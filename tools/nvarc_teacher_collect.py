# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Collect VERIFIED teacher traces for NVARC SFT: full-fidelity episodes.

The teacher plays BOTH roles through the deployment eval-sequence protocol
(advance-on-solve, revise-on-fail, final-rule credit), so the proposer rows
teach revision from behavioral evidence, not just first proposals:

- Episode: teacher proposes a canonical 4-section rule from the demos; fresh
  single-grid teacher-executor sessions apply it to held-out grids in order;
  a miss returns the server-rendered evidence (input, prediction, expected,
  diff) and the teacher revises; a final sweep re-applies the FINAL rule to
  every grid last solved by an older rule. The episode is kept only when the
  final rule solves ``--min-verified`` grids (default: all of them).
- Proposer row: the full multi-turn history. Intermediate assistant turns
  are payload-only (no think tags) -- nano-v3's chat template renders
  history turns as ``<think></think>`` + post-think payload, so storing the
  payload alone is byte-true at SFT time AND keeps NeMo-RL's per-message
  prefix-diff tokenization stable. Only the final turn carries the teacher
  CoT; train with ``sft.only_unmask_final=true`` (the SFT counterpart of
  final-rule credit -- intermediate failed rules are context, never loss).
- Executor rows: every first-try EXACT executor call inside the loop is
  harvested as a native 2-message executor row (teacher-proposed rules are
  deployment-truer inputs than reference rules). ``--executor-traces`` also
  still supports the standalone reference-rule mode.

Targets byte-continue nano-v3's generation opener (open think tag, newline,
cot, newline, close think tag, blank line, payload) and are length-capped
(``--max-cot-chars``). With ``--tokenizer-path`` set, each emitted row is
additionally rendered through the model's own chat template and rejected
when it exceeds ``--max-target-tokens`` (the SFT pack budget).

Teacher-agnostic: any OpenAI-compatible chat-completions endpoint
(``--endpoint``/``--model``); reasoning is taken from ``reasoning_content``
or a ``<think>`` block when the teacher emits one, else everything before
the payload marker.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import re
import sys
from pathlib import Path
from typing import Awaitable, Callable

from nemo_rl.data.datasets.response_datasets.nvarc_executor import (
    _load_split as load_nvarc_split,
)
from nemo_rl.environments.arc_agi_grid import serialize_grid
from tools.nvarc_cotrain_materialize import _executor_row, _proposer_row
from tools.nvarc_sft_materialize import PROPOSER_INSTRUCTIONS

_GYM_ROOT = Path(__file__).resolve().parents[1] / "3rdparty" / "Gym-workspace" / "Gym"
sys.path.insert(0, str(_GYM_ROOT))
# pyrefly: ignore  # import-error
from resources_servers.arc_agi_2.logic import (  # noqa: E402
    TransformDescriptionParseError,
    build_eval_feedback_prompt,
    build_nvarc_proposer_prompt,
    build_single_executor_prompt,
    build_single_grid_format_retry_prompt,
    compare_grid,
    parse_canonical_rule,
)

# pyrefly: ignore  # import-error
from resources_servers.arc_agi_2.scoring import extract_answer_grid  # noqa: E402

# complete(messages) -> (reasoning, visible_text); reasoning may be "" when
# the endpoint does not separate it (split_reasoning then recovers it).
CompleteFn = Callable[[list[dict]], Awaitable[tuple[str, str]]]

_THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)
_PAYLOAD_MARKERS = ("<answer>", "<rules_summary>")


def split_reasoning(reasoning: str, text: str) -> tuple[str, str]:
    """Normalize one teacher response into (cot, payload_text).

    Preference order: an endpoint-separated reasoning channel, then an inline
    ``<think>`` block, then everything before the first payload marker.
    """
    if reasoning.strip():
        return reasoning.strip(), text.strip()
    match = _THINK_RE.search(text)
    if match:
        payload = (text[: match.start()] + text[match.end() :]).strip()
        return match.group(1).strip(), payload
    for marker in _PAYLOAD_MARKERS:
        index = text.find(marker)
        if index >= 0:
            return text[:index].strip(), text[index:].strip()
    return "", text.strip()


def _think_target(cot: str, payload: str) -> str:
    return f"<think>\n{cot}\n</think>\n\n{payload}"


def _user(content: str) -> dict:
    return {"role": "user", "content": content}


def _assistant(content: str) -> dict:
    return {"role": "assistant", "content": content}


def _executor_sft_row(prompt: str, cot: str, grid: list[list[int]], task_id: str) -> dict:
    canonical = f"<answer>\n{serialize_grid(grid)}\n</answer>"
    return {
        "messages": [_user(prompt), _assistant(_think_target(cot, canonical))],
        "task_id": task_id,
        "sft_role": "executor",
    }


async def collect_executor_trace(
    puzzle: dict,
    pair: dict,
    *,
    template: str,
    complete: CompleteFn,
    samples: int,
    max_cot_chars: int,
) -> dict | None:
    """Standalone mode: sample against the puzzle's reference rule until exact."""
    row = _executor_row(puzzle, pair, bucket=0, template=template)
    prompt = row["responses_create_params"]["input"][0]["content"]
    for _ in range(samples):
        reasoning, text = await complete([_user(prompt)])
        cot, payload = split_reasoning(reasoning, text)
        if not cot or len(cot) > max_cot_chars:
            continue
        grid = extract_answer_grid(payload)
        if grid is None or grid != pair["output"]:
            continue
        return _executor_sft_row(prompt, cot, grid, puzzle["puzzle_id"])
    return None


async def _sample_rule(
    messages: list[dict],
    *,
    complete: CompleteFn,
    samples: int,
    max_cot_chars: int,
) -> tuple[str, str, str] | None:
    """Sample one proposer turn; return (cot, raw_payload, canonical_rule)."""
    for _ in range(samples):
        reasoning, text = await complete(messages)
        cot, payload = split_reasoning(reasoning, text)
        if not cot or len(cot) > max_cot_chars:
            continue
        try:
            rule = parse_canonical_rule(payload)
        except TransformDescriptionParseError:
            continue
        return cot, payload, rule
    return None


async def _execute_grid(
    rule: str,
    input_grid: list[list[int]],
    *,
    complete: CompleteFn,
    max_cot_chars: int,
) -> tuple[list[list[int]], str, str, bool] | None:
    """One fresh single-grid executor session with the deployment format retry.

    Returns (grid, cot, prompt, first_try); None when no parseable grid
    survives the one format-only retry (deployment terminates loss-masked
    there, so the episode is abandoned).
    """
    prompt = build_single_executor_prompt(description=rule, input_grid=input_grid)
    history = [_user(prompt)]
    for first_try in (True, False):
        reasoning, text = await complete(history)
        cot, payload = split_reasoning(reasoning, text)
        grid = extract_answer_grid(payload)
        if grid is not None:
            if len(cot) > max_cot_chars:
                cot = ""  # over-cap reasoning: usable for the loop, not for SFT
            return grid, cot, prompt, first_try
        if first_try:
            history.append(_assistant(payload or text.strip()))
            history.append(_user(build_single_grid_format_retry_prompt()))
    return None


async def collect_episode(
    puzzle: dict,
    *,
    demo_pairs: int,
    eval_pairs: int,
    byte_budget: int,
    rng: random.Random,
    complete: CompleteFn,
    samples: int,
    max_cot_chars: int,
    max_rounds: int,
    max_transcript_chars: int,
    min_verified: int | None,
    rejects: dict[str, int],
) -> tuple[dict, list[dict]] | None:
    """Run one full eval-sequence episode with the teacher in both roles.

    Returns (proposer_row, harvested_executor_rows) when the FINAL rule
    verifies, else None. The held-out outputs never appear in any prompt
    until a grid has been attempted (miss feedback is the deployment
    contract: it reveals that grid's expected output for revision).
    """
    row = _proposer_row(
        puzzle,
        bucket=0,
        demo_pairs=demo_pairs,
        eval_pairs=eval_pairs,
        byte_budget=byte_budget,
        rng=rng,
    )
    if row is None:
        rejects["no_row"] += 1
        return None
    grids = {f"test_{index}": pair for index, pair in enumerate(row["test"])}
    grid_ids = list(grids)
    messages = [
        {"role": "system", "content": PROPOSER_INSTRUCTIONS},
        _user(build_nvarc_proposer_prompt(demo_pairs=row["train"])),
    ]
    harvested: list[dict] = []

    def _harvest(prompt: str, cot: str, grid: list[list[int]]) -> None:
        if cot:
            harvested.append(_executor_sft_row(prompt, cot, grid, puzzle["puzzle_id"]))

    grid_index = 0
    final_cot = final_rule = None
    solved_by_final: set[str] = set()
    rounds_used = 0
    for _round in range(max_rounds):
        proposal = await _sample_rule(
            messages, complete=complete, samples=samples, max_cot_chars=max_cot_chars
        )
        if proposal is None:
            rejects["proposer_parse"] += 1
            return None
        cot, raw_payload, rule = proposal
        final_cot, final_rule = cot, rule
        rounds_used += 1
        solved_by_final = set()
        feedback: str | None = None
        while grid_index < len(grid_ids):
            grid_id = grid_ids[grid_index]
            attempt = await _execute_grid(
                rule, grids[grid_id]["input"], complete=complete, max_cot_chars=max_cot_chars
            )
            if attempt is None:
                rejects["executor_format"] += 1
                return None
            grid, exec_cot, exec_prompt, first_try = attempt
            if grid == grids[grid_id]["output"]:
                if first_try:
                    _harvest(exec_prompt, exec_cot, grid)
                solved_by_final.add(grid_id)
                grid_index += 1
                continue
            comparison = compare_grid(
                grid_id=grid_id, predicted=grid, correct=grids[grid_id]["output"]
            )
            feedback = build_eval_feedback_prompt(
                grid_id=grid_id,
                input_grid=grids[grid_id]["input"],
                predicted=grid,
                expected=grids[grid_id]["output"],
                diff_feedback=comparison.feedback,
            )
            break
        if feedback is None:
            break  # every remaining grid advanced under the current rule
        # Intermediate turns are stored payload-only: the deployment template
        # strips history thinking, so this is what the next turn conditions on.
        messages.append(_assistant(raw_payload))
        messages.append(_user(feedback))
        transcript = sum(len(m["content"]) for m in messages)
        if transcript + max_cot_chars > max_transcript_chars:
            rejects["transcript_budget"] += 1
            return None
    else:
        rejects["rounds_exhausted"] += 1
        return None

    assert final_cot is not None and final_rule is not None
    # Final-rule sweep: acceptance is judged on what the FINAL rule does to
    # every grid, exactly like the trained turn's reward.
    verified = len(solved_by_final)
    needed = len(grid_ids) if min_verified is None else min(min_verified, len(grid_ids))
    for grid_id in grid_ids:
        if grid_id in solved_by_final:
            continue
        attempt = await _execute_grid(
            final_rule, grids[grid_id]["input"], complete=complete, max_cot_chars=max_cot_chars
        )
        if attempt is None:
            continue  # sweep miss: simply not verified
        grid, exec_cot, exec_prompt, first_try = attempt
        if grid == grids[grid_id]["output"]:
            verified += 1
            if first_try:
                _harvest(exec_prompt, exec_cot, grid)
    if verified < needed:
        rejects["final_rule_unverified"] += 1
        return None
    messages.append(_assistant(_think_target(final_cot, final_rule)))
    proposer_row = {
        "messages": messages,
        "task_id": puzzle["puzzle_id"],
        "sft_role": "proposer",
        "rounds": rounds_used,
        "verified_pairs": verified,
    }
    return proposer_row, harvested


def _make_length_gate(args: argparse.Namespace) -> Callable[[dict], bool]:
    """Exact SFT-budget gate: render with the model's template, count tokens."""
    if not args.tokenizer_path:
        return lambda row: True
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    budget = args.max_target_tokens - 64  # eos + collate margin

    def fits(row: dict) -> bool:
        rendered = tokenizer.apply_chat_template(
            row["messages"], tokenize=True, add_generation_prompt=False
        )
        return len(rendered) <= budget

    return fits


async def collect(
    puzzles: list[dict],
    *,
    args: argparse.Namespace,
    template: str,
    complete: CompleteFn,
    rng: random.Random,
    fits_budget: Callable[[dict], bool],
    sink: Callable[[dict], None] | None = None,
) -> tuple[list[dict], dict]:
    """Walk the puzzle pool until both quotas are met or it is exhausted.

    Accepted rows stream to ``sink`` as they land, so a wall-clock kill
    loses at most the in-flight attempts, and progress prints keep long
    collections observable.
    """
    semaphore = asyncio.Semaphore(args.concurrency)
    rows: list[dict] = []
    counts = {"executor": 0, "proposer": 0}
    attempts = {"executor": 0, "episode": 0}
    rejects: dict[str, int] = {
        "no_row": 0,
        "proposer_parse": 0,
        "executor_format": 0,
        "transcript_budget": 0,
        "rounds_exhausted": 0,
        "final_rule_unverified": 0,
        "over_token_budget": 0,
    }

    def _accept(trace: dict) -> None:
        if not fits_budget(trace):
            rejects["over_token_budget"] += 1
            return
        rows.append(trace)
        counts[trace["sft_role"]] += 1
        if sink is not None:
            sink(trace)

    def _progress() -> None:
        total_attempts = attempts["executor"] + attempts["episode"]
        if total_attempts % 10 == 0:
            print(
                f"attempts exec {attempts['executor']} episode {attempts['episode']} | "
                f"accepted exec {counts['executor']}/{args.executor_traces} "
                f"prop {counts['proposer']}/{args.episodes} | "
                f"rejects {json.dumps(rejects)}",
                flush=True,
            )

    async def one_executor(puzzle: dict) -> None:
        pairs = json.loads(puzzle["pairs_json"])
        async with semaphore:
            attempts["executor"] += 1
            trace = await collect_executor_trace(
                puzzle,
                rng.choice(pairs),
                template=template,
                complete=complete,
                samples=args.samples_per_prompt,
                max_cot_chars=args.max_cot_chars,
            )
        if trace is not None:
            _accept(trace)
        _progress()

    async def one_episode(puzzle: dict) -> None:
        async with semaphore:
            attempts["episode"] += 1
            result = await collect_episode(
                puzzle,
                demo_pairs=args.demo_pairs,
                eval_pairs=args.eval_pairs,
                byte_budget=args.proposer_prompt_byte_budget,
                rng=rng,
                complete=complete,
                samples=args.samples_per_prompt,
                max_cot_chars=args.max_cot_chars,
                max_rounds=args.max_rounds,
                max_transcript_chars=args.max_transcript_chars,
                min_verified=args.min_verified,
                rejects=rejects,
            )
        if result is not None:
            proposer_row, harvested = result
            _accept(proposer_row)
            for trace in harvested:
                _accept(trace)
        _progress()

    pool = list(puzzles)
    rng.shuffle(pool)
    cursor = 0
    while cursor < len(pool) and (
        counts["executor"] < args.executor_traces or counts["proposer"] < args.episodes
    ):
        batch = pool[cursor : cursor + args.concurrency]
        cursor += len(batch)
        tasks = []
        for puzzle in batch:
            if counts["executor"] < args.executor_traces:
                tasks.append(one_executor(puzzle))
            if counts["proposer"] < args.episodes:
                tasks.append(one_episode(puzzle))
        await asyncio.gather(*tasks)
    stats = {
        "executor_traces": counts["executor"],
        "proposer_episodes": counts["proposer"],
        "executor_attempts": attempts["executor"],
        "episode_attempts": attempts["episode"],
        "puzzles_consumed": cursor,
        "rejects": rejects,
    }
    return rows, stats


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", required=True, help="OpenAI-compatible base URL")
    parser.add_argument("--model", required=True)
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--executor-traces",
        type=int,
        default=4096,
        help="executor-row quota; filled by episode harvest and standalone mode",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=1024,
        help="accepted full-fidelity proposer episodes to collect",
    )
    parser.add_argument("--samples-per-prompt", type=int, default=2)
    parser.add_argument(
        "--max-rounds",
        type=int,
        default=3,
        help="proposer turns per episode (1 initial + revisions)",
    )
    parser.add_argument(
        "--min-verified",
        type=int,
        default=None,
        help="grids the FINAL rule must solve (default: all of them)",
    )
    parser.add_argument(
        "--max-cot-chars",
        type=int,
        default=12_000,
        help="~4k tokens: the trainable turn must fit the RL pack",
    )
    parser.add_argument(
        "--max-transcript-chars",
        type=int,
        default=24_000,
        help="payload chars across the episode before the final think turn",
    )
    parser.add_argument(
        "--tokenizer-path",
        default=None,
        help="when set, reject rows whose templated render exceeds --max-target-tokens",
    )
    parser.add_argument(
        "--max-target-tokens",
        type=int,
        default=9216,
        help="the SFT recipe's max_total_sequence_length",
    )
    parser.add_argument("--max-output-tokens", type=int, default=8192)
    parser.add_argument(
        "--reasoning-effort",
        default=None,
        help="teacher reasoning effort (e.g. 'max'); omitted from the request when unset",
    )
    parser.add_argument("--request-timeout", type=float, default=1200.0)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--demo-pairs", type=int, default=3)
    parser.add_argument("--eval-pairs", type=int, default=2)
    parser.add_argument("--proposer-prompt-byte-budget", type=int, default=8000)
    parser.add_argument(
        "--executor-prompt-file", default="examples/prompts/nvarc_executor.txt"
    )
    parser.add_argument(
        "--val-tasks",
        type=int,
        default=32,
        help="collected puzzles held out for the SFT val split (by puzzle id)",
    )
    return parser.parse_args()


def _make_http_complete(args: argparse.Namespace) -> CompleteFn:
    import aiohttp

    headers = {"Authorization": f"Bearer {os.environ.get(args.api_key_env, 'EMPTY')}"}
    url = args.endpoint.rstrip("/") + "/chat/completions"
    # Reasoning teachers routinely exceed aiohttp's 5-minute default under
    # load (an unhandled TimeoutError killed the first shard generation), so
    # the timeout is explicit and generous, transient failures retry with
    # backoff, and a persistently failing call degrades to a rejected sample
    # instead of crashing the collection.
    timeout = aiohttp.ClientTimeout(total=args.request_timeout)

    async def complete(messages: list[dict]) -> tuple[str, str]:
        payload = {
            "model": args.model,
            "messages": messages,
            "temperature": args.temperature,
            "max_tokens": args.max_output_tokens,
        }
        if args.reasoning_effort:
            # Verified against the hub: this reaches the model (an invalid
            # value silently disables reasoning_content -- never guess here)
            # and 'max' roughly doubles reasoning depth over 'low'.
            payload["reasoning_effort"] = args.reasoning_effort
        for attempt in range(3):
            try:
                async with aiohttp.ClientSession(timeout=timeout) as session:
                    async with session.post(
                        url, json=payload, headers=headers
                    ) as response:
                        response.raise_for_status()
                        body = await response.json()
                message = body["choices"][0]["message"]
                return (
                    message.get("reasoning_content") or "",
                    message.get("content") or "",
                )
            except (aiohttp.ClientError, asyncio.TimeoutError, KeyError) as error:
                print(
                    f"teacher call failed (attempt {attempt + 1}): {error!r}",
                    flush=True,
                )
                await asyncio.sleep(10 * (attempt + 1))
        return "", ""

    return complete


def main() -> None:
    args = _parse_args()
    template = Path(args.executor_prompt_file).read_text(encoding="utf-8")
    puzzles = load_nvarc_split(args.data_dir, "train")
    rng = random.Random(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    # Stream accepted traces so a wall-clock kill loses almost nothing.
    with open(output_dir / "collected.jsonl", "w", encoding="utf-8") as collected:

        def sink(row: dict) -> None:
            collected.write(json.dumps(row) + "\n")
            collected.flush()

        rows, stats = asyncio.run(
            collect(
                puzzles,
                args=args,
                template=template,
                complete=_make_http_complete(args),
                rng=rng,
                fits_budget=_make_length_gate(args),
                sink=sink,
            )
        )
    # Split by puzzle id (the campaign invariant), directly SFT-consumable.
    task_ids = sorted({row["task_id"] for row in rows})
    rng.shuffle(task_ids)
    val_ids = set(task_ids[: args.val_tasks])
    for name, subset in (
        ("train.jsonl", [r for r in rows if r["task_id"] not in val_ids]),
        ("val.jsonl", [r for r in rows if r["task_id"] in val_ids]),
    ):
        with open(output_dir / name, "w", encoding="utf-8") as f:
            for row in subset:
                f.write(json.dumps(row) + "\n")
        stats[name.split(".")[0] + "_rows"] = len(subset)
    stats.update(
        {
            "endpoint_model": args.model,
            "seed": args.seed,
            "samples_per_prompt": args.samples_per_prompt,
            "max_cot_chars": args.max_cot_chars,
            "max_rounds": args.max_rounds,
            "temperature": args.temperature,
        }
    )
    with open(output_dir / "stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
