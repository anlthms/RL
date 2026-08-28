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
"""Collect VERIFIED teacher CoT traces for NVARC SFT (executor + proposer).

Rejection-sampled distillation against the campaign's own verifiers, so only
correct reasoning is imitated:

- Executor tasks: the teacher's ``<answer>`` grid must exactly match the
  gold output; the emitted payload is re-rendered canonically.
- Proposer tasks: the teacher sees only the demo pairs; its canonical
  4-section rule is verified BEHAVIORALLY -- the teacher itself executes the
  rule on the puzzle's held-out inputs and every held-out output must match
  exactly (``--min-verified`` relaxes this).

Targets byte-continue nano-v3's generation opener with the teacher's
reasoning inside the think block (open think tag, newline, cot, newline,
close think tag, blank line, payload) and are length-capped
(``--max-cot-chars``): data-level length selection is the anti-runaway
prior; the floor-with-loss-ON contract backstops overruns at RL time.

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
    build_nvarc_proposer_prompt,
    build_single_executor_prompt,
    parse_canonical_rule,
)

# pyrefly: ignore  # import-error
from resources_servers.arc_agi_2.scoring import extract_answer_grid  # noqa: E402

# complete(prompt) -> (reasoning, visible_text); reasoning may be "" when the
# endpoint does not separate it (split_reasoning then recovers it).
CompleteFn = Callable[[str], Awaitable[tuple[str, str]]]

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


async def collect_executor_trace(
    puzzle: dict,
    pair: dict,
    *,
    template: str,
    complete: CompleteFn,
    samples: int,
    max_cot_chars: int,
) -> dict | None:
    """Sample the teacher until one trace's answer grid is exactly right."""
    row = _executor_row(puzzle, pair, bucket=0, template=template)
    prompt = row["responses_create_params"]["input"][0]["content"]
    for _ in range(samples):
        reasoning, text = await complete(prompt)
        cot, payload = split_reasoning(reasoning, text)
        if not cot or len(cot) > max_cot_chars:
            continue
        grid = extract_answer_grid(payload)
        if grid is None or grid != pair["output"]:
            continue
        canonical = f"<answer>\n{serialize_grid(grid)}\n</answer>"
        return {
            "messages": [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": _think_target(cot, canonical)},
            ],
            "task_id": puzzle["puzzle_id"],
            "sft_role": "executor",
        }
    return None


async def collect_proposer_trace(
    puzzle: dict,
    *,
    demo_pairs: int,
    eval_pairs: int,
    byte_budget: int,
    rng: random.Random,
    complete: CompleteFn,
    samples: int,
    max_cot_chars: int,
    min_verified: int | None,
) -> dict | None:
    """Sample rules from demos only; keep one the teacher can itself execute.

    The rule is verified behaviorally: the teacher executes it on every
    held-out input, and the outputs must match exactly. The held-out targets
    never appear in any teacher prompt.
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
        return None
    prompt = build_nvarc_proposer_prompt(demo_pairs=row["train"])
    needed = len(row["test"]) if min_verified is None else min_verified
    for _ in range(samples):
        reasoning, text = await complete(prompt)
        cot, payload = split_reasoning(reasoning, text)
        if not cot or len(cot) > max_cot_chars:
            continue
        try:
            rule = parse_canonical_rule(payload)
        except TransformDescriptionParseError:
            continue
        verified = 0
        for pair in row["test"]:
            exec_prompt = build_single_executor_prompt(
                description=rule, input_grid=pair["input"]
            )
            exec_reasoning, exec_text = await complete(exec_prompt)
            _, exec_payload = split_reasoning(exec_reasoning, exec_text)
            grid = extract_answer_grid(exec_payload)
            if grid is not None and grid == pair["output"]:
                verified += 1
                if verified >= needed:
                    break  # verification met; save the remaining teacher calls
        if verified < needed:
            continue
        return {
            "messages": [
                {"role": "system", "content": PROPOSER_INSTRUCTIONS},
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": _think_target(cot, rule)},
            ],
            "task_id": puzzle["puzzle_id"],
            "sft_role": "proposer",
            "verified_pairs": verified,
        }
    return None


async def collect(
    puzzles: list[dict],
    *,
    args: argparse.Namespace,
    template: str,
    complete: CompleteFn,
    rng: random.Random,
    sink: Callable[[dict], None] | None = None,
) -> tuple[list[dict], dict]:
    """Walk the puzzle pool until both role quotas are met or it is exhausted.

    Accepted traces stream to ``sink`` as they land, so a wall-clock kill
    loses at most the in-flight attempts, and progress prints keep long
    collections observable.
    """
    semaphore = asyncio.Semaphore(args.concurrency)
    executor_rows: list[dict] = []
    proposer_rows: list[dict] = []
    attempts = {"executor": 0, "proposer": 0}

    def _accept(trace: dict, into: list[dict]) -> None:
        into.append(trace)
        if sink is not None:
            sink(trace)

    def _progress() -> None:
        total_attempts = attempts["executor"] + attempts["proposer"]
        if total_attempts % 10 == 0:
            print(
                f"attempts exec {attempts['executor']} prop {attempts['proposer']} | "
                f"accepted exec {len(executor_rows)}/{args.executor_traces} "
                f"prop {len(proposer_rows)}/{args.proposer_traces}",
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
            _accept(trace, executor_rows)
        _progress()

    async def one_proposer(puzzle: dict) -> None:
        async with semaphore:
            attempts["proposer"] += 1
            trace = await collect_proposer_trace(
                puzzle,
                demo_pairs=args.demo_pairs,
                eval_pairs=args.eval_pairs,
                byte_budget=args.proposer_prompt_byte_budget,
                rng=rng,
                complete=complete,
                samples=args.samples_per_prompt,
                max_cot_chars=args.max_cot_chars,
                min_verified=args.min_verified,
            )
        if trace is not None:
            _accept(trace, proposer_rows)
        _progress()

    pool = list(puzzles)
    rng.shuffle(pool)
    cursor = 0
    while cursor < len(pool) and (
        len(executor_rows) < args.executor_traces
        or len(proposer_rows) < args.proposer_traces
    ):
        batch = pool[cursor : cursor + args.concurrency]
        cursor += len(batch)
        tasks = []
        for puzzle in batch:
            if len(executor_rows) < args.executor_traces:
                tasks.append(one_executor(puzzle))
            if len(proposer_rows) < args.proposer_traces:
                tasks.append(one_proposer(puzzle))
        await asyncio.gather(*tasks)
    stats = {
        "executor_traces": len(executor_rows),
        "proposer_traces": len(proposer_rows),
        "executor_attempts": attempts["executor"],
        "proposer_attempts": attempts["proposer"],
        "puzzles_consumed": cursor,
    }
    return executor_rows + proposer_rows, stats


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", required=True, help="OpenAI-compatible base URL")
    parser.add_argument("--model", required=True)
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--executor-traces", type=int, default=4096)
    parser.add_argument("--proposer-traces", type=int, default=4096)
    parser.add_argument("--samples-per-prompt", type=int, default=2)
    parser.add_argument(
        "--min-verified",
        type=int,
        default=None,
        help="held-out pairs a rule must solve (default: all of them)",
    )
    parser.add_argument(
        "--max-cot-chars",
        type=int,
        default=12_000,
        help="~4k tokens: the trainable proposer turn must fit the RL pack",
    )
    parser.add_argument("--max-output-tokens", type=int, default=8192)
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

    async def complete(prompt: str) -> tuple[str, str]:
        payload = {
            "model": args.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": args.temperature,
            "max_tokens": args.max_output_tokens,
        }
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
            "temperature": args.temperature,
        }
    )
    with open(output_dir / "stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
