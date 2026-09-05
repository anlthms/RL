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
"""Run reproducible direct and proposer/executor Kimi diagnostics on NVARC.

The direct arm is a behavior-faithful port of arcprize/arc-agi-benchmarking
commit 28e67d54: one user message, its exact prompt rendering, final JSON-grid
extraction, and exact pair equality. The split arm mirrors the deployed
``hidden_test`` protocol: a demos-only canonical-rule proposer, fresh
single-grid executor sessions, deterministic public-demo miss feedback, a
final-rule demo sweep, and one untouched held-out test execution.

The harness never puts a held-out test target in a model request or revision
feedback. Every request and response is persisted for offline rescoring.
Verified proposer and first-try executor turns are also exported as directly
consumable SFT JSONL.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import random
import re
import sys
import time
from collections import Counter
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

GYM_ROOT = Path(__file__).resolve().parents[1] / "3rdparty" / "Gym-workspace" / "Gym"
if str(GYM_ROOT) not in sys.path:
    sys.path.insert(0, str(GYM_ROOT))

# pyrefly: ignore  # import-error
from resources_servers.arc_agi_2.logic import (  # noqa: E402
    TransformDescriptionParseError,
    assert_model_request_safe,
    build_eval_feedback_prompt,
    build_nvarc_proposer_prompt,
    build_single_executor_prompt,
    build_single_grid_format_retry_prompt,
    compare_grid,
    parse_canonical_rule,
)

# pyrefly: ignore  # import-error
from resources_servers.arc_agi_2.scoring import (  # noqa: E402
    ANSWER_CLOSE,
    ANSWER_OPEN,
    MAX_GRID_DIM,
    extract_answer_grid,
    serialize_grid,
)

Grid = list[list[int]]
CompleteFn = Callable[[str, list[dict[str, str]]], Awaitable[dict[str, Any]]]

PROPOSER_INSTRUCTIONS = (
    "Act only as an ARC transformation-rule proposer. Be precise and concise; "
    "stop reasoning once one rule explains every example. Return the requested "
    "transformation-description artifact; do not execute the test grid."
)

ARC_PRIZE_COMMIT = "28e67d54b05df5be10281892243c509a42a874f1"
ARC_PRIZE_PROMPT = """You are participating in a puzzle solving competition. You are an expert at solving puzzles.

Below is a list of input and output pairs with a pattern. Your goal is to identify the pattern or transformation in the training examples that maps the input to the output, then apply that pattern to the test input to give a final output.

Respond in the format of the training output examples

--Training Examples--
{training_examples}
--End of Training Examples--

--Test Input--
{test_input}
--End of Test Input--

Your response:"""

_BOXED_RE = re.compile(r"\\boxed\{(.*?)\}", re.DOTALL)
_THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)
_ANSWER_BLOCK_RE = re.compile(
    re.escape(ANSWER_OPEN) + r"(.*?)" + re.escape(ANSWER_CLOSE), re.DOTALL
)
_FORBIDDEN_TRACE_KEYS = frozenset(
    {
        "expected",
        "expected_output",
        "expected_outputs",
        "hidden_output",
        "hidden_outputs",
        "target",
        "targets",
        "test_output",
        "test_outputs",
        "test_targets",
    }
)


@dataclass(frozen=True)
class PilotCase:
    """One frozen NVARC task with public demos and one held-out pair."""

    task_id: str
    difficulty: int
    demo_pair_indices: tuple[int, ...]
    test_pair_index: int
    demos: list[dict[str, Grid]]
    test_input: Grid
    test_target: Grid


class OracleClient(Protocol):
    """Minimal client contract shared by the HTTP client and unit fakes."""

    async def complete(
        self, role: str, messages: list[dict[str, str]]
    ) -> dict[str, Any]:
        """Return a fully recorded chat-completions call."""
        raise NotImplementedError


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def split_reasoning(reasoning: str, text: str) -> tuple[str, str]:
    """Normalize an endpoint-separated or inline reasoning response."""
    if reasoning.strip():
        return reasoning.strip(), text.strip()
    match = _THINK_RE.search(text)
    if match:
        payload = (text[: match.start()] + text[match.end() :]).strip()
        return match.group(1).strip(), payload
    for marker in ("<answer>", "<rules_summary>"):
        index = text.find(marker)
        if index >= 0:
            return text[:index].strip(), text[index:].strip()
    return "", text.strip()


def _load_nvarc_split(data_dir: str, split: str) -> list[dict[str, Any]]:
    """Load the same sorted columns as the production NVARC dataset loader."""
    # Deferred import keeps prompt rendering, parsing, and comparison usable
    # without the optional PyArrow dependency.
    import pyarrow.dataset as ds

    shards = sorted(Path(data_dir).glob("data-*.parquet"))
    if not shards:
        raise FileNotFoundError(f"no data-*.parquet shards under {data_dir}")
    dataset = ds.dataset([str(shard) for shard in shards], format="parquet")
    table = dataset.to_table(
        columns=["puzzle_id", "canonical_rule", "pairs_json", "difficulty"],
        filter=ds.field("split") == split,
    )
    rows = table.to_pylist()
    if not rows:
        raise ValueError(f"no rows with split={split!r} under {data_dir}")
    rows.sort(key=lambda row: row["puzzle_id"])
    return rows


def build_arc_prize_prompt(
    *, training_pairs: list[dict[str, Grid]], test_input: Grid
) -> str:
    """Render the prompt exactly as arc-agi-benchmarking's prompt manager."""
    training_examples = ""
    for index, pair in enumerate(training_pairs):
        training_examples += f"--Example {index}-- \n\n INPUT: \n\n"
        training_examples += json.dumps(pair["input"]) + "\n\n"
        training_examples += "OUTPUT: \n\n"
        training_examples += json.dumps(pair["output"]) + "\n\n"
    return ARC_PRIZE_PROMPT.format(
        training_examples=training_examples,
        test_input=json.dumps(test_input),
    )


def _backscan_json_grid(text: str) -> Grid | None:
    """Port arc-agi-benchmarking's final balanced JSON-grid extraction."""
    last_bracket_index = -1
    closing_bracket = ""
    for index in range(len(text) - 1, -1, -1):
        if text[index] in ("]", "}"):
            last_bracket_index = index
            closing_bracket = text[index]
            break
    if last_bracket_index < 0:
        return None
    opening_bracket = "[" if closing_bracket == "]" else "{"
    depth = 1
    start_index = -1
    for index in range(last_bracket_index - 1, -1, -1):
        if text[index] == closing_bracket:
            depth += 1
        elif text[index] == opening_bracket:
            depth -= 1
            if depth == 0:
                start_index = index
                break
    if start_index < 0:
        return None
    try:
        value = json.loads(text[start_index : last_bracket_index + 1])
    except json.JSONDecodeError:
        return None
    if not isinstance(value, list) or not value:
        return None
    if not all(isinstance(row, list) for row in value):
        return None
    return value


def parse_arc_prize_grid(text: str) -> Grid | None:
    """Port the benchmark's boxed-first, final-JSON response parsing."""
    boxed_match = _BOXED_RE.search(text)
    if boxed_match:
        try:
            value = json.loads(boxed_match.group(1).strip())
        except json.JSONDecodeError:
            value = None
        if isinstance(value, list) and all(isinstance(row, list) for row in value):
            return value
    return _backscan_json_grid(text)


def diagnose_prediction(prediction: Grid | None, target: Grid) -> str:
    """Classify an ARC-Prize-parsed prediction relative to its exact target."""
    if prediction is None:
        return "no_json_grid"
    if not prediction or not all(isinstance(row, list) and row for row in prediction):
        return "invalid_grid"
    if any(
        any(type(cell) is not int or not 0 <= cell <= 9 for cell in row)
        for row in prediction
    ):
        return "invalid_token_cell"
    if any(len(row) != len(prediction[0]) for row in prediction[1:]):
        return "ragged_rows"
    if (len(prediction), len(prediction[0])) != (len(target), len(target[0])):
        return "valid_wrong_shape"
    if prediction != target:
        return "valid_right_shape_wrong_cells"
    return "exact"


def _diagnose_answer_block(block: str, *, unclosed: bool) -> str:
    lines = [line.strip() for line in block.strip().splitlines()]
    lines = [line for line in lines if line]
    if not lines:
        return "incomplete_truncated_grid" if unclosed else "empty_block"
    if len(lines) > MAX_GRID_DIM:
        return "more_than_30_rows"
    width: int | None = None
    for line in lines:
        if any(character.isspace() for character in line):
            cells = line.split()
            valid = all(len(cell) == 1 and cell.isdigit() for cell in cells)
        else:
            cells = list(line)
            valid = line.isdigit()
        if not valid:
            return "incomplete_truncated_grid" if unclosed else "invalid_token_cell"
        if len(cells) > MAX_GRID_DIM:
            return "more_than_30_columns"
        if width is not None and len(cells) != width:
            return "incomplete_truncated_grid" if unclosed else "ragged_rows"
        width = len(cells)
    return "unknown_parse_failure"


def diagnose_executor_response(response: str, target: Grid) -> tuple[Grid | None, str]:
    """Use the production parser and add a strict typed outcome taxonomy."""
    prediction = extract_answer_grid(response)
    if prediction is not None:
        return prediction, diagnose_prediction(prediction, target)

    blocks = _ANSWER_BLOCK_RE.findall(response)
    if blocks:
        return None, _diagnose_answer_block(blocks[-1], unclosed=False)
    _, delimiter, tail = response.rpartition(ANSWER_OPEN)
    if delimiter:
        return None, _diagnose_answer_block(tail, unclosed=True)
    return None, "no_answer_block"


def _puzzle_fingerprint(row: dict[str, Any]) -> str:
    return _sha256_text(row["canonical_rule"] + "\n" + row["pairs_json"])


def materialize_manifest(
    *,
    data_dir: str,
    output: Path,
    count: int,
    seed: int,
    demo_pairs: int,
    endpoint: str,
    model: str,
    temperature: float,
    max_output_tokens: int,
    reasoning_effort: str | None,
    max_rounds: int,
) -> dict[str, Any]:
    """Uniformly sample eligible train puzzles and freeze pair-level splits."""
    if count <= 0 or demo_pairs <= 0:
        raise ValueError("count and demo_pairs must be positive")
    rows = _load_nvarc_split(data_dir, "train")
    eligible = [
        row for row in rows if len(json.loads(row["pairs_json"])) >= demo_pairs + 1
    ]
    if count > len(eligible):
        raise ValueError(f"requested {count} tasks from only {len(eligible)} eligible")
    rng = random.Random(seed)
    sampled = rng.sample(eligible, count)
    manifest_rows: list[dict[str, Any]] = []
    for row in sampled:
        pairs = json.loads(row["pairs_json"])
        chosen = rng.sample(range(len(pairs)), demo_pairs + 1)
        manifest_rows.append(
            {
                "task_id": row["puzzle_id"],
                "difficulty": row["difficulty"],
                "demo_pair_indices": chosen[:-1],
                "test_pair_index": chosen[-1],
                "puzzle_sha256": _puzzle_fingerprint(row),
            }
        )
    manifest = {
        "schema_version": 1,
        "source": {
            "data_dir": str(Path(data_dir).resolve()),
            "split": "train",
            "eligible_puzzles": len(eligible),
        },
        "sampling": {
            "method": "uniform_without_replacement_over_eligible_puzzle_ids",
            "seed": seed,
            "count": count,
            "demo_pairs": demo_pairs,
            "test_pairs": 1,
        },
        "arc_prize": {
            "repository": "https://github.com/arcprize/arc-agi-benchmarking",
            "commit": ARC_PRIZE_COMMIT,
            "prompt_sha256": _sha256_text(ARC_PRIZE_PROMPT),
            "parser": "boxed_then_final_balanced_json",
            "scorer": "exact_grid_equality_per_test_pair",
            "num_attempts": 1,
        },
        "oracle": {
            "endpoint": endpoint.rstrip("/"),
            "model": model,
            "temperature": temperature,
            "max_output_tokens": max_output_tokens,
            "reasoning_effort": reasoning_effort,
        },
        "split_protocol": {
            "name": "hidden_test",
            "proposer_test_aware": False,
            "max_rounds": max_rounds,
            "executor_format_retries": 1,
        },
        "rows": manifest_rows,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def load_cases(manifest: dict[str, Any]) -> list[PilotCase]:
    """Resolve a frozen manifest against its source dataset with hash checks."""
    data_dir = manifest["source"]["data_dir"]
    source_rows = {
        row["puzzle_id"]: row
        for row in _load_nvarc_split(data_dir, manifest["source"]["split"])
    }
    cases: list[PilotCase] = []
    for frozen in manifest["rows"]:
        row = source_rows[frozen["task_id"]]
        if _puzzle_fingerprint(row) != frozen["puzzle_sha256"]:
            raise ValueError(f"source puzzle changed: {frozen['task_id']}")
        pairs = json.loads(row["pairs_json"])
        demo_indices = tuple(frozen["demo_pair_indices"])
        test_index = frozen["test_pair_index"]
        cases.append(
            PilotCase(
                task_id=row["puzzle_id"],
                difficulty=row["difficulty"],
                demo_pair_indices=demo_indices,
                test_pair_index=test_index,
                demos=[pairs[index] for index in demo_indices],
                test_input=pairs[test_index]["input"],
                test_target=pairs[test_index]["output"],
            )
        )
    return cases


class HttpOracleClient:
    """Retrying OpenAI-compatible client that returns complete raw call records."""

    def __init__(
        self,
        *,
        endpoint: str,
        model: str,
        api_key: str,
        temperature: float,
        max_output_tokens: int,
        reasoning_effort: str | None,
        timeout_seconds: float,
        stream: bool,
    ) -> None:
        self.url = endpoint.rstrip("/") + "/chat/completions"
        self.model = model
        self.api_key = api_key
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        self.reasoning_effort = reasoning_effort
        self.timeout_seconds = timeout_seconds
        self.stream = stream
        self.request_index = 0

    @staticmethod
    def _accumulate_stream_chunk(state: dict[str, Any], chunk: dict[str, Any]) -> None:
        """Merge one OpenAI-compatible SSE chunk into a final response."""
        for key in ("id", "model", "object", "system_fingerprint"):
            if key in chunk:
                state[key] = chunk[key]
        usage = chunk.get("usage")
        if isinstance(usage, dict):
            state["usage"] = usage
        choices = chunk.get("choices", [])
        if not choices:
            return
        choice = choices[0]
        delta = choice.get("delta", {})
        content = delta.get("content") or ""
        reasoning = delta.get("reasoning_content") or delta.get("reasoning") or ""
        if not isinstance(content, str) or not isinstance(reasoning, str):
            raise TypeError("streamed assistant content/reasoning must be strings")
        state["content"] += content
        state["reasoning_content"] += reasoning
        finish_reason = choice.get("finish_reason")
        if finish_reason is not None:
            state["finish_reason"] = finish_reason

    async def _read_stream(
        self, response: Any, *, role: str, request_id: int, started: float
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Read NVIDIA's data-only SSE stream and report live character counts."""
        state: dict[str, Any] = {
            "content": "",
            "reasoning_content": "",
            "finish_reason": None,
            "usage": {},
        }
        chunks = 0
        saw_done = False
        next_report_chars = 8_192
        async for raw_line in response.content:
            line = raw_line.decode("utf-8").strip()
            if not line or not line.startswith("data:"):
                continue
            data = line.removeprefix("data:").strip()
            if data == "[DONE]":
                saw_done = True
                break
            chunk = json.loads(data)
            self._accumulate_stream_chunk(state, chunk)
            chunks += 1
            streamed_chars = len(state["content"]) + len(state["reasoning_content"])
            if streamed_chars >= next_report_chars:
                print(
                    "oracle_stream: "
                    f"request={request_id} role={role} chunks={chunks} "
                    f"reasoning_chars={len(state['reasoning_content'])} "
                    f"content_chars={len(state['content'])} "
                    f"elapsed={time.perf_counter() - started:.1f}s",
                    flush=True,
                )
                next_report_chars = (streamed_chars // 8_192 + 1) * 8_192
        if not saw_done and state["finish_reason"] is None:
            raise ValueError("SSE stream ended without [DONE] or finish_reason")
        body = {
            key: state[key]
            for key in ("id", "model", "object", "system_fingerprint")
            if key in state
        }
        body.update(
            {
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": state["content"],
                            "reasoning_content": state["reasoning_content"],
                        },
                        "finish_reason": state["finish_reason"],
                    }
                ],
                "usage": state["usage"],
            }
        )
        return body, {
            "enabled": True,
            "chunks": chunks,
            "saw_done": saw_done,
            "reasoning_chars": len(state["reasoning_content"]),
            "content_chars": len(state["content"]),
        }

    async def complete(
        self, role: str, messages: list[dict[str, str]]
    ) -> dict[str, Any]:
        """Call the endpoint and retain retry errors, request, response, and usage."""
        # Deferred import keeps manifest generation and unit tests lightweight.
        import aiohttp

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_output_tokens,
        }
        if self.reasoning_effort:
            payload["reasoning_effort"] = self.reasoning_effort
        if self.stream:
            payload["stream"] = True
            payload["stream_options"] = {"include_usage": True}
        assert_model_request_safe(payload)
        timeout = aiohttp.ClientTimeout(total=self.timeout_seconds)
        retry_errors: list[str] = []
        started = time.perf_counter()
        self.request_index += 1
        request_id = self.request_index
        for attempt in range(3):
            try:
                async with aiohttp.ClientSession(timeout=timeout) as session:
                    async with session.post(
                        self.url,
                        json=payload,
                        headers={
                            "Authorization": f"Bearer {self.api_key}",
                            "Accept": (
                                "text/event-stream"
                                if self.stream
                                else "application/json"
                            ),
                            "Content-Type": "application/json",
                        },
                    ) as response:
                        if response.status >= 400:
                            response_text = await response.text()
                            raise RuntimeError(
                                f"HTTP {response.status}: {response_text[:500]}"
                            )
                        if self.stream:
                            body, stream_stats = await self._read_stream(
                                response,
                                role=role,
                                request_id=request_id,
                                started=started,
                            )
                        else:
                            body = json.loads(await response.text())
                            stream_stats = {"enabled": False}
                message = body["choices"][0]["message"]
                content = message.get("content") or ""
                reasoning = message.get("reasoning_content") or ""
                if not isinstance(content, str) or not isinstance(reasoning, str):
                    raise TypeError("assistant content/reasoning must be strings")
                return {
                    "role": role,
                    "request": payload,
                    "response": body,
                    "content": content,
                    "reasoning": reasoning,
                    "usage": body.get("usage", {}),
                    "latency_seconds": time.perf_counter() - started,
                    "retry_errors": retry_errors,
                    "stream": stream_stats,
                }
            except (
                aiohttp.ClientError,
                asyncio.TimeoutError,
                KeyError,
                TypeError,
                ValueError,
                RuntimeError,
            ) as error:
                retry_errors.append(repr(error))
                if attempt < 2:
                    await asyncio.sleep(10 * (attempt + 1))
        raise RuntimeError(f"oracle call failed after 3 attempts: {retry_errors}")


def _usage_total(call: dict[str, Any]) -> int:
    value = call.get("usage", {}).get("total_tokens", 0)
    return value if isinstance(value, int) else 0


def _executor_sft_row(
    *,
    prompt: str,
    reasoning: str,
    prediction: Grid,
    task_id: str,
    grid_id: str,
) -> dict[str, Any]:
    answer = f"<answer>\n{serialize_grid(prediction)}\n</answer>"
    return {
        "messages": [
            {"role": "user", "content": prompt},
            {
                "role": "assistant",
                "content": f"<think>\n{reasoning.strip()}\n</think>\n\n{answer}",
            },
        ],
        "task_id": task_id,
        "sft_role": "executor",
        "source": "kimi_k3_oracle_pilot",
        "grid_id": grid_id,
    }


async def evaluate_direct(case: PilotCase, client: OracleClient) -> dict[str, Any]:
    """Evaluate one case under the pinned ARC Prize single-turn method."""
    prompt = build_arc_prize_prompt(
        training_pairs=case.demos, test_input=case.test_input
    )
    call = await client.complete("direct", [{"role": "user", "content": prompt}])
    prediction = parse_arc_prize_grid(call["content"])
    outcome = diagnose_prediction(prediction, case.test_target)
    reasoning, _ = split_reasoning(call["reasoning"], call["content"])
    sft_row = None
    if prediction == case.test_target and reasoning:
        sft_row = {
            "messages": [
                {"role": "user", "content": prompt},
                {
                    "role": "assistant",
                    "content": (
                        f"<think>\n{reasoning.strip()}\n</think>\n\n"
                        + json.dumps(prediction)
                    ),
                },
            ],
            "task_id": case.task_id,
            "sft_role": "direct",
            "source": "kimi_k3_arc_prize_direct",
        }
    return {
        "schema_version": 1,
        "mode": "direct",
        "task_id": case.task_id,
        "difficulty": case.difficulty,
        "demo_pair_indices": list(case.demo_pair_indices),
        "test_pair_index": case.test_pair_index,
        "prompt_sha256": _sha256_text(prompt),
        "format_valid": prediction is not None,
        "parse_outcome": outcome,
        "test_exact": prediction == case.test_target,
        "prediction": prediction,
        "calls": [call],
        "direct_sft_row": sft_row,
    }


async def _execute_grid(
    *,
    case: PilotCase,
    client: OracleClient,
    description: str,
    grid_id: str,
    input_grid: Grid,
    target: Grid,
) -> dict[str, Any]:
    prompt = build_single_executor_prompt(
        description=description, input_grid=input_grid
    )
    history = [{"role": "user", "content": prompt}]
    calls: list[dict[str, Any]] = []
    for attempt in range(2):
        call = await client.complete("executor", history)
        prediction, parse_outcome = diagnose_executor_response(call["content"], target)
        calls.append(
            {
                **call,
                "grid_id": grid_id,
                "format_retry": bool(attempt),
                "parse_outcome": parse_outcome,
                "format_valid": prediction is not None,
                "exact": prediction == target,
            }
        )
        if prediction is not None:
            reasoning, _ = split_reasoning(call["reasoning"], call["content"])
            sft_row = None
            if attempt == 0 and prediction == target and reasoning:
                sft_row = _executor_sft_row(
                    prompt=prompt,
                    reasoning=reasoning,
                    prediction=prediction,
                    task_id=case.task_id,
                    grid_id=grid_id,
                )
            return {
                "prediction": prediction,
                "parse_outcome": parse_outcome,
                "format_valid": True,
                "format_retry_used": bool(attempt),
                "exact": prediction == target,
                "calls": calls,
                "sft_row": sft_row,
            }
        if attempt == 0:
            history.extend(
                [
                    {"role": "assistant", "content": call["content"]},
                    {
                        "role": "user",
                        "content": build_single_grid_format_retry_prompt(),
                    },
                ]
            )
    return {
        "prediction": None,
        "parse_outcome": calls[-1]["parse_outcome"],
        "format_valid": False,
        "format_retry_used": True,
        "exact": False,
        "calls": calls,
        "sft_row": None,
    }


def _trace_has_forbidden_target_key(value: Any) -> bool:
    if isinstance(value, dict):
        if any(str(key).lower() in _FORBIDDEN_TRACE_KEYS for key in value):
            return True
        return any(_trace_has_forbidden_target_key(item) for item in value.values())
    if isinstance(value, list):
        return any(_trace_has_forbidden_target_key(item) for item in value)
    return False


async def evaluate_split(
    case: PilotCase,
    client: OracleClient,
    *,
    max_rounds: int,
) -> dict[str, Any]:
    """Run one demos-only proposer/executor hidden-test episode."""
    base_prompt = build_nvarc_proposer_prompt(demo_pairs=case.demos)
    proposer_history = [
        {"role": "system", "content": PROPOSER_INSTRUCTIONS},
        {"role": "user", "content": base_prompt},
    ]
    proposer_turns: list[dict[str, Any]] = []
    executor_attempts: list[dict[str, Any]] = []
    executor_sft_rows: list[dict[str, Any]] = []
    demo_index = 0
    description: str | None = None
    attempted_with_current_rule: set[str] = set()
    final_demo_attempts: dict[str, dict[str, Any]] = {}
    termination_reason = "max_rounds"
    final_turn_is_valid = False

    for round_index in range(max_rounds):
        call = await client.complete("proposer", proposer_history)
        try:
            proposed_description = parse_canonical_rule(call["content"])
            parse_valid = True
            parse_error = None
        except TransformDescriptionParseError as error:
            proposed_description = None
            parse_valid = False
            parse_error = str(error)
        turn = {
            "round_index": round_index,
            "call": call,
            "format_valid": parse_valid,
            "parse_error": parse_error,
            "canonical_rule": proposed_description,
            "canonical_rule_sha256": (
                _sha256_text(proposed_description) if proposed_description else None
            ),
            "feedback": None,
        }
        proposer_turns.append(turn)
        if not parse_valid:
            termination_reason = "proposer_format_failure"
            final_turn_is_valid = False
            break

        assert proposed_description is not None
        description = proposed_description
        final_turn_is_valid = True
        attempted_with_current_rule = set()
        final_demo_attempts = {}
        failed_attempt: dict[str, Any] | None = None
        while demo_index < len(case.demos):
            grid_id = f"train_{demo_index}"
            demo = case.demos[demo_index]
            attempt = await _execute_grid(
                case=case,
                client=client,
                description=description,
                grid_id=grid_id,
                input_grid=demo["input"],
                target=demo["output"],
            )
            executor_attempts.append(attempt)
            attempted_with_current_rule.add(grid_id)
            final_demo_attempts[grid_id] = attempt
            if attempt["sft_row"] is not None:
                executor_sft_rows.append(attempt["sft_row"])
            if not attempt["format_valid"]:
                termination_reason = "executor_format_failure"
                return _split_result(
                    case=case,
                    proposer_turns=proposer_turns,
                    executor_attempts=executor_attempts,
                    executor_sft_rows=executor_sft_rows,
                    final_demo_attempts=final_demo_attempts,
                    description=description,
                    final_turn_is_valid=final_turn_is_valid,
                    termination_reason=termination_reason,
                    test_attempt=None,
                    proposer_history=proposer_history,
                )
            if attempt["exact"]:
                demo_index += 1
                continue
            failed_attempt = attempt
            break

        if failed_attempt is None:
            termination_reason = "train_verified"
            break
        verification = compare_grid(
            grid_id=f"train_{demo_index}",
            predicted=failed_attempt["prediction"],
            correct=case.demos[demo_index]["output"],
        )
        feedback = build_eval_feedback_prompt(
            grid_id=f"train_{demo_index}",
            input_grid=case.demos[demo_index]["input"],
            predicted=failed_attempt["prediction"],
            expected=case.demos[demo_index]["output"],
            diff_feedback=verification.feedback,
        )
        turn["feedback"] = feedback
        proposer_history.extend(
            [
                {"role": "assistant", "content": call["content"]},
                {"role": "user", "content": feedback},
            ]
        )

    if description is None:
        return _split_result(
            case=case,
            proposer_turns=proposer_turns,
            executor_attempts=executor_attempts,
            executor_sft_rows=executor_sft_rows,
            final_demo_attempts=final_demo_attempts,
            description=None,
            final_turn_is_valid=False,
            termination_reason=termination_reason,
            test_attempt=None,
            proposer_history=proposer_history,
        )

    for index, demo in enumerate(case.demos):
        grid_id = f"train_{index}"
        if grid_id in attempted_with_current_rule:
            continue
        attempt = await _execute_grid(
            case=case,
            client=client,
            description=description,
            grid_id=grid_id,
            input_grid=demo["input"],
            target=demo["output"],
        )
        executor_attempts.append(attempt)
        final_demo_attempts[grid_id] = attempt
        if attempt["sft_row"] is not None:
            executor_sft_rows.append(attempt["sft_row"])
        if not attempt["format_valid"]:
            return _split_result(
                case=case,
                proposer_turns=proposer_turns,
                executor_attempts=executor_attempts,
                executor_sft_rows=executor_sft_rows,
                final_demo_attempts=final_demo_attempts,
                description=description,
                final_turn_is_valid=final_turn_is_valid,
                termination_reason="executor_format_failure",
                test_attempt=None,
                proposer_history=proposer_history,
            )

    test_attempt = await _execute_grid(
        case=case,
        client=client,
        description=description,
        grid_id="test_0",
        input_grid=case.test_input,
        target=case.test_target,
    )
    executor_attempts.append(test_attempt)
    if test_attempt["sft_row"] is not None:
        executor_sft_rows.append(test_attempt["sft_row"])
    return _split_result(
        case=case,
        proposer_turns=proposer_turns,
        executor_attempts=executor_attempts,
        executor_sft_rows=executor_sft_rows,
        final_demo_attempts=final_demo_attempts,
        description=description,
        final_turn_is_valid=final_turn_is_valid,
        termination_reason=termination_reason,
        test_attempt=test_attempt,
        proposer_history=proposer_history,
    )


def _split_result(
    *,
    case: PilotCase,
    proposer_turns: list[dict[str, Any]],
    executor_attempts: list[dict[str, Any]],
    executor_sft_rows: list[dict[str, Any]],
    final_demo_attempts: dict[str, dict[str, Any]],
    description: str | None,
    final_turn_is_valid: bool,
    termination_reason: str,
    test_attempt: dict[str, Any] | None,
    proposer_history: list[dict[str, str]],
) -> dict[str, Any]:
    demo_exact = sum(bool(attempt["exact"]) for attempt in final_demo_attempts.values())
    demo_count = len(case.demos)
    train_gate_pass = demo_exact == demo_count
    test_exact = bool(test_attempt and test_attempt["exact"])
    proposer_sft_row = None
    if train_gate_pass and test_exact and final_turn_is_valid and proposer_turns:
        final_call = proposer_turns[-1]["call"]
        reasoning, _ = split_reasoning(final_call["reasoning"], final_call["content"])
        if reasoning and description is not None:
            proposer_sft_row = {
                "messages": proposer_history
                + [
                    {
                        "role": "assistant",
                        "content": (
                            f"<think>\n{reasoning.strip()}\n</think>\n\n{description}"
                        ),
                    }
                ],
                "task_id": case.task_id,
                "sft_role": "proposer",
                "source": "kimi_k3_oracle_pilot",
                "rounds": len(proposer_turns),
                "verified_pairs": demo_count + 1,
            }
    calls = [turn["call"] for turn in proposer_turns]
    calls.extend(call for attempt in executor_attempts for call in attempt["calls"])
    result = {
        "schema_version": 1,
        "mode": "split",
        "task_id": case.task_id,
        "difficulty": case.difficulty,
        "demo_pair_indices": list(case.demo_pair_indices),
        "test_pair_index": case.test_pair_index,
        "termination_reason": termination_reason,
        "proposer_turns": proposer_turns,
        "proposer_rule_format_valid": bool(description and final_turn_is_valid),
        "final_rule_sha256": _sha256_text(description) if description else None,
        "demo_executor_format_valid": (
            all(attempt["format_valid"] for attempt in final_demo_attempts.values())
            and len(final_demo_attempts) == demo_count
        ),
        "demo_exact_fraction": demo_exact / demo_count,
        "train_gate_pass": train_gate_pass,
        "test_executor_format_valid": bool(
            test_attempt and test_attempt["format_valid"]
        ),
        "test_parse_outcome": (
            test_attempt["parse_outcome"] if test_attempt else "not_attempted"
        ),
        "test_exact": test_exact,
        "test_prediction": test_attempt["prediction"] if test_attempt else None,
        "executor_attempts": executor_attempts,
        "model_calls": len(calls),
        "total_tokens": sum(_usage_total(call) for call in calls),
        "proposer_sft_row": proposer_sft_row,
        "executor_sft_rows": executor_sft_rows,
    }
    # Verifier-only targets may exist in local variables and SFT target text,
    # but never as structured request/feedback-trace keys.
    request_trace = [call["request"] for call in calls]
    if _trace_has_forbidden_target_key(request_trace):
        raise AssertionError("model request trace contains a verifier-only target key")
    return result


def summarize_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate authoritative and diagnostic metrics for a completed arm."""
    if not results:
        raise ValueError("cannot summarize empty results")
    mode = results[0]["mode"]
    summary: dict[str, Any] = {
        "mode": mode,
        "tasks": len(results),
        "test_exact": sum(bool(row["test_exact"]) for row in results) / len(results),
        "test_exact_count": sum(bool(row["test_exact"]) for row in results),
    }
    if mode == "direct":
        summary.update(
            {
                "format_valid": sum(bool(row["format_valid"]) for row in results)
                / len(results),
                "parse_outcomes": dict(
                    Counter(row["parse_outcome"] for row in results)
                ),
                "model_calls": len(results),
                "total_tokens": sum(_usage_total(row["calls"][0]) for row in results),
                "sft_direct_rows": sum(
                    row["direct_sft_row"] is not None for row in results
                ),
            }
        )
    else:
        summary.update(
            {
                "proposer_rule_format_valid": sum(
                    bool(row["proposer_rule_format_valid"]) for row in results
                )
                / len(results),
                "demo_executor_format_valid": sum(
                    bool(row["demo_executor_format_valid"]) for row in results
                )
                / len(results),
                "mean_demo_exact_fraction": sum(
                    float(row["demo_exact_fraction"]) for row in results
                )
                / len(results),
                "train_gate_pass": sum(bool(row["train_gate_pass"]) for row in results)
                / len(results),
                "test_executor_format_valid": sum(
                    bool(row["test_executor_format_valid"]) for row in results
                )
                / len(results),
                "test_parse_outcomes": dict(
                    Counter(row["test_parse_outcome"] for row in results)
                ),
                "termination_reasons": dict(
                    Counter(row["termination_reason"] for row in results)
                ),
                "model_calls": sum(int(row["model_calls"]) for row in results),
                "total_tokens": sum(int(row["total_tokens"]) for row in results),
                "sft_proposer_rows": sum(
                    row["proposer_sft_row"] is not None for row in results
                ),
                "sft_executor_rows": sum(
                    len(row["executor_sft_rows"]) for row in results
                ),
            }
        )
    summary["mean_calls_per_task"] = summary["model_calls"] / len(results)
    summary["mean_tokens_per_task"] = summary["total_tokens"] / len(results)
    return summary


async def run_arm(
    *,
    manifest: dict[str, Any],
    mode: str,
    output_dir: Path,
    api_key: str,
    concurrency: int,
    timeout_seconds: float,
    stream: bool = True,
    client_factory: Callable[[], OracleClient] | None = None,
) -> dict[str, Any]:
    """Run or resume one arm, streaming one complete task result per JSONL row."""
    if mode not in {"direct", "split"}:
        raise ValueError(f"unsupported mode: {mode}")
    if concurrency <= 0:
        raise ValueError("concurrency must be positive")
    cases = load_cases(manifest)
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "raw_results.jsonl"
    existing: dict[str, dict[str, Any]] = {}
    if results_path.exists():
        for line in results_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                existing[row["task_id"]] = row

    oracle = manifest["oracle"]
    client = (
        client_factory()
        if client_factory is not None
        else HttpOracleClient(
            endpoint=oracle["endpoint"],
            model=oracle["model"],
            api_key=api_key,
            temperature=oracle["temperature"],
            max_output_tokens=oracle["max_output_tokens"],
            reasoning_effort=oracle["reasoning_effort"],
            timeout_seconds=timeout_seconds,
            stream=stream,
        )
    )
    semaphore = asyncio.Semaphore(concurrency)

    async def evaluate(case: PilotCase) -> dict[str, Any]:
        async with semaphore:
            if mode == "direct":
                return await evaluate_direct(case, client)
            return await evaluate_split(
                case,
                client,
                max_rounds=manifest["split_protocol"]["max_rounds"],
            )

    pending = [case for case in cases if case.task_id not in existing]
    tasks = [asyncio.create_task(evaluate(case)) for case in pending]
    with results_path.open("a", encoding="utf-8") as sink:
        completed_now = 0
        for task in asyncio.as_completed(tasks):
            result = await task
            sink.write(json.dumps(result) + "\n")
            sink.flush()
            existing[result["task_id"]] = result
            completed_now += 1
            print(
                f"{mode}: {len(existing)}/{len(cases)} complete; "
                f"latest={result['task_id']} exact={int(result['test_exact'])}",
                flush=True,
            )
    ordered = [existing[case.task_id] for case in cases]
    summary = summarize_results(ordered)
    run_metadata = {
        "manifest_sha256": _sha256_text(_canonical_json(manifest)),
        "mode": mode,
        "concurrency": concurrency,
        "timeout_seconds": timeout_seconds,
        "stream": stream,
        "resumed_tasks": len(cases) - len(pending),
        "completed_now": completed_now,
        "summary": summary,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(run_metadata, indent=2) + "\n", encoding="utf-8"
    )
    _write_sft_exports(output_dir=output_dir, mode=mode, results=ordered)
    return run_metadata


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as sink:
        for row in rows:
            sink.write(json.dumps(row) + "\n")


def _write_sft_exports(
    *, output_dir: Path, mode: str, results: list[dict[str, Any]]
) -> None:
    if mode == "direct":
        _write_jsonl(
            output_dir / "sft" / "direct_exact.jsonl",
            [row["direct_sft_row"] for row in results if row["direct_sft_row"]],
        )
        return
    _write_jsonl(
        output_dir / "sft" / "proposer_verified.jsonl",
        [row["proposer_sft_row"] for row in results if row["proposer_sft_row"]],
    )
    _write_jsonl(
        output_dir / "sft" / "executor_exact.jsonl",
        [sft for row in results for sft in row["executor_sft_rows"]],
    )


def _bootstrap_delta(
    direct: list[bool], split: list[bool], *, seed: int, samples: int = 20_000
) -> tuple[float, float]:
    rng = random.Random(seed)
    deltas: list[float] = []
    count = len(direct)
    for _ in range(samples):
        total = 0
        for _ in range(count):
            index = rng.randrange(count)
            total += int(split[index]) - int(direct[index])
        deltas.append(total / count)
    deltas.sort()
    return deltas[int(0.025 * samples)], deltas[int(0.975 * samples)]


def compare_arms(
    *, direct_dir: Path, split_dir: Path, output_dir: Path, seed: int
) -> dict[str, Any]:
    """Produce a paired exact comparison and deterministic bootstrap interval."""

    def read_rows(path: Path) -> dict[str, dict[str, Any]]:
        return {
            row["task_id"]: row
            for row in (
                json.loads(line)
                for line in (path / "raw_results.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
                if line.strip()
            )
        }

    direct_rows = read_rows(direct_dir)
    split_rows = read_rows(split_dir)
    if set(direct_rows) != set(split_rows):
        raise ValueError("direct and split arms do not contain identical task ids")
    task_ids = sorted(direct_rows)
    direct_exact = [bool(direct_rows[task_id]["test_exact"]) for task_id in task_ids]
    split_exact = [bool(split_rows[task_id]["test_exact"]) for task_id in task_ids]
    direct_rate = sum(direct_exact) / len(task_ids)
    split_rate = sum(split_exact) / len(task_ids)
    ci_low, ci_high = _bootstrap_delta(direct_exact, split_exact, seed=seed)
    both = sum(direct and split for direct, split in zip(direct_exact, split_exact))
    direct_only = sum(
        direct and not split for direct, split in zip(direct_exact, split_exact)
    )
    split_only = sum(
        split and not direct for direct, split in zip(direct_exact, split_exact)
    )
    neither = len(task_ids) - both - direct_only - split_only
    comparison = {
        "tasks": len(task_ids),
        "direct_exact": direct_rate,
        "split_exact": split_rate,
        "split_minus_direct": split_rate - direct_rate,
        "paired_bootstrap_95_ci": [ci_low, ci_high],
        "paired_outcomes": {
            "both": both,
            "direct_only": direct_only,
            "split_only": split_only,
            "neither": neither,
        },
        "direct_summary": json.loads(
            (direct_dir / "summary.json").read_text(encoding="utf-8")
        )["summary"],
        "split_summary": json.loads(
            (split_dir / "summary.json").read_text(encoding="utf-8")
        )["summary"],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "comparison.json").write_text(
        json.dumps(comparison, indent=2) + "\n", encoding="utf-8"
    )
    report = (
        "# Kimi K3 NVARC pilot\n\n"
        f"- Tasks: {len(task_ids)}\n"
        f"- Direct exact: {direct_rate:.3f} ({sum(direct_exact)}/{len(task_ids)})\n"
        f"- Split exact: {split_rate:.3f} ({sum(split_exact)}/{len(task_ids)})\n"
        f"- Split - direct: {split_rate - direct_rate:+.3f} "
        f"(paired bootstrap 95% CI [{ci_low:+.3f}, {ci_high:+.3f}])\n"
        f"- Paired outcomes: both={both}, direct-only={direct_only}, "
        f"split-only={split_only}, neither={neither}\n"
    )
    (output_dir / "report.md").write_text(report, encoding="utf-8")
    return comparison


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    manifest_parser = subparsers.add_parser("manifest")
    manifest_parser.add_argument("--data-dir", required=True)
    manifest_parser.add_argument("--output", type=Path, required=True)
    manifest_parser.add_argument("--count", type=int, default=32)
    manifest_parser.add_argument("--seed", type=int, default=20_260_902)
    manifest_parser.add_argument("--demo-pairs", type=int, default=3)
    manifest_parser.add_argument(
        "--endpoint", default="https://inference-api.nvidia.com/v1"
    )
    manifest_parser.add_argument("--model", default="nvidia/moonshotai/eccn-kimi-k3")
    manifest_parser.add_argument("--temperature", type=float, default=0.0)
    manifest_parser.add_argument("--max-output-tokens", type=int, default=16_384)
    manifest_parser.add_argument("--reasoning-effort", default="max")
    manifest_parser.add_argument("--max-rounds", type=int, default=3)

    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--manifest", type=Path, required=True)
    run_parser.add_argument("--mode", choices=("direct", "split"), required=True)
    run_parser.add_argument("--output-dir", type=Path, required=True)
    run_parser.add_argument("--api-key-env", default="NVARC_TEACHER_KEY")
    run_parser.add_argument("--concurrency", type=int, default=8)
    run_parser.add_argument("--timeout-seconds", type=float, default=1_200.0)
    run_parser.add_argument(
        "--stream", action=argparse.BooleanOptionalAction, default=True
    )

    compare_parser = subparsers.add_parser("compare")
    compare_parser.add_argument("--direct-dir", type=Path, required=True)
    compare_parser.add_argument("--split-dir", type=Path, required=True)
    compare_parser.add_argument("--output-dir", type=Path, required=True)
    compare_parser.add_argument("--seed", type=int, default=20_260_902)
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    if args.command == "manifest":
        manifest = materialize_manifest(
            data_dir=args.data_dir,
            output=args.output,
            count=args.count,
            seed=args.seed,
            demo_pairs=args.demo_pairs,
            endpoint=args.endpoint,
            model=args.model,
            temperature=args.temperature,
            max_output_tokens=args.max_output_tokens,
            reasoning_effort=args.reasoning_effort,
            max_rounds=args.max_rounds,
        )
        print(json.dumps(manifest["sampling"], indent=2))
    elif args.command == "run":
        manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
        api_key = os.environ.get(args.api_key_env, "")
        if not api_key:
            raise SystemExit(f"missing API key in {args.api_key_env}")
        metadata = asyncio.run(
            run_arm(
                manifest=manifest,
                mode=args.mode,
                output_dir=args.output_dir,
                api_key=api_key,
                concurrency=args.concurrency,
                timeout_seconds=args.timeout_seconds,
                stream=args.stream,
            )
        )
        print(json.dumps(metadata["summary"], indent=2))
    else:
        comparison = compare_arms(
            direct_dir=args.direct_dir,
            split_dir=args.split_dir,
            output_dir=args.output_dir,
            seed=args.seed,
        )
        print(json.dumps(comparison, indent=2))


if __name__ == "__main__":
    main()
