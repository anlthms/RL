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
"""Benchmark NVARC rule execution through an OpenAI-compatible endpoint.

Samples held-out puzzles from the ingested NVARC parquet
(``tools/nvarc_ingest.py``), one pair per puzzle, deterministically by seed,
and evaluates each with the exact single-grid prompt contract used by executor
GRPO: one rule, one grid, fresh chat, one format-only retry.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import time
import urllib.error
import urllib.request
from collections import defaultdict
from collections.abc import Callable
from dataclasses import asdict, dataclass
from math import prod
from pathlib import Path
from typing import Any, Protocol

from nemo_rl.environments.arc_agi_grid import (
    Grid,
    best_alignment_cell_accuracy,
    extract_answer_grid,
    grid_shape,
    serialize_grid,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
EXECUTOR_PROMPT_TEMPLATE = (
    REPO_ROOT / "examples" / "prompts" / "nvarc_executor.txt"
).read_text(encoding="utf-8")


ORACLE_RELIABILITY_TARGET = 0.95
FORMAT_VALIDITY_TARGET = 0.99

DEFAULT_BUCKET_EDGES = (162, 380, 621, 700, 750, 783, 810, 840, 870)


class ExecutorClient(Protocol):
    """Minimal async chat interface used by the real client and unit tests."""

    async def complete(
        self, messages: list[dict[str, str]], *, seed: int | None = None
    ) -> str:
        """Return assistant text for one chat-completions request."""
        raise NotImplementedError

    async def complete_with_reasoning(
        self, messages: list[dict[str, str]], *, seed: int | None = None
    ) -> tuple[str, str]:
        """Return (reasoning, content) for one chat-completions request.

        Reasoning is empty when the endpoint has no separated channel.
        """
        return "", await self.complete(messages, seed=seed)


@dataclass(frozen=True)
class BenchmarkCase:
    """One held-out NVARC rule and one grid to apply it to.

    ``bucket`` is the same 1-indexed max-grid-area bucket the training dataset
    reports, so the per-bucket summary lines up with training validation
    metrics.
    """

    task_id: str
    bucket: int
    grid_size: int
    description: str
    input_grid: Grid
    target_grid: Grid


@dataclass(frozen=True)
class CaseResult:
    """Executor competence metrics and trace for one benchmark case."""

    task_id: str
    bucket: int
    grid_size: int
    format_valid: bool
    format_retry_used: bool
    grid_exact: bool
    shape_match: bool
    cell_accuracy: float
    latency_seconds: float
    trace: dict[str, Any]
    sample_index: int = 0


@dataclass(frozen=True)
class BenchmarkConfig:
    """Resolved CLI controls recorded beside benchmark results."""

    base_url: str
    model: str
    seed: int
    count: int
    data_dir: str
    split: str
    bucket_edges: tuple[int, ...]
    max_output_tokens: int
    concurrency: int
    timeout_seconds: float
    temperature: float
    reasoning_effort: str | None = None
    stream: bool = False
    samples_per_case: int = 1
    sample_index_offset: int = 0
    # When set, every first-try exact solve with separated reasoning is also
    # written as a ready-to-train SFT executor row (STaR self-distillation).
    star_output: str | None = None
    star_max_cot_chars: int = 12_000


class OpenAIChatCompletionsClient:
    """Small stdlib client for local vLLM or another OpenAI-compatible server."""

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        api_key: str,
        max_output_tokens: int,
        temperature: float,
        timeout_seconds: float,
        reasoning_effort: str | None = None,
    ) -> None:
        self.url = f"{base_url.rstrip('/')}/chat/completions"
        self.model = model
        self.api_key = api_key
        self.max_output_tokens = max_output_tokens
        self.temperature = temperature
        self.timeout_seconds = timeout_seconds
        self.reasoning_effort = reasoning_effort
        self.opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    async def complete(
        self, messages: list[dict[str, str]], *, seed: int | None = None
    ) -> str:
        _, content = await self.complete_with_reasoning(messages, seed=seed)
        return content

    async def complete_with_reasoning(
        self, messages: list[dict[str, str]], *, seed: int | None = None
    ) -> tuple[str, str]:
        return await asyncio.to_thread(self._complete_sync, messages, seed)

    def _complete_sync(
        self, messages: list[dict[str, str]], seed: int | None
    ) -> tuple[str, str]:
        request_payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "max_tokens": self.max_output_tokens,
            "temperature": self.temperature,
        }
        if self.reasoning_effort:
            request_payload["reasoning_effort"] = self.reasoning_effort
        if seed is not None:
            request_payload["seed"] = seed
        payload = json.dumps(request_payload).encode("utf-8")
        request = urllib.request.Request(
            self.url,
            data=payload,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with self.opener.open(request, timeout=self.timeout_seconds) as response:
                body = json.loads(response.read())
        except urllib.error.HTTPError as error:
            detail = error.read().decode("utf-8", errors="replace")
            raise RuntimeError(
                f"executor endpoint returned HTTP {error.code}: {detail}"
            ) from error
        try:
            message = body["choices"][0]["message"]
            content = message["content"]
        except (KeyError, IndexError, TypeError) as error:
            raise RuntimeError(
                f"executor endpoint returned an invalid chat response: {body}"
            ) from error
        if not isinstance(content, str):
            raise RuntimeError("executor endpoint returned non-text assistant content")
        reasoning = message.get("reasoning_content") or message.get("reasoning")
        return (reasoning if isinstance(reasoning, str) else ""), content


class RetryingStreamingChatCompletionsClient:
    """Adapt the campaign's resumable SSE client to the executor interface."""

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        api_key: str,
        max_output_tokens: int,
        temperature: float,
        timeout_seconds: float,
        reasoning_effort: str | None,
    ) -> None:
        # Deferred to keep local Megatron benchmarking and unit tests free of
        # aiohttp and the Gym-backed oracle harness until streaming is selected.
        from tools.nvarc_oracle_harness import HttpOracleClient

        self._client = HttpOracleClient(
            endpoint=base_url,
            model=model,
            api_key=api_key,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            reasoning_effort=reasoning_effort,
            timeout_seconds=timeout_seconds,
            stream=True,
        )

    async def complete(
        self, messages: list[dict[str, str]], *, seed: int | None = None
    ) -> str:
        _, content = await self.complete_with_reasoning(messages, seed=seed)
        return content

    async def complete_with_reasoning(
        self, messages: list[dict[str, str]], *, seed: int | None = None
    ) -> tuple[str, str]:
        del seed  # The external ceiling uses one greedy sample per case.
        call = await self._client.complete("executor", messages)
        return call["reasoning"], call["content"]


def build_cases(config: BenchmarkConfig) -> list[BenchmarkCase]:
    """Sample held-out NVARC puzzles deterministically, one pair per puzzle.

    Puzzles are sorted by id before sampling so the case set is independent of
    parquet shard layout; ``config.seed`` fixes both the puzzle subset and the
    pair drawn from each.
    """
    # Deferred import: pyarrow is unneeded by the unit-testable pieces above.
    import pyarrow.dataset as ds

    # Select shards by name: the ingest directory also holds stats.json, which
    # a whole-directory dataset would try to parse.
    shards = sorted(Path(config.data_dir).glob("data-*.parquet"))
    if not shards:
        raise FileNotFoundError(f"no data-*.parquet shards under {config.data_dir}")
    table = ds.dataset([str(shard) for shard in shards], format="parquet").to_table(
        columns=["puzzle_id", "canonical_rule", "pairs_json", "difficulty"],
        filter=ds.field("split") == config.split,
    )
    rows = sorted(table.to_pylist(), key=lambda row: row["puzzle_id"])
    if not rows:
        raise ValueError(f"no rows with split={config.split!r} under {config.data_dir}")
    rng = random.Random(config.seed)
    if len(rows) > config.count:
        rows = rng.sample(rows, config.count)

    def bucket(difficulty: int) -> int:
        for index, edge in enumerate(config.bucket_edges):
            if difficulty <= edge:
                return index + 1
        return len(config.bucket_edges) + 1

    cases: list[BenchmarkCase] = []
    for row in rows:
        pair = rng.choice(json.loads(row["pairs_json"]))
        grids = [pair["input"], pair["output"]]
        cases.append(
            BenchmarkCase(
                task_id=row["puzzle_id"],
                bucket=bucket(row["difficulty"]),
                grid_size=max(max(len(g), len(g[0])) for g in grids),
                description=row["canonical_rule"],
                input_grid=pair["input"],
                target_grid=pair["output"],
            )
        )
    return cases


def build_single_grid_prompt(*, description: str, input_grid: Grid) -> str:
    """Render the exact single-grid prompt contract used by executor GRPO."""
    task_body = (
        "<transformation>\n"
        f"{description}\n"
        "</transformation>\n"
        "<input>\n"
        f"{serialize_grid(input_grid)}\n"
        "</input>"
    )
    return EXECUTOR_PROMPT_TEMPLATE.format(task_body)


def build_single_grid_format_retry_prompt() -> str:
    """Request only a corrected rendering of the same single-grid answer."""
    return (
        "Your previous response did not contain a parseable grid inside an "
        "<answer> block. Do not change or reinterpret the transformation. Return "
        "exactly one grid using this form:\n\n"
        "<answer>\n"
        "0 1 2\n"
        "3 4 5\n"
        "</answer>\n\n"
        "Do not include JSON, prose, or a Markdown code fence."
    )


async def evaluate_case(
    case: BenchmarkCase,
    client: ExecutorClient,
    *,
    sample_index: int = 0,
) -> CaseResult:
    """Apply one rule description to one held-out grid in a fresh chat."""
    started = time.perf_counter()
    prompt = build_single_grid_prompt(
        description=case.description,
        input_grid=case.input_grid,
    )
    messages = [{"role": "user", "content": prompt}]
    responses: list[str] = []
    reasonings: list[str] = []
    prediction: Grid | None = None
    retry_used = False

    for attempt in range(2):
        reasoning, response = await client.complete_with_reasoning(
            messages,
            seed=sample_index + 1 + attempt * 1_000_000,
        )
        responses.append(response)
        reasonings.append(reasoning)
        prediction = extract_answer_grid(response)
        if prediction is not None:
            break
        if attempt == 0:
            retry_used = True
            messages.extend(
                [
                    {"role": "assistant", "content": response},
                    {
                        "role": "user",
                        "content": build_single_grid_format_retry_prompt(),
                    },
                ]
            )

    format_valid = prediction is not None
    grid_exact = prediction == case.target_grid
    shape_match = prediction is not None and grid_shape(prediction) == grid_shape(
        case.target_grid
    )
    cell_accuracy = (
        best_alignment_cell_accuracy(prediction, case.target_grid)
        if prediction is not None
        else 0.0
    )
    return CaseResult(
        task_id=case.task_id,
        bucket=case.bucket,
        grid_size=case.grid_size,
        format_valid=format_valid,
        format_retry_used=retry_used,
        grid_exact=grid_exact,
        shape_match=shape_match,
        cell_accuracy=cell_accuracy,
        latency_seconds=time.perf_counter() - started,
        trace={
            "description": case.description,
            "input_grid": case.input_grid,
            "target_grid": case.target_grid,
            "responses": responses,
            "reasonings": reasonings,
            "prediction": prediction,
        },
        sample_index=sample_index,
    )


def _mean(results: list[CaseResult], field: str) -> float:
    if not results:
        return 0.0
    return sum(float(getattr(result, field)) for result in results) / len(results)


def summarize_group(results: list[CaseResult]) -> dict[str, Any]:
    """Aggregate gate metrics for one complete slice."""
    return {
        "count": len(results),
        "format_valid": _mean(results, "format_valid"),
        "format_retry_rate": _mean(results, "format_retry_used"),
        "grid_exact": _mean(results, "grid_exact"),
        "shape_match": _mean(results, "shape_match"),
        "cell_accuracy": _mean(results, "cell_accuracy"),
        "single_grid_reliability": _mean(results, "grid_exact"),
        "mean_latency_seconds": _mean(results, "latency_seconds"),
    }


def summarize_results(results: list[CaseResult]) -> dict[str, Any]:
    """Report overall reliability and the per-difficulty splits."""
    if not results:
        raise ValueError("cannot summarize an empty benchmark")

    def split(key: Callable[[CaseResult], Any]) -> dict[str, dict[str, Any]]:
        groups: dict[str, list[CaseResult]] = defaultdict(list)
        for result in results:
            groups[str(key(result))].append(result)
        return {name: summarize_group(group) for name, group in sorted(groups.items())}

    overall = summarize_group(results)
    overall["gate_pass"] = (
        overall["single_grid_reliability"] >= ORACLE_RELIABILITY_TARGET
        and overall["format_valid"] >= FORMAT_VALIDITY_TARGET
    )
    return {
        "targets": {
            "single_grid_reliability": ORACLE_RELIABILITY_TARGET,
            "format_valid": FORMAT_VALIDITY_TARGET,
        },
        "overall": overall,
        "by_bucket": split(lambda result: result.bucket),
        "by_grid_size": split(lambda result: result.grid_size),
    }


def _pass_at_k(*, samples: int, correct: int, k: int) -> float:
    """Return the unbiased pass@k estimator for one prompt."""
    if not 1 <= k <= samples:
        raise ValueError("k must be between one and the sample count")
    if samples - correct < k:
        return 1.0
    return 1.0 - prod(
        (samples - correct - index) / (samples - index) for index in range(k)
    )


def summarize_sampled_results(results: list[CaseResult]) -> dict[str, Any]:
    """Summarize pass@k and modal-grid consensus for grouped samples."""
    if not results:
        raise ValueError("cannot summarize empty sampled results")
    by_task: dict[str, list[CaseResult]] = defaultdict(list)
    for result in results:
        by_task[result.task_id].append(result)
    sample_counts = {len(group) for group in by_task.values()}
    if len(sample_counts) != 1:
        raise ValueError("every task must have the same number of samples")
    samples = sample_counts.pop()
    expected_indices = list(
        range(
            min(row.sample_index for row in results),
            samples + min(row.sample_index for row in results),
        )
    )
    for group in by_task.values():
        if sorted(row.sample_index for row in group) != expected_indices:
            raise ValueError("every task must have the same contiguous sample indices")

    requested_ks = sorted({1, min(8, samples), samples})
    pass_sums = {k: 0.0 for k in requested_ks}
    consensus_exact = 0
    consensus_agreement = 0.0
    unique_valid_predictions = 0
    for group in by_task.values():
        correct = sum(row.grid_exact for row in group)
        for k in requested_ks:
            pass_sums[k] += _pass_at_k(samples=samples, correct=correct, k=k)

        prediction_counts: dict[tuple[tuple[int, ...], ...], int] = defaultdict(int)
        for row in group:
            prediction = row.trace["prediction"]
            if prediction is not None:
                prediction_counts[tuple(tuple(line) for line in prediction)] += 1
        unique_valid_predictions += len(prediction_counts)
        if prediction_counts:
            modal_prediction, modal_count = max(
                prediction_counts.items(), key=lambda item: (item[1], item[0])
            )
            target = tuple(tuple(line) for line in group[0].trace["target_grid"])
            consensus_exact += modal_prediction == target
            consensus_agreement += modal_count / samples

    task_count = len(by_task)
    return {
        "task_count": task_count,
        "samples_per_case": samples,
        "rollout_exact": _mean(results, "grid_exact"),
        "pass_at_k": {str(k): pass_sums[k] / task_count for k in requested_ks},
        "oracle_any_exact": sum(
            any(row.grid_exact for row in group) for group in by_task.values()
        )
        / task_count,
        "consensus_exact": consensus_exact / task_count,
        "mean_consensus_agreement": consensus_agreement / task_count,
        "mean_unique_valid_predictions": unique_valid_predictions / task_count,
    }


async def run_benchmark(
    *,
    cases: list[BenchmarkCase],
    client: ExecutorClient,
    concurrency: int,
    samples_per_case: int = 1,
    sample_index_offset: int = 0,
) -> list[CaseResult]:
    """Evaluate cases concurrently while preserving deterministic output order."""
    if concurrency <= 0:
        raise ValueError("concurrency must be positive")
    if samples_per_case <= 0:
        raise ValueError("samples_per_case must be positive")
    if sample_index_offset < 0:
        raise ValueError("sample_index_offset must be nonnegative")
    semaphore = asyncio.Semaphore(concurrency)

    async def evaluate(case: BenchmarkCase, sample_index: int) -> CaseResult:
        async with semaphore:
            return await evaluate_case(case, client, sample_index=sample_index)

    return list(
        await asyncio.gather(
            *(
                evaluate(case, sample_index_offset + sample_index)
                for case in cases
                for sample_index in range(samples_per_case)
            )
        )
    )


def _comma_separated_ints(value: str) -> tuple[int, ...]:
    values = tuple(int(item) for item in value.split(",") if item)
    if not values:
        raise argparse.ArgumentTypeError("value must contain at least one integer")
    return values


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:10240/v1")
    parser.add_argument("--model", required=True)
    parser.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY", "EMPTY"))
    parser.add_argument("--seed", type=int, default=93_821)
    parser.add_argument("--count", type=int, default=150)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--split", default="executor_val")
    parser.add_argument(
        "--bucket-edges", type=_comma_separated_ints, default=DEFAULT_BUCKET_EDGES
    )
    parser.add_argument("--max-output-tokens", type=int, default=6144)
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--timeout-seconds", type=float, default=300.0)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--reasoning-effort", default=None)
    parser.add_argument(
        "--stream", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument("--samples-per-case", type=int, default=1)
    parser.add_argument("--sample-index-offset", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--star-output",
        default=None,
        help="also write first-try exact solves as SFT executor rows (JSONL)",
    )
    parser.add_argument("--star-max-cot-chars", type=int, default=12_000)
    return parser.parse_args()


def execute_benchmark(
    config: BenchmarkConfig,
    *,
    api_key: str,
    output: Path,
) -> dict[str, Any]:
    """Run a benchmark against a ready endpoint and persist its full report."""
    cases = build_cases(config)
    client_class = (
        RetryingStreamingChatCompletionsClient
        if config.stream
        else OpenAIChatCompletionsClient
    )
    client = client_class(
        base_url=config.base_url,
        model=config.model,
        api_key=api_key,
        max_output_tokens=config.max_output_tokens,
        temperature=config.temperature,
        timeout_seconds=config.timeout_seconds,
        reasoning_effort=config.reasoning_effort,
    )
    results = asyncio.run(
        run_benchmark(
            cases=cases,
            client=client,
            concurrency=config.concurrency,
            samples_per_case=config.samples_per_case,
            sample_index_offset=config.sample_index_offset,
        )
    )
    summary: dict[str, Any] = summarize_results(results)
    if config.samples_per_case > 1:
        summary["sampled"] = summarize_sampled_results(results)
    report: dict[str, Any] = {
        "config": asdict(config),
        "summary": summary,
        "results": [asdict(result) for result in results],
    }
    if config.star_output:
        star_rows = 0
        star_path = Path(config.star_output)
        star_path.parent.mkdir(parents=True, exist_ok=True)
        with open(star_path, "w", encoding="utf-8") as sink:
            for result in results:
                reasoning = result.trace["reasonings"][0]
                if not (
                    result.grid_exact
                    and not result.format_retry_used
                    and reasoning
                    and len(reasoning) <= config.star_max_cot_chars
                ):
                    continue
                prompt = build_single_grid_prompt(
                    description=result.trace["description"],
                    input_grid=result.trace["input_grid"],
                )
                answer = (
                    f"<answer>\n{serialize_grid(result.trace['prediction'])}\n</answer>"
                )
                target = f"<think>\n{reasoning.strip()}\n</think>\n\n{answer}"
                sink.write(
                    json.dumps(
                        {
                            "messages": [
                                {"role": "user", "content": prompt},
                                {"role": "assistant", "content": target},
                            ],
                            "task_id": result.task_id,
                            "sft_role": "executor",
                            "source": "star",
                        }
                    )
                    + "\n"
                )
                star_rows += 1
        summary["star_rows"] = star_rows
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    args = _parse_args()
    config = BenchmarkConfig(
        base_url=args.base_url,
        model=args.model,
        seed=args.seed,
        count=args.count,
        data_dir=args.data_dir,
        split=args.split,
        bucket_edges=args.bucket_edges,
        max_output_tokens=args.max_output_tokens,
        concurrency=args.concurrency,
        timeout_seconds=args.timeout_seconds,
        temperature=args.temperature,
        reasoning_effort=args.reasoning_effort,
        stream=args.stream,
        samples_per_case=args.samples_per_case,
        sample_index_offset=args.sample_index_offset,
        star_output=args.star_output,
        star_max_cot_chars=args.star_max_cot_chars,
    )
    report = execute_benchmark(config, api_key=args.api_key, output=args.output)
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()
