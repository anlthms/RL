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
"""Benchmark oracle ARC rule execution through an OpenAI-compatible endpoint.

Two task sources share one harness and one prompt contract: ``synthetic``
regenerates the fixed held-out generator tasks (the historical 150-case
control), and ``nvarc`` samples held-out puzzles from the ingested NVARC
parquet (``tools/nvarc_ingest.py``), one pair per puzzle, deterministically by
seed. Comparing a checkpoint on both answers the transfer question directly.
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
from pathlib import Path
from typing import Any, Protocol

from nemo_rl.environments.arc_agi_grid import (
    Grid,
    best_alignment_cell_accuracy,
    extract_answer_grid,
    grid_shape,
    serialize_grid,
)
from nemo_rl.environments.arc_agi_generators import (
    SynthTask,
    generate_task,
    render_oracle_description,
    rule_composition_depth,
    rule_family,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
EXECUTOR_PROMPT_TEMPLATE = (
    REPO_ROOT / "examples" / "prompts" / "arc_executor.txt"
).read_text(encoding="utf-8")
NVARC_EXECUTOR_PROMPT_TEMPLATE = (
    REPO_ROOT / "examples" / "prompts" / "nvarc_executor.txt"
).read_text(encoding="utf-8")
PROMPT_TEMPLATES = {
    "synthetic": EXECUTOR_PROMPT_TEMPLATE,
    "nvarc": NVARC_EXECUTOR_PROMPT_TEMPLATE,
}


ORACLE_RELIABILITY_TARGET = 0.95
FORMAT_VALIDITY_TARGET = 0.99


class ExecutorClient(Protocol):
    """Minimal async chat interface used by the real client and unit tests."""

    async def complete(self, messages: list[dict[str, str]]) -> str:
        """Return assistant text for one chat-completions request."""
        raise NotImplementedError


@dataclass(frozen=True)
class BenchmarkCase:
    """One held-out transform description and one grid to apply it to.

    ``source`` selects the prompt template; for NVARC cases ``level`` is the
    difficulty (area) bucket, ``family`` is ``"nvarc"``, and the synthetic-only
    axes (``composition_depth``, ``paraphrase_id``) are pinned to 0.
    """

    task_id: str
    level: int
    rule: str
    family: str
    composition_depth: int
    grid_size: int
    paraphrase_id: int
    description: str
    input_grid: Grid
    target_grid: Grid
    source: str = "synthetic"


@dataclass(frozen=True)
class CaseResult:
    """Executor competence metrics and trace for one benchmark case."""

    task_id: str
    level: int
    rule: str
    family: str
    composition_depth: int
    grid_size: int
    paraphrase_id: int
    format_valid: bool
    format_retry_used: bool
    grid_exact: bool
    shape_match: bool
    cell_accuracy: float
    latency_seconds: float
    trace: dict[str, Any]


@dataclass(frozen=True)
class BenchmarkConfig:
    """Resolved CLI controls recorded beside benchmark results."""

    base_url: str
    model: str
    seed: int
    count: int
    levels: tuple[int, ...]
    paraphrases: tuple[int, ...]
    num_train_pairs: int
    max_input_dim: int
    max_output_tokens: int
    concurrency: int
    timeout_seconds: float
    temperature: float
    task_source: str = "synthetic"
    nvarc_data_dir: str | None = None
    nvarc_split: str = "executor_val"
    nvarc_bucket_edges: tuple[int, ...] = (380, 700, 783, 840)


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
    ) -> None:
        self.url = f"{base_url.rstrip('/')}/chat/completions"
        self.model = model
        self.api_key = api_key
        self.max_output_tokens = max_output_tokens
        self.temperature = temperature
        self.timeout_seconds = timeout_seconds
        self.opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    async def complete(self, messages: list[dict[str, str]]) -> str:
        return await asyncio.to_thread(self._complete_sync, messages)

    def _complete_sync(self, messages: list[dict[str, str]]) -> str:
        payload = json.dumps(
            {
                "model": self.model,
                "messages": messages,
                "max_tokens": self.max_output_tokens,
                "temperature": self.temperature,
            }
        ).encode("utf-8")
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
            content = body["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as error:
            raise RuntimeError(
                f"executor endpoint returned an invalid chat response: {body}"
            ) from error
        if not isinstance(content, str):
            raise RuntimeError("executor endpoint returned non-text assistant content")
        return content


def _task_grid_size(task: SynthTask) -> int:
    grids = [
        *(pair["input"] for pair in task.train_pairs),
        *(pair["output"] for pair in task.train_pairs),
        task.test_input,
        task.target,
    ]
    return max(max(len(grid), len(grid[0])) for grid in grids)


def build_cases(config: BenchmarkConfig) -> list[BenchmarkCase]:
    """Generate deterministic tasks held out by seed and index."""
    cases: list[BenchmarkCase] = []
    for index in range(config.count):
        level = config.levels[index % len(config.levels)]
        paraphrase_id = config.paraphrases[index % len(config.paraphrases)]
        task = generate_task(
            seed=config.seed,
            index=index,
            level=level,
            num_train_pairs=config.num_train_pairs,
            max_input_dim=config.max_input_dim,
        )
        cases.append(
            BenchmarkCase(
                task_id=task.task_id,
                level=task.level,
                rule=task.rule,
                family=rule_family(task.rule),
                composition_depth=rule_composition_depth(task.rule),
                grid_size=_task_grid_size(task),
                paraphrase_id=paraphrase_id,
                description=render_oracle_description(
                    task.rule, paraphrase_id=paraphrase_id
                ),
                input_grid=task.test_input,
                target_grid=task.target,
            )
        )
    return cases


def build_nvarc_cases(config: BenchmarkConfig) -> list[BenchmarkCase]:
    """Sample held-out NVARC puzzles deterministically, one pair per puzzle.

    Puzzles are sorted by id before sampling so the case set is independent of
    parquet shard layout; ``config.seed`` fixes both the puzzle subset and the
    pair drawn from each. ``level`` is the same 1-indexed area bucket the
    training dataset reports, so the per-level summary lines up with training
    validation metrics.
    """
    # Deferred import: pyarrow is unneeded (and unimported) on the synthetic path.
    import pyarrow.dataset as ds

    if not config.nvarc_data_dir:
        raise ValueError("task_source=nvarc requires --nvarc-data-dir")
    # Select shards by name: the ingest directory also holds stats.json, which
    # a whole-directory dataset would try to parse.
    shards = sorted(Path(config.nvarc_data_dir).glob("data-*.parquet"))
    if not shards:
        raise FileNotFoundError(
            f"no data-*.parquet shards under {config.nvarc_data_dir}"
        )
    table = ds.dataset([str(shard) for shard in shards], format="parquet").to_table(
        columns=["puzzle_id", "canonical_rule", "pairs_json", "difficulty"],
        filter=ds.field("split") == config.nvarc_split,
    )
    rows = sorted(table.to_pylist(), key=lambda row: row["puzzle_id"])
    if not rows:
        raise ValueError(
            f"no rows with split={config.nvarc_split!r} under {config.nvarc_data_dir}"
        )
    rng = random.Random(config.seed)
    if len(rows) > config.count:
        rows = rng.sample(rows, config.count)

    def bucket(difficulty: int) -> int:
        for index, edge in enumerate(config.nvarc_bucket_edges):
            if difficulty <= edge:
                return index + 1
        return len(config.nvarc_bucket_edges) + 1

    cases: list[BenchmarkCase] = []
    for row in rows:
        pair = rng.choice(json.loads(row["pairs_json"]))
        grids = [pair["input"], pair["output"]]
        cases.append(
            BenchmarkCase(
                task_id=row["puzzle_id"],
                level=bucket(row["difficulty"]),
                rule=row["puzzle_id"],
                family="nvarc",
                composition_depth=0,
                grid_size=max(max(len(g), len(g[0])) for g in grids),
                paraphrase_id=0,
                description=row["canonical_rule"],
                input_grid=pair["input"],
                target_grid=pair["output"],
                source="nvarc",
            )
        )
    return cases


def build_single_grid_prompt(
    *, description: str, input_grid: Grid, source: str = "synthetic"
) -> str:
    """Render the exact single-grid prompt contract used by executor GRPO."""
    task_body = (
        "<transformation>\n"
        f"{description}\n"
        "</transformation>\n"
        "<input>\n"
        f"{serialize_grid(input_grid)}\n"
        "</input>"
    )
    return PROMPT_TEMPLATES[source].format(task_body)


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


async def evaluate_case(case: BenchmarkCase, client: ExecutorClient) -> CaseResult:
    """Apply one oracle description to one held-out grid in a fresh chat."""
    started = time.perf_counter()
    prompt = build_single_grid_prompt(
        description=case.description,
        input_grid=case.input_grid,
        source=case.source,
    )
    messages = [{"role": "user", "content": prompt}]
    responses: list[str] = []
    prediction: Grid | None = None
    retry_used = False

    for attempt in range(2):
        response = await client.complete(messages)
        responses.append(response)
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
        level=case.level,
        rule=case.rule,
        family=case.family,
        composition_depth=case.composition_depth,
        grid_size=case.grid_size,
        paraphrase_id=case.paraphrase_id,
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
            "prediction": prediction,
        },
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
    """Report overall reliability and required design-document splits."""
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
        "by_level": split(lambda result: result.level),
        "by_rule_family": split(lambda result: result.family),
        "by_grid_size": split(lambda result: result.grid_size),
        "by_composition_depth": split(lambda result: result.composition_depth),
        "by_description_paraphrase": split(lambda result: result.paraphrase_id),
    }


async def run_benchmark(
    *,
    cases: list[BenchmarkCase],
    client: ExecutorClient,
    concurrency: int,
) -> list[CaseResult]:
    """Evaluate cases concurrently while preserving deterministic output order."""
    if concurrency <= 0:
        raise ValueError("concurrency must be positive")
    semaphore = asyncio.Semaphore(concurrency)

    async def evaluate(case: BenchmarkCase) -> CaseResult:
        async with semaphore:
            return await evaluate_case(case, client)

    return list(await asyncio.gather(*(evaluate(case) for case in cases)))


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
    parser.add_argument("--levels", type=_comma_separated_ints, default=(1, 2, 3, 4, 5))
    parser.add_argument("--paraphrases", type=_comma_separated_ints, default=(0, 1, 2))
    parser.add_argument("--num-train-pairs", type=int, default=3)
    parser.add_argument("--max-input-dim", type=int, default=12)
    parser.add_argument("--max-output-tokens", type=int, default=4096)
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--timeout-seconds", type=float, default=300.0)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--task-source", choices=("synthetic", "nvarc"), default="synthetic"
    )
    parser.add_argument("--nvarc-data-dir", default=None)
    parser.add_argument("--nvarc-split", default="executor_val")
    parser.add_argument(
        "--nvarc-bucket-edges",
        type=_comma_separated_ints,
        default=(380, 700, 783, 840),
    )
    return parser.parse_args()


def execute_benchmark(
    config: BenchmarkConfig,
    *,
    api_key: str,
    output: Path,
) -> dict[str, Any]:
    """Run a benchmark against a ready endpoint and persist its full report."""
    if config.task_source == "nvarc":
        cases = build_nvarc_cases(config)
    else:
        if not config.levels or any(
            level not in {0, 1, 2, 3, 4, 5} for level in config.levels
        ):
            raise ValueError("levels must contain only integers from 0 through 5")
        if not config.paraphrases or any(
            value not in {0, 1, 2} for value in config.paraphrases
        ):
            raise ValueError("paraphrases must contain only 0, 1, or 2")
        cases = build_cases(config)
    client = OpenAIChatCompletionsClient(
        base_url=config.base_url,
        model=config.model,
        api_key=api_key,
        max_output_tokens=config.max_output_tokens,
        temperature=config.temperature,
        timeout_seconds=config.timeout_seconds,
    )
    results = asyncio.run(
        run_benchmark(cases=cases, client=client, concurrency=config.concurrency)
    )
    report = {
        "config": asdict(config),
        "summary": summarize_results(results),
        "results": [asdict(result) for result in results],
    }
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
        levels=args.levels,
        paraphrases=args.paraphrases,
        num_train_pairs=args.num_train_pairs,
        max_input_dim=args.max_input_dim,
        max_output_tokens=args.max_output_tokens,
        concurrency=args.concurrency,
        timeout_seconds=args.timeout_seconds,
        temperature=args.temperature,
        task_source=args.task_source,
        nvarc_data_dir=args.nvarc_data_dir,
        nvarc_split=args.nvarc_split,
        nvarc_bucket_edges=args.nvarc_bucket_edges,
    )
    report = execute_benchmark(config, api_key=args.api_key, output=args.output)
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()
