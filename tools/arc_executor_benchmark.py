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
"""Benchmark oracle ARC rule execution through an OpenAI-compatible endpoint."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
import urllib.error
import urllib.request
from collections import defaultdict
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol

from nemo_rl.environments.arc_agi_generators import (
    SynthTask,
    generate_task,
    render_oracle_description,
    rule_composition_depth,
    rule_family,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
GYM_ROOT = REPO_ROOT / "3rdparty" / "Gym-workspace" / "Gym"
if str(GYM_ROOT) not in sys.path:
    sys.path.insert(0, str(GYM_ROOT))

from resources_servers.arc_agi.logic import (  # noqa: E402  # pyrefly: ignore[import-error]
    BatchVerification,
    Grid,
    PredictionParseError,
    build_executor_prompt,
    build_format_retry_prompt,
    build_test_followup_prompt,
    parse_tagged_grids,
    verify_predictions,
)


ORACLE_RELIABILITY_TARGET = 0.95
FORMAT_VALIDITY_TARGET = 0.99


class ExecutorClient(Protocol):
    """Minimal async chat interface used by the real client and unit tests."""

    async def complete(self, messages: list[dict[str, str]]) -> str:
        """Return assistant text for one chat-completions request."""
        raise NotImplementedError


@dataclass(frozen=True)
class BenchmarkCase:
    """One held-out synthetic transform and its oracle description."""

    task_id: str
    level: int
    rule: str
    family: str
    composition_depth: int
    grid_size: int
    paraphrase_id: int
    description: str
    train_inputs: dict[str, Grid]
    train_targets: dict[str, Grid]
    test_inputs: dict[str, Grid]
    test_targets: dict[str, Grid]


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
    train_format_valid: bool
    train_format_retry_used: bool
    train_all_exact: bool
    train_exact_fraction: float
    train_shape_match_fraction: float
    train_cell_accuracy: float
    test_attempted: bool
    test_format_valid: bool
    test_all_exact: bool
    test_exact_fraction: float
    test_shape_match_fraction: float
    test_cell_accuracy: float
    episode_reliable: bool
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
    """Generate deterministic unaugmented tasks held out by seed and index."""
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
            augment=False,
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
                train_inputs={
                    f"train_{pair_index}": pair["input"]
                    for pair_index, pair in enumerate(task.train_pairs)
                },
                train_targets={
                    f"train_{pair_index}": pair["output"]
                    for pair_index, pair in enumerate(task.train_pairs)
                },
                test_inputs={"test_0": task.test_input},
                test_targets={"test_0": task.target},
            )
        )
    return cases


def _empty_metrics() -> tuple[bool, float, float, float]:
    return False, 0.0, 0.0, 0.0


def _metrics(result: BatchVerification | None) -> tuple[bool, float, float, float]:
    if result is None:
        return _empty_metrics()
    return (
        result.all_exact,
        result.exact_fraction,
        result.shape_match_fraction,
        result.cell_accuracy,
    )


async def evaluate_case(case: BenchmarkCase, client: ExecutorClient) -> CaseResult:
    """Apply one oracle description to train grids, gate, then test grids."""
    started = time.perf_counter()
    train_prompt = build_executor_prompt(
        description=case.description,
        inputs=case.train_inputs,
        tag="predictions",
    )
    messages = [{"role": "user", "content": train_prompt}]
    train_responses: list[str] = []
    train_result: BatchVerification | None = None
    retry_used = False

    for attempt in range(2):
        response = await client.complete(messages)
        train_responses.append(response)
        try:
            predictions = parse_tagged_grids(
                response,
                tag="predictions",
                expected_ids=list(case.train_inputs),
            )
        except PredictionParseError as error:
            if attempt == 1:
                break
            retry_used = True
            messages.extend(
                [
                    {"role": "assistant", "content": response},
                    {
                        "role": "user",
                        "content": build_format_retry_prompt(
                            tag="predictions",
                            expected_ids=list(case.train_inputs),
                            error=str(error),
                        ),
                    },
                ]
            )
            continue
        train_result = verify_predictions(
            predictions=predictions,
            correct=case.train_targets,
        )
        break

    train_format_valid = train_result is not None
    train_all_exact, train_exact, train_shape, train_cell = _metrics(train_result)
    test_response: str | None = None
    test_result: BatchVerification | None = None
    test_format_valid = False
    if train_all_exact:
        messages.extend(
            [
                {"role": "assistant", "content": train_responses[-1]},
                {
                    "role": "user",
                    "content": build_test_followup_prompt(test_inputs=case.test_inputs),
                },
            ]
        )
        test_response = await client.complete(messages)
        try:
            answers = parse_tagged_grids(
                test_response,
                tag="answers",
                expected_ids=list(case.test_inputs),
            )
        except PredictionParseError:
            pass
        else:
            test_format_valid = True
            test_result = verify_predictions(
                predictions=answers, correct=case.test_targets
            )
    test_all_exact, test_exact, test_shape, test_cell = _metrics(test_result)
    return CaseResult(
        task_id=case.task_id,
        level=case.level,
        rule=case.rule,
        family=case.family,
        composition_depth=case.composition_depth,
        grid_size=case.grid_size,
        paraphrase_id=case.paraphrase_id,
        train_format_valid=train_format_valid,
        train_format_retry_used=retry_used,
        train_all_exact=train_all_exact,
        train_exact_fraction=train_exact,
        train_shape_match_fraction=train_shape,
        train_cell_accuracy=train_cell,
        test_attempted=train_all_exact,
        test_format_valid=test_format_valid,
        test_all_exact=test_all_exact,
        test_exact_fraction=test_exact,
        test_shape_match_fraction=test_shape,
        test_cell_accuracy=test_cell,
        episode_reliable=train_all_exact and test_all_exact,
        latency_seconds=time.perf_counter() - started,
        trace={
            "description": case.description,
            "train_inputs": case.train_inputs,
            "train_targets": case.train_targets,
            "train_responses": train_responses,
            "test_inputs": case.test_inputs,
            "test_targets": case.test_targets,
            "test_response": test_response,
        },
    )


def _mean(results: list[CaseResult], field: str) -> float:
    if not results:
        return 0.0
    return sum(float(getattr(result, field)) for result in results) / len(results)


def summarize_group(results: list[CaseResult]) -> dict[str, Any]:
    """Aggregate gate metrics for one complete slice."""
    test_attempts = [result for result in results if result.test_attempted]
    return {
        "count": len(results),
        "train_format_valid": _mean(results, "train_format_valid"),
        "train_format_retry_rate": _mean(results, "train_format_retry_used"),
        "train_all_exact": _mean(results, "train_all_exact"),
        "train_exact_fraction": _mean(results, "train_exact_fraction"),
        "train_shape_match_fraction": _mean(results, "train_shape_match_fraction"),
        "train_cell_accuracy": _mean(results, "train_cell_accuracy"),
        "test_attempt_rate": _mean(results, "test_attempted"),
        "test_format_valid_when_attempted": _mean(
            test_attempts,
            "test_format_valid",
        ),
        "test_all_exact": _mean(results, "test_all_exact"),
        "test_cell_accuracy": _mean(results, "test_cell_accuracy"),
        "episode_reliability": _mean(results, "episode_reliable"),
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
        overall["episode_reliability"] >= ORACLE_RELIABILITY_TARGET
        and overall["train_format_valid"] >= FORMAT_VALIDITY_TARGET
        and overall["test_format_valid_when_attempted"] >= FORMAT_VALIDITY_TARGET
    )
    return {
        "targets": {
            "episode_reliability": ORACLE_RELIABILITY_TARGET,
            "train_format_valid": FORMAT_VALIDITY_TARGET,
            "test_format_valid_when_attempted": FORMAT_VALIDITY_TARGET,
        },
        "overall": overall,
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
    return parser.parse_args()


def execute_benchmark(
    config: BenchmarkConfig,
    *,
    api_key: str,
    output: Path,
) -> dict[str, Any]:
    """Run a benchmark against a ready endpoint and persist its full report."""
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
    )
    report = execute_benchmark(config, api_key=args.api_key, output=args.output)
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()
