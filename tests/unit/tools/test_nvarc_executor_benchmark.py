import asyncio
import json

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from tools.nvarc_executor_benchmark import (
    BenchmarkCase,
    BenchmarkConfig,
    build_cases,
    build_single_grid_prompt,
    evaluate_case,
    summarize_results,
)
from tools.nvarc_executor_benchmark_megatron import _resolve_weights_path


class FakeExecutor:
    def __init__(self, responses: list[str]) -> None:
        self.responses = list(responses)
        self.messages: list[list[dict[str, str]]] = []

    async def complete(self, messages: list[dict[str, str]]) -> str:
        self.messages.append([dict(message) for message in messages])
        return self.responses.pop(0)

    async def complete_with_reasoning(
        self, messages: list[dict[str, str]]
    ) -> tuple[str, str]:
        return "", await self.complete(messages)


def _case(**overrides) -> BenchmarkCase:
    fields = {
        "task_id": "case",
        "bucket": 1,
        "grid_size": 2,
        "description": "Reverse every row.",
        "input_grid": [[4, 5]],
        "target_grid": [[5, 4]],
    }
    fields.update(overrides)
    return BenchmarkCase(**fields)


def _config(data_dir: str, **overrides) -> BenchmarkConfig:
    fields = {
        "base_url": "http://unused",
        "model": "unused",
        "seed": 7,
        "count": 4,
        "data_dir": data_dir,
        "split": "executor_val",
        "bucket_edges": (4, 9),
        "max_output_tokens": 16,
        "concurrency": 1,
        "timeout_seconds": 1.0,
        "temperature": 0.0,
    }
    fields.update(overrides)
    return BenchmarkConfig(**fields)


def _write_fixture(path) -> str:
    rows = []
    for index in range(8):
        split = "executor_val" if index < 6 else "train"
        size = 2 + index % 3
        pairs = [
            {
                "input": [[(index + offset) % 10] * size] * size,
                "output": [[offset]],
            }
            for offset in range(3)
        ]
        rows.append(
            {
                "puzzle_id": f"puzzle_{index:02d}",
                "split": split,
                "canonical_rule": f"<rules_summary>\nrule {index}\n</rules_summary>",
                "pairs_json": json.dumps(pairs),
                "difficulty": size * size,
            }
        )
    pq.write_table(pa.Table.from_pylist(rows), str(path / "data-00000.parquet"))
    return str(path)


def test_case_retries_format_then_solves_single_grid() -> None:
    executor = FakeExecutor(["no grid here", "<answer>\n5 4\n</answer>"])
    result = asyncio.run(evaluate_case(_case(), executor))

    assert result.format_retry_used
    assert result.format_valid
    assert result.grid_exact
    assert result.cell_accuracy == 1.0
    # The retry must not alter the rule: same fresh chat, plus a format-only nudge.
    assert len(executor.messages) == 2
    assert "Do not change or reinterpret the transformation" in str(
        executor.messages[1][-1]["content"]
    )


def test_case_with_two_format_failures_scores_zero() -> None:
    executor = FakeExecutor(["nope", "still nope"])
    result = asyncio.run(evaluate_case(_case(), executor))

    assert not result.format_valid
    assert not result.grid_exact
    assert result.cell_accuracy == 0.0


def test_prompt_contains_the_color_legend_and_contract() -> None:
    prompt = build_single_grid_prompt(description="Rotate.", input_grid=[[1, 2]])
    assert "0=black" in prompt and "6=fuchsia" in prompt
    assert "8=teal" in prompt and "9=brown" in prompt
    assert "<transformation>\nRotate.\n</transformation>" in prompt
    assert "<input>\n1 2\n</input>" in prompt


def test_cases_are_deterministic_and_val_only(tmp_path) -> None:
    data_dir = _write_fixture(tmp_path)
    first = build_cases(_config(data_dir))
    second = build_cases(_config(data_dir))

    assert [case.task_id for case in first] == [case.task_id for case in second]
    assert [case.input_grid for case in first] == [case.input_grid for case in second]
    assert len(first) == 4
    # train-split puzzles must never be benchmarked as held-out
    assert all(not case.task_id.endswith(("06", "07")) for case in first)
    # buckets: areas 4, 9, 16 with edges (4, 9) -> buckets 1, 2, 3
    assert {case.bucket for case in first} <= {1, 2, 3}


def test_cases_reject_a_missing_split(tmp_path) -> None:
    data_dir = _write_fixture(tmp_path)
    with pytest.raises(ValueError, match="no rows with split"):
        build_cases(_config(data_dir, split="nonexistent"))


def test_summary_splits_by_bucket_and_grid_size() -> None:
    exact = asyncio.run(
        evaluate_case(_case(), FakeExecutor(["<answer>\n5 4\n</answer>"]))
    )
    failed = asyncio.run(
        evaluate_case(
            _case(task_id="failed", bucket=3, grid_size=4),
            FakeExecutor(["<answer>\n4 5\n</answer>"]),
        )
    )
    summary = summarize_results([exact, failed])

    assert summary["overall"]["single_grid_reliability"] == 0.5
    assert not summary["overall"]["gate_pass"]
    assert set(summary["by_bucket"]) == {"1", "3"}
    assert set(summary["by_grid_size"]) == {"2", "4"}


def test_megatron_checkpoint_resolves_step_or_weights_directory(tmp_path) -> None:
    weights = tmp_path / "step_20" / "policy" / "weights"
    weights.mkdir(parents=True)

    assert _resolve_weights_path(tmp_path / "step_20") == weights
    assert _resolve_weights_path(weights) == weights


def test_megatron_checkpoint_rejects_missing_weights(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="policy/weights"):
        _resolve_weights_path(tmp_path / "step_20")
