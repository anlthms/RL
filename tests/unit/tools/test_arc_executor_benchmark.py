import asyncio
import json

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from tools.arc_executor_benchmark import (
    BenchmarkCase,
    BenchmarkConfig,
    build_nvarc_cases,
    build_single_grid_prompt,
    evaluate_case,
    summarize_results,
)
from tools.arc_executor_benchmark_megatron import _resolve_weights_path


class FakeExecutor:
    def __init__(self, responses: list[str]) -> None:
        self.responses = list(responses)
        self.messages: list[list[dict[str, str]]] = []

    async def complete(self, messages: list[dict[str, str]]) -> str:
        self.messages.append([dict(message) for message in messages])
        return self.responses.pop(0)


def _case() -> BenchmarkCase:
    return BenchmarkCase(
        task_id="case",
        level=1,
        rule="flip_h",
        family="dihedral",
        composition_depth=1,
        grid_size=2,
        paraphrase_id=0,
        description="Reverse every row.",
        input_grid=[[4, 5]],
        target_grid=[[5, 4]],
    )


def test_oracle_case_retries_format_then_solves_single_grid() -> None:
    executor = FakeExecutor(
        [
            "not json",
            "<answer>\n5 4\n</answer>",
        ]
    )
    result = asyncio.run(evaluate_case(_case(), executor))

    assert result.format_retry_used
    assert result.format_valid
    assert result.grid_exact
    assert len(executor.messages) == 2
    assert "Do not change or reinterpret" in executor.messages[1][-1]["content"]


def test_wrong_single_grid_is_parseable_but_not_reliable() -> None:
    executor = FakeExecutor(["<answer>\n4 5\n</answer>"])
    result = asyncio.run(evaluate_case(_case(), executor))

    assert result.format_valid
    assert result.shape_match
    assert not result.grid_exact
    assert len(executor.messages) == 1


def test_cell_accuracy_does_not_substitute_for_single_grid_reliability() -> None:
    executor = FakeExecutor(["<answer>\n5 0\n</answer>"])
    result = asyncio.run(evaluate_case(_case(), executor))

    assert result.cell_accuracy == 0.5
    assert not result.grid_exact


def test_single_grid_prompt_matches_executor_training_contract() -> None:
    prompt = build_single_grid_prompt(
        description="Reverse every row.", input_grid=[[4, 5]]
    )

    assert "<transformation>\nReverse every row.\n</transformation>" in prompt
    assert "<input>\n4 5\n</input>" in prompt
    assert "<answer>" in prompt
    assert "<predictions>" not in prompt


def test_summary_splits_by_required_axes() -> None:
    exact = asyncio.run(
        evaluate_case(
            _case(),
            FakeExecutor(["<answer>\n5 4\n</answer>"]),
        )
    )
    failed_case = BenchmarkCase(
        **{
            **_case().__dict__,
            "task_id": "composition",
            "level": 5,
            "family": "composition",
            "composition_depth": 2,
            "grid_size": 4,
            "paraphrase_id": 2,
        }
    )
    failed = asyncio.run(
        evaluate_case(
            failed_case,
            FakeExecutor(["<answer>\n4 5\n</answer>"]),
        )
    )
    summary = summarize_results([exact, failed])

    assert summary["overall"]["single_grid_reliability"] == 0.5
    assert not summary["overall"]["gate_pass"]
    assert set(summary["by_rule_family"]) == {"composition", "dihedral"}
    assert set(summary["by_grid_size"]) == {"2", "4"}
    assert set(summary["by_composition_depth"]) == {"1", "2"}
    assert set(summary["by_description_paraphrase"]) == {"0", "2"}


def test_megatron_checkpoint_resolves_step_or_weights_directory(tmp_path) -> None:
    weights = tmp_path / "step_20" / "policy" / "weights"
    weights.mkdir(parents=True)

    assert _resolve_weights_path(tmp_path / "step_20") == weights
    assert _resolve_weights_path(weights) == weights


def test_megatron_checkpoint_rejects_missing_weights(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="policy/weights"):
        _resolve_weights_path(tmp_path / "step_20")


def _nvarc_config(data_dir: str, **overrides) -> BenchmarkConfig:
    config = {
        "base_url": "http://unused",
        "model": "unused",
        "seed": 7,
        "count": 4,
        "levels": (1,),
        "paraphrases": (0,),
        "num_train_pairs": 3,
        "max_input_dim": 12,
        "max_output_tokens": 16,
        "concurrency": 1,
        "timeout_seconds": 1.0,
        "temperature": 0.0,
        "task_source": "nvarc",
        "nvarc_data_dir": data_dir,
        "nvarc_bucket_edges": (4, 9),
    }
    config.update(overrides)
    return BenchmarkConfig(**config)


def _write_nvarc_fixture(path) -> str:
    rows = []
    for index in range(8):
        split = "executor_val" if index < 6 else "train"
        size = 2 + index % 3
        pairs = [
            {"input": [[index % 10] * size] * size, "output": [[offset]]}
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


def test_nvarc_cases_are_deterministic_and_val_only(tmp_path) -> None:
    data_dir = _write_nvarc_fixture(tmp_path)
    first = build_nvarc_cases(_nvarc_config(data_dir))
    second = build_nvarc_cases(_nvarc_config(data_dir))

    assert [case.task_id for case in first] == [case.task_id for case in second]
    assert [case.input_grid for case in first] == [case.input_grid for case in second]
    assert len(first) == 4
    assert all(case.source == "nvarc" for case in first)
    assert all(case.family == "nvarc" for case in first)
    # train-split puzzles must never be benchmarked as held-out
    assert all(
        case.task_id != "puzzle_06" and case.task_id != "puzzle_07" for case in first
    )
    # buckets: areas 4, 9, 16 with edges (4, 9) -> buckets 1, 2, 3
    assert set(case.level for case in first) <= {1, 2, 3}


def test_nvarc_cases_require_data_dir(tmp_path) -> None:
    with pytest.raises(ValueError, match="nvarc-data-dir"):
        build_nvarc_cases(
            _nvarc_config(_write_nvarc_fixture(tmp_path), nvarc_data_dir=None)
        )


def test_nvarc_prompt_uses_the_color_legend_template() -> None:
    prompt = build_single_grid_prompt(
        description="Rotate.", input_grid=[[1, 2]], source="nvarc"
    )
    assert "0=black" in prompt and "9=brown/maroon" in prompt
    assert "<transformation>\nRotate.\n</transformation>" in prompt
    synthetic = build_single_grid_prompt(description="Rotate.", input_grid=[[1, 2]])
    assert "0=black" not in synthetic
