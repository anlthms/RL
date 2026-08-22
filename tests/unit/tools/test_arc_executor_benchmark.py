import asyncio

import pytest

from tools.arc_executor_benchmark import (
    BenchmarkCase,
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
