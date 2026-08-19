import asyncio

from tools.arc_executor_benchmark import (
    BenchmarkCase,
    evaluate_case,
    summarize_results,
)


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
        train_inputs={"train_0": [[1, 0]], "train_1": [[2, 3]]},
        train_targets={"train_0": [[0, 1]], "train_1": [[3, 2]]},
        test_inputs={"test_0": [[4, 5]]},
        test_targets={"test_0": [[5, 4]]},
    )


def test_oracle_case_retries_format_then_passes_full_episode() -> None:
    executor = FakeExecutor(
        [
            "not json",
            '<predictions>{"train_0": [[0, 1]], "train_1": [[3, 2]]}</predictions>',
            '<answers>{"test_0": [[5, 4]]}</answers>',
        ]
    )
    result = asyncio.run(evaluate_case(_case(), executor))

    assert result.train_format_retry_used
    assert result.train_all_exact
    assert result.test_all_exact
    assert result.episode_reliable
    assert len(executor.messages) == 3
    assert "Do not change or reinterpret" in executor.messages[1][-1]["content"]
    assert "Every training prediction passed" in executor.messages[2][-1]["content"]


def test_training_mismatch_blocks_hidden_test_call() -> None:
    executor = FakeExecutor(
        [
            '<predictions>{"train_0": [[1, 0]], "train_1": [[2, 3]]}</predictions>',
        ]
    )
    result = asyncio.run(evaluate_case(_case(), executor))

    assert result.train_format_valid
    assert not result.train_all_exact
    assert not result.test_attempted
    assert not result.episode_reliable
    assert len(executor.messages) == 1


def test_per_grid_accuracy_does_not_substitute_for_episode_reliability() -> None:
    executor = FakeExecutor(
        [
            '<predictions>{"train_0": [[0, 1]], "train_1": [[3, 2]]}</predictions>',
            '<answers>{"test_0": [[5, 0]]}</answers>',
        ]
    )
    result = asyncio.run(evaluate_case(_case(), executor))

    assert result.train_all_exact
    assert result.test_cell_accuracy == 0.5
    assert not result.test_all_exact
    assert not result.episode_reliable


def test_summary_splits_by_required_axes() -> None:
    exact = asyncio.run(
        evaluate_case(
            _case(),
            FakeExecutor(
                [
                    '<predictions>{"train_0": [[0, 1]], "train_1": [[3, 2]]}</predictions>',
                    '<answers>{"test_0": [[5, 4]]}</answers>',
                ]
            ),
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
            FakeExecutor(
                [
                    '<predictions>{"train_0": [[1, 0]], "train_1": [[2, 3]]}</predictions>'
                ]
            ),
        )
    )
    summary = summarize_results([exact, failed])

    assert summary["overall"]["episode_reliability"] == 0.5
    assert not summary["overall"]["gate_pass"]
    assert set(summary["by_rule_family"]) == {"composition", "dihedral"}
    assert set(summary["by_grid_size"]) == {"2", "4"}
    assert set(summary["by_composition_depth"]) == {"1", "2"}
    assert set(summary["by_description_paraphrase"]) == {"0", "2"}
