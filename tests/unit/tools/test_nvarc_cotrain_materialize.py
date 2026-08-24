import json

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from tools.nvarc_cotrain_materialize import (
    EXECUTOR_AGENT,
    PROPOSER_AGENT,
    _proposer_row,
    main,
    role_counts,
)


def _write_nvarc_fixture(path) -> str:
    """A miniature nvarc_ingest-style parquet: 12 train puzzles, 6 pairs each."""
    rows = []
    for index in range(12):
        size = 2 + index % 4  # difficulties 4, 9, 16, 25
        pairs = [
            {
                "input": [[(index + offset) % 10] * size] * size,
                "output": [[(index + offset + 1) % 10]],
            }
            for offset in range(6)
        ]
        rows.append(
            {
                "puzzle_id": f"train_{index:04d}",
                "split": "train",
                "canonical_rule": f"<rules_summary>\nrule {index}\n</rules_summary>",
                "pairs_json": json.dumps(pairs),
                "difficulty": size * size,
            }
        )
    pq.write_table(pa.Table.from_pylist(rows), str(path / "data-00000.parquet"))
    return str(path)


def _write_arc_fixture(path) -> str:
    challenges = {
        "task_a": {
            "train": [{"input": [[1, 0]], "output": [[0, 1]]}],
            "test": [{"input": [[2, 0]]}, {"input": [[0, 2]]}],
        },
        "task_b": {
            "train": [{"input": [[3]], "output": [[4]]}],
            "test": [{"input": [[5]]}],
        },
    }
    solutions = {"task_a": [[[0, 2]], [[2, 0]]], "task_b": [[[6]]]}
    (path / "arc-agi_evaluation_challenges.json").write_text(json.dumps(challenges))
    (path / "arc-agi_evaluation_solutions.json").write_text(json.dumps(solutions))
    return str(path)


def test_role_counts_moves_from_90_10_to_10_90() -> None:
    stage, executors, progress = role_counts(0, window=20, hold_steps=2, num_stages=4)
    assert (stage, executors, progress) == (0, 18, 0.0)
    stage, executors, progress = role_counts(99, window=20, hold_steps=2, num_stages=4)
    assert (stage, executors, progress) == (3, 2, 1.0)


def test_proposer_row_shrinks_demos_to_fit_the_byte_budget() -> None:
    import random

    pairs = [{"input": [[index] * 20] * 20, "output": [[index]]} for index in range(6)]
    puzzle = {"puzzle_id": "p", "pairs_json": json.dumps(pairs)}
    row = _proposer_row(
        puzzle,
        bucket=1,
        demo_pairs=4,
        eval_pairs=2,
        byte_budget=2500,
        rng=random.Random(0),
    )
    assert row is not None
    assert 2 <= len(row["train"]) < 4
    assert len(row["test"]) == 2


def test_proposer_row_skips_puzzles_with_too_few_pairs() -> None:
    import random

    puzzle = {
        "puzzle_id": "p",
        "pairs_json": json.dumps([{"input": [[1]], "output": [[2]]}] * 3),
    }
    assert (
        _proposer_row(
            puzzle,
            bucket=1,
            demo_pairs=3,
            eval_pairs=2,
            byte_budget=8000,
            rng=random.Random(0),
        )
        is None
    )


@pytest.fixture()
def materialized(tmp_path, monkeypatch):
    data_dir = tmp_path / "nvarc"
    arc_dir = tmp_path / "arc"
    out_dir = tmp_path / "out"
    data_dir.mkdir()
    arc_dir.mkdir()
    monkeypatch.setattr(
        "sys.argv",
        [
            "nvarc_cotrain_materialize.py",
            "--data-dir",
            str(data_dir),
            "--arc-data-path",
            str(arc_dir),
            "--output-dir",
            str(out_dir),
            "--steps",
            "8",
            "--window",
            "4",
            "--hold-steps",
            "2",
            "--bucket-edges",
            "4",
            "9",
            "16",
            "--demo-pairs",
            "3",
            "--eval-pairs",
            "2",
        ],
    )
    _write_nvarc_fixture(data_dir)
    _write_arc_fixture(arc_dir)
    main()
    train = [
        json.loads(line) for line in (out_dir / "train.jsonl").read_text().splitlines()
    ]
    val = [
        json.loads(line) for line in (out_dir / "val.jsonl").read_text().splitlines()
    ]
    stats = json.loads((out_dir / "stats.json").read_text())
    return train, val, stats


def test_materializer_bakes_the_role_schedule_into_row_order(materialized) -> None:
    train, _, stats = materialized
    assert len(train) == 32
    assert stats["num_stages"] == 4
    for step, entry in enumerate(stats["step_mixture"]):
        window_rows = train[step * 4 : (step + 1) * 4]
        executors = [row for row in window_rows if row["role"] == "executor"]
        proposers = [row for row in window_rows if row["role"] == "proposer"]
        assert len(executors) == entry["executor_rows"]
        assert len(proposers) == entry["proposer_rows"]
        # Pure staging: every row in the window sits in the stage's bucket.
        assert {row["bucket"] for row in window_rows} == {entry["stage"] + 1}
    # 90:10 at the first ventile, 10:90 at the last.
    assert stats["step_mixture"][0]["executor_rows"] == 4
    assert stats["step_mixture"][-1]["proposer_rows"] == 4


def test_executor_rows_carry_the_single_turn_contract(materialized) -> None:
    train, _, _ = materialized
    row = next(row for row in train if row["role"] == "executor")
    assert row["agent_ref"]["name"] == EXECUTOR_AGENT
    prompt = row["responses_create_params"]["input"][0]["content"]
    assert "<transformation>" in prompt and "<answer>" in prompt
    assert "rules_summary" in prompt
    assert row["target"] and row["test_input"]


def test_proposer_rows_hold_out_disjoint_eval_pairs(materialized) -> None:
    train, _, _ = materialized
    row = next(row for row in train if row["role"] == "proposer")
    assert row["agent_ref"]["name"] == PROPOSER_AGENT
    assert row["responses_create_params"]["input"] == []
    demo_inputs = [json.dumps(pair["input"]) for pair in row["train"]]
    eval_inputs = [json.dumps(pair["input"]) for pair in row["test"]]
    assert not set(demo_inputs) & set(eval_inputs)
    assert len(row["test"]) == 2


def test_validation_rows_are_induction_tasks_for_the_single_turn_agent(
    materialized,
) -> None:
    _, val, _ = materialized
    assert len(val) == 3
    for row in val:
        assert row["agent_ref"]["name"] == EXECUTOR_AGENT
        prompt = row["responses_create_params"]["input"][0]["content"]
        assert "<test_input>" in prompt and "<answer>" in prompt
        assert row["target"] and row["test_input"]
