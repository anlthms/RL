import importlib.util
import json

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from tools.nvarc_sft_materialize import (
    NO_THINK_PREFIX,
    PROPOSER_INSTRUCTIONS,
    executor_example,
    proposer_example,
    main,
)

CANONICAL_RULE = (
    "<rules_summary>\nReverse each row.\n</rules_summary>\n\n"
    "<solution_steps>\nReverse the cell order of every row.\n</solution_steps>\n\n"
    "<key_insight>\nOnly the horizontal order changes.\n</key_insight>\n\n"
    "<puzzle_concepts>\nreversal\n</puzzle_concepts>"
)


def _write_nvarc_fixture(path, *, puzzles: int = 8, pairs: int = 6) -> str:
    rows = []
    for index in range(puzzles):
        size = 2 + index % 3
        pair_list = [
            {
                "input": [[(index + offset) % 10] * size] * size,
                "output": [[(index + offset + 1) % 10]],
            }
            for offset in range(pairs)
        ]
        rows.append(
            {
                "puzzle_id": f"train_{index:04d}",
                "split": "train",
                "canonical_rule": CANONICAL_RULE,
                "pairs_json": json.dumps(pair_list),
                "difficulty": size * size,
            }
        )
    pq.write_table(pa.Table.from_pylist(rows), str(path / "data-00000.parquet"))
    return str(path)


def test_executor_example_targets_the_gold_grid_in_one_answer_block() -> None:
    puzzle = {
        "puzzle_id": "p",
        "canonical_rule": CANONICAL_RULE,
        "pairs_json": "[]",
    }
    pair = {"input": [[1, 0]], "output": [[0, 1]]}
    example = executor_example(puzzle, pair, template="executor task:\n{}\n")
    assert [m["role"] for m in example["messages"]] == ["user", "assistant"]
    prompt = example["messages"][0]["content"]
    assert "<transformation>" in prompt and "1 0" in prompt
    assert (
        example["messages"][1]["content"]
        == f"{NO_THINK_PREFIX}<answer>\n0 1\n</answer>"
    )


def test_proposer_example_reuses_the_gym_prompt_and_canonical_target() -> None:
    import random

    pairs = [{"input": [[index, 0]], "output": [[0, index]]} for index in range(6)]
    puzzle = {
        "puzzle_id": "p",
        "canonical_rule": CANONICAL_RULE,
        "pairs_json": json.dumps(pairs),
    }
    example = proposer_example(
        puzzle, demo_pairs=3, eval_pairs=2, byte_budget=8000, rng=random.Random(0)
    )
    assert example is not None
    assert [m["role"] for m in example["messages"]] == ["system", "user", "assistant"]
    assert example["messages"][0]["content"] == PROPOSER_INSTRUCTIONS
    prompt = example["messages"][1]["content"]
    assert "Training example demo_0" in prompt
    assert "<rules_summary>" in prompt  # the 4-section request scaffold
    target = example["messages"][2]["content"]
    assert target == NO_THINK_PREFIX + CANONICAL_RULE
    assert "<think" not in target[len(NO_THINK_PREFIX) :]


def test_proposer_example_rejects_unparseable_reference_rules() -> None:
    import random

    pairs = [{"input": [[index, 0]], "output": [[0, index]]} for index in range(6)]
    puzzle = {
        "puzzle_id": "p",
        "canonical_rule": "<rules_summary>only one section</rules_summary>",
        "pairs_json": json.dumps(pairs),
    }
    assert (
        proposer_example(
            puzzle, demo_pairs=3, eval_pairs=2, byte_budget=8000, rng=random.Random(0)
        )
        is None
    )


@pytest.mark.skipif(
    importlib.util.find_spec("nemo_gym") is None,
    reason="nemo_gym extra not installed",
)
def test_proposer_instructions_match_the_gym_agent() -> None:
    from responses_api_agents.arc_transform_refinement_agent.app import (
        PROPOSER_INSTRUCTIONS as AGENT_INSTRUCTIONS,
    )

    assert PROPOSER_INSTRUCTIONS == AGENT_INSTRUCTIONS


def test_main_proposer_only_dataset_keeps_val_role_mix(tmp_path, monkeypatch) -> None:
    """--executor-examples 0 yields a proposer-only prior, val included."""
    data_dir = tmp_path / "nvarc"
    out_dir = tmp_path / "out"
    data_dir.mkdir()
    _write_nvarc_fixture(data_dir)
    template_file = tmp_path / "executor.txt"
    template_file.write_text("executor task:\n{}\nreturn an <answer> block\n")
    monkeypatch.setattr(
        "sys.argv",
        [
            "nvarc_sft_materialize.py",
            "--data-dir",
            str(data_dir),
            "--output-dir",
            str(out_dir),
            "--executor-examples",
            "0",
            "--proposer-examples",
            "6",
            "--val-puzzles",
            "2",
            "--val-examples",
            "4",
            "--executor-prompt-file",
            str(template_file),
        ],
    )
    main()
    train = [
        json.loads(line) for line in (out_dir / "train.jsonl").read_text().splitlines()
    ]
    val = [
        json.loads(line) for line in (out_dir / "val.jsonl").read_text().splitlines()
    ]
    assert {row["sft_role"] for row in train + val} == {"proposer"}
    assert len(train) == 6 and len(val) == 4


def test_main_splits_by_puzzle_id_and_balances_roles(tmp_path, monkeypatch) -> None:
    data_dir = tmp_path / "nvarc"
    out_dir = tmp_path / "out"
    data_dir.mkdir()
    _write_nvarc_fixture(data_dir)
    template_file = tmp_path / "executor.txt"
    template_file.write_text("executor task:\n{}\nreturn an <answer> block\n")
    monkeypatch.setattr(
        "sys.argv",
        [
            "nvarc_sft_materialize.py",
            "--data-dir",
            str(data_dir),
            "--output-dir",
            str(out_dir),
            "--executor-examples",
            "6",
            "--proposer-examples",
            "6",
            "--val-puzzles",
            "2",
            "--val-examples",
            "4",
            "--executor-prompt-file",
            str(template_file),
        ],
    )
    main()
    train = [
        json.loads(line) for line in (out_dir / "train.jsonl").read_text().splitlines()
    ]
    val = [
        json.loads(line) for line in (out_dir / "val.jsonl").read_text().splitlines()
    ]
    stats = json.loads((out_dir / "stats.json").read_text())

    assert len(train) == 12 and len(val) == 4
    assert stats["train_executor_rows"] == 6 and stats["train_proposer_rows"] == 6
    # Split by puzzle id: no puzzle appears on both sides.
    assert not {row["task_id"] for row in train} & {row["task_id"] for row in val}
    for row in train + val:
        assert row["messages"][-1]["role"] == "assistant"
        target = row["messages"][-1]["content"]
        # Empty-think prefix continuing the generation opener, nothing more.
        assert target.startswith(NO_THINK_PREFIX)
        assert "<think" not in target[len(NO_THINK_PREFIX) :]
        if row["sft_role"] == "executor":
            assert target[len(NO_THINK_PREFIX) :].startswith("<answer>\n")
        else:
            assert row["messages"][0]["content"] == PROPOSER_INSTRUCTIONS
