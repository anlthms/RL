import json

from tools.arc_evaldemo_ingest import build_rows

TASK = {
    "train": [
        {"input": [[1, 1]], "output": [[2, 2], [2, 2]]},
        {"input": [[3]], "output": [[4]]},
    ],
    "test": [{"input": [[9, 9, 9]]}],
}


def test_rows_hold_only_demo_pairs_with_ingest_difficulty() -> None:
    (row,) = build_rows({"abc": TASK}, "train")
    assert row["puzzle_id"] == "abc"
    assert row["canonical_rule"] == ""
    assert row["split"] == "train"
    assert json.loads(row["pairs_json"]) == TASK["train"]
    assert row["difficulty"] == 4  # the 2x2 output, not the 1x3 test grid


def test_real_test_grids_never_reach_the_shard() -> None:
    (row,) = build_rows({"abc": TASK}, "train")
    assert "[[9, 9, 9]]" not in row["pairs_json"]
    assert "9, 9, 9" not in json.dumps(row)
