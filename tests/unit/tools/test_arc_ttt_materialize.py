import json
import random

from tools.arc_ttt_materialize import build_task_rows, schedule

TASK = {
    "train": [
        {"input": [[1]], "output": [[2]]},
        {"input": [[3]], "output": [[4]]},
    ],
    "test": [{"input": [[9, 9]]}],
}


def test_rows_verify_the_rule_on_every_demo_and_never_read_the_test() -> None:
    rows = build_task_rows({"abc": TASK})
    row = rows["abc"]
    assert row["train"] == TASK["train"]
    assert row["test"] == TASK["train"]  # all demos, verified server-side
    assert row["role"] == "proposer"
    assert "9, 9" not in json.dumps(row)


def test_schedule_cycles_full_shuffled_passes() -> None:
    order = schedule(["a", "b", "c"], slots=8, rng=random.Random(7))
    assert len(order) == 8
    # Every complete 3-slot pass covers all tasks exactly once.
    assert sorted(order[0:3]) == ["a", "b", "c"]
    assert sorted(order[3:6]) == ["a", "b", "c"]
