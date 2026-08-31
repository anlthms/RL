from tools.arc_demo_loo import build_loo_tasks

TASK = {
    "train": [
        {"input": [[1]], "output": [[2]]},
        {"input": [[3]], "output": [[4]]},
        {"input": [[5]], "output": [[6]]},
    ],
    "test": [{"input": [[7]]}],
}


def test_each_demo_is_held_out_exactly_once_with_the_rest_as_train() -> None:
    challenges, solutions, manifest = build_loo_tasks({"abc": TASK})
    assert sorted(challenges) == ["abc_loo00", "abc_loo01", "abc_loo02"]
    row = challenges["abc_loo01"]
    assert row["train"] == [TASK["train"][0], TASK["train"][2]]
    assert row["test"] == [{"input": [[3]]}]
    assert solutions["abc_loo01"] == [[[4]]]
    assert manifest["rows"]["abc_loo01"] == {
        "base_task_id": "abc",
        "held_out_demo_index": 1,
        "num_demos": 3,
    }


def test_source_test_grids_never_reach_the_loo_split() -> None:
    challenges, solutions, _ = build_loo_tasks({"abc": TASK})
    dumped = repr((challenges, solutions))
    assert "[[7]]" not in dumped


def test_single_demo_tasks_are_skipped_and_recorded() -> None:
    single = {"train": [{"input": [[1]], "output": [[2]]}], "test": []}
    challenges, solutions, manifest = build_loo_tasks({"solo": single})
    assert challenges == {} and solutions == {}
    assert manifest["skipped_task_ids"] == ["solo"]
