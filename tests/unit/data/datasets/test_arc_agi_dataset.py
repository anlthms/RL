import json

from nemo_rl.data.datasets.response_datasets.arc_agi import ArcAgiDataset
from nemo_rl.environments.arc_agi_grid import REAL_ARC_BUCKET, bucket_metric_suffix


def _write_split(path, split: str, tasks: dict) -> None:
    challenges = {
        task_id: {
            "train": task["train"],
            "test": [{"input": t["input"]} for t in task["test"]],
        }
        for task_id, task in tasks.items()
    }
    solutions = {
        task_id: [t["output"] for t in task["test"]] for task_id, task in tasks.items()
    }
    (path / f"arc-agi_{split}_challenges.json").write_text(json.dumps(challenges))
    (path / f"arc-agi_{split}_solutions.json").write_text(json.dumps(solutions))


def test_each_test_pair_becomes_one_real_bucket_row(tmp_path) -> None:
    tasks = {
        "aaaa1111": {
            "train": [{"input": [[1]], "output": [[2]]}],
            "test": [
                {"input": [[3]], "output": [[4]]},
                {"input": [[5]], "output": [[6]]},
            ],
        },
        "bbbb2222": {
            "train": [{"input": [[7]], "output": [[8]]}],
            "test": [{"input": [[9]], "output": [[0]]}],
        },
    }
    _write_split(tmp_path, "training", tasks)
    _write_split(tmp_path, "evaluation", tasks)

    dataset = ArcAgiDataset(data_path=str(tmp_path), split="evaluation")
    rows = list(dataset.dataset)
    assert len(rows) == 3  # 2 + 1 test pairs
    first = rows[0]
    assert first["task_name"] == "arc_agi"
    assert first["bucket"] == REAL_ARC_BUCKET
    assert first["train_pairs"] == tasks["aaaa1111"]["train"]
    assert first["test_input"] == [[3]]
    assert first["target"] == [[4]]


def test_real_bucket_reports_under_the_real_suffix() -> None:
    assert bucket_metric_suffix(REAL_ARC_BUCKET) == "real"
    assert bucket_metric_suffix(3) == "bucket_3"
