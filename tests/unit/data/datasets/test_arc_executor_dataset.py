import pytest

from nemo_rl.data.datasets.response_datasets.arc_executor import (
    TASK_NAME,
    ArcExecutorDataset,
)


def _dataset(**overrides) -> ArcExecutorDataset:
    config = {
        "levels": [1, 2, 3, 4, 5],
        "num_tasks": 30,
        "num_val_tasks": 15,
        "seed": 21,
        "val_seed": 22,
        "num_train_pairs": 3,
        "max_input_dim": 12,
        "paraphrase_ids": [0, 1, 2],
    }
    config.update(overrides)
    return ArcExecutorDataset(**config)


def test_executor_rows_contain_descriptions_without_targets_in_prompts() -> None:
    dataset = _dataset()
    row = dataset.dataset[0]

    assert row["task_name"] == TASK_NAME
    assert set(row) == {
        "task_name",
        "task_id",
        "transform_description",
        "description_paraphrase",
        "test_input",
        "target",
        "level",
    }
    assert sorted(set(dataset.dataset["description_paraphrase"])) == [0, 1, 2]


def test_executor_splits_are_reproducible_and_disjoint() -> None:
    first = _dataset()
    second = _dataset()

    assert first.dataset["target"] == second.dataset["target"]
    assert not set(first.dataset["task_id"]) & set(first.val_dataset["task_id"])


def test_executor_dataset_rejects_shared_seed() -> None:
    with pytest.raises(ValueError, match="must differ"):
        _dataset(val_seed=21)


def test_executor_dataset_rejects_unknown_paraphrases() -> None:
    with pytest.raises(ValueError, match="unknown paraphrase_ids"):
        _dataset(paraphrase_ids=[3])


def test_executor_dataset_validates_ramp_capacity() -> None:
    with pytest.raises(ValueError, match="cannot hold all"):
        _dataset(
            levels=[1, 2],
            max_input_dim=[6, 12, 20],
            difficulty_ramp_window=4,
            difficulty_ramp_steps=2,
        )

    with pytest.raises(ValueError, match="needs 80 rows"):
        _dataset(difficulty_ramp_window=8, difficulty_ramp_steps=10)
