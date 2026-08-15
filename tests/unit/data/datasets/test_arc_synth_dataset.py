import pytest
from datasets import concatenate_datasets

from nemo_rl.data.datasets.response_datasets.arc_synth import (
    TASK_NAME,
    ArcSynthDataset,
)
from nemo_rl.environments.arc_agi_grid import REAL_ARC_LEVEL

COLUMNS = {"task_name", "task_id", "train_pairs", "test_input", "target", "level"}


@pytest.fixture(scope="module")
def dataset():
    return ArcSynthDataset(
        levels=[0, 1, 2], num_tasks=24, num_val_tasks=12, seed=0, val_seed=99
    )


def test_row_schema(dataset):
    row = dataset.dataset[0]
    assert set(row) == COLUMNS
    assert row["task_name"] == TASK_NAME
    assert row["train_pairs"][0].keys() == {"input", "output"}


def test_levels_are_mixed_not_ramped(dataset):
    # Every batch has to hold every level: GRPO's advantage is computed within a
    # group of rollouts on one prompt, so a phase of uniformly hopeless or
    # uniformly trivial tasks contributes no gradient at all.
    assert sorted(set(dataset.dataset["level"])) == [0, 1, 2]
    assert dataset.dataset["level"][:6] == [0, 1, 2, 0, 1, 2]


def test_validation_split_is_held_out(dataset):
    train_ids = set(dataset.dataset["task_id"])
    assert not train_ids & set(dataset.val_dataset["task_id"])
    assert len(dataset.val_dataset) == 12


def test_an_unknown_level_is_rejected():
    with pytest.raises(ValueError, match="unknown levels"):
        ArcSynthDataset(levels=[0, 9], num_tasks=4, num_val_tasks=4, seed=0, val_seed=1)


def test_a_shared_seed_is_rejected():
    # Otherwise the "held-out" split is the training tasks verbatim and
    # validation silently reports a memorization score.
    with pytest.raises(ValueError, match="memorization"):
        ArcSynthDataset(levels=[0], num_tasks=4, num_val_tasks=4, seed=7, val_seed=7)


def test_reproducible_from_the_seed():
    kwargs = dict(levels=[1], num_tasks=8, num_val_tasks=4, seed=5, val_seed=6)
    first = ArcSynthDataset(**kwargs)
    second = ArcSynthDataset(**kwargs)
    assert first.dataset["target"] == second.dataset["target"]


def test_concatenates_with_the_real_arc_schema(dataset):
    # The two validation sources are merged into one dataloader, so their
    # features have to match exactly -- that is why real ARC rows carry a level.
    real = dataset.dataset.map(
        lambda row: {"task_name": "arc_agi", "level": REAL_ARC_LEVEL}
    )
    merged = concatenate_datasets([dataset.val_dataset, real])
    assert len(merged) == len(dataset.val_dataset) + len(real)
    assert set(merged.column_names) == COLUMNS
