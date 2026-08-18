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


def test_joint_curriculum_axes_are_threaded_to_both_splits():
    configured = ArcSynthDataset(
        levels=[1, 2, 3, 4, 5],
        num_tasks=32,
        num_val_tasks=30,
        seed=15,
        val_seed=16,
        num_train_pairs=[4, 3, 2],
        max_input_dim=[6, 12, 20],
        level_difficulty_order=[0, 3, 2, 1, 4, 5],
        difficulty_ramp_window=32,
        difficulty_ramp_steps=1,
    )
    train_ids = configured.dataset["task_id"]
    val_ids = configured.val_dataset["task_id"]
    # Task IDs report what was realized, so the axes are checked on the rows
    # rather than on the config that asked for them.
    for marker in ("_f4_", "_f2_", "_d6_", "_d20_"):
        assert any(marker in task_id for task_id in train_ids)
        assert any(marker in task_id for task_id in val_ids)


@pytest.mark.parametrize(
    "removed",
    ["size_ramp_window", "palette_size", "object_count", "density", "distractor_count"],
)
def test_a_removed_curriculum_key_is_rejected_rather_than_ignored(removed):
    # `extra="allow"` is needed for the merged data.default keys, so without an
    # explicit check a recipe still setting one of these would read as
    # configured and do nothing.
    with pytest.raises(ValueError, match="removed curriculum keys"):
        ArcSynthDataset(
            levels=[1], num_tasks=4, num_val_tasks=4, seed=0, val_seed=1, **{removed: 2}
        )


def test_a_window_too_small_for_the_cross_product_is_rejected():
    # A window that cannot hold every combination is a window that can be
    # uniformly hopeless, which is the failure the mixture exists to prevent.
    with pytest.raises(ValueError, match="cannot hold all 6 level/size"):
        ArcSynthDataset(
            levels=[1, 2],
            num_tasks=8,
            num_val_tasks=4,
            seed=0,
            val_seed=1,
            max_input_dim=[6, 12, 20],
            difficulty_ramp_window=4,
            difficulty_ramp_steps=2,
        )


def test_a_ramp_window_without_a_span_is_rejected():
    # Defaulting the span to the dataset length gives an inert ramp that still
    # reads as configured -- the failure mode this check exists to prevent.
    with pytest.raises(ValueError, match="difficulty_ramp_steps"):
        ArcSynthDataset(
            levels=[1, 2],
            num_tasks=8,
            num_val_tasks=4,
            seed=0,
            val_seed=1,
            difficulty_ramp_window=8,
        )


def test_concatenates_with_the_real_arc_schema(dataset):
    # The two validation sources are merged into one dataloader, so their
    # features have to match exactly -- that is why real ARC rows carry a level.
    real = dataset.dataset.map(
        lambda row: {"task_name": "arc_agi", "level": REAL_ARC_LEVEL}
    )
    merged = concatenate_datasets([dataset.val_dataset, real])
    assert len(merged) == len(dataset.val_dataset) + len(real)
    assert set(merged.column_names) == COLUMNS


def test_a_ramp_that_cannot_complete_in_the_dataset_is_rejected():
    # Both ramp knobs interpolate from grpo.*, so a recipe that does not pin
    # max_num_steps inherits the base default of 1_000_000 and spends its whole
    # run in the first fraction of a percent of the schedule -- inert, but still
    # reading as configured.
    with pytest.raises(ValueError, match="ramp can never complete"):
        ArcSynthDataset(
            levels=[1, 2],
            num_tasks=8000,
            num_val_tasks=4,
            seed=0,
            val_seed=1,
            max_input_dim=[6, 12, 20],
            difficulty_ramp_window=128,
            difficulty_ramp_steps=1_000_000,
        )
