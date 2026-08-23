import json

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from nemo_rl.data.datasets.response_datasets.nvarc_executor import (
    TASK_NAME,
    NvArcExecutorConfig,
    NvArcExecutorDataset,
)


def _write_fixture(path, *, train_puzzles: int = 12, val_puzzles: int = 6) -> str:
    """Write a miniature nvarc_ingest-style parquet directory."""
    rows = []
    specs = [("train", train_puzzles), ("executor_val", val_puzzles)]
    # A puzzle the dataset must never surface.
    specs += [("proposer_eval", 2), ("excluded", 2)]
    index = 0
    for split, count in specs:
        for _ in range(count):
            size = 2 + index % 4  # difficulties 4, 9, 16, 25
            # Inputs must differ across a puzzle's pairs: the no-repeat test
            # identifies a drawn pair by its input grid.
            pairs = [
                {
                    "input": [[(index + offset) % 10] * size] * size,
                    "output": [[(index + offset) % 10]],
                }
                for offset in range(3)
            ]
            rows.append(
                {
                    "puzzle_id": f"{split}_{index:04d}",
                    "split": split,
                    "canonical_rule": f"<rules_summary>\nrule {index}\n</rules_summary>",
                    "pairs_json": json.dumps(pairs),
                    "difficulty": size * size,
                }
            )
            index += 1
    pq.write_table(pa.Table.from_pylist(rows), str(path / "data-00000.parquet"))
    return str(path)


def _dataset(tmp_path, **overrides) -> NvArcExecutorDataset:
    config = {
        "data_dir": _write_fixture(tmp_path),
        "num_tasks": 24,
        "num_val_tasks": 6,
        "seed": 21,
        "val_seed": 22,
        "bucket_edges": [4, 9, 16],
    }
    config.update(overrides)
    return NvArcExecutorDataset(**config)


def test_rows_carry_the_arc_executor_schema(tmp_path) -> None:
    dataset = _dataset(tmp_path)
    row = dataset.dataset[0]
    assert row["task_name"] == TASK_NAME
    assert set(row) == {
        "task_name",
        "task_id",
        "transform_description",
        "test_input",
        "target",
        "level",
        "difficulty",
    }
    assert row["transform_description"].startswith("<rules_summary>")
    assert 1 <= row["level"] <= 4


def test_splits_never_cross_pools(tmp_path) -> None:
    dataset = _dataset(tmp_path)
    assert all(task_id.startswith("train_") for task_id in dataset.dataset["task_id"])
    assert all(
        task_id.startswith("executor_val_")
        for task_id in dataset.val_dataset["task_id"]
    )


def test_rows_are_reproducible(tmp_path) -> None:
    first = _dataset(tmp_path)
    second = _dataset(tmp_path)
    assert first.dataset["task_id"] == second.dataset["task_id"]
    assert first.dataset["test_input"] == second.dataset["test_input"]
    assert first.val_dataset["target"] == second.val_dataset["target"]


def test_pairs_do_not_repeat_before_a_full_pass(tmp_path) -> None:
    dataset = _dataset(tmp_path)
    seen: dict[str, list] = {}
    for row in dataset.dataset:
        key = json.dumps(row["test_input"])
        seen.setdefault(row["task_id"], []).append(key)
    for task_id, inputs in seen.items():
        # Each fixture puzzle has 3 pairs; within any window of 3 draws for a
        # puzzle there must be no duplicate input.
        first_pass = inputs[:3]
        assert len(set(first_pass)) == len(first_pass), task_id


def test_ramp_orders_buckets_easy_to_hard(tmp_path) -> None:
    dataset = _dataset(
        tmp_path,
        num_tasks=24,
        difficulty_ramp_window=8,
        difficulty_ramp_steps=3,
    )
    levels = dataset.dataset["level"]
    windows = [levels[index : index + 8] for index in range(0, 24, 8)]
    # Every window keeps every bucket.
    for window in windows:
        assert set(window) == {1, 2, 3, 4}
    # The weighting moves from the easiest bucket toward the hardest.
    assert windows[0].count(1) > windows[-1].count(1)
    assert windows[-1].count(4) > windows[0].count(4)


def test_rejects_shared_seed(tmp_path) -> None:
    with pytest.raises(ValueError, match="must differ"):
        _dataset(tmp_path, val_seed=21)


def test_rejects_bad_bucket_edges() -> None:
    with pytest.raises(ValueError, match="strictly increasing"):
        NvArcExecutorConfig(
            data_dir="x",
            num_tasks=1,
            num_val_tasks=1,
            seed=1,
            val_seed=2,
            bucket_edges=[9, 4],
        )


def test_rejects_undersized_ramp() -> None:
    with pytest.raises(ValueError, match="cannot hold all"):
        NvArcExecutorConfig(
            data_dir="x",
            num_tasks=100,
            num_val_tasks=1,
            seed=1,
            val_seed=2,
            bucket_edges=[4, 9, 16],
            difficulty_ramp_window=2,
            difficulty_ramp_steps=10,
        )
    with pytest.raises(ValueError, match="needs 80 rows"):
        NvArcExecutorConfig(
            data_dir="x",
            num_tasks=64,
            num_val_tasks=1,
            seed=1,
            val_seed=2,
            bucket_edges=[4, 9, 16],
            difficulty_ramp_window=8,
            difficulty_ramp_steps=10,
        )


def test_bucket_id_maps_edges_inclusively() -> None:
    config = NvArcExecutorConfig(
        data_dir="x",
        num_tasks=1,
        num_val_tasks=1,
        seed=1,
        val_seed=2,
        bucket_edges=[36, 100],
    )
    assert config.bucket_id(1) == 1
    assert config.bucket_id(36) == 1
    assert config.bucket_id(37) == 2
    assert config.bucket_id(100) == 2
    assert config.bucket_id(900) == 3
