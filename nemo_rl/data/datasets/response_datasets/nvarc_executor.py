# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Single-grid executor rows sourced from the ingested NVARC dataset.

Rows carry ``task_name``, ``task_id``, ``transform_description``,
``test_input``, ``target``, and ``bucket`` for the
``arc_executor_data_processor`` and ``ArcAgiEnvironment``. The rule text is
the canonical 4-section NVARC description produced by
``tools/nvarc_ingest.py``; ``bucket`` is a 1-indexed max-grid-area bucket (the
proposal's initial difficulty proxy), reported as ``bucket_<n>`` metrics.

The train/executor-val/proposer-eval pools were split by *puzzle id* at
ingestion time; this module never crosses them. Training rows sample
input/output pairs per puzzle without replacement (reshuffling only after a
full pass), so a puzzle's ~30 pairs are spread over the run instead of
repeating one.

The training curriculum is *staged*: each difficulty bucket is held for
``curriculum_hold_steps`` steps before advancing to the next, starting from
the easiest. Pure staging (the default) fills every window with the current
bucket only — maximum on-level signal, accepted on the hypothesis that the
opening ventile is solvable by the base model. ``curriculum_cumulative``
switches to cycling over the current and all earlier buckets, the fallback if
early GRPO groups degenerate (every rollout in a group equally hopeless).
"""

import json
import random
from pathlib import Path
from typing import TypeVar

from datasets import Dataset
from pydantic import BaseModel, model_validator

from nemo_rl.data.datasets.raw_dataset import RawDataset

ChoiceT = TypeVar("ChoiceT")

TASK_NAME = "nvarc_executor"

_ROW_COLUMNS = ["puzzle_id", "canonical_rule", "pairs_json", "difficulty"]


class NvArcExecutorConfig(BaseModel, extra="allow"):
    """Configuration for regenerating NVARC executor training and validation.

    Attributes:
        data_dir: Directory of ``tools/nvarc_ingest.py`` output parquet.
        num_tasks: Training rows to emit (must cover the run's step count
            times ``curriculum_window``; the preflight checks this).
        num_val_tasks: Held-out rows drawn from the ``executor_val`` pool;
            0 disables the NVARC-side validation split entirely (used when
            validation runs solely on the real ARC evaluation set).
        seed: Ordering/pair-sampling seed for the training split.
        val_seed: Pair-sampling seed for validation; must differ from ``seed``.
        bucket_edges: Strictly increasing inclusive upper edges on puzzle
            difficulty (max grid area over the puzzle). Areas above the last
            edge form the final bucket, so ``len(bucket_edges) + 1`` buckets
            exist; bucket ids are 1-indexed and reported as ``bucket_<n>``
            validation metrics. The defaults are the ingested v1 train-split
            ventiles: max-over-pairs area saturates near 900 for half the
            corpus, so equal-mass edges beat round numbers, and the opening
            ventile (area <= 91, 2,299 puzzles) is small enough for a
            non-finetuned executor. The p95 edge ties the max area (900), so
            18 edges make 19 buckets; counts are uneven at the top where
            edges tie (756, 783/784, 810/812, 840/841), which is fine.
        curriculum_window: Rows per training step (the trainer's
            ``num_prompts_per_step``). Omit for a fixed all-bucket cycle.
        curriculum_hold_steps: Steps each bucket is held before the schedule
            advances to the next; windows past the final bucket stay on it.
            Required with ``curriculum_window``.
        curriculum_cumulative: Cycle over the current and all earlier buckets
            instead of filling windows with the current bucket only. The
            fallback if pure staging degenerates early GRPO groups.
    """

    data_dir: str
    num_tasks: int
    num_val_tasks: int
    seed: int
    val_seed: int
    bucket_edges: list[int] = [
        91, 162, 264, 380, 525, 621, 667, 700, 725, 750,
        756, 783, 784, 810, 812, 840, 841, 870,
    ]  # fmt: skip
    curriculum_window: int | None = None
    curriculum_hold_steps: int | None = None
    curriculum_cumulative: bool = False

    @model_validator(mode="after")
    def _check(self) -> "NvArcExecutorConfig":
        if self.seed == self.val_seed:
            raise ValueError("seed and val_seed must differ")
        if not self.bucket_edges or sorted(set(self.bucket_edges)) != list(
            self.bucket_edges
        ):
            raise ValueError("bucket_edges must be non-empty and strictly increasing")
        if (self.curriculum_window is None) != (self.curriculum_hold_steps is None):
            raise ValueError(
                "curriculum_window and curriculum_hold_steps must be set together"
            )
        if self.curriculum_window is None or self.curriculum_hold_steps is None:
            return self
        if self.curriculum_window < 1 or self.curriculum_hold_steps < 1:
            raise ValueError(
                "curriculum_window and curriculum_hold_steps must be positive"
            )
        return self

    def bucket_id(self, difficulty: int) -> int:
        """Map a puzzle difficulty (max grid area) to its 1-indexed bucket."""
        for index, edge in enumerate(self.bucket_edges):
            if difficulty <= edge:
                return index + 1
        return len(self.bucket_edges) + 1


class _PairSampler:
    """Sample a puzzle's pairs without replacement, reshuffling per pass."""

    def __init__(self, pairs: list[dict], rng: random.Random) -> None:
        self._pairs = pairs
        self._rng = rng
        self._queue: list[int] = []

    def next(self) -> dict:
        if not self._queue:
            self._queue = list(range(len(self._pairs)))
            self._rng.shuffle(self._queue)
        return self._pairs[self._queue.pop()]


def staged_choices(
    *,
    count: int,
    choices: list[ChoiceT],
    window: int,
    hold_steps: int,
    cumulative: bool = False,
) -> list[ChoiceT]:
    """Stage ordered choices easy-to-hard, holding each for ``hold_steps`` windows.

    Window ``w`` sits at stage ``min(w // hold_steps, len(choices) - 1)``;
    windows past the final stage stay on it. Pure staging fills the window
    with the current choice only. Cumulative staging cycles over the current
    and all earlier choices with a pointer that persists across windows, so
    every included choice recurs evenly even when the window is smaller than
    the pool.

    ``window`` is the number of prompts the trainer consumes per step.
    """
    if window < 1 or hold_steps < 1:
        raise ValueError("window and hold_steps must be positive")
    windows = max(1, -(-count // window))
    out: list[ChoiceT] = []
    pointer = 0
    for window_index in range(windows):
        stage = min(window_index // hold_steps, len(choices) - 1)
        current_size = min(window, count - window_index * window)
        if cumulative:
            pool = choices[: stage + 1]
            out.extend(pool[(pointer + i) % len(pool)] for i in range(current_size))
            pointer = (pointer + current_size) % len(pool)
        else:
            out.extend([choices[stage]] * current_size)
    return out


def _load_split(data_dir: str, split: str) -> list[dict]:
    """Read one split's rows (rule text + pairs) from the ingested parquet."""
    # Deferred import: pyarrow.dataset is only needed when this dataset is
    # actually selected, and the module is imported by the registry walk.
    import pyarrow.dataset as ds

    # Select the data shards by name: the ingest directory also holds
    # stats.json, which a whole-directory dataset would try to parse.
    shards = sorted(Path(data_dir).glob("data-*.parquet"))
    if not shards:
        raise FileNotFoundError(f"no data-*.parquet shards under {data_dir}")
    dataset = ds.dataset([str(shard) for shard in shards], format="parquet")
    table = dataset.to_table(columns=_ROW_COLUMNS, filter=ds.field("split") == split)
    rows = table.to_pylist()
    if not rows:
        raise ValueError(f"no rows with split={split!r} under {data_dir}")
    # Ingestion order is enumeration order; sort so schedules are reproducible
    # regardless of shard layout.
    rows.sort(key=lambda row: row["puzzle_id"])
    return rows


def _to_row(puzzle: dict, pair: dict, *, bucket: int) -> dict:
    return {
        "task_name": TASK_NAME,
        "task_id": puzzle["puzzle_id"],
        "transform_description": puzzle["canonical_rule"],
        "test_input": pair["input"],
        "target": pair["output"],
        "bucket": bucket,
        "difficulty": puzzle["difficulty"],
    }


class NvArcExecutorDataset(RawDataset):
    """Single-grid NVARC-description execution samples for GRPO."""

    def __init__(self, seed: int, **kwargs) -> None:
        config = NvArcExecutorConfig(seed=seed, **kwargs)
        self.task_name = TASK_NAME
        self.curriculum = config
        self.dataset = Dataset.from_list(self._split_rows(config, train=True))
        # num_val_tasks == 0 opts out of NVARC-side validation, for recipes
        # that validate solely on the real ARC evaluation split.
        self.val_dataset = (
            Dataset.from_list(self._split_rows(config, train=False))
            if config.num_val_tasks
            else None
        )

    @staticmethod
    def _split_rows(config: NvArcExecutorConfig, *, train: bool) -> list[dict]:
        rows = _load_split(config.data_dir, "train" if train else "executor_val")
        count = config.num_tasks if train else config.num_val_tasks
        seed = config.seed if train else config.val_seed
        rng = random.Random(seed)

        by_bucket: dict[int, list[dict]] = {}
        for row in rows:
            by_bucket.setdefault(config.bucket_id(row["difficulty"]), []).append(row)
        buckets = sorted(by_bucket)

        if train and config.curriculum_window:
            assert config.curriculum_hold_steps is not None  # validator-paired
            schedule = staged_choices(
                count=count,
                choices=buckets,
                window=config.curriculum_window,
                hold_steps=config.curriculum_hold_steps,
                cumulative=config.curriculum_cumulative,
            )
        else:
            # Fixed cycle: the validation mixture must not move between
            # checkpoints, and an un-ramped train split should not either.
            schedule = [buckets[index % len(buckets)] for index in range(count)]

        queues: dict[int, list[dict]] = {}
        samplers: dict[str, _PairSampler] = {}
        out: list[dict] = []
        for bucket in schedule:
            if not queues.get(bucket):
                shuffled = list(by_bucket[bucket])
                rng.shuffle(shuffled)
                queues[bucket] = shuffled
            puzzle = queues[bucket].pop()
            if puzzle["puzzle_id"] not in samplers:
                samplers[puzzle["puzzle_id"]] = _PairSampler(
                    json.loads(puzzle["pairs_json"]), rng
                )
            pair = samplers[puzzle["puzzle_id"]].next()
            out.append(_to_row(puzzle, pair, bucket=bucket))
        return out
