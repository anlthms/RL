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
        num_tasks: Training rows to emit (must cover the difficulty ramp).
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
            deciles: max-over-pairs area saturates near 900 for half the
            corpus, so equal-mass edges beat round numbers, and ten buckets
            give the ramp a genuinely easy opening decile (area <= 162).
        difficulty_ramp_window: Rows per training step (the trainer's
            ``num_prompts_per_step``); every window keeps every bucket so no
            GRPO group is uniformly hopeless. Omit for a fixed mixture.
        difficulty_ramp_steps: Steps the easy-to-hard reweighting spans; must
            be the run length, not the dataset length.
    """

    data_dir: str
    num_tasks: int
    num_val_tasks: int
    seed: int
    val_seed: int
    bucket_edges: list[int] = [162, 380, 621, 700, 750, 783, 810, 840, 870]
    difficulty_ramp_window: int | None = None
    difficulty_ramp_steps: int | None = None

    @model_validator(mode="after")
    def _check(self) -> "NvArcExecutorConfig":
        if self.seed == self.val_seed:
            raise ValueError("seed and val_seed must differ")
        if not self.bucket_edges or sorted(set(self.bucket_edges)) != list(
            self.bucket_edges
        ):
            raise ValueError("bucket_edges must be non-empty and strictly increasing")
        if self.difficulty_ramp_window is None:
            return self
        buckets = len(self.bucket_edges) + 1
        if self.difficulty_ramp_window < buckets:
            raise ValueError(
                f"difficulty_ramp_window {self.difficulty_ramp_window} cannot hold "
                f"all {buckets} difficulty buckets"
            )
        if self.difficulty_ramp_steps is None:
            raise ValueError("difficulty_ramp_window requires difficulty_ramp_steps")
        needed_rows = self.difficulty_ramp_steps * self.difficulty_ramp_window
        if needed_rows > self.num_tasks:
            raise ValueError(
                f"the difficulty ramp needs {needed_rows} rows but num_tasks is "
                f"{self.num_tasks}"
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


def ramped_choices(
    *,
    count: int,
    choices: list[ChoiceT],
    multipliers: list[float],
    window: int,
    ramp_windows: int | None = None,
) -> list[ChoiceT]:
    """Reweight ordered choices while retaining all of them in every window.

    Every window keeps at least one of every choice, so no batch is ever
    uniformly hopeless or uniformly trivial. What moves is the *weighting* of
    the remaining slots: a hump centred on the easiest choice at the start of
    the run and the hardest at the end, scaled by each choice's multiplier.

    ``window`` is the number of prompts the trainer consumes per step;
    ``ramp_windows`` is how many steps the schedule spans. Largest-remainder
    apportionment, so the counts sum to the window exactly.
    """
    if window < len(choices):
        raise ValueError(
            f"difficulty_ramp_window {window} is smaller than the "
            f"{len(choices)} difficulty buckets"
        )
    windows = max(1, -(-count // window))
    span = max(1, ramp_windows or windows)
    out: list[ChoiceT] = []
    for window_index in range(windows):
        progress = min(window_index, span - 1) / max(span - 1, 1)
        centre = progress * (len(choices) - 1)
        weights = [
            multiplier * 2.0 ** -abs(index - centre)
            for index, multiplier in enumerate(multipliers)
        ]
        total = sum(weights)
        current_size = min(window, count - window_index * window)
        spare = max(0, current_size - len(choices))
        exact = [spare * weight / total for weight in weights]
        counts = [1 + int(extra) for extra in exact]
        remainder = spare - sum(int(extra) for extra in exact)
        order = sorted(
            range(len(choices)),
            key=lambda index: exact[index] - int(exact[index]),
            reverse=True,
        )
        for index in order[:remainder]:
            counts[index] += 1
        row = [choice for choice, copies in zip(choices, counts) for _ in range(copies)]
        out.extend(row[:current_size])
    return out[:count]


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

        if train and config.difficulty_ramp_window:
            schedule = ramped_choices(
                count=count,
                choices=buckets,
                multipliers=[1.0] * len(buckets),
                window=config.difficulty_ramp_window,
                ramp_windows=config.difficulty_ramp_steps,
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
