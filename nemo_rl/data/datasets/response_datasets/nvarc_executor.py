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

Rows carry the same schema as ``arc_executor`` (``task_name``, ``task_id``,
``transform_description``, ``test_input``, ``target``, ``level``) so the
``arc_executor_data_processor`` and ``ArcAgiEnvironment`` run unchanged. The
rule text is the canonical 4-section NVARC description produced by
``tools/nvarc_ingest.py``; ``level`` is a 1-indexed max-grid-area bucket (the
proposal's initial difficulty proxy) rather than a generator ladder level, so
per-level validation metrics read as per-area-bucket metrics.

The train/executor-val/proposer-eval pools were split by *puzzle id* at
ingestion time; this module never crosses them. Training rows sample
input/output pairs per puzzle without replacement (reshuffling only after a
full pass), so a puzzle's ~30 pairs are spread over the run instead of
repeating one.
"""

import json
import random
from pathlib import Path

from datasets import Dataset
from pydantic import BaseModel, model_validator

from nemo_rl.data.datasets.raw_dataset import RawDataset
from nemo_rl.environments.arc_agi_generators import ramped_choices

TASK_NAME = "nvarc_executor"

_ROW_COLUMNS = ["puzzle_id", "canonical_rule", "pairs_json", "difficulty"]


class NvArcExecutorConfig(BaseModel, extra="allow"):
    """Configuration for regenerating NVARC executor training and validation.

    Attributes:
        data_dir: Directory of ``tools/nvarc_ingest.py`` output parquet.
        num_tasks: Training rows to emit (must cover the difficulty ramp).
        num_val_tasks: Held-out rows drawn from the ``executor_val`` pool.
        seed: Ordering/pair-sampling seed for the training split.
        val_seed: Pair-sampling seed for validation; must differ from ``seed``.
        bucket_edges: Strictly increasing inclusive upper edges on puzzle
            difficulty (max grid area over the puzzle). Areas above the last
            edge form the final bucket, so ``len(bucket_edges) + 1`` buckets
            exist; bucket ids are 1-indexed and reported as ``level_<n>``
            validation metrics. The defaults are the ingested v1 train-split
            quintiles: max-over-pairs area saturates near 900 for half the
            corpus, so equal-mass edges beat round numbers.
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
    bucket_edges: list[int] = [380, 700, 783, 840]
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
        "level": bucket,
        "difficulty": puzzle["difficulty"],
    }


class NvArcExecutorDataset(RawDataset):
    """Single-grid NVARC-description execution samples for GRPO."""

    def __init__(self, seed: int, **kwargs) -> None:
        config = NvArcExecutorConfig(seed=seed, **kwargs)
        self.task_name = TASK_NAME
        self.curriculum = config
        self.dataset = Dataset.from_list(self._split_rows(config, train=True))
        self.val_dataset = Dataset.from_list(self._split_rows(config, train=False))

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
