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

from datasets import Dataset
from pydantic import BaseModel, model_validator

from nemo_rl.data.datasets.raw_dataset import RawDataset
from nemo_rl.environments.arc_agi_generators import (
    DEFAULT_MAX_INPUT_DIM,
    LEVELS,
    SynthTask,
    generate_tasks,
)

TASK_NAME = "arc_synth"

# Difficulty axes that were tried and removed for being inert or contradictory
# on most levels. They are named rather than ignored: `extra="allow"` is needed so the merged
# `data.default` keys pass through, and it would otherwise swallow these
# silently, leaving a recipe that reads as configured and does nothing.
_REMOVED_KEYS = {
    "size_ramp_window": "use difficulty_ramp_window",
    "size_ramp_steps": "use difficulty_ramp_steps",
    "palette_size": "removed; the ladder picks a palette per level",
    "object_count": "removed; needs a definition of 'object' the ladder lacks",
    "density": "removed; it meant two opposite things across samplers",
    "composition_depth": "removed; level 5 is depth 2",
    "distractor_count": "removed; was inert on four of six levels",
}


class SynthCurriculumConfig(BaseModel, extra="allow"):
    """Everything needed to regenerate a synthetic split, and nothing else.

    This is the whole reproducibility boundary: two `SynthCurriculumConfig`s
    that compare equal generate identical tasks, so an offline re-scorer can
    rebuild a run's validation rows from its recipe alone. Adding a curriculum
    axis means adding a field here and nowhere else.

    Attributes:
        levels: Ladder levels to draw from. A repeated entry weights that level.
        num_tasks: Training tasks to materialize. Deliberately far larger than
            any run, so no task is seen twice.
        num_val_tasks: Held-out tasks per validation pass.
        seed: Generator seed for the training split.
        val_seed: Generator seed for the held-out split. Must differ from
            ``seed``, or validation measures memorization.
        num_train_pairs: Few-shot pairs per task; a list ordered easy -> hard
            (more demonstrations are easier), or None to draw 2-4 per task the
            way real ARC tasks vary.
        max_input_dim: Upper bound on an input grid's side, or a list of bounds
            to mix sizes within every batch.
        level_difficulty_order: Levels ordered by *measured* solve rate, easiest
            first. Crossed with ``max_input_dim`` to form the joint schedule.
        difficulty_ramp_window: Prompts per optimizer step. Every full window
            holds every level/size combination while the weights move easy ->
            hard. Requires ``data.shuffle: false``; omitted for validation, so
            the held-out mixture is fixed across checkpoints.
        difficulty_ramp_steps: Optimizer steps the schedule spans. Must be the
            run length, not the dataset length.
    """

    levels: list[int]
    num_tasks: int
    num_val_tasks: int
    seed: int
    val_seed: int
    num_train_pairs: int | list[int] | None = None
    max_input_dim: int | list[int] = DEFAULT_MAX_INPUT_DIM
    level_difficulty_order: list[int] | None = None
    difficulty_ramp_window: int | None = None
    difficulty_ramp_steps: int | None = None

    @model_validator(mode="after")
    def _check(self) -> "SynthCurriculumConfig":
        present = sorted(_REMOVED_KEYS.keys() & (self.model_extra or {}).keys())
        if present:
            detail = "; ".join(f"{key}: {_REMOVED_KEYS[key]}" for key in present)
            raise ValueError(
                f"removed curriculum keys are still set in the config ({detail})"
            )
        if not self.levels:
            raise ValueError("levels must not be empty")
        unknown = sorted(set(self.levels) - set(LEVELS))
        if unknown:
            raise ValueError(f"unknown levels {unknown}; the ladder has {list(LEVELS)}")
        if self.seed == self.val_seed:
            raise ValueError(
                f"seed and val_seed are both {self.seed}: the held-out split "
                "would be the training tasks verbatim, and validation would "
                "measure memorization rather than generalization"
            )
        dims = self.dims
        if not dims:
            raise ValueError("max_input_dim must not be an empty list")
        combinations = len(set(self.levels)) * len(set(dims))
        if self.difficulty_ramp_window is not None:
            if self.difficulty_ramp_window < combinations:
                raise ValueError(
                    f"difficulty_ramp_window {self.difficulty_ramp_window} cannot "
                    f"hold all {combinations} level/size combinations; every "
                    "optimizer window must contain the full range"
                )
            if self.difficulty_ramp_steps is None:
                raise ValueError(
                    "difficulty_ramp_window is set without difficulty_ramp_steps, "
                    "so the schedule would span the dataset rather than the run "
                    "and barely move"
                )
            # The ramp has to fit in the rows it will actually read. Both knobs
            # interpolate from grpo.*, so a recipe that does not pin
            # max_num_steps inherits the base default of 1_000_000 and spends
            # its whole run in the first fraction of a percent of the schedule:
            # an inert ramp that still reads as configured, which is the exact
            # failure difficulty_ramp_steps was introduced to prevent.
            needed = self.difficulty_ramp_steps * self.difficulty_ramp_window
            if needed > self.num_tasks:
                raise ValueError(
                    f"difficulty_ramp_steps {self.difficulty_ramp_steps} x window "
                    f"{self.difficulty_ramp_window} needs {needed} rows but "
                    f"num_tasks is {self.num_tasks}, so the ramp can never "
                    "complete; pin grpo.max_num_steps in the recipe to the run "
                    "length it was sized for"
                )
        return self

    @property
    def dims(self) -> list[int]:
        """``max_input_dim`` as a list, however it was written."""
        if isinstance(self.max_input_dim, int):
            return [self.max_input_dim]
        return list(self.max_input_dim)

    def train_tasks(self) -> list[SynthTask]:
        """The training split, on the ramped schedule."""
        return generate_tasks(
            self.seed,
            self.num_tasks,
            self.levels,
            num_train_pairs=self.num_train_pairs,
            max_input_dim=self.max_input_dim,
            level_difficulty_order=self.level_difficulty_order,
            difficulty_ramp_window=self.difficulty_ramp_window,
            difficulty_ramp_steps=self.difficulty_ramp_steps,
        )

    def val_tasks(self) -> list[SynthTask]:
        """The held-out split, at fixed weights.

        No ramp: the mixture must not move between checkpoints or the metric is
        not comparable across them.
        """
        return generate_tasks(
            self.val_seed,
            self.num_val_tasks,
            self.levels,
            num_train_pairs=self.num_train_pairs,
            max_input_dim=self.max_input_dim,
            level_difficulty_order=self.level_difficulty_order,
        )


def _to_row(task: SynthTask) -> dict:
    """Render a generated task as an ARC dataset row.

    The schema matches ``arc_agi.py`` exactly apart from ``level``, which real
    ARC rows carry as ``REAL_ARC_LEVEL``. Identical schemas are what let the two
    be concatenated into one validation set, and one data processor serve both.
    """
    return {
        "task_name": TASK_NAME,
        "task_id": task.task_id,
        "train_pairs": task.train_pairs,
        "test_input": task.test_input,
        "target": task.target,
        "level": task.level,
    }


class ArcSynthDataset(RawDataset):
    """Synthetic ARC-style tasks from the difficulty ladder in ``arc_agi_generators``.

    Generated rather than downloaded: volume is free, and a run stays
    reproducible because every task is a pure function of its
    ``SynthCurriculumConfig`` and its row index.

    Levels and grid sizes are **mixed within every batch**. Their joint weights
    ramp across steps, but every full window retains the complete cross product.
    GRPO's advantage is computed within a group of rollouts on one prompt, so a
    phase in which every task is uniformly hopeless -- or uniformly trivial --
    contributes no exact-match signal. Keeping the range present avoids
    reintroducing that failure while still moving the curriculum frontier.

    Every field that shapes the curriculum lives on ``SynthCurriculumConfig``
    and comes from ``env_arc_synth.yaml``: a silent fallback to some other
    mixture or seed would be a different experiment reported under the same
    name, so removed keys raise rather than being ignored.
    """

    def __init__(self, **kwargs) -> None:
        config = SynthCurriculumConfig(**kwargs)
        self.task_name = TASK_NAME
        self.curriculum = config
        self.dataset = Dataset.from_list(
            [_to_row(task) for task in config.train_tasks()]
        )
        self.val_dataset = Dataset.from_list(
            [_to_row(task) for task in config.val_tasks()]
        )
