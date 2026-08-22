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
    render_oracle_description,
)

TASK_NAME = "arc_executor"


class ExecutorCurriculumConfig(BaseModel, extra="allow"):
    """Configuration required to regenerate executor training and validation."""

    levels: list[int]
    num_tasks: int
    num_val_tasks: int
    seed: int
    val_seed: int
    paraphrase_ids: list[int]
    num_train_pairs: int | list[int] | None = None
    max_input_dim: int | list[int] = DEFAULT_MAX_INPUT_DIM
    level_difficulty_order: list[int] | None = None
    difficulty_ramp_window: int | None = None
    difficulty_ramp_steps: int | None = None

    @model_validator(mode="after")
    def _check(self) -> "ExecutorCurriculumConfig":
        if not self.levels:
            raise ValueError("levels must not be empty")
        unknown_levels = sorted(set(self.levels) - set(LEVELS))
        if unknown_levels:
            raise ValueError(
                f"unknown levels {unknown_levels}; the ladder has {list(LEVELS)}"
            )
        if self.seed == self.val_seed:
            raise ValueError("seed and val_seed must differ")
        if not self.dims:
            raise ValueError("max_input_dim must not be an empty list")
        if not self.paraphrase_ids:
            raise ValueError("paraphrase_ids must not be empty")
        unknown_paraphrases = sorted(set(self.paraphrase_ids) - {0, 1, 2})
        if unknown_paraphrases:
            raise ValueError(
                f"unknown paraphrase_ids {unknown_paraphrases}; valid IDs are 0, 1, 2"
            )

        if self.difficulty_ramp_window is None:
            return self
        combinations = len(set(self.levels)) * len(set(self.dims))
        if self.difficulty_ramp_window < combinations:
            raise ValueError(
                f"difficulty_ramp_window {self.difficulty_ramp_window} cannot hold "
                f"all {combinations} level/size combinations"
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

    @property
    def dims(self) -> list[int]:
        """Return ``max_input_dim`` in normalized list form."""
        if isinstance(self.max_input_dim, int):
            return [self.max_input_dim]
        return list(self.max_input_dim)

    def _tasks(self, *, seed: int, ramp: bool) -> list[SynthTask]:
        return generate_tasks(
            seed,
            self.num_tasks if ramp else self.num_val_tasks,
            self.levels,
            num_train_pairs=self.num_train_pairs,
            max_input_dim=self.max_input_dim,
            level_difficulty_order=self.level_difficulty_order,
            difficulty_ramp_window=self.difficulty_ramp_window if ramp else None,
            difficulty_ramp_steps=self.difficulty_ramp_steps if ramp else None,
        )

    def train_tasks(self) -> list[SynthTask]:
        """Generate the training split on its configured difficulty ramp."""
        return self._tasks(seed=self.seed, ramp=True)

    def val_tasks(self) -> list[SynthTask]:
        """Generate a fixed held-out mixture for comparable validation."""
        return self._tasks(seed=self.val_seed, ramp=False)


def _to_row(task: SynthTask, *, paraphrase_id: int) -> dict:
    return {
        "task_name": TASK_NAME,
        "task_id": task.task_id,
        "transform_description": render_oracle_description(
            task.rule, paraphrase_id=paraphrase_id
        ),
        "description_paraphrase": paraphrase_id,
        "test_input": task.test_input,
        "target": task.target,
        "level": task.level,
    }


class ArcExecutorDataset(RawDataset):
    """Single-grid oracle-description execution samples for GRPO."""

    def __init__(self, seed: int, **kwargs) -> None:
        config = ExecutorCurriculumConfig(seed=seed, **kwargs)
        self.task_name = TASK_NAME
        self.curriculum = config
        self.dataset = Dataset.from_list(self._rows(config.train_tasks(), config))
        self.val_dataset = Dataset.from_list(self._rows(config.val_tasks(), config))

    @staticmethod
    def _rows(tasks: list[SynthTask], config: ExecutorCurriculumConfig) -> list[dict]:
        return [
            _to_row(
                task,
                paraphrase_id=config.paraphrase_ids[index % len(config.paraphrase_ids)],
            )
            for index, task in enumerate(tasks)
        ]
