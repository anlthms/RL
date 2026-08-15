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

from nemo_rl.data.datasets.raw_dataset import RawDataset
from nemo_rl.environments.arc_agi_generators import (
    DEFAULT_MAX_INPUT_DIM,
    LEVELS,
    SynthTask,
    generate_tasks,
)

TASK_NAME = "arc_synth"


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
    reproducible because every task is a pure function of ``(seed, index,
    level)``.

    Levels are **mixed within every batch**, not ramped across steps. GRPO's
    advantage is computed within a group of rollouts on one prompt, so a phase
    in which every task is uniformly hopeless -- or uniformly trivial --
    contributes no gradient at all. That is the same degenerate-group failure
    the curriculum exists to fix, and a scheduled ramp reintroduces it. Mixing
    guarantees some level in every batch sits at a solve rate strictly between 0
    and 1.

    Every field that shapes the curriculum is required rather than defaulted
    here: they all live in ``env_arc_synth.yaml``, and a silent fallback to some
    other mixture or seed would be a different experiment reported under the
    same name.

    Args:
        levels: Which ladder levels to draw from, out of
            ``arc_agi_generators.LEVELS``. Tasks cycle through this list in
            order, so a repeated entry weights that level.
        num_tasks: How many training tasks to materialize.
        num_val_tasks: How many held-out tasks to materialize for validation.
        seed: Generator seed for the training tasks.
        val_seed: Generator seed for the held-out tasks. Must differ from
            ``seed`` or validation is a memorization check.
        num_train_pairs: Few-shot pairs per task, or None to draw 2-4 per task
            the way real ARC tasks vary.
        max_input_dim: Upper bound on an input grid's side, or a list of them
            to mix sizes across the batch. Grid size is a difficulty axis
            independent of the transformation: exact match needs every cell
            right, so a large grid can score zero on a rule the model does
            understand. M4.2 measured level 0 at 1.000 on targets under 150
            cells against 0.750 above 300. Mixed rather than ramped, for the
            same reason levels are -- see ``generate_tasks``.
        size_ramp_window: Set to ``grpo.num_prompts_per_step`` to reweight the
            size mixture across the run: every batch still holds every size, but
            the mass shifts from small to large so mean difficulty rises step
            over step. Requires ``data.shuffle: false``, since the schedule
            lives in the row order. Applies to training only -- the held-out
            split stays a fixed mixture, or the metric would not be comparable
            between checkpoints.
        size_ramp_steps: How many steps the schedule spans. Set it to
            ``grpo.max_num_steps``: the dataset is deliberately many times
            larger than any run so no task repeats, so a ramp spread over the
            dataset leaves a 60-step run at 12% of its schedule with the mixture
            barely moved -- inert, but still looking configured.
    """

    def __init__(
        self,
        levels: list[int],
        num_tasks: int,
        num_val_tasks: int,
        seed: int,
        val_seed: int,
        num_train_pairs: int | None = None,
        max_input_dim: int | list[int] = DEFAULT_MAX_INPUT_DIM,
        size_ramp_window: int | None = None,
        size_ramp_steps: int | None = None,
        **kwargs,
    ) -> None:
        levels = list(levels)
        if isinstance(max_input_dim, list) and not max_input_dim:
            raise ValueError("max_input_dim must not be an empty list")
        unknown = sorted(set(levels) - set(LEVELS))
        if unknown:
            raise ValueError(f"unknown levels {unknown}; the ladder has {list(LEVELS)}")
        if seed == val_seed:
            raise ValueError(
                f"seed and val_seed are both {seed}: the held-out split would be "
                "the training tasks verbatim, and validation would measure "
                "memorization rather than generalization"
            )
        self.task_name = TASK_NAME
        self.dataset = Dataset.from_list(
            [
                _to_row(task)
                for task in generate_tasks(
                    seed,
                    num_tasks,
                    levels,
                    num_train_pairs,
                    max_input_dim,
                    size_ramp_window,
                    size_ramp_steps,
                )
            ]
        )
        self.val_dataset = Dataset.from_list(
            [
                _to_row(task)
                for task in generate_tasks(
                    val_seed, num_val_tasks, levels, num_train_pairs, max_input_dim
                )
            ]
        )
