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

import json
import os

from datasets import Dataset

from nemo_rl.data.datasets.raw_dataset import RawDataset
from nemo_rl.environments.arc_agi_grid import REAL_ARC_LEVEL

# ARC releases challenges and solutions as separate files; a split is the pair.
_CHALLENGES_SUFFIX = "_challenges.json"
_SOLUTIONS_SUFFIX = "_solutions.json"


def _load_split(data_dir: str, split: str) -> list[dict]:
    """Join one ARC challenges/solutions pair into one row per test pair.

    A task carries several train pairs (the few-shot context) and one or more
    test inputs. Each test input becomes its own row, repeating that task's
    train pairs.
    """
    prefix = os.path.join(data_dir, f"arc-agi_{split}")
    with open(prefix + _CHALLENGES_SUFFIX, encoding="utf-8") as f:
        challenges = json.load(f)
    with open(prefix + _SOLUTIONS_SUFFIX, encoding="utf-8") as f:
        solutions = json.load(f)

    rows = []
    for task_id, task in challenges.items():
        task_solutions = solutions[task_id]
        assert len(task_solutions) == len(task["test"]), (
            f"task {task_id} has {len(task['test'])} test inputs but "
            f"{len(task_solutions)} solutions; the files are not aligned"
        )
        for test_pair, solution in zip(task["test"], task_solutions):
            rows.append(
                {
                    "task_name": "arc_agi",
                    "task_id": task_id,
                    "train_pairs": task["train"],
                    "test_input": test_pair["input"],
                    "target": solution,
                    # Real tasks are off the synthetic ladder, but they carry
                    # the column so the two can be concatenated into one
                    # validation set and reported as separate buckets.
                    "level": REAL_ARC_LEVEL,
                }
            )
    return rows


class ArcAgiDataset(RawDataset):
    """ARC-AGI tasks loaded from a local ARC Prize data directory.

    Args:
        data_path: Directory holding ``arc-agi_<split>_{challenges,solutions}.json``.
        split: Which ARC split to load for training, default ``"training"``.
        validation_split: Which ARC split to use for validation, default
            ``"evaluation"``. The ``test`` split ships no solutions file
            (competition holdout) and cannot be used here.
    """

    def __init__(
        self,
        data_path: str,
        split: str = "training",
        validation_split: str = "evaluation",
        **kwargs,
    ) -> None:
        self.task_name = "arc_agi"
        self.dataset = Dataset.from_list(_load_split(data_path, split))
        self.val_dataset = Dataset.from_list(_load_split(data_path, validation_split))
