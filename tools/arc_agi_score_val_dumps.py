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
"""Re-score a run's validation dumps offline, against the copy-the-input hack.

``validate()`` writes every validation chat to ``val_data_step<N>.jsonl``, but
only the total reward travels with it. This recovers the per-term metrics by
matching each record's ``<test_input>`` back to the evaluation split, which needs
no GPU and works on a finished run.

Its reason for existing is the last column. Section 4 of ARC_AGI_2_ENV_PLAN.md
calls for treating a run whose cell accuracy merely tracks a copy of the input as
hacked rather than learning; ``copy_frac`` is that test, and it is what showed the
shaped reward was paying for an echo.

Usage:
    uv run tools/arc_agi_score_val_dumps.py logs/exp_002
"""

import argparse
import glob
import json
import re

from nemo_rl.environments.arc_agi_grid import (
    Grid,
    extract_answer_grid,
    overlay_cell_accuracy,
    serialize_grid,
)

DEFAULT_DATA_DIR = "/lustre/fs1/portfolios/nemotron/projects/nemotron_sw_pre/users/anthomas/ash/data/arc-prize-2025"

_TEST_INPUT_RE = re.compile(r"<test_input>\n(.*?)\n</test_input>", re.DOTALL)
_STEP_RE = re.compile(r"step(\d+)")


def load_eval_pairs(data_dir: str, split: str) -> dict[str, tuple[Grid, Grid]]:
    """Map each serialized test input to its ``(input, target)`` pair.

    Keyed by the serialized text so a record can be matched straight from the
    prompt it carries, rather than by row index, which config changes renumber.
    """
    with open(f"{data_dir}/arc-agi_{split}_challenges.json") as handle:
        challenges = json.load(handle)
    with open(f"{data_dir}/arc-agi_{split}_solutions.json") as handle:
        solutions = json.load(handle)

    by_input = {}
    for task_id, task in challenges.items():
        for index, test in enumerate(task["test"]):
            by_input[serialize_grid(test["input"])] = (
                test["input"],
                solutions[task_id][index],
            )
    return by_input


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("log_dir", help="run log dir holding val_data_step*.jsonl")
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    parser.add_argument("--split", default="evaluation")
    args = parser.parse_args()

    by_input = load_eval_pairs(args.data_dir, args.split)
    copy_baseline = sum(
        overlay_cell_accuracy(grid_in, target) for grid_in, target in by_input.values()
    ) / len(by_input)
    print(
        f"copy-the-input cell_match over {len(by_input)} {args.split} rows: "
        f"{copy_baseline:.4f}  <- a run below this line is not beating an echo\n"
    )
    print("step  grid_match  cell_match  format_valid  copy_frac")

    files = sorted(
        glob.glob(f"{args.log_dir}/val_data_step*.jsonl"),
        key=lambda path: int(_STEP_RE.search(path).group(1)),
    )
    for path in files:
        exact = valid = copied = total = unmatched = 0
        cells = 0.0
        with open(path) as handle:
            for line in handle:
                record = json.loads(line)
                for message_log in record["content"]:
                    prompt = "".join(
                        m["content"] for m in message_log if m["role"] == "user"
                    )
                    response = "".join(
                        m["content"] for m in message_log if m["role"] == "assistant"
                    )
                    match = _TEST_INPUT_RE.search(prompt)
                    pair = by_input.get(match.group(1)) if match else None
                    if pair is None:
                        unmatched += 1
                        continue
                    grid_in, target = pair
                    total += 1
                    pred = extract_answer_grid(response)
                    if pred is None:
                        continue
                    valid += 1
                    exact += pred == target
                    copied += pred == grid_in
                    cells += overlay_cell_accuracy(pred, target)

        step = _STEP_RE.search(path).group(1)
        print(
            f"{step:>4}  {exact / total:10.4f}  {cells / total:10.4f}  "
            f"{valid / total:12.4f}  {copied / total:9.4f}"
            + (f"   ({unmatched} unmatched)" if unmatched else "")
        )


if __name__ == "__main__":
    main()
