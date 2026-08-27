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
"""Materialize a standalone ARC evaluation-split validation JSONL.

Same dual-view rows as the co-training validation (single-shot induction +
hidden_test refinement loop per test pair, built by the cotrain
materializer's ``_validation_rows``), for one-off graded evaluations on an
arbitrary ARC release -- e.g. the much easier ARC-AGI-1 evaluation split as
a progress signal while ARC-AGI-2 grid_match sits at zero.

CONTAMINATION: ``--exclude-tasks-json`` marks (not drops) tasks that appear
in other splits. 376/400 ARC-AGI-1 evaluation tasks are inside the
ARC-AGI-2 TRAINING set, which seeds the NVARC training pool, so their
scores are soft upper bounds; ``stats.json`` records the clean task ids
(18 tasks for ARC-AGI-1 eval) for post-hoc slicing of the logged rewards.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tools.nvarc_cotrain_materialize import _validation_rows


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--arc-data-path",
        required=True,
        help="ARC Prize directory with the evaluation split to score",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--exclude-tasks-json",
        nargs="*",
        default=[],
        help=(
            "challenge JSONs whose task ids mark a row as contaminated "
            "(e.g. the ARC-AGI-2 training and evaluation splits)"
        ),
    )
    parser.add_argument(
        "--loop-val-context-limit",
        type=int,
        default=19456,
        help="per-row model_context_limit for the refinement-loop rows",
    )
    parser.add_argument(
        "--induction-prompt-file", default="examples/prompts/arc_agi.txt"
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    template = Path(args.induction_prompt_file).read_text(encoding="utf-8")
    rows = _validation_rows(args.arc_data_path, template, args.loop_val_context_limit)

    contaminated: set[str] = set()
    for path in args.exclude_tasks_json:
        contaminated |= set(json.loads(Path(path).read_text(encoding="utf-8")))
    task_ids = sorted({row["task_id"] for row in rows})
    clean_tasks = [task for task in task_ids if task not in contaminated]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "val.jsonl", "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    stats = {
        "arc_data_path": args.arc_data_path,
        "loop_val_context_limit": args.loop_val_context_limit,
        "rows": len(rows),
        "induction_rows": sum(1 for row in rows if row["role"] == "induction"),
        "loop_rows": sum(1 for row in rows if row["role"] == "induction_loop"),
        "tasks": len(task_ids),
        "clean_tasks": clean_tasks,
        "contaminated_tasks": len(task_ids) - len(clean_tasks),
    }
    with open(output_dir / "stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    print(
        f"{stats['rows']} rows ({stats['induction_rows']} induction + "
        f"{stats['loop_rows']} loop) over {stats['tasks']} tasks; "
        f"{len(clean_tasks)} clean / {stats['contaminated_tasks']} contaminated"
    )


if __name__ == "__main__":
    main()
