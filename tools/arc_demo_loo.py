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
"""Materialize a leave-one-out demo split for the A3 TTT pilot.

For every task in an ARC Prize challenges file, emit one derived task per
demo pair: ``train`` = the remaining demos, ``test`` = the held-out demo's
input, solution = the held-out demo's output. The result is a normal
``arc-agi_<split>_challenges.json`` / ``_solutions.json`` pair, so
``tools/arc_sampling_harness.py`` (and every other consumer of
``load_arc_rows``) runs on it unchanged via ``--data-dir <out> --split
demoloo``.

Every grid written here is a PUBLIC demo pair; real test inputs/outputs of
the source tasks never enter the output. Derived IDs are
``<task_id>_loo<NN>``; run the harness on the FULL split (``--num-tasks``
would sample LOO variants of one base task independently).

A ``demoloo_manifest.json`` maps each derived ID back to its base task and
held-out demo index for per-task aggregation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def build_loo_tasks(
    challenges: dict[str, dict],
) -> tuple[dict[str, dict], dict[str, list], dict[str, dict]]:
    """Derive (challenges, solutions, manifest) LOO triples from source tasks.

    Tasks with fewer than two demos are skipped (an empty ``train`` would
    leave nothing to induce from) and recorded in the manifest.
    """
    loo_challenges: dict[str, dict] = {}
    loo_solutions: dict[str, list] = {}
    manifest: dict[str, dict] = {"rows": {}, "skipped_task_ids": []}
    for task_id in sorted(challenges):
        demos = challenges[task_id]["train"]
        if len(demos) < 2:
            manifest["skipped_task_ids"].append(task_id)
            continue
        for held_out in range(len(demos)):
            loo_id = f"{task_id}_loo{held_out:02d}"
            loo_challenges[loo_id] = {
                "train": [demos[i] for i in range(len(demos)) if i != held_out],
                "test": [{"input": demos[held_out]["input"]}],
            }
            loo_solutions[loo_id] = [demos[held_out]["output"]]
            manifest["rows"][loo_id] = {
                "base_task_id": task_id,
                "held_out_demo_index": held_out,
                "num_demos": len(demos),
            }
    return loo_challenges, loo_solutions, manifest


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True, help="ARC Prize directory")
    parser.add_argument("--split", default="evaluation", help="source split name")
    parser.add_argument("--out-split", default="demoloo", help="derived split name")
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    source = Path(args.data_dir) / f"arc-agi_{args.split}_challenges.json"
    challenges = json.loads(source.read_text(encoding="utf-8"))
    loo_challenges, loo_solutions, manifest = build_loo_tasks(challenges)
    manifest["source"] = str(source)
    manifest["source_task_count"] = len(challenges)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    prefix = args.output_dir / f"arc-agi_{args.out_split}"
    with open(f"{prefix}_challenges.json", "w", encoding="utf-8") as f:
        json.dump(loo_challenges, f)
    with open(f"{prefix}_solutions.json", "w", encoding="utf-8") as f:
        json.dump(loo_solutions, f)
    with open(args.output_dir / "demoloo_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(
        f"{len(loo_challenges)} LOO rows from {len(challenges)} tasks "
        f"({len(manifest['skipped_task_ids'])} skipped) -> {prefix}_*.json",
        flush=True,
    )


if __name__ == "__main__":
    main()
