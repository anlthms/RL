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
"""Materialize TTT Stage-1 rows: RL on the eval tasks' own demo pairs.

One eval-sequence proposer row per (step slot, task): ``train`` = ALL of the
task's public demo pairs (shown to the proposer, outputs included) and
``test`` = the SAME demo pairs, verified server-side by the blind executor
(rule + demo input -> predicted output, scored against the demo output). The
proposer has seen the answers it is scored on -- that is the point: rewards
teach the model to articulate rules that reproduce THIS task's demos. The
executor rollouts themselves are blind. The task's real test pairs are never
read, so nothing hidden can leak.

The row schema matches ``tools/nvarc_cotrain_materialize._proposer_row``
(gym ``arc_transform_refinement_agent`` eval-sequence episodes consumed in
row order, ``data.shuffle: false``). The schedule cycles a shuffled task
order, reshuffling each full pass, for ``(steps + pad_steps) * window``
slots. Validation is not materialized here -- point the run at the existing
real-split validation JSONL.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

PROPOSER_AGENT = "arc_transform_refinement_agent"


def build_task_rows(challenges: dict[str, dict]) -> dict[str, dict]:
    """One all-demos proposer row per task; real test pairs never read."""
    rows = {}
    for task_id in sorted(challenges):
        demos = challenges[task_id]["train"]
        rows[task_id] = {
            "responses_create_params": {"input": []},
            "agent_ref": {"type": "responses_api_agents", "name": PROPOSER_AGENT},
            "train": demos,
            "test": demos,
            "task_id": task_id,
            "bucket": 0,
            "role": "proposer",
        }
    return rows


def schedule(task_ids: list[str], *, slots: int, rng: random.Random) -> list[str]:
    """Cycle shuffled full passes over the tasks until ``slots`` are filled."""
    out: list[str] = []
    while len(out) < slots:
        ordering = list(task_ids)
        rng.shuffle(ordering)
        out.extend(ordering)
    return out[:slots]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True, help="ARC Prize directory")
    parser.add_argument("--split", default="evaluation")
    parser.add_argument("--steps", type=int, required=True, help="grpo.max_num_steps")
    parser.add_argument(
        "--window", type=int, required=True, help="grpo.num_prompts_per_step"
    )
    parser.add_argument(
        "--pad-steps",
        type=int,
        default=8,
        help="extra materialized windows past --steps (collector prefetch tail)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    source = Path(args.data_dir) / f"arc-agi_{args.split}_challenges.json"
    challenges = json.loads(source.read_text(encoding="utf-8"))
    task_rows = build_task_rows(challenges)
    rng = random.Random(args.seed)
    slots = (args.steps + args.pad_steps) * args.window
    order = schedule(sorted(task_rows), slots=slots, rng=rng)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with open(args.output_dir / "train.jsonl", "w", encoding="utf-8") as f:
        for task_id in order:
            f.write(json.dumps(task_rows[task_id]) + "\n")
    demo_counts = [len(row["train"]) for row in task_rows.values()]
    stats = {
        "source": str(source),
        "tasks": len(task_rows),
        "steps": args.steps,
        "pad_steps": args.pad_steps,
        "window": args.window,
        "rows": slots,
        "seed": args.seed,
        "min_demos": min(demo_counts),
        "max_demos": max(demo_counts),
    }
    with open(args.output_dir / "stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    print(json.dumps(stats), flush=True)


if __name__ == "__main__":
    main()
