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
"""Materialize ARC eval tasks' PUBLIC demo pairs as an nvarc-ingested shard.

TTT Stage-1 adapts the model to the eval domain using only public demo
pairs. The teacher collector (``tools/nvarc_teacher_collect.py``) consumes
the nvarc-ingested parquet schema, so this tool re-expresses each eval task
as one puzzle whose ``pairs_json`` holds ONLY its demo pairs -- the task's
real test inputs/outputs are never read, so nothing quarantined can leak.
``canonical_rule`` is empty: eval tasks have no reference rule, which the
episode path never needs (run the collector with ``--executor-traces 0``;
standalone executor traces require reference rules).

``difficulty`` mirrors ``tools/nvarc_ingest.puzzle_difficulty`` (max grid
area over the emitted pairs).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def build_rows(challenges: dict[str, dict], split: str) -> list[dict]:
    """One collector-schema row per task, holding only its demo pairs."""
    rows = []
    for task_id in sorted(challenges):
        demos = challenges[task_id]["train"]
        difficulty = max(
            len(grid) * len(grid[0])
            for pair in demos
            for grid in (pair["input"], pair["output"])
        )
        rows.append(
            {
                "puzzle_id": task_id,
                "canonical_rule": "",
                "pairs_json": json.dumps(demos),
                "difficulty": difficulty,
                "split": split,
            }
        )
    return rows


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True, help="ARC Prize directory")
    parser.add_argument("--split", default="evaluation", help="source split name")
    parser.add_argument(
        "--out-split", default="train", help="split label the collector filters on"
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    args = _parse_args()
    source = Path(args.data_dir) / f"arc-agi_{args.split}_challenges.json"
    challenges = json.loads(source.read_text(encoding="utf-8"))
    rows = build_rows(challenges, args.out_split)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), args.output_dir / "data-00000.parquet")
    demo_counts = [len(json.loads(row["pairs_json"])) for row in rows]
    stats = {
        "source": str(source),
        "tasks": len(rows),
        "demo_pairs": sum(demo_counts),
        "min_demos": min(demo_counts),
        "max_demos": max(demo_counts),
    }
    with open(args.output_dir / "stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    print(json.dumps(stats), flush=True)


if __name__ == "__main__":
    main()
