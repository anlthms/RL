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
r"""Merge collected teacher shards into one SFT dataset (train/val).

- ``--episode-dirs``: full-fidelity episode shards; every row is taken.
- ``--executor-only-dirs``: older shards contributing ONLY their executor
  rows (their single-turn proposer rows are superseded by episodes).
- Exact-duplicate rows (identical message lists) are dropped.
- Every surviving row must render through the model's own chat template to
  at most ``--max-target-tokens`` minus a collate margin, and its final
  assistant turn must byte-continue the generation opener (``<think>\\n``).
- Split is by puzzle id: ``--val-proposer-tasks`` puzzles that have proposer
  rows plus ``--val-executor-tasks`` executor-only puzzles go to val with
  ALL their rows, so both roles are represented in val_loss.

Train with ``sft.only_unmask_final=true``: proposer rows carry loss-masked
intermediate turns by design (the final-rule-credit contract).
"""

from __future__ import annotations

import argparse
import glob
import json
import random
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode-dirs", nargs="+", required=True)
    parser.add_argument("--executor-only-dirs", nargs="*", default=[])
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--tokenizer-path", required=True)
    parser.add_argument("--max-target-tokens", type=int, default=9216)
    parser.add_argument("--collate-margin", type=int, default=64)
    parser.add_argument("--val-proposer-tasks", type=int, default=16)
    parser.add_argument("--val-executor-tasks", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _load(dirs: list[str], executor_only: bool) -> list[dict]:
    rows = []
    for pattern in dirs:
        for directory in sorted(glob.glob(pattern)):
            path = Path(directory) / "collected.jsonl"
            if not path.exists():
                continue
            for line in open(path, encoding="utf-8"):
                row = json.loads(line)
                if executor_only and row["sft_role"] != "executor":
                    continue
                rows.append(row)
    return rows


def main() -> None:
    args = _parse_args()
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    budget = args.max_target_tokens - args.collate_margin

    rows = _load(args.episode_dirs, executor_only=False)
    episode_count = len(rows)
    rows += _load(args.executor_only_dirs, executor_only=True)
    stats: dict = {
        "loaded_episode_rows": episode_count,
        "loaded_executor_only_rows": len(rows) - episode_count,
    }

    deduped: list[dict] = []
    seen: set[str] = set()
    for row in rows:
        key = json.dumps(row["messages"], sort_keys=True)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(row)
    stats["exact_duplicates_dropped"] = len(rows) - len(deduped)

    kept: list[dict] = []
    over_budget = bad_target = 0
    for row in deduped:
        final = row["messages"][-1]
        if final["role"] != "assistant" or not final["content"].startswith("<think>\n"):
            bad_target += 1
            continue
        rendered = tokenizer.apply_chat_template(
            row["messages"], tokenize=False, add_generation_prompt=False
        )
        tokens = tokenizer(rendered, add_special_tokens=False)["input_ids"]
        if len(tokens) > budget:
            over_budget += 1
            continue
        kept.append(row)
    stats["over_token_budget_dropped"] = over_budget
    stats["bad_target_dropped"] = bad_target

    rng = random.Random(args.seed)
    # Half the proposer val slots come from puzzles with multi-turn episodes,
    # so val_loss covers the revision shape and not just single-round rows.
    multiturn_tasks = sorted(
        {r["task_id"] for r in kept if r.get("rounds", 1) > 1}
    )
    proposer_tasks = sorted(
        {r["task_id"] for r in kept if r["sft_role"] == "proposer"}
        - set(multiturn_tasks)
    )
    executor_only_tasks = sorted(
        {r["task_id"] for r in kept}
        - set(proposer_tasks)
        - set(multiturn_tasks)
    )
    rng.shuffle(multiturn_tasks)
    rng.shuffle(proposer_tasks)
    rng.shuffle(executor_only_tasks)
    multi_quota = args.val_proposer_tasks // 2
    val_ids = (
        set(multiturn_tasks[:multi_quota])
        | set(proposer_tasks[: args.val_proposer_tasks - multi_quota])
        | set(executor_only_tasks[: args.val_executor_tasks])
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rng.shuffle(kept)
    for name, subset in (
        ("train.jsonl", [r for r in kept if r["task_id"] not in val_ids]),
        ("val.jsonl", [r for r in kept if r["task_id"] in val_ids]),
    ):
        with open(output_dir / name, "w", encoding="utf-8") as f:
            for row in subset:
                f.write(json.dumps(row) + "\n")
        split = name.split(".")[0]
        stats[f"{split}_rows"] = len(subset)
        stats[f"{split}_proposer_rows"] = sum(
            1 for r in subset if r["sft_role"] == "proposer"
        )
        stats[f"{split}_multiturn_rows"] = sum(
            1 for r in subset if r.get("rounds", 1) > 1
        )
    stats["val_task_ids"] = sorted(val_ids)
    stats["seed"] = args.seed
    with open(output_dir / "stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
