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
"""Materialize the NVARC executor+proposer co-training JSONL for NeMo-Gym.

The gym GRPO entrypoint consumes one JSONL in row order (``data.shuffle:
false``; one ``num_prompts_per_step`` window per step), so both the staged
ventile curriculum and the role schedule are baked into the row order here:

- Step ``s`` sits at curriculum stage ``min(s // hold_steps, last)`` (the
  staged ventile schedule of ``nvarc_executor.staged_choices``, pure staging).
- The role mixture follows ``P(executor | c) = 0.90 - 0.80 c`` with progress
  ``c = stage / (num_stages - 1)``: 90:10 executor:proposer at the first
  ventile, 10:90 at the last.
- Executor rows are single-turn tasks for ``arc_agi_2_simple_agent``: the
  rendered NVARC executor prompt plus verifier-side ``target``/``test_input``.
- Proposer rows are ``arc_transform_refinement_agent`` eval-sequence episodes:
  demo pairs shown, held-out evaluation pairs verified server-side. Demo count
  shrinks (never below 2) when the serialized demos would crowd the trainable
  proposer context.

Validation is the real ARC-AGI-2 evaluation split (120 tasks / 172 test-pair
rows) as single-turn induction rows for the same simple agent, so
``cell_match`` reaches checkpoint selection as a per-agent metric.

A reactive pause of the role schedule on executor-validation regression (the
proposal's pause rule) is NOT possible with a pre-materialized file; monitor
the per-agent validation metrics and re-materialize if the executor drifts.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from nemo_rl.data.datasets.response_datasets.arc_agi import (
    _load_split as load_arc_split,
)
from nemo_rl.data.datasets.response_datasets.nvarc_executor import (
    NvArcExecutorConfig,
    _load_split as load_nvarc_split,
    _PairSampler,
)
from nemo_rl.environments.arc_agi_grid import format_task_prompt, serialize_grid

EXECUTOR_AGENT = "arc_agi_2_simple_agent"
PROPOSER_AGENT = "arc_transform_refinement_agent"

# Fixed byte overhead of the proposer prompt around the serialized demo grids
# (instructions, color legend, section scaffolding), plus per-demo framing.
_PROPOSER_PROMPT_OVERHEAD = 900
_PER_DEMO_OVERHEAD = 60
_MIN_DEMO_PAIRS = 2


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir", required=True, help="nvarc_ingest.py output directory"
    )
    parser.add_argument(
        "--arc-data-path",
        required=True,
        help="ARC Prize directory with the evaluation split",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--steps", type=int, required=True, help="grpo.max_num_steps")
    parser.add_argument(
        "--window", type=int, required=True, help="grpo.num_prompts_per_step"
    )
    parser.add_argument(
        "--hold-steps", type=int, required=True, help="steps per ventile stage"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--demo-pairs", type=int, default=3)
    parser.add_argument("--eval-pairs", type=int, default=2)
    parser.add_argument(
        "--proposer-prompt-byte-budget",
        type=int,
        default=8000,
        help="shrink demo count while the estimated proposer prompt exceeds this",
    )
    parser.add_argument(
        "--executor-prompt-file", default="examples/prompts/nvarc_executor.txt"
    )
    parser.add_argument(
        "--induction-prompt-file", default="examples/prompts/arc_agi.txt"
    )
    parser.add_argument(
        "--bucket-edges",
        type=int,
        nargs="+",
        default=None,
        help="difficulty bucket edges; defaults to the ventile edges on NvArcExecutorConfig",
    )
    return parser.parse_args()


def role_counts(
    step: int, *, window: int, hold_steps: int, num_stages: int
) -> tuple[int, int, float]:
    """Stage, executor-row count, and curriculum progress for one step."""
    stage = min(step // hold_steps, num_stages - 1)
    progress = stage / (num_stages - 1) if num_stages > 1 else 1.0
    executor_rows = round(window * (0.90 - 0.80 * progress))
    return stage, executor_rows, progress


class _BucketQueue:
    """Pop puzzles from one difficulty bucket, reshuffling after a full pass."""

    def __init__(self, puzzles: list[dict], rng: random.Random) -> None:
        self._puzzles = puzzles
        self._rng = rng
        self._queue: list[dict] = []

    def next(self) -> dict:
        if not self._queue:
            self._queue = list(self._puzzles)
            self._rng.shuffle(self._queue)
        return self._queue.pop()


def _executor_row(puzzle: dict, pair: dict, *, bucket: int, template: str) -> dict:
    task_body = (
        "<transformation>\n"
        f"{puzzle['canonical_rule']}\n"
        "</transformation>\n"
        "<input>\n"
        f"{serialize_grid(pair['input'])}\n"
        "</input>"
    )
    prompt = template.format(task_body)
    return {
        "responses_create_params": {"input": [{"role": "user", "content": prompt}]},
        "agent_ref": {"type": "responses_api_agents", "name": EXECUTOR_AGENT},
        "target": pair["output"],
        "test_input": pair["input"],
        "task_id": puzzle["puzzle_id"],
        "bucket": bucket,
        "role": "executor",
    }


def _proposer_row(
    puzzle: dict,
    *,
    bucket: int,
    demo_pairs: int,
    eval_pairs: int,
    byte_budget: int,
    rng: random.Random,
) -> dict | None:
    pairs = json.loads(puzzle["pairs_json"])
    if len(pairs) < _MIN_DEMO_PAIRS + eval_pairs:
        return None
    demos_wanted = min(demo_pairs, len(pairs) - eval_pairs)
    chosen = rng.sample(range(len(pairs)), demos_wanted + eval_pairs)
    demos = [pairs[index] for index in chosen[:demos_wanted]]
    evals = [pairs[index] for index in chosen[demos_wanted:]]

    def estimated_bytes(demo_list: list[dict]) -> int:
        grids = sum(
            len(serialize_grid(pair["input"])) + len(serialize_grid(pair["output"]))
            for pair in demo_list
        )
        return _PROPOSER_PROMPT_OVERHEAD + grids + _PER_DEMO_OVERHEAD * len(demo_list)

    while len(demos) > _MIN_DEMO_PAIRS and estimated_bytes(demos) > byte_budget:
        demos.pop()
    return {
        "responses_create_params": {"input": []},
        "agent_ref": {"type": "responses_api_agents", "name": PROPOSER_AGENT},
        "train": demos,
        "test": evals,
        "task_id": puzzle["puzzle_id"],
        "bucket": bucket,
        "role": "proposer",
    }


def _validation_rows(arc_data_path: str, template: str) -> list[dict]:
    rows = []
    for source in load_arc_split(arc_data_path, "evaluation"):
        prompt = template.format(
            format_task_prompt(source["train_pairs"], source["test_input"])
        )
        rows.append(
            {
                "responses_create_params": {
                    "input": [{"role": "user", "content": prompt}]
                },
                "agent_ref": {"type": "responses_api_agents", "name": EXECUTOR_AGENT},
                "target": source["target"],
                "test_input": source["test_input"],
                "task_id": source["task_id"],
                "role": "induction",
            }
        )
    return rows


def main() -> None:
    args = _parse_args()
    executor_template = Path(args.executor_prompt_file).read_text(encoding="utf-8")
    induction_template = Path(args.induction_prompt_file).read_text(encoding="utf-8")
    config = NvArcExecutorConfig(
        data_dir=args.data_dir,
        num_tasks=1,
        num_val_tasks=0,
        seed=args.seed,
        val_seed=args.seed + 1,
        **({"bucket_edges": args.bucket_edges} if args.bucket_edges else {}),
    )
    num_stages = len(config.bucket_edges) + 1

    puzzles = load_nvarc_split(args.data_dir, "train")
    by_bucket: dict[int, list[dict]] = {}
    for puzzle in puzzles:
        by_bucket.setdefault(config.bucket_id(puzzle["difficulty"]), []).append(puzzle)
    missing = sorted(set(range(1, num_stages + 1)) - set(by_bucket))
    if missing:
        raise SystemExit(f"train pool has no puzzles in buckets {missing}")

    rng = random.Random(args.seed)
    executor_queues = {
        bucket: _BucketQueue(pool, rng) for bucket, pool in by_bucket.items()
    }
    proposer_queues = {
        bucket: _BucketQueue(pool, rng) for bucket, pool in by_bucket.items()
    }
    pair_samplers: dict[str, _PairSampler] = {}

    train_rows: list[dict] = []
    step_mixture: list[dict] = []
    for step in range(args.steps):
        stage, executor_rows, progress = role_counts(
            step, window=args.window, hold_steps=args.hold_steps, num_stages=num_stages
        )
        bucket = sorted(by_bucket)[stage]
        emitted_proposers = 0
        for _ in range(executor_rows):
            puzzle = executor_queues[bucket].next()
            if puzzle["puzzle_id"] not in pair_samplers:
                pair_samplers[puzzle["puzzle_id"]] = _PairSampler(
                    json.loads(puzzle["pairs_json"]), rng
                )
            pair = pair_samplers[puzzle["puzzle_id"]].next()
            # The prompt is structurally target-free: _executor_row renders
            # only the canonical rule and the input grid. (A substring check
            # against the serialized target would false-positive on crop-style
            # tasks whose output is legitimately a sub-grid of the input.)
            row = _executor_row(puzzle, pair, bucket=bucket, template=executor_template)
            train_rows.append(row)
        skipped = 0
        while emitted_proposers < args.window - executor_rows:
            row = _proposer_row(
                proposer_queues[bucket].next(),
                bucket=bucket,
                demo_pairs=args.demo_pairs,
                eval_pairs=args.eval_pairs,
                byte_budget=args.proposer_prompt_byte_budget,
                rng=rng,
            )
            if row is None:
                skipped += 1
                if skipped > len(by_bucket[bucket]):
                    raise SystemExit(
                        f"bucket {bucket} has no puzzle with enough pairs for "
                        f"{_MIN_DEMO_PAIRS} demos + {args.eval_pairs} evals"
                    )
                continue
            train_rows.append(row)
            emitted_proposers += 1
        step_mixture.append(
            {
                "step": step,
                "stage": stage,
                "bucket": bucket,
                "progress": round(progress, 4),
                "executor_rows": executor_rows,
                "proposer_rows": args.window - executor_rows,
            }
        )

    val_rows = _validation_rows(args.arc_data_path, induction_template)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "train.jsonl", "w", encoding="utf-8") as f:
        for row in train_rows:
            f.write(json.dumps(row) + "\n")
    with open(output_dir / "val.jsonl", "w", encoding="utf-8") as f:
        for row in val_rows:
            f.write(json.dumps(row) + "\n")
    stats = {
        "seed": args.seed,
        "steps": args.steps,
        "window": args.window,
        "hold_steps": args.hold_steps,
        "num_stages": num_stages,
        "bucket_edges": config.bucket_edges,
        "demo_pairs": args.demo_pairs,
        "eval_pairs": args.eval_pairs,
        "train_rows": len(train_rows),
        "executor_rows": sum(1 for row in train_rows if row["role"] == "executor"),
        "proposer_rows": sum(1 for row in train_rows if row["role"] == "proposer"),
        "validation_rows": len(val_rows),
        "step_mixture": step_mixture,
    }
    with open(output_dir / "stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    print(
        f"train: {stats['train_rows']} rows "
        f"({stats['executor_rows']} executor / {stats['proposer_rows']} proposer) "
        f"over {args.steps} steps; validation: {len(val_rows)} rows"
    )
    print(f"first step mixture: {step_mixture[0]}; last: {step_mixture[-1]}")


if __name__ == "__main__":
    main()
