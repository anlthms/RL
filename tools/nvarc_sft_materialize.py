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
"""Materialize the NVARC no-think SFT JSONL (executor + proposer imitation).

A SHORT format/length prior before anti-runaway RL, not a convergence run:
both roles imitate empty-think targets (``NO_THINK_PREFIX`` + payload, byte-
continuing nano-v3's generation opener) so the RL policy starts from a prior
that closes its think block and answers in-format instead of runaway
thinking (31.5% of trained v2 proposer turns truncated mid-think at the
output cap).

DELIBERATE INVARIANT RELAXATION (2026-08-26, user-approved, recorded in
ARC_AGI_2_ENV_PLAN.md): the puzzles' reference canonical rules ARE proposer
imitation targets here. Everywhere else in the campaign reference rules are
never shown to the proposer.

Prompts are byte-parity with the RL pipeline:
- Executor examples reuse the cotrain materializer's row rendering (the
  native single-turn executor contract); the target is the gold grid in one
  ``<answer>`` block, exactly what a perfect rollout emits.
- Proposer examples reuse the cotrain demo sampling (same byte-budget
  shrinking) and the gym agent's own prompt builder + system instructions;
  the target is the puzzle's canonical 4-section rule, re-rendered through
  ``parse_canonical_rule`` so it is exactly the artifact the episode parser
  would extract.

Output: ``train.jsonl`` / ``val.jsonl`` in the OpenAI messages format
(``dataset_name: openai_format``), split by puzzle id, plus ``stats.json``.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

from nemo_rl.data.datasets.response_datasets.nvarc_executor import (
    _load_split as load_nvarc_split,
)
from nemo_rl.environments.arc_agi_grid import serialize_grid
from tools.nvarc_cotrain_materialize import _executor_row, _proposer_row

# The proposer prompt must be byte-identical to the agent's episodes, so it
# is built by the gym's own code (stdlib-only module).
_GYM_ROOT = Path(__file__).resolve().parents[1] / "3rdparty" / "Gym-workspace" / "Gym"
sys.path.insert(0, str(_GYM_ROOT))
# pyrefly: ignore  # import-error
from resources_servers.arc_agi_2.logic import (  # noqa: E402
    TransformDescriptionParseError,
    build_nvarc_proposer_prompt,
    parse_canonical_rule,
)

# Mirrors PROPOSER_INSTRUCTIONS in the gym refinement agent's app.py (the
# agent module imports fastapi/nemo_gym, too heavy for this tool); a guarded
# unit test asserts the copies stay identical.
PROPOSER_INSTRUCTIONS = (
    "Act only as an ARC transformation-rule proposer. Return the requested "
    "transformation-description artifact; do not execute the test grid."
)

# nano-v3's generation prompt ends with an OPEN think tag plus newline
# ("<|im_start|>assistant\n<think>\n"), so the trained assistant turn must
# begin with those exact bytes for the taught continuation ("</think>" then
# the payload) to be in-distribution at rollout time. v1 targets carried no
# think tags, so the template rendered them as "<think></think>" + payload:
# teacher-forced loss looked great while inference (which forces the opener)
# collapsed to runaway thinking (loop-val format_valid 0.017 on the v1 SFT
# checkpoint).
NO_THINK_PREFIX = "<think>\n</think>\n\n"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir", required=True, help="nvarc_ingest.py output directory"
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--executor-examples", type=int, default=4096)
    parser.add_argument("--proposer-examples", type=int, default=4096)
    parser.add_argument(
        "--val-puzzles",
        type=int,
        default=64,
        help="puzzles held out for val examples (split by puzzle id)",
    )
    parser.add_argument(
        "--val-examples",
        type=int,
        default=256,
        help="val examples drawn from the held-out puzzles, half per role",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--demo-pairs", type=int, default=3)
    parser.add_argument("--eval-pairs", type=int, default=2)
    parser.add_argument(
        "--proposer-prompt-byte-budget",
        type=int,
        default=8000,
        help="same demo-shrinking budget as the cotrain materializer",
    )
    parser.add_argument(
        "--executor-prompt-file", default="examples/prompts/nvarc_executor.txt"
    )
    return parser.parse_args()


def executor_example(puzzle: dict, pair: dict, *, template: str) -> dict:
    """One native executor task with the gold grid as a think-free target."""
    row = _executor_row(puzzle, pair, bucket=0, template=template)
    prompt = row["responses_create_params"]["input"][0]["content"]
    answer = f"{NO_THINK_PREFIX}<answer>\n{serialize_grid(pair['output'])}\n</answer>"
    return {
        "messages": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": answer},
        ],
        "task_id": puzzle["puzzle_id"],
        "sft_role": "executor",
    }


def proposer_example(
    puzzle: dict,
    *,
    demo_pairs: int,
    eval_pairs: int,
    byte_budget: int,
    rng: random.Random,
) -> dict | None:
    """One demo-pairs induction task with the canonical rule as the target.

    Returns None when the puzzle has too few pairs (same rule as the cotrain
    materializer) or its reference rule does not parse as a canonical
    4-section rule.
    """
    row = _proposer_row(
        puzzle,
        bucket=0,
        demo_pairs=demo_pairs,
        eval_pairs=eval_pairs,
        byte_budget=byte_budget,
        rng=rng,
    )
    if row is None:
        return None
    try:
        rule = parse_canonical_rule(puzzle["canonical_rule"])
    except TransformDescriptionParseError:
        return None
    prompt = build_nvarc_proposer_prompt(demo_pairs=row["train"])
    return {
        "messages": [
            {"role": "system", "content": PROPOSER_INSTRUCTIONS},
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": NO_THINK_PREFIX + rule},
        ],
        "task_id": puzzle["puzzle_id"],
        "sft_role": "proposer",
    }


def _emit(
    puzzles: list[dict],
    *,
    executor_count: int,
    proposer_count: int,
    template: str,
    args: argparse.Namespace,
    rng: random.Random,
) -> tuple[list[dict], int]:
    """Cycle the puzzle pool until each role's quota is met."""
    examples: list[dict] = []
    skipped_proposer = 0
    emitted_executor = 0
    emitted_proposer = 0
    cursor = 0
    passes = 0
    while emitted_executor < executor_count or emitted_proposer < proposer_count:
        if cursor == 0:
            passes += 1
            if passes > 1 + (executor_count + proposer_count) // max(len(puzzles), 1):
                raise SystemExit(
                    f"puzzle pool of {len(puzzles)} cannot fill "
                    f"{executor_count}+{proposer_count} examples "
                    f"({skipped_proposer} proposer puzzles skipped)"
                )
            rng.shuffle(puzzles)
        puzzle = puzzles[cursor]
        cursor = (cursor + 1) % len(puzzles)
        if emitted_executor < executor_count:
            pairs = json.loads(puzzle["pairs_json"])
            example = executor_example(puzzle, rng.choice(pairs), template=template)
            examples.append(example)
            emitted_executor += 1
        if emitted_proposer < proposer_count:
            example = proposer_example(
                puzzle,
                demo_pairs=args.demo_pairs,
                eval_pairs=args.eval_pairs,
                byte_budget=args.proposer_prompt_byte_budget,
                rng=rng,
            )
            if example is None:
                skipped_proposer += 1
            else:
                examples.append(example)
                emitted_proposer += 1
    rng.shuffle(examples)
    return examples, skipped_proposer


def main() -> None:
    args = _parse_args()
    template = Path(args.executor_prompt_file).read_text(encoding="utf-8")
    puzzles = load_nvarc_split(args.data_dir, "train")
    rng = random.Random(args.seed)
    rng.shuffle(puzzles)
    if len(puzzles) <= args.val_puzzles:
        raise SystemExit(
            f"{len(puzzles)} train puzzles cannot spare {args.val_puzzles} for val"
        )
    val_pool = puzzles[: args.val_puzzles]
    train_pool = puzzles[args.val_puzzles :]

    train_rows, train_skipped = _emit(
        train_pool,
        executor_count=args.executor_examples,
        proposer_count=args.proposer_examples,
        template=template,
        args=args,
        rng=rng,
    )
    val_rows, val_skipped = _emit(
        val_pool,
        executor_count=args.val_examples // 2,
        proposer_count=args.val_examples - args.val_examples // 2,
        template=template,
        args=args,
        rng=rng,
    )

    for row in train_rows + val_rows:
        target = row["messages"][-1]["content"]
        assert target.startswith(NO_THINK_PREFIX), (
            "SFT targets must continue the generation opener"
        )
        assert "<think" not in target[len(NO_THINK_PREFIX) :], (
            "SFT targets must carry an empty think block only"
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "train.jsonl", "w", encoding="utf-8") as f:
        for row in train_rows:
            f.write(json.dumps(row) + "\n")
    with open(output_dir / "val.jsonl", "w", encoding="utf-8") as f:
        for row in val_rows:
            f.write(json.dumps(row) + "\n")

    def role_count(rows: list[dict], role: str) -> int:
        return sum(1 for row in rows if row["sft_role"] == role)

    stats = {
        "seed": args.seed,
        "demo_pairs": args.demo_pairs,
        "eval_pairs": args.eval_pairs,
        "proposer_prompt_byte_budget": args.proposer_prompt_byte_budget,
        "val_puzzles": args.val_puzzles,
        "train_rows": len(train_rows),
        "train_executor_rows": role_count(train_rows, "executor"),
        "train_proposer_rows": role_count(train_rows, "proposer"),
        "train_skipped_proposer_puzzles": train_skipped,
        "val_rows": len(val_rows),
        "val_executor_rows": role_count(val_rows, "executor"),
        "val_proposer_rows": role_count(val_rows, "proposer"),
        "val_skipped_proposer_puzzles": val_skipped,
        "max_example_bytes": max(
            sum(len(m["content"]) for m in row["messages"])
            for row in train_rows + val_rows
        ),
    }
    with open(output_dir / "stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    print(
        f"train: {stats['train_rows']} rows "
        f"({stats['train_executor_rows']} executor / {stats['train_proposer_rows']} proposer); "
        f"val: {stats['val_rows']} rows; max example {stats['max_example_bytes']} bytes"
    )


if __name__ == "__main__":
    main()
