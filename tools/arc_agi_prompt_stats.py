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
"""Measure ARC-AGI prompt lengths at the real tokenizer.

Milestone 0 of ARC_AGI_2_ENV_PLAN.md: confirm the serialized few-shot prompts
leave room for reasoning inside the configured context, before spending any GPU
time on a run that would silently mask every overlong sample.

``--synth`` measures the generated ladder instead of the real corpus, which is
M4.0's go check: the generated prompts have to fit the same context, and a dump
is the only way to see what a level actually looks like before spending GPU time
on it.

Usage:
    uv run tools/arc_agi_prompt_stats.py [--max-seq-length 32768]
    uv run tools/arc_agi_prompt_stats.py --dump 3 --dump-file /tmp/arc_chats.txt
    uv run tools/arc_agi_prompt_stats.py --synth --dump 2
"""

import argparse
from pathlib import Path

from transformers import AutoTokenizer

from nemo_rl.data.datasets.response_datasets.arc_agi import _load_split
from nemo_rl.data.datasets.response_datasets.arc_synth import _to_row
from nemo_rl.data.interfaces import TaskDataSpec
from nemo_rl.environments.arc_agi_generators import LEVELS, generate_task
from nemo_rl.environments.arc_agi_grid import format_task_prompt, serialize_grid

DEFAULT_DATA_DIR = "/lustre/fs1/portfolios/nemotron/projects/nemotron_sw_pre/users/anthomas/ash/data/arc-prize-2025"
# The model the async arm trains; prompt lengths are tokenizer-specific, so the
# published numbers have to be re-measured whenever this changes.
DEFAULT_MODEL = "/lustre/fsw/portfolios/llmservice/users/wdykas/data/nano-v3-sft-64gbs-nickel-capybara-5e-5-constant-wd-0-load-bal-1e-4-lcx3-pretool-base-temp1-iter-0013600-hf"


def percentile(values: list[int], fraction: float) -> int:
    return sorted(values)[min(len(values) - 1, int(fraction * len(values)))]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    parser.add_argument("--model-name", default=DEFAULT_MODEL)
    parser.add_argument("--prompt-file", default="examples/prompts/arc_agi.txt")
    parser.add_argument("--max-seq-length", type=int, default=32768)
    parser.add_argument(
        "--dump",
        type=int,
        default=0,
        help="write this many fully rendered chats per split to --dump-file, "
        "so the exact text the model sees can be eyeballed without a GPU run",
    )
    parser.add_argument("--dump-file", default="arc_agi_sample_chats.txt")
    parser.add_argument(
        "--synth",
        action="store_true",
        help="measure the synthetic ladder instead of the real corpus, "
        "reporting one split per difficulty level",
    )
    parser.add_argument("--synth-seed", type=int, default=0)
    parser.add_argument(
        "--synth-tasks",
        type=int,
        default=500,
        help="tasks to generate per level",
    )
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    task_spec = TaskDataSpec(task_name="arc_agi", prompt_file=args.prompt_file)
    dumped: list[str] = []

    if args.synth:
        splits = {
            f"level_{level}": [
                _to_row(generate_task(args.synth_seed, index, level))
                for index in range(args.synth_tasks)
            ]
            for level in LEVELS
        }
    else:
        splits = {
            split: _load_split(args.data_dir, split)
            for split in ("training", "evaluation")
        }

    for split, rows in splits.items():
        lengths = []
        dumped_this_split = 0
        for row in rows:
            body = format_task_prompt(row["train_pairs"], row["test_input"])
            message = tokenizer.apply_chat_template(
                [{"role": "user", "content": task_spec.prompt.format(body)}],
                tokenize=False,
                add_generation_prompt=True,
                add_special_tokens=False,
            )
            token_ids = tokenizer(message, add_special_tokens=False)["input_ids"]
            lengths.append(len(token_ids))

            if dumped_this_split < args.dump:
                dumped_this_split += 1
                dumped.append(
                    f"{'=' * 78}\n"
                    f"split={split} task_id={row['task_id']} "
                    f"prompt_tokens={len(token_ids)}\n"
                    f"{'=' * 78}\n{message}\n"
                    f"{'-' * 78}\nTARGET (not shown to the model):\n"
                    f"{serialize_grid(row['target'])}\n"
                )

        over = sum(length >= args.max_seq_length for length in lengths)
        print(f"\n{split}: {len(rows)} rows")
        print(f"  median {percentile(lengths, 0.50)}")
        print(f"  p90    {percentile(lengths, 0.90)}")
        print(f"  p99    {percentile(lengths, 0.99)}")
        print(f"  max    {max(lengths)}")
        print(
            f"  >= max_seq_length ({args.max_seq_length}): {over} "
            f"({100 * over / len(rows):.2f}%) -- these get loss_multiplier=0"
        )
        print(
            f"  room left for the response at p99: {args.max_seq_length - percentile(lengths, 0.99)}"
        )

    if dumped:
        Path(args.dump_file).write_text("\n".join(dumped))
        print(f"\nwrote {len(dumped)} rendered chats to {args.dump_file}")


if __name__ == "__main__":
    main()
