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

Usage:
    uv run tools/arc_agi_prompt_stats.py [--max-seq-length 32768]
"""

import argparse

from transformers import AutoTokenizer

from nemo_rl.data.datasets.response_datasets.arc_agi import _load_split
from nemo_rl.data.interfaces import TaskDataSpec
from nemo_rl.environments.arc_agi_grid import format_task_prompt

DEFAULT_DATA_DIR = "/lustre/fs1/portfolios/nemotron/projects/nemotron_sw_pre/users/anthomas/ash/data/arc-prize-2025"


def percentile(values: list[int], fraction: float) -> int:
    return sorted(values)[min(len(values) - 1, int(fraction * len(values)))]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    parser.add_argument("--model-name", default="Qwen/Qwen3-1.7B")
    parser.add_argument("--prompt-file", default="examples/prompts/arc_agi.txt")
    parser.add_argument("--max-seq-length", type=int, default=32768)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    task_spec = TaskDataSpec(task_name="arc_agi", prompt_file=args.prompt_file)

    for split in ("training", "evaluation"):
        rows = _load_split(args.data_dir, split)
        lengths = []
        for row in rows:
            body = format_task_prompt(row["train_pairs"], row["test_input"])
            message = tokenizer.apply_chat_template(
                [{"role": "user", "content": task_spec.prompt.format(body)}],
                tokenize=False,
                add_generation_prompt=True,
                add_special_tokens=False,
            )
            lengths.append(len(tokenizer(message, add_special_tokens=False)["input_ids"]))

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
        print(f"  room left for the response at p99: {args.max_seq_length - percentile(lengths, 0.99)}")


if __name__ == "__main__":
    main()
