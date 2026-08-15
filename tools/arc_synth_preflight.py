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
r"""Resolve an ARC config and build its datasets, without touching a GPU.

A misconfigured level mixture or a validation set that silently drops one of
its two sources is invisible until a run is over -- the job trains happily on
the wrong thing and reports a number. This loads the config exactly as
``run_grpo.py`` does, applies the same overrides, and materializes the datasets,
so both are checked before a launch spends nodes on them.

Usage:
    uv run tools/arc_synth_preflight.py --config examples/configs/async/qwen3_1p7b_arcsynth_colocated.yaml \\
        data.train.levels=[0] grpo.val_at_start=true
"""

import argparse
import collections

from omegaconf import OmegaConf

from nemo_rl.data.datasets.response_datasets import load_response_dataset
from nemo_rl.data.datasets.utils import update_single_dataset_config
from nemo_rl.environments.arc_agi_grid import REAL_ARC_LEVEL, level_metric_suffix
from nemo_rl.utils.config import (
    load_config,
    parse_hydra_overrides,
    register_omegaconf_resolvers,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args, overrides = parser.parse_known_args()

    register_omegaconf_resolvers()
    config = load_config(args.config)
    if overrides:
        print(f"overrides: {overrides}")
        config = parse_hydra_overrides(config, overrides)
    config = OmegaConf.to_container(config, resolve=True)

    grpo, data, policy = config["grpo"], config["data"], config["policy"]
    print("\n=== resolved ===")
    print(f"  max_num_steps           {grpo['max_num_steps']}")
    print(f"  num_prompts_per_step    {grpo['num_prompts_per_step']}")
    print(f"  val_at_start/period     {grpo['val_at_start']} / {grpo['val_period']}")
    print(f"  max_val_samples         {grpo['max_val_samples']}")
    print(f"  val_batch_size          {grpo['val_batch_size']}")
    print(f"  max_total_sequence_len  {policy['max_total_sequence_length']}")
    print(f"  max_new_tokens          {policy['generation']['max_new_tokens']}")
    print(f"  train.levels            {data['train'].get('levels')}")
    print(f"  train.max_input_dim     {data['train'].get('max_input_dim')}")
    # The size schedule advances one window per trainer step, so the window has
    # to be the trainer's prompts-per-step. A mismatch runs the ramp at the
    # wrong rate and nothing anywhere complains.
    window = data["train"].get("size_ramp_window")
    print(f"  train.size_ramp_window  {window}")
    if window and window != grpo["num_prompts_per_step"]:
        print(
            f"  !! size_ramp_window {window} != num_prompts_per_step "
            f"{grpo['num_prompts_per_step']} -- the size schedule will advance at "
            "the wrong rate"
        )
    if window and data.get("shuffle"):
        print(
            "  !! size_ramp_window is set but data.shuffle is true -- the schedule "
            "lives in the row order and shuffling destroys it"
        )

    # Every policy worker asserts max_lr >= min_lr at construction, and a
    # shared layer that sets only `lr` can drive it under a model's own
    # `min_lr` -- which killed all 32 nano-v3 workers on job 6103023 six
    # minutes in, after the 8-node queue wait.
    optimizer = policy.get("megatron_cfg", {}).get("optimizer", {})
    lr, min_lr = optimizer.get("lr"), optimizer.get("min_lr")
    print(f"  lr / min_lr             {lr} / {min_lr}")
    if lr is not None and min_lr is not None and lr < min_lr:
        print(
            f"  !! lr {lr} < min_lr {min_lr} -- every policy worker will assert on startup"
        )

    # The ARC environment reports these; a checkpoint metric outside the set
    # raises mid-run, at the first validation that tries to select on it.
    ARC_VAL_METRICS = {"accuracy", "grid_match", "cell_match", "format_valid"}
    ckpt = config.get("checkpointing") or {}
    metric = str(ckpt.get("metric_name", "")).removeprefix("val:")
    print(
        f"  checkpointing           enabled={ckpt.get('enabled')} metric={metric or None}"
    )
    if ckpt.get("enabled") and metric not in ARC_VAL_METRICS:
        print(
            f"  !! checkpointing metric {metric!r} is not one the ARC env emits "
            f"({sorted(ARC_VAL_METRICS)}) -- validation will raise mid-run"
        )

    train_cfg = dict(data["train"])
    update_single_dataset_config(train_cfg, data["default"])
    train = load_response_dataset(train_cfg)
    print(f"\n=== train: {train.task_name}, {len(train.dataset)} rows ===")
    print(f"  levels {dict(collections.Counter(train.dataset['level']))}")

    # The validation set is the concatenation of the train dataset's own
    # held-out split and every entry under data.validation. Both have to survive
    # it: a schema mismatch would drop one, and "the real ARC rows are missing"
    # looks exactly like "transfer did not happen".
    val_parts = []
    if getattr(train, "val_dataset", None) is not None:
        val_parts.append((f"{train.task_name} (held out)", train.val_dataset))
    for cfg in data["validation"] or []:
        cfg = dict(cfg)
        update_single_dataset_config(cfg, data["default"])
        val = load_response_dataset(cfg)
        val_parts.append((val.task_name, val.dataset))

    total = 0
    print("\n=== validation ===")
    for name, part in val_parts:
        counts = collections.Counter(part["level"])
        buckets = {level_metric_suffix(k): v for k, v in sorted(counts.items())}
        print(f"  {name:28} {len(part):5} rows  {buckets}")
        total += len(part)

    scored = grpo["max_val_samples"] // grpo["val_batch_size"] * grpo["val_batch_size"]
    print(f"  {'TOTAL':28} {total:5} rows")
    print(
        f"  validation scores {scored} of them (max_val_samples // val_batch_size * val_batch_size)"
    )
    if scored < total:
        print(
            f"  !! {total - scored} rows never scored -- the loader does not shuffle, "
            "so this silently drops the tail of the validation set"
        )
    if not any(REAL_ARC_LEVEL in set(part["level"]) for _, part in val_parts):
        print(
            "  !! no real ARC-AGI-2 rows in validation -- transfer cannot be measured"
        )


if __name__ == "__main__":
    main()
