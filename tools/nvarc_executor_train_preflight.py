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
"""Resolve and validate the NVARC executor training recipe.

Checks the batch/sequence/checkpoint arithmetic, the NVARC training pool
(bucket coverage, no repeated (puzzle, input) rows), and the validation
wiring: validation must be solely the real ARC-AGI-2 evaluation split,
presented as induction rows with its own prompt and processor.
"""

from __future__ import annotations

import argparse
import collections
import json
from typing import Any, cast

from omegaconf import OmegaConf

from nemo_rl.data import ResponseDatasetConfig
from nemo_rl.data.datasets.response_datasets import load_response_dataset
from nemo_rl.data.datasets.response_datasets.nvarc_executor import (
    TASK_NAME,
    NvArcExecutorConfig,
    staged_choices,
)
from nemo_rl.data.datasets.utils import update_single_dataset_config
from nemo_rl.utils.config import (
    load_config,
    parse_hydra_overrides,
    register_omegaconf_resolvers,
)


def _parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--nodes", type=int, required=True)
    return parser.parse_known_args()


def main() -> None:
    register_omegaconf_resolvers()
    args, overrides = _parse_args()
    config = load_config(args.config)
    config = parse_hydra_overrides(config, overrides)
    resolved = cast(dict[str, Any], OmegaConf.to_container(config, resolve=True))

    grpo = resolved["grpo"]
    policy = resolved["policy"]
    data = resolved["data"]
    checkpointing = resolved["checkpointing"]
    errors: list[str] = []

    expected_global_batch = (
        grpo["num_prompts_per_step"] * grpo["num_generations_per_prompt"]
    )
    if policy["train_global_batch_size"] != expected_global_batch:
        errors.append(
            "policy.train_global_batch_size must equal prompts per step x "
            f"generations per prompt ({expected_global_batch})"
        )
    # The pack ceiling is deliberately below max_total_sequence_length: the
    # engine ceiling is sized for real-ARC validation prompts, which are never
    # trained on. A training microbatch still has to hold a full response.
    if (
        policy["sequence_packing"]["train_mb_tokens"]
        <= policy["generation"]["max_new_tokens"]
    ):
        errors.append("train_mb_tokens cannot hold even one full response")
    if (
        policy["sequence_packing"]["train_mb_tokens"]
        > policy["max_total_sequence_length"]
    ):
        errors.append("train_mb_tokens exceeds max_total_sequence_length")
    if resolved["cluster"]["num_nodes"] != args.nodes:
        errors.append(
            f"recipe resolves {resolved['cluster']['num_nodes']} nodes but --nodes is {args.nodes}"
        )
    if data["train"]["dataset_name"] != TASK_NAME:
        errors.append(f"data.train.dataset_name must be {TASK_NAME!r}")
    if data["default"]["processor"] != "arc_executor_data_processor":
        errors.append("executor recipe must use arc_executor_data_processor")
    if data["default"]["prompt_file"] != "examples/prompts/nvarc_executor.txt":
        errors.append("NVARC recipe must use the color-legend executor prompt")
    if not checkpointing["enabled"]:
        errors.append("executor training must save checkpoints for re-benchmarking")
    if checkpointing["metric_name"] != "val:cell_match":
        errors.append(
            "checkpoints must select on val:cell_match (real-split exact match "
            "sits at ~0 for a long time)"
        )

    # Validation must be solely the real ARC-AGI-2 evaluation split.
    if data["train"].get("num_val_tasks") != 0:
        errors.append(
            "data.train.num_val_tasks must be 0: validation runs solely on the "
            "real ARC evaluation split"
        )
    validation = data.get("validation")
    if not isinstance(validation, list) or len(validation) != 1:
        errors.append("data.validation must be a single-entry list")
        validation = []
    for entry in validation:
        if entry.get("dataset_name") != "arc_agi":
            errors.append("validation dataset must be arc_agi")
        if entry.get("split") != "evaluation" or entry.get("validation_split") != (
            "evaluation"
        ):
            errors.append("validation must read only the evaluation files")
        if entry.get("prompt_file") != "examples/prompts/arc_agi.txt":
            errors.append("real-ARC validation must use the induction prompt")
        if entry.get("processor") != "arc_agi_data_processor":
            errors.append("real-ARC validation must use arc_agi_data_processor")

    try:
        curriculum = NvArcExecutorConfig(**data["train"])
    except ValueError as error:
        errors.append(f"invalid executor curriculum: {error}")
        curriculum = None

    train_config = dict(data["train"])
    update_single_dataset_config(train_config, data["default"])
    dataset = load_response_dataset(cast(ResponseDatasetConfig, train_config))
    if dataset.task_name != TASK_NAME:
        errors.append(f"loaded unexpected task {dataset.task_name!r}")
    if dataset.val_dataset is not None:
        errors.append("NVARC-side validation split must be disabled")

    train_ids = dataset.dataset["task_id"]
    # A puzzle may legitimately recur with different pairs, but an identical
    # (puzzle, input) row means a bucket cycled through every pair of every
    # puzzle -- the dataset is too small for the configured num_tasks.
    train_rows = list(zip(train_ids, map(json.dumps, dataset.dataset["test_input"])))
    if len(set(train_rows)) != len(train_rows):
        errors.append("training repeats a (puzzle, input) pair; reduce num_tasks")

    bucket_counts = collections.Counter(dataset.dataset["bucket"])
    final_bucket_step = None
    if curriculum is not None:
        buckets = list(range(1, len(curriculum.bucket_edges) + 2))
        if curriculum.curriculum_window is None:
            if set(bucket_counts) != set(buckets):
                errors.append(
                    f"training covers buckets {sorted(bucket_counts)} but the "
                    f"config defines {buckets}"
                )
        else:
            assert curriculum.curriculum_hold_steps is not None  # validator-paired
            if curriculum.curriculum_window != grpo["num_prompts_per_step"]:
                errors.append(
                    "curriculum_window must equal grpo.num_prompts_per_step "
                    "(one window per training step)"
                )
            run_rows = grpo["max_num_steps"] * grpo["num_prompts_per_step"]
            if len(dataset.dataset) < run_rows:
                errors.append(
                    f"the run consumes {run_rows} rows but num_tasks is only "
                    f"{len(dataset.dataset)}"
                )
            # Step (1-indexed) at which the schedule enters the final bucket;
            # a staged pass must reach it or the top ventiles go untrained.
            final_bucket_step = (
                curriculum.curriculum_hold_steps * (len(buckets) - 1) + 1
            )
            if grpo["max_num_steps"] < final_bucket_step:
                errors.append(
                    f"max_num_steps {grpo['max_num_steps']} never reaches the "
                    f"final bucket (needs {final_bucket_step})"
                )
            # The emitted rows must be exactly the staged schedule over ALL
            # config buckets — this also catches a ventile with no puzzles in
            # the pool, which staging would otherwise silently skip.
            expected = staged_choices(
                count=len(dataset.dataset),
                choices=buckets,
                window=curriculum.curriculum_window,
                hold_steps=curriculum.curriculum_hold_steps,
                cumulative=curriculum.curriculum_cumulative,
            )
            if list(dataset.dataset["bucket"]) != expected:
                errors.append(
                    "training rows do not follow the staged bucket schedule "
                    "(is a config bucket empty in the training pool?)"
                )

    val_count = 0
    for entry in validation:
        val_config = dict(entry)
        update_single_dataset_config(val_config, data["default"])
        val_data = load_response_dataset(cast(ResponseDatasetConfig, val_config))
        val_count += len(val_data.dataset)
    scored = grpo["max_val_samples"] // grpo["val_batch_size"] * grpo["val_batch_size"]
    if scored != val_count:
        errors.append(
            f"validation scores {scored} rows but the real split has {val_count}"
        )

    print(f"config: {args.config}")
    print(f"nodes: {resolved['cluster']['num_nodes']}")
    print(
        "batch: "
        f"{grpo['num_prompts_per_step']} prompts x "
        f"{grpo['num_generations_per_prompt']} generations = {expected_global_batch}"
    )
    print(
        "sequence: "
        f"{policy['max_total_sequence_length']} total / "
        f"{policy['sequence_packing']['train_mb_tokens']} train pack / "
        f"{policy['generation']['max_new_tokens']} generated"
    )
    print(f"train: {len(dataset.dataset)} rows over {len(set(train_ids))} puzzles")
    print(f"train buckets: {dict(sorted(bucket_counts.items()))}")
    if curriculum is not None and curriculum.curriculum_window is not None:
        staging = "cumulative" if curriculum.curriculum_cumulative else "pure"
        print(
            f"curriculum: {staging} staging, "
            f"{curriculum.curriculum_hold_steps} steps per bucket, final "
            f"bucket from step {final_bucket_step} of {grpo['max_num_steps']}"
        )
    print(f"validation: {val_count} real ARC-AGI-2 rows, {scored} scored")
    print(
        "checkpointing: "
        f"every {checkpointing['save_period']} steps, "
        f"keep {checkpointing['keep_top_k']}"
    )

    if errors:
        raise SystemExit("preflight failed:\n- " + "\n- ".join(errors))


if __name__ == "__main__":
    main()
