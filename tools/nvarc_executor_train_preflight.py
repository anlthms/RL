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

Mirrors ``tools/arc_executor_train_preflight.py`` for the NVARC data source:
same batch/sequence/checkpoint checks, plus split-isolation and difficulty-
bucket coverage over the ingested parquet instead of the paraphrase checks
that only exist for the synthetic generator.
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
    if (
        policy["sequence_packing"]["train_mb_tokens"]
        < policy["max_total_sequence_length"]
    ):
        errors.append("train_mb_tokens is smaller than max_total_sequence_length")
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
    if data["validation"] is not None:
        errors.append("executor validation must use only its held-out puzzle split")
    if not checkpointing["enabled"]:
        errors.append("executor training must save checkpoints for re-benchmarking")
    if checkpointing["metric_name"] != "val:accuracy":
        errors.append("executor checkpoints must select on val:accuracy")

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

    train_ids = dataset.dataset["task_id"]
    val_ids = dataset.val_dataset["task_id"]
    if set(train_ids) & set(val_ids):
        errors.append("training and held-out executor puzzles overlap")
    # A puzzle may legitimately recur with different pairs, but an identical
    # (puzzle, input) row means a bucket cycled through every pair of every
    # puzzle -- the dataset is too small for the configured num_tasks.
    train_rows = list(zip(train_ids, map(json.dumps, dataset.dataset["test_input"])))
    if len(set(train_rows)) != len(train_rows):
        errors.append("training repeats a (puzzle, input) pair; reduce num_tasks")

    bucket_counts = collections.Counter(dataset.dataset["level"])
    val_bucket_counts = collections.Counter(dataset.val_dataset["level"])
    if curriculum is not None:
        buckets = set(range(1, len(curriculum.bucket_edges) + 2))
        if set(bucket_counts) != buckets:
            errors.append(
                f"training covers buckets {sorted(bucket_counts)} but the config "
                f"defines {sorted(buckets)}"
            )
        if set(val_bucket_counts) != buckets:
            errors.append(
                f"validation covers buckets {sorted(val_bucket_counts)} but the "
                f"config defines {sorted(buckets)}"
            )
    val_count = len(dataset.val_dataset)
    scored = grpo["max_val_samples"] // grpo["val_batch_size"] * grpo["val_batch_size"]
    if scored != val_count:
        errors.append(
            f"validation scores {scored} rows but the held-out split has {val_count}"
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
        f"{policy['generation']['max_new_tokens']} generated"
    )
    print(f"train: {len(dataset.dataset)} rows over {len(set(train_ids))} puzzles")
    print(f"train buckets: {dict(sorted(bucket_counts.items()))}")
    print(f"validation: {val_count} held-out rows, {scored} scored")
    print(f"validation buckets: {dict(sorted(val_bucket_counts.items()))}")
    print(
        "checkpointing: "
        f"every {checkpointing['save_period']} steps, "
        f"keep {checkpointing['keep_top_k']}"
    )

    if errors:
        raise SystemExit("preflight failed:\n- " + "\n- ".join(errors))


if __name__ == "__main__":
    main()
