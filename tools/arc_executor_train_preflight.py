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
"""Resolve and validate the stage-4 ARC executor training recipe."""

from __future__ import annotations

import argparse
import collections
from typing import Any, cast

from omegaconf import OmegaConf

from nemo_rl.data import ResponseDatasetConfig
from nemo_rl.data.datasets.response_datasets import load_response_dataset
from nemo_rl.data.datasets.response_datasets.arc_executor import (
    TASK_NAME,
    ExecutorCurriculumConfig,
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
    if data["validation"] is not None:
        errors.append("executor validation must use only its held-out synthetic split")
    if not checkpointing["enabled"]:
        errors.append("executor training must save checkpoints for re-benchmarking")
    if checkpointing["metric_name"] != "val:accuracy":
        errors.append("executor checkpoints must select on val:accuracy")

    try:
        curriculum = ExecutorCurriculumConfig(**data["train"])
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
    if len(set(train_ids)) != len(train_ids):
        errors.append("training task IDs are not unique")
    if set(train_ids) & set(val_ids):
        errors.append("training and held-out executor tasks overlap")
    paraphrases = collections.Counter(dataset.dataset["description_paraphrase"])
    if curriculum is not None and set(paraphrases) != set(curriculum.paraphrase_ids):
        errors.append("not every configured description paraphrase appears in training")
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
    print(f"train: {len(dataset.dataset)} unique oracle tasks")
    print(f"train levels: {dict(collections.Counter(dataset.dataset['level']))}")
    print(f"train paraphrases: {dict(paraphrases)}")
    print(f"validation: {val_count} held-out tasks, {scored} scored")
    print(
        "checkpointing: "
        f"every {checkpointing['save_period']} steps, "
        f"keep {checkpointing['keep_top_k']}"
    )

    if errors:
        raise SystemExit("preflight failed:\n- " + "\n- ".join(errors))


if __name__ == "__main__":
    main()
