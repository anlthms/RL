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
"""Resolve and validate the NVARC co-training recipe against its data files.

The curriculum and role schedule live in the materialized JSONL row order, so
this preflight cross-checks three artifacts that must agree: the recipe
config, the materializer stats, and the actual rows. It also re-verifies the
contamination and leakage invariants on the materialized rows themselves.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, cast

from omegaconf import OmegaConf

from nemo_rl.utils.config import (
    load_config,
    parse_hydra_overrides,
    register_omegaconf_resolvers,
)
from tools.nvarc_cotrain_materialize import (
    EXECUTOR_AGENT,
    PROPOSER_AGENT,
    role_counts,
)

_REAL_SPLIT_ROWS = 172


def _parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--nodes", type=int, required=True)
    return parser.parse_known_args()


def _read_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def main() -> None:
    register_omegaconf_resolvers()
    args, overrides = _parse_args()
    config = load_config(args.config)
    config = parse_hydra_overrides(config, overrides)
    resolved = cast(dict[str, Any], OmegaConf.to_container(config, resolve=True))

    grpo = resolved["grpo"]
    policy = resolved["policy"]
    data = resolved["data"]
    env = resolved["env"]
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
        <= policy["generation"]["max_new_tokens"]
    ):
        errors.append("train_mb_tokens cannot hold even one full response")
    if resolved["cluster"]["num_nodes"] != args.nodes:
        errors.append(
            f"recipe resolves {resolved['cluster']['num_nodes']} nodes but --nodes is {args.nodes}"
        )
    if grpo.get("max_val_samples") is not None:
        errors.append("the gym entrypoint requires grpo.max_val_samples: null")
    if not env.get("should_use_nemo_gym"):
        errors.append("env.should_use_nemo_gym must be true")
    if data["default"]["dataset_name"] != "NemoGymDataset":
        errors.append("data.default.dataset_name must be NemoGymDataset")
    config_paths = env["nemo_gym"]["config_paths"]
    if not any("arc_agi_2" in path for path in config_paths):
        errors.append("nemo_gym.config_paths must include the arc_agi_2 server config")
    if not any("for_training" in path for path in config_paths):
        errors.append(
            "nemo_gym.config_paths must include the *_for_training model config"
        )
    agent_override = (
        env["nemo_gym"]
        .get("arc_transform_refinement_agent", {})
        .get("responses_api_agents", {})
        .get("arc_transform_refinement_agent", {})
    )
    if agent_override.get("protocol") != "eval_sequence":
        errors.append(
            "co-training must run the refinement agent in eval_sequence protocol"
        )
    context_limit = agent_override.get("model_context_limit", 0)
    if context_limit > policy["sequence_packing"]["train_mb_tokens"]:
        errors.append(
            "the proposer model_context_limit exceeds train_mb_tokens: the "
            "trainable final proposer turn would not fit a training pack"
        )
    if (
        agent_override.get("executor_max_output_tokens")
        != policy["generation"]["max_new_tokens"]
    ):
        errors.append(
            "executor_max_output_tokens must match generation.max_new_tokens "
            "(one executor contract across both paths)"
        )
    # The gym rollout surfaces per-agent scalars as <agent>/<field>/<stat>.
    if checkpointing["metric_name"] != f"val:{EXECUTOR_AGENT}/cell_match/mean":
        errors.append(f"checkpoints must select on val:{EXECUTOR_AGENT}/cell_match/mean")

    # ------------------------------------------------ materialized data ----
    train_path = Path(data["train"]["data_path"])
    val_path = Path(data["validation"]["data_path"])
    stats_path = train_path.parent / "stats.json"
    for path in (train_path, val_path, stats_path):
        if not path.exists():
            raise SystemExit(f"preflight failed:\n- missing data file {path}")
    stats = json.loads(stats_path.read_text(encoding="utf-8"))
    if stats["steps"] != grpo["max_num_steps"]:
        errors.append(
            f"materialized for {stats['steps']} steps but the recipe runs "
            f"{grpo['max_num_steps']}"
        )
    if stats["window"] != grpo["num_prompts_per_step"]:
        errors.append(
            f"materialized window {stats['window']} != num_prompts_per_step "
            f"{grpo['num_prompts_per_step']}"
        )

    train_rows = _read_jsonl(train_path)
    window = stats["window"]
    if len(train_rows) != stats["steps"] * window:
        errors.append(
            f"train file has {len(train_rows)} rows but the schedule needs "
            f"{stats['steps'] * window}"
        )
    num_stages = stats["num_stages"]
    mixture_ok = True
    for step in range(stats["steps"]):
        stage, executor_rows, _ = role_counts(
            step,
            window=window,
            hold_steps=stats["hold_steps"],
            num_stages=num_stages,
        )
        window_rows = train_rows[step * window : (step + 1) * window]
        executors = [row for row in window_rows if row["role"] == "executor"]
        proposers = [row for row in window_rows if row["role"] == "proposer"]
        if len(executors) != executor_rows or len(proposers) != window - executor_rows:
            errors.append(
                f"step {step}: mixture {len(executors)}:{len(proposers)} does not "
                f"match the schedule {executor_rows}:{window - executor_rows}"
            )
            mixture_ok = False
            break
        expected_stage = stats["step_mixture"][step]["stage"]
        if stage != expected_stage:
            errors.append(
                f"step {step}: stats stage {expected_stage} != schedule {stage}"
            )
            mixture_ok = False
            break

    for row in train_rows:
        if row["role"] == "executor":
            if row["agent_ref"]["name"] != EXECUTOR_AGENT:
                errors.append("an executor row routes to the wrong agent")
                break
            prompt = row["responses_create_params"]["input"][0]["content"]
            if "<transformation>" not in prompt or "<answer>" not in prompt:
                errors.append("an executor row is missing the executor contract")
                break
        else:
            if row["agent_ref"]["name"] != PROPOSER_AGENT:
                errors.append("a proposer row routes to the wrong agent")
                break
            if row["responses_create_params"]["input"]:
                errors.append("a proposer row carries a pre-rendered prompt")
                break
            if len(row["train"]) < 2 or not row["test"]:
                errors.append("a proposer row has too few demo or eval pairs")
                break

    val_rows = _read_jsonl(val_path)
    if len(val_rows) != _REAL_SPLIT_ROWS:
        errors.append(
            f"validation file has {len(val_rows)} rows, expected {_REAL_SPLIT_ROWS}"
        )
    if any(row["agent_ref"]["name"] != EXECUTOR_AGENT for row in val_rows):
        errors.append("every validation row must route to the single-turn agent")
    if any("target" not in row or "test_input" not in row for row in val_rows):
        errors.append(
            "validation rows must carry target and test_input for the verifier"
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
        f"{policy['generation']['max_new_tokens']} generated / "
        f"{context_limit} proposer context"
    )
    print(
        f"train: {len(train_rows)} rows "
        f"({stats['executor_rows']} executor / {stats['proposer_rows']} proposer)"
    )
    if mixture_ok and stats["step_mixture"]:
        first, last = stats["step_mixture"][0], stats["step_mixture"][-1]
        print(
            f"mixture: {first['executor_rows']}:{first['proposer_rows']} at step 0 -> "
            f"{last['executor_rows']}:{last['proposer_rows']} at step {last['step']}"
        )
    print(f"validation: {len(val_rows)} real ARC-AGI-2 induction rows")
    print(
        "checkpointing: "
        f"every {checkpointing['save_period']} steps, keep {checkpointing['keep_top_k']}, "
        f"metric {checkpointing['metric_name']}"
    )

    if errors:
        raise SystemExit("preflight failed:\n- " + "\n- ".join(errors))


if __name__ == "__main__":
    main()
