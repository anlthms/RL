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

"""Inference-only offline evaluation for a Megatron GRPO checkpoint.

Stands up a dedicated non-colocated ``MegatronGeneration`` model from the checkpoint
weights, generates on the validation set, scores reward via the env, and reports
accuracy via ``grpo.validate()``. Builds no optimizer or scheduler.
"""

import argparse
import os
import pprint

from omegaconf import OmegaConf
from torchdata.stateful_dataloader import StatefulDataLoader

from nemo_rl.algorithms.grpo import (
    MasterConfig,
    validate,
)
from nemo_rl.algorithms.utils import get_tokenizer
from nemo_rl.data.collate_fn import rl_collate_fn
from nemo_rl.data.utils import setup_response_data
from nemo_rl.distributed.virtual_cluster import RayVirtualCluster, init_ray
from nemo_rl.environments.nemo_gym import (
    setup_nemo_gym_config,
    should_use_nemo_gym,
    spinup_nemo_gym_actor,
)
from nemo_rl.models.generation import configure_generation_config
from nemo_rl.models.generation.interfaces import (
    resolve_routed_experts_dtype_name_for_model,
)
from nemo_rl.models.generation.megatron.megatron_generation import MegatronGeneration
from nemo_rl.models.megatron.router_replay import router_replay_enabled
from nemo_rl.utils.config import (
    load_config,
    parse_hydra_overrides,
    register_omegaconf_resolvers,
)


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    """Parse command line arguments (mirrors run_grpo override passthrough)."""
    parser = argparse.ArgumentParser(
        description="Inference-only offline eval of a Megatron GRPO checkpoint"
    )
    parser.add_argument(
        "--config", type=str, default=None, help="Path to YAML config file"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to the checkpoint to evaluate. Either a "
        ".../step_N/policy/weights directory or a .../step_N directory "
        "(the latter is resolved to policy/weights).",
    )
    parser.add_argument(
        "--pretrained-format",
        type=str,
        default=None,
        choices=["megatron_lm", "megatron_bridge"],
        help="Optional: override policy.pretrained_checkpoint.format. Leave unset "
        "to keep whatever the training config already specifies.",
    )

    # Everything else is a Hydra-style override, forwarded verbatim.
    args, overrides = parser.parse_known_args()
    return args, overrides


def _resolve_weights_path(checkpoint: str) -> str:
    """Resolve a checkpoint arg to the concrete ``policy/weights`` directory."""
    checkpoint = os.path.abspath(checkpoint)
    if os.path.basename(checkpoint) == "weights":
        weights_path = checkpoint
    else:
        # Accept a bare .../step_N directory.
        weights_path = os.path.join(checkpoint, "policy", "weights")
    if not os.path.isdir(weights_path):
        raise FileNotFoundError(
            f"Resolved weights path does not exist or is not a directory: "
            f"{weights_path} (from --checkpoint={checkpoint})"
        )
    return weights_path


def main() -> None:
    """Main entry point."""
    register_omegaconf_resolvers()
    args, overrides = parse_args()

    if not args.config:
        raise ValueError("--config is required")

    weights_path = _resolve_weights_path(args.checkpoint)
    print(f"Evaluating checkpoint weights: {weights_path}")

    config = load_config(args.config)
    print(f"Loaded configuration from: {args.config}")

    # Force inference-only Megatron generation on a dedicated (non-colocated)
    # cluster, then apply the caller's overrides last so they win.
    forced_overrides = [
        "policy.generation.backend=megatron",
        "policy.generation.colocated.enabled=false",
    ]
    if args.pretrained_format is not None:
        forced_overrides.append(
            f"policy.pretrained_checkpoint.format={args.pretrained_format}"
        )
    all_overrides = forced_overrides + list(overrides)
    print(f"Overrides: {all_overrides}")
    config = parse_hydra_overrides(config, all_overrides)

    config = OmegaConf.to_container(config, resolve=True)
    config = MasterConfig(**config)

    print("Final config:")
    pprint.pprint(config)

    init_ray()

    # ---- Tokenizer + generation config -------------------------------------
    tokenizer = get_tokenizer(config.policy["tokenizer"])
    assert config.policy["generation"] is not None, (
        "A generation config is required for evaluation"
    )
    config.policy["generation"] = configure_generation_config(
        config.policy["generation"], tokenizer
    )

    # ---- Data + env --------------------------------------------------------
    # NeMo-Gym builds its env after generation is up (it needs the server URLs).
    use_nemo_gym = should_use_nemo_gym(config)
    if use_nemo_gym:
        setup_nemo_gym_config(config, tokenizer)
        _train_dataset, val_dataset = setup_response_data(
            tokenizer, config.data, env_configs=None
        )
        val_task_to_env = None  # bound after the gym actor is spun up
    else:
        (
            _train_dataset,
            val_dataset,
            _task_to_env,
            val_task_to_env,
        ) = setup_response_data(tokenizer, config.data, config.env)

    assert val_dataset is not None, (
        "A validation dataset is required for offline evaluation"
    )

    # NeMo-Gym principle: no hidden pre/post processing -- validate over the whole
    # provided val set in a single batch (matches run_grpo_nemo_gym.py).
    if use_nemo_gym:
        config.grpo["max_val_samples"] = len(val_dataset)
        config.grpo["val_batch_size"] = len(val_dataset)
    if config.grpo.get("max_val_samples") is None:
        config.grpo["max_val_samples"] = len(val_dataset)

    val_dataloader = StatefulDataLoader(
        val_dataset,
        batch_size=config.grpo["val_batch_size"],
        shuffle=False,
        collate_fn=rl_collate_fn,
        num_workers=config.data["num_workers"],
    )
    print(f"  ✓ Validation dataloader loaded with {len(val_dataset)} samples")

    # ---- Dedicated non-colocated inference cluster -------------------------
    inference_resources = config.policy["generation"]["colocated"]["resources"]
    inference_gpus_per_node = inference_resources["gpus_per_node"]
    inference_nodes = inference_resources["num_nodes"]
    assert inference_gpus_per_node and inference_nodes, (
        "policy.generation.colocated.resources.{gpus_per_node,num_nodes} must be "
        "set for a dedicated (non-colocated) inference model."
    )
    cluster_config = config.cluster
    gen_cluster = RayVirtualCluster(
        name="megatron_eval_inference_cluster",
        bundle_ct_per_node_list=[inference_gpus_per_node] * inference_nodes,
        use_gpus=True,
        num_gpus_per_node=inference_gpus_per_node,
        max_colocated_worker_groups=1,
        port_range_low=cluster_config.get("master_port_range_low"),
        port_range_high=cluster_config.get("master_port_range_high"),
    )
    print(
        f"  ✓ Ray inference cluster: {inference_nodes} nodes x "
        f"{inference_gpus_per_node} GPUs"
    )

    # ---- Dedicated inference model (no optimizer, no scheduler) -------------
    config.policy["generation"]["model_name"] = config.policy["model_name"]
    # train_iters is asserted present even on the inference path; unused here.
    megatron_cfg = config.policy.get("megatron_cfg") or {}
    if megatron_cfg.get("enabled") and "train_iters" not in megatron_cfg:
        megatron_cfg["train_iters"] = 1
    policy_generation = MegatronGeneration(
        config=config.policy,
        tokenizer=tokenizer,
        cluster=gen_cluster,
        name_prefix="megatron_eval",
        weights_path=weights_path,
    )

    # NeMo-Gym env is built now that generation server URLs exist.
    if use_nemo_gym:
        model_name = config.policy["generation"]["model_name"]
        enable_router_replay = router_replay_enabled(config.policy)
        nemo_gym_actor = spinup_nemo_gym_actor(
            env_configs=config.env,
            base_urls=policy_generation.dp_openai_server_base_urls,
            model_name=model_name,
            tokenizer=tokenizer,
            enable_router_replay=enable_router_replay,
            routed_experts_dtype=(
                resolve_routed_experts_dtype_name_for_model(model_name)
                if enable_router_replay
                else "int16"
            ),
            use_fastokens=bool(config.policy["tokenizer"].get("use_fastokens")),
        )
        val_task_to_env = {"nemo_gym": nemo_gym_actor}

    # ---- Score -------------------------------------------------------------
    try:
        val_metrics, _timings = validate(
            policy_generation,
            val_dataloader,
            tokenizer,
            val_task_to_env,
            step=0,
            master_config=config,
            logger=None,
        )
    finally:
        policy_generation.shutdown()

    # Keep the exact `Accuracy:` token so existing scrapers keep working.
    print(f"Accuracy: {val_metrics['accuracy']:.4f}")
    print(f"Average reward: {val_metrics['accuracy']:.4f}")
    print(f"Average response length: {val_metrics.get('avg_length', 0.0):.1f} tokens")


if __name__ == "__main__":
    main()
