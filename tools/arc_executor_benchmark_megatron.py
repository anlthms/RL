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
"""Run the ARC executor benchmark on an inference-only Megatron backend."""

from __future__ import annotations

import argparse
import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import cast

from omegaconf import OmegaConf

from nemo_rl.algorithms.grpo import MasterConfig
from nemo_rl.algorithms.utils import get_tokenizer
from nemo_rl.distributed.virtual_cluster import RayVirtualCluster, init_ray
from nemo_rl.models.generation import configure_generation_config
from nemo_rl.models.generation.megatron.config import (
    MCoreGenerationConfig,
    MCoreGenerationSpecificArgs,
)
from nemo_rl.models.generation.megatron.megatron_generation import MegatronGeneration
from nemo_rl.utils.config import (
    load_config,
    parse_hydra_overrides,
    register_omegaconf_resolvers,
)
from tools.arc_executor_benchmark import BenchmarkConfig, execute_benchmark


class ExecutorMCoreConfig(MCoreGenerationSpecificArgs):
    """Megatron inference fields that this four-GPU benchmark requires."""

    tensor_model_parallel_size: int


def _comma_separated_ints(value: str) -> tuple[int, ...]:
    values = tuple(int(item) for item in value.split(",") if item)
    if not values:
        raise argparse.ArgumentTypeError("value must contain at least one integer")
    return values


def _parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY", "EMPTY"))
    parser.add_argument("--seed", type=int, default=93_821)
    parser.add_argument("--count", type=int, default=150)
    parser.add_argument("--levels", type=_comma_separated_ints, default=(1, 2, 3, 4, 5))
    parser.add_argument("--paraphrases", type=_comma_separated_ints, default=(0, 1, 2))
    parser.add_argument("--num-train-pairs", type=int, default=3)
    parser.add_argument("--max-input-dim", type=int, default=12)
    parser.add_argument("--max-output-tokens", type=int, default=4096)
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--timeout-seconds", type=float, default=300.0)
    parser.add_argument("--temperature", type=float, default=0.0)
    args, overrides = parser.parse_known_args()
    return args, overrides


def _build_generation(
    config: MasterConfig,
    *,
    log_dir: Path,
) -> MegatronGeneration:
    init_ray(log_dir=str(log_dir))
    tokenizer = get_tokenizer(config.policy["tokenizer"])
    generation_config = config.policy.get("generation")
    if generation_config is None:
        raise ValueError("policy.generation is required for executor benchmarking")
    configured_generation = configure_generation_config(
        generation_config,
        tokenizer,
    )
    config.policy["generation"] = configured_generation
    mcore_generation = cast(MCoreGenerationConfig, configured_generation)

    resources = mcore_generation["colocated"]["resources"]
    gpus_per_node = resources["gpus_per_node"]
    num_nodes = resources["num_nodes"]
    if gpus_per_node != 4 or num_nodes != 1:
        raise ValueError(
            "the executor measurement requires exactly one node with four GPUs; "
            f"resolved {num_nodes} node(s) and {gpus_per_node} GPU(s) per node"
        )
    mcore_config = cast(
        ExecutorMCoreConfig,
        mcore_generation["mcore_generation_config"],
    )
    if mcore_config["tensor_model_parallel_size"] != 4:
        raise ValueError(
            "the four-GPU executor measurement requires Megatron tensor parallelism 4"
        )
    if not mcore_config["expose_http_server"]:
        raise ValueError("Megatron expose_http_server must be enabled")

    cluster_config = config.cluster
    generation_cluster = RayVirtualCluster(
        name="arc_executor_megatron_cluster",
        bundle_ct_per_node_list=[gpus_per_node],
        use_gpus=True,
        num_gpus_per_node=gpus_per_node,
        max_colocated_worker_groups=1,
        port_range_low=cluster_config.get("master_port_range_low"),
        port_range_high=cluster_config.get("master_port_range_high"),
    )
    config.policy["generation"]["model_name"] = config.policy["model_name"]
    megatron_config = config.policy.get("megatron_cfg") or {}
    if megatron_config.get("enabled") and "train_iters" not in megatron_config:
        megatron_config["train_iters"] = 1
    return MegatronGeneration(
        config=config.policy,
        tokenizer=tokenizer,
        cluster=generation_cluster,
        name_prefix="arc_executor_megatron",
    )


def _wait_for_endpoint(base_url: str, timeout_seconds: float = 60.0) -> None:
    """Wait for the local Megatron endpoint without consulting proxy settings."""
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    deadline = time.monotonic() + timeout_seconds
    last_error: urllib.error.URLError | None = None
    while time.monotonic() < deadline:
        try:
            with opener.open(f"{base_url}/health", timeout=5) as response:
                if response.status == 200:
                    return
        except urllib.error.URLError as error:
            last_error = error
        time.sleep(1)
    message = f"Megatron endpoint {base_url} was not reachable after {timeout_seconds}s"
    if last_error is None:
        raise RuntimeError(message)
    raise RuntimeError(message) from last_error


def main() -> None:
    register_omegaconf_resolvers()
    args, overrides = _parse_args()
    config = load_config(args.config)
    forced_overrides = [
        "policy.generation.backend=megatron",
        "policy.generation.colocated.enabled=false",
        "policy.generation.mcore_generation_config.expose_http_server=true",
    ]
    config = parse_hydra_overrides(config, forced_overrides + overrides)
    config = MasterConfig.model_validate(OmegaConf.to_container(config, resolve=True))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    generation: MegatronGeneration | None = None
    try:
        ray_log_dir = Path("/tmp") / f"arc_executor_ray_{os.getpid()}"
        generation = _build_generation(config, log_dir=ray_log_dir)
        base_urls = [
            base_url
            for base_url in generation.dp_openai_server_base_urls
            if base_url is not None
        ]
        if len(base_urls) != 1:
            raise RuntimeError(
                "expected exactly one Megatron data-parallel HTTP endpoint, "
                f"received {base_urls}"
            )
        print(f"Megatron executor endpoint: {base_urls[0]}", flush=True)
        _wait_for_endpoint(base_urls[0])
        benchmark_config = BenchmarkConfig(
            base_url=base_urls[0],
            model=config.policy["model_name"],
            seed=args.seed,
            count=args.count,
            levels=args.levels,
            paraphrases=args.paraphrases,
            num_train_pairs=args.num_train_pairs,
            max_input_dim=args.max_input_dim,
            max_output_tokens=args.max_output_tokens,
            concurrency=args.concurrency,
            timeout_seconds=args.timeout_seconds,
            temperature=args.temperature,
        )
        report = execute_benchmark(
            benchmark_config,
            api_key=args.api_key,
            output=args.output,
        )
        print(json.dumps(report["summary"], indent=2))
    finally:
        if generation is not None:
            generation.shutdown()


if __name__ == "__main__":
    main()
