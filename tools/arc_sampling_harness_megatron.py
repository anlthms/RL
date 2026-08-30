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
"""Run the ARC sampling harness on an inference-only Megatron backend.

Thin launcher around ``tools/arc_sampling_harness.py``: brings up the same
four-GPU Megatron HTTP endpoint the executor benchmark uses, then delegates
to the harness main loop by rewriting ``sys.argv``. All harness flags apply;
``--base-url`` is supplied by the launcher.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from omegaconf import OmegaConf

from nemo_rl.algorithms.grpo import MasterConfig
from nemo_rl.utils.config import (
    load_config,
    parse_hydra_overrides,
    register_omegaconf_resolvers,
)
from tools.nvarc_executor_benchmark_megatron import (
    _build_generation,
    _resolve_weights_path,
    _wait_for_endpoint,
)


def _parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="NeMo-RL master config YAML")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        help="trained checkpoint (step directory or its policy/weights)",
    )
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        help="hydra override for the master config (repeatable)",
    )
    args, harness_args = parser.parse_known_args()
    return args, harness_args


def main() -> None:
    register_omegaconf_resolvers()
    args, harness_args = _parse_args()
    config = load_config(args.config)
    forced_overrides = [
        "policy.generation.backend=megatron",
        "policy.generation.colocated.enabled=false",
        "policy.generation.mcore_generation_config.expose_http_server=true",
    ]
    config = parse_hydra_overrides(config, forced_overrides + args.override)
    config = MasterConfig.model_validate(OmegaConf.to_container(config, resolve=True))

    weights_path = (
        _resolve_weights_path(args.checkpoint) if args.checkpoint is not None else None
    )
    print(f"Harness weights: {weights_path or 'base model'}", flush=True)
    generation = None
    try:
        ray_log_dir = Path("/tmp") / f"arc_harness_ray_{os.getpid()}"
        generation = _build_generation(
            config, log_dir=ray_log_dir, weights_path=weights_path
        )
        base_urls = [
            url for url in generation.dp_openai_server_base_urls if url is not None
        ]
        if len(base_urls) != 1:
            raise RuntimeError(
                f"expected exactly one Megatron HTTP endpoint, received {base_urls}"
            )
        print(f"Megatron endpoint: {base_urls[0]}", flush=True)
        _wait_for_endpoint(base_urls[0])

        from tools.arc_sampling_harness import main as harness_main

        sys.argv = (
            ["arc_sampling_harness"]
            + harness_args
            + ["--base-url", base_urls[0], "--model", config.policy["model_name"]]
        )
        harness_main()
    finally:
        if generation is not None:
            generation.shutdown()


if __name__ == "__main__":
    main()
