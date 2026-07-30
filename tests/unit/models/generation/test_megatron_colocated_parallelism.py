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

import pytest

from nemo_rl.models.generation.megatron.megatron_generation import (
    _validate_colocated_parallelism,
)


def _config(train: dict[str, int], generation: dict[str, int]) -> dict:
    """Build the minimal PolicyConfig slice the validator reads."""
    return {
        "megatron_cfg": train,
        "generation": {"mcore_generation_config": generation},
    }


def test_matching_parallelism_is_accepted() -> None:
    cfg = _config(
        {"tensor_model_parallel_size": 2, "expert_model_parallel_size": 4},
        {"tensor_model_parallel_size": 2, "expert_model_parallel_size": 4},
    )
    _validate_colocated_parallelism(cfg)


def test_generation_may_omit_parallelism() -> None:
    """A generation block that declares no parallelism inherits it silently-by-design."""
    cfg = _config(
        {"tensor_model_parallel_size": 2},
        {"expose_http_server": True},
    )
    _validate_colocated_parallelism(cfg)


@pytest.mark.parametrize(
    "key",
    [
        "tensor_model_parallel_size",
        "pipeline_model_parallel_size",
        "expert_model_parallel_size",
        "expert_tensor_parallel_size",
    ],
)
def test_mismatched_weight_sharding_raises(key: str) -> None:
    cfg = _config({key: 1}, {key: 2})
    with pytest.raises(ValueError, match="cannot apply a separate generation"):
        _validate_colocated_parallelism(cfg)


def test_mismatched_context_parallel_warns() -> None:
    """CP shards activations, not weights, so it is ignored rather than rejected."""
    cfg = _config({"context_parallel_size": 2}, {"context_parallel_size": 1})
    with pytest.warns(UserWarning, match="context_parallel_size"):
        _validate_colocated_parallelism(cfg)


def test_weight_sharding_mismatch_raises_before_warning() -> None:
    cfg = _config(
        {"tensor_model_parallel_size": 1, "context_parallel_size": 2},
        {"tensor_model_parallel_size": 2, "context_parallel_size": 1},
    )
    with pytest.raises(ValueError):
        _validate_colocated_parallelism(cfg)


def test_error_names_every_mismatched_key() -> None:
    cfg = _config(
        {"tensor_model_parallel_size": 1, "pipeline_model_parallel_size": 1},
        {"tensor_model_parallel_size": 2, "pipeline_model_parallel_size": 4},
    )
    with pytest.raises(ValueError) as excinfo:
        _validate_colocated_parallelism(cfg)
    message = str(excinfo.value)
    assert "tensor_model_parallel_size: megatron_cfg=1" in message
    assert "pipeline_model_parallel_size: megatron_cfg=1" in message
