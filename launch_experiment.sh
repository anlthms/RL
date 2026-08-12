#!/bin/bash
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
#
# Launch an async GRPO experiment (Megatron inference) via submit_nemorl.sh.
#
# The experiment is three orthogonal axes; every combination has a config at
# examples/configs/async/<model>_<env>_<topology>.yaml.
#
#   MODEL     qwen3_1p7b | nanov3         which policy to train
#   ENV       gym | math | arc | arcsynth NeMo-Gym servers, the built-in math env, real
#                                         ARC-AGI-2, or the synthetic ARC difficulty ladder
#   topology  colocated | non_colocated   share GPUs with generation, or split them
#
# Usage:
#   SUBMIT_ACCOUNT=<account> [MODEL=...] [ENV=...] \
#     bash launch_experiment.sh {colocated|non_colocated}
#
# Common env:
#   NUM_ACTOR_NODES   total nodes (default 8)
#   NUM_GEN_NODES     non-colocated generation nodes (default NUM_ACTOR_NODES/2)
#   RUN_TAG           label appended to the slurm + wandb run name
#   MAX_STEPS         cap grpo.max_num_steps
#   TIMEOUT_MIN       slurm wall clock in minutes (default 240)
#   CMP=1             comparison preset: validation off, periodic weights-only
#                     checkpoints, and a fixed comparison sequence length
#   COLL_TRACE=1      dump NCCL flight-recorder traces to ./nccl_traces on timeout
#   EXTRA_OVERRIDES   ad-hoc Hydra overrides, applied last, e.g. "a=1 b=2"
#
# Driver log: <jobid>-logs/ray-driver.log.
set -euo pipefail

cd "$(dirname "$0")"

die() { echo "$*" >&2; exit 1; }

# ---------------------------------------------------------------- axes -----
TOPOLOGY=${1:-}
case "${TOPOLOGY}" in
  colocated | non_colocated) ;;
  *) die "usage: bash launch_experiment.sh colocated|non_colocated" ;;
esac

MODEL="${MODEL:-qwen3_1p7b}"
case "${MODEL}" in
  qwen3_1p7b | nanov3) ;;
  *) die "invalid MODEL: ${MODEL} (expected qwen3_1p7b or nanov3)" ;;
esac

ENV="${ENV:-gym}"
case "${ENV}" in
  gym | math | arc | arcsynth) ;;
  *) die "invalid ENV: ${ENV} (expected gym, math, arc, or arcsynth)" ;;
esac

export NUM_ACTOR_NODES="${NUM_ACTOR_NODES:-8}"
export TIMEOUT_MIN="${TIMEOUT_MIN:-240}"
USER_NAME="$(whoami)"  # resolve on the host; the container runs as root

# Everything runs on the interpreter the container already ships (3.13.13); the
# branch floats requires-python down to match. Do not point uv at a newer one:
# Ray compares cluster and worker interpreter versions exactly, and Gym passes an
# explicit --python when it builds server venvs, so a partial override desyncs
# them. Unset any inherited value so a stale shell export cannot resplit them.
unset UV_PYTHON

# One name for both slurm and wandb, so a job id always maps to a run.
RUN_NAME="async_${TOPOLOGY}_${MODEL}_${ENV}${RUN_TAG:+_${RUN_TAG}}"
export JOB_NAME="${RUN_NAME}"

# ------------------------------------------------- override accumulator ----
# Layers are appended in order; the last assignment of a key wins in Hydra.
OVERRIDES=""
add_override() { OVERRIDES+=" $*"; }

SETUP_PARTS=()
add_setup() { SETUP_PARTS+=("$*"); }

# Layer 1: base.
add_override "policy.generation.backend=megatron"
add_override "cluster.num_nodes=${NUM_ACTOR_NODES}"
add_override "logger.wandb.name=${RUN_NAME}"
add_override "+logger.wandb.entity=${WANDB_ENTITY:-adlr}"
# Non-colocated reshards weights over the network: nvshmem is proven stable at
# <=4 generation nodes, while nccl hangs the reshard there. Colocated shares
# weights in-process (CUDA-IPC), so the backend is a no-op.
add_override "policy.generation.mcore_generation_config.refit_backend=${REFIT_BACKEND:-nvshmem}"
[[ -n "${WANDB_PROJECT:-}" ]] && add_override "logger.wandb.project=${WANDB_PROJECT}"
[[ -n "${MAX_STEPS:-}" ]] && add_override "grpo.max_num_steps=${MAX_STEPS}"
# Retry imports on each node: lustre loses races resolving PEP-420 namespaces.
add_setup "bash tools/retry_import.sh nemo_rl.algorithms.grpo"

# Layer 2: topology.
if [[ "${TOPOLOGY}" == non_colocated ]]; then
  # Symmetric train/gen split; asymmetric splits hang the reshard.
  NUM_GEN_NODES="${NUM_GEN_NODES:-$((NUM_ACTOR_NODES / 2))}"
  add_override "policy.generation.colocated.resources.num_nodes=${NUM_GEN_NODES}"
else
  # Every colocated arm offloads the KV cache during the training pause. mcore
  # asserts torch_memory_saver (or UVM) for any non-persist cache mode, and the
  # wheel predates the container uv.lock, so install it in the worker venv.
  add_setup "uv pip install --python /opt/ray_venvs/nemo_rl.models.policy.workers.megatron_policy_worker.MegatronPolicyWorker/bin/python --no-deps /lustre/fsw/portfolios/nemotron/users/anthomas/wheels/torch_memory_saver-0.0.9.post1-cp39-abi3-manylinux2014_aarch64.whl"
fi

# Layer 3: CMP preset -- validation off, weights-only checkpoints, fixed seq.
if [[ "${CMP:-0}" == 1 ]]; then
  # The preset's fixed 16384 is applied after the config layer and so wins over
  # env_arc*.yaml's 32768. The worst real ARC evaluation prompt is 16593 tokens,
  # and an overlong prompt is not an error -- the processor masks the row with
  # loss_multiplier=0 -- so the combination degrades the run silently. Refuse it
  # rather than leaving it as a comment nobody reads at 2am.
  case "${ENV}" in
    arc*) die "CMP=1 forces max_total_sequence_length=16384, below the 16593-token worst-case ARC prompt; do not combine it with ENV=${ENV}" ;;
  esac
  add_override "grpo.val_period=0 grpo.val_at_start=false grpo.val_at_end=false"
  add_override "checkpointing.enabled=true checkpointing.save_period=5 checkpointing.save_optimizer=false"
  add_override "policy.max_total_sequence_length=16384"
fi

# Layer 4: caller overrides, last so they win.
[[ -n "${EXTRA_OVERRIDES:-}" ]] && add_override "${EXTRA_OVERRIDES}"

# ------------------------------------------- entrypoint, config, setup -----
# The gym path drives rollouts through NeMo-Gym HTTP servers and needs its own
# entrypoint; math runs on the standard one.
case "${ENV}" in
  gym)  ENTRYPOINT="examples/nemo_gym/run_grpo_nemo_gym.py" ;;
  math | arc | arcsynth) ENTRYPOINT="examples/run_grpo.py" ;;
esac
CONFIG="examples/configs/async/${MODEL}_${ENV}_${TOPOLOGY}.yaml"
[[ -f "${CONFIG}" ]] || die "no config for ${MODEL} x ${ENV} x ${TOPOLOGY}: ${CONFIG}"

printf -v SETUP_JOINED '%s && ' "${SETUP_PARTS[@]}"
export SETUP_COMMAND="${SETUP_COMMAND:+${SETUP_COMMAND} && }${SETUP_JOINED% && }"

# COLL_TRACE=1 enables torch's NCCL flight recorder: dumps each rank's last
# collectives to ./nccl_traces/ on a watchdog timeout (shows the stuck collective).
COLL_TRACE_ENV=""
if [[ "${COLL_TRACE:-0}" == 1 ]]; then
  mkdir -p "$(pwd)/nccl_traces"
  COLL_TRACE_ENV="TORCH_NCCL_TRACE_BUFFER_SIZE=${TORCH_NCCL_TRACE_BUFFER_SIZE:-20000} TORCH_NCCL_DUMP_ON_TIMEOUT=1 TORCH_NCCL_DEBUG_INFO_TEMP_FILE=$(pwd)/nccl_traces/trace_"
fi

# NRL_MEGATRON_CHECKPOINT_DIR caches the one-time HF -> Megatron conversion.
# NCCL_GRAPH_MIXING_SUPPORT=1 lets NCCL safely mix graph-captured decode collectives
# with eager training collectives on the shared colocated communicator.
# HF_HOME must be on shared storage: OpenMathInstruct-2's mmap'd Arrow cache is
# pickled to a collector on another node, so node-local /root/.cache would miss.
# NEMO_GYM_VENV_DIR is assigned inline rather than exported: the image sets it in
# its own ENV, which enroot applies over anything inherited from the job. It also
# points at gym_venvs_main rather than the original gym_venvs, whose servers pin
# Ray 2.55.1 and are refused by main's 2.56.1 cluster.
export COMMAND="mkdir -p /tmp/nemo_rl_triton_cache && \
TRITON_CACHE_DIR=/tmp/nemo_rl_triton_cache \
NETRC=/home/${USER_NAME}/.netrc \
NCCL_GRAPH_MIXING_SUPPORT=${NCCL_GRAPH_MIXING_SUPPORT:-1} \
NCCL_DEBUG=${NCCL_DEBUG:-} \
NCCL_DEBUG_SUBSYS=${NCCL_DEBUG_SUBSYS:-} \
NRL_MEGATRON_CHECKPOINT_DIR=/lustre/fsw/portfolios/nemotron/users/${USER_NAME}/megatron_ckpt_cache \
NEMO_GYM_VENV_DIR=${NEMO_GYM_VENV_DIR:-/lustre/fs1/portfolios/nemotron/projects/nemotron_sw_pre/users/${USER_NAME}/gym_venvs_main} \
HF_HOME=${HF_HOME:-/lustre/fsw/portfolios/nemotron/users/${USER_NAME}/hf_home} \
${COLL_TRACE_ENV} \
uv run ${ENTRYPOINT} \
  --config ${CONFIG} \
    ${OVERRIDES}"

echo "run:      ${RUN_NAME}"
echo "config:   ${CONFIG}"
echo "nodes:    ${NUM_ACTOR_NODES}${NUM_GEN_NODES:+ (${NUM_GEN_NODES} generation)}, ${TIMEOUT_MIN} min"

bash submit_nemorl.sh
