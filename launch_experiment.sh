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
# Each supported environment has a config at
# examples/configs/async/nanov3_<env>_<topology>.yaml.
#
#   ENV       rl_blend                    RL-blend dataset over NeMo-Gym servers
#             nvarc_executor_4n           NVARC-description executor training
#   topology  colocated | non_colocated   share GPUs with generation, or split them
#
# Usage:
#   SUBMIT_ACCOUNT=<account> [ENV=...] \
#     bash launch_experiment.sh {colocated|non_colocated}
#
# Common env:
#   NUM_ACTOR_NODES   total nodes (default 8)
#   NUM_GEN_NODES     non-colocated generation nodes (default NUM_ACTOR_NODES/2)
#   RUN_TAG           label appended to the slurm + wandb run name
#   MAX_STEPS         cap grpo.max_num_steps
#   TIMEOUT_MIN       slurm wall clock in minutes (default 240)
#   COLL_TRACE=0      disable the NCCL flight recorder (on by default; dumps each
#                     rank's recent collectives to ./nccl_traces on a timeout)
#   WATCHDOG=0        do not arm the GPU-idleness watchdog (armed by default)
#   WATCHDOG_INTERVAL seconds between watchdog polls (default 300)
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

ENV="${ENV:-rl_blend}"
case "${ENV}" in
  rl_blend | nvarc_executor_4n) ;;
  *) die "invalid ENV: ${ENV} (expected rl_blend or nvarc_executor_4n)" ;;
esac
if [[ "${ENV}" == nvarc_executor_4n && "${TOPOLOGY}" != colocated ]]; then
  die "${ENV} supports only colocated topology"
fi

export NUM_ACTOR_NODES="${NUM_ACTOR_NODES:-8}"
export TIMEOUT_MIN="${TIMEOUT_MIN:-240}"
USER_NAME="$(whoami)"  # resolve on the host; the container runs as root

# The image ships the interpreter this checkout's requires-python resolves to.
export CONTAINER="${CONTAINER:-/lustre/fs1/portfolios/llmservice/projects/llmservice_fm_text/users/tene/nemo_rl_0807.sqsh}"

# One name for both slurm and wandb, so a job id always maps to a run.
RUN_NAME="async_${TOPOLOGY}_${ENV}${RUN_TAG:+_${RUN_TAG}}"
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

# Layer 3: caller overrides, last so they win. Newlines are folded to spaces:
# ray.sub runs ${COMMAND} verbatim with bash, so an embedded newline would
# silently truncate the command there.
if [[ -n "${EXTRA_OVERRIDES:-}" ]]; then
  add_override "$(tr '\n' ' ' <<<"${EXTRA_OVERRIDES}")"
fi

# ------------------------------------------- entrypoint, config, setup -----
# The RL-blend path drives rollouts through NeMo-Gym HTTP servers and needs its
# own entrypoint; everything else runs on the standard one.
case "${ENV}" in
  rl_blend) ENTRYPOINT="examples/nemo_gym/run_grpo_nemo_gym.py" ;;
  *)        ENTRYPOINT="examples/run_grpo.py" ;;
esac
CONFIG="examples/configs/async/nanov3_${ENV}_${TOPOLOGY}.yaml"
[[ -f "${CONFIG}" ]] || die "no config for ${ENV} x ${TOPOLOGY}: ${CONFIG}"

# Only the colocated arm adds a setup step, so the array can be empty here.
if ((${#SETUP_PARTS[@]})); then
  printf -v SETUP_JOINED '%s && ' "${SETUP_PARTS[@]}"
  export SETUP_COMMAND="${SETUP_COMMAND:+${SETUP_COMMAND} && }${SETUP_JOINED% && }"
fi

# torch's NCCL flight recorder, on by default (COLL_TRACE=0 disables): dumps
# each rank's recent collectives to ./nccl_traces/ on a collective timeout,
# naming the stuck collective. Costs only a fixed per-rank ring buffer; nothing
# is written unless a timeout fires.
COLL_TRACE_ENV=""
if [[ "${COLL_TRACE:-1}" == 1 ]]; then
  mkdir -p "$(pwd)/nccl_traces"
  COLL_TRACE_ENV="TORCH_NCCL_TRACE_BUFFER_SIZE=${TORCH_NCCL_TRACE_BUFFER_SIZE:-20000} TORCH_NCCL_DUMP_ON_TIMEOUT=1 TORCH_NCCL_DEBUG_INFO_TEMP_FILE=$(pwd)/nccl_traces/trace_"
fi

# NRL_MEGATRON_CHECKPOINT_DIR caches the one-time HF -> Megatron conversion.
# NCCL_GRAPH_MIXING_SUPPORT=1 lets NCCL safely mix graph-captured decode collectives
# with eager training collectives on the shared colocated communicator.
# HF_HOME must be on shared storage: HF Arrow caches are mmap'd and can be
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

# Refuse to submit a command that would be truncated at a newline (see above).
[[ "${COMMAND}" == *$'\n'* ]] && die "driver command contains a newline; it would be truncated there by ray.sub"

echo "run:      ${RUN_NAME}"
echo "config:   ${CONFIG}"
echo "nodes:    ${NUM_ACTOR_NODES}${NUM_GEN_NODES:+ (${NUM_GEN_NODES} generation)}, ${TIMEOUT_MIN} min"

# Capture (not tee) the submission output so the job id can be parsed: tee
# needs a terminal, which nohup/cron lack, and its failure under `set -e`
# would abort after sbatch already queued the job.
if ! SUBMIT_OUTPUT="$(bash submit_nemorl.sh 2>&1)"; then
  echo "${SUBMIT_OUTPUT}" >&2
  die "submission failed"
fi
echo "${SUBMIT_OUTPUT}"
JOB_ID="$(grep -oE 'Submitted batch job [0-9]+' <<<"${SUBMIT_OUTPUT}" | grep -oE '[0-9]+$' | tail -1)"

# ------------------------------------------------------------- watchdog -----
# Armed by default (WATCHDOG=0 disables): kills a job whose GPUs went idle,
# capturing per-rank stacks and GPU state first -- the cluster reaper would
# take hours longer and destroy that evidence. Runs on the submit host (it
# only shells out to sacct/srun/scancel); setsid+nohup outlives this shell.
if [[ -n "${JOB_ID}" && "${WATCHDOG:-1}" != 0 ]]; then
  WATCHDOG_LOG="${JOB_ID}-watchdog.log"
  setsid nohup python3 tools/run_watchdog.py "${JOB_ID}" \
    --root "$(pwd)" \
    --log-dir "$(pwd)/${JOB_ID}-logs" \
    --interval "${WATCHDOG_INTERVAL:-300}" \
    >"${WATCHDOG_LOG}" 2>&1 < /dev/null &
  echo "watchdog: armed on ${JOB_ID} (pid $!), log ${WATCHDOG_LOG}"
elif [[ -z "${JOB_ID}" ]]; then
  echo "watchdog: NOT armed -- could not parse a job id from the submission" >&2
fi
