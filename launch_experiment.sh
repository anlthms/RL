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
# The experiment is two orthogonal axes; every combination has a config at
# examples/configs/async/nanov3_<env>_<topology>.yaml.
#
#   ENV       gym                         NeMo-Gym servers
#             arcsynth[_4n|_8n]           the synthetic ARC difficulty ladder, generic or
#                                         sized for a specific allocation
#             arcsynth_early_answer       that ladder with the early-answer prompt
#   topology  colocated | non_colocated   share GPUs with generation, or split them
#
# The real-ARC-only (`arc`) and math arms are gone. Real ARC-AGI-2 is still
# validated on at every checkpoint -- it is the campaign metric -- but as a
# validation source inside env_arc_synth.yaml, not a training arm of its own.
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
#   CMP=1             comparison preset: validation off, periodic weights-only
#                     checkpoints, and a fixed comparison sequence length
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

ENV="${ENV:-gym}"
case "${ENV}" in
  gym | arcsynth | arcsynth_4n | arcsynth_8n | arcsynth_early_answer) ;;
  *) die "invalid ENV: ${ENV} (expected gym, arcsynth, arcsynth_{4n,8n}, or arcsynth_early_answer)" ;;
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
  # env_arc_synth.yaml's 32768. The synthetic prompts are smaller, but every
  # checkpoint still validates on the real ARC-AGI-2 split, whose worst row is
  # 16593 tokens -- and an overlong prompt is not an error, the processor masks
  # the row with loss_multiplier=0, so the combination degrades the run
  # silently. Refuse it rather than leaving it as a comment nobody reads at 2am.
  case "${ENV}" in
    arcsynth*) die "CMP=1 forces max_total_sequence_length=16384, below the 16593-token worst-case real ARC validation prompt; do not combine it with ENV=${ENV}" ;;
  esac
  add_override "grpo.val_period=0 grpo.val_at_start=false grpo.val_at_end=false"
  add_override "checkpointing.enabled=true checkpointing.save_period=5 checkpointing.save_optimizer=false"
  add_override "policy.max_total_sequence_length=16384"
fi

# Layer 4: caller overrides, last so they win.
#
# Fold any newline in the caller's value to a space first. ray.sub writes
# ${COMMAND} to driver_command.sh verbatim and runs it with `bash`, so an
# embedded newline does not continue the command -- it *ends* it, and everything
# after becomes a second shell command that never reaches the entrypoint.
# A run once lost `grpo.val_period=20` exactly this way (a multi-line
# EXTRA_OVERRIDES pasted without continuations) and validated every 10 steps
# while its log claimed 20. Silent, and only visible by diffing the resolved
# config against the submit command.
if [[ -n "${EXTRA_OVERRIDES:-}" ]]; then
  add_override "$(tr '\n' ' ' <<<"${EXTRA_OVERRIDES}")"
fi

# ------------------------------------------- entrypoint, config, setup -----
# The gym path drives rollouts through NeMo-Gym HTTP servers and needs its own
# entrypoint; everything else runs on the standard one.
case "${ENV}" in
  gym) ENTRYPOINT="examples/nemo_gym/run_grpo_nemo_gym.py" ;;
  *)   ENTRYPOINT="examples/run_grpo.py" ;;
esac
CONFIG="examples/configs/async/nanov3_${ENV}_${TOPOLOGY}.yaml"
[[ -f "${CONFIG}" ]] || die "no config for ${ENV} x ${TOPOLOGY}: ${CONFIG}"

printf -v SETUP_JOINED '%s && ' "${SETUP_PARTS[@]}"
export SETUP_COMMAND="${SETUP_COMMAND:+${SETUP_COMMAND} && }${SETUP_JOINED% && }"

# torch's NCCL flight recorder, on by default: a ring buffer of each rank's
# recent collectives, dumped to ./nccl_traces/ when a collective times out. It
# names the stuck collective, which is the one thing a hang post-mortem needs
# and the one thing that cannot be recovered afterwards -- a hang caught without
# it can only ever be localized to a phase, never explained.
#
# Default-on because the cost is a fixed per-rank buffer (20k entries) and a
# pointer bump per collective; nothing is written unless a timeout fires. Set
# COLL_TRACE=0 to disable.
#
# NCCL_DEBUG is deliberately left at the driver's default rather than forced to
# INFO: INFO prints per-communicator setup for every rank and would bury the
# driver log on a large job. Raise it by hand (NCCL_DEBUG=INFO) when reproducing
# a known hang, where the verbosity is the point.
COLL_TRACE_ENV=""
if [[ "${COLL_TRACE:-1}" == 1 ]]; then
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

# Belt and braces on the above: whatever assembled it, a newline anywhere in the
# driver command truncates it at that point. Refuse to submit rather than run a
# job whose overrides are quietly half-applied.
[[ "${COMMAND}" == *$'\n'* ]] && die "driver command contains a newline; it would be truncated there by ray.sub"

echo "run:      ${RUN_NAME}"
echo "config:   ${CONFIG}"
echo "nodes:    ${NUM_ACTOR_NODES}${NUM_GEN_NODES:+ (${NUM_GEN_NODES} generation)}, ${TIMEOUT_MIN} min"

# Capture the job id so the watchdog can be armed against it. Captured rather
# than tee'd: there is no controlling terminal under nohup, cron or CI, and
# `tee /dev/tty` fails there -- which under `set -e` aborts the launcher *after*
# sbatch has already queued the job, leaving it running with no watchdog.
if ! SUBMIT_OUTPUT="$(bash submit_nemorl.sh 2>&1)"; then
  echo "${SUBMIT_OUTPUT}" >&2
  die "submission failed"
fi
echo "${SUBMIT_OUTPUT}"
JOB_ID="$(grep -oE 'Submitted batch job [0-9]+' <<<"${SUBMIT_OUTPUT}" | grep -oE '[0-9]+$' | tail -1)"

# ------------------------------------------------------------- watchdog -----
# Armed by default. A hung job holds its whole allocation until the cluster's
# idle-GPU reaper collects it -- hours of GPUs producing nothing, and the reaper
# takes the process state with it, so the hang cannot be explained afterwards.
# The watchdog kills it far sooner and snapshots the evidence first.
#
# It runs here on the submit host, not as a Slurm job: it only shells out to
# sacct/srun/scancel, so it needs the Slurm client and nothing else. Queueing a
# CPU job to run `sleep` would be slower to start and no more reliable.
# setsid+nohup so it outlives this shell and the ssh session that started it.
#
# WATCHDOG=0 disables it.
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
