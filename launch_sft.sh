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
# Launch an SFT run via submit_nemorl.sh (sibling of launch_experiment.sh,
# which is GRPO-specific: its base overrides target generation config keys
# that do not exist in the SFT schema).
#
# Usage:
#   SUBMIT_ACCOUNT=<account> CONFIG=examples/configs/<recipe>.yaml \
#     [RUN_TAG=...] [NUM_ACTOR_NODES=4] [MAX_STEPS=...] [EXTRA_OVERRIDES=...] \
#     bash launch_sft.sh
#
# Driver log: <jobid>-logs/ray-driver.log.
set -euo pipefail

cd "$(dirname "$0")"

die() { echo "$*" >&2; exit 1; }

CONFIG="${CONFIG:-}"
[[ -f "${CONFIG}" ]] || die "CONFIG must point at an SFT recipe yaml (got '${CONFIG}')"

export NUM_ACTOR_NODES="${NUM_ACTOR_NODES:-4}"
export TIMEOUT_MIN="${TIMEOUT_MIN:-240}"
USER_NAME="$(whoami)"

export CONTAINER="${CONTAINER:-/lustre/fs1/portfolios/llmservice/projects/llmservice_fm_text/users/tene/nemo_rl_0807.sqsh}"

RUN_NAME="$(basename "${CONFIG}" .yaml)${RUN_TAG:+_${RUN_TAG}}"
export JOB_NAME="${RUN_NAME}"

OVERRIDES=""
add_override() { OVERRIDES+=" $*"; }
add_override "cluster.num_nodes=${NUM_ACTOR_NODES}"
add_override "logger.wandb.name=${RUN_NAME}"
add_override "+logger.wandb.entity=${WANDB_ENTITY:-adlr}"
[[ -n "${WANDB_PROJECT:-}" ]] && add_override "logger.wandb.project=${WANDB_PROJECT}"
[[ -n "${MAX_STEPS:-}" ]] && add_override "sft.max_num_steps=${MAX_STEPS}"
if [[ -n "${EXTRA_OVERRIDES:-}" ]]; then
  add_override "$(tr '\n' ' ' <<<"${EXTRA_OVERRIDES}")"
fi

# Same shared-storage cache rationale as launch_experiment.sh.
export COMMAND="mkdir -p /tmp/nemo_rl_triton_cache && \
TRITON_CACHE_DIR=/tmp/nemo_rl_triton_cache \
NETRC=/home/${USER_NAME}/.netrc \
NRL_MEGATRON_CHECKPOINT_DIR=/lustre/fsw/portfolios/nemotron/users/${USER_NAME}/megatron_ckpt_cache \
HF_HOME=${HF_HOME:-/lustre/fsw/portfolios/nemotron/users/${USER_NAME}/hf_home} \
uv run examples/run_sft.py \
  --config ${CONFIG} \
    ${OVERRIDES}"

[[ "${COMMAND}" == *$'\n'* ]] && die "driver command contains a newline; it would be truncated by ray.sub"

echo "run:      ${RUN_NAME}"
echo "config:   ${CONFIG}"
echo "nodes:    ${NUM_ACTOR_NODES}, ${TIMEOUT_MIN} min"

if ! SUBMIT_OUTPUT="$(bash submit_nemorl.sh 2>&1)"; then
  echo "${SUBMIT_OUTPUT}" >&2
  die "submission failed"
fi
echo "${SUBMIT_OUTPUT}"
JOB_ID="$(grep -oE 'Submitted batch job [0-9]+' <<<"${SUBMIT_OUTPUT}" | grep -oE '[0-9]+$' | tail -1)"

# Same watchdog rationale as launch_experiment.sh.
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
