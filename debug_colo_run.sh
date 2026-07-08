#!/bin/bash
# Interactive debug driver for the colocated first-training-pause NCCL hang.
# See async_colo_debug_guide.md §6 for the full plan.
#
# Usage:
#   1. From the repo root, get an idle cluster (no COMMAND -> interactive):
#        JOB_NAME=colo_debug SUBMIT_ACCOUNT=nemotron_sw_pre bash submit_nemorl.sh
#      (Consider raising --time in submit_nemorl.sh to 8:0:0 for the session.)
#   2. Wait ~5 min for <jobid>-attach.sh, then:  bash <jobid>-attach.sh
#   3. Inside the attach shell:  bash debug_colo_run.sh [extra hydra overrides]
#
# Probe suggestions (append as overrides, one per run — see guide §6.5):
#   policy.generation.mcore_generation_config.cuda_graph_impl=none   # kills suspects S2+S3
#   policy.megatron_cfg.inference_moe_token_dispatcher_type=alltoall # tests S1
set -eu
cd "$(dirname "$0")"

NCCL_TRACE_DIR=/lustre/fsw/portfolios/nemotron/users/anthomas/nccl_traces/$(date +%Y%m%d_%H%M%S)
mkdir -p "$NCCL_TRACE_DIR"
echo "Flight-recorder dumps will land in: $NCCL_TRACE_DIR"

mkdir -p /tmp/nemo_rl_triton_cache

# PyTorch NCCL flight recorder: on watchdog timeout each rank dumps its recent
# collectives (op, sizes, seq, state) -> directly identifies which rank is at
# which point in the collective schedule. Env propagates to all Ray workers
# via nemo_rl/distributed/virtual_cluster.py:197.
export TORCH_NCCL_TRACE_BUFFER_SIZE=20000
export TORCH_NCCL_DUMP_ON_TIMEOUT=1
export TORCH_NCCL_DEBUG_INFO_TEMP_FILE="$NCCL_TRACE_DIR/rank_"
# Fail in 5 min instead of 10 while iterating:
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=300

TRITON_CACHE_DIR=/tmp/nemo_rl_triton_cache \
NETRC=/home/anthomas/.netrc \
HF_MODULES_CACHE=/tmp/hf_modules_debug \
NRL_MEGATRON_CHECKPOINT_DIR=/lustre/fsw/portfolios/nemotron/users/anthomas/megatron_ckpt_cache \
NEMO_GYM_VENV_DIR=/lustre/fs1/portfolios/nemotron/projects/nemotron_sw_pre/users/anthomas/gym_venvs \
uv run examples/nemo_gym/run_grpo_nemo_gym.py \
  --config examples/nemo_gym/async_colocated_debug_nanov3.yaml \
  policy.generation.backend=megatron \
  "$@"
