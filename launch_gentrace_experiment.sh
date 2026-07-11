#!/bin/bash
# Launch a generation-concurrency profiling run (validate-at-start, small val set,
# NRL_GEN_TRACE=1) for either the colocated single-model or the non-colocated arm.
# Purpose: compare why colocated validation generation runs at lower effective
# concurrency than non-colocated. Kill the job once the [NRL-GEN-TRACE] lines
# during validation have stabilized.
#
# Usage: SUBMIT_ACCOUNT=<account> bash launch_gentrace_experiment.sh {colocated|noncolocated}
set -eu

MODE=${1:?"usage: bash launch_gentrace_experiment.sh colocated|noncolocated|colocated2n"}
case "$MODE" in
  colocated)     CONFIG=examples/nemo_gym/gentrace_colocated_nanov3.yaml ;;
  noncolocated)  CONFIG=examples/nemo_gym/gentrace_noncolocated_nanov3.yaml ;;
  # 2-node (single replica) fast-iteration variant; caller should also export
  # NUM_ACTOR_NODES=2 so the Slurm allocation matches cluster.num_nodes.
  colocated2n)   CONFIG=examples/nemo_gym/gentrace_colocated_2n_nanov3.yaml ;;
  *) echo "invalid mode: $MODE (expected colocated, noncolocated, or colocated2n)" >&2; exit 1 ;;
esac

cd "$(dirname "$0")"

export JOB_NAME="gentrace_${MODE}"

# Same torch_memory_saver wheel preinstall as the production launcher (needed by
# the colocated kv_cache offload path; harmless for non-colocated persist mode).
export SETUP_COMMAND="uv pip install --python /opt/ray_venvs/nemo_rl.models.policy.workers.megatron_policy_worker.MegatronPolicyWorker/bin/python --no-deps /lustre/fsw/portfolios/nemotron/users/anthomas/wheels/torch_memory_saver-0.0.9.post1-cp39-abi3-manylinux2014_aarch64.whl"

# NRL_GEN_TRACE=1 activates the opt-in per-step concurrency trace wired into the
# Megatron DynamicInferenceEngine from megatron_worker.py.
export COMMAND="mkdir -p /tmp/nemo_rl_triton_cache && \
TRITON_CACHE_DIR=/tmp/nemo_rl_triton_cache \
HF_MODULES_CACHE=/tmp/hf_modules_gentrace_${MODE} \
NETRC=/home/anthomas/.netrc \
NRL_GEN_TRACE=1 \
NRL_GEN_TRACE_EVERY=20 \
NRL_MEGATRON_CHECKPOINT_DIR=/lustre/fsw/portfolios/nemotron/users/anthomas/megatron_ckpt_cache \
NEMO_GYM_VENV_DIR=/lustre/fs1/portfolios/nemotron/projects/nemotron_sw_pre/users/anthomas/gym_venvs \
uv run examples/nemo_gym/run_grpo_nemo_gym.py \
  --config ${CONFIG} \
  policy.generation.backend=megatron"

bash submit_nemorl.sh
