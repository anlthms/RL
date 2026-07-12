#!/bin/bash
# Launch one side of the colocated vs non-colocated async GRPO experiment
# (see colo_vs_noncolo.txt). Submits an 8-node Slurm job via submit_nemorl.sh;
# the training command runs on the Ray head node (driver log: <jobid>-logs/ray-driver.log).
#
# Usage: SUBMIT_ACCOUNT=<account> bash launch_colo_experiment.sh {colocated|non_colocated}
set -eu

MODE=${1:?"usage: bash launch_colo_experiment.sh colocated|non_colocated|colocated_singlemodel|noval_colocated_singlemodel|noval_non_colocated|noval_colocated_continuous|smoke_colocated_continuous"}
case "$MODE" in
  colocated|non_colocated|colocated_singlemodel)
    CONFIG="examples/nemo_gym/async_${MODE}_nanov3.yaml" ;;
  # No-mid-validation training arms for the offline-validation comparison.
  noval_colocated_singlemodel)
    CONFIG="examples/nemo_gym/noval_colocated_singlemodel_nanov3.yaml" ;;
  noval_non_colocated)
    CONFIG="examples/nemo_gym/noval_non_colocated_nanov3.yaml" ;;
  # E4: colocated single-model arm with the continuous rollout scheduler.
  noval_colocated_continuous)
    CONFIG="examples/nemo_gym/noval_colocated_continuous_nanov3.yaml" ;;
  # Smoke-scale validation of the continuous scheduler (run with NUM_ACTOR_NODES=2).
  smoke_colocated_continuous)
    CONFIG="examples/nemo_gym/smoke_colocated_continuous_nanov3.yaml" ;;
  *) echo "invalid mode: $MODE" >&2; exit 1 ;;
esac

cd "$(dirname "$0")"

export JOB_NAME="async_${MODE}"

# NRL_MEGATRON_CHECKPOINT_DIR caches the one-time HF -> Megatron conversion of
# the initial checkpoint; both modes start from the same model, so they share it.
# Training checkpoints go to checkpointing.checkpoint_dir from the yaml instead.
# kv_cache_management_mode=offload (colocated config) requires torch_memory_saver
# in the Megatron policy worker venv; the Jun-24 container's baked venvs predate
# its addition to uv.lock. Installed on every node before Ray starts (compute
# nodes may lack egress, so the wheel is pre-staged on lustre). Harmless for
# non-colocated (persist mode never touches it).
MODEL_PATH="/lustre/fsw/portfolios/llmservice/users/wdykas/data/nano-v3-sft-64gbs-nickel-capybara-5e-5-constant-wd-0-load-bal-1e-4-lcx3-pretool-base-temp1-iter-0013600-hf"
WORKER_PY="/opt/ray_venvs/nemo_rl.models.policy.workers.megatron_policy_worker.MegatronPolicyWorker/bin/python"
# SETUP_COMMAND runs once per node, single-threaded, BEFORE Ray workers start
# (ray.sub). Pre-warming the trust_remote_code dynamic modules here (into the same
# per-node HF_MODULES_CACHE the workers use) fully materializes them before any
# concurrent worker import, eliminating the half-written-module race that caused
# intermittent "NemotronHConfig has no attribute" AutoTokenizer failures. HF's
# module copy is not atomic, so a lock inside the workers would still allow torn
# reads; doing it once, before there is any concurrency, is the robust fix.
export SETUP_COMMAND="uv pip install --python ${WORKER_PY} --no-deps /lustre/fsw/portfolios/nemotron/users/anthomas/wheels/torch_memory_saver-0.0.9.post1-cp39-abi3-manylinux2014_aarch64.whl && \
HF_MODULES_CACHE=/tmp/hf_modules_${MODE} ${WORKER_PY} -c \"from transformers import AutoConfig, AutoTokenizer; m='${MODEL_PATH}'; AutoConfig.from_pretrained(m, trust_remote_code=True); AutoTokenizer.from_pretrained(m, trust_remote_code=True); print('[prewarm] HF trust_remote_code dynamic modules materialized')\""

# HF_MODULES_CACHE is per-node-local: trust_remote_code dynamic modules are
# tiny and rewriting them in a shared cache races across concurrent jobs
# (observed: transformers_modules...configuration_nemotron_h imported while
# half-written -> AttributeError: no attribute 'NemotronHConfig').
export COMMAND="mkdir -p /tmp/nemo_rl_triton_cache && \
TRITON_CACHE_DIR=/tmp/nemo_rl_triton_cache \
HF_MODULES_CACHE=/tmp/hf_modules_${MODE} \
NETRC=/home/anthomas/.netrc \
NRL_MEGATRON_CHECKPOINT_DIR=/lustre/fsw/portfolios/nemotron/users/anthomas/megatron_ckpt_cache \
NEMO_GYM_VENV_DIR=/lustre/fs1/portfolios/nemotron/projects/nemotron_sw_pre/users/anthomas/gym_venvs \
uv run examples/nemo_gym/run_grpo_nemo_gym.py \
  --config ${CONFIG} \
  policy.generation.backend=megatron"

bash submit_nemorl.sh
