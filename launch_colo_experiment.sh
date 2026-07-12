#!/bin/bash
# Launch one side of the colocated vs non-colocated async GRPO experiment
# (see colo_vs_noncolo.txt). Submits an 8-node Slurm job via submit_nemorl.sh;
# the training command runs on the Ray head node (driver log: <jobid>-logs/ray-driver.log).
#
# Usage: SUBMIT_ACCOUNT=<account> bash launch_colo_experiment.sh {colocated|non_colocated}
set -eu

MODE=${1:?"usage: bash launch_colo_experiment.sh colocated|non_colocated|colocated_singlemodel|noval_colocated_singlemodel|noval_non_colocated"}
case "$MODE" in
  colocated|non_colocated|colocated_singlemodel)
    CONFIG="examples/nemo_gym/async_${MODE}_nanov3.yaml" ;;
  # No-mid-validation training arms for the offline-validation comparison.
  noval_colocated_singlemodel)
    CONFIG="examples/nemo_gym/noval_colocated_singlemodel_nanov3.yaml" ;;
  noval_non_colocated)
    CONFIG="examples/nemo_gym/noval_non_colocated_nanov3.yaml" ;;
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
export SETUP_COMMAND="uv pip install --python /opt/ray_venvs/nemo_rl.models.policy.workers.megatron_policy_worker.MegatronPolicyWorker/bin/python --no-deps /lustre/fsw/portfolios/nemotron/users/anthomas/wheels/torch_memory_saver-0.0.9.post1-cp39-abi3-manylinux2014_aarch64.whl"

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
