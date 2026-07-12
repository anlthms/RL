#!/bin/bash
# Offline validation harness for the colo-vs-noncolo comparison. For each saved
# checkpoint step_N in a training run's checkpoint dir, validate it (dedicated
# non-colocated inference) out of the critical path and record (step, reward).
#
# Usage:
#   SUBMIT_ACCOUNT=<acct> bash run_offline_validation.sh <checkpoint_dir> [arm_label]
# e.g.
#   SUBMIT_ACCOUNT=nemotron_sw_post bash run_offline_validation.sh \
#     /lustre/fsw/portfolios/nemotron/users/anthomas/checkpoints/noval_colocated_singlemodel colo
#
# Produces one Slurm job per checkpoint. Extract rewards afterwards with:
#   grep -h "validation.*accuracy\|Avg Reward\|Accuracy" <jobid>-logs/ray-driver.log
set -eu

CKPT_DIR=${1:?"usage: run_offline_validation.sh <checkpoint_dir> [arm_label]"}
ARM=${2:-$(basename "$CKPT_DIR")}
cd "$(dirname "$0")"

# Views live under a scratch area so we can present exactly one step_N per job
# (grpo resumes the LATEST step_* in the dir; symlink a single step to isolate it).
VIEW_ROOT="/lustre/fsw/portfolios/nemotron/users/anthomas/offline_val_views/${ARM}"
mkdir -p "$VIEW_ROOT"

steps=$(ls -d "$CKPT_DIR"/step_* 2>/dev/null | sed -E 's/.*step_//' | sort -n)
if [ -z "$steps" ]; then echo "no step_* checkpoints in $CKPT_DIR yet" >&2; exit 1; fi
echo "Validating checkpoints for arm=$ARM: steps = $(echo $steps | tr '\n' ' ')"

for N in $steps; do
  VIEW="$VIEW_ROOT/step_${N}_view"
  rm -rf "$VIEW"; mkdir -p "$VIEW"
  ln -s "$CKPT_DIR/step_${N}" "$VIEW/step_${N}"

  export NUM_ACTOR_NODES=2
  export JOB_NAME="offval_${ARM}_step${N}"
  export SETUP_COMMAND="uv pip install --python /opt/ray_venvs/nemo_rl.models.policy.workers.megatron_policy_worker.MegatronPolicyWorker/bin/python --no-deps /lustre/fsw/portfolios/nemotron/users/anthomas/wheels/torch_memory_saver-0.0.9.post1-cp39-abi3-manylinux2014_aarch64.whl"
  export COMMAND="mkdir -p /tmp/nemo_rl_triton_cache && \
TRITON_CACHE_DIR=/tmp/nemo_rl_triton_cache \
HF_MODULES_CACHE=/tmp/hf_modules_offval_${ARM}_${N} \
NETRC=/home/anthomas/.netrc \
NRL_MEGATRON_CHECKPOINT_DIR=/lustre/fsw/portfolios/nemotron/users/anthomas/megatron_ckpt_cache \
NEMO_GYM_VENV_DIR=/lustre/fs1/portfolios/nemotron/projects/nemotron_sw_pre/users/anthomas/gym_venvs \
uv run examples/nemo_gym/run_grpo_nemo_gym.py \
  --config examples/nemo_gym/offline_validate_nanov3.yaml \
  policy.generation.backend=megatron \
  checkpointing.checkpoint_dir=${VIEW} \
  logger.wandb.name=offval_${ARM}_step${N}"

  echo "submitting offline validation: arm=$ARM step=$N (view=$VIEW)"
  bash submit_nemorl.sh
done
