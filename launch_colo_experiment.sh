#!/bin/bash
# Launch one side of the colocated vs non-colocated async GRPO experiment
# (see colo_vs_noncolo.txt). Submits an 8-node Slurm job via submit_nemorl.sh;
# the training command runs on the Ray head node (driver log: <jobid>-logs/ray-driver.log).
#
# Usage: SUBMIT_ACCOUNT=<account> bash launch_colo_experiment.sh {colocated|non_colocated}
set -eu

MODE=${1:?usage: bash launch_colo_experiment.sh {colocated|non_colocated}}
case "$MODE" in
  colocated|non_colocated) ;;
  *) echo "invalid mode: $MODE (expected colocated or non_colocated)" >&2; exit 1 ;;
esac

cd "$(dirname "$0")"

export JOB_NAME="async_${MODE}"

# NRL_MEGATRON_CHECKPOINT_DIR caches the one-time HF -> Megatron conversion of
# the initial checkpoint; both modes start from the same model, so they share it.
# Training checkpoints go to checkpointing.checkpoint_dir from the yaml instead.
export COMMAND="mkdir -p /tmp/nemo_rl_triton_cache && \
TRITON_CACHE_DIR=/tmp/nemo_rl_triton_cache \
NETRC=/home/anthomas/.netrc \
NRL_MEGATRON_CHECKPOINT_DIR=/lustre/fsw/portfolios/nemotron/users/anthomas/megatron_ckpt_cache \
NEMO_GYM_VENV_DIR=/lustre/fs1/portfolios/nemotron/projects/nemotron_sw_pre/users/anthomas/gym_venvs \
uv run examples/nemo_gym/run_grpo_nemo_gym.py \
  --config examples/nemo_gym/async_${MODE}_nanov3.yaml \
  policy.generation.backend=megatron"

bash submit_nemorl.sh
