#!/bin/bash
# Launch an async GRPO experiment (Megatron inference) via submit_nemorl.sh.
# Both modes share one override set (apples-to-apples); only non-colocated adds a
# symmetric generation split. Driver log: <jobid>-logs/ray-driver.log.
#
# Usage: SUBMIT_ACCOUNT=<account> bash launch_experiment.sh {colocated|non_colocated} [--smoke]
# Env: NUM_ACTOR_NODES (default 8 full / 4 smoke); NUM_GEN_NODES (non-colo, default NUM_ACTOR_NODES/2).
set -eu

cd "$(dirname "$0")"

MODE=${1:?"usage: bash launch_experiment.sh colocated|non_colocated [--smoke]"}
case "${MODE}" in
  colocated | non_colocated) ;;
  *) echo "invalid mode: ${MODE} (expected colocated or non_colocated)" >&2; exit 1 ;;
esac

SMOKE=0
[[ "${2:-}" == "--smoke" ]] && SMOKE=1

if [[ "${SMOKE}" == 1 ]]; then
  export NUM_ACTOR_NODES="${NUM_ACTOR_NODES:-4}"
else
  export NUM_ACTOR_NODES="${NUM_ACTOR_NODES:-8}"
fi
# Symmetric train/gen split; asymmetric splits hang the reshard.
NUM_GEN_NODES="${NUM_GEN_NODES:-$((NUM_ACTOR_NODES / 2))}"

export JOB_NAME="async_${MODE}$([[ "${SMOKE}" == 1 ]] && echo _smoke)"

# Resolve on the host (the container runs as root, so not inside COMMAND).
USER_NAME="$(whoami)"

# Common overrides. refit_backend=nccl is a no-op for colocated.
overrides="policy.generation.backend=megatron"
overrides+=" cluster.num_nodes=${NUM_ACTOR_NODES}"
overrides+=" policy.generation.mcore_generation_config.refit_backend=nccl"

# Mode-specific knobs. Colocated needs no extra overrides: it builds one
# inference_optimized dual-mode model (see async_colocated_nanov3.yaml).
if [[ "${MODE}" == non_colocated ]]; then
  overrides+=" policy.generation.colocated.resources.num_nodes=${NUM_GEN_NODES}"
fi

# Scale-down: one step, no checkpoint, shorter seq (the OOM lever), smaller batch.
if [[ "${SMOKE}" == 1 ]]; then
  overrides+=" grpo.max_num_steps=1 checkpointing.enabled=false"
  overrides+=" policy.max_total_sequence_length=16384"
  overrides+=" grpo.num_prompts_per_step=8 policy.train_global_batch_size=128"
  overrides+=" logger.wandb.name=async_${MODE}_smoke"
fi

# Ad-hoc extra Hydra overrides (space-separated), e.g. EXTRA_OVERRIDES="a=1 b=2".
overrides+=" ${EXTRA_OVERRIDES:-}"

# NRL_MEGATRON_CHECKPOINT_DIR caches the one-time HF -> Megatron conversion (shared by both modes).
export COMMAND="mkdir -p /tmp/nemo_rl_triton_cache && \
TRITON_CACHE_DIR=/tmp/nemo_rl_triton_cache \
NETRC=/home/${USER_NAME}/.netrc \
NRL_MEGATRON_CHECKPOINT_DIR=/lustre/fsw/portfolios/nemotron/users/${USER_NAME}/megatron_ckpt_cache \
NEMO_GYM_VENV_DIR=/lustre/fs1/portfolios/nemotron/projects/nemotron_sw_pre/users/${USER_NAME}/gym_venvs \
uv run examples/nemo_gym/run_grpo_nemo_gym.py \
  --config examples/nemo_gym/async_${MODE}_nanov3.yaml \
    ${overrides}"

bash submit_nemorl.sh
