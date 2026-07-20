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

# RUN_TAG (optional) appends a label to the slurm job name + wandb name.
export JOB_NAME="async_${MODE}$([[ "${SMOKE}" == 1 ]] && echo _smoke)${RUN_TAG:+_${RUN_TAG}}"

# Resolve on the host (the container runs as root, so not inside COMMAND).
USER_NAME="$(whoami)"

# Common overrides.
overrides="policy.generation.backend=megatron"
overrides+=" cluster.num_nodes=${NUM_ACTOR_NODES}"

# Mode-specific knobs.
if [[ "${MODE}" == non_colocated ]]; then
  overrides+=" policy.generation.colocated.resources.num_nodes=${NUM_GEN_NODES}"
  # refit backend for the non-colo weight reshard: nvshmem is proven stable at
  # <=4 nodes; nccl hangs the reshard here (a #2884 reorchestration regression)
  # and is only needed at 8-node scale (nvshmem hits the teams limit there).
  overrides+=" policy.generation.mcore_generation_config.refit_backend=${REFIT_BACKEND:-nvshmem}"
else
  # Colocated shares weights in-process (CUDA-IPC); refit_backend is a no-op.
  overrides+=" policy.generation.mcore_generation_config.refit_backend=${REFIT_BACKEND:-nccl}"
  # Colocated builds one inference_optimized dual-mode model (see
  # async_colocated_nanov3.yaml) with graphed decode + KV offload during training,
  # which needs torch_memory_saver in the worker venv (predates the container
  # uv.lock). Installed on every node before Ray starts; wheel pre-staged on lustre.
  export SETUP_COMMAND="${SETUP_COMMAND:-uv pip install --python /opt/ray_venvs/nemo_rl.models.policy.workers.megatron_policy_worker.MegatronPolicyWorker/bin/python --no-deps /lustre/fsw/portfolios/nemotron/users/anthomas/wheels/torch_memory_saver-0.0.9.post1-cp39-abi3-manylinux2014_aarch64.whl}"
fi

# Scale-down: few steps, no checkpoint, shorter seq (the OOM lever), smaller batch.
# SMOKE_STEPS>1 exercises the colo multi-step suspend/resume path (a 1-step smoke
# cannot surface multi-step issues).
if [[ "${SMOKE}" == 1 ]]; then
  overrides+=" grpo.max_num_steps=${SMOKE_STEPS:-1} checkpointing.enabled=false"
  overrides+=" policy.max_total_sequence_length=16384"
  overrides+=" grpo.num_prompts_per_step=8 policy.train_global_batch_size=128"
  overrides+=" logger.wandb.name=async_${MODE}_smoke${RUN_TAG:+_${RUN_TAG}}"
fi

# Ad-hoc extra Hydra overrides (space-separated), e.g. EXTRA_OVERRIDES="a=1 b=2".
overrides+=" ${EXTRA_OVERRIDES:-}"

# NRL_MEGATRON_CHECKPOINT_DIR caches the one-time HF -> Megatron conversion (shared by both modes).
# NCCL_GRAPH_MIXING_SUPPORT=1 lets NCCL safely mix graph-captured decode collectives
# with eager training collectives on the shared colocated communicator.
# NCCL_DEBUG (default empty) can be set to WARN to surface a hanging collective.
export COMMAND="mkdir -p /tmp/nemo_rl_triton_cache && \
TRITON_CACHE_DIR=/tmp/nemo_rl_triton_cache \
NETRC=/home/${USER_NAME}/.netrc \
NCCL_GRAPH_MIXING_SUPPORT=${NCCL_GRAPH_MIXING_SUPPORT:-1} \
NCCL_DEBUG=${NCCL_DEBUG:-} \
NCCL_DEBUG_SUBSYS=${NCCL_DEBUG_SUBSYS:-} \
NRL_MEGATRON_CHECKPOINT_DIR=/lustre/fsw/portfolios/nemotron/users/${USER_NAME}/megatron_ckpt_cache \
NEMO_GYM_VENV_DIR=/lustre/fs1/portfolios/nemotron/projects/nemotron_sw_pre/users/${USER_NAME}/gym_venvs \
uv run examples/nemo_gym/run_grpo_nemo_gym.py \
  --config examples/nemo_gym/async_${MODE}_nanov3.yaml \
    ${overrides}"

bash submit_nemorl.sh
