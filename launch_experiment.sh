#!/bin/bash
# Launch a colocated-vs-non-colocated async-GRPO comparison (Megatron inference) via
# submit_nemorl.sh. MODEL selects scale: nanov3 (30B nemo_gym, 8 nodes) or olmoe
# (OLMoE-1B-7B MoE on the simple math env, 4 nodes -- a fast small-MoE analog that
# reuses the nano async/colo machinery). Driver log: <jobid>-logs/ray-driver.log.
#
# Usage: SUBMIT_ACCOUNT=<acct> [MODEL=nanov3|olmoe] bash launch_experiment.sh {colocated|non_colocated} [--smoke]
# Env (both models): NUM_ACTOR_NODES, NUM_GEN_NODES (non-colo), MAX_STEPS, RUN_TAG,
#   WANDB_ENTITY (adlr), WANDB_PROJECT, REFIT_BACKEND, EXTRA_OVERRIDES (Hydra, last).
#   nanov3-only: CMP=1 (comparison preset), --smoke (SMOKE_STEPS).
set -eu

cd "$(dirname "$0")"

MODE=${1:?"usage: bash launch_experiment.sh colocated|non_colocated [--smoke]"}
case "${MODE}" in
  colocated | non_colocated) ;;
  *) echo "invalid mode: ${MODE} (expected colocated or non_colocated)" >&2; exit 1 ;;
esac

SMOKE=0
[[ "${2:-}" == "--smoke" ]] && SMOKE=1

MODEL=${MODEL:-nanov3}
# Resolve on the host (the container runs as root, so not inside COMMAND).
USER_NAME="$(whoami)"

# Node count: nano runs at 8 (4 smoke); the olmoe analog at 4 (still multi-node Ray).
if [[ "${MODEL}" == olmoe ]]; then
  export NUM_ACTOR_NODES="${NUM_ACTOR_NODES:-4}"
elif [[ "${SMOKE}" == 1 ]]; then
  export NUM_ACTOR_NODES="${NUM_ACTOR_NODES:-4}"
else
  export NUM_ACTOR_NODES="${NUM_ACTOR_NODES:-8}"
fi
# Symmetric train/gen split; asymmetric splits hang the reshard.
NUM_GEN_NODES="${NUM_GEN_NODES:-$((NUM_ACTOR_NODES / 2))}"

# RUN_TAG (optional) appends a label to the slurm job name + wandb name.
export JOB_NAME="async_${MODE}$([[ "${SMOKE}" == 1 ]] && echo _smoke)${RUN_TAG:+_${RUN_TAG}}"

# Overrides common to both models. Megatron generation + wandb entity are always set;
# project/steps override the per-model config default only when the env var is given.
overrides="policy.generation.backend=megatron"
overrides+=" cluster.num_nodes=${NUM_ACTOR_NODES}"
overrides+=" +logger.wandb.entity=${WANDB_ENTITY:-adlr}"
overrides+="${WANDB_PROJECT:+ logger.wandb.project=${WANDB_PROJECT}}"
overrides+="${MAX_STEPS:+ grpo.max_num_steps=${MAX_STEPS}}"

# Mode-specific knobs shared by both models: generation split, refit backend, memory saver.
if [[ "${MODE}" == non_colocated ]]; then
  overrides+=" policy.generation.colocated.resources.num_nodes=${NUM_GEN_NODES}"
  # refit backend for the non-colo weight reshard: nvshmem is proven stable at
  # <=4 nodes; nccl hangs the reshard here (a #2884 reorchestration regression)
  # and is only needed at 8-node scale (nvshmem hits the teams limit there).
  overrides+=" policy.generation.mcore_generation_config.refit_backend=${REFIT_BACKEND:-nvshmem}"
else
  # Colocated shares weights in-process (CUDA-IPC); refit_backend is a no-op.
  overrides+=" policy.generation.mcore_generation_config.refit_backend=${REFIT_BACKEND:-nccl}"
  # Colocated builds one inference_optimized dual-mode model with graphed decode + KV
  # offload during training, which needs torch_memory_saver in the worker venv (predates
  # the container uv.lock). Installed on every node before Ray starts; wheel pre-staged.
  export SETUP_COMMAND="${SETUP_COMMAND:-uv pip install --python /opt/ray_venvs/nemo_rl.models.policy.workers.megatron_policy_worker.MegatronPolicyWorker/bin/python --no-deps /lustre/fsw/portfolios/nemotron/users/anthomas/wheels/torch_memory_saver-0.0.9.post1-cp39-abi3-manylinux2014_aarch64.whl}"
fi

if [[ "${MODEL}" == olmoe ]]; then
  # OLMoE-1B-7B small-MoE analog. async_${MODE}_olmoe.yaml holds the model, async,
  # megatron/MoE-inference, and colocated settings; only the dynamic bits are here.
  RUN_SCRIPT="examples/run_grpo.py"
  CONFIG="examples/configs/async_${MODE}_olmoe.yaml"
  EXTRA_ENV=""
  overrides+=" checkpointing.checkpoint_dir=results/olmoe_${MODE}${RUN_TAG:+_${RUN_TAG}}"
else
  # nanov3: 30B nemo_gym. Colo time-shares all nodes; non-colo splits gen/train.
  RUN_SCRIPT="examples/nemo_gym/run_grpo_nemo_gym.py"
  CONFIG="examples/nemo_gym/async_${MODE}_nanov3.yaml"
  EXTRA_ENV="NEMO_GYM_VENV_DIR=/lustre/fs1/portfolios/nemotron/projects/nemotron_sw_pre/users/${USER_NAME}/gym_venvs"

  # Scale-down smoke: few steps, no checkpoint, shorter seq (the OOM lever), smaller
  # batch. SMOKE_STEPS>1 exercises the colo multi-step suspend/resume path.
  if [[ "${SMOKE}" == 1 ]]; then
    overrides+=" grpo.max_num_steps=${SMOKE_STEPS:-1} checkpointing.enabled=false"
    overrides+=" policy.max_total_sequence_length=16384"
    overrides+=" grpo.num_prompts_per_step=8 policy.train_global_batch_size=128"
    overrides+=" logger.wandb.name=async_${MODE}_smoke${RUN_TAG:+_${RUN_TAG}}"
  fi

  # CMP=1: comparison-run preset -- validation off, checkpoint every 5 steps (weights
  # only), seq 16384. Do not combine with --smoke.
  if [[ "${CMP:-0}" == 1 && "${SMOKE}" == 0 ]]; then
    overrides+=" policy.max_total_sequence_length=16384"
    overrides+=" grpo.val_period=0 grpo.val_at_end=false grpo.val_at_start=false"
    overrides+=" checkpointing.enabled=true checkpointing.save_period=5 checkpointing.save_optimizer=false"
    overrides+=" logger.wandb.name=async_${MODE}_cmp_noval"
  fi
fi

# Ad-hoc extra Hydra overrides (space-separated), e.g. EXTRA_OVERRIDES="a=1 b=2".
overrides+=" ${EXTRA_OVERRIDES:-}"

# HF_HOME + NRL_MEGATRON_CHECKPOINT_DIR cache the HF download and the one-time HF -> Megatron
# conversion (both shared by both models). NCCL_GRAPH_MIXING_SUPPORT=1 lets NCCL safely mix
# graph-captured decode collectives with eager training collectives on the shared colocated
# communicator. NCCL_DEBUG (default empty) can be set to WARN to surface a hanging collective.
export COMMAND="mkdir -p /tmp/nemo_rl_triton_cache && \
TRITON_CACHE_DIR=/tmp/nemo_rl_triton_cache \
NETRC=/home/${USER_NAME}/.netrc \
NCCL_GRAPH_MIXING_SUPPORT=${NCCL_GRAPH_MIXING_SUPPORT:-1} \
NCCL_DEBUG=${NCCL_DEBUG:-} \
NCCL_DEBUG_SUBSYS=${NCCL_DEBUG_SUBSYS:-} \
HF_HOME=/lustre/fsw/portfolios/nemotron/users/${USER_NAME}/hf_cache \
NRL_MEGATRON_CHECKPOINT_DIR=/lustre/fsw/portfolios/nemotron/users/${USER_NAME}/megatron_ckpt_cache \
${EXTRA_ENV:+${EXTRA_ENV} }\
uv run ${RUN_SCRIPT} \
  --config ${CONFIG} \
    ${overrides}"

bash submit_nemorl.sh
