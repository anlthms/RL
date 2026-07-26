#!/bin/bash
# Launch an async GRPO experiment (Megatron inference) via submit_nemorl.sh.
# Both modes share one override set (apples-to-apples); only non-colocated adds a
# symmetric generation split. Driver log: <jobid>-logs/ray-driver.log.
#
# Usage: SUBMIT_ACCOUNT=<account> bash launch_experiment.sh {colocated|non_colocated} [--smoke]
# Env: NUM_ACTOR_NODES (default 8 full / 4 smoke); NUM_GEN_NODES (non-colo, default NUM_ACTOR_NODES/2).
#      CMP=1 reproduces the colo-vs-non-colo comparison runs (see the CMP block below).
#      MATH=1 trains on the built-in math env (OpenMathInstruct-2) with the standard
#             run_grpo.py entrypoint -- no NeMo-Gym server or gym venv (see the *_math.yaml configs).
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

# MATH=1 selects the math-only path: standard run_grpo.py entrypoint + *_math.yaml
# configs (built-in math env, no NeMo-Gym). Default (0) is the NeMo-Gym path.
MATH="${MATH:-0}"
# Not `MATH_SFX="$(...)"`: a sole-RHS command substitution that exits nonzero
# (the [[ ]] being false) trips `set -e`. An `&&` statement is exempt.
MATH_SFX=""
[[ "${MATH}" == 1 ]] && MATH_SFX="_math"

# RUN_TAG (optional) appends a label to the slurm job name + wandb name.
export JOB_NAME="async_${MODE}$([[ "${MATH}" == 1 ]] && echo _math)$([[ "${SMOKE}" == 1 ]] && echo _smoke)${RUN_TAG:+_${RUN_TAG}}"

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
  overrides+=" logger.wandb.name=async_${MODE}${MATH_SFX}_smoke${RUN_TAG:+_${RUN_TAG}}"
fi

# CMP=1: comparison-run preset -- validation off, checkpoint every 5 steps (weights
# only), seq 16384. Do not combine with --smoke.
if [[ "${CMP:-0}" == 1 && "${SMOKE}" == 0 ]]; then
  overrides+=" policy.max_total_sequence_length=16384"
  overrides+=" grpo.val_period=0 grpo.val_at_end=false grpo.val_at_start=false"
  overrides+=" checkpointing.enabled=true checkpointing.save_period=5 checkpointing.save_optimizer=false"
  overrides+=" logger.wandb.name=async_${MODE}${MATH_SFX}_cmp_noval"
fi

# Ad-hoc extra Hydra overrides (space-separated), e.g. EXTRA_OVERRIDES="a=1 b=2".
overrides+=" ${EXTRA_OVERRIDES:-}"

# Entrypoint + config: math uses the standard run_grpo.py (no NeMo-Gym); the
# default path uses the NeMo-Gym entrypoint. Both share the override set above.
if [[ "${MATH}" == 1 ]]; then
  ENTRYPOINT="examples/run_grpo.py"
  CONFIG="examples/nemo_gym/async_${MODE}_nanov3_math.yaml"
  # Shared HF cache on lustre: OpenMathInstruct-2's mmap'd Arrow cache is pickled to
  # a collector on another node, so node-local /root/.cache -> FileNotFoundError.
  HF_HOME_ENV="HF_HOME=${HF_HOME:-/lustre/fsw/portfolios/nemotron/users/${USER_NAME}/hf_home}"
else
  ENTRYPOINT="examples/nemo_gym/run_grpo_nemo_gym.py"
  CONFIG="examples/nemo_gym/async_${MODE}_nanov3.yaml"
  HF_HOME_ENV=""
fi

# COLL_TRACE=1 enables torch's NCCL flight recorder: dumps each rank's last
# collectives to ./nccl_traces/ on a watchdog timeout (shows the stuck collective).
COLL_TRACE_ENV=""
if [[ "${COLL_TRACE:-0}" == 1 ]]; then
  mkdir -p "$(pwd)/nccl_traces"
  COLL_TRACE_ENV="TORCH_NCCL_TRACE_BUFFER_SIZE=${TORCH_NCCL_TRACE_BUFFER_SIZE:-20000} TORCH_NCCL_DUMP_ON_TIMEOUT=1 TORCH_NCCL_DEBUG_INFO_TEMP_FILE=$(pwd)/nccl_traces/trace_"
fi

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
${HF_HOME_ENV} \
${COLL_TRACE_ENV} \
uv run ${ENTRYPOINT} \
  --config ${CONFIG} \
    ${overrides}"

bash submit_nemorl.sh
