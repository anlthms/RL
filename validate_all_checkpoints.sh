#!/bin/bash
# Offline-validate every checkpoint of a run and plot reward-vs-step in one wandb
# run. Loops step_N/ under the checkpoint dir, runs run_megatron_eval.py (inference
# only -- no optimizer/scheduler) per checkpoint, collects accuracy, logs the curve.
#
# Usage: SUBMIT_ACCOUNT=<acct> bash validate_all_checkpoints.sh <checkpoint_dir> [config_yaml]
#   config_yaml: eval config (default examples/configs/async/nanov3_gym_non_colocated.yaml).
#     Must match the env the checkpoint was trained on -- pass the *_math_* leaf to
#     score a math run, otherwise validation runs against the wrong environment.
# Env: SEQ_LEN (checkpoint's training seq, default 16384), NUM_ACTOR_NODES (4), QOS
#      (interactive), TIMEOUT_MIN (per ckpt, 40), WANDB_NAME/PROJECT, WANDB_ENTITY (adlr).
set -eu
cd "$(dirname "$0")"

CKPT_DIR=${1:?"usage: bash validate_all_checkpoints.sh <checkpoint_dir> [config_yaml]"}
CONFIG=${2:-examples/configs/async/nanov3_gym_non_colocated.yaml}
[ -d "${CKPT_DIR}" ] || { echo "checkpoint dir not found: ${CKPT_DIR}" >&2; exit 1; }
[ -f "${CONFIG}" ] || { echo "config not found: ${CONFIG}" >&2; exit 1; }

SEQ_LEN="${SEQ_LEN:-16384}"
TIMEOUT_MIN="${TIMEOUT_MIN:-40}"
WANDB_NAME="${WANDB_NAME:-offlineval_$(basename "${CKPT_DIR}")}"
RESULTS="$(mktemp)"
# Inference cluster: a dedicated non-colocated generation model; 4 nodes fits the
# nano-v3 parallelism and the interactive QOS (fast alloc).
export NUM_ACTOR_NODES="${NUM_ACTOR_NODES:-4}"
export QOS="${QOS:-interactive}"
USER_NAME="$(whoami)"

# Validate each step_N checkpoint in ascending order. STEPS="5 10 15" restricts to
# a subset (e.g. steps present in both runs for a matched colo-vs-non-colo curve).
if [ -n "${STEPS:-}" ]; then
  steps="${STEPS}"
else
  steps=$(ls -d "${CKPT_DIR}"/step_* 2>/dev/null | sed 's#.*/step_##' | sort -n)
fi
[ -n "${steps}" ] || { echo "no step_N checkpoints under ${CKPT_DIR}" >&2; exit 1; }

for step in ${steps}; do
  weights="${CKPT_DIR}/step_${step}/policy/weights"
  [ -d "${weights}" ] || { echo "  step ${step}: no weights dir, skipping"; continue; }
  echo "=== offline-val step ${step} ==="

  export JOB_NAME="oval_s${step}"
  export COMMAND="mkdir -p /tmp/nemo_rl_triton_cache && \
TRITON_CACHE_DIR=/tmp/nemo_rl_triton_cache \
NETRC=/home/${USER_NAME}/.netrc \
NRL_MEGATRON_CHECKPOINT_DIR=/lustre/fsw/portfolios/nemotron/users/${USER_NAME}/megatron_ckpt_cache \
NEMO_GYM_VENV_DIR=${NEMO_GYM_VENV_DIR:-/lustre/fs1/portfolios/nemotron/projects/nemotron_sw_pre/users/${USER_NAME}/gym_venvs_main} \
HF_HOME=${HF_HOME:-/lustre/fsw/portfolios/nemotron/users/${USER_NAME}/hf_home} \
uv run examples/run_megatron_eval.py \
  --config ${CONFIG} --checkpoint ${weights} \
  cluster.num_nodes=${NUM_ACTOR_NODES} policy.max_total_sequence_length=${SEQ_LEN} \
  policy.generation.mcore_generation_config.refit_backend=nvshmem"
  out=$(bash submit_nemorl.sh)
  JID=$(echo "${out}" | grep -oE "Submitted batch job [0-9]+" | grep -oE "[0-9]+" | tail -1)
  [ -n "${JID:-}" ] || { echo "  step ${step}: failed to submit"; continue; }
  L="${JID}-logs/ray-driver.log"

  # run_megatron_eval self-completes after logging Accuracy; the scancel below is a
  # safety net in case teardown wedges.
  acc=""; deadline=$(( $(date +%s) + TIMEOUT_MIN * 60 ))
  while [ "$(date +%s)" -lt "${deadline}" ]; do
    if [ -f "${L}" ]; then
      acc=$(grep -aoE "Accuracy: [0-9.]+" "${L}" 2>/dev/null | tail -1 | grep -oE "[0-9.]+" || true)
      [ -n "${acc}" ] && break
      grep -aqE "illegal memory access|ActorDiedError|OutOfMemoryError|CUDA error" "${L}" 2>/dev/null && break
    fi
    [ -z "$(squeue -j "${JID}" -h -o '%T' 2>/dev/null)" ] && break
    sleep 20
  done
  scancel "${JID}" 2>/dev/null || true

  if [ -n "${acc}" ]; then
    echo "  step ${step}: accuracy=${acc}"; printf '%s\t%s\n' "${step}" "${acc}" >> "${RESULTS}"
  else
    echo "  step ${step}: no result (timeout/crash), skipping"
  fi
done

# RESULTS_TSV keeps the step/reward rows for a caller that logs the curve itself
# (see offline_val_compare.sh, which merges both arms into one wandb run).
if [ -n "${RESULTS_TSV:-}" ]; then
  cp "${RESULTS}" "${RESULTS_TSV}"
fi
if [ "${SKIP_WANDB:-0}" = 1 ]; then
  rm -f "${RESULTS}"
  echo "=== skipping wandb (SKIP_WANDB=1) ==="
  exit 0
fi

echo "=== logging curve to wandb run '${WANDB_NAME}' ==="
NETRC="/home/$(whoami)/.netrc" WANDB_NAME="${WANDB_NAME}" \
WANDB_ENTITY="${WANDB_ENTITY:-adlr}" WANDB_PROJECT="${WANDB_PROJECT:-mllm-rl-dev}" \
RESULTS="${RESULTS}" python3 - <<'PY'
import os, wandb
rows = [l.split() for l in open(os.environ["RESULTS"]) if l.strip()]
run = wandb.init(entity=os.environ["WANDB_ENTITY"], project=os.environ["WANDB_PROJECT"],
                 name=os.environ["WANDB_NAME"])
for step, acc in rows:
    run.log({"offline_val/accuracy": float(acc)}, step=int(step))
run.finish()
print(f"logged {len(rows)} checkpoints to wandb")
PY
rm -f "${RESULTS}"
