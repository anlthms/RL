#!/bin/bash
# Offline-validate every checkpoint of a previous run and plot reward-vs-step in a
# single wandb run. Loops step_N/ under the checkpoint dir, runs one offline-val per
# checkpoint (val_at_start, no training), collects accuracy, then logs the curve to
# one wandb run so all points share a single plot.
#
# Usage: SUBMIT_ACCOUNT=<acct> [MODEL=qwen3_1p7b|nanov3] bash validate_all_checkpoints.sh <checkpoint_dir> [colocated|non_colocated]
# Env: FORMAT (pretrained_checkpoint format, default torch_dist), WANDB_NAME (plot run
#      name), WANDB_PROJECT (mllm-rl-dev), WANDB_ENTITY (adlr), TIMEOUT_MIN (per ckpt, 40).
set -eu
cd "$(dirname "$0")"

CKPT_DIR=${1:?"usage: bash validate_all_checkpoints.sh <checkpoint_dir> [mode]"}
MODE=${2:-colocated}
[ -d "${CKPT_DIR}" ] || { echo "checkpoint dir not found: ${CKPT_DIR}" >&2; exit 1; }

FORMAT="${FORMAT:-megatron_lm}"
TIMEOUT_MIN="${TIMEOUT_MIN:-40}"
WANDB_NAME="${WANDB_NAME:-offlineval_$(basename "${CKPT_DIR}")}"
RESULTS="$(mktemp)"

# Validate each step_N checkpoint in ascending order.
steps=$(ls -d "${CKPT_DIR}"/step_* 2>/dev/null | sed 's#.*/step_##' | sort -n)
[ -n "${steps}" ] || { echo "no step_N checkpoints under ${CKPT_DIR}" >&2; exit 1; }

for step in ${steps}; do
  weights="${CKPT_DIR}/step_${step}/policy/weights"
  [ -d "${weights}" ] || { echo "  step ${step}: no weights dir, skipping"; continue; }
  echo "=== offline-val step ${step} ==="

  EXTRA_OVERRIDES="grpo.val_at_start=true grpo.max_num_steps=1 checkpointing.enabled=false \
logger.wandb_enabled=false \
+checkpointing.pretrained_checkpoint.format=${FORMAT} \
+checkpointing.pretrained_checkpoint.path=${weights}"
  out=$(EXTRA_OVERRIDES="${EXTRA_OVERRIDES}" RUN_TAG="oval_s${step}" bash launch_experiment.sh "${MODE}")
  JID=$(echo "${out}" | grep -oE "Submitted batch job [0-9]+" | grep -oE "[0-9]+" | tail -1)
  [ -n "${JID:-}" ] || { echo "  step ${step}: failed to submit"; continue; }
  L="${JID}-logs/ray-driver.log"

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
