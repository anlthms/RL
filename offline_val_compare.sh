#!/bin/bash
# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Offline-validate several training runs and plot every arm's mean-reward curve in
# ONE wandb run, on a consumed-samples x-axis.
#
# Samples, not steps: arms may train at different global batch sizes (the colocated
# arm runs 2x), so equal step numbers are not equal training data and a vs-step
# comparison overstates whichever arm has the bigger batch.
#
# Usage:
#   SUBMIT_ACCOUNT=<acct> bash offline_val_compare.sh \
#     <label>:<checkpoint_dir>:<samples_per_step> [more arms...]
#
# Env: WANDB_NAME (default offlineval_compare), WANDB_PROJECT, WANDB_ENTITY,
#      CONFIG (eval config; must match the env the checkpoints were trained on),
#      plus everything validate_all_checkpoints.sh honours (SEQ_LEN, STEPS,
#      NUM_ACTOR_NODES, QOS, TIMEOUT_MIN).
set -eu
cd "$(dirname "$0")"

[ "$#" -ge 1 ] || { echo "usage: bash offline_val_compare.sh <label>:<dir>:<samples_per_step> ..." >&2; exit 1; }

CONFIG="${CONFIG:-examples/configs/async/nanov3_gym_non_colocated.yaml}"
WANDB_NAME="${WANDB_NAME:-offlineval_compare}"
WORK="$(mktemp -d)"
trap 'rm -rf "${WORK}"' EXIT

MERGED="${WORK}/merged.tsv"
: > "${MERGED}"

for spec in "$@"; do
  label="${spec%%:*}"
  rest="${spec#*:}"
  ckpt_dir="${rest%%:*}"
  samples_per_step="${rest##*:}"
  [ -n "${label}" ] && [ -d "${ckpt_dir}" ] || { echo "bad arm spec: ${spec}" >&2; exit 1; }
  case "${samples_per_step}" in
    ''|*[!0-9]*) echo "samples_per_step must be an integer: ${spec}" >&2; exit 1 ;;
  esac

  echo "########## offline validation: ${label} (${samples_per_step} samples/step) ##########"
  arm_tsv="${WORK}/${label}.tsv"
  SKIP_WANDB=1 RESULTS_TSV="${arm_tsv}" bash validate_all_checkpoints.sh "${ckpt_dir}" "${CONFIG}"

  [ -s "${arm_tsv}" ] || { echo "  no results for ${label}; it contributes no curve" >&2; continue; }
  while IFS=$'\t' read -r step acc; do
    [ -n "${step:-}" ] || continue
    printf '%s\t%s\t%s\t%s\n' "${label}" "${step}" "$((step * samples_per_step))" "${acc}" >> "${MERGED}"
  done < "${arm_tsv}"
done

[ -s "${MERGED}" ] || { echo "no arm produced any result; nothing to log" >&2; exit 1; }

echo "=== logging all arms to wandb run '${WANDB_NAME}' ==="
NETRC="/home/$(whoami)/.netrc" WANDB_NAME="${WANDB_NAME}" \
WANDB_ENTITY="${WANDB_ENTITY:-adlr}" WANDB_PROJECT="${WANDB_PROJECT:-mllm-rl-dev}" \
MERGED="${MERGED}" python3 - <<'PY'
import os

import wandb

rows = [line.split("\t") for line in open(os.environ["MERGED"]) if line.strip()]
# One wandb step per consumed-samples value, ascending, so both arms share an axis.
rows.sort(key=lambda r: int(r[2]))
labels = sorted({r[0] for r in rows})

run = wandb.init(
    entity=os.environ["WANDB_ENTITY"],
    project=os.environ["WANDB_PROJECT"],
    name=os.environ["WANDB_NAME"],
)
run.define_metric("offline_val/consumed_samples")
for label in labels:
    run.define_metric(
        f"offline_val/{label}/mean_reward", step_metric="offline_val/consumed_samples"
    )

for label, step, samples, acc in rows:
    run.log(
        {
            f"offline_val/{label}/mean_reward": float(acc),
            f"offline_val/{label}/step": int(step),
            "offline_val/consumed_samples": int(samples),
        },
        step=int(samples),
    )
run.finish()

for label in labels:
    pts = [r for r in rows if r[0] == label]
    print(f"{label}: {len(pts)} points, steps {pts[0][1]}..{pts[-1][1]}")
PY
