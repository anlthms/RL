#!/bin/bash
# Run this from the root of your NeMo-RL checkout.
set -euo pipefail

# --- Config -----------------------------------------------------------------
# Source tree that holds the patches, yaml, and sbatch script.
SRC="/lustre/fsw/portfolios/llmservice/users/tene/rl_pls_be_valuable_research"

# Repo root (this checkout). Derived from git so it is correct wherever you run.
REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "$REPO_ROOT"
echo ">> Repo root: $REPO_ROOT"

# Apply a git patch inside a submodule repo. Copies the diff into that repo
# (so 'git apply tmp.diff' works as written) and applies it, skipping cleanly
# if it was already applied.
apply_patch() {
    local repo_dir="$1" diff_src="$2"
    echo ">> Patching $repo_dir"
    ( cd "$repo_dir"
      cp "$diff_src" ./tmp.diff
      if git apply --check tmp.diff 2>/dev/null; then
          git apply tmp.diff
          echo "   applied tmp.diff"
      elif git apply --reverse --check tmp.diff 2>/dev/null; then
          echo "   tmp.diff already applied, skipping"
      else
          echo "   ERROR: tmp.diff does not apply cleanly to $repo_dir" >&2
          exit 1
      fi
    )
}

# --- 2. Init submodules -----------------------------------------------------
echo ">> git submodule update --init --recursive"
git submodule update --init --recursive

# --- 3. Patch Megatron-Bridge -----------------------------------------------
apply_patch \
    "3rdparty/Megatron-Bridge-workspace/Megatron-Bridge" \
    "$SRC/RL/3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/tmp.diff"

# --- 5. Patch Megatron-LM ---------------------------------------------------
apply_patch \
    "3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/3rdparty/Megatron-LM" \
    "$SRC/RL/3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/3rdparty/Megatron-LM/tmp.diff"

# --- 6. Copy the run yaml and point checkpoint_dir under REPO_ROOT's parent --
YAML_DST="examples/nemo_gym/async_nanov3_base.yaml"
echo ">> Copying run yaml -> $YAML_DST"
cp "$SRC/RL/examples/nemo_gym/actually_run_nanov3.yaml" "$YAML_DST"
# Rewrite the checkpoint_dir prefix to the parent of REPO_ROOT, keeping the
# 'checkpoints/${logger.wandb.name}' suffix (OmegaConf interpolation) intact.
CKPT_PARENT="$(dirname "$REPO_ROOT")"
sed -i "s#checkpoint_dir: .*/checkpoints/#checkpoint_dir: ${CKPT_PARENT}/checkpoints/#" "$YAML_DST"
echo "   checkpoint_dir now: $(grep 'checkpoint_dir:' "$YAML_DST")"

# --- 7. Copy the sbatch script and rewrite the MOUNTS line ------------------
SBATCH_DST="submit_nemorl.sh"
echo ">> Copying sbatch script -> $SBATCH_DST"
cp "$SRC/submit_nemorl.sh" "$SBATCH_DST"
# Point /opt/nemo-rl at this repo and also mount $HOME.
NEW_MOUNTS="export MOUNTS=\"/lustre:/lustre,/home:/home,${REPO_ROOT}:/opt/nemo-rl,\${HOME}:\${HOME}\""
sed -i "s#^export MOUNTS=.*#${NEW_MOUNTS}#" "$SBATCH_DST"
echo "   $(grep '^export MOUNTS=' "$SBATCH_DST")"

echo ">> Done."
