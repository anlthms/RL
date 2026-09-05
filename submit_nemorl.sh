export NUM_ACTOR_NODES=${NUM_ACTOR_NODES:-8}
export GPUS_PER_NODE=${GPUS_PER_NODE:-4}
# QOS: "interactive" bypasses the normal queue for fast allocation (limit: 4 nodes total).
# Override with QOS= (empty) or QOS=<other> for non-interactive / larger jobs.
export QOS=${QOS:-interactive}
# Slurm wall clock, in minutes.
export TIMEOUT_MIN=${TIMEOUT_MIN:-240}

export CONTAINER="${CONTAINER:-/lustre/fs1/portfolios/llmservice/projects/llmservice_fm_text/users/tene/nemo_rl_0807.sqsh}"
# Bind /opt/nemo-rl to whichever checkout this script lives in, so a second
# working tree does not silently run the first one's code.
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export MOUNTS="/lustre:/lustre,/home:/home,${REPO_ROOT}:/opt/nemo-rl,${HOME}:${HOME}"

# Leave the command below commented out to get an interactive job.

export EXTRA_NEMO_RL_ARGS="$@"

SBATCH_NODE_ARGS=()
if [[ -n "${EXCLUDE_NODES:-}" ]]; then
    SBATCH_NODE_ARGS+=(--exclude="${EXCLUDE_NODES}")
fi

SBATCH_MEMORY_ARGS=()
if [[ -n "${MEM_PER_NODE:-}" ]]; then
    SBATCH_MEMORY_ARGS+=(--mem="${MEM_PER_NODE}")
fi

# If JOB_NAME is not set, use the default
if [ -z "$JOB_NAME" ]; then
    JOB_NAME="nemo_rl_0526_interactive"
fi

sbatch \
"${SBATCH_NODE_ARGS[@]}" \
"${SBATCH_MEMORY_ARGS[@]}" \
--nodes=$NUM_ACTOR_NODES \
--account=$SUBMIT_ACCOUNT \
--job-name=$SUBMIT_ACCOUNT:$JOB_NAME \
--partition=batch \
${QOS:+--qos=$QOS} \
--time=$((TIMEOUT_MIN / 60)):$((TIMEOUT_MIN % 60)):0 \
--gres=gpu:4 \
--comment='{"OccupiedIdleGPUsJobReaper":{"exemptIdleTimeMins":"120","reason":"interactive","description":"Interactive debugging of multinode Nano training."}}' \
ray.sub
