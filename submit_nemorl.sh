export NUM_ACTOR_NODES=${NUM_ACTOR_NODES:-8}
export GPUS_PER_NODE=${GPUS_PER_NODE:-4}
# QOS: "interactive" bypasses the normal queue for fast allocation (limit: 4 nodes total).
# Override with QOS= (empty) or QOS=<other> for non-interactive / larger jobs.
export QOS=${QOS:-interactive}
# Slurm wall clock, in minutes.
export TIMEOUT_MIN=${TIMEOUT_MIN:-240}

export CONTAINER="/lustre/fs1/portfolios/llmservice/projects/llmservice_fm_text/users/tene/nemo_rl_0624.sqsh"
export MOUNTS="/lustre:/lustre,/home:/home,/lustre/fs1/portfolios/nemotron/projects/nemotron_sw_pre/users/anthomas/RL3:/opt/nemo-rl,${HOME}:${HOME}"

# Leave the command below commented out to get an interactive job.

export EXTRA_NEMO_RL_ARGS="$@"

# If JOB_NAME is not set, use the default
if [ -z "$JOB_NAME" ]; then
    JOB_NAME="nemo_rl_0526_interactive"
fi

sbatch \
--nodes=$NUM_ACTOR_NODES \
--account=$SUBMIT_ACCOUNT \
--job-name=$SUBMIT_ACCOUNT:$JOB_NAME \
--partition=batch \
${QOS:+--qos=$QOS} \
--time=$((TIMEOUT_MIN / 60)):$((TIMEOUT_MIN % 60)):0 \
--gres=gpu:4 \
--comment='{"OccupiedIdleGPUsJobReaper":{"exemptIdleTimeMins":"120","reason":"interactive","description":"Interactive debugging of multinode Nano training."}}' \
ray.sub
