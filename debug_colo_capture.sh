#!/bin/bash
# Capture per-rank state while the colocated run is hung.
# Run from a LOGIN-NODE shell (not the attach shell), while the job is alive:
#   bash debug_colo_capture.sh <slurm_job_id>
#
# For every node in the job, dumps native+python stacks of every
# MegatronPolicyWorker (all threads: main, inference loop, NCCL watchdog)
# using the pre-staged py-spy binary. Output: one file per worker pid under
# $OUT. Then diff main-thread stacks across the ranks of one EP/TP island and
# check each rank's inference-loop thread is parked (guide §6.4).
set -eu
JID=${1:?usage: bash debug_colo_capture.sh <slurm_job_id>}

PYSPY=/lustre/fsw/portfolios/nemotron/users/anthomas/wheels/py-spy
OUT=/lustre/fsw/portfolios/nemotron/users/anthomas/nccl_traces/pyspy_${JID}_$(date +%H%M%S)
mkdir -p "$OUT"
echo "Dumps -> $OUT"

nodes=$(scontrol show hostnames "$(squeue -h -j "$JID" -o %N)")
i=0
for node in $nodes; do
  # ray.sub names the head container ray-head and workers ray-worker-<i>
  if [ $i -eq 0 ]; then cname=ray-head; else cname=ray-worker-$i; fi
  echo "=== $node ($cname) ==="
  srun --overlap --jobid="$JID" -N1 -n1 -w "$node" \
    --container-name="$cname" --no-container-mount-home \
    bash -c '
      for pid in $(pgrep -f MegatronPolicyWorker); do
        echo "  py-spy dump pid=$pid"
        '"$PYSPY"' dump --pid "$pid" --native \
          > '"$OUT"'/$(hostname)_pid${pid}.txt 2>&1 || true
      done
    ' || echo "  (srun into $node/$cname failed — try the other container name)"
  i=$((i+1))
done

echo
echo "Done. Quick triage:"
echo "  grep -l 'get_logprobs' $OUT/*.txt   # ranks in the training forward"
echo "  grep -L 'get_logprobs' $OUT/*.txt   # ranks somewhere else  <- suspects"
