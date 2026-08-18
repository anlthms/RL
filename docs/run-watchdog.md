# Hang Watchdog

`tools/run_watchdog.py` kills a Slurm job whose GPUs have gone idle, after
snapshotting the evidence needed to explain why.

A hung distributed job holds its whole allocation until something reaps it.
Where a cluster's idle-GPU reaper allows two hours, that is a large number of
GPU-hours spent producing nothing — and the reaper takes the process state with
it, so the hang cannot be diagnosed afterwards.

## Usage

The watchdog is **armed automatically** by `launch_experiment.sh` after every
submission. Nothing is required to use it:

```
watchdog: armed on 1234567 (pid 98765), log 1234567-watchdog.log
```

It runs on the submit host (it only shells out to `sacct`, `srun` and
`scancel`) under `setsid`, so it outlives the launching shell.

| Variable | Default | Meaning |
|---|---|---|
| `WATCHDOG=0` | armed | do not arm the watchdog |
| `WATCHDOG_INTERVAL` | `300` | seconds between polls |

To watch a job by hand — one already running, or one launched another way:

```bash
uv run tools/run_watchdog.py <jobid> --log-dir <jobid>-logs
uv run tools/run_watchdog.py <jobid> --once      # single evaluation, no loop
```

| Flag | Default | Meaning |
|---|---|---|
| `--interval` | `300` | seconds between polls |
| `--idle-polls` | `4` | consecutive idle polls before declaring a hang |
| `--grace-minutes` | `45` | ignore idle GPUs for this long after the job starts |
| `--root` | cwd | where to write the evidence bundle |
| `--log-dir` | — | directory of `*.log` files to tail into the bundle |

## How it decides

**GPU utilization is the progress signal, not log output.** A job can be quiet
for many minutes while doing real work — a long evaluation pass, a slow
checkpoint — so treating silence as failure kills healthy runs. Busy GPUs mean
progress whatever the logs say, which also keeps the tool free of any knowledge
of the application inside the job.

A hang is declared when all of these hold:

1. Slurm still reports the job as running.
2. Fewer than 5% of its GPUs are above 5% utilization.
3. That has been true for `--idle-polls` consecutive polls.
4. The job is past `--grace-minutes`.

The grace period exists because a job's GPUs are legitimately idle while it
starts up — pulling a container, converting weights, loading a model across
ranks. That is indistinguishable from a hang by utilization alone, and it is
long: tens of minutes for a large model. Without it the watchdog kills every job
minutes after launch.

An **unavailable** utilization sample is not a vote. A probe that failed to run
is no evidence the GPUs are idle, so a transient `srun` failure can neither trip
nor reset the detector.

Detection latency is therefore `grace + idle_polls × interval` at worst, and
utilization takes a few minutes to drain after work actually stops.

## What it captures

On a trip it captures **before** it kills, because `scancel` destroys exactly
the evidence a post-mortem needs. Written to `<jobid>-hang/`:

| File | Contents |
|---|---|
| `summary.txt` | job state, GPU busy fraction, nodes, capture time |
| `py-spy.txt` | per-rank Python stacks — usually the most useful file |
| `nvidia-smi.txt` | confirms idle, and shows memory still resident |
| `sockets.txt` | a rank waiting on a peer that never connected shows here |
| `dmesg.txt` | Xid errors, OOM kills, fabric resets |
| `tail-*.log` | the last 400 lines of each log the job was writing |

Probes run via `srun --overlap` rather than ssh, which is the portable way onto
a node the job already holds and works where ssh to compute nodes is closed.

## Afterwards: the `diagnose-hung-job` skill

The watchdog does detection and the kill, both as plain thresholds. The
judgement half — root cause, code reading, what to do next — is a Claude skill,
because it needs reading and inference rather than a comparison against a
number. The watchdog ends by naming it:

```
next: run the `diagnose-hung-job` skill against /path/to/1234567-hang
```

**To use it**, start an agent session in the repo and ask for a diagnosis,
pointing at the bundle. The skill is selected automatically from the request —
no need to name it:

```
why did job 1234567 hang? the bundle is in 1234567-hang/
```

It also triggers on a run that stopped without a bundle ("the run is stuck",
"job was cancelled but I did not cancel it", "job says COMPLETED but produced
nothing"), which are the cases where the job state is misleading. To invoke it
explicitly instead, use `/diagnose-hung-job`.

What it does, given the bundle and the job's logs:

1. Locates the failure against a **healthy** run's own progress markers, turning
   "it hung" into "it hung *here*".
2. Builds a discriminating matrix across runs — the axis where one cell differs
   is the finding.
3. Reads the bundle, `py-spy.txt` first, and reads hang signatures correctly
   (heartbeat and store failures are teardown symptoms, not causes).
4. Writes `reports/hangs/<jobid>.md` with what happened, where it stopped, the
   evidence, an honestly-hedged root cause, the next command to run, and the
   GPU-hours burned.

It is written to say "cause undetermined" when the evidence does not support
more, and to prescribe instrumentation instead — which is the common outcome on
a first occurrence.

Pair it with the NCCL flight recorder, on by default in `launch_experiment.sh`
(`COLL_TRACE=0` disables). It writes nothing unless a collective times out, and
it is the one artifact that names a stuck collective.
