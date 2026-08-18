---
name: diagnose-hung-job
description: Root-cause a distributed job that hung, was reaped, or exited successfully without finishing, then write actionable recommendations to a markdown file. Covers reading an evidence bundle, locating the failure in the job's own progress markers, building a discriminating matrix across runs, and prescribing instrumentation when the evidence is insufficient.
when_to_use: A run stopped making progress or was killed by a watchdog or cluster reaper; `tools/run_watchdog.py` wrote a `<jobid>-hang/` bundle; 'why did the job hang', 'the run is stuck', 'job was cancelled but I did not cancel it', 'no steps logged', 'job says COMPLETED but produced nothing'.
---

# Diagnosing a hung job

Invoked after `tools/run_watchdog.py` confirms a hang, or by hand on a run that
stopped. The watchdog does detection and the kill, both as plain thresholds;
this skill does the judgement half — root cause, code reading, and a written
handoff another session can act on.

## 0. Two things that mislead people first

- **Job state lies about success.** A framework that catches its own async error
  can exit `COMPLETED` with status 0 having produced nothing. Never conclude
  from `sacct` alone; read the driver log and confirm the run reached the work
  it was supposed to do.
- **`CANCELLED` with no reason is usually the cluster.** Check the job's
  `--comment` for a reaper policy and compare its threshold against the elapsed
  time. An elapsed time that lands suspiciously close to a reaper's idle
  allowance means the job hung and was collected, not that someone cancelled it.

## 1. Locate the failure against the job's own progress markers

Convert "it hung" into "it hung *here*". Every training driver prints an ordered
sequence of setup and progress lines; find the last one present, and the hang is
in the phase immediately after it.

Do not hardcode a marker list — derive it from a healthy run of the same
workload, which is authoritative and stays current:

```bash
# The markers a healthy run prints, in order, with their first occurrence.
grep -anE '^(=+ Step|Setting up|Loaded |Running on |Starting |Running initial)' \
  <healthy-jobid>-logs/ray-driver.log | head -40
```

Then check which of those the hung run reached. Log size is a fast first cut: a
hung run's driver log is orders of magnitude smaller than a healthy one's.

## 2. Always diff against a healthy run

One log tells you where it stopped; two tell you what should have happened next.
Read the ~20 lines *after* the last shared marker in the healthy run — that is
the operation that hung.

## 3. Build a discriminating matrix

Never report a cause from a single observation. Tabulate every run of this
workload against the axis you suspect — parallelism, node count, batch shape,
image version — and find the one cell that differs:

| axis value | scale | result |
|---|---|---|
| … | … | ran to completion |
| … | … | hung |

The table *is* the finding. It names the suspect configuration without
pretending to know the mechanism.

## 4. Read the evidence bundle

`tools/run_watchdog.py` writes `<jobid>-hang/` **before** killing:

- `summary.txt` — job state, GPU busy fraction, nodes, capture time
- `py-spy.txt` — Python stacks per rank. The most valuable file: it names the
  exact collective, lock or socket every rank is parked on.
- `nvidia-smi.txt` — confirms idle vs busy, and shows memory still resident
- `sockets.txt` — a rank waiting on a peer that never connected shows here
- `dmesg.txt` — Xid errors, OOM kills, fabric resets
- `tail-*.log` — the last of each log the job was writing

If there is no bundle, the job was killed before anyone looked and the process
state is gone. Say so plainly and go to §6 — do not invent a mechanism.

## 5. Read hang signatures correctly

- **Heartbeat-monitor and store `recvBytes` failures** appear during *teardown*
  of a collective hang. They are the symptom: ranks lost contact. They do not
  say which collective, or why.
- **`Watchdog caught collective operation timeout`** is a real collective
  timeout and names the operation and rank. Far more informative.
- **No communication-layer messages at all** means the hang is above it — a
  Python-level deadlock, a server that never bound, a wedged actor. `py-spy` is
  then the only thing that will tell you.

## 6. When the evidence is insufficient, prescribe instrumentation

A legitimate and common outcome. Say the cause is undetermined, then make the
next occurrence explicable:

- Confirm the flight recorder was enabled (`COLL_TRACE`, on by default in
  `launch_experiment.sh`) and that `./nccl_traces/` was populated. A dump names
  the stuck collective; without one, a hang can only be localised, never
  explained.
- Raise communication-library verbosity **for the reproduction only**
  (`NCCL_DEBUG=INFO`). It is too verbose to leave on: it prints per-communicator
  setup for every rank.
- Reproduce at the smallest scale that still fails, and at the largest scale
  known to work, to tighten the matrix in §3.

## 7. Write the handoff file

Write `reports/hangs/<jobid>.md` so another session can act without re-deriving
anything. Required sections:

1. **What happened** — job id, allocation, elapsed, reported state *and* the
   real outcome, which often differ.
2. **Where it stopped** — the marker comparison against a healthy run.
3. **Discriminating matrix** — every run against the suspect axis.
4. **Evidence** — what the bundle showed, or explicitly that none was captured.
5. **Root cause** — with honest confidence. "Localised to worker
   initialisation; mechanism unknown pending traces" is a good finding. A guess
   dressed as a cause is not.
6. **Recommended next step** — a specific code fix with file and line, or the
   exact instrumented reproduction command. Include the command.
7. **Cost** — GPU-hours burned. It is what justifies the watchdog and the
   instrumentation.

## 8. Rules

- Report faithfully, including "I could not determine the cause".
- One observation is never a root cause; two configurations differing in one
  axis is a finding.
- Never propose a config change that has not been validated by whatever cheap
  preflight the repo provides — CPU minutes beat discovering it hours into a
  large allocation.
- Reproduce at the smallest scale and the fastest queue available. Wall-clock to
  first signal matters more than fidelity when you are still bisecting.
