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
r"""Kill a Slurm job whose GPUs have gone idle, capturing evidence first.

A hung distributed job holds its whole allocation until something reaps it. On a
cluster whose idle-GPU reaper allows two hours, that is a large number of
GPU-hours spent producing nothing -- and the reaper takes the process state with
it, so the hang cannot be explained afterwards.

**GPU utilisation is the progress signal.** Not log output: a job can be quiet
for many minutes while doing real work (a long evaluation pass, a slow
checkpoint), and treating silence as failure kills healthy runs. Busy GPUs mean
progress no matter what is or is not being written to a log, which also keeps
this tool free of any knowledge of the application running inside the job.

Deliberately not an LLM: whether a number is below a threshold wants a predicate
that is reviewable, testable and identical on every poll, and ``scancel`` is
destructive. The judgement work afterwards -- root cause, code reading, what to
do next -- is escalated to an agent via the ``diagnose-hung-job`` skill.

Order of operations on a trip is **capture, then kill**, because ``scancel``
destroys the stacks, the GPU state and any in-memory trace buffers, which is
precisely the evidence worth having.

Usage:
    uv run tools/run_watchdog.py JOBID           # poll until the job ends
    uv run tools/run_watchdog.py JOBID --once    # single evaluation
"""

import argparse
import dataclasses
import re
import subprocess
import time
from pathlib import Path

# A GPU reporting at or below this percent of utilisation is doing nothing worth
# counting. Kernel launches and memory traffic keep a working GPU well above it.
G_BUSY_PERCENT = 5

# Fraction of the job's GPUs that must be busy for the job to count as making
# progress. Kept low: one rank still working means the job is not wedged, and a
# straggler pattern is not this tool's business.
G_BUSY_FRACTION = 0.05

# Consecutive idle polls before a hang is declared. Multiple polls rather than
# one because a single sample can land between phases -- after a checkpoint
# write, during a data reshuffle -- when the GPUs are legitimately quiet.
G_IDLE_POLLS = 4

# Minutes from job start during which idleness is ignored. A job's GPUs are
# legitimately idle while it starts up: pulling a container, converting weights,
# loading a model across ranks. That is indistinguishable from a hang by
# utilisation alone, and it is long -- a large model takes tens of minutes to
# reach its first GPU work. Without this the watchdog kills every job it is
# armed on, a few minutes after launch.
G_GRACE_MINUTES = 45

G_TERMINAL_STATES = frozenset(
    {"COMPLETED", "FAILED", "CANCELLED", "TIMEOUT", "NODE_FAIL", "PREEMPTED", "OUT_OF"}
)


@dataclasses.dataclass(frozen=True)
class Observation:
    """One poll's view of a job."""

    state: str
    busy_fraction: float | None
    nodes: list[str]

    @property
    def is_terminal(self) -> bool:
        return any(self.state.startswith(term) for term in G_TERMINAL_STATES)

    @property
    def is_idle(self) -> bool:
        """Idle means measured and low.

        An unavailable sample is not a vote. A probe that failed to run is no
        evidence that the GPUs are doing nothing, and treating it as such would
        let a transient ``srun`` failure kill a healthy job.
        """
        return self.busy_fraction is not None and self.busy_fraction < G_BUSY_FRACTION


def _run(command: list[str], timeout_s: int = 60) -> tuple[int, str]:
    """Run a command, returning its return code and combined output."""
    try:
        done = subprocess.run(
            command, capture_output=True, text=True, timeout=timeout_s, check=False
        )
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError) as error:
        return 1, f"{type(error).__name__}: {error}"
    return done.returncode, (done.stdout or "") + (done.stderr or "")


def job_state(job_id: str) -> str:
    """The Slurm state of a job, or ``UNKNOWN`` if it cannot be read."""
    code, out = _run(["sacct", "-j", job_id, "--format=State", "-nX"])
    if code != 0 or not out.strip():
        return "UNKNOWN"
    return out.strip().splitlines()[0].strip()


def job_elapsed_minutes(job_id: str) -> float | None:
    """Minutes since the job started running, or None if not started/unknown."""
    code, out = _run(["sacct", "-j", job_id, "--format=Start", "-nXP"])
    if code != 0 or not out.strip():
        return None
    started = out.strip().splitlines()[0].strip()
    if not started or started in {"Unknown", "None"}:
        return None
    try:
        start = time.mktime(time.strptime(started, "%Y-%m-%dT%H:%M:%S"))
    except ValueError:
        return None
    return (time.time() - start) / 60.0


def job_nodes(job_id: str) -> list[str]:
    """The job's allocated hostnames, expanded from Slurm's range notation."""
    code, out = _run(["sacct", "-j", job_id, "--format=NodeList%200", "-nX"])
    if code != 0 or not out.strip():
        return []
    node_list = out.strip().splitlines()[0].strip()
    if not node_list or node_list in {"None assigned", "(null)"}:
        return []
    code, expanded = _run(["scontrol", "show", "hostnames", node_list])
    if code != 0:
        return []
    return [line.strip() for line in expanded.splitlines() if line.strip()]


def _in_job(
    job_id: str, nodes: list[str], command: list[str], timeout_s: int
) -> tuple[int, str]:
    """Run a command on every node of a running job, one task per node.

    ``srun --overlap`` rather than ssh: it is the portable way onto a node the
    job already holds, and works where direct ssh to compute nodes is closed by
    the cluster's PAM or cgroup policy.
    """
    tasks = max(len(nodes), 1)
    return _run(
        [
            "srun",
            "--jobid",
            job_id,
            "--overlap",
            f"--nodes={tasks}",
            f"--ntasks={tasks}",
            "--ntasks-per-node=1",
            *command,
        ],
        timeout_s=timeout_s,
    )


def busy_fraction(job_id: str, nodes: list[str]) -> float | None:
    """Fraction of the job's GPUs above the busy threshold, or None if unknown."""
    if not nodes:
        return None
    code, out = _in_job(
        job_id,
        nodes,
        ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
        timeout_s=120,
    )
    if code != 0:
        return None
    readings = [int(m) for m in re.findall(r"^\s*(\d+)\s*$", out, flags=re.MULTILINE)]
    if not readings:
        return None
    return sum(1 for value in readings if value > G_BUSY_PERCENT) / len(readings)


def observe(job_id: str) -> Observation:
    """Take one reading of the job."""
    state = job_state(job_id)
    if any(state.startswith(term) for term in G_TERMINAL_STATES) or state == "UNKNOWN":
        return Observation(state, None, [])
    nodes = job_nodes(job_id)
    return Observation(state, busy_fraction(job_id, nodes), nodes)


def capture_evidence(
    job_id: str, out_dir: Path, observation: Observation, log_dir: Path | None
) -> Path:
    """Snapshot a live job's state before it is killed.

    Everything here is generic to a GPU Slurm job: process stacks, GPU state,
    open sockets. Nothing is read from, or assumed about, the application.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.txt").write_text(
        "\n".join(
            [
                f"job_id: {job_id}",
                f"slurm_state: {observation.state}",
                f"gpu_busy_fraction: {observation.busy_fraction}",
                f"idle_threshold: <{G_BUSY_FRACTION} of GPUs above {G_BUSY_PERCENT}%",
                f"nodes: {' '.join(observation.nodes)}",
                f"captured: {time.strftime('%Y-%m-%d %H:%M:%S')}",
            ]
        )
        + "\n"
    )

    probes: tuple[tuple[str, list[str], int], ...] = (
        ("nvidia-smi.txt", ["nvidia-smi"], 120),
        # py-spy against every python process on the node: which collective or
        # lock each rank is parked on is the single most useful artefact, and it
        # exists only while the processes do.
        (
            "py-spy.txt",
            [
                "bash",
                "-lc",
                'for p in $(pgrep -u "$(id -u)" python 2>/dev/null); do '
                'echo "=== pid $p"; '
                "py-spy dump --pid $p --nonblocking 2>&1 | head -80; done",
            ],
            300,
        ),
        ("sockets.txt", ["bash", "-lc", "ss -tnp 2>/dev/null | head -200"], 120),
        (
            "dmesg.txt",
            ["bash", "-lc", "dmesg --time-format iso 2>/dev/null | tail -60"],
            120,
        ),
    )
    for filename, command, timeout_s in probes:
        code, out = _in_job(job_id, observation.nodes, command, timeout_s)
        (out_dir / filename).write_text(f"# rc={code}\n{out}")

    if log_dir and log_dir.is_dir():
        for path in sorted(log_dir.glob("*.log")):
            code, tail = _run(["tail", "-n", "400", str(path)])
            (out_dir / f"tail-{path.name}").write_text(f"# rc={code}\n{tail}")
    return out_dir


def watch(
    job_id: str,
    *,
    interval_s: int,
    idle_polls: int,
    grace_minutes: float,
    once: bool,
    root: Path,
    log_dir: Path | None,
) -> int:
    """Poll a job until it ends or hangs. Returns a process exit code."""
    consecutive_idle = 0
    while True:
        observation = observe(job_id)
        stamp = time.strftime("%H:%M:%S")

        if observation.is_terminal:
            print(
                f"[{stamp}] {job_id} {observation.state}; watchdog exiting", flush=True
            )
            return 0

        elapsed = job_elapsed_minutes(job_id)
        in_grace = elapsed is not None and elapsed < grace_minutes

        if in_grace:
            # Startup: idle GPUs here mean the job is still loading, not stuck.
            consecutive_idle = 0
        elif observation.is_idle:
            consecutive_idle += 1
        elif observation.busy_fraction is not None:
            consecutive_idle = 0
        # An unknown reading leaves the counter untouched: it neither confirms
        # nor clears, so a flaky probe can neither trip nor reset the detector.

        busy = (
            "unknown"
            if observation.busy_fraction is None
            else f"{observation.busy_fraction:.0%}"
        )
        grace_note = f" grace({elapsed:.0f}/{grace_minutes:.0f}m)" if in_grace else ""
        print(
            f"[{stamp}] {job_id} state={observation.state} gpu_busy={busy} "
            f"idle_polls={consecutive_idle}/{idle_polls}{grace_note}",
            flush=True,
        )

        if consecutive_idle >= idle_polls:
            minutes = idle_polls * interval_s / 60
            print(
                f"HANG CONFIRMED: GPUs idle for {minutes:.0f} min. "
                "Capturing evidence before killing.",
                flush=True,
            )
            out_dir = capture_evidence(
                job_id, root / f"{job_id}-hang", observation, log_dir
            )
            print(f"evidence: {out_dir}", flush=True)
            code, out = _run(["scancel", job_id])
            print(f"scancel rc={code} {out.strip()}", flush=True)
            print(
                f"next: run the `diagnose-hung-job` skill against {out_dir}",
                flush=True,
            )
            return 2

        if once:
            return 0
        time.sleep(interval_s)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("job_id", help="Slurm job id to watch")
    parser.add_argument(
        "--interval", type=int, default=300, help="seconds between polls"
    )
    parser.add_argument(
        "--idle-polls",
        type=int,
        default=G_IDLE_POLLS,
        help="consecutive idle polls before declaring a hang",
    )
    parser.add_argument(
        "--root", type=Path, default=Path.cwd(), help="where to write evidence"
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=None,
        help="optional directory of *.log files to tail into the evidence bundle",
    )
    parser.add_argument(
        "--grace-minutes",
        type=float,
        default=G_GRACE_MINUTES,
        help="ignore idle GPUs for this long after the job starts (startup)",
    )
    parser.add_argument("--once", action="store_true", help="evaluate once and exit")
    args = parser.parse_args()

    raise SystemExit(
        watch(
            args.job_id,
            interval_s=args.interval,
            idle_polls=args.idle_polls,
            grace_minutes=args.grace_minutes,
            once=args.once,
            root=args.root,
            log_dir=args.log_dir,
        )
    )


if __name__ == "__main__":
    main()
