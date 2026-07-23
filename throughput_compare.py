#!/usr/bin/env python3
"""Compare async-GRPO training throughput (samples/hour) across runs.

Companion to launch_experiment.sh: quantifies the colocated-vs-non-colocated gap for the
CMP=1 comparison runs. Throughput comes from wandb's per-step `_runtime` over the training
window -- excludes startup and captures the buffer-empty waits the per-step timer misses.
samples/hour = steps/hour * train_global_batch_size keeps mixed-batch runs comparable. Also
reports the exposed-generation lull share (non-overlapped inference wait: large when
generation time-shares GPUs, ~0 when it overlaps on dedicated GPUs).

Usage:
  python3 throughput_compare.py                         # the CMP=1 comparison runs
  python3 throughput_compare.py --runs colo:<id> nc:<id> --project entity/project
  python3 throughput_compare.py --csv <dir>             # also dump per-step CSVs
Requires wandb creds (NETRC / WANDB_API_KEY) and network access.
"""

import argparse
import os
import statistics as st

DEFAULT_PROJECT = "adlr/nemo-rl-megatron-backend"
# The CMP=1 comparison runs (colocated 5493479 / non-colocated 5493294).
DEFAULT_RUNS = ["colocated:i5uqv8r4", "non-colocated:mu3r2kui"]
KEYS = [
    "_runtime",
    "_step",
    "timing/train/total_step_time",
    "timing/train/exposed_generation",
    "timing/train/policy_training",
]


def fetch(project, run_id):
    import wandb

    run = wandb.Api(timeout=60).run(f"{project}/{run_id}")
    rows = list(run.history(keys=KEYS, pandas=False, samples=100000))
    rows = [r for r in rows if r.get("_runtime") is not None and r.get("_step") is not None]
    rows.sort(key=lambda r: r["_step"])
    # train_global_batch_size (config layout varies: nested dict or dotted key).
    cfg = run.config
    pol = cfg.get("policy")
    batch = (pol or {}).get("train_global_batch_size") if isinstance(pol, dict) else None
    batch = batch or cfg.get("policy.train_global_batch_size")
    return rows, batch


def summarize(name, project, run_id):
    rows, batch = fetch(project, run_id)
    # Training window = steps >= 1 (step 0 is the setup/timing log emitted post-startup).
    train = [r for r in rows if r["_step"] >= 1]
    if len(train) < 2:
        raise SystemExit(f"{name}: too few training steps in history for {run_id}")
    first, last = train[0], train[-1]
    n_steps = last["_step"] - first["_step"]
    window_s = last["_runtime"] - first["_runtime"]

    def col(key):
        return [r[key] for r in train if r.get(key) is not None]

    tst, exg, pol = col(KEYS[2]), col(KEYS[3]), col(KEYS[4])
    return {
        "name": name,
        "n_steps": n_steps,
        "window_s": window_s,
        "steps_per_hour": n_steps / window_s * 3600,
        "batch": batch,
        "samples_per_hour": (n_steps / window_s * 3600 * batch) if batch else None,
        "sec_per_step": window_s / n_steps,
        "startup_s": first["_runtime"],
        "expgen_share": (sum(exg) / sum(tst)) if tst else 0.0,
        "expgen_median": st.median(exg) if exg else 0.0,
        "expgen_max": max(exg) if exg else 0.0,
        "policy_median": st.median(pol) if pol else 0.0,
        "rows": train,
    }


def parse_runs(items):
    runs = []
    for it in items:
        if ":" not in it:
            raise SystemExit(f"--runs entry must be label:run_id, got {it!r}")
        label, rid = it.split(":", 1)
        runs.append((label, rid))
    return runs


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--project", default=DEFAULT_PROJECT, help="wandb entity/project")
    ap.add_argument("--runs", nargs="+", default=DEFAULT_RUNS, metavar="LABEL:RUN_ID")
    ap.add_argument("--csv", help="dir to write per-step (step, runtime_s) CSVs for plotting")
    args = ap.parse_args()
    os.environ.setdefault("WANDB_SILENT", "true")

    runs = [summarize(label, args.project, rid) for label, rid in parse_runs(args.runs)]
    width = max(16, max(len(r["name"]) for r in runs) + 2)

    def row(label, fn):
        return f"{label:<34}" + "".join(f"{fn(r):>{width}}" for r in runs)

    print("\n=== Training-step throughput (from wandb _runtime) ===\n")
    print(row("", lambda r: r["name"]))
    print("-" * (34 + width * len(runs)))
    print(row("training steps in window", lambda r: r["n_steps"]))
    print(row("training wallclock (s)", lambda r: f"{r['window_s']:.0f}"))
    print(row("train_global_batch_size", lambda r: f"{r['batch']}" if r["batch"] else "?"))
    print(row("steps / hour", lambda r: f"{r['steps_per_hour']:.1f}"))
    print(row("SAMPLES / HOUR", lambda r: f"{r['samples_per_hour']:.0f}" if r["samples_per_hour"] else "?"))
    print(row("sec / step", lambda r: f"{r['sec_per_step']:.1f}"))
    print(row("startup to 1st step (s)", lambda r: f"{r['startup_s']:.0f}"))
    print("-" * (34 + width * len(runs)))
    print(row("exposed-gen lull share", lambda r: f"{100 * r['expgen_share']:.1f}%"))
    print(row("  lull s/step (median | max)", lambda r: f"{r['expgen_median']:.1f} | {r['expgen_max']:.0f}"))
    print(row("train compute s/step (median)", lambda r: f"{r['policy_median']:.1f}"))

    # samples/hour is batch-invariant; steps/hour only compares at equal batch size.
    if len(runs) == 2 and all(r["samples_per_hour"] for r in runs):
        ratio = runs[0]["samples_per_hour"] / runs[1]["samples_per_hour"]
        faster = runs[0]["name"] if ratio > 1 else runs[1]["name"]
        print(f"\n{runs[0]['name']} / {runs[1]['name']} samples-per-hour = {ratio:.3f}  "
              f"({faster} faster by {abs(ratio - 1) * 100:.1f}%)")

    if args.csv:
        import csv
        for r in runs:
            path = os.path.join(args.csv, f"runtime_{r['name']}.csv")
            with open(path, "w", newline="") as fh:
                w = csv.writer(fh)
                w.writerow(["step", "runtime_s", "total_step_time_s", "exposed_generation_s"])
                for row_ in r["rows"]:
                    w.writerow([row_["_step"], f"{row_['_runtime']:.1f}",
                                row_.get(KEYS[2], ""), row_.get(KEYS[3], "")])
            print(f"wrote {path}  (plot runtime_s vs step -> the staircase)")


if __name__ == "__main__":
    main()
