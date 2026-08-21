#!/usr/bin/env python3
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
"""Compare async-GRPO throughput from unsampled W&B history.

Uses consumed-sample slope after warmup, with generated-token metrics.

Usage:
  python3 throughput_compare.py --runs colo:<id> nc:<id> --project entity/project
  python3 throughput_compare.py --runs colo:<id> nc:<id> --warmup-steps 5 --csv <dir>
Requires W&B credentials and network access.
"""

import argparse
import os
import statistics as st
from typing import Any

DEFAULT_PROJECT = "adlr/nemo-rl-megatron-backend"
# The training loop logs consumed_samples at step+1, so step 1 is the earliest
# anchor available; anchoring there also drops the compile-heavy first step.
DEFAULT_WARMUP_STEPS = 1
KEYS = [
    "_step",
    "train/consumed_samples",
    "train/elapsed_time_s",
    "train/mean_gen_tokens_per_sample",
    "timing/train/policy_training",
]


def fetch(*, project: str, run_id: str) -> tuple[list[dict[str, Any]], int | None]:
    import wandb

    run = wandb.Api(timeout=60).run(f"{project}/{run_id}")
    # Avoid endpoint bias from run.history downsampling.
    try:
        rows = list(run.scan_history(keys=KEYS, page_size=1000))
    except Exception:
        # A keyed scan needs a `_step` column, which runs that log against a
        # custom step_metric do not expose. Scan everything and project locally.
        rows = [
            {key: row[key] for key in KEYS if key in row}
            for row in run.scan_history(page_size=2000)
        ]
    cfg = run.config
    policy = cfg.get("policy")
    batch = policy.get("train_global_batch_size") if isinstance(policy, dict) else None
    batch = batch or cfg.get("policy.train_global_batch_size")
    return rows, batch


def summarize_rows(
    name: str,
    rows: list[dict[str, Any]],
    *,
    batch: int | None,
    warmup_steps: int,
) -> dict[str, Any]:
    """Summarize a run using the end-of-warmup point as the window anchor."""
    train = [
        row
        for row in rows
        if row.get("_step") is not None
        and row.get("train/consumed_samples") is not None
        and row.get("train/elapsed_time_s") is not None
    ]
    train.sort(key=lambda row: (row["_step"], row["train/elapsed_time_s"]))
    anchors = [row for row in train if row["_step"] == warmup_steps]
    measured = [row for row in train if row["_step"] > warmup_steps]
    if not anchors or not measured:
        raise ValueError(
            f"{name}: need an end-of-warmup point at step {warmup_steps} "
            "and at least one later training point"
        )

    first, last = anchors[-1], measured[-1]
    n_steps = last["_step"] - first["_step"]
    window_s = last["train/elapsed_time_s"] - first["train/elapsed_time_s"]
    consumed = last["train/consumed_samples"] - first["train/consumed_samples"]
    if n_steps <= 0 or window_s <= 0 or consumed <= 0:
        raise ValueError(
            f"{name}: non-positive measured window "
            f"(steps={n_steps}, seconds={window_s}, samples={consumed})"
        )

    def col(key: str) -> list[float]:
        return [row[key] for row in measured if row.get(key) is not None]

    policy_training = col("timing/train/policy_training")
    generation_lengths = col("train/mean_gen_tokens_per_sample")
    mean_generation_length = st.mean(generation_lengths) if generation_lengths else None
    samples_per_second = consumed / window_s

    return {
        "name": name,
        "n_steps": n_steps,
        "window_s": window_s,
        "steps_per_hour": n_steps / window_s * 3600,
        "batch": batch,
        "consumed_samples": consumed,
        "samples_per_hour": samples_per_second * 3600,
        "generated_tokens_per_second": (
            samples_per_second * mean_generation_length
            if mean_generation_length is not None
            else None
        ),
        "mean_generation_length": mean_generation_length,
        "sec_per_step": window_s / n_steps,
        "policy_median": st.median(policy_training) if policy_training else 0.0,
        "rows": [first, *measured],
    }


def summarize(
    name: str, *, project: str, run_id: str, warmup_steps: int
) -> dict[str, Any]:
    rows, batch = fetch(project=project, run_id=run_id)
    try:
        return summarize_rows(name, rows, batch=batch, warmup_steps=warmup_steps)
    except ValueError as exc:
        raise SystemExit(f"{exc} (run {run_id})") from exc


def parse_runs(items: list[str]) -> list[tuple[str, str]]:
    runs = []
    for item in items:
        if ":" not in item:
            raise SystemExit(f"--runs entry must be label:run_id, got {item!r}")
        label, run_id = item.split(":", 1)
        runs.append((label, run_id))
    return runs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--project", default=DEFAULT_PROJECT, help="W&B entity/project")
    parser.add_argument("--runs", nargs="+", required=True, metavar="LABEL:RUN_ID")
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=DEFAULT_WARMUP_STEPS,
        help="exclude this many completed steps (minimum 1)",
    )
    parser.add_argument("--csv", help="directory for per-step training-window CSVs")
    args = parser.parse_args()
    if args.warmup_steps < 1:
        parser.error("--warmup-steps must be at least 1")
    os.environ.setdefault("WANDB_SILENT", "true")

    runs = [
        summarize(
            label,
            project=args.project,
            run_id=run_id,
            warmup_steps=args.warmup_steps,
        )
        for label, run_id in parse_runs(args.runs)
    ]
    width = max(16, max(len(run["name"]) for run in runs) + 2)

    def row(label, value):
        return f"{label:<34}" + "".join(f"{value(run):>{width}}" for run in runs)

    print("\n=== Async-GRPO throughput (consumed samples / elapsed time) ===\n")
    print(row("", lambda run: run["name"]))
    print("-" * (34 + width * len(runs)))
    print(row("warmup steps excluded", lambda _: args.warmup_steps))
    print(row("training steps in window", lambda run: run["n_steps"]))
    print(row("training wallclock (s)", lambda run: f"{run['window_s']:.0f}"))
    print(row("consumed samples", lambda run: f"{run['consumed_samples']:.0f}"))
    print(
        row(
            "train_global_batch_size",
            lambda run: f"{run['batch']}" if run["batch"] else "?",
        )
    )
    print(row("steps / hour", lambda run: f"{run['steps_per_hour']:.1f}"))
    print(row("SAMPLES / HOUR", lambda run: f"{run['samples_per_hour']:.0f}"))
    print(
        row(
            "GENERATED TOKENS / SECOND",
            lambda run: (
                f"{run['generated_tokens_per_second']:.0f}"
                if run["generated_tokens_per_second"] is not None
                else "?"
            ),
        )
    )
    print(
        row(
            "mean generation length",
            lambda run: (
                f"{run['mean_generation_length']:.1f}"
                if run["mean_generation_length"] is not None
                else "?"
            ),
        )
    )
    print(row("sec / step", lambda run: f"{run['sec_per_step']:.1f}"))
    print("-" * (34 + width * len(runs)))
    print(
        row(
            "train compute s/step (median)",
            lambda run: f"{run['policy_median']:.1f}",
        )
    )

    if len(runs) == 2:
        ratio = runs[0]["samples_per_hour"] / runs[1]["samples_per_hour"]
        faster = runs[0]["name"] if ratio > 1 else runs[1]["name"]
        print(
            f"\n{runs[0]['name']} / {runs[1]['name']} samples-per-hour = "
            f"{ratio:.3f} ({faster} faster by {abs(ratio - 1) * 100:.1f}%)"
        )

    if args.csv:
        import csv

        os.makedirs(args.csv, exist_ok=True)
        for run in runs:
            path = os.path.join(args.csv, f"runtime_{run['name']}.csv")
            with open(path, "w", newline="") as file:
                writer = csv.writer(file)
                writer.writerow(["step", "elapsed_time_s", "consumed_samples"])
                for history_row in run["rows"]:
                    writer.writerow(
                        [
                            history_row["_step"],
                            f"{history_row['train/elapsed_time_s']:.3f}",
                            history_row["train/consumed_samples"],
                        ]
                    )
            print(f"wrote {path}")


if __name__ == "__main__":
    main()
