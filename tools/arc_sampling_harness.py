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
"""Shared resumable ARC sampling harness for gates A0, A1, and A4.

Samples K single-shot induction candidates per (task, test row, view) from an
OpenAI-compatible endpoint, scores every candidate by EXACT grid match in the
canonical frame, and streams one JSONL record per candidate so every metric
can be recomputed offline without another model run. Solve metrics are derived
only from the per-candidate ``grid_match`` booleans -- shaped similarity terms
are recorded as diagnostics and never feed pass@k.

- A0 (train-subset calibration): ``--split training --num-tasks N``, with
  ``--source-ids label=ids.json`` intersecting the manifest against every
  SFT/RL seed source so overlapping tasks are labeled, not silently trusted.
- A1 (latent ceiling): ``--split evaluation`` at temperature ~1.0 with K up to
  64; the report carries pass@1/8/32/64, solved IDs, and bootstrap CIs.
- A4 (augmentation invariance): ``--num-views V`` adds D8+color views; every
  prediction is inverted to the canonical frame before scoring, agreement, or
  voting (see tools/arc_augment.py).

Rows render exactly like the co-training validation induction rows: the
``examples/prompts/arc_agi.txt`` template around
``format_task_prompt(train_pairs, test_input)``.

Restarting with the same ``--output-dir`` resumes: completed
(task, test, view, candidate) keys are skipped.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import random
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from nemo_rl.environments.arc_agi_grid import (
    Grid,
    best_alignment_cell_accuracy,
    extract_answer_grid,
    format_task_prompt,
    grid_shape,
)
from tools.arc_augment import ArcView, sample_views
from tools.nvarc_executor_benchmark import ExecutorClient, OpenAIChatCompletionsClient

DEFAULT_PASS_AT = (1, 8, 32, 64)
BOOTSTRAP_RESAMPLES = 2000
WILSON_Z = 1.959964  # 95%


@dataclass(frozen=True)
class HarnessConfig:
    """Resolved CLI controls recorded beside the results."""

    base_url: str
    model: str
    data_dir: str
    split: str
    seed: int
    num_tasks: int | None
    num_candidates: int
    pass_at: tuple[int, ...]
    temperature: float
    max_output_tokens: int
    concurrency: int
    timeout_seconds: float
    num_views: int
    fix_background: bool
    color_permutations: bool
    prompt_file: str
    store_responses: bool
    source_ids_labels: tuple[str, ...] = ()


@dataclass(frozen=True)
class ManifestRow:
    """One test row of one selected puzzle, with its canonical grids."""

    task_id: str
    test_index: int
    train_pairs: list[dict[str, Grid]]
    test_input: Grid
    target: Grid
    seen_in_sources: tuple[str, ...]


def load_arc_rows(data_dir: str, split: str) -> list[dict[str, Any]]:
    """Join an ARC Prize challenges/solutions pair, one row per test pair.

    Standalone re-read of the same files
    ``nemo_rl.data.datasets.response_datasets.arc_agi._load_split`` consumes,
    so the harness stays importable without the training-data dependencies.
    """
    prefix = Path(data_dir) / f"arc-agi_{split}"
    with open(f"{prefix}_challenges.json", encoding="utf-8") as f:
        challenges = json.load(f)
    with open(f"{prefix}_solutions.json", encoding="utf-8") as f:
        solutions = json.load(f)
    rows = []
    for task_id in sorted(challenges):
        task = challenges[task_id]
        task_solutions = solutions[task_id]
        if len(task_solutions) != len(task["test"]):
            raise ValueError(
                f"task {task_id} has {len(task['test'])} test inputs but "
                f"{len(task_solutions)} solutions"
            )
        for test_index, (test_pair, solution) in enumerate(
            zip(task["test"], task_solutions)
        ):
            rows.append(
                {
                    "task_id": task_id,
                    "test_index": test_index,
                    "train_pairs": task["train"],
                    "test_input": test_pair["input"],
                    "target": solution,
                }
            )
    return rows


def load_source_ids(spec: str) -> tuple[str, set[str]]:
    """Parse one ``label=path.json`` source-seed spec into (label, task ids).

    The JSON may be an ARC challenges dict (task ids are the keys) or a plain
    list of task ids.
    """
    label, separator, path = spec.partition("=")
    if not separator or not label or not path:
        raise ValueError(f"--source-ids must look like label=path.json, got {spec!r}")
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        return label, set(payload)
    if isinstance(payload, list):
        return label, {str(item) for item in payload}
    raise ValueError(f"source-id file {path} must hold a dict or a list of task ids")


def build_manifest(
    rows: list[dict[str, Any]],
    *,
    num_tasks: int | None,
    seed: int,
    source_ids: dict[str, set[str]],
) -> tuple[list[ManifestRow], dict[str, Any]]:
    """Fix the evaluated population by puzzle ID and label source overlap.

    Sampling is by task, never by row, and every test row of a selected task
    is included -- a puzzle is only comparable to the deployment metric when
    all of its test grids are on the table. The published manifest carries the
    exact IDs so the run is auditable and reproducible.
    """
    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_task[row["task_id"]].append(row)
    task_ids = sorted(by_task)
    if num_tasks is not None and num_tasks < len(task_ids):
        task_ids = sorted(random.Random(seed).sample(task_ids, num_tasks))

    manifest_rows: list[ManifestRow] = []
    overlap: dict[str, list[str]] = {label: [] for label in source_ids}
    for task_id in task_ids:
        seen = tuple(
            sorted(label for label, ids in source_ids.items() if task_id in ids)
        )
        for label in seen:
            overlap[label].append(task_id)
        for row in sorted(by_task[task_id], key=lambda r: r["test_index"]):
            manifest_rows.append(
                ManifestRow(
                    task_id=task_id,
                    test_index=row["test_index"],
                    train_pairs=row["train_pairs"],
                    test_input=row["test_input"],
                    target=row["target"],
                    seen_in_sources=seen,
                )
            )
    seen_tasks = sorted({r.task_id for r in manifest_rows if r.seen_in_sources})
    manifest = {
        "seed": seed,
        "num_tasks_requested": num_tasks,
        "task_ids": task_ids,
        "task_count": len(task_ids),
        "row_count": len(manifest_rows),
        "seen_task_ids": seen_tasks,
        "clean_task_count": len(task_ids) - len(seen_tasks),
        "source_overlap": overlap,
    }
    return manifest_rows, manifest


def render_prompt(template: str, row: ManifestRow, view: ArcView) -> str:
    """Render one induction prompt in the view frame.

    Identical bytes to the co-training validation induction rows when the view
    is the identity.
    """
    return template.format(
        format_task_prompt(
            view.apply_pairs(row.train_pairs), view.apply_grid(row.test_input)
        )
    )


def score_candidate(
    *,
    response: str,
    row: ManifestRow,
    view: ArcView,
) -> dict[str, Any]:
    """Score one raw response in the CANONICAL frame.

    The prediction is parsed in the view frame and inverted before any
    comparison; the returned record carries no target grids.
    """
    view_prediction = extract_answer_grid(response)
    prediction = view.invert_grid(view_prediction)
    if prediction is None:
        format_valid = False
        grid_match = False
        shape_match = False
        cell_match = 0.0
    else:
        format_valid = True
        grid_match = prediction == row.target
        shape_match = grid_shape(prediction) == grid_shape(row.target)
        cell_match = best_alignment_cell_accuracy(prediction, row.target)
    return {
        "task_id": row.task_id,
        "test_index": row.test_index,
        "view_id": view.view_id,
        "format_valid": format_valid,
        "grid_match": grid_match,
        "shape_match": shape_match,
        "cell_match": cell_match,
        "copied_input": bool(format_valid and prediction == row.test_input),
        "prediction": prediction,
        "response_chars": len(response),
        "seen_in_sources": list(row.seen_in_sources),
    }


def record_key(record: dict[str, Any]) -> tuple[str, int, str, int]:
    return (
        record["task_id"],
        int(record["test_index"]),
        record["view_id"],
        int(record["candidate_index"]),
    )


def load_completed(candidates_path: Path) -> set[tuple[str, int, str, int]]:
    """Load resume keys, tolerating a truncated trailing line from a wall kill."""
    completed: set[tuple[str, int, str, int]] = set()
    if not candidates_path.exists():
        return completed
    with open(candidates_path, encoding="utf-8") as f:
        for line in f:
            try:
                completed.add(record_key(json.loads(line)))
            except (json.JSONDecodeError, KeyError):
                continue
    return completed


async def run_sampling(
    *,
    config: HarnessConfig,
    manifest_rows: list[ManifestRow],
    views: list[ArcView],
    client: ExecutorClient,
    candidates_path: Path,
) -> int:
    """Sample every missing (row, view, candidate) and stream records to JSONL."""
    template = Path(config.prompt_file).read_text(encoding="utf-8")
    completed = load_completed(candidates_path)
    jobs = [
        (row, view, candidate_index)
        for row in manifest_rows
        for view in views
        for candidate_index in range(config.num_candidates)
        if (row.task_id, row.test_index, view.view_id, candidate_index) not in completed
    ]
    print(
        f"{len(completed)} candidate records already on disk; sampling "
        f"{len(jobs)} remaining",
        flush=True,
    )
    semaphore = asyncio.Semaphore(config.concurrency)
    sink = open(candidates_path, "a", encoding="utf-8")
    progress = {"written": 0}

    async def one(row: ManifestRow, view: ArcView, candidate_index: int) -> None:
        prompt = render_prompt(template, row, view)
        async with semaphore:
            # Bounded retry: a transient endpoint failure must not kill a
            # multi-hour run; a persistent one still fails loudly. Completed
            # records are already on disk either way, so a rerun resumes.
            response: str | None = None
            started = time.perf_counter()
            for attempt in range(3):
                started = time.perf_counter()
                try:
                    response = await client.complete(
                        [{"role": "user", "content": prompt}]
                    )
                except Exception as error:
                    if attempt == 2:
                        raise
                    print(
                        f"retrying {row.task_id}:{row.test_index} candidate "
                        f"{candidate_index} after: {error}",
                        flush=True,
                    )
                    await asyncio.sleep(5 * (attempt + 1))
                else:
                    break
            assert response is not None  # the loop either bound it or raised
            latency = time.perf_counter() - started
        record = score_candidate(response=response, row=row, view=view)
        record["candidate_index"] = candidate_index
        record["latency_seconds"] = latency
        if config.store_responses:
            record["response"] = response
        # Single-threaded event loop: writes are not interleaved. Flush per
        # record so a wall kill loses at most the in-flight candidates.
        sink.write(json.dumps(record) + "\n")
        sink.flush()
        progress["written"] += 1
        if progress["written"] % 100 == 0:
            print(f"{progress['written']}/{len(jobs)} candidates sampled", flush=True)

    try:
        await asyncio.gather(*(one(*job) for job in jobs))
    finally:
        sink.close()
    return progress["written"]


def pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased pass@k estimator from n samples with c exact hits (Chen 2021)."""
    if k > n:
        raise ValueError(f"pass@{k} needs at least {k} samples, got {n}")
    if n - c < k:
        return 1.0
    # Exact integer combinatorics: lgamma would leave 1e-16 residue at c=0,
    # and "no hits" must report exactly zero.
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)


def wilson_interval(successes: int, total: int) -> tuple[float, float]:
    """95% Wilson score interval for a binomial fraction."""
    if total == 0:
        return 0.0, 0.0
    z2 = WILSON_Z**2
    center = (successes + z2 / 2) / (total + z2)
    margin = (
        WILSON_Z
        * math.sqrt(successes * (total - successes) / total + z2 / 4)
        / (total + z2)
    )
    return max(0.0, center - margin), min(1.0, center + margin)


def _bootstrap_ci(per_row_values: list[float], *, seed: int) -> tuple[float, float]:
    """95% bootstrap CI of the mean over rows."""
    if not per_row_values:
        return 0.0, 0.0
    rng = random.Random(seed)
    means = sorted(
        sum(rng.choice(per_row_values) for _ in per_row_values) / len(per_row_values)
        for _ in range(BOOTSTRAP_RESAMPLES)
    )
    return (
        means[int(0.025 * len(means))],
        means[min(len(means) - 1, int(0.975 * len(means)))],
    )


@dataclass
class _RowStats:
    """Exact-hit tally for one (task, test, view) cell of the solve matrix."""

    n: int = 0
    hits: int = 0
    format_valid: int = 0
    cell_sum: float = 0.0
    hit_candidates: list[int] = field(default_factory=list)


def _view_summary(
    cells: dict[tuple[str, int], _RowStats],
    *,
    pass_at: tuple[int, ...],
    seed: int,
) -> dict[str, Any]:
    """Aggregate one view's solve matrix into pass@k, solved IDs, and CIs."""
    rows = sorted(cells)
    summary: dict[str, Any] = {
        "rows": len(rows),
        "candidates": sum(cells[key].n for key in rows),
        "format_valid": (
            sum(cells[key].format_valid for key in rows)
            / max(1, sum(cells[key].n for key in rows))
        ),
        "mean_cell_match": (
            sum(cells[key].cell_sum for key in rows)
            / max(1, sum(cells[key].n for key in rows))
        ),
    }
    solved_rows = [key for key in rows if cells[key].hits > 0]
    summary["solved_rows"] = len(solved_rows)
    summary["solved_row_ids"] = [f"{task}:{index}" for task, index in solved_rows]
    low, high = wilson_interval(len(solved_rows), len(rows))
    summary["any_hit_fraction"] = len(solved_rows) / len(rows) if rows else 0.0
    summary["any_hit_wilson_95"] = [low, high]

    # Task-level oracle: a task is solved when EVERY one of its test rows has
    # at least one exact candidate.
    task_rows: dict[str, list[bool]] = defaultdict(list)
    for task, index in rows:
        task_rows[task].append(cells[(task, index)].hits > 0)
    solved_tasks = sorted(
        task for task, flags in task_rows.items() if flags and all(flags)
    )
    summary["tasks"] = len(task_rows)
    summary["solved_task_ids"] = solved_tasks

    summary["pass_at"] = {}
    for k in pass_at:
        eligible = [key for key in rows if cells[key].n >= k]
        if not eligible:
            continue
        values = [pass_at_k(cells[key].n, cells[key].hits, k) for key in eligible]
        low, high = _bootstrap_ci(values, seed=seed + k)
        summary["pass_at"][str(k)] = {
            "estimate": sum(values) / len(values),
            "bootstrap_95": [low, high],
            "rows": len(eligible),
        }
        if len(eligible) < len(rows):
            summary["pass_at"][str(k)]["rows_skipped_below_k_samples"] = len(
                rows
            ) - len(eligible)
    return summary


def summarize(
    records: list[dict[str, Any]],
    *,
    manifest_rows: list[ManifestRow],
    pass_at: tuple[int, ...],
    seed: int,
) -> dict[str, Any]:
    """Recompute every solve metric from the candidate records alone.

    All pass@k values derive from the per-candidate ``grid_match`` boolean
    matrix; shaped diagnostics (cell match, format validity) are reported
    beside them and never counted as solves.
    """
    seen_tasks = {row.task_id for row in manifest_rows if row.seen_in_sources}
    per_view: dict[str, dict[tuple[str, int], _RowStats]] = {}
    for record in records:
        cells = per_view.setdefault(record["view_id"], {})
        stats = cells.setdefault(
            (record["task_id"], int(record["test_index"])), _RowStats()
        )
        stats.n += 1
        stats.format_valid += int(bool(record["format_valid"]))
        stats.cell_sum += float(record["cell_match"])
        if record["grid_match"]:
            stats.hits += 1
            stats.hit_candidates.append(int(record["candidate_index"]))

    report: dict[str, Any] = {"views": {}}
    for view_id, cells in sorted(per_view.items()):
        view_report = {
            "all_rows": _view_summary(cells, pass_at=pass_at, seed=seed),
        }
        clean_cells = {
            key: stats for key, stats in cells.items() if key[0] not in seen_tasks
        }
        if len(clean_cells) < len(cells):
            view_report["clean_rows_only"] = _view_summary(
                clean_cells, pass_at=pass_at, seed=seed
            )
        report["views"][view_id] = view_report

    if len(per_view) > 1:
        report["voting"] = _voting_summary(
            records, per_view=per_view, pass_at=pass_at, seed=seed
        )
    return report


def _voting_summary(
    records: list[dict[str, Any]],
    *,
    per_view: dict[str, dict[tuple[str, int], _RowStats]],
    pass_at: tuple[int, ...],
    seed: int,
) -> dict[str, Any]:
    """Cross-view agreement and canonical-frame voting (A4).

    Ballot ``i`` for a row collects candidate ``i`` from every view -- all
    already inverted to the canonical frame at scoring time -- and majority
    votes. The voter sees only predictions; targets enter only when the voted
    grid is scored, which happens against per-candidate ``grid_match`` ==
    prediction equality established at record time.
    """
    from tools.arc_augment import vote_canonical

    by_ballot: dict[tuple[str, int, int], dict[str, dict[str, Any]]] = defaultdict(dict)
    for record in records:
        key = (
            record["task_id"],
            int(record["test_index"]),
            int(record["candidate_index"]),
        )
        by_ballot[key][record["view_id"]] = record

    view_ids = sorted(per_view)
    voted_cells: dict[tuple[str, int], _RowStats] = defaultdict(_RowStats)
    agreement_sum = 0.0
    ballots = 0
    exact_by_view: dict[str, list[bool]] = {view_id: [] for view_id in view_ids}
    for (task_id, test_index, _candidate), view_records in sorted(by_ballot.items()):
        if len(view_records) < len(view_ids):
            continue  # incomplete ballot (mid-resume); skip rather than bias
        ordered = [view_records[view_id] for view_id in view_ids]
        voted, stats = vote_canonical([r["prediction"] for r in ordered])
        # The voted grid is exact iff it equals a prediction that was exact.
        voted_exact = any(r["grid_match"] and r["prediction"] == voted for r in ordered)
        cell = voted_cells[(task_id, test_index)]
        cell.n += 1
        cell.format_valid += int(voted is not None)
        if voted_exact:
            cell.hits += 1
        agreement_sum += stats["agreement"]
        ballots += 1
        for view_id in view_ids:
            exact_by_view[view_id].append(bool(view_records[view_id]["grid_match"]))

    error_correlation: dict[str, float] = {}
    for i, left in enumerate(view_ids):
        for right in view_ids[i + 1 :]:
            error_correlation[f"{left}|{right}"] = _pearson(
                exact_by_view[left], exact_by_view[right]
            )
    return {
        "ballots": ballots,
        "mean_vote_agreement": agreement_sum / ballots if ballots else 0.0,
        "voted": _view_summary(voted_cells, pass_at=pass_at, seed=seed),
        "exact_correlation_between_views": error_correlation,
    }


def _pearson(left: list[bool], right: list[bool]) -> float:
    if not left or len(left) != len(right):
        return 0.0
    n = len(left)
    mean_l = sum(left) / n
    mean_r = sum(right) / n
    cov = sum((l - mean_l) * (r - mean_r) for l, r in zip(left, right)) / n
    var_l = sum((l - mean_l) ** 2 for l in left) / n
    var_r = sum((r - mean_r) ** 2 for r in right) / n
    if var_l == 0.0 or var_r == 0.0:
        return 0.0
    return cov / math.sqrt(var_l * var_r)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:10240/v1")
    parser.add_argument("--model", required=True)
    parser.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY", "EMPTY"))
    parser.add_argument("--data-dir", required=True, help="ARC Prize directory")
    parser.add_argument("--split", default="evaluation")
    parser.add_argument("--seed", type=int, default=20_260_830)
    parser.add_argument(
        "--num-tasks", type=int, default=None, help="sample this many puzzles by ID"
    )
    parser.add_argument("--num-candidates", "-k", type=int, default=64)
    parser.add_argument(
        "--pass-at",
        type=lambda v: tuple(int(x) for x in v.split(",") if x),
        default=DEFAULT_PASS_AT,
    )
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max-output-tokens", type=int, default=8192)
    parser.add_argument("--concurrency", type=int, default=32)
    parser.add_argument("--timeout-seconds", type=float, default=600.0)
    parser.add_argument("--num-views", type=int, default=1)
    parser.add_argument("--fix-background", action="store_true", default=True)
    parser.add_argument(
        "--no-fix-background", dest="fix_background", action="store_false"
    )
    parser.add_argument(
        "--no-color-permutations", dest="color_permutations", action="store_false"
    )
    parser.add_argument("--prompt-file", default="examples/prompts/arc_agi.txt")
    parser.add_argument("--store-responses", action="store_true")
    parser.add_argument(
        "--source-ids",
        action="append",
        default=[],
        help="label=path.json task-id sets that seeded SFT/RL data (repeatable)",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    config = HarnessConfig(
        base_url=args.base_url,
        model=args.model,
        data_dir=args.data_dir,
        split=args.split,
        seed=args.seed,
        num_tasks=args.num_tasks,
        num_candidates=args.num_candidates,
        pass_at=tuple(k for k in args.pass_at if k <= args.num_candidates),
        temperature=args.temperature,
        max_output_tokens=args.max_output_tokens,
        concurrency=args.concurrency,
        timeout_seconds=args.timeout_seconds,
        num_views=args.num_views,
        fix_background=args.fix_background,
        color_permutations=args.color_permutations,
        prompt_file=args.prompt_file,
        store_responses=args.store_responses,
        source_ids_labels=tuple(args.source_ids),
    )
    source_ids = dict(load_source_ids(spec) for spec in args.source_ids)
    rows = load_arc_rows(config.data_dir, config.split)
    manifest_rows, manifest = build_manifest(
        rows, num_tasks=config.num_tasks, seed=config.seed, source_ids=source_ids
    )
    views = sample_views(
        count=config.num_views,
        seed=config.seed,
        include_identity=True,
        color_permutations=config.color_permutations,
        fix_background=config.fix_background,
    )
    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"manifest: {manifest['task_count']} tasks / {manifest['row_count']} rows; "
        f"seen in sources: {len(manifest['seen_task_ids'])}",
        flush=True,
    )

    client = OpenAIChatCompletionsClient(
        base_url=config.base_url,
        model=config.model,
        api_key=args.api_key,
        max_output_tokens=config.max_output_tokens,
        temperature=config.temperature,
        timeout_seconds=config.timeout_seconds,
    )
    candidates_path = output_dir / "candidates.jsonl"
    asyncio.run(
        run_sampling(
            config=config,
            manifest_rows=manifest_rows,
            views=views,
            client=client,
            candidates_path=candidates_path,
        )
    )

    with open(candidates_path, encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]
    report = {
        "config": asdict(config),
        "manifest": manifest,
        "view_ids": [view.view_id for view in views],
        **summarize(
            records,
            manifest_rows=manifest_rows,
            pass_at=config.pass_at,
            seed=config.seed,
        ),
    }
    (output_dir / "report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    for view_id, view_report in report["views"].items():
        overall = view_report["all_rows"]
        printable = {k: round(v["estimate"], 4) for k, v in overall["pass_at"].items()}
        print(
            f"[{view_id}] rows={overall['rows']} pass@k={printable} "
            f"solved_rows={overall['solved_rows']} "
            f"solved_tasks={len(overall['solved_task_ids'])}",
            flush=True,
        )


if __name__ == "__main__":
    main()
