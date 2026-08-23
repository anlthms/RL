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
"""Offline NVARC ingestion: join artifact rule text to synthetic grid pairs.

Joins the NVARC Artifacts Puzzles ``mixtures`` table to the NVARC Synthetic
Puzzles grid files and emits one parquet dataset with a stable, versioned
schema, split by puzzle id into train / executor-val / proposer-eval pools.

The join key is the ordered seed-task pair: a synthetic puzzle file
``<p1>_<p2>.json`` matches the mixtures row with ``puzzle_name1 == p1`` and
``puzzle_name2 == p2``. Measured on the 2025-11-13 Kaggle releases this join
covers 100% of the 103,223 puzzle files; 150 pairs have two mixture rows and
are deduplicated deterministically (see ``select_mixture_row``).

Deliberately runnable outside the repo venv (stdlib + pyarrow only, tokenizers
optional) so it can execute on a CPU partition without the training container.
Grid constraints are therefore restated here rather than imported from
``nemo_rl.environments.arc_agi_grid``; a unit test pins the two definitions
together.
"""

import argparse
import hashlib
import itertools
import json
import multiprocessing
import random
import re
import sys
import time
from pathlib import Path
from typing import Any

# Must match nemo_rl.environments.arc_agi_grid.MAX_GRID_DIM / NUM_COLORS
# (pinned by tests/unit/tools/test_nvarc_ingest.py).
MAX_GRID_DIM = 30
NUM_COLORS = 10

SCHEMA_VERSION = 1

# The canonical proposer<->executor rule schema, in render order.
# ``input_generation`` is parsed and kept as metadata but excluded here: it
# describes how synthetic inputs were sampled, not how to transform one.
CANONICAL_SECTIONS = (
    "rules_summary",
    "solution_steps",
    "key_insight",
    "puzzle_concepts",
)
ALL_SECTIONS = CANONICAL_SECTIONS + ("input_generation",)

_NEW_PUZZLE_RE = re.compile(r"<new_puzzle>(.*?)</new_puzzle>", re.DOTALL)
_SECTION_RES = {
    name: re.compile(rf"<{name}>\s*(.*?)\s*</{name}>", re.DOTALL)
    for name in ALL_SECTIONS
}


def parse_new_puzzle_sections(completion: str) -> dict[str, str] | None:
    """Extract the puzzle-description sections from a mixtures completion.

    Prefers content inside the final ``<new_puzzle>`` block (the completion
    opens with a ``<puzzle_analysis>`` scratchpad that may mention section
    tags); falls back to the whole completion when the wrapper is absent.
    When a tag repeats, the last occurrence wins — the same final-answer
    convention the answer-grid parser uses.

    Returns None unless every canonical section is present and non-empty.
    """
    blocks = _NEW_PUZZLE_RE.findall(completion)
    text = blocks[-1] if blocks else completion

    sections: dict[str, str] = {}
    for name in ALL_SECTIONS:
        matches = _SECTION_RES[name].findall(text)
        if matches:
            sections[name] = matches[-1]
    if any(not sections.get(name) for name in CANONICAL_SECTIONS):
        return None
    sections.setdefault("input_generation", "")
    return sections


def render_canonical_rule(sections: dict[str, str]) -> str:
    """Render parsed sections into the canonical 4-section rule text."""
    return "\n\n".join(
        f"<{name}>\n{sections[name]}\n</{name}>" for name in CANONICAL_SECTIONS
    )


def select_mixture_row(candidates: list[dict]) -> dict:
    """Pick one mixtures row for a puzzle deterministically.

    A handful of join pairs (150 of 266,443 measured) have two mixture rows.
    Order by content hash — stable across shard enumeration order — so the
    ingested dataset does not depend on filesystem listing order.
    """
    return min(
        candidates,
        key=lambda row: hashlib.sha256(row["completion"].encode()).hexdigest(),
    )


def validate_grid(grid: object) -> bool:
    """Check one grid: rectangular, non-empty, ints 0-9, dims <= 30."""
    if not isinstance(grid, list) or not grid or len(grid) > MAX_GRID_DIM:
        return False
    width: int | None = None
    for row in grid:
        if not isinstance(row, list) or not row or len(row) > MAX_GRID_DIM:
            return False
        if width is None:
            width = len(row)
        elif len(row) != width:
            return False
        for cell in row:
            # bool is an int subclass; JSON true/false must not pass as 1/0.
            if isinstance(cell, bool) or not isinstance(cell, int):
                return False
            if not 0 <= cell < NUM_COLORS:
                return False
    return True


def load_puzzle_pairs(path: Path) -> tuple[list[dict], int]:
    """Read one puzzle JSON file and return (valid pairs, dropped count)."""
    with open(path) as handle:
        raw = json.load(handle)
    valid = [
        pair
        for pair in raw
        if isinstance(pair, dict)
        and validate_grid(pair.get("input"))
        and validate_grid(pair.get("output"))
    ]
    return valid, len(raw) - len(valid)


def puzzle_difficulty(pairs: list[dict]) -> tuple[int, int]:
    """Return (max h*w, max dim) over every grid in the puzzle.

    Maximum area over all grids is the proposal's initial difficulty proxy;
    the maximum single dimension is kept alongside it because sequence-length
    budgeting cares about the serialized row count, not the area.
    """
    difficulty = 0
    max_dim = 0
    for pair in pairs:
        for grid in (pair["input"], pair["output"]):
            height, width = len(grid), len(grid[0])
            difficulty = max(difficulty, height * width)
            max_dim = max(max_dim, height, width)
    return difficulty, max_dim


def assign_splits(
    puzzle_ids: list[str],
    *,
    seed: int,
    val_count: int,
    proposer_eval_count: int,
) -> dict[str, str]:
    """Assign every puzzle id to train / executor_val / proposer_eval.

    Splitting is by puzzle id — never by pair — so no rule text seen in
    training can describe a validation puzzle. Ids are sorted before
    shuffling so the assignment is independent of enumeration order.
    """
    if val_count + proposer_eval_count >= len(puzzle_ids):
        raise ValueError(
            f"cannot hold out {val_count} + {proposer_eval_count} puzzles "
            f"from {len(puzzle_ids)}"
        )
    shuffled = sorted(puzzle_ids)
    random.Random(seed).shuffle(shuffled)
    splits = {}
    for index, puzzle_id in enumerate(shuffled):
        if index < val_count:
            splits[puzzle_id] = "executor_val"
        elif index < val_count + proposer_eval_count:
            splits[puzzle_id] = "proposer_eval"
        else:
            splits[puzzle_id] = "train"
    return splits


def _percentiles(values: list, points=(50, 90, 99, 100)) -> dict[str, float]:
    if not values:
        return {}
    ordered = sorted(values)
    return {
        f"p{point}": ordered[min(len(ordered) - 1, len(ordered) * point // 100)]
        for point in points
    }


def _worker(task: tuple[str, str, str]) -> tuple[str, str, list[dict], int] | None:
    """Pool worker: read + validate one puzzle file (I/O bound, main merges)."""
    puzzle_id, source, path = task
    try:
        pairs, dropped = load_puzzle_pairs(Path(path))
    except (json.JSONDecodeError, OSError):
        return None
    return puzzle_id, source, pairs, dropped


def _arrow_schema():
    # Deferred import so the pure functions above stay importable (and unit
    # testable) without pyarrow installed.
    import pyarrow as pa

    return pa.schema(
        [
            ("puzzle_id", pa.string()),
            ("source", pa.string()),
            ("seed_puzzle_1", pa.string()),
            ("seed_puzzle_2", pa.string()),
            ("split", pa.string()),
            ("canonical_rule", pa.string()),
            ("rules_summary", pa.string()),
            ("solution_steps", pa.string()),
            ("key_insight", pa.string()),
            ("puzzle_concepts", pa.string()),
            ("input_generation", pa.string()),
            ("model_name", pa.string()),
            ("reasoning_level", pa.string()),
            ("pairs_json", pa.string()),
            ("num_pairs", pa.int32()),
            ("num_dropped_pairs", pa.int32()),
            ("difficulty", pa.int32()),
            ("max_dim", pa.int32()),
        ],
        metadata={"schema_version": str(SCHEMA_VERSION)},
    )


def _load_mixture_index(artifacts_dir: Path, wanted: set[str]) -> dict[str, dict]:
    """Load mixtures rows for the wanted puzzle ids, deduplicating pairs."""
    import pyarrow.parquet as pq

    columns = [
        "puzzle_name1",
        "puzzle_name2",
        "model_name",
        "reasoning_level",
        "completion",
    ]
    candidates: dict[str, list[dict]] = {}
    shards = sorted((artifacts_dir / "mixtures").glob("*.parquet"))
    if not shards:
        raise FileNotFoundError(f"no mixtures shards under {artifacts_dir}")
    for shard in shards:
        for row in pq.read_table(shard, columns=columns).to_pylist():
            puzzle_id = f"{row['puzzle_name1']}_{row['puzzle_name2']}"
            if puzzle_id in wanted:
                candidates.setdefault(puzzle_id, []).append(row)
    return {
        puzzle_id: select_mixture_row(rows) for puzzle_id, rows in candidates.items()
    }


def _token_stats(rows_sample: list[dict], tokenizer_path: Path) -> dict:
    """Token-length percentiles under the policy tokenizer (optional).

    Measures the canonical rule alone and the full executor task body (rule +
    the puzzle's largest serialized input grid) — the number that actually
    drives the sequence-length budget.
    """
    from tokenizers import Tokenizer  # optional dependency, checked by caller

    tokenizer = Tokenizer.from_file(str(tokenizer_path / "tokenizer.json"))
    rule_tokens, body_tokens = [], []
    for row in rows_sample:
        pairs = json.loads(row["pairs_json"])
        biggest = max(
            (pair["input"] for pair in pairs),
            key=lambda grid: len(grid) * len(grid[0]),
        )
        serialized = "\n".join(" ".join(str(c) for c in line) for line in biggest)
        body = (
            f"<transformation>\n{row['canonical_rule']}\n</transformation>\n"
            f"<input>\n{serialized}\n</input>"
        )
        rule_tokens.append(len(tokenizer.encode(row["canonical_rule"]).ids))
        body_tokens.append(len(tokenizer.encode(body).ids))
    return {
        "sampled_puzzles": len(rows_sample),
        "canonical_rule_tokens": _percentiles(rule_tokens),
        "task_body_tokens_largest_input": _percentiles(body_tokens),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts-dir", type=Path, required=True)
    parser.add_argument("--puzzles-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split-seed", type=int, default=20260822)
    parser.add_argument("--val-puzzles", type=int, default=500)
    parser.add_argument("--proposer-eval-puzzles", type=int, default=1000)
    parser.add_argument(
        "--min-pairs",
        type=int,
        default=2,
        help="drop puzzles with fewer valid pairs (need >=1 train + 1 held out)",
    )
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--rows-per-shard", type=int, default=8192)
    parser.add_argument(
        "--tokenizer-path",
        type=Path,
        default=None,
        help="HF checkpoint dir with tokenizer.json; enables token stats",
    )
    parser.add_argument(
        "--token-stat-sample",
        type=int,
        default=5000,
        help="puzzles sampled for token stats (plus every puzzle with "
        "difficulty >= 750, the sequence-budget tail)",
    )
    parser.add_argument(
        "--exclude-seed-ids",
        type=Path,
        default=None,
        help="file of ARC task ids (one per line); puzzles seeded from any of "
        "them are quarantined into split=excluded",
    )
    args = parser.parse_args()

    import pyarrow as pa
    import pyarrow.parquet as pq

    excluded_seeds: set[str] = set()
    if args.exclude_seed_ids:
        excluded_seeds = {
            line.strip()
            for line in args.exclude_seed_ids.read_text().splitlines()
            if line.strip()
        }

    started = time.time()
    tasks: list[tuple[str, str, str]] = []
    for source in ("nvarc_full", "nvarc_training"):
        source_dir = args.puzzles_dir / source
        if not source_dir.is_dir():
            continue
        for path in sorted(source_dir.glob("*/*.json")):
            tasks.append((path.stem, source, str(path)))
    if not tasks:
        raise FileNotFoundError(f"no puzzle files under {args.puzzles_dir}")
    print(f"puzzle files: {len(tasks)}", flush=True)

    mixture_index = _load_mixture_index(args.artifacts_dir, {t[0] for t in tasks})
    print(
        f"mixture rows joined: {len(mixture_index)} ({time.time() - started:.0f}s)",
        flush=True,
    )

    stats: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "split_seed": args.split_seed,
        "min_pairs": args.min_pairs,
        "puzzle_files": len(tasks),
        "joined": 0,
        "missing_artifact": 0,
        "rule_parse_failures": 0,
        "unreadable_files": 0,
        "too_few_pairs": 0,
        "excluded_by_seed_id": 0,
        "dropped_pairs": 0,
        "kept": 0,
        "kept_by_split": {},
        "kept_by_source": {},
    }

    # Parse every joined rule up front: split assignment must run over the
    # final kept-id set, and rule parsing is the only non-grid filter.
    rules: dict[str, dict] = {}
    for puzzle_id, row in mixture_index.items():
        sections = parse_new_puzzle_sections(row["completion"])
        if sections is None:
            stats["rule_parse_failures"] += 1
            continue
        rules[puzzle_id] = {
            "sections": sections,
            "model_name": row["model_name"],
            "reasoning_level": row["reasoning_level"],
        }
    stats["missing_artifact"] = len(tasks) - len(mixture_index)

    # Pass 1 over grid files: validate, so splits are assigned over exactly
    # the puzzles that will be emitted.
    grid_info: dict[str, tuple[str, list[dict], int]] = {}
    with multiprocessing.Pool(args.workers) as pool:
        for result in pool.imap_unordered(_worker, tasks, chunksize=64):
            if result is None:
                stats["unreadable_files"] += 1
                continue
            puzzle_id, source, pairs, dropped = result
            stats["dropped_pairs"] += dropped
            if puzzle_id not in rules:
                continue
            if len(pairs) < args.min_pairs:
                stats["too_few_pairs"] += 1
                continue
            grid_info[puzzle_id] = (source, pairs, dropped)
    stats["joined"] = len(grid_info)
    print(
        f"validated puzzles: {len(grid_info)} ({time.time() - started:.0f}s)",
        flush=True,
    )

    quarantined = {
        puzzle_id
        for puzzle_id in grid_info
        if excluded_seeds & set(puzzle_id.split("_"))
    }
    stats["excluded_by_seed_id"] = len(quarantined)
    splits = assign_splits(
        [puzzle_id for puzzle_id in grid_info if puzzle_id not in quarantined],
        seed=args.split_seed,
        val_count=args.val_puzzles,
        proposer_eval_count=args.proposer_eval_puzzles,
    )
    splits.update({puzzle_id: "excluded" for puzzle_id in quarantined})

    args.output_dir.mkdir(parents=True, exist_ok=True)
    schema = _arrow_schema()
    shard_counter = itertools.count()
    batch: list[dict] = []
    difficulties: list[int] = []
    pair_counts: list[int] = []
    rule_chars: list[int] = []

    def flush(force: bool = False) -> None:
        nonlocal batch
        if not batch or (len(batch) < args.rows_per_shard and not force):
            return
        pq.write_table(
            pa.Table.from_pylist(batch, schema=schema),
            args.output_dir / f"data-{next(shard_counter):05d}.parquet",
            compression="zstd",
        )
        batch = []

    sample_rows: list[dict] = []
    sampler = random.Random(args.split_seed)
    for puzzle_id in sorted(grid_info):
        source, pairs, dropped = grid_info[puzzle_id]
        rule = rules[puzzle_id]
        difficulty, max_dim = puzzle_difficulty(pairs)
        seed_1, seed_2 = puzzle_id.split("_")
        canonical_rule = render_canonical_rule(rule["sections"])
        row = {
            "puzzle_id": puzzle_id,
            "source": source,
            "seed_puzzle_1": seed_1,
            "seed_puzzle_2": seed_2,
            "split": splits[puzzle_id],
            "canonical_rule": canonical_rule,
            **{name: rule["sections"][name] for name in ALL_SECTIONS},
            "model_name": rule["model_name"],
            "reasoning_level": rule["reasoning_level"],
            "pairs_json": json.dumps(pairs, separators=(",", ":")),
            "num_pairs": len(pairs),
            "num_dropped_pairs": dropped,
            "difficulty": difficulty,
            "max_dim": max_dim,
        }
        batch.append(row)
        flush()
        stats["kept"] += 1
        stats["kept_by_split"][row["split"]] = (
            stats["kept_by_split"].get(row["split"], 0) + 1
        )
        stats["kept_by_source"][source] = stats["kept_by_source"].get(source, 0) + 1
        difficulties.append(difficulty)
        pair_counts.append(len(pairs))
        rule_chars.append(len(canonical_rule))
        if (
            len(sample_rows) < args.token_stat_sample
            and sampler.random() < args.token_stat_sample / len(grid_info)
        ) or difficulty >= 750:
            sample_rows.append({k: row[k] for k in ("canonical_rule", "pairs_json")})
    flush(force=True)

    stats["difficulty"] = _percentiles(difficulties)
    stats["num_pairs"] = _percentiles(pair_counts)
    stats["canonical_rule_chars"] = _percentiles(rule_chars)
    if args.tokenizer_path:
        stats["tokens"] = _token_stats(sample_rows, args.tokenizer_path)
    stats["elapsed_seconds"] = round(time.time() - started)

    with open(args.output_dir / "stats.json", "w") as handle:
        json.dump(stats, handle, indent=2)
    print(json.dumps(stats, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
