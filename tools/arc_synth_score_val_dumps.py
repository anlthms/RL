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
"""Re-score a synthetic-ladder run's validation dumps, per level, offline.

``tools/arc_agi_score_val_dumps.py`` can only score real ARC rows, because it
looks their targets up in the solutions file. Synthetic targets are not stored
anywhere -- but they do not need to be: a split is a pure function of its
``SynthCurriculumConfig``, so passing the run's recipe regenerates exactly the
rows it validated on, through the same code path the dataset used.

What it reports beyond ``grid_match``, and why each one earned its place:

- ``echoed`` and ``cell(echo)``. A ladder level whose transformation touches few
  cells is mostly solved by copying the input, and a run can converge on that
  while every other metric improves. Comparing ``cell(pred)`` against
  ``cell(echo)`` says whether the policy is beating the echo or has become it.
- ``shape_ok``. Separates "did not apply the rule" from "applied it and slipped":
  a rotation that does not transpose the output dimensions was never attempted.
- Area buckets. Exact match over hundreds of cells needs near-perfect per-cell
  accuracy, so a rule the model does understand can still score zero on big
  grids. Bucketing tells that apart from not knowing the rule.

Usage:
    uv run tools/arc_synth_score_val_dumps.py logs/exp_005 \
        --config examples/configs/async/nanov3_arcsynth_colocated.yaml
"""

import argparse
import json
from pathlib import Path

from omegaconf import OmegaConf

from nemo_rl.data.datasets.response_datasets.arc_synth import SynthCurriculumConfig
from nemo_rl.environments.arc_agi_grid import (
    best_alignment_cell_accuracy,
    extract_answer_grid,
    grid_shape,
)
from nemo_rl.utils.config import (
    load_config,
    parse_hydra_overrides,
    register_omegaconf_resolvers,
)

AREA_BUCKETS = ((0, 60), (60, 150), (150, 300), (300, 10**9))


def assistant_text(row: dict) -> str:
    """Pull the assistant turn out of a dumped validation row."""
    messages = row["content"][0] if isinstance(row["content"], list) else []
    return "".join(str(m["content"]) for m in messages if m.get("role") == "assistant")


def score(rows: list[dict], tasks: list, indices: list[int]) -> dict[str, float]:
    parsed = matched = echoed = shaped = 0
    cell_pred = cell_echo = 0.0
    for i in indices:
        task = tasks[i]
        cell_echo += best_alignment_cell_accuracy(task.test_input, task.target)
        pred = extract_answer_grid(assistant_text(rows[i]))
        if pred is None:
            continue
        parsed += 1
        matched += pred == task.target
        echoed += pred == task.test_input
        shaped += grid_shape(pred) == grid_shape(task.target)
        cell_pred += best_alignment_cell_accuracy(pred, task.target)
    n, p = len(indices), max(parsed, 1)
    return {
        "n": n,
        "grid": matched / n,
        "echoed": echoed / p,
        "cell_pred": cell_pred / p,
        "cell_echo": cell_echo / n,
        "shape_ok": shaped / p,
        "parsed": parsed / n,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("log_dir", help="run log dir holding val_data_step*.jsonl")
    parser.add_argument(
        "--config", required=True, help="run recipe used to generate validation"
    )
    parser.add_argument("--by-area", action="store_true")
    args, overrides = parser.parse_known_args()

    # Rebuild the run's held-out split from its own recipe, through the same
    # schema the dataset uses. Anything else is a second implementation of the
    # curriculum that silently drifts from the one that produced the dump.
    register_omegaconf_resolvers()
    config = load_config(args.config)
    if overrides:
        config = parse_hydra_overrides(config, overrides)
    resolved = OmegaConf.to_container(config, resolve=True)
    curriculum = SynthCurriculumConfig(**resolved["data"]["train"])
    levels = sorted(set(curriculum.levels))
    tasks = curriculum.val_tasks()
    dumps = sorted(
        Path(args.log_dir).glob("val_data_step*.jsonl"),
        key=lambda p: int(p.stem.removeprefix("val_data_step")),
    )
    if not dumps:
        raise SystemExit(f"no val_data_step*.jsonl under {args.log_dir}")

    header = f"{'step':>5} {'level':>6} {'n':>4} {'grid':>6} {'echoed':>7} {'cell(pred)':>11} {'cell(echo)':>11} {'shape_ok':>9} {'parsed':>7}"
    print(header)
    for dump in dumps:
        step = int(dump.stem.removeprefix("val_data_step"))
        rows = [json.loads(line) for line in dump.open()][: len(tasks)]
        if len(rows) < len(tasks):
            print(f"  !! {dump.name} has {len(rows)} rows, expected >= {len(tasks)}")
            continue
        for level in levels:
            idx = [i for i, t in enumerate(tasks) if t.level == level]
            s = score(rows, tasks, idx)
            print(
                f"{step:>5} {level:>6} {s['n']:>4} {s['grid']:>6.3f} {s['echoed']:>7.3f} "
                f"{s['cell_pred']:>11.3f} {s['cell_echo']:>11.3f} {s['shape_ok']:>9.3f} {s['parsed']:>7.3f}"
            )

    if args.by_area:
        print("\n=== final checkpoint by target area ===")
        rows = [json.loads(line) for line in dumps[-1].open()][: len(tasks)]
        print(
            f"{'level':>6} {'cells':>13} {'n':>4} {'grid':>6} {'echoed':>7} {'cell(pred)':>11} {'shape_ok':>9}"
        )
        for level in levels:
            for low, high in AREA_BUCKETS:
                idx = [
                    i
                    for i, t in enumerate(tasks)
                    if t.level == level
                    and low <= len(t.target) * len(t.target[0]) < high
                ]
                if not idx:
                    continue
                s = score(rows, tasks, idx)
                label = f"{low}-{high if high < 10**9 else '+'}"
                print(
                    f"{level:>6} {label:>13} {s['n']:>4} {s['grid']:>6.3f} {s['echoed']:>7.3f} "
                    f"{s['cell_pred']:>11.3f} {s['shape_ok']:>9.3f}"
                )


if __name__ == "__main__":
    main()
