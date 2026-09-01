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
"""Materialize deployment-loop validation rows: gym ``candidate_select``.

One row per real evaluation test pair, run through the deployment protocol:
the agent samples ``num_candidates`` independent rules from the demo prompt,
the server scores every candidate on EVERY demo (exact count first, then
cell/format tie-breaks), ``/select_candidate`` picks the winner WITHOUT test
access, and only the selected rule touches the test grid (finalize
re-verifies the selection). This measures the deployable system - proposal
diversity + demo-fitness selection - rather than single-shot induction.

Schema mirrors the cotrain hidden_test validation rows; only ``protocol``
and ``num_candidates`` differ.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tools.arc_sampling_harness import load_arc_rows

PROPOSER_AGENT = "arc_transform_refinement_agent"


def build_rows(
    sources: list[dict], *, num_candidates: int, model_context_limit: int
) -> list[dict]:
    """One candidate_select row per real test pair."""
    return [
        {
            "responses_create_params": {"input": []},
            "agent_ref": {"type": "responses_api_agents", "name": PROPOSER_AGENT},
            "protocol": "candidate_select",
            "num_candidates": num_candidates,
            "model_context_limit": model_context_limit,
            "train": source["train_pairs"],
            "test": [{"input": source["test_input"], "output": source["target"]}],
            "task_id": source["task_id"],
            "role": "candidate_select",
        }
        for source in sources
    ]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arc-data-path", required=True)
    parser.add_argument("--split", default="evaluation")
    parser.add_argument("--num-candidates", type=int, default=16)
    parser.add_argument(
        "--model-context-limit",
        type=int,
        default=19456,
        help="engine window minus generation headroom (validation is untrained)",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    sources = load_arc_rows(args.arc_data_path, args.split)
    rows = build_rows(
        sources,
        num_candidates=args.num_candidates,
        model_context_limit=args.model_context_limit,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with open(args.output_dir / "val.jsonl", "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    stats = {
        "rows": len(rows),
        "tasks": len({row["task_id"] for row in rows}),
        "num_candidates": args.num_candidates,
        "model_context_limit": args.model_context_limit,
    }
    with open(args.output_dir / "stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    print(json.dumps(stats), flush=True)


if __name__ == "__main__":
    main()
