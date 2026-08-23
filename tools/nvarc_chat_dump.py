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
"""Print a human-readable complete chat from NVARC/ARC run logs.

Three kinds:

- ``executor``: a real training rollout (NVARC rule + one grid -> answer)
  from the newest ``train_data_step*.jsonl`` dump.
- ``induction``: a real validation rollout on the official ARC-AGI-2
  evaluation split from the newest ``val_data_step*.jsonl`` dump.
- ``proposer``: the full refinement-episode chat (proposer session, fresh
  executor sessions, verification feedback, revision, test follow-up)
  rendered from a real ingested puzzle via the gym agent's own prompt
  builders. No proposer training runs exist yet, so the model turns are
  clearly marked placeholders; the prompts are exactly what the agent sends.

Examples:
    uv run tools/nvarc_chat_dump.py --kind executor
    uv run tools/nvarc_chat_dump.py --kind induction --pick best
    uv run tools/nvarc_chat_dump.py --kind proposer --data-dir <ingested-dir>
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
GYM_ROOT = REPO_ROOT / "3rdparty" / "Gym-workspace" / "Gym"

_TURN_RE = re.compile(r"<\|im_start\|>(\w+)\n(.*?)(?:<\|im_end\|>|$)", re.DOTALL)

_RULE = "=" * 70


def _print_turn(role: str, text: str) -> None:
    print(f"{_RULE}\n{role.upper()}\n{_RULE}")
    print(text.strip("\n"))
    print()


def render_templated_prompt(prompt: str) -> list[tuple[str, str]]:
    """Split a chat-templated prompt string into (role, text) turns.

    The dumps store the prompt after ``apply_chat_template``, so roles are
    embedded as template markers. Falls back to one ``user`` turn when no
    markers are present (a different template or a plain string).
    """
    turns = [(role, text) for role, text in _TURN_RE.findall(prompt)]
    return turns if turns else [("user", prompt)]


def newest_dump(log_dir: Path, prefix: str) -> Path:
    """Return the highest-step dump of the newest experiment directory."""
    exp_dirs = sorted(
        (d for d in log_dir.glob("exp_*") if d.is_dir()),
        key=lambda d: d.stat().st_mtime,
    )
    for exp_dir in reversed(exp_dirs):
        dumps = sorted(
            exp_dir.glob(f"{prefix}_step*.jsonl"),
            key=lambda f: int(f.stem.rsplit("step", 1)[1]),
        )
        if dumps:
            return dumps[-1]
    raise FileNotFoundError(f"no {prefix}_step*.jsonl under {log_dir}/exp_*")


def pick_row(rows: list[dict], pick: str, index: int | None) -> tuple[int, dict]:
    """Choose the sample to print: an explicit index, or best/worst/first reward."""
    if index is not None:
        return index, rows[index]
    if pick == "first":
        return 0, rows[0]
    chooser = max if pick == "best" else min
    position = chooser(range(len(rows)), key=lambda i: rows[i]["rewards"][0])
    return position, rows[position]


def _normalized_turns(items: list) -> list[tuple[str, str]]:
    """Normalize a logged conversation to (role, text) pairs.

    Training dumps store plain strings (prompt, response, environment
    message); validation dumps store role dicts. Both carry the templated
    prompt as the first user text.
    """
    turns: list[tuple[str, str]] = []
    for position, item in enumerate(items):
        if isinstance(item, dict):
            turns.append((str(item.get("role", "user")), str(item.get("content", ""))))
        elif position == 0:
            turns.append(("user", str(item)))
        elif position == 1:
            turns.append(("assistant", str(item)))
        else:
            turns.append(("environment", str(item)))
    return turns


def dump_logged_chat(log_dir: Path, prefix: str, pick: str, index: int | None) -> None:
    """Print one complete logged rollout: prompt turns, response, verdict."""
    dump = newest_dump(log_dir, prefix)
    rows = [json.loads(line) for line in dump.open()]
    position, row = pick_row(rows, pick, index)
    reward = row["rewards"][0]
    print(f"source: {dump}")
    print(f"sample: {position} of {len(rows)}   reward: {reward:+.3f}\n")

    for role, text in _normalized_turns(row["content"][0]):
        if "<|im_start|>" not in text:
            label = "assistant (model output)" if role == "assistant" else role
            _print_turn(label, text)
            continue
        for sub_role, sub_text in render_templated_prompt(text):
            if sub_role == "system" and not sub_text.strip():
                continue  # nano-v3's template emits an empty system turn
            if sub_role == "assistant":
                # The template ends with the generation opener the model
                # continues from (e.g. an opening think tag).
                _print_turn("assistant (generation prefix)", sub_text)
            else:
                _print_turn(sub_role, sub_text)


def _agent_instructions() -> dict[str, str]:
    """Read the agent's role-instruction constants from its source with ast."""
    import ast

    app_path = (
        GYM_ROOT / "responses_api_agents" / "arc_transform_refinement_agent" / "app.py"
    )
    wanted = {"PROPOSER_INSTRUCTIONS", "EXECUTOR_INSTRUCTIONS"}
    found: dict[str, str] = {}
    for node in ast.parse(app_path.read_text()).body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in wanted:
                    found[target.id] = ast.literal_eval(node.value)
    missing = wanted - set(found)
    if missing:
        raise RuntimeError(f"could not find {sorted(missing)} in {app_path}")
    return found


def dump_proposer_episode(data_dir: Path, seed: int) -> None:
    """Render a full refinement episode from a real proposer_eval puzzle.

    Prompts come from the gym agent's own builders; model turns are
    placeholders because no proposer runs exist yet.
    """
    # Deferred imports: the gym tree and pyarrow are only needed here.
    sys.path.insert(0, str(GYM_ROOT))
    import random

    import pyarrow.dataset as ds
    from resources_servers.arc_agi import (  # pyrefly: ignore[import-error]  resolved via the GYM_ROOT sys.path entry above
        logic,
    )

    # The agent's app.py imports the whole nemo_gym package (server deps this
    # tool does not need), so lift its two instruction constants from source.
    instructions = _agent_instructions()

    shards = sorted(data_dir.glob("data-*.parquet"))
    if not shards:
        raise FileNotFoundError(f"no data-*.parquet shards under {data_dir}")
    rows = (
        ds.dataset([str(s) for s in shards], format="parquet")
        .to_table(
            columns=["puzzle_id", "pairs_json", "difficulty"],
            filter=(ds.field("split") == "proposer_eval")
            & (ds.field("difficulty") <= 64),
        )
        .to_pylist()
    )
    row = random.Random(seed).choice(sorted(rows, key=lambda r: r["puzzle_id"]))
    pairs = json.loads(row["pairs_json"])
    train_pairs, test_pair = pairs[:2], pairs[2]
    placeholder = "[model output would appear here]"

    print(f"puzzle: {row['puzzle_id']} (proposer_eval, difficulty {row['difficulty']})")
    print("Prompts are the agent's real ones; assistant turns are placeholders.\n")

    print("################ PROPOSER SESSION (persistent) ################\n")
    _print_turn("system", instructions["PROPOSER_INSTRUCTIONS"])
    _print_turn(
        "user",
        logic.build_proposer_prompt(
            train_pairs=train_pairs, test_inputs=[test_pair["input"]]
        ),
    )
    _print_turn("assistant", f"{placeholder}\n<transform_description>...rule...")

    print("########## EXECUTOR SESSION (fresh chat per call) ##########\n")
    _print_turn("system", instructions["EXECUTOR_INSTRUCTIONS"])
    _print_turn(
        "user",
        logic.build_executor_prompt(
            description="...rule text from the proposer, verbatim...",
            inputs={
                "train_1": train_pairs[0]["input"],
                "train_2": train_pairs[1]["input"],
            },
            tag="train_predictions",
        ),
    )
    _print_turn("assistant", f"{placeholder}\n<train_predictions>{{...}}")

    # Show the deterministic feedback with one deliberately wrong prediction.
    wrong = [list(cells) for cells in train_pairs[1]["output"]]
    wrong[0][0] = (wrong[0][0] + 1) % 10
    verification = logic.verify_predictions(
        predictions={"train_1": train_pairs[0]["output"], "train_2": wrong},
        correct={
            "train_1": train_pairs[0]["output"],
            "train_2": train_pairs[1]["output"],
        },
    )

    print("############ PROPOSER SESSION, revision turn ############\n")
    _print_turn("user", logic.build_revision_prompt(verification.feedback))
    _print_turn("assistant", f"{placeholder}\n<transform_description>...revised...")

    print("###### EXECUTOR SESSION (fresh), test follow-up ######\n")
    _print_turn(
        "user",
        logic.build_test_followup_prompt(test_inputs={"test_1": test_pair["input"]}),
    )
    _print_turn("assistant", f"{placeholder}\n<answers>{{...}}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--kind", choices=("executor", "induction", "proposer"), default="executor"
    )
    parser.add_argument("--log-dir", type=Path, default=REPO_ROOT / "logs")
    parser.add_argument(
        "--pick",
        choices=("best", "worst", "first"),
        default="best",
        help="which logged sample to print, by reward",
    )
    parser.add_argument("--index", type=int, default=None)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="ingested NVARC parquet dir (proposer kind only)",
    )
    parser.add_argument("--seed", type=int, default=0, help="proposer puzzle choice")
    args = parser.parse_args()

    if args.kind == "proposer":
        data_dir: Path | None = args.data_dir
        if data_dir is None:
            parser.error("--kind proposer requires --data-dir")
        else:
            dump_proposer_episode(data_dir, args.seed)
    else:
        prefix = "train_data" if args.kind == "executor" else "val_data"
        dump_logged_chat(args.log_dir, prefix, args.pick, args.index)


if __name__ == "__main__":
    main()
