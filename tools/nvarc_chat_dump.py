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
- ``proposer``: a real trained proposer trajectory from the newest
  ``train_data_step*.jsonl`` dump, decoded from its token ids (proposer
  rows log empty ``content``), with the loss-masked trainable span printed
  separately. Requires ``--tokenizer``.

Examples:
    uv run tools/nvarc_chat_dump.py --kind executor
    uv run tools/nvarc_chat_dump.py --kind induction --pick best
    uv run tools/nvarc_chat_dump.py --kind proposer --tokenizer <model-dir>
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

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


def dump_real_proposer_chat(
    log_dir: Path,
    pick: str,
    index: int | None,
    tokenizer_path: str,
    agent_name: str = "arc_transform_refinement_agent",
) -> None:
    """Print a real trained proposer trajectory decoded from its token ids.

    Proposer rows log empty ``content`` (the episode spans gym sub-sessions),
    so the chat is recovered from the trainable trajectory itself:
    ``token_ids`` decoded with the run's tokenizer. The loss-masked span
    (the final proposer turn, per the trainable-trajectory contract) is
    printed again separately so the trained tokens are unambiguous.
    """
    from transformers import AutoTokenizer  # deferred: needs the run container

    dump = newest_dump(log_dir, "train_data")
    rows = [json.loads(line) for line in dump.open()]
    filtered = [
        (position, row)
        for position, row in enumerate(rows)
        if row.get("agent_ref") and row["agent_ref"][0].get("name") == agent_name
    ]
    if not filtered:
        raise SystemExit(f"no {agent_name} rows in {dump}")

    if index is not None:
        position, row = filtered[index]
    elif pick == "first":
        position, row = filtered[0]
    else:
        chooser = max if pick == "best" else min
        position, row = chooser(filtered, key=lambda item: item[1]["rewards"][0])

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    token_ids = row["token_ids"][0]
    loss_mask = row["token_loss_mask"][0]
    trainable = int(sum(loss_mask))
    print(f"source: {dump}")
    print(
        f"sample: row {position} of {len(rows)} ({len(filtered)} {agent_name} rows)   "
        f"reward: {row['rewards'][0]:+.3f}   "
        f"tokens: {len(token_ids)} ({trainable} trainable)\n"
    )

    for role, text in render_templated_prompt(tokenizer.decode(token_ids)):
        if role == "system" and not text.strip():
            continue
        _print_turn(role, text)

    spans: list[list[int]] = []
    previous_masked = False
    for token_id, masked in zip(token_ids, loss_mask):
        if masked:
            if not previous_masked:
                spans.append([])
            spans[-1].append(token_id)
        previous_masked = bool(masked)
    for span in spans:
        _print_turn("trainable span (loss-masked tokens)", tokenizer.decode(span))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--kind",
        choices=("executor", "induction", "proposer"),
        default="executor",
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
        "--tokenizer",
        default=None,
        help="tokenizer path for decoding token_ids (proposer kind only)",
    )
    args = parser.parse_args()

    if args.kind == "proposer":
        if args.tokenizer is None:
            parser.error("--kind proposer requires --tokenizer")
        dump_real_proposer_chat(args.log_dir, args.pick, args.index, args.tokenizer)
    else:
        prefix = "train_data" if args.kind == "executor" else "val_data"
        dump_logged_chat(args.log_dir, prefix, args.pick, args.index)


if __name__ == "__main__":
    main()
