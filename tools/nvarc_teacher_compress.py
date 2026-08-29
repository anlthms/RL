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
"""Compress the trained-turn CoT of verified SFT rows (anti-runaway pass).

SFT v4 taught max-effort exploratory reasoning whose *style* self-perpetuates
at inference: the policy stops when it solves and thinks to the output cap
when it cannot (88-95% format failure in loop-val). This pass rewrites only
the FINAL assistant turn's think block: the teacher compresses its own
verified reasoning into a short, decisive trace, and the payload (rule or
answer) is kept byte-identical, so no re-verification is needed.

Rows where compression fails validation fall back to the original
(``"compressed": false``), so the output is always a complete dataset.
Splits are preserved by compressing train and val files separately.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
from pathlib import Path

_THINK_RE = re.compile(r"^<think>\n(.*)\n</think>\n\n(.*)$", re.DOTALL)
_FORBIDDEN = ("<think>", "</think>", "<answer>", "<rules_summary>")

COMPRESS_INSTRUCTIONS = (
    "Below is a verified chain of reasoning that led to a correct answer for a "
    "grid-transformation task, followed by that answer. Rewrite ONLY the "
    "reasoning as a compressed trace of at most {budget} characters: keep the "
    "decisive observations, checks, and the commitment to the answer; drop "
    "exploration, backtracking, and hedging ('wait', 'alternatively', "
    "'let me reconsider'). Write it as terse working notes that end decisively. "
    "Return ONLY the compressed reasoning text: no tags, no headers, and do not "
    "restate the answer.\n\n"
    "REASONING:\n{cot}\n\n"
    "ANSWER (context only, do not include or restate it):\n{payload}"
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--input", required=True, help="SFT JSONL to compress")
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--max-cot-chars",
        type=int,
        default=3600,
        help="~1.2k tokens: the compressed-think budget",
    )
    parser.add_argument("--samples-per-row", type=int, default=2)
    parser.add_argument("--reasoning-effort", default="low")
    parser.add_argument("--max-output-tokens", type=int, default=4096)
    parser.add_argument("--request-timeout", type=float, default=600.0)
    parser.add_argument("--temperature", type=float, default=0.3)
    parser.add_argument("--concurrency", type=int, default=32)
    parser.add_argument(
        "--min-original-chars",
        type=int,
        default=1200,
        help="rows with shorter thinks are already decisive; passed through",
    )
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="append to an existing output, skipping row indices already done",
    )
    return parser.parse_args()


def _make_http_complete(args: argparse.Namespace):
    import aiohttp

    headers = {"Authorization": f"Bearer {os.environ.get(args.api_key_env, 'EMPTY')}"}
    url = args.endpoint.rstrip("/") + "/chat/completions"
    timeout = aiohttp.ClientTimeout(total=args.request_timeout)

    async def complete(prompt: str) -> str:
        payload = {
            "model": args.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": args.temperature,
            "max_tokens": args.max_output_tokens,
        }
        if args.reasoning_effort:
            payload["reasoning_effort"] = args.reasoning_effort
        for attempt in range(3):
            try:
                async with aiohttp.ClientSession(timeout=timeout) as session:
                    async with session.post(url, json=payload, headers=headers) as r:
                        r.raise_for_status()
                        body = await r.json()
                return body["choices"][0]["message"].get("content") or ""
            except (aiohttp.ClientError, asyncio.TimeoutError, KeyError) as error:
                print(f"compress call failed (attempt {attempt + 1}): {error!r}", flush=True)
                await asyncio.sleep(10 * (attempt + 1))
        return ""

    return complete


async def compress_row(row: dict, *, args: argparse.Namespace, complete) -> dict:
    final = row["messages"][-1]["content"]
    match = _THINK_RE.match(final)
    if match is None:
        row["compressed"] = False
        return row
    cot, payload = match.group(1), match.group(2)
    if len(cot) <= args.min_original_chars:
        row["compressed"] = "passthrough"
        return row
    prompt = COMPRESS_INSTRUCTIONS.format(
        budget=args.max_cot_chars, cot=cot, payload=payload
    )
    for _ in range(args.samples_per_row):
        compressed = (await complete(prompt)).strip()
        if (
            compressed
            and len(compressed) <= args.max_cot_chars
            and not any(tag in compressed for tag in _FORBIDDEN)
        ):
            new_row = dict(row)
            new_row["messages"] = row["messages"][:-1] + [
                {
                    "role": "assistant",
                    "content": f"<think>\n{compressed}\n</think>\n\n{payload}",
                }
            ]
            new_row["compressed"] = True
            return new_row
    row["compressed"] = False
    return row


async def run(args: argparse.Namespace) -> None:
    """Stream compressed rows to disk as they land (wall-kill loses nothing).

    Output rows carry ``row_index`` (their line number in the input file), so
    a killed shard resumes with ``--resume`` and the final dataset is the
    concatenation of all shard outputs, order-independent.
    """
    all_rows = [json.loads(line) for line in open(args.input, encoding="utf-8")]
    todo = [
        (index, row)
        for index, row in enumerate(all_rows)
        if index % args.num_shards == args.shard_index
    ]
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    done_indices: set[int] = set()
    if args.resume and output.exists():
        for line in open(output, encoding="utf-8"):
            done_indices.add(json.loads(line)["row_index"])
        todo = [(i, r) for i, r in todo if i not in done_indices]
    print(f"shard {args.shard_index}/{args.num_shards}: {len(todo)} rows to do "
          f"({len(done_indices)} resumed)", flush=True)

    complete = _make_http_complete(args)
    semaphore = asyncio.Semaphore(args.concurrency)
    stats = {"rows": len(todo), "compressed": 0, "passthrough": 0, "fallback": 0}
    lock = asyncio.Lock()
    done = {"count": 0}

    with open(output, "a" if args.resume else "w", encoding="utf-8") as sink:

        async def one(index: int, row: dict) -> None:
            async with semaphore:
                result = await compress_row(row, args=args, complete=complete)
            result["row_index"] = index
            key = {True: "compressed", "passthrough": "passthrough", False: "fallback"}[
                result["compressed"]
            ]
            async with lock:
                stats[key] += 1
                sink.write(json.dumps(result) + "\n")
                sink.flush()
            done["count"] += 1
            if done["count"] % 100 == 0:
                print(f"{done['count']}/{len(todo)} rows", flush=True)

        await asyncio.gather(*(one(i, r) for i, r in todo))
    with open(output.with_suffix(".stats.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    print(json.dumps(stats, indent=2))


def main() -> None:
    asyncio.run(run(_parse_args()))


if __name__ == "__main__":
    main()
