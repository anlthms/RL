import asyncio
import json
import random
from collections import defaultdict

from tools.nvarc_sft_materialize import PROPOSER_INSTRUCTIONS
from tools.nvarc_teacher_collect import (
    collect_episode,
    collect_executor_trace,
    split_reasoning,
)

RULE = (
    "<rules_summary>\nSwap the two cells.\n</rules_summary>\n\n"
    "<solution_steps>\nReverse each row.\n</solution_steps>\n\n"
    "<key_insight>\nOrder flips.\n</key_insight>\n\n"
    "<puzzle_concepts>\nreversal\n</puzzle_concepts>"
)


def test_split_reasoning_prefers_channel_then_think_then_marker() -> None:
    assert split_reasoning("thought", "<answer>\n1\n</answer>") == (
        "thought",
        "<answer>\n1\n</answer>",
    )
    cot, payload = split_reasoning("", "<think>inline</think>\n<answer>\n1\n</answer>")
    assert cot == "inline" and payload == "<answer>\n1\n</answer>"
    cot, payload = split_reasoning(
        "", "so I reason...\n<rules_summary>r</rules_summary>"
    )
    assert cot == "so I reason..." and payload.startswith("<rules_summary>")


def test_executor_trace_keeps_only_exact_answers_with_bounded_cot() -> None:
    puzzle = {"puzzle_id": "p", "canonical_rule": RULE, "pairs_json": "[]"}
    pair = {"input": [[1, 0]], "output": [[0, 1]]}
    responses = iter(
        [
            ("I guess", "<answer>\n1 1\n</answer>"),  # wrong grid
            ("x" * 50, "<answer>\n0 1\n</answer>"),  # exact, kept
        ]
    )

    async def complete(prompt: str) -> tuple[str, str]:
        return next(responses)

    trace = asyncio.run(
        collect_executor_trace(
            puzzle,
            pair,
            template="task:\n{}\n",
            complete=complete,
            samples=3,
            max_cot_chars=100,
        )
    )
    assert trace is not None
    target = trace["messages"][-1]["content"]
    assert target == "<think>\n" + "x" * 50 + "\n</think>\n\n<answer>\n0 1\n</answer>"
    assert trace["sft_role"] == "executor"


def test_executor_trace_rejects_overlong_cot() -> None:
    puzzle = {"puzzle_id": "p", "canonical_rule": RULE, "pairs_json": "[]"}
    pair = {"input": [[1, 0]], "output": [[0, 1]]}

    async def complete(prompt: str) -> tuple[str, str]:
        return "y" * 500, "<answer>\n0 1\n</answer>"

    trace = asyncio.run(
        collect_executor_trace(
            puzzle,
            pair,
            template="task:\n{}\n",
            complete=complete,
            samples=2,
            max_cot_chars=100,
        )
    )
    assert trace is None


def test_episode_verifies_rules_behaviorally_without_leakage() -> None:
    pairs = [{"input": [[index, 0]], "output": [[0, index]]} for index in range(6)]
    puzzle = {"puzzle_id": "p", "canonical_rule": RULE, "pairs_json": json.dumps(pairs)}
    prompts_seen: list[str] = []

    async def complete(messages: list[dict]) -> tuple[str, str]:
        prompt = messages[-1]["content"]
        prompts_seen.append(prompt)
        if "<transformation>" in prompt:
            # Executor verification call: reverse the presented input row.
            row = prompt.split("<input>\n")[1].split("\n</input>")[0]
            cells = row.split()
            return "apply", "<answer>\n" + " ".join(reversed(cells)) + "\n</answer>"
        return "induce", RULE

    result = asyncio.run(
        collect_episode(
            puzzle,
            demo_pairs=3,
            eval_pairs=2,
            byte_budget=8000,
            rng=random.Random(0),
            complete=complete,
            samples=1,
            max_cot_chars=100,
            max_rounds=3,
            max_transcript_chars=10_000,
            min_verified=None,
            rejects=defaultdict(int),
        )
    )
    assert result is not None
    trace, harvested = result
    assert trace is not None
    assert trace["verified_pairs"] == 2
    assert len(harvested) == 2
    assert trace["messages"][0]["content"] == PROPOSER_INSTRUCTIONS
    assert trace["messages"][-1]["content"].startswith("<think>\ninduce\n</think>\n\n")
    # The held-out OUTPUTS never appear in any prompt sent to the teacher.
    eval_outputs = {"0 " + str(i) for i in range(6)}
    for prompt in prompts_seen:
        if "<transformation>" in prompt:
            body = prompt.split("<input>")[0]
            assert not any(
                out in body.split("<transformation>")[0] for out in eval_outputs
            )


def test_episode_rejects_rules_the_teacher_cannot_execute() -> None:
    pairs = [{"input": [[index, 0]], "output": [[0, index]]} for index in range(6)]
    puzzle = {"puzzle_id": "p", "canonical_rule": RULE, "pairs_json": json.dumps(pairs)}

    async def complete(messages: list[dict]) -> tuple[str, str]:
        prompt = messages[-1]["content"]
        if "<transformation>" in prompt:
            return "apply", "<answer>\n9 9\n</answer>"  # never matches
        return "induce", RULE

    rejects = defaultdict(int)
    trace = asyncio.run(
        collect_episode(
            puzzle,
            demo_pairs=3,
            eval_pairs=2,
            byte_budget=8000,
            rng=random.Random(0),
            complete=complete,
            samples=2,
            max_cot_chars=100,
            max_rounds=3,
            max_transcript_chars=10_000,
            min_verified=None,
            rejects=rejects,
        )
    )
    assert trace is None
    assert rejects["rounds_exhausted"] == 1
