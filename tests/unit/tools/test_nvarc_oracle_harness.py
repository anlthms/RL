import asyncio
import hashlib
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from tools.nvarc_oracle_harness import (
    ARC_PRIZE_PROMPT,
    HttpOracleClient,
    PilotCase,
    build_arc_prize_prompt,
    diagnose_executor_response,
    evaluate_direct,
    evaluate_split,
    materialize_manifest,
    parse_arc_prize_grid,
)
from resources_servers.arc_agi_2.logic import NVARC_EXECUTOR_PROMPT_TEMPLATE


RULE = (
    "<rules_summary>\nReverse each row.\n</rules_summary>\n\n"
    "<solution_steps>\nRead every row right to left.\n</solution_steps>\n\n"
    "<key_insight>\nHorizontal reflection.\n</key_insight>\n\n"
    "<puzzle_concepts>\nreflection\n</puzzle_concepts>"
)


class FakeClient:
    def __init__(self, respond: Callable[[str, list[dict[str, str]]], str]) -> None:
        self.respond = respond
        self.requests: list[tuple[str, list[dict[str, str]]]] = []

    async def complete(
        self, role: str, messages: list[dict[str, str]]
    ) -> dict[str, Any]:
        copied = json.loads(json.dumps(messages))
        self.requests.append((role, copied))
        content = self.respond(role, copied)
        return {
            "role": role,
            "request": {"model": "fake", "messages": copied},
            "response": {"choices": [{"message": {"content": content}}]},
            "content": content,
            "reasoning": f"reasoning for {role}",
            "usage": {"total_tokens": 10},
            "latency_seconds": 0.01,
            "retry_errors": [],
        }


def _case() -> PilotCase:
    return PilotCase(
        task_id="puzzle",
        difficulty=4,
        demo_pair_indices=(0, 1),
        test_pair_index=2,
        demos=[
            {"input": [[1, 2]], "output": [[2, 1]]},
            {"input": [[3, 4]], "output": [[4, 3]]},
        ],
        test_input=[[5, 6]],
        test_target=[[6, 5]],
    )


def test_arc_prize_prompt_is_byte_exact() -> None:
    assert hashlib.sha256(ARC_PRIZE_PROMPT.encode()).hexdigest() == (
        "065e42f724bcfe3a735ded38d92b34a6a942604e653a0584cc08b1de9bdf70cc"
    )
    prompt = build_arc_prize_prompt(
        training_pairs=[{"input": [[1]], "output": [[2]]}], test_input=[[3]]
    )
    expected_examples = "--Example 0-- \n\n INPUT: \n\n[[1]]\n\nOUTPUT: \n\n[[2]]\n\n"
    assert prompt == ARC_PRIZE_PROMPT.format(
        training_examples=expected_examples, test_input="[[3]]"
    )


def test_native_and_gym_executor_templates_are_byte_exact() -> None:
    native = Path("examples/prompts/nvarc_executor.txt").read_text(encoding="utf-8")
    assert native == NVARC_EXECUTOR_PROMPT_TEMPLATE


def test_manifest_sampling_is_frozen_and_target_free(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    pairs = [{"input": [[index]], "output": [[index + 1]]} for index in range(5)]
    rows = [
        {
            "puzzle_id": f"puzzle_{index}",
            "canonical_rule": RULE,
            "pairs_json": json.dumps(pairs),
            "difficulty": index,
        }
        for index in range(10)
    ]
    monkeypatch.setattr(
        "tools.nvarc_oracle_harness._load_nvarc_split",
        lambda _data_dir, _split: rows,
    )

    def build(output: Path) -> dict[str, Any]:
        return materialize_manifest(
            data_dir=str(tmp_path),
            output=output,
            count=4,
            seed=7,
            demo_pairs=3,
            endpoint="https://example.test/v1",
            model="fake",
            temperature=0.0,
            max_output_tokens=100,
            reasoning_effort="max",
            max_rounds=3,
        )

    first = build(tmp_path / "first.json")
    second = build(tmp_path / "second.json")
    assert first == second
    assert len({row["task_id"] for row in first["rows"]}) == 4
    assert all(
        row["test_pair_index"] not in row["demo_pair_indices"] for row in first["rows"]
    )
    assert all(
        not {"target", "test_output", "output"}.intersection(row)
        for row in first["rows"]
    )


@pytest.mark.parametrize(
    ("response", "expected"),
    [
        ("analysis [[1, 2], [3, 4]]", [[1, 2], [3, 4]]),
        ("old [[9]] final [[2]]", [[2]]),
        (r"answer \\boxed{[[7]]} trailing [[8]]", [[7]]),
        ("no grid", None),
        ("[]", None),
    ],
)
def test_arc_prize_parser_parity(response: str, expected: list | None) -> None:
    assert parse_arc_prize_grid(response) == expected


def test_stream_chunks_reconstruct_reasoning_content_and_usage() -> None:
    state: dict[str, Any] = {
        "content": "",
        "reasoning_content": "",
        "finish_reason": None,
        "usage": {},
    }
    HttpOracleClient._accumulate_stream_chunk(
        state,
        {
            "id": "request",
            "model": "kimi",
            "choices": [{"delta": {"reasoning_content": "think "}}],
        },
    )
    HttpOracleClient._accumulate_stream_chunk(
        state,
        {"choices": [{"delta": {"content": "[[1]]"}, "finish_reason": "stop"}]},
    )
    HttpOracleClient._accumulate_stream_chunk(
        state, {"choices": [], "usage": {"total_tokens": 7}}
    )

    assert state == {
        "content": "[[1]]",
        "reasoning_content": "think ",
        "finish_reason": "stop",
        "usage": {"total_tokens": 7},
        "id": "request",
        "model": "kimi",
    }


@pytest.mark.parametrize(
    ("response", "reason"),
    [
        ("nothing", "no_answer_block"),
        ("<answer></answer>", "empty_block"),
        ("<answer>1 x</answer>", "invalid_token_cell"),
        ("<answer>1 2\n3</answer>", "ragged_rows"),
        ("<answer>1 2\n3", "incomplete_truncated_grid"),
        ("<answer>1 2</answer>", "valid_wrong_shape"),
        ("<answer>1 0\n3 4</answer>", "valid_right_shape_wrong_cells"),
        ("<answer>1 2\n3 4</answer>", "exact"),
    ],
)
def test_executor_failure_taxonomy(response: str, reason: str) -> None:
    _, actual = diagnose_executor_response(response, [[1, 2], [3, 4]])
    assert actual == reason


def test_direct_uses_arc_prompt_and_exact_score() -> None:
    client = FakeClient(lambda _role, _messages: "analysis\n[[6, 5]]")
    result = asyncio.run(evaluate_direct(_case(), client))
    assert result["test_exact"] is True
    assert result["parse_outcome"] == "exact"
    assert len(client.requests) == 1
    assert "--Test Input--\n[[5, 6]]" in client.requests[0][1][0]["content"]


def test_split_refines_on_demo_without_leaking_test_target() -> None:
    proposer_calls = 0

    def respond(role: str, messages: list[dict[str, str]]) -> str:
        nonlocal proposer_calls
        if role == "proposer":
            proposer_calls += 1
            return RULE
        prompt = messages[0]["content"]
        input_text = prompt.split("<input>\n", 1)[1].split("\n</input>", 1)[0]
        cells = input_text.split()
        return "<answer>\n" + " ".join(reversed(cells)) + "\n</answer>"

    client = FakeClient(respond)
    result = asyncio.run(evaluate_split(_case(), client, max_rounds=3))
    assert proposer_calls == 1
    assert result["train_gate_pass"] is True
    assert result["test_exact"] is True
    assert result["proposer_sft_row"] is not None
    assert len(result["executor_sft_rows"]) == 3
    # The held-out target is never sent to either role. Demo outputs are public
    # and may appear in a revision prompt, but this episode needs no revision.
    serialized_requests = json.dumps(client.requests)
    assert "[[6, 5]]" not in serialized_requests


def test_split_revision_feedback_changes_rule_and_final_sweep() -> None:
    proposer_calls = 0
    first_executor = True

    def respond(role: str, messages: list[dict[str, str]]) -> str:
        nonlocal proposer_calls, first_executor
        if role == "proposer":
            proposer_calls += 1
            return RULE
        if first_executor:
            first_executor = False
            return "<answer>\n9 9\n</answer>"
        prompt = messages[0]["content"]
        input_text = prompt.split("<input>\n", 1)[1].split("\n</input>", 1)[0]
        return "<answer>\n" + " ".join(reversed(input_text.split())) + "\n</answer>"

    client = FakeClient(respond)
    result = asyncio.run(evaluate_split(_case(), client, max_rounds=3))
    assert proposer_calls == 2
    assert result["train_gate_pass"] is True
    assert result["test_exact"] is True
    proposer_requests = [
        messages for role, messages in client.requests if role == "proposer"
    ]
    assert "Expected output:\n2 1" in proposer_requests[1][-1]["content"]
    # The final rule is swept over demo_0 after the revision before test use.
    executor_grid_ids = [
        attempt["calls"][0]["grid_id"] for attempt in result["executor_attempts"]
    ]
    assert executor_grid_ids == ["train_0", "train_0", "train_1", "test_0"]
