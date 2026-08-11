import time

import pytest
import ray
import torch

from nemo_rl.environments.arc_agi_environment import ArcAgiEnvConfig
from nemo_rl.environments.utils import create_env

TARGET = [[1, 2], [3, 4]]
# Shares no cells with TARGET, so the copy-the-input baseline the similarity
# terms are measured against sits at its floor.
TEST_INPUT = [[9, 9], [9, 9]]


@pytest.fixture(scope="module")
def arc_env():
    env = create_env("arc_agi", {})
    yield env
    env.shutdown.remote()
    ray.kill(env)
    time.sleep(0.1)


def message_log(response: str):
    return [
        {"role": "user", "content": "prompt"},
        {"role": "assistant", "content": response},
    ]


def answer(text: str) -> str:
    return f"reasoning\n<answer>\n{text}\n</answer>"


def test_config_defaults_are_centralized():
    # The weights must come from the schema, not from call sites.
    cfg = ArcAgiEnvConfig()
    assert cfg.exact_weight == 1.0
    assert cfg.cell_weight == 0.20
    assert cfg.edit_weight == 0.10
    assert cfg.format_weight == 0.05


def test_config_override_is_honored():
    assert ArcAgiEnvConfig(cell_weight=0.5).cell_weight == 0.5


def test_step_scores_a_batch(arc_env):
    responses = [
        answer("12\n34"),  # exact
        answer("12\n33"),  # near miss
        "no answer at all",  # unparseable
    ]
    metadata = [
        {
            "target": TARGET,
            "test_input": TEST_INPUT,
            "task_id": "t0",
            "terms": None,
        }
        for _ in responses
    ]

    result = ray.get(
        arc_env.step.remote([message_log(r) for r in responses], metadata)
    )

    assert result.rewards.shape == (3,)
    assert torch.all(result.terminateds == 1)
    assert len(result.observations) == 3
    # Ordering is the contract the whole reward design rests on.
    assert result.rewards[0] > result.rewards[1] > result.rewards[2]
    assert result.metadata[0]["terms"]["grid_match"] == 1.0
    assert result.metadata[2]["terms"]["format_valid"] == 0.0
    # No stop strings: mcore's stop-word trimming desynchronizes generated
    # tokens from their logprobs, which corrupts the importance ratios. The
    # parser tolerates an unclosed answer instead.
    assert result.next_stop_strings[0] is None


def test_step_preserves_metadata_fields(arc_env):
    metadata = [
        {
            "target": TARGET,
            "test_input": TEST_INPUT,
            "task_id": "abc123",
            "terms": None,
        }
    ]
    result = ray.get(arc_env.step.remote([message_log(answer("12\n34"))], metadata))
    assert result.metadata[0]["task_id"] == "abc123"
    assert result.metadata[0]["target"] == TARGET
    assert result.metadata[0]["test_input"] == TEST_INPUT


def test_echoing_the_test_input_loses_to_a_real_answer(arc_env):
    # End to end through the actor: on a task whose target is one cell away
    # from its input, echoing the input must not outscore solving it.
    test_input = [[1, 1], [1, 1]]
    target = [[1, 1], [1, 0]]
    metadata = [
        {"target": target, "test_input": test_input, "task_id": "t", "terms": None}
        for _ in range(2)
    ]
    result = ray.get(
        arc_env.step.remote(
            [message_log(answer("11\n11")), message_log(answer("11\n10"))], metadata
        )
    )
    assert result.metadata[0]["terms"]["copied_input"] == 1.0
    assert result.metadata[0]["terms"]["cell_gain"] == 0.0
    assert result.rewards[1] > result.rewards[0]
