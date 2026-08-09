import time

import pytest
import ray
import torch

from nemo_rl.environments.arc_agi_environment import ArcAgiEnvConfig
from nemo_rl.environments.utils import create_env

TARGET = [[1, 2], [3, 4]]


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
        {"target": TARGET, "task_id": "t0", "terms": None} for _ in responses
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
    # The environment asks generation to stop at the closing delimiter.
    assert result.next_stop_strings[0] == ["</answer>"]


def test_step_preserves_metadata_fields(arc_env):
    metadata = [{"target": TARGET, "task_id": "abc123", "terms": None}]
    result = ray.get(arc_env.step.remote([message_log(answer("12\n34"))], metadata))
    assert result.metadata[0]["task_id"] == "abc123"
    assert result.metadata[0]["target"] == TARGET
