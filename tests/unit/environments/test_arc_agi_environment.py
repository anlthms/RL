import time

import pytest
import ray
import torch

from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.environments.arc_agi_environment import ArcAgiEnvConfig
from nemo_rl.environments.arc_agi_grid import (
    REAL_ARC_LEVEL,
    RewardWeights,
    score_response,
)
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
            "level": REAL_ARC_LEVEL,
            "terms": None,
        }
        for _ in responses
    ]

    result = ray.get(arc_env.step.remote([message_log(r) for r in responses], metadata))

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
            "level": REAL_ARC_LEVEL,
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
        {
            "target": target,
            "test_input": test_input,
            "task_id": "t",
            "level": REAL_ARC_LEVEL,
            "terms": None,
        }
        for _ in range(2)
    ]
    result = ray.get(
        arc_env.step.remote(
            [message_log(answer("11\n11")), message_log(answer("11\n10"))], metadata
        )
    )
    assert result.metadata[0]["terms"]["copied_input"] == 1.0
    assert result.metadata[0]["terms"]["cell_gain"] == 0.0
    assert result.rewards[0] == pytest.approx(-0.4)
    assert result.rewards[1] > result.rewards[0]


def test_terms_are_tagged_with_the_difficulty_level(arc_env):
    # Validation averages each per-sample term over the samples that reported
    # it, so a level-tagged key *is* that level's mean with no extra plumbing.
    # Without the split, an aggregate grid match cannot tell "solving level 0
    # and nothing else" from "uniformly mediocre".
    metadata = [
        {
            "target": TARGET,
            "test_input": TEST_INPUT,
            "task_id": "t",
            "level": level,
            "terms": None,
        }
        for level in (0, 3, REAL_ARC_LEVEL)
    ]
    responses = [answer("12\n34"), answer("99\n99"), answer("12\n34")]
    result = ray.get(arc_env.step.remote([message_log(r) for r in responses], metadata))
    assert result.metadata[0]["terms"]["grid_match/level_0"] == 1.0
    assert result.metadata[1]["terms"]["grid_match/level_3"] == 0.0
    assert result.metadata[2]["terms"]["grid_match/real"] == 1.0
    # A sample only reports its own level.
    assert "grid_match/level_3" not in result.metadata[0]["terms"]


def test_global_metrics_are_reported_per_level(arc_env):
    # Terms come from the real scorer rather than a hand-written dict: the
    # metrics function reads every term by name, so a partial dict tests the
    # test rather than the code.
    weights = RewardWeights(
        exact=1.0,
        cell=0.20,
        edit=0.10,
        color=0.05,
        extraneous=0.05,
        shape=0.05,
        format=0.05,
    )
    scored = []
    for level, response in ((0, "12\n34"), (0, "99\n99"), (1, "12\n34")):
        scored.append(
            {
                "target": TARGET,
                "test_input": TEST_INPUT,
                "task_id": "t",
                "level": level,
                "terms": score_response(answer(response), TARGET, TEST_INPUT, weights),
            }
        )
    # Two of the three are exact, and the one at level 1 is one of them, so the
    # aggregate and the per-level split disagree -- which is the point.
    assert [terms["terms"]["grid_match"] for terms in scored] == [1.0, 0.0, 1.0]
    batch = BatchedDataDict(
        {
            "rewards": torch.tensor([1.0, 0.0, 1.0]),
            "is_end": torch.ones(3),
            # (b, s) token ids, not strings: calculate_pass_rate_per_prompt
            # groups rollouts by prompt with torch.unique(dim=0). The first two
            # share a prompt, so this is one solved prompt out of two.
            "text": torch.tensor([[1, 2], [1, 2], [3, 4]]),
            "generation_lengths": torch.tensor([10, 20, 30]),
            "prompt_lengths": torch.tensor([5, 5, 5]),
            "extra_env_info": scored,
        }
    )
    _, metrics = ray.get(arc_env.global_post_process_and_metrics.remote(batch))
    assert metrics["grid_match"] == pytest.approx(2 / 3)
    assert metrics["grid_match/level_0"] == pytest.approx(0.5)
    assert metrics["grid_match/level_1"] == 1.0
    assert metrics["num_problems_in_batch/level_0"] == 2
    # No real-ARC rows in this batch, so no real bucket is invented for them.
    assert "grid_match/real" not in metrics
    assert metrics["pass@samples_per_prompt"] == pytest.approx(1.0)
