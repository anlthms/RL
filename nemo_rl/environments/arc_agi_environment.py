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
"""GRPO environment for ARC-AGI tasks.

Single-turn: the model sees the few-shot pairs and the test input, and answers
with free-form reasoning followed by one delimited grid. Only the grid is
scored -- see ``arc_agi_grid.py`` for the parser and the reward terms.
"""

from typing import Any, TypedDict

import ray
import torch
from pydantic import BaseModel

from nemo_rl.data.interfaces import LLMMessageLogType
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.environments.arc_agi_grid import (
    Grid,
    RewardWeights,
    score_response,
)
from nemo_rl.environments.interfaces import EnvironmentInterface, EnvironmentReturn
from nemo_rl.environments.metrics import calculate_pass_rate_per_prompt


class ArcAgiEnvConfig(BaseModel, extra="allow"):
    """Reward weights for the ARC-AGI environment.

    The dense terms exist to break degenerate GRPO groups: on ARC-AGI-2 a small
    policy solves approximately nothing, so a binary exact-match reward gives
    every rollout in a group the same value, the advantage is zero, and the run
    produces no gradient at all. These terms make two wrong answers
    distinguishable.

    Both similarity terms are paid on their gain over echoing the test input
    rather than on their absolute value. Scored absolutely, a copy of the input
    earns ~0.61 cell accuracy on the ARC-AGI-2 evaluation split -- more than any
    run has yet earned by reasoning -- and training duly converged on the echo
    (58% of validation answers by step 60 of job 6041201). Measured against a
    copy of the same task's input, an echo is worth exactly zero.

    Attributes:
        exact_weight: Weight on exact grid match. Kept dominant so that shaping
            can never be a cheaper path to reward than solving the task.
        cell_weight: Weight on centered-overlay cell accuracy, scored as gain
            over the copy-the-input baseline.
        edit_weight: Weight on 1 - normalized edit distance, likewise scored as
            gain over the copy baseline. Complements the overlay term, which
            compares fixed positions and so writes off a prediction that is
            correct but shifted by a row; edit distance charges that same
            prediction for a single insertion.
        color_weight: Weight on the fraction of target colors present.
        extraneous_color_weight: Penalty weight on predicted colors absent from
            the target. Required alongside ``color_weight``, which is otherwise
            maxed out for free by emitting all ten colors.
        shape_weight: Penalty weight on normalized shape error.
        format_weight: Weight on emitting exactly one parseable grid.
    """

    exact_weight: float = 1.0
    cell_weight: float = 0.20
    edit_weight: float = 0.10
    color_weight: float = 0.05
    extraneous_color_weight: float = 0.05
    shape_weight: float = 0.05
    format_weight: float = 0.05


class ArcAgiEnvironmentMetadata(TypedDict):
    """Per-sample state carried through ``extra_env_info``.

    ``target`` and ``test_input`` are set by the data processor; the scoring
    terms are written back by ``step`` so ``global_post_process_and_metrics``
    can report the reward breakdown -- whether reward growth is real (exact
    match) or merely shaping is the diagnostic this whole environment is built
    around. ``test_input`` is carried because the similarity terms are scored
    as gain over echoing it, which needs the input at scoring time.
    """

    target: Grid
    test_input: Grid
    task_id: str
    terms: dict[str, float] | None


@ray.remote(
    max_restarts=-1, max_task_retries=-1, max_concurrency=1000
)  # pragma: no cover
class ArcAgiEnvironment(EnvironmentInterface[ArcAgiEnvironmentMetadata]):
    def __init__(self, cfg: dict[str, Any] | ArcAgiEnvConfig):
        # create_env() hands every environment the raw YAML block, so validate
        # it here -- that is what makes ArcAgiEnvConfig's field defaults the one
        # source of truth for the weights instead of scattering them.
        self.cfg = (
            cfg if isinstance(cfg, ArcAgiEnvConfig) else ArcAgiEnvConfig(**dict(cfg))
        )
        cfg = self.cfg
        self.weights = RewardWeights(
            exact=cfg.exact_weight,
            cell=cfg.cell_weight,
            edit=cfg.edit_weight,
            color=cfg.color_weight,
            extraneous=cfg.extraneous_color_weight,
            shape=cfg.shape_weight,
            format=cfg.format_weight,
        )

    def shutdown(self) -> None:
        pass

    def step(
        self,
        message_log_batch: list[LLMMessageLogType],
        metadata: list[ArcAgiEnvironmentMetadata],
    ) -> EnvironmentReturn[ArcAgiEnvironmentMetadata]:
        """Scores one batch of responses against their target grids.

        Args:
            message_log_batch: Batch of OpenAI-API-like message logs.
            metadata: Per-sample metadata carrying the target grid.

        Returns:
            EnvironmentReturn with the scalar reward per sample and the
            per-term breakdown written back into metadata.
        """
        responses = [
            "".join(
                str(message["content"])
                for message in conversation
                if message["role"] == "assistant"
            )
            for conversation in message_log_batch
        ]

        # Scoring is pure Python over grids of at most 900 cells, so it runs
        # inline rather than fanning out to verifier actors the way the math
        # environment must for math-verify.
        all_terms = [
            score_response(
                response, meta["target"], meta["test_input"], self.weights
            )
            for response, meta in zip(responses, metadata)
        ]

        updated_metadata: list[ArcAgiEnvironmentMetadata] = [
            {**meta, "terms": terms} for meta, terms in zip(metadata, all_terms)
        ]
        observations = [
            {
                "role": "environment",
                "content": "Environment: correct"
                if terms["grid_match"]
                else "Environment: incorrect",
            }
            for terms in all_terms
        ]
        rewards = torch.tensor([terms["reward"] for terms in all_terms]).cpu()

        return EnvironmentReturn(
            observations=observations,
            metadata=updated_metadata,
            next_stop_strings=[None] * len(message_log_batch),
            rewards=rewards,
            terminateds=torch.ones_like(rewards).cpu(),
            answers=None,
        )

    def global_post_process_and_metrics(
        self, batch: BatchedDataDict[Any]
    ) -> tuple[BatchedDataDict[Any], dict[str, float | int]]:
        """Reports the reward breakdown alongside the two headline ARC metrics.

        ``grid_match`` is the honest ARC score (exact match); ``cell_match`` is
        the centered-overlay accuracy, which moves long before grid match does
        and is the metric to watch early.
        """
        rewards = batch["rewards"] * batch["is_end"]
        all_terms = [meta["terms"] for meta in batch["extra_env_info"]]
        # A sample that never reached the environment has no terms.
        scored = [terms for terms in all_terms if terms is not None]

        def mean(name: str) -> float:
            if not scored:
                return 0.0
            return sum(terms[name] for terms in scored) / len(scored)

        metrics: dict[str, float | int] = {
            "accuracy": mean("grid_match"),
            "grid_match": mean("grid_match"),
            "cell_match": mean("cell_match"),
            "cell_gain": mean("cell_gain"),
            "edit_similarity": mean("edit_similarity"),
            "edit_gain": mean("edit_gain"),
            # The hack detector: the fraction of answers that are the test input
            # echoed back. Rising here means the run is gaming the shaped terms.
            "copied_input": mean("copied_input"),
            "color_recall": mean("color_recall"),
            "extraneous_colors": mean("extraneous_colors"),
            "shape_mismatch": mean("shape_mismatch"),
            "format_valid": mean("format_valid"),
            "pass@samples_per_prompt": calculate_pass_rate_per_prompt(
                batch["text"], (rewards >= self.cfg.exact_weight).float()
            ),
            "fraction_of_samples_properly_ended": batch["is_end"].float().mean().item(),
            "num_problems_in_batch": batch["is_end"].shape[0],
            "generation_lengths": batch["generation_lengths"].float().mean().item(),
            "prompt_lengths": batch["prompt_lengths"].float().mean().item(),
        }
        return batch, metrics
