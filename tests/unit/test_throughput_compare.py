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

import pytest

from throughput_compare import DEFAULT_WARMUP_STEPS, summarize_rows


def _row(step: int, elapsed: float, consumed: int, gen_length: float) -> dict:
    return {
        "_step": step,
        "train/elapsed_time_s": elapsed,
        "train/consumed_samples": consumed,
        "train/mean_gen_tokens_per_sample": gen_length,
        "timing/train/policy_training": 4.0,
    }


def test_summarize_rows_uses_consumed_samples_after_warmup() -> None:
    rows = [
        _row(step, elapsed=step * 10.0, consumed=step * 20, gen_length=step * 10.0)
        for step in range(1, 7)
    ]

    result = summarize_rows("run", rows, batch=20, warmup_steps=2)

    assert result["n_steps"] == 4
    assert result["window_s"] == 40.0
    assert result["consumed_samples"] == 80
    assert result["samples_per_hour"] == pytest.approx(7200.0)
    assert result["mean_generation_length"] == pytest.approx(45.0)
    assert result["generated_tokens_per_second"] == pytest.approx(90.0)
    assert result["policy_median"] == pytest.approx(4.0)


def test_default_warmup_anchors_on_the_first_logged_step() -> None:
    # The training loop logs consumed_samples at step+1, so no step-0 anchor exists.
    rows = [
        _row(step, elapsed=step * 10.0, consumed=step * 20, gen_length=100.0)
        for step in range(1, 4)
    ]

    result = summarize_rows("run", rows, batch=20, warmup_steps=DEFAULT_WARMUP_STEPS)

    assert result["n_steps"] == 2
    assert result["consumed_samples"] == 40


def test_summarize_rows_requires_exact_warmup_anchor() -> None:
    rows = [_row(1, 10.0, 20, 10.0), _row(3, 30.0, 60, 30.0)]

    with pytest.raises(ValueError, match="end-of-warmup point at step 2"):
        summarize_rows("run", rows, batch=20, warmup_steps=2)


def test_summarize_rows_rejects_nonpositive_window() -> None:
    rows = [_row(1, 10.0, 20, 10.0), _row(2, 10.0, 40, 20.0)]

    with pytest.raises(ValueError, match="non-positive measured window"):
        summarize_rows("run", rows, batch=20, warmup_steps=1)
