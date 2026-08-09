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
"""Grid serialization, answer extraction, and reward scoring for ARC-AGI tasks.

Deliberately stdlib-only so the parser and scorer can be exercised without Ray,
torch, or a GPU -- they are the two pieces most likely to silently starve or
inflate a training run, so they need to be cheap to test.
"""

import re
from dataclasses import dataclass

# An ARC grid: rectangular, symbols 0-9, at most 30x30.
Grid = list[list[int]]

ANSWER_OPEN = "<answer>"
ANSWER_CLOSE = "</answer>"
MAX_GRID_DIM = 30
NUM_COLORS = 10

_ANSWER_BLOCK_RE = re.compile(
    re.escape(ANSWER_OPEN) + r"(.*?)" + re.escape(ANSWER_CLOSE), re.DOTALL
)


@dataclass(frozen=True)
class RewardWeights:
    """Per-term reward weights.

    Internal container, not loaded from YAML: the defaults live on
    ``ArcAgiEnvConfig`` in ``arc_agi_environment.py``, which builds this. No
    field defaults here on purpose -- a second set of defaults is a second
    source of truth.
    """

    exact: float
    cell: float
    color: float
    extraneous: float
    shape: float
    format: float


def serialize_grid(grid: Grid) -> str:
    """Render a grid as one line of digits per row."""
    return "\n".join("".join(str(cell) for cell in row) for row in grid)


def parse_grid(text: str) -> Grid | None:
    """Parse digit-rows into a grid, or return None if it is not well formed.

    Rejects empty grids, ragged rows, non-digit characters, and anything larger
    than ``MAX_GRID_DIM`` in either dimension. Strict on purpose: an
    over-permissive parser inflates reward, which is indistinguishable from
    learning until the run is long over.
    """
    rows = [line.strip() for line in text.strip().splitlines()]
    rows = [row for row in rows if row]
    if not rows:
        return None
    width = len(rows[0])
    if width == 0 or width > MAX_GRID_DIM or len(rows) > MAX_GRID_DIM:
        return None
    grid: Grid = []
    for row in rows:
        if len(row) != width or not row.isdigit():
            return None
        grid.append([int(char) for char in row])
    return grid


def extract_answer_grid(response: str) -> Grid | None:
    """Extract the final answer grid from a model response.

    Scans every ``<answer>...</answer>`` block and returns the last one that
    parses, so free-form reasoning that mentions grid-like text earlier cannot
    displace the real answer.
    """
    for block in reversed(_ANSWER_BLOCK_RE.findall(response)):
        grid = parse_grid(block)
        if grid is not None:
            return grid
    return None


def grid_shape(grid: Grid) -> tuple[int, int]:
    return len(grid), len(grid[0])


def overlay_cell_accuracy(pred: Grid, target: Grid) -> float:
    """Fraction of cells that agree when ``pred`` is centered over ``target``.

    Works at any pair of shapes, which is the point: a prediction that is one
    row short still earns most of its cells, and that partial credit is what
    keeps GRPO groups from collapsing to a single reward value.

    The denominator is the larger of the two areas, so padding the prediction
    out to a huge grid that happens to cover the target cannot inflate the
    score.
    """
    pred_h, pred_w = grid_shape(pred)
    target_h, target_w = grid_shape(target)
    row_offset = (target_h - pred_h) // 2
    col_offset = (target_w - pred_w) // 2

    matches = 0
    for row in range(target_h):
        pred_row = row - row_offset
        if not 0 <= pred_row < pred_h:
            continue
        for col in range(target_w):
            pred_col = col - col_offset
            if 0 <= pred_col < pred_w and pred[pred_row][pred_col] == target[row][col]:
                matches += 1

    return matches / max(pred_h * pred_w, target_h * target_w)


def color_recall(pred: Grid, target: Grid) -> float:
    """Fraction of the target's colors that appear anywhere in the prediction."""
    target_colors = {cell for row in target for cell in row}
    pred_colors = {cell for row in pred for cell in row}
    return len(target_colors & pred_colors) / len(target_colors)


def extraneous_color_fraction(pred: Grid, target: Grid) -> float:
    """Fraction of the prediction's colors that the target does not use.

    Paired with ``color_recall``: without this penalty, emitting all ten colors
    would max out recall for free.
    """
    target_colors = {cell for row in target for cell in row}
    pred_colors = {cell for row in pred for cell in row}
    return len(pred_colors - target_colors) / len(pred_colors)


def shape_mismatch(pred: Grid, target: Grid) -> float:
    """Normalized magnitude of the shape error, in [0, 1]."""
    pred_h, pred_w = grid_shape(pred)
    target_h, target_w = grid_shape(target)
    error = abs(target_h - pred_h) + abs(target_w - pred_w)
    return min(1.0, error / (target_h + target_w))


def score_response(
    response: str, target: Grid, weights: RewardWeights
) -> dict[str, float]:
    """Score one model response against the target grid.

    Only the extracted grid is scored -- the reasoning that precedes it is never
    inspected, so it is free to be unreadable.

    Returns the total plus every term separately. The breakdown is the primary
    diagnostic for whether reward growth is real (exact match) or shaping.
    """
    pred = extract_answer_grid(response)
    if pred is None:
        # Charge the penalty terms in full and grant no format credit. This puts
        # the unparseable floor strictly below the worst parseable answer (whose
        # reward bottoms out at ``format - extraneous - shape``), so emitting
        # *something* well formed is always an improvement. Without that gap the
        # format term cannot bootstrap: at step 0, when nothing is solved, it is
        # the only reward difference the policy can act on.
        return {
            "reward": -(weights.extraneous + weights.shape),
            "exact_match": 0.0,
            "cell_match": 0.0,
            "color_recall": 0.0,
            "extraneous_colors": 1.0,
            "shape_mismatch": 1.0,
            "format_valid": 0.0,
        }

    terms = {
        "exact_match": float(pred == target),
        "cell_match": overlay_cell_accuracy(pred, target),
        "color_recall": color_recall(pred, target),
        "extraneous_colors": extraneous_color_fraction(pred, target),
        "shape_mismatch": shape_mismatch(pred, target),
        "format_valid": 1.0,
    }
    terms["reward"] = (
        weights.exact * terms["exact_match"]
        + weights.cell * terms["cell_match"]
        + weights.color * terms["color_recall"]
        - weights.extraneous * terms["extraneous_colors"]
        - weights.shape * terms["shape_mismatch"]
        + weights.format * terms["format_valid"]
    )
    return terms


def format_task_prompt(train_pairs: list[dict[str, Grid]], test_input: Grid) -> str:
    """Lay out the few-shot pairs and the test input as delimited text.

    Plain-text tags rather than added vocabulary: new tokens would need their
    embeddings trained from scratch, and GRPO's signal is far too sparse for
    that.
    """
    blocks = []
    for pair in train_pairs:
        blocks.append(
            "<example>\n"
            f"<input>\n{serialize_grid(pair['input'])}\n</input>\n"
            f"<output>\n{serialize_grid(pair['output'])}\n</output>\n"
            "</example>"
        )
    blocks.append(f"<test_input>\n{serialize_grid(test_input)}\n</test_input>")
    return "\n".join(blocks)
