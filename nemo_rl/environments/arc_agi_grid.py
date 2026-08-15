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

Deliberately free of Ray, torch, and any GPU dependency so the parser and scorer
can be exercised on their own -- they are the two pieces most likely to silently
starve or inflate a training run, so they need to be cheap to test. numpy is the
one exception, for the sliding alignment search in
``best_alignment_cell_accuracy``.
"""

import re
from dataclasses import dataclass

import numpy as np

# An ARC grid: rectangular, symbols 0-9, at most 30x30.
Grid = list[list[int]]

ANSWER_OPEN = "<answer>"
ANSWER_CLOSE = "</answer>"
MAX_GRID_DIM = 30
NUM_COLORS = 10

# Difficulty level carried by every row, so per-level metrics can separate
# "solving level 0 and nothing else" from "uniformly mediocre" -- which is the
# whole question the synthetic curriculum exists to answer. Real ARC-AGI-2 rows
# are not on the ladder and get their own bucket.
REAL_ARC_LEVEL = -1


def level_metric_suffix(level: int) -> str:
    """Metric-key suffix for a difficulty level."""
    return "real" if level == REAL_ARC_LEVEL else f"level_{level}"


# Row boundary marker for edit distance. Outside 0-9 so it can never match a cell.
_ROW_SENTINEL = -1

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
    edit: float
    color: float
    extraneous: float
    shape: float
    format: float


def serialize_grid(grid: Grid) -> str:
    """Render a grid as one line per row, cells separated by single spaces.

    The separator is not cosmetic: without it a row is a single run of digits
    that the tokenizer merges into arbitrary multi-cell chunks, so cell
    boundaries are invisible to the model. One space per cell costs prompt
    length but makes every cell its own token.
    """
    return "\n".join(" ".join(str(cell) for cell in row) for row in grid)


def _parse_row(row: str) -> list[int] | None:
    """Parse one serialized row, accepting spaced or contiguous digits.

    We serialize with spaces, but a model that answers in the compact form has
    still produced a well-formed grid, and rejecting it would throw away reward
    signal over punctuation.
    """
    if any(char.isspace() for char in row):
        cells = row.split()
        if not all(len(cell) == 1 and cell.isdigit() for cell in cells):
            return None
        return [int(cell) for cell in cells]
    if not row.isdigit():
        return None
    return [int(char) for char in row]


def parse_grid(text: str) -> Grid | None:
    """Parse digit-rows into a grid, or return None if it is not well formed.

    Rejects empty grids, ragged rows, non-digit characters, and anything larger
    than ``MAX_GRID_DIM`` in either dimension. Strict on purpose: an
    over-permissive parser inflates reward, which is indistinguishable from
    learning until the run is long over.
    """
    lines = [line.strip() for line in text.strip().splitlines()]
    lines = [line for line in lines if line]
    if not lines or len(lines) > MAX_GRID_DIM:
        return None
    grid: Grid = []
    for line in lines:
        row = _parse_row(line)
        if row is None or not row or len(row) > MAX_GRID_DIM:
            return None
        if grid and len(row) != len(grid[0]):
            return None
        grid.append(row)
    return grid


def extract_answer_grid(response: str) -> Grid | None:
    """Extract the final answer grid from a model response.

    Scans every ``<answer>...</answer>`` block and returns the last one that
    parses, so free-form reasoning that mentions grid-like text earlier cannot
    displace the real answer.

    Falls back to the text after a final unclosed ``<answer>``. Generation stops
    on the closing delimiter and may not echo it back, and a response truncated
    at the token cap has no closing tag either -- in both cases the grid is
    right there, and refusing to read it would score a complete answer as
    garbage.
    """
    for block in reversed(_ANSWER_BLOCK_RE.findall(response)):
        grid = parse_grid(block)
        if grid is not None:
            return grid

    _, delimiter, tail = response.rpartition(ANSWER_OPEN)
    if delimiter and ANSWER_CLOSE not in tail:
        return parse_grid(tail)
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


def best_alignment_cell_accuracy(pred: Grid, target: Grid) -> float:
    """Cell agreement at the best valid-mode placement of one grid in the other.

    This is the valid-mode cross-correlation of the two grids: slide the smaller
    entirely inside the larger and count agreeing cells at each placement. Colors
    are labels, not magnitudes, so the per-cell operator is equality rather than
    a product -- one-hot the two grids and it is exactly a 3D correlation, since
    a product summed over one-hot channels *is* the equality indicator.

    Preferred over ``overlay_cell_accuracy`` because that function has to pick an
    alignment, and ``(h_t - h_p) // 2`` decides every odd-sized near-miss by a
    coin flip. Here the alignment is chosen rather than assumed. When the shapes
    are equal -- 123 of the 172 evaluation rows -- there is exactly one valid
    placement and the two agree by construction.

    The denominator stays ``max`` of the two areas, so a prediction padded out to
    cover the target still cannot buy a high score. Falls back to the centered
    overlay when neither grid fits inside the other (one taller, the other
    wider), where valid mode has no placement at all.
    """
    pred_h, pred_w = grid_shape(pred)
    target_h, target_w = grid_shape(target)

    if pred_h <= target_h and pred_w <= target_w:
        small, big = np.array(pred), np.array(target)
    elif target_h <= pred_h and target_w <= pred_w:
        small, big = np.array(target), np.array(pred)
    else:
        return overlay_cell_accuracy(pred, target)

    small_h, small_w = small.shape
    big_h, big_w = big.shape
    denominator = max(pred_h * pred_w, target_h * target_w)
    best = 0
    for row in range(big_h - small_h + 1):
        for col in range(big_w - small_w + 1):
            window = big[row : row + small_h, col : col + small_w]
            best = max(best, int(np.count_nonzero(small == window)))
    return best / denominator


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


def _flatten(grid: Grid) -> list[int]:
    """Flatten a grid to a cell sequence with a sentinel between rows.

    The row sentinel is what makes edit distance shape-aware: without it,
    dropping a row and shifting every subsequent cell one place left would look
    like a handful of substitutions instead of a deleted row.
    """
    sequence: list[int] = []
    for index, row in enumerate(grid):
        if index:
            sequence.append(_ROW_SENTINEL)
        sequence.extend(row)
    return sequence


def _levenshtein(left: list[int], right: list[int]) -> int:
    """Edit distance between two cell sequences, two-row DP."""
    if left == right:
        return 0
    if not left:
        return len(right)
    if not right:
        return len(left)

    previous = list(range(len(right) + 1))
    for i, left_cell in enumerate(left, start=1):
        current = [i]
        for j, right_cell in enumerate(right, start=1):
            current.append(
                min(
                    previous[j] + 1,
                    current[j - 1] + 1,
                    previous[j - 1] + (left_cell != right_cell),
                )
            )
        previous = current
    return previous[-1]


def edit_similarity(pred: Grid, target: Grid) -> float:
    """1 - normalized edit distance between the two grids, in [0, 1].

    Complements ``overlay_cell_accuracy``, which compares cells at fixed
    positions and so scores a prediction that is right but shifted by one row as
    almost entirely wrong. Edit distance charges that same prediction for one
    insertion. Nothing here has to be differentiable, so the two can simply be
    added: they disagree exactly on the near-misses that matter.

    Cost is O(area^2), which is 930^2 only for a 30x30 pair; real ARC test grids
    are far smaller and the typical case is a few thousand operations.
    """
    left, right = _flatten(pred), _flatten(target)
    return 1.0 - _levenshtein(left, right) / max(len(left), len(right))


def gain_over_baseline(score: float, baseline: float) -> float:
    """Rescale an absolute score to its improvement over a baseline, into [-1, 1].

    ``0`` means "no better than the baseline", ``1`` means perfect, negative
    means worse than the baseline. Both directions are normalized by the room
    available in that direction, so a task where the baseline is already 0.95
    is not quietly worth twenty times less than one where it is 0.05.

    This is what stops copying the input from paying. An echo scores ~0.61 cell
    accuracy averaged over the ARC-AGI-2 evaluation split -- more than any run
    so far has earned -- because ARC grids are mostly background and the
    background usually survives the transformation. Measured against a copy of
    that same task's input, an echo is worth exactly zero.
    """
    if score >= baseline:
        headroom = 1.0 - baseline
        return 1.0 if headroom <= 0.0 else (score - baseline) / headroom
    return (score - baseline) / baseline if baseline > 0.0 else -1.0


def shape_mismatch(pred: Grid, target: Grid) -> float:
    """Normalized magnitude of the shape error, in [0, 1]."""
    pred_h, pred_w = grid_shape(pred)
    target_h, target_w = grid_shape(target)
    error = abs(target_h - pred_h) + abs(target_w - pred_w)
    return min(1.0, error / (target_h + target_w))


def score_response(
    response: str, target: Grid, test_input: Grid, weights: RewardWeights
) -> dict[str, float]:
    """Score one model response against the target grid.

    Only the extracted grid is scored -- the reasoning that precedes it is never
    inspected, so it is free to be unreadable.

    The two similarity terms are paid on the *gain over echoing ``test_input``*,
    not on their absolute value, because absolute similarity to an ARC target is
    something a copy of the input already earns most of. Their raw values are
    still returned, both as the honest metric to report and so a run can be
    compared against the ones scored the old way.

    Returns the total plus every term separately. The breakdown is the primary
    diagnostic for whether reward growth is real (exact match) or shaping.
    """
    pred = extract_answer_grid(response)
    if pred is None:
        # Charge every penalty in full and grant no format credit. This puts the
        # unparseable floor strictly below the worst parseable answer, which
        # bottoms out one format bonus above it, so emitting *something* well
        # formed is always an improvement. Without that gap the format term
        # cannot bootstrap: at step 0, when nothing is solved, it is the only
        # reward difference the policy can act on. Note the gain terms reach -1,
        # so they belong in this floor -- otherwise a badly-wrong parseable
        # answer would score below garbage and the ordering would invert.
        return {
            "reward": -(
                weights.cell + weights.edit + weights.extraneous + weights.shape
            ),
            "grid_match": 0.0,
            "cell_match": 0.0,
            "cell_gain": -1.0,
            "edit_similarity": 0.0,
            "edit_gain": -1.0,
            "color_recall": 0.0,
            "extraneous_colors": 1.0,
            "shape_mismatch": 1.0,
            "format_valid": 0.0,
            "copied_input": 0.0,
        }

    copy_cell = best_alignment_cell_accuracy(test_input, target)
    copy_edit = edit_similarity(test_input, target)

    terms = {
        "grid_match": float(pred == target),
        "cell_match": best_alignment_cell_accuracy(pred, target),
        "edit_similarity": edit_similarity(pred, target),
        "color_recall": color_recall(pred, target),
        "extraneous_colors": extraneous_color_fraction(pred, target),
        "shape_mismatch": shape_mismatch(pred, target),
        "format_valid": 1.0,
        # Not scored. Logged because "is the policy converging on an echo" is
        # the question this whole reward change exists to answer.
        "copied_input": float(pred == test_input),
    }
    terms["cell_gain"] = gain_over_baseline(terms["cell_match"], copy_cell)
    terms["edit_gain"] = gain_over_baseline(terms["edit_similarity"], copy_edit)

    terms["reward"] = (
        weights.exact * terms["grid_match"]
        + weights.cell * terms["cell_gain"]
        + weights.edit * terms["edit_gain"]
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
