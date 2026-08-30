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
"""Invertible ARC task views: D8 symmetries composed with color bijections.

A *view* applies one D8 element and one color permutation consistently to
every grid of a task (demo inputs, demo outputs, and test inputs). Because
both operations are invertible and ARC transformation rules commute with
them, a solved view is a solved task -- but only after the prediction is
mapped back into the original coordinate/color frame. Voting or exact
scoring over raw grids from different frames is meaningless, so every
consumer must call :func:`ArcView.invert_grid` on a prediction before
comparing it with anything canonical.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

Grid = list[list[int]]

NUM_COLORS = 10


def _rot90(grid: Grid) -> Grid:
    """Rotate 90 degrees clockwise."""
    return [list(row) for row in zip(*grid[::-1])]


def _rot180(grid: Grid) -> Grid:
    return [row[::-1] for row in grid[::-1]]


def _rot270(grid: Grid) -> Grid:
    """Rotate 90 degrees counterclockwise."""
    return [list(row) for row in zip(*grid)][::-1]


def _flip_h(grid: Grid) -> Grid:
    """Mirror left-right (reverse each row)."""
    return [row[::-1] for row in grid]


def _flip_v(grid: Grid) -> Grid:
    """Mirror top-bottom (reverse the row order)."""
    return [list(row) for row in grid[::-1]]


def _transpose(grid: Grid) -> Grid:
    return [list(row) for row in zip(*grid)]


def _anti_transpose(grid: Grid) -> Grid:
    """Reflect across the anti-diagonal."""
    return _rot180(_transpose(grid))


def _identity(grid: Grid) -> Grid:
    return [list(row) for row in grid]


# The eight elements of the dihedral group D8 with their inverses. Rotations
# invert to the opposite rotation; every reflection is an involution.
_D8 = {
    "identity": (_identity, "identity"),
    "rot90": (_rot90, "rot270"),
    "rot180": (_rot180, "rot180"),
    "rot270": (_rot270, "rot90"),
    "flip_h": (_flip_h, "flip_h"),
    "flip_v": (_flip_v, "flip_v"),
    "transpose": (_transpose, "transpose"),
    "anti_transpose": (_anti_transpose, "anti_transpose"),
}

D8_TRANSFORMS = tuple(_D8)


def apply_d8(grid: Grid, transform: str) -> Grid:
    if transform not in _D8:
        raise ValueError(f"unknown D8 transform {transform!r}")
    return _D8[transform][0](grid)


def invert_d8(grid: Grid, transform: str) -> Grid:
    if transform not in _D8:
        raise ValueError(f"unknown D8 transform {transform!r}")
    return _D8[_D8[transform][1]][0](grid)


@dataclass(frozen=True)
class ArcView:
    """One invertible task view: a D8 element plus a color bijection.

    ``color_map`` maps canonical color ``c`` to view color ``color_map[c]``.
    The identity view is ``ArcView()``.
    """

    transform: str = "identity"
    color_map: tuple[int, ...] = tuple(range(NUM_COLORS))

    def __post_init__(self) -> None:
        if self.transform not in _D8:
            raise ValueError(f"unknown D8 transform {self.transform!r}")
        if sorted(self.color_map) != list(range(NUM_COLORS)):
            raise ValueError("color_map must be a permutation of colors 0-9")

    @property
    def view_id(self) -> str:
        """Stable identifier recorded beside every candidate result."""
        colors = "".join(str(color) for color in self.color_map)
        return f"{self.transform}/c{colors}"

    def is_identity(self) -> bool:
        return self.transform == "identity" and self.color_map == tuple(
            range(NUM_COLORS)
        )

    def apply_grid(self, grid: Grid) -> Grid:
        """Map a canonical-frame grid into this view's frame."""
        return [
            [self.color_map[cell] for cell in row]
            for row in apply_d8(grid, self.transform)
        ]

    def invert_grid(self, grid: Grid | None) -> Grid | None:
        """Map a view-frame grid (e.g. a prediction) back to the canonical frame.

        ``None`` (a malformed or unparseable answer) passes through unchanged
        so parse failures stay parse failures in the canonical frame.
        """
        if grid is None:
            return None
        inverse_color = [0] * NUM_COLORS
        for canonical, view_color in enumerate(self.color_map):
            inverse_color[view_color] = canonical
        recolored = [[inverse_color[cell] for cell in row] for row in grid]
        return invert_d8(recolored, self.transform)

    def apply_pairs(self, pairs: list[dict[str, Grid]]) -> list[dict[str, Grid]]:
        """Map demo pairs into the view frame (both inputs and outputs)."""
        return [
            {
                "input": self.apply_grid(pair["input"]),
                "output": self.apply_grid(pair["output"]),
            }
            for pair in pairs
        ]


def sample_views(
    *,
    count: int,
    seed: int,
    include_identity: bool = True,
    color_permutations: bool = True,
    fix_background: bool = True,
) -> list[ArcView]:
    """Draw distinct views deterministically by seed.

    The identity view leads the list when requested so per-view metrics always
    have the untransformed baseline. ``fix_background`` keeps color 0 fixed,
    matching the common ARC convention that 0 is the background color.
    """
    if count <= 0:
        raise ValueError("count must be positive")
    rng = random.Random(seed)
    views: list[ArcView] = [ArcView()] if include_identity else []
    seen = {view.view_id for view in views}
    attempts = 0
    while len(views) < count:
        attempts += 1
        if attempts > 1000 * count:
            raise ValueError(
                "could not draw enough distinct views; reduce count or allow "
                "color permutations"
            )
        transform = rng.choice(D8_TRANSFORMS)
        if color_permutations:
            movable = list(range(1 if fix_background else 0, NUM_COLORS))
            rng.shuffle(movable)
            color_map = tuple(([0] if fix_background else []) + movable)
        else:
            color_map = tuple(range(NUM_COLORS))
        view = ArcView(transform=transform, color_map=color_map)
        if view.view_id in seen:
            continue
        seen.add(view.view_id)
        views.append(view)
    return views


def vote_canonical(
    predictions: list[Grid | None],
) -> tuple[Grid | None, dict[str, float]]:
    """Majority-vote canonical-frame predictions from different views.

    All inputs MUST already be in the canonical frame (invert_grid applied).
    Ties break toward the candidate seen earliest in the list, and the
    returned stats record the agreement level so a coin-flip win is visible.
    Parse failures (``None``) do not vote.
    """
    tally: dict[str, list] = {}
    order: list[str] = []
    for prediction in predictions:
        if prediction is None:
            continue
        key = repr(prediction)
        if key not in tally:
            tally[key] = [0, prediction]
            order.append(key)
        tally[key][0] += 1
    if not tally:
        return None, {"votes": 0.0, "agreement": 0.0, "tied": 0.0}
    best_key = max(order, key=lambda key: tally[key][0])
    best_votes = tally[best_key][0]
    tied = sum(1 for key in order if tally[key][0] == best_votes) > 1
    total_votes = sum(votes for votes, _ in tally.values())
    return tally[best_key][1], {
        "votes": float(best_votes),
        "agreement": best_votes / total_votes,
        "tied": float(tied),
    }
