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
"""Deterministic synthetic ARC transforms for the executor curriculum.

Tasks are regenerated from ``(seed, index, level)`` and require no stored
dataset. The module is stdlib-only; grid parsing and scoring live in
``arc_agi_grid.py``.
"""

import random
import re
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from typing import TypeVar

from nemo_rl.environments.arc_agi_grid import MAX_GRID_DIM, Grid

BACKGROUND = 0
COLORS = tuple(range(1, 10))

# 3x3 is the smallest grid that can hold a shape with an interior; 20 is the
# plan's upper bound on an *input* -- geometric rules multiply it, and every
# grid in a task must still fit MAX_GRID_DIM.
MIN_GRID_DIM = 3
# Upper bound on an input grid's side, overridable per call.
DEFAULT_MAX_INPUT_DIM = 20
# Object placement needs room. Below this, "put three shapes down with a
# background margin between them" usually fails and we would spend the whole
# attempt budget on rejections.
MIN_OBJECT_GRID_DIM = 8

LEVELS = (1, 2, 3, 4, 5)

# How many (rule, pairs) draws to make before giving up on a level. Generous:
# rejection is the normal path (a rot180 that happens to land on a symmetric
# grid is *supposed* to be thrown away), and exhausting this budget means a
# generator bug rather than bad luck, which is why it raises.
_MAX_ATTEMPTS = 200
_MAX_PLACEMENT_TRIES = 40

Pair = tuple[Grid, Grid]
AxisT = TypeVar("AxisT")
ChoiceT = TypeVar("ChoiceT")

# How many non-background colors each level's rules want, low to high. Levels 2
# and 5 need at least two so that a color op is identifiable at all -- "drop the
# 3s" cannot be read off examples that contain only 3s. Level 4's structural
# rules stay near-monochrome so the structure is what varies.
_PALETTE_RANGE: dict[int, tuple[int, int]] = {
    1: (1, 3),
    2: (2, 4),
    3: (1, 3),
    4: (1, 2),
    5: (2, 4),
}

# Levels whose rules draw a color from *outside* the palette: recolor needs a
# destination, add_border a frame color, fill_enclosed a fill. Every range above
# leaves at least five spare colors, so `_spare_color` cannot come up empty --
# but it checks, because a future range that forgets this would otherwise fail
# as a bare `IndexError` from `rng.choice` deep inside a sampler.
_SPARE_COLOR_LEVELS = frozenset({2, 3, 4, 5})


def _spare_color(rng: random.Random, palette: list[int]) -> int:
    """A color the palette does not use, for rules that must introduce one."""
    spare = [color for color in COLORS if color not in palette]
    if not spare:
        raise ValueError(
            f"palette {sorted(palette)} uses every one of the {len(COLORS)} "
            "non-background colors, leaving none for a rule that must introduce one"
        )
    return rng.choice(spare)


# ---------------------------------------------------------------------------
# Grid transformations
# ---------------------------------------------------------------------------


def rot90(grid: Grid) -> Grid:
    """Rotate a quarter turn clockwise."""
    return [list(row) for row in zip(*grid[::-1])]


def rot180(grid: Grid) -> Grid:
    return [row[::-1] for row in grid[::-1]]


def rot270(grid: Grid) -> Grid:
    return [list(row) for row in zip(*grid)][::-1]


def flip_h(grid: Grid) -> Grid:
    """Mirror left-right."""
    return [row[::-1] for row in grid]


def flip_v(grid: Grid) -> Grid:
    """Mirror top-bottom."""
    return [list(row) for row in grid[::-1]]


def transpose(grid: Grid) -> Grid:
    return [list(row) for row in zip(*grid)]


def anti_transpose(grid: Grid) -> Grid:
    """Reflect across the anti-diagonal."""
    return rot180(transpose(grid))


def identity(grid: Grid) -> Grid:
    return [list(row) for row in grid]


# The dihedral group of the square. Keyed by name so a task can say which
# element it used, and so the group property is testable by composition.
DIHEDRAL: dict[str, Callable[[Grid], Grid]] = {
    "identity": identity,
    "rot90": rot90,
    "rot180": rot180,
    "rot270": rot270,
    "flip_h": flip_h,
    "flip_v": flip_v,
    "transpose": transpose,
    "anti_transpose": anti_transpose,
}


def drop_color(color: int) -> Callable[[Grid], Grid]:
    """Send every cell of ``color`` to the background."""
    return lambda grid: [
        [BACKGROUND if cell == color else cell for cell in row] for row in grid
    ]


def recolor(source: int, destination: int) -> Callable[[Grid], Grid]:
    return lambda grid: [
        [destination if cell == source else cell for cell in row] for row in grid
    ]


def keep_only(color: int) -> Callable[[Grid], Grid]:
    """Erase everything except ``color``."""
    return lambda grid: [
        [cell if cell == color else BACKGROUND for cell in row] for row in grid
    ]


def tile(vertical: int, horizontal: int) -> Callable[[Grid], Grid]:
    """Repeat the whole grid ``vertical`` x ``horizontal`` times."""

    def apply(grid: Grid) -> Grid:
        wide = [row * horizontal for row in grid]
        return [list(row) for _ in range(vertical) for row in wide]

    return apply


def scale(factor: int) -> Callable[[Grid], Grid]:
    """Blow each cell up into a ``factor`` x ``factor`` block."""

    def apply(grid: Grid) -> Grid:
        out: Grid = []
        for row in grid:
            stretched = [cell for cell in row for _ in range(factor)]
            out.extend([list(stretched) for _ in range(factor)])
        return out

    return apply


def crop_to_bbox(grid: Grid) -> Grid:
    """Crop to the bounding box of the non-background cells.

    Returns the grid unchanged when it is entirely background, which the
    degeneracy guard then rejects -- there is nothing to crop to.
    """
    cells = [
        (r, c)
        for r, row in enumerate(grid)
        for c, cell in enumerate(row)
        if cell != BACKGROUND
    ]
    if not cells:
        return identity(grid)
    top = min(r for r, _ in cells)
    bottom = max(r for r, _ in cells)
    left = min(c for _, c in cells)
    right = max(c for _, c in cells)
    return [row[left : right + 1] for row in grid[top : bottom + 1]]


def add_border(color: int, width: int = 1) -> Callable[[Grid], Grid]:
    """Wrap the grid in a frame of ``color``."""

    def apply(grid: Grid) -> Grid:
        inner_width = len(grid[0]) + 2 * width
        frame = [[color] * inner_width for _ in range(width)]
        body = [[color] * width + list(row) + [color] * width for row in grid]
        return frame + body + [list(row) for row in frame]

    return apply


def denoise(grid: Grid) -> Grid:
    """Erase non-background cells that have no non-background 4-neighbor.

    "Keep the shapes, drop the specks" -- the isolated-cell definition is the
    one that makes a shape's own cells safe, since every cell of a 2x2-or-larger
    blob touches another.
    """
    height, width = len(grid), len(grid[0])
    out = identity(grid)
    for r in range(height):
        for c in range(width):
            if grid[r][c] == BACKGROUND:
                continue
            neighbors = [
                grid[r + dr][c + dc]
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1))
                if 0 <= r + dr < height and 0 <= c + dc < width
            ]
            if all(neighbor == BACKGROUND for neighbor in neighbors):
                out[r][c] = BACKGROUND
    return out


def complete_symmetry(axis: str) -> Callable[[Grid], Grid]:
    """Fill background cells from the grid's mirror image across ``axis``.

    The input is a symmetric picture with part of one side erased; the output
    puts it back. A cell that is already painted is never overwritten, so the
    rule is a pure completion.
    """
    mirror = flip_h if axis == "horizontal" else flip_v

    def apply(grid: Grid) -> Grid:
        reflected = mirror(grid)
        return [
            [
                cell if cell != BACKGROUND else reflected[r][c]
                for c, cell in enumerate(row)
            ]
            for r, row in enumerate(grid)
        ]

    return apply


def fill_enclosed(color: int) -> Callable[[Grid], Grid]:
    """Paint every background cell that cannot reach the border with ``color``.

    Flood from the border rather than searching for closed curves: reachability
    is the definition of "enclosed" and needs no shape analysis.
    """

    def apply(grid: Grid) -> Grid:
        height, width = len(grid), len(grid[0])
        outside = [[False] * width for _ in range(height)]
        stack = [
            (r, c)
            for r in range(height)
            for c in range(width)
            if (r in (0, height - 1) or c in (0, width - 1))
            and grid[r][c] == BACKGROUND
        ]
        for r, c in stack:
            outside[r][c] = True
        while stack:
            r, c = stack.pop()
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nr, nc = r + dr, c + dc
                if (
                    0 <= nr < height
                    and 0 <= nc < width
                    and not outside[nr][nc]
                    and grid[nr][nc] == BACKGROUND
                ):
                    outside[nr][nc] = True
                    stack.append((nr, nc))
        return [
            [
                color if cell == BACKGROUND and not outside[r][c] else cell
                for c, cell in enumerate(row)
            ]
            for r, row in enumerate(grid)
        ]

    return apply


# ---------------------------------------------------------------------------
# Input patterns
# ---------------------------------------------------------------------------
#
# Uniform noise makes several levels degenerate: crop-to-bounding-box needs a
# bounding box, denoising needs signal to distinguish from noise, and symmetry
# completion needs a symmetry. So inputs come from a small library of pattern
# samplers, and each rule names the one that gives it something to act on.
#
# Every sampler paints all of `palette`, because the identifiability guard
# requires a color rule's parameter to appear in the examples that are supposed
# to teach it. Samplers return None when placement fails, and the caller
# redraws.


def _size(
    rng: random.Random, max_dim: int, min_dim: int = MIN_GRID_DIM
) -> tuple[int, int]:
    high = min(max_dim, MAX_GRID_DIM)
    low = min(min_dim, high)
    return rng.randint(low, high), rng.randint(low, high)


def _blank(height: int, width: int) -> Grid:
    return [[BACKGROUND] * width for _ in range(height)]


def _color_cycle(rng: random.Random, palette: list[int], count: int) -> list[int]:
    """``count`` colors that use every entry of ``palette`` at least once."""
    colors = list(palette)
    colors.extend(rng.choice(palette) for _ in range(max(0, count - len(palette))))
    rng.shuffle(colors)
    return colors


# Fraction of a scattered grid's cells that are painted. A difficulty axis in
# principle, but not a wired one: driving it from config previously made it mean
# two opposite things, since the motif sampler reads the same number as a
# keep-probability, where higher is *easier*.
_SCATTER_DENSITY = 0.2
# Probability that a motif cell is painted rather than left as background.
_MOTIF_KEEP_PROB = 0.7


def pattern_scatter(
    rng: random.Random, palette: list[int], max_dim: int
) -> Grid | None:
    """Loose points on a background."""
    height, width = _size(rng, max_dim)
    count = max(len(palette), int(round(_SCATTER_DENSITY * height * width)))
    if count > height * width:
        return None
    cells = rng.sample([(r, c) for r in range(height) for c in range(width)], count)
    grid = _blank(height, width)
    for (r, c), color in zip(cells, _color_cycle(rng, palette, count)):
        grid[r][c] = color
    return grid


def _place(
    rng: random.Random,
    grid: Grid,
    occupied: list[list[bool]],
    shape: Grid,
    margin: int,
) -> bool:
    """Stamp ``shape`` somewhere free, keeping ``margin`` cells clear around it.

    The margin is what guarantees the shapes stay separate objects: without it
    two rectangles can fuse, and "remove the isolated cells" or "fill the
    enclosed region" stop having a single well-defined answer.
    """
    height, width = len(grid), len(grid[0])
    shape_h, shape_w = len(shape), len(shape[0])
    if height - shape_h - 2 * margin < 0 or width - shape_w - 2 * margin < 0:
        return False
    for _ in range(_MAX_PLACEMENT_TRIES):
        top = rng.randint(margin, height - shape_h - margin)
        left = rng.randint(margin, width - shape_w - margin)
        if any(
            occupied[r][c]
            for r in range(max(0, top - margin), min(height, top + shape_h + margin))
            for c in range(max(0, left - margin), min(width, left + shape_w + margin))
        ):
            continue
        for r in range(shape_h):
            for c in range(shape_w):
                occupied[top + r][left + c] = True
                if shape[r][c] != BACKGROUND:
                    grid[top + r][left + c] = shape[r][c]
        return True
    return False


def _filled_rect(height: int, width: int, color: int) -> Grid:
    return [[color] * width for _ in range(height)]


def _hollow_rect(height: int, width: int, color: int) -> Grid:
    return [
        [
            color if r in (0, height - 1) or c in (0, width - 1) else BACKGROUND
            for c in range(width)
        ]
        for r in range(height)
    ]


def objects_pattern(
    margin: int = 1, hollow: bool = False
) -> Callable[..., Grid | None]:
    """Sampler factory: a few rectangles on a background.

    ``margin`` > 0 leaves a background frame, which is what makes
    crop-to-bounding-box a real transformation. ``hollow`` produces rectangles
    with an interior, which is what makes fill-enclosed one.

    One color per rectangle, so a rectangle is one object under any reading --
    the object count is therefore the palette size, give or take one, rather
    than an axis of its own. Separating the two needs a definition of "object"
    that this ladder does not yet have.
    """

    def sample(rng: random.Random, palette: list[int], max_dim: int) -> Grid | None:
        height, width = _size(rng, max_dim, min_dim=MIN_OBJECT_GRID_DIM)
        grid = _blank(height, width)
        occupied = [[False] * width for _ in range(height)]
        colors = _color_cycle(rng, palette, len(palette) + rng.randint(0, 1))
        for color in colors:
            low = 3 if hollow else 2
            shape_h = rng.randint(low, min(low + 2, height - 2 * margin))
            shape_w = rng.randint(low, min(low + 2, width - 2 * margin))
            builder = _hollow_rect if hollow else _filled_rect
            if not _place(
                rng, grid, occupied, builder(shape_h, shape_w, color), margin
            ):
                return None
        return grid

    return sample


def pattern_lines(rng: random.Random, palette: list[int], max_dim: int) -> Grid | None:
    """Full rows and columns, which is where crosses and stripes come from."""
    height, width = _size(rng, max_dim)
    grid = _blank(height, width)
    for color in _color_cycle(rng, palette, len(palette)):
        if rng.random() < 0.5:
            grid[rng.randrange(height)] = [color] * width
        else:
            column = rng.randrange(width)
            for row in grid:
                row[column] = color
    return grid


def pattern_motif(rng: random.Random, palette: list[int], max_dim: int) -> Grid | None:
    """A small motif repeated to fill the grid."""
    height, width = _size(rng, max_dim)
    motif_h = rng.randint(1, min(3, height))
    motif_w = rng.randint(1, min(3, width))
    count = motif_h * motif_w
    if count < len(palette):
        return None
    colors = _color_cycle(rng, palette, count)
    motif = [
        [
            colors[r * motif_w + c] if rng.random() < _MOTIF_KEEP_PROB else BACKGROUND
            for c in range(motif_w)
        ]
        for r in range(motif_h)
    ]
    grid = [
        [motif[r % motif_h][c % motif_w] for c in range(width)] for r in range(height)
    ]
    # The masking above can erase a palette color from every copy of the motif.
    if {cell for row in grid for cell in row} - {BACKGROUND} != set(palette):
        return None
    return grid


def pattern_noisy_objects(
    rng: random.Random, palette: list[int], max_dim: int
) -> Grid | None:
    """Solid rectangles plus isolated single cells -- the denoise input.

    The specks are placed with a one-cell margin so they are genuinely isolated
    under the 4-neighbor rule, and the rectangles are at least 2x2 so none of
    their own cells is.

    The specks are the transformation's *signal*, not distractors: denoise is
    the rule "remove exactly these". Do not wire a distractor axis to them: a
    distractor is something the rule ignores, which these are not.
    """
    grid = objects_pattern(margin=1)(rng, palette, max_dim)
    if grid is None:
        return None
    height, width = len(grid), len(grid[0])
    occupied = [
        [
            any(
                grid[r + dr][c + dc] != BACKGROUND
                for dr in (-1, 0, 1)
                for dc in (-1, 0, 1)
                if 0 <= r + dr < height and 0 <= c + dc < width
            )
            for c in range(width)
        ]
        for r in range(height)
    ]
    speck_colors = _color_cycle(rng, palette, rng.randint(2, 5))
    placed = 0
    for color in speck_colors:
        if _place(rng, grid, occupied, [[color]], margin=1):
            placed += 1
    if placed == 0:
        return None
    return grid


def symmetric_holes_pattern(axis: str) -> Callable[..., Grid | None]:
    """Sampler factory: a symmetric picture with part of one side erased."""
    mirror = flip_h if axis == "horizontal" else flip_v

    def sample(rng: random.Random, palette: list[int], max_dim: int) -> Grid | None:
        base = pattern_scatter(rng, palette, max_dim)
        if base is None:
            return None
        reflected = mirror(base)
        symmetric = [
            [
                cell if cell != BACKGROUND else reflected[r][c]
                for c, cell in enumerate(row)
            ]
            for r, row in enumerate(base)
        ]
        # Erase from one half only, so the other half still carries the answer.
        height, width = len(symmetric), len(symmetric[0])
        grid = identity(symmetric)
        painted = [
            (r, c)
            for r in range(height)
            for c in range(width)
            if symmetric[r][c] != BACKGROUND
            and (c >= width // 2 if axis == "horizontal" else r >= height // 2)
        ]
        if not painted:
            return None
        for r, c in rng.sample(painted, max(1, len(painted) // 2)):
            grid[r][c] = BACKGROUND
        if {cell for row in grid for cell in row} - {BACKGROUND} != set(palette):
            return None
        return grid

    return sample


# ---------------------------------------------------------------------------
# Rules
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Rule:
    """One concrete, fully-parameterized transformation plus its input source.

    ``required_colors`` is what the identifiability guard checks: "drop color 3"
    cannot be inferred from examples that contain no 3, and a task whose rule is
    not pinned down by its own examples is unsolvable in principle -- which
    during a training run is indistinguishable from the model failing to learn.
    """

    name: str
    level: int
    stages: tuple[Callable[[Grid], Grid], ...]
    sample_input: Callable[[random.Random], Grid | None]
    required_colors: frozenset[int] = frozenset()

    def apply(self, grid: Grid) -> Grid:
        """Run every stage in order."""
        for stage in self.stages:
            grid = stage(grid)
        return grid

    def trace(self, grid: Grid) -> list[Grid]:
        """The grid after each stage, for the per-stage degeneracy guard."""
        out = []
        for stage in self.stages:
            grid = stage(grid)
            out.append(grid)
        return out


def rule_family(rule_name: str) -> str:
    """Return the curriculum family represented by a concrete rule name."""
    if "+" in rule_name:
        return "composition"
    operation = rule_name.partition("(")[0]
    if operation == "identity":
        return "identity"
    if operation in DIHEDRAL:
        return "dihedral"
    if operation in {"drop_color", "recolor", "keep_only"}:
        return "color"
    if operation in {"tile", "crop_to_bbox", "scale", "add_border"}:
        return "geometric"
    if operation in {"denoise", "complete_symmetry", "fill_enclosed"}:
        return "structure"
    raise ValueError(f"cannot classify unknown ARC rule {rule_name!r}")


def rule_composition_depth(rule_name: str) -> int:
    """Return the number of ordered stages encoded in a concrete rule name."""
    return len(rule_name.split("+"))


def _describe_rule_stage(stage_name: str) -> str:
    """Render one concrete generator stage as an operational instruction."""
    fixed = {
        "identity": "Copy every cell unchanged and preserve the input height and width.",
        "rot90": "Rotate the entire grid 90 degrees clockwise; the output height is the input width.",
        "rot180": "Rotate the entire grid 180 degrees.",
        "rot270": "Rotate the entire grid 90 degrees counterclockwise; the output height is the input width.",
        "flip_h": "Reflect the grid left-to-right by reversing the cells within every row.",
        "flip_v": "Reflect the grid top-to-bottom by reversing the order of the rows.",
        "transpose": "Reflect across the main diagonal by swapping every cell's row and column indices.",
        "anti_transpose": (
            "Reflect across the anti-diagonal: transpose the grid and then rotate that result 180 degrees."
        ),
        "crop_to_bbox": (
            "Crop away all outer rows and columns containing only black cells, leaving the smallest rectangle that "
            "contains every non-black cell."
        ),
        "denoise": (
            "Replace a non-black cell with black exactly when all of its existing up, down, left, and right "
            "neighbors are black; leave every other cell unchanged."
        ),
    }
    if stage_name in fixed:
        return fixed[stage_name]

    match = re.fullmatch(r"drop_color\((\d)\)", stage_name)
    if match:
        return f"Replace every cell of color {match.group(1)} with black (0), leaving all other cells unchanged."
    match = re.fullmatch(r"recolor\((\d)->(\d)\)", stage_name)
    if match:
        return f"Replace every cell of color {match.group(1)} with color {match.group(2)}, leaving all others unchanged."
    match = re.fullmatch(r"keep_only\((\d)\)", stage_name)
    if match:
        return f"Keep cells of color {match.group(1)} and replace every cell of every other color with black (0)."
    match = re.fullmatch(r"tile\((\d)x(\d)\)", stage_name)
    if match:
        vertical, horizontal = match.groups()
        return (
            f"Tile the whole input grid in a {vertical}-by-{horizontal} arrangement: {vertical} copies vertically "
            f"and {horizontal} copies horizontally."
        )
    match = re.fullmatch(r"scale\((\d)\)", stage_name)
    if match:
        factor = match.group(1)
        return f"Scale the grid by replacing each cell with a {factor}-by-{factor} block of that same color."
    match = re.fullmatch(r"add_border\((\d)\)", stage_name)
    if match:
        return f"Add a one-cell-wide border of color {match.group(1)} around all four sides of the unchanged grid."
    match = re.fullmatch(r"complete_symmetry\((horizontal|vertical)\)", stage_name)
    if match:
        if match.group(1) == "horizontal":
            direction = "left-to-right"
        else:
            direction = "top-to-bottom"
        return (
            f"Complete {direction} mirror symmetry: for each black cell, copy the color from its mirrored position, "
            "and never overwrite a non-black cell."
        )
    match = re.fullmatch(r"fill_enclosed\((\d)\)", stage_name)
    if match:
        return (
            f"Fill every black region that cannot reach the outer border through up/down/left/right black cells with "
            f"color {match.group(1)}; leave border-connected black cells and all colored cells unchanged."
        )
    raise ValueError(f"cannot describe unknown ARC rule stage {stage_name!r}")


def render_oracle_description(rule_name: str, *, paraphrase_id: int = 0) -> str:
    """Render a canonical, executable textual description of a sampled rule.

    The three deterministic wrappers support a held-out wording split without
    changing the underlying operation. Concrete colors, dimensions, and stage
    order come from the sampled rule parameters rather than from its examples.
    """
    if paraphrase_id not in {0, 1, 2}:
        raise ValueError("paraphrase_id must be 0, 1, or 2")
    stages = [_describe_rule_stage(stage_name) for stage_name in rule_name.split("+")]
    if paraphrase_id == 0:
        heading = "Apply the following operation to each input grid independently."
    elif paraphrase_id == 1:
        heading = "Operational transformation specification for every supplied grid:"
    else:
        heading = "Produce each output mechanically from its corresponding input using these ordered steps:"
    if len(stages) == 1:
        return f"{heading}\n{stages[0]}"
    numbered = "\n".join(
        f"{index}. {stage}" for index, stage in enumerate(stages, start=1)
    )
    return f"{heading}\n{numbered}\nComplete the stages in that order; stage 2 consumes the result of stage 1."


def _bind(
    sampler: Callable[..., Grid | None], palette: list[int], max_dim: int
) -> Callable[[random.Random], Grid | None]:
    return lambda rng: sampler(rng, palette, max_dim)


def _generic_sampler(rng: random.Random) -> Callable[..., Grid | None]:
    return rng.choice(
        [pattern_scatter, pattern_lines, pattern_motif, objects_pattern(margin=1)]
    )


def _level1_rule(rng: random.Random, palette: list[int], max_dim: int) -> Rule:
    name = rng.choice([key for key in DIHEDRAL if key != "identity"])
    return Rule(
        name=name,
        level=1,
        stages=(DIHEDRAL[name],),
        sample_input=_bind(_generic_sampler(rng), palette, max_dim),
    )


def _level2_rule(rng: random.Random, palette: list[int], max_dim: int) -> Rule:
    source = rng.choice(palette)
    kind = rng.choice(["drop_color", "recolor", "keep_only"])
    if kind == "drop_color":
        name, apply = f"drop_color({source})", drop_color(source)
    elif kind == "keep_only":
        name, apply = f"keep_only({source})", keep_only(source)
    else:
        # Recolor to a color the input never uses: mapping onto an existing one
        # merges two objects, and then the examples no longer say which of the
        # two the rule was about.
        destination = _spare_color(rng, palette)
        name, apply = f"recolor({source}->{destination})", recolor(source, destination)
    return Rule(
        name=name,
        level=2,
        stages=(apply,),
        sample_input=_bind(_generic_sampler(rng), palette, max_dim),
        required_colors=frozenset({source}),
    )


def _level3_rule(rng: random.Random, palette: list[int], max_dim: int) -> Rule:
    kind = rng.choice(["tile", "crop", "scale", "border"])
    if kind == "tile":
        vertical, horizontal = rng.randint(1, 3), rng.randint(1, 3)
        if (vertical, horizontal) == (1, 1):
            vertical = 2
        return Rule(
            name=f"tile({vertical}x{horizontal})",
            level=3,
            stages=(tile(vertical, horizontal),),
            sample_input=_bind(
                _generic_sampler(rng),
                palette,
                min(max_dim, MAX_GRID_DIM // max(vertical, horizontal)),
            ),
        )
    if kind == "scale":
        factor = rng.randint(2, 3)
        return Rule(
            name=f"scale({factor})",
            level=3,
            stages=(scale(factor),),
            sample_input=_bind(
                _generic_sampler(rng), palette, min(max_dim, MAX_GRID_DIM // factor)
            ),
        )
    if kind == "border":
        color = _spare_color(rng, palette)
        return Rule(
            name=f"add_border({color})",
            level=3,
            stages=(add_border(color),),
            sample_input=_bind(
                _generic_sampler(rng), palette, min(max_dim, MAX_GRID_DIM - 2)
            ),
        )
    # Crop needs a background frame to crop away, so it gets the margin sampler
    # rather than a generic one.
    return Rule(
        name="crop_to_bbox",
        level=3,
        stages=(crop_to_bbox,),
        sample_input=_bind(objects_pattern(margin=1), palette, max_dim),
    )


def _level4_rule(rng: random.Random, palette: list[int], max_dim: int) -> Rule:
    kind = rng.choice(["denoise", "symmetry", "fill"])
    if kind == "denoise":
        return Rule(
            name="denoise",
            level=4,
            stages=(denoise,),
            sample_input=_bind(pattern_noisy_objects, palette, max_dim),
        )
    if kind == "symmetry":
        axis = rng.choice(["horizontal", "vertical"])
        return Rule(
            name=f"complete_symmetry({axis})",
            level=4,
            stages=(complete_symmetry(axis),),
            sample_input=_bind(symmetric_holes_pattern(axis), palette, max_dim),
        )
    color = _spare_color(rng, palette)
    return Rule(
        name=f"fill_enclosed({color})",
        level=4,
        stages=(fill_enclosed(color),),
        sample_input=_bind(objects_pattern(margin=1, hollow=True), palette, max_dim),
    )


def _level5_rule(rng: random.Random, palette: list[int], max_dim: int) -> Rule:
    """A color op followed by a shape op.

    Always in that order, and never shape-then-color: a color op can erase the
    very color a following color op is parameterized on, and then the second
    stage is not identifiable from the examples. Shape ops are color-agnostic,
    so composing one after a color op can never do that.

    Both stages share one palette, and the *shape* stage supplies the input
    sampler -- it is the stage with an opinion about size (tile and scale bound
    the input so the output still fits) and about structure (crop needs a
    background border to crop away).
    """
    color = _level2_rule(rng, palette, max_dim)
    shape = rng.choice([_level1_rule, _level3_rule])(rng, palette, max_dim)
    return Rule(
        name=f"{color.name}+{shape.name}",
        level=5,
        stages=color.stages + shape.stages,
        sample_input=shape.sample_input,
        required_colors=color.required_colors,
    )


_LEVEL_RULES: dict[int, Callable[[random.Random, list[int], int], Rule]] = {
    1: _level1_rule,
    2: _level2_rule,
    3: _level3_rule,
    4: _level4_rule,
    5: _level5_rule,
}


# ---------------------------------------------------------------------------
# Guards
# ---------------------------------------------------------------------------


def grid_colors(grid: Grid) -> set[int]:
    return {cell for row in grid for cell in row}


def rule_is_identifiable(rule: Rule, train_pairs: list[Pair]) -> bool:
    """Is the rule pinned down by the few-shot pairs alone?

    Only the *train* pairs count -- they are all the model is shown. A rule
    whose parameter never appears in them is unsolvable however good the policy
    is, and in a training run that reads as the model failing to learn.
    """
    return all(rule.required_colors <= grid_colors(inp) for inp, _ in train_pairs)


def is_degenerate(rule: Rule, pairs: list[Pair]) -> bool:
    """Does any stage of the rule leave any pair's grid unchanged?

    Stricter than the "every example is unchanged" test in two ways, both
    load-bearing:

    - *Per pair.* One identity pair inside a non-identity task is itself an
      ambiguity, and a task whose *test* pair is unchanged is solved by echoing
      the input -- which is exactly the behavior four previous runs collapsed
      into.
    - *Per stage.* A composition hides a no-op that a whole-rule check would
      miss: ``recolor(4->6)`` then ``flip_v`` on a grid of identical rows is a
      recolor wearing a level-5 label, and the flip is not inferable from any
      example.

    """
    for inp, _ in pairs:
        current = inp
        for stage in rule.trace(inp):
            if stage == current:
                return True
            current = stage
    return False


def _fits(grid: Grid) -> bool:
    return (
        1 <= len(grid) <= MAX_GRID_DIM
        and bool(grid[0])
        and len(grid[0]) <= MAX_GRID_DIM
    )


# ---------------------------------------------------------------------------
# Task generation
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SynthTask:
    """One generated task, shaped like a row of the real ARC dataset."""

    task_id: str
    level: int
    rule: str
    train_pairs: list[dict[str, Grid]]
    test_input: Grid
    target: Grid


def _task_seed(seed: int, index: int, level: int) -> int:
    return (seed * 1_000_003 + index) * 131 + level


def _sample_pairs(rng: random.Random, rule: Rule, count: int) -> list[Pair] | None:
    pairs: list[Pair] = []
    for _ in range(count):
        inp = rule.sample_input(rng)
        if inp is None or not _fits(inp):
            return None
        out = rule.apply(inp)
        if not _fits(out):
            return None
        pairs.append((inp, out))
    return pairs


def generate_task(
    seed: int,
    index: int,
    level: int,
    *,
    num_train_pairs: int | None = None,
    max_input_dim: int = DEFAULT_MAX_INPUT_DIM,
) -> SynthTask:
    """Generate the task at ``(seed, index, level)``.

    Deterministic in its arguments and nothing else, so a run is reproducible
    without materializing a dataset.

    Rejection is the normal path: a rot180 that lands on a symmetric grid, a
    "drop color 3" whose examples contain no 3, and a crop of a grid with no
    background border are all generated and thrown away. Exhausting the attempt
    budget means the level's rules and patterns disagree with each other, which
    is a bug worth crashing on rather than a task worth returning.
    """
    if level not in _LEVEL_RULES:
        raise ValueError(
            f"unknown level {level}; expected one of {sorted(_LEVEL_RULES)}"
        )
    if num_train_pairs is not None and not 2 <= num_train_pairs <= 4:
        raise ValueError("num_train_pairs must be between 2 and 4")
    if max_input_dim < MIN_GRID_DIM:
        raise ValueError(f"max_input_dim must be at least {MIN_GRID_DIM}")
    rng = random.Random(_task_seed(seed, index, level))

    low, high = _PALETTE_RANGE[level]
    if level in _SPARE_COLOR_LEVELS and high >= len(COLORS):
        raise ValueError(
            f"level {level} draws a color outside its palette, so its palette "
            f"range {(low, high)} must stay below {len(COLORS)}"
        )

    for _ in range(_MAX_ATTEMPTS):
        palette = rng.sample(COLORS, rng.randint(low, high))
        rule = _LEVEL_RULES[level](rng, palette, max_input_dim)
        count = (num_train_pairs or rng.randint(2, 4)) + 1
        pairs = _sample_pairs(rng, rule, count)
        if pairs is None:
            continue
        if not rule_is_identifiable(rule, pairs[:-1]):
            continue
        if is_degenerate(rule, pairs):
            continue

        train, (test_input, target) = pairs[:-1], pairs[-1]
        return SynthTask(
            # The rule goes in the id so a dumped validation row says which
            # transformation it was -- the difference between "level 3 is hard"
            # and "tile is hard" -- without needing its own dataset column.
            task_id=(
                f"synth_L{level}_d{max_input_dim}_f{count - 1}"
                f"_{rule.name}_{seed}_{index}"
            ),
            level=level,
            rule=rule.name,
            train_pairs=[{"input": inp, "output": out} for inp, out in train],
            test_input=test_input,
            target=target,
        )

    raise RuntimeError(
        f"no valid task at level {level} in {_MAX_ATTEMPTS} attempts "
        f"(seed={seed}, index={index}) -- the level's rules and input patterns "
        "are inconsistent"
    )


def generate_tasks(
    seed: int,
    count: int,
    levels: list[int],
    *,
    num_train_pairs: int | list[int] | None = None,
    max_input_dim: int | list[int] = DEFAULT_MAX_INPUT_DIM,
    level_difficulty_order: list[int] | None = None,
    difficulty_ramp_window: int | None = None,
    difficulty_ramp_steps: int | None = None,
) -> list[SynthTask]:
    """Generate ``count`` tasks over the joint transform x size schedule.

    ``level_difficulty_order`` ranks the levels by *measured* solve rate rather
    than authored index -- the numbering is a guess, and M4.2 measured level 2 as
    easier than level 1. That rank is crossed with every ``max_input_dim``, and
    the resulting combinations are reweighted from easy to hard across
    ``difficulty_ramp_steps`` while every complete ``difficulty_ramp_window``
    still contains all of them.

    Keeping the full range in every window is the point. GRPO's advantage is
    computed within a group of rollouts on one prompt, so a phase whose tasks
    are uniformly hopeless -- or uniformly trivial -- contributes no gradient,
    which is the same degenerate-group failure the curriculum exists to fix. A
    plain ramp reintroduces it; reweighting a mixture does not.

    ``difficulty_ramp_steps`` must be the *run* length, not the dataset length.
    The dataset is deliberately many times larger than any run so that no task
    repeats, so a ramp spread over it leaves a 60-step run at 12% of its
    schedule -- inert, but still reading as configured. Rows past the end of the
    ramp stay at the final, hardest mixture.

    The schedule lives in the row order, so it needs ``data.shuffle: false``.
    Omit ``difficulty_ramp_window`` (as validation does) and the cross product is
    cycled at fixed weights instead, so the held-out mixture does not move
    between checkpoints.

    A level repeated in ``levels`` is weighted proportionally: it keeps one slot
    per window like every other combination, and takes that multiple of the
    remaining mass.

    ``num_train_pairs`` may be a list ordered easy -> hard (more demonstrations
    are easier); it is selected by the same joint difficulty rank. It is the one
    axis beyond transform and size that is wired end to end. Five others
    (palette size, object count, density, composition depth, distractors) were
    tried and removed for being inert or contradictory on most levels.
    """
    if not levels:
        raise ValueError("levels must not be empty")
    unknown = sorted(set(levels) - set(LEVELS))
    if unknown:
        raise ValueError(f"unknown levels {unknown}; the ladder has {list(LEVELS)}")
    dims = [max_input_dim] if isinstance(max_input_dim, int) else list(max_input_dim)
    if not dims:
        raise ValueError("max_input_dim must not be an empty list")
    if level_difficulty_order is None:
        # Authored index is the fallback ranking, and it is explicitly a guess.
        level_difficulty_order = list(LEVELS)

    level_sizes, difficulty_fractions = _joint_level_size_schedule(
        count=count,
        levels=levels,
        dims=dims,
        level_difficulty_order=level_difficulty_order,
        window=difficulty_ramp_window,
        ramp_windows=difficulty_ramp_steps,
    )
    train_pair_values = _axis_values(num_train_pairs)

    return [
        generate_task(
            seed,
            index,
            level,
            num_train_pairs=_difficulty_axis_value(train_pair_values, difficulty),
            max_input_dim=size,
        )
        for index, ((level, size), difficulty) in enumerate(
            zip(level_sizes, difficulty_fractions)
        )
    ]


def _axis_values(value: AxisT | list[AxisT] | None) -> list[AxisT | None]:
    """An axis as the list of values it takes, easiest first.

    ``None`` is a value, not an absence: it is how "let the level decide" is
    spelled for ``num_train_pairs``, so a one-element ``[None]`` is correct.
    """
    if value is None:
        return [None]
    values: list[AxisT | None] = []
    if isinstance(value, list):
        values.extend(value)
    else:
        values.append(value)
    if not values:
        raise ValueError("difficulty-axis lists must not be empty")
    return values


def _difficulty_axis_value(
    values: list[AxisT | None], difficulty: float
) -> AxisT | None:
    """The value this axis takes at a joint difficulty in [0, 1]."""
    return values[round(difficulty * (len(values) - 1))]


def _joint_level_size_schedule(
    *,
    count: int,
    levels: list[int],
    dims: list[int],
    level_difficulty_order: list[int],
    window: int | None,
    ramp_windows: int | None,
) -> tuple[list[tuple[int, int]], list[float]]:
    """Cross measured transform difficulty with grid size into one schedule.

    Returns the ``(level, size)`` for each row and that row's joint difficulty in
    [0, 1], which the remaining axes are selected by.
    """
    if len(set(level_difficulty_order)) != len(level_difficulty_order):
        raise ValueError("level_difficulty_order must not contain duplicates")
    missing = sorted(set(levels) - set(level_difficulty_order))
    if missing:
        raise ValueError(f"level_difficulty_order is missing active levels {missing}")

    # A repeated level is a weight, not a second combination: the cross product
    # is built from the distinct levels and the multiplicity scales the mass
    # that level receives.
    multiplicity: Counter[int] = Counter(levels)
    active_order = [level for level in level_difficulty_order if level in multiplicity]
    unique_dims = sorted(set(dims))
    level_rank = {level: rank for rank, level in enumerate(active_order)}
    dim_rank = {dim: rank for rank, dim in enumerate(unique_dims)}
    level_denominator = max(len(active_order) - 1, 1)
    dim_denominator = max(len(unique_dims) - 1, 1)

    choices = [(level, dim) for level in active_order for dim in unique_dims]
    scores = {
        (level, dim): (
            level_rank[level] / level_denominator + dim_rank[dim] / dim_denominator
        )
        / 2.0
        for level, dim in choices
    }
    choices.sort(key=lambda choice: (scores[choice], level_rank[choice[0]], choice[1]))
    weights = [float(multiplicity[level]) for level, _ in choices]

    if window:
        scheduled = ramped_choices(
            count=count,
            choices=choices,
            multipliers=weights,
            window=window,
            ramp_windows=ramp_windows,
        )
    else:
        # No ramp: cycle at fixed weights, which is what the held-out split
        # wants -- its mixture must not move between checkpoints. Multiplicity
        # is honored here too, by giving a repeated level that many slots in the
        # cycle, so `levels` means the same thing on both splits.
        cycle = [choice for choice in choices for _ in range(multiplicity[choice[0]])]
        scheduled = [cycle[index % len(cycle)] for index in range(count)]
    return scheduled, [scores[choice] for choice in scheduled]


def ramped_choices(
    *,
    count: int,
    choices: list[ChoiceT],
    multipliers: list[float],
    window: int,
    ramp_windows: int | None = None,
) -> list[ChoiceT]:
    """Reweight ordered choices while retaining all of them in every window.

    Every window keeps at least one of every choice, so no batch is ever
    uniformly hopeless or uniformly trivial. What moves is the *weighting* of
    the remaining slots: a hump centred on the easiest choice at the start of
    the run and the hardest at the end, scaled by each choice's multiplier.

    ``window`` is the number of prompts the trainer consumes per step;
    ``ramp_windows`` is how many steps the schedule spans. Largest-remainder
    apportionment, so the counts sum to the window exactly.
    """
    if window < len(choices):
        raise ValueError(
            f"difficulty_ramp_window {window} is smaller than the "
            f"{len(choices)} level/size combinations"
        )
    windows = max(1, -(-count // window))
    span = max(1, ramp_windows or windows)
    out: list[ChoiceT] = []
    for window_index in range(windows):
        progress = min(window_index, span - 1) / max(span - 1, 1)
        centre = progress * (len(choices) - 1)
        weights = [
            multiplier * 2.0 ** -abs(index - centre)
            for index, multiplier in enumerate(multipliers)
        ]
        total = sum(weights)
        current_size = min(window, count - window_index * window)
        spare = max(0, current_size - len(choices))
        exact = [spare * weight / total for weight in weights]
        counts = [1 + int(extra) for extra in exact]
        remainder = spare - sum(int(extra) for extra in exact)
        order = sorted(
            range(len(choices)),
            key=lambda index: exact[index] - int(exact[index]),
            reverse=True,
        )
        for index in order[:remainder]:
            counts[index] += 1
        row = [choice for choice, copies in zip(choices, counts) for _ in range(copies)]
        out.extend(row[:current_size])
    return out[:count]
