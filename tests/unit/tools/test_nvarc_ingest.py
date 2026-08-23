import pytest

from nemo_rl.environments import arc_agi_grid
from tools.nvarc_ingest import (
    CANONICAL_SECTIONS,
    MAX_GRID_DIM,
    NUM_COLORS,
    assign_splits,
    parse_new_puzzle_sections,
    puzzle_difficulty,
    render_canonical_rule,
    select_mixture_row,
    validate_grid,
)


def _completion(**overrides) -> str:
    sections = {
        "rules_summary": "Rotate the grid.",
        "input_generation": "Random 5x5 grids.",
        "solution_steps": "1. Rotate 90 degrees.",
        "key_insight": "Rotation preserves shape.",
        "puzzle_concepts": "- rotation",
    }
    sections.update(overrides)
    body = "\n".join(f"<{k}>\n{v}\n</{k}>" for k, v in sections.items())
    return f"<puzzle_analysis>\nscratch\n</puzzle_analysis>\n<new_puzzle>\n{body}\n</new_puzzle>"


def test_grid_constraints_match_arc_agi_grid() -> None:
    # The tool restates these so it can run without the repo venv; they must
    # never drift from the environment's definitions.
    assert MAX_GRID_DIM == arc_agi_grid.MAX_GRID_DIM
    assert NUM_COLORS == arc_agi_grid.NUM_COLORS


def test_parse_sections_prefers_new_puzzle_block() -> None:
    completion = (
        "<puzzle_analysis><rules_summary>WRONG</rules_summary></puzzle_analysis>"
        + _completion()
    )
    sections = parse_new_puzzle_sections(completion)
    assert sections is not None
    assert sections["rules_summary"] == "Rotate the grid."
    assert sections["input_generation"] == "Random 5x5 grids."


def test_parse_sections_requires_every_canonical_section() -> None:
    completion = _completion()
    for name in CANONICAL_SECTIONS:
        broken = completion.replace(f"<{name}>", "<x>").replace(f"</{name}>", "</x>")
        assert parse_new_puzzle_sections(broken) is None
    # input_generation is metadata, not required
    no_input_gen = _completion(input_generation="")
    assert parse_new_puzzle_sections(no_input_gen) is not None


def test_parse_sections_without_wrapper_falls_back_to_whole_text() -> None:
    body = _completion().split("<new_puzzle>")[1].split("</new_puzzle>")[0]
    assert parse_new_puzzle_sections(body) is not None


def test_render_canonical_rule_excludes_input_generation() -> None:
    sections = parse_new_puzzle_sections(_completion())
    rendered = render_canonical_rule(sections)
    assert "input_generation" not in rendered
    assert "Random 5x5 grids." not in rendered
    for name in CANONICAL_SECTIONS:
        assert f"<{name}>" in rendered and f"</{name}>" in rendered


def test_select_mixture_row_is_order_independent() -> None:
    rows = [{"completion": "aaa"}, {"completion": "bbb"}]
    assert select_mixture_row(rows) == select_mixture_row(rows[::-1])


@pytest.mark.parametrize(
    "grid,expected",
    [
        ([[0, 1], [2, 3]], True),
        ([[9] * 30] * 30, True),
        ([], False),
        ([[]], False),
        ([[0, 1], [2]], False),  # ragged
        ([[10]], False),  # out of palette
        ([[-1]], False),
        ([[True]], False),  # bool is not a color
        ([[0.5]], False),
        ([[0] * 31], False),  # too wide
        ([[0]] * 31, False),  # too tall
        ("nope", False),
    ],
)
def test_validate_grid(grid, expected) -> None:
    assert validate_grid(grid) is expected


def test_puzzle_difficulty_is_max_area_over_all_grids() -> None:
    pairs = [
        {"input": [[0, 1]], "output": [[0], [1], [2]]},
        {"input": [[0] * 4] * 5, "output": [[1]]},
    ]
    difficulty, max_dim = puzzle_difficulty(pairs)
    assert difficulty == 20
    assert max_dim == 5


def test_assign_splits_is_deterministic_and_disjoint() -> None:
    ids = [f"p{i}" for i in range(100)]
    first = assign_splits(ids, seed=7, val_count=10, proposer_eval_count=20)
    second = assign_splits(ids[::-1], seed=7, val_count=10, proposer_eval_count=20)
    assert first == second  # enumeration order must not matter

    by_split: dict[str, set] = {}
    for puzzle_id, split in first.items():
        by_split.setdefault(split, set()).add(puzzle_id)
    assert len(by_split["executor_val"]) == 10
    assert len(by_split["proposer_eval"]) == 20
    assert len(by_split["train"]) == 70

    different = assign_splits(ids, seed=8, val_count=10, proposer_eval_count=20)
    assert first != different


def test_assign_splits_rejects_oversized_holdout() -> None:
    with pytest.raises(ValueError, match="cannot hold out"):
        assign_splits(["a", "b"], seed=1, val_count=1, proposer_eval_count=1)
