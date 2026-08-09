import random

import pytest

from nemo_rl.environments.arc_agi_grid import (
    RewardWeights,
    color_recall,
    extract_answer_grid,
    extraneous_color_fraction,
    format_task_prompt,
    overlay_cell_accuracy,
    parse_grid,
    score_response,
    serialize_grid,
    shape_mismatch,
)

WEIGHTS = RewardWeights(
    exact=1.0, cell=0.20, color=0.05, extraneous=0.05, shape=0.05, format=0.05
)


def answer(text: str) -> str:
    return f"some free-form reasoning\n<answer>\n{text}\n</answer>"


class TestParseGrid:
    def test_well_formed(self):
        assert parse_grid("012\n345") == [[0, 1, 2], [3, 4, 5]]

    def test_surrounding_whitespace_is_tolerated(self):
        assert parse_grid("\n  012  \n  345  \n") == [[0, 1, 2], [3, 4, 5]]

    @pytest.mark.parametrize(
        "text",
        [
            "",
            "   \n  ",
            "012\n34",  # ragged
            "01a\n345",  # non-digit
            "0 1 2\n3 4 5",  # inner spaces make rows ragged/non-digit
        ],
    )
    def test_malformed_is_rejected(self, text):
        assert parse_grid(text) is None

    def test_oversize_is_rejected(self):
        assert parse_grid("\n".join("0" * 31 for _ in range(2))) is None
        assert parse_grid("\n".join("0" * 2 for _ in range(31))) is None

    def test_max_size_is_accepted(self):
        grid = parse_grid("\n".join("0" * 30 for _ in range(30)))
        assert grid is not None and len(grid) == 30 and len(grid[0]) == 30


class TestExtractAnswerGrid:
    def test_no_delimiters(self):
        assert extract_answer_grid("012\n345") is None

    def test_empty_block(self):
        assert extract_answer_grid("<answer></answer>") is None

    def test_unclosed_block(self):
        assert extract_answer_grid("<answer>\n012\n345") is None

    def test_last_valid_block_wins(self):
        response = answer("00") + answer("11\n11")
        assert extract_answer_grid(response) == [[1, 1], [1, 1]]

    def test_malformed_trailing_block_falls_back_to_earlier_one(self):
        response = answer("11\n11") + "<answer>\nnot a grid\n</answer>"
        assert extract_answer_grid(response) == [[1, 1], [1, 1]]

    def test_reasoning_mentioning_grids_does_not_displace_the_answer(self):
        response = "maybe the output is 000\n111 ?\n" + answer("22")
        assert extract_answer_grid(response) == [[2, 2]]


class TestOverlayCellAccuracy:
    def test_identical(self):
        grid = [[1, 2], [3, 4]]
        assert overlay_cell_accuracy(grid, grid) == 1.0

    def test_fully_disjoint(self):
        assert overlay_cell_accuracy([[1, 1], [1, 1]], [[2, 2], [2, 2]]) == 0.0

    def test_prediction_smaller_still_earns_credit(self):
        target = [[5, 5, 5], [5, 5, 5], [5, 5, 5]]
        # Centered 1x1 lands on the middle cell: 1 of 9 target cells.
        assert overlay_cell_accuracy([[5]], target) == pytest.approx(1 / 9)

    def test_prediction_larger_is_penalized_by_the_max_area_denominator(self):
        target = [[5]]
        pred = [[5, 5, 5], [5, 5, 5], [5, 5, 5]]
        # The single target cell matches, but the denominator is the 9-cell
        # prediction -- padding out cannot buy a high score.
        assert overlay_cell_accuracy(pred, target) == pytest.approx(1 / 9)

    def test_odd_size_delta_uses_floor_offset(self):
        # 1x2 over 1x3: floor((3-2)/2) == 0, so the prediction sits left.
        assert overlay_cell_accuracy([[1, 2]], [[1, 2, 9]]) == pytest.approx(2 / 3)

    def test_off_by_one_row_keeps_most_credit(self):
        target = [[7, 7], [7, 7], [7, 7]]
        pred = [[7, 7], [7, 7]]
        # This is the case a binary or shape-gated scorer throws away entirely.
        assert overlay_cell_accuracy(pred, target) == pytest.approx(4 / 6)


class TestColorTerms:
    def test_full_recall(self):
        assert color_recall([[1, 2]], [[1, 2]]) == 1.0

    def test_partial_recall(self):
        assert color_recall([[1, 1]], [[1, 2]]) == 0.5

    def test_no_extraneous_colors(self):
        assert extraneous_color_fraction([[1, 2]], [[1, 2]]) == 0.0

    def test_extraneous_colors_counted(self):
        # Predicted {1,2,3}; target uses {1}. Two of three predicted are wrong.
        assert extraneous_color_fraction([[1, 2, 3]], [[1]]) == pytest.approx(2 / 3)


class TestShapeMismatch:
    def test_exact_shape(self):
        assert shape_mismatch([[1, 1]], [[2, 2]]) == 0.0

    def test_magnitude_scales(self):
        small = shape_mismatch([[1] * 3] * 3, [[1] * 4] * 4)
        large = shape_mismatch([[1]], [[1] * 4] * 4)
        assert 0 < small < large <= 1.0

    def test_clipped_at_one(self):
        assert shape_mismatch([[1] * 30] * 30, [[1]]) == 1.0


class TestScoreResponse:
    def test_exact_match_scores_highest(self):
        target = [[1, 2], [3, 4]]
        terms = score_response(answer("12\n34"), target, WEIGHTS)
        assert terms["exact_match"] == 1.0
        assert terms["cell_match"] == 1.0
        assert terms["format_valid"] == 1.0
        assert terms["reward"] == pytest.approx(1.0 + 0.20 + 0.05 + 0.05)

    def test_unparseable_response_scores_the_floor(self):
        terms = score_response("no answer here", [[1]], WEIGHTS)
        assert terms["format_valid"] == 0.0
        assert terms["reward"] == pytest.approx(-(WEIGHTS.extraneous + WEIGHTS.shape))

    def test_worst_parseable_answer_still_beats_the_unparseable_floor(self):
        # Worst case for a parseable answer: no cells right, no colors right,
        # maximally wrong shape. It must still clear the floor.
        target = [[1]]
        worst = score_response(answer("2222\n2222\n2222\n2222"), target, WEIGHTS)
        floor = score_response("nothing", target, WEIGHTS)
        assert worst["reward"] > floor["reward"]

    def test_format_only_beats_nothing(self):
        # The whole point of the format term: a parseable but wrong answer must
        # outscore an unparseable one, so groups are non-degenerate at step 0.
        target = [[1, 1], [1, 1]]
        wrong = score_response(answer("22\n22"), target, WEIGHTS)
        unparseable = score_response("nothing", target, WEIGHTS)
        assert wrong["reward"] > unparseable["reward"]

    def test_near_miss_beats_far_miss(self):
        target = [[1, 1, 1], [1, 1, 1], [1, 1, 1]]
        near = score_response(answer("111\n111\n112"), target, WEIGHTS)
        far = score_response(answer("222\n222\n222"), target, WEIGHTS)
        assert near["reward"] > far["reward"]

    def test_exact_match_beats_every_partial(self):
        target = [[1, 2], [3, 4]]
        exact = score_response(answer("12\n34"), target, WEIGHTS)
        for wrong in ("12\n33", "1\n3", "1234", "11\n11"):
            assert exact["reward"] > score_response(answer(wrong), target, WEIGHTS)[
                "reward"
            ]


class TestRewardHacks:
    """The shaped terms exist to break degenerate groups; they must not become
    a cheaper path to reward than actually solving the task."""

    target = [[1, 1, 1], [1, 0, 1], [1, 1, 1]]
    genuine_partial = "111\n101\n110"  # one cell wrong

    def test_copying_the_input_scores_below_a_genuine_partial(self):
        copied = "000\n000\n000"
        assert (
            score_response(answer(copied), self.target, WEIGHTS)["reward"]
            < score_response(answer(self.genuine_partial), self.target, WEIGHTS)[
                "reward"
            ]
        )

    def test_flooding_all_colors_scores_below_a_genuine_partial(self):
        flooded = "012\n345\n678"
        assert (
            score_response(answer(flooded), self.target, WEIGHTS)["reward"]
            < score_response(answer(self.genuine_partial), self.target, WEIGHTS)[
                "reward"
            ]
        )

    def test_padding_to_a_giant_grid_scores_below_a_genuine_partial(self):
        giant = "\n".join("1" * 30 for _ in range(30))
        assert (
            score_response(answer(giant), self.target, WEIGHTS)["reward"]
            < score_response(answer(self.genuine_partial), self.target, WEIGHTS)[
                "reward"
            ]
        )


class TestSerialization:
    def test_round_trip_is_identity(self):
        rng = random.Random(0)
        for _ in range(200):
            height = rng.randint(1, 30)
            width = rng.randint(1, 30)
            grid = [
                [rng.randint(0, 9) for _ in range(width)] for _ in range(height)
            ]
            assert parse_grid(serialize_grid(grid)) == grid

    def test_prompt_layout_contains_every_pair_and_the_test_input(self):
        pairs = [{"input": [[1]], "output": [[2]]}, {"input": [[3]], "output": [[4]]}]
        prompt = format_task_prompt(pairs, [[5]])
        assert prompt.count("<example>") == 2
        assert "<test_input>\n5\n</test_input>" in prompt
        assert "<input>\n1\n</input>" in prompt and "<output>\n4\n</output>" in prompt
