import random

import pytest

from nemo_rl.environments.arc_agi_grid import (
    RewardWeights,
    color_recall,
    edit_similarity,
    extract_answer_grid,
    extraneous_color_fraction,
    format_task_prompt,
    gain_over_baseline,
    overlay_cell_accuracy,
    parse_grid,
    score_response,
    serialize_grid,
    shape_mismatch,
)

WEIGHTS = RewardWeights(
    exact=1.0,
    cell=0.20,
    edit=0.10,
    color=0.05,
    extraneous=0.05,
    shape=0.05,
    format=0.05,
)

# Scoring is relative to echoing the test input, so every score_response call
# needs one. This default shares nothing with the targets used below, which
# keeps the copy baseline at its floor unless a test sets it deliberately.
NO_OVERLAP_INPUT = [[7]]


def answer(text: str) -> str:
    return f"some free-form reasoning\n<answer>\n{text}\n</answer>"


def score(response: str, target, test_input=NO_OVERLAP_INPUT, weights=WEIGHTS):
    return score_response(response, target, test_input, weights)


class TestParseGrid:
    def test_well_formed(self):
        assert parse_grid("0 1 2\n3 4 5") == [[0, 1, 2], [3, 4, 5]]

    def test_contiguous_digits_still_accepted(self):
        # We prompt for spaces, but a compact answer is still a well-formed
        # grid -- rejecting it would discard reward signal over punctuation.
        assert parse_grid("012\n345") == [[0, 1, 2], [3, 4, 5]]

    def test_surrounding_whitespace_is_tolerated(self):
        assert parse_grid("\n  0 1 2  \n  3 4 5  \n") == [[0, 1, 2], [3, 4, 5]]

    def test_irregular_inner_spacing_is_tolerated(self):
        assert parse_grid("0  1\t2\n3 4  5") == [[0, 1, 2], [3, 4, 5]]

    @pytest.mark.parametrize(
        "text",
        [
            "",
            "   \n  ",
            "0 1 2\n3 4",  # ragged
            "012\n34",  # ragged, compact form
            "0 1 a\n3 4 5",  # non-digit
            "01a\n345",  # non-digit, compact form
            "0 12\n3 45",  # multi-digit cell
            "0 1 2\n34",  # mixed forms, widths disagree
        ],
    )
    def test_malformed_is_rejected(self, text):
        assert parse_grid(text) is None

    def test_oversize_is_rejected(self):
        assert parse_grid("\n".join(" ".join("0" * 31) for _ in range(2))) is None
        assert parse_grid("\n".join(" ".join("0" * 2) for _ in range(31))) is None

    def test_max_size_is_accepted(self):
        grid = parse_grid("\n".join(" ".join("0" * 30) for _ in range(30)))
        assert grid is not None and len(grid) == 30 and len(grid[0]) == 30


class TestExtractAnswerGrid:
    def test_no_delimiters(self):
        assert extract_answer_grid("012\n345") is None

    def test_empty_block(self):
        assert extract_answer_grid("<answer></answer>") is None

    def test_unclosed_final_block_is_still_read(self):
        # Generation stops on "</answer>" and may not echo it back; a response
        # truncated at the token cap has no closing tag either.
        assert extract_answer_grid("<answer>\n012\n345") == [[0, 1, 2], [3, 4, 5]]

    def test_unclosed_block_with_no_grid(self):
        assert extract_answer_grid("<answer>\nstill thinking about it") is None

    def test_closed_block_wins_over_later_unclosed_garbage(self):
        response = answer("11\n11") + "<answer>\nnot a grid"
        assert extract_answer_grid(response) == [[1, 1], [1, 1]]

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
        terms = score(answer("12\n34"), target)
        assert terms["grid_match"] == 1.0
        assert terms["cell_match"] == 1.0
        assert terms["format_valid"] == 1.0
        # exact + cell gain + edit gain + color recall + format, no penalties.
        assert terms["cell_gain"] == pytest.approx(1.0)
        assert terms["edit_gain"] == pytest.approx(1.0)
        assert terms["reward"] == pytest.approx(1.0 + 0.20 + 0.10 + 0.05 + 0.05)

    def test_unparseable_response_scores_the_floor(self):
        terms = score("no answer here", [[1]])
        assert terms["format_valid"] == 0.0
        # The gain terms bottom out at -1, so they are part of the floor too --
        # otherwise a badly wrong parseable answer would score below garbage.
        assert terms["reward"] == pytest.approx(
            -(WEIGHTS.cell + WEIGHTS.edit + WEIGHTS.extraneous + WEIGHTS.shape)
        )

    def test_worst_parseable_answer_still_beats_the_unparseable_floor(self):
        # Worst case for a parseable answer: no cells right, no colors right,
        # maximally wrong shape. It must still clear the floor.
        target = [[1]]
        worst = score(answer("2222\n2222\n2222\n2222"), target)
        floor = score("nothing", target)
        assert worst["reward"] > floor["reward"]

    def test_format_only_beats_nothing(self):
        # The whole point of the format term: a parseable but wrong answer must
        # outscore an unparseable one, so groups are non-degenerate at step 0.
        target = [[1, 1], [1, 1]]
        wrong = score(answer("22\n22"), target)
        unparseable = score("nothing", target)
        assert wrong["reward"] > unparseable["reward"]

    def test_near_miss_beats_far_miss(self):
        target = [[1, 1, 1], [1, 1, 1], [1, 1, 1]]
        near = score(answer("111\n111\n112"), target)
        far = score(answer("222\n222\n222"), target)
        assert near["reward"] > far["reward"]

    def test_exact_match_beats_every_partial(self):
        target = [[1, 2], [3, 4]]
        exact = score(answer("12\n34"), target)
        for wrong in ("12\n33", "1\n3", "1234", "11\n11"):
            assert exact["reward"] > score(answer(wrong), target)[
                "reward"
            ]


class TestEditSimilarity:
    def test_identical(self):
        grid = [[1, 2], [3, 4]]
        assert edit_similarity(grid, grid) == 1.0

    def test_fully_disjoint_same_shape(self):
        assert edit_similarity([[1, 1]], [[2, 2]]) == 0.0

    def test_a_dropped_row_costs_one_edit_not_a_whole_realignment(self):
        # The case the centered overlay handles badly: content is right but
        # shifted. Edit distance charges the missing row, nothing more.
        target = [[1, 2], [3, 4], [5, 6]]
        pred = [[1, 2], [3, 4]]
        # Deleting "-1 5 6" from an 8-token target is 3 of 8 edits.
        assert edit_similarity(pred, target) == pytest.approx(1 - 3 / 8)

    def test_disagrees_with_the_overlay_on_a_shifted_grid(self):
        # A grid shifted one row down: the overlay sees almost nothing right,
        # edit distance sees one inserted row. Having both is the point.
        target = [[1, 1], [2, 2], [3, 3]]
        pred = [[9, 9], [1, 1], [2, 2]]
        assert edit_similarity(pred, target) > overlay_cell_accuracy(pred, target)


class TestGainOverBaseline:
    def test_matching_the_baseline_is_worth_nothing(self):
        assert gain_over_baseline(0.6, 0.6) == 0.0

    def test_perfect_is_one_whatever_the_baseline(self):
        assert gain_over_baseline(1.0, 0.6) == pytest.approx(1.0)
        assert gain_over_baseline(1.0, 0.05) == pytest.approx(1.0)

    def test_worse_than_the_baseline_is_negative(self):
        assert gain_over_baseline(0.3, 0.6) == pytest.approx(-0.5)

    def test_both_directions_stay_in_range(self):
        for baseline in (0.0, 0.05, 0.5, 0.95, 1.0):
            for value in (0.0, 0.25, 0.5, 0.75, 1.0):
                assert -1.0 <= gain_over_baseline(value, baseline) <= 1.0

    def test_a_high_baseline_does_not_shrink_the_reward_for_solving(self):
        # An easy task (baseline 0.95) and a hard one (0.05) must both pay 1.0
        # for a solve, or GRPO would quietly prefer whichever tasks are hard.
        assert gain_over_baseline(1.0, 0.95) == gain_over_baseline(1.0, 0.05)


class TestRewardHacks:
    """The shaped terms exist to break degenerate groups; they must not become
    a cheaper path to reward than actually solving the task."""

    target = [[1, 1, 1], [1, 0, 1], [1, 1, 1]]
    genuine_partial = "111\n101\n110"  # one cell wrong

    def test_copying_the_input_scores_below_a_genuine_partial(self):
        copied = "000\n000\n000"
        assert (
            score(answer(copied), self.target)["reward"]
            < score(answer(self.genuine_partial), self.target)[
                "reward"
            ]
        )

    def test_echoing_the_test_input_earns_zero_on_both_similarity_terms(self):
        # The finding that motivated the relative scoring: on a task whose
        # output mostly resembles its input, an absolute similarity score pays
        # richly for an echo. Measured against that same echo, it pays nothing.
        test_input = [[1, 1, 1], [1, 1, 1], [1, 1, 1]]
        target = [[1, 1, 1], [1, 0, 1], [1, 1, 1]]
        echo = score(answer("111\n111\n111"), target, test_input)
        assert echo["cell_match"] > 0.8  # absolutely, an echo looks great
        assert echo["cell_gain"] == pytest.approx(0.0)
        assert echo["edit_gain"] == pytest.approx(0.0)
        assert echo["copied_input"] == 1.0

    def test_beating_the_echo_outscores_the_echo(self):
        test_input = [[1, 1, 1], [1, 1, 1], [1, 1, 1]]
        target = [[1, 1, 1], [1, 0, 1], [1, 1, 1]]
        echo = score(answer("111\n111\n111"), target, test_input)
        solved = score(answer("111\n101\n111"), target, test_input)
        assert solved["reward"] > echo["reward"]

    def test_an_echo_of_a_high_baseline_task_cannot_outscore_a_real_solve(self):
        # An echo on an input that is 8/9 of the way to its target must still
        # lose to solving a task whose input shares nothing with its target.
        near_input = [[1, 1, 1], [1, 1, 1], [1, 1, 1]]
        near_target = [[1, 1, 1], [1, 0, 1], [1, 1, 1]]
        echo = score(answer("111\n111\n111"), near_target, near_input)
        far_solve = score(answer("222"), [[2, 2, 2]], [[9, 9, 9]])
        assert far_solve["reward"] > echo["reward"]

    def test_flooding_all_colors_scores_below_a_genuine_partial(self):
        flooded = "012\n345\n678"
        assert (
            score(answer(flooded), self.target)["reward"]
            < score(answer(self.genuine_partial), self.target)[
                "reward"
            ]
        )

    def test_padding_to_a_giant_grid_scores_below_a_genuine_partial(self):
        giant = "\n".join("1" * 30 for _ in range(30))
        assert (
            score(answer(giant), self.target)["reward"]
            < score(answer(self.genuine_partial), self.target)[
                "reward"
            ]
        )


class TestSerialization:
    def test_cells_are_space_delimited(self):
        assert serialize_grid([[0, 1, 2], [3, 4, 5]]) == "0 1 2\n3 4 5"

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
        pairs = [
            {"input": [[1, 1]], "output": [[2, 2]]},
            {"input": [[3, 3]], "output": [[4, 4]]},
        ]
        prompt = format_task_prompt(pairs, [[5, 6]])
        assert prompt.count("<example>") == 2
        assert "<test_input>\n5 6\n</test_input>" in prompt
        assert "<input>\n1 1\n</input>" in prompt
        assert "<output>\n4 4\n</output>" in prompt
