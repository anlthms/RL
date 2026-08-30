import random

import pytest

from tools.arc_augment import (
    D8_TRANSFORMS,
    ArcView,
    apply_d8,
    invert_d8,
    sample_views,
    vote_canonical,
)

# Non-square on purpose: shape errors under rotation/transposition are the
# whole point of the round-trip requirement.
NON_SQUARE = [[1, 2, 3], [4, 5, 6]]


@pytest.mark.parametrize("transform", D8_TRANSFORMS)
def test_every_d8_element_round_trips_a_non_square_grid(transform) -> None:
    assert invert_d8(apply_d8(NON_SQUARE, transform), transform) == NON_SQUARE


def test_d8_orbit_has_eight_distinct_elements_for_an_asymmetric_grid() -> None:
    orbit = {repr(apply_d8(NON_SQUARE, transform)) for transform in D8_TRANSFORMS}
    assert len(orbit) == 8


def test_rot90_is_clockwise() -> None:
    assert apply_d8([[1, 2], [3, 4]], "rot90") == [[3, 1], [4, 2]]


def test_unknown_transform_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown D8 transform"):
        apply_d8(NON_SQUARE, "rot45")
    with pytest.raises(ValueError, match="unknown D8 transform"):
        ArcView(transform="rot45")


@pytest.mark.parametrize("transform", D8_TRANSFORMS)
def test_view_round_trips_with_a_random_color_bijection(transform) -> None:
    rng = random.Random(17)
    colors = list(range(10))
    rng.shuffle(colors)
    view = ArcView(transform=transform, color_map=tuple(colors))
    assert view.invert_grid(view.apply_grid(NON_SQUARE)) == NON_SQUARE


def test_non_bijective_color_map_is_rejected() -> None:
    with pytest.raises(ValueError, match="permutation"):
        ArcView(color_map=(0,) * 10)


def test_malformed_prediction_passes_through_inversion_as_none() -> None:
    view = ArcView(transform="rot90")
    assert view.invert_grid(None) is None


def test_apply_pairs_transforms_inputs_and_outputs_consistently() -> None:
    view = ArcView(transform="rot180", color_map=(1, 0, 2, 3, 4, 5, 6, 7, 8, 9))
    pairs = [{"input": [[0, 1]], "output": [[1], [0]]}]
    transformed = view.apply_pairs(pairs)
    assert transformed == [{"input": [[0, 1]], "output": [[1], [0]]}]
    # Same view inverts each grid individually.
    assert view.invert_grid(transformed[0]["input"]) == pairs[0]["input"]


def test_multi_test_task_views_share_one_frame() -> None:
    view = ArcView(transform="rot270")
    tests = [NON_SQUARE, [[7]], [[8, 9]]]
    for test_input in tests:
        assert view.invert_grid(view.apply_grid(test_input)) == test_input


def test_sample_views_is_deterministic_and_leads_with_identity() -> None:
    first = sample_views(count=5, seed=3)
    second = sample_views(count=5, seed=3)
    assert [view.view_id for view in first] == [view.view_id for view in second]
    assert first[0].is_identity()
    assert len({view.view_id for view in first}) == 5


def test_sample_views_can_fix_the_background_color() -> None:
    views = sample_views(count=6, seed=11, fix_background=True)
    assert all(view.color_map[0] == 0 for view in views)


def test_vote_requires_canonical_frame_agreement() -> None:
    # Three views agree once mapped back to canonical; one dissents; one failed
    # to parse. The vote must count canonical-frame equality, not raw grids.
    canonical = [[1, 2, 3], [4, 5, 6]]
    views = sample_views(count=3, seed=5, include_identity=False)
    predictions = [view.invert_grid(view.apply_grid(canonical)) for view in views]
    predictions.append([[9]])
    predictions.append(None)
    voted, stats = vote_canonical(predictions)
    assert voted == canonical
    assert stats["votes"] == 3.0
    assert stats["tied"] == 0.0
    assert stats["agreement"] == pytest.approx(0.75)


def test_vote_with_no_parseable_prediction_returns_none() -> None:
    voted, stats = vote_canonical([None, None])
    assert voted is None
    assert stats["votes"] == 0.0


def test_vote_tie_breaks_toward_the_earliest_candidate_and_flags_it() -> None:
    voted, stats = vote_canonical([[[1]], [[2]]])
    assert voted == [[1]]
    assert stats["tied"] == 1.0
