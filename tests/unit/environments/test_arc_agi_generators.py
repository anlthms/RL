import random
import statistics as st
from collections import Counter
from itertools import product

import pytest

from nemo_rl.environments.arc_agi_generators import (
    BACKGROUND,
    COLORS,
    _joint_level_size_schedule,
    _PALETTE_RANGE,
    _SPARE_COLOR_LEVELS,
    DIHEDRAL,
    LEVELS,
    MAX_GRID_DIM,
    Rule,
    add_border,
    anti_transpose,
    apply_color_map,
    augment_task,
    complete_symmetry,
    crop_to_bbox,
    denoise,
    drop_color,
    fill_enclosed,
    flip_h,
    flip_v,
    generate_task,
    generate_tasks,
    identity,
    is_degenerate,
    keep_only,
    recolor,
    rot90,
    rot180,
    rot270,
    rule_is_identifiable,
    scale,
    tile,
    transpose,
)

# Deliberately not symmetric under any element of the dihedral group, so every
# rotation and reflection of it is a distinct grid.
ASYMMETRIC = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]


class TestTransformations:
    def test_rot90_turns_clockwise(self):
        assert rot90([[1, 2], [3, 4]]) == [[3, 1], [4, 2]]

    @pytest.mark.parametrize(
        "transform", [rot180, flip_h, flip_v, transpose, anti_transpose]
    )
    def test_involutions(self, transform):
        assert transform(transform(ASYMMETRIC)) == ASYMMETRIC

    def test_rot90_has_order_four(self):
        grid = ASYMMETRIC
        assert rot90(rot90(grid)) == rot180(grid)
        assert rot90(rot180(grid)) == rot270(grid)
        assert rot90(rot270(grid)) == grid

    def test_dihedral_is_a_group_of_order_eight(self):
        # Every composition of two elements is again an element, and the eight
        # act distinctly -- which is what makes level 1 eight rules and not
        # fewer.
        images = {
            tuple(tuple(row) for row in transform(ASYMMETRIC))
            for transform in DIHEDRAL.values()
        }
        assert len(images) == 8
        for left, right in product(DIHEDRAL.values(), repeat=2):
            composed = tuple(tuple(row) for row in right(left(ASYMMETRIC)))
            assert composed in images

    def test_non_square_rotation_swaps_dimensions(self):
        assert rot90([[1, 2, 3], [4, 5, 6]]) == [[4, 1], [5, 2], [6, 3]]

    def test_drop_recolor_keep(self):
        grid = [[1, 2], [2, 3]]
        assert drop_color(2)(grid) == [[1, 0], [0, 3]]
        assert recolor(2, 7)(grid) == [[1, 7], [7, 3]]
        assert keep_only(2)(grid) == [[0, 2], [2, 0]]

    def test_tile_and_scale(self):
        assert tile(2, 3)([[1, 2]]) == [[1, 2, 1, 2, 1, 2], [1, 2, 1, 2, 1, 2]]
        assert scale(2)([[1, 2]]) == [[1, 1, 2, 2], [1, 1, 2, 2]]

    def test_crop_to_bbox(self):
        grid = [[0, 0, 0], [0, 5, 0], [0, 0, 0]]
        assert crop_to_bbox(grid) == [[5]]

    def test_crop_of_an_empty_grid_is_a_no_op(self):
        # Which the degeneracy guard then rejects -- there is nothing to crop to.
        grid = [[0, 0], [0, 0]]
        assert crop_to_bbox(grid) == grid

    def test_add_border(self):
        assert add_border(4)([[1]]) == [[4, 4, 4], [4, 1, 4], [4, 4, 4]]

    def test_denoise_keeps_shapes_and_drops_specks(self):
        grid = [
            [1, 1, 0, 0, 0],
            [1, 1, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 2, 0],
        ]
        assert denoise(grid) == [
            [1, 1, 0, 0, 0],
            [1, 1, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
        ]

    def test_complete_symmetry_fills_only_background(self):
        grid = [[1, 0], [2, 0]]
        assert complete_symmetry("horizontal")(grid) == [[1, 1], [2, 2]]
        # An already-painted cell is never overwritten, so the rule is a pure
        # completion rather than a mirror.
        assert complete_symmetry("horizontal")([[1, 3], [2, 4]]) == [[1, 3], [2, 4]]

    def test_fill_enclosed_fills_the_interior_only(self):
        grid = [
            [0, 0, 0, 0, 0],
            [0, 3, 3, 3, 0],
            [0, 3, 0, 3, 0],
            [0, 3, 3, 3, 0],
            [0, 0, 0, 0, 0],
        ]
        filled = fill_enclosed(7)(grid)
        assert filled[2][2] == 7
        assert filled[0][0] == BACKGROUND

    def test_fill_enclosed_ignores_a_leaky_shape(self):
        # A gap in the wall means the interior reaches the border, so nothing is
        # enclosed and nothing is filled.
        grid = [
            [0, 0, 0, 0, 0],
            [0, 3, 3, 3, 0],
            [0, 3, 0, 0, 0],
            [0, 3, 3, 3, 0],
            [0, 0, 0, 0, 0],
        ]
        assert fill_enclosed(7)(grid) == grid


class TestGuards:
    def test_identifiability_rejects_a_color_the_examples_never_show(self):
        rule = Rule(
            name="drop_color(3)",
            level=2,
            stages=(drop_color(3),),
            sample_input=lambda rng: None,
            required_colors=frozenset({3}),
        )
        without_three = [([[1, 2]], [[1, 2]])]
        with_three = [([[1, 3]], [[1, 0]])]
        assert not rule_is_identifiable(rule, without_three)
        assert rule_is_identifiable(rule, with_three)

    def test_identifiability_needs_every_example_not_just_one(self):
        rule = Rule(
            name="keep_only(5)",
            level=2,
            stages=(keep_only(5),),
            sample_input=lambda rng: None,
            required_colors=frozenset({5}),
        )
        pairs = [([[5, 1]], [[5, 0]]), ([[1, 2]], [[0, 0]])]
        assert not rule_is_identifiable(rule, pairs)

    def test_degeneracy_rejects_rot180_of_a_symmetric_grid(self):
        rule = Rule(
            name="rot180",
            level=1,
            stages=(DIHEDRAL["rot180"],),
            sample_input=lambda rng: None,
        )
        symmetric = [[1, 2, 1], [3, 0, 3], [1, 2, 1]]
        assert is_degenerate(rule, [(symmetric, DIHEDRAL["rot180"](symmetric))])
        assert not is_degenerate(rule, [(ASYMMETRIC, DIHEDRAL["rot180"](ASYMMETRIC))])

    def test_degeneracy_rejects_a_crop_with_no_background_border(self):
        rule = Rule(
            name="crop_to_bbox",
            level=3,
            stages=(crop_to_bbox,),
            sample_input=lambda rng: None,
        )
        flush = [[1, 2], [3, 4]]
        assert is_degenerate(rule, [(flush, crop_to_bbox(flush))])

    def test_degeneracy_rejects_one_bad_pair_among_good_ones(self):
        rule = Rule(
            name="flip_h", level=1, stages=(flip_h,), sample_input=lambda rng: None
        )
        pairs = [
            ([[1, 2]], flip_h([[1, 2]])),
            ([[1, 1]], flip_h([[1, 1]])),  # palindrome: unchanged
        ]
        assert is_degenerate(rule, pairs)

    def test_degeneracy_sees_a_no_op_stage_inside_a_composition(self):
        # The whole rule changes the grid, so a whole-rule check would pass it,
        # but the flip does nothing on a grid of identical rows and is therefore
        # not inferable from this example at all.
        rule = Rule(
            name="recolor(4->6)+flip_v",
            level=5,
            stages=(recolor(4, 6), flip_v),
            sample_input=lambda rng: None,
        )
        striped = [[4, 0], [4, 0]]
        assert rule.apply(striped) == [[6, 0], [6, 0]]
        assert is_degenerate(rule, [(striped, rule.apply(striped))])

    def test_level_zero_is_exempt_from_the_degeneracy_guard(self):
        rule = Rule(
            name="identity", level=0, stages=(identity,), sample_input=lambda rng: None
        )
        assert not is_degenerate(rule, [(ASYMMETRIC, ASYMMETRIC)])


class TestAugmentation:
    def test_color_permutation_commutes_with_the_rule(self):
        # This is what makes the augmentation safe: relabeling the colors and
        # then applying the rule gives the same task as applying the rule and
        # then relabeling, so the augmented pairs still follow one rule.
        mapping = {color: (color % 9) + 1 for color in range(1, 10)}
        mapping[BACKGROUND] = BACKGROUND
        rule = drop_color(5)
        permuted_rule = drop_color(mapping[5])
        assert apply_color_map(rule(ASYMMETRIC), mapping) == permuted_rule(
            apply_color_map(ASYMMETRIC, mapping)
        )

    def test_augmentation_preserves_pair_count_and_never_creates_an_identity(self):
        rng = random.Random(7)
        pairs = [(ASYMMETRIC, rot90(ASYMMETRIC)), ([[1, 0]], [[0, 1]])]
        augmented = augment_task(rng, pairs)
        assert len(augmented) == len(pairs)
        assert all(inp != out for inp, out in augmented)

    def test_augmentation_leaves_the_background_alone(self):
        rng = random.Random(3)
        grid = [[0, 1], [2, 0]]
        [(inp, _)] = augment_task(rng, [(grid, grid)])
        assert sum(cell == BACKGROUND for row in inp for cell in row) == 2


class TestGenerateTask:
    @pytest.mark.parametrize("level", LEVELS)
    def test_every_level_produces_well_formed_tasks(self, level):
        for index in range(40):
            task = generate_task(seed=11, index=index, level=level)
            assert task.level == level
            assert 2 <= len(task.train_pairs) <= 4
            grids = [task.test_input, task.target]
            for pair in task.train_pairs:
                grids.extend([pair["input"], pair["output"]])
            for grid in grids:
                assert 1 <= len(grid) <= MAX_GRID_DIM
                assert 1 <= len(grid[0]) <= MAX_GRID_DIM
                assert all(len(row) == len(grid[0]) for row in grid)
                assert all(0 <= cell <= 9 for row in grid for cell in row)

    @pytest.mark.parametrize("level", [level for level in LEVELS if level > 0])
    def test_echoing_the_input_never_solves_a_task_above_level_zero(self, level):
        # The whole point of the ladder is exact-match signal the model cannot
        # get by copying -- four previous runs converged on exactly that copy.
        for index in range(60):
            task = generate_task(seed=5, index=index, level=level)
            assert task.target != task.test_input
            for pair in task.train_pairs:
                assert pair["output"] != pair["input"]

    def test_level_zero_is_the_identity(self):
        for index in range(20):
            task = generate_task(seed=5, index=index, level=0)
            assert task.target == task.test_input
            for pair in task.train_pairs:
                assert pair["output"] == pair["input"]

    def test_generation_is_deterministic_in_seed_and_index(self):
        first = generate_task(seed=3, index=9, level=3)
        assert first == generate_task(seed=3, index=9, level=3)
        assert first != generate_task(seed=4, index=9, level=3)
        assert first != generate_task(seed=3, index=10, level=3)

    def test_num_train_pairs_is_honored(self):
        task = generate_task(seed=1, index=0, level=2, num_train_pairs=3)
        assert len(task.train_pairs) == 3

    def test_unknown_level_is_rejected(self):
        with pytest.raises(ValueError, match="unknown level"):
            generate_task(seed=1, index=0, level=99)

    def test_task_id_carries_the_level_and_the_rule(self):
        task = generate_task(seed=1, index=0, level=1)
        assert task.task_id.startswith("synth_L1_")
        assert task.rule in task.task_id


class TestMaxInputDim:
    """Grid size is a difficulty axis of its own, independent of the level.

    Exact match needs every cell right, so at 99% per-cell accuracy a 300-cell
    grid matches outright only ~5% of the time -- M4.2 measured level 0 at 1.000
    under 150 cells and 0.750 above 300. Capping the input makes exact match
    attainable while a rule is still being learned.
    """

    @pytest.mark.parametrize("level", LEVELS)
    @pytest.mark.parametrize("cap", [6, 10])
    def test_inputs_respect_the_cap(self, level, cap):
        for index in range(25):
            task = generate_task(seed=3, index=index, level=level, max_input_dim=cap)
            for pair in task.train_pairs:
                assert max(len(pair["input"]), len(pair["input"][0])) <= cap
            assert max(len(task.test_input), len(task.test_input[0])) <= cap

    @pytest.mark.parametrize("level", LEVELS)
    def test_outputs_still_fit_the_grid_limit(self, level):
        # The geometric rules multiply their input, so the caller's cap has to
        # compose with the bound they already impose rather than replace it.
        for index in range(25):
            task = generate_task(seed=4, index=index, level=level, max_input_dim=10)
            for grid in [task.target] + [p["output"] for p in task.train_pairs]:
                assert len(grid) <= MAX_GRID_DIM
                assert len(grid[0]) <= MAX_GRID_DIM

    @pytest.mark.parametrize("level", [level for level in LEVELS if level > 0])
    def test_guards_still_hold_under_a_small_cap(self, level):
        # A tight cap shrinks the space of distinct grids, which is exactly when
        # a rot180 lands on a symmetric grid -- the guards must still bite.
        for index in range(30):
            task = generate_task(seed=5, index=index, level=level, max_input_dim=6)
            assert task.target != task.test_input
            for pair in task.train_pairs:
                assert pair["output"] != pair["input"]

    def test_a_smaller_cap_yields_smaller_targets(self):
        area = lambda t: len(t.target) * len(t.target[0])
        small = [area(t) for t in generate_tasks(9, 40, [1], max_input_dim=6)]
        large = [area(t) for t in generate_tasks(9, 40, [1], max_input_dim=20)]
        assert max(small) < st.median(large)

    def test_the_cap_is_threaded_through_generate_tasks(self):
        for task in generate_tasks(11, 30, list(LEVELS), max_input_dim=7):
            assert max(len(task.test_input), len(task.test_input[0])) <= 7


class TestSizeMixing:
    """Grid size is mixed within every batch, for the same reason levels are.

    A ramp -- small grids early, large late -- puts every task in a batch at one
    difficulty, and a group of rollouts that is uniformly hopeless or uniformly
    trivial has zero advantage and contributes no gradient. Mixing keeps the
    small grids where exact match is reachable in *every* batch.
    """

    def test_a_list_cycles_sizes_across_tasks(self):
        tasks = generate_tasks(3, 60, [1, 2], max_input_dim=[6, 20])
        caps = {6: [], 20: []}
        for t in tasks:
            side = max(len(t.test_input), len(t.test_input[0]))
            caps[6 if "_d6_" in t.task_id else 20].append(side)
        assert caps[6] and caps[20]
        assert max(caps[6]) <= 6
        assert max(caps[20]) <= 20
        assert max(caps[20]) > 6

    def test_every_level_and_size_combination_appears(self):
        # Size cycles on a different modulus than level, so one size is not
        # welded to one level for the whole run.
        tasks = generate_tasks(4, 60, [1, 2, 3], max_input_dim=[6, 12])
        seen = {(t.level, "_d6_" in t.task_id) for t in tasks}
        assert seen == {(lvl, small) for lvl in (1, 2, 3) for small in (True, False)}

    def test_a_scalar_still_works(self):
        tasks = generate_tasks(5, 20, [1], max_input_dim=8)
        assert all(max(len(t.test_input), len(t.test_input[0])) <= 8 for t in tasks)

    def test_an_empty_size_list_is_rejected(self):
        with pytest.raises(ValueError, match="max_input_dim"):
            generate_tasks(6, 4, [1], max_input_dim=[])

    def test_guards_hold_for_every_size_in_the_mix(self):
        for task in generate_tasks(7, 60, [1, 2, 3], max_input_dim=[6, 12, 20]):
            assert task.target != task.test_input
            for pair in task.train_pairs:
                assert pair["output"] != pair["input"]


class TestJointDifficultyRamp:
    LEVEL_ORDER = [0, 3, 2, 1, 4, 5]

    def test_every_window_keeps_every_level_size_combination(self):
        schedule, _ = _joint_level_size_schedule(
            count=32 * 10,
            levels=[1, 2, 3, 4, 5],
            dims=[6, 12, 20],
            level_difficulty_order=self.LEVEL_ORDER,
            window=32,
            ramp_windows=10,
        )
        expected = set(product([1, 2, 3, 4, 5], [6, 12, 20]))
        for window_index in range(10):
            window = schedule[window_index * 32 : (window_index + 1) * 32]
            assert set(window) == expected

    def test_joint_mean_difficulty_moves_from_easy_to_hard(self):
        _, difficulty = _joint_level_size_schedule(
            count=32 * 20,
            levels=[1, 2, 3, 4, 5],
            dims=[6, 12, 20],
            level_difficulty_order=self.LEVEL_ORDER,
            window=32,
            ramp_windows=20,
        )
        means = [
            st.mean(difficulty[index * 32 : (index + 1) * 32]) for index in range(20)
        ]
        assert means[0] < means[-1]
        assert all(left <= right for left, right in zip(means, means[1:]))

    def test_measured_level_rank_breaks_the_authored_order(self):
        schedule, _ = _joint_level_size_schedule(
            count=15,
            levels=[1, 2, 3, 4, 5],
            dims=[6, 12, 20],
            level_difficulty_order=self.LEVEL_ORDER,
            window=None,
            ramp_windows=None,
        )
        assert schedule[0] == (3, 6)
        assert schedule[-1] == (5, 20)

    def test_window_must_fit_the_full_cross_product(self):
        with pytest.raises(ValueError, match="15 level/size"):
            _joint_level_size_schedule(
                count=32,
                levels=[1, 2, 3, 4, 5],
                dims=[6, 12, 20],
                level_difficulty_order=self.LEVEL_ORDER,
                window=14,
                ramp_windows=10,
            )

    # The four invariants below were written for the size-only `_ramped_sizes`,
    # which the joint scheduler replaced. They are the reason the ramp is a
    # reweighting rather than a ramp, so they move rather than retire.

    def test_every_window_keeps_every_size(self):
        schedule, _ = _joint_level_size_schedule(
            count=16 * 30,
            levels=[1, 2],
            dims=[6, 12, 20],
            level_difficulty_order=self.LEVEL_ORDER,
            window=16,
            ramp_windows=30,
        )
        for index in range(30):
            window = schedule[index * 16 : (index + 1) * 16]
            assert len(window) == 16
            assert {dim for _, dim in window} == {6, 12, 20}, (
                f"window {index} lost a size"
            )

    def test_it_starts_small_heavy_and_ends_large_heavy(self):
        schedule, _ = _joint_level_size_schedule(
            count=16 * 30,
            levels=[1, 2],
            dims=[6, 12, 20],
            level_difficulty_order=self.LEVEL_ORDER,
            window=16,
            ramp_windows=30,
        )
        first = [dim for _, dim in schedule[:16]]
        last = [dim for _, dim in schedule[-16:]]
        assert first.count(6) > first.count(20)
        assert last.count(20) > last.count(6)

    def test_the_ramp_spans_the_run_not_the_dataset(self):
        # The dataset is many times larger than any run so no task repeats. A
        # ramp spread over the dataset leaves the run barely moved -- inert, but
        # still looking configured.
        window, steps = 16, 60
        _, difficulty = _joint_level_size_schedule(
            count=window * 500,
            levels=[1, 2],
            dims=[6, 12, 20],
            level_difficulty_order=self.LEVEL_ORDER,
            window=window,
            ramp_windows=steps,
        )
        run = [
            st.mean(difficulty[index * window : (index + 1) * window])
            for index in range(steps)
        ]
        assert run[-1] - run[0] > 0.2, "ramp barely moved over the run"
        # Past the ramp the mixture stays at its hardest rather than resetting.
        tail = difficulty[400 * window : 401 * window]
        assert st.mean(tail) == pytest.approx(run[-1])

    def test_a_partial_final_window_is_still_filled(self):
        schedule, _ = _joint_level_size_schedule(
            count=20,
            levels=[1, 2],
            dims=[6, 12, 20],
            level_difficulty_order=self.LEVEL_ORDER,
            window=16,
            ramp_windows=2,
        )
        assert len(schedule) == 20

    def test_a_repeated_level_takes_a_larger_share_of_the_ramp(self):
        def share(levels):
            schedule, _ = _joint_level_size_schedule(
                count=32 * 6,
                levels=levels,
                dims=[6, 12, 20],
                level_difficulty_order=self.LEVEL_ORDER,
                window=32,
                ramp_windows=6,
            )
            counts = Counter(level for level, _ in schedule)
            return counts[1] / sum(counts.values())

        assert share([1, 1, 2, 3]) > share([1, 2, 3])

    def test_a_repeated_level_still_keeps_every_combination_present(self):
        # Weighting must not cost the degenerate-group property: a window that
        # dropped a combination is a window that can be uniformly hopeless.
        schedule, _ = _joint_level_size_schedule(
            count=32 * 6,
            levels=[1, 1, 2, 3],
            dims=[6, 12, 20],
            level_difficulty_order=self.LEVEL_ORDER,
            window=32,
            ramp_windows=6,
        )
        expected = set(product([1, 2, 3], [6, 12, 20]))
        for index in range(6):
            assert set(schedule[index * 32 : (index + 1) * 32]) == expected


class TestLevelPalettes:
    """Every level must be able to draw the colors its rules need.

    `palette_size` used to be a config key whose documented range crashed four
    levels: recolor needs a destination outside the palette, add_border a frame
    color, fill_enclosed a fill, and at nine colors there is none. The axis is
    gone, but the constraint it violated is permanent, so it is pinned here.
    """

    @pytest.mark.parametrize("level", sorted(LEVELS))
    def test_a_levels_palette_leaves_room_for_the_colors_its_rules_add(self, level):
        _, high = _PALETTE_RANGE[level]
        if level in _SPARE_COLOR_LEVELS:
            assert high < len(COLORS)
        assert high <= len(COLORS)

    @pytest.mark.parametrize("level", sorted(LEVELS))
    def test_generation_never_leaks_a_raw_indexerror(self, level):
        # The module's contract is that a sampler returns None and the caller
        # redraws. A helper that raises instead escapes `generate_task`
        # entirely, which is how a bare IndexError reached the dataset build.
        for index in range(40):
            generate_task(seed=4242, index=index, level=level, max_input_dim=10)

    def test_a_palette_with_no_spare_color_is_rejected_with_a_reason(self):
        from nemo_rl.environments.arc_agi_generators import _spare_color

        with pytest.raises(ValueError, match="none for a rule"):
            _spare_color(random.Random(0), list(COLORS))


class TestGenerateTasks:
    def test_levels_are_cycled_so_a_batch_holds_every_level(self):
        levels = [0, 1, 2]
        tasks = generate_tasks(seed=2, count=12, levels=levels)
        assert [task.level for task in tasks] == levels * 4

    def test_a_repeated_level_is_weighted(self):
        tasks = generate_tasks(seed=2, count=6, levels=[0, 1, 1])
        assert sum(task.level == 1 for task in tasks) == 4

    def test_empty_levels_is_rejected(self):
        with pytest.raises(ValueError, match="levels must not be empty"):
            generate_tasks(seed=2, count=4, levels=[])

    def test_tasks_are_distinct(self):
        # Volume is the reason to generate rather than materialize; if indices
        # collapsed onto a handful of tasks the ladder would be memorizable.
        tasks = generate_tasks(seed=2, count=60, levels=list(LEVELS))
        signatures = {
            (
                tuple(tuple(row) for row in task.test_input),
                tuple(tuple(row) for row in task.target),
            )
            for task in tasks
        }
        assert len(signatures) == len(tasks)
