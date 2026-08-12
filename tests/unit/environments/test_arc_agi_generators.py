import random
from itertools import product

import pytest

from nemo_rl.environments.arc_agi_generators import (
    BACKGROUND,
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
