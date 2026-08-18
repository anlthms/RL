# ARC code cleanup — findings and plan

Review of the two commits on `async_arc2`:

- `6c779ea` — feat(arc): add joint synthetic curriculum
- `6a4fe19` — feat(arc): add early-answer follow-up recipe

Nothing has been changed or committed. This document is the handoff: every finding
below was reproduced by running the generator, not inferred from the diff, and each
carries the fix. Read `ARC_AGI_2_ENV_PLAN.md` first for design context.

**Status: none of this is applied.** The working tree is as the two commits left it.

---

## 0. Decisions still open

The one decision that changes the shape of the work:

**How far to go on the five extra difficulty axes** (palette size, object count,
density, composition depth, distractors). They are the source of six of the seven
bugs below — each is wired into some levels and silently inert in others.

| Option | What it means | Cost |
|---|---|---|
| **A — keep, make each honest** *(recommended)* | Fix all seven bugs; where an axis genuinely cannot apply to a level, say so explicitly and validate it rather than silently zeroing it. Preserves `6c779ea`'s intent. | Highest |
| B — revert the five axes, keep joint level×size | Roll back to the joint level-rank × grid-size schedule, which is sound. Reintroduce axes one at a time, each with a test proving it moves difficulty. | Medium |
| C — revert to pre-`6c779ea` | Back to the size-only ramp. Discards the one part that works. | Low, but loses ground |

Recommendation is **A**, on the grounds that the axes are ~80% there and the failures
are localized; but **B** is defensible if the next campaign priority is item 7 of the
plan (the object/relational DSL), because that rewrites the rule layer anyway and
half-working axes would be rewritten with it.

Git strategy: **new commits on top**, not a rebase. The two commits reference job IDs
that the results TSV and the plan doc both cite; rewriting them costs traceability for
no benefit.

---

## 1. Verified bugs

All reproduced with `PYTHONPATH=. python3 -c ...` on a login node (the generator is
stdlib-only, so no container is needed).

### 1.1 `palette_size=9` crashes four levels — while the validator advertises it

`nemo_rl/environments/arc_agi_generators.py:987`

```python
if palette_size is not None and not 1 <= palette_size <= len(COLORS):
    raise ValueError(f"palette_size must be between 1 and {len(COLORS)}")
```

Levels 2, 3, 4 and 5 each need a colour *outside* the palette:

- `_level2_rule:684` — `destination = rng.choice([c for c in COLORS if c not in palette])`
- `_level3_rule:738` — `color = rng.choice([c for c in COLORS if c not in palette])` (border)
- `_level4_rule:784` — same (fill_enclosed)

At `palette_size=9` that list is empty and `rng.choice` raises
`IndexError: Cannot choose from an empty sequence`. Reproduced on all four levels.

**Fix.** Tighten the bound to the real one and make it level-aware. The maximum usable
palette is `len(COLORS) - 1` for any level that draws a spare colour, minus
`distractor_count`. Validate once in `generate_task`, before the attempt loop:

```python
_SPARE_COLOR_LEVELS = frozenset({2, 3, 4, 5})
max_palette = len(COLORS) - (1 if level in _SPARE_COLOR_LEVELS else 0)
if palette_size is not None and not 1 <= palette_size <= max_palette:
    raise ValueError(
        f"palette_size must be between 1 and {max_palette} at level {level}; "
        "the level's rules need a colour outside the palette"
    )
```

Note this is only *reachable* now because `palette_size` became a user-settable config
key in `6c779ea`. The latent `rng.choice([])` predates it.

### 1.2 `_multicolor_shape` raises out of the rejection loop

`nemo_rl/environments/arc_agi_generators.py:404`

```python
if len(colors) > len(cells):
    raise ValueError("shape has fewer painted cells than requested colors")
```

Every other sampler in this module returns `None` on failure and `generate_task`
redraws — that is the module's stated contract (`arc_agi_generators.py:316`,
"Samplers return None when placement fails, and the caller redraws"). This one raises,
and the exception escapes `generate_task` entirely.

Reproduced: **21 of 60** tasks at `palette_size=8, object_count=1, level=3` die with
this `ValueError`. A filled rectangle is at minimum 2×2 = 4 cells; distributing 8
palette colours over 1 shape needs 8.

**Fix.** Return `None` from `objects_pattern`'s `sample` when any colour group is
larger than its shape, so the redraw machinery handles it:

```python
def _multicolor_shape(shape: Grid, colors: list[int]) -> Grid | None:
    """Paint every requested color into one connected non-background shape.

    Returns None when the shape has fewer painted cells than colors, which is a
    redraw rather than an error -- the caller samples a new shape size.
    """
    ...
    if len(colors) > len(cells):
        return None
```

and at the call site (`objects_pattern:453`):

```python
shape = _multicolor_shape(builder(shape_h, shape_w, colors[0]), colors)
if shape is None or not _place(rng, grid, occupied, shape, margin):
    return None
```

Consider also biasing `shape_h`/`shape_w` upward when `len(colors)` is large, so the
redraw budget is not spent on shapes that cannot possibly fit — otherwise a large
palette with a small object count burns `_MAX_ATTEMPTS` and then raises `RuntimeError`
claiming the level is inconsistent, which would be a misleading message.

### 1.3 Duplicate levels now invert the weighting

`levels: [1, 1, 2]` is documented in three places as weighting level 1 twice:

- `examples/configs/async/env_arc_synth.yaml:66` — "Repeat a level here to weight it."
- `nemo_rl/data/datasets/response_datasets/arc_synth.py:67` — "a repeated entry weights that level"
- `generate_tasks` docstring (old text, now removed) — "Repeat a level in `levels` to weight it."

Reproduced with `levels=[1,1,2]`, `level_difficulty_order=[0,3,2,1,4,5]`:

```
joint schedule : Counter({2: 20, 1: 10})    <-- level 2 twice as often
legacy path    : Counter({1: 20, 2: 10})    <-- as documented
```

Cause: `_joint_level_size_schedule:1198` builds
`choices = [(level, dim) for level in levels for dim in sorted(set(dims))]`, so
duplicates enter `choices`; `scores` is a dict keyed on the same tuples and dedups
them; the hump weighting is then applied over a list with repeats and the counts come
out inverted rather than doubled.

**Fix.** Decide which semantics you want and enforce it in one place:

- *Preferred* — drop repeat-to-weight. It is a legacy affordance that no config uses,
  and the joint ramp is now the weighting mechanism. Deduplicate at the top of
  `generate_tasks` (`levels = sorted(set(levels))` after validation) and delete the
  claim from all three docstrings/comments.
- *Alternative* — keep it, by building `choices` from `sorted(set(levels))` and
  carrying a separate multiplicity vector into the weighting.

Either way, add a test that pins the resulting level histogram, because this silently
changed behaviour under a comment that still describes the old behaviour.

### 1.4 `distractor_count` is dead on every level but 2 and 5

`generate_task:1012`

```python
effective_distractors = distractor_count if level in (2, 5) else 0
```

So `pattern_noisy_objects:530` —

```python
speck_count = options.distractor_count or rng.randint(2, 5)
```

— never sees a non-zero `distractor_count`, because denoise is level 4. That branch is
unreachable. Reproduced: 20 level-4 tasks generated with `distractor_count=3` all carry
`x0` in their task ID.

Meanwhile `env_arc_synth.yaml:93` states: *"denoise uses them as irrelevant specks"*,
and `ARC_AGI_2_ENV_PLAN.md` item 6 is marked **DONE** with "distractors are
configurable, recorded in task IDs, and covered by fixed validation."

**Fix.** Pick one:

1. Wire it — let level 4 pass `distractor_count` through as the speck count. Requires
   thinking about whether specks-as-distractors and specks-as-the-thing-to-remove are
   the same axis (they are not: for denoise the specks *are* the signal).
2. Or make the restriction explicit and honest: rename the field to
   `distractor_colors_count`, document that it applies to colour-op levels only, and
   delete the dead `or` branch in `pattern_noisy_objects`. Then correct the YAML comment
   and downgrade plan item 6 from DONE to partially done.

Option 2 is the smaller, truer change.

### 1.5 `palette_size=1` is silently overridden on levels 2 and 5

`generate_task:1004`

```python
active_palette_size = max(2, palette_size) if level in (2, 5) else palette_size
```

Reproduced: `generate_task(3, 0, 2, palette_size=1)` yields a task ID containing `p2`.
The user asked for one colour and got two, with no warning. This is the
silent-on-misconfiguration smell the error-handling skill calls out explicitly.

**Fix.** Fold the level-2/5 minimum into the validation from §1.1 and raise:

```python
min_palette = 2 if level in (2, 5) else 1
if palette_size is not None and not min_palette <= palette_size <= max_palette:
    raise ValueError(...)
```

A colour op needs two colours to be identifiable; that is a real constraint and should
fail loudly rather than be papered over.

### 1.6 `density` means two opposite things

`pattern_scatter:345` — `count = max(len(palette), round(density * height * width))`.
Density is the painted fraction; higher is a busier grid.

`pattern_motif:491` — `colors[...] if rng.random() < options.density else BACKGROUND`.
Here it is the probability of *keeping* a motif cell, and it was a hardcoded `0.7`
before `6c779ea`. The config now supplies `[0.15, 0.25, 0.40]`, so motif grids went
from 70% painted to 15–40% painted.

Two consequences:

- The difficulty direction is inverted between the two samplers.
- `pattern_motif:500` rejects a motif whose masking erased a palette colour. At
  density 0.15 with a 3-colour palette that rejection fires most of the time, so the
  motif sampler burns redraw budget and is effectively deselected from the generic
  sampler pool at the easy end of the ramp — the opposite of intended.

**Fix.** Give the motif sampler its own constant back (`_MOTIF_KEEP_PROB = 0.7`) and
leave `density` meaning "painted fraction" only, in `pattern_scatter` where it is
well-defined. If motif fill should be an axis, it needs its own name and its own
easy→hard direction. Add a test asserting mean painted fraction is monotone in
`density` across every sampler that consumes it.

### 1.7 `object_count` is half-wired in `pattern_lines`

`pattern_lines:467`

```python
count = options.object_count or len(palette)
for color in _color_cycle(rng, palette, max(count, len(palette))):
```

The `max(...)` means `object_count` can only ever *raise* the number of lines, never
lower it below the palette size. `objects_pattern:435` handles the same tension the
opposite way — it distributes all palette colours across exactly `object_count` shapes.

**Fix.** Make `pattern_lines` follow `objects_pattern`: draw exactly `object_count`
lines and distribute palette colours over them, with a `None` return if the palette
cannot be covered by that many lines. Same identifiability requirement, same shape of
solution — the two samplers should not disagree about what the axis means.

---

## 2. Structural cleanups

### 2.1 Delete the size-only scheduler

`_ramped_sizes:1260` and `_ramped_choices:1221` are the same largest-remainder hump
algorithm, written twice — one over `dims`, one over `[(level, size)]` tuples. The
size-only path is dead in every shipped config, but survives as three public
parameters (`size_ramp_window`, `size_ramp_steps`, and the `elif` branch in
`generate_tasks`) plus a mutual-exclusion guard:

```python
if size_ramp_window or size_ramp_steps:
    raise ValueError("size_ramp_* and the joint level_difficulty_order schedule are mutually exclusive")
```

Two schedulers with a runtime guard against using both is the most expensive way to
express "there is one scheduler."

**Fix.** Delete `_ramped_sizes`, `size_ramp_window`, `size_ramp_steps`, the mutual
exclusion, and the `elif` branch. `_ramped_choices` with a single-element level list
reproduces the old behaviour exactly. Rename `_ramped_choices` to `_ramped_schedule`.
Port the good docstring from `_ramped_sizes` (it explains *why* the window must be
prompts-per-step and the span must be the run length) onto the survivor — that text is
load-bearing and currently sits on the function being deleted.

Migrate the four `_ramped_sizes` tests
(`tests/unit/environments/test_arc_agi_generators.py:406-444`) onto the joint schedule
rather than deleting them; they check real invariants (every window keeps every size,
mean rises, the ramp spans the run not the dataset, partial final window is filled).

### 2.2 One curriculum config object instead of twelve parameters, four times

Adding one axis today means editing four files in lockstep:

| File | What it repeats |
|---|---|
| `arc_agi_generators.py:1066` | `generate_tasks` — 12 keyword parameters |
| `arc_synth.py:111` | `ArcSynthDataset.__init__` — the same 12, re-typed and re-documented |
| `tools/arc_synth_score_val_dumps.py:113` | 11 keys hand-copied out of `data.train` |
| `tools/arc_synth_preflight.py:70` | 6 axis names hand-listed for printing |

Nothing enforces that these four lists agree. `arc_synth_score_val_dumps.py` already
omits `difficulty_ramp_window` (correct, for validation) and `size_ramp_*` (now
meaningless) — the drift has started.

**Fix.** Introduce one schema class and pass it through:

```python
# nemo_rl/environments/arc_agi_generators.py
class CurriculumConfig(BaseModel, extra="allow"):
    """The difficulty schedule for a synthetic ARC run.

    Every axis is either a scalar (held constant) or a list ordered easy -> hard
    (selected by the joint difficulty rank).
    """
    levels: list[int]
    level_difficulty_order: list[int]
    max_input_dim: int | list[int]
    num_train_pairs: int | list[int] | None
    palette_size: int | list[int] | None
    object_count: int | list[int] | None
    density: float | list[float]
    composition_depth: int | list[int]
    distractor_count: int | list[int]
    ramp_window: int | None = None   # None on the validation split
    ramp_steps: int | None = None
```

Then `generate_tasks(seed, count, config)`, `ArcSynthDataset(config, ...)`, and both
tools construct it from `data.train` in one line. Per `config-conventions`, this is
user-facing YAML-loaded config, so `pydantic.BaseModel` with `extra="allow"` — not a
`TypedDict` (forbidden for new classes) and not `dict[str, Any]`.

The validation split becomes `config.model_copy(update={"ramp_window": None, "ramp_steps": None})`,
which is more obviously correct than the current "call `generate_tasks` again but
remember to omit two of the twelve arguments" (`arc_synth.py:170`).

### 2.3 Config defaults at call sites are a forbidden pattern

`config-conventions` §Forbidden Patterns names function-parameter defaults for config
values explicitly. Present in two signatures:

- `generate_tasks(..., density: float | list[float] = 0.2, composition_depth: int | list[int] = 2, distractor_count: int | list[int] = 0)`
- `ArcSynthDataset.__init__(..., density: float | list[float] = 0.2, ...)`

And `arc_synth.py:59` still asserts the opposite in its own docstring:

> Every field that shapes the curriculum is required rather than defaulted here: they
> all live in `env_arc_synth.yaml`, and a silent fallback to some other mixture or seed
> would be a different experiment reported under the same name.

That paragraph was true before `6c779ea` and is now contradicted by the signature
directly beneath it. §2.2's `CurriculumConfig` fixes this: the defaults move onto the
BaseModel fields (one place), and the docstring becomes true again.

### 2.4 `dataclasses.replace` in `_level3_rule`

`_level3_rule:705, 723, 739` rebuilds `GenerationOptions` field-by-field three times to
change one field. Six lines each, and every future field must be added to all three.

```python
from dataclasses import dataclass, replace
bounded = replace(options, max_input_dim=min(options.max_input_dim, MAX_GRID_DIM // factor))
```

Three 8-line blocks become three 1-line calls, and the failure mode (forgetting to
copy a new field) disappears.

### 2.5 Type annotations that do not hold

`_axis_values:1163` is annotated `-> list[AxisT | None]`; `_difficulty_axis_value:1172`
takes `values: list[AxisT]`. The `None` element is real — it is how "no explicit value,
let the level choose" is expressed — so the annotation on the consumer is wrong.

Under §2.2 this simplifies: make the "unset" case `None` at the config level and have
`_axis_values` return `list[AxisT]` for set axes only.

Also add both files to `pyrefly.toml` `project-includes` if they are not there — per
the linting skill, a file absent from the allow-list has its annotations never checked,
which is presumably how the above survived.

### 2.6 `score_response` mutates the reward it just computed

`nemo_rl/environments/arc_agi_grid.py:395`

```python
terms["reward"] = weights.exact * terms["grid_match"] + ...
if terms["copied_input"] and not terms["grid_match"]:
    terms["reward"] = -(weights.cell + weights.edit + weights.extraneous + weights.shape)
```

Two small things:

- The floor expression is duplicated from the unparseable path. Name it once —
  `def _unparseable_floor(weights: RewardWeights) -> float` — so the two can never
  drift apart. The plan's §3 invariant ("the unparseable floor must sit strictly below
  the worst parseable answer") depends on them being the same number.
- Computing a value and then overwriting it reads as a patch. Prefer an early return or
  a single expression with the echo case as a guard clause, so the reward has one
  definition per branch.

Behaviourally this change is sound and it is the correct resolution of plan item 9 —
this is a readability note, not a correctness one. One thing worth *measuring* though:
an echo now scores identically to an unparseable response, so `format_valid` can no
longer distinguish "emitted a well-formed grid" from "failed to emit one" in the echo
case. If the format term is meant to bootstrap, confirm that is intended.

---

## 3. Config, tooling and launch

### 3.1 `env_arc_synth_early_answer.yaml` breaks the model-agnostic rule

The file sets `grpo.val_period: 20`. `ARC_AGI_2_ENV_PLAN.md` §5 states the rule this
violates:

> **Keep the env layers model-agnostic.** Learning rates, chat-template flags and
> KV-cache sizes belong in the model's recipe.

Validation cadence is a wall-clock/cost knob (the plan's own §7 derives `val_period: 20`
from "validation is 344 rows and ~11 min a checkpoint" — a nano-v3-on-4-nodes fact).
The file's own comment concedes the motivation was working around a launcher override
that did not take effect.

**Fix.** Move `val_period` to `nanov3_arcsynth_early_answer_colocated.yaml`, and fix the
actual bug — find out why job 6211877 received `val_period=20` and validated every 10
steps. That is an override-precedence bug in `launch_experiment.sh` or the config
merge, and pinning the value in the wrong layer hides it rather than fixing it.

### 3.2 `CONFIG_PATH` escape hatch in `launch_experiment.sh`

`launch_experiment.sh:135` gained:

```bash
CONFIG="${CONFIG_PATH:-examples/configs/async/nanov3_${ENV}_${TOPOLOGY}.yaml}"
```

solely because `nanov3_arcsynth_early_answer_colocated.yaml` does not fit
`nanov3_${ENV}_${TOPOLOGY}`. A free-form path variable defeats the naming convention and
the `[[ -f ]]` guard's usefulness for every future recipe.

**Fix.** Rename the recipe so `ENV=arcsynth_early_answer` resolves it, and add
`arcsynth_early_answer` to the `case "${ENV}"` entrypoint dispatch at
`launch_experiment.sh:133`. Revert the `CONFIG_PATH` line and its doc comment. Zero new
launcher surface.

### 3.3 `run_arc_preflight.sub` hardcodes the launch overrides

The preflight now inlines the five `EXTRA_OVERRIDES` from the known-good 4-node launch
so that it checks the schedule that will actually run — a good instinct with the wrong
mechanism. There are now two copies of the run configuration, in a `.sub` file and in
the plan doc's §7 command, and nothing keeps them in sync.

**Fix.** Put the 4-node overrides in a config overlay
(`examples/configs/async/nanov3_arcsynth_colocated_4n.yaml`) that both the preflight and
`launch_experiment.sh` reference by name. The plan doc's §7 then documents one recipe
name instead of a five-item override string, and the preflight cannot drift from the
launch because there is only one source.

### 3.4 `arc_synth_score_val_dumps.py` reimplements dataset construction

`tools/arc_synth_score_val_dumps.py:113` hand-copies 11 keys out of `data.train` into a
`generate_tasks` call, plus `parse_known_args` + `parse_hydra_overrides` to accept
Hydra-style overrides alongside argparse flags. It regenerates the validation split by
re-deriving what `ArcSynthDataset` already does.

**Fix.** Instantiate `ArcSynthDataset` from the resolved config and read `.val_dataset`.
One line replaces the whole block, and the re-scorer is then correct by construction
whenever the dataset changes. Keep the `--levels` fallback for old dumps only if there
are old dumps worth scoring; otherwise drop the dual-mode argument parsing, which is
where the `parse_known_args` awkwardness comes from.

Also: the `errors`-list-then-`raise SystemExit` restructuring of
`tools/arc_synth_preflight.py` is a genuine improvement — the tool previously printed
`!!` warnings and exited 0, so a preflight could "pass" while reporting four problems.
Keep it. The only note is §2.2's hand-listed axis names.

---

## 4. Documentation

### 4.1 `ARC_AGI_2_ENV_PLAN.md` claims more than the code delivers

- **Item 6 is marked DONE** — "Few-shot count, palette size, object count/density,
  composition depth, and distractors are configurable, recorded in task IDs, and covered
  by fixed validation." Per §1.4, §1.6 and §1.7, distractors are inert on four of six
  levels, density is conflated with motif fill, and object count is half-wired on
  `pattern_lines`. Downgrade to **PARTIAL** with the caveats named, or finish the work
  first.
- **§4's summary** — "The same joint rank controls few-shot count `[4,3,2]`, palette
  size, object count/density, composition depth, and distractors" — same overclaim.
- The DONE markers deleted the original rationale for items 5, 6 and 9. That rationale
  is the part worth keeping when the item is revisited; the plan is explicitly a design
  doc, and "DONE — one sentence" loses why the design was chosen. Prefer keeping the
  reasoning and appending the outcome.
- **§9 open question 2** and the new **Current result** block at the top state the same
  6211877 finding twice. Keep the top block (it is the live status) and make §9 point at
  it.

### 4.2 Stale comments that now describe the opposite of the behaviour

- `env_arc_synth.yaml:66` — "Repeat a level here to weight it" (§1.3: inverted).
- `env_arc_synth.yaml:93` — "denoise uses them as irrelevant specks" (§1.4: unreachable).
- `arc_synth.py:67` — "a repeated entry weights that level" (§1.3).
- `arc_synth.py:59` — "Every field that shapes the curriculum is required rather than
  defaulted here" (§2.3: twelve now have defaults).

These are worse than absent comments — each is a load-bearing claim about behaviour that
is now false, in a file whose whole style is to record *why*.

### 4.3 Commit hygiene, for the record

`6a4fe19` ("add early-answer follow-up recipe") carries every plan-doc update belonging
to `6c779ea` ("add joint synthetic curriculum") — the §4 curriculum rewrite, the item
5/6/9 DONE markers, the `train_mb_tokens` note. Reading `6c779ea` alone shows a large
behaviour change with no doc; reading `6a4fe19` alone shows a doc rewrite for code that
is not in it. Not worth a rebase to fix (see §0), but worth not repeating.

---

## 5. Tests to add

`6c779ea` added 168 lines to `tests/unit/environments/test_arc_agi_generators.py`,
which is real coverage — but every bug in §1 slipped through, so the gaps are specific:

| Test | Catches |
|---|---|
| `generate_task` at every level × `palette_size` in `1..9` either succeeds or raises `ValueError` — never `IndexError` | §1.1 |
| `generate_task` at `palette_size=8, object_count=1` over 60 seeds raises nothing | §1.2 |
| level histogram of `generate_tasks(levels=[1,1,2], ...)` matches the documented weighting under the joint schedule | §1.3 |
| every axis, on every level that claims to consume it, changes the generated task — assert on the task-ID fields, which already record realized values | §1.4, §1.7 |
| `palette_size` below a level's minimum raises rather than being silently raised to it | §1.5 |
| mean painted-cell fraction is monotone in `density` for every sampler that reads it | §1.6 |
| the four migrated `_ramped_sizes` invariants, on the joint schedule | §2.1 |
| final partial window still contains the hard end of the schedule | §2.1 (`_ramped_choices` truncates `row[:current_size]`; verify no systematic bias) |

Existing tests `test_palette_axis_controls_the_number_of_active_colors` and
`test_level_two_distractor_colors_are_left_untouched` pass because they exercise
level 2 — the one level where the axes are fully wired. Parameterize them over all
levels and they will fail, which is the point.

Per the testing skill, these run on a login node:
`PYTHONPATH=. python3 -m pytest <copy outside tests/>` — `tests/unit/conftest.py`
imports Ray. Full suite via `run_arc_tests.sub`.

---

## 6. Suggested order of work

Bugs before refactors, so each refactor is validated by tests that already pass.

1. **§1.1, §1.5** — palette validation. Smallest, and unblocks parameterizing the
   existing palette tests over all levels.
2. **§1.2** — `_multicolor_shape` returns `None`. Independent.
3. **§1.3** — settle repeat-to-weight, fix or delete, correct all three comments.
4. **§1.6, §1.7** — density and object-count semantics; one sampler each.
5. **§1.4** — distractors: wire level 4 or scope the field honestly (decision needed).
6. **§5** — add the eight tests. Everything above should now be green.
7. **§2.1** — delete the size-only scheduler; migrate its tests.
8. **§2.2, §2.3, §2.5** — `CurriculumConfig`; collapses four parameter lists into one
   and fixes the forbidden-default violation. Largest change, done last, on green tests.
9. **§2.4, §2.6** — `dataclasses.replace`; name the reward floor. Cosmetic, cheap.
10. **§3.1–§3.4** — config layering, recipe rename, preflight overlay, re-scorer reuse.
11. **§4** — reconcile the plan doc and the stale comments with what the code now does.

Steps 1–6 are the "reverse the damage" half; 7–11 are the "make it elegant" half. They
can land as two commits or eleven; either is fine as long as no commit leaves a test red.

---

## 7. What is worth keeping, unchanged

Not everything in the two commits needs work. These were good calls and should survive
any refactor:

- **The joint level-rank × grid-size schedule** is the right structure, and
  `_joint_level_size_schedule`'s separation of ranking from weighting is clean.
- **Ranking levels by measured solve rate** rather than authored index, with the comment
  admitting which parts were measured (L3, L2) and which are a tiebreaker (L1, L4, L5).
  That honesty is exactly right.
- **The preflight now exits non-zero** instead of printing `!!` and returning 0.
- **`train_mb_tokens: 16384`** with the OOM job ID recorded next to it.
- **The echo floor** (`score_response`) is the correct fix for plan item 9; §2.6 is only
  about how it is written.
- **`set -euo pipefail`** in both `.sub` files.
- **Task IDs recording every realized axis value** — that is what made most of §1
  diagnosable in minutes rather than hours.

---

## 8. Response — analysis and recommended disposition (2026-08-16)

### 8.1 Bottom line

The central diagnosis is sound: the joint level×size schedule is worth keeping,
while the added axes currently promise stronger and more uniform semantics than the
generator implements. The fail-loud validation, one config schema, scheduler
deduplication, stale-documentation cleanup, and new-commits-on-top strategy all agree
with the repository guidelines.

I would nevertheless choose **Option B for the next cleanup**, not A. The current
campaign result already makes the object/relational DSL the next curriculum question
after the early-answer experiment, and closer inspection shows that `object_count` and
`distractor_count` need semantic design, not just localized wiring. Spending the
largest cleanup budget to stabilize interfaces that item 7 is expected to replace is
unlikely to pay back. Keep the joint level-rank×size schedule (and the few-shot axis if
desired), remove the five under-specified axes from the shipped config, and reintroduce
each through the DSL with an observable invariant.

If another campaign on the current generator is required before the DSL work, A is
still viable, but the corrections below are prerequisites; the fixes as currently
written are not yet sufficient.

### 8.2 Corrections to the findings and proposed fixes

1. **The palette failure is probabilistic, and the proposed bound is incomplete.**
   Levels 2–5 *can* select a rule needing a spare color; they do not all fail on every
   task because other rule kinds do not need one. More importantly, the prose says to
   subtract `distractor_count` but the sample code does not. On the current code,
   `palette_size=8, distractor_count=1` still fills all nine non-background colors and
   leaves recolor no destination. In a 50-index check this raised `IndexError` 34 times
   at level 2 and 44 times at level 5. For the levels that support color distractors,
   the upper bound must account for all three quantities:

   ```python
   needs_spare = level in {2, 3, 4, 5}
   effective_distractors = distractor_count if level in {2, 5} else 0
   max_palette = len(COLORS) - effective_distractors - int(needs_spare)
   ```

   Combine that with the level-2/5 minimum of two and validate before constructing the
   RNG/attempt loop. A non-zero distractor value on an unsupported level must either be
   represented as an explicitly level-scoped axis or rejected; silently replacing it
   with zero is the behavior this cleanup is trying to remove. Also change the finding
   title to “`palette_size=9` can crash four levels” so it does not imply determinism.

2. **Preserve repeat-to-weight semantics.** It is documented in multiple public-facing
   locations and already has a legacy-path unit test. “No shipped config uses it” is not
   enough reason to change the API silently. Build unique `(level, size)` choices,
   retain a level-multiplicity map, and multiply the ramp weights by that multiplicity.
   Test the first, middle, and last windows; only testing an aggregate histogram can
   miss the interaction between multiplicity and the moving hump. If the behavior is
   intentionally removed later, make that a separately documented breaking change.

3. **The `_multicolor_shape` redraw fix is correct, but feasibility should be checked
   before sampling dimensions.** Returning `None` restores the sampler contract. Also
   choose a shape whose painted-cell capacity is at least the color-group size (hollow
   and filled rectangles have different capacities); otherwise valid configurations
   spend much of `_MAX_ATTEMPTS` on impossible draws and can still end in the misleading
   “level's rules and input patterns are inconsistent” error.

4. **`object_count` needs a definition before fixing `pattern_lines`.** A monochrome
   full row/column cannot cover more palette colors than the number of lines. Distributing
   several colors over one line changes the sampler from “full rows and columns”, while
   independently drawing rows/columns can collide and yield fewer visible lines than
   requested. Decide whether the axis counts primitives, connected components, or
   semantic objects, then sample without replacement and test that exact observable.
   Making the loop execute exactly `object_count` times is not by itself an honest fix.

5. **The distractor rename is directionally right but does not complete Option A.**
   `num_distractor_colors` (or equivalent) accurately describes levels 2 and 5. Merely
   documenting that it is ignored elsewhere still leaves a joint scalar/list axis
   silently inert on three fifths of the training mixture. Option A therefore needs a
   level-scoped representation; Option B can remove it until the DSL defines a
   cross-level distractor abstraction. Denoise specks should remain separate because
   they are the transformation's signal, not irrelevant distractors.

6. **The config-default history in §2.3 is inaccurate.** Before `6c779ea`,
   `ArcSynthDataset.__init__` already defaulted `num_train_pairs`, `max_input_dim`, and
   both `size_ramp_*` arguments. The new commit expanded the contradiction with the
   docstring; it did not introduce it. The recommended schema cleanup remains valid,
   but the historical claim should be corrected.

7. **The reward concern at the end of §2.6 is not present.** A penalized echo keeps
   `format_valid == 1.0` and `copied_input == 1.0`; only its scalar reward is moved to the
   unparseable floor. `test_non_answer_echo_is_on_the_unparseable_floor` already pins
   that distinction. Extracting a shared floor helper and giving `reward` one assignment
   per branch is still a worthwhile readability cleanup.

### 8.3 Structural and tooling refinements

- **Use a schema, but include the complete regeneration boundary.**
  `CurriculumConfig` should cover every value needed to regenerate a split, including
  the relevant seed/count fields, or be nested in a clearly named dataset config that
  does. Put cross-field validation there: non-empty axes, known levels, seed separation,
  ramp-window capacity, level-aware palette bounds, and supported axis/level
  combinations. Because `extra="allow"` would otherwise accept removed `size_ramp_*`
  keys silently, keep explicit deprecated fields that raise a migration error (or reject
  those names in a model validator) during the transition.

- **Do not overstate the one-line scorer change.** `ArcSynthDataset.val_dataset`
  contains row dictionaries, while `score()` currently consumes `SynthTask` attributes.
  Reusing dataset construction is the right abstraction, but the scorer must be adapted
  to the row schema and must apply `data.default` the same way `load_response_dataset`
  does. Add a test that scorer targets and IDs exactly equal the dataset's held-out rows.

- **The launcher failure mechanism is known more precisely.** In
  `6211877-logs/driver_command.sh`, a newline appears immediately after
  `buffer_size_gb=40`. `examples/run_grpo.py` consequently received overrides only
  through that key; `policy.sequence_packing.train_mb_tokens=16384` and
  `grpo.val_period=20` became a second shell command. The final config in the driver log
  confirms `val_period=10`. This is command serialization/transport, not Hydra merge
  precedence. Fix the launcher so overrides are transported as an argument array or a
  generated, safely quoted command file, and add a regression test with a long override
  list. Pinning `val_period` in YAML happened to mask the symptom.

- **No early-answer recipe rename is needed.** The existing file is already named
  `nanov3_arcsynth_early_answer_colocated.yaml`; adding `arcsynth_early_answer` to the
  accepted `ENV` values and standard entrypoint dispatch makes the conventional path
  resolve. Remove `CONFIG_PATH` after that. The proposed four-node overlay should
  likewise use a name the launcher can derive, and preflight should assert that the
  submitted node count matches the recipe's intended topology because YAML cannot
  enforce the Slurm allocation by itself.

- **Deleting the size-only scheduler is reasonable on this experimental branch.** Port
  its invariant tests first, preserve duplicate-level weighting as above, and ensure
  removed keys fail explicitly rather than disappearing into
  `BaseModel(extra="allow")`. Both ARC generator/dataset modules are currently absent
  from `pyrefly.toml`, so add them only after making the reported annotations pass.

- **The login-node test command needs an explicit exception or a container.** The
  linting skill requires `uv run`; `uv` is not installed on this login node. The direct
  `PYTHONPATH=. python3 ...` reproductions work because the generator is stdlib-only,
  but §5 should not attribute that command to the testing skill. Either document this
  narrow exception or use the existing container path for the canonical test command.

### 8.4 Revised order if Option B is selected

Replace §6's A-oriented first half with:

1. Record the decision and correct `ARC_AGI_2_ENV_PLAN.md`/YAML claims before changing
   behavior. Preserve the two existing commits for experiment traceability.
2. Retain and consolidate the joint scheduler, preserve level multiplicity, remove the
   five extra axes from the shipped recipe/API, and migrate the size-ramp invariant
   tests to the survivor.
3. Fix launcher override transport, move the 4-node cadence/capacity values into one
   conventionally named model/run recipe, and make launch plus preflight consume it.
4. Introduce the validated config boundary once its post-rollback fields are stable;
   make removed keys fail with actionable messages.
5. Apply the independent reward readability cleanup and scorer reuse.
6. Run focused unit tests plus a generator matrix over every supported level/axis
   combination and many deterministic seeds. Invalid combinations must raise
   `ValueError` before sampling; valid combinations must never leak raw `IndexError` or
   helper `ValueError`, must preserve every identifiability/degeneracy guard, and must
   realize exactly what their task IDs report.

Suggested commit slices are: generator/scheduler correctness, config/dataset/tooling,
launcher/recipes, then design/status documentation. That keeps each behavior change
reviewable without rewriting experiment history.

---

## 9. Implementation record — Option B applied (2026-08-16)

Section 8's recommendation was taken: **Option B**, on branch `arc-cleanup-option-b`, as new
commits with the two original commits preserved for experiment traceability. Section 8.2's
corrections were applied where they changed the fix; where they asked for a *definition* before
code (object count, distractors) the axis was removed rather than guessed at.

### 9.1 What changed

**Generator (`arc_agi_generators.py`).**

- The five under-specified axes are gone, along with `GenerationOptions`; samplers take a plain
  `max_dim` again. `_MOTIF_KEEP_PROB = 0.7` and `_SCATTER_DENSITY = 0.2` are named constants, so
  `density` no longer means two opposite things (§1.6). `objects_pattern` is one color per
  rectangle again, and `_multicolor_shape` — the helper that raised out of the redraw loop — is
  deleted rather than patched (§1.2, §8.2.3): with palette and object count no longer independent
  axes, the shape it existed to build has no caller.
- Palette selection moved into a per-level `_PALETTE_RANGE` table, and the three open-coded
  `rng.choice([c for c in COLORS if c not in palette])` sites became one `_spare_color` helper that
  raises with a reason instead of a bare `IndexError`. Levels 2–5 are declared in
  `_SPARE_COLOR_LEVELS`, and `generate_task` checks the range leaves a spare color *before* the
  attempt loop. This is §8.2.1's constraint, enforced structurally rather than as an arithmetic
  bound on a config key that no longer exists.
- One scheduler. `_ramped_sizes`, `size_ramp_window`, `size_ramp_steps` and the mutual-exclusion
  `raise` are deleted; `_ramped_choices` over `(level, size)` subsumes them.
- **Repeat-to-weight preserved** (§8.2.2), and on *both* paths — a multiplicity map scales the ramp
  weights, and the unramped path (which validation uses) repeats each choice in the cycle. Without
  the second half, `levels` would have meant different things on the two splits.
- `generate_tasks` validates unknown levels, and `level_difficulty_order` defaults to the authored
  order rather than silently taking a different code path.

**Config boundary (`arc_synth.py`).** `SynthCurriculumConfig` is a `pydantic.BaseModel` holding the
complete regeneration boundary — seeds and counts included, per §8.3 — with `train_tasks()` /
`val_tasks()`. `ArcSynthDataset.__init__` is now `(**kwargs)` into that schema. Cross-field
validation lives there: seed separation, unknown levels, window capacity, and a window set without
a span. Removed keys raise an actionable migration error rather than vanishing into
`extra="allow"`, which is still needed for the merged `data.default` keys.

**Reward (`arc_agi_grid.py`).** `_reward_floor(weights)` is named once and used by both the
unparseable and echo paths, so the two cannot drift; `reward` has one assignment per branch. Per
§8.2.7 the diagnostic terms were already correct and were left alone.

**Tooling.** The preflight prints curriculum fields off `SynthCurriculumConfig.model_fields` (add a
field, it appears) and delegates validation to the schema, reporting rather than raising so one run
surfaces every problem. It gained `--nodes`, checked against the recipe's `cluster.num_nodes` —
§8.3's point that YAML cannot constrain a Slurm allocation. `arc_synth_score_val_dumps.py` rebuilds
the split through the same schema; its `--levels` fallback and `parse_known_args` dual mode are
gone.

**Launcher.** The override-transport bug is fixed at its cause. §8.3 diagnosed it correctly:
`ray.sub` does `printf '%s' "$COMMAND" > driver_command.sh` then `bash driver_command.sh`, so a
newline *ends* the command. `launch_experiment.sh` now folds newlines in `EXTRA_OVERRIDES` to
spaces and refuses to submit a command containing one. `CONFIG_PATH` is removed and
`arcsynth_early_answer` / `arcsynth_8n` are accepted `ENV` values, so the conventional path
resolves (§8.3). `val_period` moved out of the env layer into the recipe, restoring §5's
model-agnostic rule.

**Recipes.** `nanov3_arcsynth_8n_colocated.yaml` carries the 8-node run shape and its intended
`cluster.num_nodes`; `run_arc_preflight.sub` resolves both recipes with `--nodes` instead of
holding a second copy of the override string.

### 9.2 Verification

169 generator + grid tests pass on a login node. The size-ramp invariants were ported rather than
deleted (every window keeps every size, starts small-heavy and ends large-heavy, the ramp spans the
run not the dataset, a partial final window is still filled), and these were added: every level's
palette leaves room for the colors its rules add; 40 seeds × 6 levels raise nothing; `_spare_color`
rejects an exhausted palette with a reason; a repeated level takes a larger share *without* dropping
a combination from any window; and each removed config key is rejected by name.

The schema was exercised standalone (8000 train / 172 val rows; first row `L3_d6`, last `L5_d20`,
so the ramp lands) with every misconfiguration rejected and legitimate passthrough keys accepted.

Not run here: the dataset and environment test files need `datasets` and Ray, so they run via
`run_arc_tests.sub`. On §8.3's note about `uv` — it is genuinely absent from this login node, so
the stdlib-only `PYTHONPATH=. python3 -m pytest` path is the only local option; that is a property
of the two modules being stdlib-only, not a testing-skill recommendation, and §6 of
`ARC_AGI_2_ENV_PLAN.md` says so.

### 9.3 Deliberately not done

- **`pyrefly.toml`.** §8.3 is right that both modules are absent from it. Adding them is a separate
  change that should follow a pass making the annotations actually check, not ride along here.
- **Wiring object count or distractors.** §8.2.4 and §8.2.5 ask for a definition first. Item 7 of
  `ARC_AGI_2_ENV_PLAN.md` (the object/relational DSL) is where that definition belongs.
- **A launcher regression test for long override lists.** The newline guard makes the failure loud
  at submit time, which was the actual gap; a test would need to stand up `ray.sub`'s file-writing
  path to be meaningful.
