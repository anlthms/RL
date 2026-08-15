# ARC-AGI-2 environment — design

A NeMo-RL environment that runs GRPO on ARC-AGI-2, plus a synthetic task generator that produces
solvable ARC-style tasks of tunable difficulty.

The prompt lays out the few-shot input/output grids as delimited text and asks the model to describe
the transformation it infers before emitting the answer grid. Nothing scores the description — it is
free-form by construction. Only the final grid is scored.

Model: **nano-v3 30B-A3B**. Context: **32768**. Branch: `async_arc2`, off `async_colo_verify2`.

> This document is the design and the operating manual. Run-by-run results are **not** kept here;
> they live in `reports/auto_research/arc-prompt/experiments.tsv`, and the per-commit history of how
> the design got here is on `async_arc`. Where a design choice exists because a run demonstrated
> something, the reason is stated inline — those sentences are the surviving evidence. Earlier work
> was measured on a 1.7B-class model since dropped from the async arm, so absolute numbers quoted
> here as scale references should be re-measured before being relied on.

---

## 1. Data

`/lustre/fs1/portfolios/nemotron/projects/nemotron_sw_pre/users/anthomas/ash/data/arc-prize-2025/`

| File | Use |
|---|---|
| `arc-agi_training_challenges.json` + `..._solutions.json` | **train.** 1000 tasks → 1076 test-pair rows |
| `arc-agi_evaluation_challenges.json` + `..._solutions.json` | **validation.** 120 tasks → 172 test-pair rows |
| `arc-agi_test_challenges.json` | unusable — no solutions file (competition holdout) |

Challenges are `{task_id: {"train": [{"input", "output"}, ...], "test": [{"input"}, ...]}}`; the
solutions file maps `task_id -> [grid, ...]`, positionally aligned with `test` (verified for every
task in both splits). One dataset row per *test pair*, carrying that task's train pairs as few-shot
context. Grids are ≤30×30, symbols 0–9.

Some tasks contribute more than one row (different test inputs, same few-shot context), so rows
within a task are not independent.

## 2. Prompt construction

Cells are **space-delimited**, one row per line. Without a separator a row is a single run of digits
that the tokenizer merges into arbitrary multi-cell chunks, so cell boundaries — the thing every ARC
transformation operates on — are invisible to the model. `parse_grid` still accepts the compact form
on the way back in; rejecting a spaceless answer would discard reward over punctuation.

The task body (`format_task_prompt`) is wrapped by `examples/prompts/arc_agi.txt`:

```
You are solving a puzzle from the ARC benchmark. Below are example input/output
grid pairs. Every pair follows the same transformation rule.

Each grid is a 2-D picture written one row per line, with the cells of a row
separated by single spaces. Each cell holds a digit naming its color:

  0 = black    1 = blue     2 = red      3 = green    4 = yellow
  5 = grey     6 = pink     7 = orange   8 = azure    9 = maroon

<example>
<input>
0 0 3
0 3 3
3 0 0
</input>
<output>
0 0 6
0 6 6
6 0 0
</output>
</example>
... (remaining train pairs)
<test_input>
0 3 0
3 3 0
0 0 3
</test_input>

Work through the puzzle in this order, then answer.

1. Describe the example input grids ... and what they have in common.
2. Describe the example output grids the same way.
3. State the transformation rule.
4. Test the rule against every example; revise it if any disagrees.
5. Apply the final rule to the test input and write the resulting grid between
   <answer> and </answer>. ... at most 30 rows and 30 cells per row.
```

Three things about this layout are load-bearing:

- **The colors are named.** 0–9 are a palette, not magnitudes.
- **The `<answer>` tags live inside step 5**, not in a trailing paragraph. Stranded at the end, the
  model produces a correct-looking grid in a markdown code block as the last item of a five-part
  write-up and never emits the tags at all.
- **Plain-text delimiters, not new vocabulary.** Added tokens need embeddings trained from scratch,
  and GRPO's signal is far too sparse for that.

### Token budget

Prompt lengths are tokenizer-specific. Re-measure with `tools/arc_agi_prompt_stats.py` whenever the
model changes; it defaults to the model the async arm trains. As a scale reference, at a 1.7B-class
tokenizer the space-delimited corpus ran p99 ≈ 12.3k (training) / 14.9k (evaluation) tokens with a
worst row of 16593 — nothing overflowing 32768, leaving room for the response.

`max_new_tokens: 8192` caps the response. The context is sized for the *prompt*; with an unbounded
response budget rollouts run to the cap, the async buffer never fills, and no training step starts.
Step 4 of the prompt means re-deriving several grids, so 4096 is not enough — running out of budget
before reaching `<answer>` looks like a reward bug rather than a length one.

## 3. Reward design

Scored on the extracted final grid only. Let `T` be the target and `P` the prediction.

```
reward = w_exact * exact_match           # 1.0 iff P == T
       + w_cell  * gain(cell_accuracy)   # vs. echoing the test input
       + w_edit  * gain(edit_similarity) # vs. echoing the test input
       + w_color * color_recall
       - w_extra * extraneous_color_fraction
       - w_shape * shape_mismatch
       + w_fmt   * format_valid
```

Weights: `exact 1.0`, `cell 0.20`, `edit 0.10`, `color 0.05`, `extraneous 0.05`, `shape 0.05`,
`format 0.05`. Defaults live on `ArcAgiEnvConfig`; every term is logged separately, and whether
reward growth is exact match or shaping is the single most important diagnostic here.

**The dense terms exist to break degenerate groups.** GRPO computes advantage within a group of
rollouts on one prompt; if every rollout scores the same, advantage is zero and the group contributes
no gradient. On ARC-AGI-2 a small policy solves approximately nothing, so a binary exact-match reward
yields approximately no training signal. These terms make two wrong answers distinguishable — their
absolute scale barely matters, their presence does.

**Both similarity terms are paid on their gain over echoing the test input**, via
`gain_over_baseline(score, baseline) -> [-1, 1]`, zero at the baseline and normalized by the room
available in each direction. Scored absolutely, a copy of the input earns ~0.61 cell accuracy on the
ARC-AGI-2 evaluation split — more than a small policy earns by reasoning — and training converges on
the echo. Re-baselining moves the zero point to where it belongs without singling anything out.

**Cell accuracy uses `best_alignment_cell_accuracy`**: slide the smaller grid entirely inside the
larger (valid-mode cross-correlation) and take the best agreement, denominated by the *larger* of the
two areas so an oversized prediction cannot buy score. Colors are labels, so the per-cell operator is
equality. Falls back to a centered overlay when neither grid fits inside the other.

**The edit-distance term earns its place by disagreeing with the overlay**: a prediction that is
correct but shifted one row is nearly worthless to the overlay and costs edit distance one insertion.
Nothing here has to be differentiable, so the two can simply be added.

**Parser contract.** Last well-formed grid between `<answer>`/`</answer>`; falls back to the text
after a final *unclosed* `<answer>`, since generation may stop on the delimiter or be cut at the
token cap and the grid is right there. Rejects ragged rows, out-of-range symbols, empty grids, and
dimensions >30. The unparseable floor is `-(cell + edit + extraneous + shape)`: full penalties, no
format credit. That floor must sit strictly below the worst parseable answer, or the format term
cannot bootstrap — at step 0, when nothing is solved, the format gap is the only reward difference
the policy can act on.

**Known hack, and the detector for it.** Copying the input beats chance on many tasks. `copied_input`
is logged per sample for exactly this reason: a run whose cell accuracy merely tracks the copy
baseline is hacked, not learning. Do not reward CoT length — if longer reasoning helps, GRPO will
find it; paying for tokens buys tokens.

### Validation metrics

- **`grid_match`** — exact-match rate, the honest ARC score.
- **`cell_match`** — best-alignment cell accuracy, which moves long before grid match does.
- **`copied_input`**, **`format_valid`**, and the full per-term breakdown.
- All of the above **per level** (`grid_match/level_3`, `grid_match/real`). Without the split an
  aggregate cannot distinguish "solving the easiest level and nothing else" from "uniformly
  mediocre", which is the whole question the curriculum asks.

Validation covers two sources at once: the synthetic held-out split (fresh seed, same mixture) and
all 172 real ARC-AGI-2 evaluation rows. They are concatenated into one dataloader, so their row
schemas must match exactly — real rows carry `level = REAL_ARC_LEVEL` for this reason. The loader
does not shuffle and stops after `max_val_samples // val_batch_size` batches, so that budget must
cover both sources or the tail is silently dropped.

## 4. The synthetic generator

ARC-AGI-2 is built so that frontier models score near zero; for a small policy it is not a hard
learning problem but a null signal, and exact match never leaves zero. The generator supplies tasks
of tunable difficulty, unlimited volume, and a regime where `grid_match` is nonzero and can be
optimized directly rather than approximated.

Every task is a pure function of `(seed, index, level)`, so a run is reproducible without a static
dataset — and validation targets can be regenerated for offline scoring rather than stored.

### The ladder

| level | transformation |
|---|---|
| 0 | **identity** — the plumbing gate, *not* a curriculum rung. See below. |
| 1 | **dihedral-8** — rot90/180/270, flip-h/v, transpose, anti-transpose |
| 2 | **single-color ops** — drop color X; recolor X→Y; keep only X |
| 3 | **geometric** — tile k×m; crop to the non-background bounding box; scale by k; add a border |
| 4 | **structure** — denoise (drop isolated cells); complete a symmetry; fill enclosed regions |
| 5 | **compositions** — a color op followed by a shape op |

**Level 0 is deliberately absent from the training mixture.** Echoing the input scores full marks
there while the copy-relative reward makes an echo worth about *zero* — not negative — everywhere
else, so mixing it in hands the policy a dominant degenerate strategy: always echo. Use it via
`data.train.levels=[0]` to check the data path, prompt, and parser end to end; never as a rung.

**Level 5 is always color-then-shape.** The other order is not identifiable — a color op can erase
the very color a following color op is parameterized on. Shape ops are color-agnostic, so this order
never can. Both stages share one palette, and the shape stage supplies the input sampler, since it is
the stage with an opinion about size (tile and scale bound the input so the output still fits) and
about structure (crop needs a background border to crop away).

### Input patterns, not noise

Uniform random grids make several levels degenerate: crop needs a bounding box, denoise needs signal
to distinguish from noise, symmetry completion needs a symmetry. Inputs come from a small library of
pattern samplers — scattered points, filled and hollow rectangles, lines, repeated motifs, objects on
a background — and each rule names the one that gives it something to act on. Every sampler paints
the whole palette, which is what makes identifiability hold by construction.

### Two guards the generator must enforce

Both produce tasks unsolvable in principle, which during a training run is indistinguishable from the
model failing to learn.

1. **Identifiability.** A rule's parameter must appear in the train pairs that are supposed to teach
   it. "Drop color 3" is not inferable from examples containing no 3.
2. **Degeneracy.** No *stage* of a rule may leave *any* pair unchanged. Stricter than "every example
   is unchanged", in two ways that both matter: one identity pair inside a non-identity task is an
   ambiguity, and a task whose test pair is unchanged is solved by echoing the input. Per-*stage*
   catches what a whole-rule check misses — `recolor(4→6)` then `flip_v` on a grid of identical rows
   changes the grid but teaches a flip no example demonstrates.

Rejection is the normal path; exhausting the attempt budget means the level's rules and patterns
disagree, which raises rather than returning a bad task.

### Augmentation

Permute the non-zero colors and apply a dihedral transform, identically across every pair of a task.
Both conjugate the rule rather than changing it, so the task stays consistent — and the model cannot
memorize "color 3 is the one that disappears" or "the answer is always wider than the input".

### Difficulty is mixed within a batch, and reweighted over the run

**Mixed, never ramped.** A phase in which every task shares a difficulty gives groups that are
uniformly hopeless or uniformly trivial, and those contribute no gradient — the degenerate-group
failure the curriculum exists to fix, reintroduced by the curriculum. Every batch spans the range.

**Reweighted, so mean difficulty still rises.** `max_input_dim` takes a list (`[6, 12, 20]`), and
with `size_ramp_window` set to the trainer's prompts-per-step the mixture shifts from small-heavy to
large-heavy across the run while every window keeps at least one task of every size.
`size_ramp_steps` must be the *run* length, not the dataset length: the dataset is deliberately many
times larger than any run so no task repeats, and a ramp spread over it leaves the run at a fraction
of its schedule — inert, but still reading as configured. The schedule lives in the row order, so it
requires `data.shuffle: false`. Both knobs interpolate from `grpo.*` so they track the model.

**Applied to size, not to level.** Grid size is monotone in difficulty; the level index is not
(single-color ops measured *easier* than dihedral). Ramping by level number would be ramping by an
authored guess. §8 covers how to fix that properly.

## 5. Where the code lives

| File | Contents |
|---|---|
| `nemo_rl/environments/arc_agi_grid.py` | Serializer, parser, scorers, reward. numpy + stdlib only. |
| `nemo_rl/environments/arc_agi_generators.py` | The ladder: transformations, patterns, guards, augmentation, size schedule. Stdlib only. |
| `nemo_rl/environments/arc_agi_environment.py` | `ArcAgiEnvironment` Ray actor + `ArcAgiEnvConfig`, per-level metrics. |
| `nemo_rl/data/datasets/response_datasets/arc_agi.py` | Real ARC: joins challenges + solutions, one row per test pair. |
| `nemo_rl/data/datasets/response_datasets/arc_synth.py` | Synthetic: materializes the mixture from `(seed, index, level)`. |
| `nemo_rl/data/processors.py` | `arc_agi_data_processor` — one processor, both datasets, identical row schema. |
| `examples/prompts/arc_agi.txt` | The prompt above. |
| `examples/configs/async/env_arc{,_synth}.yaml` | Model-agnostic data/env layers. |
| `examples/configs/async/nanov3_arc{,synth}_{colocated,non_colocated}.yaml` | Recipes. |
| `tools/arc_agi_prompt_stats.py` | Prompt lengths at the real tokenizer; `--synth` for the ladder. |
| `tools/arc_synth_preflight.py` | Resolves a recipe and builds its datasets, on CPU. |
| `tools/arc_agi_score_val_dumps.py`, `tools/arc_synth_score_val_dumps.py` | Offline re-scorers. |

**Keep the env layers model-agnostic.** Learning rates, chat-template flags, and KV-cache sizes are
properties of the model and belong in the model's recipe. A shared `lr` that lands under a model's
own `min_lr` kills every policy worker at construction.

## 6. Tests

`tests/unit/environments/test_arc_agi_{grid,generators,environment}.py`,
`tests/unit/data/datasets/test_arc_synth_dataset.py`.

Parser (well-formed, ragged, out-of-range, empty, multiple candidates, no delimiters, oversize);
scorers (identical grids, smaller/larger predictions, odd/even offsets, disjoint, the oversized-P
penalty biting via the max-area denominator); reward terms in isolation and combined, with the
copy-the-input and flood-all-colors hacks scoring below a genuine partial solve; serializer
round-trip; the `step()` contract; the dihedral group of order 8; both generator guards; augmentation
commuting with the rule; the size schedule keeping every size in every window while the mean rises.
Ray actors get `# pragma: no cover`.

The grid and generator modules avoid Ray, torch, and GPU, so their tests run on a login node:
`PYTHONPATH=. python3 -m pytest <copy of the test file outside tests/>` — `tests/unit/conftest.py`
imports Ray, so run from a scratch directory. Everything else needs the container:
`.run_arc_tests.sub`.

## 7. Running it

```
SUBMIT_ACCOUNT=nemotron_sw_post ENV=arcsynth NUM_ACTOR_NODES=4 MAX_STEPS=60 \
  TIMEOUT_MIN=240 RUN_TAG=<tag> bash launch_experiment.sh colocated
```

The launcher is `ENV × topology` over nano-v3; `ENV` is `gym | math | arc | arcsynth`. There is no
MODEL axis. Each launch needs explicit approval.

- **Pre-flight first.** `.run_arc_preflight.sub` resolves the recipe and builds its datasets on a CPU
  node. It carries four guards, each added after a bug that surfaced only at runtime and cost cluster
  time: ramp window vs prompts-per-step, ramp vs `data.shuffle`, `lr` vs `min_lr`, and the checkpoint
  metric vs the metrics the ARC environment actually emits.
- **Never combine `CMP=1` with an ARC env.** It forces `max_total_sequence_length=16384`, below the
  worst-case ARC prompt, and an overlong prompt is masked rather than raised — so the combination
  degrades a run silently. The launcher refuses it.
- **`--no-container-mount-home` is required** when running anything in the container by hand. Without
  it enroot mounts the caller's home over the container's, hiding `uv` behind a `command not found`
  that reads as a broken image.
- Runs write to `logs/exp_NNN/`, and every validation chat lands in `val_data_step<N>.jsonl` there.
  Both re-scorers work offline from those dumps; the synthetic one regenerates targets from the run's
  `val_seed`, so nothing needs to have been stored.
- A job can exit `COMPLETED` with status 0 after the async loop caught an error and shut down
  cleanly. **Check the driver log, not just `sacct`.**
- Slurm's projected start is a backfill estimate, not a reservation, and can recede indefinitely.

## 8. Next moves

The generator has two difficulty axes and only one of them is scheduled. The work below makes
difficulty a first-class, multi-dimensional quantity.

**1. Rank levels by measured difficulty, not by authored index.** The ladder's numbering is a guess,
and is already known to be wrong in at least one place — single-color ops measured easier than
dihedral. Score a fixed policy across every level and order them by solve rate. That ranking is the
prerequisite for everything else here: without it, "increase transform complexity" means "increase a
number someone made up".

**2. Ramp transform complexity alongside size.** Once levels are ranked, define a scalar difficulty
`d = f(level_rank, grid_size)` and reweight the mixture along `d` with the machinery `_ramped_sizes`
already implements — every window spans the range, the mass moves from easy to hard across the run.
This needs a second input and a joint schedule, not a new design.

**3. Add the axes the ladder does not have.** Each is cheap in the generator and independently
tunable, which is what makes a multi-dimensional curriculum worth building:
   - **few-shot count** — 2 pairs is materially harder to infer from than 4.
   - **palette size** — more colors, more to track.
   - **object count and density.**
   - **composition depth** — level 5 is depth 2; depth 3+ extends `Rule.stages`, which already
     generalizes over a tuple of stages.
   - **distractors** — elements the rule ignores. Most of what makes real ARC hard, and the ladder
     currently has none.

**4. Close the loop: frontier-weighted sampling.** Reweight toward levels whose solve rate is
strictly between 0 and 1 — the band where GRPO groups actually carry gradient. This needs a feedback
path from the environment back to the data generator across Ray actors, which is why it was deferred;
the per-level metrics it depends on now exist. Do it after the fixed schedule in (2), so there is a
baseline to beat.

**5. Make the echo strictly worse than a genuine attempt.** Under copy-relative scoring an echo still
collects `color_recall` and `format_valid` and dodges the extraneous-color and shape penalties, so it
sits at a small *positive* floor, while a real attempt that misses scores a negative gain. On any
task the policy cannot solve, echoing is therefore the reward-maximizing move. Re-baselining moved
the zero point but did not invert that ordering. The targeted fix is to score `pred == test_input` at
or below the unparseable floor wherever it is not the answer.

**6. Ask whether exact match is the right objective at all.** Exact match needs every cell right, so
a rule the model genuinely understands still scores zero on a large grid. Small grids are the cheap
test and are already supported via `max_input_dim`. If solve rates stay at zero even there, the
objective — not the curriculum — is the thing to change, and annealing `w_cell` toward 0 as grid
match rises becomes the live question rather than a footnote.

## 9. Open questions

1. **Eval-set size.** 120 tasks / 172 rows is small; step-to-step noise will swamp real movement.
   Consider holding out a slice of the 1000 training tasks for online validation and reserving the
   official evaluation split for milestone reporting.
2. **Does solving synthetic tasks transfer to real ARC at all?** The generator's premise is that
   exact-match signal on solvable tasks teaches something shaping cannot. It is entirely possible the
   model learns the generator's vocabulary and transfers nothing. Real-ARC validation runs alongside
   the synthetic split at every checkpoint precisely to measure this, and a null result is a real
   answer rather than a failed run.
3. **Does the edit-distance term earn its weight?** It shipped together with the copy-relative
   re-baselining, so its individual contribution has never been isolated. Cheap to ablate
   (`edit_weight: 0.0`) in a setting where `grid_match` is nonzero.
