# ARC-AGI-2 environment — design

GRPO on ARC-AGI-2, plus a synthetic generator that produces solvable ARC-style tasks of tunable
difficulty. The prompt shows the few-shot grid pairs and asks the model to describe the
transformation before emitting the answer grid. The description is never scored — only the grid is.

Model: **nano-v3 30B-A3B**. Context **32768**. Branch `async_arc2`, off `async_colo_verify2`.

> Design and operating manual only. Results live in `reports/auto_research/arc-prompt/experiments.tsv`;
> per-commit history is on `async_arc`. Where a choice exists because a run demonstrated something,
> the reason is stated inline. Numbers quoted as scale references came from a 1.7B-class model since
> dropped from the async arm — re-measure before relying on them.

---

## 1. Data

`.../users/anthomas/ash/data/arc-prize-2025/` — `arc-agi_{training,evaluation}_{challenges,solutions}.json`.
Training is 1000 tasks → **1076** rows; evaluation is 120 tasks → **172** rows. The `test` split ships
no solutions and is unusable. One row per *test pair*, carrying that task's train pairs as few-shot
context; a task with several test inputs yields several non-independent rows. Grids ≤30×30, symbols 0–9.

## 2. Prompt

Cells are **space-delimited**, one row per line. Without a separator the tokenizer merges a row into
arbitrary multi-cell chunks and cell boundaries — what every ARC transformation operates on — become
invisible. `parse_grid` still accepts the compact form; rejecting it would discard reward over
punctuation.

`examples/prompts/arc_agi.txt` wraps the task body:

```
... Each cell holds a digit naming its color:
  0 = black    1 = blue     2 = red      3 = green    4 = yellow
  5 = grey     6 = pink     7 = orange   8 = azure    9 = maroon

<example>
<input>          <output>
0 0 3            0 0 6
0 3 3            0 6 6
3 0 0            6 0 0
</input>         </output>
</example>
<test_input>
0 3 0
3 3 0
0 0 3
</test_input>

1. Describe the example inputs and what they share.
2. Describe the example outputs the same way.
3. State the transformation rule.
4. Test the rule against every example; revise if any disagrees.
5. Apply it to the test input, answer between <answer> and </answer>.
```

(`<input>`/`<output>` are consecutive blocks, shown side by side here for brevity.)

Load-bearing details: the colors are **named** (0–9 are a palette, not magnitudes); the `<answer>`
tags live **inside step 5** — stranded in a trailing paragraph the model emits a markdown code block
and no tags; delimiters are **plain text**, since added vocabulary needs embeddings trained from
scratch and GRPO's signal is far too sparse for that.

`max_new_tokens: 8192`. The context is sized for the prompt; an unbounded response budget means
rollouts run to the cap, the async buffer never fills, and no training step starts. 4096 is too small
once step 4 means re-deriving several grids. Prompt lengths are tokenizer-specific — re-measure with
`tools/arc_agi_prompt_stats.py`.

## 3. Reward

```
reward = 1.00 * exact_match            # P == T
       + 0.20 * gain(cell_accuracy)    # vs. echoing the test input
       + 0.10 * gain(edit_similarity)  # vs. echoing the test input
       + 0.05 * color_recall  - 0.05 * extraneous_colors
       - 0.05 * shape_mismatch + 0.05 * format_valid
```

Defaults live on `ArcAgiEnvConfig`; every term is logged separately, and whether reward growth is
exact match or shaping is the primary diagnostic.

- **Dense terms break degenerate groups.** GRPO's advantage is computed within a group of rollouts on
  one prompt; if all score the same there is no gradient. A small policy solves ~nothing on real ARC,
  so binary exact match yields ~no signal. These terms make two wrong answers distinguishable.
- **Similarity is paid as gain over echoing the input** (`gain_over_baseline -> [-1,1]`, zero at the
  baseline). Scored absolutely, a copy of the input earns ~0.61 cell accuracy on the evaluation split
  — more than a small policy earns by reasoning — and training converges on the echo.
- **`best_alignment_cell_accuracy`** slides the smaller grid inside the larger and takes the best
  agreement, over the *larger* area so an oversized prediction cannot buy score. Falls back to a
  centered overlay when neither fits inside the other.
- **Edit distance earns its keep by disagreeing with the overlay**: a correct-but-shifted grid is
  nearly worthless to the overlay and costs one insertion here. Nothing needs to be differentiable.
- **Parser**: last well-formed grid between `<answer>`/`</answer>`, falling back to text after a
  final *unclosed* tag (generation may stop on the delimiter or hit the cap). Rejects ragged rows,
  bad symbols, empty grids, dims >30. The unparseable floor is `-(cell+edit+extraneous+shape)` and
  must sit strictly below the worst parseable answer, or the format term cannot bootstrap.
- **`copied_input` is logged** as the hack detector: a run whose cell accuracy merely tracks the copy
  baseline is hacked, not learning. Never reward CoT length — paying for tokens buys tokens.

**Validation** reports `grid_match` (the honest score), `cell_match`, `copied_input`, `format_valid`
and the per-term breakdown, **each also per level** (`grid_match/level_3`, `grid_match/real`) — an
aggregate cannot tell "solving the easiest level only" from "uniformly mediocre". Two sources are
concatenated into one loader: the synthetic held-out split and all 172 real evaluation rows, so their
schemas must match (real rows carry `level = REAL_ARC_LEVEL`). The loader does not shuffle and stops
after `max_val_samples // val_batch_size` batches — that budget must cover both or the tail is
silently dropped.

## 4. Synthetic generator

Real ARC-AGI-2 is built so frontier models score near zero; for a small policy it is a null signal
and exact match never leaves zero. The generator gives tunable difficulty, unlimited volume, and a
regime where `grid_match` can be optimized directly. Every task is a pure function of
`(seed, index, level)` — reproducible without a static dataset, and validation targets are
regenerable for offline scoring rather than stored.

| level | transformation |
|---|---|
| 0 | **identity** — plumbing gate, *not* a rung |
| 1 | dihedral-8 |
| 2 | single-color ops — drop / recolor / keep-only |
| 3 | geometric — tile, crop to bbox, scale, add border |
| 4 | structure — denoise, complete symmetry, fill enclosed |
| 5 | compositions — a color op then a shape op |

- **Level 0 stays out of the training mixture.** Echoing scores full marks there while the
  copy-relative reward makes an echo worth ~zero elsewhere, so mixing it in hands the policy a
  dominant degenerate strategy. Use `data.train.levels=[0]` to check the data path end to end.
- **Level 5 is always color-then-shape**: the other order is not identifiable, since a color op can
  erase the color a following color op is parameterized on.
- **Inputs come from pattern samplers, not noise** — crop needs a bounding box, denoise needs signal
  to distinguish from noise, symmetry completion needs a symmetry. Every sampler paints the whole
  palette, which makes identifiability hold by construction.
- **Two guards**, both unit-tested, because an unsolvable task is indistinguishable from a model that
  will not learn. *Identifiability*: a rule's parameter must appear in the train pairs meant to teach
  it. *Degeneracy*: no **stage** may leave **any** pair unchanged — one identity pair is an
  ambiguity, an unchanged test pair is solved by echoing, and per-stage catches what a whole-rule
  check misses (`recolor` then `flip_v` on identical rows teaches a flip nothing demonstrates).
- **Augmentation**: permute non-zero colors and apply a dihedral transform, identically across a
  whole task. Both conjugate the rule rather than change it, and block memorizing "color 3 disappears".

**Difficulty is mixed within every batch, and reweighted across the run.** A phase where every task
shares a difficulty gives uniformly hopeless or uniformly trivial groups, which contribute no
gradient — the exact failure the curriculum exists to fix. So `max_input_dim` takes a list
(`[6, 12, 20]`) and the mixture shifts small-heavy → large-heavy while every window keeps at least one
task of every size. `size_ramp_steps` must be the **run** length, not the dataset length: the dataset
is deliberately much larger so no task repeats, and a ramp spread over it is inert while still reading
as configured. The schedule lives in row order, so `data.shuffle: false`. Both knobs interpolate from
`grpo.*`. Applied to **size, not level** — size is monotone in difficulty, the level index is not
(single-color ops measured *easier* than dihedral).

## 5. Code

| File | Contents |
|---|---|
| `nemo_rl/environments/arc_agi_grid.py` | serializer, parser, scorers, reward (numpy + stdlib) |
| `nemo_rl/environments/arc_agi_generators.py` | ladder, patterns, guards, augmentation, size schedule (stdlib) |
| `nemo_rl/environments/arc_agi_environment.py` | Ray actor, `ArcAgiEnvConfig`, per-level metrics |
| `nemo_rl/data/datasets/response_datasets/arc_{agi,synth}.py` | real / synthetic datasets, identical row schema |
| `nemo_rl/data/processors.py` | `arc_agi_data_processor`, serves both |
| `examples/prompts/arc_agi.txt`, `examples/configs/async/env_arc{,_synth}.yaml` | prompt, model-agnostic env layers |
| `examples/configs/async/nanov3_arc{,synth}_{colocated,non_colocated}.yaml` | recipes |
| `tools/arc_agi_prompt_stats.py` | prompt lengths at the real tokenizer (`--synth` for the ladder) |
| `tools/arc_synth_preflight.py` | resolves a recipe and builds its datasets, on CPU |
| `tools/arc{_agi,_synth}_score_val_dumps.py` | offline re-scorers |

**Keep the env layers model-agnostic.** Learning rates, chat-template flags and KV-cache sizes belong
in the model's recipe — a shared `lr` landing under a model's own `min_lr` kills every policy worker
at construction.

## 6. Tests

`tests/unit/environments/test_arc_agi_{grid,generators,environment}.py` and
`tests/unit/data/datasets/test_arc_synth_dataset.py`: parser rejections, scorer edge cases, each
reward term plus the copy-input and flood-colors hacks scoring below a genuine partial solve, the
`step()` contract, the dihedral group, both guards, augmentation commuting with the rule, and the
size schedule keeping every size per window while the mean rises. Ray actors get `# pragma: no cover`.

Grid and generator modules are Ray/torch/GPU-free, so they run on a login node —
`PYTHONPATH=. python3 -m pytest <copy outside tests/>`, since `tests/unit/conftest.py` imports Ray.
Everything else: `.run_arc_tests.sub`.

## 7. Running it

**4-node nano recipe — known good** (job 6174618: 60 steps + 4 validations in 2h01m, ~73 s/step,
~11 min per validation):

```
SUBMIT_ACCOUNT=nemotron_sw_post ENV=arcsynth NUM_ACTOR_NODES=4 \
  MAX_STEPS=60 TIMEOUT_MIN=240 RUN_TAG=<tag> \
  EXTRA_OVERRIDES="grpo.num_prompts_per_step=32 policy.train_global_batch_size=512 \
                   policy.generation.mcore_generation_config.buffer_size_gb=40 grpo.val_period=20" \
  bash launch_experiment.sh colocated
```

Why those overrides:

- **4 nodes = 16 GPUs, and nano's parallelism divides cleanly** — TP2 × PP1 × CP1 → DP 8, and
  ETP1 × EP4 → EDP 4. No parallelism change is needed, and memory is not the constraint (~31 GB/GPU
  of ~186 GB observed).
- **`num_prompts_per_step` 128 → 32** (global batch 2048 → 512) is what makes 60 steps fit. At nano's
  default batch a 4-node run needs ~7 h against the `batch` partition's hard **4 h** cap. Caveat:
  nano's `lr` was tuned for 2048, so gradients are noisier than its config assumes.
- **`val_period: 20`** — validation is 344 rows and ~11 min a checkpoint; four fit, seven do not.
- ≤4 nodes gets **interactive QOS**, which allocates in ~90 s. Above that, `QOS=normal` has queued
  for over 24 h. This is the single biggest lever on iteration speed.

Gotchas:

- **Pre-flight first** — `.run_arc_preflight.sub` resolves a recipe and builds its datasets on CPU.
  Its four guards each exist because a bug surfaced only at runtime and cost cluster time: ramp
  window vs prompts-per-step, ramp vs `shuffle`, `lr` vs `min_lr`, checkpoint metric vs the metrics
  the ARC env emits.
- **Never `CMP=1` with an ARC env** — it forces `max_total_sequence_length=16384`, below the
  worst-case ARC prompt, and an overlong prompt is *masked*, not raised. The launcher refuses it.
- **`--no-container-mount-home`** for anything run in the container by hand, or enroot mounts your
  home over the container's and hides `uv`.
- Validation chats land in `logs/exp_NNN/val_data_step<N>.jsonl`; both re-scorers work offline from
  them, and the synthetic one regenerates targets from the run's `val_seed`.
- **A caught async error still exits `COMPLETED` with status 0.** Read the driver log, not `sacct`.
- Slurm's projected start is a backfill estimate, not a reservation, and can recede indefinitely.

## 8. Next moves

**Prompt and inference**

1. **One-shot example in the prompt.** The five-step structure is described but never demonstrated,
   so the model invents the format. Prepend one fully worked puzzle — grids, the five steps, and the
   `<answer>` block. Safe by construction: the parser takes the *last* answer block, so an exemplar
   containing one cannot be mistaken for the response. Costs prompt length, which is already at p99
   ~15k; measure before and after.
2. **Category-conditioned one-shot.** A single exemplar biases toward its own transform type. Instead
   keep one exemplar per category, classify the puzzle first, then splice in the matching one.
   Synthetic tasks come with ground-truth categories (the level and rule name are already in
   `task_id`), so the classifier can be trained and measured cheaply before it is trusted on real
   ARC, where no label exists. Risk to measure: a misclassification supplies a *misleading* exemplar,
   which may be worse than a generic one.
3. **Execute-and-compare feedback loop.** Step 4 asks the model to test its rule against every
   example, but it does so in-context — it hallucinates the check. Instead run a second chat that
   applies the *predicted transform* to each training input, produces a grid, and diffs it against
   that pair's known output; feed the mismatches back into the original chat so the model can revise
   the rule. Two constraints: compare only against **train** pairs, never the test target, or the
   answer leaks; and this makes the environment multi-turn (`max_rollout_turns` is 1 today), so turn
   credit assignment needs deciding. This is the highest-value item here — it converts the prompt's
   self-check from narration into an actual verifier.

**Curriculum**

4. **Rank levels by measured difficulty, not authored index.** The numbering is a guess and already
   known wrong in one place. Score a fixed policy per level and order by solve rate. Prerequisite for
   the next item.
5. **Ramp transform complexity jointly with size.** With levels ranked, define `d = f(level_rank,
   grid_size)` and reweight along `d` using the machinery `_ramped_sizes` already implements — every
   window spans the range, the mass moves. Needs a second input and a joint schedule, not a new design.
6. **Add the axes the ladder lacks**: few-shot count (2 pairs is much harder than 4), palette size,
   object count and density, composition depth (level 5 is depth 2; `Rule.stages` already generalizes
   over a tuple), and **distractors** — elements the rule ignores, which are most of what makes real
   ARC hard and which the ladder has none of.
7. **Make the generator genuinely ARC-like.** The ladder is six hand-written families over a fixed
   vocabulary, and a policy can learn that vocabulary rather than the skill. Sample rules from a small
   DSL instead of enumerating them, and add the classes real ARC leans on: object-centric rules (move,
   count, sort, select the largest / odd-one-out), relational and conditional rules, and mappings that
   are not cellwise. This is also the direct test of open question (2) below.
8. **Frontier-weighted sampling.** Reweight toward levels whose solve rate is strictly between 0 and 1
   — the band where groups carry gradient. Needs an environment → generator feedback path across Ray
   actors; the per-level metrics it depends on now exist. Do it after (5), so there is a baseline.

**Reward and objective**

9. **Make the echo strictly worse than a genuine attempt.** An echo still collects `color_recall` and
   `format_valid` and dodges the extraneous-color and shape penalties, so it sits at a small
   *positive* floor while a real attempt that misses scores a negative gain. On any task the policy
   cannot solve, echoing is therefore reward-maximizing. Re-baselining moved the zero point but did
   not invert that ordering. Fix: score `pred == test_input` at or below the unparseable floor
   wherever it is not the answer.
10. **Question whether exact match is the right objective.** It needs every cell right, so a rule the
    model understands still scores zero on a large grid. Small grids are the cheap test and are
    already supported. If solve rates stay at zero there, the objective — not the curriculum — is what
    to change, and annealing `w_cell` toward 0 as grid match rises becomes the live question.

## 9. Open questions

1. **Eval-set size.** 172 rows is small; step-to-step noise swamps real movement. Consider holding out
   a slice of the 1000 training tasks for online validation and reserving the official split for
   milestone reporting.
2. **Does synthetic transfer to real ARC at all?** The generator's premise is that exact-match signal
   on solvable tasks teaches what shaping cannot. The model may instead learn the generator's
   vocabulary and transfer nothing. Real-ARC validation runs at every checkpoint to measure this; a
   null result is an answer, not a failed run.
3. **Does the edit-distance term earn its weight?** It shipped with the copy-relative re-baselining
   and has never been isolated. Cheap to ablate (`edit_weight: 0.0`) once `grid_match` is nonzero.
