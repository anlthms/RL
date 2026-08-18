# ARC-AGI-2 environment — design

GRPO on ARC-AGI-2, plus a synthetic generator that produces solvable ARC-style tasks of tunable
difficulty. The prompt shows the few-shot grid pairs and asks the model to describe the
transformation before emitting the answer grid. The description is never scored — only the grid is.

Model: **nano-v3 30B-A3B**. Context **32768**. Branch `async_arc2`, off `async_colo_verify2`.

> Design and operating manual. Results live in `reports/auto_research/arc-curriculum/experiments.tsv`;
> per-commit history is on `async_arc`. Where a choice exists because a run demonstrated something,
> the reason is stated inline. Numbers quoted as scale references came from a 1.7B-class model since
> dropped from the async arm — re-measure before relying on them.

**Current result (2026-08-16).** The campaign target is at least **9/172 exact solves (5%) on the
real ARC-AGI-2 evaluation rows**; synthetic rows do not count. Job 6211877 completed 60 steps in
2h42m: synthetic exact rose **17 → 51/172**, but real exact stayed **0/172** at every checkpoint.
Best real diagnostics were 18.55% cell match and 45.35% valid format at step 50. This is evidence
that the current generator is learnable but does not transfer exact solves. The next prepared run is
`early_answer_4k_v3`, which targets responses that exhaust the token budget before emitting a grid.

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

The baseline uses `max_new_tokens: 8192`, but job 6211877 frequently ran to that cap and spent
15–20 minutes per validation. The follow-up prompt `arc_agi_early_answer.txt` requires a provisional
answer within 1200 tokens and permits a corrected final block; its dedicated recipe safely reduces
the cap to 4096. Prompt lengths are tokenizer-specific — re-measure with
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
- **A non-answer echo receives the unparseable floor.** True identity solutions remain exact, but
  copying the test input when it is not the target can no longer beat a genuine imperfect attempt.

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

**Difficulty is mixed within every batch, and reweighted jointly across the run.** Levels use the
measured order L3, L2, L1, L4, L5 and are crossed with sizes `[6, 12, 20]`; every optimizer window
contains all 15 combinations while extra mass shifts easy → hard. The same joint rank selects
few-shot count `[4,3,2]` — the only axis beyond transform and size that is wired end to end. Five
others (palette size, object count, density, composition depth, distractors) shipped half-wired and
were removed; setting one now raises. See `ARC_CLEANUP_PLAN.md` §8–9. A level repeated in `levels` is
weighted proportionally on both splits. The schedule lives in row order (`data.shuffle: false`),
spans `grpo.max_num_steps`, and the 8000-row dataset is large enough that a 60-step run never
repeats a task.

## 5. Code

| File | Contents |
|---|---|
| `nemo_rl/environments/arc_agi_grid.py` | serializer, parser, scorers, reward (numpy + stdlib) |
| `nemo_rl/environments/arc_agi_generators.py` | ladder, patterns, guards, augmentation, size schedule (stdlib) |
| `nemo_rl/environments/arc_agi_environment.py` | Ray actor, `ArcAgiEnvConfig`, per-level metrics |
| `nemo_rl/data/datasets/response_datasets/arc_{agi,synth}.py` | real / synthetic datasets, identical row schema. Real ARC is a **validation** source only — the arm that trained on it is gone, having produced `grid_match == 0.0000` four times |
| `nemo_rl/data/processors.py` | `arc_agi_data_processor`, serves both |
| `examples/prompts/arc_agi{,_early_answer}.txt`, `examples/configs/async/env_arc_synth{,_early_answer}.yaml` | prompts and model-agnostic env layers |
| `examples/configs/async/nanov3_arcsynth{,_4n,_8n,_early_answer}_{colocated,non_colocated}.yaml` | recipes, one per allocation the run was sized for |
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
`step()` contract, the dihedral group, both guards, augmentation commuting with the rule, the joint
curriculum invariants (every window keeps every combination, mean difficulty rises, the ramp spans
the run not the dataset, a repeated level is weighted without dropping a combination), that every
level's palette leaves room for the colors its rules add, and that removed config keys raise rather
than being ignored. Ray actors get `# pragma: no cover`.

Grid and generator modules are Ray/torch/GPU-free, so they run on a login node —
`PYTHONPATH=. python3 -m pytest <copy outside tests/>`, since `tests/unit/conftest.py` imports Ray.
Everything else: `run_arc_tests.sub`.

## 7. Running it

**The run's shape lives in the recipe, not in `EXTRA_OVERRIDES`.** Every recipe named for an
allocation (`nanov3_arcsynth_8n_colocated.yaml`) carries the batch, cadence and KV budget it was
sized for, and declares `cluster.num_nodes` so the preflight can check the submission against it.
The 4-node campaign instead carried five overrides on the command line plus a second copy in
`run_arc_preflight.sub`, and the two could not be kept in agreement — see the override-transport
gotcha below for what that cost.

**4-node — known good** (job 6174618: 60 steps + 4 validations in 2h01m, ~73 s/step, ~11 min per
validation). 60 steps at 32 prompts; interactive QOS allocates in ~90 s:

```
SUBMIT_ACCOUNT=nemotron_sw_post ENV=arcsynth_4n NUM_ACTOR_NODES=4 \
  TIMEOUT_MIN=240 RUN_TAG=<tag> bash launch_experiment.sh colocated
```

**8-node** — 80 steps at 64 prompts, half of nano's tuned batch rather than a quarter:

```
SUBMIT_ACCOUNT=nemotron_sw_post ENV=arcsynth_8n NUM_ACTOR_NODES=8 QOS=normal \
  TIMEOUT_MIN=240 RUN_TAG=<tag> bash launch_experiment.sh colocated
```

`QOS=normal` is required above 4 nodes and **`QOS=` (empty) does not work**: `submit_nemorl.sh`
does `QOS=${QOS:-interactive}`, and `:-` substitutes on empty as well as unset, so an empty value
resolves back to `interactive` and the job is rejected for exceeding its 4-node limit. The comment
in that script saying otherwise is wrong.

Why the 4-node numbers:

- **4 nodes = 16 GPUs, and nano's parallelism divides cleanly** — TP2 × PP1 × CP1 → DP 8, and
  ETP1 × EP4 → EDP 4. No parallelism change is needed, and memory is not the constraint (~31 GB/GPU
  of ~186 GB observed).
- **`num_prompts_per_step` 128 → 32** (global batch 2048 → 512) is what makes 60 steps fit. At nano's
  default batch a 4-node run needs ~7 h against the `batch` partition's hard **4 h** cap. Caveat:
  nano's `lr` was tuned for 2048, so gradients are noisier than its config assumes.
- **`val_period: 20`** — validation is 344 rows and ~11 min a checkpoint; four fit, seven do not.
- **Training packs are 16384 tokens; context/logprob remain 32768.** Job 6210750 OOMed on its first
  backward pass with 32768-token training packs; job 6211877 completed all 60 steps after this change.
- ≤4 nodes gets **interactive QOS**, which allocates in ~90 s. Above that, `QOS=normal` has queued
  for over 2 h. This is the single biggest lever on iteration speed.

Gotchas:

- **A newline in `EXTRA_OVERRIDES` truncates the run silently.** `ray.sub` writes `$COMMAND` to
  `driver_command.sh` and runs it with `bash`, so an embedded newline does not continue the command —
  it *ends* it, and everything after becomes a second shell command that never reaches the
  entrypoint. Job 6211877 lost `grpo.val_period=20` this way and validated every 10 steps while its
  log claimed 20; the loss is visible only by diffing `driver_command.sh` against what was typed.
  `launch_experiment.sh` now folds newlines to spaces and refuses to submit a command containing
  one. Prefer putting run shape in a recipe over passing it on the command line.
- **Pre-flight first** — `run_arc_preflight.sub` resolves each recipe and builds its datasets on
  CPU. Its guards each exist because a bug surfaced only at runtime and cost cluster time: ramp
  window vs prompts-per-step, ramp vs `shuffle`, `lr` vs `min_lr`, checkpoint metric vs the metrics
  the ARC env emits, submitted node count vs the recipe's `cluster.num_nodes`, and everything
  `SynthCurriculumConfig` validates (removed keys, seed separation, window capacity). It exits
  non-zero rather than printing warnings and returning 0.
- **Never `CMP=1` with an ARC env** — it forces `max_total_sequence_length=16384`, below the
  worst-case ARC prompt, and an overlong prompt is *masked*, not raised. The launcher refuses it.
- **`--no-container-mount-home`** for anything run in the container by hand, or enroot mounts your
  home over the container's and hides `uv`.
- Validation chats land in `logs/exp_NNN/val_data_step<N>.jsonl`; both re-scorers work offline from
  them, and the synthetic one regenerates targets by rebuilding the run's `SynthCurriculumConfig`
  from its recipe (`--config <recipe>`), through the same code path the dataset used.
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
5. **DONE — ramp transform complexity jointly with size.** Measured level rank × grid size is mixed
   within every optimizer window and reweighted easy → hard across the run.
6. **PARTIAL — add the missing axes.** Few-shot count is wired end to end and recorded in task IDs.
   Palette size, object count, density, composition depth and distractors were shipped, found to be
   inert or contradictory on most levels, and **removed** — palette size crashed four levels at its
   own documented maximum; distractors were zeroed on four of six; density meant "fraction painted"
   in one sampler and "keep this cell" in another. Two of them need a definition before they need
   code: what an *object* is when a rectangle can hold several colors, and what a *distractor* is on
   a level where the specks are the signal. Reintroduce them through item 7's DSL, one at a time,
   each with a test that shows it moves difficulty. Details in `ARC_CLEANUP_PLAN.md` §1, §8.
7. **Make the generator genuinely ARC-like.** The ladder is six hand-written families over a fixed
   vocabulary, and a policy can learn that vocabulary rather than the skill. Sample rules from a small
   DSL instead of enumerating them, and add the classes real ARC leans on: object-centric rules (move,
   count, sort, select the largest / odd-one-out), relational and conditional rules, and mappings that
   are not cellwise. This is also the direct test of open question (2) below.
8. **Frontier-weighted sampling.** Reweight toward levels whose solve rate is strictly between 0 and 1
   — the band where groups carry gradient. Needs an environment → generator feedback path across Ray
   actors; the per-level metrics it depends on now exist. Do it after (5), so there is a baseline.

**Reward and objective**

9. **DONE — make the echo strictly worse than a genuine attempt.** A non-answer
   `pred == test_input` now receives the unparseable floor; a true identity solution remains exact.
10. **Question whether exact match is the right objective.** It needs every cell right, so a rule the
    model understands still scores zero on a large grid. Small grids are the cheap test and are
    already supported. If solve rates stay at zero there, the objective — not the curriculum — is what
    to change, and annealing `w_cell` toward 0 as grid match rises becomes the live question.

## 9. Open questions

1. **Eval-set size.** 172 rows is small; step-to-step noise swamps real movement. Consider holding out
   a slice of the 1000 training tasks for online validation and reserving the official split for
   milestone reporting.
2. **Does synthetic transfer to real ARC at all?** Current answer: not with this generator/run.
   Job 6211877 improved synthetic exact 17 → 51/172 while real exact stayed 0/172. Item 7 (a broader
   object/relational DSL) is now the main curriculum question; `early_answer_4k_v3` first isolates
   the measured output-format/token-exhaustion failure.
3. **Does the edit-distance term earn its weight?** It shipped with the copy-relative re-baselining
   and has never been isolated. Cheap to ablate (`edit_weight: 0.0`) once `grid_match` is nonzero.
