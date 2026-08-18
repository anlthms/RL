# GRPO on ARC-AGI-2

An environment for [ARC-AGI-2](https://arcprize.org/) — grid-to-grid puzzles
where a rule must be inferred from a handful of worked examples and applied to a
new input — plus a generator that manufactures solvable ARC-style tasks of
tunable difficulty.

The two exist together because real ARC-AGI-2 is built so that frontier models
score near zero. For a small policy it is not a hard learning problem, it is a
null signal: exact match never leaves zero, so every gradient comes from shaping
and shaping only buys presentation. The synthetic ladder supplies tasks the
policy can actually solve, so exact match has somewhere to move from, while the
real evaluation split is scored at every checkpoint to measure whether any of it
transfers.

## Quick start

```bash
SUBMIT_ACCOUNT=<account> ENV=arcsynth_4n NUM_ACTOR_NODES=4 \
  TIMEOUT_MIN=240 RUN_TAG=<tag> bash launch_experiment.sh colocated
```

Validate the config on CPU first — it costs six minutes and catches the
misconfigurations that otherwise surface hours into a GPU run:

```bash
sbatch run_arc_preflight.sub
```

| Recipe | `ENV=` | Shape |
|---|---|---|
| `nanov3_arcsynth_4n_colocated.yaml` | `arcsynth_4n` | 4 nodes, interactive QOS (~90 s to allocate) |
| `nanov3_arcsynth_8n_colocated.yaml` | `arcsynth_8n` | 8 nodes, needs `QOS=normal` |
| `nanov3_arcsynth_colocated.yaml` | `arcsynth` | generic, no allocation-specific sizing |
| `nanov3_arcsynth_early_answer_colocated.yaml` | `arcsynth_early_answer` | early-answer prompt, shorter response cap |

Each allocation-specific recipe carries the batch, cadence and memory settings
it was sized for, and declares `cluster.num_nodes` so the preflight can check a
submission against it.

## Pieces

| File | Contents |
|---|---|
| `nemo_rl/environments/arc_agi_environment.py` | Ray actor, `ArcAgiEnvConfig`, per-level metrics |
| `nemo_rl/environments/arc_agi_grid.py` | serializer, parser, scorers, reward |
| `nemo_rl/environments/arc_agi_generators.py` | the synthetic ladder, guards, augmentation |
| `nemo_rl/data/datasets/response_datasets/arc_agi.py` | real corpus |
| `nemo_rl/data/datasets/response_datasets/arc_synth.py` | synthetic tasks, `SynthCurriculumConfig` |
| `nemo_rl/data/processors.py` | `arc_agi_data_processor`, serves both |
| `examples/prompts/arc_agi{,_early_answer}.txt` | prompts |
| `examples/configs/async/env_arc_synth{,_early_answer}.yaml` | model-agnostic env layers |

`arc_agi_grid.py` and `arc_agi_generators.py` are Ray-, torch- and GPU-free, so
they run on a login node.

## Data

Real ARC-AGI-2 ships challenges and solutions as separate JSON files per split;
point `data_path` at the directory holding
`arc-agi_<split>_{challenges,solutions}.json`. The `test` split has no solutions
and is unusable. One row per *test pair*, carrying that task's train pairs as
few-shot context.

Synthetic tasks are generated, not downloaded: each is a pure function of its
`SynthCurriculumConfig` and row index, so a run is reproducible without storing
a dataset and validation targets can be regenerated for offline scoring.

## The prompt

Grid cells are **space-delimited**, one row per line. Without a separator the
tokenizer merges a row into arbitrary multi-cell chunks and cell boundaries —
what every ARC transformation operates on — become invisible.

The prompt names the colors (0–9 are a palette, not magnitudes), asks the model
to describe the transformation before answering, and puts the `<answer>` tags
inside the final step; stranded in a trailing paragraph the model emits a
markdown code block and no tags. Delimiters are plain text, since added
vocabulary needs embeddings trained from scratch and GRPO's signal is far too
sparse for that. The description is never scored — only the grid is.

Measure prompt lengths at the real tokenizer before changing anything:

```bash
uv run tools/arc_agi_prompt_stats.py          # real corpus
uv run tools/arc_agi_prompt_stats.py --synth  # the ladder
```

## Reward

```
reward = 1.00 * exact_match            # prediction == target
       + 0.20 * gain(cell_accuracy)    # vs. echoing the test input
       + 0.10 * gain(edit_similarity)  # vs. echoing the test input
       + 0.05 * color_recall  - 0.05 * extraneous_colors
       - 0.05 * shape_mismatch + 0.05 * format_valid
```

Weights live on `ArcAgiEnvConfig` and every term is logged separately; whether
reward growth is exact match or shaping is the primary diagnostic.

The dense terms exist because GRPO computes advantage within a group of rollouts
on one prompt — if all score the same there is no gradient, and a small policy
solves ~nothing on real ARC. They make two wrong answers distinguishable.

Two details are load-bearing:

- **Similarity is paid as gain over echoing the input.** Scored absolutely, a
  copy of the test input earns ~0.61 cell accuracy on the evaluation split —
  more than a small policy earns by reasoning — and training converges on the
  echo.
- **A non-answer echo scores the reward floor.** Gain-relative scoring moved an
  echo's similarity reward to zero, but it still collected color and format
  credit while dodging every penalty, which beat a genuine imperfect attempt. A
  true identity solution still scores exact and is exempt.

`copied_input` is logged as the hack detector: a run whose cell accuracy merely
tracks the copy baseline is hacked, not learning. Never reward CoT length —
paying for tokens buys tokens.

## The synthetic ladder

| Level | Transformation |
|---|---|
| 0 | identity — a plumbing gate, *not* a rung |
| 1 | dihedral-8 |
| 2 | single-color ops — drop / recolor / keep-only |
| 3 | geometric — tile, crop to bbox, scale, add border |
| 4 | structure — denoise, complete symmetry, fill enclosed |
| 5 | compositions — a color op then a shape op |

Level 0 stays out of the training mixture: echoing scores full marks there,
which hands the policy a dominant degenerate strategy. Use
`data.train.levels=[0]` to check the data path end to end.

Difficulty is **mixed within every batch and reweighted across the run**. Levels
are ranked by measured solve rate rather than authored index and crossed with
grid size; every optimizer window contains the full cross product while the mass
shifts easy to hard. A phase where every task shares a difficulty gives groups
that are uniformly hopeless or uniformly trivial, and those contribute no
gradient — the exact failure the curriculum exists to fix.

Two guards, both unit-tested, because an unsolvable task is indistinguishable
from a model that will not learn: *identifiability* (a rule's parameter must
appear in the train pairs meant to teach it) and *degeneracy* (no stage may
leave any pair unchanged).

Every curriculum field lives on `SynthCurriculumConfig`, which is the whole
regeneration boundary — two configs that compare equal generate identical tasks.
Removed keys raise rather than being silently ignored.

## Validation and metrics

Validation concatenates the synthetic held-out split with all real evaluation
rows, so `max_val_samples` must cover both — the loader does not shuffle and
stops after `max_val_samples // val_batch_size` batches, silently dropping the
tail otherwise.

The environment emits `grid_match` (the honest score), `cell_match`,
`copied_input`, `format_valid` and the per-term breakdown, **each also per
level** (`grid_match/level_3`, `grid_match/real`). An aggregate cannot tell
"solving the easiest level only" from "uniformly mediocre".

Validation chats land in `logs/exp_NNN/val_data_step<N>.jsonl`. Both re-scorers
work offline from them:

```bash
uv run tools/arc_agi_score_val_dumps.py logs/exp_NNN
uv run tools/arc_synth_score_val_dumps.py logs/exp_NNN --config <recipe>
```

The synthetic one rebuilds the run's held-out split from its recipe through the
same code path the dataset used, so it cannot drift.

## Gotchas

- **Preflight first.** `run_arc_preflight.sub` resolves each recipe and builds
  its datasets on CPU. Its guards each exist because a bug surfaced only at
  runtime and cost cluster time: ramp window vs prompts-per-step, ramp vs
  `shuffle`, `lr` vs `min_lr`, checkpoint metric vs the metrics the environment
  emits, `train_mb_tokens` vs the sequence length, and global batch vs
  prompts-per-step × generations.
- **Sequence packing needs `train_mb_tokens >= max_total_sequence_length`.**
  Every sequence goes in exactly one bin, so an undersized bin is not a tight
  fit but a crash — and a *late* one, since the curriculum has to ramp to a long
  enough prompt first.
- **Never `CMP=1` with an ARC env.** It forces `max_total_sequence_length=16384`,
  below the worst-case real validation prompt, and an overlong prompt is
  *masked*, not raised. The launcher refuses it.
- **A caught async error still exits `COMPLETED` with status 0.** Read the
  driver log, not `sacct`.
- Grid and generator tests run on a login node with
  `PYTHONPATH=. python3 -m pytest <copy outside tests/>`, since
  `tests/unit/conftest.py` imports Ray. Everything else: `run_arc_tests.sub`.

## Design notes

`ARC_AGI_2_ENV_PLAN.md` in the repo root is the design and operating manual:
what was measured, which choices exist because a run demonstrated something, and
the open questions.
