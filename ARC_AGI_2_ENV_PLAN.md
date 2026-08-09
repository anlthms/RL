# ARC-AGI-2 environment — plan

Add a NeMo-RL environment that runs GRPO on ARC-AGI-2 tasks. Prompts lay out the few-shot
input/output grids as text with plain-text delimiters; the prompt asks the model to first describe
the transformation it infers from the examples, then emit the answer grid. Nothing scores the
description — it is free-form by construction.

Model: **Qwen3-1.7B**. Context: **32768**. First pass stays deliberately simple; a synthetic
difficulty curriculum is deferred until we know we need it.

Branch: `async_arc`.

---

## 1. The question up front: does inducing CoT require SFT?

**No — GRPO alone can do it, and the CoT does not need to be human-readable.** But format is not
what will block this. The blocker is *reward variance*, and that distinction drives §4.

### Why GRPO alone suffices for the CoT itself

GRPO's gradient reinforces every token in a sampled sequence in proportion to that sequence's
group-relative advantage. It has no notion of "reasoning" versus "answer" — if rollouts that reason
before answering are more often correct, those tokens get upweighted. That is the entire mechanism.
It does not care whether the reasoning is English, pseudo-code, coordinate dumps, or apparent
gibberish, and no term in the objective would penalize any of those.

Three things make it work without a supervised stage:

1. **An explicit instruction to describe the transformation.** The prompt says: study the example
   pairs, state the rule you infer, then apply it to the test input. This is the cheapest possible
   CoT induction — it costs nothing, it gives the model a concrete thing to produce rather than a
   vague "think step by step," and an instruct-tuned Qwen3-1.7B already complies with instructions
   of this shape at high rate on turn one.
2. **A format-validity reward term**, small and separate from correctness, rewarding "your output
   contained exactly one parseable grid in the right place." Trivially achievable, so it produces
   non-degenerate groups from step 0 — before any task is ever solved.
3. **Structural extraction.** Stop string on the closing delimiter; the parser takes the *last*
   well-formed grid, so a description that mentions grid-like text mid-stream can't corrupt it.

### The real blocker: degenerate groups

GRPO computes advantage within a group of `num_generations_per_prompt` rollouts on the same prompt
(16 in our qwen config). If every rollout in a group gets the *same* reward, advantage is zero for
all of them and the group contributes no gradient.

ARC-AGI-2 is built so that frontier models score near zero. A 1.7B model's exact-match rate will
very likely be **0/16 on essentially every group**. Binary exact-match reward therefore yields
approximately zero training signal, and the run will look like a bug — flat reward, flat KL, nothing
moving — when it is actually a reward-design problem. §4's dense terms exist entirely to fix this:
they are what make two wrong answers distinguishable.

### When SFT would actually be warranted

Only after the dense reward demonstrably fails to bootstrap. Cheapest intervention that clears the
observed failure, in order:

| Symptom | Intervention |
|---|---|
| Model doesn't emit the delimiters | Format instruction + format reward — in from the start |
| Format fine, but zero reward variance | Dense reward (§4) |
| Still zero variance at full dense reward | Synthetic easy-task curriculum (deferred; §6) |
| Variance exists but no CoT emerges | Nothing. If accuracy improves without it, that's a fine outcome |
| Genuinely zero signal even on easy synthetic tasks | Expert iteration: rejection-sample our own solved rollouts, brief SFT, resume GRPO |

That last row needs no human CoT annotation and no external dataset.

---

## 2. Data

`/lustre/fs1/portfolios/nemotron/projects/nemotron_sw_pre/users/anthomas/ash/data/arc-prize-2025/`

| File | Use |
|---|---|
| `arc-agi_training_challenges.json` + `arc-agi_training_solutions.json` | **train.** 1000 tasks → 1076 test-pair rows |
| `arc-agi_evaluation_challenges.json` + `arc-agi_evaluation_solutions.json` | **validation.** 120 tasks → 172 test-pair rows |
| `arc-agi_test_challenges.json` | unusable — no solutions file (competition holdout) |
| `sample_submission.json` | unused |

Challenges are `{task_id: {"train": [{"input", "output"}, ...], "test": [{"input"}, ...]}}`; the
solutions file maps `task_id -> [grid, ...]` positionally aligned with `test` (verified: alignment
holds for every task in both splits). One dataset row per *test pair*, carrying that task's train
pairs as few-shot context. Grids are ≤30×30, symbols 0–9.

---

## 3. Prompt construction

### Layout

```
You are given example input/output grid pairs that all follow one transformation
rule. Study them, describe the rule you infer, then apply it to the test input.
Put your final answer grid between <answer> and </answer>.

<example>
<input>
003
033
300
</input>
<output>
006
066
600
</output>
</example>
... (remaining train pairs)
<test_input>
...
</test_input>
```

The model is expected to respond with free-form reasoning followed by
`<answer>\n<grid>\n</answer>`. Only the grid is scored.

**Plain-text delimiters, not new vocabulary entries.** Adding real tokens means resizing the
embedding matrix and training randomly-initialized embeddings, and GRPO's signal is far too sparse
to learn them. ASCII tags are already well-represented in pretraining and parse just as
unambiguously.

### Token budget — measured, not assumed

Counting one character per cell plus a newline per row, over the actual corpus:

| Split | median grid-chars/task | p90 | p99 | max |
|---|---|---|---|---|
| training | 1112 | 3524 | 7070 | 9300 |
| evaluation | 3346 | 6000 | 9300 | 9660 |

So the worst task in either split is under ~10k characters of grid content, and digits tokenize
close to 1:1. At **32768** context that leaves >20k tokens for the description plus answer even on
the largest evaluation tasks — comfortable, which is the main reason to take the context bump.
Confirm against the real tokenizer in Milestone 0 rather than trusting the character proxy.

**Watch out:** `launch_experiment.sh`'s `CMP=1` preset hard-sets
`policy.max_total_sequence_length=16384`, applied after the config layer. **Do not use `CMP=1` with
`ENV=arc`**, or pass the 32768 value through `EXTRA_OVERRIDES`, which is applied last and wins.

---

## 4. Reward design

Scored on the extracted final grid only; the reasoning is never inspected. Let `T` be the target
grid (`h_t × w_t`) and `P` the prediction (`h_p × w_p`).

```
reward = w_exact * exact_match                 # 1.0 iff P == T
       + w_cell  * overlay_cell_accuracy       # centered overlay, works at any shape
       + w_color * color_recall                # fraction of T's colors present in P
       - w_extra * extraneous_color_fraction   # colors in P that are absent from T
       - w_shape * shape_mismatch              # normalized |Δh| + |Δw|
       + w_fmt   * format_valid
```

**Centered-overlay cell accuracy — the key change from a naive scorer.** Rather than zeroing the
cell term whenever shapes disagree, center `P` over `T`:

```
row_offset = (h_t - h_p) // 2      # negative when P is larger than T
col_offset = (w_t - w_p) // 2
matches    = #{cells where the overlaid P and T agree}
overlay_cell_accuracy = matches / max(h_t * w_t, h_p * w_p)
```

Dividing by the **max** of the two areas rather than by `h_t * w_t` means an oversized prediction
that happens to cover the target is penalized for its excess, so the metric can't be gamed by
emitting a giant grid. A near-miss that is one row too short still scores most of its cells, which
is exactly the gradient a binary scorer throws away.

Starting weights: `w_exact=1.0`, `w_cell=0.20`, `w_color=0.05`, `w_extra=0.05`, `w_shape=0.05`,
`w_fmt=0.05`. All configurable; every term logged separately.

Rationale and hazards:

- **The dense terms exist to break degenerate groups.** Their absolute scale barely matters (GRPO
  normalizes within-group); their *presence* is what turns an all-zero group into one with gradient.
- **Cell accuracy is still partially hackable** — copying the input grid, or flooding the modal
  color, beats chance on many tasks. `w_exact` dominating is the primary guard. Log a
  *copy-the-input baseline* reward per task and treat a run whose cell accuracy merely tracks that
  baseline as hacked, not learning.
- **The color terms are cheap shape-independent signal.** Recall rewards discovering the right
  palette; the extraneous-color penalty discourages spraying all ten symbols to farm partial credit
  (which would otherwise be a real exploit of `color_recall`).
- **Do not reward CoT length.** If longer reasoning helps, GRPO will find it; paying for tokens
  directly buys tokens, not reasoning.
- **Log every term separately.** Whether reward growth is exact-match or shaping is the single most
  important diagnostic in this plan.

Parser contract: last well-formed grid between `<answer>`/`</answer>`; reject ragged rows,
out-of-range symbols, empty grids, oversize (>30) dimensions — all → `format_valid=0`, reward 0.
An over-permissive parser inflates reward and an over-strict one starves the run; both look like
"the model isn't learning," so every rejection path gets a unit test.

### Validation metrics

On `arc-agi_evaluation_challenges.json`, report both, per the request:

- **grid match %** — exact-match rate over the 172 evaluation test-pair rows (the honest ARC score).
- **cell match %** — mean centered-overlay cell accuracy, which will move long before grid match
  does and is the metric to watch early.

---

## 5. Where the code goes

Everything mirrors the math env, the nearest internal analog.

| File | Change |
|---|---|
| `nemo_rl/environments/arc_agi_environment.py` | **new.** `ArcAgiEnvironment(EnvironmentInterface)` as a `@ray.remote` actor with `# pragma: no cover`, mirroring `math_environment.py`. `step()` parses each rollout's final grid, scores it, returns `EnvironmentReturn` with `terminateds=True` (single-turn). Config class `ArcAgiEnvConfig` is a **`pydantic.BaseModel`** with the reward weights as typed fields carrying the defaults — new user-facing config is v2, not `TypedDict`, and weights must not be `.get(k, default)`-ed at the call site. |
| `nemo_rl/environments/utils.py` | add `"arc_agi"` to `ENV_REGISTRY` with `actor_class_fqn` + `default_processor`. |
| `nemo_rl/distributed/ray_actor_environment_registry.py` | map the FQN to `PY_EXECUTABLES.SYSTEM` — scoring is pure Python. |
| `nemo_rl/data/datasets/response_datasets/arc_agi.py` | **new.** Joins challenges + solutions from the lustre path, emits one row per test pair. |
| `nemo_rl/data/processors.py` | **new** `arc_agi_data_processor` + `PROCESSOR_REGISTRY` entry. Serializes the few-shot layout, puts the target grid in `extra_env_info`, sets `loss_multiplier=0.0` on prompt overflow (same pattern as `math_hf_data_processor`). |
| `examples/configs/async/env_arc.yaml` | **new.** `data`/`env` layer with `_override_: true`, parallel to `env_math.yaml`; sets the train/val paths, `max_input_seq_length`, and `policy.max_total_sequence_length: 32768`. |
| `examples/configs/async/qwen3_1p7b_arc_{colocated,non_colocated}.yaml` | **new.** `defaults: [qwen3_1p7b_<topology>.yaml, env_arc.yaml]`. |
| `launch_experiment.sh` | add `arc` to the `ENV` axis → `examples/run_grpo.py` (no NeMo-Gym server). |
| `tests/unit/environments/test_arc_agi_environment.py` | **new** — see §7. |

---

## 6. First pass, then iterate

Keep the first pass simple: real ARC-AGI-2 training tasks, one reward function, single-turn, no
curriculum, no augmentation.

**Milestone 0 — offline, no GPUs.** Serialize the corpus; histogram prompt lengths at the real
tokenizer; round-trip the parser and scorer over hand-written and adversarial model-style outputs.
*Go:* p99 prompt comfortably under 32768 with room for a response; parser/scorer tests green.

**Milestone 1 — reward sanity, short run.** GRPO on the training split, tens of steps. Purpose is
plumbing and reward-variance measurement, not accuracy.
*Go:* non-zero within-group reward std on a clear majority of groups, format validity climbing
toward ~1.0. If reward std is ~0 here, stop and fix the reward — nothing downstream will work.

**Milestone 2 — does anything learn?** Full batch (`num_prompts_per_step: 16`,
`train_global_batch_size: 256` — the values both topologies now share), long enough to see a trend,
validating on the 120 evaluation tasks.
*Go:* cell match % on evaluation beats the step-0 baseline by more than the noise band. Bootstrap a
CI; with 172 rows the noise is large. Expect grid match % to stay at or near zero — that is not yet
failure.

**Milestone 3 — CoT check.** On the Milestone 2 run, compare mean response length and accuracy
conditioned on response length between step 0 and the final checkpoint. This is the direct answer to
"did GRPO alone induce reasoning." If accuracy rises without longer reasoning, that is a legitimate
result, not a failure.

**Deferred until needed:** synthetic curriculum of gradually increasing difficulty, augmentation
(color permutation, rotation/reflection), multi-turn retry, expert iteration.

Campaign mechanics per the auto-research skill: work on `async_arc`, one branch per hypothesis under
it, TSV ledger under `reports/auto_research/`, baseline first. Slurm launches go through
`launch_experiment.sh` (`SUBMIT_ACCOUNT=nemotron_sw_post`, `QOS=normal` for 8 nodes) and need
explicit approval each time.

---

## 7. Tests

- **Parser:** well-formed; ragged rows; out-of-range symbols; empty grid; multiple candidate grids
  (last wins); no delimiters; empty delimiters; dimensions >30.
- **Overlay scorer:** identical grids → 1.0; P smaller than T; P larger than T; odd/even size deltas
  (offset rounding); fully disjoint → 0.0; oversized-P penalty actually bites via the `max`-area
  denominator.
- **Reward terms:** each in isolation, then combined; color recall and the extraneous-color penalty;
  shape-mismatch magnitude; the copy-the-input and flood-all-colors hacks must both score below a
  genuine partial solve.
- **Serializer round-trip:** grid → text → grid is the identity for random grids up to 30×30.
- **`step()` contract:** batch in, `EnvironmentReturn` out with correct shapes, `terminateds` all
  true.
- Ray actor gets `# pragma: no cover`.

---

## 8. Progress log

### Milestone 0 — offline plumbing: **GO** (job 5984260, COMPLETED)

Implemented on `async_arc`:

- `nemo_rl/environments/arc_agi_grid.py` — serializer, parser, overlay scorer, prompt layout.
  Stdlib-only so it runs without Ray/torch/GPU.
- `nemo_rl/environments/arc_agi_environment.py` — `ArcAgiEnvironment` Ray actor + `ArcAgiEnvConfig`.
- `nemo_rl/data/datasets/response_datasets/arc_agi.py`, `arc_agi_data_processor`, prompt template
  `examples/prompts/arc_agi.txt`, all four registry entries.
- `examples/configs/async/env_arc.yaml` + `qwen3_1p7b_arc_{colocated,non_colocated}.yaml`;
  `ENV=arc` on the launcher.
- `tests/unit/environments/test_arc_agi_{grid,environment}.py`.

**Tests:** 39/39 pass in-container.

**Prompt lengths at the real Qwen3-1.7B tokenizer** (`tools/arc_agi_prompt_stats.py`):

| Split | rows | median | p90 | p99 | max | ≥32768 |
|---|---|---|---|---|---|---|
| training | 1076 | 1103 | 3055 | 6353 | 8570 | 0 |
| evaluation | 172 | 2385 | 4091 | 7807 | 8570 | 0 |

Go criterion met: nothing overflows, ~25k tokens left for the response at p99. The character-count
proxy from §3 overestimated by ~25% — digits tokenize better than 1:1 because runs of repeated
symbols merge.

**Two design bugs the tests caught before any GPU time:**

1. *The format term was inert.* A parseable-but-maximally-wrong answer scored exactly 0.0, identical
   to unparseable garbage, because the extraneous-color penalty (0.05) exactly cancelled the format
   bonus (0.05). Since at step 0 nothing is solved, the format gap is the *only* reward difference
   the policy can act on — a zero gap makes the whole bootstrap story fail silently. Fixed by
   defining the unparseable floor as `-(extraneous + shape)`: full penalties, no format credit, so
   any well-formed answer strictly beats garbage.
2. *`create_env` passes the raw YAML dict*, not a validated config object, so a typed `cfg` annotation
   would have been a lie and every weight read would have hit a plain dict. `ArcAgiEnvironment`
   now validates into `ArcAgiEnvConfig` in `__init__`, which is what keeps the field defaults the
   single source of truth.

Also added `stop_strings: ["</answer>"]` to the datum: reasoning before the grid is free-form and
unbounded by design, but there is nothing to gain after it, and 32k of context is a lot of rope.

### Milestone 1 — reward sanity: **GO** (job 5985126, 6 steps, 2 nodes)

Go criterion was non-zero within-group reward std. Met on every step:

| step | reward min | max | mean | **std** | mean tokens/sample |
|---|---|---|---|---|---|
| 1 | -0.100 | 0.283 | -0.037 | **0.109** | 1181 |
| 2 | -0.100 | 0.274 | -0.022 | **0.110** | 1273 |
| 3 | -0.100 | 0.293 | -0.045 | **0.093** | 1032 |
| 4 | -0.100 | 0.297 | -0.039 | **0.101** | 1047 |
| 5 | -0.100 | 0.297 | -0.038 | **0.089** | 970 |
| 6 | -0.100 | 0.266 | -0.020 | **0.100** | 887 |

Advantages: std ≈ 1.07–1.12, min ≈ -1.9, max ≈ +5.3. Non-degenerate — there is a gradient.

**Two prior attempts failed, both instructive:**

*Job 5984422 — no training step in 20 minutes.* `policy.generation.max_new_tokens` inherited the
32768 context, and Qwen3 is a thinking model, so rollouts ran to the cap and the async buffer never
filled. The context is sized for the *prompt*; the response needs its own, much smaller budget.
Fixed by setting `max_new_tokens: 4096` in the env layer.

*Job 5984932 — every rollout scored exactly the -0.10 floor, reward std 0.0.* This is precisely the
degenerate-group failure the whole reward design exists to prevent, arriving through a route the
plan did not anticipate: not difficulty, but format. Mean tokens/sample was 4099 — every response
hit the 4096 cap mid-thought, having spent the entire budget inside Qwen3's thinking block without
ever emitting an answer. Two fixes:

1. **The parser was too strict.** Requiring a closing `</answer>` scores a complete answer as
   garbage whenever generation stops on the delimiter without echoing it, or the response is cut at
   the token cap. Now falls back to the text after a final unclosed `<answer>`.
2. **Thinking mode off** (`enable_thinking: false`). We want the reasoning in the *scored* response
   anyway — the prompt asks the model to describe the transformation, and those are the tokens GRPO
   reinforces. A thinking block that gets stripped before scoring is reasoning we pay for and cannot
   reward. Mean tokens/sample fell 4099 → ~1000 and rewards immediately separated.

**Format validity is the live concern.** Mean reward ≈ -0.03 against a floor of -0.10 and a typical
parsed-but-wrong score of ~+0.15 implies only roughly a quarter of rollouts produce a parseable
grid. Non-degenerate is enough for Milestone 1, but if that fraction does not climb during
Milestone 2, the format term is doing all the work and the run is learning to format rather than to
solve.

**Metrics gap found and fixed.** `EnvironmentInterface.global_post_process_and_metrics` exists for
exactly this reporting, but **nothing in the codebase calls it** — so `grid_match`/`cell_match`
never reached wandb, and validation's `accuracy` was the mean *shaped* reward, which conflates
solving with getting close. `validate()` now averages any per-sample `terms` dict an environment
attaches to its metadata. Environments that don't populate it are unaffected.

### Milestone 2 — in flight (job 5985429)

8 nodes, 60 steps, 3 h wall clock, validation every 10 steps plus a step-0 baseline, on all 172
ARC-AGI-2 evaluation rows. Queued on `QOS=normal`. Will report `val:grid_match`, `val:cell_match`,
and `val:format_valid` against the step-0 baseline.

---

## 9. Open questions

1. **Eval-set size.** 120 tasks / 172 rows is small; step-to-step validation noise will swamp real
   movement. Consider holding out a slice of the 1000 training tasks as the online validation signal
   and reserving the official evaluation split for milestone reporting.
2. **Where partial credit stops helping.** Shaped reward is a bootstrap; there may be a point where
   it teaches near-miss behavior. Annealing `w_cell` toward 0 as grid match rises is the obvious
   knob, untested.
3. **Centering convention.** `(h_t - h_p) // 2` floors; for odd deltas the alternative alignment
   scores differently. Taking the max over both roundings is more forgiving but doubles scorer cost
   — cheap enough that it's worth measuring whether it matters.
4. **Multiple test inputs per task.** 1076 train rows come from 1000 tasks, so some tasks appear
   more than once with different test inputs and the same few-shot context. Fine for training;
   worth noting it makes rows within a task non-independent.
