# ARC-AGI-2 environment — plan

Add a NeMo-RL environment that runs GRPO on ARC-AGI-2 tasks. Prompts lay out the few-shot
input/output grids as text with plain-text delimiters; the prompt asks the model to first describe
the transformation it infers from the examples, then emit the answer grid. Nothing scores the
description — it is free-form by construction.

Model: **Qwen3-1.7B**. Context: **32768**. First pass stays deliberately simple; a synthetic
difficulty curriculum is deferred until we know we need it.

Branch: `async_arc`.

---

> **Status, 2026-08-11 — start at §13.** The environment, reward, prompt, and data path are built and
> have run four times end to end (§8, §11, §12). The model has learned the output contract and
> nothing else: `grid_match` is **0.0000 at every checkpoint of every run**, so no gradient has ever
> come from exact match. Two rounds of reward redesign moved presentation only. §13 is the current
> plan — a synthetic task generator that produces solvable tasks so exact-match signal exists at all.
> §1-§7 are the original design; §8-§12 are results, and where they contradict the design the results
> win — §3's prompt and §4's reward have both been superseded, as flagged in place. §14 is open.

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

> **Superseded by §11.** Grids are now space-delimited, the prompt carries a color legend and a
> five-step structure, and the `<answer>` tags live inside its final step. The layout below is the
> original. The token-budget reasoning still holds, at roughly double the numbers.

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

> **Superseded by §12.** The two similarity terms are now paid on their gain over echoing the test
> input, an edit-distance term was added, and the cell alignment is searched rather than centered.
> The rationale below — dense terms exist to break degenerate groups, exact match stays dominant —
> is unchanged and is still the reason the reward has the shape it does.

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

### A generation bug that inverted every earlier measurement

Validation reported `format_valid = 0` while training-step rewards plainly contained parseable
answers. Instrumenting `validate()` showed 43 samples but only 18 with scoring terms, and every
unscored sample sitting at `total_reward = 0.0000` — they never reached the environment at all.
The driver log had 299 copies of:

```
RuntimeError: The expanded size of the tensor (836) must match the existing size (838)
  megatron_worker.py:557 in _parse_result_to_batched_data_dict
```

mcore strips the matched stop string from `generated_tokens` but leaves its logprobs in
`generated_log_probs`. The padded logprob row is sized from the *token* count, so the longer
logprob slice raised a shape mismatch — 838 vs 836 is exactly the two tokens of `</answer>` — and
the rollout was dropped with reward 0.

**The failure is silently selective, which is what made it so misleading: only rollouts that
actually hit their stop string crash.** The samples surviving to be scored were precisely the ones
that failed to answer, so a working environment reported a 0% format-valid rate. It also explains
the earlier `accuracy` vs `reward` discrepancy — 24 scored samples at -0.1 plus 19 dropped zeros
averages to exactly the -0.0558 that looked impossible. The terms aggregation was correct
throughout.

Fixed by clipping logprobs to the kept tokens (`megatron_worker.py`). Logprobs must align 1:1 with
the tokens retained for training, so clipping is the correct direction, not padding. After the fix:
0 generation errors, 43/43 samples scored, `accuracy == reward`.

This bug affects **any** Megatron-generation run that uses stop strings, not just ARC.

### Milestone 2 — **accuracy is increasing** (job 5988364, in flight)

8 nodes, 60 steps, 3 h, validation at step 0 / every 10 steps / at end, on all 172 ARC-AGI-2
evaluation rows.

| step | grid_match | cell_match | format_valid | color_recall | shape_mismatch ↓ | accuracy |
|---|---|---|---|---|---|---|
| 0 | 0.0000 | 0.1925 | 0.4302 | 0.3126 | 0.6490 | 0.0118 |
| 10 | 0.0000 | 0.3748 | 0.8488 | 0.6121 | 0.2976 | 0.1194 |

Go criterion (cell match beats the step-0 baseline) met at the first checkpoint: 0.19 → 0.37.

**That run then collapsed at step 20** — every metric to zero, accuracy to the -0.10 floor. Cause
below.

### The collapse: off-policy logprob drift

`train/token_mult_prob_error` (generation-vs-training logprob mismatch) grew from 1.005 to **1e10**
starting at step 7, and generation length blew up 1363 → 3841 tokens (the cap) as the policy
collapsed into responses that never answer.

Two contributing causes, found in order:

1. **The stop-string logprob clip was misaligned.** The error stayed ~1.0 in earlier runs only
   because those samples were being *dropped* by the crash above; once they entered training it
   exploded. The tail-clip guessed the wrong end. Rather than keep guessing at mcore internals, the
   environment no longer sets stop strings at all — the parser already tolerates an unclosed
   `<answer>`, so the delimiter was only ever a latency optimization.
2. **In-flight weight updates, the dominant cause.** Removing stop strings alone still hit 12379 by
   step 10. A ~1300-token ARC response straddles a weight refit, so its generation logprobs come
   from weights the trainer no longer holds. Turning async off entirely is *not* available: the
   colocated Megatron path fails in `prepare_for_generation` with the weights offloaded (verified,
   job 5989038).

With `in_flight_weight_updates: false` and `max_trajectory_age_steps: 1` (now in `env_arc.yaml`):

| | before | after |
|---|---|---|
| `token_mult_prob_error` | 86 → 12379 | ≤ 4.5 |
| generation length | 1363 → 3841 | flat ~1300 |
| reward over 12 steps | climbed then collapsed | 0.00 → 0.19, holding |

### Milestone 2 rerun — **GO** (job 5989773, 8 nodes, 60 steps, COMPLETED in 46 min)

| step | grid_match | cell_match | format_valid | accuracy | response tokens |
|---|---|---|---|---|---|
| 0 | 0.0000 | 0.2308 | 0.4593 | 0.0244 | 2076 |
| 10 | 0.0000 | 0.3854 | 0.8430 | 0.1184 | 1464 |
| 20 | 0.0000 | 0.4000 | 0.9012 | 0.1330 | 1504 |
| 30 | 0.0000 | 0.4791 | 0.9244 | 0.1599 | 2283 |
| 40 | 0.0000 | 0.5227 | 0.9477 | 0.1756 | 2287 |
| 50 | 0.0000 | **0.5601** | 0.9767 | **0.1904** | 1484 |
| 60 | 0.0000 | 0.5118 | 0.9535 | 0.1713 | 1845 |

Accuracy rose 0.024 → 0.190 (peak, step 50), cell match 0.23 → 0.56. Training stayed stable
throughout: `token_mult_prob_error` 1.01 at every late step, generation length ~900–1400.

**The gains decoupled from formatting.** Format validity saturated around step 30 (0.92 → 0.98,
+6% relative from step 30 to 50) while cell match kept climbing (0.48 → 0.56, +17% relative). Early
gains were the model learning the output contract; later ones were not. This is the §4 question
about when shaping stops helping, and the answer here is that genuine grid accuracy kept improving
well after the format term was exhausted — no `w_cell` annealing was needed within 60 steps.

The step-60 dip (0.190 → 0.171) is within what 172 evaluation rows can resolve; it is not evidence
of a trend either way.

**`grid_match` is 0.0000 at every checkpoint.** This is the honest headline: **nothing was solved
outright.** Expected for ARC-AGI-2 at 1.7B — cell match 0.56 means predictions are a bit better
than half-right on a centered overlay, which is progress on the shaped objective, not on ARC.

### Milestone 3 — CoT emergence: **no lengthening, accuracy rose anyway**

| | step 0 / first 10 | final / last 10 |
|---|---|---|
| validation response tokens | 2076 | 1845 |
| training generation tokens | 1329 | 999 |
| policy entropy | 0.054 | 0.147 |

Responses got **shorter**, not longer, while accuracy improved 7.8×. GRPO did not induce longer
reasoning here; it made the existing response more useful — the early drop (2076 → 1464) is the
model abandoning rambling that never reached an answer, and the later variation is noise around a
flat trend. Rising entropy (0.054 → 0.147) says the policy did not collapse to a single template
either.

Per the plan's own criterion this is a legitimate outcome rather than a failure: CoT was a means to
accuracy, not the goal. The prompt-level instruction to describe the transformation, plus the
format term, was sufficient to get parseable structured answers out of an instruct model **with no
SFT of any kind** — which answers the original question. What it did *not* do is produce visibly
longer deliberation, so nothing here demonstrates that ARC-style reasoning emerged.

---

## 9. Follow-up: is the async machinery broken? (No.)

An earlier draft of this document claimed in-flight weight updates were the culprit and implied the
async mechanism was at fault. **That framing was wrong**, and the controls below show why.

### Does the stop-string logprob desync affect nanov3 + gym?

**No.** The NeMo-Gym path force-sets `generation_config["stop_strings"] = None`
(`nemo_gym.py:609`) and then asserts they are unset (`rollouts.py:2152`). `grpo_nanov3.yaml`
ships `stop_strings: null`, the math environment returns `next_stop_strings = [None]`, and no
non-ARC data processor sets datum-level stop strings. Nothing outside ARC reaches mcore's
stop-word trimming, so the token/logprob length desync cannot trigger there. ARC was the first
workload on this branch to use stop strings with Megatron generation.

### Why did the importance ratio not save us?

The mechanism is exactly as designed: `generation_logprobs` are the behavior logprobs recorded at
rollout time, and the actor weight is `exp(prev_logprobs - generation_logprobs)` per token
(`loss_functions.py`). The denominator is right. The problem is that with
`truncated_importance_sampling_type: null` the weight is **unbounded**, and it is unbiased only in
expectation — its *variance* explodes once the policy moves far from the behavior policy.

`token_mult_prob_error` is `mean(exp|generation_logprobs - prev_logprobs|)`, i.e. a direct measure
of how far the policy has moved since those tokens were sampled. It is not a bug detector.

**ARC is an unusually fast-moving policy.** Format validity goes 0.43 → 0.85 in ten steps — the
output distribution is being rewritten. With `max_trajectory_age_steps: 4`, trajectories in the
buffer were sampled by a policy up to four steps back, i.e. from the middle of that rewrite. The
error explodes at exactly steps 7–10, which is exactly when reward first moves
(0.018 → 0.036 → 0.050 → 0.195). That is genuine off-policyness, not corruption.

Controls confirming the machinery is healthy elsewhere:

| workload | in-flight / age | max `token_mult_prob_error` | outcome |
|---|---|---|---|
| nanov3 + gym (async colo & non-colo) | true / 4 | **1.02 – 1.04** | rewards 0.33–0.64 |
| qwen3-1.7B + math (19–41 steps) | true / 4 | **1.0 – 1.1** | rewards 0.40–0.80 |
| qwen3-1.7B + ARC | true / 4 | **1e5 – 1e15** | collapse by step 20 |
| qwen3-1.7B + ARC | false / 1 | 170 (60 steps) | stable, accuracy 0.024 → 0.190 (superseded — see below) |

Math and gym policies start already well-formed and move slowly, so their behavior logprobs stay
close to current. ARC's do not.

Two hypotheses tested and **rejected**:

- *Truncated importance sampling fixes it* (job 6014543, `tis`, ratio 5 / min 0.2, async defaults
  restored): reward peaked 0.20 at step 11 then decayed to -0.03 by step 20, with the error still
  reaching 1e15. Clamping bounds the weight but cannot recover signal from tokens whose behavior
  policy is four steps of rapid drift away.
- *Chunked prefill misaligns logprobs for ARC's long prompts* (job 6014907,
  `enable_chunked_prefill: false`): still exploded at step 9. Prompt length is not the mechanism.

**The ratio stays unbiased across a refit.** An earlier draft claimed a straddling sequence has "no
behavior policy" and that IS therefore cannot correct it. That is wrong. The ratio is per *token*,
and the recorded `generation_logprob` for a post-refit token is the probability of the distribution
that actually sampled it — new weights over a stale KV cache. Odd proposal distribution, still a
valid one. The pathology is variance, not bias, so the fix belongs at the drift, not at the
collector.

### Controlling the drift instead (async settings restored)

Three controls, each with `in_flight_weight_updates: true` and `max_trajectory_age_steps: 4`
restored, 2 nodes, 40 steps:

| control | max `token_mult_prob_error` | outcome |
|---|---|---|
| **`lr` 5e-6 → 2e-6** | 3.8e4 (steady ~1.0) | reward 0.15–0.19 sustained |
| `ratio_clip` 0.2 → 0.1 | 2.4e13 | reward swings 0.03–0.26 |
| `seq_logprob_error_threshold: 1.5` | ~1.1 after masking | masks **238 of 256** sequences/step by step 20 |

The clip bounds the surrogate ratio but not how far the weights move; the learning rate bounds the
drift itself. Sequence masking only "works" by discarding 93% of the batch.

**Confirmation** (job 6029953, 8 nodes, 60 steps, async at defaults + `lr` 2e-6):

| step | 0 | 20 | 30 | 40 | 60 |
|---|---|---|---|---|---|
| accuracy | 0.0296 | 0.1129 | 0.0513 | 0.1857 | 0.1819 |
| cell_match | 0.2176 | 0.3930 | 0.2914 | 0.5544 | 0.5559 |
| format_valid | 0.5116 | 0.7907 | 0.5174 | 0.9709 | 0.9419 |

Matches the throttled-async result (0.190 / 0.560 / 0.977) while the collector keeps running four
steps ahead — **27 minutes versus 46**. The step-30 dip recovers fully by step 40. `lr: 2.0e-6` is
now in `env_arc.yaml`; the async overrides were reverted.

**Conclusion.** Nothing is wrong with the async machinery, and existing recipes need no change.
ARC needed a smaller step size because its early training rewrites the output distribution; the
right lever was the learning rate, not the collector.

## 10. How the CoT actually evolves (job 6015119)

Sampled validation responses at steps 0 / 10 / 20 / 30 (`logger.num_val_samples_to_print`).
Accuracy on this 2-node rerun: -0.044 → 0.100 → 0.135 → 0.132; format 0.28 → 0.77 → 0.91 → 0.86;
response length 1933 → 1410 → 1229 → 1838 tokens.

**Step 0 — a confident essay about the wrong rule, and often no answer.**

> ### Rule Inference
> After analyzing the three example input/output pairs, the transformation rule can be inferred as:
> **Each row is transformed by replacing every occurrence of the digit `2` with `2` and every
> occurrence of the digit `4` with `8`.**
> ### Applying the Rule to the Test Input
> ```
> 4242442424424242242442
> …(echoes all 22 rows of the input verbatim)…
> ```
> After applying the transformation (replacing `4` with `8`): …(echoes 22 more rows)…

Note the rule is degenerate ("`2` stays as `2`"), and the response burns most of its budget
re-printing the input grid twice — which is why format validity is only 0.28: many responses never
reach `<answer>`.

**Step 10 — the essay collapses, the answer arrives.**

> After analyzing the examples, I infer that the transformation rule is:
> **Rule**: Each digit in the input grid is replaced with the same digit in the output grid.
> However, there is a subtle change in the placement of digits… it appears that the transformation
> is **not a simple digit-wise shift**, but rather a **rearrangement of digits**…
> To apply this rule to the test input, I will examine the input and output for patterns…
> `<answer>` …grid… 

The verbatim grid echo is gone and `<answer>` is emitted reliably (0.77). The prose is *vaguer* than
step 0 — hedged, non-committal — but it is short and it terminates. GRPO bought format compliance by
cutting the part of the response that wasn't paying for itself.

**Step 30 — reasoning grounded in the actual grid.**

> ### Example Analysis:
> 1. In the first example, the input grid has a single 1 in the fourth row and the output has this 1
>    in the fourth row. This suggests that the 1 remains in the same position.
> 2. In the second example, the input has multiple 1's in the same row and the output keeps these…
> ### Applying the Rule:
> In the test input, the 4th row has "000000000000300", and the output has "000000000044300"…

The analysis now cites concrete rows and cell values from *this* task rather than generic claims,
and the structure is stable (observations → rule → application → answer).

**What changed and what didn't.** Structure and grounding improved: input echo dropped, `<answer>`
reliably emitted, observations referencing real cells. The *inferred rules stayed wrong* — degenerate
substitutions at step 0, hedging at step 10, plausible-but-incorrect row claims at step 30. That is
exactly consistent with `grid_match = 0.0000` throughout: the model learned to produce a
well-formed, on-topic, grounded answer, not to solve ARC.

## 11. Grid-aware prompt (jobs 6040390, 6041201)

Four changes to how a task is presented, on
`autoresearch/2026-08-10-arc-prompt/grid-legend-and-verify`:

1. **Cells are space-delimited** (`0 1 2` per row, not `012`). Without a separator a row is one
   run of digits that the tokenizer merges into arbitrary multi-cell chunks, so cell boundaries --
   the thing every ARC transformation operates on -- are invisible to the model. `parse_grid` still
   accepts the compact form; rejecting a spaceless answer would discard reward over punctuation.
2. **The prompt names the colors** (`0 = black ... 9 = maroon`). 0-9 are a palette, not magnitudes.
3. **Describe the inputs and the outputs separately**, including what all the inputs share and what
   all the outputs share -- the commonality across examples is where the rule lives.
4. **Test the inferred rule against every example** and revise it if any disagrees.

Prompt length roughly doubles from the spacing: p99 12316 (training) / 14930 (evaluation), max
16593 at the real tokenizer. Nothing overflows 32768. `max_new_tokens` 4096 -> 8192, because step 4
means re-deriving several grids and running out of budget before `<answer>` is a failure we already
hit once (job 5984932) and misread as a reward bug.

**The `<answer>` tags have to live inside step 5.** The first smoke (6040390) put step-0 format
validity at 0.314, *below* the 0.512 the terser prompt started from. 95 of the 118 failures never
emitted `<answer>`: they produced a correct-looking grid in a markdown code block as the last item
of a five-part write-up, with the tag instruction stranded in a trailing paragraph. Folding the tags
and the shape constraints into step 5 itself moved step-0 format validity 0.314 -> 0.663 and step-0
accuracy -0.011 -> 0.092, with no other change.

### Result: much better zero-shot, same place after 60 steps

Job 6041201, 8 nodes, 60 steps, 40 min, against baseline 6029953:

| step | cell_match | format_valid | accuracy | baseline accuracy |
|---|---|---|---|---|
| 0 | **0.3589** | 0.6628 | **0.0920** | 0.0296 |
| 10 | 0.4457 | 0.7674 | 0.1296 | -- |
| 20 | 0.4928 | 0.8372 | 0.1529 | 0.1129 |
| 30 | 0.5170 | 0.8953 | 0.1675 | 0.0513 |
| 40 | 0.5173 | 0.9244 | 0.1724 | 0.1857 |
| 50 | 0.5309 | 0.9070 | 0.1735 | -- |
| 60 | 0.5251 | 0.9012 | 0.1703 | 0.1819 |

Step 0 is far stronger -- cell match 0.359 vs 0.218, +65% relative, before any training. Step 60 is
a hair below the baseline (0.525 vs 0.556), which is inside the ±0.05 step-to-step swing 172
evaluation rows produce. **The prompt buys a much better starting point and no higher ceiling.**
`grid_match` is 0.0000 at every checkpoint, as in every prior run.

### The finding that matters: the runs are learning to copy the input

Per §4's own criterion, scoring the copy-the-input baseline over all 172 evaluation rows:

```
copy-the-input cell_match = 0.6058
```

**That is higher than any cell match either run ever reached** (0.525 new, 0.556 baseline). And the
number of predictions *literally identical to the test input* grows monotonically with training:

| step | 0 | 30 | 60 |
|---|---|---|---|
| answers identical to the test input | 50/172 | 79/172 | **99/172** |

By step 60, 58% of validation answers are the test input echoed back. The reasoning degenerates to
match: step-60 rule descriptions are formulaic hedging ("The exact rule is not explicitly stated,
but it involves modifying the color patterns in a systematic way") wrapped around a copied grid.

This reframes Milestone 2 and every accuracy number in §8. The 7.8x accuracy gain was the model
learning (a) to emit a parseable grid and (b) to copy the input -- both of which the shaped reward
pays for, neither of which is ARC. §4 anticipated exactly this and named the test; the test had
simply never been run. It also explains why `grid_match` never moved and why cell match plateaus
around 0.52-0.56: the policy is climbing toward the copy baseline, not past it.

The prompt changes are not what caused this -- the baseline's 0.556 is below 0.6058 too -- but they
do not fix it either. **The reward, not the prompt, is the next thing to change.**

## 12. Copy-relative reward (job 6056499)

Two changes, on `autoresearch/2026-08-10-arc-prompt/copy-relative-reward`:

1. **Both similarity terms are paid on their gain over echoing the test input**, via
   `gain_over_baseline(score, baseline)` -> [-1, 1], zero at the baseline, each direction normalized
   by the room available in it so that solving an easy-baseline task pays the same 1.0 as solving a
   hard-baseline one. An echo is worth exactly zero on both terms. This is a re-baselining rather
   than a `pred == test_input` penalty: nothing is singled out, the zero point moves to where it
   belongs.
2. **An edit-distance term** (`edit_weight: 0.10`), `1 - levenshtein / max(len)` over the flattened
   cells with a row sentinel between rows. Its value is that it *disagrees* with the centered
   overlay: a prediction that is correct but shifted one row is nearly worthless to the overlay and
   costs edit distance one insertion. The reward does not have to be differentiable, so the two can
   simply be added.

The unparseable floor moves to `-(cell + edit + extraneous + shape)`, since the gain terms reach -1
and otherwise a badly wrong parseable answer would score below garbage, inverting the ordering the
format bootstrap depends on.

Vetted against job 6041201's 172 stored rollouts before spending GPU time -- the main risk of
relative scoring is flattening reward into degenerate groups, and it does not: step-0 reward std is
0.2243. A pure echo earns +0.084 rather than dominating; that residual is the format and
color-recall floor any parseable answer gets.

### It stopped the drift toward copying, and bought nothing else

| step | 0 | 20 | 40 | 60 |
|---|---|---|---|---|
| `copied_input`, **absolute** reward (6041201) | 0.291 | 0.366 | 0.488 | **0.576** |
| `copied_input`, **copy-relative** (6056499) | 0.291 | 0.343 | 0.384 | **0.401** |
| `cell_match`, absolute | 0.359 | 0.493 | 0.517 | 0.525 |
| `cell_match`, copy-relative | 0.418 | 0.489 | 0.542 | 0.515 |

The monotone climb into the echo is gone: `copied_input` plateaus around 0.35-0.41 instead of
rising through 0.58. Cell match is unchanged within noise, and `grid_match` is still 0.0000.

**The metric that actually settles it is the fraction of answers strictly better than an echo:**

| step | 0 | 20 | 40 | 60 |
|---|---|---|---|---|
| beats the echo, absolute reward | 0.076 | 0.076 | 0.064 | **0.041** |
| beats the echo, copy-relative | 0.093 | 0.076 | 0.105 | **0.099** |
| mean `cell_gain`, copy-relative | -0.315 | -0.190 | -0.137 | -0.165 |

Under the old reward this fraction **declined** with training -- the run was actively learning to
stop beating the echo. Under the new one it holds around 0.10, about 2.4x higher by step 60. So the
change fixed the pathology it was aimed at.

It did not, however, produce learning. The fraction is flat, not climbing, and mean `cell_gain`
stays negative at -0.165: **the average answer is still worse than simply echoing the input.** Nine
in ten answers fail to beat a strategy that requires no reasoning at all.

That is the state of things. Across four runs the model has learned the output contract (format
validity 0.43 -> 0.91), learned to ground its prose in the actual grid, and stopped drifting into
the echo -- and has solved zero ARC-AGI-2 tasks. Every gain so far has been in presentation. The
remaining levers are the ones §1 and §6 deferred: a synthetic curriculum of easy transformations
that a 1.7B model can actually solve, so that `grid_match` has somewhere to move from, and expert
iteration on whatever it does solve. Shaping the reward further looks exhausted -- two rounds of it
have moved presentation and nothing else.

### Alignment is now chosen, not assumed (code-only, no run yet)

`best_alignment_cell_accuracy` replaces the centered overlay as the alignment behind `cell_match`:
slide the smaller grid entirely inside the larger (valid-mode cross-correlation) and take the best
cell agreement. Colors are labels, not magnitudes, so the per-cell operator is equality — which is
exactly what a product summed over one-hot channels computes, so the 3D-conv and XNOR formulations
agree and neither needs the channels materialized.

It **replaces** the centered overlay's alignment rather than adding a term beside it: on 419 real
predictions from job 6056499 the two correlate at r = 0.978 (mean |diff| 0.034), so a separate
weighted term would have been `cell_match` wearing a second weight, not new signal. What it buys is
the end of an arbitrary convention — `(h_t - h_p) // 2` scores a 1x2 patch that exactly matches the
right end of a 1x3 target as **0.0**; valid mode scores it 2/3. Where shapes are equal (123 of 172
evaluation rows) there is one placement and the two agree by construction. Falls back to the
centered overlay for the mixed-dimension cases (one grid taller, the other wider, 22 of 419) where
valid mode has no placement. The max-area denominator is unchanged, and the copy baseline uses the
same function so the relative scoring stays consistent.

**This has not been trained with yet.** It landed after job 6056499 and is unmeasured.

---

## 13. Next: the synthetic generator ladder

This section is the working plan for the next phase. Everything above is history; this is what to
build.

### Why

Four runs, zero tasks solved. `grid_match` has been 0.0000 at every checkpoint of every run, so
there has never been a single bit of exact-match signal — every gradient the policy has ever
received came from shaping, and shaping has bought presentation (format validity 0.43 -> 0.91,
grounded prose, no more drift into the echo) and nothing else. Two rounds of reward redesign have
not changed that, and there is no obvious third round that is not just a fourth way to score
near-misses.

The missing ingredient is tasks the model can actually solve. ARC-AGI-2 is built so that frontier
models score near zero; a 1.7B model on it is not a learning problem, it is a null signal. A
generator gives us tasks of tunable difficulty, unlimited volume, and — critically — a regime where
`grid_match` is nonzero and can therefore be optimized directly rather than approximated.

### Difficulty is mixed within every batch, not ramped across steps

**Decision: every batch samples across all levels.** Not a scheduled ramp.

The reason is specific to GRPO rather than general curriculum lore. Advantage is computed within a
group of `num_generations_per_prompt` rollouts on the same prompt; if every rollout in a group gets
the same reward the advantage is zero and the group contributes no gradient. A ramp produces phases
where every task is at one difficulty, so groups are uniformly hopeless early and uniformly trivial
late — the degenerate-group failure §1 identified as the central risk, reintroduced by the
curriculum meant to fix it. Mixing guarantees that some level in every batch sits at a solve rate
strictly between 0 and 1.

If a frontier-weighted mixture is wanted later (reweight toward levels whose solve rate is neither 0
nor 1), it needs a feedback path from the environment back to the data generator across Ray actors.
Do not build that first. Mixed-uniform is the baseline; earn the complexity.

### The ladder

Every task is generated from `(seed, index)` so a run is reproducible without a static dataset. Each
task emits 2-4 few-shot pairs plus one test pair, matching real ARC's shape.

| level | transformation |
|---|---|
| 0 | **identity** — output = input. The sanity gate: if this is not learned, something is broken upstream of the task |
| 1 | **dihedral-8** — rot90 / rot180 / rot270 / flip-h / flip-v / transpose / anti-transpose |
| 2 | **single-color ops** — drop color X to background; recolor X -> Y; keep only X |
| 3 | **geometric** — tile k x m; crop to the bounding box of non-background; scale by integer k; add a border |
| 4 | **structure** — denoise (remove isolated cells, keep shapes); complete a symmetry; fill enclosed regions |
| 5 | **compositions** — two level-1..3 operations applied in sequence |

Look at ARC-AGI-1's training split for more level-2/3 ideas; its tasks are markedly simpler than
ARC-AGI-2's and several are close to one-liners in this vocabulary.

### Input grids need structure, not noise

This is the part that is easy to get wrong. Uniform random grids make several levels degenerate:
crop-to-bounding-box needs a bounding box, denoising needs signal to distinguish from noise, and
symmetry completion needs a symmetry. Generate inputs from a small library of *patterns* —
scattered points, filled and hollow rectangles, lines and crosses, symmetric blobs, repeated motifs,
a background with one or two foreground objects — with random sizes (3x3 up to ~20x20) and a random
color palette drawn from 1-9 over background 0.

### Two correctness properties the generator must enforce

Both of these produce tasks that are unsolvable in principle, which in a training run is
indistinguishable from the model failing to learn. Guard them explicitly and unit-test the guards.

1. **Identifiability.** The rule must be inferable from the few-shot pairs alone. "Drop color 3" is
   not identifiable unless color 3 appears in the examples; "recolor 3 -> 5" is not identifiable if
   the examples never contain a 3. Reject and resample any task whose rule is not pinned down by its
   own examples.
2. **Non-degeneracy.** Reject any non-identity task where output == input on every example — rot180
   of a symmetric grid, dropping a color that is absent, cropping a grid with no background border.
   These are level-0 tasks wearing a level-3 label, and they inflate the solve rate of whatever
   level they land in.

Also reject tasks where echoing the input already solves the test pair, except at level 0 where that
is the point.

### Augmentation

Free volume, and it blocks memorizing specific colors rather than the rule: permute the non-zero
colors 1-9 consistently across a whole task, and apply a whole-task dihedral transform (both input
and output of every pair). Both preserve the rule exactly, which is what makes them safe.

### Where the code goes

| File | Change |
|---|---|
| `nemo_rl/environments/arc_agi_generators.py` | **new.** Pure transformation and pattern-generation functions, numpy + stdlib only, mirroring `arc_agi_grid.py` so it is testable with no Ray/torch/GPU. Exports one `generate_task(rng, level) -> dict` plus the per-level primitives and the two rejection guards. |
| `nemo_rl/data/datasets/response_datasets/arc_synth.py` | **new.** Dataset that materializes tasks from `(seed, index)` across a configured level mixture. Emits the same row schema as `arc_agi.py` (`train_pairs`, `test_input`, `target`, `task_id`) plus `level`. |
| `nemo_rl/data/processors.py` | extend `arc_agi_data_processor` to carry `level` into `extra_env_info`; keep it one processor, the row schema is identical. |
| `nemo_rl/environments/arc_agi_environment.py` | add `level` to `ArcAgiEnvironmentMetadata`, and report **per-level** `grid_match` from `global_post_process_and_metrics`. Without the per-level split an aggregate number cannot distinguish "solving level 0, nothing else" from "uniformly mediocre", which is the whole question. |
| `examples/configs/async/env_arc_synth.yaml` | **new.** Parallel to `env_arc.yaml`; sets the generator seed, the level mixture, and both validation sources. |
| `examples/configs/async/qwen3_1p7b_arcsynth_{colocated,non_colocated}.yaml` | **new.** `defaults: [qwen3_1p7b_<topology>.yaml, env_arc_synth.yaml]`. |
| `launch_experiment.sh` | add `arcsynth` to the `ENV` axis -> `examples/run_grpo.py`. |
| `tests/unit/environments/test_arc_agi_generators.py` | **new** — see below. |

### Validation: both sources

- **Synthetic held-out** (fresh seed, same mixture) — shows whether the curriculum is being learned
  at all, per level. This is where `grid_match` should finally leave zero.
- **Real ARC-AGI-2 evaluation, all 172 rows** — shows whether any of it transfers, and keeps the
  headline metric comparable to all four prior runs.

### Milestones

**M4.0 — generator offline.** Tests green; eyeball a dump of generated tasks at every level with
`tools/arc_agi_prompt_stats.py --dump`; confirm prompt lengths still fit. *Go:* both rejection
guards demonstrably fire; no level emits a task solvable by echoing the input (except level 0).

**M4.1 — the identity gate.** Level 0 only, 2 nodes, ~20 steps. *Go:* `grid_match` > 0.9. This is a
plumbing test, not a research result: if a 1.7B model cannot learn to copy a grid it is shown, the
problem is in the data path, the prompt, or the parser, and no amount of curriculum will help.
**Do not proceed past a failure here.**

**M4.2 — levels 0-2 mixed**, 8 nodes, 60 steps. *Go:* per-level `grid_match` rising on levels 1 and
2, not only 0.

**M4.3 — full ladder mixed**, 8 nodes. Report per-level `grid_match` and the real ARC-AGI-2 eval.
The question this phase exists to answer is whether real-ARC cell match improves at all once the
model has genuinely solved *something*.

### Tests

- Each transformation is its own inverse or has a known fixed point: `rot180(rot180(g)) == g`,
  `transpose(transpose(g)) == g`, dihedral-8 forms a group of order 8 on a non-symmetric grid.
- Identifiability guard rejects a "drop color X" task whose examples never contain X.
- Non-degeneracy guard rejects rot180 of a symmetric grid and a crop with no background border.
- Echo guard: for every level above 0, the generated test target != test input.
- Color-permutation augmentation preserves the rule: applying the rule then permuting equals
  permuting then applying the rule.
- Level mixture: sampling N tasks yields every configured level, and generation is deterministic in
  `(seed, index)`.

### M4.0 — the generator, built (code only, no run yet)

Everything in the table above exists on `async_arc`. Six decisions the plan left open, and
what was chosen:

1. **The degeneracy guard is per pair and per stage**, not "every example is unchanged". A single
   identity pair inside a non-identity task is itself an ambiguity, and a task whose *test* pair is
   unchanged is solved by echoing the input — the behavior four runs collapsed into — so the plan's
   separate echo guard is subsumed rather than written twice. The per-*stage* half was not
   anticipated and is what a whole-rule check misses: `recolor(4->6)` then `flip_v` on a grid of
   identical rows changes the grid, passes a whole-rule check, and teaches a flip that no example
   demonstrates. Composition hides no-ops.
2. **Level 5 is always a color op followed by a shape op**, sharing one palette, with the shape stage
   supplying the input sampler. The other order is not identifiable: `drop_color(3)` then
   `recolor(3->5)` leaves the second stage with nothing to act on. Shape ops are color-agnostic, so
   this order can never do that, and the shape stage is the one with an opinion about size (tile and
   scale bound the input so the output still fits 30x30) and structure (crop needs a background
   border).
3. **Input patterns are paired with the rules that need them**, since the sampler is what makes a
   level satisfiable at all: crop gets the margin sampler, fill-enclosed the hollow-rectangle one,
   denoise rectangles-plus-specks, symmetry completion a mirrored grid with one side erased. Every
   sampler paints the whole palette, which is what makes identifiability hold by construction rather
   than by rejection — the guard is then a real check, not a formality.
4. **Per-level metrics ride on the per-sample `terms` dict.** Validation does not call
   `global_post_process_and_metrics` at all (§8) — it averages each per-sample term over the samples
   that reported it. So a key present only on one level's samples *is* that level's mean, with no
   plumbing: `step()` copies `grid_match` and `cell_match` to `grid_match/level_3` and the
   aggregation does the rest. `global_post_process_and_metrics` computes the same split explicitly
   for training batches.
5. **Real ARC rows now carry `level = -1`** and are reported as the `/real` bucket. Both validation
   sources go through one dataloader, which concatenates them, and `concatenate_datasets` requires
   identical features — so the column has to exist on both sides. The loader does not shuffle and
   stops after `max_val_samples // val_batch_size` batches, hence `max_val_samples: 344` (172
   synthetic held-out + all 172 real evaluation rows) at `val_batch_size: 43`. Getting this wrong
   silently drops the real split, which is the measurement the phase exists for.
6. **`CMP=1` with any ARC env is now a hard error in the launcher** rather than a comment. An
   overlong prompt is not an error — the processor masks the row with `loss_multiplier=0` — so the
   combination degrades a run silently, and it has been a footnote in this document twice.

The rule name is folded into `task_id` (`synth_L3_tile(2x2)_0_41`) instead of getting its own
dataset column, so a dumped validation row says *which* transformation it was — the difference
between "level 3 is hard" and "tile is hard" — without a second schema for the real corpus to match.

Prompt lengths, generated task bodies at 500 tasks per level: median ~2.1k characters, worst 8.8k,
against 16.6k tokens for the worst real ARC row. Nothing is close to the context, and the real
evaluation split is still the binding constraint on `max_total_sequence_length`.

**Tests: 127 pass in-container** (job 6069244), across all four ARC files. `.run_arc_tests.sub` is
the runner. Two things it caught that a login-node run could not, both in the new test fixtures
rather than the code, and both worth knowing before writing the next environment test:

- `global_post_process_and_metrics` reads *every* reward term by name, so a hand-written partial
  `terms` dict raises `KeyError` and tests the test. Build terms with `score_response`.
- `batch["text"]` is a `(b, s)` tensor of token ids, not a list of strings —
  `calculate_pass_rate_per_prompt` groups rollouts by prompt with `torch.unique(dim=0)`.

Also verified out-of-container: 8000 synthetic rows materialize in 7.6 s, and a held-out synthetic
split concatenates with the 172 real evaluation rows into one 344-row validation set with matching
features and levels `[-1, 0, 1, 2, 3, 4, 5]`.

**Running the tests in the container needs `--no-container-mount-home`.** Without it enroot mounts
the caller's host home over the container's, which hides the image's `uv` and produces a
`uv: command not found` that reads as a broken image — it is not. `UV_PYTHON` must stay unset
(`launch_experiment.sh:76`): Ray compares cluster and worker interpreter versions exactly.
`.run_arc_tests.sub` runs on the `cpu` partition rather than `batch`/`interactive`, because the
interactive QOS caps a user at 4 nodes and a training run wants all of them — tests should never
queue behind, or displace, an experiment.

Not yet run. M4.1 (the identity gate, `levels: [0]`, 2 nodes, ~20 steps, go at `grid_match > 0.9`)
is the next step and must pass before anything else: if a 1.7B model cannot learn to copy a grid it
is shown, the problem is in the data path, the prompt, or the parser.

### Operational notes for whoever picks this up

- Branch `async_arc`, currently at `806ab83d`. Both prior experiment branches are preserved under
  `autoresearch/2026-08-10-arc-prompt/`. Ledger: `reports/auto_research/arc-prompt/experiments.tsv`.
- Launch: `SUBMIT_ACCOUNT=nemotron_sw_post QOS=normal MODEL=qwen3_1p7b ENV=arcsynth NUM_ACTOR_NODES=8
  MAX_STEPS=60 TIMEOUT_MIN=180 RUN_TAG=<tag> EXTRA_OVERRIDES="grpo.val_at_start=true
  grpo.val_period=10" bash launch_experiment.sh colocated`. Each launch needs explicit approval.
- **Never combine `CMP=1` with an ARC env** — it hard-sets `max_total_sequence_length=16384` after
  the config layer, and ARC prompts reach 16593 tokens.
- Runs write to `logs/exp_NNN/`, incrementing per run, and every validation chat lands in
  `val_data_step<N>.jsonl` there. `tools/arc_agi_score_val_dumps.py <log_dir>` re-scores a finished
  run offline, including the copy-the-input check.
- The grid and generator modules avoid Ray/torch/GPU, so their tests run on a login node with
  `PYTHONPATH=. python3 -m pytest <copy of the test file outside tests/>` — `tests/unit/conftest.py`
  imports Ray, so run the file from a scratch directory to skip it. Everything else needs the
  container.
- 8-node jobs have queued for 30-90 minutes; a 60-step ARC run takes ~40 minutes once started.

## 14. Open questions

1. **Eval-set size.** 120 tasks / 172 rows is small; step-to-step validation noise will swamp real
   movement. Consider holding out a slice of the 1000 training tasks as the online validation signal
   and reserving the official evaluation split for milestone reporting.
2. **Where partial credit stops helping.** Shaped reward is a bootstrap; there may be a point where
   it teaches near-miss behavior. Annealing `w_cell` toward 0 as grid match rises is the obvious
   knob, untested — and newly testable, because the generator is the first setting where grid match
   is nonzero and therefore has something to anneal against.
3. ~~**Centering convention.**~~ **Closed** by `best_alignment_cell_accuracy` (§12): the alignment
   is now searched in valid mode rather than assumed, so the `//2` rounding no longer decides
   near-misses. Untrained-with as of `806ab83d`.
4. **Multiple test inputs per task.** 1076 train rows come from 1000 tasks, so some tasks appear
   more than once with different test inputs and the same few-shot context. Fine for training;
   worth noting it makes rows within a task non-independent.
5. **Does the edit-distance term earn its weight?** It shipped together with the copy-relative
   re-baselining in job 6056499, so its individual contribution has never been isolated. Cheap to
   ablate (`edit_weight: 0.0`) once there is a setting where the result would be legible — which
   means after the generator produces nonzero `grid_match`, not on ARC-AGI-2.
6. **Does solving synthetic tasks transfer to real ARC at all?** The premise of §13 is that exact-match
   signal on solvable tasks teaches something that shaping cannot. That premise is untested and is
   the main risk of the whole phase: it is entirely possible the model learns the generator's
   vocabulary and transfers nothing. M4.3's real-ARC validation is the measurement, and a null
   result there is a real answer, not a failed run.
