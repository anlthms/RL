# ARC-AGI proposer-executor investigation closeout

## Decision

The current free-form natural-language executor is not reliable enough to
support a proposer-executor ARC loop with Nemotron Nano. We therefore stop
this investigation after the completed September 2026 diagnostics. The
benchmark, rollout capture, and general training fixes remain useful, but the
failed SFT checkpoints and executor-RL continuations are not recommended as
initializers for further work.

This result is narrower than saying that modular ARC systems cannot work. It
applies to the tested combination of Nano, prose transformation descriptions,
autoregressive SFT, and outcome-reward RL. Reopening the direction should
require a materially different interface or selection mechanism, rather than
another learning-rate, reward-weight, or epoch-count sweep.

The Kimi K3 oracle-rule ceiling experiment was deliberately deferred. No
Python executor was considered, by decision.

## Question and evaluation contract

The investigation asked whether a model could execute a supplied, correct ARC
transformation accurately enough for a separate proposer to use execution
feedback. All Nano comparisons used the same frozen 150-puzzle executor split,
kept disjoint from SFT collection. Checkpoints were compared by exact grid
match; format validity and cell accuracy were diagnostics, not substitutes for
exactness. Later evaluations allowed 16,384 output tokens.

The selected starting point for the final diagnostics was the exact-reward RL
checkpoint at global step 48, which solved 27 of 150 puzzles (18.0%). Paired
checkpoint comparisons were used where possible, because equal aggregate
scores can hide different solved cases.

## Evidence

| Test | Result | Interpretation |
| --- | --- | --- |
| Kimi K3 direct vs. proposer-executor, 32 fixed puzzles | 22/32 direct; 21/32 split | Decomposition did not improve the frontier-model result. |
| Initial Kimi-trace SFT | 20/150 base to 12/150 after either epoch | Lower validation loss accompanied worse execution. |
| Lower-LR, answer-weighted SFT | 24/150 base to 22/150, then 25/150 | The final difference was a statistical tie, not a durable gain. |
| Base vs. SFT-initialized RL | Base: 24 to 24 to 26; SFT: 25 to 18 to 13 | SFT harmed the subsequent RL trajectory. |
| Exact-reward continuation | 26/150 at step 32; 27/150 at step 48; 21/150 at step 64 | The small gain was transient. |
| Frozen curriculum control | 27/150 source; 31/150 at step 56; 20/150 at step 64 | The best observed score was unstable and not a stopping-point solution. |
| Dynamic mixed-outcome exact/DAPO | 27/150 source; 25/150 at step 56; 27/150 at step 64 | Dynamic sampling returned to the source score without improving it. |
| Nano sampling, 16 attempts per puzzle | 13.125% per rollout; 34.0% oracle-any; 20.667% modal consensus | Correct answers sometimes exist in the samples, but no usable target-time selector was found. |
| Broad gold-rationale SFT | 27/150 source to 15, 20, 16, then 4/150 | Teacher-forced learning strongly anti-correlated with exact execution. |

For the dynamic-sampling comparison at step 64, the source and trained model
both solved 27 puzzles. Each uniquely solved nine cases, giving an exact
McNemar p-value of 1.0. This is churn, not progress.

The 2,400-rollout pass-at-k experiment had 82.042% format validity, 9.92
unique valid predictions per puzzle on average, and only 22.417% mean agreement
with the modal prediction. Oracle-any pass@16 reached 34.0%, but modal-grid
consensus solved 20.667%. The gap demonstrates latent answer diversity, not a
deployable 34% executor: the oracle requires the hidden target to choose the
correct sample.

## Why the broad SFT result is decisive

The final SFT run used 45,568 unique training puzzles and 256 held-out
training-puzzle validation examples. It started from the 27/150 exact-RL
checkpoint and trained for one epoch (356 steps) with global batch size 128,
16K context, TP4/EP4, a learning rate of 5e-7, and 8x weight on the final
answer span. Answer tokens contributed 92.9% of the weighted loss mass. One
54,552-token outlier was excluded instead of truncated.

Validation loss improved monotonically through most of training:

```text
source   step 89   step 178   step 267   step 356
0.1116   0.0959    0.0883     0.0875     0.0880
27/150   15/150    20/150     16/150      4/150 exact
```

The final model's median reasoning length fell from 13,978 to 1,285
characters, so the run did teach shorter outputs. Inspection showed that the
short reasoning often restated the supplied generic `solution_steps` while
still placing output cells incorrectly. Concision and next-token fit therefore
did not imply instance-specific execution competence.

This explains why ordinary autoregressive SFT could not absorb the desired RL
objective merely by upweighting answers. Exact grid correctness is a
sequence-level, non-differentiable event. Answer weighting focuses gradient on
the output block, but token-level cross-entropy still rewards locally likely
cells and cannot directly express an all-cells-correct objective.

## Consequences for the multi-turn design

The loop assumes that executor feedback is evidence about the proposed rule.
At roughly 18--21% executor exactness, most failures are ambiguous: a correct
rule can be rejected because of execution error, and an incorrect rule can be
revised in response to a spurious output. More rounds then amplify noise rather
than provide dependable verification.

The original 95% executor gate was too ambitious as an immediate milestone,
but removing the gate did not remove the dependency. The measured ceiling is
far below the level needed to interpret executor failures reliably. The
earlier multi-turn gain in cell accuracy, with 0/172 exact solves, is
consistent with partial transformations improving while the verification
channel remains too noisy for exact ARC solutions.

No tested incremental intervention changed that conclusion:

- longer output budgets removed an artificial cap but did not create accuracy;
- lower learning rates and answer-weighted SFT did not produce a durable gain;
- shaped and exact-only RL produced small, unstable checkpoint variation;
- frozen and dynamic curricula did not improve the selected source;
- repeated sampling exposed latent solves, but simple consensus could not
  select them;
- broad gold-rationale SFT shortened reasoning while severely reducing exact
  execution.

## What remains useful

- The frozen, SFT-disjoint executor benchmark and paired checkpoint reports.
- Repeated-sampling reports with pass@k, oracle-any, consensus, validity, and
  prediction-diversity metrics.
- Complete proposer and executor rollout capture for later analysis or data
  filtering.
- The canonical ARC color mapping and shared prompt contracts.
- Credential-safe streamed-response handling in the teacher client.
- General SFT tagged-span weighting, rejected-sample masking, cycling async
  data, and Slurm resource overrides.
- Optional exact-only rollout-length shaping as an experimental primitive,
  not as a demonstrated executor-accuracy improvement.

The generated Kimi and gold-rationale data remain research artifacts. They
should not be treated as validated SFT datasets for this executor.

## Conditions for revisiting

Revisit the proposer-executor direction only with a new hypothesis that
changes the information interface or the ability to select executions. A
candidate must first show a durable, paired improvement on the same frozen
executor benchmark, survive a later checkpoint, and improve exact match rather
than only token loss or cell accuracy. Examples include a demonstrated learned
selector or a more structured neural representation; another prose-prompt or
scalar-reward sweep is not sufficient.

The Kimi K3 oracle-rule experiment remains a possible diagnostic if explicitly
reopened. It would distinguish a Nano-specific capacity problem from a more
general weakness of the prose executor contract, but it is not needed to close
the present Nano investigation.

## Reproducibility

The relevant experiment records are the `nvarc-kimi` ledger entries 108--119.
The durable tooling lives primarily in:

- `tools/nvarc_executor_benchmark_megatron.py`
- `tools/nvarc_executor_passk_report.py`
- `tools/nvarc_kimi_oracle.py`
- `tools/nvarc_teacher_collect.py`
- `examples/prompts/nvarc_executor.txt`
- `nemo_rl/environments/arc_agi/`

Large reports, datasets, checkpoints, and complete traces stay in the research
artifact directories rather than the source tree.
