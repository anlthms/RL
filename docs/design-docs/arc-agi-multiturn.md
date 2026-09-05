# Verifier-Guided Multi-Turn ARC-AGI

Revised 2026-09-04 to record the completed executor investigation and its
stopping decision. The sections below still describe the implemented system
(NeMo-Gym
`arc_transform_refinement_agent` + `arc_agi_2` resources server on the Gym
submodule, co-trained with single-turn executor rows through
`examples/nemo_gym/run_grpo_nemo_gym.py`). Sections marked *adopted
revision* record design changes made after measurement.

## Disposition

The current free-form natural-language executor is a no-go for a reliable
proposer-executor loop with Nemotron Nano. Its selected checkpoint solves
27/150 frozen executor cases, and neither SFT, continued RL, dynamic sampling,
nor repeated-sample consensus produced a durable improvement. The final broad
gold-rationale SFT run reduced exact accuracy from 27/150 to 4/150 while its
validation loss improved.

This document remains the design record for the implemented system, not a
recommendation to scale the current recipe. See the
[investigation closeout](arc-agi-multiturn-investigation.md) for the full
evidence and the conditions required to revisit the direction.

## Motivation

A single model response that infers a rule, mentally tests it, and emits the
test grid anchors on its own narration. This investigation tested replacing
the narrated self-check with an actual check: rules are executed by a separate
chat and compared against known outputs by a deterministic verifier, and
failures come back as behavioral evidence for revision.

The episode separates two skills, played by one shared policy:

1. **Rule induction (proposer):** describe the transformation represented
   by paired example grids, and revise it from execution feedback.
2. **Rule execution (executor):** apply a textual description to one grid
   without seeing the expected output.

The rule interface is the canonical 4-section schema used across the whole
stack — `<rules_summary>`, `<solution_steps>`, `<key_insight>`,
`<puzzle_concepts>` — matching the NVARC descriptions the executor is
trained on, so a co-trained policy proposes rules in exactly the format it
learned to execute.

## Goals

- Use exact execution on known pairs instead of an in-context claim that a
  rule was checked.
- Keep rule induction and rule execution in different chat histories; rule
  text is the only proposer-to-executor channel.
- Never reveal an expected output to the executor or a hidden test output
  to any model call.
- One executor contract everywhere: one rule, one grid, one fresh session,
  one `<answer>` block — byte-identical between native executor training
  rows, the executor benchmark, and executor calls inside episodes.
- Provide precise, deterministic feedback (input, executor output,
  expected output, `p-c` cell diff) when a predicted grid is wrong.
- Preserve a complete, inspectable episode trace even when only part of it
  is used for policy loss.

## Non-goals

- Generating or executing Python as the transform representation
  (natural-language rules only, by decision).
- Searching a beam of rules per episode; GRPO generations already provide
  independent rule trajectories per prompt. Multi-proposal filtering and
  test-time RL are deferred phases.
- Solving credit assignment across multiple disjoint chat histories.

## Episode protocols

The agent implements two protocols over the same machinery. A task row may
select its protocol per episode (`protocol` field), falling back to the
agent config — this is how one agent instance trains on `eval_sequence`
rows and validates on `hidden_test` rows in the same run.

### `eval_sequence` (NVARC co-training)

The proposer sees only demonstration pairs (never the evaluation inputs)
and induces a canonical rule. Fresh single-grid executor sessions apply it
to held-out evaluation grids one at a time: an exact solve advances to the
next grid with the same rule; a miss returns server-rendered behavioral
evidence for a revision. Reward is aggregated server-side over the grid
sequence with FINAL-RULE CREDIT (adopted 2026-08-27, implemented on gym
branch `anthomas/colo2`; activates with the next co-training lineage):
before finalizing, the agent re-applies the final rule once to every grid
whose last attempt used an older rule (including unreached grids), and
each grid is scored on its LAST attempt. The original best-attempt
aggregation let a degraded final revision inherit rewards that earlier
rules earned under advance-on-solve — credit misassignment on the only
trained turn, consistent with the revision-degradation observed in the
first two co-training runs (loop cell_match declining while training
reward improved). Unattempted grids still sit at the reward floor and
solving everything with the final rule earns the configured bonus.

### `hidden_test` (real ARC)

The proposer sees the puzzle's demo (train) pairs. The refinement loop
verifies the demo grids one at a time — their outputs are public at
inference, so this checks whether the rule text *operationalizes* what the
proposer saw. Whatever ends the loop — all demos verified, the context
budget, the round cap, or a parse failure after a previously valid rule —
the current rule answers every hidden test grid once, in fresh single-grid
sessions, with no feedback on the test. The server scores those final
answers (last attempt per grid; unanswered grids at the reward floor) and
reports `grid_match`/`cell_match` top-level; the demo loop surfaces as
`train_gate_pass`/`train_exact_fraction` diagnostics. A parse failure with
no prior valid rule terminates the episode (`agent_error`, masked).

```text
Persistent proposer chat                Fresh executor chat per grid
  user:      demo pairs + rule contract   user:      rule + one input grid
  assistant: 4-section rule v0            assistant: one <answer> grid
  user:      behavioral evidence for a
             missed grid (input, executor
             output, expected, p-c diff)
  assistant: 4-section rule v1
  ...
```

## Prompt contracts

- **Proposer:** color legend, demonstration pairs, and a request for an
  operational rule as exactly the four canonical sections. No Python, no
  predicted grids. The test input is *not* shown (adopted revision: the
  original design showed it for disambiguation; the implemented NVARC
  prompt shows demos only so the rule must generalize).
- **Executor:** the shared single-grid template — rule inside
  `<transformation>` tags, one input grid, strict `<answer>` contract
  (rows of space-separated ints, no JSON/prose/fences), no system message.
  This text is byte-identical to `examples/prompts/nvarc_executor.txt`. A
  format failure gets one format-only retry in the same session, worded
  identically to the benchmark's retry; a second failure terminates the
  episode as an executor failure, never as evidence against the rule.
- **Revision:** the resources server renders the complete evidence prompt
  (input, executor output, expected output, `p-c` diff) so hidden targets
  reach the proposer only through this deliberate channel; the agent never
  handles raw targets. Matching cells render as `2-2` to preserve
  alignment; shape mismatches render as a shape message without a cell
  diff.

## Reward

Every grid attempt is scored server-side with the shared gain-over-echo
terms: exact match dominates the sum of all shaping weights, dense
similarity terms are paid as gain over echoing the input (an echo sits on
the reward floor), and `copied_input` is reported as the hack detector.
`eval_sequence` aggregates over the evaluation-grid sequence as above;
`hidden_test` averages the final test answers. Test exactness dominates
shaping by construction of the weights.

## Context-window policy

Before each revision the agent requires

```text
current_proposer_tokens + next_feedback_tokens
  + reserved_proposer_output_tokens + chat_template_margin
  <= model_context_limit
```

using measured token counts from the last response and a bytes-as-tokens
upper bound for the feedback. `model_context_limit` is set so the
trainable final proposer turn fits the training pack (e.g. 12288).

Two measured caveats (adopted revisions):

- **The accounting leaks.** Client-side token counts under-measure the
  assembled history by more than the template margin in rare episodes
  (think-token accounting), so the RL side hard-clamps any sample that
  exceeds the sequence-packing bin — truncate the tail, zero the loss
  multiplier, keep the group slot (`clamp_overlong_samples_to_pack_bin`).
  Packing overflow can therefore no longer crash a run.
- **Do not throttle inference with the training limit.** Applying the
  training-pack `model_context_limit` to validation episodes (which are
  never trained, on an engine with far more context) reduced the real-ARC
  loop to ~1 revision round. Validation rows should carry a per-row
  context-limit override sized to the engine.

Termination reasons distinguish `train_verified`, `all_solved`,
`context_exhausted`, `executor_format_failure`,
`executor_context_exhausted`, `agent_error`, and `emergency_round_cap`.
The round cap is an emergency stop; the token budget governs normal
termination.

## Trainable trajectory contract

The NeMo-Gym postprocessor represents one contiguous token history per
rollout, and fresh executor chats are intentionally disjoint, so the agent
returns:

- **trainable response:** only the final proposer generation, with its
  complete proposer history as prompt tokens;
- **audit data:** every proposer call, executor call, verification, and
  state transition in the full result;
- **reward:** the server-aggregated episode reward associated with that
  final generation.

Final-turn-only training avoids positively reinforcing an early rule that
a later revision rescued. Executor generations are never proposer-loss
tokens; executor competence is trained by the separate single-turn
executor task (native rows through the same resources server), which is
also why both roles can share one GRPO run.

Loss masking (adopted revision, implemented 2026-08-26): masking is for
failures that are *not the proposer's behavior* — executor double format
failures, executor context exhaustion. Proposer-caused format failures (an
unparseable rule, a think block that runs past the output cap) were
originally masked too; measured over a full co-training run this creates a
blind spot in which think runaway grows unchecked (31.5% of trained
proposer turns truncating at the cap). The contract now gives
proposer-caused format failures the reward floor with loss ON: the agent
reports `proposer_format_failure` to `/finalize`, the server pins the
episode reward to the floor (bonus and per-grid gains included) while the
grid metrics keep reporting the real answers, and the field is surfaced as
a per-agent scalar so validation tracks the proposer's format-failure rate
directly. Context exhaustion after valid verification remains a real
behavioral outcome and is never masked.

## Executor competence

The 95%-exact / 99%-format gate before proposer RL was removed by decision
(2026-08-23): the greedy NVARC executor benchmark
(`tools/nvarc_executor_benchmark_megatron.py`, 150 held-out puzzles, one
format-only retry) remains a per-checkpoint diagnostic, re-run during
co-training because shared weights mean executor competence can drift.
Per-difficulty-bucket splits in its report are the primary view (aggregate
exactness hides "solves the easiest bucket only").

The completed investigation shows that removing the gate did not remove the
underlying dependency. The selected exact-RL checkpoint reached 27/150 exact
(18.0%). A frozen continuation briefly reached 31/150 and then regressed to
20/150; dynamic sampling ended at 27/150. Across 16 sampled attempts per case,
oracle-any exactness was 34.0%, but target-free modal consensus reached
only 20.667%. The executor is therefore too noisy for a failed execution to be
reliable evidence about the proposed rule.

## Metrics

Per-agent validation scalars surface as `val:<agent>/<field>/<stat>`.
The load-bearing ones: `grid_match`/`cell_match` (hidden_test final
answers; checkpoint selection uses the loop `cell_match` mean),
`format_valid`, `loss_masked`, `proposer_format_failure`, `rounds_used`,
`train_gate_pass`/`train_exact_fraction` (demo loop),
`eval_exact_fraction`/`eval_cell_match`/`all_solved` (eval_sequence), and
the single-turn agent's `grid_match`/`cell_match`/`copied_input`.
Validation should be greedy (per-request val sampling profile) so curves
are low-noise. Full traces label which tokens entered policy loss; W&B
full-result tables stay disabled (traces are large).

## Failure modes

| Failure | Consequence | Mitigation |
| --- | --- | --- |
| Correct rule, incorrect execution | Valid rule revised or rejected | Executor benchmark, format retry, separate executor training |
| Proposer think runaway | Rule never emitted; masked turns dodge gradient | Floor-reward with loss on for proposer-caused failures; SFT format/length prior |
| Echo hack | Copying the input outscores reasoning | Gain-over-echo scoring, echo pinned to the floor, `copied_input` metric |
| Context accounting slack | Trainable sample exceeds the pack bin | RL-side clamp + mask; margin in the agent budget |
| Training limit throttles inference | Loop cannot revise on real tasks | Per-row context-limit override for validation rows |
| Proposer patches examples instead of a rule | Demo gate passes, test fails | General-rule contract, held-out grids (eval_sequence), hidden test reward |
| Test target leaks into a prompt | Invalid evaluation | Server-side targets, forbidden-key scan on every model request |
| Disjoint chats concatenated as one trajectory | Invalid policy loss | Final-proposer-turn-only trainable response |

## Status and evidence (2026-09-04)

The protocol, agents, co-training rows, greedy validation, metric plumbing,
and trace capture work end to end. The earlier refinement loop improved
real-ARC cell accuracy from 0.059 to 0.166 at the best matched checkpoint,
but exact match remained 0/172.

Three closing diagnostics tested the executor bottleneck. Continued exact-RL
with frozen and dynamic curricula produced no durable gain over 27/150.
Sixteen-sample inference reached 34.0% oracle-any exactness but only 20.667%
modal-consensus exactness. Finally, one epoch of broad, answer-weighted
gold-rationale SFT drove the frozen benchmark from 27/150 to 4/150 even as
held-out teacher-forced loss improved from 0.1116 to 0.0880. These results
close the current investigation: the executor channel is not accurate enough
to distinguish a bad rule from a bad execution.

## Conditions for revisiting

- Demonstrate a durable paired exact-match gain on the same frozen executor
  benchmark, including a later checkpoint rather than a transient peak.
- Change the representation or selection hypothesis materially; do not reopen
  for another learning-rate, reward-weight, prompt, or epoch sweep alone.
- Keep exact grid match as the decision metric. Token loss, shorter reasoning,
  and cell accuracy are useful diagnostics but did not predict success here.
- Keep the executor benchmark disjoint from generated SFT data and continue to
  record both proposer and executor traces.
- Run the deferred Kimi K3 oracle-rule ceiling only if that diagnostic is
  explicitly reopened.

## References

- [Executor investigation closeout](arc-agi-multiturn-investigation.md)
- [NeMo-Gym integration](nemo-gym-integration.md)
- Gym components: `responses_api_agents/arc_transform_refinement_agent/`,
  `resources_servers/arc_agi_2/` (submodule branch `anthomas/colo1`)
- NeMo-RL surface: `tools/nvarc_cotrain_materialize.py`,
  `tools/nvarc_cotrain_preflight.py`,
  `tools/nvarc_executor_benchmark_megatron.py`,
  `examples/configs/async/env_nvarc_cotrain.yaml`
- NVARC: Sorokin & Puget, *NVARC solution to ARC-AGI-2 2025*;
  https://github.com/1ytic/NVARC
- ARC-Lang/Berman refinement-loop precedent:
  https://github.com/jerber/arc-lang-public
