# Verifier-Guided Multi-Turn ARC-AGI

Revised 2026-08-26 to describe the implemented system (NeMo-Gym
`arc_transform_refinement_agent` + `arc_agi_2` resources server on the Gym
submodule, co-trained with single-turn executor rows through
`examples/nemo_gym/run_grpo_nemo_gym.py`). Sections marked *adopted
revision* record design changes made after measurement.

## Motivation

A single model response that infers a rule, mentally tests it, and emits
the test grid anchors on its own narration. This design replaces the
narrated self-check with an actual check: rules are executed by a separate
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
sequence — each grid contributes its best attempt's gain-over-echo score,
unreached grids sit at the reward floor (ending early is never better than
attempting), and solving everything earns a configured bonus.

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

## Status and evidence (2026-08-26)

Implemented and proven end to end (gym branch `anthomas/colo1`; RL branch
`async_arc4`): unit-tested protocol logic and agents, executor+proposer
co-training over one JSONL (curriculum and role mixture baked into row
order), greedy dual validation (single-shot + hidden_test loop per real
row), per-agent metric plumbing and loop-metric checkpoint selection.
Measured: the refinement loop beats single-shot induction ~3x on real-ARC
cell accuracy at matched checkpoints (0.166 vs 0.059 at the best anchor);
real-ARC exact match remains 0/172; the binding constraints are proposer
think runaway and the throttled inference loop, addressed by the adopted
revisions above. Campaign status and the next-session plan live in the
untracked `ARC_AGI_2_ENV_PLAN.md`; experiment history in the untracked
research ledger.

## Open questions

- Does floor-rewarding proposer format failures suppress runaway without
  suppressing useful long reasoning, or is an explicit think budget
  needed?
- How much does multi-round refinement buy at inference once the loop is
  unthrottled — and does it convert cell-accuracy gains into exact solves?
- Is a short SFT stage on NVARC reference rules (deliberately relaxing the
  no-imitation invariant) a net win as a format/length prior before RL?
- Once the single-path loop works, does selecting among several
  independently proposed rules (deferred phase) justify the extra
  inference?

## References

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
