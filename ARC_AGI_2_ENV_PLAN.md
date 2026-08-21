# ARC-AGI-2 Execution Plan

## Objective

Improve exact match on the fixed 172-row ARC-AGI-2 validation subset. The current approach separates rule induction from deterministic rule execution and uses exact verification on the known training pairs before applying a proposed transform to a hidden test grid.

## Implementation status

- Pure ARC grid parsing, scoring, diffs, state transitions, leakage checks, and context budgeting are implemented in the NeMo-Gym ARC resource server.
- The refinement agent keeps one proposer history and uses fresh executor chats. Only proposer tokens are returned for policy loss.
- Synthetic rules have canonical textual descriptions, and the executor benchmark sends one grid per fresh chat using the same prompt contract as executor training.
- Executor GRPO training and native Megatron checkpoint benchmarking are implemented for the four-node colocated recipe.
- The current step-20 executor checkpoint reaches 94/150 exact grids (62.7%), 138/150 format-valid responses after one retry (92%), 110/150 shape matches (73.3%), and 86.8% cell accuracy. This is below the 95% exact and 99% format gates.
- End-to-end proposer training remains gated on executor reliability. The authoritative real-ARC result remains 0/172 exact.

## Retained ARC surface

| Path | Purpose |
| --- | --- |
| `nemo_rl/environments/arc_agi_generators.py` | Deterministic synthetic transforms and canonical rule descriptions |
| `nemo_rl/environments/arc_agi_grid.py` | Grid serialization, parsing, and dense/exact scoring |
| `nemo_rl/environments/arc_agi_environment.py` | Executor GRPO reward environment |
| `nemo_rl/data/datasets/response_datasets/arc_executor.py` | Reproducible executor train/validation datasets |
| `examples/configs/async/env_arc_executor.yaml` | Executor task and reward configuration |
| `examples/configs/async/nanov3_arc_executor_4n_colocated.yaml` | Four-node Megatron training recipe |
| `tools/arc_executor_benchmark.py` | Backend-neutral single-grid benchmark logic |
| `tools/arc_executor_benchmark_megatron.py` | Native Megatron inference driver |
| `3rdparty/Gym-workspace/Gym/resources_servers/arc_agi/` | Session-owned verifier and hidden targets |
| `3rdparty/Gym-workspace/Gym/responses_api_agents/arc_transform_refinement_agent/` | Persistent proposer/fresh executor orchestration |

The obsolete direct rule-inference modes, one-node executor recipe, early-answer prompt, and their dedicated scoring/preflight tools have been removed.

## Next gates

1. Continue executor training on a larger allocation while preserving the single-grid objective and fixed held-out split.
2. Re-run the 150-case Megatron benchmark. Require at least 95% exact grids and 99% format validity.
3. Run a fixed inference-only end-to-end subset and inspect complete proposer/executor traces.
4. Run a small RL plumbing check for token ownership, rewards, context termination, async streaming, and resume behavior.
5. Train the proposer only after those gates pass, then evaluate on the fixed real validation subset.

## Operational constraints

- Run every test, preflight, benchmark, and training command through Slurm on a compute node.
- Use the container selected by `launch_experiment.sh` / `submit_nemorl.sh`.
- Keep synthetic training and held-out seeds distinct; do not train on the fixed validation rows.
- Preserve full executor traces for audit, but never place executor tokens in the proposer loss-bearing history.
