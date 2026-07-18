# Colocated deadlock fence — verification handoff

Date: 2026-07-18
Branch: `colo-deadlock-instrumentation` (based on `async_colo2`)
Workspace: `/lustre/fs1/portfolios/nemotron/projects/nemotron_sw_pre/users/anthomas/RL2`

## Goal and status

The colocated Megatron async-GRPO path nondeterministically deadlocked near the
generation-to-training transition, commonly around weight version 3 to 4. The
desired behavior is to preserve active generation requests and KV state while
inference is paused for training, then resume those same requests afterward.
Waiting for all generation threads to finish is only a diagnostic control and
is not the intended solution.

This branch has two commits on top of `async_colo2`:

1. `e8c998a0 feat(megatron): trace dynamic engine pause state`
2. `fix(megatron): fence colocated train and inference collectives` (the commit
   containing this handoff)

The 4-node verification passed all eight transitions, including the historical
v3-to-v4 failure window. Because the original failure was nondeterministic and
was observed at 8-node scale, run the 8-node confirmation below before declaring
the issue fully resolved.

## Diagnosis established by the traced controls

Previous traced jobs:

- `5465557`: default in-flight updates; failed after versions 0 through 3.
- `5465652`: `ep_consensus_interval=1`; failed similarly, so a stale consensus
  interval was not the cause.
- `5465563`: `in_flight_weight_updates=false`; reached version 8, showing the
  failure requires the in-flight transition. This run drains generations and is
  not an acceptable production fix.

In both failing in-flight jobs, all ranks reported `PAUSED`, `SUSPENDED`, and
completion of the caller barrier. The CPU state machine therefore said the
engine was suspended even though the NCCL flight recorder later showed training
and decode collective sequences crossing.

Code inspection exposed two gaps in the transition fence:

- Sleep waited for `SUSPENDED`, returned from the inference-loop coroutine, and
  only then synchronized CUDA on the Ray caller thread. The inference thread
  itself never acknowledged GPU quiescence.
- Wake resumed/unpaused the inference engine before synchronizing and barriering
  the preceding training work. Decode could therefore restart before all ranks
  had retired training collectives.

## Implemented fix

`nemo_rl/models/generation/megatron/megatron_worker.py` now enforces:

### Inference to training

1. Stop new generation admission at the trajectory collector (existing behavior).
2. Pause and suspend the dynamic inference engine.
3. On the dedicated inference-loop thread, synchronize the explicit CUDA device
   identified by `LOCAL_RANK`.
4. Return the quiescence acknowledgement to the caller.
5. Complete a global barrier before training uses the shared communicator.

### Training to inference

1. Synchronize the explicit local CUDA device while inference remains suspended.
2. Complete the global barrier across all ranks.
3. Only then send resume and unpause signals to the inference engines.

Trace events record thread ID/name, `LOCAL_RANK`, and current CUDA device for both
quiescence acknowledgements. Focused tests cover sleep ordering, wake ordering,
and explicit device targeting.

## Completed verification

Job: `5467501`
Mode: colocated, 4 nodes / 16 GPUs, interactive QOS
Config: 8 steps, sequence length 16384, `in_flight_weight_updates=true`, default
EP consensus interval, EP tracing and NCCL flight recorder enabled.

Launch command:

```bash
SUBMIT_ACCOUNT=nemotron_sw_post QOS=interactive NUM_ACTOR_NODES=4 \
SMOKE_STEPS=8 RUN_TAG=epfence_fix_v1 \
NRL_COLL_TRACE=1 NRL_EP_TRACE=1 \
NCCL_TRACE_DIR=/lustre/fs1/portfolios/nemotron/projects/nemotron_sw_pre/users/anthomas/RL2/nccl_traces/20260718_epfence_fix_v1 \
EXTRA_OVERRIDES='logger.wandb.name=async_colocated_epfence_fix_v1' \
bash launch_experiment.sh colocated --smoke
```

Evidence in `5467501-logs/ray-driver.log`:

- Active requests remained present during pauses (`local_pending` was often
  nonzero), proving this was pause/resume rather than draining.
- Every cycle logged `inference_quiesce_exit` before `sleep_barrier_post`.
- Every wake logged `training_quiesce_exit`, then `wake_barrier_post`, then
  `wait_running_exit`.
- Weight versions advanced monotonically from 0 through 8.
- The configured 1000-sample validation completed after step 5.
- The driver printed `Async GRPO training complete!`.
- No NCCL watchdog timeout, CUDA error, illegal memory access, actor death, or
  `sleep_wait_timeout` occurred.

`git diff --check` passed. Standalone unit tests were not executed because the
user required all code execution to use a Slurm compute node through
`launch_experiment.sh`; the full distributed experiment exercised the changed
runtime path.

Expected shutdown-only noise includes NemoGym `ClientOSError` retries, cancelled
generation futures, and a Ray core-worker finalization assertion after training
completion. These occur after version 8 and are not the training deadlock.

## Required 8-node confirmation

Use normal QOS; interactive is capped at four nodes:

```bash
SUBMIT_ACCOUNT=nemotron_sw_post QOS=normal NUM_ACTOR_NODES=8 \
RUN_TAG=epfence_fix_8n_v1 NRL_COLL_TRACE=1 NRL_EP_TRACE=1 \
NCCL_TRACE_DIR=/lustre/fs1/portfolios/nemotron/projects/nemotron_sw_pre/users/anthomas/RL2/nccl_traces/20260718_epfence_fix_8n_v1 \
EXTRA_OVERRIDES='policy.max_total_sequence_length=16384 grpo.max_num_steps=8 checkpointing.enabled=false logger.wandb.name=async_colocated_epfence_fix_8n_v1' \
bash launch_experiment.sh colocated
```

Success criteria:

- Weight versions 0 through 8 appear in the real driver progression.
- Each sleep shows inference-thread quiescence before the sleep barrier.
- Each wake shows training quiescence and the wake barrier before `RUNNING`.
- Zero watchdog, CUDA, illegal-memory, actor-death, or pause-timeout signatures.
- Treat the completion banner as valid only if the version progression and
  negative error audit also pass.

Do not remove or overwrite unrelated untracked workspace artifacts (`reports/`,
`session/`, prior `nccl_traces/`, `ASYNC_GRPO_MEGATRON.md`, and user notes).
