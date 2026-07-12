# Plan: continuous-rollout scheduler for colocated async GRPO

Implements the "Ideal colocated execution model" section of `colo_vs_noncolo.md` with surgical,
flag-gated changes. Default behavior (target-window scheduler) is untouched; the new mode is opt-in
via config so non-colo and existing colo runs are unaffected.

## Target invariants

1. **Generation phase**: every GPU busy; a finished rollout is immediately replaced by a new one.
   No target-step gating, no collector self-pause, no lookahead window that exhausts.
2. **Switch**: triggered by buffer state ("enough complete groups"), decided while generation is
   live. Mid-flight rollouts are suspended in place; generation buffers offloaded.
3. **Training phase**: consumes only already-complete groups; **never waits on the buffer**. A
   buffer miss after the engine pause is a hard error (assert), not a retry loop.
4. **Resume**: weights updated in place (single dual-mode model), engine resumes, paused rollouts
   continue from the exact token, now decoding under the new weights. Mixed-version rollouts are
   fine: per-token generation logprobs are recorded at sampling time, so the token-level IS
   correction needs no special casing.

## Building blocks already in place (verified)

- **Engine suspend/resume retaining in-flight requests**: Megatron dynamic engine reaches a
  barrier-synchronized PAUSED state and keeps its requests across
  `finish_generation()`/`prepare_for_generation()` — the "Phase A" path documented in
  `AsyncTrajectoryCollector.prepare_for_refit` (`trajectory_collector.py:494-528`). Exercised every
  step in the E3 colo run.
- **Staleness-window replay buffer**: `ReplayBufferNew`
  (`nemo_rl/algorithms/async_utils/replay_buffer.py:557`) already implements evict-by-staleness +
  freshest-first sampling with no target matching. Its docstring states the motivation: target
  gating "causes generation pauses". WIP but small and inherits all persistence from
  `ReplayBufferImpl`.
- **Single dual-mode model** (E1b): trained weights ARE the inference weights; the colocated
  "refit" is just offload + `prepare_for_generation()` (`grpo.py:4145-4154`). Already implemented.
- **Collector pause/resume plumbing**: `_refit_pause_cleared` / `_manual_pause_cleared` events and
  the per-group worker threads + in-flight semaphore (`trajectory_collector.py:80-120`).
- **IS-corrected loss** (`use_importance_sampling_correction=true`) and per-token
  `generation_logprobs` already flow through `_build_async_grpo_train_data`.

## Changes

### 1. Config — one new key (v1 TypedDict extension, default in exemplar YAML)

`AsyncGRPOConfig` (`grpo.py:141`) is a legacy TypedDict; extending it (not adding a new TypedDict
class) is the sanctioned pattern:

```python
class AsyncGRPOConfig(TypedDict):
    ...
    # Rollout scheduling mode:
    #   "target_window" (default): collector generates per-target batches gated on
    #       [N+1..N+max_age]; training samples exact-target batches (current behavior).
    #   "continuous": collector keeps the engine saturated with untargeted rollouts;
    #       training samples any complete groups within max_trajectory_age_steps
    #       (freshest-first, ReplayBufferNew). Intended for colocated mode.
    rollout_scheduler: NotRequired[str]
```

- Default `rollout_scheduler: "target_window"` goes in the exemplar YAML
  (`examples/configs/grpo_math_1B.yaml` async_grpo block) — no `.get(k, default)` at call sites;
  read via `.get("rollout_scheduler")` and treat `None`/absent as target_window explicitly at ONE
  dispatch point in `async_grpo_train`.
- Reuse `max_trajectory_age_steps` as the staleness bound (`ReplayBufferNew.max_staleness`) — no
  second knob.
- Experiment config: `noval_colocated_continuous_nanov3.yaml` with
  `defaults: noval_colocated_singlemodel_nanov3.yaml` + `grpo.async_grpo.rollout_scheduler: continuous`.

### 2. Replay buffer (`replay_buffer.py`) — finish `ReplayBufferNew` minimally

- Add `ready(num_prompt_groups: int, current_weight_version: int) -> bool`: evict stale rows (same
  logic as `_evict`), return `len(trajectories) >= num_prompt_groups`. This is the driver's
  switch-trigger probe — checks readiness **without consuming**, so the driver can pause the engine
  and then sample with a guarantee (buffer only grows between `ready()` and `sample()`;
  eviction is version-driven and the version doesn't change in between).
- Override `load_state_dict`: skip `_prepare_for_training_step` / `_remove_incomplete_target_steps`
  (target-step bookkeeping is meaningless here); restore lists, `_evict(current_training_step)`,
  `_truncate_to_max_size`.
- Export `ReplayBufferNew` from `nemo_rl/algorithms/async_utils/__init__.py` and drop the
  "DO NOT USE" warning once tests pass (or keep the class name and soften to "experimental").
- Leave `ReplayBuffer` (target mode) untouched.

### 3. Collector (`trajectory_collector.py`) — continuous spawn mode

Gate on the config key at `__init__` (`self._continuous = async_cfg.get("rollout_scheduler") ==
"continuous"`). In continuous mode:

- `_process_batch`: skip `_get_next_target_for_generation` /
  `_should_pause_for_generation_limits` / `_generating_targets` reservation machinery entirely.
  For each prompt in the dataloader batch: acquire the in-flight semaphore, spawn a worker with
  `target_weight_version = generation_weight_version` (stored but unused for sampling). The
  semaphore is the ONLY throttle; buffer-full backpressure (`add()` → "full" → worker backoff)
  already exists as the second safety net.
- Because there is no window to exhaust, the collection loop never self-pauses waiting for a
  weight update: as each worker finishes, the semaphore slot frees and the loop immediately spawns
  the next prompt — the "rollout finishes → next rollout starts" invariant. The
  `_generation_limit_cleared` event stays permanently set in continuous mode.
- Semaphore size: keep the existing formula `num_prompts_per_step * max_trajectory_age_steps`
  (64 here). In continuous mode this is a genuine concurrency cap (tunable via the two existing
  knobs), not a self-strangling window, because slots refill continuously. If engine traces show
  concurrency-starvation at the cap, bump `max_trajectory_age_steps` — it raises both the cap and
  the staleness bound coherently.
- Keep spawn-time version tagging (conservative: the oldest tokens' version governs staleness).
  Follow-up option (not this change): tag at completion and log spawn→completion version spread.
- `prepare_for_refit` / `resume_after_refit` / `set_weight_version`: unchanged — Phase A already
  does exactly what the ideal workflow needs at the phase boundary.

### 4. Driver (`grpo.py` `async_grpo_train`) — reorder ~30 lines, flag-gated

- **Buffer construction** (`grpo.py:3591`): if continuous, instantiate `ReplayBufferNew` with
  `max_size=optimal_buffer_size, max_staleness=max_trajectory_age_steps` (same runtime env).
- **Initial fill wait** (`grpo.py:3771-3799`): if continuous, poll `ready(num_prompts_per_step,
  step)` instead of `has_complete_batch(step, ...)` / `get_trajectories_needed`.
- **Main loop** (`grpo.py:3820-3864`): the existing structure is `sample()-retry-loop → process
  rewards/data → pause engine → train`. In continuous mode replace ONLY the retry loop:

  ```
  # Generation phase: engine + collector live. This wait IS the generation phase.
  with timer.time("exposed_generation"):
      while not ray.get(replay_buffer.ready.remote(num_prompt_groups_needed, weight_version)):
          time.sleep(1.0)
      sample_result = ray.get(replay_buffer.sample.remote(...))
  assert sample_result is not None and len(sample_result["trajectories"]) == num_prompt_groups_needed, (
      "continuous scheduler invariant violated: buffer lost groups between ready() and sample()"
  )
  ```

  Everything downstream (reward processing, data prep on the driver CPU while the engine keeps
  generating, the colocated pause at `grpo.py:3981-3989`, logprobs, train, and the resume at
  `grpo.py:4145-4154`) is unchanged. Training-phase code after the pause never touches the buffer,
  so invariant 3 holds by construction.
- Target-window path: byte-for-byte unchanged.

### 5. Observability (small, but decides E4)

- Log at each switch: buffer size at switch, groups consumed, avg + max trajectory age, generation
  phase duration (`exposed_generation` now measures exactly the generation phase).
- Reuse the opt-in `NRL_GEN_TRACE` engine trace to spot-check concurrency stays near the cap for
  the whole generation phase (the E3 failure signature was concurrency decaying to 1-2 groups).

## What this fixes, mechanically

| E3 failure mode | Why it disappears |
| :---- | :---- |
| ~31-37 min full-latency stall every `max_age` steps | No exact-target sampling: any complete groups train; phase ends when `num_prompts` are ready, not when a specific target completes |
| Collector self-pause (19x) | No target window to exhaust; spawn loop refills slots continuously |
| Concurrency drains to straggler tail → reaper | Semaphore slots refill immediately; stragglers just keep running alongside fresh rollouts across phases |
| Straggler death → permanent stall (unreachable target) | No targets; a dead worker only delays the count by one group |
| Data pinned at max staleness 4.0 | Freshest-first sampling; expected steady-state age ≈ 1 |

## Testing

1. **Unit** (no Ray actors needed — test the impl classes directly, per testing-skill guidance):
   - `ReplayBufferNew`(Impl-level): `ready()` semantics incl. eviction; `sample()` freshest-first
     order; `load_state_dict` restore + evict + truncate; ready→sample consistency.
   - Collector continuous-mode spawn gating with a stubbed `policy_generation` (threads spawn up
     to semaphore, refill on completion, no self-pause).
2. **Smoke**: `async_colocated_debug_nanov3.yaml` (2 prompts) + `rollout_scheduler=continuous`,
   a few steps on a small allocation; assert no "STALLING"/"Insufficient valid groups" lines and
   that each step's post-pause sample succeeds first try.
3. **E4 (the real test)**: rerun the E3 protocol — `noval_colocated_continuous_nanov3.yaml`,
   4h, 8 nodes, `nemotron_sw_post`, fresh SFT checkpoint, ckpt every 5 steps — vs the existing
   arms (non-colo 39 steps, colo target-window 19 steps).
   - Success: ≥ ~35 optimizer steps/4h (parity or better vs non-colo; ~60 is the decode-parity
     ceiling), avg trajectory age ≈ 1, no reaper kill, engine concurrency near cap through each
     generation phase.
   - Then extend the offline-validation sweep (`run_offline_validation.sh`) to the new arm's
     checkpoints for the reward-vs-step comparison.

## Branch / ledger

- Branch per auto-research convention: `async_colo1-continuous-scheduler` off `async_colo1`
  (one commit for the scheduler, separate commit for the experiment YAML).
- Log E4 in `experiment_log.md`; update `colo_vs_noncolo.md` findings when E4 lands.

## Risks / open questions

- **Suspended-rollout memory**: in-flight requests held across the training phase occupy engine
  state that must offload cleanly; this is the same path E3 exercised every step (Phase A), so low
  risk, but watch HBM at `finish_generation` with ~64 suspended groups.
- **Version-spread of resumed rollouts**: a straggler may now span many weight versions. Loss-side
  it is exact (recorded logprobs + IS correction); quality-side we bound it with spawn-version
  staleness eviction. Monitor evicted-group counts — high eviction = wasted generation, consider
  completion-version tagging.
- **Dynamic sampling / OPD interplay**: continuous mode changes only WHICH groups train a step,
  not their shape; teacher-logprob computation happens per-worker as today. No changes expected,
  verify in smoke run.
- **Non-colocated + continuous**: should also work (and is where `ReplayBufferNew` was headed
  upstream), but out of scope for E4; keep non-colo on target_window for the controlled comparison.
