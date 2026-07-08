# Async Colocated GRPO (Megatron backend): Code-Path Guide & Debug Plan

Companion to `experiment_log.md`. All file:line references are against branch
`async_colo1` (main @ f2ab62c2 + PR 2884 cherry-pick + our fixes). Where a
line number drifts, the symbol name is the anchor.

## 1. The cast of actors

Everything is a Ray actor spawned by the driver (`examples/nemo_gym/run_grpo_nemo_gym.py`
→ `nemo_rl/algorithms/grpo.py::setup()`):

| Actor | Code | Role |
|---|---|---|
| `MegatronPolicyWorker` ×32 | `nemo_rl/models/policy/workers/megatron_policy_worker.py` | One per GPU. In colocated mode each worker does BOTH training (train/get_logprobs) and generation (hosts a shard of the persistent inference engine). |
| `Policy` (driver-side wrapper) | `nemo_rl/models/policy/lm_policy.py` | Shards data across DP ranks, fans out worker calls. |
| `MegatronGeneration` (driver-side wrapper) | `nemo_rl/models/generation/megatron/megatron_generation.py` | Generation facade. Colocated: wraps the training `Policy` (`policy is not None` branch, line ~97). Non-colocated: builds a second inference-only `Policy` on separate nodes. |
| `AsyncTrajectoryCollector` | `nemo_rl/algorithms/async_utils/trajectory_collector.py` | Background rollout production: spawns one thread per prompt group, each calling NemoGym over HTTP. |
| `ReplayBuffer` | `nemo_rl/algorithms/async_utils/replay_buffer.py` | Holds trajectory groups tagged by weight version; enforces `max_trajectory_age_steps`. |
| `NemoGym` | `nemo_rl/environments/nemo_gym.py` | Spins up 13 HTTP servers (resource servers + agents + the `policy_model` proxy) in the pre-built venvs under `$NEMO_GYM_VENV_DIR`. |

### The generation data path (colocated)

```
AsyncTrajectoryCollector thread
  → NemoGym simple_agent (HTTP)
    → policy_model proxy  (Gym/responses_api_models/vllm_model/app.py;
       clients built from MegatronGeneration.dp_openai_server_base_urls)
      → per-DP-leader OpenAI server on the worker
         (megatron_worker._setup_openai_api_server, mcore start_text_gen_server)
        → DynamicInferenceEngine + coordinator (mcore)
          → model forward on the SAME GPUs/process groups as training
```

## 2. Worker threading model — the crux

Each `MegatronPolicyWorker` has **two threads that can issue CUDA/NCCL work**:

1. **Main Ray actor thread** — executes `train`, `get_logprobs`,
   `prepare_for_generation`, `finish_generation`, etc.
2. **Inference event-loop thread** — started at engine init
   (`_start_inference_loop_thread`, megatron_worker.py:260). Runs a persistent
   asyncio loop hosting the mcore `DynamicInferenceEngine` step loop and (rank 0
   of each DP group) the HTTP server + inference coordinator client.

Both threads drive collectives on the **same NCCL process groups** (the engine
reuses the model's `pg_collection`, megatron_worker.py:97). Evidence: the EP
alltoall SeqNums in the hang dumps are in the millions — accumulated by ~2h of
inference decode steps on the training PGs. Any situation where both threads
(or two ranks in different phases) enqueue collectives on one PG is a deadlock.

## 3. Startup path (async colocated + megatron)

1. `setup()` (grpo.py:~1062): `init_policy()` first — HF→Megatron conversion
   (cached via `NRL_MEGATRON_CHECKPOINT_DIR`,
   `nemo_rl/models/policy/utils.py:236`; gated by a bare `os.path.exists`,
   `nemo_rl/models/megatron/setup.py:460,1446` — no lock, don't cold-start two
   jobs on one empty cache).
2. `MegatronGeneration(policy=policy)` — colocated branch (**our fix**): calls
   `policy.offload_before_refit()` → `prepare_for_generation()` → collects
   `dp_openai_server_base_urls` from workers. Originally returned early with an
   empty URL list → Gym proxy had 0 clients (`ZeroDivisionError`, app.py:580).
3. `prepare_for_generation` (megatron_worker.py:379):
   `model.eval()` → `toggle_cuda_graphs(lang_module, set_to=cuda_graph_impl)`
   (switches module-tree attrs; mcore `transformer/utils.py:414`) → first call
   `_initialize_inference_engine` (builds `DynamicInferenceContext` with
   `buffer_size_gb`, `kv_cache_management_mode`, `moe_pad_experts...` assert at
   `text_generation_controller.py:127`) → `_run_async_coordinator_start`
   (starts engine loop + HTTP server); later calls `_wake()` (no-op if awake).
4. `_spinup_nemo_gym(policy_generation.dp_openai_server_base_urls, ...)`
   (grpo.py:1074) — needs non-empty URLs, hence fix ordering above.
5. `async_grpo_train` (grpo.py:~3530): colocated branch re-runs
   `offload_before_refit` + `prepare_for_generation` (idempotent), THEN starts
   the collector (`start_collection`), then enters the loop.

## 4. Steady-state loop (colocated branch of `async_grpo_train`)

Per training step (grpo.py ~3944 onward):

```
A. PAUSE   print "⏸️ Pausing colocated engine + collector for training..."
           ray.get(collector.prepare_for_refit.remote())
             └─ trajectory_collector.py:462 — with OUR FIX: always
                wait_for_pending_generations() when colocated (drains the
                in-flight rollout threads; observed tail 69–99 min!)
           policy_generation.get_logger_metrics()
           policy_generation.finish_generation()
             └─ worker.finish_generation (megatron_worker.py:345):
                _sleep() [pause→suspend engine, barriers] then
                toggle_cuda_graphs(set_to="none"), gc + empty_cache
B. LOGPROBS  policy.get_logprobs(...)          ← ★ EVERY attempt hangs here
             └─ lm_policy._shard_for_logprob (packing bins computed globally
                on driver: shard_by_batch_size + sequence_packing_args)
             └─ worker.get_logprobs (megatron_policy_worker.py:917)
                → get_microbatch_iterator (models/megatron/data.py:144)
                → megatron_forward_backward (models/megatron/train.py:360)
C. TRAIN     policy.train(...)  (same fwd/bwd machinery + optimizer step)
D. RESUME    policy.offload_before_refit()      [optimizer/grads → CPU]
             policy_generation.prepare_for_generation()   [toggle graphs on,
             _wake() engine]
             weight_version += 1; collector.set_weight_version; resume_after_refit
```

No refit/weight-transfer ever happens colocated — the engine reads the shared
(updated) weights on resume. Validation (every `val_period`) uses the same
pause/resume dance.

## 5. Bug ledger

Fixed on this branch (all commits on `async_colo1`):

1. **Empty `dp_openai_server_base_urls`** in the colocated ctor →
   `megatron_generation.py` colocated branch now starts the engine + collects
   URLs (see §3.2).
2. **`moe_pad_experts_for_cuda_graph_inference` assert** — colocated reuses the
   *training* model (transformer_engine impl) and toggles `cuda_graph_impl=local`
   on it; with EP>1 mcore requires the pad flag
   (`text_generation_controller.py:127`). Set via `policy.megatron_cfg` in the
   colo yaml (plumbed at `models/megatron/setup.py:693`).
3. **Drain before pause** — `prepare_for_refit` skipped draining when
   `in_flight_weight_updates` (fine non-colocated, hazardous colocated);
   now drains when `colocated.enabled` (trajectory_collector.py:~495).
4. Launcher/env hardening: per-mode `HF_MODULES_CACHE` (trust_remote_code
   dynamic-module cache race across concurrent jobs), `SETUP_COMMAND` installing
   `torch_memory_saver` into the baked worker venv (needed only by offload mode).

### The open bug: first post-pause `get_logprobs` NCCL hang

Symptom: ~10 min after logprobs begin (or after ~2h of slow microbatches in
some attempts), the PG watchdog kills the job. Per-rank dumps show **different
ranks stuck on different collectives** — EP `ALLTOALL_BASE` both huge (~8.4GB
data dispatch) and tiny (26–79KB, token-count exchange), TP `_ALLGATHER_BASE`
big and small — i.e., ranks are executing **different points in the collective
schedule**.

Falsified (each via a full 2.5h cycle):
- Memory capacity (OOMs fixed by 32GB buffer + 32k seqlen; hang persists with ~50GB free)
- VMM allocators (hangs with standard allocator too)
- Engine paused mid-decode (hangs with fully drained + suspended engine)
- (Likely) per-rank microbatch-count mismatch (bins packed globally on driver;
  not 100% verified — see experiment idea E3 below)

Prime suspects — all "mode-switch residue" on the shared model, i.e. state the
generation phase leaves behind that makes the training forward diverge per rank:

- **S1: MoE dispatcher switching.** Training uses
  `moe_token_dispatcher_type=alltoall`; inference sets
  `inference_moe_token_dispatcher_type=nccl` (both on the same `model_cfg`,
  `models/megatron/setup.py:674-679`). Where/how mcore selects the dispatcher
  per-forward, and whether the paused engine leaves the nccl dispatcher (or its
  communicator handles) active, is unaudited.
- **S2: `toggle_cuda_graphs` cache/restore.** `transformer/utils.py:414`
  caches and rewrites `cuda_graph_impl` / `recompute_granularity` /
  `cudagraph_manager` across the module tree. If any rank's restore diverges
  (e.g., recompute_granularity left None on some ranks → different recompute →
  different collective counts inside a microbatch), you get exactly this hang.
  Note interplay: activation_checkpointing=true matters only if
  recompute attrs are restored correctly after generation toggled them off.
- **S3: `set_decode_expert_padding` residue.** The generation controller
  toggles expert padding per-step (`text_generation_controller.py:584-597`);
  if left True on some ranks' expert modules, the MoE dispatch shape/sequence
  differs per rank in the training forward.
- **S4: engine thread not fully quiet.** `_sleep` awaits PAUSED then SUSPENDED
  (megatron_worker.py:218-237) with a `torch.distributed.barrier()` — but the
  barrier is on the default PG from the *main* thread; verify no engine-loop
  task can still enqueue work after SUSPENDED (e.g., coordinator heartbeats,
  a queued chunked-prefill continuation).

## 6. Interactive debugging plan

### 6.1 Get a live cluster (no batch COMMAND)

```bash
# from repo root — omit COMMAND so ray.sub leaves the cluster idle
JOB_NAME=colo_debug SUBMIT_ACCOUNT=nemotron_sw_pre bash submit_nemorl.sh
# wait ~5 min for <jobid>-attach.sh to appear, then:
bash <jobid>-attach.sh          # shell inside the head-node container
```

Run the driver manually inside the attach shell (same env as
`launch_colo_experiment.sh`'s COMMAND, config `async_colocated_nanov3.yaml`).
The 4h limit in `submit_nemorl.sh` may need raising to 6–8h for a debug session.

### 6.2 Make the repro cheap first (do this before anything else)

The hang gates on "first training pause", which today costs ~2h of buffer
fill + ~1h drain. Shrink it to minutes with a smoke-scale override set:

```
grpo.num_prompts_per_step=2 grpo.num_generations_per_prompt=4 \
policy.train_global_batch_size=8 policy.max_total_sequence_length=8192 \
grpo.val_period=1000
```

Two outcomes, both informative:
- **Hang reproduces in minutes** → iterate freely (each probe is cheap).
- **Hang does NOT reproduce** → the bug is load/length-dependent (points to
  S4 or straggler-slowness rather than deterministic S1–S3 residue); bisect
  upward (seqlen 8k→16k→32k, prompts 2→8→16).

### 6.3 Instrument before launching

Export in the driver env (propagates to all workers via
`virtual_cluster.py:197`):

```bash
# PyTorch NCCL flight recorder: on watchdog timeout, dumps the last N
# collectives per rank (op type, sizes, seq numbers, states) — this directly
# answers "which collective did each rank last enqueue".
export TORCH_NCCL_TRACE_BUFFER_SIZE=20000
export TORCH_NCCL_DUMP_ON_TIMEOUT=1
export TORCH_NCCL_DEBUG_INFO_TEMP_FILE=/lustre/fsw/portfolios/nemotron/users/anthomas/nccl_traces/rank_
# Optional while iterating: fail fast instead of waiting 10 min
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=300
```

### 6.4 When it hangs: capture per-rank state

The workers are ordinary processes on the compute nodes. From the attach shell
(or `srun --overlap -w <node>` into each node's container):

```bash
# find the worker pids
pgrep -af MegatronPolicyWorker
# dump ALL threads of every worker (main + inference loop + watchdog):
for pid in $(pgrep -f MegatronPolicyWorker); do
  py-spy dump --pid $pid --native > /lustre/.../pyspy_$(hostname)_$pid.txt
done
```

(`py-spy` can be `uv pip install`ed into the worker venv the same way we
installed `torch_memory_saver`, or run from any python env on the node — it
attaches externally.)

What to look for, in order:
1. **Diff the main-thread stacks across the 8 ranks of one EP×TP island.**
   The deadlock requires ≥2 ranks at different microbatches / different
   collectives. The stack diff + flight-recorder seq numbers identify WHO is
   ahead and at which call site (`moe_layer.py` dispatch vs `transformer_layer`
   allgather vs somewhere unexpected).
2. **Check the inference thread's stack on every rank.** It should be parked
   in `run_forever` with no CUDA frames. Any rank whose engine thread sits in
   a collective or a CUDA call after "paused inference engine" = S4 confirmed.
3. **Inspect module state on a hung rank** (only if 1–2 are inconclusive):
   `gdb -p <pid>` + python extension, or restart the repro under a driver-side
   `ray.util.pdb`/remote-pdb breakpoint placed at the top of
   `worker.get_logprobs` to print, per rank:
   `model.config.cuda_graph_impl`, each MoE layer's dispatcher type/instance,
   expert `decode_expert_padding` flags, `recompute_granularity` per block.
   Ranks should be identical; any divergence maps directly to S1/S2/S3.

### 6.5 Cheap non-interactive probes (if an interactive slot is scarce)

- Set `mcore_generation_config.cuda_graph_impl: "none"` for one colo run:
  removes S2/S3 entirely (no graph toggling, no pad-experts flag needed).
  If the hang vanishes, the bug lives in the CUDA-graph/padding toggle path.
- Set `inference_moe_token_dispatcher_type: "alltoall"` (match training): tests S1.
- `TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=300` + flight recorder (6.3) on an
  otherwise-standard run: turns every future failure into a diagnosable dump
  at half the wait.

Each probe is one config line; run them as separate branch commits per the
experiment-log convention so results stay attributable.

## 7. Pointers

- Run history & findings: `experiment_log.md`
- Session record: `session/20260705_105458/`
- Non-colo reference results: W&B `nano-v3-megatron-inference` runs
  `async_non_colocated` (49k) and `async_non_colocated_32k`
- PR 2884: https://github.com/NVIDIA-NeMo/RL/pull/2884
