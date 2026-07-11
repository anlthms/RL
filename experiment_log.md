# Colocated vs Non-colocated Async GRPO — Experiment Log

Companion to `colo_vs_noncolo.txt`. Branch: `async_colo1` (fork `anlthms/RL`).

## Setup

| Item | Value |
|---|---|
| Base code | main @ `f2ab62c2` |
| PR 2884 cherry-pick | `369dfbb3` (async colocated GRPO w/ Megatron inference) |
| Experiment configs | `92c823d4`, launch script `9d5bba51` + brace fix |
| Submodule patches | tene tmp.diffs, uncommitted in worktrees: Megatron-Bridge `mamba_provider.py` (`mtp_mamba_stack_spec`), Megatron-LM `optimizer_config.py` (`muon_use_nesterov`) |
| Model | Nano-v3 30BA3B post-SFT HF ckpt (wdykas iter-0013600) |
| Data | rkirby train-split.jsonl / val-split.jsonl |
| Cluster | OCI-HSG, 8 nodes x 4 GB200, 4h wall clock, container `nemo_rl_0624.sqsh` |
| Scale | 16 prompts/step x 16 gens = gbs 256; TP2 EP4 PP1 CP1 (train and inference) |
| KV buffer | `buffer_size_gb: 80` both modes (watch colocated for OOM; lower if needed) |
| Metric | `validation/total_reward/mean`, W&B project `nano-v3-megatron-inference` |

## Runs (32k phase — current)

| # | Mode | Job ID | Config | W&B name | Submitted | Status | Result |
|---|---|---|---|---|---|---|---|
| 3a | non_colocated | 4218285 | 32k seqlen | `async_non_colocated_32k` | 2026-07-06 ~09:30 PDT | **COMPLETE (4h TIMEOUT)** | ~35 train steps; 7 validations (5-35); train reward mean climbed to ~0.47-0.57 at end; val curve in W&B `async_non_colocated_32k` |
| 3b | colocated | ~~4218286~~ cancelled | 32k seqlen, KV 32GB persist, drain fix | `async_colocated_32k` | 2026-07-06 ~09:30 PDT | FAILED (NCCL watchdog in first post-pause logprobs — 8th consecutive) | — |

User decision (2026-07-06): cut max_total_sequence_length 49152 -> 32768 for BOTH runs (rather than gbs, which doesn't reduce per-microbatch peak). Cascades via interpolation to mb_tokens/max_new_tokens/max_model_len. Both sides rerun; 49k non-colo result kept for reference but the primary comparison is now 32k vs 32k. Reduce further if colo still blocked.

## Runs (49k phase — concluded)

| # | Mode | Job ID | Config | W&B name | Submitted | Status | Result |
|---|---|---|---|---|---|---|---|
| 1a | non_colocated | ~~4074952~~ crashed | `examples/nemo_gym/async_non_colocated_nanov3.yaml` | `async_non_colocated` | 2026-07-05 ~11:35 | FAILED (stale gym venv) | — |
| 1b | non_colocated | ~~4076186~~ crashed | `examples/nemo_gym/async_non_colocated_nanov3.yaml` | `async_non_colocated` | 2026-07-05 ~12:10 | FAILED (train OOM) | — |
| 1c | non_colocated | 4078619 | `examples/nemo_gym/async_non_colocated_nanov3.yaml` | `async_non_colocated` | 2026-07-05 12:19 PDT | **COMPLETE (4h TIMEOUT)** | ~30 train steps; 6 validations (5-30); train reward mean ~0.34-0.36 at end; val curve in W&B |
| 2a | colocated | ~~4074953~~ crashed | `examples/nemo_gym/async_colocated_nanov3.yaml` | `async_colocated` | 2026-07-05 ~11:35 | FAILED (stale gym venv) | — |
| 2b | colocated | ~~4076187~~ crashed | `examples/nemo_gym/async_colocated_nanov3.yaml` | `async_colocated` | 2026-07-05 ~12:10 | FAILED (missing docstring_parser) | — |
| 2c | colocated | ~~4077269~~ crashed | `examples/nemo_gym/async_colocated_nanov3.yaml` | `async_colocated` | 2026-07-05 ~12:40 | FAILED (moe_pad_experts assert) | — |
| 2d | colocated | ~~4077883~~ killed | `examples/nemo_gym/async_colocated_nanov3.yaml` | `async_colocated` | 2026-07-05 ~13:25 | FAILED (empty policy URLs) | — |
| 2e | colocated | ~~4078620~~ crashed | `examples/nemo_gym/async_colocated_nanov3.yaml` | `async_colocated` | 2026-07-05 ~14:15 | FAILED (HF modules cache race) | — |
| 2f | colocated | ~~4079315~~ crashed | `examples/nemo_gym/async_colocated_nanov3.yaml` | `async_colocated` | 2026-07-05 ~14:50 | FAILED (logprob OOM w/ 80GB KV persist) | — |
| 2g | colocated | ~~4088065~~ crashed | `examples/nemo_gym/async_colocated_nanov3.yaml` (KV 48GB persist) | `async_colocated` | 2026-07-05 ~15:30 PDT | FAILED (logprob OOM again: 21.75GB contiguous vs 16GB free + 16GB fragmented) | — |
| 2h | colocated | ~~4096960~~ crashed | `examples/nemo_gym/async_colocated_nanov3.yaml` (KV 80GB + offload) | `async_colocated` | 2026-07-05 ~17:15 PDT | FAILED (torch_memory_saver missing) | — |
| 2i | colocated | ~~4097624~~ crashed | KV 80GB + offload + torch_memory_saver | `async_colocated` | 2026-07-05 ~17:50 PDT | FAILED (NCCL watchdog timeout at first post-pause forward) | — |
| 2j | colocated | ~~4106452~~ crashed | KV 48GB persist + expandable_segments | `async_colocated` | 2026-07-05 ~19:20 PDT | FAILED (NCCL hang at first post-pause forward, same as offload) | — |
| 2k | colocated | ~~4115406~~ crashed | KV 32GB persist, standard allocator | `async_colocated` | 2026-07-05 ~21:45 PDT | FAILED (same NCCL hang — allocator theory dead) | — |
| 4a | colocated | ~~4306397~~ crashed (py-spy captured!) | 32k, 64GB, RL.old fixes | `async_colocated_32k` | 2026-07-08 | FAILED (same hang — but live capture nailed root cause) | — |
| 4b | colocated | ~~4312739~~, ~~4313783~~ | + expert-padding normalization (9a77aa5c) | `async_colocated_32k` | 2026-07-08 | FAILED x2 (cluster fabric flake: init BROADCAST/ALLGATHER timeouts at SeqNum 3-16, pre-code) | — |
| 4c | colocated | 4316967 | same | `async_colocated_32k` | 2026-07-08 ~23:08 PDT | **COMPLETE (4h TIMEOUT — first crash-free colocated run)** | 2 training steps (3rd in flight); 0 validations (never reached step 5); ~75-90 min/step (drain tail + slow colocated decode + buffer refill); batch reward means 0.504, 0.551 |
| 2l | colocated | ~~4134208~~ crashed | KV 32GB persist + drain fix | `async_colocated` | 2026-07-06 ~02:30 PDT | FAILED (NCCL watchdog after logprobs ran 2h+; drain itself took 99 min) | — |

## Timeline

- **2026-07-05 11:35** — Both jobs submitted via `launch_colo_experiment.sh` (SUBMIT_ACCOUNT=nemotron_sw_pre). Both PENDING, 8 nodes each. Note: cluster must have 16 free GB200 nodes for the runs to overlap; if they run serially that is still a fair comparison (same duration each).
- First submission attempt failed: `${1:?...}` usage text contained `}` which corrupted the mode argument; fixed and committed.

- **2026-07-05 ~11:50** — Both jobs RUNNING concurrently (16 nodes granted). non-colo: nvl72028/43/75/150; colo: nvl72055/89/106/147. Ray clusters up; driver logs pending. Health monitors armed on both ray-driver.log files.

- **2026-07-05 ~12:00** — BOTH first-attempt jobs crashed identically at NemoGym spin-up: Gym's `policy_model` server (`responses_api_models/vllm_model/app.py`) died with `ModuleNotFoundError: No module named 'anthropic'`. Root cause: `env.nemo_gym.skip_venv_if_present: true` reused pre-built venvs in `gym_venvs/` (built Jun 25) that predate Gym's `anthropic<=0.109.2` dependency (types-only, added Jun 16 per Gym pyproject — venvs were built from an older Gym checkout). NOT related to colocated/non-colocated modes or PR 2884.
- **Fix**: installed `anthropic==0.109.2` via `pip install --no-deps --target` into all 8 venvs under `gym_venvs/` (`.venv/lib/python3.13/site-packages`; all other deps already present; venv pythons not executable outside the container, hence --target). Persistent fix since venvs live on lustre and are reused.
- **2026-07-05 ~12:10** — Resubmitted: 4076186 (non-colo), 4076187 (colo). PENDING (Priority).

- **2026-07-05 ~12:35** — Second colo attempt (4076187) crashed the same place: `ModuleNotFoundError: No module named 'docstring_parser'` — a dependency of `anthropic` that the earlier `--no-deps` install skipped. Installed `docstring_parser` into all 8 venvs and verified the complete anthropic Requires-Dist list (anyio, distro, docstring-parser, httpx, jiter, pydantic, sniffio, typing-extensions) is now present in every venv.
- Non-colo (4076186) survived: its Gym actor spun up after the fix landed. Colo resubmitted as 4077269. The two runs now start ~40 min apart; each still gets a full 4h window.

- **2026-07-05 ~13:00** — 4077269 (colo) RUNNING. Checked a self-inflicted risk: sharing NRL_MEGATRON_CHECKPOINT_DIR between both jobs is racy — conversion is gated by a bare `os.path.exists(iter_0000000)` check with no locking (nemo_rl/models/megatron/setup.py:460,1446). Benign here: non-colo finished the one-time HF->Megatron conversion (59G iter_0000000 + completion marker) before colo reached the check, so colo loads from warm cache. CAUTION for future reruns: don't start two jobs cold on a shared empty cache dir simultaneously; pre-warm it or use separate dirs.

- **2026-07-05 ~13:15** — Colo attempt 3 (4077269) got past Gym spin-up (all 13/13 servers ready — venv fix confirmed) but failed at colocated engine init: Megatron-LM `TextGenerationController` asserts `moe_pad_experts_for_cuda_graph_inference` when `cuda_graph_impl==local` + EP>1 + transformer_impl != inference_optimized. Non-colo is exempt because its dedicated inference workers use `transformer_impl: inference_optimized`; colocated mode reuses the training model (transformer_engine impl) and toggles cuda graphs on it (megatron_worker.py:398). This is a genuine gap in the PR 2884 path for MoE+cuda-graph configs (the PR's own recipe doesn't enable mcore cuda graphs, so CI never hit it).
- **Fix** (committed): set `policy.megatron_cfg.moe_pad_experts_for_cuda_graph_inference: true` in async_colocated_nanov3.yaml — inference-only flag, already plumbed through nemo_rl setup (setup.py:693). Resubmitted as 4077883.
- Worth reporting upstream on PR 2884: colocated + megatron + MoE + `mcore_generation_config.cuda_graph_impl: local` requires this flag or a clearer error/auto-set.

- **2026-07-05 ~14:00** — Two more failures, both fixed in commit on async_colo1:
  1. **Colo 4077883**: moe_pad fix held; engine initialized ✅ and collection started, but every rollout got HTTP 500 from Gym's `policy_model` proxy: `ZeroDivisionError: integer modulo by zero` at `vllm_model/app.py:580` (`% len(self._clients)`) — the proxy had ZERO backend URLs. Root cause: colocated `MegatronGeneration.__init__` returned early without starting the engine, leaving `dp_openai_server_base_urls=[]` when `_spinup_nemo_gym` consumed it (grpo.py:1075). Non-colo works because the dedicated-policy branch calls `prepare_for_generation()` + collects URLs in the constructor. PR 2884 never hit this: its recipe uses run_grpo.py math (no Gym/HTTP). **Fix**: mirror the dedicated branch in the colocated branch (offload_before_refit -> prepare_for_generation -> collect URLs). Report upstream with the moe_pad issue.
  2. **Non-colo 4076186**: reached training but OOMed in backward on first step (~150GB/184GB allocated, 49k-token packed microbatches, no recompute). **Fix**: `activation_checkpointing: true` in BOTH configs (fairness).
- **2026-07-05 ~14:15** — Resubmitted: 4078619 (non-colo), 4078620 (colo). Monitors re-armed with tighter filters (previous colo monitor flooded on per-request Gym tracebacks).

- **2026-07-05 ~14:45** — Colo 4078620 died at worker init (unrelated to code fixes): transformers trust_remote_code dynamic-module cache race — both jobs started simultaneously, concurrent rewrites of `transformers_modules/.../configuration_nemotron_h.py` served a half-written module (`AttributeError: no attribute 'NemotronHConfig'`). Non-colo 4078619 unaffected (13/13 gym servers, healthy). Mitigation committed: `HF_MODULES_CACHE=/tmp/hf_modules_<mode>` in launch script (node-local, per-mode; env propagates to workers via virtual_cluster.py:197). Colo resubmitted as 4079315.

- **2026-07-05 ~15:20** — Both runs healthy and past all previous failure points. Non-colo 4078619: collection running, first train step pending. Colo 4079315: engine init at construction (URL fix verified — rollouts streaming through megatron HTTP servers, no ZeroDivisionError), collection running. Remaining untested: colo's first train step (pause engine -> train -> resume cycle).

- **2026-07-05 ~17:05** — Colo 4079315 ran ~2h: rollout collection worked end-to-end (trajectory versions 0-4 generated), but at the FIRST training pause every rank OOMed in get_logprobs (~162GB resident incl. the 80GB KV buffer, which persists on GPU through training with kv_cache_management_mode=persist; logprob fwd needed +10.9GB, train fwd +21.8GB). Zero training steps completed. This is the memory-competition cost the experiment probes — 80GB KV is infeasible colocated at this scale.
- **Fix** (user pre-approved): colocated buffer_size_gb 80 -> 48 (frees 32GB for the training phase). Non-colo keeps 80GB on its dedicated inference nodes — that asymmetry is part of what colocated vs non-colocated means. Resubmitted as 4088065.
- Alternative if 48GB still OOMs or throttles generation: kv_cache_management_mode: "offload" (keeps big buffer during generation, offloads to CPU during training; costs transfer time each step).

- **2026-07-05 16:20 PDT** — Non-colo 4078619 finished its FULL 4h window (Slurm TIMEOUT as designed). ~30 training steps, validations every 5 steps (last at step 30), training reward mean drifted 0.34->0.36. This is the reference result; validation/total_reward/mean curve in W&B run async_non_colocated.
- Colo 4088065 (KV 48GB) healthy: 13/13 servers, engine init OK, collection in progress (~90-130 s/it, slower than 80GB attempt as expected). First training pause pending (~2h after start). NOTE: earlier timeline timestamps in this file were estimated loosely; job-relative elapsed times are authoritative.

- **2026-07-05 ~17:15 PDT** — Colo 4088065 (48GB persist) OOMed at the same point: training-phase get_logprobs asks for 21.75GB contiguous; 140GB allocated, 16GB free, 16GB fragmented. Conclusion: ANY persist-mode buffer at this scale starves the training phase.
- **Fix**: kv_cache_management_mode: "offload" — KV cache moves to CPU during the training pause (frees the whole buffer), so the buffer is restored to the inherited 80GB for generation parity with non-colo. Resubmitted as 4096960. Note for writeup: colocated Nano-v3 @49k seqlen on 4xGB200 nodes REQUIRES offload mode; persist is infeasible — a substantive finding about the colocated design space.

- **2026-07-05 ~17:50 PDT** — 4096960 failed fast at engine init: offload mode asserts `torch_memory_saver` or UVM (dynamic_context.py:455); neither in the Jun-24 container venvs (torch_memory_saver was added to uv.lock after the container build) and unified_memory_level is not plumbed through nemo-rl config. Fix: pre-staged the aarch64 wheel on lustre (compute nodes may lack egress) and added SETUP_COMMAND to the launch script — ray.sub runs it on every node before Ray starts, installing the wheel into the MegatronPolicyWorker venv. Resubmitted as 4097624.

- **2026-07-05 ~18:20 PDT** — 4097624: SETUP_COMMAND verified on all nodes (torch-memory-saver installed pre-Ray), 13/13 gym servers, engine initialized in OFFLOAD mode at 80GB, collection running. Next milestone: first training pause (~2h in) — offload/onload cycle test.

- **2026-07-05 ~20:15 PDT** — 4097624 (offload mode) ran 2h12m: collection fine, engine paused for training, then EVERY rank's first get_logprobs forward hung in NCCL collectives (EP ALLTOALL ~8.4GB, TP ALLGATHER) until the 600s watchdog SIGABRTed. Hypothesis: torch_memory_saver's pause unmaps the KV region while inference CUDA graphs still hold pointers, corrupting stream state -> offload mode ruled out for now (worth reporting alongside the other PR 2884 findings).
- **New approach**: back to persist@48GB but fix the REAL 48GB failure — fragmentation (21.75GB contiguous ask vs 16GB free + 16GB reserved-unalloc) — with PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True via policy.megatron_cfg.env_vars (targets exactly the megatron workers; the documented weight-transfer slowdown is irrelevant colocated — no refit). Resubmitted as 4106452.
- Escalation ladder if this fails: halve sequence_packing.logprob_mb_tokens / train_mb_tokens (24576) for colo only — affects throughput not training math.

- **2026-07-05 ~21:45 PDT** — 4106452 (persist@48 + expandable_segments): collection completed (1h54m), engine paused, then NCCL watchdog timeouts in the first get_logprobs forward — SAME signature as offload mode. Pattern across all four training-pause attempts: standard allocator -> clean OOM (persist@80, persist@48); VMM-backed allocator (torch_memory_saver offload, expandable_segments) -> NCCL collective deadlock. Conclusion: VMM allocators break NCCL here (registration/transport incompatibility); the real requirement is simply more free memory with the standard allocator.
- Also ruled out: engine-not-quiesced theory (megatron_worker._sleep does pause->suspend with barriers before training); in_flight_weight_updates drain gap noted as a theoretical colocated hazard (trajectory_collector.py:495 skips draining) but the engine suspend covers it.
- **Next**: persist@32GB, standard allocator (job 4115406) — ~32GB free for the 21.75GB ask, no VMM. Generation pays with a smaller KV cache; that cost is part of the colocated tradeoff being measured.

- **2026-07-06 ~02:30 PDT** — 4115406 (32GB standard allocator) hung NCCL at the first training pause too — allocator theory falsified; the hang is intrinsic. Revised root cause (fits all evidence): the async engine is paused MID-DECODE; ranks pause at different decode-step boundaries, leaving incomplete decode collectives (the tiny 73728-numel alltoalls seen in the dumps) queued on the same NCCL communicators training then uses -> deadlock. PR 2884's recipe never hits this because its 2k-token rollouts leave the engine idle at pause; our 49k agentic rollouts are always mid-step. The earlier standard-allocator OOMs simply crashed before the deadlock could express.
- **Code fix** (committed): trajectory_collector.prepare_for_refit now drains in-flight generations when colocated (skipping the in_flight_weight_updates fast path). Cost: each training step waits for the rollout tail — that latency is a REAL colocated cost and will show in the step-time comparison. Resubmitted as 4134208 (32GB persist kept — memory finding still stands).
- PR 2884 feedback list now: (1) moe_pad_experts assert with mcore cuda graphs, (2) empty dp_openai_server_base_urls in colocated ctor, (3) in-flight pause deadlock with long rollouts, (4) persist-mode KV memory infeasibility at 49k seqlen (needs small buffer).

- **2026-07-06 ~08:50 PDT** — 4134208 (drain fix): drain WORKED (engine fully idle at pause; in-flight tail took 99 min — itself a colocated cost problem), but get_logprobs still hit NCCL watchdog timeouts ~2h12m after it started. Falsified so far: VMM allocators, mid-decode pause, (likely) microbatch-count mismatch (driver packs bins globally in shard_by_batch_size).
- **Leading unified hypothesis**: not a deadlock — pathological SLOWNESS. Logprob forward under ~150GB residency grinds (allocator retry/fragmentation churn); EP/TP ranks progress unevenly; a waiting peer exceeds the 600s NCCL watchdog. Explains why "hang" points/collective sizes vary run to run and why plain-OOM vs hang tracks how much memory was left.
- **COLO ATTEMPTS PAUSED** pending decision. Options: (A) buffer 16GB (max training headroom, slower generation); (B) raise torch/NCCL PG timeout (cheap, decisive diagnostic: if slowness, steps complete and reveal true step time); (C) cut max_total_sequence_length for BOTH runs (changes workload; big memory relief); (D) escalate findings to PR 2884 author. Recommended: B (+A if B reveals impractical step times).

## RL.old discovery (2026-07-08)

The previous checkout (/lustre/fsw/portfolios/nemotron/users/anthomas/RL.old) is a PRIOR campaign of this same experiment, with its own issue ledger (examples/nemo_gym/colocated_vs_noncolocated.md, issues 1-12) and five uncommitted fixes. Its colocated run 3988899 completed step 1 end-to-end (train -> wake -> checkpoint -> wake) with inference CUDA graphs OFF. Re-enabling graphs (issue #11, for 10x decode) caused the same 10.88GB first-logprobs OOM we saw at 80GB (issue #12); their 64GB retry is where the ledger ends — our graphs-on 48/32GB attempts explored past it and found the NCCL hang. All five fixes ported to async_colo1 (commit f0fb5368): timeout-clock restart, checkpoint pause/resume, _wake barrier-before, pre-hook disable in prepare_for_generation, param_sync_func guard. Colo buffer set to 64GB (their issue-#12 value).
- Revised hang suspects: CUDA-graph/expert-padding residue on the shared module (graphs-off never hung in either campaign). Graphs-off is NOT an acceptable fallback (user decision 2026-07-08: 10x slower decode defeats the colocated arm's purpose). If the graphs-on hang recurs at 64GB, the path is interactive debugging of the CUDA-graph/expert-padding residue (async_colo_debug_guide.md S2/S3) — the debug kit's cheap probe `inference_moe_token_dispatcher_type=alltoall` and per-rank module-state inspection are the next moves.
- Also learned (issue #6): max_total_sequence_length too small -> empty completions -> engine crash; debug config bumped to 16384.

## ROOT CAUSE FOUND (2026-07-08) — py-spy capture of live hung job 4306397

Job 4306397 (ported fixes + 64GB, graphs on) hung again at first logprobs — but we captured py-spy dumps from the still-live hang (debug_colo_capture.sh). Findings:
- The inference event-loop thread ("Thread-3 run_loop") is IDLE on hung ranks — engine properly quiesced (S4 falsified).
- Hung ranks spin on a CUDA sync inside the logprob forward at MoE ROUTING (topk_routing_with_score_function, moe_utils.py:808), Ray task = get_logprobs on the actor's AsyncIO thread.
- Mechanism (S3 confirmed): the generation controller calls set_decode_expert_padding per DECODE STEP from rank-local state (using_cuda_graph_this_step); it mutates the SHARED model config (moe_pad_expert_input_to_capacity, moe_expert_capacity_factor) and every dispatcher's drop_and_pad. Engine suspend freezes EP peers in DIFFERENT padding states -> training forward runs shape-mismatched expert dispatch -> NCCL deadlock. Only possible on the graphs-on path (padding only toggles when cuda graphs are used) — explains the graphs-on/graphs-off split across both campaigns.
- FIX (commit 9a77aa5c): finish_generation now calls set_decode_expert_padding(model, False) after toggling graphs off, so all ranks enter the training phase with uniform unpadded dispatch; generation re-enables per step on resume. Resubmitted as 4312739.

## Final colo diagnosis (2026-07-07, superseded by the above)

The first post-pause logprob forward hangs in NCCL collectives across EVERY controlled variable: allocator (standard / expandable_segments / torch_memory_saver), KV buffer (80/48/32GB), sequence length (49k/32k), engine drained or not. Memory pressure ruled out (32k@32GB leaves ~50GB free). Remaining suspects are colocated mode-switch state bugs in the PR 2884 path: inference vs training MoE dispatcher (nccl vs alltoall) not switched back, CUDA-graph teardown (toggle_cuda_graphs) incomplete on some ranks, or set_decode_expert_padding left inconsistent. Requires interactive debugging (py-spy/gdb on a live hung rank) or upstream input — not more batch cycles. COLO ARM STOPPED after 8 attempts.

**Working answer to the experiment question at this scale/workload**: async colocated GRPO with Megatron inference (PR 2884) is not yet functional for long-generation NeMo-Gym workloads; non-colocated is stable (two clean 4h runs). Even setting the hang aside, colo showed structural costs: buffer fill took ~2.5h of the 4h window (32GB KV throttling generation concurrency), and the correctness-required drain at each training pause cost 69-99 min due to rollout stragglers.

- **2026-07-07 ~03:30 PDT** — Non-colo 32k (4218285) COMPLETE: full 4h (TIMEOUT), ~35 training steps, validations every 5 through step 35, training reward mean rising into 0.47-0.57 by the end (vs ~0.34-0.36 for the 49k run at step 30 — note: different max lengths mean these training-reward numbers aren't directly comparable; use validation/total_reward/mean in W&B for any cross-run reading).

## RESULT (2026-07-09)

Head-to-head at 32k seqlen, identical data/model/gbs, 4h each on 8x4 GB200:
- **non-colocated: ~35 training steps, 7 validations** (W&B async_non_colocated_32k)
- **colocated: 2 training steps, 0 validations** (W&B async_colocated_32k) — first crash-free colocated run after 4 root-cause fixes

Verdict at this scale/workload: **non-colocated wins by ~15x in optimizer steps per GPU-hour.** The colocated implementation's per-step costs dominate: (1) drain-before-pause (correctness-required; 30-55 min rollout tail per step at 32k), (2) ~10x slower decode through the training-impl module (no inference_optimized kernels, flash_decode disabled — RL.old issue #12 measured 640 vs 7400 tok/s/GPU), (3) buffer refill gating each step. The original hypothesis (colocated more efficient via fresher weights and no refit) is not supported: the freshness advantage is real but drowned out by generation throughput and pause costs.

## PERF PHASE A (2026-07-09) — drain → engine suspend/resume

Goal: remove the drain-before-pause cost (30-99 min/step) by relying on the Megatron dynamic
engine's own suspend()/resume(), which already retains in-flight requests across the training
pause and reaches a barrier-synced idle PAUSED state. The drain was added under the mid-decode-
pause deadlock hypothesis, later falsified (real cause = expert-padding divergence, fixed
9a77aa5c). Plan in `colo_perf_plan.md`.

Code (branch `async_colo1`, also on `colo-perf/phaseA-no-drain`):
- `02bb4b3b` feat: drop colocated drain in `trajectory_collector.prepare_for_refit` (colocated
  async now takes the in-flight path); add `[NRL-COLO-BACKEND]` banner (driver + per-rank)
  confirming rollouts are served by the native Megatron DynamicInferenceEngine; add
  `[NRL-COLO-PHASEA]` retained-tail count at each pause.
- `6ab28650` chore: W&B run name → `async_colocated_32k_phaseA`.

| # | Mode | Job ID | Config | W&B name | Submitted | Status | Result |
|---|---|---|---|---|---|---|---|
| PA | colocated | ~~4420367~~ cancelled | 32k, no-drain (engine suspend/resume) | `async_colocated_32k_phaseA` | 2026-07-09 ~17:15 PDT | CANCELLED (lost scheduling race to PA'; auto-cancelled to avoid double-run) | — |
| PA' | colocated | 4513519 | same code/config as PA | `async_colocated_32k_phaseA` | 2026-07-10 | **RUNNING** (nemotron_sw_post; won race) | in progress |

**PA' backend confirmed (2026-07-10, ~11 min in):** `[NRL-COLO-BACKEND]` across 32 ranks shows
native Megatron `DynamicInferenceEngine` serves generation (NOT vLLM); per-rank
`transformer_impl=transformer_engine` (confirms training-impl reuse = the slow-decode root cause);
`kv_cache_management_mode=persist`; NeMo-Gym proxy → Megatron OpenAI URL. Backend assumption for
Phase A verified on hardware.

**PA' RESULT — Phase A HOLDS (2026-07-10, ~1h30m in): drain-before-pause eliminated, NO deadlock.**
- `[NRL-COLO-PHASEA] retained 16 → 29 → 44 in-flight generation group(s) across refit (not drained)`
  — the engine suspend/resume carries the stragglers across the training step; they continue with
  updated weights. Suspend-and-resume works as designed.
- `Ready for refit (took 0.00s)` — the drain that cost **30-99 min/step** is now **instant**.
- First (and 2nd, 3rd, 4th) post-pause logprob forward did NOT NCCL-deadlock. The "drain is
  redundant" hypothesis is CONFIRMED on hardware: the engine's barrier-synced PAUSED quiesce +
  the expert-padding fix (9a77aa5c) are sufficient without the collector drain.
- **Step wall-clock ~37-39s/step** (train_data_step{1,2,3}.jsonl mtimes 10:01:55 / 10:02:32 /
  10:03:11) vs the 75-90 min/step baseline — while consuming the pre-filled buffer.
- Buffer trajectory 63→50→36→20: training now drains ~14-16 groups/step faster than slow colocated
  decode refills → **Issue 1 (slow decode) is now the binding constraint**, exactly as the plan
  predicted ("buffer refill gating each step"). Initial buffer fill to 63 took ~1h26m (08:35→10:01)
  — the slow-decode cost. Sustained step rate will settle to the generation refill rate.
- Note: per-step training trajectories dumped to `logs/exp_022/train_data_step*.jsonl` (another
  place to inspect actual rollouts).
**PA' FINAL (2026-07-10, 4h TIMEOUT — sacct FAILED = shutdown noise, not a crash):**
- **5 training steps + first validation (at step 5)** in 4h. vs prior colo-with-drain: **2 steps, 0
  validations**. vs non-colo: ~35 steps, 7 validations. So Phase A: **2.5x more steps AND reached
  validation for the first time** in a colocated run.
- Step timing tells the story: steps 1-4 ran at **~38s each** (consuming the initial buffer
  backlog: dumps 10:01:55/10:02:32/10:03:11/10:04:02), then step 5 took **~2h22m** (10:04→12:26) —
  the generation-bound sustained rate once the buffer emptied. Validation alone took ~76min (4566s)
  because the val set also decodes through the slow colocated engine.
- **Verdict: Phase A fully fixes Issue 2 (drain), but does NOT make colo competitive alone.** With
  the drain gone (0.00s), throughput is now entirely **generation-bound** (~2h22m/step), i.e. Issue
  1 (slow decode ~10x) is the sole remaining blocker. Colo is still ~7x behind non-colo, now purely
  on decode kernels. Validation reward at step 5 is in W&B `async_colocated_32k_phaseA`.
- **Implication:** the "non-colo wins 15x" verdict is now obsolete (it was drain-dominated). The
  updated gap is ~7x and is 100% Issue 1. Closing it requires Issue 1 option 1 (inference-optimized
  model instance for generation + local refit) — Phase A has isolated it as the only lever left.

Dual-submit note: PA (pre) and PA' (post) are identical jobs on two accounts to improve
scheduling odds. Combined monitor cancels whichever is still pending once the other reaches
RUNNING (avoids double GPU burn + W&B/checkpoint-dir collision — same run name/dir).

Watch on this run:
1. `[NRL-COLO-BACKEND]` in ray-driver.log / ray-worker logs — confirms native Megatron engine
   (user asked to verify backend is not vLLM).
2. `[NRL-COLO-PHASEA] retained N in-flight ...` at each pause — measures the tail no longer drained.
3. **DECISIVE:** first post-pause logprob forward — does it NCCL-deadlock (Phase A hypothesis
   wrong; the crash-free 4316967 had BOTH drain and padding fix) or proceed? If it proceeds,
   step time should drop far below the 75-90 min/step baseline.

## Monitoring notes

- Driver log: `<jobid>-logs/ray-driver.log` in repo root (written by `ray.sub`).
- Attach script `<jobid>-attach.sh` appears ~5 min after job start (interactive use only; not needed — COMMAND runs automatically).
- Watch colocated run early steps for CUDA OOM from the 80 GB KV buffer competing with optimizer state.

## Results

TODO after runs complete: compare `validation/total_reward/mean` at equal wall-clock in W&B.

---

## E1 — single dual-mode model (colocated fast decode without a second model)

- **Date:** 2026-07-10. **Branch:** `async_colo1`. **Commit:** `77978a8c`. **Job:** `4563657`
  (SUBMIT_ACCOUNT=nemotron_sw_post, 8 nodes, 4h).
- **Config:** `examples/nemo_gym/async_colocated_singlemodel_nanov3.yaml` — the ONE colocated model
  built as `transformer_impl=inference_optimized` (dual-mode: fast kernels under InferenceMode,
  trainable TE fallback for train/logprob). No second model, no per-step weight sync.
- **Hypothesis:** replaces the two-model `colo_decode_fast_plan.md`. If the inference_optimized layers
  train via their TE fallback, decode gets the fast kernels for free and the ~10x colo decode penalty
  closes with a single model. Config-compat verified: train == gen parallelism (TP2/PP1/EP4/ETP1),
  router fp32, dropless, num_groups=null (nano-v3 n_group=1). See `colo_single_model_plan.md`.
- **Watch (gates):**
  1. **Build gate** — model builds with inference_optimized + init_optimizer (guard relaxed via
     `allow_inference_optimized_training`). Fails fast at setup if a constraint is violated.
  2. **`[NRL-COLO-BACKEND] ... transformer_impl=inference_optimized`** in ray-driver.log.
  3. **Decode gate** — tok/s/GPU approaches ~7400 non-colo (vs ~640 today).
  4. **Train gate** — optimizer steps advance, loss/grad finite, reward tracks non-colo baseline.
- **Status:** PENDING (job 4563657, nemotron_sw_post; original 4556249 on _pre cancelled to avoid run-name collision). Result: TODO.

### E1 outcome (job 4563657) → E1b (job 4564982)

- **BUILD GATE PASSED.** The single `inference_optimized` model built and the DynamicInferenceEngine
  wrapped it on all 32 ranks: `[NRL-COLO-BACKEND] ... transformer_impl=inference_optimized`. The
  relaxed guard worked; config resolved as intended (TP2/PP1/EP4/ETP1, dropless, router fp32).
- **Crash at setup** in `prepare_refit_info()` → `ValueError: Cannot determine parallelism type for
  module 'InferenceColumnParallelLinear' at 'decoder.layers.1.mlp.shared_experts.linear_fc1.weight'`.
  Root cause: megatron-bridge's AutoMapping registry omits `InferenceColumnParallelLinear` (it has
  the Row and LayerNormColumn variants). Invisible non-colo because the inference_optimized model is
  only a refit *target* there; here it is the export *source*.
- **Fix (commit 7572f75a):** register `InferenceColumnParallelLinear`→`column` when
  transformer_impl=inference_optimized. Resubmitted as **E1b job 4564982** (nemotron_sw_post).

### E1b RESULT (job 4564982) — THESIS VALIDATED

The single dual-mode `inference_optimized` model trains AND generates in the colocated Phase-A loop.
- **Decode throughput (Generation Worker Group): ~5870–7685 tok/s/GPU** (settling ~7685) vs old
  colocated **~640** and non-colocated target **~7400**. → **~12× over old colo, at/above non-colo parity.**
- **Total step time: ~23.4s** (after a 121s first-step warmup) vs old colocated **~2h22m/step**.
- E2E ~4238–5320 tok/s/GPU; Policy Training ~7500–9800 tok/s/GPU.
- Colocated Phase-A refit loop cycles cleanly (`retained 16→32→48 in-flight groups across refit`).
- Memory comfortable: ~94–126 GB of 184 GB (optimizer offloaded during gen). No OOM.
- No second model, no per-step weight sync, no inference-model offload — generation reads the exact
  tensors training updates.

**Gates:** build ✓ · backend banner (inference_optimized) ✓ · decode-speed ✓ (7685 vs 640/7400) ·
train steps advance ✓. Remaining: E2 reward-curve correctness (validation reward tracking non-colo
baseline) — needs more wall-clock (val_period=5).

**Conclusion:** the single-model approach closes the entire ~10× colocated decode gap and makes the
two-model `colo_decode_fast_plan.md` unnecessary. Net change: guard opt-in + one config + one
missing AutoMapping registration.

### E1b full-run wrap (job 4564982, ran full 4h → TIMEOUT, not a crash)

- **Decode throughput sustained** across the whole run: ~7000–8277 tok/s/GPU (one 3005 outlier at a
  refit boundary), consistently at/above the ~7400 non-colo target. Not a first-step artifact.
- **Non-validation step time 16–36s** (121s first-step warmup). Confirms generation is no longer the
  bottleneck.
- **E2 (correctness) — sane and functioning:** validation ran at steps 5 and 10:
  - step 5:  Accuracy 0.294, Avg Reward 0.4336
  - step 10: Accuracy 0.280, Avg Reward 0.4844
  Non-degenerate rewards, training working through the TE fallback. Step-15 validation was mid-run at
  the 4h wall-clock cutoff.
- **Caveat:** only ~15 optimizer steps + 2 validation points in 4h because each validation costs
  ~43 min (val_batch_size=1000, full generation) — that overhead, NOT our change, capped step count.
  A rigorous reward-vs-non-colo curve needs a longer run and/or lighter validation cadence.

**Overall:** decode-parity thesis proven and sustained; training correct on the sampled points.
Remaining open item is a longer apples-to-apples reward comparison vs non-colo baseline a28k1rl8.

### E2b — generation-concurrency profiling (why non-colo validation is faster)

- Instrumented the Megatron DynamicInferenceEngine (opt-in NRL_GEN_TRACE, patched from
  megatron_worker.py — submodule not edited) to log per-step in-flight concurrency
  (active/paused/waiting requests, active tokens).
- Two matched validate-at-start runs, identical 256-prompt val workload (1024 seqs),
  same max_new_tokens=32768 so the KV/concurrency ceiling matches production:
  - non-colo: job 4607214 (nemotron_sw_pre)
  - colo single-model: job 4607216 (nemotron_sw_post)
- Hypothesis: colo shows lower steady-state active_reqs (KV headroom / single serving
  coordinator) → fewer sequences decode in parallel → longer validation wall-clock at
  equal per-token speed. Kill each once [NRL-GEN-TRACE] stabilizes during validation.

#### E2b colo result (job 4607216)
- 8319 trace lines during validation: active_reqs ~29-64 per replica (rank 0),
  **paused=0 and waiting=0 on EVERY line**. The colocated engine is NEVER KV-limited
  during validation — it is request-starved (~30-60 concurrent seqs), not capacity-bound.
- So colo validation slowness is NOT decode speed, NOT KV/memory — it is low effective
  generation concurrency (few sequences in flight).
- Job died near 4h wall from NeMo-Gym HTTP ClientOSError (validation genuinely ~43min).
- Next: non-colo comparison (per-rank trace) to see if it packs more per replica or
  spreads across more replicas. non-colo resubmitted on nemotron_sw_post.

#### E2b non-colo result (job 4653968)
- All-rank trace: 11 distinct gen ranks actively decoding CONCURRENTLY
  (ranks 0,1,2,3,4,5,8,11,12,13,14), each ~13-46 active_reqs, paused=0/waiting=0.
- So non-colo runs validation across MANY DP replicas in parallel (pool-wide
  concurrency = sum over replicas ≈ few hundred sequences in flight).
- Contrast with colo rank-0 ~30-64. Decisive test (job 4656156, all-rank colo trace):
  does colo spread across ranks too, or funnel to one replica?

#### E2b colo per-rank result (job 4656156) + reframe
- Colo ALSO spreads across ~11 ranks (0,2,4,7,9,12,16,18,20,24,31), ~20-67 active_reqs each,
  paused=0/waiting=0. So concurrency is SIMILAR to non-colo — funnel hypothesis REFUTED.
- Colo rank0 (from 4607216): 166,360 engine steps but only 308 finished over ~43min, rarely
  idle (active_reqs ~30-64 steady). => sequences run to ~20-32k tokens; engine decodes
  continuously but completes ~8x slower per replica than non-colo (~55 vs ~7 finished/min/replica).
- => The gap is per-GPU DECODE RATE at a given concurrency, not KV, not batch size, not idle.
- Added steps_per_s to trace (commit) to measure tokens/s/replica = active_reqs * steps_per_s.
  Rerunning both to steady state: non-colo + colo.

#### E2b decode-rate measurement (steps_per_s trace)
- NON-COLO (job 4658796): validation window steady state = active_reqs ~41, steps_per_s ~90/s,
  val time 472s. => ~41 x 90 ≈ 3700 decode-tokens/s/replica (~1850 tok/s/GPU).
- Iterating faster per user guidance: all jobs on nemotron_sw_post; measuring colo decode rate
  on a 2-node (single TP2xEP4 replica) run (job 4674235) since per-replica rate is pool-independent.
  Kill early once steps_per_s stabilizes.

#### E2b ROOT CAUSE (decode step-rate, 2-node fast iteration on post)
- COLO (2-node, 1 replica, job 4674235): active_reqs ~110, steps_per_s ~12.9 => ~1458 tok/s/replica
  = ~730 tok/s/GPU.
- NON-COLO (job 4658796): active_reqs ~41, steps_per_s ~90 => ~3690 tok/s/replica = ~1845 tok/s/GPU.
- Colo runs a BIGGER decode batch (110 vs 41) yet gets <half the per-GPU throughput => colo decode
  step is intrinsically ~2.5x less efficient per GPU. Both use CUDA graphs (colo buckets to 276,
  non-colo to 348), so it's not graphs-off. Not concurrency, KV, memory, or idle (all ruled out).
- CONCLUSION: non-colo validation is faster because the DEDICATED inference model decodes ~2.5x
  more efficiently per GPU than the colocated engine (which wraps the TRAINING model: DDP/Float16
  wrappers, optimizer attached-but-offloaded, training memory layout). Combined with a warm
  dedicated pool, this yields the ~5x validation wall-clock gap (472s non-colo vs ~2600s colo).
- Note: colo's single-model decode hit 7685 tok/s/GPU at HIGH rollout concurrency (E1b) but
  degrades at validation's lower per-replica batch — the per-step overhead dominates there.
- Deeper micro-cause (DDP-wrapper overhead vs graph/kernel differences) would need nsys profiling.
