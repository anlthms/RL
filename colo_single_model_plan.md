# Plan: colocated decode speed via a single dual-mode model (no second model)

Alternative to `colo_decode_fast_plan.md`. That plan closes the ~10× colocated decode gap by
building a **second** `inference_optimized` model and syncing weights into it each step. This plan
gets the same fast kernels from **one** model, avoiding the weight-sync, the parity check, and the
extra weight copy. All experiments stay on branch `async_colo1`.

## Key insight (verified against Megatron-LM source)

The `inference_optimized` layers are **dual-mode**, not inference-only. Each one runs the fast
inference kernel when the engine is active and falls back to the trainable Transformer-Engine path
otherwise, selected at runtime — no separate model needed:

- MoE dispatcher: `moe_layer.py:628-635` picks `_inference_token_dispatcher` iff
  `InferenceMode.is_active()`, else `_training_token_dispatcher`. Both are built at `__init__`.
- Parallel linears: `inference_layers.py:107,218,329,479` — `if self.training: return super().forward(x)`.
- Grouped MLP: `experts.py:1109` — inactive `InferenceMode` → `super().forward()` (trainable `TEGroupedMLP`).
- Router: `router.py:838-841` — inactive `InferenceMode` → parent `TopKRouter.forward` (full training router).
- The `Inference*` layers **register no new parameters** — same names/shapes as the TE parent.
- `InferenceMode` is a global flag the dynamic engine sets on `resume()` and clears on `suspend()`
  (`dynamic_engine.py:300/795/846`). Phase-A suspend/resume already toggles it for us.

So the plan's premise that "the generation and training models **must** be two separate instances"
is a NeMo-RL self-imposed guard (`megatron_policy_worker.py:318-324` from PR #2355), not a Megatron
constraint. One `inference_optimized` model trains through the TE fallback and generates through the
fast path.

## Config compatibility (verified) — mostly already satisfied

`inference_optimized` imposes constraints (`transformer_config.py:1336-1364,2552-2557`). Checked
against the colocated **training** config (`grpo_nanov3.yaml`):

| Constraint | Required | Training config today | OK? |
|---|---|---|---|
| expert_tensor_parallel_size | 1 | 1 | ✅ |
| moe_expert_capacity_factor | None (dropless) | unset → None | ✅ |
| moe_router_dtype | fp32 | fp32 | ✅ |
| RMSNorm / no bias / no GLU | required | satisfied (non-colo inference on same ckpt works) | ✅ |
| moe_router_num_groups | None | nano-v3 `n_group=1,topk_group=1` ⇒ null is equivalent | ✅ |
| pipeline_model_parallel_size | — | 1 (train, via actually_run override) == 1 (inference) | ✅ |
| expert_model_parallel_size / TP / ETP | matched | train TP2/PP1/EP4/ETP1 == gen TP2/PP1/EP4/ETP1 | ✅ |

The effective colocated parallelism (after `actually_run_nanov3.yaml` overrides `grpo_nanov3.yaml`
to PP=1, EP=4) is **identical between the training model and the generation config** — TP2, PP1,
EP4, ETP1. So there is no parallelism mismatch; every `inference_optimized` constraint is already
satisfied by the training config. The build is expected to succeed; the empirical question reduces to
decode throughput and training correctness, not feasibility.

## Changes (small)

1. **Relax the guard** (`megatron_policy_worker.py:318-324`): allow `transformer_impl=inference_optimized`
   with `init_optimizer=True` when opted in via `megatron_cfg.allow_inference_optimized_training: true`.
   Keep the assert for everyone else.
2. **New config** `examples/nemo_gym/async_colocated_singlemodel_nanov3.yaml` (extends
   `async_colocated_nanov3.yaml`), setting on `policy.megatron_cfg`:
   - `transformer_impl: inference_optimized`
   - `allow_inference_optimized_training: true`
   - `inference_moe_token_dispatcher_type: nccl`, `inference_grouped_gemm_backend: vllm`
   - `moe_router_num_groups: null`, `moe_router_group_topk: null`
   - `moe_pad_experts_for_cuda_graph_inference: false` (forbidden with inference_optimized —
     `text_generation_controller.py:588-591`; also drops the Phase-A `set_decode_expert_padding` hack)
   - keep `activation_checkpointing: true` (training memory)
3. **No weight sync, no second model, no offload/onload of an inference model, no parity check** —
   generation reads the exact tensors training just updated.

## Experiments (all on `async_colo1`, logged to `experiment_log.md` + TSV)

- **E1 — single-model build + decode throughput.** Launch the single-model colocated config.
  Success gate 1: model builds (PP=2 + inference_optimized survives config validation & engine init).
  Success gate 2: decode tok/s/GPU approaches the ~7400 non-colo number (vs ~640 today). This alone
  proves the thesis and needs no new sync code.
- **E2 — training correctness end-to-end.** Same run continues: confirm optimizer steps advance,
  loss/grad finite, and the training-reward curve tracks the non-colo baseline from the same ckpt.
  Because it is one model, "fresh weights for generation" is automatic.
- **E3 — throughput/economics.** Compare optimizer-steps/4h and sustained step time vs the ~2h22m/step
  colocated baseline and the non-colo cadence.

Fallback if E1 fails at build/engine init (e.g. PP=2 unsupported by fast kernels): fall back to the
two-model design in `colo_decode_fast_plan.md`, or retry the single model at PP=1.

## Why this is preferable to two models

- Deletes the "one genuinely new mechanism" (device-local weight sync) and its parity check.
- Saves a full BF16 weight copy (~7.5 GB/GPU) and the per-step offload/onload of a second model.
- Sidesteps the plan's #1 risk (CUDA-graph lifecycle across inference-model offload) — nothing to offload.
- Your CPU-offload doubt resolves: with one model there is no separate inference model to offload;
  the training weights *are* the generation weights.

## Result (E1b, job 4564982, 2026-07-10)

VALIDATED. Decode throughput ~7685 tok/s/GPU (Generation Worker Group) vs ~640 old-colo / ~7400
non-colo target — full parity, ~12× gain. Step time ~23.4s vs ~2h22m old-colo. Single model trains
+ generates in the colocated loop; no second model, no weight sync. Two fixes were needed beyond the
config: relax the training-worker guard (opt-in flag), and register `InferenceColumnParallelLinear`
with megatron-bridge's AutoMapping (shared_experts.linear_fc1 export). Reward-curve correctness (E2)
pending more wall-clock.
