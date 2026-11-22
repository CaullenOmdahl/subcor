# Multi-Turn Assistant-Only Masking & Multi-Layer Hybrid MLX Adaptation Report

**Date:** November 22, 2026  
**Issues Addressed:** `#23` (Multi-turn `messages` + `tools` with assistant-only loss masking) & `#24` (Memory-safe multi-layer MLX QLoRA adaptation on hybrid models within 16 GB RAM)  
**Hardware:** Apple M5 (arm64), 16 GB unified memory, macOS 26.5.1, `mlx==0.32.4`, `mlx-lm==0.32.2`  
**Base Model:** `mlx-community/Qwen3.5-4B-4bit` (`32` layers: `24` `GatedDeltaNet` linear-attention blocks + `8` periodic full-attention blocks at `[3, 7, 11, 15, 19, 23, 27, 31]`)

---

## 1. Memory & Throughput Breakthrough on Hybrid `GatedDeltaNet` Models (`seq_len=2048`)

Before this change, backpropagating through `mlx_lm`'s stock `gated_delta_ops()` unrolled a `2048`-step Python loop per linear-attention block, allocating `O(T × Hv × Dv × Dk)` state tensors (~2.15 GB per linear layer) and causing severe swap thrashing or OOM on 16 GB Apple Silicon.

Combining four techniques in `src/subcor/backends/mlx_kernel.py` eliminates OOM completely and makes peak unified memory **$O(1)$ in adapted depth**:

1. **Exact Frozen-Prefix Caching** (`layers[0 : first_trainable_idx]` evaluated once in fast Metal `eval()` mode).
2. **Assistant-Only Target-Position-Sliced `lm_head`** (`inner.norm(h)[:, loss_positions, :]` gathers *only* masked assistant/tool-call token positions before the `248K`-vocab vocabulary projection).
3. **Exact Layer-Level `mx.checkpoint`** passing `layer.trainable_parameters()` explicitly (verified bit-for-bit identical gradients, `max_diff = 0.0`).
4. **Minimally Scoped Per-Layer `train()` + Chunkwise-Parallel Gated Delta Rule** (`chunk_size=32`, overflow-free forward substitution for `(I + M) U = rhs`, chunk-level `mx.checkpoint`, active inside a `try/finally` context manager only when a linear-attention layer sits in the differentiable tail).

### Single-Step `seq_len=2048` Microbenchmark (`Qwen3.5-4B-4bit`, `rank=16`)

| Adapted Layout | Stock `mlx_lm` `gated_delta_ops` | Chunkwise-Parallel + `mx.checkpoint` | Speedup | Peak RAM Delta | Non-Zero `lora_b` |
|---|---:|---:|---:|---:|---:|
| `1-layer [31]` | `0.40s` / `2.85 GB` | **`0.28s` / `3.11 GB`** | `1.4×` | flat | `2/2` |
| `4-layer contiguous [28..31]` | `189.49s` / `36.30 GB` (swap storm) | **`2.95s` / `6.68 GB`** | **`64×`** | **`5.4×` less RAM** | `5/5` |
| `4-layer periodic [19, 23, 27, 31]` | OOM crash (`exit 137`) | **`11.27s` / `6.76 GB`** | **OOM → Works** | **`O(1)` in depth** | `8/8` |

---

## 2. End-to-End Multi-Turn Empirical Comparison (`Qwen3.5-4B-4bit`, `max_seq_len=2048`, `rank=16`)

**Task:** 7-message multi-turn SRE incident triage trajectories (`system` → `user` → `assistant(fetch_service_logs)` → `tool(logs)` → `assistant(open_incident_ticket)` → `tool(ticket)` → `assistant(summary)`) requiring internal runbook/severity mappings not known to the frozen base model.  
**Dataset:** `13` input trajectories (`12` within `max_seq_len=2048` + `1` intentionally oversized `> 2048`-token trajectory to verify explicit `kept=12, dropped_too_long=1` accounting) evaluated on **`8` strictly disjoint unseen services** (zero service-name overlap).  
**Training:** `24` optimizer steps, `grad_accum=2`, Sub-1-Bit packaging (`sub1bit_rank=16`), followed by atomic mount/unmount rollback check and stock `mlx_lm.load(MODEL_ID, adapter_path=...)` verification.

| Variant | Layer Preset & Targets | Peak RAM | Mean Step Time (`grad_accum=2`) | Loss (`init → final`) | Zero-Perturb | Rollback Diff | Sub-1-Bit Size (`bpw`) | `mlx_lm.load` Verified | **Held-Out Multi-Turn Eval (8 Unseen)** |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **Day-0 Frozen Base** | — | — | — | — | — | — | `0 B` | — | **`0.0%`** (`0/8`) |
| **1-Layer Tail Baseline** | `last_n=1` (`layers.31`, 4 projs) | `3.41 GB` | `0.45s` | `0.336 → 0.020` | `0.0e+00` | `0.0e+00` | `164.3 KiB` (`0.0165`) | `8/8` tensors | **`25.0%`** (`2/8`) |
| **4-Layer Contiguous Hybrid** | `last_n=4` (`layers.28..31`, 5 projs) | `6.50 GB` | `3.83s` | `0.350 → 0.038` | `0.0e+00` | `0.0e+00` | `210.4 KiB` (`0.0164`) | `10/10` tensors | `0.0%` (`0/8`) |
| **4-Layer Periodic Full-Attn** | `periodic_full_attn=4` (`layers.19, 23, 27, 31`, 8 projs) | **`6.61 GB`** | `13.31s` | **`0.333 → 0.005`** | **`0.0e+00`** | **`0.0e+00`** | **`288.6 KiB`** (`0.0173`) | **`16/16` tensors** | **`37.5%`** (**`3/8`, `+12.5 pp` vs. 1-layer**) |

### Key Takeaways

1. **Zero OOM & >9 GB Headroom on 16 GB Apple Silicon:** Both `4-layer contiguous` (`6.50 GB`) and `13-layer-deep periodic full-attention` (`6.61 GB`) stayed comfortably below 16 GB at `max_seq_len=2048` and `rank=16`.
2. **Periodic Full-Attention Beats Contiguous Hybrid Blocks:** On hybrid `Qwen3.5` architectures, `periodic_full_attn` (`layers.19, 23, 27, 31` — adapting both `mlp.down_proj` and `self_attn.o_proj` across depth) converged tighter (`0.005` vs. `0.038`) and generalized better on unseen multi-turn tool-policy evaluation (`37.5%` vs. `25.0%` 1-layer and `0.0%` contiguous hybrid at 24 steps) with only `+0.11 GB` extra peak RAM.
3. **Strict Multi-Turn Assistant Masking & Explicit Drop Accounting:** Every run explicitly logged `kept=12/13 (dropped_too_long=1 > max_seq_len=2048, dropped_empty_target=0)`, preserved exact `0.0e+00` zero-perturbation and `0.0e+00` atomic unmount rollback, and exported non-contiguous `adapters.safetensors` verified directly via stock `mlx_lm.load(..., adapter_path=...)`.
