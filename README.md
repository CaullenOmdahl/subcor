# SLAINGI — Self-Learning AI General Intelligence

**SLAINGI** is a governed self-learning runtime for compact local models. Instead of retraining monolithic checkpoints from scratch, SLAINGI treats an open-weight quantized model as an **immutable base kernel** and grows audited, hot-swappable **Skill Units** above it.

It combines four mechanisms that solve each other's bottlenecks:

1. **Frozen Quantized Base Kernel** — Base weights stay strictly read-only. Adaptation never mutates the underlying checkpoint.
2. **Surgical Weight-Space Patching** — Contrastive activation profiling isolates structural failure directions (such as repetition loops) and synthesizes minimal rank-$k$ ablation patches with **zero gradient descent**.
3. **Function-Preserving Dynamic Expansion + Quantized Adaptation (STE)** — When a task requires new circuits rather than unblocking latent ones, SLAINGI appends zero-initialized low-rank columns ($r \to r + k$) and adapts only those new columns using verified on-policy corrections and straight-through estimation.
4. **Sub-1-Bit Binary Factorization & Governed Skill Units** — Learned skill deltas are compressed into bit-packed binary low-rank factors ($W \approx s_1 \odot (U_{\pm 1} V_{\pm 1}^T) \odot s_2^T$), content-addressed with SHA-256 digests, gated by an immutable conformance suite and separation of duties (no self-approval), rolled back atomically, and exportable as **CaPheOS** format-2 capability units (`module.json`).

---

## Architecture Overview

```text
┌──────────────────────────────────────────────────────────────────────────┐
│                         Governed Adaptation Loop                         │
│                                                                          │
│  [1. Intern Attempt] ──► [2. Verified Evidence Gate]                     │
│         │                         │ (drops unverified trajectories)      │
│         ▼                         ▼                                      │
│  [Activation Traces]      [3. Diagnosis & Synthesis]                     │
│         │                 ├── Structural defect ─► Weight-Space Surgeon  │
│         └───────────────► └── Missing skill     ─► Dynamic Rank + STE    │
│                                   │                                      │
│                                   ▼                                      │
│                    [4. Sub-1-Bit Skill Unit Packager]                    │
│                       (bit-packed U, V ∈ {±1} + s1, s2 + SHA-256)        │
│                                   │                                      │
│                                   ▼                                      │
│               [5. Locked Conformance Gate + Separate Approver]           │
│                  ├── Fail / Regress ──► Rejected (remains DORMANT)       │
│                  └── Pass + Sign-off ─► Hot-Swapped onto Frozen Base     │
└──────────────────────────────────────────────────────────────────────────┘
```

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the complete specification, behavioral contracts, invariants, and architecture decision records, and [docs/ROADMAP.md](docs/ROADMAP.md) for the milestone plan.

---

## Quickstart

Requires Python 3.12 or 3.13 and [`uv`](https://docs.astral.sh/uv/).

### Core Proving Ground (CPU, NumPy-only, ~1s)

```bash
uv sync --extra dev
uv run pytest
uv run slaingi demo --export-capheos ./out/capheos-units
```

### Live Apple Silicon MLX Kernel (Compact 0.8B 4-bit Model)

Targets compact, capable local models (defaults to `mlx-community/Qwen3.5-0.8B-4bit`, ~652 MB, with `--model` support for Gemma 4 E2B QAT or any `mlx-lm` checkpoint):

```bash
uv sync --extra dev --extra mlx

# Standardized Longitudinal Growth Benchmark (works against local MLX or live `slaingi serve`)
uv run --extra mlx slaingi bench run --label day0-base --ledger ./benchmarks/history.jsonl
uv run --extra mlx slaingi bench run \
  --endpoint http://10.1.100.11:8420/v1 --model slaingi/company-1 \
  --label v1-after-ops --ledger ./benchmarks/history.jsonl
uv run slaingi bench report --ledger ./benchmarks/history.jsonl --out ./benchmarks/GROWTH_REPORT.md
uv run slaingi bench compare --ledger ./benchmarks/history.jsonl

# Live Multi-Tenant OpenAI-Compatible & Governed Server (for 1..N CaPheOS instances)
SLAINGI_API_KEY="replace-with-strong-key" uv run --extra mlx slaingi serve \
  --host 127.0.0.1 --port 8420 \
  --tenant company-1 --tenant company-2

# Live Routed MoE Skill Layer (Shared Dense 0.8B Core + Two Routed Sub-1-Bit Experts)
# Demonstrates multi-skill stacking with 0% general QA regression
uv run --extra mlx slaingi mlx-moe \
  --export-capheos ./out/capheos-moe-units

# Live On-Policy Intern distillation -> Sub-1-Bit Skill Unit (< 0.05 bits/weight)
# + locked conformance gate + CaPheOS & mlx-lm safetensors export + rollback proof
uv run --extra mlx slaingi mlx-intern \
  --export-capheos ./out/capheos-units \
  --export-safetensors ./out/mlx-lora

# Live zero-gradient contrastive activation diagnosis & rank-1 ablation
uv run --extra mlx slaingi mlx-surgery
```

---

## Honest Scope & Current Status (Milestones 1–3)

What is **working and verified today**:
- Append-only JSONL telemetry stores (`TrajectoryStore` & `TextTrajectoryStore`) enforcing the **Verified Evidence Gate**.
- Real **Sub-1-Bit Binary Low-Rank Factorization** with `packbits` storage, exact unpack/reconstruct, and measured `< 1.0` bits/weight compression (`~0.03` bits/weight on real `1024 x 3584` transformer MLP projections).
- **Function-Preserving Dynamic Low-Rank Expansion** ($r \to r + k$, exact `0.0` initial output/logit perturbation) with FP32, Binary STE (QAT), and live Apple Silicon MLX GPU QLoRA adaptation over frozen quantized base weights.
- **Contrastive Weight-Space Surgeon** that isolates hidden-layer failure directions from activation traces and synthesizes zero-gradient rank-$k$ ablation patches.
- **Governed Skill Unit Registry** enforcing content-addressed SHA-256 verification, immutable conformance gating, no self-approval (`proposer != approver`), atomic rollback, **CaPheOS `module.json` (format 2)** export (with runnable Node `conformance/test.js`), and stock **`mlx-lm` `adapters.safetensors`** export.
- End-to-end **Micro-Intern Proving Ground** (`slaingi demo`) and live **Apple Silicon MLX Intern + Surgery** runners (`slaingi mlx-intern`, `slaingi mlx-surgery`) verified on `mlx-community/Qwen3.5-0.8B-4bit`.

What is **next** (tracked in [docs/ROADMAP.md](docs/ROADMAP.md)):
- Scheduled overnight failure-clustering daemon & live CaPheOS container integration (Milestone 4).
