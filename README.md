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

```bash
uv sync --extra dev
uv run pytest
uv run slaingi demo
```

To export a passing Skill Unit as a CaPheOS-compatible unit bundle:

```bash
uv run slaingi demo --export-capheos ./out/capheos-units
```

---

## Honest Scope & Current Status (Milestone 1)

What is **working and verified today**:
- Append-only JSONL telemetry store enforcing the **Verified Evidence Gate**.
- Real **Sub-1-Bit Binary Low-Rank Factorization** with `packbits` storage, exact unpack/reconstruct, and measured `< 1.0` bits/weight compression.
- **Function-Preserving Dynamic Low-Rank Expansion** ($r \to r + k$, exact zero-perturbation initialization) with FP32 and binary Straight-Through Estimator (STE) micro-batch adaptation over a frozen base matrix.
- **Contrastive Weight-Space Surgeon** that isolates hidden-layer failure/looping directions from activation traces and synthesizes zero-gradient rank-$k$ patches.
- **Governed Skill Unit Registry** enforcing content-addressed SHA-256 verification, immutable conformance gating, no self-approval (`proposer != approver`), atomic rollback, and CaPheOS `module.json` (format 2) export.
- End-to-end **Micro-Intern Proving Ground** demonstrating both zero-gradient loop ablation and dynamic sub-1-bit skill growth with governance enforcement.

What is **not built yet** (tracked in [docs/ROADMAP.md](docs/ROADMAP.md)):
- Live Apple Silicon (`mlx-lm`) / `llama.cpp` inference server adapter for Gemma 4 QAT / Qwen 3.5 checkpoints (Milestone 2).
- Overnight scheduled on-policy distillation from senior agent transcripts (Milestone 3).
- Direct CaPheOS runtime mount & live company proving ground (Milestone 4).
