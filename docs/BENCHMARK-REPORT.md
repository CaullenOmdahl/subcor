# SUBCOR Functional Growth Report

- **Suite:** `subcor-core-v1@1.0.0` (`sha256:1732c1fcad55`)
- **Model:** `mlx-community/Qwen3.5-0.8B-4bit` (tenant: `growth-lab`)
- **Snapshots Recorded:** 3 (`day0-frozen-base` → `v2-ops-plus-noc`)
- **Net Overall Growth:** `41.7% → 58.3%` (`+16.7 pp`)
- **Locked Anchor Retention:** `80.0% → 80.0%` (PRESERVED)
- **Active Skill Footprint:** 2 units, 162.8 KiB (0.0302 avg bits/weight)

## Longitudinal Progression

| Snapshot | Active Units | Skill Size | Overall | Macro Axis | Anchors | Loop Fail | Avg Latency | Tok/s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `day0-frozen-base` | 0 | 0.0 KiB | 41.7% | 45.0% | 80.0% | 0.0% | 171 ms | 114.8 |
| `v1-ops-router` | 1 | 81.4 KiB | 54.2% | 57.0% | 80.0% | 0.0% | 156 ms | 101.2 |
| `v2-ops-plus-noc` | 2 | 162.8 KiB | 58.3% | 60.3% | 80.0% | 0.0% | 137 ms | 86.5 |

## Per-Axis Functional Growth

| Axis | `day0-frozen-base` | `v1-ops-router` | `v2-ops-plus-noc` | Δ (First → Latest) |
|---|---:|---:|---:|---:|
| `domain_noc_triage` | 0.0% | 0.0% | 16.7% | **+16.7 pp** |
| `domain_ops_routing` | 0.0% | 60.0% | 60.0% | **+60.0 pp** |
| `general_reasoning` *(anchor)* | 83.3% | 83.3% | 83.3% | **+0.0 pp** |
| `instruction_discipline` *(anchor)* | 75.0% | 75.0% | 75.0% | **+0.0 pp** |
| `structured_json` | 66.7% | 66.7% | 66.7% | **+0.0 pp** |

## Active Skill Units in Latest Snapshot

| Unit ID | Version | Bits/Weight | Payload Size | SHA-256 |
|---|---|---:|---:|---|
| `capheos-ops-router-v1` | `1.0.0` | 0.0302 | 81.4 KiB | `2c09843de5b282fe` |
| `capheos-incident-sla-v1` | `1.0.0` | 0.0302 | 81.4 KiB | `468a16a76f51f907` |

---

## Controlled 100-Case Held-Out Baseline Comparison (`subcor bench baselines --live-mlx`)

Measured live on Apple M5 (`mlx-community/Qwen3.5-0.8B-4bit`, `n_train = 20`, `n_test = 100` strictly disjoint procedural workflow cases, `steps = 80`, `lr = 3e-4`, `seed = 42`):

### 1. Live Procedural Workflow Baselines (100 Unseen Cases)

| Method | Accuracy (100 held-out) | Payload Size | Prompt Tokens | Mean Latency | Notes |
|---|---:|---:|---:|---:|---|
| Frozen Base (zero-shot) | `0.0%` (`0/100`) | `0 B` | `67.7` | `298.5 ms` | Cannot guess unseen org codes |
| 5-Shot In-Context Prompting | `49.0%` (`49/100`) | `0 B` | `277.7` (**4.1×**) | `198.8 ms` | High prompt overhead, lower adherence |
| Same-Byte FP16 LoRA (Rank-1) | `53.0%` (`53/100`) | `54.0 KiB` | `67.7` | `166.2 ms` | Under-parameterized at rank 1 |
| **SUBCOR Sub-1-Bit Skill Unit (Rank-8)** | **`79.0%` (`79/100`)** | **`81.4 KiB`** | **`67.7`** | **`162.7 ms`** | **+30 pp vs 5-shot, +26 pp vs Rank-1 FP16** |
| FP16 LoRA (Rank-2, +33% bytes) | `85.0%` (`85/100`) | `108.0 KiB` | `67.7` | `159.8 ms` | Native FP16 rank-2 beats post-hoc Sub-1-Bit by +6 pp |
| Frozen-Base Embedding kNN (classification only) | `86.0%` (`86/100`) | `40.0 KiB` | — | `26.6 ms` | Pure fixed-label lookup; cannot generate tool calls |

### 2. Equal-Byte Compression Audit (`1024 × 3584` Decaying-Spectrum Real-Valued Rank-8 Delta)

| Encoding | Bytes / Layer | Bits / Weight | Relative Frobenius Error |
|---|---:|---:|---:|
| `sub1bit_rank_8` | `13,857 B` | `0.03017` | **`0.5687`** |
| `fp16_lora_rank_1` | `9,216 B` | `0.02009` | `0.7888` |
| `fp16_lora_rank_2` | `18,432 B` | `0.04018` | `0.6242` |
| `fp16_lora_rank_4` | `36,864 B` | `0.08036` | `0.3716` |
| `bitdelta_dense_1bit` | `458,754 B` | `1.00000` | `0.6548` |

### 3. Broad-Distribution Router Scale Sweep (200 Isotropic OOD General Prompts)

| Skills | Flat OOD FPR | Seat-Scoped OOD FPR | Held-Out FNR | Exact Skill Acc |
|---:|---:|---:|---:|---:|
| 1 | `0.5%` | **`0.0%`** | `0.0%` | `100.0%` |
| 5 | `1.5%` | **`0.5%`** | `1.0%` | `99.0%` |
| 10 | `0.5%` | **`0.0%`** | `5.0%` | `95.0%` |
| 20 | `2.5%` | **`1.5%`** | `3.8%` | `96.2%` |

**Honest Takeaway:** On realistic 100-case procedural workflows, `SUBCOR` Skill Units outperform 5-shot prompting by **+30 pp** at **4.1× fewer prompt tokens** (`162.7 ms` vs `198.8 ms`) and beat rank-1 FP16 LoRA (`54 KiB`) by **+26 pp**. However, native rank-2 FP16 LoRA (`108 KiB`, 33% larger) edges out post-hoc compressed Sub-1-Bit Rank-8 (`81.4 KiB`) by **+6 pp** (`85%` vs `79%`), confirming a genuine byte-vs-accuracy Pareto tradeoff—which is why SUBCOR supports both `SUB1BIT_BINARY` and `LOW_RANK_FP32`/FP16 encodings under one governed Skill Unit container.
