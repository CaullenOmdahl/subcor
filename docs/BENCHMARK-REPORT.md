# SLAINGI Functional Growth Report

- **Suite:** `slaingi-core-v1@1.0.0` (`sha256:9e9383e77a24`)
- **Model:** `mlx-community/Qwen3.5-0.8B-4bit` (tenant: `growth-lab`)
- **Snapshots Recorded:** 3 (`day0-frozen-base` → `v2-ops-plus-noc`)
- **Net Overall Growth:** `41.7% → 66.7%` (`+25.0 pp`)
- **Locked Anchor Retention:** `80.0% → 80.0%` (PRESERVED)
- **Active Skill Footprint:** 2 units, 162.8 KiB (0.0302 avg bits/weight)

## Longitudinal Progression

| Snapshot | Active Units | Skill Size | Overall | Macro Axis | Anchors | Loop Fail | Avg Latency | Tok/s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `day0-frozen-base` | 0 | 0.0 KiB | 41.7% | 45.0% | 80.0% | 0.0% | 265 ms | 74.2 |
| `v1-ops-router` | 1 | 81.4 KiB | 58.3% | 61.0% | 80.0% | 4.2% | 261 ms | 64.3 |
| `v2-ops-plus-noc` | 2 | 162.8 KiB | 66.7% | 67.7% | 80.0% | 4.2% | 261 ms | 49.0 |

## Per-Axis Functional Growth

| Axis | `day0-frozen-base` | `v1-ops-router` | `v2-ops-plus-noc` | Δ (First → Latest) |
|---|---:|---:|---:|---:|
| `domain_noc_triage` | 0.0% | 0.0% | 33.3% | **+33.3 pp** |
| `domain_ops_routing` | 0.0% | 80.0% | 80.0% | **+80.0 pp** |
| `general_reasoning` *(anchor)* | 83.3% | 83.3% | 83.3% | **+0.0 pp** |
| `instruction_discipline` *(anchor)* | 75.0% | 75.0% | 75.0% | **+0.0 pp** |
| `structured_json` | 66.7% | 66.7% | 66.7% | **+0.0 pp** |

## Active Skill Units in Latest Snapshot

| Unit ID | Version | Bits/Weight | Payload Size | SHA-256 |
|---|---|---:|---:|---|
| `capheos-ops-router-v1` | `1.0.0` | 0.0302 | 81.4 KiB | `ab357893db0366ad` |
| `capheos-incident-sla-v1` | `1.0.0` | 0.0302 | 81.4 KiB | `4f8af660150bf1ac` |
