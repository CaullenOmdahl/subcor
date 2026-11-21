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
