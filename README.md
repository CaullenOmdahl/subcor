# SLAINGI — Self-Learning AI General Intelligence

**Governed sub-1-bit, MoE-routed Skill Units for self-learning compact language models.**

SLAINGI turns a tiny, frozen 4-bit open-weight model (default **`Qwen3.5-0.8B-4bit`**, 652 MB, running locally at `154 tok/s` on Apple Silicon) into a **self-learning, multi-tenant "Intern" server**. Instead of mutating monolithic checkpoints or relying solely on prompt tweaks, SLAINGI keeps the base kernel read-only and grows audited, **`~81 KiB` (`0.0302 bits/weight`) Skill Units** that are dynamically routed per prompt, gated against regression, blocked from self-approving, rolled back atomically, and exportable as portable **CaPheOS format-2 capability units**.

> **Methodology Preprint:** See [docs/PAPER.md](docs/PAPER.md) for the complete mathematical formulation, experimental methodology, empirical results (including negative findings), and limitations.

---

## Architecture

```text
Incoming Prompt (OpenAI-compatible /v1/chat/completions or local MLX)
        │
        ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ 1. Frozen Quantized Base Kernel (e.g., Qwen3.5-0.8B-4bit · 652 MB)      │
│    • Strictly read-only base weights; shared across 1..N tenants        │
│    • Early-exit tap (layers 0..18, skips lm_head) -> 18.3 ms embedding  │
└────────────────────────────────────┬────────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ 2. Activation-Space MoE Skill Router (slaingi.router)                   │
│    • Training-free contrastive RouterGate per enabled Skill Unit        │
│    • Invariant to installed units (taps frozen base with bypass=True)   │
│    • Selects top-k matching unit OR 0 units (pure base = 0% forgetting) │
└───────────────┬────────────────────┼────────────────────┬───────────────┘
                ▼                    ▼                    ▼
        [Pure Frozen Base]   [Skill Unit 1: 81 KiB] [Skill Unit 2: 81 KiB]
         General QA & Chat     0.0302 bits/weight     0.0302 bits/weight
         (+0.0 pp regression)  SHA-256 verified       SHA-256 verified
```

### Governed Adaptation Loop (How New Skill Units Are Born)

1. **Verified Evidence Gate** (`slaingi.telemetry`, `slaingi.intern`) — Only trajectories attested by a deterministic verifier or senior-seat/owner sign-off can enter diagnosis or learning; unverified outputs are excluded.
2. **Dual-Mode Weight Adaptation**:
   - **Zero-Gradient Weight-Space Surgery** (`slaingi.surgeon`) — Contrastive activation profiling isolates structural failure directions (such as repetition loops) and synthesizes rank-$k$ ablation patches $\Delta W = -\alpha U(U^T W_{\text{base}})$ without gradient descent.
   - **Function-Preserving Dynamic Expansion** (`slaingi.expansion`, `slaingi.backends.mlx_kernel`) — Appends zero-initialized low-rank branches (`max|logits_after - logits_before| = 0.0`) and trains only the new weights via MLX GPU autodiff (FP32 warm-start or opt-in Two-Stage Binary STE QAT).
3. **Sub-1-Bit Binary Compression** (`slaingi.sub1bit`) — Packs deltas as $\Delta W \approx \operatorname{diag}(s_1)(U_{\pm 1}\operatorname{diag}(\sigma)V_{\pm 1}^T)\operatorname{diag}(s_2)$ using `np.packbits` (`0.0302 bits/weight`, **5.3× smaller than FP16 `adapters.safetensors`**).
4. **Governed Lifecycle & Atomic Rollback** (`slaingi.units`) — `DORMANT → CONFORMANT → ENABLED → ROLLED_BACK` with SHA-256 tamper detection, locked regression conformance gate, **separation of duties (`approver != proposer`)**, and bit-identical rollback (`0.0e+00` logit diff).

---

## Verified Empirical Highlights (`Qwen3.5-0.8B-4bit`, Apple M5)

Measured live on the content-hashed 5-axis **`slaingi-core-v1@1.0.0`** benchmark (`sha256:9e9383e77a24`, full report in [docs/BENCHMARK-REPORT.md](docs/BENCHMARK-REPORT.md)):

| Metric | Measured Result |
|---|---|
| Frozen Base Kernel Size | **652 MB** (`mlx-community/Qwen3.5-0.8B-4bit`) |
| Sustained HTTP Decode Speed (AC Power) | **154.1 tok/s** (`245 tokens` in `1.59s`) |
| Early-Exit Routing Tap Overhead | **18.3 ms/prompt** (skips layers 19–23 & `lm_head`) |
| Skill Unit Size (6 layers, `22.0M` weights) | **81.4 KiB** (**`0.0302 bits/weight`**, vs. `433 KiB` FP16 LoRA) |
| Zero-Perturbation Expansion & Rollback | **`0.0e+00`** max logit diff |
| Unseen Ops Routing Growth | **`0.0% → 60%–80%`** (`100.0%` semantic routing accuracy) |
| Locked Anchor Retention (`general_reasoning`, `instruction_discipline`, `structured_json`) | **`+0.0 pp` regression** (`80.0% → 80.0%` preserved) |
| Multi-Unit Routed Turn Latency | **`171 ms` (base) → `137 ms` (2 routed units)** |

---

## Quickstart

Requires Python 3.12 or 3.13 and [`uv`](https://docs.astral.sh/uv/).

### 1. Core Engine & Synthetic Proving Ground (CPU, NumPy-only, ~1s)

```bash
uv sync --extra dev --python 3.12
uv run pytest -v
uv run slaingi demo --export-capheos ./out/capheos-units
```

### 2. Apple Silicon Live MLX Workflows (`Qwen3.5-0.8B-4bit` or any `--model`)

```bash
uv sync --extra dev --extra mlx --python 3.12

# Live two-expert Routed MoE Skill Layer (shared 0.8B core + 2 sub-1-bit experts)
uv run --extra mlx slaingi mlx-moe --export-capheos ./out/capheos-moe-units

# Live single-skill On-Policy Intern distillation + dual export
uv run --extra mlx slaingi mlx-intern \
  --export-capheos ./out/capheos-units \
  --export-safetensors ./out/mlx-lora

# Live zero-gradient contrastive activation diagnosis & rank-1 ablation
uv run --extra mlx slaingi mlx-surgery
```

### 3. Multi-Tenant OpenAI-Compatible & Governed Server

Serves one frozen `652 MB` base model to multiple isolated tenants (e.g., two CaPheOS instances) with OpenAI `/v1/chat/completions` (JSON + SSE streaming), Ollama `/api/tags` discovery, and governed control endpoints:

```bash
uv run --extra mlx slaingi serve \
  --host 127.0.0.1 \
  --port 8420 \
  --state-dir ~/.local/share/slaingi \
  --tenant company-a \
  --tenant company-b
```

Optional bearer auth: set `SLAINGI_API_KEY=...` or pass `--api-key`. See [docs/CAPHEOS-INTEGRATION.md](docs/CAPHEOS-INTEGRATION.md) for connecting multiple CaPheOS instances over an encrypted SSH loopback tunnel or LAN.

### 4. Standardized Longitudinal Functional Growth Benchmarks

```bash
# Run built-in slaingi-core-v1 against local MLX or any live OpenAI-compatible server
uv run --extra mlx slaingi bench run \
  --endpoint http://127.0.0.1:8420/v1 \
  --model slaingi/company-a \
  --label "day0-frozen-base" \
  --ledger ~/.local/share/slaingi/benchmarks/company-a.jsonl

# Render Markdown Growth Report and JSON delta
uv run slaingi bench report \
  --ledger ~/.local/share/slaingi/benchmarks/company-a.jsonl \
  --out ./company-a-growth.md

uv run slaingi bench compare \
  --ledger ~/.local/share/slaingi/benchmarks/company-a.jsonl

# Export canonical suite JSON to inspect or create custom company suites
uv run slaingi bench export-suite --out ./slaingi-core-v1.json
```

---

## Multi-Tenant HTTP API Reference

| Method & Path | Purpose |
|---|---|
| `GET /healthz`, `GET /readyz` | Liveness/readiness, active model, per-tenant enabled units |
| `GET /v1/models`, `GET /api/tags` | OpenAI & Ollama/StarNet model discovery (`slaingi/base`, `slaingi/<tenant>`) |
| `POST /v1/chat/completions` | Routed chat completions (non-streaming JSON & `stream: true` SSE) |
| `POST /v1/slaingi/tenants/{t}/trajectories` | Append verified or unverified Intern trajectories |
| `POST /v1/slaingi/tenants/{t}/study` | Run governed Study Cycle (`CONFORMANT`/`DORMANT`; never self-enables) |
| `GET  /v1/slaingi/tenants/{t}/units` | List tenant Skill Units with SHA-256, bits/weight, and conformance reports |
| `POST /v1/slaingi/tenants/{t}/units/{u}/promote` | Promote `CONFORMANT → ENABLED` (`approver != proposer`, else HTTP 403) |
| `POST /v1/slaingi/tenants/{t}/units/{u}/rollback` | Atomic live rollback `ENABLED → ROLLED_BACK` |
| `POST /v1/slaingi/tenants/{t}/units/{u}/export-capheos` | Export portable CaPheOS format-2 bundle (`module.json`, `weights.slun`, `router_gate.json`, `conformance/test.js`) |

---

## Repository Map

```text
src/slaingi/
├── types.py               # Domain types, lifecycle states, error hierarchy
├── telemetry.py           # Append-only JSONL TrajectoryStore (Verified Evidence Gate)
├── intern.py              # TextTrajectoryStore & n-gram repetition loop metrics
├── sub1bit.py             # NanoQuant-style bit-packed binary low-rank factorization
├── expansion.py           # Function-preserving ElasticLinear + FP32 & Binary STE
├── surgeon.py             # Zero-gradient contrastive activation WeightSpaceSurgeon
├── router.py              # Training-free frozen-base RouterGate & MoESkillRouter
├── units.py               # Governed SkillUnitRegistry, SHA-256, rollback, CaPheOS export
├── server.py              # Multi-tenant OpenAI + Control Plane HTTP server
├── benchmark.py           # Standardized slaingi-core-v1 suite, ledger, diff, Markdown reports
├── proving_ground.py      # Deterministic synthetic Micro-Intern proving ground
├── live_loop.py           # Live Apple Silicon MLX Intern, MoE, and Surgery workflows
├── cli.py                 # `slaingi` CLI entrypoint
└── backends/
    └── mlx_kernel.py      # Frozen Apple Silicon MLX kernel + native-dtype SlaingiMLXLinear

tests/                     # 25 automated unit & HTTP integration tests
docs/
├── PAPER.md               # Methodology draft paper (v0.1)
├── ARCHITECTURE.md        # Intended behavior contracts, invariants, and ADRs 001–006
├── BENCHMARK-REPORT.md    # Live verified 3-snapshot growth report on Qwen3.5-0.8B-4bit
├── CAPHEOS-INTEGRATION.md # Multi-instance CaPheOS deployment & governance runbook
└── ROADMAP.md             # Milestone exit gates and next steps
```

---

## Licensing

- **Core Runtime:** `AGPL-3.0-or-later`
- **Exported Skill Units:** Default `Apache-2.0` (compatible with Apache-2.0 base checkpoints such as Gemma 4 / Qwen 3.5 and CaPheOS unit licensing)
