# SUBCOR — Sub-Bit Unit-Based Continual On-Policy Runtime

**Governed sub-1-bit, MoE-routed Skill Units for self-learning compact language models.**

SUBCOR turns a tiny, frozen 4-bit open-weight model (default **`Qwen3.5-0.8B-4bit`**, 652 MB, running locally at `154 tok/s` on Apple Silicon) into a **self-learning, multi-tenant subcortical runtime**. Rather than cramming encyclopedic facts into weights, SUBCOR packages **procedural workflows, decision policies, and parsing discipline** into audited, **`~81 KiB` (`0.0302 bits/weight`) Skill Units**—like swappable **drill bits on an impact drill**—while factual memory stays in normal context, RAG, and MCP tools. Units are dynamically routed per prompt, monitor decode uncertainty to hand novel cases back to a frontier model (**Cortical Handoff**), recompile automatically when you upgrade the base model (`subcor recompile`), block self-approval, roll back atomically, and export as portable **CaPheOS format-2 capability units**.

> **Methodology Preprint:** See [docs/PAPER.md](docs/PAPER.md) for the Cortical-Subcortical cognitive thesis, mathematical formulation, empirical results (including negative findings), and limitations.

---

## Cognitive Model: Impact Drill + Recompilable Bits

| Brain Analogy | SUBCOR Component | Role |
|---|---|---|
| **Working Memory** | Context window, RAG, MCP tools | Holds facts, documents, and live state (never stuffed into weights) |
| **Prefrontal Cortex** | Frontier senior seats (Claude / GPT / Gemini) | Handles novel planning, ambiguous edge cases, and verifies corrections |
| **Basal Ganglia** | Frozen `MoESkillRouter` + `RouterGate` | Recognizes familiar routines (`18.3 ms`) and selects the matching `~81 KiB` drill bit |
| **Cerebellum** | Routed Sub-1-Bit Skill Units (`0.0302 bpw`) | Smooths repeatable execution: routing, tool calling, structured parsing, policy gates |
| **Thalamic Switch** | `TurnConfidence` (`X-SUBCOR-Escalate`) | Detects decode uncertainty, loops, or unmapped tasks and hands control back to Cortex |

---

## Architecture

```text
Incoming Multi-Turn Chat + OpenAI Tools (/v1/chat/completions — JSON or token SSE)
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
│ 2. Activation-Space MoE Skill Router (subcor.router)                   │
│    • Training-free contrastive RouterGate per enabled Skill Unit        │
│    • Invariant to installed units (taps frozen base with bypass=True)   │
│    • Selects top-k matching unit OR 0 units (pure base = 0% forgetting) │
└───────────────┬────────────────────┼────────────────────┬───────────────┘
                ▼                    ▼                    ▼
        [Pure Frozen Base]   [Skill Unit 1: 81 KiB] [Skill Unit 2: 81 KiB]
         General QA & Chat     0.0302 bits/weight     0.0302 bits/weight
         (+0.0 pp regression)  SHA-256 + skill_spec   SHA-256 + skill_spec
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ 3. Cortical Handoff / Thalamic Switch (subcor.confidence)              │
│    • Scores per-token decode logprobs, router margin, & loop ratio      │
│    • Emits X-SUBCOR-Escalate, X-SUBCOR-Confidence, & subcor.confidence  │
│    • Parses native XML/JSON <tool_call> blocks into OpenAI tool_calls   │
└─────────────────────────────────────────────────────────────────────────┘
```

### Governed Adaptation Loop (How New Skill Units Are Born)

1. **Verified Evidence Gate** (`subcor.telemetry`, `subcor.intern`) — Only trajectories attested by a deterministic verifier or senior-seat/owner sign-off can enter diagnosis or learning; unverified outputs are excluded.
2. **Dual-Mode Weight Adaptation**:
   - **Zero-Gradient Weight-Space Surgery** (`subcor.surgeon`) — Contrastive activation profiling isolates structural failure directions (such as repetition loops) and synthesizes rank-$k$ ablation patches $\Delta W = -\alpha U(U^T W_{\text{base}})$ without gradient descent.
   - **Function-Preserving Dynamic Expansion** (`subcor.expansion`, `subcor.backends.mlx_kernel`) — Appends zero-initialized low-rank branches (`max|logits_after - logits_before| = 0.0`) and trains only the new weights via MLX GPU autodiff (FP32 warm-start or opt-in Two-Stage Binary STE QAT).
3. **Sub-1-Bit Binary Compression** (`subcor.sub1bit`) — Packs deltas as $\Delta W \approx \operatorname{diag}(s_1)(U_{\pm 1}\operatorname{diag}(\sigma)V_{\pm 1}^T)\operatorname{diag}(s_2)$ using `np.packbits` (`0.0302 bits/weight`, **5.3× smaller than FP16 `adapters.safetensors`**).
4. **Governed Lifecycle & Atomic Rollback** (`subcor.units`) — `DORMANT → CONFORMANT → ENABLED → ROLLED_BACK` with SHA-256 tamper detection, locked regression conformance gate, **separation of duties (`approver != proposer`)**, and bit-identical rollback (`0.0e+00` logit diff).

---

## Verified Empirical Highlights (`Qwen3.5-0.8B-4bit`, Apple M5)

Measured live on the content-hashed 5-axis **`subcor-core-v1@1.0.0`** benchmark (`sha256:1732c1fcad55`, full report in [docs/BENCHMARK-REPORT.md](docs/BENCHMARK-REPORT.md)):

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
uv run subcor demo --export-capheos ./out/capheos-units
```

### 2. Apple Silicon Live MLX Workflows (`Qwen3.5-0.8B-4bit` or any `--model`)

```bash
uv sync --extra dev --extra mlx --python 3.12

# Live two-expert Routed MoE Skill Layer (shared 0.8B core + 2 sub-1-bit experts)
uv run --extra mlx subcor mlx-moe --export-capheos ./out/capheos-moe-units

# Live single-skill On-Policy Intern distillation + dual export
uv run --extra mlx subcor mlx-intern \
  --export-capheos ./out/capheos-units \
  --export-safetensors ./out/mlx-lora

# Live zero-gradient contrastive activation diagnosis & rank-1 ablation
uv run --extra mlx subcor mlx-surgery
```

### 3. Multi-Tenant OpenAI-Compatible & Governed Server

Serves one frozen `652 MB` base model to multiple isolated tenants (e.g., two CaPheOS instances) with OpenAI `/v1/chat/completions` (JSON + SSE streaming), Ollama `/api/tags` discovery, and governed control endpoints:

```bash
uv run --extra mlx subcor serve \
  --host 127.0.0.1 \
  --port 8420 \
  --state-dir ~/.local/share/subcor \
  --tenant company-a \
  --tenant company-b
```

Optional bearer auth: set `SUBCOR_API_KEY=...` or pass `--api-key`. See [docs/CAPHEOS-INTEGRATION.md](docs/CAPHEOS-INTEGRATION.md) for connecting multiple CaPheOS instances over an encrypted SSH loopback tunnel or LAN.

### 4. Standardized Longitudinal Functional Growth Benchmarks

```bash
# Run built-in subcor-core-v1 against local MLX or any live OpenAI-compatible server
uv run --extra mlx subcor bench run \
  --endpoint http://127.0.0.1:8420/v1 \
  --model subcor/company-a \
  --label "day0-frozen-base" \
  --ledger ~/.local/share/subcor/benchmarks/company-a.jsonl

# Render Markdown Growth Report and JSON delta
uv run subcor bench report \
  --ledger ~/.local/share/subcor/benchmarks/company-a.jsonl \
  --out ./company-a-growth.md

uv run subcor bench compare \
  --ledger ~/.local/share/subcor/benchmarks/company-a.jsonl

# Export canonical suite JSON to inspect or create custom company suites
uv run subcor bench export-suite --out ./subcor-core-v1.json
```

### 5. Recompiling Skill Units Across Base Kernel Upgrades

Each compiled `payload.scun` is a binary tied to its `base_kernel_id` ("the impact drill"), while its companion `skill_spec.json` is portable workflow source code ("the drill bit blueprint"). Upgrading to a newer or larger base checkpoint recompiles your entire library automatically through the locked Conformance Gate:

```bash
uv run --extra mlx subcor recompile \
  --source-units ~/.local/share/subcor/tenants/company-a/units \
  --dest-state ~/.local/share/subcor-v2 \
  --tenant company-a \
  --model mlx-community/Qwen3.5-2B-4bit
```

Recompiled units land in `CONFORMANT` (never self-enabled) so an independent approver promotes them into production.

---

## Multi-Tenant HTTP API Reference

| Method & Path | Purpose |
|---|---|
| `GET /healthz`, `GET /readyz` | Liveness/readiness, active model, per-tenant enabled units |
| `GET /v1/models`, `GET /api/tags` | OpenAI & Ollama/StarNet model discovery (`subcor/base`, `subcor/<tenant>`) |
| `POST /v1/chat/completions` | Full multi-turn chat, OpenAI `tools`/`tool_calls`, true token-by-token SSE streaming (`stream: true`), plus `X-SUBCOR-Routed-Units`, `X-SUBCOR-Escalate`, and `X-SUBCOR-Confidence` headers |
| `POST /v1/subcor/tenants/{t}/trajectories` | Append verified or unverified Intern trajectories |
| `POST /v1/subcor/tenants/{t}/study` | Run governed Study Cycle (`CONFORMANT`/`DORMANT`; saves `skill_spec.json`; never self-enables) |
| `GET  /v1/subcor/tenants/{t}/units` | List tenant Skill Units with `base_kernel_id`, SHA-256, bits/weight, and conformance reports |
| `POST /v1/subcor/tenants/{t}/units/{u}/promote` | Promote `CONFORMANT → ENABLED` (`approver != proposer`, else HTTP 403) |
| `POST /v1/subcor/tenants/{t}/units/{u}/rollback` | Atomic live rollback `ENABLED → ROLLED_BACK` |
| `POST /v1/subcor/tenants/{t}/units/{u}/export-capheos` | Export portable CaPheOS format-2 bundle (`module.json`, `weights.scun`, `router_gate.json`, `skill_spec.json`, `conformance/test.js`) |

---

## Repository Map

```text
src/subcor/
├── types.py               # Domain types, SkillSpec, lifecycle states, error hierarchy
├── telemetry.py           # Append-only JSONL TrajectoryStore (Verified Evidence Gate)
├── intern.py              # TextTrajectoryStore & n-gram repetition loop metrics
├── sub1bit.py             # NanoQuant-style bit-packed binary low-rank factorization
├── expansion.py           # Function-preserving ElasticLinear + FP32 & Binary STE
├── surgeon.py             # Zero-gradient contrastive activation WeightSpaceSurgeon
├── router.py              # Training-free frozen-base RouterGate & MoESkillRouter
├── confidence.py          # Cortical Handoff (TurnConfidence) & OpenAI tool_calls parser
├── units.py               # Governed SkillUnitRegistry, SHA-256, skill_spec, rollback, export
├── recompile.py           # Portable Skill Unit recompiler across base-kernel upgrades
├── server.py              # Multi-tenant OpenAI (multi-turn/tools/SSE) + Control Plane server
├── benchmark.py           # Standardized subcor-core-v1 suite, ledger, diff, Markdown reports
├── proving_ground.py      # Deterministic synthetic Micro-Intern proving ground
├── live_loop.py           # Live Apple Silicon MLX Intern, MoE, and Surgery workflows
├── cli.py                 # `subcor` CLI entrypoint (`serve`, `bench`, `recompile`, ...)
└── backends/
    └── mlx_kernel.py      # Frozen Apple Silicon MLX kernel, stream_generate, SubcorMLXLinear

tests/                     # 31 automated unit & HTTP integration tests
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
