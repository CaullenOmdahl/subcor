# SUBCOR

**Teach a small local model new skills without ever touching its weights, and without letting
it approve its own changes.**

[![License: AGPL-3.0-or-later](https://img.shields.io/badge/license-AGPL--3.0--or--later-blue.svg)](LICENSE)
![Python 3.12 | 3.13](https://img.shields.io/badge/python-3.12%20%7C%203.13-blue.svg)
![Backends: MLX · llama.cpp · PyTorch](https://img.shields.io/badge/backends-MLX%20%C2%B7%20llama.cpp%20%C2%B7%20PyTorch-informational.svg)

The name stands for *Sub-Bit Unit-Based Continual On-Policy Runtime*.

SUBCOR keeps a quantized open-weight model (Qwen 3.5 0.8B–9B, Gemma 4, any `mlx-lm` or GGUF
checkpoint) **frozen**, and adds what it learns as tiny, separate **Skill Units**. Each one
is a low-rank weight delta packed to well under one bit per weight, often under 100 KiB. A
training-free router mounts a unit only when a prompt calls for it. Every unit has to earn
its place through a locked regression check and a second party's approval, and can be
rolled back instantly to the exact original model.

```text
   frozen 4-bit base  +  routed ~66–290 KiB Skill Units  +  governance gates
   (never modified)      (one per skill, swappable)         (verified evidence, no
                                                             regression, no self-approval)
```

## Why it's interesting

On the official **Berkeley Function Calling Leaderboard (BFCL v3)**, scored on 120 held-out
cases with no overlap with the training stream:

| Model | Frozen base | + one SUBCOR Skill Unit | Unit size | General-QA anchors |
|---|---:|---:|---:|---:|
| Qwen3.5-0.8B-4bit | 40.8% | **62.5%** (+21.7 pp) | **66 KiB** | 80% → 80% (no forgetting) |
| Qwen3.5-4B-4bit | 81.7% | **83.3%** (+1.7 pp, no category regressed) | 164 KiB | 80% → 80% |

- **The 0.8B gain is 21.7 points in 77 seconds of training on a laptop.** The sub-1-bit
  unit matches its own 16× larger FP32 version case for case, and mean latency dropped from
  1,779 ms to 1,033 ms.
- **On-policy beats naive fine-tuning.** Training only on verified corrections of the
  model's own failures, plus replays of its own correct outputs, beats fine-tuning on ground
  truth. On the 4B, naive SFT lost 3.4 points and was blocked by the gate. Details and
  negative results are in [`docs/BFCL-POC-REPORT.md`](docs/BFCL-POC-REPORT.md).
- **Adding units doesn't cost general ability.** Routed units are mounted only when their
  gate fires, so on a fixed 5-axis benchmark the locked general-reasoning and
  instruction-following anchors moved **+0.0 pp** while two skills were added. Mounting a
  unit densely, by contrast, cost 16.6 points of general QA (83.3% → 66.7%)
  ([`docs/BENCHMARK-REPORT.md`](docs/BENCHMARK-REPORT.md)).
- **Adding or removing a unit is exact.** A new unit starts zero-initialized, and the logit
  difference before training is `0.0e+00`. Rollback restores the base logits bit for bit.
- **It runs on hardware you already have.**
  - Apple M5 laptop: 154 tok/s serving a 0.8B over HTTP.
  - AMD RX 6800 (16 GB), via a supervised `llama-server`: ~300 tok/s on a 0.8B and ~110
    tok/s on a 4B, while training units on the same card
    ([`docs/AMD-GGUF.md`](docs/AMD-GGUF.md)).

## How it works

```text
 chat request (/v1/chat/completions, multi-turn, OpenAI tools, SSE streaming)
        │
        ▼
 ┌─────────────────────────────────────────────┐
 │ Frozen quantized base model                 │  read-only, shared by every tenant
 │  └─ early-exit tap → prompt embedding       │  ~18 ms, skips the upper layers and lm_head
 └──────────────────────┬──────────────────────┘
                        ▼
 ┌─────────────────────────────────────────────┐
 │ Router: one analytic gate per enabled unit  │  no training, no router collapse
 │  top-k units that clear their threshold     │  none clears → pure base, zero perturbation
 └───────┬───────────────┬───────────────┬─────┘
         ▼               ▼               ▼
     [ base ]      [ unit A ]      [ unit B ]       low-rank deltas, sub-1-bit packed,
                                                    SHA-256 checked on every load
                        ▼
 ┌─────────────────────────────────────────────┐
 │ Confidence check                            │  low decode confidence, loops or an
 │  X-SUBCOR-Escalate / X-SUBCOR-Confidence     │  unmatched prompt → hand back to a
 └─────────────────────────────────────────────┘  bigger model
```

### How a Skill Unit is born

1. **Only verified evidence goes in.** Trajectories enter an append-only store. Only those
   confirmed by a deterministic verifier (a schema check, a test, an exit code) or a human
   or senior-model reviewer can be used for training.
2. **Adapt, one of two ways.**
   - **Zero-gradient surgery** finds a failure direction from activations and projects it
     out. This suits structural defects such as loops.
   - **Function-preserving expansion** adds a zero-initialized low-rank branch and trains
     only that branch. This suits skills the base doesn't have yet.
3. **Pack the delta to sub-1-bit.** Binary low-rank factors plus fp16 scales come to about
   0.02–0.04 bits per weight, about 5× smaller than an FP16 LoRA file.
4. **Gate it.** The unit moves `DORMANT → CONFORMANT` only if it improves its target and the
   owner's locked anchors don't regress. A proposer can make the gate stricter, never
   looser.
5. **Approve it.** It moves `CONFORMANT → ENABLED` only when someone other than the proposer
   promotes it.
6. **Roll back any time.** `ENABLED → ROLLED_BACK` atomically, per unit.

Each unit also stores a portable `skill_spec.json`. When you upgrade the base model,
`subcor recompile` retrains every unit onto the new base and re-verifies it.

The maths is in [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md#4-method-details). Where the
ideas came from, and what we think is actually new, is in
[`docs/BACKGROUND.md`](docs/BACKGROUND.md).

## Quickstart

You need Python 3.12 or 3.13 and [`uv`](https://docs.astral.sh/uv/).

### 1. See the whole loop in about a second (CPU, NumPy only)

```bash
git clone https://github.com/CaullenOmdahl/subcor.git && cd subcor
uv sync --extra dev
uv run subcor demo
```

The demo runs a synthetic network with an injected thinking loop and a missing skill. In one
run you can watch:

- zero-gradient surgery remove the loop;
- expansion learn the missing skill, packed to under 1 bit per weight;
- a poisoned patch get rejected and a self-approval get blocked;
- a rollback restore the earlier state.

### 2. Apple Silicon (MLX)

```bash
uv sync --extra dev --extra mlx
uv run --extra mlx subcor mlx-moe        # two routed skills on one frozen 0.8B, general QA unchanged
uv run --extra mlx subcor mlx-intern     # learn one skill from verified corrections, export it
uv run --extra mlx subcor mlx-surgery    # zero-gradient activation diagnosis and ablation
```

### 3. AMD, NVIDIA or CPU (GGUF via llama.cpp, PyTorch trainer)

```bash
uv sync --python 3.13 --extra dev --extra rocm
uv run --extra rocm subcor serve --backend llamacpp --gguf /path/to/Qwen3.5-0.8B-Q4_K_M.gguf --tenant acme
```

SUBCOR supervises its own `llama-server`. With `--sticky-routing` it keeps a conversation's units
unless a switch clearly wins, because every unit switch makes llama-server re-read the prompt. It trains units with PyTorch from the same GGUF,
exports them as GGUF LoRA adapters and hot-loads them. Setup, flags, VRAM budgets and
measured numbers are in [`docs/AMD-GGUF.md`](docs/AMD-GGUF.md).

### 4. Serve several tenants from one base

```bash
uv run --extra mlx subcor serve --host 127.0.0.1 --port 8420 \
  --state-dir ~/.local/share/subcor --tenant company-a --tenant company-b
```

The server speaks the OpenAI API (`/v1/chat/completions` with tools and streaming, and
`/v1/models`) and the Ollama discovery endpoint (`/api/tags`), so existing clients can point
straight at it. Each tenant gets its own units, trajectories and router. Binding to anything
but loopback requires `--api-key` or `--auth-config`. Connecting agent platforms is covered
in [`docs/CAPHEOS-INTEGRATION.md`](docs/CAPHEOS-INTEGRATION.md).

### 5. Track growth over time

```bash
uv run subcor bench run --endpoint http://127.0.0.1:8420/v1 --model subcor/company-a \
  --label day0 --ledger ./company-a.jsonl
uv run subcor bench report --ledger ./company-a.jsonl --out ./company-a-growth.md
uv run subcor bench bfcl --help        # the BFCL v3 harness used for the results above
```

## HTTP API

| Endpoint | What it does |
|---|---|
| `GET /healthz`, `GET /readyz` | Liveness and readiness, active model, enabled units per tenant |
| `GET /v1/models`, `GET /api/tags` | Model discovery (`subcor/base`, `subcor/<tenant>`) |
| `POST /v1/chat/completions` | Multi-turn chat, OpenAI `tools` and `tool_calls`, token SSE streaming, plus `X-SUBCOR-Routed-Units`, `X-SUBCOR-Escalate` and `X-SUBCOR-Confidence` headers |
| `POST /v1/subcor/tenants/{t}/trajectories` | Append verified or unverified trajectories |
| `POST /v1/subcor/tenants/{t}/study` | Run a governed study cycle (sync, or `async: true` with a job to poll). Never enables a unit. |
| `GET /v1/subcor/tenants/{t}/units` | List units with base kernel, SHA-256, bits per weight and conformance report |
| `POST /v1/subcor/tenants/{t}/units/{u}/promote` | `CONFORMANT → ENABLED`. Returns 403 if the approver is the proposer. |
| `POST /v1/subcor/tenants/{t}/units/{u}/rollback` | Atomic `ENABLED → ROLLED_BACK` |
| `POST /v1/subcor/tenants/{t}/units/{u}/export-capheos` | Export a portable bundle (`module.json`, `weights.scun`, `router_gate.json`, `skill_spec.json`, conformance test) |

## What SUBCOR is not

- **Not a way to cram facts into weights.** Facts belong in context, retrieval and tools.
  Units hold *procedures*: routing, formatting, triage, policy checks.
- **Not a replacement for a frontier model.** It's built for the repeatable middle of the
  work, and it hands hard turns back.
- **Not a proven self-improving system.** Units do get better from verified experience, but
  we haven't yet shown gains compounding over many generations of units. See
  [`docs/BACKGROUND.md`](docs/BACKGROUND.md) for limitations and what didn't work.

## Documentation

| Document | Contents |
|---|---|
| [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) | Behavior contracts, invariants, ADRs and method details (the maths) |
| [`docs/BACKGROUND.md`](docs/BACKGROUND.md) | Inspiration, prior art, what's new, negative results, limitations |
| [`docs/BFCL-POC-REPORT.md`](docs/BFCL-POC-REPORT.md) | BFCL v3 results and ablations |
| [`docs/BENCHMARK-REPORT.md`](docs/BENCHMARK-REPORT.md) | `subcor-core-v1` growth benchmark on Apple M5 |
| [`docs/MULTITURN-MULTILAYER-REPORT.md`](docs/MULTITURN-MULTILAYER-REPORT.md) | Multi-turn masking and memory-safe multi-layer training on 16 GB |
| [`docs/AMD-GGUF.md`](docs/AMD-GGUF.md) | llama.cpp serving and PyTorch training on 12–16 GB GPUs |
| [`docs/CAPHEOS-INTEGRATION.md`](docs/CAPHEOS-INTEGRATION.md) | Serving multiple agent-platform instances |
| [`docs/ROADMAP.md`](docs/ROADMAP.md) | Milestones and exit gates |
| [`docs/ALORA-NOTE.md`](docs/ALORA-NOTE.md) | Research note: llama.cpp prompt-cache cost of unit switches, and activated LoRAs |
| [`experiments/`](experiments/README.md) | Preregistered BFCL v4 study, and Skill Units trained on real coding-agent sessions |

## Repository map

```text
src/subcor/
├── types.py            domain types, SkillSpec, lifecycle states, errors
├── telemetry.py        append-only trajectory store (verified evidence gate)
├── intern.py           text trajectory store, n-gram loop metrics
├── expansion.py        function-preserving low-rank expansion, FP32 and binary STE
├── surgeon.py          zero-gradient contrastive weight-space surgery
├── sub1bit.py          bit-packed binary low-rank factorization
├── router.py           training-free frozen-base gates and top-k router
├── confidence.py       turn confidence, escalation, tool-call parsing
├── units.py            governed unit registry: SHA-256, conformance, approval, rollback, export
├── auth.py             role- and tenant-scoped principals for the HTTP server
├── server.py           multi-tenant OpenAI-compatible server and governance endpoints
├── recompile.py        recompile units onto a new base model
├── benchmark.py        subcor-core-v1 suite, ledger, diffs, reports
├── baselines.py        controlled comparisons (sub-1-bit vs FP16 LoRA, BitDelta, few-shot)
├── bfcl.py             BFCL v3 loader, grader and training harness
├── gguf_export.py      Skill Unit → llama.cpp GGUF LoRA adapter
├── proving_ground.py   the synthetic `subcor demo`
├── live_loop.py        live MLX intern, MoE and surgery workflows
├── cli.py              the `subcor` command
└── backends/
    ├── mlx_kernel.py           Apple Silicon kernel (frozen base, adapters, streaming)
    ├── torch_kernel.py         PyTorch ROCm/CUDA/CPU kernel and trainer (QLoRA, multi-turn masking)
    ├── llamacpp_kernel.py      llama-server backend (chat, tools, SSE, LoRA routing)
    └── llamacpp_supervisor.py  owns and restarts the llama-server process
tests/                  unit, property and HTTP integration tests (CPU, no model downloads)
docs/                   see the table above
experiments/            research code: BFCL v4 study, chat-history units, serving measurements
```

## Contributing and security

Contributions are welcome. [`CONTRIBUTING.md`](CONTRIBUTING.md) covers setup, the checks to
run and the invariants a change must keep. Please report vulnerabilities privately, as
described in [`SECURITY.md`](SECURITY.md).

## License

- **The runtime is [AGPL-3.0-or-later](LICENSE).**
- **Exported Skill Units default to Apache-2.0**, which fits Apache-2.0 base checkpoints such
  as Gemma 4 and Qwen 3.5.
- **Model weights are not part of SUBCOR.** Base checkpoints (GGUF, MLX or Hugging Face) are
  downloaded separately and keep their own licenses, which every deployment must follow. A
  Skill Unit is trained against a specific base, so check that the base's terms allow the
  unit's license before distributing it.
- **Third-party software keeps its own license.** NumPy and the optional extras (MLX,
  PyTorch, Transformers, `gguf`, ...) are installed as dependencies, and `llama-server`
  (llama.cpp) runs as a separate binary you install yourself. None of them is vendored in
  this repository.
