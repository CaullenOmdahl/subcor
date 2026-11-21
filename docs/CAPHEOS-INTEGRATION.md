# Serving Two CaPheOS Instances on CachyOS (`10.1.100.10`)

SLAINGI runs as a **standalone multi-tenant HTTP server** outside the CaPheOS containers. Both CaPheOS instances on `10.1.100.10` share ONE frozen 652 MB `Qwen3.5-0.8B-4bit` base kernel while keeping their Skill Units, trajectories, and MoE routers strictly isolated per company.

---

## Topology

```text
CachyOS Host (10.1.100.10)                      Apple M5 Host (10.1.100.11)
┌──────────────────────────────┐               ┌────────────────────────────────────────┐
│ CaPheOS Instance 1           │               │ SLAINGI Multi-Tenant Server (:8420)    │
│   provider: openai-compat    │               │   Frozen Base: Qwen3.5-0.8B-4bit       │
│   baseURL:  http://10.1.100.11:8420/v1  ───► │                                        │
│   model:    slaingi/company-1│               │   tenants/company-1/  (isolated units) │
├──────────────────────────────┤               │   tenants/company-2/  (isolated units) │
│ CaPheOS Instance 2           │               └────────────────────────────────────────┘
│   provider: openai-compat    │
│   baseURL:  http://10.1.100.11:8420/v1  ───►
│   model:    slaingi/company-2│
└──────────────────────────────┘
```

---

## 1. Start the SLAINGI Server on `10.1.100.11`

Generate a bearer token once (store it in your password manager and in each CaPheOS instance's `secrets/` directory, `chmod 600`):

```bash
export SLAINGI_API_KEY="$(openssl rand -hex 24)"

cd /Users/caullen/Documents/Dev/GitHub/slaingi
uv run --extra mlx slaingi serve \
  --host 10.1.100.11 \
  --port 8420 \
  --state-dir ~/.local/share/slaingi \
  --tenant company-1 \
  --tenant company-2
```

Verify readiness from `10.1.100.10`:

```bash
curl -s http://10.1.100.11:8420/readyz | jq
```

---

## 2. Connect Both CaPheOS Instances on `10.1.100.10`

In each CaPheOS instance's console (**Connections → Add OpenAI-Compatible Provider**):

| Setting | CaPheOS Instance 1 | CaPheOS Instance 2 |
|---|---|---|
| Base URL | `http://10.1.100.11:8420/v1` | `http://10.1.100.11:8420/v1` |
| API Key | `secrets/slaingi_api_key` | `secrets/slaingi_api_key` |
| Model ID | `slaingi/company-1` | `slaingi/company-2` |
| Seat Assignment | `intern` (low-risk / first-draft seat) | `intern` (low-risk / first-draft seat) |

- Using `slaingi/company-1` vs. `slaingi/company-2` guarantees **strict tenant isolation**: Skill Units promoted in Company 1 are never mounted or visible in Company 2.
- Use `slaingi/base` to bypass all Skill Units and query the pure frozen base kernel directly.

---

## 3. Governed Learning Cycle (API Calls from CaPheOS `tools` & `risk` Seats)

### A. Ingest Verified Senior Corrections (`tools` seat)
```bash
curl -s -X POST http://10.1.100.11:8420/v1/slaingi/tenants/company-1/trajectories \
  -H "Authorization: Bearer $SLAINGI_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "trajectories": [
      {
        "task_family": "ops_triage",
        "prompt": "Event: Cloud vendor invoice #4821 exceeds spending cap.",
        "response": "...",
        "senior_correction": "{\"seat\":\"finance\",\"gate\":\"owner\",\"code\":\"FIN-CAP\"}",
        "evidence": {"verified": true, "verifier": "senior_seat_ceo", "method": "senior_seat_review"}
      }
    ]
  }'
```

### B. Run Off-Peak Study Cycle (`tools` seat -> lands in `CONFORMANT`, never self-enabled)
```bash
curl -s -X POST http://10.1.100.11:8420/v1/slaingi/tenants/company-1/study \
  -H "Authorization: Bearer $SLAINGI_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "unit_id": "ops-triage-v1",
    "version": "1.0.0",
    "task_family": "ops_triage",
    "proposer": "tools",
    "steps": 108
  }'
```

### C. Approve & Hot-Swap Live (`risk` seat — self-approval by `tools` returns HTTP 403)
```bash
curl -s -X POST http://10.1.100.11:8420/v1/slaingi/tenants/company-1/units/ops-triage-v1/promote \
  -H "Authorization: Bearer $SLAINGI_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"approver": "risk"}'
```

### D. Instant Atomic Rollback (`risk` seat)
```bash
curl -s -X POST http://10.1.100.11:8420/v1/slaingi/tenants/company-1/units/ops-triage-v1/rollback \
  -H "Authorization: Bearer $SLAINGI_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"actor": "risk", "reason": "Regression detected in morning review"}'
```

---

## 4. Network & Security Checklist

1. **Always set `SLAINGI_API_KEY`** when binding `--host 10.1.100.11`.
2. **Restrict macOS firewall / `pf` on `10.1.100.11`** to allow TCP `8420` only from `10.1.100.10` (and localhost).
3. **Never place company secrets inside prompts** sent to telemetry; SLAINGI's `export-capheos` strips telemetry and exports only `module.json`, `weights.slun`, `router_gate.json`, and `conformance/test.js`.
4. **Schedule study cycles off-peak**: MLX shares a single Apple Silicon GPU queue, so running `study` (~45–90s) during overnight hours avoids latency spikes on live Intern chat turns.
