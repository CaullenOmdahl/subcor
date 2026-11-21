# Serving `ca-phe-digital` & `capheos-sandbox` on CachyOS (`10.1.100.10`)

SLAINGI runs on your **Apple M5 (`10.1.100.11`)** as a single multi-tenant HTTP server wrapping one frozen `652 MB` **`Qwen3.5-0.8B-4bit`** kernel, while isolating Skill Units, telemetry, and MoE routers per company:

- **`ca-phe-digital`** (CachyOS console `:8900`, station `:8800`) → model `slaingi/ca-phe-digital`
- **`capheos-sandbox`** (CachyOS console `:8910`, station `:8810`) → model `slaingi/capheos-sandbox`

Because CaPheOS's `custom` provider slot is already used for MiniMax, SLAINGI plugs into StarNet's built-in **`ollama` provider (`http://127.0.0.1:11434/v1`, unmetered, zero collision with MiniMax)**. Both `/v1/models` and Ollama's `/api/tags` are served natively.

---

## Option 1 (Recommended): Encrypted SSH Loopback Tunnel (Zero Open LAN Ports)

Port `11434` is free on `10.1.100.10`. Keeping SLAINGI bound strictly to `127.0.0.1:8420` on `10.1.100.11` and reverse-forwarding `127.0.0.1:11434` onto `10.1.100.10` means:
- Zero open LAN ports on macOS
- Zero systemd changes on `10.1.100.10`
- Zero edits to `.starnet-data/workspaces/.secrets/` (never trips `locked-watch.js`)

### 1. Start SLAINGI on Apple M5 (`10.1.100.11`)
```bash
cd /Users/caullen/Documents/Dev/GitHub/slaingi
uv run --extra mlx slaingi serve \
  --host 127.0.0.1 \
  --port 8420 \
  --state-dir ~/.local/share/slaingi \
  --tenant ca-phe-digital \
  --tenant capheos-sandbox
```

### 2. Forward `127.0.0.1:11434` on `10.1.100.10` → `127.0.0.1:8420` on `10.1.100.11`
```bash
ssh -N -o ExitOnForwardFailure=yes -o ServerAliveInterval=30 \
  -R 127.0.0.1:11434:127.0.0.1:8420 caullen@10.1.100.10
```

### 3. Verify from `10.1.100.10`
```bash
curl -s http://127.0.0.1:11434/readyz | jq
curl -s http://127.0.0.1:11434/v1/models | jq
```

---

## Option 2: Direct LAN Bind (`10.1.100.11:8420`) + `OLLAMA_BASE_URL`

If you prefer direct LAN traffic (4.4 ms RTT) instead of an SSH tunnel:

1. Start SLAINGI on `10.1.100.11` with `--host 10.1.100.11 --port 8420` (firewall TCP `8420` to `10.1.100.10` only).
2. Add `Environment=OLLAMA_BASE_URL=http://10.1.100.11:8420/v1` to the `capheos-sandbox-station` and `ca-phe-digital-station` systemd units on `10.1.100.10` and restart them.

---

## Registering the Intern Model in Each Company's `model-policy.json`

In `capheos-sandbox/company/model-policy.json` (and `ca-phe-digital/company/model-policy.json`), under `"allowed"` (approved by CEO/owner per `company/MODEL_POLICY.md`):

```json
"ollama": {
  "models": ["slaingi/capheos-sandbox"],
  "note": "Local SLAINGI 0.8B Intern server (unmetered, MoE-routed Sub-1-Bit Skill Units)"
}
```
*(Use `"slaingi/ca-phe-digital"` in `ca-phe-digital/company/model-policy.json`.)*

---

## Governed Learning & Benchmarking Loop

```bash
# Baseline longitudinal benchmark for capheos-sandbox
uv run slaingi bench run \
  --endpoint http://127.0.0.1:8420/v1 \
  --model slaingi/capheos-sandbox \
  --label "day0-sandbox-base" \
  --ledger ~/.local/share/slaingi/benchmarks/capheos-sandbox.jsonl

# Promote a CONFORMANT unit after risk sign-off (self-approval by tools returns 403)
curl -s -X POST http://127.0.0.1:8420/v1/slaingi/tenants/capheos-sandbox/units/<unit-id>/promote \
  -H "Content-Type: application/json" \
  -d '{"approver": "risk"}'

# Atomic rollback
curl -s -X POST http://127.0.0.1:8420/v1/slaingi/tenants/capheos-sandbox/units/<unit-id>/rollback \
  -H "Content-Type: application/json" \
  -d '{"actor": "risk", "reason": "Regression detected"}'
```
