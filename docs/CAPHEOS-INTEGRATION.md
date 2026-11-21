# Serving Multiple CaPheOS Instances

SUBCOR runs as a **standalone multi-tenant HTTP server** outside CaPheOS containers, sharing one frozen `652 MB` **`Qwen3.5-0.8B-4bit`** base kernel across multiple isolated company tenants:

- **Instance A (`company-a`)** → model `subcor/company-a`
- **Instance B (`company-b`)** → model `subcor/company-b`

Because CaPheOS's `custom` provider slot may already be assigned to another provider (e.g., MiniMax), SUBCOR plugs cleanly into StarNet's built-in **`ollama` provider (`http://127.0.0.1:11434/v1`, unmetered)** by serving both `/v1/models` and Ollama's `/api/tags`.

---

## Option 1 (Recommended): Encrypted SSH Loopback Tunnel

Keep SUBCOR bound strictly to `127.0.0.1:8420` on the GPU/Apple Silicon host and reverse-forward `127.0.0.1:11434` onto the CaPheOS host over SSH:
- Zero open LAN ports on the model host
- Zero systemd changes on the CaPheOS host
- Zero edits to `.starnet-data/workspaces/.secrets/` (never trips `locked-watch.js`)

### 1. Start SUBCOR on the Model Host (Apple Silicon / GPU)
```bash
uv run --extra mlx subcor serve \
  --host 127.0.0.1 \
  --port 8420 \
  --state-dir ~/.local/share/subcor \
  --tenant company-a \
  --tenant company-b
```

*(On macOS, install the server venv outside `~/Documents`—e.g. `~/.local/share/subcor/venv`—if running under `launchd` to avoid macOS TCC `Documents` restrictions.)*

### 2. Forward `127.0.0.1:11434` on the CaPheOS Host → `127.0.0.1:8420` on the Model Host
```bash
ssh -N -o ExitOnForwardFailure=yes -o ServerAliveInterval=30 \
  -R 127.0.0.1:11434:127.0.0.1:8420 <user>@<capheos-host>
```

### 3. Verify from the CaPheOS Host
```bash
curl -s http://127.0.0.1:11434/readyz | jq
curl -s http://127.0.0.1:11434/v1/models | jq
node company/tools/hire.js models   # should list subcor/company-a and subcor/company-b under "ollama"
```

---

## Option 2: Direct LAN Bind + `OLLAMA_BASE_URL`

1. Start SUBCOR with `--host <model-host-lan-ip> --port 8420` (restricting firewall ingress on TCP `8420` to `<capheos-host>` only).
2. Set `Environment=OLLAMA_BASE_URL=http://<model-host-lan-ip>:8420/v1` on each CaPheOS station systemd unit and restart.

---

## Registering the Intern Model in CaPheOS Governance

`company/model-policy.json` is watched by `locked-watch.js` (Class F, owned by the `people` seat). Approve and add `ollama` through your sanctioned `people`/owner workflow:

```json
"ollama": {
  "models": ["subcor/company-a"],
  "note": "Local SUBCOR 0.8B Intern server (unmetered, MoE-routed Sub-1-Bit Skill Units)"
}
```

---

## Governed Learning & Longitudinal Benchmarking

```bash
# Record Day-0 longitudinal benchmark snapshot
uv run subcor bench run \
  --endpoint http://127.0.0.1:8420/v1 \
  --model subcor/company-a \
  --label "day0-frozen-base" \
  --ledger ~/.local/share/subcor/benchmarks/company-a.jsonl

# Promote a CONFORMANT unit after independent risk sign-off (self-approval returns HTTP 403)
curl -s -X POST http://127.0.0.1:8420/v1/subcor/tenants/company-a/units/<unit-id>/promote \
  -H "Content-Type: application/json" \
  -d '{"approver": "risk"}'

# Instant atomic rollback
curl -s -X POST http://127.0.0.1:8420/v1/subcor/tenants/company-a/units/<unit-id>/rollback \
  -H "Content-Type: application/json" \
  -d '{"actor": "risk", "reason": "Regression detected"}'
```
