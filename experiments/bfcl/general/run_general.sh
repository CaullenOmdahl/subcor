#!/usr/bin/env bash
# General-capability regression evals (IFEval 541, GSM8K 500 seeded, router false-activation
# probe ~300) against an OpenAI-compatible endpoint. Per-item JSONL for paired bootstrap.
#
# Usage:
#   run_general.sh <arm-label> <base_url> <model> [limit]
#     arm-label : free name, e.g. base-llama | subcor-base | subcor-routed
#     base_url  : .../v1 of plain llama-server or `subcor serve`
#     model     : 'local' for llama-server; 'subcor/base' or 'subcor/<tenant>' for SUBCOR
#     limit     : optional N for a smoke run (first N items of each eval)
#   ROUTER_PROBE=1 also runs the probe (only meaningful for a subcor/<tenant> model).
#   TENANT_DIR=<state>/tenants/<tenant> adds gate-threshold margins to the probe summary.
#
# Compare arms:   $VENV/bin/python -I paired_bootstrap.py A.jsonl B.jsonl [--metric ...]
#
# Determinism: run every arm with the same --concurrency (default 1) and the same
# llama-server -np; batched decoding is not bit-reproducible across batch compositions.
# Expected servers (CPU smoke):
#   llama-server -m Qwen3.5-0.8B-Q4_K_M.gguf --port 8502 --jinja \
#     --chat-template-kwargs '{"enable_thinking":false}' --embeddings --pooling mean \
#     -ngl 0 --device none --no-op-offload -c 16384 -np 1
#   (cd <repo> && uv run --extra dev --extra rocm subcor serve --backend llamacpp \
#     --llamacpp-url http://127.0.0.1:8502 --model <id> --port 8592 --state-dir <state>)
set -euo pipefail

ARM=${1:?arm label}; BASE_URL=${2:?base_url}; MODEL=${3:?model}; LIMIT=${4:-}
HERE=$(cd "$(dirname "$0")" && pwd)
SCRATCH=${SCRATCH:-/tmp/claude-1000/-home-caullen-Documents-Dev-GitHub-subcor/301b3745-882c-4027-b232-a6492d126d24/scratchpad/bfcl}
VENV=${VENV:-$SCRATCH/venv}
DATA=${DATA:-$SCRATCH/general/data}
OUT=${OUT:-$SCRATCH/general/results/$ARM}
CONC=${CONCURRENCY:-1}
PY="$VENV/bin/python -I"
LIM=(); [[ -n "$LIMIT" ]] && LIM=(--limit "$LIMIT")
mkdir -p "$OUT"
cd "$OUT"  # never run from a directory holding downloaded data

# max_tokens are explicit (PREREGISTRATION.md 4.3): IFEval 1280, GSM8K 1024.
$PY "$HERE/run_ifeval.py" --base-url "$BASE_URL" --model "$MODEL" --run-label "$ARM" --max-tokens 1280 \
  --concurrency "$CONC" --data "$DATA/ifeval.jsonl" --out "$OUT/ifeval.jsonl" "${LIM[@]}"
$PY "$HERE/run_gsm8k.py" --base-url "$BASE_URL" --model "$MODEL" --run-label "$ARM" --max-tokens 1024 \
  --concurrency "$CONC" --data "$DATA/gsm8k_test.jsonl" --out "$OUT/gsm8k.jsonl" "${LIM[@]}"
if [[ "${ROUTER_PROBE:-0}" == 1 ]]; then
  TD=(); [[ -n "${TENANT_DIR:-}" ]] && TD=(--tenant-dir "$TENANT_DIR")
  $PY "$HERE/run_router_probe.py" --base-url "$BASE_URL" --model "$MODEL" --run-label "$ARM" \
    --concurrency "$CONC" --prompts "$DATA/router_probe_prompts.jsonl" --out "$OUT/router_probe.jsonl" \
    "${TD[@]}" "${LIM[@]}"
fi
echo "summaries: $OUT/*.summary.json"
