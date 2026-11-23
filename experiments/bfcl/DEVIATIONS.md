# Deviations and execution log (PREREGISTRATION.md section 12 companion)

Each entry gives the date, the change, the reason, and whether any treatment-arm (A1-A4)
result had been seen when it was made. `$S` = `<scratchpad>/bfcl`.

## D1 (2026-11-22, before any GPU result): V2 timeout fix done by the executor, not committed

- `src/subcor/cli.py`: `subcor serve` gains `--llamacpp-timeout SECONDS` (default 60.0, the
  old hard-coded value), passed to `LlamaCppServingAdapter(timeout=...)` in both the
  supervised and external branches. Every SUBCOR arm and C-nat run uses
  `--llamacpp-timeout 600`.
- Run order step 1 says "commit and record the hash". The executor may not commit; the
  change is in the working tree and is left for the orchestrator to commit. The harness
  and SUBCOR code are instead frozen by content hash: each `run_config.json` records
  `harness.files_sha256`, and the sha256 of `src/subcor/cli.py` /
  `src/subcor/backends/llamacpp_kernel.py` / `src/subcor/server.py` are recorded in the
  ledger `$S/ledger.jsonl` at every arm start.
- The `TimeoutError` is still not caught by the server's `except` list; with 600 s it
  should not occur. Any occurrence is counted against the V2 error budget.
- Treatment-arm results seen: none.

## D2 (2026-11-22, before any GPU result): run directory location

- Run directories are `$S/results/<run>/` (orchestrator instruction) instead of
  `$S/runs/<run>/` in section 10. Path only; no effect on any number.

## D3 (2026-11-22, before any GPU result): preregistration not yet in a commit

- `experiments/` is untracked in git at execution start, so "the commit that adds this
  file" does not exist yet. The file content at execution start has
  sha256 recorded in the ledger (`prereg_sha256`). Sections 1-11 were not edited.

## D4 (2026-11-22, before any GPU result): C-off llama-server flags

- Section 3 does not fix the llama-server flags for C-off. C-off uses the same flags the
  SUBCOR supervisor builds for the arms, so C-off wall time projects the arms' cost and
  C-nat - C-off is not confounded by server flags:
  `llama-server --model <gguf> -ngl 99 --ctx-size 65536 -np 1 --embeddings --pooling mean --jinja --no-webui`.
  (`--embeddings` makes llama-server force `n_batch = n_ubatch = 512`; this only affects
  prefill speed.)
