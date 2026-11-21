# BFCL v4 harness (official data and graders, native function calling)

This harness tests whether SUBCOR's routed Skill Units improve a small local model's tool
use. It runs the official Berkeley Function Calling Leaderboard v4 test data and graders
from the `bfcl-eval` package (version `2026.3.23`). The harness adds no grader of its
own:

* Test data comes from bfcl's `load_dataset_entry`. That includes the language hints for
  Java and JavaScript and the prerequisite conversations for the memory categories.
* Grading is done by bfcl's `eval_runner.main`: the AST checker, relevance and
  irrelevance, the multi-turn state and response checkers with the simulated APIs, and
  the memory/agentic checker.
* Inference runs through bfcl's `BaseHandler` loops (single-turn, and multi-turn with the
  20-step limit and the miss-func holdout turns) and bfcl's `generate_results`
  scheduler.
* The Overall score uses bfcl's `generate_leaderboard_csv` weights: non-live 10, live 10,
  irrelevance 10, multi-turn 30, agentic 40.

Version check: the board page (last updated 2026-04-12) cites `bfcl-eval==2025.12.17`.
`diff -r` of the two wheels shows that `data/`, `ast_checker.py`, `multi_turn_checker.py`,
`eval_runner*.py` and the column and weighting code are identical. The differences are
model registrations, the `REMOTE_OPENAI_*` and LoRA plumbing in the OSS handler (2026.3.23
is needed for `--handler official` against an existing server), and one regex in
`multi_turn_utils` that sanitizes simulated-API instance names (`:` added). None of
these touch grading.

The harness adds four things: a native-FC handler, per-run isolation and resume,
item subsetting, and a scorer and paired export that report on bfcl's score files.

| file | role |
|---|---|
| `native_fc_handler.py` | `NativeFCHandler`, a subclass of bfcl's own `OpenAICompletionsHandler` |
| `run_bfcl.py` | CLI with the commands `generate`, `evaluate`, `summarize`, `all` and `pair` |
| `official_data_overall.csv` | snapshot of https://gorilla.cs.berkeley.edu/data_overall.csv (2026-11-21) |

## Setup (separate eval venv, never the project env)

```bash
S=/tmp/claude-1000/-home-caullen-Documents-Dev-GitHub-subcor/301b3745-882c-4027-b232-a6492d126d24/scratchpad/bfcl/harness
uv venv --python 3.11 $S/venv
# CPU torch wheel: bfcl pulls sentence-transformers. Without the CPU index, uv downloads several GB of CUDA wheels.
uv pip install --python $S/venv/bin/python \
  --extra-index-url https://download.pytorch.org/whl/cpu --index-strategy unsafe-best-match \
  bfcl-eval==2026.3.23 soundfile      # bfcl's qwen_agent import needs soundfile and does not declare it
```

## Run

Each `--run-dir` is a separate bfcl project root (`BFCL_PROJECT_ROOT`). It holds
`run_config.json` (endpoint and settings, plus the frozen item ids when subsetting),
`result/` (bfcl's per-category JSONL files, one line per item), `score/` (bfcl's
score files and `data_*.csv`), `summary.json` and `per_item.csv`.

Generation can be resumed: rerun the same command and the items already in `result/`
are skipped. `--retry-errors` regenerates only the items that hit an inference error.
When the settings differ from the stored `run_config.json`, the run is refused unless
you pass `--force-config`.

```bash
PY=$S/venv/bin/python; H=experiments/bfcl/harness/run_bfcl.py

# llama-server directly (start it with --jinja; per-slot ctx >= 32k for multi_turn_long_context)
$PY $H all --run-dir $S/runs/q35-4b-direct \
  --base-url http://127.0.0.1:8510/v1 --served-model q35-4b --thinking off \
  --categories non_live live multi_turn --num-threads 2

# SUBCOR serve: frozen base vs. a tenant with routed Skill Units (same server, same settings)
$PY $H all --run-dir $S/runs/subcor-base   --backend subcor --base-url http://127.0.0.1:8571/v1 \
  --served-model subcor/base   --thinking off --categories non_live live multi_turn
$PY $H all --run-dir $S/runs/subcor-tenant --backend subcor --base-url http://127.0.0.1:8571/v1 \
  --served-model subcor/acme   --thinking off --categories non_live live multi_turn

# paired per-item export plus a stratified bootstrap of the deltas (B minus A)
$PY $H pair --run-a $S/runs/subcor-base --run-b $S/runs/subcor-tenant --out $S/pairs/base-vs-acme.csv

# official bfcl handler, UNMODIFIED, against llama-server /v1/completions (parity check):
# QwenFCHandler is the handler behind the board's local Qwen3 "(FC)" rows.
$PY $H all --run-dir $S/runs/qwen3-4b-2507-official --handler official \
  --official-registry Qwen/Qwen3-4B-Instruct-2507-FC --base-url http://127.0.0.1:8510/v1 \
  --categories non_live live multi_turn --num-threads 2 \
  --reference-model "Qwen3-4B-Instruct-2507 (FC)"
```

Other options:

* Subsets: `--per-category N` takes a seeded random N items per category, and
  `--per-category-multi-turn M` overrides N for the multi-turn categories. Given alone,
  M subsets only multi-turn and every other category runs in full. You can also pass
  `--ids-file`, either `{category: [ids]}` or another run's `run_config.json`, whose
  frozen `selected_ids` are reused.
* `--system-prefix FILE` (A1 arm) prepends a system message to every item, or merges it
  ahead of the item's own system message. FILE is `a1_exemplars.json` (its
  `system_prompt`, sha256-checked) or plain text.
* Each `run_config.json` records `harness.files_sha256` and the `bfcl_eval` version.
  Resuming under changed harness code is refused unless you pass `--force-config`.
* `--reference-model "<board display name>"` adds a column-by-column comparison with the
  official CSV.
* `--extra-body '{"lora":[{"id":0,"scale":1.0}]}'` adds fields to the llama-server
  request, for example per-request LoRA.
* Categories: the default is `non_live live multi_turn`. You can add `memory`. The two
  `web_search_*` categories are always skipped because they need SerpAPI, and
  `format_sensitivity` applies only to prompt-mode models. Categories that are not run
  count as 0 in `Overall Acc`, exactly as in bfcl.

`summarize` also reports two numbers that are clearly labelled as not official:

* "Overall (no web search, renormalized)": the same weights without the web-search half
  of agentic.
* "overall with official web-search/memory imputed": this run's sections, with the
  reference row's Web Search Acc filled in (and its Memory Acc when memory was not run).

These are the like-for-like overall numbers to compare with a board row. The script
also checks its own aggregation against bfcl's `data_overall.csv` (`bfcl_csv_check`)
and checks the per-item correct counts against bfcl's score headers.

## Oracle self-test (no model, no server)

`run_bfcl.py selftest [--categories non_live live] [--limit N]` checks the conversion
path. It feeds bfcl's own ground truth through the harness:

1. For each item, it builds a fake `/v1/chat/completions` response whose `tool_calls`
   are the ground truth, with dotted names rewritten as the tools were.
2. It parses that response with `NativeFCHandler` and decodes it with `decode_ast`.
3. It grades the result with bfcl's `_evaluate_single_ast_entry` and relevance checker.

Result on 2026-11-21:

* 3,593 of 3,606 checked items pass.
* irrelevance, live_irrelevance, live_relevance, live_parallel, live_parallel_multiple,
  multiple, parallel, simple_python and simple_javascript score 100%.
* The remaining 13 are limits of the oracle, not harness bugs:
  * required parameters whose only listed answer is the "may be omitted" marker;
  * nested dicts with all-optional fields;
  * Java `long` answers that need an `L` suffix;
  * 2 parallel_multiple ground truths with a key that is not in the function doc.
* 35 Java/JS items with container values are skipped, because they would need Java/JS
  source rendering.

## Which official handler produced the board rows

Read this before claiming parity. The Qwen rows the task names (`Qwen3-4B-Instruct-2507 (FC)`,
`Qwen3-0.6B (FC)`, `Qwen3-8B (FC)`) were **not** produced through an OpenAI `tools` API.
In `bfcl_eval/constants/model_config.py` they are registered as `Qwen/...-FC` with
`QwenFCHandler`, a local OSS handler:

* It renders the Qwen3 chat template by hand. The tools are the **raw BFCL function
  docs** (`"type": "dict"`, BFCL type names, dotted function names kept), not
  OpenAI-converted tools.
* It posts the prompt to `/v1/completions` (vLLM/SGLang on the board).
* It parses `<tool_call>\n{json}\n</tool_call>` blocks with a regex.
* It leaves hybrid models in thinking mode: the prompt ends at
  `<|im_start|>assistant\n`.

The API entries with the same display names (`qwen3-*-FC`, `QwenAPIHandler`) use
DashScope instead.

So there are two ways to reproduce a board row:

1. `--handler official`, which is exact: bfcl's QwenFCHandler, unmodified, pointed at
   llama-server's `/v1/completions` through `REMOTE_OPENAI_BASE_URL`, with
   `--skip-server-setup` semantics. Only the serving engine and the weights differ (GGUF
   quant on llama.cpp instead of bf16 on vLLM). This needs a GGUF of the board model
   itself (for example Qwen3-4B-Instruct-2507). None is cached locally yet, and the HF
   tokenizer and config are downloaded on first use.
2. `--handler native`, which is the path SUBCOR is measured on. Its gap to (1) on the
   same GGUF measures how much changes when the template and parser move into
   llama-server. That gap has to be known before attributing anything to Skill Units.

## Handler deviations (NativeFCHandler vs. bfcl's `OpenAICompletionsHandler`)

Inherited unchanged:

* the inference loops;
* `_pre_query_processing_FC`: no system prompt is injected, and a test's own system
  turn is passed as-is;
* `_compile_tools`: `convert_to_tool(..., GORILLA_TO_OPENAPI, OPENAI_COMPLETIONS)`,
  which rewrites `.` to `_` in function names. The registry therefore sets
  `underscore_to_dot=True`, as bfcl does for GPT FC;
* `decode_ast`: `json.loads` of each call's `arguments`;
* `decode_execute`: `convert_to_function_call`;
* first-turn and next-turn user messages;
* `role="tool"` results zipped with the `tool_call_ids`;
* `is_fc_model=True`;
* the request fields `messages`, `tools` (only when non-empty), `temperature`, `store=False`.

The deviations below could shift scores. Each one is marked as changing requests or as
bookkeeping only.

| # | Deviation | Why | Score impact |
|---|---|---|---|
| 1 | `base_url`, served `model` and API key come from `run_config.json`. The `OPENAI_*` env vars are ignored. | target llama-server or SUBCOR (`subcor/base`, `subcor/<tenant>`) | none |
| 2 | `max_tokens=4096` is sent. The official handler sends none. Use `--max-tokens 0` to send none. | an unbounded local decoder can loop until the context is full. 4096 is the cap bfcl's own OSS handler uses | changes requests. Affects only runaway generations, which are graded wrong either way unless a valid call comes after 4096 tokens |
| 3 | `enable_thinking` is sent explicitly when `--thinking on/off` (`chat_template_kwargs` for llama-server, the top-level field for SUBCOR). `--thinking` is required | **llama-server b12110 thinks by default** (`--reasoning auto` sets `enable_thinking=true`, verified on this machine). **SUBCOR serve defaults to `enable_thinking=False`**. Leaving it implicit would silently compare a thinking run with a non-thinking run | changes requests. Large for hybrid models; must be identical across paired runs |
| 4 | The assistant message in the history is a plain dict `{role, content, tool_calls[, reasoning_content]}` rebuilt from the response. The official handler appends the SDK message object | portable across servers. `reasoning_content` is kept by default (`--reasoning-history keep`), as bfcl's QwenFCHandler keeps it and the Qwen templates decide whether to render it; `strip` matches bfcl's DeepSeek-style handlers | same content as the SDK object when the server returns ids. With thinking off there is no reasoning content, so no difference |
| 5 | Tool calls with no `id` get a synthetic `call_<hex>` id | defensive: the `tool` replies need ids. llama-server always returns ids, and so does SUBCOR, including its text-parsing fallback (`confidence.parse_tool_calls`) | none in practice |
| 6 | No `RateLimitError` backoff. The OpenAI SDK's default 2 retries on connection errors remain. Timeout 1800 s | local servers do not rate-limit | none |
| 7 | `parallel_tool_calls` is not sent (same as official). `--parallel-tool-calls true/false` can force it | verified: llama-server b12110 returns several calls without it | none by default |
| 8 | `harness_trace` is added to each result line: per-step `finish_reason`, plus SUBCOR's `subcor` routing block and its `X-SUBCOR-*` headers | needed to see which units were routed on which item | bookkeeping only. The graders never read it |
| 9 | The default temperature is 0.001 (bfcl `generate`'s default). The task text said 0, and llama.cpp treats 0.001 as near-greedy (top-k 40 / top-p 0.95 / min-p 0.05 still apply) | follow the official setting | negligible. Use `--temperature 0` for strict greedy |
| 10 | `--system-prefix` (A1 arm only) adds or extends the first system message | the few-shot baseline | changes requests for that arm only; `pair` treats it as the arm's identity |

What else differs from the board (not the handler):

* **The server owns the template, parser and decoding.** llama-server `--jinja` renders
  the GGUF's embedded template and parses tool calls with its per-template parser (for
  Qwen3.5, the XML `<function=...><parameter=...>` format). For the formats it
  recognizes, llama.cpp can also attach a lazy tool-call grammar that constrains
  arguments to the schema. Whether it does so for this template on b12110 was **not
  verified**; check with `-lv 4` logs before attributing parity gaps. The board's
  QwenFCHandler decodes without constraints and silently drops `<tool_call>` blocks
  whose JSON does not parse.
* Value typing with Qwen3.5's XML call format: parameter values are raw text that
  llama.cpp converts to JSON. In the smoke run, a boolean came back as the string
  `"True"` (`live_multiple_733-167-4`, graded `value_error`). The JSON-format
  `<tool_call>` that QwenFCHandler parses would have carried `true`. This is a template
  and parser effect that the board's Qwen3 rows do not have.
* Weights and engine: GGUF Q4_K_M on llama.cpp instead of bf16 on vLLM or SGLang.
  Expect a few points of drift even with the exact official handler.
* Context: the board uses each model's full context. Here each llama-server slot has
  `-c / -np` tokens. A prompt that exceeds it is a 400 error, which becomes
  `Error during inference` and a failed item. `multi_turn_long_context` needs per-slot
  ctx >= 32k.
* SUBCOR serve, measured from its source at `exp/bfcl-eval`:
  * it forces `temperature=0.0` (greedy) and ignores the request value;
  * it drops `reasoning_content` from its responses;
  * it does not forward `parallel_tool_calls` or other extra fields;
  * it defaults `max_tokens` to 128 (the harness always sends 4096);
  * it serializes all requests under one lock, so `--num-threads > 1` does not speed it up;
  * its `top_k` field is the number of routed units, not a sampling setting, so the
    harness never sends it.

  So compare `subcor/<tenant>` against `subcor/base` through the same SUBCOR server, not
  against llama-server direct.

## Paired export

`per_item.csv` (written by `summarize`) has one row per graded item with the columns
`registry, category, id, correct, error_type, inference_error`. bfcl's score files list
only the failed items after the header, so correct = graded ids minus failed ids. The
counts are asserted equal to bfcl's header.

`pair` joins two runs on `(category, id)` and writes `<out>.csv` with the columns
`a_correct, b_correct`. It also writes `<out>.json` with:

* per-category and per-section deltas, and Overall Acc;
* the non-official no-web-search overall;
* 95% and 97.5% CIs from a bootstrap stratified by category (`--bootstrap`, default
  10000);
* per-category win/tie/loss counts with exact McNemar p-values, raw and Holm-adjusted;
* the discordant counts.

`pair` refuses two runs that are not comparable arms. Everything except `base_url`,
`served_model`, `api_key_env` and `system_prefix*` must match: handler, temperature,
thinking, `max_tokens`, backend, timeout, `extra_body`, frozen item ids, categories, and
the harness code hash recorded in `run_config.json`. Use `--allow-config-diff KEY` to
accept a listed difference. `pair` also refuses runs that graded different item sets
(`--allow-unmatched` overrides). `--exclude-errors [--exclude-errors-from RUN ...]` is
the error-sensitivity analysis.

`routing --run-dir RUN [--expect-unit U] [--compare-run OTHER]` reads `harness_trace`
and writes `routing.csv` (per item: steps, fired steps, units, fired class all/some/none)
and `routing.json` (per-category activation rates). `--expect-unit` fails on any other
routed unit or a step without a `subcor` block. `--compare-run` reports the
identical-output rate against another run per fired class. A tenant arm must reproduce
the base arm on items where its gate never fired.

## Smoke test (CPU, 2026-11-21/22)

These are the smoke test runs. The model was Qwen3.5-0.8B Q4_K_M on llama-server b12110
(CPU: `-ngl 0 --device none --no-op-offload`, `-c 65536 -np 2`), with `--thinking off`
and temperature 0.001. Ledgers are in `results/`.

* `smoke-0.8b-native`: 5 items in each of the 13 single-turn categories and 2 in each of
  the 4 multi-turn categories (73 items). It ran end to end:
  * generate took 13 min 18 s, then evaluate and summarize ran;
  * no inference errors;
  * bfcl's `data_overall.csv` equals the harness aggregation on all 23 columns.

  Non-live accuracy is low (6.7% simple) because the 0.8B model in non-thinking mode
  mostly answers in prose instead of calling the tool
  (`ast_decoder:decoder_failed ... 'str' object has no attribute 'keys'`). That is model
  behavior, not a harness fault (see `selftest`).
* `smoke-0.8b-subcor-base`: 13 items through `subcor serve --backend llamacpp` with
  `subcor/base`. The `subcor` routing block and the `X-SUBCOR-*` headers are recorded
  per step. 3 of the 13 items failed with `Connection error`:
  `LlamaCppServingAdapter._post_json` uses a hard-coded **60 s** urllib timeout. The
  resulting `TimeoutError` is not in the server's `except` list, so the handler thread
  dies and the client sees a dropped connection. On CPU, any long generation hits it.
  On GPU it is rarer, but long thinking or 9B outputs can still exceed 60 s. Those
  items are recorded as `Error during inference`, scored wrong, and counted in
  `inference_errors`; `--retry-errors` regenerates them.
* `tiny-official`: 6 items through the unmodified bfcl `QwenFCHandler` (registered as
  `Qwen/Qwen3.5-0.8B-FC`, tokenizer from the local HF cache) against llama-server
  `/v1/completions`. It decodes and grades end to end.

## Limitations

* `web_search_*` is never run (it needs SerpAPI).
* Memory has been exercised once (`smoke-0.8b-memkv`). One `memory_kv` question pulled
  its 10 prerequisite conversations through the dependency closure (they run in
  order), and bfcl's agentic checker graded it (`agentic:answer_not_found`). One
  prerequisite needed 33,001 prompt tokens and failed against a 32k slot, so the memory
  categories need **>= 64k context per slot**. On CPU this one question took about 50
  minutes. `memory_vector` (sentence-transformers download) and `memory_rec_sum` were
  not exercised.
* With only 1 graded item, bfcl's own `generate_leaderboard_csv` crashes
  (`stdev` of a single latency). `evaluate` catches this, because the score files are
  already written by then; `summarize` still works, without `bfcl_csv_check`.
* The board's Qwen3 rows can only be reproduced exactly with `--handler official` on a
  GGUF of the same Qwen3 model. The cached Qwen3.5 GGUFs have no board row.
