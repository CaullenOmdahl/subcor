# Pre-registration: do routed SUBCOR Skill Units improve a small local model's tool use on BFCL v4?

Status: **frozen before any GPU result exists** (written 2026-11-22). The git commit that
adds this file is the timestamp. After that commit, sections 1-10 may change only through
dated entries in section 12 (Deviations log), and each entry must say whether any result
of a treatment arm had been seen when the change was made.

Only CPU smoke runs (Qwen3.5-0.8B, at most 73 BFCL items, 10 IFEval/GSM8K items) existed
when this was written. They were used to debug the harness and are not data for this
study.

---

## 1. Question and hypotheses

SUBCOR claims that a Skill Unit (a low-rank delta trained from tool-use conversations,
stored as SUB1BIT, and switched on per request by a RouterGate on the frozen base's
embedding of the last user message) improves a small model's tool use without hurting
general behavior. It also claims this beats cheaper alternatives.

- **H1 (primary).** On Qwen3.5-4B Q4_K_M, arm A3 (routed SUB1BIT unit) scores higher than
  A0 (frozen base) on BFCL v4 multi-turn, or on the renormalised overall (section 6.1).
  The margin must be at least 5 points and the 95% CI lower bound above 0.
- **H2.** A3 is at least as good as A1, a 3-shot prompt with no training, on the
  renormalised overall.
- **H3.** A3 is no more than 2 points below A2, a plain always-on FP16 LoRA trained
  identically, on the renormalised overall. This is the cost of SUB1BIT compression plus
  routing.
- **H4 (no regression).** A3 drops IFEval and GSM8K by no more than 1 point, with the CI
  lower bound at or above -2.
- **H5 (router specificity).** A3's unit fires on fewer than 5% of about 300 general,
  non-tool prompts (the probe), **and** on fewer than 5% of the 541 IFEval prompts, which
  come from no source the gate was fitted on (see Review log, item 9).

**Null expectation.** Fine-tuning on public function-calling data usually moves BFCL
single-turn AST a few points for small models. Multi-turn (stateful file system, trading
and travel APIs, up to 20 steps) is far from the training distribution (section 4.4). A
null or negative H1 result is plausible, and it will be reported with the same prominence
as a positive one.

## 2. Models and fixed artifacts

| role | model | file | sha256 |
|---|---|---|---|
| primary | Qwen3.5-4B Q4_K_M (unsloth) | `Qwen3.5-4B-Q4_K_M.gguf` (32 blocks, full attention every 4) | `00fe7986ff5f6b463e62455821146049db6f9313603938a70800d1fb69ef11a4` |
| secondary | Qwen3.5-0.8B Q4_K_M (unsloth) | `Qwen3.5-0.8B-Q4_K_M.gguf` (24 blocks) | `bd258782e35f7f458f8aced1adc053e6e92e89bc735ba3be89d38a06121dc517` |
| calibration | Qwen3-4B-Instruct-2507 Q4_K_M | `unsloth/Qwen3-4B-Instruct-2507-GGUF` (fallback `bartowski/Qwen_Qwen3-4B-Instruct-2507-GGUF`) | record at download |
| calibration | Qwen3-0.6B Q4_K_M | `unsloth/Qwen3-0.6B-GGUF` (fallback `bartowski/Qwen_Qwen3-0.6B-GGUF`) | record at download |

Qwen3.5-9B is out of scope for this study.

Fixed software:

- llama.cpp b12110, Vulkan build, `/usr/bin/llama-server`.
- `bfcl-eval==2026.3.23`, with grading and data byte-identical to the board's
  `2025.12.17`; see `harness/README.md`.
- lm-eval 0.4.13 for the IFEval checker.
- SUBCOR at the commit that adds this file, plus the required timeout fix in section 8, V2.

Fixed data, under `$S = <scratchpad>/bfcl`:

| file | sha256 |
|---|---|
| `data/train.jsonl` (2,408 conv.) | `fb892d6c0d8ef00e1d8847ce86d1df1c8a1c0a4bb2fe0275a065c48cc39a0a7a` |
| `data/dev.jsonl` (205 conv.) | `96bcfb7a8152a6f03b8702d7864bf5fa57419824b75b765f18e544febb8770f6` |
| `data/router_pos.jsonl` (300; 240 fit / 60 heldout) | `c52f87e6cdbfa56621cdff7e9cbf51f3548f9c8ec54ac80a320c2abe65458090` |
| `data/router_neg.jsonl` (300; 240 fit / 60 heldout) | `d1d23e3291296955f6a67ac69d8ccd53e1b5fde50b39dcfd9f383331b0b4a42f` |
| `data/router_pos_dev.jsonl` (100) | `e1a1651f5ef6842524a1bb2309148e4b96eb5265389ec901ae63be05ded47b8d` |
| `experiments/bfcl/a1_exemplars.json` | in repo; system prompt sha256 `e418c8a1db1ed1f4c9753de6fc9a6a868bb4d360e473a3d873edc1a98de0dd22` |
| `experiments/bfcl/general/gsm8k_subset_seed1234.json` | in repo |
| `experiments/bfcl/data/results/decontam_semantic_drop.json` (30 ids: 27 train, 3 dev) | `1c877a6c3fabffe526cf9aacf543c39312d12574e4ed0c7a194d1e52e7c9e86f` |
| `general/data/router_probe_prompts.jsonl` (300, rebuilt; ids in `general/probe_set_ids.json`) | `711e62d56ef01e5adba6fc74564caf49e4571a690a951032d2445f61b893c3f7` |

The probe set was rebuilt during the review (section 9, R3); zero overlap is verified.

## 3. Calibration: establishing harness validity

The board's Qwen "(FC)" rows came from bfcl's local `QwenFCHandler`. That handler renders
the Qwen3 template by hand, posts to `/v1/completions` and regex-parses `<tool_call>`
JSON. They did not come from OpenAI `tools`; see `harness/README.md`. Calibration
therefore has two parts.

**C-off (official path).** `run_bfcl.py all --handler official` runs bfcl's unmodified
handler against llama-server `/v1/completions`, with bfcl's default sampling (temperature
0.001) and thinking as the handler leaves it:

| model | registry | thinking |
|---|---|---|
| Qwen3-4B-Instruct-2507 | `Qwen/Qwen3-4B-Instruct-2507-FC` | none; non-thinking model |
| Qwen3-0.6B | `Qwen/Qwen3-0.6B-FC` | **on**, as the board ran it |

**C-nat (native-FC path).** The same GGUF goes through the path the arms use:
`--handler native` against SUBCOR serve `subcor/base`, temperature 0, with thinking
matched to C-off. That means `off` for 4B-2507 and `on` for 0.6B. The 0.6B is also run
once with thinking `off` to measure the thinking effect.

**Items.** The full category sets of section 5. Multi-turn may use the frozen subset only
under the compute rule of section 5.

**Tolerance.** It is pre-registered against the board rows (`official_data_overall.csv`,
snapshot 2026-11-21). Each cell is |our − board| in points:

| section | Qwen3-4B-Instruct-2507 (FC) board | Qwen3-0.6B (FC) board | tolerance |
|---|---|---|---|
| Non-live (AST, 4-way mean) | 87.88 | 71.79 | ≤ 3.0 |
| Live (AST, n-weighted) | 76.39 | 56.62 | ≤ 3.0 |
| Multi-turn (4-way mean) | 22.12 | 3.62 | ≤ 5.0 |
| Irrelevance detection | 84.93 | 80.84 | ≤ 5.0 (reported; not a gate) |

These tolerances allow for Q4_K_M quantization and llama.cpp numerics against bf16 on
vLLM. They also cover sampling noise: SE ≈ 1 point on Non-live and Live, and ≈ 1.5 points
on 800 multi-turn items at p ≈ 0.2. The 0.6B multi-turn cell is nearly uninformative,
because its board value is 3.6. So **Qwen3-4B-Instruct-2507 is the deciding calibration
model**, and 0.6B is supporting evidence.

**Decision rules.** These are fixed now.

1. **C-off within tolerance on all three gated sections for 4B-2507.** The data, grader
   and llama-server pipeline count as validated, and BFCL section scores may be called
   "board-comparable in method".
2. **C-off outside tolerance.**
   - Diagnose only harness and serving configuration: template, `max_tokens`, context per
     slot, stop tokens, tokenizer path, inference errors.
   - Rerun the Q8_0 GGUF of the same model as a quantization control.
   - If Q8_0 is within tolerance and Q4_K_M is not, attribute the gap to quantization and
     proceed, stating this.
   - At most two fix-and-rerun iterations are allowed. Any fix applies to every later
     run.
   - If it still fails, every BFCL number in the report is labelled **"BFCL v4 data and
     grader, not comparable to the leaderboard"**. The paired arm comparisons (section 6)
     remain valid as within-harness comparisons, because all arms share the harness, and
     the report must say so.
3. **C-nat − C-off (the path gap).** Report it per section. If |gap| > 5 on any section
   for 4B-2507, the native path is "not board-equivalent". Arm numbers must not be placed
   next to board rows, but the paired comparisons still stand. A native path that is
   much *worse* than official is also a red flag for headroom: an adapter could "win" by
   repairing serving or parser artifacts rather than tool skill. See section 9, R7.
4. **The calibration verdict is recorded in section 12 before any treatment arm runs.**

**What calibration does not show.** Qwen3.5 has no board row, and its llama.cpp parser
uses an XML `<function=...><parameter=...>` format. Qwen3 uses JSON `<tool_call>`.
Calibration therefore validates the data, grader, pipeline and native-FC plumbing, not
the Qwen3.5 parser. The Qwen3.5 A0 failure audit (section 9, R7) covers that gap.

## 4. Arms

### 4.1 Qwen3.5-4B (primary)

| arm | what is served | request path |
|---|---|---|
| **A0** | frozen base, no units | SUBCOR serve, `model=subcor/base` |
| **A1** | frozen base + fixed 3-shot system prompt (`a1_exemplars.json`) | SUBCOR serve, `subcor/base` |
| **A2** | base + FP16 LoRA, always on at scale 1.0 | primary: SUBCOR serve `subcor/base` on a llama-server started with `--lora <a2>.gguf` (scale 1); fallback in V5 |
| **A3** | SUBCOR tenant with one enabled unit `bfcl-tools-v1-sub1bit` (SUB1BIT); the RouterGate decides per request | SUBCOR serve, `model=subcor/<tenant-a3>` |
| **A4** | same as A3, but the unit is `bfcl-tools-v1-fp32` | SUBCOR serve, `model=subcor/<tenant-a4>` |

**Routing.** No seat oracle: no `X-CaPheOS-Seat` header is sent, gates are not
seat-scoped, and `top_k = 1`. One unit is enabled per tenant. Routing happens on every
request, so within a multi-turn item the unit can be on for some steps and off for
others. That is the deployed behavior, and per-step decisions are logged in
`harness_trace`.

**Tenants.** `train_unit.py` writes both variants into one registry. `tenant-a3`'s
`units/` directory receives **only** `bfcl-tools-v1-sub1bit` (payload, manifest,
`router_gate.json`) and `tenant-a4`'s **only** `bfcl-tools-v1-fp32`. A tenant holding
both would route by tie-break between two identical gates. After each tenant arm,
`run_bfcl.py routing --expect-unit <unit>` must report 0 problems: every request carried
a `subcor` routing block, and nothing but the expected unit was ever selected (V11).

**Server instances.** All arms use one binary, one GGUF and these llama-server flags.
`subcor serve` builds them itself in supervised mode:
`subcor serve --backend llamacpp --gguf <GGUF> --n-gpu-layers 99 --ctx-size 65536
--llama-server-arg=-np --llama-server-arg=1 --llamacpp-port 85xx --port 85yy`, plus a
fresh state copy (V3). The supervisor adds `--embeddings --pooling mean --jinja
--no-webui`.
- **S1** serves A0, A1, A3 and A4 from one process: both tenants, plus `subcor/base`.
  So A3 and A4 differ from A0 only by the routed unit. The other unit's adapter is loaded
  at scale 0 and sent as scale 0 on every request.
- **S2** serves A2. It uses the same command with no tenants, plus
  `--llama-server-arg=--lora --llama-server-arg=<a2.gguf>`, so the adapter is at global
  scale 1 and SUBCOR sends no `lora` field (V5).

Never use `subcor serve`'s default port 8420.

**How the arms relate.** A2, A3 and A4 come from **one training run per seed**.
`train_unit.py` produces the FP32 adapters, packs them to SUB1BIT (A3), keeps FP32 (A4)
and writes the A2 GGUF LoRA from the same adapters. The A2 file has **f32** tensors
(`--a2-dtype f32`, the default since the review), which are the tensors SUBCOR's
supervisor exports for the A4 unit. The training budget is therefore identical by
construction. The A3 and A4 gates are fit on frozen-base embeddings of the same prompts,
so they should route identically; this is checked in V6. The design isolates two
effects:

| difference | isolates |
|---|---|
| A4 − A2 | routing only (on when the gate fires, versus always on); same f32 adapter tensors, checked in V10 |
| A3 − A4 | SUB1BIT compression at fixed routing |
| A3 − A0 | the full SUBCOR effect |

**A1 rendering.**
- The frozen `system_prompt` text is prepended as the first system message.
- If a BFCL item already has a system message, one system message is sent: the exemplar
  text, a blank line, then the item's own content.
- The exemplars were chosen by `a1_select_exemplars.py` (seed 20261122): one each of
  single, parallel_multiple and irrelevance, short single-turn rows from train.jsonl
  whose tool names appear nowhere in the BFCL v4 function docs (2,910 names checked).
- The selection was **not** re-rolled for quality.
- The harness implements this as `--system-prefix experiments/bfcl/a1_exemplars.json`
  (`NativeFCHandler.add_first_turn_message_FC`). It reads `system_prompt`, checks it
  against the recorded `system_prompt_sha256`, and stores the text and hash in
  `run_config.json`. The prefix goes on the first turn only, so in multi-turn items it
  stays in the history for every later request.
- A1 sends exemplars only on requests that carry tools, which all BFCL requests do. That
  makes it identical to A0 on IFEval and GSM8K, so A1 is not run there.

**Derived comparator ("tools-present gate").** A simple deployment rule would apply the
A2 LoRA exactly when the request carries `tools`. On BFCL this equals A2, because every
item has tools. On IFEval, GSM8K and the probe (no tools) it equals A0. It is reported
from the existing runs at no extra compute. If A3 does not beat A2 on BFCL, SUBCOR's
embedding router adds nothing over this one-line heuristic **for this skill**, and the
report must say so.

### 4.2 Qwen3.5-0.8B (secondary)

Arms A0, A2 and A3, with the same definitions and pipeline. Layers 18-23; everything else
as in section 4.4. This model has no PASS criteria; its results are reported with the same
statistics.

### 4.3 Decoding (all arms, both models)

- temperature **0**: SUBCOR forces 0 anyway, and the harness sends `--temperature 0`.
- **thinking off**: `enable_thinking=false`, harness `--thinking off`.
- `max_tokens` 4096 for BFCL, 1280 for IFEval, 1024 for GSM8K. These are the runners'
  defaults, and `run_general.sh` now passes them explicitly; the original text said 1280
  for GSM8K. No `parallel_tool_calls` field.
- One request at a time: llama-server `-np 1` and harness `--num-threads 1`.
  Concurrency 1 in `run_general.sh`.
- Identical llama-server flags for every arm, apart from A2's `--lora`:
  `--jinja --embeddings --pooling mean -ngl 99 -c 65536 -np 1`, the same GGUF file, the
  same binary.

Thinking-off is a deliberate choice: SUBCOR's default, and the cheaper edge deployment.
The Qwen3.5 model card's BFCL-v4 numbers were measured **with thinking**, so our A0 is
expected to sit below the card and is not comparable to it. Thinking-off may also leave
headroom that thinking would close; see R10.

### 4.4 Training (A2/A3/A4, identical for every seed)

| setting | value |
|---|---|
| train / dev | `train.jsonl` / `dev.jsonl` (sha256 above), with `--exclude-ids data/results/decontam_semantic_drop.json`, which removes 27 train and 3 dev conversations (R1). Early stopping on dev **only**. |
| layers | 4B: `26-31`; 0.8B: `18-23` (last 6 blocks) |
| projections | `mlp.down_proj` |
| rank / sub1bit-rank | 8 / 8 |
| lr / steps / grad-accum | 3e-4 / max 400 / 8 |
| eval-every / patience | 25 / 4 (best dev-loss checkpoint kept; `binary_ste=False`) |
| max-seq-tokens | 4096 (nothing in the data exceeds it) |
| dtype / device | fp16 / ROCm GPU, trainer resident with no llama-server running a benchmark |
| A2 export | `export_unit_to_gguf_lora(..., tensor_dtype="f32")` (`--a2-dtype f32`), scale 1.0 |
| gate | `--pos-prompts router_pos.jsonl --neg-prompts router_neg.jsonl --prompt-split fit`. `train_unit.py` now keeps only the `split=="fit"` rows; before the review it read every row, heldout included. Positives whose `row_id` is in the exclude list are dropped, which leaves 233 positives and 240 negatives. `--gate-safety-bias 0.0`, `--num-prototypes 1`. `--embed-url` = a llama-server for the same GGUF with **no adapter at nonzero global scale**. `train_unit.py` checks `GET /lora-adapters` and refuses otherwise, so the A2 server can never be used. The `heldout` rows and `router_pos_dev.jsonl` are used only for the gate diagnostics we report. |
| seeds | **seed 0 is primary.** Seeds 1 and 2 are run for A2, A3 and A4 on 4B if time permits (A2/A3/A4 share each run). Otherwise only seed 0, and the report says so. |

**Single permitted fallback.** If the best dev loss is not below the base dev loss, or
training diverges, retrain once with lr 1e-4. This decision uses **dev loss only** and is
logged in section 12. No other hyperparameter search is allowed. In particular, nothing
may be tuned on BFCL, IFEval, GSM8K or the probe.

**Governance.** Units come out DORMANT. SUBCOR's built-in conformance runs only inside
`run_study_cycle`, which trains its own adapter from tenant telemetry, so it cannot
certify a unit from `train_unit.py`. The conformance inputs are therefore fixed here, so
they cannot be chosen later or computed on test data:
- `SkillUnitRegistry.record_conformance` with its default thresholds
  (`max_regression_tolerance=0.02`, `min_target_improvement=0.05`).
- `target_before = exp(−dev_loss.base)` and `target_after = exp(−dev_loss.<variant>)`
  from `train_report.json`, where the variant is `sub1bit` for A3 and `fp32` for A4.
- `baseline_before` and `baseline_after` = substring accuracy on SUBCOR's
  `DEFAULT_BASELINE_ANCHOR_CASES` through the tenant, without and with the unit's gate
  registered, as `run_study_cycle` does.
- Then `promote` by an approver other than the proposer.

No BFCL, IFEval, GSM8K or probe item may enter conformance. If governance rejects a
unit, A3 or A4 is reported as "rejected by SUBCOR governance": as deployed, the
headline fails. A diagnostic run with the unit force-enabled may follow, but it must be
labelled non-primary.

## 5. Evaluation sets and n

BFCL v4, full sets (item counts from `bfcl_eval/data`):

| section | categories (n) | in overall? |
|---|---|---|
| Non-live AST | simple_python 400, simple_java 100, simple_javascript 50, multiple 200, parallel 200, parallel_multiple 200 (1,150) | yes |
| Live AST | live_simple 258, live_multiple 1,053, live_parallel 16, live_parallel_multiple 24 (1,351) | yes |
| Irrelevance | irrelevance 240, live_irrelevance 884 | yes |
| Relevance | live_relevance 16 | no (bfcl reports it only) |
| Multi-turn | base 200, miss_func 200, miss_param 200, long_context 200 (800) | yes |
| **Total per arm** | **4,441 items** | |

**Excluded:**
- web_search_base and web_search_no_snippet (200): need SerpAPI.
- format_sensitivity: prompt-mode models only.
- memory_kv, memory_vector, memory_rec_sum (3 × 155): excluded from all primary
  endpoints. They need 64k context per slot, and the smoke test took about 50 minutes
  for one question on CPU. They are optional exploratory runs for A0 and A3 on 4B only,
  if time remains after everything else, and are reported separately with the memory
  overall of section 6.1.

**Compute rule.** This is the only allowed reduction. It is decided **from the wall-clock
time of calibration run C-off on Qwen3-4B-Instruct-2507 only**, and recorded in
section 12 before any Qwen3.5 run. If the projected full 4B arm takes more than 14 hours,
every multi-turn category in every arm and in C-nat uses the frozen seeded subset
`--per-category-multi-turn 100 --sample-seed 20261122` (400 items). Single-turn always
runs in full. Before the review, `--per-category-multi-turn` without `--per-category` was
silently ignored and would have run the full 800. It now subsets only multi-turn, as
tested: 100 per multi-turn category and every single-turn category complete. The subset
ids are frozen in each `run_config.json`, and every arm passes
`--ids-file <A0 run>/run_config.json`. The harness takes that run's `selected_ids` and
refuses a category mismatch. `pair` refuses runs whose ids differ.

**General regression** (4B: A0, A2, A3, A4; 0.8B: A0, A2, A3):

| set | n |
|---|---|
| IFEval | all 541 prompts. Primary metric: prompt-level strict accuracy; the other three are reported. |
| GSM8K test subset | 500 (`gsm8k_subset_seed1234.json`). Strict (`####`) and flexible, **both** primary for H4. |
| Router probe | 300 prompts, rebuilt disjoint from the router calibration data (R3): 240 Dolly, 30 GSM8K train, 30 hand-written hard negatives. |

## 6. Endpoints and statistics

### 6.1 Renormalised overall (not the official Overall)

With per-category accuracies in percent, using bfcl's own section definitions:

```
S    = (simple_python + simple_java + simple_javascript) / 3
NL   = (S + multiple + parallel + parallel_multiple) / 4                      # bfcl unweighted
L    = (258·live_simple + 1053·live_multiple + 16·live_parallel + 24·live_parallel_multiple) / 1351
                                                                               # bfcl n-weighted
IRR  = (irrelevance + live_irrelevance) / 2
MT   = (mt_base + mt_miss_func + mt_miss_param + mt_long_context) / 4
O_R  = (10·NL + 10·L + 10·IRR + 30·MT) / 60
```

This is bfcl's `[10, 10, 10, 30, 40]` weighting (non-live, live, irrelevance, multi-turn,
agentic) with the agentic term removed and the rest renormalised. It is **not** the
leaderboard Overall and is never placed next to it. If memory is run (exploratory):
`O_RM = (10·NL + 10·L + 10·IRR + 30·MT + 20·MEM) / 80`, where `MEM` is the mean of the
three memory categories (bfcl's agentic term without web search). With the multi-turn
subset, `MT` is the mean over the subset's 4 categories.

### 6.2 Tests

- **Paired bootstrap.** 10,000 resamples (`run_bfcl.py pair --bootstrap 10000`), seed 0,
  stratified by category: resample items within each category, recompute every section
  and O_R through the formulas above, and take the paired difference B − A. Report the
  point estimate, the percentile 95% CI, and win/tie/loss counts. IFEval and GSM8K use
  `general/paired_bootstrap.py` (10,000 resamples).
- **McNemar.** Exact two-sided binomial on the discordant pairs, per BFCL category and
  per general metric. Reported with Holm correction across the 17 BFCL categories as
  exploratory evidence. McNemar is not part of PASS. `pair` writes `per_category`, with
  win/tie/loss counts, `mcnemar_p` and `mcnemar_p_holm`.
- **Pairing guards.** `pair` refuses two runs unless they match on everything except
  the arm-identifying fields (`base_url`, `served_model`, `system_prefix*`). That covers
  handler, temperature, thinking, `max_tokens`, backend, timeout, `extra_body`, frozen
  item ids, categories and the harness code hash. It also refuses runs that graded
  different item sets. The only permitted override is `--allow-config-diff
  endpoint.backend` for the V5 direct fallback. `general/paired_bootstrap.py` likewise
  refuses arms with failed or missing items. **Error sensitivity** is
  `pair --exclude-errors --exclude-errors-from <every other arm>`.
- **Multiplicity.** H1 is an OR over two endpoints. PASS uses the nominal 95% CI as
  specified, and the report also gives 97.5% (Bonferroni) CIs. If H1 passes at 95% but
  not at 97.5%, the report must say "passes nominally, not after multiplicity
  adjustment".
- **Router false activation.** Activations / 300 on the probe, with a Wilson 95% CI.
  Also reported excluding the hard negatives, and per source.
- **Seeds.** PASS is evaluated on seed 0. Extra seeds are reported as the mean and range
  of each delta. If any seed's A3 − A0 point estimate on the H1 endpoint is below 0, the
  claim must be qualified as "not robust across training seeds".

### 6.3 PASS criteria (4B, seed 0). All must hold for "SUBCOR works".

| # | criterion |
|---|---|
| P1 | A3 − A0 ≥ **+5.0** points on **MT** with 95% CI lower bound > 0, **or** the same on **O_R** |
| P2 | A3 − A1 ≥ 0 on O_R (point estimate; CI reported) |
| P3 | A3 − A2 ≥ −2.0 on O_R (point estimate; CI reported) |
| P4 | IFEval prompt-strict, GSM8K strict and GSM8K flexible: each A3 − A0 ≥ −1.0 with 95% CI lower bound ≥ −2.0 |
| P5 | A3 unit false activation < 5% on the 300-prompt probe **and** < 5% on the 541 IFEval prompts (point estimates; Wilson CIs reported) |

**P4 can hold by abstention.** If the unit fires on no IFEval or GSM8K item, A3 is
byte-identical to A0 there, and P4 holds with a delta of exactly 0. The report must then
say "P4 met by abstention (unit fired on k/n)". It must also show A2 − A0 and A4 − A0 on
the same sets, which measure what the unit would cost if it fired. The same applies to
GSM8K in P5: the gate's negatives include GSM8K *train*, so GSM8K-test activation is an
in-distribution check, reported and not gated. The probe's Dolly and GSM8K-train
prompts share sources with the gate negatives too, which is why P5 also gates IFEval.

**Outcome labels:**

| label | meaning |
|---|---|
| **"SUBCOR works on this test"** | P1-P5 all hold and validity checks V1-V8 pass |
| **"Partial"** | P1 holds but one of P2-P5 fails; the report names which |
| **"Does not work on this test"** | P1 fails |

All arms are reported whatever the outcome. Nothing is dropped.

### 6.4 Also reported (no thresholds)

- Every bfcl column for every arm. Deltas A2 − A0, A4 − A0, A3 − A4 and A4 − A2, with
  CIs.
- Router activation rate per BFCL category (recall proxy), on IFEval and on GSM8K test.
  BFCL rates come from `run_bfcl.py routing`, which writes `routing.csv` and
  `routing.json` per run.
- Deltas split by whether the router fired on the item: on every step, some steps, or
  none. Use `routing.csv`'s `fired_class` joined to the `pair` CSV.
- Error-type breakdown per arm from bfcl's score files: `decoder_failed`, wrong function,
  `value_error`, missing parameter, irrelevance failures, inference errors.
- **Prompt-token cost per arm**: mean, median and p95 of `usage.prompt_tokens` and
  `usage.completion_tokens` per request and per item, from server usage fields. A1's
  added tokens are reported explicitly (system prompt is 3,384 characters).
- **Latency per arm**: per request and per item, mean / p50 / p95 wall clock at
  concurrency 1. SUBCOR routing overhead (the embedding call) is measured separately on
  50 requests.
- Unit payload sizes on disk: SUB1BIT, FP32, the A2 f32 GGUF, and an f16 export of the
  same adapter. The f16 file is written for the size table only and never served.
- Gate diagnostics: activation on the heldout positives and negatives and on
  `router_pos_dev.jsonl`.

## 7. Run order

1. Fix V2 (timeout), commit, and record the commit hash in section 12.
2. Download the calibration GGUFs and record their sha256.
3. Calibration: C-off for 4B-2507, then the compute-rule decision, then C-off for 0.6B
   and C-nat for both. Record the verdict in section 12.
4. Run the Qwen3.5 A0 failure audit (R7) on A0 BFCL results. The probe set was already
   rebuilt (R3). A0 is not a treatment arm, so looking at it does not unblind anything.
   But A0 results are BFCL test results. After seeing them, only harness or serving
   fixes are allowed: they apply to every arm, and A0 is rerun. Training data, the
   exclude list, hyperparameters, exemplars, gate data and gate settings may not change
   (R4). V8 enforces this through the section-2 hashes.
5. Train seed 0 (4B, then 0.8B), then promotion.
6. Run arms A1, A2, A3 and A4 (4B), then A2 and A3 (0.8B), then the general regression
   runs, then the probe.
7. Run the pre-registered analysis, then optional seeds 1 and 2, then optional memory.

## 8. Validity preconditions (each one is checked and reported)

- **V1 Harness frozen.** `experiments/bfcl/harness/` and the analysis code are fixed at
  a recorded commit before the first treatment arm. A harness fix found later must be
  applied to all arms, and every affected arm rerun. Each `run_config.json` now records
  `harness.files_sha256` and the `bfcl_eval` version. Resuming a run under changed
  harness code is refused, and `pair` refuses arms produced by different harness code.
- **V2 SUBCOR timeout.** `LlamaCppServingAdapter` currently has a hard-coded 60-second
  urllib timeout, and the resulting `TimeoutError` drops the client connection. It must
  become configurable and be set to ≥ 600 s, identically for all arms, before any arm
  runs. This is a change to `src/`, owned by the orchestrator. After one
  `--retry-errors` pass, each arm's inference-error rate must be ≤ 0.5% of items.
  Remaining errors count as wrong. A sensitivity analysis also excludes every item that
  errored in any arm.
- **V3 State isolation.** Each arm runs on a fresh copy of a frozen SUBCOR state
  directory, because SUBCOR logs eval traffic into tenant telemetry as unverified
  examples. No Study Cycle, intern or live-learning job may run during evals. The
  sha256 of every unit payload, gate file and registry manifest is recorded before and
  after each arm and must not change.
- **V4 Determinism.** Re-run 50 seeded BFCL single-turn items and 10 multi-turn items
  for A0 and A3 on a freshly started server, in reverse order. Outputs must match the
  main run on ≥ 98% of items. The reverse order also exercises A3's alternating LoRA
  on/off against llama-server's prompt cache. If the check fails, disable prompt-cache
  reuse on the server and rerun all arms.
- **V5 A2 really is always on.**
  - `GET /lora-adapters` shows scale 1.0 before and after the A2 run.
  - At least 20 of the first 50 A2 outputs differ from A0.
  - If SUBCOR's global LoRA handling resets the scale, A2 instead runs direct on
    llama-server with `--lora` at temperature 0, together with an **A0-direct** run.
    P3 is then evaluated as (A3 − A0) − (A2 − A0-direct).
- **V6 A3 and A4 route identically.** Per-request `selected_units` agree on ≥ 99% of
  requests. If they don't, A3 − A4 is reported as confounded by routing.
- **V7 Context.** No item fails because of the context limit (65,536 per slot). Every
  `Error during inference` is classified.
- **V8 Data hashes.** Train, dev, router, exclude-list, probe and A1 files match
  section 2 at training and run time. `train_report.json` records `input_sha256`.
- **V9 The unit is the only difference (A3/A4 vs A0).** Run
  `run_bfcl.py routing --run-dir <A3> --compare-run <A0>`. On items where the gate fired
  on no step, A3's `result` must equal A0's on ≥ 98% of items, because the server and
  request are the same and no adapter is applied. Do the same for A4. A miss means
  something besides the unit differs between the tenant and base paths, for example the
  routing embedding request evicting the slot's prompt cache. In that case apply V4's
  remedy (no prompt-cache reuse, all arms rerun) before any comparison is reported.
- **V10 A2 is the A4 adapter, always on.** The tensors in the A2 GGUF must equal the
  tensors the supervisor exported for the A4 unit, compared with `gguf` tensor arrays.
  On items where A4 fired on every step, report the identical-output rate of A4 against
  A2 (`routing --run-dir <A4> --compare-run <A2>`). If it is below 90%, P3 and A4 − A2
  are flagged "routing confounded with serving differences".
- **V11 Exactly one routed unit, routed inside SUBCOR.** `routing --expect-unit` reports
  0 problems for A3 and A4.

## 9. Adversarial review: leakage and circularity risks, with mitigations

**R1. Training data overlaps the test.** The training pool (Hermes, Glaive, ToolACE)
comes from the same public function-calling ecosystem as BFCL. ToolACE was itself
evaluated on BFCL, and BFCL "live" items are user-contributed APIs.

*Done:* decontamination against every BFCL v4 file, the memory prerequisites, the
multi-turn function docs and the unused datasets (8,934 prompts and 9,476 functions):
- exact name + schema match (1,592 removed);
- name + parameter-name match (7,405);
- 13-gram overlap with the shorter prompt ≥ 0.8 (0);
- MinHash Jaccard > 0.7 (52).

APIGen-MT was removed completely.

*Second pass (review, `data/decontam_semantic.py`):*
- The build's 13-gram rule hashes a BFCL prompt shorter than 13 words as **one** gram.
  So a short BFCL prompt embedded in a longer training message was never detected.
  Example: `simple_python`'s "Calculate the area of a circle with a radius of 5 units."
  inside glaive-5459 and glaive-6352.
- A verbatim-containment rule for 6-12-word BFCL prompts found 16 train rows. Ten of
  them contain the 6-word turn "I want to watch a movie" or "I'm looking for a movie".
- BFCL functions copied under another name, or with changed parameters, were found by
  identical or near-identical descriptions (> 6 words, Jaccard > 0.7): 11 train and 3 dev
  rows. Examples: `calculate_tip`, `getCurrentWeather` ~ `get_current_weather`, and
  `analyze_sentiment` ~ `text_analysis.sentiment_analysis`.
- All 30 conversations are excluded from training, dev and router positives through
  `--exclude-ids`. No A1 exemplar is affected.
- A looser 8-gram ≥ 0.5 rule flags 7 more train rows, all template-level near-misses
  such as "radius of 4" against "radius of 5". These are reported, not dropped, and fall
  under analysis (a).

*Residual risk:* paraphrased prompts, and APIs that are semantically the same under
different names and different descriptions. Also, 366 training tools share only a
*name* with a BFCL function.

*Mitigation (pre-registered analyses):*
- (a) Embed every BFCL item and every train prompt with the base `/v1/embeddings`. Split
  the BFCL items into tertiles by nearest-neighbour cosine similarity, and report
  A3 − A0 and A2 − A0 per tertile. If the top tertile's gain is more than twice the
  bottom tertile's and the bottom tertile's CI includes 0, flag "gain concentrated on
  train-similar items".
- (b) Report the delta on BFCL items whose function names occur among the 366 shared
  names, against all other items.
- (c) Leakage in the base model's own pre-training affects every arm equally, so it
  cancels in paired deltas. It does not cancel in calibration.

**R2. Exemplar overlap (A1).** The exemplars come from the decontaminated train.jsonl.
Their tool names were checked against all 2,910 BFCL function names, and they were chosen
by a fixed seed before any result existed. The selection script and output are in the
repo. A1 uses the same data pool as A2/A3/A4, so no arm gets privileged data. *Residual
risk:* one exemplar (`toolace-4978`) has parameterless tools and is fairly weak. We keep
it: re-rolling for quality would be selection by preference.

**R3. Router calibrated on eval prompts.**
- The gates are fit only on train-derived positives and on Dolly and GSM8K-train
  negatives. The data agent checked the negatives against IFEval, the GSM8K test split
  and BFCL, with 0 hits.
- **Found while writing this document:** the current router probe set
  (`general/probe_set_ids.json`) shares **4 of 300 prompts** with `router_neg.jsonl`
  (exact match after normalisation). It was built before `router_neg.jsonl` existed,
  excluding only `neg_prompts.txt`.
- *Done (review):* the probe was rebuilt with
  `build_probe_set.py --exclude neg_prompts.txt --exclude router_neg.jsonl --exclude router_pos.jsonl --exclude router_pos_dev.jsonl`.
  The builder now excludes by **source row**: a Dolly row is dropped when any of its
  renderings matches, or its instruction equals the first paragraph of an excluded
  prompt. (`build_router_calib.py` renders summarization rows as "instr\n\nctx" and the
  probe as "instr\n\nContext:\nctx".) GSM8K rows are dropped by question text.
- Four Dolly rows (917, 3545, 12962, 3568) were replaced from the same strata. The
  builder asserts zero overlap after the build. An independent check found 0 matches,
  by exact normalized text and by instruction, against all three router files.
  `probe_set_ids.json` records the excluded rows, the hashes of the exclude files and
  the probe hash.
- *Remaining limit:* the probe draws on the same sources as the gate negatives (Dolly,
  GSM8K train), so a low probe rate is partly in-distribution. P5 therefore also gates
  IFEval.
- The gate threshold uses `safety_bias = 0.0`, fixed now, and is never adjusted after
  seeing probe, IFEval or BFCL activations.

**R4. Tuning on the test.**
- Every hyperparameter is fixed in section 4.4. Early stopping and the single lr
  fallback use dev.jsonl only, and dev shares no tool set with train.
- No checkpoint, seed, threshold, layer set or prompt may be chosen using BFCL, IFEval,
  GSM8K or probe results. Seed 0 is primary, chosen in advance, and every seed run is
  reported.
- *Residual risk:* the harness was debugged on 73 BFCL smoke items with Qwen3.5-0.8B,
  and the training-data category mix was designed with BFCL's category taxonomy in mind
  (not its items). This is disclosed. The smoke items stay in the test set; they are
  harness checks, not tuning signal for the units.
- Calibration (C-off, C-nat) and the A0 audit expose BFCL test results before training.
  Nothing on the training side may change after that (section 7, step 4). Data, the
  exclude list and all settings are frozen by hash now. The review's exclude list was
  made from BFCL **inputs** only (prompts and function docs), with no model output.

**R5. Grader differences.**
- Grading is bfcl's own `eval_runner`, unmodified. `per_item.csv` is derived from bfcl's
  score files with count assertions, and the harness aggregation is checked against
  bfcl's `data_overall.csv` (it matched on 23 of 23 columns in the smoke runs).
- The oracle self-test passes 3,593 of 3,606 items; the 13 failures are quirks in the
  answer data.
- The native-FC path converts tools (`.` becomes `_`, mapped back), the same for every
  arm. The cost of that against the board's QwenFCHandler is measured in calibration
  (C-nat − C-off).
- IFEval uses lm-eval's copy of the official checker, and it agreed with the lm-eval CLI
  on the 10 smoke items. GSM8K uses our own zero-shot prompt and is comparable only
  between arms; it is never compared with published numbers.

**R6. Serving-path asymmetry between arms.**
- Every arm goes through SUBCOR serve with identical llama-server flags, temperature 0
  (SUBCOR forces it) and concurrency 1.
- A2's always-on LoRA is a server flag rather than a per-request field; V5 covers it.
- A timeout or inference error that falls more on one arm could bias the comparison. For
  example, an adapter that lengthens outputs would get more timeouts. V2 sets the error
  budget and the sensitivity analysis.

**R7. Gains that are really format or parser repair.** Qwen3.5 on llama.cpp uses an XML
call format, and the smoke run already showed a boolean returned as the string `"True"`.
An adapter can raise scores by learning to avoid parser artifacts or to call a tool
instead of answering in prose (the dominant 0.8B failure). That is a real deployment
gain, but it is not "tool-use skill".

*Mitigation:*
- Before any treatment arm, take a seeded sample of 60 A0 failures (seed 20261122,
  stratified by section). Classify each by hand as a parser or serving artifact or a
  model error, and record the counts in section 12.
- Report each arm's error-type breakdown, and the share of A3 − A0 gains that come from
  `decoder_failed` (prose instead of a call) versus wrong or incorrect calls.

**R8. Telemetry feedback (live learning on test data).** SUBCOR writes every request to
tenant telemetry and can run Study Cycles from it. Training on BFCL traffic during the
eval would leak the test into the units. V3 covers this: fresh state copies, no study
jobs, and hash checks before and after each arm.

**R9. Prompt cache and LoRA switching.** If llama-server reused a KV prefix computed
under a different adapter, A3 outputs would depend on the order items run in. V4 covers
this with the reversed-order rerun.

**R10. A thinking-off headroom artifact.** With thinking off, the base may answer in
prose where thinking would have called the tool. The adapter could be buying back
something the base already has at extra decode cost.

*Mitigation (optional; run only if time remains):* an exploratory A0 run with thinking
on, over single-turn only. It is reported as context and does not change PASS. Its token
cost goes next to A3's.

**R11. Weak baseline choice.** A0 alone would flatter any trained arm. A1 (few-shot), A2
(plain LoRA) and the tools-present gate (section 4.1) are the realistic alternatives,
and P2 and P3 require SUBCOR not to lose to them.

**R12. Small categories.** live_parallel (16), live_parallel_multiple (24) and
live_relevance (16) have very wide CIs. They contribute only through L, which is
n-weighted, or not at all, and no conclusion is drawn from them alone.

**R13. Same team builds and judges.** This document is committed before results exist.
The analysis commands are fixed here. Deviations are logged with what had been seen.
Every arm and seed run is reported, including failures and governance rejections.

## 10. Exact commands (templates)

```bash
PY=<scratchpad>/bfcl/harness/venv/bin/python; H=experiments/bfcl/harness/run_bfcl.py
# calibration, official path
$PY $H all --run-dir $S/runs/cal-q3-4b-2507-official --handler official \
  --official-registry Qwen/Qwen3-4B-Instruct-2507-FC --base-url http://127.0.0.1:85xx/v1 \
  --categories non_live live multi_turn --num-threads 1 --reference-model "Qwen3-4B-Instruct-2507 (FC)"
# training (seed 0, 4B)
uv run --extra dev --extra rocm python experiments/bfcl/train_unit.py --base-gguf <4B GGUF> \
  --train $S/data/train.jsonl --dev $S/data/dev.jsonl \
  --exclude-ids experiments/bfcl/data/results/decontam_semantic_drop.json \
  --unit-id bfcl-tools-v1 --registry <staging>/units --out $S/train/q35-4b-s0 \
  --layers 26-31 --projections mlp.down_proj --rank 8 --steps 400 --lr 3e-4 --grad-accum 8 \
  --eval-every 25 --patience 4 --seed 0 --a2-dtype f32 \
  --pos-prompts $S/data/router_pos.jsonl --neg-prompts $S/data/router_neg.jsonl --prompt-split fit \
  --embed-url http://127.0.0.1:85xx      # a llama-server with no nonzero-scale adapter
# A0 (S1, subcor/base); with the compute rule add: --per-category-multi-turn 100 --sample-seed 20261122
$PY $H all --run-dir $S/runs/q35-4b-A0 --backend subcor --base-url http://127.0.0.1:85yy/v1 \
  --served-model subcor/base --thinking off --temperature 0 --categories non_live live multi_turn --num-threads 1
# A1 (S1): the only difference from A0 is the system prefix
$PY $H all --run-dir $S/runs/q35-4b-A1 --backend subcor --base-url http://127.0.0.1:85yy/v1 \
  --served-model subcor/base --system-prefix experiments/bfcl/a1_exemplars.json --thinking off --temperature 0 \
  --categories non_live live multi_turn --num-threads 1 --ids-file $S/runs/q35-4b-A0/run_config.json
# A3 (S1); A4 identical with subcor/<tenant-a4>; A2 identical with subcor/base on S2
$PY $H all --run-dir $S/runs/q35-4b-A3-s0 --backend subcor --base-url http://127.0.0.1:85yy/v1 \
  --served-model subcor/<tenant-a3> --thinking off --temperature 0 \
  --categories non_live live multi_turn --num-threads 1 --ids-file $S/runs/q35-4b-A0/run_config.json
$PY $H routing --run-dir $S/runs/q35-4b-A3-s0 --expect-unit bfcl-tools-v1-sub1bit --compare-run $S/runs/q35-4b-A0
$PY $H pair --run-a $S/runs/q35-4b-A0 --run-b $S/runs/q35-4b-A3-s0 --bootstrap 10000 --seed 0 \
  --out $S/pairs/A3-vs-A0.csv
# error sensitivity (V2)
$PY $H pair --run-a $S/runs/q35-4b-A0 --run-b $S/runs/q35-4b-A3-s0 --exclude-errors \
  --exclude-errors-from $S/runs/q35-4b-A1 $S/runs/q35-4b-A2-s0 $S/runs/q35-4b-A4-s0 --out $S/pairs/A3-vs-A0.noerr.csv
```

When a run directory contains no memory category, the harness value
"Overall (no web search, renormalized; NOT official)" is exactly O_R of section 6.1
(weights 10/10/10/30 over the sections bfcl defines; checked against
`run_bfcl.py` `official_aggregate`). `pair` bootstraps that value. Exploratory memory runs
therefore go in **separate run directories**, so that primary run directories never mix
O_RM into O_R. If any implementation disagrees with section 6.1, section 6.1 is
authoritative.

## 11. What gets committed

Small ledgers only: `summary.json`, `per_item.csv`, `run_config.json`, pair JSON and CSV,
and general-metric summaries, each under 1 MB, in `experiments/bfcl/results/`. Model
files, datasets and raw generations stay in the scratchpad.

## 12. Deviations log

Each entry records the date, the change, the reason, and whether any treatment-arm
result had been seen.

- 2026-11-22: pre-registration written. No GPU result exists. Known open items, all
  required before the arms run: V2 timeout fix in `src/`; harness `--system-prefix`
  option for A1; rebuild of the probe set (R3, 4/300 overlap with `router_neg.jsonl`);
  `pair --bootstrap 10000` (the default is 2000).
- 2026-11-22 (adversarial review, before the freeze commit; no GPU result and no
  treatment-arm result exists): sections 1, 2, 4, 5, 6, 7, 8, 9 and 10 were edited in
  place as listed in section 13. Resolved open items: `--system-prefix` (implemented),
  the probe rebuild (done, 0 overlap), and `pair --bootstrap` (default now 10000).
  Still open: the V2 timeout fix in `src/` (`LlamaCppServingAdapter(timeout=60.0)`; the
  CLI exposes no timeout).

## 13. Review log (adversarial review, 2026-11-22)

A skeptical pass over this document and every script under `experiments/bfcl/`. The
reviewer looked for any way the result could be invalid. Each item gives the issue, its
severity and the resolution. No GPU was used. All checks ran on CPU against the frozen
scratch data.

**Leakage into training, exemplars, router calibration or hyperparameters**

1. **Short BFCL prompts slipped through decontamination (real, fixed).** The 13-gram
   containment rule represents a BFCL prompt of fewer than 13 words as one whole-sequence
   gram. That gram can only match a training message with exactly the same words. Most
   `simple_*` prompts are that short. "Calculate the area of a circle with a radius of 5
   units." sat verbatim inside two glaive training requests.
   *Resolution:* `data/decontam_semantic.py` adds verbatim containment for 6-12-word
   BFCL prompts: 16 train rows.
2. **Renamed or reparameterised copies of BFCL functions (real, fixed).** The name-keyed
   schema and signature rules miss them. A description-identity / description-MinHash
   pass found 11 train and 3 dev rows (`calculate_tip` with other parameters,
   `getCurrentWeather`, `analyze_sentiment`, `Currency_Converter`, ...).
   *Resolution:* items 1 and 2 give 30 ids, frozen in
   `data/results/decontam_semantic_drop.json` (sha256 in section 2). They are excluded
   via `train_unit.py --exclude-ids`, which filters train, dev and router-positive rows
   by `row_id`: 7 of the 240 fit positives. The source files and their hashes are
   unchanged, so no A1 exemplar or other frozen artifact moves.
3. **Router heldout rows were fitted on (real, fixed).** The plan says fit on
   `split=="fit"` rows only, but `train_unit.read_prompts` read every row of the JSONL.
   The "heldout" gate diagnostics would have been in-sample.
   *Resolution:* `--prompt-split` (default `fit`) filters by the `split` key.
4. **Gate embeddings could come from a non-frozen server (real risk, guarded).**
   llama-server's embedding endpoints use the *global* adapter scales. Pointing
   `--embed-url` at the A2 server (`--lora` at scale 1) would fit the gate on
   adapter-modified embeddings that serving never sees.
   *Resolution:* `train_unit.py` checks `GET /lora-adapters` and refuses any nonzero
   global scale. The adapter list is recorded in the gate stats.
5. **Probe overlapped the gate negatives (known, fixed).** The overlap was 4/300.
   *Resolution:* the probe is rebuilt with source-row exclusion and a built-in
   zero-overlap assertion. An independent recheck found 0 matches by exact text and 0 by
   instruction, against router_neg, router_pos and router_pos_dev. New probe sha256 is in
   section 2.
6. **Governance was a path for test data or post-hoc metric choice (gap, specified).**
   SUBCOR's conformance only exists inside `run_study_cycle`, which cannot certify a
   `train_unit.py` unit. "Promote through the normal flow" was therefore undefined, and
   someone could have computed conformance on BFCL.
   *Resolution:* the inputs (dev loss and SUBCOR anchors) and thresholds are now fixed
   in section 4.4, and BFCL, IFEval, GSM8K and the probe are excluded from them.
7. **Training-side changes after seeing BFCL results (loophole, closed).** Calibration
   and the A0 audit show BFCL test outcomes before training. Nothing forbade adjusting
   the training mix in response.
   *Resolution:* section 7, step 4 and R4 forbid it, and V8 hashes enforce it. The
   review's exclude list uses BFCL inputs only, never model outputs.
8. **A1 exemplars:** checked, not hit by the new rules, no change. **Hyperparameters:**
   all fixed in 4.4. No BFCL-derived signal was found in their provenance: the template
   parity results are synthetic, and the smoke runs were base-only. No change.

**Thresholds that could be met trivially**

9. **P5 was nearly in-sample (real, strengthened).** The probe draws on Dolly and
   GSM8K-train, the same sources as the gate's fitted negatives, so a low rate there is
   expected by construction.
   *Resolution:* P5 and H5 now also require < 5% activation on the 541 IFEval prompts,
   a source the gate never saw. GSM8K-test activation is labelled an in-distribution
   check.
10. **P4 can hold by abstention (disclosed).** If the gate never fires on IFEval or
    GSM8K, A3 equals A0 there and P4 holds with delta 0.
    *Resolution:* the report must label this "met by abstention" and show A2 − A0 and
    A4 − A0 on the same sets.
11. **P3 could pass because A2 is broken (guarded).** A wrong export scale would make
    A2 weak.
    *Resolution:* A2 is exported with f32 tensors identical to A4's supervisor export,
    and V10 checks tensor identity and output identity on items where A4 fired on every
    step. P2 against a weak A1: if A1 < A0, P1 already implies P2. No change.

**Comparability of A0-A4 (same prompts, decoding, template, server path)**

12. **Multi-turn compute rule was a silent no-op (real bug, fixed).**
    `--per-category-multi-turn` only worked together with `--per-category`. The
    pre-registered command would have run the full 800 multi-turn items in one arm,
    while an arm launched differently might have been subset.
    *Resolution:* fixed and tested. Every arm now reuses A0's frozen ids through
    `--ids-file <A0>/run_config.json`.
13. **`pair` did not check comparability (real, fixed).** It silently paired the
    intersection of items. It accepted runs with different thinking, `max_tokens`,
    temperature, backend, `extra_body`, item ids or harness code. It also had no
    error-sensitivity mode, no 97.5% CI and no per-category McNemar.
    *Resolution:* all of these are added. It refuses mismatches unless a specific key is
    allowed, and the default is 10,000 resamples. `general/paired_bootstrap.py` now
    refuses arms with failed or missing items instead of dropping them.
14. **Harness drift between arms (guarded).** `run_config.json` records the harness
    file hashes and bfcl version. Resume refuses changed code, and `pair` refuses arms
    from different code (V1).
15. **Server-path asymmetry (specified, checked).** The document claimed "identical
    flags", but supervised `subcor serve` builds its own llama-server command line
    without `-np`. A3/A4 and A0 could have run on differently configured processes.
    *Resolution:* the S1 and S2 instance recipes (section 4.1) put A0, A1, A3 and A4 in
    one process. V9 then requires A3 to reproduce A0 byte-for-byte on items where the
    gate never fired. That also catches one plausible hidden difference: the routing
    embedding request sharing the single slot with chat and evicting its prompt cache.
16. **A4 − A2 mixed routing with FP32-vs-FP16 tensors (real, fixed).**
    *Resolution:* the A2 default is now f32 (`--a2-dtype`).
17. **GSM8K `max_tokens` mismatch (doc vs code).** The document said 1280, the runner
    used 1024.
    *Resolution:* the document now says 1024, and `run_general.sh` passes both limits
    explicitly.
18. **A1 had no implementation (open item, done).** `--system-prefix` merges the
    exemplar text into the first system message, never mutates bfcl's test entry, and
    is sha256-checked against `a1_exemplars.json`. A unit check covered the merge and
    insert cases, and multi-turn persistence follows from first-turn placement.
    `pair` treats the prefix as an arm-identifying field.

**Could the FC handler bias some arms?**

19. **Shared code path (checked, no change).** Every arm runs through the same
    `NativeFCHandler` and the same SUBCOR server code. The request fields are identical
    except `model` and, for A1, the system text. SUBCOR forces temperature 0, ignores
    `top_k` sampling, drops `reasoning_content` and applies the same `parse_tool_calls`
    text fallback to every arm. The oracle self-test still passes; its one `--limit 20`
    failure is the documented `parallel_multiple_12` answer quirk.
20. **Residual handler-level effects (disclosed).** An adapter can gain by producing
    output the Qwen3.5 XML parser types correctly ("True" against `true`) or by calling
    instead of answering in prose. R7's audit and error breakdown cover this.
    Inference errors kill a whole multi-turn item, and longer adapter outputs could err
    more. The V2 error budget and the new `pair --exclude-errors` sensitivity cover
    that.
21. **Thinking-on C-nat for Qwen3-0.6B (disclosed).** SUBCOR drops
    `reasoning_content`, so in-turn reasoning is not replayed in multi-turn history. The
    official QwenFCHandler keeps it. This is part of the measured path gap, not an arm
    difference: all arms run thinking off.

**Is "router decides" really true?**

22. **The gate is real (checked, no change).** `RouterGate` is a contrastive centroid
    score against a threshold fit from positives and negatives only (`safety_bias` 0).
    It sees only the last user message's frozen-base embedding: no tools, no
    tools-present shortcut and no seat filter, since the gates are unscoped. 0 units
    means the pure base. `top_k` is 1, and the harness never sends it.
23. **Routing was not auditable (fixed).** Nothing checked that a tenant held exactly
    one unit, or that every request was routed in SUBCOR.
    *Resolution:* `run_bfcl.py routing` writes per-item `fired_steps` / `fired_class`,
    per-category activation rates, and identity rates against a comparison run (V9,
    V10). `--expect-unit` fails on any foreign unit or missing routing block (V11).
    Section 4.1 fixes the single-unit tenant layout.
24. **The router's added value is still bounded by the tools-present heuristic
    (disclosed, unchanged).** On BFCL every request has tools, so A3 can at best match
    A2 there. The router's value shows only where tools are absent: IFEval, GSM8K and
    the probe.

**Still open (not fixed here)**

25. **V2 timeout (src/, orchestrator-owned).** `LlamaCppServingAdapter` keeps
    `timeout=60.0`, `subcor serve` has no flag for it, and `TimeoutError` still drops the
    connection. This is required before any arm runs.
26. **V10's tensor comparison needs the trained files.** It must be run after training.
    It reads the A2 GGUF and `<state>/llamacpp/loras/*.gguf` with `gguf`.
