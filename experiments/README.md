# Experiments

Research code that sits next to the `subcor` package. It isn't installed with the package,
and the core test suite doesn't depend on it.

## `bfcl/`: preregistered BFCL v4 study

Do routed SUBCOR Skill Units improve a small local model's tool use on the official
Berkeley Function Calling Leaderboard v4, and do they beat cheaper alternatives?

- **[`PREREGISTRATION.md`](bfcl/PREREGISTRATION.md)** fixes the hypotheses, arms, metrics,
  stopping rules and analysis before any GPU result exists. The arms are: frozen base,
  3-shot prompting, an always-on FP16 LoRA, and a routed sub-1-bit unit.
  [`DEVIATIONS.md`](bfcl/DEVIATIONS.md) logs every later change and whether a result had
  been seen when it was made.
- **[`harness/`](bfcl/harness/README.md)** runs the official `bfcl-eval` data, inference
  loops and graders unchanged, with native function calling. It adds no grader of its own.
- **`data/`** builds the training data from public tool-use datasets, with schema checks,
  rendering checks and semantic decontamination against the test set.
- **`general/`** contains the no-regression probes: an IFEval subset, a GSM8K subset, and a
  router probe, all with a paired bootstrap.

**Status:** the harness and data pipeline are complete and smoke-tested on CPU (the
`results/` files). The preregistered GPU arms have not been run yet.

## `chat_prep/`: Skill Units from real agent sessions

This is an end-to-end pipeline that turns a developer's own coding-agent transcripts
(Claude Code, Codex, DeepSeek Harness) into training data for per-capability Skill Units,
then measures whether the units help on held-out sessions.

| Step | Script | What it does |
|---|---|---|
| Convert | `convert.py` | Normalizes each harness's session format into one canonical multi-turn schema with OpenAI-style tool calls |
| Scrub | `scrub.py` | Removes gitleaks findings, built-in secret and PII rules, and a private blocklist, then re-scans as a gate (0 hits required). The audit log stores hashes, never the values. |
| Build | `build.py` | Cuts windows of at most 2,048 tokens, splits by session (newest 20% for eval), buckets by capability (tool-calling, code-writing, agentic decisions) and re-scrubs every record |
| Gate | `publish.py` | Runs a final leak gate over the built data before it is copied anywhere |
| Train and evaluate | `train_local.py` | Trains a unit with the PyTorch backend, then compares the base and base+unit on held-out sessions: per-window NLL with a bootstrap CI, next-action tool and argument match with exact McNemar tests, loop rate, and cross-bucket NLL as a regression check |
| Specialization | `unit_matrix.py` | Scores every unit on every bucket (3×3) |
| Habits | `habits.py` | Detects engineering habits (verify after a change, inspect first, least privilege, ...) as a benchmark |

**The data is private and never leaves the machine.** Only the code is here. The private
blocklist and the datasets live outside the repository.

**Results on held-out sessions** (aggregate metrics only), for Qwen3.5-4B with 4-bit
QLoRA on four layers, measured on an RX 6800:

| Unit | Next tool matches the developer's | Argument names match | Runaway output |
|---|---|---|---|
| Tool-calling | 62% → **97%** (p < 0.001) | 62% → **94%** (p < 0.001) | 3% vs 3% |
| Code-writing | 17% → **34.5%** (p = 0.011) | 17% → **32%** (p = 0.024) | no real loops at a 1,536-token cap |
| Agentic decisions | 29% → 36% (n.s.); exact action 13% → **24%** (p = 0.019) | | 8%, all cap hits |

- **Every unit improves every bucket's NLL by 0.10–0.17**, and each unit does best on its
  own bucket.
- **Loops were caused by decoding, not training.** Under pure greedy decoding the
  tool-calling unit kept extending chained commands, giving 23% runaway output. A decode
  repetition penalty of 1.1 (`--rep-penalty`) brought that to the base model's rate. The
  loss mask had been verified correct first.
