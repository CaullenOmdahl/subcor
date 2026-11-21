# SLAINGI Roadmap

Every milestone has an explicit exit gate. A milestone is not done until its exit gate passes with real evidence.

## Milestone 1 — Core Self-Learning Engine & Governed Proving Ground (Current)

Build and verify the full mathematical and governance loop with zero external runtime dependencies beyond Python + NumPy:

- [x] Architecture spec, behavioral contracts, and ADRs (`docs/ARCHITECTURE.md`)
- [x] Verified-evidence JSONL trajectory store (`slaingi.telemetry`)
- [x] Bit-packed Sub-1-Bit binary low-rank factorization (`slaingi.sub1bit`)
- [x] Function-preserving dynamic low-rank expansion + FP32 & binary STE adaptation (`slaingi.expansion`)
- [x] Contrastive weight-space surgeon for zero-gradient structural loop ablation (`slaingi.surgeon`)
- [x] Governed Skill Unit registry with SHA-256 integrity, conformance gate, no self-approval, atomic rollback, and CaPheOS `module.json` (format 2) export (`slaingi.units`)
- [x] Executable Micro-Intern Proving Ground (`slaingi.proving_ground`) + CLI (`slaingi`)

**Exit Gate:** `uv run ruff check`, `uv run ruff format --check`, `uv run pytest`, and `uv run slaingi demo --export-capheos ...` all pass cleanly, proving:
1. Structural loop defect eliminated with zero gradient descent while preserving baseline tasks.
2. Novel missing skill learned via zero-perturbation dynamic rank expansion + STE, compressed to `< 1.0` bits/weight.
3. Poisoned/regressive patch blocked by conformance gate; self-approval blocked; enabled unit rolled back cleanly.
4. Valid CaPheOS format-2 unit bundle exported with verified digest.

---

## Milestone 2 — Apple Silicon Local Kernel Backend (`mlx-lm` / `llama.cpp`)

Connect the frozen-kernel + hot-swappable adapter runtime to real compact QAT open-weight models on 16 GB Apple Silicon (M5):

- [x] Apple Silicon MLX frozen-kernel backend (`slaingi.backends.mlx_kernel.MLXKernel`) defaulting to compact 4-bit `mlx-community/Qwen3.5-0.8B-4bit` (652 MB) with `--model` support for Gemma 4 E2B QAT and any `mlx-lm` checkpoint.
- [x] `SlaingiMLXLinear` wrapper capturing live per-layer hidden-state activations and mounting/unmounting multi-unit Sub-1-Bit & FP32 Skill Units without mutating `QuantizedLinear` base weights.
- [x] Export SLAINGI Skill Units directly to stock `mlx-lm` `adapters.safetensors` + `adapter_config.json` alongside CaPheOS format-2 `module.json`.
- [x] Live zero-gradient contrastive weight-space diagnosis & ablation CLI (`slaingi mlx-surgery`).

**Exit Gate:** Live local quantized model serves tasks, captures activation traces, applies a hot-swapped adapter/patch without mutating the base checkpoint, and rolls back cleanly within 16 GB unified memory.

---

## Milestone 3 — On-Policy "Intern" Distillation Loop

Turn senior-seat review transcripts into scheduled overnight adapter growth:

- [x] Text-level append-only `TextTrajectoryStore` (`slaingi.intern`) enforcing the Verified Evidence Gate on live LLM `(prompt, intern_response, senior_correction)` triples.
- [x] Function-preserving Apple Silicon GPU QLoRA adaptation (`MLXKernel.train_qlora_adapters`, verified `0.0` initial logit perturbation) + Sub-1-Bit binary compression (`< 0.05` bits/weight) + locked conformance gate (`slaingi mlx-intern`).
- [ ] Scheduled overnight cron/dispatcher runner with automated failure clustering (routing structural loop defects to `WeightSpaceSurgeon` vs. capability gaps to QLoRA expansion).

**Exit Gate:** Local Intern model demonstrably improves pass rate on a held-out benchmark of real company/coding tasks across overnight cycles with preserved baseline capabilities.

---

## Milestone 4 — CaPheOS Live Integration

Run SLAINGI as a real CaPheOS proving ground:

- Mount SLAINGI as a CaPheOS `capability` / `model-ops` unit talking to a local inference endpoint outside the immutable CaPheOS container.
- Wire `people` (model choices), `tools` (unit synthesis), and `risk` (locked conformance suite & promotion sign-off) seats into the live approval workflow.
- Export battle-tested learned Skill Units from one company and import them cleanly into a fresh second company.

**Exit Gate:** End-to-end CaPheOS company run where an Intern seat earns graduated autonomy via promoted Skill Units and cleanly exports a portable unit with zero company data leakage.
