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

- Local backend interface (`InferenceBackend`) with implementations for `mlx-lm` (Gemma 4 E2B/E4B QAT, Qwen3.5-0.8B/4B) and `llama-server` LoRA hot-swap API.
- Real hidden-state activation hook capture during generation for contrastive loop detection.
- Export SLAINGI low-rank deltas directly to `safetensors` / GGUF LoRA adapter files loadable by `mlx-lm` and `llama.cpp`.
- Real reproduction of MIT Experiment 8-style structural loop ablation on a local 0.8B–4B checkpoint.

**Exit Gate:** Live local quantized model serves tasks, captures activation traces, applies a hot-swapped adapter/patch without mutating the base checkpoint, and rolls back cleanly within 16 GB unified memory.

---

## Milestone 3 — On-Policy "Intern" Distillation Loop

Turn senior-seat review transcripts into scheduled overnight adapter growth:

- Ingest `(intern_attempt, senior_correction, verified_outcome)` triples from real run journals.
- Automatic failure clustering: route structural repetition/formatting defects to `WeightSpaceSurgeon` and capability gaps to QLoRA / STE `ElasticLinear` expansion.
- Automated held-out regression suite runner with strict budget and compute caps.

**Exit Gate:** Local Intern model demonstrably improves pass rate on a held-out benchmark of real company/coding tasks across two overnight cycles with zero baseline regression.

---

## Milestone 4 — CaPheOS Live Integration

Run SLAINGI as a real CaPheOS proving ground:

- Mount SLAINGI as a CaPheOS `capability` / `model-ops` unit talking to a local inference endpoint outside the immutable CaPheOS container.
- Wire `people` (model choices), `tools` (unit synthesis), and `risk` (locked conformance suite & promotion sign-off) seats into the live approval workflow.
- Export battle-tested learned Skill Units from one company and import them cleanly into a fresh second company.

**Exit Gate:** End-to-end CaPheOS company run where an Intern seat earns graduated autonomy via promoted Skill Units and cleanly exports a portable unit with zero company data leakage.
