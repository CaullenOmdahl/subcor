# SLAINGI Architecture & Behavioral Specification

## 1. Purpose & Design Thesis

SLAINGI is a self-learning adaptation layer designed to run on consumer hardware (starting with 16 GB Apple Silicon) and integrate cleanly with governed agent operating systems such as **CaPheOS**.

Three hard engineering constraints shape the architecture:

1. **No monolithic checkpoint mutation.** Base weights are treated as an immutable OS kernel. All self-improvement happens through versioned, content-addressed delta modules (**Skill Units**) that can be mounted, composed, audited, and rolled back atomically.
2. **Match the adaptation mechanism to the failure mode.**
   - *Structural defects* (e.g., repetition loops, bad residual directions) are patched via **contrastive weight-space surgery** without gradient descent.
   - *Missing capabilities* (fuzzy knowledge or new circuits) require **dynamic low-rank capacity expansion** paired with **quantized gradient updates (STE)** over verified on-policy corrections.
3. **Governance is part of the learning loop, not an afterthought.** Self-modifying weights without verification and separation of duties silently regress or reward-hack. Every proposed delta must be grounded in verified evidence, pass a locked regression suite, and be promoted by an identity distinct from the proposer.

---

## 2. Intended Behavior Contracts

### 2.1 Verified Evidence Telemetry (`TrajectoryStore`)
- **Actor:** Intern seat (runtime execution) and Verifier / Senior reviewer.
- **Object:** Append-only JSONL trajectory log (`trajectories.jsonl`).
- **Trigger:** Task attempt finishes and emits outcome + optional activation snapshots and verification record.
- **State transition:** `recorded (unverified)` -> `verified (evidence_passed | senior_signed)` or `rejected`.
- **Invariants:**
  - `export_training_pairs()` **never** emits unverified trajectories (prevents "blind leading the blind").
  - Records are append-only and immutable once written; verification appends a signed verification event or stores immutable verified records.
- **Reversibility:** Non-destructive; filtering is deterministic over stored records.

### 2.2 Contrastive Weight-Space Surgery (`WeightSpaceSurgeon`)
- **Actor:** Auditor / `tools` seat during reflection/off-peak phase.
- **Object:** Activation snapshots across layers for failing (looping) vs. clean runs, and target base layer weights $W_{\text{base}}$.
- **Trigger:** Recurring structural failure cluster detected in verified failure telemetry.
- **State transition:** `(failing_activations, clean_activations, W_base)` -> proposed `LayerDelta` (rank-$k$ ablation patch).
- **Observable contract:**
  - Computes contrastive failure subspace $U_{\text{fail}}$ from activation differences between failing and clean runs.
  - Synthesizes a minimal rank-$k$ weight delta $\Delta W$ that suppresses projection onto $U_{\text{fail}}$ without running gradient descent or modifying $W_{\text{base}}$ in place.
  - Output delta can be stored as exact low-rank factors or compressed to a sub-1-bit binary factorized payload.
- **Reversibility:** Pure function; produces a `PROPOSED` Skill Unit without touching active inference state.

### 2.3 Function-Preserving Dynamic Expansion & Quantized Adaptation (`ElasticLinear`)
- **Actor:** Learning / `tools` seat when weight-space ablation cannot resolve a missing skill.
- **Object:** Frozen base linear layer $W_{\text{base}}$ + dynamic low-rank adapter stack $\{(A_i, B_i, \alpha_i)\}$.
- **Trigger:** Verified `(input, target/correction)` examples exist for a capability outside the base model's current competence.
- **State transition:** `rank = r` -> `rank = r + k (untrained, zero output delta)` -> `adapted rank = r + k` -> optional `sub-1-bit compressed Skill Unit`.
- **Observable contract:**
  - **Strict zero-perturbation invariant:** Immediately after `expand(k)`, for every input $x$, $\|f_{\text{expanded}}(x) - f_{\text{before}}(x)\|_\infty = 0.0$.
  - **Frozen base invariant:** `W_base` is read-only (`writeable = False`); adaptation updates *only* the newly expanded low-rank columns.
  - **STE binary mode:** Forward pass uses binarized factors ($\operatorname{sign}(U), \operatorname{sign}(V)$) with real scaling vectors while backward pass propagates gradients via Straight-Through Estimation, training directly around quantization constraints.
- **Reversibility:** Removing or disabling the adapter restores exact pre-expansion outputs.

### 2.4 Sub-1-Bit Binary Factorization (`Sub1BitFactorizer`)
- **Actor:** Skill Unit packager.
- **Object:** Dense or low-rank weight delta matrix $W \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}$.
- **Trigger:** Packaging a delta into a portable, compact Skill Unit.
- **Observable contract:**
  - Factors $W \approx \operatorname{diag}(s_1)\,(U_{\pm 1} V_{\pm 1}^T)\,\operatorname{diag}(s_2)$ where $U_{\pm 1} \in \{-1, +1\}^{d_{\text{out}} \times r}$, $V_{\pm 1} \in \{-1, +1\}^{d_{\text{in}} \times r}$, $s_1 \in \mathbb{R}^{d_{\text{out}}}$, $s_2 \in \mathbb{R}^{d_{\text{in}}}$.
  - Stores binary matrices bit-packed via `packbits` (1 bit per entry) and scaling vectors in float16/float32.
  - Reports exact `bits_per_weight = total_stored_bits / (d_out * d_in)`. For realistic matrix dimensions and small rank $r$, `bits_per_weight < 1.0`.
  - Serialization round-trip (`to_bytes` -> `from_bytes`) reproduces the bit-identical reconstructed matrix and matches the manifest SHA-256 digest.

### 2.5 Governed Skill Unit Lifecycle (`SkillUnitRegistry`)
- **Actor:** Proposer seat (`tools` / `model-ops`), Conformance Runner (locked automated gate), Approver seat (`risk` / `people` / owner).
- **Object:** Versioned, content-addressed Skill Unit (`manifest.json` + `payload.bin` + conformance report).
- **Lifecycle state machine:**

```text
PROPOSED ──(ingest + digest verified)──► DORMANT
DORMANT  ──(conformance gate passes)───► CONFORMANT
CONFORMANT ──(separate approver sign-off)──► ENABLED
ENABLED  ──(rollback / disable)────────► ROLLED_BACK
CONFORMANT | ENABLED ──(export)────────► CaPheOS module.json bundle (format 2)
```

- **Hard governance invariants:**
  1. **Digest integrity:** Loading or enabling verifies payload SHA-256 against manifest digest; any tampering fails closed.
  2. **No self-approval:** `promote(unit_id, approver=...)` raises `GovernanceError` if `approver == manifest.proposer`.
  3. **Conformance required:** A `DORMANT` or failed unit cannot be promoted to `ENABLED`. Conformance requires:
     - Baseline regression score $\ge \text{baseline\_before} - \text{max\_regression\_tolerance}$
     - Target skill/defect score strictly improved ($\ge \text{target\_before} + \text{min\_improvement}$)
  4. **Atomic rollback:** Rolling back an `ENABLED` unit immediately marks it `ROLLED_BACK` and removes its delta from the active stack without touching the frozen base or unrelated units.
  5. **Parameterization rule on export:** Exported CaPheOS bundles strip company-specific transcripts and secrets, exporting only portable weights, manifest, declared capabilities, and conformance test metadata.

---

## 3. Architecture Decision Records (ADRs)

### ADR-001: Keep the Core Engine Dependency-Minimal (`numpy` + stdlib)
- **Decision:** The core `slaingi` package depends only on `numpy` and Python's standard library. Heavy inference/training backends (`mlx`, `mlx-lm`, `llama.cpp` HTTP client, `torch`) live behind optional extras and clean backend interfaces.
- **Tradeoff:** We implement alternating binary factorization, contrastive SVD surgery, and STE low-rank micro-batch updates directly in `numpy` rather than relying on PyTorch autograd for the core engine.
- **Why:** Runs identically and deterministically in <1 second in CI and on any developer machine, stays true to CaPheOS's "one dependency is a decision" discipline, and prevents locking the architecture to one hardware vendor before Milestone 2.

### ADR-002: Separate Process / Out-of-Process Model Serving Boundary
- **Decision:** SLAINGI never embeds Python/ML runtimes inside a CaPheOS Node process. In production (Milestone 2+), the frozen base model runs in a dedicated local inference server (`mlx_lm.server` or `llama-server`), and SLAINGI manages adapter/unit lifecycle over a typed provider boundary.
- **Why:** Preserves CaPheOS's read-only immutable Node container and keeps GPU/unified-memory pressure isolated on the local host (crucial on 16 GB Apple Silicon).

### ADR-003: Dual-Mode Adaptation (Surgery + Expansion) Rather Than Pure Weight-Editing
- **Decision:** Use zero-gradient weight-space surgery *only* for structural activation defects (loops, refusal quirks, routing biases), and use function-preserving dynamic low-rank expansion + STE gradient adaptation for new skills.
- **Why:** Empirical evidence from MIT CSAIL's weight-space programming study shows direct weight editing unblocks latent defects cleanly (Qwen3.5-0.8B loops) but cannot synthesize complex missing reasoning circuits from scratch (Gemma 3 270M).

### ADR-004: Content-Addressed Skill Units Mapped 1:1 to CaPheOS Unit Standard
- **Decision:** Every weight delta or adapter is packaged as a versioned unit with SHA-256 digest, lifecycle state, conformance report, and a direct exporter to CaPheOS `module.json` (format 2).
- **Why:** Solves rollback, auditability, portability across companies/deployments, and reuse from day one.
