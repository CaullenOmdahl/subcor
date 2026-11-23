# SUBCOR Architecture & Behavioral Specification

## 1. Purpose & Design Thesis

SUBCOR is a self-learning adaptation layer designed to run on consumer hardware (starting with 16 GB Apple Silicon) and integrate cleanly with governed agent operating systems such as **CaPheOS**.

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
  4. **Owner-controlled policy:** The thresholds, anchor cases and router-gate negatives come from the server owner's `ConformancePolicy`. A proposer can tighten them or add cases, never loosen or replace them. The owner anchors are always scored, and the gate sees the worse of the anchor and proposer-baseline regressions. Non-finite or out-of-range scores and thresholds are rejected, including by direct `record_conformance` callers.
  5. **Atomic rollback:** Rolling back an `ENABLED` unit immediately marks it `ROLLED_BACK` and removes its delta from the active stack without touching the frozen base or unrelated units.
  6. **Parameterization rule on export:** Exported CaPheOS bundles strip company-specific transcripts and secrets, exporting only portable weights, manifest, declared capabilities, and conformance test metadata.

### 2.6 Apple Silicon Frozen Kernel & Live Intern Loop (`MLXKernel`, `TextTrajectoryStore`)
- **Actor:** Intern seat running a compact quantized open-weight LLM (default `mlx-community/Qwen3.5-0.8B-4bit`, 652 MB, or any `mlx-lm` checkpoint such as Gemma 4 E2B QAT).
- **Object:** Frozen `nn.QuantizedLinear` projections wrapped in `SubcorMLXLinear` + append-only `TextTrajectoryStore`.
- **Trigger:** Live inference (`generate_with_trace`), zero-gradient contrastive weight-space surgery (`subcor mlx-surgery`), or on-policy QLoRA distillation (`subcor mlx-intern`).
- **Observable contract:**
  - Base quantized model parameters remain frozen (`model.freeze()`) at all times; adapters are hot-swapped via `mount_unit` / `unmount_unit` with SHA-256 verification.
  - Live per-layer mean hidden-state activations are captured off-graph when `capture_activations=True` with zero overhead when disabled.
  - Dynamic LoRA expansion (`attach_trainable_lora`) initializes `lora_b = 0` and verifies `max|logits_after - logits_before| == 0.0` before step 0.
  - Hybrid models (such as Qwen3.5 GatedDeltaNet) switch cleanly to `model.train()` during adaptation so differentiable MLX ops run instead of inference-only Metal kernels, then return to `model.eval()` and `model.freeze()`.
  - Promoted Skill Units export to **both** CaPheOS format-2 `module.json` bundles and stock `mlx-lm` `adapters.safetensors` + `adapter_config.json`.
  - Unmounting a Skill Unit restores exact frozen base logits (`rollback_max_logit_diff == 0.0`).

### 2.7 Routed MoE Skill Layer (`RouterGate`, `MoESkillRouter`)
- **Actor:** `MLXKernel.generate_routed` selecting active Skill Units per turn.
- **Object:** Frozen-base prompt embedding at `tap_layer` (captured with adapters bypassed so coordinates never drift as units are added/removed) + registered `RouterGate`s (`router_gate.json`).
- **Trigger:** Incoming prompt routed before generation (`top_k=1` by default).
- **Observable contract:**
  - Each `RouterGate` is fitted analytically from verified positive vs. negative frozen-base embeddings (zero router backprop, zero small-batch router collapse).
  - Routing uses raw user prompt embeddings (no task-specific system prompt leakage).
  - `seat_scope` is a routing hint, not access control. A scoped gate is filtered for any other seat; a library call with no (or a blank) seat considers every gate. The HTTP chat endpoint passes a seat to every backend that accepts one, as all built-in backends do (`X-CaPheOS-Seat`, missing or blank → `intern`). Tenant and principal checks remain the authorization boundary.
  - When no expert clears its calibrated threshold, `RoutingDecision.fell_back_to_base` is `True` and 0 experts are mounted — producing bit-identical frozen-base behavior (**0.0% general regression**).
  - Exported CaPheOS format-2 bundles include both `weights.scun` and `router_gate.json`.

### 2.8 Standardized Longitudinal Benchmark System (`BenchmarkSuite`, `BenchmarkLedger`)
- **Actor:** Operator, CI, or automated post-promotion audit duty.
- **Object:** Immutable versioned `BenchmarkSuite` (with SHA-256 `suite_digest`) + append-only `BenchmarkLedger` (`history.jsonl`) + generated Markdown Growth Report.
- **Trigger:** `subcor bench run` against either local MLX state or any live OpenAI-compatible HTTP endpoint (including `subcor serve` or frontier teacher baselines).
- **Observable contract:**
  - Built-in `subcor-core-v1` evaluates 5 functional axes: locked anchors (`general_reasoning`, `instruction_discipline`), general `structured_json`, and acquired domain skills (`domain_ops_routing`, `domain_noc_triage`).
  - Records exact active Skill Units, SHA-256 digests, total adapter bytes, `bits/weight`, per-axis pass rates, base-fallback rates, token-limit loop rate, latency, and tok/s.
  - `compare_snapshots(baseline, current)` checks suite digest equality and flags any regression on locked anchor axes.

### 2.9 Cortical Handoff / Thalamic Uncertainty Switch & Tool Execution (`subcor.confidence`)
- **Actor:** Runtime inference server (`SubcorRequestHandler` / `TenantManager`) cooperating with upstream Prefrontal Cortex seats.
- **Object:** Per-token decode log-probabilities (`StreamTokenEvent.logprob`), `RoutingDecision`, repetition ratio, token-limit flag, and raw model `<tool_call>` blocks.
- **Trigger:** Every `/v1/chat/completions` turn (non-streaming JSON or true token-by-token SSE streaming).
- **Observable contract:**
  - Full multi-turn conversation history (`system`, `user`, `assistant`, `tool`) and OpenAI `tools` definitions are passed natively to the tokenizer chat template.
  - Native `<tool_call>` blocks (both Qwen3.5 XML `<function=...><parameter=...>` and JSON formats) are parsed into standard OpenAI `tool_calls` with `finish_reason="tool_calls"`.
  - `evaluate_turn_confidence()` computes geometric-mean token probability, minimum token probability, NLL, and router margin, setting `escalate_to_cortex=True` (surfaced in `X-SUBCOR-Escalate`, `X-SUBCOR-Confidence`, and `subcor.confidence`) whenever mean probability drops below threshold, a localized uncertainty spike occurs, a repetition loop or token limit is hit, or a strict habitual seat receives an unmapped task.

### 2.10 Portable Skill Specs & Cross-Kernel Recompilation (`SkillSpec`, `subcor recompile`)
- **Actor:** Operator upgrading the frozen base kernel ("impact drill") to a newer or larger checkpoint.
- **Object:** `skill_spec.json` stored alongside each Skill Unit's `manifest.json` (which records `base_kernel_id`) and exported into CaPheOS format-2 bundles.
- **Trigger:** `subcor recompile --source-units ... --dest-state ... --tenant ... --model <new-kernel>`.
- **Observable contract:**
  - Treats `payload.scun` as a kernel-specific compiled binary and `skill_spec.json` as portable workflow source code (verified training pairs, locked eval/baseline cases, negative routing prompts, and hyperparameters).
  - Recompiles every `ENABLED`/`CONFORMANT` unit onto the target kernel, runs the locked Conformance Gate, tags the new manifest with the target `base_kernel_id`, and leaves units in `CONFORMANT` (never self-enabling) for independent sign-off.
  - Gracefully skips legacy units without `skill_spec.json` (`status="skipped_no_spec"`).

---

## 3. Architecture Decision Records (ADRs)

### ADR-001: Keep the Core Engine Dependency-Minimal (`numpy` + stdlib)
- **Decision:** The core `subcor` package depends only on `numpy` and Python's standard library. Heavy inference/training backends (`mlx`, `mlx-lm`, `llama.cpp` HTTP client, `torch`) live behind optional extras and clean backend interfaces.
- **Tradeoff:** We implement alternating binary factorization, contrastive SVD surgery, and STE low-rank micro-batch updates directly in `numpy` rather than relying on PyTorch autograd for the core engine.
- **Why:** Runs identically and deterministically in <1 second in CI and on any developer machine, stays true to CaPheOS's "one dependency is a decision" discipline, and prevents locking the architecture to one hardware vendor before Milestone 2.

### ADR-002: Separate Process / Out-of-Process Model Serving Boundary
- **Decision:** SUBCOR never embeds Python/ML runtimes inside a CaPheOS Node process. In production (Milestone 2+), the frozen base model runs in a dedicated local inference server (`mlx_lm.server` or `llama-server`), and SUBCOR manages adapter/unit lifecycle over a typed provider boundary.
- **Why:** Preserves CaPheOS's read-only immutable Node container and keeps GPU/unified-memory pressure isolated on the local host (crucial on 16 GB Apple Silicon).

### ADR-003: Dual-Mode Adaptation (Surgery + Expansion) Rather Than Pure Weight-Editing
- **Decision:** Use zero-gradient weight-space surgery *only* for structural activation defects (loops, refusal quirks, routing biases), and use function-preserving dynamic low-rank expansion + STE gradient adaptation for new skills.
- **Why:** Empirical evidence from MIT CSAIL's weight-space programming study shows direct weight editing unblocks latent defects cleanly (Qwen3.5-0.8B loops) but cannot synthesize complex missing reasoning circuits from scratch (Gemma 3 270M).

### ADR-004: Content-Addressed Skill Units Mapped 1:1 to CaPheOS Unit Standard
- **Decision:** Every weight delta or adapter is packaged as a versioned unit with SHA-256 digest, lifecycle state, conformance report, and a direct exporter to CaPheOS `module.json` (format 2).
- **Why:** Solves rollback, auditability, portability across companies/deployments, and reuse from day one.

### ADR-005: Target Very Small, Capable Quantized Kernels First (0.6B–4B)
- **Decision:** Default local target is `mlx-community/Qwen3.5-0.8B-4bit` (~652 MB on disk, <1.5 GB unified RAM), with first-class `--model` support for Gemma 4 E2B QAT and compact Qwen 3/3.5 checkpoints.
- **Why:** Fast iteration cycles (<60s end-to-end on a 16 GB Apple Silicon laptop), low memory footprint alongside a running CaPheOS company, and high compression leverage when domain skills live in modular `< 0.05` bits/weight Skill Units rather than bloating a single monolithic checkpoint.

### ADR-006: Tiny Dense Shared Core + Routed MoE Skill Units Over Native Small MoE
- **Decision:** Implement MoE routing at the SUBCOR Skill Unit layer (`MoESkillRouter`) over a frozen dense 0.8B–4B shared core, rather than requiring a 26B–30B native MoE base checkpoint.
- **Why:** Native MoE checkpoints still keep all experts in unified RAM (~14–16 GB at 4-bit), whereas a frozen 652 MB shared core + ~81 KiB routed Sub-1-Bit Skill Units fits 1,000+ skills inside ~1.5 GB RAM with zero cross-skill interference and zero general-knowledge regression.

### ADR-007: Cortical-Subcortical Split ("Impact Drill + Recompilable Bits")
- **Decision:** Treat SUBCOR strictly as a **procedural subcortical runtime** (workflows, tool usage, routing, parsing discipline) paired with normal context/RAG/MCP for factual memory and frontier models as the **Prefrontal Cortex** for novel planning and exceptions—connected via a live **Thalamic Uncertainty Switch** (`TurnConfidence` -> `X-SUBCOR-Escalate`).
- **Why:** Storing dense encyclopedic knowledge in sub-1-bit weights fights physics, whereas encoding repeatable decision procedures and tool habits in `~81 KiB` routed bits lets small models outperform general-purpose multitool models on specialized routines while cleanly escalating out-of-distribution turns.

### ADR-008: Portable `skill_spec.json` Source + `subcor recompile` for Kernel Upgrades
- **Decision:** Accept that compiled `.scun` weight deltas are mathematically coupled to a specific `base_kernel_id`, and solve long-term portability by storing a sanitized `skill_spec.json` alongside every Skill Unit and providing `subcor recompile`.
- **Why:** Just as software ships portable source code and recompiles binaries for new CPU architectures, storing the verified procedural spec + locked conformance cases lets an organization upgrade from a `0.8B` kernel to a newer `2B` or `4B` kernel and automatically re-verify its entire Skill Unit library without losing institutional habits.


---

## 4. Method Details

The maths behind each component, for readers who want more than the contracts above. Where the idea for each piece came from is in [`BACKGROUND.md`](BACKGROUND.md).

### 4.1 Immutable Quantized Base Kernel

A compact open-weight quantized checkpoint $f_{\theta_0}$ (default `mlx-community/Qwen3.5-0.8B-4bit`, 24 layers, `1024 × 3584` MLP projections, `652 MB` on disk) is loaded once and frozen (`model.freeze()`). Base weights are strictly read-only (`writeable=False` in the NumPy core; `base.freeze()` in `SubcorMLXLinear`). Every target linear projection $W_{\text{base}} \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}$ is wrapped so runtime inference computes:

$$y = x W_{\text{base}}^T + \sum_{u \in \mathcal{A}(x)} \Delta_u(x)$$

where $\mathcal{A}(x)$ is the set of Skill Units selected by the router for prompt $x$. For performance, mounted adapters pre-fold their scalar scale into $B$ at mount time and execute in the base model's native compute dtype (`bfloat16`/`float16`), eliminating per-token FP32 casts during decode.

### 4.2 Verified Evidence Gate

All runtime trajectories are appended to an immutable JSONL store (`TrajectoryStore` / `TextTrajectoryStore`). Each record carries a `VerificationEvidence(verified, verifier, method)` attestation. Live generations enter the store with `verified=False` until confirmed by either a deterministic verifier (schema check, unit test, exit code) or explicit senior-seat/owner review. Both contrastive activation extraction and distillation dataset export strictly exclude unverified records, preventing hallucinated student drafts from entering weight updates.

### 4.3 Dual-Mode Adaptation Engine

Empirical evidence shows that direct weight editing unblocks latent defects cleanly but cannot synthesize complex missing circuits from scratch. SUBCOR separates adaptation into two distinct modes:

#### Mode A — Zero-Gradient Contrastive Weight-Space Surgery (`WeightSpaceSurgeon`)
Given verified activation snapshots from failing/looping runs $H_{\text{fail}} \in \mathbb{R}^{N_f \times d_{\text{out}}}$ and clean passing runs $H_{\text{clean}} \in \mathbb{R}^{N_c \times d_{\text{out}}}$ at each instrumented layer $\ell$:

1. Score layer $\ell$ by normalized contrastive separation:
   $$S(\ell) = \frac{\|\mu_{\text{fail}}^{(\ell)} - \mu_{\text{clean}}^{(\ell)}\|_2}{\sqrt{\frac{1}{N_c}\sum_{i=1}^{N_c}\|h_{\text{clean},i}^{(\ell)} - \mu_{\text{clean}}^{(\ell)}\|_2^2 + \epsilon}}$$
2. Construct a rank-$k$ orthonormal failure subspace $U_k \in \mathbb{R}^{d_{\text{out}} \times k}$ starting from the normalized mean-difference direction and supplemented by orthogonalized leading SVD components of $(H_{\text{fail}} - \mu_{\text{clean}})$.
3. Synthesize a zero-gradient rank-$k$ ablation patch for $W_{\text{base}}^{(\ell)}$:
   $$\Delta W^{(\ell)} = -\alpha \, U_k (U_k^T W_{\text{base}}^{(\ell)})$$
   so $(W_{\text{base}}^{(\ell)} + \Delta W^{(\ell)})h = (I - \alpha U_k U_k^T) W_{\text{base}}^{(\ell)} h$, attenuating the failure direction while leaving orthogonal clean features invariant.

#### Mode B — Function-Preserving Dynamic Expansion & Quantized Adaptation
When the base model genuinely lacks a skill circuit, SUBCOR appends a new zero-initialized low-rank branch ($B_{\text{new}} = 0$ in FP32, or $s_{\text{out}} = 0$ in Binary STE) and verifies the **strict zero-perturbation invariant** before step 0:

$$\max_x \|f_{\text{expanded}}(x) - f_{\text{base}}(x)\|_\infty = 0.0$$

Only the newly attached branch is unfrozen and optimized over verified `(prompt, senior_correction)` pairs via completion-masked causal cross-entropy (`AdamW`, cosine decay, gradient norm clipping at `1.0`). For hybrid architectures such as Qwen3.5's GatedDeltaNet, the model switches cleanly to `model.train()` during adaptation (using differentiable MLX ops rather than inference-only Metal kernels) and returns to `model.eval()` and `model.freeze()` immediately after extraction. SUBCOR supports both FP32 warm-start followed by sub-1-bit factorization and opt-in Two-Stage Binary STE QAT (`_ste_sign_pm1` with clipped straight-through gradient $\mathbf{1}_{|z|\le 1}$).

### 4.4 Sub-1-Bit Low-Rank Binary Factorization (`Sub1BitMatrix`)

Inspired by NanoQuant, SUBCOR compresses any layer weight delta $\Delta W \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}$ into bit-packed binary low-rank factors plus float16 scaling vectors:

$$\Delta W \approx \operatorname{diag}(s_1)\,\left(U_{\pm 1}\operatorname{diag}(\sigma)V_{\pm 1}^T\right)\,\operatorname{diag}(s_2)$$

where $U_{\pm 1} \in \{-1, +1\}^{d_{\text{out}} \times r}$ and $V_{\pm 1} \in \{-1, +1\}^{d_{\text{in}} \times r}$ are bit-packed into `uint8` buffers via `np.packbits` (1 bit per entry), and $s_1 \in \mathbb{R}^{d_{\text{out}}}$, $s_2 \in \mathbb{R}^{d_{\text{in}}}$, $\sigma \in \mathbb{R}^r$ are stored in `float16`.

Fitting proceeds via:
1. Row/column RMS pre-normalization.
2. Greedy residual binary rank-1 decomposition initialized by **power iteration** (12 iterations for matrices with $\min(d_{\text{out}}, d_{\text{in}}) > 96$, avoiding $O(m^2 n)$ dense SVD) followed by alternating sign maximization $u_k \leftarrow \operatorname{sign}(R v_k)$, $v_k \leftarrow \operatorname{sign}(R^T u_k)$.
3. Alternating closed-form least-squares refinement sweeps over $(U, V, \sigma)$, $s_1$, and $s_2$.

Effective storage per dense parameter is:

$$\text{bpw} = \frac{(d_{\text{out}} + d_{\text{in}})r + 16(d_{\text{out}} + d_{\text{in}} + r)}{d_{\text{out}} \, d_{\text{in}}}$$

For `Qwen3.5-0.8B-4bit`'s `mlp.down_proj` layers ($d_{\text{out}} = 1024$, $d_{\text{in}} = 3584$, $3.67\text{M}$ weights/layer) at rank $r = 8$:
- Parameter storage per layer is `110,720` bits (`13.5 KiB`) = **`0.0302 bits/weight`**.
- Across 6 adapted layers (`22.02M` weights), a complete Skill Unit is **`81.4 KiB`**—**5.3× smaller** than stock FP16 `adapters.safetensors` (`433 KiB`) and **~530× smaller** than dense FP16 deltas.

### 4.5 Training-Free Frozen-Base Activation-Space MoE Router (`MoESkillRouter`)

To compose multiple Skill Units without cross-skill interference or general-knowledge forgetting, SUBCOR implements a **Shared Dense Core + Routed Skill Units** topology:

1. **Invariant Prompt Tap (Early-Exit Prefill):** For any user prompt $p$ (evaluated with `system_prompt=None` so routing is strictly semantic), SUBCOR bypasses all mounted adapters (`_bypass_adapters=True`) and executes layers $0 \dots \ell_{\text{tap}}$ only—stopping immediately after $\ell_{\text{tap}}$ and skipping downstream layers plus the 248,320-token `lm_head`. The mean output activation at $\ell_{\text{tap}}$ is L2-normalized to produce an embedding $e(p) \in \mathbb{S}^{d-1}$ that is **mathematically invariant** to which Skill Units are installed or enabled.
2. **Analytical Contrastive `RouterGate`:** Each Skill Unit stores a lightweight gate fitted analytically from verified positive prompt embeddings $E^+$ vs. verified negative/general embeddings $E^-$:
   $$\mu_+ = \operatorname{unit}(\overline{E^+}), \quad \mu_- = \operatorname{unit}(\overline{E^-}), \quad s(e) = \langle e, \mu_+ \rangle - \langle e, \mu_- \rangle$$
   The decision threshold $\tau$ is calibrated conservatively above $\max_{e \in E^-} s(e)$ (a 65/35 blend between $\max(s^-)$ and the 15th percentile of $s^+$ when separable).
3. **Zero-Interference Fallback:** At inference time, `MoESkillRouter.route(e, top_k=1)` activates only units with $s_u(e) \ge \tau_u$. When no unit clears its threshold, $\mathcal{A}(x) = \emptyset$ and inference runs on the pure frozen base kernel with **exact zero perturbation**.

### 4.6 Governed Skill Unit Lifecycle & Multi-Tenant Isolation

Every Skill Unit is packaged into a content-addressed directory (`manifest.json`, `payload.scun`, optional `router_gate.json`) governed by a strict finite state machine:

$$\texttt{DORMANT} \xrightarrow{\text{locked conformance pass}} \texttt{CONFORMANT} \xrightarrow{\text{approver} \neq \text{proposer}} \texttt{ENABLED} \xrightarrow{\text{atomic rollback}} \texttt{ROLLED\_BACK}$$

Four invariants are enforced in code:
- **Content Integrity:** Every load, promotion, and export verifies payload SHA-256 against `manifest.payload_sha256` (fail-closed `IntegrityError`).
- **Locked Conformance Gate:** Candidate units must improve target capability by $\ge \Delta_{\min}$ while keeping baseline regression within tolerance $\epsilon_{\max}$. Both limits, the anchor cases and the router-gate negatives come from the server owner's `ConformancePolicy`; a study request may tighten the limits and add cases, never loosen or replace them, and every score and threshold must be finite and in range.
- **No Self-Approval:** `promote(unit_id, approver)` raises `GovernanceError` whenever `approver == manifest.proposer`.
- **Multi-Tenant Serving, Tool Calling, Streaming & Portable Export:** `SubcorHTTPServer` shares one frozen base kernel across $N$ isolated tenants (`subcor/<tenant_id>`), each with its own unit registry, trajectory store, and MoE router. `/v1/chat/completions` preserves full multi-turn conversation history, accepts OpenAI `tools` definitions and parses native XML/JSON `<tool_call>` blocks into structured `tool_calls`, and streams tokens incrementally via `mlx_lm.stream_generate` SSE. Promoted units export to both **CaPheOS format-2 `module.json` bundles** (including `router_gate.json`, `skill_spec.json`, `baseKernelId`, and runnable Node `conformance/test.js`) and stock **`mlx-lm` `adapters.safetensors`**.

### 4.7 Cortical Handoff (`TurnConfidence`) & Cross-Kernel Recompilation (`subcor recompile`)

To close the loop between subcortical speed and cortical reliability, every inference turn records per-token decode log-probabilities $\{ \ell_t \}_{t=1}^T$ alongside router margin $m = s_{u^*} - \tau_{u^*}$, repetition ratio, and token-limit exhaustion. `evaluate_turn_confidence()` computes geometric-mean token probability $\bar{p} = \exp(\frac{1}{T}\sum_t \ell_t)$ and worst-token probability $p_{\min} = \exp(\min_t \ell_t)$, raising `escalate_to_cortex = True` (surfaced via HTTP headers `X-SUBCOR-Escalate` and `X-SUBCOR-Confidence` and in `subcor.confidence`) whenever:
1. the token budget is exhausted (`token_limit_exhausted`),
2. n-gram repetition exceeds policy (`repetition_loop_detected`),
3. a strict habitual seat receives an unmapped prompt (`no_matching_skill_unit`),
4. $\bar{p} < 0.55$ (`low_decode_confidence`), or
5. a localized uncertainty spike occurs ($p_{\min} < 0.05$ with $\bar{p} < 0.72$).

To make learned habits durable across base-model upgrades, every Study Cycle persists a sanitized `skill_spec.json` (verified `(prompt, target)` pairs, locked evaluation and baseline anchor cases, negative routing prompts, and training hyperparameters) alongside the kernel-tagged manifest (`base_kernel_id`). Running `subcor recompile` against a newer or larger base checkpoint retrains each eligible unit from its `skill_spec.json`, passes it through the locked Conformance Gate, and places passing units in `CONFORMANT` (never self-enabled) for independent approval.
