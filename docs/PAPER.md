# SUBCOR: Governed Sub-1-Bit Routed Skill Units for Self-Learning Compact Language Models

**Author:** Caullen Omdahl  
**Date:** November 21, 2026  
**Status:** Working Methodology Draft (v0.1 — Timestamped Preprint)  
**Reference Implementation:** `subcor` (`0.1.0`)

---

## Abstract

Most practical "self-improving" AI agent frameworks today never modify neural network weights; they instead mutate prompts, memory stores, or retrieval indices. Conversely, recent neural weight-editing and low-bit compression papers—such as direct weight-space programming, sub-1-bit binary factorization, and quantization-aware training—are typically studied in isolation on static checkpoints without runtime governance, rollback guarantees, or protection against catastrophic forgetting.

We present **SUBCOR** (*Sub-Bit Unit-Based Continual On-Policy Runtime*), a governed runtime architecture that enables **sub-billion-parameter quantized language models** to acquire new structured capabilities and patch structural failure modes on consumer hardware without overwriting base model weights. SUBCOR treats a frozen 4-bit model (e.g., `Qwen3.5-0.8B-4bit`, 652 MB) as an immutable shared core and represents every learned capability or surgical patch as a **content-addressed, sub-1-bit binary Skill Unit** ($\approx 0.0302$ bits/weight, `81.4 KiB` across 22M adapted MLP weights). To prevent cross-skill interference and general-knowledge regression, Skill Units are selected dynamically via a **training-free, frozen-base activation-space MoE router** that falls back cleanly to zero active experts on out-of-domain queries. Every weight delta passes through four enforceable governance gates: a **Verified Evidence Gate** on incoming trajectories, **SHA-256 payload integrity**, an immutable **Conformance Gate** with locked anchor non-regression tolerances, and **separation of duties (no self-approval)** with instant atomic rollback.

We evaluate SUBCOR both on a controlled multi-layer residual proving ground and live on Apple Silicon (M5, 16 GB unified memory) running `Qwen3.5-0.8B-4bit` over a standardized, content-hashed 5-axis longitudinal benchmark (`subcor-core-v1`). Adding two independent routed sub-1-bit Skill Units (`162.8 KiB` total) increases unseen company operations routing accuracy from `0.0%` to `60.0%–80.0%` (`100.0%` semantic routing accuracy across unseen prompts), preserves all locked general-reasoning and instruction-discipline anchors at `+0.0 pp` regression (`80.0% → 80.0%`), reduces average turn latency (`171 ms → 137 ms`) via an `18.3 ms` early-exit routing tap and native-dtype mounted adapters, and supports sustained HTTP decode at `154.1 tok/s`. We also report two negative empirical findings—single-layer rank-1 live surgery failing conformance on verbose word problems and seed variance in cold-start Binary STE on 12-example batches—demonstrating the necessity of locked conformance gates in self-modifying systems.

---

## 1. Introduction

Deploying autonomous agents in real organizational workflows exposes an acute economic and architectural tension. Frontier models provide broad reasoning capability but incur high marginal token cost, latency, and rate-limit pressure on repetitive operational turns (routing, triage, JSON tool formatting, log parsing, policy enforcement). Small open-weight models (`0.6B–4B` parameters) run locally at near-zero marginal cost and $>100$ tokens/sec on consumer laptops, yet lack company-specific conventions and degrade rapidly when naively fine-tuned on narrow tasks.

Four structural barriers have prevented small local models from acting as genuinely self-improving "Intern" workers:

1. **Monolithic Checkpoint Mutation:** Fine-tuning or editing a single checkpoint entangles new skills with base capabilities, making silent regression difficult to detect and rollback costly.
2. **Catastrophic Forgetting Under Dense Adaptation:** Mounting dense low-rank adapters unconditionally perturbs activations on every token—even on general reasoning tasks unrelated to the newly learned domain.
3. **Storage and Transfer Overhead:** Standard FP16 LoRA adapters require hundreds of kilobytes to tens of megabytes per skill and lack standardized manifests, integrity digests, or capability declarations.
4. **Ungoverned Self-Modification:** Allowing an agent to collect unverified transcripts, synthesize weight updates, and promote those updates without independent verification invites compounding hallucination ("blind leading the blind") and silent capability erosion.

SUBCOR resolves these four barriers by unifying four complementary mechanisms around a single portable artifact—the **Governed Sub-1-Bit Skill Unit**—designed to run standalone or alongside governed agent operating systems such as CaPheOS.

---

## 2. Related Work

SUBCOR synthesizes five active research threads into one cohesive, governed runtime:

- **Low-Rank & Quantized Adaptation:** QLoRA (Dettmers et al., 2023) established training low-rank adapters over frozen 4-bit base models; Punica, S-LoRA, and LoRAX demonstrated multi-tenant hot-swapping of LoRA adapters over a shared base model. SUBCOR extends this multi-tenant pattern with sub-1-bit binary packing and strict per-tenant isolation.
- **MoE-Style Adapter Routing:** LoRAHub, LoRAMoE, Arrow, and PHATGOOSE route across pools of LoRA modules. Rather than training a parameterized router network (which collapses on 10–20 on-policy examples), SUBCOR uses a training-free contrastive prototype gate tapped directly from **frozen-base** hidden activations.
- **Weight-Space Programming & Directional Ablation:** Representation Engineering (Zou et al., 2023), Directional Ablation for refusal removal (Arditi et al., 2024), and MIT CSAIL's *"Sparks of Weight-Space Programming"* (Takehi et al., Nov 2026) showed that structural behavioral defects—such as thinking loops in `Qwen3.5-0.8B`—can be ablated directly in weight space without gradient descent. SUBCOR incorporates zero-gradient contrastive directional surgery as a first-class adaptation mode alongside gradient expansion.
- **Sub-1-Bit Quantization & QAT:** NanoQuant (Chong et al., ICML 2026), LittleBit, and BTC-LLM (ACL 2026) break the 1-bit-per-weight barrier using low-rank binary factorization and codebooks; Google DeepMind's Gemma 4 QAT demonstrated training around low-bit rounding constraints via Straight-Through Estimators (STE). SUBCOR applies NanoQuant-style bit-packed factorization to **portable skill deltas** rather than pre-training from scratch.
- **On-Policy Distillation:** Generalized Knowledge Distillation (Agarwal et al., 2024) and on-policy self-distillation show that training a compact student on corrections to its own trajectories outperforms passive off-policy imitation. SUBCOR embeds this pattern organizationally as an "Intern" seat supervised by senior seats and deterministic verifiers.

---

## 3. Methodology

### 3.1 Immutable Quantized Base Kernel

A compact open-weight quantized checkpoint $f_{\theta_0}$ (default `mlx-community/Qwen3.5-0.8B-4bit`, 24 layers, `1024 × 3584` MLP projections, `652 MB` on disk) is loaded once and frozen (`model.freeze()`). Base weights are strictly read-only (`writeable=False` in the NumPy core; `base.freeze()` in `SubcorMLXLinear`). Every target linear projection $W_{\text{base}} \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}$ is wrapped so runtime inference computes:

$$y = x W_{\text{base}}^T + \sum_{u \in \mathcal{A}(x)} \Delta_u(x)$$

where $\mathcal{A}(x)$ is the set of Skill Units selected by the router for prompt $x$. For performance, mounted adapters pre-fold their scalar scale into $B$ at mount time and execute in the base model's native compute dtype (`bfloat16`/`float16`), eliminating per-token FP32 casts during decode.

### 3.2 Verified Evidence Gate

All runtime trajectories are appended to an immutable JSONL store (`TrajectoryStore` / `TextTrajectoryStore`). Each record carries a `VerificationEvidence(verified, verifier, method)` attestation. Live generations enter the store with `verified=False` until confirmed by either a deterministic verifier (schema check, unit test, exit code) or explicit senior-seat/owner review. Both contrastive activation extraction and distillation dataset export strictly exclude unverified records, preventing hallucinated student drafts from entering weight updates.

### 3.3 Dual-Mode Adaptation Engine

Empirical evidence shows that direct weight editing unblocks latent defects cleanly but cannot synthesize complex missing circuits from scratch. SUBCOR therefore separates adaptation into two distinct modes:

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

### 3.4 Sub-1-Bit Low-Rank Binary Factorization (`Sub1BitMatrix`)

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

### 3.5 Training-Free Frozen-Base Activation-Space MoE Router (`MoESkillRouter`)

To compose multiple Skill Units without cross-skill interference or general-knowledge forgetting, SUBCOR implements a **Shared Dense Core + Routed Skill Units** topology:

1. **Invariant Prompt Tap (Early-Exit Prefill):** For any user prompt $p$ (evaluated with `system_prompt=None` so routing is strictly semantic), SUBCOR bypasses all mounted adapters (`_bypass_adapters=True`) and executes layers $0 \dots \ell_{\text{tap}}$ only—stopping immediately after $\ell_{\text{tap}}$ and skipping downstream layers plus the 248,320-token `lm_head`. The mean output activation at $\ell_{\text{tap}}$ is L2-normalized to produce an embedding $e(p) \in \mathbb{S}^{d-1}$ that is **mathematically invariant** to which Skill Units are installed or enabled.
2. **Analytical Contrastive `RouterGate`:** Each Skill Unit stores a lightweight gate fitted analytically from verified positive prompt embeddings $E^+$ vs. verified negative/general embeddings $E^-$:
   $$\mu_+ = \operatorname{unit}(\overline{E^+}), \quad \mu_- = \operatorname{unit}(\overline{E^-}), \quad s(e) = \langle e, \mu_+ \rangle - \langle e, \mu_- \rangle$$
   The decision threshold $\tau$ is calibrated conservatively above $\max_{e \in E^-} s(e)$ (a 65/35 blend between $\max(s^-)$ and the 15th percentile of $s^+$ when separable).
3. **Zero-Interference Fallback:** At inference time, `MoESkillRouter.route(e, top_k=1)` activates only units with $s_u(e) \ge \tau_u$. When no unit clears its threshold, $\mathcal{A}(x) = \emptyset$ and inference runs on the pure frozen base kernel with **exact zero perturbation**.

### 3.6 Governed Skill Unit Lifecycle & Multi-Tenant Isolation

Every Skill Unit is packaged into a content-addressed directory (`manifest.json`, `payload.scun`, optional `router_gate.json`) governed by a strict finite state machine:

$$\texttt{DORMANT} \xrightarrow{\text{locked conformance pass}} \texttt{CONFORMANT} \xrightarrow{\text{approver} \neq \text{proposer}} \texttt{ENABLED} \xrightarrow{\text{atomic rollback}} \texttt{ROLLED\_BACK}$$

Four invariants are enforced in code:
- **Content Integrity:** Every load, promotion, and export verifies payload SHA-256 against `manifest.payload_sha256` (fail-closed `IntegrityError`).
- **Locked Conformance Gate:** Candidate units must improve target capability by $\ge \Delta_{\min}$ while keeping baseline regression within tolerance $\epsilon_{\max}$.
- **No Self-Approval:** `promote(unit_id, approver)` raises `GovernanceError` whenever `approver == manifest.proposer`.
- **Multi-Tenant Serving & Portable Export:** `SubcorHTTPServer` shares one frozen base kernel across $N$ isolated tenants (`subcor/<tenant_id>`), each with its own unit registry, trajectory store, and MoE router. Promoted units export to both **CaPheOS format-2 `module.json` bundles** (with runnable Node `conformance/test.js`) and stock **`mlx-lm` `adapters.safetensors`**.

---

## 4. Experimental Setup & Standardized Benchmark (`subcor-core-v1`)

To prevent goalpost drift across iterations, SUBCOR defines an immutable, content-hashed longitudinal benchmark suite **`subcor-core-v1@1.0.0`** (`sha256:1732c1fcad55`, 24 deterministic test cases across 5 functional axes):

| Axis | Cases | Role | Deterministic Grader |
|---|---:|---|---|
| `general_reasoning` | 6 | Locked non-regression anchor | `numeric`, `contains` |
| `instruction_discipline` | 4 | Locked non-regression anchor | `concise_contains` (word & token limits) |
| `structured_json` | 3 | General schema/tool adherence | `exact_json`, `json_keys` |
| `domain_ops_routing` | 5 | Acquired Skill 1 (unseen company ops events) | `exact_json` (`seat`, `gate`, `code`) |
| `domain_noc_triage` | 6 | Acquired Skill 2 (unseen NOC alerts) | `exact_json` (`sev`, `notify`, `sla_min`) |

Every run appends an immutable `BenchmarkSnapshot` to a JSONL `BenchmarkLedger` recording per-axis pass rates, locked anchor retention, token-limit loop failure rate, active Skill Units with SHA-256 digests, total adapter bytes, latency, and throughput.

**Hardware:** Apple M5 (arm64), 16 GB unified memory, macOS 26.5.1, Python 3.12.16, `mlx==0.32.4`, `mlx-lm==0.32.2`.

---

## 5. Empirical Results

### 5.1 Controlled Synthetic Proving Ground (`subcor demo`)

On a 2-layer iterative residual reasoning network (`dim=64`, `max_steps=12`) with an injected structural thinking loop in `layer_1` and an unlearned orthogonal Task Family B:

| Stage | Family A Clean | Family A Loop (`loop_fail`) | Family B Novel | Unit Footprint |
|---|---:|---:|---:|---|
| Frozen Base | `100.0%` | `0.0%` (`100.0%` loop) | `33.8%` | `0 B` |
| Cycle 1: Zero-Gradient Surgery | `100.0%` | **`100.0%`** (`0.0%` loop) | `35.0%` | `312 B` (`0.535 bpw`) |
| Cycle 2: Dynamic Expansion + Binary STE | `100.0%` | `100.0%` | **`100.0%`** | `366 B` (`0.641 bpw`) |
| After Atomic Rollback of Skill 2 | `100.0%` | `100.0%` | `35.0%` | `312 B` |

Zero-perturbation expansion measured `max|y_after - y_before| = 0.0e+00`; `12` unverified trajectories were filtered; self-approval (`tools → tools`) and a deliberately regressive patch were both blocked.

### 5.2 Live Apple M5 Longitudinal Results (`Qwen3.5-0.8B-4bit`)

#### A. Dense Mounting vs. Routed MoE Mounting
- **Unrouted dense mount (`subcor mlx-intern`):** `capheos-ops-router-v1` (`81.4 KiB`, `0.0302 bpw`) lifted unseen Ops Router accuracy from **`0.0% → 80.0%`** (`loss 1.7064 → 0.0168`, zero-perturbation `0.0e+00`, rollback logit diff `0.0e+00`), **but** caused General QA to dip **`83.3% → 66.7%`** due to dense activation interference.
- **Routed MoE layer (`subcor mlx-moe`):** Mounting BOTH `capheos-ops-router-v1` (`81.4 KiB`) and `capheos-incident-sla-v1` (`81.4 KiB`) behind `MoESkillRouter` achieved **`100.0%` unseen semantic routing accuracy** on both domains and eliminated General QA regression (**`83.3% → 83.3%`**).

#### B. Standardized `subcor-core-v1` Growth Progression (AC Power, Post-Optimization)

| Snapshot | Active Units | Skill Size | Overall | Macro Axis | Locked Anchors | Loop Fail | Avg Latency | Short-Turn TPS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `day0-frozen-base` | 0 | `0.0 KiB` | `41.7%` | `45.0%` | `80.0%` | `0.0%` | `171 ms` | `114.8 tok/s` |
| `v1-ops-router` | 1 | `81.4 KiB` | `54.2%` | `57.0%` | `80.0%` | `0.0%` | `156 ms` | `101.2 tok/s` |
| `v2-ops-plus-noc` | 2 | `162.8 KiB` | **`58.3%`** | **`60.3%`** | **`80.0%`** | **`0.0%`** | **`137 ms`** | **`86.5 tok/s`** |

- **Non-regression:** `general_reasoning` (`83.3%`), `instruction_discipline` (`75.0%`), and `structured_json` (`66.7%`) all changed by **`+0.0 pp`** across all three snapshots.
- **Sustained HTTP decode (AC Power, 256-token prompt):** **`154.1 tok/s`** over `http://127.0.0.1:8420/v1/chat/completions`.
- **Early-exit routing tap overhead:** **`18.3 ms/prompt`** (`layers.0..18`, bypassing layers `19..23` and the 248k-vocab `lm_head`).
- **Speedup from native-dtype pre-scaled adapters + early-exit tap:** Two-expert routed short-turn throughput rose from `49.0 tok/s` (`261 ms/case`, eco mode) to **`86.5 tok/s` (`137 ms/case`, AC power)**.

### 5.3 Negative Findings & Governance Validations

Reporting negative outcomes is central to evaluating self-modifying systems honestly:

1. **Single-Layer Live Weight Surgery (`subcor mlx-surgery`):** Contrastive activation analysis over `Qwen3.5-0.8B-4bit` layers `[7, 11, 15, 19, 22]` cleanly ranked `layers.11.mlp.down_proj` highest (`separation=6.3808`) and produced a `9.6 KiB` (`0.0213 bpw`) rank-1 patch without gradients. However, single-layer rank-1 ablation at a 160-token cap did not shorten verbose word-problem thinking traces (`100% → 100%` limit rate), confirming MIT Experiment 8's finding that real LLM thinking-loop suppression requires multi-layer, column-targeted edits over long-context loops. The locked Conformance Gate rejected promotion and held the unit `dormant`.
2. **Cold-Start Binary STE & Scale LR Sensitivity:** Training Binary STE from random signs on 12 examples exhibited high seed variance (`50.0%` at seed 42 vs. `16.7%` at seed 142), and warm-started STE with `ste_lr = 5e-4` destabilized normalized `Sub1Bit` scaling vectors (`loss 0.029 → 8.854`), which the locked Conformance Gate likewise blocked. Restricting `ste_lr <= 8e-5` or using FP32 warm-start followed by alternating `Sub1Bit` factorization restored monotonic convergence (`loss 2.808 → 0.0021`).
3. **Runaway Hyperparameter Catch:** An early `lr = 4e-3` trial caused baseline General QA collapse (`83.3% → 0.0%`); the locked Conformance Gate automatically rejected the unit and atomic rollback restored exact `0.0e+00` base logits.

---

## 6. Limitations & Safety Boundaries

1. **Base Kernel Intelligence Ceiling:** A `0.8B` model augmented with `81 KiB` Skill Units excels at repeatable structured tasks (routing, schemas, policies, triage) but cannot replace frontier models for ambiguous long-horizon reasoning. SUBCOR targets the *Intern* tier—graduating routine work locally while escalating hard cases.
2. **Low-Rank Compression Suits Skills, Not Databases:** Binary rank-8 factorization compresses procedural skills cleanly to `0.03 bpw`, but dense factual catalogs belong in external databases, files, and MCP tools rather than weights.
3. **Flat Cosine Routing at Scale:** Single-centroid gates separated 2 distinct domains at `100%`, but scaling to `50+` overlapping skills will require hierarchical routing (e.g., scoping candidate gates by organizational seat before scoring embeddings).
4. **Compiled-Artifact Coupling:** Each `weights.scun` payload is mathematically coupled to its base kernel's coordinate system. The **verified trajectory log** is the durable cross-generation "source code"; upgrading the base kernel recompiles Skill Units from verified logs in minutes.
5. **Single-GPU Study Contention:** On consumer hardware, QLoRA study jobs (`~45–90s`) saturate the GPU queue and should run during scheduled off-peak windows.

---

## 7. Reproducibility & Future Work

All experiments are reproducible from the repository root:

```bash
uv sync --extra dev --extra mlx --python 3.12
uv run pytest -v                              # 25 deterministic unit & integration tests
uv run subcor demo                           # Milestone 1 synthetic proving ground
uv run --extra mlx subcor mlx-intern         # Live single-skill QLoRA -> Sub-1-Bit
uv run --extra mlx subcor mlx-moe            # Live two-expert Routed MoE
uv run --extra mlx subcor mlx-surgery        # Live zero-gradient activation surgery
uv run --extra mlx subcor bench run --label day0-base
```

Immediate future work includes: (i) multi-layer column-masked live surgery over long-context thinking loops, (ii) hierarchical seat-scoped MoE routing tested across `20–50` accumulated company Skill Units, (iii) live multi-week token-cost and autonomy telemetry inside CaPheOS deployments, and (iv) cross-company portability trials exporting learned `81 KiB` Skill Units between independent companies.

---

## References

1. Takehi, R., Teoh, J., Zhang, A., & Khattab, O. (Nov 2026). *Sparks of Weight-Space Programming: LLMs can natively read & write neural network weights.* MIT CSAIL.
2. Chong, H., Kim, D., Kim, C., & Choi, M. (2026). *NanoQuant: Efficient Sub-1-Bit Quantization of Large Language Models.* ICML 2026 (`arXiv:2602.06694`).
3. *BTC-LLM: Efficient Sub-1-Bit LLM Quantization via Learnable Transformation and Binary Codebook.* ACL 2026 (`arXiv:2506.12040`).
4. Google DeepMind (2026). *Gemma 4 Quantization-Aware Training (QAT) Technical Documentation.*
5. Dettmers, T., Pagnoni, A., Holtzman, A., & Zettlemoyer, L. (2023). *QLoRA: Efficient Finetuning of Quantized LLMs.* NeurIPS 2023.
6. Arditi, A., Obeso, O., Syed, A., Paleka, D., Panickssery, N., Gurnee, W., & Nanda, N. (2024). *Refusal in Language Models Is Mediated by a Single Direction.*
7. Agarwal, R., et al. (2024). *On-Policy Distillation of Language Models: Learning from Self-Generated Mistakes.* ICLR 2024.
8. Chen, T., Goodfellow, I., & Shlens, J. (2015). *Net2Net: Accelerating Learning via Knowledge Transfer.*
