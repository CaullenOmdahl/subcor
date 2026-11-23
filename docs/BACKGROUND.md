# Background and inspiration

Where the idea for SUBCOR came from, the work it builds on, and what we think is new.
This is a design note, not a paper. The measured results live in the reports under
[`docs/`](.), and the behavior contracts in [`ARCHITECTURE.md`](ARCHITECTURE.md).

## The problem we started from

Most "self-improving" agent frameworks never touch model weights. They edit prompts,
memory stores or retrieval indices. Meanwhile the research on editing weights and on
extreme low-bit compression mostly studies a single static checkpoint, with no runtime
around it: no rollback, no audit trail, no protection against an update that quietly makes
the model worse at everything else.

We wanted a small local model that could take on the repetitive parts of real work
(routing, triage, tool-call formatting, policy checks) and get better at them from verified
experience, without ever risking the base model. That led to four requirements:

1. **Never mutate the base.** New behavior lives in separate, removable pieces.
2. **Don't let a new skill leak into everything else.** A skill should only be active when
   it is relevant.
3. **Make each skill small enough to store, ship and audit like a file.**
4. **Don't let the system approve its own changes.** Only verified evidence goes in, a fixed
   regression check gates every change, and a different party approves it.

## The analogy that shaped the design

The design follows a rough cortical/subcortical division of labor:

| Role | In the brain (loosely) | In SUBCOR |
|---|---|---|
| Working memory | Holding facts in mind | Context window, RAG, MCP tools. Facts are not trained into weights. |
| Deliberate planning | Prefrontal cortex | A frontier model handles novel or ambiguous cases. |
| Recognizing a familiar routine | Basal ganglia | `MoESkillRouter`: a training-free gate on frozen-base activations. |
| Executing a practiced skill | Cerebellum | A routed Skill Unit: a small low-rank weight delta. |
| Knowing when you're out of your depth | Thalamic gating | `TurnConfidence` flags low decode confidence or loops and hands back. |

In engineering terms, the frozen base model is an impact drill and each Skill Unit is a
drill bit. Compiled bits fit one drill (`base_kernel_id`). Each bit's `skill_spec.json` is
its blueprint, so when you upgrade the drill, `subcor recompile` rebuilds and re-verifies
every bit.

## Work this builds on

No single mechanism in SUBCOR is new. Each piece comes from published work:

**Low-rank adapters on a frozen, quantized base**
- QLoRA: Dettmers et al., NeurIPS 2023, [arXiv:2305.14314](https://arxiv.org/abs/2305.14314).
- Multi-tenant adapter serving over one shared base: Punica,
  [arXiv:2310.18547](https://arxiv.org/abs/2310.18547); S-LoRA,
  [arXiv:2311.03285](https://arxiv.org/abs/2311.03285); and
  [LoRAX](https://github.com/predibase/lorax).

**Routing among adapters**
- LoraHub, [arXiv:2307.13269](https://arxiv.org/abs/2307.13269).
- LoRAMoE, [arXiv:2312.09979](https://arxiv.org/abs/2312.09979).
- Arrow, from "Towards Modular LLMs by Building and Reusing a Library of LoRAs",
  [arXiv:2405.11157](https://arxiv.org/abs/2405.11157).
- PHATGOOSE, [arXiv:2402.05859](https://arxiv.org/abs/2402.05859).
- Where SUBCOR differs: instead of training a router network, which collapses on 10–20
  examples, it fits each gate analytically from frozen-base activations. When no gate
  matches, it falls back to the unmodified base.

**Editing behavior directly in weight space**
- Representation Engineering, Zou et al. 2023,
  [arXiv:2310.01405](https://arxiv.org/abs/2310.01405).
- "Refusal in Language Models Is Mediated by a Single Direction", Arditi et al. 2024,
  [arXiv:2406.11717](https://arxiv.org/abs/2406.11717).
- "Sparks of Weight-Space Programming", Takehi, Teoh, Zhang and Khattab, a
  [blog post](https://rikiyat.github.io/blog/weight-space-programming) from November 2026
  (not peer reviewed). Its thinking-loop experiments prompted SUBCOR's zero-gradient surgery
  mode.

**Sub-1-bit compression**
- NanoQuant, ICML 2026, [arXiv:2602.06694](https://arxiv.org/abs/2602.06694).
- LittleBit, NeurIPS 2025, [arXiv:2506.13771](https://arxiv.org/abs/2506.13771).
- BTC-LLM, ACL 2026,
  [ACL Anthology](https://aclanthology.org/2026.acl-long.1066/).
- BitDelta, [arXiv:2402.10193](https://arxiv.org/abs/2402.10193), which showed that
  fine-tune deltas compress well.
- SUBCOR applies bit-packed binary low-rank factorization to *skill deltas* rather than to
  whole models.

**Quantization-aware training**
- Gemma 4 QAT,
  [Google blog](https://blog.google/innovation-and-ai/technology/developers-tools/quantization-aware-training-gemma-4/).
- SUBCOR's opt-in binary straight-through estimator follows the same idea.

**Learning from your own outputs**
- On-policy distillation (GKD): Agarwal et al., ICLR 2024,
  [arXiv:2306.13649](https://arxiv.org/abs/2306.13649).
- STaR, [arXiv:2203.14465](https://arxiv.org/abs/2203.14465).
- ReST-EM, [arXiv:2312.06585](https://arxiv.org/abs/2312.06585).
- SUBCOR's BFCL results come from this pattern: train only on verified corrections of the
  model's own failures, mixed with replays of its own passing outputs.

**Growing a network without changing its function**
- Net2Net, ICLR 2016, [arXiv:1511.05641](https://arxiv.org/abs/1511.05641).
- SUBCOR's zero-initialized expansion branches start from the same invariant: the
  network's output is exactly unchanged before training begins.

## Closest related systems

Models that keep learning after deployment already exist:

- [Cursor Tab online RL](https://cursor.com/blog/tab-rl).
- Gboard's federated learning, Hard et al. 2018,
  [arXiv:1811.03604](https://arxiv.org/abs/1811.03604).
- SEAL, [arXiv:2506.10943](https://arxiv.org/abs/2506.10943).
- A trigger-guided LoRA self-adaptation framework (SAGE),
  [arXiv:2509.05385](https://arxiv.org/abs/2509.05385).

The two closest designs we know of:

- **Macaron-V1** ([arXiv:2608.09819](https://arxiv.org/abs/2608.09819), a technical report)
  pairs a frozen base with LoRA specialists chosen per turn, plus a versioned registry.
- **"Surprise Alone Learns Lies: Evidence-Gated Continual Learning for LLMs from Streaming
  Claims"** ([arXiv:2610.39840](https://arxiv.org/abs/2610.39840), a preprint) gates
  continual updates on evidence.

## What we think is new

The combination, not any one part:

- Updates are learned only from **verified** deployment trajectories.
- **Per-skill** adapters are created automatically.
- A **locked, owner-controlled** regression gate checks every update.
- The **approver is never the proposer**.
- A **frozen-base router** falls back exactly to the base model.
- Rollback is **per unit**.
- Units are **portable** and can be recompiled onto a new base.
- Units are packed to **sub-1-bit**.

That is a claim about integration. Our confidence that no one has published this exact
combination is moderate, since it is based on public sources only. We also do not yet
claim the system is "self-improving" in the strong sense: we have not shown gains
compounding over successive generations of units.

## What didn't work

Reporting failures matters for a system that changes its own weights. Each of these was
caught by the conformance gate:

- **Single-layer rank-1 surgery doesn't stop real thinking loops.** On `Qwen3.5-0.8B`,
  activation analysis correctly picked out `layers.11.mlp.down_proj`, and the patch was only
  9.6 KiB. It still didn't shorten verbose word-problem traces: the token-limit rate stayed
  at 100%. Real loop suppression needs edits across several layers. The gate kept the unit
  dormant.
- **Cold-start binary STE is seed-sensitive on tiny batches.** On 12 examples, seed 42 scored
  50% and seed 142 scored 16.7%. A high STE learning rate (`5e-4`) also destabilized the
  scaling vectors. Warm-starting in FP32 and then factorizing fixed both.
- **A runaway learning rate is caught.** An early `lr = 4e-3` run collapsed general QA from
  83.3% to 0%. The gate rejected the unit, and rollback restored the exact base logits.
- **Naive SFT on synthetic ground truth overfits.** On BFCL with `Qwen3.5-4B` it dropped
  held-out accuracy from 81.7% to 78.3%. On-policy self-distillation, as described above,
  fixed it.

## Limitations

- **A small base has a ceiling.** Skill Units help with repeatable, structured work. They do
  not turn a 0.8B model into a frontier model for open-ended reasoning, which is why
  low-confidence turns are handed back.
- **Skills belong in weights, facts do not.** Binary low-rank deltas compress procedures
  well. Catalogs of facts belong in databases, files and tools.
- **Routing at scale is still unproven.** Single-centroid gates separate a few domains
  cleanly. Scoping by seat helps, but 50 or more overlapping skills still need testing.
- **Compiled units are tied to one base.** A unit's weights only make sense on the base it
  was trained on. The durable artifacts are `skill_spec.json` and the verified trajectories.
- **Training competes for the GPU.** On a single consumer GPU, study jobs compete with
  serving, so schedule them off-peak.
