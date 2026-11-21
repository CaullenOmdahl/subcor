"""Controlled baseline comparisons for SUBCOR Skill Units (Issue #12).

Provides reproducible, apples-to-apples comparisons answering the core empirical questions:
  1. Compression Fidelity at Equal Byte Budget:
     Sub-1-Bit (`Sub1BitMatrix`, rank-k) vs. same-byte-budget FP16 low-rank SVD
     (rank-1 / rank-2) and BitDelta (1-bit sign + global scalar) on *realistic*
     real-valued low-rank weight deltas (not just synthetic ±1 matrices).
  2. Broad Router FP/FN Sweep:
     Measures false-positive and false-negative routing rates on broad out-of-domain
     prompt distributions as skill count scales (1 -> 5 -> 20 skills), comparing
     single-centroid vs. multi-prototype and seat-scoped `MoESkillRouter`.
  3. Live MLX Procedural Workflow Head-to-Head (`run_mlx_baseline_comparison`):
     Evaluates Frozen Base, Few-Shot Prompting, Frozen-Base kNN Embedding Classifier,
     Same-Byte FP16 LoRA (rank-1 & rank-2), and SUBCOR Rank-8 Sub-1-Bit Skill Unit
     on a 100-case strictly held-out procedural workflow suite.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from subcor.router import MoESkillRouter, RouterGate, _l2_normalize
from subcor.sub1bit import factorize_sub1bit, relative_frobenius_error


@dataclass(frozen=True)
class CompressionBaselineRow:
    """Single compression method measured on a realistic low-rank delta matrix."""

    method: str
    payload_bytes: int
    bits_per_weight: float
    relative_frobenius_error: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "payload_bytes": self.payload_bytes,
            "bits_per_weight": round(self.bits_per_weight, 5),
            "relative_frobenius_error": round(self.relative_frobenius_error, 4),
        }


def compare_compression_at_equal_bytes(
    *,
    d_out: int = 1024,
    d_in: int = 3584,
    true_rank: int = 8,
    scale: float = 1e-3,
    seed: int = 42,
) -> list[CompressionBaselineRow]:
    """Compare Sub-1-Bit vs same-byte-budget FP16 SVD & BitDelta on realistic real-valued deltas."""
    rng = np.random.default_rng(seed)
    # Realistic real-valued rank-8 LoRA delta with decaying singular spectrum.
    b_true = rng.normal(0.0, 1.0, size=(d_out, true_rank))
    a_true = rng.normal(0.0, 1.0, size=(true_rank, d_in))
    decay = np.geomspace(1.0, 0.2, num=true_rank)
    w = (b_true * decay[np.newaxis, :]) @ a_true
    w = w * (scale / (float(np.linalg.norm(w)) / np.sqrt(d_out * d_in) + 1e-12))

    w_norm = float(np.linalg.norm(w))
    n_weights = d_out * d_in
    rows: list[CompressionBaselineRow] = []

    # 1. SUBCOR Sub-1-Bit rank-8 factorization
    sub1 = factorize_sub1bit(w, rank=true_rank)
    rows.append(
        CompressionBaselineRow(
            method=f"sub1bit_rank_{true_rank}",
            payload_bytes=len(sub1.to_bytes()),
            bits_per_weight=sub1.bits_per_weight,
            relative_frobenius_error=relative_frobenius_error(w, sub1),
        )
    )

    # 2. Exact low-rank FP16 SVD at rank 1, 2, 4, and true_rank (same-byte & full-rank references)
    u_svd, s_svd, vt_svd = np.linalg.svd(w, full_matrices=False)
    for r in (1, 2, 4, true_rank):
        if r > min(d_out, d_in):
            continue
        b_r = (u_svd[:, :r] * np.sqrt(s_svd[:r])[np.newaxis, :]).astype(np.float16)
        a_r = (np.sqrt(s_svd[:r])[:, np.newaxis] * vt_svd[:r, :]).astype(np.float16)
        w_hat = b_r.astype(np.float64) @ a_r.astype(np.float64)
        err = float(np.linalg.norm(w - w_hat) / (w_norm + 1e-12))
        payload_bytes = b_r.nbytes + a_r.nbytes
        rows.append(
            CompressionBaselineRow(
                method=f"fp16_lora_rank_{r}",
                payload_bytes=payload_bytes,
                bits_per_weight=(payload_bytes * 8.0) / n_weights,
                relative_frobenius_error=err,
            )
        )

    # 3. BitDelta-style dense 1-bit sign + global FP16 scale
    alpha = float(np.mean(np.abs(w)))
    w_bd = np.where(w >= 0.0, alpha, -alpha)
    bd_bytes = (n_weights + 7) // 8 + 2
    rows.append(
        CompressionBaselineRow(
            method="bitdelta_dense_1bit",
            payload_bytes=bd_bytes,
            bits_per_weight=(bd_bytes * 8.0) / n_weights,
            relative_frobenius_error=float(np.linalg.norm(w - w_bd) / (w_norm + 1e-12)),
        )
    )

    return rows


@dataclass(frozen=True)
class RouterScaleSweepRow:
    """Router FP/FN telemetry as skill library scales from 1 to N skills."""

    num_skills: int
    seat_scoped: bool
    num_prototypes: int
    false_positive_rate: float
    false_negative_rate: float
    exact_skill_accuracy: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "num_skills": self.num_skills,
            "seat_scoped": self.seat_scoped,
            "num_prototypes": self.num_prototypes,
            "false_positive_rate": round(self.false_positive_rate, 4),
            "false_negative_rate": round(self.false_negative_rate, 4),
            "exact_skill_accuracy": round(self.exact_skill_accuracy, 4),
        }


def run_router_scale_sweep(
    *,
    skill_counts: tuple[int, ...] = (1, 5, 10, 20),
    dim: int = 256,
    n_train_per_skill: int = 12,
    n_test_per_skill: int = 20,
    n_ood_general: int = 200,
    num_seats: int = 4,
    seed: int = 42,
) -> list[RouterScaleSweepRow]:
    """Measure broad-distribution router FP/FN rates as skill count grows (flat vs seat-scoped)."""
    rng = np.random.default_rng(seed)
    max_skills = max(skill_counts)

    # Generate semi-overlapping skill clusters + broad isotropic OOD general prompts.
    skill_centers = _l2_normalize(rng.normal(0.0, 1.0, size=(max_skills, dim)))
    ood_calib = _l2_normalize(rng.normal(0.0, 1.0, size=(40, dim)))
    ood_test = _l2_normalize(rng.normal(0.0, 1.0, size=(n_ood_general, dim)))

    train_by_skill = [
        _l2_normalize(skill_centers[i] + rng.normal(0.0, 0.12, size=(n_train_per_skill, dim)))
        for i in range(max_skills)
    ]
    test_by_skill = [
        _l2_normalize(skill_centers[i] + rng.normal(0.0, 0.12, size=(n_test_per_skill, dim)))
        for i in range(max_skills)
    ]

    results: list[RouterScaleSweepRow] = []
    for n_skills in skill_counts:
        for seat_scoped in (False, True):
            router = MoESkillRouter(tap_layer="layers.18.mlp.down_proj")
            for i in range(n_skills):
                seat_name = f"seat-{i % num_seats}" if seat_scoped else None
                # Include other skills' training embeddings as negatives alongside OOD calibration.
                other_neg = [train_by_skill[j] for j in range(n_skills) if j != i]
                neg_pool = np.vstack([ood_calib, *other_neg]) if other_neg else ood_calib
                gate = RouterGate.fit(
                    unit_id=f"skill-{i:02d}",
                    tap_layer="layers.18.mlp.down_proj",
                    pos_embeddings=train_by_skill[i],
                    neg_embeddings=neg_pool,
                    seat_scope=seat_name,
                    num_prototypes=1,
                )
                router.register(gate)

            # Broad OOD False Positive Rate
            fp = 0
            for idx in range(n_ood_general):
                seat_q = f"seat-{idx % num_seats}" if seat_scoped else None
                dec = router.route(ood_test[idx], top_k=1, seat=seat_q)
                if not dec.fell_back_to_base:
                    fp += 1

            # Held-out In-Domain False Negative Rate & Exact Skill Accuracy
            fn = 0
            exact = 0
            total_id = n_skills * n_test_per_skill
            for i in range(n_skills):
                expected_uid = f"skill-{i:02d}"
                seat_q = f"seat-{i % num_seats}" if seat_scoped else None
                for vec in test_by_skill[i]:
                    dec = router.route(vec, top_k=1, seat=seat_q)
                    if dec.fell_back_to_base:
                        fn += 1
                    elif dec.selected_units[0] == expected_uid:
                        exact += 1

            results.append(
                RouterScaleSweepRow(
                    num_skills=n_skills,
                    seat_scoped=seat_scoped,
                    num_prototypes=1,
                    false_positive_rate=fp / n_ood_general,
                    false_negative_rate=fn / total_id,
                    exact_skill_accuracy=exact / total_id,
                )
            )
    return results


class FrozenEmbeddingKNNBaseline:
    """Simple cosine kNN classifier on frozen-base embeddings (baseline for pure routing tasks)."""

    def __init__(self, *, k: int = 3) -> None:
        self.k = k
        self._embs: np.ndarray | None = None
        self._labels: list[str] = []

    def fit(self, embeddings: np.ndarray, labels: list[str]) -> None:
        self._embs = _l2_normalize(np.asarray(embeddings, dtype=np.float64))
        self._labels = list(labels)

    def predict(self, embedding: np.ndarray) -> str:
        if self._embs is None or not self._labels:
            raise ValueError("FrozenEmbeddingKNNBaseline must be fitted before predict()")
        e = _l2_normalize(embedding)
        sims = self._embs @ e
        top_idx = np.argsort(sims)[::-1][: min(self.k, len(self._labels))]
        counts: dict[str, float] = {}
        for idx in top_idx:
            lbl = self._labels[int(idx)]
            counts[lbl] = counts.get(lbl, 0.0) + max(0.0, float(sims[int(idx)]))
        return max(counts.items(), key=lambda kv: kv[1])[0]


def generate_heldout_procedural_suite(
    *,
    n_train: int = 20,
    n_test: int = 100,
) -> tuple[list[tuple[str, dict[str, str]]], list[tuple[str, dict[str, str]]]]:
    """Generate a deterministic 100-case strictly held-out procedural workflow routing suite.

    Covers 5 organizational decision workflows (finance, security/risk, executive,
    people/HR, platform/tools) with parameterized synthetic templates and zero
    train/test overlap so accuracy reflects generalization rather than memorization.
    """
    specs: list[tuple[str, dict[str, str], list[str]]] = [
        (
            "finance",
            {"seat": "finance", "gate": "owner", "code": "FIN-CAP"},
            [
                "Workflow event: Cloud vendor invoice #{id} for ${amt} exceeds the {dept} monthly spending cap.",
                "Workflow event: Quarterly SaaS renewal #{id} (${amt}) breached the {dept} approved budget.",
                "Workflow event: Automated billing alert #{id}: token spend hit ${amt} hard stop in {dept}.",
            ],
        ),
        (
            "risk",
            {"seat": "risk", "gate": "owner", "code": "SEC-LOCK"},
            [
                "Workflow event: Security audit #{id} detected unrotated API secret in {dept} production.",
                "Workflow event: Guardrail alert #{id}: sandbox attempted unauthorized payment origin from {dept}.",
                "Workflow event: Compliance scanner #{id} flagged missing TLS pinning on {dept} gateway.",
            ],
        ),
        (
            "ceo",
            {"seat": "ceo", "gate": "owner", "code": "EXEC-CHARTER"},
            [
                "Workflow event: Enterprise partner #{id} requested exclusive multi-year {dept} distribution terms.",
                "Workflow event: Board briefing #{id}: public roadmap commitment requested for {dept} launch.",
                "Workflow event: Acquisition inquiry #{id} arrived regarding {dept} intellectual property.",
            ],
        ),
        (
            "people",
            {"seat": "people", "gate": "auto", "code": "HR-ROLE"},
            [
                "Workflow event: Onboarding ticket #{id}: draft seat charter for new {dept} analyst hire.",
                "Workflow event: Recruiting packet #{id}: summarize panel feedback for {dept} lead finalists.",
                "Workflow event: Role review #{id}: update quarterly competency rubric for {dept} interns.",
            ],
        ),
        (
            "tools",
            {"seat": "tools", "gate": "auto", "code": "OPS-UNIT"},
            [
                "Workflow event: Runtime ticket #{id}: SSE adapter dropped callback frame in {dept} worker.",
                "Workflow event: Package update #{id}: verify SHA-256 digest for {dept} MCP tool module.",
                "Workflow event: CI pipeline #{id}: container build cache expired for {dept} runner.",
            ],
        ),
    ]
    depts_train = ["core", "edge", "billing", "infra"]
    depts_test = [
        "analytics",
        "payments",
        "search",
        "mobile",
        "ledger",
        "identity",
        "telemetry",
        "support",
        "dispatch",
        "storage",
    ]

    train: list[tuple[str, dict[str, str]]] = []
    for idx in range(n_train):
        _, label, tmpls = specs[idx % len(specs)]
        tmpl = tmpls[idx % len(tmpls)]
        dept = depts_train[idx % len(depts_train)]
        prompt = tmpl.format(id=100 + idx, amt=1500 + idx * 250, dept=dept)
        train.append((prompt, dict(label)))

    test: list[tuple[str, dict[str, str]]] = []
    for idx in range(n_test):
        _, label, tmpls = specs[idx % len(specs)]
        tmpl = tmpls[(idx // len(specs)) % len(tmpls)]
        dept = depts_test[idx % len(depts_test)]
        prompt = tmpl.format(id=5000 + idx, amt=4100 + idx * 130, dept=dept)
        test.append((prompt, dict(label)))

    return train, test


def run_mlx_baseline_comparison(
    *,
    model_id: str = "mlx-community/Qwen3.5-0.8B-4bit",
    n_train: int = 20,
    n_test: int = 100,
    steps: int = 80,
    lr: float = 3e-4,
    seed: int = 42,
) -> dict[str, Any]:
    """Run controlled live MLX baselines on a 100-case strictly held-out procedural suite.

    Compares:
      1. Frozen base (zero-shot)
      2. 5-shot in-context prompting (measures accuracy AND prompt-token/latency overhead)
      3. Frozen-base embedding kNN classifier on tap_layer
      4. Same-byte-budget FP16 Rank-1 LoRA (~54 KiB across 6 layers)
      5. Same-byte-budget FP16 Rank-2 LoRA (~108 KiB across 6 layers)
      6. SUBCOR Rank-8 Sub-1-Bit Skill Unit (~81.4 KiB across 6 layers)
    """
    import json
    import time

    from subcor.backends.mlx_kernel import MLXKernel
    from subcor.benchmark import BenchmarkCase, grade_response
    from subcor.types import DeltaEncoding
    from subcor.units import build_layer_delta, serialize_layer_deltas

    train_cases, test_cases = generate_heldout_procedural_suite(n_train=n_train, n_test=n_test)
    sys_prompt = (
        "You are an organizational workflow router. Reply ONLY with compact JSON "
        'containing keys "seat", "gate", and "code". No prose, no markdown.'
    )

    kernel = MLXKernel(model_id=model_id, target_layer_indices=[18, 19, 20, 21, 22, 23])
    layer_names = sorted(
        kernel.instrumented.keys(),
        key=lambda n: (
            int(n.split(".")[1]) if len(n.split(".")) > 1 and n.split(".")[1].isdigit() else 10**9
        ),
    )
    tap_layer = layer_names[0]

    def eval_prompts(
        cases: list[tuple[str, dict[str, str]]],
        *,
        sys_p: str,
    ) -> dict[str, Any]:
        hits = 0
        prompt_toks = 0
        t0 = time.perf_counter()
        for idx, (prompt, expected) in enumerate(cases):
            tr = kernel.generate_with_trace(
                prompt,
                system_prompt=sys_p,
                max_tokens=40,
                enable_thinking=False,
                capture_activations=False,
            )
            prompt_toks += tr.prompt_tokens
            c = BenchmarkCase(f"t-{idx}", "procedural", prompt, "exact_json", expected)
            if grade_response(c, tr.response, hit_token_limit=tr.hit_token_limit):
                hits += 1
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        return {
            "accuracy": round(hits / max(1, len(cases)), 4),
            "passed": hits,
            "total": len(cases),
            "mean_prompt_tokens": round(prompt_toks / max(1, len(cases)), 1),
            "mean_latency_ms": round(elapsed_ms / max(1, len(cases)), 1),
        }

    # 1. Zero-shot Frozen Base
    base_res = eval_prompts(test_cases, sys_p=sys_prompt)

    # 2. 5-shot In-Context Prompting (1 example per class from train_cases)
    few_shot_lines = [sys_prompt, "", "Examples:"]
    for p, lbl in train_cases[:5]:
        few_shot_lines.append(f"Input: {p}")
        few_shot_lines.append(f"Output: {json.dumps(lbl, separators=(',', ':'))}")
    few_shot_sys = "\n".join(few_shot_lines)
    few_shot_res = eval_prompts(test_cases, sys_p=few_shot_sys)

    # 3. Frozen-base Embedding kNN Classifier on tap_layer
    t_knn0 = time.perf_counter()
    train_embs = np.stack(
        [kernel.encode_prompt_embedding(p, tap_layer=tap_layer) for p, _ in train_cases]
    )
    train_labels = [
        json.dumps(lbl, separators=(",", ":"), sort_keys=True) for _, lbl in train_cases
    ]
    knn = FrozenEmbeddingKNNBaseline(k=3)
    knn.fit(train_embs, train_labels)
    knn_hits = 0
    for p, expected in test_cases:
        emb = kernel.encode_prompt_embedding(p, tap_layer=tap_layer)
        pred = knn.predict(emb)
        if pred == json.dumps(expected, separators=(",", ":"), sort_keys=True):
            knn_hits += 1
    knn_ms = (time.perf_counter() - t_knn0) * 1000.0
    knn_res = {
        "accuracy": round(knn_hits / len(test_cases), 4),
        "passed": knn_hits,
        "total": len(test_cases),
        "payload_bytes": int(train_embs.astype(np.float16).nbytes),
        "mean_latency_ms": round(knn_ms / len(test_cases), 1),
    }

    # Shared training pairs
    train_examples = [
        (sys_prompt, p, json.dumps(lbl, separators=(",", ":"))) for p, lbl in train_cases
    ]

    def run_adapter_variant(
        *,
        train_rank: int,
        compress_sub1bit: bool,
        sub1bit_rank: int = 8,
    ) -> dict[str, Any]:
        adapters, losses, _ = kernel.train_qlora_adapters(
            train_examples,
            layer_names=layer_names,
            rank=train_rank,
            steps=steps,
            lr=lr,
            seed=seed,
            binary_ste=False,
        )
        if compress_sub1bit:
            deltas = [
                build_layer_delta(
                    name,
                    adapters[name],
                    encoding=DeltaEncoding.SUB1BIT_BINARY,
                    sub1bit_rank=sub1bit_rank,
                )
                for name in layer_names
            ]
            payload_bytes = len(serialize_layer_deltas(deltas))
            eval_adapters = [d.to_adapter(adapter_name=f"cmp:{d.layer_name}") for d in deltas]
            for d, adp in zip(deltas, eval_adapters, strict=True):
                kernel.instrumented[d.layer_name].mount_adapter(adp.name, adp)
        else:
            # Exact FP16 LoRA byte budget (A + B stored in float16)
            payload_bytes = sum(
                adapters[n].a.astype(np.float16).nbytes + adapters[n].b.astype(np.float16).nbytes
                for n in layer_names
            )
            for name in layer_names:
                adp = adapters[name]
                # Quantize to FP16 to evaluate actual stored FP16 precision
                from subcor.expansion import LowRankAdapter

                adp_fp16 = LowRankAdapter(
                    name=f"fp16:{name}",
                    a=adp.a.astype(np.float16).astype(np.float64),
                    b=adp.b.astype(np.float16).astype(np.float64),
                    scale=adp.scale,
                )
                kernel.instrumented[name].mount_adapter(adp_fp16.name, adp_fp16)

        try:
            res = eval_prompts(test_cases, sys_p=sys_prompt)
        finally:
            for name in layer_names:
                for mname in list(kernel.instrumented[name].mounted_names):
                    kernel.instrumented[name].unmount_adapter(mname)

        return {
            **res,
            "payload_bytes": payload_bytes,
            "final_loss": round(float(np.mean(losses[-5:])), 4),
        }

    fp16_r1 = run_adapter_variant(train_rank=1, compress_sub1bit=False)
    fp16_r2 = run_adapter_variant(train_rank=2, compress_sub1bit=False)
    sub1_r8 = run_adapter_variant(train_rank=8, compress_sub1bit=True, sub1bit_rank=8)

    return {
        "model_id": model_id,
        "n_train": n_train,
        "n_test": n_test,
        "compression_matrix_audit": [
            r.to_dict() for r in compare_compression_at_equal_bytes(seed=seed)
        ],
        "router_scale_sweep": [r.to_dict() for r in run_router_scale_sweep(seed=seed)],
        "live_procedural_baselines": {
            "frozen_base_zero_shot": base_res,
            "few_shot_5": few_shot_res,
            "frozen_embedding_knn": knn_res,
            "fp16_lora_rank1": fp16_r1,
            "fp16_lora_rank2": fp16_r2,
            "subcor_sub1bit_rank8": sub1_r8,
        },
    }
