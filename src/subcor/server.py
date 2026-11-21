"""Multi-tenant OpenAI-compatible & Governed Control Plane HTTP server for SUBCOR.

Serves ONE frozen compact base kernel (e.g., Qwen3.5-0.8B-4bit, 652 MB) to N
isolated CaPheOS company instances simultaneously:

  - Every tenant (<tenant_id>) has strictly isolated:
      * SkillUnitRegistry  (<state_dir>/tenants/<tenant_id>/units/)
      * TextTrajectoryStore (<state_dir>/tenants/<tenant_id>/telemetry/trajectories.jsonl)
      * MoESkillRouter      (loaded only from that tenant's ENABLED Skill Units)
  - OpenAI-compatible endpoints (works directly as a CaPheOS provider):
      * GET  /healthz, /readyz
      * GET  /v1/models
      * POST /v1/chat/completions  (streaming SSE & non-streaming JSON)
  - Governed Model-Ops Control Plane (for CaPheOS tools/risk/people seats):
      * POST /v1/subcor/tenants/{tenant}/trajectories
      * POST /v1/subcor/tenants/{tenant}/study
      * GET  /v1/subcor/tenants/{tenant}/units
      * POST /v1/subcor/tenants/{tenant}/units/{unit_id}/promote
      * POST /v1/subcor/tenants/{tenant}/units/{unit_id}/rollback
      * POST /v1/subcor/tenants/{tenant}/units/{unit_id}/export-capheos

Uses Python stdlib ThreadingHTTPServer (zero extra dependencies) with a kernel
lock so concurrent requests from multiple CaPheOS instances serialize safely on
Apple Silicon MLX without Metal queue races.
"""

from __future__ import annotations

import hmac
import json
import re
import threading
import time
import uuid
import warnings
from collections.abc import Iterator
from dataclasses import dataclass, field, replace
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Protocol
from urllib.parse import urlparse

import numpy as np

from subcor.confidence import (
    ConfidencePolicy,
    TurnConfidence,
    evaluate_turn_confidence,
)
from subcor.expansion import LowRankAdapter
from subcor.intern import TextTrajectory, TextTrajectoryStore, ngram_repetition_ratio
from subcor.router import MoESkillRouter, RouterGate, RoutingDecision
from subcor.types import (
    AdaptationMode,
    DeltaEncoding,
    GovernanceError,
    IntegrityError,
    SkillSpec,
    SkillUnitManifest,
    UnitState,
    VerificationError,
    VerificationEvidence,
)
from subcor.units import LayerDeltaPayload, SkillUnitRegistry, build_layer_delta

_TENANT_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,47}$")
_LAYER_IDX_RE = re.compile(r"layers\.(\d+)\.")

# Strictly disjoint default sets so RouterGate negative calibration never overlaps
# with the locked baseline anchor questions used to score non-regression.
DEFAULT_BASELINE_ANCHOR_CASES: tuple[tuple[str, str], ...] = (
    ("What is 12 * 12? Reply with only the number.", "144"),
    ("What is the capital of Japan? Reply with only the city name.", "Tokyo"),
    ("What is 15 + 27? Reply with only the number.", "42"),
    ("What planet is closest to the Sun? Reply with only the planet name.", "Mercury"),
    ("What is 100 - 37? Reply with only the number.", "63"),
    ("Name the chemical symbol for gold in at most 4 words.", "Au"),
)

DEFAULT_NEGATIVE_CALIBRATION_PROMPTS: tuple[str, ...] = (
    "Explain briefly how photosynthesis converts sunlight into energy.",
    "Summarize the difference between HTTP GET and POST requests.",
    "What is the boiling point of water at sea level in Celsius?",
    "Give a one-sentence definition of recursion in programming.",
    "List three primary colors used in subtractive color mixing.",
    "How many minutes are in three and a half hours?",
)


def _layer_sort_key(layer_name: str) -> tuple[int, str]:
    """Sort layer names by numeric transformer layer index (e.g. layers.7 before layers.11)."""
    m = _LAYER_IDX_RE.search(layer_name)
    return (int(m.group(1)) if m else 10**9, layer_name)


def validate_tenant_id(tenant_id: str) -> str:
    if not _TENANT_RE.match(tenant_id):
        raise ValueError(
            f"Invalid tenant_id {tenant_id!r}: must match ^[a-z0-9][a-z0-9_-]{{0,47}}$"
        )
    return tenant_id


@dataclass(frozen=True)
class InferenceResult:
    """Backend-agnostic result of one routed or base chat completion."""

    text: str
    prompt_tokens: int
    completion_tokens: int
    hit_token_limit: bool
    repetition_ratio: float
    routing: RoutingDecision
    tool_calls: tuple[dict[str, Any], ...] = ()
    token_logprobs: tuple[float, ...] = ()
    confidence: TurnConfidence | None = None


class ServingKernel(Protocol):
    """Minimal kernel interface required by SubcorServer (MLX or test backend)."""

    model_id: str
    tap_layer: str
    target_layers: tuple[str, ...]

    def generate_routed_chat(
        self,
        *,
        user_prompt: str,
        system_prompt: str | None,
        router: MoESkillRouter | None,
        registry: SkillUnitRegistry | None,
        max_tokens: int,
        enable_thinking: bool,
        top_k: int,
    ) -> InferenceResult: ...

    def encode_prompt_embedding(self, user_prompt: str) -> np.ndarray: ...

    def train_skill_deltas(
        self,
        *,
        examples: list[tuple[str | None, str, str]],
        rank: int,
        sub1bit_rank: int,
        steps: int,
        lr: float,
        seed: int,
    ) -> tuple[list[LayerDeltaPayload], float, float, float]:
        """Train and sub-1-bit compress deltas.

        Returns (deltas, initial_loss, final_loss, zero_perturbation_max_diff).
        """
        ...

    def invalidate_unit(self, unit_id: str) -> None: ...


class MLXServingAdapter:
    """Wraps MLXKernel behind the ServingKernel protocol."""

    def __init__(
        self,
        model_id: str = "mlx-community/Qwen3.5-0.8B-4bit",
        *,
        target_layer_indices: list[int] | None = None,
    ) -> None:
        from subcor.backends.mlx_kernel import MLXKernel

        indices = (
            target_layer_indices if target_layer_indices is not None else [18, 19, 20, 21, 22, 23]
        )
        self._kernel = MLXKernel(model_id=model_id, target_layer_indices=indices)
        self.model_id = model_id
        self.target_layers = tuple(sorted(self._kernel.instrumented.keys(), key=_layer_sort_key))
        self.tap_layer = self.target_layers[0]

    def generate_routed_messages(
        self,
        *,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        router: MoESkillRouter | None,
        registry: SkillUnitRegistry | None,
        max_tokens: int,
        enable_thinking: bool,
        top_k: int,
    ) -> InferenceResult:
        if router is not None and registry is not None and router.registered_units:
            trace, decision = self._kernel.generate_routed_messages(
                messages,
                router=router,
                registry=registry,
                tools=tools,
                max_tokens=max_tokens,
                enable_thinking=enable_thinking,
                top_k=top_k,
            )
        else:
            trace = self._kernel.generate_messages_with_trace(
                messages,
                tools=tools,
                max_tokens=max_tokens,
                enable_thinking=enable_thinking,
            )
            decision = RoutingDecision(selected_units=(), scores={}, thresholds={})

        return InferenceResult(
            text=trace.response,
            prompt_tokens=trace.prompt_tokens,
            completion_tokens=trace.tokens_generated,
            hit_token_limit=trace.hit_token_limit,
            repetition_ratio=trace.repetition_ratio,
            routing=decision,
            tool_calls=trace.tool_calls,
            token_logprobs=trace.token_logprobs,
        )

    def stream_routed_messages(
        self,
        *,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        router: MoESkillRouter | None,
        registry: SkillUnitRegistry | None,
        max_tokens: int,
        enable_thinking: bool,
        top_k: int,
    ) -> tuple[RoutingDecision, Iterator[Any]]:
        return self._kernel.stream_routed_messages(
            messages,
            router=router,
            registry=registry,
            tools=tools,
            max_tokens=max_tokens,
            enable_thinking=enable_thinking,
            top_k=top_k,
        )

    def generate_routed_chat(
        self,
        *,
        user_prompt: str,
        system_prompt: str | None,
        router: MoESkillRouter | None,
        registry: SkillUnitRegistry | None,
        max_tokens: int,
        enable_thinking: bool,
        top_k: int,
    ) -> InferenceResult:
        messages: list[dict[str, Any]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})
        return self.generate_routed_messages(
            messages=messages,
            tools=None,
            router=router,
            registry=registry,
            max_tokens=max_tokens,
            enable_thinking=enable_thinking,
            top_k=top_k,
        )

    def encode_prompt_embedding(self, user_prompt: str) -> np.ndarray:
        return self._kernel.encode_prompt_embedding(
            user_prompt,
            tap_layer=self.tap_layer,
            system_prompt=None,
            enable_thinking=False,
        )

    def train_skill_deltas(
        self,
        *,
        examples: list[tuple[str | None, str, str]],
        rank: int,
        sub1bit_rank: int,
        steps: int,
        lr: float,
        seed: int,
        binary_ste: bool = False,
    ) -> tuple[list[LayerDeltaPayload], float, float, float]:
        adapters, losses, zero_diff = self._kernel.train_qlora_adapters(
            examples,
            layer_names=list(self.target_layers),
            enable_thinking=False,
            rank=rank,
            scale=1.0,
            binary_ste=binary_ste,
            steps=steps,
            lr=lr,
            seed=seed,
        )
        deltas = [
            build_layer_delta(
                name,
                adapters[name],
                encoding=DeltaEncoding.SUB1BIT_BINARY,
                sub1bit_rank=sub1bit_rank,
            )
            for name in self.target_layers
        ]
        head = float(np.mean(losses[: min(6, len(losses))]))
        tail = float(np.mean(losses[-min(6, len(losses)) :]))
        return deltas, head, tail, zero_diff

    def invalidate_unit(self, unit_id: str) -> None:
        self._kernel.invalidate_unit_cache(unit_id)


@dataclass
class TenantContext:
    """Strictly isolated per-company runtime state (units, telemetry, router)."""

    tenant_id: str
    root: Path
    registry: SkillUnitRegistry
    telemetry: TextTrajectoryStore
    router: MoESkillRouter
    quarantined_units: list[str] = field(default_factory=list)


class TenantManager:
    """Coordinates isolated CaPheOS tenants sharing one frozen ServingKernel."""

    def __init__(
        self,
        *,
        state_dir: Path,
        kernel: ServingKernel,
        initial_tenants: tuple[str, ...] = (),
    ) -> None:
        self.state_dir = state_dir
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.kernel = kernel
        self.lock = threading.Lock()
        self.studying_tenant: str | None = None
        self._tenants: dict[str, TenantContext] = {}

        # Discover existing tenants persisted on disk + ensure initial_tenants exist.
        tenants_root = self.state_dir / "tenants"
        if tenants_root.exists():
            for child in sorted(tenants_root.iterdir()):
                if child.is_dir() and _TENANT_RE.match(child.name):
                    self.ensure_tenant(child.name)
        for tid in initial_tenants:
            self.ensure_tenant(tid)

    def list_tenants(self) -> list[str]:
        return sorted(self._tenants.keys())

    def ensure_tenant(self, tenant_id: str) -> TenantContext:
        tid = validate_tenant_id(tenant_id)
        existing = self._tenants.get(tid)
        if existing is not None:
            return existing

        root = self.state_dir / "tenants" / tid
        registry = SkillUnitRegistry(root / "units")
        telemetry = TextTrajectoryStore(root / "telemetry" / "trajectories.jsonl")
        router = MoESkillRouter(tap_layer=self.kernel.tap_layer)

        # Hydrate all ENABLED units with valid SHA-256 digests and saved RouterGates.
        # Corrupt or tampered units are quarantined rather than bricking the whole server.
        quarantined: list[str] = []
        for manifest in registry.list_units():
            if manifest.state is UnitState.ENABLED:
                try:
                    registry.load_verified_deltas(manifest.unit_id)
                    gate = registry.load_router_gate(manifest.unit_id)
                    if gate is not None and gate.tap_layer == self.kernel.tap_layer:
                        router.register(gate)
                    else:
                        quarantined.append(manifest.unit_id)
                        warnings.warn(
                            f"Quarantined ENABLED unit '{manifest.unit_id}' for tenant '{tid}': "
                            "missing or tap-layer-mismatched RouterGate",
                            RuntimeWarning,
                            stacklevel=2,
                        )
                except (IntegrityError, FileNotFoundError, ValueError, json.JSONDecodeError) as exc:
                    quarantined.append(manifest.unit_id)
                    warnings.warn(
                        f"Quarantined corrupted ENABLED unit '{manifest.unit_id}' for tenant "
                        f"'{tid}': {exc}",
                        RuntimeWarning,
                        stacklevel=2,
                    )

        ctx = TenantContext(
            tenant_id=tid,
            root=root,
            registry=registry,
            telemetry=telemetry,
            router=router,
            quarantined_units=quarantined,
        )
        self._tenants[tid] = ctx
        return ctx

    def resolve_tenant_from_model(
        self, model_field: str | None, header_tenant: str | None
    ) -> tuple[TenantContext | None, bool]:
        """Resolve (TenantContext | None, use_base_only) from OpenAI model or header.

        Supported model conventions:
          - "subcor/base"            -> pure frozen base, no tenant routing
          - "subcor/<tenant_id>"     -> tenant <tenant_id> with MoE routing
          - "<tenant_id>"             -> if matching tenant exists
          - any other model + X-SUBCOR-Tenant header -> that tenant
        """
        if model_field == "subcor/base":
            return None, True
        if model_field and model_field.startswith("subcor/"):
            tid = model_field.split("/", 1)[1]
            return self.ensure_tenant(tid), False
        if header_tenant:
            return self.ensure_tenant(header_tenant), False
        if model_field and model_field in self._tenants:
            return self._tenants[model_field], False
        if len(self._tenants) == 1:
            return next(iter(self._tenants.values())), False
        return None, True

    def chat_completion(
        self,
        *,
        tenant: TenantContext | None,
        use_base_only: bool,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        max_tokens: int = 128,
        enable_thinking: bool = False,
        top_k: int = 1,
        seat: str = "intern",
        confidence_policy: ConfidencePolicy | None = None,
        record_telemetry: bool = True,
    ) -> InferenceResult:
        """Execute one serialized multi-turn routed chat completion with confidence scoring."""
        system_parts = [str(m.get("content", "")) for m in messages if m.get("role") == "system"]
        user_parts = [str(m.get("content", "")) for m in messages if m.get("role") == "user"]
        if not user_parts:
            raise ValueError("Chat request must contain at least one user message")
        system_prompt = "\n".join(system_parts).strip() or None
        user_prompt = user_parts[-1]

        active_router = None if (use_base_only or tenant is None) else tenant.router
        active_registry = None if (use_base_only or tenant is None) else tenant.registry

        with self.lock:
            multi_fn = getattr(self.kernel, "generate_routed_messages", None)
            if callable(multi_fn):
                result = multi_fn(
                    messages=messages,
                    tools=tools,
                    router=active_router,
                    registry=active_registry,
                    max_tokens=max_tokens,
                    enable_thinking=enable_thinking,
                    top_k=top_k,
                )
            else:
                result = self.kernel.generate_routed_chat(
                    user_prompt=user_prompt,
                    system_prompt=system_prompt,
                    router=active_router,
                    registry=active_registry,
                    max_tokens=max_tokens,
                    enable_thinking=enable_thinking,
                    top_k=top_k,
                )

        conf = evaluate_turn_confidence(
            token_logprobs=list(result.token_logprobs),
            routing=result.routing,
            hit_token_limit=result.hit_token_limit,
            repetition_ratio=result.repetition_ratio,
            policy=confidence_policy,
        )
        result = replace(result, confidence=conf)

        if record_telemetry and tenant is not None:
            # Live runtime responses start UNVERIFIED until CaPheOS verifies them.
            tenant.telemetry.append(
                TextTrajectory(
                    trajectory_id=f"live-{uuid.uuid4().hex[:12]}",
                    task_family="live_chat",
                    seat=seat,
                    prompt=user_prompt,
                    response=result.text,
                    tokens_generated=result.completion_tokens,
                    hit_token_limit=result.hit_token_limit,
                    repetition_ratio=result.repetition_ratio,
                    passed_verifier=False,
                    evidence=VerificationEvidence(
                        verified=False,
                        verifier="pending",
                        method="live_inference_unverified",
                        notes=(
                            f"routed={list(result.routing.selected_units)} "
                            f"escalate={conf.escalate_to_cortex} "
                            f"prob={conf.mean_token_prob:.3f}"
                        ),
                    ),
                )
            )
        return result

    def ingest_trajectories(
        self, tenant_id: str, raw_records: list[dict[str, Any]]
    ) -> dict[str, int]:
        """Append incoming trajectories (verified or unverified) to tenant store."""
        ctx = self.ensure_tenant(tenant_id)
        parsed: list[TextTrajectory] = []
        verified_count = 0
        for item in raw_records:
            ev_raw = item.get("evidence") or {}
            ev = VerificationEvidence(
                verified=bool(ev_raw.get("verified", False)),
                verifier=str(ev_raw.get("verifier", "unknown")),
                method=str(ev_raw.get("method", "api_ingest")),
                notes=str(ev_raw.get("notes", "")),
            )
            if ev.verified:
                verified_count += 1
            resp = str(item.get("response", ""))
            parsed.append(
                TextTrajectory(
                    trajectory_id=str(item.get("trajectory_id") or f"ing-{uuid.uuid4().hex[:10]}"),
                    task_family=str(item["task_family"]),
                    seat=str(item.get("seat", "intern")),
                    prompt=str(item["prompt"]),
                    response=resp,
                    tokens_generated=int(item.get("tokens_generated", len(resp.split()))),
                    hit_token_limit=bool(item.get("hit_token_limit", False)),
                    repetition_ratio=float(
                        item.get("repetition_ratio", ngram_repetition_ratio(resp))
                    ),
                    passed_verifier=bool(item.get("passed_verifier", False)),
                    evidence=ev,
                    senior_correction=(
                        None
                        if item.get("senior_correction") is None
                        else str(item["senior_correction"])
                    ),
                )
            )
        written = ctx.telemetry.append_many(parsed)
        return {
            "ingested": written,
            "verified": verified_count,
            "unverified": written - verified_count,
        }

    def run_study_cycle(
        self,
        tenant_id: str,
        *,
        unit_id: str,
        version: str,
        task_family: str,
        title: str,
        summary: str,
        proposer: str = "tools",
        system_prompt: str | None = None,
        eval_cases: list[dict[str, str]] | None = None,
        baseline_cases: list[dict[str, str]] | None = None,
        negative_prompts: list[str] | None = None,
        rank: int = 8,
        sub1bit_rank: int = 8,
        steps: int = 108,
        lr: float = 3e-4,
        max_regression_tolerance: float = 0.02,
        min_target_improvement: float = 0.25,
        seed: int = 42,
    ) -> dict[str, Any]:
        """Run a governed Study Cycle for one tenant using ONLY verified trajectories.

        Trains a zero-perturbation adapter, packs it into a Sub-1-Bit Skill Unit,
        fits a frozen-base RouterGate, runs the locked Conformance Gate, and stops
        at CONFORMANT (or DORMANT on fail) so an independent approver must promote.
        """
        ctx = self.ensure_tenant(tenant_id)
        pairs = ctx.telemetry.verified_distillation_pairs(task_family=task_family)

        # Build genuinely held-out target eval & strictly disjoint baseline/negative sets.
        if eval_cases:
            train_pairs = list(pairs)
            target_eval = [(c["prompt"], c["expected"]) for c in eval_cases]
            eval_mode = "explicit_held_out"
        elif len(pairs) >= 4:
            holdout_n = max(1, len(pairs) // 4)
            train_pairs = list(pairs[:-holdout_n])
            target_eval = list(pairs[-holdout_n:])
            eval_mode = "auto_held_out_split"
        else:
            train_pairs = list(pairs)
            target_eval = list(pairs)
            eval_mode = "in_sample_small_n"

        examples: list[tuple[str | None, str, str]] = [
            (system_prompt, prompt, target) for prompt, target in train_pairs
        ]

        base_eval = (
            [(c["prompt"], c["expected"]) for c in baseline_cases]
            if baseline_cases
            else list(DEFAULT_BASELINE_ANCHOR_CASES)
        )
        neg_prompts = (
            list(negative_prompts)
            if negative_prompts
            else list(DEFAULT_NEGATIVE_CALIBRATION_PROMPTS)
        )
        base_prompt_set = {p.strip().lower() for p, _ in base_eval}
        neg_prompt_set = {p.strip().lower() for p in neg_prompts}
        baseline_neg_overlap = len(base_prompt_set & neg_prompt_set)

        def score_cases(
            cases: list[tuple[str, str]],
            *,
            sys_prompt: str | None,
            use_router: bool,
        ) -> float:
            hits = 0
            for prompt, expected in cases:
                res = self.kernel.generate_routed_chat(
                    user_prompt=prompt,
                    system_prompt=sys_prompt,
                    router=ctx.router if use_router else None,
                    registry=ctx.registry if use_router else None,
                    max_tokens=64,
                    enable_thinking=False,
                    top_k=1,
                )
                if expected.strip().lower() in res.text.strip().lower():
                    hits += 1
            return hits / max(1, len(cases))

        with self.lock:
            self.studying_tenant = ctx.tenant_id
            try:
                baseline_before = score_cases(base_eval, sys_prompt=None, use_router=True)
                target_before = score_cases(target_eval, sys_prompt=system_prompt, use_router=True)

                pos_embs = np.stack(
                    [self.kernel.encode_prompt_embedding(p) for p, _ in train_pairs]
                )
                neg_embs = np.stack([self.kernel.encode_prompt_embedding(p) for p in neg_prompts])

                deltas, init_loss, final_loss, zero_diff = self.kernel.train_skill_deltas(
                    examples=examples,
                    rank=rank,
                    sub1bit_rank=sub1bit_rank,
                    steps=steps,
                    lr=lr,
                    seed=seed,
                )

                manifest = ctx.registry.propose(
                    unit_id=unit_id,
                    version=version,
                    title=title,
                    summary=summary,
                    proposer=proposer,
                    mode=AdaptationMode.DYNAMIC_EXPANSION_FP32,
                    deltas=deltas,
                    base_kernel_id=self.kernel.model_id,
                )
                gate = RouterGate.fit(
                    unit_id=unit_id,
                    tap_layer=self.kernel.tap_layer,
                    pos_embeddings=pos_embs,
                    neg_embeddings=neg_embs,
                )
                ctx.registry.save_router_gate(gate)
                ctx.registry.save_skill_spec(
                    SkillSpec(
                        unit_id=unit_id,
                        version=version,
                        title=title,
                        summary=summary,
                        task_family=task_family,
                        proposer=proposer,
                        system_prompt=system_prompt,
                        training_pairs=tuple((p, t) for p, t in train_pairs),
                        eval_cases=tuple({"prompt": p, "expected": e} for p, e in target_eval),
                        baseline_cases=tuple({"prompt": p, "expected": e} for p, e in base_eval),
                        negative_prompts=tuple(neg_prompts),
                        rank=rank,
                        sub1bit_rank=sub1bit_rank,
                        steps=steps,
                        lr=lr,
                        max_regression_tolerance=max_regression_tolerance,
                        min_target_improvement=min_target_improvement,
                    )
                )

                # Sandbox-register gate for locked conformance evaluation, then unregister
                # so the unit NEVER becomes live before independent approver sign-off.
                ctx.router.register(gate)
                try:
                    baseline_after = score_cases(base_eval, sys_prompt=None, use_router=True)
                    target_after = score_cases(
                        target_eval, sys_prompt=system_prompt, use_router=True
                    )
                finally:
                    ctx.router.unregister(unit_id)

                manifest = ctx.registry.record_conformance(
                    unit_id,
                    baseline_before=baseline_before,
                    baseline_after=baseline_after,
                    target_before=target_before,
                    target_after=target_after,
                    max_regression_tolerance=max_regression_tolerance,
                    min_target_improvement=min_target_improvement,
                )
            finally:
                self.studying_tenant = None

        return {
            "tenant_id": ctx.tenant_id,
            "verified_training_pairs": len(train_pairs),
            "eval_cases_count": len(target_eval),
            "eval_mode": eval_mode,
            "baseline_neg_overlap": baseline_neg_overlap,
            "initial_loss": init_loss,
            "final_loss": final_loss,
            "zero_perturbation_max_diff": zero_diff,
            "manifest": manifest.to_dict(),
        }

    def promote_unit(self, tenant_id: str, unit_id: str, *, approver: str) -> SkillUnitManifest:
        """Promote a CONFORMANT unit (enforcing no self-approval) and hot-mount its RouterGate."""
        ctx = self.ensure_tenant(tenant_id)
        with self.lock:
            # Validate RouterGate & tap_layer BEFORE persisting ENABLED on disk so
            # disk and in-memory router state can never diverge.
            gate = ctx.registry.load_router_gate(unit_id)
            if gate is None:
                raise GovernanceError(
                    f"Cannot promote Skill Unit '{unit_id}': missing router_gate.json"
                )
            if gate.tap_layer != self.kernel.tap_layer:
                raise GovernanceError(
                    f"Cannot promote Skill Unit '{unit_id}': RouterGate tap_layer "
                    f"'{gate.tap_layer}' does not match active kernel tap_layer "
                    f"'{self.kernel.tap_layer}'"
                )
            manifest = ctx.registry.promote(unit_id, approver=approver)
            ctx.router.register(gate)
            if unit_id in ctx.quarantined_units:
                ctx.quarantined_units.remove(unit_id)
            return manifest

    def rollback_unit(
        self, tenant_id: str, unit_id: str, *, actor: str, reason: str = ""
    ) -> SkillUnitManifest:
        """Atomically roll back an ENABLED unit and deregister it from the live MoE router."""
        ctx = self.ensure_tenant(tenant_id)
        with self.lock:
            manifest = ctx.registry.rollback(unit_id, actor=actor, reason=reason)
            ctx.router.unregister(unit_id)
            self.kernel.invalidate_unit(unit_id)
            return manifest


def _extract_bearer(auth_header: str | None) -> str | None:
    if not auth_header:
        return None
    parts = auth_header.strip().split(" ", 1)
    if len(parts) == 2 and parts[0].lower() == "bearer":
        return parts[1].strip()
    return None


class SubcorHTTPServer(ThreadingHTTPServer):
    """Multi-tenant SUBCOR HTTP server carrying TenantManager and optional API key."""

    allow_reuse_address = True

    def __init__(
        self,
        server_address: tuple[str, int],
        *,
        manager: TenantManager,
        api_key: str | None = None,
    ) -> None:
        self.manager = manager
        self.api_key = api_key
        super().__init__(server_address, SubcorRequestHandler)


class SubcorRequestHandler(BaseHTTPRequestHandler):
    """HTTP handler implementing OpenAI /v1/chat/completions + SUBCOR Control Plane."""

    server: SubcorHTTPServer
    protocol_version = "HTTP/1.1"

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        # Keep stdout clean for structured operational logs.
        return

    def _send_json(
        self,
        status: int,
        payload: dict[str, Any],
        extra_headers: dict[str, str] | None = None,
    ) -> None:
        body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        if extra_headers:
            for k, v in extra_headers.items():
                self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length) if length > 0 else b"{}"
        data = json.loads(raw.decode("utf-8"))
        if not isinstance(data, dict):
            raise ValueError("JSON request body must be an object")
        return data

    def _check_auth(self) -> bool:
        expected = self.server.api_key
        if not expected:
            return True
        provided = _extract_bearer(self.headers.get("Authorization"))
        if not provided or not hmac.compare_digest(provided, expected):
            self._send_json(
                HTTPStatus.UNAUTHORIZED,
                {"error": {"message": "Unauthorized", "type": "auth_error"}},
            )
            return False
        return True

    def do_GET(self) -> None:  # noqa: N802
        path = urlparse(self.path).path.rstrip("/") or "/"
        mgr = self.server.manager

        if path in ("/healthz", "/readyz"):
            tenants_summary = {
                tid: {
                    "enabled_units": list(mgr.ensure_tenant(tid).router.registered_units),
                    "quarantined_units": list(mgr.ensure_tenant(tid).quarantined_units),
                    "total_units": len(mgr.ensure_tenant(tid).registry.list_units()),
                }
                for tid in mgr.list_tenants()
            }
            self._send_json(
                HTTPStatus.OK,
                {
                    "status": "ok",
                    "model_id": mgr.kernel.model_id,
                    "tap_layer": mgr.kernel.tap_layer,
                    "studying_tenant": mgr.studying_tenant,
                    "tenants": tenants_summary,
                },
            )
            return

        if not self._check_auth():
            return

        if path == "/api/tags":
            # Ollama-compatible model discovery endpoint (used by StarNet's built-in `ollama` provider).
            now_iso = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            models = [
                {
                    "name": "subcor/base",
                    "model": "subcor/base",
                    "modified_at": now_iso,
                    "size": 0,
                    "digest": "subcor-frozen-base",
                }
            ]
            for tid in mgr.list_tenants():
                models.append(
                    {
                        "name": f"subcor/{tid}",
                        "model": f"subcor/{tid}",
                        "modified_at": now_iso,
                        "size": 0,
                        "digest": f"subcor-tenant-{tid}",
                    }
                )
            self._send_json(HTTPStatus.OK, {"models": models})
            return

        if path == "/v1/models":
            now = int(time.time())
            data = [
                {
                    "id": "subcor/base",
                    "object": "model",
                    "created": now,
                    "owned_by": "subcor",
                }
            ]
            for tid in mgr.list_tenants():
                data.append(
                    {
                        "id": f"subcor/{tid}",
                        "object": "model",
                        "created": now,
                        "owned_by": tid,
                    }
                )
            self._send_json(HTTPStatus.OK, {"object": "list", "data": data})
            return

        m_units = re.match(r"^/v1/subcor/tenants/([^/]+)/units$", path)
        if m_units:
            try:
                ctx = mgr.ensure_tenant(m_units.group(1))
                units = [m.to_dict() for m in ctx.registry.list_units()]
                self._send_json(
                    HTTPStatus.OK,
                    {
                        "tenant_id": ctx.tenant_id,
                        "active_routed_units": list(ctx.router.registered_units),
                        "units": units,
                    },
                )
            except ValueError as exc:
                self._send_json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
            return

        self._send_json(HTTPStatus.NOT_FOUND, {"error": f"Unknown route: {path}"})

    def do_POST(self) -> None:  # noqa: N802
        if not self._check_auth():
            return

        path = urlparse(self.path).path.rstrip("/")
        mgr = self.server.manager

        try:
            body = self._read_json()
        except Exception as exc:
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": f"Invalid JSON: {exc}"})
            return

        # 1. OpenAI-compatible chat completions (multi-turn, tools/tool_calls, true SSE streaming).
        if path == "/v1/chat/completions":
            try:
                model_field = body.get("model")
                header_tenant = self.headers.get("X-SUBCOR-Tenant")
                seat = self.headers.get("X-CaPheOS-Seat", "intern")
                tenant, use_base_only = mgr.resolve_tenant_from_model(
                    None if model_field is None else str(model_field),
                    header_tenant,
                )
                messages = body.get("messages")
                if not isinstance(messages, list):
                    raise ValueError("messages must be a list")
                tools = body.get("tools")
                if tools is not None and not isinstance(tools, list):
                    raise ValueError("tools must be a list when provided")
                max_tokens = int(body.get("max_tokens", 128))
                stream = bool(body.get("stream", False))
                enable_thinking = bool(body.get("enable_thinking", False))
                top_k = int(body.get("top_k", 1))
                conf_policy = ConfidencePolicy(
                    require_skill_match=bool(body.get("require_skill_match", False))
                )

                completion_id = f"chatcmpl-{uuid.uuid4().hex[:16]}"
                created = int(time.time())
                model_name = (
                    "subcor/base"
                    if (use_base_only or tenant is None)
                    else f"subcor/{tenant.tenant_id}"
                )

                # True token-by-token SSE streaming when backend supports stream_routed_messages
                # and no structured tool-call post-parsing is required.
                stream_fn = getattr(mgr.kernel, "stream_routed_messages", None)
                if stream and not tools and callable(stream_fn):
                    active_router = None if (use_base_only or tenant is None) else tenant.router
                    active_registry = None if (use_base_only or tenant is None) else tenant.registry
                    with mgr.lock:
                        decision, token_iter = stream_fn(
                            messages=messages,
                            tools=None,
                            router=active_router,
                            registry=active_registry,
                            max_tokens=max_tokens,
                            enable_thinking=enable_thinking,
                            top_k=top_k,
                        )
                        routed_header = ",".join(decision.selected_units) or "base"
                        self.send_response(HTTPStatus.OK)
                        self.send_header("Content-Type", "text/event-stream")
                        self.send_header("Cache-Control", "no-cache")
                        self.send_header("Connection", "close")
                        self.send_header("X-SUBCOR-Routed-Units", routed_header)
                        self.end_headers()

                        first_chunk = {
                            "id": completion_id,
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": model_name,
                            "choices": [
                                {"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}
                            ],
                        }
                        self.wfile.write(
                            f"data: {json.dumps(first_chunk, separators=(',', ':'))}\n\n".encode()
                        )
                        self.wfile.flush()

                        last_finish = "stop"
                        stream_lps: list[float] = []
                        stream_pieces: list[str] = []
                        gen_tokens = 0
                        for ev in token_iter:
                            stream_lps.append(float(ev.logprob))
                            gen_tokens = int(ev.generation_tokens)
                            if ev.text:
                                stream_pieces.append(ev.text)
                                chunk = {
                                    "id": completion_id,
                                    "object": "chat.completion.chunk",
                                    "created": created,
                                    "model": model_name,
                                    "choices": [
                                        {
                                            "index": 0,
                                            "delta": {"content": ev.text},
                                            "finish_reason": None,
                                        }
                                    ],
                                }
                                self.wfile.write(
                                    f"data: {json.dumps(chunk, separators=(',', ':'))}\n\n".encode()
                                )
                                self.wfile.flush()
                            if ev.finish_reason:
                                last_finish = ev.finish_reason

                        full_text = "".join(stream_pieces).strip()
                        hit_limit = last_finish == "length" or gen_tokens >= max_tokens
                        rep_ratio = ngram_repetition_ratio(full_text)
                        conf = evaluate_turn_confidence(
                            token_logprobs=stream_lps,
                            routing=decision,
                            hit_token_limit=hit_limit,
                            repetition_ratio=rep_ratio,
                            policy=conf_policy,
                        )
                        if tenant is not None:
                            user_parts = [
                                str(m.get("content", ""))
                                for m in messages
                                if m.get("role") == "user"
                            ]
                            last_user = user_parts[-1] if user_parts else ""
                            tenant.telemetry.append(
                                TextTrajectory(
                                    trajectory_id=f"live-{uuid.uuid4().hex[:12]}",
                                    task_family="live_chat",
                                    seat=seat,
                                    prompt=last_user,
                                    response=full_text,
                                    tokens_generated=gen_tokens,
                                    hit_token_limit=hit_limit,
                                    repetition_ratio=rep_ratio,
                                    passed_verifier=False,
                                    evidence=VerificationEvidence(
                                        verified=False,
                                        verifier="pending",
                                        method="live_inference_unverified",
                                        notes=(
                                            f"routed={list(decision.selected_units)} "
                                            f"escalate={conf.escalate_to_cortex} "
                                            f"prob={conf.mean_token_prob:.3f}"
                                        ),
                                    ),
                                )
                            )

                        done_chunk = {
                            "id": completion_id,
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": model_name,
                            "choices": [{"index": 0, "delta": {}, "finish_reason": last_finish}],
                            "subcor": {
                                "tenant": None if tenant is None else tenant.tenant_id,
                                "selected_units": list(decision.selected_units),
                                "fell_back_to_base": decision.fell_back_to_base,
                                "confidence": conf.to_dict(),
                            },
                        }
                        self.wfile.write(
                            f"data: {json.dumps(done_chunk, separators=(',', ':'))}\n\ndata: [DONE]\n\n".encode()
                        )
                        self.wfile.flush()
                    return

                res = mgr.chat_completion(
                    tenant=tenant,
                    use_base_only=use_base_only,
                    messages=messages,
                    tools=tools,
                    max_tokens=max_tokens,
                    enable_thinking=enable_thinking,
                    top_k=top_k,
                    seat=seat,
                    confidence_policy=conf_policy,
                )
            except (ValueError, IntegrityError) as exc:
                self._send_json(HTTPStatus.BAD_REQUEST, {"error": {"message": str(exc)}})
                return

            routed_header = ",".join(res.routing.selected_units) or "base"
            escalate_header = (
                "true" if (res.confidence and res.confidence.escalate_to_cortex) else "false"
            )
            conf_header = (
                f"{res.confidence.mean_token_prob:.4f}" if res.confidence is not None else "1.0000"
            )
            finish_reason = (
                "tool_calls" if res.tool_calls else ("length" if res.hit_token_limit else "stop")
            )

            if stream:
                delta_obj: dict[str, Any] = {"role": "assistant"}
                if res.text:
                    delta_obj["content"] = res.text
                if res.tool_calls:
                    delta_obj["tool_calls"] = [
                        {"index": idx, **tc} for idx, tc in enumerate(res.tool_calls)
                    ]
                chunks = [
                    {
                        "id": completion_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": model_name,
                        "choices": [{"index": 0, "delta": delta_obj, "finish_reason": None}],
                    },
                    {
                        "id": completion_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": model_name,
                        "choices": [{"index": 0, "delta": {}, "finish_reason": finish_reason}],
                    },
                ]
                sse_payload = (
                    "".join(f"data: {json.dumps(c, separators=(',', ':'))}\n\n" for c in chunks)
                    + "data: [DONE]\n\n"
                ).encode("utf-8")
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-cache")
                self.send_header("X-SUBCOR-Routed-Units", routed_header)
                self.send_header("X-SUBCOR-Escalate", escalate_header)
                self.send_header("X-SUBCOR-Confidence", conf_header)
                self.send_header("Content-Length", str(len(sse_payload)))
                self.end_headers()
                self.wfile.write(sse_payload)
                return

            msg_obj: dict[str, Any] = {
                "role": "assistant",
                "content": res.text if (res.text or not res.tool_calls) else None,
            }
            if res.tool_calls:
                msg_obj["tool_calls"] = list(res.tool_calls)

            payload = {
                "id": completion_id,
                "object": "chat.completion",
                "created": created,
                "model": model_name,
                "choices": [
                    {
                        "index": 0,
                        "message": msg_obj,
                        "finish_reason": finish_reason,
                    }
                ],
                "usage": {
                    "prompt_tokens": res.prompt_tokens,
                    "completion_tokens": res.completion_tokens,
                    "total_tokens": res.prompt_tokens + res.completion_tokens,
                },
                "subcor": {
                    "tenant": None if tenant is None else tenant.tenant_id,
                    "selected_units": list(res.routing.selected_units),
                    "fell_back_to_base": res.routing.fell_back_to_base,
                    "scores": {k: round(v, 5) for k, v in res.routing.scores.items()},
                    "confidence": None if res.confidence is None else res.confidence.to_dict(),
                },
            }
            self._send_json(
                HTTPStatus.OK,
                payload,
                extra_headers={
                    "X-SUBCOR-Routed-Units": routed_header,
                    "X-SUBCOR-Escalate": escalate_header,
                    "X-SUBCOR-Confidence": conf_header,
                },
            )
            return

        # 2. Ingest trajectories for a tenant.
        m_traj = re.match(r"^/v1/subcor/tenants/([^/]+)/trajectories$", path)
        if m_traj:
            try:
                recs = body.get("trajectories")
                if not isinstance(recs, list):
                    raise ValueError("trajectories must be a list")
                summary = mgr.ingest_trajectories(m_traj.group(1), recs)
                self._send_json(HTTPStatus.OK, summary)
            except (ValueError, KeyError) as exc:
                self._send_json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
            return

        # 3. Trigger a governed Study Cycle for a tenant.
        m_study = re.match(r"^/v1/subcor/tenants/([^/]+)/study$", path)
        if m_study:
            try:
                result = mgr.run_study_cycle(
                    m_study.group(1),
                    unit_id=str(body["unit_id"]),
                    version=str(body.get("version", "1.0.0")),
                    task_family=str(body["task_family"]),
                    title=str(body.get("title", body["unit_id"])),
                    summary=str(body.get("summary", "Study cycle Skill Unit")),
                    proposer=str(body.get("proposer", "tools")),
                    system_prompt=body.get("system_prompt"),
                    eval_cases=body.get("eval_cases"),
                    baseline_cases=body.get("baseline_cases"),
                    negative_prompts=body.get("negative_prompts"),
                    rank=int(body.get("rank", 8)),
                    sub1bit_rank=int(body.get("sub1bit_rank", 8)),
                    steps=int(body.get("steps", 108)),
                    lr=float(body.get("lr", 3e-4)),
                    max_regression_tolerance=float(body.get("max_regression_tolerance", 0.02)),
                    min_target_improvement=float(body.get("min_target_improvement", 0.25)),
                    seed=int(body.get("seed", 42)),
                )
                self._send_json(HTTPStatus.OK, result)
            except (KeyError, ValueError, VerificationError, GovernanceError) as exc:
                self._send_json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
            return

        # 4. Promote a CONFORMANT unit (enforces no self-approval).
        m_promote = re.match(r"^/v1/subcor/tenants/([^/]+)/units/([^/]+)/promote$", path)
        if m_promote:
            try:
                manifest = mgr.promote_unit(
                    m_promote.group(1),
                    m_promote.group(2),
                    approver=str(body.get("approver", "")),
                )
                self._send_json(HTTPStatus.OK, {"manifest": manifest.to_dict()})
            except GovernanceError as exc:
                self._send_json(HTTPStatus.FORBIDDEN, {"error": str(exc)})
            except (FileNotFoundError, ValueError, IntegrityError) as exc:
                self._send_json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
            return

        # 5. Atomic rollback of an ENABLED unit.
        m_rb = re.match(r"^/v1/subcor/tenants/([^/]+)/units/([^/]+)/rollback$", path)
        if m_rb:
            try:
                manifest = mgr.rollback_unit(
                    m_rb.group(1),
                    m_rb.group(2),
                    actor=str(body.get("actor", "risk")),
                    reason=str(body.get("reason", "")),
                )
                self._send_json(HTTPStatus.OK, {"manifest": manifest.to_dict()})
            except GovernanceError as exc:
                self._send_json(HTTPStatus.CONFLICT, {"error": str(exc)})
            except (FileNotFoundError, ValueError) as exc:
                self._send_json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
            return

        # 6. Export a portable CaPheOS format-2 bundle.
        m_exp = re.match(r"^/v1/subcor/tenants/([^/]+)/units/([^/]+)/export-capheos$", path)
        if m_exp:
            try:
                ctx = mgr.ensure_tenant(m_exp.group(1))
                dest = Path(str(body.get("destination", ctx.root / "exports")))
                out_dir = ctx.registry.export_capheos_unit(m_exp.group(2), dest)
                self._send_json(HTTPStatus.OK, {"exported_path": str(out_dir)})
            except GovernanceError as exc:
                self._send_json(HTTPStatus.CONFLICT, {"error": str(exc)})
            except (FileNotFoundError, ValueError, IntegrityError) as exc:
                self._send_json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
            return

        self._send_json(HTTPStatus.NOT_FOUND, {"error": f"Unknown route: {path}"})


def build_dummy_delta(
    layer_name: str, *, d_out: int = 32, d_in: int = 32, rank: int = 2
) -> LayerDeltaPayload:
    """Helper for deterministic offline testing without MLX."""
    rng = np.random.default_rng(7)
    adapter = LowRankAdapter(
        name=layer_name,
        a=rng.normal(0.0, 0.2, size=(rank, d_in)),
        b=rng.normal(0.0, 0.2, size=(d_out, rank)),
    )
    return build_layer_delta(
        layer_name,
        adapter,
        encoding=DeltaEncoding.SUB1BIT_BINARY,
        sub1bit_rank=rank,
    )
