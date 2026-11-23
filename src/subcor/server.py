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

import inspect
import json
import math
import queue
import re
import threading
import time
import uuid
import warnings
from collections.abc import Iterator
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Protocol
from urllib.parse import urlparse

import numpy as np

from subcor.auth import AuthPolicy, AuthPrincipal
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


@dataclass(frozen=True)
class ConformancePolicy:
    """Owner-controlled limits for the conformance gate of a Study Cycle.

    Set by whoever constructs the `TenantManager` (the server owner), never by the
    proposer. A study request may make the thresholds stricter, never looser, and its
    unit is always scored on `anchor_cases` in addition to any proposer-supplied
    baseline cases, so a proposer cannot swap in an easier regression suite.
    """

    max_regression_tolerance: float = 0.02
    min_target_improvement: float = 0.25
    anchor_cases: tuple[tuple[str, str], ...] = DEFAULT_BASELINE_ANCHOR_CASES
    # Always used to calibrate router gates, in addition to any proposer negatives.
    negative_prompts: tuple[str, ...] = DEFAULT_NEGATIVE_CALIBRATION_PROMPTS

    def __post_init__(self) -> None:
        if not self.anchor_cases:
            raise ValueError("anchor_cases must not be empty: they are the regression check")
        anchors = {p.strip().lower() for p, _ in self.anchor_cases}
        if not [p for p in self.negative_prompts if p.strip() and p.strip().lower() not in anchors]:
            raise ValueError("negative_prompts must include prompts that are not anchor prompts")
        _check_threshold("max_regression_tolerance", self.max_regression_tolerance, 0.0, 1.0)
        _check_threshold("min_target_improvement", self.min_target_improvement, 0.0, 1.0)

    def resolve(
        self, max_regression_tolerance: float | None, min_target_improvement: float | None
    ) -> tuple[float, float]:
        """Effective thresholds for one study: the request's if at least as strict."""
        tol = self.max_regression_tolerance
        if max_regression_tolerance is not None:
            _check_threshold("max_regression_tolerance", max_regression_tolerance, 0.0, 1.0)
            if max_regression_tolerance > tol:
                raise ValueError(
                    f"max_regression_tolerance {max_regression_tolerance} is looser than the "
                    f"owner policy limit {tol}"
                )
            tol = float(max_regression_tolerance)
        imp = self.min_target_improvement
        if min_target_improvement is not None:
            _check_threshold("min_target_improvement", min_target_improvement, 0.0, 1.0)
            if min_target_improvement < imp:
                raise ValueError(
                    f"min_target_improvement {min_target_improvement} is looser than the "
                    f"owner policy limit {imp}"
                )
            imp = float(min_target_improvement)
        return tol, imp


def _check_threshold(name: str, value: float, low: float, high: float) -> None:
    if not isinstance(value, int | float) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number, got {value!r}")
    if not low <= value <= high:
        raise ValueError(f"{name} must be in [{low}, {high}], got {value!r}")


def _optional_float(value: Any) -> float | None:
    return None if value is None else float(value)


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
        examples: list[Any],
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

    # Optional hook (not part of the required protocol; looked up with getattr):
    #   prepare_units(units: Sequence[tuple[SkillUnitRegistry, str]]) -> None
    # Called under TenantManager.lock before a unit can be routed (sandbox conformance,
    # promotion, startup hydration) so backends that must load adapters out of process
    # (llama.cpp GGUF LoRAs) can do so. MLX does not need it.


class MLXServingAdapter:
    """Wraps MLXKernel behind the ServingKernel protocol."""

    def __init__(
        self,
        model_id: str = "mlx-community/Qwen3.5-0.8B-4bit",
        *,
        target_layer_indices: list[int] | None = None,
        target_projections: tuple[str, ...] | None = None,
        layer_preset: str | None = None,
        num_layers: int | None = None,
    ) -> None:
        from subcor.backends.mlx_kernel import MLXKernel

        indices = (
            target_layer_indices
            if target_layer_indices is not None or layer_preset is not None
            else [18, 19, 20, 21, 22, 23]
        )
        self._kernel = MLXKernel(
            model_id=model_id,
            target_layer_indices=indices,
            target_projections=target_projections,
            layer_preset=layer_preset,
            num_layers=num_layers,
        )
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
        seat: str | None = None,
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
                seat=seat,
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
        seat: str | None = None,
    ) -> tuple[RoutingDecision, Iterator[Any]]:
        return self._kernel.stream_routed_messages(
            messages,
            router=router,
            registry=registry,
            tools=tools,
            max_tokens=max_tokens,
            enable_thinking=enable_thinking,
            top_k=top_k,
            seat=seat,
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
        examples: list[Any],
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


@dataclass
class StudyJob:
    """Observable status of an asynchronous background Study Cycle (Issue #4)."""

    job_id: str
    tenant_id: str
    unit_id: str
    status: str = "queued"  # queued | running | succeeded | failed
    created_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    started_at: str | None = None
    finished_at: str | None = None
    result: dict[str, Any] | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "tenant_id": self.tenant_id,
            "unit_id": self.unit_id,
            "status": self.status,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "result": self.result,
            "error": self.error,
        }


class TenantManager:
    """Coordinates isolated CaPheOS tenants sharing one frozen ServingKernel."""

    def __init__(
        self,
        *,
        state_dir: Path,
        kernel: ServingKernel,
        initial_tenants: tuple[str, ...] = (),
        allow_auto_create_tenants: bool | None = None,
        conformance_policy: ConformancePolicy | None = None,
    ) -> None:
        self.state_dir = state_dir
        self.conformance_policy = conformance_policy or ConformancePolicy()
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.kernel = kernel
        self.lock = threading.Lock()
        self.studying_tenant: str | None = None
        self._tenants: dict[str, TenantContext] = {}
        # By default disallow arbitrary tenant auto-creation when explicit tenants are provisioned.
        self.allow_auto_create_tenants = (
            (len(initial_tenants) == 0)
            if allow_auto_create_tenants is None
            else allow_auto_create_tenants
        )

        # Background Study Job queue (Issue #4) so HTTP study requests return immediately.
        self._study_jobs: dict[str, StudyJob] = {}
        self._study_jobs_lock = threading.Lock()
        self._study_queue: queue.Queue[tuple[str, dict[str, Any]]] = queue.Queue()
        self._study_worker = threading.Thread(
            target=self._study_worker_loop, name="subcor-study-worker", daemon=True
        )
        self._study_worker.start()

        # Discover existing tenants persisted on disk + ensure initial_tenants exist.
        # Adapter preparation for hydrated ENABLED units is batched into one kernel call.
        self._deferred_prepare: list[TenantContext] | None = []
        tenants_root = self.state_dir / "tenants"
        if tenants_root.exists():
            for child in sorted(tenants_root.iterdir()):
                if child.is_dir() and _TENANT_RE.match(child.name):
                    self.ensure_tenant(child.name)
        for tid in initial_tenants:
            self.ensure_tenant(tid)
        deferred, self._deferred_prepare = self._deferred_prepare, None
        if callable(getattr(self.kernel, "prepare_units", None)):
            with self.lock:
                self._prepare_hydrated_units(deferred)

    def _prepare_kernel_units(self, units: list[tuple[SkillUnitRegistry, str]]) -> None:
        """Call the kernel's optional prepare_units hook (no-op for MLX/test kernels)."""
        prepare = getattr(self.kernel, "prepare_units", None)
        if callable(prepare) and units:
            prepare(units)

    def _prepare_hydrated_units(self, contexts: list[TenantContext]) -> None:
        """Prepare routable units of `contexts`; quarantine any unit whose adapter fails."""
        if not callable(getattr(self.kernel, "prepare_units", None)):
            return
        units = [(ctx, uid) for ctx in contexts for uid in ctx.router.registered_units]
        if not units:
            return
        try:
            self._prepare_kernel_units([(ctx.registry, uid) for ctx, uid in units])
            return
        except Exception:
            pass
        for ctx, uid in units:
            try:
                self._prepare_kernel_units([(ctx.registry, uid)])
            except Exception as exc:
                ctx.router.unregister(uid)
                ctx.quarantined_units.append(uid)
                warnings.warn(
                    f"Quarantined ENABLED unit '{uid}' for tenant '{ctx.tenant_id}': "
                    f"kernel could not load its adapter: {exc}",
                    RuntimeWarning,
                    stacklevel=2,
                )

    def _study_worker_loop(self) -> None:
        while True:
            job_id, kwargs = self._study_queue.get()
            try:
                with self._study_jobs_lock:
                    job = self._study_jobs[job_id]
                    job.status = "running"
                    job.started_at = datetime.now(UTC).isoformat()
                result = self.run_study_cycle(**kwargs)
                with self._study_jobs_lock:
                    job = self._study_jobs[job_id]
                    job.status = "succeeded"
                    job.result = result
                    job.finished_at = datetime.now(UTC).isoformat()
            except Exception as exc:
                with self._study_jobs_lock:
                    job = self._study_jobs[job_id]
                    job.status = "failed"
                    job.error = str(exc)
                    job.finished_at = datetime.now(UTC).isoformat()
            finally:
                self._study_queue.task_done()

    def submit_study_job(self, tenant_id: str, **kwargs: Any) -> StudyJob:
        """Enqueue a non-blocking background Study Cycle and return its StudyJob handle."""
        ctx = self.get_tenant(tenant_id)
        job_id = f"study-{uuid.uuid4().hex[:12]}"
        job = StudyJob(
            job_id=job_id,
            tenant_id=ctx.tenant_id,
            unit_id=str(kwargs["unit_id"]),
        )
        with self._study_jobs_lock:
            self._study_jobs[job_id] = job
        self._study_queue.put((job_id, {"tenant_id": ctx.tenant_id, **kwargs}))
        return job

    def get_study_job(self, job_id: str) -> StudyJob:
        with self._study_jobs_lock:
            job = self._study_jobs.get(job_id)
            if job is None:
                raise FileNotFoundError(f"Unknown study job '{job_id}'")
            return job

    def list_study_jobs(self, tenant_id: str | None = None) -> list[StudyJob]:
        with self._study_jobs_lock:
            jobs = list(self._study_jobs.values())
        if tenant_id is not None:
            jobs = [j for j in jobs if j.tenant_id == tenant_id]
        return jobs

    def list_tenants(self) -> list[str]:
        return sorted(self._tenants.keys())

    def get_tenant(self, tenant_id: str) -> TenantContext:
        """Return an existing tenant, or auto-create only if allow_auto_create_tenants=True."""
        tid = validate_tenant_id(tenant_id)
        existing = self._tenants.get(tid)
        if existing is not None:
            return existing
        if not self.allow_auto_create_tenants:
            raise FileNotFoundError(
                f"Unknown tenant '{tid}' (auto-provisioning is disabled; provision tenant explicitly)"
            )
        return self.ensure_tenant(tid)

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
        if self._deferred_prepare is not None:
            self._deferred_prepare.append(ctx)
        elif router.registered_units and callable(getattr(self.kernel, "prepare_units", None)):
            with self.lock:
                self._prepare_hydrated_units([ctx])
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
            return self.get_tenant(tid), False
        if header_tenant:
            return self.get_tenant(header_tenant), False
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
                kwargs: dict[str, Any] = {
                    "messages": messages,
                    "tools": tools,
                    "router": active_router,
                    "registry": active_registry,
                    "max_tokens": max_tokens,
                    "enable_thinking": enable_thinking,
                    "top_k": top_k,
                }
                if "seat" in inspect.signature(multi_fn).parameters:
                    kwargs["seat"] = seat
                result = multi_fn(**kwargs)
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
        training_trajectories: list[dict[str, Any]] | None = None,
        eval_cases: list[dict[str, str]] | None = None,
        baseline_cases: list[dict[str, str]] | None = None,
        negative_prompts: list[str] | None = None,
        rank: int = 8,
        sub1bit_rank: int = 8,
        steps: int = 108,
        lr: float = 3e-4,
        max_regression_tolerance: float | None = None,
        min_target_improvement: float | None = None,
        seat_scope: str | None = None,
        num_prototypes: int = 1,
        seed: int = 42,
    ) -> dict[str, Any]:
        """Run a governed Study Cycle for one tenant using ONLY verified trajectories.

        Conformance thresholds default to `self.conformance_policy`; a caller may only
        tighten them. The policy's anchor cases are always scored, and the worse of the
        anchor and proposer-baseline regressions is what the gate sees.

        Trains a zero-perturbation adapter, packs it into a Sub-1-Bit Skill Unit,
        fits a frozen-base RouterGate (optionally seat-scoped and multi-prototype),
        runs the locked Conformance Gate, and stops at CONFORMANT (or DORMANT on fail)
        so an independent approver must promote.

        Uses phased kernel locking so live chat requests can interleave between
        baseline evaluation, training, and conformance phases rather than starving.
        """
        ctx = self.get_tenant(tenant_id)
        policy = self.conformance_policy
        max_regression_tolerance, min_target_improvement = policy.resolve(
            max_regression_tolerance, min_target_improvement
        )
        trajectories_list: list[dict[str, Any]] = list(training_trajectories or [])
        try:
            pairs = ctx.telemetry.verified_distillation_pairs(task_family=task_family)
        except VerificationError:
            if not trajectories_list:
                raise
            pairs = []

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

        examples: list[Any] = [(system_prompt, prompt, target) for prompt, target in train_pairs]
        if trajectories_list:
            examples.extend(trajectories_list)
        if not examples:
            raise ValueError(
                f"No verified training examples available for task_family={task_family!r}"
            )

        pos_prompts: list[str] = [p for p, _ in train_pairs]
        for tr in trajectories_list:
            msgs = tr.get("messages") or []
            u_prompt = next(
                (str(m.get("content", "")).strip() for m in msgs if m.get("role") == "user"),
                "",
            )
            if u_prompt:
                pos_prompts.append(u_prompt)
        if not pos_prompts:
            pos_prompts = [p for p, _ in target_eval] or ["Perform the requested skill."]

        anchor_eval = list(policy.anchor_cases)
        base_eval = (
            [(c["prompt"], c["expected"]) for c in baseline_cases]
            if baseline_cases
            else anchor_eval
        )
        # Owner anchors are scored separately when the proposer brought its own baseline.
        extra_anchor_eval = anchor_eval if base_eval != anchor_eval else []
        # The router gate is calibrated not to fire on negative prompts. A proposer who listed
        # only the owner anchors as negatives could tune the gate to stay off exactly those
        # prompts and pass the anchor check by construction. So the owner's negatives are
        # always included (a proposer can add negatives, never replace them), and anchor
        # prompts themselves are never used as negatives.
        if negative_prompts is not None and not all(isinstance(p, str) for p in negative_prompts):
            raise ValueError("negative_prompts must be a list of strings")
        anchor_prompt_set = {p.strip().lower() for p, _ in anchor_eval}
        neg_prompts = []
        seen_neg: set[str] = set()
        for p in (*policy.negative_prompts, *(negative_prompts or ())):
            key = p.strip().lower()
            if key and key not in anchor_prompt_set and key not in seen_neg:
                seen_neg.add(key)
                neg_prompts.append(p)
        base_prompt_set = {p.strip().lower() for p, _ in base_eval} | anchor_prompt_set
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

        self.studying_tenant = ctx.tenant_id
        try:
            # Phase 1: Baseline evaluation & frozen-base prompt embeddings.
            with self.lock:
                baseline_before = score_cases(base_eval, sys_prompt=None, use_router=True)
                anchor_before = (
                    score_cases(extra_anchor_eval, sys_prompt=None, use_router=True)
                    if extra_anchor_eval
                    else None
                )
                target_before = score_cases(target_eval, sys_prompt=system_prompt, use_router=True)
                pos_embs = np.stack([self.kernel.encode_prompt_embedding(p) for p in pos_prompts])
                neg_embs = np.stack([self.kernel.encode_prompt_embedding(p) for p in neg_prompts])

            # Phase 2: Adapter training (releases lock before/after so chat can interleave).
            # Kernels that can split training from packing (llama.cpp + PyTorch trainer) do
            # the CPU-only sub-1-bit packing outside the lock, so chat is not blocked by it.
            train_adapters = getattr(self.kernel, "train_skill_adapters", None)
            pack_deltas = getattr(self.kernel, "pack_skill_deltas", None)
            if callable(train_adapters) and callable(pack_deltas):
                with self.lock:
                    trained = train_adapters(
                        examples=examples,
                        rank=rank,
                        steps=steps,
                        lr=lr,
                        seed=seed,
                    )
                deltas = pack_deltas(trained, sub1bit_rank=sub1bit_rank)
                init_loss = trained.initial_loss
                final_loss = trained.final_loss
                zero_diff = trained.zero_perturbation_max_diff
            else:
                with self.lock:
                    deltas, init_loss, final_loss, zero_diff = self.kernel.train_skill_deltas(
                        examples=examples,
                        rank=rank,
                        sub1bit_rank=sub1bit_rank,
                        steps=steps,
                        lr=lr,
                        seed=seed,
                    )

            # Phase 3: Disk packaging & analytical RouterGate fitting (lock-free CPU work!).
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
                seat_scope=seat_scope,
                num_prototypes=num_prototypes,
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
                    training_trajectories=tuple(dict(tr) for tr in trajectories_list),
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

            # Phase 4: Sandbox-register gate for locked conformance evaluation, then unregister
            # so the unit NEVER becomes live before independent approver sign-off.
            with self.lock:
                # Backends that load adapters out of process (llama.cpp) load the unit now.
                self._prepare_kernel_units([(ctx.registry, unit_id)])
                ctx.router.register(gate)
                try:
                    baseline_after = score_cases(base_eval, sys_prompt=None, use_router=True)
                    anchor_after = (
                        score_cases(extra_anchor_eval, sys_prompt=None, use_router=True)
                        if extra_anchor_eval
                        else None
                    )
                    target_after = score_cases(
                        target_eval, sys_prompt=system_prompt, use_router=True
                    )
                finally:
                    ctx.router.unregister(unit_id)

            if (
                anchor_before is not None
                and anchor_after is not None
                and anchor_before - anchor_after > baseline_before - baseline_after
            ):
                baseline_before, baseline_after = anchor_before, anchor_after
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
            "seat_scope": gate.seat_scope,
            "num_prototypes": len(gate.prototypes) or 1,
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
            # Load the unit's adapter (llama.cpp) BEFORE persisting ENABLED, but only for a
            # promotable unit so rejected promotions never trigger backend work.
            pending = ctx.registry.get_manifest(unit_id)
            if (
                pending.state is UnitState.CONFORMANT
                and pending.conformance is not None
                and pending.conformance.passed
                and bool(approver and approver.strip())
                and approver.strip() != pending.proposer.strip()
            ):
                self._prepare_kernel_units([(ctx.registry, unit_id)])
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


def _resolve_sandboxed_export_dir(export_root: Path, requested: str | None) -> Path:
    """Resolve an export destination strictly inside `export_root`, blocking traversal."""
    root_resolved = export_root.expanduser().resolve()
    if not requested or not requested.strip():
        return root_resolved
    req_path = Path(requested)
    candidate = (req_path if req_path.is_absolute() else (root_resolved / req_path)).resolve()
    try:
        candidate.relative_to(root_resolved)
    except ValueError as exc:
        raise ValueError(
            f"Export destination '{requested}' escapes configured export_root '{root_resolved}'"
        ) from exc
    return candidate


class SubcorHTTPServer(ThreadingHTTPServer):
    """Multi-tenant SUBCOR HTTP server carrying TenantManager, AuthPolicy, and sandboxed exports."""

    allow_reuse_address = True

    def __init__(
        self,
        server_address: tuple[str, int],
        *,
        manager: TenantManager,
        api_key: str | None = None,
        auth_policy: AuthPolicy | None = None,
        export_root: Path | None = None,
    ) -> None:
        self.manager = manager
        self.api_key = api_key
        if auth_policy is not None:
            self.auth_policy = auth_policy
        elif api_key:
            self.auth_policy = AuthPolicy.from_single_key(api_key)
        else:
            self.auth_policy = AuthPolicy.disabled()
        self.export_root = (
            export_root.expanduser().resolve()
            if export_root is not None
            else (manager.state_dir / "exports").resolve()
        )
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

    def _authenticate(self) -> AuthPrincipal | None:
        provided = _extract_bearer(self.headers.get("Authorization"))
        principal = self.server.auth_policy.authenticate(provided)
        if principal is None:
            # Close connection on 401 so any unconsumed POST body cannot corrupt keep-alive.
            self.close_connection = True
            self._send_json(
                HTTPStatus.UNAUTHORIZED,
                {"error": {"message": "Unauthorized", "type": "auth_error"}},
                extra_headers={"Connection": "close"},
            )
            return None
        return principal

    def _authorize(
        self, principal: AuthPrincipal, *, role: str, tenant_id: str | None = None
    ) -> bool:
        if not principal.has_role(role):
            self._send_json(
                HTTPStatus.FORBIDDEN,
                {
                    "error": {
                        "message": f"Actor '{principal.actor}' lacks required role '{role}'",
                        "type": "forbidden",
                    }
                },
            )
            return False
        if tenant_id is not None and not principal.allows_tenant(tenant_id):
            self._send_json(
                HTTPStatus.FORBIDDEN,
                {
                    "error": {
                        "message": f"Actor '{principal.actor}' is not authorized for tenant '{tenant_id}'",
                        "type": "forbidden",
                    }
                },
            )
            return False
        return True

    def do_GET(self) -> None:  # noqa: N802
        path = urlparse(self.path).path.rstrip("/") or "/"
        mgr = self.server.manager

        if path == "/healthz":
            self._send_json(HTTPStatus.OK, {"status": "ok"})
            return

        if path == "/readyz":
            provided = _extract_bearer(self.headers.get("Authorization"))
            principal = self.server.auth_policy.authenticate(provided)
            if self.server.auth_policy.enabled and principal is None:
                # Minimal unauthenticated readiness probe (no tenant/unit enumeration).
                self._send_json(HTTPStatus.OK, {"status": "ok", "ready": True})
                return
            assert principal is not None
            visible_tenants = [t for t in mgr.list_tenants() if principal.allows_tenant(t)]
            tenants_summary = {
                tid: {
                    "enabled_units": list(mgr.ensure_tenant(tid).router.registered_units),
                    "quarantined_units": list(mgr.ensure_tenant(tid).quarantined_units),
                    "total_units": len(mgr.ensure_tenant(tid).registry.list_units()),
                }
                for tid in visible_tenants
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

        principal = self._authenticate()
        if principal is None:
            return

        visible_tenants = [t for t in mgr.list_tenants() if principal.allows_tenant(t)]

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
            for tid in visible_tenants:
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
            for tid in visible_tenants:
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
            tid = m_units.group(1)
            if not self._authorize(principal, role="chat", tenant_id=tid):
                return
            try:
                ctx = mgr.get_tenant(tid)
                units = [m.to_dict() for m in ctx.registry.list_units()]
                self._send_json(
                    HTTPStatus.OK,
                    {
                        "tenant_id": ctx.tenant_id,
                        "active_routed_units": list(ctx.router.registered_units),
                        "quarantined_units": list(ctx.quarantined_units),
                        "units": units,
                    },
                )
            except FileNotFoundError as exc:
                self._send_json(HTTPStatus.NOT_FOUND, {"error": str(exc)})
            except ValueError as exc:
                self._send_json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
            return

        m_jobs = re.match(r"^/v1/subcor/tenants/([^/]+)/study/jobs$", path)
        if m_jobs:
            tid = m_jobs.group(1)
            if not self._authorize(principal, role="propose", tenant_id=tid):
                return
            try:
                ctx = mgr.get_tenant(tid)
                jobs = [j.to_dict() for j in mgr.list_study_jobs(ctx.tenant_id)]
                self._send_json(HTTPStatus.OK, {"tenant_id": ctx.tenant_id, "jobs": jobs})
            except FileNotFoundError as exc:
                self._send_json(HTTPStatus.NOT_FOUND, {"error": str(exc)})
            except ValueError as exc:
                self._send_json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
            return

        m_job = re.match(r"^/v1/subcor/tenants/([^/]+)/study/jobs/([^/]+)$", path)
        if m_job:
            tid, jid = m_job.group(1), m_job.group(2)
            if not self._authorize(principal, role="propose", tenant_id=tid):
                return
            try:
                job = mgr.get_study_job(jid)
                if job.tenant_id != tid:
                    raise FileNotFoundError(f"Unknown study job '{jid}' for tenant '{tid}'")
                self._send_json(HTTPStatus.OK, job.to_dict())
            except FileNotFoundError as exc:
                self._send_json(HTTPStatus.NOT_FOUND, {"error": str(exc)})
            return

        self._send_json(HTTPStatus.NOT_FOUND, {"error": f"Unknown route: {path}"})

    def do_POST(self) -> None:  # noqa: N802
        principal = self._authenticate()
        if principal is None:
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
                # Missing or blank seat header -> the default seat, so a scoped gate never
                # widens to "all seats" over HTTP (see RouterGate.matches_seat).
                seat = (self.headers.get("X-CaPheOS-Seat") or "").strip() or "intern"
                tenant, use_base_only = mgr.resolve_tenant_from_model(
                    None if model_field is None else str(model_field),
                    header_tenant,
                )
                if not self._authorize(
                    principal,
                    role="chat",
                    tenant_id=None if (use_base_only or tenant is None) else tenant.tenant_id,
                ):
                    return
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
                        stream_kwargs: dict[str, Any] = {
                            "messages": messages,
                            "tools": None,
                            "router": active_router,
                            "registry": active_registry,
                            "max_tokens": max_tokens,
                            "enable_thinking": enable_thinking,
                            "top_k": top_k,
                        }
                        if "seat" in inspect.signature(stream_fn).parameters:
                            stream_kwargs["seat"] = seat
                        decision, token_iter = stream_fn(**stream_kwargs)
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
            except FileNotFoundError as exc:
                self._send_json(HTTPStatus.NOT_FOUND, {"error": {"message": str(exc)}})
                return
            except (ValueError, KeyError, TypeError, IntegrityError) as exc:
                self._send_json(HTTPStatus.BAD_REQUEST, {"error": {"message": str(exc)}})
                return
            except RuntimeError as exc:
                self._send_json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": {"message": str(exc)}})
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

        # 2. Ingest trajectories for a tenant (verified records require `verify` role).
        m_traj = re.match(r"^/v1/subcor/tenants/([^/]+)/trajectories$", path)
        if m_traj:
            tid = m_traj.group(1)
            if not self._authorize(principal, role="ingest", tenant_id=tid):
                return
            try:
                recs = body.get("trajectories")
                if not isinstance(recs, list):
                    raise ValueError("trajectories must be a list")
                has_verified = any(
                    bool((item.get("evidence") or {}).get("verified", False))
                    for item in recs
                    if isinstance(item, dict)
                )
                if has_verified and not principal.has_role("verify"):
                    self._send_json(
                        HTTPStatus.FORBIDDEN,
                        {
                            "error": (
                                f"Actor '{principal.actor}' lacks 'verify' role required to "
                                "ingest verified trajectories"
                            )
                        },
                    )
                    return
                # Bind verifier identity when non-admin caller submits verified evidence.
                if has_verified and not principal.is_admin:
                    sanitized: list[dict[str, Any]] = []
                    for item in recs:
                        copy_item = dict(item)
                        ev = dict(copy_item.get("evidence") or {})
                        if ev.get("verified"):
                            ev["verifier"] = principal.actor
                        copy_item["evidence"] = ev
                        sanitized.append(copy_item)
                    recs = sanitized
                summary = mgr.ingest_trajectories(tid, recs)
                self._send_json(HTTPStatus.OK, summary)
            except FileNotFoundError as exc:
                self._send_json(HTTPStatus.NOT_FOUND, {"error": str(exc)})
            except (ValueError, KeyError, TypeError) as exc:
                self._send_json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
            return

        # 3. Trigger a governed Study Cycle (synchronous or non-blocking async: true).
        m_study = re.match(r"^/v1/subcor/tenants/([^/]+)/study$", path)
        if m_study:
            tid = m_study.group(1)
            if not self._authorize(principal, role="propose", tenant_id=tid):
                return
            try:
                # Non-admin principals cannot spoof proposer identity.
                proposer = (
                    str(body.get("proposer", principal.actor))
                    if principal.is_admin
                    else principal.actor
                )
                study_kwargs: dict[str, Any] = {
                    "unit_id": str(body["unit_id"]),
                    "version": str(body.get("version", "1.0.0")),
                    "task_family": str(body["task_family"]),
                    "title": str(body.get("title", body["unit_id"])),
                    "summary": str(body.get("summary", "Study cycle Skill Unit")),
                    "proposer": proposer,
                    "system_prompt": body.get("system_prompt"),
                    "eval_cases": body.get("eval_cases"),
                    "baseline_cases": body.get("baseline_cases"),
                    "negative_prompts": body.get("negative_prompts"),
                    "rank": int(body.get("rank", 8)),
                    "sub1bit_rank": int(body.get("sub1bit_rank", 8)),
                    "steps": int(body.get("steps", 108)),
                    "lr": float(body.get("lr", 3e-4)),
                    "max_regression_tolerance": _optional_float(
                        body.get("max_regression_tolerance")
                    ),
                    "min_target_improvement": _optional_float(body.get("min_target_improvement")),
                    "seat_scope": body.get("seat_scope"),
                    "num_prototypes": int(body.get("num_prototypes", 1)),
                    "seed": int(body.get("seed", 42)),
                }
                # Reject looser or invalid thresholds before any work is queued (HTTP 400).
                mgr.conformance_policy.resolve(
                    study_kwargs["max_regression_tolerance"], study_kwargs["min_target_improvement"]
                )
                if bool(body.get("async", False)):
                    job = mgr.submit_study_job(tid, **study_kwargs)
                    self._send_json(
                        HTTPStatus.ACCEPTED,
                        {
                            **job.to_dict(),
                            "poll_url": f"/v1/subcor/tenants/{tid}/study/jobs/{job.job_id}",
                        },
                    )
                else:
                    result = mgr.run_study_cycle(tid, **study_kwargs)
                    self._send_json(HTTPStatus.OK, result)
            except FileNotFoundError as exc:
                self._send_json(HTTPStatus.NOT_FOUND, {"error": str(exc)})
            except (
                KeyError,
                ValueError,
                TypeError,
                IntegrityError,
                VerificationError,
                GovernanceError,
            ) as exc:
                self._send_json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
            return

        # 4. Promote a CONFORMANT unit (enforces no self-approval & binds approver to credential).
        m_promote = re.match(r"^/v1/subcor/tenants/([^/]+)/units/([^/]+)/promote$", path)
        if m_promote:
            tid, uid = m_promote.group(1), m_promote.group(2)
            if not self._authorize(principal, role="approve", tenant_id=tid):
                return
            try:
                approver = (
                    str(body.get("approver", principal.actor))
                    if principal.is_admin
                    else principal.actor
                )
                manifest = mgr.promote_unit(tid, uid, approver=approver)
                self._send_json(HTTPStatus.OK, {"manifest": manifest.to_dict()})
            except GovernanceError as exc:
                self._send_json(HTTPStatus.FORBIDDEN, {"error": str(exc)})
            except FileNotFoundError as exc:
                self._send_json(HTTPStatus.NOT_FOUND, {"error": str(exc)})
            except (ValueError, TypeError, IntegrityError) as exc:
                self._send_json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
            return

        # 5. Atomic rollback of an ENABLED unit.
        m_rb = re.match(r"^/v1/subcor/tenants/([^/]+)/units/([^/]+)/rollback$", path)
        if m_rb:
            tid, uid = m_rb.group(1), m_rb.group(2)
            if not self._authorize(principal, role="rollback", tenant_id=tid):
                return
            try:
                actor = (
                    str(body.get("actor", principal.actor))
                    if principal.is_admin
                    else principal.actor
                )
                manifest = mgr.rollback_unit(
                    tid,
                    uid,
                    actor=actor,
                    reason=str(body.get("reason", "")),
                )
                self._send_json(HTTPStatus.OK, {"manifest": manifest.to_dict()})
            except GovernanceError as exc:
                self._send_json(HTTPStatus.CONFLICT, {"error": str(exc)})
            except FileNotFoundError as exc:
                self._send_json(HTTPStatus.NOT_FOUND, {"error": str(exc)})
            except (ValueError, TypeError) as exc:
                self._send_json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
            return

        # 6. Export a portable CaPheOS format-2 bundle (strictly sandboxed inside export_root).
        m_exp = re.match(r"^/v1/subcor/tenants/([^/]+)/units/([^/]+)/export-capheos$", path)
        if m_exp:
            tid, uid = m_exp.group(1), m_exp.group(2)
            if not self._authorize(principal, role="export", tenant_id=tid):
                return
            try:
                ctx = mgr.get_tenant(tid)
                req_dest = body.get("destination")
                dest = _resolve_sandboxed_export_dir(
                    self.server.export_root / ctx.tenant_id,
                    None if req_dest is None else str(req_dest),
                )
                out_dir = ctx.registry.export_capheos_unit(uid, dest)
                self._send_json(HTTPStatus.OK, {"exported_path": str(out_dir)})
            except GovernanceError as exc:
                self._send_json(HTTPStatus.CONFLICT, {"error": str(exc)})
            except FileNotFoundError as exc:
                self._send_json(HTTPStatus.NOT_FOUND, {"error": str(exc)})
            except (ValueError, TypeError, IntegrityError) as exc:
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
