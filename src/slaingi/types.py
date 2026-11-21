"""Core domain types, lifecycle states, and error hierarchy for SLAINGI."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any


class SlaingiError(Exception):
    """Base exception for all SLAINGI runtime errors."""


class GovernanceError(SlaingiError):
    """Raised when a governance invariant (such as no self-approval) is violated."""


class IntegrityError(SlaingiError):
    """Raised when content-addressed digest verification fails."""


class VerificationError(SlaingiError):
    """Raised when unverified data is submitted where verified evidence is required."""


class UnitState(StrEnum):
    """Lifecycle states of a governed Skill Unit (aligned with CaPheOS §5)."""

    PROPOSED = "proposed"
    DORMANT = "dormant"
    CONFORMANT = "conformant"
    ENABLED = "enabled"
    ROLLED_BACK = "rolled_back"


class DeltaEncoding(StrEnum):
    """Supported wire representations for a LayerDelta payload."""

    LOW_RANK_FP32 = "low_rank_fp32"
    SUB1BIT_BINARY = "sub1bit_binary"


class AdaptationMode(StrEnum):
    """Origin mechanism that produced a Skill Unit."""

    WEIGHT_SPACE_SURGERY = "weight_space_surgery"
    DYNAMIC_EXPANSION_FP32 = "dynamic_expansion_fp32"
    DYNAMIC_EXPANSION_STE = "dynamic_expansion_ste"
    MANUAL_TEST = "manual_test"


@dataclass(frozen=True)
class VerificationEvidence:
    """Immutable proof that a trajectory's outcome or correction was verified."""

    verified: bool
    verifier: str
    method: str  # e.g. "deterministic_verifier", "senior_seat_review", "owner_signoff"
    notes: str = ""
    timestamp: str = field(default_factory=lambda: datetime.now(UTC).isoformat())

    def to_dict(self) -> dict[str, Any]:
        return {
            "verified": self.verified,
            "verifier": self.verifier,
            "method": self.method,
            "notes": self.notes,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> VerificationEvidence:
        return cls(
            verified=bool(data["verified"]),
            verifier=str(data["verifier"]),
            method=str(data["method"]),
            notes=str(data.get("notes", "")),
            timestamp=str(data.get("timestamp", datetime.now(UTC).isoformat())),
        )


@dataclass(frozen=True)
class TrajectoryRecord:
    """Single Intern execution attempt recorded in append-only telemetry."""

    trajectory_id: str
    task_family: str
    seat: str
    input_vector: list[float]
    predicted_label: int
    expected_label: int | None
    steps_taken: int
    hit_step_limit: bool
    succeeded: bool
    evidence: VerificationEvidence
    layer_activations: dict[str, list[float]] = field(default_factory=dict)
    senior_correction_label: int | None = None
    timestamp: str = field(default_factory=lambda: datetime.now(UTC).isoformat())

    @property
    def is_verified(self) -> bool:
        return self.evidence.verified

    def to_dict(self) -> dict[str, Any]:
        return {
            "trajectory_id": self.trajectory_id,
            "task_family": self.task_family,
            "seat": self.seat,
            "input_vector": self.input_vector,
            "predicted_label": self.predicted_label,
            "expected_label": self.expected_label,
            "steps_taken": self.steps_taken,
            "hit_step_limit": self.hit_step_limit,
            "succeeded": self.succeeded,
            "evidence": self.evidence.to_dict(),
            "layer_activations": self.layer_activations,
            "senior_correction_label": self.senior_correction_label,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TrajectoryRecord:
        return cls(
            trajectory_id=str(data["trajectory_id"]),
            task_family=str(data["task_family"]),
            seat=str(data["seat"]),
            input_vector=[float(v) for v in data["input_vector"]],
            predicted_label=int(data["predicted_label"]),
            expected_label=(
                None if data.get("expected_label") is None else int(data["expected_label"])
            ),
            steps_taken=int(data["steps_taken"]),
            hit_step_limit=bool(data["hit_step_limit"]),
            succeeded=bool(data["succeeded"]),
            evidence=VerificationEvidence.from_dict(data["evidence"]),
            layer_activations={
                str(k): [float(x) for x in v] for k, v in data.get("layer_activations", {}).items()
            },
            senior_correction_label=(
                None
                if data.get("senior_correction_label") is None
                else int(data["senior_correction_label"])
            ),
            timestamp=str(data.get("timestamp", datetime.now(UTC).isoformat())),
        )


@dataclass(frozen=True)
class ConformanceReport:
    """Result of running the locked conformance gate on a proposed Skill Unit."""

    unit_id: str
    passed: bool
    baseline_before: float
    baseline_after: float
    target_before: float
    target_after: float
    max_regression_tolerance: float
    min_target_improvement: float
    reason: str
    evaluator: str = "locked_conformance_gate"
    timestamp: str = field(default_factory=lambda: datetime.now(UTC).isoformat())

    def to_dict(self) -> dict[str, Any]:
        return {
            "unit_id": self.unit_id,
            "passed": self.passed,
            "baseline_before": self.baseline_before,
            "baseline_after": self.baseline_after,
            "target_before": self.target_before,
            "target_after": self.target_after,
            "max_regression_tolerance": self.max_regression_tolerance,
            "min_target_improvement": self.min_target_improvement,
            "reason": self.reason,
            "evaluator": self.evaluator,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ConformanceReport:
        return cls(
            unit_id=str(data["unit_id"]),
            passed=bool(data["passed"]),
            baseline_before=float(data["baseline_before"]),
            baseline_after=float(data["baseline_after"]),
            target_before=float(data["target_before"]),
            target_after=float(data["target_after"]),
            max_regression_tolerance=float(data["max_regression_tolerance"]),
            min_target_improvement=float(data["min_target_improvement"]),
            reason=str(data["reason"]),
            evaluator=str(data.get("evaluator", "locked_conformance_gate")),
            timestamp=str(data.get("timestamp", datetime.now(UTC).isoformat())),
        )


@dataclass(frozen=True)
class SkillUnitManifest:
    """Metadata manifest for a versioned, content-addressed SLAINGI Skill Unit."""

    unit_id: str
    version: str
    title: str
    summary: str
    proposer: str
    mode: AdaptationMode
    encoding: DeltaEncoding
    target_layers: tuple[str, ...]
    payload_sha256: str
    payload_bytes: int
    effective_bits_per_weight: float
    state: UnitState = UnitState.DORMANT
    approver: str | None = None
    license: str = "Apache-2.0"
    created_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    conformance: ConformanceReport | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "unit_id": self.unit_id,
            "version": self.version,
            "title": self.title,
            "summary": self.summary,
            "proposer": self.proposer,
            "mode": self.mode.value,
            "encoding": self.encoding.value,
            "target_layers": list(self.target_layers),
            "payload_sha256": self.payload_sha256,
            "payload_bytes": self.payload_bytes,
            "effective_bits_per_weight": self.effective_bits_per_weight,
            "state": self.state.value,
            "approver": self.approver,
            "license": self.license,
            "created_at": self.created_at,
            "conformance": None if self.conformance is None else self.conformance.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SkillUnitManifest:
        return cls(
            unit_id=str(data["unit_id"]),
            version=str(data["version"]),
            title=str(data["title"]),
            summary=str(data["summary"]),
            proposer=str(data["proposer"]),
            mode=AdaptationMode(data["mode"]),
            encoding=DeltaEncoding(data["encoding"]),
            target_layers=tuple(str(x) for x in data["target_layers"]),
            payload_sha256=str(data["payload_sha256"]),
            payload_bytes=int(data["payload_bytes"]),
            effective_bits_per_weight=float(data["effective_bits_per_weight"]),
            state=UnitState(data["state"]),
            approver=data.get("approver"),
            license=str(data.get("license", "Apache-2.0")),
            created_at=str(data.get("created_at", datetime.now(UTC).isoformat())),
            conformance=(
                None
                if data.get("conformance") is None
                else ConformanceReport.from_dict(data["conformance"])
            ),
        )
