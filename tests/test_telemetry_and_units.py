"""Tests for Verified Evidence Gate telemetry and Governed Skill Unit Registry."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np
import pytest

from subcor.expansion import LowRankAdapter
from subcor.telemetry import TrajectoryStore
from subcor.types import (
    AdaptationMode,
    DeltaEncoding,
    GovernanceError,
    IntegrityError,
    TrajectoryRecord,
    UnitState,
    VerificationError,
    VerificationEvidence,
)
from subcor.units import SkillUnitRegistry, build_layer_delta


def test_trajectory_store_enforces_verified_evidence_gate(tmp_path: Path) -> None:
    store = TrajectoryStore(tmp_path / "trajectories.jsonl")

    verified_rec = TrajectoryRecord(
        trajectory_id="t-1",
        task_family="triage",
        seat="intern",
        input_vector=[1.0, 0.0, 0.0, 0.0],
        predicted_label=0,
        expected_label=1,
        steps_taken=2,
        hit_step_limit=False,
        succeeded=False,
        evidence=VerificationEvidence(
            verified=True,
            verifier="senior_ceo",
            method="senior_seat_review",
        ),
        senior_correction_label=1,
    )
    unverified_rec = TrajectoryRecord(
        trajectory_id="t-2",
        task_family="triage",
        seat="intern",
        input_vector=[9.0, 9.0, 9.0, 9.0],
        predicted_label=2,
        expected_label=3,
        steps_taken=1,
        hit_step_limit=False,
        succeeded=False,
        evidence=VerificationEvidence(
            verified=False,
            verifier="none",
            method="unverified_draft",
        ),
        senior_correction_label=3,
    )

    store.append_many([verified_rec, unverified_rec])
    assert len(store.load_all()) == 2
    assert [r.trajectory_id for r in store.load_verified()] == ["t-1"]

    xs, ys = store.verified_correction_dataset(task_family="triage")
    assert xs.shape == (1, 4)
    assert ys.tolist() == [1]

    # Empty verified family raises VerificationError.
    with pytest.raises(VerificationError):
        store.verified_correction_dataset(task_family="missing")


def _sample_delta() -> LowRankAdapter:
    rng = np.random.default_rng(5)
    return LowRankAdapter(
        name="sample",
        a=rng.normal(0.0, 0.5, size=(2, 64)),
        b=rng.normal(0.0, 0.5, size=(64, 2)),
    )


def test_registry_governance_integrity_rollback_and_capheos_export(tmp_path: Path) -> None:
    registry = SkillUnitRegistry(tmp_path / "units")
    delta = build_layer_delta("layer_0", _sample_delta(), encoding=DeltaEncoding.SUB1BIT_BINARY)

    manifest = registry.propose(
        unit_id="skill-alpha",
        version="1.0.0",
        title="Alpha Skill",
        summary="Sub-1-bit test skill unit.",
        proposer="tools",
        mode=AdaptationMode.DYNAMIC_EXPANSION_STE,
        deltas=[delta],
    )
    assert manifest.state is UnitState.DORMANT
    assert manifest.effective_bits_per_weight < 1.0

    # Cannot promote before passing conformance.
    with pytest.raises(GovernanceError, match="conformance"):
        registry.promote("skill-alpha", approver="risk")

    # Failed conformance keeps unit DORMANT and blocks promotion.
    failed = registry.record_conformance(
        "skill-alpha",
        baseline_before=0.95,
        baseline_after=0.80,
        target_before=0.25,
        target_after=0.90,
        max_regression_tolerance=0.02,
    )
    assert failed.state is UnitState.DORMANT
    assert failed.conformance is not None and not failed.conformance.passed
    with pytest.raises(GovernanceError):
        registry.promote("skill-alpha", approver="risk")

    # Passing conformance transitions DORMANT -> CONFORMANT.
    conformant = registry.record_conformance(
        "skill-alpha",
        baseline_before=0.95,
        baseline_after=0.95,
        target_before=0.25,
        target_after=0.92,
        max_regression_tolerance=0.02,
        min_target_improvement=0.10,
    )
    assert conformant.state is UnitState.CONFORMANT

    # Self-approval is strictly forbidden.
    with pytest.raises(GovernanceError, match="Self-approval forbidden"):
        registry.promote("skill-alpha", approver="tools")

    # Independent risk seat promotes CONFORMANT -> ENABLED.
    enabled = registry.promote("skill-alpha", approver="risk")
    assert enabled.state is UnitState.ENABLED
    assert enabled.approver == "risk"

    # Export as CaPheOS format-2 bundle and execute its Node conformance test.
    bundle_dir = registry.export_capheos_unit("skill-alpha", tmp_path / "capheos-export")
    module_json = json.loads((bundle_dir / "module.json").read_text(encoding="utf-8"))
    assert module_json["format"] == 2
    assert module_json["id"] == "subcor-skill-alpha"
    assert module_json["subcor"]["payloadSha256"] == enabled.payload_sha256

    proc = subprocess.run(
        ["node", str(bundle_dir / "conformance" / "test.js")],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    assert "PASS subcor-skill-alpha@1.0.0" in proc.stdout

    # Atomic rollback transitions ENABLED -> ROLLED_BACK and writes audit log.
    rolled = registry.rollback("skill-alpha", actor="risk", reason="Regression drill")
    assert rolled.state is UnitState.ROLLED_BACK
    assert (tmp_path / "units" / "skill-alpha" / "audit.jsonl").exists()

    # Tampering with payload bytes fails closed with IntegrityError.
    payload_path = tmp_path / "units" / "skill-alpha" / "payload.scun"
    raw = bytearray(payload_path.read_bytes())
    raw[-1] ^= 0xFF
    payload_path.write_bytes(bytes(raw))
    with pytest.raises(IntegrityError, match="SHA-256 mismatch"):
        registry.load_verified_deltas("skill-alpha")
