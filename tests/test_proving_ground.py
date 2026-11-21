"""End-to-end test for the Milestone 1 Micro-Intern Proving Ground."""

from __future__ import annotations

import subprocess
from pathlib import Path

from slaingi.proving_ground import run_proving_ground
from slaingi.types import UnitState


def test_proving_ground_end_to_end(tmp_path: Path) -> None:
    export_dir = tmp_path / "capheos-units"
    report = run_proving_ground(
        tmp_path / "work",
        export_capheos_dir=export_dir,
        seed=42,
    )

    # 0. Baseline exhibits healthy clean tasks, severe loop defect, and missing novel skill.
    assert report.baseline_clean.accuracy >= 0.90
    assert report.baseline_loop.step_limit_rate >= 0.85
    assert report.baseline_loop.accuracy <= 0.15
    assert report.baseline_novel.accuracy <= 0.45
    assert report.unverified_filtered_count == 12

    # 1. Cycle 1: Zero-gradient weight-space surgery fixes layer_1 loop without losing clean accuracy.
    assert report.detected_loop_layer == "layer_1"
    assert report.after_surgery_loop.step_limit_rate <= 0.05
    assert report.after_surgery_loop.accuracy >= 0.90
    assert report.after_surgery_clean.accuracy >= 0.90
    assert report.surgery_unit.state is UnitState.ENABLED
    assert report.surgery_unit.effective_bits_per_weight < 1.0

    # 2. Cycle 2: Zero-perturbation dynamic expansion + Binary STE learns Family B at < 1 bit/weight.
    assert report.zero_perturbation_max_diff == 0.0
    assert report.after_expansion_novel.accuracy >= 0.85
    assert report.after_expansion_clean.accuracy >= 0.90
    assert report.after_expansion_loop.accuracy >= 0.90
    assert report.skill_unit.state is UnitState.ENABLED
    assert report.skill_unit.effective_bits_per_weight < 1.0

    # 3. Governance enforcement and exact rollback restoration.
    assert report.self_approval_blocked is True
    assert report.poisoned_patch_rejected is True
    assert report.after_rollback_clean == report.after_surgery_clean
    assert report.after_rollback_loop == report.after_surgery_loop
    assert report.after_rollback_novel == report.after_surgery_novel

    # 4. Both exported CaPheOS format-2 bundles pass their Node conformance tests.
    assert len(report.exported_capheos_paths) == 2
    for bundle in report.exported_capheos_paths:
        proc = subprocess.run(
            ["node", str(bundle / "conformance" / "test.js")],
            capture_output=True,
            text=True,
            check=False,
        )
        assert proc.returncode == 0, proc.stderr
