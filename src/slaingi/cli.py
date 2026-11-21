"""Command-line interface for SLAINGI."""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

from slaingi.proving_ground import EvalMetrics, ProvingGroundReport, run_proving_ground
from slaingi.units import SkillUnitRegistry


def _fmt_metrics(m: EvalMetrics) -> str:
    return (
        f"acc={m.accuracy * 100:5.1f}%  "
        f"loop_fail={m.step_limit_rate * 100:5.1f}%  "
        f"avg_steps={m.mean_steps:4.1f}"
    )


def _print_report(report: ProvingGroundReport) -> None:
    print("=========================================================================")
    print(" SLAINGI — Milestone 1 Proving Ground Report")
    print("=========================================================================")
    print("\n[0] Baseline (Frozen Base Kernel)")
    print(f"    Family A Clean : {_fmt_metrics(report.baseline_clean)}")
    print(f"    Family A Loop  : {_fmt_metrics(report.baseline_loop)}")
    print(f"    Family B Novel : {_fmt_metrics(report.baseline_novel)}")
    print(
        f"    Verified Gate  : filtered {report.unverified_filtered_count} unverified trajectories"
    )

    print("\n[1] After Cycle 1 — Zero-Gradient Weight-Space Surgery")
    print(f"    Detected Layer : {report.detected_loop_layer}")
    print(
        f"    Skill Unit     : {report.surgery_unit.unit_id}@{report.surgery_unit.version} "
        f"({report.surgery_unit.effective_bits_per_weight:.3f} bits/weight, "
        f"{report.surgery_unit.payload_bytes} B, sha256={report.surgery_unit.payload_sha256[:12]})"
    )
    print(f"    Family A Clean : {_fmt_metrics(report.after_surgery_clean)}")
    print(f"    Family A Loop  : {_fmt_metrics(report.after_surgery_loop)}")
    print(f"    Family B Novel : {_fmt_metrics(report.after_surgery_novel)}")

    print("\n[2] After Cycle 2 — Dynamic Low-Rank Expansion + Binary STE + Sub-1-Bit")
    print(f"    Zero-Perturb   : max|y_after - y_before| = {report.zero_perturbation_max_diff:.1e}")
    print(
        f"    Skill Unit     : {report.skill_unit.unit_id}@{report.skill_unit.version} "
        f"({report.skill_unit.effective_bits_per_weight:.3f} bits/weight, "
        f"{report.skill_unit.payload_bytes} B, sha256={report.skill_unit.payload_sha256[:12]})"
    )
    print(f"    Family A Clean : {_fmt_metrics(report.after_expansion_clean)}")
    print(f"    Family A Loop  : {_fmt_metrics(report.after_expansion_loop)}")
    print(f"    Family B Novel : {_fmt_metrics(report.after_expansion_novel)}")

    print("\n[3] Governance, Rejection & Atomic Rollback Verification")
    print(f"    No Self-Approval Blocked : {report.self_approval_blocked}")
    print(f"    Poisoned Patch Rejected  : {report.poisoned_patch_rejected}")
    print("    After Rollback of novel-skill-v1:")
    print(f"    Family A Clean : {_fmt_metrics(report.after_rollback_clean)}")
    print(f"    Family A Loop  : {_fmt_metrics(report.after_rollback_loop)}")
    print(f"    Family B Novel : {_fmt_metrics(report.after_rollback_novel)}")

    if report.exported_capheos_paths:
        print("\n[4] Exported CaPheOS Format-2 Unit Bundles")
        for path in report.exported_capheos_paths:
            print(f"    - {path}")
    print("=========================================================================")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="slaingi",
        description="SLAINGI — Governed self-learning runtime over frozen quantized kernels.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    demo_cmd = sub.add_parser(
        "demo",
        help="Run the end-to-end Micro-Intern proving ground (surgery, STE expansion, sub-1-bit, governance).",
    )
    demo_cmd.add_argument(
        "--work-dir",
        type=Path,
        default=None,
        help="Directory to persist telemetry and Skill Units (defaults to a temporary directory).",
    )
    demo_cmd.add_argument(
        "--export-capheos",
        type=Path,
        default=None,
        help="Optional directory to export promoted Skill Units as CaPheOS format-2 module.json bundles.",
    )
    demo_cmd.add_argument("--seed", type=int, default=42, help="Deterministic RNG seed.")

    inspect_cmd = sub.add_parser(
        "inspect",
        help="Inspect registered Skill Units and verify SHA-256 integrity in a unit store.",
    )
    inspect_cmd.add_argument("units_dir", type=Path, help="Path to a SLAINGI units directory.")

    export_cmd = sub.add_parser(
        "export-capheos",
        help="Export a CONFORMANT or ENABLED Skill Unit as a CaPheOS format-2 bundle.",
    )
    export_cmd.add_argument("units_dir", type=Path, help="Path to a SLAINGI units directory.")
    export_cmd.add_argument("unit_id", type=str, help="Unit ID to export.")
    export_cmd.add_argument("destination", type=Path, help="Destination directory.")

    args = parser.parse_args(argv)

    if args.command == "demo":
        if args.work_dir is None:
            with tempfile.TemporaryDirectory(prefix="slaingi-demo-") as tmp:
                report = run_proving_ground(
                    Path(tmp),
                    export_capheos_dir=args.export_capheos,
                    seed=args.seed,
                )
        else:
            report = run_proving_ground(
                args.work_dir,
                export_capheos_dir=args.export_capheos,
                seed=args.seed,
            )
        _print_report(report)
        return 0

    if args.command == "inspect":
        registry = SkillUnitRegistry(args.units_dir)
        rows = []
        for m in registry.list_units():
            registry.load_verified_deltas(m.unit_id)
            rows.append(
                {
                    "unit_id": m.unit_id,
                    "version": m.version,
                    "state": m.state.value,
                    "mode": m.mode.value,
                    "bits_per_weight": round(m.effective_bits_per_weight, 4),
                    "payload_bytes": m.payload_bytes,
                    "sha256": m.payload_sha256[:16],
                    "integrity": "verified",
                }
            )
        print(json.dumps(rows, indent=2))
        return 0

    if args.command == "export-capheos":
        registry = SkillUnitRegistry(args.units_dir)
        out = registry.export_capheos_unit(args.unit_id, args.destination)
        print(f"Exported CaPheOS unit bundle to: {out}")
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
