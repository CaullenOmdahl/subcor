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


def _run_mlx_intern_cli(args: argparse.Namespace) -> int:
    try:
        from slaingi.live_loop import run_live_intern_distillation
    except ImportError as exc:
        raise SystemExit(
            "MLX extra is required for 'slaingi mlx-intern'. "
            "Run with: uv run --extra mlx slaingi mlx-intern"
        ) from exc

    def execute(work_dir: Path) -> None:
        report = run_live_intern_distillation(
            work_dir,
            model_id=args.model,
            rank=args.rank,
            steps=args.steps,
            lr=args.lr,
            sub1bit_rank=args.sub1bit_rank,
            export_capheos_dir=args.export_capheos,
            export_safetensors_dir=args.export_safetensors,
            seed=args.seed,
        )
        print("=========================================================================")
        print(f" SLAINGI — Live Apple Silicon MLX Intern Report ({report.model_id})")
        print("=========================================================================")
        print(f"Target Layers      : {', '.join(report.target_layers)}")
        print(f"Verified Gate      : filtered {report.unverified_filtered_count} unverified traces")
        print(
            f"Zero-Perturbation  : max|logits_after - logits_before| = "
            f"{report.zero_perturbation_max_logit_diff:.1e}"
        )
        print(f"QLoRA Loss Curve   : {report.initial_loss:.4f} -> {report.final_loss:.4f}")
        print(
            f"Skill Unit         : {report.unit.unit_id}@{report.unit.version} "
            f"state={report.unit.state.value} "
            f"({report.unit.effective_bits_per_weight:.4f} bits/weight, "
            f"{report.unit.payload_bytes / 1024:.1f} KiB, "
            f"sha256={report.unit.payload_sha256[:12]})"
        )
        print("\nHeld-Out Metrics:")
        print(
            f"  Baseline Frozen  : Ops Router={report.baseline_ops_acc * 100:5.1f}%  "
            f"General QA={report.baseline_general_acc * 100:5.1f}%"
        )
        print(
            f"  Sub-1-Bit Unit   : Ops Router={report.sub1bit_ops_acc * 100:5.1f}%  "
            f"General QA={report.sub1bit_general_acc * 100:5.1f}%"
        )
        print(
            f"  After Rollback   : Ops Router={report.rollback_ops_acc * 100:5.1f}%  "
            f"General QA={report.rollback_general_acc * 100:5.1f}%  "
            f"(max_logit_diff={report.rollback_max_logit_diff:.1e})"
        )
        print(f"\nGovernance         : self_approval_blocked={report.self_approval_blocked}")
        if report.unit.conformance is not None:
            print(f"Conformance Gate   : {report.unit.conformance.reason}")
        if report.exported_capheos_dir is not None:
            print(f"CaPheOS Export     : {report.exported_capheos_dir}")
        if report.exported_safetensors_dir is not None:
            print(f"MLX LoRA Export    : {report.exported_safetensors_dir}")
        print("=========================================================================")

    if args.work_dir is None:
        with tempfile.TemporaryDirectory(prefix="slaingi-mlx-intern-") as tmp:
            execute(Path(tmp))
    else:
        execute(args.work_dir)
    return 0


def _run_mlx_surgery_cli(args: argparse.Namespace) -> int:
    try:
        from slaingi.live_loop import run_live_weight_surgery
    except ImportError as exc:
        raise SystemExit(
            "MLX extra is required for 'slaingi mlx-surgery'. "
            "Run with: uv run --extra mlx slaingi mlx-surgery"
        ) from exc

    def execute(work_dir: Path) -> None:
        report = run_live_weight_surgery(
            work_dir,
            model_id=args.model,
            max_tokens=args.max_tokens,
        )
        print("=========================================================================")
        print(f" SLAINGI — Live Zero-Gradient Weight-Space Surgery ({report.model_id})")
        print("=========================================================================")
        print("Contrastive Layer Ranking (zero gradient descent):")
        for name, score in report.ranked_layers:
            marker = " <-- peak" if name == report.selected_layer else ""
            print(f"  {name:<26} separation={score:.4f}{marker}")
        print(
            f"\nBest Calibration   : layer={report.selected_layer} alpha={report.selected_alpha:.2f}"
        )
        print(
            f"Token-Limit Rate   : {report.baseline_limit_rate * 100:5.1f}% -> "
            f"{report.patched_limit_rate * 100:5.1f}% "
            f"(avg tokens {report.baseline_mean_tokens:.1f} -> {report.patched_mean_tokens:.1f})"
        )
        print(
            f"Skill Unit         : {report.unit.unit_id}@{report.unit.version} "
            f"state={report.unit.state.value} "
            f"({report.unit.effective_bits_per_weight:.4f} bits/weight, "
            f"{report.unit.payload_bytes / 1024:.1f} KiB)"
        )
        if report.unit.conformance is not None:
            print(f"Conformance Gate   : {report.unit.conformance.reason}")
        print("=========================================================================")

    if args.work_dir is None:
        with tempfile.TemporaryDirectory(prefix="slaingi-mlx-surgery-") as tmp:
            execute(Path(tmp))
    else:
        execute(args.work_dir)
    return 0


def _run_mlx_moe_cli(args: argparse.Namespace) -> int:
    try:
        from slaingi.live_loop import run_live_moe_multi_skill
    except ImportError as exc:
        raise SystemExit(
            "MLX extra is required for 'slaingi mlx-moe'. "
            "Run with: uv run --extra mlx slaingi mlx-moe"
        ) from exc

    def execute(work_dir: Path) -> None:
        report = run_live_moe_multi_skill(
            work_dir,
            model_id=args.model,
            rank=args.rank,
            sub1bit_rank=args.sub1bit_rank,
            steps_per_skill=args.steps,
            lr=args.lr,
            export_capheos_dir=args.export_capheos,
            seed=args.seed,
        )
        print("=========================================================================")
        print(f" SLAINGI — Routed MoE Skill Layer Report ({report.model_id})")
        print("=========================================================================")
        print(f"Routing Tap Layer  : {report.tap_layer} (invariant frozen shared core)")
        print(
            f"Expert 1 (Ops)     : {report.ops_unit.unit_id}@{report.ops_unit.version} "
            f"state={report.ops_unit.state.value} "
            f"({report.ops_unit.effective_bits_per_weight:.4f} bpw, "
            f"{report.ops_unit.payload_bytes / 1024:.1f} KiB)"
        )
        print(
            f"Expert 2 (NOC SLA) : {report.noc_unit.unit_id}@{report.noc_unit.version} "
            f"state={report.noc_unit.state.value} "
            f"({report.noc_unit.effective_bits_per_weight:.4f} bpw, "
            f"{report.noc_unit.payload_bytes / 1024:.1f} KiB)"
        )
        print("\nHeld-Out 3-Domain Evaluation (Unseen Prompts):")
        print(
            f"  Ops Router       : {report.baseline_ops_acc * 100:5.1f}% -> "
            f"{report.routed_ops_acc * 100:5.1f}%  "
            f"(routed to Expert 1: {report.ops_route_acc * 100:5.1f}%)"
        )
        print(
            f"  NOC Incident SLA : {report.baseline_noc_acc * 100:5.1f}% -> "
            f"{report.routed_noc_acc * 100:5.1f}%  "
            f"(routed to Expert 2: {report.noc_route_acc * 100:5.1f}%)"
        )
        print(
            f"  General QA       : {report.baseline_general_acc * 100:5.1f}% -> "
            f"{report.routed_general_acc * 100:5.1f}%  "
            f"(routed to Base Core: {report.general_base_route_acc * 100:5.1f}%)"
        )
        if report.exported_capheos_dirs:
            print("\nExported CaPheOS Format-2 Bundles (weights + router_gate.json):")
            for p in report.exported_capheos_dirs:
                print(f"  - {p}")
        print("=========================================================================")

    if args.work_dir is None:
        with tempfile.TemporaryDirectory(prefix="slaingi-mlx-moe-") as tmp:
            execute(Path(tmp))
    else:
        execute(args.work_dir)
    return 0


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

    mlx_intern_cmd = sub.add_parser(
        "mlx-intern",
        help="Run the live Apple Silicon MLX On-Policy Intern distillation + Sub-1-Bit Unit cycle.",
    )
    mlx_intern_cmd.add_argument(
        "--model",
        type=str,
        default="mlx-community/Qwen3.5-0.8B-4bit",
        help="Compact Hugging Face MLX model ID.",
    )
    mlx_intern_cmd.add_argument("--work-dir", type=Path, default=None)
    mlx_intern_cmd.add_argument("--rank", type=int, default=8)
    mlx_intern_cmd.add_argument("--sub1bit-rank", type=int, default=8)
    mlx_intern_cmd.add_argument("--steps", type=int, default=108)
    mlx_intern_cmd.add_argument("--lr", type=float, default=3e-4)
    mlx_intern_cmd.add_argument("--seed", type=int, default=42)
    mlx_intern_cmd.add_argument("--export-capheos", type=Path, default=None)
    mlx_intern_cmd.add_argument("--export-safetensors", type=Path, default=None)

    mlx_surgery_cmd = sub.add_parser(
        "mlx-surgery",
        help="Run live zero-gradient contrastive weight-space diagnosis & ablation on an MLX model.",
    )
    mlx_surgery_cmd.add_argument(
        "--model",
        type=str,
        default="mlx-community/Qwen3.5-0.8B-4bit",
        help="Compact Hugging Face MLX model ID.",
    )
    mlx_surgery_cmd.add_argument("--work-dir", type=Path, default=None)
    mlx_surgery_cmd.add_argument("--max-tokens", type=int, default=160)

    mlx_moe_cmd = sub.add_parser(
        "mlx-moe",
        help="Train & route two independent Sub-1-Bit Skill Units over one frozen 0.8B core with 0% general QA regression.",
    )
    mlx_moe_cmd.add_argument("--model", type=str, default="mlx-community/Qwen3.5-0.8B-4bit")
    mlx_moe_cmd.add_argument("--work-dir", type=Path, default=None)
    mlx_moe_cmd.add_argument("--rank", type=int, default=8)
    mlx_moe_cmd.add_argument("--sub1bit-rank", type=int, default=8)
    mlx_moe_cmd.add_argument("--steps", type=int, default=90)
    mlx_moe_cmd.add_argument("--lr", type=float, default=3e-4)
    mlx_moe_cmd.add_argument("--seed", type=int, default=42)
    mlx_moe_cmd.add_argument("--export-capheos", type=Path, default=None)

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

    if args.command == "mlx-intern":
        return _run_mlx_intern_cli(args)

    if args.command == "mlx-surgery":
        return _run_mlx_surgery_cli(args)

    if args.command == "mlx-moe":
        return _run_mlx_moe_cli(args)

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
