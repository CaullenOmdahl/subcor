"""Command-line interface for SUBCOR."""

from __future__ import annotations

import argparse
import json
import os
import signal
import tempfile
import threading
from pathlib import Path
from typing import TYPE_CHECKING

from subcor.proving_ground import EvalMetrics, ProvingGroundReport, run_proving_ground
from subcor.units import SkillUnitRegistry

if TYPE_CHECKING:
    from subcor.router import StickyRoutingPolicy

_MLX_DEFAULT_MODEL = "mlx-community/Qwen3.5-0.8B-4bit"


def _fmt_metrics(m: EvalMetrics) -> str:
    return (
        f"acc={m.accuracy * 100:5.1f}%  "
        f"loop_fail={m.step_limit_rate * 100:5.1f}%  "
        f"avg_steps={m.mean_steps:4.1f}"
    )


def _print_report(report: ProvingGroundReport) -> None:
    print("=========================================================================")
    print(" SUBCOR — Milestone 1 Proving Ground Report")
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
        from subcor.live_loop import run_live_intern_distillation
    except ImportError as exc:
        raise SystemExit(
            "MLX extra is required for 'subcor mlx-intern'. "
            "Run with: uv run --extra mlx subcor mlx-intern"
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
        print(f" SUBCOR — Live Apple Silicon MLX Intern Report ({report.model_id})")
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
        with tempfile.TemporaryDirectory(prefix="subcor-mlx-intern-") as tmp:
            execute(Path(tmp))
    else:
        execute(args.work_dir)
    return 0


def _run_mlx_surgery_cli(args: argparse.Namespace) -> int:
    try:
        from subcor.live_loop import run_live_weight_surgery
    except ImportError as exc:
        raise SystemExit(
            "MLX extra is required for 'subcor mlx-surgery'. "
            "Run with: uv run --extra mlx subcor mlx-surgery"
        ) from exc

    def execute(work_dir: Path) -> None:
        report = run_live_weight_surgery(
            work_dir,
            model_id=args.model,
            max_tokens=args.max_tokens,
        )
        print("=========================================================================")
        print(f" SUBCOR — Live Zero-Gradient Weight-Space Surgery ({report.model_id})")
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
        with tempfile.TemporaryDirectory(prefix="subcor-mlx-surgery-") as tmp:
            execute(Path(tmp))
    else:
        execute(args.work_dir)
    return 0


def _run_mlx_moe_cli(args: argparse.Namespace) -> int:
    try:
        from subcor.live_loop import run_live_moe_multi_skill
    except ImportError as exc:
        raise SystemExit(
            "MLX extra is required for 'subcor mlx-moe'. "
            "Run with: uv run --extra mlx subcor mlx-moe"
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
        print(f" SUBCOR — Routed MoE Skill Layer Report ({report.model_id})")
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
        with tempfile.TemporaryDirectory(prefix="subcor-mlx-moe-") as tmp:
            execute(Path(tmp))
    else:
        execute(args.work_dir)
    return 0


def _sticky_policy(args: argparse.Namespace) -> StickyRoutingPolicy | None:
    if not getattr(args, "sticky_routing", False):
        return None
    from subcor.router import StickyRoutingPolicy

    return StickyRoutingPolicy(
        switch_margin=args.sticky_switch_margin,
        min_context_tokens=args.sticky_min_tokens,
    )


def _parse_lora_id_map(spec: str | None) -> dict[str, int]:
    """Parse `--lora-id-map`: a JSON object or comma-separated `unit_id=lora_id` pairs."""
    if not spec or not spec.strip():
        return {}
    text = spec.strip()
    try:
        if text.startswith("{"):
            raw = json.loads(text)
            if not isinstance(raw, dict):
                raise ValueError("expected a JSON object")
            items = list(raw.items())
        else:
            items = []
            for part in text.split(","):
                if not part.strip():
                    continue
                unit_id, sep, lora_id = part.partition("=")
                if not sep or not unit_id.strip():
                    raise ValueError(f"expected unit_id=lora_id, got {part.strip()!r}")
                items.append((unit_id.strip(), lora_id.strip()))
        out: dict[str, int] = {}
        for unit_id, lora_id in items:
            if isinstance(lora_id, bool):
                raise ValueError(f"LoRA id for {unit_id!r} must be an integer")
            lid = int(lora_id)
            if lid < 0:
                raise ValueError(f"LoRA id for {unit_id!r} must be >= 0")
            out[str(unit_id)] = lid
        return out
    except (ValueError, TypeError) as exc:
        raise SystemExit(f"Invalid --lora-id-map {spec!r}: {exc}") from exc


def _run_serve_cli(args: argparse.Namespace) -> int:
    """Run `subcor serve`. SIGTERM is turned into KeyboardInterrupt for the WHOLE run.

    The handler is installed before a supervised llama-server is started, so a SIGTERM
    while the model loads or TenantManager hydrates still stops llama-server.
    """

    def _on_sigterm(_signum: int, _frame: object) -> None:
        raise KeyboardInterrupt

    in_main = threading.current_thread() is threading.main_thread()
    previous_sigterm = signal.signal(signal.SIGTERM, _on_sigterm) if in_main else None
    try:
        return _run_serve(args)
    except KeyboardInterrupt:
        print("\nInterrupted; SUBCOR server stopped.")
        return 130
    finally:
        if in_main:
            signal.signal(signal.SIGTERM, previous_sigterm)


def _run_serve(args: argparse.Namespace) -> int:
    from subcor.auth import AuthPolicy
    from subcor.server import MLXServingAdapter, SubcorHTTPServer, TenantManager

    api_key = args.api_key or os.environ.get("SUBCOR_API_KEY")
    if args.auth_config is not None:
        auth_policy = AuthPolicy.from_json_file(args.auth_config.expanduser().resolve())
    elif api_key:
        auth_policy = AuthPolicy.from_single_key(api_key)
    else:
        auth_policy = AuthPolicy.disabled()

    is_loopback = args.host in ("127.0.0.1", "localhost", "::1")
    if not is_loopback and not auth_policy.enabled and not args.allow_unauthenticated:
        raise SystemExit(
            f"Refusing to bind to non-loopback address '{args.host}' without authentication. "
            "Pass --api-key / SUBCOR_API_KEY, --auth-config, or --allow-unauthenticated."
        )

    state_dir = args.state_dir.expanduser().resolve()
    export_root = (
        args.export_root.expanduser().resolve()
        if args.export_root is not None
        else (state_dir / "exports").resolve()
    )
    tenants = tuple(t.strip() for t in (args.tenants or []) if t.strip())
    supervisor = None
    if args.backend == "llamacpp":
        from subcor.backends.llamacpp_kernel import LlamaCppServingAdapter

        projections = (
            tuple(p.strip() for p in args.projections.split(",") if p.strip())
            if args.projections
            else None
        )
        if args.gguf is not None:
            from subcor.backends.llamacpp_supervisor import LlamaServerSupervisor

            gguf_path = args.gguf.expanduser().absolute()
            if not gguf_path.is_file():
                raise SystemExit(f"--gguf file not found: {gguf_path}")
            model_id = args.model if args.model != _MLX_DEFAULT_MODEL else gguf_path.stem
            llama_dir = state_dir / "llamacpp"
            supervisor = LlamaServerSupervisor(
                base_gguf=gguf_path,
                binary=args.llama_server_bin,
                host="127.0.0.1",
                port=args.llamacpp_port,
                n_gpu_layers=args.n_gpu_layers,
                ctx_size=args.ctx_size,
                extra_args=tuple(args.llama_server_args or ()),
                lora_dir=llama_dir / "loras",
                log_path=llama_dir / "llama-server.log",
            )
            trainer_layers = (
                [int(x) for x in args.trainer_layers.split(",") if x.strip()]
                if args.trainer_layers
                else None
            )
            kernel = LlamaCppServingAdapter(
                model_id=model_id,
                supervisor=supervisor,
                lora_mode=args.lora_mode,
                trainer_device=args.trainer_device,
                trainer_dtype=args.trainer_dtype,
                trainer_layer_indices=trainer_layers,
                trainer_projections=projections,
                trainer_quantize_4bit=args.trainer_quantize_4bit,
                unload_trainer_after_study=args.unload_trainer_after_study,
                timeout=args.llamacpp_timeout,
                sticky_routing=_sticky_policy(args),
            )
            if args.lora_id_map:
                raise SystemExit("--lora-id-map applies only to an external server (no --gguf)")
            print(
                f"Starting supervised llama-server for '{gguf_path.name}' on "
                f"{supervisor.base_url} (-ngl {args.n_gpu_layers}, log {supervisor.log_path}) ..."
            )
            supervisor.start()
        else:
            print(f"Connecting to llama-server backend at '{args.llamacpp_url}' ({args.model}) ...")
            print("  (inference only; pass --gguf to let SUBCOR train Skill Units)")
            lora_id_map = _parse_lora_id_map(args.lora_id_map)
            if lora_id_map:
                print(f"  static LoRA ids: {lora_id_map}")
            kernel = LlamaCppServingAdapter(
                base_url=args.llamacpp_url,
                model_id=args.model,
                lora_mode=args.lora_mode,
                lora_id_map=lora_id_map,
                timeout=args.llamacpp_timeout,
                sticky_routing=_sticky_policy(args),
            )
    else:
        projections = (
            tuple(p.strip() for p in args.projections.split(",") if p.strip())
            if args.projections
            else None
        )
        print(f"Loading frozen base kernel '{args.model}' ...")
        kernel = MLXServingAdapter(
            model_id=args.model,
            target_projections=projections,
            layer_preset=args.layer_preset,
            num_layers=args.num_layers,
        )
    try:
        manager = TenantManager(state_dir=state_dir, kernel=kernel, initial_tenants=tenants)
        if supervisor is not None and not supervisor.is_running:
            raise SystemExit(
                "Supervised llama-server is not running after startup (it could not be "
                f"restarted with the ENABLED adapters). See {supervisor.log_path}."
            )
        server = SubcorHTTPServer(
            (args.host, args.port),
            manager=manager,
            auth_policy=auth_policy,
            export_root=export_root,
        )
    except BaseException:
        if supervisor is not None:
            supervisor.stop()
        raise

    print("=========================================================================")
    print(" SUBCOR — Multi-Tenant Server Ready")
    print("=========================================================================")
    print(f"Listen         : http://{args.host}:{args.port}")
    print(f"OpenAI BaseURL : http://{args.host}:{args.port}/v1")
    print(f"Frozen Kernel  : {kernel.model_id}")
    if supervisor is not None:
        print(f"llama-server   : {supervisor.base_url} (pid {supervisor.pid}, supervised)")
    print(f"State Dir      : {state_dir}")
    print(f"Export Root    : {export_root}")
    print(f"Auth           : {'Enabled' if auth_policy.enabled else 'DISABLED (loopback dev)'}")
    print(f"Tenants        : {', '.join(manager.list_tenants()) or '(auto-create on request)'}")
    print("=========================================================================")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down SUBCOR server cleanly.")
    finally:
        server.server_close()
        if supervisor is not None:
            supervisor.stop()
    return 0


def _run_bench_cli(args: argparse.Namespace) -> int:
    from subcor.benchmark import (
        BenchmarkLedger,
        BenchmarkSuite,
        compare_snapshots,
        core_v1_suite,
        render_markdown_report,
        run_benchmark_http,
        run_benchmark_local,
    )

    if args.bench_cmd == "export-suite":
        suite = core_v1_suite()
        suite.to_json_file(args.out)
        print(
            f"Exported {suite.suite_id}@{suite.version} "
            f"(sha256={suite.digest()[:12]}, {len(suite.cases)} cases) -> {args.out}"
        )
        return 0

    if args.bench_cmd == "baselines":
        from subcor.baselines import (
            compare_compression_at_equal_bytes,
            run_mlx_baseline_comparison,
            run_router_scale_sweep,
        )

        if args.live_mlx:
            report = run_mlx_baseline_comparison(
                model_id=args.model,
                n_train=args.n_train,
                n_test=args.n_test,
                steps=args.steps,
                seed=args.seed,
            )
        else:
            report = {
                "compression_matrix_audit": [
                    r.to_dict() for r in compare_compression_at_equal_bytes(seed=args.seed)
                ],
                "router_scale_sweep": [r.to_dict() for r in run_router_scale_sweep(seed=args.seed)],
            }
        if args.out is not None:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(report, indent=2))
        return 0

    if args.bench_cmd == "bfcl":
        from subcor.bfcl import run_bfcl_poc

        cats = tuple(c.strip() for c in args.categories.split(",") if c.strip())
        projs = tuple(p.strip() for p in args.projections.split(",") if p.strip())
        with tempfile.TemporaryDirectory(prefix="subcor-bfcl-") as tmp:
            wdir = args.work_dir.expanduser().resolve() if args.work_dir else Path(tmp)
            report = run_bfcl_poc(
                work_dir=wdir,
                model_id=args.model,
                categories=cats,
                max_per_category=args.max_per_category,
                train_source=args.train_source,
                n_single=args.n_single,
                n_parallel=args.n_parallel,
                n_abstain=args.n_abstain,
                rank=args.rank,
                steps=args.steps,
                lr=args.lr,
                grad_accum=args.grad_accum,
                max_tokens=args.max_tokens,
                num_adapted_layers=args.num_layers,
                projections=projs,
                seed=args.seed,
            )
        if args.out is not None:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(report, indent=2))
        return 0

    ledger = BenchmarkLedger(args.ledger.expanduser().resolve())

    if args.bench_cmd == "run":
        suite = BenchmarkSuite.from_json_file(args.suite) if args.suite else core_v1_suite()
        if args.endpoint:
            if args.tenant is not None:
                print(
                    "WARNING: --tenant is ignored with --endpoint; select tenant via --model subcor/<tenant>."
                )
            api_key = args.api_key or os.environ.get("SUBCOR_API_KEY")
            snap = run_benchmark_http(
                suite=suite,
                base_url=args.endpoint,
                model=args.model,
                label=args.label,
                api_key=api_key,
            )
        else:
            from subcor.server import MLXServingAdapter, TenantManager

            kernel = MLXServingAdapter(model_id=args.model)
            tenant_ctx = None
            if args.tenant:
                mgr = TenantManager(
                    state_dir=args.state_dir.expanduser().resolve(),
                    kernel=kernel,
                )
                tenant_ctx = mgr.ensure_tenant(args.tenant)
            snap = run_benchmark_local(
                suite=suite,
                kernel=kernel,
                label=args.label,
                tenant=tenant_ctx,
            )

        ledger.append(snap)
        print("=========================================================================")
        print(f" SUBCOR Benchmark Snapshot: {snap.label}")
        print("=========================================================================")
        print(
            f"Suite         : {snap.suite_id}@{snap.suite_version} (sha256={snap.suite_digest[:12]})"
        )
        print(
            f"Model         : {snap.model_id}"
            + (f" (tenant={snap.tenant_id})" if snap.tenant_id else "")
        )
        print(
            f"Active Units  : {len(snap.active_units)} "
            f"({snap.total_unit_bytes / 1024:.1f} KiB, {snap.mean_bits_per_weight:.4f} avg bpw)"
        )
        print(
            f"Overall       : {snap.overall_pass_rate * 100:5.1f}%  "
            f"Macro-Axis: {snap.macro_axis_pass_rate * 100:5.1f}%  "
            f"Anchors: {snap.anchor_pass_rate * 100:5.1f}%"
        )
        print(
            f"Efficiency    : {snap.mean_latency_ms:.0f} ms/case  "
            f"{snap.tokens_per_second:.1f} tok/s  "
            f"loop_fail={snap.token_limit_rate * 100:.1f}%"
        )
        print("\nPer-Axis Breakdown:")
        for ax, s in snap.axis_summaries.items():
            print(
                f"  {ax:<24} {s.passed:2d}/{s.total:2d} ({s.pass_rate * 100:5.1f}%)  "
                f"base_fallback={s.base_fallback_rate * 100:5.1f}%"
            )
        print(f"\nAppended to ledger: {ledger.path}")
        print("=========================================================================")
        return 0

    if args.bench_cmd == "report":
        snaps = ledger.load_all()
        md = render_markdown_report(snaps)
        if args.out is not None:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(md, encoding="utf-8")
            print(f"Wrote longitudinal growth report ({len(snaps)} snapshots) -> {args.out}")
        else:
            print(md)
        return 0

    if args.bench_cmd == "compare":
        snaps = ledger.load_all()
        if len(snaps) < 2:
            raise SystemExit(
                f"Need at least 2 snapshots in {ledger.path} to compare (found {len(snaps)})"
            )
        delta = compare_snapshots(snaps[0], snaps[-1])
        print(json.dumps(delta.to_dict(), indent=2))
        return 0 if delta.ci_passed else 2

    return 1


def _run_recompile_cli(args: argparse.Namespace) -> int:
    from subcor.recompile import recompile_units_for_kernel
    from subcor.server import MLXServingAdapter

    kernel = MLXServingAdapter(model_id=args.target_model)
    report = recompile_units_for_kernel(
        source_units_dir=args.source_units.expanduser().resolve(),
        dest_state_dir=args.dest_state_dir.expanduser().resolve(),
        target_tenant=args.tenant,
        kernel=kernel,
        steps_override=args.steps,
        seed=args.seed,
    )
    print(json.dumps(report.to_dict(), indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="subcor",
        description="SUBCOR — Governed self-learning runtime over frozen quantized kernels.",
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
        help="Train & route two independent Sub-1-Bit Skill Units over one frozen 0.8B core with 0%% general QA regression.",
    )
    mlx_moe_cmd.add_argument("--model", type=str, default="mlx-community/Qwen3.5-0.8B-4bit")
    mlx_moe_cmd.add_argument("--work-dir", type=Path, default=None)
    mlx_moe_cmd.add_argument("--rank", type=int, default=8)
    mlx_moe_cmd.add_argument("--sub1bit-rank", type=int, default=8)
    mlx_moe_cmd.add_argument("--steps", type=int, default=108)
    mlx_moe_cmd.add_argument("--lr", type=float, default=3e-4)
    mlx_moe_cmd.add_argument("--seed", type=int, default=42)
    mlx_moe_cmd.add_argument("--export-capheos", type=Path, default=None)

    serve_cmd = sub.add_parser(
        "serve",
        help="Run the multi-tenant OpenAI-compatible & governed SUBCOR HTTP server.",
    )
    serve_cmd.add_argument(
        "--model",
        type=str,
        default="mlx-community/Qwen3.5-0.8B-4bit",
        help="Compact Hugging Face MLX model ID.",
    )
    serve_cmd.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Bind host (keep 127.0.0.1 with an SSH tunnel, or bind a LAN IP with --api-key).",
    )
    serve_cmd.add_argument("--port", type=int, default=8420, help="Bind port.")
    serve_cmd.add_argument(
        "--state-dir",
        type=Path,
        default=Path.home() / ".local" / "share" / "subcor",
        help="Persistent directory for per-tenant Skill Units and trajectories.",
    )
    serve_cmd.add_argument(
        "--tenant",
        dest="tenants",
        action="append",
        default=[],
        help="Pre-register a CaPheOS tenant ID (repeatable, e.g. --tenant company-a --tenant company-b).",
    )
    serve_cmd.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="Optional bearer API key (or set SUBCOR_API_KEY env var).",
    )
    serve_cmd.add_argument(
        "--auth-config",
        type=Path,
        default=None,
        help="JSON file mapping bearer tokens to scoped actors, roles, and tenants.",
    )
    serve_cmd.add_argument(
        "--export-root",
        type=Path,
        default=None,
        help="Sandboxed root directory for HTTP export-capheos requests (default: <state-dir>/exports).",
    )
    serve_cmd.add_argument(
        "--backend",
        choices=("mlx", "llamacpp"),
        default="mlx",
        help="Inference backend: 'mlx' (Apple Silicon) or 'llamacpp' (Linux/CUDA/ROCm/CPU llama-server).",
    )
    serve_cmd.add_argument(
        "--llamacpp-url",
        type=str,
        default="http://127.0.0.1:8080",
        help=(
            "Base URL of an external llama-server when --backend llamacpp is selected without "
            "--gguf (inference only; Study Cycles then fail)."
        ),
    )
    serve_cmd.add_argument(
        "--lora-id-map",
        type=str,
        default=None,
        help=(
            "External llama-server only (no --gguf): unit_id -> LoRA id of the adapters that "
            "server was started with, as JSON ('{\"ops-v1\": 0}') or 'ops-v1=0,fin-v2=1'. "
            "Routed units without an entry run on the base model (with a warning)."
        ),
    )
    serve_cmd.add_argument(
        "--gguf",
        type=Path,
        default=None,
        help=(
            "llamacpp backend: local base GGUF. SUBCOR then runs llama-server itself, trains "
            "Skill Units on a PyTorch trainer loaded from the same GGUF, and loads them as "
            "GGUF LoRA adapters."
        ),
    )
    serve_cmd.add_argument(
        "--llama-server-bin",
        type=str,
        default="llama-server",
        help="llama-server executable used with --gguf (default: from PATH).",
    )
    serve_cmd.add_argument(
        "--llamacpp-port",
        type=int,
        default=8091,
        help="Loopback port for the supervised llama-server (with --gguf).",
    )
    serve_cmd.add_argument(
        "--llamacpp-timeout",
        type=float,
        default=60.0,
        help="Per-request HTTP timeout (seconds) for calls from SUBCOR to llama-server.",
    )
    serve_cmd.add_argument(
        "--n-gpu-layers",
        type=int,
        default=99,
        help="llama-server --n-gpu-layers for the supervised server (0 = CPU only).",
    )
    serve_cmd.add_argument(
        "--ctx-size",
        type=int,
        default=8192,
        help="llama-server --ctx-size for the supervised server.",
    )
    serve_cmd.add_argument(
        "--llama-server-arg",
        dest="llama_server_args",
        action="append",
        default=[],
        help="Extra argument passed to the supervised llama-server (repeatable).",
    )
    serve_cmd.add_argument(
        "--lora-mode",
        choices=("auto", "request", "global"),
        default="auto",
        help=(
            "How routed LoRAs are selected: per-request 'lora' field, or global "
            "/lora-adapters. 'auto' = request when supervised, global for an external server."
        ),
    )
    serve_cmd.add_argument(
        "--sticky-routing",
        action="store_true",
        help=(
            "llama.cpp: keep a conversation's previous Skill Units unless the fresh routing "
            "decision clearly beats them. Every unit switch makes llama-server re-read the "
            "whole conversation, so this trades some routing freshness for prompt-cache hits."
        ),
    )
    serve_cmd.add_argument(
        "--sticky-switch-margin",
        type=float,
        default=0.05,
        help="With --sticky-routing: margin above threshold a new unit needs to win a switch.",
    )
    serve_cmd.add_argument(
        "--sticky-min-tokens",
        type=int,
        default=2048,
        help="With --sticky-routing: estimated context size below which routing stays fresh.",
    )
    serve_cmd.add_argument(
        "--trainer-device",
        type=str,
        default=None,
        help="PyTorch device for the --gguf trainer (default: cuda if available, else cpu).",
    )
    serve_cmd.add_argument(
        "--trainer-dtype",
        choices=("float16", "bfloat16", "float32"),
        default=None,
        help="PyTorch dtype for the --gguf trainer (default: float16 on GPU, float32 on CPU).",
    )
    serve_cmd.add_argument(
        "--trainer-layers",
        type=str,
        default=None,
        help="Comma-separated layer indices to train (default: the last 6 layers of the GGUF).",
    )
    serve_cmd.add_argument(
        "--trainer-quantize-4bit",
        action="store_true",
        help=(
            "Pack frozen decoder-layer base weights into pure-PyTorch 4-bit INT4 buffers "
            "(~4.5 bpw, on-the-fly dequant in forward/backward) for 12-16 GB ROCm/CUDA GPUs."
        ),
    )
    serve_cmd.add_argument(
        "--unload-trainer-after-study",
        action="store_true",
        help="Free the trainer's GPU memory after every Study Cycle (slower, less VRAM).",
    )
    serve_cmd.add_argument(
        "--projections",
        type=str,
        default=None,
        help=(
            "Comma-separated linear projections to instrument (e.g. "
            "'mlp.down_proj,mlp.gate_proj,mlp.up_proj,self_attn.o_proj'). Default: mlp.down_proj."
        ),
    )
    serve_cmd.add_argument(
        "--layer-preset",
        choices=("default", "last_n", "periodic_full_attn", "every_k"),
        default=None,
        help="Multi-layer adaptation preset for MLX backend (e.g. 'periodic_full_attn', 'last_n').",
    )
    serve_cmd.add_argument(
        "--num-layers",
        type=int,
        default=None,
        help="Number of blocks to adapt when --layer-preset is set (e.g. 4).",
    )
    serve_cmd.add_argument(
        "--allow-unauthenticated",
        action="store_true",
        help="Explicitly allow binding a non-loopback host without auth (unsafe on shared LANs).",
    )

    bench_cmd = sub.add_parser(
        "bench",
        help="Run standardized longitudinal benchmarks documenting functional growth over time.",
    )
    bench_sub = bench_cmd.add_subparsers(dest="bench_cmd", required=True)

    b_run = bench_sub.add_parser(
        "run",
        help="Execute a versioned benchmark suite and append an immutable snapshot to the ledger.",
    )
    b_run.add_argument(
        "--label", type=str, required=True, help="Snapshot label (e.g. day0-base, v1-ops)."
    )
    b_run.add_argument(
        "--ledger",
        type=Path,
        default=Path.home() / ".local" / "share" / "subcor" / "benchmarks" / "history.jsonl",
    )
    b_run.add_argument("--suite", type=Path, default=None, help="Optional custom JSON suite file.")
    b_run.add_argument(
        "--endpoint",
        type=str,
        default=None,
        help="OpenAI-compatible base URL (e.g. http://127.0.0.1:8420/v1). If omitted, runs local MLX.",
    )
    b_run.add_argument("--api-key", type=str, default=None)
    b_run.add_argument("--model", type=str, default="mlx-community/Qwen3.5-0.8B-4bit")
    b_run.add_argument(
        "--state-dir", type=Path, default=Path.home() / ".local" / "share" / "subcor"
    )
    b_run.add_argument(
        "--tenant", type=str, default=None, help="Tenant ID when benchmarking local MLX state."
    )

    b_rep = bench_sub.add_parser(
        "report",
        help="Render a longitudinal Markdown Functional Growth Report from a benchmark ledger.",
    )
    b_rep.add_argument(
        "--ledger",
        type=Path,
        default=Path.home() / ".local" / "share" / "subcor" / "benchmarks" / "history.jsonl",
    )
    b_rep.add_argument("--out", type=Path, default=None, help="Write Markdown report to file.")

    b_cmp = bench_sub.add_parser(
        "compare",
        help="Compare Day-0 baseline snapshot against latest snapshot in a benchmark ledger.",
    )
    b_cmp.add_argument(
        "--ledger",
        type=Path,
        default=Path.home() / ".local" / "share" / "subcor" / "benchmarks" / "history.jsonl",
    )

    b_base = bench_sub.add_parser(
        "baselines",
        help="Run controlled same-byte-budget FP16 LoRA, few-shot, kNN, and router-scale baselines.",
    )
    b_base.add_argument(
        "--live-mlx",
        action="store_true",
        help="Also run the 100-case held-out procedural workflow comparison on a live MLX model.",
    )
    b_base.add_argument("--model", type=str, default="mlx-community/Qwen3.5-0.8B-4bit")
    b_base.add_argument("--n-train", type=int, default=20)
    b_base.add_argument("--n-test", type=int, default=100)
    b_base.add_argument("--steps", type=int, default=80)
    b_base.add_argument("--seed", type=int, default=42)
    b_base.add_argument("--out", type=Path, default=None)

    b_bfcl = bench_sub.add_parser(
        "bfcl",
        help="Run the official Berkeley Function Calling Leaderboard (BFCL v3) PoC on MLX.",
    )
    b_bfcl.add_argument("--model", type=str, default="mlx-community/Qwen3.5-4B-4bit")
    b_bfcl.add_argument(
        "--categories",
        type=str,
        default="simple,multiple,parallel,irrelevance",
    )
    b_bfcl.add_argument("--max-per-category", type=int, default=50)
    b_bfcl.add_argument(
        "--train-source",
        choices=("hermes", "bfcl_onpolicy", "bfcl_self_distill"),
        default="hermes",
        help=(
            "Strictly disjoint training source: 'hermes' (cross-dataset), 'bfcl_onpolicy' "
            "(BFCL[50:] GT), or 'bfcl_self_distill' (on-policy error corrections + base self-replay)."
        ),
    )
    b_bfcl.add_argument("--n-single", type=int, default=35)
    b_bfcl.add_argument("--n-parallel", type=int, default=35)
    b_bfcl.add_argument("--n-abstain", type=int, default=30)
    b_bfcl.add_argument("--rank", type=int, default=16)
    b_bfcl.add_argument("--steps", type=int, default=120)
    b_bfcl.add_argument("--lr", type=float, default=2e-4)
    b_bfcl.add_argument("--grad-accum", type=int, default=4)
    b_bfcl.add_argument("--max-tokens", type=int, default=512)
    b_bfcl.add_argument("--num-layers", type=int, default=8)
    b_bfcl.add_argument("--projections", type=str, default="mlp.down_proj,self_attn.o_proj")
    b_bfcl.add_argument("--work-dir", type=Path, default=None)
    b_bfcl.add_argument("--seed", type=int, default=42)
    b_bfcl.add_argument("--out", type=Path, default=None)

    b_exp = bench_sub.add_parser(
        "export-suite",
        help="Export the canonical built-in subcor-core-v1 benchmark suite to JSON.",
    )
    b_exp.add_argument("--out", type=Path, required=True)

    recomp_cmd = sub.add_parser(
        "recompile",
        help="Recompile portable Skill Units (from skill_spec.json) onto a new base kernel.",
    )
    recomp_cmd.add_argument(
        "--source-units", type=Path, required=True, help="Source units directory."
    )
    recomp_cmd.add_argument(
        "--target-model", type=str, required=True, help="New target MLX base model ID."
    )
    recomp_cmd.add_argument(
        "--dest-state-dir",
        type=Path,
        default=Path.home() / ".local" / "share" / "subcor",
    )
    recomp_cmd.add_argument("--tenant", type=str, required=True, help="Target tenant ID.")
    recomp_cmd.add_argument(
        "--steps", type=int, default=None, help="Optional training step override."
    )
    recomp_cmd.add_argument("--seed", type=int, default=42)

    inspect_cmd = sub.add_parser(
        "inspect",
        help="Inspect registered Skill Units and verify SHA-256 integrity in a unit store.",
    )
    inspect_cmd.add_argument("units_dir", type=Path, help="Path to a SUBCOR units directory.")

    export_cmd = sub.add_parser(
        "export-capheos",
        help="Export a CONFORMANT or ENABLED Skill Unit as a CaPheOS format-2 bundle.",
    )
    export_cmd.add_argument("units_dir", type=Path, help="Path to a SUBCOR units directory.")
    export_cmd.add_argument("unit_id", type=str, help="Unit ID to export.")
    export_cmd.add_argument("destination", type=Path, help="Destination directory.")

    args = parser.parse_args(argv)

    if args.command == "demo":
        if args.work_dir is None:
            with tempfile.TemporaryDirectory(prefix="subcor-demo-") as tmp:
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

    if args.command == "serve":
        return _run_serve_cli(args)

    if args.command == "bench":
        return _run_bench_cli(args)

    if args.command == "recompile":
        return _run_recompile_cli(args)

    if args.command == "inspect":
        from subcor.types import IntegrityError

        registry = SkillUnitRegistry(args.units_dir)
        rows = []
        any_failed = False
        for m in registry.list_units():
            row: dict[str, object] = {
                "unit_id": m.unit_id,
                "version": m.version,
                "base_kernel_id": m.base_kernel_id,
                "state": m.state.value,
                "mode": m.mode.value,
                "bits_per_weight": round(m.effective_bits_per_weight, 4),
                "payload_bytes": m.payload_bytes,
                "sha256": m.payload_sha256[:16],
            }
            try:
                registry.load_verified_deltas(m.unit_id)
                row["integrity"] = "verified"
            except (IntegrityError, FileNotFoundError, ValueError) as exc:
                any_failed = True
                row["integrity"] = "failed"
                row["error"] = str(exc)
            rows.append(row)
        print(json.dumps(rows, indent=2))
        return 1 if any_failed else 0

    if args.command == "export-capheos":
        registry = SkillUnitRegistry(args.units_dir)
        out = registry.export_capheos_unit(args.unit_id, args.destination)
        print(f"Exported CaPheOS unit bundle to: {out}")
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
