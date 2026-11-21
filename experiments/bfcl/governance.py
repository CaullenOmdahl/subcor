"""Governed promotion of train_unit.py units (PREREGISTRATION.md section 4.4, "Governance").

SUBCOR's own conformance runs only inside ``run_study_cycle``. This script applies the
same locked gate to a unit trained by ``train_unit.py``, with the inputs fixed by the
preregistration:

* ``SkillUnitRegistry.record_conformance`` with its default thresholds
  (``max_regression_tolerance=0.02``, ``min_target_improvement=0.05``);
* ``target_before = exp(-dev_loss.base)``, ``target_after = exp(-dev_loss.<variant>)``
  from ``train_report.json``;
* ``baseline_before`` / ``baseline_after`` = substring accuracy on SUBCOR's
  ``DEFAULT_BASELINE_ANCHOR_CASES`` through the tenant (``generate_routed_chat`` with the
  tenant router, ``max_tokens=64``, thinking off, ``top_k=1``), without and with the
  unit's RouterGate registered, exactly as ``run_study_cycle`` phases 1 and 4 do;
* then ``TenantManager.promote_unit`` by an approver other than the proposer.

No BFCL, IFEval, GSM8K or probe item is used. One supervised llama-server is started for
the base GGUF and stopped at the end. Output: a JSON report (``--out``).

  uv run --extra dev --extra rocm python experiments/bfcl/governance.py \\
      --state-dir STATE --gguf Qwen3.5-4B-Q4_K_M.gguf --llamacpp-port 8531 \\
      --train-report OUT/train_report.json \\
      --unit tenant-a3=bfcl-tools-v1-sub1bit:sub1bit --unit tenant-a4=bfcl-tools-v1-fp32:fp32 \\
      --approver bfcl-governance-reviewer --out OUT/governance.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

from subcor.backends.llamacpp_kernel import LlamaCppServingAdapter
from subcor.backends.llamacpp_supervisor import LlamaServerSupervisor
from subcor.server import DEFAULT_BASELINE_ANCHOR_CASES, TenantManager


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--state-dir", type=Path, required=True)
    ap.add_argument("--gguf", type=Path, required=True)
    ap.add_argument("--llamacpp-port", type=int, required=True)
    ap.add_argument("--ctx-size", type=int, default=65536)
    ap.add_argument("--llama-server-arg", dest="llama_server_args", action="append", default=[])
    ap.add_argument("--train-report", type=Path, required=True)
    ap.add_argument(
        "--unit",
        action="append",
        required=True,
        help="tenant=unit_id:variant (variant = key of train_report dev_loss)",
    )
    ap.add_argument("--approver", required=True)
    ap.add_argument("--timeout", type=float, default=600.0)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)

    report = json.loads(args.train_report.read_text())
    dev_loss = report["dev_loss"]
    state_dir = args.state_dir.expanduser().resolve()
    gguf = args.gguf.expanduser().absolute()
    supervisor = LlamaServerSupervisor(
        base_gguf=gguf,
        binary="/usr/bin/llama-server",
        host="127.0.0.1",
        port=args.llamacpp_port,
        n_gpu_layers=99,
        ctx_size=args.ctx_size,
        extra_args=tuple(args.llama_server_args),
        lora_dir=state_dir / "llamacpp" / "loras",
        log_path=state_dir / "llamacpp" / "llama-server-governance.log",
    )
    kernel = LlamaCppServingAdapter(model_id=gguf.stem, supervisor=supervisor, timeout=args.timeout)
    supervisor.start()
    out: dict = {"started": time.time(), "anchors": [list(c) for c in DEFAULT_BASELINE_ANCHOR_CASES]}
    results = []
    try:
        mgr = TenantManager(state_dir=state_dir, kernel=kernel, initial_tenants=())
        for spec in args.unit:
            tenant, rest = spec.split("=", 1)
            unit_id, variant = rest.rsplit(":", 1)
            ctx = mgr.get_tenant(tenant)

            def score(texts: list) -> float:
                hits = 0
                for prompt, expected in DEFAULT_BASELINE_ANCHOR_CASES:
                    res = kernel.generate_routed_chat(
                        user_prompt=prompt,
                        system_prompt=None,
                        router=ctx.router,
                        registry=ctx.registry,
                        max_tokens=64,
                        enable_thinking=False,
                        top_k=1,
                    )
                    texts.append(
                        {"prompt": prompt, "text": res.text, "units": list(res.routing.selected_units)}
                    )
                    if expected.strip().lower() in res.text.strip().lower():
                        hits += 1
                return hits / len(DEFAULT_BASELINE_ANCHOR_CASES)

            before_txt: list = []
            after_txt: list = []
            with mgr.lock:
                baseline_before = score(before_txt)
                gate = ctx.registry.load_router_gate(unit_id)
                if gate is None:
                    raise SystemExit(f"{unit_id}: no router_gate.json")
                mgr._prepare_kernel_units([(ctx.registry, unit_id)])
                ctx.router.register(gate)
                try:
                    baseline_after = score(after_txt)
                finally:
                    ctx.router.unregister(unit_id)
            t_before = math.exp(-float(dev_loss["base"]))
            t_after = math.exp(-float(dev_loss[variant]))
            manifest = ctx.registry.record_conformance(
                unit_id,
                baseline_before=baseline_before,
                baseline_after=baseline_after,
                target_before=t_before,
                target_after=t_after,
            )
            row = {
                "tenant": tenant,
                "unit_id": unit_id,
                "variant": variant,
                "baseline_before": baseline_before,
                "baseline_after": baseline_after,
                "target_before": t_before,
                "target_after": t_after,
                "conformance": manifest.conformance.to_dict() if manifest.conformance else None,
                "state_after_conformance": manifest.state.value,
                "anchor_outputs_before": before_txt,
                "anchor_outputs_after": after_txt,
            }
            if manifest.conformance is not None and manifest.conformance.passed:
                promoted = mgr.promote_unit(tenant, unit_id, approver=args.approver)
                row["state_final"] = promoted.state.value
                row["approver"] = args.approver
                row["proposer"] = promoted.proposer
            else:
                row["state_final"] = manifest.state.value
                row["rejected_by_governance"] = True
            results.append(row)
            print(json.dumps({k: v for k, v in row.items() if not k.startswith("anchor_")}), flush=True)
    finally:
        supervisor.stop()
    out["results"] = results
    out["finished"] = time.time()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2) + "\n")
    return 0 if all(r["state_final"] == "enabled" for r in results) else 2


if __name__ == "__main__":
    sys.exit(main())
