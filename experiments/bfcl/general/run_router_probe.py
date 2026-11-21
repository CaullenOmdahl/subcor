"""Router false-activation probe: general (non-tool) prompts -> SUBCOR serve.

Sends each prompt as a single user turn with NO tools to SUBCOR's
/v1/chat/completions (model `subcor/<tenant>`) and records the response's
`subcor.selected_units` / `subcor.scores` / `fell_back_to_base`. Any non-empty
selected_units on these prompts is a false activation. Generation is irrelevant to
routing (SUBCOR routes on the frozen-base embedding of the last user message before
decoding), so --max-tokens defaults to 1.

Note: every request through a tenant is appended to that tenant's live telemetry
(trajectories.jsonl, unverified). Run against a throwaway copy of the state dir, or
expect ~300 extra unverified rows.

Summary: false-activation rate with Wilson 95% CI, overall / general-only (excludes
`hard_negative`) / per category / per unit, plus per-unit score quantiles.

  python -I run_router_probe.py --base-url http://127.0.0.1:8592/v1 --model subcor/probe \
      --prompts .../general/data/router_probe_prompts.jsonl --out .../results/probe.jsonl
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

import common  # noqa: E402


def wilson(k: int, n: int, z: float = 1.959964) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (max(0.0, c - h), min(1.0, c + h))


def rate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    k = sum(1 for r in rows if r["selected_units"])
    lo, hi = wilson(k, len(rows))
    return {"n": len(rows), "activated": k, "rate": k / len(rows) if rows else float("nan"), "wilson95": [lo, hi]}


def quantiles(xs: list[float]) -> dict[str, float]:
    if not xs:
        return {}
    s = sorted(xs)

    def q(p: float) -> float:
        return s[min(len(s) - 1, int(round(p * (len(s) - 1))))]

    return {"min": s[0], "p50": q(0.5), "p90": q(0.9), "p99": q(0.99), "max": s[-1]}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    common.add_endpoint_args(ap)
    ap.add_argument("--prompts", type=Path, required=True, help="jsonl with id, prompt, category")
    ap.add_argument("--max-tokens", type=int, default=1)
    ap.add_argument(
        "--tenant-dir",
        type=Path,
        default=None,
        help="optional <state-dir>/tenants/<tenant>: read each unit's router_gate.json threshold "
        "to report score-minus-threshold margins (how close general prompts come to firing)",
    )
    args = ap.parse_args()

    cfg = common.config_from_args(args)
    items = common.read_jsonl(args.prompts)
    if args.limit is not None:
        items = items[: args.limit]
    fp = common.fingerprint_of("router_probe", cfg.describe(), args.max_tokens, args.run_label)

    def worker(it: dict[str, Any]) -> dict[str, Any]:
        rec = common.chat(cfg, [{"role": "user", "content": it["prompt"]}], max_tokens=args.max_tokens)
        row: dict[str, Any] = {
            "id": it["id"],
            "task": "router_probe",
            "run_label": args.run_label,
            "model": cfg.model,
            "category": it.get("category"),
            "source": it.get("source"),
            "ok": rec["ok"],
        }
        if not rec["ok"]:
            row["error"] = rec["error"]
            return row
        sub = rec["subcor"] if isinstance(rec["subcor"], dict) else None
        if sub is None:
            row.update({"ok": False, "error": "response has no 'subcor' field (not a SUBCOR endpoint?)"})
            return row
        row.update(
            {
                "selected_units": list(sub.get("selected_units") or []),
                "scores": sub.get("scores") or {},
                "fell_back_to_base": sub.get("fell_back_to_base"),
                "tenant": sub.get("tenant"),
                "routed_header": rec["routed_header"],
                "latency_s": rec["latency_s"],
            }
        )
        return row

    print(f"router probe: {len(items)} prompts -> {args.out} ({cfg.base_url}, model={cfg.model})", file=sys.stderr)
    rows = common.run_items(items, worker, out_path=args.out, concurrency=args.concurrency, fingerprint=fp)
    ok = [r for r in rows if r.get("ok")]
    cats = sorted({r["category"] for r in ok})
    units = sorted({u for r in ok for u in (r.get("scores") or {})} | {u for r in ok for u in r["selected_units"]})
    tenants = sorted({str(r.get("tenant")) for r in ok})
    summary = {
        "task": "router_probe",
        "endpoint": cfg.describe(),
        "run_label": args.run_label,
        "fingerprint": fp,
        "tenants_seen": tenants,
        "n_failed_requests": len(rows) - len(ok),
        "registered_units_seen_in_scores": units,
        "overall": rate(ok),
        "general_only": rate([r for r in ok if r["category"] != "hard_negative"]),
        "per_category": {c: rate([r for r in ok if r["category"] == c]) for c in cats},
        "per_unit": {
            u: {
                **rate([{"selected_units": [x for x in r["selected_units"] if x == u]} for r in ok]),
                "score_quantiles": quantiles([float(r["scores"][u]) for r in ok if u in (r.get("scores") or {})]),
            }
            for u in units
        },
    }
    if args.tenant_dir is not None:
        for u in units:
            gate_path = args.tenant_dir / "units" / u / "router_gate.json"
            if gate_path.exists():
                thr = float(json.loads(gate_path.read_text())["threshold"])
                summary["per_unit"][u]["threshold"] = thr
                summary["per_unit"][u]["margin_quantiles"] = quantiles(
                    [float(r["scores"][u]) - thr for r in ok if u in (r.get("scores") or {})]
                )
    if not units:
        summary["warning"] = "no unit scores in any response: tenant has no ENABLED units, so 0% is trivial"
    common.write_summary(common.default_summary_path(args.out, args.summary), summary)
    print(json.dumps({k: v for k, v in summary.items() if k not in ("endpoint", "per_unit")}, indent=2))
    return 0 if not summary["n_failed_requests"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
