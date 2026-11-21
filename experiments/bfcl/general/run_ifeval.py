"""IFEval (all 541 prompts) against an OpenAI-compatible chat endpoint.

Generation: one user turn = the IFEval prompt verbatim (as lm-eval's `ifeval` task with
--apply_chat_template does), temperature 0, thinking off, max_tokens 1280 (lm-eval's
max_gen_toks for ifeval), no stop strings.

Grading: lm-evaluation-harness's `lm_eval.tasks.ifeval.utils.process_results`, a port of
google-research/instruction_following_eval (strict + loose, prompt- and instruction-level).
Needs the eval venv (lm-eval[ifeval]); see run_smoke.sh.

Per-item JSONL row: key, response, the four metric fields, finish_reason, selected_units...
Summary: the four standard IFEval numbers + SUBCOR routing activation rate.

  python -I run_ifeval.py --base-url http://127.0.0.1:8502/v1 --model local \
      --data .../general/data/ifeval.jsonl --out .../results/ifeval_base.jsonl
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

import common  # noqa: E402

PROMPT_TEMPLATE_VERSION = "ifeval-verbatim-v1"


def grade(doc: dict[str, Any], response: str) -> dict[str, Any]:
    from lm_eval.tasks.ifeval import utils as ifeval_utils

    res = ifeval_utils.process_results(doc, [response])
    return {
        "prompt_level_strict_acc": bool(res["prompt_level_strict_acc"]),
        "inst_level_strict_acc": [bool(x) for x in res["inst_level_strict_acc"]],
        "prompt_level_loose_acc": bool(res["prompt_level_loose_acc"]),
        "inst_level_loose_acc": [bool(x) for x in res["inst_level_loose_acc"]],
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    ok = [r for r in rows if r.get("ok")]
    n = len(ok)

    def mean(xs: list[float]) -> float:
        return sum(xs) / len(xs) if xs else float("nan")

    inst_s = [x for r in ok for x in r["inst_level_strict_acc"]]
    inst_l = [x for r in ok for x in r["inst_level_loose_acc"]]
    return {
        "n_items": n,
        "n_failed_requests": len(rows) - n,
        "prompt_level_strict_acc": mean([float(r["prompt_level_strict_acc"]) for r in ok]),
        "inst_level_strict_acc": mean([float(x) for x in inst_s]),
        "prompt_level_loose_acc": mean([float(r["prompt_level_loose_acc"]) for r in ok]),
        "inst_level_loose_acc": mean([float(x) for x in inst_l]),
        "n_instructions": len(inst_s),
        "finish_length": sum(1 for r in ok if r.get("finish_reason") == "length"),
        "had_think_block": sum(1 for r in ok if r.get("had_think")),
        "empty_response": sum(1 for r in ok if not r.get("response", "").strip()),
        "routing": common.routing_summary(ok),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    common.add_endpoint_args(ap)
    ap.add_argument("--data", type=Path, required=True, help="IFEval jsonl (google/IFEval)")
    ap.add_argument("--max-tokens", type=int, default=1280)
    args = ap.parse_args()

    cfg = common.config_from_args(args)
    docs = common.read_jsonl(args.data)
    if len(docs) != 541:
        print(f"warning: {len(docs)} IFEval docs (expected 541)", file=sys.stderr)
    items = [{"id": int(d["key"]), "doc": d} for d in docs]
    if args.limit is not None:
        items = items[: args.limit]
    fp = common.fingerprint_of(
        "ifeval", cfg.describe(), args.max_tokens, PROMPT_TEMPLATE_VERSION, args.run_label
    )

    def worker(it: dict[str, Any]) -> dict[str, Any]:
        doc = it["doc"]
        rec = common.chat(cfg, [{"role": "user", "content": doc["prompt"]}], max_tokens=args.max_tokens)
        row: dict[str, Any] = {
            "id": it["id"],
            "task": "ifeval",
            "run_label": args.run_label,
            "model": cfg.model,
            "ok": rec["ok"],
            "instruction_id_list": doc["instruction_id_list"],
        }
        if not rec["ok"]:
            row["error"] = rec["error"]
            return row
        row.update(
            {
                "response": rec["text"],
                "had_think": rec["had_think"],
                "finish_reason": rec["finish_reason"],
                "completion_tokens": rec["usage"].get("completion_tokens"),
                "latency_s": rec["latency_s"],
                "selected_units": rec["selected_units"],
                "subcor": rec["subcor"],
            }
        )
        row.update(grade(doc, rec["text"]))
        return row

    print(f"IFEval: {len(items)} items -> {args.out} ({cfg.base_url}, model={cfg.model})", file=sys.stderr)
    rows = common.run_items(items, worker, out_path=args.out, concurrency=args.concurrency, fingerprint=fp)
    summary = {
        "task": "ifeval",
        "grader": "lm_eval.tasks.ifeval.utils.process_results (google-research IFEval port)",
        "endpoint": cfg.describe(),
        "run_label": args.run_label,
        "max_tokens": args.max_tokens,
        "limit": args.limit,
        "fingerprint": fp,
        **summarize(rows),
    }
    out_summary = common.default_summary_path(args.out, args.summary)
    common.write_summary(out_summary, summary)
    print(common.json.dumps({k: v for k, v in summary.items() if k != "endpoint"}, indent=2))
    return 0 if summary["n_failed_requests"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
