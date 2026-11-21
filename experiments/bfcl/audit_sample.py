"""R7 A0 failure audit sampler (PREREGISTRATION.md section 9, R7).

Draws a seeded sample of failed A0 items (seed 20261122), stratified by section
(non-live AST, live AST, irrelevance, multi-turn; 15 each = 60) and dumps, for each,
the prompt summary, the raw model output, bfcl's error and error_type, so each can be
classified by hand as "parser/serving artifact" or "model error".

  python -I audit_sample.py --run-dir RUN --out audit.json [--n 60] [--seed 20261122]
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

SECTIONS = {
    "non_live": {"simple_python", "simple_java", "simple_javascript", "multiple", "parallel", "parallel_multiple"},
    "live": {"live_simple", "live_multiple", "live_parallel", "live_parallel_multiple"},
    "irrelevance": {"irrelevance", "live_irrelevance"},
    "multi_turn": {"multi_turn_base", "multi_turn_miss_func", "multi_turn_miss_param", "multi_turn_long_context"},
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--seed", type=int, default=20261122)
    args = ap.parse_args()

    rows = list(csv.DictReader((args.run_dir / "per_item.csv").open()))
    fails = [r for r in rows if r["correct"] == "0"]
    rng = random.Random(args.seed)
    per = args.n // len(SECTIONS)
    picked = []
    for sec, cats in SECTIONS.items():
        pool = sorted((r for r in fails if r["category"] in cats), key=lambda r: (r["category"], r["id"]))
        rng.shuffle(pool)
        picked += [(sec, r) for r in pool[:per]]

    # bfcl score files: one JSON line per failed item after the header
    score_lines: dict[str, dict] = {}
    for f in (args.run_dir / "score").rglob("*_score.json"):
        with f.open() as fh:
            next(fh, None)
            for line in fh:
                if line.strip():
                    d = json.loads(line)
                    score_lines[d["id"]] = d
    result_lines: dict[str, dict] = {}
    for f in (args.run_dir / "result").rglob("*_result.json"):
        for line in f.open():
            if line.strip():
                d = json.loads(line)
                result_lines[d["id"]] = d

    out = []
    for sec, r in picked:
        s = score_lines.get(r["id"], {})
        res = result_lines.get(r["id"], {})
        prompt = s.get("prompt", {})
        q = prompt.get("question")
        out.append(
            {
                "section": sec,
                "category": r["category"],
                "id": r["id"],
                "error_type": r["error_type"],
                "inference_error": r["inference_error"],
                "error": s.get("error"),
                "question_head": json.dumps(q)[:600] if q is not None else None,
                "functions": [fn.get("name") for fn in prompt.get("function", [])][:20],
                "model_result_raw": (json.dumps(s.get("model_result_raw", res.get("result")))[:1500]),
                "possible_answer": json.dumps(s.get("possible_answer"))[:800],
                "finish_reasons": [t.get("finish_reason") for t in res.get("harness_trace", [])][:30],
                "classification": None,
                "note": None,
            }
        )
    args.out.write_text(json.dumps(out, indent=1) + "\n")
    print(f"{len(out)} items -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
