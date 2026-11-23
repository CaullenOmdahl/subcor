"""Validate the built tool-calling splits; exit 1 on any failure.

Checks, per file:
  * every row: structure + every assistant tool_call is valid JSON whose object matches
    the called function's JSON Schema (schema_check.check_conversation)
  * the stored ``category`` equals a fresh categorize() (derived clarify rows excepted)
  * independent re-run of the BFCL v4 decontamination rules -> must be 0 hits
  * train/dev disjointness: no shared ids, no shared tool sets, no dev user prompt with
    13-gram containment >= 0.8 or MinHash Jaccard > 0.7 against a train user prompt
  * (router files, if given) positives/negatives have no overlap and negatives contain no
    prompt matching IFEval / GSM8K-test by the same n-gram / MinHash rules

Usage:
  $VENV -I experiments/bfcl/data/validate_tool_data.py --data $SCRATCH/bfcl/data \
      --bfcl-data $VENV_SITE/bfcl_eval/data [--raw $SCRATCH/bfcl/raw]
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from build_tool_data import _norm_prompt, categorize, load_bfcl  # noqa: E402
from common import MinHashLSH, NgramIndex, name_key, schema_key, signature_key, user_texts  # noqa: E402
from schema_check import check_conversation  # noqa: E402


def read_jsonl(p: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]


def group(r: dict[str, Any]) -> str:
    return ",".join(sorted(t["function"]["name"].lower() for t in r["tools"]))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--bfcl-data", type=Path, required=True)
    ap.add_argument("--raw", type=Path, help="raw dir (for ifeval/ gsm8k/ router checks)")
    args = ap.parse_args()
    fails: list[str] = []
    report: dict[str, Any] = {}

    bprompts, bfuncs, _ = load_bfcl(args.bfcl_data)
    b_schema = {schema_key(f) for f in bfuncs}
    b_sig = {signature_key(f) for f in bfuncs}
    b_name = {name_key(f) for f in bfuncs}
    b_exact = {_norm_prompt(p) for p in bprompts}
    ng, mh = NgramIndex(13), MinHashLSH()
    for p in bprompts:
        ng.add(p)
        mh.add(p)

    splits = {}
    for name in ("train", "dev", "train_long"):
        p = args.data / f"{name}.jsonl"
        if p.exists():
            splits[name] = read_jsonl(p)

    for name, rows in splits.items():
        errs: Counter = Counter()
        n_calls = n_asst = 0
        hits: Counter = Counter()
        cat_mismatch = 0
        name_only = 0
        for r in rows:
            e = check_conversation(r)
            for x in e:
                errs[x.split(":", 1)[-1].strip()[:60]] += 1
            if e and len(fails) < 20:
                fails.append(f"{name}/{r['id']}: {e[:3]}")
            n_asst += sum(1 for m in r["messages"] if m["role"] == "assistant")
            n_calls += sum(len(m.get("tool_calls") or []) for m in r["messages"])
            if not r["id"].endswith("-clarify") and categorize(r) != r["category"]:
                cat_mismatch += 1
            for t in r["tools"]:
                if schema_key(t) in b_schema:
                    hits["func_schema_exact"] += 1
                if signature_key(t) in b_sig:
                    hits["func_signature"] += 1
                if name_key(t) in b_name:
                    name_only += 1
            for u in user_texts(r["messages"]):
                if _norm_prompt(u) in b_exact:
                    hits["prompt_exact"] += 1
                if ng.max_overlap(u)[0] >= 0.8:
                    hits["ngram13_overlap"] += 1
                if mh.max_jaccard(u)[0] > 0.7:
                    hits["minhash_jaccard"] += 1
        if errs:
            fails.append(
                f"{name}: {sum(errs.values())} validation errors {dict(errs.most_common(5))}"
            )
        if cat_mismatch:
            fails.append(f"{name}: {cat_mismatch} category mismatches")
        if hits:
            fails.append(f"{name}: BFCL contamination hits {dict(hits)}")
        report[name] = {
            "rows": len(rows),
            "assistant_turns": n_asst,
            "tool_calls_validated": n_calls,
            "validation_errors": sum(errs.values()),
            "bfcl_hits": dict(hits),
            "tools_sharing_only_a_name_with_bfcl": name_only,
            "categories": dict(sorted(Counter(r["category"] for r in rows).items())),
        }

    if "train" in splits and "dev" in splits:
        tr, dv = splits["train"], splits["dev"]
        ids = {r["id"] for r in tr} & {r["id"] for r in dv}
        tg = {group(r) for r in tr}
        shared_groups = sum(1 for r in dv if group(r) in tg)
        tmh, tng = MinHashLSH(), NgramIndex(13)
        for r in tr:
            for u in user_texts(r["messages"]):
                tmh.add(u)
                tng.add(u)
        near = sum(
            1
            for r in dv
            if any(
                tmh.max_jaccard(u)[0] > 0.7 or tng.max_overlap(u)[0] >= 0.8
                for u in user_texts(r["messages"])
            )
        )
        report["train_dev_disjoint"] = {
            "shared_ids": len(ids),
            "dev_rows_sharing_train_toolset": shared_groups,
            "dev_rows_near_dup_train": near,
        }
        if ids or shared_groups or near:
            fails.append(f"train/dev not disjoint: {report['train_dev_disjoint']}")

    pos_p, neg_p = args.data / "router_pos.jsonl", args.data / "router_neg.jsonl"
    if pos_p.exists() and neg_p.exists():
        pos = [r["prompt"] for r in read_jsonl(pos_p)]
        neg = [r["prompt"] for r in read_jsonl(neg_p)]
        both = {_norm_prompt(x) for x in pos} & {_norm_prompt(x) for x in neg}
        rr: dict[str, Any] = {"pos": len(pos), "neg": len(neg), "pos_neg_overlap": len(both)}
        if both:
            fails.append(f"router pos/neg overlap {len(both)}")
        trp = {_norm_prompt(u) for r in splits.get("train", []) for u in user_texts(r["messages"])}
        rr["pos_not_from_train"] = sum(1 for x in pos if _norm_prompt(x) not in trp)
        if rr["pos_not_from_train"]:
            fails.append(f"{rr['pos_not_from_train']} router positives not drawn from train")
        if args.raw:
            from build_router_calib import load_eval_prompts  # noqa: PLC0415

            ev = load_eval_prompts(args.raw)
            eng, emh = NgramIndex(13), MinHashLSH()
            for p in ev:
                eng.add(p)
                emh.add(p)
            bad = sum(1 for x in neg if eng.max_overlap(x)[0] >= 0.8 or emh.max_jaccard(x)[0] > 0.7)
            bad_b = sum(1 for x in neg if ng.max_overlap(x)[0] >= 0.8 or mh.max_jaccard(x)[0] > 0.7)
            rr["neg_hits_ifeval_gsm8k_test"] = bad
            rr["neg_hits_bfcl"] = bad_b
            if bad or bad_b:
                fails.append(f"router negatives contaminated: ifeval/gsm8k={bad} bfcl={bad_b}")
        report["router"] = rr

    print(json.dumps(report, indent=1))
    if fails:
        print("FAIL:\n  " + "\n  ".join(fails))
        return 1
    print("OK: all checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
