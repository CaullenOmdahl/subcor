"""Second-pass decontamination audit: BFCL functions copied under a DIFFERENT name.

`build_tool_data.py` drops a training conversation when one of its tools has the same
sanitized name as a BFCL v4 function (with the same schema, or the same property-name
set). A renamed copy (`get_weather` -> `fetch_weather`, same description and
parameters) passes those rules. This audit looks for such copies, and for looser
prompt near-duplicates, in train / dev / router positives / A1 exemplars:

  desc_exact        normalized function description (>= 6 words) equals a BFCL one
  desc_minhash      word-3-gram Jaccard of the descriptions > 0.7 (>= 6 words)
  props_desc        same sorted property-name set (>= 2 props) AND description Jaccard > 0.5
  prompt_contained  a whole BFCL prompt of 6-12 words occurs verbatim (normalized words,
                    contiguous) inside a user message. The build's 13-gram rule hashes a
                    BFCL prompt shorter than 13 words as ONE gram, which only matches a
                    train message of exactly the same words, so "Calculate the area of a
                    circle with a radius of 5 units." embedded in a longer request slipped
                    through. This rule closes that hole.
  prompt_ngram8     a user message shares >= 0.5 of its word 8-grams with one BFCL prompt
                    (looser than the build's 13-gram / 0.8 rule; >= 8 words)

Read-only: it writes a JSON report and, with --drop-list, the ids of conversations hit
by any FUNCTION rule or by prompt_contained. prompt_ngram8 is reported, not dropped
(template-level paraphrases such as "radius of 4" vs "radius of 5" are the residual
risk that PREREGISTRATION.md R1(a) analyses). Run it with the eval venv, `python -I`:

  $VENV -I experiments/bfcl/data/decontam_semantic.py --data $S/data \
      --bfcl-data $VENV_SITE/bfcl_eval/data --a1 experiments/bfcl/a1_exemplars.json \
      --out experiments/bfcl/data/results/decontam_semantic.json \
      --drop-list experiments/bfcl/data/results/decontam_semantic_drop.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from build_tool_data import load_bfcl  # noqa: E402
from common import MinHashLSH, NgramIndex, norm_parameters, user_texts, words  # noqa: E402

MIN_DESC_WORDS = 6


def _fn(t: dict[str, Any]) -> dict[str, Any]:
    return (
        t["function"] if t.get("type") == "function" and isinstance(t.get("function"), dict) else t
    )


def _desc(fn: dict[str, Any]) -> str:
    return " ".join(words(str(fn.get("description") or "")))


def _props(fn: dict[str, Any]) -> str:
    p = norm_parameters(fn.get("parameters"))
    return ",".join(sorted(k.lower() for k in p["properties"]))


def read_jsonl(p: Path) -> list[dict[str, Any]]:
    return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--bfcl-data", type=Path, required=True)
    ap.add_argument("--a1", type=Path, default=None, help="a1_exemplars.json (checked by row id)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--drop-list", type=Path, default=None)
    args = ap.parse_args()

    bprompts, bfuncs, _ = load_bfcl(args.bfcl_data)
    b_desc_exact: dict[str, str] = {}
    b_props: dict[str, list[int]] = {}
    mh = MinHashLSH()
    b_names: list[str] = []
    for f in bfuncs:
        d = _desc(f)
        i = mh.add(d if len(d.split()) >= MIN_DESC_WORDS else "")
        b_names.append(str(f.get("name")))
        if len(d.split()) >= MIN_DESC_WORDS:
            b_desc_exact.setdefault(d, str(f.get("name")))
        pk = _props(f)
        if pk.count(",") >= 1:
            b_props.setdefault(pk, []).append(i)
    ng8 = NgramIndex(8)
    short: dict[int, set[str]] = {}
    for p in bprompts:
        ng8.add(p)
        w = words(p)
        if 6 <= len(w) < 13:
            short.setdefault(len(w), set()).add(" ".join(w))

    def contained(text: str) -> bool:
        w = words(text)
        for n, pool in short.items():
            for i in range(len(w) - n + 1):
                if " ".join(w[i : i + n]) in pool:
                    return True
        return False

    report: dict[str, Any] = {"bfcl_functions": len(bfuncs), "bfcl_prompts": len(bprompts)}
    drop: dict[str, str] = {}
    examples: list[dict[str, Any]] = []
    for split in ("train", "dev"):
        p = args.data / f"{split}.jsonl"
        if not p.exists():
            continue
        rows = read_jsonl(p)
        hits: Counter = Counter()
        rows_hit: Counter = Counter()
        for r in rows:
            fired: set[str] = set()
            for t in r["tools"]:
                fn = _fn(t)
                d = _desc(fn)
                long_enough = len(d.split()) >= MIN_DESC_WORDS
                if long_enough and d in b_desc_exact:
                    fired.add("desc_exact")
                    if len(examples) < 25:
                        examples.append(
                            {
                                "split": split,
                                "id": r["id"],
                                "rule": "desc_exact",
                                "train_fn": fn.get("name"),
                                "bfcl_fn": b_desc_exact[d],
                            }
                        )
                jac, bi = mh.max_jaccard(d) if long_enough else (0.0, -1)
                if jac > 0.7:
                    fired.add("desc_minhash")
                    if len(examples) < 25 and "desc_exact" not in fired:
                        examples.append(
                            {
                                "split": split,
                                "id": r["id"],
                                "rule": "desc_minhash",
                                "train_fn": fn.get("name"),
                                "bfcl_fn": b_names[bi],
                                "jaccard": round(jac, 3),
                            }
                        )
                pk = _props(fn)
                if pk.count(",") >= 1 and pk in b_props and long_enough:
                    sh = mh.shingles(d)
                    best = max(
                        (len(sh & mh.sets[j]) / max(1, len(sh | mh.sets[j])) for j in b_props[pk]),
                        default=0.0,
                    )
                    if best > 0.5:
                        fired.add("props_desc")
            for u in user_texts(r["messages"]):
                if contained(u):
                    fired.add("prompt_contained")
                if len(words(u)) >= 8 and ng8.max_overlap(u)[0] >= 0.5:
                    fired.add("prompt_ngram8")
            for rule in fired:
                rows_hit[rule] += 1
            func_rules = fired - {"prompt_ngram8"}
            if func_rules:
                drop[r["id"]] = sorted(func_rules)[0]
            if fired:
                hits["any"] += 1
        report[split] = {
            "rows": len(rows),
            "rows_hit_by_rule": dict(rows_hit),
            "rows_hit_by_function_rule": sum(1 for r in rows if r["id"] in drop),
        }

    for name in ("router_pos.jsonl", "router_pos_dev.jsonl"):
        p = args.data / name
        if p.exists():
            rows = read_jsonl(p)
            report[name] = {
                "rows": len(rows),
                "from_rows_hit_by_function_rule": sum(1 for r in rows if r.get("row_id") in drop),
            }
    if args.a1 is not None and args.a1.exists():
        a1 = json.loads(args.a1.read_text(encoding="utf-8"))
        report["a1_exemplars_hit"] = [e["id"] for e in a1["exemplars"] if e["id"] in drop]
    report["examples"] = examples
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if args.drop_list is not None:
        args.drop_list.write_text(json.dumps(drop, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "examples"}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
