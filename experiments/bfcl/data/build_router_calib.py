"""Router calibration prompts for the tool-use Skill Unit gate.

router_pos.jsonl     ~300 tool requests (tools present) drawn from train.jsonl: the latest
                     user message before an assistant tool_call turn (= the text SUBCOR
                     serving embeds), stratified over the calling categories.
router_pos_dev.jsonl the same construction over dev.jsonl (held out from the unit's training).
router_neg.jsonl     ~300 general prompts with no tools: chat/QA/brainstorm/writing/
                     classification/summarisation from databricks-dolly-15k (CC-BY-SA-3.0)
                     plus grade-school math from the GSM8K *train* split (MIT). Negatives
                     are dropped if they match IFEval (google/IFEval), the GSM8K test split,
                     or any BFCL v4 prompt by 13-gram containment >= 0.8 / Jaccard > 0.7.
Each row: {"prompt", "split": "fit"|"heldout" (80/20), "category", "source", ...}.
train_unit.py reads the "prompt" key (--pos-prompts / --neg-prompts).

Usage:
  $VENV -I experiments/bfcl/data/build_router_calib.py --data $SCRATCH/bfcl/data \
      --raw $SCRATCH/bfcl/raw --bfcl-data $VENV_SITE/bfcl_eval/data
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from build_tool_data import _norm_prompt, load_bfcl  # noqa: E402
from common import MinHashLSH, NgramIndex  # noqa: E402

NEG_MIX = {  # dolly category -> count ; "gsm8k_train" -> count
    "open_qa": 70,
    "general_qa": 50,
    "brainstorming": 35,
    "creative_writing": 45,
    "classification": 20,
    "summarization": 20,
    "gsm8k_train": 60,
}
POS_N = 300
CALL_CATS = ("single", "multiple", "parallel", "parallel_multiple", "multi_step", "multi_turn", "multi_turn_clarify")


def load_eval_prompts(raw: Path) -> list[str]:
    import pyarrow.parquet as pq  # noqa: PLC0415

    out = [json.loads(line)["prompt"] for line in (raw / "ifeval/ifeval_input_data.jsonl").read_text().splitlines() if line.strip()]
    out += [str(q) for q in pq.read_table(raw / "gsm8k/main/test-00000-of-00001.parquet").column("question").to_pylist()]
    return out


def routing_prompts(row: dict[str, Any]) -> list[str]:
    msgs, out, last_user = row["messages"], [], None
    for m in msgs:
        if m["role"] == "user":
            last_user = m["content"]
        if m["role"] == "assistant" and m.get("tool_calls") and last_user:
            out.append(last_user)
            last_user = None  # one per user turn
    return out


def pick_pos(rows: list[dict[str, Any]], n: int, rng: random.Random) -> list[dict[str, Any]]:
    by_cat: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        if r["category"] in CALL_CATS:
            by_cat[r["category"]].append(r)
    for v in by_cat.values():
        rng.shuffle(v)
    seen: set[str] = set()
    out: list[dict[str, Any]] = []
    cats = sorted(by_cat)
    while len(out) < n and any(by_cat[c] for c in cats):
        for c in cats:
            if not by_cat[c] or len(out) >= n:
                continue
            r = by_cat[c].pop()
            ps = [p for p in routing_prompts(r) if len(p) <= 4000]
            if not ps:
                continue
            p = rng.choice(ps)
            k = _norm_prompt(p)
            if k in seen:
                continue
            seen.add(k)
            out.append({"prompt": p, "category": r["category"], "source": r["source"], "row_id": r["id"],
                        "tool_names": [t["function"]["name"] for t in r["tools"]]})
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--raw", type=Path, required=True)
    ap.add_argument("--bfcl-data", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=20261121)
    args = ap.parse_args()
    rng = random.Random(args.seed)

    def rd(p: Path) -> list[dict[str, Any]]:
        return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]

    pos = pick_pos(rd(args.data / "train.jsonl"), POS_N, rng)
    pos_dev = pick_pos(rd(args.data / "dev.jsonl"), 100, rng)

    # contamination indexes for negatives: IFEval + GSM8K test + BFCL prompts
    ev = load_eval_prompts(args.raw)
    bprompts, _, _ = load_bfcl(args.bfcl_data)
    ng, mh = NgramIndex(13), MinHashLSH()
    for p in ev + bprompts:
        ng.add(p)
        mh.add(p)
    tool_side = {_norm_prompt(p["prompt"]) for p in pos + pos_dev}

    import pyarrow.parquet as pq  # noqa: PLC0415

    cands: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for line in (args.raw / "dolly/databricks-dolly-15k.jsonl").read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        cat = r.get("category")
        if cat not in NEG_MIX:
            continue
        text = r["instruction"].strip()
        if cat == "summarization":
            if not r.get("context"):
                continue
            text = f"{text}\n\n{r['context'].strip()}"
        elif r.get("context"):
            continue  # keep the no-context prompts for the other categories
        if 15 <= len(text) <= 2500:
            cands[cat].append((text, "databricks/databricks-dolly-15k"))
    for q in pq.read_table(args.raw / "gsm8k/main/train-00000-of-00001.parquet").column("question").to_pylist():
        cands["gsm8k_train"].append((str(q), "openai/gsm8k (train split)"))

    drops: Counter = Counter()
    neg: list[dict[str, Any]] = []
    seen: set[str] = set()
    for cat, n in NEG_MIX.items():
        pool = cands[cat]
        rng.shuffle(pool)
        took = 0
        for text, src in pool:
            if took >= n:
                break
            k = _norm_prompt(text)
            if k in seen or k in tool_side:
                drops["duplicate"] += 1
                continue
            if ng.max_overlap(text)[0] >= 0.8:
                drops["ngram13_vs_ifeval_gsm8ktest_bfcl"] += 1
                continue
            if mh.max_jaccard(text)[0] > 0.7:
                drops["minhash_vs_ifeval_gsm8ktest_bfcl"] += 1
                continue
            seen.add(k)
            neg.append({"prompt": text, "category": "math" if cat == "gsm8k_train" else cat, "source": src})
            took += 1

    def write(name: str, rows: list[dict[str, Any]]) -> None:
        rng.shuffle(rows)
        cut = int(round(0.8 * len(rows)))
        with (args.data / name).open("w", encoding="utf-8") as fh:
            for i, r in enumerate(rows):
                fh.write(json.dumps({**r, "split": "fit" if i < cut else "heldout"}, ensure_ascii=False) + "\n")

    write("router_pos.jsonl", pos)
    write("router_pos_dev.jsonl", pos_dev)
    write("router_neg.jsonl", neg)
    stats = {
        "pos": {"n": len(pos), "by_category": dict(Counter(p["category"] for p in pos))},
        "pos_dev": {"n": len(pos_dev), "by_category": dict(Counter(p["category"] for p in pos_dev))},
        "neg": {"n": len(neg), "by_category": dict(Counter(p["category"] for p in neg)),
                "licences": {"databricks/databricks-dolly-15k": "cc-by-sa-3.0", "openai/gsm8k": "mit"},
                "decontam_drops": dict(drops), "checked_against": "google/IFEval (541), GSM8K test (1319), BFCL v4 prompts"},
    }
    (args.data / "router_stats.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()
