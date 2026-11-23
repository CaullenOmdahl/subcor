"""Build the ~300-prompt general (non-tool) set for the router false-activation probe.

Composition (deterministic, seed 7):
  * 240 from databricks-dolly-15k: 30 per category (8 categories), context inlined as
    "<instruction>\n\nContext:\n<context>"; context capped at 1500 chars, instruction >= 15.
  * 30 GSM8K *train* questions (never test; the GSM8K regression eval uses test).
  * 30 hand-written hard negatives (category `hard_negative`): coding / JSON / API-concept
    prompts that share vocabulary with tool use but ask for no tool call. Reported
    separately; whether a tool-use unit *should* fire on them is a judgement call.

Disjointness: prompts also used as router-gate negatives would make the probe in-sample,
so `--exclude FILE` (repeatable; .txt one prompt per line, or .jsonl with prompt /
instruction / question / messages[-1].content) removes exact matches (whitespace/case
normalized) and the replacement is drawn from the same stratum. The data agent's
`neg_prompts.txt` is excluded by default when present. The selected dolly / gsm8k-train
row indices are written to probe_set_ids.json (commit this, not the prompts) so gate
training can exclude them in turn.

  python -I build_probe_set.py --dolly .../dolly15k.jsonl --gsm8k-train .../train.parquet \
      --out .../general/data/router_probe_prompts.jsonl
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

import common  # noqa: E402

SEED = 7
PER_CATEGORY = 30
N_GSM8K = 30
MAX_CONTEXT = 1500
IDS_FILE = Path(__file__).resolve().parent / "probe_set_ids.json"

HARD_NEGATIVES = [
    "Write a Python function that checks whether a string is a palindrome.",
    "Explain what a REST API is to a beginner.",
    "What is the difference between a function and a method in Python?",
    "Convert this list to JSON: name Alice, age 30, city Paris.",
    "Write a SQL query that returns the ten most recent orders from an orders table.",
    "In mathematics, what does it mean for a function to be injective?",
    "Explain how OAuth 2.0 authorization codes work.",
    "Write a bash one-liner that counts the lines in every .txt file in a directory.",
    "What does HTTP status code 429 mean and how should a client handle it?",
    "Describe what a webhook is and give an everyday analogy.",
    "Refactor this JavaScript to use async/await: fetch(url).then(r => r.json()).then(console.log)",
    "What is a JSON schema and why would you use one?",
    "Write a regular expression that matches US ZIP codes.",
    "Explain recursion using the factorial function as an example.",
    "What are the pros and cons of GraphQL compared with REST?",
    "Write a short docstring for a function that computes the median of a list.",
    "How do I reverse a linked list in C? Explain the idea, no code needed.",
    "What is the time complexity of binary search and why?",
    "Explain what an API rate limit is in two sentences.",
    "Write a haiku about debugging a program late at night.",
    "Summarize what the `map` function does in functional programming.",
    "What is the difference between `let` and `const` in JavaScript?",
    "Explain what a callback function is.",
    "Give three tips for writing clear commit messages.",
    "What is dependency injection, in plain English?",
    "Write a YAML snippet describing a person with a name and two hobbies.",
    "Explain the difference between authentication and authorization.",
    "What does the acronym CRUD stand for?",
    "Describe how a hash table handles collisions.",
    "Write pseudocode for bubble sort.",
]

_WS = re.compile(r"\s+")


def norm(text: str) -> str:
    return _WS.sub(" ", text).strip().lower()


def first_paragraph(text: str) -> str:
    """The instruction part of a "<instruction>\n\n<context>" prompt."""
    return norm(text.split("\n\n", 1)[0])


def dolly_forms(instr: str, ctx: str) -> set[str]:
    """Every way a dolly row is rendered as a prompt by this repo's builders: this file
    ("<instr>\n\nContext:\n<ctx>") and data/build_router_calib.py ("<instr>\n\n<ctx>")."""
    forms = {norm(instr)}
    if ctx:
        forms |= {norm(f"{instr}\n\nContext:\n{ctx}"), norm(f"{instr}\n\n{ctx}")}
    return forms


def load_exclusions(paths: list[Path]) -> set[str]:
    out: set[str] = set()
    for p in paths:
        if not p.exists():
            print(f"  exclude file missing (skipped): {p}", file=sys.stderr)
            continue
        if p.suffix == ".jsonl":
            for r in common.read_jsonl(p):
                for key in ("prompt", "instruction", "question", "text"):
                    if isinstance(r.get(key), str):
                        out.add(norm(r[key]))
                msgs = r.get("messages")
                if isinstance(msgs, list):
                    for m in msgs:
                        if isinstance(m, dict) and m.get("role") == "user":
                            out.add(norm(str(m.get("content", ""))))
        else:
            out.update(
                norm(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()
            )
    return out


def load_gsm8k_train(path: Path) -> list[str]:
    if path.suffix == ".parquet":
        import pyarrow.parquet as pq

        return [
            str(q) for q in pq.read_table(path, columns=["question"]).column("question").to_pylist()
        ]
    return [r["question"] for r in common.read_jsonl(path)]


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--dolly", type=Path, required=True)
    ap.add_argument("--gsm8k-train", type=Path, required=True)
    ap.add_argument("--exclude", type=Path, action="append", default=[])
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    excl = load_exclusions(args.exclude)
    # Exclusion by SOURCE ROW, not only by exact prompt text: a dolly row whose
    # instruction matches the first paragraph of an excluded prompt is the same row
    # rendered with or without context (build_router_calib.py writes summarization rows
    # as "<instr>\n\n<ctx>", this file as "<instr>\n\nContext:\n<ctx>", so exact-text
    # matching alone misses them).
    excl_instr = set(excl)  # `excl` is whitespace-normalized, so take paragraphs from raw rows
    for p in args.exclude:
        if p.exists() and p.suffix == ".jsonl":
            for r in common.read_jsonl(p):
                if isinstance(r.get("prompt"), str):
                    excl_instr.add(first_paragraph(r["prompt"]))
    excluded_rows: dict[str, list[int]] = {"dolly": [], "gsm8k_train": []}
    rng = random.Random(SEED)
    items: list[dict[str, Any]] = []

    dolly = common.read_jsonl(args.dolly)
    by_cat: dict[str, list[int]] = {}
    for i, r in enumerate(dolly):
        if len(r["instruction"].strip()) < 15 or len(r.get("context") or "") > MAX_CONTEXT:
            continue
        by_cat.setdefault(r["category"], []).append(i)
    n_excluded = 0
    for cat in sorted(by_cat):
        pool = by_cat[cat][:]
        rng.shuffle(pool)
        picked = 0
        for i in pool:
            r = dolly[i]
            instr = r["instruction"].strip()
            ctx = (r.get("context") or "").strip()
            prompt = f"{instr}\n\nContext:\n{ctx}" if ctx else instr
            if (dolly_forms(instr, ctx) & excl) or norm(instr) in excl_instr:
                n_excluded += 1
                excluded_rows["dolly"].append(i)
                continue
            items.append(
                {
                    "id": f"dolly-{i}",
                    "source": "dolly",
                    "category": cat,
                    "src_index": i,
                    "prompt": prompt,
                }
            )
            picked += 1
            if picked == PER_CATEGORY:
                break

    gsm = load_gsm8k_train(args.gsm8k_train)
    order = list(range(len(gsm)))
    rng.shuffle(order)
    picked = 0
    for i in order:
        if norm(gsm[i]) in excl:
            n_excluded += 1
            excluded_rows["gsm8k_train"].append(i)
            continue
        items.append(
            {
                "id": f"gsm8ktrain-{i}",
                "source": "gsm8k_train",
                "category": "math",
                "src_index": i,
                "prompt": gsm[i],
            }
        )
        picked += 1
        if picked == N_GSM8K:
            break

    for j, p in enumerate(HARD_NEGATIVES):
        if norm(p) in excl:
            n_excluded += 1
            continue
        items.append(
            {
                "id": f"hard-{j}",
                "source": "handwritten",
                "category": "hard_negative",
                "src_index": j,
                "prompt": p,
            }
        )

    # Verify: no probe prompt (full text or instruction part) is in any excluded file.
    overlap = sorted(
        it["id"]
        for it in items
        if norm(it["prompt"]) in excl or first_paragraph(it["prompt"]) in excl_instr
    )
    if overlap:
        raise SystemExit(f"probe still overlaps the excluded files: {overlap[:10]}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as fh:
        for it in items:
            fh.write(json.dumps(it, ensure_ascii=False) + "\n")
    counts: dict[str, int] = {}
    for it in items:
        counts[it["category"]] = counts.get(it["category"], 0) + 1
    IDS_FILE.write_text(
        json.dumps(
            {
                "seed": SEED,
                "n": len(items),
                "counts": counts,
                "excluded_matches": n_excluded,
                "exclude_files": [str(p.name) for p in args.exclude],
                "exclude_files_sha256": {
                    p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in args.exclude
                    if p.exists()
                },
                "excluded_source_rows": excluded_rows,
                "overlap_with_excluded_after_build": len(overlap),
                "probe_sha256": hashlib.sha256(
                    "".join(json.dumps(it, ensure_ascii=False) + "\n" for it in items).encode(
                        "utf-8"
                    )
                ).hexdigest(),
                "dolly_indices": sorted(it["src_index"] for it in items if it["source"] == "dolly"),
                "gsm8k_train_indices": sorted(
                    it["src_index"] for it in items if it["source"] == "gsm8k_train"
                ),
                "prompt_sha16": [common.sha_text(it["prompt"]) for it in items],
            }
        )
        + "\n"
    )
    print(
        json.dumps(
            {
                "n": len(items),
                "counts": counts,
                "excluded_matches": n_excluded,
                "out": str(args.out),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
