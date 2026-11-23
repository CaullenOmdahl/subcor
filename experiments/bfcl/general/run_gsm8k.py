"""GSM8K test split, fixed 500-item seeded subset, against an OpenAI-compatible chat endpoint.

Subset: `random.Random(SEED).sample(range(1319), 500)` sorted, SEED=1234; the index list is
committed as gsm8k_subset_seed1234.json and verified against the data on every run.

Prompt (zero-shot CoT, one user turn):
    Solve the following math problem step by step. At the end, write the final answer
    on its own line in the format "#### <number>".

    Question: <question>

Scoring (exact match on the final number, numeric equality after stripping `,` `$` and a
trailing `.`), two standard extractors as in lm-eval's gsm8k tasks:
  * strict:   last "#### <number>" in the response
  * flexible: last number anywhere in the response (primary metric; robust to format slips)
Gold = the number after "####" in the reference answer.

  python -I run_gsm8k.py --base-url http://127.0.0.1:8502/v1 --model local \
      --data .../general/data/gsm8k_test.jsonl --out .../results/gsm8k_base.jsonl
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

import common  # noqa: E402

SEED = 1234
N_SUBSET = 500
N_TEST = 1319
SUBSET_FILE = Path(__file__).resolve().parent / f"gsm8k_subset_seed{SEED}.json"
PROMPT_TEMPLATE_VERSION = "gsm8k-zs-cot-hash-v1"
PROMPT = (
    "Solve the following math problem step by step. At the end, write the final answer "
    'on its own line in the format "#### <number>".\n\nQuestion: {question}'
)

_NUM = r"-?[$]?\s?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?"
STRICT_RE = re.compile(r"####\s*\**\s*(" + _NUM + r")")
FLEX_RE = re.compile(_NUM)


def subset_indices() -> list[int]:
    idx = sorted(random.Random(SEED).sample(range(N_TEST), N_SUBSET))
    if SUBSET_FILE.exists():
        frozen = json.loads(SUBSET_FILE.read_text())["indices"]
        if frozen != idx:
            raise SystemExit(f"{SUBSET_FILE} disagrees with the seeded sample; refusing to run")
    return idx


def normalize(num: str | None) -> Decimal | None:
    if num is None:
        return None
    s = num.replace(",", "").replace("$", "").replace(" ", "").rstrip(".")
    try:
        return Decimal(s)
    except InvalidOperation:
        return None


def gold_of(answer: str) -> Decimal:
    g = normalize(answer.split("####")[-1].strip())
    if g is None:
        raise ValueError(f"unparseable gold: {answer[-80:]!r}")
    return g


def extract(response: str) -> tuple[str | None, str | None]:
    strict = STRICT_RE.findall(response)
    flex = FLEX_RE.findall(response)
    return (strict[-1] if strict else None, flex[-1] if flex else None)


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    common.add_endpoint_args(ap)
    ap.add_argument("--data", type=Path, required=True, help="GSM8K main/test jsonl (1319 rows)")
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument(
        "--write-subset", action="store_true", help="(re)write the frozen index file and exit"
    )
    args = ap.parse_args()

    cfg = common.config_from_args(args)
    docs = common.read_jsonl(args.data)
    if len(docs) != N_TEST:
        raise SystemExit(f"expected {N_TEST} GSM8K test rows, got {len(docs)}")
    if args.write_subset:
        idx = sorted(random.Random(SEED).sample(range(N_TEST), N_SUBSET))
        SUBSET_FILE.write_text(
            json.dumps(
                {
                    "dataset": "openai/gsm8k main test",
                    "seed": SEED,
                    "recipe": "sorted(random.Random(seed).sample(range(1319), 500))",
                    "question_sha16": [common.sha_text(docs[i]["question"]) for i in idx],
                    "indices": idx,
                }
            )
            + "\n"
        )
        print(f"wrote {SUBSET_FILE}")
        return 0
    idx = subset_indices()
    if SUBSET_FILE.exists():
        shas = json.loads(SUBSET_FILE.read_text()).get("question_sha16")
        if shas and shas != [common.sha_text(docs[i]["question"]) for i in idx]:
            raise SystemExit("GSM8K data file does not match the frozen subset hashes")
    items = [{"id": i, "doc": docs[i]} for i in idx]
    if args.limit is not None:
        items = items[: args.limit]
    fp = common.fingerprint_of(
        "gsm8k", cfg.describe(), args.max_tokens, PROMPT_TEMPLATE_VERSION, args.run_label
    )

    def worker(it: dict[str, Any]) -> dict[str, Any]:
        doc = it["doc"]
        gold = gold_of(doc["answer"])
        rec = common.chat(
            cfg,
            [{"role": "user", "content": PROMPT.format(question=doc["question"])}],
            max_tokens=args.max_tokens,
        )
        row: dict[str, Any] = {
            "id": it["id"],
            "task": "gsm8k",
            "run_label": args.run_label,
            "model": cfg.model,
            "ok": rec["ok"],
            "gold": str(gold),
        }
        if not rec["ok"]:
            row["error"] = rec["error"]
            return row
        s_raw, f_raw = extract(rec["text"])
        s_val, f_val = normalize(s_raw), normalize(f_raw)
        row.update(
            {
                "response": rec["text"],
                "had_think": rec["had_think"],
                "finish_reason": rec["finish_reason"],
                "completion_tokens": rec["usage"].get("completion_tokens"),
                "latency_s": rec["latency_s"],
                "selected_units": rec["selected_units"],
                "subcor": rec["subcor"],
                "pred_strict": None if s_val is None else str(s_val),
                "pred_flexible": None if f_val is None else str(f_val),
                "exact_match_strict": s_val is not None and s_val == gold,
                "exact_match_flexible": f_val is not None and f_val == gold,
            }
        )
        return row

    print(
        f"GSM8K: {len(items)} items -> {args.out} ({cfg.base_url}, model={cfg.model})",
        file=sys.stderr,
    )
    rows = common.run_items(
        items, worker, out_path=args.out, concurrency=args.concurrency, fingerprint=fp
    )
    ok = [r for r in rows if r.get("ok")]
    n = max(len(ok), 1)
    summary = {
        "task": "gsm8k",
        "subset": f"seed{SEED} n={N_SUBSET} of test",
        "endpoint": cfg.describe(),
        "run_label": args.run_label,
        "max_tokens": args.max_tokens,
        "limit": args.limit,
        "fingerprint": fp,
        "n_items": len(ok),
        "n_failed_requests": len(rows) - len(ok),
        "exact_match_flexible": sum(r["exact_match_flexible"] for r in ok) / n,
        "exact_match_strict": sum(r["exact_match_strict"] for r in ok) / n,
        "finish_length": sum(1 for r in ok if r.get("finish_reason") == "length"),
        "had_think_block": sum(1 for r in ok if r.get("had_think")),
        "routing": common.routing_summary(ok),
    }
    common.write_summary(common.default_summary_path(args.out, args.summary), summary)
    print(json.dumps({k: v for k, v in summary.items() if k != "endpoint"}, indent=2))
    return 0 if summary["n_failed_requests"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
