"""Smoke test: replay the first assistant turn of N dev rows against an OpenAI-compatible
/v1/chat/completions (llama-server --jinja) with the row's tools, and compare with gold.

Checks the data is servable as-is (tools accepted, tool_calls parsed back) and gives a
rough first-turn agreement number. Not a benchmark.

  python3 -I experiments/bfcl/data/smoke_serve.py --data $SCRATCH/bfcl/data/dev.jsonl \
      --url http://127.0.0.1:8577 --n 40
"""

from __future__ import annotations

import argparse
import json
import random
import time
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--url", default="http://127.0.0.1:8577")
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-tokens", type=int, default=384)
    args = ap.parse_args()
    rows = [json.loads(x) for x in args.data.read_text().splitlines() if x.strip()]
    by_cat = defaultdict(list)
    for r in rows:
        by_cat[r["category"]].append(r)
    rng = random.Random(args.seed)
    pick = []
    cats = sorted(by_cat)
    while len(pick) < args.n and any(by_cat.values()):
        for c in cats:
            if by_cat[c] and len(pick) < args.n:
                pick.append(by_cat[c].pop(rng.randrange(len(by_cat[c]))))
    stats: dict[str, Counter] = defaultdict(Counter)
    t0 = time.time()
    for r in pick:
        k = next(i for i, m in enumerate(r["messages"]) if m["role"] == "assistant")
        gold = r["messages"][k]
        body = {
            "messages": r["messages"][:k],
            "tools": r["tools"],
            "temperature": 0,
            "max_tokens": args.max_tokens,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        req = urllib.request.Request(
            args.url + "/v1/chat/completions",
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
        )
        s = stats[r["category"]]
        s["n"] += 1
        try:
            with urllib.request.urlopen(req, timeout=300) as resp:
                out = json.loads(resp.read())
        except Exception as e:  # noqa: BLE001
            s["http_error"] += 1
            print("ERR", r["id"], str(e)[:200])
            continue
        s["accepted"] += 1
        msg = out["choices"][0]["message"]
        got = [c["function"]["name"] for c in msg.get("tool_calls") or []]
        want = [c["function"]["name"] for c in gold.get("tool_calls") or []]

        def _obj(a: str) -> bool:
            try:
                return isinstance(json.loads(a), dict)
            except json.JSONDecodeError:
                return False

        args_ok = all(_obj(c["function"]["arguments"]) for c in msg.get("tool_calls") or [])
        s["got_tool_calls"] += bool(got)
        s["args_json_ok"] += bool(got) and args_ok
        s["call_vs_nocall_agree"] += bool(got) == bool(want)
        s["names_match"] += sorted(got) == sorted(want)
    tot = Counter()
    for s in stats.values():
        tot.update(s)
    print(
        json.dumps(
            {
                "per_category": {c: dict(v) for c, v in sorted(stats.items())},
                "total": dict(tot),
                "seconds": round(time.time() - t0, 1),
            },
            indent=1,
        )
    )


if __name__ == "__main__":
    main()
