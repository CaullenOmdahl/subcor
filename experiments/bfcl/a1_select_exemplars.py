"""Freeze the A1 (3-shot) exemplars named in PREREGISTRATION.md.

Deterministic: picks one `single`, one `parallel_multiple` and one `irrelevance` row from
the decontaminated train.jsonl with `random.Random(20261122)`, keeping only short
single-turn rows (user -> assistant) whose tool names do not occur anywhere in the BFCL v4
function docs. Writes the ids, the SHA-256 of each source row and the exact system-prompt
text that the A1 arm prepends.

    python3 -I experiments/bfcl/a1_select_exemplars.py --train $S/data/train.jsonl \
        --bfcl-data $BD --out experiments/bfcl/a1_exemplars.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

SEED = 20261122
CATEGORIES = ("single", "parallel_multiple", "irrelevance")
MAX_EST_TOKENS = 450

HEADER = (
    "You can call functions. Below are three worked examples of how to respond. "
    "They use different tools from the ones you have now; only use the tools provided "
    "in this conversation. Call a function only when one of the provided tools fits the "
    "request, and fill every argument with a value taken from the request. If no provided "
    "tool fits, answer in text without calling any function.\n"
)


def bfcl_function_names(bfcl_data: Path) -> set[str]:
    names: set[str] = set()

    def walk(obj: object) -> None:
        if isinstance(obj, dict):
            if "name" in obj and "parameters" in obj and isinstance(obj["name"], str):
                names.add(obj["name"])
                names.add(obj["name"].replace(".", "_"))
            for v in obj.values():
                walk(v)
        elif isinstance(obj, list):
            for v in obj:
                walk(v)

    for f in sorted(bfcl_data.rglob("*.json")):
        with f.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    try:
                        walk(json.loads(line))
                    except json.JSONDecodeError:
                        continue
    return names


def eligible(row: dict, bfcl_names: set[str]) -> bool:
    msgs = row["messages"]
    if row["est_tokens"] > MAX_EST_TOKENS or len(msgs) != 2:
        return False
    if msgs[0]["role"] != "user" or msgs[1]["role"] != "assistant":
        return False
    tool_names = {t["function"]["name"] for t in row["tools"]}
    if tool_names & bfcl_names:
        return False
    calls = msgs[1].get("tool_calls") or []
    if row["category"] == "irrelevance":
        return not calls and bool((msgs[1].get("content") or "").strip())
    if row["category"] == "single":
        return len(calls) == 1
    return len(calls) >= 2 and len({c["function"]["name"] for c in calls}) >= 2


def render(rows: list[dict]) -> str:
    parts = [HEADER]
    for k, row in enumerate(rows, 1):
        user, asst = row["messages"]
        tools = [t["function"] for t in row["tools"]]
        parts.append(f"\n### Example {k}")
        parts.append("Available tools: " + json.dumps(tools, ensure_ascii=False))
        parts.append("User: " + user["content"].strip())
        calls = asst.get("tool_calls") or []
        if calls:
            rendered = [
                {"name": c["function"]["name"], "arguments": json.loads(c["function"]["arguments"])}
                for c in calls
            ]
            parts.append("Assistant function call(s): " + json.dumps(rendered, ensure_ascii=False))
        else:
            parts.append("Assistant (no function call): " + asst["content"].strip())
    parts.append("\n### End of examples")
    return "\n".join(parts) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=Path, required=True)
    ap.add_argument("--bfcl-data", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    bfcl_names = bfcl_function_names(a.bfcl_data)
    raw_lines = [ln for ln in a.train.read_text(encoding="utf-8").splitlines() if ln.strip()]
    rows = [json.loads(ln) for ln in raw_lines]
    rng = random.Random(SEED)
    chosen, hashes = [], []
    for cat in CATEGORIES:
        pool = [i for i, r in enumerate(rows) if r["category"] == cat and eligible(r, bfcl_names)]
        if not pool:
            raise SystemExit(f"no eligible exemplar for {cat}")
        i = rng.choice(sorted(pool))
        chosen.append(rows[i])
        hashes.append(hashlib.sha256(raw_lines[i].encode("utf-8")).hexdigest())
    text = render(chosen)
    out = {
        "seed": SEED,
        "train_sha256": hashlib.sha256(a.train.read_bytes()).hexdigest(),
        "bfcl_function_names_checked": len(bfcl_names),
        "exemplars": [
            {"id": r["id"], "category": r["category"], "source": r["source"], "row_sha256": h}
            for r, h in zip(chosen, hashes, strict=True)
        ],
        "system_prompt": text,
        "system_prompt_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "system_prompt_chars": len(text),
    }
    a.out.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in out.items() if k != "system_prompt"}, indent=2))


if __name__ == "__main__":
    main()
