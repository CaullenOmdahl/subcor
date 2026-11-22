"""Engineering-habit detectors, benchmark builder and scorer for chat-history units.

The habits come from "Enterprise Engineering Habits" (ChiefGyk3D/Clouds-Against-Humanity wiki,
Apache-2.0), used as a yardstick only: no text from the page is trained on. Each detector is a
mechanical check on agent actions in the user's own transcripts:

  verify_after_change  (#9 verify, then record) next action after an edit/write is a check
  stage_gate           (#6 stage gates)         no second edit before the first was checked
  inspect_first        (precondition of #4/#6)  first action on a change request inspects, not edits
  commit_why           (#1 code with a why)     commit messages explain, not just name
  least_privilege      (#3 least privilege)     no needless sudo, no chmod 777, secrets get 600

Usage:
  python habits.py score <03-capabilities dir>            habit evidence per bucket/split (train weighting)
  python habits.py build <03-capabilities dir> <out.jsonl> benchmark items from the eval splits only
"""

from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path

EDIT = {"edit_file", "write_file", "apply_patch"}
INSPECT = {"read_file", "search_files"}
CHECK_RE = re.compile(
    r"\b(pytest|unittest|go test|cargo (test|check|build|clippy)|npm (test|run (test|build|lint|typecheck))|pnpm "
    r"(test|build|lint)|yarn (test|build)|make( \w+)?\b|ruff|mypy|tsc\b|eslint|shellcheck|node --check|python3? -m "
    r"(py_compile|pytest)|bash -n|git (diff|status|show)|systemctl (status|is-active)|journalctl|curl\b|wget\b|"
    r"--version|--help|test -[efd]|ls\b|cat\b|head\b|tail\b|sed -n|grep\b|rg\b|diff\b|sha256sum|md5sum|wc\b|stat\b)")
INSPECT_CMD_RE = re.compile(r"^\s*(cd [^;&|]+(&&|;)\s*)?(ls|cat|head|tail|sed -n|grep|rg|find|git (status|log|diff|show)|"
                            r"stat|file|wc|tree|which|type|systemctl status|journalctl)\b")
COMMIT_RE = re.compile(r"git commit\b[^\n]*?-m\s+(\"([^\"]*)\"|'([^']*)')")
WHY_RE = re.compile(r"\b(because|so that|so the|to (avoid|prevent|keep|allow|fix|make|stop)|fixes|why|instead|otherwise|"
                    r"caused|prevents?|ensures?)\b", re.I)


def _calls(msgs: list[dict]) -> list[tuple[int, str, dict]]:
    out = []
    for i, m in enumerate(msgs):
        if m["role"] != "assistant":
            continue
        for c in m.get("tool_calls") or []:
            fn = c.get("function") or c
            args = fn.get("arguments")
            out.append((i, fn.get("name"), args if isinstance(args, dict) else {}))
    return out


def is_check(name: str, args: dict) -> bool:
    if name in INSPECT:
        return True
    return name == "run_command" and bool(CHECK_RE.search(args.get("command") or ""))


def is_inspect(name: str, args: dict) -> bool:
    if name in INSPECT:
        return True
    return name == "run_command" and bool(INSPECT_CMD_RE.search(args.get("command") or ""))


def privilege_issues(cmd: str) -> int:
    n = 0
    n += len(re.findall(r"chmod\s+(-R\s+)?0?777\b", cmd))
    n += len(re.findall(r"\bsudo\s+(cat|ls|grep|head|tail|echo|less)\b", cmd))  # read-only commands rarely need root
    return n


def window_habits(rec: dict) -> dict:
    """Habit evidence in one training/eval window (counts; None when the habit had no chance to show)."""
    calls = _calls(rec["messages"])
    h: dict = {}
    edits = [k for k, (_, n, _) in enumerate(calls) if n in EDIT]
    if edits:
        verified = sum(1 for k in edits if k + 1 < len(calls) and is_check(calls[k + 1][1], calls[k + 1][2]))
        followed = sum(1 for k in edits if k + 1 < len(calls))
        h["verify_after_change"] = (verified, followed) if followed else None
        stacked = sum(1 for a, b in zip(edits, edits[1:], strict=False)
                      if not any(is_check(n, x) for _, n, x in calls[a + 1:b]))
        h["stage_gate"] = (len(edits) - 1 - stacked, len(edits) - 1) if len(edits) > 1 else None
    if calls and any(n in EDIT for _, n, _ in calls) and rec["messages"][1]["role"] == "user" \
            and "(Earlier in this task" not in rec["messages"][1]["content"]:
        h["inspect_first"] = (int(is_inspect(calls[0][1], calls[0][2])), 1)
    msgs_commits = [m for _, n, a in calls if n == "run_command" for m in COMMIT_RE.findall(a.get("command") or "")]
    if msgs_commits:
        texts = [m[1] or m[2] for m in msgs_commits]
        h["commit_why"] = (sum(1 for t in texts if WHY_RE.search(t) or len(t) >= 72 or "\n" in t), len(texts))
    cmds = [a.get("command") or "" for _, n, a in calls if n == "run_command"]
    if cmds:
        h["least_privilege"] = (sum(1 for c in cmds if privilege_issues(c) == 0), len(cmds))
    return h


def score(data: Path) -> None:
    for cap in ("tool-calling", "code-writing", "agentic-decisions"):
        for split in ("train", "eval"):
            tot: dict[str, list[int]] = {}
            n = 0
            for line in (data / cap / f"{split}.jsonl").open():
                n += 1
                for k, v in window_habits(json.loads(line)).items():
                    if v:
                        t = tot.setdefault(k, [0, 0, 0])
                        t[0] += v[0]
                        t[1] += v[1]
                        t[2] += 1
            print(f"{cap:18} {split:5} n={n:5} " + "  ".join(
                f"{k}={a / b:.0%}({w}w)" for k, (a, b, w) in sorted(tot.items()) if b))


def build(data: Path, out: Path) -> None:
    """Benchmark items from eval splits: context up to the decision point + what passing means."""
    items, kinds = [], Counter()
    for cap in ("tool-calling", "code-writing", "agentic-decisions"):
        for line in (data / cap / "eval.jsonl").open():
            rec = json.loads(line)
            msgs = rec["messages"]
            calls = _calls(msgs)
            # verify_after_change: cut right after the first edit's tool result
            for i, n, _ in calls:
                if n in EDIT and i + 1 < len(msgs) and msgs[i + 1]["role"] == "tool":
                    j = i + 1
                    while j + 1 < len(msgs) and msgs[j + 1]["role"] == "tool":
                        j += 1
                    if j + 1 < len(msgs) and msgs[j + 1]["role"] == "assistant":
                        items.append({"id": rec["id"] + ":verify", "habit": "verify_after_change", "capability": cap,
                                      "messages": msgs[:j + 1], "tools": rec["tools"]})
                        kinds["verify_after_change"] += 1
                    break
            # inspect_first: a fresh change request (no progress note) whose window later edits
            if calls and any(n in EDIT for _, n, _ in calls) and "(Earlier in this task" not in msgs[1]["content"]:
                items.append({"id": rec["id"] + ":inspect", "habit": "inspect_first", "capability": cap,
                              "messages": msgs[:2], "tools": rec["tools"]})
                kinds["inspect_first"] += 1
    with out.open("w") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    print(json.dumps({"items": len(items), **kinds}))


def passes(habit: str, name: str | None, args: dict | None) -> bool:
    """Does a generated next action satisfy the habit item?"""
    if name is None:
        return False
    if habit == "verify_after_change":
        return is_check(name, args or {})
    if habit == "inspect_first":
        return is_inspect(name, args or {})
    raise ValueError(habit)


if __name__ == "__main__":
    {"score": lambda: score(Path(sys.argv[2])), "build": lambda: build(Path(sys.argv[2]), Path(sys.argv[3]))}[sys.argv[1]]()
