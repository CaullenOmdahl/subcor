"""Build capability datasets (canonical multi-turn JSONL) from scrubbed sessions.

Output under <out>/03-capabilities/<capability>/{train,eval,negative_prompts}.jsonl + stats.json
and <out>/03-capabilities/manifest.json. Each record:
  {id, session_id, source, project, timestamp, capability, signal_type, tools, messages,
   tokens_qwen35}
Windows are <= MAX_TOKENS Qwen3.5 tokens (chat template incl. tool schemas). Filter on
tokens_qwen35 <= 1400 for the single-turn-era trainer.

Usage: python build.py <scrubbed_dir> <out_dir> <tokenizer_dir> <blocklist.tsv>
Every final record is re-scrubbed (unwrapping JSON tool output can expose text the first
pass saw only in escaped form).
"""

from __future__ import annotations

import hashlib
import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import scrub  # noqa: E402

MAX_TOKENS = 2048
TOOL_OUT_TOKENS = 400
MAX_PROMPT_TOKENS = 1000
PROJECT_CAP = 0.20
EVAL_FRAC = 0.20

SYSTEM = ("You are a careful, concise software and operations agent working in a Linux shell. "
          "Inspect before changing things, make minimal edits, and verify results before reporting.")

TOOL_SCHEMAS = {
    "run_command": ("Run a shell command and return its output.", {"command": "string"}, ["command"]),
    "read_file": ("Read a file, optionally a line range.", {"path": "string", "offset": "integer", "limit": "integer"}, ["path"]),
    "edit_file": ("Replace exact text in a file.", {"path": "string", "old_text": "string", "new_text": "string"}, ["path", "old_text", "new_text"]),
    "write_file": ("Create or overwrite a file.", {"path": "string", "content": "string"}, ["path", "content"]),
    "apply_patch": ("Apply a patch in *** Begin Patch format.", {"patch": "string"}, ["patch"]),
    "search_files": ("Search file names or contents for a pattern.", {"pattern": "string", "path": "string"}, ["pattern"]),
    "web_search": ("Search the web.", {"query": "string"}, ["query"]),
    "web_fetch": ("Fetch a web page.", {"url": "string"}, ["url"]),
    "ask_user": ("Ask the user a clarifying question.", {"question": "string"}, ["question"]),
}
TOOLS = [{"type": "function", "function": {"name": n, "description": d,
          "parameters": {"type": "object", "properties": {k: {"type": t} for k, t in p.items()}, "required": r}}}
         for n, (d, p, r) in TOOL_SCHEMAS.items()]
EDIT_TOOLS = {"edit_file", "write_file", "apply_patch"}
TOOL_BY_NAME = {t["function"]["name"]: t for t in TOOLS}


def tools_for(names: set[str], seed: str) -> list[dict]:
    """Tools used in the window plus two deterministic decoys (keeps selection non-trivial)."""
    rest = sorted(set(TOOL_BY_NAME) - names, key=lambda n: hashlib.sha256((seed + n).encode()).hexdigest())
    keep = (names & set(TOOL_BY_NAME)) | set(rest[:2])
    return [TOOL_BY_NAME[n] for n in TOOL_BY_NAME if n in keep]
CORR_RE = re.compile(r"^(no\b|nope|not\b|that'?s (wrong|not)|wrong|don'?t|do not|stop|actually|wait|"
                     r"i (said|asked|meant|wanted)|why (did|are) you|you (didn'?t|missed|forgot)|instead|undo|revert)", re.I)
APOLOGY_RE = re.compile(r"^(you'?re (right|correct)|my (mistake|bad)|sorry|apologies|good catch)[^\n.!]*[.!:—-]+\s*", re.I)
ERRLINE_RE = re.compile(r"(error|exception|traceback|failed|fatal|\[exit -?\d+\]|exit code|panic)", re.I)


class Tok:
    def __init__(self, path: str):
        from transformers import AutoTokenizer
        self.t = AutoTokenizer.from_pretrained(path, local_files_only=True)

    def n(self, s: str) -> int:
        return len(self.t.encode(s, add_special_tokens=False))

    def chat(self, messages: list[dict], tools: list[dict] | None = None) -> int:
        return self.n(self.t.apply_chat_template(messages, tools=tools or TOOLS, tokenize=False))

    def truncate(self, s: str, budget: int) -> str:
        if len(s) < budget * 2 or self.n(s) <= budget:
            return s
        lines = s.splitlines()
        err = [ln for ln in lines if ERRLINE_RE.search(ln)][-6:]
        head, tail = [], []
        used = sum(self.n(x) for x in err) + 12
        for ln in lines:
            c = self.n(ln) + 1
            if used + c > budget * 0.45:
                break
            head.append(ln)
            used += c
        for ln in reversed(lines[len(head):]):
            c = self.n(ln) + 1
            if used + c > budget:
                break
            tail.insert(0, ln)
            used += c
        mid = [x for x in err if x not in head and x not in tail]
        return "\n".join(head + ["[... output truncated ...]"] + mid + (["[...]"] if mid else []) + tail)


BOILER_RE = re.compile(r"\s*\(file state is current in your context[^)]*\)|"
                       r"\s*Here's the result of running `cat -n` on a snippet of the edited file:")


def clean_tool_output(txt: str) -> str:
    t = txt.strip()
    if t.startswith("{") and '"output"' in t[:400]:
        try:
            j = json.loads(t)
            while isinstance(j, dict) and "output" not in j and ("result" in j or "value" in j):
                j = j.get("result", j.get("value"))  # Promise.allSettled wrappers
            if isinstance(j, dict) and j.get("status") == "rejected":
                return "Error: " + str(j.get("reason", ""))[:2000]
            if isinstance(j, dict) and "output" in j:
                code = j.get("exit_code", j.get("exitCode"))
                t = str(j["output"]).strip() + (f"\n[exit {code}]" if code not in (None, 0) else "")
        except Exception:
            pass
    return BOILER_RE.sub("", t).strip()


def project_key(p: str | None) -> str:
    p = (p or "?").rstrip("/")
    p = re.sub(r"/\.codex/worktrees/[^/]+/", "/", p)
    p = re.sub(r"/\.paseo/worktrees/[^/]+/", "/", p)
    return p.split("/")[-1] or "home"


def to_msgs(step: dict, results: list[dict]) -> list[dict]:
    step = step["a"]
    a = {"role": "assistant", "content": APOLOGY_RE.sub("", step.get("content") or "")}
    if step["tool_calls"]:
        a["tool_calls"] = [{"id": c["id"], "type": "function",
                            "function": {"name": c["name"], "arguments": c["arguments"]}} for c in step["tool_calls"]]
    return [a] + [{"role": "tool", "tool_call_id": r["tool_call_id"], "name": r["name"], "content": r["content"]}
                  for r in results]


def turns_of(session: dict) -> list[dict]:
    turns, cur = [], None
    for e in session["events"]:
        if e["role"] == "user":
            cur = {"user": e["content"], "ts": e.get("ts"), "steps": [], "interrupted": False}
            turns.append(cur)
        elif e["role"] == "break":
            if cur is not None and e["reason"] in ("user_interrupt", "harness_error"):
                cur["interrupted"] = True
            cur = None if e["reason"] == "compaction" else cur
        elif cur is None:
            continue
        elif e["role"] == "assistant":
            cur["steps"].append({"a": e, "results": []})
        elif e["role"] == "tool" and cur["steps"]:
            cur["steps"][-1]["results"].append(e)
    return turns


def is_correction(prev: dict, nxt: dict) -> bool:
    txt = nxt["user"].strip()
    if len(txt) > 240:
        return False
    if CORR_RE.match(txt):
        return True
    prev_edits = {c["arguments"].get("path") for s in prev["steps"] for c in s["a"]["tool_calls"] if c["name"] in EDIT_TOOLS}
    prev_failed = {c["arguments"].get("command") for s in prev["steps"] for c, r in zip(s["a"]["tool_calls"], s["results"])
                   if c["name"] == "run_command" and r.get("is_error")}
    for s in nxt["steps"][:4]:
        for c in s["a"]["tool_calls"]:
            if c["name"] in EDIT_TOOLS and c["arguments"].get("path") in prev_edits - {None}:
                return True
            if c["name"] == "run_command" and c["arguments"].get("command") in prev_failed - {None}:
                return True
    return False


def _same(a: str, b: str) -> bool:
    return re.sub(r"\s+", " ", a).strip().lower() == re.sub(r"\s+", " ", b).strip().lower()


def windows_for(turn: dict, prompt: str, signal: str, tok: Tok, drops: Counter) -> list[tuple[list[dict], str, int, list[dict]]]:
    steps = []
    for s in turn["steps"]:
        if any(c["name"].startswith("UNSUPPORTED") for c in s["a"]["tool_calls"]):
            drops["skipped_unsupported_step"] += 1
            steps.append(None)  # gap: windows never span it; the next window gets a progress note
            continue
        res = [dict(r, content=tok.truncate(clean_tool_output(r["content"]), TOOL_OUT_TOKENS)) for r in s["results"]]
        steps.append((s, res))
    if not any(steps):
        drops["turn_no_usable_steps"] += 1
        return []
    if tok.n(prompt) > MAX_PROMPT_TOKENS:
        drops["prompt_truncated"] += 1
        prompt = tok.truncate(prompt, MAX_PROMPT_TOKENS - 60)
    used = {c["name"] for x in steps if x for c in x[0]["a"]["tool_calls"]}
    wtools = tools_for(used, prompt[:200])
    out, i, done_cmds = [], 0, []
    while i < len(steps):
        if steps[i] is None:
            done_cmds.append("other_tool(...)")
            i += 1
            continue
        note = ""
        if i > 0:
            recent = "; ".join(done_cmds[-5:])[:600]
            note = f"\n\n[Continuing this task. {i} earlier steps done; most recent: {recent}]"
        msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt + note}]
        j, n_tok = i, None
        while j < len(steps) and steps[j] is not None:
            cand = msgs + to_msgs(*steps[j])
            n = tok.chat(cand, wtools)
            if n > MAX_TOKENS:
                # try ending on this step's assistant message alone (drop its results)
                alone = msgs + to_msgs(steps[j][0], [])
                na = tok.chat(alone, wtools)
                if na <= MAX_TOKENS and j >= i:
                    msgs, n_tok = alone, na
                    j += 1  # its results are shown at the start of the next window via the note
                break
            msgs, n_tok = cand, n
            j += 1
        if j == i or n_tok is None:
            drops["step_too_long"] += 1
            i += 1
            continue
        shown = {m["tool_call_id"] for m in msgs if m["role"] == "tool"}
        errs = sum(1 for _, rs in steps[i:j] for r in rs if r.get("is_error") and r["tool_call_id"] in shown)
        if errs > 1:
            drops["dead_end_multi_error"] += 1
        else:
            sig = signal if signal != "accepted" else ("error_recovery" if errs == 1 else "accepted")
            out.append((msgs, sig, n_tok, wtools))
        for s, _ in steps[i:j]:
            for c in s["a"]["tool_calls"]:
                a = c["arguments"]
                done_cmds.append(f"{c['name']}({(a.get('command') or a.get('path') or a.get('query') or '')[:80]})")
        i = j
    return out


def capability(msgs: list[dict], signal: str) -> str:
    names = [c["function"]["name"] for m in msgs if m["role"] == "assistant" for c in m.get("tool_calls", [])]
    if signal in ("correction_resolved", "error_recovery") or "ask_user" in names or not names:
        return "agentic-decisions"
    if EDIT_TOOLS & set(names):
        return "code-writing"
    return "tool-calling"


def cap_by_project(recs: list[dict], frac: float) -> list[dict]:
    by = defaultdict(list)
    for r in recs:
        by[r["project"]].append(r)
    if len(by) < 1 / frac:  # cap would be infeasible; fall back to equal share
        frac = 1 / len(by)
    lo, hi = 0, len(recs)
    while lo < hi:  # largest N with sum(min(c, frac*N)) >= N
        mid = (lo + hi + 1) // 2
        if sum(min(len(v), int(frac * mid)) for v in by.values()) >= mid:
            lo = mid
        else:
            hi = mid - 1
    cap = max(10, int(frac * lo))  # floor keeps small buckets usable
    out = []
    for p, v in by.items():
        v = sorted(v, key=lambda r: hashlib.sha256(r["id"].encode()).hexdigest())
        out.extend(v[:cap])
    return out


def norm_key(r: dict) -> str:
    u = re.sub(r"\s+", " ", r["messages"][1]["content"].lower())[:400]
    a = json.dumps(r["messages"][2], sort_keys=True)[:400]
    return hashlib.sha256((u + "|" + a).encode()).hexdigest()


def main() -> None:
    src, out, tokdir = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
    rules = scrub.RULES + scrub.load_blocklist(Path(sys.argv[4]))
    rescrub = Counter()
    tok = Tok(tokdir)
    drops, signals = Counter(), Counter()
    sessions = [json.loads(f.read_text()) for f in sorted(src.glob("*.json"))]
    # session-level time split within each source
    eval_ids = set()
    for source in {s["source"] for s in sessions}:
        ss = sorted((s for s in sessions if s["source"] == source), key=lambda s: s.get("started_at") or "")
        eval_ids |= {s["session_id"] for s in ss[int(len(ss) * (1 - EVAL_FRAC)):]}
    recs = []
    for s in sessions:
        turns = turns_of(s)
        for k, t in enumerate(turns):
            nxt = turns[k + 1] if k + 1 < len(turns) else None
            if t["interrupted"]:
                drops["turn_interrupted"] += 1
                continue
            if nxt is not None and not _same(nxt["user"], t["user"]) and is_correction(t, nxt):
                drops["turn_superseded_by_correction"] += 1
                continue
            prev = turns[k - 1] if k > 0 else None
            if prev is not None and _same(prev["user"], t["user"]):
                prompt, signal = t["user"], "accepted"  # repeated message, not a correction
            elif prev is not None and is_correction(prev, t):
                prompt, signal = prev["user"] + "\n\n" + t["user"], "correction_resolved"
            else:
                prompt, signal = t["user"], "accepted"
            for w, (msgs, sig, n, wtools) in enumerate(windows_for(t, prompt, signal, tok, drops)):
                msgs = scrub.walk(msgs, lambda x: scrub.scrub_string(x, [], rules, rescrub, {}))
                signals[sig] += 1
                recs.append({"id": f"{s['session_id']}:t{k:03d}:w{w}", "session_id": s["session_id"],
                             "source": s["source"], "project": project_key(s.get("project")),
                             "timestamp": t.get("ts") or s.get("started_at"),
                             "split": "eval" if s["session_id"] in eval_ids else "train",
                             "capability": capability(msgs, sig), "signal_type": sig,
                             "tools": wtools, "messages": msgs, "tokens_qwen35": n})
    # dedup (train first, then drop eval items duplicating train)
    seen, deduped = set(), []
    for r in sorted(recs, key=lambda r: r["split"] != "train"):
        k = norm_key(r)
        if k in seen:
            drops["duplicate"] += 1
            continue
        seen.add(k)
        deduped.append(r)
    base = out / "03-capabilities"
    manifest = {"max_tokens": MAX_TOKENS, "tool_output_tokens": TOOL_OUT_TOKENS, "project_cap": PROJECT_CAP,
                "eval_frac_by_session": EVAL_FRAC, "system_prompt": SYSTEM, "drops": dict(drops),
                "signals_before_cap": dict(signals), "rescrub_hits": dict(rescrub), "capabilities": {}}
    rng = random.Random(0)
    for cap in ("tool-calling", "code-writing", "agentic-decisions"):
        d = base / cap
        d.mkdir(parents=True, exist_ok=True)
        stats = {}
        kept = {}
        for split in ("train", "eval"):
            rs = [r for r in deduped if r["capability"] == cap and r["split"] == split]
            capped = cap_by_project(rs, PROJECT_CAP) if rs else []
            capped.sort(key=lambda r: r["timestamp"] or "")
            kept[split] = capped
            with (d / f"{split}.jsonl").open("w") as fh:
                for r in capped:
                    fh.write(json.dumps({k: v for k, v in r.items() if k != "split"}, ensure_ascii=False) + "\n")
            toks = sorted(r["tokens_qwen35"] for r in capped)
            stats[split] = {"before_cap": len(rs), "after_cap": len(capped),
                            "projects": dict(Counter(r["project"] for r in capped).most_common()),
                            "sources": dict(Counter(r["source"] for r in capped)),
                            "signals": dict(Counter(r["signal_type"] for r in capped)),
                            "le_1400_tokens": sum(t <= 1400 for t in toks),
                            "tokens_p50_p90_max": [toks[len(toks) // 2], toks[int(len(toks) * 0.9)], toks[-1]] if toks else []}
        tr_s = {r["session_id"] for r in kept["train"]}
        ev_s = {r["session_id"] for r in kept["eval"]}
        tr_k = {norm_key(r) for r in kept["train"]}
        stats["overlap_sessions"] = len(tr_s & ev_s)
        stats["overlap_prompts"] = sum(norm_key(r) in tr_k for r in kept["eval"])
        # router negatives: raw first-user prompts from the other capabilities' train windows
        others = [r for r in deduped if r["capability"] != cap and r["split"] == "train"
                  and r["id"].endswith(":w0")]
        rng.shuffle(others)
        negs, seen_p = [], set()
        for r in others:
            p = r["messages"][1]["content"]
            if p not in seen_p and tok.n(p) <= 500:
                seen_p.add(p)
                negs.append({"prompt": p, "from_capability": r["capability"], "project": r["project"]})
            if len(negs) >= 100:
                break
        with (d / "negative_prompts.jsonl").open("w") as fh:
            for x in negs:
                fh.write(json.dumps(x, ensure_ascii=False) + "\n")
        stats["negative_prompts"] = len(negs)
        (d / "stats.json").write_text(json.dumps(stats, indent=1))
        manifest["capabilities"][cap] = {s: stats[s]["after_cap"] for s in ("train", "eval")}
    # gitleaks over the built files: placeholder substitution and unwrapping can create new
    # key:value shapes (e.g. "<VENDOR_A>-api:<tag>"); replace whatever it flags, like scrub.py does
    post = Counter()
    for _ in range(3):
        found = scrub.run_gitleaks(base)
        if not found:
            break
        lit: dict[str, list[tuple[str, str]]] = {}
        for x in found:
            for v in scrub._secret_variants(x.get("Secret", "")):
                lit.setdefault(x.get("File", ""), []).append((v, f"gitleaks:{x.get('RuleID')}"))
        found.clear()
        for fname, pairs in lit.items():
            fp = Path(fname)
            pairs = sorted(set(pairs), key=lambda x: -len(x[0]))
            if fp.suffix == ".jsonl":
                lines = [json.dumps(scrub.walk(json.loads(ln), lambda t, pr=pairs: scrub.scrub_string(
                    t, pr, [], post, {})), ensure_ascii=False) for ln in fp.read_text().splitlines()]
                fp.write_text("\n".join(lines) + "\n")
            else:
                fp.write_text(json.dumps(scrub.walk(json.loads(fp.read_text()), lambda t, pr=pairs: scrub.scrub_string(
                    t, pr, [], post, {})), indent=1))
        lit.clear()
    manifest["post_build_gitleaks_replaced"] = dict(post)
    (base / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(json.dumps(manifest, indent=1))


if __name__ == "__main__":
    main()
