"""Build a BFCL-v4-DISJOINT tool-calling train/dev set in OpenAI messages+tools format.

Sources (public, no credentials; see SOURCES for licences):
  NousResearch/hermes-function-calling-v1  func-calling-singleturn.json, func-calling.json
  glaiveai/glaive-function-calling-v2       glaive-function-calling-v2.json
  Team-ACE/ToolACE                          data.json
  Salesforce/APIGen-MT-5k                   apigen-mt_5k.json   (CC-BY-NC-4.0: research only)
  (Salesforce/xlam-function-calling-60k is gated -> skipped)

Pipeline: parse + normalise every source -> drop malformed / schema-invalid items ->
decontaminate against ALL BFCL v4 test items (bfcl-eval package data) -> intra-set dedup ->
categorise -> quota sample -> tool-set-grouped train/dev split (dev disjoint in tool sets
and near-duplicate prompts) -> write JSONL + stats.

Decontamination rules (an item is dropped if ANY rule fires; first firing rule is logged):
  func_schema_exact   a tool's sanitized name + normalized parameter schema equals a BFCL function
  func_signature      a tool's sanitized name + property-name set equals a BFCL function (stricter)
  prompt_exact        a user message equals a BFCL prompt (normalized words)
  ngram13_overlap     a user message shares >= 0.8 of word 13-grams with a BFCL prompt
                      (containment vs the smaller of the two; < 13 words = whole sequence)
  minhash_jaccard     exact Jaccard of word 3-gram shingles > 0.7 (MinHash-LSH candidates)
Name-only collisions with BFCL functions are counted but not dropped.

Usage:
  VENV=$SCRATCH/bfcl/venv/bin/python
  $VENV -I experiments/bfcl/data/build_tool_data.py \
      --raw $SCRATCH/bfcl/raw --bfcl-data $VENV_SITE/bfcl_eval/data --out $SCRATCH/bfcl/data
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import (  # noqa: E402
    MinHashLSH,
    NgramIndex,
    est_tokens,
    name_key,
    norm_parameters,
    sanitize_name,
    schema_key,
    signature_key,
    to_openai_tool,
    user_texts,
    words,
)
from schema_check import check_conversation  # noqa: E402

SOURCES = {
    "hermes": {"hf": "NousResearch/hermes-function-calling-v1", "license": "apache-2.0"},
    "glaive": {"hf": "glaiveai/glaive-function-calling-v2", "license": "apache-2.0"},
    "toolace": {"hf": "Team-ACE/ToolACE", "license": "apache-2.0"},
    "apigenmt": {"hf": "Salesforce/APIGen-MT-5k", "license": "cc-by-nc-4.0"},
}
SKIPPED = {"Salesforce/xlam-function-calling-60k": "gated (HTTP 401 without HF credentials)"}

# Target composition of the train+dev pool (before the ~10% dev split).
QUOTAS = {
    "single": 300,
    "multiple": 400,
    "parallel": 220,
    "parallel_multiple": 180,
    "irrelevance": 380,
    "clarify": 170,
    "multi_step": 180,
    "multi_turn": 600,
    "multi_turn_clarify": 270,
}
MAX_EST_TOKENS = 3500
# APIGen-MT conversations carry a ~1.6k-token policy system prompt + ~2.9k tokens of tool
# schemas, so none fit MAX_EST_TOKENS; they go to an optional train_long.jsonl instead.
LONG_MAX_EST_TOKENS = 7000
LONG_SOURCES = {"apigenmt"}
LONG_LIMIT = 300


class Drop(Exception):
    pass


# --------------------------------------------------------------------------------------
# Builder for normalized conversations
# --------------------------------------------------------------------------------------


class Conv:
    def __init__(self, tools: list[dict[str, Any]]) -> None:
        self.tools: list[dict[str, Any]] = []
        self.raw2san: dict[str, str] = {}
        seen: set[str] = set()
        for t in tools:
            if not isinstance(t, dict):
                raise Drop("bad_tool")
            fn = t["function"] if t.get("type") == "function" and isinstance(t.get("function"), dict) else t
            if not fn.get("name"):
                raise Drop("bad_tool")
            ot = to_openai_tool(fn)
            n = ot["function"]["name"]
            if n in seen:
                raise Drop("tool_name_collision")
            seen.add(n)
            self.raw2san[str(fn["name"])] = n
            self.tools.append(ot)
        self.messages: list[dict[str, Any]] = []
        self.pending: list[str] = []
        self.ncall = 0

    def system(self, text: str) -> None:
        self.messages.append({"role": "system", "content": text})

    def user(self, text: str) -> None:
        text = text.strip()
        if not text:
            raise Drop("empty_user")
        if self.pending:
            raise Drop("unanswered_calls")
        self.messages.append({"role": "user", "content": text})

    def assistant(self, text: str = "", calls: list[tuple[str, dict[str, Any]]] | None = None) -> None:
        text = (text or "").strip()
        if self.pending:
            raise Drop("unanswered_calls")
        if not text and not calls:
            raise Drop("empty_assistant")
        msg: dict[str, Any] = {"role": "assistant", "content": text}
        if calls:
            tcs = []
            for name, args in calls:
                san = self.raw2san.get(name) or (sanitize_name(name) if sanitize_name(name) in self.raw2san.values() else None)
                if san is None:
                    raise Drop("unknown_function")
                if not isinstance(args, dict):
                    raise Drop("args_not_object")
                self.ncall += 1
                cid = f"call_{self.ncall}"
                tcs.append({"id": cid, "type": "function",
                            "function": {"name": san, "arguments": json.dumps(args, ensure_ascii=False)}})
                self.pending.append(cid)
            msg["tool_calls"] = tcs
        # merge consecutive text assistant + call assistant (APIGen-MT style)
        if self.messages and self.messages[-1]["role"] == "assistant" and not self.messages[-1].get("tool_calls") and calls and not text:
            self.messages[-1]["tool_calls"] = msg["tool_calls"]
            return
        if self.messages and self.messages[-1]["role"] == "assistant" and not self.messages[-1].get("tool_calls") and not calls:
            raise Drop("double_assistant_text")
        self.messages.append(msg)

    def tool_result(self, content: Any) -> None:
        if not self.pending:
            raise Drop("orphan_tool_result")
        cid = self.pending.pop(0)
        if not isinstance(content, str):
            content = json.dumps(content, ensure_ascii=False)
        self.messages.append({"role": "tool", "tool_call_id": cid, "content": content})

    def finish(self) -> dict[str, Any]:
        while self.messages and self.messages[-1]["role"] != "assistant":
            # dangling tool results without a final answer: cut back to the last assistant
            m = self.messages.pop()
            if m["role"] == "tool":
                self.pending = []
        if not any(m["role"] == "user" for m in self.messages) or not self.messages:
            raise Drop("no_turns")
        self.pending = []
        return {"tools": self.tools, "messages": self.messages}


# --------------------------------------------------------------------------------------
# Source parsers (each yields (id, row) or raises Drop)
# --------------------------------------------------------------------------------------

_TC = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.S)
_TR = re.compile(r"<tool_response>\s*(.*?)\s*</tool_response>", re.S)


def _loose_json(s: str) -> Any:
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        try:
            return ast.literal_eval(s)
        except Exception as e:  # noqa: BLE001
            raise Drop("unparseable_json") from e


def parse_hermes(rec: dict[str, Any]) -> dict[str, Any]:
    conv = rec["conversations"]
    if not conv or conv[0]["from"] != "system":
        raise Drop("no_system")
    sysv = conv[0]["value"]
    a = sysv.rfind("<tools>")  # the first "<tools> </tools>" is prose describing the tags
    b = sysv.find("</tools>", a)
    if a < 0 or b < 0:
        raise Drop("no_tools")
    tools = _loose_json(sysv[a + len("<tools>") : b].strip())
    if isinstance(tools, dict):
        tools = [tools]
    c = Conv(tools)
    for t in conv[1:]:
        f, v = t["from"], t["value"]
        if f == "human":
            c.user(v)
        elif f == "gpt":
            calls = []
            for blob in _TC.findall(v):
                j = _loose_json(blob)
                if not isinstance(j, dict) or "name" not in j:
                    raise Drop("bad_call")
                a = j.get("arguments", {})
                if isinstance(a, str):
                    a = _loose_json(a) if a.strip() else {}
                calls.append((j["name"], a))
            text = _TC.sub("", v).strip()
            if "<tool_call>" in text:
                raise Drop("bad_call")
            c.assistant(text, calls or None)
        elif f == "tool":
            rs = _TR.findall(v)
            if not rs or len(rs) != len(c.pending):
                raise Drop("result_count_mismatch")
            for r in rs:
                try:
                    j = json.loads(r)
                    r = json.dumps(j.get("content", j) if isinstance(j, dict) else j, ensure_ascii=False)
                except json.JSONDecodeError:
                    pass
                c.tool_result(r)
        else:
            raise Drop("bad_role")
    return c.finish()


_GL_SPLIT = re.compile(r"(USER:|ASSISTANT:|FUNCTION RESPONSE:)")
_GL_CALL = re.compile(r"<functioncall>\s*(.*)$", re.S)
_GL_ARGS = re.compile(r'^\{\s*"name"\s*:\s*"([^"]+)"\s*,\s*"arguments"\s*:\s*\'(.*)\'\s*\}\s*$', re.S)


def _json_objects(s: str) -> list[Any]:
    dec, out, i = json.JSONDecoder(), [], 0
    while True:
        j = s.find("{", i)
        if j < 0:
            return out
        try:
            obj, end = dec.raw_decode(s, j)
        except json.JSONDecodeError as e:
            raise Drop("bad_function_spec") from e
        out.append(obj)
        i = end


def parse_glaive(rec: dict[str, Any]) -> dict[str, Any]:
    sysm = rec["system"]
    if "Use them if required" not in sysm:
        raise Drop("no_tools")
    tools = _json_objects(sysm.split("Use them if required", 1)[1])
    if not tools:
        raise Drop("no_tools")
    c = Conv(tools)
    parts = _GL_SPLIT.split(rec["chat"])
    if parts[0].strip():
        raise Drop("odd_format")
    for tag, body in zip(parts[1::2], parts[2::2]):
        body = body.replace("<|endoftext|>", "").strip()
        if tag == "USER:":
            c.user(body)
        elif tag == "ASSISTANT:":
            m = _GL_CALL.search(body)
            if m:
                text = body[: m.start()].strip()
                blob = m.group(1).strip()
                g = _GL_ARGS.match(blob)
                if g:
                    name, a = g.group(1), g.group(2)
                    args = _loose_json(a) if a.strip() else {}
                else:
                    j = _loose_json(blob)
                    if not isinstance(j, dict) or "name" not in j:
                        raise Drop("bad_call")
                    name, args = j["name"], j.get("arguments", {})
                    if isinstance(args, str):
                        args = _loose_json(args) if args.strip() else {}
                c.assistant(text, [(name, args)])
            else:
                c.assistant(body)
        else:
            if not c.pending:
                raise Drop("orphan_tool_result")
            try:
                body = json.dumps(json.loads(body), ensure_ascii=False)
            except json.JSONDecodeError:
                pass
            c.tool_result(body)
    return c.finish()


_TA_TOOLS = "Here is a list of functions in JSON format that you can invoke:"


def _pyval(node: ast.AST) -> Any:
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, (ast.List, ast.Tuple)):
        return [_pyval(e) for e in node.elts]
    if isinstance(node, ast.Dict):
        return {_pyval(k): _pyval(v) for k, v in zip(node.keys, node.values)}
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        v = _pyval(node.operand)
        if isinstance(v, (int, float)):
            return -v if isinstance(node.op, ast.USub) else v
    if isinstance(node, ast.Name) and node.id in ("true", "false", "null", "True", "False", "None"):
        return {"true": True, "True": True, "false": False, "False": False}.get(node.id)
    raise Drop("unparseable_call")


def parse_python_calls(s: str, names: list[str]) -> list[tuple[str, dict[str, Any]]] | None:
    """``[Fn A(x=1), fn_b(y="z")]`` -> calls; None when the text is not a call list."""
    s = s.strip()
    if not (s.startswith("[") and s.endswith("]") and "(" in s):
        return None
    ph = {}
    for i, n in enumerate(sorted(names, key=len, reverse=True)):
        tok = f"__F{i}__"
        if n + "(" in s:
            s = s.replace(n + "(", tok + "(")
            ph[tok] = n
    try:
        tree = ast.parse(s, mode="eval")
    except SyntaxError as e:
        raise Drop("unparseable_call") from e
    if not isinstance(tree.body, ast.List):
        raise Drop("unparseable_call")
    out = []
    for call in tree.body.elts:
        if not isinstance(call, ast.Call) or not isinstance(call.func, ast.Name) or call.args:
            raise Drop("unparseable_call")
        name = ph.get(call.func.id, call.func.id)
        out.append((name, {kw.arg: _pyval(kw.value) for kw in call.keywords if kw.arg}))
    return out


def looks_like_odd_call(text: str, names: list[str]) -> bool:
    t = text.strip()
    for n in names:
        if re.search(re.escape(n) + r"\s*[\(\|:\]]", t):
            return True
    return t[:1] in "[{(<" and any(n in t[:200] for n in names)


def parse_toolace(rec: dict[str, Any]) -> dict[str, Any]:
    sysm = rec["system"]
    if _TA_TOOLS not in sysm:
        raise Drop("no_tools")
    tail = sysm.split(_TA_TOOLS, 1)[1]
    j = tail.find("[")
    try:
        tools, _ = json.JSONDecoder().raw_decode(tail, j)
    except json.JSONDecodeError as e:
        raise Drop("bad_function_spec") from e
    c = Conv(tools)
    names = [str(t.get("name")) for t in tools]
    for t in rec["conversations"]:
        f, v = t["from"], t["value"]
        if f == "user":
            if v.lstrip().startswith("Role definition"):
                raise Drop("odd_format")
            c.user(v)
        elif f == "assistant":
            calls = parse_python_calls(v, names)
            if calls is None:
                if looks_like_odd_call(v, names):
                    raise Drop("unparseable_call")
                c.assistant(v)
            else:
                c.assistant("", calls)
        elif f == "tool":
            res = _loose_json(v)
            if not isinstance(res, list) or len(res) != len(c.pending):
                raise Drop("result_count_mismatch")
            for r in res:
                c.tool_result(r.get("results", r) if isinstance(r, dict) else r)
        else:
            raise Drop("bad_role")
    return c.finish()


def parse_apigenmt(rec: dict[str, Any]) -> dict[str, Any]:
    tools = rec["tools"]
    if isinstance(tools, str):
        tools = json.loads(tools)
    tools = [t for t in tools if t.get("name") != "think"]  # scratchpad tool, not a real API
    c = Conv(tools)
    if rec.get("system"):
        c.system(rec["system"])
    turns = rec["conversations"]
    skip_obs = 0
    for t in turns:
        f, v = t["from"], t["value"]
        if f == "human":
            c.user(v)
        elif f == "gpt":
            if c.messages and c.messages[-1]["role"] == "assistant" and not c.messages[-1].get("tool_calls"):
                c.messages[-1]["content"] += "\n\n" + v.strip()
            else:
                c.assistant(v)
        elif f == "function_call":
            j = _loose_json(v)
            if j.get("name") == "think":
                skip_obs += 1
                continue
            args = j.get("arguments", {})
            if isinstance(args, str):
                args = _loose_json(args) if args.strip() else {}
            if c.messages and c.messages[-1]["role"] == "assistant" and c.messages[-1].get("tool_calls") and c.pending and not any(m["role"] == "tool" for m in c.messages[-1:]):
                # consecutive function_call turns -> parallel calls in one assistant message
                saved = c.pending
                c.pending = []
                c.assistant("", [(j["name"], args)])
                new = c.messages.pop()
                c.messages[-1]["tool_calls"].extend(new["tool_calls"])
                c.pending = saved + c.pending
            else:
                c.assistant("", [(j["name"], args)])
        elif f == "observation":
            if skip_obs:
                skip_obs -= 1
                continue
            c.tool_result(v)
        else:
            raise Drop("bad_role")
    return c.finish()


def truncate_to_fit(row: dict[str, Any], cap: int) -> dict[str, Any] | None:
    """Longest prefix ending on a text-only assistant turn (after >= 1 tool call) under cap."""
    if est_tokens(row) <= cap:
        return row
    msgs = row["messages"]
    for k in range(len(msgs), 0, -1):
        m = msgs[k - 1]
        if m["role"] != "assistant" or m.get("tool_calls"):
            continue
        pre = msgs[:k]
        if not any(x.get("tool_calls") for x in pre):
            break
        cand = {**row, "messages": pre, "truncated": True}
        if est_tokens(cand) <= cap:
            return cand
    return None


# --------------------------------------------------------------------------------------
# Categorisation
# --------------------------------------------------------------------------------------

_REFUSE = re.compile(
    r"\b(sorry|unable|can't|cannot|can not|don't have|do not have|not able|no (available )?function|"
    r"none of the (provided |available )?functions|outside|beyond|limited to|not (currently )?possible|"
    r"capabilit|not equipped|doesn't (allow|support)|does not (allow|support))", re.I)
_ASK = re.compile(
    r"\b(provide|specify|could you (please )?(tell|give|share|let me know)|can you (please )?(tell|give|share)|"
    r"what (is|are|would|should)|which|need (the|some|more|a few|to know)|missing|please (tell|share|let me know))", re.I)


def categorize(row: dict[str, Any]) -> str:
    msgs = [m for m in row["messages"] if m["role"] != "system"]
    users = [i for i, m in enumerate(msgs) if m["role"] == "user"]
    call_idx = [i for i, m in enumerate(msgs) if m.get("tool_calls")]
    if not call_idx:
        first = next(m for m in msgs if m["role"] == "assistant")
        txt = first.get("content", "")
        if "?" in txt and _ASK.search(txt) and not _REFUSE.search(txt):
            return "clarify"
        return "irrelevance"
    if len(users) >= 2:
        # clarify -> call: an assistant text turn asking for info, then user, then a call
        first_call = call_idx[0]
        before = [m for m in msgs[:first_call] if m["role"] == "assistant"]
        if before and "?" in before[-1].get("content", "") and _ASK.search(before[-1]["content"]):
            return "multi_turn_clarify"
        return "multi_turn"
    if len(call_idx) >= 2:
        return "multi_step"
    calls = msgs[call_idx[0]]["tool_calls"]
    if len(calls) == 1:
        return "single" if len(row["tools"]) == 1 else "multiple"
    return "parallel" if len({c["function"]["name"] for c in calls}) == 1 else "parallel_multiple"


# --------------------------------------------------------------------------------------
# BFCL index
# --------------------------------------------------------------------------------------


def load_bfcl(data_dir: Path) -> tuple[list[str], list[dict[str, Any]], dict[str, int]]:
    prompts: list[str] = []
    funcs: list[dict[str, Any]] = []
    stats: Counter = Counter()

    def take_question(q: Any) -> None:
        if isinstance(q, str):
            prompts.append(q)
            return
        for turn in q or []:
            if isinstance(turn, dict):
                turn = [turn]
            for m in turn or []:
                if isinstance(m, dict) and m.get("role") == "user" and m.get("content"):
                    prompts.append(str(m["content"]))

    files = sorted(data_dir.glob("BFCL_v4_*.json")) + sorted((data_dir / "memory_prereq_conversation").glob("*.json"))
    files += sorted((data_dir / "unused_datasets" / "question").glob("*.json"))
    for f in files:
        if f.name == "BFCL_v4_format_sensitivity.json":
            continue  # an id index over other categories, no new items
        for line in f.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            it = json.loads(line)
            stats[f.name] += 1
            take_question(it.get("question"))
            for fn in it.get("function") or []:
                if isinstance(fn, dict):
                    funcs.append(fn)
            for fn in it.get("missed_function", {}).values() if isinstance(it.get("missed_function"), dict) else []:
                funcs.extend(x for x in fn if isinstance(x, dict))
    for f in sorted((data_dir / "multi_turn_func_doc").glob("*.json")):
        for line in f.read_text(encoding="utf-8").splitlines():
            if line.strip():
                funcs.append(json.loads(line))
                stats["func_doc:" + f.name] += 1
    return prompts, funcs, dict(stats)


# --------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------


def _norm_prompt(t: str) -> str:
    return " ".join(words(t))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--raw", type=Path, required=True, help="dir with hermes/ glaive/ toolace/ apigenmt/")
    ap.add_argument("--bfcl-data", type=Path, required=True, help="bfcl_eval/data dir from the package")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=20261121)
    ap.add_argument("--dev-frac", type=float, default=0.1)
    ap.add_argument("--glaive-limit", type=int, default=40000, help="first N glaive records screened")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)

    # ---- BFCL index
    bprompts, bfuncs, bstats = load_bfcl(args.bfcl_data)
    b_schema = {schema_key(f) for f in bfuncs}
    b_sig = {signature_key(f) for f in bfuncs}
    b_name = {name_key(f) for f in bfuncs}
    b_exact = {_norm_prompt(p) for p in bprompts}
    ng = NgramIndex(13)
    mh = MinHashLSH()
    for p in bprompts:
        ng.add(p)
        mh.add(p)
    print(f"BFCL: {len(bprompts)} prompts, {len(bfuncs)} functions ({len(b_schema)} unique schemas)", flush=True)

    # ---- parse sources
    parsed: list[dict[str, Any]] = []
    parse_drops: dict[str, Counter] = defaultdict(Counter)
    seen_raw: Counter = Counter()

    def ingest(src: str, recs: list[dict[str, Any]], fn: Any, tag: str) -> None:
        cap = LONG_MAX_EST_TOKENS if src in LONG_SOURCES else MAX_EST_TOKENS
        for i, rec in enumerate(recs):
            seen_raw[src] += 1
            try:
                row = fn(rec)
            except Drop as d:
                parse_drops[src][str(d)] += 1
                continue
            except (KeyError, TypeError, AttributeError, ValueError):
                parse_drops[src]["malformed"] += 1
                continue
            errs = check_conversation(row)
            if errs:
                parse_drops[src]["schema_invalid" if any("fail schema" in e or "invalid schema" in e for e in errs) else "structure_invalid"] += 1
                continue
            row2 = truncate_to_fit(row, cap)
            if row2 is None:
                parse_drops[src]["too_long"] += 1
                continue
            row2["id"] = f"{tag}-{i}"
            row2["source"] = SOURCES[src]["hf"]
            row2["license"] = SOURCES[src]["license"]
            row2["long"] = src in LONG_SOURCES
            parsed.append(row2)

    raw = args.raw
    ingest("hermes", json.loads((raw / "hermes/func-calling-singleturn.json").read_text()), parse_hermes, "hermes-st")
    ingest("hermes", json.loads((raw / "hermes/func-calling.json").read_text()), parse_hermes, "hermes-mt")
    ingest("toolace", json.loads((raw / "toolace/data.json").read_text()), parse_toolace, "toolace")
    ingest("apigenmt", json.loads((raw / "apigenmt/apigen-mt_5k.json").read_text()), parse_apigenmt, "apigenmt")
    gl = json.loads((raw / "glaive/glaive-function-calling-v2.json").read_text())
    gl_idx = list(range(len(gl)))
    random.Random(args.seed).shuffle(gl_idx)
    ingest("glaive", [gl[i] for i in gl_idx[: args.glaive_limit]], parse_glaive, "glaive")
    del gl
    print(f"parsed {len(parsed)} valid conversations", flush=True)

    # ---- decontaminate vs BFCL + intra dedup
    removed: Counter = Counter()
    removed_by_src: dict[str, Counter] = defaultdict(Counter)
    name_only = 0
    removal_log = (args.out / "decontam_removed.jsonl").open("w", encoding="utf-8")
    kept: list[dict[str, Any]] = []
    seen_prompt: set[str] = set()
    for row in parsed:
        rule, detail = None, None
        for t in row["tools"]:
            if schema_key(t) in b_schema:
                rule, detail = "func_schema_exact", t["function"]["name"]
                break
            if signature_key(t) in b_sig:
                rule, detail = "func_signature", t["function"]["name"]
                break
        if rule is None:
            if any(name_key(t) in b_name for t in row["tools"]):
                name_only += 1
            for u in user_texts(row["messages"]):
                if _norm_prompt(u) in b_exact:
                    rule, detail = "prompt_exact", u[:200]
                    break
                ov, bi = ng.max_overlap(u)
                if ov >= 0.8:
                    rule, detail = "ngram13_overlap", {"train": u[:200], "bfcl": bprompts[bi][:200], "overlap": round(ov, 3)}
                    break
                jac, bi = mh.max_jaccard(u)
                if jac > 0.7:
                    rule, detail = "minhash_jaccard", {"train": u[:200], "bfcl": bprompts[bi][:200], "jaccard": round(jac, 3)}
                    break
        if rule is not None:
            removed[rule] += 1
            removed_by_src[row["source"]][rule] += 1
            removal_log.write(json.dumps({"id": row["id"], "rule": rule, "detail": detail}, ensure_ascii=False) + "\n")
            continue
        key = _norm_prompt(" || ".join(user_texts(row["messages"])))
        if key in seen_prompt:
            removed["intra_duplicate"] += 1
            removed_by_src[row["source"]]["intra_duplicate"] += 1
            continue
        seen_prompt.add(key)
        row["category"] = categorize(row)
        kept.append(row)
    removal_log.close()
    print(f"after decontam: {len(kept)}; removed {dict(removed)}", flush=True)

    # ---- quota sampling with source balancing
    long_rows = [r for r in kept if r.get("long")]
    kept = [r for r in kept if not r.get("long")]
    by_cat: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in kept:
        by_cat[r["category"]].append(r)
    # Single-turn "clarify" (ask for a missing required parameter) is rare as a standalone
    # item; derive it from multi_turn_clarify conversations cut right after the assistant's
    # question. Each source conversation is used for one category only.
    mtc = by_cat.get("multi_turn_clarify", [])
    rng.shuffle(mtc)
    need = max(0, QUOTAS["clarify"] - len(by_cat.get("clarify", [])))
    derived = 0
    while need > derived and len(mtc) > QUOTAS["multi_turn_clarify"]:
        r = mtc.pop()
        msgs = r["messages"]
        first_call = next(i for i, m in enumerate(msgs) if m.get("tool_calls"))
        k = max(i for i in range(first_call) if msgs[i]["role"] == "assistant")
        by_cat["clarify"].append({**r, "id": r["id"] + "-clarify", "messages": msgs[: k + 1], "category": "clarify",
                                  "derived_from": "multi_turn_clarify"})
        derived += 1
    by_cat["multi_turn_clarify"] = mtc
    avail = {c: Counter(r["source"] for r in rs) for c, rs in by_cat.items()}
    selected: list[dict[str, Any]] = []
    for cat, quota in QUOTAS.items():
        pool = by_cat.get(cat, [])
        by_src: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for r in pool:
            by_src[r["source"]].append(r)
        for lst in by_src.values():
            rng.shuffle(lst)
        # round-robin across sources -> as balanced as availability allows
        take: list[dict[str, Any]] = []
        srcs = sorted(by_src)
        while len(take) < quota and any(by_src[s] for s in srcs):
            for s in srcs:
                if by_src[s] and len(take) < quota:
                    take.append(by_src[s].pop())
        selected.extend(take)

    # ---- dev split grouped by tool set, then strip dev items near-duplicating train
    def group(r: dict[str, Any]) -> str:
        return ",".join(sorted(t["function"]["name"].lower() for t in r["tools"]))

    def is_dev(r: dict[str, Any]) -> bool:
        if r["source"] == SOURCES["apigenmt"]["hf"]:
            # 2 shared tool sets (airline/retail): split by conversation instead
            g = r["id"]
        else:
            g = group(r)
        h = int.from_bytes(hashlib.sha256(f"{args.seed}:{g}".encode()).digest()[:4], "big")
        return (h % 1000) < args.dev_frac * 1000

    train = [r for r in selected if not is_dev(r)]
    dev = [r for r in selected if is_dev(r)]
    tr_mh = MinHashLSH()
    tr_ng = NgramIndex(13)
    for r in train:
        for u in user_texts(r["messages"]):
            tr_mh.add(u)
            tr_ng.add(u)
    dev_clean, dev_drop = [], 0
    train_groups = {group(r) for r in train if r["source"] != SOURCES["apigenmt"]["hf"]}
    for r in dev:
        bad = any(tr_mh.max_jaccard(u)[0] > 0.7 or tr_ng.max_overlap(u)[0] >= 0.8 for u in user_texts(r["messages"]))
        if bad:
            dev_drop += 1
            continue
        dev_clean.append(r)
    dev = dev_clean
    assert not ({group(r) for r in dev if r["source"] != SOURCES["apigenmt"]["hf"]} & train_groups)
    rng.shuffle(train)
    rng.shuffle(dev)

    def write(path: Path, rows: list[dict[str, Any]]) -> None:
        with path.open("w", encoding="utf-8") as fh:
            for r in rows:
                out = {"id": r["id"], "category": r["category"], "source": r["source"], "license": r["license"],
                       "est_tokens": est_tokens(r), "truncated": bool(r.get("truncated")),
                       "tools": r["tools"], "messages": r["messages"]}
                fh.write(json.dumps(out, ensure_ascii=False) + "\n")

    write(args.out / "train.jsonl", train)
    write(args.out / "dev.jsonl", dev)
    rng.shuffle(long_rows)
    long_rows = long_rows[:LONG_LIMIT]
    if long_rows:
        write(args.out / "train_long.jsonl", long_rows)
    else:
        (args.out / "train_long.jsonl").unlink(missing_ok=True)

    def counts(rows: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "total": len(rows),
            "by_category": dict(sorted(Counter(r["category"] for r in rows).items())),
            "by_source": dict(sorted(Counter(r["source"] for r in rows).items())),
            "by_category_source": {c: dict(Counter(r["source"] for r in rows if r["category"] == c)) for c in QUOTAS},
            "assistant_turns": sum(1 for r in rows for m in r["messages"] if m["role"] == "assistant"),
            "tool_calls": sum(len(m.get("tool_calls") or []) for r in rows for m in r["messages"]),
            "truncated": sum(1 for r in rows if r.get("truncated")),
            "est_tokens_max": max((est_tokens(r) for r in rows), default=0),
        }

    stats = {
        "seed": args.seed,
        "sources": SOURCES,
        "skipped_sources": SKIPPED,
        "bfcl_index": {"prompts": len(bprompts), "functions": len(bfuncs), "unique_schemas": len(b_schema),
                       "files": bstats},
        "raw_records_screened": dict(seen_raw),
        "parse_drops": {k: dict(v) for k, v in parse_drops.items()},
        "parsed_valid": len(parsed),
        "decontam_removed": dict(removed),
        "decontam_removed_by_source": {k: dict(v) for k, v in removed_by_src.items()},
        "bfcl_name_only_collisions_kept": name_only,
        "available_after_decontam": {c: dict(v) for c, v in avail.items()},
        "quotas": QUOTAS,
        "dev_dropped_near_dup_of_train": dev_drop,
        "clarify_derived_from_multi_turn_clarify": derived,
        "train": counts(train),
        "dev": counts(dev),
        "train_long": counts(long_rows),
    }
    (args.out / "build_stats.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps({"train": stats["train"]["by_category"], "dev": stats["dev"]["by_category"]}, indent=1))


if __name__ == "__main__":
    main()
