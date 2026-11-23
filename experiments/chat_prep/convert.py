"""Convert raw Claude Code, Codex and DeepSeek Harness transcripts into one canonical session format.

Canonical session (one JSON file per session):
  {"session_id", "source", "project", "started_at",
   "events": [{"role": "user", "content", "ts"},
              {"role": "assistant", "content", "tool_calls": [{"id", "name", "arguments"}]},
              {"role": "tool", "tool_call_id", "name", "content", "is_error"},
              {"role": "break", "reason"}]}

Tools are mapped onto a small canonical set (see CANONICAL_TOOLS). Calls to anything else
(subagents, browsers, workflows, harness MCP tools) become name "UNSUPPORTED:<orig>"; the
builder cuts windows there. Thinking/reasoning blocks are dropped. Sidechain (subagent)
records are skipped.

Usage: python convert.py <raw_root> <out_dir>
  raw_root contains claude-borg/, claude-local/ (optional), codex-borg/, deepseek-local/ (optional,
  a copy of ~/.dsh/sessions)
"""

from __future__ import annotations

import contextlib
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

CANONICAL_TOOLS = (
    "run_command",
    "read_file",
    "edit_file",
    "write_file",
    "apply_patch",
    "search_files",
    "web_search",
    "web_fetch",
    "ask_user",
)

USER_SKIP_PREFIXES = (
    "<command-",
    "<local-command",
    "Caveat:",
    "<system-reminder",
    "<environment_context",
    "<user_instructions",
    "# AGENTS.md",
    "<permissions",
    "<turn_aborted",
    "<task-notification",
    "[Request interrupted",
)


def _text_of(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for b in content:
            if isinstance(b, dict):
                if b.get("type") in ("text", "input_text", "output_text"):
                    parts.append(b.get("text", ""))
                elif b.get("type") == "image":
                    parts.append("[image]")
            elif isinstance(b, str):
                parts.append(b)
        return "\n".join(p for p in parts if p)
    return "" if content is None else str(content)


# ---------------------------------------------------------------- Claude Code


def _map_claude_tool(name: str, inp: dict) -> tuple[str, dict]:
    if name == "Bash":
        return "run_command", {"command": inp.get("command", "")}
    if name == "Read":
        a = {"path": inp.get("file_path", "")}
        for k in ("offset", "limit"):
            if k in inp:
                a[k] = inp[k]
        return "read_file", a
    if name == "Edit":
        return "edit_file", {
            "path": inp.get("file_path", ""),
            "old_text": inp.get("old_string", ""),
            "new_text": inp.get("new_string", ""),
        }
    if name == "Write":
        return "write_file", {"path": inp.get("file_path", ""), "content": inp.get("content", "")}
    if name in ("Grep", "Glob"):
        a = {"pattern": inp.get("pattern", "")}
        if inp.get("path"):
            a["path"] = inp["path"]
        return "search_files", a
    if name == "WebSearch":
        return "web_search", {"query": inp.get("query", "")}
    if name == "WebFetch":
        return "web_fetch", {"url": inp.get("url", "")}
    if name == "AskUserQuestion":
        qs = inp.get("questions") or []
        q = "\n".join(x.get("question", "") for x in qs if isinstance(x, dict)) or str(inp)
        return "ask_user", {"question": q}
    return f"UNSUPPORTED:{name}", {}


def convert_claude(path: Path) -> dict | None:
    events: list[dict] = []
    sid = path.stem
    project = None
    started = None
    cur_msg_id = None
    for line in path.open(errors="ignore"):
        try:
            r = json.loads(line)
        except Exception:
            continue
        if r.get("isSidechain") or r.get("isMeta"):
            continue
        t = r.get("type")
        if t not in ("user", "assistant"):
            if t == "summary" or r.get("subtype") == "compact_boundary":
                events.append({"role": "break", "reason": "compaction"})
            continue
        project = project or r.get("cwd")
        started = started or r.get("timestamp")
        m = r.get("message") or {}
        content = m.get("content")
        if t == "user":
            cur_msg_id = None
            if isinstance(content, list) and any(
                isinstance(b, dict) and b.get("type") == "tool_result" for b in content
            ):
                for b in content:
                    if isinstance(b, dict) and b.get("type") == "tool_result":
                        events.append(
                            {
                                "role": "tool",
                                "tool_call_id": b.get("tool_use_id"),
                                "content": _text_of(b.get("content")),
                                "is_error": bool(b.get("is_error")),
                            }
                        )
                continue
            txt = _text_of(content).strip()
            if txt.startswith("[Request interrupted"):
                events.append({"role": "break", "reason": "user_interrupt"})
                continue
            if not txt or txt.startswith(USER_SKIP_PREFIXES):
                continue
            events.append({"role": "user", "content": txt, "ts": r.get("timestamp")})
            continue
        # assistant: one content block per line, merge by message id
        mid = m.get("id")
        if mid != cur_msg_id or not events or events[-1]["role"] != "assistant":
            events.append({"role": "assistant", "content": "", "tool_calls": []})
            cur_msg_id = mid
        a = events[-1]
        for b in content if isinstance(content, list) else []:
            if not isinstance(b, dict):
                continue
            if b.get("type") == "text":
                a["content"] = (a["content"] + "\n" + b.get("text", "")).strip()
            elif b.get("type") == "tool_use":
                name, args = _map_claude_tool(b.get("name", ""), b.get("input") or {})
                a["tool_calls"].append({"id": b.get("id"), "name": name, "arguments": args})
    # attach tool names to results
    names = {
        c["id"]: c["name"] for e in events if e["role"] == "assistant" for c in e["tool_calls"]
    }
    for e in events:
        if e["role"] == "tool":
            e["name"] = names.get(e["tool_call_id"], "UNSUPPORTED:?")
    if not any(e["role"] == "user" for e in events):
        return None
    return {
        "session_id": f"cc-{sid}",
        "source": "claude_code",
        "project": project,
        "started_at": started,
        "events": events,
    }


# ---------------------------------------------------------------- Codex


class _JSLiteral:
    """Tiny parser for the JS object literals Codex passes to tools.<name>({...})."""

    def __init__(self, s: str):
        self.s, self.i = s, 0

    def ws(self):
        while self.i < len(self.s) and self.s[self.i] in " \t\r\n":
            self.i += 1

    def value(self):
        self.ws()
        c = self.s[self.i]
        if c == "{":
            return self.obj()
        if c == "[":
            return self.arr()
        if c in "\"'`":
            return self.string()
        m = re.compile(r"-?\d+(\.\d+)?([eE][-+]?\d+)?").match(self.s, self.i)
        if m:
            self.i = m.end()
            return json.loads(m.group())
        for lit, v in (("true", True), ("false", False), ("null", None), ("undefined", None)):
            if self.s.startswith(lit, self.i):
                self.i += len(lit)
                return v
        raise ValueError("unsupported js value")

    def string(self):
        q = self.s[self.i]
        j = self.i + 1
        out = []
        while True:
            c = self.s[j]
            if c == "\\":
                out.append(self.s[j : j + 2])
                j += 2
                continue
            if q == "`" and c == "$" and self.s[j + 1 : j + 2] == "{":
                raise ValueError("template interpolation")
            if c == q:
                break
            out.append('\\"' if c == '"' else ("\\n" if c == "\n" else c))
            j += 1
        self.i = j + 1
        raw = "".join(out)
        if q != '"':
            raw = raw.replace("\\'", "'").replace("\\`", "`")
        return json.loads('"' + raw + '"')

    def obj(self):
        self.i += 1
        out = {}
        while True:
            self.ws()
            if self.s[self.i] == "}":
                self.i += 1
                return out
            if self.s[self.i] in "\"'":
                k = self.string()
            else:
                m = re.compile(r"[A-Za-z_$][\w$]*").match(self.s, self.i)
                if not m:
                    raise ValueError("bad key")
                k = m.group()
                self.i = m.end()
            self.ws()
            if self.s[self.i] != ":":
                raise ValueError("expected :")
            self.i += 1
            out[k] = self.value()
            self.ws()
            if self.s[self.i] == ",":
                self.i += 1

    def arr(self):
        self.i += 1
        out = []
        while True:
            self.ws()
            if self.s[self.i] == "]":
                self.i += 1
                return out
            out.append(self.value())
            self.ws()
            if self.s[self.i] == ",":
                self.i += 1


_CALL_RE = re.compile(r"tools\.(\w+)\(")


_ASSIGN_RE = re.compile(r"\b(?:const|let|var)\s+([A-Za-z_$][\w$]*)\s*=\s*")
_IDENT_RE = re.compile(r"([A-Za-z_$][\w$]*)\s*\)")


def _parse_exec(src: str) -> list[tuple[str, Any]] | None:
    env: dict[str, Any] = {}
    for m in _ASSIGN_RE.finditer(src):
        p = _JSLiteral(src)
        p.i = m.end()
        with contextlib.suppress(Exception):
            env[m.group(1)] = p.value()
    calls = []
    for m in _CALL_RE.finditer(src):
        p = _JSLiteral(src)
        p.i = m.end()
        try:
            p.ws()
            if p.s[p.i] == ")":
                arg = {}
            else:
                im = _IDENT_RE.match(src, p.i)
                arg = env[im.group(1)] if im and im.group(1) in env else p.value()
        except Exception:
            return None
        calls.append((m.group(1), arg))
    return calls


def _map_codex_tool(name: str, arg: Any) -> tuple[str, dict]:
    if name == "exec_command" and isinstance(arg, dict):
        return "run_command", {"command": arg.get("cmd", "")}
    if name == "apply_patch":
        patch = (
            arg
            if isinstance(arg, str)
            else (arg.get("input") or arg.get("patch") or "")
            if isinstance(arg, dict)
            else ""
        )
        return "apply_patch", {"patch": patch}
    if name == "web__run" and isinstance(arg, dict):
        q = arg.get("search_query") or arg.get("query")
        if isinstance(q, list) and q:
            q = q[0].get("q") if isinstance(q[0], dict) else q[0]
        if q:
            return "web_search", {"query": str(q)}
        if arg.get("open"):
            o = arg["open"][0] if isinstance(arg["open"], list) else arg["open"]
            return "web_fetch", {"url": o.get("ref_id", "") if isinstance(o, dict) else str(o)}
    return f"UNSUPPORTED:{name}", {}


_EXIT_RE = re.compile(r"Process exited with code (-?\d+)")


def _codex_output(raw: Any) -> tuple[str, bool]:
    if isinstance(raw, list):
        txt = _text_of(raw)
    else:
        txt = str(raw or "")
        try:
            j = json.loads(txt)
            if isinstance(j, dict) and "output" in j:
                txt = str(j["output"])
        except Exception:
            pass
    txt = re.sub(r"^Script completed\nWall time [^\n]*\nOutput:\n", "", txt)
    txt = re.sub(r"^(Chunk ID: \S+\n)?Wall time: [^\n]*\n", "", txt, flags=re.M)
    m = _EXIT_RE.search(txt)
    is_err = bool(m and m.group(1) != "0") or txt.startswith(("Script failed", "Error:"))
    txt = _EXIT_RE.sub(lambda mm: f"[exit {mm.group(1)}]", txt)
    txt = re.sub(r"^Original token count: \d+\n", "", txt, flags=re.M)
    return txt.strip(), is_err


def convert_codex(path: Path) -> dict | None:
    if path.suffix == ".zst":
        stream = subprocess.Popen(
            ["zstdcat", str(path)], stdout=subprocess.PIPE, text=True, errors="ignore"
        ).stdout
    else:
        stream = path.open(errors="ignore")
    events: list[dict] = []
    sid = project = started = None
    pending: dict[str, dict] = {}
    for line in stream:
        try:
            r = json.loads(line)
        except Exception:
            continue
        p = r.get("payload") or {}
        typ = r.get("type")
        if typ == "session_meta":
            sid = sid or p.get("id")
            project = project or p.get("cwd")
            started = started or p.get("timestamp") or r.get("timestamp")
            continue
        if typ == "compacted":
            events.append({"role": "break", "reason": "compaction"})
            continue
        if typ != "response_item":
            continue
        pt = p.get("type")
        if pt == "message":
            role = p.get("role")
            txt = _text_of(p.get("content")).strip()
            if role == "user" and txt.lstrip().startswith("<turn_aborted"):
                events.append({"role": "break", "reason": "user_interrupt"})
            elif role == "user" and txt and not txt.lstrip().startswith(USER_SKIP_PREFIXES):
                events.append({"role": "user", "content": txt, "ts": r.get("timestamp")})
            elif role == "assistant" and txt:
                events.append({"role": "assistant", "content": txt, "tool_calls": []})
        elif pt in ("custom_tool_call", "function_call"):
            cid = p.get("call_id")
            name = p.get("name")
            if name == "exec":
                parsed = _parse_exec(p.get("input") or "")
                mapped = (
                    [_map_codex_tool(*c) for c in parsed]
                    if parsed
                    else [("UNSUPPORTED:exec-parse", {})]
                )
            elif name == "request_user_input_async":
                mapped = [("ask_user", {"question": str(p.get("arguments", ""))[:2000]})]
            else:
                mapped = [(f"UNSUPPORTED:{name}", {})]
            if not events or events[-1]["role"] != "assistant":
                events.append({"role": "assistant", "content": "", "tool_calls": []})
            calls = [
                {"id": f"{cid}#{k}" if len(mapped) > 1 else cid, "name": n, "arguments": a}
                for k, (n, a) in enumerate(mapped)
            ]
            events[-1]["tool_calls"].extend(calls)
            pending[cid] = calls
        elif pt in ("custom_tool_call_output", "function_call_output"):
            cid = p.get("call_id")
            calls = pending.pop(cid, None) or [{"id": cid, "name": "UNSUPPORTED:?"}]
            raw_out = p.get("output")
            if len(calls) > 1:
                items = (
                    [b.get("text", "") for b in raw_out if isinstance(b, dict)]
                    if isinstance(raw_out, list)
                    else []
                )
                if items and items[0].startswith("Script completed"):
                    items = items[1:]
                if len(items) != len(calls):
                    for c in calls:  # outputs can't be attributed per call
                        c["name"] = "UNSUPPORTED:exec-unsplittable"
                    items = ["\n".join(items)] + [""] * (len(calls) - 1)
                for c, it in zip(calls, items, strict=False):
                    txt, err = _codex_output(it)
                    events.append(
                        {
                            "role": "tool",
                            "tool_call_id": c["id"],
                            "name": c["name"],
                            "content": txt,
                            "is_error": err,
                        }
                    )
            else:
                txt, err = _codex_output(raw_out)
                events.append(
                    {
                        "role": "tool",
                        "tool_call_id": calls[0]["id"],
                        "name": calls[0]["name"],
                        "content": txt,
                        "is_error": err,
                    }
                )
    if not any(e["role"] == "user" for e in events):
        return None
    return {
        "session_id": f"cx-{sid or path.stem}",
        "source": "codex",
        "project": project,
        "started_at": started,
        "events": events,
    }


# ---------------------------------------------------------------- DeepSeek Harness (dsh)

_DSH_EXIT_RE = re.compile(r"\[exit code: (-?\d+)\]")
_DSH_READ_RE = re.compile(
    r"\A<path>[^\n]*</path>\n<type>[^\n]*</type>\n<content>\n?(.*?)(?:\n?</content>)?\s*\Z", re.S
)


def _ms_iso(ms: Any) -> str | None:
    import datetime as _dt

    try:
        return _dt.datetime.fromtimestamp(int(ms) / 1000, _dt.UTC).strftime("%Y-%m-%dT%H:%M:%S.%fZ")
    except Exception:
        return None


def _map_dsh_tool(name: str, inp: dict) -> tuple[str, dict]:
    if name == "bash":
        if inp.get("run_in_background"):
            return "UNSUPPORTED:bash-background", {}
        return "run_command", {"command": inp.get("command", "")}
    if name == "read":
        a = {"path": inp.get("file_path", "")}
        for k in ("offset", "limit"):
            if k in inp:
                a[k] = inp[k]
        return "read_file", a
    if name == "edit":
        return "edit_file", {
            "path": inp.get("file_path", ""),
            "old_text": inp.get("old_string", ""),
            "new_text": inp.get("new_string", ""),
        }
    if name == "write":
        return "write_file", {"path": inp.get("file_path", ""), "content": inp.get("content", "")}
    if name in ("glob", "grep"):
        a = {"pattern": inp.get("pattern", "")}
        if inp.get("path"):
            a["path"] = inp["path"]
        return "search_files", a
    if name == "web_search":
        qs = inp.get("queries") or ([inp["query"]] if inp.get("query") else [])
        return (
            ("web_search", {"query": str(qs[0])})
            if len(qs) == 1
            else ("UNSUPPORTED:web_search-multi", {})
        )
    if name == "web_fetch":
        return "web_fetch", {"url": inp.get("url", "")}
    if name == "ask_user_question":
        qs = inp.get("questions") or []
        q = "\n".join(x.get("question", "") for x in qs if isinstance(x, dict)) or str(inp)
        return "ask_user", {"question": q}
    return f"UNSUPPORTED:{name}", {}


def _dsh_output(name: str, content: Any, is_error: bool) -> tuple[str, bool]:
    txt = _text_of(content)
    if name == "read_file":
        m = _DSH_READ_RE.match(txt)
        if m:
            txt = m.group(1)
    m = _DSH_EXIT_RE.search(txt)
    if m and m.group(1) != "0":
        is_error = True
    txt = _DSH_EXIT_RE.sub(lambda mm: f"[exit {mm.group(1)}]", txt)
    return txt.strip(), bool(is_error or txt.startswith("Error:"))


def convert_dsh(path: Path) -> dict | None:
    stream = subprocess.Popen(
        ["zstdcat", str(path)], stdout=subprocess.PIPE, text=True, errors="ignore"
    ).stdout
    events: list[dict] = []
    sid = project = started = None
    names: dict[str, str] = {}
    for line in stream:
        try:
            r = json.loads(line)
        except Exception:
            continue
        t = r.get("type")
        d = r.get("data") or {}
        if t == "session":
            if (r.get("delegationDepth") or 0) > 0:
                return (
                    None  # subagent session: kept raw, not used (same policy as Claude sidechains)
                )
            sid, project, started = r.get("id"), r.get("cwd"), _ms_iso(r.get("createdAt"))
        elif t == "user/message":
            kind = (d.get("source") or {}).get("kind")
            if kind == "compact-checkpoint":
                events.append({"role": "break", "reason": "compaction"})
                continue
            if kind != "user":
                continue
            txt = _text_of(d.get("content")).strip()
            if txt and not txt.startswith(USER_SKIP_PREFIXES):
                events.append({"role": "user", "content": txt, "ts": _ms_iso(r.get("time"))})
        elif t == "assistant/message":
            m = d.get("message") or {}
            a = {"role": "assistant", "content": "", "tool_calls": []}
            for b in m.get("content") or []:
                if not isinstance(b, dict):
                    continue
                if b.get("type") == "text":
                    a["content"] = (a["content"] + "\n" + b.get("text", "")).strip()
                elif b.get("type") == "tool-call":
                    try:
                        inp = json.loads(b.get("arguments") or "{}")
                    except Exception:
                        inp = None
                    name, args = (
                        _map_dsh_tool(b.get("name", ""), inp)
                        if isinstance(inp, dict)
                        else ("UNSUPPORTED:bad-args", {})
                    )
                    names[b.get("id")] = name
                    a["tool_calls"].append({"id": b.get("id"), "name": name, "arguments": args})
            if a["content"] or a["tool_calls"]:
                events.append(a)
        elif t == "tool/result":
            m = d.get("message") or {}
            cid = m.get("toolCallId")
            name = names.get(cid, "UNSUPPORTED:?")
            txt, err = _dsh_output(name, m.get("content"), bool(m.get("isError")))
            events.append(
                {"role": "tool", "tool_call_id": cid, "name": name, "content": txt, "is_error": err}
            )
        elif t == "turn/end":
            kind = (d.get("reason") or {}).get("kind")
            if kind in ("aborted", "interrupted", "error"):
                events.append(
                    {
                        "role": "break",
                        "reason": "user_interrupt" if kind != "error" else "harness_error",
                    }
                )
    if not any(e["role"] == "user" for e in events):
        return None
    return {
        "session_id": f"ds-{sid or path.parent.name}",
        "source": "deepseek",
        "project": project,
        "started_at": started,
        "events": events,
    }


def main() -> None:
    raw, out = Path(sys.argv[1]), Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    n = {"claude_code": 0, "codex": 0, "deepseek": 0, "skipped": 0}
    seen = set()
    for sub in ("claude-borg", "claude-local"):
        for f in sorted((raw / sub).rglob("*.jsonl")) if (raw / sub).exists() else []:
            if "subagents" in f.parts or f.stem in seen:
                continue
            seen.add(f.stem)
            s = convert_claude(f)
            if s is None:
                n["skipped"] += 1
                continue
            (out / f"{s['session_id']}.json").write_text(json.dumps(s, ensure_ascii=False))
            n["claude_code"] += 1
    for f in sorted((raw / "codex-borg").rglob("rollout-*")):
        if f.name.endswith(".jsonl.zst") and f.with_suffix("").exists():
            continue
        s = convert_codex(f)
        if s is None or s["session_id"] in seen:
            n["skipped"] += 1
            continue
        seen.add(s["session_id"])
        (out / f"{s['session_id']}.json").write_text(json.dumps(s, ensure_ascii=False))
        n["codex"] += 1
    for f in (
        sorted((raw / "deepseek-local").rglob("session.v4.jsonl.zstd"))
        if (raw / "deepseek-local").exists()
        else []
    ):
        s = convert_dsh(f)
        if s is None or s["session_id"] in seen:
            n["skipped"] += 1
            continue
        seen.add(s["session_id"])
        (out / f"{s['session_id']}.json").write_text(json.dumps(s, ensure_ascii=False))
        n["deepseek"] += 1
    print(json.dumps(n))


if __name__ == "__main__":
    main()
