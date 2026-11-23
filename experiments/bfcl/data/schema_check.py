"""Validate one OpenAI-format tool conversation (``{"messages": [...], "tools": [...]}``).

Checks every assistant ``tool_calls`` entry: the function exists in ``tools``, its
``arguments`` is a JSON string decoding to an object, and the object validates against the
function's ``parameters`` JSON Schema (jsonschema Draft 7). Also checks role order and that
each ``tool`` message answers a pending ``tool_call_id``.
"""

from __future__ import annotations

import json
from typing import Any

import jsonschema
from jsonschema import Draft7Validator

_cache: dict[str, Draft7Validator] = {}


def _validator(schema: dict[str, Any]) -> Draft7Validator:
    key = json.dumps(schema, sort_keys=True)
    v = _cache.get(key)
    if v is None:
        Draft7Validator.check_schema(schema)
        v = Draft7Validator(schema)
        if len(_cache) > 50000:
            _cache.clear()
        _cache[key] = v
    return v


def check_conversation(row: dict[str, Any]) -> list[str]:
    """Return a list of error strings (empty = valid)."""
    errs: list[str] = []
    tools = row.get("tools") or []
    by_name: dict[str, dict[str, Any]] = {}
    for t in tools:
        fn = t.get("function") if isinstance(t, dict) else None
        if t.get("type") != "function" or not isinstance(fn, dict) or not fn.get("name"):
            errs.append("bad tool entry")
            continue
        if fn["name"] in by_name:
            errs.append(f"duplicate tool name {fn['name']}")
        by_name[fn["name"]] = fn
        try:
            _validator(fn.get("parameters") or {"type": "object"})
        except jsonschema.SchemaError as e:
            errs.append(f"invalid schema for {fn['name']}: {e.message[:120]}")
    msgs = row.get("messages")
    if not isinstance(msgs, list) or not msgs:
        return errs + ["no messages"]
    pending: list[str] = []
    seen_ids: set[str] = set()
    n_user = 0
    for i, m in enumerate(msgs):
        role = m.get("role")
        if role not in ("system", "user", "assistant", "tool"):
            errs.append(f"msg {i}: bad role {role}")
            continue
        if role == "system" and i != 0:
            errs.append(f"msg {i}: system not first")
        if role == "user":
            n_user += 1
            if pending:
                errs.append(f"msg {i}: user turn while tool results pending")
                pending = []
        if role == "assistant":
            if pending:
                errs.append(f"msg {i}: assistant turn while tool results pending")
                pending = []
            calls = m.get("tool_calls") or []
            if not calls and not str(m.get("content") or "").strip():
                errs.append(f"msg {i}: empty assistant turn")
            for c in calls:
                fn = c.get("function") or {}
                name = fn.get("name")
                cid = c.get("id")
                if c.get("type") != "function" or not cid:
                    errs.append(f"msg {i}: malformed tool_call")
                if cid in seen_ids:
                    errs.append(f"msg {i}: duplicate tool_call id {cid}")
                seen_ids.add(cid)
                pending.append(cid)
                if name not in by_name:
                    errs.append(f"msg {i}: call to unknown function {name}")
                    continue
                a = fn.get("arguments")
                if not isinstance(a, str):
                    errs.append(f"msg {i}: arguments not a JSON string")
                    continue
                try:
                    args = json.loads(a)
                except json.JSONDecodeError:
                    errs.append(f"msg {i}: arguments not valid JSON")
                    continue
                if not isinstance(args, dict):
                    errs.append(f"msg {i}: arguments not an object")
                    continue
                schema = by_name[name].get("parameters") or {"type": "object"}
                try:
                    v = _validator(schema)
                except jsonschema.SchemaError:
                    continue  # already reported
                e = next(iter(v.iter_errors(args)), None)
                if e is not None:
                    errs.append(f"msg {i}: {name} args fail schema: {e.message[:160]}")
        if role == "tool":
            tid = m.get("tool_call_id")
            if tid not in pending:
                errs.append(f"msg {i}: tool message for unknown/answered id {tid}")
            else:
                pending.remove(tid)
            if not isinstance(m.get("content"), str):
                errs.append(f"msg {i}: tool content not a string")
    if n_user == 0:
        errs.append("no user turn")
    if msgs[-1].get("role") != "assistant":
        errs.append("does not end on an assistant turn")
    return errs
