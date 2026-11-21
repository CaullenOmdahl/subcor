"""Shared helpers for the BFCL tool-calling data pipeline (stdlib + numpy only).

Normalisation (OpenAI ``messages`` + ``tools``), JSON-schema type repair, text
shingling and a small deterministic MinHash/LSH used for decontamination.

The scripts in this directory are meant to be run with ``python -I`` against data in a
scratch directory; they add their own directory to ``sys.path`` explicitly.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import defaultdict
from typing import Any

import numpy as np

# --------------------------------------------------------------------------------------
# Tool schemas
# --------------------------------------------------------------------------------------

_TYPE_MAP = {
    "dict": "object",
    "object": "object",
    "float": "number",
    "double": "number",
    "number": "number",
    "int": "integer",
    "integer": "integer",
    "long": "integer",
    "str": "string",
    "string": "string",
    "char": "string",
    "bool": "boolean",
    "boolean": "boolean",
    "list": "array",
    "tuple": "array",
    "array": "array",
    "arraylist": "array",
    "hashmap": "object",
    "null": "null",
}

_NAME_BAD = re.compile(r"[^A-Za-z0-9_-]+")


def sanitize_name(name: str) -> str:
    """OpenAI-legal function name (``^[A-Za-z0-9_-]{1,64}$``); BFCL FC mode maps '.'→'_'."""
    out = _NAME_BAD.sub("_", str(name).strip()).strip("_")
    return out[:64] or "fn"


def norm_schema(s: Any) -> dict[str, Any]:
    """Repair BFCL/ToolACE-style pseudo types (dict/float/tuple/any) into JSON Schema."""
    if not isinstance(s, dict):
        return {}
    out: dict[str, Any] = {}
    for k, v in s.items():
        if k == "type":
            if isinstance(v, list):
                ts = [_TYPE_MAP.get(str(t).lower()) for t in v]
                ts = [t for t in ts if t]
                if ts:
                    out["type"] = ts if len(ts) > 1 else ts[0]
            else:
                t = _TYPE_MAP.get(str(v).lower())
                if t:
                    out["type"] = t
                # "any" / unknown -> no type constraint
        elif k == "properties":
            if isinstance(v, dict):
                out["properties"] = {str(pk): norm_schema(pv) for pk, pv in v.items()}
        elif k == "items":
            if isinstance(v, dict):
                out["items"] = norm_schema(v)
            elif isinstance(v, list) and v:
                out["items"] = norm_schema(v[0]) if isinstance(v[0], dict) else {}
        elif k == "additionalProperties":
            if isinstance(v, dict):
                out[k] = norm_schema(v)
            elif isinstance(v, bool):
                out[k] = v
        elif k == "required":
            if isinstance(v, list):
                out["required"] = [str(x) for x in v if isinstance(x, str)]
        elif k in ("anyOf", "oneOf", "allOf"):
            if isinstance(v, list):
                out[k] = [norm_schema(x) for x in v if isinstance(x, dict)]
        elif k == "enum":
            if isinstance(v, list) and v:
                out["enum"] = v
        elif k in ("minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum"):
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                out[k] = v
        elif k in ("minItems", "maxItems", "minLength", "maxLength"):
            if isinstance(v, int) and not isinstance(v, bool):
                out[k] = v
        elif k == "pattern":
            if isinstance(v, str):
                try:
                    re.compile(v)
                    out[k] = v
                except re.error:
                    pass
        elif k in ("description", "title", "format"):
            if isinstance(v, str):
                out[k] = v
        elif k == "default":
            out[k] = v
        # everything else (e.g. ToolACE "required": null, "optional", "response") dropped
    if "properties" in out and "type" not in out:
        out["type"] = "object"
    if out.get("type") == "object" and "required" in out and "properties" in out:
        out["required"] = [r for r in out["required"] if r in out["properties"]]
    if "enum" in out and out.get("type") in ("number", "integer", "boolean", "string"):
        t = out["type"]
        ok = {
            "number": lambda x: isinstance(x, (int, float)) and not isinstance(x, bool),
            "integer": lambda x: isinstance(x, int) and not isinstance(x, bool),
            "boolean": lambda x: isinstance(x, bool),
            "string": lambda x: isinstance(x, str),
        }[t]
        if not all(ok(x) for x in out["enum"]):
            out.pop("enum")
    return out


def norm_parameters(p: Any) -> dict[str, Any]:
    s = norm_schema(p) if isinstance(p, dict) else {}
    s["type"] = "object"
    s.setdefault("properties", {})
    if not isinstance(s.get("required"), list):
        s["required"] = []
    s["required"] = [r for r in s["required"] if r in s["properties"]]
    return s


def to_openai_tool(fn: dict[str, Any]) -> dict[str, Any]:
    """``{name, description, parameters}`` (or already-wrapped) -> OpenAI tool entry."""
    if fn.get("type") == "function" and isinstance(fn.get("function"), dict):
        fn = fn["function"]
    return {
        "type": "function",
        "function": {
            "name": sanitize_name(fn.get("name", "")),
            "description": str(fn.get("description") or ""),
            "parameters": norm_parameters(fn.get("parameters")),
        },
    }


def _strip_descriptions(s: Any) -> Any:
    if isinstance(s, dict):
        return {k: _strip_descriptions(v) for k, v in s.items() if k not in ("description", "title")}
    if isinstance(s, list):
        return [_strip_descriptions(x) for x in s]
    return s


def schema_key(fn: dict[str, Any]) -> str:
    """Exact identity of a function: sanitized lowercase name + canonical normalized params."""
    if fn.get("type") == "function" and isinstance(fn.get("function"), dict):
        fn = fn["function"]
    params = norm_parameters(fn.get("parameters"))
    return sanitize_name(fn.get("name", "")).lower() + "|" + json.dumps(params, sort_keys=True)


def signature_key(fn: dict[str, Any]) -> str:
    """Looser identity: sanitized lowercase name + sorted property names (types/descriptions ignored)."""
    if fn.get("type") == "function" and isinstance(fn.get("function"), dict):
        fn = fn["function"]
    params = norm_parameters(fn.get("parameters"))
    return sanitize_name(fn.get("name", "")).lower() + "|" + ",".join(sorted(params["properties"]))


def name_key(fn: dict[str, Any]) -> str:
    if fn.get("type") == "function" and isinstance(fn.get("function"), dict):
        fn = fn["function"]
    return sanitize_name(fn.get("name", "")).lower()


# --------------------------------------------------------------------------------------
# Text: tokens, n-grams, MinHash
# --------------------------------------------------------------------------------------

_WORD = re.compile(r"\w+", re.UNICODE)


def words(text: str) -> list[str]:
    return _WORD.findall(str(text).lower())


def _h64(s: str) -> int:
    return int.from_bytes(hashlib.blake2b(s.encode("utf-8"), digest_size=8).digest(), "little")


def ngram_hashes(toks: list[str], n: int) -> set[int]:
    """Word n-gram hashes; a text shorter than n words is one gram (the whole sequence)."""
    if not toks:
        return set()
    if len(toks) < n:
        return {_h64(" ".join(toks))}
    return {_h64(" ".join(toks[i : i + n])) for i in range(len(toks) - n + 1)}


class NgramIndex:
    """Containment index: overlap(a, b) = |A∩B| / min(|A|, |B|) over word 13-grams."""

    def __init__(self, n: int = 13) -> None:
        self.n = n
        self.post: dict[int, list[int]] = defaultdict(list)
        self.size: list[int] = []

    def add(self, text: str) -> int:
        grams = ngram_hashes(words(text), self.n)
        i = len(self.size)
        self.size.append(len(grams))
        for g in grams:
            self.post[g].append(i)
        return i

    def max_overlap(self, text: str) -> tuple[float, int]:
        grams = ngram_hashes(words(text), self.n)
        if not grams:
            return 0.0, -1
        hits: dict[int, int] = defaultdict(int)
        for g in grams:
            for i in self.post.get(g, ()):
                hits[i] += 1
        best, arg = 0.0, -1
        for i, c in hits.items():
            ov = c / max(1, min(len(grams), self.size[i]))
            if ov > best:
                best, arg = ov, i
        return best, arg


_P = (1 << 31) - 1


class MinHashLSH:
    """Deterministic MinHash over word 3-gram shingles + banded LSH; candidates are then
    scored with the EXACT Jaccard of the shingle sets (no estimation error in the verdict)."""

    def __init__(self, num_perm: int = 128, bands: int = 32, shingle: int = 3, seed: int = 1) -> None:
        assert num_perm % bands == 0
        rng = np.random.RandomState(seed)
        self.a = rng.randint(1, _P, size=num_perm, dtype=np.int64).astype(np.uint64)
        self.b = rng.randint(0, _P, size=num_perm, dtype=np.int64).astype(np.uint64)
        self.bands, self.rows, self.k = bands, num_perm // bands, shingle
        self.tables: list[dict[bytes, list[int]]] = [defaultdict(list) for _ in range(bands)]
        self.sets: list[frozenset[int]] = []

    def shingles(self, text: str) -> frozenset[int]:
        return frozenset(ngram_hashes(words(text), self.k))

    def _sig(self, sh: frozenset[int]) -> np.ndarray:
        x = (np.fromiter(sh, dtype=np.uint64, count=len(sh)) % np.uint64(_P))[None, :]
        return ((self.a[:, None] * x + self.b[:, None]) % np.uint64(_P)).min(axis=1)

    def add(self, text: str) -> int:
        sh = self.shingles(text)
        i = len(self.sets)
        self.sets.append(sh)
        if sh:
            sig = self._sig(sh)
            for bi in range(self.bands):
                self.tables[bi][sig[bi * self.rows : (bi + 1) * self.rows].tobytes()].append(i)
        return i

    def max_jaccard(self, text: str) -> tuple[float, int]:
        sh = self.shingles(text)
        if not sh:
            return 0.0, -1
        sig = self._sig(sh)
        cands: set[int] = set()
        for bi in range(self.bands):
            cands.update(self.tables[bi].get(sig[bi * self.rows : (bi + 1) * self.rows].tobytes(), ()))
        best, arg = 0.0, -1
        for i in cands:
            o = self.sets[i]
            j = len(sh & o) / max(1, len(sh | o))
            if j > best:
                best, arg = j, i
        return best, arg


# --------------------------------------------------------------------------------------
# Conversation helpers
# --------------------------------------------------------------------------------------


def user_texts(messages: list[dict[str, Any]]) -> list[str]:
    return [str(m.get("content") or "") for m in messages if m.get("role") == "user"]


def est_tokens(row: dict[str, Any]) -> int:
    """Cheap length estimate (~3.6 chars/token over the JSON of tools + messages)."""
    return int(len(json.dumps(row.get("tools", []))) / 3.6 + len(json.dumps(row["messages"])) / 3.6)
