"""Shared client + JSONL plumbing for the general-capability regression evals.

Every eval talks to an OpenAI-compatible ``POST {base_url}/chat/completions`` endpoint:

  * plain llama-server (``--jinja``): the "base, no SUBCOR" arm, or
  * ``subcor serve`` (model ``subcor/<tenant>`` = routed, ``subcor/base`` = forced base),
    so routing decisions happen inside SUBCOR's server and are read back from the
    response's ``subcor`` field / ``X-SUBCOR-Routed-Units`` header.

Decoding is fixed: temperature 0, thinking disabled. Thinking is disabled both ways
because the two servers take different knobs: llama-server reads
``chat_template_kwargs.enable_thinking``; SUBCOR reads the top-level ``enable_thinking``
(and itself always sends temperature 0 + ``chat_template_kwargs`` to its backend).
Any residual ``<think>...</think>`` block is stripped before grading and counted.

Stdlib only (urllib), so it runs in the eval venv or the project env.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sys
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

_THINK_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL)


@dataclass
class ChatConfig:
    base_url: str  # e.g. http://127.0.0.1:8502/v1
    model: str
    api_key: str | None = None
    timeout_s: float = 900.0
    retries: int = 3
    extra_body: dict[str, Any] = field(default_factory=dict)

    def describe(self) -> dict[str, Any]:
        return {
            "base_url": self.base_url,
            "model": self.model,
            "extra_body": self.extra_body,
        }


def add_endpoint_args(ap: Any) -> None:
    ap.add_argument("--base-url", default="http://127.0.0.1:8502/v1", help="OpenAI base URL (.../v1)")
    ap.add_argument("--model", default="local", help="model field ('subcor/<tenant>', 'subcor/base', ...)")
    ap.add_argument("--api-key", default=os.environ.get("EVAL_API_KEY"))
    ap.add_argument("--concurrency", type=int, default=1)
    ap.add_argument("--timeout", type=float, default=900.0)
    ap.add_argument("--limit", type=int, default=None, help="first N items only (smoke tests)")
    ap.add_argument("--out", type=Path, required=True, help="per-item JSONL (resumable)")
    ap.add_argument("--summary", type=Path, default=None, help="summary JSON (default: <out>.summary.json)")
    ap.add_argument("--run-label", default=None, help="free-form arm label stored in every row")
    ap.add_argument(
        "--extra-body",
        default=None,
        help="JSON merged into every request body (e.g. '{\"top_k\":2}' for SUBCOR routing top-k)",
    )


def config_from_args(args: Any) -> ChatConfig:
    return ChatConfig(
        base_url=args.base_url.rstrip("/"),
        model=args.model,
        api_key=args.api_key,
        timeout_s=args.timeout,
        extra_body=json.loads(args.extra_body) if args.extra_body else {},
    )


def strip_think(text: str) -> tuple[str, bool]:
    """Remove <think> blocks (and a dangling unterminated one). Returns (text, had_think)."""
    had = "<think>" in text
    out = _THINK_RE.sub("", text)
    if "<think>" in out:  # unterminated think (hit the token limit while thinking)
        out = out.split("<think>", 1)[0]
    if out.lstrip().startswith("</think>"):
        out = out.lstrip()[len("</think>") :]
    return out.strip(), had


def chat(
    cfg: ChatConfig,
    messages: list[dict[str, Any]],
    *,
    max_tokens: int,
    tools: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """One non-streaming chat completion. Returns a normalized record (never raises on HTTP)."""
    body: dict[str, Any] = {
        "model": cfg.model,
        "messages": messages,
        "max_tokens": int(max_tokens),
        "temperature": 0.0,
        "seed": 0,
        "stream": False,
        # llama-server knob:
        "chat_template_kwargs": {"enable_thinking": False},
        # SUBCOR knob (ignored by llama-server):
        "enable_thinking": False,
    }
    if tools:
        body["tools"] = tools
    body.update(cfg.extra_body)
    data = json.dumps(body).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if cfg.api_key:
        headers["Authorization"] = f"Bearer {cfg.api_key}"

    last_err = ""
    for attempt in range(cfg.retries + 1):
        t0 = time.perf_counter()
        try:
            req = urllib.request.Request(
                f"{cfg.base_url}/chat/completions", data=data, headers=headers, method="POST"
            )
            with urllib.request.urlopen(req, timeout=cfg.timeout_s) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
                hdr_units = resp.headers.get("X-SUBCOR-Routed-Units")
            latency = time.perf_counter() - t0
            choice = (payload.get("choices") or [{}])[0]
            msg = choice.get("message") or {}
            raw = str(msg.get("content") or "")
            text, had_think = strip_think(raw)
            sub = payload.get("subcor")
            return {
                "ok": True,
                "raw_content": raw,
                "text": text,
                "had_think": had_think,
                "reasoning_content": msg.get("reasoning_content") or None,
                "tool_calls": msg.get("tool_calls") or None,
                "finish_reason": choice.get("finish_reason"),
                "usage": payload.get("usage") or {},
                "latency_s": round(latency, 3),
                "served_model": payload.get("model"),
                "subcor": sub,
                "selected_units": (
                    list(sub.get("selected_units") or []) if isinstance(sub, dict) else None
                ),
                "routed_header": hdr_units,
                "attempts": attempt + 1,
            }
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")[:500]
            last_err = f"HTTP {exc.code}: {detail}"
            if 400 <= exc.code < 500 and exc.code != 429:
                break  # deterministic client error; do not retry
        except Exception as exc:  # noqa: BLE001 - network/timeouts
            last_err = f"{type(exc).__name__}: {exc}"
        time.sleep(min(2**attempt, 10))
    return {"ok": False, "error": last_err, "text": "", "selected_units": None}


# ----------------------------------------------------------------------------- JSONL


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def sha_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def run_items(
    items: list[dict[str, Any]],
    worker: Callable[[dict[str, Any]], dict[str, Any]],
    *,
    out_path: Path,
    concurrency: int,
    fingerprint: str,
    id_key: str = "id",
) -> list[dict[str, Any]]:
    """Run `worker` on every item not already (successfully) in out_path; append rows.

    Resumable: rows with ok=True are kept, failed rows are retried on the next invocation.
    Every row carries `fingerprint` (endpoint+model+decoding config); resuming into a file
    written under a different fingerprint is refused so arms never mix in one file.
    Returns all rows for the requested items, in item order.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    prior = read_jsonl(out_path)
    foreign = {r.get("fingerprint") for r in prior} - {fingerprint}
    if foreign:
        raise SystemExit(
            f"{out_path} holds rows from a different run config {sorted(map(str, foreign))}; "
            "use a new --out path per arm."
        )
    existing = {r[id_key]: r for r in prior if r.get("ok")}
    todo = [it for it in items if it[id_key] not in existing]
    if existing:
        # Rewrite without failed rows so the file holds one row per id.
        with out_path.open("w", encoding="utf-8") as fh:
            for r in existing.values():
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    elif out_path.exists():
        out_path.unlink()
    lock = threading.Lock()
    done = 0
    t_start = time.perf_counter()

    def _write(row: dict[str, Any]) -> None:
        nonlocal done
        row["fingerprint"] = fingerprint
        with lock:
            with out_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            existing[row[id_key]] = row
            done += 1
            if done % 10 == 0 or done == len(todo):
                rate = done / max(time.perf_counter() - t_start, 1e-9)
                print(f"  [{done}/{len(todo)}] {rate:.2f} items/s", file=sys.stderr, flush=True)

    if concurrency <= 1:
        for it in todo:
            _write(worker(it))
    else:
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            futs = [pool.submit(worker, it) for it in todo]
            for fut in as_completed(futs):
                _write(fut.result())
    return [existing[it[id_key]] for it in items if it[id_key] in existing]


def routing_summary(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Fraction of requests where SUBCOR activated >=1 unit, plus per-unit counts."""
    rows = [r for r in rows if r.get("ok")]
    seen = [r for r in rows if r.get("selected_units") is not None]
    if not seen:
        return {"n": len(rows), "subcor_field_present": False}
    per_unit: dict[str, int] = {}
    active = 0
    for r in seen:
        units = r["selected_units"]
        if units:
            active += 1
        for u in units:
            per_unit[u] = per_unit.get(u, 0) + 1
    return {
        "n": len(seen),
        "subcor_field_present": True,
        "any_unit_active": active,
        "any_unit_active_rate": active / len(seen),
        "per_unit_counts": dict(sorted(per_unit.items())),
    }


def fingerprint_of(*parts: Any) -> str:
    return sha_text(json.dumps(parts, sort_keys=True, default=str))


def write_summary(path: Path, summary: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def default_summary_path(out: Path, summary: Path | None) -> Path:
    return summary if summary is not None else out.with_suffix(".summary.json")
