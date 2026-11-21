"""Standardized longitudinal benchmark system for documenting SUBCOR functional growth.

Provides:
  - Versioned, content-hashed BenchmarkSuites (SHA-256 suite_digest prevents
    moving goalposts across snapshots).
  - Deterministic graders: exact_json, json_keys, numeric, contains, regex.
  - Built-in standardized `subcor-core-v1` suite covering 5 functional axes:
      1. general_reasoning      (locked non-regression anchor)
      2. instruction_discipline (locked non-regression anchor: concise stopping, no loops)
      3. structured_json        (strict JSON tool/schema formatting)
      4. domain_ops_routing     (CaPheOS company ops governance routing)
      5. domain_noc_triage      (CaPheOS overnight NOC incident SLA policy)
  - Execution against EITHER a local ServingKernel/TenantManager OR any live
    OpenAI-compatible HTTP endpoint (including `subcor serve` or frontier
    teacher models for Teacher-vs-Intern gap tracking).
  - Append-only JSONL BenchmarkLedger + diff/regression analyzer + Markdown
    longitudinal Growth Report generator.
"""

from __future__ import annotations

import hashlib
import json
import re
import threading
import time
import urllib.request
import warnings
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from subcor.server import ServingKernel, TenantContext

LOCKED_ANCHOR_AXES: frozenset[str] = frozenset({"general_reasoning", "instruction_discipline"})


@dataclass(frozen=True)
class BenchmarkCase:
    """Single deterministic evaluation case belonging to one functional axis."""

    case_id: str
    axis: str
    prompt: str
    grader: str  # "exact_json" | "json_keys" | "numeric" | "contains" | "regex"
    expected: Any
    system_prompt: str | None = None
    max_tokens: int = 48
    enable_thinking: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "axis": self.axis,
            "prompt": self.prompt,
            "grader": self.grader,
            "expected": self.expected,
            "system_prompt": self.system_prompt,
            "max_tokens": self.max_tokens,
            "enable_thinking": self.enable_thinking,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BenchmarkCase:
        return cls(
            case_id=str(data["case_id"]),
            axis=str(data["axis"]),
            prompt=str(data["prompt"]),
            grader=str(data["grader"]),
            expected=data["expected"],
            system_prompt=data.get("system_prompt"),
            max_tokens=int(data.get("max_tokens", 48)),
            enable_thinking=bool(data.get("enable_thinking", False)),
        )


@dataclass(frozen=True)
class BenchmarkSuite:
    """Immutable, content-hashed collection of BenchmarkCases across capability axes."""

    suite_id: str
    version: str
    description: str
    cases: tuple[BenchmarkCase, ...]
    anchor_axes: tuple[str, ...] = ("general_reasoning", "instruction_discipline")

    @property
    def axes(self) -> tuple[str, ...]:
        seen: list[str] = []
        for c in self.cases:
            if c.axis not in seen:
                seen.append(c.axis)
        return tuple(seen)

    def digest(self) -> str:
        """Deterministic SHA-256 over canonical suite definition (prevents goalpost drift)."""
        canonical = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
        return hashlib.sha256(canonical).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        return {
            "suite_id": self.suite_id,
            "version": self.version,
            "description": self.description,
            "anchor_axes": list(self.anchor_axes),
            "cases": [c.to_dict() for c in self.cases],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BenchmarkSuite:
        return cls(
            suite_id=str(data["suite_id"]),
            version=str(data["version"]),
            description=str(data.get("description", "")),
            anchor_axes=tuple(str(x) for x in data.get("anchor_axes", [])),
            cases=tuple(BenchmarkCase.from_dict(c) for c in data["cases"]),
        )

    @classmethod
    def from_json_file(cls, path: Path) -> BenchmarkSuite:
        return cls.from_dict(json.loads(path.read_text(encoding="utf-8")))

    def to_json_file(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8")


def _strip_code_fences(text: str) -> str:
    clean = text.strip()
    if clean.startswith("```"):
        clean = re.sub(r"^```[a-zA-Z]*\s*|\s*```$", "", clean).strip()
    return clean


_NEGATION_PREFIX_RE = re.compile(
    r"(?:\b(?:not|never|neither|no|isn't|is\s+not|aren't|are\s+not|wasn't|was\s+not|rather\s+than|instead\s+of)\s+)(?:\w+\s+){0,1}$",
    flags=re.IGNORECASE,
)
_NUMERIC_TOKEN_RE = re.compile(r"(?<![\w.])-?\d+(?:\.\d+)?(?!\w|\.\d)")


def _strict_json_equal(actual: Any, expected: Any) -> bool:
    """JSON equality where booleans never compare equal to ints (True != 1)."""
    if isinstance(expected, bool) or isinstance(actual, bool):
        return type(actual) is type(expected) and actual == expected
    if isinstance(expected, dict) and isinstance(actual, dict):
        return actual.keys() == expected.keys() and all(
            _strict_json_equal(actual[k], expected[k]) for k in expected
        )
    if isinstance(expected, list) and isinstance(actual, list):
        return len(actual) == len(expected) and all(
            _strict_json_equal(a, e) for a, e in zip(actual, expected, strict=True)
        )
    return actual == expected


def _affirmative_contains(text: str, needle: str, *, case_sensitive: bool = False) -> bool:
    """Return True if needle appears affirmatively (not immediately negated) in text."""
    flags = 0 if case_sensitive else re.IGNORECASE
    pattern = re.compile(
        rf"(?<!\w){re.escape(needle)}(?!\w)" if needle.isalnum() else re.escape(needle), flags
    )
    for m in pattern.finditer(text):
        prefix = text[max(0, m.start() - 32) : m.start()]
        if not _NEGATION_PREFIX_RE.search(prefix):
            return True
    return False


def grade_response(case: BenchmarkCase, response: str, *, hit_token_limit: bool) -> bool:
    """Deterministically grade a model response against a BenchmarkCase."""
    if case.grader == "exact_json":
        try:
            parsed = json.loads(_strip_code_fences(response))
        except json.JSONDecodeError:
            return False
        if not isinstance(parsed, dict) or not isinstance(case.expected, dict):
            return False
        return all(
            k in parsed and _strict_json_equal(parsed[k], v) for k, v in case.expected.items()
        )

    if case.grader == "json_keys":
        try:
            parsed = json.loads(_strip_code_fences(response))
        except json.JSONDecodeError:
            return False
        if not isinstance(parsed, dict):
            return False
        req_keys = [str(k) for k in case.expected]
        return all(k in parsed for k in req_keys)

    if case.grader == "numeric":
        target = str(case.expected).strip()
        nums = _NUMERIC_TOKEN_RE.findall(response.strip())
        if not nums:
            return False
        return nums[-1] == target

    if case.grader == "contains":
        if isinstance(case.expected, dict):
            needle = str(case.expected["contains"]).strip()
            case_sensitive = bool(case.expected.get("case_sensitive", False))
        else:
            needle = str(case.expected).strip()
            case_sensitive = False
        return _affirmative_contains(response.strip(), needle, case_sensitive=case_sensitive)

    if case.grader == "regex":
        return bool(re.search(str(case.expected), response.strip(), flags=re.IGNORECASE))

    if case.grader == "concise_contains":
        if hit_token_limit:
            return False
        if isinstance(case.expected, dict):
            needle = str(case.expected["contains"])
            max_words = int(case.expected.get("max_words", 25))
            case_sensitive = bool(case.expected.get("case_sensitive", False))
            words = response.strip().split()
            return _affirmative_contains(
                response.strip(), needle, case_sensitive=case_sensitive
            ) and (len(words) <= max_words)
        return _affirmative_contains(response.strip(), str(case.expected), case_sensitive=False)

    raise ValueError(f"Unsupported grader type: {case.grader!r}")


@dataclass(frozen=True)
class CaseResult:
    """Observed result for one BenchmarkCase execution."""

    case_id: str
    axis: str
    passed: bool
    response: str
    completion_tokens: int
    hit_token_limit: bool
    repetition_ratio: float
    latency_ms: float
    routed_units: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "axis": self.axis,
            "passed": self.passed,
            "response": self.response,
            "completion_tokens": self.completion_tokens,
            "hit_token_limit": self.hit_token_limit,
            "repetition_ratio": round(self.repetition_ratio, 4),
            "latency_ms": round(self.latency_ms, 2),
            "routed_units": list(self.routed_units),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CaseResult:
        return cls(
            case_id=str(data["case_id"]),
            axis=str(data["axis"]),
            passed=bool(data["passed"]),
            response=str(data["response"]),
            completion_tokens=int(data["completion_tokens"]),
            hit_token_limit=bool(data["hit_token_limit"]),
            repetition_ratio=float(data["repetition_ratio"]),
            latency_ms=float(data["latency_ms"]),
            routed_units=tuple(str(u) for u in data.get("routed_units", [])),
        )


@dataclass(frozen=True)
class AxisSummary:
    """Aggregated score & efficiency summary for one capability axis."""

    axis: str
    total: int
    passed: int
    pass_rate: float
    token_limit_rate: float
    mean_tokens: float
    mean_latency_ms: float
    base_fallback_rate: float
    routed_unit_counts: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "axis": self.axis,
            "total": self.total,
            "passed": self.passed,
            "pass_rate": round(self.pass_rate, 6),
            "token_limit_rate": round(self.token_limit_rate, 6),
            "mean_tokens": round(self.mean_tokens, 2),
            "mean_latency_ms": round(self.mean_latency_ms, 2),
            "base_fallback_rate": round(self.base_fallback_rate, 6),
            "routed_unit_counts": dict(self.routed_unit_counts),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AxisSummary:
        return cls(
            axis=str(data["axis"]),
            total=int(data["total"]),
            passed=int(data["passed"]),
            pass_rate=float(data["pass_rate"]),
            token_limit_rate=float(data["token_limit_rate"]),
            mean_tokens=float(data["mean_tokens"]),
            mean_latency_ms=float(data["mean_latency_ms"]),
            base_fallback_rate=float(data["base_fallback_rate"]),
            routed_unit_counts={
                str(k): int(v) for k, v in data.get("routed_unit_counts", {}).items()
            },
        )


@dataclass(frozen=True)
class BenchmarkSnapshot:
    """Immutable longitudinal benchmark snapshot stored in BenchmarkLedger."""

    run_id: str
    label: str
    suite_id: str
    suite_version: str
    suite_digest: str
    model_id: str
    tenant_id: str | None
    active_units: tuple[dict[str, Any], ...]
    total_unit_bytes: int
    mean_bits_per_weight: float
    overall_pass_rate: float
    macro_axis_pass_rate: float
    anchor_pass_rate: float
    token_limit_rate: float
    mean_tokens: float
    mean_latency_ms: float
    tokens_per_second: float
    axis_summaries: dict[str, AxisSummary]
    case_results: tuple[CaseResult, ...]
    timestamp: str = field(default_factory=lambda: datetime.now(UTC).isoformat())

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "label": self.label,
            "suite_id": self.suite_id,
            "suite_version": self.suite_version,
            "suite_digest": self.suite_digest,
            "model_id": self.model_id,
            "tenant_id": self.tenant_id,
            "active_units": list(self.active_units),
            "total_unit_bytes": self.total_unit_bytes,
            "mean_bits_per_weight": round(self.mean_bits_per_weight, 6),
            "overall_pass_rate": round(self.overall_pass_rate, 6),
            "macro_axis_pass_rate": round(self.macro_axis_pass_rate, 6),
            "anchor_pass_rate": round(self.anchor_pass_rate, 6),
            "token_limit_rate": round(self.token_limit_rate, 6),
            "mean_tokens": round(self.mean_tokens, 2),
            "mean_latency_ms": round(self.mean_latency_ms, 2),
            "tokens_per_second": round(self.tokens_per_second, 2),
            "axis_summaries": {k: v.to_dict() for k, v in self.axis_summaries.items()},
            "case_results": [r.to_dict() for r in self.case_results],
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BenchmarkSnapshot:
        return cls(
            run_id=str(data["run_id"]),
            label=str(data["label"]),
            suite_id=str(data["suite_id"]),
            suite_version=str(data["suite_version"]),
            suite_digest=str(data["suite_digest"]),
            model_id=str(data["model_id"]),
            tenant_id=data.get("tenant_id"),
            active_units=tuple(dict(u) for u in data.get("active_units", [])),
            total_unit_bytes=int(data.get("total_unit_bytes", 0)),
            mean_bits_per_weight=float(data.get("mean_bits_per_weight", 0.0)),
            overall_pass_rate=float(data["overall_pass_rate"]),
            macro_axis_pass_rate=float(data["macro_axis_pass_rate"]),
            anchor_pass_rate=float(data["anchor_pass_rate"]),
            token_limit_rate=float(data["token_limit_rate"]),
            mean_tokens=float(data["mean_tokens"]),
            mean_latency_ms=float(data["mean_latency_ms"]),
            tokens_per_second=float(data["tokens_per_second"]),
            axis_summaries={
                str(k): AxisSummary.from_dict(v) for k, v in data.get("axis_summaries", {}).items()
            },
            case_results=tuple(CaseResult.from_dict(r) for r in data.get("case_results", [])),
            timestamp=str(data.get("timestamp", datetime.now(UTC).isoformat())),
        )


def core_v1_suite() -> BenchmarkSuite:
    """Built-in standardized 5-axis SUBCOR functional growth benchmark (`subcor-core-v1`)."""
    ops_sys = (
        "You are the CaPheOS Intern router. Reply ONLY with compact JSON containing "
        'keys "seat", "gate", and "code". No prose, no markdown.'
    )
    noc_sys = (
        "You are the CaPheOS overnight NOC triage seat. Reply ONLY with compact JSON "
        'containing keys "sev", "notify", and "sla_min". No prose, no markdown.'
    )
    json_sys = "Reply ONLY with valid JSON. No prose, no markdown fences."

    cases: list[BenchmarkCase] = [
        # --- Axis 1: general_reasoning (locked anchor) ---
        BenchmarkCase(
            case_id="gen-01",
            axis="general_reasoning",
            prompt="What is 12 * 12? Reply with only the number.",
            grader="numeric",
            expected="144",
            max_tokens=32,
        ),
        BenchmarkCase(
            case_id="gen-02",
            axis="general_reasoning",
            prompt="What is the capital of Japan? Reply with only the city name.",
            grader="contains",
            expected="Tokyo",
            max_tokens=32,
        ),
        BenchmarkCase(
            case_id="gen-03",
            axis="general_reasoning",
            prompt="Solve for x: 3x + 7 = 28. Reply with only the number.",
            grader="numeric",
            expected="7",
            max_tokens=32,
        ),
        BenchmarkCase(
            case_id="gen-04",
            axis="general_reasoning",
            prompt="What is 15 + 27? Reply with only the number.",
            grader="numeric",
            expected="42",
            max_tokens=32,
        ),
        BenchmarkCase(
            case_id="gen-05",
            axis="general_reasoning",
            prompt="What planet is closest to the Sun? Reply with only the planet name.",
            grader="contains",
            expected="Mercury",
            max_tokens=32,
        ),
        BenchmarkCase(
            case_id="gen-06",
            axis="general_reasoning",
            prompt="What is 100 - 37? Reply with only the number.",
            grader="numeric",
            expected="63",
            max_tokens=32,
        ),
        # --- Axis 2: instruction_discipline (locked anchor: concise, no runaway) ---
        BenchmarkCase(
            case_id="disc-01",
            axis="instruction_discipline",
            prompt="Respond with exactly the single word READY and nothing else.",
            grader="concise_contains",
            expected={"contains": "ready", "max_words": 3},
            max_tokens=16,
        ),
        BenchmarkCase(
            case_id="disc-02",
            axis="instruction_discipline",
            prompt="Is 19 a prime number? Answer YES or NO in at most 5 words.",
            grader="concise_contains",
            expected={"contains": "yes", "max_words": 5},
            max_tokens=20,
        ),
        BenchmarkCase(
            case_id="disc-03",
            axis="instruction_discipline",
            prompt="Convert 'hello world' to uppercase. Reply in at most 4 words.",
            grader="concise_contains",
            expected={"contains": "HELLO WORLD", "max_words": 4, "case_sensitive": True},
            max_tokens=20,
        ),
        BenchmarkCase(
            case_id="disc-04",
            axis="instruction_discipline",
            prompt="Name the chemical symbol for gold in at most 4 words.",
            grader="concise_contains",
            expected={"contains": "au", "max_words": 4},
            max_tokens=20,
        ),
        # --- Axis 3: structured_json (general tool-call schema discipline) ---
        BenchmarkCase(
            case_id="json-01",
            axis="structured_json",
            system_prompt=json_sys,
            prompt='Extract name and age from "Alice is 29 years old" into JSON keys "name" and "age".',
            grader="exact_json",
            expected={"name": "Alice", "age": 29},
            max_tokens=48,
        ),
        BenchmarkCase(
            case_id="json-02",
            axis="structured_json",
            system_prompt=json_sys,
            prompt='Return JSON with keys "status" set to "ok" and "count" set to 3.',
            grader="exact_json",
            expected={"status": "ok", "count": 3},
            max_tokens=48,
        ),
        BenchmarkCase(
            case_id="json-03",
            axis="structured_json",
            system_prompt=json_sys,
            prompt='Emit a tool call object with keys "tool" and "args" for reading file "README.md".',
            grader="json_keys",
            expected=["tool", "args"],
            max_tokens=64,
        ),
        # --- Axis 4: domain_ops_routing (acquired Skill Unit 1 — unseen events) ---
        BenchmarkCase(
            case_id="ops-01",
            axis="domain_ops_routing",
            system_prompt=ops_sys,
            prompt="Event: Strategic partner proposes an exclusive multi-year distribution contract.",
            grader="exact_json",
            expected={"seat": "ceo", "gate": "owner", "code": "EXEC-CHARTER"},
            max_tokens=48,
        ),
        BenchmarkCase(
            case_id="ops-02",
            axis="domain_ops_routing",
            system_prompt=ops_sys,
            prompt="Event: Annual software renewal bill of $4,200 breaches the approved budget threshold.",
            grader="exact_json",
            expected={"seat": "finance", "gate": "owner", "code": "FIN-CAP"},
            max_tokens=48,
        ),
        BenchmarkCase(
            case_id="ops-03",
            axis="domain_ops_routing",
            system_prompt=ops_sys,
            prompt="Event: Compliance officer flagged a missing key rotation on an API secret file.",
            grader="exact_json",
            expected={"seat": "risk", "gate": "owner", "code": "SEC-LOCK"},
            max_tokens=48,
        ),
        BenchmarkCase(
            case_id="ops-04",
            axis="domain_ops_routing",
            system_prompt=ops_sys,
            prompt="Event: Compile interview notes for two finalists applying to the NOC lead seat.",
            grader="exact_json",
            expected={"seat": "people", "gate": "auto", "code": "HR-ROLE"},
            max_tokens=48,
        ),
        BenchmarkCase(
            case_id="ops-05",
            axis="domain_ops_routing",
            system_prompt=ops_sys,
            prompt="Event: Install and verify digest for the updated calendar capability unit.",
            grader="exact_json",
            expected={"seat": "tools", "gate": "auto", "code": "OPS-UNIT"},
            max_tokens=48,
        ),
        # --- Axis 5: domain_noc_triage (acquired Skill Unit 2 — unseen alerts) ---
        BenchmarkCase(
            case_id="noc-01",
            axis="domain_noc_triage",
            system_prompt=noc_sys,
            prompt="Alert: Core billing service completely offline; all customer transactions failing.",
            grader="exact_json",
            expected={"sev": "P1", "notify": "pager", "sla_min": 15},
            max_tokens=48,
        ),
        BenchmarkCase(
            case_id="noc-02",
            axis="domain_noc_triage",
            system_prompt=noc_sys,
            prompt="Alert: Production DNS zone resolution failing across North America edge nodes.",
            grader="exact_json",
            expected={"sev": "P1", "notify": "pager", "sla_min": 15},
            max_tokens=48,
        ),
        BenchmarkCase(
            case_id="noc-03",
            axis="domain_noc_triage",
            system_prompt=noc_sys,
            prompt="Alert: Notification worker queue backed up with 8,000 pending retries.",
            grader="exact_json",
            expected={"sev": "P2", "notify": "slack", "sla_min": 60},
            max_tokens=48,
        ),
        BenchmarkCase(
            case_id="noc-04",
            axis="domain_noc_triage",
            system_prompt=noc_sys,
            prompt="Alert: API gateway p99 response time spiked to 2.1s after cache flush.",
            grader="exact_json",
            expected={"sev": "P2", "notify": "slack", "sla_min": 60},
            max_tokens=48,
        ),
        BenchmarkCase(
            case_id="noc-05",
            axis="domain_noc_triage",
            system_prompt=noc_sys,
            prompt="Alert: Dev sandbox disk usage reached 80%; prune old container layers this week.",
            grader="exact_json",
            expected={"sev": "P3", "notify": "digest", "sla_min": 480},
            max_tokens=48,
        ),
        BenchmarkCase(
            case_id="noc-06",
            axis="domain_noc_triage",
            system_prompt=noc_sys,
            prompt="Alert: Staging SSL certificate renewal window opens in 21 days.",
            grader="exact_json",
            expected={"sev": "P3", "notify": "digest", "sla_min": 480},
            max_tokens=48,
        ),
    ]

    return BenchmarkSuite(
        suite_id="subcor-core-v1",
        version="1.0.0",
        description=(
            "Standardized 5-axis functional growth suite measuring general reasoning, "
            "instruction discipline, structured JSON formatting, company ops routing, "
            "and overnight NOC triage."
        ),
        cases=tuple(cases),
        anchor_axes=("general_reasoning", "instruction_discipline"),
    )


def _summarize_snapshot(
    *,
    label: str,
    suite: BenchmarkSuite,
    model_id: str,
    tenant_id: str | None,
    active_units: list[dict[str, Any]],
    case_results: list[CaseResult],
) -> BenchmarkSnapshot:
    by_axis: dict[str, list[CaseResult]] = {}
    for res in case_results:
        by_axis.setdefault(res.axis, []).append(res)

    axis_summaries: dict[str, AxisSummary] = {}
    for axis in suite.axes:
        rows = by_axis.get(axis, [])
        n = len(rows)
        if n == 0:
            continue
        passed = sum(1 for r in rows if r.passed)
        lim = sum(1 for r in rows if r.hit_token_limit)
        base_fb = sum(1 for r in rows if len(r.routed_units) == 0)
        unit_counts: dict[str, int] = {}
        for r in rows:
            for u in r.routed_units:
                unit_counts[u] = unit_counts.get(u, 0) + 1
        axis_summaries[axis] = AxisSummary(
            axis=axis,
            total=n,
            passed=passed,
            pass_rate=passed / n,
            token_limit_rate=lim / n,
            mean_tokens=sum(r.completion_tokens for r in rows) / n,
            mean_latency_ms=sum(r.latency_ms for r in rows) / n,
            base_fallback_rate=base_fb / n,
            routed_unit_counts=unit_counts,
        )

    total_n = max(1, len(case_results))
    overall_pass = sum(1 for r in case_results if r.passed) / total_n
    macro_axis = (
        sum(s.pass_rate for s in axis_summaries.values()) / len(axis_summaries)
        if axis_summaries
        else 0.0
    )
    anchor_rows = [r for r in case_results if r.axis in suite.anchor_axes]
    anchor_pass = (
        sum(1 for r in anchor_rows if r.passed) / len(anchor_rows) if anchor_rows else overall_pass
    )
    token_limit_rate = sum(1 for r in case_results if r.hit_token_limit) / total_n
    total_tokens = sum(r.completion_tokens for r in case_results)
    total_ms = sum(r.latency_ms for r in case_results)
    mean_tokens = total_tokens / total_n
    mean_latency_ms = total_ms / total_n
    tok_per_sec = (total_tokens / (total_ms / 1000.0)) if total_ms > 0 else 0.0

    total_unit_bytes = sum(int(u.get("payload_bytes", 0)) for u in active_units)
    bpws = [float(u["bits_per_weight"]) for u in active_units if "bits_per_weight" in u]
    mean_bpw = (sum(bpws) / len(bpws)) if bpws else 0.0

    run_id = f"bench-{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}-{hashlib.sha256(label.encode()).hexdigest()[:6]}"
    return BenchmarkSnapshot(
        run_id=run_id,
        label=label,
        suite_id=suite.suite_id,
        suite_version=suite.version,
        suite_digest=suite.digest(),
        model_id=model_id,
        tenant_id=tenant_id,
        active_units=tuple(active_units),
        total_unit_bytes=total_unit_bytes,
        mean_bits_per_weight=mean_bpw,
        overall_pass_rate=overall_pass,
        macro_axis_pass_rate=macro_axis,
        anchor_pass_rate=anchor_pass,
        token_limit_rate=token_limit_rate,
        mean_tokens=mean_tokens,
        mean_latency_ms=mean_latency_ms,
        tokens_per_second=tok_per_sec,
        axis_summaries=axis_summaries,
        case_results=tuple(case_results),
    )


def run_benchmark_local(
    *,
    suite: BenchmarkSuite,
    kernel: ServingKernel,
    label: str,
    tenant: TenantContext | None = None,
    top_k: int = 1,
) -> BenchmarkSnapshot:
    """Run a BenchmarkSuite directly against a local ServingKernel (+ optional TenantContext)."""
    active_units: list[dict[str, Any]] = []
    if tenant is not None:
        for uid in tenant.router.registered_units:
            m = tenant.registry.get_manifest(uid)
            active_units.append(
                {
                    "unit_id": m.unit_id,
                    "version": m.version,
                    "sha256": m.payload_sha256[:16],
                    "payload_bytes": m.payload_bytes,
                    "bits_per_weight": round(m.effective_bits_per_weight, 4),
                }
            )

    results: list[CaseResult] = []
    for case in suite.cases:
        t0 = time.perf_counter()
        res = kernel.generate_routed_chat(
            user_prompt=case.prompt,
            system_prompt=case.system_prompt,
            router=None if tenant is None else tenant.router,
            registry=None if tenant is None else tenant.registry,
            max_tokens=case.max_tokens,
            enable_thinking=case.enable_thinking,
            top_k=top_k,
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        passed = grade_response(case, res.text, hit_token_limit=res.hit_token_limit)
        results.append(
            CaseResult(
                case_id=case.case_id,
                axis=case.axis,
                passed=passed,
                response=res.text,
                completion_tokens=res.completion_tokens,
                hit_token_limit=res.hit_token_limit,
                repetition_ratio=res.repetition_ratio,
                latency_ms=elapsed_ms,
                routed_units=res.routing.selected_units,
            )
        )

    return _summarize_snapshot(
        label=label,
        suite=suite,
        model_id=kernel.model_id,
        tenant_id=None if tenant is None else tenant.tenant_id,
        active_units=active_units,
        case_results=results,
    )


def run_benchmark_http(
    *,
    suite: BenchmarkSuite,
    base_url: str,
    model: str,
    label: str,
    api_key: str | None = None,
    timeout: float = 60.0,
) -> BenchmarkSnapshot:
    """Run a BenchmarkSuite against any OpenAI-compatible HTTP endpoint (e.g. `subcor serve`)."""
    root = base_url.rstrip("/")
    headers: dict[str, str] = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    tenant_id: str | None = None
    active_units: list[dict[str, Any]] = []
    if model.startswith("subcor/") and model != "subcor/base":
        tenant_id = model.split("/", 1)[1]
        # Query SUBCOR control plane if present to record exact active Skill Units.
        units_url = f"{root}/subcor/tenants/{tenant_id}/units"
        try:
            req_u = urllib.request.Request(units_url, headers=headers, method="GET")
            with urllib.request.urlopen(req_u, timeout=10) as resp:
                udata = json.loads(resp.read().decode("utf-8"))
                active_set = set(udata.get("active_routed_units", []))
                for u in udata.get("units", []):
                    if u.get("unit_id") in active_set:
                        active_units.append(
                            {
                                "unit_id": u["unit_id"],
                                "version": u["version"],
                                "sha256": str(u["payload_sha256"])[:16],
                                "payload_bytes": int(u["payload_bytes"]),
                                "bits_per_weight": round(float(u["effective_bits_per_weight"]), 4),
                            }
                        )
        except Exception:
            pass

    results: list[CaseResult] = []
    chat_url = f"{root}/chat/completions"

    for case in suite.cases:
        messages: list[dict[str, str]] = []
        if case.system_prompt:
            messages.append({"role": "system", "content": case.system_prompt})
        messages.append({"role": "user", "content": case.prompt})

        payload = {
            "model": model,
            "messages": messages,
            "max_tokens": case.max_tokens,
            "temperature": 0.0,
            "enable_thinking": case.enable_thinking,
        }
        req = urllib.request.Request(
            chat_url,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        t0 = time.perf_counter()
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = json.loads(resp.read().decode("utf-8"))
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        choice = body["choices"][0]
        text = str(choice["message"]["content"])
        hit_limit = choice.get("finish_reason") == "length"
        usage = body.get("usage") or {}
        comp_tokens = int(usage.get("completion_tokens", max(1, len(text.split()))))
        subcor_meta = body.get("subcor") or {}
        routed = tuple(str(u) for u in subcor_meta.get("selected_units", []))

        passed = grade_response(case, text, hit_token_limit=hit_limit)
        results.append(
            CaseResult(
                case_id=case.case_id,
                axis=case.axis,
                passed=passed,
                response=text,
                completion_tokens=comp_tokens,
                hit_token_limit=hit_limit,
                repetition_ratio=0.0,
                latency_ms=elapsed_ms,
                routed_units=routed,
            )
        )

    return _summarize_snapshot(
        label=label,
        suite=suite,
        model_id=model,
        tenant_id=tenant_id,
        active_units=active_units,
        case_results=results,
    )


class BenchmarkLedger:
    """Append-only JSONL history documenting longitudinal functional growth."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._io_lock = threading.Lock()
        self.corrupted_line_count: int = 0

    def append(self, snapshot: BenchmarkSnapshot) -> None:
        line = json.dumps(snapshot.to_dict(), separators=(",", ":"), sort_keys=True)
        with self._io_lock, self.path.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")

    def load_all(self, *, suite_digest: str | None = None) -> list[BenchmarkSnapshot]:
        if not self.path.exists():
            return []
        out: list[BenchmarkSnapshot] = []
        corrupted = 0
        with self._io_lock, self.path.open("r", encoding="utf-8") as fh:
            for lineno, raw in enumerate(fh, start=1):
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    snap = BenchmarkSnapshot.from_dict(json.loads(raw))
                except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
                    corrupted += 1
                    warnings.warn(
                        f"Skipping corrupted benchmark ledger line {lineno} in {self.path}: {exc}",
                        RuntimeWarning,
                        stacklevel=2,
                    )
                    continue
                if suite_digest is None or snap.suite_digest == suite_digest:
                    out.append(snap)
        self.corrupted_line_count = corrupted
        return out


@dataclass(frozen=True)
class GrowthDelta:
    """Comparison between a baseline BenchmarkSnapshot and a later snapshot."""

    baseline_label: str
    current_label: str
    suite_digest_match: bool
    overall_delta: float
    macro_axis_delta: float
    anchor_delta: float
    anchor_regressed: bool
    axis_deltas: dict[str, float]
    unit_count_delta: int
    unit_bytes_delta: int
    latency_ms_delta: float
    regressed_anchor_axes: tuple[str, ...] = ()

    @property
    def ci_passed(self) -> bool:
        """True only when suite digests match AND no pooled or individual anchor axis regressed."""
        return self.suite_digest_match and not self.anchor_regressed

    def to_dict(self) -> dict[str, Any]:
        return {
            "baseline_label": self.baseline_label,
            "current_label": self.current_label,
            "suite_digest_match": self.suite_digest_match,
            "ci_passed": self.ci_passed,
            "overall_delta": round(self.overall_delta, 6),
            "macro_axis_delta": round(self.macro_axis_delta, 6),
            "anchor_delta": round(self.anchor_delta, 6),
            "anchor_regressed": self.anchor_regressed,
            "regressed_anchor_axes": list(self.regressed_anchor_axes),
            "axis_deltas": {k: round(v, 6) for k, v in self.axis_deltas.items()},
            "unit_count_delta": self.unit_count_delta,
            "unit_bytes_delta": self.unit_bytes_delta,
            "latency_ms_delta": round(self.latency_ms_delta, 2),
        }


def compare_snapshots(
    baseline: BenchmarkSnapshot,
    current: BenchmarkSnapshot,
    *,
    anchor_regression_tolerance: float = 0.02,
    anchor_axes: frozenset[str] = LOCKED_ANCHOR_AXES,
) -> GrowthDelta:
    """Compute exact functional growth & per-axis anchor non-regression deltas."""
    axes = list(baseline.axis_summaries.keys())
    for ax in current.axis_summaries:
        if ax not in axes:
            axes.append(ax)

    axis_deltas: dict[str, float] = {}
    for ax in axes:
        b_rate = baseline.axis_summaries[ax].pass_rate if ax in baseline.axis_summaries else 0.0
        c_rate = current.axis_summaries[ax].pass_rate if ax in current.axis_summaries else 0.0
        axis_deltas[ax] = c_rate - b_rate

    regressed_axes = tuple(
        ax for ax in axes if ax in anchor_axes and axis_deltas[ax] < -anchor_regression_tolerance
    )
    anchor_delta = current.anchor_pass_rate - baseline.anchor_pass_rate
    suite_match = baseline.suite_digest == current.suite_digest
    return GrowthDelta(
        baseline_label=baseline.label,
        current_label=current.label,
        suite_digest_match=suite_match,
        overall_delta=current.overall_pass_rate - baseline.overall_pass_rate,
        macro_axis_delta=current.macro_axis_pass_rate - baseline.macro_axis_pass_rate,
        anchor_delta=anchor_delta,
        anchor_regressed=(anchor_delta < -anchor_regression_tolerance) or bool(regressed_axes),
        regressed_anchor_axes=regressed_axes,
        axis_deltas=axis_deltas,
        unit_count_delta=len(current.active_units) - len(baseline.active_units),
        unit_bytes_delta=current.total_unit_bytes - baseline.total_unit_bytes,
        latency_ms_delta=current.mean_latency_ms - baseline.mean_latency_ms,
    )


def render_markdown_report(snapshots: list[BenchmarkSnapshot]) -> str:
    """Render a longitudinal Markdown Functional Growth Report from ordered snapshots."""
    if not snapshots:
        return "# SUBCOR Functional Growth Report\n\nNo benchmark snapshots recorded yet.\n"

    first = snapshots[0]
    latest = snapshots[-1]
    delta = compare_snapshots(first, latest)
    axes = list(first.axis_summaries.keys())
    for snap in snapshots:
        for ax in snap.axis_summaries:
            if ax not in axes:
                axes.append(ax)

    if not delta.suite_digest_match:
        anchor_verdict = "INVALID (SUITE DIGEST MISMATCH)"
    elif delta.anchor_regressed:
        reg_str = ", ".join(delta.regressed_anchor_axes) or "pooled"
        anchor_verdict = f"REGRESSED ({reg_str})"
    else:
        anchor_verdict = "PRESERVED"

    lines: list[str] = [
        "# SUBCOR Functional Growth Report",
        "",
        f"- **Suite:** `{first.suite_id}@{first.suite_version}` (`sha256:{first.suite_digest[:12]}`)",
        f"- **Model:** `{latest.model_id}`"
        + (f" (tenant: `{latest.tenant_id}`)" if latest.tenant_id else ""),
        f"- **Snapshots Recorded:** {len(snapshots)} (`{first.label}` → `{latest.label}`)",
        f"- **Net Overall Growth:** `{first.overall_pass_rate * 100:.1f}% → {latest.overall_pass_rate * 100:.1f}%` "
        f"(`{delta.overall_delta * 100:+.1f} pp`)",
        f"- **Locked Anchor Retention:** `{first.anchor_pass_rate * 100:.1f}% → {latest.anchor_pass_rate * 100:.1f}%` "
        f"({anchor_verdict})",
        f"- **Active Skill Footprint:** {len(latest.active_units)} units, "
        f"{latest.total_unit_bytes / 1024:.1f} KiB "
        f"({latest.mean_bits_per_weight:.4f} avg bits/weight)",
        "",
        "## Longitudinal Progression",
        "",
        "| Snapshot | Active Units | Skill Size | Overall | Macro Axis | Anchors | Loop Fail | Avg Latency | Tok/s |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for s in snapshots:
        lines.append(
            f"| `{s.label}` | {len(s.active_units)} | {s.total_unit_bytes / 1024:.1f} KiB | "
            f"{s.overall_pass_rate * 100:.1f}% | {s.macro_axis_pass_rate * 100:.1f}% | "
            f"{s.anchor_pass_rate * 100:.1f}% | {s.token_limit_rate * 100:.1f}% | "
            f"{s.mean_latency_ms:.0f} ms | {s.tokens_per_second:.1f} |"
        )

    lines.extend(["", "## Per-Axis Functional Growth", ""])
    header = "| Axis | " + " | ".join(f"`{s.label}`" for s in snapshots) + " | Δ (First → Latest) |"
    sep = "|---|" + "|".join("---:" for _ in snapshots) + "|---:|"
    lines.extend([header, sep])

    for ax in axes:
        cells = []
        for s in snapshots:
            rate = s.axis_summaries[ax].pass_rate if ax in s.axis_summaries else 0.0
            cells.append(f"{rate * 100:.1f}%")
        d_ax = delta.axis_deltas.get(ax, 0.0)
        tag = " *(anchor)*" if ax in LOCKED_ANCHOR_AXES else ""
        lines.append(f"| `{ax}`{tag} | " + " | ".join(cells) + f" | **{d_ax * 100:+.1f} pp** |")

    if latest.active_units:
        lines.extend(
            [
                "",
                "## Active Skill Units in Latest Snapshot",
                "",
                "| Unit ID | Version | Bits/Weight | Payload Size | SHA-256 |",
                "|---|---|---:|---:|---|",
            ]
        )
        for u in latest.active_units:
            lines.append(
                f"| `{u['unit_id']}` | `{u['version']}` | {u['bits_per_weight']:.4f} | "
                f"{u['payload_bytes'] / 1024:.1f} KiB | `{u['sha256']}` |"
            )

    lines.append("")
    return "\n".join(lines)
