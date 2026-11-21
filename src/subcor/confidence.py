"""Cortical Handoff (Thalamic Uncertainty Switch) and OpenAI tool-call parsing.

Implements two runtime primitives for the Cortical-Subcortical architecture:

1. TurnConfidence / ConfidencePolicy:
   Scores each local SUBCOR turn using both RouterGate margin (how squarely
   inside a habitual Skill Unit the request sits) and decode-time token
   log-probabilities (cerebellar certainty). Emits `escalate_to_cortex` when
   confidence drops, a thinking loop is detected, or a strict-habit seat
   encounters an unmapped routine.

2. parse_tool_calls:
   Parses native model tool invocations (both Qwen3.5 XML `<tool_call><function=...>
   <parameter=...>...</function></tool_call>` and JSON `<tool_call>{"name":...}</tool_call>`)
   into standard OpenAI `tool_calls` objects.
"""

from __future__ import annotations

import json
import math
import re
import uuid
from dataclasses import dataclass
from typing import Any

from subcor.router import RoutingDecision


@dataclass(frozen=True)
class ConfidencePolicy:
    """Thresholds governing when SUBCOR hands control back to the Prefrontal Cortex."""

    min_mean_token_prob: float = 0.55
    min_worst_token_prob: float = 0.05
    max_repetition_ratio: float = 0.45
    require_skill_match: bool = False


@dataclass(frozen=True)
class TurnConfidence:
    """Observable confidence and cortical-escalation telemetry for one turn."""

    mean_token_prob: float
    min_token_prob: float
    mean_nll: float
    router_margin: float | None
    matched_skill_unit: bool
    escalate_to_cortex: bool
    escalation_reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "mean_token_prob": round(self.mean_token_prob, 4),
            "min_token_prob": round(self.min_token_prob, 4),
            "mean_nll": round(self.mean_nll, 4),
            "router_margin": None if self.router_margin is None else round(self.router_margin, 5),
            "matched_skill_unit": self.matched_skill_unit,
            "escalate_to_cortex": self.escalate_to_cortex,
            "escalation_reason": self.escalation_reason,
        }


def evaluate_turn_confidence(
    *,
    token_logprobs: list[float],
    routing: RoutingDecision,
    hit_token_limit: bool,
    repetition_ratio: float,
    policy: ConfidencePolicy | None = None,
) -> TurnConfidence:
    """Compute geometric-mean token confidence and Cortical Handoff decision."""
    pol = policy or ConfidencePolicy()

    if token_logprobs:
        # Clamp extreme logprobs to avoid exp underflow/overflow.
        clamped = [max(-30.0, min(0.0, float(lp))) for lp in token_logprobs]
        mean_lp = sum(clamped) / len(clamped)
        min_lp = min(clamped)
        mean_prob = math.exp(mean_lp)
        min_prob = math.exp(min_lp)
        mean_nll = -mean_lp
    else:
        mean_prob = 1.0
        min_prob = 1.0
        mean_nll = 0.0

    matched = not routing.fell_back_to_base
    margin: float | None = None
    if matched:
        top_uid = routing.selected_units[0]
        if top_uid in routing.scores and top_uid in routing.thresholds:
            margin = routing.scores[top_uid] - routing.thresholds[top_uid]

    if hit_token_limit:
        escalate = True
        reason = "token_limit_exhausted"
    elif repetition_ratio > pol.max_repetition_ratio:
        escalate = True
        reason = "repetition_loop_detected"
    elif pol.require_skill_match and not matched:
        escalate = True
        reason = "no_matching_skill_unit"
    elif token_logprobs and mean_prob < pol.min_mean_token_prob:
        escalate = True
        reason = "low_decode_confidence"
    elif token_logprobs and min_prob < pol.min_worst_token_prob and mean_prob < 0.72:
        escalate = True
        reason = "localized_uncertainty_spike"
    else:
        escalate = False
        reason = "ok_habitual_execution" if matched else "ok_base_confident"

    return TurnConfidence(
        mean_token_prob=mean_prob,
        min_token_prob=min_prob,
        mean_nll=mean_nll,
        router_margin=margin,
        matched_skill_unit=matched,
        escalate_to_cortex=escalate,
        escalation_reason=reason,
    )


_TOOL_BLOCK_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", flags=re.DOTALL)
_XML_FUNC_RE = re.compile(r"<function=([^\s>]+)>\s*(.*?)\s*</function>", flags=re.DOTALL)
_XML_PARAM_RE = re.compile(r"<parameter=([^\s>]+)>\s*(.*?)\s*</parameter>", flags=re.DOTALL)


def _coerce_param_value(raw: str) -> Any:
    val = raw.strip()
    if not val:
        return ""
    if val in ("true", "false", "null") or (
        (val.startswith("{") and val.endswith("}")) or (val.startswith("[") and val.endswith("]"))
    ):
        try:
            return json.loads(val)
        except json.JSONDecodeError:
            return val
    try:
        if "." in val:
            return float(val)
        return int(val)
    except ValueError:
        return val


def parse_tool_calls(text: str) -> tuple[str, list[dict[str, Any]]]:
    """Extract OpenAI-formatted tool_calls from raw model output.

    Supports both:
      - Qwen3.5 XML tags: <tool_call><function=fn><parameter=k>v</parameter></function></tool_call>
      - JSON blocks:      <tool_call>{"name": "fn", "arguments": {"k": "v"}}</tool_call>

    Returns (remaining_clean_content, openai_tool_calls).
    """
    calls: list[dict[str, Any]] = []
    spans_to_remove: list[tuple[int, int]] = []

    for match in _TOOL_BLOCK_RE.finditer(text):
        inner = match.group(1).strip()
        parsed_call: tuple[str, dict[str, Any]] | None = None

        # 1. Try JSON format first.
        if inner.startswith("{"):
            try:
                obj = json.loads(inner)
                if isinstance(obj, dict) and "name" in obj:
                    raw_args = obj.get("arguments", {})
                    if isinstance(raw_args, str):
                        try:
                            raw_args = json.loads(raw_args)
                        except json.JSONDecodeError:
                            raw_args = {"raw": raw_args}
                    if not isinstance(raw_args, dict):
                        raw_args = {"value": raw_args}
                    parsed_call = (str(obj["name"]), raw_args)
            except json.JSONDecodeError:
                parsed_call = None

        # 2. Try Qwen3.5 XML function/parameter format.
        if parsed_call is None:
            fn_match = _XML_FUNC_RE.search(inner)
            if fn_match:
                fn_name = fn_match.group(1).strip()
                fn_body = fn_match.group(2)
                args: dict[str, Any] = {}
                for pm in _XML_PARAM_RE.finditer(fn_body):
                    args[pm.group(1).strip()] = _coerce_param_value(pm.group(2))
                parsed_call = (fn_name, args)

        if parsed_call is not None:
            fn_name, fn_args = parsed_call
            calls.append(
                {
                    "id": f"call_{uuid.uuid4().hex[:12]}",
                    "type": "function",
                    "function": {
                        "name": fn_name,
                        "arguments": json.dumps(fn_args, separators=(",", ":")),
                    },
                }
            )
            spans_to_remove.append(match.span())

    if not spans_to_remove:
        return text, []

    pieces: list[str] = []
    cursor = 0
    for start, end in spans_to_remove:
        pieces.append(text[cursor:start])
        cursor = end
    pieces.append(text[cursor:])
    remaining = "".join(pieces).strip()
    return remaining, calls
