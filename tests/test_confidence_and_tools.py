"""Tests for Cortical Handoff (TurnConfidence) and OpenAI tool-call parsing."""

from __future__ import annotations

import json
import math

from subcor.confidence import (
    ConfidencePolicy,
    evaluate_turn_confidence,
    parse_tool_calls,
)
from subcor.router import RoutingDecision


def test_parse_qwen_xml_tool_calls_with_type_coercion() -> None:
    raw = (
        "Let me look that up.\n"
        "<tool_call>\n"
        "<function=get_invoice>\n"
        "<parameter=invoice_id>INV-42</parameter>\n"
        "<parameter=limit>10</parameter>\n"
        "<parameter=include_tax>true</parameter>\n"
        "<parameter=python_bool>True</parameter>\n"
        "<parameter=python_false>False</parameter>\n"
        "<parameter=threshold>99.5</parameter>\n"
        '<parameter=tags>["urgent","finance"]</parameter>\n'
        "</function>\n"
        "</tool_call>"
    )
    clean, calls = parse_tool_calls(raw)
    assert clean == "Let me look that up."
    assert len(calls) == 1
    call = calls[0]
    assert call["id"].startswith("call_")
    assert call["type"] == "function"
    assert call["function"]["name"] == "get_invoice"
    args = json.loads(call["function"]["arguments"])
    assert args == {
        "invoice_id": "INV-42",
        "limit": 10,
        "include_tax": True,
        "python_bool": True,
        "python_false": False,
        "threshold": 99.5,
        "tags": ["urgent", "finance"],
    }


def test_parse_json_tool_calls_and_multiple_blocks() -> None:
    raw = (
        '<tool_call>{"name": "lookup_policy", "arguments": {"dept": "finance"}}</tool_call>\n'
        '<tool_call>{"name": "notify_owner", "arguments": "{\\"urgent\\": true}"}</tool_call>'
    )
    clean, calls = parse_tool_calls(raw)
    assert clean == ""
    assert len(calls) == 2
    assert calls[0]["function"]["name"] == "lookup_policy"
    assert json.loads(calls[0]["function"]["arguments"]) == {"dept": "finance"}
    assert calls[1]["function"]["name"] == "notify_owner"
    assert json.loads(calls[1]["function"]["arguments"]) == {"urgent": True}


def test_parse_tool_calls_passthrough_plain_text() -> None:
    text = '{"seat":"finance","gate":"owner"}'
    clean, calls = parse_tool_calls(text)
    assert clean == text
    assert calls == []


def test_evaluate_turn_confidence_habitual_and_base_ok() -> None:
    routed = RoutingDecision(
        selected_units=("ops-skill-v1",),
        scores={"ops-skill-v1": 0.91},
        thresholds={"ops-skill-v1": 0.75},
    )
    conf = evaluate_turn_confidence(
        token_logprobs=[math.log(0.95), math.log(0.92), math.log(0.88)],
        routing=routed,
        hit_token_limit=False,
        repetition_ratio=0.0,
    )
    assert conf.matched_skill_unit is True
    assert conf.escalate_to_cortex is False
    assert conf.escalation_reason == "ok_habitual_execution"
    assert conf.router_margin is not None
    assert abs(conf.router_margin - 0.16) < 1e-6
    assert conf.mean_token_prob > 0.90

    base = RoutingDecision(
        selected_units=(), scores={"ops-skill-v1": 0.2}, thresholds={"ops-skill-v1": 0.75}
    )
    base_conf = evaluate_turn_confidence(
        token_logprobs=[math.log(0.90), math.log(0.85)],
        routing=base,
        hit_token_limit=False,
        repetition_ratio=0.0,
    )
    assert base_conf.matched_skill_unit is False
    assert base_conf.escalate_to_cortex is False
    assert base_conf.escalation_reason == "ok_base_confident"
    assert base_conf.router_margin is None


def test_evaluate_turn_confidence_escalation_triggers() -> None:
    routed = RoutingDecision(
        selected_units=("ops-skill-v1",),
        scores={"ops-skill-v1": 0.85},
        thresholds={"ops-skill-v1": 0.75},
    )
    base = RoutingDecision(selected_units=(), scores={}, thresholds={})

    # 1. Low mean decode probability -> escalate
    low_prob = evaluate_turn_confidence(
        token_logprobs=[math.log(0.30), math.log(0.25), math.log(0.40)],
        routing=routed,
        hit_token_limit=False,
        repetition_ratio=0.0,
    )
    assert low_prob.escalate_to_cortex is True
    assert low_prob.escalation_reason == "low_decode_confidence"

    # 2. Localized uncertainty spike (0.55 <= mean < 0.72 and min < 0.05) -> escalate
    spike = evaluate_turn_confidence(
        token_logprobs=[
            math.log(0.96),
            math.log(0.96),
            math.log(0.96),
            math.log(0.96),
            math.log(0.96),
            math.log(0.96),
            math.log(0.02),
        ],
        routing=routed,
        hit_token_limit=False,
        repetition_ratio=0.0,
    )
    assert 0.55 <= spike.mean_token_prob < 0.72
    assert spike.min_token_prob < 0.05
    assert spike.escalate_to_cortex is True
    assert spike.escalation_reason == "localized_uncertainty_spike"

    # 3. Token limit exhausted -> escalate
    limit = evaluate_turn_confidence(
        token_logprobs=[math.log(0.95)],
        routing=routed,
        hit_token_limit=True,
        repetition_ratio=0.0,
    )
    assert limit.escalate_to_cortex is True
    assert limit.escalation_reason == "token_limit_exhausted"

    # 4. Repetition loop -> escalate
    rep = evaluate_turn_confidence(
        token_logprobs=[math.log(0.95)],
        routing=routed,
        hit_token_limit=False,
        repetition_ratio=0.60,
    )
    assert rep.escalate_to_cortex is True
    assert rep.escalation_reason == "repetition_loop_detected"

    # 5. Strict habitual seat (require_skill_match=True) with no matching skill -> escalate
    strict = evaluate_turn_confidence(
        token_logprobs=[math.log(0.95)],
        routing=base,
        hit_token_limit=False,
        repetition_ratio=0.0,
        policy=ConfidencePolicy(require_skill_match=True),
    )
    assert strict.escalate_to_cortex is True
    assert strict.escalation_reason == "no_matching_skill_unit"
