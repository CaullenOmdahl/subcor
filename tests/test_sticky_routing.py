"""Sticky per-conversation routing: the policy and its use in the llama.cpp kernel."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from subcor.backends.llamacpp_kernel import LlamaCppServingAdapter
from subcor.router import RoutingDecision, StickyRoutingPolicy

LONG = 10_000  # context tokens well above min_context_tokens


def _decision(margins: dict[str, float], top_k: int = 1) -> RoutingDecision:
    """Decision whose units have the given margins above a 0.5 threshold."""
    scores = {uid: 0.5 + m for uid, m in margins.items()}
    passing = sorted((m, uid) for uid, m in margins.items() if m >= 0)[::-1]
    return RoutingDecision(
        selected_units=tuple(uid for _, uid in passing[:top_k]),
        scores=scores,
        thresholds=dict.fromkeys(margins, 0.5),
    )


POLICY = StickyRoutingPolicy(switch_margin=0.05, min_context_tokens=2048)


def test_first_turn_and_unchanged_decisions_pass_through() -> None:
    fresh = _decision({"a": 0.2, "b": 0.1})
    assert POLICY.apply(None, fresh, context_tokens=LONG) is fresh
    assert POLICY.apply(("a",), fresh, context_tokens=LONG) is fresh


def test_keeps_previous_unit_when_new_one_wins_by_less_than_the_margin() -> None:
    out = POLICY.apply(("b",), _decision({"a": 0.12, "b": 0.1}), context_tokens=LONG)
    assert out.selected_units == ("b",) and out.sticky


def test_switches_when_new_unit_wins_by_more_than_the_margin() -> None:
    out = POLICY.apply(("b",), _decision({"a": 0.3, "b": 0.1}), context_tokens=LONG)
    assert out.selected_units == ("a",) and not out.sticky


def test_switches_when_previous_unit_no_longer_clears_its_threshold() -> None:
    out = POLICY.apply(("b",), _decision({"a": 0.01, "b": -0.01}), context_tokens=LONG)
    assert out.selected_units == ("a",) and not out.sticky
    to_base = POLICY.apply(("b",), _decision({"a": -0.2, "b": -0.01}), context_tokens=LONG)
    assert to_base.selected_units == () and not to_base.sticky


def test_switches_when_previous_unit_is_no_longer_routable() -> None:
    out = POLICY.apply(("gone",), _decision({"a": 0.01}), context_tokens=LONG)
    assert out.selected_units == ("a",)


def test_base_counts_as_margin_zero() -> None:
    weak = POLICY.apply((), _decision({"a": 0.03}), context_tokens=LONG)
    assert weak.selected_units == () and weak.sticky
    strong = POLICY.apply((), _decision({"a": 0.2}), context_tokens=LONG)
    assert strong.selected_units == ("a",)


def test_short_context_routes_fresh() -> None:
    out = POLICY.apply(("b",), _decision({"a": 0.12, "b": 0.1}), context_tokens=100)
    assert out.selected_units == ("a",) and not out.sticky


def test_multi_unit_previous_uses_its_weakest_margin() -> None:
    fresh = _decision({"a": 0.2, "b": 0.19, "c": 0.12}, top_k=2)
    assert POLICY.apply(("b", "c"), fresh, context_tokens=LONG).selected_units == ("a", "b")
    close = _decision({"a": 0.2, "b": 0.19, "c": 0.17}, top_k=2)
    assert POLICY.apply(("b", "c"), close, context_tokens=LONG).selected_units == ("b", "c")


@pytest.mark.parametrize("kwargs", [{"switch_margin": -0.1}, {"min_context_tokens": -1}])
def test_rejects_negative_settings(kwargs: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        StickyRoutingPolicy(**kwargs)


# ---------------------------------------------------------------- kernel integration
class _ScriptedRouter:
    """Router stub that returns queued decisions in order."""

    registered_units = ("a", "b")

    def __init__(self, decisions: list[RoutingDecision]) -> None:
        self.decisions = list(decisions)

    def route(self, emb: Any, *, top_k: int, seat: str | None) -> RoutingDecision:
        return self.decisions.pop(0)


class _Registry:
    def __init__(self) -> None:
        self.verified: list[str] = []

    def load_verified_deltas(self, uid: str) -> None:
        self.verified.append(uid)


def _kernel(policy: StickyRoutingPolicy | None) -> LlamaCppServingAdapter:
    kernel = LlamaCppServingAdapter(base_url="http://127.0.0.1:9", sticky_routing=policy)
    kernel.encode_prompt_embedding = lambda text: np.ones(4)  # type: ignore[method-assign]
    return kernel


def _conversation(opening: str, turns: int) -> list[dict[str, Any]]:
    msgs: list[dict[str, Any]] = [{"role": "system", "content": "sys " * 3000}]
    msgs.append({"role": "user", "content": opening})
    for i in range(turns):
        msgs += [{"role": "assistant", "content": f"r{i}"}, {"role": "user", "content": f"q{i}"}]
    return msgs


def test_kernel_keeps_a_conversations_unit_across_turns() -> None:
    close = _decision({"a": 0.12, "b": 0.1})
    router = _ScriptedRouter([_decision({"a": -0.1, "b": 0.1}), close, close])
    registry = _Registry()
    kernel = _kernel(POLICY)
    first = kernel._route(_conversation("hello", 0), router, registry, 1, None)  # type: ignore[arg-type]
    second = kernel._route(_conversation("hello", 1), router, registry, 1, None)  # type: ignore[arg-type]
    assert first.selected_units == ("b",)
    assert second.selected_units == ("b",) and second.sticky
    assert registry.verified == ["b", "b"]  # integrity is checked on the units that run
    # A different conversation (other opening) has no history, so it routes fresh.
    other = kernel._route(_conversation("other", 1), router, registry, 1, None)  # type: ignore[arg-type]
    assert other.selected_units == ("a",) and not other.sticky


def test_kernel_without_policy_routes_fresh_every_turn() -> None:
    close = _decision({"a": 0.12, "b": 0.1})
    router = _ScriptedRouter([_decision({"a": -0.1, "b": 0.1}), close])
    kernel = _kernel(None)
    kernel._route(_conversation("hello", 0), router, _Registry(), 1, None)  # type: ignore[arg-type]
    second = kernel._route(_conversation("hello", 1), router, _Registry(), 1, None)  # type: ignore[arg-type]
    assert second.selected_units == ("a",) and not second.sticky


def test_kernel_conversation_memory_is_bounded(monkeypatch: pytest.MonkeyPatch) -> None:
    import subcor.backends.llamacpp_kernel as mod

    monkeypatch.setattr(mod, "_STICKY_MAX_CONVERSATIONS", 3)
    decisions = [_decision({"a": 0.1, "b": -0.1}) for _ in range(5)]
    kernel = _kernel(POLICY)
    router = _ScriptedRouter(decisions)
    for i in range(5):
        kernel._route(_conversation(f"c{i}", 0), router, _Registry(), 1, None)  # type: ignore[arg-type]
    assert len(kernel._sticky_units) == 3
