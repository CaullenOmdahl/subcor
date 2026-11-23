"""Issue #33: proposers cannot weaken the conformance gate; non-finite values are rejected."""

from __future__ import annotations

import math
import threading
from pathlib import Path
from typing import Any

import pytest

from subcor.server import (
    DEFAULT_BASELINE_ANCHOR_CASES,
    ConformancePolicy,
    SubcorHTTPServer,
    TenantManager,
    build_dummy_delta,
)
from subcor.types import AdaptationMode, UnitState
from subcor.units import SkillUnitRegistry
from test_p0_correctness_and_eval import _MinimalKernel
from test_server import _FakeMultiSkillKernel, _http_json

# Scores from the issue: baseline 0.9 -> 0.1, target 0.8 -> 0.1.
BAD_SCORES = {
    "baseline_before": 0.9,
    "baseline_after": 0.1,
    "target_before": 0.8,
    "target_after": 0.1,
}


@pytest.fixture
def registry(tmp_path: Path) -> SkillUnitRegistry:
    reg = SkillUnitRegistry(tmp_path / "units")
    reg.propose(
        unit_id="u1",
        version="1.0.0",
        title="U1",
        summary="Test",
        proposer="tools",
        mode=AdaptationMode.DYNAMIC_EXPANSION_FP32,
        deltas=[build_dummy_delta("layers.18.mlp.down_proj", rank=2)],
    )
    return reg


def test_default_thresholds_reject_a_regressing_unit(registry: SkillUnitRegistry) -> None:
    manifest = registry.record_conformance("u1", **BAD_SCORES)
    assert manifest.state is UnitState.DORMANT


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
@pytest.mark.parametrize(
    "field",
    [
        "baseline_before",
        "baseline_after",
        "target_before",
        "target_after",
        "max_regression_tolerance",
        "min_target_improvement",
    ],
)
def test_registry_rejects_non_finite_values(
    registry: SkillUnitRegistry, field: str, bad: float
) -> None:
    kwargs: dict[str, Any] = {**BAD_SCORES, field: bad}
    with pytest.raises(ValueError, match="finite"):
        registry.record_conformance("u1", **kwargs)
    assert registry.get_manifest("u1").conformance is None


@pytest.mark.parametrize(
    "overrides",
    [
        {"baseline_before": 1.5},
        {"target_after": -0.1},
        {"max_regression_tolerance": -0.01},
        {"max_regression_tolerance": 1.5},
        {"min_target_improvement": -1.5},
    ],
)
def test_registry_rejects_out_of_range_values(
    registry: SkillUnitRegistry, overrides: dict[str, float]
) -> None:
    with pytest.raises(ValueError, match="must be in"):
        registry.record_conformance("u1", **{**BAD_SCORES, **overrides})


def _manager(tmp_path: Path, **kwargs: Any) -> TenantManager:
    mgr = TenantManager(
        state_dir=tmp_path / "st", kernel=_MinimalKernel(), initial_tenants=("t",), **kwargs
    )
    mgr.ingest_trajectories(
        "t",
        [
            {
                "task_family": "ops",
                "prompt": f"Event: Invoice #{i} over cap",
                "response": "bad",
                "senior_correction": '{"code":"FIN-CAP"}',
                "evidence": {"verified": True, "verifier": "senior"},
            }
            for i in range(4)
        ],
    )
    return mgr


def _study(mgr: TenantManager, **kwargs: Any) -> dict[str, Any]:
    return mgr.run_study_cycle(
        "t",
        unit_id="ops-v1",
        version="1.0.0",
        task_family="ops",
        title="Ops",
        summary="Ops",
        **kwargs,
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_regression_tolerance": 1.0},
        {"min_target_improvement": -1.0},
        {"min_target_improvement": 0.0},
        {"max_regression_tolerance": math.nan},
        {"min_target_improvement": math.nan},
        {"max_regression_tolerance": math.inf},
    ],
)
def test_study_cycle_rejects_looser_or_non_finite_thresholds(
    tmp_path: Path, kwargs: dict[str, float]
) -> None:
    with pytest.raises(ValueError):
        _study(_manager(tmp_path), **kwargs)


def test_study_cycle_accepts_stricter_thresholds_and_records_them(tmp_path: Path) -> None:
    res = _study(_manager(tmp_path), max_regression_tolerance=0.0, min_target_improvement=0.5)
    report = res["manifest"]["conformance"]
    assert (report["max_regression_tolerance"], report["min_target_improvement"]) == (0.0, 0.5)


def test_owner_policy_sets_the_defaults(tmp_path: Path) -> None:
    policy = ConformancePolicy(max_regression_tolerance=0.01, min_target_improvement=0.9)
    res = _study(_manager(tmp_path, conformance_policy=policy))
    report = res["manifest"]["conformance"]
    assert (report["max_regression_tolerance"], report["min_target_improvement"]) == (0.01, 0.9)
    with pytest.raises(ValueError, match="looser"):
        _study(_manager(tmp_path / "b", conformance_policy=policy), min_target_improvement=0.5)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_regression_tolerance": math.nan},
        {"max_regression_tolerance": 1.5},
        {"min_target_improvement": -0.1},
    ],
)
def test_policy_itself_validates_its_limits(kwargs: dict[str, float]) -> None:
    with pytest.raises(ValueError):
        ConformancePolicy(**kwargs)


class _AnchorBreakingKernel(_MinimalKernel):
    """A routed unit breaks the owner anchors but not the proposer's easy baseline."""

    def generate_routed_chat(self, **kwargs: Any) -> Any:
        res = super().generate_routed_chat(**kwargs)
        anchor_prompts = {p for p, _ in DEFAULT_BASELINE_ANCHOR_CASES}
        routed = kwargs["router"] is not None and bool(kwargs["router"].registered_units)
        if routed and kwargs["user_prompt"] in anchor_prompts:
            return type(res)(**{**res.__dict__, "text": "wrong"})
        return res


def test_owner_anchors_are_scored_even_with_proposer_baseline(tmp_path: Path) -> None:
    mgr = TenantManager(
        state_dir=tmp_path / "st", kernel=_AnchorBreakingKernel(), initial_tenants=("t",)
    )
    mgr.ingest_trajectories(
        "t",
        [
            {
                "task_family": "ops",
                "prompt": f"Event: Invoice #{i} over cap",
                "response": "bad",
                "senior_correction": '{"code":"FIN-CAP"}',
                "evidence": {"verified": True, "verifier": "senior"},
            }
            for i in range(4)
        ],
    )
    # The proposer's own baseline never matches the unit's routing, so it never regresses.
    easy = [{"prompt": "Say the word base.", "expected": "base"}]
    res = _study(mgr, baseline_cases=easy)
    report = res["manifest"]["conformance"]
    assert res["manifest"]["state"] == "dormant"
    assert "Baseline regression" in report["reason"]


def test_http_study_rejects_looser_thresholds(tmp_path: Path) -> None:
    manager = TenantManager(
        state_dir=tmp_path / "state", kernel=_FakeMultiSkillKernel(), initial_tenants=("a",)
    )
    server = SubcorHTTPServer(("127.0.0.1", 0), manager=manager, api_key="secret-key")
    host, port = server.server_address
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        code, body, _ = _http_json(
            f"http://{host}:{port}/v1/subcor/tenants/a/study",
            method="POST",
            body={
                "unit_id": "ops-v1",
                "task_family": "ops",
                "max_regression_tolerance": 1.0,
                "min_target_improvement": -1.0,
            },
        )
        assert code == 400
        assert "looser" in body["error"]
    finally:
        server.shutdown()
        server.server_close()
