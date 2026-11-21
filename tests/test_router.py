"""Unit tests for the Activation-Space MoE Skill Router and CaPheOS gate export."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from slaingi.expansion import LowRankAdapter
from slaingi.router import MoESkillRouter, RouterGate
from slaingi.types import AdaptationMode, DeltaEncoding
from slaingi.units import SkillUnitRegistry, build_layer_delta


def test_moe_router_routes_two_skills_and_falls_back_to_base_on_general() -> None:
    rng = np.random.default_rng(7)
    dim = 64
    q, _ = np.linalg.qr(rng.normal(size=(dim, dim)))
    c_ops = q[:, 0]
    c_noc = q[:, 1]
    c_gen = q[:, 2]

    ops_train = c_ops + rng.normal(0.0, 0.12, size=(15, dim))
    noc_train = c_noc + rng.normal(0.0, 0.12, size=(15, dim))
    gen_train = c_gen + rng.normal(0.0, 0.12, size=(15, dim))

    gate_ops = RouterGate.fit(
        unit_id="ops-router-v1",
        tap_layer="layers.18.mlp.down_proj",
        pos_embeddings=ops_train,
        neg_embeddings=np.vstack([gen_train, noc_train]),
    )
    gate_noc = RouterGate.fit(
        unit_id="incident-sla-v1",
        tap_layer="layers.18.mlp.down_proj",
        pos_embeddings=noc_train,
        neg_embeddings=np.vstack([gen_train, ops_train]),
    )

    router = MoESkillRouter(tap_layer="layers.18.mlp.down_proj")
    router.register(gate_ops)
    router.register(gate_noc)

    # Unseen test embeddings
    ops_test = c_ops + rng.normal(0.0, 0.15, size=(10, dim))
    noc_test = c_noc + rng.normal(0.0, 0.15, size=(10, dim))
    gen_test = c_gen + rng.normal(0.0, 0.15, size=(10, dim))

    assert all(router.route(v, top_k=1).selected_units == ("ops-router-v1",) for v in ops_test)
    assert all(router.route(v, top_k=1).selected_units == ("incident-sla-v1",) for v in noc_test)
    assert all(router.route(v, top_k=1).fell_back_to_base for v in gen_test)

    # Unregistering a rolled-back unit immediately routes its queries back to base.
    router.unregister("incident-sla-v1")
    assert all(router.route(v, top_k=1).fell_back_to_base for v in noc_test)


def test_router_gate_persists_in_registry_and_exports_in_capheos_bundle(tmp_path: Path) -> None:
    registry = SkillUnitRegistry(tmp_path / "units")
    rng = np.random.default_rng(13)
    adapter = LowRankAdapter(
        name="a",
        a=rng.normal(0.0, 0.3, size=(2, 64)),
        b=rng.normal(0.0, 0.3, size=(64, 2)),
    )
    delta = build_layer_delta(
        "layers.18.mlp.down_proj",
        adapter,
        encoding=DeltaEncoding.SUB1BIT_BINARY,
        sub1bit_rank=2,
    )
    registry.propose(
        unit_id="routed-ops-v1",
        version="1.0.0",
        title="Routed Ops",
        summary="Test routed skill unit",
        proposer="tools",
        mode=AdaptationMode.DYNAMIC_EXPANSION_FP32,
        deltas=[delta],
    )
    gate = RouterGate.fit(
        unit_id="routed-ops-v1",
        tap_layer="layers.18.mlp.down_proj",
        pos_embeddings=rng.normal(1.0, 0.1, size=(6, 16)),
        neg_embeddings=rng.normal(-1.0, 0.1, size=(6, 16)),
    )
    registry.save_router_gate(gate)
    reloaded = registry.load_router_gate("routed-ops-v1")
    assert reloaded is not None
    assert reloaded.unit_id == "routed-ops-v1"
    np.testing.assert_allclose(reloaded.pos_centroid, gate.pos_centroid, atol=1e-5)

    registry.record_conformance(
        "routed-ops-v1",
        baseline_before=0.85,
        baseline_after=0.85,
        target_before=0.0,
        target_after=0.90,
    )
    registry.promote("routed-ops-v1", approver="risk")

    bundle = registry.export_capheos_unit("routed-ops-v1", tmp_path / "capheos")
    manifest = json.loads((bundle / "module.json").read_text(encoding="utf-8"))
    assert manifest["slaingi"]["routerGateFile"] == "router_gate.json"
    assert (bundle / "router_gate.json").exists()
