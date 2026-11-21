"""Tests for portable SkillSpec persistence and `subcor recompile` kernel migration."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from subcor.expansion import LowRankAdapter
from subcor.recompile import recompile_units_for_kernel
from subcor.router import MoESkillRouter, RoutingDecision
from subcor.server import InferenceResult, TenantManager
from subcor.types import AdaptationMode, DeltaEncoding, UnitState
from subcor.units import LayerDeltaPayload, SkillUnitRegistry, build_layer_delta


def _seeded_delta(layer_name: str, seed: int) -> LayerDeltaPayload:
    rng = np.random.default_rng(seed)
    adapter = LowRankAdapter(
        name=layer_name,
        a=rng.normal(0.0, 0.2, size=(2, 32)),
        b=rng.normal(0.0, 0.2, size=(32, 2)),
    )
    return build_layer_delta(
        layer_name,
        adapter,
        encoding=DeltaEncoding.SUB1BIT_BINARY,
        sub1bit_rank=2,
    )


class _KernelStub:
    """Mock base kernel whose model_id and compiled delta bytes depend on its version."""

    tap_layer = "layers.18.mlp.down_proj"
    target_layers = ("layers.18.mlp.down_proj",)

    def __init__(self, model_id: str, delta_seed_offset: int) -> None:
        self.model_id = model_id
        self._seed_offset = delta_seed_offset
        self._dim = 16
        self._ops_vec = np.eye(self._dim, dtype=np.float64)[0]
        self._gen_vec = np.eye(self._dim, dtype=np.float64)[2]

    def encode_prompt_embedding(self, user_prompt: str) -> np.ndarray:
        low = user_prompt.lower()
        if "invoice" in low or "event:" in low:
            return self._ops_vec.copy()
        return self._gen_vec.copy()

    def generate_routed_chat(
        self,
        *,
        user_prompt: str,
        system_prompt: str | None,
        router: MoESkillRouter | None,
        registry: SkillUnitRegistry | None,
        max_tokens: int,
        enable_thinking: bool,
        top_k: int,
    ) -> InferenceResult:
        emb = self.encode_prompt_embedding(user_prompt)
        decision = (
            router.route(emb, top_k=top_k)
            if router is not None and router.registered_units
            else RoutingDecision(selected_units=(), scores={}, thresholds={})
        )
        if "ops-skill-v1" in decision.selected_units:
            text = '{"seat":"finance","gate":"owner","code":"FIN-CAP"}'
        elif "12 * 12" in user_prompt:
            text = "144"
        elif "japan" in user_prompt.lower():
            text = "Tokyo"
        else:
            text = "base-untrained"
        return InferenceResult(
            text=text,
            prompt_tokens=10,
            completion_tokens=6,
            hit_token_limit=False,
            repetition_ratio=0.0,
            routing=decision,
        )

    def train_skill_deltas(
        self,
        *,
        examples: list[tuple[str | None, str, str]],
        rank: int,
        sub1bit_rank: int,
        steps: int,
        lr: float,
        seed: int,
    ) -> tuple[list[LayerDeltaPayload], float, float, float]:
        return (
            [_seeded_delta(self.tap_layer, seed=seed + self._seed_offset)],
            1.4,
            0.04,
            0.0,
        )

    def invalidate_unit(self, unit_id: str) -> None:
        return


def test_recompile_migrates_skill_units_to_new_base_kernel(tmp_path: Path) -> None:
    # 1. Train and promote a Skill Unit on kernel v1.
    v1_state = tmp_path / "state-v1"
    v1_kernel = _KernelStub("mock/qwen-v1-0.8b", delta_seed_offset=10)
    mgr_v1 = TenantManager(state_dir=v1_state, kernel=v1_kernel, initial_tenants=("acme",))
    mgr_v1.ingest_trajectories(
        "acme",
        [
            {
                "task_family": "ops",
                "prompt": "Event: Cloud invoice #101 exceeds cap",
                "response": "bad",
                "senior_correction": '{"seat":"finance","gate":"owner","code":"FIN-CAP"}',
                "evidence": {"verified": True, "verifier": "senior_ceo"},
            }
        ],
    )
    study = mgr_v1.run_study_cycle(
        "acme",
        unit_id="ops-skill-v1",
        version="1.0.0",
        task_family="ops",
        title="Ops Policy",
        summary="Routes invoice cap events",
        proposer="tools",
        eval_cases=[{"prompt": "Event: Vendor invoice #999 over cap", "expected": "FIN-CAP"}],
    )
    assert study["manifest"]["base_kernel_id"] == "mock/qwen-v1-0.8b"
    assert study["manifest"]["state"] == "conformant"

    v1_promoted = mgr_v1.promote_unit("acme", "ops-skill-v1", approver="risk")
    assert v1_promoted.state is UnitState.ENABLED

    # Verify skill_spec.json is saved and CaPheOS export references it + baseKernelId.
    v1_units_dir = v1_state / "tenants" / "acme" / "units"
    v1_registry = SkillUnitRegistry(v1_units_dir)
    spec = v1_registry.load_skill_spec("ops-skill-v1")
    assert spec is not None
    assert spec.unit_id == "ops-skill-v1"
    assert len(spec.training_pairs) == 1

    export_dir = v1_registry.export_capheos_unit("ops-skill-v1", tmp_path / "exported")
    mod = json.loads((export_dir / "module.json").read_text(encoding="utf-8"))
    assert mod["subcor"]["baseKernelId"] == "mock/qwen-v1-0.8b"
    assert mod["subcor"]["skillSpecFile"] == "skill_spec.json"
    assert (export_dir / "skill_spec.json").exists()

    # Also propose a legacy unit WITHOUT a skill_spec.json to verify graceful skip.
    v1_registry.propose(
        unit_id="legacy-no-spec",
        version="0.1.0",
        title="Legacy",
        summary="No spec",
        proposer="tools",
        mode=AdaptationMode.DYNAMIC_EXPANSION_FP32,
        deltas=[_seeded_delta(v1_kernel.tap_layer, seed=99)],
        base_kernel_id="mock/qwen-v1-0.8b",
    )
    v1_registry.record_conformance(
        "legacy-no-spec",
        baseline_before=1.0,
        baseline_after=1.0,
        target_before=0.0,
        target_after=1.0,
    )

    # 2. Upgrade base kernel to v2 and recompile all units automatically.
    v2_state = tmp_path / "state-v2"
    v2_kernel = _KernelStub("mock/qwen-v2-1.5b", delta_seed_offset=200)
    report = recompile_units_for_kernel(
        source_units_dir=v1_units_dir,
        dest_state_dir=v2_state,
        target_tenant="acme",
        kernel=v2_kernel,
    )

    assert report.target_kernel_id == "mock/qwen-v2-1.5b"
    assert report.conformant_count == 1
    assert report.failed_count == 0
    assert report.skipped_count == 1

    outcomes_by_id = {o.unit_id: o for o in report.outcomes}
    recompiled = outcomes_by_id["ops-skill-v1"]
    assert recompiled.status == "conformant"
    assert recompiled.source_kernel_id == "mock/qwen-v1-0.8b"
    assert recompiled.target_kernel_id == "mock/qwen-v2-1.5b"
    assert recompiled.manifest is not None
    # Recompiled units MUST land in CONFORMANT (never self-enabled).
    assert recompiled.manifest.state is UnitState.CONFORMANT
    assert recompiled.manifest.base_kernel_id == "mock/qwen-v2-1.5b"
    # Freshly compiled weights for v2 produce a new SHA-256 artifact digest.
    assert recompiled.manifest.payload_sha256 != v1_promoted.payload_sha256

    assert outcomes_by_id["legacy-no-spec"].status == "skipped_no_spec"
