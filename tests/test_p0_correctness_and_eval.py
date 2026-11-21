"""Regression tests for P0 Issues #10 (mount/state atomicity) and #11 (eval & grader rigor)."""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pytest

from subcor.benchmark import (
    AxisSummary,
    BenchmarkCase,
    BenchmarkLedger,
    BenchmarkSnapshot,
    compare_snapshots,
    core_v1_suite,
    grade_response,
    render_markdown_report,
)
from subcor.cli import main as cli_main
from subcor.intern import TextTrajectory, TextTrajectoryStore
from subcor.router import MoESkillRouter, RouterGate, RoutingDecision
from subcor.server import (
    DEFAULT_BASELINE_ANCHOR_CASES,
    DEFAULT_NEGATIVE_CALIBRATION_PROMPTS,
    InferenceResult,
    TenantManager,
    _layer_sort_key,
    build_dummy_delta,
)
from subcor.telemetry import TrajectoryStore
from subcor.types import (
    AdaptationMode,
    GovernanceError,
    TrajectoryRecord,
    UnitState,
    VerificationEvidence,
)
from subcor.units import LayerDeltaPayload, SkillUnitRegistry


class _MinimalKernel:
    model_id = "mock/p0-kernel"
    tap_layer = "layers.18.mlp.down_proj"
    target_layers = ("layers.18.mlp.down_proj",)

    def encode_prompt_embedding(self, user_prompt: str) -> np.ndarray:
        v = np.zeros(16, dtype=np.float64)
        v[0 if "event:" in user_prompt.lower() else 1] = 1.0
        return v

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
        if decision.selected_units:
            text = '{"code":"FIN-CAP"}'
        elif "12 * 12" in user_prompt:
            text = "144"
        elif "japan" in user_prompt.lower():
            text = "Tokyo"
        elif "15 + 27" in user_prompt:
            text = "42"
        elif "sun" in user_prompt.lower():
            text = "Mercury"
        elif "100 - 37" in user_prompt:
            text = "63"
        elif "gold" in user_prompt.lower():
            text = "Au"
        else:
            text = "base"
        return InferenceResult(
            text=text,
            prompt_tokens=8,
            completion_tokens=4,
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
        return [build_dummy_delta(self.tap_layer, rank=2)], 1.2, 0.05, 0.0

    def invalidate_unit(self, unit_id: str) -> None:
        return


def test_layer_sort_key_orders_numerically() -> None:
    names = [
        "layers.22.mlp.down_proj",
        "layers.11.mlp.down_proj",
        "layers.7.mlp.down_proj",
        "layers.15.mlp.down_proj",
    ]
    ordered = sorted(names, key=_layer_sort_key)
    assert ordered == [
        "layers.7.mlp.down_proj",
        "layers.11.mlp.down_proj",
        "layers.15.mlp.down_proj",
        "layers.22.mlp.down_proj",
    ]


def test_record_conformance_blocks_enabled_and_rolled_back_units(tmp_path: Path) -> None:
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
    reg.record_conformance(
        "u1",
        baseline_before=1.0,
        baseline_after=1.0,
        target_before=0.0,
        target_after=1.0,
    )
    reg.promote("u1", approver="risk")

    # Cannot re-certify an ENABLED unit.
    with pytest.raises(GovernanceError, match="enabled"):
        reg.record_conformance(
            "u1",
            baseline_before=1.0,
            baseline_after=0.0,
            target_before=0.0,
            target_after=0.0,
        )
    assert reg.get_manifest("u1").state is UnitState.ENABLED

    # Cannot re-certify a ROLLED_BACK unit.
    reg.rollback("u1", actor="risk", reason="test")
    with pytest.raises(GovernanceError, match="rolled_back"):
        reg.record_conformance(
            "u1",
            baseline_before=1.0,
            baseline_after=1.0,
            target_before=0.0,
            target_after=1.0,
        )
    assert reg.get_manifest("u1").state is UnitState.ROLLED_BACK


def test_promote_validates_gate_before_disk_write_and_startup_quarantines_tampered(
    tmp_path: Path,
) -> None:
    state_dir = tmp_path / "state"
    kernel = _MinimalKernel()
    mgr = TenantManager(state_dir=state_dir, kernel=kernel, initial_tenants=("t1",))
    ctx = mgr.ensure_tenant("t1")

    # Propose & certify a unit WITHOUT a router gate -> promote must fail AND disk stays CONFORMANT.
    ctx.registry.propose(
        unit_id="no-gate-u",
        version="1.0.0",
        title="No Gate",
        summary="Missing gate",
        proposer="tools",
        mode=AdaptationMode.DYNAMIC_EXPANSION_FP32,
        deltas=[build_dummy_delta(kernel.tap_layer, rank=2)],
    )
    ctx.registry.record_conformance(
        "no-gate-u",
        baseline_before=1.0,
        baseline_after=1.0,
        target_before=0.0,
        target_after=1.0,
    )
    with pytest.raises(GovernanceError, match="missing router_gate.json"):
        mgr.promote_unit("t1", "no-gate-u", approver="risk")
    assert ctx.registry.get_manifest("no-gate-u").state is UnitState.CONFORMANT

    # Save a mismatched tap_layer gate -> promote must fail AND disk stays CONFORMANT.
    bad_gate = RouterGate.fit(
        unit_id="no-gate-u",
        tap_layer="layers.7.mlp.down_proj",
        pos_embeddings=np.array([[1.0, 0.0]]),
        neg_embeddings=np.array([[0.0, 1.0]]),
    )
    ctx.registry.save_router_gate(bad_gate)
    with pytest.raises(GovernanceError, match="tap_layer"):
        mgr.promote_unit("t1", "no-gate-u", approver="risk")
    assert ctx.registry.get_manifest("no-gate-u").state is UnitState.CONFORMANT

    # Now save a valid gate and promote cleanly; then tamper payload on disk and restart.
    good_gate = RouterGate.fit(
        unit_id="no-gate-u",
        tap_layer=kernel.tap_layer,
        pos_embeddings=np.array([[1.0, 0.0]]),
        neg_embeddings=np.array([[0.0, 1.0]]),
    )
    ctx.registry.save_router_gate(good_gate)
    mgr.promote_unit("t1", "no-gate-u", approver="risk")
    assert ctx.registry.get_manifest("no-gate-u").state is UnitState.ENABLED

    payload_path = state_dir / "tenants" / "t1" / "units" / "no-gate-u" / "payload.scun"
    payload_path.write_bytes(b"TAMPERED-BYTES")

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        mgr2 = TenantManager(state_dir=state_dir, kernel=kernel, initial_tenants=("t1",))
    ctx2 = mgr2.ensure_tenant("t1")
    assert "no-gate-u" in ctx2.quarantined_units
    assert "no-gate-u" not in ctx2.router.registered_units
    assert any("Quarantined" in str(w.message) for w in caught)


def test_jsonl_stores_tolerate_corrupted_lines(tmp_path: Path) -> None:
    text_store = TextTrajectoryStore(tmp_path / "text.jsonl")
    text_store.append(
        TextTrajectory(
            trajectory_id="ok-1",
            task_family="ops",
            seat="intern",
            prompt="p",
            response="r",
            tokens_generated=1,
            hit_token_limit=False,
            repetition_ratio=0.0,
            passed_verifier=True,
            evidence=VerificationEvidence(verified=True, verifier="v", method="test"),
        )
    )
    with text_store.path.open("a", encoding="utf-8") as fh:
        fh.write("{corrupted truncated json\n")

    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        loaded = text_store.load_all()
    assert len(loaded) == 1
    assert text_store.corrupted_line_count == 1

    vec_store = TrajectoryStore(tmp_path / "vec.jsonl")
    vec_store.append(
        TrajectoryRecord(
            trajectory_id="v-1",
            task_family="a",
            seat="intern",
            input_vector=[1.0, 0.0],
            predicted_label=1,
            expected_label=1,
            steps_taken=1,
            hit_step_limit=False,
            succeeded=True,
            evidence=VerificationEvidence(verified=True, verifier="v", method="test"),
        )
    )
    with vec_store.path.open("a", encoding="utf-8") as fh:
        fh.write("not-json\n")
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        vec_loaded = vec_store.load_all()
    assert len(vec_loaded) == 1
    assert vec_store.corrupted_line_count == 1


def test_strict_graders_reject_false_positives() -> None:
    # 1. Numeric: repeating equation "3x + 7 = 28" must NOT pass as answer 7.
    c_num = BenchmarkCase("gen-03", "general_reasoning", "3x+7=28", "numeric", "7")
    assert not grade_response(c_num, "3x + 7 = 28", hit_token_limit=False)
    assert grade_response(c_num, "3x = 21, so x = 7", hit_token_limit=False)
    assert grade_response(c_num, "7", hit_token_limit=False)

    # 2. Contains: negated mention "Not Tokyo, Kyoto" must NOT pass as Tokyo.
    c_city = BenchmarkCase("gen-02", "general_reasoning", "Capital?", "contains", "Tokyo")
    assert not grade_response(c_city, "Not Tokyo, Kyoto", hit_token_limit=False)
    assert not grade_response(c_city, "It is not Tokyo.", hit_token_limit=False)
    assert grade_response(c_city, "Tokyo", hit_token_limit=False)

    # 3. Uppercase discipline case in core_v1_suite must reject lowercase "hello world".
    suite = core_v1_suite()
    disc03 = next(c for c in suite.cases if c.case_id == "disc-03")
    assert not grade_response(disc03, "hello world", hit_token_limit=False)
    assert grade_response(disc03, "HELLO WORLD", hit_token_limit=False)

    # 4. exact_json: bool True must NOT equal int 1.
    c_bool = BenchmarkCase("j1", "structured_json", "p", "exact_json", {"ok": True, "n": 1})
    assert grade_response(c_bool, '{"ok": true, "n": 1}', hit_token_limit=False)
    assert not grade_response(c_bool, '{"ok": 1, "n": 1}', hit_token_limit=False)
    assert not grade_response(c_bool, '{"ok": true, "n": true}', hit_token_limit=False)


def _make_snap(
    label: str,
    *,
    digest: str = "abc123",
    gen_rate: float = 0.8,
    disc_rate: float = 0.8,
) -> BenchmarkSnapshot:
    return BenchmarkSnapshot(
        run_id=f"run-{label}",
        label=label,
        suite_id="subcor-core-v1",
        suite_version="1.0.0",
        suite_digest=digest,
        model_id="mock/m",
        tenant_id=None,
        active_units=(),
        total_unit_bytes=0,
        mean_bits_per_weight=0.0,
        overall_pass_rate=(gen_rate + disc_rate) / 2,
        macro_axis_pass_rate=(gen_rate + disc_rate) / 2,
        anchor_pass_rate=(gen_rate + disc_rate) / 2,
        token_limit_rate=0.0,
        mean_tokens=10.0,
        mean_latency_ms=50.0,
        tokens_per_second=100.0,
        axis_summaries={
            "general_reasoning": AxisSummary(
                axis="general_reasoning",
                total=4,
                passed=int(gen_rate * 4),
                pass_rate=gen_rate,
                token_limit_rate=0.0,
                mean_tokens=5.0,
                mean_latency_ms=50.0,
                base_fallback_rate=1.0,
            ),
            "instruction_discipline": AxisSummary(
                axis="instruction_discipline",
                total=4,
                passed=int(disc_rate * 4),
                pass_rate=disc_rate,
                token_limit_rate=0.0,
                mean_tokens=5.0,
                mean_latency_ms=50.0,
                base_fallback_rate=1.0,
            ),
        },
        case_results=(),
    )


def test_compare_snapshots_catches_per_axis_regression_and_digest_mismatch_with_cli_exit(
    tmp_path: Path,
) -> None:
    # Pooled anchor average stays 0.50 -> 0.50, but general_reasoning drops -50 pp!
    s0 = _make_snap("base", gen_rate=1.0, disc_rate=0.0)
    s1 = _make_snap("cand", gen_rate=0.5, disc_rate=0.5)
    delta = compare_snapshots(s0, s1)
    assert delta.anchor_delta == 0.0
    assert delta.anchor_regressed is True
    assert delta.regressed_anchor_axes == ("general_reasoning",)
    assert delta.ci_passed is False

    ledger_path = tmp_path / "reg.jsonl"
    ledger = BenchmarkLedger(ledger_path)
    ledger.append(s0)
    ledger.append(s1)
    exit_code = cli_main(["bench", "compare", "--ledger", str(ledger_path)])
    assert exit_code == 2

    # Digest mismatch also fails ci_passed and CLI exit code, and warns in Markdown report.
    s_other = _make_snap("other-suite", digest="zzz999", gen_rate=1.0, disc_rate=1.0)
    delta_mismatch = compare_snapshots(s0, s_other)
    assert delta_mismatch.suite_digest_match is False
    assert delta_mismatch.ci_passed is False
    md = render_markdown_report([s0, s_other])
    assert "SUITE DIGEST MISMATCH" in md


def test_default_study_cycle_separates_calibration_negatives_and_auto_holds_out_eval(
    tmp_path: Path,
) -> None:
    # Verify default anchor cases and default negative calibration prompts are strictly disjoint.
    base_set = {p.strip().lower() for p, _ in DEFAULT_BASELINE_ANCHOR_CASES}
    neg_set = {p.strip().lower() for p in DEFAULT_NEGATIVE_CALIBRATION_PROMPTS}
    assert base_set.isdisjoint(neg_set)

    mgr = TenantManager(state_dir=tmp_path / "st", kernel=_MinimalKernel(), initial_tenants=("t",))
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
    res = mgr.run_study_cycle(
        "t",
        unit_id="ops-heldout-v1",
        version="1.0.0",
        task_family="ops",
        title="Ops",
        summary="Ops heldout test",
    )
    assert res["eval_mode"] == "auto_held_out_split"
    assert res["verified_training_pairs"] == 3
    assert res["eval_cases_count"] == 1
    assert res["baseline_neg_overlap"] == 0
    assert res["manifest"]["state"] == "conformant"


def test_mlx_mount_rollback_prevents_cross_request_leak(tmp_path: Path) -> None:
    """Verify MLXKernel.mount_unit rolls back partial layer mounts on error (Issue #10)."""
    pytest.importorskip("mlx")
    from subcor.backends.mlx_kernel import MLXKernel

    class _FakeLayer:
        def __init__(self, *, fail_on_mount: bool = False) -> None:
            self.fail_on_mount = fail_on_mount
            self.mounted: dict[str, object] = {}

        @property
        def mounted_names(self) -> tuple[str, ...]:
            return tuple(self.mounted.keys())

        def mount_adapter(self, name: str, adapter: object) -> None:
            if self.fail_on_mount:
                raise ValueError("Simulated shape mismatch on layer 2")
            self.mounted[name] = adapter

        def unmount_adapter(self, name: str) -> None:
            self.mounted.pop(name, None)

    reg = SkillUnitRegistry(tmp_path / "units")
    reg.propose(
        unit_id="u-partial",
        version="1.0.0",
        title="Partial",
        summary="2-layer unit",
        proposer="tools",
        mode=AdaptationMode.DYNAMIC_EXPANSION_FP32,
        deltas=[
            build_dummy_delta("layers.18.mlp.down_proj", rank=2),
            build_dummy_delta("layers.19.mlp.down_proj", rank=2),
        ],
    )

    kernel = object.__new__(MLXKernel)
    layer18 = _FakeLayer(fail_on_mount=False)
    layer19 = _FakeLayer(fail_on_mount=True)
    kernel.instrumented = {
        "layers.18.mlp.down_proj": layer18,
        "layers.19.mlp.down_proj": layer19,
    }
    kernel._verified_unit_cache = {}

    with pytest.raises(ValueError, match="shape mismatch"):
        kernel.mount_unit("u-partial", reg)

    # Layer 18 MUST be cleanly unmounted (no leaked adapter!).
    assert layer18.mounted_names == ()
    assert layer19.mounted_names == ()
