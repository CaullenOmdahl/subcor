"""Tests for the standardized longitudinal functional growth benchmark system."""

from __future__ import annotations

import threading
from pathlib import Path

import numpy as np

from subcor.benchmark import (
    BenchmarkCase,
    BenchmarkLedger,
    BenchmarkSuite,
    compare_snapshots,
    core_v1_suite,
    grade_response,
    render_markdown_report,
    run_benchmark_http,
    run_benchmark_local,
)
from subcor.router import MoESkillRouter, RoutingDecision
from subcor.server import (
    InferenceResult,
    SubcorHTTPServer,
    TenantManager,
    build_dummy_delta,
)
from subcor.units import LayerDeltaPayload, SkillUnitRegistry


class _MockBenchKernel:
    model_id = "mock/qwen3.5-0.8b"
    tap_layer = "layers.18.mlp.down_proj"
    target_layers = ("layers.18.mlp.down_proj",)

    def __init__(self) -> None:
        self._ops = np.eye(16, dtype=np.float64)[0]
        self._gen = np.eye(16, dtype=np.float64)[1]

    def encode_prompt_embedding(self, user_prompt: str) -> np.ndarray:
        return self._ops.copy() if "event:" in user_prompt.lower() else self._gen.copy()

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
        return [build_dummy_delta(self.tap_layer, rank=2)], 1.4, 0.04, 0.0

    def invalidate_unit(self, unit_id: str) -> None:
        return None


def test_suite_digest_and_roundtrip_and_graders(tmp_path: Path) -> None:
    suite = core_v1_suite()
    assert suite.suite_id == "subcor-core-v1"
    assert len(suite.axes) == 5
    digest_1 = suite.digest()

    out_path = tmp_path / "suite.json"
    suite.to_json_file(out_path)
    reloaded = BenchmarkSuite.from_json_file(out_path)
    assert reloaded.digest() == digest_1

    # Verify all deterministic grader types.
    c_json = BenchmarkCase("1", "a", "p", "exact_json", {"seat": "ceo", "gate": "owner"})
    assert grade_response(c_json, '{"seat":"ceo","gate":"owner"}', hit_token_limit=False)
    assert grade_response(
        c_json, '```json\n{"seat":"ceo","gate":"owner"}\n```', hit_token_limit=False
    )
    assert not grade_response(c_json, '{"seat":"risk"}', hit_token_limit=False)

    c_keys = BenchmarkCase("2", "a", "p", "json_keys", ["tool", "args"])
    assert grade_response(c_keys, '{"tool":"read","args":{"p":"a"}}', hit_token_limit=False)
    assert not grade_response(c_keys, '{"tool":"read"}', hit_token_limit=False)

    c_num = BenchmarkCase("3", "a", "p", "numeric", "144")
    assert grade_response(c_num, "The answer is 144.", hit_token_limit=False)
    assert not grade_response(c_num, "1440", hit_token_limit=False)

    c_concise = BenchmarkCase(
        "4", "a", "p", "concise_contains", {"contains": "ready", "max_words": 3}
    )
    assert grade_response(c_concise, "READY", hit_token_limit=False)
    assert not grade_response(
        c_concise, "I am definitely very much READY right now", hit_token_limit=False
    )
    assert not grade_response(c_concise, "READY", hit_token_limit=True)


def test_longitudinal_benchmark_ledger_diff_and_http_runner(tmp_path: Path) -> None:
    mini_suite = BenchmarkSuite(
        suite_id="mini-growth-v1",
        version="1.0.0",
        description="Compact longitudinal test suite",
        anchor_axes=("general_reasoning",),
        cases=(
            BenchmarkCase("g1", "general_reasoning", "What is 12 * 12?", "numeric", "144"),
            BenchmarkCase("g2", "general_reasoning", "Capital of Japan?", "contains", "Tokyo"),
            BenchmarkCase(
                "o1",
                "domain_ops_routing",
                "Event: Vendor invoice #999 over budget",
                "exact_json",
                {"seat": "finance", "gate": "owner", "code": "FIN-CAP"},
            ),
        ),
    )

    kernel = _MockBenchKernel()
    manager = TenantManager(
        state_dir=tmp_path / "state",
        kernel=kernel,
        initial_tenants=("company-alpha",),
    )
    ledger = BenchmarkLedger(tmp_path / "benchmarks" / "history.jsonl")

    # Snapshot 0: Day-0 frozen base (via local runner).
    snap0 = run_benchmark_local(
        suite=mini_suite,
        kernel=kernel,
        label="day0-frozen-base",
        tenant=manager.ensure_tenant("company-alpha"),
    )
    ledger.append(snap0)
    assert snap0.axis_summaries["general_reasoning"].pass_rate == 1.0
    assert snap0.axis_summaries["domain_ops_routing"].pass_rate == 0.0
    assert len(snap0.active_units) == 0

    # Train & promote ops-skill-v1 on company-alpha.
    manager.ingest_trajectories(
        "company-alpha",
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
    manager.run_study_cycle(
        "company-alpha",
        unit_id="ops-skill-v1",
        version="1.0.0",
        task_family="ops",
        title="Ops Skill",
        summary="Ops skill unit",
        proposer="tools",
        eval_cases=[{"prompt": "Event: Vendor invoice #999 over budget", "expected": "FIN-CAP"}],
    )
    manager.promote_unit("company-alpha", "ops-skill-v1", approver="risk")

    # Snapshot 1: Live HTTP benchmark against SubcorHTTPServer!
    server = SubcorHTTPServer(("127.0.0.1", 0), manager=manager, api_key="bench-key")
    host, port = server.server_address
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        snap1 = run_benchmark_http(
            suite=mini_suite,
            base_url=f"http://{host}:{port}/v1",
            model="subcor/company-alpha",
            label="v1-after-ops-unit",
            api_key="bench-key",
        )
    finally:
        server.shutdown()
        server.server_close()

    ledger.append(snap1)
    assert snap1.axis_summaries["general_reasoning"].pass_rate == 1.0
    assert snap1.axis_summaries["domain_ops_routing"].pass_rate == 1.0
    assert len(snap1.active_units) == 1
    assert snap1.active_units[0]["unit_id"] == "ops-skill-v1"

    # Compare Day-0 vs v1 and render longitudinal Markdown report.
    loaded = ledger.load_all(suite_digest=mini_suite.digest())
    assert len(loaded) == 2

    delta = compare_snapshots(loaded[0], loaded[1])
    assert delta.suite_digest_match is True
    assert delta.anchor_regressed is False
    assert delta.anchor_delta == 0.0
    assert delta.axis_deltas["domain_ops_routing"] == 1.0
    assert delta.overall_delta > 0.3
    assert delta.unit_count_delta == 1

    md = render_markdown_report(loaded)
    assert "# SUBCOR Functional Growth Report" in md
    assert "`day0-frozen-base`" in md
    assert "`v1-after-ops-unit`" in md
    assert "PRESERVED" in md
    assert "`ops-skill-v1`" in md
