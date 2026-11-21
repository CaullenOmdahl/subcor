"""Tests for P1 Issues #7 (hierarchical/multi-prototype MoE), #13 (RBAC/sandbox), and #4 (async study)."""

from __future__ import annotations

import json
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np

from subcor.auth import AuthPolicy, AuthPrincipal
from subcor.router import MoESkillRouter, RouterGate, RoutingDecision
from subcor.server import (
    InferenceResult,
    SubcorHTTPServer,
    TenantManager,
    build_dummy_delta,
)
from subcor.units import LayerDeltaPayload, SkillUnitRegistry


class _P1MockKernel:
    model_id = "mock/p1-kernel"
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
        time.sleep(0.05)  # Simulate brief GPU study work
        return [build_dummy_delta(self.tap_layer, rank=2)], 1.2, 0.05, 0.0

    def invalidate_unit(self, unit_id: str) -> None:
        return


def _req(
    url: str,
    *,
    method: str = "GET",
    token: str | None = None,
    body: dict[str, Any] | None = None,
) -> tuple[int, dict[str, Any], dict[str, str]]:
    headers: dict[str, str] = {}
    if token is not None:
        headers["Authorization"] = f"Bearer {token}"
    data = None
    if body is not None:
        data = json.dumps(body).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8")), dict(resp.headers.items())
    except urllib.error.HTTPError as err:
        return err.code, json.loads(err.read().decode("utf-8")), dict(err.headers.items())


def test_hierarchical_seat_scoped_and_multi_prototype_routing() -> None:
    rng = np.random.default_rng(7)
    dim = 32
    # Multimodal skill with two orthogonal clusters (axis 0 and axis 1); negatives on axis 2.
    c0 = np.eye(dim)[0]
    c1 = np.eye(dim)[1]
    neg_c = np.eye(dim)[2]

    pos_embs = np.vstack(
        [c0 + rng.normal(0, 0.03, size=(6, dim)), c1 + rng.normal(0, 0.03, size=(6, dim))]
    )
    neg_embs = neg_c + rng.normal(0, 0.03, size=(8, dim))

    gate_fin = RouterGate.fit(
        unit_id="finance-multimodal-v1",
        tap_layer="layers.18.mlp.down_proj",
        pos_embeddings=pos_embs,
        neg_embeddings=neg_embs,
        seat_scope="finance",
        num_prototypes=2,
    )
    assert len(gate_fin.prototypes) == 2
    assert gate_fin.seat_scope == "finance"

    # Serialization round-trip preserves prototypes and seat_scope.
    restored = RouterGate.from_dict(gate_fin.to_dict())
    assert len(restored.prototypes) == 2
    assert restored.seat_scope == "finance"

    # Competing NOC gate on identical embedding space to test strict seat scoping.
    gate_noc = RouterGate.fit(
        unit_id="noc-overlap-v1",
        tap_layer="layers.18.mlp.down_proj",
        pos_embeddings=pos_embs,
        neg_embeddings=neg_embs,
        seat_scope="noc",
        num_prototypes=2,
    )

    router = MoESkillRouter(tap_layer="layers.18.mlp.down_proj")
    router.register(restored)
    router.register(gate_noc)

    # Finance seat routes ONLY to finance gate on both clusters, never to NOC gate.
    d_fin_0 = router.route(c0, top_k=1, seat="finance")
    d_fin_1 = router.route(c1, top_k=1, seat="finance")
    assert d_fin_0.selected_units == ("finance-multimodal-v1",)
    assert d_fin_1.selected_units == ("finance-multimodal-v1",)
    assert "noc-overlap-v1" not in d_fin_0.scores

    # NOC seat routes ONLY to NOC gate.
    d_noc = router.route(c0, top_k=1, seat="noc")
    assert d_noc.selected_units == ("noc-overlap-v1",)
    assert "finance-multimodal-v1" not in d_noc.scores

    # Unrelated negative falls back cleanly to frozen base.
    d_neg = router.route(neg_c, top_k=1, seat="finance")
    assert d_neg.fell_back_to_base is True


def test_rbac_auth_tenant_isolation_sandboxed_export_and_async_study(tmp_path: Path) -> None:
    policy = AuthPolicy(
        {
            "tok-intern-a": AuthPrincipal(
                actor="intern-a", roles=frozenset({"chat", "ingest"}), tenants=frozenset({"alpha"})
            ),
            "tok-verify-a": AuthPrincipal(
                actor="verifier-a",
                roles=frozenset({"ingest", "verify"}),
                tenants=frozenset({"alpha"}),
            ),
            "tok-tools-a": AuthPrincipal(
                actor="tools-a",
                roles=frozenset({"propose", "approve", "export"}),
                tenants=frozenset({"alpha"}),
            ),
            "tok-risk-a": AuthPrincipal(
                actor="risk-a",
                roles=frozenset({"approve", "rollback"}),
                tenants=frozenset({"alpha"}),
            ),
            "tok-beta": AuthPrincipal(
                actor="intern-b", roles=frozenset({"chat"}), tenants=frozenset({"beta"})
            ),
        }
    )
    mgr = TenantManager(
        state_dir=tmp_path / "state",
        kernel=_P1MockKernel(),
        initial_tenants=("alpha", "beta"),
    )
    server = SubcorHTTPServer(
        ("127.0.0.1", 0),
        manager=mgr,
        auth_policy=policy,
        export_root=tmp_path / "sandboxed-exports",
    )
    host, port = server.server_address
    base = f"http://{host}:{port}"
    th = threading.Thread(target=server.serve_forever, daemon=True)
    th.start()
    try:
        # 1. 401 sets Connection: close to protect keep-alive pipelines.
        code, _, hdrs = _req(f"{base}/v1/models", token="bad-token")
        assert code == 401
        assert hdrs.get("Connection", "").lower() == "close"

        # 2. Tenant scoping: intern-a sees only alpha; cannot chat with beta or auto-create tenants.
        code, models, _ = _req(f"{base}/v1/models", token="tok-intern-a")
        assert code == 200
        assert {m["id"] for m in models["data"]} == {"subcor/base", "subcor/alpha"}

        code, _, _ = _req(
            f"{base}/v1/chat/completions",
            method="POST",
            token="tok-intern-a",
            body={"model": "subcor/beta", "messages": [{"role": "user", "content": "hi"}]},
        )
        assert code == 403

        code, _, _ = _req(
            f"{base}/v1/chat/completions",
            method="POST",
            token="tok-intern-a",
            body={"model": "subcor/unprovisioned", "messages": [{"role": "user", "content": "hi"}]},
        )
        assert code == 404
        assert not (tmp_path / "state" / "tenants" / "unprovisioned").exists()

        # 3. Verified evidence spoofing blocked for intern-a; allowed for verifier-a.
        verified_payload = {
            "trajectories": [
                {
                    "task_family": "ops",
                    "prompt": f"Event: Invoice #{i} over cap",
                    "response": "bad",
                    "senior_correction": '{"code":"FIN-CAP"}',
                    "evidence": {"verified": True, "verifier": "spoofed"},
                }
                for i in range(4)
            ]
        }
        code, _, _ = _req(
            f"{base}/v1/subcor/tenants/alpha/trajectories",
            method="POST",
            token="tok-intern-a",
            body=verified_payload,
        )
        assert code == 403

        code, ing, _ = _req(
            f"{base}/v1/subcor/tenants/alpha/trajectories",
            method="POST",
            token="tok-verify-a",
            body=verified_payload,
        )
        assert code == 200
        assert ing["verified"] == 4
        # Verifier identity was bound to verifier-a, not client-supplied "spoofed".
        stored = mgr.ensure_tenant("alpha").telemetry.load_verified()
        assert all(r.evidence.verifier == "verifier-a" for r in stored)

        # 4. Async Study Cycle returns HTTP 202 immediately & binds proposer to "tools-a".
        code, accepted, _ = _req(
            f"{base}/v1/subcor/tenants/alpha/study",
            method="POST",
            token="tok-tools-a",
            body={
                "async": True,
                "unit_id": "async-ops-v1",
                "task_family": "ops",
                "proposer": "spoofed-proposer",
                "seat_scope": "finance",
                "num_prototypes": 2,
            },
        )
        assert code == 202
        job_id = accepted["job_id"]
        assert accepted["poll_url"] == f"/v1/subcor/tenants/alpha/study/jobs/{job_id}"

        # Poll until background worker completes.
        job_state: dict[str, Any] = {}
        for _ in range(40):
            code, job_state, _ = _req(
                f"{base}/v1/subcor/tenants/alpha/study/jobs/{job_id}",
                token="tok-tools-a",
            )
            assert code == 200
            if job_state["status"] in ("succeeded", "failed"):
                break
            time.sleep(0.05)
        assert job_state["status"] == "succeeded"
        assert job_state["result"]["manifest"]["proposer"] == "tools-a"
        assert job_state["result"]["manifest"]["state"] == "conformant"
        assert job_state["result"]["seat_scope"] == "finance"

        # 5. Separation of duties bound to token identity: tools-a CANNOT self-approve
        #    even by passing approver="risk-a" in the JSON body!
        code, _, _ = _req(
            f"{base}/v1/subcor/tenants/alpha/units/async-ops-v1/promote",
            method="POST",
            token="tok-tools-a",
            body={"approver": "risk-a"},
        )
        assert code == 403

        # Independent risk-a token promotes cleanly.
        code, prom, _ = _req(
            f"{base}/v1/subcor/tenants/alpha/units/async-ops-v1/promote",
            method="POST",
            token="tok-risk-a",
            body={},
        )
        assert code == 200
        assert prom["manifest"]["state"] == "enabled"
        assert prom["manifest"]["approver"] == "risk-a"

        # 6. Export path traversal is blocked (400); relative path inside export_root succeeds.
        code, _, _ = _req(
            f"{base}/v1/subcor/tenants/alpha/units/async-ops-v1/export-capheos",
            method="POST",
            token="tok-tools-a",
            body={"destination": "../../escape"},
        )
        assert code == 400

        code, _, _ = _req(
            f"{base}/v1/subcor/tenants/alpha/units/async-ops-v1/export-capheos",
            method="POST",
            token="tok-tools-a",
            body={"destination": "/tmp/subcor-escape"},
        )
        assert code == 400

        code, exp, _ = _req(
            f"{base}/v1/subcor/tenants/alpha/units/async-ops-v1/export-capheos",
            method="POST",
            token="tok-tools-a",
            body={"destination": "bundle-v1"},
        )
        assert code == 200
        assert str(tmp_path / "sandboxed-exports" / "alpha" / "bundle-v1") in exp["exported_path"]
    finally:
        server.shutdown()
        server.server_close()
