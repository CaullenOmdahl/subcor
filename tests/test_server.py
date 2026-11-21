"""End-to-end tests for the multi-tenant SLAINGI OpenAI + Control Plane HTTP server."""

from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from slaingi.router import MoESkillRouter, RoutingDecision
from slaingi.server import (
    InferenceResult,
    SlaingiHTTPServer,
    TenantManager,
    build_dummy_delta,
)
from slaingi.units import LayerDeltaPayload, SkillUnitRegistry


class _FakeMultiSkillKernel:
    """Deterministic test kernel simulating a frozen base + routed Skill Units."""

    model_id = "fake/qwen3.5-0.8b-test"
    tap_layer = "layers.18.mlp.down_proj"
    target_layers = ("layers.18.mlp.down_proj",)

    def __init__(self) -> None:
        self._dim = 16
        self._ops_vec = np.eye(self._dim, dtype=np.float64)[0]
        self._noc_vec = np.eye(self._dim, dtype=np.float64)[1]
        self._gen_vec = np.eye(self._dim, dtype=np.float64)[2]
        self.invalidated: list[str] = []

    def encode_prompt_embedding(self, user_prompt: str) -> np.ndarray:
        low = user_prompt.lower()
        if "invoice" in low or "contract" in low or "event:" in low:
            return self._ops_vec.copy()
        if "alert:" in low or "outage" in low or "503" in low:
            return self._noc_vec.copy()
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
        elif "noc-skill-v1" in decision.selected_units:
            text = '{"sev":"P1","notify":"pager","sla_min":15}'
        elif "12 * 12" in user_prompt:
            text = "144"
        elif "japan" in user_prompt.lower():
            text = "Tokyo"
        else:
            text = "untrained-base-response"
        return InferenceResult(
            text=text,
            prompt_tokens=12,
            completion_tokens=8,
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
        return [build_dummy_delta(self.tap_layer, rank=2)], 1.5, 0.05, 0.0

    def invalidate_unit(self, unit_id: str) -> None:
        self.invalidated.append(unit_id)


def _http_json(
    url: str,
    *,
    method: str = "GET",
    body: dict[str, Any] | None = None,
    api_key: str | None = "secret-key",
    extra_headers: dict[str, str] | None = None,
) -> tuple[int, dict[str, Any], dict[str, str]]:
    headers: dict[str, str] = {}
    if api_key is not None:
        headers["Authorization"] = f"Bearer {api_key}"
    if extra_headers:
        headers.update(extra_headers)
    data = None
    if body is not None:
        data = json.dumps(body).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            raw = resp.read().decode("utf-8")
            return resp.status, json.loads(raw), dict(resp.headers.items())
    except urllib.error.HTTPError as err:
        raw = err.read().decode("utf-8")
        return err.code, json.loads(raw), dict(err.headers.items())


def test_multi_tenant_server_isolation_study_governance_streaming_and_rollback(
    tmp_path: Path,
) -> None:
    kernel = _FakeMultiSkillKernel()
    manager = TenantManager(
        state_dir=tmp_path / "state",
        kernel=kernel,
        initial_tenants=("company-alpha", "company-beta"),
    )
    server = SlaingiHTTPServer(("127.0.0.1", 0), manager=manager, api_key="secret-key")
    host, port = server.server_address
    base_url = f"http://{host}:{port}"

    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        # 1. /readyz works unauthenticated; /v1/models requires bearer token.
        code, health, _ = _http_json(f"{base_url}/readyz", api_key=None)
        assert code == 200
        assert sorted(health["tenants"].keys()) == ["company-alpha", "company-beta"]

        unauth_code, _, _ = _http_json(f"{base_url}/v1/models", api_key="wrong")
        assert unauth_code == 401

        code, models, _ = _http_json(f"{base_url}/v1/models")
        assert code == 200
        model_ids = {m["id"] for m in models["data"]}
        assert {"slaingi/base", "slaingi/company-alpha", "slaingi/company-beta"} <= model_ids

        # 2. Ingest verified + unverified trajectories into company-alpha ONLY.
        code, ingest, _ = _http_json(
            f"{base_url}/v1/slaingi/tenants/company-alpha/trajectories",
            method="POST",
            body={
                "trajectories": [
                    {
                        "task_family": "ops",
                        "prompt": "Event: Cloud invoice #101 exceeds cap",
                        "response": "bad",
                        "senior_correction": '{"seat":"finance","gate":"owner","code":"FIN-CAP"}',
                        "evidence": {"verified": True, "verifier": "senior_ceo"},
                    },
                    {
                        "task_family": "ops",
                        "prompt": "Event: Poison unverified",
                        "response": "bad",
                        "senior_correction": "POISON",
                        "evidence": {"verified": False, "verifier": "none"},
                    },
                ]
            },
        )
        assert code == 200
        assert ingest == {"ingested": 2, "verified": 1, "unverified": 1}

        # 3. Study cycle for company-alpha -> CONFORMANT (not yet enabled!).
        code, study, _ = _http_json(
            f"{base_url}/v1/slaingi/tenants/company-alpha/study",
            method="POST",
            body={
                "unit_id": "ops-skill-v1",
                "task_family": "ops",
                "proposer": "tools",
                "eval_cases": [
                    {
                        "prompt": "Event: Vendor invoice #999 over budget",
                        "expected": "FIN-CAP",
                    }
                ],
            },
        )
        assert code == 200
        assert study["verified_training_pairs"] == 1
        assert study["manifest"]["state"] == "conformant"

        # 4. Self-approval blocked (403); independent risk seat promotes to ENABLED.
        code, _, _ = _http_json(
            f"{base_url}/v1/slaingi/tenants/company-alpha/units/ops-skill-v1/promote",
            method="POST",
            body={"approver": "tools"},
        )
        assert code == 403

        code, promoted, _ = _http_json(
            f"{base_url}/v1/slaingi/tenants/company-alpha/units/ops-skill-v1/promote",
            method="POST",
            body={"approver": "risk"},
        )
        assert code == 200
        assert promoted["manifest"]["state"] == "enabled"

        # 5. Tenant isolation proof:
        #    - company-alpha routes Ops prompts to ops-skill-v1
        #    - company-alpha routes General QA to base (0 experts)
        #    - company-beta NEVER sees company-alpha's ops-skill-v1!
        code, alpha_ops, hdrs = _http_json(
            f"{base_url}/v1/chat/completions",
            method="POST",
            body={
                "model": "slaingi/company-alpha",
                "messages": [{"role": "user", "content": "Event: Vendor invoice #555 over cap"}],
            },
        )
        assert code == 200
        assert "FIN-CAP" in alpha_ops["choices"][0]["message"]["content"]
        assert alpha_ops["slaingi"]["selected_units"] == ["ops-skill-v1"]
        assert hdrs.get("X-SLAINGI-Routed-Units") == "ops-skill-v1"

        code, alpha_gen, _ = _http_json(
            f"{base_url}/v1/chat/completions",
            method="POST",
            body={
                "model": "slaingi/company-alpha",
                "messages": [{"role": "user", "content": "What is 12 * 12?"}],
            },
        )
        assert code == 200
        assert alpha_gen["choices"][0]["message"]["content"] == "144"
        assert alpha_gen["slaingi"]["fell_back_to_base"] is True

        code, beta_ops, _ = _http_json(
            f"{base_url}/v1/chat/completions",
            method="POST",
            body={
                "model": "slaingi/company-beta",
                "messages": [{"role": "user", "content": "Event: Vendor invoice #555 over cap"}],
            },
        )
        assert code == 200
        assert beta_ops["choices"][0]["message"]["content"] == "untrained-base-response"
        assert beta_ops["slaingi"]["fell_back_to_base"] is True

        # 6. Streaming SSE endpoint works cleanly for CaPheOS streamed turns.
        sse_req = urllib.request.Request(
            f"{base_url}/v1/chat/completions",
            data=json.dumps(
                {
                    "model": "slaingi/company-alpha",
                    "stream": True,
                    "messages": [{"role": "user", "content": "Event: invoice stream test"}],
                }
            ).encode("utf-8"),
            headers={
                "Authorization": "Bearer secret-key",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        with urllib.request.urlopen(sse_req, timeout=5) as resp:
            sse_text = resp.read().decode("utf-8")
            assert resp.status == 200
            assert "FIN-CAP" in sse_text
            assert "data: [DONE]" in sse_text

        # 7. Export CaPheOS format-2 bundle via API.
        code, exp, _ = _http_json(
            f"{base_url}/v1/slaingi/tenants/company-alpha/units/ops-skill-v1/export-capheos",
            method="POST",
            body={"destination": str(tmp_path / "exported")},
        )
        assert code == 200
        exp_path = Path(exp["exported_path"])
        assert (exp_path / "module.json").exists()
        assert (exp_path / "router_gate.json").exists()

        # 8. Atomic rollback deregisters unit live -> immediate revert to frozen base.
        code, rb, _ = _http_json(
            f"{base_url}/v1/slaingi/tenants/company-alpha/units/ops-skill-v1/rollback",
            method="POST",
            body={"actor": "risk", "reason": "Drill"},
        )
        assert code == 200
        assert rb["manifest"]["state"] == "rolled_back"
        assert "ops-skill-v1" in kernel.invalidated

        code, after_rb, _ = _http_json(
            f"{base_url}/v1/chat/completions",
            method="POST",
            body={
                "model": "slaingi/company-alpha",
                "messages": [{"role": "user", "content": "Event: Vendor invoice #555 over cap"}],
            },
        )
        assert code == 200
        assert after_rb["choices"][0]["message"]["content"] == "untrained-base-response"
        assert after_rb["slaingi"]["fell_back_to_base"] is True
    finally:
        server.shutdown()
        server.server_close()


def test_invalid_tenant_id_rejected(tmp_path: Path) -> None:
    mgr = TenantManager(state_dir=tmp_path / "s", kernel=_FakeMultiSkillKernel())
    with pytest.raises(ValueError, match="Invalid tenant_id"):
        mgr.ensure_tenant("../escape")
