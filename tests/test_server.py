"""End-to-end tests for the multi-tenant SUBCOR OpenAI + Control Plane HTTP server."""

from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from subcor.router import MoESkillRouter, RoutingDecision
from subcor.server import (
    InferenceResult,
    SubcorHTTPServer,
    TenantManager,
    build_dummy_delta,
)
from subcor.units import LayerDeltaPayload, SkillUnitRegistry


@dataclass(frozen=True)
class _FakeStreamEvent:
    text: str
    token_id: int
    logprob: float
    prompt_tokens: int
    generation_tokens: int
    finish_reason: str | None = None


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
        self.last_messages: list[dict[str, Any]] = []
        self.last_tools: list[dict[str, Any]] | None = None

    def encode_prompt_embedding(self, user_prompt: str) -> np.ndarray:
        low = user_prompt.lower()
        if "invoice" in low or "contract" in low or "event:" in low:
            return self._ops_vec.copy()
        if "alert:" in low or "outage" in low or "503" in low:
            return self._noc_vec.copy()
        return self._gen_vec.copy()

    def _route_for_prompt(
        self,
        user_prompt: str,
        router: MoESkillRouter | None,
        top_k: int,
    ) -> RoutingDecision:
        emb = self.encode_prompt_embedding(user_prompt)
        return (
            router.route(emb, top_k=top_k)
            if router is not None and router.registered_units
            else RoutingDecision(selected_units=(), scores={}, thresholds={})
        )

    def generate_routed_messages(
        self,
        *,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        router: MoESkillRouter | None,
        registry: SkillUnitRegistry | None,
        max_tokens: int,
        enable_thinking: bool,
        top_k: int,
    ) -> InferenceResult:
        self.last_messages = list(messages)
        self.last_tools = tools
        user_prompt = ""
        for msg in reversed(messages):
            if msg.get("role") == "user":
                user_prompt = str(msg.get("content", ""))
                break
        decision = self._route_for_prompt(user_prompt, router, top_k)

        # Simulate tool calling when tools are provided and prompt asks to lookup.
        if tools and "lookup" in user_prompt.lower():
            return InferenceResult(
                text="",
                prompt_tokens=20,
                completion_tokens=10,
                hit_token_limit=False,
                repetition_ratio=0.0,
                routing=decision,
                tool_calls=(
                    {
                        "id": "call_fake001",
                        "type": "function",
                        "function": {
                            "name": tools[0]["function"]["name"],
                            "arguments": '{"query":"invoice-555"}',
                        },
                    },
                ),
                token_logprobs=(-0.05, -0.04, -0.06),
            )

        # Simulate low confidence when prompt mentions "ambiguous".
        if "ambiguous" in user_prompt.lower():
            return InferenceResult(
                text="not sure",
                prompt_tokens=14,
                completion_tokens=2,
                hit_token_limit=False,
                repetition_ratio=0.0,
                routing=decision,
                token_logprobs=(-1.8, -1.6),
            )

        # Simulate multi-turn memory recall across earlier turns.
        if "my codename" in user_prompt.lower():
            joined = " ".join(str(m.get("content", "")) for m in messages)
            if "cobalt-7" in joined.lower():
                return InferenceResult(
                    text="Your codename is Cobalt-7.",
                    prompt_tokens=24,
                    completion_tokens=6,
                    hit_token_limit=False,
                    repetition_ratio=0.0,
                    routing=decision,
                    token_logprobs=(-0.03, -0.04, -0.02),
                )

        return self.generate_routed_chat(
            user_prompt=user_prompt,
            system_prompt=None,
            router=router,
            registry=registry,
            max_tokens=max_tokens,
            enable_thinking=enable_thinking,
            top_k=top_k,
        )

    def stream_routed_messages(
        self,
        *,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        router: MoESkillRouter | None,
        registry: SkillUnitRegistry | None,
        max_tokens: int,
        enable_thinking: bool,
        top_k: int,
    ) -> tuple[RoutingDecision, Iterator[_FakeStreamEvent]]:
        user_prompt = ""
        for msg in reversed(messages):
            if msg.get("role") == "user":
                user_prompt = str(msg.get("content", ""))
                break
        decision = self._route_for_prompt(user_prompt, router, top_k)
        tokens = (
            ['{"seat":"finance",', '"gate":"owner",', '"code":"FIN-CAP"}']
            if "ops-skill-v1" in decision.selected_units
            else ["base-", "stream"]
        )

        def _gen() -> Iterator[_FakeStreamEvent]:
            for i, tok in enumerate(tokens, start=1):
                yield _FakeStreamEvent(
                    text=tok,
                    token_id=i,
                    logprob=-0.05,
                    prompt_tokens=12,
                    generation_tokens=i,
                    finish_reason="stop" if i == len(tokens) else None,
                )

        return decision, _gen()

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
        decision = self._route_for_prompt(user_prompt, router, top_k)
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
            token_logprobs=(-0.04, -0.05, -0.03),
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
    server = SubcorHTTPServer(("127.0.0.1", 0), manager=manager, api_key="secret-key")
    host, port = server.server_address
    base_url = f"http://{host}:{port}"

    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        # 1. /readyz unauthenticated returns minimal status; authenticated returns tenant details.
        code, unauth_health, _ = _http_json(f"{base_url}/readyz", api_key=None)
        assert code == 200
        assert unauth_health == {"status": "ok", "ready": True}

        code, health, _ = _http_json(f"{base_url}/readyz")
        assert code == 200
        assert sorted(health["tenants"].keys()) == ["company-alpha", "company-beta"]

        unauth_code, _, _ = _http_json(f"{base_url}/v1/models", api_key="wrong")
        assert unauth_code == 401

        code, models, _ = _http_json(f"{base_url}/v1/models")
        assert code == 200
        model_ids = {m["id"] for m in models["data"]}
        assert {"subcor/base", "subcor/company-alpha", "subcor/company-beta"} <= model_ids

        # Ollama-compatible /api/tags discovery endpoint (for StarNet's built-in `ollama` provider).
        code, tags, _ = _http_json(f"{base_url}/api/tags")
        assert code == 200
        tag_names = {m["name"] for m in tags["models"]}
        assert {"subcor/base", "subcor/company-alpha", "subcor/company-beta"} <= tag_names

        # 2. Ingest verified + unverified trajectories into company-alpha ONLY.
        code, ingest, _ = _http_json(
            f"{base_url}/v1/subcor/tenants/company-alpha/trajectories",
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
            f"{base_url}/v1/subcor/tenants/company-alpha/study",
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
            f"{base_url}/v1/subcor/tenants/company-alpha/units/ops-skill-v1/promote",
            method="POST",
            body={"approver": "tools"},
        )
        assert code == 403

        code, promoted, _ = _http_json(
            f"{base_url}/v1/subcor/tenants/company-alpha/units/ops-skill-v1/promote",
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
                "model": "subcor/company-alpha",
                "messages": [{"role": "user", "content": "Event: Vendor invoice #555 over cap"}],
            },
        )
        assert code == 200
        assert "FIN-CAP" in alpha_ops["choices"][0]["message"]["content"]
        assert alpha_ops["subcor"]["selected_units"] == ["ops-skill-v1"]
        assert hdrs.get("X-SUBCOR-Routed-Units") == "ops-skill-v1"
        assert hdrs.get("X-SUBCOR-Escalate") == "false"
        assert float(hdrs.get("X-SUBCOR-Confidence", "0")) > 0.9
        assert alpha_ops["subcor"]["confidence"]["escalate_to_cortex"] is False
        assert alpha_ops["subcor"]["confidence"]["escalation_reason"] == "ok_habitual_execution"

        # 5b. Cortical Handoff (Thalamic Switch) flags low-confidence turns for escalation.
        code, ambig, ambig_hdrs = _http_json(
            f"{base_url}/v1/chat/completions",
            method="POST",
            body={
                "model": "subcor/company-alpha",
                "messages": [{"role": "user", "content": "This is an ambiguous novel request"}],
            },
        )
        assert code == 200
        assert ambig_hdrs.get("X-SUBCOR-Escalate") == "true"
        assert ambig["subcor"]["confidence"]["escalate_to_cortex"] is True
        assert ambig["subcor"]["confidence"]["escalation_reason"] == "low_decode_confidence"

        # 5c. Multi-turn chat history is passed intact (no longer truncated to last user turn).
        code, multiturn, _ = _http_json(
            f"{base_url}/v1/chat/completions",
            method="POST",
            body={
                "model": "subcor/company-alpha",
                "messages": [
                    {"role": "system", "content": "Be concise."},
                    {"role": "user", "content": "Remember my codename is Cobalt-7."},
                    {"role": "assistant", "content": "Noted."},
                    {"role": "user", "content": "What is my codename?"},
                ],
            },
        )
        assert code == 200
        assert "Cobalt-7" in multiturn["choices"][0]["message"]["content"]
        assert len(kernel.last_messages) == 4

        # 5d. OpenAI tools/tool_calls support returns finish_reason="tool_calls".
        code, tool_resp, _ = _http_json(
            f"{base_url}/v1/chat/completions",
            method="POST",
            body={
                "model": "subcor/company-alpha",
                "messages": [{"role": "user", "content": "Please lookup invoice-555"}],
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "search_ledger",
                            "description": "Search invoice ledger",
                            "parameters": {
                                "type": "object",
                                "properties": {"query": {"type": "string"}},
                            },
                        },
                    }
                ],
            },
        )
        assert code == 200
        assert tool_resp["choices"][0]["finish_reason"] == "tool_calls"
        assert tool_resp["choices"][0]["message"]["content"] is None
        tcs = tool_resp["choices"][0]["message"]["tool_calls"]
        assert len(tcs) == 1
        assert tcs[0]["function"]["name"] == "search_ledger"

        code, alpha_gen, _ = _http_json(
            f"{base_url}/v1/chat/completions",
            method="POST",
            body={
                "model": "subcor/company-alpha",
                "messages": [{"role": "user", "content": "What is 12 * 12?"}],
            },
        )
        assert code == 200
        assert alpha_gen["choices"][0]["message"]["content"] == "144"
        assert alpha_gen["subcor"]["fell_back_to_base"] is True

        code, beta_ops, _ = _http_json(
            f"{base_url}/v1/chat/completions",
            method="POST",
            body={
                "model": "subcor/company-beta",
                "messages": [{"role": "user", "content": "Event: Vendor invoice #555 over cap"}],
            },
        )
        assert code == 200
        assert beta_ops["choices"][0]["message"]["content"] == "untrained-base-response"
        assert beta_ops["subcor"]["fell_back_to_base"] is True

        # 6. True token-by-token streaming SSE endpoint emits multiple incremental chunks.
        sse_req = urllib.request.Request(
            f"{base_url}/v1/chat/completions",
            data=json.dumps(
                {
                    "model": "subcor/company-alpha",
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
            data_lines = [
                line[6:]
                for line in sse_text.splitlines()
                if line.startswith("data: ") and line != "data: [DONE]"
            ]
            # role chunk + 3 token chunks + final done chunk = 5 SSE JSON chunks
            assert len(data_lines) == 5
            combined = "".join(
                json.loads(d)["choices"][0]["delta"].get("content", "") for d in data_lines
            )
            assert "FIN-CAP" in combined
            final_chunk = json.loads(data_lines[-1])
            assert final_chunk["choices"][0]["finish_reason"] == "stop"
            assert final_chunk["subcor"]["confidence"]["escalate_to_cortex"] is False
            assert "data: [DONE]" in sse_text

        # 7. Export CaPheOS format-2 bundle via API (sandboxed inside export_root).
        code, exp, _ = _http_json(
            f"{base_url}/v1/subcor/tenants/company-alpha/units/ops-skill-v1/export-capheos",
            method="POST",
            body={"destination": "exported"},
        )
        assert code == 200
        exp_path = Path(exp["exported_path"])
        assert (exp_path / "module.json").exists()
        assert (exp_path / "router_gate.json").exists()
        assert (exp_path / "skill_spec.json").exists()

        # 8. Atomic rollback deregisters unit live -> immediate revert to frozen base.
        code, rb, _ = _http_json(
            f"{base_url}/v1/subcor/tenants/company-alpha/units/ops-skill-v1/rollback",
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
                "model": "subcor/company-alpha",
                "messages": [{"role": "user", "content": "Event: Vendor invoice #555 over cap"}],
            },
        )
        assert code == 200
        assert after_rb["choices"][0]["message"]["content"] == "untrained-base-response"
        assert after_rb["subcor"]["fell_back_to_base"] is True
    finally:
        server.shutdown()
        server.server_close()


def test_invalid_tenant_id_rejected(tmp_path: Path) -> None:
    mgr = TenantManager(state_dir=tmp_path / "s", kernel=_FakeMultiSkillKernel())
    with pytest.raises(ValueError, match="Invalid tenant_id"):
        mgr.ensure_tenant("../escape")
