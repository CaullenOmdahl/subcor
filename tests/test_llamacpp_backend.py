"""Tests for P3 Issue #8: cross-platform llama.cpp (`llama-server`) ServingKernel backend."""

from __future__ import annotations

import json
import threading
import urllib.error
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from subcor.backends.llamacpp_kernel import LlamaCppServingAdapter
from subcor.router import MoESkillRouter, RouterGate
from subcor.server import SubcorHTTPServer, TenantManager, build_dummy_delta
from subcor.types import AdaptationMode
from subcor.units import SkillUnitRegistry


class _MockLlamaServer(ThreadingHTTPServer):
    allow_reuse_address = True

    def __init__(self, addr: tuple[str, int]) -> None:
        self.lora_history: list[list[dict[str, Any]]] = []
        self.active_lora_ids: set[int] = set()
        self.fail_next_chat: bool = False
        super().__init__(addr, _MockLlamaHandler)


class _MockLlamaHandler(BaseHTTPRequestHandler):
    server: _MockLlamaServer

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        return

    def _read_json(self) -> Any:
        length = int(self.headers.get("Content-Length", "0"))
        return json.loads(self.rfile.read(length).decode("utf-8")) if length > 0 else {}

    def _send_json(self, status: int, payload: Any) -> None:
        raw = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_POST(self) -> None:  # noqa: N802
        body = self._read_json()

        if self.path == "/lora-adapters":
            self.server.lora_history.append(body)
            self.server.active_lora_ids = {
                int(item["id"]) for item in body if float(item.get("scale", 0.0)) > 0.0
            }
            self._send_json(HTTPStatus.OK, {"status": "ok"})
            return

        if self.path == "/v1/embeddings":
            inp = str(body.get("input", "")).lower()
            vec = [1.0, 0.0, 0.0, 0.0] if "invoice" in inp else [0.0, 1.0, 0.0, 0.0]
            self._send_json(HTTPStatus.OK, {"data": [{"embedding": vec}]})
            return

        if self.path == "/v1/chat/completions":
            if self.server.fail_next_chat:
                self.server.fail_next_chat = False
                self._send_json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": "boom"})
                return

            routed_ops = 0 in self.server.active_lora_ids
            if body.get("stream"):
                chunks = [
                    {
                        "choices": [
                            {
                                "delta": {"content": '{"code":' if routed_ops else "base-"},
                                "finish_reason": None,
                                "logprobs": {"content": [{"logprob": -0.05}]},
                            }
                        ]
                    },
                    {
                        "choices": [
                            {
                                "delta": {"content": '"FIN-CAP"}' if routed_ops else "ok"},
                                "finish_reason": "stop",
                                "logprobs": {"content": [{"logprob": -0.04}]},
                            }
                        ]
                    },
                ]
                sse = (
                    "".join(f"data: {json.dumps(c)}\n\n" for c in chunks) + "data: [DONE]\n\n"
                ).encode("utf-8")
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Content-Length", str(len(sse)))
                self.end_headers()
                self.wfile.write(sse)
                return

            content = '{"code":"FIN-CAP"}' if routed_ops else "base-answer"
            self._send_json(
                HTTPStatus.OK,
                {
                    "choices": [
                        {
                            "message": {"role": "assistant", "content": content},
                            "finish_reason": "stop",
                            "logprobs": {"content": [{"logprob": -0.05}, {"logprob": -0.06}]},
                        }
                    ],
                    "usage": {"prompt_tokens": 12, "completion_tokens": 4},
                },
            )
            return

        self._send_json(HTTPStatus.NOT_FOUND, {"error": "unknown"})


def test_llamacpp_backend_routes_streams_and_clears_loras_on_error(tmp_path: Path) -> None:
    llama = _MockLlamaServer(("127.0.0.1", 0))
    host, port = llama.server_address
    th = threading.Thread(target=llama.serve_forever, daemon=True)
    th.start()
    try:
        adapter = LlamaCppServingAdapter(
            base_url=f"http://{host}:{port}",
            model_id="qwen3.5-0.8b.gguf",
            lora_id_map={"ops-gguf-v1": 0},
        )
        reg = SkillUnitRegistry(tmp_path / "units")
        reg.propose(
            unit_id="ops-gguf-v1",
            version="1.0.0",
            title="Ops GGUF",
            summary="Cross-platform unit",
            proposer="tools",
            mode=AdaptationMode.DYNAMIC_EXPANSION_FP32,
            deltas=[build_dummy_delta(adapter.tap_layer, rank=2)],
            base_kernel_id=adapter.model_id,
        )
        gate = RouterGate.fit(
            unit_id="ops-gguf-v1",
            tap_layer=adapter.tap_layer,
            pos_embeddings=np.array([[1.0, 0.0, 0.0, 0.0]]),
            neg_embeddings=np.array([[0.0, 1.0, 0.0, 0.0]]),
        )
        router = MoESkillRouter(tap_layer=adapter.tap_layer)
        router.register(gate)

        # 1. Routed chat mounts LoRA 0, gets FIN-CAP, captures logprobs, and clears LoRA afterward.
        res = adapter.generate_routed_messages(
            messages=[{"role": "user", "content": "Invoice #10 over cap"}],
            tools=None,
            router=router,
            registry=reg,
            max_tokens=32,
            enable_thinking=False,
            top_k=1,
        )
        assert res.text == '{"code":"FIN-CAP"}'
        assert res.routing.selected_units == ("ops-gguf-v1",)
        assert len(res.token_logprobs) == 2
        assert llama.active_lora_ids == set()

        # 2. General prompt falls back to base without activating LoRA 0.
        res_base = adapter.generate_routed_messages(
            messages=[{"role": "user", "content": "What is 2 + 2?"}],
            tools=None,
            router=router,
            registry=reg,
            max_tokens=32,
            enable_thinking=False,
            top_k=1,
        )
        assert res_base.text == "base-answer"
        assert res_base.routing.fell_back_to_base is True

        # 3. True token-by-token SSE streaming yields incremental tokens and clears LoRA on exit.
        dec, events = adapter.stream_routed_messages(
            messages=[{"role": "user", "content": "Invoice stream check"}],
            tools=None,
            router=router,
            registry=reg,
            max_tokens=32,
            enable_thinking=False,
            top_k=1,
        )
        assert dec.selected_units == ("ops-gguf-v1",)
        collected = list(events)
        assert "".join(e.text for e in collected) == '{"code":"FIN-CAP"}'
        assert llama.active_lora_ids == set()

        # 4. Failed upstream chat MUST still reset LoRA scales to 0.0 (no cross-request leak).
        llama.fail_next_chat = True
        with pytest.raises(urllib.error.HTTPError):
            adapter.generate_routed_messages(
                messages=[{"role": "user", "content": "Invoice error test"}],
                tools=None,
                router=router,
                registry=reg,
                max_tokens=32,
                enable_thinking=False,
                top_k=1,
            )
        assert llama.active_lora_ids == set()

        # 5. Mounts cleanly behind SubcorHTTPServer + TenantManager.
        mgr = TenantManager(
            state_dir=tmp_path / "state", kernel=adapter, initial_tenants=("linux-co",)
        )
        srv = SubcorHTTPServer(("127.0.0.1", 0), manager=mgr)
        srv.server_close()
    finally:
        llama.shutdown()
        llama.server_close()
