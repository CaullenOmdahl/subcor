"""Embedding fallback of `LlamaCppServingAdapter.encode_prompt_embedding`.

A stdlib mock llama-server reproduces the real llama.cpp b12110 behaviours:

  * default pooling (`none`): POST /v1/embeddings -> 400
    {"error": {"code": 400, "message": "Pooling type 'none' is not OAI compatible. ..."}}
    while POST /embedding returns per-token vectors [{"index": 0, "embedding": [[...], ...]}];
  * a server that returns a list-of-lists from /v1/embeddings;
  * a correctly pooled server (`--pooling mean`).

The adapter must work with every configuration and return the L2-normalised MEAN of the
per-token vectors when it has to pool itself. Token vectors are chosen so that first-token,
last-token, or normalise-then-mean pooling all give a different answer.
"""

from __future__ import annotations

import json
import threading
import urllib.error
from collections.abc import Iterator
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import numpy as np
import pytest

from subcor.backends.llamacpp_kernel import LlamaCppServingAdapter

POOLING_NONE_ERROR = {
    "error": {
        "code": 400,
        "message": "Pooling type 'none' is not OAI compatible. Please use a different pooling type",
        "type": "invalid_request_error",
    }
}

BATCH_LIMIT = 512

TOKENS_INVOICE = [[3.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
TOKENS_OTHER = [[0.0, 2.0, 0.0], [0.0, 0.0, 1.0]]


def _tokens_for(text: str) -> list[list[float]]:
    return TOKENS_INVOICE if "invoice" in text.lower() else TOKENS_OTHER


def _expected(tokens: list[list[float]]) -> np.ndarray:
    mean = np.asarray(tokens, dtype=np.float64).mean(axis=0)
    return mean / np.linalg.norm(mean)


class _MockServer(ThreadingHTTPServer):
    allow_reuse_address = True

    def __init__(self, mode: str) -> None:
        self.mode = mode
        self.calls: list[tuple[str, Any]] = []
        super().__init__(("127.0.0.1", 0), _Handler)


class _Handler(BaseHTTPRequestHandler):
    server: _MockServer

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        return

    def _send(self, status: int, payload: Any) -> None:
        raw = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self) -> None:  # noqa: N802
        self.server.calls.append((f"GET {self.path}", None))
        if self.path == "/lora-adapters":
            self._send(HTTPStatus.OK, [])
        elif self.path == "/health":
            self._send(HTTPStatus.OK, {"status": "ok"})
        else:
            self._send(HTTPStatus.NOT_FOUND, {"error": "unknown"})

    def do_POST(self) -> None:  # noqa: N802
        length = int(self.headers.get("Content-Length", "0"))
        body = json.loads(self.rfile.read(length).decode("utf-8")) if length else {}
        self.server.calls.append((self.path, body))
        mode = self.server.mode
        text = ""
        if isinstance(body, dict):
            text = str(body.get("input", body.get("content", "")))

        if self.path == "/lora-adapters":
            self._send(HTTPStatus.OK, {"success": True})
        elif self.path == "/tokenize":
            # Mock vocabulary: word "w<N>" is token N.
            self._send(HTTPStatus.OK, {"tokens": [int(w[1:]) for w in text.split()]})
        elif self.path == "/detokenize":
            self._send(HTTPStatus.OK, {"content": " ".join(f"w{t}" for t in body["tokens"])})
        elif self.path == "/v1/embeddings":
            if mode == "batch_limit" and len(text.split()) + 1 > BATCH_LIMIT:  # +1: BOS
                self._send(
                    HTTPStatus.INTERNAL_SERVER_ERROR,
                    {
                        "error": {
                            "code": 500,
                            "message": f"input ({len(text.split()) + 1} tokens) is too large "
                            "to process. increase the physical batch size (current batch "
                            f"size: {BATCH_LIMIT})",
                            "type": "server_error",
                        }
                    },
                )
            elif mode == "pooling_none":
                self._send(HTTPStatus.BAD_REQUEST, POOLING_NONE_ERROR)
            elif mode == "other_400":
                self._send(
                    HTTPStatus.BAD_REQUEST,
                    {"error": {"code": 400, "message": "input is too large to process"}},
                )
            elif mode == "server_error":
                self._send(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": {"message": "boom"}})
            elif mode == "v1_per_token":
                self._send(HTTPStatus.OK, {"data": [{"index": 0, "embedding": _tokens_for(text)}]})
            else:  # pooled
                vec = [3.0, 4.0, 0.0] if "invoice" in text.lower() else [0.0, 0.0, 2.0]
                self._send(
                    HTTPStatus.OK,
                    {"object": "list", "data": [{"index": 0, "embedding": vec}], "usage": {}},
                )
        elif self.path == "/embedding":
            self._send(HTTPStatus.OK, [{"index": 0, "embedding": _tokens_for(text)}])
        else:
            self._send(HTTPStatus.NOT_FOUND, {"error": "unknown"})


@pytest.fixture
def mock_server(request: pytest.FixtureRequest) -> Iterator[_MockServer]:
    srv = _MockServer(request.param)
    th = threading.Thread(target=srv.serve_forever, daemon=True)
    th.start()
    try:
        yield srv
    finally:
        srv.shutdown()
        srv.server_close()


def _adapter(srv: _MockServer) -> LlamaCppServingAdapter:
    host, port = srv.server_address[:2]
    return LlamaCppServingAdapter(base_url=f"http://{host}:{port}", model_id="mock.gguf")


def _embedding_calls(srv: _MockServer) -> list[tuple[str, Any]]:
    return [c for c in srv.calls if c[0] in ("/v1/embeddings", "/embedding")]


@pytest.mark.parametrize("mock_server", ["pooling_none"], indirect=True)
def test_pooling_none_400_falls_back_to_embedding_and_mean_pools(mock_server: _MockServer) -> None:
    adapter = _adapter(mock_server)
    prompt = "Invoice #10 over cap"
    emb = adapter.encode_prompt_embedding(prompt)
    assert emb.dtype == np.float64 and emb.shape == (3,)
    np.testing.assert_allclose(emb, _expected(TOKENS_INVOICE), atol=1e-12)

    paths = [p for p, _ in _embedding_calls(mock_server)]
    assert "/embedding" in paths, paths
    fallback_bodies = [b for p, b in _embedding_calls(mock_server) if p == "/embedding"]
    assert any(prompt in json.dumps(b) for b in fallback_bodies), fallback_bodies

    # Subsequent prompts keep working (and are pooled per prompt, not cached).
    emb2 = adapter.encode_prompt_embedding("What is 2 + 2?")
    np.testing.assert_allclose(emb2, _expected(TOKENS_OTHER), atol=1e-12)
    assert not np.allclose(emb, emb2)


@pytest.mark.parametrize("mock_server", ["v1_per_token"], indirect=True)
def test_list_of_lists_from_v1_embeddings_is_mean_pooled(mock_server: _MockServer) -> None:
    adapter = _adapter(mock_server)
    emb = adapter.encode_prompt_embedding("invoice due")
    assert emb.shape == (3,)
    np.testing.assert_allclose(emb, _expected(TOKENS_INVOICE), atol=1e-12)
    emb2 = adapter.encode_prompt_embedding("other")
    np.testing.assert_allclose(emb2, _expected(TOKENS_OTHER), atol=1e-12)


@pytest.mark.parametrize("mock_server", ["pooled"], indirect=True)
def test_pooled_server_uses_v1_embeddings_directly(mock_server: _MockServer) -> None:
    adapter = _adapter(mock_server)
    emb = adapter.encode_prompt_embedding("invoice due")
    np.testing.assert_allclose(emb, [0.6, 0.8, 0.0], atol=1e-12)
    emb2 = adapter.encode_prompt_embedding("hello")
    np.testing.assert_allclose(emb2, [0.0, 0.0, 1.0], atol=1e-12)
    paths = [p for p, _ in _embedding_calls(mock_server)]
    assert paths and all(p == "/v1/embeddings" for p in paths), paths


@pytest.mark.parametrize("mock_server", ["server_error"], indirect=True)
def test_non_pooling_errors_are_not_masked_by_the_fallback(mock_server: _MockServer) -> None:
    """Only the pooling-'none' 400 triggers the fallback; a real server error must surface."""
    adapter = _adapter(mock_server)
    with pytest.raises((urllib.error.HTTPError, RuntimeError)):
        adapter.encode_prompt_embedding("invoice due")


@pytest.mark.parametrize("mock_server", ["other_400"], indirect=True)
def test_unrelated_400_is_raised_and_does_not_switch_to_the_fallback(
    mock_server: _MockServer,
) -> None:
    """A 400 that is not the pooling-'none' error (e.g. prompt too long) must surface and
    must NOT permanently switch the adapter to /embedding."""
    adapter = _adapter(mock_server)
    for _ in range(2):
        with pytest.raises(urllib.error.HTTPError) as excinfo:
            adapter.encode_prompt_embedding("invoice due")
        assert excinfo.value.code == 400
    paths = [p for p, _ in _embedding_calls(mock_server)]
    assert paths == ["/v1/embeddings", "/v1/embeddings"], paths


@pytest.mark.parametrize("mock_server", ["batch_limit"], indirect=True)
def test_prompt_longer_than_physical_batch_is_clipped_to_its_last_tokens(
    mock_server: _MockServer,
) -> None:
    """llama-server rejects embedding inputs above `-ub` tokens; routing must clip and retry."""
    adapter = _adapter(mock_server)
    prompt = " ".join(f"w{i}" for i in range(1000))
    emb = adapter.encode_prompt_embedding(prompt)
    assert emb.shape == (3,)

    bodies = [b for p, b in _embedding_calls(mock_server)]
    assert len(bodies) == 2, bodies
    retried = bodies[-1]["input"].split()
    assert len(retried) + 1 <= BATCH_LIMIT
    assert retried[-1] == "w999"  # the END of the message is what routing keeps

    # A prompt that fits goes through in a single call.
    mock_server.calls.clear()
    adapter.encode_prompt_embedding("w1 w2 w3")
    assert len(_embedding_calls(mock_server)) == 1
