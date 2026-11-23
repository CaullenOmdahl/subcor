"""Issue #34: seat scope is a routing hint; omitted/blank seat semantics are explicit."""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from subcor.router import MoESkillRouter, RouterGate
from subcor.server import SubcorHTTPServer, TenantManager
from test_server import _FakeMultiSkillKernel, _http_json

TAP = "layers.18.mlp.down_proj"
POS = np.eye(8)[0]
NEG = np.eye(8)[1]


def _gate(seat_scope: str | None) -> RouterGate:
    return RouterGate.fit(
        unit_id="fin-v1",
        tap_layer=TAP,
        pos_embeddings=np.stack([POS, POS]),
        neg_embeddings=np.stack([NEG, NEG]),
        seat_scope=seat_scope,
    )


@pytest.mark.parametrize(
    ("seat", "eligible"),
    [
        ("finance", True),  # correct seat
        ("  FINANCE ", True),  # normalized
        ("tools", False),  # wrong seat
        (None, True),  # omitted: no seat context, every gate eligible
        ("", True),  # blank is the same as omitted
        ("   ", True),
    ],
)
def test_scoped_gate_eligibility(seat: str | None, eligible: bool) -> None:
    gate = _gate("finance")
    assert gate.matches_seat(seat) is eligible
    router = MoESkillRouter(tap_layer=TAP)
    router.register(gate)
    selected = router.route(POS, seat=seat).selected_units
    assert selected == (("fin-v1",) if eligible else ())


@pytest.mark.parametrize("scope", [None, "", "  "])
def test_unscoped_gate_is_eligible_for_every_seat(scope: str | None) -> None:
    gate = _gate(scope)
    assert gate.seat_scope is None
    assert all(gate.matches_seat(s) for s in ("finance", "tools", None, ""))


def test_scope_is_normalized_when_loaded_from_disk() -> None:
    data = _gate("finance").to_dict()
    data["seat_scope"] = " Finance "
    gate = RouterGate.from_dict(data)
    assert gate.seat_scope == "finance"
    assert gate.matches_seat("FINANCE") and not gate.matches_seat("tools")
    data["seat_scope"] = "  "
    assert RouterGate.from_dict(data).seat_scope is None


class _SeatRecordingKernel(_FakeMultiSkillKernel):
    def __init__(self) -> None:
        super().__init__()
        self.seats: list[str | None] = []

    def generate_routed_messages(self, *, seat: str | None = None, **kwargs: Any) -> Any:
        self.seats.append(seat)
        return super().generate_routed_messages(**kwargs)


@pytest.mark.parametrize(
    ("header", "expected"),
    [(None, "intern"), ("", "intern"), ("   ", "intern"), ("finance", "finance")],
)
def test_http_chat_always_routes_with_a_seat(
    tmp_path: Path, header: str | None, expected: str
) -> None:
    kernel = _SeatRecordingKernel()
    manager = TenantManager(state_dir=tmp_path / "state", kernel=kernel, initial_tenants=("a",))
    server = SubcorHTTPServer(("127.0.0.1", 0), manager=manager, api_key="secret-key")
    host, port = server.server_address
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        code, _, _ = _http_json(
            f"http://{host}:{port}/v1/chat/completions",
            method="POST",
            body={"model": "subcor/a", "messages": [{"role": "user", "content": "hi"}]},
            extra_headers=None if header is None else {"X-CaPheOS-Seat": header},
        )
        assert code == 200
        assert kernel.seats == [expected]
    finally:
        server.shutdown()
        server.server_close()
