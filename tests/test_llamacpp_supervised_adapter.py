"""Supervised llama.cpp serving adapter: per-request LoRA, prepare_units, training, and
the TenantManager `prepare_units` hook.

Fully offline: a stdlib mock llama-server records request bodies, a fake supervisor
records restarts (no subprocess), and GGUF export is stubbed so no base GGUF is needed.
"""

from __future__ import annotations

import contextlib
import json
import threading
import warnings
import weakref
from collections.abc import Iterator, Sequence
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from subcor.backends.llamacpp_kernel import LlamaCppServingAdapter
from subcor.expansion import LowRankAdapter
from subcor.router import MoESkillRouter, RouterGate, RoutingDecision
from subcor.server import InferenceResult, TenantManager, build_dummy_delta
from subcor.types import AdaptationMode, DeltaEncoding, GovernanceError, UnitState
from subcor.units import LayerDeltaPayload, SkillUnitRegistry

# --------------------------------------------------------------------------- mock server


class _Llama(ThreadingHTTPServer):
    allow_reuse_address = True

    def __init__(self) -> None:
        self.chat_bodies: list[dict[str, Any]] = []
        self.lora_posts: list[Any] = []
        super().__init__(("127.0.0.1", 0), _Handler)


class _Handler(BaseHTTPRequestHandler):
    server: _Llama

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        return

    def _send(self, payload: Any) -> None:
        raw = json.dumps(payload).encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_POST(self) -> None:  # noqa: N802
        n = int(self.headers.get("Content-Length", "0"))
        body = json.loads(self.rfile.read(n).decode("utf-8")) if n else {}
        if self.path == "/lora-adapters":
            self.server.lora_posts.append(body)
            self._send({"success": True})
        elif self.path == "/v1/embeddings":
            inp = str(body.get("input", "")).lower()
            vec = [1.0, 0.0, 0.0] if "invoice" in inp else [0.0, 1.0, 0.0]
            self._send({"data": [{"embedding": vec}]})
        elif self.path == "/v1/chat/completions":
            self.server.chat_bodies.append(body)
            if body.get("stream"):
                chunk = {"choices": [{"delta": {"content": "s"}, "finish_reason": "stop"}]}
                sse = f"data: {json.dumps(chunk)}\n\ndata: [DONE]\n\n".encode()
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Content-Length", str(len(sse)))
                self.end_headers()
                self.wfile.write(sse)
                return
            self._send(
                {
                    "choices": [
                        {"message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}
                    ],
                    "usage": {"prompt_tokens": 3, "completion_tokens": 1},
                }
            )
        else:
            self.send_response(HTTPStatus.NOT_FOUND)
            self.end_headers()


@pytest.fixture
def llama() -> Iterator[_Llama]:
    srv = _Llama()
    th = threading.Thread(target=srv.serve_forever, daemon=True)
    th.start()
    try:
        yield srv
    finally:
        srv.shutdown()
        srv.server_close()


class _FakeSupervisor:
    """Stands in for LlamaServerSupervisor: ids follow load order, records restarts."""

    def __init__(self, base_url: str, tmp: Path) -> None:
        self.base_url = base_url
        tmp.mkdir(parents=True, exist_ok=True)
        self.base_gguf = tmp / "base.gguf"
        self.base_gguf.write_bytes(b"GGUF-base")
        self.lora_dir = tmp / "loras"
        self.restarts: list[dict[str, Path]] = []
        self.fail_next = False
        self.fail_count = 0  # fail this many upcoming restarts
        self.fail_when: Any = None  # optional predicate(adapters) -> bool
        self.running = True
        self.stopped = False

    @property
    def is_running(self) -> bool:
        return self.running

    def restart(self, adapters: dict[str, Path]) -> dict[str, int]:
        self.restarts.append(dict(adapters))
        fail = self.fail_next or self.fail_count > 0
        if self.fail_when is not None and self.fail_when(adapters):
            fail = True
        if fail:
            self.fail_next = False
            self.fail_count = max(0, self.fail_count - 1)
            self.running = False
            raise RuntimeError("llama-server rejected the adapter")
        self.running = True
        return {k: i for i, k in enumerate(adapters)}

    def stop(self) -> None:
        self.stopped = True
        self.running = False


def _make_adapter(llama: _Llama, tmp: Path, **kw: Any) -> tuple[LlamaCppServingAdapter, Any]:
    host, port = llama.server_address[:2]
    sup = _FakeSupervisor(f"http://{host}:{port}", tmp)
    adapter = LlamaCppServingAdapter(model_id="base.gguf", supervisor=sup, **kw)  # type: ignore[arg-type]
    exported: list[str] = []

    def _fake_export(registry: SkillUnitRegistry, unit_id: str, key: str) -> Path:
        out = tmp / "loras" / f"{key}.gguf"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(b"lora")
        exported.append(unit_id)
        return out

    adapter._export_unit = _fake_export  # type: ignore[method-assign]
    adapter.exported = exported  # type: ignore[attr-defined]
    return adapter, sup


def _propose(reg: SkillUnitRegistry, uid: str, tap: str, seed_vec: list[float]) -> RouterGate:
    reg.propose(
        unit_id=uid,
        version="1.0.0",
        title=uid,
        summary=uid,
        proposer="tools",
        mode=AdaptationMode.DYNAMIC_EXPANSION_FP32,
        deltas=[build_dummy_delta(tap, rank=2)],
    )
    neg = [0.0, 1.0, 0.0] if seed_vec[1] == 0.0 else [1.0, 0.0, 0.0]
    return RouterGate.fit(
        unit_id=uid,
        tap_layer=tap,
        pos_embeddings=np.array([seed_vec]),
        neg_embeddings=np.array([neg]),
    )


def _chat(adapter: LlamaCppServingAdapter, router: Any, reg: Any, text: str) -> InferenceResult:
    return adapter.generate_routed_messages(
        messages=[{"role": "user", "content": text}],
        tools=None,
        router=router,
        registry=reg,
        max_tokens=8,
        enable_thinking=False,
        top_k=1,
    )


# --------------------------------------------------------------------------- adapter tests


def test_supervised_prepare_units_and_per_request_lora(llama: _Llama, tmp_path: Path) -> None:
    adapter, sup = _make_adapter(llama, tmp_path)
    assert adapter.lora_mode == "request"
    reg = SkillUnitRegistry(tmp_path / "units")
    gate = _propose(reg, "inv", adapter.tap_layer, [1.0, 0.0, 0.0])
    router = MoESkillRouter(tap_layer=adapter.tap_layer)

    # No adapters loaded: no "lora" field at all, thinking flag always explicit.
    _chat(adapter, router, reg, "hello")
    assert "lora" not in llama.chat_bodies[-1]
    assert llama.chat_bodies[-1]["chat_template_kwargs"] == {"enable_thinking": False}

    adapter.prepare_units([(reg, "inv")])
    key = adapter.adapter_key(reg, "inv")
    assert len(sup.restarts) == 1 and list(sup.restarts[0]) == [key]
    assert adapter.lora_id_map == {key: 0}
    adapter.prepare_units([(reg, "inv")])  # already loaded -> no export, no restart
    assert len(sup.restarts) == 1 and adapter.exported == ["inv"]  # type: ignore[attr-defined]

    router.register(gate)
    res = _chat(adapter, router, reg, "Invoice 12 over cap")
    assert res.routing.selected_units == ("inv",)
    assert llama.chat_bodies[-1]["lora"] == [{"id": 0, "scale": 1.0}]
    res = _chat(adapter, router, reg, "what is the weather")
    assert res.routing.selected_units == ()
    # Unselected adapters are listed explicitly at 0 ("lora": [] would mean "global").
    assert llama.chat_bodies[-1]["lora"] == [{"id": 0, "scale": 0.0}]

    dec, events = adapter.stream_routed_messages(
        messages=[{"role": "user", "content": "Invoice stream"}],
        tools=None,
        router=router,
        registry=reg,
        max_tokens=8,
        enable_thinking=True,
        top_k=1,
    )
    assert "".join(e.text for e in events) == "s"
    assert dec.selected_units == ("inv",)
    assert llama.chat_bodies[-1]["lora"] == [{"id": 0, "scale": 1.0}]
    assert llama.chat_bodies[-1]["chat_template_kwargs"] == {"enable_thinking": True}

    # Request mode never touches the global scales (embeddings depend on them staying 0).
    assert llama.lora_posts == []

    # A second unit restarts once with the union, previous adapter first.
    _propose(reg, "other", adapter.tap_layer, [0.0, 1.0, 0.0])
    adapter.prepare_units([(reg, "other"), (reg, "inv")])
    key2 = adapter.adapter_key(reg, "other")
    assert list(sup.restarts[-1]) == [key, key2]
    assert adapter.lora_id_map == {key: 0, key2: 1}

    adapter.close()
    assert sup.stopped


def test_prepare_units_failure_restores_previous_adapter_set(llama: _Llama, tmp_path: Path) -> None:
    adapter, sup = _make_adapter(llama, tmp_path)
    reg = SkillUnitRegistry(tmp_path / "units")
    _propose(reg, "a", adapter.tap_layer, [1.0, 0.0, 0.0])
    _propose(reg, "b", adapter.tap_layer, [0.0, 1.0, 0.0])
    adapter.prepare_units([(reg, "a")])
    key_a = adapter.adapter_key(reg, "a")

    sup.fail_next = True
    with pytest.raises(RuntimeError, match="rejected"):
        adapter.prepare_units([(reg, "b")])
    assert list(sup.restarts[-1]) == [key_a]  # serving brought back with the old set
    assert adapter.lora_id_map == {key_a: 0}


def test_routed_unit_without_loaded_adapter_self_heals(llama: _Llama, tmp_path: Path) -> None:
    adapter, sup = _make_adapter(llama, tmp_path)
    reg = SkillUnitRegistry(tmp_path / "units")
    gate = _propose(reg, "inv", adapter.tap_layer, [1.0, 0.0, 0.0])
    router = MoESkillRouter(tap_layer=adapter.tap_layer)
    router.register(gate)
    _chat(adapter, router, reg, "Invoice 9")
    assert len(sup.restarts) == 1
    assert llama.chat_bodies[-1]["lora"] == [{"id": 0, "scale": 1.0}]


def test_global_mode_sets_and_resets_scales(llama: _Llama, tmp_path: Path) -> None:
    adapter, _ = _make_adapter(llama, tmp_path, lora_mode="global")
    reg = SkillUnitRegistry(tmp_path / "units")
    gate = _propose(reg, "inv", adapter.tap_layer, [1.0, 0.0, 0.0])
    adapter.prepare_units([(reg, "inv")])
    router = MoESkillRouter(tap_layer=adapter.tap_layer)
    router.register(gate)
    llama.lora_posts.clear()
    _chat(adapter, router, reg, "Invoice 1")
    assert "lora" not in llama.chat_bodies[-1]
    # Exactly: pre-embedding reset, activation, then the try/finally reset.
    assert llama.lora_posts == [
        [{"id": 0, "scale": 0.0}],
        [{"id": 0, "scale": 1.0}],
        [{"id": 0, "scale": 0.0}],
    ]
    # An unrouted request only resets (for the embedding) and never activates anything.
    llama.lora_posts.clear()
    _chat(adapter, router, reg, "what is the weather")
    assert llama.lora_posts == [[{"id": 0, "scale": 0.0}]]


# 12 distinct losses: head = mean(12..7) = 9.5, tail = mean(6..1) = 3.5.
_LOSSES = [float(x) for x in range(12, 0, -1)]


class _FakeTrainer:
    def __init__(self, names: Sequence[str], *, fail: bool = False) -> None:
        self.instrumented = {n: object() for n in names}
        self.calls: list[dict[str, Any]] = []
        self.fail = fail

    def train_qlora_adapters(
        self, examples: Any, *, layer_names: list[str], **kw: Any
    ) -> tuple[dict[str, LowRankAdapter], list[float], float]:
        self.calls.append({"examples": examples, "layer_names": layer_names, **kw})
        if self.fail:
            raise RuntimeError("trainer blew up")
        rng = np.random.default_rng(0)
        adapters = {
            n: LowRankAdapter(name=n, a=rng.normal(size=(4, 16)), b=rng.normal(size=(8, 4)))
            for n in layer_names
        }
        return adapters, list(_LOSSES), 0.0


def test_train_skill_deltas_uses_trainer_and_returns_sub1bit(llama: _Llama, tmp_path: Path) -> None:
    names = ["layers.11.mlp.down_proj", "layers.3.self_attn.o_proj"]
    trainer = _FakeTrainer(names)
    adapter, _ = _make_adapter(llama, tmp_path, trainer_factory=lambda: trainer)
    examples = [(None, "p", "c")]
    deltas, head, tail, zero = adapter.train_skill_deltas(
        examples=examples, rank=4, sub1bit_rank=2, steps=3, lr=1e-3, seed=7
    )
    # Layers ordered numerically, every delta SUB1BIT at the requested rank.
    assert [d.layer_name for d in deltas] == ["layers.3.self_attn.o_proj", names[0]]
    assert all(d.encoding is DeltaEncoding.SUB1BIT_BINARY for d in deltas)
    assert all(d.sub1bit is not None and d.sub1bit.rank == 2 for d in deltas)
    assert all(np.linalg.matrix_rank(d.sub1bit.reconstruct()) == 2 for d in deltas)  # type: ignore[union-attr]
    # Every Study Cycle hyperparameter reaches the trainer unchanged.
    assert trainer.calls == [
        {
            "examples": examples,
            "layer_names": ["layers.3.self_attn.o_proj", names[0]],
            "enable_thinking": False,
            "rank": 4,
            "scale": 1.0,
            "binary_ste": False,
            "steps": 3,
            "lr": 1e-3,
            "seed": 7,
        }
    ]
    # (initial = head mean, final = tail mean): distinct, so a swap would be caught.
    assert (head, tail, zero) == (9.5, 3.5, 0.0)

    adapter.train_skill_deltas(
        examples=examples, rank=2, sub1bit_rank=1, steps=5, lr=0.5, seed=1, binary_ste=True
    )
    assert {k: trainer.calls[-1][k] for k in ("rank", "steps", "lr", "seed", "binary_ste")} == {
        "rank": 2,
        "steps": 5,
        "lr": 0.5,
        "seed": 1,
        "binary_ste": True,
    }

    bad = _FakeTrainer(["layers.1.mlp.mystery_proj"])
    adapter_bad, _ = _make_adapter(llama, tmp_path / "b", trainer_factory=lambda: bad)
    with pytest.raises(ValueError, match="cannot be exported"):
        adapter_bad.train_skill_deltas(
            examples=[(None, "p", "c")], rank=4, sub1bit_rank=2, steps=1, lr=1e-3, seed=0
        )
    assert bad.calls == []  # rejected before any training


def test_external_mode_is_inference_only() -> None:
    adapter = LlamaCppServingAdapter(base_url="http://127.0.0.1:9", model_id="x")
    assert adapter.lora_mode == "global"
    adapter.prepare_units([])  # no-op
    with pytest.raises(NotImplementedError, match="--gguf"):
        adapter.train_skill_deltas(
            examples=[(None, "p", "c")], rank=4, sub1bit_rank=2, steps=1, lr=1e-3, seed=0
        )


# --------------------------------------------------------------------------- TenantManager hook


class _HookKernel:
    model_id = "mock/hook"
    tap_layer = "layers.18.mlp.down_proj"
    target_layers = ("layers.18.mlp.down_proj",)

    def __init__(self) -> None:
        self.prepare_calls: list[list[str]] = []
        self.registered_at_prepare: list[tuple[str, ...]] = []
        self.fail_units: set[str] = set()
        self.router_ref: MoESkillRouter | None = None

    def prepare_units(self, units: Sequence[tuple[SkillUnitRegistry, str]]) -> None:
        uids = [uid for _, uid in units]
        self.prepare_calls.append(uids)
        if self.router_ref is not None:
            self.registered_at_prepare.append(self.router_ref.registered_units)
        if self.fail_units & set(uids):
            raise RuntimeError("adapter failed to load")

    def encode_prompt_embedding(self, user_prompt: str) -> np.ndarray:
        v = np.zeros(8)
        v[0 if "event:" in user_prompt.lower() else 1] = 1.0
        return v

    def generate_routed_chat(self, *, user_prompt: str, router: Any, top_k: int, **_: Any):
        decision = (
            router.route(self.encode_prompt_embedding(user_prompt), top_k=top_k)
            if router is not None and router.registered_units
            else RoutingDecision(selected_units=(), scores={}, thresholds={})
        )
        text = '{"code":"FIN-CAP"}' if decision.selected_units else "base"
        if not decision.selected_units:
            low = user_prompt.lower()
            for k, ans in (
                ("12 * 12", "144"),
                ("japan", "Tokyo"),
                ("15 + 27", "42"),
                ("sun", "Mercury"),
                ("100 - 37", "63"),
                ("gold", "Au"),
            ):
                if k in low:
                    text = ans
        return InferenceResult(
            text=text,
            prompt_tokens=4,
            completion_tokens=2,
            hit_token_limit=False,
            repetition_ratio=0.0,
            routing=decision,
        )

    def train_skill_deltas(self, **_: Any) -> tuple[list[LayerDeltaPayload], float, float, float]:
        return [build_dummy_delta(self.tap_layer, rank=2)], 1.0, 0.1, 0.0

    def invalidate_unit(self, unit_id: str) -> None:
        return


def test_tenant_manager_calls_prepare_units_hook(tmp_path: Path) -> None:
    state = tmp_path / "state"
    kernel = _HookKernel()
    mgr = TenantManager(state_dir=state, kernel=kernel, initial_tenants=("t",))
    assert kernel.prepare_calls == []  # nothing ENABLED yet -> no hydration call
    ctx = mgr.ensure_tenant("t")
    kernel.router_ref = ctx.router
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
        "t", unit_id="ops-v1", version="1.0.0", task_family="ops", title="Ops", summary="s"
    )
    assert res["manifest"]["state"] == "conformant"
    # Prepared for sandbox conformance BEFORE its gate was registered.
    assert kernel.prepare_calls == [["ops-v1"]]
    assert kernel.registered_at_prepare == [()]

    with pytest.raises(GovernanceError, match="Self-approval"):
        mgr.promote_unit("t", "ops-v1", approver="tools")
    assert kernel.prepare_calls == [["ops-v1"]]  # rejected promotion: no backend work

    kernel.fail_units = {"ops-v1"}
    with pytest.raises(RuntimeError, match="adapter failed"):
        mgr.promote_unit("t", "ops-v1", approver="risk")
    # Adapter failed to load -> unit is NOT persisted ENABLED nor routed.
    assert ctx.registry.get_manifest("ops-v1").state is UnitState.CONFORMANT
    assert "ops-v1" not in ctx.router.registered_units

    kernel.fail_units = set()
    mgr.promote_unit("t", "ops-v1", approver="risk")
    assert kernel.prepare_calls[-1] == ["ops-v1"]
    assert ctx.router.registered_units == ("ops-v1",)

    # Restart: ENABLED units are prepared in one batched call at startup.
    kernel2 = _HookKernel()
    TenantManager(state_dir=state, kernel=kernel2, initial_tenants=("t",))
    assert kernel2.prepare_calls == [["ops-v1"]]

    # Restart where the adapter cannot load: the unit is quarantined, not routed.
    kernel3 = _HookKernel()
    kernel3.fail_units = {"ops-v1"}
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        mgr3 = TenantManager(state_dir=state, kernel=kernel3, initial_tenants=("t",))
    ctx3 = mgr3.ensure_tenant("t")
    assert "ops-v1" in ctx3.quarantined_units
    assert ctx3.router.registered_units == ()
    assert any("could not load its adapter" in str(w.message) for w in caught)


# --------------------------------------------------------------------------- trainer memory


def _track_empty_cache(monkeypatch: pytest.MonkeyPatch) -> tuple[list[Any], list[list[bool]]]:
    """Fake CUDA: records which tracked trainers are still alive at each empty_cache()."""
    torch = pytest.importorskip("torch")
    refs: list[Any] = []
    seen: list[list[bool]] = []
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        torch.cuda, "empty_cache", lambda: seen.append([r() is not None for r in refs])
    )
    return refs, seen


@pytest.mark.parametrize("fail", [False, True])
def test_unload_trainer_after_study_frees_trainer_before_empty_cache(
    llama: _Llama, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fail: bool
) -> None:
    refs, seen = _track_empty_cache(monkeypatch)

    def factory() -> _FakeTrainer:
        t = _FakeTrainer(["layers.18.mlp.down_proj"], fail=fail)
        refs.append(weakref.ref(t))
        return t

    adapter, _ = _make_adapter(
        llama, tmp_path, trainer_factory=factory, unload_trainer_after_study=True
    )
    kw = {"examples": [(None, "p", "c")], "rank": 4, "sub1bit_rank": 2, "steps": 1, "lr": 1e-3}
    if fail:
        with pytest.raises(RuntimeError, match="blew up"):
            adapter.train_skill_deltas(**kw, seed=0)
    else:
        adapter.train_skill_deltas(**kw, seed=0)
    # When the VRAM is handed back, nothing (local, frame, traceback) holds the trainer.
    assert seen == [[False]]
    assert adapter._trainer is None
    # The next Study Cycle builds a fresh trainer.
    with pytest.raises(RuntimeError) if fail else contextlib.nullcontext():
        adapter.train_skill_deltas(**kw, seed=1)
    assert len(refs) == 2 and seen[-1] == [False, False]


def test_resident_trainer_still_releases_cached_memory(
    llama: _Llama, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    refs, seen = _track_empty_cache(monkeypatch)

    def factory() -> _FakeTrainer:
        t = _FakeTrainer(["layers.18.mlp.down_proj"])
        refs.append(weakref.ref(t))
        return t

    adapter, _ = _make_adapter(llama, tmp_path, trainer_factory=factory)
    kw = {"examples": [(None, "p", "c")], "rank": 4, "sub1bit_rank": 2, "steps": 1, "lr": 1e-3}
    adapter.train_skill_deltas(**kw, seed=0)
    adapter.train_skill_deltas(**kw, seed=1)
    # Kept resident (built once), but the allocator cache is emptied after each cycle.
    assert len(refs) == 1 and adapter._trainer is not None
    assert seen == [[True], [True]]


# --------------------------------------------------------------------------- production trainer


def _write_base_gguf(path: Path, *, block_count: int, nextn: int | None = None) -> Path:
    gguf = pytest.importorskip("gguf")
    writer = gguf.GGUFWriter(str(path), arch="qwen35")
    writer.add_block_count(block_count)
    if nextn is not None:
        writer.add_uint32("qwen35.nextn_predict_layers", nextn)
    writer.add_tensor("blk.0.ffn_down.weight", np.zeros((4, 8), dtype=np.float32))
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    return path


@pytest.mark.parametrize(("block_count", "nextn"), [(24, None), (24, 0), (25, 1)])
def test_build_trainer_loads_gguf_with_last_six_decoder_layers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, block_count: int, nextn: int | None
) -> None:
    """Production path: no trainer_factory -> load_trainer_from_gguf(<served GGUF>, ...).

    An MTP block (`nextn_predict_layers`) is not a decoder layer, so a 25-block GGUF with one
    MTP layer trains the same layers 18..23 as a 24-block one.
    """
    torch_kernel = pytest.importorskip("subcor.backends.torch_kernel")
    base = _write_base_gguf(tmp_path / "base.gguf", block_count=block_count, nextn=nextn)
    calls: list[tuple[Path, dict[str, Any]]] = []
    sentinel = _FakeTrainer(["layers.18.mlp.down_proj"])

    def spy(path: Any, **kw: Any) -> Any:
        calls.append((Path(path), kw))
        return sentinel

    monkeypatch.setattr(torch_kernel, "load_trainer_from_gguf", spy)
    adapter = LlamaCppServingAdapter(
        base_url="http://127.0.0.1:9",
        base_gguf=base,
        trainer_device="cpu",
        trainer_dtype="float32",
        trainer_projections=("mlp.down_proj", "self_attn.o_proj"),
    )
    assert adapter.trainer is sentinel
    assert adapter.trainer is sentinel  # built once
    assert calls == [
        (
            base.absolute(),
            {
                "target_layer_indices": [18, 19, 20, 21, 22, 23],
                "device": "cpu",
                "dtype": "float32",
                "target_projections": ("mlp.down_proj", "self_attn.o_proj"),
            },
        )
    ]

    # Explicit --trainer-layers win over the GGUF default; no projections -> TorchKernel default.
    calls.clear()
    explicit = LlamaCppServingAdapter(
        base_url="http://127.0.0.1:9", base_gguf=base, trainer_layer_indices=[2, 5]
    )
    _ = explicit.trainer
    assert calls == [
        (base.absolute(), {"target_layer_indices": [2, 5], "device": None, "dtype": None})
    ]


# --------------------------------------------------------------------------- crash recovery


def test_dead_llama_server_is_restarted_with_last_good_adapters(
    llama: _Llama, tmp_path: Path
) -> None:
    adapter, sup = _make_adapter(llama, tmp_path)
    reg = SkillUnitRegistry(tmp_path / "units")
    gate = _propose(reg, "inv", adapter.tap_layer, [1.0, 0.0, 0.0])
    adapter.prepare_units([(reg, "inv")])
    key = adapter.adapter_key(reg, "inv")
    router = MoESkillRouter(tap_layer=adapter.tap_layer)
    router.register(gate)

    sup.running = False  # llama-server crashed (segfault / device loss / OOM)
    with pytest.warns(RuntimeWarning, match="not running"):
        res = _chat(adapter, router, reg, "Invoice 3")
    assert sup.running and list(sup.restarts[-1]) == [key]
    assert res.routing.selected_units == ("inv",)
    assert llama.chat_bodies[-1]["lora"] == [{"id": 0, "scale": 1.0}]

    # Adapters cannot be reloaded (e.g. VRAM): fall back to the base model and keep serving.
    sup.running = False
    sup.fail_when = lambda adapters: bool(adapters)
    with pytest.warns(RuntimeWarning) as rec:
        res = _chat(adapter, None, None, "hello")
    assert any("base model only" in str(w.message) for w in rec)
    assert sup.running and sup.restarts[-2:] == [{key: sup.restarts[-2][key]}, {}]
    assert res.text == "ok" and "lora" not in llama.chat_bodies[-1]
    assert adapter.lora_id_map == {}

    # Streaming and embeddings also bring a dead server back.
    sup.fail_when = None
    sup.running = False
    with pytest.warns(RuntimeWarning):
        adapter.encode_prompt_embedding("x")
    assert sup.running
    sup.running = False
    with pytest.warns(RuntimeWarning):
        _, events = adapter.stream_routed_messages(
            messages=[{"role": "user", "content": "hi"}],
            tools=None,
            router=None,
            registry=None,
            max_tokens=4,
            enable_thinking=False,
            top_k=1,
        )
    assert "".join(e.text for e in events) == "s" and sup.running

    # After close() a stopped server is never restarted behind the operator's back.
    adapter.close()
    assert not sup.running
    n = len(sup.restarts)
    adapter.encode_prompt_embedding("x")  # (the mock HTTP server is still up)
    assert len(sup.restarts) == n and not sup.running


def test_prepare_units_double_failure_falls_back_to_base_server(
    llama: _Llama, tmp_path: Path
) -> None:
    adapter, sup = _make_adapter(llama, tmp_path)
    reg = SkillUnitRegistry(tmp_path / "units")
    _propose(reg, "a", adapter.tap_layer, [1.0, 0.0, 0.0])
    _propose(reg, "b", adapter.tap_layer, [0.0, 1.0, 0.0])
    adapter.prepare_units([(reg, "a")])
    key_a = adapter.adapter_key(reg, "a")

    sup.fail_count = 2  # the new set AND the previous set both fail
    with pytest.raises(RuntimeError, match="rejected"):
        adapter.prepare_units([(reg, "b")])
    assert [list(r) for r in sup.restarts[-3:]] == [
        [key_a, adapter.adapter_key(reg, "b")],
        [key_a],
        [],
    ]
    assert sup.running and adapter.lora_id_map == {}


# --------------------------------------------------------------------------- adapter pruning


def test_rolled_back_and_failed_units_are_dropped_at_next_restart(
    llama: _Llama, tmp_path: Path
) -> None:
    adapter, sup = _make_adapter(llama, tmp_path)
    reg = SkillUnitRegistry(tmp_path / "units")
    for uid, vec in (("keep", [1.0, 0.0, 0.0]), ("gone", [0.0, 1.0, 0.0])):
        _propose(reg, uid, adapter.tap_layer, vec)
        reg.record_conformance(
            uid,
            baseline_before=1.0,
            baseline_after=1.0,
            target_before=0.0,
            target_after=1.0,
            max_regression_tolerance=0.02,
            min_target_improvement=0.25,
        )
        reg.promote(uid, approver="risk")
    _propose(reg, "bad", adapter.tap_layer, [0.0, 0.0, 1.0])
    adapter.prepare_units([(reg, "keep"), (reg, "gone"), (reg, "bad")])
    k_keep, k_gone, k_bad = (adapter.adapter_key(reg, u) for u in ("keep", "gone", "bad"))
    assert list(sup.restarts[-1]) == [k_keep, k_gone, k_bad]

    # "bad" fails conformance (DORMANT with a failed report); "gone" is rolled back.
    reg.record_conformance(
        "bad",
        baseline_before=1.0,
        baseline_after=0.0,
        target_before=0.0,
        target_after=0.0,
        max_regression_tolerance=0.02,
        min_target_improvement=0.25,
    )
    assert reg.get_manifest("bad").state is UnitState.DORMANT
    reg.rollback("gone", actor="ops", reason="regression")
    adapter.invalidate_unit("gone")
    n = len(sup.restarts)
    # Still loaded (no restart just for pruning), and listed at scale 0 per request.
    assert len(sup.restarts) == n
    _chat(adapter, None, None, "hello")
    assert {e["id"] for e in llama.chat_bodies[-1]["lora"]} == {0, 1, 2}
    assert all(e["scale"] == 0.0 for e in llama.chat_bodies[-1]["lora"])

    # The next restart (a new unit) reloads only the live adapters.
    _propose(reg, "new", adapter.tap_layer, [0.5, 0.5, 0.0])
    adapter.prepare_units([(reg, "new")])
    k_new = adapter.adapter_key(reg, "new")
    assert list(sup.restarts[-1]) == [k_keep, k_new]
    assert adapter.lora_id_map == {k_keep: 0, k_new: 1}
    _chat(adapter, None, None, "hello")
    assert llama.chat_bodies[-1]["lora"] == [{"id": 0, "scale": 0.0}, {"id": 1, "scale": 0.0}]


def test_retired_but_still_loaded_adapter_is_readopted_without_restart(
    llama: _Llama, tmp_path: Path
) -> None:
    adapter, sup = _make_adapter(llama, tmp_path)
    reg = SkillUnitRegistry(tmp_path / "units")
    _propose(reg, "u", adapter.tap_layer, [1.0, 0.0, 0.0])
    adapter.prepare_units([(reg, "u")])
    key = adapter.adapter_key(reg, "u")
    reg.record_conformance(
        "u",
        baseline_before=1.0,
        baseline_after=0.0,
        target_before=0.0,
        target_after=0.0,
        max_regression_tolerance=0.02,
        min_target_improvement=0.25,
    )
    adapter.invalidate_unit("u")  # failed -> retired from the next restart
    assert key not in adapter._managed_paths and adapter.lora_id_map == {key: 0}
    n = len(sup.restarts)
    adapter.prepare_units([(reg, "u")])  # asked for again while still loaded
    assert len(sup.restarts) == n and key in adapter._managed_paths


# --------------------------------------------------------------------------- external mode


def test_external_mode_static_map_and_warning_for_unmapped_units(
    llama: _Llama, tmp_path: Path
) -> None:
    host, port = llama.server_address[:2]
    reg = SkillUnitRegistry(tmp_path / "units")
    tap = "llamacpp.embedding"
    g_inv = _propose(reg, "inv", tap, [1.0, 0.0, 0.0])
    g_other = _propose(reg, "other", tap, [0.0, 1.0, 0.0])
    router = MoESkillRouter(tap_layer=tap)
    router.register(g_inv)
    router.register(g_other)
    adapter = LlamaCppServingAdapter(
        base_url=f"http://{host}:{port}", model_id="x", lora_mode="request", lora_id_map={"inv": 3}
    )
    _chat(adapter, router, reg, "Invoice 4")
    assert llama.chat_bodies[-1]["lora"] == [{"id": 3, "scale": 1.0}]
    with pytest.warns(RuntimeWarning, match="'other' has no llama-server LoRA id"):
        res = _chat(adapter, router, reg, "what is the weather")
    assert res.routing.selected_units == ("other",)
    assert llama.chat_bodies[-1]["lora"] == [{"id": 3, "scale": 0.0}]
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _chat(adapter, router, reg, "what is the weather")  # warned once per unit
