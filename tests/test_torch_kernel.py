"""CPU tests for the PyTorch trainer backend (`subcor.backends.torch_kernel.TorchKernel`).

Every test runs fully offline on CPU in float32 against `tests/torch_tiny.py`'s tiny
hybrid (GatedDeltaNet + full attention) Qwen3_5ForCausalLM. No GPU is ever touched:
every kernel is constructed with ``device="cpu"``. Each assertion is chosen so that the
property it names fails when the property is broken (mutation thinking), e.g. exact
``== 0.0`` / ``np.array_equal`` where the contract promises bit-exactness, and a
non-trivial positive control next to every "nothing changed" check.
"""

from __future__ import annotations

import math
import os
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
pytest.importorskip("tokenizers")

sys.path.insert(0, str(Path(__file__).resolve().parent))

from subcor.backends.torch_kernel import (  # noqa: E402
    DEFAULT_TORCH_MODEL,
    TorchKernel,
    load_trainer_from_gguf,
)
from subcor.expansion import LowRankAdapter  # noqa: E402
from subcor.router import MoESkillRouter, RouterGate  # noqa: E402
from subcor.types import AdaptationMode, DeltaEncoding, IntegrityError  # noqa: E402
from subcor.units import SkillUnitRegistry, build_layer_delta  # noqa: E402
from torch_tiny import build_tiny_qwen35  # noqa: E402

HIDDEN = 64  # tiny d_model -> down_proj d_out
INTER = 128  # tiny d_ff    -> down_proj d_in
PROMPT = "Event: cloud invoice #101 exceeds cap"
MEMO_PAIRS = [("Say the code.", "XQ-17"), ("Name the color.", "teal")]


def _name(i: int, proj: str = "mlp.down_proj") -> str:
    return f"layers.{i}.{proj}"


def _make_kernel(
    indices: tuple[int, ...] | None = (1, 2, 3),
    *,
    num_layers: int = 4,
    seed: int = 0,
    **kwargs: Any,
) -> TorchKernel:
    model, tok = build_tiny_qwen35(seed=seed, num_layers=num_layers)
    return TorchKernel(
        model=model,
        tokenizer=tok,
        target_layer_indices=None if indices is None else list(indices),
        device="cpu",
        dtype=torch.float32,
        **kwargs,
    )


def _random_adapter(
    name: str,
    *,
    d_out: int = HIDDEN,
    d_in: int = INTER,
    rank: int = 2,
    seed: int = 0,
    std: float = 0.3,
) -> LowRankAdapter:
    rng = np.random.default_rng(seed)
    return LowRankAdapter(
        name=name,
        a=rng.normal(0.0, std, size=(rank, d_in)),
        b=rng.normal(0.0, std, size=(d_out, rank)),
    )


def _propose_unit(
    registry: SkillUnitRegistry,
    unit_id: str,
    layer_names: list[str],
    *,
    encoding: DeltaEncoding = DeltaEncoding.SUB1BIT_BINARY,
    rank: int = 2,
    seed: int = 0,
    shapes: dict[str, tuple[int, int]] | None = None,
) -> None:
    shapes = shapes or {}
    deltas = []
    for j, ln in enumerate(layer_names):
        d_out, d_in = shapes.get(ln, (HIDDEN, INTER))
        adapter = _random_adapter(ln, d_out=d_out, d_in=d_in, rank=rank, seed=seed + j)
        deltas.append(build_layer_delta(ln, adapter, encoding=encoding, sub1bit_rank=rank))
    registry.propose(
        unit_id=unit_id,
        version="1.0.0",
        title=f"Test unit {unit_id}",
        summary="torch backend test unit",
        proposer="tools",
        mode=AdaptationMode.DYNAMIC_EXPANSION_STE,
        deltas=deltas,
    )


def _all_unmounted(kernel: TorchKernel) -> bool:
    return all(layer.mounted_names == () for layer in kernel.instrumented.values())


def _trainable(layer: Any) -> list[Any]:
    """Trainable branch tensors as named by `trainable_lora_keys()` (mirrors MLX)."""
    params = [getattr(layer, k) for k in layer.trainable_lora_keys()]
    assert all(isinstance(p, torch.Tensor) for p in params), layer.trainable_lora_keys()
    return params


def _target_nll(kernel: TorchKernel, prompt: str, target: str) -> float:
    """Masked next-token NLL of `target` (+eos) after the chat-formatted prompt."""
    tok = kernel.tokenizer
    prefix = tok.encode(kernel.format_chat_prompt(prompt))
    tgt = tok.encode(target, add_special_tokens=False) + [int(tok.eos_token_id)]
    ids = torch.tensor([prefix + tgt], dtype=torch.long)
    with torch.no_grad():
        logits = kernel.model(input_ids=ids).logits[0].double()
    logp = torch.log_softmax(logits[:-1], dim=-1)
    targets = ids[0, 1:]
    picked = logp[torch.arange(targets.shape[0]), targets]
    return float(-picked[len(prefix) - 1 :].mean())


# --------------------------------------------------------------------------------------
# Module hygiene / construction / naming
# --------------------------------------------------------------------------------------


def test_torch_kernel_module_imports_without_mlx() -> None:
    """The trainer must import on Linux with MLX absent (mlx/mlx_lm imports are blocked)."""
    code = (
        "import sys\n"
        "sys.modules['mlx'] = None\n"
        "sys.modules['mlx_lm'] = None\n"
        "import subcor.backends.torch_kernel as tk\n"
        "assert tk.DEFAULT_TORCH_MODEL == 'Qwen/Qwen3.5-0.8B'\n"
        "print('OK')\n"
    )
    env = {**os.environ, "HIP_VISIBLE_DEVICES": "", "CUDA_VISIBLE_DEVICES": ""}
    proc = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=120, env=env
    )
    assert proc.returncode == 0 and "OK" in proc.stdout, proc.stderr[-2000:]


def test_canonical_layer_naming_and_in_place_wrapping() -> None:
    model, tok = build_tiny_qwen35()
    originals = {i: model.model.layers[i].mlp.down_proj.weight.detach().clone() for i in range(4)}
    unwrapped = {i: model.model.layers[i].mlp.down_proj for i in range(4)}
    kernel = TorchKernel(
        model=model,
        tokenizer=tok,
        target_layer_indices=[3, 1],
        device="cpu",
        dtype=torch.float32,
    )
    assert DEFAULT_TORCH_MODEL == "Qwen/Qwen3.5-0.8B"
    assert kernel.model is model and kernel.tokenizer is tok
    # Keys are the SAME canonical names MLX uses (`layers.{i}.{proj}`), not HF paths.
    assert set(kernel.instrumented) == {_name(1), _name(3)}
    assert all(p.device.type == "cpu" for p in model.parameters())
    assert all(p.dtype == torch.float32 for p in model.parameters())

    for i in (1, 3):
        wrapped = kernel.instrumented[_name(i)]
        # The HF module tree itself routes through the wrapper (not a detached copy).
        assert model.model.layers[i].mlp.down_proj is wrapped
        assert wrapped is not unwrapped[i]
        assert wrapped.layer_name == _name(i)
        assert tuple(wrapped.shape) == (HIDDEN, INTER)
        assert wrapped.mounted_names == ()
        w = wrapped.dequantize_base_weight()
        assert w.dtype == np.float64 and w.shape == (HIDDEN, INTER)
        np.testing.assert_array_equal(w, originals[i].double().numpy())
    for i in (0, 2):
        assert model.model.layers[i].mlp.down_proj is unwrapped[i]

    # Every base parameter is frozen after construction.
    assert not any(p.requires_grad for p in model.parameters())
    assert kernel.trainable_parameter_count() == 0


@pytest.mark.parametrize(("num_layers", "expected"), [(4, [1, 2, 3]), (12, [7, 9, 10, 11])])
def test_default_target_layer_indices_follow_mlx_rule(num_layers: int, expected: list[int]) -> None:
    # MLX rule: {7, 11, 15, 19, L-3, L-2, L-1} ∩ [0, L).
    kernel = _make_kernel(None, num_layers=num_layers)
    assert set(kernel.instrumented) == {_name(i) for i in expected}


def test_projection_attr_and_multi_projection_surface_skip_hybrid_layers() -> None:
    single = _make_kernel((2,), projection_attr="mlp.up_proj")
    assert set(single.instrumented) == {_name(2, "mlp.up_proj")}
    assert tuple(single.instrumented[_name(2, "mlp.up_proj")].shape) == (INTER, HIDDEN)
    assert single.model.model.layers[2].mlp.up_proj is single.instrumented[_name(2, "mlp.up_proj")]

    # Layer 2 is linear_attention (no self_attn) -> its o_proj is silently skipped, as in MLX.
    multi = _make_kernel((2, 3), target_projections=("mlp.down_proj", "self_attn.o_proj"))
    assert set(multi.instrumented) == {
        _name(2),
        _name(3),
        _name(3, "self_attn.o_proj"),
    }
    o_proj = multi.instrumented[_name(3, "self_attn.o_proj")]
    assert multi.model.model.layers[3].self_attn.o_proj is o_proj
    assert tuple(o_proj.shape) == (HIDDEN, 4 * 16)  # (hidden, heads * head_dim)

    with pytest.raises(ValueError):
        _make_kernel((0, 1, 2), target_projections=("self_attn.o_proj",))  # nothing matches


# --------------------------------------------------------------------------------------
# Instrumented-layer behaviour (exercised through the kernel's own wrappers)
# --------------------------------------------------------------------------------------


def test_wrapper_mount_bypass_record_and_restore() -> None:
    model, tok = build_tiny_qwen35(seed=3)
    base = model.model.layers[3].mlp.down_proj
    kernel = TorchKernel(
        model=model, tokenizer=tok, target_layer_indices=[3], device="cpu", dtype=torch.float32
    )
    wrapped = kernel.instrumented[_name(3)]
    assert not base.weight.requires_grad  # base frozen on wrap

    x = torch.randn(2, 5, INTER, generator=torch.Generator().manual_seed(0))
    with torch.no_grad():
        y_base = wrapped(x)
        assert torch.equal(y_base, base(x))

    # Recording: exact mean over every (batch, position); nothing recorded -> None.
    wrapped.start_recording()
    assert wrapped.stop_recording() is None
    wrapped.start_recording()
    with torch.no_grad():
        wrapped(x)
    mean_act = wrapped.stop_recording()
    assert mean_act is not None and mean_act.shape == (HIDDEN,)
    np.testing.assert_allclose(mean_act, y_base.double().mean(dim=(0, 1)).numpy(), atol=1e-6)

    # Mount: output gains exactly x @ (scale * B @ A)^T (checks orientation AND scale folding).
    adapter = _random_adapter("unit:u1", seed=1)
    adapter.scale = 0.7
    wrapped.mount_adapter("unit:u1", adapter)
    assert wrapped.mounted_names == ("unit:u1",)
    with torch.no_grad():
        y_m = wrapped(x)
    expected = x.double().numpy() @ adapter.delta_matrix().T
    np.testing.assert_allclose((y_m - y_base).double().numpy(), expected, atol=1e-4)

    # Two units stack additively.
    adapter2 = _random_adapter("unit:u2", seed=2)
    wrapped.mount_adapter("unit:u2", adapter2)
    assert set(wrapped.mounted_names) == {"unit:u1", "unit:u2"}
    with torch.no_grad():
        y_mm = wrapped(x)
    expected2 = x.double().numpy() @ (adapter.delta_matrix() + adapter2.delta_matrix()).T
    np.testing.assert_allclose((y_mm - y_base).double().numpy(), expected2, atol=1e-4)

    # Bypass reproduces the frozen base bit-exactly.
    wrapped._bypass_adapters = True
    try:
        with torch.no_grad():
            assert torch.equal(wrapped(x), y_base)
    finally:
        wrapped._bypass_adapters = False

    # Wrong-shaped adapter is rejected and leaves nothing behind.
    with pytest.raises(ValueError):
        wrapped.mount_adapter("bad", _random_adapter("bad", d_out=INTER, d_in=HIDDEN))
    assert "bad" not in wrapped.mounted_names

    wrapped.unmount_adapter("unit:u1")
    wrapped.unmount_adapter("unit:u2")
    wrapped.unmount_adapter("never-mounted")  # idempotent, like MLX
    assert wrapped.mounted_names == ()
    with torch.no_grad():
        assert torch.equal(wrapped(x), y_base)


@pytest.mark.parametrize("binary_ste", [False, True])
def test_trainable_lora_zero_init_grads_and_extraction(binary_ste: bool) -> None:
    model, tok = build_tiny_qwen35(seed=1)
    base = model.model.layers[3].mlp.down_proj
    kernel = TorchKernel(
        model=model, tokenizer=tok, target_layer_indices=[3], device="cpu", dtype=torch.float32
    )
    wrapped = kernel.instrumented[_name(3)]
    x = torch.randn(3, 4, INTER, generator=torch.Generator().manual_seed(1))
    with torch.no_grad():
        y_base = wrapped(x)

    wrapped.attach_trainable_lora(rank=4, scale=1.0, binary_ste=binary_ste, seed=7)
    trainable = _trainable(wrapped)
    assert trainable, "attach_trainable_lora must expose trainable tensors"
    for p in trainable:
        p.requires_grad_(True)
    # Exact zero perturbation right after attach (B = 0 / s_out = 0).
    with torch.no_grad():
        assert float((wrapped(x) - y_base).abs().max()) == 0.0
    assert all(p is not base.weight for p in trainable)

    # Gradients reach the new branch but never the frozen base.
    loss = ((wrapped(x) - (y_base + 0.5)) ** 2).mean()
    loss.backward()
    assert base.weight.grad is None
    assert any(p.grad is not None and float(p.grad.abs().sum()) > 0.0 for p in trainable)

    # Randomise the branch; the extracted frozen adapter must reproduce its forward exactly.
    with torch.no_grad():
        gen = torch.Generator().manual_seed(9)
        for p in trainable:
            p.add_(0.3 * torch.randn(p.shape, generator=gen, dtype=p.dtype))
        y_new = wrapped(x)
    extracted = wrapped.extract_trainable_lora("extracted")
    assert isinstance(extracted, LowRankAdapter)
    assert extracted.frozen is True
    assert extracted.binary_ste is binary_ste
    assert extracted.a.shape == (4, INTER)  # (rank, d_in)
    assert extracted.b.shape == (HIDDEN, 4)  # (d_out, rank)
    delta = extracted.delta_matrix()
    assert float(np.abs(delta).max()) > 1e-4
    np.testing.assert_allclose(
        (y_new - y_base).double().numpy(), x.double().numpy() @ delta.T, atol=1e-4
    )

    wrapped.detach_trainable_lora()
    with torch.no_grad():
        assert torch.equal(wrapped(x), y_base)
    with pytest.raises(RuntimeError):
        wrapped.extract_trainable_lora("gone")


def test_convert_lora_to_binary_ste_warm_starts_from_sub1bit_factorisation() -> None:
    model, tok = build_tiny_qwen35(seed=2)
    kernel = TorchKernel(
        model=model, tokenizer=tok, target_layer_indices=[3], device="cpu", dtype=torch.float32
    )
    wrapped = kernel.instrumented[_name(3)]
    x = torch.randn(2, 3, INTER, generator=torch.Generator().manual_seed(2))
    with torch.no_grad():
        y_base = wrapped(x)

    wrapped.attach_trainable_lora(rank=4, binary_ste=False, seed=3)
    with torch.no_grad():
        gen = torch.Generator().manual_seed(4)
        for p in _trainable(wrapped):
            p.add_(0.2 * torch.randn(p.shape, generator=gen, dtype=p.dtype))
    fp32 = wrapped.extract_trainable_lora("fp32")
    expected = fp32.compress_sub1bit(rank=4).reconstruct()

    wrapped.convert_lora_to_binary_ste()
    ste = wrapped.extract_trainable_lora("ste")
    assert ste.binary_ste is True
    np.testing.assert_allclose(ste.delta_matrix(), expected, atol=1e-5)
    with torch.no_grad():
        y_ste = wrapped(x)
    np.testing.assert_allclose(
        (y_ste - y_base).double().numpy(), x.double().numpy() @ expected.T, atol=1e-4
    )
    assert set(wrapped.trainable_lora_keys()) >= {"lora_a", "lora_b", "s_in", "s_out", "sigma"}
    wrapped.detach_trainable_lora()


# --------------------------------------------------------------------------------------
# Kernel-level invariants
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("binary_ste", [False, True])
def test_attach_is_exact_zero_perturbation_on_live_logits(binary_ste: bool) -> None:
    kernel = _make_kernel()
    before = kernel.probe_logits(PROMPT)
    assert before.dtype == np.float64
    assert before.shape == (len(kernel.tokenizer),)

    for layer in kernel.instrumented.values():
        layer.attach_trainable_lora(rank=4, binary_ste=binary_ste, seed=5)
    after = kernel.probe_logits(PROMPT)
    assert float(np.max(np.abs(after - before))) == 0.0
    for layer in kernel.instrumented.values():
        layer.detach_trainable_lora()
    assert np.array_equal(kernel.probe_logits(PROMPT), before)


def test_train_reports_nonzero_diff_when_attach_is_not_function_preserving(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Guards against a hard-coded zero_perturbation_max_diff."""
    kernel = _make_kernel()
    layer_cls = type(kernel.instrumented[_name(3)])
    original = layer_cls.attach_trainable_lora

    def perturbing_attach(self: Any, *args: Any, **kwargs: Any) -> None:
        original(self, *args, **kwargs)
        with torch.no_grad():
            for p in _trainable(self):
                p.add_(0.5)

    monkeypatch.setattr(layer_cls, "attach_trainable_lora", perturbing_attach)
    _, _, diff = kernel.train_qlora_adapters(
        [("Say the code.", "XQ")], layer_names=[_name(3)], rank=2, steps=1, lr=1e-4
    )
    assert diff > 1e-6


def test_train_rejects_bad_inputs() -> None:
    kernel = _make_kernel()
    with pytest.raises(ValueError):
        kernel.train_qlora_adapters([], layer_names=[_name(3)], steps=1)
    with pytest.raises(KeyError):
        kernel.train_qlora_adapters(MEMO_PAIRS, layer_names=[_name(0)], steps=1)
    assert kernel.trainable_parameter_count() == 0


def test_mount_then_unmount_restores_probe_logits_bit_exactly() -> None:
    kernel = _make_kernel()
    before = kernel.probe_logits(PROMPT)
    for j, (name, layer) in enumerate(kernel.instrumented.items()):
        layer.mount_adapter(f"unit:t:{name}", _random_adapter(name, seed=10 + j))
    mounted = kernel.probe_logits(PROMPT)
    assert float(np.max(np.abs(mounted - before))) > 1e-4
    for name, layer in kernel.instrumented.items():
        layer.unmount_adapter(f"unit:t:{name}")
    assert _all_unmounted(kernel)
    assert np.array_equal(kernel.probe_logits(PROMPT), before)


def test_unmount_unit_removes_only_that_units_adapters(tmp_path: Path) -> None:
    kernel = _make_kernel()
    registry = SkillUnitRegistry(tmp_path / "units")
    _propose_unit(registry, "u-one", [_name(2), _name(3)], seed=1)
    _propose_unit(registry, "u-two", [_name(3)], seed=7)
    before = kernel.probe_logits(PROMPT)

    kernel.mount_unit("u-two", registry)
    only_two = kernel.probe_logits(PROMPT)
    assert float(np.max(np.abs(only_two - before))) > 1e-5
    kernel.mount_unit("u-one", registry)
    both = kernel.probe_logits(PROMPT)
    assert float(np.max(np.abs(both - only_two))) > 1e-5
    assert len(kernel.instrumented[_name(3)].mounted_names) == 2

    kernel.unmount_unit("u-one")
    assert len(kernel.instrumented[_name(2)].mounted_names) == 0
    assert len(kernel.instrumented[_name(3)].mounted_names) == 1
    np.testing.assert_allclose(kernel.probe_logits(PROMPT), only_two, atol=1e-6)
    kernel.unmount_unit("u-two")
    assert _all_unmounted(kernel)
    assert np.array_equal(kernel.probe_logits(PROMPT), before)


@pytest.mark.parametrize("failure", ["shape_mismatch", "non_instrumented_layer", "tampered"])
def test_mount_unit_is_atomic_on_failure(tmp_path: Path, failure: str) -> None:
    """A unit whose 2nd layer fails to mount must leave NO layer mounted (Issue #10 parity)."""
    kernel = _make_kernel()
    registry = SkillUnitRegistry(tmp_path / "units")
    baseline = kernel.probe_logits(PROMPT)

    if failure == "shape_mismatch":
        # layers.2 is valid and listed FIRST; layers.3 carries a transposed (wrong) shape.
        _propose_unit(
            registry,
            "u-bad",
            [_name(2), _name(3)],
            shapes={_name(3): (INTER, HIDDEN)},
        )
        expected_exc: Any = ValueError
    elif failure == "non_instrumented_layer":
        _propose_unit(registry, "u-bad", [_name(2), _name(0)])  # layer 0 not instrumented
        expected_exc = (KeyError, ValueError)
    else:
        _propose_unit(registry, "u-bad", [_name(2), _name(3)])
        payload = tmp_path / "units" / "u-bad" / "payload.scun"
        raw = bytearray(payload.read_bytes())
        raw[-1] ^= 0xFF
        payload.write_bytes(bytes(raw))
        expected_exc = IntegrityError

    with pytest.raises(expected_exc):
        kernel.mount_unit("u-bad", registry)
    assert _all_unmounted(kernel)
    assert np.array_equal(kernel.probe_logits(PROMPT), baseline)

    # Positive control: a valid unit still mounts and changes logits afterwards.
    _propose_unit(registry, "u-good", [_name(2), _name(3)], seed=4)
    kernel.mount_unit("u-good", registry)
    assert float(np.max(np.abs(kernel.probe_logits(PROMPT) - baseline))) > 1e-5
    kernel.unmount_unit("u-good")
    assert np.array_equal(kernel.probe_logits(PROMPT), baseline)


def test_generate_with_trace_returns_trace_and_captures_activations() -> None:
    kernel = _make_kernel()
    trace = kernel.generate_with_trace(PROMPT, max_tokens=6, capture_activations=True)
    assert trace.prompt == PROMPT
    assert isinstance(trace.response, str)
    assert 1 <= trace.tokens_generated <= 6
    assert trace.prompt_tokens == len(kernel.tokenizer.encode(kernel.format_chat_prompt(PROMPT)))
    assert len(trace.token_logprobs) == trace.tokens_generated
    assert all(lp <= 1e-6 and math.isfinite(lp) for lp in trace.token_logprobs)
    assert trace.hit_token_limit == (trace.tokens_generated >= 6)
    assert set(trace.layer_activations) == set(kernel.instrumented)
    for vec in trace.layer_activations.values():
        assert vec.shape == (HIDDEN,)
        assert np.all(np.isfinite(vec))
    # Greedy decode is deterministic; no capture -> no activations; recorders switched off.
    again = kernel.generate_with_trace(PROMPT, max_tokens=6)
    assert again.response == trace.response
    assert again.token_logprobs == pytest.approx(trace.token_logprobs, abs=1e-5)
    assert again.layer_activations == {}
    assert all(layer.stop_recording() is None for layer in kernel.instrumented.values())


def _always_gate(unit_id: str, tap: str, emb: np.ndarray, threshold: float) -> RouterGate:
    # score(emb) = cos(emb, emb) - cos(emb, -emb) = 2.0 for the routing prompt.
    return RouterGate(
        unit_id=unit_id,
        tap_layer=tap,
        pos_centroid=emb.copy(),
        neg_centroid=-emb.copy(),
        threshold=threshold,
        pos_mean_score=2.0,
        neg_mean_score=-2.0,
    )


@pytest.mark.parametrize("failure", ["tampered_payload", "non_instrumented_layer"])
def test_generate_routed_failure_on_second_unit_leaves_nothing_mounted(
    tmp_path: Path, failure: str
) -> None:
    kernel = _make_kernel((1, 2, 3))
    tap = _name(1)
    registry = SkillUnitRegistry(tmp_path / "units")
    _propose_unit(registry, "unit-a", [_name(1), _name(2)], seed=1)
    if failure == "tampered_payload":
        _propose_unit(registry, "unit-b", [_name(3)], seed=5)
        payload = tmp_path / "units" / "unit-b" / "payload.scun"
        raw = bytearray(payload.read_bytes())
        raw[-1] ^= 0xFF
        payload.write_bytes(bytes(raw))
        expected_exc: Any = IntegrityError
    else:
        _propose_unit(registry, "unit-b", [_name(0)], seed=5)  # layer 0 is not instrumented
        expected_exc = (KeyError, ValueError)

    emb = kernel.encode_prompt_embedding(PROMPT, tap_layer=tap)
    router = MoESkillRouter(tap_layer=tap)
    router.register(_always_gate("unit-a", tap, emb, threshold=-1.0))  # margin 3.0 -> first
    router.register(_always_gate("unit-b", tap, emb, threshold=0.0))  # margin 2.0 -> second
    assert router.route(emb, top_k=2).selected_units == ("unit-a", "unit-b")

    # Positive control: unit-a alone really is mounted during generation, then removed.
    seen: list[tuple[str, ...]] = []
    hook = kernel.instrumented[_name(1)].register_forward_pre_hook(
        lambda mod, _inp: seen.append(tuple(mod.mounted_names))
    )
    baseline = kernel.probe_logits(PROMPT)
    seen.clear()
    try:
        trace, decision = kernel.generate_routed(
            PROMPT, router=router, registry=registry, max_tokens=3, top_k=1
        )
    finally:
        hook.remove()
    assert decision.selected_units == ("unit-a",)
    assert any(names for names in seen), "selected unit must be mounted during generation"
    assert isinstance(trace.response, str)
    assert _all_unmounted(kernel)

    # The second selected unit fails to load -> error propagates, NOTHING stays mounted.
    with pytest.raises(expected_exc):
        kernel.generate_routed(PROMPT, router=router, registry=registry, max_tokens=3, top_k=2)
    assert _all_unmounted(kernel)
    assert np.array_equal(kernel.probe_logits(PROMPT), baseline)


def test_encode_prompt_embedding_is_unit_invariant_and_matches_full_forward_tap(
    tmp_path: Path,
) -> None:
    kernel = _make_kernel((1, 2, 3))
    tap = _name(2)
    e0 = kernel.encode_prompt_embedding(PROMPT, tap_layer=tap)
    assert e0.shape == (HIDDEN,) and e0.dtype == np.float64
    assert math.isclose(float(np.linalg.norm(e0)), 1.0, abs_tol=1e-9)

    def tap_mean() -> np.ndarray:
        captured: list[Any] = []
        hook = kernel.instrumented[tap].register_forward_hook(
            lambda _m, _i, out: captured.append(out.detach().double())
        )
        ids = torch.tensor([kernel.tokenizer.encode(kernel.format_chat_prompt(PROMPT))])
        try:
            with torch.no_grad():
                kernel.model(input_ids=ids)
        finally:
            hook.remove()
        assert len(captured) == 1
        return captured[0].reshape(-1, HIDDEN).mean(dim=0).numpy()

    ref = tap_mean()
    np.testing.assert_allclose(e0, ref / np.linalg.norm(ref), atol=1e-5)

    # Mount a unit on the tap layer AND upstream/downstream layers: the embedding must not move.
    registry = SkillUnitRegistry(tmp_path / "units")
    _propose_unit(registry, "u-mounted", [_name(1), _name(2), _name(3)], seed=3)
    kernel.mount_unit("u-mounted", registry)
    assert all(layer.mounted_names for layer in kernel.instrumented.values())
    mounted_ref = tap_mean()
    assert float(np.max(np.abs(mounted_ref - ref))) > 1e-4  # non-trivial: units do change it
    e1 = kernel.encode_prompt_embedding(PROMPT, tap_layer=tap)
    assert np.array_equal(e0, e1)
    # Bypass is restored afterwards and the unit is still mounted.
    assert all(not layer._bypass_adapters for layer in kernel.instrumented.values())
    assert all(layer.mounted_names for layer in kernel.instrumented.values())
    np.testing.assert_allclose(tap_mean(), mounted_ref, atol=1e-9)

    kernel.unmount_unit("u-mounted")
    assert _all_unmounted(kernel)

    with pytest.raises((KeyError, ValueError)):
        kernel.encode_prompt_embedding(PROMPT, tap_layer=_name(0))


def test_encode_prompt_embedding_early_exits_after_tap_layer() -> None:
    kernel = _make_kernel((1, 2, 3))
    inner = kernel.model.model
    counts: dict[str, int] = defaultdict(int)
    hooks = []

    def counter(key: str) -> Any:
        def _hook(*_args: Any) -> None:
            counts[key] += 1

        return _hook

    for i, layer in enumerate(inner.layers):
        hooks.append(layer.input_layernorm.register_forward_hook(counter(f"ln{i}")))
        hooks.append(layer.mlp.register_forward_hook(counter(f"mlp{i}")))
    hooks.append(inner.norm.register_forward_hook(counter("final_norm")))
    hooks.append(kernel.model.lm_head.register_forward_hook(counter("lm_head")))
    hooks.append(kernel.instrumented[_name(1)].register_forward_hook(counter("tap")))
    try:
        kernel.encode_prompt_embedding(PROMPT, tap_layer=_name(1))
    finally:
        for h in hooks:
            h.remove()

    assert counts["ln0"] >= 1 and counts["ln1"] >= 1 and counts["tap"] >= 1
    for key in ("ln2", "mlp2", "ln3", "mlp3", "final_norm", "lm_head"):
        assert counts[key] == 0, f"{key} ran during early-exit embedding ({dict(counts)})"

    # The kernel still works normally after an early exit (no stuck hooks / state).
    logits = kernel.probe_logits(PROMPT)
    assert logits.shape == (len(kernel.tokenizer),) and np.all(np.isfinite(logits))


# --------------------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------------------


def test_train_qlora_reduces_loss_and_keeps_base_bit_identical() -> None:
    kernel = _make_kernel((1, 2, 3))
    model = kernel.model
    snapshot = {n: p.detach().clone() for n, p in model.named_parameters()}
    logits_before = kernel.probe_logits(MEMO_PAIRS[0][0])
    nll_before = _target_nll(kernel, *MEMO_PAIRS[0])
    layer_names = [_name(1), _name(2), _name(3)]

    # Kernel mirrors MLX's fp32 lr cap (4e-4), so memorisation needs ~200 tiny steps.
    adapters, losses, zero_diff = kernel.train_qlora_adapters(
        MEMO_PAIRS, layer_names=layer_names, rank=8, steps=200, lr=3e-3, seed=0
    )

    assert zero_diff == 0.0
    assert len(losses) == 200
    assert all(math.isfinite(v) for v in losses)
    head, tail = float(np.mean(losses[:6])), float(np.mean(losses[-6:]))
    assert tail < head * 0.85, f"loss did not drop enough: {head:.4f} -> {tail:.4f}"

    assert set(adapters) == set(layer_names)
    for ad in adapters.values():
        assert isinstance(ad, LowRankAdapter)
        assert ad.frozen is True
        assert ad.binary_ste is False
        assert ad.a.shape == (8, INTER) and ad.b.shape == (HIDDEN, 8)
        assert float(np.abs(ad.delta_matrix()).max()) > 0.0

    # Base weights bit-identical, no leftover LoRA params, everything frozen again.
    after = dict(model.named_parameters())
    assert set(after) == set(snapshot)
    for n, p in after.items():
        assert torch.equal(p.detach(), snapshot[n]), f"base parameter {n} changed"
        assert not p.requires_grad, f"{n} left trainable"
    assert kernel.trainable_parameter_count() == 0
    assert _all_unmounted(kernel)
    assert not model.training, "model must be returned to eval mode"
    assert np.array_equal(kernel.probe_logits(MEMO_PAIRS[0][0]), logits_before)

    # The extracted adapters carry what was learned: mounting them lowers target NLL.
    for name in layer_names:
        kernel.instrumented[name].mount_adapter(f"trained:{name}", adapters[name])
    nll_after = _target_nll(kernel, *MEMO_PAIRS[0])
    for name in layer_names:
        kernel.instrumented[name].unmount_adapter(f"trained:{name}")
    assert nll_after < nll_before - 0.05, f"{nll_before:.4f} -> {nll_after:.4f}"


def test_train_qlora_binary_ste_two_stage_path() -> None:
    kernel = _make_kernel((1, 2, 3))
    snapshot = {n: p.detach().clone() for n, p in kernel.model.named_parameters()}
    layer_names = [_name(3)]
    adapters, losses, zero_diff = kernel.train_qlora_adapters(
        MEMO_PAIRS, layer_names=layer_names, rank=4, binary_ste=True, steps=10, lr=3e-3, seed=1
    )
    assert zero_diff == 0.0
    assert len(losses) == 10 and all(math.isfinite(v) for v in losses)
    ad = adapters[_name(3)]
    assert ad.binary_ste is True  # stage 2 (Binary STE QAT) actually ran
    assert ad.a.shape == (4, INTER) and ad.b.shape == (HIDDEN, 4)
    delta = ad.delta_matrix()
    assert float(np.abs(delta).max()) > 0.0
    # QAT output packs into Sub1BitMatrix without a quantization cliff (fp16 scale rounding only).
    packed = ad.compress_sub1bit()
    np.testing.assert_allclose(packed.reconstruct(), delta, atol=1e-2 * float(np.abs(delta).max()))
    for n, p in kernel.model.named_parameters():
        assert torch.equal(p.detach(), snapshot[n])
    assert kernel.trainable_parameter_count() == 0


def test_trained_adapter_to_sub1bit_unit_round_trip(tmp_path: Path) -> None:
    kernel = _make_kernel((1, 2, 3))
    layer_names = [_name(2), _name(3)]
    adapters, _, _ = kernel.train_qlora_adapters(
        MEMO_PAIRS, layer_names=layer_names, rank=4, steps=12, lr=3e-3, seed=2
    )
    registry = SkillUnitRegistry(tmp_path / "units")
    deltas = [
        build_layer_delta(n, adapters[n], encoding=DeltaEncoding.SUB1BIT_BINARY, sub1bit_rank=4)
        for n in layer_names
    ]
    registry.propose(
        unit_id="trained-unit",
        version="1.0.0",
        title="trained",
        summary="trained on tiny memorisation pairs",
        proposer="tools",
        mode=AdaptationMode.DYNAMIC_EXPANSION_STE,
        deltas=deltas,
    )

    before = kernel.probe_logits(PROMPT)
    kernel.mount_unit("trained-unit", registry)
    assert len(kernel.instrumented[_name(1)].mounted_names) == 0
    for n in layer_names:
        assert len(kernel.instrumented[n].mounted_names) == 1
    mounted = kernel.probe_logits(PROMPT)
    assert float(np.max(np.abs(mounted - before))) > 1e-6

    # Mounted unit == manually mounting the registry's reconstructed adapters.
    kernel.unmount_unit("trained-unit")
    assert _all_unmounted(kernel)
    assert np.array_equal(kernel.probe_logits(PROMPT), before)
    for d in registry.load_verified_deltas("trained-unit"):
        kernel.instrumented[d.layer_name].mount_adapter("manual", d.to_adapter("manual"))
    np.testing.assert_allclose(kernel.probe_logits(PROMPT), mounted, atol=1e-5)
    for n in layer_names:
        kernel.instrumented[n].unmount_adapter("manual")
    assert np.array_equal(kernel.probe_logits(PROMPT), before)

    # Re-mounting after invalidation still works (cache invalidation path).
    kernel.invalidate_unit_cache("trained-unit")
    kernel.mount_unit("trained-unit", registry)
    np.testing.assert_allclose(kernel.probe_logits(PROMPT), mounted, atol=1e-6)
    kernel.unmount_unit("trained-unit")
    assert np.array_equal(kernel.probe_logits(PROMPT), before)


# --------------------------------------------------------------------------------------
# Loading from GGUF (transformers is stubbed: no network, no real checkpoint)
# --------------------------------------------------------------------------------------


class _LoadSpy:
    """Records AutoModelForCausalLM / AutoTokenizer.from_pretrained calls; returns tiny objects."""

    def __init__(self) -> None:
        self.model_calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
        self.tok_calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []

    def install(self, monkeypatch: pytest.MonkeyPatch) -> None:
        spy = self

        def fake_model(*args: Any, **kwargs: Any) -> Any:
            spy.model_calls.append((args, kwargs))
            return build_tiny_qwen35(seed=0)[0]

        def fake_tok(*args: Any, **kwargs: Any) -> Any:
            spy.tok_calls.append((args, kwargs))
            return build_tiny_qwen35(seed=0)[1]

        monkeypatch.setenv("HF_HUB_OFFLINE", "1")
        monkeypatch.setattr(transformers.AutoModelForCausalLM, "from_pretrained", fake_model)
        monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", fake_tok)

    @staticmethod
    def source_and_file(call: tuple[tuple[Any, ...], dict[str, Any]]) -> tuple[str, str | None]:
        args, kwargs = call
        src = args[0] if args else kwargs.get("pretrained_model_name_or_path")
        return str(src), kwargs.get("gguf_file")


def _assert_loaded_local_gguf(spy: _LoadSpy, gguf_path: Path) -> None:
    assert spy.model_calls, "AutoModelForCausalLM.from_pretrained was not called"
    for call in (spy.model_calls[-1], *spy.tok_calls[-1:]):
        src, gguf_file = _LoadSpy.source_and_file(call)
        assert gguf_file is not None, f"gguf_file not forwarded to transformers: {call}"
        # transformers resolves gguf_file relative to the source directory.
        assert (Path(src) / gguf_file).resolve() == gguf_path.resolve()


def test_gguf_file_from_hf_repo_is_forwarded_to_transformers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spy = _LoadSpy()
    spy.install(monkeypatch)
    kernel = TorchKernel(
        "unsloth/Qwen3.5-0.8B-GGUF",
        gguf_file="Qwen3.5-0.8B-Q4_K_M.gguf",
        target_layer_indices=[3],
        device="cpu",
        dtype=torch.float32,
    )
    assert spy.model_calls and spy.tok_calls
    for call in (spy.model_calls[-1], spy.tok_calls[-1]):
        src, gguf_file = _LoadSpy.source_and_file(call)
        assert src == "unsloth/Qwen3.5-0.8B-GGUF"
        assert gguf_file == "Qwen3.5-0.8B-Q4_K_M.gguf"
    assert set(kernel.instrumented) == {_name(3)}
    assert kernel.probe_logits(PROMPT).shape == (len(kernel.tokenizer),)


def test_gguf_file_absolute_local_path_loads_from_its_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spy = _LoadSpy()
    spy.install(monkeypatch)
    gguf_path = tmp_path / "models" / "tiny-Q4_K_M.gguf"
    gguf_path.parent.mkdir()
    gguf_path.write_bytes(b"GGUF")
    kernel = TorchKernel(
        DEFAULT_TORCH_MODEL,
        gguf_file=str(gguf_path),
        target_layer_indices=[2],
        device="cpu",
        dtype=torch.float32,
    )
    _assert_loaded_local_gguf(spy, gguf_path)
    src, _ = _LoadSpy.source_and_file(spy.model_calls[-1])
    assert Path(src).resolve() == gguf_path.parent.resolve()  # NOT the HF repo id
    assert set(kernel.instrumented) == {_name(2)}


def test_load_trainer_from_gguf_convenience(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spy = _LoadSpy()
    spy.install(monkeypatch)
    gguf_path = tmp_path / "base.gguf"
    gguf_path.write_bytes(b"GGUF")
    kernel = load_trainer_from_gguf(
        gguf_path, target_layer_indices=[1, 3], device="cpu", dtype=torch.float32
    )
    assert isinstance(kernel, TorchKernel)
    _assert_loaded_local_gguf(spy, gguf_path)
    assert set(kernel.instrumented) == {_name(1), _name(3)}
    assert all(p.device.type == "cpu" for p in kernel.model.parameters())


def test_default_device_and_dtype_fall_back_to_cpu_fp32(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    model, tok = build_tiny_qwen35()
    kernel = TorchKernel(model=model, tokenizer=tok, target_layer_indices=[3])
    assert all(p.device.type == "cpu" for p in kernel.model.parameters())
    assert all(p.dtype == torch.float32 for p in kernel.model.parameters())
    assert np.all(np.isfinite(kernel.probe_logits(PROMPT)))


def test_gguf_load_always_requests_dequantized_weights(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Without GgufConfig(dequantize=True), installing `kernels` would make transformers keep
    the GGUF packed (GgufLinear modules, not nn.Linear) and the trainer could not wrap it."""
    spy = _LoadSpy()
    spy.install(monkeypatch)
    gguf_path = tmp_path / "base.gguf"
    gguf_path.write_bytes(b"GGUF")
    load_trainer_from_gguf(gguf_path, target_layer_indices=[3], device="cpu", dtype=torch.float32)
    _, kwargs = spy.model_calls[-1]
    qc = kwargs.get("quantization_config")
    assert isinstance(qc, transformers.GgufConfig) and qc.dequantize is True
    # Tokenizers take no quantization config.
    assert "quantization_config" not in spy.tok_calls[-1][1]


def test_non_float_or_non_linear_projection_is_rejected() -> None:
    model, tok = build_tiny_qwen35(seed=0)
    down = model.model.layers[3].mlp.down_proj
    down.weight = torch.nn.Parameter(
        torch.zeros(down.weight.shape, dtype=torch.uint8), requires_grad=False
    )
    with pytest.raises(TypeError, match="floating-point"):
        TorchKernel(model=model, tokenizer=tok, target_layer_indices=[3], device="cpu")

    class GgufLinear(torch.nn.Module):  # stand-in for transformers' packed GGUF module
        def forward(self, x: Any) -> Any:
            return x

    model2, tok2 = build_tiny_qwen35(seed=0)
    model2.model.layers[3].mlp.down_proj = GgufLinear()
    with pytest.raises(TypeError, match="expected nn.Linear"):
        TorchKernel(model=model2, tokenizer=tok2, target_layer_indices=[3], device="cpu")


def test_out_of_range_target_layer_index_raises() -> None:
    """An explicit index past the decoder stack (e.g. a GGUF MTP block) is an error, not a
    silently dropped layer."""
    with pytest.raises(ValueError, match=r"\[4\] are out of range.*4 decoder layers"):
        _make_kernel((1, 2, 3, 4), num_layers=4)
    with pytest.raises(ValueError, match="out of range"):
        _make_kernel((-1, 2), num_layers=4)
