"""Unit tests for Issue #30: pure-PyTorch packed 4-bit on-the-fly QLoRA base weights."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from torch import nn  # noqa: E402

from subcor.backends.torch_kernel import (  # noqa: E402
    QuantizedFrozenLinear,
    SubcorTorchLinear,
    TorchKernel,
    quantize_model_linears_4bit,
)
from subcor.types import AdaptationMode, DeltaEncoding  # noqa: E402
from subcor.units import SkillUnitRegistry, build_layer_delta  # noqa: E402
from torch_tiny import build_tiny_qwen35  # noqa: E402


@pytest.mark.parametrize("in_features", [128, 100])
@pytest.mark.parametrize("use_bias", [False, True])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_quantized_frozen_linear_forward_and_dx_parity(
    in_features: int,
    use_bias: bool,
    dtype: torch.dtype,
) -> None:
    """QuantizedFrozenLinear matches explicit dequantized nn.Linear on forward & dx."""
    torch.manual_seed(42)
    out_features = 96
    dense = nn.Linear(in_features, out_features, bias=use_bias).to(dtype=dtype)
    qmod = QuantizedFrozenLinear.from_linear(dense, group_size=64, compute_dtype=dtype)

    # Strictly frozen: zero trainable parameters & real memory savings over FP16/FP32.
    assert list(qmod.parameters()) == []
    dense_bytes = dense.weight.numel() * dense.weight.element_size()
    if dtype == torch.float16:
        assert qmod.nbytes_packed < dense_bytes * 0.45
    else:
        assert qmod.nbytes_packed < dense_bytes * 0.25

    q_before = qmod.qweight.clone()
    s_before = qmod.scales.clone()
    z_before = qmod.zeros.clone()

    # Build explicit dequantized nn.Linear reference from the exact packed weights.
    ref = nn.Linear(in_features, out_features, bias=use_bias).to(dtype=dtype)
    with torch.no_grad():
        ref.weight.copy_(qmod.dequantize_weight(dtype=dtype))
        if use_bias and qmod.bias is not None:
            ref.bias.copy_(qmod.bias.to(dtype=dtype))
    ref.weight.requires_grad_(False)
    if ref.bias is not None:
        ref.bias.requires_grad_(False)

    x_q = (torch.randn(2, 5, in_features, dtype=dtype) * 0.25).requires_grad_(True)
    x_r = x_q.detach().clone().requires_grad_(True)

    y_q = qmod(x_q)
    y_r = ref(x_r)
    tol = 1e-3 if dtype == torch.float16 else 1e-5
    assert float((y_q - y_r).abs().max().item()) < tol

    grad_out = torch.randn_like(y_q) * 0.1
    y_q.backward(grad_out)
    y_r.backward(grad_out)
    assert x_q.grad is not None and x_r.grad is not None
    assert float((x_q.grad - x_r.grad).abs().max().item()) < tol

    # Base buffers must be bit-identical (0.0 mutation).
    assert torch.equal(qmod.qweight, q_before)
    assert torch.equal(qmod.scales, s_before)
    assert torch.equal(qmod.zeros, z_before)


def test_subcor_torch_linear_over_quantized_base_zero_perturb_and_rollback() -> None:
    """SubcorTorchLinear over QuantizedFrozenLinear preserves 0.0 init perturb & atomic rollback."""
    torch.manual_seed(7)
    dense = nn.Linear(128, 64, bias=False)
    qbase = QuantizedFrozenLinear.from_linear(dense, group_size=64, compute_dtype=torch.float32)
    wrapped = SubcorTorchLinear(qbase, layer_name="layers.0.mlp.down_proj")
    assert wrapped.shape == (64, 128)
    assert wrapped.dequantize_base_weight().shape == (64, 128)

    x = torch.randn(1, 4, 128, dtype=torch.float32)
    y_base = wrapped(x).detach().clone()

    # Zero-initialized LoRA branch causes exact 0.0 perturbation.
    wrapped.attach_trainable_lora(rank=4, scale=1.0, binary_ste=False, seed=11)
    y_attach = wrapped(x).detach()
    assert float((y_attach - y_base).abs().max().item()) == 0.0

    # Backprop flows cleanly into LoRA params without mutating packed base buffers.
    q_snap = qbase.qweight.clone()
    x_in = x.clone().requires_grad_(True)
    loss = wrapped(x_in).pow(2).mean()
    loss.backward()
    assert wrapped.lora_b is not None and wrapped.lora_b.grad is not None
    assert float(wrapped.lora_b.grad.norm().item()) > 0.0
    assert x_in.grad is not None and float(x_in.grad.norm().item()) > 0.0
    assert torch.equal(qbase.qweight, q_snap)

    # Extract, compress, mount, verify delta effect, unmount -> exact 0.0 rollback.
    with torch.no_grad():
        wrapped.lora_b.add_(0.05)
    adapter = wrapped.extract_trainable_lora("trained:test")
    wrapped.detach_trainable_lora()
    assert float((wrapped(x).detach() - y_base).abs().max().item()) == 0.0

    wrapped.mount_adapter("u1", adapter)
    assert float((wrapped(x).detach() - y_base).abs().max().item()) > 1e-4
    wrapped.unmount_adapter("u1")
    assert float((wrapped(x).detach() - y_base).abs().max().item()) == 0.0


def test_torch_kernel_quantized_4bit_multiturn_training_and_rollback(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """End-to-end 4-bit QLoRA TorchKernel training on tiny hybrid Qwen3.5 + multi-turn masking."""
    model, tok = build_tiny_qwen35(seed=3, num_layers=4)
    kernel = TorchKernel(
        "tiny-qwen35",
        model=model,
        tokenizer=tok,
        target_layer_indices=[2, 3],
        target_projections=("mlp.down_proj",),
        quantize_base_weights=True,
        quantize_group_size=32,
        device="cpu",
        dtype=torch.float32,
    )

    assert kernel.quantization_stats["quantized_layers"] > 0
    assert kernel.quantization_stats["packed_bytes"] < kernel.quantization_stats["dense_bytes"]
    assert isinstance(kernel.model.lm_head, nn.Linear)
    for lyr in kernel.instrumented.values():
        assert isinstance(lyr.base, QuantizedFrozenLinear)

    # Multi-turn trajectories: one valid with trailing tool turn (must be trimmed & kept!),
    # one oversized (must be dropped & logged!).
    traj_valid = {
        "messages": [
            {"role": "system", "content": "Be brief."},
            {"role": "user", "content": "ping"},
            {"role": "assistant", "content": "pong"},
            {"role": "tool", "name": "noop", "content": "x" * 200},
        ]
    }
    traj_too_long = {
        "messages": [
            {"role": "user", "content": "y" * 260},
            {"role": "assistant", "content": "ok"},
        ]
    }
    layer_names = sorted(kernel.instrumented.keys())
    with caplog.at_level(logging.WARNING, logger="subcor.backends.torch_kernel"):
        adapters, losses, zero_diff = kernel.train_qlora_adapters(
            [traj_valid, traj_too_long],
            layer_names=layer_names,
            rank=4,
            steps=4,
            grad_accum=2,
            lr=1e-3,
            max_seq_len=160,
            seed=42,
        )

    assert zero_diff == 0.0
    assert len(losses) == 4 and all(np.isfinite(losses))
    assert kernel.last_train_stats["kept"] == 1
    assert kernel.last_train_stats["dropped_too_long"] == 1
    assert kernel.last_train_stats["quantize_base_weights"] is True
    assert "dropped_too_long=1" in caplog.text
    for nm in layer_names:
        assert float(np.linalg.norm(adapters[nm].delta_matrix())) > 0.0

    # Package Sub-1-Bit unit, mount, verify rollback is bit-identical (0.0 diff).
    reg = SkillUnitRegistry(tmp_path / "units")
    deltas = [
        build_layer_delta(nm, adapters[nm], encoding=DeltaEncoding.SUB1BIT_BINARY, sub1bit_rank=4)
        for nm in layer_names
    ]
    reg.propose(
        unit_id="q4-unit",
        version="1.0.0",
        title="4-bit QLoRA Unit",
        summary="Trained over QuantizedFrozenLinear",
        proposer="tools",
        mode=AdaptationMode.DYNAMIC_EXPANSION_FP32,
        deltas=deltas,
        base_kernel_id="tiny-qwen35",
    )
    base_logits = kernel.probe_logits("ping", system_prompt="Be brief.", enable_thinking=False)
    kernel.mount_unit("q4-unit", reg)
    mounted_logits = kernel.probe_logits("ping", system_prompt="Be brief.", enable_thinking=False)
    assert float(np.max(np.abs(mounted_logits - base_logits))) > 1e-6
    kernel.unmount_unit("q4-unit")
    restored_logits = kernel.probe_logits("ping", system_prompt="Be brief.", enable_thinking=False)
    assert float(np.max(np.abs(restored_logits - base_logits))) == 0.0


def test_quantize_model_linears_4bit_skips_lm_head() -> None:
    model, _ = build_tiny_qwen35(seed=5, num_layers=2)
    n_rep, d_bytes, p_bytes = quantize_model_linears_4bit(model, group_size=32)
    assert n_rep > 0
    assert p_bytes < d_bytes * 0.3
    assert isinstance(model.lm_head, nn.Linear)
