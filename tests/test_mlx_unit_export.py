"""Fast MLX backend unit tests (runs on tiny synthetic MLX layers; skips if mlx absent)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
nn = pytest.importorskip("mlx.nn")

from slaingi.backends.mlx_kernel import SlaingiMLXLinear, export_mlx_lora_safetensors  # noqa: E402
from slaingi.expansion import LowRankAdapter  # noqa: E402
from slaingi.types import AdaptationMode, DeltaEncoding  # noqa: E402
from slaingi.units import SkillUnitRegistry, build_layer_delta  # noqa: E402


def test_slaingi_mlx_linear_zero_perturbation_recording_and_rollback() -> None:
    base_lin = nn.Linear(64, 32, bias=False)
    q_lin = nn.QuantizedLinear.from_linear(base_lin, group_size=32, bits=4)
    mx.eval(q_lin.parameters())

    wrapped = SlaingiMLXLinear(q_lin, layer_name="layers.0.mlp.down_proj")
    assert wrapped.shape == (32, 64)

    x = mx.random.normal((2, 5, 64))
    y_base = wrapped(x)
    mx.eval(y_base)

    # Zero-initialized trainable LoRA produces exact zero output perturbation.
    wrapped.attach_trainable_lora(rank=4, scale=2.0, seed=7)
    y_expanded = wrapped(x)
    mx.eval(y_expanded)
    assert float(mx.max(mx.abs(y_expanded - y_base)).item()) == 0.0
    wrapped.detach_trainable_lora()

    # Activation recording captures exact mean vector across batch & seq dims.
    wrapped.start_recording()
    y_rec = wrapped(x)
    mx.eval(y_rec)
    mean_act = wrapped.stop_recording()
    assert mean_act is not None
    assert mean_act.shape == (32,)
    expected_mean = np.mean(np.array(y_rec.astype(mx.float32)), axis=(0, 1))
    np.testing.assert_allclose(mean_act, expected_mean, atol=1e-4)

    # Mounting & unmounting a Sub-1-Bit adapter modifies output then restores exact base output.
    rng = np.random.default_rng(11)
    adapter = LowRankAdapter(
        name="test-delta",
        a=rng.normal(0.0, 0.2, size=(2, 64)),
        b=rng.normal(0.0, 0.2, size=(32, 2)),
    )
    wrapped.mount_adapter("unit:u1", adapter)
    y_mounted = wrapped(x)
    mx.eval(y_mounted)
    assert float(mx.max(mx.abs(y_mounted - y_base)).item()) > 1e-3

    # Bypassing adapters for frozen-base routing tap reproduces exact base output.
    wrapped._bypass_adapters = True
    y_bypassed = wrapped(x)
    mx.eval(y_bypassed)
    wrapped._bypass_adapters = False
    assert float(mx.max(mx.abs(y_bypassed - y_base)).item()) == 0.0

    wrapped.unmount_adapter("unit:u1")
    y_restored = wrapped(x)
    mx.eval(y_restored)
    assert float(mx.max(mx.abs(y_restored - y_base)).item()) == 0.0


def test_export_mlx_lora_safetensors(tmp_path: Path) -> None:
    registry = SkillUnitRegistry(tmp_path / "units")
    rng = np.random.default_rng(15)
    adapter = LowRankAdapter(
        name="a",
        a=rng.normal(0.0, 0.3, size=(4, 64)),
        b=rng.normal(0.0, 0.3, size=(64, 4)),
    )
    delta = build_layer_delta(
        "layers.15.mlp.down_proj",
        adapter,
        encoding=DeltaEncoding.SUB1BIT_BINARY,
        sub1bit_rank=4,
    )
    registry.propose(
        unit_id="u-mlx",
        version="1.0.0",
        title="Test MLX Export",
        summary="Verify safetensors export",
        proposer="tools",
        mode=AdaptationMode.WEIGHT_SPACE_SURGERY,
        deltas=[delta],
    )
    out_dir = export_mlx_lora_safetensors("u-mlx", registry, tmp_path / "st")
    assert (out_dir / "adapters.safetensors").exists()
    assert (out_dir / "adapter_config.json").exists()

    loaded = mx.load(str(out_dir / "adapters.safetensors"))
    assert "model.layers.15.mlp.down_proj.lora_a" in loaded
    assert "model.layers.15.mlp.down_proj.lora_b" in loaded
    assert loaded["model.layers.15.mlp.down_proj.lora_a"].shape == (64, 4)
    assert loaded["model.layers.15.mlp.down_proj.lora_b"].shape == (4, 64)
