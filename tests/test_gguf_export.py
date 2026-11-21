"""Tests for `subcor.gguf_export`: Skill Unit -> llama.cpp LoRA adapter GGUF.

Default tests are offline and tiny: they write a fake base GGUF with gguf-py, export a
registry unit, then read the adapter back with `gguf.GGUFReader` and check the exact
llama.cpp conventions (llama-adapter.cpp / convert_lora_to_gguf.py):

  * KV: general.type == "adapter", adapter.type == "lora", general.architecture copied
    from the base, adapter.lora.alpha such that the effective scale is exactly 1.0
    (llama.cpp: scale = alpha ? user_scale * alpha / rank : user_scale).
  * Tensors: "<base>.weight.lora_a" with ne = [d_in, rank] (numpy (rank, d_in)) and
    "<base>.weight.lora_b" with ne = [rank, d_out] (numpy (d_out, rank)); llama.cpp
    rejects anything else at load ("lora_a tensor is not transposed" / shape mismatch).
  * effective_scale * B @ A reconstructs the unit's verified delta.

`test_real_llama_server_loads_exported_lora_scale0_matches_base` is the single optional
real-binary integration test (CPU only); enable it with SUBCOR_LLAMA_INTEGRATION=1.
"""

from __future__ import annotations

import importlib
import json
import os
import shutil
import socket
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np
import pytest

gguf = pytest.importorskip("gguf")

from subcor.expansion import LowRankAdapter  # noqa: E402
from subcor.gguf_export import (  # noqa: E402
    canonical_to_gguf_base_tensor,
    export_unit_to_gguf_lora,
    read_gguf_architecture,
)
from subcor.types import AdaptationMode, DeltaEncoding, IntegrityError  # noqa: E402
from subcor.units import SkillUnitRegistry, build_layer_delta  # noqa: E402

# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------


def _write_base_gguf(
    path: Path,
    *,
    arch: str,
    tensors: dict[str, tuple[int, int]],
    kv_u32: dict[str, int] | None = None,
) -> Path:
    """Write a minimal base-model GGUF: architecture KV + F32 weights (numpy (d_out, d_in))."""
    writer = gguf.GGUFWriter(str(path), arch=arch)
    writer.add_block_count(4)
    for key, value in (kv_u32 or {}).items():
        writer.add_uint32(key, int(value))
    for name, (d_out, d_in) in tensors.items():
        writer.add_tensor(f"{name}.weight", np.zeros((d_out, d_in), dtype=np.float32))
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    return path


def _field_value(reader: Any, key: str) -> Any:
    field = reader.fields[key]
    part = field.parts[field.data[0]]
    if field.types and field.types[0] == gguf.GGUFValueType.STRING:
        return bytes(part).decode("utf-8")
    return part.tolist()[0] if hasattr(part, "tolist") else part


def _random_adapter(
    name: str, *, d_out: int, d_in: int, rank: int, seed: int, std: float = 0.3
) -> LowRankAdapter:
    rng = np.random.default_rng(seed)
    return LowRankAdapter(
        name=name,
        a=rng.normal(0.0, std, size=(rank, d_in)),
        b=rng.normal(0.0, std, size=(d_out, rank)),
    )


def _propose(registry: SkillUnitRegistry, unit_id: str, deltas: list[Any]) -> None:
    registry.propose(
        unit_id=unit_id,
        version="1.0.0",
        title=unit_id,
        summary="gguf export test",
        proposer="tools",
        mode=AdaptationMode.DYNAMIC_EXPANSION_STE,
        deltas=deltas,
    )


def _as_f64(tensor: Any) -> np.ndarray:
    return np.asarray(tensor.data, dtype=np.float64)


def _recon_tolerance(tensor_type: Any, ref: np.ndarray) -> float:
    scale = max(1.0, float(np.abs(ref).max()))
    if tensor_type == gguf.GGMLQuantizationType.F32:
        return 1e-5 * scale
    return 2e-2 * scale  # F16 / BF16 rounding of the factors


# Tiny shapes: hidden 64, ffn 128, attention heads*head_dim 64.
H, FF = 64, 128
DOWN = "layers.1.mlp.down_proj"
OPROJ = "layers.3.self_attn.o_proj"
BASE_TENSORS = {
    "blk.1.ffn_down": (H, FF),
    "blk.3.attn_output": (H, H),
    "blk.3.ffn_down": (H, FF),
}


@pytest.fixture
def base_gguf(tmp_path: Path) -> Path:
    return _write_base_gguf(tmp_path / "base.gguf", arch="qwen35", tensors=BASE_TENSORS)


@pytest.fixture
def mixed_unit(tmp_path: Path) -> SkillUnitRegistry:
    """Unit with a SUB1BIT layer (rank 4) and an FP32 layer (rank 2, scale 0.5): mixed ranks."""
    registry = SkillUnitRegistry(tmp_path / "units")
    sub = build_layer_delta(
        DOWN,
        _random_adapter(DOWN, d_out=H, d_in=FF, rank=4, seed=1),
        encoding=DeltaEncoding.SUB1BIT_BINARY,
        sub1bit_rank=4,
    )
    fp_adapter = _random_adapter(OPROJ, d_out=H, d_in=H, rank=2, seed=2)
    fp_adapter.scale = 0.5  # exporter must fold this into the stored factors / alpha
    fp = build_layer_delta(OPROJ, fp_adapter, encoding=DeltaEncoding.LOW_RANK_FP32)
    _propose(registry, "u-mixed", [sub, fp])
    return registry


# --------------------------------------------------------------------------------------
# Name mapping / architecture
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("canonical", "expected"),
    [
        ("layers.18.mlp.down_proj", "blk.18.ffn_down"),
        ("layers.0.mlp.gate_proj", "blk.0.ffn_gate"),
        ("layers.7.mlp.up_proj", "blk.7.ffn_up"),
        ("layers.3.self_attn.q_proj", "blk.3.attn_q"),
        ("layers.11.self_attn.k_proj", "blk.11.attn_k"),
        ("layers.23.self_attn.v_proj", "blk.23.attn_v"),
        ("layers.15.self_attn.o_proj", "blk.15.attn_output"),
        ("layers.123.mlp.down_proj", "blk.123.ffn_down"),
    ],
)
def test_canonical_to_gguf_base_tensor_maps_supported_projections(
    canonical: str, expected: str
) -> None:
    assert canonical_to_gguf_base_tensor(canonical) == expected


# Qwen3.5 GatedDeltaNet projections, as named in the real Qwen3.5-0.8B-Q4_K_M.gguf
# (blk.N.attn_qkv, attn_gate, ssm_alpha, ssm_beta, ssm_out). Officially supported.
@pytest.mark.parametrize(
    ("canonical", "expected"),
    [
        ("layers.2.linear_attn.in_proj_qkv", "blk.2.attn_qkv"),
        ("layers.2.linear_attn.in_proj_z", "blk.2.attn_gate"),
        ("layers.2.linear_attn.in_proj_a", "blk.2.ssm_alpha"),
        ("layers.2.linear_attn.in_proj_b", "blk.2.ssm_beta"),
        ("layers.2.linear_attn.out_proj", "blk.2.ssm_out"),
    ],
)
def test_linear_attention_projection_mapping(canonical: str, expected: str) -> None:
    assert canonical_to_gguf_base_tensor(canonical) == expected


@pytest.mark.parametrize(
    "bad",
    [
        "layers.3.mlp.foo_proj",
        "layers.3.self_attn.rotary_emb",
        "embed_tokens",
        "lm_head",
        "llamacpp.embedding",
        "layers.x.mlp.down_proj",
        "layers.3",
    ],
)
def test_canonical_to_gguf_base_tensor_rejects_unsupported(bad: str) -> None:
    with pytest.raises(KeyError):
        canonical_to_gguf_base_tensor(bad)


@pytest.mark.parametrize("arch", ["qwen35", "llama"])
def test_read_gguf_architecture(tmp_path: Path, arch: str) -> None:
    path = _write_base_gguf(
        tmp_path / f"{arch}.gguf", arch=arch, tensors={"blk.0.ffn_down": (4, 8)}
    )
    assert read_gguf_architecture(path) == arch
    assert read_gguf_architecture(str(path)) == arch


# --------------------------------------------------------------------------------------
# Export round trip
# --------------------------------------------------------------------------------------


def test_export_writes_llamacpp_lora_gguf_that_reconstructs_unit_delta(
    tmp_path: Path, base_gguf: Path, mixed_unit: SkillUnitRegistry
) -> None:
    out = tmp_path / "out" / "u-mixed.gguf"
    out.parent.mkdir()
    ret = export_unit_to_gguf_lora(mixed_unit, "u-mixed", out, base_gguf=base_gguf)
    assert Path(ret) == out and out.is_file()

    reader = gguf.GGUFReader(str(out))
    assert _field_value(reader, "general.type") == "adapter"
    assert _field_value(reader, "adapter.type") == "lora"
    assert _field_value(reader, "general.architecture") == "qwen35"
    alpha = float(_field_value(reader, "adapter.lora.alpha"))

    tensors = {t.name: t for t in reader.tensors}
    expected_names = set()
    for base in ("blk.1.ffn_down", "blk.3.attn_output"):
        expected_names |= {f"{base}.weight.lora_a", f"{base}.weight.lora_b"}
    assert set(tensors) == expected_names

    base_reader = gguf.GGUFReader(str(base_gguf))
    base_ne = {t.name: [int(v) for v in t.shape] for t in base_reader.tensors}

    for delta in mixed_unit.load_verified_deltas("u-mixed"):
        base = canonical_to_gguf_base_tensor(delta.layer_name)
        ta = tensors[f"{base}.weight.lora_a"]
        tb = tensors[f"{base}.weight.lora_b"]
        assert ta.tensor_type in (gguf.GGMLQuantizationType.F32, gguf.GGMLQuantizationType.F16)
        assert tb.tensor_type in (gguf.GGMLQuantizationType.F32, gguf.GGMLQuantizationType.F16)

        want = delta.to_adapter("ref").delta_matrix()  # (d_out, d_in), scale folded
        d_out, d_in = want.shape
        a = _as_f64(ta)
        b = _as_f64(tb)
        rank = a.shape[0]
        # numpy orientation == LowRankAdapter: A (rank, d_in), B (d_out, rank).
        assert a.shape == (rank, d_in)
        assert b.shape == (d_out, rank)
        # GGUF ne orientation and llama-adapter.cpp's load-time validation, verbatim:
        a_ne = [int(v) for v in ta.shape]
        b_ne = [int(v) for v in tb.shape]
        assert a_ne == [d_in, rank] and b_ne == [rank, d_out]
        w_ne = base_ne[f"{base}.weight"]
        assert w_ne[0] == a_ne[0], "model_tensor->ne[0] != a->ne[0]"
        assert w_ne[1] == b_ne[1], "model_tensor->ne[1] != b->ne[1]"
        assert a_ne[1] == b_ne[0], "lora_a tensor is not transposed"

        # Effective llama.cpp scale (user scale 1.0) must be exactly 1.0 for every tensor,
        # including when ranks differ across layers (rank 4 sub-1-bit vs rank 2 fp32 here).
        effective = alpha / rank if alpha else 1.0
        assert effective == pytest.approx(1.0, abs=1e-6), (alpha, rank)
        got = effective * (b @ a)
        np.testing.assert_allclose(got, want, atol=_recon_tolerance(ta.tensor_type, want))

        if delta.encoding is DeltaEncoding.SUB1BIT_BINARY:
            # Exact for the stored sub-1-bit factors (not a re-fit of something else).
            assert delta.sub1bit is not None
            np.testing.assert_allclose(
                got,
                delta.sub1bit.reconstruct(),
                atol=_recon_tolerance(ta.tensor_type, want),
            )
        else:
            # FP32 unit scale 0.5 is folded in: plain B@A of the payload would be 2x too big.
            assert delta.a_fp32 is not None and delta.b_fp32 is not None
            raw = delta.b_fp32.astype(np.float64) @ delta.a_fp32.astype(np.float64)
            np.testing.assert_allclose(got, 0.5 * raw, atol=1e-5 * max(1.0, np.abs(raw).max()))


def _export_and_read(
    registry: SkillUnitRegistry, unit_id: str, out: Path, base: Path
) -> tuple[float, dict[str, tuple[np.ndarray, np.ndarray]]]:
    """Export and return (alpha, {base_tensor: (A (rank, d_in), B (d_out, rank))})."""
    export_unit_to_gguf_lora(registry, unit_id, out, base_gguf=base)
    reader = gguf.GGUFReader(str(out))
    alpha = float(_field_value(reader, "adapter.lora.alpha"))
    by_name = {t.name: _as_f64(t) for t in reader.tensors}
    pairs: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for name in by_name:
        if name.endswith(".weight.lora_a"):
            base_name = name[: -len(".weight.lora_a")]
            pairs[base_name] = (by_name[name], by_name[f"{base_name}.weight.lora_b"])
    return alpha, pairs


@pytest.mark.parametrize("encoding", [DeltaEncoding.LOW_RANK_FP32, DeltaEncoding.SUB1BIT_BINARY])
def test_uniform_rank_unit_exports_alpha_equal_rank_and_unit_scale(
    tmp_path: Path, base_gguf: Path, encoding: DeltaEncoding
) -> None:
    """Production units use ONE rank for every layer: alpha must equal that rank exactly."""
    registry = SkillUnitRegistry(tmp_path / "units")
    rank = 3
    deltas = []
    for name, (d_out, d_in), seed in ((DOWN, (H, FF), 5), (OPROJ, (H, H), 6)):
        adapter = _random_adapter(name, d_out=d_out, d_in=d_in, rank=rank, seed=seed)
        if encoding is DeltaEncoding.SUB1BIT_BINARY:
            deltas.append(build_layer_delta(name, adapter, encoding=encoding, sub1bit_rank=rank))
        else:
            deltas.append(build_layer_delta(name, adapter, encoding=encoding))
    _propose(registry, "u-uni", deltas)
    alpha, pairs = _export_and_read(registry, "u-uni", tmp_path / "uni.gguf", base_gguf)
    assert set(pairs) == {"blk.1.ffn_down", "blk.3.attn_output"}
    assert alpha == float(rank)
    for delta in registry.load_verified_deltas("u-uni"):
        a, b = pairs[canonical_to_gguf_base_tensor(delta.layer_name)]
        assert a.shape[0] == b.shape[1] == rank
        effective = alpha / a.shape[0]  # llama.cpp: scale = user_scale * alpha / rank
        assert effective == 1.0
        want = delta.to_adapter("ref").delta_matrix()
        np.testing.assert_allclose(effective * (b @ a), want, atol=1e-5 * max(1.0, abs(want).max()))


# Reference re-implementations of llama.cpp's convert_hf_to_gguf.py permutations, written
# as explicit index loops (independent of the exporter's reshape/transpose code).
def _ref_llama_permute_rows(w: np.ndarray, n_head: int) -> np.ndarray:
    """LlamaModel.permute: per head, HF rows [first half | second half] -> interleaved."""
    head_dim = w.shape[0] // n_head
    half = head_dim // 2
    out = np.empty_like(w)
    for h in range(n_head):
        for i in range(half):
            for j in range(2):
                out[h * head_dim + 2 * i + j] = w[h * head_dim + j * half + i]
    return out


def _ref_grouped_to_tiled(n_k: int, v_per_k: int, head_dim: int) -> list[int]:
    """_reorder_v_heads: tiled position (v, k, d) holds grouped row (k, v, d)."""
    return [
        (k * v_per_k + v) * head_dim + d
        for v in range(v_per_k)
        for k in range(n_k)
        for d in range(head_dim)
    ]


def _fp32_unit(registry: SkillUnitRegistry, unit_id: str, shapes: dict[str, tuple[int, int]]):
    deltas = [
        build_layer_delta(
            name,
            _random_adapter(name, d_out=d_out, d_in=d_in, rank=2, seed=10 + i),
            encoding=DeltaEncoding.LOW_RANK_FP32,
        )
        for i, (name, (d_out, d_in)) in enumerate(shapes.items())
    ]
    _propose(registry, unit_id, deltas)
    return {d.layer_name: d.to_adapter("ref").delta_matrix() for d in deltas}


def test_llama_q_k_deltas_are_rotary_permuted_into_gguf_layout(tmp_path: Path) -> None:
    n_head, n_kv, head_dim, hidden = 4, 2, 8, 32
    base = _write_base_gguf(
        tmp_path / "llama.gguf",
        arch="llama",
        tensors={
            "blk.0.attn_q": (n_head * head_dim, hidden),
            "blk.0.attn_k": (n_kv * head_dim, hidden),
            "blk.0.attn_v": (n_kv * head_dim, hidden),
            "blk.0.ffn_down": (hidden, 2 * hidden),
        },
        kv_u32={"llama.attention.head_count": n_head, "llama.attention.head_count_kv": n_kv},
    )
    registry = SkillUnitRegistry(tmp_path / "units")
    dense = _fp32_unit(
        registry,
        "u-llama",
        {
            "layers.0.self_attn.q_proj": (n_head * head_dim, hidden),
            "layers.0.self_attn.k_proj": (n_kv * head_dim, hidden),
            "layers.0.self_attn.v_proj": (n_kv * head_dim, hidden),
            "layers.0.mlp.down_proj": (hidden, 2 * hidden),
        },
    )
    alpha, pairs = _export_and_read(registry, "u-llama", tmp_path / "l.gguf", base)
    assert alpha == 2.0

    def got(base_name: str) -> np.ndarray:
        a, b = pairs[base_name]
        return (alpha / a.shape[0]) * (b @ a)

    tol = {"atol": 1e-5}
    q_ref = _ref_llama_permute_rows(dense["layers.0.self_attn.q_proj"], n_head)
    k_ref = _ref_llama_permute_rows(dense["layers.0.self_attn.k_proj"], n_kv)  # head_count_kv
    np.testing.assert_allclose(got("blk.0.attn_q"), q_ref, **tol)
    np.testing.assert_allclose(got("blk.0.attn_k"), k_ref, **tol)
    assert not np.allclose(q_ref, dense["layers.0.self_attn.q_proj"])  # permute is not a no-op
    # v_proj and down_proj are never permuted.
    np.testing.assert_allclose(got("blk.0.attn_v"), dense["layers.0.self_attn.v_proj"], **tol)
    np.testing.assert_allclose(got("blk.0.ffn_down"), dense["layers.0.mlp.down_proj"], **tol)


def _qwen35_linear_base(path: Path, *, n_k: int, n_v: int, hk: int, hv: int, hidden: int) -> Path:
    return _write_base_gguf(
        path,
        arch="qwen35",
        tensors={
            "blk.0.attn_qkv": (2 * hk * n_k + n_v * hv, hidden),
            "blk.0.attn_gate": (n_v * hv, hidden),
            "blk.0.ssm_alpha": (n_v, hidden),
            "blk.0.ssm_beta": (n_v, hidden),
            "blk.0.ssm_out": (hidden, n_v * hv),
        },
        kv_u32={
            "qwen35.ssm.group_count": n_k,
            "qwen35.ssm.time_step_rank": n_v,
            "qwen35.ssm.inner_size": n_v * hv,
            "qwen35.ssm.state_size": hk,
        },
    )


def _qwen35_linear_shapes(n_k: int, n_v: int, hk: int, hv: int, hidden: int):
    return {
        "layers.0.linear_attn.in_proj_qkv": (2 * hk * n_k + n_v * hv, hidden),
        "layers.0.linear_attn.in_proj_z": (n_v * hv, hidden),
        "layers.0.linear_attn.in_proj_a": (n_v, hidden),
        "layers.0.linear_attn.in_proj_b": (n_v, hidden),
        "layers.0.linear_attn.out_proj": (hidden, n_v * hv),
    }


_QWEN35_LEAVES = {
    "in_proj_qkv": "blk.0.attn_qkv",
    "in_proj_z": "blk.0.attn_gate",
    "in_proj_a": "blk.0.ssm_alpha",
    "in_proj_b": "blk.0.ssm_beta",
    "out_proj": "blk.0.ssm_out",
}


@pytest.mark.parametrize(("n_k", "n_v"), [(2, 4), (2, 2)])
def test_qwen35_linear_attention_value_heads_grouped_to_tiled(
    tmp_path: Path, n_k: int, n_v: int
) -> None:
    """num_k_heads != num_v_heads (larger Qwen3.5 sizes): value heads are reordered from
    the HF grouped layout to llama.cpp's tiled layout. Equal counts (0.8B): unchanged."""
    hk, hv, hidden = 4, 3, 16
    base = _qwen35_linear_base(tmp_path / "q.gguf", n_k=n_k, n_v=n_v, hk=hk, hv=hv, hidden=hidden)
    registry = SkillUnitRegistry(tmp_path / "units")
    dense = _fp32_unit(registry, "u-la", _qwen35_linear_shapes(n_k, n_v, hk, hv, hidden))
    alpha, pairs = _export_and_read(registry, "u-la", tmp_path / "la.gguf", base)
    assert alpha == 2.0

    v_per_k = n_v // n_k
    head_idx = _ref_grouped_to_tiled(n_k, v_per_k, hv)
    scalar_idx = _ref_grouped_to_tiled(n_k, v_per_k, 1)
    qk_rows = 2 * hk * n_k
    want = {}
    d = {k.rsplit(".", 1)[1]: v for k, v in dense.items()}
    want["in_proj_qkv"] = np.concatenate(
        [d["in_proj_qkv"][:qk_rows], d["in_proj_qkv"][qk_rows:][head_idx]]
    )
    want["in_proj_z"] = d["in_proj_z"][head_idx]
    want["in_proj_a"] = d["in_proj_a"][scalar_idx]
    want["in_proj_b"] = d["in_proj_b"][scalar_idx]
    want["out_proj"] = d["out_proj"][:, head_idx]

    for proj, base_name in _QWEN35_LEAVES.items():
        a, b = pairs[base_name]
        got = (alpha / a.shape[0]) * (b @ a)
        np.testing.assert_allclose(got, want[proj], atol=1e-5, err_msg=proj)
        if n_k == n_v:
            np.testing.assert_allclose(got, d[proj], atol=1e-5, err_msg=proj)
        else:
            assert not np.allclose(got, d[proj]), f"{proj}: reorder must not be a no-op"


def test_export_copies_architecture_from_base(
    tmp_path: Path, mixed_unit: SkillUnitRegistry
) -> None:
    base = _write_base_gguf(tmp_path / "llama-base.gguf", arch="llama", tensors=BASE_TENSORS)
    out = export_unit_to_gguf_lora(mixed_unit, "u-mixed", tmp_path / "x.gguf", base_gguf=base)
    reader = gguf.GGUFReader(str(out))
    assert _field_value(reader, "general.architecture") == "llama"


def test_export_rejects_tampered_payload(
    tmp_path: Path, base_gguf: Path, mixed_unit: SkillUnitRegistry
) -> None:
    payload = mixed_unit.root / "u-mixed" / "payload.scun"
    raw = bytearray(payload.read_bytes())
    raw[-1] ^= 0xFF
    payload.write_bytes(bytes(raw))
    out = tmp_path / "tampered.gguf"
    with pytest.raises(IntegrityError):
        export_unit_to_gguf_lora(mixed_unit, "u-mixed", out, base_gguf=base_gguf)
    assert not out.exists() or out.stat().st_size == 0


def test_export_rejects_unknown_unit_and_unsupported_layer(tmp_path: Path, base_gguf: Path) -> None:
    registry = SkillUnitRegistry(tmp_path / "units")
    with pytest.raises(FileNotFoundError):
        export_unit_to_gguf_lora(registry, "nope", tmp_path / "nope.gguf", base_gguf=base_gguf)

    bad = build_layer_delta(
        "llamacpp.embedding",
        _random_adapter("e", d_out=8, d_in=8, rank=2, seed=0),
        encoding=DeltaEncoding.LOW_RANK_FP32,
    )
    _propose(registry, "u-unsupported", [bad])
    with pytest.raises(KeyError):
        export_unit_to_gguf_lora(
            registry, "u-unsupported", tmp_path / "bad.gguf", base_gguf=base_gguf
        )


def test_export_missing_base_gguf_raises(tmp_path: Path, mixed_unit: SkillUnitRegistry) -> None:
    with pytest.raises((FileNotFoundError, OSError, ValueError)):
        export_unit_to_gguf_lora(
            mixed_unit, "u-mixed", tmp_path / "o.gguf", base_gguf=tmp_path / "missing.gguf"
        )


def test_module_imports_without_gguf_and_export_raises_clear_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, base_gguf: Path, mixed_unit: SkillUnitRegistry
) -> None:
    """gguf-py is imported lazily: the module loads without it; export fails with a clear error."""
    import subcor.gguf_export as ge

    monkeypatch.setitem(sys.modules, "gguf", None)  # any `import gguf` now raises ImportError
    try:
        reloaded = importlib.reload(ge)
        assert reloaded.canonical_to_gguf_base_tensor(DOWN) == "blk.1.ffn_down"
        with pytest.raises((ImportError, RuntimeError)) as excinfo:
            reloaded.export_unit_to_gguf_lora(
                mixed_unit, "u-mixed", tmp_path / "o.gguf", base_gguf=base_gguf
            )
        assert "gguf" in str(excinfo.value).lower()
    finally:
        monkeypatch.undo()
        importlib.reload(ge)


# --------------------------------------------------------------------------------------
# Optional real llama-server integration (CPU only)
# --------------------------------------------------------------------------------------

_DEFAULT_REAL_GGUF = (
    Path.home()
    / ".cache/huggingface/hub/models--unsloth--Qwen3.5-0.8B-GGUF/snapshots"
    / "6ab461498e2023f6e3c1baea90a8f0fe38ab64d0/Qwen3.5-0.8B-Q4_K_M.gguf"
)


def _free_port_in(candidates: range) -> int | None:
    for port in candidates:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("127.0.0.1", port))
            except OSError:
                continue
            return port
    return None


def _http_json(url: str, payload: Any | None = None, timeout: float = 300.0) -> Any:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="GET" if payload is None else "POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _complete(base_url: str, prompt: str, **extra: Any) -> str:
    body = {
        "prompt": prompt,
        "n_predict": 12,
        "temperature": 0.0,
        "top_k": 1,
        "seed": 0,
        "cache_prompt": False,
        **extra,
    }
    return str(_http_json(f"{base_url}/completion", body)["content"])


def _rewrite_lora_alpha(src: Path, dst: Path, *, alpha: float) -> Path:
    """Copy a LoRA GGUF's tensors into a new adapter GGUF with a different alpha."""
    reader = gguf.GGUFReader(str(src))
    writer = gguf.GGUFWriter(str(dst), str(_field_value(reader, "general.architecture")))
    writer.add_type(gguf.GGUFType.ADAPTER)
    writer.add_string(gguf.Keys.Adapter.TYPE, "lora")
    writer.add_float32(gguf.Keys.Adapter.LORA_ALPHA, float(alpha))
    for t in reader.tensors:
        writer.add_tensor(t.name, np.array(t.data, dtype=np.float32))
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    return dst


def _server_pid(sup: Any) -> int:
    pid = getattr(sup, "pid", None)
    if pid is None:
        proc = getattr(sup, "_proc", None)
        pid = getattr(proc, "pid", None)
    assert isinstance(pid, int), "cannot determine llama-server pid from supervisor"
    return pid


def _pid_alive(pid: int) -> bool:
    try:
        stat = Path(f"/proc/{pid}/stat").read_text()
    except FileNotFoundError:
        return False
    return stat.rsplit(")", 1)[1].split()[0] != "Z"


@pytest.mark.skipif(
    os.environ.get("SUBCOR_LLAMA_INTEGRATION") != "1",
    reason="set SUBCOR_LLAMA_INTEGRATION=1 to run the real llama-server integration test",
)
def test_real_llama_server_loads_exported_lora_scale0_matches_base(tmp_path: Path) -> None:
    from subcor.backends.llamacpp_supervisor import LlamaServerSupervisor

    binary = os.environ.get("SUBCOR_LLAMA_SERVER_BIN") or shutil.which("llama-server")
    base = Path(os.environ.get("SUBCOR_LLAMA_BASE_GGUF", str(_DEFAULT_REAL_GGUF)))
    if not binary or not base.is_file():
        pytest.skip(f"llama-server ({binary}) or base GGUF ({base}) unavailable")
    port = _free_port_in(range(8095, 8100))
    if port is None:
        pytest.skip("no free port in 8095-8099")

    # Unit on the real Qwen3.5-0.8B surface: a large sub-1-bit ffn_down delta plus an FP32
    # attn_q delta (full-attention layer 3). Large magnitudes guarantee a visible effect.
    registry = SkillUnitRegistry(tmp_path / "units")
    down = build_layer_delta(
        "layers.18.mlp.down_proj",
        _random_adapter("d", d_out=1024, d_in=3584, rank=4, seed=11, std=0.5),
        encoding=DeltaEncoding.SUB1BIT_BINARY,
        sub1bit_rank=4,
    )
    q = build_layer_delta(
        "layers.3.self_attn.q_proj",
        _random_adapter("q", d_out=4096, d_in=1024, rank=2, seed=12, std=0.5),
        encoding=DeltaEncoding.LOW_RANK_FP32,
    )
    _propose(registry, "u-int", [down, q])
    lora_path = export_unit_to_gguf_lora(registry, "u-int", tmp_path / "u-int.gguf", base_gguf=base)
    # The same two layers as separate uniform-rank units (alpha = rank branch).
    _propose(registry, "u-down", [down])
    _propose(registry, "u-q", [q])
    down_path = export_unit_to_gguf_lora(
        registry, "u-down", tmp_path / "u-down.gguf", base_gguf=base
    )
    q_path = export_unit_to_gguf_lora(registry, "u-q", tmp_path / "u-q.gguf", base_gguf=base)
    # Production-shaped unit: every layer SUB1BIT at one rank (alpha = rank branch)...
    uni = [
        build_layer_delta(
            f"layers.{i}.mlp.down_proj",
            _random_adapter(f"u{i}", d_out=1024, d_in=3584, rank=4, seed=20 + i, std=0.3),
            encoding=DeltaEncoding.SUB1BIT_BINARY,
            sub1bit_rank=4,
        )
        for i in (19, 21)
    ]
    _propose(registry, "u-uni", uni)
    uni_path = export_unit_to_gguf_lora(registry, "u-uni", tmp_path / "u-uni.gguf", base_gguf=base)
    # ...and an independent reference: the same tensors with alpha = 0, for which llama.cpp
    # applies exactly the requested scale (no alpha/rank factor at all).
    ref_path = _rewrite_lora_alpha(uni_path, tmp_path / "u-uni-ref.gguf", alpha=0.0)

    threads = str(max(1, min(8, (os.cpu_count() or 2) // 2)))
    sup = LlamaServerSupervisor(
        base_gguf=base,
        binary=binary,
        host="127.0.0.1",
        port=port,
        n_gpu_layers=0,  # GPU discipline: CPU only
        ctx_size=1024,
        extra_args=("--device", "none", "-t", threads),
        lora_dir=tmp_path / "loras",
    )
    prompt = "The capital of France is"
    pids: list[int] = []
    try:
        sup.start()
        pids.append(_server_pid(sup))
        url = sup.base_url
        base_text = _complete(url, prompt)
        base_emb = np.asarray(
            _http_json(f"{url}/v1/embeddings", {"input": prompt})["data"][0]["embedding"]
        )
        assert base_emb.shape == (1024,)

        adapters = {
            "u-int": Path(lora_path),
            "u-down": Path(down_path),
            "u-q": Path(q_path),
            "u-uni": Path(uni_path),
            "u-ref": Path(ref_path),
        }
        ids = sup.restart(adapters)
        pids.append(_server_pid(sup))
        assert not _pid_alive(pids[0]), "restart() left the previous llama-server running"
        assert set(ids) == set(adapters)
        lora_id = ids["u-int"]
        listed = _http_json(f"{url}/lora-adapters")
        assert sorted(int(e["id"]) for e in listed) == sorted(ids.values())
        # Loaded but NOT applied: every adapter's global scale must start at 0.
        assert all(float(e["scale"]) == 0.0 for e in listed), listed

        assert _complete(url, prompt) == base_text  # scale 0 == base output
        emb0 = np.asarray(
            _http_json(f"{url}/v1/embeddings", {"input": prompt})["data"][0]["embedding"]
        )
        np.testing.assert_allclose(emb0, base_emb, atol=1e-4)

        # Positive control: the adapter really is applied at scale 1 (per-request selection)...
        lora_text = _complete(url, prompt, lora=[{"id": lora_id, "scale": 1.0}])
        assert lora_text != base_text
        # ...and does not leak into the next request.
        assert _complete(url, prompt) == base_text

        # Quantitative scale check (embeddings use the GLOBAL scales).
        def emb_with(scales: dict[str, float]) -> np.ndarray:
            _http_json(
                f"{url}/lora-adapters",
                [{"id": ids[k], "scale": float(scales.get(k, 0.0))} for k in ids],
            )
            try:
                out = _http_json(f"{url}/v1/embeddings", {"input": prompt})
            finally:
                _http_json(f"{url}/lora-adapters", [{"id": i, "scale": 0.0} for i in ids.values()])
            return np.asarray(out["data"][0]["embedding"], dtype=np.float64)

        for scale in (1.0, 0.5):
            e_uni = emb_with({"u-uni": scale})
            e_ref = emb_with({"u-ref": scale})
            assert np.abs(e_uni - base_emb).max() > 1e-3, "uniform unit had no effect"
            # alpha = rank exported for a uniform-rank unit == exactly the requested scale.
            np.testing.assert_allclose(e_uni, e_ref, atol=1e-5, rtol=0)
        # Mixed ranks (alpha = 0) == the same layers as two uniform-rank units applied together.
        e_mixed = emb_with({"u-int": 1.0})
        e_split = emb_with({"u-down": 1.0, "u-q": 1.0})
        assert np.abs(e_mixed - base_emb).max() > 1e-3
        np.testing.assert_allclose(e_mixed, e_split, atol=1e-5, rtol=0)
        # Same through the per-request field used for routed chat.
        uni_req = _complete(url, prompt, lora=[{"id": ids["u-uni"], "scale": 1.0}])
        ref_req = _complete(url, prompt, lora=[{"id": ids["u-ref"], "scale": 1.0}])
        assert uni_req == ref_req
        listed = _http_json(f"{url}/lora-adapters")
        assert all(float(e["scale"]) == 0.0 for e in listed), listed
    finally:
        sup.stop()
    deadline = time.monotonic() + 15
    while any(_pid_alive(p) for p in pids) and time.monotonic() < deadline:
        time.sleep(0.2)
    assert not any(_pid_alive(p) for p in pids)
    with pytest.raises((urllib.error.URLError, ConnectionError, OSError)):
        _http_json(f"http://127.0.0.1:{port}/health", timeout=2)
