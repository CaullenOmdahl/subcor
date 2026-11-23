"""Export governed Skill Units as llama.cpp-compatible GGUF LoRA adapters.

A Skill Unit stores one low-rank delta per instrumented projection, keyed by the
canonical SUBCOR layer name (``layers.{i}.{proj}``, identical across the MLX and
torch kernels). llama-server can only apply such a delta when it is packaged as a
GGUF *adapter* file, which this module writes with pure numpy + gguf-py.

File layout (mirrors ``convert_lora_to_gguf.py`` and ``src/llama-adapter.cpp``
from ggml-org/llama.cpp):

* KV metadata: ``general.architecture`` copied from the base GGUF (llama.cpp
  rejects an arch mismatch), ``general.type = "adapter"``,
  ``adapter.type = "lora"`` and ``adapter.lora.alpha`` (see *Scale* below), plus
  informational ``subcor.*`` keys (unit id, version, payload SHA-256, base kernel).
* For every target base tensor ``blk.{i}.{leaf}.weight`` two F32 tensors:

  - ``blk.{i}.{leaf}.weight.lora_a``: ggml ``ne = [d_in, rank]``, i.e. a numpy
    array of shape ``(rank, d_in)``. This is SUBCOR's ``A`` as-is.
  - ``blk.{i}.{leaf}.weight.lora_b``: ggml ``ne = [rank, d_out]``, i.e. a numpy
    array of shape ``(d_out, rank)``. This is SUBCOR's ``B`` as-is (with the
    payload's per-layer scale folded in).

  llama.cpp validates ``base.ne[0] == a.ne[0]`` (d_in), ``base.ne[1] == b.ne[1]``
  (d_out) and ``a.ne[1] == b.ne[0]`` (rank) at load time, and computes
  ``y += scale * B @ (A @ x)``. This exporter checks the same shapes against the
  base GGUF up front, so a mismatched unit fails here instead of at server start.

Scale: llama.cpp uses ``scale = adapter_scale * alpha / rank`` when ``alpha != 0``
and ``scale = adapter_scale`` when ``alpha == 0`` (``rank = lora_b.ne[0]``, per
tensor). When every tensor in the unit has the same rank we write
``alpha = rank`` (the conventional PEFT setting, exactly ``alpha / rank == 1.0`` in
IEEE arithmetic); with mixed ranks we write ``alpha = 0.0``. Either way the
effective multiplier is exactly the requested llama-server LoRA scale, so scale
1.0 applies the stored delta ``B @ A`` unchanged and scale 0.0 is the base model.

Basis: the trainer trains against weights loaded by ``transformers`` (from the
GGUF itself or from the HF checkpoint), i.e. the Hugging Face tensor layout.
Where llama.cpp's converter permutes a tensor and ``transformers``' GGUF loader
undoes it, the delta is re-permuted into the GGUF layout here:

* ``llama`` arch: ``q_proj`` / ``k_proj`` rows (rotary interleave permute),
  applied to ``B`` rows.
* ``qwen35`` / ``qwen35moe`` linear attention with ``num_k_heads != num_v_heads``:
  value heads grouped -> tiled, applied to ``B`` rows of ``in_proj_qkv`` (value
  block only), ``in_proj_z``, ``in_proj_a``, ``in_proj_b`` and to ``A`` columns of
  ``out_proj``. (A no-op for e.g. Qwen3.5-0.8B where the head counts are equal.)

Sub-1-bit units are materialised through ``LayerDeltaPayload.to_adapter`` into
dense low-rank factors ``A = V^T diag(s2)``, ``B = diag(s1) U diag(sigma)`` and
written as F32. The GGUF LoRA is therefore the *reconstructed* delta: exact for
the stored sub-1-bit factors (up to float32 rounding of the scales), not a
re-approximation, but it is ~32x larger on disk than the bit-packed payload.
"""

from __future__ import annotations

import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from subcor.units import SkillUnitRegistry

__all__ = [
    "SUPPORTED_PROJECTIONS",
    "GGUFLoraTensor",
    "canonical_to_gguf_base_tensor",
    "export_unit_to_gguf_lora",
    "read_gguf_architecture",
    "unit_to_gguf_lora_tensors",
]

# Canonical SUBCOR projection (HF module path below `layers.{i}.`) -> ggml leaf name.
SUPPORTED_PROJECTIONS: dict[str, str] = {
    "mlp.gate_proj": "ffn_gate",
    "mlp.up_proj": "ffn_up",
    "mlp.down_proj": "ffn_down",
    "self_attn.q_proj": "attn_q",
    "self_attn.k_proj": "attn_k",
    "self_attn.v_proj": "attn_v",
    "self_attn.o_proj": "attn_output",
    # Qwen3.5 / Qwen3-Next GatedDeltaNet linear-attention projections.
    "linear_attn.in_proj_qkv": "attn_qkv",
    "linear_attn.in_proj_z": "attn_gate",
    "linear_attn.in_proj_a": "ssm_alpha",
    "linear_attn.in_proj_b": "ssm_beta",
    "linear_attn.out_proj": "ssm_out",
}

_TENSOR_DTYPES: dict[str, Any] = {"f32": np.float32, "f16": np.float16}

_LAYER_RE = re.compile(r"(?:^|\.)layers\.(\d+)\.(.+)$")
_QWEN35_ARCHES = frozenset({"qwen35", "qwen35moe"})
_LLAMA_PERMUTE_ARCHES = frozenset({"llama"})


def _require_gguf() -> Any:
    try:
        import gguf  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover - depends on optional extra
        raise ImportError(
            "Exporting Skill Units as GGUF LoRA adapters requires the 'gguf' package. "
            "Install it with `uv sync --extra gguf` (or `pip install gguf`)."
        ) from exc
    return gguf


def canonical_to_gguf_base_tensor(layer_name: str) -> str:
    """Map a canonical SUBCOR layer name to its llama.cpp base tensor (without `.weight`).

    `'layers.18.mlp.down_proj'` -> `'blk.18.ffn_down'`. A model prefix before
    `layers.` (e.g. `model.` or `language_model.model.`) is tolerated. Raises
    KeyError for projections llama.cpp LoRA cannot target via this exporter
    (embeddings, lm_head, MoE experts, norms, ...).
    """
    match = _LAYER_RE.search(layer_name)
    if match is None:
        raise KeyError(
            f"Not a canonical SUBCOR layer name (expected 'layers.<i>.<proj>'): {layer_name!r}"
        )
    idx, proj = match.group(1), match.group(2)
    if proj.endswith(".weight"):
        proj = proj[: -len(".weight")]
    leaf = SUPPORTED_PROJECTIONS.get(proj)
    if leaf is None:
        raise KeyError(
            f"Projection {proj!r} in {layer_name!r} has no llama.cpp LoRA mapping; "
            f"supported: {sorted(SUPPORTED_PROJECTIONS)}"
        )
    return f"blk.{int(idx)}.{leaf}"


@dataclass(frozen=True)
class _BaseInfo:
    arch: str
    # ggml `ne` order: (ne0, ne1) == (d_in, d_out) for a 2D projection weight.
    tensor_ne: dict[str, tuple[int, ...]]
    kv: dict[str, Any]


def _field_value(field: Any) -> Any:
    try:
        return field.contents()
    except Exception:  # pragma: no cover - older gguf-py without contents()
        if len(field.types) == 1 and field.data:
            part = field.parts[field.data[0]]
            if field.types[0].name == "STRING":
                return bytes(part).decode("utf-8")
            return part.tolist()[0]
        return None


def _read_base(base_gguf: Path) -> _BaseInfo:
    gguf = _require_gguf()
    path = Path(base_gguf)
    if not path.is_file():
        raise FileNotFoundError(f"Base GGUF not found: {path}")
    reader = gguf.GGUFReader(str(path), "r")
    arch_field = reader.fields.get("general.architecture")
    if arch_field is None:
        raise ValueError(f"Base GGUF {path} has no general.architecture")
    arch = str(_field_value(arch_field))
    kv: dict[str, Any] = {}
    for key, field in reader.fields.items():
        if key.startswith(f"{arch}.") or key == "general.type":
            kv[key] = _field_value(field)
    tensor_ne = {t.name: tuple(int(x) for x in t.shape) for t in reader.tensors}
    return _BaseInfo(arch=arch, tensor_ne=tensor_ne, kv=kv)


def read_gguf_architecture(base_gguf: Path) -> str:
    """Return `general.architecture` of a GGUF file (e.g. `'qwen35'`, `'llama'`)."""
    return _read_base(Path(base_gguf)).arch


@dataclass(frozen=True)
class GGUFLoraTensor:
    """One llama.cpp LoRA pair in numpy layout: a (rank, d_in), b (d_out, rank)."""

    base_tensor: str  # e.g. 'blk.18.ffn_down.weight'
    a: np.ndarray
    b: np.ndarray

    @property
    def rank(self) -> int:
        return int(self.a.shape[0])


def _grouped_to_tiled_index(num_k_heads: int, v_per_k: int, head_dim: int) -> np.ndarray:
    """llama.cpp `_reorder_v_heads` gather index: tiled[i] = grouped[idx[i]]."""
    n = num_k_heads * v_per_k * head_dim
    return np.arange(n).reshape(num_k_heads, v_per_k, head_dim).transpose(1, 0, 2).reshape(n)


def _llama_permute_rows(b: np.ndarray, n_head: int) -> np.ndarray:
    """llama.cpp `LlamaModel.permute` on rows (HF rotary half-split -> GGUF interleave)."""
    d_out = b.shape[0]
    if d_out % (2 * n_head) != 0:
        raise ValueError(f"Cannot apply llama q/k permute: d_out={d_out} n_head={n_head}")
    return b.reshape(n_head, 2, d_out // n_head // 2, *b.shape[1:]).swapaxes(1, 2).reshape(b.shape)


def _to_gguf_basis(
    proj: str, a: np.ndarray, b: np.ndarray, info: _BaseInfo
) -> tuple[np.ndarray, np.ndarray]:
    """Re-express an HF-layout delta B @ A in the GGUF tensor layout (see module docstring)."""
    arch = info.arch
    if arch in _LLAMA_PERMUTE_ARCHES and proj in ("self_attn.q_proj", "self_attn.k_proj"):
        n_head = int(info.kv[f"{arch}.attention.head_count"])
        if proj == "self_attn.k_proj":
            n_head = int(info.kv.get(f"{arch}.attention.head_count_kv", n_head) or n_head)
        return a, _llama_permute_rows(b, n_head)

    if arch in _QWEN35_ARCHES and proj.startswith("linear_attn."):
        num_k = int(info.kv[f"{arch}.ssm.group_count"])
        num_v = int(info.kv[f"{arch}.ssm.time_step_rank"])
        if num_k <= 0 or num_v <= 0 or num_k == num_v:
            return a, b
        if num_v % num_k != 0:
            raise ValueError(f"{arch}: num_v_heads={num_v} not a multiple of num_k_heads={num_k}")
        v_per_k = num_v // num_k
        head_v_dim = int(info.kv[f"{arch}.ssm.inner_size"]) // num_v
        head_k_dim = int(info.kv[f"{arch}.ssm.state_size"])
        if proj == "linear_attn.in_proj_qkv":
            qk_rows = 2 * head_k_dim * num_k
            idx = _grouped_to_tiled_index(num_k, v_per_k, head_v_dim)
            b = np.concatenate([b[:qk_rows], b[qk_rows:][idx]], axis=0)
        elif proj == "linear_attn.in_proj_z":
            b = b[_grouped_to_tiled_index(num_k, v_per_k, head_v_dim)]
        elif proj in ("linear_attn.in_proj_a", "linear_attn.in_proj_b"):
            b = b[_grouped_to_tiled_index(num_k, v_per_k, 1)]
        elif proj == "linear_attn.out_proj":
            a = a[:, _grouped_to_tiled_index(num_k, v_per_k, head_v_dim)]
        return a, b

    return a, b


def unit_to_gguf_lora_tensors(
    registry: SkillUnitRegistry, unit_id: str, *, base_gguf: Path
) -> tuple[str, list[GGUFLoraTensor]]:
    """Load a unit (SHA-256 verified) and build its GGUF LoRA tensors.

    Returns `(architecture, tensors)`; tensors are float32, already in the GGUF
    basis and shape-checked against the base GGUF. Several payloads that target
    the same layer are merged exactly by rank concatenation.
    """
    info = _read_base(Path(base_gguf))
    deltas = registry.load_verified_deltas(unit_id)

    merged: dict[str, tuple[list[np.ndarray], list[np.ndarray]]] = {}
    order: list[str] = []
    for delta in deltas:
        base_name = canonical_to_gguf_base_tensor(delta.layer_name) + ".weight"
        proj = _LAYER_RE.search(delta.layer_name).group(2)  # type: ignore[union-attr]
        proj = proj[: -len(".weight")] if proj.endswith(".weight") else proj
        adapter = delta.to_adapter(f"{unit_id}:{delta.layer_name}")
        if adapter.binary_ste:  # pragma: no cover - to_adapter never returns STE adapters today
            u, s, vt = np.linalg.svd(adapter.delta_matrix(), full_matrices=False)
            r = min(adapter.rank, len(s))
            b64, a64 = u[:, :r] * s[:r], vt[:r, :]
        else:
            a64 = np.asarray(adapter.a, dtype=np.float64)
            b64 = np.asarray(adapter.b, dtype=np.float64) * float(adapter.scale)
        a64, b64 = _to_gguf_basis(proj, a64, b64, info)

        ne = info.tensor_ne.get(base_name)
        if ne is None:
            raise KeyError(
                f"Unit '{unit_id}' targets {delta.layer_name!r} -> '{base_name}', which is not a "
                f"tensor of the base GGUF ({info.arch}); wrong base model or layer type?"
            )
        d_out, d_in = b64.shape[0], a64.shape[1]
        if len(ne) != 2 or ne[0] != d_in or ne[1] != d_out:
            raise ValueError(
                f"Unit '{unit_id}' layer {delta.layer_name!r} delta is (d_out={d_out}, d_in={d_in}) "
                f"but base tensor '{base_name}' has ggml ne={list(ne)} (expected [d_in, d_out])"
            )
        if base_name not in merged:
            merged[base_name] = ([], [])
            order.append(base_name)
        merged[base_name][0].append(a64)
        merged[base_name][1].append(b64)

    tensors: list[GGUFLoraTensor] = []
    for base_name in order:
        a_parts, b_parts = merged[base_name]
        a = np.ascontiguousarray(np.concatenate(a_parts, axis=0), dtype=np.float32)
        b = np.ascontiguousarray(np.concatenate(b_parts, axis=1), dtype=np.float32)
        tensors.append(GGUFLoraTensor(base_tensor=base_name, a=a, b=b))
    return info.arch, tensors


def export_unit_to_gguf_lora(
    registry: SkillUnitRegistry,
    unit_id: str,
    out_path: Path,
    *,
    base_gguf: Path,
    tensor_dtype: str = "f32",
) -> Path:
    """Write Skill Unit `unit_id` as a llama.cpp LoRA adapter GGUF at `out_path`.

    The payload is SHA-256 verified first. The file is written atomically (temp
    file in the destination directory, then `os.replace`), so a concurrent
    llama-server restart never sees a half-written adapter. Load it with
    `llama-server --lora <file>` (or `--lora-scaled <file>:0` and select it per
    request; `--lora-init-without-apply` alone still applies it at scale 1.0 on
    llama.cpp b12110); scale 1.0 applies the unit's delta exactly, scale 0.0 is the
    base model. Sub-1-bit units are written as their reconstructed dense
    low-rank factors (see module docstring). `tensor_dtype="f16"` stores the
    `lora_a` / `lora_b` tensors as F16 (what `convert_lora_to_gguf.py --outtype f16`
    writes) instead of F32. Returns `out_path`.
    """
    if tensor_dtype not in _TENSOR_DTYPES:
        raise ValueError(f"tensor_dtype must be one of {sorted(_TENSOR_DTYPES)}")
    np_dtype = _TENSOR_DTYPES[tensor_dtype]
    gguf = _require_gguf()
    out_path = Path(out_path)
    manifest = registry.get_manifest(unit_id)
    arch, tensors = unit_to_gguf_lora_tensors(registry, unit_id, base_gguf=Path(base_gguf))
    if not tensors:  # pragma: no cover - propose() rejects empty units
        raise ValueError(f"Skill Unit '{unit_id}' has no layer deltas to export")

    ranks = {t.rank for t in tensors}
    # alpha == rank -> alpha / rank == 1.0 exactly; alpha == 0 -> llama.cpp skips alpha scaling.
    alpha = float(next(iter(ranks))) if len(ranks) == 1 else 0.0

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        dir=str(out_path.parent), prefix=f".{out_path.name}.", suffix=".tmp"
    )
    os.close(fd)
    try:
        writer = gguf.GGUFWriter(tmp_name, arch)
        writer.add_type(gguf.GGUFType.ADAPTER)
        writer.add_string(gguf.Keys.Adapter.TYPE, "lora")
        writer.add_float32(gguf.Keys.Adapter.LORA_ALPHA, alpha)
        writer.add_name(f"subcor-{manifest.unit_id}")
        writer.add_string("subcor.unit_id", manifest.unit_id)
        writer.add_string("subcor.unit_version", manifest.version)
        writer.add_string("subcor.payload_sha256", manifest.payload_sha256)
        writer.add_string("subcor.base_kernel_id", manifest.base_kernel_id)
        writer.add_string("subcor.encoding", manifest.encoding.value)
        for t in tensors:
            for suffix, arr in (("lora_a", t.a), ("lora_b", t.b)):
                cast = np.ascontiguousarray(arr, dtype=np_dtype)
                if not np.all(np.isfinite(cast)):
                    raise ValueError(
                        f"Unit '{unit_id}' tensor {t.base_tensor}.{suffix} is not finite "
                        f"as {tensor_dtype}"
                    )
                writer.add_tensor(f"{t.base_tensor}.{suffix}", cast)
        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file()
        writer.close()
        with open(tmp_name, "rb+") as fh:
            os.fsync(fh.fileno())
        os.replace(tmp_name, out_path)
    except BaseException:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)
        raise
    return out_path
