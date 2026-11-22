"""PyTorch (ROCm / CUDA / CPU) frozen-kernel runtime and Skill Unit trainer.

Mirrors :mod:`subcor.backends.mlx_kernel` on top of Hugging Face ``transformers``
so the governed Skill Unit loop runs on AMD GPUs via ROCm (exposed by PyTorch as
``cuda``), on NVIDIA GPUs, or on plain CPU. Its main job on Linux is to be the
**trainer** next to a llama.cpp server: ``transformers`` can load a GGUF directly
(dequantized to fp16/fp32), so Skill Units are trained against the same weights
``llama-server`` serves.

  - Base ``nn.Linear`` weights are strictly frozen (``requires_grad=False``)
    and never overwritten; they stay bit-identical across mount / train cycles.
  - Instrumented projections are wrapped in :class:`SubcorTorchLinear`, supporting:
      * Mean hidden-state activation recording during generation (detached
        float32 sums, no autograd graph retention).
      * Multi-unit hot-swap mounting and atomic rollback of Sub-1-Bit or FP32
        Skill Units verified by SHA-256 via SkillUnitRegistry.
      * Function-preserving zero-initialized LoRA expansion (B = 0) and
        QLoRA-style adaptation (FP32 LoRA branch over a frozen fp16/bf16/fp32
        base), with optional two-stage Binary STE (QAT) refinement.
      * Base weight export for zero-gradient weight-space surgery.

Canonical layer names are identical to the MLX backend (``layers.{i}.{projection}``)
so Skill Units, router gates and telemetry are portable between backends.
"""

from __future__ import annotations

import contextlib
import functools
import inspect
import json
import logging
import math
import sys
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from subcor.confidence import parse_tool_calls
from subcor.expansion import LowRankAdapter
from subcor.intern import ngram_repetition_ratio
from subcor.router import MoESkillRouter, RoutingDecision
from subcor.units import LayerDeltaPayload, SkillUnitRegistry

DEFAULT_TORCH_MODEL = "Qwen/Qwen3.5-0.8B"
logger = logging.getLogger(__name__)

# Candidate attribute paths (relative to the loaded model) of the decoder-layer list.
_DECODER_LAYER_PATHS: tuple[str, ...] = (
    "model.layers",
    "model.language_model.layers",
    "model.model.layers",
    "language_model.model.layers",
    "language_model.layers",
    "model.decoder.layers",
    "layers",
    "transformer.h",
)

_DTYPE_ALIASES: dict[str, torch.dtype] = {
    "float16": torch.float16,
    "fp16": torch.float16,
    "half": torch.float16,
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
    "float32": torch.float32,
    "fp32": torch.float32,
    "float": torch.float32,
}


def _ste_sign_pm1(z: torch.Tensor) -> torch.Tensor:
    """Binary sign in {-1, +1} with clipped Straight-Through Estimator (|z| <= 1)."""
    hard = torch.where(z >= 0, torch.ones_like(z), -torch.ones_like(z))
    clipped = torch.clamp(z, -1.0, 1.0)
    return clipped + (hard - clipped).detach()


def _sign_pm1_np(z: np.ndarray) -> np.ndarray:
    return np.where(z >= 0, 1.0, -1.0)


def _cosine_decay_lr(init: float, decay_steps: int, end: float, step: int) -> float:
    """Same schedule as ``mlx.optimizers.cosine_decay(init, decay_steps, end)``."""
    s = min(step, decay_steps)
    return end + (init - end) * 0.5 * (1.0 + math.cos(math.pi * s / max(1, decay_steps)))


def _resolve_device(device: str | torch.device | None) -> torch.device:
    if isinstance(device, torch.device):
        return device
    if device is None or str(device).strip().lower() in {"", "auto"}:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(str(device))


def _resolve_dtype(dtype: str | torch.dtype | None, device: torch.device) -> torch.dtype:
    if isinstance(dtype, torch.dtype):
        return dtype
    if dtype is None or (isinstance(dtype, str) and dtype.strip().lower() in {"", "auto"}):
        # fp16 is the fastest dtype on RDNA2 (bf16 is emulated / slower there).
        return torch.float16 if device.type in {"cuda", "mps"} else torch.float32
    key = str(dtype).strip().lower().removeprefix("torch.")
    if key not in _DTYPE_ALIASES:
        raise ValueError(f"Unsupported dtype {dtype!r}; expected one of {sorted(_DTYPE_ALIASES)}")
    return _DTYPE_ALIASES[key]


def _get_attr_path(root: Any, dotted: str) -> Any:
    obj = root
    for part in dotted.split("."):
        obj = obj[int(part)] if part.isdigit() else getattr(obj, part)
    return obj


def _resolve_parent_and_attr(root: Any, dotted: str) -> tuple[Any, str]:
    parts = dotted.split(".")
    parent = root
    for part in parts[:-1]:
        parent = parent[int(part)] if part.isdigit() else getattr(parent, part)
    return parent, parts[-1]


def _layer_index(canonical_name: str) -> int | None:
    parts = canonical_name.split(".")
    if len(parts) >= 2 and parts[0] == "layers" and parts[1].isdigit():
        return int(parts[1])
    return None


# Module-level kernel functions that transformers may bind to Triton implementations
# (flash-linear-attention / causal-conv1d) at import time, independent of device.
_DEVICE_DISPATCH_FUNCS: tuple[str, ...] = (
    "torch_chunk_gated_delta_rule",
    "torch_recurrent_gated_delta_rule",
    "chunk_gated_delta_rule",
    "fused_recurrent_gated_delta_rule",
    "causal_conv1d_fn",
    "causal_conv1d_update",
)


def _first_tensor(args: tuple[Any, ...], kwargs: dict[str, Any]) -> torch.Tensor | None:
    for val in (*args, *kwargs.values()):
        if isinstance(val, torch.Tensor):
            return val
    return None


def _make_cpu_fallback_dispatch(fast: Callable[..., Any], ref: Callable[..., Any]) -> Callable:
    params = inspect.signature(ref).parameters
    accepts_var_kw = any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values())

    @functools.wraps(fast)
    def dispatch(*args: Any, **kwargs: Any) -> Any:
        t = _first_tensor(args, kwargs)
        if t is not None and t.device.type == "cpu":
            if not accepts_var_kw:
                kwargs = {k: v for k, v in kwargs.items() if k in params}
            return ref(*args, **kwargs)
        return fast(*args, **kwargs)

    dispatch._subcor_cpu_dispatch = True  # type: ignore[attr-defined]
    return dispatch


def _install_cpu_safe_kernels(model: nn.Module) -> None:
    """Route CPU tensors in hub/Triton-backed modeling functions to their torch reference.

    transformers binds e.g. Qwen3.5 GatedDeltaNet's ``torch_chunk_gated_delta_rule`` to
    flash-linear-attention's Triton kernel whenever ``fla`` is importable, regardless of
    the tensor device, which crashes on CPU. GPU tensors keep the fast implementation.
    """
    module_names = {type(m).__module__ for m in model.modules()}
    for mod_name in module_names:
        mod = sys.modules.get(mod_name)
        if mod is None:
            continue
        for fn_name in _DEVICE_DISPATCH_FUNCS:
            fn = getattr(mod, fn_name, None)
            if not callable(fn) or getattr(fn, "_subcor_cpu_dispatch", False):
                continue
            ref = inspect.unwrap(fn)
            if ref is fn or not inspect.isfunction(ref) or ref.__module__ != mod_name:
                continue
            setattr(mod, fn_name, _make_cpu_fallback_dispatch(fn, ref))


def _resolve_gguf_source(model_id: str, gguf_file: str | Path | None) -> tuple[str, str | None]:
    """Return (from_pretrained source, gguf filename) for a repo-relative or local GGUF."""
    if gguf_file is None:
        return model_id, None
    candidate = Path(gguf_file).expanduser()
    if candidate.is_absolute() or candidate.is_file():
        # absolute() (not resolve()) keeps HF-cache snapshot symlinks readable.
        resolved = candidate.absolute()
        if not resolved.is_file():
            raise FileNotFoundError(f"GGUF file not found: {resolved}")
        return str(resolved.parent), resolved.name
    return model_id, str(gguf_file)


class _StopForward(Exception):  # noqa: N818 - private control-flow signal, not an error
    """Private control-flow signal used to early-exit a forward pass after the tap layer."""


@dataclass(frozen=True)
class GenerationTrace:
    """Result of one deterministic generation with logprobs, tool calls, and activations."""

    prompt: str
    response: str
    tokens_generated: int
    hit_token_limit: bool
    repetition_ratio: float
    layer_activations: dict[str, np.ndarray]
    prompt_tokens: int = 0
    token_logprobs: tuple[float, ...] = ()
    tool_calls: tuple[dict[str, Any], ...] = ()
    raw_response: str = ""


@dataclass(frozen=True)
class StreamTokenEvent:
    """Incremental token emitted during true token-by-token streaming."""

    text: str
    token_id: int
    logprob: float
    prompt_tokens: int
    generation_tokens: int
    finish_reason: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)


def _pack_int4_affine(
    weight: torch.Tensor,
    *,
    group_size: int = 64,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pack a 2D float weight ``(d_out, d_in)`` into group-wise affine INT4 ``uint8`` + fp16 scale/zero."""
    if weight.ndim != 2:
        raise ValueError(f"Expected 2D weight tensor, got shape {tuple(weight.shape)}")
    if group_size <= 0 or group_size % 2 != 0:
        raise ValueError(f"group_size must be a positive even integer, got {group_size}")

    d_out, d_in = int(weight.shape[0]), int(weight.shape[1])
    w_f32 = weight.detach().to(torch.float32)
    pad = (-d_in) % group_size
    if pad:
        w_f32 = F.pad(w_f32, (0, pad), value=0.0)

    n_groups = (d_in + pad) // group_size
    grouped = w_f32.reshape(d_out, n_groups, group_size)
    w_min = grouped.amin(dim=-1, keepdim=True)
    w_max = grouped.amax(dim=-1, keepdim=True)
    scale = ((w_max - w_min) / 15.0).clamp(min=1e-8)

    # Quantize against the fp16-rounded scale & zero so unpack matches pack tightly.
    scale_f16 = scale.to(torch.float16)
    zero_f16 = w_min.to(torch.float16)
    scale_q = scale_f16.to(torch.float32).clamp(min=1e-8)
    zero_q = zero_f16.to(torch.float32)

    q = torch.round((grouped - zero_q) / scale_q).clamp(0.0, 15.0).to(torch.uint8)
    lo = q[..., 0::2] & 0x0F
    hi = (q[..., 1::2] & 0x0F) << 4
    qweight = (lo | hi).contiguous()  # (d_out, n_groups, group_size // 2)
    return qweight, scale_f16.squeeze(-1).contiguous(), zero_f16.squeeze(-1).contiguous()


def _unpack_int4_affine(
    qweight: torch.Tensor,
    scales: torch.Tensor,
    zeros: torch.Tensor,
    *,
    out_features: int,
    in_features: int,
    group_size: int,
    compute_dtype: torch.dtype,
) -> torch.Tensor:
    """Dequantize packed 4-bit affine weight buffers transiently into ``(out_features, in_features)``."""
    lo = (qweight & 0x0F).to(compute_dtype)
    hi = ((qweight >> 4) & 0x0F).to(compute_dtype)
    n_groups = int(qweight.shape[1])
    q_unpacked = torch.stack((lo, hi), dim=-1).reshape(out_features, n_groups, group_size)
    w = q_unpacked * scales.to(compute_dtype).unsqueeze(-1) + zeros.to(compute_dtype).unsqueeze(-1)
    w = w.reshape(out_features, n_groups * group_size)
    if w.shape[1] != in_features:
        w = w[:, :in_features]
    return w


class _QuantizedFrozenLinearFn(torch.autograd.Function):
    """On-the-fly 4-bit dequantizing linear op: never caches dense W for backward."""

    @staticmethod
    def forward(  # type: ignore[override]
        ctx: Any,
        x: torch.Tensor,
        qweight: torch.Tensor,
        scales: torch.Tensor,
        zeros: torch.Tensor,
        bias: torch.Tensor | None,
        out_features: int,
        in_features: int,
        group_size: int,
        compute_dtype: torch.dtype,
    ) -> torch.Tensor:
        w = _unpack_int4_affine(
            qweight,
            scales,
            zeros,
            out_features=out_features,
            in_features=in_features,
            group_size=group_size,
            compute_dtype=compute_dtype,
        )
        b = bias.to(w.dtype) if bias is not None else None
        y = F.linear(x.to(w.dtype), w, b)
        if y.dtype != x.dtype:
            y = y.to(x.dtype)
        # Save ONLY packed 4-bit buffers (~4.5 bpw); dense `w` is discarded immediately.
        ctx.save_for_backward(qweight, scales, zeros)
        ctx.meta = (out_features, in_features, group_size, compute_dtype)
        return y

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> tuple[Any, ...]:  # type: ignore[override]
        qweight, scales, zeros = ctx.saved_tensors
        out_features, in_features, group_size, compute_dtype = ctx.meta
        w = _unpack_int4_affine(
            qweight,
            scales,
            zeros,
            out_features=out_features,
            in_features=in_features,
            group_size=group_size,
            compute_dtype=compute_dtype,
        )
        grad_x = grad_output.to(w.dtype) @ w
        if grad_x.dtype != grad_output.dtype:
            grad_x = grad_x.to(grad_output.dtype)
        # Base weights are strictly frozen, so only grad_x is needed.
        return grad_x, None, None, None, None, None, None, None, None


class QuantizedFrozenLinear(nn.Module):
    """Pure-PyTorch packed 4-bit frozen linear layer with transient per-layer dequantization.

    Stores weights as packed ``uint8`` nibbles + ``fp16`` per-group scale/zero (~4.5 bpw,
    ~3.5x smaller than FP16) with zero external CUDA/HIP/bitsandbytes build dependencies.
    During training, forward and backward each dequantize only this single layer transiently
    and discard the dense matrix immediately, matching MLX's ``nn.QuantizedLinear`` footprint.
    """

    qweight: torch.Tensor
    scales: torch.Tensor
    zeros: torch.Tensor
    bias: torch.Tensor | None

    def __init__(
        self,
        qweight: torch.Tensor,
        scales: torch.Tensor,
        zeros: torch.Tensor,
        *,
        out_features: int,
        in_features: int,
        group_size: int = 64,
        compute_dtype: torch.dtype = torch.float16,
        bias: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.out_features = int(out_features)
        self.in_features = int(in_features)
        self.group_size = int(group_size)
        self.compute_dtype = compute_dtype
        self.register_buffer("qweight", qweight)
        self.register_buffer("scales", scales)
        self.register_buffer("zeros", zeros)
        if bias is not None:
            self.register_buffer("bias", bias.detach().clone())
        else:
            self.register_buffer("bias", None)

    @classmethod
    def from_linear(
        cls,
        linear: nn.Linear,
        *,
        group_size: int = 64,
        compute_dtype: torch.dtype | None = None,
    ) -> QuantizedFrozenLinear:
        """Pack an existing dense ``nn.Linear`` into a frozen 4-bit module."""
        w = linear.weight.detach()
        out_features, in_features = int(w.shape[0]), int(w.shape[1])
        eff_dtype = compute_dtype if compute_dtype is not None else w.dtype
        qweight, scales, zeros = _pack_int4_affine(w, group_size=group_size)
        bias = (
            linear.bias.detach().to(device=qweight.device, dtype=eff_dtype)
            if linear.bias is not None
            else None
        )
        return cls(
            qweight=qweight,
            scales=scales,
            zeros=zeros,
            out_features=out_features,
            in_features=in_features,
            group_size=group_size,
            compute_dtype=eff_dtype,
            bias=bias,
        )

    @property
    def shape(self) -> tuple[int, int]:
        return self.out_features, self.in_features

    @property
    def device(self) -> torch.device:
        return self.qweight.device

    @property
    def nbytes_packed(self) -> int:
        total = int(self.qweight.numel() * self.qweight.element_size())
        total += int(self.scales.numel() * self.scales.element_size())
        total += int(self.zeros.numel() * self.zeros.element_size())
        if self.bias is not None:
            total += int(self.bias.numel() * self.bias.element_size())
        return total

    def dequantize_weight(self, *, dtype: torch.dtype | None = None) -> torch.Tensor:
        """Return a detached dense copy ``(out_features, in_features)`` of the frozen weight."""
        with torch.no_grad():
            return _unpack_int4_affine(
                self.qweight,
                self.scales,
                self.zeros,
                out_features=self.out_features,
                in_features=self.in_features,
                group_size=self.group_size,
                compute_dtype=dtype if dtype is not None else self.compute_dtype,
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _QuantizedFrozenLinearFn.apply(
            x,
            self.qweight,
            self.scales,
            self.zeros,
            self.bias,
            self.out_features,
            self.in_features,
            self.group_size,
            self.compute_dtype,
        )

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"group_size={self.group_size}, compute_dtype={self.compute_dtype}, "
            f"bias={self.bias is not None}"
        )


def quantize_model_linears_4bit(
    model: nn.Module,
    *,
    group_size: int = 64,
    compute_dtype: torch.dtype | None = None,
    skip_names: tuple[str, ...] = ("lm_head",),
    min_in_features: int = 16,
) -> tuple[int, int, int]:
    """Replace eligible frozen ``nn.Linear`` layers in-place with :class:`QuantizedFrozenLinear`.

    Returns ``(replaced_count, dense_bytes_before, packed_bytes_after)``.
    """
    replaced = 0
    dense_bytes = 0
    packed_bytes = 0

    def _walk(parent: nn.Module, prefix: str) -> None:
        nonlocal replaced, dense_bytes, packed_bytes
        for child_name, child in list(parent.named_children()):
            full = f"{prefix}.{child_name}" if prefix else child_name
            if child_name in skip_names or full in skip_names:
                continue
            if isinstance(child, SubcorTorchLinear):
                continue
            if (
                isinstance(child, nn.Linear)
                and child.weight is not None
                and child.weight.ndim == 2
                and child.weight.is_floating_point()
                and int(child.in_features) >= min_in_features
            ):
                dense_bytes += int(child.weight.numel() * child.weight.element_size())
                if child.bias is not None:
                    dense_bytes += int(child.bias.numel() * child.bias.element_size())
                qmod = QuantizedFrozenLinear.from_linear(
                    child,
                    group_size=group_size,
                    compute_dtype=compute_dtype,
                )
                packed_bytes += qmod.nbytes_packed
                setattr(parent, child_name, qmod)
                replaced += 1
            else:
                _walk(child, full)

    _walk(model, "")
    return replaced, dense_bytes, packed_bytes


class SubcorTorchLinear(nn.Module):
    """Hot-swappable adapter wrapper around a frozen ``nn.Linear`` or ``QuantizedFrozenLinear``."""

    lora_a: nn.Parameter | None
    lora_b: nn.Parameter | None
    s_in: nn.Parameter | None
    s_out: nn.Parameter | None
    sigma: nn.Parameter | None

    def __init__(self, base: nn.Linear | QuantizedFrozenLinear, layer_name: str) -> None:
        super().__init__()
        if isinstance(base, QuantizedFrozenLinear):
            self._compute_dtype: torch.dtype = base.compute_dtype
        elif hasattr(base, "weight") and base.weight is not None and base.weight.ndim == 2:
            self._compute_dtype = base.weight.dtype
        else:
            raise TypeError(
                f"SubcorTorchLinear expects a 2D nn.Linear or QuantizedFrozenLinear, "
                f"got {type(base).__name__}"
            )
        self.base = base
        self.layer_name = layer_name

        # Mounted frozen adapters: name -> (a_t (d_in, r), b_t_scaled (r, d_out)).
        self._mounted: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}

        # Optional trainable LoRA / Binary STE (QAT) branch (same orientation as MLX).
        self.register_parameter("lora_a", None)  # (d_in, rank)
        self.register_parameter("lora_b", None)  # (rank, d_out)
        self.register_parameter("s_in", None)  # (d_in,) used when _lora_binary_ste=True
        self.register_parameter("s_out", None)  # (d_out,) used when _lora_binary_ste=True
        self.register_parameter("sigma", None)  # (rank,) used when _lora_binary_ste=True
        self._lora_scale: float = 1.0
        self._lora_binary_ste: bool = False
        self._lora_active: bool = False

        # Toggleable mean activation recorder (detached, off-graph fp32 accumulator).
        self._recording: bool = False
        self._act_sum: torch.Tensor | None = None
        self._act_count: int = 0
        self._bypass_adapters: bool = False

        # Freeze base layer strictly.
        for p in self.base.parameters():
            p.requires_grad_(False)

    # ------------------------------------------------------------------ properties
    @property
    def shape(self) -> tuple[int, int]:
        """Return (d_out, d_in) of the underlying weight matrix."""
        if isinstance(self.base, QuantizedFrozenLinear):
            return self.base.shape
        out_dims, in_dims = self.base.weight.shape
        return int(out_dims), int(in_dims)

    @property
    def in_features(self) -> int:
        return self.shape[1]

    @property
    def out_features(self) -> int:
        return self.shape[0]

    @property
    def weight(self) -> torch.Tensor:
        """Frozen base weight (read-only view for HF utilities that probe ``.weight``)."""
        if isinstance(self.base, QuantizedFrozenLinear):
            return self.base.dequantize_weight()
        return self.base.weight

    @property
    def bias(self) -> torch.Tensor | None:
        return self.base.bias

    @property
    def mounted_names(self) -> tuple[str, ...]:
        return tuple(self._mounted.keys())

    @property
    def _device(self) -> torch.device:
        if isinstance(self.base, QuantizedFrozenLinear):
            return self.base.device
        return self.base.weight.device

    # ------------------------------------------------------------------ base weight
    def dequantize_base_weight(self) -> np.ndarray:
        """Return a float64 copy of the frozen base weight matrix (d_out, d_in)."""
        with torch.no_grad():
            if isinstance(self.base, QuantizedFrozenLinear):
                w = self.base.dequantize_weight(dtype=torch.float32).to(device="cpu")
            else:
                w = self.base.weight.detach().to(device="cpu", dtype=torch.float32)
        return np.array(w.numpy(), dtype=np.float64, copy=True)

    # ------------------------------------------------------------------ recording
    def start_recording(self) -> None:
        self._recording = True
        self._act_sum = None
        self._act_count = 0

    def stop_recording(self) -> np.ndarray | None:
        self._recording = False
        if self._act_sum is None or self._act_count == 0:
            self._act_sum = None
            self._act_count = 0
            return None
        total = self._act_sum.detach().to(device="cpu", dtype=torch.float64).numpy()
        out = np.array(total, dtype=np.float64, copy=True) / float(self._act_count)
        self._act_sum = None
        self._act_count = 0
        return out

    # ------------------------------------------------------------------ mounting
    def mount_adapter(self, name: str, adapter: LowRankAdapter) -> None:
        """Mount a frozen LowRankAdapter (Delta W = scale * B @ A) in the base compute dtype."""
        d_out, d_in = self.shape
        if adapter.d_out != d_out or adapter.d_in != d_in:
            raise ValueError(
                f"Adapter '{name}' shape ({adapter.d_out}, {adapter.d_in}) does not match "
                f"layer '{self.layer_name}' shape ({d_out}, {d_in})"
            )
        if adapter.binary_ste:
            # Exact low-rank factorisation of the binary-STE delta:
            #   Delta W = scale * diag(s_out) sign(B) diag(sigma) sign(A) diag(s_in)
            assert adapter.s_in is not None and adapter.s_out is not None
            assert adapter.sigma is not None
            a_np = _sign_pm1_np(adapter.a) * adapter.s_in[np.newaxis, :]
            b_np = (
                adapter.scale
                * (adapter.s_out[:, np.newaxis] * _sign_pm1_np(adapter.b))
                * adapter.sigma[np.newaxis, :]
            )
        else:
            a_np = adapter.a
            b_np = adapter.scale * adapter.b
        # Always create normal (non-inference) tensors so mounted units remain usable
        # inside autograd-tracked training forwards.
        with torch.inference_mode(False), torch.no_grad():
            a_t = torch.from_numpy(np.ascontiguousarray(a_np.T, dtype=np.float32)).to(
                device=self._device, dtype=self._compute_dtype
            )
            b_t = torch.from_numpy(np.ascontiguousarray(b_np.T, dtype=np.float32)).to(
                device=self._device, dtype=self._compute_dtype
            )
        self._mounted[name] = (a_t, b_t)

    def unmount_adapter(self, name: str) -> None:
        self._mounted.pop(name, None)

    # ------------------------------------------------------------------ trainable LoRA
    def _new_param(self, arr: np.ndarray) -> nn.Parameter:
        with torch.inference_mode(False), torch.no_grad():
            t = torch.from_numpy(np.ascontiguousarray(arr, dtype=np.float32)).to(self._device)
        return nn.Parameter(t, requires_grad=True)

    def attach_trainable_lora(
        self,
        *,
        rank: int,
        scale: float = 1.0,
        binary_ste: bool = False,
        seed: int = 0,
    ) -> None:
        """Attach a zero-initialized trainable LoRA or Binary STE (QAT) branch."""
        if rank <= 0:
            raise ValueError("rank must be positive")
        d_out, d_in = self.shape
        rng = np.random.default_rng(seed)
        self._lora_scale = float(scale)
        self._lora_binary_ste = binary_ste
        self._lora_active = True
        for p in self.base.parameters():
            p.requires_grad_(False)

        if binary_ste:
            # Matches Sub1BitMatrix 1-to-1; s_out=0 guarantees exact zero initial perturbation.
            self.lora_a = self._new_param(rng.uniform(-0.4, 0.4, size=(d_in, rank)))
            self.lora_b = self._new_param(rng.uniform(-0.4, 0.4, size=(rank, d_out)))
            self.s_in = self._new_param(np.ones((d_in,)))
            self.s_out = self._new_param(np.zeros((d_out,)))
            self.sigma = self._new_param(np.full((rank,), 1.0 / math.sqrt(d_in * rank)))
        else:
            bound = 1.0 / math.sqrt(d_in)
            self.lora_a = self._new_param(rng.uniform(-bound, bound, size=(d_in, rank)))
            self.lora_b = self._new_param(np.zeros((rank, d_out)))
            self.s_in = None
            self.s_out = None
            self.sigma = None

    def convert_lora_to_binary_ste(self) -> None:
        """Warm-start Binary STE (QAT) parameters directly from the current FP32 LoRA delta.

        Mirrors the MLX backend: continuous warm-start lands in a good basin first,
        then initializing STE from Sub1BitMatrix lets STE refine around {-1, +1}.
        """
        if not self._lora_active or self._lora_binary_ste:
            return
        fp32_adapter = self.extract_trainable_lora(adapter_name=f"warm:{self.layer_name}")
        fac = fp32_adapter.compress_sub1bit(rank=fp32_adapter.rank)
        u_bin, v_bin = fac.unpack_factors()  # u: (d_out, r), v: (d_in, r)

        # Latent weights start at +-0.5 inside [-1, 1] so STE gradients remain responsive.
        self.lora_a = self._new_param(0.5 * v_bin)  # (d_in, rank)
        self.lora_b = self._new_param(0.5 * u_bin.T)  # (rank, d_out)
        self.s_in = self._new_param(fac.s2.astype(np.float32))
        self.s_out = self._new_param(fac.s1.astype(np.float32))
        self.sigma = self._new_param(fac.sigma.astype(np.float32))
        self._lora_scale = 1.0
        self._lora_binary_ste = True

    def trainable_lora_keys(self) -> list[str]:
        if self._lora_binary_ste:
            return ["lora_a", "lora_b", "s_in", "s_out", "sigma"]
        return ["lora_a", "lora_b"]

    def trainable_lora_parameters(self) -> list[nn.Parameter]:
        """Return the live trainable LoRA parameters (in ``trainable_lora_keys`` order)."""
        if not self._lora_active:
            return []
        params: list[nn.Parameter] = []
        for key in self.trainable_lora_keys():
            p = getattr(self, key)
            if p is not None:
                params.append(p)
        return params

    @staticmethod
    def _to_np(t: torch.Tensor) -> np.ndarray:
        return np.array(
            t.detach().to(device="cpu", dtype=torch.float32).numpy(), dtype=np.float64, copy=True
        )

    def extract_trainable_lora(self, adapter_name: str) -> LowRankAdapter:
        """Convert the active trainable branch into a frozen SUBCOR LowRankAdapter.

        Returns a (rank, d_in) and b (d_out, rank), the SUBCOR / PEFT orientation.
        """
        if not self._lora_active or self.lora_a is None or self.lora_b is None:
            raise RuntimeError(f"No trainable LoRA attached on layer '{self.layer_name}'")
        a_np = self._to_np(self.lora_a).T  # (rank, d_in)
        b_np = self._to_np(self.lora_b).T  # (d_out, rank)
        if self._lora_binary_ste:
            assert self.s_in is not None and self.s_out is not None and self.sigma is not None
            return LowRankAdapter(
                name=adapter_name,
                a=a_np,
                b=b_np,
                scale=self._lora_scale,
                binary_ste=True,
                s_in=self._to_np(self.s_in),
                s_out=self._to_np(self.s_out),
                sigma=self._to_np(self.sigma),
                frozen=True,
            )
        return LowRankAdapter(
            name=adapter_name,
            a=a_np,
            b=b_np,
            scale=self._lora_scale,
            binary_ste=False,
            frozen=True,
        )

    def detach_trainable_lora(self) -> None:
        """Remove trainable LoRA branch without touching frozen base or mounted units."""
        self.lora_a = None
        self.lora_b = None
        self.s_in = None
        self.s_out = None
        self.sigma = None
        self._lora_binary_ste = False
        self._lora_active = False

    # ------------------------------------------------------------------ forward
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.base(x)

        if not self._bypass_adapters:
            if self._mounted:
                x_c = x if x.dtype == self._compute_dtype else x.to(self._compute_dtype)
                for a_t, b_t in self._mounted.values():
                    y = y + ((x_c @ a_t) @ b_t).to(y.dtype)

            if self._lora_active and self.lora_a is not None and self.lora_b is not None:
                x_f32 = x.float()
                if self._lora_binary_ste:
                    assert (
                        self.s_in is not None and self.s_out is not None and self.sigma is not None
                    )
                    a_q = _ste_sign_pm1(self.lora_a)
                    b_q = _ste_sign_pm1(self.lora_b)
                    h = ((x_f32 * self.s_in) @ a_q) * self.sigma
                    z = ((h @ b_q) * self.s_out) * self._lora_scale
                else:
                    z = ((x_f32 @ self.lora_a) @ self.lora_b) * self._lora_scale
                y = y + z.to(y.dtype)

        if self._recording:
            # Mean hidden output across batch & sequence positions, strictly off-graph.
            with torch.no_grad():
                vec_sum = y.detach().reshape(-1, y.shape[-1]).float().sum(dim=0)
                count = int(y.numel() // y.shape[-1])
            # Out-of-place accumulate: safe whether or not we're inside inference_mode.
            self._act_sum = vec_sum if self._act_sum is None else self._act_sum + vec_sum
            self._act_count += count

        return y

    def extra_repr(self) -> str:
        d_out, d_in = self.shape
        return (
            f"layer_name={self.layer_name!r}, d_in={d_in}, d_out={d_out}, "
            f"mounted={len(self._mounted)}, lora_active={self._lora_active}"
        )


class TorchKernel:
    """Frozen PyTorch base model with governed hot-swap Skill Units (ROCm / CUDA / CPU)."""

    def __init__(
        self,
        model_id: str = DEFAULT_TORCH_MODEL,
        *,
        gguf_file: str | Path | None = None,
        target_layer_indices: list[int] | None = None,
        projection_attr: str = "mlp.down_proj",
        target_projections: tuple[str, ...] | None = None,
        device: str | torch.device | None = None,
        dtype: str | torch.dtype | None = None,
        quantize_base_weights: bool = False,
        quantize_group_size: int = 64,
        model: Any = None,
        tokenizer: Any = None,
    ) -> None:
        self.model_id = model_id
        self.gguf_file: str | None = None
        self.quantize_base_weights: bool = bool(quantize_base_weights)
        self.quantize_group_size: int = int(quantize_group_size)
        self.quantization_stats: dict[str, int] = {}
        self.last_train_stats: dict[str, Any] = {}
        if (model is None) != (tokenizer is None):
            raise ValueError("Pass both model and tokenizer, or neither")

        if model is None:
            try:
                from transformers import AutoModelForCausalLM, AutoTokenizer
            except ImportError as exc:  # pragma: no cover - environment guard
                raise ImportError(
                    "TorchKernel requires transformers; install with `uv sync --extra rocm`"
                ) from exc
            self.device = _resolve_device(device)
            self.dtype = _resolve_dtype(dtype, self.device)
            source, gguf_name = _resolve_gguf_source(model_id, gguf_file)
            load_kw: dict[str, Any] = {}
            if gguf_name is not None:
                load_kw["gguf_file"] = gguf_name
                self.gguf_file = str(Path(source) / gguf_name) if source != model_id else gguf_name
            tokenizer = AutoTokenizer.from_pretrained(source, **load_kw)
            model_kw = dict(load_kw)
            if gguf_name is not None:
                # Always dequantize to plain nn.Linear on CPU first. When
                # quantize_base_weights=True, we immediately pack decoder linears into
                # 4-bit QuantizedFrozenLinear on CPU before moving to VRAM so even 9B
                # models fit cleanly inside 16 GB VRAM.
                model_kw["quantization_config"] = _dequantize_gguf_config()
            model = AutoModelForCausalLM.from_pretrained(source, dtype=self.dtype, **model_kw)
            if self.quantize_base_weights:
                n_rep, d_bytes, p_bytes = quantize_model_linears_4bit(
                    model,
                    group_size=self.quantize_group_size,
                    compute_dtype=self.dtype,
                )
                self.quantization_stats = {
                    "quantized_layers": n_rep,
                    "dense_bytes": d_bytes,
                    "packed_bytes": p_bytes,
                }
            model = model.to(self.device)
        else:
            first = next(iter(model.parameters()), None)
            model_device = first.device if first is not None else torch.device("cpu")
            model_dtype = first.dtype if first is not None else torch.float32
            self.device = _resolve_device(device) if device is not None else model_device
            self.dtype = _resolve_dtype(dtype, self.device) if dtype is not None else model_dtype
            if self.quantize_base_weights:
                n_rep, d_bytes, p_bytes = quantize_model_linears_4bit(
                    model,
                    group_size=self.quantize_group_size,
                    compute_dtype=self.dtype,
                )
                self.quantization_stats = {
                    "quantized_layers": n_rep,
                    "dense_bytes": d_bytes,
                    "packed_bytes": p_bytes,
                }
            if self.device != model_device or self.dtype != model_dtype:
                model = model.to(device=self.device, dtype=self.dtype)
                if self.quantize_base_weights:
                    for mod in model.modules():
                        if isinstance(mod, QuantizedFrozenLinear):
                            mod.compute_dtype = self.dtype

        self.model = model
        self.tokenizer = tokenizer
        _install_cpu_safe_kernels(self.model)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

        self._layers_path, self._layers = self._find_decoder_layers()
        # HF parameter prefix preceding canonical ``layers.{i}`` (e.g. ``model.``).
        self.layer_prefix: str = (
            self._layers_path[: -len("layers")]
            if self._layers_path.endswith("layers")
            else f"{self._layers_path}."
        )
        num_layers = len(self._layers)
        self.total_layers: int = num_layers
        if target_layer_indices is None:
            # Default: compact set covering middle + upper layers (same rule as MLXKernel).
            candidates = [7, 11, 15, 19, num_layers - 3, num_layers - 2, num_layers - 1]
            target_layer_indices = sorted({i for i in candidates if 0 <= i < num_layers})

        projections = target_projections if target_projections is not None else (projection_attr,)
        self.projection_attr = projection_attr
        self.target_projections: tuple[str, ...] = tuple(projections)
        self.instrumented: dict[str, SubcorTorchLinear] = {}
        out_of_range = [int(i) for i in target_layer_indices if not 0 <= int(i) < num_layers]
        if out_of_range:
            raise ValueError(
                f"target_layer_indices {out_of_range} are out of range for '{model_id}', which "
                f"has {num_layers} decoder layers (valid: 0..{num_layers - 1})"
            )
        for idx in target_layer_indices:
            for proj in projections:
                canonical = f"layers.{idx}.{proj}"
                try:
                    parent, attr = _resolve_parent_and_attr(self._layers[idx], proj)
                    base_mod = getattr(parent, attr)
                except (AttributeError, IndexError, KeyError):
                    # Hybrid architectures (e.g. Qwen3.5 GatedDeltaNet) omit self_attn on
                    # linear-attention layers (and vice versa), exactly like MLXKernel.
                    continue
                if isinstance(base_mod, SubcorTorchLinear):
                    wrapped = base_mod
                elif isinstance(base_mod, QuantizedFrozenLinear):
                    wrapped = SubcorTorchLinear(base_mod, layer_name=canonical)
                    setattr(parent, attr, wrapped)
                else:
                    if not isinstance(base_mod, nn.Linear):
                        raise TypeError(
                            f"Module at '{canonical}' is {type(base_mod).__name__}, "
                            "expected nn.Linear or QuantizedFrozenLinear"
                        )
                    if not base_mod.weight.is_floating_point():
                        raise TypeError(
                            f"Module at '{canonical}' has a {base_mod.weight.dtype} weight; "
                            "expected a dense floating-point nn.Linear (dequantized base)"
                        )
                    wrapped = SubcorTorchLinear(base_mod, layer_name=canonical)
                    setattr(parent, attr, wrapped)
                self.instrumented[canonical] = wrapped

        if not self.instrumented:
            raise ValueError(
                f"No target projections matched on '{model_id}' for layers={target_layer_indices} "
                f"and projections={tuple(projections)}"
            )

        fwd_params = inspect.signature(self.model.forward).parameters
        self._supports_logits_to_keep: bool = "logits_to_keep" in fwd_params
        self._eos_ids: list[int] = self._collect_eos_ids()
        self._verified_unit_cache: dict[str, tuple[str, list[tuple[str, LowRankAdapter]]]] = {}

    # ------------------------------------------------------------------ model plumbing
    def _find_decoder_layers(self) -> tuple[str, nn.ModuleList]:
        for path in _DECODER_LAYER_PATHS:
            try:
                obj = _get_attr_path(self.model, path)
            except (AttributeError, IndexError, TypeError):
                continue
            if isinstance(obj, nn.ModuleList) and len(obj) > 0:
                return path, obj
        raise AttributeError(
            f"Could not locate decoder layers on {type(self.model).__name__}; tried "
            f"{list(_DECODER_LAYER_PATHS)}"
        )

    def hf_module_path(self, layer_name: str) -> str:
        """Map a canonical ``layers.{i}.{proj}`` name to the HF module path on this model."""
        idx = _layer_index(layer_name)
        if idx is None:
            raise ValueError(f"Not a canonical layer name: {layer_name!r}")
        suffix = layer_name.split(".", 2)[2]
        return f"{self._layers_path}.{idx}.{suffix}"

    def _collect_eos_ids(self) -> list[int]:
        ids: list[int] = []

        def _add(val: Any) -> None:
            if val is None:
                return
            if isinstance(val, (list, tuple)):
                for v in val:
                    _add(v)
                return
            iv = int(val)
            if iv not in ids:
                ids.append(iv)

        gen_cfg = getattr(self.model, "generation_config", None)
        if gen_cfg is not None:
            _add(getattr(gen_cfg, "eos_token_id", None))
        _add(getattr(self.tokenizer, "eos_token_id", None))
        return ids

    def _encode(self, text: str) -> list[int]:
        ids = [int(t) for t in self.tokenizer.encode(text)]
        bos = getattr(self.tokenizer, "bos_token_id", None)
        # Chat templates usually embed BOS already; avoid a doubled BOS.
        if bos is not None and len(ids) >= 2 and ids[0] == ids[1] == int(bos):
            ids = ids[1:]
        return ids

    def _decode(self, ids: list[int]) -> str:
        return str(
            self.tokenizer.decode(
                ids, skip_special_tokens=False, clean_up_tokenization_spaces=False
            )
        )

    def _ids_tensor(self, ids: list[int]) -> torch.Tensor:
        return torch.tensor([ids], dtype=torch.long, device=self.device)

    def _logits_kw(self, n: int) -> dict[str, Any]:
        return {"logits_to_keep": int(n)} if self._supports_logits_to_keep else {}

    def _iter_trainable(self) -> Iterator[nn.Parameter]:
        for layer in self.instrumented.values():
            yield from layer.trainable_lora_parameters()

    # ------------------------------------------------------------------ prompting
    def format_messages(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        enable_thinking: bool = False,
        add_generation_prompt: bool = True,
    ) -> str:
        """Render full multi-turn OpenAI messages (+ optional tool schemas) via chat template."""
        cleaned: list[dict[str, Any]] = []
        for msg in messages:
            role = str(msg.get("role", "user"))
            entry: dict[str, Any] = {"role": role}
            content = msg.get("content")
            if content is not None:
                entry["content"] = str(content)
            elif role != "assistant":
                entry["content"] = ""
            if "tool_calls" in msg and msg["tool_calls"]:
                norm_calls: list[dict[str, Any]] = []
                for tc in msg["tool_calls"]:
                    tc_copy = dict(tc)
                    fn = tc_copy.get("function")
                    if isinstance(fn, dict):
                        fn_copy = dict(fn)
                        raw_args = fn_copy.get("arguments")
                        if isinstance(raw_args, str):
                            try:
                                parsed = json.loads(raw_args)
                                if isinstance(parsed, dict):
                                    fn_copy["arguments"] = parsed
                            except json.JSONDecodeError:
                                pass
                        tc_copy["function"] = fn_copy
                    norm_calls.append(tc_copy)
                entry["tool_calls"] = norm_calls
            if "tool_call_id" in msg and msg["tool_call_id"]:
                entry["tool_call_id"] = str(msg["tool_call_id"])
            if "name" in msg and msg["name"]:
                entry["name"] = str(msg["name"])
            cleaned.append(entry)

        kwargs: dict[str, Any] = {
            "tokenize": False,
            "add_generation_prompt": add_generation_prompt,
        }
        template = getattr(self.tokenizer, "chat_template", "") or ""
        if "enable_thinking" in template:
            kwargs["enable_thinking"] = enable_thinking
        if tools:
            kwargs["tools"] = tools
        try:
            return str(self.tokenizer.apply_chat_template(cleaned, **kwargs))
        except TypeError:
            # Fallback if a custom tokenizer template does not accept `tools`.
            kwargs.pop("tools", None)
            return str(self.tokenizer.apply_chat_template(cleaned, **kwargs))

    def format_chat_prompt(
        self,
        user_prompt: str,
        *,
        system_prompt: str | None = None,
        enable_thinking: bool = False,
    ) -> str:
        messages: list[dict[str, Any]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})
        return self.format_messages(messages, enable_thinking=enable_thinking)

    # ------------------------------------------------------------------ generation
    def _greedy_step(self, input_ids: torch.Tensor, cache: Any) -> tuple[int, float, Any]:
        """One frozen-base forward under inference_mode -> (token, logprob, new cache)."""
        with torch.inference_mode():
            out = self.model(
                input_ids=input_ids,
                past_key_values=cache,
                use_cache=True,
                **self._logits_kw(1),
            )
            logits = out.logits[0, -1, :].float()
            tok = int(torch.argmax(logits).item())
            lp = float((logits[tok] - torch.logsumexp(logits, dim=-1)).item())
        return tok, lp, out.past_key_values

    def stream_formatted(
        self,
        formatted_prompt: str,
        *,
        max_tokens: int = 128,
        capture_activations: bool = False,
    ) -> Iterator[StreamTokenEvent]:
        """Yield true token-by-token greedy StreamTokenEvents with exact per-token logprobs.

        Event semantics mirror ``mlx_lm.stream_generate``: one event per generated
        token; EOS is not emitted as text but its event carries ``finish_reason="stop"``
        (and counts toward ``generation_tokens``); hitting ``max_tokens`` sets
        ``finish_reason="length"`` on the final event. Each forward runs in its own
        ``torch.inference_mode`` block so the grad mode never leaks across ``yield``.
        """
        if capture_activations:
            for layer in self.instrumented.values():
                layer.start_recording()
        if max_tokens <= 0:
            return

        prompt_ids = self._encode(formatted_prompt)
        n_prompt = len(prompt_ids)
        eos = set(self._eos_ids)
        generated: list[int] = []
        emitted = ""
        cache: Any = None
        next_input = self._ids_tensor(prompt_ids)

        for n in range(max_tokens):
            tok, lp, cache = self._greedy_step(next_input, cache)
            if tok in eos:
                rest = self._decode(generated)[len(emitted) :] if generated else ""
                yield StreamTokenEvent(
                    text=rest,
                    token_id=tok,
                    logprob=lp,
                    prompt_tokens=n_prompt,
                    generation_tokens=n + 1,
                    finish_reason="stop",
                )
                return
            generated.append(tok)
            full = self._decode(generated)
            last = n + 1 == max_tokens
            if last or not full.endswith("�"):
                piece = full[len(emitted) :] if full.startswith(emitted) else ""
                emitted = full if full.startswith(emitted) else emitted
            else:
                # Hold back incomplete multi-byte sequences until the next token completes them.
                piece = ""
            yield StreamTokenEvent(
                text=piece,
                token_id=tok,
                logprob=lp,
                prompt_tokens=n_prompt,
                generation_tokens=n + 1,
                finish_reason="length" if last else None,
            )
            next_input = self._ids_tensor([tok])

    def generate_messages_with_trace(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        max_tokens: int = 128,
        enable_thinking: bool = False,
        capture_activations: bool = False,
    ) -> GenerationTrace:
        """Run deterministic greedy multi-turn generation capturing logprobs & tool calls."""
        formatted = self.format_messages(
            messages,
            tools=tools,
            enable_thinking=enable_thinking,
        )
        chunks: list[str] = []
        logprobs: list[float] = []
        prompt_tokens = 0
        n_tokens = 0
        finish_reason: str | None = None

        try:
            for ev in self.stream_formatted(
                formatted,
                max_tokens=max_tokens,
                capture_activations=capture_activations,
            ):
                chunks.append(ev.text)
                logprobs.append(ev.logprob)
                prompt_tokens = ev.prompt_tokens
                n_tokens = ev.generation_tokens
                finish_reason = ev.finish_reason
        finally:
            activations: dict[str, np.ndarray] = {}
            if capture_activations:
                for name, layer in self.instrumented.items():
                    mean_vec = layer.stop_recording()
                    if mean_vec is not None:
                        activations[name] = mean_vec

        raw_response = "".join(chunks)
        clean_response, parsed_tools = (
            parse_tool_calls(raw_response) if tools else (raw_response, [])
        )
        hit_limit = (finish_reason == "length") or (n_tokens >= max_tokens)
        rep = ngram_repetition_ratio(raw_response)
        last_user = next(
            (str(m.get("content", "")) for m in reversed(messages) if m.get("role") == "user"),
            "",
        )

        return GenerationTrace(
            prompt=last_user,
            response=clean_response,
            tokens_generated=n_tokens,
            hit_token_limit=hit_limit,
            repetition_ratio=rep,
            layer_activations=activations,
            prompt_tokens=prompt_tokens,
            token_logprobs=tuple(logprobs),
            tool_calls=tuple(parsed_tools),
            raw_response=raw_response,
        )

    def generate_with_trace(
        self,
        user_prompt: str,
        *,
        system_prompt: str | None = None,
        max_tokens: int = 128,
        enable_thinking: bool = False,
        capture_activations: bool = False,
    ) -> GenerationTrace:
        """Run deterministic greedy single-turn generation with logprobs & activation capture."""
        messages: list[dict[str, Any]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})
        return self.generate_messages_with_trace(
            messages,
            max_tokens=max_tokens,
            enable_thinking=enable_thinking,
            capture_activations=capture_activations,
        )

    def probe_logits(
        self,
        user_prompt: str,
        *,
        system_prompt: str | None = None,
        enable_thinking: bool = False,
    ) -> np.ndarray:
        """Return next-token logits for a prompt (used to prove exact zero-perturbation)."""
        formatted = self.format_chat_prompt(
            user_prompt,
            system_prompt=system_prompt,
            enable_thinking=enable_thinking,
        )
        ids = self._ids_tensor(self._encode(formatted))
        with torch.inference_mode():
            out = self.model(input_ids=ids, use_cache=False, **self._logits_kw(1))
            logits = out.logits[0, -1, :].float().cpu()
        return np.array(logits.numpy(), dtype=np.float64, copy=True)

    # ------------------------------------------------------------------ skill units
    def _load_cached_unit(
        self, unit_id: str, registry: SkillUnitRegistry
    ) -> list[tuple[str, LowRankAdapter]]:
        """Verify SHA-256 on disk and cache unpacked adapters keyed by digest."""
        manifest = registry.get_manifest(unit_id)
        cached = self._verified_unit_cache.get(unit_id)
        if cached is not None and cached[0] == manifest.payload_sha256:
            return cached[1]
        deltas = registry.load_verified_deltas(unit_id)
        pairs: list[tuple[str, LowRankAdapter]] = []
        for delta in deltas:
            if delta.layer_name not in self.instrumented:
                raise KeyError(
                    f"Skill Unit '{unit_id}' targets layer '{delta.layer_name}' which is not "
                    f"instrumented in this TorchKernel (have {sorted(self.instrumented)})"
                )
            adapter = delta.to_adapter(adapter_name=f"unit:{unit_id}:{delta.layer_name}")
            pairs.append((delta.layer_name, adapter))
        self._verified_unit_cache[unit_id] = (manifest.payload_sha256, pairs)
        return pairs

    def invalidate_unit_cache(self, unit_id: str) -> None:
        self._verified_unit_cache.pop(unit_id, None)

    def mount_unit(self, unit_id: str, registry: SkillUnitRegistry) -> None:
        """Verify SHA-256 integrity and atomically hot-swap a Skill Unit onto instrumented layers.

        If any layer fails to mount (or the mount is interrupted), every layer adapter
        mounted by THIS call is unmounted in reverse order before re-raising, so no
        partial unit ever leaks into a later request. Adapters that were already
        mounted before the call (same verified payload) are left as they were.
        """
        loaded = self._load_cached_unit(unit_id, registry)
        mounted_layers: list[tuple[str, str]] = []
        try:
            for layer_name, adapter in loaded:
                layer = self.instrumented[layer_name]
                was_mounted = adapter.name in layer.mounted_names
                layer.mount_adapter(adapter.name, adapter)
                if not was_mounted:
                    mounted_layers.append((layer_name, adapter.name))
        except BaseException:
            for layer_name, adapter_name in reversed(mounted_layers):
                self.instrumented[layer_name].unmount_adapter(adapter_name)
            raise

    def unmount_unit(self, unit_id: str) -> None:
        """Atomically unmount all layer adapters belonging to unit_id."""
        prefix = f"unit:{unit_id}:"
        for layer in self.instrumented.values():
            for name in list(layer.mounted_names):
                if name.startswith(prefix):
                    layer.unmount_adapter(name)

    # ------------------------------------------------------------------ routing
    def encode_prompt_embedding(
        self,
        user_prompt: str,
        *,
        tap_layer: str,
        system_prompt: str | None = None,
        enable_thinking: bool = False,
    ) -> np.ndarray:
        """Compute an invariant frozen-base hidden activation embedding at tap_layer.

        Bypasses all mounted skill adapters (and any trainable branch) AND early-exits
        right after the tap decoder layer via a forward hook that raises a private
        control-flow exception, so downstream layers and the vocab lm_head never run.
        """
        if tap_layer not in self.instrumented:
            raise KeyError(f"tap_layer '{tap_layer}' is not instrumented in this TorchKernel")
        formatted = self.format_chat_prompt(
            user_prompt,
            system_prompt=system_prompt,
            enable_thinking=enable_thinking,
        )
        ids = self._ids_tensor(self._encode(formatted))

        tap = self.instrumented[tap_layer]
        tap_idx = _layer_index(tap_layer)
        stop_module: nn.Module = (
            self._layers[tap_idx] if tap_idx is not None and tap_idx < len(self._layers) else tap
        )

        def _stop_hook(_module: nn.Module, _args: Any, _output: Any) -> None:
            raise _StopForward

        prev_bypass = {name: layer._bypass_adapters for name, layer in self.instrumented.items()}
        for layer in self.instrumented.values():
            layer._bypass_adapters = True
        handle = stop_module.register_forward_hook(_stop_hook)
        tap.start_recording()
        mean_act: np.ndarray | None = None
        try:
            with torch.inference_mode(), contextlib.suppress(_StopForward):
                self.model(input_ids=ids, use_cache=False)
        finally:
            handle.remove()
            mean_act = tap.stop_recording()
            for name, layer in self.instrumented.items():
                layer._bypass_adapters = prev_bypass[name]

        if mean_act is None:
            raise RuntimeError(f"Failed to capture activation at tap_layer '{tap_layer}'")
        norm = float(np.linalg.norm(mean_act))
        return mean_act / norm if norm > 1e-12 else mean_act

    def generate_routed_messages(
        self,
        messages: list[dict[str, Any]],
        *,
        router: MoESkillRouter,
        registry: SkillUnitRegistry,
        tools: list[dict[str, Any]] | None = None,
        max_tokens: int = 128,
        enable_thinking: bool = False,
        top_k: int = 1,
        seat: str | None = None,
        capture_activations: bool = False,
    ) -> tuple[GenerationTrace, RoutingDecision]:
        """Route full multi-turn messages using frozen-base activations of the active user turn."""
        last_user = next(
            (str(m.get("content", "")) for m in reversed(messages) if m.get("role") == "user"),
            "",
        )
        emb = self.encode_prompt_embedding(
            last_user,
            tap_layer=router.tap_layer,
            system_prompt=None,
            enable_thinking=False,
        )
        decision = router.route(emb, top_k=top_k, seat=seat)

        mounted_units: list[str] = []
        try:
            for uid in decision.selected_units:
                self.mount_unit(uid, registry)
                mounted_units.append(uid)
            trace = self.generate_messages_with_trace(
                messages,
                tools=tools,
                max_tokens=max_tokens,
                enable_thinking=enable_thinking,
                capture_activations=capture_activations,
            )
        finally:
            for uid in reversed(mounted_units):
                self.unmount_unit(uid)

        return trace, decision

    def stream_routed_messages(
        self,
        messages: list[dict[str, Any]],
        *,
        router: MoESkillRouter | None,
        registry: SkillUnitRegistry | None,
        tools: list[dict[str, Any]] | None = None,
        max_tokens: int = 128,
        enable_thinking: bool = False,
        top_k: int = 1,
        seat: str | None = None,
    ) -> tuple[RoutingDecision, Iterator[StreamTokenEvent]]:
        """Route once, then return a live token-by-token iterator that cleans up mounts on exit."""
        last_user = next(
            (str(m.get("content", "")) for m in reversed(messages) if m.get("role") == "user"),
            "",
        )
        if router is not None and registry is not None and router.registered_units:
            emb = self.encode_prompt_embedding(
                last_user,
                tap_layer=router.tap_layer,
                system_prompt=None,
                enable_thinking=False,
            )
            decision = router.route(emb, top_k=top_k, seat=seat)
        else:
            decision = RoutingDecision(selected_units=(), scores={}, thresholds={})

        formatted = self.format_messages(messages, tools=tools, enable_thinking=enable_thinking)

        def _gen() -> Iterator[StreamTokenEvent]:
            mounted_units: list[str] = []
            try:
                for uid in decision.selected_units:
                    assert registry is not None
                    self.mount_unit(uid, registry)
                    mounted_units.append(uid)
                yield from self.stream_formatted(formatted, max_tokens=max_tokens)
            finally:
                for uid in reversed(mounted_units):
                    self.unmount_unit(uid)

        return decision, _gen()

    def generate_routed(
        self,
        user_prompt: str,
        *,
        router: MoESkillRouter,
        registry: SkillUnitRegistry,
        system_prompt: str | None = None,
        max_tokens: int = 128,
        enable_thinking: bool = False,
        top_k: int = 1,
        capture_activations: bool = False,
    ) -> tuple[GenerationTrace, RoutingDecision]:
        """Route a single user prompt through MoESkillRouter using frozen-base activations."""
        messages: list[dict[str, Any]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})
        return self.generate_routed_messages(
            messages,
            router=router,
            registry=registry,
            max_tokens=max_tokens,
            enable_thinking=enable_thinking,
            top_k=top_k,
            capture_activations=capture_activations,
        )

    # ------------------------------------------------------------------ training
    def _tokenize_training_example(
        self,
        item: (
            tuple[str, str]
            | tuple[str | None, str, str]
            | tuple[str | None, str, str, list[dict[str, Any]] | None]
            | dict[str, Any]
        ),
        *,
        default_system_prompt: str | None,
        enable_thinking: bool,
        max_seq_len: int,
    ) -> tuple[tuple[torch.Tensor, torch.Tensor, torch.Tensor, bool] | None, str]:
        """Tokenize a single-turn tuple or multi-turn trajectory with assistant-only loss masking.

        Returns ``((x_ids, y_target, loss_positions, is_contiguous_suffix), "ok")`` or
        ``(None, reason)`` where ``reason`` is ``"too_long"`` or ``"empty_target"``.
        """
        eos_id = getattr(self.tokenizer, "eos_token_id", None)
        eos_str = getattr(self.tokenizer, "eos_token", None)

        if isinstance(item, dict):
            raw_msgs = item.get("messages")
            if not isinstance(raw_msgs, list) or not raw_msgs:
                return None, "empty_target"
            messages = [dict(m) for m in raw_msgs]
            if default_system_prompt and messages[0].get("role") != "system":
                messages.insert(0, {"role": "system", "content": default_system_prompt})
            last_asst_idx = -1
            for idx, msg in enumerate(messages):
                if msg.get("role") == "assistant":
                    last_asst_idx = idx
            if last_asst_idx < 0:
                return None, "empty_target"
            messages = messages[: last_asst_idx + 1]
            tools = item.get("tools")

            full_text = self.format_messages(
                messages,
                tools=tools,
                enable_thinking=enable_thinking,
                add_generation_prompt=False,
            )
            full_ids = self._encode(full_text)
            if len(full_ids) < 2:
                return None, "empty_target"
            if len(full_ids) > max_seq_len:
                return None, "too_long"

            pos_set: set[int] = set()
            for idx, msg in enumerate(messages):
                if msg.get("role") != "assistant":
                    continue
                pref_text = self.format_messages(
                    messages[:idx],
                    tools=tools,
                    enable_thinking=enable_thinking,
                    add_generation_prompt=True,
                )
                upto_text = self.format_messages(
                    messages[: idx + 1],
                    tools=tools,
                    enable_thinking=enable_thinking,
                    add_generation_prompt=False,
                )
                if eos_str and upto_text.endswith(f"{eos_str}\n"):
                    upto_text = upto_text[:-1]
                if upto_text.startswith(pref_text):
                    start_tok = len(self._encode(pref_text))
                else:
                    prev_text = self.format_messages(
                        messages[:idx],
                        tools=tools,
                        enable_thinking=enable_thinking,
                        add_generation_prompt=False,
                    )
                    turn_chunk = upto_text[len(prev_text) :]
                    nl = turn_chunk.find("\n")
                    header_text = prev_text + (turn_chunk[: nl + 1] if nl != -1 else "")
                    start_tok = len(self._encode(header_text))
                end_tok = min(len(self._encode(upto_text)), len(full_ids))
                for tok_pos in range(start_tok, end_tok):
                    pred_pos = tok_pos - 1
                    if 0 <= pred_pos < len(full_ids) - 1:
                        pos_set.add(pred_pos)

            if not pos_set:
                return None, "empty_target"

            pos_list = sorted(pos_set)
            x_ids = self._ids_tensor(full_ids[:-1])
            loss_positions = torch.tensor(pos_list, dtype=torch.long, device=self.device)
            y_ids = torch.tensor(
                [full_ids[p + 1] for p in pos_list], dtype=torch.long, device=self.device
            )
            is_suffix = pos_list == list(
                range(len(full_ids) - 1 - len(pos_list), len(full_ids) - 1)
            )
            return (x_ids, y_ids, loss_positions, is_suffix), "ok"

        if len(item) == 2:
            ex_sys, user_prompt, target, ex_tools = (
                default_system_prompt,
                item[0],
                item[1],
                None,
            )
        elif len(item) == 3:
            ex_sys, user_prompt, target, ex_tools = item[0], item[1], item[2], None
        else:
            ex_sys, user_prompt, target, ex_tools = item[0], item[1], item[2], item[3]

        msgs: list[dict[str, Any]] = []
        if ex_sys:
            msgs.append({"role": "system", "content": ex_sys})
        msgs.append({"role": "user", "content": user_prompt})
        prefix = self.format_messages(
            msgs,
            tools=ex_tools,
            enable_thinking=enable_thinking,
            add_generation_prompt=True,
        )
        prompt_ids = self._encode(prefix)
        target_ids = [int(t) for t in self.tokenizer.encode(target, add_special_tokens=False)]
        if eos_id is not None and (not target_ids or target_ids[-1] != int(eos_id)):
            target_ids = [*target_ids, int(eos_id)]
        full_ids = prompt_ids + target_ids
        if len(full_ids) < 2 or not prompt_ids or not target_ids:
            return None, "empty_target"
        if len(full_ids) > max_seq_len:
            return None, "too_long"

        x_ids = self._ids_tensor(full_ids[:-1])
        pos_list = list(range(len(prompt_ids) - 1, len(full_ids) - 1))
        loss_positions = torch.tensor(pos_list, dtype=torch.long, device=self.device)
        y_ids = torch.tensor(target_ids, dtype=torch.long, device=self.device)
        return (x_ids, y_ids, loss_positions, True), "ok"

    def train_qlora_adapters(
        self,
        pairs: Sequence[
            tuple[str, str]
            | tuple[str | None, str, str]
            | tuple[str | None, str, str, list[dict[str, Any]] | None]
            | dict[str, Any]
        ],
        *,
        layer_names: list[str],
        system_prompt: str | None = None,
        enable_thinking: bool = False,
        rank: int = 8,
        scale: float = 1.0,
        binary_ste: bool = False,
        steps: int = 108,
        lr: float = 3e-4,
        max_seq_len: int = 2048,
        grad_accum: int = 1,
        seed: int = 42,
        gradient_checkpointing: bool = False,
    ) -> tuple[dict[str, LowRankAdapter], list[float], float]:
        """Function-preserving LoRA expansion + QLoRA adaptation on a frozen base.

        Freezes the entire base model (dense or packed 4-bit :class:`QuantizedFrozenLinear`),
        attaches zero-initialized FP32 LoRA branches to ``layer_names``, verifies exact zero
        logit perturbation before step 0, trains ONLY the new LoRA weights via assistant-only
        masked causal-LM cross-entropy on single-turn tuples or multi-turn trajectory dicts
        (``{"messages": [...], "tools": [...]}``), then extracts and detaches the trained
        adapters. With ``binary_ste=True`` runs Two-Stage QAT (70% FP32 warm start, then
        Binary STE refinement).

        Only masked assistant/completion positions are projected through ``lm_head``
        (``logits_to_keep`` for contiguous suffixes, or direct hidden-state gathering before
        ``lm_head`` for multi-turn trajectories), keeping peak memory bounded on 12-16 GB
        GPUs.

        Returns ``(adapters_by_layer, loss_history, zero_perturbation_max_diff)``.
        """
        if not pairs:
            raise ValueError("pairs must be non-empty")
        if not layer_names:
            raise ValueError("layer_names must be non-empty")
        for name in layer_names:
            if name not in self.instrumented:
                raise KeyError(f"Layer '{name}' is not instrumented")

        first_item = pairs[0]
        if isinstance(first_item, dict):
            first_msgs = list(first_item.get("messages") or [])
            probe_sys = next(
                (str(m["content"]) for m in first_msgs if m.get("role") == "system"),
                system_prompt,
            )
            probe_prompt = next(
                (str(m["content"]) for m in first_msgs if m.get("role") == "user"),
                "Hello",
            )
        elif len(first_item) == 2:
            probe_sys, probe_prompt = system_prompt, first_item[0]
        else:
            probe_sys, probe_prompt = first_item[0], first_item[1]

        # Pre-tokenize all examples with assistant-only loss masking and explicit drop stats.
        encoded_examples: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, bool]] = []
        dropped_too_long = 0
        dropped_empty_target = 0
        for item in pairs:
            tok_ex, reason = self._tokenize_training_example(
                item,
                default_system_prompt=system_prompt,
                enable_thinking=enable_thinking,
                max_seq_len=max_seq_len,
            )
            if tok_ex is not None:
                encoded_examples.append(tok_ex)
            elif reason == "too_long":
                dropped_too_long += 1
            else:
                dropped_empty_target += 1

        if dropped_too_long > 0 or dropped_empty_target > 0:
            logger.warning(
                "Torch QLoRA tokenization kept %d/%d examples (dropped_too_long=%d > max_seq_len=%d, "
                "dropped_empty_target=%d).",
                len(encoded_examples),
                len(pairs),
                dropped_too_long,
                max_seq_len,
                dropped_empty_target,
            )
        if not encoded_examples:
            raise ValueError(
                f"No valid tokenized examples produced from {len(pairs)} inputs "
                f"(dropped_too_long={dropped_too_long} > max_seq_len={max_seq_len}, "
                f"dropped_empty_target={dropped_empty_target})"
            )

        prev_training = bool(self.model.training)
        prev_requires_grad = [(p, p.requires_grad) for p in self.model.parameters()]
        extracted: dict[str, LowRankAdapter] = {}
        losses: list[float] = []
        gc_enabled = False
        accum = max(1, int(grad_accum))

        # Locate inner base model + output head for sparse assistant-only slicing before lm_head.
        base_backbone = getattr(self.model, "model", None)
        out_head = (
            self.model.get_output_embeddings()
            if hasattr(self.model, "get_output_embeddings")
            else getattr(self.model, "lm_head", None)
        )

        try:
            # Measure live logits BEFORE expansion (in eval mode).
            self.model.eval()
            logits_before = self.probe_logits(
                probe_prompt, system_prompt=probe_sys, enable_thinking=enable_thinking
            )

            # Freeze entire model, then attach zero-initialized FP32 LoRA branches.
            for p in self.model.parameters():
                p.requires_grad_(False)
            for idx, name in enumerate(layer_names):
                self.instrumented[name].attach_trainable_lora(
                    rank=rank, scale=scale, binary_ste=False, seed=seed + idx
                )

            # Verify the strict zero-perturbation invariant on live logits.
            logits_after = self.probe_logits(
                probe_prompt, system_prompt=probe_sys, enable_thinking=enable_thinking
            )
            zero_perturb_diff = float(np.max(np.abs(logits_after - logits_before)))

            fp32_steps = int(round(steps * 0.70)) if binary_ste and steps >= 6 else steps
            ste_steps = steps - fp32_steps
            fp32_lr = min(lr, 4e-4)
            ste_lr = min(lr * 0.25, 8e-5)

            rng = np.random.default_rng(seed)
            order = np.arange(len(encoded_examples))
            ex_cursor = 0

            def _next_example() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, bool]:
                nonlocal ex_cursor
                if ex_cursor % len(encoded_examples) == 0:
                    rng.shuffle(order)
                idx = int(order[ex_cursor % len(encoded_examples)])
                ex_cursor += 1
                return encoded_examples[idx]

            def _loss(
                x_ids: torch.Tensor,
                y_ids: torch.Tensor,
                loss_positions: torch.Tensor,
                is_suffix: bool,
            ) -> torch.Tensor:
                if is_suffix and self._supports_logits_to_keep:
                    n_tgt = int(y_ids.shape[0])
                    logits = self.model(
                        input_ids=x_ids, use_cache=False, **self._logits_kw(n_tgt)
                    ).logits
                    logits = logits[0, -n_tgt:, :].float()
                    return F.cross_entropy(logits, y_ids, reduction="mean")
                if base_backbone is not None and out_head is not None:
                    # Gather ONLY assistant/tool-call hidden positions before the vocab lm_head!
                    hidden = base_backbone(input_ids=x_ids, use_cache=False)[0]
                    h_sel = hidden[:, loss_positions, :]
                    logits = out_head(h_sel)[0].float()
                    return F.cross_entropy(logits, y_ids, reduction="mean")
                logits = self.model(input_ids=x_ids, use_cache=False).logits
                logits = logits[0, loss_positions, :].float()
                return F.cross_entropy(logits, y_ids, reduction="mean")

            def _run_stage(
                n_steps: int,
                init_lr: float,
                end_lr: float,
                weight_decay: float,
                clip_latents: bool,
            ) -> None:
                params = list(self._iter_trainable())
                if n_steps <= 0 or not params:
                    return
                opt = torch.optim.AdamW(
                    params, lr=init_lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=weight_decay
                )
                for step in range(n_steps):
                    for group in opt.param_groups:
                        group["lr"] = _cosine_decay_lr(init_lr, max(1, n_steps), end_lr, step)
                    opt.zero_grad(set_to_none=True)
                    step_loss = 0.0
                    for _ in range(accum):
                        x_ids, y_ids, loss_positions, is_suffix = _next_example()
                        loss = _loss(x_ids, y_ids, loss_positions, is_suffix)
                        (loss / float(accum)).backward()
                        step_loss += float(loss.detach().item())
                    grad_norm = torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
                    mean_loss = step_loss / float(accum)
                    if math.isfinite(mean_loss) and bool(torch.isfinite(grad_norm).item()):
                        opt.step()
                    if clip_latents:
                        with torch.no_grad():
                            for name in layer_names:
                                lyr = self.instrumented[name]
                                if lyr.lora_a is not None and lyr.lora_b is not None:
                                    lyr.lora_a.clamp_(-1.0, 1.0)
                                    lyr.lora_b.clamp_(-1.0, 1.0)
                    losses.append(mean_loss)
                opt.zero_grad(set_to_none=True)

            if gradient_checkpointing and getattr(
                self.model, "supports_gradient_checkpointing", False
            ):
                if hasattr(self.model, "activation_checkpointing_enable"):
                    self.model.activation_checkpointing_enable(
                        activation_checkpointing_kwargs={"use_reentrant": False}
                    )
                else:  # transformers < 5.x name
                    self.model.gradient_checkpointing_enable(
                        gradient_checkpointing_kwargs={"use_reentrant": False}
                    )
                gc_enabled = True

            self.model.train()
            with torch.inference_mode(False), torch.enable_grad():
                # Stage 1: FP32 continuous warm-start.
                _run_stage(fp32_steps, fp32_lr, fp32_lr * 0.15, 0.01, clip_latents=False)

                # Stage 2: Binary STE (QAT) refinement on {-1, +1} factors + scales.
                if ste_steps > 0:
                    for name in layer_names:
                        self.instrumented[name].convert_lora_to_binary_ste()
                    _run_stage(ste_steps, ste_lr, ste_lr * 0.1, 0.0, clip_latents=True)

            self.last_train_stats = {
                "total_input": len(pairs),
                "kept": len(encoded_examples),
                "dropped_too_long": dropped_too_long,
                "dropped_empty_target": dropped_empty_target,
                "max_seq_len": max_seq_len,
                "grad_accum": accum,
                "quantize_base_weights": self.quantize_base_weights,
                "gradient_checkpointing": gc_enabled,
            }

            for name in layer_names:
                extracted[name] = self.instrumented[name].extract_trainable_lora(
                    adapter_name=f"trained:{name}"
                )
        finally:
            for name in layer_names:
                self.instrumented[name].detach_trainable_lora()
            if gc_enabled:
                if hasattr(self.model, "activation_checkpointing_disable"):
                    self.model.activation_checkpointing_disable()
                else:
                    self.model.gradient_checkpointing_disable()
            for p, flag in prev_requires_grad:
                p.requires_grad_(flag)
                p.grad = None
            self.model.train(prev_training)

        return extracted, losses, zero_perturb_diff

    def trainable_parameter_count(self) -> int:
        return int(sum(p.numel() for p in self.model.parameters() if p.requires_grad))


def _dequantize_gguf_config() -> Any:
    """`transformers.GgufConfig(dequantize=True)` (None on transformers without GgufConfig)."""
    try:
        from transformers import GgufConfig
    except ImportError:  # pragma: no cover - older transformers always dequantize
        return None
    return GgufConfig(dequantize=True)


def load_trainer_from_gguf(gguf_path: str | Path, **kw: Any) -> TorchKernel:
    """Build a TorchKernel trainer over the dequantized weights of a local GGUF file.

    ``transformers`` dequantizes the GGUF (to ``dtype``: fp16 on GPU, fp32 on CPU by
    default), so Skill Units are trained against the same weights ``llama-server``
    serves from that file. ``model_id`` defaults to the absolute GGUF path; every
    other keyword is forwarded to :class:`TorchKernel`.
    """
    path = Path(gguf_path).expanduser().absolute()
    if not path.is_file():
        raise FileNotFoundError(f"GGUF file not found: {path}")
    model_id = str(kw.pop("model_id", str(path)))
    return TorchKernel(model_id, gguf_file=str(path), **kw)


def export_peft_lora_safetensors(
    unit_id: str,
    registry: SkillUnitRegistry,
    destination_dir: Path,
    *,
    base_model_name_or_path: str = DEFAULT_TORCH_MODEL,
    module_prefix: str = "model.",
) -> Path:
    """Export a verified Skill Unit as a stock PEFT LoRA adapter directory.

    Writes ``adapter_model.safetensors`` (fp32 ``lora_A`` (r, d_in) / ``lora_B`` (d_out, r),
    with the unit scale folded into ``lora_B``) and ``adapter_config.json`` with
    ``lora_alpha == r`` so PEFT's scaling is exactly 1.0. ``module_prefix`` is the HF
    path preceding ``layers.`` (``TorchKernel.layer_prefix``, e.g. ``model.``).
    """
    from safetensors.torch import save_file

    manifest = registry.get_manifest(unit_id)
    deltas: list[LayerDeltaPayload] = registry.load_verified_deltas(unit_id)

    destination_dir = Path(destination_dir)
    destination_dir.mkdir(parents=True, exist_ok=True)
    tensors: dict[str, torch.Tensor] = {}
    module_ranks: dict[str, int] = {}

    for delta in deltas:
        adapter = delta.to_adapter(adapter_name=delta.layer_name)
        a_np, b_np = adapter.a, adapter.scale * adapter.b
        module_path = f"{module_prefix}{delta.layer_name}"
        prefix = f"base_model.model.{module_path}"
        tensors[f"{prefix}.lora_A.weight"] = torch.from_numpy(
            np.ascontiguousarray(a_np, dtype=np.float32)
        )
        tensors[f"{prefix}.lora_B.weight"] = torch.from_numpy(
            np.ascontiguousarray(b_np, dtype=np.float32)
        )
        module_ranks[module_path] = int(a_np.shape[0])

    save_file(
        tensors, str(destination_dir / "adapter_model.safetensors"), metadata={"format": "pt"}
    )

    r_main = max(module_ranks.values()) if module_ranks else 1
    rank_pattern = {m: rk for m, rk in module_ranks.items() if rk != r_main}
    config: dict[str, Any] = {
        "peft_type": "LORA",
        "task_type": "CAUSAL_LM",
        "base_model_name_or_path": base_model_name_or_path,
        "r": r_main,
        "lora_alpha": r_main,
        "lora_dropout": 0.0,
        "bias": "none",
        "fan_in_fan_out": False,
        "use_rslora": False,
        "use_dora": False,
        "init_lora_weights": True,
        "inference_mode": True,
        "target_modules": sorted(module_ranks),
        "rank_pattern": rank_pattern,
        "alpha_pattern": dict(rank_pattern),
        "modules_to_save": None,
        "subcor_unit": {
            "unit_id": manifest.unit_id,
            "version": manifest.version,
            "base_kernel_id": manifest.base_kernel_id,
            "payload_sha256": manifest.payload_sha256,
            "effective_bits_per_weight": round(manifest.effective_bits_per_weight, 6),
        },
    }
    (destination_dir / "adapter_config.json").write_text(
        json.dumps(config, indent=2) + "\n",
        encoding="utf-8",
    )
    return destination_dir
