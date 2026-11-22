"""Apple Silicon MLX frozen-kernel runtime with hot-swappable SUBCOR Skill Units.

Wraps any compact mlx-lm quantized checkpoint (e.g., Qwen3.5-0.8B-4bit or
Gemma-4-E2B-it-qat-4bit) as an immutable base kernel:

  - Base QuantizedLinear weights are strictly frozen and never overwritten.
  - Instrumented layers are wrapped in SubcorMLXLinear, supporting:
      * Live mean hidden-state activation recording during generation.
      * Multi-unit hot-swap mounting and atomic rollback of Sub-1-Bit or FP32
        Skill Units verified by SHA-256 via SkillUnitRegistry.
      * Function-preserving zero-initialized LoRA expansion (B = 0) and
        Apple Silicon GPU QLoRA adaptation via mlx.nn.value_and_grad.
      * On-demand base weight dequantization for zero-gradient weight-space
        surgery without mutating the quantized base weights.
      * Export to stock mlx-lm adapters.safetensors + adapter_config.json.
"""

from __future__ import annotations

import contextlib
import json
import logging
import math
import time
from collections import OrderedDict
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from mlx.utils import tree_flatten
from mlx_lm import load, stream_generate
from mlx_lm.models.base import create_attention_mask, create_ssm_mask
from mlx_lm.sample_utils import make_sampler

from subcor.confidence import parse_tool_calls
from subcor.expansion import LowRankAdapter
from subcor.intern import ngram_repetition_ratio
from subcor.router import MoESkillRouter, RoutingDecision
from subcor.units import LayerDeltaPayload, SkillUnitRegistry

DEFAULT_COMPACT_MODEL = "mlx-community/Qwen3.5-0.8B-4bit"
logger = logging.getLogger(__name__)


def chunked_gated_delta_ops(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    g: mx.array,
    beta: mx.array,
    state: mx.array | None = None,
    mask: mx.array | None = None,
    *,
    chunk_size: int = 32,
    fallback_ops: Any = None,
) -> tuple[mx.array, mx.array]:
    """Memory-safe, mathematically exact chunkwise-parallel Gated Delta Rule (Yang et al., 2024).

    Replaces mlx-lm's O(T) unrolled Python recurrence during backprop (which allocates
    O(T * Hv * Dv * Dk) intermediate state tensors and OOMs on 16 GB unified memory at
    seq_len=2048) with chunk-checkpointed FP32 matmuls + overflow-free forward substitution
    for `(I + M) U = rhs`, keeping peak RAM flat across multi-layer hybrid training.
    """
    b_size, seq_len, hk, dk = q.shape
    hv, dv = v.shape[-2:]
    in_dtype = q.dtype
    state_f32 = (
        mx.zeros((b_size, hv, dv, dk), dtype=mx.float32)
        if state is None
        else state.astype(mx.float32)
    )
    if mask is not None or g.ndim == 4:
        if fallback_ops is not None:
            return fallback_ops(q, k, v, g, beta, state_f32, mask)
        from mlx_lm.models.gated_delta import gated_delta_ops as _stock_ops

        return _stock_ops(q, k, v, g, beta, state_f32, mask)

    if (repeat_factor := hv // hk) > 1:
        q = mx.repeat(q, repeat_factor, -2)
        k = mx.repeat(k, repeat_factor, -2)

    q_h = q.transpose(0, 2, 1, 3).astype(mx.float32)
    k_h = k.transpose(0, 2, 1, 3).astype(mx.float32)
    v_h = v.transpose(0, 2, 1, 3).astype(mx.float32)
    g_h = mx.clip(g.transpose(0, 2, 1).astype(mx.float32), 1e-6, 1.0)
    b_h = beta.transpose(0, 2, 1).astype(mx.float32)
    log_g = mx.log(g_h)

    def _run_chunk(
        qc: mx.array,
        kc: mx.array,
        vc: mx.array,
        lgc: mx.array,
        bc: mx.array,
        s_prev: mx.array,
    ) -> tuple[mx.array, mx.array]:
        c_len = qc.shape[2]
        gamma = mx.cumsum(lgc, axis=-1)  # (B, Hv, C)
        idx = mx.arange(c_len)
        tril_strict = idx[:, None] > idx[None, :]
        tril_incl = idx[:, None] >= idx[None, :]
        diff = gamma[:, :, :, None] - gamma[:, :, None, :]
        safe_strict = mx.clip(mx.where(tril_strict, diff, -40.0), -40.0, 0.0)
        safe_incl = mx.clip(mx.where(tril_incl, diff, -40.0), -40.0, 0.0)
        decay_strict = mx.where(tril_strict, mx.exp(safe_strict), 0.0)
        decay_incl = mx.where(tril_incl, mx.exp(safe_incl), 0.0)

        ktk = kc @ kc.swapaxes(-1, -2)
        m_mat = (bc[:, :, :, None] * decay_strict) * ktk

        # Overflow-free forward substitution for (I + M) U = rhs (never forms M^p).
        exp_gamma = mx.exp(mx.clip(gamma, -40.0, 0.0))
        ks = kc @ s_prev.swapaxes(-1, -2)
        rhs = bc[:, :, :, None] * (vc - exp_gamma[:, :, :, None] * ks)
        u_rows = [rhs[:, :, 0:1, :]]
        for t in range(1, c_len):
            u_prev = mx.concatenate(u_rows, axis=2)
            m_row = m_mat[:, :, t : t + 1, :t]
            u_rows.append(rhs[:, :, t : t + 1, :] - (m_row @ u_prev))
        u_mat = mx.concatenate(u_rows, axis=2)

        qtk = qc @ kc.swapaxes(-1, -2)
        y_chunk = (exp_gamma[:, :, :, None] * qc) @ s_prev.swapaxes(-1, -2) + (
            qtk * decay_incl
        ) @ u_mat
        g_end = gamma[:, :, -1:]
        scale_k = mx.exp(mx.clip(g_end - gamma, -40.0, 0.0))
        s_next = mx.exp(mx.clip(g_end, -40.0, 0.0))[:, :, :, None] * s_prev + u_mat.swapaxes(
            -1, -2
        ) @ (scale_k[:, :, :, None] * kc)
        return y_chunk, s_next

    ckpt_chunk = mx.checkpoint(_run_chunk)
    ys: list[mx.array] = []
    cur_state = state_f32
    for start in range(0, seq_len, chunk_size):
        end = min(start + chunk_size, seq_len)
        yc, cur_state = ckpt_chunk(
            q_h[:, :, start:end, :],
            k_h[:, :, start:end, :],
            v_h[:, :, start:end, :],
            log_g[:, :, start:end],
            b_h[:, :, start:end],
            cur_state,
        )
        ys.append(yc)

    y_all = mx.concatenate(ys, axis=2).transpose(0, 2, 1, 3).astype(in_dtype)
    return y_all, cur_state


@contextlib.contextmanager
def _chunked_gated_delta_context(*, enabled: bool = True, chunk_size: int = 32) -> Iterator[bool]:
    """Temporarily enable chunkwise-parallel GatedDeltaNet ops during training only."""
    if not enabled:
        yield False
        return
    try:
        import mlx_lm.models.gated_delta as gd
    except ImportError:
        yield False
        return

    orig_ops = gd.gated_delta_ops

    def _patched(
        q: mx.array,
        k: mx.array,
        v: mx.array,
        g: mx.array,
        beta: mx.array,
        state: mx.array | None = None,
        mask: mx.array | None = None,
    ) -> tuple[mx.array, mx.array]:
        return chunked_gated_delta_ops(
            q,
            k,
            v,
            g,
            beta,
            state=state,
            mask=mask,
            chunk_size=chunk_size,
            fallback_ops=orig_ops,
        )

    gd.gated_delta_ops = _patched
    try:
        yield True
    finally:
        gd.gated_delta_ops = orig_ops


def select_layer_indices(
    model: nn.Module,
    *,
    preset: str = "default",
    num_layers: int | None = None,
    stride: int = 4,
) -> list[int]:
    """Select transformer block indices for multi-layer adaptation on standard or hybrid models.

    Supported presets:
      - ``"default"``: compact middle + upper layer spread (backward-compatible default).
      - ``"last_n"``: the final ``num_layers`` contiguous blocks (default 4).
      - ``"periodic_full_attn"``: the last ``num_layers`` full-attention blocks
        (``not getattr(layer, "is_linear", False)``) across depth on hybrid models such as
        Qwen3.5 (e.g. ``[19, 23, 27, 31]`` on 32-layer 4B); falls back to strided layers on
        pure full-attention architectures.
      - ``"every_k"``: every ``stride``-th block ending at ``total_layers - 1``.
    """
    layers = getattr(model, "layers", None)
    if layers is None:
        inner = getattr(model, "model", None)
        layers = getattr(inner, "layers", [])
    total = len(layers)
    if total == 0:
        raise ValueError("Model exposes 0 transformer layers")

    key = preset.strip().lower()
    if key == "default":
        candidates = [7, 11, 15, 19, total - 3, total - 2, total - 1]
        return sorted({i for i in candidates if 0 <= i < total})

    if key == "last_n":
        count = max(1, min(total, num_layers if num_layers is not None else 4))
        return list(range(total - count, total))

    if key == "periodic_full_attn":
        full_attn = [i for i, lyr in enumerate(layers) if not getattr(lyr, "is_linear", False)]
        count = max(1, num_layers if num_layers is not None else 4)
        if full_attn and len(full_attn) < total:
            # Hybrid model (e.g. Qwen3.5 GatedDeltaNet + periodic full-attention).
            return full_attn[-count:]
        # Pure full-attention model: distribute periodically from the top down.
        step = max(1, stride)
        picked = list(range(total - 1, -1, -step))[:count]
        return sorted(picked)

    if key == "every_k":
        step = max(1, stride)
        picked = list(range(total - 1, -1, -step))
        if num_layers is not None:
            picked = picked[: max(1, num_layers)]
        return sorted(picked)

    raise ValueError(
        f"Unknown layer_preset={preset!r}; expected one of 'default', 'last_n', "
        "'periodic_full_attn', 'every_k'"
    )


def _normalize_chat_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Normalize OpenAI chat messages (including JSON-string tool_call arguments) for Jinja."""
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
    return cleaned


def _ste_sign_pm1(z: mx.array) -> mx.array:
    """Binary sign in {-1, +1} with clipped Straight-Through Estimator (|z| <= 1)."""
    hard = mx.where(z >= 0, mx.array(1.0, dtype=z.dtype), mx.array(-1.0, dtype=z.dtype))
    clipped = mx.clip(z, -1.0, 1.0)
    return clipped + mx.stop_gradient(hard - clipped)


@dataclass(frozen=True)
class GenerationTrace:
    """Result of one deterministic MLX generation with logprobs, tool calls, and activations."""

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
    """Incremental token emitted during true token-by-token MLX streaming."""

    text: str
    token_id: int
    logprob: float
    prompt_tokens: int
    generation_tokens: int
    finish_reason: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)


class SubcorMLXLinear(nn.Module):
    """Hot-swappable adapter wrapper around a frozen MLX Linear / QuantizedLinear layer."""

    def __init__(self, base: nn.Module, layer_name: str) -> None:
        super().__init__()
        self.base = base
        self.layer_name = layer_name
        # Native compute dtype (e.g. bfloat16/float16) avoids per-token FP32 casts on decode.
        if isinstance(base, nn.QuantizedLinear) and hasattr(base, "scales"):
            self._compute_dtype = base.scales.dtype
        elif hasattr(base, "weight"):
            self._compute_dtype = base.weight.dtype
        else:
            self._compute_dtype = mx.float16

        # Mounted frozen adapters stored pre-scaled in native compute dtype: (a_t, b_t_scaled).
        self._mounted: dict[str, tuple[mx.array, mx.array]] = {}

        # Optional trainable LoRA / Binary STE (QAT) branch for MLX autodiff.
        self.lora_a: mx.array | None = None  # (d_in, rank)
        self.lora_b: mx.array | None = None  # (rank, d_out)
        self.s_in: mx.array | None = None  # (d_in,) used when _lora_binary_ste=True
        self.s_out: mx.array | None = None  # (d_out,) used when _lora_binary_ste=True
        self.sigma: mx.array | None = None  # (rank,) used when _lora_binary_ste=True
        self._lora_scale: float = 1.0
        self._lora_binary_ste: bool = False
        self._lora_active: bool = False

        # Toggleable mean activation recorder (stores numpy accumulator off-graph).
        self._recording: bool = False
        self._act_sum: np.ndarray | None = None
        self._act_count: int = 0
        self._bypass_adapters: bool = False

        # Freeze base layer strictly.
        self.base.freeze()

    @property
    def shape(self) -> tuple[int, int]:
        """Return (d_out, d_in) of the underlying weight matrix."""
        out_dims, in_dims = self.base.weight.shape
        if isinstance(self.base, nn.QuantizedLinear):
            in_dims = in_dims * 32 // self.base.bits
        return int(out_dims), int(in_dims)

    @property
    def mounted_names(self) -> tuple[str, ...]:
        return tuple(self._mounted.keys())

    def dequantize_base_weight(self) -> np.ndarray:
        """Return a float64 copy of the frozen base weight matrix (d_out, d_in)."""
        if isinstance(self.base, nn.QuantizedLinear):
            w = mx.dequantize(
                self.base.weight,
                self.base.scales,
                self.base.biases,
                group_size=self.base.group_size,
                bits=self.base.bits,
                mode=self.base.mode,
            )
        else:
            w = self.base.weight
        mx.eval(w)
        return np.array(w.astype(mx.float32), dtype=np.float64, copy=True)

    def start_recording(self) -> None:
        self._recording = True
        self._act_sum = None
        self._act_count = 0

    def stop_recording(self) -> np.ndarray | None:
        self._recording = False
        if self._act_sum is None or self._act_count == 0:
            return None
        return self._act_sum / float(self._act_count)

    def mount_adapter(self, name: str, adapter: LowRankAdapter) -> None:
        """Mount a frozen LowRankAdapter (Delta W = scale * B @ A) in native compute dtype."""
        d_out, d_in = self.shape
        if adapter.d_out != d_out or adapter.d_in != d_in:
            raise ValueError(
                f"Adapter '{name}' shape ({adapter.d_out}, {adapter.d_in}) does not match "
                f"layer '{self.layer_name}' shape ({d_out}, {d_in})"
            )
        # Pre-fold scale into b_t and store in native compute dtype -> zero casts per decode token.
        a_t = mx.array(adapter.a.T.astype(np.float32)).astype(self._compute_dtype)
        b_t = mx.array((adapter.scale * adapter.b.T).astype(np.float32)).astype(self._compute_dtype)
        mx.eval(a_t, b_t)
        self._mounted[name] = (a_t, b_t)

    def unmount_adapter(self, name: str) -> None:
        self._mounted.pop(name, None)

    def attach_trainable_lora(
        self,
        *,
        rank: int,
        scale: float = 1.0,
        binary_ste: bool = False,
        seed: int = 0,
    ) -> None:
        """Attach a zero-initialized trainable LoRA or Binary STE (QAT) branch."""
        d_out, d_in = self.shape
        rng = np.random.default_rng(seed)
        self._lora_scale = float(scale)
        self._lora_binary_ste = binary_ste
        self._lora_active = True
        self.base.freeze()

        if binary_ste:
            # Matches Sub1BitMatrix 1-to-1; s_out=0 guarantees exact zero initial perturbation.
            self.lora_a = mx.array(rng.uniform(-0.4, 0.4, size=(d_in, rank)).astype(np.float32))
            self.lora_b = mx.array(rng.uniform(-0.4, 0.4, size=(rank, d_out)).astype(np.float32))
            self.s_in = mx.ones((d_in,), dtype=mx.float32)
            self.s_out = mx.zeros((d_out,), dtype=mx.float32)
            self.sigma = mx.full((rank,), 1.0 / math.sqrt(d_in * rank), dtype=mx.float32)
            mx.eval(self.lora_a, self.lora_b, self.s_in, self.s_out, self.sigma)
        else:
            bound = 1.0 / math.sqrt(d_in)
            self.lora_a = mx.array(rng.uniform(-bound, bound, size=(d_in, rank)).astype(np.float32))
            self.lora_b = mx.zeros((rank, d_out), dtype=mx.float32)
            self.s_in = None
            self.s_out = None
            self.sigma = None
            mx.eval(self.lora_a, self.lora_b)

    def convert_lora_to_binary_ste(self) -> None:
        """Warm-start Binary STE (QAT) parameters directly from the current FP32 LoRA delta.

        Mirrors Gemma 4 QAT: continuous warm-start lands in a good basin first, then
        initializing STE from Sub1BitMatrix lets STE refine around {-1, +1} rounding
        without random-sign seed instability.
        """
        if not self._lora_active or self._lora_binary_ste:
            return
        fp32_adapter = self.extract_trainable_lora(adapter_name=f"warm:{self.layer_name}")
        fac = fp32_adapter.compress_sub1bit(rank=fp32_adapter.rank)
        u_bin, v_bin = fac.unpack_factors()  # u: (d_out, r), v: (d_in, r)

        # Latent weights start at +-0.5 inside [-1, 1] so STE gradients remain responsive.
        self.lora_a = mx.array((0.5 * v_bin).astype(np.float32))  # (d_in, rank)
        self.lora_b = mx.array((0.5 * u_bin.T).astype(np.float32))  # (rank, d_out)
        self.s_in = mx.array(fac.s2.astype(np.float32))
        self.s_out = mx.array(fac.s1.astype(np.float32))
        self.sigma = mx.array(fac.sigma.astype(np.float32))
        self._lora_scale = 1.0
        self._lora_binary_ste = True
        mx.eval(self.lora_a, self.lora_b, self.s_in, self.s_out, self.sigma)

    def trainable_lora_keys(self) -> list[str]:
        if self._lora_binary_ste:
            return ["lora_a", "lora_b", "s_in", "s_out", "sigma"]
        return ["lora_a", "lora_b"]

    def extract_trainable_lora(self, adapter_name: str) -> LowRankAdapter:
        """Convert the active trainable MLX branch into a SUBCOR LowRankAdapter."""
        if not self._lora_active or self.lora_a is None or self.lora_b is None:
            raise RuntimeError(f"No trainable LoRA attached on layer '{self.layer_name}'")
        if self._lora_binary_ste:
            assert self.s_in is not None and self.s_out is not None and self.sigma is not None
            mx.eval(self.lora_a, self.lora_b, self.s_in, self.s_out, self.sigma)
            return LowRankAdapter(
                name=adapter_name,
                a=np.array(self.lora_a, dtype=np.float64).T,
                b=np.array(self.lora_b, dtype=np.float64).T,
                scale=self._lora_scale,
                binary_ste=True,
                s_in=np.array(self.s_in, dtype=np.float64),
                s_out=np.array(self.s_out, dtype=np.float64),
                sigma=np.array(self.sigma, dtype=np.float64),
                frozen=True,
            )
        mx.eval(self.lora_a, self.lora_b)
        a_np = np.array(self.lora_a, dtype=np.float64).T  # (rank, d_in)
        b_np = np.array(self.lora_b, dtype=np.float64).T  # (d_out, rank)
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

    def __call__(self, x: mx.array) -> mx.array:
        y = self.base(x)

        if not self._bypass_adapters:
            if self._mounted:
                for a_t, b_t in self._mounted.values():
                    y = y + (x @ a_t) @ b_t

            if self._lora_active and self.lora_a is not None and self.lora_b is not None:
                dtype = x.dtype
                x_f32 = x.astype(mx.float32)
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
                y = y + z.astype(dtype)

        if self._recording:
            # Record mean hidden output across batch & sequence positions off-graph.
            y_flat = y.reshape(-1, y.shape[-1]).astype(mx.float32)
            mx.eval(y_flat)
            arr = np.array(y_flat, dtype=np.float64)
            vec_sum = np.sum(arr, axis=0)
            if self._act_sum is None:
                self._act_sum = vec_sum
            else:
                self._act_sum += vec_sum
            self._act_count += arr.shape[0]

        return y


def _resolve_parent_and_attr(model: nn.Module, dotted_path: str) -> tuple[Any, str]:
    """Resolve e.g. 'layers.15.mlp.down_proj' on an mlx-lm model."""
    parts = dotted_path.split(".")
    obj: Any = model
    for part in parts[:-1]:
        obj = obj[int(part)] if part.isdigit() else getattr(obj, part)
    return obj, parts[-1]


def _detect_layer_prefix(model: nn.Module) -> str:
    """Detect the exact mlx-lm parameter prefix preceding `layers.<idx>...` (e.g. `language_model.model.`)."""
    for key, _ in tree_flatten(model.parameters()):
        marker = "layers.0."
        pos = key.find(marker)
        if pos != -1:
            return key[:pos]
    return "model."


class MLXKernel:
    """Frozen Apple Silicon quantized base model with governed hot-swap Skill Units."""

    def __init__(
        self,
        model_id: str = DEFAULT_COMPACT_MODEL,
        *,
        target_layer_indices: list[int] | None = None,
        projection_attr: str = "mlp.down_proj",
        target_projections: tuple[str, ...] | None = None,
        layer_preset: str | None = None,
        num_layers: int | None = None,
        layer_stride: int = 4,
    ) -> None:
        self.model_id = model_id
        self.model, self.tokenizer = load(model_id)
        self.model.freeze()
        self.layer_prefix: str = _detect_layer_prefix(self.model)

        self.total_layers: int = len(self.model.layers)
        if target_layer_indices is None:
            target_layer_indices = select_layer_indices(
                self.model,
                preset=layer_preset or "default",
                num_layers=num_layers,
                stride=layer_stride,
            )

        projections = target_projections if target_projections is not None else (projection_attr,)
        self.instrumented: dict[str, SubcorMLXLinear] = {}
        for idx in target_layer_indices:
            for proj in projections:
                dotted = f"layers.{idx}.{proj}"
                try:
                    parent, attr = _resolve_parent_and_attr(self.model, dotted)
                    base_mod = getattr(parent, attr)
                except (AttributeError, IndexError, KeyError):
                    # Hybrid architectures (e.g. Qwen3.5 GatedDeltaNet) omit self_attn on linear layers.
                    continue
                wrapped = SubcorMLXLinear(base_mod, layer_name=dotted)
                setattr(parent, attr, wrapped)
                self.instrumented[dotted] = wrapped

        if not self.instrumented:
            raise ValueError(
                f"No target projections matched on '{model_id}' for layers={target_layer_indices} "
                f"and projections={projections}"
            )

        self._greedy_sampler = make_sampler(temp=0.0)
        self._verified_unit_cache: dict[str, tuple[str, list[tuple[str, LowRankAdapter]]]] = {}
        self.last_train_stats: dict[str, Any] = {}

    def format_messages(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        enable_thinking: bool = False,
        add_generation_prompt: bool = True,
    ) -> str:
        """Render full multi-turn OpenAI messages (+ optional tool schemas) via chat template."""
        cleaned = _normalize_chat_messages(messages)
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

    def stream_formatted(
        self,
        formatted_prompt: str,
        *,
        max_tokens: int = 128,
        capture_activations: bool = False,
    ) -> Iterator[StreamTokenEvent]:
        """Yield true token-by-token StreamTokenEvents with exact per-token logprobs."""
        if capture_activations:
            for layer in self.instrumented.values():
                layer.start_recording()

        for resp in stream_generate(
            self.model,
            self.tokenizer,
            prompt=formatted_prompt,
            max_tokens=max_tokens,
            sampler=self._greedy_sampler,
        ):
            tok_id = int(resp.token)
            lp = 0.0
            if resp.logprobs is not None:
                lp = float(resp.logprobs[tok_id].item())
            yield StreamTokenEvent(
                text=str(resp.text),
                token_id=tok_id,
                logprob=lp,
                prompt_tokens=int(resp.prompt_tokens),
                generation_tokens=int(resp.generation_tokens),
                finish_reason=resp.finish_reason,
            )

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
        ids = mx.array([self.tokenizer.encode(formatted)])
        logits = self.model(ids)[:, -1, :].astype(mx.float32)
        mx.eval(logits)
        return np.array(logits[0], dtype=np.float64, copy=True)

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
                    f"instrumented in this MLXKernel (have {sorted(self.instrumented)})"
                )
            adapter = delta.to_adapter(adapter_name=f"unit:{unit_id}:{delta.layer_name}")
            pairs.append((delta.layer_name, adapter))
        self._verified_unit_cache[unit_id] = (manifest.payload_sha256, pairs)
        return pairs

    def invalidate_unit_cache(self, unit_id: str) -> None:
        self._verified_unit_cache.pop(unit_id, None)

    def mount_unit(self, unit_id: str, registry: SkillUnitRegistry) -> None:
        """Verify SHA-256 integrity and atomically hot-swap a Skill Unit onto instrumented layers."""
        loaded = self._load_cached_unit(unit_id, registry)
        mounted_layers: list[tuple[str, str]] = []
        try:
            for layer_name, adapter in loaded:
                self.instrumented[layer_name].mount_adapter(adapter.name, adapter)
                mounted_layers.append((layer_name, adapter.name))
        except Exception:
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

    def _forward_until_layer(self, ids: mx.array, tap_layer: str) -> mx.array:
        """Run frozen-base prefill up through `tap_layer`, skipping later layers and lm_head."""
        parts = tap_layer.split(".")
        tap_idx = (
            int(parts[1])
            if len(parts) >= 2 and parts[0] == "layers" and parts[1].isdigit()
            else None
        )
        inner = getattr(self.model, "model", None)

        if (
            tap_idx is not None
            and inner is not None
            and hasattr(inner, "embed_tokens")
            and hasattr(inner, "layers")
        ):
            hidden = inner.embed_tokens(ids)
            fa_mask = create_attention_mask(hidden, None)
            ssm_mask = create_ssm_mask(hidden, None) if hasattr(inner, "ssm_idx") else None
            for layer in inner.layers[: tap_idx + 1]:
                mask = ssm_mask if getattr(layer, "is_linear", False) else fa_mask
                hidden = layer(hidden, mask=mask, cache=None)
            return hidden

        if inner is not None:
            return inner(ids)
        return self.model(ids)

    def encode_prompt_embedding(
        self,
        user_prompt: str,
        *,
        tap_layer: str,
        system_prompt: str | None = None,
        enable_thinking: bool = False,
    ) -> np.ndarray:
        """Compute an invariant frozen-base hidden activation embedding at tap_layer.

        Bypasses all mounted skill adapters AND early-exits right after `tap_layer`
        (skipping downstream layers and the large vocab lm_head) so routing is fast
        and strictly invariant to installed Skill Units.
        """
        if tap_layer not in self.instrumented:
            raise KeyError(f"tap_layer '{tap_layer}' is not instrumented in this MLXKernel")
        formatted = self.format_chat_prompt(
            user_prompt,
            system_prompt=system_prompt,
            enable_thinking=enable_thinking,
        )
        ids = mx.array([self.tokenizer.encode(formatted)], dtype=mx.int32)

        for layer in self.instrumented.values():
            layer._bypass_adapters = True
        tap = self.instrumented[tap_layer]
        tap.start_recording()
        try:
            out = self._forward_until_layer(ids, tap_layer)
            mx.eval(out)
            mean_act = tap.stop_recording()
        finally:
            for layer in self.instrumented.values():
                layer._bypass_adapters = False

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
    ) -> tuple[tuple[mx.array, mx.array, mx.array] | None, str]:
        """Tokenize a single-turn tuple or multi-turn trajectory with assistant-only loss positions.

        Returns ``((x_ids, y_target, loss_positions), "ok")`` or ``(None, reason)`` where
        ``reason`` is ``"too_long"`` or ``"empty_target"``.
        """
        eos_id = self.tokenizer.eos_token_id
        eos_str = getattr(self.tokenizer, "eos_token", None)

        if isinstance(item, dict):
            raw_msgs = item.get("messages")
            if not isinstance(raw_msgs, list) or not raw_msgs:
                return None, "empty_target"
            messages = [dict(m) for m in raw_msgs]
            if default_system_prompt and messages[0].get("role") != "system":
                messages.insert(0, {"role": "system", "content": default_system_prompt})
            # Trim any trailing non-assistant turns (e.g., trailing tool output or
            # unanswered user turn) before formatting & max_seq_len checks so
            # unsupervised tail tokens never waste compute or falsely trigger too_long.
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
            full_ids = self.tokenizer.encode(full_text)
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
                # Keep supervision through the assistant turn's EOS token, excluding the
                # template's inter-turn separator newline right after EOS.
                if eos_str and upto_text.endswith(f"{eos_str}\n"):
                    upto_text = upto_text[:-1]
                start_tok = len(self.tokenizer.encode(pref_text))
                end_tok = min(len(self.tokenizer.encode(upto_text)), len(full_ids))
                for tok_pos in range(start_tok, end_tok):
                    pred_pos = tok_pos - 1
                    if 0 <= pred_pos < len(full_ids) - 1:
                        pos_set.add(pred_pos)

            if not pos_set:
                return None, "empty_target"

            pos_list = sorted(pos_set)
            x_ids = mx.array([full_ids[:-1]], dtype=mx.int32)
            loss_positions = mx.array(pos_list, dtype=mx.int32)
            y_target = mx.array([[full_ids[p + 1] for p in pos_list]], dtype=mx.int32)
            return (x_ids, y_target, loss_positions), "ok"

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
        prompt_ids = self.tokenizer.encode(prefix)
        target_ids = self.tokenizer.encode(target, add_special_tokens=False)
        if eos_id is not None and (not target_ids or target_ids[-1] != eos_id):
            target_ids = [*target_ids, int(eos_id)]
        if not target_ids:
            return None, "empty_target"

        full_ids = prompt_ids + target_ids
        if len(full_ids) < 2:
            return None, "empty_target"
        if len(full_ids) > max_seq_len:
            return None, "too_long"

        x_ids = mx.array([full_ids[:-1]], dtype=mx.int32)
        target_start = len(prompt_ids) - 1
        pos_list = list(range(target_start, len(full_ids) - 1))
        loss_positions = mx.array(pos_list, dtype=mx.int32)
        y_target = mx.array([full_ids[target_start + 1 :]], dtype=mx.int32)
        return (x_ids, y_target, loss_positions), "ok"

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
        grad_checkpoint: bool | None = None,
        max_prefix_cache_mb: float = 512.0,
        seed: int = 42,
    ) -> tuple[dict[str, LowRankAdapter], list[float], float]:
        """Function-preserving LoRA expansion + Apple Silicon GPU QLoRA adaptation.

        Freezes the entire base model, attaches zero-initialized LoRA branches to
        ``layer_names``, verifies exact zero logit perturbation before step 0, trains
        ONLY the new LoRA weights via assistant-only masked causal LM cross-entropy on
        either single-turn tuples or canonical multi-turn trajectory dicts
        (``{"messages": [...], "tools": [...]}``), then extracts and detaches the
        trained adapters.

        Uses frozen-prefix caching, target-position-sliced ``lm_head`` projection,
        layer-level ``mx.checkpoint`` for multi-layer adaptation, minimal per-layer
        ``train()`` scoping, and exact chunkwise-parallel GatedDeltaNet ops so hybrid
        4B models train multi-layer adapters cleanly within 16 GB unified memory.

        Returns ``(adapters_by_layer, loss_history, zero_perturbation_max_diff)``.
        """
        if not pairs:
            raise ValueError("pairs must be non-empty")
        for name in layer_names:
            if name not in self.instrumented:
                raise KeyError(f"Layer '{name}' is not instrumented")

        # Extract a representative probe prompt for zero-perturbation verification.
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

        # Measure live logits BEFORE expansion (in eval mode).
        self.model.eval()
        logits_before = self.probe_logits(
            probe_prompt,
            system_prompt=probe_sys,
            enable_thinking=enable_thinking,
        )

        # Freeze entire model, then attach zero-initialized FP32 LoRA branches.
        self.model.freeze()
        for idx, name in enumerate(layer_names):
            layer = self.instrumented[name]
            layer.attach_trainable_lora(rank=rank, scale=scale, binary_ste=False, seed=seed + idx)
            layer.unfreeze(keys=layer.trainable_lora_keys(), recurse=False)

        # Verify strict zero-perturbation invariant on live transformer logits.
        logits_after = self.probe_logits(
            probe_prompt,
            system_prompt=probe_sys,
            enable_thinking=enable_thinking,
        )
        zero_perturb_diff = float(np.max(np.abs(logits_after - logits_before)))

        # Pre-tokenize all examples with assistant-only loss position indices and explicit drop stats.
        raw_examples: list[tuple[mx.array, mx.array, mx.array]] = []
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
                raw_examples.append(tok_ex)
            elif reason == "too_long":
                dropped_too_long += 1
            else:
                dropped_empty_target += 1

        if dropped_too_long > 0 or dropped_empty_target > 0:
            logger.warning(
                "MLX QLoRA tokenization kept %d/%d examples (dropped_too_long=%d > max_seq_len=%d, "
                "dropped_empty_target=%d).",
                len(raw_examples),
                len(pairs),
                dropped_too_long,
                max_seq_len,
                dropped_empty_target,
            )
        if not raw_examples:
            for name in layer_names:
                self.instrumented[name].detach_trainable_lora()
            self.model.freeze()
            self.model.eval()
            raise ValueError(
                f"No valid tokenized examples produced from {len(pairs)} inputs "
                f"(dropped_too_long={dropped_too_long} > max_seq_len={max_seq_len}, "
                f"dropped_empty_target={dropped_empty_target})"
            )

        # Determine if structured layer-by-layer forward + target-sliced lm_head is supported
        # (works even when first_trainable_idx == 0!) and whether a frozen prefix exists.
        layer_indices = [
            int(n.split(".")[1])
            for n in layer_names
            if len(n.split(".")) >= 2 and n.split(".")[0] == "layers" and n.split(".")[1].isdigit()
        ]
        first_trainable_idx = min(layer_indices) if layer_indices else 0
        inner_model = getattr(self.model, "model", None)
        lm_owner = getattr(self.model, "language_model", self.model)
        can_use_structured_forward = (
            inner_model is not None
            and hasattr(inner_model, "embed_tokens")
            and hasattr(inner_model, "layers")
            and hasattr(inner_model, "norm")
            and (
                hasattr(lm_owner, "lm_head")
                or getattr(getattr(lm_owner, "args", None), "tie_word_embeddings", False)
            )
        )
        has_frozen_prefix = can_use_structured_forward and first_trainable_idx > 0

        tail_layers = (
            list(inner_model.layers[first_trainable_idx:])
            if (can_use_structured_forward and inner_model is not None)
            else []
        )
        use_grad_checkpoint = (
            (len(tail_layers) > 1) if grad_checkpoint is None else bool(grad_checkpoint)
        )
        has_linear_in_tail = (
            any(getattr(lyr, "is_linear", False) for lyr in tail_layers)
            if can_use_structured_forward
            else (
                any(getattr(lyr, "is_linear", False) for lyr in inner_model.layers)
                if inner_model is not None and hasattr(inner_model, "layers")
                else False
            )
        )

        accum = max(1, int(grad_accum))
        total_microsteps = max(1, steps * accum)

        # Reset peak memory BEFORE any prefix precomputation so peak_memory_gb reflects true
        # end-to-end training memory (including any cached prefix tensors).
        mx.reset_peak_memory()

        def _compute_frozen_prefix(x_ids: mx.array) -> mx.array:
            assert inner_model is not None
            h_pref = inner_model.embed_tokens(x_ids)
            fa_mask = create_attention_mask(h_pref, None)
            ssm_mask = create_ssm_mask(h_pref, None) if hasattr(inner_model, "ssm_idx") else None
            for layer in inner_model.layers[:first_trainable_idx]:
                mask = ssm_mask if getattr(layer, "is_linear", False) else fa_mask
                h_pref = layer(h_pref, mask=mask, cache=None)
            h_pref = mx.stop_gradient(h_pref)
            mx.eval(h_pref)
            return h_pref

        # Select bounded frozen-prefix caching policy ("full" | "lru" | "on_the_fly" | "none"):
        # never materialize unbounded prefix tensors in RAM on large datasets.
        prefix_cache_mode = "none"
        budget_bytes = max(0, int(float(max_prefix_cache_mb) * 1024.0 * 1024.0))
        cached_examples: list[tuple[mx.array, mx.array, mx.array]] = []
        lru_cache: OrderedDict[int, tuple[mx.array, int]] = OrderedDict()
        lru_cur_bytes = 0
        max_cache_bytes = 0

        if has_frozen_prefix:
            assert inner_model is not None
            self.model.eval()
            if budget_bytes <= 0:
                prefix_cache_mode = "on_the_fly"
            else:
                probe_x = raw_examples[0][0]
                probe_h = _compute_frozen_prefix(probe_x)
                bytes_per_tok = probe_h.nbytes / max(1, int(probe_x.shape[1]))
                estimated_full_bytes = int(
                    sum(int(ex[0].shape[1]) for ex in raw_examples) * bytes_per_tok
                )
                if estimated_full_bytes <= budget_bytes and len(raw_examples) <= total_microsteps:
                    prefix_cache_mode = "full"
                    cached_examples.append((probe_h, raw_examples[0][1], raw_examples[0][2]))
                    max_cache_bytes = probe_h.nbytes
                    for x_ids, y_target, loss_positions in raw_examples[1:]:
                        h_pref = _compute_frozen_prefix(x_ids)
                        max_cache_bytes += h_pref.nbytes
                        cached_examples.append((h_pref, y_target, loss_positions))
                    mx.clear_cache()
                else:
                    prefix_cache_mode = "lru"
                    # Seed LRU with index 0 if it fits in the budget; otherwise discard.
                    if probe_h.nbytes <= budget_bytes:
                        lru_cache[0] = (probe_h, probe_h.nbytes)
                        lru_cur_bytes = probe_h.nbytes
                        max_cache_bytes = lru_cur_bytes
                    del probe_h
                    mx.clear_cache()

        if can_use_structured_forward:

            def loss_fn(
                model: nn.Module,
                inp: mx.array,
                y_target: mx.array,
                loss_positions: mx.array,
            ) -> mx.array:
                inner = model.model
                lm = getattr(model, "language_model", model)
                h = inp if has_frozen_prefix else inner.embed_tokens(inp)
                fa_mask = create_attention_mask(h, None)
                ssm_mask = create_ssm_mask(h, None) if hasattr(inner, "ssm_idx") else None
                for layer in inner.layers[first_trainable_idx:]:
                    mask = ssm_mask if getattr(layer, "is_linear", False) else fa_mask
                    if use_grad_checkpoint:

                        def _ckpt_layer(
                            params: Any,
                            x_in: mx.array,
                            m_in: Any,
                            _layer: nn.Module = layer,
                        ) -> mx.array:
                            _layer.update(params)
                            return _layer(x_in, mask=m_in, cache=None)

                        h = mx.checkpoint(_ckpt_layer)(layer.trainable_parameters(), h, mask)
                    else:
                        h = layer(h, mask=mask, cache=None)
                # Gather ONLY masked assistant/tool-call positions before the 248k-vocab lm_head!
                h_sel = inner.norm(h)[:, loss_positions, :]
                if getattr(getattr(lm, "args", None), "tie_word_embeddings", False):
                    logits = inner.embed_tokens.as_linear(h_sel).astype(mx.float32)
                else:
                    logits = lm.lm_head(h_sel).astype(mx.float32)
                return mx.mean(nn.losses.cross_entropy(logits, y_target, reduction="none"))
        else:

            def loss_fn(
                model: nn.Module,
                x_ids: mx.array,
                y_target: mx.array,
                loss_positions: mx.array,
            ) -> mx.array:
                logits = model(x_ids)[:, loss_positions, :].astype(mx.float32)
                return mx.mean(nn.losses.cross_entropy(logits, y_target, reduction="none"))

        loss_and_grad = nn.value_and_grad(self.model, loss_fn)

        fp32_steps = int(round(steps * 0.70)) if binary_ste and steps >= 6 else steps
        ste_steps = steps - fp32_steps
        fp32_lr = min(lr, 4e-4)
        ste_lr = min(lr * 0.25, 8e-5)

        losses: list[float] = []
        rng = np.random.default_rng(seed)
        num_examples = len(raw_examples)
        order = np.arange(num_examples)

        # Minimally scope train() ONLY to linear-attention (GatedDeltaNet) layers inside the
        # differentiable tail; all prefix & full-attention layers keep fast Metal eval() kernels.
        self.model.eval()
        if has_linear_in_tail and inner_model is not None and hasattr(inner_model, "layers"):
            scoped_layers = (
                inner_model.layers[first_trainable_idx:]
                if can_use_structured_forward
                else inner_model.layers
            )
            for lyr in scoped_layers:
                if getattr(lyr, "is_linear", False):
                    lyr.train()

        def _add_grads(acc: Any, g: Any) -> Any:
            if acc is None:
                return g
            if isinstance(acc, dict):
                return {k: _add_grads(acc[k], g[k]) for k in acc}
            if isinstance(acc, list):
                return [_add_grads(a, b) for a, b in zip(acc, g, strict=True)]
            return acc + g

        def _scale_grads(g: Any, inv: float) -> Any:
            if isinstance(g, dict):
                return {k: _scale_grads(v, inv) for k, v in g.items()}
            if isinstance(g, list):
                return [_scale_grads(v, inv) for v in g]
            return g * inv

        ex_cursor = 0

        def _next_example() -> tuple[mx.array, mx.array, mx.array]:
            nonlocal ex_cursor, lru_cur_bytes, max_cache_bytes
            if ex_cursor % num_examples == 0:
                rng.shuffle(order)
            idx = int(order[ex_cursor % num_examples])
            ex_cursor += 1

            if prefix_cache_mode == "full":
                return cached_examples[idx]
            x_ids, y_target, loss_positions = raw_examples[idx]
            if prefix_cache_mode == "on_the_fly":
                return _compute_frozen_prefix(x_ids), y_target, loss_positions
            if prefix_cache_mode == "lru":
                hit = lru_cache.get(idx)
                if hit is not None:
                    lru_cache.move_to_end(idx)
                    return hit[0], y_target, loss_positions
                h_pref = _compute_frozen_prefix(x_ids)
                nbytes = h_pref.nbytes
                if nbytes <= budget_bytes:
                    while lru_cache and lru_cur_bytes + nbytes > budget_bytes:
                        _, (_, evicted_bytes) = lru_cache.popitem(last=False)
                        lru_cur_bytes -= evicted_bytes
                    lru_cache[idx] = (h_pref, nbytes)
                    lru_cur_bytes += nbytes
                    max_cache_bytes = max(max_cache_bytes, lru_cur_bytes)
                return h_pref, y_target, loss_positions
            return x_ids, y_target, loss_positions

        t_loop0 = time.perf_counter()
        try:
            with _chunked_gated_delta_context(enabled=has_linear_in_tail) as chunked_gdn_active:
                # Stage 1: FP32 continuous warm-start.
                opt_fp32 = optim.AdamW(
                    learning_rate=optim.cosine_decay(
                        fp32_lr, max(1, fp32_steps), end=fp32_lr * 0.15
                    ),
                    weight_decay=0.01,
                )
                for step in range(fp32_steps):
                    acc_grads = None
                    step_loss = 0.0
                    for _ in range(accum):
                        inp, y_target, loss_positions = _next_example()
                        loss_val, grads = loss_and_grad(self.model, inp, y_target, loss_positions)
                        acc_grads = _add_grads(acc_grads, grads)
                        mx.eval(loss_val, acc_grads)
                        step_loss += float(loss_val.item())
                    avg_grads = _scale_grads(acc_grads, 1.0 / accum) if accum > 1 else acc_grads
                    avg_grads, _ = optim.clip_grad_norm(avg_grads, max_norm=1.0)
                    opt_fp32.update(self.model, avg_grads)
                    mx.eval(self.model.parameters(), opt_fp32.state)
                    losses.append(step_loss / accum)
                    if (step + 1) % 8 == 0:
                        mx.clear_cache()

                # Stage 2: Binary STE (QAT) refinement directly on {-1, +1} factors + scales.
                if ste_steps > 0:
                    self.model.freeze()
                    for name in layer_names:
                        lyr = self.instrumented[name]
                        lyr.convert_lora_to_binary_ste()
                        lyr.unfreeze(keys=lyr.trainable_lora_keys(), recurse=False)

                    opt_ste = optim.AdamW(
                        learning_rate=optim.cosine_decay(
                            ste_lr, max(1, ste_steps), end=ste_lr * 0.1
                        ),
                        weight_decay=0.0,
                    )
                    for step in range(ste_steps):
                        acc_grads = None
                        step_loss = 0.0
                        for _ in range(accum):
                            inp, y_target, loss_positions = _next_example()
                            loss_val, grads = loss_and_grad(
                                self.model, inp, y_target, loss_positions
                            )
                            acc_grads = _add_grads(acc_grads, grads)
                            mx.eval(loss_val, acc_grads)
                            step_loss += float(loss_val.item())
                        avg_grads = _scale_grads(acc_grads, 1.0 / accum) if accum > 1 else acc_grads
                        avg_grads, _ = optim.clip_grad_norm(avg_grads, max_norm=1.0)
                        opt_ste.update(self.model, avg_grads)
                        for name in layer_names:
                            lyr = self.instrumented[name]
                            if lyr.lora_a is not None and lyr.lora_b is not None:
                                lyr.lora_a = mx.clip(lyr.lora_a, -1.0, 1.0)
                                lyr.lora_b = mx.clip(lyr.lora_b, -1.0, 1.0)
                        mx.eval(self.model.parameters(), opt_ste.state)
                        losses.append(step_loss / accum)
                        if (step + 1) % 8 == 0:
                            mx.clear_cache()

            elapsed = time.perf_counter() - t_loop0
            peak_gb = mx.get_peak_memory() / (1024**3)
            self.last_train_stats = {
                "total_input": len(pairs),
                "kept": len(raw_examples),
                "dropped_too_long": dropped_too_long,
                "dropped_empty_target": dropped_empty_target,
                "max_seq_len": max_seq_len,
                "grad_accum": accum,
                "grad_checkpoint": use_grad_checkpoint,
                "chunked_gated_delta": chunked_gdn_active,
                "prefix_cache_mode": prefix_cache_mode,
                "prefix_cache_mb": round(max_cache_bytes / (1024.0 * 1024.0), 4),
                "peak_memory_gb": round(peak_gb, 3),
                "train_seconds": round(elapsed, 3),
                "mean_step_seconds": round(elapsed / max(1, len(losses)), 4),
            }

            extracted: dict[str, LowRankAdapter] = {}
            for name in layer_names:
                extracted[name] = self.instrumented[name].extract_trainable_lora(
                    adapter_name=f"trained:{name}"
                )
        finally:
            cached_examples.clear()
            lru_cache.clear()
            for name in layer_names:
                self.instrumented[name].detach_trainable_lora()
            self.model.freeze()
            self.model.eval()
            mx.clear_cache()

        return extracted, losses, zero_perturb_diff

    def trainable_parameter_count(self) -> int:
        return sum(v.size for _, v in tree_flatten(self.model.trainable_parameters()))


def export_mlx_lora_safetensors(
    unit_id: str,
    registry: SkillUnitRegistry,
    destination_dir: Path,
    *,
    model: nn.Module | None = None,
    layer_prefix: str | None = None,
    total_layers: int | None = None,
    exact_layer_keys: bool = False,
) -> Path:
    """Export a verified Skill Unit as stock mlx-lm adapters.safetensors + adapter_config.json.

    Detects the exact model key prefix (e.g. `language_model.model.` on Qwen3.5 or
    `model.` on standard Llama/Gemma) and sets `num_layers` to match mlx-lm's
    last-N-layers convention so `mlx_lm.load(model_id, adapter_path=...)` works natively
    for both contiguous and non-contiguous multi-layer units. Pass `exact_layer_keys=True`
    to emit full dotted module paths with `num_layers=0` (avoiding zero-initialized LoRA
    wrappers on untouched intermediate blocks for sparse periodic layouts).
    """
    manifest = registry.get_manifest(unit_id)
    deltas: list[LayerDeltaPayload] = registry.load_verified_deltas(unit_id)

    resolved_prefix = (
        layer_prefix
        if layer_prefix is not None
        else (_detect_layer_prefix(model) if model is not None else "model.")
    )
    resolved_total_layers = (
        total_layers
        if total_layers is not None
        else (len(model.layers) if model is not None and hasattr(model, "layers") else None)
    )

    destination_dir.mkdir(parents=True, exist_ok=True)
    tensors: dict[str, mx.array] = {}
    ranks: set[int] = set()
    keys: set[str] = set()
    full_keys: set[str] = set()
    layer_indices: list[int] = []

    for delta in deltas:
        adapter = delta.to_adapter(adapter_name=delta.layer_name)
        ranks.add(adapter.rank)
        # mlx-lm adapter naming: <layer_prefix><layer_path>.lora_a (d_in, r), .lora_b (r, d_out)
        prefix = f"{resolved_prefix}{delta.layer_name}"
        tensors[f"{prefix}.lora_a"] = mx.array(adapter.a.T.astype(np.float16))
        tensors[f"{prefix}.lora_b"] = mx.array((adapter.scale * adapter.b).T.astype(np.float16))
        full_keys.add(prefix)
        parts = delta.layer_name.split(".")
        if len(parts) >= 2 and parts[0] == "layers" and parts[1].isdigit():
            layer_indices.append(int(parts[1]))
            if len(parts) >= 3:
                keys.add(".".join(parts[2:]))
        elif len(parts) >= 3:
            keys.add(".".join(parts[2:]))

    safetensors_path = destination_dir / "adapters.safetensors"
    mx.save_safetensors(str(safetensors_path), tensors)

    unique_indices = sorted(set(layer_indices))
    if exact_layer_keys and full_keys:
        num_layers = 0
        exported_keys = sorted(full_keys)
    elif resolved_total_layers is not None and unique_indices:
        num_layers = resolved_total_layers - unique_indices[0]
        exported_keys = sorted(keys)
    else:
        num_layers = -1
        exported_keys = sorted(keys)

    config = {
        "fine_tune_type": "lora",
        "num_layers": num_layers,
        "lora_parameters": {
            "rank": max(ranks) if ranks else 1,
            "scale": 1.0,
            "dropout": 0.0,
            "keys": exported_keys,
        },
        "subcor_unit": {
            "unit_id": manifest.unit_id,
            "version": manifest.version,
            "base_kernel_id": manifest.base_kernel_id,
            "payload_sha256": manifest.payload_sha256,
            "effective_bits_per_weight": round(manifest.effective_bits_per_weight, 6),
            "target_layers": sorted(full_keys),
        },
    }
    (destination_dir / "adapter_config.json").write_text(
        json.dumps(config, indent=2) + "\n",
        encoding="utf-8",
    )
    return destination_dir
