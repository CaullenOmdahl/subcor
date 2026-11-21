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

import json
import math
from collections.abc import Iterator
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
    ) -> None:
        self.model_id = model_id
        self.model, self.tokenizer = load(model_id)
        self.model.freeze()
        self.layer_prefix: str = _detect_layer_prefix(self.model)

        num_layers = len(self.model.layers)
        self.total_layers: int = num_layers
        if target_layer_indices is None:
            # Default: compact set covering middle + upper layers (including MIT Exp 8's 15 & 19).
            candidates = [7, 11, 15, 19, num_layers - 3, num_layers - 2, num_layers - 1]
            target_layer_indices = sorted({i for i in candidates if 0 <= i < num_layers})

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

    def format_messages(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        enable_thinking: bool = False,
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
                entry["tool_calls"] = msg["tool_calls"]
            if "tool_call_id" in msg and msg["tool_call_id"]:
                entry["tool_call_id"] = str(msg["tool_call_id"])
            if "name" in msg and msg["name"]:
                entry["name"] = str(msg["name"])
            cleaned.append(entry)

        kwargs: dict[str, Any] = {
            "tokenize": False,
            "add_generation_prompt": True,
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

    def train_qlora_adapters(
        self,
        pairs: (
            list[tuple[str, str]]
            | list[tuple[str | None, str, str]]
            | list[tuple[str | None, str, str, list[dict[str, Any]] | None]]
        ),
        *,
        layer_names: list[str],
        system_prompt: str | None = None,
        enable_thinking: bool = False,
        rank: int = 8,
        scale: float = 1.0,
        binary_ste: bool = False,
        steps: int = 108,
        lr: float = 3e-4,
        max_seq_len: int = 1536,
        grad_accum: int = 1,
        seed: int = 42,
    ) -> tuple[dict[str, LowRankAdapter], list[float], float]:
        """Function-preserving LoRA expansion + Apple Silicon GPU QLoRA adaptation.

        Freezes the entire base model, attaches zero-initialized LoRA branches to
        layer_names, verifies exact zero logit perturbation before step 0, trains
        ONLY the new LoRA weights via masked causal LM cross-entropy on verified
        (prompt, target), (system_prompt, prompt, target), or
        (system_prompt, prompt, target, tools) examples, then extracts and detaches
        the trained adapters.

        Returns (adapters_by_layer, loss_history, zero_perturbation_max_diff).
        """
        if not pairs:
            raise ValueError("pairs must be non-empty")
        for name in layer_names:
            if name not in self.instrumented:
                raise KeyError(f"Layer '{name}' is not instrumented")

        normalized: list[tuple[str | None, str, str, list[dict[str, Any]] | None]] = []
        for item in pairs:
            if len(item) == 2:
                normalized.append((system_prompt, item[0], item[1], None))
            elif len(item) == 3:
                normalized.append((item[0], item[1], item[2], None))
            else:
                normalized.append((item[0], item[1], item[2], item[3]))

        # Measure live logits BEFORE expansion (in eval mode).
        self.model.eval()
        probe_sys, probe_prompt, _, _ = normalized[0]
        logits_before = self.probe_logits(
            probe_prompt,
            system_prompt=probe_sys,
            enable_thinking=enable_thinking,
        )

        # Freeze entire model, then attach zero-initialized FP32 LoRA branches.
        # When binary_ste=True, we run Two-Stage QAT:
        #   Stage 1 (70% steps): FP32 warm-start for seed-stable basin discovery
        #   Stage 2 (30% steps): Convert to Sub1Bit Binary STE and refine around {-1, +1}
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

        # Pre-tokenize all examples with completion-only loss spans.
        eos_id = self.tokenizer.eos_token_id
        raw_examples: list[tuple[mx.array, mx.array, int]] = []
        for ex_sys, user_prompt, target, ex_tools in normalized:
            msgs: list[dict[str, Any]] = []
            if ex_sys:
                msgs.append({"role": "system", "content": ex_sys})
            msgs.append({"role": "user", "content": user_prompt})
            prefix = self.format_messages(
                msgs,
                tools=ex_tools,
                enable_thinking=enable_thinking,
            )
            prompt_ids = self.tokenizer.encode(prefix)
            target_ids = self.tokenizer.encode(target, add_special_tokens=False)
            if eos_id is not None and (not target_ids or target_ids[-1] != eos_id):
                target_ids = [*target_ids, int(eos_id)]
            full_ids = prompt_ids + target_ids
            if len(full_ids) < 2 or len(full_ids) > max_seq_len:
                continue

            x_ids = mx.array([full_ids[:-1]], dtype=mx.int32)
            target_start = len(prompt_ids) - 1
            y_target = mx.array([full_ids[target_start + 1 :]], dtype=mx.int32)
            raw_examples.append((x_ids, y_target, target_start))

        if not raw_examples:
            raise ValueError("No valid tokenized examples produced from pairs")

        # Determine if exact frozen-prefix caching + target-sliced lm_head is supported:
        # cuts peak memory ~10-15x and speeds up upper-layer LoRA training ~5x with
        # mathematically identical gradients (all layers < first_trainable_idx are frozen).
        layer_indices = [
            int(n.split(".")[1])
            for n in layer_names
            if len(n.split(".")) >= 2 and n.split(".")[0] == "layers" and n.split(".")[1].isdigit()
        ]
        first_trainable_idx = min(layer_indices) if layer_indices else 0
        inner_model = getattr(self.model, "model", None)
        lm_owner = getattr(self.model, "language_model", self.model)
        can_use_prefix_cache = (
            first_trainable_idx > 0
            and inner_model is not None
            and hasattr(inner_model, "embed_tokens")
            and hasattr(inner_model, "layers")
            and hasattr(inner_model, "norm")
            and (
                hasattr(lm_owner, "lm_head")
                or getattr(getattr(lm_owner, "args", None), "tie_word_embeddings", False)
            )
        )

        cached_examples: list[tuple[mx.array, mx.array, int]] = []
        if can_use_prefix_cache:
            assert inner_model is not None
            self.model.eval()
            for x_ids, y_target, target_start in raw_examples:
                h = inner_model.embed_tokens(x_ids)
                fa_mask = create_attention_mask(h, None)
                ssm_mask = create_ssm_mask(h, None) if hasattr(inner_model, "ssm_idx") else None
                for layer in inner_model.layers[:first_trainable_idx]:
                    mask = ssm_mask if getattr(layer, "is_linear", False) else fa_mask
                    h = layer(h, mask=mask, cache=None)
                mx.eval(h)
                cached_examples.append((h, y_target, target_start))
            mx.clear_cache()

            def loss_fn(
                model: nn.Module,
                h_prefix: mx.array,
                y_target: mx.array,
                target_start: int,
            ) -> mx.array:
                inner = model.model
                lm = getattr(model, "language_model", model)
                h = h_prefix
                fa_mask = create_attention_mask(h, None)
                ssm_mask = create_ssm_mask(h, None) if hasattr(inner, "ssm_idx") else None
                for layer in inner.layers[first_trainable_idx:]:
                    mask = ssm_mask if getattr(layer, "is_linear", False) else fa_mask
                    h = layer(h, mask=mask, cache=None)
                # Slice ONLY the completion token positions before the 248k-vocab lm_head!
                h_tail = inner.norm(h)[:, target_start:, :]
                if getattr(getattr(lm, "args", None), "tie_word_embeddings", False):
                    logits = inner.embed_tokens.as_linear(h_tail).astype(mx.float32)
                else:
                    logits = lm.lm_head(h_tail).astype(mx.float32)
                return mx.mean(nn.losses.cross_entropy(logits, y_target, reduction="none"))
        else:
            cached_examples = raw_examples

            def loss_fn(
                model: nn.Module,
                x_ids: mx.array,
                y_target: mx.array,
                target_start: int,
            ) -> mx.array:
                logits = model(x_ids)[:, target_start:, :].astype(mx.float32)
                return mx.mean(nn.losses.cross_entropy(logits, y_target, reduction="none"))

        loss_and_grad = nn.value_and_grad(self.model, loss_fn)

        fp32_steps = int(round(steps * 0.70)) if binary_ste and steps >= 6 else steps
        ste_steps = steps - fp32_steps
        fp32_lr = min(lr, 4e-4)
        ste_lr = min(lr * 0.25, 8e-5)

        losses: list[float] = []
        rng = np.random.default_rng(seed)
        order = np.arange(len(cached_examples))

        # Only switch to train() if a linear-attention (GatedDeltaNet) layer lies inside
        # the differentiable tail; when adapting only full-attention blocks (e.g. final
        # layer), eval() keeps fast Metal kernels and runs 20-50x faster!
        needs_train_mode = True
        if can_use_prefix_cache and inner_model is not None:
            tail_layers = inner_model.layers[first_trainable_idx:]
            needs_train_mode = any(getattr(lyr, "is_linear", False) for lyr in tail_layers)
        if needs_train_mode:
            self.model.train()
        else:
            self.model.eval()
        accum = max(1, int(grad_accum))

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

        def _next_example() -> tuple[mx.array, mx.array, int]:
            nonlocal ex_cursor
            if ex_cursor % len(cached_examples) == 0:
                rng.shuffle(order)
            idx = int(order[ex_cursor % len(cached_examples)])
            ex_cursor += 1
            return cached_examples[idx]

        try:
            # Stage 1: FP32 continuous warm-start.
            opt_fp32 = optim.AdamW(
                learning_rate=optim.cosine_decay(fp32_lr, max(1, fp32_steps), end=fp32_lr * 0.15),
                weight_decay=0.01,
            )
            for step in range(fp32_steps):
                acc_grads = None
                step_loss = 0.0
                for _ in range(accum):
                    inp, y_target, target_start = _next_example()
                    loss_val, grads = loss_and_grad(self.model, inp, y_target, target_start)
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
                    learning_rate=optim.cosine_decay(ste_lr, max(1, ste_steps), end=ste_lr * 0.1),
                    weight_decay=0.0,
                )
                for step in range(ste_steps):
                    acc_grads = None
                    step_loss = 0.0
                    for _ in range(accum):
                        inp, y_target, target_start = _next_example()
                        loss_val, grads = loss_and_grad(self.model, inp, y_target, target_start)
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

            extracted: dict[str, LowRankAdapter] = {}
            for name in layer_names:
                extracted[name] = self.instrumented[name].extract_trainable_lora(
                    adapter_name=f"trained:{name}"
                )
        finally:
            for name in layer_names:
                self.instrumented[name].detach_trainable_lora()
            self.model.freeze()
            self.model.eval()

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
) -> Path:
    """Export a verified Skill Unit as stock mlx-lm adapters.safetensors + adapter_config.json.

    Detects the exact model key prefix (e.g. `language_model.model.` on Qwen3.5 or
    `model.` on standard Llama/Gemma) and sets `num_layers` to match mlx-lm's
    last-N-layers convention so `mlx_lm.load(model_id, adapter_path=...)` works natively.
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
    layer_indices: list[int] = []

    for delta in deltas:
        adapter = delta.to_adapter(adapter_name=delta.layer_name)
        ranks.add(adapter.rank)
        # mlx-lm adapter naming: <layer_prefix><layer_path>.lora_a (d_in, r), .lora_b (r, d_out)
        prefix = f"{resolved_prefix}{delta.layer_name}"
        tensors[f"{prefix}.lora_a"] = mx.array(adapter.a.T.astype(np.float16))
        tensors[f"{prefix}.lora_b"] = mx.array((adapter.scale * adapter.b).T.astype(np.float16))
        parts = delta.layer_name.split(".")
        if len(parts) >= 2 and parts[0] == "layers" and parts[1].isdigit():
            layer_indices.append(int(parts[1]))
        if len(parts) >= 3:
            keys.add(".".join(parts[2:]))

    safetensors_path = destination_dir / "adapters.safetensors"
    mx.save_safetensors(str(safetensors_path), tensors)

    # mlx-lm interprets num_layers as "the last N layers" (or -1 for all layers).
    if resolved_total_layers is not None and layer_indices:
        num_layers = resolved_total_layers - min(layer_indices)
    else:
        num_layers = -1

    config = {
        "fine_tune_type": "lora",
        "num_layers": num_layers,
        "lora_parameters": {
            "rank": max(ranks) if ranks else 1,
            "scale": 1.0,
            "dropout": 0.0,
            "keys": sorted(keys),
        },
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
