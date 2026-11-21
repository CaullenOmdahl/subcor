"""Apple Silicon MLX frozen-kernel runtime with hot-swappable SLAINGI Skill Units.

Wraps any compact mlx-lm quantized checkpoint (e.g., Qwen3.5-0.8B-4bit or
Gemma-4-E2B-it-qat-4bit) as an immutable base kernel:

  - Base QuantizedLinear weights are strictly frozen and never overwritten.
  - Instrumented layers are wrapped in SlaingiMLXLinear, supporting:
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
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from mlx.utils import tree_flatten
from mlx_lm import generate, load
from mlx_lm.sample_utils import make_sampler

from slaingi.expansion import LowRankAdapter
from slaingi.intern import ngram_repetition_ratio
from slaingi.units import LayerDeltaPayload, SkillUnitRegistry

DEFAULT_COMPACT_MODEL = "mlx-community/Qwen3.5-0.8B-4bit"


@dataclass(frozen=True)
class GenerationTrace:
    """Result of one deterministic MLX generation with optional activation capture."""

    prompt: str
    response: str
    tokens_generated: int
    hit_token_limit: bool
    repetition_ratio: float
    layer_activations: dict[str, np.ndarray]


class SlaingiMLXLinear(nn.Module):
    """Hot-swappable adapter wrapper around a frozen MLX Linear / QuantizedLinear layer."""

    def __init__(self, base: nn.Module, layer_name: str) -> None:
        super().__init__()
        self.base = base
        self.layer_name = layer_name
        self._mounted: dict[str, tuple[mx.array, mx.array, float]] = {}

        # Optional trainable LoRA branch (stored as direct attributes for MLX autodiff).
        self.lora_a: mx.array | None = None  # (d_in, rank)
        self.lora_b: mx.array | None = None  # (rank, d_out)
        self._lora_scale: float = 1.0
        self._lora_active: bool = False

        # Toggleable mean activation recorder (stores numpy accumulator off-graph).
        self._recording: bool = False
        self._act_sum: np.ndarray | None = None
        self._act_count: int = 0

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
        """Mount a frozen LowRankAdapter (Delta W = scale * B @ A) onto this layer."""
        d_out, d_in = self.shape
        if adapter.d_out != d_out or adapter.d_in != d_in:
            raise ValueError(
                f"Adapter '{name}' shape ({adapter.d_out}, {adapter.d_in}) does not match "
                f"layer '{self.layer_name}' shape ({d_out}, {d_in})"
            )
        # Store as MLX (A_T: d_in x r, B_T: r x d_out) for efficient (x @ A_T) @ B_T.
        a_t = mx.array(adapter.a.T.astype(np.float32))
        b_t = mx.array(adapter.b.T.astype(np.float32))
        mx.eval(a_t, b_t)
        self._mounted[name] = (a_t, b_t, float(adapter.scale))

    def unmount_adapter(self, name: str) -> None:
        self._mounted.pop(name, None)

    def attach_trainable_lora(
        self,
        *,
        rank: int,
        scale: float = 1.0,
        seed: int = 0,
    ) -> None:
        """Attach a zero-initialized trainable LoRA branch (exact zero perturbation)."""
        d_out, d_in = self.shape
        rng = np.random.default_rng(seed)
        bound = 1.0 / math.sqrt(d_in)
        self.lora_a = mx.array(rng.uniform(-bound, bound, size=(d_in, rank)).astype(np.float32))
        self.lora_b = mx.zeros((rank, d_out), dtype=mx.float32)
        self._lora_scale = float(scale)
        self._lora_active = True
        self.base.freeze()
        mx.eval(self.lora_a, self.lora_b)

    def extract_trainable_lora(self, adapter_name: str) -> LowRankAdapter:
        """Convert the active trainable MLX LoRA branch into a SLAINGI LowRankAdapter."""
        if not self._lora_active or self.lora_a is None or self.lora_b is None:
            raise RuntimeError(f"No trainable LoRA attached on layer '{self.layer_name}'")
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
        self._lora_active = False

    def __call__(self, x: mx.array) -> mx.array:
        y = self.base(x)
        dtype = x.dtype

        if self._mounted:
            x_f32 = x.astype(mx.float32)
            for a_t, b_t, scale in self._mounted.values():
                delta = ((x_f32 @ a_t) @ b_t) * scale
                y = y + delta.astype(dtype)

        if self._lora_active and self.lora_a is not None and self.lora_b is not None:
            x_f32 = x.astype(mx.float32)
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


class MLXKernel:
    """Frozen Apple Silicon quantized base model with governed hot-swap Skill Units."""

    def __init__(
        self,
        model_id: str = DEFAULT_COMPACT_MODEL,
        *,
        target_layer_indices: list[int] | None = None,
        projection_attr: str = "mlp.down_proj",
    ) -> None:
        self.model_id = model_id
        self.model, self.tokenizer = load(model_id)
        self.model.freeze()

        num_layers = len(self.model.layers)
        if target_layer_indices is None:
            # Default: compact set covering middle + upper layers (including MIT Exp 8's 15 & 19).
            candidates = [7, 11, 15, 19, num_layers - 3, num_layers - 2, num_layers - 1]
            target_layer_indices = sorted({i for i in candidates if 0 <= i < num_layers})

        self.instrumented: dict[str, SlaingiMLXLinear] = {}
        for idx in target_layer_indices:
            dotted = f"layers.{idx}.{projection_attr}"
            parent, attr = _resolve_parent_and_attr(self.model, dotted)
            base_mod = getattr(parent, attr)
            wrapped = SlaingiMLXLinear(base_mod, layer_name=dotted)
            setattr(parent, attr, wrapped)
            self.instrumented[dotted] = wrapped

        self._greedy_sampler = make_sampler(temp=0.0)

    def format_chat_prompt(
        self,
        user_prompt: str,
        *,
        system_prompt: str | None = None,
        enable_thinking: bool = False,
    ) -> str:
        messages: list[dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})
        kwargs: dict[str, Any] = {
            "tokenize": False,
            "add_generation_prompt": True,
        }
        template = getattr(self.tokenizer, "chat_template", "") or ""
        if "enable_thinking" in template:
            kwargs["enable_thinking"] = enable_thinking
        return str(self.tokenizer.apply_chat_template(messages, **kwargs))

    def generate_with_trace(
        self,
        user_prompt: str,
        *,
        system_prompt: str | None = None,
        max_tokens: int = 128,
        enable_thinking: bool = False,
        capture_activations: bool = False,
    ) -> GenerationTrace:
        """Run deterministic greedy generation with optional per-layer activation capture."""
        formatted = self.format_chat_prompt(
            user_prompt,
            system_prompt=system_prompt,
            enable_thinking=enable_thinking,
        )
        if capture_activations:
            for layer in self.instrumented.values():
                layer.start_recording()

        try:
            response = generate(
                self.model,
                self.tokenizer,
                prompt=formatted,
                max_tokens=max_tokens,
                sampler=self._greedy_sampler,
                verbose=False,
            )
        finally:
            activations: dict[str, np.ndarray] = {}
            if capture_activations:
                for name, layer in self.instrumented.items():
                    mean_vec = layer.stop_recording()
                    if mean_vec is not None:
                        activations[name] = mean_vec

        token_ids = self.tokenizer.encode(response, add_special_tokens=False)
        n_tokens = len(token_ids)
        hit_limit = n_tokens >= max_tokens
        rep = ngram_repetition_ratio(response)

        return GenerationTrace(
            prompt=user_prompt,
            response=response,
            tokens_generated=n_tokens,
            hit_token_limit=hit_limit,
            repetition_ratio=rep,
            layer_activations=activations,
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

    def mount_unit(self, unit_id: str, registry: SkillUnitRegistry) -> None:
        """Verify SHA-256 integrity and hot-swap a Skill Unit onto instrumented layers."""
        deltas = registry.load_verified_deltas(unit_id)
        for delta in deltas:
            if delta.layer_name not in self.instrumented:
                raise KeyError(
                    f"Skill Unit '{unit_id}' targets layer '{delta.layer_name}' which is not "
                    f"instrumented in this MLXKernel (have {sorted(self.instrumented)})"
                )
            adapter = delta.to_adapter(adapter_name=f"unit:{unit_id}:{delta.layer_name}")
            self.instrumented[delta.layer_name].mount_adapter(adapter.name, adapter)

    def unmount_unit(self, unit_id: str) -> None:
        """Atomically unmount all layer adapters belonging to unit_id."""
        prefix = f"unit:{unit_id}:"
        for layer in self.instrumented.values():
            for name in list(layer.mounted_names):
                if name.startswith(prefix):
                    layer.unmount_adapter(name)

    def train_qlora_adapters(
        self,
        pairs: list[tuple[str, str]] | list[tuple[str | None, str, str]],
        *,
        layer_names: list[str],
        system_prompt: str | None = None,
        enable_thinking: bool = False,
        rank: int = 8,
        scale: float = 2.0,
        steps: int = 40,
        lr: float = 3e-3,
        seed: int = 42,
    ) -> tuple[dict[str, LowRankAdapter], list[float], float]:
        """Function-preserving LoRA expansion + Apple Silicon GPU QLoRA adaptation.

        Freezes the entire base model, attaches zero-initialized LoRA branches to
        layer_names, verifies exact zero logit perturbation before step 0, trains
        ONLY the new LoRA weights via masked causal LM cross-entropy on verified
        (prompt, target) or (system_prompt, prompt, target) examples, then
        extracts and detaches the trained adapters.

        Returns (adapters_by_layer, loss_history, zero_perturbation_max_diff).
        """
        if not pairs:
            raise ValueError("pairs must be non-empty")
        for name in layer_names:
            if name not in self.instrumented:
                raise KeyError(f"Layer '{name}' is not instrumented")

        normalized: list[tuple[str | None, str, str]] = []
        for item in pairs:
            if len(item) == 2:
                normalized.append((system_prompt, item[0], item[1]))
            else:
                normalized.append((item[0], item[1], item[2]))

        # Measure live logits BEFORE expansion (in eval mode).
        self.model.eval()
        probe_sys, probe_prompt, _ = normalized[0]
        logits_before = self.probe_logits(
            probe_prompt,
            system_prompt=probe_sys,
            enable_thinking=enable_thinking,
        )

        # Freeze entire model, then attach zero-initialized trainable LoRA branches.
        self.model.freeze()
        for idx, name in enumerate(layer_names):
            layer = self.instrumented[name]
            layer.attach_trainable_lora(rank=rank, scale=scale, seed=seed + idx)
            layer.unfreeze(keys=["lora_a", "lora_b"], recurse=False)

        # Verify strict zero-perturbation invariant on live transformer logits.
        logits_after = self.probe_logits(
            probe_prompt,
            system_prompt=probe_sys,
            enable_thinking=enable_thinking,
        )
        zero_perturb_diff = float(np.max(np.abs(logits_after - logits_before)))

        # Pre-tokenize all examples with completion-only loss masks.
        eos_id = self.tokenizer.eos_token_id
        encoded_examples: list[tuple[mx.array, mx.array, mx.array]] = []
        for ex_sys, user_prompt, target in normalized:
            prefix = self.format_chat_prompt(
                user_prompt,
                system_prompt=ex_sys,
                enable_thinking=enable_thinking,
            )
            prompt_ids = self.tokenizer.encode(prefix)
            target_ids = self.tokenizer.encode(target, add_special_tokens=False)
            if eos_id is not None and (not target_ids or target_ids[-1] != eos_id):
                target_ids = [*target_ids, int(eos_id)]
            full_ids = prompt_ids + target_ids
            if len(full_ids) < 2:
                continue

            x_ids = mx.array([full_ids[:-1]], dtype=mx.int32)
            y_ids = mx.array([full_ids[1:]], dtype=mx.int32)
            mask_list = [0.0] * (len(prompt_ids) - 1) + [1.0] * len(target_ids)
            mask = mx.array([mask_list], dtype=mx.float32)
            encoded_examples.append((x_ids, y_ids, mask))

        if not encoded_examples:
            raise ValueError("No valid tokenized examples produced from pairs")

        def loss_fn(model: nn.Module, x_ids: mx.array, y_ids: mx.array, mask: mx.array) -> mx.array:
            logits = model(x_ids).astype(mx.float32)
            ce = nn.losses.cross_entropy(logits, y_ids, reduction="none")
            return mx.sum(ce * mask) / mx.maximum(mx.sum(mask), mx.array(1.0))

        loss_and_grad = nn.value_and_grad(self.model, loss_fn)
        lr_schedule = optim.cosine_decay(lr, max(1, steps), end=lr * 0.1)
        optimizer = optim.AdamW(learning_rate=lr_schedule, weight_decay=0.01)

        losses: list[float] = []
        rng = np.random.default_rng(seed)
        order = np.arange(len(encoded_examples))

        # Switch to train() so hybrid models (e.g. Qwen3.5 GatedDeltaNet) use
        # differentiable ops instead of inference-only custom Metal kernels.
        self.model.train()
        try:
            for step in range(steps):
                if step % len(encoded_examples) == 0:
                    rng.shuffle(order)
                ex_idx = int(order[step % len(encoded_examples)])
                x_ids, y_ids, mask = encoded_examples[ex_idx]

                loss_val, grads = loss_and_grad(self.model, x_ids, y_ids, mask)
                grads, _ = optim.clip_grad_norm(grads, max_norm=1.0)
                optimizer.update(self.model, grads)
                mx.eval(loss_val, self.model.parameters(), optimizer.state)
                losses.append(float(loss_val.item()))

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
) -> Path:
    """Export a verified Skill Unit as stock mlx-lm adapters.safetensors + adapter_config.json."""
    manifest = registry.get_manifest(unit_id)
    deltas: list[LayerDeltaPayload] = registry.load_verified_deltas(unit_id)

    destination_dir.mkdir(parents=True, exist_ok=True)
    tensors: dict[str, mx.array] = {}
    ranks: set[int] = set()
    keys: set[str] = set()

    for delta in deltas:
        adapter = delta.to_adapter(adapter_name=delta.layer_name)
        ranks.add(adapter.rank)
        # mlx-lm adapter naming: model.<layer_path>.lora_a (d_in, r), .lora_b (r, d_out)
        prefix = f"model.{delta.layer_name}"
        tensors[f"{prefix}.lora_a"] = mx.array(adapter.a.T.astype(np.float16))
        tensors[f"{prefix}.lora_b"] = mx.array((adapter.scale * adapter.b).T.astype(np.float16))
        parts = delta.layer_name.split(".")
        if len(parts) >= 3:
            keys.add(".".join(parts[2:]))

    safetensors_path = destination_dir / "adapters.safetensors"
    mx.save_safetensors(str(safetensors_path), tensors)

    config = {
        "fine_tune_type": "lora",
        "num_layers": len(deltas),
        "lora_parameters": {
            "rank": max(ranks) if ranks else 1,
            "scale": 1.0,
            "dropout": 0.0,
            "keys": sorted(keys),
        },
        "slaingi_unit": {
            "unit_id": manifest.unit_id,
            "version": manifest.version,
            "payload_sha256": manifest.payload_sha256,
            "effective_bits_per_weight": round(manifest.effective_bits_per_weight, 6),
        },
    }
    (destination_dir / "adapter_config.json").write_text(
        json.dumps(config, indent=2) + "\n",
        encoding="utf-8",
    )
    return destination_dir
