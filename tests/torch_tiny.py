"""Offline tiny random Qwen3.5 causal LM + char-level tokenizer for CPU torch tests.

Builds a structurally faithful (hybrid GatedDeltaNet linear_attention + gated
full_attention) but tiny, randomly initialised Qwen3_5ForCausalLM from a config,
plus a character-level PreTrainedTokenizerFast with Qwen-style chat special
tokens and a Jinja chat template. Needs no network and no GPU.
"""

from __future__ import annotations

import string
from typing import Any

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
tokenizers = pytest.importorskip("tokenizers")

from tokenizers import Tokenizer, decoders, models  # noqa: E402
from transformers import PreTrainedTokenizerFast  # noqa: E402
from transformers.models.qwen3_5 import Qwen3_5ForCausalLM, Qwen3_5TextConfig  # noqa: E402

SPECIAL_TOKENS = ("<|endoftext|>", "<|im_start|>", "<|im_end|>", "<think>", "</think>")

# Qwen-style template; mentions enable_thinking so kernels forward the kwarg.
TINY_CHAT_TEMPLATE = (
    "{%- for message in messages -%}"
    "{{ '<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>\n' }}"
    "{%- endfor -%}"
    "{%- if add_generation_prompt -%}"
    "{{ '<|im_start|>assistant\n' }}"
    "{%- if enable_thinking is defined and enable_thinking is false -%}"
    "{{ '<think>\n\n</think>\n\n' }}"
    "{%- endif -%}"
    "{%- endif -%}"
)


_LINEAR_ATTN_FUNCS = ("torch_chunk_gated_delta_rule", "torch_recurrent_gated_delta_rule")


def ensure_cpu_safe_linear_attention() -> None:
    """Route GatedDeltaNet kernels to the torch reference path for non-CUDA tensors.

    When flash-linear-attention (``fla``) is installed, transformers binds its
    Triton kernels at import time regardless of device, so a CPU forward crashes
    with "Pointer argument ... cannot be accessed from Triton (cpu tensor?)".
    This idempotently wraps the module-level functions so CPU tensors use the
    original torch implementation while CUDA/ROCm tensors keep the fast path.
    """
    from transformers.models.qwen3_5 import modeling_qwen3_5 as mod

    for fname in _LINEAR_ATTN_FUNCS:
        fast = getattr(mod, fname, None)
        if fast is None or getattr(fast, "_subcor_cpu_safe", False):
            continue
        reference = getattr(fast, "__wrapped__", None)
        if reference is None:
            continue

        def dispatch(*args: Any, _fast: Any = fast, _ref: Any = reference, **kwargs: Any) -> Any:
            first = args[0] if args else kwargs.get("query")
            if isinstance(first, torch.Tensor) and first.device.type == "cuda":
                return _fast(*args, **kwargs)
            if kwargs.get("cu_seqlens") is None:
                kwargs.pop("cu_seqlens", None)
            return _ref(*args, **kwargs)

        dispatch._subcor_cpu_safe = True  # type: ignore[attr-defined]
        setattr(mod, fname, dispatch)


def build_tiny_tokenizer() -> PreTrainedTokenizerFast:
    """Character-level fast tokenizer over printable ASCII + Qwen chat specials."""
    chars = sorted(set(string.printable) - set("\x0b\x0c\r"))
    vocab: dict[str, int] = {}
    for tok in ("<unk>", *SPECIAL_TOKENS):
        vocab[tok] = len(vocab)
    for ch in chars:
        vocab.setdefault(ch, len(vocab))

    # BPE with no merges == pure character-level model; Fuse() decodes by concatenation.
    core = Tokenizer(models.BPE(vocab=vocab, merges=[], unk_token="<unk>"))
    core.decoder = decoders.Fuse()

    tok = PreTrainedTokenizerFast(
        tokenizer_object=core,
        unk_token="<unk>",
        eos_token="<|im_end|>",
        pad_token="<|endoftext|>",
        additional_special_tokens=list(SPECIAL_TOKENS[1:]),
    )
    tok.chat_template = TINY_CHAT_TEMPLATE
    return tok


def build_tiny_config(vocab_size: int, *, num_layers: int = 4, **overrides: Any) -> Any:
    """Tiny hybrid Qwen3.5 text config (full_attention every 4th layer, like the real model)."""
    layer_types = [
        "full_attention" if (i + 1) % 4 == 0 else "linear_attention" for i in range(num_layers)
    ]
    kwargs: dict[str, Any] = {
        "vocab_size": vocab_size,
        "hidden_size": 64,
        "intermediate_size": 128,
        "num_hidden_layers": num_layers,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 16,
        "linear_conv_kernel_dim": 4,
        "linear_key_head_dim": 16,
        "linear_value_head_dim": 16,
        "linear_num_key_heads": 2,
        "linear_num_value_heads": 2,
        "layer_types": layer_types,
        "max_position_embeddings": 512,
        "initializer_range": 0.1,
        "rms_norm_eps": 1e-6,
        "tie_word_embeddings": True,
        # partial rotary: rotary dim 8 -> 4 freqs, so the mrope section must sum to 4.
        "rope_parameters": {
            "rope_type": "default",
            "rope_theta": 10000.0,
            "partial_rotary_factor": 0.5,
            "mrope_interleaved": True,
            "mrope_section": [2, 1, 1],
        },
    }
    kwargs.update(overrides)
    return Qwen3_5TextConfig(**kwargs)


def build_tiny_qwen35(
    seed: int = 0,
    *,
    num_layers: int = 4,
    cpu_safe_linear_attention: bool = False,
) -> tuple[Any, PreTrainedTokenizerFast]:
    """Return (model, tokenizer): a tiny random fp32 Qwen3_5ForCausalLM on CPU in eval mode.

    `cpu_safe_linear_attention` defaults to False so tests exercise TorchKernel's own
    CPU-safe GatedDeltaNet dispatch; pass True to forward the raw model without a kernel.
    """
    if cpu_safe_linear_attention:
        ensure_cpu_safe_linear_attention()
    tokenizer = build_tiny_tokenizer()
    config = build_tiny_config(len(tokenizer), num_layers=num_layers)
    config.eos_token_id = tokenizer.eos_token_id
    config.pad_token_id = tokenizer.pad_token_id
    torch.manual_seed(seed)
    model = Qwen3_5ForCausalLM(config).to(dtype=torch.float32, device="cpu")
    model.eval()
    if getattr(model, "generation_config", None) is not None:
        model.generation_config.eos_token_id = tokenizer.eos_token_id
        model.generation_config.pad_token_id = tokenizer.pad_token_id
    return model, tokenizer
