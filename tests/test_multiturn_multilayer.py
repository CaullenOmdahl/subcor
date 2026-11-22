"""Unit tests for Issue #23 (multi-turn assistant-only masking) and Issue #24 (multi-layer hybrid MLX)."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest
from mlx.utils import tree_flatten

from subcor.backends.mlx_kernel import (
    MLXKernel,
    SubcorMLXLinear,
    chunked_gated_delta_ops,
    export_mlx_lora_safetensors,
    select_layer_indices,
)
from subcor.expansion import LowRankAdapter
from subcor.recompile import recompile_units_for_kernel
from subcor.router import RoutingDecision
from subcor.server import InferenceResult, TenantManager
from subcor.types import AdaptationMode, DeltaEncoding, SkillSpec
from subcor.units import LayerDeltaPayload, SkillUnitRegistry, build_layer_delta


class _MiniTokenizer:
    """Deterministic whitespace/tag tokenizer with a Qwen-like chat template for fast unit tests."""

    def __init__(self) -> None:
        self.eos_token = "<|im_end|>"
        self.chat_template = "multi_turn_with_tools"
        self._vocab: dict[str, int] = {"<pad>": 0, "<|im_start|>": 1, "<|im_end|>": 2}
        self.eos_token_id = 2

    def _tok_id(self, piece: str) -> int:
        if piece not in self._vocab:
            self._vocab[piece] = len(self._vocab) % 97 + 3
        return self._vocab[piece]

    def apply_chat_template(
        self,
        messages: list[dict[str, Any]],
        *,
        tokenize: bool = False,
        add_generation_prompt: bool = False,
        tools: list[dict[str, Any]] | None = None,
        **_: Any,
    ) -> str:
        assert not tokenize
        parts: list[str] = []
        if tools:
            parts.append(f"<|im_start|>tools {json.dumps(tools, sort_keys=True)} <|im_end|>\n")
        for m in messages:
            role = m["role"]
            body = m.get("content", "") or ""
            if m.get("tool_calls"):
                tc_chunks = []
                for tc in m["tool_calls"]:
                    fn = tc["function"]
                    tc_chunks.append(
                        f"<tool_call> {fn['name']} {json.dumps(fn['arguments'], sort_keys=True)} </tool_call>"
                    )
                body = (body + " " + " ".join(tc_chunks)).strip()
            parts.append(f"<|im_start|>{role} {body} <|im_end|>\n")
        if add_generation_prompt:
            parts.append("<|im_start|>assistant ")
        return "".join(parts)

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        del add_special_tokens
        return [self._tok_id(tok) for tok in text.replace("\n", " <nl> ").split() if tok]


class _TinyBlock(nn.Module):
    def __init__(self, d: int, *, is_linear: bool = False) -> None:
        super().__init__()
        self.is_linear = is_linear
        self.mlp = SimpleNamespace(down_proj=nn.Linear(d, d, bias=False))
        self.down_proj = self.mlp.down_proj  # register as MLX submodule

    def __call__(self, x: mx.array, mask: Any = None, cache: Any = None) -> mx.array:
        del mask, cache
        return x + nn.relu(self.mlp.down_proj(x))


class _TinyInnerModel(nn.Module):
    def __init__(self, vocab: int = 128, d: int = 16, num_layers: int = 5) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab, d)
        self.layers = [_TinyBlock(d, is_linear=(i % 2 == 1)) for i in range(num_layers)]
        self.norm = nn.RMSNorm(d)


class _TinyLM(nn.Module):
    def __init__(self, vocab: int = 128, d: int = 16, num_layers: int = 5) -> None:
        super().__init__()
        self.model = _TinyInnerModel(vocab=vocab, d=d, num_layers=num_layers)
        self.lm_head = nn.Linear(d, vocab, bias=False)
        self.args = SimpleNamespace(tie_word_embeddings=False)

    @property
    def layers(self) -> list[_TinyBlock]:
        return self.model.layers

    def __call__(self, x_ids: mx.array) -> mx.array:
        h = self.model.embed_tokens(x_ids)
        for lyr in self.model.layers:
            h = lyr(h)
        return self.lm_head(self.model.norm(h))


def _make_tiny_kernel(target_indices: list[int]) -> MLXKernel:
    mx.random.seed(11)
    model = _TinyLM(vocab=128, d=16, num_layers=5)
    model.freeze()
    kernel = object.__new__(MLXKernel)
    kernel.model_id = "tiny-test"
    kernel.model = model
    kernel.tokenizer = _MiniTokenizer()
    kernel.layer_prefix = "model."
    kernel.total_layers = len(model.layers)
    kernel.instrumented = {}
    for idx in target_indices:
        dotted = f"layers.{idx}.mlp.down_proj"
        blk = model.layers[idx]
        wrapped = SubcorMLXLinear(blk.mlp.down_proj, layer_name=dotted)
        blk.mlp.down_proj = wrapped
        blk.down_proj = wrapped
        kernel.instrumented[dotted] = wrapped
    kernel._verified_unit_cache = {}
    kernel.last_train_stats = {}
    return kernel


def test_multiturn_assistant_only_loss_masking_and_gradient_invariance() -> None:
    """Loss & gradients depend ONLY on assistant/tool-call spans, never system/user/tool spans."""
    kernel = _make_tiny_kernel([3, 4])
    tools = [
        {
            "type": "function",
            "function": {
                "name": "run_tests",
                "description": "Run tests",
                "parameters": {"type": "object", "properties": {"path": {"type": "string"}}},
            },
        }
    ]
    traj = {
        "tools": tools,
        "messages": [
            {"role": "system", "content": "SYS_TOKEN_A SYS_TOKEN_B"},
            {"role": "user", "content": "USER_TOKEN_X USER_TOKEN_Y"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "run_tests", "arguments": '{"path":"app.py"}'},
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_1",
                "name": "run_tests",
                "content": "TOOL_OUT_ERR TOOL_OUT_TRACE",
            },
            {"role": "assistant", "content": "ASSIST_FIX_ONE ASSIST_FIX_TWO"},
        ],
    }

    tok_ex, reason = kernel._tokenize_training_example(
        traj,
        default_system_prompt=None,
        enable_thinking=False,
        max_seq_len=2048,
    )
    assert reason == "ok" and tok_ex is not None
    x_ids, y_target, loss_positions = tok_ex

    # Verify target tokens strictly belong to assistant/tool_call spans (never system/user/tool).
    inv_vocab = {v: k for k, v in kernel.tokenizer._vocab.items()}
    supervised_tokens = [inv_vocab.get(int(t), "") for t in np.array(y_target[0]).tolist()]
    for forbidden in (
        "SYS_TOKEN_A",
        "SYS_TOKEN_B",
        "USER_TOKEN_X",
        "USER_TOKEN_Y",
        "TOOL_OUT_ERR",
        "TOOL_OUT_TRACE",
    ):
        assert forbidden not in supervised_tokens
    assert "<tool_call>" in supervised_tokens
    assert "run_tests" in supervised_tokens
    assert "ASSIST_FIX_ONE" in supervised_tokens
    assert "ASSIST_FIX_TWO" in supervised_tokens

    # Attach LoRA and verify gradients are 100% invariant to masking out non-assistant hidden outputs
    # (and strictly sensitive to assistant targets).
    for lyr in kernel.instrumented.values():
        lyr.attach_trainable_lora(rank=4, scale=1.0, binary_ste=False, seed=42)
        lyr.unfreeze(keys=lyr.trainable_lora_keys(), recurse=False)

    seq_len = x_ids.shape[1]
    pos_set = set(np.array(loss_positions).tolist())
    non_assist_mask = mx.array(
        [[[0.0 if i in pos_set else 1.0] for i in range(seq_len)]],
        dtype=mx.float32,
    )

    def _loss_with_poison(model: nn.Module, poison_scale: float, y_tgt: mx.array) -> mx.array:
        h = model.model.embed_tokens(x_ids)
        for lyr in model.model.layers:
            h = lyr(h)
        h_norm = model.model.norm(h)
        # Inject huge arbitrary perturbation at non-assistant positions ONLY before slicing:
        h_poisoned = h_norm + non_assist_mask * poison_scale
        h_sel = h_poisoned[:, loss_positions, :]
        logits = model.lm_head(h_sel).astype(mx.float32)
        return mx.mean(nn.losses.cross_entropy(logits, y_tgt, reduction="none"))

    v0, g0 = nn.value_and_grad(kernel.model, _loss_with_poison)(kernel.model, 0.0, y_target)
    v1, g1 = nn.value_and_grad(kernel.model, _loss_with_poison)(kernel.model, 1e6, y_target)
    y_mut = (y_target + 1) % 97
    v2, g2 = nn.value_and_grad(kernel.model, _loss_with_poison)(kernel.model, 0.0, y_mut)
    mx.eval(v0, g0, v1, g1, v2, g2)

    assert abs(float(v0.item()) - float(v1.item())) == 0.0
    f0 = dict(tree_flatten(g0))
    f1 = dict(tree_flatten(g1))
    f2 = dict(tree_flatten(g2))
    for k in f0:
        assert float(mx.max(mx.abs(f0[k] - f1[k])).item()) == 0.0
    # Mutating assistant targets MUST change the loss and gradients.
    assert abs(float(v0.item()) - float(v2.item())) > 1e-4
    assert any(float(mx.max(mx.abs(f0[k] - f2[k])).item()) > 1e-6 for k in f0)


def test_multiturn_dropped_example_accounting_and_logging(caplog: pytest.LogCaptureFixture) -> None:
    """Sequences exceeding max_seq_len or lacking assistant spans are counted and logged."""
    kernel = _make_tiny_kernel([3, 4])
    valid = {
        "messages": [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello world"},
        ]
    }
    too_long = {
        "messages": [
            {"role": "user", "content": " ".join(f"tok{i}" for i in range(120))},
            {"role": "assistant", "content": "ok"},
        ]
    }
    no_assistant = {
        "messages": [{"role": "user", "content": "only user message"}],
    }

    with caplog.at_level(logging.WARNING, logger="subcor.backends.mlx_kernel"):
        adapters, losses, zero_diff = kernel.train_qlora_adapters(
            [valid, too_long, no_assistant],
            layer_names=sorted(kernel.instrumented.keys()),
            rank=4,
            steps=4,
            lr=1e-3,
            max_seq_len=64,
            seed=7,
        )

    assert zero_diff == 0.0
    assert len(losses) == 4
    assert len(adapters) == 2
    assert kernel.last_train_stats["total_input"] == 3
    assert kernel.last_train_stats["kept"] == 1
    assert kernel.last_train_stats["dropped_too_long"] == 1
    assert kernel.last_train_stats["dropped_empty_target"] == 1
    assert "dropped_too_long=1" in caplog.text
    assert "dropped_empty_target=1" in caplog.text


def test_checkpointed_multilayer_gradients_match_uncheckpointed() -> None:
    """Layer-level mx.checkpoint produces identical updates and exact zero perturbation."""
    traj = {
        "messages": [
            {"role": "system", "content": "Be concise."},
            {"role": "user", "content": "Fix bug in parser."},
            {"role": "assistant", "content": "Check line 42 and return None."},
        ]
    }
    k_plain = _make_tiny_kernel([2, 3, 4])
    k_ckpt = _make_tiny_kernel([2, 3, 4])
    names = sorted(k_plain.instrumented.keys())

    a_plain, l_plain, z_plain = k_plain.train_qlora_adapters(
        [traj],
        layer_names=names,
        rank=4,
        steps=5,
        lr=2e-3,
        grad_checkpoint=False,
        seed=13,
    )
    a_ckpt, l_ckpt, z_ckpt = k_ckpt.train_qlora_adapters(
        [traj],
        layer_names=names,
        rank=4,
        steps=5,
        lr=2e-3,
        grad_checkpoint=True,
        seed=13,
    )
    assert z_plain == 0.0 and z_ckpt == 0.0
    assert k_plain.last_train_stats["grad_checkpoint"] is False
    assert k_ckpt.last_train_stats["grad_checkpoint"] is True
    assert np.allclose(l_plain, l_ckpt, atol=1e-5)
    for name in names:
        assert np.allclose(a_plain[name].delta_matrix(), a_ckpt[name].delta_matrix(), atol=1e-5)
        assert float(np.linalg.norm(a_ckpt[name].delta_matrix())) > 0.0


def test_chunked_gated_delta_ops_matches_reference_forward_and_backward() -> None:
    """Chunkwise-parallel GatedDeltaNet matches stock gated_delta_ops forward & backward."""
    from mlx_lm.models.gated_delta import gated_delta_ops

    mx.random.seed(19)
    b_size, seq_len, hk, hv, dim = 1, 70, 2, 4, 16  # non-multiple of chunk_size=32
    q = mx.random.normal((b_size, seq_len, hk, dim)) * 0.15
    k = mx.random.normal((b_size, seq_len, hk, dim)) * 0.15
    v = mx.random.normal((b_size, seq_len, hv, dim)) * 0.15
    g = mx.sigmoid(mx.random.normal((b_size, seq_len, hv)))
    beta = mx.sigmoid(mx.random.normal((b_size, seq_len, hv)))

    y_ref, s_ref = gated_delta_ops(q, k, v, g, beta)
    y_chk, s_chk = chunked_gated_delta_ops(q, k, v, g, beta, chunk_size=32)
    mx.eval(y_ref, s_ref, y_chk, s_chk)
    assert float(mx.max(mx.abs(y_ref - y_chk)).item()) < 2e-4
    assert float(mx.max(mx.abs(s_ref - s_chk)).item()) < 2e-4

    def f_ref(q_: mx.array, k_: mx.array, v_: mx.array, g_: mx.array, b_: mx.array) -> mx.array:
        y, s = gated_delta_ops(q_, k_, v_, g_, b_)
        return mx.sum(y**2) + 0.1 * mx.sum(s**2)

    def f_chk(q_: mx.array, k_: mx.array, v_: mx.array, g_: mx.array, b_: mx.array) -> mx.array:
        y, s = chunked_gated_delta_ops(q_, k_, v_, g_, b_, chunk_size=32)
        return mx.sum(y**2) + 0.1 * mx.sum(s**2)

    g_ref = mx.grad(f_ref, argnums=(0, 1, 2, 3, 4))(q, k, v, g, beta)
    g_chk = mx.grad(f_chk, argnums=(0, 1, 2, 3, 4))(q, k, v, g, beta)
    mx.eval(g_ref, g_chk)
    for arr_ref, arr_chk in zip(g_ref, g_chk, strict=True):
        assert float(mx.max(mx.abs(arr_ref - arr_chk)).item()) < 5e-4

    # Stress-test aligned unit keys (previously overflowed matrix powers M^p).
    k_aligned = mx.broadcast_to(
        mx.ones((1, 1, hk, dim), dtype=mx.float32) / np.sqrt(dim), (1, 64, hk, dim)
    )
    q_aligned = k_aligned
    v_aligned = mx.ones((1, 64, hv, dim), dtype=mx.float32) * 0.5
    g_ones = mx.ones((1, 64, hv), dtype=mx.float32) * 0.99
    b_ones = mx.ones((1, 64, hv), dtype=mx.float32) * 0.9
    y_al, s_al = chunked_gated_delta_ops(q_aligned, k_aligned, v_aligned, g_ones, b_ones)
    mx.eval(y_al, s_al)
    assert not bool(mx.any(mx.isnan(y_al)).item())
    assert not bool(mx.any(mx.isnan(s_al)).item())


def test_layer_selection_presets_standard_and_hybrid() -> None:
    hybrid = SimpleNamespace(layers=[SimpleNamespace(is_linear=(i % 4 != 3)) for i in range(32)])
    assert select_layer_indices(hybrid, preset="last_n", num_layers=4) == [28, 29, 30, 31]
    assert select_layer_indices(hybrid, preset="periodic_full_attn", num_layers=4) == [
        19,
        23,
        27,
        31,
    ]
    assert select_layer_indices(hybrid, preset="every_k", num_layers=3, stride=4) == [23, 27, 31]

    pure_attn = SimpleNamespace(layers=[SimpleNamespace(is_linear=False) for _ in range(24)])
    assert select_layer_indices(pure_attn, preset="periodic_full_attn", num_layers=4, stride=4) == [
        11,
        15,
        19,
        23,
    ]
    with pytest.raises(ValueError, match="Unknown layer_preset"):
        select_layer_indices(pure_attn, preset="bogus")


def test_noncontiguous_and_heterogeneous_export_mlx_lora_safetensors(tmp_path: Path) -> None:
    registry = SkillUnitRegistry(tmp_path / "units")
    rng = np.random.default_rng(3)

    def _adapter(name: str, d_in: int = 16, d_out: int = 8, rank: int = 4) -> LowRankAdapter:
        return LowRankAdapter(
            name=name,
            a=rng.standard_normal((rank, d_in)),
            b=rng.standard_normal((d_out, rank)),
            scale=1.0,
            frozen=True,
        )

    # Non-contiguous periodic full-attention unit: layers 19, 23, 27, 31
    deltas = [
        build_layer_delta(
            f"layers.{idx}.mlp.down_proj",
            _adapter(f"l{idx}"),
            encoding=DeltaEncoding.SUB1BIT_BINARY,
            sub1bit_rank=4,
        )
        for idx in (19, 23, 27, 31)
    ]
    registry.propose(
        unit_id="periodic-unit",
        version="1.0.0",
        title="Periodic Unit",
        summary="Non-contiguous periodic full-attention unit",
        proposer="tools",
        mode=AdaptationMode.DYNAMIC_EXPANSION_FP32,
        deltas=deltas,
        base_kernel_id="test-4b",
    )
    # Default export preserves mlx-lm last-N-layers convention (32 - 19 = 13) + target_layers metadata.
    out_default = export_mlx_lora_safetensors(
        "periodic-unit",
        registry,
        tmp_path / "export_periodic_default",
        layer_prefix="language_model.model.",
        total_layers=32,
    )
    cfg_def = json.loads((out_default / "adapter_config.json").read_text(encoding="utf-8"))
    assert cfg_def["num_layers"] == 13
    assert cfg_def["lora_parameters"]["keys"] == ["mlp.down_proj"]
    assert len(cfg_def["subcor_unit"]["target_layers"]) == 4

    # Exact-keys export emits num_layers=0 + exact full dotted paths for zero dummy intermediate layers.
    out_dir = export_mlx_lora_safetensors(
        "periodic-unit",
        registry,
        tmp_path / "export_periodic_exact",
        layer_prefix="language_model.model.",
        total_layers=32,
        exact_layer_keys=True,
    )
    cfg = json.loads((out_dir / "adapter_config.json").read_text(encoding="utf-8"))
    assert cfg["num_layers"] == 0
    assert cfg["lora_parameters"]["keys"] == [
        "language_model.model.layers.19.mlp.down_proj",
        "language_model.model.layers.23.mlp.down_proj",
        "language_model.model.layers.27.mlp.down_proj",
        "language_model.model.layers.31.mlp.down_proj",
    ]
    weights = mx.load(str(out_dir / "adapters.safetensors"))
    assert len(weights) == 8
    assert weights["language_model.model.layers.19.mlp.down_proj.lora_a"].shape == (16, 4)


def test_skill_spec_multiturn_trajectories_roundtrip_and_recompile(tmp_path: Path) -> None:
    traj = {
        "messages": [
            {"role": "user", "content": "Diagnose build failure"},
            {"role": "assistant", "content": "Run pytest to inspect the traceback."},
        ]
    }
    spec = SkillSpec(
        unit_id="mt-skill",
        version="1.0.0",
        title="Multi-Turn Debugging",
        summary="Portable multi-turn trajectory spec",
        task_family="debugging",
        proposer="tools",
        system_prompt="Be concise.",
        training_pairs=(),
        training_trajectories=(traj,),
        eval_cases=({"prompt": "Diagnose build failure", "expected": "pytest"},),
        baseline_cases=({"prompt": "Capital of France?", "expected": "Paris"},),
        negative_prompts=("Tell a joke.",),
        steps=4,
    )
    restored = SkillSpec.from_dict(spec.to_dict())
    assert restored.training_trajectories == (traj,)

    # Backward compatibility when older skill_spec.json lacks training_trajectories.
    legacy = spec.to_dict()
    legacy.pop("training_trajectories")
    assert SkillSpec.from_dict(legacy).training_trajectories == ()

    # Recompile passes training_trajectories to the target kernel.
    src_reg = SkillUnitRegistry(tmp_path / "src_units")
    delta = build_layer_delta(
        "layers.0.mlp.down_proj",
        LowRankAdapter(
            name="seed",
            a=np.ones((2, 8)),
            b=np.ones((4, 2)),
            scale=1.0,
            frozen=True,
        ),
        encoding=DeltaEncoding.SUB1BIT_BINARY,
        sub1bit_rank=2,
    )
    src_reg.propose(
        unit_id="mt-skill",
        version="1.0.0",
        title=spec.title,
        summary=spec.summary,
        proposer="tools",
        mode=AdaptationMode.DYNAMIC_EXPANSION_FP32,
        deltas=[delta],
        base_kernel_id="old-base",
    )
    src_reg.record_conformance(
        "mt-skill",
        baseline_before=1.0,
        baseline_after=1.0,
        target_before=0.0,
        target_after=1.0,
    )
    src_reg.save_skill_spec(spec)

    seen_examples: list[Any] = []

    class _FakeKernel:
        model_id = "new-base"
        tap_layer = "layers.0.mlp.down_proj"

        def generate_routed_chat(
            self, *, user_prompt: str, router: Any, **_: Any
        ) -> InferenceResult:
            empty_routing = RoutingDecision(selected_units=(), scores={}, thresholds={})
            if "France" in user_prompt:
                return InferenceResult(
                    text="Paris",
                    prompt_tokens=4,
                    completion_tokens=1,
                    hit_token_limit=False,
                    repetition_ratio=0.0,
                    routing=empty_routing,
                )
            has_unit = bool(router and router.registered_units)
            return InferenceResult(
                text="Run pytest" if has_unit else "Unknown",
                prompt_tokens=4,
                completion_tokens=2,
                hit_token_limit=False,
                repetition_ratio=0.0,
                routing=empty_routing,
            )

        def encode_prompt_embedding(self, user_prompt: str) -> np.ndarray:
            return np.array([1.0, 0.0] if "Diagnose" in user_prompt else [0.0, 1.0])

        def train_skill_deltas(
            self, *, examples: list[Any], **_: Any
        ) -> tuple[list[LayerDeltaPayload], float, float, float]:
            seen_examples.extend(examples)
            return [delta], 1.0, 0.2, 0.0

        def invalidate_unit(self, unit_id: str) -> None:
            del unit_id

    report = recompile_units_for_kernel(
        source_units_dir=tmp_path / "src_units",
        dest_state_dir=tmp_path / "dest_state",
        target_tenant="acme",
        kernel=_FakeKernel(),
    )
    assert report.conformant_count == 1
    assert any(
        isinstance(ex, dict) and ex.get("messages") == traj["messages"] for ex in seen_examples
    )
    dest_mgr = TenantManager(state_dir=tmp_path / "dest_state", kernel=_FakeKernel())
    saved_spec = dest_mgr.get_tenant("acme").registry.load_skill_spec("mt-skill")
    assert saved_spec is not None
    assert saved_spec.training_trajectories == (traj,)
