"""Cross-platform Linux/Windows/macOS GPU & CPU backend via `llama-server` (Issue #8).

Connects SUBCOR's multi-tenant governed runtime (`SubcorHTTPServer`, `MoESkillRouter`,
`TurnConfidence`, `SkillUnitRegistry`) to any `llama.cpp` `llama-server` instance
running on NVIDIA CUDA, AMD ROCm, Vulkan, Metal, or CPU:

  - Frozen-base prompt embeddings via `POST /v1/embeddings` (or `/embedding`).
  - Per-turn multi-turn chat, OpenAI `tools`/`tool_calls`, logprobs, and true
    token-by-token SSE streaming via `POST /v1/chat/completions`.
  - Dynamic LoRA hot-swap via `llama-server`'s `POST /lora-adapters` API (with
    strict `try/finally` reset to scale=0.0 so failed turns never leak adapters).
  - Zero extra Python dependencies (Python stdlib HTTP + NumPy only).
"""

from __future__ import annotations

import json
import urllib.request
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import numpy as np

from subcor.confidence import parse_tool_calls
from subcor.intern import ngram_repetition_ratio
from subcor.router import MoESkillRouter, RoutingDecision
from subcor.server import InferenceResult
from subcor.units import LayerDeltaPayload, SkillUnitRegistry


@dataclass(frozen=True)
class LlamaCppStreamEvent:
    """Incremental token event compatible with SubcorRequestHandler SSE streaming."""

    text: str
    token_id: int
    logprob: float
    prompt_tokens: int
    generation_tokens: int
    finish_reason: str | None = None


class LlamaCppServingAdapter:
    """ServingKernel backend backed by an external `llama-server` (CUDA / ROCm / Vulkan / CPU)."""

    def __init__(
        self,
        *,
        base_url: str = "http://127.0.0.1:8080",
        model_id: str = "llamacpp/local-gguf",
        api_key: str | None = None,
        tap_layer: str = "llamacpp.embedding",
        lora_id_map: dict[str, int] | None = None,
        timeout: float = 60.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model_id = model_id
        self.api_key = api_key
        self.tap_layer = tap_layer
        self.target_layers = (tap_layer,)
        self._lora_id_map: dict[str, int] = dict(lora_id_map or {})
        self.timeout = timeout

    def _headers(self) -> dict[str, str]:
        hdrs = {"Content-Type": "application/json"}
        if self.api_key:
            hdrs["Authorization"] = f"Bearer {self.api_key}"
        return hdrs

    def _post_json(self, path: str, payload: dict[str, Any] | list[Any]) -> Any:
        req = urllib.request.Request(
            f"{self.base_url}{path}",
            data=json.dumps(payload).encode("utf-8"),
            headers=self._headers(),
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            raw = resp.read().decode("utf-8")
            return json.loads(raw) if raw.strip() else {}

    def _set_active_loras(self, selected_units: tuple[str, ...]) -> bool:
        """Activate mapped `llama-server` LoRA adapter IDs via POST /lora-adapters."""
        if not self._lora_id_map:
            return False
        active_set = set(selected_units)
        payload = [
            {"id": int(lora_id), "scale": 1.0 if uid in active_set else 0.0}
            for uid, lora_id in self._lora_id_map.items()
        ]
        self._post_json("/lora-adapters", payload)
        return True

    def _clear_active_loras(self) -> None:
        if not self._lora_id_map:
            return
        payload = [{"id": int(lora_id), "scale": 0.0} for lora_id in self._lora_id_map.values()]
        self._post_json("/lora-adapters", payload)

    def encode_prompt_embedding(self, user_prompt: str) -> np.ndarray:
        """Fetch an invariant frozen-base embedding from llama-server /v1/embeddings."""
        # Ensure all LoRAs are zeroed before embedding so router coordinates stay invariant.
        self._clear_active_loras()
        resp = self._post_json("/v1/embeddings", {"model": self.model_id, "input": user_prompt})
        if isinstance(resp, dict) and "data" in resp and resp["data"]:
            vec = resp["data"][0]["embedding"]
        elif isinstance(resp, dict) and "embedding" in resp:
            vec = resp["embedding"]
        else:
            raise RuntimeError(f"Unexpected llama-server embedding response: {resp!r}")
        arr = np.asarray(vec, dtype=np.float64)
        norm = float(np.linalg.norm(arr))
        return arr / norm if norm > 1e-12 else arr

    def _route(
        self,
        messages: list[dict[str, Any]],
        router: MoESkillRouter | None,
        registry: SkillUnitRegistry | None,
        top_k: int,
        seat: str | None,
    ) -> RoutingDecision:
        if router is None or registry is None or not router.registered_units:
            return RoutingDecision(selected_units=(), scores={}, thresholds={})
        last_user = next(
            (str(m.get("content", "")) for m in reversed(messages) if m.get("role") == "user"),
            "",
        )
        emb = self.encode_prompt_embedding(last_user)
        decision = router.route(emb, top_k=top_k, seat=seat)
        # Verify SHA-256 payload integrity on disk for every selected Skill Unit.
        for uid in decision.selected_units:
            registry.load_verified_deltas(uid)
        return decision

    def generate_routed_messages(
        self,
        *,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        router: MoESkillRouter | None,
        registry: SkillUnitRegistry | None,
        max_tokens: int,
        enable_thinking: bool,
        top_k: int,
        seat: str | None = None,
    ) -> InferenceResult:
        decision = self._route(messages, router, registry, top_k, seat)
        loras_toggled = False
        try:
            if decision.selected_units:
                loras_toggled = self._set_active_loras(decision.selected_units)
            req_body: dict[str, Any] = {
                "model": self.model_id,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": 0.0,
                "logprobs": True,
            }
            if tools:
                req_body["tools"] = tools
            body = self._post_json("/v1/chat/completions", req_body)
        finally:
            if loras_toggled:
                self._clear_active_loras()

        choice = body["choices"][0]
        msg = choice.get("message") or {}
        raw_content = str(msg.get("content") or "")
        native_tool_calls = msg.get("tool_calls")
        if isinstance(native_tool_calls, list) and native_tool_calls:
            clean_text = raw_content.strip()
            tool_calls = tuple(native_tool_calls)
        else:
            clean_text, parsed_calls = parse_tool_calls(raw_content)
            tool_calls = tuple(parsed_calls)

        # Extract per-token logprobs when provided by llama-server.
        token_lps: list[float] = []
        lp_obj = choice.get("logprobs")
        if isinstance(lp_obj, dict) and isinstance(lp_obj.get("content"), list):
            for entry in lp_obj["content"]:
                if isinstance(entry, dict) and entry.get("logprob") is not None:
                    token_lps.append(float(entry["logprob"]))

        usage = body.get("usage") or {}
        prompt_tokens = int(usage.get("prompt_tokens", 0))
        completion_tokens = int(usage.get("completion_tokens", max(1, len(raw_content.split()))))
        hit_limit = choice.get("finish_reason") == "length"

        return InferenceResult(
            text=clean_text,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            hit_token_limit=hit_limit,
            repetition_ratio=ngram_repetition_ratio(raw_content),
            routing=decision,
            tool_calls=tool_calls,
            token_logprobs=tuple(token_lps),
        )

    def stream_routed_messages(
        self,
        *,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        router: MoESkillRouter | None,
        registry: SkillUnitRegistry | None,
        max_tokens: int,
        enable_thinking: bool,
        top_k: int,
        seat: str | None = None,
    ) -> tuple[RoutingDecision, Iterator[LlamaCppStreamEvent]]:
        decision = self._route(messages, router, registry, top_k, seat)

        def _gen() -> Iterator[LlamaCppStreamEvent]:
            loras_toggled = False
            try:
                if decision.selected_units:
                    loras_toggled = self._set_active_loras(decision.selected_units)
                req_body: dict[str, Any] = {
                    "model": self.model_id,
                    "messages": messages,
                    "max_tokens": max_tokens,
                    "temperature": 0.0,
                    "stream": True,
                    "logprobs": True,
                }
                if tools:
                    req_body["tools"] = tools
                req = urllib.request.Request(
                    f"{self.base_url}/v1/chat/completions",
                    data=json.dumps(req_body).encode("utf-8"),
                    headers=self._headers(),
                    method="POST",
                )
                tok_idx = 0
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    for raw_line in resp:
                        line = raw_line.decode("utf-8").strip()
                        if not line.startswith("data: "):
                            continue
                        payload = line[6:]
                        if payload == "[DONE]":
                            break
                        chunk = json.loads(payload)
                        choice = chunk["choices"][0]
                        delta = choice.get("delta") or {}
                        text = str(delta.get("content") or "")
                        finish = choice.get("finish_reason")
                        lp = 0.0
                        lp_obj = choice.get("logprobs")
                        if (
                            isinstance(lp_obj, dict)
                            and isinstance(lp_obj.get("content"), list)
                            and lp_obj["content"]
                            and lp_obj["content"][0].get("logprob") is not None
                        ):
                            lp = float(lp_obj["content"][0]["logprob"])
                        if text or finish:
                            tok_idx += 1
                            yield LlamaCppStreamEvent(
                                text=text,
                                token_id=tok_idx,
                                logprob=lp,
                                prompt_tokens=0,
                                generation_tokens=tok_idx,
                                finish_reason=finish,
                            )
            finally:
                if loras_toggled:
                    self._clear_active_loras()

        return decision, _gen()

    def generate_routed_chat(
        self,
        *,
        user_prompt: str,
        system_prompt: str | None,
        router: MoESkillRouter | None,
        registry: SkillUnitRegistry | None,
        max_tokens: int,
        enable_thinking: bool,
        top_k: int,
    ) -> InferenceResult:
        messages: list[dict[str, Any]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})
        return self.generate_routed_messages(
            messages=messages,
            tools=None,
            router=router,
            registry=registry,
            max_tokens=max_tokens,
            enable_thinking=enable_thinking,
            top_k=top_k,
        )

    def train_skill_deltas(
        self,
        *,
        examples: list[tuple[str | None, str, str]],
        rank: int,
        sub1bit_rank: int,
        steps: int,
        lr: float,
        seed: int,
    ) -> tuple[list[LayerDeltaPayload], float, float, float]:
        raise NotImplementedError(
            "LlamaCppServingAdapter is an inference/routing backend for llama-server. "
            "Train Skill Units on an MLX or PyTorch trainer node and mount exported GGUF LoRAs."
        )

    def invalidate_unit(self, unit_id: str) -> None:
        self._lora_id_map.pop(unit_id, None)
