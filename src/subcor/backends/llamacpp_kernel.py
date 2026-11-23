"""Cross-platform Linux/Windows/macOS GPU & CPU backend via `llama-server` (Issue #8).

Connects SUBCOR's multi-tenant governed runtime (`SubcorHTTPServer`, `MoESkillRouter`,
`TurnConfidence`, `SkillUnitRegistry`) to a `llama.cpp` `llama-server` instance
running on NVIDIA CUDA, AMD ROCm/HIP, Vulkan, Metal, or CPU:

  - Frozen-base prompt embeddings via `POST /v1/embeddings`. If the server runs with
    the default pooling (`none`), that endpoint answers 400 and the adapter falls back
    to `POST /embedding` and mean-pools the per-token vectors. Starting the server
    with `--embeddings --pooling mean` is recommended (the supervisor always does).
  - Per-turn multi-turn chat, OpenAI `tools`/`tool_calls`, logprobs, and true
    token-by-token SSE streaming via `POST /v1/chat/completions`.
  - LoRA selection for routed Skill Units, in one of two modes:
      * `request`: the per-request `"lora": [{"id": N, "scale": S}]` field of the chat
        request. Nothing global changes, so concurrent requests cannot leak adapters.
        Verified against llama.cpp b12110 and 1.0.0 (b12359).
      * `global`: `POST /lora-adapters` before the request and a strict `try/finally`
        reset to scale 0.0 afterwards, for older servers.
  - Optional `LlamaServerSupervisor` (SUBCOR owns `llama-server`) plus a PyTorch
    trainer built from the SAME base GGUF, so Study Cycles can train Skill Units,
    export them as GGUF LoRA adapters, and load them into the serving process.

Without a supervisor or trainer this backend is inference-only, with zero extra
Python dependencies (stdlib HTTP + NumPy).
"""

from __future__ import annotations

import gc
import hashlib
import json
import re
import threading
import traceback
import urllib.error
import urllib.request
import warnings
from collections import OrderedDict
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from subcor.confidence import parse_tool_calls
from subcor.expansion import LowRankAdapter
from subcor.intern import ngram_repetition_ratio
from subcor.router import MoESkillRouter, RoutingDecision, StickyRoutingPolicy
from subcor.server import InferenceResult
from subcor.types import DeltaEncoding, UnitState
from subcor.units import LayerDeltaPayload, SkillUnitRegistry, build_layer_delta

if TYPE_CHECKING:
    from subcor.backends.llamacpp_supervisor import LlamaServerSupervisor

_LORA_MODES = ("auto", "request", "global")
# Conversations remembered for sticky routing (least recently used are dropped).
_STICKY_MAX_CONVERSATIONS = 4096
# Rough characters per token for the sticky-routing context-length estimate.
_CHARS_PER_TOKEN = 4
_UNSAFE_KEY_CHARS = re.compile(r"[^A-Za-z0-9._-]+")
_LAYER_IDX_RE = re.compile(r"layers\.(\d+)\.")
# Unit states whose supervised adapter is kept across llama-server restarts. A DORMANT
# unit is kept only while it has no conformance result yet (a fresh candidate); one that
# failed conformance, like a ROLLED_BACK one, is dropped at the next restart.
_LIVE_ADAPTER_STATES = frozenset({UnitState.PROPOSED, UnitState.CONFORMANT, UnitState.ENABLED})


def _layer_sort_key(layer_name: str) -> tuple[int, str]:
    m = _LAYER_IDX_RE.search(layer_name)
    return (int(m.group(1)) if m else 10**9, layer_name)


@dataclass(frozen=True)
class TrainedAdapters:
    """Trained-but-unpacked Study Cycle adapters (CPU NumPy; no accelerator memory held)."""

    layer_names: tuple[str, ...]
    adapters: dict[str, LowRankAdapter]
    initial_loss: float
    final_loss: float
    zero_perturbation_max_diff: float


@dataclass(frozen=True)
class LlamaCppStreamEvent:
    """Incremental token event compatible with SubcorRequestHandler SSE streaming."""

    text: str
    token_id: int
    logprob: float
    prompt_tokens: int
    generation_tokens: int
    finish_reason: str | None = None


def _mean_pool_embedding(vec: Any) -> list[float]:
    """Return a 1-D vector, mean-pooling a per-token (list of lists) embedding."""
    arr = np.asarray(vec, dtype=np.float64)
    if arr.ndim == 2:
        if arr.shape[0] == 0:
            raise RuntimeError("llama-server returned an empty per-token embedding")
        arr = arr.mean(axis=0)
    if arr.ndim != 1:
        raise RuntimeError(f"Unexpected llama-server embedding shape {arr.shape}")
    return arr.tolist()


def _is_pooling_none_error(exc: urllib.error.HTTPError) -> bool:
    if exc.code != 400:
        return False
    try:
        body = exc.read().decode("utf-8", errors="replace").lower()
    except Exception:  # pragma: no cover - defensive
        return False
    return "pooling" in body


# llama-server embeds a whole input in one physical batch and rejects anything longer
# ("input (N tokens) is too large to process. increase the physical batch size (current
# batch size: M)"). Clipped routing inputs leave this many tokens of headroom for BOS/EOS
# and re-tokenization drift.
_EMBED_TOKEN_MARGIN = 16
_TOO_LARGE_RE = re.compile(r"too large to process.*current batch size:\s*(\d+)", re.DOTALL)


def _input_too_large_limit(exc: urllib.error.HTTPError) -> int | None:
    """Physical batch size from llama-server's 'input too large' error, else None."""
    try:
        body = exc.read().decode("utf-8", errors="replace")
    except Exception:  # pragma: no cover - defensive
        return None
    m = _TOO_LARGE_RE.search(body)
    return int(m.group(1)) if m else None


def _read_gguf_block_count(base_gguf: Path) -> int | None:
    """Best-effort number of DECODER layers from a GGUF header (needs gguf-py).

    `<arch>.block_count` minus `<arch>.nextn_predict_layers`: llama.cpp counts a
    multi-token-prediction block (Qwen3.5 / Qwen3-Next, GLM, ...) as an extra `blk.N`,
    while transformers builds `num_hidden_layers = block_count - nextn_predict_layers`
    and maps that block to `mtp.*`, so it is not a trainable decoder layer.
    """
    try:
        import gguf  # type: ignore[import-not-found]

        reader = gguf.GGUFReader(str(base_gguf))
        arch_field = reader.fields["general.architecture"]
        arch = bytes(arch_field.parts[arch_field.data[0]]).decode("utf-8")
        bc = reader.fields[f"{arch}.block_count"]
        n_blocks = int(bc.parts[bc.data[0]][0])
        mtp_field = reader.fields.get(f"{arch}.nextn_predict_layers")
        n_mtp = int(mtp_field.parts[mtp_field.data[0]][0]) if mtp_field is not None else 0
        n_layers = n_blocks - max(0, n_mtp)
        return n_layers if n_layers > 0 else None
    except Exception:
        return None


def _release_accelerator_cache() -> None:
    """Collect garbage and return cached PyTorch accelerator memory to the driver.

    PyTorch's caching allocator keeps freed blocks reserved for the process; another
    process (llama-server) can only use that VRAM after `empty_cache()`.
    """
    gc.collect()
    try:
        import torch
    except ImportError:
        return
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


class LlamaCppServingAdapter:
    """ServingKernel backend backed by `llama-server` (CUDA / ROCm / Vulkan / CPU).

    External mode (`base_url`, optional static `lora_id_map` keyed by unit_id): inference
    and routing only. Supervised mode (`supervisor=`): SUBCOR owns `llama-server`;
    `train_skill_deltas` trains on a PyTorch trainer loaded from the same base GGUF, and
    `prepare_units` exports units as GGUF LoRAs and restarts `llama-server` so they load.
    """

    def __init__(
        self,
        *,
        base_url: str = "http://127.0.0.1:8080",
        model_id: str = "llamacpp/local-gguf",
        api_key: str | None = None,
        tap_layer: str = "llamacpp.embedding",
        lora_id_map: dict[str, int] | None = None,
        timeout: float = 60.0,
        supervisor: LlamaServerSupervisor | None = None,
        base_gguf: str | Path | None = None,
        lora_dir: str | Path | None = None,
        lora_mode: str = "auto",
        trainer_factory: Callable[[], Any] | None = None,
        trainer_device: str | None = None,
        trainer_dtype: Any = None,
        trainer_layer_indices: Sequence[int] | None = None,
        trainer_projections: tuple[str, ...] | None = None,
        trainer_quantize_4bit: bool = False,
        unload_trainer_after_study: bool = False,
        sticky_routing: StickyRoutingPolicy | None = None,
    ) -> None:
        if lora_mode not in _LORA_MODES:
            raise ValueError(f"lora_mode must be one of {_LORA_MODES}, got {lora_mode!r}")
        self.supervisor = supervisor
        self.base_url = (supervisor.base_url if supervisor is not None else base_url).rstrip("/")
        self.model_id = model_id
        self.api_key = api_key
        self.tap_layer = tap_layer
        self.target_layers = (tap_layer,)
        self._lora_id_map: dict[str, int] = dict(lora_id_map or {})
        self.timeout = timeout
        self.lora_mode = (
            ("request" if supervisor is not None else "global")
            if (lora_mode == "auto")
            else lora_mode
        )

        if base_gguf is None and supervisor is not None:
            base_gguf = supervisor.base_gguf
        self.base_gguf: Path | None = (
            Path(base_gguf).expanduser().absolute() if base_gguf is not None else None
        )
        if lora_dir is None and supervisor is not None:
            lora_dir = supervisor.lora_dir
        self.lora_dir: Path | None = (
            Path(lora_dir).expanduser().resolve() if lora_dir is not None else None
        )

        # Supervised adapters: adapter key (unit_id + payload digest) -> llama LoRA id / path.
        self._managed_ids: dict[str, int] = {}
        self._managed_paths: dict[str, Path] = {}
        # adapter key -> (registry, unit_id) pairs that asked for it (liveness checks).
        self._adapter_owners: dict[str, list[tuple[SkillUnitRegistry, str]]] = {}
        self._units_lock = threading.RLock()
        self._closed = False
        self._warned_unmapped: set[str] = set()
        self._embedding_path = "/v1/embeddings"

        # Sticky routing: conversation key -> units the previous turn ran with.
        self.sticky_routing = sticky_routing
        self._sticky_units: OrderedDict[str, tuple[str, ...]] = OrderedDict()
        self._sticky_lock = threading.Lock()

        self._trainer_factory = trainer_factory
        self.trainer_device = trainer_device
        self.trainer_dtype = trainer_dtype
        self.trainer_layer_indices = (
            list(trainer_layer_indices) if trainer_layer_indices is not None else None
        )
        self.trainer_projections = trainer_projections
        self.trainer_quantize_4bit = bool(trainer_quantize_4bit)
        self.unload_trainer_after_study = unload_trainer_after_study
        self._trainer: Any = None

    # ------------------------------------------------------------------ HTTP
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

    # ------------------------------------------------------------------ LoRA selection
    @property
    def lora_id_map(self) -> dict[str, int]:
        """All known adapter ids: static unit_id map plus supervised adapter keys."""
        with self._units_lock:
            return {**self._lora_id_map, **self._managed_ids}

    def _all_lora_ids(self) -> list[int]:
        with self._units_lock:
            return sorted(set(self._lora_id_map.values()) | set(self._managed_ids.values()))

    @staticmethod
    def adapter_key(registry: SkillUnitRegistry, unit_id: str) -> str:
        """Stable adapter key for one unit payload (unit_id + payload SHA-256 prefix).

        Keyed by content so two tenants with the same unit_id never share an adapter.
        """
        manifest = registry.get_manifest(unit_id)
        safe_uid = _UNSAFE_KEY_CHARS.sub("_", unit_id)[:64]
        return f"{safe_uid}-{manifest.payload_sha256[:16]}"

    def _resolve_lora_ids(
        self, selected_units: tuple[str, ...], registry: SkillUnitRegistry | None
    ) -> list[int]:
        """Map routed unit_ids to llama-server LoRA ids (loading supervised units if needed)."""
        ids: list[int] = []
        for uid in selected_units:
            if self.supervisor is not None and registry is not None:
                key = self.adapter_key(registry, uid)
                with self._units_lock:
                    lid = self._managed_ids.get(key)
                if lid is None:
                    # Self-heal: a routable unit whose adapter is not loaded yet.
                    self.prepare_units([(registry, uid)])
                    with self._units_lock:
                        lid = self._managed_ids.get(key)
                if lid is None:
                    raise RuntimeError(f"Skill Unit '{uid}' has no loaded GGUF LoRA adapter")
                ids.append(lid)
            elif uid in self._lora_id_map:
                ids.append(int(self._lora_id_map[uid]))
            elif uid not in self._warned_unmapped:
                # External mode: routing selected a unit the external llama-server has no
                # adapter for (no --lora-id-map entry), so the request runs on the base.
                self._warned_unmapped.add(uid)
                warnings.warn(
                    f"Routed Skill Unit '{uid}' has no llama-server LoRA id (pass "
                    "--lora-id-map, or use --gguf so SUBCOR loads adapters itself); "
                    "the request runs on the base model.",
                    RuntimeWarning,
                    stacklevel=2,
                )
        return ids

    def _lora_request_field(self, active_ids: list[int]) -> list[dict[str, Any]]:
        active = set(active_ids)
        return [{"id": int(i), "scale": 1.0 if i in active else 0.0} for i in self._all_lora_ids()]

    def _set_active_loras(self, active_ids: list[int]) -> bool:
        """Global mode: activate LoRA ids via POST /lora-adapters (all others scale 0)."""
        all_ids = self._all_lora_ids()
        if not all_ids:
            return False
        self._post_json("/lora-adapters", self._lora_request_field(active_ids))
        return True

    def _clear_active_loras(self) -> None:
        all_ids = self._all_lora_ids()
        if not all_ids:
            return
        self._post_json("/lora-adapters", [{"id": int(i), "scale": 0.0} for i in all_ids])

    def _chat_body(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        max_tokens: int,
        enable_thinking: bool,
        active_ids: list[int],
        *,
        stream: bool,
    ) -> dict[str, Any]:
        body: dict[str, Any] = {
            "model": self.model_id,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": 0.0,
            "logprobs": True,
            # Qwen3.x templates think by default; honour the caller's choice explicitly.
            "chat_template_kwargs": {"enable_thinking": bool(enable_thinking)},
        }
        if stream:
            body["stream"] = True
        if tools:
            body["tools"] = tools
        if self.lora_mode == "request" and self._all_lora_ids():
            body["lora"] = self._lora_request_field(active_ids)
        return body

    # ------------------------------------------------------------------ embeddings
    def _extract_embedding(self, resp: Any) -> list[float]:
        if isinstance(resp, dict) and resp.get("data"):
            return _mean_pool_embedding(resp["data"][0]["embedding"])
        if isinstance(resp, dict) and "embedding" in resp:
            return _mean_pool_embedding(resp["embedding"])
        if isinstance(resp, list) and resp and isinstance(resp[0], dict) and "embedding" in resp[0]:
            return _mean_pool_embedding(resp[0]["embedding"])
        raise RuntimeError(f"Unexpected llama-server embedding response: {resp!r}")

    def encode_prompt_embedding(self, user_prompt: str) -> np.ndarray:
        """Fetch an invariant frozen-base embedding from llama-server.

        Embeddings always use llama-server's GLOBAL adapter scales (the per-request
        `lora` field is ignored by the embedding endpoints), so global scales are kept at
        0: in `global` mode they are reset here; in `request` mode they are never set.

        llama-server embeds an input in one physical batch (`-ub`, default 512 tokens). A
        longer prompt is clipped to its last tokens that fit, so routing keeps working on
        long user messages instead of failing the whole chat request.
        """
        self._ensure_server()
        if self.lora_mode == "global":
            self._clear_active_loras()
        try:
            return self._embed_text(user_prompt)
        except urllib.error.HTTPError as exc:
            limit = _input_too_large_limit(exc)
            if limit is None:
                raise
        return self._embed_text(self._clip_to_last_tokens(user_prompt, limit))

    def _clip_to_last_tokens(self, text: str, batch_limit: int) -> str:
        keep = max(1, batch_limit - _EMBED_TOKEN_MARGIN)
        tokens = self._post_json("/tokenize", {"content": text, "add_special": False})["tokens"]
        if len(tokens) <= keep:
            keep = max(1, len(tokens) - _EMBED_TOKEN_MARGIN)
        return str(self._post_json("/detokenize", {"tokens": tokens[-keep:]})["content"])

    def _embed_text(self, user_prompt: str) -> np.ndarray:
        if self._embedding_path == "/v1/embeddings":
            try:
                resp = self._post_json(
                    "/v1/embeddings", {"model": self.model_id, "input": user_prompt}
                )
            except urllib.error.HTTPError as exc:
                if not _is_pooling_none_error(exc):
                    raise
                # Server started without --pooling: use the native per-token endpoint.
                self._embedding_path = "/embedding"
            else:
                return self._normalize(self._extract_embedding(resp))
        resp = self._post_json("/embedding", {"content": user_prompt})
        return self._normalize(self._extract_embedding(resp))

    @staticmethod
    def _normalize(vec: list[float]) -> np.ndarray:
        arr = np.asarray(vec, dtype=np.float64)
        norm = float(np.linalg.norm(arr))
        return arr / norm if norm > 1e-12 else arr

    # ------------------------------------------------------------------ routing + chat
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
        if self.sticky_routing is not None:
            decision = self._apply_sticky(messages, router, seat, decision)
        # Verify SHA-256 payload integrity on disk for every selected Skill Unit.
        for uid in decision.selected_units:
            registry.load_verified_deltas(uid)
        return decision

    @staticmethod
    def _conversation_key(
        messages: list[dict[str, Any]], router: MoESkillRouter, seat: str | None
    ) -> str | None:
        """Identify a conversation by its opening system and user messages.

        Chat requests carry no conversation id, but every turn of one conversation resends
        the same opening. Two conversations with an identical opening share a key; the
        policy only biases which units run, so the cost of a collision is a less fresh
        routing decision, not wrong output from another tenant (the router is part of the
        key, and each tenant has its own router).
        """
        system = next((m.get("content") for m in messages if m.get("role") == "system"), None)
        first_user = next((m.get("content") for m in messages if m.get("role") == "user"), None)
        if first_user is None:
            return None
        raw = json.dumps([id(router), seat, system, first_user], sort_keys=True, default=str)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    @staticmethod
    def _estimate_context_tokens(messages: list[dict[str, Any]]) -> int:
        chars = sum(len(json.dumps(m.get("content", ""), default=str)) for m in messages)
        return chars // _CHARS_PER_TOKEN

    def _apply_sticky(
        self,
        messages: list[dict[str, Any]],
        router: MoESkillRouter,
        seat: str | None,
        fresh: RoutingDecision,
    ) -> RoutingDecision:
        policy = self.sticky_routing
        key = self._conversation_key(messages, router, seat)
        if policy is None or key is None:
            return fresh
        with self._sticky_lock:
            previous = self._sticky_units.get(key)
        decision = policy.apply(
            previous, fresh, context_tokens=self._estimate_context_tokens(messages)
        )
        with self._sticky_lock:
            self._sticky_units[key] = decision.selected_units
            self._sticky_units.move_to_end(key)
            while len(self._sticky_units) > _STICKY_MAX_CONVERSATIONS:
                self._sticky_units.popitem(last=False)
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
        self._ensure_server()
        decision = self._route(messages, router, registry, top_k, seat)
        active_ids = (
            self._resolve_lora_ids(decision.selected_units, registry)
            if decision.selected_units
            else []
        )
        loras_toggled = False
        try:
            if active_ids and self.lora_mode == "global":
                loras_toggled = self._set_active_loras(active_ids)
            req_body = self._chat_body(
                messages, tools, max_tokens, enable_thinking, active_ids, stream=False
            )
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
        timings = body.get("timings")

        return InferenceResult(
            text=clean_text,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            hit_token_limit=hit_limit,
            repetition_ratio=ngram_repetition_ratio(raw_content),
            routing=decision,
            tool_calls=tool_calls,
            token_logprobs=tuple(token_lps),
            server_timings=dict(timings) if isinstance(timings, dict) else None,
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
        self._ensure_server()
        decision = self._route(messages, router, registry, top_k, seat)
        active_ids = (
            self._resolve_lora_ids(decision.selected_units, registry)
            if decision.selected_units
            else []
        )

        def _gen() -> Iterator[LlamaCppStreamEvent]:
            loras_toggled = False
            try:
                if active_ids and self.lora_mode == "global":
                    loras_toggled = self._set_active_loras(active_ids)
                req_body = self._chat_body(
                    messages, tools, max_tokens, enable_thinking, active_ids, stream=True
                )
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
                        if not chunk.get("choices"):
                            continue
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

    # ------------------------------------------------------------------ training
    def _default_trainer_layer_indices(self) -> list[int] | None:
        """Mirror MLXServingAdapter (last 6 layers, i.e. 18..23 on Qwen3.5-0.8B)."""
        if self.trainer_layer_indices is not None:
            return list(self.trainer_layer_indices)
        if self.base_gguf is None:
            return None
        n_layers = _read_gguf_block_count(self.base_gguf)
        if n_layers is None:
            return None
        return list(range(max(0, n_layers - 6), n_layers))

    def _build_trainer(self) -> Any:
        if self._trainer_factory is not None:
            return self._trainer_factory()
        if self.base_gguf is None:
            raise NotImplementedError(
                "This llama.cpp backend is inference-only: it is connected to an external "
                "llama-server without a base GGUF. Start SUBCOR with "
                "'subcor serve --backend llamacpp --gguf /path/to/model.gguf' so it can train "
                "Skill Units on a PyTorch trainer loaded from the same GGUF."
            )
        try:
            from subcor.backends.torch_kernel import load_trainer_from_gguf
        except ImportError as exc:
            raise RuntimeError(
                "Training on the llama.cpp backend needs the PyTorch trainer "
                "(torch + transformers + gguf). Install with "
                "'uv sync --extra rocm' (AMD) or an equivalent CUDA/CPU torch install."
            ) from exc
        kwargs: dict[str, Any] = {
            "target_layer_indices": self._default_trainer_layer_indices(),
            "device": self.trainer_device,
            "dtype": self.trainer_dtype,  # str or torch.dtype; TorchKernel resolves it
        }
        if self.trainer_quantize_4bit:
            kwargs["quantize_base_weights"] = True
        if self.trainer_projections is not None:
            kwargs["target_projections"] = self.trainer_projections
        return load_trainer_from_gguf(self.base_gguf, **kwargs)

    @property
    def trainer(self) -> Any:
        """The lazily built PyTorch trainer (loaded from the serving GGUF on first use)."""
        if self._trainer is None:
            self._trainer = self._build_trainer()
        return self._trainer

    def unload_trainer(self) -> None:
        """Drop the trainer and release its accelerator memory.

        Only frees the memory if nothing else still references the trainer, so callers
        must drop their own references first (see `train_skill_deltas`).
        """
        self._trainer = None
        _release_accelerator_cache()

    def _check_layers_exportable(self, layer_names: Sequence[str]) -> None:
        if self.supervisor is None:
            return
        from subcor.gguf_export import canonical_to_gguf_base_tensor

        unsupported = []
        for name in layer_names:
            try:
                canonical_to_gguf_base_tensor(name)
            except KeyError:
                unsupported.append(name)
        if unsupported:
            raise ValueError(
                f"Projections {unsupported} cannot be exported as GGUF LoRA tensors; "
                "choose supported --projections for the llama.cpp backend."
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
        binary_ste: bool = False,
    ) -> tuple[list[LayerDeltaPayload], float, float, float]:
        """Train on the PyTorch trainer (dequantized from the served GGUF) -> SUB1BIT deltas.

        With `unload_trainer_after_study` the trainer is dropped afterwards and its VRAM
        returned to the driver (so llama-server can use it). Otherwise the trainer stays
        resident, but PyTorch's cached activation memory is still released.
        """
        trained = self.train_skill_adapters(
            examples=examples,
            rank=rank,
            steps=steps,
            lr=lr,
            seed=seed,
            binary_ste=binary_ste,
        )
        deltas = self.pack_skill_deltas(trained, sub1bit_rank=sub1bit_rank)
        return deltas, trained.initial_loss, trained.final_loss, trained.zero_perturbation_max_diff

    def train_skill_adapters(
        self,
        *,
        examples: list[tuple[str | None, str, str]],
        rank: int,
        steps: int,
        lr: float,
        seed: int,
        binary_ste: bool = False,
    ) -> TrainedAdapters:
        """GPU half of a Study Cycle: train adapters, then release (or unload) the trainer.

        Returns plain CPU (NumPy) adapters, so nothing on the accelerator is held after this
        returns. `pack_skill_deltas` does the CPU-only sub-1-bit packing; the server calls it
        outside its kernel lock so live chat is not blocked while units are packed.
        """
        try:
            # The trainer is only referenced inside `_train`'s frame, which is gone once it
            # returns, so the memory release below can actually free it.
            result = self._train(
                examples=examples,
                rank=rank,
                steps=steps,
                lr=lr,
                seed=seed,
                binary_ste=binary_ste,
            )
        except BaseException as exc:
            if self.unload_trainer_after_study:
                # The traceback keeps the finished frames (and their `trainer` local) alive.
                traceback.clear_frames(exc.__traceback__)
                self.unload_trainer()
            else:
                _release_accelerator_cache()
            raise
        if self.unload_trainer_after_study:
            self.unload_trainer()
        else:
            _release_accelerator_cache()
        return result

    def pack_skill_deltas(
        self, trained: TrainedAdapters, *, sub1bit_rank: int
    ) -> list[LayerDeltaPayload]:
        """CPU-only half of a Study Cycle: pack trained adapters as SUB1BIT layer deltas."""
        return [
            build_layer_delta(
                name,
                trained.adapters[name],
                encoding=DeltaEncoding.SUB1BIT_BINARY,
                sub1bit_rank=sub1bit_rank,
            )
            for name in trained.layer_names
        ]

    def _train(
        self,
        *,
        examples: list[tuple[str | None, str, str]],
        rank: int,
        steps: int,
        lr: float,
        seed: int,
        binary_ste: bool,
    ) -> TrainedAdapters:
        trainer = self.trainer
        layer_names = sorted(trainer.instrumented.keys(), key=_layer_sort_key)
        self._check_layers_exportable(layer_names)
        adapters, losses, zero_diff = trainer.train_qlora_adapters(
            examples,
            layer_names=list(layer_names),
            enable_thinking=False,
            rank=rank,
            scale=1.0,
            binary_ste=binary_ste,
            steps=steps,
            lr=lr,
            seed=seed,
        )
        del trainer
        head = float(np.mean(losses[: min(6, len(losses))]))
        tail = float(np.mean(losses[-min(6, len(losses)) :]))
        return TrainedAdapters(
            layer_names=tuple(layer_names),
            adapters=dict(adapters),
            initial_loss=head,
            final_loss=tail,
            zero_perturbation_max_diff=float(zero_diff),
        )

    # ------------------------------------------------------------------ unit loading
    def _export_unit(self, registry: SkillUnitRegistry, unit_id: str, key: str) -> Path:
        assert self.lora_dir is not None and self.base_gguf is not None
        from subcor.gguf_export import export_unit_to_gguf_lora

        # One sub-directory per base GGUF so a cached adapter is never loaded on another base.
        base_tag = _UNSAFE_KEY_CHARS.sub("_", self.base_gguf.stem)[:64]
        base_tag = f"{base_tag}-{self.base_gguf.stat().st_size:x}"
        out = self.lora_dir / base_tag / f"{key}.gguf"
        if not out.is_file():
            # export_unit_to_gguf_lora verifies the payload digest and writes atomically.
            export_unit_to_gguf_lora(registry, unit_id, out, base_gguf=self.base_gguf)
        return out

    def _adapter_live(self, key: str) -> bool:
        """True while some owner of `key` is a candidate, CONFORMANT or ENABLED unit.

        Keys are payload-digest specific, so a re-proposed unit (new payload) no longer
        owns its old adapter. Unknown keys are conservatively treated as live.
        """
        owners = self._adapter_owners.get(key)
        if not owners:
            return True
        for registry, unit_id in owners:
            try:
                if self.adapter_key(registry, unit_id) != key:
                    continue
                manifest = registry.get_manifest(unit_id)
                if manifest.state in _LIVE_ADAPTER_STATES:
                    return True
                if manifest.state is UnitState.DORMANT and manifest.conformance is None:
                    return True  # proposed, conformance not evaluated yet
            except Exception:
                continue
        return False

    def _retire_dead_adapters(self, keep: set[str] | frozenset[str] = frozenset()) -> None:
        """Drop adapters of rolled-back / failed-conformance units from the next restart.

        They stay loaded in the running llama-server (listed at scale 0 per request and
        unroutable, because their gates are unregistered) until that restart.
        """
        for key in list(self._managed_paths):
            if key not in keep and not self._adapter_live(key):
                del self._managed_paths[key]
                self._adapter_owners.pop(key, None)

    def _restart_supervisor(
        self, wanted: dict[str, Path], fallbacks: Sequence[dict[str, Path]]
    ) -> BaseException | None:
        """Restart llama-server with `wanted`, else each fallback set in turn.

        Updates the managed id/path maps to whatever is actually loaded and returns the
        exception of the `wanted` attempt (None on success). If every attempt fails the
        maps are cleared and the last error is raised.
        """
        assert self.supervisor is not None
        try:
            ids = self.supervisor.restart(wanted)
        except Exception as exc:
            first_error: BaseException = exc
        else:
            self._managed_paths, self._managed_ids = dict(wanted), dict(ids)
            for key in list(self._adapter_owners):
                if key not in wanted:
                    del self._adapter_owners[key]
            return None
        tried = [wanted]
        last_error: BaseException = first_error
        for fallback in fallbacks:
            if fallback in tried:
                continue
            tried.append(fallback)
            try:
                ids = self.supervisor.restart(fallback)
            except Exception as exc:
                last_error = exc
                continue
            self._managed_paths, self._managed_ids = dict(fallback), dict(ids)
            for key in list(self._adapter_owners):
                if key not in fallback:
                    del self._adapter_owners[key]
            return first_error
        self._managed_ids, self._managed_paths = {}, {}
        self._adapter_owners.clear()
        raise last_error

    def _ensure_server(self) -> None:
        """Supervised mode: bring a dead llama-server back before serving a request.

        llama-server can die on its own (segfault, device loss, OOM) or be left stopped by
        a failed restart. It is restarted with the last known-good adapter set, falling
        back to the plain base model, so base-only requests keep working.
        """
        sup = self.supervisor
        if sup is None or self._closed or sup.is_running:
            return
        with self._units_lock:
            if self._closed or sup.is_running:
                return
            self._retire_dead_adapters()
            wanted = dict(self._managed_paths)
            warnings.warn(
                f"Supervised llama-server is not running; restarting it with "
                f"{len(wanted)} adapter(s).",
                RuntimeWarning,
                stacklevel=3,
            )
            error = self._restart_supervisor(wanted, fallbacks=[{}])
            if error is not None:
                warnings.warn(
                    f"llama-server could not reload its adapters ({error}); serving the base "
                    "model only until the next unit load.",
                    RuntimeWarning,
                    stacklevel=3,
                )

    def prepare_units(self, units: Sequence[tuple[SkillUnitRegistry, str]]) -> None:
        """Make sure every (registry, unit_id) has its GGUF LoRA loaded in llama-server.

        Called by TenantManager (under its kernel lock) before sandbox conformance
        evaluation, before promotion, and when hydrating ENABLED units. Missing adapters
        are exported via `subcor.gguf_export` and llama-server is restarted once with the
        loaded adapters that are still live plus the new ones (adapters of rolled-back
        units and of units that failed conformance are left out). If that restart fails,
        serving is brought back with the previous set, or with no adapters, and the error
        is re-raised. No-op in external (unsupervised) mode.
        """
        if self.supervisor is None or not units:
            return
        if self.base_gguf is None or self.lora_dir is None:
            raise RuntimeError("Supervised llama.cpp backend needs base_gguf and lora_dir")
        with self._units_lock:
            requested: dict[str, tuple[SkillUnitRegistry, str]] = {}
            for registry, unit_id in units:
                key = self.adapter_key(registry, unit_id)
                requested.setdefault(key, (registry, unit_id))
                owners = self._adapter_owners.setdefault(key, [])
                if not any(r is registry and u == unit_id for r, u in owners):
                    owners.append((registry, unit_id))
            self._retire_dead_adapters(keep=set(requested))
            new_paths: dict[str, Path] = {}
            for key, (registry, unit_id) in requested.items():
                if key in self._managed_paths and key in self._managed_ids:
                    continue
                path = self._export_unit(registry, unit_id, key)
                if key in self._managed_ids:
                    # Retired earlier but still loaded in the running server: re-adopt it.
                    self._managed_paths[key] = path
                else:
                    new_paths[key] = path
            if not new_paths and self.supervisor.is_running:
                return
            previous = dict(self._managed_paths)
            error = self._restart_supervisor({**previous, **new_paths}, fallbacks=[previous, {}])
            if error is not None:
                raise error

    def invalidate_unit(self, unit_id: str) -> None:
        """Forget a rolled-back unit's adapter.

        Static map: the entry is removed. Supervised adapters: the unit's adapter stays
        loaded at scale 0 (unroutable once its gate is unregistered) and is left out of
        the next llama-server restart.
        """
        self._lora_id_map.pop(unit_id, None)
        with self._units_lock:
            self._retire_dead_adapters()

    def close(self) -> None:
        """Stop a supervised llama-server and drop the trainer."""
        self._closed = True
        if self.supervisor is not None:
            self.supervisor.stop()
        self._trainer = None
