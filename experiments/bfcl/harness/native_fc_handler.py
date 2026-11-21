"""Native function-calling BFCL handler for OpenAI-compatible local servers.

`NativeFCHandler` subclasses bfcl-eval's own ``OpenAICompletionsHandler`` (the
handler BFCL uses for OpenAI-compatible ``/v1/chat/completions`` FC endpoints) and
changes only what a local endpoint needs. Everything else is inherited unchanged:
the BFCL ``BaseHandler`` single-turn / multi-turn inference loops, tool
compilation (``convert_to_tool(..., GORILLA_TO_OPENAPI, OPENAI_COMPLETIONS)``),
``decode_ast`` / ``decode_execute``, how the first and following user turns are
added, and how execution results go back as ``role="tool"`` messages carrying
the ``tool_call_id``.

What changes (all listed in README.md, "Handler deviations"):

* endpoint, API key and served ``model`` come from the run config, not from
  ``OPENAI_*`` env vars, so one run can target llama-server directly or SUBCOR
  serve (``subcor/base`` or ``subcor/<tenant>``);
* ``max_tokens`` is sent (default 4096, the cap bfcl's OSS handler uses); the
  official OpenAI handler sends none, and an unbounded local server can loop;
* thinking is chosen explicitly: ``chat_template_kwargs.enable_thinking`` for
  llama-server, the top-level ``enable_thinking`` field for SUBCOR serve;
* the assistant message put back into the history is a plain dict rebuilt from
  the response, not the SDK object. ``reasoning_content`` is kept when present
  (as bfcl's local ``QwenFCHandler`` does) unless ``reasoning_history="strip"``;
* tool calls with no ``id`` get a synthetic one, so the ``tool`` replies can be
  matched to their calls;
* optional ``system_prefix`` (the A1 few-shot arm only): prepended as the first system
  message of each item, or merged ahead of the item's own system message;
* each item's result also stores the routing metadata SUBCOR returns
  (``subcor`` field and ``X-SUBCOR-*`` headers) plus the finish reasons, under
  ``harness_trace``. The graders never read this field.
"""

from __future__ import annotations

import json
import os
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any

from bfcl_eval.constants.enums import ModelStyle
from bfcl_eval.model_handler.api_inference.openai_completion import (
    OpenAICompletionsHandler,
)
from overrides import override


@dataclass
class EndpointConfig:
    """How one registry name reaches its server (serialized into run_config.json)."""

    base_url: str  # e.g. http://127.0.0.1:8501/v1
    served_model: (
        str  # the "model" field sent: anything for llama-server, subcor/<tenant> for SUBCOR
    )
    backend: str = "llama-server"  # "llama-server" | "subcor"
    api_key_env: str | None = None  # env var holding the bearer key (never stored on disk)
    max_tokens: int | None = 4096
    # None -> leave the chat template default; True/False -> send it explicitly.
    enable_thinking: bool | None = None
    reasoning_history: str = "keep"  # "keep" | "strip"
    parallel_tool_calls: bool | None = None  # None -> do not send (OpenAI handler default)
    request_timeout_s: float = 1800.0
    extra_body: dict[str, Any] = field(default_factory=dict)
    # A1 arm: text prepended as the first system message of every item (merged into the
    # item's own system message when it has one). None for every other arm.
    system_prefix: str | None = None
    system_prefix_sha256: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def api_key(self) -> str:
        return (os.getenv(self.api_key_env) if self.api_key_env else None) or "EMPTY"


# registry_name -> EndpointConfig. Filled by run_bfcl.register_run() before bfcl
# builds any handler (bfcl builds handlers itself from MODEL_CONFIG_MAPPING and only
# passes model_name / temperature / registry_name / is_fc_model).
ENDPOINTS: dict[str, EndpointConfig] = {}

_trace_local = threading.local()


class NativeFCHandler(OpenAICompletionsHandler):
    def __init__(self, model_name, temperature, registry_name, is_fc_model, **kwargs) -> None:
        cfg = ENDPOINTS.get(registry_name)
        if cfg is None:
            raise KeyError(
                f"No endpoint registered for '{registry_name}'. Use run_bfcl.py, which registers it."
            )
        self.endpoint = cfg
        # The parent constructor builds its OpenAI client from _build_client_kwargs()
        # (overridden below), so OPENAI_* env vars cannot redirect a run.
        super().__init__(model_name, temperature, registry_name, is_fc_model, **kwargs)
        self.model_style = ModelStyle.OPENAI_COMPLETIONS

    @override
    def _build_client_kwargs(self):
        return {
            "base_url": self.endpoint.base_url,
            "api_key": self.endpoint.api_key(),
            "timeout": self.endpoint.request_timeout_s,
        }

    # ------------------------------------------------------------------ tracing
    @override
    def inference(self, test_entry: dict, include_input_log: bool, exclude_state_log: bool):
        _trace_local.steps = []
        result, metadata = super().inference(test_entry, include_input_log, exclude_state_log)
        steps = getattr(_trace_local, "steps", [])
        if steps:
            metadata["harness_trace"] = steps
        _trace_local.steps = []
        return result, metadata

    def _record(self, item: dict) -> None:
        steps = getattr(_trace_local, "steps", None)
        if steps is not None:
            steps.append(item)

    # ------------------------------------------------------------------ query
    @override
    def _query_FC(self, inference_data: dict):
        message: list[dict] = inference_data["message"]
        tools = inference_data["tools"]
        inference_data["inference_input_log"] = {"message": repr(message), "tools": tools}

        # Same fields as the official OpenAICompletionsHandler._query_FC ...
        kwargs: dict[str, Any] = {
            "messages": message,
            "model": self.endpoint.served_model,
            "temperature": self.temperature,
            "store": False,
        }
        if len(tools) > 0:
            kwargs["tools"] = tools
        # ... plus the local-endpoint fields described in the module docstring.
        cfg = self.endpoint
        if cfg.max_tokens is not None:
            kwargs["max_tokens"] = int(cfg.max_tokens)
        if cfg.parallel_tool_calls is not None and len(tools) > 0:
            kwargs["parallel_tool_calls"] = bool(cfg.parallel_tool_calls)
        extra_body: dict[str, Any] = dict(cfg.extra_body)
        if cfg.enable_thinking is not None:
            if cfg.backend == "subcor":
                extra_body["enable_thinking"] = bool(cfg.enable_thinking)
            else:
                ctk = dict(extra_body.get("chat_template_kwargs") or {})
                ctk["enable_thinking"] = bool(cfg.enable_thinking)
                extra_body["chat_template_kwargs"] = ctk
        if extra_body:
            kwargs["extra_body"] = extra_body

        return self.generate_with_backoff(**kwargs)

    @override
    def generate_with_backoff(self, **kwargs):
        # Same timing as the parent. with_raw_response keeps the HTTP headers, which
        # is where SUBCOR reports the routed units.
        start_time = time.time()
        raw = self.client.chat.completions.with_raw_response.create(**kwargs)
        api_response = raw.parse()
        end_time = time.time()
        try:
            headers = {k: v for k, v in raw.headers.items() if k.lower().startswith("x-subcor-")}
        except Exception:
            headers = {}
        api_response._harness_headers = headers  # type: ignore[attr-defined]
        return api_response, end_time - start_time

    # ------------------------------------------------------------------ parse
    @override
    def _parse_query_response_FC(self, api_response: Any) -> dict:
        choice = api_response.choices[0]
        message = choice.message
        raw_tool_calls = getattr(message, "tool_calls", None)

        # Identical branching to the official parser: tool_calls -> list of
        # {name: arguments_json_string}; otherwise the text content.
        tool_calls_for_history: list[dict] = []
        try:
            model_responses = [
                {func_call.function.name: func_call.function.arguments}
                for func_call in raw_tool_calls
            ]
            tool_call_ids = []
            for func_call in raw_tool_calls:
                call_id = func_call.id or f"call_{uuid.uuid4().hex[:24]}"
                tool_call_ids.append(call_id)
                tool_calls_for_history.append(
                    {
                        "id": call_id,
                        "type": "function",
                        "function": {
                            "name": func_call.function.name,
                            "arguments": func_call.function.arguments,
                        },
                    }
                )
        except Exception:
            model_responses = message.content
            tool_call_ids = []

        extra = getattr(message, "model_extra", None) or {}
        reasoning_content = getattr(message, "reasoning_content", None) or extra.get(
            "reasoning_content"
        )

        history_msg: dict[str, Any] = {"role": "assistant", "content": message.content}
        if tool_calls_for_history:
            history_msg["tool_calls"] = tool_calls_for_history
        if reasoning_content and self.endpoint.reasoning_history == "keep":
            history_msg["reasoning_content"] = reasoning_content

        usage = getattr(api_response, "usage", None)
        response_data = {
            "model_responses": model_responses,
            "model_responses_message_for_chat_history": history_msg,
            "tool_call_ids": tool_call_ids,
            "input_token": getattr(usage, "prompt_tokens", 0) if usage else 0,
            "output_token": getattr(usage, "completion_tokens", 0) if usage else 0,
        }
        if reasoning_content:
            response_data["reasoning_content"] = reasoning_content

        resp_extra = getattr(api_response, "model_extra", None) or {}
        trace: dict[str, Any] = {"finish_reason": choice.finish_reason}
        if "subcor" in resp_extra:
            trace["subcor"] = resp_extra["subcor"]
        headers = getattr(api_response, "_harness_headers", None)
        if headers:
            trace["headers"] = headers
        self._record(trace)
        return response_data

    @override
    def add_first_turn_message_FC(
        self, inference_data: dict, first_turn_message: list[dict]
    ) -> dict:
        inference_data = super().add_first_turn_message_FC(inference_data, first_turn_message)
        prefix = self.endpoint.system_prefix
        if prefix:
            msgs: list[dict] = inference_data["message"]
            if msgs and msgs[0].get("role") == "system":
                # One system message: the prefix, a blank line, then the item's own content.
                # A new dict, so bfcl's test_entry is never mutated.
                own = str(msgs[0].get("content") or "")
                msgs[0] = {**msgs[0], "content": prefix.rstrip("\n") + "\n\n" + own}
            else:
                msgs.insert(0, {"role": "system", "content": prefix})
        return inference_data

    @override
    def _add_assistant_message_FC(self, inference_data: dict, model_response_data: dict) -> dict:
        inference_data["message"].append(
            model_response_data["model_responses_message_for_chat_history"]
        )
        return inference_data


def endpoint_from_dict(d: dict[str, Any]) -> EndpointConfig:
    d = dict(d)
    return EndpointConfig(**d)


def dumps_endpoint(cfg: EndpointConfig) -> str:
    return json.dumps(cfg.to_dict(), indent=2, sort_keys=True)
