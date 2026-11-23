"""Check that SUBCOR's trainer sees EXACTLY the token ids llama-server serves.

For every assistant turn of every conversation this compares, against a running
``llama-server --jinja`` (same GGUF):

1. prompt text: ``torch_kernel.assistant_turns`` prompt == ``POST /apply-template``
   for ``messages[:k]`` (``enable_thinking=false``, the SUBCOR serving default);
2. prompt ids: trainer tokenizer ids == ``POST /tokenize`` of the server prompt
   (``add_special=true, parse_special=true``, i.e. how the server tokenizes prompts);
3. target ids: trainer ids of the assistant target == ``/tokenize`` of the same text;
4. target serialisation: the server's own rendering of the conversation with turn
   ``k`` in its history starts with ``prompt_k + target_k`` (text and ids), so the
   trained tool-call text is the template's (= the model's native) format;
5. training sequences: for every labelled span of every sequence produced by
   ``encode_chat_training_example`` the ids before the span == the server's prompt
   ids for that turn, and the span == the target ids.

Usage (CPU server, no GPU):

    llama-server -m Qwen3.5-0.8B-Q4_K_M.gguf --port 8503 -ngl 0 --device none \\
        --jinja --embeddings --pooling mean
    uv run --extra dev --extra rocm python experiments/bfcl/check_template_parity.py \\
        --gguf /path/Qwen3.5-0.8B-Q4_K_M.gguf --server http://127.0.0.1:8503 \\
        [--jsonl more.jsonl] [--dump parity.jsonl] [--out result.json]

Exit status 0 iff every check passes.
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from pathlib import Path
from typing import Any

from subcor.backends.torch_kernel import (
    IGNORE_INDEX,
    _resolve_gguf_source,
    assistant_turns,
    encode_chat_training_example,
    encode_prompt_text,
)


# --------------------------------------------------------------------------------------
# Built-in tool conversations (BFCL-like shapes; synthetic, no BFCL data)
# --------------------------------------------------------------------------------------
def _fn(name: str, desc: str, props: dict[str, Any], required: list[str]) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": desc,
            "parameters": {"type": "object", "properties": props, "required": required},
        },
    }


def _call(name: str, args: Any, cid: str) -> dict[str, Any]:
    arguments = args if isinstance(args, str) else json.dumps(args, ensure_ascii=False)
    return {"id": cid, "type": "function", "function": {"name": name, "arguments": arguments}}


WEATHER = _fn(
    "get_weather",
    "Get the current weather for a city.",
    {
        "city": {"type": "string", "description": "City name"},
        "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
    },
    ["city"],
)
CALC = _fn(
    "calculate_area",
    "Area of a shape.",
    {
        "shape": {"type": "string"},
        "dimensions": {"type": "array", "items": {"type": "number"}},
        "precise": {"type": "boolean"},
    },
    ["shape", "dimensions"],
)
BOOK = _fn(
    "book_flight",
    "Book a flight.",
    {
        "origin": {"type": "string"},
        "destination": {"type": "string"},
        "date": {"type": "string", "description": "YYYY-MM-DD"},
        "passengers": {"type": "integer"},
        "options": {"type": "object", "properties": {"seat": {"type": "string"}}},
    },
    ["origin", "destination", "date"],
)
SEARCH = _fn("search_files", "Find files.", {"pattern": {"type": "string"}}, ["pattern"])
NOARGS = _fn("get_time", "Current UTC time.", {}, [])
MATH = _fn(
    "math_factorial", "Factorial of n.", {"n": {"type": "integer", "description": "n >= 0"}}, ["n"]
)


def parity_conversations() -> list[dict[str, Any]]:
    """~20 tool conversations covering the formatting edge cases that matter."""
    convs: list[dict[str, Any]] = []

    def add(name: str, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None) -> None:
        entry: dict[str, Any] = {"id": name, "messages": messages}
        if tools is not None:
            entry["tools"] = tools
        convs.append(entry)

    u = "user"
    add(
        "single_call",
        [
            {"role": u, "content": "What's the weather in Paris?"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [_call("get_weather", {"city": "Paris"}, "c1")],
            },
        ],
        [WEATHER],
    )
    add(
        "single_call_system",
        [
            {"role": "system", "content": "You are a travel assistant. Be concise."},
            {"role": u, "content": "Weather in Tokyo in fahrenheit please"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [_call("get_weather", {"city": "Tokyo", "unit": "fahrenheit"}, "c1")],
            },
        ],
        [WEATHER, CALC],
    )
    add(
        "parallel_calls",
        [
            {"role": u, "content": "Weather in Berlin and Madrid?"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    _call("get_weather", {"city": "Berlin"}, "c1"),
                    _call("get_weather", {"city": "Madrid"}, "c2"),
                ],
            },
        ],
        [WEATHER],
    )
    add(
        "array_bool_float",
        [
            {"role": u, "content": "Area of a rectangle 3.5 by 2, precise."},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    _call(
                        "calculate_area",
                        {"shape": "rectangle", "dimensions": [3.5, 2], "precise": True},
                        "c1",
                    )
                ],
            },
        ],
        [CALC],
    )
    add(
        "nested_object_int",
        [
            {"role": u, "content": "Book 2 seats SFO to JFK on 2026-12-01, window."},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    _call(
                        "book_flight",
                        {
                            "origin": "SFO",
                            "destination": "JFK",
                            "date": "2026-12-01",
                            "passengers": 2,
                            "options": {"seat": "window"},
                        },
                        "c1",
                    )
                ],
            },
        ],
        [BOOK],
    )
    add(
        "unicode_args",
        [
            {"role": u, "content": "Météo à São Paulo et Zürich ?"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    _call("get_weather", {"city": "São Paulo"}, "c1"),
                    _call("get_weather", {"city": "Zürich"}, "c2"),
                ],
            },
        ],
        [WEATHER],
    )
    add(
        "content_then_call",
        [
            {"role": u, "content": "Find all python files."},
            {
                "role": "assistant",
                "content": "I'll search for them.",
                "tool_calls": [_call("search_files", {"pattern": "**/*.py"}, "c1")],
            },
        ],
        [SEARCH],
    )
    add(
        "empty_args",
        [
            {"role": u, "content": "What time is it?"},
            {"role": "assistant", "content": None, "tool_calls": [_call("get_time", {}, "c1")]},
        ],
        [NOARGS, WEATHER],
    )
    add(
        "string_with_newlines_quotes",
        [
            {"role": u, "content": 'Search for files named "a\nb"'},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [_call("search_files", {"pattern": 'line1\nline2 "q" \\ end'}, "c1")],
            },
        ],
        [SEARCH],
    )
    add(
        "tool_result_then_answer",
        [
            {"role": u, "content": "Weather in Oslo?"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [_call("get_weather", {"city": "Oslo"}, "c1")],
            },
            {"role": "tool", "tool_call_id": "c1", "content": '{"temp_c": -3, "sky": "snow"}'},
            {"role": "assistant", "content": "It is -3°C and snowing in Oslo."},
        ],
        [WEATHER],
    )
    add(
        "multi_step_chain",
        [
            {"role": u, "content": "Compute 5! then the area of a square with that side."},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [_call("math_factorial", {"n": 5}, "c1")],
            },
            {"role": "tool", "tool_call_id": "c1", "content": "120"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    _call("calculate_area", {"shape": "square", "dimensions": [120]}, "c2")
                ],
            },
            {"role": "tool", "tool_call_id": "c2", "content": "14400"},
            {"role": "assistant", "content": "5! = 120 and the square's area is 14400."},
        ],
        [MATH, CALC],
    )
    add(
        "parallel_results",
        [
            {"role": u, "content": "Weather in Rome and Lima?"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    _call("get_weather", {"city": "Rome"}, "c1"),
                    _call("get_weather", {"city": "Lima"}, "c2"),
                ],
            },
            {"role": "tool", "tool_call_id": "c1", "content": "sunny 24C"},
            {"role": "tool", "tool_call_id": "c2", "content": "cloudy 18C"},
            {"role": "assistant", "content": "Rome: sunny, 24C. Lima: cloudy, 18C."},
        ],
        [WEATHER],
    )
    add(
        "two_user_queries",
        [
            {"role": u, "content": "Weather in Cairo?"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [_call("get_weather", {"city": "Cairo"}, "c1")],
            },
            {"role": "tool", "tool_call_id": "c1", "content": "hot"},
            {"role": "assistant", "content": "It's hot in Cairo."},
            {"role": u, "content": "And in Nairobi, in fahrenheit?"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    _call("get_weather", {"city": "Nairobi", "unit": "fahrenheit"}, "c2")
                ],
            },
        ],
        [WEATHER],
    )
    add(
        "three_user_queries_multi_turn",
        [
            {"role": "system", "content": "You manage flights."},
            {"role": u, "content": "Book LAX to SEA on 2026-11-30."},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    _call(
                        "book_flight",
                        {"origin": "LAX", "destination": "SEA", "date": "2026-11-30"},
                        "c1",
                    )
                ],
            },
            {"role": "tool", "tool_call_id": "c1", "content": '{"status": "booked", "id": "F1"}'},
            {"role": "assistant", "content": "Booked, confirmation F1."},
            {"role": u, "content": "Thanks! What's the weather in Seattle?"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [_call("get_weather", {"city": "Seattle"}, "c2")],
            },
            {"role": "tool", "tool_call_id": "c2", "content": "rain"},
            {"role": "assistant", "content": "Rainy in Seattle."},
            {"role": u, "content": "Ok bye"},
            {"role": "assistant", "content": "Goodbye!"},
        ],
        [BOOK, WEATHER],
    )
    add(
        "irrelevance_no_call",
        [
            {"role": u, "content": "Tell me a joke about cats."},
            {
                "role": "assistant",
                "content": "Why did the cat sit on the computer? To keep an eye on the mouse.",
            },
        ],
        [WEATHER, CALC],
    )
    add(
        "no_tools_plain_chat",
        [
            {"role": "system", "content": "You are helpful."},
            {"role": u, "content": "Say hi."},
            {"role": "assistant", "content": "Hi!"},
            {"role": u, "content": "Again."},
            {"role": "assistant", "content": "Hi again!"},
        ],
        None,
    )
    add(
        "null_and_numbers",
        [
            {"role": u, "content": "Book 1 passenger NYC to BOS 2027-01-05, no seat pref."},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    _call(
                        "book_flight",
                        {
                            "origin": "NYC",
                            "destination": "BOS",
                            "date": "2027-01-05",
                            "passengers": 1,
                            "options": None,
                        },
                        "c1",
                    )
                ],
            },
        ],
        [BOOK],
    )
    add(
        "large_and_negative_numbers",
        [
            {"role": u, "content": "Area for shape 'line' with dims -1e-3 and 12345678901."},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    _call(
                        "calculate_area",
                        {"shape": "line", "dimensions": [-0.001, 12345678901], "precise": False},
                        "c1",
                    )
                ],
            },
        ],
        [CALC],
    )
    add(
        "whitespace_content",
        [
            {"role": u, "content": "  padded question?  \n"},
            {"role": "assistant", "content": "  ", "tool_calls": [_call("get_time", {}, "c1")]},
            {"role": "tool", "tool_call_id": "c1", "content": "  12:00Z \n"},
            {"role": "assistant", "content": "It's 12:00 UTC."},
        ],
        [NOARGS],
    )
    add(
        "many_tools_schema",
        [
            {"role": u, "content": "What's 7 factorial?"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [_call("math_factorial", {"n": 7}, "c1")],
            },
            {"role": "tool", "tool_call_id": "c1", "content": "5040"},
            {"role": "assistant", "content": "7! = 5040."},
        ],
        [WEATHER, CALC, BOOK, SEARCH, NOARGS, MATH],
    )
    add(
        "code_in_args",
        [
            {"role": u, "content": "Search for `def main(`"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    _call("search_files", {"pattern": "def main(<args>): {x: [1]}"}, "c1")
                ],
            },
        ],
        [SEARCH],
    )
    return convs


# --------------------------------------------------------------------------------------
# llama-server helpers
# --------------------------------------------------------------------------------------
def _post(server: str, path: str, payload: dict[str, Any]) -> Any:
    req = urllib.request.Request(
        f"{server.rstrip('/')}{path}",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        return json.loads(resp.read().decode("utf-8"))


def server_prompt(server: str, messages: list[dict[str, Any]], tools: Any) -> str:
    body: dict[str, Any] = {
        "messages": messages,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    if tools:
        body["tools"] = tools
    return str(_post(server, "/apply-template", body)["prompt"])


def server_tokens(server: str, text: str, *, add_special: bool) -> list[int]:
    out = _post(
        server, "/tokenize", {"content": text, "add_special": add_special, "parse_special": True}
    )
    return [int(t) for t in out["tokens"]]


def _first_diff(a: Any, b: Any) -> int:
    for i, (x, y) in enumerate(zip(a, b, strict=False)):
        if x != y:
            return i
    return min(len(a), len(b))


def check_conversation(tok: Any, server: str, conv: dict[str, Any]) -> dict[str, Any]:
    msgs, tools = conv["messages"], conv.get("tools")
    turns = assistant_turns(tok, msgs, tools=tools)
    failures: list[str] = []
    srv_prompt_ids: dict[int, list[int]] = {}
    tgt_ids: dict[int, list[int]] = {}
    for t in turns:
        k = t.message_index
        sp = server_prompt(server, msgs[:k], tools)
        if sp != t.prompt:
            i = _first_diff(sp, t.prompt)
            failures.append(
                f"turn {k}: prompt text differs at char {i}: server={sp[i - 40 : i + 40]!r} "
                f"ours={t.prompt[i - 40 : i + 40]!r}"
            )
        sids = server_tokens(server, sp, add_special=True)
        ours = encode_prompt_text(tok, t.prompt)
        srv_prompt_ids[k] = sids
        if sids != ours:
            failures.append(f"turn {k}: prompt ids differ at {_first_diff(sids, ours)}")
        t_srv = server_tokens(server, t.target, add_special=False)
        t_ours = [int(x) for x in tok.encode(t.target, add_special_tokens=False)]
        tgt_ids[k] = t_ours
        if t_srv != t_ours:
            failures.append(f"turn {k}: target ids differ at {_first_diff(t_srv, t_ours)}")
        # The server's own history rendering of turn k (a tool message keeps the latest
        # user query, so turn k keeps its think block, exactly as when it was generated).
        hist = server_prompt(
            server, [*msgs[: k + 1], {"role": "tool", "tool_call_id": "x", "content": "ok"}], tools
        )
        if not hist.startswith(t.prompt + t.target):
            i = _first_diff(hist, t.prompt + t.target)
            failures.append(
                f"turn {k}: server history does not contain target verbatim at char {i}: "
                f"{hist[i - 40 : i + 40]!r}"
            )
        hist_ids = server_tokens(server, hist, add_special=True)
        if hist_ids[: len(ours) + len(t_ours)] != ours + t_ours:
            failures.append(f"turn {k}: server history ids != prompt ids + target ids")

    # Training sequences: every labelled span sits on the exact serving prompt ids.
    seqs = encode_chat_training_example(tok, msgs, tools=tools)
    trained = [t for t in turns if t.trained]
    spans: list[tuple[int, int, int]] = []  # (seq index, start, end)
    for si, seq in enumerate(seqs):
        lab = seq.labels
        i = 0
        while i < len(lab):
            if lab[i] != IGNORE_INDEX:
                j = i
                while j < len(lab) and lab[j] != IGNORE_INDEX:
                    j += 1
                spans.append((si, i, j))
                i = j
            else:
                i += 1
    if len(spans) != len(trained):
        failures.append(f"{len(spans)} labelled spans for {len(trained)} trained turns")
    for (si, start, end), t in zip(spans, trained, strict=False):
        seq = seqs[si]
        if list(seq.input_ids[:start]) != srv_prompt_ids[t.message_index]:
            failures.append(f"turn {t.message_index}: sequence context != server prompt ids")
        if list(seq.labels[start:end]) != tgt_ids[t.message_index] + (
            [] if tgt_ids[t.message_index][-1:] == [tok.eos_token_id] else [tok.eos_token_id]
        ):
            failures.append(f"turn {t.message_index}: labelled span != target ids")
    return {
        "id": conv.get("id"),
        "turns": len(turns),
        "sequences": len(seqs),
        "prompt_tokens_max": max((len(v) for v in srv_prompt_ids.values()), default=0),
        "ok": not failures,
        "failures": failures,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--gguf", required=True, help="Base GGUF (the one llama-server serves)")
    ap.add_argument("--server", default="http://127.0.0.1:8503")
    ap.add_argument("--jsonl", action="append", default=[], help="Extra conversations (JSONL)")
    ap.add_argument("--no-builtin", action="store_true", help="Skip the built-in conversations")
    ap.add_argument("--dump", type=Path, help="Write the built-in conversations as JSONL here")
    ap.add_argument("--out", type=Path, help="Write a JSON result summary here")
    args = ap.parse_args(argv)

    from transformers import AutoTokenizer

    source, name = _resolve_gguf_source(args.gguf, args.gguf)
    tok = AutoTokenizer.from_pretrained(source, gguf_file=name)

    convs = [] if args.no_builtin else parity_conversations()
    if args.dump:
        args.dump.parent.mkdir(parents=True, exist_ok=True)
        with args.dump.open("w", encoding="utf-8") as fh:
            for c in parity_conversations():
                fh.write(json.dumps(c, ensure_ascii=False) + "\n")
    for path in args.jsonl:
        with open(path, encoding="utf-8") as fh:
            convs.extend(json.loads(line) for line in fh if line.strip())

    results = [check_conversation(tok, args.server, c) for c in convs]
    n_ok = sum(r["ok"] for r in results)
    n_turns = sum(r["turns"] for r in results)
    for r in results:
        status = "OK  " if r["ok"] else "FAIL"
        print(
            f"{status} {r['id']!s:<32} turns={r['turns']} seqs={r['sequences']} "
            f"prompt_tok<={r['prompt_tokens_max']}"
        )
        for f in r["failures"]:
            print(f"     - {f}")
    summary = {
        "conversations": len(results),
        "conversations_ok": n_ok,
        "assistant_turns": n_turns,
        "all_ok": n_ok == len(results),
        "server": args.server,
        "gguf": str(args.gguf),
        "results": results,
    }
    print(
        f"\n{n_ok}/{len(results)} conversations ({n_turns} assistant turns) token-identical "
        "to llama-server"
    )
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return 0 if summary["all_ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
