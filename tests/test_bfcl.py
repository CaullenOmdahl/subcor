"""Unit tests for the official BFCL v3 AST grader and Qwen3.5 XML tool-call formatter."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from subcor.bfcl import (
    BFCLCase,
    _gt_entry_to_call_dict,
    bfcl_functions_to_openai_tools,
    collect_bfcl_onpolicy_self_distill_examples,
    format_qwen35_xml_tool_calls,
    grade_bfcl_case,
    load_bfcl_disjoint_onpolicy_examples,
)
from subcor.confidence import parse_tool_calls


def _tc(name: str, args: dict) -> dict:
    return {
        "id": "call_1",
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(args)},
    }


def test_bfcl_simple_optional_args_and_strict_types() -> None:
    case = BFCLCase(
        case_id="simple_0",
        category="simple",
        messages=({"role": "user", "content": "Area of triangle base 10 height 5"},),
        tools=(),
        ground_truth=(
            {"calculate_triangle_area": {"base": [10], "height": [5], "unit": ["units", ""]}},
        ),
    )
    # Exact match with optional unit omitted or included passes.
    assert grade_bfcl_case(case, [_tc("calculate_triangle_area", {"base": 10, "height": 5})])
    assert grade_bfcl_case(
        case, [_tc("calculate_triangle_area", {"base": 10.0, "height": 5, "unit": "Units"})]
    )
    # Hallucinated extra arg fails.
    assert not grade_bfcl_case(
        case, [_tc("calculate_triangle_area", {"base": 10, "height": 5, "color": "red"})]
    )
    # Wrong value fails.
    assert not grade_bfcl_case(case, [_tc("calculate_triangle_area", {"base": 11, "height": 5})])
    # Bool True must NOT match int 1.
    bool_case = BFCLCase(
        case_id="simple_b",
        category="simple",
        messages=(),
        tools=(),
        ground_truth=({"fn": {"flag": [True], "count": [1]}},),
    )
    assert grade_bfcl_case(bool_case, [_tc("fn", {"flag": True, "count": 1})])
    assert not grade_bfcl_case(bool_case, [_tc("fn", {"flag": 1, "count": 1})])
    assert not grade_bfcl_case(bool_case, [_tc("fn", {"flag": True, "count": True})])


def test_bfcl_parallel_order_independent_and_irrelevance() -> None:
    par = BFCLCase(
        case_id="parallel_0",
        category="parallel",
        messages=(),
        tools=(),
        ground_truth=(
            {"spotify.play": {"artist": ["Taylor Swift"], "duration": [20]}},
            {"spotify.play": {"artist": ["Maroon 5"], "duration": [15]}},
        ),
    )
    # Reversed order passes (bipartite match), dot/underscore normalized passes.
    assert grade_bfcl_case(
        par,
        [
            _tc("spotify_play", {"artist": "Maroon 5", "duration": 15}),
            _tc("spotify.play", {"artist": "Taylor Swift", "duration": 20}),
        ],
    )
    # Missing one call fails.
    assert not grade_bfcl_case(
        par, [_tc("spotify.play", {"artist": "Taylor Swift", "duration": 20})]
    )

    # Irrelevance passes ONLY when zero tool calls are emitted.
    irr = BFCLCase(case_id="irr_0", category="irrelevance", messages=(), tools=(), ground_truth=())
    assert grade_bfcl_case(irr, [])
    assert not grade_bfcl_case(irr, [_tc("determine_bmi", {"weight": 70})])


def test_bfcl_schema_normalization_and_xml_roundtrip() -> None:
    tools = bfcl_functions_to_openai_tools(
        [
            {
                "name": "calc",
                "description": "d",
                "parameters": {
                    "type": "dict",
                    "properties": {"x": {"type": "float"}, "items": {"type": "tuple"}},
                },
            }
        ]
    )
    assert tools[0]["function"]["parameters"]["type"] == "object"
    assert tools[0]["function"]["parameters"]["properties"]["x"]["type"] == "number"
    assert tools[0]["function"]["parameters"]["properties"]["items"]["type"] == "array"

    calls = [_tc("spotify.play", {"artist": "Taylor Swift", "duration": 20, "shuffle": True})]
    xml = format_qwen35_xml_tool_calls(calls)
    clean, parsed = parse_tool_calls(xml)
    assert clean == ""
    assert len(parsed) == 1
    assert parsed[0]["function"]["name"] == "spotify.play"
    assert json.loads(parsed[0]["function"]["arguments"]) == {
        "artist": "Taylor Swift",
        "duration": 20,
        "shuffle": True,
    }


def test_gt_entry_to_call_dict_handles_optional_and_falsy_values() -> None:
    entry = {
        "math.solve": {
            "x": [0, 1],
            "strict": [False],
            "note": ["", "optional"],
            "unused": [""],
        }
    }
    call = _gt_entry_to_call_dict(entry)
    assert call == {
        "function": {
            "name": "math.solve",
            "arguments": {"x": 0, "strict": False},
        }
    }
    xml = format_qwen35_xml_tool_calls([call])
    _, parsed = parse_tool_calls(xml)
    case = BFCLCase(
        case_id="simple_gt",
        category="simple",
        messages=(),
        tools=(),
        ground_truth=(entry,),
    )
    assert grade_bfcl_case(case, parsed)


def test_load_bfcl_disjoint_onpolicy_examples_zero_leakage(tmp_path: Path, monkeypatch) -> None:
    import huggingface_hub

    q_simple = tmp_path / "BFCL_v3_simple.json"
    a_simple = tmp_path / "ans_simple.json"
    q_irr = tmp_path / "BFCL_v3_irrelevance.json"

    q_simple.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "id": "simple_0",
                        "question": [[{"role": "user", "content": "Held-out test prompt"}]],
                        "function": [
                            {
                                "name": "f",
                                "description": "d",
                                "parameters": {"type": "dict", "properties": {}},
                            }
                        ],
                    }
                ),
                json.dumps(
                    {
                        "id": "simple_1",
                        "question": [[{"role": "user", "content": "HELD-OUT TEST PROMPT"}]],
                        "function": [
                            {
                                "name": "f",
                                "description": "d",
                                "parameters": {"type": "dict", "properties": {}},
                            }
                        ],
                    }
                ),
                json.dumps(
                    {
                        "id": "simple_2",
                        "question": [[{"role": "user", "content": "Disjoint train prompt"}]],
                        "function": [
                            {
                                "name": "f",
                                "description": "d",
                                "parameters": {
                                    "type": "dict",
                                    "properties": {"a": {"type": "integer"}},
                                },
                            }
                        ],
                    }
                ),
            ]
        ),
        encoding="utf-8",
    )
    a_simple.write_text(
        "\n".join(
            [
                json.dumps({"id": "simple_0", "ground_truth": [{"f": {}}]}),
                json.dumps({"id": "simple_1", "ground_truth": [{"f": {}}]}),
                json.dumps({"id": "simple_2", "ground_truth": [{"f": {"a": [7]}}]}),
            ]
        ),
        encoding="utf-8",
    )
    q_irr.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "id": "irr_0",
                        "question": [[{"role": "user", "content": "Held-out irrelevance"}]],
                        "function": [
                            {
                                "name": "g",
                                "description": "d",
                                "parameters": {"type": "dict", "properties": {}},
                            }
                        ],
                    }
                ),
                json.dumps(
                    {
                        "id": "irr_1",
                        "question": [[{"role": "user", "content": "Disjoint irrelevance prompt"}]],
                        "function": [
                            {
                                "name": "g",
                                "description": "d",
                                "parameters": {"type": "dict", "properties": {}},
                            }
                        ],
                    }
                ),
            ]
        ),
        encoding="utf-8",
    )

    file_map = {
        "BFCL_v3_simple.json": str(q_simple),
        "possible_answer/BFCL_v3_simple.json": str(a_simple),
        "BFCL_v3_irrelevance.json": str(q_irr),
    }
    monkeypatch.setattr(
        huggingface_hub,
        "hf_hub_download",
        lambda repo, filename, **kw: file_map[filename],
    )

    test_cases = [
        BFCLCase(
            case_id="simple_0",
            category="simple",
            messages=({"role": "user", "content": "Held-out test prompt"},),
            tools=(),
            ground_truth=(),
        ),
        BFCLCase(
            case_id="irr_0",
            category="irrelevance",
            messages=({"role": "user", "content": "Held-out irrelevance"},),
            tools=(),
            ground_truth=(),
        ),
    ]
    examples = load_bfcl_disjoint_onpolicy_examples(
        bfcl_test_cases=test_cases,
        categories=("simple", "irrelevance"),
        start_offset=1,
        max_per_category=10,
    )
    # simple_1 must be skipped due to case-insensitive prompt collision with simple_0!
    assert len(examples) == 2
    assert examples[0][1] == "Disjoint train prompt"
    assert "<function=f>" in examples[0][2]
    assert examples[1][1] == "Disjoint irrelevance prompt"
    assert "<tool_call>" not in examples[1][2]


def test_collect_bfcl_onpolicy_self_distill_separates_corrections_and_replays(
    tmp_path: Path, monkeypatch
) -> None:
    import huggingface_hub

    q_simple = tmp_path / "BFCL_v3_simple.json"
    a_simple = tmp_path / "ans_simple.json"
    q_simple.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "id": "simple_0",
                        "question": [[{"role": "user", "content": "Held-out test"}]],
                        "function": [
                            {
                                "name": "f",
                                "description": "d",
                                "parameters": {
                                    "type": "dict",
                                    "properties": {"x": {"type": "integer"}},
                                },
                            }
                        ],
                    }
                ),
                json.dumps(
                    {
                        "id": "simple_1",
                        "question": [[{"role": "user", "content": "Dev passing prompt"}]],
                        "function": [
                            {
                                "name": "f",
                                "description": "d",
                                "parameters": {
                                    "type": "dict",
                                    "properties": {"x": {"type": "integer"}},
                                },
                            }
                        ],
                    }
                ),
                json.dumps(
                    {
                        "id": "simple_2",
                        "question": [[{"role": "user", "content": "Dev failing prompt"}]],
                        "function": [
                            {
                                "name": "f",
                                "description": "d",
                                "parameters": {
                                    "type": "dict",
                                    "properties": {"x": {"type": "integer"}},
                                },
                            }
                        ],
                    }
                ),
            ]
        ),
        encoding="utf-8",
    )
    a_simple.write_text(
        "\n".join(
            [
                json.dumps({"id": "simple_0", "ground_truth": [{"f": {"x": [1]}}]}),
                json.dumps({"id": "simple_1", "ground_truth": [{"f": {"x": [2]}}]}),
                json.dumps({"id": "simple_2", "ground_truth": [{"f": {"x": [3]}}]}),
            ]
        ),
        encoding="utf-8",
    )
    file_map = {
        "BFCL_v3_simple.json": str(q_simple),
        "possible_answer/BFCL_v3_simple.json": str(a_simple),
    }
    monkeypatch.setattr(
        huggingface_hub,
        "hf_hub_download",
        lambda repo, filename, **kw: file_map[filename],
    )

    class _FakeKernel:
        def generate_messages_with_trace(self, msgs, **kw):
            u = msgs[-1]["content"]
            if "passing" in u:
                raw = "<tool_call>\n<function=f>\n<parameter=x>\n2\n</parameter>\n</function>\n</tool_call>"
                return SimpleNamespace(
                    tool_calls=[_tc("f", {"x": 2})],
                    raw_response=raw,
                    hit_token_limit=False,
                )
            return SimpleNamespace(
                tool_calls=[_tc("f", {"x": 999})],
                raw_response="<tool_call>wrong</tool_call>",
                hit_token_limit=False,
            )

    test_cases = [
        BFCLCase("simple_0", "simple", ({"role": "user", "content": "Held-out test"},), (), ())
    ]
    examples, stats = collect_bfcl_onpolicy_self_distill_examples(
        kernel=_FakeKernel(),
        bfcl_test_cases=test_cases,
        categories=("simple",),
        start_offset=1,
        stream_per_category=10,
    )
    assert stats == {"stream_total": 2, "failed_corrections": 1, "passed_replays": 1}
    # 2x correction for simple_2 + 1x self-replay for simple_1 = 3 training examples
    assert len(examples) == 3
    targets_by_prompt: dict[str, list[str]] = {}
    for _, prompt, target, _ in examples:
        targets_by_prompt.setdefault(prompt, []).append(target)
    assert len(targets_by_prompt["Dev failing prompt"]) == 2
    assert "\n3\n" in targets_by_prompt["Dev failing prompt"][0]
    assert len(targets_by_prompt["Dev passing prompt"]) == 1
    assert "\n2\n" in targets_by_prompt["Dev passing prompt"][0]
