"""Unit tests for the official BFCL v3 AST grader and Qwen3.5 XML tool-call formatter."""

from __future__ import annotations

import json

from subcor.bfcl import (
    BFCLCase,
    bfcl_functions_to_openai_tools,
    format_qwen35_xml_tool_calls,
    grade_bfcl_case,
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
