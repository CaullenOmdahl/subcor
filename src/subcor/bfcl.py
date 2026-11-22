"""Official Berkeley Function Calling Leaderboard (BFCL v3) PoC harness for SUBCOR.

Provides:
  1. Official BFCL v3 dataset loader (`gorilla-llm/Berkeley-Function-Calling-Leaderboard`)
     converting BFCL schemas into standard OpenAI `tools` definitions.
  2. Strict official BFCL v3 AST / argument-matching grader (`grade_bfcl_case`)
     supporting `simple`, `multiple`, `parallel`, and `irrelevance` (and `live_*`)
     categories, including order-independent parallel matching, optional-parameter
     handling (`""` in `possible_answer`), numeric equality, and strict bool vs int checks.
  3. Strictly disjoint open training loader (`NousResearch/hermes-function-calling-v1`)
     formatting single, parallel, and abstention examples in native Qwen3.5 XML
     `<tool_call>` syntax with an explicit zero-overlap assertion against BFCL test prompts.
  4. End-to-end governed PoC runner (`run_bfcl_poc`) measuring Day-0 Frozen Base vs.
     SUBCOR Skill Unit(s) on official BFCL v3 + locked general-QA non-regression anchors
     + Cortical Handoff (`TurnConfidence`) hybrid accuracy.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from subcor.confidence import ConfidencePolicy, evaluate_turn_confidence, parse_tool_calls
from subcor.router import MoESkillRouter, RouterGate, RoutingDecision

BFCL_HF_REPO = "gorilla-llm/Berkeley-Function-Calling-Leaderboard"
HERMES_HF_REPO = "NousResearch/hermes-function-calling-v1"

DEFAULT_BFCL_CATEGORIES: tuple[str, ...] = ("simple", "multiple", "parallel", "irrelevance")


@dataclass(frozen=True)
class BFCLCase:
    """One official BFCL v3 evaluation case."""

    case_id: str
    category: str
    messages: tuple[dict[str, str], ...]
    tools: tuple[dict[str, Any], ...]
    ground_truth: tuple[dict[str, dict[str, list[Any]]], ...]  # empty for irrelevance


def _normalize_schema_types(obj: Any) -> Any:
    """Convert BFCL's Python-style schema types (`dict`, `float`, `tuple`) to JSON Schema."""
    if isinstance(obj, list):
        return [_normalize_schema_types(x) for x in obj]
    if not isinstance(obj, dict):
        return obj
    type_map = {
        "dict": "object",
        "float": "number",
        "tuple": "array",
        "any": "string",
    }
    out: dict[str, Any] = {}
    for k, v in obj.items():
        if k == "type" and isinstance(v, str):
            out[k] = type_map.get(v.lower(), v.lower())
        else:
            out[k] = _normalize_schema_types(v)
    return out


def bfcl_functions_to_openai_tools(functions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Wrap BFCL function specs into standard OpenAI `tools` objects."""
    tools: list[dict[str, Any]] = []
    for fn in functions:
        fn_copy = _normalize_schema_types(fn)
        tools.append({"type": "function", "function": fn_copy})
    return tools


def load_bfcl_v3_cases(
    *,
    categories: tuple[str, ...] = DEFAULT_BFCL_CATEGORIES,
    max_per_category: int = 50,
) -> list[BFCLCase]:
    """Download and parse official BFCL v3 test cases + ground-truth possible answers."""
    from huggingface_hub import hf_hub_download

    cases: list[BFCLCase] = []
    for cat in categories:
        q_path = hf_hub_download(BFCL_HF_REPO, f"BFCL_v3_{cat}.json", repo_type="dataset")
        questions = [
            json.loads(line)
            for line in Path(q_path).read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

        gt_by_id: dict[str, list[dict[str, dict[str, list[Any]]]]] = {}
        if "irrelevance" not in cat:
            a_path = hf_hub_download(
                BFCL_HF_REPO, f"possible_answer/BFCL_v3_{cat}.json", repo_type="dataset"
            )
            for line in Path(a_path).read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                row = json.loads(line)
                gt_by_id[str(row["id"])] = row["ground_truth"]

        for row in questions[:max_per_category]:
            cid = str(row["id"])
            raw_turns = row["question"]
            # BFCL v3 stores single-turn prompts as [[{role, content}, ...]].
            turn_msgs = (
                raw_turns[0] if (raw_turns and isinstance(raw_turns[0], list)) else raw_turns
            )
            msgs = tuple({"role": str(m["role"]), "content": str(m["content"])} for m in turn_msgs)
            tools = tuple(bfcl_functions_to_openai_tools(row.get("function", [])))
            gt = tuple(gt_by_id.get(cid, []))
            cases.append(
                BFCLCase(
                    case_id=cid,
                    category=cat,
                    messages=msgs,
                    tools=tools,
                    ground_truth=gt,
                )
            )
    return cases


def _norm_bfcl_str(s: str) -> str:
    """Official BFCL-style string normalization (lowercase, collapse spaces, strip outer punct)."""
    cleaned = re.sub(r"[\s,.;:!?\"'`]+", " ", s.strip().lower()).strip()
    return cleaned


def _value_matches_bfcl(pred: Any, expected: Any) -> bool:
    """Check whether predicted argument value matches one BFCL ground-truth candidate."""
    if isinstance(expected, bool) or isinstance(pred, bool):
        return type(pred) is type(expected) and pred == expected
    if isinstance(expected, (int, float)) and isinstance(pred, (int, float)):
        return abs(float(pred) - float(expected)) <= 1e-6
    if isinstance(expected, str) and isinstance(pred, str):
        return _norm_bfcl_str(pred) == _norm_bfcl_str(expected)
    if isinstance(expected, list) and isinstance(pred, list):
        return len(pred) == len(expected) and all(
            _value_matches_bfcl(p, e) for p, e in zip(pred, expected, strict=True)
        )
    if isinstance(expected, dict) and isinstance(pred, dict):
        return pred.keys() == expected.keys() and all(
            _value_matches_bfcl(pred[k], expected[k]) for k in expected
        )
    # Allow stringified numbers ONLY if expected is string representation of pred or vice versa
    if isinstance(expected, str) and isinstance(pred, (int, float)):
        return _norm_bfcl_str(str(pred)) == _norm_bfcl_str(expected)
    return pred == expected


def _norm_fn_name(name: str) -> str:
    return name.strip().replace(".", "_")


def _call_matches_gt(
    pred_name: str,
    pred_args: dict[str, Any],
    gt_entry: dict[str, dict[str, list[Any]]],
) -> bool:
    """Check if one parsed OpenAI tool call matches one BFCL ground-truth spec."""
    if len(gt_entry) != 1:
        return False
    gt_name, gt_params = next(iter(gt_entry.items()))
    if pred_name != gt_name and _norm_fn_name(pred_name) != _norm_fn_name(gt_name):
        return False

    # Reject hallucinated extra arguments not listed in ground truth.
    for pk in pred_args:
        if pk not in gt_params:
            return False

    for param_name, acceptable_vals in gt_params.items():
        if param_name not in pred_args or pred_args[param_name] is None:
            # BFCL marks optional parameters by including "" or None in acceptable values.
            if "" in acceptable_vals or None in acceptable_vals:
                continue
            return False
        pval = pred_args[param_name]
        if not any(_value_matches_bfcl(pval, cand) for cand in acceptable_vals):
            return False
    return True


def _bipartite_match(
    preds: list[tuple[str, dict[str, Any]]],
    gts: tuple[dict[str, dict[str, list[Any]]], ...],
) -> bool:
    """Order-independent exact matching between predicted tool calls and ground-truth calls."""
    n = len(gts)
    if len(preds) != n:
        return False
    used = [False] * n

    def _dfs(gt_idx: int) -> bool:
        if gt_idx == n:
            return True
        gt_item = gts[gt_idx]
        for p_idx in range(n):
            if not used[p_idx] and _call_matches_gt(preds[p_idx][0], preds[p_idx][1], gt_item):
                used[p_idx] = True
                if _dfs(gt_idx + 1):
                    return True
                used[p_idx] = False
        return False

    return _dfs(0)


def grade_bfcl_case(case: BFCLCase, predicted_tool_calls: list[dict[str, Any]]) -> bool:
    """Deterministically grade predicted OpenAI `tool_calls` against an official BFCL v3 case."""
    if "irrelevance" in case.category:
        return len(predicted_tool_calls) == 0

    if len(predicted_tool_calls) != len(case.ground_truth):
        return False

    parsed_preds: list[tuple[str, dict[str, Any]]] = []
    for tc in predicted_tool_calls:
        fn = tc.get("function") or {}
        name = str(fn.get("name", ""))
        raw_args = fn.get("arguments", "{}")
        if isinstance(raw_args, str):
            try:
                args = json.loads(raw_args)
            except json.JSONDecodeError:
                return False
        elif isinstance(raw_args, dict):
            args = raw_args
        else:
            return False
        if not isinstance(args, dict):
            return False
        parsed_preds.append((name, args))

    return _bipartite_match(parsed_preds, case.ground_truth)


def format_qwen35_xml_tool_calls(calls: list[dict[str, Any]]) -> str:
    """Render OpenAI/Hermes tool calls into native Qwen3.5 XML `<tool_call>` syntax."""
    blocks: list[str] = []
    for call in calls:
        fn = call.get("function", call)
        name = str(fn["name"])
        args = fn.get("arguments", {})
        if isinstance(args, str):
            args = json.loads(args)
        lines = ["<tool_call>", f"<function={name}>"]
        for k, v in args.items():
            val_str = (
                json.dumps(v, separators=(",", ":"))
                if isinstance(v, (dict, list, bool))
                else str(v)
            )
            lines.append(f"<parameter={k}>")
            lines.append(val_str)
            lines.append("</parameter>")
        lines.append("</function>")
        lines.append("</tool_call>")
        blocks.append("\n".join(lines))
    return "\n".join(blocks)


_TOOLS_TAG_RE = re.compile(r"<tools>\s*(.*?)\s*</tools>", flags=re.DOTALL)


def load_hermes_disjoint_training_examples(
    *,
    bfcl_test_cases: list[BFCLCase],
    n_single: int = 35,
    n_parallel: int = 35,
    n_abstain: int = 30,
    max_tools: int = 6,
) -> list[tuple[str | None, str, str, list[dict[str, Any]] | None]]:
    """Load strictly disjoint single, parallel, and abstention tool examples from Hermes/Glaive.

    Formats targets in native Qwen3.5 XML `<tool_call>` syntax and asserts zero prompt
    overlap against `bfcl_test_cases`.
    """
    from huggingface_hub import hf_hub_download

    test_prompts = {
        m["content"].strip().lower()
        for c in bfcl_test_cases
        for m in c.messages
        if m["role"] == "user"
    }
    sys_tool = (
        "You are a precise function-calling assistant. Call tools ONLY when a provided tool "
        "matches the user's request. If multiple calls are needed, emit all of them. "
        "If no provided tool is relevant, reply concisely in plain text without calling any tool."
    )

    singles: list[tuple[str | None, str, str, list[dict[str, Any]] | None]] = []
    parallels: list[tuple[str | None, str, str, list[dict[str, Any]] | None]] = []
    abstains: list[tuple[str | None, str, str, list[dict[str, Any]] | None]] = []

    def _extract_tools(sys_text: str) -> list[dict[str, Any]] | None:
        # Hermes system prompts mention "<tools> </tools>" in prose before the real block;
        # scan all <tools>...</tools> spans and parse the first non-empty JSON tool list.
        for m in _TOOLS_TAG_RE.finditer(sys_text):
            raw = m.group(1).strip()
            if not raw:
                continue
            parsed: Any = None
            try:
                parsed = json.loads(raw)
            except json.JSONDecodeError:
                # Some Hermes rows contain newline-delimited JSON objects inside <tools>.
                objs: list[Any] = []
                for line in raw.splitlines():
                    line = line.strip()
                    if line:
                        try:
                            objs.append(json.loads(line))
                        except json.JSONDecodeError:
                            objs = []
                            break
                if objs:
                    parsed = objs
            if parsed is None:
                continue
            if isinstance(parsed, dict):
                parsed = [parsed]
            if not isinstance(parsed, list) or not parsed or len(parsed) > max_tools:
                continue
            tools_out: list[dict[str, Any]] = []
            for item in parsed:
                if isinstance(item, dict) and "function" in item:
                    tools_out.append(item)
                elif isinstance(item, dict) and "name" in item:
                    tools_out.append({"type": "function", "function": item})
            if tools_out:
                return tools_out
        return None

    # 1. Single & parallel calls from func-calling-singleturn.json (also collect
    #    mismatched-tool irrelevance pairs by pairing a prompt with unrelated Hermes tools).
    p_single = hf_hub_download(HERMES_HF_REPO, "func-calling-singleturn.json", repo_type="dataset")
    parsed_hermes: list[tuple[str, str, list[dict[str, Any]], int]] = []
    for row in json.loads(Path(p_single).read_text(encoding="utf-8")):
        conv = row.get("conversations") or []
        sys_msg = next((c["value"] for c in conv if c.get("from") == "system"), "")
        user_msg = next((c["value"] for c in conv if c.get("from") == "human"), "")
        asst_msg = next((c["value"] for c in conv if c.get("from") == "gpt"), "")
        if not user_msg or not asst_msg or user_msg.strip().lower() in test_prompts:
            continue
        tools = _extract_tools(sys_msg)
        if not tools:
            continue
        _, calls = parse_tool_calls(asst_msg)
        if not calls:
            continue
        target_xml = format_qwen35_xml_tool_calls(calls)
        parsed_hermes.append((user_msg.strip(), target_xml, tools, len(calls)))
        if len(calls) == 1 and len(singles) < n_single:
            singles.append((sys_tool, user_msg.strip(), target_xml, tools))
        elif len(calls) >= 2 and len(parallels) < n_parallel:
            parallels.append((sys_tool, user_msg.strip(), target_xml, tools))
        if (
            len(singles) >= n_single
            and len(parallels) >= n_parallel
            and len(parsed_hermes) >= n_abstain + 60
        ):
            break

    # 2a. Real mismatched-tool abstention pairs: user asks for task A while only unrelated
    #     Hermes tools from a distant row are available -> model must abstain cleanly.
    n_mismatch_target = (n_abstain * 2) // 3
    used_mismatch: set[str] = set()
    for idx, (u_prompt, _, orig_tools, _) in enumerate(parsed_hermes):
        if len(abstains) >= n_mismatch_target:
            break
        if u_prompt in used_mismatch:
            continue
        other_tools = parsed_hermes[(idx + 53) % len(parsed_hermes)][2]
        orig_names = {t["function"]["name"] for t in orig_tools}
        other_names = {t["function"]["name"] for t in other_tools}
        if orig_names & other_names:
            continue
        used_mismatch.add(u_prompt)
        abstains.append(
            (
                sys_tool,
                u_prompt,
                "I cannot complete this request because none of the provided tools support it.",
                other_tools,
            )
        )

    # 2b. Conversational no-tool-needed examples from glaive-function-calling-5k.json
    p_glaive = hf_hub_download(
        HERMES_HF_REPO, "glaive-function-calling-5k.json", repo_type="dataset"
    )
    for row in json.loads(Path(p_glaive).read_text(encoding="utf-8")):
        if len(abstains) >= n_abstain:
            break
        conv = row.get("conversations") or []
        sys_msg = next((c["value"] for c in conv if c.get("from") == "system"), "")
        user_msg = next((c["value"] for c in conv if c.get("from") == "human"), "")
        asst_msg = next((c["value"] for c in conv if c.get("from") == "gpt"), "")
        if not user_msg or not asst_msg or user_msg.strip().lower() in test_prompts:
            continue
        if "<tool_call>" in asst_msg or len(asst_msg) > 350:
            continue
        tools = _extract_tools(sys_msg)
        if not tools:
            continue
        abstains.append((sys_tool, user_msg.strip(), asst_msg.strip(), tools))

    combined = singles + parallels + abstains
    train_prompt_set = {p.strip().lower() for _, p, _, _ in combined}
    overlap = train_prompt_set & test_prompts
    if overlap:
        raise AssertionError(f"Train/test leakage detected: {len(overlap)} overlapping prompts")
    return combined


def _gt_entry_to_call_dict(gt_entry: dict[str, dict[str, list[Any]]]) -> dict[str, Any]:
    """Convert one BFCL possible_answer ground-truth entry into a canonical OpenAI tool call."""
    fn_name, params = next(iter(gt_entry.items()))
    args: dict[str, Any] = {}
    for k, candidates in params.items():
        if not candidates:
            continue
        first = candidates[0]
        if first == "" and len(candidates) == 1:
            continue
        if first != "":
            args[k] = first
    return {"function": {"name": fn_name, "arguments": args}}


def load_bfcl_disjoint_onpolicy_examples(
    *,
    bfcl_test_cases: list[BFCLCase],
    categories: tuple[str, ...] = DEFAULT_BFCL_CATEGORIES,
    start_offset: int = 50,
    max_per_category: int = 40,
) -> list[tuple[str | None, str, str, list[dict[str, Any]] | None]]:
    """Load strictly disjoint on-policy training examples from official BFCL[start_offset:].

    Asserts ZERO case-ID overlap and ZERO prompt overlap with `bfcl_test_cases`.
    """
    from huggingface_hub import hf_hub_download

    test_ids = {c.case_id for c in bfcl_test_cases}
    test_prompts = {
        m["content"].strip().lower()
        for c in bfcl_test_cases
        for m in c.messages
        if m["role"] == "user"
    }
    sys_tool = (
        "You are a precise function-calling assistant. Call tools ONLY when a provided tool "
        "matches the user's request. If multiple calls are needed, emit all of them. "
        "If no provided tool is relevant, reply concisely in plain text without calling any tool."
    )

    examples: list[tuple[str | None, str, str, list[dict[str, Any]] | None]] = []
    for cat in categories:
        q_path = hf_hub_download(BFCL_HF_REPO, f"BFCL_v3_{cat}.json", repo_type="dataset")
        questions = [
            json.loads(line)
            for line in Path(q_path).read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        gt_by_id: dict[str, list[dict[str, dict[str, list[Any]]]]] = {}
        if "irrelevance" not in cat:
            a_path = hf_hub_download(
                BFCL_HF_REPO, f"possible_answer/BFCL_v3_{cat}.json", repo_type="dataset"
            )
            for line in Path(a_path).read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                row = json.loads(line)
                gt_by_id[str(row["id"])] = row["ground_truth"]

        count = 0
        for row in questions[start_offset:]:
            if count >= max_per_category:
                break
            cid = str(row["id"])
            if cid in test_ids:
                continue
            raw_turns = row["question"]
            turn_msgs = (
                raw_turns[0] if (raw_turns and isinstance(raw_turns[0], list)) else raw_turns
            )
            user_msg = next(
                (str(m["content"]).strip() for m in reversed(turn_msgs) if m.get("role") == "user"),
                "",
            )
            if not user_msg or user_msg.lower() in test_prompts:
                continue
            tools = bfcl_functions_to_openai_tools(row.get("function", []))
            if "irrelevance" in cat:
                target = (
                    "I cannot complete this request because none of the provided tools support it."
                )
            else:
                gt_calls = gt_by_id.get(cid) or []
                if not gt_calls:
                    continue
                target = format_qwen35_xml_tool_calls(
                    [_gt_entry_to_call_dict(entry) for entry in gt_calls]
                )
            examples.append((sys_tool, user_msg, target, tools))
            count += 1

    train_prompts = {p.strip().lower() for _, p, _, _ in examples}
    if train_prompts & test_prompts:
        raise AssertionError("On-policy BFCL train/test prompt leakage detected!")
    return examples


def collect_bfcl_onpolicy_self_distill_examples(
    *,
    kernel: Any,
    bfcl_test_cases: list[BFCLCase],
    categories: tuple[str, ...] = DEFAULT_BFCL_CATEGORIES,
    start_offset: int = 50,
    stream_per_category: int = 25,
    max_tokens: int = 512,
    seed: int = 42,
) -> tuple[list[tuple[str | None, str, str, list[dict[str, Any]] | None]], dict[str, int]]:
    """Collect true on-policy error corrections + base self-replay KL anchors from BFCL[start_offset:].

    Runs the frozen base kernel on strictly disjoint experience cases (zero case-ID and zero
    prompt overlap with `bfcl_test_cases`):
    - On FAILED cases: records the verified ground-truth correction.
    - On PASSED cases: records the model's OWN raw output as a KL-stabilizing self-replay
      so LoRA updates fix real failure modes without drifting away from passing behaviors.
    """
    from huggingface_hub import hf_hub_download

    test_ids = {c.case_id for c in bfcl_test_cases}
    test_prompts = {
        m["content"].strip().lower()
        for c in bfcl_test_cases
        for m in c.messages
        if m["role"] == "user"
    }
    sys_tool = (
        "You are a precise function-calling assistant. Call tools ONLY when a provided tool "
        "matches the user's request. If multiple calls are needed, emit all of them. "
        "If no provided tool is relevant, reply concisely in plain text without calling any tool."
    )

    corrections: list[tuple[str | None, str, str, list[dict[str, Any]] | None]] = []
    replays_by_cat: dict[str, list[tuple[str | None, str, str, list[dict[str, Any]] | None]]] = {
        c: [] for c in categories
    }
    stats: dict[str, int] = {"stream_total": 0, "failed_corrections": 0, "passed_replays": 0}

    for cat in categories:
        q_path = hf_hub_download(BFCL_HF_REPO, f"BFCL_v3_{cat}.json", repo_type="dataset")
        questions = [
            json.loads(line)
            for line in Path(q_path).read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        gt_by_id: dict[str, list[dict[str, dict[str, list[Any]]]]] = {}
        if "irrelevance" not in cat:
            a_path = hf_hub_download(
                BFCL_HF_REPO, f"possible_answer/BFCL_v3_{cat}.json", repo_type="dataset"
            )
            for line in Path(a_path).read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                row = json.loads(line)
                gt_by_id[str(row["id"])] = row["ground_truth"]

        seen = 0
        for row in questions[start_offset:]:
            if seen >= stream_per_category:
                break
            cid = str(row["id"])
            if cid in test_ids:
                continue
            raw_turns = row["question"]
            turn_msgs = (
                raw_turns[0] if (raw_turns and isinstance(raw_turns[0], list)) else raw_turns
            )
            user_msg = next(
                (str(m["content"]).strip() for m in reversed(turn_msgs) if m.get("role") == "user"),
                "",
            )
            if not user_msg or user_msg.lower() in test_prompts:
                continue
            tools = bfcl_functions_to_openai_tools(row.get("function", []))
            gt = tuple(gt_by_id.get(cid, []))
            if "irrelevance" not in cat and not gt:
                continue

            case = BFCLCase(
                case_id=cid,
                category=cat,
                messages=tuple(
                    {"role": str(m["role"]), "content": str(m["content"])} for m in turn_msgs
                ),
                tools=tuple(tools),
                ground_truth=gt,
            )
            seen += 1
            stats["stream_total"] += 1

            msgs = [{"role": "system", "content": sys_tool}, *[dict(m) for m in case.messages]]
            tr = kernel.generate_messages_with_trace(
                msgs,
                tools=list(case.tools),
                max_tokens=max_tokens,
                enable_thinking=False,
            )
            passed = grade_bfcl_case(case, list(tr.tool_calls))
            if passed:
                raw_out = (tr.raw_response or "").strip()
                if raw_out and not tr.hit_token_limit and len(raw_out) <= 1600:
                    replays_by_cat[cat].append((sys_tool, user_msg, raw_out, tools))
                    stats["passed_replays"] += 1
            else:
                if "irrelevance" in cat:
                    target = (
                        "None of the available tools can fulfill this request, "
                        "so I cannot call a tool for it."
                    )
                else:
                    target = format_qwen35_xml_tool_calls(
                        [_gt_entry_to_call_dict(entry) for entry in gt]
                    )
                corrections.append((sys_tool, user_msg, target, tools))
                stats["failed_corrections"] += 1

    # Build a balanced buffer: 2x verified corrections + equal KL-anchor replays spread evenly across categories.
    rng = np.random.default_rng(seed)
    target_replays = max(len(corrections) * 2, 24)
    per_cat_quota = max(4, target_replays // max(1, len(categories)))
    chosen_replays: list[tuple[str | None, str, str, list[dict[str, Any]] | None]] = []
    for cat in categories:
        pool = list(replays_by_cat[cat])
        if len(pool) > per_cat_quota:
            idxs = rng.choice(len(pool), size=per_cat_quota, replace=False)
            pool = [pool[int(i)] for i in sorted(idxs)]
        chosen_replays.extend(pool)

    combined = [*corrections, *corrections, *chosen_replays]
    if combined:
        order = rng.permutation(len(combined))
        combined = [combined[int(i)] for i in order]

    train_prompts = {p.strip().lower() for _, p, _, _ in combined}
    if train_prompts & test_prompts:
        raise AssertionError("Self-distill BFCL train/test prompt leakage detected!")
    return combined, stats


def run_bfcl_poc(
    *,
    work_dir: Path,
    model_id: str = "mlx-community/Qwen3.5-4B-4bit",
    categories: tuple[str, ...] = DEFAULT_BFCL_CATEGORIES,
    max_per_category: int = 50,
    train_source: str = "hermes",  # "hermes" | "bfcl_onpolicy" | "bfcl_self_distill"
    n_single: int = 35,
    n_parallel: int = 35,
    n_abstain: int = 30,
    rank: int = 16,
    steps: int = 120,
    lr: float = 2e-4,
    grad_accum: int = 4,
    max_tokens: int = 512,
    num_adapted_layers: int = 8,
    projections: tuple[str, ...] = ("mlp.down_proj", "self_attn.o_proj"),
    seed: int = 42,
) -> dict[str, Any]:
    """Run the full Day-0 vs. SUBCOR BFCL v3 PoC on Apple Silicon MLX."""
    from subcor.backends.mlx_kernel import MLXKernel
    from subcor.benchmark import LOCKED_ANCHOR_AXES, core_v1_suite, grade_response
    from subcor.server import DEFAULT_NEGATIVE_CALIBRATION_PROMPTS
    from subcor.types import AdaptationMode, DeltaEncoding, SkillSpec
    from subcor.units import SkillUnitRegistry, build_layer_delta

    work_dir.mkdir(parents=True, exist_ok=True)
    cases = load_bfcl_v3_cases(categories=categories, max_per_category=max_per_category)

    # Probe layer count from model config via MLXKernel.
    from huggingface_hub import hf_hub_download

    cfg = json.loads(Path(hf_hub_download(model_id, "config.json")).read_text(encoding="utf-8"))
    total_layers = int(cfg.get("text_config", cfg).get("num_hidden_layers", 24))
    target_indices = list(range(max(0, total_layers - num_adapted_layers), total_layers))

    kernel = MLXKernel(
        model_id=model_id,
        target_layer_indices=target_indices,
        target_projections=projections,
    )
    self_distill_stats: dict[str, int] | None = None
    if train_source == "bfcl_self_distill":
        train_examples, self_distill_stats = collect_bfcl_onpolicy_self_distill_examples(
            kernel=kernel,
            bfcl_test_cases=cases,
            categories=categories,
            start_offset=max(max_per_category, 50),
            stream_per_category=max(n_single, n_parallel, n_abstain),
            max_tokens=max_tokens,
            seed=seed,
        )
        print(
            f"[BFCL PoC] On-policy self-distill stream (BFCL[{max(max_per_category, 50)}:], 0 overlap): "
            f"{self_distill_stats}",
            flush=True,
        )
    elif train_source == "bfcl_onpolicy":
        train_examples = load_bfcl_disjoint_onpolicy_examples(
            bfcl_test_cases=cases,
            categories=categories,
            start_offset=max(max_per_category, 50),
            max_per_category=max(n_single, n_parallel, n_abstain),
        )
    else:
        train_examples = load_hermes_disjoint_training_examples(
            bfcl_test_cases=cases,
            n_single=n_single,
            n_parallel=n_parallel,
            n_abstain=n_abstain,
        )
    if not train_examples:
        raise ValueError(f"Loaded 0 disjoint training examples for train_source={train_source!r}")
    print(
        f"[BFCL PoC] Loaded {len(cases)} official BFCL v3 test cases & "
        f"{len(train_examples)} disjoint {train_source} training examples (0 overlap).",
        flush=True,
    )
    layer_names = sorted(
        kernel.instrumented.keys(),
        key=lambda n: (
            int(n.split(".")[1]) if len(n.split(".")) > 1 and n.split(".")[1].isdigit() else 10**9,
            n,
        ),
    )
    tap_layer = next(
        (n for n in layer_names if n.endswith("mlp.down_proj")),
        layer_names[0],
    )

    sys_tool = (
        "You are a precise function-calling assistant. Call tools ONLY when a provided tool "
        "matches the user's request. If multiple calls are needed, emit all of them. "
        "If no provided tool is relevant, reply concisely in plain text without calling any tool."
    )
    anchor_cases = [c for c in core_v1_suite().cases if c.axis in LOCKED_ANCHOR_AXES]

    def eval_suite(
        *,
        router: MoESkillRouter | None,
        registry: SkillUnitRegistry | None,
    ) -> dict[str, Any]:
        by_cat: dict[str, dict[str, int]] = {
            cat: {
                "passed": 0,
                "total": 0,
                "escalated": 0,
                "accepted_passed": 0,
                "accepted_total": 0,
            }
            for cat in categories
        }
        latencies: list[float] = []
        conf_policy = ConfidencePolicy()

        for case in cases:
            msgs = [{"role": "system", "content": sys_tool}, *[dict(m) for m in case.messages]]
            t0 = time.perf_counter()
            if router is not None and registry is not None and router.registered_units:
                trace, decision = kernel.generate_routed_messages(
                    msgs,
                    router=router,
                    registry=registry,
                    tools=list(case.tools),
                    max_tokens=max_tokens,
                    enable_thinking=False,
                    top_k=1,
                    seat="tool_calling",
                )
            else:
                trace = kernel.generate_messages_with_trace(
                    msgs,
                    tools=list(case.tools),
                    max_tokens=max_tokens,
                    enable_thinking=False,
                )
                decision = RoutingDecision(selected_units=(), scores={}, thresholds={})
            latencies.append((time.perf_counter() - t0) * 1000.0)

            passed = grade_bfcl_case(case, list(trace.tool_calls))
            conf = evaluate_turn_confidence(
                token_logprobs=list(trace.token_logprobs),
                routing=decision,
                hit_token_limit=trace.hit_token_limit,
                repetition_ratio=trace.repetition_ratio,
                policy=conf_policy,
            )
            st = by_cat[case.category]
            st["total"] += 1
            if passed:
                st["passed"] += 1
            if conf.escalate_to_cortex:
                st["escalated"] += 1
            else:
                st["accepted_total"] += 1
                if passed:
                    st["accepted_passed"] += 1

        # Evaluate locked general-QA non-regression anchors (seat="general", no tools).
        anchor_hits = 0
        for ac in anchor_cases:
            a_msgs: list[dict[str, Any]] = []
            if ac.system_prompt:
                a_msgs.append({"role": "system", "content": ac.system_prompt})
            a_msgs.append({"role": "user", "content": ac.prompt})
            if router is not None and registry is not None and router.registered_units:
                a_tr, _ = kernel.generate_routed_messages(
                    a_msgs,
                    router=router,
                    registry=registry,
                    tools=None,
                    max_tokens=ac.max_tokens,
                    enable_thinking=False,
                    top_k=1,
                    seat="general",
                )
            else:
                a_tr = kernel.generate_messages_with_trace(
                    a_msgs,
                    tools=None,
                    max_tokens=ac.max_tokens,
                    enable_thinking=False,
                )
            if grade_response(ac, a_tr.response, hit_token_limit=a_tr.hit_token_limit):
                anchor_hits += 1

        cat_summary: dict[str, Any] = {}
        total_pass = 0
        total_n = 0
        total_esc = 0
        total_acc_pass = 0
        total_acc_n = 0
        for cat, st in by_cat.items():
            total_pass += st["passed"]
            total_n += st["total"]
            total_esc += st["escalated"]
            total_acc_pass += st["accepted_passed"]
            total_acc_n += st["accepted_total"]
            cat_summary[cat] = {
                "accuracy": round(st["passed"] / max(1, st["total"]), 4),
                "passed": st["passed"],
                "total": st["total"],
                "escalation_rate": round(st["escalated"] / max(1, st["total"]), 4),
                "local_confident_accuracy": round(
                    st["accepted_passed"] / max(1, st["accepted_total"]), 4
                ),
            }

        return {
            "overall_bfcl_accuracy": round(total_pass / max(1, total_n), 4),
            "passed": total_pass,
            "total": total_n,
            "escalation_rate": round(total_esc / max(1, total_n), 4),
            "local_confident_accuracy": round(total_acc_pass / max(1, total_acc_n), 4),
            "locked_anchor_accuracy": round(anchor_hits / max(1, len(anchor_cases)), 4),
            "mean_latency_ms": round(float(np.mean(latencies)), 1),
            "categories": cat_summary,
        }

    # 1. Day-0 Frozen Base evaluation
    print(f"[BFCL PoC] Evaluating Day-0 frozen base ({model_id})...", flush=True)
    day0 = eval_suite(router=None, registry=None)
    print(
        f"[BFCL PoC] Day-0 Frozen Base: overall={day0['overall_bfcl_accuracy'] * 100:.1f}% "
        f"anchors={day0['locked_anchor_accuracy'] * 100:.1f}% "
        f"cats={ {k: v['accuracy'] for k, v in day0['categories'].items()} }",
        flush=True,
    )

    # 2. Train SUBCOR Skill Unit on strictly disjoint Hermes/Glaive tool examples
    print(
        f"[BFCL PoC] Training rank-{rank} adapter ({steps} steps, {len(layer_names)} layers)...",
        flush=True,
    )
    t_train0 = time.perf_counter()
    adapters, losses, zero_diff = kernel.train_qlora_adapters(
        train_examples,
        layer_names=layer_names,
        rank=rank,
        steps=steps,
        lr=lr,
        grad_accum=grad_accum,
        binary_ste=False,
        seed=seed,
    )
    train_sec = time.perf_counter() - t_train0
    print(
        f"[BFCL PoC] Training done in {train_sec:.1f}s "
        f"(loss {float(np.mean(losses[:5])):.4f} -> {float(np.mean(losses[-5:])):.4f}, "
        f"zero_perturb={zero_diff:.1e})",
        flush=True,
    )

    # Package BOTH exact FP32/low-rank AND Sub-1-Bit encodings so we can compare cleanly.
    registry = SkillUnitRegistry(work_dir / "units")
    router = MoESkillRouter(tap_layer=tap_layer)

    pos_embs = np.stack(
        [
            kernel.encode_prompt_embedding(p, tap_layer=tap_layer)
            for _, p, _, _ in train_examples[:24]
        ]
    )
    neg_embs = np.stack(
        [
            kernel.encode_prompt_embedding(p, tap_layer=tap_layer)
            for p in DEFAULT_NEGATIVE_CALIBRATION_PROMPTS
        ]
    )

    results_by_encoding: dict[str, Any] = {}
    for enc_label, encoding in (
        ("low_rank_fp32", DeltaEncoding.LOW_RANK_FP32),
        ("sub1bit_binary", DeltaEncoding.SUB1BIT_BINARY),
    ):
        uid = f"bfcl-tool-skill-{enc_label}-v1"
        deltas = [
            build_layer_delta(
                name,
                adapters[name],
                encoding=encoding,
                sub1bit_rank=min(rank, 16),
            )
            for name in layer_names
        ]
        manifest = registry.propose(
            unit_id=uid,
            version="1.0.0",
            title=f"BFCL Tool-Calling Skill Unit ({enc_label})",
            summary=f"Disjoint {train_source} tool-calling unit on {model_id}",
            proposer="tools",
            mode=AdaptationMode.DYNAMIC_EXPANSION_FP32,
            deltas=deltas,
            base_kernel_id=model_id,
        )
        # Seat-scoped to "tool_calling" with ultra-conservative threshold so general QA stays pure base.
        gate = RouterGate.fit(
            unit_id=uid,
            tap_layer=tap_layer,
            pos_embeddings=pos_embs,
            neg_embeddings=neg_embs,
            seat_scope="tool_calling",
            safety_bias=-1.5,  # Always active when seat="tool_calling"; strictly skipped on seat="general"
        )
        registry.save_router_gate(gate)
        registry.save_skill_spec(
            SkillSpec(
                unit_id=uid,
                version="1.0.0",
                title=manifest.title,
                summary=manifest.summary,
                task_family="tool_calling",
                proposer="tools",
                system_prompt=sys_tool,
                training_pairs=tuple((p, t) for _, p, t, _ in train_examples),
                eval_cases=(),
                baseline_cases=(),
                negative_prompts=tuple(DEFAULT_NEGATIVE_CALIBRATION_PROMPTS),
                rank=rank,
                sub1bit_rank=min(rank, 16),
                steps=steps,
                lr=lr,
            )
        )
        router.register(gate)
        try:
            eval_res = eval_suite(router=router, registry=registry)
        finally:
            router.unregister(uid)

        manifest = registry.record_conformance(
            uid,
            baseline_before=day0["locked_anchor_accuracy"],
            baseline_after=eval_res["locked_anchor_accuracy"],
            target_before=day0["overall_bfcl_accuracy"],
            target_after=eval_res["overall_bfcl_accuracy"],
            max_regression_tolerance=0.02,
            min_target_improvement=0.01,
        )
        results_by_encoding[enc_label] = {
            "unit_id": uid,
            "state": manifest.state.value,
            "payload_bytes": manifest.payload_bytes,
            "effective_bits_per_weight": round(manifest.effective_bits_per_weight, 5),
            "conformance_reason": manifest.conformance.reason if manifest.conformance else "",
            **eval_res,
        }

    return {
        "model_id": model_id,
        "train_source": train_source,
        "categories": list(categories),
        "total_bfcl_test_cases": len(cases),
        "disjoint_train_examples": len(train_examples),
        "adapted_layers": layer_names,
        "rank": rank,
        "steps": steps,
        "grad_accum": grad_accum,
        "max_tokens": max_tokens,
        "lr": lr,
        "self_distill_stats": self_distill_stats,
        "train_seconds": round(train_sec, 1),
        "zero_perturbation_max_logit_diff": zero_diff,
        "initial_loss": round(float(np.mean(losses[:5])), 4),
        "final_loss": round(float(np.mean(losses[-5:])), 4),
        "day0_frozen_base": day0,
        "subcor_adapted": results_by_encoding,
    }
