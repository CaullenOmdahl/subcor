#!/usr/bin/env python
"""BFCL v4 runner built on the official bfcl-eval package (data, checkers and score weighting).

Run it with the separate eval venv, never the project env:

    BFCL_VENV=/path/to/venv
    $BFCL_VENV/bin/python experiments/bfcl/harness/run_bfcl.py <command> --run-dir RUN ...

Commands
  generate   run inference for a run dir and resume where it stopped (bfcl's per-category
             JSONL result files; items already present are skipped)
  evaluate   grade the run with bfcl's own eval runner (AST / relevance / multi-turn / memory)
  summarize  print official-style columns, coverage, and a comparison with an official board row;
             also writes summary.json and per_item.csv
  all        generate + evaluate + summarize
  pair       paired per-item join of two runs, with a stratified bootstrap of the deltas
             (refuses runs whose settings or item sets differ)
  routing    per-item RouterGate activation of a SUBCOR tenant run (from harness_trace)

Two handler families:
  --handler native    NativeFCHandler: OpenAI tools on /v1/chat/completions
                      (llama-server --jinja, or SUBCOR serve)
  --handler official  any handler registered in bfcl-eval, unmodified (for example the local
                      QwenFCHandler that produced the Qwen3 "(FC)" board rows), pointed at an
                      existing /v1/completions server through REMOTE_OPENAI_BASE_URL
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import re
import sys
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from typing import Any

HERE = Path(__file__).resolve().parent
DEFAULT_CATEGORIES = ["non_live", "live", "multi_turn"]
SKIPPED_ALWAYS = ["web_search_base", "web_search_no_snippet"]  # need SerpAPI
ERROR_PREFIX = "Error during inference"


# --------------------------------------------------------------------------- bootstrap env
def _early_run_dir(argv: list[str]) -> Path | None:
    for i, a in enumerate(argv):
        if a == "--run-dir" and i + 1 < len(argv):
            return Path(argv[i + 1]).resolve()
        if a.startswith("--run-dir="):
            return Path(a.split("=", 1)[1]).resolve()
    return None


def _set_project_root(run_dir: Path) -> None:
    # bfcl-eval reads BFCL_PROJECT_ROOT at import time (result/, score/, locks, id files).
    # One root per run dir keeps runs from colliding.
    run_dir.mkdir(parents=True, exist_ok=True)
    os.environ["BFCL_PROJECT_ROOT"] = str(run_dir)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


if __name__ == "__main__":
    _rd = _early_run_dir(sys.argv[1:])
    if _rd is not None:
        _set_project_root(_rd)

sys.path.insert(0, str(HERE))


def _bfcl():
    """Import bfcl lazily (after BFCL_PROJECT_ROOT is set)."""
    import bfcl_eval  # noqa: F401
    from bfcl_eval.constants import model_config as mc

    return mc


# --------------------------------------------------------------------------- run config
def _slug(text: str) -> str:
    s = re.sub(r"[^A-Za-z0-9.-]+", "-", text).strip("-")
    return s or "run"


def load_run_config(run_dir: Path) -> dict[str, Any] | None:
    p = run_dir / "run_config.json"
    return json.loads(p.read_text()) if p.exists() else None


def save_run_config(run_dir: Path, cfg: dict[str, Any]) -> None:
    (run_dir / "run_config.json").write_text(json.dumps(cfg, indent=2, sort_keys=True) + "\n")


def register_run(cfg: dict[str, Any]) -> str:
    """Put this run's registry entry into bfcl's MODEL_CONFIG_MAPPING (in place, so every
    bfcl module that imported the dict sees it). Returns the registry name."""
    mc = _bfcl()
    registry = cfg["registry"]
    if "_" in registry:
        raise ValueError("registry names must not contain '_' (bfcl maps '_' back to '/')")
    if cfg["handler"] == "native":
        import native_fc_handler as nfh

        nfh.ENDPOINTS[registry] = nfh.endpoint_from_dict(cfg["endpoint"])
        mc.MODEL_CONFIG_MAPPING[registry] = mc.ModelConfig(
            model_name=cfg["endpoint"]["served_model"],
            display_name=cfg.get("display_name") or registry,
            url="local",
            org="local",
            license="local",
            model_handler=nfh.NativeFCHandler,
            input_price=None,
            output_price=None,
            is_fc_model=True,
            # Tools are compiled in OPENAI_COMPLETIONS style, which rewrites '.' to '_'
            # in function names; the AST checker must then expect '_' (as for GPT FC).
            underscore_to_dot=True,
        )
    else:
        os.environ["REMOTE_OPENAI_BASE_URL"] = cfg["official"]["base_url"]
        if cfg["official"].get("tokenizer_path"):
            os.environ["REMOTE_OPENAI_TOKENIZER_PATH"] = cfg["official"]["tokenizer_path"]
        if registry not in mc.MODEL_CONFIG_MAPPING:
            # A model bfcl has no entry for (e.g. Qwen3.5) run through an existing bfcl
            # handler class, unchanged.
            mod_name, cls_name = cfg["official"]["handler_class"].rsplit(".", 1)
            import importlib

            cls = getattr(importlib.import_module(mod_name), cls_name)
            mc.MODEL_CONFIG_MAPPING[registry] = mc.ModelConfig(
                model_name=cfg["official"]["hf_model"],
                display_name=cfg.get("display_name") or registry,
                url="local",
                org="local",
                license="local",
                model_handler=cls,
                input_price=None,
                output_price=None,
                is_fc_model=bool(cfg["official"].get("is_fc_model", True)),
                underscore_to_dot=bool(cfg["official"].get("underscore_to_dot", False)),
            )
    return registry


def build_config(args: argparse.Namespace, run_dir: Path) -> dict[str, Any]:
    if args.handler == "native":
        if not args.base_url or not args.served_model:
            raise SystemExit("--handler native needs --base-url and --served-model")
        if args.thinking is None:
            raise SystemExit("--handler native needs --thinking on|off|default (see --help)")
        thinking = {"default": None, "on": True, "off": False}[args.thinking]
        ptc = {"unset": None, "true": True, "false": False}[args.parallel_tool_calls]
        endpoint = {
            "base_url": args.base_url.rstrip("/"),
            "served_model": args.served_model,
            "backend": args.backend,
            "api_key_env": args.api_key_env,
            "max_tokens": None if args.max_tokens <= 0 else args.max_tokens,
            "enable_thinking": thinking,
            "reasoning_history": args.reasoning_history,
            "parallel_tool_calls": ptc,
            "request_timeout_s": args.request_timeout,
            "extra_body": json.loads(args.extra_body) if args.extra_body else {},
        }
        if args.system_prefix:
            text, digest = load_system_prefix(Path(args.system_prefix))
            # Only present for the A1 arm, so older run_config.json files stay comparable.
            endpoint["system_prefix"] = text
            endpoint["system_prefix_sha256"] = digest
        registry = args.registry or f"harness/{_slug(run_dir.name)}-FC"
        cfg = {"handler": "native", "registry": registry, "endpoint": endpoint}
    else:
        if not args.official_registry or not args.base_url:
            raise SystemExit("--handler official needs --official-registry and --base-url")
        cfg = {
            "handler": "official",
            "registry": args.official_registry,
            "official": {
                "base_url": args.base_url.rstrip("/"),
                "tokenizer_path": args.tokenizer_path,
                "handler_class": args.official_handler_class,
                "hf_model": args.hf_model,
                "is_fc_model": True,
                "underscore_to_dot": False,
            },
        }
    cfg["display_name"] = args.display_name
    cfg["temperature"] = args.temperature
    cfg["categories_arg"] = args.categories
    cfg["per_category"] = args.per_category
    cfg["per_category_multi_turn"] = args.per_category_multi_turn
    cfg["sample_seed"] = args.sample_seed
    cfg["notes"] = args.notes
    cfg["harness"] = harness_fingerprint()
    return cfg


def load_system_prefix(path: Path) -> tuple[str, str]:
    """A1 system prefix: a .json with "system_prompt" (a1_exemplars.json, whose recorded
    "system_prompt_sha256" is verified) or a plain text file used verbatim."""
    import hashlib

    raw = path.read_text(encoding="utf-8")
    if path.suffix == ".json":
        data = json.loads(raw)
        text = data["system_prompt"]
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        want = data.get("system_prompt_sha256")
        if want and want != digest:
            raise SystemExit(f"{path}: system_prompt sha256 {digest} != recorded {want}")
        return text, digest
    return raw, hashlib.sha256(raw.encode("utf-8")).hexdigest()


def harness_fingerprint() -> dict[str, Any]:
    """sha256 of the harness code and the bfcl-eval version (V1: harness frozen)."""
    import hashlib
    from importlib import metadata

    files = {
        f.name: hashlib.sha256(f.read_bytes()).hexdigest()
        for f in (HERE / "run_bfcl.py", HERE / "native_fc_handler.py")
    }
    try:
        bfcl_version = metadata.version("bfcl-eval")
    except metadata.PackageNotFoundError:  # pragma: no cover
        bfcl_version = None
    return {"files_sha256": files, "bfcl_eval": bfcl_version}


# --------------------------------------------------------------------------- selection
def resolve_categories(category_args: list[str]) -> list[str]:
    from bfcl_eval.utils import parse_test_category_argument

    cats = parse_test_category_argument(category_args)
    return [c for c in cats if c not in SKIPPED_ALWAYS and c != "format_sensitivity"]


def select_entries(cfg: dict[str, Any], ids_file: str | None) -> tuple[list[str], list[dict]]:
    """bfcl's own loader (incl. language hints and memory prereqs), optionally subset.

    The chosen ids are frozen into run_config.json, so resumes and re-evaluations always
    use the same items."""
    from bfcl_eval.utils import is_memory_prereq, load_dataset_entry

    cats = resolve_categories(cfg["categories_arg"])
    entries: list[dict] = []
    for c in cats:
        entries.extend(load_dataset_entry(c))
    by_id = {e["id"]: e for e in entries}

    if "selected_ids" not in cfg:
        selected: dict[str, list[str]] | None = None
        if ids_file:
            selected = json.loads(Path(ids_file).read_text())
            if isinstance(selected, dict) and "selected_ids" in selected:
                # Another run's run_config.json: reuse exactly its frozen ids (all arms
                # reuse the A0 ids; PREREGISTRATION.md section 5).
                selected = selected["selected_ids"]
            if selected is not None:
                unknown = sorted(set(selected) - set(cats))
                missing_cats = sorted(set(cats) - set(selected))
                if unknown or missing_cats:
                    raise SystemExit(
                        f"--ids-file categories do not match --categories: not requested "
                        f"{unknown}, not in ids file {missing_cats}"
                    )
            cfg["ids_file"] = str(Path(ids_file).resolve())
        elif cfg.get("per_category") or cfg.get("per_category_multi_turn"):
            # --per-category-multi-turn alone subsets ONLY multi-turn; every other
            # category runs in full (the section-5 compute rule). Before this fix the
            # multi-turn override was ignored unless --per-category was also set.
            n = int(cfg.get("per_category") or 0)
            seed = int(cfg.get("sample_seed") or 0)
            selected = {}
            for c in cats:
                pool = [e["id"] for e in entries if _category_of(e["id"]) == c]
                pool = [i for i in pool if not is_memory_prereq(i)]
                k = n if n > 0 else len(pool)
                if c.startswith("multi_turn") and cfg.get("per_category_multi_turn"):
                    k = int(cfg["per_category_multi_turn"])
                # One RNG per category and a shuffled prefix: the same seed picks the same
                # items in a category whatever else the run includes, and a smaller N is a
                # subset of a larger N, so runs stay pairable.
                order = sorted(pool, key=_id_sort)
                random.Random(f"{seed}:{c}").shuffle(order)
                selected[c] = sorted(order[:k], key=_id_sort)
        cfg["selected_ids"] = selected

    selected = cfg["selected_ids"]
    if selected is None:
        return cats, entries

    keep: set[str] = set()
    stack = [i for ids in selected.values() for i in ids]
    while stack:  # memory questions need their prerequisite conversations
        i = stack.pop()
        if i in keep or i not in by_id:
            continue
        keep.add(i)
        stack.extend(by_id[i].get("depends_on", []) or [])
    missing = [i for ids in selected.values() for i in ids if i not in by_id]
    if missing:
        raise SystemExit(f"selected ids not in the dataset: {missing[:5]} ...")
    return cats, [e for e in entries if e["id"] in keep]


def _category_of(test_id: str) -> str:
    from bfcl_eval.utils import extract_test_category_from_id

    return extract_test_category_from_id(test_id)


def _id_sort(test_id: str):
    nums = [int(x) for x in re.findall(r"\d+", test_id)]
    return (re.sub(r"\d+", "", test_id), nums)


# --------------------------------------------------------------------------- generate
def drop_error_entries(result_dir: Path) -> int:
    from bfcl_eval.constants.eval_config import RESULT_FILE_PATTERN

    dropped = 0
    for f in result_dir.rglob(RESULT_FILE_PATTERN):
        lines = f.read_text().splitlines()
        keep = []
        for line in lines:
            if not line.strip():
                continue
            entry = json.loads(line)
            if isinstance(entry.get("result"), str) and entry["result"].startswith(ERROR_PREFIX):
                dropped += 1
                continue
            keep.append(line)
        f.write_text("".join(k + "\n" for k in keep))
    return dropped


def _lock_run_dir(run_dir: Path):
    """Exclusive per-run-dir lock. Two generators appending to the same result files
    produce duplicate ids, which bfcl's evaluator then rejects."""
    import fcntl

    fh = open(run_dir / ".generate.lock", "w")  # noqa: SIM115 - held for the process lifetime
    try:
        fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit(f"another generate is already running on {run_dir}") from None
    fh.write(str(os.getpid()))
    fh.flush()
    return fh


def cmd_generate(args: argparse.Namespace) -> None:
    run_dir = Path(args.run_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    _lock = _lock_run_dir(run_dir)  # noqa: F841 - released at process exit
    existing = load_run_config(run_dir)
    new_cfg = build_config(args, run_dir)
    if existing is not None:
        keys = (
            "handler",
            "registry",
            "endpoint",
            "official",
            "temperature",
            "categories_arg",
            "per_category",
            "per_category_multi_turn",
            "sample_seed",
            "harness",
        )
        diff = {
            k: (existing.get(k), new_cfg.get(k))
            for k in keys
            # unset and 0/empty are the same setting (older run_config.json lack new keys)
            if (existing.get(k) or None) != (new_cfg.get(k) or None)
        }
        if diff and not args.force_config:
            raise SystemExit(
                "run dir already has a different run_config.json (resume needs identical settings; "
                f"pass --force-config to override): {json.dumps(diff, default=str)}"
            )
        cfg = existing if not diff else {**new_cfg, "selected_ids": existing.get("selected_ids")}
    else:
        cfg = new_cfg

    registry = register_run(cfg)
    cats, entries = select_entries(cfg, args.ids_file)
    cfg["categories"] = cats
    save_run_config(run_dir, cfg)

    import multiprocessing as mp

    from bfcl_eval._llm_response_generation import collect_test_cases, generate_results
    from bfcl_eval.constants.eval_config import RESULT_FILE_PATTERN, RESULT_PATH
    from bfcl_eval.utils import sort_file_content_by_id

    # Same process-level settings as bfcl's generation main().
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    mp.set_start_method("spawn", force=True)

    if args.retry_errors:
        n = drop_error_entries(RESULT_PATH)
        print(f"[harness] dropped {n} errored entries for retry")

    ns = SimpleNamespace(
        model=[registry],
        test_category=cats,
        temperature=cfg["temperature"],
        include_input_log=args.include_input_log,
        exclude_state_log=args.exclude_state_log,
        num_gpus=1,
        num_threads=args.num_threads,
        gpu_memory_utilization=0.9,
        backend="vllm",
        skip_server_setup=True,  # never spawn vLLM/SGLang; the server is already running
        local_model_path=args.local_model_path,
        result_dir=RESULT_PATH,
        allow_overwrite=False,  # resume: keep what is there
        run_ids=False,
        enable_lora=False,
        max_lora_rank=None,
        lora_modules=None,
    )
    todo = collect_test_cases(ns, registry, cats, deepcopy(entries))
    print(
        f"[harness] registry={registry} categories={cats} selected={len(entries)} to_generate={len(todo)}"
    )
    if todo:
        generate_results(ns, registry, todo)
    for f in RESULT_PATH.rglob(RESULT_FILE_PATTERN):
        sort_file_content_by_id(f)


# --------------------------------------------------------------------------- evaluate
def cmd_evaluate(args: argparse.Namespace) -> None:
    run_dir = Path(args.run_dir).resolve()
    cfg = load_run_config(run_dir)
    if cfg is None:
        raise SystemExit(f"no run_config.json in {run_dir}")
    registry = register_run(cfg)
    # Grade only categories that have result files (bfcl's runner works per result file).
    import statistics

    from bfcl_eval.constants.eval_config import RESULT_PATH, SCORE_PATH
    from bfcl_eval.eval_checker.eval_runner import main as evaluation_main

    try:
        evaluation_main(
            [registry], cfg["categories"], str(RESULT_PATH), str(SCORE_PATH), partial_eval=True
        )
    except statistics.StatisticsError as exc:
        # bfcl's generate_leaderboard_csv takes stdev(latencies) and fails with < 2 latency
        # points (a 1-item run). The per-category score files are already written by then,
        # so summarize still works; only bfcl's data_*.csv (and bfcl_csv_check) are missing.
        print(f"[harness] bfcl leaderboard CSV skipped ({exc}); score files are complete")


# --------------------------------------------------------------------------- scoring
def _score_files(score_dir: Path) -> dict[str, Path]:
    from bfcl_eval.utils import extract_test_category

    return {extract_test_category(p): p for p in score_dir.rglob("BFCL_v4_*_score.json")}


def per_item_correctness(run_dir: Path, cfg: dict[str, Any]) -> list[dict[str, Any]]:
    """One row per graded item. bfcl's score files hold a header line followed by
    one line per FAILED item, so correct = graded ids minus failed ids."""
    from bfcl_eval.constants.eval_config import RESULT_FILE_PATTERN
    from bfcl_eval.utils import extract_test_category, is_memory_prereq

    reg_dir = cfg["registry"].replace("/", "_")
    result_dir = run_dir / "result" / reg_dir
    score_dir = run_dir / "score" / reg_dir
    scores = _score_files(score_dir)
    rows: list[dict[str, Any]] = []
    for rf in sorted(result_dir.rglob(RESULT_FILE_PATTERN)):
        cat = extract_test_category(rf)
        if is_memory_prereq(cat) or cat not in scores:
            continue
        lines = [json.loads(x) for x in scores[cat].read_text().splitlines() if x.strip()]
        header, failures = lines[0], lines[1:]
        failed = {f["id"]: f for f in failures if "id" in f}
        results = [json.loads(x) for x in rf.read_text().splitlines() if x.strip()]
        n_correct = 0
        for r in results:
            f = failed.get(r["id"])
            ok = f is None
            n_correct += ok
            err = (f or {}).get("error_type")
            if err is None and isinstance((f or {}).get("error"), dict):
                err = f["error"].get("error_type")  # agentic/memory checker nests it
            rows.append(
                {
                    "category": cat,
                    "id": r["id"],
                    "correct": int(ok),
                    "error_type": err or "",
                    "inference_error": int(
                        isinstance(r.get("result"), str) and r["result"].startswith(ERROR_PREFIX)
                    ),
                }
            )
        if n_correct != header.get("correct_count"):
            raise RuntimeError(
                f"{cat}: per-item correct {n_correct} != score header {header.get('correct_count')}"
            )
    return rows


_COUNT_CACHE: dict[str, int] = {}


def official_aggregate(cat_acc: dict[str, dict[str, float]]) -> dict[str, Any]:
    """The column formulas of bfcl's generate_leaderboard_csv, built from bfcl's own
    helpers (get_category_score / calculate_*). `cat_acc` maps category ->
    {"accuracy", "total_count"}. Missing categories count as 0, as in bfcl.
    Checked against bfcl's data_overall.csv in summarize()."""
    from bfcl_eval.eval_checker import eval_runner_helper as h

    def score(cat: str) -> dict:
        if cat in cat_acc:
            return {**cat_acc[cat], "display_accuracy": cat_acc[cat]["accuracy"]}
        if cat not in _COUNT_CACHE:
            _COUNT_CACHE[cat] = h.get_category_score({}, cat)["total_count"]
        return {"accuracy": 0, "total_count": _COUNT_CACHE[cat], "display_accuracy": "N/A"}

    s = {
        c: score(c)
        for c in (
            "simple_python",
            "simple_java",
            "simple_javascript",
            "multiple",
            "parallel",
            "parallel_multiple",
            "irrelevance",
            "live_simple",
            "live_multiple",
            "live_parallel",
            "live_parallel_multiple",
            "live_irrelevance",
            "live_relevance",
            "multi_turn_base",
            "multi_turn_miss_func",
            "multi_turn_miss_param",
            "multi_turn_long_context",
            "web_search_base",
            "web_search_no_snippet",
            "memory_kv",
            "memory_vector",
            "memory_rec_sum",
        )
    }
    uw, w, pw = (
        h.calculate_unweighted_accuracy,
        h.calculate_weighted_accuracy,
        h.calculate_percentage_weighted_accuracy,
    )
    simple_nl = uw([s["simple_python"], s["simple_java"], s["simple_javascript"]])
    nl_parts = [simple_nl, s["multiple"], s["parallel"], s["parallel_multiple"]]
    non_live = uw(nl_parts, display_na_if_category_missing=False)
    live_parts = [
        s["live_simple"],
        s["live_multiple"],
        s["live_parallel"],
        s["live_parallel_multiple"],
    ]
    live = w(live_parts, display_na_if_category_missing=False)
    mt = uw(
        [
            s["multi_turn_base"],
            s["multi_turn_miss_func"],
            s["multi_turn_miss_param"],
            s["multi_turn_long_context"],
        ],
        display_na_if_category_missing=False,
    )
    ws = uw([s["web_search_base"], s["web_search_no_snippet"]])
    mem = uw([s["memory_kv"], s["memory_vector"], s["memory_rec_sum"]])
    agentic = uw([ws, mem], display_na_if_category_missing=False)
    irrel = uw([s["irrelevance"], s["live_irrelevance"]])
    overall = pw(
        [non_live, live, irrel, mt, agentic],
        [10, 10, 10, 30, 40],
        display_na_if_category_missing=False,
    )
    # Not an official number: the same weights without the web-search half of agentic,
    # renormalized (non_live 10, live 10, irrelevance 10, multi_turn 30, memory 20).
    ran_mem = any(c in cat_acc for c in ("memory_kv", "memory_vector", "memory_rec_sum"))
    no_ws_parts = [non_live, live, irrel, mt] + ([mem] if ran_mem else [])
    no_ws_w = [10, 10, 10, 30] + ([20] if ran_mem else [])
    overall_no_ws = pw(no_ws_parts, no_ws_w, display_na_if_category_missing=False)

    pct = lambda d: None if d["display_accuracy"] == "N/A" else round(100 * d["accuracy"], 2)  # noqa: E731
    return {
        "Overall Acc": round(100 * overall["accuracy"], 2),
        "Overall (no web search, renormalized; NOT official)": round(
            100 * overall_no_ws["accuracy"], 2
        ),
        "Non-Live AST Acc": pct(uw(nl_parts)),
        "Non-Live Simple AST": pct(simple_nl),
        "Non-Live Multiple AST": pct(s["multiple"]),
        "Non-Live Parallel AST": pct(s["parallel"]),
        "Non-Live Parallel Multiple AST": pct(s["parallel_multiple"]),
        "Live Acc": round(100 * live["accuracy"], 2),
        "Live Simple AST": pct(s["live_simple"]),
        "Live Multiple AST": pct(s["live_multiple"]),
        "Live Parallel AST": pct(s["live_parallel"]),
        "Live Parallel Multiple AST": pct(s["live_parallel_multiple"]),
        "Multi Turn Acc": round(100 * mt["accuracy"], 2),
        "Multi Turn Base": pct(s["multi_turn_base"]),
        "Multi Turn Miss Func": pct(s["multi_turn_miss_func"]),
        "Multi Turn Miss Param": pct(s["multi_turn_miss_param"]),
        "Multi Turn Long Context": pct(s["multi_turn_long_context"]),
        "Web Search Acc": pct(ws),
        "Memory Acc": pct(mem),
        "Memory KV": pct(s["memory_kv"]),
        "Memory Vector": pct(s["memory_vector"]),
        "Memory Recursive Summarization": pct(s["memory_rec_sum"]),
        "Relevance Detection": pct(s["live_relevance"]),
        "Irrelevance Detection": pct(irrel),
        "_components": {
            "non_live": non_live["accuracy"],
            "live": live["accuracy"],
            "irrelevance": irrel["accuracy"],
            "multi_turn": mt["accuracy"],
            "web_search": ws["accuracy"],
            "memory": mem["accuracy"],
        },
    }


def _pct_to_float(v: str) -> float | None:
    v = (v or "").strip()
    if not v or v == "N/A":
        return None
    return float(v.rstrip("%"))


def load_official_row(csv_path: Path, display_name: str) -> dict[str, Any] | None:
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            if row.get("Model") == display_name:
                return row
    return None


def cmd_summarize(args: argparse.Namespace) -> dict[str, Any]:
    run_dir = Path(args.run_dir).resolve()
    cfg = load_run_config(run_dir)
    if cfg is None:
        raise SystemExit(f"no run_config.json in {run_dir}")
    register_run(cfg)
    from bfcl_eval.utils import load_dataset_entry

    rows = per_item_correctness(run_dir, cfg)
    by_cat: dict[str, list[int]] = {}
    for r in rows:
        by_cat.setdefault(r["category"], []).append(r["correct"])
    cat_acc = {c: {"accuracy": sum(v) / len(v), "total_count": len(v)} for c, v in by_cat.items()}
    coverage = {}
    for c in sorted(by_cat):
        full = len(
            load_dataset_entry(c, include_prereq=False, include_language_specific_hint=False)
        )
        coverage[c] = {
            "n": len(by_cat[c]),
            "full": full,
            "correct": sum(by_cat[c]),
            "acc": round(100 * cat_acc[c]["accuracy"], 2),
            "inference_errors": sum(r["inference_error"] for r in rows if r["category"] == c),
        }
    agg = official_aggregate(cat_acc)
    components = agg.pop("_components")

    # Check against bfcl's own data_overall.csv (written by `evaluate`).
    bfcl_csv = run_dir / "score" / "data_overall.csv"
    bfcl_check = None
    if bfcl_csv.exists():
        with open(bfcl_csv, newline="") as f:
            csv_rows = list(csv.DictReader(f))
        want = cfg.get("display_name") or cfg["registry"]
        match = [r for r in csv_rows if r["Model"] == want] or (
            csv_rows if len(csv_rows) == 1 else []
        )
        if match:
            bfcl_cols = {k: _pct_to_float(v) for k, v in match[0].items() if k in agg}
            mismatched = {
                k: (v, agg[k])
                for k, v in bfcl_cols.items()
                if v is not None and agg[k] is not None and abs(v - agg[k]) > 0.011
            }
            bfcl_check = {
                "bfcl_overall": bfcl_cols.get("Overall Acc"),
                "harness_overall": agg["Overall Acc"],
                "columns_compared": len(bfcl_cols),
                "mismatched": mismatched,
            }

    subset = any(v["n"] < v["full"] for v in coverage.values())
    summary: dict[str, Any] = {
        "run_dir": str(run_dir),
        "registry": cfg["registry"],
        "handler": cfg["handler"],
        "endpoint": cfg.get("endpoint") or cfg.get("official"),
        "temperature": cfg.get("temperature"),
        "categories_run": sorted(by_cat),
        "categories_not_run": [c for c in _all_scoring() if c not in by_cat],
        "subset": subset,
        "coverage": coverage,
        "columns": agg,
        "components": components,
        "bfcl_csv_check": bfcl_check,
    }

    if args.reference_model:
        ref = load_official_row(Path(args.official_csv), args.reference_model)
        if ref is None:
            print(f"[harness] reference model '{args.reference_model}' not in {args.official_csv}")
        else:
            cmp = {}
            for col, ours in agg.items():
                theirs = _pct_to_float(ref.get(col, "")) if col in ref else None
                if ours is not None and theirs is not None:
                    cmp[col] = {"ours": ours, "official": theirs, "delta": round(ours - theirs, 2)}
            # Diagnostic only: our overall with the official row's web-search accuracy (and
            # memory, when not run) plugged in, to compare like for like.
            ws_off = (_pct_to_float(ref["Web Search Acc"]) or 0) / 100
            mem_off = (_pct_to_float(ref["Memory Acc"]) or 0) / 100
            mem = components["memory"] if any(c.startswith("memory_") for c in by_cat) else mem_off
            imputed = 100 * (
                0.1 * components["non_live"]
                + 0.1 * components["live"]
                + 0.1 * components["irrelevance"]
                + 0.3 * components["multi_turn"]
                + 0.4 * (ws_off + mem) / 2
            )
            summary["reference"] = {
                "model": args.reference_model,
                "official_overall": _pct_to_float(ref["Overall Acc"]),
                "overall_with_official_web_search_and_missing_memory_imputed": round(imputed, 2),
                "columns": cmp,
            }

    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    with open(run_dir / "per_item.csv", "w", newline="") as f:
        wr = csv.DictWriter(
            f, fieldnames=["registry", "category", "id", "correct", "error_type", "inference_error"]
        )
        wr.writeheader()
        for r in rows:
            wr.writerow({"registry": cfg["registry"], **r})
    _print_summary(summary)
    return summary


def _all_scoring() -> list[str]:
    from bfcl_eval.constants.category_mapping import ALL_SCORING_CATEGORIES

    return list(ALL_SCORING_CATEGORIES)


def _print_summary(s: dict[str, Any]) -> None:
    print(f"\n== {s['registry']}  ({'SUBSET' if s['subset'] else 'FULL'})  handler={s['handler']}")
    print(f"{'category':28s} {'n':>5s} {'full':>5s} {'acc%':>7s} {'err':>4s}")
    for c, v in s["coverage"].items():
        print(f"{c:28s} {v['n']:5d} {v['full']:5d} {v['acc']:7.2f} {v['inference_errors']:4d}")
    print("-- official-style columns (bfcl formulas; categories not run count as 0 in Overall)")
    for k, v in s["columns"].items():
        print(f"  {k:52s} {'N/A' if v is None else f'{v:.2f}'}")
    if s.get("bfcl_csv_check"):
        print(f"  bfcl data_overall.csv check: {s['bfcl_csv_check']}")
    if s.get("reference"):
        r = s["reference"]
        print(f"-- vs official '{r['model']}' (official Overall {r['official_overall']})")
        print(
            f"  overall with official web-search/memory imputed: {r['overall_with_official_web_search_and_missing_memory_imputed']}"
        )
        for k, v in r["columns"].items():
            print(
                f"  {k:52s} ours {v['ours']:6.2f}  official {v['official']:6.2f}  delta {v['delta']:+6.2f}"
            )


# --------------------------------------------------------------------------- pair / bootstrap
# Endpoint keys that may legitimately differ between two arms of one comparison. Every
# other setting (decoding, thinking, max_tokens, backend, timeout, extra fields, items,
# harness code) must match, or `pair` refuses unless the key is passed to
# --allow-config-diff (e.g. the V5 A2-direct fallback pairs backend llama-server runs).
_ARM_KEYS = ("base_url", "served_model", "api_key_env", "system_prefix", "system_prefix_sha256")


def config_differences(cfg_a: dict[str, Any], cfg_b: dict[str, Any]) -> dict[str, Any]:
    diff: dict[str, Any] = {}
    for k in ("handler", "temperature", "selected_ids", "categories", "harness"):
        if (cfg_a.get(k) or None) != (cfg_b.get(k) or None):
            diff[k] = (
                "differs" if k in ("selected_ids", "harness") else (cfg_a.get(k), cfg_b.get(k))
            )
    ea = cfg_a.get("endpoint") or cfg_a.get("official") or {}
    eb = cfg_b.get("endpoint") or cfg_b.get("official") or {}
    for k in sorted(set(ea) | set(eb)):
        if k in _ARM_KEYS:
            continue
        if (ea.get(k) or None) != (eb.get(k) or None):
            diff[f"endpoint.{k}"] = (ea.get(k), eb.get(k))
    return diff


def mcnemar_exact(b01: int, b10: int) -> float:
    """Two-sided exact binomial p-value on the discordant pairs."""
    from math import comb

    n = b01 + b10
    if n == 0:
        return 1.0
    k = min(b01, b10)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2**n)


def holm(pvals: dict[str, float]) -> dict[str, float]:
    order = sorted(pvals, key=lambda c: pvals[c])
    m, running, out = len(order), 0.0, {}
    for rank, c in enumerate(order):
        running = max(running, min(1.0, (m - rank) * pvals[c]))
        out[c] = running
    return out


def errored_ids(run_dir: Path) -> set[tuple[str, str]]:
    cfg = load_run_config(run_dir)
    register_run(cfg)
    return {
        (r["category"], r["id"]) for r in per_item_correctness(run_dir, cfg) if r["inference_error"]
    }


def cmd_pair(args: argparse.Namespace) -> None:
    import numpy as np

    runs = []
    for rd in (args.run_a, args.run_b):
        rd = Path(rd).resolve()
        cfg = load_run_config(rd)
        if cfg is None:
            raise SystemExit(f"no run_config.json in {rd}")
        register_run(cfg)
        rows = per_item_correctness(rd, cfg)
        runs.append((rd, cfg, {(r["category"], r["id"]): r for r in rows}))
    (_, cfg_a, ra), (_, cfg_b, rb) = runs

    # Comparability: identical settings except the arm-identifying fields.
    diff = config_differences(cfg_a, cfg_b)
    allowed = set(args.allow_config_diff or [])
    blocking = {k: v for k, v in diff.items() if k not in allowed}
    if blocking:
        raise SystemExit(
            "runs are not comparable arms (pass --allow-config-diff KEY to accept a listed "
            f"difference): {json.dumps(blocking, default=str)}"
        )

    # Same items: a silently shorter arm would bias the comparison.
    only = {"only_a": len(set(ra) - set(rb)), "only_b": len(set(rb) - set(ra))}
    if (only["only_a"] or only["only_b"]) and not args.allow_unmatched:
        raise SystemExit(f"runs graded different item sets {only}; pass --allow-unmatched")

    excluded: set[tuple[str, str]] = set()
    if args.exclude_errors:
        excluded |= {k for k, r in ra.items() if r["inference_error"]}
        excluded |= {k for k, r in rb.items() if r["inference_error"]}
        for extra in args.exclude_errors_from or []:
            excluded |= errored_ids(Path(extra).resolve())
    a = {k: r["correct"] for k, r in ra.items()}
    b = {k: r["correct"] for k, r in rb.items()}
    keys = sorted((set(a) & set(b)) - excluded, key=lambda k: (k[0], _id_sort(k[1])))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(
            ["category", "id", "a_correct", "b_correct", "a_inference_error", "b_inference_error"]
        )
        for k in keys:
            wr.writerow(
                [k[0], k[1], a[k], b[k], ra[k]["inference_error"], rb[k]["inference_error"]]
            )

    cats = sorted({k[0] for k in keys})
    idx = {c: [i for i, k in enumerate(keys) if k[0] == c] for c in cats}
    A = np.array([a[k] for k in keys], dtype=float)
    B = np.array([b[k] for k in keys], dtype=float)

    def agg(vec, sel):
        cat_acc = {
            c: {"accuracy": float(vec[sel[c]].mean()), "total_count": len(sel[c])} for c in cats
        }
        res = official_aggregate(cat_acc)
        comp = res.pop("_components")
        return {
            "Overall (no web search, renormalized; NOT official)": res[
                "Overall (no web search, renormalized; NOT official)"
            ],
            "Overall Acc": res["Overall Acc"],
            "non_live": 100 * comp["non_live"],
            "live": 100 * comp["live"],
            "irrelevance": 100 * comp["irrelevance"],
            "multi_turn": 100 * comp["multi_turn"],
            **{c: 100 * cat_acc[c]["accuracy"] for c in cats},
        }

    sel0 = {c: np.array(idx[c]) for c in cats}
    pa, pb = agg(A, sel0), agg(B, sel0)
    rng = np.random.default_rng(args.seed)
    deltas: dict[str, list[float]] = {k: [] for k in pa}
    for _ in range(args.bootstrap):
        sel = {c: rng.choice(sel0[c], size=len(sel0[c]), replace=True) for c in cats}
        ra_, rb_ = agg(A, sel), agg(B, sel)
        for k in deltas:
            deltas[k].append(rb_[k] - ra_[k])
    report: dict[str, Any] = {
        "a": cfg_a["registry"],
        "b": cfg_b["registry"],
        "n_paired": len(keys),
        **only,
        "excluded_errored_items": len(excluded & (set(a) | set(b))),
        "config_differences_allowed": {k: v for k, v in diff.items() if k in allowed},
        "bootstrap": args.bootstrap,
        "seed": args.seed,
        "metrics": {},
    }
    for k in pa:
        d = np.array(deltas[k])
        report["metrics"][k] = {
            "a": round(pa[k], 2),
            "b": round(pb[k], 2),
            "delta": round(pb[k] - pa[k], 2),
            "ci95": [
                round(float(np.percentile(d, 2.5)), 2),
                round(float(np.percentile(d, 97.5)), 2),
            ],
            # Bonferroni for the two H1 endpoints (PREREGISTRATION.md section 6.2).
            "ci97_5": [
                round(float(np.percentile(d, 1.25)), 2),
                round(float(np.percentile(d, 98.75)), 2),
            ],
        }
    per_cat: dict[str, Any] = {}
    for c in cats:
        ia = A[sel0[c]]
        ib = B[sel0[c]]
        b_only = int(((ia == 0) & (ib == 1)).sum())
        a_only = int(((ia == 1) & (ib == 0)).sum())
        per_cat[c] = {
            "n": int(len(ia)),
            "b_wins": b_only,
            "ties": int((ia == ib).sum()),
            "b_losses": a_only,
            "mcnemar_p": mcnemar_exact(b_only, a_only),
        }
    adj = holm({c: v["mcnemar_p"] for c, v in per_cat.items()})
    for c in per_cat:
        per_cat[c]["mcnemar_p_holm"] = adj[c]
    report["per_category"] = per_cat
    report["discordant"] = {
        "a_only_correct": int(((A == 1) & (B == 0)).sum()),
        "b_only_correct": int(((A == 0) & (B == 1)).sum()),
    }
    out.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


# --------------------------------------------------------------------------- routing audit
def cmd_routing(args: argparse.Namespace) -> None:
    """Per-item routing record from harness_trace: how often the RouterGate fired.

    Writes <run-dir>/routing.csv (category, id, steps, fired_steps, units, fired_class)
    and routing.json (per-category request and item activation rates). With
    --expect-unit U, any routed unit other than U (or a missing subcor field) is an
    error: it proves the tenant had exactly one unit and that routing really happened
    inside SUBCOR for every request."""
    from bfcl_eval.constants.eval_config import RESULT_FILE_PATTERN
    from bfcl_eval.utils import extract_test_category, is_memory_prereq

    run_dir = Path(args.run_dir).resolve()
    cfg = load_run_config(run_dir)
    if cfg is None:
        raise SystemExit(f"no run_config.json in {run_dir}")
    result_dir = run_dir / "result" / cfg["registry"].replace("/", "_")
    rows: list[dict[str, Any]] = []
    problems: list[str] = []
    for rf in sorted(result_dir.rglob(RESULT_FILE_PATTERN)):
        cat = extract_test_category(rf)
        if is_memory_prereq(cat):
            continue
        for line in rf.read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            if isinstance(r.get("result"), str) and r["result"].startswith(ERROR_PREFIX):
                rows.append(
                    {
                        "category": cat,
                        "id": r["id"],
                        "steps": 0,
                        "fired_steps": 0,
                        "units": "",
                        "fired_class": "error",
                    }
                )
                continue
            steps = r.get("harness_trace") or []
            fired = 0
            units: set[str] = set()
            for st in steps:
                sub = st.get("subcor")
                if not isinstance(sub, dict):
                    problems.append(f"{r['id']}: step without a subcor routing block")
                    continue
                sel = list(sub.get("selected_units") or [])
                fired += bool(sel)
                units.update(sel)
            if not steps:
                problems.append(f"{r['id']}: no harness_trace")
            if args.expect_unit and units - {args.expect_unit}:
                problems.append(f"{r['id']}: routed unexpected units {sorted(units)}")
            cls = "none" if fired == 0 else ("all" if fired == len(steps) else "some")
            rows.append(
                {
                    "category": cat,
                    "id": r["id"],
                    "steps": len(steps),
                    "fired_steps": fired,
                    "units": ",".join(sorted(units)),
                    "fired_class": cls,
                }
            )
    with open(run_dir / "routing.csv", "w", newline="") as f:
        wr = csv.DictWriter(
            f, fieldnames=["category", "id", "steps", "fired_steps", "units", "fired_class"]
        )
        wr.writeheader()
        wr.writerows(rows)
    per_cat: dict[str, Any] = {}
    for c in sorted({r["category"] for r in rows}):
        rc = [r for r in rows if r["category"] == c]
        n_steps = sum(r["steps"] for r in rc)
        per_cat[c] = {
            "items": len(rc),
            "requests": n_steps,
            "request_activation_rate": round(sum(r["fired_steps"] for r in rc) / n_steps, 4)
            if n_steps
            else None,
            "items_by_fired_class": {
                k: sum(1 for r in rc if r["fired_class"] == k)
                for k in ("all", "some", "none", "error")
            },
        }
    summary = {
        "run_dir": str(run_dir),
        "expect_unit": args.expect_unit,
        "per_category": per_cat,
        "problems": problems[:50],
        "n_problems": len(problems),
    }
    if args.compare_run:
        # V9/V10: an item on which the gate never fired must reproduce the base arm's
        # output exactly (same server, same request, no adapter), and an item on which it
        # fired on every step must reproduce the always-on arm built from the same adapter.
        summary["identity_vs_compare_run"] = _identity_by_class(
            rows, result_dir, Path(args.compare_run).resolve()
        )
    (run_dir / "routing.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    if problems:
        raise SystemExit(
            f"[harness] routing audit found {len(problems)} problems (see routing.json)"
        )


def _result_lines(result_dir: Path) -> dict[str, Any]:
    from bfcl_eval.constants.eval_config import RESULT_FILE_PATTERN

    out: dict[str, Any] = {}
    for rf in result_dir.rglob(RESULT_FILE_PATTERN):
        for line in rf.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                out[r["id"]] = r.get("result")
    return out


def _identity_by_class(
    rows: list[dict[str, Any]], result_dir: Path, other_run: Path
) -> dict[str, Any]:
    ocfg = load_run_config(other_run)
    if ocfg is None:
        raise SystemExit(f"no run_config.json in {other_run}")
    mine = _result_lines(result_dir)
    theirs = _result_lines(other_run / "result" / ocfg["registry"].replace("/", "_"))
    out: dict[str, Any] = {"compare_run": str(other_run)}
    for cls in ("none", "all", "some"):
        ids = [r["id"] for r in rows if r["fired_class"] == cls and r["id"] in theirs]
        same = sum(
            json.dumps(mine.get(i), sort_keys=True) == json.dumps(theirs[i], sort_keys=True)
            for i in ids
        )
        out[cls] = {
            "items": len(ids),
            "identical": same,
            "identical_rate": round(same / len(ids), 4) if ids else None,
        }
    return out


# --------------------------------------------------------------------------- selftest
def _oracle_pick(options):
    """First accepted value that is not bfcl's "may be omitted" marker ("")."""
    return next((x for x in options if x != ""), "")


def _oracle_value(v, lang, top=True):
    """One accepted value from a bfcl possible-answer value. Nested dicts hold lists of
    accepted values. Java/JS scalar answers are compared as source strings."""
    if isinstance(v, dict):
        v = {
            k: _oracle_value(_oracle_pick(x), lang, False)
            for k, x in v.items()
            if isinstance(x, list) and _oracle_pick(x) != ""
        }
    elif isinstance(v, list):
        v = [_oracle_value(x, lang, False) for x in v]
    if (
        top
        and getattr(lang, "name", "") in ("JAVA", "JAVASCRIPT")
        and not isinstance(v, (str, dict, list))
    ):
        return json.dumps(v) if isinstance(v, bool) or v is None else str(v)
    return v


def cmd_selftest(args: argparse.Namespace) -> None:
    """Oracle check of the native path with no model or server involved.

    For single-turn items it builds a fake chat-completions response whose tool_calls are
    the ground truth: the first accepted value per parameter, and optional parameters
    left out. The response goes through NativeFCHandler's parser and decode_ast, then
    through bfcl's own _evaluate_single_ast_entry / relevance checker. Anything short of
    ~100% means the harness (name mangling, argument encoding) mis-converts calls."""
    from types import SimpleNamespace as NS

    from bfcl_eval.constants.enums import Language, ReturnFormat
    from bfcl_eval.eval_checker import eval_runner as er
    from bfcl_eval.utils import is_java, is_js, load_dataset_entry, load_ground_truth_entry

    cfg = {
        "handler": "native",
        "registry": "harness/selftest-FC",
        "display_name": "selftest",
        "endpoint": {"base_url": "http://127.0.0.1:9/v1", "served_model": "x"},
    }
    registry = register_run(cfg)
    handler = er.get_handler(registry)
    model_dir = registry.replace("/", "_")

    def fake(calls=None, text=None):
        tcs = None
        if calls is not None:
            tcs = [
                NS(id=f"c{i}", function=NS(name=n, arguments=json.dumps(a)))
                for i, (n, a) in enumerate(calls)
            ]
        msg = NS(content=text if calls is None else "", tool_calls=tcs, model_extra={})
        return NS(
            choices=[NS(message=msg, finish_reason="stop")],
            usage=NS(prompt_tokens=1, completion_tokens=1),
            model_extra={},
        )

    cats = [
        c for c in resolve_categories(args.categories) if not c.startswith(("multi_turn", "memory"))
    ]
    total_bad = 0
    for cat in cats:
        prompts = load_dataset_entry(
            cat, include_prereq=False, include_language_specific_hint=False
        )
        n = len(prompts) if args.limit <= 0 else min(args.limit, len(prompts))
        bad = []
        skipped = 0
        if "relevance" in cat:  # irrelevance / live_irrelevance / live_relevance
            for pe in prompts[:n]:
                if "irrelevance" in cat:
                    resp = fake(text="I cannot help with that using the given tools.")
                else:
                    f0 = pe["function"][0]
                    req = f0["parameters"].get("required", [])
                    resp = fake(calls=[(re.sub(r"\.", "_", f0["name"]), {k: "x" for k in req})])
                out = handler._parse_query_response_FC(resp)["model_responses"]
                r = er._evaluate_single_relevance_entry(handler, pe["id"], out, pe, model_dir, cat)
                if not r["valid"]:
                    bad.append(pe["id"])
        else:
            gts = load_ground_truth_entry(cat)
            lang, fmt = (
                (Language.JAVA, ReturnFormat.JAVA)
                if is_java(cat)
                else (Language.JAVASCRIPT, ReturnFormat.JAVASCRIPT)
                if is_js(cat)
                else (Language.PYTHON, ReturnFormat.PYTHON)
            )
            for pe, gt in list(zip(prompts, gts, strict=True))[:n]:
                if lang in (Language.JAVA, Language.JAVASCRIPT) and any(
                    isinstance(_oracle_pick(v), (dict, list))
                    for call in gt["ground_truth"]
                    for params in call.values()
                    for v in params.values()
                ):
                    skipped += 1  # would need Java/JS source rendering of containers
                    continue
                calls = []
                for call in gt["ground_truth"]:
                    ((name, params),) = call.items()
                    args_ = {
                        k: _oracle_value(_oracle_pick(v), lang)
                        for k, v in params.items()
                        if _oracle_pick(v) != ""
                    }
                    calls.append((re.sub(r"\.", "_", name), args_))
                out = handler._parse_query_response_FC(fake(calls=calls))["model_responses"]
                r = er._evaluate_single_ast_entry(
                    handler,
                    pe["id"],
                    out,
                    gt["ground_truth"],
                    pe,
                    model_dir,
                    cat,
                    language=lang,
                    return_format=fmt,
                    has_tool_call_tag=False,
                )
                if not r["valid"]:
                    bad.append((pe["id"], r.get("error_type"), str(r.get("error"))[:160]))
        total_bad += len(bad)
        m = n - skipped
        print(
            f"{cat:28s} oracle {m - len(bad):5d}/{m:<5d} pass"
            + (f" ({skipped} skipped)" if skipped else "")
            + (f"   e.g. {bad[:2]}" if bad else "")
        )
    print(f"[selftest] failures: {total_bad}")


# --------------------------------------------------------------------------- CLI
def _add_gen_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--handler", choices=["native", "official"], default="native")
    p.add_argument("--base-url", help="OpenAI-compatible base URL ending in /v1")
    p.add_argument(
        "--served-model", help="native: the 'model' field (e.g. subcor/base, subcor/<tenant>)"
    )
    p.add_argument("--backend", choices=["llama-server", "subcor"], default="llama-server")
    p.add_argument(
        "--registry", help="native: bfcl registry name (default harness/<run-dir>-FC), no '_'"
    )
    p.add_argument("--display-name")
    p.add_argument("--api-key-env", help="env var holding a bearer key (SUBCOR auth)")
    p.add_argument(
        "--max-tokens", type=int, default=4096, help="<=0 sends none (official OpenAI handler)"
    )
    p.add_argument(
        "--thinking",
        choices=["default", "on", "off"],
        default=None,
        help="native, required: on/off sends enable_thinking explicitly. 'default' sends nothing: "
        "llama-server b12110 then THINKS (its --reasoning auto), SUBCOR serve does NOT",
    )
    p.add_argument("--reasoning-history", choices=["keep", "strip"], default="keep")
    p.add_argument("--parallel-tool-calls", choices=["unset", "true", "false"], default="unset")
    p.add_argument("--request-timeout", type=float, default=1800.0)
    p.add_argument(
        "--extra-body", help="JSON merged into every request body (e.g. llama-server lora)"
    )
    p.add_argument(
        "--system-prefix",
        help="native: A1 arm only. a1_exemplars.json (its system_prompt, sha256-checked) or a "
        "text file, prepended as the first system message of every item",
    )
    p.add_argument(
        "--official-registry",
        help="official: bfcl registry name, e.g. Qwen/Qwen3-4B-Instruct-2507-FC",
    )
    p.add_argument(
        "--official-handler-class",
        default="bfcl_eval.model_handler.local_inference.qwen_fc.QwenFCHandler",
        help="official: handler class when the registry name is not in bfcl",
    )
    p.add_argument(
        "--hf-model", help="official: HF id/path for the tokenizer+config (new registry names)"
    )
    p.add_argument("--tokenizer-path", help="official: REMOTE_OPENAI_TOKENIZER_PATH")
    p.add_argument("--local-model-path", help="official: local HF dir with config.json + tokenizer")
    p.add_argument(
        "--categories",
        nargs="+",
        default=DEFAULT_CATEGORIES,
        help="bfcl categories/groups (web_search is always skipped)",
    )
    p.add_argument(
        "--per-category", type=int, default=0, help="seeded random subset per category (0 = all)"
    )
    p.add_argument(
        "--per-category-multi-turn",
        type=int,
        default=0,
        help="override --per-category for multi_turn_* categories",
    )
    p.add_argument("--sample-seed", type=int, default=0)
    p.add_argument(
        "--ids-file",
        help="JSON {category: [ids]} subset, or another run's run_config.json (its frozen "
        "selected_ids), instead of --per-category",
    )
    p.add_argument(
        "--temperature", type=float, default=0.001, help="bfcl generate default is 0.001"
    )
    p.add_argument(
        "--num-threads", type=int, default=1, help="concurrent requests (match server -np)"
    )
    p.add_argument("--retry-errors", action="store_true", help="regenerate items that errored")
    p.add_argument("--include-input-log", action="store_true")
    p.add_argument("--exclude-state-log", action="store_true")
    p.add_argument("--force-config", action="store_true")
    p.add_argument("--notes", default="")


def _add_sum_args(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--reference-model", help="official board display name, e.g. 'Qwen3-4B-Instruct-2507 (FC)'"
    )
    p.add_argument("--official-csv", default=str(HERE / "official_data_overall.csv"))


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("generate")
    g.add_argument("--run-dir", required=True)
    _add_gen_args(g)
    e = sub.add_parser("evaluate")
    e.add_argument("--run-dir", required=True)
    s = sub.add_parser("summarize")
    s.add_argument("--run-dir", required=True)
    _add_sum_args(s)
    a = sub.add_parser("all")
    a.add_argument("--run-dir", required=True)
    _add_gen_args(a)
    _add_sum_args(a)
    t = sub.add_parser("selftest", help="oracle check of the native conversion path (no server)")
    t.add_argument("--categories", nargs="+", default=["non_live", "live"])
    t.add_argument("--limit", type=int, default=0, help="items per category (0 = all)")
    p = sub.add_parser("pair")
    p.add_argument("--run-a", required=True)
    p.add_argument("--run-b", required=True)
    p.add_argument(
        "--out", required=True, help="paired CSV path (a .json report is written next to it)"
    )
    p.add_argument("--bootstrap", type=int, default=10000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--allow-config-diff",
        action="append",
        help="accept this differing setting (e.g. endpoint.backend for the V5 fallback)",
    )
    p.add_argument("--allow-unmatched", action="store_true", help="pair the item intersection")
    p.add_argument(
        "--exclude-errors",
        action="store_true",
        help="sensitivity analysis: drop items with an inference error in A or B "
        "(and in every --exclude-errors-from run)",
    )
    p.add_argument("--exclude-errors-from", nargs="*", default=[])
    r = sub.add_parser("routing", help="per-item RouterGate activation from harness_trace")
    r.add_argument("--run-dir", required=True)
    r.add_argument("--expect-unit", help="the one unit id the tenant may route to")
    r.add_argument(
        "--compare-run",
        help="another run dir (e.g. A0, or A2): identical-output rate per fired class (V9/V10)",
    )
    args = ap.parse_args(argv)

    if args.cmd == "selftest":
        _set_project_root(Path(os.environ.get("TMPDIR", "/tmp")) / "bfcl-harness-selftest")
        cmd_selftest(args)
        return
    if args.cmd == "pair":
        # Both runs are registered in one process; bfcl only needs some project root.
        _set_project_root(Path(args.out).resolve().parent / ".bfcl-root")
        cmd_pair(args)
    elif args.cmd == "generate":
        cmd_generate(args)
    elif args.cmd == "evaluate":
        cmd_evaluate(args)
    elif args.cmd == "summarize":
        cmd_summarize(args)
    elif args.cmd == "routing":
        cmd_routing(args)
    elif args.cmd == "all":
        cmd_generate(args)
        cmd_evaluate(args)
        cmd_summarize(args)


if __name__ == "__main__":
    main()
