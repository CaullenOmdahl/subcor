"""Measure llama-server prompt-cache reuse across Skill Unit switches, plus a low-VRAM profile.

llama-server clears a slot's cached prompt whenever a request's LoRA set changes
(tools/server/server-context.cpp, `lora_should_clear_cache`). SUBCOR re-routes every turn
and sends the full adapter list with scale 1/0, so a unit switch (or unit <-> base) makes
the server re-read the whole conversation. This script measures that cost through the
real supervised `LlamaCppServingAdapter` path, with synthetic random LoRA units (the
cache behaviour does not depend on adapter content).

Modes:
  switch      warm a growing conversation, then probe one turn per condition at each size
  regression  greedy text + first-token logprobs on fixed prompts (base and one unit)
  edge        VRAM / decode tok/s / cold TTFT for low-VRAM configs (partial -ngl etc.)

Conversation text is Python stdlib source (public), never chat-history data. Outputs hold
numbers (and, for regression, model outputs on the fixed prompts) only.

Usage:
  python cache_switch.py switch --gguf <base.gguf> --out results/x.json [--np 1]
  python cache_switch.py regression --gguf <base.gguf> --out results/r.json [--compare old.json]
  python cache_switch.py edge --config NAME:GGUF:NGL[:EXTRA,ARGS] ... --out results/e.json
"""

from __future__ import annotations

import argparse
import inspect
import json
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from subcor.backends.llamacpp_kernel import LlamaCppServingAdapter
from subcor.backends.llamacpp_supervisor import LlamaServerSupervisor
from subcor.expansion import LowRankAdapter
from subcor.router import RoutingDecision
from subcor.types import AdaptationMode, DeltaEncoding
from subcor.units import SkillUnitRegistry, build_layer_delta

UNITS = ("unit-a", "unit-b", "unit-c")
VRAM_USED = Path("/sys/class/drm/card1/device/mem_info_vram_used")


# ----------------------------------------------------------------------------- setup
class ScriptedRouter:
    """Stands in for MoESkillRouter: returns whatever unit the experiment selects next."""

    def __init__(self) -> None:
        self.tap_layer = "llamacpp.embedding"
        self.next: tuple[str, ...] = ()

    @property
    def registered_units(self) -> tuple[str, ...]:
        return UNITS

    def route(self, embedding: Any, *, top_k: int = 1, seat: str | None = None) -> RoutingDecision:
        return RoutingDecision(selected_units=self.next, scores={}, thresholds={})


def _down_proj_dims(base_gguf: Path) -> tuple[int, int, int]:
    import gguf

    reader = gguf.GGUFReader(str(base_gguf))
    downs = [t for t in reader.tensors if t.name.endswith("ffn_down.weight")]
    d_in, d_out = (int(x) for x in downs[0].shape[:2])  # GGUF ne order: (d_in, d_out)
    return len(downs), d_in, d_out


def build_units(registry_dir: Path, base_gguf: Path, *, rank: int, std: float) -> SkillUnitRegistry:
    """Synthetic units on mlp.down_proj of the last 4 layers (small std keeps text sane)."""
    n_layers, d_in, d_out = _down_proj_dims(base_gguf)
    registry = SkillUnitRegistry(registry_dir)
    for seed, uid in enumerate(UNITS):
        rng = np.random.default_rng(seed)
        deltas = []
        for layer in range(n_layers - 4, n_layers):
            name = f"layers.{layer}.mlp.down_proj"
            adapter = LowRankAdapter(
                name=name,
                a=rng.normal(0.0, std, size=(rank, d_in)),
                b=rng.normal(0.0, std, size=(d_out, rank)),
            )
            deltas.append(build_layer_delta(name, adapter, encoding=DeltaEncoding.LOW_RANK_FP32))
        registry.propose(
            unit_id=uid,
            version="1.0.0",
            title=uid,
            summary="synthetic unit for cache-switch measurement",
            proposer="tools",
            mode=AdaptationMode.DYNAMIC_EXPANSION_FP32,
            deltas=deltas,
        )
    return registry


def start_kernel(
    base_gguf: Path, work: Path, *, ngl: int, ctx: int, extra: list[str], port: int
) -> LlamaCppServingAdapter:
    sup = LlamaServerSupervisor(
        base_gguf=base_gguf,
        port=port,
        n_gpu_layers=ngl,
        ctx_size=ctx,
        extra_args=extra,
        lora_dir=work / "loras",
        log_path=work / "llama-server.log",
        startup_timeout=600.0,
    )
    return LlamaCppServingAdapter(
        supervisor=sup, model_id=base_gguf.name, lora_mode="request", timeout=1800.0
    )


def server_build() -> str:
    out = subprocess.run(["llama-server", "--version"], capture_output=True, text=True)
    text = (out.stdout + out.stderr).strip().splitlines()
    return text[0] if text else "unknown"


# ----------------------------------------------------------------------------- text
def corpus_chunks(chunk_chars: int) -> list[str]:
    """Deterministic public text: Python stdlib module sources."""
    import argparse as m1
    import dataclasses as m2
    import email.message as m3
    import json.decoder as m4
    import logging as m5
    import pathlib as m6
    import subprocess as m7
    import typing as m8

    text = "\n".join(inspect.getsource(m) for m in (m1, m2, m3, m4, m5, m6, m7, m8))
    return [text[i : i + chunk_chars] for i in range(0, len(text), chunk_chars)]


def user_turn(chunk: str, idx: int) -> dict[str, str]:
    return {
        "role": "user",
        "content": f"File part {idx}:\n```python\n{chunk}\n```\nReply with one word: ok.",
    }


# ----------------------------------------------------------------------------- calls
@dataclass
class Turn:
    prompt_tokens: int
    cache_n: int | None
    prompt_n: int | None
    prompt_ms: float | None
    wall_s: float
    text: str


def routed(
    kernel: LlamaCppServingAdapter,
    router: ScriptedRouter,
    registry: SkillUnitRegistry,
    messages: list[dict[str, Any]],
    unit: str | None,
    max_tokens: int = 4,
) -> Turn:
    router.next = (unit,) if unit else ()
    t0 = time.perf_counter()
    res = kernel.generate_routed_messages(
        messages=messages,
        tools=None,
        router=router,  # type: ignore[arg-type]
        registry=registry,
        max_tokens=max_tokens,
        enable_thinking=False,
        top_k=1,
    )
    wall = time.perf_counter() - t0
    t = res.server_timings or {}
    return Turn(
        res.prompt_tokens, t.get("cache_n"), t.get("prompt_n"), t.get("prompt_ms"), wall, res.text
    )


def direct(
    kernel: LlamaCppServingAdapter,
    registry: SkillUnitRegistry,
    messages: list[dict[str, Any]],
    unit: str | None,
    max_tokens: int = 4,
) -> Turn:
    """Same chat request as `routed`, but without the routing embedding call."""
    ids = kernel._resolve_lora_ids((unit,), registry) if unit else []
    body = kernel._chat_body(messages, None, max_tokens, False, ids, stream=False)
    t0 = time.perf_counter()
    resp = kernel._post_json("/v1/chat/completions", body)
    wall = time.perf_counter() - t0
    t = resp.get("timings") or {}
    usage = resp.get("usage") or {}
    text = str((resp["choices"][0].get("message") or {}).get("content") or "")
    return Turn(
        int(usage.get("prompt_tokens", 0)),
        t.get("cache_n"),
        t.get("prompt_n"),
        t.get("prompt_ms"),
        wall,
        text,
    )


# ----------------------------------------------------------------------------- modes
# (label, warm unit, probe unit, routed?)  every probe sits on a cache warmed with `warm`.
CONDITIONS = (
    ("same_unit", "unit-a", "unit-a", True),
    ("same_unit_no_embedding", "unit-a", "unit-a", False),
    ("switch_unit", "unit-a", "unit-b", True),
    ("unit_to_base", "unit-a", None, True),
    ("base_to_unit", None, "unit-a", True),
    ("same_base", None, None, True),
)


def run_switch(args: argparse.Namespace, kernel, router, registry) -> dict[str, Any]:
    chunks = corpus_chunks(args.chunk_chars)
    messages: list[dict[str, Any]] = [{"role": "system", "content": "You are a terse assistant."}]
    targets = sorted(args.sizes)
    results: list[dict[str, Any]] = []
    idx = 0
    for target in targets:
        # Grow the conversation on unit-a until the prompt reaches `target` tokens.
        while True:
            messages.append(user_turn(chunks[idx % len(chunks)], idx))
            idx += 1
            turn = routed(kernel, router, registry, messages, "unit-a")
            messages.append({"role": "assistant", "content": turn.text or "ok"})
            if turn.prompt_tokens >= target:
                break
        for label, warm, probe, use_router in CONDITIONS:
            probe_msgs = messages + [{"role": "user", "content": f"Probe {label}: reply ok."}]
            reps = []
            for rep in range(args.reps):
                # Warm: same history under `warm` (a cold re-read is fine here, it is not measured).
                warm_msgs = messages + [
                    {"role": "user", "content": f"Warm {label} {rep}: reply ok."}
                ]
                routed(kernel, router, registry, warm_msgs, warm)
                routed(kernel, router, registry, warm_msgs, warm)  # second call: confirmed warm
                probe_turn = (
                    routed(kernel, router, registry, probe_msgs, probe)
                    if use_router
                    else direct(kernel, registry, probe_msgs, probe)
                )
                reps.append(probe_turn.__dict__ | {"text": None})
            results.append({"target_tokens": target, "condition": label, "reps": reps})
            r0 = reps[-1]
            print(
                f"[{target:>6}] {label:<24} prompt={r0['prompt_tokens']:>6} cache_n={r0['cache_n']} "
                f"prompt_n={r0['prompt_n']} prompt_ms={r0['prompt_ms']:.0f} wall={r0['wall_s']:.2f}s",
                flush=True,
            )
    return {"probes": results}


REG_PROMPTS = (
    "Write a Python function that returns the n-th Fibonacci number.",
    "Explain what a mutex is in two sentences.",
    "List three prime numbers greater than 100.",
    "Translate 'good morning' into French and Spanish.",
    "What does HTTP status 404 mean?",
    "Give a one-line shell command that counts lines in all .py files.",
    "Summarize the rules of tic-tac-toe in one sentence.",
    "What is the capital of Australia?",
    "Convert 72 degrees Fahrenheit to Celsius.",
    "Name a sorting algorithm with O(n log n) worst case.",
)


def run_regression(args: argparse.Namespace, kernel, router, registry) -> dict[str, Any]:
    out: list[dict[str, Any]] = []
    for unit in (None, "unit-a"):
        for i, prompt in enumerate(REG_PROMPTS):
            router.next = (unit,) if unit else ()
            res = kernel.generate_routed_messages(
                messages=[{"role": "user", "content": prompt}],
                tools=None,
                router=router,  # type: ignore[arg-type]
                registry=registry,
                max_tokens=32,
                enable_thinking=False,
                top_k=1,
            )
            out.append(
                {
                    "unit": unit,
                    "prompt": i,
                    "text": res.text,
                    "logprobs": list(res.token_logprobs[:8]),
                }
            )
    report: dict[str, Any] = {"items": out}
    if args.compare:
        old = {
            (r["unit"], r["prompt"]): r for r in json.loads(Path(args.compare).read_text())["items"]
        }
        same_text, lp_diffs = 0, []
        for r in out:
            o = old[(r["unit"], r["prompt"])]
            same_text += int(o["text"] == r["text"])
            n = min(len(o["logprobs"]), len(r["logprobs"]))
            lp_diffs += [
                abs(a - b) for a, b in zip(o["logprobs"][:n], r["logprobs"][:n], strict=True)
            ]
        report["compare"] = {
            "identical_text": f"{same_text}/{len(out)}",
            "median_abs_logprob_diff": float(np.median(lp_diffs)) if lp_diffs else None,
            "max_abs_logprob_diff": float(np.max(lp_diffs)) if lp_diffs else None,
        }
        print(json.dumps(report["compare"]), flush=True)
    return report


def vram_used() -> int:
    return int(VRAM_USED.read_text()) if VRAM_USED.exists() else -1


def run_edge(args: argparse.Namespace) -> dict[str, Any]:
    rows = []
    prompt_text = "\n".join(corpus_chunks(args.chunk_chars))
    for spec in args.config:
        name, gguf_path, ngl, *rest = spec.split(":")
        extra = rest[0].split(",") if rest and rest[0] else []
        base = Path(gguf_path).expanduser()
        with tempfile.TemporaryDirectory(prefix="subcor-edge-") as tmp:
            work = Path(tmp)
            registry = build_units(work / "units", base, rank=args.rank, std=args.std)
            idle = vram_used()
            kernel = start_kernel(
                base, work, ngl=int(ngl), ctx=args.ctx, extra=extra, port=args.port
            )
            router = ScriptedRouter()
            try:
                kernel.prepare_units([(registry, "unit-a")])
                loaded = vram_used()
                # Cold prompt of ~edge_prompt_tokens (character budget, ~3.5 chars/token for code).
                msgs = [
                    {
                        "role": "user",
                        "content": prompt_text[: args.edge_prompt_tokens * 7 // 2] + "\nReply ok.",
                    }
                ]
                cold = routed(kernel, router, registry, msgs, "unit-a")
                peak = vram_used()
                router.next = ("unit-a",)
                dec = kernel.generate_routed_messages(
                    messages=[
                        {"role": "user", "content": "Write a long story about a lighthouse keeper."}
                    ],
                    tools=None,
                    router=router,  # type: ignore[arg-type]
                    registry=registry,
                    max_tokens=256,
                    enable_thinking=False,
                    top_k=1,
                )
                t = dec.server_timings or {}
                decode_tps = (
                    1000.0 * t["predicted_n"] / t["predicted_ms"] if t.get("predicted_ms") else None
                )
                row = {
                    "config": name,
                    "gguf": base.name,
                    "ngl": int(ngl),
                    "extra": extra,
                    "vram_loaded_gib": (loaded - idle) / 2**30,
                    "vram_peak_gib": (peak - idle) / 2**30,
                    "cold_prompt_tokens": cold.prompt_tokens,
                    "cold_prompt_ms": cold.prompt_ms,
                    "cold_wall_s": cold.wall_s,
                    "decode_tok_s": decode_tps,
                }
            finally:
                kernel.close()
        rows.append(row)
        print(json.dumps(row), flush=True)
    return {"rows": rows}


# ----------------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("switch", "regression", "edge"))
    ap.add_argument("--gguf", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--ctx", type=int, default=32768)
    ap.add_argument("--ngl", type=int, default=99)
    ap.add_argument("--np", type=int, default=None, help="llama-server --parallel")
    ap.add_argument("--extra", action="append", default=[], help="extra llama-server arg")
    ap.add_argument("--port", type=int, default=8197)
    ap.add_argument("--sizes", type=int, nargs="+", default=[4000, 16000, 30000])
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--chunk-chars", type=int, default=6000)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--std", type=float, default=0.01)
    ap.add_argument("--compare", type=Path)
    ap.add_argument("--config", action="append", default=[], help="edge: NAME:GGUF:NGL[:A,B]")
    ap.add_argument("--edge-prompt-tokens", type=int, default=4000)
    ap.add_argument("--workdir", type=Path, help="keep units/adapters/server log here")
    args = ap.parse_args()

    # Paths are reduced to their file names so results carry no local directory layout.
    meta = {
        "mode": args.mode,
        "build": server_build(),
        "args": {
            k: (v.name if isinstance(v, Path) else v)
            for k, v in vars(args).items()
            if k not in ("config", "out")
        },
    }
    if args.mode == "edge":
        report = run_edge(args)
    else:
        if args.gguf is None:
            ap.error("--gguf is required for switch/regression")
        base = args.gguf.expanduser()
        extra = list(args.extra) + (["--parallel", str(args.np)] if args.np is not None else [])
        with tempfile.TemporaryDirectory(prefix="subcor-cache-") as tmp:
            work = args.workdir or Path(tmp)
            work.mkdir(parents=True, exist_ok=True)
            registry = build_units(work / "units", base, rank=args.rank, std=args.std)
            kernel = start_kernel(
                base, work, ngl=args.ngl, ctx=args.ctx, extra=extra, port=args.port
            )
            router = ScriptedRouter()
            try:
                kernel.prepare_units([(registry, u) for u in UNITS])
                fn = run_switch if args.mode == "switch" else run_regression
                report = fn(args, kernel, router, registry)
            finally:
                kernel.close()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(meta | report, indent=2) + "\n")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
