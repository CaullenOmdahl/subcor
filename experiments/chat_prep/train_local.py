"""Train one Skill Unit on a chat-history capability bucket with TorchKernel, then compare
base vs base+unit on held-out (eval-split) sessions.

Metrics on the eval split (never seen in training, split by session), paired base vs unit:
  - nll: assistant-token NLL per window (training masking); bootstrap CI of the per-window gain
  - next: greedy first assistant action from (system, user, tools): same tool name / same
    argument keys / identical (whitespace-normalized) arguments; exact McNemar on name and args
  - cross_nll: NLL on the OTHER capabilities' eval windows, to catch regressions
Per-item results store window ids and outcomes only (no message text).

Usage:
  python train_local.py <03-capabilities dir> <capability> <out_dir>
      [--model Qwen/Qwen3.5-0.8B] [--steps 300] [--rank 16] [--layers 8]
      [--quantize-4bit] [--n-eval 200] [--n-gen 60]
Outputs <out_dir>/<capability>-<model-slug><tag>/{result.json, adapters.npz}. Local only.
  --adapters <npz> re-evaluates a saved unit without training.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from subcor.backends.torch_kernel import TorchKernel
from subcor.expansion import LowRankAdapter

_FN_RE = re.compile(r"<function=([^>\s]+)>")


def load(path: Path, max_tokens: int) -> list[dict]:
    return [r for r in map(json.loads, path.open()) if r["tokens_qwen35"] <= max_tokens]


@torch.no_grad()
def eval_nll(k: TorchKernel, recs: list[dict], max_seq_len: int) -> list[dict]:
    """Per-record assistant-token NLL (same masking as training). Ids and numbers only."""
    backbone = k.model.model
    head = k.model.get_output_embeddings()
    out = []
    for r in recs:
        ex, _ = k._tokenize_training_example(
            {"messages": r["messages"], "tools": r["tools"]},
            default_system_prompt=None, enable_thinking=False, max_seq_len=max_seq_len)
        if ex is None:
            continue
        x, y, pos, _ = ex
        h = backbone(input_ids=x, use_cache=False)[0][:, pos, :]
        logits = head(h)[0].float()
        out.append({"id": r["id"], "nll_sum": float(F.cross_entropy(logits, y, reduction="sum")),
                    "n_tok": int(y.shape[0])})
        del h, logits
    return out


def _call(c: dict | None) -> tuple[str | None, dict | None]:
    if not c:
        return None, None
    fn = c.get("function") or c
    args = fn.get("arguments")
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except json.JSONDecodeError:
            args = None
    return fn.get("name"), args if isinstance(args, dict) else None


def _norm(v):
    if isinstance(v, str):
        return " ".join(v.split())
    if isinstance(v, dict):
        return {k: _norm(x) for k, x in sorted(v.items())}
    return v


def next_action(k: TorchKernel, recs: list[dict], max_tokens: int = 192) -> list[dict]:
    """First assistant turn of each window: does the model take the same next action?"""
    out = []
    for r in recs:
        msgs = r["messages"]
        i = next(j for j, m in enumerate(msgs) if m["role"] == "assistant")
        ref_name, ref_args = _call((msgs[i].get("tool_calls") or [None])[0])
        tr = k.generate_messages_with_trace(msgs[:i], tools=r["tools"], max_tokens=max_tokens)
        if tr.tool_calls:
            got_name, got_args = _call(tr.tool_calls[0])
        else:
            m = _FN_RE.search(tr.raw_response or tr.response)
            got_name, got_args = (m.group(1) if m else None), None
        names = {t["function"]["name"] for t in r["tools"]}
        out.append({
            "id": r["id"],
            "ref_tool": ref_name, "got_tool": got_name,
            "name_match": got_name == ref_name,
            "valid_name": got_name is None or got_name in names,
            "args_exact": got_name == ref_name and (ref_name is None or (
                got_args is not None and _norm(got_args) == _norm(ref_args or {}))),
            "args_keys": got_name == ref_name and (ref_name is None or (
                got_args is not None and set(got_args) == set(ref_args or {}))),
            "hit_limit": bool(tr.hit_token_limit),
            "repetition": round(float(tr.repetition_ratio), 3),
            # runaway output: cut off by the cap, or heavily repeated text
            "loop": bool(tr.hit_token_limit) or float(tr.repetition_ratio) >= 0.3,
            "n_calls": len(tr.tool_calls),
        })
    return out


def mcnemar(base: list[bool], unit: list[bool]) -> dict:
    """Exact two-sided McNemar (binomial on discordant pairs)."""
    b = sum(1 for x, y in zip(base, unit, strict=True) if x and not y)
    c = sum(1 for x, y in zip(base, unit, strict=True) if y and not x)
    n = b + c
    if n == 0:
        return {"base_only": 0, "unit_only": 0, "p": 1.0}
    tail = sum(math.comb(n, i) for i in range(0, min(b, c) + 1)) / 2**n
    return {"base_only": b, "unit_only": c, "p": round(min(1.0, 2 * tail), 5)}


def nll_compare(base: list[dict], unit: list[dict], seed: int = 0) -> dict:
    bm = {x["id"]: x for x in base}
    pairs = [(bm[u["id"]], u) for u in unit if u["id"] in bm]
    db = np.array([b["nll_sum"] / b["n_tok"] - u["nll_sum"] / u["n_tok"] for b, u in pairs])
    tb = sum(b["nll_sum"] for b, _ in pairs) / sum(b["n_tok"] for b, _ in pairs)
    tu = sum(u["nll_sum"] for _, u in pairs) / sum(u["n_tok"] for _, u in pairs)
    rng = np.random.default_rng(seed)
    boots = [db[rng.integers(0, len(db), len(db))].mean() for _ in range(2000)]
    return {"n": len(pairs), "base_nll": round(tb, 4), "unit_nll": round(tu, 4),
            "mean_per_window_gain": round(float(db.mean()), 4),
            "gain_ci95": [round(float(np.percentile(boots, 2.5)), 4), round(float(np.percentile(boots, 97.5)), 4)],
            "windows_improved": round(float((db > 0).mean()), 3)}


def summarize(items: list[dict]) -> dict:
    n = max(1, len(items))
    return {"n": len(items), **{k: round(sum(x[k] for x in items) / n, 3)
                                for k in ("name_match", "args_keys", "args_exact", "valid_name", "hit_limit", "loop")}}


def mount(k: TorchKernel, adapters: dict) -> None:
    for name, a in adapters.items():
        k.instrumented[name].mount_adapter("unit", a)


def unmount(k: TorchKernel, adapters: dict) -> None:
    for name in adapters:
        k.instrumented[name].unmount_adapter("unit")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("data", type=Path)
    ap.add_argument("capability")
    ap.add_argument("out", type=Path)
    ap.add_argument("--model", default="Qwen/Qwen3.5-0.8B")
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--layers", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--grad-accum", type=int, default=1)
    ap.add_argument("--max-seq-len", type=int, default=2048)
    ap.add_argument("--projections", default="mlp.down_proj")
    ap.add_argument("--quantize-4bit", action="store_true")
    ap.add_argument("--grad-ckpt", action="store_true")
    ap.add_argument("--n-eval", type=int, default=200)
    ap.add_argument("--n-gen", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-cross", type=int, default=100, help="eval windows per OTHER capability (regression check)")
    ap.add_argument("--adapters", type=Path, help="evaluate a saved adapters.npz instead of training")
    ap.add_argument("--tag", default="", help="suffix for the output directory")
    ap.add_argument("--gen-tokens", type=int, default=192, help="generation cap for the next-action check")
    a = ap.parse_args()

    d = a.data / a.capability
    train = load(d / "train.jsonl", a.max_seq_len)
    ev = load(d / "eval.jsonl", a.max_seq_len)
    rnd = random.Random(a.seed)
    rnd.shuffle(ev)
    ev_nll, ev_gen = ev[: a.n_eval], ev[: a.n_gen]

    from transformers import AutoConfig
    cfg = AutoConfig.from_pretrained(a.model)
    n_layers = int(getattr(cfg, "text_config", cfg).num_hidden_layers)
    t0 = time.perf_counter()
    k = TorchKernel(a.model, target_layer_indices=list(range(n_layers - a.layers, n_layers)),
                    target_projections=tuple(a.projections.split(",")),
                    quantize_base_weights=a.quantize_4bit)
    layer_names = sorted(k.instrumented)
    print(f"loaded {a.model} in {time.perf_counter() - t0:.0f}s; {len(layer_names)} adapted layers; "
          f"train={len(train)} eval={len(ev)}", flush=True)

    others = [c for c in ("tool-calling", "code-writing", "agentic-decisions")
              if c != a.capability] if a.n_cross > 0 else []
    cross = {}
    for c in others:
        ce = load(a.data / c / "eval.jsonl", a.max_seq_len)
        random.Random(a.seed).shuffle(ce)
        cross[c] = ce[: a.n_cross]

    res: dict = {"model": a.model, "capability": a.capability, "args": {k_: str(v) for k_, v in vars(a).items()},
                 "layers": layer_names, "n_train": len(train)}

    def evaluate(tag: str) -> dict:
        t0 = time.perf_counter()
        r = {"nll": eval_nll(k, ev_nll, a.max_seq_len), "next": next_action(k, ev_gen, a.gen_tokens),
             "cross_nll": {c: eval_nll(k, v, a.max_seq_len) for c, v in cross.items()}}
        print(f"{tag}: nll={sum(x['nll_sum'] for x in r['nll']) / sum(x['n_tok'] for x in r['nll']):.4f} "
              f"next={summarize(r['next'])} ({time.perf_counter() - t0:.0f}s)", flush=True)
        return r

    base = evaluate("base")

    if a.adapters:
        z = np.load(a.adapters)
        adapters = {n: LowRankAdapter(name=f"trained:{n}", a=z[f"{n}.a"], b=z[f"{n}.b"]) for n in layer_names}
        res["train"] = {"loaded_from": str(a.adapters)}
    else:
        t0 = time.perf_counter()
        examples = [{"messages": r["messages"], "tools": r["tools"]} for r in train]
        adapters, losses, zero = k.train_qlora_adapters(
            examples, layer_names=layer_names, rank=a.rank, steps=a.steps, lr=a.lr,
            grad_accum=a.grad_accum, max_seq_len=a.max_seq_len, seed=a.seed,
            gradient_checkpointing=a.grad_ckpt)
        res["train"] = {"seconds": round(time.perf_counter() - t0, 1), "zero_perturb": zero,
                        "loss_first20": float(np.mean(losses[:20])), "loss_last20": float(np.mean(losses[-20:])),
                        "losses": [round(x, 4) for x in losses], "stats": k.last_train_stats,
                        "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2)
                        if torch.cuda.is_available() else None}
        print(f"train: {res['train']['seconds']}s loss {res['train']['loss_first20']:.3f} -> "
              f"{res['train']['loss_last20']:.3f} peak {res['train']['peak_vram_gb']} GB", flush=True)

    mount(k, adapters)
    unit = evaluate("unit")
    unmount(k, adapters)

    bn = {x["id"]: x for x in base["next"]}
    pairs = [(bn[u["id"]], u) for u in unit["next"]]
    res["summary"] = {
        "nll": nll_compare(base["nll"], unit["nll"], a.seed),
        "next_base": summarize(base["next"]), "next_unit": summarize(unit["next"]),
        "mcnemar_name": mcnemar([b["name_match"] for b, _ in pairs], [u["name_match"] for _, u in pairs]),
        "mcnemar_args_exact": mcnemar([b["args_exact"] for b, _ in pairs], [u["args_exact"] for _, u in pairs]),
        "mcnemar_args_keys": mcnemar([b["args_keys"] for b, _ in pairs], [u["args_keys"] for _, u in pairs]),
        "mcnemar_no_loop": mcnemar([not b["loop"] for b, _ in pairs], [not u["loop"] for _, u in pairs]),
        "cross_nll": {c: nll_compare(base["cross_nll"][c], unit["cross_nll"][c], a.seed) for c in cross},
    }
    print("summary: " + json.dumps(res["summary"]), flush=True)
    res["items"] = {"base": base, "unit": unit}

    out = a.out / f"{a.capability}-{a.model.split('/')[-1].lower()}{a.tag}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "result.json").write_text(json.dumps(res, indent=1))
    if not a.adapters:
        np.savez(out / "adapters.npz", **{f"{n}.{f}": getattr(ad, f) for n, ad in adapters.items()
                                           for f in ("a", "b")})
    print(f"wrote {out}", flush=True)

if __name__ == "__main__":
    main()
