"""Train one SUBCOR Skill Unit on tool-calling conversations, on the serving GGUF.

Pipeline (one run = one unit):

1. Load the PyTorch trainer from the SAME GGUF llama-server serves
   (``load_trainer_from_gguf``: weights dequantized from the quantized file).
2. Read OpenAI-format conversations (JSONL, one ``{"messages": [...], "tools": [...]}``
   per line; extra keys ignored) and split off a dev set (``--dev`` file or
   ``--dev-fraction`` of the conversations, by seed). Every conversation is rendered with
   the GGUF's chat template exactly as ``llama-server --jinja`` renders it (tool-call
   ``arguments`` parsed into objects, ``enable_thinking=false``), with loss on assistant
   tokens only (tool-call turns included). ``check_template_parity.py`` verifies this
   token-for-token against a running llama-server.
3. Train a zero-initialised FP32 LoRA on ``--layers`` x ``--projections`` with
   early stopping on the dev loss (the best dev checkpoint is kept).
4. Pack two variants and propose both (DORMANT) into ``--registry``:
   ``<unit-id>-sub1bit`` (SUB1BIT_BINARY, the SUBCOR unit) and ``<unit-id>-fp32``
   (LOW_RANK_FP32, the same adapters unpacked). Dev loss is reported for base / FP32 /
   SUB1BIT so the packing cost is visible.
5. Fit a RouterGate for each variant from ``--pos-prompts`` / ``--neg-prompts`` using the
   SERVING backend's embeddings: ``LlamaCppServingAdapter.encode_prompt_embedding`` on
   ``--embed-url`` (llama-server ``/v1/embeddings``, mean-pooled, L2-normalised, the
   last user message as raw text), exactly what ``subcor serve --backend llamacpp``
   routes on. Without ``--pos-prompts`` the positives are the latest user message
   before every trained assistant turn of the training set.
6. Write an always-on plain LoRA GGUF of the FP32 adapters (the "A2" arm:
   ``llama-server --lora <file>``), via ``gguf_export.export_unit_to_gguf_lora``, by
   default with f32 tensors: the same tensors SUBCOR's supervisor exports for the A4
   fp32 unit, so A4 - A2 differs only in routing (``--a2-dtype f16`` for the old file).

Everything is written under ``--out`` (``train_report.json``, the A2 GGUF) and
``--registry``. Units are left DORMANT: promoting them (conformance + approval) is the
governed server's job.

Example (GPU trainer; llama-server with embeddings on :8503):

    uv run --extra dev --extra rocm python experiments/bfcl/train_unit.py \\
        --base-gguf Qwen3.5-4B-Q4_K_M.gguf --train train.jsonl --dev-fraction 0.1 \\
        --unit-id bfcl-tools-v1 --registry STATE/tenants/default/units --out OUT \\
        --layers 26-31 --projections mlp.down_proj --rank 8 --steps 400 --lr 3e-4 \\
        --eval-every 25 --patience 4 --neg-prompts general.txt \\
        --embed-url http://127.0.0.1:8503
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from subcor.backends.llamacpp_kernel import LlamaCppServingAdapter, _read_gguf_block_count
from subcor.router import RouterGate
from subcor.types import AdaptationMode, DeltaEncoding
from subcor.units import SkillUnitRegistry, build_layer_delta


# --------------------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------------------
def read_conversations(path: Path) -> list[dict[str, Any]]:
    convs: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as fh:
        for n, line in enumerate(fh, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict) or not isinstance(row.get("messages"), list):
                raise ValueError(f"{path}:{n}: expected an object with a 'messages' list")
            conv: dict[str, Any] = {"messages": row["messages"]}
            if row.get("tools"):
                conv["tools"] = row["tools"]
            if "id" in row:
                conv["id"] = row["id"]
            convs.append(conv)
    return convs


def read_prompts(
    path: Path, *, split: str = "fit", exclude_ids: set[str] | frozenset[str] = frozenset()
) -> list[str]:
    """Plain text (one prompt per line) or JSONL rows with 'prompt' or 'messages'.

    JSONL rows that carry a ``split`` key (``build_router_calib.py`` writes "fit" /
    "heldout") are kept only when it equals ``split``; ``split="all"`` keeps every row.
    Without this filter the heldout rows that PREREGISTRATION.md reserves for gate
    diagnostics would be fitted on. Rows whose ``row_id`` (the source conversation) is in
    ``exclude_ids`` are dropped, like the conversations themselves.
    """
    out: list[str] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.rstrip("\n")
            if not line.strip():
                continue
            if path.suffix == ".jsonl":
                row = json.loads(line)
                if isinstance(row, str):
                    out.append(row)
                    continue
                if split != "all" and "split" in row and row["split"] != split:
                    continue
                if row.get("row_id") in exclude_ids:
                    continue
                if "prompt" in row:
                    out.append(str(row["prompt"]))
                else:
                    out.append(last_user_text(row["messages"]))
            else:
                out.append(line)
    return out


def read_exclude_ids(path: Path | None) -> set[str]:
    """Conversation ids to leave out: a JSON list, or a JSON object keyed by id."""
    if path is None:
        return set()
    data = json.loads(path.read_text(encoding="utf-8"))
    return {str(x) for x in (data.keys() if isinstance(data, dict) else data)}


def sha256_file(path: Path | None) -> str | None:
    if path is None:
        return None
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()


def assert_embed_server_frozen(base_url: str) -> list[dict[str, Any]]:
    """Refuse to fit a gate on a llama-server whose GLOBAL LoRA scales are not all 0.

    llama-server's embedding endpoints always use the global adapter scales, so gate
    embeddings taken from, e.g., the A2 server (``--lora a2.gguf`` at scale 1) would not
    be frozen-base embeddings and would not match what serving routes on.
    """
    import urllib.error
    import urllib.request

    try:
        with urllib.request.urlopen(f"{base_url.rstrip('/')}/lora-adapters", timeout=30) as r:
            listed = json.loads(r.read().decode("utf-8") or "[]")
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            return []
        raise
    listed = listed if isinstance(listed, list) else []
    nonzero = [a for a in listed if isinstance(a, dict) and float(a.get("scale") or 0.0) != 0.0]
    if nonzero:
        raise SystemExit(
            f"--embed-url {base_url} has LoRA adapters with nonzero global scale {nonzero}; "
            "router gates must be fit on frozen-base embeddings"
        )
    return listed


def last_user_text(messages: list[dict[str, Any]]) -> str:
    """The text SUBCOR serving routes on (same expression as LlamaCppServingAdapter._route)."""
    return next(
        (str(m.get("content", "")) for m in reversed(messages) if m.get("role") == "user"), ""
    )


def routing_prompts_from_conversations(convs: list[dict[str, Any]]) -> list[str]:
    """Latest user message before every trained assistant turn (what serving would embed)."""
    seen: dict[str, None] = {}
    for conv in convs:
        msgs = conv["messages"]
        for k, m in enumerate(msgs):
            if m.get("role") == "assistant" and m.get("weight", 1):
                text = last_user_text(msgs[:k])
                if text:
                    seen.setdefault(text, None)
    return list(seen)


def parse_layers(spec: str | None, base_gguf: Path) -> list[int] | None:
    if spec is None:
        n = _read_gguf_block_count(base_gguf)
        return None if n is None else list(range(max(0, n - 6), n))
    out: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = (int(x) for x in part.split("-", 1))
            out.extend(range(lo, hi + 1))
        else:
            out.append(int(part))
    return sorted(set(out))


# --------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--base-gguf", type=Path, required=True)
    ap.add_argument("--train", type=Path, required=True, help="Conversations JSONL")
    ap.add_argument("--dev", type=Path, help="Dev conversations JSONL (else --dev-fraction)")
    ap.add_argument("--dev-fraction", type=float, default=0.1)
    ap.add_argument("--unit-id", required=True)
    ap.add_argument("--version", default="1.0.0")
    ap.add_argument("--title", default=None)
    ap.add_argument("--summary", default="Tool-calling Skill Unit trained on chat conversations")
    ap.add_argument("--proposer", default="experiments/bfcl/train_unit.py")
    ap.add_argument("--base-kernel-id", default=None, help="Default: GGUF stem (= subcor serve)")
    ap.add_argument("--registry", type=Path, required=True, help="SkillUnitRegistry root")
    ap.add_argument("--out", type=Path, required=True, help="Report / A2 LoRA directory")
    ap.add_argument("--layers", default=None, help="e.g. '18-23' or '7,11,15' (default: last 6)")
    ap.add_argument("--projections", default="mlp.down_proj", help="comma separated")
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--sub1bit-rank", type=int, default=None, help="Default: --rank")
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--lr", type=float, default=3e-4, help="Trainer caps FP32 lr at 4e-4")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--grad-accum", type=int, default=1)
    ap.add_argument("--eval-every", type=int, default=25)
    ap.add_argument("--patience", type=int, default=4, help="0 = never stop early")
    ap.add_argument("--max-seq-tokens", type=int, default=4096, help="Drop longer convs")
    ap.add_argument("--max-train", type=int, default=0, help="Cap #train convs (0 = all)")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--dtype", default="auto")
    ap.add_argument("--gradient-checkpointing", action="store_true")
    ap.add_argument("--variants", default="sub1bit,fp32")
    ap.add_argument("--pos-prompts", type=Path, help="Router positives (.txt lines or .jsonl)")
    ap.add_argument("--neg-prompts", type=Path, help="Router negatives (.txt lines or .jsonl)")
    ap.add_argument(
        "--prompt-split",
        default="fit",
        help="Keep only .jsonl prompt rows whose 'split' equals this ('all' = every row)",
    )
    ap.add_argument(
        "--exclude-ids",
        type=Path,
        default=None,
        help="JSON list/object of conversation ids to drop from --train/--dev and from "
        "prompt rows by 'row_id' (e.g. data/results/decontam_semantic_drop.json)",
    )
    ap.add_argument("--no-gate", action="store_true", help="Skip RouterGate fitting")
    ap.add_argument("--embed-url", default="http://127.0.0.1:8503")
    ap.add_argument("--tap-layer", default="llamacpp.embedding")
    ap.add_argument("--gate-safety-bias", type=float, default=0.0)
    ap.add_argument("--num-prototypes", type=int, default=1)
    ap.add_argument("--a2-lora", type=Path, default=None, help="Default: OUT/<unit>-a2-f16.gguf")
    ap.add_argument("--no-a2", action="store_true")
    ap.add_argument(
        "--a2-dtype",
        default="f32",
        choices=["f32", "f16"],
        help="A2 GGUF tensor dtype. f32 (default) = the same tensors the supervisor exports "
        "for the A4 fp32 unit, so A4 - A2 isolates routing only",
    )
    return ap


def _split(
    convs: list[dict[str, Any]], frac: float, seed: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    idx = list(range(len(convs)))
    random.Random(seed).shuffle(idx)
    n_dev = max(1, int(round(frac * len(convs)))) if frac > 0 and len(convs) > 1 else 0
    dev_idx = set(idx[:n_dev])
    return [c for i, c in enumerate(convs) if i not in dev_idx], [convs[i] for i in sorted(dev_idx)]


def _filter_len(trainer: Any, convs: list[dict[str, Any]], max_tokens: int) -> tuple[list, int]:
    kept, dropped = [], 0
    for conv in convs:
        seqs = trainer.encode_training_examples([conv])
        if not seqs or max(len(s.input_ids) for s in seqs) > max_tokens:
            dropped += 1
            continue
        kept.append(conv)
    return kept, dropped


def _dev_loss_with(trainer: Any, adapters: dict[str, Any], dev_seqs: list[Any]) -> float:
    for name, ad in adapters.items():
        trainer.instrumented[name].mount_adapter(f"eval:{name}", ad)
    try:
        return float(trainer.evaluate_loss(dev_seqs))
    finally:
        for name in adapters:
            trainer.instrumented[name].unmount_adapter(f"eval:{name}")


def _fit_gate(
    unit_id: str,
    pos: np.ndarray,
    neg: np.ndarray,
    args: argparse.Namespace,
) -> tuple[RouterGate, dict[str, Any]]:
    gate = RouterGate.fit(
        unit_id=unit_id,
        tap_layer=args.tap_layer,
        pos_embeddings=pos,
        neg_embeddings=neg,
        safety_bias=args.gate_safety_bias,
        num_prototypes=args.num_prototypes,
    )
    pos_hit = float(np.mean([gate.matches(e) for e in pos]))
    neg_hit = float(np.mean([gate.matches(e) for e in neg]))
    stats = {
        "threshold": gate.threshold,
        "pos_mean_score": gate.pos_mean_score,
        "neg_mean_score": gate.neg_mean_score,
        "fit_pos_activation_rate": pos_hit,
        "fit_neg_activation_rate": neg_hit,
        "n_pos": int(pos.shape[0]),
        "n_neg": int(neg.shape[0]),
    }
    return gate, stats


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    t0 = time.time()
    base_gguf = args.base_gguf.expanduser().absolute()
    args.out.mkdir(parents=True, exist_ok=True)
    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    bad = set(variants) - {"sub1bit", "fp32"}
    if bad or not variants:
        raise SystemExit(f"--variants must be from sub1bit,fp32 (got {args.variants!r})")
    if not args.no_gate and args.neg_prompts is None:
        raise SystemExit("--neg-prompts is required to fit a RouterGate (or pass --no-gate)")
    if not args.no_a2 and "fp32" not in variants:
        raise SystemExit(
            "The A2 FP16 LoRA is exported from the fp32 variant; keep it in --variants"
        )
    registry = SkillUnitRegistry(args.registry)
    unit_ids = {v: f"{args.unit_id}-{v}" for v in variants}
    for uid in unit_ids.values():
        if (args.registry / uid).exists():
            raise SystemExit(f"Unit '{uid}' already exists in {args.registry}; bump --unit-id")

    # ---------------------------------------------------------------- data
    exclude_ids = read_exclude_ids(args.exclude_ids)
    convs = read_conversations(args.train)
    n_train_raw = len(convs)
    convs = [c for c in convs if str(c.get("id")) not in exclude_ids]
    excluded = {"train": n_train_raw - len(convs), "dev": 0}
    if args.dev is not None:
        dev_raw = read_conversations(args.dev)
        dev_convs = [c for c in dev_raw if str(c.get("id")) not in exclude_ids]
        excluded["dev"] = len(dev_raw) - len(dev_convs)
        train_convs = convs
    else:
        train_convs, dev_convs = _split(convs, args.dev_fraction, args.seed)
    if exclude_ids:
        print(f"[train_unit] --exclude-ids removed {excluded}", flush=True)
    if args.max_train > 0:
        train_convs = train_convs[: args.max_train]

    # ---------------------------------------------------------------- trainer
    from subcor.backends.torch_kernel import load_trainer_from_gguf

    projections = tuple(p.strip() for p in args.projections.split(",") if p.strip())
    layer_idx = parse_layers(args.layers, base_gguf)
    print(f"[train_unit] loading trainer from {base_gguf.name} (layers={layer_idx})", flush=True)
    trainer = load_trainer_from_gguf(
        base_gguf,
        target_layer_indices=layer_idx,
        target_projections=projections,
        device=args.device,
        dtype=args.dtype,
    )
    layer_names = sorted(
        trainer.instrumented,
        key=lambda n: (int(n.split(".")[1]), n),
    )
    from subcor.gguf_export import canonical_to_gguf_base_tensor

    for name in layer_names:
        canonical_to_gguf_base_tensor(name)  # fail fast on non-exportable projections
    t_load = time.time() - t0

    train_convs, dropped_train = _filter_len(trainer, train_convs, args.max_seq_tokens)
    dev_convs, dropped_dev = _filter_len(trainer, dev_convs, args.max_seq_tokens)
    if not train_convs:
        raise SystemExit("No training conversations left after filtering")
    dev_seqs = trainer.encode_training_examples(dev_convs) if dev_convs else []
    print(
        f"[train_unit] {len(train_convs)} train / {len(dev_convs)} dev conversations "
        f"(dropped {dropped_train}/{dropped_dev} > {args.max_seq_tokens} tokens); "
        f"{len(layer_names)} layers x rank {args.rank} on {trainer.device} {trainer.dtype}",
        flush=True,
    )

    # ---------------------------------------------------------------- train
    report: dict[str, Any] = {}
    use_dev = bool(dev_convs) and args.eval_every > 0
    t1 = time.time()
    adapters, losses, zero_diff = trainer.train_qlora_adapters(
        train_convs,
        layer_names=list(layer_names),
        enable_thinking=False,
        rank=args.rank,
        scale=args.scale,
        binary_ste=False,
        steps=args.steps,
        lr=args.lr,
        seed=args.seed,
        gradient_checkpointing=args.gradient_checkpointing,
        grad_accum_steps=args.grad_accum,
        eval_examples=dev_convs if use_dev else None,
        eval_every=args.eval_every if use_dev else 0,
        early_stopping_patience=args.patience,
        report=report,
    )
    t_train = time.time() - t1
    print(
        f"[train_unit] {report['steps_run']} steps in {t_train:.1f}s; best step "
        f"{report['best_step']} dev={report['best_dev_loss']}; zero-perturbation diff "
        f"{zero_diff:.3g}",
        flush=True,
    )

    # ---------------------------------------------------------------- pack + propose
    sub1bit_rank = args.sub1bit_rank or args.rank
    deltas_by_variant: dict[str, list[Any]] = {}
    if "sub1bit" in variants:
        deltas_by_variant["sub1bit"] = [
            build_layer_delta(
                n,
                adapters[n],
                encoding=DeltaEncoding.SUB1BIT_BINARY,
                sub1bit_rank=sub1bit_rank,
            )
            for n in layer_names
        ]
    if "fp32" in variants:
        deltas_by_variant["fp32"] = [
            build_layer_delta(n, adapters[n], encoding=DeltaEncoding.LOW_RANK_FP32)
            for n in layer_names
        ]

    dev_loss: dict[str, float | None] = {"base": None}
    if dev_seqs:
        dev_loss["base"] = float(trainer.evaluate_loss(dev_seqs))
        for variant, deltas in deltas_by_variant.items():
            ads = {d.layer_name: d.to_adapter(f"{variant}:{d.layer_name}") for d in deltas}
            dev_loss[variant] = _dev_loss_with(trainer, ads, dev_seqs)

    recon: dict[str, float] = {}
    if "sub1bit" in deltas_by_variant:
        for d in deltas_by_variant["sub1bit"]:
            ref = adapters[d.layer_name].delta_matrix()
            got = d.to_adapter("r").delta_matrix()
            denom = float(np.linalg.norm(ref)) or 1.0
            recon[d.layer_name] = float(np.linalg.norm(got - ref) / denom)

    base_kernel_id = args.base_kernel_id or base_gguf.stem
    manifests: dict[str, Any] = {}
    for variant, deltas in deltas_by_variant.items():
        m = registry.propose(
            unit_id=unit_ids[variant],
            version=args.version,
            title=args.title or f"{args.unit_id} ({variant})",
            summary=args.summary,
            proposer=args.proposer,
            mode=AdaptationMode.DYNAMIC_EXPANSION_FP32,
            deltas=deltas,
            base_kernel_id=base_kernel_id,
        )
        manifests[variant] = {
            "unit_id": m.unit_id,
            "encoding": m.encoding.value,
            "payload_sha256": m.payload_sha256,
            "payload_bytes": m.payload_bytes,
            "effective_bits_per_weight": m.effective_bits_per_weight,
            "state": m.state.value,
            "target_layers": list(m.target_layers),
        }

    # ---------------------------------------------------------------- router gate
    gates: dict[str, Any] = {}
    if not args.no_gate:
        embed_loras = assert_embed_server_frozen(args.embed_url)
        embedder = LlamaCppServingAdapter(base_url=args.embed_url, tap_layer=args.tap_layer)
        pos_prompts = (
            read_prompts(args.pos_prompts, split=args.prompt_split, exclude_ids=exclude_ids)
            if args.pos_prompts is not None
            else routing_prompts_from_conversations(train_convs)
        )
        neg_prompts = read_prompts(
            args.neg_prompts, split=args.prompt_split, exclude_ids=exclude_ids
        )
        if not pos_prompts or not neg_prompts:
            raise SystemExit("RouterGate needs non-empty positive and negative prompt sets")
        pos = np.stack([embedder.encode_prompt_embedding(p) for p in pos_prompts])
        neg = np.stack([embedder.encode_prompt_embedding(p) for p in neg_prompts])
        dev_pos = routing_prompts_from_conversations(dev_convs)
        dev_pos_emb = (
            np.stack([embedder.encode_prompt_embedding(p) for p in dev_pos]) if dev_pos else None
        )
        for variant, uid in unit_ids.items():
            gate, stats = _fit_gate(uid, pos, neg, args)
            if dev_pos_emb is not None:
                stats["dev_pos_activation_rate"] = float(
                    np.mean([gate.matches(e) for e in dev_pos_emb])
                )
                stats["n_dev_pos"] = len(dev_pos)
            stats["embed_server_lora_adapters"] = embed_loras
            stats["prompt_split"] = args.prompt_split
            registry.save_router_gate(gate)
            gates[variant] = stats
        print(f"[train_unit] RouterGate: {json.dumps(gates.get(variants[0]))}", flush=True)

    # ---------------------------------------------------------------- A2 LoRA
    a2_path: Path | None = None
    if not args.no_a2:
        from subcor.gguf_export import export_unit_to_gguf_lora

        a2_path = args.a2_lora or (args.out / f"{args.unit_id}-a2-{args.a2_dtype}.gguf")
        export_unit_to_gguf_lora(
            registry, unit_ids["fp32"], a2_path, base_gguf=base_gguf, tensor_dtype=args.a2_dtype
        )

    # ---------------------------------------------------------------- report
    finite = [v for v in losses if math.isfinite(v)]
    head = float(np.mean(finite[: min(10, len(finite))])) if finite else None
    tail = float(np.mean(finite[-min(10, len(finite)) :])) if finite else None
    out = {
        "unit_id": args.unit_id,
        "units": manifests,
        "base_gguf": str(base_gguf),
        "base_gguf_bytes": base_gguf.stat().st_size,
        "base_kernel_id": base_kernel_id,
        "config": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        "input_sha256": {
            "train": sha256_file(args.train),
            "dev": sha256_file(args.dev),
            "pos_prompts": sha256_file(args.pos_prompts),
            "neg_prompts": sha256_file(args.neg_prompts),
            "exclude_ids": sha256_file(args.exclude_ids),
            "base_gguf_bytes": base_gguf.stat().st_size,
        },
        "excluded_conversations": excluded,
        "layers": list(layer_names),
        "device": str(trainer.device),
        "dtype": str(trainer.dtype),
        "data": {
            "train_conversations": len(train_convs),
            "dev_conversations": len(dev_convs),
            "dropped_train_too_long": dropped_train,
            "dropped_dev_too_long": dropped_dev,
            "train_sequences": report.get("num_train_sequences"),
            "train_target_tokens": report.get("num_train_target_tokens"),
            "dev_sequences": len(dev_seqs),
            "dev_target_tokens": sum(s.num_target_tokens for s in dev_seqs),
        },
        "training": {
            "steps_run": report.get("steps_run"),
            "stopped_early": report.get("stopped_early"),
            "best_step": report.get("best_step"),
            "best_dev_loss": report.get("best_dev_loss"),
            "dev_curve": report.get("dev_losses"),
            "train_loss_head10": head,
            "train_loss_tail10": tail,
            "train_losses": losses,
            "zero_perturbation_max_diff": zero_diff,
        },
        "dev_loss": dev_loss,
        "sub1bit_relative_reconstruction_error": recon,
        "router_gates": gates,
        "a2_lora_gguf": str(a2_path) if a2_path else None,
        "timing_s": {
            "load": round(t_load, 1),
            "train": round(t_train, 1),
            "total": round(time.time() - t0, 1),
        },
    }
    report_path = args.out / "train_report.json"
    report_path.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(
        f"[train_unit] dev loss base={dev_loss.get('base')} fp32={dev_loss.get('fp32')} "
        f"sub1bit={dev_loss.get('sub1bit')}\n[train_unit] units: "
        f"{', '.join(unit_ids.values())} -> {args.registry}\n[train_unit] A2: {a2_path}\n"
        f"[train_unit] report: {report_path}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
