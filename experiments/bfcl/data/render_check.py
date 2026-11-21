"""Render every conversation through SUBCOR's trainer encoding (the Qwen3.5 chat template as
llama-server --jinja applies it) and report real token lengths. Run in the PROJECT env:

  cd <repo> && uv run --extra dev --extra rocm python experiments/bfcl/data/render_check.py \
      --data $SCRATCH/bfcl/data [--tokenizer Qwen/Qwen3.5-0.8B] [--max-tokens 4096]

Exit 1 if any row fails to render/encode. Qwen3.5 0.8B/4B/9B share the tokenizer + template.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from transformers import AutoTokenizer

from subcor.backends.torch_kernel import encode_chat_training_example


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--tokenizer", default="Qwen/Qwen3.5-0.8B")
    ap.add_argument("--max-tokens", type=int, default=4096)
    args = ap.parse_args()
    tok = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    bad = 0
    out = {}
    for name in ("train", "dev", "train_long"):
        p = args.data / f"{name}.jsonl"
        if not p.exists():
            continue
        lens, tgt, over, errs = [], 0, 0, Counter()
        nseq = 0
        for line in p.read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            try:
                seqs = encode_chat_training_example(tok, r["messages"], tools=r["tools"])
            except Exception as e:  # noqa: BLE001
                errs[type(e).__name__ + ": " + str(e)[:80]] += 1
                continue
            if not seqs:
                errs["no trainable sequence"] += 1
                continue
            m = max(len(s.input_ids) for s in seqs)
            nseq += len(seqs)
            lens.append(m)
            tgt += sum(s.num_target_tokens for s in seqs)
            over += m > args.max_tokens
        lens.sort()
        q = lambda f: lens[min(len(lens) - 1, int(f * len(lens)))] if lens else 0  # noqa: E731
        out[name] = {"rows_ok": len(lens), "errors": dict(errs), "sequences": nseq, "target_tokens": tgt,
                     "max_seq_tokens": {"p50": q(0.5), "p90": q(0.9), "p99": q(0.99), "max": lens[-1] if lens else 0},
                     f"rows_over_{args.max_tokens}": over}
        bad += sum(errs.values())
    print(json.dumps(out, indent=1))
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
