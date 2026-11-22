"""Print raw base vs unit generations for a few eval windows (local debugging only; prints personal data).

Usage: python inspect_outputs.py <03-capabilities dir> <capability> <adapters.npz> [n=8] [gen_tokens=384]
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np
from transformers import AutoConfig

from subcor.backends.torch_kernel import TorchKernel
from subcor.expansion import LowRankAdapter

sys.path.insert(0, str(Path(__file__).parent))
from train_local import load, mount, unmount  # noqa: E402


def main() -> None:
    data, cap, npz = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
    n = int(sys.argv[4]) if len(sys.argv) > 4 else 8
    gen = int(sys.argv[5]) if len(sys.argv) > 5 else 384
    model = "Qwen/Qwen3.5-0.8B"
    cfg = AutoConfig.from_pretrained(model)
    total = int(getattr(cfg, "text_config", cfg).num_hidden_layers)
    k = TorchKernel(model, target_layer_indices=list(range(total - 8, total)), target_projections=("mlp.down_proj",))
    names = sorted(k.instrumented)
    z = np.load(npz)
    ad = {nm: LowRankAdapter(name=f"trained:{nm}", a=z[f"{nm}.a"], b=z[f"{nm}.b"]) for nm in names}
    ev = load(data / cap / "eval.jsonl", 2048)
    random.Random(0).shuffle(ev)
    for r in ev[:n]:
        msgs = r["messages"]
        i = next(j for j, m in enumerate(msgs) if m["role"] == "assistant")
        print("=" * 100)
        print("USER:", str(msgs[i - 1]["content"])[-300:].replace("\n", " "))
        print("REF :", (str(msgs[i].get("content") or "")[:200] + " | calls=" + str(msgs[i].get("tool_calls"))[:300]).replace("\n", " "))
        for tag in ("base", "unit"):
            if tag == "unit":
                mount(k, ad)
            tr = k.generate_messages_with_trace(msgs[:i], tools=r["tools"], max_tokens=gen)
            if tag == "unit":
                unmount(k, ad)
            print(f"--- {tag.upper()} (tokens={tr.tokens_generated}, limit={tr.hit_token_limit}, "
                  f"rep={tr.repetition_ratio:.2f}, calls={len(tr.tool_calls)})")
            print((tr.raw_response or tr.response)[:1500])


if __name__ == "__main__":
    main()
