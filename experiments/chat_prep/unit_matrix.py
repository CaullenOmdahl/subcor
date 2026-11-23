"""Score every trained unit on every capability's eval windows (same windows for all).

If units specialize, each row's best column is its own capability. If they all learn the
same general "house style", every unit helps every bucket by about the same amount.

Usage: [SUBCOR_MODEL=.. SUBCOR_LAYERS=.. SUBCOR_PROJ=.. SUBCOR_Q4=1] python unit_matrix.py <03-capabilities dir> <out.json> name=path/adapters.npz ...
"""

from __future__ import annotations

import json
import os
import random
import sys
from pathlib import Path

import numpy as np
from transformers import AutoConfig

from subcor.backends.torch_kernel import TorchKernel
from subcor.expansion import LowRankAdapter

sys.path.insert(0, str(Path(__file__).parent))
from train_local import eval_nll, load, mount, unmount  # noqa: E402

CAPS = ("tool-calling", "code-writing", "agentic-decisions")


def main() -> None:
    data, out = Path(sys.argv[1]), Path(sys.argv[2])
    units = dict(a.split("=", 1) for a in sys.argv[3:])
    # SUBCOR_MODEL / SUBCOR_LAYERS ("19,23,27,31" or "last8") / SUBCOR_PROJ / SUBCOR_Q4=1 must match training
    model, n = os.environ.get("SUBCOR_MODEL", "Qwen/Qwen3.5-0.8B"), 100
    cfg = AutoConfig.from_pretrained(model)
    total = int(getattr(cfg, "text_config", cfg).num_hidden_layers)
    spec = os.environ.get("SUBCOR_LAYERS", "last8")
    idx = (
        list(range(total - int(spec[4:]), total))
        if spec.startswith("last")
        else [int(x) for x in spec.split(",")]
    )
    k = TorchKernel(
        model,
        target_layer_indices=idx,
        target_projections=tuple(os.environ.get("SUBCOR_PROJ", "mlp.down_proj").split(",")),
        quantize_base_weights=os.environ.get("SUBCOR_Q4") == "1",
    )
    names = sorted(k.instrumented)
    sets = {}
    for c in CAPS:
        ev = load(data / c / "eval.jsonl", 2048)
        random.Random(0).shuffle(ev)
        sets[c] = ev[:n]

    def score() -> dict:
        res = {}
        for c, recs in sets.items():
            items = eval_nll(k, recs, 2048)
            res[c] = round(sum(x["nll_sum"] for x in items) / sum(x["n_tok"] for x in items), 4)
        return res

    table = {"base": score()}
    print("base", table["base"], flush=True)
    for uname, path in units.items():
        z = np.load(path)
        ad = {
            nm: LowRankAdapter(name=f"trained:{nm}", a=z[f"{nm}.a"], b=z[f"{nm}.b"]) for nm in names
        }
        mount(k, ad)
        table[uname] = score()
        unmount(k, ad)
        print(uname, table[uname], flush=True)
    gains = {
        u: {c: round(table["base"][c] - v, 4) for c, v in row.items()}
        for u, row in table.items()
        if u != "base"
    }
    out.write_text(
        json.dumps(
            {
                "nll": table,
                "gain_vs_base": gains,
                "n_per_capability": {c: len(v) for c, v in sets.items()},
            },
            indent=1,
        )
    )
    print("gains", json.dumps(gains), flush=True)


if __name__ == "__main__":
    main()
