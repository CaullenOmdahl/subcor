"""Paired bootstrap on two per-item JSONL files (arm A = baseline, arm B = treatment).

Items are joined on `id`; only ids present (and ok) in both files are used, and the
counts of unmatched ids are reported. The statistic is mean(B) - mean(A) of a per-item
0/1 (or float) metric; resampling is over items (paired), 10k resamples, seed 0.

Metric keys:
  gsm8k:  exact_match_flexible (default) | exact_match_strict
  ifeval: prompt_level_strict_acc (default) | prompt_level_loose_acc |
          inst_level_strict_acc | inst_level_loose_acc   (inst-level: per-item mean of
          the instruction list; the paired unit stays the prompt)
  any:    activated  (1 if selected_units non-empty)

  python -I paired_bootstrap.py A.jsonl B.jsonl --metric exact_match_flexible
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

import common  # noqa: E402

DEFAULT_METRIC = {"gsm8k": "exact_match_flexible", "ifeval": "prompt_level_strict_acc"}


def value(row: dict[str, Any], metric: str) -> float:
    if metric == "activated":
        return 1.0 if row.get("selected_units") else 0.0
    v = row[metric]
    if isinstance(v, list):
        return sum(map(float, v)) / len(v) if v else 0.0
    return float(v)


def paired_bootstrap(
    a: list[float], b: list[float], *, n_boot: int = 10000, seed: int = 0
) -> dict[str, Any]:
    n = len(a)
    diffs = [y - x for x, y in zip(a, b, strict=True)]
    obs = sum(diffs) / n
    rng = random.Random(seed)
    boots = []
    for _ in range(n_boot):
        s = 0.0
        for _ in range(n):
            s += diffs[rng.randrange(n)]
        boots.append(s / n)
    boots.sort()
    lo, hi = boots[int(0.025 * n_boot)], boots[int(0.975 * n_boot) - 1]
    # two-sided p: how often the centred bootstrap is at least as extreme as observed
    p = sum(1 for d in boots if abs(d - obs) >= abs(obs)) / n_boot
    return {
        "n": n,
        "mean_a": sum(a) / n,
        "mean_b": sum(b) / n,
        "diff_b_minus_a": obs,
        "ci95": [lo, hi],
        "p_two_sided": p,
        "b_wins": sum(1 for d in diffs if d > 0),
        "a_wins": sum(1 for d in diffs if d < 0),
        "ties": sum(1 for d in diffs if d == 0),
    }


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("a", type=Path)
    ap.add_argument("b", type=Path)
    ap.add_argument("--metric", default=None)
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--allow-unmatched",
        action="store_true",
        help="pair the shared ok ids even when an arm has failed/missing items (default: refuse)",
    )
    args = ap.parse_args()

    all_a = {r["id"]: r for r in common.read_jsonl(args.a)}
    all_b = {r["id"]: r for r in common.read_jsonl(args.b)}
    ra = {i: r for i, r in all_a.items() if r.get("ok")}
    rb = {i: r for i, r in all_b.items() if r.get("ok")}
    task = next(iter(ra.values()), {}).get("task")
    metric = args.metric or DEFAULT_METRIC.get(str(task), "activated")
    ids = sorted(set(ra) & set(rb), key=str)
    if not ids:
        raise SystemExit("no shared ok ids")
    unmatched = {
        "failed_a": len(all_a) - len(ra),
        "failed_b": len(all_b) - len(rb),
        "only_in_a": len(set(ra) - set(rb)),
        "only_in_b": len(set(rb) - set(ra)),
    }
    if any(unmatched.values()) and not args.allow_unmatched:
        # An arm whose failed requests are silently dropped is not paired on the same items.
        raise SystemExit(
            f"arms are not paired on the same ok items {unmatched}; rerun the "
            "failed items (the runners resume) or pass --allow-unmatched"
        )
    res = paired_bootstrap(
        [value(ra[i], metric) for i in ids],
        [value(rb[i], metric) for i in ids],
        n_boot=args.n_boot,
        seed=args.seed,
    )
    res.update(
        {
            "task": task,
            "metric": metric,
            "a": str(args.a),
            "b": str(args.b),
            **unmatched,
        }
    )
    print(json.dumps(res, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
