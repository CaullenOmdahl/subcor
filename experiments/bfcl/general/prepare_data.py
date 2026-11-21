"""Download the general-eval datasets into <scratch>/general/data (never into the repo).

  google/IFEval (train = the 541 eval prompts)      -> ifeval.jsonl
  openai/gsm8k main test (1319)                      -> gsm8k_test.jsonl
  openai/gsm8k main train (probe-set source only)    -> gsm8k_train.jsonl
  databricks/databricks-dolly-15k (probe-set source) -> dolly15k.jsonl
then builds router_probe_prompts.jsonl (build_probe_set.py) and checks the frozen GSM8K
subset hashes. Run with the eval venv:  $VENV/bin/python -I prepare_data.py --out-dir ...
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--hf-home", type=Path, default=None)
    ap.add_argument("--exclude", type=Path, action="append", default=[], help="passed to build_probe_set.py")
    args = ap.parse_args()
    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    if args.hf_home:
        os.environ["HF_HOME"] = str(args.hf_home)
    import datasets

    datasets.load_dataset("google/IFEval", split="train").to_json(str(out / "ifeval.jsonl"))
    datasets.load_dataset("openai/gsm8k", "main", split="test").to_json(str(out / "gsm8k_test.jsonl"))
    datasets.load_dataset("openai/gsm8k", "main", split="train").to_json(str(out / "gsm8k_train.jsonl"))
    datasets.load_dataset("databricks/databricks-dolly-15k", split="train").to_json(str(out / "dolly15k.jsonl"))
    cmd = [sys.executable, "-I", str(HERE / "build_probe_set.py"), "--dolly", str(out / "dolly15k.jsonl"),
           "--gsm8k-train", str(out / "gsm8k_train.jsonl"), "--out", str(out / "router_probe_prompts.jsonl")]
    for e in args.exclude:
        cmd += ["--exclude", str(e)]
    subprocess.run(cmd, check=True, cwd=out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
