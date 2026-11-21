"""Final leak gate on the built datasets, then copy them to a transfer drive.

Only <out>/03-capabilities is copied. Raw, canonical and scrubbed sessions, the blocklist
and the scrub audit stay on this machine. The gate re-runs gitleaks and every scrub rule
(plus the blocklist) over the decoded JSONL and refuses to copy on any hit.

Usage: python publish.py <out_dir> <blocklist.tsv> <dest_dir>
"""

from __future__ import annotations

import json
import re
import shutil
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import scrub  # noqa: E402

README = """# SUBCOR 4B POC chat-history datasets

Built on the workstation from Claude Code + Codex transcripts (scrubbed, 0 leak-gate hits).
PERSONAL DATA, LOCAL ONLY: do not publish, push, or train units you distribute from this.

Layout: <capability>/{train,eval,negative_prompts}.jsonl + stats.json; manifest.json.
Record: {id, session_id, source, project, timestamp, capability, signal_type, tools,
         messages (OpenAI/Qwen3.5 multi-turn, assistant tool_calls with dict arguments),
         tokens_qwen35 (Qwen3.5 chat template incl. the record's tools)}.
- Windows <= 2048 tokens; filter tokens_qwen35 <= 1400 for the single-turn trainer.
- Split by session, newest 20% of each source -> eval; per-project cap 20% per capability.
- signal_type: accepted | error_recovery | correction_resolved.
- Placeholders: <CLIENT_*>, <VENDOR_*>, <SECRET>, <API_KEY>, <TOKEN>, <EMAIL>, <IP>, <MAC>,
  <PHONE>, <NAME>, /home/user.
- Train with assistant-only loss masking; tool messages are context, not targets.
"""


def main() -> None:
    out, blk, dest = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    src = out / "03-capabilities"
    rules = scrub.RULES + scrub.load_blocklist(blk)
    leaks = scrub.run_gitleaks(src)
    by_rule = Counter(x.get("RuleID") for x in leaks)
    leaks.clear()
    hits = Counter()
    ph = re.compile(r"^<[A-Z_]+>$")

    def check(s: str) -> str:
        for rid, rx, _ in rules:
            if rid == "home_path":
                continue
            for m in rx.finditer(s):
                if rid in ("kv_secret", "bearer") and ph.match(m.group(m.lastindex or 0) or ""):
                    continue
                hits[rid] += 1
        return s

    for f in src.rglob("*.json*"):
        if f.suffix == ".jsonl":
            for line in f.open():
                scrub.walk(json.loads(line), check)
        else:
            scrub.walk(json.loads(f.read_text()), check)
    print(json.dumps({"gitleaks": dict(by_rule), "rules": dict(hits)}))
    if by_rule or hits:
        sys.exit("PUBLISH GATE FAILED: nothing copied")
    target = dest / "subcor-4b-poc"
    if target.exists():
        shutil.rmtree(target)
    shutil.copytree(src, target, copy_function=shutil.copyfile)
    (target / "README.md").write_text(README)
    print(f"PUBLISH GATE PASSED: copied to {target}")


if __name__ == "__main__":
    main()
