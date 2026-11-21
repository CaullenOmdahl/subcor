"""Deterministic local scrub of canonical sessions, with an independent re-scan gate.

1. gitleaks scans the canonical sessions; its findings (raw secrets) go to a chmod-600 temp
   report that is deleted as soon as the replacements are built.
2. Every string in every session is rewritten: gitleaks secrets, built-in regex rules
   (keys, tokens, PEM blocks, JWTs, URL credentials, emails, IPs, MACs, phones, home paths)
   and a private blocklist file (client/personal names -> typed placeholders).
3. Audit log (chmod 600): {file, rule, count, sha256_prefix} only, never the matched value.
4. Gate: gitleaks re-scan of the output plus re-running every rule must give 0 hits.

Usage: python scrub.py <canonical_dir> <out_dir> <blocklist.tsv> <audit.jsonl>
  blocklist.tsv: one "<regex>\t<replacement>" per line, case-insensitive, '#' comments.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path

RULES: list[tuple[str, re.Pattern, str]] = [
    ("pem_private_key", re.compile(r"-----BEGIN [A-Z0-9 ]*PRIVATE KEY-----[\s\S]*?-----END [A-Z0-9 ]*PRIVATE KEY-----"), "<PRIVATE_KEY>"),
    ("jwt", re.compile(r"\beyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}"), "<JWT>"),
    ("anthropic_key", re.compile(r"\bsk-ant-[A-Za-z0-9_-]{20,}"), "<API_KEY>"),
    ("openai_key", re.compile(r"\bsk-(?:proj-)?[A-Za-z0-9_-]{20,}"), "<API_KEY>"),
    ("github_token", re.compile(r"\b(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,})"), "<API_KEY>"),
    ("slack_token", re.compile(r"\bxox[abprs]-[A-Za-z0-9-]{10,}"), "<API_KEY>"),
    ("aws_key", re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b"), "<API_KEY>"),
    ("google_key", re.compile(r"\bAIza[0-9A-Za-z_-]{35}\b"), "<API_KEY>"),
    ("hf_token", re.compile(r"\bhf_[A-Za-z0-9]{30,}\b"), "<API_KEY>"),
    ("gitlab_token", re.compile(r"\bglpat-[A-Za-z0-9_-]{20,}"), "<API_KEY>"),
    ("stripe_key", re.compile(r"\b(?:sk|rk|pk)_(?:live|test)_[A-Za-z0-9]{16,}"), "<API_KEY>"),
    ("bearer", re.compile(r"(?i)\b(bearer\s+)[A-Za-z0-9._~+/=-]{16,}"), r"\1<TOKEN>"),
    ("kv_secret", re.compile(
        r"(?i)\b((?:api[_-]?key|access[_-]?key|secret[_-]?key|access[_-]?token|auth[_-]?token|refresh[_-]?token|"
        r"client[_-]?secret|secret|password|passwd|pwd|token|private[_-]?key)[\"']?\s*[:=]\s*[\"']?)"
        r"(?!<)([^\s\"',;}{)(\]\[<>]{8,})"), r"\1<SECRET>"),
    ("url_creds", re.compile(r"\b([a-z][a-z0-9+.-]*://)[^/\s:@\"']+:[^/\s@\"']+@"), r"\1<CREDS>@"),
    ("email", re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"), "<EMAIL>"),
    ("ipv4", re.compile(r"\b(?!127\.0\.0\.1\b)(?!0\.0\.0\.0\b)(?:25[0-5]|2[0-4]\d|1?\d?\d)(?:\.(?:25[0-5]|2[0-4]\d|1?\d?\d)){3}\b"), "<IP>"),
    ("mac_addr", re.compile(r"\b[0-9A-Fa-f]{2}(?::[0-9A-Fa-f]{2}){5}\b"), "<MAC>"),
    ("phone", re.compile(r"(?<![\w.])\+\d[\d \-]{8,14}\d\b"), "<PHONE>"),
    ("home_path", re.compile(r"/(?:home|Users)/caullen\b"), "/home/user"),
    ("username", re.compile(r"(?i)\bcaullen\b"), "user"),
    ("surname", re.compile(r"(?i)\bomdahl\b"), "<NAME>"),
]


def load_blocklist(path: Path) -> list[tuple[str, re.Pattern, str]]:
    out = []
    for i, line in enumerate(path.read_text().splitlines()):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        pat, _, rep = line.partition("\t")
        out.append((f"blocklist:{i + 1}", re.compile(pat, re.I), rep or "<REDACTED>"))
    return out


def run_gitleaks(target: Path) -> list[dict]:
    fd, rpt = tempfile.mkstemp(prefix=".gitleaks-", suffix=".json", dir=target.parent)
    os.close(fd)
    os.chmod(rpt, 0o600)
    try:
        subprocess.run(["gitleaks", "dir", str(target), "--report-format", "json", "--report-path", rpt,
                        "--no-banner", "--log-level", "error", "--exit-code", "0"], check=True)
        return json.loads(Path(rpt).read_text() or "[]")
    finally:
        os.unlink(rpt)


def _h(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8", "ignore")).hexdigest()[:10]


def scrub_string(s: str, literal: list[tuple[str, str]], rules, counts: Counter, hashes: dict) -> str:
    for secret, rid in literal:
        if secret in s:
            counts[rid] += s.count(secret)
            hashes.setdefault(rid, set()).add(_h(secret))
            s = s.replace(secret, "<SECRET>")
    for rid, rx, rep in rules:
        def _sub(m, rid=rid, rx=rx, rep=rep):
            counts[rid] += 1
            hashes.setdefault(rid, set()).add(_h(m.group(0)))
            return m.expand(rep)
        s = rx.sub(_sub, s)
    return s


def walk(o, fn):
    if isinstance(o, str):
        return fn(o)
    if isinstance(o, list):
        return [walk(x, fn) for x in o]
    if isinstance(o, dict):
        return {k: walk(v, fn) for k, v in o.items()}
    return o


def _secret_variants(sec: str) -> list[str]:
    out = {sec}
    try:
        out.add(json.loads('"' + sec + '"'))
    except Exception:
        pass
    return [v for v in out if len(v) >= 6]


def main() -> None:
    src, dst, blk, audit = map(Path, sys.argv[1:5])
    dst.mkdir(parents=True, exist_ok=True)
    rules = RULES + load_blocklist(blk)
    findings = run_gitleaks(src)
    per_file: dict[str, list[tuple[str, str]]] = {}
    for f in findings:
        name = Path(f.get("File", "")).name
        for v in _secret_variants(f.get("Secret", "")):
            per_file.setdefault(name, []).append((v, f"gitleaks:{f.get('RuleID')}"))
    findings.clear()
    audit.touch(mode=0o600)
    os.chmod(audit, 0o600)
    total = Counter()
    with audit.open("w") as log:
        for f in sorted(src.glob("*.json")):
            lit = sorted(set(per_file.get(f.name, [])), key=lambda x: -len(x[0]))
            counts: Counter = Counter()
            hashes: dict = {}
            data = walk(json.loads(f.read_text()), lambda s: scrub_string(s, lit, rules, counts, hashes))
            (dst / f.name).write_text(json.dumps(data, ensure_ascii=False))
            for rid, n in counts.items():
                log.write(json.dumps({"file": f.name, "rule": rid, "count": n,
                                      "sha256_prefixes": sorted(hashes.get(rid, ()))[:20]}) + "\n")
            total.update(counts)
    per_file.clear()
    # ---- extra gitleaks passes: replacements can expose secrets gitleaks missed in context
    with audit.open("a") as log:
        for _ in range(3):
            leftover = run_gitleaks(dst)
            if not leftover:
                break
            extra: dict[str, list[tuple[str, str]]] = {}
            for x in leftover:
                for v in _secret_variants(x.get("Secret", "")):
                    extra.setdefault(Path(x.get("File", "")).name, []).append((v, f"gitleaks2:{x.get('RuleID')}"))
            leftover.clear()
            for name, lit in extra.items():
                counts, hashes = Counter(), {}
                p = dst / name
                data = walk(json.loads(p.read_text()), lambda s: scrub_string(s, lit, [], counts, hashes))
                p.write_text(json.dumps(data, ensure_ascii=False))
                for rid, n in counts.items():
                    log.write(json.dumps({"file": name, "rule": rid, "count": n,
                                          "sha256_prefixes": sorted(hashes.get(rid, ()))[:20]}) + "\n")
                total.update(counts)
    # ---- gate (on decoded strings, the same representation the scrub rewrote)
    leftover = run_gitleaks(dst)
    left_by_rule = Counter(x.get("RuleID") for x in leftover)
    leftover.clear()
    rule_hits = Counter()
    ok_placeholder = re.compile(r"^<[A-Z_]+>$")

    def _check(s: str) -> str:
        for rid, rx, rep in rules:
            if rid == "home_path":
                continue
            for m in rx.finditer(s):
                if rid in ("kv_secret", "bearer") and ok_placeholder.match(m.group(m.lastindex or 0) or ""):
                    continue
                rule_hits[rid] += 1
        return s

    for f in dst.glob("*.json"):
        walk(json.loads(f.read_text()), _check)
    print(json.dumps({"replaced": dict(total.most_common()), "gate_gitleaks": dict(left_by_rule),
                      "gate_rules": dict(rule_hits)}, indent=1))
    if left_by_rule or rule_hits:
        sys.exit("SCRUB GATE FAILED: leftover hits above")
    print("SCRUB GATE PASSED")


if __name__ == "__main__":
    main()
