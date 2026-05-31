#!/usr/bin/env python3
"""One-shot OPERATOR-RUN driver: sign the reaudit-ACCEPT tier_model allowlist backlog.

Context: the 2026-05-30 ``reaudit --include-pre-judge`` sweep classified the 536
pre-judge ``allow_hits`` entries. The 245 that the cicd-judge ACCEPTED are honest
trust-boundary suppressions that still lack signed judge metadata. This driver runs
``elspeth-lints justify`` for each so they acquire a verdict + HMAC signature.

OPERATOR-ONLY. The HMAC signing key is operator-custody (see CLAUDE.md "CICD Judge
Gate: HMAC Key Custody" + filigree O1 elspeth-b3a3335c9f). An autonomous agent must
NOT run this in signing mode. Required environment for a real run:
  - ELSPETH_JUDGE_METADATA_HMAC_KEY   (signing; operator-held)
  - OPENROUTER_API_KEY                (the judge LLM)

Each entry is RE-JUDGED by ``justify`` (the judge is always called; that is the
guardrail). On ACCEPT it is signed + written; an entry that now flips to BLOCK
fails closed — it is logged, NOT signed. The driver is idempotent: entries that
already carry ``judge_verdict`` are skipped, so a re-run resumes after an interrupt.

Modes:
  --print-only   Print the justify command per entry; run nothing. (Free; agent-safe.)
  --dry-run      Run ``justify --dry-run`` (judge, no sign). Needs OPENROUTER_API_KEY only.
  (default)      Run ``justify`` for real (judge + sign). Needs BOTH keys. OPERATOR.

Usage:
  PYTHONPATH=elspeth-lints/src python scripts/cicd/sign_accept_backlog.py [--print-only|--dry-run] [--limit N]
"""

from __future__ import annotations

import argparse
import contextlib
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SIGNLIST = REPO_ROOT / "notes" / "reaudit-accept-signlist-2026-05-30.txt"
DEFAULT_ALLOWLIST_DIR = REPO_ROOT / "config" / "cicd" / "enforce_tier_model"
DEFAULT_SOURCE_ROOT = REPO_ROOT / "src" / "elspeth"

_VALID_RULE_IDS = {f"R{i}" for i in range(0, 20)} | {f"L{i}" for i in range(0, 10)}


def _parse_key(key: str) -> tuple[str, str, str, str | None]:
    """Split a tier_model allowlist key into (file_path, rule, symbol, fingerprint)."""
    parts = key.split(":")
    fingerprint = None
    if parts and parts[-1].startswith("fp="):
        fingerprint = parts[-1].split("=", 1)[1]
        parts = parts[:-1]
    file_path = parts[0]
    rule = parts[1] if len(parts) > 1 else ""
    symbol = ".".join(parts[2:]) if len(parts) > 2 else ""
    return file_path, rule, symbol, fingerprint


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--print-only", action="store_true", help="Print commands; run nothing (free).")
    mode.add_argument("--dry-run", action="store_true", help="Run justify --dry-run (judge, no sign).")
    ap.add_argument("--limit", type=int, default=None, help="Process only the first N entries.")
    ap.add_argument("--signlist", type=Path, default=DEFAULT_SIGNLIST)
    ap.add_argument("--allowlist-dir", type=Path, default=DEFAULT_ALLOWLIST_DIR)
    ap.add_argument("--root", type=Path, default=DEFAULT_SOURCE_ROOT)
    args = ap.parse_args()

    sys.path.insert(0, str(REPO_ROOT / "elspeth-lints" / "src"))
    from elspeth_lints.core import allowlist as allowlist_mod

    allowlist = allowlist_mod.load_allowlist(args.allowlist_dir, valid_rule_ids=_VALID_RULE_IDS)
    by_key = {entry.key: entry for entry in allowlist.entries}

    keys = [ln.strip() for ln in args.signlist.read_text().splitlines() if ln.strip() and not ln.startswith("#")]
    if args.limit is not None:
        keys = keys[: args.limit]

    tally: Counter[str] = Counter()
    failures: list[tuple[str, str]] = []

    for idx, key in enumerate(keys, 1):
        entry = by_key.get(key)
        if entry is None:
            tally["MISSING"] += 1
            failures.append((key, "not present in current allowlist (rotated/removed?)"))
            continue
        if entry.judge_verdict is not None:
            tally["SKIP_ALREADY_JUDGED"] += 1
            continue

        file_path, rule, symbol, fingerprint = _parse_key(key)
        cmd = [
            sys.executable,
            "-m",
            "elspeth_lints.core.cli",
            "justify",
            "--root",
            str(args.root),
            "--allowlist-dir",
            str(args.allowlist_dir),
            "--file-path",
            file_path,
            "--rule",
            rule,
            "--symbol",
            symbol,
            "--owner",
            entry.owner,
            "--rationale",
            entry.reason,
            "--format",
            "json",
        ]
        if fingerprint:
            cmd += ["--fingerprint", fingerprint]
        if args.dry_run:
            cmd.append("--dry-run")

        if args.print_only:
            import shlex

            print(f"[{idx}/{len(keys)}] " + " ".join(shlex.quote(c) for c in cmd))
            tally["PRINTED"] += 1
            continue

        proc = subprocess.run(cmd, capture_output=True, text=True)
        verdict = "UNKNOWN"
        with contextlib.suppress(json.JSONDecodeError):
            verdict = (json.loads(proc.stdout) or {}).get("verdict", "UNKNOWN")
        if proc.returncode == 0 and verdict == "ACCEPTED":
            tally["SIGNED" if not args.dry_run else "WOULD_SIGN"] += 1
        else:
            tally["BLOCKED_OR_FAILED"] += 1
            failures.append((key, f"rc={proc.returncode} verdict={verdict} :: {proc.stderr.strip()[:200]}"))
        print(f"[{idx}/{len(keys)}] {verdict:9s} {key}")

    print("\n=== summary ===")
    for k, v in tally.most_common():
        print(f"  {v:4d}  {k}")
    if failures:
        print(f"\n=== {len(failures)} need attention (flipped to BLOCK / failed / missing) ===")
        for key, why in failures:
            print(f"  - {key}\n      {why}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
