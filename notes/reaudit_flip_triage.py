#!/usr/bin/env python
"""List + scaffold the WAS_ACCEPTED_NOW_BLOCKED flips from a reaudit run.

Reads the reaudit run-state sidecar (works on a live, in-progress run — the
file is append-only) and emits a triage-ready list of every verdict flip, with
the entry key, the resolved source path, and the judge's fresh rationale, ready
to bucket under the three-way lens:

  1. WAS ALWAYS WRONG          -> remediate code (guard was redundant from the start)
  2. STANDARD LEGITIMATELY RAISED -> remediate code (was fine before; bar moved)
  3. JUDGE OVER-CORRECTING     -> keep code; if 3s cluster, dampen the prompt

Buckets 1+2 both change code; only 3 pushes back. Reading each site's code is
what separates them — this script just lays them out.

Usage:
  .venv/bin/python notes/reaudit_flip_triage.py <run_id> [--full] [--all-divergences]

  --full              print the entire fresh rationale (default truncates to 400 chars)
  --all-divergences   include STILL_AGREES too (default: flips only)
"""

from __future__ import annotations

import collections
import json
import sys
from pathlib import Path

STATE_DIR = Path("config/cicd/enforce_tier_model/.reaudit-state")


def _parse_key(key: str) -> tuple[str, str, str]:
    """key = '<file>:<rule>:<symbol...>:fp=<fp>' -> (file, rule, symbol)."""
    prefix = key.split(":fp=", 1)[0]
    tokens = prefix.split(":")
    for i, tok in enumerate(tokens):
        if tok.endswith(".py"):
            file = ":".join(tokens[: i + 1])
            rule = tokens[i + 1] if i + 1 < len(tokens) else "?"
            symbol = ":".join(tokens[i + 2 :]) if i + 2 < len(tokens) else "?"
            return file, rule, symbol
    return prefix, "?", "?"


def main(argv: list[str]) -> int:
    if not argv or argv[0].startswith("-"):
        print(__doc__)
        return 2
    run_id = argv[0]
    full = "--full" in argv
    all_div = "--all-divergences" in argv

    path = STATE_DIR / f"{run_id}.jsonl"
    if not path.exists():
        print(f"no run-state at {path}", file=sys.stderr)
        return 1

    outcomes = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue  # tolerate a mid-write final line on a live run
        if rec.get("type") == "outcome":
            outcomes.append(rec)

    div_counts = collections.Counter(o.get("divergence") for o in outcomes)
    print(f"# reaudit {run_id}: {len(outcomes)} outcomes scored")
    for k, v in sorted(div_counts.items(), key=lambda kv: (kv[0] or "")):
        print(f"#   {k}: {v}")

    selected = [
        o for o in outcomes
        if all_div or o.get("divergence") not in (None, "STILL_AGREES")
    ]
    print(f"\n# {len(selected)} entr{'y' if len(selected)==1 else 'ies'} to triage "
          f"(bucket each: 1=always-wrong  2=standard-raised  3=judge-wrong)\n")

    for o in selected:
        entry = o.get("entry") or {}
        key = entry.get("key", "?")
        file, rule, symbol = _parse_key(key)
        rationale = o.get("fresh_rationale") or ""
        if not full and len(rationale) > 400:
            rationale = rationale[:400] + " …"
        print(f"[ ] bucket=__  {o.get('divergence')}  ({o.get('cause')})")
        print(f"      key:    {key}")
        print(f"      source: src/elspeth/{file}  ::  {symbol}  [{rule}]")
        print(f"      fresh:  {rationale}")
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
