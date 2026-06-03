"""Compute redaction snapshot direction (weaken vs strengthen) for CI.

Reads base and head snapshot JSON files. Compares ``sensitive_path_count``
across changed entries. Prints diagnostic information to stdout AND
appends ``direction=<value>`` to a GitHub Actions output file
(``GITHUB_OUTPUT`` semantics).

Used by ``.github/workflows/composer-redaction-gate.yml`` (spec §4.4.5,
Task 18). The logic lives here rather than inline in the workflow so it
can be exercised by the four-combination test suite
(``tests/unit/web/composer/test_label_gate_direction.py``) via
``subprocess.run``.

Direction semantics
-------------------
For each manifest entry present in either snapshot:

* If ``base[entry] != head[entry]`` it is a *changed entry*.
* ``base_total`` = sum of ``sensitive_path_count`` for changed entries on
  the base side (entries removed in head contribute their base count).
* ``head_total`` = sum of ``sensitive_path_count`` for changed entries on
  the head side (entries added in head contribute their head count).
* ``direction = "weaken"`` if ``head_total < base_total`` (coverage
  reduced); otherwise ``direction = "strengthen"``.

This script is observational — it always exits 0 even on the weakening
direction. The direction string is the output the next workflow step
consumes for label assertion. Missing entries on either side are treated
as having ``sensitive_path_count = 0`` for the sum calculation, which
correctly attributes removal as base-side coverage that disappeared and
addition as head-side coverage that appeared.

Usage
-----
::

    python3 scripts/cicd/check_redaction_direction.py \\
        --base-snapshot base.json \\
        --head-snapshot head.json \\
        --output-file "$GITHUB_OUTPUT"

Exit code is always 0 (observational). Errors reading the inputs raise
and exit nonzero — that's a CI infrastructure failure, not a policy
verdict.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any


def _load_snapshot(path: Path) -> Mapping[str, Mapping[str, Any]]:
    """Load a snapshot JSON file. Returns the top-level mapping.

    The snapshot is Tier 1 data (our own audit artefact) — if it is
    malformed, we crash. ``json.loads`` raises ``ValueError`` on parse
    failure and that propagates out of ``main`` so the CI step fails.
    """
    raw = path.read_text(encoding="utf-8")
    data = json.loads(raw)
    if not isinstance(data, dict):
        raise ValueError(f"snapshot at {path} is not a JSON object (got {type(data).__name__})")
    return data


def compute_direction(
    base: Mapping[str, Mapping[str, Any]],
    head: Mapping[str, Mapping[str, Any]],
) -> tuple[str, list[str], int, int]:
    """Return (direction, changed_entries, base_total, head_total).

    ``direction`` is ``"weaken"`` when ``head_total < base_total`` and
    ``"strengthen"`` otherwise. Callers should treat ``"strengthen"`` as
    the default (covers no-change-in-count cases such as entry rename
    where the head count equals the base count for changed entries).
    """
    all_keys = set(base) | set(head)
    changed_entries = sorted(k for k in all_keys if base.get(k) != head.get(k))

    base_total = sum(int(base.get(k, {}).get("sensitive_path_count", 0)) for k in changed_entries)
    head_total = sum(int(head.get(k, {}).get("sensitive_path_count", 0)) for k in changed_entries)

    direction = "weaken" if head_total < base_total else "strengthen"
    return direction, changed_entries, base_total, head_total


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-snapshot",
        required=True,
        type=Path,
        help="Path to the base-branch snapshot JSON file.",
    )
    parser.add_argument(
        "--head-snapshot",
        required=True,
        type=Path,
        help="Path to the head (PR) snapshot JSON file.",
    )
    parser.add_argument(
        "--output-file",
        required=True,
        type=Path,
        help=("Path to a file (typically $GITHUB_OUTPUT) to which 'direction=<value>' will be appended."),
    )
    args = parser.parse_args(argv)

    base = _load_snapshot(args.base_snapshot)
    head = _load_snapshot(args.head_snapshot)

    direction, changed, base_total, head_total = compute_direction(base, head)

    print(f"Changed entries: {changed}")
    print(f"Base sensitive_path_count (changed entries): {base_total}")
    print(f"Head sensitive_path_count (changed entries): {head_total}")
    print(f"Direction: {direction}")

    with args.output_file.open("a", encoding="utf-8") as handle:
        handle.write(f"direction={direction}\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
