"""Assert correct redaction PR label for a given snapshot-diff direction.

Used by ``.github/workflows/composer-redaction-gate.yml`` (spec §4.4.5,
Task 18). The logic lives here rather than inline in the workflow's bash
block so it can be exercised by the four-combination test suite
(``tests/unit/web/composer/test_label_gate_direction.py``) via
``subprocess.run``.

The rules (rev-2 BLOCKER_B, Appendix B):

* ``direction=weaken`` (coverage reduced):

  - PR labels MUST include ``policy-weaken-justified``.
  - PR body MUST include a ``Redaction policy weakening rationale``
    section (exact phrase, matched via ``in`` on the body string).
  - PR labels MUST NOT include ``policy-strengthen``.

* ``direction=strengthen`` (coverage increased / new entries):

  - PR labels MUST include ``policy-strengthen``.
  - PR labels MUST NOT include ``policy-weaken-justified``.

On success, exits 0. On any rule violation, prints a ``::error::``-prefixed
message to stderr (the GitHub Actions log-annotation convention) naming
both the wrong label and the correct label so the PR author sees the
exact remediation, then exits 1.

Usage
-----
::

    python3 scripts/cicd/assert_redaction_label.py \\
        --direction weaken \\
        --pr-labels-json '["policy-weaken-justified"]' \\
        --pr-body 'Redaction policy weakening rationale: ...'

The ``--pr-body`` argument may be empty. The ``--pr-labels-json``
argument must be a JSON array of strings. Malformed JSON crashes — that's
a workflow misconfiguration, not a policy verdict.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence

WEAKEN_LABEL = "policy-weaken-justified"
STRENGTHEN_LABEL = "policy-strengthen"
RATIONALE_SECTION = "Redaction policy weakening rationale"


def _parse_labels(raw: str) -> Sequence[str]:
    """Parse a JSON array of label strings.

    Tier 3-style boundary: the JSON payload comes from a workflow
    ``toJson`` expression on a GitHub-API-furnished list. We crash on
    malformed JSON (that's a CI-config bug, not a runtime policy
    decision) but treat the contents as trusted strings.
    """
    parsed = json.loads(raw)
    if not isinstance(parsed, list):
        raise ValueError(f"--pr-labels-json must be a JSON array, got {type(parsed).__name__}")
    return [str(item) for item in parsed]


def assert_label_correct(direction: str, labels: Sequence[str], body: str) -> list[str]:
    """Return a list of ``::error::`` messages; empty list means pass."""
    errors: list[str] = []
    label_set = set(labels)

    if direction == "weaken":
        if WEAKEN_LABEL not in label_set:
            errors.append(
                f"::error::snapshot diff shows coverage reduction (weakening); "
                f"PR must carry '{WEAKEN_LABEL}' label "
                f"(current labels: {sorted(label_set) or 'none'})."
            )
        if RATIONALE_SECTION not in body:
            errors.append(f"::error::'{WEAKEN_LABEL}' label requires a '{RATIONALE_SECTION}' section in the PR body.")
        if STRENGTHEN_LABEL in label_set:
            errors.append(
                f"::error::snapshot diff shows weakening; '{STRENGTHEN_LABEL}' is incorrect for this change (use '{WEAKEN_LABEL}' instead)."
            )
    elif direction == "strengthen":
        if STRENGTHEN_LABEL not in label_set:
            errors.append(
                f"::error::redaction snapshot changed (strengthening); "
                f"PR must carry '{STRENGTHEN_LABEL}' label "
                f"(current labels: {sorted(label_set) or 'none'})."
            )
        if WEAKEN_LABEL in label_set:
            errors.append(
                f"::error::snapshot diff shows no coverage reduction; "
                f"do not use '{WEAKEN_LABEL}' for this change "
                f"(use '{STRENGTHEN_LABEL}' instead)."
            )
    else:
        raise ValueError(f"--direction must be 'weaken' or 'strengthen', got {direction!r}")

    return errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--direction",
        required=True,
        choices=("weaken", "strengthen"),
        help="Snapshot-diff direction computed by check_redaction_direction.py.",
    )
    parser.add_argument(
        "--pr-labels-json",
        required=True,
        help="JSON array of PR label name strings (from github.event.pull_request.labels.*.name).",
    )
    parser.add_argument(
        "--pr-body",
        required=True,
        help="PR body text (may be empty).",
    )
    args = parser.parse_args(argv)

    labels = _parse_labels(args.pr_labels_json)
    errors = assert_label_correct(args.direction, labels, args.pr_body)

    if errors:
        for message in errors:
            print(message, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
