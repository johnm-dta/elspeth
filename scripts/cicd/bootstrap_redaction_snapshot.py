"""Regenerate the redaction policy snapshot file (spec §4.4.3).

Reads ``MANIFEST``, computes ``_entry_hash`` for each entry, and writes the
result to ``tests/unit/web/composer/redaction_policy_snapshot.json``.

Idempotent: re-running on an unchanged ``MANIFEST`` produces a byte-identical
output (``sort_keys=True`` on ``json.dumps``; manifest is iterated in sorted
key order via ``compute_manifest_snapshot``).

Usage::

    # Dry-run (print what would be written, do not modify the file):
    .venv/bin/python scripts/cicd/bootstrap_redaction_snapshot.py

    # Write the snapshot (commit the change after reviewing the diff):
    .venv/bin/python scripts/cicd/bootstrap_redaction_snapshot.py --write

After regenerating, review the diff against your manifest changes before
merging.  The label-gate CI step (Task 18 / spec §4.4.5) requires a
direction-appropriate PR label when the snapshot changes.

Idempotency verification::

    .venv/bin/python scripts/cicd/bootstrap_redaction_snapshot.py --write
    .venv/bin/python scripts/cicd/bootstrap_redaction_snapshot.py --write
    git diff --exit-code tests/unit/web/composer/redaction_policy_snapshot.json
    # Must exit 0 (no diff after the second run).

Implementation note — shared hash logic
---------------------------------------
``_entry_hash`` and ``compute_manifest_snapshot`` live in the test helper
module ``tests/unit/web/composer/_adequacy_helpers.py`` rather than here so
the bootstrap script and the adequacy-guard test file share a single
canonical implementation.  This script adds ``tests/`` to ``sys.path`` (via
an absolute path resolved from ``__file__``) to make the import possible
without requiring the tests package to be installed.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Resolve the project root and inject ``tests/`` into sys.path so we can
# import from the test helper module without installing the test package.
# ``__file__`` is ``<project_root>/scripts/cicd/bootstrap_redaction_snapshot.py``
# so parents[2] is the project root.
# ---------------------------------------------------------------------------
_SCRIPT_PATH = Path(__file__).resolve()
_PROJECT_ROOT = _SCRIPT_PATH.parents[2]
_TESTS_DIR = _PROJECT_ROOT / "tests"
if str(_TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(_TESTS_DIR))

# These imports depend on the sys.path injection above.
from elspeth.web.composer.redaction import MANIFEST  # noqa: E402
from unit.web.composer._adequacy_helpers import compute_manifest_snapshot  # type: ignore[import-not-found]  # noqa: E402

_SNAPSHOT_PATH = _PROJECT_ROOT / "tests" / "unit" / "web" / "composer" / "redaction_policy_snapshot.json"


def _build_snapshot_text() -> str:
    """Compute the canonical JSON text for the current MANIFEST state."""
    snapshot = compute_manifest_snapshot(MANIFEST)
    return json.dumps(snapshot, indent=2, sort_keys=True) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Regenerate the redaction policy snapshot file (spec §4.4.3).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--write",
        action="store_true",
        default=False,
        help=(
            "Write the computed snapshot to "
            "tests/unit/web/composer/redaction_policy_snapshot.json. "
            "Without this flag the script prints the computed content and exits "
            "without modifying any file (dry-run mode)."
        ),
    )
    args = parser.parse_args()

    snapshot_text = _build_snapshot_text()

    if args.write:
        _SNAPSHOT_PATH.write_text(snapshot_text)
        print(f"Written: {_SNAPSHOT_PATH}")

        # Verify idempotency: recompute and compare.
        recomputed = _build_snapshot_text()
        if recomputed != snapshot_text:
            # This is an internal consistency error — crash immediately.
            raise RuntimeError(
                "Idempotency check failed: recomputing _build_snapshot_text() "
                "after writing produced a different result. "
                "The hash encoding has non-determinism — fix _entry_hash before "
                "committing the snapshot."
            )
        print("Idempotency verified: second computation is byte-identical.")
    else:
        print("--- computed snapshot (dry-run, not written) ---")
        print(snapshot_text, end="")
        print("--- end ---")
        print("\nTo write the file, re-run with --write.  Review the diff before committing.")


if __name__ == "__main__":
    main()
