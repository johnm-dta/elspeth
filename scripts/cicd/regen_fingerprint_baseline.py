#!/usr/bin/env python3
"""Regenerate the elspeth-lints fingerprint baseline — gated on a green SIGNED enforce check.

The committed ``tests/unit/elspeth_lints/fixtures/fingerprint_baseline.json`` pins the
*raw* (pre-suppression) fingerprints of every shipped elspeth-lints rule. When source
edits add or rotate ``trust_tier.tier_model`` findings, the raw baseline drifts and
``test_baseline_capture_is_self_consistent`` goes red.

Re-capturing the baseline blesses whatever raw findings exist *right now*. That is only
safe when those findings are legitimately SIGNED-SUPPRESSED in the enforce allowlist —
otherwise we would launder an unsuppressed (or unsigned) trust-tier violation into a
green baseline. So this script REFUSES to regenerate unless the signed enforce gate
(``trust_tier.tier_model`` run in ``required`` signature-verify mode, under the
operator-held ``ELSPETH_JUDGE_METADATA_HMAC_KEY``) is green. It never signs anything:
signing is the operator's attestation, made separately via ``sign-judge-signatures``.

Order of operations:
  1. Verify the signed enforce gate is green (HMAC signatures checked, not just shape).
     On failure: abort, print the diagnose/sign guidance, change nothing.
  2. Re-capture every rule's raw fingerprints (the exact ``capture_all`` the test uses)
     and write the baseline fixture.
  3. Re-run ``test_baseline_capture_is_self_consistent`` to prove the fixture now matches.

Requirements:
  * Python 3.13 — baselines are interpreter-version-specific (see the test's skipif).
  * ``ELSPETH_JUDGE_METADATA_HMAC_KEY`` in the environment.
  * Run from a checkout whose ``.venv`` is the canonical 3.13 venv (the capture shells
    out to ``.venv/bin/python``, mirroring CI and the test).

Usage:
  ELSPETH_JUDGE_METADATA_HMAC_KEY=... python scripts/cicd/regen_fingerprint_baseline.py
  ELSPETH_JUDGE_METADATA_HMAC_KEY=... python scripts/cicd/regen_fingerprint_baseline.py --commit
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import NoReturn, cast

REPO_ROOT = Path(__file__).resolve().parents[2]
VENV_PYTHON = REPO_ROOT / ".venv" / "bin" / "python"
ELSPETH_LINTS_SRC = REPO_ROOT / "elspeth-lints" / "src"
TEST_FILE = REPO_ROOT / "tests" / "unit" / "elspeth_lints" / "test_allowlist_loader_unification.py"
BASELINE = REPO_ROOT / "tests" / "unit" / "elspeth_lints" / "fixtures" / "fingerprint_baseline.json"
HMAC_ENV = "ELSPETH_JUDGE_METADATA_HMAC_KEY"

# Mirror the CI "Run trust-tier elspeth-lints rule" step exactly: same rule, same root,
# no --allowlist-dir (the rule's default is the signed enforce allowlist CI uses).
GATE_RULE = "trust_tier.tier_model"
GATE_ROOT = "src/elspeth"

DIAGNOSE_HINT = (
    "Signed enforce gate is RED. The raw baseline cannot be blessed while a tier_model\n"
    "finding is unsuppressed or its signature does not verify — that would launder a\n"
    "real trust-tier violation into a green baseline.\n\n"
    "Diagnose:\n"
    "  diagnose-judge-signatures --root src/elspeth --allowlist-dir config/cicd/enforce_tier_model\n\n"
    "If the diagnosis is genuine new drift you have adjudicated, sign it (operator shell,\n"
    "key loaded), then re-run this script:\n"
    "  sign-judge-signatures --root src/elspeth --allowlist-dir config/cicd/enforce_tier_model \\\n"
    '      --env-file /path/to/operator.env --owner "$USER"\n'
)


def _fail(message: str) -> NoReturn:
    print(f"\n✗ {message}", file=sys.stderr)
    raise SystemExit(1)


def _check_preconditions() -> None:
    if sys.version_info[:2] != (3, 13):
        _fail(
            f"baselines are Python-3.13-specific; this interpreter is "
            f"{sys.version_info.major}.{sys.version_info.minor}. Run under the 3.13 venv."
        )
    if not VENV_PYTHON.exists():
        _fail(f"expected canonical venv interpreter at {VENV_PYTHON} (the capture shells out to it).")
    if not os.environ.get(HMAC_ENV):
        _fail(
            f"{HMAC_ENV} is not set. The signed enforce gate cannot be verified without it,\n"
            f"  and an unverified gate must not bless a baseline. Load the operator env and retry."
        )
    if not TEST_FILE.exists() or not BASELINE.exists():
        _fail("could not locate the baseline test file or fixture — is this the elspeth repo root?")


def _verify_signed_gate() -> None:
    """Run the enforce gate in `required` signature mode. Exit 0 == every tier_model
    finding is suppressed by a valid, HMAC-verified allowlist entry."""
    print(f"→ Verifying signed enforce gate: {GATE_RULE} (required signature mode) ...\n")
    env = {
        **os.environ,
        "PYTHONPATH": str(ELSPETH_LINTS_SRC),
        "ELSPETH_JUDGE_METADATA_SIGNATURE_VERIFY_MODE": "required",
    }
    result = subprocess.run(
        [
            str(VENV_PYTHON),
            "-m",
            "elspeth_lints.core.cli",
            "check",
            "--rules",
            GATE_RULE,
            "--root",
            GATE_ROOT,
        ],
        cwd=str(REPO_ROOT),
        env=env,
        check=False,
    )
    if result.returncode != 0:
        _fail(DIAGNOSE_HINT)
    print("\n✓ Signed enforce gate is GREEN — raw findings are signature-verified suppressed.\n")


def _load_capture_all() -> Callable[..., dict[str, list[dict[str, str]]]]:
    """Load the test module's exact `capture_all` so the regen matches the test byte-for-byte."""
    spec = importlib.util.spec_from_file_location("_baseline_capture_module", TEST_FILE)
    if spec is None or spec.loader is None:
        _fail(f"could not load {TEST_FILE}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return cast("Callable[..., dict[str, list[dict[str, str]]]]", module.capture_all)


def _regenerate() -> None:
    capture_all = _load_capture_all()
    print("→ Re-capturing raw fingerprints across all shipped rules (this runs every rule) ...")
    with tempfile.TemporaryDirectory() as tmp:
        empty_allowlist = Path(tmp) / "empty-allowlist"
        empty_allowlist.mkdir()
        captured = capture_all(allowlist_dir=empty_allowlist)
    BASELINE.write_text(json.dumps(captured, indent=2) + "\n", encoding="utf-8")
    total = sum(len(v) for v in captured.values())
    print(f"✓ Wrote {BASELINE.relative_to(REPO_ROOT)} ({total} findings across {len(captured)} rules).\n")


def _confirm() -> None:
    print("→ Re-running test_baseline_capture_is_self_consistent to confirm ...\n")
    result = subprocess.run(
        [
            str(VENV_PYTHON),
            "-m",
            "pytest",
            str(TEST_FILE),
            "-k",
            "test_baseline_capture_is_self_consistent",
            "-q",
        ],
        cwd=str(REPO_ROOT),
        check=False,
    )
    if result.returncode != 0:
        _fail("self-consistency test still failing after regen — do NOT commit; investigate.")
    print("\n✓ Baseline self-consistency test passes.\n")


def _git_status_and_maybe_commit(*, commit: bool) -> None:
    diff = subprocess.run(
        ["git", "diff", "--stat", "--", str(BASELINE)],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    if not diff.stdout.strip():
        print("→ Baseline already matched — no change to commit.")
        return
    print("→ Baseline change:\n" + diff.stdout)
    if not commit:
        print(
            "Left uncommitted for review. To land it:\n"
            f"  git add {BASELINE.relative_to(REPO_ROOT)}\n"
            '  git commit -m "chore(cicd): regenerate fingerprint baseline for RC5.3 tier_model drift"\n'
        )
        return
    subprocess.run(["git", "add", str(BASELINE)], cwd=str(REPO_ROOT), check=True)
    subprocess.run(
        [
            "git",
            "commit",
            "-m",
            "chore(cicd): regenerate fingerprint baseline after signed enforce gate verified green",
        ],
        cwd=str(REPO_ROOT),
        check=True,
    )
    print("✓ Committed the regenerated baseline.")


def _load_env_file(env_file: Path | None) -> None:
    """Load signing-relevant keys from a dotenv file via the shared lints loader.

    Same semantics as ``sign-judge-signatures --env-file``: existing environment
    values win, unrelated keys are ignored, secret values are never printed.
    """
    if env_file is None:
        return
    sys.path.insert(0, str(ELSPETH_LINTS_SRC))
    from elspeth_lints.core.cli import _load_judge_signing_env_file

    _load_judge_signing_env_file(env_file)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--commit",
        action="store_true",
        help="git-commit the regenerated fixture (default: leave uncommitted for review).",
    )
    parser.add_argument(
        "--env-file",
        type=Path,
        default=None,
        help=(
            f"Dotenv file containing signing-relevant keys such as {HMAC_ENV}. Existing environment values win; unrelated keys are ignored."
        ),
    )
    args = parser.parse_args()

    _load_env_file(args.env_file)
    _check_preconditions()
    _verify_signed_gate()  # refuses (exit 1) if the signed gate is not green
    _regenerate()
    _confirm()
    _git_status_and_maybe_commit(commit=args.commit)
    print("Done.")


if __name__ == "__main__":
    main()
