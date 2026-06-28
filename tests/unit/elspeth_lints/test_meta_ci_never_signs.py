"""Standing meta-gate: the judge-gates CI workflow VERIFIES, it never SIGNS.

Security invariant #4 (spec [O1] / elspeth-2b351cd004): the HMAC signing key
must never be reachable from PR-controlled code, so signing runs only in an
operator-controlled shell and CI keeps *verifying*, never signing. This test is
a regression guard, not a one-time grep — a future edit that re-adds a signing
verb to ``enforce-allowlist-judge-gates.yaml`` fails the suite instead of
slipping past a manual review.

Mirrors the ``test_meta_no_new_bespoke_cicd_enforcer.py`` idiom: assert
properties over the real CI config rather than over a fixture.

Scope note: this parses ``enforce-allowlist-judge-gates.yaml`` *specifically*,
and scans only the bodies of ``run:`` steps (the commands CI actually executes)
rather than the whole file. Two reasons:
  * ``ci.yaml`` carries the token ``sign-judge-signatures`` inside inert
    ``echo "::error..."`` operator-hint strings (drift-repair guidance, not
    invocations) — broadening to every workflow would false-positive on those.
  * the workflow header comment legitimately uses the word "signing" to state
    the verify-only contract; scanning run bodies lets the comment coexist with
    the guard.
"""

from __future__ import annotations

import re
from pathlib import Path

import yaml

# tests/unit/elspeth_lints/test_meta_ci_never_signs.py -> repo root is parents[3].
REPO_ROOT = Path(__file__).resolve().parents[3]
WORKFLOW = REPO_ROOT / ".github/workflows/enforce-allowlist-judge-gates.yaml"

# The two verify-only elspeth-lints gates the workflow is permitted to invoke.
ALLOWED_GATE_SUBCOMMANDS = frozenset({"check-override-rate", "check-judge-quality"})

# Any elspeth-lints subcommand that mints or rotates a judge-metadata HMAC
# signature. None of these may appear in a CI ``run`` body.
SIGNING_VERBS = frozenset({"sign-bundle", "rekey", "justify", "sign-judge-signatures", "migrate-judge-scope"})


def _run_step_bodies(workflow_text: str) -> list[str]:
    """Return the body of every ``run:`` step across all jobs in the workflow."""
    doc = yaml.safe_load(workflow_text)
    bodies: list[str] = []
    for job in doc["jobs"].values():
        for step in job.get("steps", []):
            run = step.get("run")
            if run is not None:
                bodies.append(run)
    return bodies


def _invoked_cli_subcommands(bodies: list[str]) -> set[str]:
    """Extract the subcommand token following each ``elspeth_lints.core.cli`` call."""
    invoked: set[str] = set()
    for body in bodies:
        invoked |= set(re.findall(r"elspeth_lints\.core\.cli\b[\s\\]*([a-z][a-z0-9_-]+)", body))
    return invoked


def _signing_verbs_present(bodies: list[str]) -> set[str]:
    """Return signing verbs that appear as a whole token in any run body.

    Invocation-form-agnostic: catches ``elspeth-lints sign-bundle`` as well as
    ``python -m elspeth_lints.core.cli sign-bundle``.
    """
    present: set[str] = set()
    for verb in SIGNING_VERBS:
        pattern = r"\b" + re.escape(verb) + r"\b"
        if any(re.search(pattern, body) for body in bodies):
            present.add(verb)
    return present


def test_enforce_workflow_exists() -> None:
    """Guard against silently passing because the workflow was renamed/moved."""
    assert WORKFLOW.is_file(), f"expected CI workflow at {WORKFLOW}"


def test_enforce_workflow_invokes_only_verify_gates() -> None:
    """The workflow runs exactly the two verify-only gates and nothing else."""
    bodies = _run_step_bodies(WORKFLOW.read_text(encoding="utf-8"))
    invoked = _invoked_cli_subcommands(bodies)

    # Non-vacuity: the workflow must actually invoke the verify gates, so an
    # empty/garbled parse cannot pass this assertion.
    assert invoked == set(ALLOWED_GATE_SUBCOMMANDS), (
        f"enforce-allowlist-judge-gates.yaml must invoke only {sorted(ALLOWED_GATE_SUBCOMMANDS)}; found {sorted(invoked)}"
    )


def test_enforce_workflow_contains_no_signing_verb() -> None:
    """No signing verb may appear in any CI run body (CI verifies, never signs)."""
    bodies = _run_step_bodies(WORKFLOW.read_text(encoding="utf-8"))
    present = _signing_verbs_present(bodies)

    assert present == set(), (
        f"signing must run only in an operator shell, never in CI; enforce-allowlist-judge-gates.yaml run bodies invoke: {sorted(present)}"
    )


def test_signing_verb_guard_is_non_vacuous() -> None:
    """Prove the guard bites: a workflow that DOES sign is flagged, a clean one is not.

    Without this the green result above could merely mean the scanner never
    matches anything. Here a synthetic workflow whose run body fires
    ``sign-bundle`` (and a second, ``elspeth-lints rekey`` in non-module form)
    is parsed through the exact same pipeline and must be caught.
    """
    dirty = yaml.safe_dump(
        {
            "jobs": {
                "bad": {
                    "steps": [
                        {
                            "run": (
                                "PYTHONPATH=elspeth-lints/src python -m "
                                "elspeth_lints.core.cli \\\n"
                                "  sign-bundle .elspeth/staged-reviews/b.json \\\n"
                                "  --owner ci\n"
                            )
                        },
                        {"run": "elspeth-lints rekey --in b.json --old-key-env A --new-key-env B\n"},
                    ]
                }
            }
        }
    )
    dirty_bodies = _run_step_bodies(dirty)

    assert _signing_verbs_present(dirty_bodies) == {"sign-bundle", "rekey"}
    assert "sign-bundle" in _invoked_cli_subcommands(dirty_bodies)

    clean = yaml.safe_dump(
        {"jobs": {"ok": {"steps": [{"run": ("python -m elspeth_lints.core.cli check-override-rate --max-rate 0.10\n")}]}}}
    )
    clean_bodies = _run_step_bodies(clean)
    assert _signing_verbs_present(clean_bodies) == set()
    assert _invoked_cli_subcommands(clean_bodies) == {"check-override-rate"}
