"""Four-combination direction-by-label tests for the redaction CI gate.

Closes rev-2 BLOCKER_B. Each test invokes the two CI scripts via
``subprocess.run`` against the real Python interpreter — the actual
shell behaviour is exercised, not a re-implementation of the logic.

Combinations covered (per plan body §Task 18):

* (a) weakening diff + ``policy-weaken-justified`` (with rationale) → pass
* (b) weakening diff + ``policy-strengthen`` → fail (direction-mismatch)
* (c) strengthening diff + ``policy-strengthen`` → pass
* (d) strengthening diff + ``policy-weaken-justified`` → fail
       ("no coverage reduction" message)

Snapshot fixtures are tiny synthetic dicts written into ``tmp_path``; the
direction script is path-agnostic so it reads them like the real
snapshot.
"""

from __future__ import annotations

import json
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[4]
CHECK_SCRIPT = REPO_ROOT / "scripts" / "cicd" / "check_redaction_direction.py"
ASSERT_SCRIPT = REPO_ROOT / "scripts" / "cicd" / "assert_redaction_label.py"


def _write_snapshot(path: Path, content: Mapping[str, object]) -> Path:
    path.write_text(json.dumps(content, sort_keys=True), encoding="utf-8")
    return path


def _run_direction(
    tmp_path: Path,
    base: Mapping[str, object],
    head: Mapping[str, object],
) -> tuple[str, subprocess.CompletedProcess[str]]:
    """Run the direction script; return (direction, completed-process)."""
    base_file = _write_snapshot(tmp_path / "base.json", base)
    head_file = _write_snapshot(tmp_path / "head.json", head)
    output_file = tmp_path / "github_output"
    output_file.touch()

    proc = subprocess.run(
        [
            sys.executable,
            str(CHECK_SCRIPT),
            "--base-snapshot",
            str(base_file),
            "--head-snapshot",
            str(head_file),
            "--output-file",
            str(output_file),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    direction = None
    for line in output_file.read_text("utf-8").splitlines():
        if line.startswith("direction="):
            direction = line.split("=", 1)[1]
    assert direction is not None, "check script did not emit direction line"
    return direction, proc


def _run_assert(
    direction: str,
    labels: list[str],
    body: str,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(ASSERT_SCRIPT),
            "--direction",
            direction,
            "--pr-labels-json",
            json.dumps(labels),
            "--pr-body",
            body,
        ],
        capture_output=True,
        text=True,
        check=False,
    )


# Reusable snapshot shapes — keep tiny so the test diffs are obvious.
_BASE_WEAKEN = {
    "tool_a": {"hash": "h1", "sensitive_path_count": 3, "shape": "type_driven"},
    "tool_b": {"hash": "h2", "sensitive_path_count": 2, "shape": "declarative"},
}
_HEAD_WEAKEN = {
    # tool_a sensitive_path_count dropped 3 → 1 (weakening)
    "tool_a": {"hash": "h1b", "sensitive_path_count": 1, "shape": "type_driven"},
    "tool_b": {"hash": "h2", "sensitive_path_count": 2, "shape": "declarative"},
}

_BASE_STRENGTHEN = {
    "tool_a": {"hash": "h1", "sensitive_path_count": 1, "shape": "type_driven"},
}
_HEAD_STRENGTHEN = {
    # tool_a sensitive_path_count rose 1 → 3 (strengthening)
    "tool_a": {"hash": "h1b", "sensitive_path_count": 3, "shape": "type_driven"},
}


def test_combination_a_weakening_with_correct_label_passes(tmp_path: Path) -> None:
    """(a) weakening diff + policy-weaken-justified + rationale → exit 0."""
    direction, _ = _run_direction(tmp_path, _BASE_WEAKEN, _HEAD_WEAKEN)
    assert direction == "weaken"

    body = (
        "## Summary\nDrop one redaction path on tool_a.\n\n"
        "## Redaction policy weakening rationale\n"
        "The dropped path is now covered by a new upstream sanitiser; "
        "no auditor-visible field is at risk."
    )
    proc = _run_assert(direction, ["policy-weaken-justified"], body)
    assert proc.returncode == 0, f"expected exit 0, got {proc.returncode}; stderr={proc.stderr!r}"
    assert proc.stderr == ""


def test_combination_b_weakening_with_strengthen_label_fails(tmp_path: Path) -> None:
    """(b) weakening diff + policy-strengthen → exit 1 (direction-mismatch)."""
    direction, _ = _run_direction(tmp_path, _BASE_WEAKEN, _HEAD_WEAKEN)
    assert direction == "weaken"

    proc = _run_assert(direction, ["policy-strengthen"], "no rationale here")
    assert proc.returncode == 1
    # Must name BOTH the wrong label and the correct one.
    assert "policy-strengthen" in proc.stderr
    assert "policy-weaken-justified" in proc.stderr
    # And must communicate the direction-mismatch framing.
    assert "weaken" in proc.stderr.lower()


def test_combination_c_strengthening_with_correct_label_passes(tmp_path: Path) -> None:
    """(c) strengthening diff + policy-strengthen → exit 0."""
    direction, _ = _run_direction(tmp_path, _BASE_STRENGTHEN, _HEAD_STRENGTHEN)
    assert direction == "strengthen"

    proc = _run_assert(direction, ["policy-strengthen"], "Adds one redaction path.")
    assert proc.returncode == 0, f"expected exit 0, got {proc.returncode}; stderr={proc.stderr!r}"
    assert proc.stderr == ""


def test_combination_d_strengthening_with_weaken_label_fails(tmp_path: Path) -> None:
    """(d) strengthening diff + policy-weaken-justified → exit 1 with 'no coverage reduction'."""
    direction, _ = _run_direction(tmp_path, _BASE_STRENGTHEN, _HEAD_STRENGTHEN)
    assert direction == "strengthen"

    body = "## Redaction policy weakening rationale\nirrelevant"
    proc = _run_assert(direction, ["policy-weaken-justified"], body)
    assert proc.returncode == 1
    assert "no coverage reduction" in proc.stderr.lower()
    # Must name BOTH the wrong label and the correct one.
    assert "policy-weaken-justified" in proc.stderr
    assert "policy-strengthen" in proc.stderr


# Edge cases beyond the four canonical combinations -----------------------


def test_mixed_entry_changes_with_any_sensitive_count_drop_yield_weaken(tmp_path: Path) -> None:
    """A per-entry redaction drop must not be hidden by an unrelated increase."""
    base = {
        "tool_a": {"hash": "a1", "sensitive_path_count": 3, "shape": "type_driven"},
        "tool_b": {"hash": "b1", "sensitive_path_count": 0, "shape": "declarative"},
    }
    head = {
        "tool_a": {"hash": "a2", "sensitive_path_count": 1, "shape": "type_driven"},
        "tool_b": {"hash": "b2", "sensitive_path_count": 2, "shape": "declarative"},
    }

    direction, _ = _run_direction(tmp_path, base, head)

    assert direction == "weaken"


def test_same_count_sensitive_entry_replacement_yields_weaken(tmp_path: Path) -> None:
    """A changed existing sensitive entry with equal count needs weakening review.

    The snapshot does not carry stable sensitive-path identities, so a hash
    change at the same count can be a path/key replacement that removes coverage
    for the original field.
    """
    base = {"tool_a": {"hash": "api-key-policy", "sensitive_path_count": 1, "shape": "declarative"}}
    head = {"tool_a": {"hash": "token-policy", "sensitive_path_count": 1, "shape": "declarative"}}

    direction, _ = _run_direction(tmp_path, base, head)

    assert direction == "weaken"


def test_weaken_label_without_rationale_section_fails(tmp_path: Path) -> None:
    """A weakening PR carrying the correct label but missing the rationale
    section must fail — the rationale is the substantive control."""
    direction, _ = _run_direction(tmp_path, _BASE_WEAKEN, _HEAD_WEAKEN)
    assert direction == "weaken"

    proc = _run_assert(direction, ["policy-weaken-justified"], "no rationale section")
    assert proc.returncode == 1
    assert "Redaction policy weakening rationale" in proc.stderr


def test_unchanged_snapshot_yields_strengthen_with_empty_changed_set(
    tmp_path: Path,
) -> None:
    """If somehow both files are identical (no changed entries), the
    script reports ``strengthen`` (the default-when-equal branch); the
    workflow does not invoke the assertion step in this case because the
    bash ``git diff --quiet`` guard short-circuits — but the script must
    still emit a valid direction for unit-test isolation."""
    direction, _ = _run_direction(tmp_path, _BASE_WEAKEN, _BASE_WEAKEN)
    assert direction == "strengthen"


@pytest.mark.parametrize(
    "labels",
    [
        [],
        ["unrelated-label"],
        ["policy-weaken-justified", "policy-strengthen"],  # both labels present
    ],
)
def test_strengthening_requires_strengthen_label(tmp_path: Path, labels: list[str]) -> None:
    """Various wrong-label states for a strengthening PR all fail."""
    direction, _ = _run_direction(tmp_path, _BASE_STRENGTHEN, _HEAD_STRENGTHEN)
    assert direction == "strengthen"
    proc = _run_assert(direction, labels, "body")
    assert proc.returncode == 1
