"""Structural assertions for the composer redaction-gate workflow.

Closes rev-2 BLOCKER_B / rev-1 B4 / W9 / M10. The workflow itself lives
at ``.github/workflows/composer-redaction-gate.yml`` and its logic is
extracted into ``scripts/cicd/check_redaction_direction.py`` and
``scripts/cicd/assert_redaction_label.py``. These tests assert the
workflow has the required paths, the direction output step, separate
weaken/strengthen handling, and that it calls the extracted scripts.

We use ``yaml.safe_load`` to parse; we do not regex over raw text. The
"separate weaken/strengthen branches" requirement is satisfied by the
assert-label step calling
``scripts/cicd/assert_redaction_label.py`` which contains the two
direction-specific branches verifiable by the four-combination tests in
``test_label_gate_direction.py``.
"""

from __future__ import annotations

from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[4]
WORKFLOW_PATH = REPO_ROOT / ".github" / "workflows" / "composer-redaction-gate.yml"

REQUIRED_PATHS = frozenset(
    {
        "src/elspeth/web/composer/redaction.py",
        "src/elspeth/web/composer/redaction_telemetry.py",
        "src/elspeth/web/composer/tools.py",
        "tests/unit/web/composer/redaction_policy_snapshot.json",
        "tests/unit/web/composer/test_adequacy_guard.py",
        "tests/unit/web/composer/test_walk_model_schema.py",
    }
)


def _load_workflow() -> dict:
    raw = WORKFLOW_PATH.read_text(encoding="utf-8")
    parsed = yaml.safe_load(raw)
    assert isinstance(parsed, dict), "workflow YAML root must be a mapping"
    return parsed


def test_workflow_parses_as_yaml() -> None:
    """The workflow file must be valid YAML mapping."""
    workflow = _load_workflow()
    assert workflow["name"] == "composer-redaction-gate"


def test_workflow_triggers_on_required_paths() -> None:
    """The path-filter must cover every redaction-touching file."""
    workflow = _load_workflow()
    # PyYAML maps the ``on:`` key as the string "on", but in some
    # environments it is loaded as boolean True (YAML 1.1 quirk). Accept
    # whichever the parser produced.
    on_block = workflow.get("on") or workflow.get(True)
    assert on_block is not None, "workflow must declare an 'on:' trigger block"
    pull_request = on_block["pull_request"]
    paths = frozenset(pull_request.get("paths", []))
    if not paths:
        steps = workflow["jobs"]["redaction-gate"]["steps"]
        diff_steps = [s for s in steps if s.get("id") == "diff"]
        assert len(diff_steps) == 1, "exactly one step must carry id=diff"
        run_block = diff_steps[0]["run"]
        paths = frozenset(path for path in REQUIRED_PATHS if path in run_block)

    missing = REQUIRED_PATHS - paths
    assert not missing, f"workflow paths missing required entries: {sorted(missing)}"


def test_workflow_has_direction_output_step() -> None:
    """The first job step computing diff must expose a ``direction`` output."""
    workflow = _load_workflow()
    steps = workflow["jobs"]["redaction-gate"]["steps"]
    diff_steps = [s for s in steps if s.get("id") == "diff"]
    assert len(diff_steps) == 1, "exactly one step must carry id=diff"
    run_block = diff_steps[0]["run"]
    # Must write the direction= line either inline (none case) or via the
    # check script (changed case). Both paths must appear.
    assert "direction=none" in run_block, "no-change branch must write direction=none"
    assert "check_redaction_direction.py" in run_block, "changed branch must invoke check_redaction_direction.py"


def test_workflow_handles_first_snapshot_introduction() -> None:
    """A PR adding the snapshot for the first time must not fail on ``git show``.

    ``origin/main`` did not always contain the redaction snapshot. When a PR
    introduces the file, the diff step still needs a base-side empty object so
    the direction script can classify the change as strengthening.
    """
    workflow = _load_workflow()
    steps = workflow["jobs"]["redaction-gate"]["steps"]
    diff_steps = [s for s in steps if s.get("id") == "diff"]
    run_block = diff_steps[0]["run"]

    assert 'git cat-file -e "${BASE_REMOTE}:${SNAPSHOT_PATH}"' in run_block
    assert "printf '{}' > \"$BASE_TMP\"" in run_block


def test_workflow_calls_assert_redaction_label_script() -> None:
    """The label-assertion step must call the extracted script with direction."""
    workflow = _load_workflow()
    steps = workflow["jobs"]["redaction-gate"]["steps"]
    assert_steps = [s for s in steps if "assert_redaction_label.py" in str(s.get("run", ""))]
    assert len(assert_steps) == 1, "exactly one step must invoke assert_redaction_label.py"
    step = assert_steps[0]
    run_block = step["run"]
    assert "--direction" in run_block
    assert "--pr-labels-json" in run_block
    assert "--pr-body" in run_block
    # The step must be guarded so it only fires when the snapshot
    # changed (no-change PRs should not be label-gated).
    assert step.get("if") == "steps.diff.outputs.snapshot_changed == 'true'", "label-assertion step must gate on snapshot_changed == 'true'"
    # Env wiring: PR_LABELS, PR_BODY, DIRECTION must come from the
    # workflow context (not be hardcoded).
    env = step["env"]
    assert "PR_LABELS" in env and "toJson" in env["PR_LABELS"]
    assert "PR_BODY" in env
    assert env["DIRECTION"] == "${{ steps.diff.outputs.direction }}"


def test_workflow_separates_weaken_and_strengthen_branches() -> None:
    """The assertion logic must branch on direction; verify via the script.

    The workflow itself delegates the weaken vs strengthen branching to
    ``assert_redaction_label.py``, which contains explicit ``elif`` for
    the two directions. We assert here that:

    * ``check_redaction_direction.py`` (compute) emits both branch
      strings, and
    * ``assert_redaction_label.py`` (enforce) handles both branch
      strings.

    The four-combination tests in ``test_label_gate_direction.py``
    exercise the actual behaviour end-to-end.
    """
    scripts_dir = REPO_ROOT / "scripts" / "cicd"
    check_script = (scripts_dir / "check_redaction_direction.py").read_text("utf-8")
    assert_script = (scripts_dir / "assert_redaction_label.py").read_text("utf-8")

    # The compute script must be able to emit both directions.
    assert '"weaken"' in check_script
    assert '"strengthen"' in check_script

    # The assertion script must branch on both directions.
    assert 'direction == "weaken"' in assert_script
    assert 'direction == "strengthen"' in assert_script
