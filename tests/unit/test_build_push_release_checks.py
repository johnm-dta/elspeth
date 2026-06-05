"""Build-push workflow release proof invariants."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
BUILD_PUSH_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "build-push.yaml"


def _workflow() -> dict[str, Any]:
    raw = yaml.safe_load(BUILD_PUSH_WORKFLOW.read_text(encoding="utf-8"))
    assert isinstance(raw, dict)
    return raw


def _build_push_job() -> dict[str, Any]:
    workflow = _workflow()
    job = workflow["jobs"]["build-push"]
    assert isinstance(job, dict)
    return job


def _step_run(job: dict[str, Any], step_name: str) -> str:
    for step in job["steps"]:
        if step.get("name") == step_name:
            run = step.get("run")
            assert isinstance(run, str)
            return run
    raise AssertionError(f"Missing build-push step {step_name!r}")


def test_build_push_verifies_ruleset_required_checks_for_image_sha() -> None:
    """Release image publication must mirror live branch-protection contexts."""
    job = _build_push_job()

    run = _step_run(job, "Verify required checks for image commit")

    assert "scripts/cicd/check_release_required_checks.py" in run
    assert '--sha "$IMAGE_SHA"' in run
    assert '--repo "$GITHUB_REPOSITORY"' in run
    assert "check_name=CI%20Success" not in run
    assert "Workflow run trigger already supplied successful CI conclusion" not in run


def test_build_push_grants_read_permissions_for_required_check_verifier() -> None:
    """The verifier must be able to read rulesets, check runs, and statuses."""
    job = _build_push_job()

    assert job["permissions"]["actions"] == "read"
    assert job["permissions"]["checks"] == "read"
    assert job["permissions"]["statuses"] == "read"
