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


def _job(name: str) -> dict[str, Any]:
    workflow = _workflow()
    job = workflow["jobs"][name]
    assert isinstance(job, dict)
    return job


def _step(job: dict[str, Any], step_name: str) -> dict[str, Any]:
    for step in job["steps"]:
        if step.get("name") == step_name:
            assert isinstance(step, dict)
            return step
    raise AssertionError(f"Missing step {step_name!r}")


def _step_run(job: dict[str, Any], step_name: str) -> str:
    run = _step(job, step_name).get("run")
    assert isinstance(run, str)
    return run


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


# ---------------------------------------------------------------------------
# elspeth-118bf5ea8c / elspeth-8cb798c3fd:
# An ACR-only manual dispatch sets push_ghcr=false, so no GHCR image is pushed.
# The smoke-test job must NOT unconditionally pull the GHCR image — it must test
# whichever registry was actually pushed.
# ---------------------------------------------------------------------------


def test_build_push_exposes_registry_push_decisions_as_outputs() -> None:
    """Downstream jobs need to know which registries were actually pushed."""
    job = _build_push_job()
    outputs = job["outputs"]

    assert "steps.registries.outputs.push_ghcr" in outputs["push_ghcr"]
    assert "steps.registries.outputs.push_acr" in outputs["push_acr"]


def test_smoke_test_skipped_when_no_image_was_pushed() -> None:
    """If neither registry was pushed there is nothing to smoke-test."""
    job = _job("smoke-test")
    condition = job["if"]

    assert "needs.build-push.outputs.push_ghcr" in condition
    assert "needs.build-push.outputs.push_acr" in condition


def test_smoke_test_image_selection_is_registry_aware() -> None:
    """The smoke image must be chosen from the pushed registry, not hardcoded GHCR."""
    job = _job("smoke-test")
    run = _step_run(job, "Determine smoke-test image")

    # Picks GHCR when pushed, else the ACR image — written to the job env.
    assert "SMOKE_IMAGE=ghcr.io/" in run
    assert "SMOKE_IMAGE=${ACR_REGISTRY}/" in run
    assert "GITHUB_ENV" in run


def test_smoke_test_run_steps_use_the_selected_image() -> None:
    """No smoke run step may hardcode the GHCR image; all use $SMOKE_IMAGE."""
    job = _job("smoke-test")
    for step in job["steps"]:
        run = step.get("run")
        if not isinstance(run, str):
            continue
        assert 'IMAGE="ghcr.io/${REPO_OWNER}' not in run, f"hardcoded GHCR image in step {step.get('name')!r}"


def test_smoke_test_logs_into_the_pushed_registry() -> None:
    """GHCR login fires only for a GHCR smoke; an ACR login path exists too."""
    job = _job("smoke-test")
    ghcr_login = _step(job, "Login to GHCR (to pull image)")
    acr_login = _step(job, "Login to ACR (to pull image)")

    assert "ghcr" in ghcr_login["if"]
    assert "acr" in acr_login["if"]
    assert acr_login["with"]["registry"] == "${{ secrets.ACR_REGISTRY }}"
