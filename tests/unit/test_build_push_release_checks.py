"""Build-push workflow release proof invariants."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path
from typing import Any

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
BUILD_PUSH_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "build-push.yaml"
DOCKERFILE = REPO_ROOT / "Dockerfile"
DOCKERIGNORE = REPO_ROOT / ".dockerignore"


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


def test_release_dockerfile_builds_frontend_dist_before_python_install() -> None:
    """Release image must build the SPA instead of relying on ignored host dist."""
    dockerfile = DOCKERFILE.read_text(encoding="utf-8")

    assert "AS frontend-builder" in dockerfile
    assert "npm ci" in dockerfile
    assert "npm run build" in dockerfile
    assert "COPY --from=frontend-builder /frontend/dist /tmp/frontend-dist/" in dockerfile
    assert dockerfile.index("npm run build") < dockerfile.index('uv sync --frozen "$@" --no-editable --active')


def test_release_build_context_excludes_host_node_modules() -> None:
    """Host-installed frontend dependencies must not enter the Docker context."""
    raw_lines = DOCKERIGNORE.read_text(encoding="utf-8").splitlines()
    dockerignore_patterns = {line.strip() for line in raw_lines if line.strip() and not line.lstrip().startswith("#")}

    assert "**/node_modules/" in dockerignore_patterns


def test_release_build_context_excludes_frontend_unit_tests() -> None:
    """Production SPA compilation must not depend on test-only source fixtures."""
    raw_lines = DOCKERIGNORE.read_text(encoding="utf-8").splitlines()
    dockerignore_patterns = {line.strip() for line in raw_lines if line.strip() and not line.lstrip().startswith("#")}

    assert "src/elspeth/web/frontend/src/**/*.test.ts" in dockerignore_patterns
    assert "src/elspeth/web/frontend/src/**/*.test.tsx" in dockerignore_patterns


def test_release_dockerfile_copies_local_uv_sources_before_dependency_sync() -> None:
    """Root pyproject local uv sources must exist before Docker runs uv sync."""
    dockerfile = DOCKERFILE.read_text(encoding="utf-8")

    assert "COPY elspeth-lints/ ./elspeth-lints/" in dockerfile
    assert dockerfile.index("COPY elspeth-lints/ ./elspeth-lints/") < dockerfile.index(
        'uv sync --frozen "$@" --no-install-project --active'
    )


def test_release_dockerfile_copies_frontend_dist_into_installed_package() -> None:
    """The non-editable Docker install must carry generated SPA assets into site-packages."""
    dockerfile = DOCKERFILE.read_text(encoding="utf-8")

    assert "COPY --from=frontend-builder /frontend/dist /tmp/frontend-dist/" in dockerfile
    assert "import elspeth.web" in dockerfile
    assert 'shutil.copytree("/tmp/frontend-dist", target)' in dockerfile
    assert dockerfile.index('uv sync --frozen "$@" --no-editable --active') < dockerfile.index(
        'shutil.copytree("/tmp/frontend-dist", target)'
    )


def _extras_validation_scripts() -> list[str]:
    dockerfile = DOCKERFILE.read_text(encoding="utf-8")
    blocks = re.findall(
        r'test -n "\$INSTALL_EXTRAS" && \\\n.*?uv sync --frozen "\$@" --no-(?:install-project|editable) --active',
        dockerfile,
        flags=re.DOTALL,
    )
    assert len(blocks) == 2, "both dependency sync layers must use the same extras validator"
    return [
        re.sub(
            r'uv sync --frozen "\$@" --no-(?:install-project|editable) --active',
            lambda _match: "printf '%s\\n' \"$@\"",
            block,
        )
        for block in blocks
    ]


def _run_extras_validator(script: str, extras: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["/bin/sh", "-c", script],
        env={"INSTALL_EXTRAS": extras},
        capture_output=True,
        text=True,
        check=False,
    )


def test_release_dockerfile_defaults_to_all_extras_and_validates_both_sync_layers() -> None:
    dockerfile = DOCKERFILE.read_text(encoding="utf-8")

    assert dockerfile.count('ARG INSTALL_EXTRAS="all"') == 1
    assert dockerfile.count('test -n "$INSTALL_EXTRAS"') == 2
    assert dockerfile.count('case "$e" in [a-z0-9]*) ;; *) exit 2 ;; esac;') == 2
    assert dockerfile.count('case "$e" in *[!a-z0-9-]*) exit 2 ;; esac;') == 2
    assert dockerfile.count('set -- "$@" --extra "$e"') == 2
    assert dockerfile.count('test "$#" -gt 0') == 2


@pytest.mark.parametrize("extras", ["", "   ", "--no-dev", "ALL", "webui*", "webui;llm"])
def test_release_dockerfile_rejects_invalid_install_extras_in_both_sync_layers(extras: str) -> None:
    for script in _extras_validation_scripts():
        result = _run_extras_validator(script, extras)
        assert result.returncode != 0, (extras, result.stdout, result.stderr)


@pytest.mark.parametrize(
    ("extras", "expected"),
    [
        ("all", ["--extra", "all"]),
        (
            "webui llm aws postgres",
            ["--extra", "webui", "--extra", "llm", "--extra", "aws", "--extra", "postgres"],
        ),
    ],
)
def test_release_dockerfile_expands_valid_install_extras_in_both_sync_layers(
    extras: str,
    expected: list[str],
) -> None:
    for script in _extras_validation_scripts():
        result = _run_extras_validator(script, extras)
        assert result.returncode == 0, (extras, result.stdout, result.stderr)
        assert result.stdout.splitlines() == expected


def test_release_dockerfile_documents_orchestrator_owned_probe_wiring() -> None:
    dockerfile = DOCKERFILE.read_text(encoding="utf-8")

    assert "Web task definitions: loopback GET /api/health" in dockerfile
    assert "ALB target groups:     GET /api/ready" in dockerfile
    assert "Batch tasks:           process exit code" in dockerfile
    assert "elspeth health --port 8451" not in dockerfile
