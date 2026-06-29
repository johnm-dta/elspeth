"""Regression checks for resume recovery documentation."""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
README = REPO_ROOT / "README.md"
INCIDENT_RESPONSE = REPO_ROOT / "docs" / "runbooks" / "incident-response.md"
DOCKER_GUIDE = REPO_ROOT / "docs" / "guides" / "docker.md"


def test_resume_recovery_examples_use_execute_flag() -> None:
    readme = README.read_text(encoding="utf-8")
    incident_response = INCIDENT_RESPONSE.read_text(encoding="utf-8")
    docker_guide = DOCKER_GUIDE.read_text(encoding="utf-8")

    assert "elspeth resume <run_id> --execute" in readme
    assert "elspeth resume abc123 --execute" in readme
    assert "elspeth resume <RUN_ID> --execute" in incident_response
    assert "resume abc123 --execute" in docker_guide


def test_readme_mentions_resume_default_is_dry_run() -> None:
    text = README.read_text(encoding="utf-8")
    normalized_text = " ".join(text.split())

    assert "Without `--execute`" in text
    assert "checks whether the run can be resumed" in normalized_text
