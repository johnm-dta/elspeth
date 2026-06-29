"""Regression checks for Docker guide release examples."""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
DOCKER_GUIDE = REPO_ROOT / "docs" / "guides" / "docker.md"
STALE_IMAGE_TAG = "elspeth:v0.1.0"


def test_docker_guide_uses_release_tag_variable_for_image_examples() -> None:
    text = DOCKER_GUIDE.read_text(encoding="utf-8")

    assert STALE_IMAGE_TAG not in text
    assert "IMAGE_TAG=" in text
    assert "ghcr.io/johnm-dta/elspeth:${IMAGE_TAG}" in text
    assert "your-acr.azurecr.io/elspeth:${IMAGE_TAG}" in text


def test_docker_guide_links_to_active_user_manual() -> None:
    text = DOCKER_GUIDE.read_text(encoding="utf-8")

    assert "../USER_MANUAL.md" not in text
    assert "(user-manual.md#cli-commands)" in text
    assert (DOCKER_GUIDE.parent / "user-manual.md").exists()
