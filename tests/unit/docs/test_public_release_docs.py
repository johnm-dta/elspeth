"""Regression checks for public-facing release documents."""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
PUBLIC_RELEASE_DOCS = (
    REPO_ROOT / "docs" / "release" / "README.md",
    REPO_ROOT / "docs" / "release" / "executive-summary.md",
    REPO_ROOT / "docs" / "release" / "composer-guide.md",
    REPO_ROOT / "docs" / "release" / "platform-architecture.md",
    REPO_ROOT / "docs" / "release" / "assessment-mapping.md",
    REPO_ROOT / "docs" / "release" / "guarantees.md",
)
INTERNAL_TRACKER_TERMS = ("Filigree", "filigree", "session-context")


def test_public_release_docs_do_not_route_readers_to_internal_tracker() -> None:
    offenders: list[str] = []

    for path in PUBLIC_RELEASE_DOCS:
        text = path.read_text(encoding="utf-8")
        rel_path = path.relative_to(REPO_ROOT).as_posix()
        for term in INTERNAL_TRACKER_TERMS:
            if term in text:
                offenders.append(f"{rel_path}: {term}")

    assert offenders == []
