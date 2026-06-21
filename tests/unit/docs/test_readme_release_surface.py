"""Regression checks for the public README release surface."""

from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
README = REPO_ROOT / "README.md"


def _readme_text() -> str:
    return README.read_text(encoding="utf-8")


def test_readme_advertises_current_release_surface() -> None:
    text = _readme_text()

    # Current release line is shown, not a stale RC badge.
    assert "Status: 0.6.0" in text
    assert "status-0.6.0" in text
    assert "## What Changed In 0.6.0" in text

    # Key evaluator-facing release references remain.
    assert "[Audit and Lineage Guarantees](docs/release/guarantees.md)" in text
    assert "[docs/release/](docs/release/)" in text


def test_readme_release_links_resolve() -> None:
    text = _readme_text()
    linked_paths = set(re.findall(r"\]\((docs/release/[^)#]+)", text))

    assert linked_paths, "README should reference at least one docs/release/ document"
    for relative_path in linked_paths:
        assert (REPO_ROOT / relative_path).exists(), relative_path
