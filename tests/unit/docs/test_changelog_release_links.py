"""Regression checks for root changelog release reference links."""

from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
CHANGELOG = REPO_ROOT / "CHANGELOG.md"
LINKED_RELEASE_HEADING_RE = re.compile(r"^## \[([^\]]+)\]", re.MULTILINE)
REFERENCE_DEFINITION_RE = re.compile(r"^\[([^\]]+)\]:", re.MULTILINE)


def test_linked_changelog_release_headings_have_reference_definitions() -> None:
    text = CHANGELOG.read_text(encoding="utf-8")

    linked_release_headings = set(LINKED_RELEASE_HEADING_RE.findall(text))
    reference_definitions = set(REFERENCE_DEFINITION_RE.findall(text))

    assert sorted(linked_release_headings - reference_definitions) == []
