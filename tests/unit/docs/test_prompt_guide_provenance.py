"""Regression checks for transcript citation placeholder hygiene."""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
PLACEHOLDERS = ("filecite", "cite", "turn0file", "turn7view")


def test_public_docs_have_no_transcript_citation_placeholders() -> None:
    offenders: list[str] = []

    for path in [*REPO_ROOT.glob("*.md"), *REPO_ROOT.glob("docs/**/*.md")]:
        text = path.read_text(encoding="utf-8")
        for placeholder in PLACEHOLDERS:
            if placeholder in text:
                offenders.append(f"{path.relative_to(REPO_ROOT).as_posix()}: {placeholder}")

    assert offenders == []
