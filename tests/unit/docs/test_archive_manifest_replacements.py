"""Regression checks for maintainer-local docs archive references."""

from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
SPECIFIC_ARCHIVE_PATH = re.compile(r"docs-archive/\d{4}-\d{2}-\d{2}[^)\s`]*")


def _public_text_paths() -> list[Path]:
    paths = [
        *REPO_ROOT.glob("*.md"),
        *REPO_ROOT.glob("docs/**/*.md"),
        *REPO_ROOT.glob("config/cicd/**/*.md"),
        *REPO_ROOT.glob("elspeth-lints/**/*.md"),
    ]
    return sorted(path for path in paths if path.is_file())


def test_public_docs_do_not_link_specific_local_archive_paths() -> None:
    offenders: list[str] = []

    for path in _public_text_paths():
        text = path.read_text(encoding="utf-8")
        if SPECIFIC_ARCHIVE_PATH.search(text):
            offenders.append(path.relative_to(REPO_ROOT).as_posix())

    assert offenders == []
