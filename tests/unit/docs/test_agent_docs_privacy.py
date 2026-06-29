"""Regression guard for sensitive operator attributes in tracked docs."""

from __future__ import annotations

import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
SENSITIVE_OPERATOR_DISCLOSURE_TERMS = ("neuro" + "diverse",)


def _tracked_markdown_files() -> list[Path]:
    proc = subprocess.run(
        ["git", "ls-files", "*.md"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return [REPO_ROOT / line for line in proc.stdout.splitlines()]


def test_tracked_markdown_docs_do_not_disclose_sensitive_operator_attributes() -> None:
    offenders: list[str] = []
    for path in _tracked_markdown_files():
        text = path.read_text(encoding="utf-8")
        lower_text = text.lower()
        for term in SENSITIVE_OPERATOR_DISCLOSURE_TERMS:
            if term in lower_text:
                offenders.append(str(path.relative_to(REPO_ROOT)))

    assert offenders == []
