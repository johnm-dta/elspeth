"""Release PDF distribution label guard."""

from __future__ import annotations

import re
from pathlib import Path

PDF_DIR = Path("docs/release/pdf")
RELEASE_INDEX = Path("docs/release/README.md")
PDF_README = Path("docs/release/pdf/README.md")
INTERNAL_ONLY_RE = re.compile(r"\binternal\b.*\bonly\b", re.IGNORECASE)
DISTRIBUTION_RE = re.compile(r'distribution:\s*"([^"]+)"')


def test_non_draft_release_pdfs_do_not_use_internal_only_distribution_labels() -> None:
    """Public/evaluator release PDFs must not self-label as internal-only."""
    release_text = RELEASE_INDEX.read_text(encoding="utf-8")
    pdf_readme_text = PDF_README.read_text(encoding="utf-8")
    assert "public-sector evaluation material" in release_text
    assert "Audience:" in pdf_readme_text

    offenders: list[str] = []
    for source in sorted(PDF_DIR.glob("*.typ")):
        text = source.read_text(encoding="utf-8")
        if "#cover-page(" not in text or "draft: true" in text:
            continue
        match = DISTRIBUTION_RE.search(text)
        if match is not None and INTERNAL_ONLY_RE.search(match.group(1)):
            offenders.append(f"{source.as_posix()}: {match.group(1)}")

    assert offenders == []
