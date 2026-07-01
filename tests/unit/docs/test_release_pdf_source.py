"""Release PDF source reproducibility guards."""

from __future__ import annotations

import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
PDF_DIR = REPO_ROOT / "docs" / "release" / "pdf"


def _git_check_ignore_pattern(path: str) -> str:
    result = subprocess.run(
        ["git", "check-ignore", "--no-index", "-v", path],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    metadata = result.stdout.strip().split("\t", maxsplit=1)[0]
    return metadata.split(":", maxsplit=2)[2]


def test_release_pdf_make_dry_run_resolves_source_dependencies() -> None:
    """The incremental PDF build must resolve source dependencies from checkout."""
    result = subprocess.run(
        ["make", "-n", "-B", "-C", str(PDF_DIR), "all"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr


def test_release_pdf_tokens_source_is_not_gitignored() -> None:
    """tokens.typ is source, not generated output, so .gitignore must not hide it."""
    assert _git_check_ignore_pattern("docs/release/pdf/tokens.typ") == "!docs/release/pdf/*.typ"


def test_release_pdf_generated_outputs_remain_gitignored() -> None:
    """Unignoring PDF source must not expose generated PDFs to git status."""
    for path in ("docs/release/pdf/out/example.pdf", "docs/release/pdf/output/example.pdf"):
        assert not _git_check_ignore_pattern(path).startswith("!")
