"""Regression checks for dated docs archive replacement manifests."""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
MAY_2026_MANIFEST = REPO_ROOT / "docs-archive" / "2026-05-19-docs-cleanout" / "MANIFEST.md"
ACTIVE_GUARANTEES = "docs/release/guarantees.md"
MISSING_ASSURANCE_CONTRACT = "docs/contracts/assurance-contract.md"


def _manifest_row_for_archived_group(manifest_text: str, archived_group: str) -> str:
    for line in manifest_text.splitlines():
        if line.startswith("|") and f"`{archived_group}`" in line:
            return line
    raise AssertionError(f"missing archive manifest row for {archived_group}")


def test_may_2026_archive_manifest_keeps_release_guarantees_active() -> None:
    text = MAY_2026_MANIFEST.read_text(encoding="utf-8")
    row = _manifest_row_for_archived_group(text, ACTIVE_GUARANTEES)

    assert MISSING_ASSURANCE_CONTRACT not in row
    assert f"`{ACTIVE_GUARANTEES}`" in row
    assert (REPO_ROOT / ACTIVE_GUARANTEES).exists()
