"""Regression checks for redaction policy operator guidance."""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
GUIDE = REPO_ROOT / "docs" / "guides" / "redaction-policy-changes.md"
BOOTSTRAP_COMMAND = ".venv/bin/python scripts/cicd/bootstrap_redaction_snapshot.py"


def test_redaction_snapshot_regeneration_guide_includes_write_command() -> None:
    text = GUIDE.read_text(encoding="utf-8")

    assert f"{BOOTSTRAP_COMMAND}\n" in text
    assert f"{BOOTSTRAP_COMMAND} --write" in text
    assert "dry-run" in text.lower()
