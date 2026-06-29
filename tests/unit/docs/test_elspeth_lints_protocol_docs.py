"""Regression checks for elspeth-lints protocol documentation."""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
PROTOCOLS_DOC = REPO_ROOT / "docs" / "elspeth-lints" / "protocols.md"


def test_dump_edges_json_example_includes_required_output_path() -> None:
    text = PROTOCOLS_DOC.read_text(encoding="utf-8")

    assert "elspeth-lints dump-edges --root . --format json --output" in text
    assert "elspeth-lints dump-edges --root . --format json\n" not in text
