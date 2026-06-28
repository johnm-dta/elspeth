"""Unit tests for the codex_panel_review foundation (no codex calls)."""

from __future__ import annotations

from pathlib import Path

from scripts import codex_panel_review as cpr


def test_module_constants_present_and_typed():
    assert isinstance(cpr.PANEL_FINDING_SCHEMA, Path)
    assert cpr.PANEL_FINDING_SCHEMA.name == "panel_finding.schema.json"
    assert isinstance(cpr.LENSES_DIR, Path)
    # priority-bearing categories that must carry a file:line anchor
    assert frozenset({"bug", "correctness", "security", "smell"}) == cpr.STRICT_CATEGORIES
    assert frozenset({"improvement", "efficiency"}) == cpr.RELAXED_CATEGORIES
    assert cpr.STRICT_CATEGORIES | frozenset({"easy-win"}) == cpr.ANCHOR_REQUIRED_CATEGORIES
