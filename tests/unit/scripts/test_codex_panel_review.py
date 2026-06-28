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


def test_panel_schema_shape():
    import json

    schema = json.loads(cpr.PANEL_FINDING_SCHEMA.read_text(encoding="utf-8"))
    assert schema["type"] == "object"
    # markdown_report is mandatory: run_codex_once extracts it (common.py:365)
    assert "markdown_report" in schema["required"]
    assert schema["properties"]["markdown_report"]["type"] == "string"
    finding = schema["properties"]["findings"]["items"]
    assert finding["properties"]["priority"]["enum"] == ["P0", "P1", "P2", "P3"]
    # priority, NOT severity, carries the P-level
    assert "severity" not in finding["properties"]
    for req in ("priority", "lens", "category", "summary", "evidence"):
        assert req in finding["required"]
    # strict structured-outputs contract: under additionalProperties:false the API
    # rejects any optional property, so EVERY declared property must be required
    # (genuinely-optional fields are nullable union types, never absent from required).
    assert set(finding["required"]) == set(finding["properties"])
    evidence_item = finding["properties"]["evidence"]["items"]
    assert set(evidence_item["required"]) == set(evidence_item["properties"])
