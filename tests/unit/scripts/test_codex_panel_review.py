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


def test_layered_prompt_orders_stable_content_first():
    prompt = cpr.build_layered_prompt(
        context="CONTEXT_MARKER project rules",
        file_source="SOURCE_MARKER def f(): ...",
        file_path="src/elspeth/web/foo.py",
        persona="PERSONA_MARKER act as a security architect",
        lens="security-architect",
    )
    i_ctx = prompt.index("CONTEXT_MARKER")
    i_src = prompt.index("SOURCE_MARKER")
    i_persona = prompt.index("PERSONA_MARKER")
    i_path = prompt.index("src/elspeth/web/foo.py")
    # stable (cacheable) content first, variable per-call content last
    assert i_ctx < i_src < i_persona < i_path
    # the focus path must NOT appear in the cacheable head (before persona)
    assert prompt[:i_persona].count("src/elspeth/web/foo.py") == 0


def test_load_persona_reads_and_errors(tmp_path):
    (tmp_path / "demo.md").write_text("PERSONA BODY", encoding="utf-8")
    assert cpr.load_persona("demo", lenses_dir=tmp_path) == "PERSONA BODY"
    import pytest

    with pytest.raises(FileNotFoundError):
        cpr.load_persona("missing", lenses_dir=tmp_path)


def test_route_lenses_default_and_override():
    py = Path("src/elspeth/web/foo.py")
    routed = cpr.route_lenses(py)
    assert "solution-architect" in routed and "security-architect" in routed
    # explicit override is returned verbatim
    assert cpr.route_lenses(py, override=["security-architect"]) == ["security-architect"]
