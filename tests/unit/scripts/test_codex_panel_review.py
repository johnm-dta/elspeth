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


def _write_sidecar(tmp_path, findings):
    import json

    md = tmp_path / "src__x.md"
    md.write_text("narration\n", encoding="utf-8")
    sidecar = tmp_path / "src__x.md.structured.json"
    sidecar.write_text(json.dumps({"markdown_report": "n", "findings": findings}), encoding="utf-8")
    return md, sidecar


def test_gate_strict_downgrades_without_line(tmp_path):
    import json

    _, sidecar = _write_sidecar(
        tmp_path,
        [
            {
                "priority": "P1",
                "lens": "security-architect",
                "category": "security",
                "summary": "s",
                "evidence": [{"path": "src/x.py", "claim": "no line"}],
            },
        ],
    )
    n = cpr.apply_panel_evidence_gate(sidecar, lens="security-architect")
    assert n == 1
    assert json.loads(sidecar.read_text())["findings"][0]["priority"] == "P3"


def test_gate_relaxed_keeps_lineless_improvement(tmp_path):
    import json

    _, sidecar = _write_sidecar(
        tmp_path,
        [
            {
                "priority": "P2",
                "lens": "solution-architect",
                "category": "improvement",
                "impact": "removes a whole retry class",
                "summary": "add retry abstraction",
                "evidence": [],
            },
        ],
    )
    n = cpr.apply_panel_evidence_gate(sidecar, lens="solution-architect")
    assert n == 0
    assert json.loads(sidecar.read_text())["findings"][0]["priority"] == "P2"


def test_gate_easy_win_requires_anchor(tmp_path):
    import json

    _, sidecar = _write_sidecar(
        tmp_path,
        [
            {
                "priority": "P2",
                "lens": "solution-architect",
                "category": "easy-win",
                "impact": "tiny",
                "summary": "rename for clarity",
                "evidence": [],
            },
        ],
    )
    assert cpr.apply_panel_evidence_gate(sidecar, lens="solution-architect") == 1
    assert json.loads(sidecar.read_text())["findings"][0]["priority"] == "P3"


def test_gate_design_without_impact_downgrades(tmp_path):
    # fail-closed: `design` is in no category set, so an unsubstantiated design
    # finding (no anchor, no impact) is downgraded rather than riding through.
    import json

    _, sidecar = _write_sidecar(
        tmp_path,
        [
            {
                "priority": "P0",
                "lens": "solution-architect",
                "category": "design",
                "summary": "module splits two responsibilities",
                "evidence": [],
            },
        ],
    )
    assert cpr.apply_panel_evidence_gate(sidecar, lens="solution-architect") == 1
    assert json.loads(sidecar.read_text())["findings"][0]["priority"] == "P3"


def test_gate_design_with_impact_kept(tmp_path):
    import json

    _, sidecar = _write_sidecar(
        tmp_path,
        [
            {
                "priority": "P1",
                "lens": "solution-architect",
                "category": "design",
                "impact": "two responsibilities inflate every change's blast radius",
                "summary": "split module",
                "evidence": [],
            },
        ],
    )
    assert cpr.apply_panel_evidence_gate(sidecar, lens="solution-architect") == 0
    assert json.loads(sidecar.read_text())["findings"][0]["priority"] == "P1"


def test_gate_stamps_lens_over_model_value(tmp_path):
    # the gate trusts the caller's lens, never the model-supplied field.
    import json

    _, sidecar = _write_sidecar(
        tmp_path,
        [
            {"priority": "P2", "lens": "WRONG-model-value", "category": "improvement", "impact": "real", "summary": "s", "evidence": []},
        ],
    )
    cpr.apply_panel_evidence_gate(sidecar, lens="solution-architect")
    assert json.loads(sidecar.read_text())["findings"][0]["lens"] == "solution-architect"


def test_gate_no_category_rides_through_ungated(tmp_path):
    # drift guard: with neither a file:line anchor nor an impact rationale, NO
    # category in the schema enum survives at its original priority — a future
    # enum value with no gate handling would fail here.
    import json

    schema = json.loads(cpr.PANEL_FINDING_SCHEMA.read_text(encoding="utf-8"))
    enum = schema["properties"]["findings"]["items"]["properties"]["category"]["enum"]
    for category in [c for c in enum if c is not None]:
        _, sidecar = _write_sidecar(
            tmp_path,
            [
                {"priority": "P0", "lens": "solution-architect", "category": category, "summary": "s", "evidence": []},
            ],
        )
        assert cpr.apply_panel_evidence_gate(sidecar, lens="solution-architect") == 1, category
        assert json.loads(sidecar.read_text())["findings"][0]["priority"] == "P3"


def test_gate_rewrites_sidecar_last_so_not_stale(tmp_path):
    # Reproduces the mtime trap: run_codex_once writes the .md AFTER the sidecar,
    # so the fresh sidecar looks stale until the gate rewrites it.
    import json
    import time

    try:
        from codex_audit_common import _structured_findings, structured_output_path_for_report
    except ModuleNotFoundError:
        from scripts.codex_audit_common import _structured_findings, structured_output_path_for_report

    md = tmp_path / "src__y.md"
    sidecar = structured_output_path_for_report(md)
    sidecar.write_text(
        json.dumps(
            {
                "markdown_report": "n",
                "findings": [
                    {
                        "priority": "P1",
                        "lens": "security-architect",
                        "category": "security",
                        "summary": "s",
                        "evidence": [{"path": "src/y.py", "line": 5, "claim": "c"}],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    time.sleep(0.02)
    md.write_text("narration\n", encoding="utf-8")  # .md now newer -> sidecar looks stale
    assert _structured_findings(md) is None  # trap confirmed
    cpr.apply_panel_evidence_gate(sidecar, lens="security-architect")  # rewrites sidecar last
    findings = _structured_findings(md)
    assert findings is not None and findings[0]["priority"] == "P1"
