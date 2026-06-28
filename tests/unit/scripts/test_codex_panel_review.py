"""Unit tests for the codex_panel_review foundation (no codex calls)."""

from __future__ import annotations

import asyncio
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
    # the ROOT object also has additionalProperties:false, so it carries the same
    # strict-mode contract — dropping `findings` from required (or adding a property
    # not in required) keeps the rest of this test green yet makes codex reject the
    # schema with HTTP 400, failing every pilot lens silently (gated behind Task 9).
    assert set(schema["required"]) == set(schema["properties"])
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


def test_run_file_lenses_loops_gates_and_aggregates(tmp_path, monkeypatch):
    import json

    try:
        from codex_audit_common import _structured_findings, structured_output_path_for_report
    except ModuleNotFoundError:
        from scripts.codex_audit_common import _structured_findings, structured_output_path_for_report

    # run_file_lenses reads the target's source, so z.py must exist on disk.
    src_dir = tmp_path / "src"
    src_dir.mkdir()
    (src_dir / "z.py").write_text("def z(): ...", encoding="utf-8")

    calls = []

    async def fake_run_codex_once(**kwargs):
        # Simulate codex: write the structured sidecar then the .md (engine order).
        calls.append(kwargs["output_path"])
        out = kwargs["output_path"]
        sidecar = structured_output_path_for_report(out)
        sidecar.write_text(
            json.dumps(
                {
                    "markdown_report": "n",
                    "findings": [
                        {
                            "priority": "P1",
                            "lens": "x",
                            "category": "security",
                            "summary": "s",
                            # lineless on purpose: an anchor-required `security`
                            # finding with no int line MUST be downgraded by the gate,
                            # giving the integrated test an observable gate effect.
                            "evidence": [{"path": "src/z.py", "claim": "c"}],
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        out.write_text("narration\n", encoding="utf-8")
        return {"input_tokens": 100, "cached_input_tokens": 60, "output_tokens": 10, "total_tokens": 110}

    monkeypatch.setattr(cpr, "run_codex_once", fake_run_codex_once)

    stats = asyncio.run(
        cpr.run_file_lenses(
            file_path=tmp_path / "src" / "z.py",
            lenses=["security-architect", "solution-architect"],
            output_dir=tmp_path / "out",
            repo_root=tmp_path,
            context="CTX",
            model=None,
            reasoning_effort=None,
            rate_limiter=None,
            log_path=tmp_path / "log.md",
            log_lock=asyncio.Lock(),
        )
    )
    assert len(calls) == 2  # one call per lens, serial
    assert stats["cached_input_tokens"] == 120  # 60 * 2 aggregated
    assert stats["input_tokens"] == 200 and stats["output_tokens"] == 20  # full usage aggregated
    assert stats["failed"] == 0
    # Gate observability through the INTEGRATED runner path (not just the isolated
    # gate test): the lineless `security` finding must be downgraded once per lens.
    assert stats["gated"] == 2
    # Re-reading each per-lens sidecar via _structured_findings proves the gate (a)
    # rewrote the sidecar LAST so it is not stale against the .md, and (b) stamped
    # the caller's lens over the model's "x". Drop the gate call and the sidecar
    # stays stale -> _structured_findings returns None -> this block fails.
    for out_path, expected_lens in zip(calls, ["security-architect", "solution-architect"], strict=True):
        findings = _structured_findings(out_path)
        assert findings is not None, "gate did not rewrite the sidecar fresh"
        assert findings[0]["lens"] == expected_lens
        assert findings[0]["priority"] == "P3"


def test_run_file_lenses_captures_and_continues_on_lens_failure(tmp_path, monkeypatch):
    # The pilot's resilience contract: a lens whose codex call raises is counted in
    # `failed`, does NOT re-raise, and the loop continues to the next lens.
    import json

    try:
        from codex_audit_common import structured_output_path_for_report
    except ModuleNotFoundError:
        from scripts.codex_audit_common import structured_output_path_for_report

    calls = []

    async def fake_run_codex_once(**kwargs):
        calls.append(kwargs["output_path"])
        if len(calls) == 1:
            raise RuntimeError("codex exec failed (e.g. schema rejected)")
        out = kwargs["output_path"]
        # success path: write a (gate-valid, empty) sidecar then the .md, so the
        # post-call gate succeeds and only the FIRST lens is counted failed.
        structured_output_path_for_report(out).write_text(json.dumps({"markdown_report": "n", "findings": []}), encoding="utf-8")
        out.write_text("narration\n", encoding="utf-8")
        return {"input_tokens": 100, "cached_input_tokens": 60, "output_tokens": 10, "total_tokens": 110}

    monkeypatch.setattr(cpr, "run_codex_once", fake_run_codex_once)

    src_dir = tmp_path / "src"
    src_dir.mkdir()
    (src_dir / "z.py").write_text("def z(): ...", encoding="utf-8")

    stats = asyncio.run(
        cpr.run_file_lenses(
            file_path=tmp_path / "src" / "z.py",
            lenses=["security-architect", "solution-architect"],
            output_dir=tmp_path / "out",
            repo_root=tmp_path,
            context="CTX",
            model=None,
            reasoning_effort=None,
            rate_limiter=None,
            log_path=tmp_path / "log.md",
            log_lock=asyncio.Lock(),
        )
    )
    assert len(calls) == 2  # loop continued past the failing lens
    assert stats["failed"] == 1  # the raising lens was captured, not propagated
    assert stats["cached_input_tokens"] == 60  # only the surviving lens aggregated


def test_main_non_dry_run_exit_code_follows_failed(tmp_path, monkeypatch, capsys):
    target = tmp_path / "src" / "x.py"
    target.parent.mkdir(parents=True)
    target.write_text("def x(): ...", encoding="utf-8")
    monkeypatch.setattr(cpr, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(cpr.shutil, "which", lambda _name: "/usr/bin/codex")
    monkeypatch.setattr(cpr, "ensure_log_file", lambda *a, **k: None)
    monkeypatch.setattr(cpr, "load_context", lambda *a, **k: "CTX")
    monkeypatch.setattr(cpr, "make_codex_rate_limiter", lambda *a, **k: None)

    def fake_stats(failed):
        async def _run(**kwargs):
            return {
                "input_tokens": 100,
                "cached_input_tokens": 50,
                "output_tokens": 10,
                "total_tokens": 110,
                "gated": 0,
                "failed": failed,
            }

        return _run

    # fail-closed: a failed lens makes main exit 1
    monkeypatch.setattr(cpr, "run_file_lenses", fake_stats(1))
    assert cpr.main(["--file", str(target)]) == 1
    out = capsys.readouterr().out
    assert "cache_hit=50.0%" in out and "failed=1" in out

    # clean run exits 0
    monkeypatch.setattr(cpr, "run_file_lenses", fake_stats(0))
    assert cpr.main(["--file", str(target)]) == 0


def test_main_rejects_missing_and_out_of_repo_file(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(cpr, "REPO_ROOT", tmp_path)
    # nonexistent --file -> 1
    assert cpr.main(["--file", str(tmp_path / "nope.py")]) == 1
    # existing file OUTSIDE the repo -> clean 1, not an uncaught ValueError
    outside = tmp_path.parent / "outside.py"
    outside.write_text("x = 1\n", encoding="utf-8")
    try:
        assert cpr.main(["--file", str(outside)]) == 1
        assert "outside the repo" in capsys.readouterr().err
    finally:
        outside.unlink()


def test_main_rejects_unknown_lens_before_spend(tmp_path, monkeypatch, capsys):
    target = tmp_path / "src" / "x.py"
    target.parent.mkdir(parents=True)
    target.write_text("def x(): ...", encoding="utf-8")
    monkeypatch.setattr(cpr, "REPO_ROOT", tmp_path)
    # a --lenses typo is rejected up front (return 1) — no codex call is attempted.
    assert cpr.main(["--file", str(target), "--lenses", "no-such-lens"]) == 1
    assert "unknown lens" in capsys.readouterr().err


def test_main_rejects_context_file_traversal(tmp_path, monkeypatch, capsys):
    target = tmp_path / "src" / "x.py"
    target.parent.mkdir(parents=True)
    target.write_text("def x(): ...", encoding="utf-8")
    monkeypatch.setattr(cpr, "REPO_ROOT", tmp_path)
    # a --context-files entry that traverses outside the repo is rejected before any
    # codex call, so it can never be inlined into the egressed prompt.
    assert cpr.main(["--file", str(target), "--context-files", "../../../etc/passwd"]) == 1
    assert "context-files" in capsys.readouterr().err


def test_dry_run_prints_lens_plan(tmp_path, capsys, monkeypatch):
    target = tmp_path / "src" / "elspeth" / "web" / "foo.py"
    target.parent.mkdir(parents=True)
    target.write_text("def foo(): ...", encoding="utf-8")
    monkeypatch.setattr(cpr, "REPO_ROOT", tmp_path)
    rc = cpr.main(["--file", str(target), "--dry-run"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "security-architect" in out and "solution-architect" in out
    assert "foo.py" in out
