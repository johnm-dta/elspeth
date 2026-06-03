"""Golden tests for elspeth-lints emitters."""

from __future__ import annotations

import json
from pathlib import Path

from elspeth_lints.core.protocols import Category, Finding, RuleMetadata, RuleScope, Severity

FIXTURES = Path("tests/fixtures/elspeth_lints/emitters")


def test_text_emitter_matches_golden() -> None:
    from elspeth_lints.core.emitters.text import render_text

    assert render_text([_finding()]) == (FIXTURES / "text.txt").read_text(encoding="utf-8")


def test_github_emitter_matches_golden() -> None:
    from elspeth_lints.core.emitters.github import render_github

    assert render_github([_finding()]) == (FIXTURES / "github.txt").read_text(encoding="utf-8")


def test_sarif_emitter_matches_golden_and_has_required_shape() -> None:
    from elspeth_lints.core.emitters.sarif import render_sarif

    rendered = render_sarif([_finding()], metadata=[_metadata()])

    assert json.loads(rendered) == json.loads((FIXTURES / "sarif.json").read_text(encoding="utf-8"))
    payload = json.loads(rendered)
    assert payload["version"] == "2.1.0"
    assert payload["runs"][0]["tool"]["driver"]["rules"][0]["id"] == "demo.rule"
    assert payload["runs"][0]["results"][0]["locations"][0]["physicalLocation"]["region"]["startColumn"] == 3


def _finding() -> Finding:
    return Finding(
        rule_id="demo.rule",
        file_path="src/example.py",
        line=7,
        column=2,
        message="Demo finding",
        fingerprint="demo-fingerprint",
        severity=Severity.ERROR,
        suggestion="Fix the demo",
    )


def _metadata() -> RuleMetadata:
    return RuleMetadata(
        id="demo.rule",
        name="Demo rule",
        description="Demonstrates emitter output.",
        severity=Severity.ERROR,
        category=Category.MANIFEST,
        cwe=(),
        scope=RuleScope.INCREMENTAL,
        path_filter=r".*\.py$",
        examples_violation_count=1,
        examples_clean_count=1,
    )
