"""Smoke tests for the elspeth-lints package skeleton."""

from __future__ import annotations

import ast
import json
from pathlib import Path


def test_cli_accepts_empty_rule_set(tmp_path: Path, capsys: object) -> None:
    """The package CLI can run with no rules as the initial skeleton state."""
    from elspeth_lints.core.cli import main

    exit_code = main(["check", "--rules", "nothing", "--root", str(tmp_path)])

    assert exit_code == 0
    captured = capsys.readouterr()
    assert captured.err == ""


def test_cli_dump_edges_contract_is_available(tmp_path: Path, capsys: object) -> None:
    """The foundation CLI exposes the import-edge command reserved by ADR-023."""
    from elspeth_lints.core.cli import main

    output = tmp_path / "edges.json"
    exit_code = main(["dump-edges", "--root", str(tmp_path), "--format", "json", "--output", str(output), "--no-timestamp"])

    assert exit_code == 0
    captured = capsys.readouterr()
    assert captured.err == ""
    assert captured.out == ""
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["nodes"] == []
    assert payload["edges"] == []


def test_cli_refuses_explicit_files_outside_rule_path_filter(tmp_path: Path, capsys: object) -> None:
    """Explicit --files input must respect each rule's metadata path filter."""
    import argparse
    from collections.abc import Iterable

    from elspeth_lints.core.cli import _run_check
    from elspeth_lints.core.protocols import Category, Finding, RuleContext, RuleMetadata, RuleScope, Severity
    from elspeth_lints.core.registry import RuleRegistry

    class ScopedRule:
        id = "demo.scoped"
        scope = RuleScope.INCREMENTAL
        metadata = RuleMetadata(
            id=id,
            name="Scoped",
            description="Only applies to allowed.py.",
            severity=Severity.ERROR,
            category=Category.MANIFEST,
            cwe=(),
            scope=scope,
            path_filter=r"^allowed\.py$",
            examples_violation_count=1,
            examples_clean_count=1,
        )

        def analyze(self, tree: ast.AST, file_path: Path, context: RuleContext) -> Iterable[Finding]:
            del tree, context
            return (
                Finding(
                    rule_id=self.id,
                    file_path=file_path.name,
                    line=1,
                    column=0,
                    message="should not run",
                    fingerprint="demo",
                    severity=self.metadata.severity,
                ),
            )

    outside = tmp_path / "outside.py"
    outside.write_text("value = 1\n", encoding="utf-8")
    registry = RuleRegistry()
    registry.register(ScopedRule())

    exit_code = _run_check(
        argparse.Namespace(
            rules="demo.scoped",
            rule_set="static",
            format="text",
            root=tmp_path,
            allowlist_dir=None,
            files=[outside],
        ),
        registry=registry,
    )

    assert exit_code == 2
    captured = capsys.readouterr()
    assert "outside.py" in captured.err
    assert "demo.scoped" in captured.err


def test_registry_decorator_registers_rule() -> None:
    """The registry exposes a decorator-based API for future rule modules."""
    from elspeth_lints.core.findings import Finding
    from elspeth_lints.core.protocols import RuleContext
    from elspeth_lints.core.registry import RuleRegistry

    registry = RuleRegistry()

    @registry.rule("demo.rule")
    def demo_rule(_tree: ast.AST, _file_path: Path) -> list[Finding]:
        return [
            Finding(
                rule_id="demo.rule",
                file_path="demo.py",
                line=1,
                column=0,
                message="demo finding",
                fingerprint="demo",
            )
        ]

    registered = registry.get("demo.rule")
    assert registry.ids() == ("demo.rule",)
    findings = list(registered.analyze(ast.Module(body=[], type_ignores=[]), Path("demo.py"), RuleContext(root=Path("."))))
    assert findings == demo_rule(ast.Module(body=[], type_ignores=[]), Path("demo.py"))


def test_python_walker_reports_syntax_errors_without_aborting(tmp_path: Path) -> None:
    """Syntax errors become parse results instead of crashing the whole run."""
    from elspeth_lints.core.ast_walker import ParsedPythonFile, PythonSyntaxError, walk_python_files

    good = tmp_path / "good.py"
    good.write_text("value = 1\n", encoding="utf-8")
    bad = tmp_path / "bad.py"
    bad.write_text("def broken(:\n", encoding="utf-8")

    results = list(walk_python_files(tmp_path))

    parsed = [item for item in results if isinstance(item, ParsedPythonFile)]
    syntax_errors = [item for item in results if isinstance(item, PythonSyntaxError)]
    assert [item.path.name for item in parsed] == ["good.py"]
    assert isinstance(parsed[0].tree, ast.Module)
    assert [item.path.name for item in syntax_errors] == ["bad.py"]
    assert syntax_errors[0].line == 1


def test_allowlist_loads_per_file_rules_and_exact_entries(tmp_path: Path) -> None:
    """The shared allowlist loader supports per-file and exact-key suppressions."""
    from elspeth_lints.core.allowlist import FindingKey, load_allowlist

    allowlist_file = tmp_path / "allowlist.yaml"
    allowlist_file.write_text(
        """
allow_hits:
  - key: src/example.py:R1:func:fp=abc123
    owner: tests
    reason: exact suppression
    safety: synthetic test
    expires: 2099-01-01
per_file_rules:
  - pattern: src/generated/*
    rules: [R2]
    reason: generated source
    expires: 2099-01-01
    max_hits: 2
""",
        encoding="utf-8",
    )

    allowlist = load_allowlist(allowlist_file, valid_rule_ids={"R1", "R2"})

    exact = FindingKey(
        file_path="src/example.py",
        rule_id="R1",
        symbol_context=("func",),
        fingerprint="abc123",
    )
    generated = FindingKey(
        file_path="src/generated/a.py",
        rule_id="R2",
        symbol_context=(),
        fingerprint="def456",
    )

    assert allowlist.match(exact) is not None
    assert allowlist.match(generated) is not None
    assert allowlist.get_exceeded_rules() == []


def test_allowlist_accepts_protocol_schema_fields(tmp_path: Path) -> None:
    """The allowlist schema includes the C8-3 binding fields on judge-gated entries.

    Per invariant 8, binding fields (file_fingerprint + ast_path) are
    required co-presence companions of judge_verdict; per invariant 4,
    pre-judge entries (no judge_verdict) must omit them. This test
    pins the post-C8-3 shape: binding fields ride with the judge
    quartet, not as a standalone schema extension.
    """
    from elspeth_lints.core.allowlist import FindingKey, load_allowlist
    from elspeth_lints.core.judge import DEFAULT_JUDGE_MODEL, JUDGE_POLICY_HASH

    allowlist_file = tmp_path / "allowlist.yaml"
    allowlist_file.write_text(
        f"""
allow_hits:
  - key: src/example.py:R1:Class.method:fp=abc123
    owner: platform
    reason: existing migrated finding
    safety: reviewed before migration
    expires_at: 2099-01-01
    judge_verdict: ACCEPTED
    judge_recorded_at: '2026-05-23T00:00:00+00:00'
    judge_model: {DEFAULT_JUDGE_MODEL}
    judge_policy_hash: '{JUDGE_POLICY_HASH}'
    judge_rationale: the boundary is legitimate
    file_fingerprint: sha256:source
    ast_path: Module.body[0].body[0]
""",
        encoding="utf-8",
    )

    # source_root=None: this test exercises the schema shape only, not
    # the file_fingerprint live-recompute (no source tree on disk).
    allowlist = load_allowlist(allowlist_file, valid_rule_ids={"R1"})
    finding = FindingKey(
        file_path="src/example.py",
        rule_id="R1",
        symbol_context=("Class.method",),
        fingerprint="abc123",
    )

    match = allowlist.match(finding)

    assert match is not None
    assert match.file_fingerprint == "sha256:source"
    assert match.ast_path == "Module.body[0].body[0]"
