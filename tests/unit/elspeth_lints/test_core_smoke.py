"""Smoke tests for the elspeth-lints package skeleton."""

from __future__ import annotations

import ast
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

    exit_code = main(["dump-edges", "--root", str(tmp_path), "--format", "json"])

    assert exit_code == 0
    captured = capsys.readouterr()
    assert captured.err == ""
    assert captured.out == "[]\n"


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
    """The allowlist schema includes the protocol-layer fingerprint fields."""
    from elspeth_lints.core.allowlist import FindingKey, load_allowlist

    allowlist_file = tmp_path / "allowlist.yaml"
    allowlist_file.write_text(
        """
allow_hits:
  - key: src/example.py:R1:Class.method:fp=abc123
    owner: platform
    reason: existing migrated finding
    safety: reviewed before migration
    expires_at: 2099-01-01
    file_fingerprint: sha256:source
    ast_path: Module.body[0].body[0]
""",
        encoding="utf-8",
    )

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
