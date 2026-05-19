"""Unit tests for the guard symmetry enforcement tool."""

from __future__ import annotations

import argparse
import ast
import json
import subprocess
import sys
import tempfile
from collections.abc import Generator
from pathlib import Path
from textwrap import dedent

import pytest
from scripts.cicd.enforce_guard_symmetry import (
    RULES,
    Allowlist,
    DataclassInfo,
    Finding,
    GuardSymmetryVisitor,
    LoaderInfo,
    PerFileRule,
    expected_loader_name,
    find_unguarded_pairs,
    format_finding,
    load_allowlist,
    run_check,
    scan_files,
)


def test_guard_symmetry_json_mode_succeeds_on_current_codebase() -> None:
    """Legacy guard-symmetry scanner emits JSON for shadow-mode parity."""
    result = subprocess.run(
        [
            sys.executable,
            "scripts/cicd/enforce_guard_symmetry.py",
            "check",
            "--root",
            "src/elspeth",
            "--allowlist",
            "config/cicd/enforce_guard_symmetry",
            "--format",
            "json",
        ],
        capture_output=True,
        text=True,
        cwd=Path(__file__).resolve().parents[4],
    )

    assert result.returncode == 0, f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    assert json.loads(result.stdout) == []


def scan_source(source: str, filename: str = "test.py") -> GuardSymmetryVisitor:
    """Helper to parse source and run the visitor."""
    tree = ast.parse(source, filename=filename)
    source_lines = source.splitlines()
    visitor = GuardSymmetryVisitor(filename, source_lines)
    visitor.visit(tree)
    return visitor


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


class TestDataStructures:
    """Tests for core data structures."""

    def test_rules_defined(self) -> None:
        assert "GS1" in RULES
        assert RULES["GS1"]["name"] == "missing-read-guard"

    def test_finding_canonical_key(self) -> None:
        f = Finding(
            rule_id="GS1",
            file_path="core/landscape/model_loaders.py",
            line=10,
            col=0,
            symbol_context=("RunLoader", "load"),
            fingerprint="abc123",
            code_snippet="return Run(",
            message="test",
        )
        assert f.canonical_key == "core/landscape/model_loaders.py:GS1:RunLoader:load:fp=abc123"

    def test_allowlist_match(self) -> None:
        rule = PerFileRule(
            pattern="core/landscape/*",
            rules=["GS1"],
            reason="test",
            expires=None,
        )
        al = Allowlist(per_file_rules=[rule])
        f = Finding(
            rule_id="GS1",
            file_path="core/landscape/model_loaders.py",
            line=10,
            col=0,
            symbol_context=("RunLoader",),
            fingerprint="abc123",
            code_snippet="return Run(",
            message="test",
        )
        assert al.match(f) is not None

    def test_allowlist_no_match(self) -> None:
        al = Allowlist(per_file_rules=[])
        f = Finding(
            rule_id="GS1",
            file_path="core/landscape/model_loaders.py",
            line=10,
            col=0,
            symbol_context=("RunLoader",),
            fingerprint="abc123",
            code_snippet="return Run(",
            message="test",
        )
        assert al.match(f) is None


class TestDataclassDiscovery:
    """Tests for discovering dataclasses with __post_init__ validation."""

    def test_finds_dataclass_with_validation_post_init(self) -> None:
        source = dedent("""
            from dataclasses import dataclass

            @dataclass(frozen=True)
            class Token:
                token_id: str
                step: int

                def __post_init__(self) -> None:
                    if not isinstance(self.step, int):
                        raise TypeError("step must be int")
        """)
        visitor = scan_source(source)
        assert len(visitor.dataclasses) == 1
        assert visitor.dataclasses[0].name == "Token"

    def test_ignores_dataclass_without_post_init(self) -> None:
        source = dedent("""
            from dataclasses import dataclass

            @dataclass(frozen=True)
            class Simple:
                name: str
        """)
        visitor = scan_source(source)
        assert len(visitor.dataclasses) == 0

    def test_ignores_freeze_only_post_init(self) -> None:
        source = dedent("""
            from dataclasses import dataclass

            @dataclass(frozen=True)
            class FreezeOnly:
                data: dict

                def __post_init__(self) -> None:
                    freeze_fields(self, "data")
        """)
        visitor = scan_source(source)
        assert len(visitor.dataclasses) == 0

    def test_detects_require_int_as_validation(self) -> None:
        source = dedent("""
            from dataclasses import dataclass

            @dataclass(frozen=True)
            class Row:
                row_index: int

                def __post_init__(self) -> None:
                    require_int(self.row_index, "row_index", min_value=0)
        """)
        visitor = scan_source(source)
        assert len(visitor.dataclasses) == 1
        assert visitor.dataclasses[0].name == "Row"

    def test_detects_validate_enum_as_validation(self) -> None:
        source = dedent("""
            from dataclasses import dataclass

            @dataclass(frozen=True)
            class TokenOutcome:
                outcome: str

                def __post_init__(self) -> None:
                    _validate_enum(self.outcome, TerminalOutcome, "outcome")
        """)
        visitor = scan_source(source)
        assert len(visitor.dataclasses) == 1

    def test_mixed_freeze_and_validation(self) -> None:
        """__post_init__ with both freeze_fields and validation should be detected."""
        source = dedent("""
            from dataclasses import dataclass

            @dataclass(frozen=True)
            class Mixed:
                data: dict
                step: int

                def __post_init__(self) -> None:
                    freeze_fields(self, "data")
                    require_int(self.step, "step", min_value=0)
        """)
        visitor = scan_source(source)
        assert len(visitor.dataclasses) == 1


class TestLoaderDiscovery:
    """Tests for discovering *Loader classes and checking for AuditIntegrityError."""

    def test_finds_loader_with_audit_integrity_error(self) -> None:
        source = dedent("""
            class TokenOutcomeLoader:
                def load(self, row):
                    if row.is_terminal not in (0, 1):
                        raise AuditIntegrityError("bad is_terminal")
                    return TokenOutcome(outcome_id=row.outcome_id)
        """)
        visitor = scan_source(source)
        assert len(visitor.loaders) == 1
        assert visitor.loaders[0].name == "TokenOutcomeLoader"
        assert visitor.loaders[0].target_class == "TokenOutcome"
        assert visitor.loaders[0].has_audit_integrity_error is True

    def test_finds_loader_without_audit_integrity_error(self) -> None:
        source = dedent("""
            class RunLoader:
                def load(self, row):
                    return Run(run_id=row.run_id, status=RunStatus(row.status))
        """)
        visitor = scan_source(source)
        assert len(visitor.loaders) == 1
        assert visitor.loaders[0].name == "RunLoader"
        assert visitor.loaders[0].target_class == "Run"
        assert visitor.loaders[0].has_audit_integrity_error is False

    def test_ignores_class_without_load_method(self) -> None:
        source = dedent("""
            class NotALoader:
                def process(self, row):
                    return row
        """)
        visitor = scan_source(source)
        assert len(visitor.loaders) == 0

    def test_ignores_class_not_ending_in_loader(self) -> None:
        source = dedent("""
            class RunHelper:
                def load(self, row):
                    return row
        """)
        visitor = scan_source(source)
        assert len(visitor.loaders) == 0

    def test_target_class_derived_from_name(self) -> None:
        source = dedent("""
            class RoutingEventLoader:
                def load(self, row):
                    return RoutingEvent(event_id=row.event_id)
        """)
        visitor = scan_source(source)
        assert visitor.loaders[0].target_class == "RoutingEvent"

    def test_skips_abstract_loader_with_ellipsis_body(self) -> None:
        """Protocol/ABC loaders with stub bodies should not be collected."""
        source = dedent("""
            class SecretLoader:
                def load(self, key: str) -> str:
                    ...
        """)
        visitor = scan_source(source)
        assert len(visitor.loaders) == 0

    def test_skips_abstract_loader_with_pass_body(self) -> None:
        source = dedent("""
            class BaseLoader:
                def load(self, row):
                    pass
        """)
        visitor = scan_source(source)
        assert len(visitor.loaders) == 0

    def test_skips_abstract_loader_with_not_implemented(self) -> None:
        source = dedent("""
            class AbstractLoader:
                def load(self, row):
                    raise NotImplementedError
        """)
        visitor = scan_source(source)
        assert len(visitor.loaders) == 0


class TestPairing:
    """Tests for dataclass→loader pairing and finding generation."""

    def test_expected_loader_name_default(self) -> None:
        assert expected_loader_name("Run") == "RunLoader"
        assert expected_loader_name("TokenOutcome") == "TokenOutcomeLoader"

    def test_expected_loader_name_node_state_overrides(self) -> None:
        assert expected_loader_name("NodeStateOpen") == "NodeStateLoader"
        assert expected_loader_name("NodeStatePending") == "NodeStateLoader"
        assert expected_loader_name("NodeStateCompleted") == "NodeStateLoader"
        assert expected_loader_name("NodeStateFailed") == "NodeStateLoader"

    def test_expected_loader_name_record_suffix_overrides(self) -> None:
        """*Record-suffixed dataclasses map to loaders that drop the 'Record' suffix."""
        assert expected_loader_name("TransformErrorRecord") == "TransformErrorLoader"
        assert expected_loader_name("ValidationErrorRecord") == "ValidationErrorLoader"

    def test_unguarded_pair_produces_finding(self) -> None:
        dcs = [DataclassInfo(name="Run", file_path="contracts/audit.py", line=48)]
        loaders = [
            LoaderInfo(
                name="RunLoader",
                file_path="core/landscape/model_loaders.py",
                line=53,
                target_class="Run",
                has_audit_integrity_error=False,
            )
        ]
        findings = find_unguarded_pairs(dcs, loaders)
        assert len(findings) == 1
        assert findings[0].rule_id == "GS1"
        assert "RunLoader" in findings[0].message

    def test_guarded_pair_produces_no_finding(self) -> None:
        dcs = [DataclassInfo(name="TokenOutcome", file_path="contracts/audit.py", line=642)]
        loaders = [
            LoaderInfo(
                name="TokenOutcomeLoader",
                file_path="core/landscape/model_loaders.py",
                line=467,
                target_class="TokenOutcome",
                has_audit_integrity_error=True,
            )
        ]
        findings = find_unguarded_pairs(dcs, loaders)
        assert len(findings) == 0

    def test_dataclass_without_loader_skipped(self) -> None:
        dcs = [DataclassInfo(name="Orphan", file_path="contracts/audit.py", line=100)]
        loaders: list[LoaderInfo] = []
        findings = find_unguarded_pairs(dcs, loaders)
        assert len(findings) == 0

    def test_node_state_variants_use_override(self) -> None:
        """All 4 NodeState variants should check NodeStateLoader."""
        dcs = [
            DataclassInfo(name="NodeStateOpen", file_path="contracts/audit.py", line=174),
            DataclassInfo(name="NodeStateCompleted", file_path="contracts/audit.py", line=232),
        ]
        loaders = [
            LoaderInfo(
                name="NodeStateLoader",
                file_path="core/landscape/model_loaders.py",
                line=253,
                target_class="NodeState",
                has_audit_integrity_error=True,
            )
        ]
        findings = find_unguarded_pairs(dcs, loaders)
        assert len(findings) == 0

    def test_multiple_findings_from_multiple_pairs(self) -> None:
        dcs = [
            DataclassInfo(name="Run", file_path="contracts/audit.py", line=48),
            DataclassInfo(name="Call", file_path="contracts/audit.py", line=298),
        ]
        loaders = [
            LoaderInfo(
                name="RunLoader", file_path="core/landscape/model_loaders.py", line=53, target_class="Run", has_audit_integrity_error=False
            ),
            LoaderInfo(
                name="CallLoader",
                file_path="core/landscape/model_loaders.py",
                line=190,
                target_class="Call",
                has_audit_integrity_error=False,
            ),
        ]
        findings = find_unguarded_pairs(dcs, loaders)
        assert len(findings) == 2


class TestFileScanning:
    """Tests for scanning files and producing findings."""

    def test_scan_finds_unguarded_pair_across_files(self, temp_dir: Path) -> None:
        """Dataclass in one file, loader in another — should produce finding."""
        contracts_dir = temp_dir / "contracts"
        contracts_dir.mkdir()
        core_dir = temp_dir / "core" / "landscape"
        core_dir.mkdir(parents=True)

        # Dataclass file
        (contracts_dir / "audit.py").write_text(
            dedent("""
            from dataclasses import dataclass

            @dataclass(frozen=True)
            class Run:
                run_id: str
                status: str

                def __post_init__(self) -> None:
                    if self.status not in ("running", "done"):
                        raise ValueError("bad status")
        """)
        )

        # Loader file — no AuditIntegrityError
        (core_dir / "model_loaders.py").write_text(
            dedent("""
            class RunLoader:
                def load(self, row):
                    return Run(run_id=row.run_id, status=row.status)
        """)
        )

        findings = scan_files(temp_dir)
        assert len(findings) == 1
        assert findings[0].rule_id == "GS1"
        assert "RunLoader" in findings[0].message

    def test_scan_no_findings_when_guarded(self, temp_dir: Path) -> None:
        contracts_dir = temp_dir / "contracts"
        contracts_dir.mkdir()
        core_dir = temp_dir / "core" / "landscape"
        core_dir.mkdir(parents=True)

        (contracts_dir / "audit.py").write_text(
            dedent("""
            from dataclasses import dataclass

            @dataclass(frozen=True)
            class Run:
                run_id: str
                status: str

                def __post_init__(self) -> None:
                    if self.status not in ("running", "done"):
                        raise ValueError("bad status")
        """)
        )

        (core_dir / "model_loaders.py").write_text(
            dedent("""
            class RunLoader:
                def load(self, row):
                    if row.status not in ("running", "done"):
                        raise AuditIntegrityError("invalid status")
                    return Run(run_id=row.run_id, status=row.status)
        """)
        )

        findings = scan_files(temp_dir)
        assert len(findings) == 0


class TestAllowlistLoading:
    """Tests for allowlist YAML loading."""

    def test_load_from_directory(self, temp_dir: Path) -> None:
        (temp_dir / "_defaults.yaml").write_text("version: 1\ndefaults:\n  fail_on_stale: true\n")
        (temp_dir / "landscape.yaml").write_text(
            dedent("""
            per_file_rules:
              - pattern: "core/landscape/*"
                rules: [GS1]
                reason: "int-only validation covered by __post_init__"
                expires: null
                max_hits: 5
        """)
        )
        al = load_allowlist(temp_dir)
        assert len(al.per_file_rules) == 1
        assert al.fail_on_stale is True

    def test_load_empty_returns_default(self, temp_dir: Path) -> None:
        nonexistent = temp_dir / "no_such_file.yaml"
        al = load_allowlist(nonexistent)
        assert len(al.per_file_rules) == 0


class TestReporting:
    """Tests for finding formatting."""

    def test_format_finding_includes_key_info(self) -> None:
        f = Finding(
            rule_id="GS1",
            file_path="core/landscape/model_loaders.py",
            line=53,
            col=0,
            symbol_context=("RunLoader", "load"),
            fingerprint="abc123",
            code_snippet="class RunLoader:",
            message="Run has __post_init__ validation but RunLoader.load() has no AuditIntegrityError",
        )
        text = format_finding(f)
        assert "model_loaders.py:53:0" in text
        assert "GS1" in text
        assert "missing-read-guard" in text
        assert "RunLoader:load" in text
        assert "fp=abc123" in text


class TestCLI:
    """Tests for the check command."""

    def test_check_returns_0_when_no_findings(self, temp_dir: Path) -> None:
        contracts_dir = temp_dir / "contracts"
        contracts_dir.mkdir()
        (contracts_dir / "audit.py").write_text(
            dedent("""
            from dataclasses import dataclass

            @dataclass(frozen=True)
            class Foo:
                name: str
        """)
        )

        args = argparse.Namespace(
            root=temp_dir,
            allowlist=temp_dir / "no_such_allowlist",
            files=[],
        )
        assert run_check(args) == 0

    def test_check_returns_1_when_violations(self, temp_dir: Path) -> None:
        contracts_dir = temp_dir / "contracts"
        contracts_dir.mkdir()
        core_dir = temp_dir / "core"
        core_dir.mkdir()

        (contracts_dir / "audit.py").write_text(
            dedent("""
            from dataclasses import dataclass

            @dataclass(frozen=True)
            class Widget:
                size: int

                def __post_init__(self) -> None:
                    if self.size < 0:
                        raise ValueError("negative size")
        """)
        )

        (core_dir / "loaders.py").write_text(
            dedent("""
            class WidgetLoader:
                def load(self, row):
                    return Widget(size=row.size)
        """)
        )

        args = argparse.Namespace(
            root=temp_dir,
            allowlist=temp_dir / "no_such_allowlist",
            files=[],
        )
        assert run_check(args) == 1
