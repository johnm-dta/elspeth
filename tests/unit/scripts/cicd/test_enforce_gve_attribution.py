"""Tests for enforce_gve_attribution.py CI scanner.

Tests the GA1 rule: raise GraphValidationError(...) without component_id=.
Uses tmp_path fixtures with real files.

Follows the pattern established by test_enforce_component_type.py.
"""

from __future__ import annotations

import argparse
import textwrap
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
import yaml
from scripts.cicd.enforce_gve_attribution import (
    Allowlist,
    Finding,
    PerFileRule,
    load_allowlist,
    run_check,
    scan_all,
)

# =============================================================================
# Helpers
# =============================================================================


def _write_py(directory: Path, name: str, source: str) -> Path:
    """Write a dedented source string to a Python file in directory."""
    directory.mkdir(parents=True, exist_ok=True)
    f = directory / name
    f.write_text(textwrap.dedent(source))
    return f


def _scan(tmp_path: Path) -> list[Finding]:
    """Scan all .py files under tmp_path."""
    return scan_all(tmp_path)


# =============================================================================
# GA1: Missing component_id detection
# =============================================================================


class TestGA1Detection:
    """GA1 detects raise GraphValidationError(...) without component_id=."""

    def test_missing_component_id_flagged(self, tmp_path: Path) -> None:
        """raise GraphValidationError('msg') is flagged."""
        _write_py(
            tmp_path,
            "bad.py",
            """
            from elspeth.core.dag.models import GraphValidationError

            def validate():
                raise GraphValidationError("bad")
            """,
        )
        findings = _scan(tmp_path)
        assert len(findings) == 1
        assert findings[0].rule_id == "GA1"

    def test_with_component_id_not_flagged(self, tmp_path: Path) -> None:
        """raise GraphValidationError('msg', component_id='x') is not flagged."""
        _write_py(
            tmp_path,
            "good.py",
            """
            from elspeth.core.dag.models import GraphValidationError

            def validate():
                raise GraphValidationError("ok", component_id="node_1")
            """,
        )
        findings = _scan(tmp_path)
        assert len(findings) == 0

    def test_with_both_kwargs_not_flagged(self, tmp_path: Path) -> None:
        """raise GraphValidationError('msg', component_id='x', component_type='y') is not flagged."""
        _write_py(
            tmp_path,
            "good.py",
            """
            from elspeth.core.dag.models import GraphValidationError

            def validate():
                raise GraphValidationError("ok", component_id="n", component_type="gate")
            """,
        )
        findings = _scan(tmp_path)
        assert len(findings) == 0

    def test_component_type_only_still_flagged(self, tmp_path: Path) -> None:
        """component_type without component_id is still flagged."""
        _write_py(
            tmp_path,
            "partial.py",
            """
            from elspeth.core.dag.models import GraphValidationError

            def validate():
                raise GraphValidationError("partial", component_type="coalesce")
            """,
        )
        findings = _scan(tmp_path)
        assert len(findings) == 1

    def test_non_gve_raise_ignored(self, tmp_path: Path) -> None:
        """raise ValueError(...) is not checked."""
        _write_py(
            tmp_path,
            "other.py",
            """
            def validate():
                raise ValueError("not a GVE")
            """,
        )
        findings = _scan(tmp_path)
        assert len(findings) == 0

    def test_multiple_violations_in_one_file(self, tmp_path: Path) -> None:
        """Multiple raise sites without component_id are all flagged."""
        _write_py(
            tmp_path,
            "multi.py",
            """
            from elspeth.core.dag.models import GraphValidationError

            def first():
                raise GraphValidationError("a")

            def second():
                raise GraphValidationError("b")
            """,
        )
        findings = _scan(tmp_path)
        assert len(findings) == 2

    def test_mixed_good_and_bad(self, tmp_path: Path) -> None:
        """Only raise sites without component_id are flagged."""
        _write_py(
            tmp_path,
            "mixed.py",
            """
            from elspeth.core.dag.models import GraphValidationError

            def bad():
                raise GraphValidationError("no id")

            def good():
                raise GraphValidationError("has id", component_id="x")
            """,
        )
        findings = _scan(tmp_path)
        assert len(findings) == 1
        assert "bad" in findings[0].symbol_context

    def test_attribute_access_call_detected(self, tmp_path: Path) -> None:
        """raise models.GraphValidationError(...) is detected."""
        _write_py(
            tmp_path,
            "qualified.py",
            """
            from elspeth.core.dag import models

            def validate():
                raise models.GraphValidationError("qualified call")
            """,
        )
        findings = _scan(tmp_path)
        assert len(findings) == 1

    def test_relative_file_path_in_finding(self, tmp_path: Path) -> None:
        """Finding file_path is relative to scan root."""
        subdir = tmp_path / "subdir"
        _write_py(
            subdir,
            "inner.py",
            """
            from elspeth.core.dag.models import GraphValidationError

            def validate():
                raise GraphValidationError("nested")
            """,
        )
        findings = _scan(tmp_path)
        assert len(findings) == 1
        assert findings[0].file_path == "subdir/inner.py"

    def test_enclosing_class_in_symbol_context(self, tmp_path: Path) -> None:
        """Symbol context includes enclosing class name."""
        _write_py(
            tmp_path,
            "classed.py",
            """
            from elspeth.core.dag.models import GraphValidationError

            class MyGraph:
                def validate(self):
                    raise GraphValidationError("in class")
            """,
        )
        findings = _scan(tmp_path)
        assert len(findings) == 1
        assert findings[0].symbol_context == ("MyGraph", "validate")

    def test_syntax_error_file_skipped(self, tmp_path: Path) -> None:
        """Files with syntax errors are skipped without crashing."""
        _write_py(tmp_path, "bad_syntax.py", "def foo(:\n    pass\n")
        _write_py(
            tmp_path,
            "good.py",
            """
            from elspeth.core.dag.models import GraphValidationError

            def validate():
                raise GraphValidationError("ok", component_id="x")
            """,
        )
        findings = _scan(tmp_path)
        assert len(findings) == 0


# =============================================================================
# Allowlist
# =============================================================================


class TestAllowlist:
    """Allowlist matching, staleness, and loading."""

    def _make_finding(self, file_path: str = "core/dag/graph.py") -> Finding:
        return Finding(
            rule_id="GA1",
            file_path=file_path,
            line=100,
            col=8,
            symbol_context=("ExecutionGraph", "validate"),
            fingerprint="abc123",
            code_snippet="raise GraphValidationError('cycle')",
            message="test",
        )

    def test_suppresses_finding(self) -> None:
        """Allowlisted finding is matched."""
        allowlist = Allowlist(
            per_file_rules=[
                PerFileRule(
                    pattern="core/dag/graph.py",
                    rules=["GA1"],
                    reason="structural",
                    expires=None,
                ),
            ]
        )
        assert allowlist.match(self._make_finding()) is not None

    def test_non_matching_pattern_not_suppressed(self) -> None:
        """Finding with non-matching pattern is not suppressed."""
        allowlist = Allowlist(
            per_file_rules=[
                PerFileRule(
                    pattern="other/*.py",
                    rules=["GA1"],
                    reason="test",
                    expires=None,
                ),
            ]
        )
        assert allowlist.match(self._make_finding()) is None

    def test_glob_pattern_matching(self) -> None:
        """Glob patterns match file paths correctly."""
        rule = PerFileRule(pattern="core/dag/*.py", rules=["GA1"], reason="test", expires=None)
        assert rule.matches("core/dag/graph.py", "GA1") is True
        assert rule.matches("core/other.py", "GA1") is False

    def test_max_hits_exceeded_reported(self) -> None:
        """Rules with matched_count > max_hits are reported."""
        rule = PerFileRule(pattern="test.py", rules=["GA1"], reason="test", expires=None, max_hits=1)
        rule.matched_count = 2
        allowlist = Allowlist(per_file_rules=[rule])
        assert len(allowlist.get_exceeded_rules()) == 1

    def test_unused_rule_reported(self) -> None:
        """Rules with zero matches are reported as unused."""
        rule = PerFileRule(pattern="nonexistent.py", rules=["GA1"], reason="test", expires=None)
        allowlist = Allowlist(per_file_rules=[rule])
        assert len(allowlist.get_unused_rules()) == 1

    def test_expired_rule_reported(self) -> None:
        """Rules past expiry date are reported."""
        yesterday = (datetime.now(UTC) - timedelta(days=1)).date()
        rule = PerFileRule(pattern="test.py", rules=["GA1"], reason="test", expires=yesterday)
        allowlist = Allowlist(per_file_rules=[rule])
        assert len(allowlist.get_expired_rules()) == 1

    def test_non_expired_rule_not_reported(self) -> None:
        """Rules before expiry are not reported."""
        tomorrow = (datetime.now(UTC) + timedelta(days=1)).date()
        rule = PerFileRule(pattern="test.py", rules=["GA1"], reason="test", expires=tomorrow)
        allowlist = Allowlist(per_file_rules=[rule])
        assert len(allowlist.get_expired_rules()) == 0

    def test_directory_loading(self, tmp_path: Path) -> None:
        """load_allowlist reads a directory of YAML files with _defaults.yaml."""
        (tmp_path / "_defaults.yaml").write_text("defaults:\n  fail_on_stale: false\n")
        (tmp_path / "rules.yaml").write_text(yaml.dump({"per_file_rules": [{"pattern": "foo.py", "rules": ["GA1"], "reason": "test"}]}))

        allowlist = load_allowlist(tmp_path)
        assert allowlist.fail_on_stale is False
        assert len(allowlist.per_file_rules) == 1

    def test_unknown_rule_id_exits(self, tmp_path: Path) -> None:
        """Unknown rule IDs in allowlist cause sys.exit(1)."""
        (tmp_path / "bad.yaml").write_text(yaml.dump({"per_file_rules": [{"pattern": "foo.py", "rules": ["BAD"], "reason": "test"}]}))
        with pytest.raises(SystemExit) as exc_info:
            load_allowlist(tmp_path)
        assert exc_info.value.code == 1

    def test_nonexistent_path_returns_empty_allowlist(self) -> None:
        """load_allowlist with nonexistent path returns empty allowlist."""
        allowlist = load_allowlist(Path("/nonexistent/path.yaml"))
        assert len(allowlist.per_file_rules) == 0
        assert allowlist.fail_on_stale is True


# =============================================================================
# run_check integration
# =============================================================================


class TestRunCheck:
    """Integration tests for run_check — the CI entry point."""

    def _make_args(
        self,
        root: Path,
        allowlist: Path | None = None,
        files: list[Path] | None = None,
    ) -> argparse.Namespace:
        return argparse.Namespace(
            command="check",
            root=root,
            allowlist=allowlist,
            files=files or [],
        )

    def test_clean_codebase_returns_zero(self, tmp_path: Path) -> None:
        """No violations -> exit 0."""
        src = tmp_path / "src"
        _write_py(
            src,
            "good.py",
            """
            from elspeth.core.dag.models import GraphValidationError

            def validate():
                raise GraphValidationError("ok", component_id="node_1")
            """,
        )
        allowlist_dir = tmp_path / "allowlist"
        allowlist_dir.mkdir()

        assert run_check(self._make_args(root=src, allowlist=allowlist_dir)) == 0

    def test_violation_returns_one(self, tmp_path: Path) -> None:
        """Unallowlisted violation -> exit 1."""
        src = tmp_path / "src"
        _write_py(
            src,
            "bad.py",
            """
            from elspeth.core.dag.models import GraphValidationError

            def validate():
                raise GraphValidationError("no id")
            """,
        )
        allowlist_dir = tmp_path / "allowlist"
        allowlist_dir.mkdir()

        assert run_check(self._make_args(root=src, allowlist=allowlist_dir)) == 1

    def test_allowlisted_violation_returns_zero(self, tmp_path: Path) -> None:
        """Violation covered by allowlist -> exit 0."""
        src = tmp_path / "src"
        _write_py(
            src,
            "structural.py",
            """
            from elspeth.core.dag.models import GraphValidationError

            def validate():
                raise GraphValidationError("cycle detected")
            """,
        )
        allowlist_dir = tmp_path / "allowlist"
        allowlist_dir.mkdir()
        (allowlist_dir / "rules.yaml").write_text(
            yaml.dump({"per_file_rules": [{"pattern": "structural.py", "rules": ["GA1"], "reason": "structural", "max_hits": 1}]})
        )

        assert run_check(self._make_args(root=src, allowlist=allowlist_dir)) == 0

    def test_stale_rule_fails_in_full_scan(self, tmp_path: Path) -> None:
        """Unused allowlist rules fail in full-scan mode when fail_on_stale is true."""
        src = tmp_path / "src"
        src.mkdir()
        (src / "clean.py").write_text("x = 1\n")

        allowlist_dir = tmp_path / "allowlist"
        allowlist_dir.mkdir()
        (allowlist_dir / "_defaults.yaml").write_text("defaults:\n  fail_on_stale: true\n")
        (allowlist_dir / "rules.yaml").write_text(
            yaml.dump({"per_file_rules": [{"pattern": "nonexistent.py", "rules": ["GA1"], "reason": "stale"}]})
        )

        assert run_check(self._make_args(root=src, allowlist=allowlist_dir)) == 1

    def test_precommit_mode_skips_staleness(self, tmp_path: Path) -> None:
        """Pre-commit mode (files arg) skips unused rule check."""
        src = tmp_path / "src"
        src.mkdir()
        clean_file = src / "clean.py"
        clean_file.write_text("x = 1\n")

        allowlist_dir = tmp_path / "allowlist"
        allowlist_dir.mkdir()
        (allowlist_dir / "rules.yaml").write_text(
            yaml.dump({"per_file_rules": [{"pattern": "nonexistent.py", "rules": ["GA1"], "reason": "stale"}]})
        )

        assert run_check(self._make_args(root=src, allowlist=allowlist_dir, files=[clean_file])) == 0


# =============================================================================
# Finding data structure
# =============================================================================


class TestFinding:
    """Tests for Finding data structure."""

    def test_canonical_key_format(self) -> None:
        """canonical_key includes file path, rule, scope, and fingerprint."""
        finding = Finding(
            rule_id="GA1",
            file_path="core/dag/graph.py",
            line=100,
            col=8,
            symbol_context=("ExecutionGraph", "validate"),
            fingerprint="abc123",
            code_snippet="raise GraphValidationError('test')",
            message="test",
        )
        key = finding.canonical_key
        assert "core/dag/graph.py" in key
        assert "GA1" in key
        assert "ExecutionGraph:validate" in key
        assert "fp=" in key

    def test_canonical_key_module_level_sentinel(self) -> None:
        """canonical_key uses _module_ when no symbol context."""
        finding = Finding(
            rule_id="GA1",
            file_path="test.py",
            line=1,
            col=0,
            symbol_context=(),
            fingerprint="abc123",
            code_snippet="raise GraphValidationError('test')",
            message="test",
        )
        assert "_module_" in finding.canonical_key


# =============================================================================
# Real codebase integration — scanner + allowlist against src/elspeth
# =============================================================================


class TestRealCodebase:
    """Verify the scanner passes against the actual codebase with allowlist."""

    def test_scanner_passes_with_allowlist(self) -> None:
        """The scanner must pass against src/elspeth with the committed allowlist."""
        # tests/unit/scripts/cicd/ → 5 parents to reach repo root
        repo_root = Path(__file__).resolve().parent.parent.parent.parent.parent
        root = repo_root / "src" / "elspeth"
        allowlist_path = repo_root / "config" / "cicd" / "enforce_gve_attribution"

        args = argparse.Namespace(
            command="check",
            root=root,
            allowlist=allowlist_path,
            files=[],
        )
        assert run_check(args) == 0, "enforce_gve_attribution check failed against real codebase"
