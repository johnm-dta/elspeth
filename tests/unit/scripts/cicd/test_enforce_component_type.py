"""Tests for enforce_component_type.py CI scanner.

Tests the ComponentTypeScanner AST analysis, cross-file inheritance resolution,
allowlist matching, and scan integration. Uses tmp_path fixtures with real files
because cross-file inheritance resolution requires a filesystem.

Follows the pattern established by test_enforce_freeze_guards.py.
"""

from __future__ import annotations

import argparse
import textwrap
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
import yaml
from scripts.cicd.enforce_component_type import (
    Allowlist,
    Finding,
    PerFileRule,
    load_allowlist,
    run_check,
    scan_and_resolve,
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
    """Scan all .py files under tmp_path and resolve inheritance."""
    return scan_and_resolve(tmp_path)


# =============================================================================
# CT1: Missing _plugin_component_type detection
# =============================================================================


class TestCT1Detection:
    """CT1 detects DataPluginConfig subclasses missing _plugin_component_type."""

    def test_direct_subclass_missing_type(self, tmp_path: Path) -> None:
        """Direct DataPluginConfig subclass without _plugin_component_type is flagged."""
        _write_py(
            tmp_path,
            "bad.py",
            """
            from elspeth.plugins.infrastructure.config_base import DataPluginConfig

            class BadConfig(DataPluginConfig):
                path: str
            """,
        )
        findings = _scan(tmp_path)
        assert len(findings) == 1
        assert findings[0].rule_id == "CT1"
        assert "BadConfig" in findings[0].message

    def test_direct_subclass_with_type_not_flagged(self, tmp_path: Path) -> None:
        """Direct DataPluginConfig subclass with _plugin_component_type is not flagged."""
        _write_py(
            tmp_path,
            "good.py",
            """
            from elspeth.plugins.infrastructure.config_base import DataPluginConfig
            from typing import ClassVar

            class GoodConfig(DataPluginConfig):
                _plugin_component_type: ClassVar[str | None] = "transform"
                path: str
            """,
        )
        findings = _scan(tmp_path)
        assert len(findings) == 0

    def test_exempt_class_not_flagged(self, tmp_path: Path) -> None:
        """Class with _component_type_exempt = True is not flagged."""
        _write_py(
            tmp_path,
            "exempt.py",
            """
            from elspeth.plugins.infrastructure.config_base import DataPluginConfig
            from typing import ClassVar

            class IntermediateConfig(DataPluginConfig):
                _component_type_exempt: ClassVar[bool] = True
                path: str
            """,
        )
        findings = _scan(tmp_path)
        assert len(findings) == 0

    def test_exempt_does_not_propagate_to_children(self, tmp_path: Path) -> None:
        """Children of an exempt class are still checked."""
        _write_py(
            tmp_path,
            "hierarchy.py",
            """
            from elspeth.plugins.infrastructure.config_base import DataPluginConfig
            from typing import ClassVar

            class MiddleConfig(DataPluginConfig):
                _component_type_exempt: ClassVar[bool] = True

            class ChildConfig(MiddleConfig):
                pass
            """,
        )
        findings = _scan(tmp_path)
        assert len(findings) == 1
        assert "ChildConfig" in findings[0].message

    def test_inherited_type_from_known_base(self, tmp_path: Path) -> None:
        """Subclass of a class that sets _plugin_component_type is not flagged."""
        _write_py(
            tmp_path,
            "good_source.py",
            """
            from elspeth.plugins.infrastructure.config_base import SourceDataConfig

            class MySourceConfig(SourceDataConfig):
                extra: str = "default"
            """,
        )
        findings = _scan(tmp_path)
        assert len(findings) == 0

    def test_inherited_type_through_local_chain(self, tmp_path: Path) -> None:
        """Deeply nested class inheriting through chain is not flagged."""
        _write_py(
            tmp_path,
            "chain.py",
            """
            from elspeth.plugins.infrastructure.config_base import TransformDataConfig

            class MiddleConfig(TransformDataConfig):
                buffer: int = 100

            class LeafConfig(MiddleConfig):
                mode: str = "fast"
            """,
        )
        findings = _scan(tmp_path)
        assert len(findings) == 0

    def test_path_config_subclass_without_type_flagged(self, tmp_path: Path) -> None:
        """PathConfig subclass without _plugin_component_type is flagged."""
        _write_py(
            tmp_path,
            "bad_path.py",
            """
            from elspeth.plugins.infrastructure.config_base import PathConfig

            class BadPathConfig(PathConfig):
                encoding: str = "utf-8"
            """,
        )
        findings = _scan(tmp_path)
        assert len(findings) == 1
        assert "BadPathConfig" in findings[0].message

    def test_non_data_plugin_config_class_ignored(self, tmp_path: Path) -> None:
        """Classes not inheriting from DataPluginConfig hierarchy are not checked."""
        _write_py(
            tmp_path,
            "unrelated.py",
            """
            from elspeth.plugins.infrastructure.config_base import PluginConfig

            class UnrelatedConfig(PluginConfig):
                name: str

            class PlainClass:
                x: int = 0
            """,
        )
        findings = _scan(tmp_path)
        assert len(findings) == 0

    def test_multiple_violations_in_one_file(self, tmp_path: Path) -> None:
        """Multiple classes missing type in one file all get flagged."""
        _write_py(
            tmp_path,
            "multi.py",
            """
            from elspeth.plugins.infrastructure.config_base import DataPluginConfig

            class Bad1(DataPluginConfig):
                pass

            class Bad2(DataPluginConfig):
                pass
            """,
        )
        findings = _scan(tmp_path)
        assert len(findings) == 2
        names = {f.message for f in findings}
        assert any("Bad1" in n for n in names)
        assert any("Bad2" in n for n in names)

    def test_relative_file_path_in_finding(self, tmp_path: Path) -> None:
        """Finding file_path is relative to scan root."""
        subdir = tmp_path / "subdir"
        _write_py(
            subdir,
            "missing.py",
            """
            from elspeth.plugins.infrastructure.config_base import DataPluginConfig

            class MissConfig(DataPluginConfig):
                pass
            """,
        )
        findings = _scan(tmp_path)
        assert len(findings) == 1
        assert findings[0].file_path == "subdir/missing.py"


# =============================================================================
# Cross-file inheritance resolution
# =============================================================================


class TestCrossFileResolution:
    """Tests for cross-file inheritance chain resolution."""

    def test_type_set_in_separate_base_file(self, tmp_path: Path) -> None:
        """Class in file B inheriting from class in file A that sets type."""
        _write_py(
            tmp_path,
            "base_config.py",
            """
            from elspeth.plugins.infrastructure.config_base import DataPluginConfig
            from typing import ClassVar

            class MyBaseConfig(DataPluginConfig):
                _plugin_component_type: ClassVar[str | None] = "sink"
            """,
        )
        _write_py(
            tmp_path,
            "leaf_config.py",
            """
            from base_config import MyBaseConfig

            class LeafConfig(MyBaseConfig):
                output_format: str = "json"
            """,
        )
        findings = _scan(tmp_path)
        assert len(findings) == 0

    def test_type_missing_across_files(self, tmp_path: Path) -> None:
        """Class in file B inheriting from exempt class in file A — still flagged."""
        _write_py(
            tmp_path,
            "base_config.py",
            """
            from elspeth.plugins.infrastructure.config_base import DataPluginConfig
            from typing import ClassVar

            class MyMiddle(DataPluginConfig):
                _component_type_exempt: ClassVar[bool] = True
            """,
        )
        _write_py(
            tmp_path,
            "leaf_config.py",
            """
            from base_config import MyMiddle

            class BadLeaf(MyMiddle):
                pass
            """,
        )
        findings = _scan(tmp_path)
        assert len(findings) == 1
        assert "BadLeaf" in findings[0].message

    def test_syntax_error_file_skipped(self, tmp_path: Path) -> None:
        """Files with syntax errors are skipped without crashing."""
        _write_py(tmp_path, "bad_syntax.py", "def foo(:\n    pass\n")
        _write_py(
            tmp_path,
            "good.py",
            """
            from elspeth.plugins.infrastructure.config_base import DataPluginConfig
            from typing import ClassVar

            class GoodConfig(DataPluginConfig):
                _plugin_component_type: ClassVar[str | None] = "source"
            """,
        )
        findings = _scan(tmp_path)
        assert len(findings) == 0


# =============================================================================
# Allowlist
# =============================================================================


class TestAllowlist:
    """Allowlist matching, staleness, and loading."""

    def _make_finding(self, file_path: str = "plugins/bad.py") -> Finding:
        return Finding(
            rule_id="CT1",
            file_path=file_path,
            line=10,
            col=0,
            symbol_context=("BadConfig",),
            fingerprint="abc123",
            code_snippet="class BadConfig(DataPluginConfig):",
            message="test",
        )

    def test_suppresses_finding(self) -> None:
        """Allowlisted finding is matched."""
        allowlist = Allowlist(
            per_file_rules=[
                PerFileRule(
                    pattern="plugins/bad.py",
                    rules=["CT1"],
                    reason="intermediate base",
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
                    rules=["CT1"],
                    reason="test",
                    expires=None,
                ),
            ]
        )
        assert allowlist.match(self._make_finding()) is None

    def test_glob_pattern_matching(self) -> None:
        """Glob patterns match file paths correctly."""
        rule = PerFileRule(pattern="plugins/*.py", rules=["CT1"], reason="test", expires=None)
        assert rule.matches("plugins/bad.py", "CT1") is True
        assert rule.matches("other/bad.py", "CT1") is False

    def test_max_hits_exceeded_reported(self) -> None:
        """Rules with matched_count > max_hits are reported."""
        rule = PerFileRule(pattern="test.py", rules=["CT1"], reason="test", expires=None, max_hits=1)
        rule.matched_count = 2
        allowlist = Allowlist(per_file_rules=[rule])
        assert len(allowlist.get_exceeded_rules()) == 1

    def test_unused_rule_reported(self) -> None:
        """Rules with zero matches are reported as unused."""
        rule = PerFileRule(pattern="nonexistent.py", rules=["CT1"], reason="test", expires=None)
        allowlist = Allowlist(per_file_rules=[rule])
        assert len(allowlist.get_unused_rules()) == 1

    def test_expired_rule_reported(self) -> None:
        """Rules past expiry date are reported."""
        yesterday = (datetime.now(UTC) - timedelta(days=1)).date()
        rule = PerFileRule(pattern="test.py", rules=["CT1"], reason="test", expires=yesterday)
        allowlist = Allowlist(per_file_rules=[rule])
        assert len(allowlist.get_expired_rules()) == 1

    def test_non_expired_rule_not_reported(self) -> None:
        """Rules before expiry are not reported."""
        tomorrow = (datetime.now(UTC) + timedelta(days=1)).date()
        rule = PerFileRule(pattern="test.py", rules=["CT1"], reason="test", expires=tomorrow)
        allowlist = Allowlist(per_file_rules=[rule])
        assert len(allowlist.get_expired_rules()) == 0

    def test_directory_loading(self, tmp_path: Path) -> None:
        """load_allowlist reads a directory of YAML files with _defaults.yaml."""
        (tmp_path / "_defaults.yaml").write_text("defaults:\n  fail_on_stale: false\n")
        (tmp_path / "rules.yaml").write_text(yaml.dump({"per_file_rules": [{"pattern": "foo.py", "rules": ["CT1"], "reason": "test"}]}))

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
        """No violations → exit 0."""
        src = tmp_path / "src"
        _write_py(
            src,
            "good.py",
            """
            from elspeth.plugins.infrastructure.config_base import DataPluginConfig
            from typing import ClassVar

            class GoodConfig(DataPluginConfig):
                _plugin_component_type: ClassVar[str | None] = "transform"
            """,
        )
        allowlist_dir = tmp_path / "allowlist"
        allowlist_dir.mkdir()

        assert run_check(self._make_args(root=src, allowlist=allowlist_dir)) == 0

    def test_violation_returns_one(self, tmp_path: Path) -> None:
        """Unallowlisted violation → exit 1."""
        src = tmp_path / "src"
        _write_py(
            src,
            "bad.py",
            """
            from elspeth.plugins.infrastructure.config_base import DataPluginConfig

            class BadConfig(DataPluginConfig):
                path: str
            """,
        )
        allowlist_dir = tmp_path / "allowlist"
        allowlist_dir.mkdir()

        assert run_check(self._make_args(root=src, allowlist=allowlist_dir)) == 1

    def test_allowlisted_violation_returns_zero(self, tmp_path: Path) -> None:
        """Violation covered by allowlist → exit 0."""
        src = tmp_path / "src"
        _write_py(
            src,
            "bad.py",
            """
            from elspeth.plugins.infrastructure.config_base import DataPluginConfig

            class BadConfig(DataPluginConfig):
                path: str
            """,
        )
        allowlist_dir = tmp_path / "allowlist"
        allowlist_dir.mkdir()
        (allowlist_dir / "rules.yaml").write_text(
            yaml.dump({"per_file_rules": [{"pattern": "bad.py", "rules": ["CT1"], "reason": "test", "max_hits": 1}]})
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
            yaml.dump({"per_file_rules": [{"pattern": "nonexistent.py", "rules": ["CT1"], "reason": "stale"}]})
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
            yaml.dump({"per_file_rules": [{"pattern": "nonexistent.py", "rules": ["CT1"], "reason": "stale"}]})
        )

        assert run_check(self._make_args(root=src, allowlist=allowlist_dir, files=[clean_file])) == 0


# =============================================================================
# Finding data structure
# =============================================================================


class TestFinding:
    """Tests for Finding data structure."""

    def test_canonical_key_format(self) -> None:
        """canonical_key includes file path, rule, class name, and fingerprint."""
        finding = Finding(
            rule_id="CT1",
            file_path="plugins/bad.py",
            line=10,
            col=0,
            symbol_context=("BadConfig",),
            fingerprint="abc123",
            code_snippet="class BadConfig(DataPluginConfig):",
            message="test",
        )
        key = finding.canonical_key
        assert "plugins/bad.py" in key
        assert "CT1" in key
        assert "BadConfig" in key
        assert "fp=" in key

    def test_canonical_key_module_level_sentinel(self) -> None:
        """canonical_key uses _module_ when no symbol context."""
        finding = Finding(
            rule_id="CT1",
            file_path="test.py",
            line=1,
            col=0,
            symbol_context=(),
            fingerprint="abc123",
            code_snippet="x = 1",
            message="test",
        )
        assert "_module_" in finding.canonical_key
