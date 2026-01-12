"""
Unit tests for the no_bug_hiding enforcement tool.

Tests cover:
- Detection of each rule (R1-R4)
- Allowlist matching
- Stale allowlist detection
- Expiry behavior
"""

from __future__ import annotations

import ast
import tempfile
from datetime import date, timedelta
from pathlib import Path
from textwrap import dedent

import pytest

# Import the module under test
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "scripts" / "cicd"))
from no_bug_hiding import (
    Allowlist,
    AllowlistEntry,
    BugHidingVisitor,
    Finding,
    load_allowlist,
    scan_file,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


def parse_and_visit(source: str, filename: str = "test.py") -> list[Finding]:
    """Helper to parse source and run the visitor."""
    tree = ast.parse(source, filename=filename)
    source_lines = source.splitlines()
    visitor = BugHidingVisitor(filename, source_lines)
    visitor.visit(tree)
    return visitor.findings


# =============================================================================
# R1: dict.get() detection
# =============================================================================


class TestR1DictGet:
    """Tests for R1: dict.get() detection."""

    def test_detects_dict_get_call(self):
        """dict.get() calls should be flagged."""
        source = dedent("""
            data = {"key": "value"}
            result = data.get("key")
        """)
        findings = parse_and_visit(source)

        assert len(findings) == 1
        assert findings[0].rule_id == "R1"
        assert findings[0].line == 3

    def test_detects_dict_get_with_default(self):
        """dict.get() with default should be flagged."""
        source = dedent("""
            data = {"key": "value"}
            result = data.get("missing", "default")
        """)
        findings = parse_and_visit(source)

        assert len(findings) == 1
        assert findings[0].rule_id == "R1"

    def test_detects_chained_dict_get(self):
        """Chained .get() calls should each be flagged."""
        source = dedent("""
            nested = {"a": {"b": "value"}}
            result = nested.get("a").get("b")
        """)
        findings = parse_and_visit(source)

        # Both .get() calls should be detected
        r1_findings = [f for f in findings if f.rule_id == "R1"]
        assert len(r1_findings) == 2

    def test_get_in_function_context(self):
        """dict.get() in a function should include function in context."""
        source = dedent("""
            def process_data(data):
                return data.get("key")
        """)
        findings = parse_and_visit(source)

        assert len(findings) == 1
        assert findings[0].symbol_context == ("process_data",)

    def test_get_in_class_method_context(self):
        """dict.get() in a class method should include class and method in context."""
        source = dedent("""
            class DataProcessor:
                def process(self, data):
                    return data.get("key")
        """)
        findings = parse_and_visit(source)

        assert len(findings) == 1
        assert findings[0].symbol_context == ("DataProcessor", "process")


# =============================================================================
# R2: getattr() detection
# =============================================================================


class TestR2Getattr:
    """Tests for R2: getattr() with default detection."""

    def test_detects_getattr_with_default(self):
        """getattr() with 3 args (including default) should be flagged."""
        source = dedent("""
            class Foo:
                pass
            obj = Foo()
            value = getattr(obj, "attr", None)
        """)
        findings = parse_and_visit(source)

        r2_findings = [f for f in findings if f.rule_id == "R2"]
        assert len(r2_findings) == 1
        assert r2_findings[0].line == 5

    def test_ignores_getattr_without_default(self):
        """getattr() with only 2 args should NOT be flagged."""
        source = dedent("""
            class Foo:
                attr = "value"
            obj = Foo()
            value = getattr(obj, "attr")
        """)
        findings = parse_and_visit(source)

        r2_findings = [f for f in findings if f.rule_id == "R2"]
        assert len(r2_findings) == 0

    def test_detects_getattr_with_keyword_default(self):
        """getattr() with default as keyword arg should be flagged."""
        source = dedent("""
            obj = object()
            value = getattr(obj, "attr", default=None)
        """)
        findings = parse_and_visit(source)

        r2_findings = [f for f in findings if f.rule_id == "R2"]
        assert len(r2_findings) == 1


# =============================================================================
# R3: hasattr() detection
# =============================================================================


class TestR3Hasattr:
    """Tests for R3: hasattr() detection."""

    def test_detects_hasattr(self):
        """hasattr() calls should be flagged."""
        source = dedent("""
            obj = object()
            if hasattr(obj, "method"):
                obj.method()
        """)
        findings = parse_and_visit(source)

        r3_findings = [f for f in findings if f.rule_id == "R3"]
        assert len(r3_findings) == 1
        assert r3_findings[0].line == 3

    def test_hasattr_in_condition(self):
        """hasattr() in conditions should be flagged."""
        source = dedent("""
            result = obj.method() if hasattr(obj, "method") else None
        """)
        findings = parse_and_visit(source)

        r3_findings = [f for f in findings if f.rule_id == "R3"]
        assert len(r3_findings) == 1


# =============================================================================
# R4: Broad exception handling
# =============================================================================


class TestR4BroadExcept:
    """Tests for R4: broad exception handling detection."""

    def test_detects_bare_except(self):
        """Bare except should be flagged."""
        source = dedent("""
            try:
                risky_operation()
            except:
                pass
        """)
        findings = parse_and_visit(source)

        r4_findings = [f for f in findings if f.rule_id == "R4"]
        assert len(r4_findings) == 1

    def test_detects_except_exception(self):
        """except Exception should be flagged."""
        source = dedent("""
            try:
                risky_operation()
            except Exception:
                pass
        """)
        findings = parse_and_visit(source)

        r4_findings = [f for f in findings if f.rule_id == "R4"]
        assert len(r4_findings) == 1

    def test_detects_except_exception_as_e(self):
        """except Exception as e without re-raise should be flagged."""
        source = dedent("""
            try:
                risky_operation()
            except Exception as e:
                log_error(e)
        """)
        findings = parse_and_visit(source)

        r4_findings = [f for f in findings if f.rule_id == "R4"]
        assert len(r4_findings) == 1

    def test_ignores_except_with_reraise(self):
        """except Exception with re-raise should NOT be flagged."""
        source = dedent("""
            try:
                risky_operation()
            except Exception as e:
                log_error(e)
                raise
        """)
        findings = parse_and_visit(source)

        r4_findings = [f for f in findings if f.rule_id == "R4"]
        assert len(r4_findings) == 0

    def test_ignores_except_with_raise_new(self):
        """except Exception with raise NewError should NOT be flagged."""
        source = dedent("""
            try:
                risky_operation()
            except Exception as e:
                raise RuntimeError("wrapped") from e
        """)
        findings = parse_and_visit(source)

        r4_findings = [f for f in findings if f.rule_id == "R4"]
        assert len(r4_findings) == 0

    def test_ignores_specific_exceptions(self):
        """Catching specific exceptions should NOT be flagged."""
        source = dedent("""
            try:
                int("not a number")
            except ValueError:
                return None
        """)
        findings = parse_and_visit(source)

        r4_findings = [f for f in findings if f.rule_id == "R4"]
        assert len(r4_findings) == 0

    def test_detects_except_base_exception(self):
        """except BaseException should be flagged."""
        source = dedent("""
            try:
                risky_operation()
            except BaseException:
                pass
        """)
        findings = parse_and_visit(source)

        r4_findings = [f for f in findings if f.rule_id == "R4"]
        assert len(r4_findings) == 1


# =============================================================================
# Finding and canonical key generation
# =============================================================================


class TestFinding:
    """Tests for Finding dataclass and key generation."""

    def test_canonical_key_module_level(self):
        """Module-level finding should have _module_ in key."""
        finding = Finding(
            rule_id="R1",
            file_path="src/module.py",
            line=10,
            col=0,
            symbol_context=(),
            code_snippet="data.get('key')",
            message="test",
        )

        assert finding.canonical_key == "src/module.py:R1:_module_:line=10"

    def test_canonical_key_function(self):
        """Function-level finding should include function name."""
        finding = Finding(
            rule_id="R2",
            file_path="src/module.py",
            line=25,
            col=4,
            symbol_context=("process_data",),
            code_snippet="getattr(obj, 'x', None)",
            message="test",
        )

        assert finding.canonical_key == "src/module.py:R2:process_data:line=25"

    def test_canonical_key_class_method(self):
        """Class method finding should include class and method."""
        finding = Finding(
            rule_id="R3",
            file_path="src/handler.py",
            line=42,
            col=8,
            symbol_context=("Handler", "process"),
            code_snippet="hasattr(obj, 'attr')",
            message="test",
        )

        assert finding.canonical_key == "src/handler.py:R3:Handler:process:line=42"


# =============================================================================
# Allowlist matching
# =============================================================================


class TestAllowlistMatching:
    """Tests for allowlist entry matching."""

    def test_exact_match(self):
        """Allowlist entry should match finding with exact key."""
        entry = AllowlistEntry(
            key="src/module.py:R1:process:line=10",
            owner="test",
            reason="test",
            safety="test",
            expires=None,
        )
        allowlist = Allowlist(entries=[entry])

        finding = Finding(
            rule_id="R1",
            file_path="src/module.py",
            line=10,
            col=0,
            symbol_context=("process",),
            code_snippet="data.get('key')",
            message="test",
        )

        matched = allowlist.match(finding)
        assert matched is not None
        assert matched.key == entry.key
        assert entry.matched is True

    def test_no_match(self):
        """Finding without matching allowlist entry should return None."""
        entry = AllowlistEntry(
            key="src/other.py:R1:process:line=10",
            owner="test",
            reason="test",
            safety="test",
            expires=None,
        )
        allowlist = Allowlist(entries=[entry])

        finding = Finding(
            rule_id="R1",
            file_path="src/module.py",
            line=10,
            col=0,
            symbol_context=("process",),
            code_snippet="data.get('key')",
            message="test",
        )

        matched = allowlist.match(finding)
        assert matched is None
        assert entry.matched is False


# =============================================================================
# Stale allowlist detection
# =============================================================================


class TestStaleDetection:
    """Tests for stale allowlist entry detection."""

    def test_unmatched_entry_is_stale(self):
        """Entry that doesn't match any finding should be stale."""
        entry = AllowlistEntry(
            key="src/removed.py:R1:old_function:line=10",
            owner="test",
            reason="test",
            safety="test",
            expires=None,
        )
        allowlist = Allowlist(entries=[entry])

        # No findings matched
        stale = allowlist.get_stale_entries()
        assert len(stale) == 1
        assert stale[0].key == entry.key

    def test_matched_entry_not_stale(self):
        """Entry that matched a finding should not be stale."""
        entry = AllowlistEntry(
            key="src/module.py:R1:process:line=10",
            owner="test",
            reason="test",
            safety="test",
            expires=None,
        )
        allowlist = Allowlist(entries=[entry])

        # Simulate matching
        entry.matched = True

        stale = allowlist.get_stale_entries()
        assert len(stale) == 0


# =============================================================================
# Expiry detection
# =============================================================================


class TestExpiryDetection:
    """Tests for allowlist entry expiry detection."""

    def test_expired_entry_detected(self):
        """Entry with past expiry date should be detected."""
        yesterday = date.today() - timedelta(days=1)
        entry = AllowlistEntry(
            key="src/module.py:R1:process:line=10",
            owner="test",
            reason="test",
            safety="test",
            expires=yesterday,
        )
        allowlist = Allowlist(entries=[entry])

        expired = allowlist.get_expired_entries()
        assert len(expired) == 1
        assert expired[0].key == entry.key

    def test_future_entry_not_expired(self):
        """Entry with future expiry date should not be detected."""
        tomorrow = date.today() + timedelta(days=1)
        entry = AllowlistEntry(
            key="src/module.py:R1:process:line=10",
            owner="test",
            reason="test",
            safety="test",
            expires=tomorrow,
        )
        allowlist = Allowlist(entries=[entry])

        expired = allowlist.get_expired_entries()
        assert len(expired) == 0

    def test_no_expiry_not_expired(self):
        """Entry without expiry date should not be detected."""
        entry = AllowlistEntry(
            key="src/module.py:R1:process:line=10",
            owner="test",
            reason="test",
            safety="test",
            expires=None,
        )
        allowlist = Allowlist(entries=[entry])

        expired = allowlist.get_expired_entries()
        assert len(expired) == 0


# =============================================================================
# YAML loading
# =============================================================================


class TestYAMLLoading:
    """Tests for allowlist YAML file loading."""

    def test_load_empty_file(self, temp_dir):
        """Empty allowlist file should produce empty allowlist."""
        allowlist_path = temp_dir / "allowlist.yaml"
        allowlist_path.write_text("")

        allowlist = load_allowlist(allowlist_path)
        assert len(allowlist.entries) == 0

    def test_load_with_entries(self, temp_dir):
        """Allowlist with entries should be parsed correctly."""
        allowlist_path = temp_dir / "allowlist.yaml"
        allowlist_path.write_text("""
version: 1
defaults:
  fail_on_stale: true
  fail_on_expired: false
allow_hits:
  - key: "src/module.py:R1:process:line=10"
    owner: "john"
    reason: "Legacy code"
    safety: "Will be refactored"
    expires: "2026-06-01"
""")

        allowlist = load_allowlist(allowlist_path)
        assert len(allowlist.entries) == 1
        assert allowlist.entries[0].owner == "john"
        assert allowlist.entries[0].expires == date(2026, 6, 1)
        assert allowlist.fail_on_stale is True
        assert allowlist.fail_on_expired is False

    def test_load_nonexistent_file(self, temp_dir):
        """Missing allowlist file should produce empty allowlist."""
        allowlist_path = temp_dir / "missing.yaml"

        allowlist = load_allowlist(allowlist_path)
        assert len(allowlist.entries) == 0


# =============================================================================
# File scanning
# =============================================================================


class TestFileScanning:
    """Tests for scanning Python files."""

    def test_scan_file_with_violations(self, temp_dir):
        """File with violations should produce findings."""
        py_file = temp_dir / "test_module.py"
        py_file.write_text(dedent("""
            def process(data):
                return data.get("key", None)
        """))

        findings = scan_file(py_file, temp_dir)
        assert len(findings) == 1
        assert findings[0].rule_id == "R1"
        assert findings[0].file_path == "test_module.py"

    def test_scan_file_no_violations(self, temp_dir):
        """Clean file should produce no findings."""
        py_file = temp_dir / "clean_module.py"
        py_file.write_text(dedent("""
            def process(data):
                return data["key"]
        """))

        findings = scan_file(py_file, temp_dir)
        assert len(findings) == 0

    def test_scan_file_syntax_error(self, temp_dir):
        """File with syntax error should not crash."""
        py_file = temp_dir / "broken.py"
        py_file.write_text("def broken(\n")  # syntax error

        findings = scan_file(py_file, temp_dir)
        assert len(findings) == 0  # No crash, just empty


# =============================================================================
# Integration tests
# =============================================================================


class TestIntegration:
    """End-to-end integration tests."""

    def test_finding_allowlisted_and_stale_detection(self, temp_dir):
        """Full workflow: findings, allowlisting, and stale detection."""
        # Create a file with one violation
        py_file = temp_dir / "module.py"
        py_file.write_text(dedent("""
            def process(data):
                return data.get("key")
        """))

        # Scan and get finding
        findings = scan_file(py_file, temp_dir)
        assert len(findings) == 1

        finding = findings[0]

        # Create allowlist with matching entry and one stale entry
        entry_matching = AllowlistEntry(
            key=finding.canonical_key,
            owner="test",
            reason="test",
            safety="test",
            expires=None,
        )
        entry_stale = AllowlistEntry(
            key="module.py:R1:old_function:line=999",
            owner="test",
            reason="test",
            safety="test",
            expires=None,
        )
        allowlist = Allowlist(entries=[entry_matching, entry_stale])

        # Match finding
        matched = allowlist.match(finding)
        assert matched is not None

        # Check stale entries
        stale = allowlist.get_stale_entries()
        assert len(stale) == 1
        assert stale[0].key == entry_stale.key
