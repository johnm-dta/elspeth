"""
Unit tests for the tier model enforcement tool.

Tests cover:
- Detection of each rule (R1-R4)
- Allowlist matching
- Stale allowlist detection
- Expiry behavior
"""

from __future__ import annotations

import argparse
import ast
import tempfile
from collections.abc import Generator
from datetime import UTC, datetime, timedelta
from pathlib import Path
from textwrap import dedent

import pytest
from scripts.cicd.enforce_tier_model import (
    Allowlist,
    AllowlistEntry,
    Finding,
    PerFileRule,
    TierModelVisitor,
    _parse_allow_hits,
    _parse_per_file_rules,
    _suggest_module_file,
    format_stale_entry_text,
    load_allowlist,
    run_check,
    scan_file,
)

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


def parse_and_visit(source: str, filename: str = "test.py") -> list[Finding]:
    """Helper to parse source and run the visitor."""
    tree = ast.parse(source, filename=filename)
    source_lines = source.splitlines()
    visitor = TierModelVisitor(filename, source_lines)
    visitor.visit(tree)
    return visitor.findings


# =============================================================================
# R1: dict.get() detection
# =============================================================================


class TestR1DictGet:
    """Tests for R1: dict.get() detection."""

    def test_detects_dict_get_call(self) -> None:
        """dict.get() calls should be flagged."""
        source = dedent("""
            data = {"key": "value"}
            result = data.get("key")
        """)
        findings = parse_and_visit(source)

        assert len(findings) == 1
        assert findings[0].rule_id == "R1"
        assert findings[0].line == 3

    def test_detects_dict_get_with_default(self) -> None:
        """dict.get() with default should be flagged."""
        source = dedent("""
            data = {"key": "value"}
            result = data.get("missing", "default")
        """)
        findings = parse_and_visit(source)

        assert len(findings) == 1
        assert findings[0].rule_id == "R1"

    def test_detects_chained_dict_get(self) -> None:
        """Chained .get() calls should each be flagged."""
        source = dedent("""
            nested = {"a": {"b": "value"}}
            result = nested.get("a").get("b")
        """)
        findings = parse_and_visit(source)

        # Both .get() calls should be detected
        r1_findings = [f for f in findings if f.rule_id == "R1"]
        assert len(r1_findings) == 2

    def test_get_in_function_context(self) -> None:
        """dict.get() in a function should include function in context."""
        source = dedent("""
            def process_data(data):
                return data.get("key")
        """)
        findings = parse_and_visit(source)

        assert len(findings) == 1
        assert findings[0].symbol_context == ("process_data",)

    def test_get_in_class_method_context(self) -> None:
        """dict.get() in a class method should include class and method in context."""
        source = dedent("""
            class DataProcessor:
                def process(self, data):
                    return data.get("key")
        """)
        findings = parse_and_visit(source)

        assert len(findings) == 1
        assert findings[0].symbol_context == ("DataProcessor", "process")


class TestR1FalsePositiveFiltering:
    """Tests for R1 false positive filtering.

    The R1 rule flags .get() calls, but many are false positives:
    - FastAPI router decorators (@router.get('/path'))
    - HTTP client methods (client.get('http://...'))
    - ChromaDB SDK calls (collection.get(ids=[...]))

    Tests 1-4 and 6 document filtering that SHOULD happen (TDD red phase).
    Tests 5, 7-10 are regression guards that must continue to pass.
    """

    def test_fastapi_router_decorator_not_flagged(self) -> None:
        """@router.get('/path') on sync def should NOT be flagged."""
        source = dedent("""
            from fastapi import APIRouter
            router = APIRouter()

            @router.get('/users')
            def list_users():
                return []
        """)
        findings = parse_and_visit(source)

        r1_findings = [f for f in findings if f.rule_id == "R1"]
        assert len(r1_findings) == 0, "FastAPI router.get() decorator should not be flagged"

    def test_async_router_decorator_not_flagged(self) -> None:
        """@router.get('/path') on async def should NOT be flagged."""
        source = dedent("""
            from fastapi import APIRouter
            router = APIRouter()

            @router.get('/items/{item_id}')
            async def get_item(item_id: int):
                return {"item_id": item_id}
        """)
        findings = parse_and_visit(source)

        r1_findings = [f for f in findings if f.rule_id == "R1"]
        assert len(r1_findings) == 0, "FastAPI router.get() decorator should not be flagged"

    def test_httpx_client_get_not_flagged(self) -> None:
        """client.get('http://...') should NOT be flagged."""
        source = dedent("""
            import httpx

            def fetch_data():
                client = httpx.Client()
                response = client.get('https://api.example.com/data')
                return response.json()
        """)
        findings = parse_and_visit(source)

        r1_findings = [f for f in findings if f.rule_id == "R1"]
        assert len(r1_findings) == 0, "HTTP client.get() with URL should not be flagged"

    def test_get_with_url_path_not_flagged(self) -> None:
        """client.get('/api/path') should NOT be flagged."""
        source = dedent("""
            def make_request(client):
                return client.get('/api/v1/users')
        """)
        findings = parse_and_visit(source)

        r1_findings = [f for f in findings if f.rule_id == "R1"]
        assert len(r1_findings) == 0, "client.get() with URL path should not be flagged"

    def test_fstring_url_still_flagged(self) -> None:
        """client.get(f"/api/{id}") SHOULD still be flagged (documents limitation).

        This test documents a known limitation: f-strings are not string literals
        in the AST, so we cannot determine if the argument is a URL. The heuristic
        errs on the side of flagging, requiring an allowlist entry if needed.
        """
        source = dedent("""
            def fetch_user(client, user_id):
                return client.get(f'/api/users/{user_id}')
        """)
        findings = parse_and_visit(source)

        r1_findings = [f for f in findings if f.rule_id == "R1"]
        assert len(r1_findings) == 1, "f-string URLs cannot be detected as URLs - must be flagged"

    def test_collection_get_with_ids_kwarg_not_flagged(self) -> None:
        """collection.get(ids=[...]) should NOT be flagged.

        ChromaDB and similar SDK patterns use .get(ids=[...]) which is clearly
        not a dict.get() call.
        """
        source = dedent("""
            def fetch_documents(collection):
                return collection.get(ids=['doc1', 'doc2', 'doc3'])
        """)
        findings = parse_and_visit(source)

        r1_findings = [f for f in findings if f.rule_id == "R1"]
        assert len(r1_findings) == 0, "collection.get(ids=[...]) should not be flagged"

    def test_limit_kwarg_still_flagged(self) -> None:
        """session.get(key, limit=10) SHOULD still be flagged (regression guard).

        The 'ids' kwarg is special-cased for ChromaDB SDK patterns. Other kwargs
        like 'limit' do not indicate non-dict usage and should still be flagged.
        """
        source = dedent("""
            def get_cached(session, key):
                return session.get(key, limit=10)
        """)
        findings = parse_and_visit(source)

        r1_findings = [f for f in findings if f.rule_id == "R1"]
        assert len(r1_findings) == 1, "get() with arbitrary kwargs should still be flagged"

    def test_real_dict_get_still_flagged(self) -> None:
        """data.get("key", "default") SHOULD be flagged."""
        source = dedent("""
            def process(data):
                return data.get("key", "default")
        """)
        findings = parse_and_visit(source)

        r1_findings = [f for f in findings if f.rule_id == "R1"]
        assert len(r1_findings) == 1, "Standard dict.get() must be flagged"

    def test_dict_get_without_default_still_flagged(self) -> None:
        """config.get("setting") SHOULD be flagged."""
        source = dedent("""
            def read_config(config):
                return config.get("setting")
        """)
        findings = parse_and_visit(source)

        r1_findings = [f for f in findings if f.rule_id == "R1"]
        assert len(r1_findings) == 1, "dict.get() without default must be flagged"

    def test_ambiguous_get_still_flagged(self) -> None:
        """obj.get("field") SHOULD be flagged.

        Ambiguous .get() calls with string arguments that don't look like URLs
        should be flagged. The allowlist mechanism exists for justified exceptions.
        """
        source = dedent("""
            def extract(obj):
                return obj.get("field")
        """)
        findings = parse_and_visit(source)

        r1_findings = [f for f in findings if f.rule_id == "R1"]
        assert len(r1_findings) == 1, "Ambiguous get() must be flagged"


# =============================================================================
# R2: getattr() detection
# =============================================================================


class TestR2Getattr:
    """Tests for R2: getattr() with default detection."""

    def test_detects_getattr_with_default(self) -> None:
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

    def test_ignores_getattr_without_default(self) -> None:
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

    def test_detects_getattr_with_keyword_default(self) -> None:
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

    def test_detects_hasattr(self) -> None:
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

    def test_hasattr_in_condition(self) -> None:
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

    def test_detects_bare_except(self) -> None:
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

    def test_detects_except_exception(self) -> None:
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

    def test_detects_except_exception_as_e(self) -> None:
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

    def test_ignores_except_with_reraise(self) -> None:
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

    def test_ignores_except_with_raise_new(self) -> None:
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

    def test_ignores_specific_exceptions(self) -> None:
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

    def test_detects_except_base_exception(self) -> None:
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

    def test_canonical_key_module_level(self) -> None:
        """Module-level finding should have _module_ in key."""
        finding = Finding(
            rule_id="R1",
            file_path="src/module.py",
            line=10,
            col=0,
            symbol_context=(),
            fingerprint="deadbeefcafebabe",
            code_snippet="data.get('key')",
            message="test",
        )

        assert finding.canonical_key == "src/module.py:R1:_module_:fp=deadbeefcafebabe"

    def test_canonical_key_function(self) -> None:
        """Function-level finding should include function name."""
        finding = Finding(
            rule_id="R2",
            file_path="src/module.py",
            line=25,
            col=4,
            symbol_context=("process_data",),
            fingerprint="0123456789abcdef",
            code_snippet="getattr(obj, 'x', None)",
            message="test",
        )

        assert finding.canonical_key == "src/module.py:R2:process_data:fp=0123456789abcdef"

    def test_canonical_key_class_method(self) -> None:
        """Class method finding should include class and method."""
        finding = Finding(
            rule_id="R3",
            file_path="src/handler.py",
            line=42,
            col=8,
            symbol_context=("Handler", "process"),
            fingerprint="feedfacecafed00d",
            code_snippet="hasattr(obj, 'attr')",
            message="test",
        )

        assert finding.canonical_key == "src/handler.py:R3:Handler:process:fp=feedfacecafed00d"


# =============================================================================
# Allowlist matching
# =============================================================================


class TestAllowlistMatching:
    """Tests for allowlist entry matching."""

    def test_exact_match(self) -> None:
        """Allowlist entry should match finding with exact key."""
        entry = AllowlistEntry(
            key="src/module.py:R1:process:fp=deadbeefcafebabe",
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
            fingerprint="deadbeefcafebabe",
            code_snippet="data.get('key')",
            message="test",
        )

        matched = allowlist.match(finding)
        assert matched is not None
        assert isinstance(matched, AllowlistEntry)
        assert matched.key == entry.key
        assert entry.matched is True

    def test_no_match(self) -> None:
        """Finding without matching allowlist entry should return None."""
        entry = AllowlistEntry(
            key="src/other.py:R1:process:fp=deadbeefcafebabe",
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
            fingerprint="deadbeefcafebabe",
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

    def test_unmatched_entry_is_stale(self) -> None:
        """Entry that doesn't match any finding should be stale."""
        entry = AllowlistEntry(
            key="src/removed.py:R1:old_function:fp=deadbeefcafebabe",
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

    def test_matched_entry_not_stale(self) -> None:
        """Entry that matched a finding should not be stale."""
        entry = AllowlistEntry(
            key="src/module.py:R1:process:fp=deadbeefcafebabe",
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

    def test_expired_entry_detected(self) -> None:
        """Entry with past expiry date should be detected."""
        yesterday = datetime.now(UTC).date() - timedelta(days=1)
        entry = AllowlistEntry(
            key="src/module.py:R1:process:fp=deadbeefcafebabe",
            owner="test",
            reason="test",
            safety="test",
            expires=yesterday,
        )
        allowlist = Allowlist(entries=[entry])

        expired = allowlist.get_expired_entries()
        assert len(expired) == 1
        assert expired[0].key == entry.key

    def test_future_entry_not_expired(self) -> None:
        """Entry with future expiry date should not be detected."""
        tomorrow = datetime.now(UTC).date() + timedelta(days=1)
        entry = AllowlistEntry(
            key="src/module.py:R1:process:fp=deadbeefcafebabe",
            owner="test",
            reason="test",
            safety="test",
            expires=tomorrow,
        )
        allowlist = Allowlist(entries=[entry])

        expired = allowlist.get_expired_entries()
        assert len(expired) == 0

    def test_no_expiry_not_expired(self) -> None:
        """Entry without expiry date should not be detected."""
        entry = AllowlistEntry(
            key="src/module.py:R1:process:fp=deadbeefcafebabe",
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

    def test_load_empty_file(self, temp_dir: Path) -> None:
        """Empty allowlist file should produce empty allowlist."""
        allowlist_path = temp_dir / "allowlist.yaml"
        allowlist_path.write_text("")

        allowlist = load_allowlist(allowlist_path)
        assert len(allowlist.entries) == 0

    def test_load_with_entries(self, temp_dir: Path) -> None:
        """Allowlist with entries should be parsed correctly."""
        allowlist_path = temp_dir / "allowlist.yaml"
        allowlist_path.write_text("""
version: 1
defaults:
  fail_on_stale: true
  fail_on_expired: false
allow_hits:
  - key: "src/module.py:R1:process:fp=deadbeefcafebabe"
    owner: "john"
    reason: "Legacy code"
    safety: "Will be refactored"
    expires: "2026-06-01"
""")

        allowlist = load_allowlist(allowlist_path)
        assert len(allowlist.entries) == 1
        assert allowlist.entries[0].owner == "john"
        assert allowlist.entries[0].expires == datetime(2026, 6, 1, tzinfo=UTC).date()
        assert allowlist.fail_on_stale is True
        assert allowlist.fail_on_expired is False

    def test_load_nonexistent_file(self, temp_dir: Path) -> None:
        """Missing allowlist file should produce empty allowlist."""
        allowlist_path = temp_dir / "missing.yaml"

        allowlist = load_allowlist(allowlist_path)
        assert len(allowlist.entries) == 0


# =============================================================================
# File scanning
# =============================================================================


class TestFileScanning:
    """Tests for scanning Python files."""

    def test_scan_file_with_violations(self, temp_dir: Path) -> None:
        """File with violations should produce findings."""
        py_file = temp_dir / "test_module.py"
        py_file.write_text(
            dedent("""
            def process(data):
                return data.get("key", None)
        """)
        )

        findings = scan_file(py_file, temp_dir)
        assert len(findings) == 1
        assert findings[0].rule_id == "R1"
        assert findings[0].file_path == "test_module.py"

    def test_scan_file_no_violations(self, temp_dir: Path) -> None:
        """Clean file should produce no findings."""
        py_file = temp_dir / "clean_module.py"
        py_file.write_text(
            dedent("""
            def process(data):
                return data["key"]
        """)
        )

        findings = scan_file(py_file, temp_dir)
        assert len(findings) == 0

    def test_scan_file_syntax_error(self, temp_dir: Path) -> None:
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

    def test_finding_allowlisted_and_stale_detection(self, temp_dir: Path) -> None:
        """Full workflow: findings, allowlisting, and stale detection."""
        # Create a file with one violation
        py_file = temp_dir / "module.py"
        py_file.write_text(
            dedent("""
            def process(data):
                return data.get("key")
        """)
        )

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
            key="module.py:R1:old_function:fp=deadbeefcafebabe",
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


# =============================================================================
# Directory loading
# =============================================================================


class TestDirectoryLoading:
    """Tests for loading allowlist from a directory of per-module YAML files."""

    def test_load_directory_merges_entries(self, temp_dir: Path) -> None:
        """Directory with defaults + module files should merge into single Allowlist."""
        allowlist_dir = temp_dir / "allowlist"
        allowlist_dir.mkdir()

        (allowlist_dir / "_defaults.yaml").write_text("version: 1\ndefaults:\n  fail_on_stale: true\n  fail_on_expired: false\n")
        (allowlist_dir / "core.yaml").write_text(
            dedent("""\
            per_file_rules:
              - pattern: core/config.py
                rules: [R1, R5]
                reason: Config parsing
                expires: null
            allow_hits:
              - key: "core/events.py:R1:EventBus:emit:fp=aaa"
                owner: test
                reason: test
                safety: test
            """)
        )
        (allowlist_dir / "plugins.yaml").write_text(
            dedent("""\
            allow_hits:
              - key: "plugins/sinks/csv_sink.py:R1:CSVSink:open:fp=bbb"
                owner: test
                reason: test
                safety: test
              - key: "plugins/sinks/json_sink.py:R1:JSONSink:open:fp=ccc"
                owner: test
                reason: test
                safety: test
            """)
        )

        allowlist = load_allowlist(allowlist_dir)
        assert allowlist.fail_on_stale is True
        assert allowlist.fail_on_expired is False
        assert len(allowlist.entries) == 3
        assert len(allowlist.per_file_rules) == 1

    def test_load_directory_sorted_order(self, temp_dir: Path) -> None:
        """Entries should merge in sorted filename order."""
        allowlist_dir = temp_dir / "allowlist"
        allowlist_dir.mkdir()

        (allowlist_dir / "_defaults.yaml").write_text("version: 1\ndefaults: {}\n")
        (allowlist_dir / "b_module.yaml").write_text(
            dedent("""\
            allow_hits:
              - key: "b/file.py:R1:func:fp=bbb"
                owner: test
                reason: from b
                safety: test
            """)
        )
        (allowlist_dir / "a_module.yaml").write_text(
            dedent("""\
            allow_hits:
              - key: "a/file.py:R1:func:fp=aaa"
                owner: test
                reason: from a
                safety: test
            """)
        )

        allowlist = load_allowlist(allowlist_dir)
        # a_module.yaml sorts before b_module.yaml
        assert allowlist.entries[0].reason == "from a"
        assert allowlist.entries[1].reason == "from b"

    def test_load_directory_empty(self, temp_dir: Path) -> None:
        """Empty directory should give empty allowlist with defaults."""
        allowlist_dir = temp_dir / "allowlist"
        allowlist_dir.mkdir()

        allowlist = load_allowlist(allowlist_dir)
        assert len(allowlist.entries) == 0
        assert len(allowlist.per_file_rules) == 0
        assert allowlist.fail_on_stale is True
        assert allowlist.fail_on_expired is True

    def test_load_directory_no_defaults(self, temp_dir: Path) -> None:
        """Missing _defaults.yaml should use hardcoded defaults."""
        allowlist_dir = temp_dir / "allowlist"
        allowlist_dir.mkdir()

        (allowlist_dir / "core.yaml").write_text(
            dedent("""\
            allow_hits:
              - key: "core/events.py:R1:EventBus:emit:fp=aaa"
                owner: test
                reason: test
                safety: test
            """)
        )

        allowlist = load_allowlist(allowlist_dir)
        assert len(allowlist.entries) == 1
        # Defaults: fail_on_stale=True, fail_on_expired=True
        assert allowlist.fail_on_stale is True
        assert allowlist.fail_on_expired is True

    def test_load_file_backward_compat(self, temp_dir: Path) -> None:
        """Single file path should still work (backward compatibility)."""
        allowlist_path = temp_dir / "allowlist.yaml"
        allowlist_path.write_text(
            dedent("""\
            version: 1
            defaults:
              fail_on_stale: true
              fail_on_expired: true
            allow_hits:
              - key: "src/module.py:R1:process:fp=deadbeef"
                owner: john
                reason: test
                safety: test
                expires: "2026-12-01"
            """)
        )

        allowlist = load_allowlist(allowlist_path)
        assert len(allowlist.entries) == 1
        assert allowlist.entries[0].owner == "john"

    def test_stale_detection_across_files(self, temp_dir: Path) -> None:
        """Stale entries should be detected in merged allowlist from directory."""
        allowlist_dir = temp_dir / "allowlist"
        allowlist_dir.mkdir()

        (allowlist_dir / "_defaults.yaml").write_text("version: 1\ndefaults: {}\n")
        (allowlist_dir / "core.yaml").write_text(
            dedent("""\
            allow_hits:
              - key: "core/events.py:R1:EventBus:emit:fp=aaa"
                owner: test
                reason: stale entry
                safety: test
            """)
        )
        (allowlist_dir / "plugins.yaml").write_text(
            dedent("""\
            allow_hits:
              - key: "plugins/sinks/csv_sink.py:R1:CSVSink:open:fp=bbb"
                owner: test
                reason: also stale
                safety: test
            """)
        )

        allowlist = load_allowlist(allowlist_dir)
        # No findings matched — all entries are stale
        stale = allowlist.get_stale_entries()
        assert len(stale) == 2

    def test_source_file_tracking(self, temp_dir: Path) -> None:
        """Entries should carry their source filename."""
        allowlist_dir = temp_dir / "allowlist"
        allowlist_dir.mkdir()

        (allowlist_dir / "_defaults.yaml").write_text("version: 1\ndefaults: {}\n")
        (allowlist_dir / "core.yaml").write_text(
            dedent("""\
            per_file_rules:
              - pattern: core/config.py
                rules: [R1]
                reason: test
                expires: null
            allow_hits:
              - key: "core/events.py:R1:EventBus:emit:fp=aaa"
                owner: test
                reason: test
                safety: test
            """)
        )

        allowlist = load_allowlist(allowlist_dir)
        assert allowlist.entries[0].source_file == "core.yaml"
        assert allowlist.per_file_rules[0].source_file == "core.yaml"

    def test_format_stale_entry_with_source(self) -> None:
        """Stale entry formatting should include source file when set."""
        entry = AllowlistEntry(
            key="core/events.py:R1:emit:fp=aaa",
            owner="test",
            reason="test reason",
            safety="test",
            expires=None,
            source_file="core.yaml",
        )
        text = format_stale_entry_text(entry)
        assert "Source: core.yaml" in text
        assert "Key: core/events.py:R1:emit:fp=aaa" in text

    def test_format_stale_entry_without_source(self) -> None:
        """Stale entry formatting should omit source when empty."""
        entry = AllowlistEntry(
            key="core/events.py:R1:emit:fp=aaa",
            owner="test",
            reason="test reason",
            safety="test",
            expires=None,
        )
        text = format_stale_entry_text(entry)
        assert "Source:" not in text

    def test_suggest_module_file_directory(self, temp_dir: Path) -> None:
        """_suggest_module_file should map findings to module YAML files."""
        allowlist_dir = temp_dir / "allowlist"
        allowlist_dir.mkdir()

        finding = Finding(
            rule_id="R1",
            file_path="core/events.py",
            line=10,
            col=0,
            symbol_context=("EventBus", "emit"),
            fingerprint="aaa",
            code_snippet="data.get('key')",
            message="test",
        )
        result = _suggest_module_file(finding, allowlist_dir)
        assert result.endswith("core.yaml")

    def test_suggest_module_file_cli(self, temp_dir: Path) -> None:
        """Bare cli.py should map to cli.yaml."""
        allowlist_dir = temp_dir / "allowlist"
        allowlist_dir.mkdir()

        finding = Finding(
            rule_id="R1",
            file_path="cli.py",
            line=10,
            col=0,
            symbol_context=(),
            fingerprint="aaa",
            code_snippet="data.get('key')",
            message="test",
        )
        result = _suggest_module_file(finding, allowlist_dir)
        assert result.endswith("cli.yaml")


# =============================================================================
# Per-file rule max_hits
# =============================================================================


class TestPerFileRuleMaxHits:
    """Tests for max_hits cap on per-file rules."""

    def _make_finding(self, file_path: str, rule_id: str = "R5") -> Finding:
        """Create a minimal Finding for testing."""
        return Finding(
            rule_id=rule_id,
            file_path=file_path,
            line=10,
            col=0,
            symbol_context=("SomeClass", "method"),
            fingerprint="deadbeef",
            code_snippet="isinstance(x, int)",
            message="test",
        )

    def test_max_hits_none_allows_unlimited(self) -> None:
        """Per-file rule with no max_hits should allow any number of matches."""
        rule = PerFileRule(
            pattern="core/canonical.py",
            rules=["R5"],
            reason="Type dispatch for normalization",
            expires=None,
            max_hits=None,
        )
        allowlist = Allowlist(entries=[], per_file_rules=[rule])

        for _ in range(50):
            allowlist.match(self._make_finding("core/canonical.py"))

        assert rule.matched_count == 50
        assert allowlist.get_exceeded_file_rules() == []

    def test_max_hits_within_limit(self) -> None:
        """Per-file rule with matched_count <= max_hits should not be exceeded."""
        rule = PerFileRule(
            pattern="core/canonical.py",
            rules=["R5"],
            reason="Type dispatch",
            expires=None,
            max_hits=18,
        )
        allowlist = Allowlist(entries=[], per_file_rules=[rule])

        for _ in range(18):
            allowlist.match(self._make_finding("core/canonical.py"))

        assert rule.matched_count == 18
        assert allowlist.get_exceeded_file_rules() == []

    def test_max_hits_exceeded(self) -> None:
        """Per-file rule exceeding max_hits should be reported."""
        rule = PerFileRule(
            pattern="core/canonical.py",
            rules=["R5"],
            reason="Type dispatch",
            expires=None,
            max_hits=5,
        )
        allowlist = Allowlist(entries=[], per_file_rules=[rule])

        for _ in range(8):
            allowlist.match(self._make_finding("core/canonical.py"))

        assert rule.matched_count == 8
        exceeded = allowlist.get_exceeded_file_rules()
        assert len(exceeded) == 1
        assert exceeded[0] is rule

    def test_max_hits_only_counts_matching_rule(self) -> None:
        """max_hits should only count hits for the matching rule, not other rules."""
        rule = PerFileRule(
            pattern="core/canonical.py",
            rules=["R5"],
            reason="Type dispatch",
            expires=None,
            max_hits=2,
        )
        allowlist = Allowlist(entries=[], per_file_rules=[rule])

        # R5 matches the rule
        allowlist.match(self._make_finding("core/canonical.py", rule_id="R5"))
        allowlist.match(self._make_finding("core/canonical.py", rule_id="R5"))
        # R1 does NOT match this rule (rules=["R5"])
        result = allowlist.match(self._make_finding("core/canonical.py", rule_id="R1"))
        assert result is None  # R1 not in rule's rules list

        assert rule.matched_count == 2
        assert allowlist.get_exceeded_file_rules() == []

    def test_max_hits_parsed_from_yaml(self, temp_dir: Path) -> None:
        """max_hits should be parsed from YAML per_file_rules."""
        allowlist_dir = temp_dir / "allowlist"
        allowlist_dir.mkdir()

        (allowlist_dir / "_defaults.yaml").write_text("version: 1\ndefaults: {}\n")
        (allowlist_dir / "core.yaml").write_text(
            dedent("""\
            per_file_rules:
              - pattern: core/canonical.py
                rules: [R5]
                reason: Type dispatch for normalization
                expires: null
                max_hits: 18
            """)
        )

        allowlist = load_allowlist(allowlist_dir)
        assert len(allowlist.per_file_rules) == 1
        assert allowlist.per_file_rules[0].max_hits == 18

    def test_max_hits_defaults_to_none(self, temp_dir: Path) -> None:
        """Omitting max_hits in YAML should default to None (unlimited)."""
        allowlist_dir = temp_dir / "allowlist"
        allowlist_dir.mkdir()

        (allowlist_dir / "_defaults.yaml").write_text("version: 1\ndefaults: {}\n")
        (allowlist_dir / "core.yaml").write_text(
            dedent("""\
            per_file_rules:
              - pattern: core/canonical.py
                rules: [R5]
                reason: Type dispatch
                expires: null
            """)
        )

        allowlist = load_allowlist(allowlist_dir)
        assert allowlist.per_file_rules[0].max_hits is None


class TestDirectoryLoadingSuggestModuleFile:
    """Tests for _suggest_module_file and related directory loading."""

    def test_suggest_module_file_single_file(self, temp_dir: Path) -> None:
        """Single file path should return the file path as-is."""
        allowlist_path = temp_dir / "allowlist.yaml"
        allowlist_path.write_text("")

        finding = Finding(
            rule_id="R1",
            file_path="core/events.py",
            line=10,
            col=0,
            symbol_context=(),
            fingerprint="aaa",
            code_snippet="data.get('key')",
            message="test",
        )
        result = _suggest_module_file(finding, allowlist_path)
        assert result == str(allowlist_path)


# =============================================================================
# Bug fix tests: enforce_tier_model.py bug cluster
# =============================================================================


class TestBannedRuleKeyValidation:
    """Tests for elspeth-9f34362456: banned-rule key-format check validates rule ID."""

    def test_valid_banned_rule_in_allow_hits_rejected(self, capsys: pytest.CaptureFixture[str]) -> None:
        """allow_hits entry with banned rule R3 should be rejected."""
        data = {
            "allow_hits": [
                {
                    "key": "core/events.py:R3:SomeClass:fp=abc123",
                    "owner": "test",
                    "reason": "test",
                    "safety": "test",
                    "expires": "2099-01-01",
                }
            ]
        }
        with pytest.raises(SystemExit) as exc_info:
            _parse_allow_hits(data)
        assert exc_info.value.code == 1
        assert "banned rule R3" in capsys.readouterr().err

    def test_invalid_rule_id_in_key_rejected(self, capsys: pytest.CaptureFixture[str]) -> None:
        """allow_hits entry with invalid (non-existent) rule ID should be rejected.

        Previously, a malformed key like 'foo.py:GARBAGE:bar:fp=abc' would silently
        pass because 'GARBAGE' is not in _BANNED_RULES. The validation should also
        verify that the rule ID is valid.
        """
        data = {
            "allow_hits": [
                {
                    "key": "core/events.py:NONEXISTENT_RULE:SomeClass:fp=abc123",
                    "owner": "test",
                    "reason": "test",
                    "safety": "test",
                    "expires": "2099-01-01",
                }
            ]
        }
        with pytest.raises(SystemExit) as exc_info:
            _parse_allow_hits(data)
        assert exc_info.value.code == 1
        assert "unknown rule ID" in capsys.readouterr().err
        # Distinct from banned-rule rejection — error names the offending ID

    def test_malformed_key_missing_rule_id_rejected(self, capsys: pytest.CaptureFixture[str]) -> None:
        """allow_hits entry with no colon (no rule ID extractable) should be rejected."""
        data = {
            "allow_hits": [
                {
                    "key": "bare-key-no-colons",
                    "owner": "test",
                    "reason": "test",
                    "safety": "test",
                    "expires": "2099-01-01",
                }
            ]
        }
        with pytest.raises(SystemExit) as exc_info:
            _parse_allow_hits(data)
        assert exc_info.value.code == 1
        assert "malformed key" in capsys.readouterr().err

    def test_valid_rule_id_in_key_accepted(self) -> None:
        """allow_hits entry with valid non-banned rule ID should be accepted."""
        data = {
            "allow_hits": [
                {
                    "key": "core/events.py:R1:SomeClass:fp=abc123",
                    "owner": "test",
                    "reason": "test",
                    "safety": "test",
                    "expires": "2099-01-01",
                }
            ]
        }
        entries = _parse_allow_hits(data)
        assert len(entries) == 1
        assert entries[0].key == "core/events.py:R1:SomeClass:fp=abc123"


class TestMaxHitsParseError:
    """Tests for elspeth-cdeeeccde3: int(raw_max_hits) should give contextual error."""

    def test_non_numeric_max_hits_gives_context(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Non-numeric max_hits should produce an error message with pattern context.

        Previously, int('five') raised a bare ValueError with no indication of
        which per_file_rules entry or YAML file contained the error.
        """
        data = {
            "per_file_rules": [
                {
                    "pattern": "plugins/sources/*",
                    "rules": ["R1"],
                    "reason": "test",
                    "max_hits": "five",
                }
            ]
        }
        with pytest.raises(SystemExit) as exc_info:
            _parse_per_file_rules(data, source_file="plugins.yaml")
        assert exc_info.value.code == 1
        err = capsys.readouterr().err
        assert "plugins/sources/*" in err
        assert "plugins.yaml" in err
        assert "'five'" in err

    def test_numeric_string_max_hits_parses(self) -> None:
        """Numeric string max_hits like '18' should still parse correctly."""
        data = {
            "per_file_rules": [
                {
                    "pattern": "plugins/sources/*",
                    "rules": ["R1"],
                    "reason": "test",
                    "max_hits": "18",
                }
            ]
        }
        rules = _parse_per_file_rules(data)
        assert rules[0].max_hits == 18


class TestPerFileRulesUnknownRuleValidation:
    """Tests for symmetric unknown-rule-ID validation in per_file_rules."""

    def test_unknown_rule_id_in_per_file_rules_rejected(self, capsys: pytest.CaptureFixture[str]) -> None:
        """per_file_rules with unknown rule ID should be rejected at parse time."""
        data = {
            "per_file_rules": [
                {
                    "pattern": "plugins/sources/*",
                    "rules": ["R1", "TYPO_RULE"],
                    "reason": "test",
                }
            ]
        }
        with pytest.raises(SystemExit) as exc_info:
            _parse_per_file_rules(data, source_file="plugins.yaml")
        assert exc_info.value.code == 1
        err = capsys.readouterr().err
        assert "unknown rule ID" in err
        assert "TYPO_RULE" in err

    def test_valid_rule_ids_in_per_file_rules_accepted(self) -> None:
        """per_file_rules with all valid non-banned rule IDs should be accepted."""
        data = {
            "per_file_rules": [
                {
                    "pattern": "plugins/sources/*",
                    "rules": ["R1", "R4", "R5"],
                    "reason": "test",
                }
            ]
        }
        rules = _parse_per_file_rules(data)
        assert len(rules) == 1
        assert rules[0].rules == ["R1", "R4", "R5"]


class TestExceededFileRulesPreCommitMode:
    """Tests for elspeth-d224bb2575: exceeded_file_rules in pre-commit mode."""

    @staticmethod
    def _make_finding(file_path: str = "core/canonical.py", rule_id: str = "R5") -> Finding:
        return Finding(
            rule_id=rule_id,
            file_path=file_path,
            line=10,
            col=0,
            symbol_context=("SomeClass", "method"),
            fingerprint="deadbeef",
            code_snippet="isinstance(x, int)",
            message="test",
        )

    def test_exceeded_file_rules_suppressed_in_precommit_mode(self) -> None:
        """In pre-commit mode (args.files is set), exceeded_file_rules should be
        suppressed because the partial scan produces non-deterministic match counts.

        Previously, stale/expired/unused checks were correctly suppressed in pre-commit
        mode, but exceeded_file_rules was always checked — asymmetric behavior.
        """
        rule = PerFileRule(
            pattern="core/canonical.py",
            rules=["R5"],
            reason="Type dispatch",
            expires=None,
            max_hits=2,
        )
        allowlist = Allowlist(entries=[], per_file_rules=[rule])

        # Simulate 5 matches — exceeds max_hits=2
        for _ in range(5):
            allowlist.match(self._make_finding())

        # In a full scan, this would be exceeded
        assert allowlist.get_exceeded_file_rules() == [rule]

        # But get_exceeded_file_rules should NOT contribute to failure
        # when we're in pre-commit mode. The fix should suppress this
        # at the call site in run_check(), same as stale/expired/unused.
        # This test documents the data model behavior.

    def test_run_check_precommit_ignores_exceeded_max_hits(self, temp_dir: Path) -> None:
        """run_check in pre-commit mode (with files arg) must not fail on exceeded max_hits.

        This exercises the actual code path in run_check() where pre-commit mode
        suppresses exceeded_file_rules, verifying the fix at the call site.
        """
        # Create a Python file with enough isinstance() calls to exceed max_hits=1
        src_dir = temp_dir / "src"
        src_dir.mkdir()
        py_file = src_dir / "example.py"
        py_file.write_text(
            "def f(x):\n    if isinstance(x, int): pass\n    if isinstance(x, str): pass\n    if isinstance(x, float): pass\n"
        )

        # Create allowlist with max_hits=1 for R5 (isinstance) — will be exceeded
        allowlist_dir = temp_dir / "allowlist"
        allowlist_dir.mkdir()
        (allowlist_dir / "_defaults.yaml").write_text("version: 1\ndefaults: {}\n")
        (allowlist_dir / "test.yaml").write_text(
            "per_file_rules:\n  - pattern: example.py\n    rules: [R5]\n    reason: test\n    expires: null\n    max_hits: 1\n"
        )

        # Build args simulating pre-commit mode (with specific files)
        args = argparse.Namespace(
            root=src_dir,
            allowlist=allowlist_dir,
            exclude=[],
            format="text",
            files=[py_file],
        )

        # In pre-commit mode, exceeded max_hits should NOT cause failure
        result = run_check(args)
        assert result == 0, "pre-commit mode should suppress exceeded_file_rules"
