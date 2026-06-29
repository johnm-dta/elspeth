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
import hashlib
import json
import sys
import tempfile
from collections.abc import Generator
from datetime import UTC, datetime, timedelta
from pathlib import Path
from textwrap import dedent

import pytest

from elspeth_lints.core.allowlist import JudgeVerdict
from elspeth_lints.core.judge import DEFAULT_JUDGE_MODEL, JUDGE_POLICY_HASH
from elspeth_lints.rules.trust_tier.tier_model.rule import (
    Allowlist,
    AllowlistBudgetViolation,
    AllowlistEntry,
    Finding,
    PerFileRule,
    TierModelVisitor,
    _match_finding,
    _suggest_module_file,
    _validate_allowlist_governance,
    format_stale_entry_text,
    report_json,
    run_check,
    scan_file,
    scan_layer_imports_file,
)
from elspeth_lints.rules.trust_tier.tier_model.rule import (
    _load_tier_model_allowlist as load_allowlist,
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
    file_fingerprint = hashlib.sha256(source.encode("utf-8")).hexdigest()
    visitor = TierModelVisitor(filename, source_lines, file_fingerprint)
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

    def test_unknown_receiver_fstring_url_still_flagged(self) -> None:
        """client.get(f"/api/{id}") SHOULD still be flagged for unknown receivers.

        F-string URLs are safe to suppress only when the receiver is known to be
        an HTTP client. Unknown receivers may still be dict-like objects.
        """
        source = dedent("""
            def fetch_user(client, user_id):
                return client.get(f'/api/users/{user_id}')
        """)
        findings = parse_and_visit(source)

        r1_findings = [f for f in findings if f.rule_id == "R1"]
        assert len(r1_findings) == 1, "f-string URLs on unknown receivers must be flagged"

    def test_httpx_async_client_get_with_variable_url_not_flagged(self) -> None:
        """httpx.AsyncClient.get(variable_url) is HTTP transport, not dict access."""
        source = dedent("""
            import httpx

            async def fetch_jwks(issuer):
                async with httpx.AsyncClient(timeout=10.0) as client:
                    discovery_url = f"{issuer}/.well-known/openid-configuration"
                    response = await client.get(discovery_url)
                    return response.json()
        """)
        findings = parse_and_visit(source)

        r1_findings = [f for f in findings if f.rule_id == "R1"]
        assert len(r1_findings) == 0, "httpx.AsyncClient.get() should not be flagged"

    def test_starlette_headers_get_not_flagged(self) -> None:
        """starlette Headers.get() is the multidict Mapping accessor, not dict.get()."""
        source = dedent("""
            from starlette.datastructures import Headers

            async def __call__(self, scope, receive, send):
                headers = Headers(scope=scope)
                supplied = headers.get("x-request-id")
                return supplied
        """)
        findings = parse_and_visit(source)

        r1_findings = [f for f in findings if f.rule_id == "R1"]
        assert len(r1_findings) == 0, "starlette Headers.get() should not be flagged as dict.get()"

    def test_httpx_module_get_with_variable_url_not_flagged(self) -> None:
        """httpx.get(variable_url) is module-level HTTP transport, not dict access."""
        source = dedent("""
            import httpx

            def check_readiness(count_url):
                response = httpx.get(count_url, timeout=10.0)
                return response.status_code
        """)
        findings = parse_and_visit(source)

        r1_findings = [f for f in findings if f.rule_id == "R1"]
        assert len(r1_findings) == 0, "httpx.get() should not be flagged"

    def test_asyncio_queue_get_not_flagged(self) -> None:
        """asyncio.Queue.get() is an awaitable queue API, not dict access."""
        source = dedent("""
            import asyncio

            async def wait_for_event():
                queue = asyncio.Queue()
                event = await queue.get()
                return event
        """)
        findings = parse_and_visit(source)

        r1_findings = [f for f in findings if f.rule_id == "R1"]
        assert len(r1_findings) == 0, "asyncio.Queue.get() should not be flagged"

    def test_httpx_client_instance_attribute_get_not_flagged(self) -> None:
        """self._client.get() is HTTP transport when __init__ constructs httpx.Client."""
        source = dedent("""
            import httpx

            class AuditedHTTPClient:
                def __init__(self):
                    self._client = httpx.Client()

                def fetch(self, url):
                    return self._client.get(url, timeout=10.0)
        """)
        findings = parse_and_visit(source)

        r1_findings = [f for f in findings if f.rule_id == "R1"]
        assert len(r1_findings) == 0, "self._client.get() backed by httpx.Client should not be flagged"

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


class TestR1SourceRegressions:
    """Source-level regressions for known R1 allowlist burn-downs."""

    def test_landscape_exporter_iter_records_has_no_r1_findings(self) -> None:
        findings = scan_file(
            Path("src/elspeth/core/landscape/exporter.py"),
            Path("src/elspeth"),
        )

        r1_findings = [
            finding for finding in findings if finding.rule_id == "R1" and finding.symbol_context == ("LandscapeExporter", "_iter_records")
        ]
        assert r1_findings == []

    def test_sink_display_mapping_call_sites_have_no_r1_findings(self) -> None:
        findings = [
            *scan_file(
                Path("src/elspeth/plugins/sinks/csv_sink.py"),
                Path("src/elspeth"),
            ),
            *scan_file(
                Path("src/elspeth/plugins/sinks/json_sink.py"),
                Path("src/elspeth"),
            ),
        ]

        target_contexts = {
            ("CSVSink", "validate_output_target"),
            ("CSVSink", "_open_file"),
            ("CSVSink", "_get_field_names_and_display"),
            ("JSONSink", "validate_output_target"),
        }
        r1_findings = [finding for finding in findings if finding.rule_id == "R1" and finding.symbol_context in target_contexts]
        assert r1_findings == []

    def test_session_fork_blob_rewrite_call_sites_have_no_r1_findings(self) -> None:
        findings = scan_file(
            Path("src/elspeth/web/sessions/routes.py"),
            Path("src/elspeth"),
        )

        r1_findings = [
            finding
            for finding in findings
            if finding.rule_id == "R1" and finding.symbol_context == ("create_session_router", "fork_from_message")
        ]
        assert r1_findings == []

    @pytest.mark.skipif(
        sys.version_info[:2] != (3, 13),
        reason="tier-model allowlist fingerprints are version-specific; Python 3.13 is canonical",
    )
    def test_source_boundary_non_r5_findings_are_site_allowlisted(self) -> None:
        allowlist = load_allowlist(Path("config/cicd/enforce_tier_model"))

        blanket_source_rules = [
            rule
            for rule in allowlist.per_file_rules
            if rule.pattern == "plugins/sources/*" and {"R1", "R2", "R4", "R6", "R9"} & set(rule.rules)
        ]
        assert blanket_source_rules == []

        source_files = [
            Path("src/elspeth/plugins/sources/azure_blob_source.py"),
            Path("src/elspeth/plugins/sources/csv_source.py"),
            Path("src/elspeth/plugins/sources/dataverse.py"),
            Path("src/elspeth/plugins/sources/json_source.py"),
            Path("src/elspeth/plugins/sources/text_source.py"),
        ]
        findings = [
            finding
            for source_file in source_files
            for finding in scan_file(source_file, Path("src/elspeth"))
            if finding.rule_id in {"R1", "R2", "R4", "R6", "R9"}
        ]
        allowed_keys = {entry.key for entry in allowlist.entries}

        missing_keys = [finding.canonical_key for finding in findings if finding.canonical_key not in allowed_keys]
        assert missing_keys == []

    def test_grouping_setdefault_call_sites_have_no_r8_findings(self) -> None:
        findings = [
            *scan_file(
                Path("src/elspeth/plugins/transforms/field_mapper.py"),
                Path("src/elspeth"),
            ),
            *scan_file(
                Path("src/elspeth/plugins/sources/field_normalization.py"),
                Path("src/elspeth"),
            ),
        ]

        target_contexts = {
            ("FieldMapperConfig", "_reject_duplicate_targets"),
            ("check_normalization_collisions",),
            ("check_mapping_collisions",),
        }
        r8_findings = [finding for finding in findings if finding.rule_id == "R8" and finding.symbol_context in target_contexts]
        assert r8_findings == []


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

    def test_nested_helper_raise_does_not_satisfy_outer_handler(self) -> None:
        """A raise inside a nested helper is not a re-raise by the handler."""
        source = dedent("""
            try:
                risky_operation()
            except Exception:
                def helper():
                    raise RuntimeError("not the handler")
                record_problem()
        """)
        findings = parse_and_visit(source)

        r4_findings = [f for f in findings if f.rule_id == "R4"]
        assert len(r4_findings) == 1

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
# R6: Silent specific exception handling
# =============================================================================


class TestR6SilentExcept:
    """Specific exception handlers must be judged in their own lexical scope."""

    def test_nested_raise_does_not_make_handler_non_silent(self) -> None:
        source = dedent("""
            try:
                int("not a number")
            except ValueError:
                def helper():
                    raise RuntimeError("not the handler")
                helper
        """)
        findings = parse_and_visit(source)

        r6_findings = [f for f in findings if f.rule_id == "R6"]
        assert len(r6_findings) == 1

    def test_transform_result_error_routed_to_completion_is_explicit(self) -> None:
        source = dedent("""
            class Worker:
                def accept_row(self):
                    try:
                        self._batch_executor.submit(self._process)
                    except RuntimeError:
                        from elspeth.contracts import TransformResult

                        shutdown_result = TransformResult.error(
                            {"reason": "shutdown_requested"},
                            retryable=False,
                        )
                        self._complete_ticket(ticket, token, shutdown_result, state_id)
        """)
        findings = parse_and_visit(source)

        r6_findings = [f for f in findings if f.rule_id == "R6"]
        assert r6_findings == []

    def test_transform_result_error_without_completion_route_still_fires(self) -> None:
        source = dedent("""
            class Worker:
                def accept_row(self):
                    try:
                        self._batch_executor.submit(self._process)
                    except RuntimeError:
                        from elspeth.contracts import TransformResult

                        TransformResult.error(
                            {"reason": "shutdown_requested"},
                            retryable=False,
                        )
        """)
        findings = parse_and_visit(source)

        r6_findings = [f for f in findings if f.rule_id == "R6"]
        assert len(r6_findings) == 1

    def test_nested_non_default_return_does_not_make_handler_non_silent(self) -> None:
        source = dedent("""
            try:
                int("not a number")
            except ValueError:
                def helper():
                    return {"handled": True}
                helper
        """)
        findings = parse_and_visit(source)

        r6_findings = [f for f in findings if f.rule_id == "R6"]
        assert len(r6_findings) == 1


# =============================================================================
# R8: setdefault() detection
# =============================================================================


class TestR8Setdefault:
    """Tests for R8: setdefault() detection."""

    def test_ignores_immediate_append_grouping_idiom(self) -> None:
        """setdefault(k, []).append(v) constructs a grouping bucket."""
        source = dedent("""
            def group(items):
                buckets = {}
                for item in items:
                    buckets.setdefault(item.key, []).append(item)
                return buckets
        """)
        findings = parse_and_visit(source)

        r8_findings = [f for f in findings if f.rule_id == "R8"]
        assert r8_findings == []

    def test_ignores_immediate_extend_grouping_idiom(self) -> None:
        """setdefault(k, []).extend(values) constructs a grouping bucket."""
        source = dedent("""
            def group(items):
                buckets = {}
                for key, values in items:
                    buckets.setdefault(key, []).extend(values)
                return buckets
        """)
        findings = parse_and_visit(source)

        r8_findings = [f for f in findings if f.rule_id == "R8"]
        assert r8_findings == []

    def test_flags_setdefault_assigned_for_later_use(self) -> None:
        """setdefault() remains flagged when it returns a value used later."""
        source = dedent("""
            def process(key):
                cache = {}
                bucket = cache.setdefault(key, [])
                return bucket
        """)
        findings = parse_and_visit(source)

        r8_findings = [f for f in findings if f.rule_id == "R8"]
        assert len(r8_findings) == 1

    def test_flags_setdefault_subscript_mutation(self) -> None:
        """setdefault()[name] mutation is not the narrow append/extend grouping idiom."""
        source = dedent("""
            def subscribe(run_id, queue, state):
                subscribers = {}
                subscribers.setdefault(run_id, {})[queue] = state
                return subscribers
        """)
        findings = parse_and_visit(source)

        r8_findings = [f for f in findings if f.rule_id == "R8"]
        assert len(r8_findings) == 1


# =============================================================================
# R5: isinstance() lattice classification
# =============================================================================


class TestR5IsinstanceClassification:
    """Tests for R5: isinstance() should only flag Tier-2 defensive checks."""

    @staticmethod
    def _r5_findings(source: str, filename: str = "test.py") -> list[Finding]:
        findings = parse_and_visit(source, filename=filename)
        return [f for f in findings if f.rule_id == "R5"]

    def test_regular_isinstance_still_flagged(self) -> None:
        """Ordinary isinstance() remains R5c and should be flagged."""
        source = dedent("""
            def process(value):
                if isinstance(value, str):
                    return value.strip()
                return value
        """)

        assert len(self._r5_findings(source)) == 1

    def test_frozen_dataclass_post_init_self_field_guard_not_flagged(self) -> None:
        """Frozen dataclass __post_init__ self-field guards are Tier-1 offensive guards."""
        source = dedent("""
            from dataclasses import dataclass

            @dataclass(frozen=True, slots=True)
            class TokenInfo:
                row_id: str

                def __post_init__(self) -> None:
                    if not isinstance(self.row_id, str):
                        raise TypeError("row_id must be str")
        """)

        assert self._r5_findings(source) == []

    def test_frozen_dataclass_post_init_non_self_value_still_flagged(self) -> None:
        """The post-init exclusion is limited to self.<field> invariant guards."""
        source = dedent("""
            from dataclasses import dataclass

            @dataclass(frozen=True)
            class TokenInfo:
                row_id: str

                def __post_init__(self) -> None:
                    value = object()
                    if isinstance(value, str):
                        raise TypeError("unexpected value")
        """)

        assert len(self._r5_findings(source)) == 1

    def test_frozen_dataclass_post_init_self_field_alias_not_flagged(self) -> None:
        """A local alias of self.<field> in frozen __post_init__ is still an invariant guard."""
        source = dedent("""
            from dataclasses import dataclass

            @dataclass(frozen=True)
            class ExampleBundle:
                args: tuple[object, ...]

                def __post_init__(self) -> None:
                    value = self.args
                    if isinstance(value, list):
                        object.__setattr__(self, "args", tuple(value))
        """)

        assert self._r5_findings(source) == []

    def test_frozen_dataclass_post_init_later_self_field_alias_still_flagged(self) -> None:
        """Only aliases assigned before the isinstance() call count as self-field guards."""
        source = dedent("""
            from dataclasses import dataclass

            @dataclass(frozen=True)
            class ExampleBundle:
                args: tuple[object, ...]

                def __post_init__(self) -> None:
                    if isinstance(value, list):
                        object.__setattr__(self, "args", tuple(value))
                    value = self.args
        """)

        assert len(self._r5_findings(source)) == 1

    def test_non_frozen_dataclass_post_init_self_field_still_flagged(self) -> None:
        """Mutable dataclasses are not part of the Tier-1 frozen-DTO guard exclusion."""
        source = dedent("""
            from dataclasses import dataclass

            @dataclass
            class TokenInfo:
                row_id: str

                def __post_init__(self) -> None:
                    if not isinstance(self.row_id, str):
                        raise TypeError("row_id must be str")
        """)

        assert len(self._r5_findings(source)) == 1

    def test_pydantic_before_validator_boundary_not_flagged(self) -> None:
        """Pydantic before validators consume Tier-3 input and may use isinstance."""
        source = dedent("""
            from pydantic import BaseModel, field_validator

            class RunEvent(BaseModel):
                payload: object

                @field_validator("payload", mode="before")
                @classmethod
                def _validate_payload(cls, value):
                    if not isinstance(value, dict):
                        raise ValueError("payload must be an object")
                    return value
        """)

        assert self._r5_findings(source, filename="web/execution/schemas.py") == []

    def test_fastapi_route_handler_boundary_not_flagged(self) -> None:
        """FastAPI route handlers are Tier-3 request boundaries."""
        source = dedent("""
            from fastapi import APIRouter

            router = APIRouter()

            @router.post("/sessions")
            async def create_session(payload):
                if not isinstance(payload, dict):
                    raise ValueError("payload must be an object")
                return payload
        """)

        assert self._r5_findings(source, filename="web/sessions/routes.py") == []

    def test_named_tier3_boundary_helper_not_flagged(self) -> None:
        """Closed-list boundary helper contexts can validate external provider payloads."""
        source = dedent("""
            from collections.abc import Mapping

            def token_usage_from_response(response):
                usage = getattr(response, "usage", None)
                if isinstance(usage, Mapping):
                    return usage
                return None
        """)

        assert self._r5_findings(source, filename="web/composer/llm_response_parsing.py") == []

    def test_unlisted_web_helper_still_flagged(self) -> None:
        """The boundary-helper split must not suppress arbitrary web helpers."""
        source = dedent("""
            def ordinary_helper(value):
                if isinstance(value, str):
                    return value.strip()
                return value
        """)

        assert len(self._r5_findings(source, filename="web/composer/service.py")) == 1


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

        matched = _match_finding(allowlist, finding)
        assert matched is not None
        assert isinstance(matched, AllowlistEntry)
        assert matched.key == entry.key
        assert entry.matched is True

    def test_scope_fallback_does_not_reuse_exact_matched_entry(self) -> None:
        """One allowlist entry must not suppress two distinct live findings."""
        scope_fingerprint = "a" * 64
        entry = AllowlistEntry(
            key="src/module.py:R1:process:fp=first",
            owner="test",
            reason="test",
            safety="test",
            expires=None,
            ast_path="body[0]/body[0]/value",
            scope_fingerprint=scope_fingerprint,
            judge_signature_version=2,
            judge_verdict=JudgeVerdict.ACCEPTED,
        )
        allowlist = Allowlist(entries=[entry])

        first = Finding(
            rule_id="R1",
            file_path="src/module.py",
            line=10,
            col=0,
            symbol_context=("process",),
            fingerprint="first",
            code_snippet="data.get('first')",
            message="test",
            ast_path="body[0]/body[0]/value",
            scope_fingerprint=scope_fingerprint,
            scope_depth=1,
        )
        second = Finding(
            rule_id="R1",
            file_path="src/module.py",
            line=11,
            col=0,
            symbol_context=("process",),
            fingerprint="second",
            code_snippet="data.get('second')",
            message="test",
            ast_path="body[1]/body[0]/value",
            scope_fingerprint=scope_fingerprint,
            scope_depth=1,
        )

        assert _match_finding(allowlist, first) is entry
        assert entry.matched is True
        assert _match_finding(allowlist, second) is None

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

        matched = _match_finding(allowlist, finding)
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
        stale = allowlist.get_unused_entries()
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

        stale = allowlist.get_unused_entries()
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

    @pytest.mark.parametrize(
        ("yaml_body", "expected_field"),
        [
            (
                """
version: 1
allow_hits:
  - key: "src/module.py:R1:process:fp=deadbeefcafebabe"
    owner: "john"
    reason: "Legacy code"
    safety: "Will be refactored"
    expires: 2026-12-31 23:59:59
""",
                r"allow_hits\[0\]\.expires",
            ),
            (
                """
version: 1
per_file_rules:
  - pattern: "src/module.py"
    rules: ["R1"]
    reason: "Legacy code"
    expires: 2026-12-31 23:59:59
""",
                r"per_file_rules\[0\]\.expires",
            ),
        ],
    )
    def test_load_rejects_timestamp_expiry_fields(self, temp_dir: Path, yaml_body: str, expected_field: str) -> None:
        """Expiry fields are date-only; YAML timestamps must not flow through as datetimes."""
        allowlist_path = temp_dir / "allowlist.yaml"
        allowlist_path.write_text(dedent(yaml_body))

        with pytest.raises(ValueError, match=rf"{expected_field} must be YYYY-MM-DD"):
            load_allowlist(allowlist_path)

    def test_load_nonexistent_file(self, temp_dir: Path) -> None:
        """Missing allowlist file is Tier-1 audit-data loss."""
        allowlist_path = temp_dir / "missing.yaml"

        with pytest.raises(FileNotFoundError, match="allowlist YAML file is required"):
            load_allowlist(allowlist_path)


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
        matched = _match_finding(allowlist, finding)
        assert matched is not None

        # Check stale entries
        stale = allowlist.get_unused_entries()
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
        stale = allowlist.get_unused_entries()
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
            rules=("R5",),
            reason="Type dispatch for normalization",
            expires=None,
            max_hits=None,
        )
        allowlist = Allowlist(entries=[], per_file_rules=[rule])

        for _ in range(50):
            _match_finding(allowlist, self._make_finding("core/canonical.py"))

        assert rule.matched_count == 50
        assert allowlist.get_exceeded_rules() == []

    def test_max_hits_within_limit(self) -> None:
        """Per-file rule with matched_count <= max_hits should not be exceeded."""
        rule = PerFileRule(
            pattern="core/canonical.py",
            rules=("R5",),
            reason="Type dispatch",
            expires=None,
            max_hits=18,
        )
        allowlist = Allowlist(entries=[], per_file_rules=[rule])

        for _ in range(18):
            _match_finding(allowlist, self._make_finding("core/canonical.py"))

        assert rule.matched_count == 18
        assert allowlist.get_exceeded_rules() == []

    def test_max_hits_exceeded(self) -> None:
        """Per-file rule exceeding max_hits should be reported."""
        rule = PerFileRule(
            pattern="core/canonical.py",
            rules=("R5",),
            reason="Type dispatch",
            expires=None,
            max_hits=5,
        )
        allowlist = Allowlist(entries=[], per_file_rules=[rule])

        for _ in range(8):
            _match_finding(allowlist, self._make_finding("core/canonical.py"))

        assert rule.matched_count == 8
        exceeded = allowlist.get_exceeded_rules()
        assert len(exceeded) == 1
        assert exceeded[0] is rule

    def test_max_hits_only_counts_matching_rule(self) -> None:
        """max_hits should only count hits for the matching rule, not other rules."""
        rule = PerFileRule(
            pattern="core/canonical.py",
            rules=("R5",),
            reason="Type dispatch",
            expires=None,
            max_hits=2,
        )
        allowlist = Allowlist(entries=[], per_file_rules=[rule])

        # R5 matches the rule
        _match_finding(allowlist, self._make_finding("core/canonical.py", rule_id="R5"))
        _match_finding(allowlist, self._make_finding("core/canonical.py", rule_id="R5"))
        # R1 does NOT match this rule (rules=("R5",))
        result = _match_finding(allowlist, self._make_finding("core/canonical.py", rule_id="R1"))
        assert result is None  # R1 not in rule's rules list

        assert rule.matched_count == 2
        assert allowlist.get_exceeded_rules() == []

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

    def test_load_missing_single_allowlist_file_crashes(self, temp_dir: Path) -> None:
        """A missing Tier-1 allowlist file is corruption, not an empty allowlist."""
        missing = temp_dir / "missing.yaml"

        with pytest.raises(FileNotFoundError, match="allowlist YAML file is required"):
            load_allowlist(missing)

    def test_load_yaml_file_rejects_oversized_input(self, temp_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """The YAML loader checks file size before calling yaml.safe_load."""
        allowlist_path = temp_dir / "oversized.yaml"
        allowlist_path.write_text("allow_hits: []\n")
        monkeypatch.setattr(
            "elspeth_lints.core.allowlist._MAX_ALLOWLIST_YAML_BYTES",
            len("allow_hits: []\n") - 1,
        )

        with pytest.raises(ValueError, match="exceeds maximum allowlist YAML size"):
            load_allowlist(allowlist_path)

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
# Bug fix tests: trust-tier analyzer bug cluster
# =============================================================================


class TestBannedRuleKeyValidation:
    """Tests for elspeth-9f34362456: banned-rule key-format check validates rule ID.

    Plan A Task 7: governance moved out of ``_parse_allow_hits`` (now in core)
    and into ``_validate_allowlist_governance``. The previous ``sys.exit(1)`` +
    stderr-print pattern was replaced with ``ValueError`` per CLAUDE.md
    audit-primacy order. Tests now exercise the validator directly.
    """

    def test_valid_banned_rule_in_allow_hits_rejected(self) -> None:
        """allow_hits entry with banned rule R3 should be rejected."""
        al = Allowlist(
            entries=[
                AllowlistEntry(
                    key="core/events.py:R3:SomeClass:fp=abc123",
                    owner="test",
                    reason="test",
                    safety="test",
                    expires=datetime(2099, 1, 1, tzinfo=UTC).date(),
                )
            ]
        )
        with pytest.raises(ValueError, match="banned rule R3"):
            _validate_allowlist_governance(al)

    def test_invalid_rule_id_in_key_rejected(self) -> None:
        """allow_hits entry with invalid (non-existent) rule ID should be rejected.

        A malformed key like 'foo.py:GARBAGE:bar:fp=abc' must not silently pass
        just because 'GARBAGE' is not banned — the rule-id is also checked
        against the live registry.
        """
        al = Allowlist(
            entries=[
                AllowlistEntry(
                    key="core/events.py:NONEXISTENT_RULE:SomeClass:fp=abc123",
                    owner="test",
                    reason="test",
                    safety="test",
                    expires=datetime(2099, 1, 1, tzinfo=UTC).date(),
                )
            ]
        )
        with pytest.raises(ValueError, match="unknown rule ID"):
            _validate_allowlist_governance(al)

    def test_malformed_key_missing_rule_id_rejected(self) -> None:
        """allow_hits entry with no colon (no rule ID extractable) should be rejected."""
        al = Allowlist(
            entries=[
                AllowlistEntry(
                    key="bare-key-no-colons",
                    owner="test",
                    reason="test",
                    safety="test",
                    expires=datetime(2099, 1, 1, tzinfo=UTC).date(),
                )
            ]
        )
        with pytest.raises(ValueError, match="malformed key"):
            _validate_allowlist_governance(al)

    def test_valid_rule_id_in_key_accepted(self) -> None:
        """allow_hits entry with valid non-banned rule ID should be accepted."""
        al = Allowlist(
            entries=[
                AllowlistEntry(
                    key="core/events.py:R1:SomeClass:fp=abc123",
                    owner="test",
                    reason="test",
                    safety="test",
                    expires=datetime(2099, 1, 1, tzinfo=UTC).date(),
                )
            ]
        )
        _validate_allowlist_governance(al)  # must not raise
        assert al.entries[0].key == "core/events.py:R1:SomeClass:fp=abc123"


class TestAllowHitPatternTags:
    """Tests for allow_hit pattern tag schema validation."""

    def test_valid_pattern_tag_is_preserved(self) -> None:
        """A valid pattern tag is parsed into the AllowlistEntry."""
        al = Allowlist(
            entries=[
                AllowlistEntry(
                    key="contracts/transform_contract.py:R2:_get_python_type:fp=abc123",
                    owner="bugfix",
                    reason="Type object display fallback",
                    safety="Used only in error message text",
                    expires=None,
                    pattern="display-fallback",
                )
            ]
        )
        _validate_allowlist_governance(al)
        assert al.entries[0].pattern == "display-fallback"

    def test_unknown_pattern_tag_is_rejected(self) -> None:
        """Pattern tags must come from the closed project vocabulary."""
        al = Allowlist(
            entries=[
                AllowlistEntry(
                    key="contracts/transform_contract.py:R2:_get_python_type:fp=abc123",
                    owner="bugfix",
                    reason="Type object display fallback",
                    safety="Used only in error message text",
                    expires=None,
                    pattern="whatever-this-is",
                )
            ]
        )
        with pytest.raises(ValueError, match="unknown pattern tag"):
            _validate_allowlist_governance(al)

    def test_permanent_bugfix_entry_requires_pattern(self) -> None:
        """owner=bugfix entries need expires or a pattern tag."""
        al = Allowlist(
            entries=[
                AllowlistEntry(
                    key="contracts/transform_contract.py:R2:_get_python_type:fp=abc123",
                    owner="bugfix",
                    reason="Type object display fallback",
                    safety="Used only in error message text",
                    expires=None,
                    pattern=None,
                )
            ]
        )
        with pytest.raises(ValueError, match="owner=bugfix"):
            _validate_allowlist_governance(al)

    def test_bugfix_entry_with_expiry_does_not_require_pattern(self) -> None:
        """Bounded bugfix entries may rely on expiry instead of a permanent pattern."""
        al = Allowlist(
            entries=[
                AllowlistEntry(
                    key="contracts/data.py:R6:_get_allow_inf_nan:fp=abc123",
                    owner="bugfix",
                    reason="Temporary narrow TypeError catch",
                    safety="Only catches TypeError from vars()",
                    expires=datetime(2099, 1, 1, tzinfo=UTC).date(),
                    pattern=None,
                )
            ]
        )
        _validate_allowlist_governance(al)
        assert al.entries[0].pattern is None


class TestMaxHitsParseError:
    """Tests for elspeth-cdeeeccde3: non-numeric max_hits is rejected by the core loader."""

    def test_non_numeric_max_hits_rejected(self, tmp_path: Path) -> None:
        """Non-numeric max_hits should raise a ``ValueError`` from the core loader.

        Core's loader rejects bool / non-int types for max_hits unconditionally,
        replacing tier_model's bespoke ``int(...)`` coercion + sys.exit pattern.
        The yaml-parsed string ``"five"`` reaches the loader as a ``str`` (not
        an int), so the typed parser rejects it.
        """
        al_file = tmp_path / "al.yaml"
        al_file.write_text(
            dedent("""\
                per_file_rules:
                  - pattern: "plugins/sources/*"
                    rules: [R1]
                    reason: test
                    max_hits: "five"
            """)
        )
        with pytest.raises(ValueError, match="max_hits"):
            load_allowlist(al_file)

    def test_numeric_int_max_hits_parses(self, tmp_path: Path) -> None:
        """Numeric ``max_hits: 18`` should parse correctly."""
        al_file = tmp_path / "al.yaml"
        al_file.write_text(
            dedent("""\
                per_file_rules:
                  - pattern: "plugins/sources/*"
                    rules: [R1]
                    reason: test
                    max_hits: 18
            """)
        )
        al = load_allowlist(al_file)
        assert al.per_file_rules[0].max_hits == 18


class TestPerFileRulesUnknownRuleValidation:
    """Tests for symmetric unknown-rule-ID validation in per_file_rules.

    Core's loader validates ``per_file_rules[].rules`` against the
    ``valid_rule_ids`` collection passed in (tier_model passes ``_ALL_RULE_IDS``
    via ``_load_tier_model_allowlist``). Unknown ids surface as ``ValueError``.
    """

    def test_unknown_rule_id_in_per_file_rules_rejected(self, tmp_path: Path) -> None:
        al_file = tmp_path / "al.yaml"
        al_file.write_text(
            dedent("""\
                per_file_rules:
                  - pattern: "plugins/sources/*"
                    rules: [R1, TYPO_RULE]
                    reason: test
            """)
        )
        with pytest.raises(ValueError, match="TYPO_RULE"):
            load_allowlist(al_file)

    def test_valid_rule_ids_in_per_file_rules_accepted(self, tmp_path: Path) -> None:
        al_file = tmp_path / "al.yaml"
        al_file.write_text(
            dedent("""\
                per_file_rules:
                  - pattern: "plugins/sources/*"
                    rules: [R1, R4, R5]
                    reason: test
            """)
        )
        al = load_allowlist(al_file)
        assert len(al.per_file_rules) == 1
        assert al.per_file_rules[0].rules == ("R1", "R4", "R5")

    def test_banned_rule_in_per_file_rules_rejected(self, tmp_path: Path) -> None:
        """Banned rules (RULES[id].banned=True) cannot appear in per_file_rules."""
        al_file = tmp_path / "al.yaml"
        al_file.write_text(
            dedent("""\
                per_file_rules:
                  - pattern: "plugins/sources/*"
                    rules: [R3]
                    reason: test
            """)
        )
        with pytest.raises(ValueError, match="banned rule"):
            load_allowlist(al_file)


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
            rules=("R5",),
            reason="Type dispatch",
            expires=None,
            max_hits=2,
        )
        allowlist = Allowlist(entries=[], per_file_rules=[rule])

        # Simulate 5 matches — exceeds max_hits=2
        for _ in range(5):
            _match_finding(allowlist, self._make_finding())

        # In a full scan, this would be exceeded
        assert allowlist.get_exceeded_rules() == [rule]

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


# =============================================================================
# Allowlist budget ratchet
# =============================================================================


class TestAllowlistBudgetRatchet:
    """Tests for hard allowlist-count budget ratchets."""

    def test_suggested_allowlist_entry_uses_quarterly_expiry(self) -> None:
        """New suggested allowlist entries default to a quarterly review window."""
        finding = Finding(
            rule_id="R1",
            file_path="core/events.py",
            line=10,
            col=0,
            symbol_context=("EventBus", "emit"),
            fingerprint="aaa",
            code_snippet="payload.get('event')",
            message="test",
        )

        suggested = finding.suggested_allowlist_entry()
        expires = datetime.strptime(suggested["expires"], "%Y-%m-%d").replace(tzinfo=UTC).date()

        assert expires >= datetime.now(UTC).date() + timedelta(days=80)

    def test_load_directory_reads_budget_defaults(self, temp_dir: Path) -> None:
        """Directory defaults may define hard allowlist count ceilings."""
        allowlist_dir = temp_dir / "allowlist"
        allowlist_dir.mkdir()
        (allowlist_dir / "_defaults.yaml").write_text(
            dedent("""\
            version: 1
            defaults:
              fail_on_stale: true
              fail_on_expired: true
              allowlist_budget:
                max_allow_hits: 3
                max_per_file_rules: 1
                max_total_entries: 4
                max_permanent_allow_hits: 2
                max_permanent_per_file_rules: 0
                max_permanent_total_entries: 2
            """)
        )

        allowlist = load_allowlist(allowlist_dir)

        assert allowlist.max_allow_hits == 3
        assert allowlist.max_per_file_rules == 1
        assert allowlist.max_total_entries == 4
        assert allowlist.max_permanent_allow_hits == 2
        assert allowlist.max_permanent_per_file_rules == 0
        assert allowlist.max_permanent_total_entries == 2

    def test_no_budget_defaults_to_no_budget_violations(self) -> None:
        """Existing callers without budget config keep current behavior."""
        allowlist = Allowlist(
            entries=[
                AllowlistEntry(
                    key="core/events.py:R1:EventBus:emit:fp=aaa",
                    owner="test",
                    reason="test",
                    safety="test",
                    expires=None,
                )
            ]
        )

        assert allowlist.get_budget_violations() == []

    def test_budget_violation_reports_allow_hits_per_file_and_total(self) -> None:
        """Each configured ceiling reports its own over-budget category."""
        allowlist = Allowlist(
            entries=[
                AllowlistEntry(
                    key="core/events.py:R1:EventBus:emit:fp=aaa",
                    owner="test",
                    reason="test",
                    safety="test",
                    expires=None,
                ),
                AllowlistEntry(
                    key="core/events.py:R1:EventBus:emit:fp=bbb",
                    owner="test",
                    reason="test",
                    safety="test",
                    expires=None,
                ),
            ],
            per_file_rules=[
                PerFileRule(pattern="core/config.py", rules=("R1",), reason="test", expires=None),
                PerFileRule(pattern="core/canonical.py", rules=("R5",), reason="test", expires=None),
            ],
            max_allow_hits=1,
            max_per_file_rules=1,
            max_total_entries=3,
        )

        violations = allowlist.get_budget_violations()

        assert [(v.category, v.current, v.max_allowed) for v in violations] == [
            ("allow_hits", 2, 1),
            ("per_file_rules", 2, 1),
            ("total_entries", 4, 3),
        ]

    def test_budget_violation_reports_permanent_entries(self) -> None:
        """Permanent expires:null entries have their own ratchet categories."""
        bounded_expiry = datetime(2099, 1, 1, tzinfo=UTC).date()
        allowlist = Allowlist(
            entries=[
                AllowlistEntry(
                    key="core/events.py:R1:EventBus:emit:fp=aaa",
                    owner="test",
                    reason="test",
                    safety="test",
                    expires=None,
                ),
                AllowlistEntry(
                    key="core/events.py:R1:EventBus:emit:fp=bbb",
                    owner="test",
                    reason="test",
                    safety="test",
                    expires=bounded_expiry,
                ),
            ],
            per_file_rules=[
                PerFileRule(pattern="core/config.py", rules=("R1",), reason="test", expires=None),
                PerFileRule(pattern="core/canonical.py", rules=("R5",), reason="test", expires=bounded_expiry),
            ],
            max_permanent_allow_hits=0,
            max_permanent_per_file_rules=0,
            max_permanent_total_entries=0,
        )

        violations = allowlist.get_budget_violations()

        assert [(v.category, v.current, v.max_allowed) for v in violations] == [
            ("permanent_allow_hits", 1, 0),
            ("permanent_per_file_rules", 1, 0),
            ("permanent_total_entries", 2, 0),
        ]

    def test_report_json_includes_budget_violations(self) -> None:
        """JSON report exposes budget overruns for CI annotations."""
        payload = json.loads(
            report_json(
                violations=[],
                stale_entries=[],
                expired_entries=[],
                budget_violations=[
                    AllowlistBudgetViolation(category="allow_hits", current=2, max_allowed=1),
                ],
            )
        )

        assert payload["allowlist_budget_violations"] == [
            {"category": "allow_hits", "current": 2, "max_allowed": 1},
        ]

    def test_run_check_fails_when_budget_exceeded(
        self,
        temp_dir: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """Full check fails when the loaded allowlist exceeds configured budget."""
        src_dir = temp_dir / "src"
        src_dir.mkdir()
        (src_dir / "example.py").write_text("def ok():\n    return 1\n")

        allowlist_dir = temp_dir / "allowlist"
        allowlist_dir.mkdir()
        (allowlist_dir / "_defaults.yaml").write_text(
            dedent("""\
            version: 1
            defaults:
              fail_on_stale: false
              fail_on_expired: false
              allowlist_budget:
                max_allow_hits: 0
            """)
        )
        (allowlist_dir / "core.yaml").write_text(
            dedent("""\
            allow_hits:
              - key: "core/events.py:R1:EventBus:emit:fp=aaa"
                owner: test
                reason: test
                safety: test
            """)
        )

        args = argparse.Namespace(
            root=src_dir,
            allowlist=allowlist_dir,
            exclude=[],
            format="text",
            files=[],
        )

        assert run_check(args) == 1
        captured = capsys.readouterr()
        assert "ALLOWLIST BUDGET EXCEEDED" in captured.out
        assert "allow_hits" in captured.out

    def test_run_check_fails_when_permanent_budget_exceeded(
        self,
        temp_dir: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """Full check fails when permanent exemptions exceed configured budget."""
        src_dir = temp_dir / "src"
        src_dir.mkdir()
        (src_dir / "example.py").write_text("def ok():\n    return 1\n")

        allowlist_dir = temp_dir / "allowlist"
        allowlist_dir.mkdir()
        (allowlist_dir / "_defaults.yaml").write_text(
            dedent("""\
            version: 1
            defaults:
              fail_on_stale: false
              fail_on_expired: false
              allowlist_budget:
                max_permanent_allow_hits: 0
            """)
        )
        (allowlist_dir / "core.yaml").write_text(
            dedent("""\
            allow_hits:
              - key: "core/events.py:R1:EventBus:emit:fp=aaa"
                owner: test
                reason: test
                safety: test
                expires: null
            """)
        )

        args = argparse.Namespace(
            root=src_dir,
            allowlist=allowlist_dir,
            exclude=[],
            format="text",
            files=[],
        )

        assert run_check(args) == 1
        captured = capsys.readouterr()
        assert "ALLOWLIST BUDGET EXCEEDED" in captured.out
        assert "permanent_allow_hits" in captured.out


# =============================================================================
# Plan A Task 7 consolidation invariants
# =============================================================================


class TestCoreAllowlistConsolidation:
    """Plan A Task 7: tier_model must reuse core's allowlist dataclasses.

    These tests pin the consolidation contract: identity (not just equivalence)
    of the dataclasses, absence of the deleted parser functions, and the
    governance validator's ``ValueError`` semantics.
    """

    def test_tier_model_imports_dataclasses_from_core(self) -> None:
        import elspeth_lints.rules.trust_tier.tier_model.rule as r
        from elspeth_lints.core.allowlist import Allowlist as CoreAllowlist
        from elspeth_lints.core.allowlist import AllowlistBudgetViolation as CoreBudget
        from elspeth_lints.core.allowlist import AllowlistEntry as CoreEntry
        from elspeth_lints.core.allowlist import PerFileRule as CorePerFileRule

        assert r.Allowlist is CoreAllowlist
        assert r.AllowlistBudgetViolation is CoreBudget
        assert r.AllowlistEntry is CoreEntry
        assert r.PerFileRule is CorePerFileRule

    def test_tier_model_does_not_export_duplicate_loaders(self) -> None:
        """The duplicate parser/loader helpers must be gone, not shadowed."""
        import elspeth_lints.rules.trust_tier.tier_model.rule as r

        assert "load_allowlist_from_directory" not in vars(r), "duplicate loader must be removed"
        assert "_parse_allow_hits" not in vars(r), "duplicate allow_hits parser must be removed"
        assert "_parse_per_file_rules" not in vars(r), "duplicate per_file_rules parser must be removed"
        assert "_parse_allowlist_budget" not in vars(r), "duplicate budget parser must be removed"
        assert "_load_yaml_file" not in vars(r), "duplicate yaml loader must be removed"

    def test_governance_validator_raises_valueerror_on_banned_rule(self) -> None:
        """sys.exit(1) was replaced with ValueError per CLAUDE.md audit-primacy."""
        # Pick a banned rule from the live registry rather than hard-coding "R3"
        import re as _re

        import elspeth_lints.rules.trust_tier.tier_model.rule as r

        banned_rule_id = next(iter(r._BANNED_RULES))
        bad_key = f"some/file.py:{banned_rule_id}:_module_:fp=abc"
        al = Allowlist(
            entries=[
                AllowlistEntry(
                    key=bad_key,
                    owner="x",
                    reason="x",
                    safety="x",
                    expires=datetime(2099, 1, 1, tzinfo=UTC).date(),
                )
            ]
        )
        with pytest.raises(ValueError, match=_re.compile(r"banned", _re.IGNORECASE)):
            _validate_allowlist_governance(al)

    def test_match_finding_preserves_per_file_first_order(self) -> None:
        """tier_model's historical match order (per_file_rules first) is preserved.

        Core's ``Allowlist.match`` checks entries first; we wrap it in
        ``_match_finding`` to preserve tier_model's historical accounting
        (per_file_rule gets the credit when both could match).
        """
        finding = Finding(
            rule_id="R1",
            file_path="plugins/foo.py",
            line=1,
            col=0,
            symbol_context=("f",),
            fingerprint="abc",
            code_snippet="x",
            message="m",
        )
        entry = AllowlistEntry(
            key=finding.canonical_key,
            owner="t",
            reason="t",
            safety="t",
            expires=None,
        )
        rule = PerFileRule(
            pattern="plugins/foo.py",
            rules=("R1",),
            reason="t",
            expires=None,
        )
        al = Allowlist(entries=[entry], per_file_rules=[rule])
        matched = _match_finding(al, finding)
        # The per_file_rule should be credited, not the exact entry.
        assert matched is rule
        assert rule.matched_count == 1
        assert entry.matched is False


# =============================================================================
# C8-3: matcher-side binding verification — in-file transplant defence.
#
# Cross-file transplant is caught at load time (test in
# test_allowlist_judge_metadata_integrity.py). In-file transplant — the same
# file's bytes pass the load-time file_fingerprint check, but the persisted
# ast_path points at a different AST node than the live finding does — is
# caught here at match time, in the rule's ``_match_finding``.
# =============================================================================


class TestC83InFileTransplantDefence:
    """The matcher refuses to honor a judge verdict whose ast_path no longer matches."""

    def test_matcher_rejects_judge_gated_entry_with_mismatched_ast_path(self) -> None:
        """An entry whose persisted ast_path differs from the live finding's fails the match.

        Construction: build a synthetic Finding and a synthetic
        AllowlistEntry whose canonical_key matches the finding (so
        ``entry.matches(finding_key)`` returns True), and whose
        judge_verdict is set (so binding verification arms). The
        entry's persisted ast_path deliberately differs from the
        finding's live ast_path — that is the in-file transplant
        shape. ``_match_finding`` must raise.
        """
        finding = Finding(
            rule_id="R1",
            file_path="plugins/widget.py",
            line=10,
            col=4,
            symbol_context=("Widget", "lookup"),
            fingerprint="livefp00",
            code_snippet="payload.get(...)",
            message="dict.get on Tier-2 data",
            ast_path="body[0]/body[1]/body[2]/value",  # the LIVE address
        )
        entry = AllowlistEntry(
            key=finding.canonical_key,  # canonical key MATCHES the finding
            owner="historic-agent",
            reason="r",
            safety="s",
            expires=None,
            file_fingerprint="0" * 64,
            ast_path="body[0]/body[1]/body[99]/value",  # transplanted: DIFFERENT
            judge_verdict=JudgeVerdict.ACCEPTED,
            judge_recorded_at=datetime(2026, 5, 1, tzinfo=UTC),
            judge_model=DEFAULT_JUDGE_MODEL,
            judge_policy_hash=JUDGE_POLICY_HASH,
            judge_rationale="judge accepted at a different node entirely",
        )
        al = Allowlist(entries=[entry], per_file_rules=[])
        with pytest.raises(ValueError, match=r"ast_path mismatch.*plugins/widget\.py"):
            _match_finding(al, finding)

    def test_matcher_accepts_judge_gated_entry_with_matching_ast_path(self) -> None:
        """Happy path: entry's persisted ast_path equals the live finding's, match succeeds."""
        finding = Finding(
            rule_id="R1",
            file_path="plugins/widget.py",
            line=10,
            col=4,
            symbol_context=("Widget", "lookup"),
            fingerprint="livefp00",
            code_snippet="payload.get(...)",
            message="dict.get on Tier-2 data",
            ast_path="body[0]/body[1]/body[2]/value",
        )
        entry = AllowlistEntry(
            key=finding.canonical_key,
            owner="historic-agent",
            reason="r",
            safety="s",
            expires=None,
            file_fingerprint="0" * 64,
            ast_path=finding.ast_path,  # matches
            judge_verdict=JudgeVerdict.ACCEPTED,
            judge_recorded_at=datetime(2026, 5, 1, tzinfo=UTC),
            judge_model=DEFAULT_JUDGE_MODEL,
            judge_policy_hash=JUDGE_POLICY_HASH,
            judge_rationale="judge accepted at this exact AST node",
        )
        al = Allowlist(entries=[entry], per_file_rules=[])
        matched = _match_finding(al, finding)
        assert matched is entry
        assert entry.matched is True

    def test_matcher_skips_binding_check_for_pre_judge_entry(self) -> None:
        """Pre-judge entries (no judge_verdict) carry no binding fields; matcher must skip the check.

        Pre-judge era entries predate the C8-3 binding wiring. They
        match purely on canonical key (as before); the binding gate
        only arms when ``judge_verdict`` is set. This test pins that
        backward-compatible behaviour so existing allowlists (the
        whole ~700-entry historical corpus) keep matching cleanly.
        """
        finding = Finding(
            rule_id="R1",
            file_path="plugins/widget.py",
            line=10,
            col=4,
            symbol_context=("Widget", "lookup"),
            fingerprint="livefp00",
            code_snippet="payload.get(...)",
            message="dict.get on Tier-2 data",
            ast_path="body[0]/body[1]/body[2]/value",
        )
        entry = AllowlistEntry(
            key=finding.canonical_key,
            owner="historic-agent",
            reason="r",
            safety="s",
            expires=None,
            # No judge_verdict, no binding fields — the pre-C8-3 shape.
        )
        al = Allowlist(entries=[entry], per_file_rules=[])
        matched = _match_finding(al, finding)
        assert matched is entry


# =============================================================================
# Layer-import (L1 / TC) scanner — relative & package-root upward imports
# (elspeth-b8b600e213) and nested TYPE_CHECKING classification (elspeth-b7ef37c4a9)
# =============================================================================


class TestLayerImportScanner:
    """scan_layer_imports_file: upward layer-import detection and TC classification."""

    @staticmethod
    def _scan(tmp_path: Path, rel: str, source: str, *, subpackages: tuple[str, ...] = ()) -> tuple[list[Finding], list[Finding]]:
        # tmp_path acts as the --root (src/elspeth style: paths like core/...).
        for pkg in subpackages:
            pkg_dir = tmp_path / pkg
            pkg_dir.mkdir(parents=True, exist_ok=True)
            (pkg_dir / "__init__.py").write_text("", encoding="utf-8")
        file_path = tmp_path / rel
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(dedent(source), encoding="utf-8")
        return scan_layer_imports_file(file_path, tmp_path)

    @staticmethod
    def _file_fingerprint(source: str) -> str:
        return hashlib.sha256(dedent(source).encode("utf-8")).hexdigest()

    def test_flags_absolute_upward_import(self, tmp_path: Path) -> None:
        # NO-REGRESSION (the case with no real exemplar): a plain absolute
        # level-0 upward import must STILL be flagged after the rewrite.
        source = "from elspeth.plugins import transforms\n"
        violations, tc = self._scan(tmp_path, "core/mod.py", source)
        assert [f.rule_id for f in violations] == ["L1"]
        assert violations[0].file_fingerprint == self._file_fingerprint(source)
        assert tc == []

    def test_flags_relative_upward_import(self, tmp_path: Path) -> None:
        # elspeth-b8b600e213: `from ..plugins import x` in a core file resolves to
        # elspeth.plugins (L3) and must be flagged.
        violations, _ = self._scan(tmp_path, "core/mod.py", "from ..plugins import transforms\n")
        assert [f.rule_id for f in violations] == ["L1"]

    def test_flags_package_root_subpackage_import(self, tmp_path: Path) -> None:
        # elspeth-b8b600e213: `from elspeth import plugins` (plugins IS a real
        # subpackage) must be flagged.
        violations, _ = self._scan(tmp_path, "core/mod.py", "from elspeth import plugins\n", subpackages=("plugins",))
        assert [f.rule_id for f in violations] == ["L1"]

    def test_ignores_package_root_attribute_import(self, tmp_path: Path) -> None:
        # FP trap: `from elspeth import __version__` imports an attribute, not a
        # subpackage — must NOT be flagged.
        violations, tc = self._scan(tmp_path, "core/mod.py", "from elspeth import __version__\n")
        assert violations == []
        assert tc == []

    def test_one_finding_per_multi_alias_import(self, tmp_path: Path) -> None:
        # Per-node (not per-alias) emission: `from elspeth.plugins import a, b, c`
        # is a single upward edge -> exactly one finding.
        violations, _ = self._scan(tmp_path, "core/mod.py", "from elspeth.plugins import a, b, c\n")
        assert len(violations) == 1

    def test_nested_type_checking_import_is_warning_not_violation(self, tmp_path: Path) -> None:
        # elspeth-b7ef37c4a9: an import nested in try/ inside `if TYPE_CHECKING:`
        # is annotation-only -> TC warning, not an L1 runtime violation.
        source = """
            from typing import TYPE_CHECKING

            if TYPE_CHECKING:
                try:
                    from elspeth.plugins import transforms
                except ImportError:
                    transforms = None
        """
        violations, tc = self._scan(tmp_path, "core/mod.py", source)
        assert violations == []
        assert [f.rule_id for f in tc] == ["TC"]
        assert tc[0].file_fingerprint == self._file_fingerprint(source)

    def test_type_checking_else_branch_is_runtime(self, tmp_path: Path) -> None:
        # The else: of `if TYPE_CHECKING:` is the runtime fallback — its imports
        # are genuine runtime and must stay L1 (getting the walk wrong fails OPEN).
        source = """
            from typing import TYPE_CHECKING

            if TYPE_CHECKING:
                pass
            else:
                from elspeth.plugins import transforms
        """
        violations, tc = self._scan(tmp_path, "core/mod.py", source)
        assert [f.rule_id for f in violations] == ["L1"]
        assert tc == []

    def test_sideways_and_downward_imports_not_flagged(self, tmp_path: Path) -> None:
        # engine (L2) importing core (L1) and contracts (L0) is allowed.
        violations, _ = self._scan(
            tmp_path,
            "engine/mod.py",
            "from elspeth.core import x\nfrom elspeth.contracts import y\n",
        )
        assert violations == []
