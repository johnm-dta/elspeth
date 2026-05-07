"""Tests for RAG query construction."""

import os

import pytest

from elspeth.plugins.infrastructure.templates import TemplateError
from elspeth.plugins.transforms.rag.query import QueryBuilder

# =============================================================================
# Field-only mode
# =============================================================================


class TestFieldOnlyMode:
    def test_extracts_value_verbatim(self):
        builder = QueryBuilder(query_field="question")
        result = builder.build({"question": "What is RAG?"})
        assert result.query == "What is RAG?"

    def test_missing_field_crashes(self):
        """Missing field is a contract violation (Tier 2) — crash, don't quarantine."""
        builder = QueryBuilder(query_field="question")
        with pytest.raises(KeyError):
            builder.build({"other_field": "value"})

    def test_none_value_returns_error(self):
        builder = QueryBuilder(query_field="question")
        result = builder.build({"question": None})
        assert result.error is not None
        assert result.error["reason"] == "invalid_input"
        assert result.error["cause"] == "null_value"

    def test_empty_string_returns_error(self):
        builder = QueryBuilder(query_field="question")
        result = builder.build({"question": ""})
        assert result.error is not None
        assert result.error["reason"] == "invalid_input"
        assert result.error["cause"] == "empty_query"

    def test_whitespace_only_returns_error(self):
        builder = QueryBuilder(query_field="question")
        result = builder.build({"question": "   \t\n  "})
        assert result.error is not None
        assert result.error["reason"] == "invalid_input"
        assert result.error["cause"] == "empty_query"


# =============================================================================
# Template mode
# =============================================================================


class TestTemplateMode:
    def test_renders_with_query_and_row(self):
        builder = QueryBuilder(
            query_field="topic",
            query_template="Find documents about {{ query }} for {{ row.category }}",
        )
        result = builder.build({"topic": "compliance", "category": "finance"})
        assert result.query == "Find documents about compliance for finance"

    def test_structural_error_at_compile_time(self):
        with pytest.raises(TemplateError):
            QueryBuilder(
                query_field="topic",
                query_template="{% if unclosed",
            )

    def test_render_error_returns_error(self):
        builder = QueryBuilder(
            query_field="topic",
            query_template="{{ query }} for {{ row.missing_field }}",
        )
        result = builder.build({"topic": "test"})
        assert result.error is not None
        assert result.error["reason"] == "template_rendering_failed"


# =============================================================================
# Regex mode
# =============================================================================


class TestRegexMode:
    def test_captures_first_group(self):
        builder = QueryBuilder(
            query_field="text",
            query_pattern=r"issue:\s*(.+?)(?:\n|$)",
        )
        result = builder.build({"text": "issue: payment failed\nother stuff"})
        assert result.query == "payment failed"

    def test_full_match_when_no_groups(self):
        builder = QueryBuilder(
            query_field="text",
            query_pattern=r"\w+@\w+\.\w+",
        )
        result = builder.build({"text": "contact user@example.com for help"})
        assert result.query == "user@example.com"

    def test_no_match_returns_error(self):
        builder = QueryBuilder(
            query_field="text",
            query_pattern=r"issue:\s*(.+)",
        )
        result = builder.build({"text": "no issue here"})
        assert result.error is not None
        assert result.error["reason"] == "no_regex_match"

    def test_non_participating_group_returns_error(self):
        """Optional capture group that didn't participate."""
        builder = QueryBuilder(
            query_field="text",
            query_pattern=r"(?:issue|problem)(?::\s*(.+?))?$",
        )
        result = builder.build({"text": "issue"})
        assert result.error is not None
        assert result.error["reason"] == "no_regex_match"
        assert result.error["cause"] == "capture_group_empty"

    def test_timeout_on_catastrophic_backtracking(self):
        """ReDoS protection: pathological pattern with adversarial input."""
        builder = QueryBuilder(
            query_field="text",
            query_pattern=r"(a+)+b",
            regex_timeout=0.1,  # Short timeout for test
        )
        result = builder.build({"text": "a" * 30})
        assert result.error is not None
        assert result.error["reason"] == "no_regex_match"
        assert result.error["cause"] == "regex_timeout"


# =============================================================================
# Subprocess crash detection
# =============================================================================


class TestWorkerFailureDetection:
    """Worker failures in the ProcessPoolExecutor surface as RuntimeError.

    _regex_worker is system-owned code. A crash or exception in the worker
    is a code bug, not a data issue. Per offensive programming rules: plugin
    bugs crash immediately — they don't silently quarantine.
    """

    def test_worker_exception_raises_runtime_error(self):
        """When the pool future raises, build() must raise RuntimeError."""
        from concurrent.futures import Future
        from unittest.mock import MagicMock

        builder = QueryBuilder(
            query_field="text",
            query_pattern=r"issue:\s*(.+)",
        )

        failed_future = Future()
        failed_future.set_exception(ValueError("simulated worker bug"))
        mock_pool = MagicMock()
        mock_pool.submit.return_value = failed_future
        builder._regex_pool = mock_pool

        with pytest.raises(RuntimeError, match="Regex worker failed"):
            builder.build({"text": "issue: payment failed"})

    def test_worker_error_includes_pattern_and_cause(self):
        """RuntimeError from worker failure includes the pattern for diagnostics."""
        from concurrent.futures import Future
        from unittest.mock import MagicMock

        builder = QueryBuilder(
            query_field="text",
            query_pattern=r"issue:\s*(.+)",
        )

        failed_future = Future()
        failed_future.set_exception(ValueError("kaboom"))
        mock_pool = MagicMock()
        mock_pool.submit.return_value = failed_future
        builder._regex_pool = mock_pool

        with pytest.raises(RuntimeError, match="issue:") as exc_info:
            builder.build({"text": "issue: payment failed"})
        assert "kaboom" in str(exc_info.value)


# =============================================================================
# Pool lifecycle
# =============================================================================


class TestPoolLifecycle:
    """ProcessPoolExecutor is created for regex mode and shut down on close()."""

    def test_pool_created_only_for_regex_mode(self):
        builder_field = QueryBuilder(query_field="text")
        assert builder_field._regex_pool is None

        builder_template = QueryBuilder(query_field="text", query_template="{{ query }}")
        assert builder_template._regex_pool is None

        builder_regex = QueryBuilder(query_field="text", query_pattern=r"\w+")
        assert builder_regex._regex_pool is not None
        builder_regex.close()

    def test_close_shuts_down_pool(self):
        builder = QueryBuilder(query_field="text", query_pattern=r"\w+")
        assert builder._regex_pool is not None
        builder.close()
        assert builder._regex_pool is None

    @pytest.mark.skipif(not os.path.exists("/proc"), reason="Linux /proc required")
    def test_no_fd_leak_after_repeated_evaluations(self):
        """FDs don't accumulate over many calls with pool reuse."""
        builder = QueryBuilder(
            query_field="text",
            query_pattern=r"issue:\s*(.+)",
        )
        pid = os.getpid()
        fd_count_before = len(os.listdir(f"/proc/{pid}/fd"))

        for _ in range(20):
            result = builder.build({"text": "issue: payment failed"})
            assert result.query is not None

        fd_count_after = len(os.listdir(f"/proc/{pid}/fd"))

        fd_count_after = len(os.listdir(f"/proc/{pid}/fd"))
        builder.close()

        assert fd_count_after - fd_count_before < 10, (
            f"File descriptor leak: {fd_count_after - fd_count_before} new FDs "
            f"after 20 regex evaluations (before={fd_count_before}, after={fd_count_after})"
        )
