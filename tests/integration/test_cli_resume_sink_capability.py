"""Integration tests for CLI resume sink capability validation.

Tests that CLI resume properly uses the polymorphic sink resume capability
instead of blindly injecting mode=append into all sinks.

BUG-CLI-02 FIX: The old code injected mode=append into sink options before
instantiation. This crashed for sinks that don't have a 'mode' field
(DatabaseSink, AzureBlobSink). The fix uses sink.supports_resume and
sink.configure_for_resume() to let sinks self-configure.
"""

import pytest

from elspeth.plugins.sinks.csv_sink import CSVSink
from elspeth.plugins.sinks.database_sink import DatabaseSink
from elspeth.plugins.sinks.json_sink import JSONSink


class TestCLIResumeCallsConfigureForResume:
    """Tests that CLI resume properly calls configure_for_resume on sinks."""

    def test_resume_calls_configure_for_resume_on_resumable_sinks(self):
        """Resume should call configure_for_resume on each sink."""
        # Create sink and verify configure_for_resume changes mode
        sink = CSVSink(
            {
                "path": "/tmp/test.csv",
                "schema": {"mode": "strict", "fields": ["id: int", "name: str"]},
                "mode": "write",
            }
        )

        assert sink._mode == "write"
        assert sink.supports_resume is True

        sink.configure_for_resume()

        assert sink._mode == "append"

    def test_non_resumable_sink_detected(self):
        """Non-resumable sinks should be detectable before resume."""
        # JSON array format does not support resume
        sink = JSONSink(
            {
                "path": "/tmp/test.json",
                "schema": {"fields": "dynamic"},
                "format": "json",
            }
        )

        assert sink.supports_resume is False

        with pytest.raises(NotImplementedError):
            sink.configure_for_resume()

    def test_jsonl_sink_is_resumable(self):
        """JSONL format should support resume."""
        sink = JSONSink(
            {
                "path": "/tmp/test.jsonl",
                "schema": {"fields": "dynamic"},
                "format": "jsonl",
            }
        )

        assert sink.supports_resume is True

        sink.configure_for_resume()

        assert sink._mode == "append"


class TestDatabaseSinkResumeCapability:
    """Tests for DatabaseSink resume capability."""

    def test_database_sink_supports_resume(self):
        """DatabaseSink should support resume (append to table)."""
        sink = DatabaseSink(
            {
                "url": "sqlite:///:memory:",
                "table": "test_output",
                "schema": {"mode": "strict", "fields": ["id: int", "name: str"]},
                "if_exists": "replace",
            }
        )

        assert sink.supports_resume is True

    def test_database_sink_configure_for_resume(self):
        """configure_for_resume should switch to append mode."""
        sink = DatabaseSink(
            {
                "url": "sqlite:///:memory:",
                "table": "test_output",
                "schema": {"mode": "strict", "fields": ["id: int", "name: str"]},
                "if_exists": "replace",
            }
        )

        assert sink._if_exists == "replace"

        sink.configure_for_resume()

        assert sink._if_exists == "append"


class TestCSVSinkResumeCapability:
    """Tests for CSVSink resume capability."""

    def test_csv_sink_supports_resume(self):
        """CSVSink should support resume."""
        sink = CSVSink(
            {
                "path": "/tmp/test.csv",
                "schema": {"mode": "strict", "fields": ["id: int", "name: str"]},
            }
        )

        assert sink.supports_resume is True

    def test_csv_sink_configure_for_resume(self):
        """configure_for_resume should switch mode to append."""
        sink = CSVSink(
            {
                "path": "/tmp/test.csv",
                "schema": {"mode": "strict", "fields": ["id: int", "name: str"]},
                "mode": "write",
            }
        )

        assert sink._mode == "write"

        sink.configure_for_resume()

        assert sink._mode == "append"


class TestJSONSinkResumeCapability:
    """Tests for JSONSink resume capability (format-dependent)."""

    def test_jsonl_format_supports_resume(self):
        """JSONL format should support resume."""
        sink = JSONSink(
            {
                "path": "/tmp/test.jsonl",
                "schema": {"fields": "dynamic"},
                "format": "jsonl",
            }
        )

        assert sink.supports_resume is True

    def test_json_array_format_does_not_support_resume(self):
        """JSON array format should NOT support resume."""
        sink = JSONSink(
            {
                "path": "/tmp/test.json",
                "schema": {"fields": "dynamic"},
                "format": "json",
            }
        )

        assert sink.supports_resume is False

    def test_json_array_configure_for_resume_raises(self):
        """configure_for_resume should raise for JSON array format."""
        sink = JSONSink(
            {
                "path": "/tmp/test.json",
                "schema": {"fields": "dynamic"},
                "format": "json",
            }
        )

        with pytest.raises(NotImplementedError) as exc_info:
            sink.configure_for_resume()

        assert "JSON array format rewrites" in str(exc_info.value)

    def test_jsonl_configure_for_resume(self):
        """configure_for_resume should switch JSONL mode to append."""
        sink = JSONSink(
            {
                "path": "/tmp/test.jsonl",
                "schema": {"fields": "dynamic"},
                "format": "jsonl",
                "mode": "write",
            }
        )

        assert sink._mode == "write"

        sink.configure_for_resume()

        assert sink._mode == "append"

    def test_json_format_autodetected_from_extension(self):
        """Format should be autodetected from file extension."""
        # .json extension => JSON array format
        json_sink = JSONSink(
            {
                "path": "/tmp/test.json",
                "schema": {"fields": "dynamic"},
            }
        )
        assert json_sink.supports_resume is False

        # .jsonl extension => JSONL format
        jsonl_sink = JSONSink(
            {
                "path": "/tmp/test.jsonl",
                "schema": {"fields": "dynamic"},
            }
        )
        assert jsonl_sink.supports_resume is True
