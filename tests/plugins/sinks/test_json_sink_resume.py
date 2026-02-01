"""Tests for JSONSink resume capability."""

import pytest

from elspeth.plugins.sinks.json_sink import JSONSink


class TestJSONSinkResumeCapability:
    """Tests for JSONSink resume declaration."""

    def test_jsonl_sink_supports_resume(self):
        """JSONL format should support resume."""
        sink = JSONSink(
            {
                "path": "/tmp/test.jsonl",
                "schema": {"fields": "dynamic"},
                "format": "jsonl",
            }
        )
        assert sink.supports_resume is True

    def test_json_array_sink_does_not_support_resume(self):
        """JSON array format should NOT support resume."""
        sink = JSONSink(
            {
                "path": "/tmp/test.json",
                "schema": {"fields": "dynamic"},
                "format": "json",
            }
        )
        assert sink.supports_resume is False

    def test_json_sink_auto_detect_jsonl_supports_resume(self):
        """Auto-detected JSONL format should support resume."""
        sink = JSONSink(
            {
                "path": "/tmp/test.jsonl",  # .jsonl extension
                "schema": {"fields": "dynamic"},
                # No format specified - auto-detect
            }
        )
        assert sink.supports_resume is True

    def test_json_sink_auto_detect_json_does_not_support_resume(self):
        """Auto-detected JSON array format should NOT support resume."""
        sink = JSONSink(
            {
                "path": "/tmp/test.json",  # .json extension
                "schema": {"fields": "dynamic"},
                # No format specified - auto-detect
            }
        )
        assert sink.supports_resume is False


class TestJSONSinkConfigureForResume:
    """Tests for JSONSink configure_for_resume behavior."""

    def test_jsonl_configure_for_resume_sets_append_mode(self):
        """JSONL sink configure_for_resume should set append mode."""
        sink = JSONSink(
            {
                "path": "/tmp/test.jsonl",
                "schema": {"fields": "dynamic"},
                "format": "jsonl",
                "mode": "write",  # Explicit write mode
            }
        )

        assert sink._mode == "write"

        sink.configure_for_resume()

        assert sink._mode == "append"

    def test_json_array_configure_for_resume_raises(self):
        """JSON array sink configure_for_resume should raise NotImplementedError."""
        sink = JSONSink(
            {
                "path": "/tmp/test.json",
                "schema": {"fields": "dynamic"},
                "format": "json",
            }
        )

        with pytest.raises(NotImplementedError) as exc_info:
            sink.configure_for_resume()

        assert "JSON array" in str(exc_info.value) or "json" in str(exc_info.value).lower()
        assert "jsonl" in str(exc_info.value).lower()

    def test_jsonl_configure_for_resume_idempotent(self):
        """Calling configure_for_resume multiple times should be safe."""
        sink = JSONSink(
            {
                "path": "/tmp/test.jsonl",
                "schema": {"fields": "dynamic"},
                "format": "jsonl",
            }
        )

        sink.configure_for_resume()
        sink.configure_for_resume()  # Second call

        assert sink._mode == "append"


class TestJSONSinkModeField:
    """Tests for JSONSink mode configuration field."""

    def test_json_sink_mode_default_is_write(self):
        """JSONSink should default to mode='write'."""
        sink = JSONSink(
            {
                "path": "/tmp/test.jsonl",
                "schema": {"fields": "dynamic"},
            }
        )
        assert sink._mode == "write"

    def test_json_sink_respects_append_mode(self):
        """JSONSink should respect mode='append' config."""
        sink = JSONSink(
            {
                "path": "/tmp/test.jsonl",
                "schema": {"fields": "dynamic"},
                "mode": "append",
            }
        )
        assert sink._mode == "append"
