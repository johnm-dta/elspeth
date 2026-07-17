"""Tests for JSONSink resume capability."""

import pytest

from elspeth.contracts.plugin_context import PluginContext
from elspeth.plugins.sinks.json_sink import JSONSink
from tests.fixtures.base_classes import inject_write_failure
from tests.fixtures.factories import make_context
from tests.fixtures.landscape import make_factory, make_landscape_db


@pytest.fixture
def ctx() -> PluginContext:
    """Create test context."""
    db = make_landscape_db()
    factory = make_factory(db)
    return make_context(landscape=factory.plugin_audit_writer())


class TestJSONSinkResumeCapability:
    """Tests for JSONSink resume declaration."""

    def test_jsonl_sink_supports_resume(self):
        """JSONL format should support resume."""
        sink = inject_write_failure(
            JSONSink(
                {
                    "path": "/tmp/test.jsonl",
                    "schema": {"mode": "observed"},
                    "format": "jsonl",
                }
            )
        )
        assert sink.supports_resume is True

    def test_json_array_sink_does_not_support_resume(self):
        """JSON array format should NOT support resume."""
        sink = inject_write_failure(
            JSONSink(
                {
                    "path": "/tmp/test.json",
                    "schema": {"mode": "observed"},
                    "format": "json",
                }
            )
        )
        assert sink.supports_resume is False

    def test_json_sink_auto_detect_jsonl_supports_resume(self):
        """Auto-detected JSONL format should support resume."""
        sink = inject_write_failure(
            JSONSink(
                {
                    "path": "/tmp/test.jsonl",  # .jsonl extension
                    "schema": {"mode": "observed"},
                    # No format specified - auto-detect
                }
            )
        )
        assert sink.supports_resume is True

    def test_json_sink_auto_detect_json_does_not_support_resume(self):
        """Auto-detected JSON array format should NOT support resume."""
        sink = inject_write_failure(
            JSONSink(
                {
                    "path": "/tmp/test.json",  # .json extension
                    "schema": {"mode": "observed"},
                    # No format specified - auto-detect
                }
            )
        )
        assert sink.supports_resume is False


class TestJSONSinkConfigureForResume:
    """Tests for JSONSink configure_for_resume error contract."""

    def test_json_array_configure_for_resume_raises(self):
        """JSON array sink configure_for_resume should raise NotImplementedError."""
        sink = inject_write_failure(
            JSONSink(
                {
                    "path": "/tmp/test.json",
                    "schema": {"mode": "observed"},
                    "format": "json",
                }
            )
        )

        with pytest.raises(NotImplementedError):
            sink.configure_for_resume()


class TestJSONSinkResumeModeResolution:
    """elspeth-fc9906e398: resolved effect mode must claim what resume executes."""

    def test_resume_purpose_resolves_post_resume_append_mode(self) -> None:
        """A write-configured JSONL sink resumes in append mode; the resolver must say so."""
        from elspeth.contracts.sink_effects import ResolvedSinkEffectMode, SinkEffectExecutionPurpose

        config = {"path": "/tmp/out.jsonl", "schema": {"mode": "observed"}, "format": "jsonl", "mode": "write"}

        resolved = JSONSink._resolve_sink_effect_mode(config, purpose=SinkEffectExecutionPurpose.RESUME)

        assert resolved == ResolvedSinkEffectMode("append")

    def test_fresh_purpose_keeps_configured_mode(self) -> None:
        from elspeth.contracts.sink_effects import ResolvedSinkEffectMode, SinkEffectExecutionPurpose

        config = {"path": "/tmp/out.jsonl", "schema": {"mode": "observed"}, "format": "jsonl", "mode": "write"}

        resolved = JSONSink._resolve_sink_effect_mode(config, purpose=SinkEffectExecutionPurpose.FRESH)

        assert resolved == ResolvedSinkEffectMode("write")
