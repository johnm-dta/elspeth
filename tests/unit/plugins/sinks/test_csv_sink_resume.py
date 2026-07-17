"""Tests for CSVSink resume capability."""

import pytest

from elspeth.contracts.plugin_context import PluginContext
from elspeth.plugins.sinks.csv_sink import CSVSink
from tests.fixtures.base_classes import inject_write_failure
from tests.fixtures.factories import make_context
from tests.fixtures.landscape import make_factory, make_landscape_db

# Strict schema for tests - CSVSink requires fixed columns
STRICT_SCHEMA = {"mode": "fixed", "fields": ["id: int"]}


@pytest.fixture
def ctx() -> PluginContext:
    """Create test context."""
    db = make_landscape_db()
    factory = make_factory(db)
    return make_context(landscape=factory.plugin_audit_writer())


class TestCSVSinkResumeContract:
    """Tests for CSVSink resume contract (public API)."""

    def test_csv_sink_supports_resume(self) -> None:
        """CSVSink should declare supports_resume=True."""
        assert CSVSink.supports_resume is True


class TestCSVSinkResumeModeResolution:
    """elspeth-fc9906e398: resolved effect mode must claim what resume executes."""

    def test_resume_purpose_resolves_post_resume_append_mode(self) -> None:
        """A write-configured CSV sink resumes in append mode; the resolver must say so."""
        from elspeth.contracts.sink_effects import ResolvedSinkEffectMode, SinkEffectExecutionPurpose

        config = {"path": "/tmp/out.csv", "schema": STRICT_SCHEMA, "mode": "write"}

        resolved = CSVSink._resolve_sink_effect_mode(config, purpose=SinkEffectExecutionPurpose.RESUME)

        assert resolved == ResolvedSinkEffectMode("append")

    def test_resume_purpose_matches_configure_for_resume_live_mode(self) -> None:
        """Resolver claim and configure_for_resume() mutation must agree exactly."""
        from elspeth.contracts.sink_effects import SinkEffectExecutionPurpose

        config = {"path": "/tmp/out.csv", "schema": STRICT_SCHEMA, "mode": "write"}
        sink = inject_write_failure(CSVSink(dict(config)))
        sink.configure_for_resume()

        resolved = CSVSink._resolve_sink_effect_mode(config, purpose=SinkEffectExecutionPurpose.RESUME)

        assert resolved is not None
        assert resolved.value == sink._mode == "append"

    def test_fresh_purpose_keeps_configured_mode(self) -> None:
        from elspeth.contracts.sink_effects import ResolvedSinkEffectMode, SinkEffectExecutionPurpose

        config = {"path": "/tmp/out.csv", "schema": STRICT_SCHEMA, "mode": "write"}

        resolved = CSVSink._resolve_sink_effect_mode(config, purpose=SinkEffectExecutionPurpose.FRESH)

        assert resolved == ResolvedSinkEffectMode("write")
