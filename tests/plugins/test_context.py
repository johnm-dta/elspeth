# tests/plugins/test_context.py
"""Tests for plugin context."""

from contextlib import nullcontext


class TestPluginContext:
    """Context passed to all plugin operations."""

    def test_minimal_context(self) -> None:
        from elspeth.plugins.context import PluginContext

        ctx = PluginContext(run_id="run-001", config={})
        assert ctx.run_id == "run-001"
        assert ctx.config == {}

    def test_optional_integrations_default_none(self) -> None:
        from elspeth.plugins.context import PluginContext

        ctx = PluginContext(run_id="run-001", config={})
        # Phase 3 integration points - optional in Phase 2
        assert ctx.landscape is None
        assert ctx.tracer is None
        assert ctx.payload_store is None

    def test_start_span_without_tracer(self) -> None:
        from elspeth.plugins.context import PluginContext

        ctx = PluginContext(run_id="run-001", config={})
        # Should return nullcontext when no tracer
        span_ctx = ctx.start_span("test_operation")
        assert isinstance(span_ctx, nullcontext)

    def test_get_config_value(self) -> None:
        from elspeth.plugins.context import PluginContext

        ctx = PluginContext(
            run_id="run-001",
            config={"threshold": 0.5, "nested": {"key": "value"}},
        )
        assert ctx.get("threshold") == 0.5
        assert ctx.get("nested.key") == "value"
        assert ctx.get("missing", default="default") == "default"
