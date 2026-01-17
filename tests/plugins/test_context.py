# tests/plugins/test_context.py
"""Tests for plugin context."""

from contextlib import nullcontext
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pytest


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


class TestValidationErrorRecording:
    """Tests for recording validation errors from sources."""

    def test_record_validation_error_exists(self) -> None:
        """PluginContext has record_validation_error method."""
        from elspeth.plugins.context import PluginContext

        ctx = PluginContext(run_id="test-run", config={})
        assert hasattr(ctx, "record_validation_error")
        assert callable(ctx.record_validation_error)

    def test_record_validation_error_returns_quarantine_token(self) -> None:
        """record_validation_error returns token for tracking quarantined row."""
        from elspeth.plugins.context import PluginContext, ValidationErrorToken

        ctx = PluginContext(run_id="test-run", config={}, node_id="source_node")

        token = ctx.record_validation_error(
            row={"id": 42, "invalid": "data"},
            error="validation failed",
            schema_mode="strict",
            destination="discard",
        )

        assert token is not None
        assert isinstance(token, ValidationErrorToken)
        assert token.row_id is not None
        assert token.node_id == "source_node"

    def test_record_validation_error_without_landscape_logs_warning(
        self, caplog: "pytest.LogCaptureFixture"
    ) -> None:
        """record_validation_error logs warning when no landscape configured."""
        import logging

        from elspeth.plugins.context import PluginContext

        ctx = PluginContext(run_id="test-run", config={}, node_id="source_node")

        with caplog.at_level(logging.WARNING):
            token = ctx.record_validation_error(
                row={"id": 42, "invalid": "data"},
                error="validation failed",
                schema_mode="strict",
                destination="discard",
            )

        # Should still return a token even without landscape
        assert token is not None
        assert token.error_id is None  # Not recorded to landscape
        # Check that warning was logged
        assert "no landscape" in caplog.text.lower()

    def test_record_validation_error_uses_id_field_as_row_id(self) -> None:
        """record_validation_error uses row's id field if present."""
        from elspeth.plugins.context import PluginContext

        ctx = PluginContext(run_id="test-run", config={}, node_id="source_node")

        token = ctx.record_validation_error(
            row={"id": "row-42", "invalid": "data"},
            error="validation failed",
            schema_mode="strict",
            destination="discard",
        )

        assert token.row_id == "row-42"

    def test_record_validation_error_generates_row_id_from_hash(self) -> None:
        """record_validation_error generates row_id from content hash if no id field."""
        from elspeth.plugins.context import PluginContext

        ctx = PluginContext(run_id="test-run", config={}, node_id="source_node")

        token = ctx.record_validation_error(
            row={"no_id_field": "data"},
            error="validation failed",
            schema_mode="strict",
            destination="discard",
        )

        # Should generate a row_id from hash (16 chars)
        assert token.row_id is not None
        assert len(token.row_id) == 16

    def test_record_validation_error_with_landscape(self) -> None:
        """record_validation_error records to landscape when configured."""
        from unittest.mock import MagicMock

        from elspeth.plugins.context import PluginContext

        # Create mock landscape recorder
        mock_landscape = MagicMock()
        mock_landscape.record_validation_error.return_value = "verr_abc123"

        ctx = PluginContext(
            run_id="test-run",
            config={},
            node_id="source_node",
            landscape=mock_landscape,
        )

        token = ctx.record_validation_error(
            row={"id": 42, "invalid": "data"},
            error="validation failed",
            schema_mode="strict",
            destination="quarantine_sink",
        )

        # Should have called landscape
        mock_landscape.record_validation_error.assert_called_once()
        call_kwargs = mock_landscape.record_validation_error.call_args[1]
        assert call_kwargs["run_id"] == "test-run"
        assert call_kwargs["node_id"] == "source_node"
        assert call_kwargs["row_data"] == {"id": 42, "invalid": "data"}
        assert call_kwargs["error"] == "validation failed"
        assert call_kwargs["schema_mode"] == "strict"
        assert call_kwargs["destination"] == "quarantine_sink"

        # Token should have error_id from landscape
        assert token.error_id == "verr_abc123"
        assert token.destination == "quarantine_sink"


class TestValidationErrorDestination:
    """Tests for validation error destination tracking."""

    def test_validation_error_token_has_destination(self) -> None:
        """ValidationErrorToken includes destination field."""
        from elspeth.plugins.context import ValidationErrorToken

        token = ValidationErrorToken(
            row_id="row_1",
            node_id="source_node",
            destination="quarantine_sink",
        )

        assert token.destination == "quarantine_sink"

    def test_validation_error_token_defaults_to_discard(self) -> None:
        """ValidationErrorToken defaults to 'discard' if not specified."""
        from elspeth.plugins.context import ValidationErrorToken

        token = ValidationErrorToken(row_id="row_1", node_id="source_node")

        assert token.destination == "discard"

    def test_record_validation_error_requires_destination(self) -> None:
        """record_validation_error requires destination parameter."""
        from elspeth.plugins.context import PluginContext

        ctx = PluginContext(run_id="test-run", config={}, node_id="source_node")

        # Should work with destination
        token = ctx.record_validation_error(
            row={"id": 1, "bad": "data"},
            error="validation failed",
            schema_mode="strict",
            destination="quarantine_sink",
        )

        assert token.destination == "quarantine_sink"

    def test_record_validation_error_with_discard_destination(self) -> None:
        """record_validation_error accepts 'discard' as destination."""
        from elspeth.plugins.context import PluginContext

        ctx = PluginContext(run_id="test-run", config={}, node_id="source_node")

        token = ctx.record_validation_error(
            row={"id": 1, "bad": "data"},
            error="validation failed",
            schema_mode="strict",
            destination="discard",
        )

        assert token.destination == "discard"


class TestRouteToSink:
    """Tests for route_to_sink method."""

    def test_route_to_sink_exists(self) -> None:
        """PluginContext has route_to_sink method."""
        from elspeth.plugins.context import PluginContext

        ctx = PluginContext(run_id="test-run", config={}, node_id="test_node")
        assert hasattr(ctx, "route_to_sink")
        assert callable(ctx.route_to_sink)

    def test_route_to_sink_logs_action(
        self, caplog: "pytest.LogCaptureFixture"
    ) -> None:
        """route_to_sink logs the routing action."""
        import logging

        from elspeth.plugins.context import PluginContext

        ctx = PluginContext(run_id="test-run", config={}, node_id="source_node")

        with caplog.at_level(logging.INFO):
            ctx.route_to_sink(
                sink_name="quarantine",
                row={"id": 1, "bad": "data"},
                metadata={"reason": "validation failed"},
            )

        assert "route_to_sink" in caplog.text
        assert "quarantine" in caplog.text

    def test_route_to_sink_accepts_none_metadata(self) -> None:
        """route_to_sink works without metadata."""
        from elspeth.plugins.context import PluginContext

        ctx = PluginContext(run_id="test-run", config={}, node_id="test_node")

        # Should not raise
        ctx.route_to_sink(sink_name="error_sink", row={"id": 42})
