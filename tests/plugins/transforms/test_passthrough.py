"""Tests for PassThrough transform."""

from typing import Any

import pytest

from elspeth.plugins.context import PluginContext
from elspeth.plugins.protocols import TransformProtocol

# Common schema config for dynamic field handling (accepts any fields)
DYNAMIC_SCHEMA = {"fields": "dynamic"}


class TestPassThrough:
    """Tests for PassThrough transform plugin."""

    @pytest.fixture
    def ctx(self) -> PluginContext:
        """Create minimal plugin context."""
        return PluginContext(run_id="test-run", config={})

    def test_implements_protocol(self) -> None:
        """PassThrough implements TransformProtocol."""
        from elspeth.plugins.transforms.passthrough import PassThrough

        transform = PassThrough({"schema": DYNAMIC_SCHEMA})
        assert isinstance(transform, TransformProtocol)

    def test_has_required_attributes(self) -> None:
        """PassThrough has name."""
        from elspeth.plugins.transforms.passthrough import PassThrough

        assert PassThrough.name == "passthrough"

    def test_instance_has_schemas(self) -> None:
        """PassThrough instance has input/output schemas."""
        from elspeth.plugins.transforms.passthrough import PassThrough

        transform = PassThrough({"schema": DYNAMIC_SCHEMA})
        assert hasattr(transform, "input_schema")
        assert hasattr(transform, "output_schema")

    def test_process_returns_unchanged_row(self, ctx: PluginContext) -> None:
        """process() returns row data unchanged."""
        from elspeth.plugins.transforms.passthrough import PassThrough

        transform = PassThrough({"schema": DYNAMIC_SCHEMA})
        row = {"id": 1, "name": "alice", "value": 100}

        result = transform.process(row, ctx)

        assert result.status == "success"
        assert result.row == row
        assert result.row is not row  # Should be a copy, not the same object

    def test_process_with_nested_data(self, ctx: PluginContext) -> None:
        """Handles nested structures correctly."""
        from elspeth.plugins.transforms.passthrough import PassThrough

        transform = PassThrough({"schema": DYNAMIC_SCHEMA})
        row: dict[str, Any] = {"id": 1, "meta": {"source": "test", "tags": ["a", "b"]}}

        result = transform.process(row, ctx)

        assert result.status == "success"
        assert result.row is not None
        assert result.row == row
        # Nested structures should be deep copied
        assert result.row["meta"] is not row["meta"]
        assert result.row["meta"]["tags"] is not row["meta"]["tags"]

    def test_process_with_empty_row(self, ctx: PluginContext) -> None:
        """Handles empty row."""
        from elspeth.plugins.transforms.passthrough import PassThrough

        transform = PassThrough({"schema": DYNAMIC_SCHEMA})
        row: dict[str, Any] = {}

        result = transform.process(row, ctx)

        assert result.status == "success"
        assert result.row == {}

    def test_close_is_idempotent(self) -> None:
        """close() can be called multiple times."""
        from elspeth.plugins.transforms.passthrough import PassThrough

        transform = PassThrough({"schema": DYNAMIC_SCHEMA})
        transform.close()
        transform.close()  # Should not raise

    def test_requires_schema_config(self) -> None:
        """PassThrough requires schema configuration."""
        from elspeth.plugins.config_base import PluginConfigError
        from elspeth.plugins.transforms.passthrough import PassThrough

        with pytest.raises(PluginConfigError, match="schema"):
            PassThrough({})

    def test_validate_input_rejects_wrong_type(self, ctx: PluginContext) -> None:
        """validate_input=True crashes on wrong types (upstream bug).

        Per three-tier trust model: transforms use allow_coercion=False,
        so string "42" is NOT coerced to int 42 - it raises ValidationError.
        """
        from pydantic import ValidationError

        from elspeth.plugins.transforms.passthrough import PassThrough

        transform = PassThrough(
            {
                "schema": {"mode": "strict", "fields": ["count: int"]},
                "validate_input": True,
            }
        )

        with pytest.raises(ValidationError):
            transform.process({"count": "not_an_int"}, ctx)

    def test_validate_input_disabled_passes_wrong_type(self, ctx: PluginContext) -> None:
        """validate_input=False (default) passes wrong types through.

        When validation is disabled, the transform doesn't check types.
        This is the default to avoid breaking existing pipelines.
        """
        from elspeth.plugins.transforms.passthrough import PassThrough

        transform = PassThrough(
            {
                "schema": {"mode": "strict", "fields": ["count: int"]},
                "validate_input": False,  # Explicit default
            }
        )

        # String passes through without validation
        result = transform.process({"count": "not_an_int"}, ctx)
        assert result.status == "success"
        assert result.row is not None
        assert result.row["count"] == "not_an_int"

    def test_validate_input_skipped_for_dynamic_schema(self, ctx: PluginContext) -> None:
        """validate_input=True with dynamic schema skips validation.

        Dynamic schemas accept anything, so validation is a no-op.
        """
        from elspeth.plugins.transforms.passthrough import PassThrough

        transform = PassThrough(
            {
                "schema": {"fields": "dynamic"},
                "validate_input": True,  # Would validate, but schema is dynamic
            }
        )

        # Any data passes with dynamic schema
        result = transform.process({"anything": "goes", "count": "string"}, ctx)
        assert result.status == "success"
