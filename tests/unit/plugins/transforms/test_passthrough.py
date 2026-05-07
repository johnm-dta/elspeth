"""Tests for PassThrough transform."""

from typing import Any

import pytest

from elspeth.contracts.plugin_context import PluginContext
from elspeth.testing import make_pipeline_row
from tests.fixtures.factories import make_source_context

# Common schema config for dynamic field handling (accepts any fields)
DYNAMIC_SCHEMA = {"mode": "observed"}


class TestPassThrough:
    """Tests for PassThrough transform plugin."""

    @pytest.fixture
    def ctx(self) -> PluginContext:
        """Create minimal plugin context."""
        return make_source_context()

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

        result = transform.process(make_pipeline_row(row), ctx)

        assert result.status == "success"
        assert result.row is not None
        assert result.row.to_dict() == row
        assert id(result.row) != id(row)  # Should be a copy, not the same object

    def test_process_with_nested_data(self, ctx: PluginContext) -> None:
        """Handles nested structures correctly."""
        from elspeth.plugins.transforms.passthrough import PassThrough

        transform = PassThrough({"schema": DYNAMIC_SCHEMA})
        row: dict[str, Any] = {"id": 1, "meta": {"source": "test", "tags": ["a", "b"]}}

        result = transform.process(make_pipeline_row(row), ctx)

        assert result.status == "success"
        assert result.row is not None
        assert result.row.to_dict() == row
        # Nested structures should be deep copied
        assert result.row["meta"] is not row["meta"]
        assert result.row["meta"]["tags"] is not row["meta"]["tags"]

    def test_process_with_empty_row(self, ctx: PluginContext) -> None:
        """Handles empty row."""
        from elspeth.plugins.transforms.passthrough import PassThrough

        transform = PassThrough({"schema": DYNAMIC_SCHEMA})
        row: dict[str, Any] = {}

        result = transform.process(make_pipeline_row(row), ctx)

        assert result.status == "success"
        assert result.row is not None
        assert result.row.to_dict() == {}

    def test_close_is_idempotent(self) -> None:
        """close() can be called multiple times."""
        from elspeth.plugins.transforms.passthrough import PassThrough

        transform = PassThrough({"schema": DYNAMIC_SCHEMA})
        transform.close()
        transform.close()  # Should not raise

    def test_requires_schema_config(self) -> None:
        """PassThrough requires schema configuration."""
        from elspeth.plugins.infrastructure.config_base import PluginConfigError
        from elspeth.plugins.transforms.passthrough import PassThrough

        with pytest.raises(PluginConfigError, match="schema"):
            PassThrough({})

    def test_no_validate_input_attribute(self) -> None:
        """PassThrough does not carry a validate_input attribute.

        Input validation is unconditional in the executor — plugins
        no longer control this via a flag.
        """
        from elspeth.plugins.transforms.passthrough import PassThrough

        transform = PassThrough(
            {
                "schema": {"mode": "fixed", "fields": ["count: int"]},
            }
        )

        assert not hasattr(transform, "validate_input")

    def test_passthrough_validates_schema_compatibility(self) -> None:
        """PassThrough should validate schema is self-compatible."""
        from elspeth.plugins.transforms.passthrough import PassThrough

        # Valid: Dynamic schema (always compatible with itself)
        dynamic_config = {"schema": {"mode": "observed"}}
        PassThrough(dynamic_config)  # Should succeed

        # Valid: Explicit schema (compatible with itself)
        explicit_config = {"schema": {"mode": "fixed", "fields": ["id: int", "name: str"]}}
        PassThrough(explicit_config)  # Should succeed

        # Note: PassThrough has same input/output schema, so always compatible
        # More complex transforms tested separately

    def test_fixed_schema_initializes_output_schema_config_and_aligns_output_contract(self, ctx: PluginContext) -> None:
        """Declared fixed schemas must drive runtime output contracts."""
        from elspeth.plugins.transforms.passthrough import PassThrough

        transform = PassThrough(
            {
                "schema": {
                    "mode": "fixed",
                    "fields": ["id: int", "name: str"],
                },
            }
        )
        row = make_pipeline_row({"id": 1, "name": "alice"})

        result = transform.process(row, ctx)

        assert transform._output_schema_config is not None
        assert transform._output_schema_config.mode == "fixed"
        assert result.row is not None
        assert result.row.contract.mode == "FIXED"
        assert result.row.contract.locked is True
