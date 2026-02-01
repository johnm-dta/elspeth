"""Tests for BatchReplicate aggregation transform.

BatchReplicate replicates rows based on a copies field. It is batch-aware,
meaning it receives lists of rows when aggregation triggers fire.

Contract enforcement tests verify that wrong types raise TypeError per
the Tier 2 trust model - transforms must not coerce pipeline data types.
"""

import pytest

from elspeth.plugins.context import PluginContext
from elspeth.plugins.protocols import TransformProtocol

# Common schema config for dynamic field handling (accepts any fields)
DYNAMIC_SCHEMA = {"fields": "dynamic"}


class TestBatchReplicateHappyPath:
    """Happy path tests for BatchReplicate transform."""

    @pytest.fixture
    def ctx(self) -> PluginContext:
        """Create minimal plugin context."""
        return PluginContext(run_id="test-run", config={})

    def test_implements_protocol(self) -> None:
        """BatchReplicate implements TransformProtocol."""
        from elspeth.plugins.transforms.batch_replicate import BatchReplicate

        transform = BatchReplicate(
            {
                "schema": DYNAMIC_SCHEMA,
                "copies_field": "copies",
            }
        )
        assert isinstance(transform, TransformProtocol)  # type: ignore[unreachable]

    def test_has_required_attributes(self) -> None:
        """BatchReplicate has name and is_batch_aware."""
        from elspeth.plugins.transforms.batch_replicate import BatchReplicate

        assert BatchReplicate.name == "batch_replicate"
        assert BatchReplicate.is_batch_aware is True

    def test_replicates_rows_by_copies_field(self, ctx: PluginContext) -> None:
        """BatchReplicate creates N copies of each row based on copies field."""
        from elspeth.plugins.transforms.batch_replicate import BatchReplicate

        transform = BatchReplicate(
            {
                "schema": DYNAMIC_SCHEMA,
                "copies_field": "copies",
            }
        )

        rows = [
            {"id": 1, "copies": 2},
            {"id": 2, "copies": 3},
            {"id": 3, "copies": 1},
        ]

        result = transform.process(rows, ctx)

        assert result.status == "success"
        assert result.rows is not None
        assert len(result.rows) == 6  # 2 + 3 + 1
        # Check copy indices
        assert result.rows[0]["copy_index"] == 0
        assert result.rows[1]["copy_index"] == 1
        assert result.rows[2]["copy_index"] == 0
        assert result.rows[3]["copy_index"] == 1
        assert result.rows[4]["copy_index"] == 2

    def test_uses_default_copies_when_field_missing(self, ctx: PluginContext) -> None:
        """Missing copies field uses default_copies value."""
        from elspeth.plugins.transforms.batch_replicate import BatchReplicate

        transform = BatchReplicate(
            {
                "schema": DYNAMIC_SCHEMA,
                "copies_field": "copies",
                "default_copies": 2,
            }
        )

        rows = [
            {"id": 1},  # No copies field - use default
            {"id": 2, "copies": 3},
        ]

        result = transform.process(rows, ctx)

        assert result.status == "success"
        assert result.rows is not None
        assert len(result.rows) == 5  # 2 (default) + 3

    def test_empty_batch_returns_marker(self, ctx: PluginContext) -> None:
        """Empty batch returns success with marker row."""
        from elspeth.plugins.transforms.batch_replicate import BatchReplicate

        transform = BatchReplicate(
            {
                "schema": DYNAMIC_SCHEMA,
                "copies_field": "copies",
            }
        )

        result = transform.process([], ctx)

        assert result.status == "success"
        assert result.row is not None
        assert result.row["batch_empty"] is True


class TestBatchReplicateTypeEnforcement:
    """Contract enforcement tests - transforms must not coerce types.

    Per CLAUDE.md Tier 2 trust model and docs/contracts/plugin-protocol.md:
    Transforms receive pipeline data that should already be type-validated.
    Wrong types indicate upstream bugs and must raise TypeError, not be coerced.
    """

    @pytest.fixture
    def ctx(self) -> PluginContext:
        """Create minimal plugin context."""
        return PluginContext(run_id="test-run", config={})

    def test_string_copies_raises_type_error(self, ctx: PluginContext) -> None:
        """String value in copies field raises TypeError (no coercion)."""
        from elspeth.plugins.transforms.batch_replicate import BatchReplicate

        transform = BatchReplicate(
            {
                "schema": DYNAMIC_SCHEMA,
                "copies_field": "copies",
            }
        )

        rows = [{"id": 1, "copies": "3"}]  # String "3" instead of int 3

        with pytest.raises(TypeError, match="must be int, got str"):
            transform.process(rows, ctx)

    def test_float_copies_raises_type_error(self, ctx: PluginContext) -> None:
        """Float value in copies field raises TypeError (no coercion)."""
        from elspeth.plugins.transforms.batch_replicate import BatchReplicate

        transform = BatchReplicate(
            {
                "schema": DYNAMIC_SCHEMA,
                "copies_field": "copies",
            }
        )

        rows = [{"id": 1, "copies": 3.5}]  # Float instead of int

        with pytest.raises(TypeError, match="must be int, got float"):
            transform.process(rows, ctx)

    def test_none_copies_raises_type_error(self, ctx: PluginContext) -> None:
        """None value in copies field raises TypeError."""
        from elspeth.plugins.transforms.batch_replicate import BatchReplicate

        transform = BatchReplicate(
            {
                "schema": DYNAMIC_SCHEMA,
                "copies_field": "copies",
            }
        )

        rows = [{"id": 1, "copies": None}]

        with pytest.raises(TypeError, match="must be int, got NoneType"):
            transform.process(rows, ctx)

    def test_zero_copies_raises_value_error(self, ctx: PluginContext) -> None:
        """Zero copies value raises ValueError."""
        from elspeth.plugins.transforms.batch_replicate import BatchReplicate

        transform = BatchReplicate(
            {
                "schema": DYNAMIC_SCHEMA,
                "copies_field": "copies",
            }
        )

        rows = [{"id": 1, "copies": 0}]

        with pytest.raises(ValueError, match="must be >= 1"):
            transform.process(rows, ctx)

    def test_negative_copies_raises_value_error(self, ctx: PluginContext) -> None:
        """Negative copies value raises ValueError."""
        from elspeth.plugins.transforms.batch_replicate import BatchReplicate

        transform = BatchReplicate(
            {
                "schema": DYNAMIC_SCHEMA,
                "copies_field": "copies",
            }
        )

        rows = [{"id": 1, "copies": -1}]

        with pytest.raises(ValueError, match="must be >= 1"):
            transform.process(rows, ctx)

    def test_error_message_indicates_upstream_bug(self, ctx: PluginContext) -> None:
        """Error message explicitly indicates upstream validation bug."""
        from elspeth.plugins.transforms.batch_replicate import BatchReplicate

        transform = BatchReplicate(
            {
                "schema": DYNAMIC_SCHEMA,
                "copies_field": "copies",
            }
        )

        rows = [{"id": 1, "copies": "invalid"}]

        with pytest.raises(TypeError, match="upstream validation bug"):
            transform.process(rows, ctx)


class TestBatchReplicateConfigValidation:
    """Config validation tests."""

    def test_default_copies_zero_rejected(self) -> None:
        """Config with default_copies=0 is rejected at validation time."""
        from elspeth.plugins.config_base import PluginConfigError
        from elspeth.plugins.transforms.batch_replicate import BatchReplicate

        with pytest.raises(PluginConfigError, match="default_copies"):
            BatchReplicate(
                {
                    "schema": {"fields": "dynamic"},
                    "default_copies": 0,
                }
            )

    def test_default_copies_negative_rejected(self) -> None:
        """Config with negative default_copies is rejected at validation time."""
        from elspeth.plugins.config_base import PluginConfigError
        from elspeth.plugins.transforms.batch_replicate import BatchReplicate

        with pytest.raises(PluginConfigError, match="default_copies"):
            BatchReplicate(
                {
                    "schema": {"fields": "dynamic"},
                    "default_copies": -1,
                }
            )


class TestBatchReplicateSchemaContract:
    """Schema contract tests."""

    def test_output_schema_is_dynamic_when_copy_index_enabled(self) -> None:
        """Output schema is dynamic to accommodate copy_index field."""
        from elspeth.plugins.transforms.batch_replicate import BatchReplicate

        transform = BatchReplicate(
            {
                "schema": {"fields": [{"id": "int"}], "mode": "strict"},
                "include_copy_index": True,
            }
        )

        # Output schema should accept the copy_index field (dynamic schema)
        output_schema = transform.output_schema
        # Dynamic schemas accept any fields
        validated = output_schema.model_validate({"id": 1, "copy_index": 0})
        assert validated.copy_index == 0

    def test_output_schema_accepts_copy_index_field(self) -> None:
        """Output schema validation passes for rows with copy_index."""
        from elspeth.plugins.transforms.batch_replicate import BatchReplicate

        transform = BatchReplicate(
            {
                "schema": {"fields": "dynamic"},
                "include_copy_index": True,
            }
        )

        # Simulate what process() outputs
        output_row = {"original_field": "value", "copy_index": 2}
        # This should not raise
        transform.output_schema.model_validate(output_row)
