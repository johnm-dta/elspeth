"""Tests for BatchReplicate aggregation transform.

BatchReplicate replicates rows based on a copies field. It is batch-aware,
meaning it receives lists of rows when aggregation triggers fire.

Contract enforcement tests verify that wrong types raise TypeError per
the Tier 2 trust model - transforms must not coerce pipeline data types.
"""

import pytest

from elspeth.contracts.plugin_context import PluginContext
from elspeth.testing import make_pipeline_row
from tests.fixtures.factories import make_context

# Common schema config for dynamic field handling (accepts any fields)
DYNAMIC_SCHEMA = {"mode": "observed"}


class TestBatchReplicateHappyPath:
    """Happy path tests for BatchReplicate transform."""

    @pytest.fixture
    def ctx(self) -> PluginContext:
        """Create minimal plugin context."""
        return make_context()

    def test_has_required_attributes(self) -> None:
        """BatchReplicate has name and is_batch_aware."""
        from elspeth.plugins.transforms.batch_replicate import BatchReplicate

        assert BatchReplicate.name == "batch_replicate"
        assert BatchReplicate.is_batch_aware is True

    def test_is_not_declared_pass_through_when_invalid_rows_can_be_quarantined(self) -> None:
        """Mixed-validity batches make unconditional pass-through dishonest.

        ``process()`` can quarantine some inputs while still succeeding for the
        rest of the batch, so the class-level declaration must stay
        conservative until ELSPETH has a more precise partial-pass-through
        contract.
        """
        from elspeth.plugins.transforms.batch_replicate import BatchReplicate

        assert BatchReplicate.passes_through_input is False

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
            make_pipeline_row({"id": 1, "copies": 2}),
            make_pipeline_row({"id": 2, "copies": 3}),
            make_pipeline_row({"id": 3, "copies": 1}),
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
            make_pipeline_row({"id": 1}),  # No copies field - use default
            make_pipeline_row({"id": 2, "copies": 3}),
        ]

        result = transform.process(rows, ctx)

        assert result.status == "success"
        assert result.rows is not None
        assert len(result.rows) == 5  # 2 (default) + 3

    def test_empty_batch_returns_error(self, ctx: PluginContext) -> None:
        """Empty batch returns error — not fabricated data."""
        from elspeth.plugins.transforms.batch_replicate import BatchReplicate

        transform = BatchReplicate(
            {
                "schema": DYNAMIC_SCHEMA,
                "copies_field": "copies",
            }
        )

        result = transform.process([], ctx)

        assert result.status == "error"
        assert result.reason is not None
        assert result.reason["reason"] == "empty_batch"
        assert not result.retryable


class TestBatchReplicateTypeEnforcement:
    """Contract enforcement tests - transforms must not coerce types.

    Per CLAUDE.md Tier 2 trust model and docs/contracts/plugin-protocol.md:
    Transforms receive pipeline data that should already be type-validated.
    Wrong types indicate upstream bugs and must raise TypeError, not be coerced.
    """

    @pytest.fixture
    def ctx(self) -> PluginContext:
        """Create minimal plugin context."""
        return make_context()

    def test_string_copies_raises_type_error(self, ctx: PluginContext) -> None:
        """String value in copies field raises TypeError (no coercion)."""
        from elspeth.plugins.transforms.batch_replicate import BatchReplicate

        transform = BatchReplicate(
            {
                "schema": DYNAMIC_SCHEMA,
                "copies_field": "copies",
            }
        )

        rows = [make_pipeline_row({"id": 1, "copies": "3"})]  # String "3" instead of int 3

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

        rows = [make_pipeline_row({"id": 1, "copies": 3.5})]  # Float instead of int

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

        rows = [make_pipeline_row({"id": 1, "copies": None})]

        with pytest.raises(TypeError, match="must be int, got NoneType"):
            transform.process(rows, ctx)

    def test_bool_true_copies_raises_type_error(self, ctx: PluginContext) -> None:
        """Bool True in copies field raises TypeError (not silently treated as 1).

        Python's isinstance(True, int) returns True because bool is a subclass
        of int. The fix uses `type(x) is int` for strict type checking, so
        True/False are rejected as distinct logical types.
        """
        from elspeth.plugins.transforms.batch_replicate import BatchReplicate

        transform = BatchReplicate(
            {
                "schema": DYNAMIC_SCHEMA,
                "copies_field": "copies",
            }
        )

        rows = [make_pipeline_row({"id": 1, "copies": True})]

        with pytest.raises(TypeError, match="must be int, got bool"):
            transform.process(rows, ctx)

    def test_bool_false_copies_raises_type_error(self, ctx: PluginContext) -> None:
        """Bool False in copies field raises TypeError (not silently treated as 0).

        Without strict type checking, False would be treated as 0 copies,
        which would then be quarantined as invalid (< 1). The bug is that
        the type check passes at all - bool is not int for contract purposes.
        """
        from elspeth.plugins.transforms.batch_replicate import BatchReplicate

        transform = BatchReplicate(
            {
                "schema": DYNAMIC_SCHEMA,
                "copies_field": "copies",
            }
        )

        rows = [make_pipeline_row({"id": 1, "copies": False})]

        with pytest.raises(TypeError, match="must be int, got bool"):
            transform.process(rows, ctx)

    def test_zero_copies_returns_error_when_all_invalid(self, ctx: PluginContext) -> None:
        """All rows with zero copies returns error result (no valid output to expand)."""
        from elspeth.plugins.transforms.batch_replicate import BatchReplicate

        transform = BatchReplicate(
            {
                "schema": DYNAMIC_SCHEMA,
                "copies_field": "copies",
            }
        )

        rows = [make_pipeline_row({"id": 1, "copies": 0})]

        result = transform.process(rows, ctx)
        assert result.status == "error"
        assert result.reason is not None
        assert result.reason["reason"] == "all_rows_failed"
        assert "1 rows quarantined" in result.reason["error"]
        assert result.reason["row_errors"][0]["reason"] == "invalid_copies"

    def test_negative_copies_returns_error_when_all_invalid(self, ctx: PluginContext) -> None:
        """All rows with negative copies returns error result."""
        from elspeth.plugins.transforms.batch_replicate import BatchReplicate

        transform = BatchReplicate(
            {
                "schema": DYNAMIC_SCHEMA,
                "copies_field": "copies",
            }
        )

        rows = [make_pipeline_row({"id": 1, "copies": -1})]

        result = transform.process(rows, ctx)
        assert result.status == "error"
        assert result.reason is not None
        assert result.reason["reason"] == "all_rows_failed"
        assert result.reason["row_errors"][0]["reason"] == "invalid_copies"

    def test_invalid_copies_quarantined_alongside_valid_rows(self, ctx: PluginContext) -> None:
        """Rows with invalid copies are quarantined; valid rows still replicated."""
        from elspeth.plugins.transforms.batch_replicate import BatchReplicate

        transform = BatchReplicate(
            {
                "schema": DYNAMIC_SCHEMA,
                "copies_field": "copies",
            }
        )

        rows = [
            make_pipeline_row({"id": 1, "copies": -1}),
            make_pipeline_row({"id": 2, "copies": 2}),
        ]

        result = transform.process(rows, ctx)
        assert result.status == "success"
        assert result.rows is not None
        assert len(result.rows) == 2  # Only valid copies of row 2
        assert result.rows[0]["id"] == 2
        assert result.rows[0]["copy_index"] == 0
        assert result.rows[1]["id"] == 2
        assert result.rows[1]["copy_index"] == 1
        # Quarantine info in success_reason.metadata
        assert result.success_reason is not None
        assert result.success_reason["metadata"]["quarantined_count"] == 1
        assert result.success_reason["metadata"]["quarantined"][0]["reason"] == "invalid_copies"
        assert result.success_reason["metadata"]["quarantined"][0]["row_data"]["id"] == 1

    def test_error_message_indicates_upstream_bug(self, ctx: PluginContext) -> None:
        """Error message explicitly indicates upstream validation bug."""
        from elspeth.plugins.transforms.batch_replicate import BatchReplicate

        transform = BatchReplicate(
            {
                "schema": DYNAMIC_SCHEMA,
                "copies_field": "copies",
            }
        )

        rows = [make_pipeline_row({"id": 1, "copies": "invalid"})]

        with pytest.raises(TypeError, match="upstream validation bug"):
            transform.process(rows, ctx)


class TestBatchReplicateConfigValidation:
    """Config validation tests."""

    def test_default_copies_zero_rejected(self) -> None:
        """Config with default_copies=0 is rejected at validation time."""
        from elspeth.plugins.infrastructure.config_base import PluginConfigError
        from elspeth.plugins.transforms.batch_replicate import BatchReplicate

        with pytest.raises(PluginConfigError, match="default_copies"):
            BatchReplicate(
                {
                    "schema": {"mode": "observed"},
                    "default_copies": 0,
                }
            )

    def test_default_copies_negative_rejected(self) -> None:
        """Config with negative default_copies is rejected at validation time."""
        from elspeth.plugins.infrastructure.config_base import PluginConfigError
        from elspeth.plugins.transforms.batch_replicate import BatchReplicate

        with pytest.raises(PluginConfigError, match="default_copies"):
            BatchReplicate(
                {
                    "schema": {"mode": "observed"},
                    "default_copies": -1,
                }
            )

    def test_default_copies_above_max_rejected(self) -> None:
        """Config with default_copies > max_copies is rejected at validation time."""
        from elspeth.plugins.infrastructure.config_base import PluginConfigError
        from elspeth.plugins.transforms.batch_replicate import BatchReplicate

        with pytest.raises(PluginConfigError, match="exceeds max_copies"):
            BatchReplicate(
                {
                    "schema": {"mode": "observed"},
                    "default_copies": 11,
                    "max_copies": 10,
                }
            )

    def test_explicit_schema_declaring_copy_index_is_rejected(self) -> None:
        """Fixed/flexible schemas must fail closed when copy_index would collide."""
        from elspeth.plugins.infrastructure.config_base import PluginConfigError
        from elspeth.plugins.transforms.batch_replicate import BatchReplicate

        with pytest.raises(PluginConfigError, match="copy_index"):
            BatchReplicate(
                {
                    "schema": {
                        "mode": "fixed",
                        "fields": [
                            "id: int",
                            "copies: int",
                            "copy_index: int",
                        ],
                    },
                    "include_copy_index": True,
                }
            )


class TestBatchReplicateSchemaContract:
    """Schema contract tests."""

    def test_output_schema_is_observed_when_copy_index_enabled(self) -> None:
        """Output schema is dynamic to accommodate copy_index field."""
        from elspeth.plugins.transforms.batch_replicate import BatchReplicate

        transform = BatchReplicate(
            {
                "schema": {"fields": [{"id": "int"}], "mode": "fixed"},
                "include_copy_index": True,
            }
        )

        # Output schema should accept the copy_index field (dynamic schema)
        output_schema = transform.output_schema
        # Dynamic schemas accept any fields
        validated = output_schema.model_validate({"id": 1, "copy_index": 0})
        assert validated.copy_index == 0  # type: ignore[attr-defined]

    def test_output_schema_accepts_copy_index_field(self) -> None:
        """Output schema validation passes for rows with copy_index."""
        from elspeth.plugins.transforms.batch_replicate import BatchReplicate

        transform = BatchReplicate(
            {
                "schema": {"mode": "observed"},
                "include_copy_index": True,
            }
        )

        # Simulate what process() outputs
        output_row = {"original_field": "value", "copy_index": 2}
        # This should not raise
        transform.output_schema.model_validate(output_row)


class TestBatchReplicateDeepCopy:
    """Verify replicated rows don't share mutable nested references."""

    @pytest.fixture
    def ctx(self) -> PluginContext:
        return make_context()

    def test_nested_list_mutation_does_not_cross_contaminate(self, ctx: PluginContext) -> None:
        """Mutating a nested list in one copy must not affect other copies."""
        from elspeth.plugins.transforms.batch_replicate import BatchReplicate

        transform = BatchReplicate({"schema": DYNAMIC_SCHEMA, "copies_field": "copies", "include_copy_index": True})
        row = make_pipeline_row({"id": 1, "tags": ["a", "b"], "copies": 3})

        result = transform.process([row], ctx)

        assert result.status == "success"
        assert result.rows is not None
        assert len(result.rows) == 3

        # Mutate nested list in first copy
        first = result.rows[0].to_dict()
        first["tags"].append("MUTATED")

        # Other copies must be unaffected
        for copy_row in result.rows[1:]:
            assert "MUTATED" not in copy_row["tags"]

    def test_nested_dict_mutation_does_not_cross_contaminate(self, ctx: PluginContext) -> None:
        """Mutating a nested dict in one copy must not affect other copies."""
        from elspeth.plugins.transforms.batch_replicate import BatchReplicate

        transform = BatchReplicate({"schema": DYNAMIC_SCHEMA, "copies_field": "copies", "include_copy_index": True})
        row = make_pipeline_row({"id": 1, "meta": {"source": "csv"}, "copies": 2})

        result = transform.process([row], ctx)

        assert result.rows is not None
        first = result.rows[0].to_dict()
        first["meta"]["injected"] = True

        second = result.rows[1].to_dict()
        assert "injected" not in second["meta"]

    def test_runtime_collision_on_copy_index_raises_plugin_contract_violation(self, ctx: PluginContext) -> None:
        """Observed batches must reject incoming copy_index instead of overwriting it."""
        from elspeth.contracts.errors import PluginContractViolation
        from elspeth.plugins.transforms.batch_replicate import BatchReplicate

        transform = BatchReplicate({"schema": DYNAMIC_SCHEMA, "copies_field": "copies", "include_copy_index": True})
        rows = [make_pipeline_row({"id": 1, "copies": 2, "copy_index": "source-value"})]

        with pytest.raises(PluginContractViolation, match=r"would overwrite existing input fields \['copy_index'\]"):
            transform.process(rows, ctx)


class TestBatchReplicateDeclaredOutputFields:
    """Tests for declared_output_fields — centralized collision detection support.

    Field collision detection is enforced centrally by TransformExecutor
    (see TestTransformExecutor in test_executors.py). These tests verify
    that BatchReplicate correctly declares its output fields so the executor
    can perform pre-execution collision checks.
    """

    def test_declared_output_fields_contains_copy_index_when_enabled(self) -> None:
        """declared_output_fields includes copy_index when include_copy_index is True."""
        from elspeth.plugins.transforms.batch_replicate import BatchReplicate

        transform = BatchReplicate(
            {
                "schema": DYNAMIC_SCHEMA,
                "copies_field": "copies",
                "include_copy_index": True,
            }
        )

        assert "copy_index" in transform.declared_output_fields

    def test_declared_output_fields_empty_when_copy_index_disabled(self) -> None:
        """declared_output_fields is empty when include_copy_index is False."""
        from elspeth.plugins.transforms.batch_replicate import BatchReplicate

        transform = BatchReplicate(
            {
                "schema": DYNAMIC_SCHEMA,
                "copies_field": "copies",
                "include_copy_index": False,
            }
        )

        assert len(transform.declared_output_fields) == 0

    def test_declared_output_fields_drives_schema_evolution(self) -> None:
        """declared_output_fields is non-empty when include_copy_index=True, enabling schema evolution."""
        from elspeth.plugins.transforms.batch_replicate import BatchReplicate

        transform = BatchReplicate(
            {
                "schema": DYNAMIC_SCHEMA,
                "copies_field": "copies",
                "include_copy_index": True,
            }
        )

        assert transform.declared_output_fields


class TestOutputSchemaConfig:
    def test_guaranteed_fields_with_copy_index(self):
        from elspeth.plugins.transforms.batch_replicate import BatchReplicate

        transform = BatchReplicate(
            {
                "include_copy_index": True,
                "schema": {"mode": "observed"},
            }
        )
        assert transform._output_schema_config is not None
        assert frozenset(transform._output_schema_config.guaranteed_fields) == frozenset({"copy_index"})

    def test_guaranteed_fields_without_copy_index(self):
        from elspeth.plugins.transforms.batch_replicate import BatchReplicate

        transform = BatchReplicate(
            {
                "include_copy_index": False,
                "schema": {"mode": "observed"},
            }
        )
        assert transform._output_schema_config is not None
        # No upstream guaranteed_fields + no declared fields → abstain (None)
        assert transform._output_schema_config.guaranteed_fields is None


class TestBatchReplicateContractPreservation:
    """Tests that output rows preserve input contract metadata.

    Bug fix: BatchReplicate was building SchemaContract from scratch with
    mode=OBSERVED, python_type=object, source=inferred for ALL keys. This
    discards the input contracts' mode, original_name, python_type, and field types.
    The fix merges input row contracts to preserve metadata.
    """

    @pytest.fixture
    def ctx(self) -> PluginContext:
        return make_context()

    def test_output_contract_mode_aligns_to_declared_output_schema(self, ctx: PluginContext) -> None:
        """Emitted contract mode follows the transform's declared output schema."""
        from elspeth.contracts.schema_contract import FieldContract, PipelineRow, SchemaContract
        from elspeth.plugins.transforms.batch_replicate import BatchReplicate

        transform = BatchReplicate(
            {
                "schema": DYNAMIC_SCHEMA,
                "copies_field": "copies",
                "include_copy_index": True,
            }
        )

        contract = SchemaContract(
            mode="FIXED",
            fields=(
                FieldContract(normalized_name="id", original_name="id", python_type=int, required=True, source="declared"),
                FieldContract(normalized_name="copies", original_name="copies", python_type=int, required=True, source="declared"),
            ),
            locked=True,
        )
        rows = [PipelineRow({"id": 1, "copies": 2}, contract)]

        result = transform.process(rows, ctx)

        assert result.status == "success"
        assert result.rows is not None
        output_contract = result.rows[0].contract
        assert output_contract.mode == "OBSERVED"
        assert output_contract.locked is True

    def test_output_preserves_field_python_type(self, ctx: PluginContext) -> None:
        """Output contract fields preserve python_type from input contract."""
        from elspeth.contracts.schema_contract import FieldContract, PipelineRow, SchemaContract
        from elspeth.plugins.transforms.batch_replicate import BatchReplicate

        transform = BatchReplicate(
            {
                "schema": DYNAMIC_SCHEMA,
                "copies_field": "copies",
                "include_copy_index": True,
            }
        )

        contract = SchemaContract(
            mode="OBSERVED",
            fields=(
                FieldContract(normalized_name="id", original_name="ID", python_type=int, required=True, source="declared"),
                FieldContract(normalized_name="copies", original_name="copies", python_type=int, required=False, source="inferred"),
            ),
            locked=True,
        )
        rows = [PipelineRow({"id": 1, "copies": 1}, contract)]

        result = transform.process(rows, ctx)

        assert result.status == "success"
        assert result.rows is not None
        output_contract = result.rows[0].contract

        id_field = output_contract.get_field("id")
        assert id_field is not None
        assert id_field.python_type is int
        assert id_field.original_name == "ID"
        assert id_field.source == "declared"

    def test_output_copy_index_has_int_type(self, ctx: PluginContext) -> None:
        """copy_index field in output contract has python_type=int, not object."""
        from elspeth.contracts.schema_contract import FieldContract, PipelineRow, SchemaContract
        from elspeth.plugins.transforms.batch_replicate import BatchReplicate

        transform = BatchReplicate(
            {
                "schema": DYNAMIC_SCHEMA,
                "copies_field": "copies",
                "include_copy_index": True,
            }
        )

        contract = SchemaContract(
            mode="OBSERVED",
            fields=(
                FieldContract(normalized_name="id", original_name="id", python_type=int, required=False, source="inferred"),
                FieldContract(normalized_name="copies", original_name="copies", python_type=int, required=False, source="inferred"),
            ),
            locked=True,
        )
        rows = [PipelineRow({"id": 1, "copies": 2}, contract)]

        result = transform.process(rows, ctx)

        assert result.status == "success"
        assert result.rows is not None
        copy_idx_field = result.rows[0].contract.get_field("copy_index")
        assert copy_idx_field is not None
        assert copy_idx_field.python_type is int
