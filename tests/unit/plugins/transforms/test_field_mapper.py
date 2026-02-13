"""Tests for FieldMapper transform."""

from pathlib import Path

import pytest

from elspeth.contracts.plugin_context import PluginContext
from elspeth.contracts.schema_contract import PipelineRow, SchemaContract
from elspeth.testing import make_field, make_pipeline_row

# Common schema config for dynamic field handling (accepts any fields)
DYNAMIC_SCHEMA = {"mode": "observed"}


class TestFieldMapper:
    """Tests for FieldMapper transform plugin."""

    @pytest.fixture
    def ctx(self) -> PluginContext:
        """Create minimal plugin context."""
        return PluginContext(run_id="test-run", config={})

    def test_has_required_attributes(self) -> None:
        """FieldMapper has name and schemas."""
        from elspeth.plugins.transforms.field_mapper import FieldMapper

        assert FieldMapper.name == "field_mapper"

    def test_rename_single_field(self, ctx: PluginContext) -> None:
        """Rename a single field."""
        from elspeth.plugins.transforms.field_mapper import FieldMapper

        transform = FieldMapper(
            {
                "schema": DYNAMIC_SCHEMA,
                "mapping": {"old_name": "new_name"},
            }
        )
        row = {"old_name": "value", "other": 123}

        result = transform.process(make_pipeline_row(row), ctx)

        assert result.status == "success"
        assert result.row is not None
        assert result.row.to_dict() == {"new_name": "value", "other": 123}
        # Original name remains accessible via contract metadata lineage.
        assert "old_name" in result.row
        assert result.row["old_name"] == "value"

    def test_rename_multiple_fields(self, ctx: PluginContext) -> None:
        """Rename multiple fields at once."""
        from elspeth.plugins.transforms.field_mapper import FieldMapper

        transform = FieldMapper(
            {
                "schema": DYNAMIC_SCHEMA,
                "mapping": {
                    "first_name": "firstName",
                    "last_name": "lastName",
                },
            }
        )
        row = {"first_name": "Alice", "last_name": "Smith", "id": 1}

        result = transform.process(make_pipeline_row(row), ctx)

        assert result.status == "success"
        assert result.row is not None
        assert result.row.to_dict() == {"firstName": "Alice", "lastName": "Smith", "id": 1}

    def test_mapping_source_can_use_original_field_name(self, ctx: PluginContext) -> None:
        """Mapping sources resolve original names through the input contract."""
        from elspeth.plugins.transforms.field_mapper import FieldMapper

        transform = FieldMapper(
            {
                "schema": DYNAMIC_SCHEMA,
                "mapping": {"Amount USD": "price"},
            }
        )
        contract = SchemaContract(
            mode="OBSERVED",
            fields=(make_field("amount_usd", float, original_name="Amount USD", required=False, source="inferred"),),
            locked=True,
        )
        row = PipelineRow({"amount_usd": 12.5}, contract)

        result = transform.process(row, ctx)

        assert result.status == "success"
        assert result.row is not None
        assert result.row.to_dict() == {"price": 12.5}

    def test_select_fields_only(self, ctx: PluginContext) -> None:
        """Only include specified fields (drop others)."""
        from elspeth.plugins.transforms.field_mapper import FieldMapper

        transform = FieldMapper(
            {
                "schema": DYNAMIC_SCHEMA,
                "mapping": {"id": "id", "name": "name"},
                "select_only": True,
            }
        )
        row = {"id": 1, "name": "alice", "secret": "password", "extra": "data"}

        result = transform.process(make_pipeline_row(row), ctx)

        assert result.status == "success"
        assert result.row is not None
        assert result.row.to_dict() == {"id": 1, "name": "alice"}
        assert "secret" not in result.row
        assert "extra" not in result.row

    def test_missing_field_error(self, ctx: PluginContext) -> None:
        """Error when required field is missing and strict mode enabled."""
        from elspeth.plugins.transforms.field_mapper import FieldMapper

        transform = FieldMapper(
            {
                "schema": DYNAMIC_SCHEMA,
                "mapping": {"required_field": "output"},
                "strict": True,
            }
        )
        row = {"other_field": "value"}

        result = transform.process(make_pipeline_row(row), ctx)

        assert result.status == "error"
        assert "required_field" in str(result.reason)

    def test_missing_field_skip_non_strict(self, ctx: PluginContext) -> None:
        """Skip missing fields when strict mode disabled."""
        from elspeth.plugins.transforms.field_mapper import FieldMapper

        transform = FieldMapper(
            {
                "schema": DYNAMIC_SCHEMA,
                "mapping": {"maybe_field": "output"},
                "strict": False,
            }
        )
        row = {"other_field": "value"}

        result = transform.process(make_pipeline_row(row), ctx)

        assert result.status == "success"
        assert result.row is not None
        assert result.row.to_dict() == {"other_field": "value"}
        assert "output" not in result.row

    def test_default_is_non_strict(self, ctx: PluginContext) -> None:
        """Default behavior is non-strict (skip missing)."""
        from elspeth.plugins.transforms.field_mapper import FieldMapper

        transform = FieldMapper(
            {
                "schema": DYNAMIC_SCHEMA,
                "mapping": {"missing": "output"},
            }
        )
        row = {"exists": "value"}

        result = transform.process(make_pipeline_row(row), ctx)

        assert result.status == "success"

    def test_nested_field_access(self, ctx: PluginContext) -> None:
        """Access nested fields with dot notation."""
        from elspeth.plugins.transforms.field_mapper import FieldMapper

        transform = FieldMapper(
            {
                "schema": DYNAMIC_SCHEMA,
                "mapping": {"meta.source": "origin"},
            }
        )
        row = {"id": 1, "meta": {"source": "api", "timestamp": 123}}

        result = transform.process(make_pipeline_row(row), ctx)

        assert result.status == "success"
        assert result.row is not None
        assert result.row["origin"] == "api"
        assert "meta" in result.row  # Original nested structure preserved

    def test_nested_field_type_mismatch_raises_in_non_strict_mode(self, ctx: PluginContext) -> None:
        """Non-dict intermediate on dotted path raises type error (upstream bug)."""
        from elspeth.plugins.transforms.field_mapper import FieldMapper

        transform = FieldMapper(
            {
                "schema": DYNAMIC_SCHEMA,
                "mapping": {"user.name": "origin"},
                "strict": False,
            }
        )

        with pytest.raises(TypeError, match="expected dict"):
            transform.process(make_pipeline_row({"user": "string_not_dict"}), ctx)

    def test_nested_field_type_mismatch_raises_in_strict_mode(self, ctx: PluginContext) -> None:
        """Strict mode should not mask type mismatches as missing fields."""
        from elspeth.plugins.transforms.field_mapper import FieldMapper

        transform = FieldMapper(
            {
                "schema": DYNAMIC_SCHEMA,
                "mapping": {"user.name": "origin"},
                "strict": True,
            }
        )

        with pytest.raises(TypeError, match="expected dict"):
            transform.process(make_pipeline_row({"user": "string_not_dict"}), ctx)

    def test_empty_mapping_passthrough(self, ctx: PluginContext) -> None:
        """Empty mapping acts as passthrough."""
        from elspeth.plugins.transforms.field_mapper import FieldMapper

        transform = FieldMapper(
            {
                "schema": DYNAMIC_SCHEMA,
                "mapping": {},
            }
        )
        row = {"a": 1, "b": 2}

        result = transform.process(make_pipeline_row(row), ctx)

        assert result.status == "success"
        assert result.row is not None
        assert result.row.to_dict() == row

    def test_requires_schema_config(self) -> None:
        """FieldMapper requires schema configuration."""
        from elspeth.plugins.config_base import PluginConfigError
        from elspeth.plugins.transforms.field_mapper import FieldMapper

        with pytest.raises(PluginConfigError, match="schema"):
            FieldMapper({"mapping": {"a": "b"}})

    def test_validate_input_rejects_wrong_type(self, ctx: PluginContext) -> None:
        """validate_input=True crashes on wrong types (upstream bug).

        Per three-tier trust model: transforms use allow_coercion=False,
        so string "42" is NOT coerced to int 42 - it raises ValidationError.
        """
        from pydantic import ValidationError

        from elspeth.plugins.transforms.field_mapper import FieldMapper

        transform = FieldMapper(
            {
                "schema": {"mode": "fixed", "fields": ["count: int"]},
                "mapping": {},
                "validate_input": True,
            }
        )

        with pytest.raises(ValidationError):
            transform.process(make_pipeline_row({"count": "not_an_int"}), ctx)

    def test_validate_input_disabled_passes_wrong_type(self, ctx: PluginContext) -> None:
        """validate_input=False (default) passes wrong types through.

        When validation is disabled, the transform doesn't check types.
        This is the default to avoid breaking existing pipelines.
        """
        from elspeth.plugins.transforms.field_mapper import FieldMapper

        transform = FieldMapper(
            {
                "schema": {"mode": "fixed", "fields": ["count: int"]},
                "mapping": {},
                "validate_input": False,  # Explicit default
            }
        )

        # String passes through without validation
        result = transform.process(make_pipeline_row({"count": "not_an_int"}), ctx)
        assert result.status == "success"
        assert result.row is not None
        assert result.row["count"] == "not_an_int"

    def test_validate_input_skipped_for_dynamic_schema(self, ctx: PluginContext) -> None:
        """validate_input=True with dynamic schema skips validation.

        Dynamic schemas accept anything, so validation is a no-op.
        """
        from elspeth.plugins.transforms.field_mapper import FieldMapper

        transform = FieldMapper(
            {
                "schema": {"mode": "observed"},
                "mapping": {},
                "validate_input": True,  # Would validate, but schema is dynamic
            }
        )

        # Any data passes with dynamic schema
        result = transform.process(make_pipeline_row({"anything": "goes", "count": "string"}), ctx)
        assert result.status == "success"


class TestFieldMapperOutputSchema:
    """Tests for output schema behavior of shape-changing transforms.

    Per P1-2026-01-19-shape-changing-transforms-output-schema-mismatch:
    Shape-changing transforms must use dynamic output_schema because their
    output shape depends on config (mapping, select_only), not input schema.
    """

    def test_select_only_uses_dynamic_output_schema(self) -> None:
        """FieldMapper with select_only=True uses dynamic output_schema.

        When select_only=True, the output only includes mapped fields,
        which depends on config, not the input schema. Therefore output_schema
        must be dynamic (accepts any fields) to avoid false schema validation.
        """
        from elspeth.plugins.transforms.field_mapper import FieldMapper

        # Explicit schema: expects a, b, c
        transform = FieldMapper(
            {
                "schema": {"mode": "fixed", "fields": ["a: str", "b: int", "c: float"]},
                "mapping": {"a": "a"},  # Only select field 'a'
                "select_only": True,
            }
        )

        # Output schema should be dynamic (accepts any fields)
        # because output shape depends on mapping config, not input schema
        output_fields = transform.output_schema.model_fields

        # The fix: output_schema should be dynamic (empty required fields, extra="allow")
        # Currently fails because output_schema = input_schema, which has a, b, c
        assert len(output_fields) == 0, f"Expected dynamic schema with no required fields, got: {list(output_fields.keys())}"

        # Additionally verify extra fields are allowed (dynamic schema behavior)
        config = transform.output_schema.model_config
        assert config.get("extra") == "allow", "Output schema should allow extra fields (dynamic)"


class TestFieldMapperContractPropagation:
    """Tests for FieldMapper contract propagation."""

    @pytest.fixture
    def ctx(self) -> PluginContext:
        """Create minimal plugin context."""
        return PluginContext(run_id="test-run", config={})

    def test_contract_contains_renamed_field(self, ctx: PluginContext) -> None:
        """Output contract contains renamed field, not original field name."""
        from elspeth.plugins.transforms.field_mapper import FieldMapper

        transform = FieldMapper(
            {
                "schema": DYNAMIC_SCHEMA,
                "mapping": {"old_field": "new_field"},
            }
        )

        row = make_pipeline_row({"old_field": "value", "other": 42})
        result = transform.process(row, ctx)

        assert result.status == "success"
        assert isinstance(result.row, PipelineRow)

        field_names = {f.normalized_name for f in result.row.contract.fields}
        assert "new_field" in field_names
        assert "old_field" not in field_names
        assert "other" in field_names

    def test_contract_reflects_field_removal(self, ctx: PluginContext) -> None:
        """Output contract doesn't contain removed fields when select_only=True."""
        from elspeth.plugins.transforms.field_mapper import FieldMapper

        transform = FieldMapper(
            {
                "schema": DYNAMIC_SCHEMA,
                "mapping": {"keep_me": "kept"},
                "select_only": True,
            }
        )

        row = make_pipeline_row({"keep_me": "value", "remove_me": 42, "also_remove": "bye"})
        result = transform.process(row, ctx)

        assert result.status == "success"
        assert isinstance(result.row, PipelineRow)

        field_names = {f.normalized_name for f in result.row.contract.fields}
        assert field_names == {"kept"}  # Only the mapped field should remain

    def test_downstream_can_access_renamed_field(self, ctx: PluginContext) -> None:
        """Downstream transforms can access renamed fields via contract."""
        from elspeth.plugins.transforms.field_mapper import FieldMapper

        mapper = FieldMapper(
            {
                "schema": DYNAMIC_SCHEMA,
                "mapping": {"source": "target"},
            }
        )

        row = make_pipeline_row({"source": "value", "other": 42})
        result = mapper.process(row, ctx)

        assert result.status == "success"
        assert result.row is not None
        assert isinstance(result.row, PipelineRow)

        # result.row IS already a PipelineRow with contract
        output_row = result.row

        # Downstream access via contract should work
        assert output_row["target"] == "value"
        assert output_row["other"] == 42

        # Original field name remains accessible via contract lineage.
        assert output_row["source"] == "value"

    def test_renamed_field_preserves_original_name_metadata(self, ctx: PluginContext) -> None:
        """Renamed fields preserve source original_name lineage in contract."""
        from elspeth.plugins.transforms.field_mapper import FieldMapper

        transform = FieldMapper(
            {
                "schema": DYNAMIC_SCHEMA,
                "mapping": {"amount_usd": "price"},
            }
        )

        input_contract = SchemaContract(
            mode="OBSERVED",
            fields=(
                make_field("amount_usd", float, original_name="Amount USD", required=True, source="declared"),
                make_field("other", int, original_name="Other", required=False, source="inferred"),
            ),
            locked=True,
        )
        row = PipelineRow({"amount_usd": 12.5, "other": 1}, input_contract)

        result = transform.process(row, ctx)
        assert result.status == "success"
        assert isinstance(result.row, PipelineRow)

        renamed = result.row.contract.get_field("price")
        assert renamed is not None
        assert renamed.original_name == "Amount USD"
        assert renamed.python_type is float
        assert renamed.required is True
        assert renamed.source == "declared"

    def test_headers_original_uses_preserved_source_name_after_rename(self, tmp_path: Path, ctx: PluginContext) -> None:
        """Sink headers: original emits source header after FieldMapper rename."""
        from elspeth.plugins.sinks.csv_sink import CSVSink
        from elspeth.plugins.transforms.field_mapper import FieldMapper

        transform = FieldMapper(
            {
                "schema": DYNAMIC_SCHEMA,
                "mapping": {"amount_usd": "price"},
            }
        )

        input_contract = SchemaContract(
            mode="OBSERVED",
            fields=(make_field("amount_usd", float, original_name="Amount USD", required=True, source="declared"),),
            locked=True,
        )
        result = transform.process(PipelineRow({"amount_usd": 12.5}, input_contract), ctx)
        assert result.status == "success"
        assert isinstance(result.row, PipelineRow)

        output_path = tmp_path / "output.csv"
        sink = CSVSink(
            {
                "path": str(output_path),
                "schema": {"mode": "observed"},
                "headers": "original",
            }
        )
        sink_ctx = PluginContext(
            run_id="test-run",
            config={},
            contract=result.row.contract,
        )
        sink.write([result.row.to_dict()], sink_ctx)
        sink.close()

        header = output_path.read_text().splitlines()[0]
        assert header == "Amount USD"
