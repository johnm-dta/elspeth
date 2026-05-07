"""Tests for FieldMapper transform."""

from pathlib import Path

import pytest

from elspeth.contracts.plugin_context import PluginContext
from elspeth.contracts.schema_contract import PipelineRow, SchemaContract
from elspeth.testing import make_field, make_pipeline_row
from tests.fixtures.base_classes import inject_write_failure
from tests.fixtures.factories import make_context
from tests.fixtures.landscape import make_factory

# Common schema config for dynamic field handling (accepts any fields)
DYNAMIC_SCHEMA = {"mode": "observed"}


class TestFieldMapper:
    """Tests for FieldMapper transform plugin."""

    @pytest.fixture
    def ctx(self) -> PluginContext:
        """Create minimal plugin context."""
        return make_context()

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

    def test_backward_probe_rows_drop_mapped_source_field(self, ctx: PluginContext) -> None:
        """Backward invariant probe keeps the real rename/drop path under test."""
        from elspeth.plugins.transforms.field_mapper import FieldMapper

        transform = FieldMapper(FieldMapper.probe_config())
        probe = make_pipeline_row({"baseline": "kept"})

        result = transform.execute_backward_invariant_probe(
            transform.backward_invariant_probe_rows(probe),
            ctx,
        )

        assert result.status == "success"
        assert result.row is not None
        assert result.row["baseline"] == "kept"
        assert result.row["field_mapper_probe_target"] == "mapped"
        assert "field_mapper_probe_source" not in result.row.to_dict()

    def test_requires_schema_config(self) -> None:
        """FieldMapper requires schema configuration."""
        from elspeth.plugins.infrastructure.config_base import PluginConfigError
        from elspeth.plugins.transforms.field_mapper import FieldMapper

        with pytest.raises(PluginConfigError, match="schema"):
            FieldMapper({"mapping": {"a": "b"}})

    def test_no_validate_input_attribute(self) -> None:
        """FieldMapper does not carry a validate_input attribute.

        Input validation is unconditional in the executor — plugins
        no longer control this via a flag.
        """
        from elspeth.plugins.transforms.field_mapper import FieldMapper

        transform = FieldMapper(
            {
                "schema": {"mode": "fixed", "fields": ["count: int"]},
                "mapping": {},
            }
        )

        assert not hasattr(transform, "validate_input")

    def test_dynamic_schema_accepts_any_types(self, ctx: PluginContext) -> None:
        """Dynamic schema imposes no type constraints on input.

        The executor validates unconditionally, but dynamic schemas
        accept everything — validation is a no-op.
        """
        from elspeth.plugins.transforms.field_mapper import FieldMapper

        transform = FieldMapper(
            {
                "schema": {"mode": "observed"},
                "mapping": {},
            }
        )

        result = transform.process(make_pipeline_row({"anything": "goes", "count": "string"}), ctx)
        assert result.status == "success"


class TestFieldMapperDuplicateTargetRejection:
    """Tests for duplicate target field name rejection.

    When multiple source fields map to the same target, the last write wins
    and earlier values are silently lost. This also corrupts contract metadata
    (type/original_name lineage from the wrong source field). The fix rejects
    such mappings at config time.
    """

    def test_duplicate_targets_rejected_at_config_time(self) -> None:
        """Two sources mapping to the same target raises PluginConfigError."""
        from elspeth.plugins.infrastructure.config_base import PluginConfigError
        from elspeth.plugins.transforms.field_mapper import FieldMapper

        with pytest.raises(PluginConfigError, match="duplicate target"):
            FieldMapper(
                {
                    "schema": DYNAMIC_SCHEMA,
                    "mapping": {"a": "x", "b": "x"},
                }
            )

    def test_triple_duplicate_targets_rejected(self) -> None:
        """Three sources mapping to the same target raises PluginConfigError."""
        from elspeth.plugins.infrastructure.config_base import PluginConfigError
        from elspeth.plugins.transforms.field_mapper import FieldMapper

        with pytest.raises(PluginConfigError, match="duplicate target"):
            FieldMapper(
                {
                    "schema": DYNAMIC_SCHEMA,
                    "mapping": {"a": "z", "b": "z", "c": "z"},
                }
            )

    def test_multiple_distinct_duplicate_targets_rejected(self) -> None:
        """Multiple groups of duplicate targets are all reported."""
        from elspeth.plugins.infrastructure.config_base import PluginConfigError
        from elspeth.plugins.transforms.field_mapper import FieldMapper

        with pytest.raises(PluginConfigError, match="duplicate target"):
            FieldMapper(
                {
                    "schema": DYNAMIC_SCHEMA,
                    "mapping": {"a": "x", "b": "x", "c": "y", "d": "y"},
                }
            )

    def test_unique_targets_accepted(self) -> None:
        """Mappings with unique targets are accepted normally."""
        from elspeth.plugins.transforms.field_mapper import FieldMapper

        # Should not raise
        transform = FieldMapper(
            {
                "schema": DYNAMIC_SCHEMA,
                "mapping": {"a": "x", "b": "y", "c": "z"},
            }
        )
        assert transform._mapping == {"a": "x", "b": "y", "c": "z"}

    def test_identity_mapping_accepted(self) -> None:
        """Identity mappings (source == target) are accepted."""
        from elspeth.plugins.transforms.field_mapper import FieldMapper

        # Should not raise
        transform = FieldMapper(
            {
                "schema": DYNAMIC_SCHEMA,
                "mapping": {"a": "a", "b": "b"},
            }
        )
        assert transform._mapping == {"a": "a", "b": "b"}

    def test_error_message_includes_collision_details(self) -> None:
        """Error message includes which sources collide on which target."""
        from elspeth.plugins.infrastructure.config_base import PluginConfigError
        from elspeth.plugins.transforms.field_mapper import FieldMapper

        with pytest.raises(PluginConfigError, match="silent data loss"):
            FieldMapper(
                {
                    "schema": DYNAMIC_SCHEMA,
                    "mapping": {"first_name": "name", "last_name": "name"},
                }
            )

    @pytest.mark.parametrize(
        "mapping",
        [
            {"a": "b", "b": "a"},
            {"a": "b", "b": "c"},
            {"b": "c", "a": "b"},
        ],
    )
    def test_overlapping_rename_graphs_rejected_at_config_time(self, mapping: dict[str, str]) -> None:
        """Targets that are also sources are rejected before they can lose data."""
        from elspeth.plugins.infrastructure.config_base import PluginConfigError
        from elspeth.plugins.transforms.field_mapper import FieldMapper

        with pytest.raises(PluginConfigError, match="overlapping rename"):
            FieldMapper(
                {
                    "schema": DYNAMIC_SCHEMA,
                    "mapping": mapping,
                }
            )


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
        return make_context()

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
        sink = inject_write_failure(
            CSVSink(
                {
                    "path": str(output_path),
                    "schema": {"mode": "observed"},
                    "headers": "original",
                }
            )
        )
        factory = make_factory()
        sink_ctx = PluginContext(
            run_id="test-run",
            config={},
            landscape=factory.plugin_audit_writer(),
            contract=result.row.contract,
        )
        sink.write([result.row.to_dict()], sink_ctx)
        sink.close()

        header = output_path.read_text().splitlines()[0]
        assert header == "Amount USD"


class TestOutputSchemaConfig:
    def test_guaranteed_fields_from_mapping_targets(self):
        from elspeth.plugins.transforms.field_mapper import FieldMapper

        transform = FieldMapper(
            {
                "mapping": {"old_name": "new_name", "source": "target"},
                "schema": {"mode": "observed"},
                "strict": True,
            }
        )
        assert transform._output_schema_config is not None
        assert frozenset(transform._output_schema_config.guaranteed_fields) == frozenset({"new_name", "target"})

    def test_non_strict_optional_mapping_does_not_declare_or_guarantee_target(self) -> None:
        """A skipped non-strict mapping target is not present on every successful row."""
        from elspeth.engine.executors.declared_output_fields import verify_declared_output_fields
        from elspeth.plugins.transforms.field_mapper import FieldMapper

        transform = FieldMapper(
            {
                "mapping": {"maybe_field": "output"},
                "schema": {"mode": "observed"},
                "strict": False,
            }
        )
        assert transform.declared_output_fields == frozenset()
        assert transform._output_schema_config is not None
        assert transform._output_schema_config.guaranteed_fields is None

        result = transform.process(make_pipeline_row({"other_field": "value"}), make_context())

        assert result.status == "success"
        assert result.row is not None
        assert result.row.to_dict() == {"other_field": "value"}
        verify_declared_output_fields(
            declared_output_fields=transform.declared_output_fields,
            emitted_rows=(result.row,),
            plugin_name=transform.name,
            node_id="node",
            run_id="run",
            row_id="row",
            token_id="token",
        )

    def test_non_strict_mapping_from_guaranteed_source_declares_target(self) -> None:
        """A non-strict mapping can guarantee its target when the source is guaranteed upstream."""
        from elspeth.plugins.transforms.field_mapper import FieldMapper

        transform = FieldMapper(
            {
                "mapping": {"source": "target"},
                "schema": {"mode": "observed", "guaranteed_fields": ["source", "kept"]},
                "strict": False,
            }
        )

        assert transform.declared_output_fields == frozenset({"target"})
        assert transform._output_schema_config is not None
        assert frozenset(transform._output_schema_config.guaranteed_fields) == frozenset({"target", "kept"})

    def test_guaranteed_fields_empty_mapping(self):
        from elspeth.plugins.transforms.field_mapper import FieldMapper

        transform = FieldMapper(
            {
                "mapping": {},
                "schema": {"mode": "observed"},
            }
        )
        assert transform._output_schema_config is not None
        # Empty mapping with no upstream guaranteed_fields → abstain (None)
        assert transform._output_schema_config.guaranteed_fields is None

    def test_upstream_none_guaranteed_with_mapping_produces_explicit(self):
        """Strict mapping with upstream guaranteed_fields=None can declare target guarantees."""
        from elspeth.plugins.transforms.field_mapper import FieldMapper

        transform = FieldMapper(
            {
                "mapping": {"old": "new"},
                "schema": {"mode": "observed"},
                "strict": True,
                # No guaranteed_fields key → upstream is None (abstain)
            }
        )
        assert transform._output_schema_config is not None
        # Transform adds "new" via mapping, so it CAN guarantee something
        assert transform._output_schema_config.guaranteed_fields is not None
        assert "new" in transform._output_schema_config.guaranteed_fields

    def test_upstream_declared_empty_produces_explicit_empty(self):
        """Upstream guaranteed_fields=[] (parsed as None) + empty mapping → abstain.

        When upstream has no guaranteed_fields AND the mapping adds nothing,
        the transform should abstain (None), not declare empty guarantees.
        """
        from elspeth.plugins.transforms.field_mapper import FieldMapper

        transform = FieldMapper(
            {
                "mapping": {},
                "schema": {"mode": "observed", "guaranteed_fields": ["x"]},
                "select_only": True,
            }
        )
        assert transform._output_schema_config is not None
        # select_only with empty mapping produces no fields, but upstream declared → ()
        # Actually mapping is empty so output_fields is empty, but upstream declared
        # so we should get explicit empty tuple
        assert transform._output_schema_config.guaranteed_fields is not None
        assert transform._output_schema_config.guaranteed_fields == ()

    def test_declared_output_fields_set_from_mapping(self):
        from elspeth.plugins.transforms.field_mapper import FieldMapper

        transform = FieldMapper(
            {
                "mapping": {"a": "b", "c": "d"},
                "schema": {"mode": "observed"},
                "strict": True,
            }
        )
        assert transform.declared_output_fields == frozenset({"b", "d"})

    def test_declared_output_fields_excludes_identity_mappings(self):
        """Identity mappings (same source and target) are excluded from declared fields."""
        from elspeth.plugins.transforms.field_mapper import FieldMapper

        transform = FieldMapper(
            {
                "mapping": {"score": "score", "name": "display_name"},
                "schema": {"mode": "observed"},
                "strict": True,
            }
        )
        # "score" → "score" is identity (excluded), "name" → "display_name" is a rename (included)
        assert transform.declared_output_fields == frozenset({"display_name"})

    def test_original_header_rename_does_not_retain_unresolved_source_guarantee(self) -> None:
        """Original-header sources are not treated as normalized guarantee keys."""
        from elspeth.plugins.transforms.field_mapper import FieldMapper

        transform = FieldMapper(
            {
                "mapping": {"Amount USD": "price"},
                "schema": {"mode": "observed", "guaranteed_fields": ["amount_usd"]},
            }
        )

        assert transform.declared_output_fields == frozenset()
        assert transform._output_schema_config is not None
        assert transform._output_schema_config.guaranteed_fields == ()

    def test_original_header_identity_mapping_does_not_declare_target_as_new_field(self) -> None:
        """Original-name identity mappings must not trigger false collision checks."""
        from elspeth.plugins.transforms.field_mapper import FieldMapper

        transform = FieldMapper(
            {
                "mapping": {"Amount USD": "amount_usd"},
                "schema": {"mode": "observed"},
            }
        )

        assert transform.declared_output_fields == frozenset()


class TestFieldMapperOutputSchemaContract:
    """Tests for FieldMapper _output_schema_config reflecting actual output shape.

    Bug fix: FieldMapper called _build_output_schema_config() which copies input
    fields into output guarantees. But FieldMapper removes/renames fields, so the
    output shape differs from input. The fix builds a custom output schema config.
    """

    def test_select_only_output_guarantees_are_only_targets(self):
        """select_only=True: guaranteed fields are ONLY the mapping targets."""
        from elspeth.plugins.transforms.field_mapper import FieldMapper

        transform = FieldMapper(
            {
                "mapping": {"a": "x", "b": "y"},
                "select_only": True,
                "schema": {"mode": "observed", "guaranteed_fields": ["a", "b", "c"]},
            }
        )
        assert transform._output_schema_config is not None
        guaranteed = frozenset(transform._output_schema_config.guaranteed_fields)
        # Only mapping targets, not input fields
        assert guaranteed == frozenset({"x", "y"})
        # Input fields that were dropped should NOT be present
        assert "a" not in guaranteed
        assert "b" not in guaranteed
        assert "c" not in guaranteed

    def test_rename_removes_source_adds_target_in_guarantees(self):
        """Rename mapping removes source field and adds target in guaranteed_fields."""
        from elspeth.plugins.transforms.field_mapper import FieldMapper

        transform = FieldMapper(
            {
                "mapping": {"old_name": "new_name"},
                "schema": {"mode": "observed", "guaranteed_fields": ["old_name", "keep_me"]},
            }
        )
        assert transform._output_schema_config is not None
        guaranteed = frozenset(transform._output_schema_config.guaranteed_fields)
        assert "new_name" in guaranteed
        assert "keep_me" in guaranteed
        assert "old_name" not in guaranteed

    def test_identity_mapping_preserves_field_in_guarantees(self):
        """Identity mapping (source == target) keeps the field in guaranteed_fields."""
        from elspeth.plugins.transforms.field_mapper import FieldMapper

        transform = FieldMapper(
            {
                "mapping": {"id": "id"},
                "schema": {"mode": "observed", "guaranteed_fields": ["id", "name"]},
            }
        )
        assert transform._output_schema_config is not None
        guaranteed = frozenset(transform._output_schema_config.guaranteed_fields)
        assert "id" in guaranteed
        assert "name" in guaranteed
