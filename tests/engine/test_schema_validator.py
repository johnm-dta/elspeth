# tests/engine/test_schema_validator.py
"""Tests for pipeline schema validation."""

from __future__ import annotations

from typing import Any


class TestPipelineSchemaValidator:
    """Tests for optional schema compatibility checking."""

    def test_validates_compatible_schemas(self) -> None:
        """Compatible schemas pass validation."""
        from elspeth.contracts import PluginSchema
        from elspeth.engine.schema_validator import validate_pipeline_schemas

        class SourceOutput(PluginSchema):
            name: str
            value: int

        class TransformInput(PluginSchema):
            name: str
            value: int

        # Mock pipeline config
        errors = validate_pipeline_schemas(
            source_output=SourceOutput,
            transform_inputs=[TransformInput],
            transform_outputs=[SourceOutput],  # Pass-through
            sink_inputs=[SourceOutput],
        )

        assert len(errors) == 0

    def test_detects_missing_field(self) -> None:
        """Detects when consumer expects field producer doesn't provide."""
        from elspeth.contracts import PluginSchema
        from elspeth.engine.schema_validator import validate_pipeline_schemas

        class SourceOutput(PluginSchema):
            name: str

        class TransformInput(PluginSchema):
            name: str
            value: int  # Source doesn't provide this!

        errors = validate_pipeline_schemas(
            source_output=SourceOutput,
            transform_inputs=[TransformInput],
            transform_outputs=[SourceOutput],
            sink_inputs=[SourceOutput],
        )

        assert len(errors) == 1
        assert "value" in errors[0]

    def test_detects_transform_chain_incompatibility(self) -> None:
        """Detects incompatibility between transforms in a chain."""
        from elspeth.contracts import PluginSchema
        from elspeth.engine.schema_validator import validate_pipeline_schemas

        class Schema1(PluginSchema):
            field_a: str

        class Schema2(PluginSchema):
            field_a: str
            field_b: int  # Added by transform 0

        class Schema3(PluginSchema):
            field_a: str
            field_b: int
            field_c: float  # Transform 1 needs this, but transform 0 doesn't output it

        errors = validate_pipeline_schemas(
            source_output=Schema1,
            transform_inputs=[Schema1, Schema3],  # Transform 1 expects field_c
            transform_outputs=[
                Schema2,
                Schema2,
            ],  # Transform 0 outputs Schema2 (no field_c)
            sink_inputs=[Schema2],
        )

        assert len(errors) == 1
        assert "field_c" in errors[0]
        assert "transform[1]" in errors[0].lower() or "Transform[1]" in errors[0]

    def test_detects_sink_incompatibility(self) -> None:
        """Detects when sink requires field that final transform doesn't provide."""
        from elspeth.contracts import PluginSchema
        from elspeth.engine.schema_validator import validate_pipeline_schemas

        class TransformOutput(PluginSchema):
            result: str

        class SinkInput(PluginSchema):
            result: str
            metadata: dict[str, Any]  # Not provided by transform!

        errors = validate_pipeline_schemas(
            source_output=TransformOutput,
            transform_inputs=[],
            transform_outputs=[],
            sink_inputs=[SinkInput],
        )

        # With no transforms, source output goes directly to sink
        assert len(errors) == 1
        assert "metadata" in errors[0]

    def test_skips_validation_for_none_schemas(self) -> None:
        """Plugins with None schemas (dynamic) skip validation."""
        from elspeth.contracts import PluginSchema
        from elspeth.engine.schema_validator import validate_pipeline_schemas

        class SomeSchema(PluginSchema):
            field: str

        # None schemas indicate dynamic/unknown schemas - skip validation
        errors = validate_pipeline_schemas(
            source_output=None,
            transform_inputs=[SomeSchema],
            transform_outputs=[SomeSchema],
            sink_inputs=[SomeSchema],
        )

        assert len(errors) == 0

    def test_optional_fields_not_required(self) -> None:
        """Optional fields in consumer schema are not required from producer."""
        from elspeth.contracts import PluginSchema
        from elspeth.engine.schema_validator import validate_pipeline_schemas

        class SourceOutput(PluginSchema):
            name: str

        class TransformInput(PluginSchema):
            name: str
            optional_field: int | None = None  # Has default, not required

        errors = validate_pipeline_schemas(
            source_output=SourceOutput,
            transform_inputs=[TransformInput],
            transform_outputs=[SourceOutput],
            sink_inputs=[SourceOutput],
        )

        assert len(errors) == 0

    def test_multiple_sinks_validated_independently(self) -> None:
        """Each sink is validated against final transform output."""
        from elspeth.contracts import PluginSchema
        from elspeth.engine.schema_validator import validate_pipeline_schemas

        class TransformOutput(PluginSchema):
            data: str

        class GoodSink(PluginSchema):
            data: str

        class BadSink(PluginSchema):
            data: str
            extra: int  # Not provided!

        errors = validate_pipeline_schemas(
            source_output=TransformOutput,
            transform_inputs=[TransformOutput],
            transform_outputs=[TransformOutput],
            sink_inputs=[GoodSink, BadSink],
        )

        # Only BadSink should have an error
        assert len(errors) == 1
        assert "extra" in errors[0]
        assert "sink[1]" in errors[0].lower() or "sink" in errors[0].lower()
