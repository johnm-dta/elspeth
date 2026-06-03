"""Tests for NullSource - a source that yields nothing for resume operations."""

import pytest

from elspeth.contracts import Determinism
from elspeth.contracts.plugin_context import PluginContext
from tests.fixtures.factories import make_source_context


class TestNullSource:
    """Tests for NullSource."""

    @pytest.fixture
    def ctx(self) -> PluginContext:
        """Create a minimal plugin context."""
        return make_source_context(plugin_name="null")

    def test_null_source_yields_nothing(self, ctx: PluginContext) -> None:
        """NullSource.load() yields no rows."""
        from elspeth.plugins.sources.null_source import NullSource

        source = NullSource({})

        rows = list(source.load(ctx))

        assert rows == []

    def test_null_source_has_name(self) -> None:
        """NullSource has 'null' as its name."""
        from elspeth.plugins.sources.null_source import NullSource

        source = NullSource({})
        assert source.name == "null"

    def test_null_source_has_output_schema(self) -> None:
        """NullSource has an output_schema attribute that is a PluginSchema subclass."""
        from elspeth.contracts import PluginSchema
        from elspeth.plugins.sources.null_source import NullSource

        source = NullSource({})
        # Direct access - no hasattr() per CLAUDE.md
        assert issubclass(source.output_schema, PluginSchema)

    def test_null_source_close_is_idempotent(self) -> None:
        """close() can be called multiple times safely."""
        from elspeth.plugins.sources.null_source import NullSource

        source = NullSource({})
        source.close()
        source.close()  # Should not raise

    def test_null_source_has_determinism(self) -> None:
        """NullSource has appropriate determinism marking."""
        from elspeth.plugins.sources.null_source import NullSource

        source = NullSource({})
        # NullSource is deterministic - always yields nothing
        assert source.determinism == Determinism.DETERMINISTIC

    def test_null_source_has_plugin_version(self) -> None:
        """NullSource has a plugin_version."""
        from elspeth.plugins.sources.null_source import NullSource

        source = NullSource({})
        assert hasattr(source, "plugin_version")
        assert isinstance(source.plugin_version, str)
        assert source.plugin_version != ""

    def test_null_source_on_success_via_bridge(self) -> None:
        """NullSource on_success is set by instantiation bridge, not config."""
        from elspeth.plugins.sources.null_source import NullSource

        # NullSource no longer validates on_success from config dict;
        # it's injected post-construction by the instantiation bridge.
        source = NullSource({})
        source.on_success = "output"
        assert source.on_success == "output"

    def test_null_source_schema_is_observed(self) -> None:
        """NullSourceSchema must be recognized as observed by DAG validation.

        Bug fix: P2-2026-02-05-nullsourceschema-treated-as-explicit-schema

        NullSourceSchema must have extra="allow" to be recognized as observed.
        The DAG validator checks: len(model_fields) == 0 AND model_config["extra"] == "allow"
        Without this, resume execution graph validation fails when downstream
        transforms have explicit input schemas.
        """
        from elspeth.plugins.sources.null_source import NullSourceSchema

        # Structural observed check: no fields + extra="allow"
        # This mirrors the check in dag.py:_is_observed_schema
        assert len(NullSourceSchema.model_fields) == 0
        assert NullSourceSchema.model_config["extra"] == "allow"

    def test_null_source_schema_with_explicit_downstream_schema(self) -> None:
        """Resume graph with NullSource and explicit downstream schema must validate.

        Bug fix: P2-2026-02-05-nullsourceschema-treated-as-explicit-schema

        This simulates a resume scenario where:
        - NullSource (observed schema) is the source
        - Downstream transform has explicit input schema with required fields

        If NullSourceSchema is treated as explicit (extra="ignore"), validation fails
        with "Missing fields" because an empty explicit schema cannot satisfy
        required fields. With extra="allow" (observed), validation passes because
        observed schemas bypass static type validation.
        """
        from pydantic import ConfigDict

        from elspeth.contracts import NodeType, PluginSchema
        from elspeth.core.dag import ExecutionGraph
        from elspeth.plugins.sources.null_source import NullSourceSchema

        # Create an explicit input schema with required fields
        class ExplicitInputSchema(PluginSchema):
            """Transform input schema that requires specific fields."""

            model_config = ConfigDict(extra="forbid", strict=True)
            customer_id: str
            amount: int

        # Build resume-like graph: NullSource -> Transform (explicit input) -> Sink
        graph = ExecutionGraph()
        graph.add_node(
            "source",
            node_type=NodeType.SOURCE,
            plugin_name="null",
            output_schema=NullSourceSchema,
        )
        graph.add_node(
            "transform",
            node_type=NodeType.TRANSFORM,
            plugin_name="some_transform",
            input_schema=ExplicitInputSchema,
        )
        graph.add_node(
            "sink",
            node_type=NodeType.SINK,
            plugin_name="csv",
        )
        graph.add_edge("source", "transform", label="continue")
        graph.add_edge("transform", "sink", label="continue")

        # Validation should pass because NullSourceSchema is observed
        # and observed schemas bypass static type validation (dag.py lines 1005-1008)
        # Before the fix, this would raise GraphValidationError with "Missing fields: customer_id, amount"
        graph.validate()  # Should not raise
