# tests/integration/test_llm_contract_validation.py
"""Integration tests for LLM transform schema contract validation.

These tests verify that the contract validation system catches missing
template fields at configuration time, before any data processing occurs.
"""

import pytest


class TestLLMContractValidationBasics:
    """Basic tests for LLM transform required_input_fields."""

    def test_llm_config_accepts_required_input_fields(self) -> None:
        """LLMConfig parses required_input_fields correctly."""
        from elspeth.plugins.llm.base import LLMConfig

        config = LLMConfig.from_dict(
            {
                "schema": {"fields": "dynamic"},
                "model": "gpt-4",
                "template": "Hello {{ row.customer_name }}",
                "required_input_fields": ["customer_name"],
            }
        )

        assert config.required_input_fields == ["customer_name"]

    def test_llm_config_validates_required_fields_format(self) -> None:
        """LLMConfig validates required_input_fields are valid identifiers."""
        from elspeth.plugins.config_base import PluginConfigError
        from elspeth.plugins.llm.base import LLMConfig

        with pytest.raises(PluginConfigError, match="valid Python identifier"):
            LLMConfig.from_dict(
                {
                    "schema": {"fields": "dynamic"},
                    "model": "gpt-4",
                    "template": "Hello",
                    "required_input_fields": ["valid", "invalid-field"],
                }
            )


class TestLLMTemplateFieldDeclarationRequired:
    """Tests for error-with-opt-out when templates reference row fields."""

    def test_error_when_template_uses_row_without_declaration(self) -> None:
        """Error raised when template references row fields but required_input_fields not declared."""
        from elspeth.plugins.config_base import PluginConfigError
        from elspeth.plugins.llm.base import LLMConfig

        with pytest.raises(PluginConfigError) as exc_info:
            LLMConfig.from_dict(
                {
                    "schema": {"fields": "dynamic"},
                    "model": "gpt-4",
                    "template": "Customer: {{ row.customer_id }}, Amount: {{ row.amount }}",
                    # No required_input_fields - this is now an error
                }
            )

        error = str(exc_info.value)
        assert "customer_id" in error
        assert "amount" in error
        assert "required_input_fields" in error

    def test_explicit_empty_list_allows_opt_out(self) -> None:
        """Empty list [] is explicit opt-out - no error raised."""
        from elspeth.plugins.llm.base import LLMConfig

        # This should NOT raise - empty list is explicit opt-out
        config = LLMConfig.from_dict(
            {
                "schema": {"fields": "dynamic"},
                "model": "gpt-4",
                "template": "Customer: {{ row.customer_id }}",
                "required_input_fields": [],  # Explicit: "I accept runtime risk"
            }
        )

        assert config.required_input_fields == []

    def test_no_error_when_fields_declared(self) -> None:
        """No error when required_input_fields properly declared."""
        from elspeth.plugins.llm.base import LLMConfig

        # Should not raise
        config = LLMConfig.from_dict(
            {
                "schema": {"fields": "dynamic"},
                "model": "gpt-4",
                "template": "Customer: {{ row.customer_id }}",
                "required_input_fields": ["customer_id"],
            }
        )

        assert config.required_input_fields == ["customer_id"]

    def test_no_error_for_non_row_templates(self) -> None:
        """No error when template doesn't reference row namespace."""
        from elspeth.plugins.llm.base import LLMConfig

        # Should not raise - no row references
        config = LLMConfig.from_dict(
            {
                "schema": {"fields": "dynamic"},
                "model": "gpt-4",
                "template": "Static prompt with {{ lookup.value }}",
            }
        )

        assert config.required_input_fields is None


class TestDAGContractValidationWithLLMConfig:
    """Tests for DAG-level contract validation with LLM transforms."""

    def test_dag_validation_catches_missing_template_field(self) -> None:
        """DAG validation fails when LLM requires field source doesn't provide."""
        from elspeth.contracts import NodeType
        from elspeth.core.dag import ExecutionGraph

        graph = ExecutionGraph()

        # Source only guarantees 'id'
        graph.add_node(
            "source_1",
            node_type=NodeType.SOURCE,
            plugin_name="csv",
            config={"schema": {"fields": "dynamic", "guaranteed_fields": ["id"]}},
        )

        # LLM requires 'id' and 'customer_name'
        graph.add_node(
            "llm_1",
            node_type=NodeType.TRANSFORM,
            plugin_name="azure_llm",
            config={
                "schema": {"fields": "dynamic"},
                "required_input_fields": ["id", "customer_name"],
            },
        )

        graph.add_node(
            "sink_1",
            node_type=NodeType.SINK,
            plugin_name="csv",
            config={"schema": {"fields": "dynamic"}},
        )

        graph.add_edge("source_1", "llm_1", label="continue")
        graph.add_edge("llm_1", "sink_1", label="continue")

        with pytest.raises(ValueError) as exc_info:
            graph.validate_edge_compatibility()

        error = str(exc_info.value)
        assert "customer_name" in error
        assert "Missing fields" in error

    def test_dag_validation_passes_when_source_provides_all(self) -> None:
        """DAG validation passes when source provides all required fields."""
        from elspeth.contracts import NodeType
        from elspeth.core.dag import ExecutionGraph

        graph = ExecutionGraph()

        # Source guarantees both fields
        graph.add_node(
            "source_1",
            node_type=NodeType.SOURCE,
            plugin_name="csv",
            config={"schema": {"fields": "dynamic", "guaranteed_fields": ["id", "customer_name"]}},
        )

        # LLM requires both
        graph.add_node(
            "llm_1",
            node_type=NodeType.TRANSFORM,
            plugin_name="azure_llm",
            config={
                "schema": {"fields": "dynamic"},
                "required_input_fields": ["id", "customer_name"],
            },
        )

        graph.add_node(
            "sink_1",
            node_type=NodeType.SINK,
            plugin_name="csv",
            config={"schema": {"fields": "dynamic"}},
        )

        graph.add_edge("source_1", "llm_1", label="continue")
        graph.add_edge("llm_1", "sink_1", label="continue")

        # Should not raise
        graph.validate_edge_compatibility()

    def test_explicit_override_for_conditional_template(self) -> None:
        """Developer can declare subset of template fields as required."""
        from elspeth.contracts import NodeType
        from elspeth.core.dag import ExecutionGraph

        # This tests the use case where a template has conditional logic:
        # {% if row.premium %}{{ row.discount }}{% endif %}
        # The developer only declares 'id' as truly required
        # (premium and discount are optional/conditional)

        graph = ExecutionGraph()

        # Source only guarantees 'id'
        graph.add_node(
            "source_1",
            node_type=NodeType.SOURCE,
            plugin_name="csv",
            config={"schema": {"fields": "dynamic", "guaranteed_fields": ["id"]}},
        )

        # LLM template might reference premium, discount, but only id is REQUIRED
        graph.add_node(
            "llm_1",
            node_type=NodeType.TRANSFORM,
            plugin_name="azure_llm",
            config={
                "schema": {"fields": "dynamic"},
                "required_input_fields": ["id"],  # Explicit: only id is required
            },
        )

        graph.add_node(
            "sink_1",
            node_type=NodeType.SINK,
            plugin_name="csv",
            config={"schema": {"fields": "dynamic"}},
        )

        graph.add_edge("source_1", "llm_1", label="continue")
        graph.add_edge("llm_1", "sink_1", label="continue")

        # Should pass - developer explicitly says only 'id' is required
        graph.validate_edge_compatibility()


class TestMultiTransformChain:
    """Tests for contract validation in multi-transform pipelines."""

    def test_chain_with_field_transformation(self) -> None:
        """Pipeline where transform changes available fields."""
        from elspeth.contracts import NodeType
        from elspeth.core.dag import ExecutionGraph

        graph = ExecutionGraph()

        # Source guarantees raw fields
        graph.add_node(
            "source_1",
            node_type=NodeType.SOURCE,
            plugin_name="csv",
            config={"schema": {"fields": "dynamic", "guaranteed_fields": ["raw_text"]}},
        )

        # First LLM requires raw_text, produces classification
        graph.add_node(
            "llm_classify",
            node_type=NodeType.TRANSFORM,
            plugin_name="azure_llm",
            config={
                "schema": {"fields": "dynamic", "guaranteed_fields": ["raw_text", "classification"]},
                "required_input_fields": ["raw_text"],
            },
        )

        # Second LLM requires classification (from first LLM)
        graph.add_node(
            "llm_action",
            node_type=NodeType.TRANSFORM,
            plugin_name="azure_llm",
            config={
                "schema": {"fields": "dynamic"},
                "required_input_fields": ["classification"],
            },
        )

        graph.add_node(
            "sink_1",
            node_type=NodeType.SINK,
            plugin_name="csv",
            config={"schema": {"fields": "dynamic"}},
        )

        graph.add_edge("source_1", "llm_classify", label="continue")
        graph.add_edge("llm_classify", "llm_action", label="continue")
        graph.add_edge("llm_action", "sink_1", label="continue")

        # Should pass - each stage provides what the next needs
        graph.validate_edge_compatibility()

    def test_chain_fails_if_intermediate_drops_field(self) -> None:
        """Pipeline fails if intermediate transform doesn't guarantee needed field."""
        from elspeth.contracts import NodeType
        from elspeth.core.dag import ExecutionGraph

        graph = ExecutionGraph()

        # Source guarantees both fields
        graph.add_node(
            "source_1",
            node_type=NodeType.SOURCE,
            plugin_name="csv",
            config={"schema": {"fields": "dynamic", "guaranteed_fields": ["a", "b"]}},
        )

        # First transform only guarantees 'a' (effectively drops 'b')
        graph.add_node(
            "transform_1",
            node_type=NodeType.TRANSFORM,
            plugin_name="mapper",
            config={
                "schema": {"fields": "dynamic", "guaranteed_fields": ["a"]},
                "required_input_fields": ["a"],
            },
        )

        # Second transform requires both 'a' and 'b'
        graph.add_node(
            "transform_2",
            node_type=NodeType.TRANSFORM,
            plugin_name="llm",
            config={
                "schema": {"fields": "dynamic"},
                "required_input_fields": ["a", "b"],
            },
        )

        graph.add_node(
            "sink_1",
            node_type=NodeType.SINK,
            plugin_name="csv",
            config={"schema": {"fields": "dynamic"}},
        )

        graph.add_edge("source_1", "transform_1", label="continue")
        graph.add_edge("transform_1", "transform_2", label="continue")
        graph.add_edge("transform_2", "sink_1", label="continue")

        with pytest.raises(ValueError, match=r"Missing fields.*'b'"):
            graph.validate_edge_compatibility()
