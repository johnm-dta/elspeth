"""Composer/runtime schema-contract characterization.

This suite covers two categories:
- shared contract cases where composer preview and runtime should agree
- documented runtime-only gaps where composer stays permissive and the runtime
  validator remains authoritative

It does not claim global equivalence between preview validation and runtime DAG
validation.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from elspeth.cli_helpers import instantiate_plugins_from_config
from elspeth.contracts.errors import FrameworkBugError
from elspeth.core.config import (
    AggregationSettings,
    CoalesceSettings,
    ElspethSettings,
    GateSettings,
    SinkSettings,
    SourceSettings,
    TransformSettings,
    TriggerConfig,
)
from elspeth.core.dag import ExecutionGraph, GraphValidationError
from elspeth.engine.orchestrator.preflight import assemble_and_validate_pipeline_config
from elspeth.engine.orchestrator.types import RouteValidationError
from elspeth.plugins.infrastructure.config_base import PluginConfigError
from elspeth.web.composer import yaml_generator as composer_yaml_generator
from elspeth.web.composer.state import (
    CompositionState,
    EdgeSpec,
    NodeSpec,
    OutputSpec,
    PipelineMetadata,
    SourceSpec,
)
from elspeth.web.execution.validation import validate_pipeline


class TestComposerRuntimeAgreement:
    """Shared agreement checks plus documented runtime-only gap characterization."""

    def _empty_state(self) -> CompositionState:
        return CompositionState(
            source=None,
            nodes=(),
            edges=(),
            outputs=(),
            metadata=PipelineMetadata(),
            version=1,
        )

    def _build_runtime_graph(
        self,
        *,
        source_plugin: str,
        source_options: dict[str, Any],
        sink_options: dict[str, Any],
        transform_options: dict[str, Any] | None = None,
        transform_plugin: str | None = "value_transform",
        aggregation_options: dict[str, Any] | None = None,
        aggregation_plugin: str | None = None,
    ) -> ExecutionGraph:
        """Build a runtime ExecutionGraph through the production assembly path."""
        if transform_plugin is not None and aggregation_plugin is not None:
            raise AssertionError(
                "Task 8 agreement helper supports either a transform chain or an aggregation chain, not both in the same case."
            )

        source_on_success = "agg1" if aggregation_plugin is not None else ("t1" if transform_plugin is not None else "main")
        transforms: list[TransformSettings] = []
        aggregations: list[AggregationSettings] = []

        if transform_plugin is not None:
            transforms.append(
                TransformSettings(
                    name="t1",
                    plugin=transform_plugin,
                    input="t1",
                    on_success="main",
                    on_error="discard",
                    options=transform_options or {},
                )
            )

        if aggregation_plugin is not None:
            aggregations.append(
                AggregationSettings(
                    name="agg1",
                    plugin=aggregation_plugin,
                    input="agg1",
                    on_success="main",
                    on_error="discard",
                    trigger=TriggerConfig(count=1),
                    options=aggregation_options or {},
                )
            )

        config = ElspethSettings(
            source=SourceSettings(
                plugin=source_plugin,
                on_success=source_on_success,
                options={**source_options, "on_validation_failure": "discard"},
            ),
            transforms=transforms,
            aggregations=aggregations,
            sinks={
                "main": SinkSettings(
                    plugin="csv",
                    on_write_failure="discard",
                    options=sink_options,
                )
            },
        )
        return self._build_runtime_graph_from_settings(config)

    def _build_runtime_graph_from_settings(self, config: ElspethSettings) -> ExecutionGraph:
        """Build a runtime graph from full settings through the production path."""
        plugins = instantiate_plugins_from_config(config)
        return ExecutionGraph.from_plugin_instances(
            source=plugins.source,
            source_settings=plugins.source_settings,
            transforms=plugins.transforms,
            sinks=plugins.sinks,
            aggregations=plugins.aggregations,
            gates=list(config.gates),
            coalesce_settings=list(config.coalesce) if config.coalesce else None,
        )

    def test_both_reject_missing_required_field(self, tmp_path: Path) -> None:
        """Both validators reject when a consumer requires an unsatisfied field."""
        text_path = tmp_path / "input.txt"
        text_path.write_text("hello\n", encoding="utf-8")
        output_path = tmp_path / "out.csv"

        state = self._empty_state()
        state = state.with_source(
            SourceSpec(
                plugin="text",
                on_success="t1",
                options={
                    "path": str(text_path),
                    "column": "line",
                    "schema": {"mode": "observed"},
                },
                on_validation_failure="quarantine",
            )
        )
        state = state.with_node(
            NodeSpec(
                id="t1",
                node_type="transform",
                plugin="value_transform",
                input="t1",
                on_success="main",
                on_error="discard",
                options={
                    "required_input_fields": ["text"],
                    "operations": [
                        {
                            "target": "out",
                            "expression": "row['text'] + ' world'",
                        }
                    ],
                    "schema": {"mode": "observed"},
                },
                condition=None,
                routes=None,
                fork_to=None,
                branches=None,
                policy=None,
                merge=None,
            )
        )
        state = state.with_output(
            OutputSpec(
                name="main",
                plugin="csv",
                options={
                    "path": str(output_path),
                    "schema": {"mode": "observed"},
                },
                on_write_failure="discard",
            )
        )
        state = state.with_edge(
            EdgeSpec(
                id="e1",
                from_node="source",
                to_node="t1",
                edge_type="on_success",
                label=None,
            )
        )

        composer_result = state.validate()
        assert not composer_result.is_valid, "Composer should reject: source column is 'line' but consumer requires 'text'."
        assert any("schema contract violation" in entry.message.lower() for entry in composer_result.errors)

        with pytest.raises(GraphValidationError) as exc_info:
            graph = self._build_runtime_graph(
                source_plugin="text",
                source_options={
                    "path": str(text_path),
                    "column": "line",
                    "schema": {"mode": "observed"},
                },
                transform_options={
                    "required_input_fields": ["text"],
                    "operations": [
                        {
                            "target": "out",
                            "expression": "row['text'] + ' world'",
                        }
                    ],
                    "schema": {"mode": "observed"},
                },
                sink_options={
                    "path": str(output_path),
                    "schema": {"mode": "observed"},
                },
            )
            graph.validate_edge_compatibility()
        assert "text" in str(exc_info.value).lower()

    def test_both_accept_observed_text_source_with_auto_guarantee(
        self,
        tmp_path: Path,
    ) -> None:
        """Both validators accept the observed-text special-case contract."""
        text_path = tmp_path / "input.txt"
        text_path.write_text("hello\n", encoding="utf-8")
        output_path = tmp_path / "out.csv"

        state = self._empty_state()
        state = state.with_source(
            SourceSpec(
                plugin="text",
                on_success="t1",
                options={
                    "path": str(text_path),
                    "column": "text",
                    "schema": {"mode": "observed"},
                },
                on_validation_failure="quarantine",
            )
        )
        state = state.with_node(
            NodeSpec(
                id="t1",
                node_type="transform",
                plugin="value_transform",
                input="t1",
                on_success="main",
                on_error="discard",
                options={
                    "required_input_fields": ["text"],
                    "operations": [
                        {
                            "target": "out",
                            "expression": "row['text'] + ' world'",
                        }
                    ],
                    "schema": {"mode": "observed"},
                },
                condition=None,
                routes=None,
                fork_to=None,
                branches=None,
                policy=None,
                merge=None,
            )
        )
        state = state.with_output(
            OutputSpec(
                name="main",
                plugin="csv",
                options={
                    "path": str(output_path),
                    "schema": {"mode": "observed"},
                },
                on_write_failure="discard",
            )
        )
        state = state.with_edge(
            EdgeSpec(
                id="e1",
                from_node="source",
                to_node="t1",
                edge_type="on_success",
                label=None,
            )
        )

        composer_result = state.validate()
        assert composer_result.is_valid, composer_result.errors

        graph = self._build_runtime_graph(
            source_plugin="text",
            source_options={
                "path": str(text_path),
                "column": "text",
                "schema": {"mode": "observed"},
            },
            transform_options={
                "required_input_fields": ["text"],
                "operations": [
                    {
                        "target": "out",
                        "expression": "row['text'] + ' world'",
                    }
                ],
                "schema": {"mode": "observed"},
            },
            sink_options={
                "path": str(output_path),
                "schema": {"mode": "observed"},
            },
        )
        graph.validate_edge_compatibility()

    def test_both_accept_source_schema_config_alias_contract(
        self,
        tmp_path: Path,
    ) -> None:
        """Source schema_config aliases must drive the same contract in preview and runtime."""
        text_path = tmp_path / "input.txt"
        text_path.write_text("hello\n", encoding="utf-8")
        output_path = tmp_path / "out.csv"

        state = self._empty_state()
        state = state.with_source(
            SourceSpec(
                plugin="text",
                on_success="t1",
                options={
                    "path": str(text_path),
                    "column": "line",
                    "schema_config": {"mode": "observed", "guaranteed_fields": ["text"]},
                },
                on_validation_failure="quarantine",
            )
        )
        state = state.with_node(
            NodeSpec(
                id="t1",
                node_type="transform",
                plugin="value_transform",
                input="t1",
                on_success="main",
                on_error="discard",
                options={
                    "required_input_fields": ["text"],
                    "operations": [
                        {
                            "target": "out",
                            "expression": "row['text'] + ' world'",
                        }
                    ],
                    "schema": {"mode": "observed"},
                },
                condition=None,
                routes=None,
                fork_to=None,
                branches=None,
                policy=None,
                merge=None,
            )
        )
        state = state.with_output(
            OutputSpec(
                name="main",
                plugin="csv",
                options={
                    "path": str(output_path),
                    "schema": {"mode": "observed"},
                },
                on_write_failure="discard",
            )
        )
        state = state.with_edge(
            EdgeSpec(
                id="e1",
                from_node="source",
                to_node="t1",
                edge_type="on_success",
                label=None,
            )
        )

        composer_result = state.validate()
        assert composer_result.is_valid, composer_result.errors
        source_contract = next(ec for ec in composer_result.edge_contracts if ec.to_id == "t1")
        assert source_contract.producer_guarantees == ("text",)
        assert source_contract.satisfied is True

        graph = self._build_runtime_graph(
            source_plugin="text",
            source_options={
                "path": str(text_path),
                "column": "line",
                "schema_config": {"mode": "observed", "guaranteed_fields": ["text"]},
            },
            transform_options={
                "required_input_fields": ["text"],
                "operations": [
                    {
                        "target": "out",
                        "expression": "row['text'] + ' world'",
                    }
                ],
                "schema": {"mode": "observed"},
            },
            sink_options={
                "path": str(output_path),
                "schema": {"mode": "observed"},
            },
        )
        graph.validate_edge_compatibility()

    def test_both_reject_observed_text_source_keyword_column(self, tmp_path: Path) -> None:
        """Invalid keyword columns must not create a false composer/runtime accept."""
        text_path = tmp_path / "input.txt"
        text_path.write_text("hello\n", encoding="utf-8")
        output_path = tmp_path / "out.csv"

        state = self._empty_state()
        state = state.with_source(
            SourceSpec(
                plugin="text",
                on_success="t1",
                options={
                    "path": str(text_path),
                    "column": "class",
                    "schema": {"mode": "observed"},
                },
                on_validation_failure="quarantine",
            )
        )
        state = state.with_node(
            NodeSpec(
                id="t1",
                node_type="transform",
                plugin="value_transform",
                input="t1",
                on_success="main",
                on_error="discard",
                options={
                    "required_input_fields": ["class"],
                    "operations": [
                        {
                            "target": "out",
                            "expression": "row['class'] + ' world'",
                        }
                    ],
                    "schema": {"mode": "observed"},
                },
                condition=None,
                routes=None,
                fork_to=None,
                branches=None,
                policy=None,
                merge=None,
            )
        )
        state = state.with_output(
            OutputSpec(
                name="main",
                plugin="csv",
                options={
                    "path": str(output_path),
                    "schema": {"mode": "observed"},
                },
                on_write_failure="discard",
            )
        )
        state = state.with_edge(
            EdgeSpec(
                id="e1",
                from_node="source",
                to_node="t1",
                edge_type="on_success",
                label=None,
            )
        )

        composer_result = state.validate()
        assert not composer_result.is_valid, (
            "Composer must not infer an observed-text guarantee for a keyword column name that runtime text-source config rejects."
        )
        assert any("class" in entry.message.lower() for entry in composer_result.errors)

        with pytest.raises(PluginConfigError, match="Python keyword"):
            self._build_runtime_graph(
                source_plugin="text",
                source_options={
                    "path": str(text_path),
                    "column": "class",
                    "schema": {"mode": "observed"},
                },
                transform_options={
                    "required_input_fields": ["class"],
                    "operations": [
                        {
                            "target": "out",
                            "expression": "row['class'] + ' world'",
                        }
                    ],
                    "schema": {"mode": "observed"},
                },
                sink_options={
                    "path": str(output_path),
                    "schema": {"mode": "observed"},
                },
            )

    def test_both_reject_strict_sink_typed_requirement_without_upstream_guarantee(
        self,
        tmp_path: Path,
    ) -> None:
        """Both validators reject when a strict sink requires an ungiven field."""
        text_path = tmp_path / "input.txt"
        text_path.write_text("hello\n", encoding="utf-8")
        output_path = tmp_path / "out.csv"

        state = self._empty_state()
        state = state.with_source(
            SourceSpec(
                plugin="text",
                on_success="main",
                options={
                    "path": str(text_path),
                    "column": "line",
                    "schema": {"mode": "observed"},
                },
                on_validation_failure="quarantine",
            )
        )
        state = state.with_output(
            OutputSpec(
                name="main",
                plugin="csv",
                options={
                    "path": str(output_path),
                    "schema": {"mode": "fixed", "fields": ["text: str"]},
                },
                on_write_failure="discard",
            )
        )

        composer_result = state.validate()
        assert not composer_result.is_valid, "Composer should reject: strict sink requires 'text' but upstream guarantees only 'line'."
        assert any(contract.to_id == "output:main" and not contract.satisfied for contract in composer_result.edge_contracts)

        with pytest.raises(GraphValidationError) as exc_info:
            graph = self._build_runtime_graph(
                source_plugin="text",
                source_options={
                    "path": str(text_path),
                    "column": "line",
                    "schema": {"mode": "observed"},
                },
                transform_plugin=None,
                sink_options={
                    "path": str(output_path),
                    "schema": {"mode": "fixed", "fields": ["text: str"]},
                },
            )
            graph.validate_edge_compatibility()
        assert "requires" in str(exc_info.value).lower()

    def test_both_reject_aggregation_nested_required_input_fields_without_upstream_guarantee(
        self,
        tmp_path: Path,
    ) -> None:
        """Composer rejects at preview time and runtime rejects during plugin wiring."""
        csv_path = tmp_path / "input.csv"
        csv_path.write_text("line\nhello\n", encoding="utf-8")
        output_path = tmp_path / "out.csv"

        state = self._empty_state()
        state = state.with_source(
            SourceSpec(
                plugin="csv",
                on_success="agg1",
                options={
                    "path": str(csv_path),
                    "schema": {"mode": "fixed", "fields": ["line: str"]},
                },
                on_validation_failure="quarantine",
            )
        )
        state = state.with_node(
            NodeSpec(
                id="agg1",
                node_type="aggregation",
                plugin="batch_stats",
                input="agg1",
                on_success="main",
                on_error="discard",
                options={
                    "value_field": "value",
                    "required_input_fields": ["value"],
                    "schema": {"mode": "observed"},
                },
                condition=None,
                routes=None,
                fork_to=None,
                branches=None,
                policy=None,
                merge=None,
            )
        )
        state = state.with_output(
            OutputSpec(
                name="main",
                plugin="csv",
                options={
                    "path": str(output_path),
                    "schema": {"mode": "observed"},
                },
                on_write_failure="discard",
            )
        )
        state = state.with_edge(
            EdgeSpec(
                id="e1",
                from_node="source",
                to_node="agg1",
                edge_type="on_success",
                label=None,
            )
        )

        composer_result = state.validate()
        assert not composer_result.is_valid
        assert any("value" in entry.message.lower() for entry in composer_result.errors)

        with pytest.raises(FrameworkBugError) as exc_info:
            self._build_runtime_graph(
                source_plugin="csv",
                source_options={
                    "path": str(csv_path),
                    "schema": {"mode": "fixed", "fields": ["line: str"]},
                },
                transform_plugin=None,
                aggregation_plugin="batch_stats",
                aggregation_options={
                    "value_field": "value",
                    "required_input_fields": ["value"],
                    "schema": {"mode": "observed"},
                },
                sink_options={
                    "path": str(output_path),
                    "schema": {"mode": "observed"},
                },
            )
        assert "value" in str(exc_info.value).lower()

    def test_both_reject_aggregation_nested_schema_required_fields_without_upstream_guarantee(
        self,
        tmp_path: Path,
    ) -> None:
        """Aggregation wrapper schema.required_fields must match runtime validation."""
        csv_path = tmp_path / "input.csv"
        csv_path.write_text("line\nhello\n", encoding="utf-8")
        output_path = tmp_path / "out.csv"

        state = self._empty_state()
        state = state.with_source(
            SourceSpec(
                plugin="csv",
                on_success="agg1",
                options={
                    "path": str(csv_path),
                    "schema": {"mode": "fixed", "fields": ["line: str"]},
                },
                on_validation_failure="quarantine",
            )
        )
        state = state.with_node(
            NodeSpec(
                id="agg1",
                node_type="aggregation",
                plugin="batch_stats",
                input="agg1",
                on_success="main",
                on_error="discard",
                options={
                    "options": {
                        "value_field": "value",
                        "schema": {"mode": "observed", "required_fields": ["value"]},
                    }
                },
                condition=None,
                routes=None,
                fork_to=None,
                branches=None,
                policy=None,
                merge=None,
            )
        )
        state = state.with_output(
            OutputSpec(
                name="main",
                plugin="csv",
                options={
                    "path": str(output_path),
                    "schema": {"mode": "observed"},
                },
                on_write_failure="discard",
            )
        )
        state = state.with_edge(
            EdgeSpec(
                id="e1",
                from_node="source",
                to_node="agg1",
                edge_type="on_success",
                label=None,
            )
        )

        composer_result = state.validate()
        assert not composer_result.is_valid
        assert any("value" in entry.message.lower() for entry in composer_result.errors)

        with pytest.raises(GraphValidationError) as exc_info:
            graph = self._build_runtime_graph(
                source_plugin="csv",
                source_options={
                    "path": str(csv_path),
                    "schema": {"mode": "fixed", "fields": ["line: str"]},
                },
                transform_plugin=None,
                aggregation_plugin="batch_stats",
                aggregation_options={
                    "value_field": "value",
                    "schema": {"mode": "observed", "required_fields": ["value"]},
                },
                sink_options={
                    "path": str(output_path),
                    "schema": {"mode": "observed"},
                },
            )
            graph.validate_edge_compatibility()
        assert "value" in str(exc_info.value).lower()

    def test_both_reject_direct_fork_to_sink_required_field_mismatch(
        self,
        tmp_path: Path,
    ) -> None:
        """Direct fork-to-sink edges stay statically checkable in preview and runtime."""
        text_path = tmp_path / "input.txt"
        text_path.write_text("hello\n", encoding="utf-8")
        output_path = tmp_path / "out.csv"

        state = self._empty_state()
        state = state.with_source(
            SourceSpec(
                plugin="text",
                on_success="gate_in",
                options={
                    "path": str(text_path),
                    "column": "line",
                    "schema": {"mode": "observed"},
                },
                on_validation_failure="quarantine",
            )
        )
        state = state.with_node(
            NodeSpec(
                id="fork_gate",
                node_type="gate",
                plugin=None,
                input="gate_in",
                on_success=None,
                on_error=None,
                options={},
                condition="True",
                routes={"true": "fork"},
                fork_to=("main",),
                branches=None,
                policy=None,
                merge=None,
            )
        )
        state = state.with_output(
            OutputSpec(
                name="main",
                plugin="csv",
                options={
                    "path": str(output_path),
                    "schema": {"mode": "fixed", "fields": ["text: str"]},
                },
                on_write_failure="discard",
            )
        )
        state = state.with_edge(
            EdgeSpec(
                id="e1",
                from_node="source",
                to_node="fork_gate",
                edge_type="on_success",
                label=None,
            )
        )
        state = state.with_edge(
            EdgeSpec(
                id="e2",
                from_node="fork_gate",
                to_node="main",
                edge_type="fork",
                label="main",
            )
        )

        composer_result = state.validate()
        assert not composer_result.is_valid
        sink_contract = next(contract for contract in composer_result.edge_contracts if contract.to_id == "output:main")
        assert sink_contract.from_id == "source"
        assert sink_contract.satisfied is False
        assert not any(
            "fork gate" in warning.message.lower() and "contract check skipped" in warning.message.lower()
            for warning in composer_result.warnings
        )

        config = ElspethSettings(
            source=SourceSettings(
                plugin="text",
                on_success="gate_in",
                options={
                    "path": str(text_path),
                    "column": "line",
                    "schema": {"mode": "observed"},
                    "on_validation_failure": "discard",
                },
            ),
            gates=[
                GateSettings(
                    name="fork_gate",
                    input="gate_in",
                    condition="True",
                    routes={"true": "fork", "false": "fork"},
                    fork_to=["main"],
                )
            ],
            sinks={
                "main": SinkSettings(
                    plugin="csv",
                    on_write_failure="discard",
                    options={
                        "path": str(output_path),
                        "schema": {"mode": "fixed", "fields": ["text: str"]},
                    },
                )
            },
        )

        with pytest.raises(GraphValidationError) as exc_info:
            graph = self._build_runtime_graph_from_settings(config)
            graph.validate_edge_compatibility()
        assert "text" in str(exc_info.value).lower()

    def test_both_accept_pass_through_downstream_of_coalesce(
        self,
        tmp_path: Path,
    ) -> None:
        """Pass-through preview must inherit coalesce guarantees after fan-in."""
        csv_path = tmp_path / "input.csv"
        csv_path.write_text("id,value\n1,2\n", encoding="utf-8")
        output_path = tmp_path / "out.csv"

        state = self._empty_state()
        state = state.with_source(
            SourceSpec(
                plugin="csv",
                on_success="gate_in",
                options={
                    "path": str(csv_path),
                    "schema": {"mode": "fixed", "fields": ["id: int", "value: int"]},
                },
                on_validation_failure="quarantine",
            )
        )
        state = state.with_node(
            NodeSpec(
                id="fork_gate",
                node_type="gate",
                plugin=None,
                input="gate_in",
                on_success=None,
                on_error=None,
                options={},
                condition="True",
                routes={"true": "fork", "false": "fork"},
                fork_to=("path_a", "path_b"),
                branches=None,
                policy=None,
                merge=None,
            )
        )
        state = state.with_node(
            NodeSpec(
                id="merge_results",
                node_type="coalesce",
                plugin=None,
                input="path_a",
                on_success=None,
                on_error=None,
                options={},
                condition=None,
                routes=None,
                fork_to=None,
                branches=("path_a", "path_b"),
                policy="best_effort",
                merge="union",
            )
        )
        state = state.with_node(
            NodeSpec(
                id="pt_after_merge",
                node_type="transform",
                plugin="passthrough",
                input="merge_results",
                on_success="main",
                on_error="discard",
                options={"schema": {"mode": "observed"}},
                condition=None,
                routes=None,
                fork_to=None,
                branches=None,
                policy=None,
                merge=None,
            )
        )
        state = state.with_output(
            OutputSpec(
                name="main",
                plugin="csv",
                options={
                    "path": str(output_path),
                    "schema": {"mode": "observed", "required_fields": ["id"]},
                },
                on_write_failure="discard",
            )
        )
        state = state.with_edge(
            EdgeSpec(
                id="e1",
                from_node="source",
                to_node="fork_gate",
                edge_type="on_success",
                label=None,
            )
        )
        state = state.with_edge(
            EdgeSpec(
                id="e2",
                from_node="fork_gate",
                to_node="merge_results",
                edge_type="fork",
                label="path_a",
            )
        )
        state = state.with_edge(
            EdgeSpec(
                id="e3",
                from_node="merge_results",
                to_node="pt_after_merge",
                edge_type="on_success",
                label=None,
            )
        )

        composer_result = state.validate()
        assert composer_result.is_valid, composer_result.errors
        sink_contract = next(contract for contract in composer_result.edge_contracts if contract.to_id == "output:main")
        assert sink_contract.from_id == "pt_after_merge"
        assert sink_contract.producer_guarantees == ("id", "value")
        assert sink_contract.consumer_requires == ("id",)
        assert sink_contract.satisfied is True

        config = ElspethSettings(
            source=SourceSettings(
                plugin="csv",
                on_success="gate_in",
                options={
                    "path": str(csv_path),
                    "schema": {"mode": "fixed", "fields": ["id: int", "value: int"]},
                    "on_validation_failure": "discard",
                },
            ),
            transforms=[
                TransformSettings(
                    name="pt_after_merge",
                    plugin="passthrough",
                    input="merge_results",
                    on_success="main",
                    on_error="discard",
                    options={"schema": {"mode": "observed"}},
                )
            ],
            gates=[
                GateSettings(
                    name="fork_gate",
                    input="gate_in",
                    condition="True",
                    routes={"true": "fork", "false": "fork"},
                    fork_to=["path_a", "path_b"],
                )
            ],
            coalesce=[
                CoalesceSettings(
                    name="merge_results",
                    branches={"path_a": "path_a", "path_b": "path_b"},
                    policy="best_effort",
                    merge="union",
                    timeout_seconds=1,
                )
            ],
            sinks={
                "main": SinkSettings(
                    plugin="csv",
                    on_write_failure="discard",
                    options={
                        "path": str(output_path),
                        "schema": {"mode": "observed", "required_fields": ["id"]},
                    },
                )
            },
        )

        graph = self._build_runtime_graph_from_settings(config)
        graph.validate_edge_compatibility()

    def test_composer_warns_but_runtime_rejects_mixed_coalesce_branch_schemas(
        self,
        tmp_path: Path,
    ) -> None:
        """Coalesce merge semantics stay runtime-authoritative beyond composer preview."""
        csv_path = tmp_path / "input.csv"
        csv_path.write_text("id,value\n1,2\n", encoding="utf-8")
        output_path = tmp_path / "out.csv"

        state = self._empty_state()
        state = state.with_source(
            SourceSpec(
                plugin="csv",
                on_success="gate_in",
                options={
                    "path": str(csv_path),
                    "schema": {"mode": "fixed", "fields": ["id: int", "value: int"]},
                },
                on_validation_failure="quarantine",
            )
        )
        state = state.with_node(
            NodeSpec(
                id="fork_gate",
                node_type="gate",
                plugin=None,
                input="gate_in",
                on_success=None,
                on_error=None,
                options={},
                condition="True",
                routes={"true": "fork", "false": "fork"},
                fork_to=("path_a", "path_b"),
                branches=None,
                policy=None,
                merge=None,
            )
        )
        state = state.with_node(
            NodeSpec(
                id="branch_b",
                node_type="transform",
                plugin="value_transform",
                input="path_b",
                on_success="path_b_done",
                on_error="discard",
                options={
                    "operations": [
                        {
                            "target": "value",
                            "expression": "row['value']",
                        }
                    ],
                    "schema": {"mode": "observed"},
                },
                condition=None,
                routes=None,
                fork_to=None,
                branches=None,
                policy=None,
                merge=None,
            )
        )
        state = state.with_node(
            NodeSpec(
                id="merge_results",
                node_type="coalesce",
                plugin=None,
                input="path_a",
                on_success="main",
                on_error=None,
                options={},
                condition=None,
                routes=None,
                fork_to=None,
                branches=("path_a", "path_b_done"),
                policy="require_all",
                merge="union",
            )
        )
        state = state.with_output(
            OutputSpec(
                name="main",
                plugin="csv",
                options={
                    "path": str(output_path),
                    "schema": {"mode": "fixed", "fields": ["id: int", "value: int"]},
                },
                on_write_failure="discard",
            )
        )
        state = state.with_edge(
            EdgeSpec(
                id="e1",
                from_node="source",
                to_node="fork_gate",
                edge_type="on_success",
                label=None,
            )
        )
        state = state.with_edge(
            EdgeSpec(
                id="e2",
                from_node="fork_gate",
                to_node="branch_b",
                edge_type="fork",
                label="path_b",
            )
        )
        state = state.with_edge(
            EdgeSpec(
                id="e3",
                from_node="fork_gate",
                to_node="merge_results",
                edge_type="fork",
                label="path_a",
            )
        )
        state = state.with_edge(
            EdgeSpec(
                id="e4",
                from_node="branch_b",
                to_node="merge_results",
                edge_type="on_success",
                label=None,
            )
        )

        composer_result = state.validate()
        assert composer_result.is_valid, composer_result.errors
        assert any("coalesce node" in warning.message.lower() for warning in composer_result.warnings)
        assert not any(contract.to_id == "output:main" for contract in composer_result.edge_contracts)

        config = ElspethSettings(
            source=SourceSettings(
                plugin="csv",
                on_success="gate_in",
                options={
                    "path": str(csv_path),
                    "schema": {"mode": "fixed", "fields": ["id: int", "value: int"]},
                    "on_validation_failure": "discard",
                },
            ),
            transforms=[
                TransformSettings(
                    name="branch_b",
                    plugin="value_transform",
                    input="path_b",
                    on_success="path_b_done",
                    on_error="discard",
                    options={
                        "operations": [
                            {
                                "target": "value",
                                "expression": "row['value']",
                            }
                        ],
                        "schema": {"mode": "observed"},
                    },
                )
            ],
            gates=[
                GateSettings(
                    name="fork_gate",
                    input="gate_in",
                    condition="True",
                    routes={"true": "fork", "false": "fork"},
                    fork_to=["path_a", "path_b"],
                )
            ],
            coalesce=[
                CoalesceSettings(
                    name="merge_results",
                    branches={"path_a": "path_a", "path_b": "path_b_done"},
                    policy="require_all",
                    merge="union",
                    on_success="main",
                )
            ],
            sinks={
                "main": SinkSettings(
                    plugin="csv",
                    on_write_failure="discard",
                    options={
                        "path": str(output_path),
                        "schema": {"mode": "fixed", "fields": ["id: int", "value: int"]},
                    },
                )
            },
        )

        with pytest.raises(GraphValidationError) as exc_info:
            graph = self._build_runtime_graph_from_settings(config)
            graph.validate_edge_compatibility()
        message = str(exc_info.value).lower()
        assert "coalesce" in message
        assert "observed" in message
        assert "explicit" in message

    def test_composer_accepts_field_names_but_runtime_rejects_type_mismatch(
        self,
        tmp_path: Path,
    ) -> None:
        """Type compatibility remains runtime-only even when contract fields line up."""
        csv_path = tmp_path / "input.csv"
        csv_path.write_text("value\nhello\n", encoding="utf-8")
        output_path = tmp_path / "out.csv"

        state = self._empty_state()
        state = state.with_source(
            SourceSpec(
                plugin="csv",
                on_success="main",
                options={
                    "path": str(csv_path),
                    "schema": {"mode": "fixed", "fields": ["value: str"]},
                },
                on_validation_failure="quarantine",
            )
        )
        state = state.with_output(
            OutputSpec(
                name="main",
                plugin="csv",
                options={
                    "path": str(output_path),
                    "schema": {"mode": "fixed", "fields": ["value: int"]},
                },
                on_write_failure="discard",
            )
        )

        composer_result = state.validate()
        assert composer_result.is_valid, composer_result.errors
        sink_contract = next(contract for contract in composer_result.edge_contracts if contract.to_id == "output:main")
        assert sink_contract.satisfied is True
        assert sink_contract.producer_guarantees == ("value",)
        assert sink_contract.consumer_requires == ("value",)

        with pytest.raises(GraphValidationError) as exc_info:
            graph = self._build_runtime_graph(
                source_plugin="csv",
                source_options={
                    "path": str(csv_path),
                    "schema": {"mode": "fixed", "fields": ["value: str"]},
                },
                transform_plugin=None,
                sink_options={
                    "path": str(output_path),
                    "schema": {"mode": "fixed", "fields": ["value: int"]},
                },
            )
            graph.validate_edge_compatibility()
        message = str(exc_info.value).lower()
        assert "incompatible" in message
        assert "value" in message

    def test_both_accept_aggregation_with_input_fields_and_required_fields(
        self,
        tmp_path: Path,
    ) -> None:
        """Regression for elspeth-f5f798f797.

        S2 v2 from notes/composer-llm-eval-2026-05-01.md: a ``batch_stats``
        aggregation with ``schema: {mode: flexible, fields: [...],
        required_fields: [...]}`` was accepted by composer ``/validate`` but
        rejected at runtime with ``SchemaConfigModeViolation`` because
        ``BaseTransform._build_output_schema_config`` propagated the user's
        input ``fields``/``required_fields`` into the aggregation's output
        schema config — which then required fields the aggregation never
        emits and expected types the OBSERVED-typed output cannot satisfy.

        After the fix in ``BatchStats._build_output_schema_config``, both
        composer preview and runtime emission verification accept this
        config. The aggregation honestly declares its output as observed
        with ``guaranteed_fields`` matching what the aggregation emits.
        """
        from elspeth.contracts.schema_contract import PipelineRow
        from elspeth.contracts.schema_contract_factory import create_contract_from_config
        from elspeth.engine.executors.schema_config_mode import verify_schema_config_mode
        from elspeth.plugins.transforms.batch_stats import BatchStats

        csv_path = tmp_path / "input.csv"
        csv_path.write_text(
            "customer_tier,amount\nenterprise,100.0\nenterprise,150.0\npro,50.0\n",
            encoding="utf-8",
        )
        output_path = tmp_path / "out.csv"

        state = self._empty_state()
        state = state.with_source(
            SourceSpec(
                plugin="csv",
                on_success="agg1",
                options={
                    "path": str(csv_path),
                    "schema": {
                        "mode": "flexible",
                        "fields": ["customer_tier: str", "amount: float"],
                    },
                },
                on_validation_failure="quarantine",
            )
        )
        state = state.with_node(
            NodeSpec(
                id="agg1",
                node_type="aggregation",
                plugin="batch_stats",
                input="agg1",
                on_success="main",
                on_error="discard",
                options={
                    "schema": {
                        "mode": "flexible",
                        "fields": ["customer_tier: str", "amount: float"],
                        "required_fields": ["customer_tier", "amount"],
                    },
                    "value_field": "amount",
                    "group_by": "customer_tier",
                    "compute_mean": False,
                },
                condition=None,
                routes=None,
                fork_to=None,
                branches=None,
                policy=None,
                merge=None,
                trigger={"count": 3},
                output_mode="transform",
                expected_output_count=2,
            )
        )
        state = state.with_output(
            OutputSpec(
                name="main",
                plugin="csv",
                options={
                    "path": str(output_path),
                    "schema": {"mode": "observed"},
                },
                on_write_failure="discard",
            )
        )
        state = state.with_edge(EdgeSpec(id="e1", from_node="source", to_node="agg1", edge_type="on_success", label=None))

        composer_result = state.validate()
        assert composer_result.is_valid, "\n".join(err.message for err in composer_result.errors)

        graph = self._build_runtime_graph(
            source_plugin="csv",
            source_options={
                "path": str(csv_path),
                "schema": {
                    "mode": "flexible",
                    "fields": ["customer_tier: str", "amount: float"],
                },
            },
            transform_plugin=None,
            aggregation_plugin="batch_stats",
            aggregation_options={
                "schema": {
                    "mode": "flexible",
                    "fields": ["customer_tier: str", "amount: float"],
                    "required_fields": ["customer_tier", "amount"],
                },
                "value_field": "amount",
                "group_by": "customer_tier",
                "compute_mean": False,
            },
            sink_options={
                "path": str(output_path),
                "schema": {"mode": "observed"},
            },
        )
        graph.validate_edge_compatibility()

        # Final tier of the agreement: simulate aggregate emission and
        # verify the runtime SchemaConfigModeViolation predicate accepts
        # the output. Pre-fix this raised; post-fix it must not raise.
        transform = BatchStats(
            {
                "schema": {
                    "mode": "flexible",
                    "fields": ["customer_tier: str", "amount: float"],
                    "required_fields": ["customer_tier", "amount"],
                },
                "value_field": "amount",
                "group_by": "customer_tier",
                "compute_mean": False,
            }
        )
        from elspeth.contracts.schema import SchemaConfig as _SchemaConfig

        input_contract = create_contract_from_config(
            _SchemaConfig.from_dict({"mode": "flexible", "fields": ["customer_tier: str", "amount: float"]})
        )
        rows = [
            PipelineRow({"customer_tier": "enterprise", "amount": 100.0}, input_contract),
            PipelineRow({"customer_tier": "enterprise", "amount": 150.0}, input_contract),
            PipelineRow({"customer_tier": "pro", "amount": 50.0}, input_contract),
        ]
        results: list[dict[str, object]] = []
        for group_value, grouped in transform._group_rows(rows):
            aggregate, error = transform._aggregate_group(grouped, group_value)
            assert error is None
            results.append(aggregate)
        emitted_contract = transform._output_contract_for(results)
        emitted_rows = [PipelineRow(r, emitted_contract) for r in results]

        verify_schema_config_mode(
            output_schema_config=transform._output_schema_config,
            emitted_rows=emitted_rows,
            plugin_name="batch_stats",
            node_id="agg1",
            run_id="r",
            row_id="row1",
            token_id="t1",
        )


class TestComposerRuntimeRouteTargetAgreement:
    """Composer ``/validate`` and runtime preflight agree on dangling route
    targets — closes the parity gap from elspeth-127de6865a.

    Empirical scope of the original gap (post-investigation):

    * Aggregation ``on_error`` -> unknown sink: composer was silent (the
      original reproducer). Now caught at ``route_target_resolution``.
    * Source ``on_validation_failure`` -> unknown sink: composer was silent.
      Now caught at ``route_target_resolution``.
    * Transform ``on_error`` -> unknown sink: was already caught at
      ``graph_structure`` (``builder.py:839``). The new check is
      defense-in-depth.
    * Sink ``on_write_failure`` -> unknown sink: was already caught at
      ``graph_structure`` (``builder.py:859``). The new check is
      defense-in-depth.

    Each gap-closing test (aggregation/source) exercises both paths from
    independent inputs and asserts the error messages are byte-identical.
    Each defense-in-depth test asserts both layers reject and the dangling
    target name is present in both messages.
    """

    @staticmethod
    def _validation_settings(data_dir: Path) -> SimpleNamespace:
        # ValidationSettings is a Protocol that only requires ``data_dir``.
        return SimpleNamespace(data_dir=data_dir)

    @staticmethod
    def _composer_route_target_failure(state: CompositionState, data_dir: Path) -> str:
        """Run validate_pipeline on a CompositionState and return the
        route_target_resolution check detail. Asserts the failure happened on
        that specific check (not graph_structure, not schema_compatibility)."""
        result = validate_pipeline(
            state,
            TestComposerRuntimeRouteTargetAgreement._validation_settings(data_dir),
            composer_yaml_generator,
        )
        assert result.is_valid is False, "Composer should reject pipelines with dangling route targets"
        check_by_name = {check.name: check for check in result.checks}
        assert "route_target_resolution" in check_by_name, "Missing route_target_resolution check in result"
        rt_check = check_by_name["route_target_resolution"]
        assert rt_check.passed is False, f"route_target_resolution should have failed; got {rt_check.detail}"
        # graph_structure should have passed — these dangling references are
        # NOT structural DAG errors.
        assert check_by_name["graph_structure"].passed is True, "graph_structure must pass — only route targets are bad"
        return rt_check.detail

    @staticmethod
    def _runtime_route_target_failure(config: ElspethSettings) -> str:
        """Instantiate plugins, build graph, call assemble_and_validate_pipeline_config.
        Returns the str(RouteValidationError) message."""
        plugins = instantiate_plugins_from_config(config)
        graph = ExecutionGraph.from_plugin_instances(
            source=plugins.source,
            source_settings=plugins.source_settings,
            transforms=plugins.transforms,
            sinks=plugins.sinks,
            aggregations=plugins.aggregations,
            gates=list(config.gates),
            coalesce_settings=list(config.coalesce) if config.coalesce else None,
        )
        graph.validate()  # Structural DAG check — must pass for these cases
        with pytest.raises(RouteValidationError) as exc_info:
            assemble_and_validate_pipeline_config(
                source=plugins.source,
                transforms=plugins.transforms,
                sinks=plugins.sinks,
                aggregations=plugins.aggregations,
                settings=config,
                graph=graph,
            )
        return str(exc_info.value)

    @staticmethod
    def _csv_input(tmp_path: Path) -> Path:
        # Sources must live under data_dir/blobs/ for the path allowlist.
        path = tmp_path / "blobs" / "input.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("value\n1\n", encoding="utf-8")
        return path

    @staticmethod
    def _csv_output(tmp_path: Path, name: str = "out.csv") -> Path:
        # Sinks must live under data_dir/outputs/ (or blobs/) for the allowlist.
        out_dir = tmp_path / "outputs"
        out_dir.mkdir(parents=True, exist_ok=True)
        return out_dir / name

    def test_both_reject_aggregation_on_error_dangling_sink(self, tmp_path: Path) -> None:
        """Original reproducer (S2 v1 from notes/composer-llm-eval-2026-05-01.md):
        aggregation ``on_error: aggregation_errors`` with no sink of that name."""
        csv_path = self._csv_input(tmp_path)
        output_path = self._csv_output(tmp_path)

        # Composer state: aggregation routes errors to a sink that doesn't exist.
        state = CompositionState(
            source=SourceSpec(
                plugin="csv",
                on_success="agg1",
                options={"path": str(csv_path), "schema": {"mode": "observed"}},
                on_validation_failure="discard",
            ),
            nodes=(
                NodeSpec(
                    id="agg1",
                    node_type="aggregation",
                    plugin="batch_stats",
                    input="agg1",
                    on_success="main",
                    on_error="aggregation_errors",  # ← dangling
                    options={"schema": {"mode": "observed"}, "value_field": "value"},
                    condition=None,
                    routes=None,
                    fork_to=None,
                    branches=None,
                    policy=None,
                    merge=None,
                    trigger={"count": 1},
                    output_mode="transform",
                    expected_output_count=1,
                ),
            ),
            edges=(
                EdgeSpec(id="e1", from_node="source", to_node="agg1", edge_type="on_success", label=None),
                EdgeSpec(id="e2", from_node="agg1", to_node="main", edge_type="on_success", label=None),
            ),
            outputs=(
                OutputSpec(
                    name="main",
                    plugin="csv",
                    options={"path": str(output_path), "schema": {"mode": "observed"}},
                    on_write_failure="discard",
                ),
            ),
            metadata=PipelineMetadata(),
            version=1,
        )
        composer_detail = self._composer_route_target_failure(state, tmp_path)

        # Runtime: equivalent ElspethSettings.
        config = ElspethSettings(
            source=SourceSettings(
                plugin="csv",
                on_success="agg1",
                options={"path": str(csv_path), "schema": {"mode": "observed"}, "on_validation_failure": "discard"},
            ),
            aggregations=[
                AggregationSettings(
                    name="agg1",
                    plugin="batch_stats",
                    input="agg1",
                    on_success="main",
                    on_error="aggregation_errors",  # ← dangling
                    trigger=TriggerConfig(count=1),
                    options={"schema": {"mode": "observed"}, "value_field": "value"},
                ),
            ],
            sinks={
                "main": SinkSettings(
                    plugin="csv",
                    on_write_failure="discard",
                    options={"path": str(output_path), "schema": {"mode": "observed"}},
                ),
            },
        )
        runtime_msg = self._runtime_route_target_failure(config)

        assert "aggregation_errors" in composer_detail
        assert "aggregation_errors" in runtime_msg
        assert composer_detail == runtime_msg, "Composer and runtime must surface identical RouteValidationError"

    def test_both_reject_transform_on_error_dangling_sink(self, tmp_path: Path) -> None:
        """Defense-in-depth axis: the DAG builder (``graph.validate()`` via
        ``from_plugin_instances``) already catches transform ``on_error`` ->
        unknown sink at ``builder.py:839``. The new ``route_target_resolution``
        check is a second wall behind it. This test asserts both walls agree:
        composer ``/validate`` rejects, runtime construction rejects, and the
        dangling target name appears in both messages."""
        csv_path = self._csv_input(tmp_path)
        output_path = self._csv_output(tmp_path)

        state = CompositionState(
            source=SourceSpec(
                plugin="csv",
                on_success="t1",
                options={"path": str(csv_path), "schema": {"mode": "observed"}},
                on_validation_failure="discard",
            ),
            nodes=(
                NodeSpec(
                    id="t1",
                    node_type="transform",
                    plugin="value_transform",
                    input="t1",
                    on_success="main",
                    on_error="missing_error_sink",
                    options={
                        "schema": {"mode": "observed"},
                        "operations": [{"target": "doubled", "expression": "row['value']"}],
                    },
                    condition=None,
                    routes=None,
                    fork_to=None,
                    branches=None,
                    policy=None,
                    merge=None,
                ),
            ),
            edges=(
                EdgeSpec(id="e1", from_node="source", to_node="t1", edge_type="on_success", label=None),
                EdgeSpec(id="e2", from_node="t1", to_node="main", edge_type="on_success", label=None),
            ),
            outputs=(
                OutputSpec(
                    name="main",
                    plugin="csv",
                    options={"path": str(output_path), "schema": {"mode": "observed"}},
                    on_write_failure="discard",
                ),
            ),
            metadata=PipelineMetadata(),
            version=1,
        )
        composer_result = validate_pipeline(state, self._validation_settings(tmp_path), composer_yaml_generator)
        assert composer_result.is_valid is False
        composer_messages = " | ".join(err.message for err in composer_result.errors)
        assert "missing_error_sink" in composer_messages

        config = ElspethSettings(
            source=SourceSettings(
                plugin="csv",
                on_success="t1",
                options={"path": str(csv_path), "schema": {"mode": "observed"}, "on_validation_failure": "discard"},
            ),
            transforms=[
                TransformSettings(
                    name="t1",
                    plugin="value_transform",
                    input="t1",
                    on_success="main",
                    on_error="missing_error_sink",
                    options={
                        "schema": {"mode": "observed"},
                        "operations": [{"target": "doubled", "expression": "row['value']"}],
                    },
                ),
            ],
            sinks={
                "main": SinkSettings(
                    plugin="csv",
                    on_write_failure="discard",
                    options={"path": str(output_path), "schema": {"mode": "observed"}},
                ),
            },
        )
        plugins = instantiate_plugins_from_config(config)
        with pytest.raises(GraphValidationError) as runtime_exc:
            ExecutionGraph.from_plugin_instances(
                source=plugins.source,
                source_settings=plugins.source_settings,
                transforms=plugins.transforms,
                sinks=plugins.sinks,
                aggregations=plugins.aggregations,
                gates=list(config.gates),
                coalesce_settings=list(config.coalesce) if config.coalesce else None,
            )
        assert "missing_error_sink" in str(runtime_exc.value)

    def test_both_reject_source_on_validation_failure_dangling_sink(self, tmp_path: Path) -> None:
        csv_path = self._csv_input(tmp_path)
        output_path = self._csv_output(tmp_path)

        state = CompositionState(
            source=SourceSpec(
                plugin="csv",
                on_success="main",
                options={"path": str(csv_path), "schema": {"mode": "observed"}},
                on_validation_failure="missing_quarantine_sink",  # ← dangling
            ),
            nodes=(),
            edges=(EdgeSpec(id="e1", from_node="source", to_node="main", edge_type="on_success", label=None),),
            outputs=(
                OutputSpec(
                    name="main",
                    plugin="csv",
                    options={"path": str(output_path), "schema": {"mode": "observed"}},
                    on_write_failure="discard",
                ),
            ),
            metadata=PipelineMetadata(),
            version=1,
        )
        composer_detail = self._composer_route_target_failure(state, tmp_path)

        config = ElspethSettings(
            source=SourceSettings(
                plugin="csv",
                on_success="main",
                options={
                    "path": str(csv_path),
                    "schema": {"mode": "observed"},
                    "on_validation_failure": "missing_quarantine_sink",
                },
            ),
            sinks={
                "main": SinkSettings(
                    plugin="csv",
                    on_write_failure="discard",
                    options={"path": str(output_path), "schema": {"mode": "observed"}},
                ),
            },
        )
        runtime_msg = self._runtime_route_target_failure(config)

        assert "missing_quarantine_sink" in composer_detail
        assert "missing_quarantine_sink" in runtime_msg
        assert composer_detail == runtime_msg

    def test_both_reject_sink_on_write_failure_dangling_sink(self, tmp_path: Path) -> None:
        """Defense-in-depth axis: ``builder.py:859`` already catches sink
        ``on_write_failure`` -> unknown sink at ``graph.validate()``. The
        helper provides a second wall via
        ``validate_sink_failsink_destinations``."""
        csv_path = self._csv_input(tmp_path)
        output_path = self._csv_output(tmp_path)

        state = CompositionState(
            source=SourceSpec(
                plugin="csv",
                on_success="main",
                options={"path": str(csv_path), "schema": {"mode": "observed"}},
                on_validation_failure="discard",
            ),
            nodes=(),
            edges=(EdgeSpec(id="e1", from_node="source", to_node="main", edge_type="on_success", label=None),),
            outputs=(
                OutputSpec(
                    name="main",
                    plugin="csv",
                    options={"path": str(output_path), "schema": {"mode": "observed"}},
                    on_write_failure="missing_failsink",
                ),
            ),
            metadata=PipelineMetadata(),
            version=1,
        )
        composer_result = validate_pipeline(state, self._validation_settings(tmp_path), composer_yaml_generator)
        assert composer_result.is_valid is False
        composer_messages = " | ".join(err.message for err in composer_result.errors)
        assert "missing_failsink" in composer_messages

        config = ElspethSettings(
            source=SourceSettings(
                plugin="csv",
                on_success="main",
                options={"path": str(csv_path), "schema": {"mode": "observed"}, "on_validation_failure": "discard"},
            ),
            sinks={
                "main": SinkSettings(
                    plugin="csv",
                    on_write_failure="missing_failsink",
                    options={"path": str(output_path), "schema": {"mode": "observed"}},
                ),
            },
        )
        plugins = instantiate_plugins_from_config(config)
        with pytest.raises(GraphValidationError) as runtime_exc:
            ExecutionGraph.from_plugin_instances(
                source=plugins.source,
                source_settings=plugins.source_settings,
                transforms=plugins.transforms,
                sinks=plugins.sinks,
                aggregations=plugins.aggregations,
                gates=list(config.gates),
                coalesce_settings=list(config.coalesce) if config.coalesce else None,
            )
        assert "missing_failsink" in str(runtime_exc.value)

    def test_both_reject_gate_routes_dangling_sink(self, tmp_path: Path) -> None:
        """Defense-in-depth axis: a gate ``routes`` target that is neither
        ``"fork"`` nor a real sink name nor a connection name. The DAG builder
        falls through to producer-registration (``builder.py:583+``); if no
        consumer claims that name, downstream graph checks reject. The
        ``route_target_resolution`` step is the second wall — when the runtime
        ultimately resolves the route, ``validate_route_destinations`` would
        also catch it.

        This test asserts agreement: composer ``/validate`` rejects, the
        independent runtime construction rejects, and both messages name the
        dangling target."""
        csv_path = self._csv_input(tmp_path)
        output_path = self._csv_output(tmp_path)

        state = CompositionState(
            source=SourceSpec(
                plugin="csv",
                on_success="g1",
                options={"path": str(csv_path), "schema": {"mode": "observed"}},
                on_validation_failure="discard",
            ),
            nodes=(
                NodeSpec(
                    id="g1",
                    node_type="gate",
                    plugin=None,
                    input="g1",
                    on_success=None,
                    on_error=None,
                    options={},
                    condition="row['value'] != ''",
                    routes={"true": "main", "false": "missing_route_sink"},
                    fork_to=None,
                    branches=None,
                    policy=None,
                    merge=None,
                ),
            ),
            edges=(
                EdgeSpec(id="e1", from_node="source", to_node="g1", edge_type="on_success", label=None),
                EdgeSpec(id="e2", from_node="g1", to_node="main", edge_type="route_true", label="true"),
            ),
            outputs=(
                OutputSpec(
                    name="main",
                    plugin="csv",
                    options={"path": str(output_path), "schema": {"mode": "observed"}},
                    on_write_failure="discard",
                ),
            ),
            metadata=PipelineMetadata(),
            version=1,
        )
        composer_result = validate_pipeline(state, self._validation_settings(tmp_path), composer_yaml_generator)
        assert composer_result.is_valid is False
        composer_messages = " | ".join(err.message for err in composer_result.errors)
        assert "missing_route_sink" in composer_messages

        config = ElspethSettings(
            source=SourceSettings(
                plugin="csv",
                on_success="g1",
                options={"path": str(csv_path), "schema": {"mode": "observed"}, "on_validation_failure": "discard"},
            ),
            gates=[
                GateSettings(
                    name="g1",
                    input="g1",
                    condition="row['value'] != ''",
                    routes={"true": "main", "false": "missing_route_sink"},
                ),
            ],
            sinks={
                "main": SinkSettings(
                    plugin="csv",
                    on_write_failure="discard",
                    options={"path": str(output_path), "schema": {"mode": "observed"}},
                ),
            },
        )
        plugins = instantiate_plugins_from_config(config)
        with pytest.raises(GraphValidationError) as runtime_exc:
            ExecutionGraph.from_plugin_instances(
                source=plugins.source,
                source_settings=plugins.source_settings,
                transforms=plugins.transforms,
                sinks=plugins.sinks,
                aggregations=plugins.aggregations,
                gates=list(config.gates),
                coalesce_settings=list(config.coalesce) if config.coalesce else None,
            )
        assert "missing_route_sink" in str(runtime_exc.value)

    def test_both_accept_aggregation_on_error_discard(self, tmp_path: Path) -> None:
        """Positive control: ``on_error: discard`` (the eval's S2 v3 fix) passes
        both composer and runtime. Confirms the new check is not over-eager."""
        csv_path = self._csv_input(tmp_path)
        output_path = self._csv_output(tmp_path)

        state = CompositionState(
            source=SourceSpec(
                plugin="csv",
                on_success="agg1",
                options={"path": str(csv_path), "schema": {"mode": "observed"}},
                on_validation_failure="discard",
            ),
            nodes=(
                NodeSpec(
                    id="agg1",
                    node_type="aggregation",
                    plugin="batch_stats",
                    input="agg1",
                    on_success="main",
                    on_error="discard",  # ← legal escape valve
                    options={"schema": {"mode": "observed"}, "value_field": "value"},
                    condition=None,
                    routes=None,
                    fork_to=None,
                    branches=None,
                    policy=None,
                    merge=None,
                    trigger={"count": 1},
                    output_mode="transform",
                    expected_output_count=1,
                ),
            ),
            edges=(
                EdgeSpec(id="e1", from_node="source", to_node="agg1", edge_type="on_success", label=None),
                EdgeSpec(id="e2", from_node="agg1", to_node="main", edge_type="on_success", label=None),
            ),
            outputs=(
                OutputSpec(
                    name="main",
                    plugin="csv",
                    options={"path": str(output_path), "schema": {"mode": "observed"}},
                    on_write_failure="discard",
                ),
            ),
            metadata=PipelineMetadata(),
            version=1,
        )
        result = validate_pipeline(
            state,
            self._validation_settings(tmp_path),
            composer_yaml_generator,
        )
        assert result.is_valid, "\n".join(err.message for err in result.errors)
        rt_check = next(c for c in result.checks if c.name == "route_target_resolution")
        assert rt_check.passed is True
        assert rt_check.detail == "All route targets resolve to existing sinks"

        # Runtime: must also assemble cleanly, no RouteValidationError.
        config = ElspethSettings(
            source=SourceSettings(
                plugin="csv",
                on_success="agg1",
                options={"path": str(csv_path), "schema": {"mode": "observed"}, "on_validation_failure": "discard"},
            ),
            aggregations=[
                AggregationSettings(
                    name="agg1",
                    plugin="batch_stats",
                    input="agg1",
                    on_success="main",
                    on_error="discard",
                    trigger=TriggerConfig(count=1),
                    options={"schema": {"mode": "observed"}, "value_field": "value"},
                ),
            ],
            sinks={
                "main": SinkSettings(
                    plugin="csv",
                    on_write_failure="discard",
                    options={"path": str(output_path), "schema": {"mode": "observed"}},
                ),
            },
        )
        plugins = instantiate_plugins_from_config(config)
        graph = ExecutionGraph.from_plugin_instances(
            source=plugins.source,
            source_settings=plugins.source_settings,
            transforms=plugins.transforms,
            sinks=plugins.sinks,
            aggregations=plugins.aggregations,
            gates=list(config.gates),
            coalesce_settings=list(config.coalesce) if config.coalesce else None,
        )
        graph.validate()
        # Should not raise.
        assemble_and_validate_pipeline_config(
            source=plugins.source,
            transforms=plugins.transforms,
            sinks=plugins.sinks,
            aggregations=plugins.aggregations,
            settings=config,
            graph=graph,
        )
