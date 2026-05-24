"""Tests for deterministic YAML generation from CompositionState."""

from __future__ import annotations

import pytest
import yaml

from elspeth.web.composer.state import (
    CompositionState,
    EdgeSpec,
    NodeSpec,
    OutputSpec,
    PipelineMetadata,
    SourceSpec,
)
from elspeth.web.composer.yaml_generator import generate_pipeline_dict, generate_yaml
from elspeth.web.interpretation_state import INTERPRETATION_REQUIREMENTS_KEY, PROMPT_TEMPLATE_PARTS_KEY, SOURCE_AUTHORING_KEY


def _make_linear_pipeline() -> CompositionState:
    """Source -> transform -> sink."""
    return CompositionState(
        source=SourceSpec(
            plugin="csv",
            on_success="transform_1",
            options={"path": "/data/input.csv", "schema": {"fields": ["name", "age"]}},
            on_validation_failure="quarantine",
        ),
        nodes=(
            NodeSpec(
                id="transform_1",
                node_type="transform",
                plugin="passthrough",
                input="source_out",
                on_success="main_output",
                on_error="discard",
                options={"field": "name"},
                condition=None,
                routes=None,
                fork_to=None,
                branches=None,
                policy=None,
                merge=None,
            ),
        ),
        edges=(
            EdgeSpec(id="e1", from_node="source", to_node="transform_1", edge_type="on_success", label=None),
            EdgeSpec(id="e2", from_node="transform_1", to_node="main_output", edge_type="on_success", label=None),
        ),
        outputs=(
            OutputSpec(
                name="main_output",
                plugin="csv",
                options={"path": "/data/output.csv"},
                on_write_failure="quarantine",
            ),
        ),
        metadata=PipelineMetadata(name="Linear Pipeline", description="A simple pipeline"),
        version=5,
    )


def _make_gate_pipeline() -> CompositionState:
    """Source -> gate -> two sinks."""
    return CompositionState(
        source=SourceSpec(
            plugin="csv",
            on_success="quality_check",
            options={"path": "/data/in.csv"},
            on_validation_failure="discard",
        ),
        nodes=(
            NodeSpec(
                id="quality_check",
                node_type="gate",
                plugin=None,
                input="source_out",
                on_success=None,
                on_error=None,
                options={},
                condition="row['confidence'] >= 0.85",
                routes={"high": "good_output", "low": "review_output"},
                fork_to=None,
                branches=None,
                policy=None,
                merge=None,
            ),
        ),
        edges=(),
        outputs=(
            OutputSpec(name="good_output", plugin="csv", options={"path": "/good.csv"}, on_write_failure="quarantine"),
            OutputSpec(name="review_output", plugin="csv", options={"path": "/review.csv"}, on_write_failure="discard"),
        ),
        metadata=PipelineMetadata(name="Gate Pipeline"),
        version=3,
    )


def _make_aggregation_pipeline() -> CompositionState:
    """Source -> aggregation -> sink."""
    return CompositionState(
        source=SourceSpec(
            plugin="csv",
            on_success="batch_agg",
            options={"path": "/data/in.csv"},
            on_validation_failure="quarantine",
        ),
        nodes=(
            NodeSpec(
                id="batch_agg",
                node_type="aggregation",
                plugin="batch_counter",
                input="source_out",
                on_success="main_output",
                on_error="discard",
                options={"batch_size": 10},
                condition=None,
                routes=None,
                fork_to=None,
                branches=None,
                policy=None,
                merge=None,
                trigger={"count": 10},
            ),
        ),
        edges=(),
        outputs=(OutputSpec(name="main_output", plugin="csv", options={}, on_write_failure="discard"),),
        metadata=PipelineMetadata(),
        version=1,
    )


def _make_fork_coalesce_pipeline() -> CompositionState:
    """Source -> fork gate -> two paths -> coalesce -> sink."""
    return CompositionState(
        source=SourceSpec(
            plugin="csv",
            on_success="fork_gate",
            options={},
            on_validation_failure="quarantine",
        ),
        nodes=(
            NodeSpec(
                id="fork_gate",
                node_type="gate",
                plugin=None,
                input="source_out",
                on_success=None,
                on_error=None,
                options={},
                condition="True",
                routes={"all": "fork"},
                fork_to=("path_a", "path_b"),
                branches=None,
                policy=None,
                merge=None,
            ),
            NodeSpec(
                id="merge_point",
                node_type="coalesce",
                plugin=None,
                input="join",
                on_success="main_output",
                on_error=None,
                options={},
                condition=None,
                routes=None,
                fork_to=None,
                branches=("path_a", "path_b"),
                policy="require_all",
                merge="nested",
            ),
        ),
        edges=(),
        outputs=(OutputSpec(name="main_output", plugin="csv", options={}, on_write_failure="discard"),),
        metadata=PipelineMetadata(),
        version=1,
    )


class TestGenerateYaml:
    def test_generate_pipeline_dict_matches_generate_yaml_shape(self) -> None:
        state = _make_fork_coalesce_pipeline()

        pipeline_dict = generate_pipeline_dict(state)

        assert pipeline_dict == yaml.safe_load(generate_yaml(state))
        assert set(pipeline_dict) == {"source", "gates", "coalesce", "sinks"}

    def test_generate_pipeline_dict_strips_web_metadata(self) -> None:
        state = _make_linear_pipeline().with_source(
            SourceSpec(
                plugin="csv",
                on_success="transform_1",
                options={"path": "/data/input.csv", "blob_ref": "web-only"},
                on_validation_failure="quarantine",
            )
        )

        pipeline_dict = generate_pipeline_dict(state)

        assert "blob_ref" not in pipeline_dict["source"]["options"]
        assert "blob_ref" not in yaml.safe_load(generate_yaml(state))["source"]["options"]

    def test_generate_pipeline_dict_strips_source_authoring_metadata(self) -> None:
        state = _make_linear_pipeline().with_source(
            SourceSpec(
                plugin="csv",
                on_success="transform_1",
                options={
                    "path": "/data/input.csv",
                    "blob_ref": "web-only",
                    SOURCE_AUTHORING_KEY: {
                        "modality": "llm_generated",
                        "content_hash": "abc123",
                        "review_event_id": None,
                        "resolved_kind": None,
                    },
                    "schema": {"mode": "observed"},
                },
                on_validation_failure="quarantine",
            )
        )

        pipeline_dict = generate_pipeline_dict(state)
        yaml_options = yaml.safe_load(generate_yaml(state))["source"]["options"]

        assert SOURCE_AUTHORING_KEY not in pipeline_dict["source"]["options"]
        assert SOURCE_AUTHORING_KEY not in yaml_options
        assert yaml_options["schema"] == {"mode": "observed"}

    def test_generate_pipeline_dict_strips_interpretation_authoring_metadata_from_transforms(self) -> None:
        state = _make_linear_pipeline().with_node(
            NodeSpec(
                id="transform_1",
                node_type="transform",
                plugin="llm",
                input="source_out",
                on_success="main_output",
                on_error="discard",
                options={
                    "prompt_template": "Rate resolved meaning: {{ row.text }}",
                    PROMPT_TEMPLATE_PARTS_KEY: [{"kind": "text", "text": "ignored"}],
                    INTERPRETATION_REQUIREMENTS_KEY: [],
                    "resolved_prompt_template_hash": "sha256-rfc8785-v1:abc123",
                },
                condition=None,
                routes=None,
                fork_to=None,
                branches=None,
                policy=None,
                merge=None,
            )
        )

        pipeline_dict = generate_pipeline_dict(state)
        options = pipeline_dict["transforms"][0]["options"]

        assert PROMPT_TEMPLATE_PARTS_KEY not in options
        assert INTERPRETATION_REQUIREMENTS_KEY not in options
        assert options["prompt_template"] == "Rate resolved meaning: {{ row.text }}"
        assert options["resolved_prompt_template_hash"] == "sha256-rfc8785-v1:abc123"

    def test_generate_pipeline_dict_rejects_unknown_node_type(self) -> None:
        """YAML export must not silently drop nodes outside the closed set."""
        state = _make_linear_pipeline().with_node(
            NodeSpec.from_dict(
                {
                    "id": "mystery",
                    "node_type": "bogus",
                    "plugin": "passthrough",
                    "input": "source_out",
                    "on_success": "main_output",
                    "on_error": "discard",
                    "options": {},
                    "condition": None,
                    "routes": None,
                    "fork_to": None,
                    "branches": None,
                    "policy": None,
                    "merge": None,
                }
            )
        )

        with pytest.raises(ValueError, match="Unknown node_type 'bogus' for node 'mystery'"):
            generate_pipeline_dict(state)

    def test_linear_pipeline(self) -> None:
        state = _make_linear_pipeline()
        yaml_str = generate_yaml(state)
        parsed = yaml.safe_load(yaml_str)

        # Source
        assert parsed["source"]["plugin"] == "csv"
        assert parsed["source"]["on_success"] == "transform_1"
        assert parsed["source"]["options"]["path"] == "/data/input.csv"
        assert parsed["source"]["options"]["on_validation_failure"] == "quarantine"

        # Transform
        assert len(parsed["transforms"]) == 1
        t = parsed["transforms"][0]
        assert t["name"] == "transform_1"
        assert t["plugin"] == "passthrough"
        assert t["input"] == "source_out"
        assert t["on_success"] == "main_output"
        assert t["on_error"] == "discard"
        assert t["options"]["field"] == "name"

        # Sink
        assert "main_output" in parsed["sinks"]
        s = parsed["sinks"]["main_output"]
        assert s["plugin"] == "csv"

    def test_gate_pipeline(self) -> None:
        state = _make_gate_pipeline()
        yaml_str = generate_yaml(state)
        parsed = yaml.safe_load(yaml_str)

        assert "gates" in parsed
        assert len(parsed["gates"]) == 1
        g = parsed["gates"][0]
        assert g["name"] == "quality_check"
        assert g["condition"] == "row['confidence'] >= 0.85"
        assert g["routes"]["high"] == "good_output"
        assert g["routes"]["low"] == "review_output"

    def test_gate_route_to_discard_is_exported_as_virtual_destination(self) -> None:
        from elspeth.core.config import load_settings_from_yaml_string

        state = _make_gate_pipeline()
        gate = state.nodes[0]
        state = state.with_node(
            NodeSpec(
                id=gate.id,
                node_type=gate.node_type,
                plugin=gate.plugin,
                input=gate.input,
                on_success=gate.on_success,
                on_error=gate.on_error,
                options=gate.options,
                condition=gate.condition,
                routes={"true": "good_output", "false": "discard"},
                fork_to=gate.fork_to,
                branches=gate.branches,
                policy=gate.policy,
                merge=gate.merge,
            )
        )

        yaml_str = generate_yaml(state)
        parsed = yaml.safe_load(yaml_str)

        assert parsed["gates"][0]["routes"]["false"] == "discard"
        settings = load_settings_from_yaml_string(yaml_str)
        assert settings.gates[0].routes["false"] == "discard"

    def test_aggregation_pipeline(self) -> None:
        state = _make_aggregation_pipeline()
        yaml_str = generate_yaml(state)
        parsed = yaml.safe_load(yaml_str)

        assert "aggregations" in parsed
        assert len(parsed["aggregations"]) == 1
        a = parsed["aggregations"][0]
        assert a["name"] == "batch_agg"
        assert a["plugin"] == "batch_counter"
        assert a["trigger"] == {"count": 10}
        assert a["on_error"] == "discard"
        assert a["options"]["batch_size"] == 10

    def test_fork_coalesce_pipeline(self) -> None:
        state = _make_fork_coalesce_pipeline()
        yaml_str = generate_yaml(state)
        parsed = yaml.safe_load(yaml_str)

        assert "gates" in parsed
        gate = parsed["gates"][0]
        assert gate["fork_to"] == ["path_a", "path_b"]

        assert "coalesce" in parsed
        coal = parsed["coalesce"][0]
        assert coal["branches"] == ["path_a", "path_b"]
        assert coal["policy"] == "require_all"
        assert coal["merge"] == "nested"
        assert coal["on_success"] == "main_output"

    def test_deterministic(self) -> None:
        """Same state produces byte-identical YAML."""
        state = _make_linear_pipeline()
        yaml1 = generate_yaml(state)
        yaml2 = generate_yaml(state)
        assert yaml1 == yaml2

    def test_landscape_key_never_emitted(self) -> None:
        """landscape key is never emitted -- URL comes from WebSettings at execution time (S1 fix)."""
        state = _make_linear_pipeline()
        yaml_str = generate_yaml(state)
        parsed = yaml.safe_load(yaml_str)
        assert "landscape" not in parsed

    def test_blob_ref_stripped_from_source_options(self) -> None:
        """blob_ref is web-specific metadata and should not appear in engine YAML.

        The web composer tracks file provenance via blob_ref in source options,
        but plugin configs use Pydantic extra="forbid" and will reject it.
        The YAML generator must strip these web-only keys before output.
        """
        state = CompositionState(
            source=SourceSpec(
                plugin="text",
                on_success="out",
                options={
                    "path": "/data/input.txt",
                    "blob_ref": "20b944e3-fd46-434f-b9a2-4fb508db30f0",  # Should be stripped
                    "column": "line",
                },
                on_validation_failure="discard",
            ),
            nodes=(),
            edges=(),
            outputs=(OutputSpec(name="out", plugin="csv", options={}, on_write_failure="discard"),),
            metadata=PipelineMetadata(),
            version=1,
        )
        yaml_str = generate_yaml(state)
        parsed = yaml.safe_load(yaml_str)

        # blob_ref must not appear in the YAML
        assert "blob_ref" not in parsed["source"]["options"]
        # Other options should still be present
        assert parsed["source"]["options"]["path"] == "/data/input.txt"
        assert parsed["source"]["options"]["column"] == "line"

    def test_on_error_emitted_when_set(self) -> None:
        state = CompositionState(
            source=SourceSpec(
                plugin="csv",
                on_success="t1",
                options={},
                on_validation_failure="quarantine",
            ),
            nodes=(
                NodeSpec(
                    id="t1",
                    node_type="transform",
                    plugin="passthrough",
                    input="in",
                    on_success="out",
                    on_error="error_sink",
                    options={},
                    condition=None,
                    routes=None,
                    fork_to=None,
                    branches=None,
                    policy=None,
                    merge=None,
                ),
            ),
            edges=(),
            outputs=(
                OutputSpec(name="out", plugin="csv", options={}, on_write_failure="discard"),
                OutputSpec(name="error_sink", plugin="csv", options={}, on_write_failure="discard"),
            ),
            metadata=PipelineMetadata(),
            version=1,
        )
        yaml_str = generate_yaml(state)
        parsed = yaml.safe_load(yaml_str)
        assert parsed["transforms"][0]["on_error"] == "error_sink"

    def test_on_error_discard_emitted_for_transform(self) -> None:
        """on_error='discard' must appear in generated YAML, not be omitted."""
        state = CompositionState(
            source=SourceSpec(
                plugin="csv",
                on_success="t1",
                options={},
                on_validation_failure="quarantine",
            ),
            nodes=(
                NodeSpec(
                    id="t1",
                    node_type="transform",
                    plugin="passthrough",
                    input="in",
                    on_success="out",
                    on_error="discard",
                    options={},
                    condition=None,
                    routes=None,
                    fork_to=None,
                    branches=None,
                    policy=None,
                    merge=None,
                ),
            ),
            edges=(),
            outputs=(OutputSpec(name="out", plugin="csv", options={}, on_write_failure="discard"),),
            metadata=PipelineMetadata(),
            version=1,
        )
        yaml_str = generate_yaml(state)
        parsed = yaml.safe_load(yaml_str)
        assert parsed["transforms"][0]["on_error"] == "discard"

    def test_on_error_discard_emitted_for_aggregation(self) -> None:
        """Aggregation on_error='discard' must appear in generated YAML."""
        state = CompositionState(
            source=SourceSpec(
                plugin="csv",
                on_success="agg1",
                options={},
                on_validation_failure="quarantine",
            ),
            nodes=(
                NodeSpec(
                    id="agg1",
                    node_type="aggregation",
                    plugin="batch_counter",
                    input="in",
                    on_success="out",
                    on_error="discard",
                    options={"batch_size": 10},
                    condition=None,
                    routes=None,
                    fork_to=None,
                    branches=None,
                    policy=None,
                    merge=None,
                    trigger={"count": 5},
                ),
            ),
            edges=(),
            outputs=(OutputSpec(name="out", plugin="csv", options={}, on_write_failure="discard"),),
            metadata=PipelineMetadata(),
            version=1,
        )
        yaml_str = generate_yaml(state)
        parsed = yaml.safe_load(yaml_str)
        assert parsed["aggregations"][0]["on_error"] == "discard"

    def test_on_error_none_raises_for_transform(self) -> None:
        """on_error=None on a transform is a contract violation — generator must crash.

        The "discard" default belongs at the mutation boundary (upsert_node),
        not at the serialization layer. If on_error is still None here,
        upstream code has a bug.
        """
        state = CompositionState(
            source=SourceSpec(
                plugin="csv",
                on_success="t1",
                options={},
                on_validation_failure="quarantine",
            ),
            nodes=(
                NodeSpec(
                    id="t1",
                    node_type="transform",
                    plugin="passthrough",
                    input="in",
                    on_success="out",
                    on_error=None,
                    options={},
                    condition=None,
                    routes=None,
                    fork_to=None,
                    branches=None,
                    policy=None,
                    merge=None,
                ),
            ),
            edges=(),
            outputs=(OutputSpec(name="out", plugin="csv", options={}, on_write_failure="discard"),),
            metadata=PipelineMetadata(),
            version=1,
        )
        with pytest.raises(ValueError, match="on_error=None"):
            generate_yaml(state)

    def test_on_error_none_raises_for_aggregation(self) -> None:
        """on_error=None on an aggregation is a contract violation — generator must crash."""
        state = CompositionState(
            source=SourceSpec(
                plugin="csv",
                on_success="agg1",
                options={},
                on_validation_failure="quarantine",
            ),
            nodes=(
                NodeSpec(
                    id="agg1",
                    node_type="aggregation",
                    plugin="batch_counter",
                    input="in",
                    on_success="out",
                    on_error=None,
                    options={"batch_size": 10},
                    condition=None,
                    routes=None,
                    fork_to=None,
                    branches=None,
                    policy=None,
                    merge=None,
                    trigger={"count": 5},
                ),
            ),
            edges=(),
            outputs=(OutputSpec(name="out", plugin="csv", options={}, on_write_failure="discard"),),
            metadata=PipelineMetadata(),
            version=1,
        )
        with pytest.raises(ValueError, match="on_error=None"):
            generate_yaml(state)

    def test_aggregation_without_trigger_omits_key(self) -> None:
        """Aggregation with trigger=None must not crash yaml_generator."""
        state = CompositionState(
            source=SourceSpec(
                plugin="csv",
                on_success="agg1",
                options={},
                on_validation_failure="quarantine",
            ),
            nodes=(
                NodeSpec(
                    id="agg1",
                    node_type="aggregation",
                    plugin="batch_counter",
                    input="in",
                    on_success="out",
                    on_error="discard",
                    options={},
                    condition=None,
                    routes=None,
                    fork_to=None,
                    branches=None,
                    policy=None,
                    merge=None,
                    trigger=None,
                ),
            ),
            edges=(),
            outputs=(OutputSpec(name="out", plugin="csv", options={}, on_write_failure="discard"),),
            metadata=PipelineMetadata(),
            version=1,
        )
        yaml_str = generate_yaml(state)
        parsed = yaml.safe_load(yaml_str)
        # trigger absent from YAML — runtime treats this as end-of-source-only.
        assert "trigger" not in parsed["aggregations"][0]

    def test_frozen_state_serializes_without_error(self) -> None:
        """generate_yaml() handles frozen state objects (MappingProxyType, tuple).

        AC #15: No RepresenterError from PyYAML on frozen containers.
        Verifies that generate_yaml() correctly calls state.to_dict()
        before yaml.dump().
        """
        state = _make_linear_pipeline()
        # State has been through freeze_fields() -- options are MappingProxyType
        yaml_str = generate_yaml(state)
        parsed = yaml.safe_load(yaml_str)
        assert parsed["source"]["plugin"] == "csv"
        # Nested frozen options must serialize correctly
        assert parsed["source"]["options"]["schema"]["fields"] == ["name", "age"]

    def test_empty_state_minimal_yaml(self) -> None:
        """Empty state produces minimal valid YAML (no source, no sinks)."""
        state = CompositionState(
            source=None,
            nodes=(),
            edges=(),
            outputs=(),
            metadata=PipelineMetadata(),
            version=1,
        )
        yaml_str = generate_yaml(state)
        parsed = yaml.safe_load(yaml_str)
        # Empty state should produce an empty YAML doc (no source, no sinks)
        assert parsed is None or parsed == {}
