"""Tests for shared PipelineConfig assembly."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

from elspeth.engine.orchestrator.preflight import assemble_and_validate_pipeline_config


def test_assemble_pipeline_config_preserves_resolved_audit_config() -> None:
    """The shared runtime assembly path must not create empty Landscape run configs."""
    source = MagicMock()
    source.name = "text"
    source._on_validation_failure = "discard"
    sink = MagicMock()
    sink.name = "json"
    sink._on_write_failure = "discard"
    settings = SimpleNamespace(
        gates=[],
        coalesce=[],
        model_dump=MagicMock(
            return_value={
                "sources": {
                    "primary": {
                        "plugin": "text",
                        "options": {"path": "/tmp/input.txt", "column": "value"},
                    }
                },
                "transforms": [],
                "sinks": {
                    "output": {
                        "plugin": "json",
                        "options": {"path": "/tmp/out.jsonl", "format": "jsonl"},
                    }
                },
            }
        ),
    )
    graph = MagicMock()
    graph.get_aggregation_id_map.return_value = {}
    graph.get_route_resolution_map.return_value = {}
    graph.get_transform_id_map.return_value = {}
    graph.get_config_gate_id_map.return_value = {}

    pipeline_config = assemble_and_validate_pipeline_config(
        sources={"primary": source},
        transforms=[],
        sinks={"output": sink},
        aggregations={},
        settings=settings,
        graph=graph,
    )

    # The runtime config is computed via resolve_config(settings) which still
    # exposes ``sources.primary.plugin`` on the underlying ElspethSettings.
    assert pipeline_config.config["sources"]["primary"]["plugin"] == "text"
    assert pipeline_config.config["sinks"]["output"]["plugin"] == "json"
