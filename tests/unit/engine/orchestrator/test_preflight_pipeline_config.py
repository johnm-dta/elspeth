"""Tests for shared PipelineConfig assembly."""

from __future__ import annotations

from dataclasses import dataclass, field

from elspeth.engine.orchestrator.preflight import assemble_and_validate_pipeline_config


@dataclass
class _SourceStub:
    name: str = "text"
    _on_validation_failure: str = "discard"


@dataclass
class _SinkStub:
    name: str = "json"
    _on_write_failure: str = "discard"


@dataclass
class _SettingsStub:
    gates: list[object] = field(default_factory=list)
    coalesce: list[object] = field(default_factory=list)

    def model_dump(self, *, mode: str = "python") -> dict[str, object]:
        assert mode == "json"
        return {
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


class _GraphStub:
    def get_aggregation_id_map(self) -> dict[object, object]:
        return {}

    def get_route_resolution_map(self) -> dict[object, object]:
        return {}

    def get_transform_id_map(self) -> dict[object, object]:
        return {}

    def get_config_gate_id_map(self) -> dict[object, object]:
        return {}


def test_assemble_pipeline_config_preserves_resolved_audit_config() -> None:
    """The shared runtime assembly path must not create empty Landscape run configs."""
    source = _SourceStub()
    sink = _SinkStub()
    settings = _SettingsStub()
    graph = _GraphStub()

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
