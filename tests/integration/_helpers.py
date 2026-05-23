"""Integration-test helpers for ADR-019 behaviour-change verification."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from elspeth.cli_helpers import instantiate_plugins_from_config
from elspeth.contracts.run_result import RunResult
from elspeth.core.config import ElspethSettings, load_settings_from_yaml_string
from elspeth.core.dag import ExecutionGraph
from elspeth.core.landscape.database import LandscapeDB
from elspeth.core.payload_store import FilesystemPayloadStore
from elspeth.engine.orchestrator import Orchestrator, PipelineConfig
from elspeth.engine.orchestrator.preflight import assemble_and_validate_pipeline_config
from tests.integration._adr019_test_plugins import install_adr019_test_plugin_manager


def make_settings_yaml_for_test_plugins(
    *,
    source_plugin: str,
    source_config: dict[str, object],
    sinks: dict[str, dict[str, object]],
    transforms: list[dict[str, object]] | None = None,
    gates: list[dict[str, object]] | None = None,
) -> str:
    """Serialize the minimal production settings shape used by the helpers."""
    source_options = dict(source_config)
    source_on_success = str(source_options.pop("on_success"))
    sink_payload: dict[str, dict[str, object]] = {}
    for sink_name, sink_spec in sinks.items():
        sink_payload[sink_name] = {
            "plugin": sink_spec["plugin"],
            "on_write_failure": sink_spec.get("on_write_failure", "discard"),
            "options": sink_spec.get("options", sink_spec.get("config", {})),
        }
    payload: dict[str, object] = {
        "sources": {
            "primary": {
                "plugin": source_plugin,
                "on_success": source_on_success,
                "options": source_options,
            }
        },
        "sinks": sink_payload,
    }
    if transforms:
        payload["transforms"] = transforms
    if gates:
        payload["gates"] = gates
    return yaml.safe_dump(payload, sort_keys=False)


def _make_db_and_store(tmp_path: Path) -> tuple[LandscapeDB, FilesystemPayloadStore]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    db = LandscapeDB(f"sqlite:///{tmp_path / 'audit.db'}")
    store = FilesystemPayloadStore(tmp_path / "payloads")
    return db, store


def _pipeline_from_settings(
    settings: ElspethSettings,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[PipelineConfig, ExecutionGraph, LandscapeDB, FilesystemPayloadStore]:
    """Build a pipeline through production configuration and graph assembly."""
    db, store = _make_db_and_store(tmp_path)
    install_adr019_test_plugin_manager(monkeypatch)
    bundle = instantiate_plugins_from_config(settings)
    graph = ExecutionGraph.from_plugin_instances(
        sources=bundle.sources,
        source_settings_map=bundle.source_settings_map,
        transforms=bundle.transforms,
        sinks=bundle.sinks,
        aggregations=bundle.aggregations,
        gates=list(settings.gates),
        coalesce_settings=(list(settings.coalesce) if settings.coalesce else None),
    )
    config = assemble_and_validate_pipeline_config(
        sources=bundle.sources,
        transforms=bundle.transforms,
        sinks=bundle.sinks,
        aggregations=bundle.aggregations,
        settings=settings,
        graph=graph,
    )
    return config, graph, db, store


def _yaml_for_discard_sink(rows: list[dict[str, object]], discard_row_count: int) -> str:
    return make_settings_yaml_for_test_plugins(
        source_plugin="list_source",
        source_config={"rows": rows, "on_success": "default"},
        sinks={
            "default": {
                "plugin": "diverting_sink",
                "config": {"divert_count": discard_row_count},
            },
        },
    )


def _yaml_for_failsink_diversion(
    rows: list[dict[str, object]],
    diverted_row_count: int,
) -> str:
    return make_settings_yaml_for_test_plugins(
        source_plugin="list_source",
        source_config={"rows": rows, "on_success": "default"},
        sinks={
            "default": {
                "plugin": "diverting_sink",
                "on_write_failure": "failsink",
                "config": {"divert_count": diverted_row_count},
            },
            "failsink": {
                "plugin": "json",
                "config": {"name": "failsink"},
            },
        },
    )


def _yaml_for_source_quarantine(rows: list[dict[str, object]]) -> str:
    return make_settings_yaml_for_test_plugins(
        source_plugin="quarantine_source",
        source_config={
            "rows": rows,
            "on_success": "default",
            "quarantine_destination": "quarantine",
            "on_validation_failure": "quarantine",
        },
        sinks={
            "default": {"plugin": "collect_sink", "config": {"name": "default"}},
            "quarantine": {"plugin": "collect_sink", "config": {"name": "quarantine"}},
        },
    )


def _yaml_for_gate_route(rows: list[dict[str, object]]) -> str:
    # NOTE: the sink that captures the gate "false" branch is named
    # ``gate_default_sink`` (not ``primary``) because the migration to plural
    # ``sources:`` uses the source-name token ``primary`` by default; reusing
    # ``primary`` for a sink would collide with the source key and fail
    # ``ElspethSettings`` validation ("Node name 'primary' is used by both
    # source and sink").
    return make_settings_yaml_for_test_plugins(
        source_plugin="list_source",
        source_config={"rows": rows, "on_success": "default"},
        gates=[
            {
                "name": "gate",
                "input": "default",
                "condition": "row['route'] == 'move'",
                "routes": {"true": "routed", "false": "gate_default_sink"},
            }
        ],
        sinks={
            "routed": {"plugin": "collect_sink", "config": {"name": "routed"}},
            "gate_default_sink": {"plugin": "collect_sink", "config": {"name": "gate_default_sink"}},
        },
    )


def _yaml_for_on_error_route(rows: list[dict[str, object]]) -> str:
    # NOTE: ``primary`` would collide with the implicit source key; the
    # success sink is named ``success_sink`` instead. See ``_yaml_for_gate_route``.
    return make_settings_yaml_for_test_plugins(
        source_plugin="list_source",
        source_config={"rows": rows, "on_success": "default"},
        transforms=[
            {
                "name": "maybe_fail",
                "plugin": "conditional_error",
                "input": "default",
                "on_success": "success_sink",
                "on_error": "error_sink",
                "options": {},
            }
        ],
        sinks={
            "success_sink": {"plugin": "collect_sink", "config": {"name": "success_sink"}},
            "error_sink": {"plugin": "collect_sink", "config": {"name": "error_sink"}},
        },
    )


def build_test_pipeline_with_discard_sink(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    success_row_count: int,
    discard_row_count: int,
) -> tuple[PipelineConfig, ExecutionGraph, LandscapeDB, FilesystemPayloadStore]:
    """Pipeline with successful rows plus discard-mode sink diversions."""
    rows: list[dict[str, object]] = [{"id": i, "expected": "discard"} for i in range(discard_row_count)] + [
        {"id": discard_row_count + i, "expected": "success"} for i in range(success_row_count)
    ]
    settings = load_settings_from_yaml_string(_yaml_for_discard_sink(rows, discard_row_count))
    return _pipeline_from_settings(settings, tmp_path, monkeypatch)


def build_test_pipeline_with_gate_route(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    routed_row_count: int,
    default_flow_row_count: int,
) -> tuple[PipelineConfig, ExecutionGraph, LandscapeDB, FilesystemPayloadStore]:
    """Pipeline with a config gate exercising ``(SUCCESS, GATE_ROUTED)``."""
    rows: list[dict[str, object]] = [{"id": i, "route": "move"} for i in range(routed_row_count)] + [
        {"id": routed_row_count + i, "route": "default"} for i in range(default_flow_row_count)
    ]
    settings = load_settings_from_yaml_string(_yaml_for_gate_route(rows))
    return _pipeline_from_settings(settings, tmp_path, monkeypatch)


def build_test_pipeline_with_failsink_diversion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    diverted_row_count: int,
    success_row_count: int,
) -> tuple[PipelineConfig, ExecutionGraph, LandscapeDB, FilesystemPayloadStore]:
    """Pipeline where diverted sink rows go to a configured failsink."""
    rows: list[dict[str, object]] = [{"id": i, "expected": "failsink"} for i in range(diverted_row_count)] + [
        {"id": diverted_row_count + i, "expected": "success"} for i in range(success_row_count)
    ]
    settings = load_settings_from_yaml_string(_yaml_for_failsink_diversion(rows, diverted_row_count))
    return _pipeline_from_settings(settings, tmp_path, monkeypatch)


def build_test_pipeline_with_source_quarantine(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    quarantine_row_count: int,
) -> tuple[PipelineConfig, ExecutionGraph, LandscapeDB, FilesystemPayloadStore]:
    """Pipeline whose source yields quarantined rows to a quarantine sink."""
    rows: list[dict[str, object]] = [{"id": i, "invalid": True} for i in range(quarantine_row_count)]
    settings = load_settings_from_yaml_string(_yaml_for_source_quarantine(rows))
    return _pipeline_from_settings(settings, tmp_path, monkeypatch)


def build_test_pipeline_with_on_error_route(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    on_error_routed_count: int,
    success_count: int,
) -> tuple[PipelineConfig, ExecutionGraph, LandscapeDB, FilesystemPayloadStore]:
    """Pipeline with transform on_error routing to an error sink."""
    rows: list[dict[str, object]] = []
    rows.extend({"id": i, "fail": True} for i in range(on_error_routed_count))
    rows.extend({"id": on_error_routed_count + i, "fail": False} for i in range(success_count))
    settings = load_settings_from_yaml_string(_yaml_for_on_error_route(rows))
    return _pipeline_from_settings(settings, tmp_path, monkeypatch)


def run_pipeline(
    config: PipelineConfig,
    graph: ExecutionGraph,
    db: LandscapeDB,
    store: FilesystemPayloadStore,
) -> RunResult:
    """Run a helper pipeline through the production orchestrator path."""
    return Orchestrator(db).run(
        config,
        graph=graph,
        payload_store=store,
    )


__all__ = [
    "build_test_pipeline_with_discard_sink",
    "build_test_pipeline_with_failsink_diversion",
    "build_test_pipeline_with_gate_route",
    "build_test_pipeline_with_on_error_route",
    "build_test_pipeline_with_source_quarantine",
    "make_settings_yaml_for_test_plugins",
    "run_pipeline",
]
