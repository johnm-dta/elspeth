"""Production-path execution harness for the maintained DAG scenario corpus."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from string import Template
from typing import Any

from elspeth.core.checkpoint.compatibility import CheckpointCompatibilityValidator
from elspeth.core.config import ElspethSettings, load_settings_from_yaml_string
from elspeth.core.dag import ExecutionGraph
from elspeth.core.landscape import LandscapeDB, LandscapeExporter
from elspeth.core.payload_store import FilesystemPayloadStore
from elspeth.engine.orchestrator import Orchestrator, PipelineConfig
from elspeth.engine.orchestrator.preflight import assemble_and_validate_pipeline_config, execution_sinks_for_runtime
from elspeth.plugins.infrastructure.runtime_factory import PluginBundle, instantiate_plugins_from_config
from tests.fixtures.dag_scenario_corpus.loader import resolve_fixture_path
from tests.fixtures.dag_scenario_corpus.schema import (
    AuditEvidence,
    AuditRecordCount,
    ConfigEvidence,
    GraphEvidence,
    HarnessCaseSpec,
    RecoveryEvidence,
    RuntimeEvidence,
    ScenarioRunEvidence,
    ScenarioSpec,
)


@dataclass(frozen=True, slots=True)
class RenderedScenario:
    settings: ElspethSettings
    settings_yaml: str
    settings_sha256: str
    fixture_sha256: str
    output_path: Path
    fault_marker: Path


@dataclass(frozen=True, slots=True)
class BuiltScenario:
    rendered: RenderedScenario
    bundle: PluginBundle
    graph: ExecutionGraph
    config: PipelineConfig
    graph_evidence: GraphEvidence


def render_settings(case: HarnessCaseSpec, tmp_path: Path) -> RenderedScenario:
    """Resolve and load one trusted corpus fixture without environment expansion."""

    fixture_path = resolve_fixture_path(case.fixture)
    input_path = resolve_fixture_path(case.input_fixture)
    output_path = tmp_path / "output.jsonl"
    fault_marker = tmp_path / "fault-triggered.marker"
    fixture_bytes = fixture_path.read_bytes()
    input_bytes = input_path.read_bytes()
    rendered = Template(fixture_bytes.decode("utf-8")).substitute(
        input_csv=str(input_path),
        output_jsonl=str(output_path),
        fault_marker=str(fault_marker),
    )
    if "${" in rendered:
        raise ValueError(f"Unresolved DAG scenario template variable in {fixture_path}")
    return RenderedScenario(
        settings=load_settings_from_yaml_string(rendered),
        settings_yaml=rendered,
        settings_sha256=hashlib.sha256(rendered.encode("utf-8")).hexdigest(),
        fixture_sha256=hashlib.sha256(fixture_bytes + b"\0" + input_bytes).hexdigest(),
        output_path=output_path,
        fault_marker=fault_marker,
    )


def build_scenario(rendered: RenderedScenario) -> BuiltScenario:
    """Build and validate a scenario through the production assembly sequence."""

    settings = rendered.settings
    bundle = instantiate_plugins_from_config(settings, preflight_mode=True)
    execution_sinks = execution_sinks_for_runtime(settings, bundle.sinks)
    graph = ExecutionGraph.from_plugin_instances(
        sources=bundle.sources,
        source_settings_map=bundle.source_settings_map,
        transforms=bundle.transforms,
        sinks=execution_sinks,
        aggregations=bundle.aggregations,
        gates=list(settings.gates),
        coalesce_settings=list(settings.coalesce) if settings.coalesce else None,
        queues=settings.queues,
    )
    graph.validate()
    graph.validate_edge_compatibility()
    config = assemble_and_validate_pipeline_config(
        sources=bundle.sources,
        transforms=bundle.transforms,
        sinks=bundle.sinks,
        aggregations=bundle.aggregations,
        settings=settings,
        graph=graph,
    )
    graph_evidence = GraphEvidence(
        accepted=True,
        node_count=len(graph.get_nodes()),
        edge_count=len(graph.get_edges()),
        topology_hash=CheckpointCompatibilityValidator().compute_full_topology_hash(graph),
    )
    return BuiltScenario(rendered, bundle, graph, config, graph_evidence)


def _audit_evidence(records: list[dict[str, Any]]) -> AuditEvidence:
    counts = Counter(str(record["record_type"]) for record in records)
    return AuditEvidence(
        attempted=True,
        total_records=len(records),
        record_counts=tuple(AuditRecordCount(record_type=record_type, count=count) for record_type, count in sorted(counts.items())),
        source_operation_count=sum(
            1 for record in records if record.get("record_type") == "operation" and record.get("operation_type") == "source_load"
        ),
    )


def _run_case(scenario: ScenarioSpec, case: HarnessCaseSpec, tmp_path: Path) -> ScenarioRunEvidence:
    rendered = render_settings(case, tmp_path)
    built = build_scenario(rendered)
    db = LandscapeDB(f"sqlite:///{tmp_path / 'audit.db'}")
    try:
        result = Orchestrator(db).run(
            built.config,
            graph=built.graph,
            settings=built.rendered.settings,
            payload_store=FilesystemPayloadStore(tmp_path / "payloads"),
        )
        output_rows = [json.loads(line) for line in rendered.output_path.read_text(encoding="utf-8").splitlines()]
        audit = _audit_evidence(list(LandscapeExporter(db).export_run(result.run_id)))
        result_data = result.to_dict()
        return ScenarioRunEvidence(
            schema_version=1,
            scenario_id=scenario.id,
            case_id=case.id,
            fixture_sha256=rendered.fixture_sha256,
            config=ConfigEvidence(loaded=True, settings_sha256=rendered.settings_sha256),
            graph=built.graph_evidence,
            runtime=RuntimeEvidence(
                attempted=True,
                run_id=result.run_id,
                status=str(result_data["status"]),
                rows_processed=result_data["rows_processed"],
                rows_succeeded=result_data["rows_succeeded"],
                rows_failed=result_data["rows_failed"],
                output_rows=len(output_rows),
            ),
            audit=audit,
            recovery=RecoveryEvidence(
                attempted=False,
                database_reopened=False,
                can_resume=False,
                source_replayed=False,
                checkpoint_removed=False,
            ),
            completed_stages=("config", "build", "runtime", "audit"),
        )
    finally:
        db.close()


def run_scenario_case(scenario: ScenarioSpec, case: HarnessCaseSpec, tmp_path: Path) -> ScenarioRunEvidence:
    """Execute a declared case through the workflow implemented for this task."""

    if case.workflow == "run":
        return _run_case(scenario, case, tmp_path)
    raise NotImplementedError("DAG scenario recovery workflow is implemented in Task 5")
