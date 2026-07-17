"""Production-path execution harness for the maintained DAG scenario corpus."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from string import Template
from typing import Any

from sqlalchemy import select

from elspeth.contracts import RunStatus
from elspeth.contracts.config.runtime import RuntimeCheckpointConfig
from elspeth.contracts.sink_effects import SinkEffectExecutionPurpose, SinkEffectInputKind
from elspeth.core.checkpoint import CheckpointManager, RecoveryManager
from elspeth.core.checkpoint.compatibility import CheckpointCompatibilityValidator
from elspeth.core.config import ElspethSettings, load_settings_from_yaml_string
from elspeth.core.dag import ExecutionGraph
from elspeth.core.landscape import LandscapeDB, LandscapeExporter, RecorderFactory
from elspeth.core.landscape.schema import node_states_table, token_work_items_table
from elspeth.core.payload_store import FilesystemPayloadStore
from elspeth.engine.orchestrator import Orchestrator, PipelineConfig
from elspeth.engine.orchestrator.preflight import (
    assemble_and_validate_pipeline_config,
    execution_sink_bindings_for_runtime,
    execution_sinks_for_runtime,
    sink_effect_modes_from_runtime_bindings,
    validate_pipeline_sink_effect_capabilities,
)
from elspeth.plugins.infrastructure.runtime_factory import PluginBundle, instantiate_plugins_from_config
from elspeth.plugins.transforms.llm.model_catalog import read_openrouter_catalog_snapshot_id
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
        input_csv=json.dumps(str(input_path)),
        output_jsonl=json.dumps(str(output_path)),
        fault_marker=json.dumps(str(fault_marker)),
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


def build_scenario(
    rendered: RenderedScenario,
    *,
    purpose: SinkEffectExecutionPurpose = SinkEffectExecutionPurpose.FRESH,
) -> BuiltScenario:
    """Build and validate a scenario through the production assembly sequence."""

    settings = rendered.settings
    bundle = instantiate_plugins_from_config(settings, preflight_mode=True, sink_effect_purpose=purpose)
    execution_sinks = execution_sinks_for_runtime(settings, bundle.sinks)
    if purpose is SinkEffectExecutionPurpose.RESUME:
        for sink_name, sink in execution_sinks.items():
            if not sink.supports_resume:
                raise ValueError(f"DAG scenario sink {sink_name!r} does not support resume")
            sink.configure_for_resume()
    execution_bindings = execution_sink_bindings_for_runtime(settings, bundle.sink_effect_bindings)
    sink_effect_modes = sink_effect_modes_from_runtime_bindings(
        execution_sinks,
        execution_bindings,
        purpose=purpose,
        configured_options={name: settings.sinks[name].options for name in execution_sinks},
    )
    sink_effect_admission = validate_pipeline_sink_effect_capabilities(
        execution_sinks,
        configured_modes=sink_effect_modes,
        required_input_kind=SinkEffectInputKind.PIPELINE_MEMBERS,
    )
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
        sink_effect_modes=sink_effect_modes,
        sink_effect_admission=sink_effect_admission,
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
        catalog_sha256, catalog_source = read_openrouter_catalog_snapshot_id()
        result = Orchestrator(db).run(
            built.config,
            graph=built.graph,
            settings=built.rendered.settings,
            payload_store=FilesystemPayloadStore(tmp_path / "payloads"),
            openrouter_catalog_sha256=catalog_sha256,
            openrouter_catalog_source=catalog_source,
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


def _require_exact_eof_crash(orchestrator: Orchestrator, built: BuiltScenario, payload_store: FilesystemPayloadStore) -> None:
    catalog_sha256, catalog_source = read_openrouter_catalog_snapshot_id()
    try:
        orchestrator.run(
            built.config,
            graph=built.graph,
            settings=built.rendered.settings,
            payload_store=payload_store,
            openrouter_catalog_sha256=catalog_sha256,
            openrouter_catalog_source=catalog_source,
        )
    except RuntimeError as exc:
        if str(exc) != "injected DAG corpus EOF flush crash":
            raise
    else:
        raise AssertionError("DAG recovery corpus run did not inject the EOF flush crash")


def _assert_terminal_recovery_state(
    db: LandscapeDB,
    *,
    run_id: str,
    checkpoint_id: str,
    payload_store: FilesystemPayloadStore,
) -> None:
    repositories = RecorderFactory.read_only(db, payload_store=payload_store)
    run = repositories.run_lifecycle.get_run(run_id)
    if run is None or run.status is not RunStatus.COMPLETED:
        raise AssertionError(f"DAG recovery corpus did not persist a completed run: {run!r}")
    source_records = repositories.run_lifecycle.get_run_source_lifecycle_records(run_id)
    if not source_records or any(record.lifecycle_state != "exhausted" for record in source_records.values()):
        raise AssertionError(f"DAG recovery corpus sources lost their exhausted state: {source_records!r}")

    tokens = repositories.query.get_all_tokens_for_run(run_id)
    outcomes = repositories.query.get_all_token_outcomes_for_run(run_id)
    latest_outcomes = {outcome.token_id: outcome for outcome in outcomes}
    token_ids = {token.token_id for token in tokens}
    if (
        not token_ids
        or set(latest_outcomes) != token_ids
        or not all(outcome.completed and outcome.outcome is not None for outcome in latest_outcomes.values())
    ):
        raise AssertionError(
            "DAG recovery corpus requires every token's latest exported outcome to be terminal: "
            f"tokens={sorted(token_ids)!r}, latest_outcomes={latest_outcomes!r}"
        )

    node_states = repositories.query.get_all_node_states_for_run(run_id)
    if not any(state.attempt > 0 for state in node_states):
        raise AssertionError("DAG recovery corpus requires a resumed node-state attempt")

    with db.connection() as conn:
        work_statuses = (
            conn.execute(select(token_work_items_table.c.status).where(token_work_items_table.c.run_id == run_id)).scalars().all()
        )
        resumed_markers = (
            conn.execute(
                select(node_states_table.c.resume_checkpoint_id).where(
                    node_states_table.c.run_id == run_id,
                    node_states_table.c.resume_checkpoint_id == checkpoint_id,
                )
            )
            .scalars()
            .all()
        )
    if not work_statuses or set(work_statuses) != {"terminal"}:
        raise AssertionError(f"DAG recovery corpus left non-terminal scheduler work: {work_statuses!r}")
    if not resumed_markers:
        raise AssertionError("DAG recovery corpus requires durable node-state resume checkpoint evidence")


def _recovery_case(scenario: ScenarioSpec, case: HarnessCaseSpec, tmp_path: Path) -> ScenarioRunEvidence:
    db_url = f"sqlite:///{tmp_path / 'audit.db'}"
    payload_root = tmp_path / "payloads"
    initial_rendered = render_settings(case, tmp_path)
    initial_built = build_scenario(initial_rendered)
    initial_store = FilesystemPayloadStore(payload_root)
    initial_db = LandscapeDB(db_url)
    initial_checkpoint_manager = CheckpointManager(initial_db)
    checkpoint_config = RuntimeCheckpointConfig.from_settings(initial_rendered.settings.checkpoint)
    initial_orchestrator = Orchestrator(
        initial_db,
        checkpoint_manager=initial_checkpoint_manager,
        checkpoint_config=checkpoint_config,
    )

    try:
        _require_exact_eof_crash(initial_orchestrator, initial_built, initial_store)
        initial_repositories = RecorderFactory.read_only(initial_db, payload_store=initial_store)
        runs = initial_repositories.run_lifecycle.list_runs()
        if len(runs) != 1:
            raise AssertionError(f"DAG recovery corpus expected exactly one failed run, got {len(runs)}")
        failed_run = runs[0]
        if failed_run.status is not RunStatus.FAILED:
            raise AssertionError(f"DAG recovery corpus expected failed run, got {failed_run.status.value!r}")
        run_id = failed_run.run_id
        source_records = initial_repositories.run_lifecycle.get_run_source_lifecycle_records(run_id)
        if not source_records or any(record.lifecycle_state != "exhausted" for record in source_records.values()):
            raise AssertionError(f"DAG recovery corpus sources were not exhausted before the crash: {source_records!r}")
        checkpoint = initial_checkpoint_manager.get_latest_checkpoint(run_id)
        if checkpoint is None:
            raise AssertionError("DAG recovery corpus crash did not preserve a checkpoint")
        if checkpoint.upstream_topology_hash != initial_built.graph_evidence.topology_hash:
            raise AssertionError("DAG recovery corpus checkpoint topology does not match the initial graph")
        checkpoint_id = checkpoint.checkpoint_id
        checkpoint_sequence = checkpoint.sequence_number
        checkpoint_topology_hash = checkpoint.upstream_topology_hash
    finally:
        initial_db.close()

    del initial_orchestrator, initial_checkpoint_manager, initial_repositories
    del initial_built, initial_rendered, initial_store, failed_run, source_records, checkpoint, runs

    reopened_db = LandscapeDB.from_url(db_url, create_tables=False)
    try:
        reopened_store = FilesystemPayloadStore(payload_root)
        reopened_checkpoint_manager = CheckpointManager(reopened_db)
        reopened_checkpoint = reopened_checkpoint_manager.get_latest_checkpoint(run_id)
        if reopened_checkpoint is None:
            raise AssertionError("DAG recovery corpus checkpoint disappeared across database reopen")
        if (
            reopened_checkpoint.checkpoint_id,
            reopened_checkpoint.sequence_number,
            reopened_checkpoint.upstream_topology_hash,
        ) != (checkpoint_id, checkpoint_sequence, checkpoint_topology_hash):
            raise AssertionError("DAG recovery corpus checkpoint changed across database reopen")

        fresh_rendered = render_settings(case, tmp_path)
        fresh_built = build_scenario(fresh_rendered, purpose=SinkEffectExecutionPurpose.RESUME)
        if fresh_built.graph_evidence.topology_hash != checkpoint_topology_hash:
            raise AssertionError("DAG recovery corpus fresh graph does not match the persisted checkpoint topology")
        recovery = RecoveryManager(reopened_db, reopened_checkpoint_manager)
        resume_check = recovery.can_resume(run_id, fresh_built.graph)
        if not resume_check.can_resume:
            raise AssertionError(f"DAG recovery corpus run is not resumable: {resume_check.reason}")
        resume_point = recovery.get_resume_point(run_id, fresh_built.graph)
        if resume_point is None:
            raise AssertionError("DAG recovery corpus did not produce a public resume point")
        if resume_point.checkpoint.checkpoint_id != checkpoint_id:
            raise AssertionError("DAG recovery corpus resume point does not use the reopened checkpoint")

        result = Orchestrator(
            reopened_db,
            checkpoint_manager=reopened_checkpoint_manager,
            checkpoint_config=checkpoint_config,
        ).resume(
            resume_point,
            fresh_built.config,
            fresh_built.graph,
            payload_store=reopened_store,
            settings=fresh_rendered.settings,
        )
        if result.run_id != run_id:
            raise AssertionError(f"DAG recovery corpus resumed the wrong run: expected {run_id!r}, got {result.run_id!r}")
        output_rows = [json.loads(line) for line in fresh_rendered.output_path.read_text(encoding="utf-8").splitlines()]
        if output_rows != [{"value": 60, "count": 3}]:
            raise AssertionError(f"DAG recovery corpus emitted unexpected output: {output_rows!r}")

        records = list(LandscapeExporter(reopened_db).export_run(run_id))
        audit = _audit_evidence(records)
        if audit.source_operation_count != 1:
            raise AssertionError(f"DAG recovery corpus replayed its source: source_load count={audit.source_operation_count}")
        _assert_terminal_recovery_state(
            reopened_db,
            run_id=run_id,
            checkpoint_id=checkpoint_id,
            payload_store=reopened_store,
        )
        if reopened_checkpoint_manager.get_latest_checkpoint(run_id) is not None:
            raise AssertionError("DAG recovery corpus retained a checkpoint after successful resume")
        if not fresh_rendered.fault_marker.is_file():
            raise AssertionError("DAG recovery corpus fault marker is missing after resume")

        result_data = result.to_dict()
        return ScenarioRunEvidence(
            schema_version=1,
            scenario_id=scenario.id,
            case_id=case.id,
            fixture_sha256=fresh_rendered.fixture_sha256,
            config=ConfigEvidence(loaded=True, settings_sha256=fresh_rendered.settings_sha256),
            graph=fresh_built.graph_evidence,
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
                attempted=True,
                database_reopened=True,
                checkpoint_id=checkpoint_id,
                checkpoint_sequence=checkpoint_sequence,
                can_resume=True,
                source_replayed=False,
                checkpoint_removed=True,
            ),
            completed_stages=("config", "build", "runtime", "audit", "recovery"),
        )
    finally:
        reopened_db.close()


def run_scenario_case(scenario: ScenarioSpec, case: HarnessCaseSpec, tmp_path: Path) -> ScenarioRunEvidence:
    """Execute a declared case through the workflow implemented for this task."""

    if case.workflow == "run":
        return _run_case(scenario, case, tmp_path)
    return _recovery_case(scenario, case, tmp_path)
