"""ADR-019 Phase 4 sweep and real-time invariant durability tests."""

from __future__ import annotations

import json
from collections.abc import Callable
from datetime import UTC, datetime

import pytest

from elspeth.contracts import NodeType, RunStatus
from elspeth.contracts.audit import DISCARD_SINK_NAME, TokenRef
from elspeth.contracts.enums import BatchStatus, Determinism, NodeStateStatus, RoutingMode, TerminalOutcome, TerminalPath
from elspeth.contracts.errors import AuditIntegrityError, GracefulShutdownError
from elspeth.contracts.runtime_val_manifest import build_runtime_val_manifest
from elspeth.contracts.schema import SchemaConfig
from elspeth.contracts.schema_contract import FieldContract, SchemaContract
from elspeth.core.canonical import canonical_json
from elspeth.core.landscape.factory import RecorderFactory
from elspeth.engine.orchestrator import Orchestrator, PipelineConfig, prepare_for_run
from tests.fixtures.base_classes import as_sink, as_source, as_transform
from tests.fixtures.pipeline import build_linear_pipeline
from tests.fixtures.plugins import CollectSink, PassTransform
from tests.fixtures.stores import MockPayloadStore

_DYNAMIC_SCHEMA = SchemaConfig.from_dict({"mode": "observed"})
_ERROR_HASH = "facefeed" * 8


def _runtime_val_manifest_json() -> str:
    prepare_for_run()
    return canonical_json(build_runtime_val_manifest())


def _build_minimal_run():
    source, _transforms, sinks, graph = build_linear_pipeline([{"value": 1}], transforms=[])
    config = PipelineConfig(
        source=as_source(source),
        transforms=[],
        sinks={"default": as_sink(sinks["default"])},
    )
    return config, graph


def _plant_orphan_fork_parent(
    factory: RecorderFactory,
    run_id: str,
    *,
    row_index: int = 900,
    complete_row_for_resume: bool = False,
) -> str:
    source = factory.data_flow.register_node(
        run_id=run_id,
        plugin_name=f"durability_source_{row_index}",
        node_type=NodeType.SOURCE,
        plugin_version="1.0",
        config={},
        schema_config=_DYNAMIC_SCHEMA,
    )
    row = factory.data_flow.create_row(
        run_id=run_id,
        source_node_id=source.node_id,
        row_index=row_index,
        data={"planted": True},
        source_row_index=row_index,
        ingest_sequence=row_index,
    )
    token = factory.data_flow.create_token(row_id=row.row_id)
    factory.data_flow.record_token_outcome(
        ref=TokenRef(token_id=token.token_id, run_id=run_id),
        outcome=TerminalOutcome.TRANSIENT,
        path=TerminalPath.FORK_PARENT,
        fork_group_id=f"fg_durability_{row_index}",
    )
    if complete_row_for_resume:
        sibling = factory.data_flow.create_token(row_id=row.row_id)
        factory.data_flow.record_token_outcome(
            ref=TokenRef(token_id=sibling.token_id, run_id=run_id),
            outcome=TerminalOutcome.SUCCESS,
            path=TerminalPath.DEFAULT_FLOW,
            sink_name="default",
        )
    return token.token_id


def _plant_orphan_batch_consumed(
    factory: RecorderFactory,
    run_id: str,
    *,
    row_index: int = 901,
    complete_row_for_resume: bool = False,
) -> str:
    del complete_row_for_resume
    source = factory.data_flow.register_node(
        run_id=run_id,
        plugin_name=f"durability_batch_source_{row_index}",
        node_type=NodeType.SOURCE,
        plugin_version="1.0",
        config={},
        schema_config=_DYNAMIC_SCHEMA,
    )
    row = factory.data_flow.create_row(
        run_id=run_id,
        source_node_id=source.node_id,
        row_index=row_index,
        data={"planted_i1b": True},
        source_row_index=row_index,
        ingest_sequence=row_index,
    )
    token = factory.data_flow.create_token(row_id=row.row_id)
    batch_id = f"batch_durability_{row_index}"
    factory.execution.create_batch(
        run_id=run_id,
        aggregation_node_id=source.node_id,
        batch_id=batch_id,
    )
    factory.data_flow.record_token_outcome(
        ref=TokenRef(token_id=token.token_id, run_id=run_id),
        outcome=TerminalOutcome.TRANSIENT,
        path=TerminalPath.BATCH_CONSUMED,
        batch_id=batch_id,
    )
    return token.token_id


def _assert_failed_with_preserved_outcome(
    factory: RecorderFactory,
    run_id: str,
    token_id: str,
    expected_path: TerminalPath,
) -> None:
    run_row = factory.run_lifecycle.get_run(run_id)
    assert run_row is not None
    assert run_row.status == RunStatus.FAILED

    outcome = factory.data_flow.get_token_outcome(token_id)
    assert outcome is not None
    assert outcome.outcome == TerminalOutcome.TRANSIENT
    assert outcome.path == expected_path
    if expected_path == TerminalPath.BATCH_CONSUMED:
        assert outcome.batch_id is not None
        batch = factory.execution.get_batch(outcome.batch_id)
        assert batch is not None
        assert batch.status != BatchStatus.COMPLETED


PlantFn = Callable[..., str]


@pytest.mark.parametrize(
    ("label", "plant", "expected_path"),
    [
        ("I1a", _plant_orphan_fork_parent, TerminalPath.FORK_PARENT),
        ("I1b", _plant_orphan_batch_consumed, TerminalPath.BATCH_CONSUMED),
    ],
)
def test_fresh_run_sweep_crash_finalizes_failed_and_preserves_evidence(
    monkeypatch: pytest.MonkeyPatch,
    label: str,
    plant: PlantFn,
    expected_path: TerminalPath,
) -> None:
    from elspeth.core.landscape.database import LandscapeDB

    db = LandscapeDB.in_memory()
    payload_store = MockPayloadStore()
    captured: dict[str, str] = {}
    original_init = Orchestrator._initialize_database_phase

    def _init_and_plant(
        self: Orchestrator,
        config,
        payload_store,
        secret_resolutions,
        *,
        run_id=None,
        initiated_by_user_id=None,
        auth_provider_type=None,
        openrouter_catalog_sha256: str = "0" * 64,
        openrouter_catalog_source: str = "bundled",
    ):
        factory, run = original_init(
            self,
            config,
            payload_store,
            secret_resolutions,
            run_id=run_id,
            initiated_by_user_id=initiated_by_user_id,
            auth_provider_type=auth_provider_type,
            openrouter_catalog_sha256=openrouter_catalog_sha256,
            openrouter_catalog_source=openrouter_catalog_source,
        )
        captured["run_id"] = run.run_id
        captured["token_id"] = plant(factory, run.run_id)
        return factory, run

    monkeypatch.setattr(Orchestrator, "_initialize_database_phase", _init_and_plant)
    config, graph = _build_minimal_run()

    with pytest.raises(AuditIntegrityError, match=label):
        Orchestrator(db).run(config, graph=graph, payload_store=payload_store)

    _assert_failed_with_preserved_outcome(
        RecorderFactory(db),
        captured["run_id"],
        captured["token_id"],
        expected_path,
    )


def _setup_adr019_failed_resume_run(
    db,
    payload_store,
    run_id: str,
    *,
    num_rows: int,
    processed_count: int,
):
    from sqlalchemy import insert

    from elspeth.contracts.contract_records import ContractAuditRecord
    from elspeth.core.checkpoint import CheckpointManager
    from elspeth.core.landscape.schema import edges_table, nodes_table, rows_table, runs_table, tokens_table

    now = datetime.now(UTC)
    source_data = [{"value": i} for i in range(num_rows)]
    transform = PassTransform()
    _, _, _, graph = build_linear_pipeline(source_data, transforms=[as_transform(transform)])

    source_nid = graph.get_source()
    assert source_nid is not None
    transform_id_map = graph.get_transform_id_map()
    sink_id_map = graph.get_sink_id_map()
    xform_nid = str(transform_id_map[0])
    sink_nid = str(next(iter(sink_id_map.values())))

    contract = SchemaContract(
        mode="FIXED",
        fields=(
            FieldContract(
                normalized_name="value",
                original_name="value",
                python_type=int,
                required=True,
                source="declared",
            ),
        ),
        locked=True,
    )
    audit_record = ContractAuditRecord.from_contract(contract)

    with db.engine.begin() as conn:
        conn.execute(
            insert(runs_table).values(
                run_id=run_id,
                started_at=now,
                config_hash="test",
                settings_json="{}",
                canonical_version="v1",
                status=RunStatus.FAILED,
                source_schema_json=json.dumps({"properties": {"value": {"type": "integer"}}, "required": ["value"]}),
                schema_contract_json=audit_record.to_json(),
                schema_contract_hash=contract.version_hash(),
                runtime_val_manifest_json=_runtime_val_manifest_json(),
                openrouter_catalog_sha256="0" * 64,
                openrouter_catalog_source="bundled",
            )
        )
        for node_id, plugin_name, node_type in [
            (source_nid, "list_source", NodeType.SOURCE),
            (xform_nid, "passthrough", NodeType.TRANSFORM),
            (sink_nid, "collect_sink", NodeType.SINK),
        ]:
            conn.execute(
                insert(nodes_table).values(
                    node_id=node_id,
                    run_id=run_id,
                    plugin_name=plugin_name,
                    node_type=node_type,
                    plugin_version="1.0.0",
                    determinism=Determinism.DETERMINISTIC if node_type != NodeType.SINK else Determinism.IO_WRITE,
                    config_hash="test",
                    config_json="{}",
                    registered_at=now,
                )
            )
        for edge_id, from_node, to_node in [
            ("e1", source_nid, xform_nid),
            ("e2", xform_nid, sink_nid),
        ]:
            conn.execute(
                insert(edges_table).values(
                    edge_id=edge_id,
                    run_id=run_id,
                    from_node_id=from_node,
                    to_node_id=to_node,
                    label="continue",
                    default_mode=RoutingMode.MOVE,
                    created_at=now,
                )
            )
        for i in range(num_rows):
            row_data = {"value": i}
            ref = payload_store.store(json.dumps(row_data).encode())
            conn.execute(
                insert(rows_table).values(
                    row_id=f"r{i}",
                    run_id=run_id,
                    source_node_id=source_nid,
                    row_index=i,
                    source_row_index=i,
                    ingest_sequence=i,
                    source_data_hash=f"h{i}",
                    source_data_ref=ref,
                    created_at=now,
                )
            )
            conn.execute(
                insert(tokens_table).values(
                    token_id=f"t{i}",
                    row_id=f"r{i}",
                    run_id=run_id,
                    created_at=now,
                )
            )

    factory = RecorderFactory(db)
    for i in range(processed_count):
        factory.data_flow.record_token_outcome(
            ref=TokenRef(token_id=f"t{i}", run_id=run_id),
            outcome=TerminalOutcome.SUCCESS,
            path=TerminalPath.DEFAULT_FLOW,
            sink_name="default",
        )

    if processed_count > 0:
        CheckpointManager(db).create_checkpoint(
            run_id=run_id,
            token_id=f"t{processed_count - 1}",
            node_id=xform_nid,
            sequence_number=processed_count - 1,
            graph=graph,
        )

    return graph


def _build_resume_environment(run_id: str, *, num_rows: int, processed_count: int):
    from elspeth.contracts.config.runtime import RuntimeCheckpointConfig
    from elspeth.core.checkpoint import CheckpointManager, RecoveryManager
    from elspeth.core.config import CheckpointSettings
    from elspeth.core.landscape.database import LandscapeDB
    from elspeth.plugins.sources.null_source import NullSource
    from elspeth.plugins.transforms.passthrough import PassThrough

    db = LandscapeDB.in_memory()
    payload_store = MockPayloadStore()
    checkpoint_mgr = CheckpointManager(db)
    checkpoint_config = RuntimeCheckpointConfig.from_settings(CheckpointSettings(enabled=True, frequency="every_row"))
    graph = _setup_adr019_failed_resume_run(
        db,
        payload_store,
        run_id,
        num_rows=num_rows,
        processed_count=processed_count,
    )

    resume_point = RecoveryManager(db, checkpoint_mgr).get_resume_point(run_id, graph)
    assert resume_point is not None

    transform = PassThrough({"schema": {"mode": "observed"}})
    transform.on_success = "default"
    transform.on_error = "discard"
    source = NullSource({})
    source.on_success = "default"
    sink = CollectSink()
    config = PipelineConfig(
        source=as_source(source),
        transforms=[as_transform(transform)],
        sinks={"default": as_sink(sink)},
    )
    orchestrator = Orchestrator(
        db=db,
        checkpoint_manager=checkpoint_mgr,
        checkpoint_config=checkpoint_config,
    )
    return db, payload_store, graph, resume_point, config, orchestrator


@pytest.mark.parametrize(
    ("label", "plant", "expected_path"),
    [
        ("I1a", _plant_orphan_fork_parent, TerminalPath.FORK_PARENT),
        ("I1b", _plant_orphan_batch_consumed, TerminalPath.BATCH_CONSUMED),
    ],
)
def test_resume_sweep_crash_finalizes_failed_and_preserves_evidence(
    label: str,
    plant: PlantFn,
    expected_path: TerminalPath,
) -> None:
    run_id = f"adr019-resume-{label.lower()}"
    db, payload_store, graph, resume_point, config, orchestrator = _build_resume_environment(
        run_id,
        num_rows=4,
        processed_count=2,
    )
    factory = RecorderFactory(db)
    token_id = plant(factory, run_id, complete_row_for_resume=True)

    with pytest.raises(AuditIntegrityError, match=label):
        orchestrator.resume(
            resume_point=resume_point,
            config=config,
            graph=graph,
            payload_store=payload_store,
        )

    _assert_failed_with_preserved_outcome(factory, run_id, token_id, expected_path)


@pytest.mark.parametrize(
    ("label", "plant", "expected_path"),
    [
        ("I1a", _plant_orphan_fork_parent, TerminalPath.FORK_PARENT),
        ("I1b", _plant_orphan_batch_consumed, TerminalPath.BATCH_CONSUMED),
    ],
)
def test_resume_no_work_sweep_crash_finalizes_failed_and_preserves_evidence(
    monkeypatch: pytest.MonkeyPatch,
    label: str,
    plant: PlantFn,
    expected_path: TerminalPath,
) -> None:
    run_id = f"adr019-resume-no-work-{label.lower()}"
    db, payload_store, graph, resume_point, config, orchestrator = _build_resume_environment(
        run_id,
        num_rows=2,
        processed_count=2,
    )
    factory = RecorderFactory(db)
    token_id = plant(factory, run_id, complete_row_for_resume=True)

    process_calls: list[str] = []

    def _fail_if_processed(self: Orchestrator, *args, **kwargs):
        process_calls.append("process")
        raise AssertionError("no-work resume branch must not call _process_resumed_rows")

    monkeypatch.setattr(Orchestrator, "_process_resumed_rows", _fail_if_processed)

    with pytest.raises(AuditIntegrityError, match=label):
        orchestrator.resume(
            resume_point=resume_point,
            config=config,
            graph=graph,
            payload_store=payload_store,
        )

    assert process_calls == []
    _assert_failed_with_preserved_outcome(factory, run_id, token_id, expected_path)


@pytest.mark.parametrize("kind", ["I1c", "I3"])
def test_realtime_invariant_crash_finalizes_failed_and_preserves_witnesses(
    monkeypatch: pytest.MonkeyPatch,
    kind: str,
) -> None:
    from elspeth.core.landscape.database import LandscapeDB

    db = LandscapeDB.in_memory()
    payload_store = MockPayloadStore()
    captured: dict[str, str] = {}
    original_loop = Orchestrator._run_main_processing_loop

    def _corrupting_loop(self: Orchestrator, loop_ctx, factory, run_id, source_id, edge_map, *, shutdown_event=None):
        captured["run_id"] = run_id
        sink = factory.data_flow.register_node(
            run_id=run_id,
            plugin_name=f"corrupt_{kind.lower()}_sink",
            node_type=NodeType.SINK,
            plugin_version="1.0",
            config={},
            schema_config=_DYNAMIC_SCHEMA,
        )
        row = factory.data_flow.create_row(
            run_id=run_id,
            source_node_id=source_id,
            row_index=700 if kind == "I1c" else 701,
            data={"corrupt": kind},
            source_row_index=700 if kind == "I1c" else 701,
            ingest_sequence=700 if kind == "I1c" else 701,
        )
        token = factory.data_flow.create_token(row_id=row.row_id)
        captured["token_id"] = token.token_id
        state = factory.execution.begin_node_state(
            token_id=token.token_id,
            node_id=sink.node_id,
            run_id=run_id,
            step_index=0,
            input_data={},
        )
        factory.execution.complete_node_state(
            state_id=state.state_id,
            status=NodeStateStatus.COMPLETED,
            output_data={"written": True},
            duration_ms=1.0,
        )
        if kind == "I1c":
            sibling = factory.data_flow.register_node(
                run_id=run_id,
                plugin_name="wrong_failsink",
                node_type=NodeType.SINK,
                plugin_version="1.0",
                config={},
                schema_config=_DYNAMIC_SCHEMA,
            )
            artifact = factory.execution.register_artifact(
                run_id=run_id,
                state_id=state.state_id,
                sink_node_id=sink.node_id,
                artifact_type="test",
                path="memory://wrong-failsink",
                content_hash="deadbeef" * 8,
                size_bytes=0,
            )
            captured["artifact_id"] = artifact.artifact_id
            factory.data_flow.record_token_outcome(
                ref=TokenRef(token_id=token.token_id, run_id=run_id),
                outcome=TerminalOutcome.TRANSIENT,
                path=TerminalPath.SINK_FALLBACK_TO_FAILSINK,
                sink_name="failsink",
                sink_node_id=sibling.node_id,
                artifact_id=artifact.artifact_id,
                error_hash=_ERROR_HASH,
            )
        else:
            factory.data_flow.record_token_outcome(
                ref=TokenRef(token_id=token.token_id, run_id=run_id),
                outcome=TerminalOutcome.FAILURE,
                path=TerminalPath.SINK_DISCARDED,
                sink_name=DISCARD_SINK_NAME,
                error_hash=_ERROR_HASH,
            )
        return original_loop(self, loop_ctx, factory, run_id, source_id, edge_map, shutdown_event=None)

    monkeypatch.setattr(Orchestrator, "_run_main_processing_loop", _corrupting_loop)
    config, graph = _build_minimal_run()

    with pytest.raises(AuditIntegrityError, match=kind):
        Orchestrator(db).run(config, graph=graph, payload_store=payload_store)

    factory = RecorderFactory(db)
    run_row = factory.run_lifecycle.get_run(captured["run_id"])
    assert run_row is not None
    assert run_row.status == RunStatus.FAILED
    assert factory.data_flow.get_token_outcome(captured["token_id"]) is None
    if kind == "I1c":
        artifacts = factory.execution.get_artifacts(captured["run_id"])
        assert any(artifact.artifact_id == captured["artifact_id"] for artifact in artifacts)


def test_sweep_skipped_on_graceful_shutdown(monkeypatch: pytest.MonkeyPatch) -> None:
    from elspeth.core.landscape.data_flow_repository import DataFlowRepository
    from elspeth.core.landscape.database import LandscapeDB

    sweep_calls: list[str] = []

    def _fail_if_swept(self: DataFlowRepository, run_id: str) -> None:
        sweep_calls.append(run_id)
        raise AssertionError("sweep must not run after graceful shutdown")

    def _raise_shutdown_from_flush(self: Orchestrator, factory, run_id, loop_ctx, *args, **kwargs) -> None:
        raise GracefulShutdownError(
            rows_processed=loop_ctx.counters.rows_processed,
            run_id=run_id,
            rows_succeeded=loop_ctx.counters.rows_succeeded,
            rows_failed=loop_ctx.counters.rows_failed,
            rows_quarantined=loop_ctx.counters.rows_quarantined,
            rows_routed_success=loop_ctx.counters.rows_routed_success,
            rows_routed_failure=loop_ctx.counters.rows_routed_failure,
        )

    monkeypatch.setattr(DataFlowRepository, "sweep_deferred_invariants_or_crash", _fail_if_swept)
    monkeypatch.setattr(Orchestrator, "_flush_and_write_sinks", _raise_shutdown_from_flush)

    db = LandscapeDB.in_memory()
    config, graph = _build_minimal_run()

    with pytest.raises(GracefulShutdownError):
        Orchestrator(db).run(config, graph=graph, payload_store=MockPayloadStore())

    assert sweep_calls == []
