"""Real-process SQLite proofs for artifact logical-effect idempotency."""

from __future__ import annotations

import multiprocessing
from pathlib import Path
from queue import Empty
from typing import Any

from elspeth.contracts.enums import NodeType
from elspeth.contracts.schema import SchemaConfig
from elspeth.core.landscape.database import LandscapeDB
from elspeth.core.landscape.factory import RecorderFactory

_DYNAMIC_SCHEMA = SchemaConfig.from_dict({"mode": "observed"})


def _register_contending_artifact(
    db_url: str,
    ready: Any,
    start: Any,
    results: Any,
    proposed_artifact_id: str,
) -> None:
    """Open a process-local connection and contend on one durable key."""
    db: LandscapeDB | None = None
    try:
        db = LandscapeDB.from_url(db_url, create_tables=False)
        factory = RecorderFactory(db)
        ready.put(proposed_artifact_id)
        if not start.wait(timeout=15):
            raise TimeoutError("artifact contention start gate was not released")
        artifact = factory.execution.register_artifact(
            run_id="run-artifact-contention",
            state_id="state-artifact-contention",
            sink_node_id="sink-artifact-contention",
            artifact_type="csv",
            path="/output/contention.csv",
            content_hash="sha256:contention",
            size_bytes=128,
            artifact_id=proposed_artifact_id,
            idempotency_key="run-artifact-contention:row-artifact-contention:csv_sink",
        )
        results.put(("ok", artifact.artifact_id))
    except BaseException as exc:
        results.put(("error", f"{type(exc).__name__}: {exc}"))
    finally:
        if db is not None:
            db.close()


def _seed_contention_database(db_path: Path) -> str:
    db_url = f"sqlite:///{db_path}"
    db = LandscapeDB.from_url(db_url)
    try:
        factory = RecorderFactory(db)
        run = factory.run_lifecycle.begin_run(
            config={},
            canonical_version="v1",
            run_id="run-artifact-contention",
            openrouter_catalog_sha256="0" * 64,
            openrouter_catalog_source="bundled",
        )
        factory.data_flow.register_node(
            run_id=run.run_id,
            plugin_name="source",
            node_type=NodeType.SOURCE,
            plugin_version="1.0",
            config={},
            node_id="source-artifact-contention",
            schema_config=_DYNAMIC_SCHEMA,
        )
        factory.data_flow.register_node(
            run_id=run.run_id,
            plugin_name="csv_sink",
            node_type=NodeType.SINK,
            plugin_version="1.0",
            config={},
            node_id="sink-artifact-contention",
            schema_config=_DYNAMIC_SCHEMA,
        )
        row = factory.data_flow.create_row(
            run_id=run.run_id,
            source_node_id="source-artifact-contention",
            row_index=0,
            data={"value": 1},
            row_id="row-artifact-contention",
            source_row_index=0,
            ingest_sequence=0,
        )
        token = factory.data_flow.create_token(row.row_id, token_id="token-artifact-contention")
        factory.execution.begin_node_state(
            token_id=token.token_id,
            node_id="sink-artifact-contention",
            run_id=run.run_id,
            step_index=0,
            input_data={"value": 1},
            state_id="state-artifact-contention",
        )
    finally:
        db.close()
    return db_url


def test_two_processes_converge_on_one_artifact_identity(tmp_path: Path) -> None:
    db_url = _seed_contention_database(tmp_path / "artifact-contention.db")
    context = multiprocessing.get_context("spawn")
    ready = context.Queue()
    start = context.Event()
    results = context.Queue()
    processes = [
        context.Process(
            target=_register_contending_artifact,
            args=(db_url, ready, start, results, f"artifact-proposal-{ordinal}"),
        )
        for ordinal in range(2)
    ]

    for process in processes:
        process.start()
    try:
        proposals = {ready.get(timeout=20), ready.get(timeout=20)}
        assert proposals == {"artifact-proposal-0", "artifact-proposal-1"}
        start.set()
        outcomes = [results.get(timeout=20), results.get(timeout=20)]
    except Empty as exc:
        raise AssertionError("artifact contender did not report before timeout") from exc
    finally:
        start.set()
        for process in processes:
            process.join(timeout=20)
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)

    assert all(process.exitcode == 0 for process in processes)
    assert {status for status, _value in outcomes} == {"ok"}
    winning_ids = {artifact_id for _status, artifact_id in outcomes}
    assert len(winning_ids) == 1

    verify_db = LandscapeDB.from_url(db_url, read_only=True, create_tables=False)
    try:
        artifacts = RecorderFactory(verify_db).execution.get_artifacts("run-artifact-contention")
    finally:
        verify_db.close()
    assert len(artifacts) == 1
    assert artifacts[0].artifact_id == winning_ids.pop()
