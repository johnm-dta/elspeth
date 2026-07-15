"""Cross-process call-index collision recovery.

The external effect happens before its Landscape call row is recorded.  Two
independent recorders can therefore allocate the same process-local index and
both complete their effects before either INSERT establishes DB authority.
The audit write must remap the losing allocation without repeating the effect.
"""

from __future__ import annotations

import multiprocessing
import os
from pathlib import Path
from typing import Any, Literal

import pytest

from elspeth.contracts import CallStatus, CallType, NodeType
from elspeth.contracts.call_data import RawCallPayload
from elspeth.contracts.schema import SchemaConfig
from elspeth.core.landscape.database import LandscapeDB
from elspeth.core.landscape.factory import RecorderFactory

_SCHEMA = SchemaConfig.from_dict({"mode": "observed"})


def _record_after_effect(
    db_url: str,
    parent_kind: Literal["state", "operation"],
    parent_id: str,
    worker_id: str,
    effect_path: str,
    ready: Any,
    release: Any,
    results: Any,
) -> None:
    """Allocate, durably perform one effect, then record it from a child."""
    db = LandscapeDB(db_url)
    try:
        factory = RecorderFactory(db)
        if parent_kind == "state":
            proposed_index = factory.execution.allocate_call_index(parent_id)
        else:
            proposed_index = factory.execution.allocate_operation_call_index(parent_id)
        ready.put((worker_id, proposed_index))
        if not release.wait(timeout=30):
            results.put((worker_id, "release-timeout", proposed_index, None))
            return

        # This append+fsync represents the already-completed external effect.
        # It deliberately precedes the audit INSERT and must run exactly once.
        fd = os.open(effect_path, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o600)
        try:
            os.write(fd, f"{worker_id}\n".encode())
            os.fsync(fd)
        finally:
            os.close(fd)

        try:
            if parent_kind == "state":
                call = factory.execution.record_call(
                    parent_id,
                    proposed_index,
                    CallType.HTTP,
                    CallStatus.SUCCESS,
                    request_data=RawCallPayload({"worker": worker_id}),
                    response_data=RawCallPayload({"ok": True}),
                )
            else:
                call = factory.execution.record_operation_call(
                    parent_id,
                    CallType.HTTP,
                    CallStatus.SUCCESS,
                    request_data=RawCallPayload({"worker": worker_id}),
                    response_data=RawCallPayload({"ok": True}),
                    call_index=proposed_index,
                )
        except Exception as exc:  # pragma: no cover - asserted in parent
            results.put((worker_id, type(exc).__name__, proposed_index, None))
        else:
            results.put((worker_id, "ok", proposed_index, call.call_index))
    finally:
        db.close()


def _seed_parent(db_url: str, parent_kind: Literal["state", "operation"]) -> str:
    db = LandscapeDB(db_url)
    try:
        factory = RecorderFactory(db)
        factory.run_lifecycle.begin_run(config={}, canonical_version="v1", run_id="call-race")
        factory.data_flow.register_node(
            run_id="call-race",
            plugin_name="source",
            node_type=NodeType.SOURCE,
            plugin_version="1.0",
            config={},
            node_id="source-0",
            schema_config=_SCHEMA,
        )
        if parent_kind == "operation":
            return factory.execution.begin_operation("call-race", "source-0", "source_load").operation_id

        factory.data_flow.register_node(
            run_id="call-race",
            plugin_name="transform",
            node_type=NodeType.TRANSFORM,
            plugin_version="1.0",
            config={},
            node_id="transform-0",
            schema_config=_SCHEMA,
        )
        row = factory.data_flow.create_row(
            "call-race",
            "source-0",
            0,
            {"value": 1},
            source_row_index=0,
            ingest_sequence=0,
        )
        token = factory.data_flow.create_token(row.row_id)
        return factory.execution.begin_node_state(
            token.token_id,
            "transform-0",
            "call-race",
            0,
            {"value": 1},
        ).state_id
    finally:
        db.close()


@pytest.mark.parametrize("parent_kind", ["state", "operation"])
def test_completed_effects_survive_cross_process_call_index_collision(
    tmp_path: Path,
    parent_kind: Literal["state", "operation"],
) -> None:
    db_url = f"sqlite:///{tmp_path / f'{parent_kind}.db'}"
    parent_id = _seed_parent(db_url, parent_kind)
    effect_path = tmp_path / f"{parent_kind}.effects"
    ctx = multiprocessing.get_context("spawn")
    ready = ctx.Queue()
    release = ctx.Event()
    results = ctx.Queue()
    workers = [
        ctx.Process(
            target=_record_after_effect,
            args=(db_url, parent_kind, parent_id, f"worker-{index}", str(effect_path), ready, release, results),
        )
        for index in range(2)
    ]

    for worker in workers:
        worker.start()
    allocations = [ready.get(timeout=30) for _ in workers]
    assert sorted(index for _worker_id, index in allocations) == [0, 0]
    release.set()

    for worker in workers:
        worker.join(timeout=30)
        assert not worker.is_alive()
        assert worker.exitcode == 0
    outcomes = [results.get(timeout=5) for _ in workers]

    assert sorted(effect_path.read_text().splitlines()) == ["worker-0", "worker-1"]
    assert sorted((status, proposed, recorded) for _worker, status, proposed, recorded in outcomes) == [
        ("ok", 0, 0),
        ("ok", 0, 1),
    ]

    db = LandscapeDB(db_url)
    try:
        factory = RecorderFactory(db)
        calls = factory.query.get_calls(parent_id) if parent_kind == "state" else factory.execution.get_operation_calls(parent_id)
        assert [call.call_index for call in calls] == [0, 1]
    finally:
        db.close()
