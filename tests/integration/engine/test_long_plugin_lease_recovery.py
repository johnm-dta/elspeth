"""Registered-process proof for a plugin call beyond lease and stall bounds.

The first follower blocks inside a real ``BaseTransform`` after claiming one
READY item.  The leader then advances a shared test clock beyond both the item
lease and the hard stall budget, recovers the item, and lets a second follower
re-drive it.  Only after the replacement disposition is durable does the first
plugin return.

Synchronization is event/clock driven: no correctness assertion depends on a
wall-clock sleep.
"""

from __future__ import annotations

import json
import multiprocessing
import queue
import traceback
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Literal

import pytest
from sqlalchemy import select, update

from elspeth.contracts import Determinism, PipelineRow
from elspeth.contracts.plugin_context import PluginContext
from elspeth.contracts.scheduler import SchedulerEventType, TokenWorkStatus
from elspeth.core.config import QueueSettings, SourceSettings, TransformSettings
from elspeth.core.dag import ExecutionGraph
from elspeth.core.dag.wiring import WiredTransform
from elspeth.core.landscape import LandscapeDB
from elspeth.core.landscape.factory import RecorderFactory
from elspeth.core.landscape.schema import (
    run_coordination_events_table,
    run_coordination_table,
    run_workers_table,
    scheduler_events_table,
    token_work_items_table,
)
from elspeth.core.payload_store import FilesystemPayloadStore
from elspeth.engine.clock import MockClock
from elspeth.engine.orchestrator import PipelineConfig
from elspeth.engine.orchestrator.follower import build_follower_processor
from elspeth.plugins.infrastructure.base import BaseTransform
from elspeth.plugins.infrastructure.results import TransformResult
from tests.e2e.recovery.harness import (
    _T0,
    _CrashRowSchema,
    _InterruptibleSource,
    _run_to_interrupted_checkpoint,
)
from tests.e2e.recovery.test_follower_join_and_drain import (
    _join_follower,
    _seat_run_with_live_leader,
    _seed_ready_row,
)
from tests.fixtures.base_classes import as_sink, as_source, as_transform
from tests.fixtures.plugins import CollectSink

_LEASE_SECONDS = 2
_HEARTBEAT_SECONDS = 1
_STALL_BUDGET_SECONDS = 1
_PROCESS_TIMEOUT_SECONDS = 30.0


class _SharedFileClock:
    """Cross-process UTC clock controlled by the parent test."""

    def __init__(self, path: Path) -> None:
        self._path = path

    def now_utc(self) -> datetime:
        return datetime.fromisoformat(self._path.read_text(encoding="utf-8"))

    def monotonic(self) -> float:
        return self.now_utc().timestamp()


class _LeaseStallTransform(BaseTransform):
    """Real transform with an externally visible call/effect trace."""

    name = "passthrough"
    determinism = Determinism.IO_WRITE
    input_schema = _CrashRowSchema
    output_schema = _CrashRowSchema
    on_error = "discard"

    def __init__(
        self,
        *,
        worker_label: str,
        trace_path: Path,
        entered: Any,
        release: Any,
        block: bool,
        late_outcome: Literal["success", "failure"],
    ) -> None:
        super().__init__({"schema": {"mode": "observed"}})
        self.on_success = "output"
        self.input = "inbound"
        self._worker_label = worker_label
        self._trace_path = trace_path
        self._entered = entered
        self._release = release
        self._block = block
        self._late_outcome = late_outcome

    def _trace(self, event: str) -> None:
        with self._trace_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"event": event, "worker": self._worker_label}, sort_keys=True) + "\n")
            handle.flush()

    def process(self, row: PipelineRow, ctx: Any) -> TransformResult:
        self._trace("call_started")
        self._entered.set()
        if self._block and not self._release.wait(_PROCESS_TIMEOUT_SECONDS):
            raise TimeoutError(f"test release gate timed out for {self._worker_label}")
        # This file append deliberately models an external plugin effect that
        # the scheduler cannot transact with its disposition CAS.
        self._trace("external_effect")
        if self._late_outcome == "failure":
            raise RuntimeError(f"late plugin failure from {self._worker_label}")
        return TransformResult.success(row, success_reason={"action": "lease-stall-proof"})


def _build_worker_pipeline(
    *,
    worker_label: str,
    trace_path: Path,
    entered: Any,
    release: Any,
    block: bool,
    late_outcome: Literal["success", "failure"],
) -> tuple[PipelineConfig, ExecutionGraph, _LeaseStallTransform, CollectSink]:
    source = _InterruptibleSource([], on_success="inbound")
    transform = _LeaseStallTransform(
        worker_label=worker_label,
        trace_path=trace_path,
        entered=entered,
        release=release,
        block=block,
        late_outcome=late_outcome,
    )
    sink = CollectSink("output")
    sources = {"primary": as_source(source)}
    wrapped_transform = as_transform(transform)
    graph = ExecutionGraph.from_plugin_instances(
        sources=sources,
        source_settings_map={"primary": SourceSettings(plugin=source.name, on_success="inbound", options={})},
        transforms=[
            WiredTransform(
                plugin=wrapped_transform,
                settings=TransformSettings(
                    name="passthrough_0",
                    plugin=transform.name,
                    input="inbound",
                    on_success="output",
                    on_error="discard",
                    options={},
                ),
            )
        ],
        sinks={"output": as_sink(sink)},
        queues={"inbound": QueueSettings(description="crash-resume fan-in")},
    )
    config = PipelineConfig(
        sources=sources,
        transforms=[wrapped_transform],
        sinks={"output": as_sink(sink)},
    )
    return config, graph, transform, sink


def _run_follower_attempt(
    *,
    db_url: str,
    payload_path: str,
    run_id: str,
    worker_id: str,
    worker_label: str,
    clock_path: str,
    trace_path: str,
    entered: Any,
    release: Any,
    block: bool,
    late_outcome: Literal["success", "failure"],
    result_queue: Any,
) -> None:
    """Spawn target: run one production follower drain over its own DB handle."""

    db = LandscapeDB(db_url)
    payload_store = FilesystemPayloadStore(Path(payload_path))
    transform: _LeaseStallTransform | None = None
    sink: CollectSink | None = None
    try:
        factory = RecorderFactory(db, payload_store=payload_store)
        config, graph, transform, sink = _build_worker_pipeline(
            worker_label=worker_label,
            trace_path=Path(trace_path),
            entered=entered,
            release=release,
            block=block,
            late_outcome=late_outcome,
        )
        clock = _SharedFileClock(Path(clock_path))
        follower = build_follower_processor(
            factory=factory,
            run_id=run_id,
            worker_id=worker_id,
            graph=graph,
            config=config,
            payload_store=payload_store,
            clock=clock,
            scheduler_lease_seconds=_LEASE_SECONDS,
            scheduler_heartbeat_seconds=_HEARTBEAT_SECONDS,
        )
        ctx = PluginContext(
            run_id=run_id,
            config={},
            landscape=factory.plugin_audit_writer(),
            payload_store=payload_store,
        )
        transform.on_start(ctx)
        sink.on_start(ctx)
        # build_follower_processor deliberately wraps the RowProcessor in the
        # outer idle/heartbeat loop; this process drives one bounded production
        # drain pass so the parent owns release and termination.
        drained = follower._processor.drain_follower_ready_work(ctx)
        result_queue.put({"drained": len(drained), "ok": True})
    except BaseException as exc:  # child must return the exact loser failure image
        result_queue.put(
            {
                "exception": str(exc),
                "exception_type": type(exc).__name__,
                "ok": False,
                "traceback": traceback.format_exc(),
            }
        )
    finally:
        if transform is not None:
            transform.close()
        if sink is not None:
            sink.close()
        db.close()


@dataclass(frozen=True)
class _ScenarioResult:
    old_worker: dict[str, Any]
    replacement_worker: dict[str, Any]
    old_worker_id: str
    replacement_worker_id: str
    leader_epoch: int
    token_id: str
    original_work_item_id: str
    final_item: dict[str, Any]
    scheduler_events: tuple[dict[str, Any], ...]
    coordination_events: tuple[dict[str, Any], ...]
    plugin_trace: tuple[dict[str, Any], ...]


def _queue_result(result_queue: Any) -> dict[str, Any]:
    try:
        result = result_queue.get(timeout=_PROCESS_TIMEOUT_SECONDS)
    except queue.Empty as exc:  # pragma: no cover - timeout is a harness failure with explicit diagnostics
        raise AssertionError("worker produced no result artifact") from exc
    assert isinstance(result, dict)
    return result


def _join_process(process: multiprocessing.Process, *, label: str) -> None:
    process.join(_PROCESS_TIMEOUT_SECONDS)
    if process.is_alive():  # pragma: no cover - timeout is a harness failure with explicit diagnostics
        process.terminate()
        process.join()
        raise AssertionError(f"{label} did not terminate after its bounded release")
    assert process.exitcode == 0, f"{label} exited with code {process.exitcode}"


def _run_stall_scenario(tmp_path: Path, *, late_outcome: Literal["success", "failure"]) -> _ScenarioResult:
    clock = MockClock(start=_T0)
    crashed = _run_to_interrupted_checkpoint(tmp_path, clock)
    leader_id = f"worker:{crashed.run_id}:leader-stall-proof"
    leader_token = _seat_run_with_live_leader(crashed, leader_id=leader_id)
    old_worker_id = _join_follower(crashed, leader_token)
    replacement_worker_id = _join_follower(crashed, leader_token)
    token_id, original_work_item_id = _seed_ready_row(crashed, ingest_sequence=100)

    mp = multiprocessing.get_context("spawn")
    old_entered = mp.Event()
    old_release = mp.Event()
    replacement_entered = mp.Event()
    replacement_release = mp.Event()
    old_results = mp.Queue()
    replacement_results = mp.Queue()
    trace_path = tmp_path / f"plugin-effects-{late_outcome}.jsonl"
    clock_path = tmp_path / "shared-clock.txt"
    initial_now = clock.now_utc()
    clock_path.write_text(initial_now.isoformat(), encoding="utf-8")

    # The replacement graph must address the exact durable transform cursor
    # created by the setup run; the plugin implementation may differ, its
    # declared runtime identity/config may not.
    _probe_config, probe_graph, _probe_transform, _probe_sink = _build_worker_pipeline(
        worker_label="probe",
        trace_path=trace_path,
        entered=mp.Event(),
        release=mp.Event(),
        block=False,
        late_outcome="success",
    )
    assert probe_graph.get_transform_id_map() == crashed.graph.get_transform_id_map()

    common = {
        "db_url": str(crashed.db.engine.url),
        "payload_path": str(crashed.payload_store.base_path),
        "run_id": crashed.run_id,
        "clock_path": str(clock_path),
        "trace_path": str(trace_path),
    }
    old_process = mp.Process(
        target=_run_follower_attempt,
        kwargs={
            **common,
            "worker_id": old_worker_id,
            "worker_label": "old",
            "entered": old_entered,
            "release": old_release,
            "block": True,
            "late_outcome": late_outcome,
            "result_queue": old_results,
        },
    )
    replacement_process = mp.Process(
        target=_run_follower_attempt,
        kwargs={
            **common,
            "worker_id": replacement_worker_id,
            "worker_label": "replacement",
            "entered": replacement_entered,
            "release": replacement_release,
            "block": False,
            "late_outcome": "success",
            "result_queue": replacement_results,
        },
    )

    old_process.start()
    try:
        assert old_entered.wait(_PROCESS_TIMEOUT_SECONDS), "old worker never entered the real plugin call"
        with crashed.db.engine.connect() as conn:
            claimed = dict(
                conn.execute(select(token_work_items_table).where(token_work_items_table.c.token_id == token_id)).mappings().one()
            )
        assert claimed["status"] == TokenWorkStatus.LEASED.value
        assert claimed["attempt"] == 1
        assert claimed["lease_owner"] == old_worker_id

        lease_expires_at = claimed["lease_expires_at"].replace(tzinfo=UTC)
        recovery_now = lease_expires_at + timedelta(seconds=_STALL_BUDGET_SECONDS + 1)
        clock_path.write_text(recovery_now.isoformat(), encoding="utf-8")
        far_future = recovery_now + timedelta(hours=1)
        with crashed.db.engine.begin() as conn:
            conn.execute(
                update(run_workers_table)
                .where(run_workers_table.c.run_id == crashed.run_id)
                .where(run_workers_table.c.worker_id.in_([leader_id, old_worker_id, replacement_worker_id]))
                .values(heartbeat_expires_at=far_future.replace(tzinfo=None))
            )
            conn.execute(
                update(run_coordination_table)
                .where(run_coordination_table.c.run_id == crashed.run_id)
                .values(leader_heartbeat_expires_at=far_future.replace(tzinfo=None))
            )

        recovered = crashed.repo.recover_expired_leases(
            now=recovery_now,
            coordination_token=leader_token,
            stall_budget_seconds=_STALL_BUDGET_SECONDS,
        )
        assert recovered == 1, "leader must rotate the registry-live old worker after the hard stall budget"

        replacement_process.start()
        _join_process(replacement_process, label="replacement worker")
        replacement_result = _queue_result(replacement_results)
        assert replacement_result == {"drained": 1, "ok": True}

        old_release.set()
        _join_process(old_process, label="old worker")
        old_result = _queue_result(old_results)
    finally:
        old_release.set()
        if old_process.is_alive():
            old_process.terminate()
            old_process.join()
        if replacement_process.is_alive():
            replacement_process.terminate()
            replacement_process.join()

    with crashed.db.engine.connect() as conn:
        final_item = dict(
            conn.execute(select(token_work_items_table).where(token_work_items_table.c.token_id == token_id)).mappings().one()
        )
        scheduler_events = tuple(
            dict(row)
            for row in conn.execute(
                select(scheduler_events_table)
                .where(scheduler_events_table.c.token_id == token_id)
                .order_by(scheduler_events_table.c.recorded_at, scheduler_events_table.c.event_id)
            ).mappings()
        )
        coordination_events = tuple(
            dict(row)
            for row in conn.execute(
                select(run_coordination_events_table)
                .where(run_coordination_events_table.c.run_id == crashed.run_id)
                .where(run_coordination_events_table.c.event_type == "worker_stalled")
                .order_by(run_coordination_events_table.c.seq)
            ).mappings()
        )
    plugin_trace = tuple(json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines())
    crashed.db.close()
    return _ScenarioResult(
        old_worker=old_result,
        replacement_worker=replacement_result,
        old_worker_id=old_worker_id,
        replacement_worker_id=replacement_worker_id,
        leader_epoch=leader_token.leader_epoch,
        token_id=token_id,
        original_work_item_id=original_work_item_id,
        final_item=final_item,
        scheduler_events=scheduler_events,
        coordination_events=coordination_events,
        plugin_trace=plugin_trace,
    )


def _assert_recovered_loser_contract(result: _ScenarioResult) -> None:
    """Assert the exact durable winner/loser image shared by both outcomes."""

    assert result.old_worker == {"drained": 0, "ok": True}
    assert result.replacement_worker == {"drained": 1, "ok": True}
    assert result.final_item["status"] == TokenWorkStatus.PENDING_SINK.value
    assert result.final_item["attempt"] == 2
    assert result.final_item["lease_owner"] == result.replacement_worker_id
    assert result.final_item["work_item_id"] != result.original_work_item_id

    event_types = [event["event_type"] for event in result.scheduler_events]
    assert event_types.count(SchedulerEventType.RECOVER_EXPIRED_LEASE.value) == 1
    assert event_types.count(SchedulerEventType.LEASE_LOST.value) == 1
    assert event_types.count(SchedulerEventType.MARK_PENDING_SINK.value) == 1
    assert event_types.count(SchedulerEventType.MARK_FAILED.value) == 0
    assert event_types.count(SchedulerEventType.MARK_TERMINAL.value) == 0

    recovery_event = next(
        event for event in result.scheduler_events if event["event_type"] == SchedulerEventType.RECOVER_EXPIRED_LEASE.value
    )
    assert recovery_event["from_lease_owner"] == result.old_worker_id
    assert recovery_event["from_attempt"] == 1
    assert recovery_event["to_attempt"] == 2
    assert json.loads(recovery_event["context_json"])["previous_work_item_id"] == result.original_work_item_id

    lease_lost_event = next(event for event in result.scheduler_events if event["event_type"] == SchedulerEventType.LEASE_LOST.value)
    assert lease_lost_event["caller_owner"] == result.old_worker_id
    assert lease_lost_event["work_item_id"] == result.original_work_item_id
    assert json.loads(lease_lost_event["context_json"])["reason"] == "heartbeat_cas_miss_after_recovery"

    disposition_event = next(
        event for event in result.scheduler_events if event["event_type"] == SchedulerEventType.MARK_PENDING_SINK.value
    )
    assert disposition_event["caller_owner"] == result.replacement_worker_id
    assert disposition_event["from_attempt"] == 2

    assert len(result.coordination_events) == 1
    assert result.coordination_events[0]["worker_id"] == result.old_worker_id
    assert result.coordination_events[0]["leader_epoch"] == result.leader_epoch
    assert json.loads(result.coordination_events[0]["context_json"])["reaped_work_item_id"] == result.original_work_item_id
    assert result.plugin_trace == (
        {"event": "call_started", "worker": "old"},
        {"event": "call_started", "worker": "replacement"},
        {"event": "external_effect", "worker": "replacement"},
        {"event": "external_effect", "worker": "old"},
    )


@pytest.mark.timeout(120)
def test_late_success_observes_lease_loss_before_stale_disposition(tmp_path: Path) -> None:
    result = _run_stall_scenario(tmp_path, late_outcome="success")
    _assert_recovered_loser_contract(result)


@pytest.mark.timeout(120)
def test_late_failure_observes_lease_loss_before_stale_failure_disposition(tmp_path: Path) -> None:
    result = _run_stall_scenario(tmp_path, late_outcome="failure")
    _assert_recovered_loser_contract(result)
