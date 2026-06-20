# tests/integration/engine/test_multi_source_chaos.py
"""Multi-source scheduler chaos proofs (filigree elspeth-7bb7124e8f).

Deterministic failure injection against REAL multi-source pipelines — real
Orchestrator.run, real SQLite LandscapeDB, real durable token scheduler —
asserting only through durable, observable surfaces (token_work_items,
scheduler_events, node_states, terminal-outcome journal, rows, run_sources, runs,
terminal RunStatus). Nothing here reads checkpoint-blob internals, so these
tests survive the F1 durability unification that deletes the checkpoint-blob
layer.

Chaos instruments:

- ChaosLLM (tests/fixtures/chaosllm.py): in-process Starlette TestClient fake
  LLM API. Determinism note: error-injection percentages are pinned to 100.0
  or 0.0 per call window, so every injection decision is forced — no RNG
  outcome dependence, no real network socket, default-CI safe.
- Injected MockClock (the test_rc6_lease_recovery_sweep.py pattern) to force
  lease expiry mid-transform without wall-clock sleeps.
- A peer-owner ``recover_expired_leases`` sweep issued from INSIDE a
  transform, simulating a concurrent recovery worker racing the engine.

Scenarios:

1. Two sources + a ChaosLLM-backed transform failing a deterministic subset
   of calls (every refunds row's first attempt): engine retries are auditable
   per-attempt in node_states; the scheduler journal's attempt column is
   UNTOUCHED by sub-lease retries; ingest_sequence is preserved; no orphaned
   READY rows remain.
2. Lease expiry forced mid-transform + a peer sweep: reclaim bumps the
   durable attempt and rotates the work item; the original holder is fenced
   (its heartbeat CAS-misses, durably journaled as ``lease_lost``) and the
   run REFUSES completion rather than silently dropping the reclaimed token.
3. Asymmetric sources (100 rows vs 1 row) where the external-call outage hits
   ONLY the small source: big source unaffected, per-source provenance in the
   journal proves the isolation, terminal status is
   COMPLETED_WITH_FAILURES.
4. Source error mid-stream alongside a surviving source — both the
   plugin-exception flavor (whole-run abort, current semantics) and the
   data-error flavor (quarantine; Wave-1 unmasked
   COMPLETED_WITH_FAILURES, commit f487d7b13).
"""

from __future__ import annotations

import inspect
import json
from pathlib import Path
from typing import Any, ClassVar

import pytest
from sqlalchemy import and_, literal_column, select

from elspeth.cli_helpers import instantiate_plugins_from_config
from elspeth.contracts import Determinism, PluginSchema, RunStatus
from elspeth.contracts.errors import OrchestrationInvariantError
from elspeth.contracts.scheduler import SchedulerEventType, TokenWorkStatus
from elspeth.core.config import (
    ElspethSettings,
    QueueSettings,
    SourceSettings,
    TransformSettings,
    load_settings_from_yaml_string,
)
from elspeth.core.dag import ExecutionGraph
from elspeth.core.dag.models import WiredTransform
from elspeth.core.landscape import LandscapeDB
from elspeth.core.landscape.scheduler_repository import TokenSchedulerRepository
from elspeth.core.landscape.schema import (
    node_states_table,
    nodes_table,
    rows_table,
    run_sources_table,
    runs_table,
    scheduler_events_table,
    token_outcomes_table,
    token_work_items_table,
    tokens_table,
)
from elspeth.core.payload_store import FilesystemPayloadStore
from elspeth.engine.clock import MockClock
from elspeth.engine.orchestrator import Orchestrator, PipelineConfig
from elspeth.engine.orchestrator.preflight import assemble_and_validate_pipeline_config
from elspeth.engine.processor import RowProcessor
from elspeth.plugins.infrastructure.base import BaseTransform
from elspeth.plugins.infrastructure.results import TransformResult
from tests.fixtures.base_classes import _TestSchema, _TestSourceBase, as_sink, as_source, as_transform
from tests.fixtures.chaosllm import ChaosLLMFixture
from tests.fixtures.plugins import CollectSink, ListSource

# A deliberately non-real epoch anchor for the injected clock; nothing here
# depends on wall time.
_CLOCK_EPOCH = 1_750_000_000.0

# The production scheduler lease duration, read from RowProcessor's own
# signature rather than hardcoded: the orchestrator constructs RowProcessor
# without overriding scheduler_lease_seconds, so the constructor default IS
# the lease the engine runs with. Deriving it here means a future change to
# the default (e.g. by the F1 durability rewrite) moves the lease-buster's
# clock advance with it instead of silently un-expiring the lease and failing
# far from the cause.
_SCHEDULER_LEASE_SECONDS = inspect.signature(RowProcessor.__init__).parameters["scheduler_lease_seconds"].default
assert isinstance(_SCHEDULER_LEASE_SECONDS, int), (
    f"RowProcessor.scheduler_lease_seconds default is no longer a plain int "
    f"({_SCHEDULER_LEASE_SECONDS!r}); update the lease-expiry chaos test to "
    f"derive the lease duration from wherever it now lives."
)
# Margin past expiry; comfortably larger than clock-granularity effects but
# still far smaller than the lease itself.
_LEASE_EXPIRY_ADVANCE = float(_SCHEDULER_LEASE_SECONDS) + 100.0


# ---------------------------------------------------------------------------
# Chaos plugins (test-local, deterministic)
# ---------------------------------------------------------------------------


class _ChaosLLMFlakyTransform(BaseTransform):
    """Real-HTTP transform whose marked rows fail their FIRST attempt.

    Every process() call issues a real chat-completion request to the
    in-process ChaosLLM server. For rows whose ``src`` matches
    ``fail_marker`` the FIRST attempt pins the server to 100% internal_error
    injection (deterministic — every request fails), receives the injected
    HTTP 500, and raises ``ConnectionError`` (an engine-retryable class).
    All other calls pin injection to 0% (deterministic success). The set of
    failing CALLS is therefore an exact, deterministic subset: one per
    marked row.
    """

    name = "chaosllm_flaky_transform"
    determinism = Determinism.DETERMINISTIC
    input_schema: ClassVar[type[PluginSchema]] = _TestSchema
    output_schema: ClassVar[type[PluginSchema]] = _TestSchema

    def __init__(
        self,
        chaos: ChaosLLMFixture,
        *,
        fail_marker: str,
        input_connection: str,
        on_success: str,
    ) -> None:
        super().__init__({"schema": {"mode": "observed"}})
        self.input = input_connection
        self.on_success = on_success
        self.on_error = "discard"
        self._chaos = chaos
        self._fail_marker = fail_marker
        self._failed_once: set[tuple[str, int]] = set()

    def process(self, row: Any, ctx: Any) -> TransformResult:
        key = (str(row["src"]), int(row["value"]))
        should_fail = row["src"] == self._fail_marker and key not in self._failed_once
        self._chaos.update_config(internal_error_pct=100.0 if should_fail else 0.0)
        response = self._chaos.post_completion(messages=[{"role": "user", "content": f"rate {key}"}])
        if should_fail:
            self._failed_once.add(key)
            if response.status_code != 500:
                raise AssertionError(f"ChaosLLM at 100% internal_error returned HTTP {response.status_code}, expected 500")
            # ConnectionError is in the engine's retryable class set
            # (processor._execute_transform_with_retry is_retryable).
            raise ConnectionError(f"chaosllm injected HTTP 500 for {key}")
        if response.status_code != 200:
            raise AssertionError(f"ChaosLLM at 0% injection returned HTTP {response.status_code}, expected 200")
        return TransformResult.success(row, success_reason={"action": "chaosllm_call"})


class _ChaosLLMOutageTransform(BaseTransform):
    """Real-HTTP transform whose marked rows hit a permanent ChaosLLM outage.

    Marked rows call ChaosLLM pinned to 100% gateway_timeout (deterministic
    HTTP 504) and return ``TransformResult.error`` — a legitimate processing
    failure routed via on_error, NOT an exception, so the engine does not
    retry it. Unmarked rows call at 0% injection and succeed.
    """

    name = "chaosllm_outage_transform"
    determinism = Determinism.DETERMINISTIC
    input_schema: ClassVar[type[PluginSchema]] = _TestSchema
    output_schema: ClassVar[type[PluginSchema]] = _TestSchema

    def __init__(
        self,
        chaos: ChaosLLMFixture,
        *,
        doom_marker: str,
        input_connection: str,
        on_success: str,
        on_error: str,
    ) -> None:
        super().__init__({"schema": {"mode": "observed"}})
        self.input = input_connection
        self.on_success = on_success
        self.on_error = on_error
        self._chaos = chaos
        self._doom_marker = doom_marker

    def process(self, row: Any, ctx: Any) -> TransformResult:
        doomed = row["src"] == self._doom_marker
        self._chaos.update_config(gateway_timeout_pct=100.0 if doomed else 0.0)
        response = self._chaos.post_completion(messages=[{"role": "user", "content": f"rate {row['src']}:{row['value']}"}])
        if doomed:
            if response.status_code != 504:
                raise AssertionError(f"ChaosLLM at 100% gateway_timeout returned HTTP {response.status_code}, expected 504")
            return TransformResult.error(
                {"reason": "api_call_failed", "error": f"chaosllm injected HTTP {response.status_code}"},
                retryable=False,
            )
        if response.status_code != 200:
            raise AssertionError(f"ChaosLLM at 0% injection returned HTTP {response.status_code}, expected 200")
        return TransformResult.success(row, success_reason={"action": "chaosllm_call"})


class _LeaseBusterTransform(BaseTransform):
    """Transform that forces its own lease past expiry mid-process.

    On the marked row's first arrival it advances the injected MockClock past
    ``scheduler_lease_seconds`` (the RowProcessor constructor default, read
    via ``_SCHEDULER_LEASE_SECONDS``) and then runs a PEER-owner
    ``recover_expired_leases`` sweep against the same audit DB — simulating a
    concurrent recovery worker reaping an expired lease while the original
    worker is still mid-flight.
    """

    name = "lease_buster_transform"
    determinism = Determinism.DETERMINISTIC
    input_schema: ClassVar[type[PluginSchema]] = _TestSchema
    output_schema: ClassVar[type[PluginSchema]] = _TestSchema

    def __init__(
        self,
        clock: MockClock,
        db: LandscapeDB,
        *,
        peer_owner: str,
        bust_key: tuple[str, int],
        input_connection: str,
        on_success: str,
    ) -> None:
        super().__init__({"schema": {"mode": "observed"}})
        self.input = input_connection
        self.on_success = on_success
        self.on_error = "discard"
        self._clock = clock
        self._db = db
        self._peer_owner = peer_owner
        self._bust_key = bust_key
        self.peer_recovered_count: int | None = None
        self._fired = False

    def process(self, row: Any, ctx: Any) -> TransformResult:
        if not self._fired and (row["src"], row["value"]) == self._bust_key:
            self._fired = True
            # Blow past the production-default scheduler lease (and a
            # fortiori the shorter heartbeat interval — the RowProcessor
            # constructor enforces heartbeat < lease) without sleeping.
            self._clock.advance(_LEASE_EXPIRY_ADVANCE)
            peer_repo = TokenSchedulerRepository(self._db.engine)
            self.peer_recovered_count = peer_repo.recover_expired_leases(
                run_id=ctx.run_id,
                now=self._clock.now_utc(),
                caller_owner=self._peer_owner,
            )
        return TransformResult.success(row, success_reason={"action": "lease_buster"})


class _PassthroughTransform(BaseTransform):
    """Plain passthrough used to guarantee a node-iteration heartbeat boundary."""

    name = "chaos_passthrough_transform"
    determinism = Determinism.DETERMINISTIC
    input_schema: ClassVar[type[PluginSchema]] = _TestSchema
    output_schema: ClassVar[type[PluginSchema]] = _TestSchema

    def __init__(self, *, input_connection: str, on_success: str) -> None:
        super().__init__({"schema": {"mode": "observed"}})
        self.input = input_connection
        self.on_success = on_success
        self.on_error = "discard"

    def process(self, row: Any, ctx: Any) -> TransformResult:
        return TransformResult.success(row, success_reason={"action": "passthrough"})


class _ExplodingSource(_TestSourceBase):
    """Source that yields ``explode_after`` valid rows, then raises mid-stream."""

    name = "exploding_source"
    output_schema = ListSource.output_schema

    def __init__(self, rows: list[dict[str, Any]], *, explode_after: int, on_success: str) -> None:
        super().__init__()
        self._rows = rows
        self._explode_after = explode_after
        self.on_success = on_success

    def load(self, ctx: Any) -> Any:
        for yielded, source_row in enumerate(self.wrap_rows(self._rows)):
            if yielded >= self._explode_after:
                raise RuntimeError("injected mid-stream source failure")
            yield source_row


# ---------------------------------------------------------------------------
# Pipeline assembly + durable-surface query helpers
# ---------------------------------------------------------------------------


def _build_two_source_pipeline(
    sources: dict[str, Any],
    transforms: list[BaseTransform],
    sinks: dict[str, CollectSink],
    *,
    queue: str = "inbound",
) -> tuple[PipelineConfig, ExecutionGraph]:
    """Wire test sources through chained transforms into CollectSinks."""
    typed_sources = {name: as_source(plugin) for name, plugin in sources.items()}
    source_settings = {name: SourceSettings(plugin=plugin.name, on_success=queue) for name, plugin in sources.items()}
    wired = [
        WiredTransform(
            plugin=as_transform(transform),
            settings=TransformSettings(
                name=f"{transform.name}_{index}",
                plugin=transform.name,
                input=transform.input,
                on_success=transform.on_success,
                on_error=transform.on_error or "discard",
                options={},
            ),
        )
        for index, transform in enumerate(transforms)
    ]
    typed_sinks = {name: as_sink(sink) for name, sink in sinks.items()}
    graph = ExecutionGraph.from_plugin_instances(
        sources=typed_sources,
        source_settings_map=source_settings,
        transforms=wired,
        sinks=typed_sinks,
        queues={queue: QueueSettings()},
    )
    config = PipelineConfig(
        sources=typed_sources,
        transforms=[as_transform(transform) for transform in transforms],
        sinks=typed_sinks,
    )
    return config, graph


def _fast_retry_settings() -> ElspethSettings:
    """Minimal settings whose ONLY runtime contribution is the retry policy.

    ``Orchestrator.run`` builds its RetryManager exclusively from
    ``settings.retry`` (run_core.build_row_processor); the rest of the run is
    driven by the explicit config/graph arguments. The stub source/sink decls
    exist purely to satisfy settings-model validation and are never
    instantiated. max_delay_seconds caps tenacity's jittered backoff, keeping
    the retried attempts fast and the test deterministic in outcome.
    """
    return load_settings_from_yaml_string(
        """
sources:
  stub:
    plugin: csv
    on_success: output
    options:
      path: /nonexistent/never-instantiated.csv
      on_validation_failure: discard
      schema:
        mode: observed
sinks:
  output:
    plugin: json
    on_write_failure: discard
    options:
      path: /nonexistent/never-instantiated.jsonl
      format: jsonl
      schema:
        mode: observed
retry:
  max_attempts: 3
  initial_delay_seconds: 0.01
  max_delay_seconds: 0.1
"""
    )


def _source_node_ids(db: LandscapeDB, run_id: str) -> dict[str, str]:
    """Map source_name -> source_node_id from the run_sources audit table."""
    with db.connection() as conn:
        rows = conn.execute(
            select(run_sources_table.c.source_name, run_sources_table.c.source_node_id).where(run_sources_table.c.run_id == run_id)
        ).all()
    return {str(row.source_name): str(row.source_node_id) for row in rows}


def _token_source_names(db: LandscapeDB, run_id: str) -> dict[str, str]:
    """Map token_id -> source_name via the durable token->row->source join."""
    with db.connection() as conn:
        rows = conn.execute(
            select(tokens_table.c.token_id, run_sources_table.c.source_name)
            .join(rows_table, and_(rows_table.c.run_id == tokens_table.c.run_id, rows_table.c.row_id == tokens_table.c.row_id))
            .join(
                run_sources_table,
                and_(
                    run_sources_table.c.run_id == rows_table.c.run_id,
                    run_sources_table.c.source_node_id == rows_table.c.source_node_id,
                ),
            )
            .where(tokens_table.c.run_id == run_id)
        ).all()
    return {str(row.token_id): str(row.source_name) for row in rows}


def _work_items(db: LandscapeDB, run_id: str) -> list[dict[str, Any]]:
    with db.connection() as conn:
        return [
            dict(row)
            for row in conn.execute(
                select(
                    token_work_items_table.c.token_id,
                    token_work_items_table.c.work_item_id,
                    token_work_items_table.c.status,
                    token_work_items_table.c.attempt,
                    token_work_items_table.c.lease_owner,
                    token_work_items_table.c.lease_expires_at,
                    token_work_items_table.c.ingest_sequence,
                )
                .where(token_work_items_table.c.run_id == run_id)
                .order_by(token_work_items_table.c.ingest_sequence)
            ).mappings()
        ]


def _scheduler_events(db: LandscapeDB, run_id: str) -> list[dict[str, Any]]:
    """All scheduler events in durable insertion (rowid) order."""
    with db.connection() as conn:
        return [
            dict(row)
            for row in conn.execute(
                select(
                    literal_column("scheduler_events.rowid").label("seq"),
                    scheduler_events_table.c.token_id,
                    scheduler_events_table.c.event_type,
                    scheduler_events_table.c.from_status,
                    scheduler_events_table.c.to_status,
                    scheduler_events_table.c.from_lease_owner,
                    scheduler_events_table.c.to_lease_owner,
                    scheduler_events_table.c.from_attempt,
                    scheduler_events_table.c.to_attempt,
                    scheduler_events_table.c.caller_owner,
                    scheduler_events_table.c.context_json,
                )
                .where(scheduler_events_table.c.run_id == run_id)
                .order_by(literal_column("scheduler_events.rowid"))
            ).mappings()
        ]


def _outcomes_by_source(db: LandscapeDB, run_id: str) -> list[tuple[str, str, str, str | None]]:
    """(source_name, outcome, path, sink_name) for every token outcome."""
    stmt = (
        select(
            run_sources_table.c.source_name,
            token_outcomes_table.c.outcome,
            token_outcomes_table.c.path,
            token_outcomes_table.c.sink_name,
        )
        .join(
            tokens_table,
            and_(
                tokens_table.c.run_id == token_outcomes_table.c.run_id,
                tokens_table.c.token_id == token_outcomes_table.c.token_id,
            ),
        )
        .join(rows_table, and_(rows_table.c.run_id == tokens_table.c.run_id, rows_table.c.row_id == tokens_table.c.row_id))
        .join(
            run_sources_table,
            and_(
                run_sources_table.c.run_id == rows_table.c.run_id,
                run_sources_table.c.source_node_id == rows_table.c.source_node_id,
            ),
        )
        .where(token_outcomes_table.c.run_id == run_id)
        .order_by(run_sources_table.c.source_name, token_outcomes_table.c.path)
    )
    with db.connection() as conn:
        return [(row.source_name, row.outcome, row.path, row.sink_name) for row in conn.execute(stmt)]


def _node_id_for_plugin(db: LandscapeDB, run_id: str, plugin_name: str) -> str:
    with db.connection() as conn:
        return str(
            conn.execute(
                select(nodes_table.c.node_id).where(nodes_table.c.run_id == run_id).where(nodes_table.c.plugin_name == plugin_name)
            ).scalar_one()
        )


def _node_state_attempts(db: LandscapeDB, run_id: str, node_id: str) -> dict[str, dict[int, str]]:
    """Map token_id -> {attempt: status} for one node's node_states."""
    with db.connection() as conn:
        rows = conn.execute(
            select(node_states_table.c.token_id, node_states_table.c.attempt, node_states_table.c.status)
            .where(node_states_table.c.run_id == run_id)
            .where(node_states_table.c.node_id == node_id)
        ).all()
    attempts: dict[str, dict[int, str]] = {}
    for row in rows:
        attempts.setdefault(str(row.token_id), {})[int(row.attempt)] = str(row.status)
    return attempts


def _single_run_id(db: LandscapeDB) -> str:
    with db.connection() as conn:
        return str(conn.execute(select(runs_table.c.run_id)).scalar_one())


def _run_status(db: LandscapeDB, run_id: str) -> str:
    with db.connection() as conn:
        return str(conn.execute(select(runs_table.c.status).where(runs_table.c.run_id == run_id)).scalar_one())


# ---------------------------------------------------------------------------
# 1. Deterministic ChaosLLM failures + engine retries across two sources
# ---------------------------------------------------------------------------


@pytest.mark.timeout(60)
def test_chaosllm_first_attempt_failures_retry_with_durable_per_attempt_audit(
    chaosllm_server: ChaosLLMFixture,
    tmp_path: Path,
) -> None:
    """Two sources + ChaosLLM transform failing a deterministic call subset.

    The transform makes one real HTTP call per attempt against the in-process
    ChaosLLM server. Every refunds row's FIRST attempt is forced to fail
    (100% internal_error -> HTTP 500 -> retryable ConnectionError); the
    engine's RetryManager retries and the second attempt is forced to succeed
    (0% injection). Orders rows never fail. The failing-call subset is exact:
    one call per refunds row.

    Durable invariants pinned:
    - retries are auditable PER ATTEMPT in node_states: refunds tokens carry
      {attempt 0: failed, attempt 1: completed} at the transform node under
      the UNIQUE (token_id, node_id, attempt) audit identity; orders tokens
      carry only {attempt 0: completed};
    - sub-lease engine retries do NOT touch the scheduler journal's attempt
      column (attempt stays 1 for every work item — the journal attempt moves
      only on lease recovery);
    - ingest_sequence is preserved across retries (work items still carry the
      original gapless cross-source ordering 0..4);
    - no orphaned READY rows at end of run: every work item is TERMINAL;
    - the run completes COMPLETED and the ChaosLLM request ledger shows
      exactly rows + retried-rows calls (5 + 2 = 7) — the deterministic
      failure subset, no more, no fewer.
    """
    orders = ListSource([{"src": "orders", "value": i} for i in range(3)], name="orders_source", on_success="inbound")
    refunds = ListSource([{"src": "refunds", "value": i} for i in range(2)], name="refunds_source", on_success="inbound")
    flaky = _ChaosLLMFlakyTransform(chaosllm_server, fail_marker="refunds", input_connection="inbound", on_success="output")
    sink = CollectSink("output")
    config, graph = _build_two_source_pipeline({"orders": orders, "refunds": refunds}, [flaky], {"output": sink})

    db = LandscapeDB(f"sqlite:///{tmp_path / 'audit.db'}")
    result = Orchestrator(db).run(
        config,
        graph=graph,
        settings=_fast_retry_settings(),
        payload_store=FilesystemPayloadStore(tmp_path / "payloads"),
    )

    assert result.status == RunStatus.COMPLETED
    assert result.rows_processed == 5
    assert result.rows_succeeded == 5
    assert len(sink.results) == 5

    # Exactly one extra HTTP call per deterministically-failed row.
    assert chaosllm_server.get_stats()["total_requests"] == 7

    run_id = result.run_id
    token_sources = _token_source_names(db, run_id)
    transform_node = _node_id_for_plugin(db, run_id, flaky.name)
    attempts = _node_state_attempts(db, run_id, transform_node)

    refund_tokens = {token_id for token_id, name in token_sources.items() if name == "refunds"}
    order_tokens = {token_id for token_id, name in token_sources.items() if name == "orders"}
    assert len(refund_tokens) == 2 and len(order_tokens) == 3

    # Per-attempt audit: failed first attempt + completed retry, per refunds token.
    for token_id in refund_tokens:
        assert attempts[token_id] == {0: "failed", 1: "completed"}
    for token_id in order_tokens:
        assert attempts[token_id] == {0: "completed"}

    # Scheduler journal: retries are sub-lease — attempt stays 1, ordering
    # primitive preserved, everything TERMINAL (no orphaned READY rows).
    items = _work_items(db, run_id)
    assert len(items) == 5
    assert [item["ingest_sequence"] for item in items] == [0, 1, 2, 3, 4]
    assert all(item["attempt"] == 1 for item in items)
    assert all(item["status"] == TokenWorkStatus.TERMINAL.value for item in items)

    # Every token reached a clean terminal outcome attributed to its source.
    outcomes = _outcomes_by_source(db, run_id)
    assert sorted(outcomes) == sorted(
        [("orders", "success", "default_flow", "output")] * 3 + [("refunds", "success", "default_flow", "output")] * 2
    )


# ---------------------------------------------------------------------------
# 2. Lease expiry forced mid-transform; peer sweep reclaims; stale owner fenced
# ---------------------------------------------------------------------------


@pytest.mark.timeout(60)
def test_lease_expiry_mid_transform_peer_reclaim_bumps_attempt_and_fences_stale_owner(tmp_path: Path) -> None:
    """Mid-transform lease expiry + peer recovery sweep, on the real engine path.

    A transform advances the injected MockClock past the production-default
    scheduler lease (``_SCHEDULER_LEASE_SECONDS``, read from RowProcessor's
    signature so a default change moves this test with it) while processing
    orders row #1, then runs a PEER-owner
    ``recover_expired_leases`` sweep — exactly what a concurrent recovery
    worker does. Current-model semantics pinned through durable surfaces:

    - the peer sweep reclaims exactly one item: the in-flight lease — durable
      ``recover_expired_lease`` event with attempt bump 1 -> 2, lease_owner
      cleared, work_item_id rotated, status back to READY;
    - the ORIGINAL holder is fenced: its next heartbeat (node-iteration
      boundary before the passthrough node) CAS-misses against the rotated
      work item and the engine abandons the in-flight result — durably
      journaled as a ``lease_lost`` event attributed to the stale owner, and
      no mark_* transition for the busted token is ever written by it;
    - the engine keeps pumping the remaining source (refunds rows are claimed
      and parked PENDING_SINK after the fence), proving the fence is
      token-scoped, not run-scoped;
    - the run REFUSES completion: the reclaimed READY continuation is never
      re-claimed in-run (per-row drains drive only their own token, and the
      G1 self-steal guard forbids the run's own maintenance sweep from
      recovering peer-reclaimed work back), so the post-source invariant
      check fails the run rather than silently dropping the token. The
      FAILED run + READY-attempt-2 journal row is precisely the durable
      state a resume sweep recovers from.
    """
    clock = MockClock(start=_CLOCK_EPOCH)
    db = LandscapeDB(f"sqlite:///{tmp_path / 'audit.db'}")
    peer_owner = "chaos-peer-sweeper"

    orders = ListSource([{"src": "orders", "value": i} for i in range(2)], name="orders_source", on_success="inbound")
    refunds = ListSource([{"src": "refunds", "value": i} for i in range(2)], name="refunds_source", on_success="inbound")
    buster = _LeaseBusterTransform(
        clock,
        db,
        peer_owner=peer_owner,
        bust_key=("orders", 1),
        input_connection="inbound",
        on_success="mid",
    )
    passthrough = _PassthroughTransform(input_connection="mid", on_success="output")
    sink = CollectSink("output")
    config, graph = _build_two_source_pipeline({"orders": orders, "refunds": refunds}, [buster, passthrough], {"output": sink})

    with pytest.raises(OrchestrationInvariantError, match="non-terminal scheduler work"):
        Orchestrator(db, clock=clock).run(
            config,
            graph=graph,
            payload_store=FilesystemPayloadStore(tmp_path / "payloads"),
        )

    run_id = _single_run_id(db)
    assert _run_status(db, run_id) == RunStatus.FAILED.value
    assert buster.peer_recovered_count == 1

    token_sources = _token_source_names(db, run_id)
    items = _work_items(db, run_id)
    assert len(items) == 4

    # The busted token: reclaimed READY, attempt bumped, lease cleared.
    busted = [item for item in items if item["status"] == TokenWorkStatus.READY.value]
    assert len(busted) == 1
    busted_item = busted[0]
    assert token_sources[busted_item["token_id"]] == "orders"
    assert busted_item["ingest_sequence"] == 1
    assert busted_item["attempt"] == 2
    assert busted_item["lease_owner"] is None
    assert busted_item["lease_expires_at"] is None

    # Every other token (orders row 0 + both refunds rows, claimed AFTER the
    # fence) is durably parked PENDING_SINK awaiting sink delivery; nothing
    # is LEASED or FAILED. The sink itself never ran (the run refused
    # completion before sink writes), so no result was emitted twice or at all.
    others = [item for item in items if item["token_id"] != busted_item["token_id"]]
    assert all(item["status"] == TokenWorkStatus.PENDING_SINK.value for item in others)
    assert sorted(token_sources[item["token_id"]] for item in others) == ["orders", "refunds", "refunds"]
    assert sink.results == []

    # Durable event journal: exactly one peer reclaim (attempt 1 -> 2) and
    # exactly one stale-owner fence (lease_lost), in that order.
    events = _scheduler_events(db, run_id)
    recoveries = [event for event in events if event["event_type"] == SchedulerEventType.RECOVER_EXPIRED_LEASE.value]
    fences = [event for event in events if event["event_type"] == SchedulerEventType.LEASE_LOST.value]
    assert len(recoveries) == 1
    assert len(fences) == 1
    recovery, fence = recoveries[0], fences[0]

    assert recovery["token_id"] == busted_item["token_id"]
    assert recovery["caller_owner"] == peer_owner
    assert recovery["from_attempt"] == 1 and recovery["to_attempt"] == 2
    assert recovery["to_status"] == TokenWorkStatus.READY.value
    stale_owner = recovery["from_lease_owner"]
    # Epoch 21 (ADR-030 §A.1): the engine threads the registered worker
    # identity (worker:{run_id}:{uuid}) into RowProcessor as the scheduler
    # lease_owner; row-processor:{run_id}:{uuid} remains only as the fallback
    # mint for direct repository-level construction.
    assert isinstance(stale_owner, str) and stale_owner.startswith("worker:")

    assert fence["token_id"] == busted_item["token_id"]
    assert fence["caller_owner"] == stale_owner
    assert fence["from_lease_owner"] == stale_owner
    assert recovery["seq"] < fence["seq"]

    # Fencing is complete: the stale owner never wrote any post-reclaim
    # transition for the busted token — no mark_* event exists for it.
    busted_marks = [event for event in events if event["token_id"] == busted_item["token_id"] and event["event_type"].startswith("mark_")]
    assert busted_marks == []

    # No token reached a terminal outcome: the fence abandoned the in-flight
    # result without fabricating completion, and the parked PENDING_SINK
    # tokens were refused sink delivery when the run failed its invariant.
    assert _outcomes_by_source(db, run_id) == []


# ---------------------------------------------------------------------------
# 3. Asymmetric sources: outage hits only the 1-row source
# ---------------------------------------------------------------------------


@pytest.mark.timeout(120)
def test_asymmetric_sources_outage_on_small_source_is_isolated(
    chaosllm_server: ChaosLLMFixture,
    tmp_path: Path,
) -> None:
    """100-row source + 1-row source; the external outage hits ONLY the small one.

    Every refunds (1-row source) call is forced to a deterministic ChaosLLM
    HTTP 504 and the transform routes the row via on_error to the errors
    sink. All 100 orders calls are forced clean. Durable invariants pinned:

    - source isolation in the audit DB: per-source token outcomes show
      100x (orders, success, default_flow, output) and exactly
      1x (refunds, failure, on_error_routed, errors) — the failing source
      never contaminates the big source's outcomes;
    - per-source provenance primitives hold under failure: source_row_index
      restarts per source while ingest_sequence stays globally gapless with
      the 1-row source last (declaration order);
    - the journal fully resolves: all 101 work items TERMINAL, attempt 1
      (an on_error route is a clean terminal, not a scheduler retry);
    - terminal status is COMPLETED_WITH_FAILURES (Wave-1 semantics,
      f487d7b13: a run with real per-row failures must say so).
    """
    orders = ListSource([{"src": "orders", "value": i} for i in range(100)], name="orders_source", on_success="inbound")
    refunds = ListSource([{"src": "refunds", "value": 0}], name="refunds_source", on_success="inbound")
    outage = _ChaosLLMOutageTransform(
        chaosllm_server,
        doom_marker="refunds",
        input_connection="inbound",
        on_success="output",
        on_error="errors",
    )
    output_sink = CollectSink("output")
    errors_sink = CollectSink("errors")
    config, graph = _build_two_source_pipeline(
        {"orders": orders, "refunds": refunds},
        [outage],
        {"output": output_sink, "errors": errors_sink},
    )

    db = LandscapeDB(f"sqlite:///{tmp_path / 'audit.db'}")
    result = Orchestrator(db).run(
        config,
        graph=graph,
        payload_store=FilesystemPayloadStore(tmp_path / "payloads"),
    )

    assert result.status == RunStatus.COMPLETED_WITH_FAILURES
    assert result.rows_processed == 101
    assert result.rows_succeeded == 100
    assert result.rows_failed == 1

    # The big source's rows all arrived; the small source's row was diverted.
    assert len(output_sink.results) == 100
    assert {row["src"] for row in output_sink.results} == {"orders"}
    assert len(errors_sink.results) == 1
    assert errors_sink.results[0]["src"] == "refunds"

    # One real HTTP call per row — the outage subset is exactly one call.
    assert chaosllm_server.get_stats()["total_requests"] == 101

    run_id = result.run_id

    # Source isolation in token outcomes.
    outcomes = _outcomes_by_source(db, run_id)
    assert outcomes.count(("orders", "success", "default_flow", "output")) == 100
    assert outcomes.count(("refunds", "failure", "on_error_routed", "errors")) == 1
    assert len(outcomes) == 101

    # Provenance primitives under failure (ADR-025/026).
    source_ids = _source_node_ids(db, run_id)
    with db.connection() as conn:
        row_records = conn.execute(
            select(rows_table.c.source_node_id, rows_table.c.source_row_index, rows_table.c.ingest_sequence)
            .where(rows_table.c.run_id == run_id)
            .order_by(rows_table.c.ingest_sequence)
        ).all()
    assert [record.ingest_sequence for record in row_records] == list(range(101))
    orders_indices = [record.source_row_index for record in row_records if record.source_node_id == source_ids["orders"]]
    refunds_indices = [record.source_row_index for record in row_records if record.source_node_id == source_ids["refunds"]]
    assert orders_indices == list(range(100))
    assert refunds_indices == [0]
    assert row_records[-1].source_node_id == source_ids["refunds"]

    # Journal fully resolved: failure-routed rows terminalize like successes.
    items = _work_items(db, run_id)
    assert len(items) == 101
    assert all(item["status"] == TokenWorkStatus.TERMINAL.value for item in items)
    assert all(item["attempt"] == 1 for item in items)


# ---------------------------------------------------------------------------
# 4. Source error mid-stream alongside a surviving source
# ---------------------------------------------------------------------------


@pytest.mark.timeout(60)
def test_source_plugin_exception_mid_stream_aborts_run_as_failed(tmp_path: Path) -> None:
    """A source PLUGIN raising mid-stream aborts the whole run (current semantics).

    ADR-025 ingest is sequential: the surviving orders source is declared
    first and fully ingests its rows (they are durably parked PENDING_SINK in
    the scheduler journal); the exploding source then yields one valid row
    and raises. Current engine semantics — pinned here, deliberately distinct
    from the data-level quarantine flavor below: a source plugin exception is
    a RUN-level fault, not a row-level one. The exception propagates out of
    ``Orchestrator.run`` unchanged, the failure ceremony finalizes the run
    FAILED, and NO sink writes occur — even the surviving source's parked
    rows stay undelivered (durably PENDING_SINK, recoverable by resume, never
    silently dropped). The Wave-1 COMPLETED_WITH_FAILURES family
    (f487d7b13) is for runs where rows reached clean/failed terminals; a
    mid-stream source crash leaves in-flight rows non-terminal, so FAILED is
    the correct (and now unmasked-by-cleanup) status.
    """
    orders = ListSource([{"src": "orders", "value": i} for i in range(3)], name="orders_source", on_success="inbound")
    exploding = _ExplodingSource(
        [{"src": "exploding", "value": i} for i in range(3)],
        explode_after=1,
        on_success="inbound",
    )
    passthrough = _PassthroughTransform(input_connection="inbound", on_success="output")
    sink = CollectSink("output")
    config, graph = _build_two_source_pipeline({"orders": orders, "exploding": exploding}, [passthrough], {"output": sink})

    db = LandscapeDB(f"sqlite:///{tmp_path / 'audit.db'}")
    with pytest.raises(RuntimeError, match="injected mid-stream source failure"):
        Orchestrator(db).run(
            config,
            graph=graph,
            payload_store=FilesystemPayloadStore(tmp_path / "payloads"),
        )

    run_id = _single_run_id(db)
    assert _run_status(db, run_id) == RunStatus.FAILED.value
    assert sink.results == []

    # The surviving source finished its iteration ('loaded'); the exploding
    # source died mid-load and its lifecycle never advanced past 'loading'.
    with db.connection() as conn:
        lifecycle = {
            str(row.source_name): str(row.lifecycle_state)
            for row in conn.execute(
                select(run_sources_table.c.source_name, run_sources_table.c.lifecycle_state).where(run_sources_table.c.run_id == run_id)
            )
        }
    assert lifecycle == {"orders": "exhausted", "exploding": "loading"}

    # All four ingested rows (3 surviving + 1 pre-crash) are durably parked
    # PENDING_SINK — present, attributed, and non-terminal: the exact shape a
    # resume sweep needs. Nothing was silently terminalized.
    token_sources = _token_source_names(db, run_id)
    items = _work_items(db, run_id)
    assert len(items) == 4
    assert all(item["status"] == TokenWorkStatus.PENDING_SINK.value for item in items)
    assert sorted(token_sources[item["token_id"]] for item in items) == ["exploding", "orders", "orders", "orders"]


@pytest.mark.timeout(60)
def test_source_data_errors_mid_stream_quarantine_and_surviving_source_completes(tmp_path: Path) -> None:
    """Data-level mid-stream source errors quarantine; the run is NOT masked.

    The refunds source (declared first) hits a validation failure on its
    MIDDLE row under a fixed contract and — with
    ``on_validation_failure: quarantine`` — emits a quarantined SourceRow,
    then keeps emitting; the orders source (still to emit under ADR-025
    sequential ingest) afterwards completes untouched. Wave-1 semantics
    (f487d7b13, cleanup no longer masks ceremonies +
    derive_terminal_run_status): a quarantined row is a CLEAN deliberate
    determination, but any run containing one must terminate in the
    COMPLETED_WITH_FAILURES family rather than pretending to be COMPLETED —
    that is asserted here as the run's terminal status, durably in the runs
    table and on the public RunResult.

    Durable isolation invariants: both sources reach lifecycle 'loaded'; the
    quarantined row lands in the quarantine sink with QUARANTINED_AT_SOURCE
    provenance attributed to refunds only; every surviving row delivers; the
    scheduler journal ends with zero non-terminal work.
    """
    refunds_path = tmp_path / "refunds.csv"
    orders_path = tmp_path / "orders.csv"
    output_path = tmp_path / "out.jsonl"
    quarantine_path = tmp_path / "quarantine.jsonl"
    refunds_path.write_text("id,amount\nr1,5\nr2,oops\nr3,7\n")
    orders_path.write_text("id,amount\no1,10\no2,20\no3,30\n")

    settings = load_settings_from_yaml_string(
        f"""
sources:
  refunds:
    plugin: csv
    on_success: inbound
    options:
      path: {refunds_path}
      on_validation_failure: quarantine
      schema:
        mode: fixed
        fields:
          - "id: str"
          - "amount: int"
  orders:
    plugin: csv
    on_success: inbound
    options:
      path: {orders_path}
      on_validation_failure: quarantine
      schema:
        mode: fixed
        fields:
          - "id: str"
          - "amount: int"
queues:
  inbound: {{}}
transforms:
  - name: normalize_rows
    plugin: passthrough
    input: inbound
    on_success: output
    on_error: discard
    options:
      schema:
        mode: observed
sinks:
  output:
    plugin: json
    on_write_failure: discard
    options:
      path: {output_path}
      format: jsonl
      schema:
        mode: observed
  quarantine:
    plugin: json
    on_write_failure: discard
    options:
      path: {quarantine_path}
      format: jsonl
      schema:
        mode: observed
"""
    )
    bundle = instantiate_plugins_from_config(settings)
    graph = ExecutionGraph.from_plugin_instances(
        sources=bundle.sources,
        source_settings_map=bundle.source_settings_map,
        transforms=bundle.transforms,
        sinks=bundle.sinks,
        aggregations=bundle.aggregations,
        gates=list(settings.gates),
        queues=settings.queues,
    )
    config = assemble_and_validate_pipeline_config(
        sources=bundle.sources,
        transforms=bundle.transforms,
        sinks=bundle.sinks,
        aggregations=bundle.aggregations,
        settings=settings,
        graph=graph,
    )
    db = LandscapeDB(f"sqlite:///{tmp_path / 'audit.db'}")
    result = Orchestrator(db).run(
        config,
        graph=graph,
        settings=settings,
        payload_store=FilesystemPayloadStore(tmp_path / "payloads"),
    )

    # Wave-1 unmasking: a quarantine-bearing run is COMPLETED_WITH_FAILURES.
    assert result.status == RunStatus.COMPLETED_WITH_FAILURES
    assert result.rows_processed == 6
    assert result.rows_succeeded == 5
    assert result.rows_quarantined == 1

    output_rows = [json.loads(line) for line in output_path.read_text().splitlines()]
    quarantine_rows = [json.loads(line) for line in quarantine_path.read_text().splitlines()]
    assert {row["id"] for row in output_rows} == {"r1", "r3", "o1", "o2", "o3"}
    assert [row["id"] for row in quarantine_rows] == ["r2"]
    assert quarantine_rows[0]["amount"] == "oops"

    run_id = result.run_id
    assert _run_status(db, run_id) == RunStatus.COMPLETED_WITH_FAILURES.value

    # Both sources finished their iterations despite the mid-stream data error.
    with db.connection() as conn:
        lifecycle = {
            str(row.source_name): str(row.lifecycle_state)
            for row in conn.execute(
                select(run_sources_table.c.source_name, run_sources_table.c.lifecycle_state).where(run_sources_table.c.run_id == run_id)
            )
        }
    assert lifecycle == {"refunds": "exhausted", "orders": "exhausted"}

    # Quarantine provenance is attributed to refunds only; orders untouched.
    outcomes = _outcomes_by_source(db, run_id)
    quarantined = [outcome for outcome in outcomes if outcome[2] == "quarantined_at_source"]
    assert quarantined == [("refunds", "failure", "quarantined_at_source", "quarantine")]
    assert outcomes.count(("orders", "success", "default_flow", "output")) == 3
    assert outcomes.count(("refunds", "success", "default_flow", "output")) == 2

    # The scheduler journal ends fully resolved: no orphaned READY/BLOCKED/
    # LEASED/PENDING_SINK work survives the run.
    items = _work_items(db, run_id)
    assert items, "expected scheduler work items for processed rows"
    assert all(item["status"] == TokenWorkStatus.TERMINAL.value for item in items)
