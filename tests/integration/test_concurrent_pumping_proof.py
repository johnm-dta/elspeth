# tests/integration/test_concurrent_pumping_proof.py
"""C2 proofs: the durable token scheduler pumps overlapping token lifecycles.

The claim under test (ADR-026): token progress through the DAG is interleaved
via the durable work queue — a later-admitted token can be claimed and advanced
while an earlier-admitted token is still mid-flight (non-terminal BLOCKED /
PENDING_SINK). The degenerate failure mode ruled out here is a scheduler that
drives each token all the way to TERMINAL before admitting the next.

Scope note (ADR-025 / ADR-026, "sequential multi-source ingest"): source
ITERATION is contractually sequential — source B's rows are not read until
source A exhausts — and a claimed token executes its transform segment under
one lease, single-threaded. "Concurrent pumping" therefore means overlapping
token LIFECYCLES through the scheduler states, which is exactly what the
durable ``scheduler_events`` journal proves below. Cross-source compute
overlap would violate the ADR contract and is asserted ABSENT (the
sequential-ingest characterization), not worked around.

Proof instruments:
- a recording transform (in-process processing order), and
- the ``scheduler_events`` audit journal in SQLite insertion (rowid) order,
  which is the durable, deterministic record of every lifecycle transition.
"""

from __future__ import annotations

import json
from typing import Any, ClassVar

import pytest
from sqlalchemy import literal_column, select

from elspeth.contracts import Determinism, PluginSchema, RunStatus
from elspeth.core.config import QueueSettings, SourceSettings, TransformSettings, load_settings_from_yaml_string
from elspeth.core.dag import ExecutionGraph
from elspeth.core.dag.wiring import WiredTransform
from elspeth.core.landscape import LandscapeDB
from elspeth.core.landscape.schema import rows_table, scheduler_events_table, tokens_table
from elspeth.core.payload_store import FilesystemPayloadStore
from elspeth.engine.orchestrator import Orchestrator, PipelineConfig
from elspeth.plugins.infrastructure.base import BaseTransform
from elspeth.plugins.infrastructure.results import TransformResult
from tests.fixtures.base_classes import _TestSchema, as_sink, as_source, as_transform
from tests.fixtures.plugins import CollectSink, ListSource


class _RecordingTransform(BaseTransform):
    """Passthrough transform that records each row's source marker in call order."""

    name = "recording_transform"
    determinism = Determinism.DETERMINISTIC
    input_schema: ClassVar[type[PluginSchema]] = _TestSchema
    output_schema: ClassVar[type[PluginSchema]] = _TestSchema

    def __init__(self, processed: list[str], *, input_connection: str, on_success: str) -> None:
        super().__init__({"schema": {"mode": "observed"}})
        self.input = input_connection
        self.on_success = on_success
        self.on_error = "discard"
        self._processed = processed

    def process(self, row: Any, ctx: Any) -> TransformResult:
        self._processed.append(row["src"])
        return TransformResult.success(row, success_reason={"action": "recorded"})


def _token_source_map(db: LandscapeDB, run_id: str) -> dict[str, str]:
    """Map each token_id to the source_node_id of the row it carries."""
    with db.connection() as conn:
        return dict(
            conn.execute(
                select(tokens_table.c.token_id, rows_table.c.source_node_id)
                .join(rows_table, tokens_table.c.row_id == rows_table.c.row_id)
                .where(tokens_table.c.run_id == run_id)
            ).all()
        )


def _scheduler_timeline(db: LandscapeDB, run_id: str) -> list[tuple[int, str, str, str]]:
    """Scheduler events as (insertion_seq, token_id, event_type, to_status).

    ``scheduler_events.event_id`` is a content hash, so SQLite rowid is the
    insertion-order key. Within one run the repository writes events inside
    the transition transactions, so rowid order IS transition order.
    """
    with db.connection() as conn:
        rows = conn.execute(
            select(
                literal_column("scheduler_events.rowid").label("seq"),
                scheduler_events_table.c.token_id,
                scheduler_events_table.c.event_type,
                scheduler_events_table.c.to_status,
            )
            .where(scheduler_events_table.c.run_id == run_id)
            .order_by(literal_column("scheduler_events.rowid"))
        ).all()
    return [(row.seq, row.token_id, row.event_type, row.to_status) for row in rows]


def _events_for(
    timeline: list[tuple[int, str, str, str]],
    token_source: dict[str, str],
    source_node_id: str,
    event_type: str,
) -> list[int]:
    """Insertion sequences of ``event_type`` events for tokens of one source."""
    return [seq for seq, token_id, etype, _ in timeline if etype == event_type and token_source.get(token_id) == source_node_id]


@pytest.mark.timeout(60)
class TestConcurrentTokenPumping:
    """C2: overlapping token lifecycles through the durable work queue."""

    def test_second_source_tokens_advance_while_first_source_tokens_are_mid_flight(self, landscape_db: LandscapeDB, payload_store) -> None:
        """Tokens of the second-admitted source are claimed and pumped while every
        first-source token is still mid-flight (durably parked at PENDING_SINK,
        not TERMINAL) — refuting drain-each-token-to-terminal-before-admitting-next.
        """
        processed: list[str] = []
        orders = ListSource([{"src": "orders", "value": i} for i in range(5)], name="orders_source", on_success="inbound")
        refunds = ListSource([{"src": "refunds", "value": i} for i in range(5)], name="refunds_source", on_success="inbound")
        recorder = _RecordingTransform(processed, input_connection="inbound", on_success="output")
        sink = CollectSink("output")

        sources = {"orders": as_source(orders), "refunds": as_source(refunds)}
        source_settings = {
            "orders": SourceSettings(plugin=orders.name, on_success="inbound"),
            "refunds": SourceSettings(plugin=refunds.name, on_success="inbound"),
        }
        wired_recorder = WiredTransform(
            plugin=as_transform(recorder),
            settings=TransformSettings(
                name="recording_transform_0",
                plugin=recorder.name,
                input="inbound",
                on_success="output",
                on_error="discard",
                options={},
            ),
        )
        graph = ExecutionGraph.from_plugin_instances(
            sources=sources,
            source_settings_map=source_settings,
            transforms=[wired_recorder],
            sinks={"output": as_sink(sink)},
            queues={"inbound": QueueSettings()},
        )
        config = PipelineConfig(
            sources=sources,
            transforms=[as_transform(recorder)],
            sinks={"output": as_sink(sink)},
        )

        run_result = Orchestrator(landscape_db).run(config, graph=graph, payload_store=payload_store)

        assert run_result.status == RunStatus.COMPLETED
        assert run_result.rows_processed == 10
        assert len(sink.results) == 10

        # ADR-025/026 sequential-ingest characterization: transform-level
        # compute order is strictly source-by-source in declaration order.
        # This is the contracted NON-overlap surface — overlap lives in the
        # scheduler lifecycle below, not in cross-source row admission.
        assert processed == ["orders"] * 5 + ["refunds"] * 5

        token_source = _token_source_map(landscape_db, run_result.run_id)
        with landscape_db.engine.connect() as conn:
            source_ids = (
                conn.execute(select(rows_table.c.source_node_id).where(rows_table.c.run_id == run_result.run_id).distinct()).scalars().all()
            )
        orders_node = next(node_id for node_id in source_ids if "orders" in node_id)
        refunds_node = next(node_id for node_id in source_ids if "refunds" in node_id)

        timeline = _scheduler_timeline(landscape_db, run_result.run_id)
        orders_parked = _events_for(timeline, token_source, orders_node, "mark_pending_sink")
        orders_terminal = _events_for(timeline, token_source, orders_node, "mark_pending_sink_terminal")
        refunds_claimed = _events_for(timeline, token_source, refunds_node, "claim_ready")
        refunds_parked = _events_for(timeline, token_source, refunds_node, "mark_pending_sink")

        assert len(orders_parked) == 5
        assert len(orders_terminal) == 5
        assert len(refunds_claimed) == 5

        # Overlap proof, part 1: every orders token was parked NON-TERMINAL
        # (PENDING_SINK) before the first refunds token was even claimed...
        first_refunds_claim = min(refunds_claimed)
        assert all(seq < first_refunds_claim for seq in orders_parked)

        # ...part 2: refunds tokens were claimed AND advanced to their own
        # parked state while every orders token was still non-terminal —
        # orders terminalization happens only after sink durability, after
        # all ten lifecycles were simultaneously open.
        first_orders_terminal = min(orders_terminal)
        assert first_refunds_claim < first_orders_terminal
        assert all(seq < first_orders_terminal for seq in refunds_parked)

        # Degenerate-mode refutation within a single source's batch too:
        # orders token #2 is enqueued before orders token #1 terminalizes.
        orders_enqueues = _events_for(timeline, token_source, orders_node, "enqueue")
        assert sorted(orders_enqueues)[1] < first_orders_terminal

    def test_aggregation_rendezvous_completes_only_via_overlapping_lifecycles(self, tmp_path) -> None:
        """A count-2 aggregation barrier fed by two single-row sources is a
        rendezvous: the first source's token can reach TERMINAL only if it is
        still alive (durably BLOCKED) when the second source's token arrives.

        Atomic per-token draining could not satisfy the barrier — the run
        would stall awaiting a second token that is never admitted (caught by
        the timeout) or flush a batch of one (caught by the batch_size
        assertion). A single flush covering both sources, with the durable
        event order BLOCKED(A) < ENQUEUE(B) < CLAIM(B) < BARRIER_TERMINAL(A),
        is positive proof of overlapping token lifecycles.
        """
        orders_path = tmp_path / "orders.csv"
        refunds_path = tmp_path / "refunds.csv"
        output_path = tmp_path / "totals.jsonl"
        orders_path.write_text("id,amount\n1,10\n")
        refunds_path.write_text("id,amount\nr1,5\n")

        settings = load_settings_from_yaml_string(
            f"""
sources:
  orders:
    plugin: csv
    on_success: batch_in
    options:
      path: {orders_path}
      on_validation_failure: discard
      schema:
        mode: fixed
        fields:
          - "id: str"
          - "amount: int"
  refunds:
    plugin: csv
    on_success: batch_in
    options:
      path: {refunds_path}
      on_validation_failure: discard
      schema:
        mode: fixed
        fields:
          - "id: str"
          - "amount: int"
queues:
  batch_in: {{}}
aggregations:
  - name: total_amounts
    plugin: batch_stats
    input: batch_in
    on_success: output
    on_error: discard
    trigger:
      count: 2
    output_mode: transform
    options:
      schema:
        mode: observed
      value_field: amount
sinks:
  output:
    plugin: json
    on_write_failure: discard
    options:
      path: {output_path}
      format: jsonl
      schema:
        mode: observed
"""
        )

        from elspeth.cli_helpers import instantiate_plugins_from_config
        from elspeth.engine.orchestrator.preflight import assemble_and_validate_pipeline_config

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

        assert result.status == RunStatus.COMPLETED
        assert result.rows_processed == 2

        # The rendezvous completed: exactly one flush containing BOTH rows.
        flushes = [json.loads(line) for line in output_path.read_text().splitlines()]
        assert len(flushes) == 1
        assert flushes[0]["batch_size"] == 2
        assert flushes[0]["sum"] == 15

        token_source = _token_source_map(db, result.run_id)
        with db.engine.connect() as conn:
            source_ids = (
                conn.execute(select(rows_table.c.source_node_id).where(rows_table.c.run_id == result.run_id).distinct()).scalars().all()
            )
        orders_node = next(node_id for node_id in source_ids if "orders" in node_id)
        refunds_node = next(node_id for node_id in source_ids if "refunds" in node_id)

        timeline = _scheduler_timeline(db, result.run_id)
        (orders_blocked,) = _events_for(timeline, token_source, orders_node, "mark_blocked")
        (orders_released,) = _events_for(timeline, token_source, orders_node, "mark_blocked_barrier_terminal")
        (refunds_enqueued,) = _events_for(timeline, token_source, refunds_node, "enqueue")
        (refunds_claimed,) = _events_for(timeline, token_source, refunds_node, "claim_ready")

        # Durable rendezvous ordering: A parks mid-flight, B is admitted and
        # claimed while A is still BLOCKED, and only then does A terminalize.
        assert orders_blocked < refunds_enqueued < refunds_claimed < orders_released
