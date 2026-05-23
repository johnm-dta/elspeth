# tests/integration/audit/test_recorder_routing_events.py
"""Tests for RecorderFactory routing event operations."""

from __future__ import annotations

import re
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest
from sqlalchemy import select

from elspeth.contracts import NodeType, RoutingMode, RoutingSpec, TerminalOutcome, TerminalPath
from elspeth.contracts.audit import TokenRef
from elspeth.contracts.errors import ConfigGateReason
from elspeth.contracts.schema import SchemaConfig
from elspeth.core.config import GateSettings
from elspeth.core.landscape.database import LandscapeDB
from elspeth.core.landscape.errors import LandscapeRecordError
from elspeth.core.landscape.factory import RecorderFactory
from elspeth.core.landscape.lineage import explain
from elspeth.core.landscape.schema import (
    edges_table,
    node_states_table,
    nodes_table,
    routing_events_table,
    token_outcomes_table,
)
from elspeth.engine.orchestrator import Orchestrator, PipelineConfig
from tests.fixtures.base_classes import as_sink, as_source, as_transform
from tests.fixtures.pipeline import build_production_graph
from tests.fixtures.plugins import CollectSink, ConditionalErrorTransform, ListSource

# Dynamic schema for tests that don't care about specific fields
DYNAMIC_SCHEMA = SchemaConfig.from_dict({"mode": "observed"})


class TestRecorderFactoryRouting:
    """Routing event recording (gate decisions)."""

    def test_record_routing_event(self) -> None:
        from elspeth.core.landscape.database import LandscapeDB
        from elspeth.core.landscape.factory import RecorderFactory

        db = LandscapeDB.in_memory()
        factory = RecorderFactory(db)
        run = factory.run_lifecycle.begin_run(config={}, canonical_version="v1")

        source = factory.data_flow.register_node(
            run_id=run.run_id,
            plugin_name="source",
            node_type=NodeType.SOURCE,
            plugin_version="1.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )
        gate = factory.data_flow.register_node(
            run_id=run.run_id,
            plugin_name="gate",
            node_type=NodeType.GATE,
            plugin_version="1.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )
        sink = factory.data_flow.register_node(
            run_id=run.run_id,
            plugin_name="sink",
            node_type=NodeType.SINK,
            plugin_version="1.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )
        edge = factory.data_flow.register_edge(
            run_id=run.run_id,
            from_node_id=gate.node_id,
            to_node_id=sink.node_id,
            label="high_value",
            mode=RoutingMode.MOVE,
        )

        row = factory.data_flow.create_row(
            run_id=run.run_id,
            source_node_id=source.node_id,
            row_index=0,
            data={},
            source_row_index=0,
            ingest_sequence=0,
        )
        token = factory.data_flow.create_token(row_id=row.row_id)
        state = factory.execution.begin_node_state(
            token_id=token.token_id,
            node_id=gate.node_id,
            run_id=run.run_id,
            step_index=0,
            input_data={},
        )

        event = factory.execution.record_routing_event(
            state_id=state.state_id,
            edge_id=edge.edge_id,
            mode=RoutingMode.MOVE,
            reason={"condition": "value > 1000", "result": "true"},
        )

        assert event.event_id is not None
        assert event.routing_group_id is not None  # Auto-generated
        assert event.edge_id == edge.edge_id
        assert event.mode == "move"

    def test_record_multiple_routing_events(self) -> None:
        """Test recording fork to multiple destinations."""
        from elspeth.core.landscape.database import LandscapeDB
        from elspeth.core.landscape.factory import RecorderFactory

        db = LandscapeDB.in_memory()
        factory = RecorderFactory(db)
        run = factory.run_lifecycle.begin_run(config={}, canonical_version="v1")

        gate = factory.data_flow.register_node(
            run_id=run.run_id,
            plugin_name="gate",
            node_type=NodeType.GATE,
            plugin_version="1.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )
        sink_a = factory.data_flow.register_node(
            run_id=run.run_id,
            plugin_name="sink_a",
            node_type=NodeType.SINK,
            plugin_version="1.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )
        sink_b = factory.data_flow.register_node(
            run_id=run.run_id,
            plugin_name="sink_b",
            node_type=NodeType.SINK,
            plugin_version="1.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )
        edge_a = factory.data_flow.register_edge(
            run_id=run.run_id,
            from_node_id=gate.node_id,
            to_node_id=sink_a.node_id,
            label="path_a",
            mode=RoutingMode.COPY,
        )
        edge_b = factory.data_flow.register_edge(
            run_id=run.run_id,
            from_node_id=gate.node_id,
            to_node_id=sink_b.node_id,
            label="path_b",
            mode=RoutingMode.COPY,
        )

        row = factory.data_flow.create_row(
            run_id=run.run_id,
            source_node_id=gate.node_id,
            row_index=0,
            data={},
            source_row_index=0,
            ingest_sequence=0,
        )
        token = factory.data_flow.create_token(row_id=row.row_id)
        state = factory.execution.begin_node_state(
            token_id=token.token_id,
            node_id=gate.node_id,
            run_id=run.run_id,
            step_index=0,
            input_data={},
        )

        # Fork to both paths using batch method
        events = factory.execution.record_routing_events(
            state_id=state.state_id,
            routes=[
                RoutingSpec(edge_id=edge_a.edge_id, mode=RoutingMode.COPY),
                RoutingSpec(edge_id=edge_b.edge_id, mode=RoutingMode.COPY),
            ],
            reason={"condition": "fork_to_paths", "result": "path_a,path_b"},
        )

        assert len(events) == 2
        # All events share the same routing_group_id
        assert events[0].routing_group_id == events[1].routing_group_id
        assert events[0].ordinal == 0
        assert events[1].ordinal == 1

    def test_routing_event_stores_reason_payload_single_route(self) -> None:
        """Test that single routing event stores reason payload."""
        import json

        from elspeth.core.landscape.database import LandscapeDB
        from elspeth.core.landscape.factory import RecorderFactory
        from elspeth.core.payload_store import FilesystemPayloadStore

        db = LandscapeDB.in_memory()
        with tempfile.TemporaryDirectory() as tmp_dir:
            payload_store = FilesystemPayloadStore(Path(tmp_dir))
            factory = RecorderFactory(db, payload_store=payload_store)
            run = factory.run_lifecycle.begin_run(config={}, canonical_version="v1")

            source = factory.data_flow.register_node(
                run_id=run.run_id,
                plugin_name="source",
                node_type=NodeType.SOURCE,
                plugin_version="1.0",
                config={},
                schema_config=DYNAMIC_SCHEMA,
            )
            gate = factory.data_flow.register_node(
                run_id=run.run_id,
                plugin_name="gate",
                node_type=NodeType.GATE,
                plugin_version="1.0",
                config={},
                schema_config=DYNAMIC_SCHEMA,
            )
            sink = factory.data_flow.register_node(
                run_id=run.run_id,
                plugin_name="sink",
                node_type=NodeType.SINK,
                plugin_version="1.0",
                config={},
                schema_config=DYNAMIC_SCHEMA,
            )
            edge = factory.data_flow.register_edge(
                run_id=run.run_id,
                from_node_id=gate.node_id,
                to_node_id=sink.node_id,
                label="high_value",
                mode=RoutingMode.MOVE,
            )

            row = factory.data_flow.create_row(
                run_id=run.run_id,
                source_node_id=source.node_id,
                row_index=0,
                data={},
                source_row_index=0,
                ingest_sequence=0,
            )
            token = factory.data_flow.create_token(row_id=row.row_id)
            state = factory.execution.begin_node_state(
                token_id=token.token_id,
                node_id=gate.node_id,
                run_id=run.run_id,
                step_index=0,
                input_data={},
            )

            reason: ConfigGateReason = {"condition": "value > 1000", "result": "true"}
            event = factory.execution.record_routing_event(
                state_id=state.state_id,
                edge_id=edge.edge_id,
                mode=RoutingMode.MOVE,
                reason=reason,
            )

            # Verify reason payload was stored
            assert event.reason_ref is not None, "reason_ref should be set when reason is provided"
            assert event.reason_hash is not None, "reason_hash should be computed from reason"

            # Verify we can retrieve the payload
            retrieved = payload_store.retrieve(event.reason_ref)
            retrieved_reason = json.loads(retrieved.decode("utf-8"))
            assert retrieved_reason == reason

    def test_routing_events_stores_reason_payload_multi_route(self) -> None:
        """Test that multi-route events store shared reason payload."""
        import json

        from elspeth.core.landscape.database import LandscapeDB
        from elspeth.core.landscape.factory import RecorderFactory
        from elspeth.core.payload_store import FilesystemPayloadStore

        db = LandscapeDB.in_memory()
        with tempfile.TemporaryDirectory() as tmp_dir:
            payload_store = FilesystemPayloadStore(Path(tmp_dir))
            factory = RecorderFactory(db, payload_store=payload_store)
            run = factory.run_lifecycle.begin_run(config={}, canonical_version="v1")

            gate = factory.data_flow.register_node(
                run_id=run.run_id,
                plugin_name="gate",
                node_type=NodeType.GATE,
                plugin_version="1.0",
                config={},
                schema_config=DYNAMIC_SCHEMA,
            )
            sink_a = factory.data_flow.register_node(
                run_id=run.run_id,
                plugin_name="sink_a",
                node_type=NodeType.SINK,
                plugin_version="1.0",
                config={},
                schema_config=DYNAMIC_SCHEMA,
            )
            sink_b = factory.data_flow.register_node(
                run_id=run.run_id,
                plugin_name="sink_b",
                node_type=NodeType.SINK,
                plugin_version="1.0",
                config={},
                schema_config=DYNAMIC_SCHEMA,
            )
            edge_a = factory.data_flow.register_edge(
                run_id=run.run_id,
                from_node_id=gate.node_id,
                to_node_id=sink_a.node_id,
                label="path_a",
                mode=RoutingMode.COPY,
            )
            edge_b = factory.data_flow.register_edge(
                run_id=run.run_id,
                from_node_id=gate.node_id,
                to_node_id=sink_b.node_id,
                label="path_b",
                mode=RoutingMode.COPY,
            )

            row = factory.data_flow.create_row(
                run_id=run.run_id,
                source_node_id=gate.node_id,
                row_index=0,
                data={},
                source_row_index=0,
                ingest_sequence=0,
            )
            token = factory.data_flow.create_token(row_id=row.row_id)
            state = factory.execution.begin_node_state(
                token_id=token.token_id,
                node_id=gate.node_id,
                run_id=run.run_id,
                step_index=0,
                input_data={},
            )

            reason: ConfigGateReason = {"condition": "fork_to_paths", "result": "path_a,path_b"}
            events = factory.execution.record_routing_events(
                state_id=state.state_id,
                routes=[
                    RoutingSpec(edge_id=edge_a.edge_id, mode=RoutingMode.COPY),
                    RoutingSpec(edge_id=edge_b.edge_id, mode=RoutingMode.COPY),
                ],
                reason=reason,
            )

            # Both events should reference the same payload
            assert len(events) == 2
            assert events[0].reason_ref is not None, "reason_ref should be set when reason is provided"
            assert events[1].reason_ref is not None, "reason_ref should be set when reason is provided"
            assert events[0].reason_ref == events[1].reason_ref, "All events should share same reason payload"
            assert events[0].reason_hash == events[1].reason_hash, "All events should share same reason hash"

            # Verify we can retrieve the shared payload
            retrieved = payload_store.retrieve(events[0].reason_ref)
            retrieved_reason = json.loads(retrieved.decode("utf-8"))
            assert retrieved_reason == reason


# ── elspeth-5069612f3c — producer-scoped MOVE-vs-DIVERT distinguishability ──
#
# These tests pin the **producer-scoped biconditional** for the rows_routed
# split: gate ``route_to_sink`` MOVE rows produce ``SUCCESS/GATE_ROUTED`` with
# ``error_hash IS NULL``, while transform ``on_error`` DIVERT rows produce
# ``FAILURE/ON_ERROR_ROUTED`` with a non-empty ``error_hash``.  The scope is
# intentionally narrow: ``RoutingMode.DIVERT`` is also used by source
# quarantine and sink failsink edges, so a global ``RoutingMode.DIVERT ⇔
# token_outcomes.outcome='routed_on_error'`` biconditional would be false.
# Only the two named producer sites — gate MOVE and transform on_error DIVERT
# — are under test here.


def _run_mixed_move_and_divert_pipeline(
    landscape_db: LandscapeDB,
    payload_store,
):
    """Run source -> transform -> gate with one MOVE-routed row and one
    on_error-DIVERT-routed row.

    The transform's on_error sends the failing row to ``error_sink`` (DIVERT);
    the gate's ``route_to_sink`` sends the surviving row to ``high_priority``
    (MOVE).
    """
    source = ListSource(
        [
            {"id": 1, "priority": "high", "fail": False},
            {"id": 2, "priority": "low", "fail": True},
        ],
        on_success="transform_in",
    )
    transform = ConditionalErrorTransform(
        name="may_fail",
        input_connection="transform_in",
        on_success="route_in",
        on_error="error_sink",
    )
    gate = GateSettings(
        name="priority_gate",
        input="route_in",
        condition="row['priority'] == 'high'",
        routes={"true": "high_priority", "false": "default"},
    )
    sinks = {
        "high_priority": CollectSink(name="high_priority"),
        "default": CollectSink(name="default"),
        "error_sink": CollectSink(name="error_sink"),
    }
    config = PipelineConfig(
        sources={"primary": as_source(source)},
        transforms=[as_transform(transform)],
        sinks={name: as_sink(sink) for name, sink in sinks.items()},
        gates=[gate],
    )

    return Orchestrator(landscape_db).run(
        config,
        graph=build_production_graph(config),
        payload_store=payload_store,
    )


def _fetch_single_outcome_for_sink(
    landscape_db: LandscapeDB,
    *,
    run_id: str,
    sink_name: str,
) -> Mapping[str, Any]:
    stmt = (
        select(
            token_outcomes_table.c.token_id,
            token_outcomes_table.c.outcome,
            token_outcomes_table.c.path,
            token_outcomes_table.c.error_hash,
            token_outcomes_table.c.sink_name,
        )
        .where(token_outcomes_table.c.run_id == run_id)
        .where(token_outcomes_table.c.sink_name == sink_name)
        .order_by(token_outcomes_table.c.recorded_at, token_outcomes_table.c.outcome_id)
    )
    with landscape_db.engine.connect() as conn:
        rows = conn.execute(stmt).mappings().all()
    assert len(rows) == 1, f"expected one token_outcomes row for sink {sink_name!r}, got {rows}"
    return dict(rows[0])


def _fetch_routing_modes_to_sink(
    landscape_db: LandscapeDB,
    *,
    run_id: str,
    sink_plugin_name: str,
) -> list[str]:
    stmt = (
        select(routing_events_table.c.mode)
        .select_from(
            routing_events_table.join(
                node_states_table,
                routing_events_table.c.state_id == node_states_table.c.state_id,
            )
            .join(edges_table, routing_events_table.c.edge_id == edges_table.c.edge_id)
            .join(
                nodes_table,
                (edges_table.c.to_node_id == nodes_table.c.node_id) & (node_states_table.c.run_id == nodes_table.c.run_id),
            )
        )
        .where(node_states_table.c.run_id == run_id)
        .where(nodes_table.c.plugin_name == sink_plugin_name)
        .order_by(routing_events_table.c.ordinal, routing_events_table.c.event_id)
    )
    with landscape_db.engine.connect() as conn:
        return [str(row.mode) for row in conn.execute(stmt)]


class TestRoutingEventDistinguishability:
    """elspeth-5069612f3c — pin the producer-scoped Tier-1 audit
    distinguishability between gate-routed MOVE tokens and transform
    on_error-routed DIVERT tokens.

    These assertions are deliberately producer-scoped: gate ``route_to_sink``
    MOVE -> ``SUCCESS/GATE_ROUTED`` with ``error_hash IS NULL``; transform
    ``on_error`` DIVERT -> ``FAILURE/ON_ERROR_ROUTED`` with a non-empty
    ``error_hash``.  ``RoutingMode.DIVERT`` is also emitted by source
    quarantine and sink failsink edges, so a global
    ``RoutingMode.DIVERT ⇔ token_outcomes.outcome='routed_on_error'``
    biconditional is false in the current architecture and is intentionally
    NOT pinned here.
    """

    def test_gate_routed_token_records_routed_outcome_with_null_error_hash(self, landscape_db: LandscapeDB, payload_store) -> None:
        """A row routed via gate route_to_sink (intentional MOVE) must record:
        - token_outcomes.outcome == 'routed'
        - token_outcomes.error_hash IS NULL
        - routing_events.mode == 'move' on the edge that carried it
        """
        run_result = _run_mixed_move_and_divert_pipeline(landscape_db, payload_store)

        outcome = _fetch_single_outcome_for_sink(
            landscape_db,
            run_id=run_result.run_id,
            sink_name="high_priority",
        )
        assert outcome["outcome"] == TerminalOutcome.SUCCESS.value
        assert outcome["path"] == TerminalPath.GATE_ROUTED.value
        assert outcome["error_hash"] is None

        modes = _fetch_routing_modes_to_sink(
            landscape_db,
            run_id=run_result.run_id,
            sink_plugin_name="high_priority",
        )
        assert modes == [RoutingMode.MOVE.value]

    def test_on_error_routed_token_records_routed_on_error_with_non_empty_error_hash(
        self, landscape_db: LandscapeDB, payload_store
    ) -> None:
        """A row routed via transform on_error (DIVERT) must record:
        - token_outcomes.outcome == 'routed_on_error'  (NOT 'routed')
        - token_outcomes.error_hash IS NOT NULL and matches the canonical
          16-char sha256 prefix recipe
        - routing_events.mode == 'divert' on the edge that carried it
        """
        run_result = _run_mixed_move_and_divert_pipeline(landscape_db, payload_store)

        outcome = _fetch_single_outcome_for_sink(
            landscape_db,
            run_id=run_result.run_id,
            sink_name="error_sink",
        )
        assert outcome["outcome"] == TerminalOutcome.FAILURE.value
        assert outcome["path"] == TerminalPath.ON_ERROR_ROUTED.value
        assert outcome["error_hash"] is not None
        assert re.fullmatch(r"[0-9a-f]{16}", str(outcome["error_hash"]))

        modes = _fetch_routing_modes_to_sink(
            landscape_db,
            run_id=run_result.run_id,
            sink_plugin_name="error_sink",
        )
        assert modes == [RoutingMode.DIVERT.value]

    def test_explain_recovers_routing_intent_for_both_variants(self, landscape_db: LandscapeDB, payload_store) -> None:
        """The explain() function (the contractual audit-attributability surface)
        must distinguish the two producer-scoped variants single-hop. Run a
        mixed pipeline (one row gate-routed, one on_error-routed), call
        explain() for each, and assert the returned record contains:
        - For gate-routed token: outcome=ROUTED, sink_name set, error_hash absent
        - For on_error token: outcome=ROUTED_ON_ERROR, sink_name set,
          error_hash present and non-empty
        """
        run_result = _run_mixed_move_and_divert_pipeline(landscape_db, payload_store)
        factory = RecorderFactory(landscape_db, payload_store=payload_store)

        gate_outcome = _fetch_single_outcome_for_sink(
            landscape_db,
            run_id=run_result.run_id,
            sink_name="high_priority",
        )
        error_outcome = _fetch_single_outcome_for_sink(
            landscape_db,
            run_id=run_result.run_id,
            sink_name="error_sink",
        )

        gate_lineage = explain(
            factory.query,
            factory.data_flow,
            run_result.run_id,
            token_id=str(gate_outcome["token_id"]),
        )
        error_lineage = explain(
            factory.query,
            factory.data_flow,
            run_result.run_id,
            token_id=str(error_outcome["token_id"]),
        )

        assert gate_lineage is not None
        assert gate_lineage.outcome is not None
        assert gate_lineage.outcome.outcome == TerminalOutcome.SUCCESS
        assert gate_lineage.outcome.path == TerminalPath.GATE_ROUTED
        assert gate_lineage.outcome.sink_name == "high_priority"
        assert gate_lineage.outcome.error_hash is None
        assert any(event.mode == RoutingMode.MOVE for event in gate_lineage.routing_events)

        assert error_lineage is not None
        assert error_lineage.outcome is not None
        assert error_lineage.outcome.outcome == TerminalOutcome.FAILURE
        assert error_lineage.outcome.path == TerminalPath.ON_ERROR_ROUTED
        assert error_lineage.outcome.sink_name == "error_sink"
        assert error_lineage.outcome.error_hash is not None
        assert re.fullmatch(r"[0-9a-f]{16}", error_lineage.outcome.error_hash)
        assert any(event.mode == RoutingMode.DIVERT for event in error_lineage.routing_events)

    def test_token_outcome_unique_constraint_admits_routed_on_error(self, landscape_db: LandscapeDB) -> None:
        """The token_outcomes partial unique index (one terminal outcome per
        token, see docs/contracts/token-outcomes/00-token-outcome-contract.md)
        must admit ROUTED_ON_ERROR like any other terminal outcome.  Recording
        a second terminal outcome on the same token AFTER ROUTED_ON_ERROR
        must raise the unique-violation error (not silently overwrite, not
        silently drop).
        """
        factory = RecorderFactory(landscape_db)
        run = factory.run_lifecycle.begin_run(config={}, canonical_version="v1")
        source = factory.data_flow.register_node(
            run_id=run.run_id,
            plugin_name="source",
            node_type=NodeType.SOURCE,
            plugin_version="1.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )
        row = factory.data_flow.create_row(
            run_id=run.run_id,
            source_node_id=source.node_id,
            row_index=0,
            data={"id": 1, "fail": True},
            source_row_index=0,
            ingest_sequence=0,
        )
        token = factory.data_flow.create_token(row_id=row.row_id)
        ref = TokenRef(token_id=token.token_id, run_id=run.run_id)

        factory.data_flow.record_token_outcome(
            ref,
            TerminalOutcome.FAILURE,
            TerminalPath.ON_ERROR_ROUTED,
            sink_name="error_sink",
            error_hash="0123456789abcdef",
        )

        with pytest.raises(LandscapeRecordError, match="UNIQUE constraint failed"):
            factory.data_flow.record_token_outcome(
                ref,
                TerminalOutcome.SUCCESS,
                TerminalPath.DEFAULT_FLOW,
                sink_name="default",
            )
