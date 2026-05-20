# tests/fixtures/multi_run.py
"""Multi-run landscape fixture for SQL WHERE exactness tests.

Populates 3 runs with deterministic, lexicographically ordered IDs so that
``>=`` / ``<=`` comparisons would return additional rows beyond the intended
exact match.  Every run contains rows, tokens, node states, calls, batches,
and routing events — enough audit surface for WHERE clause mutation tests.

ID ordering scheme (alphabetical sort matches inequality direction):
    run-A < run-B < run-C

Within each run the same alphabetical ordering applies to row_ids, token_ids,
state_ids, etc.  This means a query that accidentally uses ``>=`` instead of
``==`` on run-B will also pick up run-C data, making the mutation detectable.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from elspeth.contracts import (
    BatchStatus,
    CallStatus,
    CallType,
    NodeStateStatus,
    NodeType,
    RoutingMode,
    TerminalOutcome,
    TerminalPath,
    TriggerType,
)
from elspeth.contracts.audit import TokenRef
from elspeth.contracts.call_data import RawCallPayload
from elspeth.contracts.schema import SchemaConfig
from elspeth.core.landscape.database import LandscapeDB
from elspeth.core.landscape.factory import RecorderFactory

_OBSERVED_SCHEMA = SchemaConfig.from_dict({"mode": "observed"})

# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------

RUN_SUFFIXES = ("A", "B", "C")
"""Three ordered suffixes used for deterministic IDs."""


@dataclass(frozen=True, slots=True)
class TokenInfo:
    """All IDs associated with a single token."""

    token_id: str
    row_id: str
    state_id: str
    call_id: str | None = None
    routing_event_id: str | None = None


@dataclass(frozen=True, slots=True)
class RunInfo:
    """All IDs associated with a single run."""

    run_id: str
    source_node_id: str
    transform_node_id: str
    sink_node_id: str
    edge_id_source_to_transform: str
    edge_id_transform_to_sink: str
    row_ids: tuple[str, ...]
    tokens: tuple[TokenInfo, ...]
    batch_id: str


@dataclass(frozen=True, slots=True)
class MultiRunFixture:
    """Structured result from the ``multi_run_landscape`` fixture.

    Provides the factory plus per-run metadata so tests can assert
    exact-match behaviour against any entity type.
    """

    factory: RecorderFactory
    db: LandscapeDB
    runs: tuple[RunInfo, ...]

    def run(self, suffix: str) -> RunInfo:
        """Look up a RunInfo by its suffix letter (A / B / C)."""
        run_id = f"run-{suffix}"
        for r in self.runs:
            if r.run_id == run_id:
                return r
        raise KeyError(f"No run with suffix {suffix!r} (run_id={run_id!r})")


# ---------------------------------------------------------------------------
# Fixture builder
# ---------------------------------------------------------------------------


def _build_multi_run_landscape() -> MultiRunFixture:
    """Create a fully populated multi-run landscape in an in-memory DB.

    Deterministic ID scheme per run suffix ``X`` (A / B / C):
        run_id          = run-X
        source_node_id  = src-X
        transform_node  = xform-X
        sink_node       = sink-X
        row_ids          = row-X-0, row-X-1
        token_ids        = tok-X-0, tok-X-1
        state_ids        = st-X-0, st-X-1
        call              = call-X-0  (on first token only)
        routing_event     = re-X-0    (on first token only)
        batch_id          = batch-X
        edge_ids          = edge-X-s2t, edge-X-t2k
    """

    db = LandscapeDB.in_memory()
    factory = RecorderFactory(db)

    run_infos: list[RunInfo] = []

    for suffix in RUN_SUFFIXES:
        run_id = f"run-{suffix}"
        src_nid = f"src-{suffix}"
        xform_nid = f"xform-{suffix}"
        sink_nid = f"sink-{suffix}"

        # -- run + nodes --
        factory.run_lifecycle.begin_run(config={}, canonical_version="v1", run_id=run_id)

        factory.data_flow.register_node(
            run_id=run_id,
            plugin_name="csv",
            node_type=NodeType.SOURCE,
            plugin_version="1.0",
            config={},
            node_id=src_nid,
            schema_config=_OBSERVED_SCHEMA,
        )
        factory.data_flow.register_node(
            run_id=run_id,
            plugin_name="passthrough",
            node_type=NodeType.TRANSFORM,
            plugin_version="1.0",
            config={},
            node_id=xform_nid,
            schema_config=_OBSERVED_SCHEMA,
        )
        factory.data_flow.register_node(
            run_id=run_id,
            plugin_name="csv_sink",
            node_type=NodeType.SINK,
            plugin_version="1.0",
            config={},
            node_id=sink_nid,
            schema_config=_OBSERVED_SCHEMA,
        )

        # -- edges --
        edge_s2t = f"edge-{suffix}-s2t"
        edge_t2k = f"edge-{suffix}-t2k"
        factory.data_flow.register_edge(run_id, src_nid, xform_nid, "continue", RoutingMode.MOVE, edge_id=edge_s2t)
        factory.data_flow.register_edge(run_id, xform_nid, sink_nid, "continue", RoutingMode.MOVE, edge_id=edge_t2k)

        # -- rows, tokens, states, calls, routing, outcomes --
        row_ids: list[str] = []
        token_infos: list[TokenInfo] = []

        for i in range(2):
            row_id = f"row-{suffix}-{i}"
            tok_id = f"tok-{suffix}-{i}"
            state_id = f"st-{suffix}-{i}"

            factory.data_flow.create_row(run_id, src_nid, i, {"val": f"{suffix}-{i}"}, row_id=row_id, source_row_index=i, ingest_sequence=i)
            factory.data_flow.create_token(row_id, token_id=tok_id)

            # begin + complete a node state on the transform node
            factory.execution.begin_node_state(
                tok_id, xform_nid, run_id, step_index=1, input_data={"val": f"{suffix}-{i}"}, state_id=state_id
            )
            factory.execution.complete_node_state(
                state_id,
                NodeStateStatus.COMPLETED,
                output_data={"val": f"{suffix}-{i}", "processed": True},
                duration_ms=10.0,
            )

            call_id: str | None = None
            re_id: str | None = None

            if i == 0:
                # Record an external call on first token's state
                call_obj = factory.execution.record_call(
                    state_id,
                    call_index=0,
                    call_type=CallType.HTTP,
                    status=CallStatus.SUCCESS,
                    request_data=RawCallPayload({"url": f"https://api.example.com/{suffix}"}),
                    response_data=RawCallPayload({"status": 200}),
                    latency_ms=42.0,
                )
                call_id = call_obj.call_id

                # Record a routing event on first token's state
                re_obj = factory.execution.record_routing_event(
                    state_id,
                    edge_t2k,
                    RoutingMode.MOVE,
                    event_id=f"re-{suffix}-0",
                )
                re_id = re_obj.event_id

            # Record token outcome
            factory.data_flow.record_token_outcome(
                ref=TokenRef(token_id=tok_id, run_id=run_id),
                outcome=TerminalOutcome.SUCCESS,
                path=TerminalPath.DEFAULT_FLOW,
                sink_name="output",
            )

            row_ids.append(row_id)
            token_infos.append(TokenInfo(token_id=tok_id, row_id=row_id, state_id=state_id, call_id=call_id, routing_event_id=re_id))

        # -- batch (on transform node, using aggregation node type conceptually) --
        batch_id = f"batch-{suffix}"
        factory.execution.create_batch(run_id, xform_nid, batch_id=batch_id)
        factory.execution.add_batch_member(batch_id, token_infos[0].token_id, ordinal=0)
        factory.execution.add_batch_member(batch_id, token_infos[1].token_id, ordinal=1)
        factory.execution.update_batch_status(batch_id, BatchStatus.EXECUTING, trigger_type=TriggerType.COUNT, trigger_reason="count=2")

        run_infos.append(
            RunInfo(
                run_id=run_id,
                source_node_id=src_nid,
                transform_node_id=xform_nid,
                sink_node_id=sink_nid,
                edge_id_source_to_transform=edge_s2t,
                edge_id_transform_to_sink=edge_t2k,
                row_ids=tuple(row_ids),
                tokens=tuple(token_infos),
                batch_id=batch_id,
            )
        )

    return MultiRunFixture(factory=factory, db=db, runs=tuple(run_infos))


@pytest.fixture
def multi_run_landscape() -> MultiRunFixture:
    """Function-scoped multi-run landscape fixture.

    Creates 3 runs (run-A, run-B, run-C) each with 2 rows, 2 tokens,
    node states, calls, routing events, batches, and token outcomes.
    All IDs are deterministic and ordered so that inequality operators
    (``>=``, ``<=``) would include data from adjacent runs.
    """
    return _build_multi_run_landscape()
