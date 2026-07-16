from __future__ import annotations

import json
from datetime import UTC, datetime

import pytest

from elspeth.contracts import (
    CallStatus,
    CallType,
    NodeStateStatus,
    NodeType,
    RoutingMode,
    RoutingSpec,
    TerminalOutcome,
    TerminalPath,
)
from elspeth.contracts.audit import TokenRef
from elspeth.contracts.call_data import RawCallPayload
from elspeth.contracts.errors import AuditIntegrityError, CoalesceFailureReason
from elspeth.contracts.payload_store import (
    IntegrityError as PayloadIntegrityError,
)
from elspeth.contracts.payload_store import (
    PayloadNotFoundError,
    PayloadStore,
)
from elspeth.contracts.schema import SchemaConfig
from elspeth.contracts.schema_contract import PipelineRow, SchemaContract
from elspeth.core.landscape import LandscapeDB, QueryRepository
from elspeth.core.landscape._database_ops import DatabaseOps
from elspeth.core.landscape.factory import RecorderFactory
from elspeth.core.landscape.model_loaders import (
    CallLoader,
    NodeStateLoader,
    RoutingEventLoader,
    RowLoader,
    TokenLoader,
    TokenOutcomeLoader,
    TokenParentLoader,
)
from elspeth.core.landscape.row_data import RowDataResult, RowDataState
from elspeth.core.landscape.run_status_projection import AuditRunStatusProjection
from tests.fixtures.landscape import make_factory, make_landscape_db, make_recorder_with_run, register_test_node

_DYNAMIC_SCHEMA = SchemaConfig.from_dict({"mode": "observed"})

# Minimal contract for tests that only care about token lifecycle, not contract content.
_MINIMAL_CONTRACT = SchemaContract(mode="OBSERVED", fields=(), locked=True)


class _PayloadStoreStub:
    """Configurable PayloadStore fake for row payload retrieval tests."""

    def __init__(
        self,
        *,
        store_ref: str = "ref-hash",
        retrieve_result: bytes | Exception = b'{"x": 1}',
    ) -> None:
        self._store_ref = store_ref
        self._retrieve_result = retrieve_result
        self.stored_payloads: list[bytes] = []
        self.retrieved_hashes: list[str] = []
        self.deleted_hashes: list[str] = []

    def store(self, content: bytes) -> str:
        self.stored_payloads.append(content)
        return self._store_ref

    def retrieve(self, content_hash: str) -> bytes:
        self.retrieved_hashes.append(content_hash)
        if isinstance(self._retrieve_result, Exception):
            raise self._retrieve_result
        return self._retrieve_result

    def exists(self, content_hash: str) -> bool:
        return content_hash == self._store_ref

    def delete(self, content_hash: str) -> bool:
        self.deleted_hashes.append(content_hash)
        return content_hash == self._store_ref


def _setup(*, run_id: str = "run-1") -> tuple[LandscapeDB, RecorderFactory]:
    setup = make_recorder_with_run(run_id=run_id, source_node_id="source-0", source_plugin_name="csv")
    db, factory = setup.db, setup.factory
    register_test_node(factory.data_flow, run_id, "transform-1", plugin_name="transform")
    return db, factory


def _setup_full(*, run_id: str = "run-1"):
    """Build a full environment with nodes, edge, row, token, state."""
    db, factory = _setup(run_id=run_id)
    factory.data_flow.register_edge(run_id, "source-0", "transform-1", "continue", RoutingMode.MOVE, edge_id="edge-1")
    factory.data_flow.create_row(run_id, "source-0", 0, {"name": "test"}, row_id="row-1", source_row_index=0, ingest_sequence=0)
    factory.data_flow.create_token("row-1", token_id="tok-1")
    factory.execution.begin_node_state("tok-1", "transform-1", run_id, 0, {"name": "test"}, state_id="state-1")
    return db, factory


class TestQueryRepositoryCapabilityBoundary:
    """Pin the factory wiring for the read-side capability split."""

    def test_factory_injects_read_only_ops_into_query_repository(self) -> None:
        db = make_landscape_db()
        factory = make_factory(db)

        assert not hasattr(factory.query._ops, "execute_insert")
        assert not hasattr(factory.query._ops, "execute_update")

    def test_factory_exposes_run_status_projection_outside_query_repository(self) -> None:
        db = make_landscape_db()
        factory = make_factory(db)

        assert isinstance(factory.run_status_projection, AuditRunStatusProjection)
        assert not hasattr(factory.query, "count_distinct_source_rows_with_terminal_outcome")
        assert not hasattr(factory.query, "count_failed_coalesce_barrier_rows")


class TestGetRows:
    """Tests for RecorderFactory query — retrieves rows for a run ordered by row_index."""

    def test_returns_rows_ordered_by_index(self):
        _db, factory = _setup()
        factory.data_flow.create_row("run-1", "source-0", 2, {"c": 3}, row_id="row-c", source_row_index=2, ingest_sequence=2)
        factory.data_flow.create_row("run-1", "source-0", 0, {"a": 1}, row_id="row-a", source_row_index=0, ingest_sequence=0)
        factory.data_flow.create_row("run-1", "source-0", 1, {"b": 2}, row_id="row-b", source_row_index=1, ingest_sequence=1)

        rows = factory.query.get_rows("run-1")

        assert len(rows) == 3
        assert [r.row_id for r in rows] == ["row-a", "row-b", "row-c"]
        assert [r.row_index for r in rows] == [0, 1, 2]

    def test_exposes_source_local_and_ingest_row_identity(self) -> None:
        """Read-side row contracts must distinguish source-local order from ingest order."""
        _db, factory = _setup()
        register_test_node(factory.data_flow, "run-1", "source-1", node_type=NodeType.SOURCE, plugin_name="csv")
        factory.data_flow.create_row(
            "run-1",
            "source-0",
            50,
            {"order_id": "o-1"},
            row_id="row-orders",
            source_row_index=0,
            ingest_sequence=1,
        )
        factory.data_flow.create_row(
            "run-1",
            "source-1",
            99,
            {"refund_id": "r-1"},
            row_id="row-refunds",
            source_row_index=0,
            ingest_sequence=0,
        )

        rows = factory.query.get_rows("run-1")

        assert [r.row_id for r in rows] == ["row-refunds", "row-orders"]
        assert [r.source_node_id for r in rows] == ["source-1", "source-0"]
        assert [r.source_row_index for r in rows] == [0, 0]
        assert [r.ingest_sequence for r in rows] == [0, 1]

    def test_empty_for_unknown_run(self):
        _, factory = _setup()

        rows = factory.query.get_rows("nonexistent-run")

        assert rows == []

    def test_single_row(self):
        _, factory = _setup()
        factory.data_flow.create_row("run-1", "source-0", 0, {"x": 1}, row_id="row-1", source_row_index=0, ingest_sequence=0)

        rows = factory.query.get_rows("run-1")

        assert len(rows) == 1
        assert rows[0].row_id == "row-1"
        assert rows[0].row_index == 0

    def test_rows_scoped_to_run(self):
        db = make_landscape_db()
        factory = make_factory(db)
        factory.run_lifecycle.begin_run(config={}, canonical_version="v1", run_id="run-a")
        factory.data_flow.register_node(
            run_id="run-a",
            plugin_name="csv",
            node_type=NodeType.SOURCE,
            plugin_version="1.0",
            config={},
            node_id="src-a",
            schema_config=_DYNAMIC_SCHEMA,
        )
        factory.run_lifecycle.begin_run(config={}, canonical_version="v1", run_id="run-b")
        factory.data_flow.register_node(
            run_id="run-b",
            plugin_name="csv",
            node_type=NodeType.SOURCE,
            plugin_version="1.0",
            config={},
            node_id="src-b",
            schema_config=_DYNAMIC_SCHEMA,
        )
        factory.data_flow.create_row("run-a", "src-a", 0, {"v": 1}, row_id="row-a1", source_row_index=0, ingest_sequence=0)
        factory.data_flow.create_row("run-b", "src-b", 0, {"v": 2}, row_id="row-b1", source_row_index=0, ingest_sequence=0)

        rows_a = factory.query.get_rows("run-a")
        rows_b = factory.query.get_rows("run-b")

        assert len(rows_a) == 1
        assert rows_a[0].row_id == "row-a1"
        assert len(rows_b) == 1
        assert rows_b[0].row_id == "row-b1"


class TestGetTokens:
    """Tests for RecorderFactory query — retrieves tokens for a row."""

    def test_returns_tokens_for_row(self):
        _, factory = _setup()
        factory.data_flow.create_row("run-1", "source-0", 0, {"x": 1}, row_id="row-1", source_row_index=0, ingest_sequence=0)
        factory.data_flow.create_token("row-1", token_id="tok-1")
        factory.data_flow.create_token("row-1", token_id="tok-2")

        tokens = factory.query.get_tokens("row-1")

        assert len(tokens) == 2
        token_ids = {t.token_id for t in tokens}
        assert token_ids == {"tok-1", "tok-2"}

    def test_empty_for_unknown_row(self):
        _, factory = _setup()

        tokens = factory.query.get_tokens("nonexistent-row")

        assert tokens == []

    def test_single_token(self):
        _, factory = _setup()
        factory.data_flow.create_row("run-1", "source-0", 0, {"x": 1}, row_id="row-1", source_row_index=0, ingest_sequence=0)
        factory.data_flow.create_token("row-1", token_id="tok-1")

        tokens = factory.query.get_tokens("row-1")

        assert len(tokens) == 1
        assert tokens[0].token_id == "tok-1"
        assert tokens[0].row_id == "row-1"

    def test_tokens_scoped_to_row(self):
        _, factory = _setup()
        factory.data_flow.create_row("run-1", "source-0", 0, {"a": 1}, row_id="row-a", source_row_index=0, ingest_sequence=0)
        factory.data_flow.create_row("run-1", "source-0", 1, {"b": 2}, row_id="row-b", source_row_index=1, ingest_sequence=1)
        factory.data_flow.create_token("row-a", token_id="tok-a")
        factory.data_flow.create_token("row-b", token_id="tok-b")

        tokens_a = factory.query.get_tokens("row-a")
        tokens_b = factory.query.get_tokens("row-b")

        assert len(tokens_a) == 1
        assert tokens_a[0].token_id == "tok-a"
        assert len(tokens_b) == 1
        assert tokens_b[0].token_id == "tok-b"


class TestGetNodeStatesForToken:
    """Tests for RecorderFactory query — states ordered by (step_index, attempt)."""

    def test_returns_states_ordered_by_step_index(self):
        _, factory = _setup_full()
        # state-1 already exists at step_index=0
        factory.data_flow.register_node(
            run_id="run-1",
            plugin_name="transform2",
            node_type=NodeType.TRANSFORM,
            plugin_version="1.0",
            config={},
            node_id="transform-2",
            schema_config=_DYNAMIC_SCHEMA,
        )
        factory.execution.begin_node_state("tok-1", "transform-2", "run-1", 1, {"name": "test"}, state_id="state-2")

        states = factory.query.get_node_states_for_token("tok-1")

        assert len(states) == 2
        assert states[0].state_id == "state-1"
        assert states[0].step_index == 0
        assert states[1].state_id == "state-2"
        assert states[1].step_index == 1

    def test_orders_by_attempt_within_step(self):
        _, factory = _setup_full()
        # state-1 is step_index=0, attempt=0
        # Add a retry at the same step
        factory.execution.begin_node_state(
            "tok-1",
            "transform-1",
            "run-1",
            0,
            {"name": "test"},
            state_id="state-retry",
            attempt=1,
        )

        states = factory.query.get_node_states_for_token("tok-1")

        assert len(states) == 2
        assert states[0].state_id == "state-1"
        assert states[0].attempt == 0
        assert states[1].state_id == "state-retry"
        assert states[1].attempt == 1

    def test_empty_for_unknown_token(self):
        _, factory = _setup()

        states = factory.query.get_node_states_for_token("nonexistent-tok")

        assert states == []

    def test_single_state(self):
        _, factory = _setup_full()

        states = factory.query.get_node_states_for_token("tok-1")

        assert len(states) == 1
        assert states[0].state_id == "state-1"
        assert states[0].token_id == "tok-1"
        assert states[0].node_id == "transform-1"


class TestGetRow:
    """Tests for RecorderFactory query — retrieves a single row by ID."""

    def test_roundtrip(self):
        _, factory = _setup()
        factory.data_flow.create_row("run-1", "source-0", 0, {"field": "value"}, row_id="row-1", source_row_index=0, ingest_sequence=0)

        row = factory.query.get_row("row-1")

        assert row is not None
        assert row.row_id == "row-1"
        assert row.run_id == "run-1"
        assert row.source_node_id == "source-0"
        assert row.row_index == 0

    def test_none_for_unknown(self):
        _, factory = _setup()

        row = factory.query.get_row("nonexistent-row")

        assert row is None


class TestGetRowData:
    """Tests for RecorderFactory query — retrieves payload data with state information."""

    def test_row_not_found(self):
        _, factory = _setup()

        result = factory.query.get_row_data("nonexistent-row")

        assert isinstance(result, RowDataResult)
        assert result.state == RowDataState.ROW_NOT_FOUND
        assert result.data is None

    def test_never_stored_when_no_payload_ref(self):
        """Row without source_data_ref → NEVER_STORED."""
        # Explicitly create factory with NO payload store — create_row will not store payload
        # (Note: make_factory defaults to MockPayloadStore to support expand/coalesce;
        # this test needs a storeless factory to verify the NEVER_STORED code path.)
        db = make_landscape_db()
        factory = RecorderFactory(db)  # No payload store
        factory.run_lifecycle.begin_run(config={}, canonical_version="v1", run_id="run-1")
        factory.data_flow.register_node(
            run_id="run-1",
            plugin_name="csv",
            node_type=NodeType.SOURCE,
            plugin_version="1.0",
            config={},
            node_id="source-0",
            schema_config=_DYNAMIC_SCHEMA,
        )
        factory.data_flow.create_row("run-1", "source-0", 0, {"x": 1}, row_id="row-1", source_row_index=0, ingest_sequence=0)

        # With no store, create_row produces no source_data_ref
        row = factory.query.get_row("row-1")
        assert row is not None
        assert row.source_data_ref is None, "Factory with no payload store must not set source_data_ref"
        result = factory.query.get_row_data("row-1")
        assert result.state == RowDataState.NEVER_STORED
        assert result.data is None

    def test_store_not_configured_when_ref_exists_but_no_store(self):
        """Row has source_data_ref but QueryRepository has no payload_store → STORE_NOT_CONFIGURED."""
        db = make_landscape_db()
        # Create a factory WITH a payload store so the row gets a ref
        payload_store = _PayloadStoreStub(store_ref="abc123")
        factory = RecorderFactory(db, payload_store=payload_store)
        factory.run_lifecycle.begin_run(config={}, canonical_version="v1", run_id="run-1")
        factory.data_flow.register_node(
            run_id="run-1",
            plugin_name="csv",
            node_type=NodeType.SOURCE,
            plugin_version="1.0",
            config={},
            node_id="source-0",
            schema_config=_DYNAMIC_SCHEMA,
        )
        factory.data_flow.create_row("run-1", "source-0", 0, {"x": 1}, row_id="row-1", source_row_index=0, ingest_sequence=0)

        # Query through a repo WITHOUT a payload store
        ops = DatabaseOps(db)
        repo = QueryRepository(
            ops,
            row_loader=RowLoader(),
            token_loader=TokenLoader(),
            token_parent_loader=TokenParentLoader(),
            node_state_loader=NodeStateLoader(),
            routing_event_loader=RoutingEventLoader(),
            call_loader=CallLoader(),
            token_outcome_loader=TokenOutcomeLoader(),
            payload_store=None,
        )
        result = repo.get_row_data("row-1")
        assert result.state == RowDataState.STORE_NOT_CONFIGURED
        assert result.data is None


class TestGetRowDataReprFallback:
    """Tests for get_row_data REPR_FALLBACK detection on read-back.

    The write path stores {"_repr": repr(data)} for quarantined rows that
    can't be canonically serialized. The read path must detect this sentinel
    and return REPR_FALLBACK state instead of AVAILABLE.
    """

    def test_repr_fallback_detected_on_readback(self, tmp_path):
        """Payload containing only the _repr sentinel returns REPR_FALLBACK."""
        import json

        from elspeth.core.payload_store import FilesystemPayloadStore

        db = make_landscape_db()
        store = FilesystemPayloadStore(tmp_path / "payloads")

        # Store a repr-fallback payload directly
        sentinel_payload = json.dumps({"_repr": "repr(some_unparseable_data)"}).encode("utf-8")
        ref = store.store(sentinel_payload)

        factory = RecorderFactory(db, payload_store=store)
        factory.run_lifecycle.begin_run(config={}, canonical_version="v1", run_id="run-1")
        register_test_node(factory.data_flow, "run-1", "source-0", node_type=NodeType.SOURCE, plugin_name="csv")

        # Create the row, then patch the source_data_ref to point to our sentinel
        factory.data_flow.create_row("run-1", "source-0", 0, {"x": 1}, row_id="row-1", source_row_index=0, ingest_sequence=0)

        from sqlalchemy import update

        from elspeth.core.landscape.schema import rows_table

        with db.connection() as conn:
            conn.execute(update(rows_table).where(rows_table.c.row_id == "row-1").values(source_data_ref=ref))

        ops = DatabaseOps(db)
        repo = QueryRepository(
            ops,
            row_loader=RowLoader(),
            token_loader=TokenLoader(),
            token_parent_loader=TokenParentLoader(),
            node_state_loader=NodeStateLoader(),
            routing_event_loader=RoutingEventLoader(),
            call_loader=CallLoader(),
            token_outcome_loader=TokenOutcomeLoader(),
            payload_store=store,
        )

        result = repo.get_row_data("row-1")

        assert result.state == RowDataState.REPR_FALLBACK
        assert result.data is not None
        assert result.data["_repr"] == "repr(some_unparseable_data)"

    def test_dict_with_repr_plus_other_keys_is_available(self, tmp_path):
        """A dict containing _repr along with other keys is not a sentinel — returns AVAILABLE."""
        import json

        from elspeth.core.payload_store import FilesystemPayloadStore

        db = make_landscape_db()
        store = FilesystemPayloadStore(tmp_path / "payloads")

        not_sentinel = json.dumps({"_repr": "something", "other_key": "value"}).encode("utf-8")
        ref = store.store(not_sentinel)

        factory = RecorderFactory(db, payload_store=store)
        factory.run_lifecycle.begin_run(config={}, canonical_version="v1", run_id="run-1")
        register_test_node(factory.data_flow, "run-1", "source-0", node_type=NodeType.SOURCE, plugin_name="csv")
        factory.data_flow.create_row("run-1", "source-0", 0, {"x": 1}, row_id="row-1", source_row_index=0, ingest_sequence=0)

        from sqlalchemy import update

        from elspeth.core.landscape.schema import rows_table

        with db.connection() as conn:
            conn.execute(update(rows_table).where(rows_table.c.row_id == "row-1").values(source_data_ref=ref))

        ops = DatabaseOps(db)
        repo = QueryRepository(
            ops,
            row_loader=RowLoader(),
            token_loader=TokenLoader(),
            token_parent_loader=TokenParentLoader(),
            node_state_loader=NodeStateLoader(),
            routing_event_loader=RoutingEventLoader(),
            call_loader=CallLoader(),
            token_outcome_loader=TokenOutcomeLoader(),
            payload_store=store,
        )

        result = repo.get_row_data("row-1")

        assert result.state == RowDataState.AVAILABLE


class TestGetToken:
    """Tests for RecorderFactory query — retrieves a single token by ID."""

    def test_roundtrip(self):
        _, factory = _setup()
        factory.data_flow.create_row("run-1", "source-0", 0, {"x": 1}, row_id="row-1", source_row_index=0, ingest_sequence=0)
        factory.data_flow.create_token("row-1", token_id="tok-1")

        token = factory.query.get_token("tok-1")

        assert token is not None
        assert token.token_id == "tok-1"
        assert token.row_id == "row-1"

    def test_none_for_unknown(self):
        _, factory = _setup()

        token = factory.query.get_token("nonexistent-tok")

        assert token is None

    def test_get_token_for_run_hides_foreign_run_token(self):
        _, factory = _setup(run_id="run-a")
        factory.run_lifecycle.begin_run(config={}, canonical_version="v1", run_id="run-b")
        factory.data_flow.register_node(
            run_id="run-b",
            plugin_name="csv",
            node_type=NodeType.SOURCE,
            plugin_version="1.0",
            config={},
            node_id="src-b",
            schema_config=_DYNAMIC_SCHEMA,
        )
        factory.data_flow.create_row("run-b", "src-b", 0, {"v": 2}, row_id="row-b1", source_row_index=0, ingest_sequence=0)
        factory.data_flow.create_token("row-b1", token_id="tok-b1")

        unscoped_token = factory.query.get_token("tok-b1")
        scoped_miss = factory.query.get_token_for_run("run-a", "tok-b1")
        scoped_hit = factory.query.get_token_for_run("run-b", "tok-b1")

        assert unscoped_token is not None
        assert unscoped_token.run_id == "run-b"
        assert scoped_miss is None
        assert scoped_hit is not None
        assert scoped_hit.token_id == "tok-b1"


class TestGetTokensByIds:
    """Set-scoped token-id reads preserve caller order for lineage hydration."""

    def _setup_tokens(self):
        _db, factory = _setup()
        factory.data_flow.create_row("run-1", "source-0", 0, {"a": 1}, row_id="row-a", source_row_index=0, ingest_sequence=0)
        factory.data_flow.create_row("run-1", "source-0", 1, {"b": 2}, row_id="row-b", source_row_index=1, ingest_sequence=1)
        factory.data_flow.create_token("row-a", token_id="tok-a")
        factory.data_flow.create_token("row-b", token_id="tok-b")
        return factory

    def test_preserves_input_order_and_ignores_missing_tokens(self):
        factory = self._setup_tokens()

        tokens = factory.query.get_tokens_by_ids(["tok-b", "missing", "tok-a"])

        assert [token.token_id for token in tokens] == ["tok-b", "tok-a"]

    def test_empty_input_returns_empty(self):
        factory = self._setup_tokens()

        assert factory.query.get_tokens_by_ids([]) == []

    def test_chunked_input_matches_unchunked(self):
        factory = self._setup_tokens()
        unchunked = factory.query.get_tokens_by_ids(["tok-b", "tok-a"])

        factory.query._QUERY_CHUNK_SIZE = 1

        chunked = factory.query.get_tokens_by_ids(["tok-b", "tok-a"])
        assert [token.token_id for token in chunked] == [token.token_id for token in unchunked]


class TestGetTokenParents:
    """Tests for RecorderFactory query — parent relationships ordered by ordinal."""

    def test_empty_when_no_parents(self):
        _, factory = _setup()
        factory.data_flow.create_row("run-1", "source-0", 0, {"x": 1}, row_id="row-1", source_row_index=0, ingest_sequence=0)
        factory.data_flow.create_token("row-1", token_id="tok-1")

        parents = factory.query.get_token_parents("tok-1")

        assert parents == []

    def test_returns_parents_after_fork(self):
        _, factory = _setup_full()
        # fork_token creates children with parent relationships
        children, _fork_group_id = factory.data_flow.fork_token(
            parent_ref=TokenRef(token_id="tok-1", run_id="run-1"),
            row_id="row-1",
            branches=["path-a", "path-b"],
        )

        # Each child should have tok-1 as parent
        for child in children:
            parents = factory.query.get_token_parents(child.token_id)
            assert len(parents) == 1
            assert parents[0].parent_token_id == "tok-1"

    def test_returns_parents_after_coalesce(self):
        _, factory = _setup_full()
        # Create a second token to coalesce with
        factory.data_flow.create_token("row-1", token_id="tok-2")

        merged = factory.data_flow.coalesce_tokens(
            parent_refs=[TokenRef(token_id="tok-1", run_id="run-1"), TokenRef(token_id="tok-2", run_id="run-1")],
            row_id="row-1",
            merged_payload={"merged": True},
            merged_contract=_MINIMAL_CONTRACT,
        )

        parents = factory.query.get_token_parents(merged.token_id)

        assert len(parents) == 2
        parent_ids = [p.parent_token_id for p in parents]
        assert "tok-1" in parent_ids
        assert "tok-2" in parent_ids
        # Ordered by ordinal
        assert parents[0].ordinal == 0
        assert parents[1].ordinal == 1

    def test_empty_for_unknown_token(self):
        _, factory = _setup()

        parents = factory.query.get_token_parents("nonexistent-tok")

        assert parents == []


class TestGetRoutingEvents:
    """Tests for RecorderFactory query — events for a state."""

    def test_returns_events_for_state(self):
        _, factory = _setup_full()
        factory.execution.record_routing_event(
            state_id="state-1",
            edge_id="edge-1",
            mode=RoutingMode.MOVE,
        )

        events = factory.query.get_routing_events("state-1")

        assert len(events) == 1
        assert events[0].state_id == "state-1"
        assert events[0].edge_id == "edge-1"

    def test_events_ordered_by_ordinal(self):
        _, factory = _setup_full()
        # Register additional infrastructure for second event
        factory.data_flow.register_node(
            run_id="run-1",
            plugin_name="sink",
            node_type=NodeType.SINK,
            plugin_version="1.0",
            config={},
            node_id="sink-0",
            schema_config=_DYNAMIC_SCHEMA,
        )
        factory.data_flow.register_edge("run-1", "transform-1", "sink-0", "route_to_sink", RoutingMode.MOVE, edge_id="edge-2")
        factory.execution.record_routing_events(
            "state-1",
            [
                RoutingSpec(edge_id="edge-2", mode=RoutingMode.MOVE),
                RoutingSpec(edge_id="edge-1", mode=RoutingMode.MOVE),
            ],
        )

        events = factory.query.get_routing_events("state-1")

        assert len(events) == 2
        assert events[0].ordinal == 0
        assert events[1].ordinal == 1

    def test_empty_for_unknown_state(self):
        _, factory = _setup()

        events = factory.query.get_routing_events("nonexistent-state")

        assert events == []


class TestGetCalls:
    """Tests for RecorderFactory query — calls for a state ordered by call_index."""

    def test_returns_calls_for_state(self):
        _, factory = _setup_full()
        factory.execution.record_call(
            state_id="state-1",
            call_index=0,
            call_type=CallType.LLM,
            status=CallStatus.SUCCESS,
            request_data=RawCallPayload({"model": "gpt-4", "prompt": "Hello"}),
            response_data=RawCallPayload({"completion": "Hi"}),
            latency_ms=100.0,
        )

        calls = factory.query.get_calls("state-1")

        assert len(calls) == 1
        assert calls[0].state_id == "state-1"
        assert calls[0].call_index == 0
        assert calls[0].status == CallStatus.SUCCESS

    def test_calls_ordered_by_call_index(self):
        _, factory = _setup_full()
        factory.execution.record_call(
            state_id="state-1",
            call_index=1,
            call_type=CallType.LLM,
            status=CallStatus.SUCCESS,
            request_data=RawCallPayload({"prompt": "second"}),
            response_data=RawCallPayload({"out": "b"}),
            latency_ms=50.0,
        )
        factory.execution.record_call(
            state_id="state-1",
            call_index=0,
            call_type=CallType.HTTP,
            status=CallStatus.SUCCESS,
            request_data=RawCallPayload({"url": "https://example.com"}),
            response_data=RawCallPayload({"body": "ok"}),
            latency_ms=75.0,
        )

        calls = factory.query.get_calls("state-1")

        assert len(calls) == 2
        assert calls[0].call_index == 0
        assert calls[1].call_index == 1

    def test_empty_for_unknown_state(self):
        _, factory = _setup()

        calls = factory.query.get_calls("nonexistent-state")

        assert calls == []


class TestGetRoutingEventsForStates:
    """Tests for RecorderFactory query — batch query for multiple state IDs."""

    def test_batch_query_returns_events(self):
        _, factory = _setup_full()
        # Create a second state
        factory.data_flow.create_row("run-1", "source-0", 1, {"name": "test2"}, row_id="row-2", source_row_index=1, ingest_sequence=1)
        factory.data_flow.create_token("row-2", token_id="tok-2")
        factory.execution.begin_node_state("tok-2", "transform-1", "run-1", 0, {"name": "test2"}, state_id="state-2")
        factory.execution.record_routing_event(
            state_id="state-1",
            edge_id="edge-1",
            mode=RoutingMode.MOVE,
        )
        factory.execution.record_routing_event(
            state_id="state-2",
            edge_id="edge-1",
            mode=RoutingMode.MOVE,
        )

        events = factory.query.get_routing_events_for_states(["state-1", "state-2"])

        assert len(events) == 2
        state_ids = {e.state_id for e in events}
        assert state_ids == {"state-1", "state-2"}

    def test_empty_input_returns_empty(self):
        _, factory = _setup()

        events = factory.query.get_routing_events_for_states([])

        assert events == []

    def test_single_state_id(self):
        _, factory = _setup_full()
        factory.execution.record_routing_event(
            state_id="state-1",
            edge_id="edge-1",
            mode=RoutingMode.MOVE,
        )

        events = factory.query.get_routing_events_for_states(["state-1"])

        assert len(events) == 1
        assert events[0].state_id == "state-1"


class TestGetCallsForStates:
    """Tests for RecorderFactory query — batch query for multiple state IDs."""

    def test_batch_query_returns_calls(self):
        _, factory = _setup_full()
        factory.data_flow.create_row("run-1", "source-0", 1, {"name": "test2"}, row_id="row-2", source_row_index=1, ingest_sequence=1)
        factory.data_flow.create_token("row-2", token_id="tok-2")
        factory.execution.begin_node_state("tok-2", "transform-1", "run-1", 0, {"name": "test2"}, state_id="state-2")
        factory.execution.record_call(
            state_id="state-1",
            call_index=0,
            call_type=CallType.LLM,
            status=CallStatus.SUCCESS,
            request_data=RawCallPayload({"prompt": "a"}),
            response_data=RawCallPayload({"out": "x"}),
            latency_ms=50.0,
        )
        factory.execution.record_call(
            state_id="state-2",
            call_index=0,
            call_type=CallType.HTTP,
            status=CallStatus.SUCCESS,
            request_data=RawCallPayload({"url": "https://example.com"}),
            response_data=RawCallPayload({"body": "ok"}),
            latency_ms=75.0,
        )

        calls = factory.query.get_calls_for_states(["state-1", "state-2"])

        assert len(calls) == 2
        state_ids = {c.state_id for c in calls}
        assert state_ids == {"state-1", "state-2"}

    def test_empty_input_returns_empty(self):
        _, factory = _setup()

        calls = factory.query.get_calls_for_states([])

        assert calls == []

    def test_single_state_id(self):
        _, factory = _setup_full()
        factory.execution.record_call(
            state_id="state-1",
            call_index=0,
            call_type=CallType.LLM,
            status=CallStatus.SUCCESS,
            request_data=RawCallPayload({"prompt": "test"}),
            response_data=RawCallPayload({"out": "ok"}),
            latency_ms=100.0,
        )

        calls = factory.query.get_calls_for_states(["state-1"])

        assert len(calls) == 1
        assert calls[0].state_id == "state-1"


class TestGetAllTokensForRun:
    """Tests for RecorderFactory query — all tokens across rows via JOIN."""

    def test_returns_all_tokens_across_rows(self):
        _, factory = _setup()
        factory.data_flow.create_row("run-1", "source-0", 0, {"a": 1}, row_id="row-1", source_row_index=0, ingest_sequence=0)
        factory.data_flow.create_row("run-1", "source-0", 1, {"b": 2}, row_id="row-2", source_row_index=1, ingest_sequence=1)
        factory.data_flow.create_token("row-1", token_id="tok-1")
        factory.data_flow.create_token("row-1", token_id="tok-2")
        factory.data_flow.create_token("row-2", token_id="tok-3")

        tokens = factory.query.get_all_tokens_for_run("run-1")

        assert len(tokens) == 3
        token_ids = {t.token_id for t in tokens}
        assert token_ids == {"tok-1", "tok-2", "tok-3"}

    def test_empty_for_unknown_run(self):
        _, factory = _setup()

        tokens = factory.query.get_all_tokens_for_run("nonexistent-run")

        assert tokens == []

    def test_scoped_to_run(self):
        db = make_landscape_db()
        factory = make_factory(db)
        factory.run_lifecycle.begin_run(config={}, canonical_version="v1", run_id="run-a")
        factory.data_flow.register_node(
            run_id="run-a",
            plugin_name="csv",
            node_type=NodeType.SOURCE,
            plugin_version="1.0",
            config={},
            node_id="src-a",
            schema_config=_DYNAMIC_SCHEMA,
        )
        factory.run_lifecycle.begin_run(config={}, canonical_version="v1", run_id="run-b")
        factory.data_flow.register_node(
            run_id="run-b",
            plugin_name="csv",
            node_type=NodeType.SOURCE,
            plugin_version="1.0",
            config={},
            node_id="src-b",
            schema_config=_DYNAMIC_SCHEMA,
        )
        factory.data_flow.create_row("run-a", "src-a", 0, {"v": 1}, row_id="row-a1", source_row_index=0, ingest_sequence=0)
        factory.data_flow.create_token("row-a1", token_id="tok-a1")
        factory.data_flow.create_row("run-b", "src-b", 0, {"v": 2}, row_id="row-b1", source_row_index=0, ingest_sequence=0)
        factory.data_flow.create_token("row-b1", token_id="tok-b1")

        tokens_a = factory.query.get_all_tokens_for_run("run-a")
        tokens_b = factory.query.get_all_tokens_for_run("run-b")

        assert len(tokens_a) == 1
        assert tokens_a[0].token_id == "tok-a1"
        assert len(tokens_b) == 1
        assert tokens_b[0].token_id == "tok-b1"

    def test_scoped_to_run_same_node_id(self):
        """Cross-run isolation with SAME node_id reused across runs.

        The existing test_scoped_to_run uses different node_ids per run,
        which doesn't exercise the composite PK concern: same node_id +
        different run_id must still isolate correctly.
        """
        db = make_landscape_db()
        factory = make_factory(db)
        factory.run_lifecycle.begin_run(config={}, canonical_version="v1", run_id="run-a")
        factory.data_flow.register_node(
            run_id="run-a",
            plugin_name="csv",
            node_type=NodeType.SOURCE,
            plugin_version="1.0",
            config={},
            node_id="shared-source",
            schema_config=_DYNAMIC_SCHEMA,
        )
        factory.run_lifecycle.begin_run(config={}, canonical_version="v1", run_id="run-b")
        factory.data_flow.register_node(
            run_id="run-b",
            plugin_name="csv",
            node_type=NodeType.SOURCE,
            plugin_version="1.0",
            config={},
            node_id="shared-source",
            schema_config=_DYNAMIC_SCHEMA,
        )
        factory.data_flow.create_row("run-a", "shared-source", 0, {"v": 1}, row_id="row-a1", source_row_index=0, ingest_sequence=0)
        factory.data_flow.create_token("row-a1", token_id="tok-a1")
        factory.data_flow.create_row("run-b", "shared-source", 0, {"v": 2}, row_id="row-b1", source_row_index=0, ingest_sequence=0)
        factory.data_flow.create_token("row-b1", token_id="tok-b1")

        tokens_a = factory.query.get_all_tokens_for_run("run-a")
        tokens_b = factory.query.get_all_tokens_for_run("run-b")

        assert len(tokens_a) == 1
        assert tokens_a[0].token_id == "tok-a1"
        assert len(tokens_b) == 1
        assert tokens_b[0].token_id == "tok-b1"


class TestGetAllNodeStatesForRun:
    """Tests for RecorderFactory query — uses denormalized run_id."""

    def test_returns_all_states(self):
        _, factory = _setup_full()
        # state-1 already exists
        factory.data_flow.create_row("run-1", "source-0", 1, {"b": 2}, row_id="row-2", source_row_index=1, ingest_sequence=1)
        factory.data_flow.create_token("row-2", token_id="tok-2")
        factory.execution.begin_node_state("tok-2", "transform-1", "run-1", 0, {"b": 2}, state_id="state-2")

        states = factory.query.get_all_node_states_for_run("run-1")

        assert len(states) == 2
        state_ids = {s.state_id for s in states}
        assert state_ids == {"state-1", "state-2"}

    def test_empty_for_unknown_run(self):
        _, factory = _setup()

        states = factory.query.get_all_node_states_for_run("nonexistent-run")

        assert states == []

    def test_scoped_to_run(self):
        db = make_landscape_db()
        factory = make_factory(db)

        factory.run_lifecycle.begin_run(config={}, canonical_version="v1", run_id="run-a")
        factory.data_flow.register_node(
            run_id="run-a",
            plugin_name="csv",
            node_type=NodeType.SOURCE,
            plugin_version="1.0",
            config={},
            node_id="src-a",
            schema_config=_DYNAMIC_SCHEMA,
        )
        factory.data_flow.register_node(
            run_id="run-a",
            plugin_name="tx",
            node_type=NodeType.TRANSFORM,
            plugin_version="1.0",
            config={},
            node_id="tx-a",
            schema_config=_DYNAMIC_SCHEMA,
        )
        factory.data_flow.create_row("run-a", "src-a", 0, {"v": 1}, row_id="row-a1", source_row_index=0, ingest_sequence=0)
        factory.data_flow.create_token("row-a1", token_id="tok-a1")
        factory.execution.begin_node_state("tok-a1", "tx-a", "run-a", 0, {"v": 1}, state_id="state-a1")

        factory.run_lifecycle.begin_run(config={}, canonical_version="v1", run_id="run-b")
        factory.data_flow.register_node(
            run_id="run-b",
            plugin_name="csv",
            node_type=NodeType.SOURCE,
            plugin_version="1.0",
            config={},
            node_id="src-b",
            schema_config=_DYNAMIC_SCHEMA,
        )
        factory.data_flow.register_node(
            run_id="run-b",
            plugin_name="tx",
            node_type=NodeType.TRANSFORM,
            plugin_version="1.0",
            config={},
            node_id="tx-b",
            schema_config=_DYNAMIC_SCHEMA,
        )
        factory.data_flow.create_row("run-b", "src-b", 0, {"v": 2}, row_id="row-b1", source_row_index=0, ingest_sequence=0)
        factory.data_flow.create_token("row-b1", token_id="tok-b1")
        factory.execution.begin_node_state("tok-b1", "tx-b", "run-b", 0, {"v": 2}, state_id="state-b1")

        states_a = factory.query.get_all_node_states_for_run("run-a")
        states_b = factory.query.get_all_node_states_for_run("run-b")

        assert len(states_a) == 1
        assert states_a[0].state_id == "state-a1"
        assert len(states_b) == 1
        assert states_b[0].state_id == "state-b1"

    def test_scoped_to_run_same_node_id(self):
        """Cross-run isolation with SAME node_id reused across runs.

        Exercises the composite PK (run_id, node_id) on node_states:
        two runs sharing a node_id must isolate their states.
        """
        db = make_landscape_db()
        factory = make_factory(db)

        factory.run_lifecycle.begin_run(config={}, canonical_version="v1", run_id="run-a")
        factory.data_flow.register_node(
            run_id="run-a",
            plugin_name="csv",
            node_type=NodeType.SOURCE,
            plugin_version="1.0",
            config={},
            node_id="shared-source",
            schema_config=_DYNAMIC_SCHEMA,
        )
        factory.data_flow.register_node(
            run_id="run-a",
            plugin_name="tx",
            node_type=NodeType.TRANSFORM,
            plugin_version="1.0",
            config={},
            node_id="shared-tx",
            schema_config=_DYNAMIC_SCHEMA,
        )
        factory.data_flow.create_row("run-a", "shared-source", 0, {"v": 1}, row_id="row-a1", source_row_index=0, ingest_sequence=0)
        factory.data_flow.create_token("row-a1", token_id="tok-a1")
        factory.execution.begin_node_state("tok-a1", "shared-tx", "run-a", 0, {"v": 1}, state_id="state-a1")

        factory.run_lifecycle.begin_run(config={}, canonical_version="v1", run_id="run-b")
        factory.data_flow.register_node(
            run_id="run-b",
            plugin_name="csv",
            node_type=NodeType.SOURCE,
            plugin_version="1.0",
            config={},
            node_id="shared-source",
            schema_config=_DYNAMIC_SCHEMA,
        )
        factory.data_flow.register_node(
            run_id="run-b",
            plugin_name="tx",
            node_type=NodeType.TRANSFORM,
            plugin_version="1.0",
            config={},
            node_id="shared-tx",
            schema_config=_DYNAMIC_SCHEMA,
        )
        factory.data_flow.create_row("run-b", "shared-source", 0, {"v": 2}, row_id="row-b1", source_row_index=0, ingest_sequence=0)
        factory.data_flow.create_token("row-b1", token_id="tok-b1")
        factory.execution.begin_node_state("tok-b1", "shared-tx", "run-b", 0, {"v": 2}, state_id="state-b1")

        states_a = factory.query.get_all_node_states_for_run("run-a")
        states_b = factory.query.get_all_node_states_for_run("run-b")

        assert len(states_a) == 1
        assert states_a[0].state_id == "state-a1"
        assert len(states_b) == 1
        assert states_b[0].state_id == "state-b1"


class TestGetAllRoutingEventsForRun:
    """Tests for RecorderFactory query — batch via JOIN through node_states."""

    def test_returns_all_events(self):
        _, factory = _setup_full()
        factory.data_flow.create_row("run-1", "source-0", 1, {"b": 2}, row_id="row-2", source_row_index=1, ingest_sequence=1)
        factory.data_flow.create_token("row-2", token_id="tok-2")
        factory.execution.begin_node_state("tok-2", "transform-1", "run-1", 0, {"b": 2}, state_id="state-2")
        factory.execution.record_routing_event(
            state_id="state-1",
            edge_id="edge-1",
            mode=RoutingMode.MOVE,
        )
        factory.execution.record_routing_event(
            state_id="state-2",
            edge_id="edge-1",
            mode=RoutingMode.MOVE,
        )

        events = factory.query.get_all_routing_events_for_run("run-1")

        assert len(events) == 2
        state_ids = {e.state_id for e in events}
        assert state_ids == {"state-1", "state-2"}

    def test_empty_for_unknown_run(self):
        _, factory = _setup()

        events = factory.query.get_all_routing_events_for_run("nonexistent-run")

        assert events == []

    def test_empty_when_no_events_recorded(self):
        _, factory = _setup_full()

        events = factory.query.get_all_routing_events_for_run("run-1")

        assert events == []


class TestGetAllCallsForRun:
    """Tests for RecorderFactory query — state-parented calls via JOIN."""

    def test_returns_all_calls(self):
        _, factory = _setup_full()
        factory.data_flow.create_row("run-1", "source-0", 1, {"b": 2}, row_id="row-2", source_row_index=1, ingest_sequence=1)
        factory.data_flow.create_token("row-2", token_id="tok-2")
        factory.execution.begin_node_state("tok-2", "transform-1", "run-1", 0, {"b": 2}, state_id="state-2")
        factory.execution.record_call(
            state_id="state-1",
            call_index=0,
            call_type=CallType.LLM,
            status=CallStatus.SUCCESS,
            request_data=RawCallPayload({"prompt": "a"}),
            response_data=RawCallPayload({"out": "x"}),
            latency_ms=50.0,
        )
        factory.execution.record_call(
            state_id="state-2",
            call_index=0,
            call_type=CallType.HTTP,
            status=CallStatus.SUCCESS,
            request_data=RawCallPayload({"url": "https://example.com"}),
            response_data=RawCallPayload({"body": "ok"}),
            latency_ms=75.0,
        )

        calls = factory.query.get_all_calls_for_run("run-1")

        assert len(calls) == 2
        state_ids = {c.state_id for c in calls}
        assert state_ids == {"state-1", "state-2"}

    def test_empty_for_unknown_run(self):
        _, factory = _setup()

        calls = factory.query.get_all_calls_for_run("nonexistent-run")

        assert calls == []

    def test_empty_when_no_calls_recorded(self):
        _, factory = _setup_full()

        calls = factory.query.get_all_calls_for_run("run-1")

        assert calls == []


class TestGetAllTokenParentsForRun:
    """Tests for RecorderFactory query — batch via JOIN through tokens and rows."""

    def test_returns_all_parent_relationships_from_fork(self):
        _, factory = _setup_full()
        children, _ = factory.data_flow.fork_token(
            parent_ref=TokenRef(token_id="tok-1", run_id="run-1"),
            row_id="row-1",
            branches=["path-a", "path-b"],
        )

        parents = factory.query.get_all_token_parents_for_run("run-1")

        assert len(parents) == 2
        child_ids = {p.token_id for p in parents}
        assert child_ids == {children[0].token_id, children[1].token_id}
        for p in parents:
            assert p.parent_token_id == "tok-1"

    def test_empty_for_unknown_run(self):
        _, factory = _setup()

        parents = factory.query.get_all_token_parents_for_run("nonexistent-run")

        assert parents == []

    def test_empty_when_no_forks(self):
        _, factory = _setup()
        factory.data_flow.create_row("run-1", "source-0", 0, {"x": 1}, row_id="row-1", source_row_index=0, ingest_sequence=0)
        factory.data_flow.create_token("row-1", token_id="tok-1")

        parents = factory.query.get_all_token_parents_for_run("run-1")

        assert parents == []

    def test_returns_parents_from_coalesce(self):
        _, factory = _setup_full()
        factory.data_flow.create_token("row-1", token_id="tok-2")

        merged = factory.data_flow.coalesce_tokens(
            parent_refs=[TokenRef(token_id="tok-1", run_id="run-1"), TokenRef(token_id="tok-2", run_id="run-1")],
            row_id="row-1",
            merged_payload={"merged": True},
            merged_contract=_MINIMAL_CONTRACT,
        )

        parents = factory.query.get_all_token_parents_for_run("run-1")

        assert len(parents) == 2
        parent_token_ids = {p.parent_token_id for p in parents}
        assert parent_token_ids == {"tok-1", "tok-2"}
        for p in parents:
            assert p.token_id == merged.token_id


class TestExplainRow:
    """Tests for RecorderFactory query — RowLineage with graceful payload degradation."""

    def test_returns_row_lineage(self):
        _, factory = _setup()
        factory.data_flow.create_row("run-1", "source-0", 12, {"field": "value"}, row_id="row-1", source_row_index=3, ingest_sequence=7)

        lineage = factory.query.explain_row("run-1", "row-1")

        assert lineage is not None
        assert lineage.row_id == "row-1"
        assert lineage.run_id == "run-1"
        assert lineage.source_node_id == "source-0"
        assert lineage.row_index == 12
        assert lineage.source_row_index == 3
        assert lineage.ingest_sequence == 7

    def test_none_for_unknown_row(self):
        _, factory = _setup()

        lineage = factory.query.explain_row("run-1", "nonexistent-row")

        assert lineage is None

    def test_raises_for_wrong_run_id(self):
        _, factory = _setup()
        factory.data_flow.create_row("run-1", "source-0", 0, {"x": 1}, row_id="row-1", source_row_index=0, ingest_sequence=0)

        with pytest.raises(AuditIntegrityError, match="Row row-1 belongs to run run-1, not wrong-run"):
            factory.query.explain_row("wrong-run", "row-1")

    def test_payload_available_false_when_no_payload_store(self):
        # Explicitly use a factory with no payload store so source_data_ref is absent.
        # (make_factory now defaults to MockPayloadStore; override for this path.)
        db = make_landscape_db()
        factory = RecorderFactory(db)  # No payload store
        factory.run_lifecycle.begin_run(config={}, canonical_version="v1", run_id="run-1")
        factory.data_flow.register_node(
            run_id="run-1",
            plugin_name="csv",
            node_type=NodeType.SOURCE,
            plugin_version="1.0",
            config={},
            node_id="source-0",
            schema_config=_DYNAMIC_SCHEMA,
        )
        factory.data_flow.create_row("run-1", "source-0", 0, {"x": 1}, row_id="row-1", source_row_index=0, ingest_sequence=0)

        lineage = factory.query.explain_row("run-1", "row-1")

        assert lineage is not None
        assert lineage.payload_available is False

    def test_source_data_hash_present(self):
        _, factory = _setup()
        factory.data_flow.create_row("run-1", "source-0", 0, {"key": "val"}, row_id="row-1", source_row_index=0, ingest_sequence=0)

        lineage = factory.query.explain_row("run-1", "row-1")

        assert lineage is not None
        assert lineage.source_data_hash is not None
        assert isinstance(lineage.source_data_hash, str)
        assert len(lineage.source_data_hash) > 0

    def test_created_at_present(self):
        _, factory = _setup()
        factory.data_flow.create_row("run-1", "source-0", 0, {"x": 1}, row_id="row-1", source_row_index=0, ingest_sequence=0)

        lineage = factory.query.explain_row("run-1", "row-1")

        assert lineage is not None
        assert lineage.created_at is not None

    def test_source_data_none_without_payload_store(self):
        # Explicitly use a factory with no payload store so source_data_ref is absent.
        # (make_factory now defaults to MockPayloadStore; override for this path.)
        db = make_landscape_db()
        factory = RecorderFactory(db)  # No payload store
        factory.run_lifecycle.begin_run(config={}, canonical_version="v1", run_id="run-1")
        factory.data_flow.register_node(
            run_id="run-1",
            plugin_name="csv",
            node_type=NodeType.SOURCE,
            plugin_version="1.0",
            config={},
            node_id="source-0",
            schema_config=_DYNAMIC_SCHEMA,
        )
        factory.data_flow.create_row("run-1", "source-0", 0, {"x": 1}, row_id="row-1", source_row_index=0, ingest_sequence=0)

        lineage = factory.query.explain_row("run-1", "row-1")

        assert lineage is not None
        assert lineage.source_data is None


class TestRoutingEventsOrderedByExecution:
    """Verify batch routing event queries return execution order, not state_id order.

    Regression test for elspeth-rapid-11eh: the N+1 query refactor (ech8)
    introduced ordering by state_id (UUID4 hex — random) instead of
    execution order (step_index, attempt).
    """

    def _setup_three_states(self):
        """Create 3 node states with state_ids that sort opposite to execution order.

        State IDs are chosen so that lexicographic sort (zzz > bbb > aaa)
        is the *reverse* of execution order (step=0/att=0, step=0/att=1, step=1/att=0).
        If the query still sorts by state_id, the test will fail.
        """
        setup = make_recorder_with_run(run_id="run-1", source_node_id="source-0", source_plugin_name="csv")
        factory = setup.factory
        register_test_node(factory.data_flow, "run-1", "transform-1", plugin_name="t1")
        register_test_node(factory.data_flow, "run-1", "transform-2", plugin_name="t2")
        factory.data_flow.register_edge("run-1", "source-0", "transform-1", "continue", RoutingMode.MOVE, edge_id="edge-1")
        factory.data_flow.register_edge("run-1", "transform-1", "transform-2", "continue", RoutingMode.MOVE, edge_id="edge-2")
        factory.data_flow.create_row("run-1", "source-0", 0, {"x": 1}, row_id="row-1", source_row_index=0, ingest_sequence=0)
        factory.data_flow.create_token("row-1", token_id="tok-1")

        # State IDs chosen to sort OPPOSITE to execution order:
        # zzz > bbb > aaa lexicographically, but execution order is aaa, bbb, zzz
        # step=0, attempt=0 → state_id="zzz..." (sorts LAST lexicographically)
        factory.execution.begin_node_state("tok-1", "transform-1", "run-1", 0, {"x": 1}, state_id="zzz-state-first-exec")
        # step=0, attempt=1 (retry) → state_id="bbb..." (sorts MIDDLE)
        factory.execution.begin_node_state("tok-1", "transform-1", "run-1", 0, {"x": 1}, state_id="bbb-state-retry", attempt=1)
        # step=1, attempt=0 → state_id="aaa..." (sorts FIRST lexicographically)
        factory.execution.begin_node_state("tok-1", "transform-2", "run-1", 1, {"x": 1}, state_id="aaa-state-second-step")

        return factory

    def test_routing_events_for_states_ordered_by_step_index_and_attempt(self):
        factory = self._setup_three_states()
        state_ids = ["zzz-state-first-exec", "bbb-state-retry", "aaa-state-second-step"]
        for sid in state_ids:
            factory.execution.record_routing_event(state_id=sid, edge_id="edge-1", mode=RoutingMode.MOVE)

        events = factory.query.get_routing_events_for_states(state_ids)

        assert len(events) == 3
        # Execution order: step=0/att=0, step=0/att=1, step=1/att=0
        assert events[0].state_id == "zzz-state-first-exec"
        assert events[1].state_id == "bbb-state-retry"
        assert events[2].state_id == "aaa-state-second-step"

    def test_all_routing_events_for_run_ordered_by_step_index_and_attempt(self):
        factory = self._setup_three_states()
        state_ids = ["zzz-state-first-exec", "bbb-state-retry", "aaa-state-second-step"]
        for sid in state_ids:
            factory.execution.record_routing_event(state_id=sid, edge_id="edge-1", mode=RoutingMode.MOVE)

        events = factory.query.get_all_routing_events_for_run("run-1")

        assert len(events) == 3
        assert events[0].state_id == "zzz-state-first-exec"
        assert events[1].state_id == "bbb-state-retry"
        assert events[2].state_id == "aaa-state-second-step"


class TestCallsOrderedByExecution:
    """Verify batch call queries return execution order, not state_id order.

    Regression test for elspeth-rapid-11eh: same root cause as above.
    """

    def _setup_three_states(self):
        """Create 3 node states with state_ids that sort opposite to execution order."""
        setup = make_recorder_with_run(run_id="run-1", source_node_id="source-0", source_plugin_name="csv")
        factory = setup.factory
        register_test_node(factory.data_flow, "run-1", "transform-1", plugin_name="t1")
        register_test_node(factory.data_flow, "run-1", "transform-2", plugin_name="t2")
        factory.data_flow.register_edge("run-1", "source-0", "transform-1", "continue", RoutingMode.MOVE, edge_id="edge-1")
        factory.data_flow.register_edge("run-1", "transform-1", "transform-2", "continue", RoutingMode.MOVE, edge_id="edge-2")
        factory.data_flow.create_row("run-1", "source-0", 0, {"x": 1}, row_id="row-1", source_row_index=0, ingest_sequence=0)
        factory.data_flow.create_token("row-1", token_id="tok-1")

        # Same strategy: state_ids sort opposite to execution order
        factory.execution.begin_node_state("tok-1", "transform-1", "run-1", 0, {"x": 1}, state_id="zzz-state-first-exec")
        factory.execution.begin_node_state("tok-1", "transform-1", "run-1", 0, {"x": 1}, state_id="bbb-state-retry", attempt=1)
        factory.execution.begin_node_state("tok-1", "transform-2", "run-1", 1, {"x": 1}, state_id="aaa-state-second-step")

        return factory

    def test_calls_for_states_ordered_by_step_index_and_attempt(self):
        factory = self._setup_three_states()
        state_ids = ["zzz-state-first-exec", "bbb-state-retry", "aaa-state-second-step"]
        for i, sid in enumerate(state_ids):
            factory.execution.record_call(
                state_id=sid,
                call_index=0,
                call_type=CallType.LLM,
                status=CallStatus.SUCCESS,
                request_data=RawCallPayload({"prompt": f"call-{i}"}),
                response_data=RawCallPayload({"out": f"resp-{i}"}),
                latency_ms=50.0,
            )

        calls = factory.query.get_calls_for_states(state_ids)

        assert len(calls) == 3
        # Execution order: step=0/att=0, step=0/att=1, step=1/att=0
        assert calls[0].state_id == "zzz-state-first-exec"
        assert calls[1].state_id == "bbb-state-retry"
        assert calls[2].state_id == "aaa-state-second-step"

    def test_all_calls_for_run_ordered_by_step_index_and_attempt(self):
        factory = self._setup_three_states()
        state_ids = ["zzz-state-first-exec", "bbb-state-retry", "aaa-state-second-step"]
        for i, sid in enumerate(state_ids):
            factory.execution.record_call(
                state_id=sid,
                call_index=0,
                call_type=CallType.LLM,
                status=CallStatus.SUCCESS,
                request_data=RawCallPayload({"prompt": f"call-{i}"}),
                response_data=RawCallPayload({"out": f"resp-{i}"}),
                latency_ms=50.0,
            )

        calls = factory.query.get_all_calls_for_run("run-1")

        assert len(calls) == 3
        assert calls[0].state_id == "zzz-state-first-exec"
        assert calls[1].state_id == "bbb-state-retry"
        assert calls[2].state_id == "aaa-state-second-step"


class TestChunkedQueryMethods:
    """Bug 68zb: IN queries must chunk state_ids for SQLite variable limit."""

    def _setup_many_states(self, factory: RecorderFactory, run_id: str, count: int) -> list[str]:
        """Create many row/token/state triples, return state_ids.

        Uses offset indices (100+) to avoid conflicts with _setup_full which
        creates row-1/tok-1/state-1 at row_index=0.
        """
        state_ids = []
        for i in range(count):
            row_id = f"row-chunk-{i}"
            token_id = f"tok-chunk-{i}"
            state_id = f"state-chunk-{i}"
            factory.data_flow.create_row(
                run_id, "source-0", 100 + i, {"idx": i}, row_id=row_id, source_row_index=100 + i, ingest_sequence=100 + i
            )
            factory.data_flow.create_token(row_id, token_id=token_id)
            factory.execution.begin_node_state(token_id, "transform-1", run_id, 100 + i, {"idx": i}, state_id=state_id)
            state_ids.append(state_id)
        return state_ids

    def test_routing_events_for_states_with_many_state_ids(self):
        """Chunked query returns same results as small query."""
        _, factory = _setup_full()

        # Create enough states to exceed one chunk
        state_ids = self._setup_many_states(factory, "run-1", 10)

        # Record a routing event for each state
        for sid in state_ids:
            factory.execution.record_routing_event(
                state_id=sid,
                edge_id="edge-1",
                mode=RoutingMode.MOVE,
            )

        events = factory.query.get_routing_events_for_states(state_ids)

        assert len(events) == 10
        returned_state_ids = {e.state_id for e in events}
        assert returned_state_ids == set(state_ids)

    def test_calls_for_states_with_many_state_ids(self):
        """Chunked query returns same results as small query."""
        _, factory = _setup_full()

        state_ids = self._setup_many_states(factory, "run-1", 10)

        # Record a call for each state
        for sid in state_ids:
            factory.execution.record_call(
                state_id=sid,
                call_index=0,
                call_type=CallType.LLM,
                status=CallStatus.SUCCESS,
                request_data=RawCallPayload({"prompt": f"call-{sid}"}),
                response_data=RawCallPayload({"out": "ok"}),
                latency_ms=50.0,
            )

        calls = factory.query.get_calls_for_states(state_ids)

        assert len(calls) == 10
        returned_state_ids = {c.state_id for c in calls}
        assert returned_state_ids == set(state_ids)

    def test_routing_events_ordering_preserved_across_chunks(self):
        """Results must maintain execution order even across chunks."""
        from unittest.mock import patch

        _, factory = _setup_full()
        state_ids = self._setup_many_states(factory, "run-1", 5)

        for sid in state_ids:
            factory.execution.record_routing_event(
                state_id=sid,
                edge_id="edge-1",
                mode=RoutingMode.MOVE,
            )

        # Force tiny chunk size to exercise merging
        with patch("elspeth.core.landscape.query_repository.QueryRepository._QUERY_CHUNK_SIZE", 2):
            events = factory.query.get_routing_events_for_states(state_ids)

        assert len(events) == 5
        # step_index increases 0..4, so events should be in state order
        event_state_ids = [e.state_id for e in events]
        assert event_state_ids == state_ids

    def test_routing_events_for_states_use_one_read_snapshot_across_chunks(self):
        """Chunked state-set reads must not open one read snapshot per chunk."""
        from unittest.mock import patch

        db, factory = _setup_full()
        state_ids = self._setup_many_states(factory, "run-1", 5)

        for sid in state_ids:
            factory.execution.record_routing_event(
                state_id=sid,
                edge_id="edge-1",
                mode=RoutingMode.MOVE,
            )

        with (
            patch("elspeth.core.landscape.query_repository.QueryRepository._QUERY_CHUNK_SIZE", 2),
            patch.object(db, "read_only_connection", wraps=db.read_only_connection) as read_only_connection,
        ):
            events = factory.query.get_routing_events_for_states(state_ids)

        assert len(events) == 5
        assert read_only_connection.call_count == 1

    def test_calls_ordering_preserved_across_chunks(self):
        """Results must maintain execution order even across chunks."""
        from unittest.mock import patch

        _, factory = _setup_full()
        state_ids = self._setup_many_states(factory, "run-1", 5)

        for sid in state_ids:
            factory.execution.record_call(
                state_id=sid,
                call_index=0,
                call_type=CallType.LLM,
                status=CallStatus.SUCCESS,
                request_data=RawCallPayload({"prompt": f"call-{sid}"}),
                response_data=RawCallPayload({"out": "ok"}),
                latency_ms=50.0,
            )

        # Force tiny chunk size to exercise merging
        with patch("elspeth.core.landscape.query_repository.QueryRepository._QUERY_CHUNK_SIZE", 2):
            calls = factory.query.get_calls_for_states(state_ids)

        assert len(calls) == 5
        call_state_ids = [c.state_id for c in calls]
        assert call_state_ids == state_ids

    def test_calls_for_states_use_one_read_snapshot_across_chunks(self):
        """Chunked state-set reads must not open one read snapshot per chunk."""
        from unittest.mock import patch

        db, factory = _setup_full()
        state_ids = self._setup_many_states(factory, "run-1", 5)

        for sid in state_ids:
            factory.execution.record_call(
                state_id=sid,
                call_index=0,
                call_type=CallType.LLM,
                status=CallStatus.SUCCESS,
                request_data=RawCallPayload({"prompt": f"call-{sid}"}),
                response_data=RawCallPayload({"out": "ok"}),
                latency_ms=50.0,
            )

        with (
            patch("elspeth.core.landscape.query_repository.QueryRepository._QUERY_CHUNK_SIZE", 2),
            patch.object(db, "read_only_connection", wraps=db.read_only_connection) as read_only_connection,
        ):
            calls = factory.query.get_calls_for_states(state_ids)

        assert len(calls) == 5
        assert read_only_connection.call_count == 1


# === Direct QueryRepository helpers and tests (M8, C1, C2, C3, H1, H2/M1, M3) ===


def _make_repo(
    *,
    run_id: str = "run-1",
    payload_store: PayloadStore | None = None,
) -> tuple[LandscapeDB, QueryRepository, RecorderFactory]:
    """Create a QueryRepository with supporting infrastructure.

    Returns (db, repo, factory) — factory is for graph setup only.
    """
    setup = make_recorder_with_run(run_id=run_id, source_node_id="source-0", source_plugin_name="csv")
    db, factory = setup.db, setup.factory
    register_test_node(factory.data_flow, run_id, "transform-1", plugin_name="transform")
    ops = DatabaseOps(db)
    repo = QueryRepository(
        ops,
        row_loader=RowLoader(),
        token_loader=TokenLoader(),
        token_parent_loader=TokenParentLoader(),
        node_state_loader=NodeStateLoader(),
        routing_event_loader=RoutingEventLoader(),
        call_loader=CallLoader(),
        token_outcome_loader=TokenOutcomeLoader(),
        payload_store=payload_store,
    )
    return db, repo, factory


class TestDirectQueryRepositoryConstruction:
    """M8: Direct QueryRepository constructor tests — not through RecorderFactory."""

    def test_smoke_get_rows_on_empty_db(self):
        _db, repo, _factory = _make_repo()

        rows = repo.get_rows("run-1")

        assert rows == []

    def test_get_row_data_store_not_configured(self):
        """payload_store=None → STORE_NOT_CONFIGURED for rows with refs."""
        payload_store = _PayloadStoreStub()
        # Create factory WITH payload store so create_row sets source_data_ref
        db = make_landscape_db()
        factory = RecorderFactory(db, payload_store=payload_store)
        factory.run_lifecycle.begin_run(config={}, canonical_version="v1", run_id="run-1")
        factory.data_flow.register_node(
            run_id="run-1",
            plugin_name="csv",
            node_type=NodeType.SOURCE,
            plugin_version="1.0",
            config={},
            node_id="source-0",
            schema_config=_DYNAMIC_SCHEMA,
        )
        factory.data_flow.create_row("run-1", "source-0", 0, {"x": 1}, row_id="row-1", source_row_index=0, ingest_sequence=0)

        # Create a QueryRepository WITHOUT a payload store, same DB
        ops = DatabaseOps(db)
        repo_no_store = QueryRepository(
            ops,
            row_loader=RowLoader(),
            token_loader=TokenLoader(),
            token_parent_loader=TokenParentLoader(),
            node_state_loader=NodeStateLoader(),
            routing_event_loader=RoutingEventLoader(),
            call_loader=CallLoader(),
            token_outcome_loader=TokenOutcomeLoader(),
            payload_store=None,
        )
        result = repo_no_store.get_row_data("row-1")
        assert result.state == RowDataState.STORE_NOT_CONFIGURED

    def test_get_row_data_with_valid_payload(self):
        """Payload store returns valid JSON → AVAILABLE with data."""
        payload = json.dumps({"key": "value"}).encode("utf-8")
        payload_store = _PayloadStoreStub(retrieve_result=payload)
        db = make_landscape_db()
        factory = RecorderFactory(db, payload_store=payload_store)
        factory.run_lifecycle.begin_run(config={}, canonical_version="v1", run_id="run-1")
        factory.data_flow.register_node(
            run_id="run-1",
            plugin_name="csv",
            node_type=NodeType.SOURCE,
            plugin_version="1.0",
            config={},
            node_id="source-0",
            schema_config=_DYNAMIC_SCHEMA,
        )
        factory.data_flow.create_row("run-1", "source-0", 0, {"key": "value"}, row_id="row-1", source_row_index=0, ingest_sequence=0)

        ops = DatabaseOps(db)
        repo = QueryRepository(
            ops,
            row_loader=RowLoader(),
            token_loader=TokenLoader(),
            token_parent_loader=TokenParentLoader(),
            node_state_loader=NodeStateLoader(),
            routing_event_loader=RoutingEventLoader(),
            call_loader=CallLoader(),
            token_outcome_loader=TokenOutcomeLoader(),
            payload_store=payload_store,
        )
        result = repo.get_row_data("row-1")

        assert result.state == RowDataState.AVAILABLE
        assert result.data == {"key": "value"}


class TestGetRowDataErrorHandling:
    """C1 + M3: Error handling for get_row_data() payload retrieval.

    Tests cover JSONDecodeError, UnicodeDecodeError, PayloadIntegrityError,
    OSError — all should raise AuditIntegrityError with row context.
    """

    def _make_repo_with_row(self, payload_store: PayloadStore) -> QueryRepository:
        """Create a repo+row where the row has a source_data_ref.

        The payload_store is used for BOTH the factory (so create_row stores a ref)
        and the QueryRepository (so retrieval goes through the fake).
        """
        db = make_landscape_db()
        # Factory WITH payload store — so create_row sets source_data_ref
        factory = RecorderFactory(db, payload_store=payload_store)
        factory.run_lifecycle.begin_run(config={}, canonical_version="v1", run_id="run-1")
        factory.data_flow.register_node(
            run_id="run-1",
            plugin_name="csv",
            node_type=NodeType.SOURCE,
            plugin_version="1.0",
            config={},
            node_id="source-0",
            schema_config=_DYNAMIC_SCHEMA,
        )
        factory.data_flow.create_row("run-1", "source-0", 0, {"x": 1}, row_id="row-1", source_row_index=0, ingest_sequence=0)
        # QueryRepository with SAME mock store — so retrieval hits our mock
        ops = DatabaseOps(db)
        return QueryRepository(
            ops,
            row_loader=RowLoader(),
            token_loader=TokenLoader(),
            token_parent_loader=TokenParentLoader(),
            node_state_loader=NodeStateLoader(),
            routing_event_loader=RoutingEventLoader(),
            call_loader=CallLoader(),
            token_outcome_loader=TokenOutcomeLoader(),
            payload_store=payload_store,
        )

    def test_json_decode_error_raises_audit_integrity(self):
        payload_store = _PayloadStoreStub(retrieve_result=b"not-json{{")
        repo = self._make_repo_with_row(payload_store)

        with pytest.raises(AuditIntegrityError, match="Corrupt payload for row row-1"):
            repo.get_row_data("row-1")

    def test_unicode_decode_error_raises_audit_integrity(self):
        payload_store = _PayloadStoreStub(retrieve_result=b"\xff\xfe")
        repo = self._make_repo_with_row(payload_store)

        with pytest.raises(AuditIntegrityError, match="Corrupt payload for row row-1"):
            repo.get_row_data("row-1")

    def test_payload_integrity_error_raises_audit_integrity(self):
        payload_store = _PayloadStoreStub(retrieve_result=PayloadIntegrityError("hash mismatch"))
        repo = self._make_repo_with_row(payload_store)

        with pytest.raises(AuditIntegrityError, match="Payload integrity check failed for row row-1"):
            repo.get_row_data("row-1")

    def test_os_error_raises_audit_integrity(self):
        payload_store = _PayloadStoreStub(retrieve_result=OSError("Permission denied"))
        repo = self._make_repo_with_row(payload_store)

        with pytest.raises(AuditIntegrityError, match="Payload retrieval failed for row row-1"):
            repo.get_row_data("row-1")

    def test_os_error_message_does_not_expose_backend_details(self):
        payload_store = _PayloadStoreStub(retrieve_result=OSError(13, "Permission denied", "/srv/private/payloads/ref-hash"))
        repo = self._make_repo_with_row(payload_store)

        with pytest.raises(AuditIntegrityError) as exc_info:
            repo.get_row_data("row-1")

        message = str(exc_info.value)
        assert message == "Payload retrieval failed for row row-1: reason=payload_store_os_error"
        assert "ref=ref-hash" not in message
        assert "reason=payload_store_os_error" in message
        assert "Permission denied" not in message
        assert "/srv/private/payloads" not in message
        assert "Errno 13" not in message

    def test_non_dict_json_raises_audit_integrity(self):
        payload_store = _PayloadStoreStub(retrieve_result=json.dumps([1, 2, 3]).encode("utf-8"))
        repo = self._make_repo_with_row(payload_store)

        with pytest.raises(AuditIntegrityError, match="expected JSON object, got list"):
            repo.get_row_data("row-1")

    def test_valid_json_returns_available(self):
        payload_store = _PayloadStoreStub(retrieve_result=json.dumps({"key": "val"}).encode("utf-8"))
        repo = self._make_repo_with_row(payload_store)

        result = repo.get_row_data("row-1")

        assert result.state == RowDataState.AVAILABLE
        assert result.data == {"key": "val"}


class TestExplainRowErrorHandling:
    """C2 + H2/M1 + M3: Error handling for explain_row() payload retrieval.

    Tests cover UnicodeDecodeError, OSError, and PayloadIntegrityError — all
    should raise AuditIntegrityError. The OSError handler was changed from
    silent degradation to crash per Tier 1 trust model.
    """

    def _make_repo_with_row(self, payload_store: PayloadStore) -> QueryRepository:
        """Create a repo+row where the row has a source_data_ref."""
        db = make_landscape_db()
        factory = RecorderFactory(db, payload_store=payload_store)
        factory.run_lifecycle.begin_run(config={}, canonical_version="v1", run_id="run-1")
        factory.data_flow.register_node(
            run_id="run-1",
            plugin_name="csv",
            node_type=NodeType.SOURCE,
            plugin_version="1.0",
            config={},
            node_id="source-0",
            schema_config=_DYNAMIC_SCHEMA,
        )
        factory.data_flow.create_row("run-1", "source-0", 0, {"x": 1}, row_id="row-1", source_row_index=0, ingest_sequence=0)
        ops = DatabaseOps(db)
        return QueryRepository(
            ops,
            row_loader=RowLoader(),
            token_loader=TokenLoader(),
            token_parent_loader=TokenParentLoader(),
            node_state_loader=NodeStateLoader(),
            routing_event_loader=RoutingEventLoader(),
            call_loader=CallLoader(),
            token_outcome_loader=TokenOutcomeLoader(),
            payload_store=payload_store,
        )

    def test_unicode_decode_error_raises_audit_integrity(self):
        payload_store = _PayloadStoreStub(retrieve_result=b"\xff\xfe")
        repo = self._make_repo_with_row(payload_store)

        with pytest.raises(AuditIntegrityError, match="Corrupt payload for row row-1"):
            repo.explain_row("run-1", "row-1")

    def test_os_error_raises_audit_integrity(self):
        """H2: OSError during payload retrieval is infrastructure failure — crash, don't degrade."""
        payload_store = _PayloadStoreStub(retrieve_result=OSError("NFS timeout"))
        repo = self._make_repo_with_row(payload_store)

        with pytest.raises(AuditIntegrityError, match="Payload retrieval failed for row row-1"):
            repo.explain_row("run-1", "row-1")

    def test_payload_integrity_error_raises_audit_integrity(self):
        payload_store = _PayloadStoreStub(retrieve_result=PayloadIntegrityError("hash mismatch"))
        repo = self._make_repo_with_row(payload_store)

        with pytest.raises(AuditIntegrityError, match="Payload integrity check failed for row row-1"):
            repo.explain_row("run-1", "row-1")

    def test_json_decode_error_raises_audit_integrity(self):
        payload_store = _PayloadStoreStub(retrieve_result=b"not-json{{")
        repo = self._make_repo_with_row(payload_store)

        with pytest.raises(AuditIntegrityError, match="Corrupt payload for row row-1"):
            repo.explain_row("run-1", "row-1")

    def test_purged_payload_returns_lineage_without_data(self):
        """PayloadNotFoundError (purged) is the only graceful degradation — not a crash."""
        payload_store = _PayloadStoreStub(retrieve_result=PayloadNotFoundError("deadbeef" * 8))
        repo = self._make_repo_with_row(payload_store)

        lineage = repo.explain_row("run-1", "row-1")

        assert lineage is not None
        assert lineage.source_data is None
        assert lineage.payload_available is False

    def test_get_row_data_purged_returns_purged_state(self):
        """get_row_data returns PURGED when payload was removed by retention policy."""
        payload_store = _PayloadStoreStub(retrieve_result=PayloadNotFoundError("deadbeef" * 8))
        repo = self._make_repo_with_row(payload_store)

        result = repo.get_row_data("row-1")

        assert result.state == RowDataState.PURGED
        assert result.data is None

    def test_run_id_mismatch_raises_value_error(self):
        """H3: Cross-run mismatch is a caller bug, not a normal 'not found'."""
        payload_store = _PayloadStoreStub()
        _db, repo, factory = _make_repo(payload_store=payload_store)
        factory.data_flow.create_row("run-1", "source-0", 0, {"x": 1}, row_id="row-1", source_row_index=0, ingest_sequence=0)

        with pytest.raises(AuditIntegrityError, match="Row row-1 belongs to run run-1, not wrong-run"):
            repo.explain_row("wrong-run", "row-1")

    def test_none_for_unknown_row(self):
        _db, repo, _factory = _make_repo()

        result = repo.explain_row("run-1", "nonexistent-row")

        assert result is None


class TestGetAllTokenOutcomesForRun:
    """C3: Behavioral tests for get_all_token_outcomes_for_run().

    This method had zero behavioral tests — only mocked away in test_exporter.py.
    """

    def test_happy_path_multiple_outcomes(self):
        _, factory = _setup()
        factory.data_flow.create_row("run-1", "source-0", 0, {"a": 1}, row_id="row-1", source_row_index=0, ingest_sequence=0)
        factory.data_flow.create_row("run-1", "source-0", 1, {"b": 2}, row_id="row-2", source_row_index=1, ingest_sequence=1)
        factory.data_flow.create_token("row-1", token_id="tok-1")
        factory.data_flow.create_token("row-2", token_id="tok-2")
        factory.data_flow.record_token_outcome(
            ref=TokenRef(token_id="tok-1", run_id="run-1"),
            outcome=TerminalOutcome.SUCCESS,
            path=TerminalPath.DEFAULT_FLOW,
            sink_name="output",
        )
        factory.data_flow.record_token_outcome(
            ref=TokenRef(token_id="tok-2", run_id="run-1"),
            outcome=TerminalOutcome.FAILURE,
            path=TerminalPath.QUARANTINED_AT_SOURCE,
            error_hash="abc123",
        )

        outcomes = factory.query.get_all_token_outcomes_for_run("run-1")

        assert len(outcomes) == 2
        pair_map = {o.token_id: (o.outcome, o.path) for o in outcomes}
        assert pair_map["tok-1"] == (TerminalOutcome.SUCCESS, TerminalPath.DEFAULT_FLOW)
        assert pair_map["tok-2"] == (TerminalOutcome.FAILURE, TerminalPath.QUARANTINED_AT_SOURCE)

    def test_run_isolation(self):
        """Outcomes from other runs must not leak."""
        db = make_landscape_db()
        factory = make_factory(db)

        factory.run_lifecycle.begin_run(config={}, canonical_version="v1", run_id="run-a")
        factory.data_flow.register_node(
            run_id="run-a",
            plugin_name="csv",
            node_type=NodeType.SOURCE,
            plugin_version="1.0",
            config={},
            node_id="src-a",
            schema_config=_DYNAMIC_SCHEMA,
        )
        factory.data_flow.create_row("run-a", "src-a", 0, {"v": 1}, row_id="row-a1", source_row_index=0, ingest_sequence=0)
        factory.data_flow.create_token("row-a1", token_id="tok-a1")
        factory.data_flow.record_token_outcome(
            ref=TokenRef(token_id="tok-a1", run_id="run-a"),
            outcome=TerminalOutcome.SUCCESS,
            path=TerminalPath.DEFAULT_FLOW,
            sink_name="output",
        )

        factory.run_lifecycle.begin_run(config={}, canonical_version="v1", run_id="run-b")
        factory.data_flow.register_node(
            run_id="run-b",
            plugin_name="csv",
            node_type=NodeType.SOURCE,
            plugin_version="1.0",
            config={},
            node_id="src-b",
            schema_config=_DYNAMIC_SCHEMA,
        )
        factory.data_flow.create_row("run-b", "src-b", 0, {"v": 2}, row_id="row-b1", source_row_index=0, ingest_sequence=0)
        factory.data_flow.create_token("row-b1", token_id="tok-b1")
        factory.data_flow.record_token_outcome(
            ref=TokenRef(token_id="tok-b1", run_id="run-b"),
            outcome=TerminalOutcome.FAILURE,
            path=TerminalPath.UNROUTED,
            error_hash="err-hash-1",
        )

        outcomes_a = factory.query.get_all_token_outcomes_for_run("run-a")
        outcomes_b = factory.query.get_all_token_outcomes_for_run("run-b")

        assert len(outcomes_a) == 1
        assert outcomes_a[0].token_id == "tok-a1"
        assert len(outcomes_b) == 1
        assert outcomes_b[0].token_id == "tok-b1"

    def test_empty_for_unknown_run(self):
        _, factory = _setup()

        outcomes = factory.query.get_all_token_outcomes_for_run("nonexistent-run")

        assert outcomes == []

    def test_ordering_by_token_id_then_recorded_at(self):
        """Results must be ordered by (token_id, recorded_at)."""
        _, factory = _setup()
        factory.data_flow.create_row("run-1", "source-0", 0, {"a": 1}, row_id="row-1", source_row_index=0, ingest_sequence=0)
        # Create tokens with IDs that sort in known order
        factory.data_flow.create_token("row-1", token_id="tok-aaa")
        factory.data_flow.create_token("row-1", token_id="tok-zzz")
        factory.data_flow.record_token_outcome(
            ref=TokenRef(token_id="tok-zzz", run_id="run-1"),
            outcome=TerminalOutcome.SUCCESS,
            path=TerminalPath.DEFAULT_FLOW,
            sink_name="output",
        )
        factory.data_flow.record_token_outcome(
            ref=TokenRef(token_id="tok-aaa", run_id="run-1"),
            outcome=TerminalOutcome.SUCCESS,
            path=TerminalPath.DEFAULT_FLOW,
            sink_name="output",
        )

        outcomes = factory.query.get_all_token_outcomes_for_run("run-1")

        assert len(outcomes) == 2
        assert outcomes[0].token_id == "tok-aaa"
        assert outcomes[1].token_id == "tok-zzz"

    def test_multiple_outcomes_per_token(self):
        """A token can have multiple outcomes (e.g., fork then complete children)."""
        _, factory = _setup_full()
        # First outcome: FORKED
        factory.data_flow.record_token_outcome(
            ref=TokenRef(token_id="tok-1", run_id="run-1"),
            outcome=TerminalOutcome.TRANSIENT,
            path=TerminalPath.FORK_PARENT,
            fork_group_id="fg-1",
        )

        outcomes = factory.query.get_all_token_outcomes_for_run("run-1")

        assert len(outcomes) == 1
        assert outcomes[0].token_id == "tok-1"
        assert outcomes[0].outcome == TerminalOutcome.TRANSIENT
        assert outcomes[0].path == TerminalPath.FORK_PARENT


def _coalesce_failure(reason: str = "quorum_not_met_at_timeout") -> CoalesceFailureReason:
    return CoalesceFailureReason(
        failure_reason=reason,
        expected_branches=("branch_a", "branch_b"),
        branches_arrived=("branch_a",),
        merge_policy="nested",
    )


class TestAuditRunStatusProjection:
    """Pin the anchor for ``rows_coalesce_failed`` audit derivation (elspeth-7294de558e).

    The counter is per failed BARRIER — one pending key ``(coalesce_name,
    row_id)`` — reconstructed as DISTINCT ``(node_id, row_id)`` pairs over
    FAILED node_states at nodes registered with ``node_type='coalesce'``
    (the structural, indexed anchor), with the single
    ``late_arrival_after_merge`` reason excluded (a straggler rejected after
    the barrier resolved is not itself a barrier failure).
    """

    def _setup_coalesce(self, *, run_id: str = "run-1"):
        db, factory = _setup(run_id=run_id)
        register_test_node(factory.data_flow, run_id, "coalesce-1", node_type=NodeType.COALESCE, plugin_name="coalesce")
        factory.data_flow.create_row(run_id, "source-0", 0, {"value": 1}, row_id="row-1", source_row_index=0, ingest_sequence=0)
        return db, factory

    def _fail_state(
        self, factory, *, token_id: str, node_id: str, state_id: str, reason: str = "quorum_not_met_at_timeout", run_id: str = "run-1"
    ) -> None:
        factory.execution.begin_node_state(token_id, node_id, run_id, 0, {"value": 1}, state_id=state_id)
        factory.execution.complete_node_state(
            state_id=state_id,
            status=NodeStateStatus.FAILED,
            error=_coalesce_failure(reason),
            duration_ms=0.0,
        )

    def test_granularity_one_barrier_per_node_row_pair_not_per_branch_token(self):
        """A 2-branch barrier failure writes 2 FAILED states (one per arrived
        branch token, same row) but counts as ONE failed barrier — the naive
        per-state (or per-token-outcome) tally over-reports it as 2."""
        _db, factory = self._setup_coalesce()
        factory.data_flow.create_token("row-1", token_id="tok-branch-a")
        factory.data_flow.create_token("row-1", token_id="tok-branch-b")
        self._fail_state(factory, token_id="tok-branch-a", node_id="coalesce-1", state_id="cs-a")
        self._fail_state(factory, token_id="tok-branch-b", node_id="coalesce-1", state_id="cs-b")

        assert factory.run_status_projection.count_failed_coalesce_barrier_rows("run-1") == 1

    def test_attribution_failed_states_at_non_coalesce_nodes_do_not_count(self):
        """The anchor is nodes.node_type='coalesce': an ordinary transform
        failure must not register as a coalesce barrier failure."""
        _db, factory = self._setup_coalesce()
        factory.data_flow.create_token("row-1", token_id="tok-1")
        self._fail_state(factory, token_id="tok-1", node_id="transform-1", state_id="ts-1")

        assert factory.run_status_projection.count_failed_coalesce_barrier_rows("run-1") == 0

    def test_distinct_rows_count_as_distinct_barriers(self):
        """Two rows failing at the same coalesce node are two failed barriers."""
        _db, factory = self._setup_coalesce()
        factory.data_flow.create_row("run-1", "source-0", 1, {"value": 2}, row_id="row-2", source_row_index=1, ingest_sequence=1)
        factory.data_flow.create_token("row-1", token_id="tok-r1")
        factory.data_flow.create_token("row-2", token_id="tok-r2")
        self._fail_state(factory, token_id="tok-r1", node_id="coalesce-1", state_id="cs-r1")
        self._fail_state(factory, token_id="tok-r2", node_id="coalesce-1", state_id="cs-r2")

        assert factory.run_status_projection.count_failed_coalesce_barrier_rows("run-1") == 2

    def test_late_arrival_after_merge_is_excluded(self):
        """A straggler rejected AFTER the barrier resolved (e.g. after a
        successful quorum merge) is not a barrier failure: counting it would
        report a coalesce-failure for a row whose coalesce SUCCEEDED."""
        _db, factory = self._setup_coalesce()
        factory.data_flow.create_token("row-1", token_id="tok-late")
        self._fail_state(factory, token_id="tok-late", node_id="coalesce-1", state_id="cs-late", reason="late_arrival_after_merge")

        assert factory.run_status_projection.count_failed_coalesce_barrier_rows("run-1") == 0

    def test_late_arrival_does_not_mask_a_real_barrier_failure_on_same_pair(self):
        """A failed barrier followed by a late straggler still counts exactly
        once — the DISTINCT pair collapse absorbs the straggler state."""
        _db, factory = self._setup_coalesce()
        factory.data_flow.create_token("row-1", token_id="tok-branch-a")
        factory.data_flow.create_token("row-1", token_id="tok-late")
        self._fail_state(factory, token_id="tok-branch-a", node_id="coalesce-1", state_id="cs-a")
        self._fail_state(factory, token_id="tok-late", node_id="coalesce-1", state_id="cs-late", reason="late_arrival_after_merge")

        assert factory.run_status_projection.count_failed_coalesce_barrier_rows("run-1") == 1

    def test_zero_when_no_coalesce_failures(self):
        _db, factory = self._setup_coalesce()

        assert factory.run_status_projection.count_failed_coalesce_barrier_rows("run-1") == 0


# =============================================================================
# Chunked export read APIs (elspeth-3ae79a4775)
#
# These methods let the exporter stream a run in bounded row batches instead
# of preloading every child collection. The contract under test: grouping a
# set-scoped getter's result by parent id yields exactly the same per-parent
# sequences as grouping the corresponding full-run getter's result.
# =============================================================================


def _grouped_by(items: list, key: str) -> dict[str, list]:
    grouped: dict[str, list] = {}
    for item in items:
        grouped.setdefault(getattr(item, key), []).append(item)
    return grouped


class TestIterRowsForRun:
    """Keyset-paginated row batches must partition get_rows() exactly."""

    def _setup_rows(self, count: int = 5):
        _db, factory = _setup()
        # Insert out of ingest order to prove ordering comes from the query.
        for i in reversed(range(count)):
            factory.data_flow.create_row("run-1", "source-0", i, {"v": i}, row_id=f"row-{i}", source_row_index=i, ingest_sequence=i)
        return factory

    def test_batches_partition_get_rows_order(self):
        factory = self._setup_rows(5)

        batches = list(factory.query.iter_rows_for_run("run-1", batch_size=2))

        assert [len(b) for b in batches] == [2, 2, 1]
        flat = [r.row_id for batch in batches for r in batch]
        assert flat == [r.row_id for r in factory.query.get_rows("run-1")]
        assert flat == ["row-0", "row-1", "row-2", "row-3", "row-4"]

    def test_single_batch_when_batch_size_exceeds_row_count(self):
        factory = self._setup_rows(3)

        batches = list(factory.query.iter_rows_for_run("run-1", batch_size=100))

        assert [len(b) for b in batches] == [3]

    def test_exact_multiple_of_batch_size(self):
        factory = self._setup_rows(4)

        batches = list(factory.query.iter_rows_for_run("run-1", batch_size=2))

        assert [len(b) for b in batches] == [2, 2]

    def test_empty_run_yields_no_batches(self):
        _db, factory = _setup()

        assert list(factory.query.iter_rows_for_run("run-1", batch_size=2)) == []

    def test_rejects_non_positive_batch_size(self):
        _db, factory = _setup()

        with pytest.raises(ValueError, match="batch_size"):
            list(factory.query.iter_rows_for_run("run-1", batch_size=0))

    def test_scoped_to_run(self):
        db = make_landscape_db()
        factory = make_factory(db)
        for run_id, src in (("run-a", "src-a"), ("run-b", "src-b")):
            factory.run_lifecycle.begin_run(config={}, canonical_version="v1", run_id=run_id)
            factory.data_flow.register_node(
                run_id=run_id,
                plugin_name="csv",
                node_type=NodeType.SOURCE,
                plugin_version="1.0",
                config={},
                node_id=src,
                schema_config=_DYNAMIC_SCHEMA,
            )
        factory.data_flow.create_row("run-a", "src-a", 0, {"v": 1}, row_id="row-a1", source_row_index=0, ingest_sequence=0)
        factory.data_flow.create_row("run-b", "src-b", 0, {"v": 2}, row_id="row-b1", source_row_index=0, ingest_sequence=0)

        batches = list(factory.query.iter_rows_for_run("run-a", batch_size=10))

        assert [[r.row_id for r in batch] for batch in batches] == [["row-a1"]]


class TestGetTokensForRows:
    """Set-scoped token reads must match per-row order of get_tokens()."""

    def _setup_tokens(self):
        _db, factory = _setup()
        factory.data_flow.create_row("run-1", "source-0", 0, {"a": 1}, row_id="row-a", source_row_index=0, ingest_sequence=0)
        factory.data_flow.create_row("run-1", "source-0", 1, {"b": 2}, row_id="row-b", source_row_index=1, ingest_sequence=1)
        # Interleave creation across rows so per-row order is not insert order.
        factory.data_flow.create_token("row-b", token_id="tok-b2")
        factory.data_flow.create_token("row-a", token_id="tok-a2")
        factory.data_flow.create_token("row-b", token_id="tok-b1")
        factory.data_flow.create_token("row-a", token_id="tok-a1")
        return factory

    def test_per_row_grouping_matches_per_row_getter(self):
        factory = self._setup_tokens()

        tokens = factory.query.get_tokens_for_rows("run-1", ["row-a", "row-b"])

        grouped = _grouped_by(tokens, "row_id")
        for row_id in ("row-a", "row-b"):
            assert [t.token_id for t in grouped[row_id]] == [t.token_id for t in factory.query.get_tokens(row_id)]

    def test_only_requested_rows_returned(self):
        factory = self._setup_tokens()

        tokens = factory.query.get_tokens_for_rows("run-1", ["row-a"])

        assert {t.row_id for t in tokens} == {"row-a"}
        assert {t.token_id for t in tokens} == {"tok-a1", "tok-a2"}

    def test_empty_input_returns_empty(self):
        factory = self._setup_tokens()

        assert factory.query.get_tokens_for_rows("run-1", []) == []

    def test_run_mismatch_returns_empty(self):
        factory = self._setup_tokens()

        assert factory.query.get_tokens_for_rows("other-run", ["row-a", "row-b"]) == []

    def test_chunked_input_equals_unchunked(self):
        factory = self._setup_tokens()
        unchunked = factory.query.get_tokens_for_rows("run-1", ["row-a", "row-b"])

        factory.query._QUERY_CHUNK_SIZE = 1  # force one IN-chunk per row

        chunked = factory.query.get_tokens_for_rows("run-1", ["row-a", "row-b"])
        assert _grouped_by(chunked, "row_id").keys() == _grouped_by(unchunked, "row_id").keys()
        for row_id, group in _grouped_by(unchunked, "row_id").items():
            assert [t.token_id for t in _grouped_by(chunked, "row_id")[row_id]] == [t.token_id for t in group]


class TestGetTokenParentsForTokens:
    """Set-scoped parent reads must match per-token order of get_token_parents()."""

    def _setup_fork(self):
        _db, factory = _setup()
        factory.data_flow.create_row("run-1", "source-0", 0, {"a": 1}, row_id="row-1", source_row_index=0, ingest_sequence=0)
        parent = factory.data_flow.create_token("row-1", token_id="tok-parent")
        children, _ = factory.data_flow.fork_token(
            TokenRef(token_id=parent.token_id, run_id="run-1"),
            "row-1",
            ["left", "right"],
            step_in_pipeline=1,
        )
        return factory, [child.token_id for child in children]

    def test_parents_grouped_match_per_token_getter(self):
        factory, child_ids = self._setup_fork()

        parents = factory.query.get_token_parents_for_tokens(child_ids)

        grouped = _grouped_by(parents, "token_id")
        assert set(grouped.keys()) == set(child_ids)
        for child_id in child_ids:
            assert [(p.parent_token_id, p.ordinal) for p in grouped[child_id]] == [
                (p.parent_token_id, p.ordinal) for p in factory.query.get_token_parents(child_id)
            ]

    def test_only_requested_tokens_returned(self):
        factory, child_ids = self._setup_fork()

        parents = factory.query.get_token_parents_for_tokens(child_ids[:1])

        assert {p.token_id for p in parents} == {child_ids[0]}

    def test_empty_input_returns_empty(self):
        factory, _child_ids = self._setup_fork()

        assert factory.query.get_token_parents_for_tokens([]) == []


class TestGetNodeStatesForTokens:
    """Set-scoped state reads must match per-token order of get_node_states_for_token()."""

    def _setup_states(self):
        _db, factory = _setup()
        # node_states is UNIQUE on (token_id, node_id, attempt), so a token's
        # two states must sit on different nodes.
        register_test_node(factory.data_flow, "run-1", "transform-2", plugin_name="transform")
        factory.data_flow.create_row("run-1", "source-0", 0, {"a": 1}, row_id="row-1", source_row_index=0, ingest_sequence=0)
        factory.data_flow.create_token("row-1", token_id="tok-1")
        factory.data_flow.create_token("row-1", token_id="tok-2")
        # Insert out of step order to prove ordering comes from the query.
        factory.execution.begin_node_state("tok-1", "transform-2", "run-1", 2, {"a": 1}, state_id="st-1-late")
        factory.execution.begin_node_state("tok-2", "transform-1", "run-1", 0, {"a": 1}, state_id="st-2")
        factory.execution.begin_node_state("tok-1", "transform-1", "run-1", 0, {"a": 1}, state_id="st-1-early")
        return factory

    def test_states_grouped_match_per_token_getter(self):
        factory = self._setup_states()

        states = factory.query.get_node_states_for_tokens("run-1", ["tok-1", "tok-2"])

        grouped = _grouped_by(states, "token_id")
        for token_id in ("tok-1", "tok-2"):
            assert [s.state_id for s in grouped[token_id]] == [s.state_id for s in factory.query.get_node_states_for_token(token_id)]
        assert [s.state_id for s in grouped["tok-1"]] == ["st-1-early", "st-1-late"]

    def test_only_requested_tokens_returned(self):
        factory = self._setup_states()

        states = factory.query.get_node_states_for_tokens("run-1", ["tok-2"])

        assert [s.state_id for s in states] == ["st-2"]

    def test_run_mismatch_returns_empty(self):
        factory = self._setup_states()

        assert factory.query.get_node_states_for_tokens("other-run", ["tok-1", "tok-2"]) == []

    def test_empty_input_returns_empty(self):
        factory = self._setup_states()

        assert factory.query.get_node_states_for_tokens("run-1", []) == []


class TestGetTokenOutcomesForTokens:
    """Set-scoped outcome reads must group identically to the full-run getter."""

    def _setup_outcomes(self):
        _db, factory = _setup()
        factory.data_flow.create_row("run-1", "source-0", 0, {"a": 1}, row_id="row-1", source_row_index=0, ingest_sequence=0)
        factory.data_flow.create_token("row-1", token_id="tok-1")
        factory.data_flow.create_token("row-1", token_id="tok-2")
        factory.data_flow.record_token_outcome(
            TokenRef(token_id="tok-1", run_id="run-1"), TerminalOutcome.SUCCESS, TerminalPath.DEFAULT_FLOW, sink_name="out"
        )
        factory.data_flow.record_token_outcome(
            TokenRef(token_id="tok-2", run_id="run-1"),
            TerminalOutcome.FAILURE,
            TerminalPath.UNROUTED,
            error_hash="0" * 64,
        )
        return factory

    def test_outcomes_grouped_match_full_run_getter(self):
        factory = self._setup_outcomes()

        outcomes = factory.query.get_token_outcomes_for_tokens("run-1", ["tok-1", "tok-2"])

        grouped = _grouped_by(outcomes, "token_id")
        full = _grouped_by(factory.query.get_all_token_outcomes_for_run("run-1"), "token_id")
        assert grouped.keys() == full.keys()
        for token_id, group in full.items():
            assert [o.outcome_id for o in grouped[token_id]] == [o.outcome_id for o in group]

    def test_only_requested_tokens_returned(self):
        factory = self._setup_outcomes()

        outcomes = factory.query.get_token_outcomes_for_tokens("run-1", ["tok-2"])

        assert {o.token_id for o in outcomes} == {"tok-2"}

    def test_run_mismatch_returns_empty(self):
        factory = self._setup_outcomes()

        assert factory.query.get_token_outcomes_for_tokens("other-run", ["tok-1", "tok-2"]) == []

    def test_empty_input_returns_empty(self):
        factory = self._setup_outcomes()

        assert factory.query.get_token_outcomes_for_tokens("run-1", []) == []


class TestGetSchedulerEventsForTokens:
    """Set-scoped scheduler-event reads must group identically to get_scheduler_events()."""

    def _setup_events(self):
        _db, factory = _setup()
        factory.data_flow.create_row("run-1", "source-0", 0, {"a": 1}, row_id="row-1", source_row_index=0, ingest_sequence=0)
        factory.data_flow.create_row("run-1", "source-0", 1, {"b": 2}, row_id="row-2", source_row_index=1, ingest_sequence=1)
        factory.data_flow.create_token("row-1", token_id="tok-1")
        factory.data_flow.create_token("row-2", token_id="tok-2")
        payload = factory.scheduler.serialize_row_payload(PipelineRow({"id": 1}, _MINIMAL_CONTRACT))
        now = datetime.now(UTC)
        for token_id, row_id, ingest in (("tok-1", "row-1", 0), ("tok-2", "row-2", 1)):
            factory.scheduler.enqueue_ready(
                run_id="run-1",
                token_id=token_id,
                row_id=row_id,
                node_id="transform-1",
                step_index=1,
                ingest_sequence=ingest,
                available_at=now,
                row_payload_json=payload,
            )
        return factory

    def test_events_grouped_match_full_run_getter(self):
        factory = self._setup_events()

        events = factory.query.get_scheduler_events_for_tokens("run-1", ["tok-1", "tok-2"])

        grouped = _grouped_by(events, "token_id")
        full = _grouped_by(factory.query.get_scheduler_events(run_id="run-1"), "token_id")
        assert grouped.keys() == full.keys()
        for token_id, group in full.items():
            assert [e.event_id for e in grouped[token_id]] == [e.event_id for e in group]

    def test_only_requested_tokens_returned(self):
        factory = self._setup_events()

        events = factory.query.get_scheduler_events_for_tokens("run-1", ["tok-2"])

        assert {e.token_id for e in events} == {"tok-2"}

    def test_run_mismatch_returns_empty(self):
        factory = self._setup_events()

        assert factory.query.get_scheduler_events_for_tokens("other-run", ["tok-1", "tok-2"]) == []

    def test_empty_input_returns_empty(self):
        factory = self._setup_events()

        assert factory.query.get_scheduler_events_for_tokens("run-1", []) == []
