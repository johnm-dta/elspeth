"""Tests for Landscape repository layer.

Verifies that repositories correctly convert database strings to enum types,
and crash on invalid data per Data Manifesto.
"""

from dataclasses import dataclass
from datetime import UTC, datetime

import pytest

from elspeth.contracts import (
    BatchStatus,
    CallStatus,
    CallType,
    Determinism,
    ExportStatus,
    NodeType,
    RoutingMode,
    RunStatus,
)
from elspeth.core.landscape.repositories import (
    BatchRepository,
    CallRepository,
    EdgeRepository,
    NodeRepository,
    RoutingEventRepository,
    RowRepository,
    RunRepository,
    TokenParentRepository,
    TokenRepository,
)


@dataclass
class MockDbRow:
    """Mock SQLAlchemy row for testing.

    Simulates database rows that store enums as strings.
    """

    pass


class TestRunRepository:
    """Tests for RunRepository."""

    def test_load_converts_status_to_enum(self) -> None:
        """Repository converts string status to RunStatus enum."""

        @dataclass
        class RunRow:
            run_id: str
            started_at: datetime
            config_hash: str
            settings_json: str
            canonical_version: str
            status: str  # String in DB
            completed_at: datetime | None = None
            reproducibility_grade: str | None = None
            export_status: str | None = None
            export_error: str | None = None
            exported_at: datetime | None = None
            export_format: str | None = None
            export_sink: str | None = None

        db_row = RunRow(
            run_id="run-123",
            started_at=datetime.now(UTC),
            config_hash="abc123",
            settings_json="{}",
            canonical_version="1.0.0",
            status="running",  # String from DB
        )

        repo = RunRepository(session=None)
        run = repo.load(db_row)

        assert run.status == RunStatus.RUNNING
        assert isinstance(run.status, RunStatus)

    def test_load_converts_export_status_to_enum(self) -> None:
        """Repository converts export_status string to ExportStatus enum."""

        @dataclass
        class RunRow:
            run_id: str
            started_at: datetime
            config_hash: str
            settings_json: str
            canonical_version: str
            status: str
            completed_at: datetime | None = None
            reproducibility_grade: str | None = None
            export_status: str | None = None
            export_error: str | None = None
            exported_at: datetime | None = None
            export_format: str | None = None
            export_sink: str | None = None

        db_row = RunRow(
            run_id="run-123",
            started_at=datetime.now(UTC),
            config_hash="abc123",
            settings_json="{}",
            canonical_version="1.0.0",
            status="completed",
            export_status="completed",  # String from DB
        )

        repo = RunRepository(session=None)
        run = repo.load(db_row)

        assert run.export_status == ExportStatus.COMPLETED
        assert isinstance(run.export_status, ExportStatus)

    def test_load_handles_null_export_status(self) -> None:
        """Repository handles NULL export_status correctly."""

        @dataclass
        class RunRow:
            run_id: str
            started_at: datetime
            config_hash: str
            settings_json: str
            canonical_version: str
            status: str
            completed_at: datetime | None = None
            reproducibility_grade: str | None = None
            export_status: str | None = None
            export_error: str | None = None
            exported_at: datetime | None = None
            export_format: str | None = None
            export_sink: str | None = None

        db_row = RunRow(
            run_id="run-123",
            started_at=datetime.now(UTC),
            config_hash="abc123",
            settings_json="{}",
            canonical_version="1.0.0",
            status="running",
            export_status=None,  # NULL in DB
        )

        repo = RunRepository(session=None)
        run = repo.load(db_row)

        assert run.export_status is None

    def test_load_crashes_on_invalid_status(self) -> None:
        """Repository crashes on invalid status per Data Manifesto."""

        @dataclass
        class RunRow:
            run_id: str
            started_at: datetime
            config_hash: str
            settings_json: str
            canonical_version: str
            status: str
            completed_at: datetime | None = None
            reproducibility_grade: str | None = None
            export_status: str | None = None
            export_error: str | None = None
            exported_at: datetime | None = None
            export_format: str | None = None
            export_sink: str | None = None

        db_row = RunRow(
            run_id="run-123",
            started_at=datetime.now(UTC),
            config_hash="abc123",
            settings_json="{}",
            canonical_version="1.0.0",
            status="invalid_garbage",  # Invalid!
        )

        repo = RunRepository(session=None)

        with pytest.raises(ValueError):
            repo.load(db_row)


class TestNodeRepository:
    """Tests for NodeRepository."""

    def test_load_converts_node_type_to_enum(self) -> None:
        """Repository converts node_type string to NodeType enum."""

        @dataclass
        class NodeRow:
            node_id: str
            run_id: str
            plugin_name: str
            node_type: str  # String in DB
            plugin_version: str
            determinism: str  # String in DB
            config_hash: str
            config_json: str
            registered_at: datetime
            schema_hash: str | None = None
            sequence_in_pipeline: int | None = None

        db_row = NodeRow(
            node_id="node-123",
            run_id="run-456",
            plugin_name="csv_source",
            node_type="source",  # String from DB
            plugin_version="1.0.0",
            determinism="io_read",  # String from DB
            config_hash="abc123",
            config_json="{}",
            registered_at=datetime.now(UTC),
        )

        repo = NodeRepository(session=None)
        node = repo.load(db_row)

        assert node.node_type == NodeType.SOURCE
        assert isinstance(node.node_type, NodeType)
        assert node.determinism == Determinism.IO_READ
        assert isinstance(node.determinism, Determinism)

    def test_load_crashes_on_invalid_node_type(self) -> None:
        """Repository crashes on invalid node_type per Data Manifesto."""

        @dataclass
        class NodeRow:
            node_id: str
            run_id: str
            plugin_name: str
            node_type: str
            plugin_version: str
            determinism: str
            config_hash: str
            config_json: str
            registered_at: datetime
            schema_hash: str | None = None
            sequence_in_pipeline: int | None = None

        db_row = NodeRow(
            node_id="node-123",
            run_id="run-456",
            plugin_name="plugin",
            node_type="invalid_type",  # Invalid!
            plugin_version="1.0.0",
            determinism="deterministic",
            config_hash="abc123",
            config_json="{}",
            registered_at=datetime.now(UTC),
        )

        repo = NodeRepository(session=None)

        with pytest.raises(ValueError):
            repo.load(db_row)

    def test_load_crashes_on_invalid_determinism(self) -> None:
        """Repository crashes on invalid determinism per Data Manifesto."""

        @dataclass
        class NodeRow:
            node_id: str
            run_id: str
            plugin_name: str
            node_type: str
            plugin_version: str
            determinism: str
            config_hash: str
            config_json: str
            registered_at: datetime
            schema_hash: str | None = None
            sequence_in_pipeline: int | None = None

        db_row = NodeRow(
            node_id="node-123",
            run_id="run-456",
            plugin_name="plugin",
            node_type="transform",
            plugin_version="1.0.0",
            determinism="maybe_deterministic",  # Invalid!
            config_hash="abc123",
            config_json="{}",
            registered_at=datetime.now(UTC),
        )

        repo = NodeRepository(session=None)

        with pytest.raises(ValueError):
            repo.load(db_row)


class TestEdgeRepository:
    """Tests for EdgeRepository."""

    def test_load_converts_default_mode_to_enum(self) -> None:
        """Repository converts default_mode string to RoutingMode enum."""

        @dataclass
        class EdgeRow:
            edge_id: str
            run_id: str
            from_node_id: str
            to_node_id: str
            label: str
            default_mode: str  # String in DB
            created_at: datetime

        db_row = EdgeRow(
            edge_id="edge-123",
            run_id="run-456",
            from_node_id="node-1",
            to_node_id="node-2",
            label="continue",
            default_mode="move",  # String from DB
            created_at=datetime.now(UTC),
        )

        repo = EdgeRepository(session=None)
        edge = repo.load(db_row)

        assert edge.default_mode == RoutingMode.MOVE
        assert isinstance(edge.default_mode, RoutingMode)

    def test_load_crashes_on_invalid_default_mode(self) -> None:
        """Repository crashes on invalid default_mode per Data Manifesto."""

        @dataclass
        class EdgeRow:
            edge_id: str
            run_id: str
            from_node_id: str
            to_node_id: str
            label: str
            default_mode: str
            created_at: datetime

        db_row = EdgeRow(
            edge_id="edge-123",
            run_id="run-456",
            from_node_id="node-1",
            to_node_id="node-2",
            label="continue",
            default_mode="teleport",  # Invalid!
            created_at=datetime.now(UTC),
        )

        repo = EdgeRepository(session=None)

        with pytest.raises(ValueError):
            repo.load(db_row)


class TestRowRepository:
    """Tests for RowRepository."""

    def test_load_primitive_fields(self) -> None:
        """Repository loads Row with all primitive fields."""

        @dataclass
        class RowRow:
            row_id: str
            run_id: str
            source_node_id: str
            row_index: int
            source_data_hash: str
            created_at: datetime
            source_data_ref: str | None = None

        db_row = RowRow(
            row_id="row-123",
            run_id="run-456",
            source_node_id="node-1",
            row_index=42,
            source_data_hash="abc123",
            created_at=datetime.now(UTC),
            source_data_ref="payload://xyz",
        )

        repo = RowRepository(session=None)
        row = repo.load(db_row)

        assert row.row_id == "row-123"
        assert row.row_index == 42
        assert row.source_data_ref == "payload://xyz"


class TestTokenRepository:
    """Tests for TokenRepository."""

    def test_load_primitive_fields(self) -> None:
        """Repository loads Token with all primitive fields."""

        @dataclass
        class TokenRow:
            token_id: str
            row_id: str
            created_at: datetime
            fork_group_id: str | None = None
            join_group_id: str | None = None
            branch_name: str | None = None
            step_in_pipeline: int | None = None

        db_row = TokenRow(
            token_id="tok-123",
            row_id="row-456",
            created_at=datetime.now(UTC),
            fork_group_id="fork-789",
            branch_name="sentiment",
        )

        repo = TokenRepository(session=None)
        token = repo.load(db_row)

        assert token.token_id == "tok-123"
        assert token.fork_group_id == "fork-789"
        assert token.branch_name == "sentiment"


class TestTokenParentRepository:
    """Tests for TokenParentRepository."""

    def test_load_primitive_fields(self) -> None:
        """Repository loads TokenParent with all primitive fields."""

        @dataclass
        class TokenParentRow:
            token_id: str
            parent_token_id: str
            ordinal: int

        db_row = TokenParentRow(
            token_id="tok-child",
            parent_token_id="tok-parent",
            ordinal=0,
        )

        repo = TokenParentRepository(session=None)
        parent = repo.load(db_row)

        assert parent.token_id == "tok-child"
        assert parent.parent_token_id == "tok-parent"
        assert parent.ordinal == 0


class TestCallRepository:
    """Tests for CallRepository."""

    def test_load_converts_enums(self) -> None:
        """Repository converts call_type and status strings to enums."""

        @dataclass
        class CallRow:
            call_id: str
            state_id: str
            call_index: int
            call_type: str  # String in DB
            status: str  # String in DB
            request_hash: str
            created_at: datetime
            request_ref: str | None = None
            response_hash: str | None = None
            response_ref: str | None = None
            error_json: str | None = None
            latency_ms: float | None = None

        db_row = CallRow(
            call_id="call-123",
            state_id="state-456",
            call_index=0,
            call_type="llm",  # String from DB
            status="success",  # String from DB
            request_hash="abc123",
            created_at=datetime.now(UTC),
            latency_ms=150.5,
        )

        repo = CallRepository(session=None)
        call = repo.load(db_row)

        assert call.call_type == CallType.LLM
        assert isinstance(call.call_type, CallType)
        assert call.status == CallStatus.SUCCESS
        assert isinstance(call.status, CallStatus)

    def test_load_crashes_on_invalid_call_type(self) -> None:
        """Repository crashes on invalid call_type per Data Manifesto."""

        @dataclass
        class CallRow:
            call_id: str
            state_id: str
            call_index: int
            call_type: str
            status: str
            request_hash: str
            created_at: datetime
            request_ref: str | None = None
            response_hash: str | None = None
            response_ref: str | None = None
            error_json: str | None = None
            latency_ms: float | None = None

        db_row = CallRow(
            call_id="call-123",
            state_id="state-456",
            call_index=0,
            call_type="invalid_call_type",  # Invalid!
            status="success",
            request_hash="abc123",
            created_at=datetime.now(UTC),
        )

        repo = CallRepository(session=None)

        with pytest.raises(ValueError):
            repo.load(db_row)

    def test_load_crashes_on_invalid_status(self) -> None:
        """Repository crashes on invalid status per Data Manifesto."""

        @dataclass
        class CallRow:
            call_id: str
            state_id: str
            call_index: int
            call_type: str
            status: str
            request_hash: str
            created_at: datetime
            request_ref: str | None = None
            response_hash: str | None = None
            response_ref: str | None = None
            error_json: str | None = None
            latency_ms: float | None = None

        db_row = CallRow(
            call_id="call-123",
            state_id="state-456",
            call_index=0,
            call_type="llm",
            status="invalid_status",  # Invalid!
            request_hash="abc123",
            created_at=datetime.now(UTC),
        )

        repo = CallRepository(session=None)

        with pytest.raises(ValueError):
            repo.load(db_row)


class TestRoutingEventRepository:
    """Tests for RoutingEventRepository."""

    def test_load_converts_mode_to_enum(self) -> None:
        """Repository converts mode string to RoutingMode enum."""

        @dataclass
        class RoutingEventRow:
            event_id: str
            state_id: str
            edge_id: str
            routing_group_id: str
            ordinal: int
            mode: str  # String in DB
            created_at: datetime
            reason_hash: str | None = None
            reason_ref: str | None = None

        db_row = RoutingEventRow(
            event_id="evt-123",
            state_id="state-456",
            edge_id="edge-789",
            routing_group_id="group-1",
            ordinal=0,
            mode="move",  # String from DB
            created_at=datetime.now(UTC),
        )

        repo = RoutingEventRepository(session=None)
        event = repo.load(db_row)

        assert event.mode == RoutingMode.MOVE
        assert isinstance(event.mode, RoutingMode)

    def test_load_crashes_on_invalid_mode(self) -> None:
        """Repository crashes on invalid mode per Data Manifesto."""

        @dataclass
        class RoutingEventRow:
            event_id: str
            state_id: str
            edge_id: str
            routing_group_id: str
            ordinal: int
            mode: str
            created_at: datetime
            reason_hash: str | None = None
            reason_ref: str | None = None

        db_row = RoutingEventRow(
            event_id="evt-123",
            state_id="state-456",
            edge_id="edge-789",
            routing_group_id="group-1",
            ordinal=0,
            mode="teleport",  # Invalid!
            created_at=datetime.now(UTC),
        )

        repo = RoutingEventRepository(session=None)

        with pytest.raises(ValueError):
            repo.load(db_row)


class TestBatchRepository:
    """Tests for BatchRepository."""

    def test_load_converts_status_to_enum(self) -> None:
        """Repository converts status string to BatchStatus enum."""

        @dataclass
        class BatchRow:
            batch_id: str
            run_id: str
            aggregation_node_id: str
            attempt: int
            status: str  # String in DB
            created_at: datetime
            aggregation_state_id: str | None = None
            trigger_reason: str | None = None
            completed_at: datetime | None = None

        db_row = BatchRow(
            batch_id="batch-123",
            run_id="run-456",
            aggregation_node_id="node-1",
            attempt=1,
            status="executing",  # String from DB
            created_at=datetime.now(UTC),
        )

        repo = BatchRepository(session=None)
        batch = repo.load(db_row)

        assert batch.status == BatchStatus.EXECUTING
        assert isinstance(batch.status, BatchStatus)

    def test_load_crashes_on_invalid_status(self) -> None:
        """Repository crashes on invalid status per Data Manifesto."""

        @dataclass
        class BatchRow:
            batch_id: str
            run_id: str
            aggregation_node_id: str
            attempt: int
            status: str
            created_at: datetime
            aggregation_state_id: str | None = None
            trigger_reason: str | None = None
            completed_at: datetime | None = None

        db_row = BatchRow(
            batch_id="batch-123",
            run_id="run-456",
            aggregation_node_id="node-1",
            attempt=1,
            status="invalid_status",  # Invalid!
            created_at=datetime.now(UTC),
        )

        repo = BatchRepository(session=None)

        with pytest.raises(ValueError):
            repo.load(db_row)
