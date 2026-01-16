"""Tests for audit trail contracts."""

from datetime import UTC, datetime

from elspeth.contracts import (
    Determinism,
    Edge,
    ExportStatus,
    Node,
    NodeType,
    RoutingMode,
    Row,
    Run,
    RunStatus,
    Token,
    TokenParent,
)


class TestRun:
    """Tests for Run audit model."""

    def test_create_run_with_required_fields(self) -> None:
        """Can create Run with required fields and RunStatus enum."""
        now = datetime.now(UTC)
        run = Run(
            run_id="run-123",
            started_at=now,
            config_hash="abc123",
            settings_json="{}",
            canonical_version="1.0.0",
            status=RunStatus.RUNNING,
        )

        assert run.run_id == "run-123"
        assert run.started_at == now
        assert run.status == RunStatus.RUNNING
        assert run.completed_at is None
        assert run.export_status is None

    def test_run_status_must_be_enum(self) -> None:
        """Run.status must be RunStatus enum, not string."""
        run = Run(
            run_id="run-123",
            started_at=datetime.now(UTC),
            config_hash="abc123",
            settings_json="{}",
            canonical_version="1.0.0",
            status=RunStatus.COMPLETED,
        )

        # Status is enum type, not just string
        assert run.status == RunStatus.COMPLETED
        assert isinstance(run.status, RunStatus)
        assert run.status.value == "completed"

    def test_run_with_export_status(self) -> None:
        """Run can have ExportStatus enum."""
        run = Run(
            run_id="run-123",
            started_at=datetime.now(UTC),
            config_hash="abc123",
            settings_json="{}",
            canonical_version="1.0.0",
            status=RunStatus.COMPLETED,
            export_status=ExportStatus.PENDING,
        )

        assert run.export_status == ExportStatus.PENDING
        assert run.export_status.value == "pending"


class TestNode:
    """Tests for Node audit model."""

    def test_create_node_with_enum_fields(self) -> None:
        """Node requires NodeType and Determinism enums."""
        now = datetime.now(UTC)
        node = Node(
            node_id="node-123",
            run_id="run-456",
            plugin_name="csv_source",
            node_type=NodeType.SOURCE,
            plugin_version="1.0.0",
            determinism=Determinism.IO_READ,
            config_hash="abc123",
            config_json="{}",
            registered_at=now,
        )

        assert node.node_id == "node-123"
        assert node.node_type == NodeType.SOURCE
        assert node.determinism == Determinism.IO_READ

    def test_node_type_is_enum(self) -> None:
        """Node.node_type must be NodeType enum."""
        node = Node(
            node_id="node-123",
            run_id="run-456",
            plugin_name="gate",
            node_type=NodeType.GATE,
            plugin_version="1.0.0",
            determinism=Determinism.DETERMINISTIC,
            config_hash="abc123",
            config_json="{}",
            registered_at=datetime.now(UTC),
        )

        assert node.node_type == NodeType.GATE
        assert isinstance(node.node_type, NodeType)
        assert node.node_type.value == "gate"

    def test_determinism_is_enum(self) -> None:
        """Node.determinism must be Determinism enum."""
        node = Node(
            node_id="node-123",
            run_id="run-456",
            plugin_name="llm_transform",
            node_type=NodeType.TRANSFORM,
            plugin_version="1.0.0",
            determinism=Determinism.EXTERNAL_CALL,
            config_hash="abc123",
            config_json="{}",
            registered_at=datetime.now(UTC),
        )

        assert node.determinism == Determinism.EXTERNAL_CALL
        assert isinstance(node.determinism, Determinism)
        assert node.determinism.value == "external_call"


class TestEdge:
    """Tests for Edge audit model."""

    def test_create_edge_with_routing_mode(self) -> None:
        """Edge requires RoutingMode enum."""
        now = datetime.now(UTC)
        edge = Edge(
            edge_id="edge-123",
            run_id="run-456",
            from_node_id="node-1",
            to_node_id="node-2",
            label="continue",
            default_mode=RoutingMode.MOVE,
            created_at=now,
        )

        assert edge.edge_id == "edge-123"
        assert edge.default_mode == RoutingMode.MOVE

    def test_default_mode_is_enum(self) -> None:
        """Edge.default_mode must be RoutingMode enum."""
        edge = Edge(
            edge_id="edge-123",
            run_id="run-456",
            from_node_id="node-1",
            to_node_id="node-2",
            label="fork",
            default_mode=RoutingMode.COPY,
            created_at=datetime.now(UTC),
        )

        assert edge.default_mode == RoutingMode.COPY
        assert isinstance(edge.default_mode, RoutingMode)
        assert edge.default_mode.value == "copy"


class TestRow:
    """Tests for Row audit model."""

    def test_create_row(self) -> None:
        """Can create Row with all primitive fields."""
        now = datetime.now(UTC)
        row = Row(
            row_id="row-123",
            run_id="run-456",
            source_node_id="node-1",
            row_index=0,
            source_data_hash="abc123",
            created_at=now,
        )

        assert row.row_id == "row-123"
        assert row.row_index == 0
        assert row.source_data_ref is None

    def test_row_with_payload_ref(self) -> None:
        """Row can have source_data_ref for payload store."""
        row = Row(
            row_id="row-123",
            run_id="run-456",
            source_node_id="node-1",
            row_index=0,
            source_data_hash="abc123",
            created_at=datetime.now(UTC),
            source_data_ref="payload://abc123",
        )

        assert row.source_data_ref == "payload://abc123"


class TestToken:
    """Tests for Token audit model."""

    def test_create_token(self) -> None:
        """Can create Token with required fields."""
        now = datetime.now(UTC)
        token = Token(
            token_id="tok-123",
            row_id="row-456",
            created_at=now,
        )

        assert token.token_id == "tok-123"
        assert token.row_id == "row-456"
        assert token.fork_group_id is None
        assert token.branch_name is None

    def test_token_with_fork_fields(self) -> None:
        """Token can have fork/join metadata."""
        token = Token(
            token_id="tok-123",
            row_id="row-456",
            created_at=datetime.now(UTC),
            fork_group_id="fork-789",
            branch_name="sentiment",
            step_in_pipeline=3,
        )

        assert token.fork_group_id == "fork-789"
        assert token.branch_name == "sentiment"
        assert token.step_in_pipeline == 3


class TestTokenParent:
    """Tests for TokenParent audit model."""

    def test_create_token_parent(self) -> None:
        """Can create TokenParent for lineage tracking."""
        parent = TokenParent(
            token_id="tok-child",
            parent_token_id="tok-parent",
            ordinal=0,
        )

        assert parent.token_id == "tok-child"
        assert parent.parent_token_id == "tok-parent"
        assert parent.ordinal == 0

    def test_multi_parent_ordinal(self) -> None:
        """Ordinal supports multi-parent joins."""
        parent1 = TokenParent(
            token_id="tok-joined",
            parent_token_id="tok-a",
            ordinal=0,
        )
        parent2 = TokenParent(
            token_id="tok-joined",
            parent_token_id="tok-b",
            ordinal=1,
        )

        assert parent1.ordinal == 0
        assert parent2.ordinal == 1
