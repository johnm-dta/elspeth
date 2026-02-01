# tests/core/landscape/test_models.py
"""Tests for Landscape audit contracts instantiation.

NOTE: These tests were previously testing elspeth.core.landscape.models
which has been removed. Tests now verify the canonical contracts in
elspeth.contracts.audit behave correctly.
"""

from datetime import UTC, datetime

from elspeth.contracts import (
    Determinism,
    Node,
    NodeType,
    Row,
    Run,
    RunStatus,
    Token,
)


class TestRunModel:
    """Run audit contract."""

    def test_create_run(self) -> None:
        run = Run(
            run_id="run-001",
            started_at=datetime.now(UTC),
            config_hash="abc123",
            settings_json="{}",
            canonical_version="sha256-rfc8785-v1",
            status=RunStatus.RUNNING,
        )
        assert run.run_id == "run-001"
        assert run.status == RunStatus.RUNNING


class TestNodeModel:
    """Node audit contract."""

    def test_create_node(self) -> None:
        node = Node(
            node_id="node-001",
            run_id="run-001",
            plugin_name="csv",
            node_type=NodeType.SOURCE,
            plugin_version="1.0.0",
            determinism=Determinism.DETERMINISTIC,
            config_hash="def456",
            config_json="{}",
            registered_at=datetime.now(UTC),
        )
        assert node.node_type == NodeType.SOURCE
        assert node.determinism == Determinism.DETERMINISTIC


class TestRowModel:
    """Row audit contract."""

    def test_create_row(self) -> None:
        row = Row(
            row_id="row-001",
            run_id="run-001",
            source_node_id="source-001",
            row_index=0,
            source_data_hash="ghi789",
            created_at=datetime.now(UTC),
        )
        assert row.row_index == 0


class TestTokenModel:
    """Token audit contract."""

    def test_create_token(self) -> None:
        token = Token(
            token_id="token-001",
            row_id="row-001",
            created_at=datetime.now(UTC),
        )
        assert token.token_id == "token-001"
