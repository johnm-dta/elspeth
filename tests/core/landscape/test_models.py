# tests/core/landscape/test_models.py
"""Tests for Landscape database models."""

from datetime import UTC, datetime


class TestRunModel:
    """Run table model."""

    def test_create_run(self) -> None:
        from elspeth.core.landscape.models import Run, RunStatus

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
    """Node table model."""

    def test_create_node(self) -> None:
        from elspeth.core.landscape.models import Node

        node = Node(
            node_id="node-001",
            run_id="run-001",
            plugin_name="csv",
            node_type="source",
            plugin_version="1.0.0",
            determinism="deterministic",
            config_hash="def456",
            config_json="{}",
            registered_at=datetime.now(UTC),
        )
        assert node.node_type == "source"
        assert node.determinism == "deterministic"


class TestRowModel:
    """Row table model."""

    def test_create_row(self) -> None:
        from elspeth.core.landscape.models import Row

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
    """Token table model."""

    def test_create_token(self) -> None:
        from elspeth.core.landscape.models import Token

        token = Token(
            token_id="token-001",
            row_id="row-001",
            created_at=datetime.now(UTC),
        )
        assert token.token_id == "token-001"
