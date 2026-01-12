# tests/core/landscape/test_models.py
"""Tests for Landscape database models."""

from datetime import datetime, timezone

import pytest


class TestRunModel:
    """Run table model."""

    def test_create_run(self) -> None:
        from elspeth.core.landscape.models import Run

        run = Run(
            run_id="run-001",
            started_at=datetime.now(timezone.utc),
            config_hash="abc123",
            settings_json="{}",
            canonical_version="sha256-rfc8785-v1",
            status="running",
        )
        assert run.run_id == "run-001"
        assert run.status == "running"


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
            config_hash="def456",
            config_json="{}",
            registered_at=datetime.now(timezone.utc),
        )
        assert node.node_type == "source"


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
            created_at=datetime.now(timezone.utc),
        )
        assert row.row_index == 0


class TestTokenModel:
    """Token table model."""

    def test_create_token(self) -> None:
        from elspeth.core.landscape.models import Token

        token = Token(
            token_id="token-001",
            row_id="row-001",
            created_at=datetime.now(timezone.utc),
        )
        assert token.token_id == "token-001"
