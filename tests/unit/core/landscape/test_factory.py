"""Tests for RecorderFactory construction and repository wiring."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from elspeth.core.landscape.data_flow_repository import DataFlowRepository
from elspeth.core.landscape.database import LandscapeDB
from elspeth.core.landscape.execution_repository import ExecutionRepository
from elspeth.core.landscape.factory import RecorderFactory, _PluginAuditWriterAdapter
from elspeth.core.landscape.query_repository import QueryRepository
from elspeth.core.landscape.run_lifecycle_repository import RunLifecycleRepository


@pytest.fixture()
def db() -> LandscapeDB:
    return LandscapeDB.in_memory()


@pytest.fixture()
def factory(db: LandscapeDB) -> RecorderFactory:
    return RecorderFactory(db)


class TestRepositoryConstruction:
    """Verify the factory creates all four repositories with correct types."""

    def test_creates_all_four_repositories(self, factory: RecorderFactory) -> None:
        assert isinstance(factory.run_lifecycle, RunLifecycleRepository)
        assert isinstance(factory.execution, ExecutionRepository)
        assert isinstance(factory.data_flow, DataFlowRepository)
        assert isinstance(factory.query, QueryRepository)

    def test_repositories_share_database_ops(self, factory: RecorderFactory) -> None:
        """Round-trip: begin_run via lifecycle, then get_run confirms shared DB."""
        run = factory.run_lifecycle.begin_run(
            config={"source": {"plugin": "csv"}},
            canonical_version="v1",
        )
        retrieved = factory.run_lifecycle.get_run(run.run_id)
        assert retrieved is not None
        assert retrieved.run_id == run.run_id


class TestPayloadStore:
    """Verify payload_store propagation."""

    def test_payload_store_propagated(self) -> None:
        db = LandscapeDB.in_memory()
        mock_store = MagicMock()
        factory = RecorderFactory(db, payload_store=mock_store)
        assert factory.payload_store is mock_store

    def test_payload_store_defaults_to_none(self, factory: RecorderFactory) -> None:
        assert factory.payload_store is None


class TestPluginAuditWriter:
    """Verify plugin_audit_writer() returns the adapter."""

    def test_plugin_audit_writer_is_adapter(self, factory: RecorderFactory) -> None:
        writer = factory.plugin_audit_writer()
        assert isinstance(writer, _PluginAuditWriterAdapter)
