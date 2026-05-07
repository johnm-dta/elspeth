# tests/integration/pipeline/test_resume_schema_required.py
"""Integration tests for Bug #4: Type Degradation on Resume (Schema Required).

These tests verify that resume requires source schema for type fidelity
and fails early with a clear error if schema is not available.
"""

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from elspeth.contracts import Determinism, NodeType, RunStatus
from elspeth.contracts.contexts import LifecycleContext, SourceContext
from elspeth.contracts.errors import AuditIntegrityError
from elspeth.contracts.results import SourceRow
from elspeth.core.checkpoint import CheckpointManager
from elspeth.core.dag import ExecutionGraph
from elspeth.core.landscape.database import LandscapeDB
from elspeth.core.payload_store import FilesystemPayloadStore
from elspeth.plugins.infrastructure.base import BaseSource
from elspeth.testing import make_contract
from tests.fixtures.landscape import make_factory


class SourceWithoutSchema(BaseSource):
    """Test source that does NOT provide _schema_class attribute."""

    name = "source_without_schema"
    plugin_version = "1.0.0"
    determinism = Determinism.DETERMINISTIC

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(config)
        # Intentionally do NOT set self._schema_class
        # This simulates old source plugins that don't support type fidelity

    def load(self, ctx: SourceContext) -> Iterator[SourceRow]:
        """Yield test rows."""
        yield SourceRow.valid({"id": 1, "value": "row1"}, contract=make_contract({"id": 1, "value": "row1"}))
        yield SourceRow.valid({"id": 2, "value": "row2"}, contract=make_contract({"id": 2, "value": "row2"}))

    def on_start(self, ctx: LifecycleContext) -> None:
        pass

    def on_complete(self, ctx: LifecycleContext) -> None:
        pass

    def close(self) -> None:
        pass


class TestResumeSchemaRequired:
    """Integration tests for required schema on resume."""

    @pytest.fixture
    def test_env(self, tmp_path: Path) -> dict[str, Any]:
        """Set up test environment with database and payload store."""
        db = LandscapeDB(f"sqlite:///{tmp_path}/test.db")
        payload_store = FilesystemPayloadStore(tmp_path / "payloads")
        checkpoint_mgr = CheckpointManager(db)
        factory = make_factory(db)

        return {
            "db": db,
            "payload_store": payload_store,
            "checkpoint_manager": checkpoint_mgr,
            "factory": factory,
            "tmp_path": tmp_path,
        }

    @pytest.fixture
    def simple_graph(self) -> ExecutionGraph:
        """Create a simple source -> sink graph."""
        graph = ExecutionGraph()
        schema_config = {"schema": {"mode": "observed"}}
        graph.add_node("source", node_type=NodeType.SOURCE, plugin_name="source_without_schema", config=schema_config)
        graph.add_node("sink", node_type=NodeType.SINK, plugin_name="csv", config=schema_config)
        graph.add_edge("source", "sink", label="continue")
        return graph

    def test_resume_fails_early_without_source_schema(
        self,
        test_env: dict[str, Any],
        simple_graph: ExecutionGraph,
    ) -> None:
        """Verify resume fails early if source doesn't provide schema.

        Scenario:
        1. Create run WITHOUT storing source schema (simulating old run)
        2. Attempt to retrieve schema via recorder.get_source_schema()
        3. Verify: Fails with clear error message (Bug #4 fix)

        This is Bug #4 fix: resume REQUIRES schema to preserve type fidelity.
        Without schema, resumed rows would have degraded types (str instead of
        datetime/Decimal), violating the Tier 2 pipeline data trust model.
        """
        factory = test_env["factory"]
        checkpoint_mgr = test_env["checkpoint_manager"]
        db = test_env["db"]

        # 1. Create run WITHOUT storing source schema (simulating old run)
        run = factory.run_lifecycle.begin_run(config={}, canonical_version="v1")

        # Register nodes using raw SQL
        from datetime import UTC, datetime

        from elspeth.core.landscape.schema import nodes_table

        now = datetime.now(UTC)
        with db.engine.connect() as conn:
            # Source node
            conn.execute(
                nodes_table.insert().values(
                    node_id="source",
                    run_id=run.run_id,
                    plugin_name="source_without_schema",
                    node_type=NodeType.SOURCE,
                    plugin_version="1.0",
                    determinism=Determinism.DETERMINISTIC,
                    config_hash="test",
                    config_json="{}",
                    registered_at=now,
                )
            )

            # Sink node
            conn.execute(
                nodes_table.insert().values(
                    node_id="sink",
                    run_id=run.run_id,
                    plugin_name="csv",
                    node_type=NodeType.SINK,
                    plugin_version="1.0",
                    determinism=Determinism.IO_WRITE,
                    config_hash="test",
                    config_json="{}",
                    registered_at=now,
                )
            )

            conn.commit()

        # 2. Create rows and checkpoint
        row_data = {"id": 1, "value": "test"}
        row = factory.data_flow.create_row(
            run_id=run.run_id,
            source_node_id="source",
            row_index=0,
            data=row_data,
        )
        token = factory.data_flow.create_token(row_id=row.row_id)

        # Create checkpoint
        checkpoint_mgr.create_checkpoint(
            run_id=run.run_id,
            token_id=token.token_id,
            node_id="source",
            sequence_number=0,
            graph=simple_graph,
        )

        # Mark run as failed (so it can be resumed)
        factory.run_lifecycle.complete_run(run.run_id, status=RunStatus.FAILED)

        # 3. Test production code: get_source_schema() should raise AuditIntegrityError
        with pytest.raises(AuditIntegrityError) as exc_info:
            factory.run_lifecycle.get_source_schema(run.run_id)

        # 4. Verify error message is clear and helpful
        error_msg = str(exc_info.value)
        assert run.run_id in error_msg
        assert "has no source schema stored" in error_msg
        assert "Cannot resume without schema" in error_msg
        assert "type fidelity" in error_msg

        # SUCCESS: Production code catches missing schema early with clear error message
