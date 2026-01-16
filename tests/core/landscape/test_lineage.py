"""Tests for lineage query functionality."""

from datetime import UTC, datetime


class TestLineageResult:
    """Tests for LineageResult data structure."""

    def test_lineage_result_exists(self) -> None:
        """LineageResult can be imported."""
        from elspeth.core.landscape.lineage import LineageResult

        assert LineageResult is not None

    def test_lineage_result_fields(self) -> None:
        """LineageResult has expected fields."""
        from elspeth.contracts import RowLineage
        from elspeth.core.landscape.lineage import LineageResult
        from elspeth.core.landscape.models import Token

        now = datetime.now(UTC)
        result = LineageResult(
            token=Token(
                token_id="t1",
                row_id="r1",
                created_at=now,
            ),
            source_row=RowLineage(
                row_id="r1",
                run_id="run1",
                source_node_id="src",
                row_index=0,
                source_data_hash="abc",
                created_at=now,
                source_data={"field": "value"},
                payload_available=True,
            ),
            node_states=[],
            routing_events=[],
            calls=[],
            parent_tokens=[],
        )
        assert result.token.token_id == "t1"
        assert result.source_row.row_id == "r1"
        assert result.source_row.payload_available is True


class TestExplainFunction:
    """Tests for explain() lineage query function."""

    def test_explain_exists(self) -> None:
        """explain function can be imported."""
        from elspeth.core.landscape.lineage import explain

        assert callable(explain)

    def test_explain_returns_lineage_result(self) -> None:
        """explain returns LineageResult."""
        from elspeth.core.landscape.database import LandscapeDB
        from elspeth.core.landscape.lineage import LineageResult, explain
        from elspeth.core.landscape.recorder import LandscapeRecorder

        # Setup: create a minimal run with one row
        db = LandscapeDB.in_memory()
        recorder = LandscapeRecorder(db)

        run = recorder.begin_run(config={}, canonical_version="sha256-rfc8785-v1")
        node = recorder.register_node(
            run_id=run.run_id,
            plugin_name="csv",
            node_type="source",
            plugin_version="1.0",
            config={},
        )
        row = recorder.create_row(
            run_id=run.run_id,
            source_node_id=node.node_id,
            row_index=0,
            data={"id": 1},
        )
        token = recorder.create_token(row_id=row.row_id)
        recorder.complete_run(run.run_id, status="completed")

        # Query lineage
        result = explain(recorder, run_id=run.run_id, token_id=token.token_id)

        assert isinstance(result, LineageResult)
        assert result.token.token_id == token.token_id
        assert result.source_row.row_id == row.row_id

    def test_explain_by_row_id(self) -> None:
        """explain can query by row_id instead of token_id."""
        from elspeth.core.landscape.database import LandscapeDB
        from elspeth.core.landscape.lineage import explain
        from elspeth.core.landscape.recorder import LandscapeRecorder

        db = LandscapeDB.in_memory()
        recorder = LandscapeRecorder(db)

        run = recorder.begin_run(config={}, canonical_version="sha256-rfc8785-v1")
        node = recorder.register_node(
            run_id=run.run_id,
            plugin_name="csv",
            node_type="source",
            plugin_version="1.0",
            config={},
        )
        row = recorder.create_row(
            run_id=run.run_id,
            source_node_id=node.node_id,
            row_index=0,
            data={"id": 1},
        )
        _token = recorder.create_token(row_id=row.row_id)
        recorder.complete_run(run.run_id, status="completed")

        # Query by row_id (token must exist but we don't use it directly)
        result = explain(recorder, run_id=run.run_id, row_id=row.row_id)

        assert result is not None
        assert result.source_row.row_id == row.row_id

    def test_explain_nonexistent_returns_none(self) -> None:
        """explain returns None for nonexistent token."""
        from elspeth.core.landscape.database import LandscapeDB
        from elspeth.core.landscape.lineage import explain
        from elspeth.core.landscape.recorder import LandscapeRecorder

        db = LandscapeDB.in_memory()
        recorder = LandscapeRecorder(db)

        result = explain(recorder, run_id="fake", token_id="fake")
        assert result is None
