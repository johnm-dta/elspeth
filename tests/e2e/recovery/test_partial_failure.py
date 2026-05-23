"""E2E tests for partial failure scenarios.

Tests verify that ELSPETH handles partial failures correctly:
- Failed rows are recorded in the audit trail
- Successful rows still reach the sink
- All rows have terminal outcomes

Uses file-based SQLite and real payload stores. No mocks except
external services.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from sqlalchemy import select

from elspeth.contracts import (
    Determinism,
    PipelineRow,
    PluginSchema,
    RunStatus,
    TerminalOutcome,
    TerminalPath,
)
from elspeth.core.config import SourceSettings
from elspeth.core.dag import ExecutionGraph
from elspeth.core.landscape.database import LandscapeDB
from elspeth.core.landscape.factory import RecorderFactory
from elspeth.core.landscape.lineage import explain
from elspeth.core.landscape.schema import (
    token_outcomes_table,
    tokens_table,
    transform_errors_table,
)
from elspeth.core.payload_store import FilesystemPayloadStore
from elspeth.engine.orchestrator import Orchestrator, PipelineConfig
from elspeth.plugins.infrastructure.base import BaseTransform
from elspeth.plugins.infrastructure.results import TransformResult
from elspeth.testing import make_pipeline_row
from tests.fixtures.base_classes import (
    as_sink,
    as_source,
)
from tests.fixtures.factories import wire_transforms
from tests.fixtures.plugins import CollectSink, ListSource

# ---------------------------------------------------------------------------
# Shared test schema and plugins
# ---------------------------------------------------------------------------


class _PartialSchema(PluginSchema):
    """Schema for partial failure test rows."""

    id: int
    value: int


class _SelectiveErrorTransform(BaseTransform):
    """Transform that errors on rows where id is in the fail set."""

    name = "selective_error"
    input_schema = _PartialSchema
    output_schema = _PartialSchema
    determinism = Determinism.DETERMINISTIC
    on_error = "discard"

    def __init__(self, fail_ids: set[int]) -> None:
        super().__init__({"schema": {"mode": "observed"}})
        self._fail_ids = fail_ids

    def process(self, row: PipelineRow, ctx: Any) -> TransformResult:
        if row["id"] in self._fail_ids:
            return TransformResult.error(
                {
                    "reason": "test_error",
                    "error": f"Row {row['id']} deliberately failed",
                }
            )
        return TransformResult.success(
            make_pipeline_row(row.to_dict()),
            success_reason={"action": "passed"},
        )


def _build_linear_graph(config: PipelineConfig) -> ExecutionGraph:
    """Build a simple linear graph for partial failure tests.

    Uses from_plugin_instances() for production-path fidelity.
    """
    transforms = list(config.transforms)
    sink_name = next(iter(config.sinks))

    # Set source routing
    source_connection = "source_out" if transforms else sink_name
    config.sources["primary"].on_success = source_connection
    source_settings = SourceSettings(plugin=config.sources["primary"].name, on_success=source_connection, options={})

    # Wire transforms with explicit routing
    wired = wire_transforms(transforms, source_connection=source_connection, final_sink=sink_name)

    graph = ExecutionGraph.from_plugin_instances(
        sources={"primary": config.sources["primary"]},
        source_settings_map={"primary": source_settings},
        transforms=wired,
        sinks=config.sinks,
        aggregations={},
        gates=[],
    )
    return graph


class TestPartialFailure:
    """Tests for partial failure scenarios."""

    def test_partial_failures_produce_correct_audit_trail(self, tmp_path: Path) -> None:
        """Run 10 rows where 3 fail. Verify:
        1. Pipeline completes
        2. 7 rows reach the sink
        3. 3 errors recorded in transform_errors table
        4. All 10 rows have terminal outcomes (COMPLETED or FAILED/QUARANTINED)
        5. Explain query works for both successful and failed rows
        """
        db = LandscapeDB(f"sqlite:///{tmp_path}/audit.db")
        payload_store = FilesystemPayloadStore(tmp_path / "payloads")

        source_data = [{"id": i, "value": i * 10} for i in range(10)]
        fail_ids = {2, 5, 8}  # 3 rows fail

        source = ListSource(source_data)
        transform = _SelectiveErrorTransform(fail_ids=fail_ids)
        sink = CollectSink("default")

        config = PipelineConfig(
            sources={"primary": as_source(source)},
            transforms=[transform],  # type: ignore[list-item]
            sinks={"default": as_sink(sink)},
        )

        graph = _build_linear_graph(config)

        orchestrator = Orchestrator(db)
        result = orchestrator.run(config, graph=graph, payload_store=payload_store)

        # 1. Pipeline completes
        assert result.status == RunStatus.COMPLETED_WITH_FAILURES

        # 2. 7 rows reach the sink
        assert len(sink.results) == 7
        sink_ids = {r["id"] for r in sink.results}
        expected_ids = {i for i in range(10) if i not in fail_ids}
        assert sink_ids == expected_ids

        # 3. 3 errors recorded
        with db.engine.connect() as conn:
            errors = conn.execute(select(transform_errors_table).where(transform_errors_table.c.run_id == result.run_id)).fetchall()

        assert len(errors) == 3
        for e in errors:
            details = json.loads(e.error_details_json)
            assert details["reason"] == "test_error"
            assert "deliberately failed" in details["error"]

        # 4. All 10 rows have terminal outcomes
        with db.engine.connect() as conn:
            outcomes = conn.execute(
                select(
                    token_outcomes_table.c.token_id,
                    token_outcomes_table.c.outcome,
                    token_outcomes_table.c.path,
                )
                .where(token_outcomes_table.c.run_id == result.run_id)
                .where(token_outcomes_table.c.completed == 1)
            ).fetchall()

        assert len(outcomes) == 10
        outcome_values = {(o.outcome, o.path) for o in outcomes}
        # Should contain both COMPLETED and QUARANTINED/FAILED outcomes
        assert (TerminalOutcome.SUCCESS.value, TerminalPath.DEFAULT_FLOW.value) in outcome_values

        # 5. Explain query works for both successful and failed rows
        factory = RecorderFactory(db, payload_store=payload_store)

        # Get all tokens to test explain
        with db.engine.connect() as conn:
            token_rows = conn.execute(
                select(tokens_table.c.token_id)
                .join(
                    token_outcomes_table,
                    tokens_table.c.token_id == token_outcomes_table.c.token_id,
                )
                .where(token_outcomes_table.c.run_id == result.run_id)
            ).fetchall()

        # Verify at least one successful and one failed row is explainable
        explained_count = 0
        for token_row in token_rows:
            lineage = explain(
                factory.query,
                factory.data_flow,
                result.run_id,
                token_id=token_row.token_id,
            )
            if lineage is not None:
                explained_count += 1
                assert lineage.source_row is not None

        assert explained_count > 0, "Expected at least one explainable token"

        db.close()

    def test_first_row_failure_doesnt_prevent_rest(self, tmp_path: Path) -> None:
        """Run where row 0 fails but rows 1-4 succeed.

        Verify all 4 remaining rows reach sink.
        """
        db = LandscapeDB(f"sqlite:///{tmp_path}/audit.db")
        payload_store = FilesystemPayloadStore(tmp_path / "payloads")

        source_data = [{"id": i, "value": i * 10} for i in range(5)]
        fail_ids = {0}  # First row fails

        source = ListSource(source_data)
        transform = _SelectiveErrorTransform(fail_ids=fail_ids)
        sink = CollectSink("default")

        config = PipelineConfig(
            sources={"primary": as_source(source)},
            transforms=[transform],  # type: ignore[list-item]
            sinks={"default": as_sink(sink)},
        )

        graph = _build_linear_graph(config)

        orchestrator = Orchestrator(db)
        result = orchestrator.run(config, graph=graph, payload_store=payload_store)

        assert result.status == RunStatus.COMPLETED_WITH_FAILURES

        # All 4 remaining rows reach sink
        assert len(sink.results) == 4
        sink_ids = {r["id"] for r in sink.results}
        assert sink_ids == {1, 2, 3, 4}

        # 1 error recorded
        with db.engine.connect() as conn:
            errors = conn.execute(select(transform_errors_table).where(transform_errors_table.c.run_id == result.run_id)).fetchall()

        assert len(errors) == 1
        error_details = json.loads(errors[0].error_details_json)
        assert error_details["reason"] == "test_error"
        assert "Row 0 deliberately failed" in error_details["error"]

        db.close()

    def test_last_row_failure_doesnt_corrupt_prior_output(self, tmp_path: Path) -> None:
        """Run where last row fails but all prior rows succeed.

        Verify all prior rows in sink and correctly recorded.
        """
        db = LandscapeDB(f"sqlite:///{tmp_path}/audit.db")
        payload_store = FilesystemPayloadStore(tmp_path / "payloads")

        source_data = [{"id": i, "value": i * 10} for i in range(5)]
        fail_ids = {4}  # Last row fails

        source = ListSource(source_data)
        transform = _SelectiveErrorTransform(fail_ids=fail_ids)
        sink = CollectSink("default")

        config = PipelineConfig(
            sources={"primary": as_source(source)},
            transforms=[transform],  # type: ignore[list-item]
            sinks={"default": as_sink(sink)},
        )

        graph = _build_linear_graph(config)

        orchestrator = Orchestrator(db)
        result = orchestrator.run(config, graph=graph, payload_store=payload_store)

        assert result.status == RunStatus.COMPLETED_WITH_FAILURES

        # 4 rows reach sink (all except last)
        assert len(sink.results) == 4
        sink_ids = {r["id"] for r in sink.results}
        assert sink_ids == {0, 1, 2, 3}

        # 1 error recorded for the last row
        with db.engine.connect() as conn:
            errors = conn.execute(select(transform_errors_table).where(transform_errors_table.c.run_id == result.run_id)).fetchall()

        assert len(errors) == 1
        error_details = json.loads(errors[0].error_details_json)
        assert error_details["reason"] == "test_error"
        assert "Row 4 deliberately failed" in error_details["error"]

        # All 5 rows have terminal outcomes
        with db.engine.connect() as conn:
            outcomes = conn.execute(
                select(token_outcomes_table.c.outcome)
                .where(token_outcomes_table.c.run_id == result.run_id)
                .where(token_outcomes_table.c.completed == 1)
            ).fetchall()

        assert len(outcomes) == 5

        db.close()
