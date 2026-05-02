"""Regression tests for MCP report analyzer functions.

Bug fixes covered:
- Phase 0 fix #10: MCP Mermaid non-unique IDs (node_id[:8] truncation)
- P1-2026-02-14: get_performance_report truncates node_id
- P1-2026-02-14: get_outcome_analysis returns is_terminal as DB integer

Coverage additions:
- get_error_analysis: corruption guard, validation/transform grouping, sample data, run not found
"""

from __future__ import annotations

import json
import re
from typing import Any, cast
from unittest.mock import MagicMock

import pytest

from elspeth.contracts.enums import CallStatus, NodeType, RoutingMode
from elspeth.contracts.errors import AuditIntegrityError
from elspeth.mcp.analyzers.reports import (
    get_dag_structure,
    get_error_analysis,
    get_llm_usage_report,
    get_outcome_analysis,
    get_performance_report,
)
from elspeth.mcp.types import ErrorAnalysisReport, ErrorResult, LLMUsageReport


def _make_node(node_id: str, plugin_name: str, node_type: NodeType) -> MagicMock:
    """Create a mock node row."""
    node = MagicMock()
    node.node_id = node_id
    node.plugin_name = plugin_name
    node.node_type = node_type
    node.sequence_in_pipeline = 0
    return node


def _make_edge(from_id: str, to_id: str, label: str = "continue", mode: RoutingMode = RoutingMode.MOVE) -> MagicMock:
    """Create a mock edge row."""
    edge = MagicMock()
    edge.from_node_id = from_id
    edge.to_node_id = to_id
    edge.label = label
    edge.default_mode = mode
    return edge


class TestMermaidUniqueNodeIDs:
    """Verify Mermaid diagrams have unique node IDs even with similar prefixes."""

    def test_similar_prefixed_nodes_get_unique_mermaid_ids(self) -> None:
        """Nodes like "transform_classifier" and "transform_mapper" must
        produce different Mermaid node IDs. Before the fix, both would
        truncate to "transfor" and collide.
        """
        nodes = [
            _make_node("source_csv_abc123", "csv", NodeType.SOURCE),
            _make_node("transform_classifier_def456", "llm_classifier", NodeType.TRANSFORM),
            _make_node("transform_mapper_ghi789", "field_mapper", NodeType.TRANSFORM),
            _make_node("transform_truncate_jkl012", "truncate", NodeType.TRANSFORM),
            _make_node("sink_output_mno345", "csv_sink", NodeType.SINK),
        ]

        edges = [
            _make_edge("source_csv_abc123", "transform_classifier_def456"),
            _make_edge("transform_classifier_def456", "transform_mapper_ghi789"),
            _make_edge("transform_mapper_ghi789", "transform_truncate_jkl012"),
            _make_edge("transform_truncate_jkl012", "sink_output_mno345", "on_success"),
        ]

        db = MagicMock()
        factory = MagicMock()
        factory.data_flow.get_nodes.return_value = nodes
        factory.data_flow.get_edges.return_value = edges

        result = get_dag_structure(db, factory, "run-123")

        # Should not be an error
        assert "error" not in result

        mermaid = result["mermaid"]

        # Extract all node definition IDs (e.g., N0, N1, etc.)
        node_defs = re.findall(r'^\s+(\S+)\["', mermaid, re.MULTILINE)

        # All node IDs must be unique
        assert len(node_defs) == len(set(node_defs)), f"Mermaid diagram has duplicate node IDs: {node_defs}"

        # Verify we got sequential aliases (N0, N1, ...)
        for i, node_def in enumerate(node_defs):
            assert node_def == f"N{i}", f"Expected sequential alias N{i}, got {node_def}"


class TestPerformanceReportNodeId:
    """Verify get_performance_report returns full node_id.

    Regression: P1-2026-02-14 — node_id was truncated to 12 chars + "...",
    making node identity ambiguous and preventing exact cross-referencing.
    """

    def test_node_id_is_not_truncated(self) -> None:
        """Node IDs in performance report must be the full canonical ID."""
        full_node_id = "transform_llm_classifier_abc123def456"

        # Mock a stats row with full node_id
        stats_row = MagicMock()
        stats_row.node_id = full_node_id
        stats_row.plugin_name = "llm_classifier"
        stats_row.node_type = "transform"
        stats_row.executions = 10
        stats_row.avg_ms = 150.0
        stats_row.min_ms = 50.0
        stats_row.max_ms = 500.0
        stats_row.total_ms = 1500.0

        # Mock database connection and query results
        mock_conn = MagicMock()
        mock_conn.execute.return_value.fetchall.side_effect = [
            [stats_row],  # stats_query
            [],  # failed_query
        ]

        db = MagicMock()
        db.connection.return_value.__enter__ = MagicMock(return_value=mock_conn)
        db.connection.return_value.__exit__ = MagicMock(return_value=False)

        factory = MagicMock()
        factory.run_lifecycle.get_run.return_value = MagicMock(
            started_at=None,
            completed_at=None,
            status=MagicMock(value="completed"),
        )

        result = get_performance_report(db, factory, "run-123")

        assert "error" not in result
        node_perf = result["node_performance"]
        assert len(node_perf) == 1
        # The key assertion: full node_id, no truncation
        assert node_perf[0]["node_id"] == full_node_id
        assert "..." not in node_perf[0]["node_id"]


class TestOutcomeAnalysisIsTerminal:
    """Verify get_outcome_analysis returns is_terminal as bool.

    Regression: P1-2026-02-14 — is_terminal was returned as DB integer (0/1)
    instead of bool, violating the OutcomeDistributionEntry contract.
    """

    def test_is_terminal_is_bool_not_int(self) -> None:
        """is_terminal must be a Python bool, not an integer 0/1."""
        # Mock outcome rows with integer is_terminal (as from SQLite)
        outcome_row_terminal = MagicMock()
        outcome_row_terminal.outcome = "COMPLETED"
        outcome_row_terminal.is_terminal = 1  # DB integer
        outcome_row_terminal.count = 10

        outcome_row_non_terminal = MagicMock()
        outcome_row_non_terminal.outcome = "BUFFERED"
        outcome_row_non_terminal.is_terminal = 0  # DB integer
        outcome_row_non_terminal.count = 3

        # Mock sink rows (empty)
        # Mock fork/join counts
        mock_conn = MagicMock()
        mock_conn.execute.return_value.fetchall.side_effect = [
            [outcome_row_terminal, outcome_row_non_terminal],  # outcome_dist
            [],  # sink_dist
        ]
        mock_conn.execute.return_value.scalar.side_effect = [0, 0]  # fork_count, join_count

        # Need to handle multiple calls to execute() returning different results
        call_count = 0

        def side_effect_execute(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            result = MagicMock()
            if call_count == 1:
                result.fetchall.return_value = [outcome_row_terminal, outcome_row_non_terminal]
            elif call_count == 2:
                result.fetchall.return_value = []
            elif call_count in (3, 4):
                result.scalar.return_value = 0
            return result

        mock_conn.execute = side_effect_execute

        db = MagicMock()
        db.connection.return_value.__enter__ = MagicMock(return_value=mock_conn)
        db.connection.return_value.__exit__ = MagicMock(return_value=False)

        factory = MagicMock()
        factory.run_lifecycle.get_run.return_value = MagicMock(
            started_at=None,
            completed_at=None,
            status=MagicMock(value="completed"),
        )

        result = get_outcome_analysis(db, factory, "run-123")

        assert "error" not in result
        outcomes = result["outcome_distribution"]
        assert len(outcomes) == 2

        for outcome in outcomes:
            assert isinstance(outcome["is_terminal"], bool), f"is_terminal must be bool, got {type(outcome['is_terminal']).__name__}"

        # Verify correct boolean values
        terminal = next(o for o in outcomes if o["outcome"] == "COMPLETED")
        non_terminal = next(o for o in outcomes if o["outcome"] == "BUFFERED")
        assert terminal["is_terminal"] is True
        assert non_terminal["is_terminal"] is False

    def test_routed_on_error_outcome_preserved_in_distribution(self) -> None:
        """ROUTED_ON_ERROR (elspeth-5069612f3c) is surfaced as its own bucket.

        MCP does NOT split historical "routed" rows into success/failure
        buckets — the runbook/ADR handles the upgrade-boundary limitation.
        ROUTED_ON_ERROR rows pass through as the "routed_on_error" bucket
        (terminal), separate from the legacy "routed" bucket.
        """
        # Mock terminal outcomes including ROUTED_ON_ERROR
        outcome_row_completed = MagicMock()
        outcome_row_completed.outcome = "COMPLETED"
        outcome_row_completed.is_terminal = 1
        outcome_row_completed.count = 5

        outcome_row_routed = MagicMock()
        outcome_row_routed.outcome = "ROUTED"
        outcome_row_routed.is_terminal = 1
        outcome_row_routed.count = 3

        outcome_row_routed_on_error = MagicMock()
        outcome_row_routed_on_error.outcome = "ROUTED_ON_ERROR"
        outcome_row_routed_on_error.is_terminal = 1
        outcome_row_routed_on_error.count = 2

        mock_conn = MagicMock()
        call_count = 0

        def side_effect_execute(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            result = MagicMock()
            if call_count == 1:
                result.fetchall.return_value = [
                    outcome_row_completed,
                    outcome_row_routed,
                    outcome_row_routed_on_error,
                ]
            elif call_count == 2:
                result.fetchall.return_value = []
            elif call_count in (3, 4):
                result.scalar.return_value = 0
            return result

        mock_conn.execute = side_effect_execute

        db = MagicMock()
        db.connection.return_value.__enter__ = MagicMock(return_value=mock_conn)
        db.connection.return_value.__exit__ = MagicMock(return_value=False)

        factory = MagicMock()
        factory.run_lifecycle.get_run.return_value = MagicMock(
            started_at=None,
            completed_at=None,
            status=MagicMock(value="completed"),
        )

        result = get_outcome_analysis(db, factory, "run-123")

        assert "error" not in result
        outcomes = result["outcome_distribution"]
        outcome_keys = {o["outcome"] for o in outcomes}
        assert "ROUTED_ON_ERROR" in outcome_keys, "ROUTED_ON_ERROR must surface as its own outcome_distribution bucket"
        # Legacy ROUTED bucket is preserved separately.
        assert "ROUTED" in outcome_keys
        routed_on_error = next(o for o in outcomes if o["outcome"] == "ROUTED_ON_ERROR")
        assert routed_on_error["count"] == 2
        assert routed_on_error["is_terminal"] is True


class TestHighVarianceZeroDuration:
    """Verify high_variance filter includes nodes with zero avg_ms.

    Bug: T3 — `if n["avg_ms"] and n["max_ms"]` excluded nodes where
    avg_ms=0.0, because 0.0 is falsy in Python. A node with avg_ms=0.0
    and max_ms=100.0 (high variance!) was silently dropped.
    """

    def test_zero_avg_ms_included_in_high_variance(self) -> None:
        """Node with avg_ms=0.0 and high max_ms should appear in high_variance."""
        fast_node = MagicMock()
        fast_node.node_id = "transform_fast_abc123"
        fast_node.plugin_name = "fast_transform"
        fast_node.node_type = "transform"
        fast_node.executions = 100
        fast_node.avg_ms = 0.0  # Zero average — the key trigger
        fast_node.min_ms = 0.0
        fast_node.max_ms = 100.0  # But max is high — this IS high variance
        fast_node.total_ms = 5.0

        mock_conn = MagicMock()
        mock_conn.execute.return_value.fetchall.side_effect = [
            [fast_node],  # stats_query
            [],  # failed_query
        ]

        db = MagicMock()
        db.connection.return_value.__enter__ = MagicMock(return_value=mock_conn)
        db.connection.return_value.__exit__ = MagicMock(return_value=False)

        factory = MagicMock()
        factory.run_lifecycle.get_run.return_value = MagicMock(
            started_at=None,
            completed_at=None,
            status=MagicMock(value="completed"),
        )

        result = get_performance_report(db, factory, "run-123")

        assert "error" not in result
        # Before fix: high_variance was [] because `0.0 and 100.0` is falsy
        # After fix: node with avg_ms=0.0 but max_ms=100.0 IS high variance
        high_variance = result["high_variance_nodes"]
        assert len(high_variance) == 1
        assert high_variance[0]["node_id"] == "transform_fast_abc123"

    def test_none_avg_ms_excluded_from_high_variance(self) -> None:
        """Node with avg_ms=None (no timing data) should NOT appear in high_variance."""
        no_timing_node = MagicMock()
        no_timing_node.node_id = "transform_notimed_abc123"
        no_timing_node.plugin_name = "notimed_transform"
        no_timing_node.node_type = "transform"
        no_timing_node.executions = 1
        no_timing_node.avg_ms = None
        no_timing_node.min_ms = None
        no_timing_node.max_ms = None
        no_timing_node.total_ms = None

        mock_conn = MagicMock()
        mock_conn.execute.return_value.fetchall.side_effect = [
            [no_timing_node],  # stats_query
            [],  # failed_query
        ]

        db = MagicMock()
        db.connection.return_value.__enter__ = MagicMock(return_value=mock_conn)
        db.connection.return_value.__exit__ = MagicMock(return_value=False)

        factory = MagicMock()
        factory.run_lifecycle.get_run.return_value = MagicMock(
            started_at=None,
            completed_at=None,
            status=MagicMock(value="completed"),
        )

        result = get_performance_report(db, factory, "run-123")

        assert "error" not in result
        high_variance = result["high_variance_nodes"]
        assert len(high_variance) == 0


# ---------------------------------------------------------------------------
# get_error_analysis tests
# ---------------------------------------------------------------------------


def _make_db_and_factory(run_exists: bool = True) -> tuple[MagicMock, MagicMock]:
    """Create mock db/factory pair for get_error_analysis tests."""
    db = MagicMock()
    factory = MagicMock()
    if run_exists:
        factory.run_lifecycle.get_run.return_value = MagicMock()
    else:
        factory.run_lifecycle.get_run.return_value = None
    return db, factory


def _wire_conn(db: MagicMock, val_rows: list[Any], trans_rows: list[Any], sample_val: list[Any], sample_trans: list[Any]) -> None:
    """Wire up mock connection with 4 sequential execute().fetchall() calls."""
    mock_conn = MagicMock()
    call_count = 0

    def side_effect_execute(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        result = MagicMock()
        if call_count == 1:
            result.fetchall.return_value = val_rows
        elif call_count == 2:
            result.fetchall.return_value = trans_rows
        elif call_count == 3:
            result.fetchall.return_value = sample_val
        elif call_count == 4:
            result.fetchall.return_value = sample_trans
        return result

    mock_conn.execute = side_effect_execute
    db.connection.return_value.__enter__ = MagicMock(return_value=mock_conn)
    db.connection.return_value.__exit__ = MagicMock(return_value=False)


def _mock_row(**kwargs: object) -> MagicMock:
    """Create a mock DB row with named attributes."""
    row = MagicMock()
    for k, v in kwargs.items():
        setattr(row, k, v)
    return row


class TestErrorAnalysisRunNotFound:
    """get_error_analysis returns error dict when run_id doesn't exist."""

    def test_returns_error_when_run_not_found(self) -> None:
        db, factory = _make_db_and_factory(run_exists=False)

        result = get_error_analysis(db, factory, "nonexistent-run")

        assert result == {"error": "Run 'nonexistent-run' not found"}


class TestErrorAnalysisCorruptionGuard:
    """Tier 1 corruption guard: None plugin_name in transform errors raises AuditIntegrityError."""

    def test_none_plugin_name_raises_audit_integrity_error(self) -> None:
        """Transform errors referencing a non-existent node must crash, not silently pass."""
        db, factory = _make_db_and_factory()
        corrupt_row = _mock_row(plugin_name=None, count=3)
        _wire_conn(db, val_rows=[], trans_rows=[corrupt_row], sample_val=[], sample_trans=[])

        with pytest.raises(AuditIntegrityError, match="Tier-1 corruption"):
            get_error_analysis(db, factory, "run-corrupt")

    def test_corruption_guard_includes_count_and_run_id(self) -> None:
        """Error message must include the orphan count and run_id for diagnostics."""
        db, factory = _make_db_and_factory()
        corrupt_row = _mock_row(plugin_name=None, count=7)
        _wire_conn(db, val_rows=[], trans_rows=[corrupt_row], sample_val=[], sample_trans=[])

        with pytest.raises(AuditIntegrityError, match=r"7 transform_errors row.*run_id='run-abc'"):
            get_error_analysis(db, factory, "run-abc")

    def test_corruption_guard_fires_even_with_valid_rows_present(self) -> None:
        """A single None plugin_name row triggers the guard even alongside valid rows."""
        db, factory = _make_db_and_factory()
        valid_row = _mock_row(plugin_name="good_transform", count=10)
        corrupt_row = _mock_row(plugin_name=None, count=1)
        _wire_conn(db, val_rows=[], trans_rows=[valid_row, corrupt_row], sample_val=[], sample_trans=[])

        with pytest.raises(AuditIntegrityError):
            get_error_analysis(db, factory, "run-mixed")


class TestErrorAnalysisValidationGrouping:
    """Validation errors are grouped by source plugin_name and schema_mode."""

    def test_groups_validation_errors_by_plugin_and_schema_mode(self) -> None:
        db, factory = _make_db_and_factory()
        val_row_1 = _mock_row(plugin_name="csv_source", schema_mode="strict", count=5)
        val_row_2 = _mock_row(plugin_name="csv_source", schema_mode="coerce", count=2)
        _wire_conn(db, val_rows=[val_row_1, val_row_2], trans_rows=[], sample_val=[], sample_trans=[])

        result = get_error_analysis(db, factory, "run-val")

        assert "error" not in result
        val_errors = result["validation_errors"]
        assert val_errors["total"] == 7
        assert len(val_errors["by_source"]) == 2
        assert val_errors["by_source"][0] == {"source_plugin": "csv_source", "schema_mode": "strict", "count": 5}
        assert val_errors["by_source"][1] == {"source_plugin": "csv_source", "schema_mode": "coerce", "count": 2}

    def test_empty_validation_errors(self) -> None:
        db, factory = _make_db_and_factory()
        _wire_conn(db, val_rows=[], trans_rows=[], sample_val=[], sample_trans=[])

        result = get_error_analysis(db, factory, "run-empty")

        report = cast(ErrorAnalysisReport, result)
        assert report["validation_errors"]["total"] == 0
        assert report["validation_errors"]["by_source"] == []


class TestErrorAnalysisTransformGrouping:
    """Transform errors are grouped by transform plugin_name."""

    def test_groups_transform_errors_by_plugin(self) -> None:
        db, factory = _make_db_and_factory()
        trans_row_1 = _mock_row(plugin_name="llm_classifier", count=3)
        trans_row_2 = _mock_row(plugin_name="field_mapper", count=1)
        _wire_conn(db, val_rows=[], trans_rows=[trans_row_1, trans_row_2], sample_val=[], sample_trans=[])

        result = get_error_analysis(db, factory, "run-trans")

        assert "error" not in result
        trans_errors = result["transform_errors"]
        assert trans_errors["total"] == 4
        assert len(trans_errors["by_transform"]) == 2
        assert trans_errors["by_transform"][0] == {"transform_plugin": "llm_classifier", "count": 3}
        assert trans_errors["by_transform"][1] == {"transform_plugin": "field_mapper", "count": 1}


class TestErrorAnalysisSampleData:
    """Sample error data is extracted and JSON-parsed."""

    def test_parses_sample_validation_data(self) -> None:
        db, factory = _make_db_and_factory()
        sample_json = json.dumps({"field": "age", "value": "not_a_number"})
        sample_row = MagicMock()
        sample_row.__getitem__ = lambda self, idx: sample_json if idx == 0 else None
        _wire_conn(db, val_rows=[], trans_rows=[], sample_val=[sample_row], sample_trans=[])

        result = get_error_analysis(db, factory, "run-sample")

        report = cast(ErrorAnalysisReport, result)
        assert report["validation_errors"]["sample_data"] == [{"field": "age", "value": "not_a_number"}]

    def test_parses_sample_transform_details(self) -> None:
        db, factory = _make_db_and_factory()
        details_json = json.dumps({"error": "division by zero", "node": "calc"})
        sample_row = MagicMock()
        sample_row.__getitem__ = lambda self, idx: details_json if idx == 0 else None
        _wire_conn(db, val_rows=[], trans_rows=[], sample_val=[], sample_trans=[sample_row])

        result = get_error_analysis(db, factory, "run-sample-trans")

        report = cast(ErrorAnalysisReport, result)
        assert report["transform_errors"]["sample_details"] == [{"error": "division by zero", "node": "calc"}]

    def test_none_sample_data_preserved_as_none(self) -> None:
        """When row_data_json is NULL/None, the sample entry should be None, not crash."""
        db, factory = _make_db_and_factory()
        null_row = MagicMock()
        null_row.__getitem__ = lambda self, idx: None
        _wire_conn(db, val_rows=[], trans_rows=[], sample_val=[null_row], sample_trans=[])

        result = get_error_analysis(db, factory, "run-null-sample")

        report = cast(ErrorAnalysisReport, result)
        assert report["validation_errors"]["sample_data"] == [None]


# ---------------------------------------------------------------------------
# LLM Usage Report Tests
# ---------------------------------------------------------------------------


def _make_llm_row(
    plugin_name: str,
    call_type: str,
    status: CallStatus,
    count: int,
    avg_latency: float,
    min_latency: float,
    max_latency: float,
    total_latency: float,
) -> MagicMock:
    """Create a mock aggregated LLM row (result of GROUP BY query)."""
    row = MagicMock()
    row.plugin_name = plugin_name
    row.call_type = call_type
    row.status = status
    row.count = count
    row.avg_latency = avg_latency
    row.min_latency = min_latency
    row.max_latency = max_latency
    row.total_latency = total_latency
    return row


def _make_call_type_row(call_type: str, count: int) -> MagicMock:
    """Create a mock call type summary row."""
    row = MagicMock()
    row.call_type = call_type
    row.count = count
    return row


def _make_llm_db_and_factory(
    run_exists: bool = True,
) -> tuple[MagicMock, MagicMock]:
    """Create db and factory mocks for LLM usage report tests."""
    db = MagicMock()
    mock_conn = MagicMock()
    db.connection.return_value.__enter__ = MagicMock(return_value=mock_conn)
    db.connection.return_value.__exit__ = MagicMock(return_value=False)

    factory = MagicMock()
    if run_exists:
        factory.run_lifecycle.get_run.return_value = MagicMock(run_id="test-run")
    else:
        factory.run_lifecycle.get_run.return_value = None

    return db, factory


def _wire_llm_conn(
    db: MagicMock,
    llm_rows: list[MagicMock],
    call_type_rows: list[MagicMock],
) -> None:
    """Wire mock connection to return llm_rows then call_type_rows on sequential execute calls."""
    mock_conn = db.connection.return_value.__enter__.return_value
    call_count = 0

    def side_effect_execute(*args: object, **kwargs: object) -> MagicMock:
        nonlocal call_count
        call_count += 1
        result = MagicMock()
        if call_count == 1:
            result.fetchall.return_value = llm_rows
        elif call_count == 2:
            result.fetchall.return_value = call_type_rows
        return result

    mock_conn.execute = side_effect_execute


class TestLLMUsageReportRunNotFound:
    """get_llm_usage_report returns error when run doesn't exist."""

    def test_returns_error_for_missing_run(self) -> None:
        db, factory = _make_llm_db_and_factory(run_exists=False)

        result = get_llm_usage_report(db, factory, "nonexistent-run")

        assert "error" in result
        error_result = cast(ErrorResult, result)
        assert "nonexistent-run" in error_result["error"]


class TestLLMUsageReportNoLLMCalls:
    """get_llm_usage_report handles runs with no LLM calls."""

    def test_returns_message_when_no_llm_calls(self) -> None:
        db, factory = _make_llm_db_and_factory()
        _wire_llm_conn(
            db,
            llm_rows=[],
            call_type_rows=[
                _make_call_type_row("http", 5),
                _make_call_type_row("database", 3),
            ],
        )

        result = get_llm_usage_report(db, factory, "test-run")

        report = cast(LLMUsageReport, result)
        assert report["message"] == "No LLM calls found in this run"
        assert report["call_types"] == {"http": 5, "database": 3}
        assert "llm_summary" not in report
        assert "by_plugin" not in report

    def test_returns_empty_call_types_when_no_calls_at_all(self) -> None:
        db, factory = _make_llm_db_and_factory()
        _wire_llm_conn(db, llm_rows=[], call_type_rows=[])

        result = get_llm_usage_report(db, factory, "test-run")

        report = cast(LLMUsageReport, result)
        assert report["message"] == "No LLM calls found in this run"
        assert report["call_types"] == {}


class TestLLMUsageReportSinglePlugin:
    """get_llm_usage_report aggregates correctly for a single plugin."""

    def test_single_plugin_success_only(self) -> None:
        db, factory = _make_llm_db_and_factory()
        _wire_llm_conn(
            db,
            llm_rows=[
                _make_llm_row(
                    plugin_name="llm_classifier",
                    call_type="llm",
                    status=CallStatus.SUCCESS,
                    count=10,
                    avg_latency=150.0,
                    min_latency=50.0,
                    max_latency=300.0,
                    total_latency=1500.0,
                ),
            ],
            call_type_rows=[
                _make_call_type_row("llm", 10),
            ],
        )

        result = get_llm_usage_report(db, factory, "test-run")

        report = cast(LLMUsageReport, result)
        assert report["run_id"] == "test-run"
        assert report["call_types"] == {"llm": 10}

        plugin_stats = report["by_plugin"]["llm_classifier"]
        assert plugin_stats["total_calls"] == 10
        assert plugin_stats["successful"] == 10
        assert plugin_stats["failed"] == 0
        assert plugin_stats["avg_latency_ms"] == 150.0
        assert plugin_stats["total_latency_ms"] == 1500.0

        assert report["llm_summary"]["total_calls"] == 10
        assert report["llm_summary"]["total_latency_ms"] == 1500.0
        assert report["llm_summary"]["avg_latency_ms"] == 150.0


class TestLLMUsageReportSuccessFailureSplit:
    """get_llm_usage_report correctly splits successful and failed counts using CallStatus."""

    def test_success_and_failure_counts_split_correctly(self) -> None:
        db, factory = _make_llm_db_and_factory()
        _wire_llm_conn(
            db,
            llm_rows=[
                _make_llm_row(
                    plugin_name="llm_classifier",
                    call_type="llm",
                    status=CallStatus.SUCCESS,
                    count=8,
                    avg_latency=100.0,
                    min_latency=50.0,
                    max_latency=200.0,
                    total_latency=800.0,
                ),
                _make_llm_row(
                    plugin_name="llm_classifier",
                    call_type="llm",
                    status=CallStatus.ERROR,
                    count=2,
                    avg_latency=500.0,
                    min_latency=400.0,
                    max_latency=600.0,
                    total_latency=1000.0,
                ),
            ],
            call_type_rows=[
                _make_call_type_row("llm", 10),
            ],
        )

        result = get_llm_usage_report(db, factory, "test-run")

        report = cast(LLMUsageReport, result)
        plugin_stats = report["by_plugin"]["llm_classifier"]
        assert plugin_stats["total_calls"] == 10
        assert plugin_stats["successful"] == 8
        assert plugin_stats["failed"] == 2
        assert plugin_stats["total_latency_ms"] == 1800.0


class TestLLMUsageReportAverageLatency:
    """get_llm_usage_report calculates average latency as total_latency_ms / total_calls, rounded to 2dp."""

    def test_average_latency_calculation(self) -> None:
        db, factory = _make_llm_db_and_factory()
        _wire_llm_conn(
            db,
            llm_rows=[
                _make_llm_row(
                    plugin_name="llm_summarizer",
                    call_type="llm",
                    status=CallStatus.SUCCESS,
                    count=3,
                    avg_latency=100.0,
                    min_latency=80.0,
                    max_latency=120.0,
                    total_latency=333.33,
                ),
            ],
            call_type_rows=[_make_call_type_row("llm", 3)],
        )

        result = get_llm_usage_report(db, factory, "test-run")

        report = cast(LLMUsageReport, result)
        plugin_stats = report["by_plugin"]["llm_summarizer"]
        # 333.33 / 3 = 111.11
        assert plugin_stats["avg_latency_ms"] == 111.11

        assert report["llm_summary"]["avg_latency_ms"] == 111.11

    def test_average_latency_rounds_to_two_decimals(self) -> None:
        db, factory = _make_llm_db_and_factory()
        _wire_llm_conn(
            db,
            llm_rows=[
                _make_llm_row(
                    plugin_name="llm_router",
                    call_type="llm",
                    status=CallStatus.SUCCESS,
                    count=7,
                    avg_latency=100.0,
                    min_latency=50.0,
                    max_latency=200.0,
                    total_latency=1000.0,
                ),
            ],
            call_type_rows=[_make_call_type_row("llm", 7)],
        )

        result = get_llm_usage_report(db, factory, "test-run")

        report = cast(LLMUsageReport, result)
        # 1000.0 / 7 = 142.857142... -> 142.86
        assert report["by_plugin"]["llm_router"]["avg_latency_ms"] == 142.86
        assert report["llm_summary"]["avg_latency_ms"] == 142.86


class TestLLMUsageReportMultiplePlugins:
    """get_llm_usage_report aggregates correctly across multiple plugins."""

    def test_multiple_plugins_aggregated_independently(self) -> None:
        db, factory = _make_llm_db_and_factory()
        _wire_llm_conn(
            db,
            llm_rows=[
                _make_llm_row(
                    plugin_name="llm_classifier",
                    call_type="llm",
                    status=CallStatus.SUCCESS,
                    count=5,
                    avg_latency=100.0,
                    min_latency=50.0,
                    max_latency=150.0,
                    total_latency=500.0,
                ),
                _make_llm_row(
                    plugin_name="llm_classifier",
                    call_type="llm",
                    status=CallStatus.ERROR,
                    count=1,
                    avg_latency=800.0,
                    min_latency=800.0,
                    max_latency=800.0,
                    total_latency=800.0,
                ),
                _make_llm_row(
                    plugin_name="llm_summarizer",
                    call_type="llm",
                    status=CallStatus.SUCCESS,
                    count=10,
                    avg_latency=200.0,
                    min_latency=100.0,
                    max_latency=300.0,
                    total_latency=2000.0,
                ),
            ],
            call_type_rows=[
                _make_call_type_row("llm", 16),
                _make_call_type_row("http", 4),
            ],
        )

        result = get_llm_usage_report(db, factory, "test-run")

        report = cast(LLMUsageReport, result)
        # Classifier: 5 success + 1 error = 6 total, 1300ms total latency
        classifier = report["by_plugin"]["llm_classifier"]
        assert classifier["total_calls"] == 6
        assert classifier["successful"] == 5
        assert classifier["failed"] == 1
        assert classifier["total_latency_ms"] == 1300.0
        assert classifier["avg_latency_ms"] == round(1300.0 / 6, 2)

        # Summarizer: 10 success, 2000ms total latency
        summarizer = report["by_plugin"]["llm_summarizer"]
        assert summarizer["total_calls"] == 10
        assert summarizer["successful"] == 10
        assert summarizer["failed"] == 0
        assert summarizer["total_latency_ms"] == 2000.0
        assert summarizer["avg_latency_ms"] == 200.0

        # Overall summary: 16 total calls, 3300ms total latency
        assert report["llm_summary"]["total_calls"] == 16
        assert report["llm_summary"]["total_latency_ms"] == 3300.0
        assert report["llm_summary"]["avg_latency_ms"] == round(3300.0 / 16, 2)

        # Call types include non-LLM types
        assert report["call_types"] == {"llm": 16, "http": 4}
