# src/elspeth/mcp/server.py
"""MCP server for ELSPETH Landscape audit database analysis.

A lightweight read-only server that exposes tools for querying
the audit trail. Uses the existing LandscapeDB and LandscapeRecorder
infrastructure.

Usage:
    # Direct execution
    python -m elspeth.mcp.server --database sqlite:///./state/audit.db

    # Or as an MCP server
    elspeth-mcp --database sqlite:///./state/audit.db
"""

import argparse
import json
import logging
import sys
from datetime import UTC, datetime
from typing import Any, cast

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, Tool

from elspeth.contracts.enums import CallStatus
from elspeth.core.landscape.database import LandscapeDB
from elspeth.core.landscape.formatters import dataclass_to_dict, serialize_datetime
from elspeth.core.landscape.lineage import explain
from elspeth.core.landscape.recorder import LandscapeRecorder

logger = logging.getLogger(__name__)

# JSON-serializable result type
JsonResult = dict[str, Any] | list[dict[str, Any]]


# Use shared implementations from landscape.formatters
_serialize_datetime = serialize_datetime
_dataclass_to_dict = dataclass_to_dict


class LandscapeAnalyzer:
    """Read-only analyzer for the Landscape audit database."""

    def __init__(self, database_url: str) -> None:
        """Initialize analyzer with database connection.

        Args:
            database_url: SQLAlchemy connection URL (e.g., sqlite:///./state/audit.db)
        """
        self._db = LandscapeDB.from_url(database_url, create_tables=False)
        self._recorder = LandscapeRecorder(self._db)

    def close(self) -> None:
        """Close database connection."""
        self._db.close()

    def list_runs(self, limit: int = 50, status: str | None = None) -> list[dict[str, Any]]:
        """List pipeline runs.

        Args:
            limit: Maximum number of runs to return (default 50)
            status: Filter by status (PENDING, RUNNING, COMPLETED, FAILED)

        Returns:
            List of run records with id, status, timestamps
        """
        from sqlalchemy import select

        from elspeth.contracts import RunStatus
        from elspeth.core.landscape.schema import runs_table

        with self._db.connection() as conn:
            query = select(runs_table).order_by(runs_table.c.started_at.desc()).limit(limit)

            if status is not None:
                # Validate status
                try:
                    RunStatus(status)
                except ValueError:
                    valid = [s.value for s in RunStatus]
                    raise ValueError(f"Invalid status '{status}'. Valid: {valid}") from None
                query = query.where(runs_table.c.status == status)

            rows = conn.execute(query).fetchall()

        return [
            {
                "run_id": row.run_id,
                "status": row.status,
                "started_at": row.started_at.isoformat() if row.started_at else None,
                "completed_at": row.completed_at.isoformat() if row.completed_at else None,
                "config_hash": row.config_hash,
                "export_status": row.export_status,
            }
            for row in rows
        ]

    def get_run(self, run_id: str) -> dict[str, Any] | None:
        """Get details of a specific run.

        Args:
            run_id: The run ID to retrieve

        Returns:
            Run record or None if not found
        """
        run = self._recorder.get_run(run_id)
        if run is None:
            return None
        return cast(dict[str, Any], _dataclass_to_dict(run))

    def get_run_summary(self, run_id: str) -> dict[str, Any]:
        """Get summary statistics for a run.

        Args:
            run_id: The run ID to analyze

        Returns:
            Summary with counts, durations, and error information
        """
        from sqlalchemy import func, select

        from elspeth.core.landscape.schema import (
            node_states_table,
            nodes_table,
            operations_table,
            rows_table,
            token_outcomes_table,
            tokens_table,
            transform_errors_table,
            validation_errors_table,
        )

        run = self._recorder.get_run(run_id)
        if run is None:
            return {"error": f"Run '{run_id}' not found"}

        with self._db.connection() as conn:
            # Count rows
            row_count = conn.execute(select(func.count()).select_from(rows_table).where(rows_table.c.run_id == run_id)).scalar() or 0

            # Count tokens
            token_count = (
                conn.execute(
                    select(func.count())
                    .select_from(tokens_table)
                    .join(rows_table, tokens_table.c.row_id == rows_table.c.row_id)
                    .where(rows_table.c.run_id == run_id)
                ).scalar()
                or 0
            )

            # Count nodes
            node_count = conn.execute(select(func.count()).select_from(nodes_table).where(nodes_table.c.run_id == run_id)).scalar() or 0

            # Count node states
            state_count = (
                conn.execute(select(func.count()).select_from(node_states_table).where(node_states_table.c.run_id == run_id)).scalar() or 0
            )

            # Count operations (source/sink I/O)
            operation_count = (
                conn.execute(select(func.count()).select_from(operations_table).where(operations_table.c.run_id == run_id)).scalar() or 0
            )

            # Count source_load operations
            source_load_count = (
                conn.execute(
                    select(func.count())
                    .select_from(operations_table)
                    .where((operations_table.c.run_id == run_id) & (operations_table.c.operation_type == "source_load"))
                ).scalar()
                or 0
            )

            # Count sink_write operations
            sink_write_count = (
                conn.execute(
                    select(func.count())
                    .select_from(operations_table)
                    .where((operations_table.c.run_id == run_id) & (operations_table.c.operation_type == "sink_write"))
                ).scalar()
                or 0
            )

            # Count validation errors
            validation_error_count = (
                conn.execute(
                    select(func.count()).select_from(validation_errors_table).where(validation_errors_table.c.run_id == run_id)
                ).scalar()
                or 0
            )

            # Count transform errors
            transform_error_count = (
                conn.execute(
                    select(func.count()).select_from(transform_errors_table).where(transform_errors_table.c.run_id == run_id)
                ).scalar()
                or 0
            )

            # Get outcome distribution
            outcome_query = (
                select(token_outcomes_table.c.outcome, func.count().label("count"))
                .where(token_outcomes_table.c.run_id == run_id)
                .group_by(token_outcomes_table.c.outcome)
            )
            outcome_rows = conn.execute(outcome_query).fetchall()
            outcome_distribution = {row.outcome: row.count for row in outcome_rows}

            # Calculate average processing duration per node state
            avg_duration = conn.execute(
                select(func.avg(node_states_table.c.duration_ms)).where(node_states_table.c.run_id == run_id)
            ).scalar()

        # Calculate run duration
        run_duration_seconds = None
        if run.started_at and run.completed_at:
            run_duration_seconds = (run.completed_at - run.started_at).total_seconds()

        return {
            "run_id": run_id,
            "status": run.status.value,
            "started_at": run.started_at.isoformat() if run.started_at else None,
            "completed_at": run.completed_at.isoformat() if run.completed_at else None,
            "run_duration_seconds": run_duration_seconds,
            "counts": {
                "rows": row_count,
                "tokens": token_count,
                "nodes": node_count,
                "node_states": state_count,
                "operations": operation_count,
                "source_loads": source_load_count,
                "sink_writes": sink_write_count,
            },
            "errors": {
                "validation": validation_error_count,
                "transform": transform_error_count,
                "total": validation_error_count + transform_error_count,
            },
            "outcome_distribution": outcome_distribution,
            "avg_state_duration_ms": round(avg_duration, 2) if avg_duration else None,
        }

    def list_rows(self, run_id: str, limit: int = 100, offset: int = 0) -> list[dict[str, Any]]:
        """List source rows for a run.

        Args:
            run_id: Run ID to query
            limit: Maximum rows to return (default 100)
            offset: Number of rows to skip (default 0)

        Returns:
            List of row records
        """
        from sqlalchemy import select

        from elspeth.core.landscape.schema import rows_table

        with self._db.connection() as conn:
            query = select(rows_table).where(rows_table.c.run_id == run_id).order_by(rows_table.c.row_index).limit(limit).offset(offset)
            rows = conn.execute(query).fetchall()

        return [
            {
                "row_id": row.row_id,
                "run_id": row.run_id,
                "source_node_id": row.source_node_id,
                "row_index": row.row_index,
                "source_data_hash": row.source_data_hash,
                "source_data_ref": row.source_data_ref,
                "created_at": row.created_at.isoformat() if row.created_at else None,
            }
            for row in rows
        ]

    def list_nodes(self, run_id: str) -> list[dict[str, Any]]:
        """List all nodes (plugin instances) for a run.

        Args:
            run_id: Run ID to query

        Returns:
            List of node records with plugin info
        """
        nodes = self._recorder.get_nodes(run_id)
        return [_dataclass_to_dict(node) for node in nodes]

    def list_tokens(self, run_id: str, row_id: str | None = None, limit: int = 100) -> list[dict[str, Any]]:
        """List tokens for a run or specific row.

        Args:
            run_id: Run ID to query
            row_id: Optional row ID to filter by
            limit: Maximum tokens to return

        Returns:
            List of token records
        """
        from sqlalchemy import select

        from elspeth.core.landscape.schema import rows_table, tokens_table

        with self._db.connection() as conn:
            query = (
                select(tokens_table)
                .join(rows_table, tokens_table.c.row_id == rows_table.c.row_id)
                .where(rows_table.c.run_id == run_id)
                .limit(limit)
            )

            if row_id is not None:
                query = query.where(tokens_table.c.row_id == row_id)

            rows = conn.execute(query).fetchall()

        return [
            {
                "token_id": row.token_id,
                "row_id": row.row_id,
                "branch_name": row.branch_name,
                "fork_group_id": row.fork_group_id,
                "join_group_id": row.join_group_id,
                "step_in_pipeline": row.step_in_pipeline,
                "expand_group_id": row.expand_group_id,
                "created_at": row.created_at.isoformat() if row.created_at else None,
            }
            for row in rows
        ]

    def list_operations(
        self,
        run_id: str,
        operation_type: str | None = None,
        status: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """List source/sink operations for a run.

        Operations are the source/sink equivalent of node_states. They track
        external I/O operations (blob downloads, file writes, database inserts)
        and provide a parent context for external calls made during source.load()
        or sink.write().

        Args:
            run_id: Run ID to query
            operation_type: Filter by type ('source_load' or 'sink_write')
            status: Filter by status ('open', 'completed', 'failed', 'pending')
            limit: Maximum operations to return

        Returns:
            List of operation records with node info
        """
        from sqlalchemy import select

        from elspeth.core.landscape.schema import nodes_table, operations_table

        with self._db.connection() as conn:
            query = (
                select(
                    operations_table.c.operation_id,
                    operations_table.c.run_id,
                    operations_table.c.node_id,
                    operations_table.c.operation_type,
                    operations_table.c.started_at,
                    operations_table.c.completed_at,
                    operations_table.c.status,
                    operations_table.c.error_message,
                    operations_table.c.duration_ms,
                    nodes_table.c.plugin_name,
                )
                .join(
                    nodes_table,
                    (operations_table.c.node_id == nodes_table.c.node_id) & (operations_table.c.run_id == nodes_table.c.run_id),
                )
                .where(operations_table.c.run_id == run_id)
                .order_by(operations_table.c.started_at.desc())
                .limit(limit)
            )

            if operation_type is not None:
                if operation_type not in ("source_load", "sink_write"):
                    raise ValueError(f"Invalid operation_type '{operation_type}'. Valid: source_load, sink_write")
                query = query.where(operations_table.c.operation_type == operation_type)

            if status is not None:
                valid_statuses = ("open", "completed", "failed", "pending")
                if status not in valid_statuses:
                    raise ValueError(f"Invalid status '{status}'. Valid: {valid_statuses}")
                query = query.where(operations_table.c.status == status)

            rows = conn.execute(query).fetchall()

        return [
            {
                "operation_id": row.operation_id,
                "run_id": row.run_id,
                "node_id": row.node_id,
                "plugin_name": row.plugin_name,
                "operation_type": row.operation_type,
                "status": row.status,
                "started_at": row.started_at.isoformat() if row.started_at else None,
                "completed_at": row.completed_at.isoformat() if row.completed_at else None,
                "duration_ms": row.duration_ms,
                "error_message": row.error_message,
            }
            for row in rows
        ]

    def get_operation_calls(self, operation_id: str) -> list[dict[str, Any]]:
        """Get external calls for a source/sink operation.

        Unlike get_calls() which takes a state_id for transform calls, this
        method returns calls made during source.load() or sink.write().

        Args:
            operation_id: Operation ID to query

        Returns:
            List of call records (HTTP, SQL, etc.)
        """
        from sqlalchemy import select

        from elspeth.core.landscape.schema import calls_table

        with self._db.connection() as conn:
            query = select(calls_table).where(calls_table.c.operation_id == operation_id).order_by(calls_table.c.call_index)
            rows = conn.execute(query).fetchall()

        return [
            {
                "call_id": row.call_id,
                "operation_id": row.operation_id,
                "call_index": row.call_index,
                "call_type": row.call_type,
                "status": row.status,
                "latency_ms": row.latency_ms,
                "request_hash": row.request_hash,
                "response_hash": row.response_hash,
                "created_at": row.created_at.isoformat() if row.created_at else None,
            }
            for row in rows
        ]

    def explain_token(
        self,
        run_id: str,
        token_id: str | None = None,
        row_id: str | None = None,
        sink: str | None = None,
    ) -> dict[str, Any]:
        """Get complete lineage for a token or row.

        Args:
            run_id: Run ID to query
            token_id: Token ID for precise lineage (preferred for DAGs with forks)
            row_id: Row ID (requires disambiguation if multiple terminal tokens)
            sink: Sink name to disambiguate when row has multiple terminal tokens

        Returns:
            Complete lineage including source row, node states, calls, errors, outcome
        """
        result = explain(self._recorder, run_id, token_id=token_id, row_id=row_id, sink=sink)
        if result is None:
            return {"error": "Token or row not found, or no terminal tokens exist yet"}
        return cast(dict[str, Any], _dataclass_to_dict(result))

    def get_errors(
        self,
        run_id: str,
        error_type: str = "all",
        limit: int = 100,
    ) -> dict[str, Any]:
        """Get validation and/or transform errors for a run.

        Args:
            run_id: Run ID to query
            error_type: "validation", "transform", or "all" (default)
            limit: Maximum errors to return per type

        Returns:
            Errors grouped by type
        """
        from sqlalchemy import select

        from elspeth.core.landscape.schema import transform_errors_table, validation_errors_table

        result: dict[str, Any] = {"run_id": run_id}

        with self._db.connection() as conn:
            if error_type in ("all", "validation"):
                query = (
                    select(validation_errors_table)
                    .where(validation_errors_table.c.run_id == run_id)
                    .order_by(validation_errors_table.c.created_at.desc())
                    .limit(limit)
                )
                rows = conn.execute(query).fetchall()
                result["validation_errors"] = [
                    {
                        "error_id": row.error_id,
                        "node_id": row.node_id,
                        "row_hash": row.row_hash,
                        "row_data": json.loads(row.row_data_json) if row.row_data_json else None,
                        "schema_mode": row.schema_mode,
                        "created_at": row.created_at.isoformat() if row.created_at else None,
                    }
                    for row in rows
                ]

            if error_type in ("all", "transform"):
                query = (
                    select(transform_errors_table)
                    .where(transform_errors_table.c.run_id == run_id)
                    .order_by(transform_errors_table.c.created_at.desc())
                    .limit(limit)
                )
                rows = conn.execute(query).fetchall()
                result["transform_errors"] = [
                    {
                        "error_id": row.error_id,
                        "token_id": row.token_id,
                        "transform_id": row.transform_id,
                        "row_data": json.loads(row.row_data_json) if row.row_data_json else None,
                        "error_details": json.loads(row.error_details_json) if row.error_details_json else None,
                        "created_at": row.created_at.isoformat() if row.created_at else None,
                    }
                    for row in rows
                ]

        return result

    def get_node_states(
        self,
        run_id: str,
        node_id: str | None = None,
        status: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """Get node states (processing records) for a run.

        Args:
            run_id: Run ID to query
            node_id: Optional filter by node ID
            status: Optional filter by status (PENDING, RUNNING, COMPLETED, FAILED)
            limit: Maximum states to return

        Returns:
            List of node state records
        """
        from sqlalchemy import select

        from elspeth.contracts import NodeStateStatus
        from elspeth.core.landscape.schema import node_states_table

        with self._db.connection() as conn:
            query = select(node_states_table).where(node_states_table.c.run_id == run_id).limit(limit)

            if node_id is not None:
                query = query.where(node_states_table.c.node_id == node_id)

            if status is not None:
                try:
                    NodeStateStatus(status)
                except ValueError:
                    valid = [s.value for s in NodeStateStatus]
                    raise ValueError(f"Invalid status '{status}'. Valid: {valid}") from None
                query = query.where(node_states_table.c.status == status)

            query = query.order_by(node_states_table.c.step_index)
            rows = conn.execute(query).fetchall()

        return [
            {
                "state_id": row.state_id,
                "token_id": row.token_id,
                "node_id": row.node_id,
                "step_index": row.step_index,
                "attempt": row.attempt,
                "status": row.status,
                "input_hash": row.input_hash,
                "output_hash": row.output_hash,
                "duration_ms": row.duration_ms,
                "started_at": row.started_at.isoformat() if row.started_at else None,
                "completed_at": row.completed_at.isoformat() if row.completed_at else None,
            }
            for row in rows
        ]

    def get_calls(self, state_id: str) -> list[dict[str, Any]]:
        """Get external calls for a node state.

        Args:
            state_id: Node state ID to query

        Returns:
            List of call records (LLM calls, HTTP requests, etc.)
        """
        calls = self._recorder.get_calls(state_id)
        return [_dataclass_to_dict(call) for call in calls]

    def query(self, sql: str, params: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        """Execute a read-only SQL query.

        Args:
            sql: SQL query (must be SELECT)
            params: Optional query parameters

        Returns:
            Query results as list of dicts

        Raises:
            ValueError: If query is not a SELECT statement
        """
        # Safety check: only allow SELECT statements
        sql_normalized = sql.strip().upper()
        if not sql_normalized.startswith("SELECT"):
            raise ValueError("Only SELECT queries are allowed")

        # Reject dangerous keywords even in SELECT
        dangerous = ["INSERT", "UPDATE", "DELETE", "DROP", "CREATE", "ALTER", "TRUNCATE", "GRANT", "REVOKE"]
        for keyword in dangerous:
            if keyword in sql_normalized:
                raise ValueError(f"Query contains forbidden keyword: {keyword}")

        from sqlalchemy import text

        with self._db.connection() as conn:
            result = conn.execute(text(sql), params or {})
            columns = result.keys()
            rows = result.fetchall()

        return [dict(zip(columns, [_serialize_datetime(v) for v in row], strict=False)) for row in rows]

    def get_dag_structure(self, run_id: str) -> dict[str, Any]:
        """Get the DAG structure for a run as a structured object.

        Returns nodes, edges, and a mermaid diagram for visualization.

        Args:
            run_id: Run ID to analyze

        Returns:
            DAG structure with nodes, edges, and mermaid diagram
        """
        nodes = self._recorder.get_nodes(run_id)
        edges = self._recorder.get_edges(run_id)

        if not nodes:
            return {"error": f"Run '{run_id}' not found or has no nodes"}

        # Build simplified structure
        node_list = [
            {
                "node_id": n.node_id,
                "plugin_name": n.plugin_name,
                "node_type": n.node_type.value,
                "sequence": n.sequence_in_pipeline,
            }
            for n in nodes
        ]

        edge_list = [
            {
                "from": e.from_node_id,
                "to": e.to_node_id,
                "label": e.label,
                "mode": e.default_mode.value,
            }
            for e in edges
        ]

        # Generate mermaid diagram
        lines = ["graph TD"]
        for n in nodes:
            label = f"{n.plugin_name}[{n.node_type.value}]"
            lines.append(f'    {n.node_id[:8]}["{label}"]')
        for e in edges:
            arrow = "-->" if e.label == "continue" else f"-->|{e.label}|"
            lines.append(f"    {e.from_node_id[:8]} {arrow} {e.to_node_id[:8]}")

        return {
            "run_id": run_id,
            "nodes": node_list,
            "edges": edge_list,
            "node_count": len(nodes),
            "edge_count": len(edges),
            "mermaid": "\n".join(lines),
        }

    def get_performance_report(self, run_id: str) -> dict[str, Any]:
        """Get performance analysis for a run.

        Identifies slow nodes, outliers, and processing bottlenecks.

        Args:
            run_id: Run ID to analyze

        Returns:
            Performance report with timing analysis
        """
        from sqlalchemy import func, select

        from elspeth.core.landscape.schema import node_states_table, nodes_table

        run = self._recorder.get_run(run_id)
        if run is None:
            return {"error": f"Run '{run_id}' not found"}

        with self._db.connection() as conn:
            # Get per-node timing statistics using composite key join
            stats_query = (
                select(
                    nodes_table.c.node_id,
                    nodes_table.c.plugin_name,
                    nodes_table.c.node_type,
                    func.count(node_states_table.c.state_id).label("executions"),
                    func.avg(node_states_table.c.duration_ms).label("avg_ms"),
                    func.min(node_states_table.c.duration_ms).label("min_ms"),
                    func.max(node_states_table.c.duration_ms).label("max_ms"),
                    func.sum(node_states_table.c.duration_ms).label("total_ms"),
                )
                .join(
                    nodes_table,
                    (node_states_table.c.node_id == nodes_table.c.node_id) & (node_states_table.c.run_id == nodes_table.c.run_id),
                )
                .where(node_states_table.c.run_id == run_id)
                .group_by(nodes_table.c.node_id, nodes_table.c.plugin_name, nodes_table.c.node_type)
                .order_by(func.sum(node_states_table.c.duration_ms).desc())
            )
            stats_rows = conn.execute(stats_query).fetchall()

            # Get failed states count per node
            failed_query = (
                select(
                    node_states_table.c.node_id,
                    func.count(node_states_table.c.state_id).label("failures"),
                )
                .where((node_states_table.c.run_id == run_id) & (node_states_table.c.status == "failed"))
                .group_by(node_states_table.c.node_id)
            )
            failed_rows = conn.execute(failed_query).fetchall()
            failures_by_node = {row.node_id: row.failures for row in failed_rows}

        # Calculate total processing time
        total_time_ms = sum(row.total_ms or 0 for row in stats_rows)

        # Build node performance list
        node_performance = []
        for row in stats_rows:
            pct_of_total = ((row.total_ms or 0) / total_time_ms * 100) if total_time_ms > 0 else 0
            node_performance.append(
                {
                    "node_id": row.node_id[:12] + "...",
                    "plugin": row.plugin_name,
                    "type": row.node_type,
                    "executions": row.executions,
                    "avg_ms": round(row.avg_ms, 2) if row.avg_ms else None,
                    "min_ms": row.min_ms,
                    "max_ms": row.max_ms,
                    "total_ms": row.total_ms,
                    "pct_of_total": round(pct_of_total, 1),
                    "failures": failures_by_node.get(row.node_id, 0),
                }
            )

        # Identify bottlenecks (nodes taking > 20% of total time)
        bottlenecks = [n for n in node_performance if n["pct_of_total"] > 20]

        # Identify high-variance nodes (max > 5x avg)
        high_variance = [n for n in node_performance if n["avg_ms"] and n["max_ms"] and n["max_ms"] > 5 * n["avg_ms"]]

        return {
            "run_id": run_id,
            "total_processing_time_ms": total_time_ms,
            "node_count": len(node_performance),
            "bottlenecks": bottlenecks,
            "high_variance_nodes": high_variance,
            "node_performance": node_performance,
        }

    def get_error_analysis(self, run_id: str) -> dict[str, Any]:
        """Analyze errors for a run, grouping by type and identifying patterns.

        Args:
            run_id: Run ID to analyze

        Returns:
            Error analysis with groupings and patterns
        """
        from sqlalchemy import func, select

        from elspeth.core.landscape.schema import (
            nodes_table,
            transform_errors_table,
            validation_errors_table,
        )

        run = self._recorder.get_run(run_id)
        if run is None:
            return {"error": f"Run '{run_id}' not found"}

        with self._db.connection() as conn:
            # Validation errors by source node using composite key
            val_by_node = (
                select(
                    nodes_table.c.plugin_name,
                    validation_errors_table.c.schema_mode,
                    func.count(validation_errors_table.c.error_id).label("count"),
                )
                .outerjoin(
                    nodes_table,
                    (validation_errors_table.c.node_id == nodes_table.c.node_id)
                    & (validation_errors_table.c.run_id == nodes_table.c.run_id),
                )
                .where(validation_errors_table.c.run_id == run_id)
                .group_by(nodes_table.c.plugin_name, validation_errors_table.c.schema_mode)
            )
            val_rows = conn.execute(val_by_node).fetchall()

            # Transform errors by transform node using composite key
            trans_by_node = (
                select(
                    nodes_table.c.plugin_name,
                    func.count(transform_errors_table.c.error_id).label("count"),
                )
                .outerjoin(
                    nodes_table,
                    (transform_errors_table.c.transform_id == nodes_table.c.node_id)
                    & (transform_errors_table.c.run_id == nodes_table.c.run_id),
                )
                .where(transform_errors_table.c.run_id == run_id)
                .group_by(nodes_table.c.plugin_name)
            )
            trans_rows = conn.execute(trans_by_node).fetchall()

            # Sample error details for pattern matching
            sample_val = conn.execute(
                select(validation_errors_table.c.row_data_json).where(validation_errors_table.c.run_id == run_id).limit(5)
            ).fetchall()

            sample_trans = conn.execute(
                select(transform_errors_table.c.error_details_json).where(transform_errors_table.c.run_id == run_id).limit(5)
            ).fetchall()

        validation_summary = [
            {
                "source_plugin": row.plugin_name or "unknown",
                "schema_mode": row.schema_mode,
                "count": row.count,
            }
            for row in val_rows
        ]

        transform_summary = [{"transform_plugin": row.plugin_name or "unknown", "count": row.count} for row in trans_rows]

        return {
            "run_id": run_id,
            "validation_errors": {
                "total": sum(r["count"] for r in validation_summary),
                "by_source": validation_summary,
                "sample_data": [json.loads(r[0]) if r[0] else None for r in sample_val],
            },
            "transform_errors": {
                "total": sum(r["count"] for r in transform_summary),
                "by_transform": transform_summary,
                "sample_details": [json.loads(r[0]) if r[0] else None for r in sample_trans],
            },
        }

    def get_llm_usage_report(self, run_id: str) -> dict[str, Any]:
        """Get LLM usage statistics for a run.

        Analyzes external calls that were LLM API calls.

        Args:
            run_id: Run ID to analyze

        Returns:
            LLM usage report with call counts, latencies, and token estimates
        """
        from sqlalchemy import func, select, union_all

        from elspeth.core.landscape.schema import (
            calls_table,
            node_states_table,
            nodes_table,
            operations_table,
        )

        run = self._recorder.get_run(run_id)
        if run is None:
            return {"error": f"Run '{run_id}' not found"}

        with self._db.connection() as conn:
            # LLM calls via transform state_id path
            llm_state_query = (
                select(
                    nodes_table.c.plugin_name,
                    calls_table.c.call_type,
                    calls_table.c.status,
                    calls_table.c.latency_ms,
                )
                .join(node_states_table, calls_table.c.state_id == node_states_table.c.state_id)
                .join(
                    nodes_table,
                    (node_states_table.c.node_id == nodes_table.c.node_id) & (node_states_table.c.run_id == nodes_table.c.run_id),
                )
                .where(node_states_table.c.run_id == run_id)
                .where(calls_table.c.call_type == "llm")
            )

            # LLM calls via operation_id path (source/sink operations)
            llm_op_query = (
                select(
                    nodes_table.c.plugin_name,
                    calls_table.c.call_type,
                    calls_table.c.status,
                    calls_table.c.latency_ms,
                )
                .join(operations_table, calls_table.c.operation_id == operations_table.c.operation_id)
                .join(
                    nodes_table,
                    (operations_table.c.node_id == nodes_table.c.node_id) & (operations_table.c.run_id == nodes_table.c.run_id),
                )
                .where(operations_table.c.run_id == run_id)
                .where(calls_table.c.call_type == "llm")
            )

            # Union both paths and aggregate
            combined_llm = union_all(llm_state_query, llm_op_query).subquery()
            llm_query = select(
                combined_llm.c.plugin_name,
                combined_llm.c.call_type,
                combined_llm.c.status,
                func.count().label("count"),
                func.avg(combined_llm.c.latency_ms).label("avg_latency"),
                func.min(combined_llm.c.latency_ms).label("min_latency"),
                func.max(combined_llm.c.latency_ms).label("max_latency"),
                func.sum(combined_llm.c.latency_ms).label("total_latency"),
            ).group_by(combined_llm.c.plugin_name, combined_llm.c.call_type, combined_llm.c.status)
            llm_rows = conn.execute(llm_query).fetchall()

            # Count all call types (both state_id and operation_id paths)
            call_types_state_query = (
                select(calls_table.c.call_type)
                .join(node_states_table, calls_table.c.state_id == node_states_table.c.state_id)
                .where(node_states_table.c.run_id == run_id)
            )
            call_types_op_query = (
                select(calls_table.c.call_type)
                .join(operations_table, calls_table.c.operation_id == operations_table.c.operation_id)
                .where(operations_table.c.run_id == run_id)
            )
            combined_types = union_all(call_types_state_query, call_types_op_query).subquery()
            call_types_query = select(
                combined_types.c.call_type,
                func.count().label("count"),
            ).group_by(combined_types.c.call_type)
            call_type_rows = conn.execute(call_types_query).fetchall()

        if not llm_rows and not call_type_rows:
            return {
                "run_id": run_id,
                "message": "No external calls found in this run",
                "call_types": {},
            }

        call_type_summary = {row.call_type: row.count for row in call_type_rows}

        llm_by_plugin: dict[str, dict[str, Any]] = {}
        for row in llm_rows:
            plugin = row.plugin_name
            if plugin not in llm_by_plugin:
                llm_by_plugin[plugin] = {
                    "total_calls": 0,
                    "successful": 0,
                    "failed": 0,
                    "avg_latency_ms": 0.0,
                    "total_latency_ms": 0,
                }
            stats = llm_by_plugin[plugin]
            call_count: int = row.count  # type: ignore[assignment]
            stats["total_calls"] += call_count
            if row.status == CallStatus.SUCCESS.value:
                stats["successful"] += call_count
            else:
                stats["failed"] += call_count
            stats["total_latency_ms"] += row.total_latency or 0

        # Calculate averages
        for _plugin, stats in llm_by_plugin.items():
            if stats["total_calls"] > 0:
                stats["avg_latency_ms"] = round(float(stats["total_latency_ms"]) / stats["total_calls"], 2)

        total_llm_calls = sum(s["total_calls"] for s in llm_by_plugin.values())
        total_llm_latency = sum(s["total_latency_ms"] for s in llm_by_plugin.values())

        return {
            "run_id": run_id,
            "call_types": call_type_summary,
            "llm_summary": {
                "total_calls": total_llm_calls,
                "total_latency_ms": total_llm_latency,
                "avg_latency_ms": round(total_llm_latency / total_llm_calls, 2) if total_llm_calls > 0 else None,
            },
            "by_plugin": llm_by_plugin,
        }

    def describe_schema(self) -> dict[str, Any]:
        """Describe the database schema for ad-hoc query exploration.

        Returns:
            Schema description with tables and columns
        """
        from sqlalchemy import inspect

        inspector = inspect(self._db.engine)
        tables = {}

        for table_name in inspector.get_table_names():
            columns = []
            for col in inspector.get_columns(table_name):
                columns.append(
                    {
                        "name": col["name"],
                        "type": str(col["type"]),
                        "nullable": col.get("nullable", True),
                    }
                )

            # Get primary key
            pk = inspector.get_pk_constraint(table_name)
            pk_columns = pk.get("constrained_columns", []) if pk else []

            # Get foreign keys
            fks = inspector.get_foreign_keys(table_name)
            fk_info = [
                {
                    "columns": fk["constrained_columns"],
                    "references": f"{fk['referred_table']}({', '.join(fk['referred_columns'])})",
                }
                for fk in fks
            ]

            tables[table_name] = {
                "columns": columns,
                "primary_key": pk_columns,
                "foreign_keys": fk_info,
            }

        return {
            "tables": tables,
            "table_count": len(tables),
            "hint": "Use the 'query' tool with SELECT statements to explore data",
        }

    def get_outcome_analysis(self, run_id: str) -> dict[str, Any]:
        """Analyze token outcomes for a run.

        Shows terminal state distribution, fork/join patterns, and sink routing.

        Args:
            run_id: Run ID to analyze

        Returns:
            Outcome analysis with distributions and patterns
        """
        from sqlalchemy import func, select

        from elspeth.core.landscape.schema import token_outcomes_table

        run = self._recorder.get_run(run_id)
        if run is None:
            return {"error": f"Run '{run_id}' not found"}

        with self._db.connection() as conn:
            # Outcome distribution
            outcome_dist = (
                select(
                    token_outcomes_table.c.outcome,
                    token_outcomes_table.c.is_terminal,
                    func.count(token_outcomes_table.c.outcome_id).label("count"),
                )
                .where(token_outcomes_table.c.run_id == run_id)
                .group_by(token_outcomes_table.c.outcome, token_outcomes_table.c.is_terminal)
            )
            outcome_rows = conn.execute(outcome_dist).fetchall()

            # Sink distribution
            sink_dist = (
                select(
                    token_outcomes_table.c.sink_name,
                    func.count(token_outcomes_table.c.outcome_id).label("count"),
                )
                .where((token_outcomes_table.c.run_id == run_id) & (token_outcomes_table.c.sink_name.isnot(None)))
                .group_by(token_outcomes_table.c.sink_name)
            )
            sink_rows = conn.execute(sink_dist).fetchall()

            # Fork/join counts (outcomes with fork_group_id or join_group_id, scoped to run)
            fork_count = (
                conn.execute(
                    select(func.count(func.distinct(token_outcomes_table.c.fork_group_id)))
                    .select_from(token_outcomes_table)
                    .where((token_outcomes_table.c.run_id == run_id) & (token_outcomes_table.c.fork_group_id.isnot(None)))
                ).scalar()
                or 0
            )

            join_count = (
                conn.execute(
                    select(func.count(func.distinct(token_outcomes_table.c.join_group_id)))
                    .select_from(token_outcomes_table)
                    .where((token_outcomes_table.c.run_id == run_id) & (token_outcomes_table.c.join_group_id.isnot(None)))
                ).scalar()
                or 0
            )

        outcomes = [
            {
                "outcome": row.outcome,
                "is_terminal": row.is_terminal,
                "count": row.count,
            }
            for row in outcome_rows
        ]

        sinks = {row.sink_name: row.count for row in sink_rows}

        terminal_count = sum(o["count"] for o in outcomes if o["is_terminal"])
        non_terminal_count = sum(o["count"] for o in outcomes if not o["is_terminal"])

        return {
            "run_id": run_id,
            "summary": {
                "terminal_tokens": terminal_count,
                "non_terminal_tokens": non_terminal_count,
                "fork_operations": fork_count,
                "join_operations": join_count,
            },
            "outcome_distribution": outcomes,
            "sink_distribution": sinks,
        }

    # === Emergency Diagnostic Tools (for when everything is on fire) ===

    def diagnose(self) -> dict[str, Any]:
        """Emergency diagnostic: What's broken right now?

        Scans for failed runs, high error rates, stuck runs, and recent problems.
        This is the first tool to use when something is wrong.

        Returns:
            Diagnostic summary with problems, recent failures, and recommendations
        """
        from sqlalchemy import func, select

        from elspeth.core.landscape.schema import (
            operations_table,
            runs_table,
            token_outcomes_table,
            validation_errors_table,
        )

        problems: list[dict[str, Any]] = []
        recommendations: list[str] = []

        with self._db.connection() as conn:
            # Find failed runs (most critical)
            failed_runs = conn.execute(
                select(runs_table.c.run_id, runs_table.c.started_at, runs_table.c.completed_at)
                .where(runs_table.c.status == "failed")
                .order_by(runs_table.c.started_at.desc())
                .limit(5)
            ).fetchall()

            if failed_runs:
                problems.append(
                    {
                        "severity": "CRITICAL",
                        "type": "failed_runs",
                        "count": len(failed_runs),
                        "run_ids": [r.run_id for r in failed_runs],
                        "message": f"{len(failed_runs)} failed run(s) found",
                    }
                )
                recommendations.append("Use get_run_summary(run_id) on failed runs to see error counts")

            # Find stuck runs (running for > 1 hour with no completion)
            stuck_runs = conn.execute(
                select(runs_table.c.run_id, runs_table.c.started_at)
                .where(runs_table.c.status == "running")
                .where(runs_table.c.completed_at.is_(None))
                .order_by(runs_table.c.started_at)
                .limit(5)
            ).fetchall()

            if stuck_runs:
                problems.append(
                    {
                        "severity": "WARNING",
                        "type": "stuck_runs",
                        "count": len(stuck_runs),
                        "run_ids": [r.run_id for r in stuck_runs],
                        "message": f"{len(stuck_runs)} run(s) still in 'running' status",
                    }
                )
                recommendations.append("Check if process is still running; may need manual intervention")

            # Find stuck operations (open for > 1 hour)
            # These indicate source/sink I/O that never completed
            stuck_ops = conn.execute(
                select(
                    operations_table.c.operation_id,
                    operations_table.c.run_id,
                    operations_table.c.operation_type,
                    operations_table.c.started_at,
                )
                .where(operations_table.c.status == "open")
                .where(operations_table.c.started_at < (datetime.now(UTC) - __import__("datetime").timedelta(hours=1)))
                .order_by(operations_table.c.started_at)
                .limit(5)
            ).fetchall()

            if stuck_ops:
                problems.append(
                    {
                        "severity": "WARNING",
                        "type": "stuck_operations",
                        "count": len(stuck_ops),
                        "operations": [
                            {
                                "operation_id": op.operation_id[:12] + "...",
                                "run_id": op.run_id[:12] + "...",
                                "type": op.operation_type,
                                "started": op.started_at.isoformat() if op.started_at else None,
                            }
                            for op in stuck_ops
                        ],
                        "message": f"{len(stuck_ops)} operation(s) stuck in 'open' status for > 1 hour",
                    }
                )
                recommendations.append(
                    "Stuck operations may indicate: (1) streaming source still active, "
                    "(2) process crashed during I/O, (3) database connection issue"
                )
                recommendations.append("Use list_operations(run_id, status='open') to see all open operations")

            # Find runs with high error rates
            error_rates = conn.execute(
                select(
                    runs_table.c.run_id,
                    func.count(validation_errors_table.c.error_id).label("val_errors"),
                )
                .outerjoin(validation_errors_table, runs_table.c.run_id == validation_errors_table.c.run_id)
                .where(runs_table.c.status == "completed")
                .group_by(runs_table.c.run_id)
                .having(func.count(validation_errors_table.c.error_id) > 10)
                .order_by(func.count(validation_errors_table.c.error_id).desc())
                .limit(5)
            ).fetchall()

            if error_rates:
                problems.append(
                    {
                        "severity": "WARNING",
                        "type": "high_error_rates",
                        "runs": [{"run_id": r.run_id, "errors": r.val_errors} for r in error_rates],
                        "message": f"{len(error_rates)} run(s) with high error counts",
                    }
                )
                recommendations.append("Use get_error_analysis(run_id) to identify error patterns")

            # Get recent run summary
            recent_runs = conn.execute(
                select(
                    runs_table.c.run_id,
                    runs_table.c.status,
                    runs_table.c.started_at,
                    runs_table.c.completed_at,
                )
                .order_by(runs_table.c.started_at.desc())
                .limit(10)
            ).fetchall()

            # Count QUARANTINED outcomes across recent runs
            quarantined_count = (
                conn.execute(
                    select(func.count(token_outcomes_table.c.outcome_id)).where(token_outcomes_table.c.outcome == "quarantined")
                ).scalar()
                or 0
            )

            if quarantined_count > 0:
                problems.append(
                    {
                        "severity": "INFO",
                        "type": "quarantined_rows",
                        "count": quarantined_count,
                        "message": f"{quarantined_count} row(s) have been quarantined across all runs",
                    }
                )
                recommendations.append("Quarantined rows indicate data quality issues at source")

        recent_summary = [
            {
                "run_id": r.run_id[:12] + "...",
                "status": r.status,
                "started": r.started_at.isoformat() if r.started_at else None,
            }
            for r in recent_runs
        ]

        return {
            "status": "CRITICAL" if any(p["severity"] == "CRITICAL" for p in problems) else "OK" if not problems else "WARNING",
            "problems": problems,
            "recent_runs": recent_summary,
            "recommendations": recommendations,
            "next_steps": [
                "Use list_runs() to see all runs with their status",
                "Use get_run_summary(run_id) on a specific run for details",
                "Use get_error_analysis(run_id) to understand error patterns",
                "Use explain_token(run_id, token_id=...) to trace a specific failure",
            ],
        }

    def get_failure_context(self, run_id: str, limit: int = 10) -> dict[str, Any]:
        """Get comprehensive context about failures in a run.

        Use this when investigating why a run failed. Returns failed node states,
        error samples, and the surrounding context.

        Args:
            run_id: Run ID to investigate
            limit: Max failures to return

        Returns:
            Failure context with node states, errors, and recommendations
        """
        from sqlalchemy import select

        from elspeth.core.landscape.schema import (
            node_states_table,
            nodes_table,
            transform_errors_table,
            validation_errors_table,
        )

        run = self._recorder.get_run(run_id)
        if run is None:
            return {"error": f"Run '{run_id}' not found"}

        with self._db.connection() as conn:
            # Get failed node states with node info using composite key
            failed_states = conn.execute(
                select(
                    node_states_table.c.state_id,
                    node_states_table.c.token_id,
                    node_states_table.c.node_id,
                    node_states_table.c.step_index,
                    node_states_table.c.attempt,
                    node_states_table.c.started_at,
                    node_states_table.c.completed_at,
                    nodes_table.c.plugin_name,
                    nodes_table.c.node_type,
                )
                .join(
                    nodes_table,
                    (node_states_table.c.node_id == nodes_table.c.node_id) & (node_states_table.c.run_id == nodes_table.c.run_id),
                )
                .where(node_states_table.c.run_id == run_id)
                .where(node_states_table.c.status == "failed")
                .order_by(node_states_table.c.started_at.desc())
                .limit(limit)
            ).fetchall()

            # Get transform errors with details
            transform_errors = conn.execute(
                select(
                    transform_errors_table.c.token_id,
                    transform_errors_table.c.transform_id,
                    transform_errors_table.c.error_details_json,
                    nodes_table.c.plugin_name,
                )
                .outerjoin(
                    nodes_table,
                    (transform_errors_table.c.transform_id == nodes_table.c.node_id)
                    & (transform_errors_table.c.run_id == nodes_table.c.run_id),
                )
                .where(transform_errors_table.c.run_id == run_id)
                .order_by(transform_errors_table.c.created_at.desc())
                .limit(limit)
            ).fetchall()

            # Get validation errors
            validation_errors = conn.execute(
                select(
                    validation_errors_table.c.node_id,
                    validation_errors_table.c.row_hash,
                    validation_errors_table.c.row_data_json,
                    nodes_table.c.plugin_name,
                )
                .outerjoin(
                    nodes_table,
                    (validation_errors_table.c.node_id == nodes_table.c.node_id)
                    & (validation_errors_table.c.run_id == nodes_table.c.run_id),
                )
                .where(validation_errors_table.c.run_id == run_id)
                .limit(limit)
            ).fetchall()

        failed_state_list = [
            {
                "state_id": s.state_id,
                "token_id": s.token_id,
                "plugin": s.plugin_name,
                "type": s.node_type,
                "step": s.step_index,
                "attempt": s.attempt,
                "started": s.started_at.isoformat() if s.started_at else None,
            }
            for s in failed_states
        ]

        transform_error_list = [
            {
                "token_id": e.token_id,
                "plugin": e.plugin_name,
                "details": json.loads(e.error_details_json) if e.error_details_json else None,
            }
            for e in transform_errors
        ]

        validation_error_list = [
            {
                "plugin": e.plugin_name,
                "row_hash": e.row_hash[:12] + "..." if e.row_hash else None,
                "sample_data": json.loads(e.row_data_json) if e.row_data_json else None,
            }
            for e in validation_errors
        ]

        # Identify patterns
        plugins_with_failures = list({s["plugin"] for s in failed_state_list if s["plugin"]})
        has_retries = any(s["attempt"] > 1 for s in failed_state_list)

        return {
            "run_id": run_id,
            "run_status": run.status.value,
            "failed_node_states": failed_state_list,
            "transform_errors": transform_error_list,
            "validation_errors": validation_error_list,
            "patterns": {
                "plugins_failing": plugins_with_failures,
                "has_retries": has_retries,
                "failure_count": len(failed_state_list),
                "transform_error_count": len(transform_error_list),
                "validation_error_count": len(validation_error_list),
            },
            "next_steps": [
                f"Use explain_token(run_id='{run_id}', token_id='...') to trace a specific failure",
                "Check transform error details for exception messages",
                "Look at validation errors for data quality issues at source",
            ],
        }

    def get_recent_activity(self, minutes: int = 60) -> dict[str, Any]:
        """Get recent pipeline activity timeline.

        Use this to understand what happened recently when investigating issues.

        Args:
            minutes: Look back this many minutes (default 60)

        Returns:
            Timeline of recent runs and their status
        """
        from datetime import timedelta

        from sqlalchemy import func, select

        from elspeth.core.landscape.schema import (
            node_states_table,
            rows_table,
            runs_table,
        )

        cutoff = datetime.now(tz=UTC) - timedelta(minutes=minutes)

        with self._db.connection() as conn:
            # Get runs started in the time window
            recent_runs = conn.execute(
                select(
                    runs_table.c.run_id,
                    runs_table.c.status,
                    runs_table.c.started_at,
                    runs_table.c.completed_at,
                )
                .where(runs_table.c.started_at >= cutoff)
                .order_by(runs_table.c.started_at.desc())
            ).fetchall()

            # Get processing stats for each run
            run_stats = []
            for run in recent_runs:
                row_count = (
                    conn.execute(select(func.count()).select_from(rows_table).where(rows_table.c.run_id == run.run_id)).scalar() or 0
                )

                state_count = (
                    conn.execute(
                        select(func.count()).select_from(node_states_table).where(node_states_table.c.run_id == run.run_id)
                    ).scalar()
                    or 0
                )

                duration = None
                if run.started_at and run.completed_at:
                    duration = (run.completed_at - run.started_at).total_seconds()

                run_stats.append(
                    {
                        "run_id": run.run_id[:12] + "...",
                        "full_run_id": run.run_id,
                        "status": run.status,
                        "started": run.started_at.isoformat() if run.started_at else None,
                        "duration_seconds": duration,
                        "rows_processed": row_count,
                        "node_executions": state_count,
                    }
                )

        status_counts: dict[str, int] = {}
        for r in run_stats:
            status_counts[r["status"]] = status_counts.get(r["status"], 0) + 1

        return {
            "time_window_minutes": minutes,
            "total_runs": len(run_stats),
            "status_summary": status_counts,
            "runs": run_stats,
        }


def create_server(database_url: str) -> Server:
    """Create MCP server with Landscape analysis tools.

    Args:
        database_url: SQLAlchemy connection URL

    Returns:
        Configured MCP Server
    """
    server = Server("elspeth-landscape")
    analyzer = LandscapeAnalyzer(database_url)

    @server.list_tools()  # type: ignore[misc, no-untyped-call, untyped-decorator]
    async def list_tools() -> list[Tool]:
        """List available analysis tools."""
        return [
            Tool(
                name="list_runs",
                description="List pipeline runs with optional status filter",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "limit": {"type": "integer", "description": "Max runs to return (default 50)", "default": 50},
                        "status": {
                            "type": "string",
                            "description": "Filter by status",
                            "enum": ["running", "completed", "failed"],
                        },
                    },
                },
            ),
            Tool(
                name="get_run",
                description="Get details of a specific pipeline run",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "run_id": {"type": "string", "description": "Run ID to retrieve"},
                    },
                    "required": ["run_id"],
                },
            ),
            Tool(
                name="get_run_summary",
                description="Get summary statistics for a run: counts, durations, errors, outcome distribution",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "run_id": {"type": "string", "description": "Run ID to analyze"},
                    },
                    "required": ["run_id"],
                },
            ),
            Tool(
                name="list_nodes",
                description="List all nodes (plugin instances) for a run",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "run_id": {"type": "string", "description": "Run ID to query"},
                    },
                    "required": ["run_id"],
                },
            ),
            Tool(
                name="list_rows",
                description="List source rows for a run with pagination",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "run_id": {"type": "string", "description": "Run ID to query"},
                        "limit": {"type": "integer", "description": "Max rows (default 100)", "default": 100},
                        "offset": {"type": "integer", "description": "Rows to skip (default 0)", "default": 0},
                    },
                    "required": ["run_id"],
                },
            ),
            Tool(
                name="list_tokens",
                description="List tokens for a run or specific row",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "run_id": {"type": "string", "description": "Run ID to query"},
                        "row_id": {"type": "string", "description": "Optional row ID to filter by"},
                        "limit": {"type": "integer", "description": "Max tokens (default 100)", "default": 100},
                    },
                    "required": ["run_id"],
                },
            ),
            Tool(
                name="list_operations",
                description="List source/sink operations for a run (blob downloads, file writes, database inserts)",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "run_id": {"type": "string", "description": "Run ID to query"},
                        "operation_type": {
                            "type": "string",
                            "description": "Filter by type",
                            "enum": ["source_load", "sink_write"],
                        },
                        "status": {
                            "type": "string",
                            "description": "Filter by status",
                            "enum": ["open", "completed", "failed", "pending"],
                        },
                        "limit": {"type": "integer", "description": "Max operations (default 100)", "default": 100},
                    },
                    "required": ["run_id"],
                },
            ),
            Tool(
                name="get_operation_calls",
                description="Get external calls (HTTP, SQL, etc.) made during a source/sink operation",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "operation_id": {"type": "string", "description": "Operation ID to query"},
                    },
                    "required": ["operation_id"],
                },
            ),
            Tool(
                name="explain_token",
                description="Get complete lineage for a token: source row, node states, calls, routing, errors, outcome",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "run_id": {"type": "string", "description": "Run ID"},
                        "token_id": {"type": "string", "description": "Token ID (preferred for DAGs with forks)"},
                        "row_id": {"type": "string", "description": "Row ID (alternative to token_id)"},
                        "sink": {"type": "string", "description": "Sink name to disambiguate multiple terminals"},
                    },
                    "required": ["run_id"],
                },
            ),
            Tool(
                name="get_errors",
                description="Get validation and/or transform errors for a run",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "run_id": {"type": "string", "description": "Run ID to query"},
                        "error_type": {
                            "type": "string",
                            "description": "Error type filter",
                            "enum": ["all", "validation", "transform"],
                            "default": "all",
                        },
                        "limit": {"type": "integer", "description": "Max errors per type (default 100)", "default": 100},
                    },
                    "required": ["run_id"],
                },
            ),
            Tool(
                name="get_node_states",
                description="Get node states (processing records) for a run",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "run_id": {"type": "string", "description": "Run ID to query"},
                        "node_id": {"type": "string", "description": "Optional node ID filter"},
                        "status": {
                            "type": "string",
                            "description": "Optional status filter",
                            "enum": ["open", "pending", "completed", "failed"],
                        },
                        "limit": {"type": "integer", "description": "Max states (default 100)", "default": 100},
                    },
                    "required": ["run_id"],
                },
            ),
            Tool(
                name="get_calls",
                description="Get external calls (LLM, HTTP, etc.) for a node state",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "state_id": {"type": "string", "description": "Node state ID"},
                    },
                    "required": ["state_id"],
                },
            ),
            Tool(
                name="query",
                description="Execute a read-only SQL query against the audit database (SELECT only)",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "sql": {"type": "string", "description": "SQL SELECT query"},
                        "params": {"type": "object", "description": "Optional query parameters"},
                    },
                    "required": ["sql"],
                },
            ),
            # === Precomputed Analysis Tools ===
            Tool(
                name="get_dag_structure",
                description="Get the DAG structure for a run: nodes, edges, and mermaid diagram for visualization",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "run_id": {"type": "string", "description": "Run ID to analyze"},
                    },
                    "required": ["run_id"],
                },
            ),
            Tool(
                name="get_performance_report",
                description="Get performance analysis: slow nodes, bottlenecks, timing statistics, high-variance nodes",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "run_id": {"type": "string", "description": "Run ID to analyze"},
                    },
                    "required": ["run_id"],
                },
            ),
            Tool(
                name="get_error_analysis",
                description="Analyze errors: grouped by type, by node, with sample data for pattern matching",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "run_id": {"type": "string", "description": "Run ID to analyze"},
                    },
                    "required": ["run_id"],
                },
            ),
            Tool(
                name="get_llm_usage_report",
                description="Get LLM usage statistics: call counts, latencies, success rates by plugin",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "run_id": {"type": "string", "description": "Run ID to analyze"},
                    },
                    "required": ["run_id"],
                },
            ),
            Tool(
                name="describe_schema",
                description="Describe the database schema: tables, columns, primary keys, foreign keys (for ad-hoc SQL)",
                inputSchema={
                    "type": "object",
                    "properties": {},
                },
            ),
            Tool(
                name="get_outcome_analysis",
                description="Analyze token outcomes: terminal states, fork/join patterns, sink routing distribution",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "run_id": {"type": "string", "description": "Run ID to analyze"},
                    },
                    "required": ["run_id"],
                },
            ),
            # === Emergency Diagnostic Tools ===
            Tool(
                name="diagnose",
                description="🚨 EMERGENCY: What's broken right now? Scans for failed runs, stuck runs, high error rates",
                inputSchema={
                    "type": "object",
                    "properties": {},
                },
            ),
            Tool(
                name="get_failure_context",
                description="🔍 Deep dive: Get comprehensive context about failures in a run (failed states, errors, patterns)",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "run_id": {"type": "string", "description": "Run ID to investigate"},
                        "limit": {"type": "integer", "description": "Max failures to return", "default": 10},
                    },
                    "required": ["run_id"],
                },
            ),
            Tool(
                name="get_recent_activity",
                description="📊 Timeline: What happened recently? Shows runs in the last N minutes",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "minutes": {"type": "integer", "description": "Look back this many minutes", "default": 60},
                    },
                },
            ),
        ]

    @server.call_tool()  # type: ignore[misc, untyped-decorator]
    async def call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
        """Handle tool calls."""
        try:
            result: Any
            if name == "list_runs":
                result = analyzer.list_runs(
                    limit=arguments.get("limit", 50),
                    status=arguments.get("status"),
                )
            elif name == "get_run":
                result = analyzer.get_run(arguments["run_id"])
            elif name == "get_run_summary":
                result = analyzer.get_run_summary(arguments["run_id"])
            elif name == "list_nodes":
                result = analyzer.list_nodes(arguments["run_id"])
            elif name == "list_rows":
                result = analyzer.list_rows(
                    run_id=arguments["run_id"],
                    limit=arguments.get("limit", 100),
                    offset=arguments.get("offset", 0),
                )
            elif name == "list_tokens":
                result = analyzer.list_tokens(
                    run_id=arguments["run_id"],
                    row_id=arguments.get("row_id"),
                    limit=arguments.get("limit", 100),
                )
            elif name == "list_operations":
                result = analyzer.list_operations(
                    run_id=arguments["run_id"],
                    operation_type=arguments.get("operation_type"),
                    status=arguments.get("status"),
                    limit=arguments.get("limit", 100),
                )
            elif name == "get_operation_calls":
                result = analyzer.get_operation_calls(arguments["operation_id"])
            elif name == "explain_token":
                result = analyzer.explain_token(
                    run_id=arguments["run_id"],
                    token_id=arguments.get("token_id"),
                    row_id=arguments.get("row_id"),
                    sink=arguments.get("sink"),
                )
            elif name == "get_errors":
                result = analyzer.get_errors(
                    run_id=arguments["run_id"],
                    error_type=arguments.get("error_type", "all"),
                    limit=arguments.get("limit", 100),
                )
            elif name == "get_node_states":
                result = analyzer.get_node_states(
                    run_id=arguments["run_id"],
                    node_id=arguments.get("node_id"),
                    status=arguments.get("status"),
                    limit=arguments.get("limit", 100),
                )
            elif name == "get_calls":
                result = analyzer.get_calls(arguments["state_id"])
            elif name == "query":
                result = analyzer.query(
                    sql=arguments["sql"],
                    params=arguments.get("params"),
                )
            # === Precomputed Analysis Tools ===
            elif name == "get_dag_structure":
                result = analyzer.get_dag_structure(arguments["run_id"])
            elif name == "get_performance_report":
                result = analyzer.get_performance_report(arguments["run_id"])
            elif name == "get_error_analysis":
                result = analyzer.get_error_analysis(arguments["run_id"])
            elif name == "get_llm_usage_report":
                result = analyzer.get_llm_usage_report(arguments["run_id"])
            elif name == "describe_schema":
                result = analyzer.describe_schema()
            elif name == "get_outcome_analysis":
                result = analyzer.get_outcome_analysis(arguments["run_id"])
            # === Emergency Diagnostic Tools ===
            elif name == "diagnose":
                result = analyzer.diagnose()
            elif name == "get_failure_context":
                result = analyzer.get_failure_context(
                    run_id=arguments["run_id"],
                    limit=arguments.get("limit", 10),
                )
            elif name == "get_recent_activity":
                result = analyzer.get_recent_activity(
                    minutes=arguments.get("minutes", 60),
                )
            else:
                return [TextContent(type="text", text=f"Unknown tool: {name}")]

            return [TextContent(type="text", text=json.dumps(result, indent=2))]

        except Exception as e:
            return [TextContent(type="text", text=f"Error: {e!s}")]

    return server


async def run_server(database_url: str) -> None:
    """Run the MCP server with stdio transport."""
    server = create_server(database_url)
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


def _find_audit_databases(search_dir: str, max_depth: int = 5) -> list[str]:
    """Find potential audit databases in the given directory.

    Looks for .db files that might be ELSPETH audit databases,
    prioritizing files named 'audit.db' or 'landscape.db'.

    Args:
        search_dir: Directory to search from
        max_depth: Maximum directory depth to search

    Returns:
        List of absolute paths to found database files, sorted by relevance
    """
    from pathlib import Path

    found: list[tuple[int, float, str]] = []  # (priority, neg_mtime, path)
    search_path = Path(search_dir).resolve()

    for db_file in search_path.rglob("*.db"):
        # Skip hidden directories and common non-audit locations
        parts = db_file.relative_to(search_path).parts
        if any(p.startswith(".") for p in parts):
            continue
        if len(parts) > max_depth:
            continue
        if "node_modules" in parts or "__pycache__" in parts:
            continue

        # Prioritize by name and location
        name = db_file.name.lower()
        in_runs_dir = "runs" in parts

        # Databases in runs/ directories are likely active pipeline outputs
        if in_runs_dir and name == "audit.db":
            priority = 0
        elif in_runs_dir and "audit" in name:
            priority = 1
        elif name == "audit.db":
            priority = 2
        elif name == "landscape.db":
            priority = 3
        elif "audit" in name:
            priority = 4
        elif "landscape" in name:
            priority = 5
        else:
            priority = 10

        # Get modification time for sorting (most recent first)
        try:
            mtime = db_file.stat().st_mtime
        except OSError:
            mtime = 0

        found.append((priority, -mtime, str(db_file)))

    # Sort by priority, then by modification time (most recent first via negative mtime)
    found.sort(key=lambda x: (x[0], x[1]))
    return [path for _, _, path in found]


def _prompt_for_database(databases: list[str], search_dir: str) -> str | None:
    """Prompt user to select a database from the list.

    Args:
        databases: List of database paths
        search_dir: Directory that was searched (for display)

    Returns:
        Selected database path, or None if user cancelled
    """
    from pathlib import Path

    search_path = Path(search_dir).resolve()

    sys.stderr.write(f"\nFound {len(databases)} database(s) in {search_path}:\n\n")

    for i, db_path in enumerate(databases, 1):
        # Show relative path if possible
        try:
            rel_path = Path(db_path).relative_to(search_path)
            display = f"./{rel_path}"
        except ValueError:
            display = db_path
        sys.stderr.write(f"  [{i}] {display}\n")

    sys.stderr.write("\n")

    while True:
        sys.stderr.write("Select database [1]: ")
        sys.stderr.flush()

        try:
            choice = input().strip()
        except (EOFError, KeyboardInterrupt):
            sys.stderr.write("\nCancelled.\n")
            return None

        if not choice:
            choice = "1"

        try:
            idx = int(choice) - 1
            if 0 <= idx < len(databases):
                return databases[idx]
            sys.stderr.write(f"Please enter a number between 1 and {len(databases)}\n")
        except ValueError:
            sys.stderr.write("Please enter a number\n")


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="ELSPETH Landscape MCP Server - Audit database analysis tools",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run with SQLite database
    elspeth-mcp --database sqlite:///./state/audit.db

    # Run with PostgreSQL
    elspeth-mcp --database postgresql://user:pass@host/dbname

    # Interactive mode - finds and prompts for databases
    elspeth-mcp

Environment Variables:
    ELSPETH_DATABASE_URL: Default database URL if --database not specified
""",
    )
    parser.add_argument(
        "--database",
        "-d",
        default=None,
        help="Database connection URL (SQLAlchemy format)",
    )
    parser.add_argument(
        "--search-dir",
        default=".",
        help="Directory to search for databases (default: current directory)",
    )

    args = parser.parse_args()

    # Get database URL from args or environment
    import os

    database_url: str | None = args.database
    if database_url is None and "ELSPETH_DATABASE_URL" in os.environ:
        database_url = os.environ["ELSPETH_DATABASE_URL"]

    if database_url is None:
        # Auto-discovery mode: find databases in search directory
        databases = _find_audit_databases(args.search_dir)

        if not databases:
            sys.stderr.write(f"No .db files found in {os.path.abspath(args.search_dir)}\n")
            sys.stderr.write("Use --database to specify a database URL directly.\n")
            sys.exit(1)

        # Check if we're running interactively (TTY) or as MCP server (stdio)
        is_interactive = sys.stdin.isatty()

        db_path: str
        if len(databases) == 1:
            # Only one database - use it directly
            db_path = databases[0]
            sys.stderr.write(f"Using database: {db_path}\n")
        elif is_interactive:
            # Multiple databases in interactive mode - prompt for selection
            selected = _prompt_for_database(databases, args.search_dir)
            if selected is None:
                sys.exit(1)
            db_path = selected
        else:
            # Multiple databases in non-interactive mode - use best match
            db_path = databases[0]
            sys.stderr.write(f"Auto-selected database: {db_path}\n")
            sys.stderr.write("(Use --database to specify a different one)\n")

        database_url = f"sqlite:///{db_path}"

    import asyncio

    asyncio.run(run_server(database_url))


if __name__ == "__main__":
    main()
