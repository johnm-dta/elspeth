"""Computed analysis report functions for the Landscape audit database.

Functions: get_run_summary, get_dag_structure, get_performance_report,
get_error_analysis, get_llm_usage_report, describe_schema, get_outcome_analysis.

All functions accept (db, factory) as their first two parameters.
"""

from __future__ import annotations

import json
from typing import Any

from elspeth.contracts.enums import CallStatus, NodeType, RoutingMode
from elspeth.contracts.errors import AuditIntegrityError
from elspeth.core.landscape.database import LandscapeDB
from elspeth.core.landscape.factory import RecorderFactory
from elspeth.mcp.types import (
    DAGStructureReport,
    ErrorAnalysisReport,
    ErrorResult,
    LLMUsageReport,
    OutcomeAnalysisReport,
    PerformanceReport,
    RunSummaryReport,
    SchemaDescription,
)


def get_run_summary(db: LandscapeDB, factory: RecorderFactory, run_id: str) -> RunSummaryReport | ErrorResult:
    """Get summary statistics for a run.

    Args:
        db: Database connection
        factory: Recorder factory
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

    run = factory.run_lifecycle.get_run(run_id)
    if run is None:
        return {"error": f"Run '{run_id}' not found"}

    with db.connection() as conn:
        # Count rows
        row_count = conn.execute(select(func.count()).select_from(rows_table).where(rows_table.c.run_id == run_id)).scalar() or 0

        # Count tokens
        token_count = conn.execute(select(func.count()).select_from(tokens_table).where(tokens_table.c.run_id == run_id)).scalar() or 0

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
            conn.execute(select(func.count()).select_from(transform_errors_table).where(transform_errors_table.c.run_id == run_id)).scalar()
            or 0
        )

        # Get outcome distribution. ADR-019 makes outcome path-aware:
        # outcome is TerminalOutcome.value or NULL, path is always populated.
        outcome_query = (
            select(
                token_outcomes_table.c.outcome,
                token_outcomes_table.c.path,
                token_outcomes_table.c.completed,
                func.count().label("count"),
            )
            .where(token_outcomes_table.c.run_id == run_id)
            .group_by(
                token_outcomes_table.c.outcome,
                token_outcomes_table.c.path,
                token_outcomes_table.c.completed,
            )
        )
        outcome_rows = conn.execute(outcome_query).fetchall()
        outcome_distribution = [
            {
                "outcome": row.outcome,
                "path": row.path,
                "completed": bool(row.completed),
                "count": row.count,
            }
            for row in outcome_rows
        ]

        # Calculate average processing duration per node state
        avg_duration = conn.execute(select(func.avg(node_states_table.c.duration_ms)).where(node_states_table.c.run_id == run_id)).scalar()

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
        "outcome_distribution": outcome_distribution,  # type: ignore[typeddict-item]  # SA Row attr types
        "avg_state_duration_ms": round(avg_duration, 2) if avg_duration is not None else None,
    }


def get_dag_structure(db: LandscapeDB, factory: RecorderFactory, run_id: str) -> DAGStructureReport | ErrorResult:
    """Get the DAG structure for a run as a structured object.

    Returns nodes, edges, and a mermaid diagram for visualization.

    Args:
        db: Database connection
        factory: Recorder factory
        run_id: Run ID to analyze

    Returns:
        DAG structure with nodes, edges, and mermaid diagram
    """
    nodes = factory.data_flow.get_nodes(run_id)
    edges = factory.data_flow.get_edges(run_id)

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
            "flow_type": "divert" if e.default_mode == RoutingMode.DIVERT else "normal",
        }
        for e in edges
    ]

    # Build terminal sink map: processing node → sink name for on_success/MOVE edges
    sink_id_to_name = {n.node_id: n.plugin_name for n in nodes if n.node_type == NodeType.SINK}
    terminal_sink_map: dict[str, str] = {}
    for e in edges:
        if e.label == "on_success" and e.default_mode == RoutingMode.MOVE and e.to_node_id in sink_id_to_name:
            terminal_sink_map[e.from_node_id] = sink_id_to_name[e.to_node_id]

    # Generate mermaid diagram
    # Use sequential aliases (N0, N1, ...) for unique Mermaid node IDs.
    # Truncating node_id to 8 chars caused collisions (e.g. all transforms
    # shared the "transfor" prefix).
    node_alias = {n.node_id: f"N{i}" for i, n in enumerate(nodes)}
    lines = ["graph TD"]
    for n in nodes:
        alias = node_alias[n.node_id]
        label = f"{n.plugin_name}[{n.node_type.value}]"
        lines.append(f'    {alias}["{label}"]')
    for e in edges:
        from_alias = node_alias[e.from_node_id]
        to_alias = node_alias[e.to_node_id]
        if e.default_mode == RoutingMode.DIVERT:
            arrow = f"-.->|{e.label}|"
        elif e.label == "continue":
            arrow = "-->"
        else:
            arrow = f"-->|{e.label}|"
        lines.append(f"    {from_alias} {arrow} {to_alias}")

    return {
        "run_id": run_id,
        "nodes": node_list,  # type: ignore[typeddict-item]  # structurally correct dict literals
        "edges": edge_list,  # type: ignore[typeddict-item]  # functional TypedDict with "from" keyword
        "node_count": len(nodes),
        "edge_count": len(edges),
        "terminal_sink_map": terminal_sink_map,
        "mermaid": "\n".join(lines),
    }


def get_performance_report(db: LandscapeDB, factory: RecorderFactory, run_id: str) -> PerformanceReport | ErrorResult:
    """Get performance analysis for a run.

    Identifies slow nodes, outliers, and processing bottlenecks.

    Args:
        db: Database connection
        factory: Recorder factory
        run_id: Run ID to analyze

    Returns:
        Performance report with timing analysis
    """
    from sqlalchemy import func, select

    from elspeth.core.landscape.schema import node_states_table, nodes_table

    run = factory.run_lifecycle.get_run(run_id)
    if run is None:
        return {"error": f"Run '{run_id}' not found"}

    with db.connection() as conn:
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
                "node_id": row.node_id,
                "plugin": row.plugin_name,
                "type": row.node_type,
                "executions": row.executions,
                "avg_ms": round(row.avg_ms, 2) if row.avg_ms is not None else None,
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
    high_variance = [n for n in node_performance if n["avg_ms"] is not None and n["max_ms"] is not None and n["max_ms"] > 5 * n["avg_ms"]]

    return {
        "run_id": run_id,
        "total_processing_time_ms": total_time_ms,
        "node_count": len(node_performance),
        "bottlenecks": bottlenecks,  # type: ignore[typeddict-item]  # structurally correct dict literals
        "high_variance_nodes": high_variance,  # type: ignore[typeddict-item]
        "node_performance": node_performance,  # type: ignore[typeddict-item]
    }


def get_error_analysis(db: LandscapeDB, factory: RecorderFactory, run_id: str) -> ErrorAnalysisReport | ErrorResult:
    """Analyze errors for a run, grouping by type and identifying patterns.

    Args:
        db: Database connection
        factory: Recorder factory
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

    run = factory.run_lifecycle.get_run(run_id)
    if run is None:
        return {"error": f"Run '{run_id}' not found"}

    with db.connection() as conn:
        # Validation errors by source node using composite key
        val_by_node = (
            select(
                nodes_table.c.plugin_name,
                validation_errors_table.c.schema_mode,
                func.count(validation_errors_table.c.error_id).label("count"),
            )
            .outerjoin(
                nodes_table,
                (validation_errors_table.c.node_id == nodes_table.c.node_id) & (validation_errors_table.c.run_id == nodes_table.c.run_id),
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
            "source_plugin": row.plugin_name,
            "schema_mode": row.schema_mode,
            "count": row.count,
        }
        for row in val_rows
    ]

    # Corruption guard: transform_errors always reference a transform node.
    # A None plugin_name bucket means the outerjoin found no matching node row.
    for row in trans_rows:
        if row.plugin_name is None:
            msg = (
                f"Tier-1 corruption: {row.count} transform_errors row(s) reference "
                f"transform_id(s) with no matching node in nodes table "
                f"for run_id={run_id!r}"
            )
            raise AuditIntegrityError(msg)

    transform_summary = [{"transform_plugin": row.plugin_name, "count": row.count} for row in trans_rows]

    return {
        "run_id": run_id,
        "validation_errors": {
            "total": sum(r["count"] for r in validation_summary),
            "by_source": validation_summary,  # type: ignore[typeddict-item]
            "sample_data": [json.loads(r[0]) if r[0] else None for r in sample_val],
        },
        "transform_errors": {
            "total": sum(r["count"] for r in transform_summary),
            "by_transform": transform_summary,  # type: ignore[typeddict-item]
            "sample_details": [json.loads(r[0]) if r[0] else None for r in sample_trans],
        },
    }


def get_llm_usage_report(db: LandscapeDB, factory: RecorderFactory, run_id: str) -> LLMUsageReport | ErrorResult:
    """Get LLM usage statistics for a run.

    Analyzes external calls that were LLM API calls.

    Args:
        db: Database connection
        factory: Recorder factory
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

    run = factory.run_lifecycle.get_run(run_id)
    if run is None:
        return {"error": f"Run '{run_id}' not found"}

    with db.connection() as conn:
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

    call_type_summary = {row.call_type: row.count for row in call_type_rows}

    if not llm_rows:
        return {
            "run_id": run_id,
            "message": "No LLM calls found in this run",
            "call_types": call_type_summary,  # type: ignore[typeddict-item]  # SA Row attr types
        }

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
        call_count: int = row.count  # type: ignore[assignment]  # SA Row attribute from COUNT() aggregate; typed as Any
        stats["total_calls"] += call_count
        if row.status == CallStatus.SUCCESS:
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
        "call_types": call_type_summary,  # type: ignore[typeddict-item]  # SA Row attr types
        "llm_summary": {
            "total_calls": total_llm_calls,
            "total_latency_ms": total_llm_latency,
            "avg_latency_ms": round(total_llm_latency / total_llm_calls, 2) if total_llm_calls > 0 else None,
        },
        "by_plugin": llm_by_plugin,  # type: ignore[typeddict-item]  # incrementally-built dict
    }


def describe_schema(db: LandscapeDB, factory: RecorderFactory) -> SchemaDescription:
    """Describe the database schema for ad-hoc query exploration.

    Args:
        db: Database connection
        factory: Recorder factory

    Returns:
        Schema description with tables and columns
    """
    from sqlalchemy import inspect

    inspector = inspect(db.engine)
    tables = {}

    for table_name in inspector.get_table_names():
        columns = []
        for col in inspector.get_columns(table_name):
            columns.append(
                {
                    "name": col["name"],
                    "type": str(col["type"]),
                    "nullable": col["nullable"],
                }
            )

        # Get primary key
        pk = inspector.get_pk_constraint(table_name)
        pk_columns = pk["constrained_columns"] if pk else []

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
        "tables": tables,  # type: ignore[typeddict-item]  # nested dict literals from SA inspector
        "table_count": len(tables),
        "hint": "Use the 'query' tool with SELECT statements to explore data",
    }


def get_outcome_analysis(db: LandscapeDB, factory: RecorderFactory, run_id: str) -> OutcomeAnalysisReport | ErrorResult:
    """Analyze token outcomes for a run.

    Shows terminal state distribution, fork/join patterns, and sink routing.

    Args:
        db: Database connection
        factory: Recorder factory
        run_id: Run ID to analyze

    Returns:
        Outcome analysis with distributions and patterns
    """
    from sqlalchemy import func, select

    from elspeth.core.landscape.schema import token_outcomes_table

    run = factory.run_lifecycle.get_run(run_id)
    if run is None:
        return {"error": f"Run '{run_id}' not found"}

    with db.connection() as conn:
        # Outcome distribution
        outcome_dist = (
            select(
                token_outcomes_table.c.outcome,
                token_outcomes_table.c.path,
                token_outcomes_table.c.completed,
                func.count(token_outcomes_table.c.outcome_id).label("count"),
            )
            .where(token_outcomes_table.c.run_id == run_id)
            .group_by(
                token_outcomes_table.c.outcome,
                token_outcomes_table.c.path,
                token_outcomes_table.c.completed,
            )
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
            "path": row.path,
            "completed": bool(row.completed),
            "count": row.count,
        }
        for row in outcome_rows
    ]

    sinks = {row.sink_name: row.count for row in sink_rows}

    terminal_count = sum(o["count"] for o in outcomes if o["completed"])
    non_terminal_count = sum(o["count"] for o in outcomes if not o["completed"])

    return {
        "run_id": run_id,
        "summary": {
            "terminal_tokens": terminal_count,
            "non_terminal_tokens": non_terminal_count,
            "fork_operations": fork_count,
            "join_operations": join_count,
        },
        "outcome_distribution": outcomes,  # type: ignore[typeddict-item]  # structurally correct dict literals
        "sink_distribution": sinks,  # type: ignore[typeddict-item]  # SA Row attr types
    }
