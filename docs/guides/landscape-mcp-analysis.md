# Landscape MCP Analysis Server

Current as of 2026-05-20.

`elspeth-mcp` is a read-only MCP server for querying an ELSPETH Landscape audit
database. It is intended for debugging, audit investigation, and agent-assisted
analysis. It is not a production write surface.

The authoritative tool registry is `_TOOLS` in
`src/elspeth/mcp/server.py`; analyzer behavior is implemented through
`src/elspeth/mcp/analyzer.py` and `src/elspeth/mcp/analyzers/`.

## Quick Start

Install with MCP support:

```bash
uv pip install -e ".[mcp]"
```

Run with database auto-discovery:

```bash
elspeth-mcp
```

Run against an explicit Landscape database:

```bash
elspeth-mcp --database sqlite:///./examples/threshold_gate/runs/audit.db
```

Or use an environment variable:

```bash
export ELSPETH_DATABASE_URL=sqlite:///./state/audit.db
elspeth-mcp
```

## Database Discovery

When `--database` is not provided, the server searches downward from the current
directory, prefers `runs/audit.db`, accepts `landscape.db`, sorts candidates by
modification time, and selects the best non-interactively for MCP clients. In an
interactive terminal it can prompt for the target database.

## Claude Code Configuration

```json
{
  "mcpServers": {
    "elspeth-landscape": {
      "command": "elspeth-mcp",
      "args": [],
      "description": "ELSPETH Landscape audit database analysis"
    }
  }
}
```

With an explicit database:

```json
{
  "mcpServers": {
    "elspeth-landscape": {
      "command": "elspeth-mcp",
      "args": ["--database", "sqlite:///./examples/my_pipeline/runs/audit.db"]
    }
  }
}
```

## When To Use

- Diagnose failed, stuck, or high-error runs.
- Explain a row or token through the pipeline DAG.
- Inspect source/sink operations and their external calls.
- Investigate routing, batch, coalesce, and terminal-outcome behavior.
- Inspect schema contracts and contract violations for a run.
- Run tightly scoped read-only SQL when the maintained analysis tools are not
  enough.

For operator command-line investigations, start with
[`elspeth explain`](../runbooks/investigate-routing.md). Use MCP when an MCP
client needs structured, iterative audit queries.

## Tool Registry

The current registry exposes 27 tools:

| Category | Tools |
|----------|-------|
| Run queries | `list_runs`, `get_run`, `get_run_summary` |
| Graph and rows | `list_nodes`, `list_rows`, `list_tokens`, `get_token_children`, `get_dag_structure` |
| Lineage | `explain_token`, `get_node_states`, `get_calls`, `get_errors`, `list_collisions` |
| Operations | `list_operations`, `get_operation_calls` |
| Reports | `get_performance_report`, `get_error_analysis`, `get_llm_usage_report`, `get_outcome_analysis`, `describe_schema` |
| Diagnostics | `diagnose`, `get_failure_context`, `get_recent_activity` |
| Contracts | `get_run_contract`, `explain_field`, `list_contract_violations` |
| SQL | `query` |

Refresh the list from source with:

```bash
python - <<'PY'
from elspeth.mcp.server import _TOOLS
for name in _TOOLS:
    print(name)
PY
```

## Common Workflows

### Something Failed

1. Run `diagnose`.
2. Use `get_recent_activity` to identify the affected run.
3. Use `get_failure_context` for failed states and error patterns.
4. Use `explain_token` on a representative token or row.

### A Row Was Routed Unexpectedly

1. Use `list_rows` to find the row.
2. Use `list_tokens` to find terminal and branch tokens for that row.
3. Use `explain_token` with `token_id` when the row forked or expanded.
4. Use `list_collisions` if merged context fields or union origins are in
   question.

### Source Or Sink I/O Looks Suspicious

1. Use `list_operations` for the run.
2. Use `get_operation_calls` on the relevant operation.
3. Compare operation status, call status, latency, and error detail.

### Contract Or Schema Behavior Is In Question

1. Use `get_run_contract` to inspect the run-level schema contract.
2. Use `explain_field` for a specific field.
3. Use `list_contract_violations` for producer/consumer contract failures.

### Ad Hoc SQL

Use `query` only for read-only investigation. The server rejects non-SELECT
statements and validates SQL before execution, but callers should still prefer
maintained tools when a tool exists for the question.

## Schema Notes

- Landscape uses SQLAlchemy Core table definitions in
  `src/elspeth/core/landscape/schema.py`.
- `nodes` is keyed by `(node_id, run_id)`; joins must include both values when
  joining node records.
- Token outcomes use `completed`, `outcome`, and `path`, not the retired
  `is_terminal` column.
- External calls are parented by either `state_id` or `operation_id`, never both.
- See [Landscape System Architecture](../architecture/landscape.md) for the
  current table grouping and repository layout.
