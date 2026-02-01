# Landscape MCP Analysis Server

A lightweight MCP (Model Context Protocol) server for querying and analyzing the ELSPETH audit database. Designed to help agentic AI (or any MCP client) investigate pipeline runs, debug failures, and understand data flow. This is intended for debugging and analysis. It is not approved for production use.

## Quick Start

```bash
# Install with MCP support
uv pip install -e ".[mcp]"

# Run the server with auto-discovery (recommended)
# Finds .db files in current directory, prioritizes audit.db in runs/ directories
elspeth-mcp

# Or specify a database explicitly
elspeth-mcp --database sqlite:///./examples/threshold_gate/runs/audit.db

# Or use environment variable
export ELSPETH_DATABASE_URL=sqlite:///./state/audit.db
elspeth-mcp
```

### Database Auto-Discovery

When run without `--database`, the server automatically discovers SQLite databases:

1. **Searches** the current directory (up to 5 levels deep)
2. **Prioritizes** databases in `runs/` directories (pipeline outputs)
3. **Prefers** files named `audit.db` over `landscape.db`
4. **Sorts** by modification time (most recent first)

**Interactive mode** (terminal): Prompts you to select from found databases
**Non-interactive mode** (MCP): Auto-selects the best match and logs the choice

### Claude Code Configuration

Add to your project's `.mcp.json`:

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

Or specify a database path explicitly:

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

## When To Use This

Use the MCP server when you need to:

- **Debug a failed pipeline run** - Find what went wrong, which rows failed, which transforms errored
- **Investigate data quality issues** - Find quarantined rows, validation errors, schema violations
- **Analyze performance** - Find slow nodes, bottlenecks, high-latency LLM calls
- **Trace lineage** - Follow a specific row through the entire pipeline to see every transform
- **Understand pipeline structure** - Visualize the DAG, see node types and connections

## Tool Reference

### Emergency Diagnostics (Start Here When Things Are Broken)

#### `diagnose`

**Use first when something is wrong.** Scans for failed runs, stuck runs, high error rates.

```
Returns:
- status: "OK", "WARNING", or "CRITICAL"
- problems: List of issues found with severity
- recent_runs: Last 10 runs with status
- recommendations: Suggested next steps
```

#### `get_failure_context`

**Deep dive on a specific failed run.** Returns failed node states, transform errors, validation errors, and patterns.

```
Arguments:
- run_id (required): The run to investigate
- limit: Max failures to return (default 10)
```

#### `get_recent_activity`

**What happened recently?** Shows timeline of runs in the last N minutes.

```
Arguments:
- minutes: Look back window (default 60)
```

### Core Query Tools

#### `list_runs`

List pipeline runs with optional filtering.

```
Arguments:
- limit: Max runs to return (default 50)
- status: Filter by "PENDING", "RUNNING", "COMPLETED", or "FAILED"
```

#### `get_run`

Get full details of a specific run.

```
Arguments:
- run_id (required): The run ID
```

#### `get_run_summary`

Get summary statistics for a run: row counts, token counts, error counts, outcome distribution.

```
Arguments:
- run_id (required): The run ID
```

#### `list_nodes`

List all nodes (plugin instances) registered in a run.

```
Arguments:
- run_id (required): The run ID
```

#### `list_rows`

List source rows for a run with pagination.

```
Arguments:
- run_id (required): The run ID
- limit: Max rows (default 100)
- offset: Rows to skip (default 0)
```

#### `list_tokens`

List tokens (row instances in DAG paths) for a run or specific row.

```
Arguments:
- run_id (required): The run ID
- row_id: Optional filter by source row
- limit: Max tokens (default 100)
```

#### `get_node_states`

Get node states (processing records) showing how each token was processed at each node.

```
Arguments:
- run_id (required): The run ID
- node_id: Optional filter by node
- status: Optional filter by "PENDING", "RUNNING", "COMPLETED", "FAILED"
- limit: Max states (default 100)
```

#### `get_calls`

Get external calls (LLM, HTTP, etc.) made during a specific node state execution.

```
Arguments:
- state_id (required): The node state ID
```

#### `explain_token`

**The lineage tool.** Get complete processing history for a token: source row, all node states, all calls, routing events, errors, and final outcome.

```
Arguments:
- run_id (required): The run ID
- token_id: Token ID (preferred for DAGs with forks)
- row_id: Row ID (alternative - requires disambiguation if multiple terminals)
- sink: Sink name to disambiguate when row has multiple terminal tokens
```

#### `get_errors`

Get validation and/or transform errors for a run.

```
Arguments:
- run_id (required): The run ID
- error_type: "all", "validation", or "transform" (default "all")
- limit: Max errors per type (default 100)
```

#### `query`

Execute a read-only SQL query against the audit database. **SELECT only** - other statements are rejected.

```
Arguments:
- sql (required): SQL SELECT query
- params: Optional query parameters as object
```

### Precomputed Analysis Tools

#### `get_dag_structure`

Get the DAG structure as a structured object with nodes, edges, and a mermaid diagram.

```
Arguments:
- run_id (required): The run ID

Returns:
- nodes: List of {node_id, plugin_name, node_type, sequence}
- edges: List of {from, to, label, mode}
- mermaid: Mermaid diagram source for visualization
```

#### `get_performance_report`

Analyze node performance: timing statistics, bottlenecks, high-variance nodes.

```
Arguments:
- run_id (required): The run ID

Returns:
- total_processing_time_ms
- bottlenecks: Nodes taking >20% of total time
- high_variance_nodes: Nodes where max > 5x average
- node_performance: Per-node stats (avg, min, max, total, failures)
```

#### `get_error_analysis`

Analyze errors grouped by type and source, with sample data for pattern matching.

```
Arguments:
- run_id (required): The run ID

Returns:
- validation_errors: {total, by_source, sample_data}
- transform_errors: {total, by_transform, sample_details}
```

#### `get_llm_usage_report`

Analyze LLM API usage: call counts, latencies, success rates by plugin.

```
Arguments:
- run_id (required): The run ID

Returns:
- call_types: Count by call type (llm, http, etc.)
- llm_summary: {total_calls, total_latency_ms, avg_latency_ms}
- by_plugin: Per-plugin stats
```

#### `get_outcome_analysis`

Analyze token outcomes: terminal state distribution, fork/join patterns, sink routing.

```
Arguments:
- run_id (required): The run ID

Returns:
- summary: {terminal_tokens, non_terminal_tokens, fork_operations, join_operations}
- outcome_distribution: Count by outcome type
- sink_distribution: Count by sink name
```

#### `describe_schema`

Describe the database schema for ad-hoc SQL exploration.

```
Returns:
- tables: Dict of table_name -> {columns, primary_key, foreign_keys}
- table_count: Number of tables
```

## Common Workflows

### "Something failed, what happened?"

```
1. diagnose()                              # What's broken?
2. get_failure_context(run_id="...")       # Deep dive on the failure
3. explain_token(run_id="...", token_id="...")  # Trace a specific failed row
```

### "Pipeline is slow, where's the bottleneck?"

```
1. get_run_summary(run_id="...")           # Overview of the run
2. get_performance_report(run_id="...")    # Find slow nodes
3. get_llm_usage_report(run_id="...")      # If LLM transforms are suspected
```

### "Data quality issues, what's being rejected?"

```
1. get_run_summary(run_id="...")           # Check error counts
2. get_error_analysis(run_id="...")        # See error patterns
3. get_errors(run_id="...", error_type="validation")  # Get sample data
```

### "I need to understand this pipeline"

```
1. get_dag_structure(run_id="...")         # Visualize the DAG
2. list_nodes(run_id="...")                # See all plugins
3. describe_schema()                       # Understand the database
```

### "Ad-hoc investigation"

```
1. describe_schema()                       # Learn the tables
2. query(sql="SELECT ...")                 # Run custom queries
```

## Database Schema Quick Reference

### Key Tables

| Table | Purpose | Key Columns |
|-------|---------|-------------|
| `runs` | Pipeline executions | run_id, status, started_at, completed_at |
| `nodes` | Plugin instances | node_id, run_id (composite PK!), plugin_name, node_type |
| `edges` | DAG connections | from_node_id, to_node_id, label |
| `rows` | Source data ingestion | row_id, run_id, source_data_hash |
| `tokens` | Row instances in DAG | token_id, row_id, branch_name, fork_group_id |
| `node_states` | Processing records | state_id, token_id, node_id, status, duration_ms |
| `calls` | External API calls | call_id, state_id, call_type, latency_ms |
| `token_outcomes` | Terminal states | outcome, is_terminal, sink_name |
| `validation_errors` | Source validation failures | row_hash, row_data_json |
| `transform_errors` | Transform failures | token_id, error_details_json |

### Critical: Composite Primary Key on `nodes`

The `nodes` table uses `(node_id, run_id)` as a composite primary key. The same `node_id` can exist in multiple runs. Always filter by `run_id` when querying node-related data:

```sql
-- WRONG (ambiguous if node_id is reused)
SELECT * FROM node_states
JOIN nodes ON node_states.node_id = nodes.node_id

-- CORRECT (use run_id from node_states)
SELECT * FROM node_states
WHERE node_states.run_id = 'your-run-id'

-- CORRECT (if you must join to nodes)
SELECT * FROM node_states
JOIN nodes ON node_states.node_id = nodes.node_id
         AND node_states.run_id = nodes.run_id
WHERE node_states.run_id = 'your-run-id'
```

## Tips for Claude

1. **Start with `diagnose()`** when investigating problems - it tells you what's wrong and suggests next steps

2. **Use `explain_token()` for lineage** - it's the single most complete view of how a row was processed

3. **The `mermaid` output from `get_dag_structure()`** can be rendered in most markdown viewers to visualize the pipeline

4. **`get_performance_report()` identifies bottlenecks** automatically - look at the `bottlenecks` field first

5. **For ad-hoc SQL**, use `describe_schema()` first to understand the tables, then `query()` with SELECT statements

6. **Remember the composite PK** - always include `run_id` when querying nodes or joining to the nodes table
