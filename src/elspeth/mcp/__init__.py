"""MCP (Model Context Protocol) server for ELSPETH Landscape analysis.

Provides read-only tools for querying the audit database:
- list_runs: List all pipeline runs
- get_run: Get details of a specific run
- get_run_summary: Statistics for a run (now includes operation counts)
- list_rows: List rows for a run
- list_operations: List source/sink operations for a run
- explain_token: Complete lineage for a token
- get_operation_calls: Get calls made during a source/sink operation
- get_errors: Get validation and transform errors
- diagnose: Emergency diagnostic (now detects stuck operations)
- query: Execute read-only SQL queries
"""


def create_server(*args, **kwargs):  # type: ignore[no-untyped-def]
    """Create an MCP server instance. Requires the [mcp] extra."""
    try:
        from elspeth.mcp.server import create_server as _create_server
    except ModuleNotFoundError as exc:
        if "mcp" in str(exc):
            raise ImportError("MCP server requires the [mcp] extra. Install with: uv pip install -e '.[mcp]'") from exc
        raise

    return _create_server(*args, **kwargs)


def main(*args, **kwargs):  # type: ignore[no-untyped-def]
    """Run the MCP server. Requires the [mcp] extra."""
    try:
        from elspeth.mcp.server import main as _main
    except ModuleNotFoundError as exc:
        if "mcp" in str(exc):
            raise ImportError("MCP server requires the [mcp] extra. Install with: uv pip install -e '.[mcp]'") from exc
        raise

    return _main(*args, **kwargs)


__all__ = ["create_server", "main"]
