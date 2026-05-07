"""Analyzer submodules for the ELSPETH Landscape MCP server.

Each submodule exports standalone functions that accept
(db: LandscapeDB, factory: RecorderFactory) parameters.
The LandscapeAnalyzer facade delegates to these functions.

Submodules:
    queries      -- Core CRUD operations (list_runs, list_rows, etc.)
    reports      -- Computed analysis (get_run_summary, get_performance_report, etc.)
    diagnostics  -- Emergency tools (diagnose, get_failure_context, etc.)
    contracts    -- Schema contract tools (get_run_contract, explain_field, etc.)
"""
