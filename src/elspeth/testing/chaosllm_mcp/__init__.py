# src/elspeth/testing/chaosllm_mcp/__init__.py
"""MCP server for analyzing ChaosLLM test results.

Provides Claude-optimized analysis tools:
- diagnose(): One-paragraph summary with actionable insights
- analyze_aimd_behavior(): Recovery times, backoff effectiveness
- analyze_errors(): Error breakdown by category
- analyze_latency(): p50/p95/p99 with correlations
- find_anomalies(): Auto-detected unusual patterns

Usage:
    elspeth chaosllm-mcp --database=./chaosllm-metrics.db
"""

from elspeth.testing.chaosllm_mcp.server import (
    ChaosLLMAnalyzer,
    create_server,
    main,
    run_server,
)

__all__ = [
    "ChaosLLMAnalyzer",
    "create_server",
    "main",
    "run_server",
]
