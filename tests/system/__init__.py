"""System tests for ELSPETH.

System tests verify full pipeline scenarios end-to-end, including:
- Audit trail completeness and explain query functionality
- Crash recovery and resume scenarios
- Multi-stage pipeline execution with real plugins

These tests are slower than unit/integration tests (<30s each)
but verify the system works as a whole.
"""
