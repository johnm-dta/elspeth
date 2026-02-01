"""Tests for contracts package.

This package contains:
1. Unit tests for contract types (test_results.py, test_audit.py, etc.)
2. Contract verification tests for plugin interfaces (source_contracts/, etc.)

Contract tests verify that plugin implementations honor their protocol contracts.
These tests focus on interface guarantees, not implementation details.

Key contracts verified:
- Source: load() yields SourceRow objects, close() is idempotent
- Transform: process() returns TransformResult, success has data, error has reason
- Sink: write() returns ArtifactDescriptor with content_hash (audit integrity!)
"""
