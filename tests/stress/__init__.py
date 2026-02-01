# tests/stress/__init__.py
"""Stress tests for ELSPETH components.

These tests run large workloads (10K-100K requests) to verify
behavior under load. They are NOT run in CI by default due to
execution time - use `pytest tests/stress -m stress` to run.
"""
