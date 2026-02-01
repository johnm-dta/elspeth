# tests/property/canonical/__init__.py
"""Property tests for canonical JSON and hashing.

The audit trail depends on stable, deterministic hashes. These tests
verify that canonical_json() and stable_hash() behave correctly for
ALL possible inputs, not just the examples we write manually.
"""
