# tests/property/contracts/__init__.py
"""Property tests for contract invariants.

These tests verify that enum coercion, data model invariants, and
type contracts hold for all possible values. Per ELSPETH's Three-Tier
Trust Model, audit data must be pristine - invalid enum strings must
crash, not silently coerce to defaults.
"""
