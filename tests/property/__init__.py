# tests/property/__init__.py
"""Property-based tests for ELSPETH.

Property-based testing validates invariants that must hold for ALL inputs,
not just the specific examples we think of. This is critical for an
auditable system where determinism and data integrity are non-negotiable.

Test categories:
- canonical/: Hash determinism, NaN rejection, canonical JSON properties
- contracts/: Enum coercion, type contracts, data model invariants
"""
