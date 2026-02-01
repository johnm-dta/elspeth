# tests/contracts/transform_contracts/__init__.py
"""Contract tests for Transform plugins.

Transforms process pipeline data (elevated trust):
- Types are expected to be correct (source validated them)
- No coercion allowed (wrong types = upstream bug)
- Operations on row values may fail (wrap those)

Contract guarantees:
1. process() MUST return TransformResult
2. Success results MUST have output data (row or rows)
3. Error results MUST have reason dict
4. close() MUST be idempotent
5. Lifecycle hooks on_start/on_complete MUST not raise
"""
