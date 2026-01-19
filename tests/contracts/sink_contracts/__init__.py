# tests/contracts/sink_contracts/__init__.py
"""Contract tests for Sink plugins.

Sinks output data to external destinations:
- Receive pipeline data (elevated trust)
- No coercion allowed (wrong types = upstream bug)
- MUST return ArtifactDescriptor with content_hash for audit

Contract guarantees:
1. write() MUST return ArtifactDescriptor
2. ArtifactDescriptor MUST have content_hash (SHA-256, 64 hex chars)
3. ArtifactDescriptor MUST have size_bytes
4. flush() MUST be idempotent
5. close() MUST be idempotent
6. Same data MUST produce same content_hash (determinism)
"""
