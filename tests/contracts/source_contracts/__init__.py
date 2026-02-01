# tests/contracts/source_contracts/__init__.py
"""Contract tests for Source plugins.

Sources are the data ingestion boundary where:
- External data enters the system (zero trust)
- Coercion is allowed (normalizing external data)
- Invalid rows can be quarantined

Contract guarantees:
1. load() MUST yield SourceRow objects (not raw dicts)
2. Valid rows MUST have non-None data
3. Quarantined rows MUST have error and destination
4. close() MUST be idempotent (safe to call multiple times)
5. Lifecycle hooks on_start/on_complete MUST not raise
"""
