# src/elspeth/core/retention/__init__.py
"""Retention management for PayloadStore content.

Provides PurgeManager for identifying and deleting expired payloads
while preserving audit trail integrity (hashes remain in Landscape).
"""

from elspeth.core.retention.purge import PurgeManager, PurgeResult

__all__ = ["PurgeManager", "PurgeResult"]
