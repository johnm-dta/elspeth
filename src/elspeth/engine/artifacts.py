# src/elspeth/engine/artifacts.py
"""Unified artifact descriptors for all sink types.

IMPORTANT: ArtifactDescriptor is now defined in elspeth.contracts.results.
This module re-exports it for backwards compatibility.
"""

from elspeth.contracts import ArtifactDescriptor

__all__ = ["ArtifactDescriptor"]
