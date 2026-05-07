"""Converters between session-layer records and composer-layer domain objects.

This module bridges the gap between CompositionStateRecord (the database
representation with raw dict fields) and CompositionState (the typed domain
object with SourceSpec, NodeSpec, etc.).

Both sessions/routes.py and execution/service.py need this conversion.
It lives here (not in routes.py) to avoid forcing execution/ to import
from a route module.
"""

from __future__ import annotations

from elspeth.contracts.freeze import deep_thaw
from elspeth.web.composer.state import CompositionState
from elspeth.web.sessions.protocol import CompositionStateRecord


def state_from_record(record: CompositionStateRecord) -> CompositionState:
    """Reconstruct a CompositionState from a CompositionStateRecord.

    Thaws frozen container fields (MappingProxyType, tuple) back to plain
    dicts/lists so CompositionState.from_dict() can re-freeze them.

    Tier 1: metadata_ must always be populated. A None here indicates
    database corruption or a migration gap — crash immediately.
    """
    if record.metadata_ is None:
        msg = f"CompositionStateRecord {record.id} has None metadata_ — database corruption or migration gap"
        raise ValueError(msg)

    state_dict = {
        "version": record.version,
        "source": deep_thaw(record.source) if record.source is not None else None,
        # nodes/edges/outputs: None is the legitimate initial state when no
        # nodes have been added yet. Mapping None -> [] is meaning-preserving
        # (empty collection, not fabricated data).
        "nodes": [deep_thaw(n) for n in record.nodes] if record.nodes is not None else [],
        "edges": [deep_thaw(e) for e in record.edges] if record.edges is not None else [],
        "outputs": [deep_thaw(o) for o in record.outputs] if record.outputs is not None else [],
        "metadata": deep_thaw(record.metadata_),
    }
    return CompositionState.from_dict(state_dict)
