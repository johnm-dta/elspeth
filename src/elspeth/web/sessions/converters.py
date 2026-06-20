"""Converters between session-layer records and composer-layer domain objects.

This module bridges the gap between CompositionStateRecord (the database
representation with raw dict fields) and CompositionState (the typed domain
object with SourceSpec, NodeSpec, etc.).

Both sessions/routes.py and execution/service.py need this conversion.
It lives here (not in routes.py) to avoid forcing execution/ to import
from a route module.

GuidedSession persistence:
  ``guided_session`` is NOT a first-class column in ``composition_states``; it
  rides in ``composer_meta["guided_session"]`` as a serialised dict.  When
  ``composer_meta`` is present and contains a ``"guided_session"`` key,
  ``state_from_record`` reconstructs the full ``GuidedSession`` object and
  attaches it to the returned ``CompositionState``.  The absence of the key is
  honest (freeform session — ``guided_session`` stays ``None``).
"""

from __future__ import annotations

from typing import Any

from elspeth.contracts.freeze import deep_thaw
from elspeth.web.composer.guided.state_machine import GuidedSession
from elspeth.web.composer.state import CompositionState
from elspeth.web.composer.yaml_generator import generate_pipeline_dict
from elspeth.web.sessions.protocol import CompositionStateRecord


def state_from_record(record: CompositionStateRecord) -> CompositionState:
    """Reconstruct a CompositionState from a CompositionStateRecord.

    Thaws frozen container fields (MappingProxyType, tuple) back to plain
    dicts/lists so CompositionState.from_dict() can re-freeze them.

    Tier 1: metadata_ must always be populated. A None here indicates
    database corruption or a migration gap — crash immediately.

    Guided-mode: if ``composer_meta["guided_session"]`` is present, the
    GuidedSession is restored via ``GuidedSession.from_dict()``.
    """
    if record.metadata_ is None:
        msg = f"CompositionStateRecord {record.id} has None metadata_ — database corruption or migration gap"
        raise ValueError(msg)

    sources = deep_thaw(record.sources) if record.sources is not None else None
    if sources is None and record.source is not None:
        sources = {"source": deep_thaw(record.source)}

    state_dict = {
        "version": record.version,
        "sources": sources,
        # nodes/edges/outputs: None is the legitimate initial state when no
        # nodes have been added yet. Mapping None -> [] is meaning-preserving
        # (empty collection, not fabricated data).
        "nodes": [deep_thaw(n) for n in record.nodes] if record.nodes is not None else [],
        "edges": [deep_thaw(e) for e in record.edges] if record.edges is not None else [],
        "outputs": [deep_thaw(o) for o in record.outputs] if record.outputs is not None else [],
        "metadata": deep_thaw(record.metadata_),
    }
    state = CompositionState.from_dict(state_dict)

    # Restore guided_session from composer_meta side-channel.
    # Tier 1: any malformed guided_session dict is a corruption event — crash.
    if record.composer_meta is not None:
        thawed_meta = deep_thaw(record.composer_meta)
        if "guided_session" in thawed_meta:
            guided_session_raw = thawed_meta["guided_session"]
            if guided_session_raw is None:
                # Explicitly-null key: freeform session, no session to restore.
                return state
            guided_session = GuidedSession.from_dict(guided_session_raw)
            from dataclasses import replace

            state = replace(state, guided_session=guided_session)

    return state


def pipeline_dict_from_record(record: CompositionStateRecord) -> dict[str, Any]:
    """Return the canonical runtime/YAML-shape dict for a DB composition row.

    ``CompositionStateRecord`` is raw session storage shape: flat ``nodes`` and
    ``outputs`` collections. Walkers and analyzers should use this adapter
    instead of hand-partitioning those collections or round-tripping through
    YAML text.
    """
    return generate_pipeline_dict(state_from_record(record))
