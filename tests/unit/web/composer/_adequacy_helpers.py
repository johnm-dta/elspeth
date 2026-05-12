"""Adequacy guard helpers — registry name collection (spec §4.4.1)."""

from __future__ import annotations

from elspeth.web.composer.tools import (
    _BLOB_DISCOVERY_TOOLS,
    _BLOB_MUTATION_TOOLS,
    _DISCOVERY_TOOLS,
    _MUTATION_TOOLS,
    _SECRET_DISCOVERY_TOOLS,
    _SECRET_MUTATION_TOOLS,
)


def collect_registry_names() -> frozenset[str]:
    """Union of every tool name dispatchable through the composer.

    Spec §4.4.1: the manifest's set-equality contract is keyed on this
    set. Adding a tool to a dispatch dict without a manifest entry —
    or removing a manifest entry without removing the dispatch dict
    entry — must fail this guard at CI.

    The six dispatch dicts (tools.py lines 5274-5340) are unioned here.
    preview_pipeline, diff_pipeline, and set_pipeline already appear in
    _DISCOVERY_TOOLS / _MUTATION_TOOLS; their extended-signature
    special-casing in execute_tool() is an implementation detail, not a
    separate registration.

    request_advisor_hint is intercepted at service.py:2070 before
    execute_tool() is called, so it never appears in the six dispatch
    dicts. It requires a MANIFEST entry and is enumerated here explicitly.
    """
    return frozenset(
        set(_DISCOVERY_TOOLS)
        | set(_MUTATION_TOOLS)
        | set(_BLOB_DISCOVERY_TOOLS)
        | set(_BLOB_MUTATION_TOOLS)
        | set(_SECRET_DISCOVERY_TOOLS)
        | set(_SECRET_MUTATION_TOOLS)
        | {"request_advisor_hint"}  # advisor escape hatch — intercepted at service.py
    )
