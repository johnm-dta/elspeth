"""Composer tool registry — single aggregation site for plane TOOLS_IN_MODULE tuples.

Ticket ``elspeth-6c9972ccbf`` (Composer tools — ToolDeclaration paradigm),
Step 4.

This module is the **single aggregation site** for every plane's
``TOOLS_IN_MODULE`` tuple. It holds the registered-tool universe and the
derived projections every consumer reads:

- ``_REGISTERED_TOOLS`` — the immutable union of every plane's declarations,
  in stable order (plane order then in-plane order).
- ``_TOOL_DEFS_BY_NAME`` — name → ``get_tool_definitions()``-shaped dict.
- ``_DISCOVERY_TOOLS`` / ``_MUTATION_TOOLS`` / ``_BLOB_DISCOVERY_TOOLS`` /
  ``_BLOB_MUTATION_TOOLS`` / ``_SECRET_DISCOVERY_TOOLS`` /
  ``_SECRET_MUTATION_TOOLS`` — derived name → handler maps consumed by
  ``_dispatch.execute_tool``.
- ``_DISCOVERY_TOOL_NAMES`` / ``_MUTATION_TOOL_NAMES`` / ...
  ``_CACHEABLE_DISCOVERY_TOOL_NAMES`` / ``_BLOB_STORE_ONLY_MUTATION_TOOL_NAMES``
  — derived frozensets consumed by ``discovery.py`` predicates and by
  ``service.py`` via those predicates.

Topology — why this lives in its own module
-------------------------------------------

Before Step 4 the aggregation lived in ``_dispatch.py``. That made
``discovery.py`` (the classification module) unable to derive its name
frozensets from declarations without re-aggregating plane tuples, which
duplicates the source of truth. Extracting the aggregation to this leaf
module breaks that — both ``_dispatch.py`` and ``discovery.py`` now read
from here without cycle:

    plane modules → declarations.py (pure leaf)
    plane modules ← _registry.py (aggregates planes)
    _registry.py ← discovery.py (derives name frozensets)
    _registry.py ← _dispatch.py (derives handler maps)
    discovery.py ← _dispatch.py

``request_interpretation_review`` (the session-aware async carve-out) is
intentionally NOT in ``_REGISTERED_TOOLS`` — see ``declarations.py`` and
ticket ``elspeth-f5da936747`` for the design rationale. Its
session-aware classification lives in ``discovery._SESSION_AWARE_TOOL_NAMES``
as a hand-maintained one-element frozenset; widening
``ToolDeclaration.handler`` to admit async handlers is deferred to a later
ticket.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Final

from elspeth.web.composer.tools._common import ToolHandler
from elspeth.web.composer.tools.blobs import (
    TOOLS_IN_MODULE as _BLOBS_TOOLS_IN_MODULE,
)
from elspeth.web.composer.tools.declarations import (
    ToolDeclaration,
    ToolKind,
    assert_unique_names,
    derive_augments_on_failure_names,
    derive_blob_store_only_names,
    derive_cacheable_names,
    derive_handler_map_for,
    derive_name_set_for,
    derive_tool_definitions_by_name,
)
from elspeth.web.composer.tools.generation import (
    TOOLS_IN_MODULE as _GENERATION_TOOLS_IN_MODULE,
)
from elspeth.web.composer.tools.outputs import (
    TOOLS_IN_MODULE as _OUTPUTS_TOOLS_IN_MODULE,
)
from elspeth.web.composer.tools.recipes import (
    TOOLS_IN_MODULE as _RECIPES_TOOLS_IN_MODULE,
)
from elspeth.web.composer.tools.secrets import (
    TOOLS_IN_MODULE as _SECRETS_TOOLS_IN_MODULE,
)
from elspeth.web.composer.tools.sessions import (
    TOOLS_IN_MODULE as _SESSIONS_TOOLS_IN_MODULE,
)
from elspeth.web.composer.tools.sources import (
    TOOLS_IN_MODULE as _SOURCES_TOOLS_IN_MODULE,
)
from elspeth.web.composer.tools.transforms import (
    TOOLS_IN_MODULE as _TRANSFORMS_TOOLS_IN_MODULE,
)

# ---------------------------------------------------------------------------
# Aggregated declaration tuple
# ---------------------------------------------------------------------------

_REGISTERED_TOOLS: Final[tuple[ToolDeclaration, ...]] = (
    *_BLOBS_TOOLS_IN_MODULE,
    *_SOURCES_TOOLS_IN_MODULE,
    *_SESSIONS_TOOLS_IN_MODULE,
    *_GENERATION_TOOLS_IN_MODULE,
    *_RECIPES_TOOLS_IN_MODULE,
    *_TRANSFORMS_TOOLS_IN_MODULE,
    *_OUTPUTS_TOOLS_IN_MODULE,
    *_SECRETS_TOOLS_IN_MODULE,
)
assert_unique_names(_REGISTERED_TOOLS)


# ---------------------------------------------------------------------------
# Derived tool-definitions map (consumed by ``_dispatch.get_tool_definitions``)
# ---------------------------------------------------------------------------

_TOOL_DEFS_BY_NAME: Final[Mapping[str, Mapping[str, Any]]] = derive_tool_definitions_by_name(_REGISTERED_TOOLS)


# ---------------------------------------------------------------------------
# Derived per-kind handler maps (consumed by ``_dispatch.execute_tool``)
# ---------------------------------------------------------------------------

_DISCOVERY_TOOLS: Final[Mapping[str, ToolHandler]] = derive_handler_map_for(_REGISTERED_TOOLS, ToolKind.DISCOVERY)
_MUTATION_TOOLS: Final[Mapping[str, ToolHandler]] = derive_handler_map_for(_REGISTERED_TOOLS, ToolKind.MUTATION)
_BLOB_DISCOVERY_TOOLS: Final[Mapping[str, ToolHandler]] = derive_handler_map_for(_REGISTERED_TOOLS, ToolKind.BLOB_DISCOVERY)
_BLOB_MUTATION_TOOLS: Final[Mapping[str, ToolHandler]] = derive_handler_map_for(_REGISTERED_TOOLS, ToolKind.BLOB_MUTATION)
_SECRET_DISCOVERY_TOOLS: Final[Mapping[str, ToolHandler]] = derive_handler_map_for(_REGISTERED_TOOLS, ToolKind.SECRET_DISCOVERY)
_SECRET_MUTATION_TOOLS: Final[Mapping[str, ToolHandler]] = derive_handler_map_for(_REGISTERED_TOOLS, ToolKind.SECRET_MUTATION)


# ---------------------------------------------------------------------------
# Derived per-kind name frozensets (consumed by ``discovery.py`` predicates
# and re-exported under their historical names for external consumers).
# ---------------------------------------------------------------------------

_DISCOVERY_TOOL_NAMES: Final[frozenset[str]] = derive_name_set_for(_REGISTERED_TOOLS, ToolKind.DISCOVERY)
_MUTATION_TOOL_NAMES: Final[frozenset[str]] = derive_name_set_for(_REGISTERED_TOOLS, ToolKind.MUTATION)
_BLOB_DISCOVERY_TOOL_NAMES: Final[frozenset[str]] = derive_name_set_for(_REGISTERED_TOOLS, ToolKind.BLOB_DISCOVERY)
_BLOB_MUTATION_TOOL_NAMES: Final[frozenset[str]] = derive_name_set_for(_REGISTERED_TOOLS, ToolKind.BLOB_MUTATION)
_SECRET_DISCOVERY_TOOL_NAMES: Final[frozenset[str]] = derive_name_set_for(_REGISTERED_TOOLS, ToolKind.SECRET_DISCOVERY)
_SECRET_MUTATION_TOOL_NAMES: Final[frozenset[str]] = derive_name_set_for(_REGISTERED_TOOLS, ToolKind.SECRET_MUTATION)

# Cacheable discovery tools — explicit opt-in via ``ToolDeclaration.cacheable``.
# The ``__post_init__`` invariant forbids any mutation kind from setting
# ``cacheable=True``; today only DISCOVERY-kind declarations may opt in, and
# the audit-integrity rule (every new discovery tool is non-cacheable by
# default, opt-in by name) is preserved because the per-declaration field
# is required at declaration time, not inherited from kind.
_CACHEABLE_DISCOVERY_TOOL_NAMES: Final[frozenset[str]] = derive_cacheable_names(_REGISTERED_TOOLS)

# Session-mutable discovery tools — the complement of cacheable within
# DISCOVERY. Tracked explicitly (as a named, derived constant) so the
# audit-integrity disjointness assertion below can detect any future drift.
_SESSION_MUTABLE_DISCOVERY_TOOL_NAMES: Final[frozenset[str]] = _DISCOVERY_TOOL_NAMES - _CACHEABLE_DISCOVERY_TOOL_NAMES

# Blob-store side-effect tools that never advance ``CompositionState``.
# Excluded from the ``trust_mode == "explicit_approve"`` proposal-interception
# gate because they have no state delta to approve — the audit trail still
# records the blob write but the composition itself is unchanged.
_BLOB_STORE_ONLY_MUTATION_TOOL_NAMES: Final[frozenset[str]] = derive_blob_store_only_names(_REGISTERED_TOOLS)

# Mutation tools whose failure results carry inline plugin schemas. Derived
# from ``ToolDeclaration.augments_on_failure`` (closing the SSOT loop on what
# used to be a shadow ``Final[frozenset[str]]`` in ``_common.py``). The
# declaration-side ``__post_init__`` invariant restricts membership to
# MUTATION / BLOB_MUTATION kinds. Core-reviewer I5 review finding, 2026-05-24.
_AUGMENTS_ON_FAILURE_TOOL_NAMES: Final[frozenset[str]] = derive_augments_on_failure_names(_REGISTERED_TOOLS)


def should_augment_with_plugin_schemas(tool_name: str) -> bool:
    """Return True when failures from ``tool_name`` should carry inline plugin schemas.

    Backs ``_dispatch.py``'s decision to call
    ``build_plugin_schemas_for_failure``. Replaces the prior ``_common.py``
    function that gated on the now-deleted ``_PLUGIN_SCHEMA_AUGMENTATION_TOOLS``
    shadow frozenset; the set lives here as
    ``_AUGMENTS_ON_FAILURE_TOOL_NAMES``, derived from
    ``ToolDeclaration.augments_on_failure``.
    """
    return tool_name in _AUGMENTS_ON_FAILURE_TOOL_NAMES


# ---------------------------------------------------------------------------
# Audit-integrity invariants on the derived sets
#
# These assertions are content-equivalent to the ones that previously lived
# in ``discovery.py``. They are retained at the module that owns the source
# of truth (the declarations) so a future drift fails the build at import
# time of the registry, not at the consumer.
# ---------------------------------------------------------------------------

if not _CACHEABLE_DISCOVERY_TOOL_NAMES <= _DISCOVERY_TOOL_NAMES:
    raise RuntimeError(
        "_CACHEABLE_DISCOVERY_TOOL_NAMES contains names that are not declared "
        "discovery tools: "
        f"{_CACHEABLE_DISCOVERY_TOOL_NAMES - _DISCOVERY_TOOL_NAMES}"
    )
if _CACHEABLE_DISCOVERY_TOOL_NAMES & _SESSION_MUTABLE_DISCOVERY_TOOL_NAMES:
    raise RuntimeError(
        "Session-mutable discovery tools must NEVER be cacheable — caching "
        "them would serve stale audit-trail data. Intersection: "
        f"{_CACHEABLE_DISCOVERY_TOOL_NAMES & _SESSION_MUTABLE_DISCOVERY_TOOL_NAMES}"
    )
if not _BLOB_STORE_ONLY_MUTATION_TOOL_NAMES <= _BLOB_MUTATION_TOOL_NAMES:
    raise RuntimeError(
        "_BLOB_STORE_ONLY_MUTATION_TOOL_NAMES contains names that are not "
        "declared BLOB_MUTATION tools: "
        f"{_BLOB_STORE_ONLY_MUTATION_TOOL_NAMES - _BLOB_MUTATION_TOOL_NAMES}"
    )
