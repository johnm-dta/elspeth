"""Composer tool classification — name predicates and the session-aware carve-out.

This module is the public-API surface composer plumbing uses to ask "is
this tool a discovery tool / a mutation / cacheable / session-aware?".
The per-kind name frozensets are **derived from ``_REGISTERED_TOOLS``** in
``_registry.py`` (single source of truth: every plane's
``TOOLS_IN_MODULE`` tuple). Consumers that need a name frozenset directly
should reach for ``_registry``; this module is the predicate surface.

Layering:

- ``_registry.py`` is a leaf relative to ``discovery.py`` — it imports
  every plane module to aggregate declarations.
- ``discovery.py`` imports from ``_registry`` and exposes the predicates.
- ``_dispatch.py`` imports both ``_registry`` (for handler maps and the
  derived name frozensets) and ``discovery`` (for ``_SESSION_AWARE_TOOL_NAMES``).

Session-aware carve-out
-----------------------

``request_interpretation_review`` is intentionally NOT a ``ToolDeclaration``
(see ``declarations.py`` and ticket ``elspeth-f5da936747``). Its handler is
async and dispatched outside ``execute_tool`` via the compose loop. The
``_SESSION_AWARE_TOOL_NAMES`` frozenset below is therefore hand-maintained
— a one-element set today. The parity assertion in ``_dispatch.py``
asserts the ``_SESSION_AWARE_TOOL_HANDLERS`` dict in ``sessions.py``
agrees with this set.
"""

from __future__ import annotations

from typing import Final

from elspeth.web.composer.tools._registry import (
    _BLOB_DISCOVERY_TOOL_NAMES,
    _BLOB_MUTATION_TOOL_NAMES,
    _BLOB_STORE_ONLY_MUTATION_TOOL_NAMES,
    _CACHEABLE_DISCOVERY_TOOL_NAMES,
    _DISCOVERY_TOOL_NAMES,
    _MUTATION_TOOL_NAMES,
    _SECRET_DISCOVERY_TOOL_NAMES,
    _SECRET_MUTATION_TOOL_NAMES,
)

__all__ = [
    "_SESSION_AWARE_TOOL_NAMES",
    "is_blob_store_only_mutation_tool",
    "is_cacheable_discovery_tool",
    "is_discovery_tool",
    "is_mutation_tool",
    "is_session_aware_tool",
]

# ---------------------------------------------------------------------------
# Session-aware carve-out — the only hand-maintained name set on this module.
#
# Tools listed here are dispatched through the async session-aware path
# (``sessions._SESSION_AWARE_TOOL_HANDLERS``), not through ``execute_tool``.
# They do NOT have a ``ToolDeclaration`` because the declaration's handler
# typing is synchronous; widening it to admit async handlers is tracked
# under ``elspeth-f5da936747``.
# ---------------------------------------------------------------------------

_SESSION_AWARE_TOOL_NAMES: Final[frozenset[str]] = frozenset(
    {
        "request_interpretation_review",
    }
)


# ---------------------------------------------------------------------------
# Membership predicates
#
# Predicates are the public-API surface composer plumbing uses to ask "is
# this tool a discovery tool / a mutation / cacheable / session-aware?".
# They live here so plane modules and ``_dispatch.py`` agree on the answer.
# ---------------------------------------------------------------------------


def is_discovery_tool(name: str) -> bool:
    """Return True if the tool is a discovery (read-only) tool."""
    return name in _DISCOVERY_TOOL_NAMES or name in _BLOB_DISCOVERY_TOOL_NAMES or name in _SECRET_DISCOVERY_TOOL_NAMES


def is_mutation_tool(name: str) -> bool:
    """Return True when a composer tool can mutate session state or owned artifacts."""
    return name in _MUTATION_TOOL_NAMES or name in _BLOB_MUTATION_TOOL_NAMES or name in _SECRET_MUTATION_TOOL_NAMES


def is_cacheable_discovery_tool(name: str) -> bool:
    """Return True if the tool's results can be cached within a compose() call."""
    return name in _CACHEABLE_DISCOVERY_TOOL_NAMES


def is_session_aware_tool(name: str) -> bool:
    """Return True if the tool is dispatched through the async session-aware path."""
    return name in _SESSION_AWARE_TOOL_NAMES


def is_blob_store_only_mutation_tool(name: str) -> bool:
    """Return True for blob-store side-effect tools that never advance CompositionState.

    See ``_BLOB_STORE_ONLY_MUTATION_TOOL_NAMES`` for the rationale on excluding
    these from the ``trust_mode == "explicit_approve"`` proposal-interception
    gate.
    """
    return name in _BLOB_STORE_ONLY_MUTATION_TOOL_NAMES
