"""Composer tool classification — canonical tool-name sets and predicates.

This module is the single source of truth for the *names* of composer tools
and their classification (discovery vs mutation, sync vs session-aware,
cacheable, blob-store-only). The actual handler dictionaries live in
``_dispatch.py`` (sync registries) and ``sessions.py``
(``_SESSION_AWARE_TOOL_HANDLERS``); ``_dispatch.py`` asserts at import time
that the registries' keys match the name sets declared here, so a tool added
to a registry without being declared here (or vice versa) fails the build
before any compose() call could see it.

Layering note: ``discovery.py`` is a leaf module — it imports nothing from
other tool-plane files. The plane files and ``_dispatch.py`` import from
here. ``_BLOB_STORE_ONLY_MUTATION_TOOL_NAMES`` is centralised here even
though the kwarg-shape sets (``_BLOB_QUOTA_MUTATION_TOOLS``,
``_BLOB_PROVENANCE_MUTATION_TOOLS``) remain in ``blobs.py`` — those describe
*how* a blob handler is dispatched (which extended kwargs it receives), not
*what kind* of tool it is.
"""

from __future__ import annotations

from typing import Final

# ---------------------------------------------------------------------------
# Canonical tool-name frozensets
#
# Every name a composer tool can dispatch under appears in exactly one of
# the discovery / mutation / blob / secret sets, plus possibly in the
# session-aware set if it is async. ``_dispatch.py`` asserts that the
# six sync-registry key sets equal these declarations at import time;
# ``sessions.py`` asserts the same for ``_SESSION_AWARE_TOOL_NAMES``.
# ---------------------------------------------------------------------------

_DISCOVERY_TOOL_NAMES: Final[frozenset[str]] = frozenset(
    {
        "list_sources",
        "list_transforms",
        "list_sinks",
        "get_plugin_schema",
        "get_expression_grammar",
        "explain_validation_error",
        "get_plugin_assistance",
        "list_models",
        "get_audit_info",
        "list_recipes",
        "get_pipeline_state",
        "preview_pipeline",
        "diff_pipeline",
    }
)

_MUTATION_TOOL_NAMES: Final[frozenset[str]] = frozenset(
    {
        "set_source",
        "upsert_node",
        "upsert_edge",
        "remove_node",
        "remove_edge",
        "set_metadata",
        "set_output",
        "remove_output",
        "patch_source_options",
        "patch_node_options",
        "patch_output_options",
        "set_pipeline",
        "clear_source",
    }
)

_BLOB_DISCOVERY_TOOL_NAMES: Final[frozenset[str]] = frozenset(
    {
        "list_blobs",
        "get_blob_metadata",
        "get_blob_content",
        "inspect_source",
    }
)

_BLOB_MUTATION_TOOL_NAMES: Final[frozenset[str]] = frozenset(
    {
        "set_source_from_blob",
        "create_blob",
        "update_blob",
        "delete_blob",
        "apply_pipeline_recipe",
    }
)

_SECRET_DISCOVERY_TOOL_NAMES: Final[frozenset[str]] = frozenset(
    {
        "list_secret_refs",
        "validate_secret_ref",
    }
)

_SECRET_MUTATION_TOOL_NAMES: Final[frozenset[str]] = frozenset(
    {
        "wire_secret_ref",
    }
)

_SESSION_AWARE_TOOL_NAMES: Final[frozenset[str]] = frozenset(
    {
        "request_interpretation_review",
    }
)

# Cacheable discovery tools — **EXPLICIT OPT-IN** set.
#
# Audit-integrity rule: any new discovery tool MUST be added here on
# purpose. The previous opt-out construction (subtract the known-unsafe
# names from ``_DISCOVERY_TOOL_NAMES``) silently auto-cached every newly
# added discovery tool, which is the wrong default when the failure mode
# is stale data being served from cache and recorded as the live answer.
# Session-mutable-state tools — currently ``diff_pipeline``,
# ``get_pipeline_state``, and ``preview_pipeline`` — MUST remain absent
# from this set (the import-time assertion below enforces that).
#
# To add a new entry: confirm the tool's result is a pure function of
# the catalog and the *committed* CompositionState (no runtime preflight,
# no baseline-diff inputs, no live-state materialisation that the
# compose-loop cache could not safely reuse), then add the name here.
_CACHEABLE_DISCOVERY_TOOL_NAMES: Final[frozenset[str]] = frozenset(
    {
        "list_sources",
        "list_transforms",
        "list_sinks",
        "get_plugin_schema",
        "get_expression_grammar",
        "explain_validation_error",
        "get_plugin_assistance",
        "list_models",
        "get_audit_info",
        "list_recipes",
    }
)

# Names that must NEVER appear in ``_CACHEABLE_DISCOVERY_TOOL_NAMES``
# because their result depends on session-mutable state — caching would
# serve stale audit-trail data. Tracked explicitly so the import-time
# assertion below can detect accidental inclusion (e.g. a copy-paste edit
# moving one of these names into the opt-in set).
_SESSION_MUTABLE_DISCOVERY_TOOL_NAMES: Final[frozenset[str]] = frozenset(
    {
        "diff_pipeline",
        "get_pipeline_state",
        "preview_pipeline",
    }
)

# Import-time assertion: every cacheable name must be a registered
# discovery tool, and the cacheable set must not intersect the
# session-mutable forbidden set. ``_dispatch.py`` separately asserts that
# every cacheable name appears in the live ``_DISCOVERY_TOOLS`` handler
# registry; this check enforces the audit-integrity contract at the name
# layer so a stale ``discovery.py`` edit cannot pass through review.
assert _CACHEABLE_DISCOVERY_TOOL_NAMES <= _DISCOVERY_TOOL_NAMES, (
    "_CACHEABLE_DISCOVERY_TOOL_NAMES contains names that are not declared "
    "discovery tools: "
    f"{_CACHEABLE_DISCOVERY_TOOL_NAMES - _DISCOVERY_TOOL_NAMES}"
)
assert _SESSION_MUTABLE_DISCOVERY_TOOL_NAMES <= _DISCOVERY_TOOL_NAMES, (
    "_SESSION_MUTABLE_DISCOVERY_TOOL_NAMES contains names that are not "
    "declared discovery tools: "
    f"{_SESSION_MUTABLE_DISCOVERY_TOOL_NAMES - _DISCOVERY_TOOL_NAMES}"
)
assert not (_CACHEABLE_DISCOVERY_TOOL_NAMES & _SESSION_MUTABLE_DISCOVERY_TOOL_NAMES), (
    "Session-mutable discovery tools must NEVER be cacheable — caching "
    "them would serve stale audit-trail data. Intersection: "
    f"{_CACHEABLE_DISCOVERY_TOOL_NAMES & _SESSION_MUTABLE_DISCOVERY_TOOL_NAMES}"
)

# Blob-store side-effect tools that never advance ``CompositionState``.
# Excluded from the ``trust_mode == "explicit_approve"`` proposal-interception
# gate because they have no state delta to approve — the audit trail still
# records the blob write but the composition itself is unchanged.
_BLOB_STORE_ONLY_MUTATION_TOOL_NAMES: Final[frozenset[str]] = frozenset({"create_blob", "update_blob", "delete_blob"})


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
