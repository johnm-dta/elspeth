"""Composer tool declarations — single-declaration-per-tool primitive.

This module defines the ``ToolDeclaration`` frozen dataclass that consolidates
the per-tool registry data previously scattered across the ``_dispatch.py``
handler dicts, the inline JSON schema list in ``get_tool_definitions()``, the
blob kwarg-shape frozensets in ``blobs.py``, and the ``__all__`` re-export list
in ``__init__.py``.

The intervention closes the registry-fragmentation growth mechanism diagnosed
in ``notes/composer-tools-growth-mechanism-2026-05-23.md`` (Shifting the Burden
+ Fixes That Fail). It builds on the precursor PR which centralised tool-name
classification in ``discovery.py`` and collapsed handler-signature divergence
via ``ToolContext``.

What this primitive deliberately does NOT consolidate
-----------------------------------------------------

**Redaction MANIFEST stays in ``redaction.py``.** ``ToolDeclaration`` has no
redaction field. The argument-model classes (``CreateBlobArgumentsModel`` and
peers) carry ``Sensitive[T]`` annotations the redaction walker reads, and
``HandlesNoSensitiveDataReason`` carries prose security justifications subject
to human review at manifest-load time. Keeping them in one auditor-readable
file preserves the security-review property called out in ticket
``elspeth-6c9972ccbf`` Refinement 1; carrying redaction here would replicate
the Shifting-the-Burden archetype the epic is meant to defeat (two sources of
truth for the same fact, locally cheap, structurally wrong).

**Pydantic argument-models stay in ``redaction.py``** for the same reason —
they are the security policy. ``ToolDeclaration.json_schema`` is the LLM-facing
schema (richer prose, more description text). The two surfaces are structurally
distinct artifacts, not duplicates; ``model_json_schema()`` would not reproduce
the LLM-facing prose.

Architectural override versus the precursor PR
----------------------------------------------

The precursor writer kept ``_BLOB_QUOTA_MUTATION_TOOLS`` and
``_BLOB_PROVENANCE_MUTATION_TOOLS`` in ``blobs.py`` with the rationale that
they describe "the kwarg-shape dispatch, not tool classification". Under the
declaration model that distinction collapses: a tool's blob-kwarg shape *is*
a property of the tool, so it belongs in the declaration
(``needs_blob_quota`` / ``needs_blob_provenance``). The locality property
the precursor cared about is preserved by a different mechanism — every
declaration lives in the plane module next to its handler, so blob-kwarg
data still co-locates with the blob handler that consumes it.

Import topology — no cyclic aggregation
---------------------------------------

This module is a **pure-data leaf**: it defines ``ToolKind`` and
``ToolDeclaration`` plus stateless derivation helpers that take an
``Iterable[ToolDeclaration]`` parameter. It imports nothing from plane modules.
Plane modules import this module to construct their declarations and expose
``TOOLS_IN_MODULE`` tuples; ``_dispatch.py`` (which already imports every
plane) is the single aggregation site that concatenates those tuples and feeds
them to the derivation helpers. This breaks the import cycle — Python's
partial-module-load window cannot leave a derivation function seeing an empty
registered set, because the registered set is built at the consumer site after
every plane has finished loading.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Final

from elspeth.contracts.freeze import deep_thaw, freeze_fields

if TYPE_CHECKING:
    from elspeth.web.composer.tools._common import ToolHandler


class ToolKind(Enum):
    """Tool-classification kinds.

    Each declaration carries exactly one kind. The kind determines (a) which
    dispatcher path the handler is routed through (sync vs async, plain vs
    blob vs secret kwarg shape) and (b) which ``discovery.py`` name-set the
    tool's name must appear in. Discovery vs mutation is the cacheability
    boundary; blob vs secret is the kwarg-shape boundary; session-aware is
    the sync/async boundary.
    """

    DISCOVERY = "discovery"
    MUTATION = "mutation"
    BLOB_DISCOVERY = "blob_discovery"
    BLOB_MUTATION = "blob_mutation"
    SECRET_DISCOVERY = "secret_discovery"
    SECRET_MUTATION = "secret_mutation"
    SESSION_AWARE = "session_aware"


_MUTATION_KINDS: Final[frozenset[ToolKind]] = frozenset(
    {
        ToolKind.MUTATION,
        ToolKind.BLOB_MUTATION,
        ToolKind.SECRET_MUTATION,
        ToolKind.SESSION_AWARE,
    }
)


@dataclass(frozen=True, slots=True)
class ToolDeclaration:
    """One composer tool, declared at one site, co-located with its handler.

    Every declared tool is registered by being placed in a plane module's
    ``TOOLS_IN_MODULE`` tuple. ``_dispatch.py`` aggregates every plane's tuple
    into a single registered set; pure-function derivation helpers below
    project that set to the surfaces the dispatcher consumes.

    Fields:
        name: The tool name as the LLM sees it. Must equal the dict key the
            tool is dispatched under and the name in the corresponding
            ``discovery.py`` name-set.
        handler: The ``ToolHandler`` callable
            ``(arguments, state, context) -> ToolResult``. ``SESSION_AWARE``
            tools carry an async coroutine handler instead; the dispatcher
            distinguishes by ``kind``.
        kind: One of the seven ``ToolKind`` categories. Determines the
            dispatch path and the name-set the tool's name must appear in.
        description: The LLM-facing description prose. This is the
            single source of truth for tool description text used by both
            ``get_tool_definitions()`` and (when Step 4 lands) the
            skill markdown's tool-inventory bullets.
        json_schema: The "parameters" object — a JSON Schema dict describing
            the tool's argument shape. This is the LLM-facing schema, not the
            Pydantic validation model; the two surfaces are deliberately
            distinct (see module docstring).
        cacheable: True if results can be cached within a compose() call.
            Forbidden for any mutation kind; for discovery tools this is the
            **explicit opt-in** flag that today drives
            ``_CACHEABLE_DISCOVERY_TOOL_NAMES``.
        needs_blob_quota: True if the dispatcher should pass
            ``max_blob_storage_per_session_bytes`` to the handler via
            ``ToolContext``. Only meaningful for ``BLOB_MUTATION``.
        needs_blob_provenance: True if the dispatcher should pass
            ``user_message_id`` to the handler via ``ToolContext``. Only
            meaningful for ``BLOB_MUTATION``.
        blob_store_only: True if the tool writes only to blob storage and
            never advances ``CompositionState`` (excluded from the
            ``trust_mode == "explicit_approve"`` proposal-interception gate).
            Only meaningful for ``BLOB_MUTATION``.
    """

    name: str
    handler: ToolHandler
    kind: ToolKind
    description: str
    json_schema: Mapping[str, Any]
    cacheable: bool = False
    needs_blob_quota: bool = False
    needs_blob_provenance: bool = False
    blob_store_only: bool = False

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("ToolDeclaration.name must be non-empty.")

        if not isinstance(self.kind, ToolKind):
            raise TypeError(f"ToolDeclaration({self.name!r}).kind must be a ToolKind member, got {type(self.kind).__name__}.")

        # Cacheability invariant — mutation/session-aware tools MUST NOT be
        # cacheable. Caching a mutation serves a stale write result; caching
        # a session-aware async response defeats the per-call semantics.
        if self.cacheable and self.kind in _MUTATION_KINDS:
            raise ValueError(
                f"ToolDeclaration({self.name!r}).cacheable=True is forbidden "
                f"for kind {self.kind.value!r}. Caching a mutation or "
                "session-aware tool serves stale state to the LLM."
            )

        # Blob-kwarg invariants — only BLOB_MUTATION may set these.
        if self.kind is not ToolKind.BLOB_MUTATION:
            if self.needs_blob_quota:
                raise ValueError(
                    f"ToolDeclaration({self.name!r}).needs_blob_quota=True is "
                    "only valid for BLOB_MUTATION; the dispatcher does not "
                    "pass max_blob_storage_per_session_bytes through other "
                    "dispatch paths."
                )
            if self.needs_blob_provenance:
                raise ValueError(
                    f"ToolDeclaration({self.name!r}).needs_blob_provenance=True "
                    "is only valid for BLOB_MUTATION; the dispatcher does not "
                    "pass user_message_id through other dispatch paths."
                )
            if self.blob_store_only:
                raise ValueError(f"ToolDeclaration({self.name!r}).blob_store_only=True is only valid for BLOB_MUTATION.")

        freeze_fields(self, "json_schema")


# ---------------------------------------------------------------------------
# Pure derivation helpers
#
# Each function below projects an Iterable[ToolDeclaration] onto one of the
# previously hand-maintained surfaces (handler map, name set, tool-definition
# list, blob-kwarg name sets, cacheable name set). ``_dispatch.py`` aggregates
# every plane's TOOLS_IN_MODULE tuple and calls these helpers; parity between
# the derived shape and the hand-maintained shape is asserted at import time
# in ``_dispatch.py`` so a mismatch fails the build.
#
# All helpers are pure functions of the input iterable — they hold no module
# state. This is deliberate: the absence of module-level mutable state
# eliminates the import-order trap a registry-with-side-effects would carry.
# ---------------------------------------------------------------------------


def derive_handler_map_for(tools: Iterable[ToolDeclaration], kind: ToolKind) -> Mapping[str, ToolHandler]:
    """Return the name → handler map for declarations of one kind."""
    return MappingProxyType({decl.name: decl.handler for decl in tools if decl.kind is kind})


def derive_name_set_for(tools: Iterable[ToolDeclaration], kind: ToolKind) -> frozenset[str]:
    """Return the set of tool names for one kind."""
    return frozenset(decl.name for decl in tools if decl.kind is kind)


def derive_tool_definitions_by_name(
    tools: Iterable[ToolDeclaration],
) -> Mapping[str, dict[str, Any]]:
    """Return name → ``get_tool_definitions()``-shaped dict for every declaration.

    Each value has the legacy ``{"name", "description", "parameters"}`` shape
    used by the LLM tool-list.  ``_dispatch.py:get_tool_definitions()`` reads
    this map and substitutes the declaration's entry in place of the
    corresponding hand-maintained inline schema during the migration.

    The ``parameters`` value is **deep-thawed** (``MappingProxyType`` → ``dict``,
    ``tuple`` → ``list``) so external consumers (LiteLLM, MCP clients) see the
    same JSON-shaped structure they did pre-migration. The declaration stores
    the schema frozen for immutability; emission unfreezes for compatibility.
    """
    return MappingProxyType(
        {
            decl.name: {
                "name": decl.name,
                "description": decl.description,
                "parameters": deep_thaw(decl.json_schema),
            }
            for decl in tools
        }
    )


def derive_blob_quota_names(tools: Iterable[ToolDeclaration]) -> frozenset[str]:
    """Return the names of BLOB_MUTATION tools that consume the quota kwarg."""
    return frozenset(decl.name for decl in tools if decl.kind is ToolKind.BLOB_MUTATION and decl.needs_blob_quota)


def derive_blob_provenance_names(tools: Iterable[ToolDeclaration]) -> frozenset[str]:
    """Return the names of BLOB_MUTATION tools that consume the provenance kwarg."""
    return frozenset(decl.name for decl in tools if decl.kind is ToolKind.BLOB_MUTATION and decl.needs_blob_provenance)


def derive_blob_store_only_names(tools: Iterable[ToolDeclaration]) -> frozenset[str]:
    """Return the names of BLOB_MUTATION tools that never advance CompositionState."""
    return frozenset(decl.name for decl in tools if decl.kind is ToolKind.BLOB_MUTATION and decl.blob_store_only)


def derive_cacheable_names(tools: Iterable[ToolDeclaration]) -> frozenset[str]:
    """Return the names of declarations marked cacheable.

    Today only ``DISCOVERY``-kind tools may set ``cacheable=True``; the
    ``ToolDeclaration`` constructor enforces this. ``discovery.py`` separately
    asserts the cacheable set is a subset of the discovery name-set and
    disjoint from the session-mutable forbidden set; this derivation feeds
    those assertions during the migration.
    """
    return frozenset(decl.name for decl in tools if decl.cacheable)


def assert_unique_names(tools: Iterable[ToolDeclaration]) -> None:
    """Raise if any two declarations share a name. ``_dispatch.py`` calls this
    once after aggregation, before any other derivation runs."""
    seen: set[str] = set()
    for decl in tools:
        if decl.name in seen:
            raise RuntimeError(
                f"ToolDeclaration({decl.name!r}) is registered more than once. "
                "Each tool name must appear in exactly one plane's "
                "TOOLS_IN_MODULE tuple."
            )
        seen.add(decl.name)
