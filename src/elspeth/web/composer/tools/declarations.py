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
from typing import TYPE_CHECKING, Any

from jsonschema import Draft202012Validator
from jsonschema.exceptions import SchemaError

from elspeth.contracts.freeze import freeze_fields

if TYPE_CHECKING:
    from elspeth.web.composer.tools._common import ToolHandler


class ToolKind(Enum):
    """Tool-classification kinds.

    Each declaration carries exactly one kind. The kind determines (a) which
    dispatcher path the handler is routed through (plain vs blob vs secret
    kwarg shape) and (b) which ``_registry`` name-set the tool's name must
    appear in. Discovery vs mutation is the cacheability boundary; blob vs
    secret is the kwarg-shape boundary.

    Session-aware async tools do NOT have a ``ToolKind`` member today —
    they are dispatched outside ``execute_tool`` via the async session
    path and do not carry a ``ToolDeclaration`` because the declaration's
    ``handler`` is typed synchronously. Widening to admit async is tracked
    under ``elspeth-f5da936747``; when that lands, a ``SESSION_AWARE`` kind
    will be re-added concurrently with the first declaration that uses it.
    Advertising an enum value with no callers today would be a dead
    forward-pretend that this commit removes (solution-architect M1
    review finding, 2026-05-23).
    """

    DISCOVERY = "discovery"
    MUTATION = "mutation"
    BLOB_DISCOVERY = "blob_discovery"
    BLOB_MUTATION = "blob_mutation"
    SECRET_DISCOVERY = "secret_discovery"
    SECRET_MUTATION = "secret_mutation"


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
            Only ``DISCOVERY`` may set this — mutation tools must never be
            cached (would serve a stale write result), and BLOB/SECRET
            discovery tools today are not part of the per-call cache
            (``_registry.py`` asserts ``_CACHEABLE_DISCOVERY_TOOL_NAMES``
            is a subset of the DISCOVERY name-set). The construction-time
            invariant below catches the wider cacheable!=DISCOVERY shape
            at declaration time rather than deferring to the registry's
            import-time subset check (Python-engineer M3 review finding,
            2026-05-23).
        blob_store_only: True if the tool writes only to blob storage and
            never advances ``CompositionState`` (excluded from the
            ``trust_mode == "explicit_approve"`` proposal-interception gate).
            Only meaningful for ``BLOB_MUTATION``.
        augments_on_failure: True if a failure result from this tool should be
            decorated with inline plugin schemas via
            ``build_plugin_schemas_for_failure``. Set on mutation tools that
            route through ``_prevalidate_plugin_options`` and therefore emit
            ``Invalid options for <kind> '<plugin>'`` rejection messages the
            augmentation walker can parse. Closing the SSOT loop — previously
            this was a shadow ``Final[frozenset[str]]`` in ``_common.py`` that
            would have drifted on rename. Only meaningful for ``MUTATION`` /
            ``BLOB_MUTATION`` (DISCOVERY tools never write plugin config to
            state and so cannot surface plugin-option validation failures).
            Core-reviewer I5 review finding, 2026-05-24.
    """

    name: str
    handler: ToolHandler
    kind: ToolKind
    description: str
    json_schema: Mapping[str, Any]
    cacheable: bool = False
    blob_store_only: bool = False
    augments_on_failure: bool = False

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("ToolDeclaration.name must be non-empty.")

        if not isinstance(self.kind, ToolKind):
            raise TypeError(f"ToolDeclaration({self.name!r}).kind must be a ToolKind member, got {type(self.kind).__name__}.")

        # Cacheability invariant — only DISCOVERY tools may be cacheable.
        # Inverted (against the previous ``kind in _MUTATION_KINDS`` check)
        # to fire on construction for BLOB_DISCOVERY / SECRET_DISCOVERY too,
        # rather than deferring to the ``_registry.py`` subset assertion
        # (Python-engineer M3 review finding, 2026-05-23). The previous
        # invariant exclusively named mutation kinds; with the inverted
        # form an inadvertent ``cacheable=True`` on a non-DISCOVERY tool of
        # any kind fails at the declaration site.
        if self.cacheable and self.kind is not ToolKind.DISCOVERY:
            raise ValueError(
                f"ToolDeclaration({self.name!r}).cacheable=True is forbidden "
                f"for kind {self.kind.value!r}. Only DISCOVERY tools may be "
                "cacheable — mutation tools would serve stale write results, "
                "and BLOB/SECRET discovery tools are not part of the per-call "
                "cache contract (_registry.py enforces "
                "_CACHEABLE_DISCOVERY_TOOL_NAMES ⊆ _DISCOVERY_TOOL_NAMES)."
            )

        # JSON Schema validity invariant — the ``parameters`` shape we ship
        # to the LLM must meta-validate against the Draft 2020-12 spec. A
        # malformed schema (typo on ``type``, mistyped enum, structural
        # error) would otherwise escape to the model provider and fail at
        # compose time with a 400 response — the diagnostic the user pays
        # for is then opaque ("invalid_request_error" from upstream) and
        # the compose turn is lost. ``check_schema`` walks the metaschema,
        # not user data, so this is purely a construction-time check.
        # Validation must happen BEFORE ``freeze_fields`` because
        # ``Draft202012Validator.check_schema`` uses ``isinstance(x, dict)``
        # in its type checker and rejects ``MappingProxyType`` /
        # tuple-coerced ``required`` arrays. Systems-thinker
        # recommendation #3 (2026-05-23).
        try:
            Draft202012Validator.check_schema(self.json_schema)
        except SchemaError as exc:
            raise ValueError(
                f"ToolDeclaration({self.name!r}).json_schema is not a valid JSON Schema (Draft 2020-12): {exc.message}"
            ) from exc

        # blob_store_only invariant — only BLOB_MUTATION may set this.
        if self.blob_store_only and self.kind is not ToolKind.BLOB_MUTATION:
            raise ValueError(f"ToolDeclaration({self.name!r}).blob_store_only=True is only valid for BLOB_MUTATION.")

        # augments_on_failure invariant — only mutation-family kinds may set
        # this. DISCOVERY tools never call ``_prevalidate_plugin_options`` and
        # cannot emit the option-shape rejection messages the augmentation
        # walker matches against; advertising augments_on_failure on a
        # non-mutation tool would be a dead forward-pretend that this
        # invariant rejects at the declaration site. Mirrors the cacheable /
        # blob_store_only invariants above (Core-reviewer I5 review finding,
        # 2026-05-24).
        if self.augments_on_failure and self.kind not in {ToolKind.MUTATION, ToolKind.BLOB_MUTATION}:
            raise ValueError(
                f"ToolDeclaration({self.name!r}).augments_on_failure=True is "
                f"forbidden for kind {self.kind.value!r}. Only MUTATION / "
                "BLOB_MUTATION tools route through _prevalidate_plugin_options "
                "and therefore emit the option-shape rejection messages the "
                "augmentation walker matches against."
            )

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
) -> Mapping[str, Mapping[str, Any]]:
    """Return name → ``get_tool_definitions()``-shaped frozen mapping for every declaration.

    Each value has the legacy ``{"name", "description", "parameters"}`` shape
    used by the LLM tool-list. ``_dispatch.py:get_tool_definitions()`` reads
    this map and ``deep_thaw``s each entry per call to produce fresh mutable
    structures for the LLM-facing tool list.

    The returned mapping is deeply immutable: the outer mapping is a
    ``MappingProxyType``; each value is a ``MappingProxyType`` over its three
    fields; ``parameters`` is the declaration's already-deep-frozen
    ``json_schema``. This prevents the alias-mutation bug where
    ``get_tool_definitions()[i]["parameters"]`` would have been the same
    mutable dict across every call (Python-engineer H1 review finding,
    2026-05-23). Callers that need mutable JSON-shaped structures must
    ``deep_thaw`` the values they obtain — emission paths under
    ``_dispatch.py:get_tool_definitions`` do this.
    """
    return MappingProxyType(
        {
            decl.name: MappingProxyType(
                {
                    "name": decl.name,
                    "description": decl.description,
                    "parameters": decl.json_schema,
                }
            )
            for decl in tools
        }
    )


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


def derive_augments_on_failure_names(tools: Iterable[ToolDeclaration]) -> frozenset[str]:
    """Return the names of declarations that decorate failures with plugin schemas.

    Only ``MUTATION`` / ``BLOB_MUTATION`` tools may set
    ``augments_on_failure=True``; the ``ToolDeclaration`` constructor enforces
    this. ``_registry.py`` derives the set the dispatcher consumes via
    ``should_augment_with_plugin_schemas`` to decide whether to call
    ``build_plugin_schemas_for_failure`` on a failed result.

    Closing the SSOT loop: this used to be a hand-maintained
    ``Final[frozenset[str]]`` in ``_common.py`` that would have drifted on
    rename of any of the nine tools it named. Core-reviewer I5 review finding,
    2026-05-24.
    """
    return frozenset(decl.name for decl in tools if decl.augments_on_failure)


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
