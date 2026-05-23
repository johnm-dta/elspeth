"""Composer dispatch — the tool execution surface.

Tool registration is aggregated in ``_registry.py`` (the single site that
imports every plane's ``TOOLS_IN_MODULE`` tuple and derives the per-kind
handler maps and name sets). This module hosts the dispatcher
(``execute_tool``), the LLM-facing tool-definitions emitter
(``get_tool_definitions``), and the import-time invariants on async / sync
registry separation.

The per-kind registries (``_DISCOVERY_TOOLS``, ``_MUTATION_TOOLS``, ...)
are re-exported from this module under their historical names so external
consumers (tests, ``tools/__init__.py`` facade) continue to import them
from ``elspeth.web.composer.tools._dispatch``. Their canonical home is
``_registry``.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Mapping
from dataclasses import replace
from typing import Any, Final, cast

from sqlalchemy import Engine

from elspeth.contracts.freeze import deep_freeze, deep_thaw
from elspeth.contracts.secrets import WebSecretResolver
from elspeth.web.catalog.protocol import CatalogService
from elspeth.web.composer.state import (
    CompositionState,
    ValidationSummary,
)
from elspeth.web.composer.tools._common import (
    RuntimePreflight,
    ToolContext,
    ToolHandler,
    ToolResult,
    _failure_result,
    build_plugin_schemas_for_failure,
    should_augment_with_plugin_schemas,
)
from elspeth.web.composer.tools._registry import (
    _BLOB_DISCOVERY_TOOLS,
    _BLOB_MUTATION_TOOLS,
    _DISCOVERY_TOOLS,
    _MUTATION_TOOLS,
    _REGISTERED_TOOLS,
    _SECRET_DISCOVERY_TOOLS,
    _SECRET_MUTATION_TOOLS,
    _TOOL_DEFS_BY_NAME,
)
from elspeth.web.composer.tools.discovery import (
    _BLOB_MUTATION_TOOL_NAMES,
    _CACHEABLE_DISCOVERY_TOOL_NAMES,
    _MUTATION_TOOL_NAMES,
    _SECRET_MUTATION_TOOL_NAMES,
    _SESSION_AWARE_TOOL_NAMES,
)
from elspeth.web.composer.tools.sessions import (
    _SESSION_AWARE_TOOL_HANDLERS,
    ADVISOR_TRIGGER_VALUES,
)

# Backwards-compatible alias — the canonical name set lives in
# ``_registry._CACHEABLE_DISCOVERY_TOOL_NAMES``. ``tools/__init__.py``
# re-exports ``_CACHEABLE_DISCOVERY_TOOLS`` from here for external
# consumers historically reaching for it.
_CACHEABLE_DISCOVERY_TOOLS: Final[frozenset[str]] = _CACHEABLE_DISCOVERY_TOOL_NAMES


# The six sync per-kind registries (``_DISCOVERY_TOOLS`` etc.) are now
# derived in ``_registry`` from the declaration tuple; this module imports
# them above and re-exports them. ``_REGISTERED_TOOLS`` is referenced by
# the async/sync separation invariants below to validate every declared
# handler.

# Re-export the public ``ADVISOR_TRIGGER_VALUES`` constant — historically
# imported from ``_dispatch`` by callers reading the advisor schema. The
# canonical declaration lives in ``sessions.py``.
__all__ = [
    "ADVISOR_TRIGGER_VALUES",
    "_BLOB_DISCOVERY_TOOLS",
    "_BLOB_MUTATION_TOOLS",
    "_CACHEABLE_DISCOVERY_TOOLS",
    "_DISCOVERY_TOOLS",
    "_MUTATION_TOOLS",
    "_SECRET_DISCOVERY_TOOLS",
    "_SECRET_MUTATION_TOOLS",
    "_inject_prior_validation",
    "execute_tool",
    "get_tool_definitions",
]


# ---------------------------------------------------------------------------
# LLM-facing tool catalogue
#
# The bulk of the catalogue is derived from ``_TOOL_DEFS_BY_NAME``. Two tools
# are dispatched *outside* ``execute_tool`` and therefore do not carry a
# ``ToolDeclaration``:
#
# - ``request_advisor_hint`` — intercepted in ``service.py`` before dispatch.
# - ``request_interpretation_review`` — session-aware async handler in
#   ``sessions._SESSION_AWARE_TOOL_HANDLERS``; widening the declaration's
#   handler typing to admit async is deferred to ``elspeth-f5da936747``.
#
# These two definitions are emitted inline below.  ``wire_secret_ref`` is
# placed last so the Anthropic prompt-cache marker (pinned by
# ``test_trailing_tool_name_is_locked``) stays attached to it across deploys.
# ---------------------------------------------------------------------------

_REQUEST_ADVISOR_HINT_DEFINITION: Final[Mapping[str, Any]] = deep_freeze(
    {
        "name": "request_advisor_hint",
        "description": (
            "ESCAPE HATCH — call when one of the declared trigger criteria applies: "
            "reactive validation-loop recovery after two or more unchanged failures, "
            "proactive security/safety wiring review before `set_pipeline`, or "
            "proactive red-listed plugin review before `set_pipeline`. The proactive "
            "security trigger covers content moderation, prompt-injection defence, "
            "secret routing, PII/regulatory sinks, and externally fetched content "
            "flowing toward LLMs. Forwards your problem statement and context to a "
            "frontier model and returns guidance text. The reply is ADVICE, not "
            "configuration — you must still call the appropriate mutation tool "
            "yourself to apply any change. Budget is finite (sized per compose "
            "request, not per session lifetime) and exhausting it returns a "
            "structured error rather than crashing — inspect budget_remaining "
            "in each response. Do NOT call this tool in a loop, do NOT use it "
            "as a substitute for reading validator output. Disabled by default; "
            "only available when the operator has explicitly enabled it."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "trigger": {
                    "type": "string",
                    "enum": list(ADVISOR_TRIGGER_VALUES),
                    "description": (
                        "Why this advisor call is allowed. Use reactive_validation_loop "
                        "only after the recovery sequence and at least two unchanged "
                        "validator failures. Use proactive_security_safety before "
                        "set_pipeline for security/safety-sensitive flows. Use "
                        "proactive_red_listed_plugin before set_pipeline when the plan "
                        "uses a red-listed plugin such as llm, database, dataverse, "
                        "Azure safety transforms, RAG retrieval, or Chroma sinks."
                    ),
                },
                "problem_summary": {
                    "type": "string",
                    "description": (
                        "Your own statement of what you are trying to do and "
                        "why you are stuck. One or two sentences. Be specific: "
                        "'I cannot get llm transform options to validate against "
                        "the Azure provider schema' is useful; 'help' is not."
                    ),
                },
                "recent_errors": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": ("The last validator error messages verbatim, most recent first. Include up to 5; do not paraphrase."),
                },
                "attempted_actions": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "What you have already tried, one item per attempt. "
                        "Include the tool name and a one-line summary of the "
                        "argument shape. The advisor uses this to avoid "
                        "suggesting things you have already ruled out."
                    ),
                },
                "schema_excerpt": {
                    "type": "string",
                    "description": (
                        "Optional — the relevant plugin schema snippet you are "
                        "working against, as returned by `get_plugin_schema`. "
                        "Including this lets the advisor give field-level "
                        "guidance grounded in the exact contract."
                    ),
                },
            },
            "required": ["trigger", "problem_summary", "recent_errors", "attempted_actions"],
        },
    }
)


# The description below is normative documentation for the LLM (mirrored
# in the composer skill markdown) and is reviewed by the audit panel as
# part of the request_interpretation_review event row's provenance.
_REQUEST_INTERPRETATION_REVIEW_DEFINITION: Final[Mapping[str, Any]] = deep_freeze(
    {
        "name": "request_interpretation_review",
        "description": (
            "Ask the user to review your interpretation of a subjective or "
            "underspecified term they used. Call this BEFORE you finalise "
            "the prompt template for any LLM transform whose prompt depends "
            "on the term. Surface ONE term per call. The composition state "
            "MUST already contain the affected LLM transform (call upsert_node "
            "first) and its prompt_template MUST contain the placeholder "
            "{{interpretation:<term>}}. The user will see your draft and "
            "either accept it or amend it. Do not ask the user in assistant "
            "prose; this tool is the review surface. If no composition state "
            "exists yet, stage the LLM transform with a placeholder first, "
            "wait for that tool result, then call this tool. Do not call this "
            "for concrete operators (e.g., 'rate 1-10') or for terms the "
            "user already defined in the conversation."
        ),
        "parameters": {
            "type": "object",
            "additionalProperties": False,
            "required": ["affected_node_id", "user_term", "llm_draft"],
            "properties": {
                "affected_node_id": {
                    "type": "string",
                    "description": "node_id of the LLM transform whose prompt template depends on this term",
                },
                "user_term": {
                    "type": "string",
                    "description": "The user-provided term, verbatim (e.g., 'cool', 'important', 'risky')",
                },
                "llm_draft": {
                    "type": "string",
                    "description": "Your draft interpretation of the term, in your own words, suitable to embed as a phrase in the prompt template",
                },
            },
        },
    }
)


def get_tool_definitions() -> list[dict[str, Any]]:
    """Return JSON Schema tool definitions for the LLM.

    Returns 40 tools: 13 discovery + 13 mutation + 9 blob tools + 3 secret
    tools + 1 advisor tool + 1 session-aware interpretation-review tool.
    ``request_advisor_hint`` is filtered out of the LLM-visible list when
    the operator's ``composer_advisor_enabled`` flag is False (the default)
    — see ``ComposerServiceImpl._get_litellm_tools``.

    The tool catalogue is derived from ``_TOOL_DEFS_BY_NAME`` (every
    declared tool) plus two inline definitions for the dispatch-outside-
    execute_tool carve-outs (``request_advisor_hint`` and
    ``request_interpretation_review``). The trailing entry is pinned to
    ``wire_secret_ref`` by ``test_trailing_tool_name_is_locked`` —
    Anthropic prompt-cache markers attach to the last tool, and reordering
    invalidates the cache for every follow-up turn.

    The skill at ``src/elspeth/web/composer/skills/pipeline_composer.md``
    enumerates the same tool set in its Foundation-knowledge section
    (under "## CRITICAL: Tool Schema Availability"). Any skill ↔ runtime
    drift is caught by the per-tool prose tests in ``test_skill_drift.py``;
    the older inventory-parser drift gate (which existed because
    ``get_tool_definitions()`` was hand-maintained) was retired when
    declarations became the source of truth.
    """
    # Every entry on every call is freshly ``deep_thaw``ed from the deeply
    # immutable module-level registry. This guarantees mutually-isolated
    # mutable copies — a caller modifying ``result[0]["parameters"]["required"]``
    # cannot taint the registry's source-of-truth, nor a subsequent call's
    # output (Python-engineer H1 review finding, 2026-05-23). The cost is
    # one deep-walk per emission; ``get_tool_definitions`` is called once per
    # compose turn, so the cost is negligible against correctness.
    declared = [deep_thaw(defn) for name, defn in _TOOL_DEFS_BY_NAME.items() if name != "wire_secret_ref"]
    return [
        *declared,
        deep_thaw(_REQUEST_ADVISOR_HINT_DEFINITION),
        deep_thaw(_REQUEST_INTERPRETATION_REVIEW_DEFINITION),
        deep_thaw(_TOOL_DEFS_BY_NAME["wire_secret_ref"]),
    ]


def _inject_prior_validation(
    result: ToolResult,
    prior: ValidationSummary,
) -> ToolResult:
    """Attach prior validation to a successful mutation result for delta computation.

    Returns the result unchanged if the mutation failed or already carries
    prior_validation (set explicitly by the handler).
    """
    if result.success and result.prior_validation is None:
        return replace(result, prior_validation=prior)
    return result


# Tools that must scope their context with the live ValidationSummary so
# they can produce a delta against the caller's pre-mutation state. Every
# tool in the union of the three mutation registries; aliased here for the
# ``_inject_prior_validation`` wrap step in ``execute_tool``.
_ALL_MUTATION_TOOL_NAMES: Final[frozenset[str]] = _MUTATION_TOOL_NAMES | _BLOB_MUTATION_TOOL_NAMES | _SECRET_MUTATION_TOOL_NAMES


def _augment_with_plugin_schemas(
    result: ToolResult,
    tool_name: str,
    catalog: CatalogService,
) -> ToolResult:
    """Attach inline ``plugin_schemas`` to a failed option-shape rejection.

    For the mutation tools listed in
    ``_PLUGIN_SCHEMA_AUGMENTATION_TOOLS``, scan ``result.validation.errors``
    for ``Invalid options for <kind> '<plugin>'`` messages and embed the
    full ``get_plugin_schema`` payload for every named plugin. Eliminates
    the second round-trip the LLM would otherwise burn calling
    ``get_plugin_schema`` after each rejection (see composer session
    47cfbb5e on staging: 13 tool calls + 18 LLM rounds to converge a
    4-plugin pipeline because the model never preloaded schemas).

    No-op when the mutation succeeded, when no error message matches the
    option-shape pattern, when the result already carries
    ``plugin_schemas`` (handler set it directly), or when ``tool_name`` is
    not one of the augmentation-eligible tools.
    """
    if not should_augment_with_plugin_schemas(tool_name):
        return result
    if result.success or result.plugin_schemas is not None:
        return result
    schemas = build_plugin_schemas_for_failure(result, catalog)
    if schemas is None:
        return result
    return replace(result, plugin_schemas=schemas)


def execute_tool(
    tool_name: str,
    arguments: dict[str, Any],
    state: CompositionState,
    catalog: CatalogService,
    data_dir: str | None = None,
    session_engine: Engine | None = None,
    session_id: str | None = None,
    secret_service: WebSecretResolver | None = None,
    user_id: str | None = None,
    baseline: CompositionState | None = None,
    prior_validation: ValidationSummary | None = None,
    runtime_preflight: RuntimePreflight | None = None,
    max_blob_storage_per_session_bytes: int | None = None,
    user_message_id: str | None = None,
) -> ToolResult:
    """Execute a composition tool by name.

    Dispatches via a single registry lookup against the union of all six
    sync registries (``_DISCOVERY_TOOLS``, ``_MUTATION_TOOLS``,
    ``_BLOB_DISCOVERY_TOOLS``, ``_BLOB_MUTATION_TOOLS``,
    ``_SECRET_DISCOVERY_TOOLS``, ``_SECRET_MUTATION_TOOLS``). Every handler
    accepts the uniform ``(arguments, state, context)`` signature; the
    per-tool extended kwargs (``runtime_preflight`` for preview,
    ``baseline`` for diff, ``session_engine`` / ``user_message_id`` for
    blob tools, ``secret_service`` for secret tools) are threaded through
    the ``ToolContext`` constructed below from the caller's kwargs.

    Mutation tools are wrapped by ``_inject_prior_validation`` so the
    returned ToolResult carries the pre-mutation ValidationSummary on
    ``prior_validation`` (used by the compose loop to compute a delta
    against the post-mutation validation without re-running
    ``state.validate()``).

    Args:
        data_dir: Base data directory for S2 path allowlist enforcement.
            When provided, source options containing ``path`` or ``file``
            keys are restricted to ``{data_dir}/blobs/``.
        session_engine: SQLAlchemy engine for the session database.
            Required for blob tools to perform synchronous blob lookups.
        session_id: Current session ID. Required for blob tools.
        secret_service: ``WebSecretResolver`` — auth-scoped secret-reference
            resolver. Required for secret tools. Production wiring passes a
            ``ScopedSecretResolver`` (``elspeth.web.secrets.service``).
        user_id: Current user ID. Required for secret tools.
        baseline: Baseline state for diff_pipeline comparisons.
        prior_validation: Pre-computed validation for the current state.
            When provided, mutation tools reuse this instead of calling
            state.validate() for the pre-mutation delta. Callers should
            thread the previous ToolResult.validation forward — the state
            is immutable, so validation is deterministic.
        runtime_preflight: Optional callback for runtime-equivalent preflight.
            Only applied to preview_pipeline. Pre-computed in the async
            compose loop and injected here as a cheap synchronous callback
            so execute_tool() stays synchronous.
        max_blob_storage_per_session_bytes: Configured per-session blob
            storage quota for assistant-created session artifacts. Defaults
            to ``None`` so the blob plane can fall back to its historical
            BlobServiceImpl-compatible value for direct tests and non-web
            callers.
        user_message_id: Provenance pointer for blob writes that record
            ``created_from_message_id``.
    """
    all_handlers: dict[str, ToolHandler] = {
        **_DISCOVERY_TOOLS,
        **_MUTATION_TOOLS,
        **_BLOB_DISCOVERY_TOOLS,
        **_BLOB_MUTATION_TOOLS,
        **_SECRET_DISCOVERY_TOOLS,
        **_SECRET_MUTATION_TOOLS,
    }
    handler = all_handlers.get(tool_name)
    if handler is None:
        return _failure_result(state, f"Unknown tool: {tool_name}")

    # ``current_validation`` carries the live state's ValidationSummary
    # into ``diff_pipeline`` so its delta against the baseline is computed
    # using the caller's pre-mutation validation rather than re-running
    # ``state.validate()`` inside the handler. For every other handler,
    # the field is unused.
    context = ToolContext(
        catalog=catalog,
        data_dir=data_dir,
        session_engine=session_engine,
        session_id=session_id,
        secret_service=secret_service,
        user_id=user_id,
        baseline=baseline,
        current_validation=prior_validation,
        runtime_preflight=runtime_preflight,
        max_blob_storage_per_session_bytes=max_blob_storage_per_session_bytes,
        user_message_id=user_message_id,
    )

    if tool_name in _ALL_MUTATION_TOOL_NAMES:
        prior = prior_validation if prior_validation is not None else state.validate()
        result = handler(arguments, state, context)
        result = _inject_prior_validation(result, prior)
    else:
        result = handler(arguments, state, context)

    # Failure-response augmentation: when an option-shape mutation rejects,
    # embed the get_plugin_schema payloads for every plugin named in the
    # validation errors so the LLM avoids a follow-up discovery round-trip.
    # No-op for non-augmentation-eligible tools or successful results
    # (gated inside ``_augment_with_plugin_schemas``).
    return _augment_with_plugin_schemas(result, tool_name, catalog)


# ---------------------------------------------------------------------------
# Module-level invariants — F-18 dual-registry invariant enforcement.
#
# These execute at module import, so a regression (e.g., copy-pasting an async
# handler into a sync registry, or registering one tool name in two registries)
# fails the build before any compose() call could trigger silent
# "coroutine was never awaited" warnings or first-registry-wins overrides.
#
# The six sync per-kind registries are derived in ``_registry`` from
# ``_REGISTERED_TOOLS``; ``derive_handler_map_for`` partitions by kind so
# within-kind overlap is impossible by construction. The cross-kind overlap
# (a name appearing under two kinds) is prevented by ``assert_unique_names``
# in ``_registry``. The session-aware-vs-sync overlap below catches the
# remaining gap: a tool name shared between ``_SESSION_AWARE_TOOL_HANDLERS``
# (hand-maintained in ``sessions.py``) and any declared sync handler.
# ---------------------------------------------------------------------------

# Session-aware ↔ declared name parity. The hand-maintained
# ``_SESSION_AWARE_TOOL_HANDLERS`` dict in ``sessions.py`` is the source of
# truth for session-aware handlers; ``_SESSION_AWARE_TOOL_NAMES`` in
# ``discovery.py`` enumerates the same set. Drift would mean a tool is
# dispatched without a name-set membership (or vice versa).
assert set(_SESSION_AWARE_TOOL_HANDLERS) == _SESSION_AWARE_TOOL_NAMES, (
    "_SESSION_AWARE_TOOL_HANDLERS keys diverge from _SESSION_AWARE_TOOL_NAMES: "
    f"+{set(_SESSION_AWARE_TOOL_HANDLERS) - _SESSION_AWARE_TOOL_NAMES} "
    f"-{_SESSION_AWARE_TOOL_NAMES - set(_SESSION_AWARE_TOOL_HANDLERS)}"
)

# No tool name may be both a declared sync handler and a session-aware
# async handler. Detects the regression of a sync handler being given the
# same name as an existing session-aware async one.
_sync_declared_names: frozenset[str] = frozenset(decl.name for decl in _REGISTERED_TOOLS)
assert not (_sync_declared_names & set(_SESSION_AWARE_TOOL_HANDLERS)), (
    "Tool name appears in both _REGISTERED_TOOLS and _SESSION_AWARE_TOOL_HANDLERS: "
    f"{_sync_declared_names & set(_SESSION_AWARE_TOOL_HANDLERS)}"
)

# Every session-aware handler must be a coroutine function. A sync function
# accidentally registered here would silently return a non-Awaitable; the
# compose-loop ``await`` would crash with TypeError at the worst time.
for _name, _handler in _SESSION_AWARE_TOOL_HANDLERS.items():
    assert asyncio.iscoroutinefunction(_handler), (
        f"_SESSION_AWARE_TOOL_HANDLERS[{_name!r}] is not async; sync handlers belong in _MUTATION_TOOLS / _DISCOVERY_TOOLS instead."
    )

# Every declared (non-session-aware) handler must NOT be a coroutine.
# Catches the reverse regression: an async handler dropped into a
# ``ToolDeclaration`` with a sync kind that would return a coroutine
# object as if it were a ToolResult.
for _decl in _REGISTERED_TOOLS:
    assert not asyncio.iscoroutinefunction(_decl.handler), (
        f"ToolDeclaration({_decl.name!r}).handler is async but kind={_decl.kind.value!r} is a sync kind. "
        "Async handlers belong in _SESSION_AWARE_TOOL_HANDLERS instead."
    )

del _decl  # keep module namespace clean

# Cast the heterogeneous-handler-typed sync registries to a uniform
# ``Mapping[str, Callable[..., Any]]`` for the async-detection sweep
# below.  Kept for parity with the historical assertion shape even though
# the per-declaration sweep above is the authoritative check.
_sync_registries_for_check: tuple[tuple[str, Mapping[str, Callable[..., Any]]], ...] = (
    ("_DISCOVERY_TOOLS", cast(Mapping[str, Callable[..., Any]], _DISCOVERY_TOOLS)),
    ("_MUTATION_TOOLS", cast(Mapping[str, Callable[..., Any]], _MUTATION_TOOLS)),
    ("_BLOB_DISCOVERY_TOOLS", cast(Mapping[str, Callable[..., Any]], _BLOB_DISCOVERY_TOOLS)),
    ("_BLOB_MUTATION_TOOLS", cast(Mapping[str, Callable[..., Any]], _BLOB_MUTATION_TOOLS)),
    ("_SECRET_DISCOVERY_TOOLS", cast(Mapping[str, Callable[..., Any]], _SECRET_DISCOVERY_TOOLS)),
    ("_SECRET_MUTATION_TOOLS", cast(Mapping[str, Callable[..., Any]], _SECRET_MUTATION_TOOLS)),
)
for _sync_registry_name, _sync_registry in _sync_registries_for_check:
    for _name, _handler in _sync_registry.items():
        assert not asyncio.iscoroutinefunction(_handler), (
            f"{_sync_registry_name}[{_name!r}] is async; async handlers belong in _SESSION_AWARE_TOOL_HANDLERS instead."
        )
