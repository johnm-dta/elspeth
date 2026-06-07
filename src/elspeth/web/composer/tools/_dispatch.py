"""Composer dispatch — the tool execution surface.

Tool registration is aggregated in ``_registry.py`` (the single site that
imports every plane's ``TOOLS_IN_MODULE`` tuple and derives the per-kind
handler maps and name sets). This module hosts the dispatcher
(``execute_tool``), the LLM-facing tool-definitions emitter
(``get_tool_definitions``), and the import-time invariants on async / sync
registry separation, trailing-tool cache pin, and MANIFEST<->registry
set-equality.

Imports from ``_registry`` and ``discovery`` are for internal use inside
this module. Consumers needing the per-kind handler maps or name sets
should reach for ``_registry`` directly — ``_dispatch.__all__`` lists
only what this module defines (``execute_tool``, ``get_tool_definitions``,
``_inject_prior_validation``).
"""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from dataclasses import replace
from typing import Any, Final

from jsonschema import Draft202012Validator
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
)
from elspeth.web.composer.tools._registry import (
    _BLOB_DISCOVERY_TOOLS,
    _BLOB_MUTATION_TOOL_NAMES,
    _BLOB_MUTATION_TOOLS,
    _DISCOVERY_TOOLS,
    _MUTATION_TOOL_NAMES,
    _MUTATION_TOOLS,
    _REGISTERED_TOOLS,
    _SECRET_DISCOVERY_TOOLS,
    _SECRET_MUTATION_TOOL_NAMES,
    _SECRET_MUTATION_TOOLS,
    _TOOL_DEFS_BY_NAME,
    should_augment_with_plugin_schemas,
)
from elspeth.web.composer.tools.discovery import _SESSION_AWARE_TOOL_NAMES
from elspeth.web.composer.tools.sessions import (
    _SESSION_AWARE_TOOL_HANDLERS,
    ADVISOR_TRIGGER_VALUES,
)

__all__ = [
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


def _validate_and_freeze_tool_definition(defn: dict[str, Any]) -> Mapping[str, Any]:
    """Meta-validate ``defn["parameters"]`` against JSON Schema Draft 2020-12, then deep-freeze.

    Declared tools meta-validate their ``json_schema`` inside
    ``ToolDeclaration.__post_init__``. The two inline definitions below
    (``request_advisor_hint`` and ``request_interpretation_review``) do
    NOT flow through ``ToolDeclaration`` because their handlers are
    async / coroutine-shaped — but they ARE emitted on every
    ``get_tool_definitions`` call alongside the declared tools, so they
    must meet the same schema-validity contract. Without this check a
    typo in either inline schema would escape to the LLM API edge and
    fail at compose time with an opaque upstream 400. Systems-thinker
    recommendation #3 (2026-05-23).

    Validation runs BEFORE ``deep_freeze`` because
    ``Draft202012Validator.check_schema`` uses ``isinstance(x, dict)`` in
    its type checker and rejects ``MappingProxyType`` / tuple-coerced
    arrays the deep-freeze produces.
    """
    Draft202012Validator.check_schema(defn["parameters"])
    frozen: Mapping[str, Any] = deep_freeze(defn)
    return frozen


_REQUEST_ADVISOR_HINT_DEFINITION: Final[Mapping[str, Any]] = _validate_and_freeze_tool_definition(
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
                        "Why this advisor call is allowed. Use "
                        "proactive_security_safety before "
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
_REQUEST_INTERPRETATION_REVIEW_DEFINITION: Final[Mapping[str, Any]] = _validate_and_freeze_tool_definition(
    {
        "name": "request_interpretation_review",
        "description": (
            "Ask the user to review an LLM-authored assumption before it is "
            "finalised into the pipeline. This session-aware, kind-tagged "
            "review surface handles vague terms, invented source data, "
            "LLM prompt templates, and pipeline-shaping decisions. Use "
            "affected_node_id='source' for "
            "invented_source; use the LLM node id for vague_term and "
            "llm_prompt_template; use the implementing node id for "
            "pipeline_decision. Surface ONE assumption per call. The user "
            "will see your draft and resolve it in the review surface. Do not ask "
            "the user in assistant prose; this tool is the review surface. If "
            "no composition state exists yet, stage the affected source or node "
            "first, wait for that tool result, then call this tool. "
            "Do not call this merely because a concrete operator is present "
            "(e.g., 'rate 1-10'), but do call it when you authored the scale "
            "semantics, rubric, thresholds, category meaning, or subjective "
            "criterion definition behind that operator. Prompt-template review "
            "is not a substitute for an authored rubric/definition review. "
            "For LLM prompt templates, copying the user's supplied prompt "
            "verbatim is user-authored; creating a prompt template from the "
            "user's goal, data, or prose is LLM-authored and must be reviewed. "
            "Do not call this for terms the user already defined in the "
            "conversation."
        ),
        "parameters": {
            "type": "object",
            "additionalProperties": False,
            "required": ["affected_node_id", "kind", "user_term", "llm_draft"],
            "properties": {
                "affected_node_id": {
                    "type": "string",
                    "description": "Component id. Use 'source' for invented source data; use the LLM node id for vague terms and prompt templates.",
                },
                "kind": {
                    "type": "string",
                    "enum": ["vague_term", "invented_source", "llm_prompt_template", "pipeline_decision", "llm_model_choice"],
                    "description": "Class of assumption being surfaced for review.",
                },
                "user_term": {
                    "type": "string",
                    "description": "Stable user-facing label for the assumption being reviewed.",
                },
                "llm_draft": {
                    "type": "string",
                    "description": "LLM-authored interpretation, source data, or prompt template text.",
                },
            },
        },
    }
)


def get_tool_definitions() -> list[dict[str, Any]]:
    """Return JSON Schema tool definitions for the LLM.

    Returns 42 tools: 13 discovery + 13 mutation + 10 blob tools + 3 secret
    tools + 1 advisor tool + 1 session-aware interpretation-review tool.
    ``request_advisor_hint`` is always part of the LLM-visible list —
    advisor is mandatory — see ``ComposerServiceImpl._get_litellm_tools``.

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


# Import-time trailing-pin invariant. ``test_trailing_tool_name_is_locked``
# also tests this, but a test only fires at CI time; this check fires at
# module import, including at every web-service boot. An accidental
# reordering of ``get_tool_definitions()`` (e.g., moving ``wire_secret_ref``
# out of the trailing slot) would otherwise silently invalidate Anthropic's
# prompt-cache marker for every follow-up turn until the new trailing-tool
# prefix warmed up — visible only as a deploy-time cost spike and a
# latency regression, never as a crash. LLM-safety review finding #1 +
# solution-architect L1 + adapted from the test docstring (2026-05-23).
_TRAILING_TOOL_NAME: Final[str] = "wire_secret_ref"
_trailing_seen = get_tool_definitions()[-1]["name"]
if _trailing_seen != _TRAILING_TOOL_NAME:
    raise RuntimeError(
        f"Trailing tool in get_tool_definitions() is {_trailing_seen!r}, expected {_TRAILING_TOOL_NAME!r}. "
        "Anthropic cache_control markers attach to the last tool; reordering invalidates the "
        "prompt cache for every follow-up turn. If intentional, update _TRAILING_TOOL_NAME here "
        "and test_trailing_tool_name_is_locked, and consider the cache-miss cost on next deploy."
    )
del _trailing_seen


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

    For the mutation tools whose declarations set
    ``augments_on_failure=True`` (derived into
    ``_registry._AUGMENTS_ON_FAILURE_TOOL_NAMES``), scan
    ``result.validation.errors``
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
    user_message_content: str | None = None,
    composer_model_identifier: str | None = None,
    composer_model_version: str | None = None,
    composer_provider: str | None = None,
    composer_skill_hash: str | None = None,
    tool_arguments_hash: str | None = None,
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
        user_message_content: Triggering user chat-message content used to
            classify inline blob content as verbatim vs composer-authored.
        composer_model_identifier: Requested composer model identifier for
            LLM-authored blob/source provenance.
        composer_model_version: Provider-returned model/version when
            available, otherwise the requested model identifier.
        composer_provider: Composer LLM provider.
        composer_skill_hash: Hash of the composer skill markdown used for
            the request.
        tool_arguments_hash: Canonical audited arguments hash for this tool
            call.
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
        user_message_content=user_message_content,
        composer_model_identifier=composer_model_identifier,
        composer_model_version=composer_model_version,
        composer_provider=composer_provider,
        composer_skill_hash=composer_skill_hash,
        tool_arguments_hash=tool_arguments_hash,
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
if set(_SESSION_AWARE_TOOL_HANDLERS) != _SESSION_AWARE_TOOL_NAMES:
    raise RuntimeError(
        "_SESSION_AWARE_TOOL_HANDLERS keys diverge from _SESSION_AWARE_TOOL_NAMES: "
        f"+{set(_SESSION_AWARE_TOOL_HANDLERS) - _SESSION_AWARE_TOOL_NAMES} "
        f"-{_SESSION_AWARE_TOOL_NAMES - set(_SESSION_AWARE_TOOL_HANDLERS)}"
    )

# No tool name may be both a declared sync handler and a session-aware
# async handler. Detects the regression of a sync handler being given the
# same name as an existing session-aware async one.
_sync_declared_names: frozenset[str] = frozenset(decl.name for decl in _REGISTERED_TOOLS)
if _sync_declared_names & set(_SESSION_AWARE_TOOL_HANDLERS):
    raise RuntimeError(
        "Tool name appears in both _REGISTERED_TOOLS and _SESSION_AWARE_TOOL_HANDLERS: "
        f"{_sync_declared_names & set(_SESSION_AWARE_TOOL_HANDLERS)}"
    )

# Every session-aware handler must be a coroutine function. A sync function
# accidentally registered here would silently return a non-Awaitable; the
# compose-loop ``await`` would crash with TypeError at the worst time.
for _name, _handler in _SESSION_AWARE_TOOL_HANDLERS.items():
    if not asyncio.iscoroutinefunction(_handler):
        raise RuntimeError(
            f"_SESSION_AWARE_TOOL_HANDLERS[{_name!r}] is not async; sync handlers belong in _MUTATION_TOOLS / _DISCOVERY_TOOLS instead."
        )

# Every declared (non-session-aware) handler must NOT be a coroutine.
# Catches the reverse regression: an async handler dropped into a
# ``ToolDeclaration`` with a sync kind that would return a coroutine
# object as if it were a ToolResult.
for _decl in _REGISTERED_TOOLS:
    if asyncio.iscoroutinefunction(_decl.handler):
        raise RuntimeError(
            f"ToolDeclaration({_decl.name!r}).handler is async but kind={_decl.kind.value!r} is a sync kind. "
            "Async handlers belong in _SESSION_AWARE_TOOL_HANDLERS instead."
        )

del _decl, _name, _handler  # keep module namespace clean


# ---------------------------------------------------------------------------
# MANIFEST ↔ registry set-equality invariant.
#
# Closes LLM-safety review finding #2 (2026-05-23): a tool registered in the
# composer-tools surface but not present in ``redaction.MANIFEST`` is the
# asymmetric failure mode of the existing call-time guard. The dispatcher
# would route the call, the handler would run, and the response-redaction
# path at ``service.py`` would skip redaction silently because its lookup
# (``if tool_name in MANIFEST:``) gates on presence. Raw LLM-supplied
# arguments and raw handler responses would then land in ``chat_messages``
# without sanitisation — a Tier-1 audit-integrity breach.
#
# The reverse direction (MANIFEST entry without registered tool) was already
# caught at call time by ``redact_tool_call_response``'s ``not in MANIFEST``
# guard. This block closes the forward direction.
#
# Cycle note: ``redaction.py`` cannot perform this check itself — it is
# imported BY every plane module (which ``_registry.py`` aggregates). At
# the point ``_dispatch.py``'s module body executes, both ``redaction.MANIFEST``
# and the registry have fully loaded, so the import is safe here.
#
# ``request_advisor_hint`` lives in MANIFEST but not in any handler dict —
# it is intercepted in ``service.py`` before ``execute_tool``. Including
# it in the expected set keeps the equality precise.
# ---------------------------------------------------------------------------

from elspeth.web.composer.redaction import MANIFEST as _MANIFEST  # noqa: E402  (after-registry by design)

_expected_manifest_names: frozenset[str] = (
    frozenset(decl.name for decl in _REGISTERED_TOOLS) | frozenset(_SESSION_AWARE_TOOL_HANDLERS) | frozenset({"request_advisor_hint"})
)
_manifest_names: frozenset[str] = frozenset(_MANIFEST.keys())
if _expected_manifest_names != _manifest_names:
    raise RuntimeError(
        "redaction.MANIFEST diverges from the composer tool registry — every "
        "dispatchable tool MUST have a MANIFEST entry or its response will be "
        "persisted without redaction. "
        f"+{_expected_manifest_names - _manifest_names} (registered but no MANIFEST entry) "
        f"-{_manifest_names - _expected_manifest_names} (MANIFEST entry without a registered tool)"
    )
del _expected_manifest_names, _manifest_names
