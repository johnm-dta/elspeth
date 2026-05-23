"""Composer dispatch + registry — the tool execution surface (merges every plane's handlers)."""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Mapping
from dataclasses import replace
from typing import Any, Final, cast

from sqlalchemy import Engine

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
from elspeth.web.composer.tools.blobs import (
    TOOLS_IN_MODULE as _BLOBS_TOOLS_IN_MODULE,
)
from elspeth.web.composer.tools.blobs import (
    _execute_create_blob,
    _execute_delete_blob,
    _execute_get_blob_content,
    _execute_update_blob,
    _handle_get_blob_metadata,
    _handle_list_blobs,
)
from elspeth.web.composer.tools.declarations import (
    ToolDeclaration,
    ToolKind,
    assert_unique_names,
    derive_tool_definitions_by_name,
)
from elspeth.web.composer.tools.discovery import (
    _BLOB_DISCOVERY_TOOL_NAMES,
    _BLOB_MUTATION_TOOL_NAMES,
    _BLOB_STORE_ONLY_MUTATION_TOOL_NAMES,
    _CACHEABLE_DISCOVERY_TOOL_NAMES,
    _DISCOVERY_TOOL_NAMES,
    _MUTATION_TOOL_NAMES,
    _SECRET_DISCOVERY_TOOL_NAMES,
    _SECRET_MUTATION_TOOL_NAMES,
    _SESSION_AWARE_TOOL_NAMES,
)
from elspeth.web.composer.tools.generation import (
    TOOLS_IN_MODULE as _GENERATION_TOOLS_IN_MODULE,
)
from elspeth.web.composer.tools.generation import (
    _execute_diff_pipeline,
    _execute_explain_validation_error,
    _execute_get_audit_info,
    _execute_get_plugin_assistance,
    _execute_list_models,
    _execute_preview_pipeline,
    _handle_get_expression_grammar,
    _handle_get_plugin_schema,
)
from elspeth.web.composer.tools.outputs import (
    TOOLS_IN_MODULE as _OUTPUTS_TOOLS_IN_MODULE,
)
from elspeth.web.composer.tools.outputs import (
    _handle_patch_output_options,
    _handle_remove_output,
    _handle_set_output,
)
from elspeth.web.composer.tools.recipes import (
    TOOLS_IN_MODULE as _RECIPES_TOOLS_IN_MODULE,
)
from elspeth.web.composer.tools.recipes import (
    _execute_list_recipes,
)
from elspeth.web.composer.tools.secrets import (
    _execute_wire_secret_ref,
    _handle_list_secret_refs,
    _handle_validate_secret_ref,
)
from elspeth.web.composer.tools.sessions import (
    _SESSION_AWARE_TOOL_HANDLERS,
    ADVISOR_TRIGGER_VALUES,
    _execute_apply_pipeline_recipe,
    _execute_get_pipeline_state,
    _handle_set_pipeline,
)
from elspeth.web.composer.tools.sessions import (
    TOOLS_IN_MODULE as _SESSIONS_TOOLS_IN_MODULE,
)
from elspeth.web.composer.tools.sources import (
    TOOLS_IN_MODULE as _SOURCES_TOOLS_IN_MODULE,
)
from elspeth.web.composer.tools.sources import (
    _execute_inspect_source,
    _execute_set_source_from_blob,
    _handle_clear_source,
    _handle_list_sources,
    _handle_patch_source_options,
    _handle_set_source,
)
from elspeth.web.composer.tools.transforms import (
    TOOLS_IN_MODULE as _TRANSFORMS_TOOLS_IN_MODULE,
)
from elspeth.web.composer.tools.transforms import (
    _handle_list_sinks,
    _handle_list_transforms,
    _handle_patch_node_options,
    _handle_remove_edge,
    _handle_remove_node,
    _handle_set_metadata,
    _handle_upsert_edge,
    _handle_upsert_node,
)

# ---------------------------------------------------------------------------
# Registered tool declarations
#
# This is the single aggregation site for every plane's TOOLS_IN_MODULE tuple
# (ticket elspeth-6c9972ccbf). Aggregation lives here — not in
# ``declarations.py`` — to break the import cycle that would otherwise let
# a plane module observe an empty registry during partial-module-load.
#
# Step 1 of the migration registers only ``blobs.TOOLS_IN_MODULE``; Steps 2/3
# extend the tuple plane-by-plane. ``assert_unique_names`` fires at import
# time so two declarations with the same name fail the build, not silently
# at dispatch.
# ---------------------------------------------------------------------------

_REGISTERED_TOOLS: Final[tuple[ToolDeclaration, ...]] = (
    *_BLOBS_TOOLS_IN_MODULE,
    *_SOURCES_TOOLS_IN_MODULE,
    *_SESSIONS_TOOLS_IN_MODULE,
    *_GENERATION_TOOLS_IN_MODULE,
    *_RECIPES_TOOLS_IN_MODULE,
    *_TRANSFORMS_TOOLS_IN_MODULE,
    *_OUTPUTS_TOOLS_IN_MODULE,
)
assert_unique_names(_REGISTERED_TOOLS)
_TOOL_DEFS_BY_NAME: Final[Mapping[str, dict[str, Any]]] = derive_tool_definitions_by_name(_REGISTERED_TOOLS)


def get_tool_definitions() -> list[dict[str, Any]]:
    """Return JSON Schema tool definitions for the LLM.

    Returns 39 tools: 13 discovery + 13 mutation + 9 blob tools + 3 secret
    tools + 1 advisor tool. ``request_advisor_hint`` is the only tool that
    is filtered out of the LLM-visible list when the operator's
    ``composer_advisor_enabled`` flag is False (the default) — see
    ``ComposerServiceImpl._get_litellm_tools``.

    The skill at ``src/elspeth/web/composer/skills/pipeline_composer.md``
    enumerates the same tool set in its Foundation-knowledge section
    (under "## CRITICAL: Tool Schema Availability"). The drift gate
    ``TestComposerToolNameDrift::test_skill_tool_inventory_matches_get_tool_definitions``
    in ``tests/unit/web/composer/test_skill_drift.py`` enforces equality
    between the runtime list returned here and the skill's bulleted
    categories — adding a tool without updating both sides fails CI.
    """
    return [
        # Discovery tools
        _TOOL_DEFS_BY_NAME["list_sources"],
        _TOOL_DEFS_BY_NAME["list_transforms"],
        _TOOL_DEFS_BY_NAME["list_sinks"],
        _TOOL_DEFS_BY_NAME["get_plugin_schema"],
        _TOOL_DEFS_BY_NAME["get_expression_grammar"],
        # Mutation tools
        _TOOL_DEFS_BY_NAME["set_source"],
        _TOOL_DEFS_BY_NAME["upsert_node"],
        _TOOL_DEFS_BY_NAME["upsert_edge"],
        _TOOL_DEFS_BY_NAME["remove_node"],
        _TOOL_DEFS_BY_NAME["remove_edge"],
        _TOOL_DEFS_BY_NAME["set_metadata"],
        _TOOL_DEFS_BY_NAME["set_output"],
        _TOOL_DEFS_BY_NAME["remove_output"],
        _TOOL_DEFS_BY_NAME["patch_source_options"],
        _TOOL_DEFS_BY_NAME["patch_node_options"],
        _TOOL_DEFS_BY_NAME["patch_output_options"],
        _TOOL_DEFS_BY_NAME["set_pipeline"],
        # Source-reset and validation-explanation tools.
        _TOOL_DEFS_BY_NAME["clear_source"],
        _TOOL_DEFS_BY_NAME["explain_validation_error"],
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
                        "description": (
                            "The last validator error messages verbatim, most recent first. Include up to 5; do not paraphrase."
                        ),
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
        },
        _TOOL_DEFS_BY_NAME["get_plugin_assistance"],
        _TOOL_DEFS_BY_NAME["list_models"],
        _TOOL_DEFS_BY_NAME["get_audit_info"],
        _TOOL_DEFS_BY_NAME["preview_pipeline"],
        _TOOL_DEFS_BY_NAME["get_pipeline_state"],
        _TOOL_DEFS_BY_NAME["diff_pipeline"],
        # Blob tools
        {
            "name": "list_blobs",
            "description": "List uploaded/created files (blobs) in this session with metadata.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
        {
            "name": "get_blob_metadata",
            "description": "Get metadata for a specific blob (file) by ID.",
            "parameters": {
                "type": "object",
                "properties": {
                    "blob_id": {"type": "string", "description": "Blob ID."},
                },
                "required": ["blob_id"],
            },
        },
        _TOOL_DEFS_BY_NAME["set_source_from_blob"],
        _TOOL_DEFS_BY_NAME["create_blob"],
        _TOOL_DEFS_BY_NAME["update_blob"],
        _TOOL_DEFS_BY_NAME["delete_blob"],
        {
            "name": "get_blob_content",
            "description": "Retrieve the content of a blob (file) for inspection. Large files are truncated to 50,000 characters.",
            "parameters": {
                "type": "object",
                "properties": {
                    "blob_id": {
                        "type": "string",
                        "description": "ID of the blob to read.",
                    },
                },
                "required": ["blob_id"],
            },
        },
        _TOOL_DEFS_BY_NAME["list_recipes"],
        _TOOL_DEFS_BY_NAME["apply_pipeline_recipe"],
        {
            "name": "inspect_source",
            "description": (
                "Return bounded structural facts about a blob-backed source: source kind, observed "
                "headers, sample row count, inferred scalar types per column, URL candidates, and "
                "warnings. Reads at most 8 KiB of the blob and parses at most 100 rows. Use this "
                "before declaring a fixed CSV/JSON schema — observed headers and inferred types "
                "tell you which fields the source actually contains and what numeric coercion is "
                "needed before any gate or value_transform numeric op. Never returns raw row "
                "content; only summary facts."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "blob_id": {
                        "type": "string",
                        "description": "ID of the blob to inspect.",
                    },
                },
                "required": ["blob_id"],
            },
        },
        # Secret tools
        {
            "name": "list_secret_refs",
            "description": "List available secret references (API keys, credentials). Shows names and scopes, never values.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
        {
            "name": "validate_secret_ref",
            "description": "Check if a secret reference exists and is accessible to the current user.",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Secret reference name (e.g. 'OPENROUTER_API_KEY')."},
                },
                "required": ["name"],
            },
        },
        # Composer-LLM-callable tool surface for surfacing an interpretation
        # of a subjective or under-specified term for user review.
        # The description below is normative documentation for the LLM (mirrored
        # in the composer skill markdown) and is reviewed by the audit panel as
        # part of the request_interpretation_review event row's provenance.
        #
        # Position note: this tool is inserted BEFORE ``wire_secret_ref`` so
        # the trailing tool name remains ``wire_secret_ref`` — the Anthropic
        # cache-marker test (``test_trailing_tool_name_is_locked``) pins the
        # trailing position to preserve prompt-cache stability across deploys.
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
        },
        {
            "name": "wire_secret_ref",
            "description": "Place a secret reference marker in the pipeline config. The secret will be resolved at execution time.",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Secret reference name."},
                    "target": {
                        "type": "string",
                        "enum": ["source", "node", "output"],
                        "description": "Which component to wire the secret into.",
                    },
                    "target_id": {"type": "string", "description": "Node ID or output name (required for node/output targets)."},
                    "option_key": {"type": "string", "description": "Config option key to set (e.g. 'api_key')."},
                },
                "required": ["name", "target", "option_key"],
            },
        },
    ]


_DISCOVERY_TOOLS: dict[str, ToolHandler] = {
    "list_sources": _handle_list_sources,
    "list_transforms": _handle_list_transforms,
    "list_sinks": _handle_list_sinks,
    "get_plugin_schema": _handle_get_plugin_schema,
    "get_expression_grammar": _handle_get_expression_grammar,
    "explain_validation_error": _execute_explain_validation_error,
    "get_plugin_assistance": _execute_get_plugin_assistance,
    "list_models": _execute_list_models,
    "get_audit_info": _execute_get_audit_info,
    "list_recipes": _execute_list_recipes,
    "get_pipeline_state": _execute_get_pipeline_state,
    "preview_pipeline": _execute_preview_pipeline,
    "diff_pipeline": _execute_diff_pipeline,
}

# Backwards-compatible alias — the canonical declaration lives in
# ``tools.discovery``. Re-exported here because ``tools/__init__.py`` and a
# few external call sites historically imported the name from
# ``_dispatch``.
_CACHEABLE_DISCOVERY_TOOLS: frozenset[str] = _CACHEABLE_DISCOVERY_TOOL_NAMES


_MUTATION_TOOLS: dict[str, ToolHandler] = {
    "set_source": _handle_set_source,
    "upsert_node": _handle_upsert_node,
    "upsert_edge": _handle_upsert_edge,
    "remove_node": _handle_remove_node,
    "remove_edge": _handle_remove_edge,
    "set_metadata": _handle_set_metadata,
    "set_output": _handle_set_output,
    "remove_output": _handle_remove_output,
    "patch_source_options": _handle_patch_source_options,
    "patch_node_options": _handle_patch_node_options,
    "patch_output_options": _handle_patch_output_options,
    "set_pipeline": _handle_set_pipeline,
    "clear_source": _handle_clear_source,
}

_BLOB_DISCOVERY_TOOLS: dict[str, ToolHandler] = {
    "list_blobs": _handle_list_blobs,
    "get_blob_metadata": _handle_get_blob_metadata,
    "get_blob_content": _execute_get_blob_content,
    "inspect_source": _execute_inspect_source,
}


_BLOB_MUTATION_TOOLS: dict[str, ToolHandler] = {
    "set_source_from_blob": _execute_set_source_from_blob,
    "create_blob": _execute_create_blob,
    "update_blob": _execute_update_blob,
    "delete_blob": _execute_delete_blob,
    "apply_pipeline_recipe": _execute_apply_pipeline_recipe,
}

_SECRET_DISCOVERY_TOOLS: dict[str, ToolHandler] = {
    "list_secret_refs": _handle_list_secret_refs,
    "validate_secret_ref": _handle_validate_secret_ref,
}


_SECRET_MUTATION_TOOLS: dict[str, ToolHandler] = {
    "wire_secret_ref": _execute_wire_secret_ref,
}

# Registry-vs-declaration parity assertions — the canonical name sets live in
# ``tools.discovery``; the handler maps above are subordinate. A regression
# in either direction (a name in the registry without a declaration, or a
# declaration without a registered handler) fails the build at import time.
assert set(_DISCOVERY_TOOLS) == _DISCOVERY_TOOL_NAMES, (
    "_DISCOVERY_TOOLS keys diverge from _DISCOVERY_TOOL_NAMES: "
    f"+{set(_DISCOVERY_TOOLS) - _DISCOVERY_TOOL_NAMES} "
    f"-{_DISCOVERY_TOOL_NAMES - set(_DISCOVERY_TOOLS)}"
)
assert set(_MUTATION_TOOLS) == _MUTATION_TOOL_NAMES, (
    "_MUTATION_TOOLS keys diverge from _MUTATION_TOOL_NAMES: "
    f"+{set(_MUTATION_TOOLS) - _MUTATION_TOOL_NAMES} "
    f"-{_MUTATION_TOOL_NAMES - set(_MUTATION_TOOLS)}"
)
assert set(_BLOB_DISCOVERY_TOOLS) == _BLOB_DISCOVERY_TOOL_NAMES, (
    "_BLOB_DISCOVERY_TOOLS keys diverge from _BLOB_DISCOVERY_TOOL_NAMES: "
    f"+{set(_BLOB_DISCOVERY_TOOLS) - _BLOB_DISCOVERY_TOOL_NAMES} "
    f"-{_BLOB_DISCOVERY_TOOL_NAMES - set(_BLOB_DISCOVERY_TOOLS)}"
)
assert set(_BLOB_MUTATION_TOOLS) == _BLOB_MUTATION_TOOL_NAMES, (
    "_BLOB_MUTATION_TOOLS keys diverge from _BLOB_MUTATION_TOOL_NAMES: "
    f"+{set(_BLOB_MUTATION_TOOLS) - _BLOB_MUTATION_TOOL_NAMES} "
    f"-{_BLOB_MUTATION_TOOL_NAMES - set(_BLOB_MUTATION_TOOLS)}"
)
assert set(_SECRET_DISCOVERY_TOOLS) == _SECRET_DISCOVERY_TOOL_NAMES, (
    "_SECRET_DISCOVERY_TOOLS keys diverge from _SECRET_DISCOVERY_TOOL_NAMES: "
    f"+{set(_SECRET_DISCOVERY_TOOLS) - _SECRET_DISCOVERY_TOOL_NAMES} "
    f"-{_SECRET_DISCOVERY_TOOL_NAMES - set(_SECRET_DISCOVERY_TOOLS)}"
)
assert set(_SECRET_MUTATION_TOOLS) == _SECRET_MUTATION_TOOL_NAMES, (
    "_SECRET_MUTATION_TOOLS keys diverge from _SECRET_MUTATION_TOOL_NAMES: "
    f"+{set(_SECRET_MUTATION_TOOLS) - _SECRET_MUTATION_TOOL_NAMES} "
    f"-{_SECRET_MUTATION_TOOL_NAMES - set(_SECRET_MUTATION_TOOLS)}"
)
assert set(_SESSION_AWARE_TOOL_HANDLERS) == _SESSION_AWARE_TOOL_NAMES, (
    "_SESSION_AWARE_TOOL_HANDLERS keys diverge from _SESSION_AWARE_TOOL_NAMES: "
    f"+{set(_SESSION_AWARE_TOOL_HANDLERS) - _SESSION_AWARE_TOOL_NAMES} "
    f"-{_SESSION_AWARE_TOOL_NAMES - set(_SESSION_AWARE_TOOL_HANDLERS)}"
)
assert set(_DISCOVERY_TOOLS) >= _CACHEABLE_DISCOVERY_TOOL_NAMES, (
    f"Cacheable tools not in discovery registry: {_CACHEABLE_DISCOVERY_TOOL_NAMES - set(_DISCOVERY_TOOLS)}"
)


# ---------------------------------------------------------------------------
# Declaration ↔ hand-maintained registry parity (migration safety net)
#
# For every ToolDeclaration registered in this build, assert that the
# declaration's data agrees with the still-canonical hand-maintained
# surfaces for that tool (registry handler dict, blob-store-only set).
# A mismatch fails the build before any dispatch runs — protecting against
# a declaration that drifts away from the registry partway through the
# migration. Once every tool has a declaration (Step 3/4), the assertion
# inverts: the hand-maintained registries are decommissioned and the
# declarations become the canonical source the dispatcher reads.
# ---------------------------------------------------------------------------

_REGISTRY_BY_KIND: Final[Mapping[ToolKind, Mapping[str, ToolHandler]]] = {
    ToolKind.DISCOVERY: _DISCOVERY_TOOLS,
    ToolKind.MUTATION: _MUTATION_TOOLS,
    ToolKind.BLOB_DISCOVERY: _BLOB_DISCOVERY_TOOLS,
    ToolKind.BLOB_MUTATION: _BLOB_MUTATION_TOOLS,
    ToolKind.SECRET_DISCOVERY: _SECRET_DISCOVERY_TOOLS,
    ToolKind.SECRET_MUTATION: _SECRET_MUTATION_TOOLS,
}

for _decl in _REGISTERED_TOOLS:
    if _decl.kind is ToolKind.SESSION_AWARE:
        # Session-aware handlers live in ``_SESSION_AWARE_TOOL_HANDLERS`` and
        # are dispatched outside ``execute_tool``; their parity assertion
        # belongs in a later migration step.
        continue
    _registry = _REGISTRY_BY_KIND[_decl.kind]
    assert _decl.name in _registry, (
        f"ToolDeclaration({_decl.name!r}) has no entry in the hand-maintained registry for {_decl.kind.value!r}."
    )
    assert _registry[_decl.name] is _decl.handler, (
        f"ToolDeclaration({_decl.name!r}).handler diverges from the hand-maintained registry's handler for {_decl.kind.value!r}."
    )
    if _decl.kind is ToolKind.BLOB_MUTATION:
        assert (_decl.name in _BLOB_STORE_ONLY_MUTATION_TOOL_NAMES) == _decl.blob_store_only, (
            f"ToolDeclaration({_decl.name!r}).blob_store_only "
            f"({_decl.blob_store_only}) disagrees with "
            f"_BLOB_STORE_ONLY_MUTATION_TOOL_NAMES membership "
            f"({_decl.name in _BLOB_STORE_ONLY_MUTATION_TOOL_NAMES})."
        )

del _decl  # keep module namespace clean


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


# Module-level assertions — F-18 dual-registry invariant enforcement.
#
# These execute at module import, so a regression (e.g., copy-pasting an async
# handler into a sync registry, or registering one tool name in two registries)
# fails the build before any compose() call could trigger silent
# "coroutine was never awaited" warnings or first-registry-wins overrides.
_all_tools = (
    set(_DISCOVERY_TOOLS)
    | set(_MUTATION_TOOLS)
    | set(_BLOB_DISCOVERY_TOOLS)
    | set(_BLOB_MUTATION_TOOLS)
    | set(_SECRET_DISCOVERY_TOOLS)
    | set(_SECRET_MUTATION_TOOLS)
    | set(_SESSION_AWARE_TOOL_HANDLERS)
)
assert len(_all_tools) == (
    len(_DISCOVERY_TOOLS)
    + len(_MUTATION_TOOLS)
    + len(_BLOB_DISCOVERY_TOOLS)
    + len(_BLOB_MUTATION_TOOLS)
    + len(_SECRET_DISCOVERY_TOOLS)
    + len(_SECRET_MUTATION_TOOLS)
    + len(_SESSION_AWARE_TOOL_HANDLERS)
), (
    "Tool registry overlap detected — a tool name appears in more than one of "
    "_DISCOVERY_TOOLS / _MUTATION_TOOLS / blob / secret / _SESSION_AWARE_TOOL_HANDLERS"
)

# Every session-aware handler must be a coroutine function. A sync function
# accidentally registered here would silently return a non-Awaitable; the
# compose-loop ``await`` would crash with TypeError at the worst time.
for _name, _handler in _SESSION_AWARE_TOOL_HANDLERS.items():
    assert asyncio.iscoroutinefunction(_handler), (
        f"_SESSION_AWARE_TOOL_HANDLERS[{_name!r}] is not async; sync handlers belong in _MUTATION_TOOLS / _DISCOVERY_TOOLS instead."
    )

# Every sync-registry handler must NOT be a coroutine. Catches the reverse
# regression: an async handler dropped into the sync dispatch path that
# would return a coroutine object as if it were a ToolResult.
#
# The six sync registries have heterogeneous handler value-types (the blob
# and secret registries carry handlers with extra session-context kwargs),
# so the local ``_sync_registry`` is typed broadly as
# ``Mapping[str, Callable[..., Any]]`` for the duration of this check.
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
