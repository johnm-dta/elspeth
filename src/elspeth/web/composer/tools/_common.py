"""Shared toolkit for the composer-tool plane modules.

Centralises:

- ``ToolResult`` (the canonical response shape every handler returns) and the
  leaf response helpers (``_failure_result`` / ``_discovery_result`` /
  ``_mutation_result`` / ``_prepend_rejection_entry`` / ``_attach_post_call_hints``).
- Validation-delta and graph-repair-suggestion synthesis used by
  ``ToolResult.to_dict`` and the high-level ``diff_states`` reporter.
- The Pydantic mutation-argument validator and merge-patch helper used by every
  per-resource mutation handler.
- Base serialisation helpers for source/node/output/edge — leaves that the
  repair-suggestion generator and downstream pipeline-state serialiser both consume.
- The TypedDicts shared across pipeline-state, edge-contract, and repair payloads.

Layer: L3 (application). Imports from L0 contracts and the ``web.composer.state`` /
``web.composer.protocol`` / ``web.catalog.protocol`` / ``web.execution.schemas``
surfaces only — no sibling-plane imports.
"""

from __future__ import annotations

# Slice 4 — additional imports for shared validation/repair helpers.
import json
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from typing import Any, Final, TypedDict, cast

from pydantic import BaseModel
from pydantic import ValidationError as PydanticValidationError
from sqlalchemy import Engine

from elspeth.contracts.blobs_inline import is_widened_blob_ref
from elspeth.contracts.composer_interpretation import InterpretationKind
from elspeth.contracts.freeze import deep_thaw, freeze_fields
from elspeth.contracts.hashing import stable_hash
from elspeth.contracts.secrets import WebSecretResolver
from elspeth.contracts.sink import FILE_SINK_PLUGINS, FILE_SINK_REPAIR_EXTENSIONS
from elspeth.core.config import TriggerConfig
from elspeth.core.secrets import (
    collect_credential_field_violations,
    collect_disallowed_secret_ref_markers,
)
from elspeth.engine.orchestrator.preflight import check_config_value_sources
from elspeth.plugins.infrastructure.config_base import PluginConfigError
from elspeth.plugins.infrastructure.validation import (
    UnknownPluginTypeError,
    get_sink_config_model,
    get_source_config_model,
    get_transform_config_model,
)
from elspeth.web.catalog.policy_view import PolicyCatalogView
from elspeth.web.catalog.protocol import CatalogService, PluginKind
from elspeth.web.catalog.schemas import PluginSchemaInfo
from elspeth.web.composer.protocol import ToolArgumentError
from elspeth.web.composer.state import (
    CompositionState,
    EdgeSpec,
    NodeSpec,
    OutputSpec,
    SourceSpec,
    ValidationEntry,
    ValidationSummary,
    _coalesce_branch_connections,
    _coalesce_branch_names,
    _serialize_branches,
)
from elspeth.web.execution.schemas import ValidationResult
from elspeth.web.interpretation_state import (
    INTERPRETATION_REQUIREMENTS_KEY,
    SOURCE_AUTHORING_KEY,
    InterpretationRequirement,
    strip_authoring_options,
)
from elspeth.web.paths import (
    NESTED_LOCAL_PATH_OPTION_KEYS,
    SINK_LOCAL_PATH_OPTION_KEYS,
    SOURCE_LOCAL_PATH_OPTION_KEYS,
    allowed_sink_directories,
    allowed_source_directories,
    resolve_data_path,
)
from elspeth.web.plugin_policy.models import PluginAvailabilitySnapshot, PluginId, PluginUnavailableReason
from elspeth.web.provider_config_policy import web_llm_retry_budget_policy_error, web_rag_provider_config_policy_error
from elspeth.web.secrets.ref_policy import (
    allowed_secret_ref_fields,
    allowed_secret_ref_fields_text,
)
from elspeth.web.validation import (
    INTERPRETATION_PLACEHOLDER_RE,
)

_FULL_STATE_COMPONENT_ALIASES: Final[tuple[str, ...]] = ("", "full", "all", "pipeline")
_FULL_STATE_COMPONENT_ALIAS_SET: Final[frozenset[str]] = frozenset(_FULL_STATE_COMPONENT_ALIASES)
_DATA_ERROR_KEY: Final[str] = "error"
_RUNTIME_OWNED_LLM_OPTION_KEYS: Final[frozenset[str]] = frozenset({"resolved_prompt_template_hash"})
_RESOLVER_OWNED_INTERPRETATION_REQUIREMENT_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "event_id",
        "accepted_value",
        "accepted_artifact_hash",
        "resolved_prompt_template_hash",
    }
)


def _pending_interpretation_requirement(
    *,
    requirement_id: str,
    kind: InterpretationKind,
    user_term: str,
    draft: str,
) -> InterpretationRequirement:
    """Return a pending interpretation-review requirement row."""
    requirement: InterpretationRequirement = {
        "id": requirement_id,
        "kind": kind.value,
        "user_term": user_term,
        "status": "pending",
        "draft": draft,
        "event_id": None,
        "accepted_value": None,
        "accepted_artifact_hash": None,
        "resolved_prompt_template_hash": None,
    }
    return requirement


def _requirement_matches_field_value(requirement: Mapping[str, Any], field_value: str) -> bool:
    """True when ``requirement``'s draft/resolved hash already matches ``field_value``.

    Polymorphic by status: pending requirements carry the raw ``draft``
    string and compare equal; resolved requirements carry the stable hash
    of the accepted value in ``resolved_prompt_template_hash`` (a
    historical field name retained across kinds — the field is the
    universal resolved-value hash, not prompt-template-specific).
    """
    status = requirement["status"] if "status" in requirement else None
    if status == "pending":
        return requirement.get("draft") == field_value
    if status != "resolved":
        return False
    return requirement.get("resolved_prompt_template_hash") == stable_hash(field_value)


def _options_with_pending_requirement(
    options: Mapping[str, Any],
    *,
    requirement: Mapping[str, Any],
    replace_kind: InterpretationKind | None = None,
    current_field_value: str | None = None,
) -> Mapping[str, Any]:
    """Append or refresh a pending requirement without mutating ``options``.

    When ``replace_kind`` matches an existing requirement and
    ``current_field_value`` already equals that requirement's draft (or
    resolved hash), the existing requirement is kept — the call is
    idempotent so re-issuing the same mutation does not churn the review
    state. Otherwise the existing requirement of ``replace_kind`` is
    replaced with the supplied one.
    """
    requirements_value = options[INTERPRETATION_REQUIREMENTS_KEY] if INTERPRETATION_REQUIREMENTS_KEY in options else None
    if requirements_value is not None and not isinstance(requirements_value, (list, tuple)):
        return dict(options)

    requirements: list[Any] = list(requirements_value or ())
    if replace_kind is not None:
        next_requirements: list[Any] = []
        replaced = False
        for existing in requirements:
            if not isinstance(existing, Mapping) or existing.get("kind", InterpretationKind.VAGUE_TERM.value) != replace_kind.value:
                next_requirements.append(existing)
                continue
            if current_field_value is not None and _requirement_matches_field_value(existing, current_field_value):
                next_requirements.append(existing)
                replaced = True
                continue
            next_requirements.append(requirement)
            replaced = True
        if replaced:
            patched = dict(options)
            patched[INTERPRETATION_REQUIREMENTS_KEY] = next_requirements
            return patched

    requirements.append(requirement)
    patched = dict(options)
    patched[INTERPRETATION_REQUIREMENTS_KEY] = requirements
    return patched


def _options_with_default_prompt_template_review(
    *,
    node_id: str,
    plugin: str | None,
    options: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Ensure LLM-authored prompt templates carry a Class 3 review gate."""
    if plugin != "llm":
        return options
    prompt_template = options["prompt_template"] if "prompt_template" in options else None
    if not isinstance(prompt_template, str) or not prompt_template:
        return options
    requirement = _pending_interpretation_requirement(
        requirement_id=f"prompt_template_review:{node_id}",
        kind=InterpretationKind.LLM_PROMPT_TEMPLATE,
        user_term=f"llm_prompt_template:{node_id}",
        draft=prompt_template,
    )
    return _options_with_pending_requirement(
        options,
        requirement=requirement,
        replace_kind=InterpretationKind.LLM_PROMPT_TEMPLATE,
        current_field_value=prompt_template,
    )


def _options_with_default_model_choice_review(
    *,
    node_id: str,
    plugin: str | None,
    options: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Ensure LLM-authored model choices carry a review gate.

    Algorithmic enforcement of the "every model choice must be surfaced
    to the user" contract — every state mutation that sets ``options.model``
    on an LLM node MUST stage a pending interpretation requirement of
    kind ``llm_model_choice``. Mirrors
    :func:`_options_with_default_prompt_template_review` so the composer
    cannot ship a model identifier without surfacing it for review.

    The surfacing is provider-agnostic and unconditional. Whether the
    model id is also enforced by a live catalog (currently OpenRouter
    via ``CatalogValueSource`` in
    ``elspeth.plugins.transforms.llm.providers.openrouter``) is decided
    separately by the validator at preflight time — the operator's
    directive is "reliable surfacing", so we surface for every provider
    including operator-overridden ``base_url`` (chaos servers, private
    OpenAI-compatible gateways). The catalog SHA the choice was made
    against is captured separately on the audit ``runs`` row at
    execution time (``openrouter_catalog_sha256``), not here — that
    keeps the requirement shape symmetric with ``llm_prompt_template``
    (``draft == options.<field>`` invariant) and avoids leaking
    provider-specific metadata into a provider-agnostic surface.
    """
    if plugin != "llm":
        return options
    model = options["model"] if "model" in options else None
    if not isinstance(model, str) or not model:
        return options
    requirement = _pending_interpretation_requirement(
        requirement_id=f"model_choice_review:{node_id}",
        kind=InterpretationKind.LLM_MODEL_CHOICE,
        user_term=f"llm_model_choice:{node_id}",
        draft=model,
    )
    return _options_with_pending_requirement(
        options,
        requirement=requirement,
        replace_kind=InterpretationKind.LLM_MODEL_CHOICE,
        current_field_value=model,
    )


# Typographic punctuation an LLM routinely emits, mapped to its ASCII equivalent.
# web_scrape's ``http.scraping_reason`` / ``http.abuse_contact`` are sent verbatim
# as the X-Scraping-Reason / X-Abuse-Contact request headers, which must be
# ASCII-encodable (WebScrapeHTTPConfig). Folding the common typographic cases here
# lets composer-built pipelines (the first-run tutorial) round-trip; characters
# with no ASCII mapping are left untouched so the WebScrapeHTTPConfig validator
# still rejects them as a configuration error on hand-authored / YAML configs.
_TYPOGRAPHIC_TO_ASCII = {
    "\u2010": "-",  # hyphen
    "\u2011": "-",  # non-breaking hyphen
    "\u2012": "-",  # figure dash
    "\u2013": "-",  # en dash
    "\u2014": "-",  # em dash
    "\u2015": "-",  # horizontal bar
    "\u2018": "'",  # left single quotation mark
    "\u2019": "'",  # right single quotation mark / apostrophe
    "\u201a": "'",  # single low-9 quotation mark
    "\u201b": "'",  # single high-reversed-9 quotation mark
    "\u201c": '"',  # left double quotation mark
    "\u201d": '"',  # right double quotation mark
    "\u201e": '"',  # double low-9 quotation mark
    "\u201f": '"',  # double high-reversed-9 quotation mark
    "\u2032": "'",  # prime
    "\u2033": '"',  # double prime
    "\u2026": "...",  # horizontal ellipsis
    "\u00a0": " ",  # no-break space
    "\u2009": " ",  # thin space
    "\u202f": " ",  # narrow no-break space
}
_TYPOGRAPHIC_TRANSLATION = str.maketrans(_TYPOGRAPHIC_TO_ASCII)

_WIRE_VISIBLE_SCRAPE_HEADER_FIELDS = ("scraping_reason", "abuse_contact")


def _options_with_ascii_safe_scrape_headers(
    *,
    plugin: str | None,
    options: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Fold common typographic punctuation to ASCII in web_scrape header fields.

    No-op unless ``plugin == "web_scrape"`` and a header value actually changes,
    so it is safe to compose for every node. Only the wire-visible header fields
    are touched (a scrape node's prompt-like fields, and every other plugin's
    body text, are left alone). Characters with no ASCII mapping are preserved —
    the ``WebScrapeHTTPConfig`` validator rejects those as a configuration error.
    """
    if plugin != "web_scrape":
        return options
    http = options.get("http")
    if not isinstance(http, Mapping):
        return options
    folded_http: dict[str, Any] | None = None
    for field in _WIRE_VISIBLE_SCRAPE_HEADER_FIELDS:
        value = http.get(field)
        if not isinstance(value, str):
            continue
        folded = value.translate(_TYPOGRAPHIC_TRANSLATION)
        if folded != value:
            if folded_http is None:
                folded_http = dict(http)
            folded_http[field] = folded
    if folded_http is None:
        return options
    new_options = dict(options)
    new_options["http"] = folded_http
    return new_options


def _options_with_default_llm_reviews(
    *,
    node_id: str,
    plugin: str | None,
    options: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Apply every default review auto-stager for an LLM node, in order.

    Composes the per-field auto-stagers (prompt template, model choice) plus the
    web_scrape wire-visible-header ASCII fold, so call sites do not have to
    remember the full set. Each individual helper is a no-op when its trigger
    condition doesn't hold (non-llm plugin, missing field, non-scrape plugin), so
    the composition is safe for non-llm nodes and for partial node options.

    Adding a new default auto-stager here is the canonical extension point —
    callers stay on the composite and acquire the new gate automatically.
    """
    staged = _options_with_default_prompt_template_review(node_id=node_id, plugin=plugin, options=options)
    staged = _options_with_default_model_choice_review(node_id=node_id, plugin=plugin, options=staged)
    staged = _options_with_ascii_safe_scrape_headers(plugin=plugin, options=staged)
    return staged


class _SemanticEdgeContractPayload(TypedDict):
    """Wire shape for a serialized SemanticEdgeContract.

    Mirrors composer_mcp.server._SemanticEdgeContractPayload and
    web.execution.schemas.SemanticEdgeContractResponse exactly so HTTP,
    MCP, and ToolResult surfaces stay identical modulo transport
    envelope. If a field changes here, change it in all three places.
    """

    from_id: str
    to_id: str
    consumer_plugin: str
    producer_plugin: str | None
    producer_field: str
    consumer_field: str
    outcome: str
    requirement_code: str


class _FullPipelineStateMetadataPayload(TypedDict):
    """Metadata payload nested in full get_pipeline_state responses."""

    name: str | None
    description: str | None


class _FullPipelineStateInspectionPayload(TypedDict):
    """Inspection payload documenting how a full-state alias resolved."""

    requested_component: Any
    resolved_component: str
    accepted_full_state_aliases: list[str]


class _FullPipelineStatePayload(TypedDict):
    """Full-state payload returned by get_pipeline_state."""

    sources: dict[str, dict[str, Any]]
    nodes: list[dict[str, Any]]
    outputs: list[dict[str, Any]]
    edges: list[dict[str, Any]]
    metadata: _FullPipelineStateMetadataPayload
    version: int
    inspection: _FullPipelineStateInspectionPayload


class _RepairToolCall(TypedDict):
    tool: str
    arguments: Mapping[str, object]


class _AffectedConsumer(TypedDict):
    id: str
    current_input: str
    new_input: str


class _GraphRepairSuggestion(TypedDict):
    code: str
    connection: str
    strategy: str
    reason: str
    affected_consumers: list[_AffectedConsumer]
    tool_sequence: list[_RepairToolCall]


def _semantic_contracts_payload(
    contracts: tuple[Any, ...],
) -> list[_SemanticEdgeContractPayload]:
    """Serialize a SemanticEdgeContract tuple to JSON-friendly dicts.

    Centralized so ToolResult.to_dict and _execute_preview_pipeline
    emit identical shapes — and so adding a field updates both
    surfaces in one place.

    SemanticEdgeContract intentionally has no .to_dict() of its own:
    serialization happens at consumption sites so L0 stays free of
    JSON-encoding concerns. (See composer_mcp/server.py for the same
    pattern.)
    """
    return [
        _SemanticEdgeContractPayload(
            from_id=sc.from_id,
            to_id=sc.to_id,
            consumer_plugin=sc.consumer_plugin,
            producer_plugin=sc.producer_plugin,
            producer_field=sc.producer_field,
            consumer_field=sc.consumer_field,
            outcome=sc.outcome.value,
            requirement_code=sc.requirement.requirement_code,
        )
        for sc in contracts
    ]


def _compute_validation_delta(
    before: ValidationSummary,
    after: ValidationSummary,
) -> dict[str, Any]:
    """Compute new/resolved entries between two validation states.

    Compares by (component, message) tuple since ValidationEntry
    instances are recreated on each validate() call (no stable identity).
    """
    before_errors = {(e.component, e.message) for e in before.errors}
    after_errors = {(e.component, e.message) for e in after.errors}
    before_warnings = {(w.component, w.message) for w in before.warnings}
    after_warnings = {(w.component, w.message) for w in after.warnings}

    new_errors = [e.to_dict() for e in after.errors if (e.component, e.message) not in before_errors]
    resolved_errors = [e.to_dict() for e in before.errors if (e.component, e.message) not in after_errors]
    new_warnings = [w.to_dict() for w in after.warnings if (w.component, w.message) not in before_warnings]
    resolved_warnings = [w.to_dict() for w in before.warnings if (w.component, w.message) not in after_warnings]

    return {
        "new_errors": new_errors,
        "resolved_errors": resolved_errors,
        "new_warnings": new_warnings,
        "resolved_warnings": resolved_warnings,
    }


def _repair_identifier_fragment(value: str, *, fallback: str) -> str:
    """Return a connection-safe identifier fragment for generated repair skeletons."""
    fragment = re.sub(r"[^A-Za-z0-9_-]+", "_", value).strip("_-")
    if not fragment:
        return fallback
    if not fragment[0].isalnum():
        return f"{fallback}_{fragment}"
    return fragment


def _unique_name(candidate: str, reserved: set[str]) -> str:
    """Return candidate or a suffixed variant that does not collide with reserved."""
    if candidate not in reserved:
        reserved.add(candidate)
        return candidate

    index = 2
    while f"{candidate}_{index}" in reserved:
        index += 1
    unique = f"{candidate}_{index}"
    reserved.add(unique)
    return unique


def _reserved_connection_names(state: CompositionState) -> set[str]:
    """Collect existing route/connection/sink names a repair branch must avoid."""
    names: set[str] = {output.name for output in state.outputs}
    for source in state.sources.values():
        names.add(source.on_success)
        if source.on_validation_failure != "discard":
            names.add(source.on_validation_failure)

    for node in state.nodes:
        names.add(node.input)
        if node.on_success is not None:
            names.add(node.on_success)
        if node.on_error is not None and node.on_error != "discard":
            names.add(node.on_error)
        if node.routes is not None:
            names.update(node.routes.values())
        if node.fork_to is not None:
            names.update(node.fork_to)
        if node.branches is not None:
            names.update(_coalesce_branch_names(node.branches))
            names.update(_coalesce_branch_connections(node.branches))
    return names


def _duplicate_consumer_repair_suggestions(
    state: CompositionState,
    validation: ValidationSummary,
) -> list[_GraphRepairSuggestion]:
    """Build copyable repair skeletons for duplicate-consumer validation failures."""
    duplicate_error_components = {
        error.component
        for error in validation.errors
        if error.component.startswith("connection:") and error.message.startswith("Duplicate consumer for connection ")
    }
    if not duplicate_error_components:
        return []

    consumers_by_connection: dict[str, list[NodeSpec]] = {}
    for node in state.nodes:
        if node.node_type == "coalesce":
            continue
        if node.input not in consumers_by_connection:
            consumers_by_connection[node.input] = []
        consumers_by_connection[node.input].append(node)

    reserved_node_ids = {node.id for node in state.nodes}
    reserved_connection_names = _reserved_connection_names(state)
    suggestions: list[_GraphRepairSuggestion] = []

    for connection_name, consumer_nodes in sorted(consumers_by_connection.items()):
        if len(consumer_nodes) < 2 or f"connection:{connection_name}" not in duplicate_error_components:
            continue

        connection_fragment = _repair_identifier_fragment(connection_name, fallback="connection")
        gate_id = _unique_name(f"fork_{connection_fragment}", reserved_node_ids)
        branch_names = [
            _unique_name(
                f"{connection_fragment}_to_{_repair_identifier_fragment(node.id, fallback='node')}",
                reserved_connection_names,
            )
            for node in consumer_nodes
        ]
        gate_args: dict[str, object] = {
            "id": gate_id,
            "node_type": "gate",
            "plugin": None,
            "input": connection_name,
            "on_success": None,
            "on_error": None,
            "options": {},
            "condition": "True",
            "routes": {},
            "fork_to": branch_names,
            "branches": None,
            "policy": None,
            "merge": None,
            "trigger": None,
            "output_mode": None,
            "expected_output_count": None,
        }
        tool_sequence: list[_RepairToolCall] = []
        affected_consumers: list[_AffectedConsumer] = []
        for node, branch_name in zip(consumer_nodes, branch_names, strict=True):
            patched_consumer = _serialize_node(node)
            patched_consumer["input"] = branch_name
            tool_sequence.append({"tool": "upsert_node", "arguments": patched_consumer})
            affected_consumers.append(
                {
                    "id": node.id,
                    "current_input": connection_name,
                    "new_input": branch_name,
                }
            )
        tool_sequence.append({"tool": "upsert_node", "arguments": gate_args})
        tool_sequence.append({"tool": "preview_pipeline", "arguments": {}})
        suggestions.append(
            {
                "code": "duplicate_consumer_connection",
                "connection": connection_name,
                "strategy": "insert_fork_gate",
                "reason": "One connection can feed one processing node. Give each consumer a unique branch input, then insert a fork gate that consumes the shared connection and publishes those branch inputs from gate.fork_to.",
                "affected_consumers": affected_consumers,
                "tool_sequence": tool_sequence,
            }
        )

    return suggestions


def _graph_repair_suggestions(
    state: CompositionState,
    validation: ValidationSummary,
) -> list[_GraphRepairSuggestion]:
    """Return structured graph repair suggestions for validation failures."""
    return _duplicate_consumer_repair_suggestions(state, validation)


@dataclass(frozen=True, slots=True)
class ToolResult:
    """Result of a tool execution.

    Attributes:
        success: Whether the operation succeeded.
        updated_state: Full state after mutation (or original if success=False).
        validation: Stage 1 validation result for the updated state.
        affected_nodes: Node IDs changed or with changed edges.
        data: Optional data payload for discovery tools.
        prior_validation: Validation from before the mutation. When set,
            to_dict() includes a ``validation_delta`` showing new and
            resolved entries so the agent can focus on what changed.
        post_call_hints: Forward-looking coaching hints from the plugin
            that was just configured. Resolved by the catalog from
            ``BaseX.get_post_call_hints`` (see
            ``contracts/plugin_assistance.py``). Advisory only — not
            part of any audit hash. ``to_dict`` emits this field
            *only when non-empty* so existing tool consumers see no
            schema change.
        plugin_schemas: Inline ``get_plugin_schema`` payloads for every
            plugin named in a validation error of the form
            ``Invalid options for <kind> '<plugin>'``. Populated only on
            failed mutations (``success=False``) for the option-shape
            tools by ``execute_tool``. Keys are ``"<kind>/<plugin>"``
            strings sorted deterministically. ``to_dict`` emits this
            field *only when non-empty*. Eliminates the second
            round-trip the LLM would otherwise burn calling
            ``get_plugin_schema`` separately after each rejection.
    """

    success: bool
    updated_state: CompositionState
    validation: ValidationSummary
    affected_nodes: tuple[str, ...]
    data: Any = None
    prior_validation: ValidationSummary | None = None
    runtime_preflight: ValidationResult | None = None
    post_call_hints: tuple[str, ...] = ()
    plugin_schemas: Mapping[str, Mapping[str, Any]] | None = None

    def __post_init__(self) -> None:
        freeze_fields(self, "affected_nodes", "post_call_hints")
        if self.data is not None:
            freeze_fields(self, "data")
        if self.plugin_schemas is not None:
            freeze_fields(self, "plugin_schemas")

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a dict suitable for LLM tool response.

        Validation entries are serialized as structured dicts with
        component, message, and severity fields (B2 requirement).

        When prior_validation is set, includes a validation_delta with
        new_errors, resolved_errors, new_warnings, resolved_warnings to
        help the agent focus on what changed rather than re-reading the
        full validation state.
        """

        result: dict[str, Any] = {
            "success": self.success,
            "validation": {
                "is_valid": self.validation.is_valid,
                "errors": [e.to_dict() for e in self.validation.errors],
                "warnings": [e.to_dict() for e in self.validation.warnings],
                "suggestions": [e.to_dict() for e in self.validation.suggestions],
                "semantic_contracts": _semantic_contracts_payload(
                    self.validation.semantic_contracts,
                ),
                "graph_repair_suggestions": _graph_repair_suggestions(
                    self.updated_state,
                    self.validation,
                ),
            },
            "affected_nodes": list(self.affected_nodes),
            "version": self.updated_state.version,
        }
        if self.data is not None:
            result["data"] = deep_thaw(self.data)

        if self.runtime_preflight is not None:
            result["runtime_preflight"] = self.runtime_preflight.model_dump()

        if self.prior_validation is not None:
            result["validation_delta"] = _compute_validation_delta(
                self.prior_validation,
                self.validation,
            )

        if self.post_call_hints:
            result["post_call_hints"] = list(self.post_call_hints)

        if self.plugin_schemas:
            result["plugin_schemas"] = deep_thaw(self.plugin_schemas)

        return result


def diff_states(
    baseline: CompositionState,
    current: CompositionState,
    *,
    baseline_validation: ValidationSummary | None = None,
    current_validation: ValidationSummary | None = None,
) -> dict[str, Any]:
    """Compare two composition states and return a structured change summary.

    Reports added, removed, and modified sources/nodes/edges/outputs, plus
    and metadata changes. Used by the diff_pipeline MCP tool (B5).

    Args:
        baseline_validation: Pre-computed validation for the baseline state.
        current_validation: Pre-computed validation for the current state.
    """
    changes: dict[str, Any] = {
        "from_version": baseline.version,
        "to_version": current.version,
        "sources_changed": False,
        "metadata_changed": False,
        "nodes": {"added": [], "removed": [], "modified": []},
        "edges": {"added": [], "removed": [], "modified": []},
        "outputs": {"added": [], "removed": [], "modified": []},
    }

    if baseline.sources != current.sources:
        changes["sources_changed"] = True
        baseline_names = set(baseline.sources)
        current_names = set(current.sources)
        changes["sources"] = {
            "added": sorted(current_names - baseline_names),
            "removed": sorted(baseline_names - current_names),
            "modified": sorted(name for name in baseline_names & current_names if baseline.sources[name] != current.sources[name]),
        }

    # Metadata
    if baseline.metadata != current.metadata:
        changes["metadata_changed"] = True

    # Nodes
    baseline_nodes = {n.id: n for n in baseline.nodes}
    current_nodes = {n.id: n for n in current.nodes}
    for nid in current_nodes:
        if nid not in baseline_nodes:
            changes["nodes"]["added"].append(nid)
        elif current_nodes[nid] != baseline_nodes[nid]:
            changes["nodes"]["modified"].append(nid)
    for nid in baseline_nodes:
        if nid not in current_nodes:
            changes["nodes"]["removed"].append(nid)

    # Edges
    baseline_edges = {e.id: e for e in baseline.edges}
    current_edges = {e.id: e for e in current.edges}
    for eid in current_edges:
        if eid not in baseline_edges:
            changes["edges"]["added"].append(eid)
        elif current_edges[eid] != baseline_edges[eid]:
            changes["edges"]["modified"].append(eid)
    for eid in baseline_edges:
        if eid not in current_edges:
            changes["edges"]["removed"].append(eid)

    # Outputs
    baseline_outputs = {o.name: o for o in baseline.outputs}
    current_outputs = {o.name: o for o in current.outputs}
    for name in current_outputs:
        if name not in baseline_outputs:
            changes["outputs"]["added"].append(name)
        elif current_outputs[name] != baseline_outputs[name]:
            changes["outputs"]["modified"].append(name)
    for name in baseline_outputs:
        if name not in current_outputs:
            changes["outputs"]["removed"].append(name)

    # Validation delta — reuse pre-computed validations when available
    if baseline_validation is None:
        baseline_validation = baseline.validate()
    if current_validation is None:
        current_validation = current.validate()
    baseline_warnings = {e.message for e in baseline_validation.warnings}
    current_warnings = {e.message for e in current_validation.warnings}
    changes["warnings_introduced"] = sorted(current_warnings - baseline_warnings)
    changes["warnings_resolved"] = sorted(baseline_warnings - current_warnings)

    # Summary stats
    total = sum(len(changes[k][action]) for k in ("nodes", "edges", "outputs") for action in ("added", "removed", "modified"))
    total += int(changes["sources_changed"]) + int(changes["metadata_changed"])
    changes["total_changes"] = total

    return changes


def _validate_mutation_arguments(model: type[BaseModel], arguments: object, argument_name: str) -> BaseModel:
    try:
        return model.model_validate(arguments)
    except PydanticValidationError as exc:
        raise ToolArgumentError(
            argument=argument_name,
            expected=f"object conforming to {model.__name__}",
            actual_type=type(exc).__name__,
        ) from exc


def _attach_post_call_hints(
    result: ToolResult,
    catalog: CatalogService,
    *,
    plugin_type: PluginKind,
    tool_name: str,
    plugin_name: str | None,
    config_snapshot: Mapping[str, object],
) -> ToolResult:
    """Resolve postscript hints from the catalog and attach them to a successful result.

    No-ops when the mutation failed (we don't second-guess validation
    errors with coaching), when ``plugin_name`` is ``None`` (gates,
    coalesces — no plugin to resolve against), or when the plugin's
    ``get_post_call_hints`` returns an empty tuple (no hint to attach,
    so emit a result that doesn't carry the optional field at all).

    See ``contracts/plugin_assistance.py`` for the discipline and
    ``ToolResult.to_dict`` for the emission rule.
    """
    if not result.success or plugin_name is None:
        return result
    hints = catalog.post_call_hints(
        plugin_type=plugin_type,
        plugin_name=plugin_name,
        tool_name=tool_name,
        config_snapshot=config_snapshot,
    )
    if not hints:
        return result
    return replace(result, post_call_hints=hints)


def _discovery_result(state: CompositionState, data: Any) -> ToolResult:
    """Build a ToolResult for a discovery (read-only) tool."""
    validation = state.validate()
    return ToolResult(
        success=True,
        updated_state=state,
        validation=validation,
        affected_nodes=(),
        data=data,
    )


def _failure_result(
    state: CompositionState,
    error_msg: str,
    *,
    error_code: str | None = None,
) -> ToolResult:
    """Build a ToolResult for a failed mutation.

    The rejection reason (``error_msg``) is also prepended to
    ``validation.errors`` as a synthetic ``ValidationEntry`` with
    component ``"rejected_mutation"``. State-level errors from
    ``state.validate()`` (e.g. "No source configured.") describe the
    *unchanged* state and follow the rejection reason. This puts the
    action-rejection signal ahead of the stale-state signal for any
    consumer that reads ``validation.errors`` in array order — closing
    the convergence gap surfaced by composer session 58d7ede3 where the
    LLM repeated a near-identical ``set_pipeline`` because the array led
    with "No source configured." instead of the real option-shape
    error.
    """
    validation = _prepend_rejection_entry(state.validate(), error_msg)
    data = {_DATA_ERROR_KEY: error_msg}
    if error_code is not None:
        data["error_code"] = error_code
    return ToolResult(
        success=False,
        updated_state=state,
        validation=validation,
        affected_nodes=(),
        data=data,
    )


# Regex matching the option-shape failure messages emitted by
# ``_prevalidate_plugin_options`` (see ``_prevalidate_source`` /
# ``_prevalidate_transform`` / ``_prevalidate_sink``). The kind token is
# pinned to the three valid PluginKind values so an unrelated message
# containing ``Invalid options for ...`` text cannot trigger augmentation.
# The plugin name group accepts any non-apostrophe characters because
# plugin names are validated upstream.
_INVALID_OPTIONS_PLUGIN_RE: Final[re.Pattern[str]] = re.compile(
    r"Invalid options for (source|transform|sink) '([^']+)'",
)


def build_plugin_schemas_for_failure(
    result: ToolResult,
    catalog: CatalogService,
    *,
    schema_unavailable_message: Callable[[PluginSchemaInfo], str | None] | None = None,
) -> Mapping[str, Mapping[str, Any]] | None:
    """Build the ``plugin_schemas`` augmentation dict for a failed mutation.

    Scans every entry in ``result.validation.errors`` (including both the
    leading ``rejected_mutation`` entry and any state-level errors that
    follow). Each entry's ``message`` is regex-matched against
    ``_INVALID_OPTIONS_PLUGIN_RE``; every distinct ``(kind, plugin)`` pair
    is resolved through ``catalog.get_schema`` and dumped to a plain dict
    via ``PluginSchemaInfo.model_dump()`` so the payload is byte-identical
    to what the LLM would otherwise receive from a discrete
    ``get_plugin_schema`` tool call. When ``schema_unavailable_message`` is
    supplied, plugins hidden by the same availability gate as
    ``get_plugin_schema`` are omitted rather than inlining a forbidden schema.

    Returns ``None`` when the result is successful or when no error
    message matches the option-shape pattern. The caller is responsible
    for restricting the call to declarations that set
    ``augments_on_failure=True`` (gated by
    ``_registry.should_augment_with_plugin_schemas``).

    Trust tier: server-controlled response shaping. A regex match implies
    the validator already resolved the plugin in the catalog (the unknown
    -plugin path emits ``"Unknown <kind> plugin '<name>'"`` instead).
    Therefore ``catalog.get_schema`` returning ``ValueError`` here is a
    Tier-1 anomaly — propagate, do not silently omit.
    """
    if result.success:
        return None
    discovered: dict[tuple[str, str], Mapping[str, Any]] = {}
    for entry in result.validation.errors:
        for match in _INVALID_OPTIONS_PLUGIN_RE.finditer(entry.message):
            kind = cast(PluginKind, match.group(1))
            plugin_name = match.group(2)
            key = (kind, plugin_name)
            if key in discovered:
                continue
            schema = catalog.get_schema(kind, plugin_name)
            if schema_unavailable_message is not None and schema_unavailable_message(schema) is not None:
                continue
            discovered[key] = schema.model_dump()
    if not discovered:
        return None
    return {f"{kind}/{plugin_name}": payload for (kind, plugin_name), payload in sorted(discovered.items())}


def _prepend_rejection_entry(
    base: ValidationSummary,
    error_msg: str,
) -> ValidationSummary:
    """Return a ValidationSummary with a leading rejected_mutation entry.

    Preserves all non-error fields (warnings, suggestions,
    edge_contracts, semantic_contracts) verbatim. ``is_valid`` is
    forced to False because a rejection entry is by construction a
    high-severity error.
    """
    rejection = ValidationEntry(
        component="rejected_mutation",
        message=error_msg,
        severity="high",
    )
    return ValidationSummary(
        is_valid=False,
        errors=(rejection, *base.errors),
        warnings=base.warnings,
        suggestions=base.suggestions,
        edge_contracts=base.edge_contracts,
        semantic_contracts=base.semantic_contracts,
    )


def _mutation_result(
    new_state: CompositionState,
    affected: tuple[str, ...],
    *,
    prior_validation: ValidationSummary | None = None,
    data: Any = None,
    post_call_hints: tuple[str, ...] = (),
) -> ToolResult:
    """Build a ToolResult for a successful mutation.

    ``post_call_hints`` is the (possibly empty) tuple returned by the
    catalog's ``post_call_hints`` method for the just-configured
    plugin. Tool handlers compute it before calling here so the hint
    surface participates in the same envelope as ``validation`` and
    ``affected_nodes``. See ``contracts/plugin_assistance.py`` for
    the discipline; ``ToolResult.to_dict`` emits the field only when
    non-empty.
    """
    validation = new_state.validate()
    return ToolResult(
        success=True,
        updated_state=new_state,
        validation=validation,
        affected_nodes=affected,
        prior_validation=prior_validation,
        data=data,
        post_call_hints=post_call_hints,
    )


def _vf_destination_note(
    state: CompositionState,
    on_vf: str,
) -> dict[str, str] | None:
    """Advisory note when on_validation_failure references an unknown output.

    Returns a dict with a ``note`` key suitable for ``ToolResult.data``,
    or ``None`` when no advisory is needed (destination is ``"discard"``
    or matches a configured output).
    """
    if on_vf == "discard":
        return None
    output_names = {o.name for o in state.outputs}
    if on_vf not in output_names:
        current = sorted(output_names) if output_names else "(none)"
        return {
            "note": (
                f"on_validation_failure='{on_vf}' does not match any configured output. "
                "Use 'discard' to drop invalid rows without routing, or "
                f"add an output named '{on_vf}' before running the pipeline. "
                f"Current outputs: {current}."
            ),
        }
    return None


def _apply_merge_patch(
    target: Mapping[str, Any],
    patch: dict[str, Any],
) -> dict[str, Any]:
    """Shallow merge-patch: overwrite or delete top-level keys in target."""
    result = dict(target)
    for key, value in patch.items():
        if value is None:
            # Delete-if-present: a None patch value removes the key. Access the
            # key directly (R9 remediation) rather than pop-with-default; the
            # membership guard preserves the silent no-op on an absent key.
            if key in result:
                del result[key]
        else:
            result[key] = value
    return result


def _serialize_source(source: SourceSpec) -> dict[str, Any]:
    """Serialize a SourceSpec to a plain dict for LLM consumption."""
    return {
        "plugin": source.plugin,
        "on_success": source.on_success,
        "options": deep_thaw(source.options),
        "on_validation_failure": source.on_validation_failure,
    }


def _serialize_node(node: NodeSpec) -> dict[str, Any]:
    """Serialize a NodeSpec to a plain dict for LLM consumption.

    Includes all fields (even None) so the LLM sees the full schema.
    """
    return {
        "id": node.id,
        "node_type": node.node_type,
        "plugin": node.plugin,
        "input": node.input,
        "on_success": node.on_success,
        "on_error": node.on_error,
        "options": deep_thaw(node.options),
        "condition": node.condition,
        "routes": deep_thaw(node.routes) if node.routes else None,
        "fork_to": list(node.fork_to) if node.fork_to else None,
        "branches": _serialize_branches(node.branches) if node.branches else None,
        "policy": node.policy,
        "merge": node.merge,
        "trigger": deep_thaw(node.trigger) if node.trigger else None,
        "output_mode": node.output_mode,
        "expected_output_count": node.expected_output_count,
    }


def _serialize_output(output: OutputSpec) -> dict[str, Any]:
    """Serialize an OutputSpec to a plain dict for LLM consumption."""
    return {
        "sink_name": output.name,
        "plugin": output.plugin,
        "options": deep_thaw(output.options),
        "on_write_failure": output.on_write_failure,
    }


def _serialize_edge(edge: EdgeSpec) -> dict[str, Any]:
    """Serialize an EdgeSpec to a plain dict for LLM consumption."""
    return {
        "id": edge.id,
        "from_node": edge.from_node,
        "to_node": edge.to_node,
        "edge_type": edge.edge_type,
        "label": edge.label,
    }


def _serialize_full_pipeline_state(state: CompositionState, *, requested_component: Any) -> _FullPipelineStatePayload:
    """Serialize the full state and expose accepted full-state spellings."""
    return {
        "sources": {name: _serialize_source(source) for name, source in state.sources.items()},
        "nodes": [_serialize_node(n) for n in state.nodes],
        "outputs": [_serialize_output(o) for o in state.outputs],
        "edges": [_serialize_edge(e) for e in state.edges],
        "metadata": {"name": state.metadata.name, "description": state.metadata.description},
        "version": state.version,
        "inspection": {
            "requested_component": requested_component,
            "resolved_component": "full",
            "accepted_full_state_aliases": list(_FULL_STATE_COMPONENT_ALIASES),
        },
    }


# Slice 4 additions — shared validation/repair helpers, file-sink collision-policy
# cluster, and source-validation policy strings. Pulled to ``_common`` so the
# per-plane files (sources/transforms/sinks/outputs/sessions) can avoid importing
# each other.

_DEFAULT_SOURCE_VALIDATION_FAILURE: Final[str] = "discard"

_SOURCE_VALIDATION_FAILURE_DESCRIPTION: Final[str] = (
    "How to handle source validation failures. Use 'discard' to drop invalid rows without routing. "
    "Any other value, including 'quarantine', must match a configured output/sink name."
)


def _credential_wiring_contract_failure(
    state: CompositionState,
    *,
    component_id: str,
    component_type: str,
    plugin_type: PluginKind | None = None,
    plugin_name: str | None = None,
    options: Any,
) -> ToolResult | None:
    """Reject literal credentials before a mutation writes them into state.

    The returned message advertises the *inline* secret_ref form first
    because that is the only path that works for new nodes:

    - ``set_pipeline`` is atomic, so a node whose options omit a required
      credential field fails pydantic validation and the whole mutation
      rolls back — meaning ``wire_secret_ref`` cannot be used to attach
      the secret post-hoc (the node never lands in state).
    - ``collect_credential_field_violations`` short-circuits on
      ``{secret_ref: NAME}`` markers and ``set_pipeline`` strips those
      markers before pydantic validation, so passing the marker inline
      in the node's options is the supported new-node path.

    The post-hoc ``wire_secret_ref`` sequence is still documented as
    the secondary path for nodes that already exist in state.
    """
    plugin_specific_fields = (
        allowed_secret_ref_fields(plugin_type, plugin_name) if plugin_type is not None and plugin_name is not None else frozenset()
    )
    fields = tuple(
        dict.fromkeys(
            collect_credential_field_violations(
                options,
                additional_credential_fields=plugin_specific_fields,
            )
        )
    )
    if not fields:
        return None

    credential_fields = tuple(f"{component_id}:{field}" for field in fields)
    field_list = ", ".join(credential_fields)
    repair_sequence = ("list_secret_refs", "validate_secret_ref", "wire_secret_ref")
    repair_text = "list_secret_refs -> validate_secret_ref -> wire_secret_ref"
    inline_instruction = (
        "Set `<field>: {secret_ref: NAME}` directly in the node's options "
        "when calling set_pipeline / upsert_node. (The marker is stripped "
        "before option validation and resolved at execution time.) This "
        "rejection left pipeline state unchanged: repair by re-issuing only "
        "the rejected call with the marker substituted for the literal "
        "value — do not rebuild the pipeline from scratch. For a component "
        "already in state, patching just that component "
        "(patch_source_options / patch_node_options / patch_output_options) "
        "with the marker is the minimal correction."
    )
    post_hoc_instruction = f"Alternatively, after the node already exists in state, call {repair_text} to attach the marker post-hoc."
    error_msg = (
        f"Credential field(s) contain literal value(s): {field_list}. "
        f"Literal credential values were not stored. {inline_instruction} "
        f"{post_hoc_instruction}"
    )
    # Symmetric with _failure_result: lead validation.errors with the
    # rejection reason so LLMs reading the array in order see the
    # actionable message before any stale-state errors.
    validation = _prepend_rejection_entry(state.validate(), error_msg)
    return ToolResult(
        success=False,
        updated_state=state,
        validation=validation,
        affected_nodes=(),
        data={
            _DATA_ERROR_KEY: error_msg,
            "credential_fields": credential_fields,
            "components": (
                {
                    "component_id": component_id,
                    "component_type": component_type,
                    "fields": fields,
                },
            ),
            "repair": {
                "inline_form": {
                    "instruction": inline_instruction,
                    "example_options": {field: {"secret_ref": "<NAME>"} for field in fields},
                },
                "post_hoc_form": {
                    "instruction": post_hoc_instruction,
                    "tool_sequence": repair_sequence,
                },
            },
        },
    )


@dataclass(frozen=True, slots=True)
class PluginPolicyViolation:
    error_code: PluginUnavailableReason
    message: str


def _validate_plugin_name(
    context: ToolContext,
    plugin_type: PluginKind,
    name: str,
) -> PluginPolicyViolation | None:
    """Validate a new plugin selection against one request policy view."""
    try:
        plugin_id = PluginId(plugin_type, name)
    except ValueError:
        return PluginPolicyViolation(
            error_code=PluginUnavailableReason.NOT_INSTALLED,
            message=f"{plugin_type} plugin selection is unavailable ({PluginUnavailableReason.NOT_INSTALLED.value})",
        )
    reason = context.catalog.unavailable_reason(plugin_id)
    if reason is not None:
        return PluginPolicyViolation(
            error_code=reason,
            message=f"{plugin_type} plugin selection is unavailable ({reason.value})",
        )
    try:
        context.catalog.get_schema(plugin_type, name)
    except (ValueError, KeyError):
        return PluginPolicyViolation(
            error_code=PluginUnavailableReason.LOCAL_REQUIREMENT_MISSING,
            message=f"{plugin_type} plugin selection is unavailable ({PluginUnavailableReason.LOCAL_REQUIREMENT_MISSING.value})",
        )
    return None


def _plugin_policy_failure(
    state: CompositionState,
    violation: PluginPolicyViolation,
    *,
    component: str | None = None,
) -> ToolResult:
    message = violation.message if component is None else f"{component}: {violation.message}"
    return _failure_result(state, message, error_code=violation.error_code.value)


def _validate_aggregation_trigger(trigger: Any) -> str | None:
    """Return an error message if an aggregation trigger does not match runtime settings."""
    if trigger is None:
        return None
    try:
        TriggerConfig.model_validate(trigger)
    except PydanticValidationError as exc:
        detail = "; ".join(str(error["msg"]) for error in exc.errors())
        return f"Invalid aggregation trigger: {detail}"
    return None


def _validate_source_path(
    options: Mapping[str, Any],
    data_dir: str | None,
    *,
    require_data_dir: bool = False,
) -> str | None:
    """S2: Validate that path/file options are under allowed source directories.

    Returns an error message if validation fails, None if OK.
    Uses Path.resolve() + is_relative_to() to defeat ../ traversal.
    """
    for key in SOURCE_LOCAL_PATH_OPTION_KEYS:
        if key in options:
            if data_dir is None:
                if not require_data_dir:
                    return None
                return (
                    "Path violation (S2): source path options require data_dir "
                    "for allowlist enforcement. Bind uploaded files through "
                    "set_source_from_blob or provide the dispatcher data_dir."
                )
            allowed = allowed_source_directories(data_dir)
            resolved = resolve_data_path(options[key], data_dir)
            if not any(resolved.is_relative_to(d) for d in allowed):
                return (
                    f"Path violation (S2): '{options[key]}' is outside the "
                    f"allowed directories. Source file paths "
                    f"must be under {data_dir}/blobs/."
                )
    return None


def _validate_sink_path(
    options: dict[str, Any],
    data_dir: str | None,
    *,
    session_id: str | None,
) -> str | None:
    """Validate that sink path options are under allowed output directories.

    Returns an error message if validation fails, None if OK.
    Mirrors _validate_source_path but uses _allowed_sink_directories.
    Blob-directory writes are confined to the caller's own session subtree
    (elspeth-bdc17cfdb1); ``session_id=None`` fails closed to outputs only.
    """
    if data_dir is None:
        return None

    allowed = allowed_sink_directories(data_dir, session_id=session_id)

    for key in SINK_LOCAL_PATH_OPTION_KEYS:
        if key in options:
            resolved = resolve_data_path(options[key], data_dir)
            if not any(resolved.is_relative_to(d) for d in allowed):
                return (
                    f"Path violation (S2): '{key}' value '{options[key]}' is outside the "
                    f"allowed directories. Sink output paths "
                    f"must be under {data_dir}/outputs/ or this session's own "
                    f"{data_dir}/blobs/<session>/ subtree."
                )
    return None


def _validate_transform_provider_config_path(
    options: Mapping[str, Any],
    data_dir: str | None,
    *,
    session_id: str | None,
) -> str | None:
    """Validate nested provider_config path options are under allowed dirs.

    RAG retrieval transforms carry a local Chroma persist_directory under
    ``options["provider_config"]``. It is a read/write target like a sink, so
    it is confined to the allowed sink directories — including the per-session
    blob confinement (elspeth-bdc17cfdb1): a persist_directory pointed at
    another session's blob subtree would disclose that session's data on read
    as well as corrupt it on write. Non-RAG transforms have no provider_config
    dict and are skipped cleanly.

    Returns an error message if validation fails, None if OK.
    """
    if data_dir is None:
        return None

    provider_config = options.get("provider_config")
    if not isinstance(provider_config, Mapping):
        return None

    allowed = allowed_sink_directories(data_dir, session_id=session_id)

    for key in NESTED_LOCAL_PATH_OPTION_KEYS:
        if key not in provider_config:
            continue
        value = provider_config[key]
        # A null nested path must be skipped, not resolved — Path(None) raises.
        # Mirrors the runtime siblings (service/validation) which guard on
        # ``value is not None`` before resolving.
        if value is None:
            continue
        resolved = resolve_data_path(value, data_dir)
        if not any(resolved.is_relative_to(d) for d in allowed):
            return (
                f"Path violation (S2): provider_config '{key}' value "
                f"'{value}' is outside the allowed directories. "
                f"Transform provider paths must be under {data_dir}/outputs/ "
                f"or this session's own {data_dir}/blobs/<session>/ subtree."
            )
    return None


def _validate_transform_provider_config_policy(options: Mapping[str, Any], *, plugin: str | None = None) -> str | None:
    """Validate non-path web transform configuration policy constraints."""
    provider_policy_error = web_rag_provider_config_policy_error(options)
    if provider_policy_error is not None:
        return provider_policy_error
    return web_llm_retry_budget_policy_error(plugin, options)


def _prevalidate_plugin_options(
    plugin_type: PluginKind,
    plugin_name: str,
    options: dict[str, Any],
    *,
    injected_fields: dict[str, Any] | None = None,
) -> str | None:
    """Pre-validate plugin options against the plugin's config model.

    Catches missing required options (e.g., schema, operations) and
    malformed values (e.g., invalid field specs) BEFORE storing them in
    CompositionState. Returns None if valid, or a descriptive error
    message suitable for returning to the LLM agent.

    The plugin's own Pydantic config model is the authority — this
    function asks the plugin what it needs rather than hardcoding
    knowledge about individual plugins.

    Secret-ref markers (``{"secret_ref": "NAME"}``) are stripped before
    validation. The underlying Pydantic errors are filtered to exclude
    errors on secret-ref'd fields — those fields ARE provisioned, just
    deferred to execution time when ``resolve_secret_refs`` replaces them
    with actual values.

    Args:
        plugin_type: "source", "transform", or "sink".
        plugin_name: Plugin name (e.g., "csv", "value_transform").
        options: Options dict as provided by the LLM agent.
        injected_fields: Synthetic values for fields that come from
            other parts of the pipeline spec (e.g., on_validation_failure
            for sources). Merged into options for validation only —
            not stored.
    """
    secret_ref_placement_error = _secret_ref_placement_error(plugin_type, plugin_name, options)
    if secret_ref_placement_error is not None:
        return secret_ref_placement_error

    try:
        if plugin_type == "source":
            config_cls = get_source_config_model(plugin_name)
        elif plugin_type == "transform":
            config_cls = get_transform_config_model(plugin_name, options)
        elif plugin_type == "sink":
            config_cls = get_sink_config_model(plugin_name)
        else:
            # PluginKind is Literal["source", "transform", "sink"] — unreachable.
            raise AssertionError(f"_prevalidate_plugin_options: unexpected plugin_type={plugin_type!r}")
    except UnknownPluginTypeError:
        return f"Unknown {plugin_type} plugin '{plugin_name}'. Call list_{plugin_type}s to see available {plugin_type} plugins."
    except ValueError as exc:
        # Config model selection raised (e.g. unknown LLM provider) — surface it.
        return f"Invalid options for {plugin_type} '{plugin_name}': {exc}"

    if config_cls is None:
        return None

    # Options may contain frozen containers (MappingProxyType, tuple) from
    # CompositionState.  Thaw them so Pydantic receives plain dicts/lists.
    merged = deep_thaw(options)
    if injected_fields:
        for k, v in injected_fields.items():
            if k not in merged:
                merged[k] = v
    if plugin_type == "transform" and plugin_name == "llm":
        _mask_pending_interpretation_placeholders_for_authoring_validation(merged)

    # Strip secret_ref markers before validation.  A secret-ref'd field
    # IS provisioned (the user called wire_secret_ref), just deferred to
    # execution time.  Stripping it may cause Pydantic to report
    # "field required" — we filter those errors out below.
    secret_ref_keys: set[str] = set()
    for key, value in list(merged.items()):
        if isinstance(value, Mapping) and len(value) == 1 and "secret_ref" in value and isinstance(value["secret_ref"], str):
            secret_ref_keys.add(key)
            del merged[key]

    # Strip widened blob_ref(inline_content) markers before validation.  Like
    # secret_ref, these fields are provisioned but deferred to runtime
    # resolution; bind_source remains source-only and is deliberately not
    # stripped here.
    blob_inline_ref_keys: set[str] = set()
    for key, value in list(merged.items()):
        shape = is_widened_blob_ref(value)
        if shape is not None and shape.mode == "inline_content":
            blob_inline_ref_keys.add(key)
            del merged[key]

    try:
        config = config_cls.from_dict(merged, plugin_name=plugin_name)
    except PluginConfigError as exc:
        if not secret_ref_keys and not blob_inline_ref_keys:
            # No secret refs were stripped — report the error as-is.
            msg = exc.cause if exc.cause is not None else str(exc)
            return f"Invalid options for {plugin_type} '{plugin_name}': {msg}"

        # Secret refs were stripped.  Filter out errors on those fields.
        cause = exc.__cause__
        if not isinstance(cause, PydanticValidationError):
            # ValueError path (model validators) — can't filter per-field.
            msg = exc.cause if exc.cause is not None else str(exc)
            return f"Invalid options for {plugin_type} '{plugin_name}': {msg}"

        stripped_keys = secret_ref_keys | blob_inline_ref_keys
        remaining = [e for e in cause.errors() if not (e["loc"] and e["loc"][0] in stripped_keys)]
        if not remaining:
            return None

        # Re-format only the non-secret errors.
        lines = "; ".join(f"{'.'.join(str(p) for p in e['loc'])}: {e['msg']}" for e in remaining)
        return f"Invalid options for {plugin_type} '{plugin_name}': {lines}"

    # Construction passed type/required validation. Now enforce the config's
    # VALUE_SOURCES declarations (e.g. OpenRouter ``model`` catalog membership)
    # at authoring time — the same structured check the bundle walker runs at
    # instantiation (engine/orchestrator/preflight.py). This catches a
    # hallucinated catalog value here, with an actionable ``list_models`` hint,
    # instead of letting it slip through prevalidation. Catalog membership is a
    # value-source concern, deliberately NOT enforced in config construction.
    value_source_findings = check_config_value_sources(config, component_id=plugin_name)
    if value_source_findings:
        return f"Invalid options for {plugin_type} '{plugin_name}': " + "; ".join(f.reason for f in value_source_findings)
    return None


def _mask_pending_interpretation_placeholders_for_authoring_validation(
    options: dict[str, Any],
) -> None:
    """Allow unresolved interpretation placeholders during composer authoring.

    ``{{interpretation:<term>}}`` is a Phase 5b composer-review token, not a
    runtime Jinja variable. The LLM must be able to stage a pending LLM node
    carrying that token so ``request_interpretation_review`` can create the
    audit row and the user can resolve it. Runtime remains strict: execution
    rejects unresolved placeholders before YAML generation, and resolved
    prompts validate through the normal LLM config path.
    """

    if "resolved_prompt_template_hash" in options:
        return
    prompt_template = options.get("prompt_template")
    if not isinstance(prompt_template, str):
        return
    options["prompt_template"] = INTERPRETATION_PLACEHOLDER_RE.sub(
        "pending interpretation",
        prompt_template,
    )


def _resolver_owned_interpretation_requirement_error(
    options: Mapping[str, Any],
    *,
    tool_name: str,
) -> str | None:
    """Reject LLM-supplied ``interpretation_requirements`` carrying resolver-owned review metadata.

    Composer tool input may stage only PENDING review requirements; a resolved
    ``status`` or any resolver-owned field (``event_id`` / ``accepted_value`` /
    ``accepted_artifact_hash`` / ``resolved_prompt_template_hash``) may be written
    ONLY by ``resolve_interpretation_event``, which records a real human
    resolution in the interpretation-events audit DB.

    This check is PLUGIN-AGNOSTIC and must guard every write path to a spec's
    ``options`` — both LLM-node options (``vague_term`` / ``llm_prompt_template``
    / ``llm_model_choice`` requirements) and SOURCE options
    (``invented_source`` requirements). The read side that decides whether an
    LLM-authored ("invented") source still needs human review
    (``interpretation_state._pending_source_sites``) trusts a self-reported
    ``status == "resolved"`` + ``accepted_artifact_hash`` match without
    consulting the events DB, so this write-boundary guard is the only real
    defence against a forged "resolved" requirement. Apply it to the
    LLM-SUPPLIED delta (full options on a create, the ``patch`` on a merge) so a
    legitimately-resolved requirement already in stored state is not re-flagged.
    """
    requirements_value = options[INTERPRETATION_REQUIREMENTS_KEY] if INTERPRETATION_REQUIREMENTS_KEY in options else None
    if not isinstance(requirements_value, (list, tuple)):
        return None

    for index, requirement in enumerate(requirements_value):
        if not isinstance(requirement, Mapping):
            continue
        status = requirement["status"] if "status" in requirement else None
        if status not in (None, "pending"):
            return (
                f"{tool_name} options.{INTERPRETATION_REQUIREMENTS_KEY}[{index}] includes "
                f"resolver-owned status {status!r}. Composer tool input may stage pending "
                "review requirements only; resolved review metadata may only be written by "
                "resolve_interpretation_event."
            )
        resolver_owned_fields = sorted(
            field for field in _RESOLVER_OWNED_INTERPRETATION_REQUIREMENT_FIELDS if requirement.get(field) is not None
        )
        if resolver_owned_fields:
            field_names = ", ".join(resolver_owned_fields)
            return (
                f"{tool_name} options.{INTERPRETATION_REQUIREMENTS_KEY}[{index}] includes "
                f"resolver-owned field(s): {field_names}. Composer tool input may stage "
                "pending review requirements only; resolved review metadata may only be "
                "written by resolve_interpretation_event."
            )
    return None


def _runtime_owned_llm_option_error(
    plugin_name: str | None,
    options: Mapping[str, Any],
    *,
    tool_name: str,
) -> str | None:
    """Reject composer-authored writes to runtime-owned LLM audit fields.

    Two checks: (1) the LLM-only runtime-owned option keys
    (``_RUNTIME_OWNED_LLM_OPTION_KEYS``, e.g. ``resolved_prompt_template_hash``
    at the top level), gated on ``plugin_name == "llm"``; and (2) the
    plugin-agnostic resolver-owned interpretation-requirement check, which also
    guards source write paths via
    :func:`_resolver_owned_interpretation_requirement_error`.
    """
    if plugin_name != "llm":
        return None
    supplied = sorted(key for key in _RUNTIME_OWNED_LLM_OPTION_KEYS if key in options)
    if supplied:
        field_names = ", ".join(supplied)
        return (
            f"{tool_name} options include runtime-owned LLM option(s): {field_names}. "
            "These audit-link fields may only be written by resolve_interpretation_event, "
            "not by composer tool input."
        )

    return _resolver_owned_interpretation_requirement_error(options, tool_name=tool_name)


def _secret_ref_placement_error(
    plugin_type: PluginKind,
    plugin_name: str,
    options: dict[str, Any],
) -> str | None:
    """Return a policy error for secret_ref markers in non-credential fields."""
    secret_ref_placement_violations = collect_disallowed_secret_ref_markers(
        options,
        additional_allowed_fields=allowed_secret_ref_fields(plugin_type, plugin_name),
    )
    if not secret_ref_placement_violations:
        return None

    violation_text = ", ".join(f"{v.field_path} -> {v.secret_name}" for v in secret_ref_placement_violations)
    allowed_text = allowed_secret_ref_fields_text(plugin_type, plugin_name)
    return (
        f"Invalid secret_ref placement for {plugin_type} '{plugin_name}': {violation_text}; "
        "only credential-bearing fields may carry secret_ref markers. "
        f"Allowed credential-bearing fields: {allowed_text}."
    )


_WEB_ONLY_SOURCE_KEYS = frozenset({"blob_ref", SOURCE_AUTHORING_KEY})


def _source_options_for_prevalidation(options: Mapping[str, Any]) -> dict[str, Any]:
    """Strip source blob-binding metadata before plugin config validation."""
    filtered = strip_authoring_options(options)
    for key in _WEB_ONLY_SOURCE_KEYS:
        if key in filtered:
            del filtered[key]
    if options.get("blob_ref") is not None and options.get("mode") == "bind_source" and "mode" in filtered:
        del filtered["mode"]
    return filtered


_WRITE_COLLISION_POLICIES = frozenset({"fail_if_exists", "auto_increment"})

_APPEND_COLLISION_POLICIES = frozenset({"append_or_create"})


def _missing_output_options_repair_error(
    *,
    sink_name: str,
    plugin_name: str,
    on_write_failure: str,
    validation_error: str | None,
) -> str:
    """Return an exact output-object repair hint for omitted sink options."""
    if plugin_name == "text":
        path_fragment = _repair_identifier_fragment(sink_name, fallback="output")
        repair_output = {
            "sink_name": sink_name,
            "plugin": plugin_name,
            "options": {
                "path": f"outputs/{path_fragment}.txt",
                "schema": {"mode": "observed"},
                "field": "line_text",
                "mode": "write",
                "collision_policy": "auto_increment",
            },
            "on_write_failure": on_write_failure,
        }
        detail = f" Empty options were rejected: {validation_error}" if validation_error is not None else ""
        return (
            f"Output '{sink_name}' is missing options. For the text file sink, include path, schema, field, mode, "
            f"and collision_policy. Use this runnable output object and replace line_text with the actual selected "
            f"string field: {json.dumps(repair_output)}.{detail}"
        )

    if plugin_name in FILE_SINK_REPAIR_EXTENSIONS:
        path_fragment = _repair_identifier_fragment(sink_name, fallback="output")
        extension = FILE_SINK_REPAIR_EXTENSIONS[plugin_name]
        repair_output = {
            "sink_name": sink_name,
            "plugin": plugin_name,
            "options": {
                "path": f"outputs/{path_fragment}.{extension}",
                "schema": {"mode": "observed"},
                "collision_policy": "auto_increment",
            },
            "on_write_failure": on_write_failure,
        }
        detail = f" Empty options were rejected: {validation_error}" if validation_error is not None else ""
        return (
            f"Output '{sink_name}' is missing options. For {plugin_name} file sinks, include "
            f"an options object with path, schema, and collision_policy. Use this output object "
            f"shape and adjust the path/schema if needed: {json.dumps(repair_output)}.{detail}"
        )

    repair_output = {
        "sink_name": sink_name,
        "plugin": plugin_name,
        "options": {},
        "on_write_failure": on_write_failure,
    }
    detail = f" Empty options were rejected: {validation_error}" if validation_error is not None else ""
    return (
        f"Output '{sink_name}' is missing options. Include the sink plugin's options object. "
        f"If this sink accepts empty configuration, use: {json.dumps(repair_output)}; otherwise "
        f"call get_plugin_schema for sink '{plugin_name}' and fill the required options.{detail}"
    )


def validate_composer_file_sink_collision_policy(
    plugin_name: str,
    options: Mapping[str, Any],
    *,
    require_explicit: bool,
) -> str | None:
    """Require generated runnable file sinks to choose collision behavior."""
    if not require_explicit or plugin_name not in FILE_SINK_PLUGINS:
        return None

    if "collision_policy" not in options:
        return (
            f"File sink '{plugin_name}' must set collision_policy explicitly. "
            "Use 'fail_if_exists' to refuse a taken output path, "
            "'auto_increment' to choose a free sibling path, or "
            "'append_or_create' with mode='append'."
        )

    # mode is a safety-critical operator decision (truncate vs. append) — same
    # rationale as the collision_policy presence check above.  Mirroring
    # ``csv_sink.py:57`` / ``json_sink.py:63``'s ``Field(default="write")``
    # via ``options.get("mode", "write")`` here would silently paper over
    # every upstream null source — LLM omission, operator omission,
    # merge-patch strip, incomplete fixture — none of which is a correct
    # state for a runnable file sink at this validator's call sites.  The
    # operator-supplied options must name ``mode`` explicitly so the
    # write-vs-append branch selection below is authoritative rather than
    # inferred.  Closes I3 review finding (2026-05-24).
    if "mode" not in options:
        return (
            f"File sink '{plugin_name}' must set mode explicitly. "
            "Use 'write' to create or replace the file, or 'append' to "
            "add rows to an existing file."
        )

    mode = options["mode"]
    policy = options["collision_policy"]
    if mode == "append":
        if policy not in _APPEND_COLLISION_POLICIES:
            return f"File sink '{plugin_name}' with mode='append' must use collision_policy='append_or_create'."
    else:
        if policy not in _WRITE_COLLISION_POLICIES:
            return (
                f"File sink '{plugin_name}' with mode='write' must use "
                "collision_policy='fail_if_exists' or collision_policy='auto_increment'."
            )

    return None


def _prevalidate_source(
    plugin_name: str,
    options: Mapping[str, Any],
    on_validation_failure: str = _DEFAULT_SOURCE_VALIDATION_FAILURE,
) -> str | None:
    """Pre-validate source options, injecting on_validation_failure and filtering web-only keys."""
    filtered = _source_options_for_prevalidation(options)
    return _prevalidate_plugin_options(
        "source",
        plugin_name,
        filtered,
        injected_fields={"on_validation_failure": on_validation_failure},
    )


def _prevalidate_transform(plugin_name: str, options: Mapping[str, Any]) -> str | None:
    """Pre-validate transform options."""
    return _prevalidate_plugin_options("transform", plugin_name, strip_authoring_options(options))


def _prevalidate_sink(plugin_name: str, options: dict[str, Any]) -> str | None:
    """Pre-validate sink options."""
    return _prevalidate_plugin_options("sink", plugin_name, options)


# Type aliases shared by ``_dispatch`` and ``generation`` (and any plane that
# needs to talk about runtime-preflight callables or generic tool handlers).

RuntimePreflight = Callable[[CompositionState], ValidationResult]


@dataclass(frozen=True, slots=True)
class ToolContext:
    """Immutable per-call context threaded through every ``execute_tool``
    dispatch.

    Collapsing the previously-divergent kwarg surfaces of the six sync tool
    registries (and the three hardcoded ``if tool_name == ...`` branches for
    ``preview_pipeline`` / ``diff_pipeline`` / ``set_pipeline``) into a
    single frozen dataclass means every handler takes the same shape:
    ``(arguments, state, context) -> ToolResult``. The previous per-registry
    kwarg gymnastics is reduced to "the handler reads what it needs off
    ``context``".

    Fields:
        catalog: The catalog service the tool consults for plugin metadata.
        data_dir: Base data directory enforced for S2 path allowlist checks
            on source/sink options. ``None`` when the caller is not a web
            request (legacy direct tests).
        require_data_dir_for_paths: Fail closed when a source-local path
            option appears without ``data_dir``. Enabled for audited web/LLM
            dispatches.
        session_engine: SQLAlchemy engine for the session database. Required
            for blob tools to perform synchronous lookups; ``None`` for
            non-session callers.
        session_id: Current session ID. Required for blob tools.
        secret_service: ``WebSecretResolver`` (L0 protocol from
            ``elspeth.contracts.secrets``) — the auth-scoped resolver
            surface composer tools consult.  Production wiring passes
            ``ScopedSecretResolver`` (``elspeth.web.secrets.service``),
            which binds the deployment's ``auth_provider_type`` so the
            composer plane never has to know about it.  Required for
            secret tools (``list_secret_refs`` / ``validate_secret_ref``
            / ``wire_secret_ref``); ``None`` for non-secret-aware callers.
        user_id: Current user ID. Required for secret tools.
        baseline: Baseline state for ``diff_pipeline`` comparisons.
        current_validation: Pre-computed validation of the live state, used
            by ``diff_pipeline`` so its delta is computed against the same
            ValidationSummary the caller is already holding.
        runtime_preflight: Optional callback for runtime-equivalent
            preflight, applied only to ``preview_pipeline``. Pre-computed in
            the async compose loop and injected here as a cheap synchronous
            callback so ``execute_tool`` stays synchronous.
        max_blob_storage_per_session_bytes: Configured per-session blob
            storage quota for assistant-created session artifacts. Defaults
            to ``None`` (no override) so the blob plane can fall back to its
            historical BlobServiceImpl-compatible value for direct tests
            and non-web callers.
        user_message_id: Provenance pointer for blob writes that record
            ``created_from_message_id``. Only handlers that actually persist
            a new blob row read it.
        user_message_content: Triggering user chat-message content. Blob
            writers use this to distinguish byte-identical user-authored
            content from composer-authored content.
        composer_model_identifier: Requested composer model identifier for
            LLM-authored blob provenance.
        composer_model_version: Provider-returned model/version string when
            available, falling back to the requested model.
        composer_provider: Composer LLM provider name.
        composer_skill_hash: SHA-256 hash of the composer skill markdown used
            for the request.
        tool_arguments_hash: Canonical audited hash of the tool-call
            arguments that produced an LLM-authored blob.
    """

    catalog: PolicyCatalogView
    plugin_snapshot: PluginAvailabilitySnapshot
    data_dir: str | None = None
    require_data_dir_for_paths: bool = False
    session_engine: Engine | None = None
    session_id: str | None = None
    secret_service: WebSecretResolver | None = None
    user_id: str | None = None
    baseline: CompositionState | None = None
    current_validation: ValidationSummary | None = None
    runtime_preflight: RuntimePreflight | None = None
    max_blob_storage_per_session_bytes: int | None = None
    user_message_id: str | None = None
    user_message_content: str | None = None
    composer_model_identifier: str | None = None
    composer_model_version: str | None = None
    composer_provider: str | None = None
    composer_skill_hash: str | None = None
    tool_arguments_hash: str | None = None


ToolHandler = Callable[
    [dict[str, Any], CompositionState, ToolContext],
    ToolResult,
]
