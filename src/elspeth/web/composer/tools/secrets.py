"""Composer secrets plane — secret-reference discovery, validation, and wiring."""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Literal, TypedDict

from pydantic import BaseModel, ConfigDict
from pydantic import ValidationError as PydanticValidationError

from elspeth.contracts.freeze import deep_thaw
from elspeth.contracts.secrets import SecretInventoryItem, SecretScope, SecretUnavailabilityReason
from elspeth.web.composer.protocol import ToolArgumentError
from elspeth.web.composer.state import (
    CompositionState,
)
from elspeth.web.composer.tools._common import (
    ToolContext,
    ToolResult,
    _discovery_result,
    _failure_result,
    _mutation_result,
    _secret_ref_placement_error,
)
from elspeth.web.composer.tools.declarations import (
    ToolDeclaration,
    ToolKind,
)


# Tier-3 argument-shape models.  Secret tools are POLICY-driven in the
# redaction MANIFEST (``redaction.py:2662-2680`` — ``handles_no_sensitive_data=True``
# with a :class:`HandlesNoSensitiveDataReason` justification struct), because
# secret VALUES never traverse the composer tool surface — only refs (names +
# scopes).  These argument models therefore live module-local and are used
# ONLY for shape validation; they are intentionally NOT registered in the
# redaction MANIFEST so the declarative reason struct is preserved.
#
# Every other tier module wraps LLM-supplied ``arguments`` with
# ``Model.model_validate`` → ``except ValidationError: raise ToolArgumentError``
# (sources.py:279, blobs.py:553, outputs.py:205, sessions.py:181).  Secrets
# previously omitted this layer: presence-of-required-keys is upstream-guarded
# by ``_TOOL_REQUIRED_PATHS`` (``service.py:2516``) so a literal missing key
# does not reach the handler, but type-mismatch (``name: 42``), extra fields,
# and the ``target`` enum constraint were unenforced — type-mismatch in
# particular could crash deep in ``secret_service.has_ref(user_id, 42)`` and
# be laundered into ``ComposerPluginCrashError`` → HTTP 500 instead of a
# recoverable ARG_ERROR the LLM can act on.
class _ValidateSecretRefArgumentsModel(BaseModel):
    """Tier-3 shape contract for ``validate_secret_ref``.

    Mirrors the JSON schema at ``_VALIDATE_SECRET_REF_DECLARATION`` —
    a single ``name`` string identifying the secret reference to check.
    ``extra="forbid"`` rejects misrouted argument shapes early (e.g. an
    LLM accidentally sending ``wire_secret_ref`` args here).
    """

    name: str

    model_config = ConfigDict(extra="forbid")


class _WireSecretRefArgumentsModel(BaseModel):
    """Tier-3 shape contract for ``wire_secret_ref``.

    Mirrors the JSON schema at ``_WIRE_SECRET_REF_DECLARATION`` — three
    required strings (``name``, ``target``, ``option_key``) plus optional
    ``target_id`` for node/output targets (validated at the handler's
    semantic layer, not the shape layer — same channel discipline as
    ``set_pipeline``'s post-validation ``blob_id`` / ``inline_blob``
    mutual-exclusion check).

    ``target`` is ``Literal[...]`` to lift the enum constraint into the
    Pydantic layer; the runtime ``else: Unknown target type`` branch in
    :func:`_execute_wire_secret_ref` becomes belt-and-suspenders (only
    reachable if Pydantic validation is bypassed) but is retained for
    defense-in-depth.
    """

    name: str
    target: Literal["source", "node", "output"]
    target_id: str | None = None
    option_key: str

    model_config = ConfigDict(extra="forbid")


class _SecretInventoryItemPayload(TypedDict):
    name: str
    scope: SecretScope
    available: bool
    source_kind: str
    reason: SecretUnavailabilityReason | None


def _inventory_item_payload(item: SecretInventoryItem) -> _SecretInventoryItemPayload:
    return {
        "name": item.name,
        "scope": item.scope,
        "available": item.available,
        "source_kind": item.source_kind,
        "reason": item.reason,
    }


def _live_validation_failure_reason(item: SecretInventoryItem) -> SecretUnavailabilityReason:
    if item.reason is not None:
        return item.reason
    if item.source_kind == "env" or item.scope == "server":
        return "env_var_not_set"
    return "value_decryption_failed"


def _handle_list_secret_refs(
    arguments: dict[str, Any],
    state: CompositionState,
    context: ToolContext,
) -> ToolResult:
    if context.secret_service is None or context.user_id is None:
        return _failure_result(state, "Secret tools require secret service context.")
    items = context.secret_service.list_refs(context.user_id)
    # Return inventory dicts — NEVER include values
    data = [_inventory_item_payload(item) for item in items]
    return _discovery_result(state, data)


_LIST_SECRET_REFS_DECLARATION = ToolDeclaration(
    name="list_secret_refs",
    handler=_handle_list_secret_refs,
    kind=ToolKind.SECRET_DISCOVERY,
    description="List available secret references (API keys, credentials). Shows names and scopes, never values.",
    json_schema={"type": "object", "properties": {}, "required": []},
)


def _handle_validate_secret_ref(
    arguments: dict[str, Any],
    state: CompositionState,
    context: ToolContext,
) -> ToolResult:
    if context.secret_service is None or context.user_id is None:
        return _failure_result(state, "Secret tools require secret service context.")
    try:
        validated = _ValidateSecretRefArgumentsModel.model_validate(arguments)
    except PydanticValidationError as exc:
        raise ToolArgumentError(
            argument="validate_secret_ref arguments",
            expected="object conforming to _ValidateSecretRefArgumentsModel",
            actual_type=type(exc).__name__,
        ) from exc
    name = validated.name
    matching_item: SecretInventoryItem | None = None
    for item in context.secret_service.list_refs(context.user_id):
        if item.name == name:
            matching_item = item
            break

    available = context.secret_service.has_ref(context.user_id, name)
    if matching_item is not None:
        payload = _inventory_item_payload(matching_item)
        payload["available"] = available
        payload["reason"] = None if available else _live_validation_failure_reason(matching_item)
        return _discovery_result(state, payload)
    return _discovery_result(state, {"name": name, "available": available})


_VALIDATE_SECRET_REF_DECLARATION = ToolDeclaration(
    name="validate_secret_ref",
    handler=_handle_validate_secret_ref,
    kind=ToolKind.SECRET_DISCOVERY,
    description="Check if a secret reference exists and is accessible to the current user.",
    json_schema={
        "type": "object",
        "properties": {
            "name": {"type": "string", "description": "Secret reference name (e.g. 'OPENROUTER_API_KEY')."},
        },
        "required": ["name"],
    },
)


def _execute_wire_secret_ref(
    arguments: dict[str, Any],
    state: CompositionState,
    context: ToolContext,
) -> ToolResult:
    if context.secret_service is None or context.user_id is None:
        return _failure_result(state, "Secret tools require secret service context.")

    try:
        validated = _WireSecretRefArgumentsModel.model_validate(arguments)
    except PydanticValidationError as exc:
        raise ToolArgumentError(
            argument="wire_secret_ref arguments",
            expected="object conforming to _WireSecretRefArgumentsModel",
            actual_type=type(exc).__name__,
        ) from exc

    name = validated.name
    target = validated.target
    option_key = validated.option_key
    target_id = validated.target_id

    # Validate the secret ref exists
    if not context.secret_service.has_ref(context.user_id, name):
        return _failure_result(state, f"Secret reference '{name}' not found or not accessible.")

    marker = {"secret_ref": name}

    if target == "source":
        if not state.sources:
            return _failure_result(state, "No source configured — set a source first.")
        if target_id is None:
            if len(state.sources) == 1:
                source_name = next(iter(state.sources))
            else:
                return _failure_result(state, "target_id is required for source targets when multiple sources are configured.")
        else:
            source_name = str(target_id)
        source = state.sources[source_name] if source_name in state.sources else None
        if source is None:
            return _failure_result(state, f"Source '{source_name}' not found.")
        patched_options = dict(deep_thaw(source.options))
        patched_options[option_key] = marker
        placement_error = _secret_ref_placement_error("source", source.plugin, patched_options)
        if placement_error is not None:
            return _failure_result(state, placement_error)
        new_source = replace(source, options=patched_options)
        new_state = state.with_named_source(source_name, new_source)
        affected = "source" if source_name == "source" else f"source:{source_name}"
        return _mutation_result(new_state, (affected,))

    elif target == "node":
        if target_id is None:
            return _failure_result(state, "target_id is required for node targets.")
        node = next((n for n in state.nodes if n.id == target_id), None)
        if node is None:
            return _failure_result(state, f"Node '{target_id}' not found.")
        if node.node_type not in ("transform", "aggregation") or node.plugin is None:
            return _failure_result(
                state,
                "Secret references can only be wired into source, transform, aggregation, or output plugin options.",
            )
        patched_options = dict(deep_thaw(node.options))
        patched_options[option_key] = marker
        placement_error = _secret_ref_placement_error("transform", node.plugin, patched_options)
        if placement_error is not None:
            return _failure_result(state, placement_error)
        new_node = replace(node, options=patched_options)
        new_state = state.with_node(new_node)
        return _mutation_result(new_state, (target_id,))

    else:
        # ``target == "output"`` — Pydantic ``Literal["source", "node", "output"]``
        # guarantees this is the only remaining variant; explicit
        # ``elif`` plus an "unknown target" else was dead code post-validation.
        if target_id is None:
            return _failure_result(state, "target_id is required for output targets.")
        output = next((o for o in state.outputs if o.name == target_id), None)
        if output is None:
            return _failure_result(state, f"Output '{target_id}' not found.")
        patched_options = dict(deep_thaw(output.options))
        patched_options[option_key] = marker
        placement_error = _secret_ref_placement_error("sink", output.plugin, patched_options)
        if placement_error is not None:
            return _failure_result(state, placement_error)
        new_output = replace(output, options=patched_options)
        new_state = state.with_output(new_output)
        return _mutation_result(new_state, (target_id,))


_WIRE_SECRET_REF_DECLARATION = ToolDeclaration(
    name="wire_secret_ref",
    handler=_execute_wire_secret_ref,
    kind=ToolKind.SECRET_MUTATION,
    description="Place a secret reference marker in the pipeline config. The secret will be resolved at execution time.",
    json_schema={
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
)


TOOLS_IN_MODULE: tuple[ToolDeclaration, ...] = (
    _LIST_SECRET_REFS_DECLARATION,
    _VALIDATE_SECRET_REF_DECLARATION,
    _WIRE_SECRET_REF_DECLARATION,
)
"""Every tool declared in this module, in stable order.

``_dispatch.py`` aggregates this tuple alongside every other plane's
TOOLS_IN_MODULE to build the registered-tool universe."""
