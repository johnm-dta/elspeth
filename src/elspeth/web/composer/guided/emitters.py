"""Turn-emitter helpers for guided-mode endpoints.

These helpers build Turn payloads deterministically from CompositionState and
optional blob-inspection facts.  They are pure functions — no I/O, no clock,
no uuid — so the caller (route handler) can hash the result, persist it as a
TurnRecord, and emit the corresponding audit event independently.

Exported:
    build_initial_step_1_turn — build the initial Step 1 turn payload.
    build_step_1_inspect_and_confirm_turn_from_intent — inspect_and_confirm from SourceIntent.
    build_step_1_schema_form_turn — schema_form for a chosen source plugin.
    build_step_2_single_select_turn — single_select for sink plugin selection.
    build_step_2_schema_form_turn — schema_form for a chosen sink plugin.
    build_step_2_multi_select_turn — multi_select_with_custom for required fields.
    build_step_4_wire_turn — confirm_wiring turn with topology + validation two-read merge.

Trust tier: L3 web layer.  No upward imports.  Payloads are Tier 2 pipeline
data constructed from system-owned state; the Turn dict itself is not persisted
— only its hash (via stable_hash) enters the audit trail.
"""

from __future__ import annotations

import keyword
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any, Literal, cast

from elspeth.contracts.freeze import deep_thaw
from elspeth.contracts.schema import FieldDefinition
from elspeth.web.catalog.knob_schema import KnobSchema
from elspeth.web.composer._producer_resolver import source_producer_id
from elspeth.web.composer.guided._display import plugin_display_label
from elspeth.web.composer.guided.errors import InvariantError
from elspeth.web.composer.guided.protocol import (
    BLOB_REF_PATH_PREFIX,
    GuidedStep,
    InspectAndConfirmPayload,
    MultiSelectWithCustomPayload,
    SchemaFormPayload,
    SingleSelectPayload,
    Turn,
    TurnType,
    WireStageData,
    _Observed,
    _Option,
    _WireBusinessSchema,
    _WireConnectionReview,
    _WireNodeReview,
    _WireOutputReview,
    _WireProjection,
    _WireRowCardinality,
    _WireSchemaField,
    _WireSourceReview,
    _WireStructuredOutputField,
)
from elspeth.web.composer.tools._common import _semantic_contracts_payload

if TYPE_CHECKING:
    from elspeth.web.catalog.protocol import CatalogService as CatalogServiceProtocol
    from elspeth.web.composer.guided.protocol import ProposePipelinePayload
    from elspeth.web.composer.guided.resolved import SinkOutputResolved, SinkResolved, SourceResolved
    from elspeth.web.composer.guided.state_machine import GuidedSession, SourceIntent
    from elspeth.web.composer.source_inspection import SourceInspectionFacts
    from elspeth.web.composer.state import CompositionState, ValidationSummary


def build_initial_step_1_turn(
    state: CompositionState,
    *,
    blob_inspection: SourceInspectionFacts | None,
    catalog: CatalogServiceProtocol,
) -> Turn:
    """Build the initial Step 1 turn for a fresh guided session.

    Decision rule:
    - If ``blob_inspection`` is not None and carries header information →
      emit ``inspect_and_confirm`` so the user can confirm the observed schema
      before proceeding.
    - Otherwise → emit ``single_select`` listing all registered source plugins
      so the user can pick one explicitly.

    Args:
        state: Current CompositionState.  Used for ``step_index`` derivation
            (version is not used — ``GuidedStep.STEP_1_SOURCE`` is the index).
        blob_inspection: Result of ``inspect_blob_content`` on an uploaded blob,
            or ``None`` when no blob is attached to this session.
        catalog: Plugin catalog for listing registered source plugins.

    Returns:
        A ``Turn`` TypedDict ready for serialisation and hash.
    """
    if blob_inspection is not None and blob_inspection.observed_headers is not None:
        return _build_inspect_and_confirm_turn(blob_inspection)
    return _build_step_1_single_select_turn(catalog)


def build_step_1_inspect_and_confirm_turn_from_intent(
    intent: SourceIntent,
) -> Turn:
    """Build an ``inspect_and_confirm`` Turn from a persisted SourceIntent.

    Schema-8 inspection-review intents retain the authoritative
    ``SourceInspectionFacts`` used for the original turn. Reuse those facts so
    columns and warnings rebuild byte-for-byte from persisted custody instead
    of maintaining a second warning field on the intent.

    Args:
        intent: A staged schema-8 SourceIntent in ``inspection_review`` phase.

    Returns:
        A ``Turn`` TypedDict ready for serialisation and hash.
    """
    if intent.inspection_facts is None:
        raise InvariantError("inspection-review source intent requires persisted inspection facts")
    return _build_inspect_and_confirm_turn(intent.inspection_facts)


def build_step_1_schema_form_turn(
    plugin: str,
    catalog: CatalogServiceProtocol,
    *,
    inspection_facts: SourceInspectionFacts | None = None,
) -> Turn:
    """Build a ``schema_form`` Turn for the chosen source plugin.

    ``inspection_facts`` (when present) is merged into ``prefilled`` using only
    facts carried by the current SourceInspectionFacts model. When absent,
    ``prefilled`` falls back to ``{"schema": {"mode": "observed"}}``.

    Args:
        plugin: The plugin name chosen by the user (e.g. ``"csv"``).
        catalog: Plugin catalog for retrieving the plugin's knob schema.

    Returns:
        A ``Turn`` TypedDict ready for serialisation and hash.
    """
    schema_info = catalog.get_schema("source", plugin)
    prefilled = build_step_1_source_prefill(plugin, inspection_facts=inspection_facts)
    payload: SchemaFormPayload = {
        "mode": "plugin_options",
        "plugin": plugin,
        "knobs": cast(KnobSchema, schema_info.knob_schema),
        "prefilled": prefilled,
    }
    return Turn(
        type=TurnType.SCHEMA_FORM.value,
        step_index=_step_index(GuidedStep.STEP_1_SOURCE),
        payload=payload,
    )


def build_step_1_source_prefill(
    plugin: str,
    *,
    inspection_facts: SourceInspectionFacts | None = None,
) -> dict[str, object]:
    """Build source schema-form defaults from the selected plugin and blob facts."""
    prefilled: dict[str, object] = {"schema": {"mode": "observed"}}
    if inspection_facts is not None:
        _merge_inspection_into_prefill(prefilled, plugin=plugin, facts=inspection_facts)
    return prefilled


def _merge_inspection_into_prefill(
    prefilled: dict[str, object],
    *,
    plugin: str,
    facts: SourceInspectionFacts,
) -> None:
    """Conservatively prefill source schema from inspection facts."""
    blob_id = facts.redacted_identity.get("blob_id")
    if blob_id and _inspection_matches_source_plugin(plugin, facts):
        prefilled["path"] = f"{BLOB_REF_PATH_PREFIX}{blob_id}"
        prefilled["on_validation_failure"] = "discard"
    if facts.observed_headers and facts.inferred_types:
        fields = _schema_field_specs_from_inspection_headers(facts)
        if fields is not None:
            prefilled["schema"] = {"mode": "flexible", "fields": fields}
        else:
            prefilled["schema"] = {"mode": "observed"}
    elif facts.observed_headers:
        prefilled["schema"] = {"mode": "observed"}
    # Delimiter and encoding are deliberately not prefilled here: the live
    # SourceInspectionFacts model does not carry those fields yet.


def _inspection_matches_source_plugin(plugin: str, facts: SourceInspectionFacts) -> bool:
    """Return whether an uploaded blob inspection can safely prefill this source plugin."""
    if facts.source_kind == "csv":
        return plugin == "csv"
    if facts.source_kind in ("json", "jsonl"):
        return plugin == "json"
    if facts.source_kind == "text":
        return plugin == "text"
    return False


def _schema_field_specs_from_inspection_headers(facts: SourceInspectionFacts) -> list[str] | None:
    """Return safe explicit schema field specs, or ``None`` to stay observed.

    Blob inspection facts originate from Tier-3 uploaded bytes. Explicit
    ``schema.fields`` are later loaded through the runtime YAML settings loader,
    whose string values support ``${VAR}`` expansion on the operator CLI path.
    Do not place raw external headers into those strings unless the header is
    already a safe schema field identifier; otherwise the source runtime's
    mandatory header normalization remains the authoritative boundary. Safe
    names must already match the CSV source normalization convention (lowercase
    Python identifiers) so the prefill does not declare fields that runtime
    header normalization would rename.
    """
    inferred_types = facts.inferred_types
    if inferred_types is None:
        return None
    fields: list[str] = []
    for header in facts.observed_headers or ():
        if not _is_safe_schema_field_name(header):
            return None
        inferred = inferred_types[header]
        field_type = "any" if inferred == "null" else inferred
        fields.append(f"{header}: {field_type}")
    return fields


def _is_safe_schema_field_name(header: str) -> bool:
    """Return whether ``header`` can safely appear in ``schema.fields``."""
    return header.isidentifier() and header == header.lower() and not keyword.iskeyword(header)


def build_step_2_single_select_turn(
    catalog: CatalogServiceProtocol,
) -> Turn:
    """Build a ``single_select`` Turn listing available sink plugins.

    Emitted when the wizard enters Step 2 (the user has committed their
    source choice in Step 1).

    Args:
        catalog: Plugin catalog for listing registered sink plugins.

    Returns:
        A ``Turn`` TypedDict ready for serialisation and hash.
    """
    sinks = catalog.list_sinks()
    options: list[_Option] = [
        _Option(
            id=plugin.name,
            # Human display label; the option id (the submitted VALUE) stays
            # the raw plugin id (elspeth-5ee1f76e39 backend half).
            label=plugin_display_label(plugin.name),
            hint=plugin.description if plugin.description else None,
        )
        for plugin in sinks
    ]
    payload: SingleSelectPayload = {
        "question": "What format should the output be in?",
        "options": options,
        "allow_custom": False,
    }
    return Turn(
        type=TurnType.SINGLE_SELECT.value,
        step_index=_step_index(GuidedStep.STEP_2_SINK),
        payload=payload,
    )


def build_step_2_schema_form_turn(
    plugin: str,
    catalog: CatalogServiceProtocol,
    *,
    prefilled_options: Mapping[str, Any] | None = None,
) -> Turn:
    """Build a ``schema_form`` Turn for the chosen sink plugin.

    Emitted after the user picks a sink plugin in Step 2's ``single_select``
    turn.  The schema block is the plugin's full JSON schema; ``prefilled``
    seeds ``schema.mode: "observed"`` and merges any staged chat-resolution
    options (``SinkIntent.options``) so a chat-resolved sink renders as a
    live, continuable form instead of a bare one.

    Args:
        plugin: The plugin name chosen by the user (e.g. ``"json"``).
        catalog: Plugin catalog for retrieving the plugin's JSON schema.
        prefilled_options: Staged sink options from a chat resolution.

    Returns:
        A ``Turn`` TypedDict ready for serialisation and hash.
    """
    schema_info = catalog.get_schema("sink", plugin)
    prefilled: dict[str, Any] = {"schema": {"mode": "observed"}}
    if prefilled_options is not None:
        prefilled.update(deep_thaw(prefilled_options))
    payload: SchemaFormPayload = {
        "mode": "plugin_options",
        "plugin": plugin,
        "knobs": cast(KnobSchema, schema_info.knob_schema),
        "prefilled": prefilled,
    }
    return Turn(
        type=TurnType.SCHEMA_FORM.value,
        step_index=_step_index(GuidedStep.STEP_2_SINK),
        payload=payload,
    )


def build_component_review_turn(
    guided: GuidedSession,
    component_kind: Literal["source", "output"],
) -> Turn:
    """Build the public plural-component controller from reviewed custody."""

    expected_step = GuidedStep.STEP_1_SOURCE if component_kind == "source" else GuidedStep.STEP_2_SINK
    if guided.step is not expected_step:
        raise InvariantError(f"{component_kind} component review is not legal at {guided.step.value}")
    reviewed: Mapping[str, SourceResolved | SinkOutputResolved]
    if component_kind == "source":
        order = guided.source_order
        reviewed = guided.reviewed_sources
        if guided.pending_source_intents:
            raise InvariantError("source component review requires no pending source intents")
    else:
        order = guided.output_order
        reviewed = guided.reviewed_outputs
        if guided.pending_output_intents:
            raise InvariantError("output component review requires no pending output intents")
    if not reviewed or set(order) != set(reviewed):
        raise InvariantError(f"{component_kind} component review requires a complete reviewed collection")
    actions = ["add", "edit"]
    if len(order) > 1:
        actions.append("remove")
    actions.extend(("reorder", "finish"))
    return Turn(
        type=TurnType.REVIEW_COMPONENTS.value,
        step_index=_step_index(expected_step),
        payload={
            "component_kind": component_kind,
            "items": [
                {
                    "stable_id": stable_id,
                    "name": reviewed[stable_id].name,
                    "plugin": reviewed[stable_id].plugin,
                    "status": "reviewed",
                }
                for stable_id in order
            ],
            "allowed_actions": actions,
        },
    )


def build_step_1_schema_form_turn_from_resolved(
    source: SourceResolved,
    catalog: CatalogServiceProtocol,
) -> Turn:
    """Build the STEP_1 ``schema_form`` populated from an APPLIED source.

    Unlike :func:`build_step_1_schema_form_turn` (which seeds an empty
    ``prefilled``), this renders the committed ``source.options`` so the
    editable form shows what the LLM (or the manual path) built. Used by the
    chat-apply in-place re-render and by GET /guided when editing a reviewed
    source.
    """
    from elspeth.contracts.freeze import deep_thaw

    schema_info = catalog.get_schema("source", source.plugin)
    prefilled: dict[str, Any] = {"schema": {"mode": "observed"}, **dict(deep_thaw(source.options))}
    # Mask a blob-backed source's absolute storage_path behind a stable
    # ``blob:<ref>`` sentinel so the deploy dir + OS username never reach the wire
    # (the un-gated egress that bypasses the blobs/schemas.py "storage_path is
    # never exposed" doctrine). The proposal custody boundary re-resolves the
    # sentinel before commit. Gated on blob_ref so an operator-typed path knob
    # is left untouched.
    blob_ref = source.options.get("blob_ref")
    if blob_ref is not None and isinstance(prefilled.get("path"), str):
        prefilled["path"] = f"{BLOB_REF_PATH_PREFIX}{blob_ref}"
    # on_validation_failure is a required-no-default source-node knob, so the
    # schema_form's Continue stays disabled until it has a value. Seed it from the
    # resolved node field (assigned AFTER the options spread so the authoritative
    # node value wins over any stray same-named key in options).
    prefilled["on_validation_failure"] = source.on_validation_failure
    payload: SchemaFormPayload = {
        "mode": "plugin_options",
        "plugin": source.plugin,
        "knobs": cast(KnobSchema, schema_info.knob_schema),
        "prefilled": prefilled,
    }
    return Turn(
        type=TurnType.SCHEMA_FORM.value,
        step_index=_step_index(GuidedStep.STEP_1_SOURCE),
        payload=payload,
    )


def build_step_2_schema_form_turn_from_resolved(
    sink: SinkResolved,
    catalog: CatalogServiceProtocol,
) -> Turn:
    """Build the STEP_2 ``schema_form`` populated from an APPLIED sink.

    Renders the first reviewed output's committed ``options``. Used by the
    chat-apply in-place re-render and by GET /guided when editing a reviewed
    output.
    """
    from elspeth.contracts.freeze import deep_thaw

    if not sink.outputs:
        raise InvariantError("build_step_2_schema_form_turn_from_resolved: sink has no outputs")
    output = sink.outputs[0]
    schema_info = catalog.get_schema("sink", output.plugin)
    # deep_thaw: a rehydrated SinkResolved (GET /guided after apply) carries
    # deep-frozen ``mappingproxy`` options whose NESTED maps (e.g. ``schema``)
    # Pydantic rejects on serialisation. Mirror the step_1 builder's thaw.
    prefilled: dict[str, Any] = {"schema": {"mode": "observed"}, **dict(deep_thaw(output.options))}
    prefilled["on_write_failure"] = output.on_write_failure
    payload: SchemaFormPayload = {
        "mode": "plugin_options",
        "plugin": output.plugin,
        "knobs": cast(KnobSchema, schema_info.knob_schema),
        "prefilled": prefilled,
    }
    return Turn(
        type=TurnType.SCHEMA_FORM.value,
        step_index=_step_index(GuidedStep.STEP_2_SINK),
        payload=payload,
    )


def build_step_2_multi_select_turn(
    observed_columns: Sequence[str],
) -> Turn:
    """Build a ``multi_select_with_custom`` Turn for declaring required fields.

    Emitted after the user fills in sink options (Step 2 ``schema_form``).
    The options are pre-populated from Step 1's observed columns; the user
    ticks which fields must appear in the output, adds custom fields, or
    clicks the escape label to let the source decide.

    ``escape_label`` wire contract (elspeth-948eb9c0b8 C-3(a)): clicking the
    escape action MUST submit ``control_signal: "passthrough"`` (see
    ``ControlSignal.PASSTHROUGH``) alongside ``chosen: []`` and
    ``custom_inputs: []``. A bare empty ``chosen``/``custom_inputs`` pair
    *without* the signal is fail-closed rejected with a structured 400
    (``code: "guided_step2_no_fields_selected"``) — it is indistinguishable
    on the wire from a stale/buggy client submitting nothing, so the current
    STEP_2_SINK field-review transition refuses to guess. Sending
    ``control_signal: "passthrough"`` together with a
    non-empty ``chosen``/``custom_inputs`` is likewise rejected (400,
    ``code: "guided_step2_passthrough_conflict"``) as a contradictory
    payload.

    Args:
        observed_columns: The columns observed from the source in Step 1.
            Comes from the reviewed source's observed columns.

    Returns:
        A ``Turn`` TypedDict ready for serialisation and hash.
    """
    options: list[_Option] = [_Option(id=col, label=col, hint=None) for col in observed_columns]
    payload: MultiSelectWithCustomPayload = {
        "question": "Which fields must appear in the output?",
        "options": options,
        "default_chosen": list(observed_columns),
        "escape_label": "Let source decide (pass all fields through)",
    }
    return Turn(
        type=TurnType.MULTI_SELECT_WITH_CUSTOM.value,
        step_index=_step_index(GuidedStep.STEP_2_SINK),
        payload=payload,
    )


def build_step_4_wire_turn(
    state: CompositionState,
    *,
    proposal_projection: ProposePipelinePayload,
    guided: GuidedSession,
    catalog: CatalogServiceProtocol | None = None,
    validation_state: CompositionState | None = None,
    validation_summary: ValidationSummary | None = None,
) -> Turn:
    """Build one stable-ID wire review from an immutable proposal candidate."""
    del catalog  # Reserved for future catalog-backed presentation enrichment.
    validation = validation_summary or (validation_state or state).validate()
    executable_state = validation_state or state
    projected = _build_wire_projection(
        state,
        executable_state=executable_state,
        proposal_projection=proposal_projection,
        guided=guided,
        validation=validation,
    )
    payload: WireStageData = {
        "proposal_id": proposal_projection["proposal_id"],
        "draft_hash": proposal_projection["draft_hash"],
        "sources": projected["sources"],
        "nodes": projected["nodes"],
        "outputs": projected["outputs"],
        "connections": projected["connections"],
        "semantic_contracts": _semantic_contracts_payload(validation.semantic_contracts),
        "warnings": [w.to_dict() for w in validation.warnings],
        "blockers": [error.to_dict() for error in validation.errors],
        "can_confirm": validation.is_valid,
    }
    return Turn(
        type=TurnType.CONFIRM_WIRING.value,
        step_index=_step_index(GuidedStep.STEP_4_WIRE),
        payload=payload,
    )


def _wire_schema(options: Mapping[str, Any]) -> _WireBusinessSchema:
    """Project only the business schema, never adjacent path/secret options."""

    raw = options.get("schema", options.get("schema_config", {}))
    schema = raw if isinstance(raw, Mapping) else {}
    fields: list[_WireSchemaField] = []
    raw_fields = schema.get("fields")
    if isinstance(raw_fields, Sequence) and not isinstance(raw_fields, str | bytes):
        for field in raw_fields:
            if not isinstance(field, str | Mapping):
                continue
            try:
                parsed = FieldDefinition.parse(field)
            except ValueError:
                continue
            fields.append(cast(_WireSchemaField, parsed.to_dict()))

    def names(key: str) -> list[str]:
        value = schema.get(key)
        if not isinstance(value, Sequence) or isinstance(value, str | bytes):
            return []
        return [item for item in value if type(item) is str]

    mode = schema.get("mode")
    return {
        "mode": mode if type(mode) is str else "observed",
        "fields": fields,
        "guaranteed_fields": names("guaranteed_fields"),
        "required_fields": names("required_fields"),
    }


def _structured_output_fields(options: Mapping[str, Any]) -> list[_WireStructuredOutputField]:
    """Return typed LLM result fields without prompts, templates, or values."""

    queries = options.get("queries")
    if isinstance(queries, Mapping):
        entries = [(name, value) for name, value in queries.items() if type(name) is str and isinstance(value, Mapping)]
    elif isinstance(queries, Sequence) and not isinstance(queries, str | bytes):
        entries = [(item["name"], item) for item in queries if isinstance(item, Mapping) and type(item.get("name")) is str]
    else:
        return []
    projected: list[_WireStructuredOutputField] = []
    for query_name, query in entries:
        fields = query.get("output_fields")
        if not isinstance(fields, Sequence) or isinstance(fields, str | bytes):
            continue
        for field in fields:
            if not isinstance(field, Mapping) or type(field.get("suffix")) is not str or type(field.get("type")) is not str:
                continue
            values = field.get("values")
            projected.append(
                {
                    "query": query_name,
                    "field": f"{query_name}_{field['suffix']}",
                    "type": field["type"],
                    "enum_values": (
                        [item for item in values if type(item) is str]
                        if isinstance(values, Sequence) and not isinstance(values, str | bytes)
                        else []
                    ),
                }
            )
    return projected


def _node_cardinality(node: Any, executable_node: Any) -> _WireRowCardinality:
    if node.node_type == "aggregation":
        if node.expected_output_count is not None:
            return {"input": "batch", "output": "expected_count", "expected_output_count": str(node.expected_output_count)}
        return {"input": "batch", "output": "zero_or_many", "expected_output_count": None}
    if node.node_type == "coalesce":
        return {"input": "branches", "output": "one_per_branch_set", "expected_output_count": None}
    if node.node_type == "queue":
        return {"input": "many_producers", "output": "one_per_item", "expected_output_count": None}
    if node.node_type == "gate":
        return {"input": "one", "output": "one", "expected_output_count": None}
    if executable_node.plugin is None:
        raise InvariantError("wire projection transform lost its plugin")
    from elspeth.plugins.infrastructure.manager import get_shared_plugin_manager
    from elspeth.web.composer._validation_probe import prepare_validation_probe_options

    transform = get_shared_plugin_manager().create_transform(
        executable_node.plugin,
        prepare_validation_probe_options(executable_node.options),
    )
    try:
        output: Literal["one", "zero_or_one", "zero_or_many"]
        if transform.creates_tokens:
            output = "zero_or_many"
        elif transform.can_drop_rows:
            output = "zero_or_one"
        else:
            output = "one"
        return {"input": "one", "output": output, "expected_output_count": None}
    finally:
        transform.close()


def _build_wire_projection(
    state: CompositionState,
    *,
    executable_state: CompositionState,
    proposal_projection: ProposePipelinePayload,
    guided: GuidedSession,
    validation: ValidationSummary,
) -> _WireProjection:
    """Bind candidate semantics to the proposal's already-advertised IDs."""

    public_sources = list(proposal_projection["graph"]["sources"])
    public_nodes = list(proposal_projection["nodes"])
    public_outputs = list(proposal_projection["outputs"])
    if len(public_sources) != len(state.sources) or len(public_nodes) != len(state.nodes) or len(public_outputs) != len(state.outputs):
        raise InvariantError("wire projection component counts differ from the proposal")
    source_canonical = {public_sources[index]["stable_id"]: source_producer_id(name) for index, name in enumerate(state.sources)}
    node_canonical = {public_nodes[index]["stable_id"]: node.id for index, node in enumerate(state.nodes)}
    output_canonical = {public_outputs[index]["stable_id"]: f"output:{output.name}" for index, output in enumerate(state.outputs)}
    canonical_by_stable = {**source_canonical, **node_canonical, **output_canonical}
    contracts = {(item.from_id, item.to_id): item.to_dict() for item in validation.edge_contracts}

    def fields_for(stable_id: str, *, produced: bool) -> list[str]:
        canonical = canonical_by_stable[stable_id]
        values: set[str] = set()
        for (from_id, to_id), contract in contracts.items():
            if produced and from_id == canonical:
                values.update(contract["producer_guarantees"])
            if not produced and to_id == canonical:
                values.update(contract["consumer_requires"])
        return sorted(values)

    sources: list[_WireSourceReview] = [
        {
            "stable_id": public["stable_id"],
            "label": public["label"],
            "plugin": source.plugin,
            "on_validation_failure": source.on_validation_failure,
            "guaranteed_fields": fields_for(public["stable_id"], produced=True),
            "row_cardinality": {"input": "none", "output": "zero_or_many", "expected_output_count": None},
        }
        for public, source in zip(public_sources, state.sources.values(), strict=True)
    ]
    executable_nodes = {node.id: node for node in executable_state.nodes}
    nodes: list[_WireNodeReview] = [
        {
            "stable_id": public["stable_id"],
            "label": public["label"],
            "node_type": node.node_type,
            "plugin": node.plugin,
            "behavior": public["behavior"],
            "required_fields": fields_for(public["stable_id"], produced=False),
            "guaranteed_fields": fields_for(public["stable_id"], produced=True),
            "row_cardinality": _node_cardinality(node, executable_nodes[node.id]),
            "structured_output_fields": _structured_output_fields(node.options) if node.plugin == "llm" else [],
        }
        for public, node in zip(public_nodes, state.nodes, strict=True)
    ]
    outputs: list[_WireOutputReview] = [
        {
            "stable_id": public["stable_id"],
            "label": public["label"],
            "plugin": output.plugin,
            "on_write_failure": output.on_write_failure,
            "required_fields": list(guided.reviewed_outputs[public["stable_id"]].required_fields),
            "business_schema": _wire_schema(output.options),
        }
        for public, output in zip(public_outputs, state.outputs, strict=True)
    ]
    connections: list[_WireConnectionReview] = []
    for edge in proposal_projection["graph"]["edges"]:
        from_id = canonical_by_stable[edge["from_endpoint"]["stable_id"]]
        destination = edge["to_endpoint"]
        to_id = canonical_by_stable[destination["stable_id"]] if destination["kind"] != "discard" else None
        connections.append(
            {
                "stable_id": edge["stable_id"],
                "from_endpoint": edge["from_endpoint"],
                "to_endpoint": destination,
                "flow": edge["flow"],
                "schema_contract": contracts.get((from_id, to_id)) if to_id is not None else None,
            }
        )
    return {"sources": sources, "nodes": nodes, "outputs": outputs, "connections": connections}


def _build_inspect_and_confirm_turn(
    inspection: SourceInspectionFacts,
) -> Turn:
    """Build an ``inspect_and_confirm`` Turn from blob inspection facts.

    ``samples`` is intentionally empty: ``SourceInspectionFacts`` carries
    ``sample_row_count`` (a count) but not the actual row payloads — the
    inspection layer deliberately redacts row content to avoid passing
    user data through the audit trail.  The UI renders the count + column
    list; full row previews are handled by the blob preview endpoint.
    """
    observed: _Observed = {
        "columns": list(inspection.observed_headers or ()),
        "samples": [],
        "warnings": list(inspection.warnings),
    }
    payload: InspectAndConfirmPayload = {"observed": observed}
    return Turn(
        type=TurnType.INSPECT_AND_CONFIRM.value,
        step_index=_step_index(GuidedStep.STEP_1_SOURCE),
        payload=payload,
    )


# Degenerate sources hidden from the guided discovery picker. ``null`` yields no
# rows — it is never a sensible pipeline INPUT to pick first-hand, and surfacing
# it in the (especially first-run) source list is pure noise. It remains fully
# usable via explicit YAML / freeform composition.
_GUIDED_HIDDEN_SOURCES = frozenset({"null"})


def _build_step_1_single_select_turn(
    catalog: CatalogServiceProtocol,
) -> Turn:
    """Build a ``single_select`` Turn listing selectable source plugins.

    Excludes the degenerate sources in ``_GUIDED_HIDDEN_SOURCES`` (see note).
    """
    sources = catalog.list_sources()
    options: list[_Option] = [
        _Option(
            id=plugin.name,
            # Human display label; the option id (the submitted VALUE) stays
            # the raw plugin id (elspeth-5ee1f76e39 backend half).
            label=plugin_display_label(plugin.name),
            hint=plugin.description if plugin.description else None,
        )
        for plugin in sources
        if plugin.name not in _GUIDED_HIDDEN_SOURCES
    ]
    payload: SingleSelectPayload = {
        "question": "Which data source would you like to use?",
        "options": options,
        "allow_custom": False,
    }
    return Turn(
        type=TurnType.SINGLE_SELECT.value,
        step_index=_step_index(GuidedStep.STEP_1_SOURCE),
        payload=payload,
    )


def _step_index(step: GuidedStep) -> int:
    """Map GuidedStep to its 0-based integer index.

    ``Turn.step_index`` is ``int`` (spec §4), not the StrEnum value.
    The mapping is the canonical ordered position in the wizard sequence.
    """
    _ORDER: tuple[GuidedStep, ...] = (
        GuidedStep.STEP_1_SOURCE,
        GuidedStep.STEP_2_SINK,
        GuidedStep.STEP_3_TRANSFORMS,
        GuidedStep.STEP_4_WIRE,
    )
    return _ORDER.index(step)
