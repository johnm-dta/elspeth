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
    build_step_2_5_recipe_offer_turn — recipe_offer from a RecipeMatch.
    build_step_3_propose_chain_turn — propose_chain from a ChainProposal.
    build_step_3_schema_form_turn — schema_form for editing one proposed transform.

Trust tier: L3 web layer.  No upward imports.  Payloads are Tier 2 pipeline
data constructed from system-owned state; the Turn dict itself is not persisted
— only its hash (via stable_hash) enters the audit trail.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any, cast

from elspeth.web.catalog.knob_schema import KnobSchema, lower_slot_specs_to_knob_schema
from elspeth.web.composer.guided.protocol import (
    GuidedStep,
    InspectAndConfirmPayload,
    MultiSelectWithCustomPayload,
    ProposeChainPayload,
    SchemaFormPayload,
    SingleSelectPayload,
    Turn,
    TurnType,
    _Observed,
    _Option,
)

if TYPE_CHECKING:
    from elspeth.web.catalog.protocol import CatalogService as CatalogServiceProtocol
    from elspeth.web.composer.guided.recipe_match import RecipeMatch
    from elspeth.web.composer.guided.state_machine import ChainProposal, SourceIntent
    from elspeth.web.composer.source_inspection import SourceInspectionFacts
    from elspeth.web.composer.state import CompositionState


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

    Called by GET /guided to rebuild the INSPECT_AND_CONFIRM turn on refresh
    when ``guided.step_1_source_intent`` is set.  The intent carries the
    plugin's observed_columns (the key data the widget needs), but not
    the original blob-inspection warnings — those are not stored on
    SourceIntent because they are observations about the blob, not about the
    plugin choices.  The rebuild emits ``warnings=[]`` for this reason.
    Clients that need the original warnings must resubmit the prior response;
    the reconstructed turn is functionally equivalent for resuming the flow.

    Args:
        intent: The staged SourceIntent from GuidedSession.step_1_source_intent.

    Returns:
        A ``Turn`` TypedDict ready for serialisation and hash.
    """
    observed: _Observed = {
        "columns": list(intent.observed_columns),
        "samples": [],
        "warnings": [],
    }
    payload: InspectAndConfirmPayload = {"observed": observed}
    return Turn(
        type=TurnType.INSPECT_AND_CONFIRM.value,
        step_index=_step_index(GuidedStep.STEP_1_SOURCE),
        payload=payload,
    )


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
    prefilled: dict[str, object] = {"schema": {"mode": "observed"}}
    if inspection_facts is not None:
        _merge_inspection_into_prefill(prefilled, inspection_facts)
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


def _merge_inspection_into_prefill(
    prefilled: dict[str, object],
    facts: SourceInspectionFacts,
) -> None:
    """Conservatively prefill source schema from inspection facts."""
    if facts.observed_headers and facts.inferred_types:
        fields: list[str] = []
        for header in facts.observed_headers:
            inferred = facts.inferred_types[header]
            field_type = "any" if inferred == "null" else inferred
            fields.append(f"{header}: {field_type}")
        prefilled["schema"] = {"mode": "flexible", "fields": fields}
    elif facts.observed_headers:
        prefilled["schema"] = {"mode": "observed"}
    # Delimiter and encoding are deliberately not prefilled here: the live
    # SourceInspectionFacts model does not carry those fields yet.


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
            label=plugin.name,
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
) -> Turn:
    """Build a ``schema_form`` Turn for the chosen sink plugin.

    Emitted after the user picks a sink plugin in Step 2's ``single_select``
    turn.  The schema block is the plugin's full JSON schema; ``prefilled``
    seeds ``schema.mode: "observed"``.

    Args:
        plugin: The plugin name chosen by the user (e.g. ``"json"``).
        catalog: Plugin catalog for retrieving the plugin's JSON schema.

    Returns:
        A ``Turn`` TypedDict ready for serialisation and hash.
    """
    schema_info = catalog.get_schema("sink", plugin)
    prefilled: dict[str, Any] = {"schema": {"mode": "observed"}}
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


def build_step_2_multi_select_turn(
    observed_columns: Sequence[str],
) -> Turn:
    """Build a ``multi_select_with_custom`` Turn for declaring required fields.

    Emitted after the user fills in sink options (Step 2 ``schema_form``).
    The options are pre-populated from Step 1's observed columns; the user
    ticks which fields must appear in the output, adds custom fields, or
    clicks the escape label to let the source decide.

    Args:
        observed_columns: The columns observed from the source in Step 1.
            Comes from ``GuidedSession.step_1_result.observed_columns``.

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


def build_step_2_5_recipe_offer_turn(
    match: RecipeMatch,
) -> Turn:
    """Build a ``recipe_offer`` Turn with a schema-form recipe decision payload.

    ``TurnType.RECIPE_OFFER`` is retained for guided state-machine routing,
    while ``payload.mode == "recipe_decision"`` routes the shared one-knob
    renderer on the frontend.

    Args:
        match: The matched recipe with its partial slot map.

    Returns:
        A ``Turn`` TypedDict ready for serialisation and hash.
    """
    from elspeth.contracts.freeze import deep_thaw
    from elspeth.web.composer.guided.errors import InvariantError
    from elspeth.web.composer.recipes import get_recipe

    recipe = get_recipe(match.recipe_name)
    if recipe is None:
        raise InvariantError(f"Recipe {match.recipe_name!r} disappeared from registry")

    payload: SchemaFormPayload = {
        "mode": "recipe_decision",
        "knobs": lower_slot_specs_to_knob_schema(match.unsatisfied_slots),
        "prefilled": dict(deep_thaw(match.slots)),
        "recipe_context": {
            "recipe_name": match.recipe_name,
            "description": recipe.description,
            "alternatives": ["build_manually"],
        },
    }
    return Turn(
        type=TurnType.RECIPE_OFFER.value,
        step_index=_step_index(GuidedStep.STEP_2_5_RECIPE_MATCH),
        payload=payload,
    )


def build_step_3_propose_chain_turn(
    proposal: ChainProposal,
) -> Turn:
    """Build a ``propose_chain`` Turn from a ChainProposal.

    Emitted at Step 3 after ``solve_chain`` returns a complete chain proposal.

    The ``blockers`` field is always ``[]`` for chain proposals emitted
    server-side after a successful ``solve_chain`` call.  The LLM may use
    ``blockers`` when it can only produce a partial chain; for the MVP we
    do not surface that path — ``solve_chain`` either returns a complete
    proposal or raises (so the dispatcher punts).

    Args:
        proposal: The chain proposal returned by ``solve_chain``.

    Returns:
        A ``Turn`` TypedDict ready for serialisation and hash.
    """
    from elspeth.contracts.freeze import deep_thaw
    from elspeth.web.composer.guided.protocol import _ProposedStep

    # Thaw the frozen ChainProposal step mappings into plain dicts so the
    # payload is JSON-serialisable.  ``_ProposedStep`` is a TypedDict so
    # mypy needs the explicit shape — the chain solver guarantees these
    # keys (validate against the LLM tool schema at solve_chain time).
    thawed_steps: list[_ProposedStep] = [
        _ProposedStep(
            plugin=str(s["plugin"]),
            options=dict(deep_thaw(s["options"])),
            rationale=str(s["rationale"]),
        )
        for s in proposal.steps
    ]
    payload: ProposeChainPayload = {
        "steps": thawed_steps,
        "why": proposal.why,
        "blockers": [],
    }
    return Turn(
        type=TurnType.PROPOSE_CHAIN.value,
        step_index=_step_index(GuidedStep.STEP_3_TRANSFORMS),
        payload=payload,
    )


def build_step_3_schema_form_turn(
    *,
    plugin: str,
    options: Mapping[str, Any],
    catalog: CatalogServiceProtocol,
) -> Turn:
    """Build a ``schema_form`` Turn for editing a proposed transform step."""
    schema_info = catalog.get_schema("transform", plugin)
    payload: SchemaFormPayload = {
        "mode": "plugin_options",
        "plugin": plugin,
        "knobs": cast(KnobSchema, schema_info.knob_schema),
        "prefilled": dict(options),
    }
    return Turn(
        type=TurnType.SCHEMA_FORM.value,
        step_index=_step_index(GuidedStep.STEP_3_TRANSFORMS),
        payload=payload,
    )


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


def _build_step_1_single_select_turn(
    catalog: CatalogServiceProtocol,
) -> Turn:
    """Build a ``single_select`` Turn listing available source plugins."""
    sources = catalog.list_sources()
    options: list[_Option] = [
        _Option(
            id=plugin.name,
            label=plugin.name,
            hint=plugin.description if plugin.description else None,
        )
        for plugin in sources
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
        GuidedStep.STEP_2_5_RECIPE_MATCH,
        GuidedStep.STEP_3_TRANSFORMS,
    )
    return _ORDER.index(step)
