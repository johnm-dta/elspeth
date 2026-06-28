"""Guided-mode protocol: turn types, payloads, responses, legal-turn matrix.

See docs/superpowers/specs/2026-05-11-composer-guided-mode-design.md §4.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, NotRequired, TypedDict

from elspeth.web.catalog.knob_schema import SchemaFormPayload as SchemaFormPayload

# Wire sentinel for a blob-backed source's ``path`` knob in a schema_form payload.
# The emitter renders ``blob:<blob_ref>`` instead of the blob's ABSOLUTE
# storage_path (which would leak the deploy dir + OS username — see
# blobs/schemas.py, where storage_path is forbidden from HTTP responses); the
# step_1 commit handler re-resolves it to the real path via an authoritative
# by-id blob lookup. A filesystem path never starts with this prefix, so the
# sentinel is unambiguous.
BLOB_REF_PATH_PREFIX = "blob:"


class TurnType(StrEnum):
    """The closed taxonomy of turn types the protocol allows."""

    INSPECT_AND_CONFIRM = "inspect_and_confirm"
    SINGLE_SELECT = "single_select"
    MULTI_SELECT_WITH_CUSTOM = "multi_select_with_custom"
    SCHEMA_FORM = "schema_form"
    PROPOSE_CHAIN = "propose_chain"
    RECIPE_OFFER = "recipe_offer"
    CONFIRM_WIRING = "confirm_wiring"


class _Option(TypedDict):
    id: str
    label: str
    hint: str | None


class _Observed(TypedDict):
    columns: Sequence[str]
    samples: Sequence[Mapping[str, Any]]
    warnings: Sequence[str]


class InspectAndConfirmPayload(TypedDict):
    observed: _Observed


class SingleSelectPayload(TypedDict):
    question: str
    options: Sequence[_Option]
    allow_custom: bool


class MultiSelectWithCustomPayload(TypedDict):
    question: str
    options: Sequence[_Option]
    default_chosen: Sequence[str]
    escape_label: str | None


class _ProposedStep(TypedDict):
    plugin: str
    options: Mapping[str, Any]
    rationale: str


class ProposeChainPayload(TypedDict):
    steps: Sequence[_ProposedStep]
    why: str
    blockers: Sequence[str]


class _WireSourceTopo(TypedDict):
    id: str
    plugin: str
    on_success: str | None
    on_validation_failure: str


class _WireNodeTopo(TypedDict):
    id: str
    node_type: str
    plugin: str | None
    input: str | None
    on_success: str | None
    on_error: str | None
    routes: Mapping[str, str] | None
    fork_to: Sequence[str] | None
    branches: Sequence[str] | Mapping[str, str] | None


class _WireOutputTopo(TypedDict):
    id: str
    sink_name: str
    plugin: str
    on_write_failure: str


class WireTopology(TypedDict):
    """Connection-label topology for the wire stage (from get_pipeline_state)."""

    sources: Mapping[str, _WireSourceTopo]
    nodes: Sequence[_WireNodeTopo]
    outputs: Sequence[_WireOutputTopo]


class WireStageData(TypedDict):
    """STEP_4_WIRE turn payload: topology + validate() contract overlay.

    edge_contracts entries carry keys from/to, not from_id/to_id. warnings carries the live prompt-shield advisory. Renderers reconstruct edges from topology labels, never state.edges. Source rows carry id values matching validation producer ids (`source` or `source:<name>`); output rows carry id values matching validation sink ids (`output:<sink_name>`). sink_name remains the connection label; output.id is the edge target for overlay.
    """

    topology: WireTopology
    edge_contracts: Sequence[Mapping[str, Any]]
    semantic_contracts: Sequence[Mapping[str, Any]]
    warnings: Sequence[Mapping[str, Any]]
    advisor_findings: NotRequired[str]
    signoff_outcome: NotRequired[str]


class ControlSignal(StrEnum):
    """Out-of-band signals carried in a TurnResponse instead of (or alongside) data."""

    EXIT_TO_FREEFORM = "exit_to_freeform"
    REQUEST_ADVISOR = "request_advisor"
    REJECT = "reject"
    BACK = "back"


class TurnResponse(TypedDict):
    """The user's typed response to a turn."""

    chosen: Sequence[str] | None
    edited_values: Mapping[str, Any] | None
    custom_inputs: Sequence[str] | None
    accepted_step_index: int | None
    edit_step_index: int | None
    control_signal: ControlSignal | None


class Turn(TypedDict):
    """A turn emitted to the user (server-emitted or LLM-emitted)."""

    type: str  # TurnType value
    step_index: int
    payload: Mapping[str, Any]


class GuidedStep(StrEnum):
    """Wizard step pointer."""

    STEP_1_SOURCE = "step_1_source"
    STEP_2_SINK = "step_2_sink"
    STEP_2_5_RECIPE_MATCH = "step_2_5_recipe_match"
    STEP_3_TRANSFORMS = "step_3_transforms"
    STEP_4_WIRE = "step_4_wire"


class ChatRole(StrEnum):
    """Closed taxonomy of chat-turn authors.

    A second author class for Phase A.5 — ``step_entry_opener`` (proactive
    server-initiated turn) — is intentionally absent here.  Phase A only
    distinguishes user-initiated and assistant-reply turns.  When openers
    land, the discriminator moves to ``ComposerChatTurn.initiator`` (audit
    record), not the ``ChatRole`` enum on the user-visible history — both
    a user prompt and a step-entry opener produce an ``assistant`` turn
    on the wire.  The plan §"Opener-specific invariant" relies on this
    separation.
    """

    USER = "user"
    ASSISTANT = "assistant"


@dataclass(frozen=True, slots=True)
class ChatTurn:
    """One conversational message in the per-step chat history (Phase A slice 5).

    Persisted in ``GuidedSession.chat_history``.  Trust tier: Tier 1
    (audit) — every field is server-authoritative.  ``seq`` is monotonic
    per session across all chat turns (user + assistant share the
    counter); on reload the frontend renders entries in ``seq`` order.

    ``ts_iso`` is the server-recorded ISO 8601 timestamp at which the
    turn entered the history.  It is informational for the UI; chrono
    ordering should still be driven by ``seq`` to avoid sort drift when
    the same wall clock-second carries two turns.

    ``step`` records the wizard step the user was on when the turn was
    produced.  Slice 5 keeps the same step for both user message and
    assistant reply (chat does not advance step state); the field is
    load-bearing for Phase A.5 openers and Phase B tool palettes where
    the step at *emission* may differ from the step at *display* if the
    user back-buttons.
    """

    role: ChatRole
    content: str
    seq: int
    step: GuidedStep
    ts_iso: str

    def __post_init__(self) -> None:
        if type(self.role) is not ChatRole:
            raise TypeError(f"role must be ChatRole, got {type(self.role).__name__}")
        if type(self.step) is not GuidedStep:
            raise TypeError(f"step must be GuidedStep, got {type(self.step).__name__}")
        if type(self.seq) is not int:
            raise TypeError(f"seq must be int, got {type(self.seq).__name__}")
        if self.seq < 0:
            raise ValueError("seq must be >= 0")
        if type(self.content) is not str:
            raise TypeError(f"content must be str, got {type(self.content).__name__}")
        if self.content == "":
            raise ValueError("content must be non-empty")
        if type(self.ts_iso) is not str:
            raise TypeError(f"ts_iso must be str, got {type(self.ts_iso).__name__}")
        if self.ts_iso == "":
            raise ValueError("ts_iso must be non-empty")


_LEGAL_TURN_MATRIX: Mapping[GuidedStep, frozenset[TurnType]] = {
    GuidedStep.STEP_1_SOURCE: frozenset(
        {
            TurnType.INSPECT_AND_CONFIRM,
            TurnType.SINGLE_SELECT,
            TurnType.SCHEMA_FORM,
        }
    ),
    GuidedStep.STEP_2_SINK: frozenset(
        {
            TurnType.SINGLE_SELECT,
            TurnType.MULTI_SELECT_WITH_CUSTOM,
            TurnType.SCHEMA_FORM,
        }
    ),
    GuidedStep.STEP_2_5_RECIPE_MATCH: frozenset({TurnType.RECIPE_OFFER}),
    GuidedStep.STEP_3_TRANSFORMS: frozenset(
        {
            TurnType.PROPOSE_CHAIN,
            TurnType.SINGLE_SELECT,
            TurnType.SCHEMA_FORM,
        }
    ),
    GuidedStep.STEP_4_WIRE: frozenset({TurnType.CONFIRM_WIRING}),
}


def legal_turn_types_for(step: GuidedStep) -> frozenset[TurnType]:
    """Return the frozen set of TurnType values legal at the given step."""
    return _LEGAL_TURN_MATRIX[step]


_REQUIRED_KEYS: Mapping[TurnType, frozenset[str]] = {
    TurnType.INSPECT_AND_CONFIRM: frozenset({"observed"}),
    TurnType.SINGLE_SELECT: frozenset({"question", "options", "allow_custom"}),
    TurnType.MULTI_SELECT_WITH_CUSTOM: frozenset(
        {
            "question",
            "options",
            "default_chosen",
            "escape_label",
        }
    ),
    TurnType.SCHEMA_FORM: frozenset({"mode", "knobs", "prefilled"}),
    TurnType.PROPOSE_CHAIN: frozenset({"steps", "why", "blockers"}),
    # The turn discriminator remains RECIPE_OFFER so the guided state machine
    # can route Step 2.5 without changing its legal-turn matrix. The payload
    # itself uses the SchemaFormPayload discriminator, where
    # mode="recipe_decision" routes the shared one-knob renderer.
    TurnType.RECIPE_OFFER: frozenset({"mode", "knobs", "prefilled", "recipe_context"}),
    TurnType.CONFIRM_WIRING: frozenset({"topology", "edge_contracts", "semantic_contracts", "warnings"}),
}

# Nested shape spec for recursive payload validation.
#
# Each entry maps a TurnType to a list of (field_path_component, field_kind,
# required_keys) triples that describe nested structures.
#
# ``field_kind`` is one of:
#   "mapping"  — the field value must be a Mapping; validate its required_keys.
#   "sequence_of_mappings" — the field value must be a Sequence; each element
#                            must be a Mapping with required_keys.
#
# Path-rooted error messages use dot notation: ``"payload.observed.columns"``.
#
# Only fields whose nested shapes are meaningful to validate are listed. Scalar
# and pass-through fields (e.g., ``allow_custom: bool``, ``why: str``) are
# covered by the top-level required-key check and need no further descend.
#
# This mapping is TOTAL over ``TurnType``: every turn type has an entry, and
# turn types with no nested shapes to validate carry an explicit empty tuple.
# Totality lets ``validate_payload`` direct-subscript ``_NESTED_SHAPES[turn_type]``
# (mirroring ``_REQUIRED_KEYS[turn_type]``) — a missing key is then a code bug
# (a new ``TurnType`` added without registering its nested shapes) that crashes
# loudly rather than silently skipping nested validation.
_NestedSpec = tuple[str, str, frozenset[str]]
_NESTED_SHAPES: Mapping[TurnType, tuple[_NestedSpec, ...]] = {
    TurnType.INSPECT_AND_CONFIRM: (
        # "observed" must be a Mapping with these keys
        ("observed", "mapping", frozenset({"columns", "samples", "warnings"})),
    ),
    TurnType.SINGLE_SELECT: (),
    TurnType.MULTI_SELECT_WITH_CUSTOM: (),
    TurnType.SCHEMA_FORM: (("knobs", "mapping", frozenset({"fields"})),),
    TurnType.PROPOSE_CHAIN: (),
    TurnType.RECIPE_OFFER: (("knobs", "mapping", frozenset({"fields"})),),
    TurnType.CONFIRM_WIRING: (("topology", "mapping", frozenset({"sources", "nodes", "outputs"})),),
}


def validate_payload(turn_type: TurnType, payload: Mapping[str, Any]) -> str | None:
    """Validate that *payload* satisfies the schema for *turn_type*.

    Validates top-level required keys and recursively walks nested TypedDicts
    listed in ``_NESTED_SHAPES``.  Error messages are path-rooted using dot
    notation so the caller can locate the offending field:
    ``"payload.observed.columns missing"`` rather than ``"columns missing"``.

    Returns None on success, or a human-readable error string on failure.
    Raises ValueError if turn_type is not a known TurnType.
    """
    if not isinstance(turn_type, TurnType):
        raise ValueError(f"unknown turn type: {turn_type!r}")
    required = _REQUIRED_KEYS[turn_type]
    missing = required - payload.keys()
    if missing:
        return f"payload for {turn_type.value} missing required keys: {sorted(missing)}"

    # Recursive nested-shape validation.
    for field_name, field_kind, nested_required in _NESTED_SHAPES[turn_type]:
        field_value = payload[field_name]
        prefix = f"payload.{field_name}"
        if field_kind == "mapping":
            if not isinstance(field_value, Mapping):
                return f"{prefix} must be a mapping (got {type(field_value).__name__})"
            nested_missing = nested_required - field_value.keys()
            if nested_missing:
                return f"{prefix} missing required keys: {sorted(nested_missing)}"
        elif field_kind == "sequence_of_mappings":
            if not isinstance(field_value, Sequence) or isinstance(field_value, str):
                return f"{prefix} must be a sequence (got {type(field_value).__name__})"
            for idx, item in enumerate(field_value):
                if not isinstance(item, Mapping):
                    return f"{prefix}[{idx}] must be a mapping (got {type(item).__name__})"
                item_missing = nested_required - item.keys()
                if item_missing:
                    return f"{prefix}[{idx}] missing required keys: {sorted(item_missing)}"

    if turn_type in {TurnType.SCHEMA_FORM, TurnType.RECIPE_OFFER}:
        mode = payload["mode"]
        if mode == "plugin_options":
            if turn_type is TurnType.RECIPE_OFFER:
                return "payload for recipe_offer must use mode='recipe_decision'"
            if "plugin" not in payload:
                return "payload for schema_form mode=plugin_options missing required keys: ['plugin']"
        elif mode == "recipe_decision":
            if "recipe_context" not in payload:
                return "payload for schema_form mode=recipe_decision missing required keys: ['recipe_context']"
            recipe_context = payload["recipe_context"]
            if not isinstance(recipe_context, Mapping):
                return f"payload.recipe_context must be a mapping (got {type(recipe_context).__name__})"
            recipe_context_missing = {"recipe_name", "description", "alternatives"} - recipe_context.keys()
            if recipe_context_missing:
                return f"payload.recipe_context missing required keys: {sorted(recipe_context_missing)}"
        else:
            return f"payload.mode must be 'plugin_options' or 'recipe_decision' (got {mode!r})"

    return None
