"""Schema-8 GET rebuild and structured Step-1 plugin-hint tests."""

from __future__ import annotations

from typing import Literal

from elspeth.web.composer.guided.protocol import GuidedStep, TurnType
from elspeth.web.composer.guided.resolved import SourceResolved
from elspeth.web.composer.guided.state_machine import (
    ComponentTarget,
    GuidedSession,
    SourceIntent,
    TurnRecord,
)
from elspeth.web.composer.source_inspection import SourceInspectionFacts
from elspeth.web.sessions.routes.composer.guided import (
    _build_get_guided_turn,
    _step_1_plugin_hint,
)

SOURCE_ID = "00000000-0000-4000-8000-000000000401"


def _history_with_changed_copy() -> tuple[TurnRecord, ...]:
    return (
        TurnRecord(
            step=GuidedStep.STEP_1_SOURCE,
            turn_type=TurnType.SINGLE_SELECT,
            payload_hash="a" * 64,
            response_hash="b" * 64,
            emitter="server",
            summary="You picked: json (new copy, no Selected prefix)",
        ),
    )


def _source_intent(phase: Literal["plugin_selection", "plugin_options", "inspection_review"]) -> SourceIntent:
    if phase == "plugin_selection":
        return SourceIntent(
            name="source",
            phase="plugin_selection",
            plugin=None,
            options=None,
            inspection_facts=None,
            observed_columns=(),
            sample_rows=(),
        )
    if phase == "plugin_options":
        return SourceIntent(
            name="source",
            phase="plugin_options",
            plugin="csv",
            options=None,
            inspection_facts=None,
            observed_columns=(),
            sample_rows=(),
        )
    facts = SourceInspectionFacts(
        source_kind="csv",
        redacted_identity={"filename": "input.csv"},
        byte_range_inspected=(0, 10),
        sample_row_count=1,
        observed_headers=("text",),
        inferred_types={"text": "str"},
        url_candidates=(),
        warnings=(),
    )
    return SourceIntent(
        name="source",
        phase="inspection_review",
        plugin="csv",
        options={"path": "/data/input.csv"},
        inspection_facts=facts,
        observed_columns=("text",),
        sample_rows=({"text": "hello"},),
    )


def _pending_session(phase: Literal["plugin_selection", "plugin_options", "inspection_review"]) -> GuidedSession:
    return GuidedSession(
        step=GuidedStep.STEP_1_SOURCE,
        history=_history_with_changed_copy(),
        source_order=(SOURCE_ID,),
        pending_source_intents={SOURCE_ID: _source_intent(phase)},
    )


def test_hint_from_pending_plugin_options_ignores_summary_copy() -> None:
    assert _step_1_plugin_hint(_pending_session("plugin_options")) == "csv"


def test_hint_from_pending_inspection_review() -> None:
    assert _step_1_plugin_hint(_pending_session("inspection_review")) == "csv"


def test_plugin_selection_has_no_hint_even_with_selected_summary() -> None:
    assert _step_1_plugin_hint(_pending_session("plugin_selection")) is None


def test_hint_from_reviewed_source_requires_active_edit_target() -> None:
    source = SourceResolved(
        name="source",
        plugin="json",
        options={"path": "/data/input.json"},
        observed_columns=("text",),
        sample_rows=({"text": "hello"},),
        on_validation_failure="discard",
    )
    without_edit = GuidedSession(
        step=GuidedStep.STEP_1_SOURCE,
        source_order=(SOURCE_ID,),
        reviewed_sources={SOURCE_ID: source},
    )
    with_edit = GuidedSession(
        step=GuidedStep.STEP_1_SOURCE,
        source_order=(SOURCE_ID,),
        reviewed_sources={SOURCE_ID: source},
        active_edit_target=ComponentTarget(kind="source", stable_id=SOURCE_ID),
    )

    assert _step_1_plugin_hint(without_edit) is None
    assert _step_1_plugin_hint(with_edit) == "json"


def test_step_3_checkpoint_does_not_read_removed_schema_7_proposal_fields() -> None:
    guided = GuidedSession(step=GuidedStep.STEP_3_TRANSFORMS)

    assert _build_get_guided_turn(None, guided, catalog=None) is None
