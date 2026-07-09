"""Package A unit tests — GET /guided STEP_3 rebuild degrade (A1) and the
structured Step-1 plugin hint (A3).

A1: ``_build_get_guided_turn`` must treat a stale / out-of-range
``step_3_edit_index`` as no-edit-in-progress (render ``propose_chain``)
instead of raising IndexError.

A3: ``_step_1_plugin_hint`` must derive the Step-1 chat plugin hint from
structured session state (``step_1_source_intent`` / ``step_1_chosen_plugin``
/ ``step_1_result``), never by parsing the denormalised display summary —
so a copy change to the "Selected: " prefix cannot break chat resolution.
"""

from __future__ import annotations

from typing import ClassVar

from elspeth.web.composer.guided.resolved import SourceResolved
from elspeth.web.composer.guided.state_machine import (
    ChainProposal,
    GuidedSession,
    GuidedStep,
    SourceIntent,
    TurnRecord,
    TurnType,
)
from elspeth.web.sessions.routes.composer.guided import (
    _build_get_guided_turn,
    _step_1_plugin_hint,
)


def _step_3_session(*, steps: int, edit_index: int | None) -> GuidedSession:
    proposal = ChainProposal(
        steps=tuple(
            {
                "plugin": "passthrough",
                "options": {"schema": {"mode": "observed"}},
                "rationale": f"step {i}",
            }
            for i in range(steps)
        ),
        why="test proposal",
    )
    return GuidedSession(
        step=GuidedStep.STEP_3_TRANSFORMS,
        history=(),
        step_1_result=None,
        step_2_result=None,
        step_3_proposal=proposal,
        step_3_edit_index=edit_index,
    )


class TestBuildGetGuidedTurnStaleEditIndex:
    def test_out_of_range_edit_index_degrades_to_propose_chain(self) -> None:
        guided = _step_3_session(steps=1, edit_index=1)
        turn = _build_get_guided_turn(None, guided, catalog=None)
        assert turn is not None
        assert turn["type"] == TurnType.PROPOSE_CHAIN.value
        assert len(turn["payload"]["steps"]) == 1

    def test_in_range_edit_index_still_renders_schema_form(self) -> None:
        """The valid-edit path must be untouched by the stale-index guard."""

        class _FakeSchemaInfo:
            knob_schema: ClassVar[dict] = {"type": "object", "properties": {}}
            plugin_description = "passthrough transform"

        class _FakeCatalog:
            def get_schema(self, kind: str, name: str) -> _FakeSchemaInfo:
                assert kind == "transform"
                assert name == "passthrough"
                return _FakeSchemaInfo()

        guided = _step_3_session(steps=2, edit_index=1)
        turn = _build_get_guided_turn(None, guided, catalog=_FakeCatalog())
        assert turn is not None
        assert turn["type"] == TurnType.SCHEMA_FORM.value


def _step_1_session(
    *,
    chosen_plugin: str | None = None,
    intent_plugin: str | None = None,
    result_plugin: str | None = None,
    history: tuple[TurnRecord, ...] = (),
) -> GuidedSession:
    intent = None
    if intent_plugin is not None:
        intent = SourceIntent(
            plugin=intent_plugin,
            options={"schema": {"mode": "observed"}},
            observed_columns=("text",),
            sample_rows=({"text": "hello"},),
        )
    result = None
    if result_plugin is not None:
        result = SourceResolved(
            plugin=result_plugin,
            options={"schema": {"mode": "observed"}},
            observed_columns=("text",),
            sample_rows=({"text": "hello"},),
        )
    return GuidedSession(
        step=GuidedStep.STEP_1_SOURCE,
        history=history,
        step_1_result=result,
        step_2_result=None,
        step_3_proposal=None,
        step_1_chosen_plugin=chosen_plugin,
        step_1_source_intent=intent,
    )


class TestStep1PluginHint:
    def test_hint_from_chosen_plugin_ignores_summary_copy(self) -> None:
        """A copy change to the SINGLE_SELECT summary must not break the hint."""
        tampered_summary_record = TurnRecord(
            step=GuidedStep.STEP_1_SOURCE,
            turn_type=TurnType.SINGLE_SELECT,
            payload_hash="p" * 8,
            response_hash="r" * 8,
            emitter="server",
            summary="You picked: json (new copy, no 'Selected: ' prefix)",
        )
        guided = _step_1_session(chosen_plugin="csv", history=(tampered_summary_record,))
        assert _step_1_plugin_hint(guided) == "csv"

    def test_hint_from_source_intent_when_awaiting_inspect_and_confirm(self) -> None:
        guided = _step_1_session(intent_plugin="csv")
        assert _step_1_plugin_hint(guided) == "csv"

    def test_hint_from_committed_result_when_no_staging_fields(self) -> None:
        guided = _step_1_session(result_plugin="json")
        assert _step_1_plugin_hint(guided) == "json"

    def test_no_structured_state_yields_no_hint_even_with_selected_summary(self) -> None:
        """No structured plugin anywhere → None; the display summary is never parsed."""
        legacy_summary_record = TurnRecord(
            step=GuidedStep.STEP_1_SOURCE,
            turn_type=TurnType.SINGLE_SELECT,
            payload_hash="p" * 8,
            response_hash="r" * 8,
            emitter="server",
            summary="Selected: csv, comma-separated values",
        )
        guided = _step_1_session(history=(legacy_summary_record,))
        assert _step_1_plugin_hint(guided) is None

    def test_intent_wins_over_chosen_and_result(self) -> None:
        """Priority mirrors the GET /guided rebuild: intent → chosen → result."""
        guided = _step_1_session(chosen_plugin="json", intent_plugin="csv", result_plugin="azure_blob")
        assert _step_1_plugin_hint(guided) == "csv"
