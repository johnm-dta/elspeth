"""p1 Task 2.5 — _build_get_guided_turn re-renders the applied form in place."""

from __future__ import annotations

from unittest.mock import MagicMock

from elspeth.web.catalog.protocol import CatalogService
from elspeth.web.composer.guided.protocol import GuidedStep
from elspeth.web.composer.guided.resolved import (
    SinkOutputResolved,
    SinkResolved,
    SourceResolved,
)
from elspeth.web.sessions.routes.composer.guided import _build_get_guided_turn


def _catalog() -> CatalogService:
    catalog = MagicMock(spec=CatalogService)
    catalog.get_schema.return_value = MagicMock(knob_schema={"fields": []})
    return catalog


def test_get_rebuild_step_1_uses_from_resolved_when_applied_in_place() -> None:
    source = SourceResolved(
        plugin="csv",
        options={"path": "/data/x.csv", "schema": {"mode": "observed"}},
        observed_columns=("a", "b"),
        sample_rows=({"a": "1", "b": "2"},),
    )
    # In-place applied STEP_1 state: result set, BOTH staging fields cleared.
    guided = MagicMock()
    guided.step = GuidedStep.STEP_1_SOURCE
    guided.step_1_source_intent = None
    guided.step_1_chosen_plugin = None
    guided.step_1_result = source
    guided.step_1_inspection_facts = None
    turn = _build_get_guided_turn(MagicMock(), guided, catalog=_catalog())
    assert turn["type"] == "schema_form"
    assert turn["step_index"] == 0
    assert turn["payload"]["plugin"] == "csv"
    assert turn["payload"]["prefilled"]["path"] == "/data/x.csv"


def test_get_rebuild_step_2_uses_from_resolved_when_applied_in_place() -> None:
    sink = SinkResolved(
        outputs=(
            SinkOutputResolved(
                plugin="json",
                options={"path": "/out/y.jsonl"},
                required_fields=(),
                schema_mode="observed",
            ),
        )
    )
    guided = MagicMock()
    guided.step = GuidedStep.STEP_2_SINK
    guided.step_2_sink_intent = None
    guided.step_2_chosen_plugin = None
    guided.step_2_result = sink
    turn = _build_get_guided_turn(MagicMock(), guided, catalog=_catalog())
    assert turn["type"] == "schema_form"
    assert turn["step_index"] == 1
    assert turn["payload"]["plugin"] == "json"
    assert turn["payload"]["prefilled"]["path"] == "/out/y.jsonl"
