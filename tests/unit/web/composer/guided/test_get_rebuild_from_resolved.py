"""Schema-8 active-edit rebuilds render the reviewed component in place."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from elspeth.web.catalog.protocol import CatalogService
from elspeth.web.composer.guided.protocol import GuidedStep
from elspeth.web.composer.guided.resolved import SinkOutputResolved, SourceResolved
from elspeth.web.composer.guided.state_machine import ComponentTarget, GuidedSession
from elspeth.web.sessions.routes.composer.guided import _build_get_guided_turn

SOURCE_ID = "00000000-0000-4000-8000-000000000301"
OUTPUT_ID = "00000000-0000-4000-8000-000000000302"


@dataclass(frozen=True, slots=True)
class _SchemaInfo:
    knob_schema: dict[str, Any]


class _CatalogFake:
    def get_schema(self, _kind: str, _plugin: str) -> _SchemaInfo:
        return _SchemaInfo(knob_schema={"fields": []})


class _StatePlaceholder:
    pass


def _catalog() -> CatalogService:
    return _CatalogFake()  # type: ignore[return-value]


def test_get_rebuild_step_1_uses_reviewed_source_for_active_edit() -> None:
    source = SourceResolved(
        name="source",
        plugin="csv",
        options={"path": "/data/x.csv", "schema": {"mode": "observed"}},
        observed_columns=("a", "b"),
        sample_rows=({"a": "1", "b": "2"},),
        on_validation_failure="discard",
    )
    guided = GuidedSession(
        step=GuidedStep.STEP_1_SOURCE,
        source_order=(SOURCE_ID,),
        reviewed_sources={SOURCE_ID: source},
        active_edit_target=ComponentTarget(kind="source", stable_id=SOURCE_ID),
    )

    turn = _build_get_guided_turn(_StatePlaceholder(), guided, catalog=_catalog())

    assert turn is not None
    assert turn["type"] == "schema_form"
    assert turn["step_index"] == 0
    assert turn["payload"]["plugin"] == "csv"
    assert turn["payload"]["prefilled"]["path"] == "/data/x.csv"


def test_get_rebuild_step_2_uses_reviewed_output_for_active_edit() -> None:
    output = SinkOutputResolved(
        name="main",
        plugin="json",
        options={"path": "/out/y.jsonl"},
        required_fields=(),
        schema_mode="observed",
        on_write_failure="discard",
    )
    guided = GuidedSession(
        step=GuidedStep.STEP_2_SINK,
        output_order=(OUTPUT_ID,),
        reviewed_outputs={OUTPUT_ID: output},
        active_edit_target=ComponentTarget(kind="output", stable_id=OUTPUT_ID),
    )

    turn = _build_get_guided_turn(_StatePlaceholder(), guided, catalog=_catalog())

    assert turn is not None
    assert turn["type"] == "schema_form"
    assert turn["step_index"] == 1
    assert turn["payload"]["plugin"] == "json"
    assert turn["payload"]["prefilled"]["path"] == "/out/y.jsonl"
