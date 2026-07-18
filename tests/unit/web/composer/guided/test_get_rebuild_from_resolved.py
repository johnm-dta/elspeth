"""p1 Task 2.5 — _build_get_guided_turn re-renders the applied form in place."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from elspeth.web.catalog.protocol import CatalogService
from elspeth.web.composer.guided.protocol import GuidedStep
from elspeth.web.composer.guided.resolved import (
    SinkOutputResolved,
    SinkResolved,
    SourceResolved,
)
from elspeth.web.sessions.routes.composer.guided import _build_get_guided_turn


@dataclass(frozen=True, slots=True)
class _SchemaInfo:
    knob_schema: dict[str, Any]


class _CatalogFake:
    def get_schema(self, _kind: str, _plugin: str) -> _SchemaInfo:
        return _SchemaInfo(knob_schema={"fields": []})


@dataclass(slots=True)
class _GuidedFake:
    step: GuidedStep
    step_1_source_intent: Any = None
    step_1_chosen_plugin: str | None = None
    step_1_result: SourceResolved | None = None
    step_1_inspection_facts: Any = None
    step_2_sink_intent: Any = None
    step_2_chosen_plugin: str | None = None
    step_2_result: SinkResolved | None = None


class _StatePlaceholder:
    pass


def _catalog() -> CatalogService:
    return _CatalogFake()  # type: ignore[return-value]


def test_get_rebuild_step_1_uses_from_resolved_when_applied_in_place() -> None:
    source = SourceResolved(
        name="source",
        plugin="csv",
        options={"path": "/data/x.csv", "schema": {"mode": "observed"}},
        observed_columns=("a", "b"),
        sample_rows=({"a": "1", "b": "2"},),
        on_validation_failure="discard",
    )
    # In-place applied STEP_1 state: result set, BOTH staging fields cleared.
    guided = _GuidedFake(step=GuidedStep.STEP_1_SOURCE, step_1_result=source)
    turn = _build_get_guided_turn(_StatePlaceholder(), guided, catalog=_catalog())
    assert turn["type"] == "schema_form"
    assert turn["step_index"] == 0
    assert turn["payload"]["plugin"] == "csv"
    assert turn["payload"]["prefilled"]["path"] == "/data/x.csv"


def test_get_rebuild_step_2_uses_from_resolved_when_applied_in_place() -> None:
    sink = SinkResolved(
        outputs=(
            SinkOutputResolved(
                name="main",
                plugin="json",
                options={"path": "/out/y.jsonl"},
                required_fields=(),
                schema_mode="observed",
                on_write_failure="discard",
            ),
        )
    )
    guided = _GuidedFake(step=GuidedStep.STEP_2_SINK, step_2_result=sink)
    turn = _build_get_guided_turn(_StatePlaceholder(), guided, catalog=_catalog())
    assert turn["type"] == "schema_form"
    assert turn["step_index"] == 1
    assert turn["payload"]["plugin"] == "json"
    assert turn["payload"]["prefilled"]["path"] == "/out/y.jsonl"
