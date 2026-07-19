"""Schema-8 active-edit rebuilds render the reviewed component in place."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from elspeth.web.catalog.protocol import CatalogService
from elspeth.web.composer.guided.protocol import GuidedStep
from elspeth.web.composer.guided.resolved import SinkOutputResolved, SourceResolved
from elspeth.web.composer.guided.state_machine import ComponentTarget, GuidedSession, SinkIntent, SourceIntent
from elspeth.web.composer.source_inspection import SourceInspectionFacts
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
    assert turn["payload"]["prefilled"]["on_write_failure"] == "discard"


def test_get_rebuild_active_source_edit_resumes_inspection_review() -> None:
    facts = SourceInspectionFacts(
        source_kind="csv",
        redacted_identity={"blob_id": "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa", "filename": "input.csv"},
        byte_range_inspected=(0, 10),
        sample_row_count=1,
        observed_headers=("id",),
        inferred_types={"id": "str"},
        url_candidates=(),
        warnings=(),
    )
    source = SourceResolved(
        name="source",
        plugin="csv",
        options={"path": "blob:aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa"},
        observed_columns=("old",),
        sample_rows=(),
        on_validation_failure="discard",
    )
    guided = GuidedSession(
        step=GuidedStep.STEP_1_SOURCE,
        source_order=(SOURCE_ID,),
        reviewed_sources={SOURCE_ID: source},
        pending_source_intents={
            SOURCE_ID: SourceIntent(
                name="source",
                phase="inspection_review",
                plugin="csv",
                options={
                    "path": "blob:aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa",
                    "on_validation_failure": "discard",
                },
                inspection_facts=facts,
                observed_columns=("id",),
                sample_rows=(),
            )
        },
        active_edit_target=ComponentTarget(kind="source", stable_id=SOURCE_ID),
    )

    turn = _build_get_guided_turn(_StatePlaceholder(), guided, catalog=_catalog())

    assert turn is not None
    assert turn["type"] == "inspect_and_confirm"


def test_get_rebuild_active_output_edit_resumes_field_review() -> None:
    output = SinkOutputResolved(
        name="main",
        plugin="json",
        options={"path": "/out/old.jsonl"},
        required_fields=("old",),
        schema_mode="observed",
        on_write_failure="discard",
    )
    guided = GuidedSession(
        step=GuidedStep.STEP_2_SINK,
        output_order=(OUTPUT_ID,),
        reviewed_outputs={OUTPUT_ID: output},
        pending_output_intents={
            OUTPUT_ID: SinkIntent(
                name="main",
                phase="field_review",
                plugin="json",
                options={"path": "/out/new.jsonl", "on_write_failure": "discard"},
            )
        },
        active_edit_target=ComponentTarget(kind="output", stable_id=OUTPUT_ID),
    )

    turn = _build_get_guided_turn(_StatePlaceholder(), guided, catalog=_catalog())

    assert turn is not None
    assert turn["type"] == "multi_select_with_custom"
