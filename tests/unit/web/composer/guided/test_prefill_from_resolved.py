"""p1 Task 2.5 — from-resolved schema_form builders prefill the applied config."""

from __future__ import annotations

from unittest.mock import MagicMock

from elspeth.web.catalog.protocol import CatalogService
from elspeth.web.composer.guided.emitters import (
    build_step_1_schema_form_turn_from_resolved,
    build_step_2_schema_form_turn_from_resolved,
)
from elspeth.web.composer.guided.resolved import (
    SinkOutputResolved,
    SinkResolved,
    SourceResolved,
)


def _catalog() -> CatalogService:
    catalog = MagicMock(spec=CatalogService)
    catalog.get_schema.return_value = MagicMock(knob_schema={"fields": []})
    return catalog


def test_source_prefill_carries_applied_options() -> None:
    source = SourceResolved(
        plugin="csv",
        options={"path": "/data/x.csv", "schema": {"mode": "observed"}},
        observed_columns=("a", "b"),
        sample_rows=({"a": "1", "b": "2"},),
    )
    turn = build_step_1_schema_form_turn_from_resolved(source, _catalog())
    assert turn["type"] == "schema_form"
    assert turn["step_index"] == 0
    assert turn["payload"]["plugin"] == "csv"
    assert turn["payload"]["prefilled"]["path"] == "/data/x.csv"


def test_sink_prefill_carries_applied_options() -> None:
    sink = SinkResolved(
        outputs=(
            SinkOutputResolved(
                plugin="json",
                options={"path": "/out/y.jsonl", "collision_policy": "auto_increment"},
                required_fields=(),
                schema_mode="observed",
            ),
        )
    )
    turn = build_step_2_schema_form_turn_from_resolved(sink, _catalog())
    assert turn["type"] == "schema_form"
    assert turn["step_index"] == 1
    assert turn["payload"]["plugin"] == "json"
    assert turn["payload"]["prefilled"]["path"] == "/out/y.jsonl"
    assert turn["payload"]["prefilled"]["collision_policy"] == "auto_increment"
