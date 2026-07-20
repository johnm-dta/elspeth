"""p1 Task 2.5 — from-resolved schema_form builders prefill the applied config."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from elspeth.web.catalog.protocol import CatalogService
from elspeth.web.catalog.schemas import PluginSchemaInfo
from elspeth.web.composer.guided.emitters import (
    build_step_1_schema_form_turn_from_resolved,
    build_step_2_schema_form_turn_from_resolved,
)
from elspeth.web.composer.guided.errors import InvariantError
from elspeth.web.composer.guided.resolved import (
    SinkOutputResolved,
    SinkResolved,
    SourceResolved,
)


def _catalog() -> CatalogService:
    catalog = MagicMock(spec=CatalogService)
    catalog.get_schema.return_value = PluginSchemaInfo(
        name="test",
        plugin_type="source",
        description="Test schema",
        json_schema={"type": "object", "properties": {}},
        knob_schema={"fields": []},
    )
    return catalog


def test_source_prefill_carries_applied_options() -> None:
    source = SourceResolved(
        name="source",
        plugin="csv",
        options={"path": "/data/x.csv", "schema": {"mode": "observed"}},
        observed_columns=("a", "b"),
        sample_rows=({"a": "1", "b": "2"},),
        on_validation_failure="discard",
    )
    turn = build_step_1_schema_form_turn_from_resolved(source, _catalog())
    assert turn["type"] == "schema_form"
    assert turn["step_index"] == 0
    assert turn["payload"]["plugin"] == "csv"
    assert turn["payload"]["prefilled"]["path"] == "/data/x.csv"


def test_source_prefill_carries_on_validation_failure() -> None:
    """The resolved source-node routing must land in ``prefilled`` so the
    schema_form's required-no-default ``on_validation_failure`` knob has a value
    and the disabled Continue can enable. A non-default sentinel proves it is the
    resolved value (not a coincidental 'discard')."""
    source = SourceResolved(
        name="source",
        plugin="csv",
        options={"path": "/data/x.csv", "schema": {"mode": "observed"}},
        observed_columns=("a", "b"),
        sample_rows=({"a": "1", "b": "2"},),
        on_validation_failure="quarantine_sink",
    )
    turn = build_step_1_schema_form_turn_from_resolved(source, _catalog())
    assert turn["payload"]["prefilled"]["on_validation_failure"] == "quarantine_sink"


def test_json_source_schema_marks_on_validation_failure_required_no_default() -> None:
    """Load-bearing premise of the whole feature, asserted against the REAL catalog:
    ``on_validation_failure`` is a required knob with no default, which is why the
    passive learner (who types nothing) leaves the schema_form's Continue disabled
    until it is prefilled."""
    from elspeth.web.dependencies import create_catalog_service

    knob_schema = create_catalog_service().get_schema("source", "json").knob_schema
    knobs = {f["name"]: f for f in knob_schema["fields"]}
    ovf = knobs["on_validation_failure"]
    assert ovf["required"] is True
    assert "default" not in ovf


def test_source_resolved_round_trips_on_validation_failure() -> None:
    """to_dict/from_dict preserves the field; a record missing it is malformed
    and crashes (Tier-1 strict rehydrate — no compat default)."""
    source = SourceResolved(
        name="source",
        plugin="json",
        options={"schema": {"mode": "observed"}},
        observed_columns=("url",),
        sample_rows=({"url": "https://example.test/a"},),
        on_validation_failure="quarantine_sink",
    )
    payload = source.to_dict()
    assert payload["on_validation_failure"] == "quarantine_sink"
    assert SourceResolved.from_dict(payload).on_validation_failure == "quarantine_sink"

    truncated = {k: v for k, v in payload.items() if k != "on_validation_failure"}
    with pytest.raises(InvariantError):
        SourceResolved.from_dict(truncated)


def test_sink_prefill_carries_applied_options() -> None:
    sink = SinkResolved(
        outputs=(
            SinkOutputResolved(
                name="main",
                plugin="json",
                options={"path": "/out/y.jsonl", "collision_policy": "auto_increment"},
                required_fields=(),
                schema_mode="observed",
                on_write_failure="discard",
            ),
        )
    )
    turn = build_step_2_schema_form_turn_from_resolved(sink, _catalog())
    assert turn["type"] == "schema_form"
    assert turn["step_index"] == 1
    assert turn["payload"]["plugin"] == "json"
    assert turn["payload"]["prefilled"]["path"] == "/out/y.jsonl"
    assert turn["payload"]["prefilled"]["collision_policy"] == "auto_increment"
