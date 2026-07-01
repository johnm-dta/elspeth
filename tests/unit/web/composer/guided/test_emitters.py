"""Tests for guided-mode turn emitters (Task 10.0 — Gap 6 wire shape)."""

from __future__ import annotations

from types import SimpleNamespace

from elspeth.web.composer.guided.emitters import (
    _step_index,
    build_initial_step_1_turn,
    build_step_1_schema_form_turn,
    build_step_1_schema_form_turn_from_resolved,
    build_step_2_schema_form_turn,
    build_step_3_schema_form_turn,
    build_step_4_wire_turn,
)
from elspeth.web.composer.guided.protocol import GuidedStep, TurnType, validate_payload
from elspeth.web.composer.guided.state_machine import SourceResolved
from elspeth.web.composer.source_inspection import SourceInspectionFacts
from elspeth.web.composer.state import CompositionState, PipelineMetadata
from elspeth.web.sessions.routes._helpers import _guided_step_index


class _Catalog:
    def get_schema(self, plugin_type: str, plugin_name: str) -> SimpleNamespace:
        return SimpleNamespace(
            json_schema={"properties": {"path": {"type": "string"}}},
            knob_schema={
                "fields": [
                    {
                        "name": "path",
                        "label": "Path",
                        "kind": "text",
                        "required": True,
                        "nullable": False,
                    }
                ]
            },
        )


def _empty_state() -> CompositionState:
    return CompositionState(
        source=None,
        nodes=(),
        edges=(),
        outputs=(),
        metadata=PipelineMetadata(),
        version=1,
    )


class TestBuildSchemaFormTurns:
    def test_step_1_schema_form_uses_knob_schema_payload(self) -> None:
        turn = build_step_1_schema_form_turn("csv", _Catalog())

        assert turn["type"] == TurnType.SCHEMA_FORM.value
        payload = turn["payload"]
        assert payload["mode"] == "plugin_options"
        assert payload["plugin"] == "csv"
        assert payload["knobs"]["fields"][0]["name"] == "path"
        assert "schema_block" not in payload
        assert validate_payload(TurnType.SCHEMA_FORM, payload) is None

    def test_step_1_schema_form_prefills_from_inspection_facts(self) -> None:
        facts = SourceInspectionFacts(
            source_kind="csv",
            redacted_identity={"filename": "input.csv"},
            byte_range_inspected=(0, 32),
            sample_row_count=1,
            observed_headers=("name", "age"),
            inferred_types={"name": "str", "age": "int"},
            url_candidates=(),
            warnings=(),
        )

        turn = build_step_1_schema_form_turn("csv", _Catalog(), inspection_facts=facts)

        schema_prefill = turn["payload"]["prefilled"]["schema"]
        assert schema_prefill == {"mode": "flexible", "fields": ["name: str", "age: int"]}

    def test_step_2_schema_form_uses_sink_knobs(self) -> None:
        turn = build_step_2_schema_form_turn("json", _Catalog())

        payload = turn["payload"]
        assert payload["mode"] == "plugin_options"
        assert payload["plugin"] == "json"
        assert payload["knobs"]["fields"][0]["label"] == "Path"
        assert payload["prefilled"] == {"schema": {"mode": "observed"}}

    def test_step_3_schema_form_uses_transform_knobs_and_options_prefill(self) -> None:
        turn = build_step_3_schema_form_turn(
            plugin="type_coerce",
            options={"conversions": {"age": "int"}},
            catalog=_Catalog(),
        )

        payload = turn["payload"]
        assert payload["mode"] == "plugin_options"
        assert payload["plugin"] == "type_coerce"
        assert payload["knobs"]["fields"][0]["name"] == "path"
        assert payload["prefilled"] == {"conversions": {"age": "int"}}


class TestStep4WireEmitter:
    def test_step_4_wire_index_matches_guided_order(self) -> None:
        assert _step_index(GuidedStep.STEP_4_WIRE) == 4
        assert _guided_step_index(GuidedStep.STEP_4_WIRE) == 4

    def test_builds_confirm_wiring_skeleton_payload(self) -> None:
        turn = build_step_4_wire_turn(_empty_state())

        assert turn["type"] == TurnType.CONFIRM_WIRING.value
        assert turn["step_index"] == 4
        assert validate_payload(TurnType.CONFIRM_WIRING, turn["payload"]) is None
        payload = turn["payload"]
        assert set(payload.keys()) == {
            "topology",
            "edge_contracts",
            "semantic_contracts",
            "warnings",
        }
        assert payload["topology"]["sources"] == {}
        assert payload["edge_contracts"] == []
        assert payload["semantic_contracts"] == []
        assert payload["warnings"] == []


class _SourceCatalog:
    """Catalog stub exposing list_sources for the step-1 single_select path."""

    def list_sources(self) -> list[SimpleNamespace]:
        return [
            SimpleNamespace(name="csv", description="Load rows from a CSV file."),
            SimpleNamespace(name="null", description="A source that yields no rows."),
            SimpleNamespace(name="json", description="Load rows from a JSON file."),
        ]


class TestStep1SourcePicker:
    def test_single_select_excludes_the_degenerate_null_source(self) -> None:
        turn = build_initial_step_1_turn(_empty_state(), blob_inspection=None, catalog=_SourceCatalog())

        assert turn["type"] == TurnType.SINGLE_SELECT.value
        option_ids = [opt["id"] for opt in turn["payload"]["options"]]
        assert "null" not in option_ids
        assert option_ids == ["csv", "json"]


class TestSchemaFormPathMask:
    """Fix B: a blob-backed source's absolute storage_path must never reach the
    wire — it is rendered as a stable blob:<ref> sentinel that the step_1 commit
    handler re-resolves to the real path."""

    def test_step_1_from_resolved_masks_blob_backed_path(self) -> None:
        source = SourceResolved(
            plugin="json",
            options={
                "path": "/home/someuser/elspeth/data/blobs/sess/abc123_urls.json",
                "blob_ref": "abc123",
                "schema": {"mode": "observed"},
            },
            observed_columns=("url",),
            sample_rows=(),
        )
        turn = build_step_1_schema_form_turn_from_resolved(source, _Catalog())
        assert turn["payload"]["prefilled"]["path"] == "blob:abc123"
        # the absolute path (deploy dir + OS username) is gone from the payload
        assert "/home/someuser" not in str(turn["payload"]["prefilled"])

    def test_step_1_from_resolved_leaves_non_blob_path_untouched(self) -> None:
        source = SourceResolved(
            plugin="json",
            options={"path": "data/input.json", "schema": {"mode": "observed"}},
            observed_columns=("url",),
            sample_rows=(),
        )
        turn = build_step_1_schema_form_turn_from_resolved(source, _Catalog())
        assert turn["payload"]["prefilled"]["path"] == "data/input.json"
