from elspeth.web.catalog.knob_schema import (
    KnobField,
    SchemaFormPayload,
    VisibilityPredicate,
)


def test_knob_field_minimal_shape():
    field: KnobField = {
        "name": "path",
        "label": "Input file path",
        "kind": "text",
        "required": True,
        "nullable": False,
    }
    assert field["name"] == "path"


def test_plugin_options_payload_carries_plugin_and_knobs():
    payload: SchemaFormPayload = {
        "mode": "plugin_options",
        "plugin": "csv",
        "knobs": {"fields": []},
        "prefilled": {},
    }
    assert payload["plugin"] == "csv"


def test_visibility_predicate_shape():
    predicate: VisibilityPredicate = {"field": "provider", "equals": "azure"}
    assert predicate["field"] == "provider"
