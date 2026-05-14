from elspeth.web.catalog.knob_schema import (
    KnobField,
    RecipeContext,
    VisibilityPredicate,
    _PluginOptionsPayload,
    _RecipeDecisionPayload,
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


def test_recipe_decision_payload_carries_recipe_context():
    context: RecipeContext = {
        "recipe_name": "classify-rows-llm-jsonl",
        "description": "Classify each row via an LLM",
        "alternatives": ["build_manually"],
    }
    payload: _RecipeDecisionPayload = {
        "mode": "recipe_decision",
        "knobs": {"fields": []},
        "prefilled": {},
        "recipe_context": context,
    }
    assert payload["mode"] == "recipe_decision"


def test_plugin_options_payload_carries_plugin_and_knobs():
    payload: _PluginOptionsPayload = {
        "mode": "plugin_options",
        "plugin": "csv",
        "knobs": {"fields": []},
        "prefilled": {},
    }
    assert payload["plugin"] == "csv"


def test_visibility_predicate_shape():
    predicate: VisibilityPredicate = {"field": "provider", "equals": "azure"}
    assert predicate["field"] == "provider"
