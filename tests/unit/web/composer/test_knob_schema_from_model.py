from typing import Annotated, Literal

from pydantic import BaseModel, Field

from elspeth.web.catalog.knob_schema import lower_model_to_knob_schema


def test_simple_str_field_lowers_to_text():
    class Opts(BaseModel):
        name: Annotated[str, Field(title="Name", description="Your name")]

    ks = lower_model_to_knob_schema(Opts, plugin_kind="source", plugin_name="test")
    assert len(ks["fields"]) == 1
    f = ks["fields"][0]
    assert f["name"] == "name"
    assert f["label"] == "Name"
    assert f["description"] == "Your name"
    assert f["kind"] == "text"
    assert f["required"] is True
    assert f["nullable"] is False


def test_optional_str_lowers_to_nullable_text():
    class Opts(BaseModel):
        encoding: Annotated[str | None, Field(title="Encoding", description="File encoding")] = None

    ks = lower_model_to_knob_schema(Opts, plugin_kind="source", plugin_name="test")
    f = ks["fields"][0]
    assert f["kind"] == "text"
    assert f["nullable"] is True
    assert f["required"] is False
    assert "default" in f and f["default"] is None


def test_int_with_default_keeps_default():
    class Opts(BaseModel):
        skip_rows: Annotated[int, Field(title="Skip rows", description="Rows to skip")] = 0

    ks = lower_model_to_knob_schema(Opts, plugin_kind="source", plugin_name="test")
    f = ks["fields"][0]
    assert f["kind"] == "number-int"
    assert f["default"] == 0


def test_required_int_omits_default():
    class Opts(BaseModel):
        port: Annotated[int, Field(title="Port", description="TCP port")]

    ks = lower_model_to_knob_schema(Opts, plugin_kind="source", plugin_name="test")
    f = ks["fields"][0]
    assert "default" not in f


def test_literal_lowers_to_enum():
    class Opts(BaseModel):
        mode: Annotated[Literal["a", "b"], Field(title="Mode", description="Pick one")]

    ks = lower_model_to_knob_schema(Opts, plugin_kind="source", plugin_name="test")
    f = ks["fields"][0]
    assert f["kind"] == "enum"
    assert f["enum"] == ["a", "b"]


def test_tier_annotation_emitted_when_set():
    class Opts(BaseModel):
        debug: Annotated[
            bool,
            Field(
                title="Debug",
                description="Verbose output",
                json_schema_extra={"composer_tier": "advanced"},
            ),
        ] = False

    ks = lower_model_to_knob_schema(Opts, plugin_kind="source", plugin_name="test")
    f = ks["fields"][0]
    assert f["tier"] == "advanced"


def test_tier_absent_when_unannotated():
    class Opts(BaseModel):
        debug: Annotated[bool, Field(title="Debug", description="Verbose output")] = False

    ks = lower_model_to_knob_schema(Opts, plugin_kind="source", plugin_name="test")
    f = ks["fields"][0]
    assert "tier" not in f


def test_string_list_kind_for_list_of_str():
    class Opts(BaseModel):
        tags: Annotated[list[str], Field(title="Tags", description="Tag list")] = []

    ks = lower_model_to_knob_schema(Opts, plugin_kind="source", plugin_name="test")
    f = ks["fields"][0]
    assert f["kind"] == "string-list"
    assert f["item_kind"] == "text"


def test_object_map_lowers_to_json_object():
    class Opts(BaseModel):
        weird: Annotated[dict[str, int], Field(title="W", description="W")] = {}

    ks = lower_model_to_knob_schema(Opts, plugin_kind="source", plugin_name="test")
    f = ks["fields"][0]
    assert f["kind"] == "json-object"


def test_non_string_array_lowers_to_json_array():
    class Opts(BaseModel):
        rows: Annotated[list[dict[str, str]], Field(title="Rows", description="Rows")] = []

    ks = lower_model_to_knob_schema(Opts, plugin_kind="source", plugin_name="test")
    f = ks["fields"][0]
    assert f["kind"] == "json-array"


def test_optional_int_clear_round_trips_through_set_source_validator():
    from elspeth.web.composer.redaction import SetSourceArgumentsModel

    payload = {
        "plugin": "csv",
        "on_success": "continue",
        "on_validation_failure": "quarantine",
        "options": {"schema": {"mode": "observed"}, "skip_rows": None},
    }
    validated = SetSourceArgumentsModel.model_validate(payload)
    assert validated.options["skip_rows"] is None


def test_optional_str_absent_round_trips_through_set_source_validator():
    from elspeth.web.composer.redaction import SetSourceArgumentsModel

    payload = {
        "plugin": "csv",
        "on_success": "continue",
        "on_validation_failure": "quarantine",
        "options": {"schema": {"mode": "observed"}},
    }
    validated = SetSourceArgumentsModel.model_validate(payload)
    assert "encoding" not in validated.options


def test_optional_str_clear_round_trips_through_set_source_validator():
    from elspeth.web.composer.redaction import SetSourceArgumentsModel

    payload = {
        "plugin": "csv",
        "on_success": "continue",
        "on_validation_failure": "quarantine",
        "options": {"schema": {"mode": "observed"}, "encoding": None},
    }
    validated = SetSourceArgumentsModel.model_validate(payload)
    assert validated.options["encoding"] is None
