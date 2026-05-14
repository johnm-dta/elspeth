from typing import Annotated, Literal

import pytest
from pydantic import BaseModel, Field

from elspeth.web.catalog.knob_schema import (
    KnobSchemaLoweringError,
    lower_discriminated_to_knob_schema,
)


class _AzureCfg(BaseModel):
    provider: Literal["azure"] = "azure"
    deployment: Annotated[str, Field(title="Deployment", description="Azure deployment name")]


class _OpenRouterCfg(BaseModel):
    provider: Literal["openrouter"] = "openrouter"
    model: Annotated[str, Field(title="Model", description="OpenRouter model id")]


class _StubPlugin:
    @classmethod
    def discriminated_variants(cls):
        return ("provider", {"azure": _AzureCfg, "openrouter": _OpenRouterCfg})


def test_discriminator_emitted_first_as_enum():
    ks = lower_discriminated_to_knob_schema(_StubPlugin, plugin_kind="transform", plugin_name="llm")
    first = ks["fields"][0]
    assert first["name"] == "provider"
    assert first["kind"] == "enum"
    assert set(first["enum"]) == {"azure", "openrouter"}


def test_variant_fields_get_visible_when():
    ks = lower_discriminated_to_knob_schema(_StubPlugin, plugin_kind="transform", plugin_name="llm")
    deployment = next(f for f in ks["fields"] if f["name"] == "deployment")
    assert deployment["visible_when"] == {"field": "provider", "equals": "azure"}
    model = next(f for f in ks["fields"] if f["name"] == "model")
    assert model["visible_when"] == {"field": "provider", "equals": "openrouter"}


def test_non_discriminated_plugin_raises():
    class _NotDiscriminated:
        pass

    with pytest.raises(KnobSchemaLoweringError) as exc:
        lower_discriminated_to_knob_schema(_NotDiscriminated, plugin_kind="transform", plugin_name="x")
    assert "discriminated_variants" in exc.value.constraint


def test_llm_transform_real_lowering():
    from elspeth.plugins.transforms.llm.transform import LLMTransform

    ks = lower_discriminated_to_knob_schema(LLMTransform, plugin_kind="transform", plugin_name="llm")
    assert ks["fields"][0]["name"] == "provider"
    assert ks["fields"][0]["kind"] == "enum"
    for f in ks["fields"][1:]:
        assert "visible_when" in f
        assert f["visible_when"]["field"] == "provider"
