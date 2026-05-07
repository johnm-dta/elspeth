"""Contract tests for LLMTransform.get_config_schema().

The catalog publishes this schema as the authoritative configuration contract
for the ``llm`` transform. It must be a Pydantic discriminated union over
``provider`` so that downstream consumers (MCP composer, frontend PluginCard,
validation pre-flight) see the full required-field sets for every provider
variant at schema-discovery time — not just the base LLMConfig, which is a
thin discriminator anchor with no provider-specific constraints.

Regression coverage for bug elspeth-dcf12c061b.
"""

from __future__ import annotations

from copy import deepcopy
from functools import reduce
from operator import or_
from typing import Annotated

import jsonschema
from pydantic import Field, TypeAdapter

from elspeth.plugins.transforms.llm.providers.azure import AzureOpenAIConfig
from elspeth.plugins.transforms.llm.providers.openrouter import OpenRouterConfig
from elspeth.plugins.transforms.llm.transform import _PROVIDERS, LLMTransform


class TestLLMConfigSchema:
    """LLMTransform.get_config_schema() emits a discriminated union."""

    def test_top_level_shape(self) -> None:
        schema = LLMTransform.get_config_schema()
        assert "oneOf" in schema, "LLM schema must be a discriminated union"
        assert "$defs" in schema, "Each provider variant must live in $defs"
        assert "discriminator" in schema

    def test_discriminator_is_provider(self) -> None:
        schema = LLMTransform.get_config_schema()
        discriminator = schema["discriminator"]
        assert discriminator["propertyName"] == "provider"
        # Every provider registered in the runtime registry must appear in
        # the schema discriminator mapping — registry is the single source
        # of truth for both runtime dispatch and schema publication.
        assert set(discriminator["mapping"].keys()) == set(_PROVIDERS.keys())

    def test_oneof_has_one_branch_per_provider(self) -> None:
        schema = LLMTransform.get_config_schema()
        assert len(schema["oneOf"]) == len(_PROVIDERS)

    def test_azure_variant_publishes_required_provider_fields(self) -> None:
        """Azure-specific mandatory fields must appear in the published schema.

        These are the fields whose omission this bug exposed as invisible to
        catalog consumers: deployment_name, endpoint, api_key.
        """
        schema = LLMTransform.get_config_schema()
        azure = schema["$defs"]["AzureOpenAIConfig"]
        assert set(azure["required"]) >= {"deployment_name", "endpoint", "api_key", "template"}

    def test_openrouter_variant_publishes_required_provider_fields(self) -> None:
        """Mirror of the Azure test for the other provider — fail loudly if the
        OpenRouter variant's published required set drifts from its Pydantic
        model. ``model`` is OpenRouter-specific (Azure uses ``deployment_name``)
        and must appear explicitly in the published contract.
        """
        schema = LLMTransform.get_config_schema()
        openrouter = schema["$defs"]["OpenRouterConfig"]
        assert set(openrouter["required"]) >= {"api_key", "model", "template"}

    def test_matches_typeadapter_fixture(self) -> None:
        """Schema drift detector: compare against an explicit TypeAdapter.

        If providers are added to ``_PROVIDERS`` but LLMTransform's schema
        emission path diverges from the canonical Pydantic construction,
        this test fails as a loud regression signal.

        The emitted schema is the TypeAdapter output with one authored
        deviation: the discriminator field (``provider``) is republished as
        required on every variant, because Pydantic suppresses it from
        ``required`` when a default is set, which silently disagrees with the
        runtime contract. The expected fixture applies that same post-process
        so drift is still detected but this single deliberate difference does
        not trip the comparison.
        """
        variants = tuple(cfg for cfg, _ in _PROVIDERS.values())
        union_type = reduce(or_, variants[1:], variants[0])
        expected = TypeAdapter(Annotated[union_type, Field(discriminator="provider")]).json_schema(ref_template="#/$defs/{model}")
        for variant_cls, _ in _PROVIDERS.values():
            variant_def = expected["$defs"][variant_cls.__name__]
            required = variant_def.setdefault("required", [])
            if "provider" not in required:
                required.append("provider")
        assert LLMTransform.get_config_schema() == expected

    def test_explicit_union_matches(self) -> None:
        """Belt-and-braces: hard-code the current provider set in the fixture.

        If a third provider is added to ``_PROVIDERS``, this test fails and
        forces the author to update the fixture — a deliberate speed bump
        so the schema surface change is reviewed consciously.
        """
        expected = TypeAdapter(Annotated[AzureOpenAIConfig | OpenRouterConfig, Field(discriminator="provider")]).json_schema(
            ref_template="#/$defs/{model}"
        )
        for name in ("AzureOpenAIConfig", "OpenRouterConfig"):
            required = expected["$defs"][name].setdefault("required", [])
            if "provider" not in required:
                required.append("provider")
        assert LLMTransform.get_config_schema() == expected

    def test_provider_is_required_in_every_variant(self) -> None:
        """Regression for PR review P1: discriminator must be declared required.

        Pydantic omits ``provider`` from each variant's ``required`` list
        because the field has a default. The catalog must publish the
        discriminator as required so downstream JSON-Schema consumers reject
        configs that the runtime (``LLMTransform.__init__``) would also reject.
        """
        schema = LLMTransform.get_config_schema()
        for variant_cls, _ in _PROVIDERS.values():
            required = set(schema["$defs"][variant_cls.__name__].get("required", []))
            assert "provider" in required, (
                f"{variant_cls.__name__}.required must include 'provider' so the "
                "published contract matches the runtime dispatch requirement."
            )

    def test_jsonschema_rejects_payload_missing_provider(self) -> None:
        """Regression for PR review P1: runtime/schema contract parity.

        Prior to the fix, a payload that matched exactly one variant's
        non-discriminator fields was accepted by ``jsonschema.validate()``
        because each branch's ``provider`` was optional (default-backed).
        The runtime's ``LLMTransform.__init__`` always rejected such a
        payload (``ValueError: missing required 'provider' key``). This
        asymmetry let the catalog advertise acceptance of configs the
        pipeline would refuse to instantiate. Pin parity here.
        """
        schema = LLMTransform.get_config_schema()
        azure_valid = {
            "provider": "azure",
            "template": "hello",
            "api_key": "k",
            "endpoint": "https://example.invalid/",
            "deployment_name": "d",
            "schema": {"mode": "observed", "fields": None},
        }
        # Sanity: payload with provider is accepted.
        jsonschema.validate(azure_valid, schema)
        # Same payload without provider must now be rejected.
        without_provider = deepcopy(azure_valid)
        del without_provider["provider"]
        try:
            jsonschema.validate(without_provider, schema)
        except jsonschema.ValidationError:
            pass
        else:
            raise AssertionError(
                "Published schema must reject configs missing the 'provider' discriminator to match runtime LLMTransform.__init__ contract."
            )
