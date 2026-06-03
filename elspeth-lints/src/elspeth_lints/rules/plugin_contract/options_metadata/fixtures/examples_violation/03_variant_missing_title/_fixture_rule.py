from __future__ import annotations

from pydantic import BaseModel, Field

from elspeth_lints.rules.plugin_contract.options_metadata.rule import OptionsMetadataRule


class _AlphaOptions(BaseModel):
    variant_field: str = Field(description="Has description but no title")


class _Transform:
    name = "metadata_variant"
    config_model = None

    @classmethod
    def discriminated_variants(cls) -> tuple[str, dict[str, type[BaseModel]]]:
        return "provider", {"alpha": _AlphaOptions}


class _Manager:
    def get_sources(self) -> list[type]:
        return []

    def get_transforms(self) -> list[type]:
        return [_Transform]

    def get_sinks(self) -> list[type]:
        return []


RULE = OptionsMetadataRule(plugin_manager_factory=_Manager)
