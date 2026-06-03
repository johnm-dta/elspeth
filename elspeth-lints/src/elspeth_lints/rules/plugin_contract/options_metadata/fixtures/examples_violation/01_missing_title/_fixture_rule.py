from __future__ import annotations

from pydantic import BaseModel, Field

from elspeth_lints.rules.plugin_contract.options_metadata.rule import OptionsMetadataRule


class _Options(BaseModel):
    missing_title: str = Field(description="Has description but no title")


class _Source:
    name = "metadata_gap"
    config_model = _Options


class _Manager:
    def get_sources(self) -> list[type]:
        return [_Source]

    def get_transforms(self) -> list[type]:
        return []

    def get_sinks(self) -> list[type]:
        return []


RULE = OptionsMetadataRule(plugin_manager_factory=_Manager)
