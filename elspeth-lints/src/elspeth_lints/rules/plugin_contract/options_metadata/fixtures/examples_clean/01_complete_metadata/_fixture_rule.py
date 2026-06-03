from __future__ import annotations

from pydantic import BaseModel, Field

from elspeth_lints.rules.plugin_contract.options_metadata.rule import OptionsMetadataRule


class _Options(BaseModel):
    complete: str = Field(title="Complete", description="Complete metadata")


class _Source:
    name = "metadata_ok"
    config_model = _Options


class _Manager:
    def get_sources(self) -> list[type]:
        return [_Source]

    def get_transforms(self) -> list[type]:
        return []

    def get_sinks(self) -> list[type]:
        return []


RULE = OptionsMetadataRule(plugin_manager_factory=_Manager)
