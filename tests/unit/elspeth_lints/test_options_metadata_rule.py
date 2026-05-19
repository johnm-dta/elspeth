"""Tests for the plugin options metadata elspeth-lints rule."""

from __future__ import annotations

from pydantic import BaseModel, Field

from elspeth_lints.rules.plugin_contract.options_metadata import RULE
from elspeth_lints.rules.plugin_contract.options_metadata.rule import OptionsMetadataRule, collect_metadata_findings


def test_options_metadata_rule_reports_missing_title() -> None:
    findings = collect_metadata_findings(plugin_manager=_fake_manager_with_missing_title(), allowlist=set(), root=None)

    assert [finding.message for finding in findings] == ["source/metadata_gap:missing_title: missing title"]
    assert findings[0].rule_id == "plugin_contract.options_metadata"
    assert findings[0].fingerprint == "source/metadata_gap:missing_title:missing-title"


def test_options_metadata_rule_reports_missing_description() -> None:
    findings = collect_metadata_findings(plugin_manager=_fake_manager_with_missing_description(), allowlist=set(), root=None)

    assert [finding.message for finding in findings] == ["sink/metadata_gap:missing_description: missing description"]
    assert findings[0].fingerprint == "sink/metadata_gap:missing_description:missing-description"


def test_options_metadata_rule_checks_discriminated_variants() -> None:
    findings = collect_metadata_findings(plugin_manager=_fake_manager_with_broken_variant(), allowlist=set(), root=None)

    assert [finding.message for finding in findings] == ["transform/metadata_variant[alpha]:variant_field: missing title"]


def test_options_metadata_rule_uses_legacy_identifier_allowlist() -> None:
    findings = collect_metadata_findings(
        plugin_manager=_fake_manager_with_missing_title(),
        allowlist={"source/metadata_gap:missing_title"},
        root=None,
    )

    assert findings == []


def test_registered_rule_is_production_rule() -> None:
    assert isinstance(RULE, OptionsMetadataRule)
    assert RULE.id == "plugin_contract.options_metadata"


def _fake_manager_with_missing_title() -> object:
    class _Options(BaseModel):
        missing_title: str = Field(description="Has description but no title")

    class _Source:
        name = "metadata_gap"
        config_model = _Options

    return _FakePluginManager(sources=[_Source])


def _fake_manager_with_missing_description() -> object:
    class _Options(BaseModel):
        missing_description: str = Field(title="Missing description")

    class _Sink:
        name = "metadata_gap"
        config_model = _Options

    return _FakePluginManager(sinks=[_Sink])


def _fake_manager_with_broken_variant() -> object:
    class _AlphaOptions(BaseModel):
        variant_field: str = Field(description="Has description but no title")

    class _Transform:
        name = "metadata_variant"
        config_model = None

        @classmethod
        def discriminated_variants(cls) -> tuple[str, dict[str, type[BaseModel]]]:
            return "provider", {"alpha": _AlphaOptions}

    return _FakePluginManager(transforms=[_Transform])


class _FakePluginManager:
    def __init__(self, *, sources: list[type] | None = None, transforms: list[type] | None = None, sinks: list[type] | None = None) -> None:
        self._sources = sources or []
        self._transforms = transforms or []
        self._sinks = sinks or []

    def get_sources(self) -> list[type]:
        return self._sources

    def get_transforms(self) -> list[type]:
        return self._transforms

    def get_sinks(self) -> list[type]:
        return self._sinks
