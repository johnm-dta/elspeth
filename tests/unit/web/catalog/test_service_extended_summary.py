"""Tests for CatalogServiceImpl._to_summary covering Phase-7A fields.

This file uses a hand-rolled PluginManager fake so the test doesn't
depend on the real plugin registry (which evolves). The fakes mimic
just enough of the protocol surface for _to_summary to work.
"""

from __future__ import annotations

from typing import Any, ClassVar

from elspeth.contracts.enums import AuditCharacteristic, Determinism
from elspeth.contracts.plugin_assistance import PluginAssistance
from elspeth.contracts.plugin_capabilities import CapabilityDeclaration, WebConfigAuthority
from elspeth.web.catalog.service import CatalogServiceImpl


class _BareTransform:
    """A transform with no reference-content fields filled in.

    Mimics the post-Phase-7A.1 baseline: defaults exist but no author
    has populated them yet. The catalog summary should round-trip the
    defaults as-is.
    """

    name = "bare"
    determinism = Determinism.DETERMINISTIC
    plugin_version = "1.0.0"
    source_file_hash: str | None = None
    usage_when_to_use: ClassVar[str | None] = None
    usage_when_not_to_use: ClassVar[str | None] = None
    example_use: ClassVar[str | None] = None
    capability_tags: ClassVar[tuple[str, ...]] = ()
    audit_characteristics: ClassVar[frozenset[AuditCharacteristic]] = frozenset()
    discovery_secret_requirements: ClassVar[dict[str, tuple[str, ...]]] = {}
    web_config_authority = WebConfigAuthority.USER_CONFIGURABLE
    policy_capabilities: ClassVar[frozenset[CapabilityDeclaration]] = frozenset()
    config_model = None
    is_batch_aware = False

    @classmethod
    def get_config_schema(cls) -> dict[str, Any]:
        return {}

    @classmethod
    def get_config_model(cls) -> Any:
        return None

    @classmethod
    def get_agent_assistance(cls, *, issue_code: str | None = None) -> Any:
        return None

    @classmethod
    def get_post_call_hints(cls, *, tool_name: str, config_snapshot: Any) -> tuple[str, ...]:
        return ()


class _FilledSource:
    """A source with every reference-content field filled in.

    Mimics csv_source.py post-Phase-7A.5 — the canonical example.
    """

    name = "filled"
    determinism = Determinism.IO_READ
    plugin_version = "1.0.0"
    source_file_hash: str | None = None
    usage_when_to_use: ClassVar[str | None] = "When you have a CSV file."
    usage_when_not_to_use: ClassVar[str | None] = "When the data is inline; use chat instead."
    example_use: ClassVar[str | None] = "source:\n  plugin: filled"
    capability_tags: ClassVar[tuple[str, ...]] = ("file", "csv")
    discovery_secret_requirements: ClassVar[dict[str, tuple[str, ...]]] = {}
    web_config_authority = WebConfigAuthority.USER_CONFIGURABLE
    policy_capabilities: ClassVar[frozenset[CapabilityDeclaration]] = frozenset()
    # Declare both "coerce" (Tier-3 boundary trait) and "quarantine"
    # (runtime quarantine routing) explicitly; the catalog service does
    # not infer either, because `_on_validation_failure` is per-instance.
    audit_characteristics: ClassVar[frozenset[AuditCharacteristic]] = frozenset(
        {AuditCharacteristic.COERCE, AuditCharacteristic.QUARANTINE}
    )
    config_model = None

    @classmethod
    def get_config_schema(cls) -> dict[str, Any]:
        return {}

    @classmethod
    def get_config_model(cls) -> Any:
        return None

    @classmethod
    def get_agent_assistance(cls, *, issue_code: str | None = None) -> Any:
        return None

    @classmethod
    def get_post_call_hints(cls, *, tool_name: str, config_snapshot: Any) -> tuple[str, ...]:
        return ()


class _FakePluginManager:
    def __init__(self, sources, transforms, sinks):
        self._sources = sources
        self._transforms = transforms
        self._sinks = sinks

    def get_sources(self):
        return self._sources

    def get_transforms(self):
        return self._transforms

    def get_sinks(self):
        return self._sinks


def test_bare_plugin_summary_uses_defaults() -> None:
    pm = _FakePluginManager(sources=[], transforms=[_BareTransform], sinks=[])
    svc = CatalogServiceImpl(pm)  # type: ignore[arg-type]
    summaries = svc.list_transforms()
    assert len(summaries) == 1
    s = summaries[0]
    assert s.usage_when_to_use is None
    assert s.usage_when_not_to_use is None
    assert s.example_use is None
    assert s.capability_tags == ()
    # _BareTransform.determinism = Determinism.DETERMINISTIC, which is the
    # transform kind-default. The catalog suppresses default-derived flags
    # so an unfilled, default-determinism transform shows nothing.
    assert s.audit_characteristics == ()


class _HintedTransform:
    """A transform that publishes discovery-time composer_hints.

    Exercises the JIT-hints Phase 1 surface: ``_to_summary`` and
    ``_build_schema_info`` must pull the hint tuple out of the plugin's
    ``get_agent_assistance(issue_code=None)`` return.
    """

    name = "hinted"
    determinism = Determinism.DETERMINISTIC
    plugin_version = "1.0.0"
    source_file_hash: str | None = None
    usage_when_to_use: ClassVar[str | None] = None
    usage_when_not_to_use: ClassVar[str | None] = None
    example_use: ClassVar[str | None] = None
    capability_tags: ClassVar[tuple[str, ...]] = ()
    audit_characteristics: ClassVar[frozenset[AuditCharacteristic]] = frozenset()
    discovery_secret_requirements: ClassVar[dict[str, tuple[str, ...]]] = {}
    web_config_authority = WebConfigAuthority.USER_CONFIGURABLE
    policy_capabilities: ClassVar[frozenset[CapabilityDeclaration]] = frozenset()
    config_model = None
    is_batch_aware = False

    @classmethod
    def get_config_schema(cls) -> dict[str, Any]:
        return {}

    @classmethod
    def get_config_model(cls) -> Any:
        return None

    @classmethod
    def get_agent_assistance(cls, *, issue_code: str | None = None) -> PluginAssistance | None:
        if issue_code is None:
            return PluginAssistance(
                plugin_name="hinted",
                issue_code=None,
                summary="A transform with discovery-time hints.",
                composer_hints=(
                    "First imperative.",
                    "Second imperative.",
                ),
            )
        return None

    @classmethod
    def get_post_call_hints(cls, *, tool_name: str, config_snapshot: Any) -> tuple[str, ...]:
        return ()


def test_summary_carries_discovery_composer_hints() -> None:
    """PluginSummary.composer_hints reflects get_agent_assistance(issue_code=None)."""
    pm = _FakePluginManager(sources=[], transforms=[_HintedTransform], sinks=[])
    svc = CatalogServiceImpl(pm)  # type: ignore[arg-type]
    [s] = svc.list_transforms()
    assert s.composer_hints == ("First imperative.", "Second imperative.")


def test_summary_empty_hints_when_plugin_does_not_override() -> None:
    """A plugin that returns None from get_agent_assistance gets empty composer_hints."""
    pm = _FakePluginManager(sources=[], transforms=[_BareTransform], sinks=[])
    svc = CatalogServiceImpl(pm)  # type: ignore[arg-type]
    [s] = svc.list_transforms()
    assert s.composer_hints == ()


def test_schema_info_carries_discovery_composer_hints() -> None:
    """PluginSchemaInfo.composer_hints mirrors PluginSummary.composer_hints."""
    pm = _FakePluginManager(sources=[], transforms=[_HintedTransform], sinks=[])
    svc = CatalogServiceImpl(pm)  # type: ignore[arg-type]
    info = svc.get_schema("transform", "hinted")
    assert info.composer_hints == ("First imperative.", "Second imperative.")


def test_filled_source_summary_propagates_all_fields() -> None:
    pm = _FakePluginManager(sources=[_FilledSource], transforms=[], sinks=[])
    svc = CatalogServiceImpl(pm)  # type: ignore[arg-type]
    summaries = svc.list_sources()
    assert len(summaries) == 1
    s = summaries[0]
    assert s.usage_when_to_use == "When you have a CSV file."
    assert s.usage_when_not_to_use == "When the data is inline; use chat instead."
    assert s.example_use == "source:\n  plugin: filled"
    assert s.capability_tags == ("file", "csv")
    # _FilledSource.determinism = Determinism.IO_READ, which is the source
    # kind-default — the catalog suppresses default-derived flags so
    # `io_read` is NOT inferred. Only author-declared flags remain.
    assert "coerce" in s.audit_characteristics
    assert "io_read" not in s.audit_characteristics
    assert "quarantine" in s.audit_characteristics
