"""Tests for CatalogServiceImpl._to_summary covering Phase-7A fields.

This file uses a hand-rolled PluginManager fake so the test doesn't
depend on the real plugin registry (which evolves). The fakes mimic
just enough of the protocol surface for _to_summary to work.
"""

from __future__ import annotations

from typing import Any, ClassVar

from elspeth.contracts.enums import AuditCharacteristic, Determinism
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
    config_model = None
    is_batch_aware = False

    @classmethod
    def get_config_schema(cls) -> dict[str, Any]:
        return {}

    @classmethod
    def get_config_model(cls) -> Any:
        return None


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
    # Derived: DETERMINISTIC -> {"deterministic"}; no declared chars.
    # The response model exposes the composed set as a sorted tuple.
    assert s.audit_characteristics == ("deterministic",)


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
    # Composed: declared {"coerce", "quarantine"} + inferred {"io_read"}.
    # quarantine is author-declared, not inferred from instance state.
    assert "coerce" in s.audit_characteristics
    assert "io_read" in s.audit_characteristics
    assert "quarantine" in s.audit_characteristics
