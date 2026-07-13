"""Tests for the Phase-7A reference-content fields on plugin bases.

The new fields live on BaseSource / BaseTransform / BaseSink as class
attributes (matching the existing determinism / plugin_version /
source_file_hash precedent). They default to None or empty so existing
plugins keep working unchanged; authors fill them in when documenting a
plugin for the catalog's reference surface.
"""

from __future__ import annotations

from elspeth.contracts.plugin_capabilities import WebConfigAuthority
from elspeth.plugins.infrastructure.base import BaseSink, BaseSource, BaseTransform


def test_base_source_has_reference_fields() -> None:
    assert BaseSource.usage_when_to_use is None
    assert BaseSource.usage_when_not_to_use is None
    assert BaseSource.example_use is None
    assert BaseSource.capability_tags == ()
    assert BaseSource.audit_characteristics == frozenset()


def test_base_transform_has_reference_fields() -> None:
    assert BaseTransform.usage_when_to_use is None
    assert BaseTransform.usage_when_not_to_use is None
    assert BaseTransform.example_use is None
    assert BaseTransform.capability_tags == ()
    assert BaseTransform.audit_characteristics == frozenset()


def test_base_sink_has_reference_fields() -> None:
    assert BaseSink.usage_when_to_use is None
    assert BaseSink.usage_when_not_to_use is None
    assert BaseSink.example_use is None
    assert BaseSink.capability_tags == ()
    assert BaseSink.audit_characteristics == frozenset()


def test_capability_tags_is_a_tuple_not_a_list() -> None:
    """Tuples are hashable and frozen — they should not be mutable list
    defaults that could surprise the next author."""
    assert isinstance(BaseSource.capability_tags, tuple)
    assert isinstance(BaseTransform.capability_tags, tuple)
    assert isinstance(BaseSink.capability_tags, tuple)


def test_audit_characteristics_is_a_frozenset() -> None:
    """Frozensets prevent accidental mutation of the class default and
    compose cleanly under set-union in the derivation logic."""
    assert isinstance(BaseSource.audit_characteristics, frozenset)
    assert isinstance(BaseTransform.audit_characteristics, frozenset)
    assert isinstance(BaseSink.audit_characteristics, frozenset)


def test_policy_metadata_defaults_are_typed_and_immutable() -> None:
    for base in (BaseSource, BaseTransform, BaseSink):
        assert base.web_config_authority is WebConfigAuthority.USER_CONFIGURABLE
        assert base.policy_capabilities == frozenset()
        assert isinstance(base.policy_capabilities, frozenset)
