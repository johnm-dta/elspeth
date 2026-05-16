"""Tests for plugin-trust classification."""

from __future__ import annotations

import pytest

from elspeth.web.audit_readiness.trust import (
    EXTERNAL_BOUNDARY_SINKS,
    EXTERNAL_BOUNDARY_TRANSFORMS,
    PluginTrust,
    classify_plugin,
)


def test_source_kind_always_boundary():
    assert classify_plugin("source", "csv") is PluginTrust.BOUNDARY
    assert classify_plugin("source", "json") is PluginTrust.BOUNDARY


def test_external_call_transforms_are_boundary():
    for name in EXTERNAL_BOUNDARY_TRANSFORMS:
        assert classify_plugin("transform", name) is PluginTrust.BOUNDARY


def test_internal_transforms_are_internal():
    assert classify_plugin("transform", "passthrough") is PluginTrust.INTERNAL


def test_external_sinks_are_boundary():
    for name in EXTERNAL_BOUNDARY_SINKS:
        assert classify_plugin("sink", name) is PluginTrust.BOUNDARY


def test_internal_sinks_are_internal():
    assert classify_plugin("sink", "csv") is PluginTrust.INTERNAL


def test_unknown_plugin_kind_raises():
    with pytest.raises(ValueError, match="unknown plugin kind"):
        classify_plugin("gate", "anything")  # type: ignore[arg-type]


def test_external_boundary_transforms_subset_of_catalog():
    """Allowlist drift guard: every entry must resolve via the live catalog."""
    from elspeth.plugins.infrastructure.manager import PluginManager

    pm = PluginManager()
    pm.register_builtin_plugins()
    transform_names = {cls.name for cls in pm.get_transforms()}
    missing = EXTERNAL_BOUNDARY_TRANSFORMS - transform_names
    assert not missing, (
        f"EXTERNAL_BOUNDARY_TRANSFORMS has unregistered plugins: {sorted(missing)}. Either register the plugin or drop the entry."
    )


def test_external_boundary_sinks_subset_of_catalog():
    from elspeth.plugins.infrastructure.manager import PluginManager

    pm = PluginManager()
    pm.register_builtin_plugins()
    sink_names = {cls.name for cls in pm.get_sinks()}
    missing = EXTERNAL_BOUNDARY_SINKS - sink_names
    assert not missing, (
        f"EXTERNAL_BOUNDARY_SINKS has unregistered plugins: {sorted(missing)}. Either register the plugin or drop the entry."
    )


def test_every_external_call_plugin_is_on_allowlist_or_explicitly_excepted():
    """Every Determinism.EXTERNAL_CALL plugin must be on an allowlist or EXTERNAL_CALL_EXCEPTIONS.

    This test FAILS when a new external-call plugin is added without being
    categorised. Keep EXTERNAL_CALL_EXCEPTIONS empty; populate only with
    an explicit written justification.
    """
    from elspeth.contracts.enums import Determinism
    from elspeth.plugins.infrastructure.manager import PluginManager

    EXTERNAL_CALL_EXCEPTIONS: frozenset[str] = frozenset()
    pm = PluginManager()
    pm.register_builtin_plugins()
    external_call_plugins = {
        cls.name for cls in list(pm.get_transforms()) + list(pm.get_sinks()) if cls.determinism is Determinism.EXTERNAL_CALL
    }
    covered = EXTERNAL_BOUNDARY_TRANSFORMS | EXTERNAL_BOUNDARY_SINKS | EXTERNAL_CALL_EXCEPTIONS
    uncategorised = external_call_plugins - covered
    assert not uncategorised, (
        f"External-call plugins not categorised for audit-readiness: "
        f"{sorted(uncategorised)}. Add to an allowlist or EXTERNAL_CALL_EXCEPTIONS."
    )
