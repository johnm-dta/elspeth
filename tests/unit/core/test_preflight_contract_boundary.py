"""Boundary tests for preflight result contracts."""

from __future__ import annotations

from importlib import import_module


def test_preflight_result_dtos_live_in_contracts_not_dependency_config() -> None:
    preflight_contracts = import_module("elspeth.contracts.preflight")
    dependency_config = import_module("elspeth.core.dependency_config")

    for name in ("DependencyRunResult", "CommencementGateResult", "PreflightResult"):
        assert hasattr(preflight_contracts, name)
        assert not hasattr(dependency_config, name)


def test_dependency_config_keeps_preflight_declarations() -> None:
    dependency_config = import_module("elspeth.core.dependency_config")

    for name in ("DependencyConfig", "CommencementGateConfig", "CollectionProbeConfig"):
        assert hasattr(dependency_config, name)
