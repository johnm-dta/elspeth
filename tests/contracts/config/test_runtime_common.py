# tests/contracts/config/test_runtime_common.py
"""Common tests for all Runtime*Config dataclasses.

These parametrized tests verify patterns that apply to ALL runtime configs:
1. Frozen dataclass (immutability for thread safety)
2. __slots__ (memory efficiency)
3. Protocol compliance (structural typing via runtime_checkable)
4. No orphan fields (all fields traceable to Settings or INTERNAL_DEFAULTS)

This consolidates duplicate tests from individual Runtime*Config test files.
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    pass


# =============================================================================
# TEST CONFIGURATION
# =============================================================================

# Each tuple: (config_class_name, protocol_class_name, settings_class_name, internal_defaults_key)
RUNTIME_CONFIGS = [
    ("RuntimeRetryConfig", "RuntimeRetryProtocol", "RetrySettings", "retry"),
    ("RuntimeConcurrencyConfig", "RuntimeConcurrencyProtocol", "ConcurrencySettings", None),
    ("RuntimeRateLimitConfig", "RuntimeRateLimitProtocol", "RateLimitSettings", None),
    ("RuntimeCheckpointConfig", "RuntimeCheckpointProtocol", "CheckpointSettings", None),
    ("RuntimeTelemetryConfig", "RuntimeTelemetryProtocol", "TelemetrySettings", None),
]


def get_config_class(name: str) -> type:
    """Import and return a RuntimeConfig class by name."""
    from elspeth.contracts.config import runtime

    return getattr(runtime, name)


def get_protocol_class(name: str) -> type:
    """Import and return a Protocol class by name."""
    from elspeth.contracts import config

    return getattr(config, name)


def get_settings_class(name: str) -> type:
    """Import and return a Settings class by name.

    Settings classes are in core.config, NOT contracts.config (leaf boundary fix).
    """
    from elspeth.core import config

    return getattr(config, name)


# =============================================================================
# FROZEN DATACLASS TESTS
# =============================================================================


class TestRuntimeConfigImmutability:
    """All Runtime*Config classes must be frozen (immutable)."""

    @pytest.mark.parametrize(
        "config_name",
        [cfg[0] for cfg in RUNTIME_CONFIGS],
        ids=[cfg[0] for cfg in RUNTIME_CONFIGS],
    )
    def test_frozen_dataclass(self, config_name: str) -> None:
        """Runtime configs are frozen (immutable) for thread safety."""
        config_cls = get_config_class(config_name)
        config = config_cls.default()

        # Get first field name to attempt mutation
        field_name = next(iter(config_cls.__dataclass_fields__.keys()))

        with pytest.raises(FrozenInstanceError):
            setattr(config, field_name, "mutated_value")

    @pytest.mark.parametrize(
        "config_name",
        [cfg[0] for cfg in RUNTIME_CONFIGS],
        ids=[cfg[0] for cfg in RUNTIME_CONFIGS],
    )
    def test_has_slots(self, config_name: str) -> None:
        """Runtime configs use __slots__ for memory efficiency."""
        config_cls = get_config_class(config_name)

        assert hasattr(config_cls, "__slots__"), f"{config_name} should have __slots__"


# =============================================================================
# PROTOCOL COMPLIANCE TESTS
# =============================================================================


class TestRuntimeConfigProtocolCompliance:
    """All Runtime*Config classes must implement their protocols."""

    @pytest.mark.parametrize(
        "config_name,protocol_name",
        [(cfg[0], cfg[1]) for cfg in RUNTIME_CONFIGS],
        ids=[cfg[0] for cfg in RUNTIME_CONFIGS],
    )
    def test_implements_protocol(self, config_name: str, protocol_name: str) -> None:
        """Runtime config implements its protocol (runtime_checkable)."""
        config_cls = get_config_class(config_name)
        protocol_cls = get_protocol_class(protocol_name)

        config = config_cls.default()

        assert isinstance(config, protocol_cls), (
            f"{config_name} does not implement {protocol_name}. Check that all protocol properties are present with correct types."
        )


# =============================================================================
# ORPHAN FIELD DETECTION TESTS
# =============================================================================


class TestRuntimeConfigNoOrphanFields:
    """All Runtime*Config fields must have documented origin."""

    @pytest.mark.parametrize(
        "config_name,settings_name,internal_key",
        [(cfg[0], cfg[2], cfg[3]) for cfg in RUNTIME_CONFIGS],
        ids=[cfg[0] for cfg in RUNTIME_CONFIGS],
    )
    def test_no_orphan_fields(self, config_name: str, settings_name: str, internal_key: str | None) -> None:
        """Every field must come from Settings or INTERNAL_DEFAULTS."""
        from elspeth.contracts.config import FIELD_MAPPINGS, INTERNAL_DEFAULTS

        config_cls = get_config_class(config_name)
        settings_cls = get_settings_class(settings_name)

        # Get all runtime config fields
        runtime_fields = set(config_cls.__dataclass_fields__.keys())

        # Get Settings fields (with their runtime names via mapping)
        settings_fields = set(settings_cls.model_fields.keys())
        field_mappings = FIELD_MAPPINGS.get(settings_name, {})
        runtime_from_settings = {field_mappings.get(f, f) for f in settings_fields}

        # Get internal-only fields if applicable
        internal_fields: set[str] = set()
        if internal_key:
            internal_fields = set(INTERNAL_DEFAULTS.get(internal_key, {}).keys())

        # All runtime fields must be accounted for
        expected_fields = runtime_from_settings | internal_fields
        orphan_fields = runtime_fields - expected_fields

        assert not orphan_fields, (
            f"{config_name} has orphan fields: {orphan_fields}. "
            f"These must be mapped from {settings_name} or documented in INTERNAL_DEFAULTS."
        )

    @pytest.mark.parametrize(
        "config_name,settings_name",
        [(cfg[0], cfg[2]) for cfg in RUNTIME_CONFIGS],
        ids=[cfg[0] for cfg in RUNTIME_CONFIGS],
    )
    def test_no_missing_settings_fields(self, config_name: str, settings_name: str) -> None:
        """All Settings fields must exist in Runtime config."""
        from elspeth.contracts.config import FIELD_MAPPINGS

        config_cls = get_config_class(config_name)
        settings_cls = get_settings_class(settings_name)

        # Get all runtime config fields
        runtime_fields = set(config_cls.__dataclass_fields__.keys())

        # Get Settings fields (with their runtime names via mapping)
        settings_fields = set(settings_cls.model_fields.keys())
        field_mappings = FIELD_MAPPINGS.get(settings_name, {})
        runtime_from_settings = {field_mappings.get(f, f) for f in settings_fields}

        # All settings fields must exist in runtime
        missing_fields = runtime_from_settings - runtime_fields

        assert not missing_fields, f"{config_name} is missing Settings fields: {missing_fields}. Add these fields to {config_name}."
