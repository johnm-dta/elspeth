# tests/unit/contracts/config/test_runtime_common.py
"""Common tests for all Runtime*Config dataclasses.

These parametrized tests verify patterns that apply to ALL runtime configs:
1. Frozen dataclass (immutability for thread safety)
2. __slots__ (memory efficiency)
3. Protocol compliance (structural typing via runtime_checkable)
4. No orphan fields (all fields traceable to Settings or INTERNAL_DEFAULTS)

This consolidates duplicate tests from individual Runtime*Config test files.

``RUNTIME_CONFIGS`` is derived from runtime introspection of
``elspeth.contracts.config.runtime``. A new ``Runtime*Config`` frozen
dataclass added in production is automatically picked up by every
parametrised test below — no test-side edit required. Naming convention:

  - Runtime config class name: ``Runtime<Name>Config`` (frozen dataclass)
  - Paired Protocol class name: ``Runtime<Name>Protocol``
  - Paired Settings class name: ``<Name>Settings``
  - Internal defaults key (optional): snake_case of ``<Name>``;
    looked up in ``INTERNAL_DEFAULTS``. Returns ``None`` if the kind has no
    internal defaults.

Discovery raises ``AttributeError`` at module-load / test-collection time
if the protocol or Settings class is missing — making the convention
violation a hard error rather than a silent test gap.
"""

from __future__ import annotations

import dataclasses
import inspect
import re
from dataclasses import FrozenInstanceError
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

import pytest

from elspeth.contracts.config import INTERNAL_DEFAULTS
from elspeth.contracts.config import runtime as _runtime_module

if TYPE_CHECKING:
    pass


# =============================================================================
# TEST CONFIGURATION (derived from runtime introspection)
# =============================================================================


def _is_runtime_config_class(obj: Any) -> bool:
    """Match top-level ``Runtime*Config`` frozen dataclasses in the runtime module."""
    if not inspect.isclass(obj):
        return False
    if not (obj.__name__.startswith("Runtime") and obj.__name__.endswith("Config")):
        return False
    if not dataclasses.is_dataclass(obj):
        return False
    # Frozen dataclasses raise FrozenInstanceError on mutation. Probe the
    # dataclass metadata rather than constructing an instance (some configs
    # require non-trivial fixtures via from_settings/default).
    return obj.__dataclass_params__.frozen  # type: ignore[attr-defined]


def _camel_to_snake(camel: str) -> str:
    """Convert ``RateLimit`` → ``rate_limit`` (single underscore between camel boundaries)."""
    return re.sub(r"(?<!^)(?=[A-Z])", "_", camel).lower()


def _runtime_config_tuple(config_cls: type) -> tuple[str, str, str, str | None]:
    """Return (config_name, protocol_name, settings_name, internal_defaults_key) per convention."""
    config_name = config_cls.__name__
    kind = config_name.removeprefix("Runtime").removesuffix("Config")  # e.g. "RateLimit"

    protocol_name = f"Runtime{kind}Protocol"
    settings_name = f"{kind}Settings"
    internal_key: str | None = _camel_to_snake(kind)
    if internal_key not in INTERNAL_DEFAULTS:
        internal_key = None

    return (config_name, protocol_name, settings_name, internal_key)


def _discover_runtime_configs() -> list[tuple[str, str, str, str | None]]:
    """Return RUNTIME_CONFIGS tuples for every ``Runtime*Config`` in the runtime module."""
    configs = [obj for _name, obj in inspect.getmembers(_runtime_module) if _is_runtime_config_class(obj)]
    # Sort by class name for deterministic test order.
    configs.sort(key=lambda cls: cls.__name__)
    return [_runtime_config_tuple(cls) for cls in configs]


# Each tuple: (config_class_name, protocol_class_name, settings_class_name, internal_defaults_key)
RUNTIME_CONFIGS: list[tuple[str, str, str, str | None]] = _discover_runtime_configs()


def get_config_class(name: str) -> Any:
    """Import and return a RuntimeConfig class by name."""
    from elspeth.contracts.config import runtime

    return getattr(runtime, name)


def get_protocol_class(name: str) -> Any:
    """Import and return a Protocol class by name."""
    from elspeth.contracts import config

    protocol_cls = getattr(config, name, None)
    if protocol_cls is None:
        raise AttributeError(
            f"No Protocol class named {name!r} found. Per the "
            "Runtime<Name>Config → Runtime<Name>Protocol naming convention, "
            "every runtime config must have a paired protocol. Add the "
            "protocol or rename the config."
        )
    return protocol_cls


def get_settings_class(name: str) -> Any:
    """Import and return a Settings class by name.

    Settings classes are in core.config, NOT contracts.config (leaf boundary fix).
    """
    from elspeth.core import config

    settings_cls = getattr(config, name, None)
    if settings_cls is None:
        raise AttributeError(
            f"No Settings class named {name!r} found in elspeth.core.config. "
            "Per the Runtime<Name>Config → <Name>Settings naming convention, "
            "every runtime config must map back to a Pydantic settings class. "
            "Add the Settings class or rename the config."
        )
    return settings_cls


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

        # __slots__ MUST be declared on the class itself, not inherited.
        # `hasattr(cls, "__slots__")` is satisfied by an ancestor's slots,
        # which doesn't constrain *this* class's memory layout. Using
        # `__dict__` lookup ensures the declaration is local.
        assert "__slots__" in config_cls.__dict__, (
            f"{config_name} should declare __slots__ on the class itself (inherited __slots__ does not enforce this class's layout)"
        )


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
        field_mappings: dict[str, str] | MappingProxyType[str, str] = FIELD_MAPPINGS.get(settings_name, {})
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
        field_mappings: dict[str, str] | MappingProxyType[str, str] = FIELD_MAPPINGS.get(settings_name, {})
        runtime_from_settings = {field_mappings.get(f, f) for f in settings_fields}

        # All settings fields must exist in runtime
        missing_fields = runtime_from_settings - runtime_fields

        assert not missing_fields, f"{config_name} is missing Settings fields: {missing_fields}. Add these fields to {config_name}."
