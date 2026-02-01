"""Tests for configuration contracts.

After fixing P2-2026-01-20-contracts-config-reexport-breaks-leaf-boundary,
Settings classes are NO LONGER re-exported from contracts. They must be
imported directly from elspeth.core.config.

This test verifies:
1. Runtime protocols and config classes ARE available in contracts.config
2. Settings classes are NOT in contracts.config (leaf boundary preserved)
3. Settings classes ARE available in core.config
"""

import pytest

from elspeth.contracts import config as contract_config
from elspeth.core import config as core_config

# Settings classes that should ONLY be in core.config (NOT re-exported from contracts)
# This is the fix for P2-2026-01-20 - maintaining contracts as a leaf module
SETTINGS_IN_CORE_ONLY = {
    "AggregationSettings",
    "CheckpointSettings",
    "CoalesceSettings",
    "ConcurrencySettings",
    "DatabaseSettings",
    "ElspethSettings",
    "ExporterSettings",
    "GateSettings",
    "LandscapeExportSettings",
    "LandscapeSettings",
    "PayloadStoreSettings",
    "RateLimitSettings",
    "RetrySettings",
    "ServiceRateLimit",
    "SinkSettings",
    "SourceSettings",
    "TelemetrySettings",
    "TransformSettings",
    "TriggerConfig",
}

# Items defined in the contracts.config subpackage (these ARE exported)
CONTRACTS_CONFIG_ITEMS = {
    # Runtime protocols
    "RuntimeCheckpointProtocol",
    "RuntimeConcurrencyProtocol",
    "RuntimeRateLimitProtocol",
    "RuntimeRetryProtocol",
    "RuntimeTelemetryProtocol",
    # Runtime configuration dataclasses
    "ExporterConfig",
    "RuntimeCheckpointConfig",
    "RuntimeConcurrencyConfig",
    "RuntimeRateLimitConfig",
    "RuntimeRetryConfig",
    "RuntimeTelemetryConfig",
    # Default registries
    "INTERNAL_DEFAULTS",
    "POLICY_DEFAULTS",
    "get_internal_default",
    "get_policy_default",
    # Alignment documentation
    "EXEMPT_SETTINGS",
    "FIELD_MAPPINGS",
    "SETTINGS_TO_RUNTIME",
    "get_runtime_field_name",
    "get_settings_field_name",
    "is_exempt_settings",
}


class TestConfigLeafBoundary:
    """Verify Settings are NOT re-exported from contracts (leaf boundary)."""

    @pytest.mark.parametrize("name", sorted(SETTINGS_IN_CORE_ONLY))
    def test_settings_not_in_contracts_config(self, name: str) -> None:
        """Settings classes must NOT be in contracts.config (leaf boundary fix).

        This is a regression test for P2-2026-01-20. Settings classes were
        previously re-exported from contracts.config, which broke the leaf
        boundary and caused 1,200+ module imports.
        """
        assert not hasattr(contract_config, name), (
            f"{name} should NOT be in contracts.config - import from core.config instead. "
            "Re-exporting Settings would break the leaf boundary (P2-2026-01-20)."
        )

    @pytest.mark.parametrize("name", sorted(SETTINGS_IN_CORE_ONLY))
    def test_settings_available_in_core_config(self, name: str) -> None:
        """Settings classes are available in core.config."""
        assert hasattr(core_config, name), f"{name} should be in core.config"


class TestContractsConfigItems:
    """Verify contracts.config items that ARE exported."""

    @pytest.mark.parametrize("name", sorted(CONTRACTS_CONFIG_ITEMS))
    def test_contracts_config_items_exist(self, name: str) -> None:
        """Items defined in contracts.config subpackage are accessible."""
        item = getattr(contract_config, name)
        assert item is not None

    def test_all_exports_match_expected(self) -> None:
        """All items in __all__ should be in CONTRACTS_CONFIG_ITEMS."""
        all_exports = set(contract_config.__all__)

        missing = all_exports - CONTRACTS_CONFIG_ITEMS
        extra = CONTRACTS_CONFIG_ITEMS - all_exports

        assert not missing, f"Unexpected exports in __all__: {missing}. Add to CONTRACTS_CONFIG_ITEMS or remove from __all__."
        assert not extra, f"Expected items not in __all__: {extra}. Remove from test or add to __all__."


class TestCoreConfigSettings:
    """Tests for Settings classes in core.config."""

    def test_settings_are_pydantic_models(self) -> None:
        """Config types are Pydantic (trust boundary validation)."""
        from pydantic import BaseModel

        from elspeth.core.config import ElspethSettings, SourceSettings

        assert issubclass(ElspethSettings, BaseModel)
        assert issubclass(SourceSettings, BaseModel)

    def test_settings_are_frozen(self) -> None:
        """Config is immutable after construction."""
        from pydantic import ValidationError

        from elspeth.core.config import SourceSettings

        settings = SourceSettings(plugin="csv_local")

        with pytest.raises(ValidationError):
            settings.plugin = "other"  # type: ignore[misc]
