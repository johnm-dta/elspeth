# tests/contracts/config/test_runtime_concurrency.py
"""Tests for RuntimeConcurrencyConfig.

Concurrency-specific tests only. Common tests (frozen, slots, protocol, orphan fields)
are in test_runtime_common.py.
"""

import pytest

# NOTE: Protocol compliance, orphan detection, frozen/slots tests are in test_runtime_common.py


class TestRuntimeConcurrencyAllFields:
    """Verify RuntimeConcurrencyConfig has all expected fields from Settings."""

    def test_has_all_settings_fields(self) -> None:
        """RuntimeConcurrencyConfig must have all ConcurrencySettings fields."""
        from elspeth.contracts.config.runtime import RuntimeConcurrencyConfig

        # ConcurrencySettings has only one field: max_workers
        expected_fields = {"max_workers"}

        actual_fields = set(RuntimeConcurrencyConfig.__dataclass_fields__.keys())

        assert expected_fields == actual_fields, (
            f"RuntimeConcurrencyConfig fields mismatch.\n"
            f"Missing: {expected_fields - actual_fields}\n"
            f"Extra: {actual_fields - expected_fields}"
        )


class TestConcurrencyFieldNameMapping:
    """Verify field name mappings are correct.

    ConcurrencySettings uses same field names as RuntimeConcurrencyConfig,
    so there should be NO entries in FIELD_MAPPINGS.
    """

    def test_no_field_name_mappings_needed(self) -> None:
        """ConcurrencySettings uses same names - no mappings needed."""
        from elspeth.contracts.config import FIELD_MAPPINGS

        # ConcurrencySettings should not have any field mappings
        # because all field names are the same between Settings and Runtime
        mappings = FIELD_MAPPINGS.get("ConcurrencySettings", {})

        assert mappings == {}, f"ConcurrencySettings should have no field mappings (same names), but found: {mappings}"


class TestRuntimeConcurrencyFromSettings:
    """Test from_settings() factory method."""

    def test_from_settings_maps_all_fields(self) -> None:
        """from_settings() must map all Settings fields correctly."""
        from elspeth.contracts.config.runtime import RuntimeConcurrencyConfig
        from elspeth.core.config import ConcurrencySettings

        # Create settings with non-default values
        settings = ConcurrencySettings(max_workers=16)

        config = RuntimeConcurrencyConfig.from_settings(settings)

        # Verify field mapped correctly
        assert config.max_workers == 16, "max_workers not mapped correctly"

    def test_from_settings_with_defaults(self) -> None:
        """from_settings() should handle default values from Settings."""
        from elspeth.contracts.config.runtime import RuntimeConcurrencyConfig
        from elspeth.core.config import ConcurrencySettings

        # Use Settings defaults
        settings = ConcurrencySettings()
        config = RuntimeConcurrencyConfig.from_settings(settings)

        # ConcurrencySettings default: max_workers=4
        assert config.max_workers == 4


class TestRuntimeConcurrencyConvenienceFactories:
    """Test convenience factory methods."""

    def test_default_creates_config_with_default_workers(self) -> None:
        """default() should create a config with default max_workers (4)."""
        from elspeth.contracts.config.runtime import RuntimeConcurrencyConfig

        config = RuntimeConcurrencyConfig.default()

        # Default max_workers should be 4 (same as ConcurrencySettings)
        assert config.max_workers == 4


class TestRuntimeConcurrencyValidation:
    """Test validation in RuntimeConcurrencyConfig."""

    def test_max_workers_must_be_positive(self) -> None:
        """max_workers < 1 should raise ValueError."""
        from elspeth.contracts.config.runtime import RuntimeConcurrencyConfig

        with pytest.raises(ValueError, match="max_workers must be >= 1"):
            RuntimeConcurrencyConfig(max_workers=0)

    def test_max_workers_negative_raises(self) -> None:
        """Negative max_workers should raise ValueError."""
        from elspeth.contracts.config.runtime import RuntimeConcurrencyConfig

        with pytest.raises(ValueError, match="max_workers must be >= 1"):
            RuntimeConcurrencyConfig(max_workers=-1)
