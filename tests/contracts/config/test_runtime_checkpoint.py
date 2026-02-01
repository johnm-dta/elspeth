# tests/contracts/config/test_runtime_checkpoint.py
"""Tests for RuntimeCheckpointConfig.

Checkpoint-specific tests only. Common tests (frozen, slots, protocol, orphan fields)
are in test_runtime_common.py.

Note on frequency field:
    CheckpointSettings.frequency is a Literal["every_row", "every_n", "aggregation_only"]
    RuntimeCheckpointConfig.frequency is an int (checkpoint every N rows):
    - "every_row" -> 1
    - "every_n" -> checkpoint_interval value
    - "aggregation_only" -> 0

    This is documented in FIELD_MAPPINGS but the type transformation
    happens in from_settings(), not through field name mapping.
"""

import pytest

# NOTE: Protocol compliance, orphan detection, frozen/slots tests are in test_runtime_common.py


class TestRuntimeCheckpointAllFields:
    """Verify RuntimeCheckpointConfig has all expected fields from Settings."""

    def test_has_all_settings_fields(self) -> None:
        """RuntimeCheckpointConfig must have all CheckpointSettings fields.

        CheckpointSettings has:
        - enabled: bool
        - frequency: Literal["every_row", "every_n", "aggregation_only"]
        - checkpoint_interval: int | None
        - aggregation_boundaries: bool

        RuntimeCheckpointConfig maps these to:
        - enabled: bool (direct)
        - frequency: int (computed from Settings.frequency + checkpoint_interval)
        - checkpoint_interval: int | None (preserved for full Settings fidelity)
        - aggregation_boundaries: bool (direct)
        """
        from elspeth.contracts.config.runtime import RuntimeCheckpointConfig

        # All 4 fields from CheckpointSettings (preserving all even if not in protocol)
        expected_fields = {
            "enabled",
            "frequency",
            "checkpoint_interval",
            "aggregation_boundaries",
        }

        actual_fields = set(RuntimeCheckpointConfig.__dataclass_fields__.keys())

        assert expected_fields == actual_fields, (
            f"RuntimeCheckpointConfig fields mismatch.\n"
            f"Missing: {expected_fields - actual_fields}\n"
            f"Extra: {actual_fields - expected_fields}"
        )


class TestCheckpointFieldNameMapping:
    """Verify field name mappings are correct.

    CheckpointSettings uses same field names as RuntimeCheckpointConfig,
    so there should be NO entries in FIELD_MAPPINGS.

    Note: The 'frequency' field has a TYPE transformation (Literal -> int)
    but not a NAME change, so it's not in FIELD_MAPPINGS.
    """

    def test_no_field_name_mappings_needed(self) -> None:
        """CheckpointSettings uses same names - no mappings needed."""
        from elspeth.contracts.config import FIELD_MAPPINGS

        # CheckpointSettings should not have any field mappings
        # because all field names are the same between Settings and Runtime
        mappings = FIELD_MAPPINGS.get("CheckpointSettings", {})

        assert mappings == {}, f"CheckpointSettings should have no field mappings (same names), but found: {mappings}"


class TestRuntimeCheckpointFromSettings:
    """Test from_settings() factory method."""

    def test_from_settings_every_row(self) -> None:
        """from_settings() with frequency='every_row' produces frequency=1."""
        from elspeth.contracts.config.runtime import RuntimeCheckpointConfig
        from elspeth.core.config import CheckpointSettings

        settings = CheckpointSettings(
            enabled=True,
            frequency="every_row",
            aggregation_boundaries=True,
        )

        config = RuntimeCheckpointConfig.from_settings(settings)

        assert config.enabled is True
        assert config.frequency == 1, "every_row should map to frequency=1"
        assert config.checkpoint_interval is None
        assert config.aggregation_boundaries is True

    def test_from_settings_every_n(self) -> None:
        """from_settings() with frequency='every_n' uses checkpoint_interval."""
        from elspeth.contracts.config.runtime import RuntimeCheckpointConfig
        from elspeth.core.config import CheckpointSettings

        settings = CheckpointSettings(
            enabled=True,
            frequency="every_n",
            checkpoint_interval=50,
            aggregation_boundaries=False,
        )

        config = RuntimeCheckpointConfig.from_settings(settings)

        assert config.enabled is True
        assert config.frequency == 50, "every_n should map to checkpoint_interval value"
        assert config.checkpoint_interval == 50
        assert config.aggregation_boundaries is False

    def test_from_settings_aggregation_only(self) -> None:
        """from_settings() with frequency='aggregation_only' produces frequency=0."""
        from elspeth.contracts.config.runtime import RuntimeCheckpointConfig
        from elspeth.core.config import CheckpointSettings

        settings = CheckpointSettings(
            enabled=True,
            frequency="aggregation_only",
            aggregation_boundaries=True,
        )

        config = RuntimeCheckpointConfig.from_settings(settings)

        assert config.enabled is True
        assert config.frequency == 0, "aggregation_only should map to frequency=0"
        assert config.checkpoint_interval is None
        assert config.aggregation_boundaries is True

    def test_from_settings_disabled(self) -> None:
        """from_settings() preserves enabled=False."""
        from elspeth.contracts.config.runtime import RuntimeCheckpointConfig
        from elspeth.core.config import CheckpointSettings

        settings = CheckpointSettings(
            enabled=False,
            frequency="every_row",
            aggregation_boundaries=True,
        )

        config = RuntimeCheckpointConfig.from_settings(settings)

        assert config.enabled is False

    def test_from_settings_with_defaults(self) -> None:
        """from_settings() should handle default values from Settings."""
        from elspeth.contracts.config.runtime import RuntimeCheckpointConfig
        from elspeth.core.config import CheckpointSettings

        # Use Settings defaults
        settings = CheckpointSettings()
        config = RuntimeCheckpointConfig.from_settings(settings)

        # CheckpointSettings defaults:
        # enabled=True, frequency="every_row", checkpoint_interval=None, aggregation_boundaries=True
        assert config.enabled is True
        assert config.frequency == 1  # "every_row" -> 1
        assert config.checkpoint_interval is None
        assert config.aggregation_boundaries is True


class TestRuntimeCheckpointConvenienceFactories:
    """Test convenience factory methods."""

    def test_default_creates_enabled_every_row_config(self) -> None:
        """default() should create an enabled every-row checkpoint config.

        This matches CheckpointSettings defaults.
        """
        from elspeth.contracts.config.runtime import RuntimeCheckpointConfig

        config = RuntimeCheckpointConfig.default()

        # Defaults match CheckpointSettings: enabled, every_row (frequency=1), aggregation_boundaries
        assert config.enabled is True
        assert config.frequency == 1  # "every_row" -> 1
        assert config.checkpoint_interval is None
        assert config.aggregation_boundaries is True


class TestRuntimeCheckpointValidation:
    """Test validation in RuntimeCheckpointConfig."""

    def test_frequency_must_be_non_negative(self) -> None:
        """frequency < 0 should raise ValueError."""
        from elspeth.contracts.config.runtime import RuntimeCheckpointConfig

        with pytest.raises(ValueError, match="frequency must be >= 0"):
            RuntimeCheckpointConfig(
                enabled=True,
                frequency=-1,
                checkpoint_interval=None,
                aggregation_boundaries=True,
            )
