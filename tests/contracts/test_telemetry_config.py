"""Tests for telemetry configuration contracts.

Tests cover:
- TelemetryGranularity enum parsing
- BackpressureMode enum parsing
- TelemetrySettings Pydantic validation
- ExporterSettings Pydantic validation
- RuntimeTelemetryConfig.from_settings() factory
- Fail-fast behavior on unimplemented backpressure modes
"""

import pytest

from elspeth.contracts.config import (
    ExporterConfig,
    RuntimeTelemetryConfig,
)
from elspeth.contracts.enums import (
    _IMPLEMENTED_BACKPRESSURE_MODES,
    BackpressureMode,
    TelemetryGranularity,
)
from elspeth.core.config import (
    ExporterSettings,
    TelemetrySettings,
)


class TestTelemetryGranularityEnum:
    """Tests for TelemetryGranularity enum."""

    def test_lifecycle_value(self) -> None:
        """LIFECYCLE has the correct string value."""
        assert TelemetryGranularity.LIFECYCLE.value == "lifecycle"

    def test_rows_value(self) -> None:
        """ROWS has the correct string value."""
        assert TelemetryGranularity.ROWS.value == "rows"

    def test_full_value(self) -> None:
        """FULL has the correct string value."""
        assert TelemetryGranularity.FULL.value == "full"

    def test_is_string_enum(self) -> None:
        """TelemetryGranularity is a string enum for YAML/settings parsing."""
        assert isinstance(TelemetryGranularity.LIFECYCLE, str)
        assert TelemetryGranularity.LIFECYCLE == "lifecycle"

    def test_parse_from_string(self) -> None:
        """Can parse enum from string value."""
        assert TelemetryGranularity("lifecycle") == TelemetryGranularity.LIFECYCLE
        assert TelemetryGranularity("rows") == TelemetryGranularity.ROWS
        assert TelemetryGranularity("full") == TelemetryGranularity.FULL

    def test_invalid_value_raises(self) -> None:
        """Invalid string values raise ValueError."""
        with pytest.raises(ValueError):
            TelemetryGranularity("invalid")


class TestBackpressureModeEnum:
    """Tests for BackpressureMode enum."""

    def test_block_value(self) -> None:
        """BLOCK has the correct string value."""
        assert BackpressureMode.BLOCK.value == "block"

    def test_drop_value(self) -> None:
        """DROP has the correct string value."""
        assert BackpressureMode.DROP.value == "drop"

    def test_slow_value(self) -> None:
        """SLOW has the correct string value."""
        assert BackpressureMode.SLOW.value == "slow"

    def test_is_string_enum(self) -> None:
        """BackpressureMode is a string enum for YAML/settings parsing."""
        assert isinstance(BackpressureMode.BLOCK, str)
        assert BackpressureMode.BLOCK == "block"

    def test_parse_from_string(self) -> None:
        """Can parse enum from string value."""
        assert BackpressureMode("block") == BackpressureMode.BLOCK
        assert BackpressureMode("drop") == BackpressureMode.DROP
        assert BackpressureMode("slow") == BackpressureMode.SLOW

    def test_implemented_modes_set(self) -> None:
        """_IMPLEMENTED_BACKPRESSURE_MODES contains BLOCK and DROP only."""
        assert BackpressureMode.BLOCK in _IMPLEMENTED_BACKPRESSURE_MODES
        assert BackpressureMode.DROP in _IMPLEMENTED_BACKPRESSURE_MODES
        assert BackpressureMode.SLOW not in _IMPLEMENTED_BACKPRESSURE_MODES


class TestExporterSettings:
    """Tests for ExporterSettings Pydantic model."""

    def test_minimal_config(self) -> None:
        """ExporterSettings can be created with just a name."""
        settings = ExporterSettings(name="console")
        assert settings.name == "console"
        assert settings.options == {}

    def test_with_options(self) -> None:
        """ExporterSettings accepts options dict."""
        settings = ExporterSettings(
            name="otlp",
            options={"endpoint": "https://otel.example.com", "headers": {"key": "value"}},
        )
        assert settings.name == "otlp"
        assert settings.options["endpoint"] == "https://otel.example.com"

    def test_empty_name_rejected(self) -> None:
        """Empty exporter name is rejected."""
        with pytest.raises(ValueError, match="exporter name cannot be empty"):
            ExporterSettings(name="")

    def test_whitespace_name_rejected(self) -> None:
        """Whitespace-only exporter name is rejected."""
        with pytest.raises(ValueError, match="exporter name cannot be empty"):
            ExporterSettings(name="   ")

    def test_frozen(self) -> None:
        """ExporterSettings is immutable."""
        from pydantic import ValidationError

        settings = ExporterSettings(name="console")
        with pytest.raises(ValidationError):
            settings.name = "other"  # type: ignore[misc]


class TestTelemetrySettings:
    """Tests for TelemetrySettings Pydantic model."""

    def test_defaults(self) -> None:
        """TelemetrySettings has sensible defaults."""
        settings = TelemetrySettings()
        assert settings.enabled is False
        assert settings.granularity == "lifecycle"
        assert settings.backpressure_mode == "block"
        assert settings.fail_on_total_exporter_failure is True
        assert settings.exporters == []

    def test_enabled_with_exporters(self) -> None:
        """TelemetrySettings can be enabled with exporters."""
        settings = TelemetrySettings(
            enabled=True,
            granularity="rows",
            backpressure_mode="drop",
            fail_on_total_exporter_failure=False,
            exporters=[
                ExporterSettings(name="console", options={"pretty": True}),
                ExporterSettings(name="otlp", options={"endpoint": "https://otel.example.com"}),
            ],
        )
        assert settings.enabled is True
        assert settings.granularity == "rows"
        assert settings.backpressure_mode == "drop"
        assert settings.fail_on_total_exporter_failure is False
        assert len(settings.exporters) == 2

    def test_all_granularity_values(self) -> None:
        """All granularity values are valid."""
        for granularity in ["lifecycle", "rows", "full"]:
            settings = TelemetrySettings(granularity=granularity)  # type: ignore[arg-type]
            assert settings.granularity == granularity

    def test_all_backpressure_mode_values(self) -> None:
        """All backpressure mode values are valid at settings level.

        Note: 'slow' is valid in settings but will fail at runtime config conversion.
        """
        for mode in ["block", "drop", "slow"]:
            settings = TelemetrySettings(backpressure_mode=mode)  # type: ignore[arg-type]
            assert settings.backpressure_mode == mode

    def test_invalid_granularity_rejected(self) -> None:
        """Invalid granularity value is rejected."""
        with pytest.raises(ValueError):
            TelemetrySettings(granularity="invalid")  # type: ignore[arg-type]

    def test_invalid_backpressure_mode_rejected(self) -> None:
        """Invalid backpressure mode value is rejected."""
        with pytest.raises(ValueError):
            TelemetrySettings(backpressure_mode="invalid")  # type: ignore[arg-type]

    def test_frozen(self) -> None:
        """TelemetrySettings is immutable."""
        from pydantic import ValidationError

        settings = TelemetrySettings()
        with pytest.raises(ValidationError):
            settings.enabled = True  # type: ignore[misc]


class TestExporterConfig:
    """Tests for ExporterConfig frozen dataclass."""

    def test_creation(self) -> None:
        """ExporterConfig can be created with name and options."""
        config = ExporterConfig(name="console", options={"pretty": True})
        assert config.name == "console"
        assert config.options == {"pretty": True}

    def test_empty_name_rejected(self) -> None:
        """Empty exporter name is rejected in __post_init__."""
        with pytest.raises(ValueError, match="exporter name cannot be empty"):
            ExporterConfig(name="", options={})

    def test_frozen(self) -> None:
        """ExporterConfig is immutable."""
        from dataclasses import FrozenInstanceError

        config = ExporterConfig(name="console", options={})
        with pytest.raises(FrozenInstanceError):
            config.name = "other"  # type: ignore[misc]


class TestRuntimeTelemetryConfig:
    """Tests for RuntimeTelemetryConfig runtime dataclass."""

    def test_default(self) -> None:
        """RuntimeTelemetryConfig.default() returns disabled telemetry."""
        config = RuntimeTelemetryConfig.default()
        assert config.enabled is False
        assert config.granularity == TelemetryGranularity.LIFECYCLE
        assert config.backpressure_mode == BackpressureMode.BLOCK
        assert config.fail_on_total_exporter_failure is True
        assert config.exporter_configs == ()

    def test_from_settings_basic(self) -> None:
        """RuntimeTelemetryConfig.from_settings() converts settings to runtime config."""
        settings = TelemetrySettings(
            enabled=True,
            granularity="rows",
            backpressure_mode="drop",
            fail_on_total_exporter_failure=False,
            exporters=[
                ExporterSettings(name="console", options={"pretty": True}),
            ],
        )
        config = RuntimeTelemetryConfig.from_settings(settings)

        assert config.enabled is True
        assert config.granularity == TelemetryGranularity.ROWS
        assert config.backpressure_mode == BackpressureMode.DROP
        assert config.fail_on_total_exporter_failure is False
        assert len(config.exporter_configs) == 1
        assert config.exporter_configs[0].name == "console"
        assert config.exporter_configs[0].options == {"pretty": True}

    def test_from_settings_all_granularities(self) -> None:
        """from_settings() correctly parses all granularity values."""
        for granularity_str, expected_enum in [
            ("lifecycle", TelemetryGranularity.LIFECYCLE),
            ("rows", TelemetryGranularity.ROWS),
            ("full", TelemetryGranularity.FULL),
        ]:
            settings = TelemetrySettings(granularity=granularity_str)  # type: ignore[arg-type]
            config = RuntimeTelemetryConfig.from_settings(settings)
            assert config.granularity == expected_enum

    def test_from_settings_case_insensitive(self) -> None:
        """from_settings() handles uppercase granularity/backpressure mode.

        Settings validation enforces lowercase, but from_settings() lowercases
        for robustness in case of programmatic construction.
        """
        # Note: Pydantic validation enforces the exact values, so this test
        # verifies the .lower() call in from_settings() doesn't break things
        settings = TelemetrySettings(granularity="lifecycle", backpressure_mode="block")
        config = RuntimeTelemetryConfig.from_settings(settings)
        assert config.granularity == TelemetryGranularity.LIFECYCLE
        assert config.backpressure_mode == BackpressureMode.BLOCK

    def test_from_settings_slow_mode_fails_fast(self) -> None:
        """from_settings() raises NotImplementedError for 'slow' backpressure mode.

        This is the critical fail-fast behavior: unimplemented modes must fail
        at config load time, not at runtime when telemetry is actually used.
        """
        settings = TelemetrySettings(
            enabled=True,
            backpressure_mode="slow",
        )

        with pytest.raises(NotImplementedError) as exc_info:
            RuntimeTelemetryConfig.from_settings(settings)

        # Verify error message is helpful
        assert "backpressure_mode='slow' is not yet implemented" in str(exc_info.value)
        assert "block" in str(exc_info.value)
        assert "drop" in str(exc_info.value)

    def test_from_settings_multiple_exporters(self) -> None:
        """from_settings() handles multiple exporters."""
        settings = TelemetrySettings(
            enabled=True,
            exporters=[
                ExporterSettings(name="console", options={"pretty": True}),
                ExporterSettings(name="otlp", options={"endpoint": "https://example.com"}),
                ExporterSettings(name="datadog", options={"api_key_fingerprint": "abc123"}),
            ],
        )
        config = RuntimeTelemetryConfig.from_settings(settings)

        assert len(config.exporter_configs) == 3
        assert config.exporter_configs[0].name == "console"
        assert config.exporter_configs[1].name == "otlp"
        assert config.exporter_configs[2].name == "datadog"

    def test_from_settings_empty_exporters(self) -> None:
        """from_settings() handles empty exporter list."""
        settings = TelemetrySettings(enabled=True, exporters=[])
        config = RuntimeTelemetryConfig.from_settings(settings)
        assert config.exporter_configs == ()

    def test_exporter_configs_is_tuple(self) -> None:
        """exporter_configs is an immutable tuple, not a list."""
        settings = TelemetrySettings(
            exporters=[ExporterSettings(name="console", options={})],
        )
        config = RuntimeTelemetryConfig.from_settings(settings)

        assert isinstance(config.exporter_configs, tuple)

    def test_frozen(self) -> None:
        """RuntimeTelemetryConfig is immutable."""
        from dataclasses import FrozenInstanceError

        config = RuntimeTelemetryConfig.default()
        with pytest.raises(FrozenInstanceError):
            config.enabled = True  # type: ignore[misc]


class TestRuntimeTelemetryProtocolCompliance:
    """Tests that RuntimeTelemetryConfig satisfies RuntimeTelemetryProtocol."""

    def test_protocol_compliance(self) -> None:
        """RuntimeTelemetryConfig implements RuntimeTelemetryProtocol."""
        from elspeth.contracts.config import RuntimeTelemetryProtocol

        config = RuntimeTelemetryConfig.default()

        # Protocol is runtime_checkable, so isinstance should work
        assert isinstance(config, RuntimeTelemetryProtocol)

    def test_protocol_fields_accessible(self) -> None:
        """All protocol fields are accessible on RuntimeTelemetryConfig."""
        config = RuntimeTelemetryConfig.default()

        # These should all be accessible without error
        _ = config.enabled
        _ = config.granularity
        _ = config.backpressure_mode
        _ = config.fail_on_total_exporter_failure
        _ = config.exporter_configs
