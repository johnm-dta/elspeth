# tests/unit/telemetry/test_factory.py
"""Unit tests for telemetry factory functions.

Tests cover:
- Creating TelemetryManager from RuntimeTelemetryConfig
- Disabled telemetry returns None
- Unknown exporter raises error
- Multiple exporters are configured correctly
"""

import pytest

from elspeth.contracts.config import ExporterConfig, RuntimeTelemetryConfig
from elspeth.contracts.enums import BackpressureMode, TelemetryGranularity
from elspeth.telemetry.errors import TelemetryExporterError
from elspeth.telemetry.factory import create_telemetry_manager


class TestCreateTelemetryManager:
    """Tests for create_telemetry_manager factory function."""

    def test_returns_none_when_disabled(self) -> None:
        """Returns None when telemetry is disabled."""
        config = RuntimeTelemetryConfig(
            enabled=False,
            granularity=TelemetryGranularity.LIFECYCLE,
            backpressure_mode=BackpressureMode.BLOCK,
            fail_on_total_exporter_failure=True,
            exporter_configs=(),
        )

        result = create_telemetry_manager(config)

        assert result is None

    def test_creates_manager_with_console_exporter(self) -> None:
        """Creates TelemetryManager with configured console exporter."""
        config = RuntimeTelemetryConfig(
            enabled=True,
            granularity=TelemetryGranularity.FULL,
            backpressure_mode=BackpressureMode.BLOCK,
            fail_on_total_exporter_failure=False,
            exporter_configs=(ExporterConfig(name="console", options={"format": "pretty"}),),
        )

        result = create_telemetry_manager(config)
        try:
            assert result is not None
            assert len(result._exporters) == 1
            assert result._exporters[0].name == "console"
        finally:
            if result is not None:
                result.close()

    def test_creates_manager_with_multiple_exporters(self) -> None:
        """Creates TelemetryManager with multiple exporters."""
        config = RuntimeTelemetryConfig(
            enabled=True,
            granularity=TelemetryGranularity.ROWS,
            backpressure_mode=BackpressureMode.DROP,
            fail_on_total_exporter_failure=False,
            exporter_configs=(
                ExporterConfig(name="console", options={}),
                ExporterConfig(name="otlp", options={"endpoint": "http://localhost:4317"}),
            ),
        )

        result = create_telemetry_manager(config)
        try:
            assert result is not None
            assert len(result._exporters) == 2
            exporter_names = {e.name for e in result._exporters}
            assert exporter_names == {"console", "otlp"}
        finally:
            if result is not None:
                result.close()

    def test_raises_on_unknown_exporter(self) -> None:
        """Raises TelemetryExporterError for unknown exporter name."""
        config = RuntimeTelemetryConfig(
            enabled=True,
            granularity=TelemetryGranularity.LIFECYCLE,
            backpressure_mode=BackpressureMode.BLOCK,
            fail_on_total_exporter_failure=True,
            exporter_configs=(ExporterConfig(name="nonexistent", options={}),),
        )

        with pytest.raises(TelemetryExporterError, match=r"nonexistent.*Unknown exporter"):
            create_telemetry_manager(config)

    def test_passes_config_to_manager(self) -> None:
        """Passes RuntimeTelemetryConfig to TelemetryManager."""
        config = RuntimeTelemetryConfig(
            enabled=True,
            granularity=TelemetryGranularity.FULL,
            backpressure_mode=BackpressureMode.DROP,
            fail_on_total_exporter_failure=True,
            exporter_configs=(ExporterConfig(name="console", options={}),),
        )

        result = create_telemetry_manager(config)
        try:
            assert result is not None
            assert result._config is config
        finally:
            if result is not None:
                result.close()

    def test_creates_manager_with_no_exporters_when_enabled(self) -> None:
        """Creates TelemetryManager even with no exporters (warns but doesn't fail)."""
        config = RuntimeTelemetryConfig(
            enabled=True,
            granularity=TelemetryGranularity.LIFECYCLE,
            backpressure_mode=BackpressureMode.BLOCK,
            fail_on_total_exporter_failure=False,
            exporter_configs=(),  # No exporters configured
        )

        result = create_telemetry_manager(config)
        try:
            assert result is not None
            assert len(result._exporters) == 0
        finally:
            if result is not None:
                result.close()
