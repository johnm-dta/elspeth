"""Tests for telemetry.factory -- TelemetryManager creation from config."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from elspeth.contracts.config.runtime import ExporterConfig, RuntimeTelemetryConfig
from elspeth.contracts.enums import BackpressureMode, TelemetryGranularity
from elspeth.telemetry.errors import TelemetryExporterError
from elspeth.telemetry.exporters import (
    AzureMonitorExporter,
    ConsoleExporter,
    DatadogExporter,
    OTLPExporter,
)
from elspeth.telemetry.factory import _discover_exporter_registry, create_telemetry_manager
from elspeth.telemetry.hookspecs import hookimpl
from elspeth.telemetry.manager import TelemetryManager


def _make_config(
    *,
    enabled: bool = True,
    exporter_configs: tuple[ExporterConfig, ...] = (),
) -> RuntimeTelemetryConfig:
    return RuntimeTelemetryConfig(
        enabled=enabled,
        granularity=TelemetryGranularity.FULL,
        backpressure_mode=BackpressureMode.DROP,
        fail_on_total_exporter_failure=False,
        max_consecutive_failures=10,
        exporter_configs=exporter_configs,
    )


class TestCreateTelemetryManagerDisabled:
    def test_disabled_returns_none(self):
        config = _make_config(enabled=False)
        result = create_telemetry_manager(config)
        assert result is None

    def test_disabled_with_exporters_still_returns_none(self):
        config = _make_config(
            enabled=False,
            exporter_configs=(ExporterConfig(name="console", options={}),),
        )
        result = create_telemetry_manager(config)
        assert result is None

    def test_disabled_does_not_run_discovery(self):
        config = _make_config(
            enabled=False,
            exporter_configs=(ExporterConfig(name="console", options={}),),
        )
        with patch("elspeth.telemetry.factory._discover_exporter_registry") as mock_discover:
            result = create_telemetry_manager(config)
            assert result is None
            mock_discover.assert_not_called()


class TestCreateTelemetryManagerEnabled:
    def test_enabled_no_exporters_returns_manager(self):
        config = _make_config(enabled=True, exporter_configs=())
        manager = create_telemetry_manager(config)
        try:
            assert manager is not None
            assert isinstance(manager, TelemetryManager)
        finally:
            if manager is not None:
                manager.close()

    def test_enabled_with_console_exporter(self):
        config = _make_config(
            enabled=True,
            exporter_configs=(ExporterConfig(name="console", options={}),),
        )
        manager = create_telemetry_manager(config)
        try:
            assert manager is not None
            assert isinstance(manager, TelemetryManager)
        finally:
            if manager is not None:
                manager.close()

    def test_enabled_with_console_exporter_and_options(self):
        config = _make_config(
            enabled=True,
            exporter_configs=(ExporterConfig(name="console", options={"format": "pretty", "output": "stderr"}),),
        )
        manager = create_telemetry_manager(config)
        try:
            assert manager is not None
            assert isinstance(manager, TelemetryManager)
        finally:
            if manager is not None:
                manager.close()

    def test_exporter_configure_called_with_options(self):
        mock_exporter = MagicMock()
        mock_class = MagicMock(return_value=mock_exporter)
        options = {"endpoint": "http://localhost:4317"}

        with patch("elspeth.telemetry.factory._discover_exporter_registry", return_value={"mock_exp": mock_class}):
            config = _make_config(
                enabled=True,
                exporter_configs=(ExporterConfig(name="mock_exp", options=options),),
            )
            manager = create_telemetry_manager(config)
            try:
                assert manager is not None
                mock_class.assert_called_once()
                mock_exporter.configure.assert_called_once_with(options)
            finally:
                if manager is not None:
                    manager.close()

    def test_exporter_configure_called_with_empty_options(self):
        mock_exporter = MagicMock()
        mock_class = MagicMock(return_value=mock_exporter)

        with patch("elspeth.telemetry.factory._discover_exporter_registry", return_value={"mock_exp": mock_class}):
            config = _make_config(
                enabled=True,
                exporter_configs=(ExporterConfig(name="mock_exp", options={}),),
            )
            manager = create_telemetry_manager(config)
            try:
                assert manager is not None
                mock_exporter.configure.assert_called_once_with({})
            finally:
                if manager is not None:
                    manager.close()

    def test_multiple_exporters_all_configured(self):
        mock_exporter_a = MagicMock()
        mock_class_a = MagicMock(return_value=mock_exporter_a)
        mock_exporter_b = MagicMock()
        mock_class_b = MagicMock(return_value=mock_exporter_b)

        with patch(
            "elspeth.telemetry.factory._discover_exporter_registry",
            return_value={"exp_a": mock_class_a, "exp_b": mock_class_b},
        ):
            config = _make_config(
                enabled=True,
                exporter_configs=(
                    ExporterConfig(name="exp_a", options={"key": "val_a"}),
                    ExporterConfig(name="exp_b", options={"key": "val_b"}),
                ),
            )
            manager = create_telemetry_manager(config)
            try:
                assert manager is not None
                mock_class_a.assert_called_once()
                mock_class_b.assert_called_once()
                mock_exporter_a.configure.assert_called_once_with({"key": "val_a"})
                mock_exporter_b.configure.assert_called_once_with({"key": "val_b"})
            finally:
                if manager is not None:
                    manager.close()

    def test_exporters_instantiated_in_order(self):
        call_order: list[str] = []

        def make_mock(label: str) -> MagicMock:
            mock = MagicMock()
            mock.configure.side_effect = lambda _opts: call_order.append(label)
            return mock

        mock_a = make_mock("first")
        mock_b = make_mock("second")
        mock_c = make_mock("third")

        with patch(
            "elspeth.telemetry.factory._discover_exporter_registry",
            return_value={"ea": MagicMock(return_value=mock_a), "eb": MagicMock(return_value=mock_b), "ec": MagicMock(return_value=mock_c)},
        ):
            config = _make_config(
                enabled=True,
                exporter_configs=(
                    ExporterConfig(name="ea", options={}),
                    ExporterConfig(name="eb", options={}),
                    ExporterConfig(name="ec", options={}),
                ),
            )
            manager = create_telemetry_manager(config)
            try:
                assert call_order == ["first", "second", "third"]
            finally:
                if manager is not None:
                    manager.close()

    def test_returned_manager_receives_config(self):
        mock_exporter = MagicMock()
        mock_class = MagicMock(return_value=mock_exporter)

        with patch("elspeth.telemetry.factory._discover_exporter_registry", return_value={"mock_exp": mock_class}):
            config = _make_config(
                enabled=True,
                exporter_configs=(ExporterConfig(name="mock_exp", options={}),),
            )
            manager = create_telemetry_manager(config)
            try:
                assert manager is not None
                assert manager._config is config
            finally:
                if manager is not None:
                    manager.close()

    def test_returned_manager_receives_exporters_list(self):
        mock_exporter = MagicMock()
        mock_class = MagicMock(return_value=mock_exporter)

        with patch("elspeth.telemetry.factory._discover_exporter_registry", return_value={"mock_exp": mock_class}):
            config = _make_config(
                enabled=True,
                exporter_configs=(ExporterConfig(name="mock_exp", options={}),),
            )
            manager = create_telemetry_manager(config)
            try:
                assert manager is not None
                assert len(manager._exporters) == 1
                assert manager._exporters[0] is mock_exporter
            finally:
                if manager is not None:
                    manager.close()


class TestHookDiscovery:
    def test_custom_hook_exporter_is_discovered(self):
        class CustomExporter:
            _name = "custom_exporter"

            @property
            def name(self) -> str:
                return self._name

            def configure(self, config: dict[str, object]) -> None:
                self._config = config

            def export(self, event: object) -> None:
                return None

            def flush(self) -> None:
                return None

            def close(self) -> None:
                return None

        class CustomPlugin:
            @hookimpl
            def elspeth_get_exporters(self) -> list[type[CustomExporter]]:
                return [CustomExporter]

        config = _make_config(
            enabled=True,
            exporter_configs=(ExporterConfig(name="custom_exporter", options={"k": "v"}),),
        )

        manager = create_telemetry_manager(config, exporter_plugins=(CustomPlugin(),))
        try:
            assert manager is not None
            assert len(manager._exporters) == 1
            assert manager._exporters[0].name == "custom_exporter"
        finally:
            manager.close()

    def test_duplicate_exporter_names_across_hooks_raise(self):
        class ExporterA:
            _name = "dup"

            @property
            def name(self) -> str:
                return self._name

            def configure(self, config: dict[str, object]) -> None:
                return None

            def export(self, event: object) -> None:
                return None

            def flush(self) -> None:
                return None

            def close(self) -> None:
                return None

        class ExporterB:
            _name = "dup"

            @property
            def name(self) -> str:
                return self._name

            def configure(self, config: dict[str, object]) -> None:
                return None

            def export(self, event: object) -> None:
                return None

            def flush(self) -> None:
                return None

            def close(self) -> None:
                return None

        class PluginA:
            @hookimpl
            def elspeth_get_exporters(self) -> list[type[ExporterA]]:
                return [ExporterA]

        class PluginB:
            @hookimpl
            def elspeth_get_exporters(self) -> list[type[ExporterB]]:
                return [ExporterB]

        config = _make_config(enabled=True, exporter_configs=())

        with pytest.raises(TelemetryExporterError, match="Duplicate telemetry exporter name"):
            create_telemetry_manager(config, exporter_plugins=(PluginA(), PluginB()))

    def test_invalid_exporter_plugin_hook_raises(self):
        class InvalidPlugin:
            @hookimpl
            def elspeth_get_exporter(self) -> list[type]:  # pragma: no cover - typo under test
                return []

        config = _make_config(enabled=True, exporter_configs=())

        with pytest.raises(TelemetryExporterError, match="Invalid telemetry exporter plugin"):
            create_telemetry_manager(config, exporter_plugins=(InvalidPlugin(),))

    def test_hook_returning_none_raises_actionable_error(self):
        class NoneReturningPlugin:
            @hookimpl
            def elspeth_get_exporters(self):  # type: ignore[no-untyped-def]
                return None

        config = _make_config(enabled=True, exporter_configs=())

        with pytest.raises(TelemetryExporterError, match="returned None"):
            create_telemetry_manager(config, exporter_plugins=(NoneReturningPlugin(),))

    def test_hook_returning_non_iterable_raises_actionable_error(self):
        class NonIterablePlugin:
            @hookimpl
            def elspeth_get_exporters(self):  # type: ignore[no-untyped-def]
                return 42

        config = _make_config(enabled=True, exporter_configs=())

        with pytest.raises(TelemetryExporterError, match="returned int"):
            create_telemetry_manager(config, exporter_plugins=(NonIterablePlugin(),))


class TestUnknownExporter:
    def test_unknown_exporter_raises_error(self):
        config = _make_config(
            enabled=True,
            exporter_configs=(ExporterConfig(name="nonexistent_exporter", options={}),),
        )
        with pytest.raises(TelemetryExporterError) as exc_info:
            create_telemetry_manager(config)
        assert exc_info.value.exporter_name == "nonexistent_exporter"

    def test_unknown_exporter_error_mentions_available(self):
        config = _make_config(
            enabled=True,
            exporter_configs=(ExporterConfig(name="bad_name", options={}),),
        )
        with pytest.raises(TelemetryExporterError) as exc_info:
            create_telemetry_manager(config)
        error_message = str(exc_info.value)
        assert "console" in error_message
        assert "otlp" in error_message
        assert "azure_monitor" in error_message
        assert "datadog" in error_message

    def test_unknown_exporter_error_message_contains_unknown(self):
        config = _make_config(
            enabled=True,
            exporter_configs=(ExporterConfig(name="phantom", options={}),),
        )
        with pytest.raises(TelemetryExporterError) as exc_info:
            create_telemetry_manager(config)
        assert "Unknown exporter" in exc_info.value.message

    def test_unknown_exporter_does_not_produce_manager(self):
        config = _make_config(
            enabled=True,
            exporter_configs=(
                ExporterConfig(name="console", options={}),
                ExporterConfig(name="does_not_exist", options={}),
            ),
        )
        with pytest.raises(TelemetryExporterError):
            create_telemetry_manager(config)


class TestExporterDiscoveryRegistry:
    def test_registry_contains_console(self):
        registry = _discover_exporter_registry()
        assert "console" in registry

    def test_registry_contains_otlp(self):
        registry = _discover_exporter_registry()
        assert "otlp" in registry

    def test_registry_contains_azure_monitor(self):
        registry = _discover_exporter_registry()
        assert "azure_monitor" in registry

    def test_registry_contains_datadog(self):
        registry = _discover_exporter_registry()
        assert "datadog" in registry

    def test_registry_has_exactly_four_builtin_entries(self):
        registry = _discover_exporter_registry()
        assert len(registry) == 4

    def test_registry_console_is_console_exporter(self):
        registry = _discover_exporter_registry()
        assert registry["console"] is ConsoleExporter

    def test_registry_otlp_is_otlp_exporter(self):
        registry = _discover_exporter_registry()
        assert registry["otlp"] is OTLPExporter

    def test_registry_azure_monitor_is_azure_monitor_exporter(self):
        registry = _discover_exporter_registry()
        assert registry["azure_monitor"] is AzureMonitorExporter

    def test_registry_datadog_is_datadog_exporter(self):
        registry = _discover_exporter_registry()
        assert registry["datadog"] is DatadogExporter

    def test_registry_values_are_types(self):
        registry = _discover_exporter_registry()
        for name, cls in registry.items():
            assert isinstance(cls, type), f"Registry entry '{name}' is not a type: {cls!r}"

    def test_duplicate_plugin_object_raises_telemetry_error(self):
        """Registering the same plugin object twice should raise TelemetryExporterError.

        Regression test: pluggy.PluginManager.register() raises ValueError
        for duplicate plugin objects, which must be wrapped in the function's
        documented TelemetryExporterError contract.
        """

        class DuplicatePlugin:
            @hookimpl
            def elspeth_get_exporters(self) -> list[type]:
                return []

        same_instance = DuplicatePlugin()

        with pytest.raises(TelemetryExporterError, match="Invalid telemetry exporter plugin"):
            _discover_exporter_registry(exporter_plugins=(same_instance, same_instance))
