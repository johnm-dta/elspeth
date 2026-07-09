"""Tests for telemetry.factory -- TelemetryManager creation from config."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, ClassVar
from unittest.mock import patch

import pytest

from elspeth.contracts.config.runtime import ExporterConfig, RuntimeTelemetryConfig
from elspeth.contracts.enums import BackpressureMode, TelemetryGranularity
from elspeth.telemetry.errors import TelemetryExporterError
from elspeth.telemetry.factory import (
    _discover_exporter_registry,
    _resolve_exporter_name,
    create_telemetry_manager,
)
from elspeth.telemetry.hookspecs import hookimpl
from elspeth.telemetry.manager import TelemetryManager


@dataclass
class _RecordingExporter:
    _name: ClassVar[str] = "recording_exporter"
    _configure_label: ClassVar[str | None] = None
    _configure_order: ClassVar[list[str] | None] = None
    instances: ClassVar[list[_RecordingExporter]] = []

    configured_with: list[Mapping[str, Any]] = field(default_factory=list)

    @property
    def name(self) -> str:
        return type(self)._name

    def __post_init__(self) -> None:
        type(self).instances.append(self)

    def configure(self, config: Mapping[str, Any]) -> None:
        self.configured_with.append(config)
        label = type(self)._configure_label
        order = type(self)._configure_order
        if label is not None and order is not None:
            order.append(label)

    def export(self, event: object) -> None:
        return None

    def flush(self) -> None:
        return None

    def close(self) -> None:
        return None


def _make_recording_exporter_class(
    name: str,
    *,
    configure_label: str | None = None,
    configure_order: list[str] | None = None,
) -> type[_RecordingExporter]:
    class RecordingExporterClass(_RecordingExporter):
        _name: ClassVar[str] = name
        _configure_label: ClassVar[str | None] = configure_label
        _configure_order: ClassVar[list[str] | None] = configure_order
        instances: ClassVar[list[_RecordingExporter]] = []

    return RecordingExporterClass


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
        with patch("elspeth.telemetry.factory._discover_exporter_registry", autospec=True) as mock_discover:
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
        exporter_class = _make_recording_exporter_class("mock_exp")
        options = {"endpoint": "http://localhost:4317"}

        with patch(
            "elspeth.telemetry.factory._discover_exporter_registry",
            autospec=True,
            return_value={"mock_exp": exporter_class},
        ):
            config = _make_config(
                enabled=True,
                exporter_configs=(ExporterConfig(name="mock_exp", options=options),),
            )
            manager = create_telemetry_manager(config)
            try:
                assert manager is not None
                assert len(exporter_class.instances) == 1
                assert exporter_class.instances[0].configured_with == [options]
            finally:
                if manager is not None:
                    manager.close()

    def test_exporter_configure_called_with_empty_options(self):
        exporter_class = _make_recording_exporter_class("mock_exp")

        with patch(
            "elspeth.telemetry.factory._discover_exporter_registry",
            autospec=True,
            return_value={"mock_exp": exporter_class},
        ):
            config = _make_config(
                enabled=True,
                exporter_configs=(ExporterConfig(name="mock_exp", options={}),),
            )
            manager = create_telemetry_manager(config)
            try:
                assert manager is not None
                assert len(exporter_class.instances) == 1
                assert exporter_class.instances[0].configured_with == [{}]
            finally:
                if manager is not None:
                    manager.close()

    def test_multiple_exporters_all_configured(self):
        exporter_class_a = _make_recording_exporter_class("exp_a")
        exporter_class_b = _make_recording_exporter_class("exp_b")

        with patch(
            "elspeth.telemetry.factory._discover_exporter_registry",
            autospec=True,
            return_value={"exp_a": exporter_class_a, "exp_b": exporter_class_b},
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
                assert len(exporter_class_a.instances) == 1
                assert len(exporter_class_b.instances) == 1
                assert exporter_class_a.instances[0].configured_with == [{"key": "val_a"}]
                assert exporter_class_b.instances[0].configured_with == [{"key": "val_b"}]
            finally:
                if manager is not None:
                    manager.close()

    def test_exporters_instantiated_in_order(self):
        call_order: list[str] = []
        exporter_class_a = _make_recording_exporter_class("ea", configure_label="first", configure_order=call_order)
        exporter_class_b = _make_recording_exporter_class("eb", configure_label="second", configure_order=call_order)
        exporter_class_c = _make_recording_exporter_class("ec", configure_label="third", configure_order=call_order)

        with patch(
            "elspeth.telemetry.factory._discover_exporter_registry",
            autospec=True,
            return_value={
                "ea": exporter_class_a,
                "eb": exporter_class_b,
                "ec": exporter_class_c,
            },
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
        exporter_class = _make_recording_exporter_class("mock_exp")

        with patch(
            "elspeth.telemetry.factory._discover_exporter_registry",
            autospec=True,
            return_value={"mock_exp": exporter_class},
        ):
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
        exporter_class = _make_recording_exporter_class("mock_exp")

        with patch(
            "elspeth.telemetry.factory._discover_exporter_registry",
            autospec=True,
            return_value={"mock_exp": exporter_class},
        ):
            config = _make_config(
                enabled=True,
                exporter_configs=(ExporterConfig(name="mock_exp", options={}),),
            )
            manager = create_telemetry_manager(config)
            try:
                assert manager is not None
                assert len(manager._exporters) == 1
                assert manager._exporters[0] is exporter_class.instances[0]
            finally:
                if manager is not None:
                    manager.close()


class TestHookDiscovery:
    def test_custom_hook_exporter_is_discovered(self):
        class CustomExporter:
            _name: str = "custom_exporter"

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
            _name: str = "dup"

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
            _name: str = "dup"

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
            def elspeth_get_exporters(self):
                return None

        config = _make_config(enabled=True, exporter_configs=())

        with pytest.raises(TelemetryExporterError, match="returned None"):
            create_telemetry_manager(config, exporter_plugins=(NoneReturningPlugin(),))

    def test_hook_returning_non_iterable_raises_actionable_error(self):
        class NonIterablePlugin:
            @hookimpl
            def elspeth_get_exporters(self):
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


class TestResolveExporterNameEdgeCases:
    """Tests for _resolve_exporter_name validation of class-level _name attribute."""

    def test_class_name_attribute_non_string_raises(self):
        """_name as non-string (e.g. int) should raise TelemetryExporterError."""

        class BadExporter:
            _name = 42

        with pytest.raises(TelemetryExporterError, match="non-empty string"):
            _resolve_exporter_name(BadExporter)

    def test_class_name_attribute_empty_string_raises(self):
        """_name as empty string should raise TelemetryExporterError."""

        class BadExporter:
            _name = ""

        with pytest.raises(TelemetryExporterError, match="non-empty string"):
            _resolve_exporter_name(BadExporter)

    def test_class_name_attribute_valid_string_returns_it(self):
        """Valid _name class attribute should be returned directly without instantiation."""

        class GoodExporter:
            _name = "my_exporter"

        assert _resolve_exporter_name(GoodExporter) == "my_exporter"


class TestDiscoverExporterRegistryEdgeCases:
    """Tests for _discover_exporter_registry hook return validation."""

    def test_hook_returning_string_raises_actionable_error(self):
        """Hook returning a string instead of list of classes should raise with type info."""

        class StringPlugin:
            @hookimpl
            def elspeth_get_exporters(self):
                return "not_a_list"  # Common mistake: returning name instead of [class]

        with pytest.raises(TelemetryExporterError, match=r"str.*expected iterable"):
            _discover_exporter_registry(exporter_plugins=(StringPlugin(),))

    def test_hook_returning_bytes_raises_actionable_error(self):
        """Hook returning bytes instead of list of classes should raise with type info."""

        class BytesPlugin:
            @hookimpl
            def elspeth_get_exporters(self):
                return b"not_a_list"

        with pytest.raises(TelemetryExporterError, match=r"bytes.*expected iterable"):
            _discover_exporter_registry(exporter_plugins=(BytesPlugin(),))
