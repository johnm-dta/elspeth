# src/elspeth/telemetry/factory.py
"""Factory functions for creating TelemetryManager from configuration.

This module provides the glue between configuration (RuntimeTelemetryConfig)
and the runtime TelemetryManager instance. It handles:
1. Discovering exporter classes via telemetry pluggy hooks
2. Instantiating and configuring exporters
3. Creating the TelemetryManager with configured exporters

Usage:
    from elspeth.contracts.config import RuntimeTelemetryConfig
    from elspeth.telemetry.factory import create_telemetry_manager

    config = RuntimeTelemetryConfig.from_settings(settings.telemetry)
    manager = create_telemetry_manager(config)
    # manager is ready to use with Orchestrator
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import pluggy
import structlog

from elspeth.contracts.config import RuntimeTelemetryConfig
from elspeth.telemetry.errors import TelemetryExporterError
from elspeth.telemetry.exporters import BuiltinExportersPlugin
from elspeth.telemetry.hookspecs import PROJECT_NAME, ElspethTelemetrySpec
from elspeth.telemetry.manager import TelemetryManager
from elspeth.telemetry.protocols import ExporterProtocol

logger = structlog.get_logger(__name__)


def _resolve_exporter_name(exporter_class: type[ExporterProtocol]) -> str:
    """Resolve exporter name from class metadata or a temporary instance.

    Args:
        exporter_class: Exporter class returned from hook discovery.

    Returns:
        Exporter name used in telemetry config.

    Raises:
        TelemetryExporterError: If the class cannot be instantiated for name
            resolution or resolves to an invalid name.
    """
    try:
        class_name = exporter_class.__name__
    except AttributeError as e:  # pragma: no cover - defensive boundary for plugin code
        raise TelemetryExporterError(
            "telemetry_plugins",
            f"Invalid exporter declaration without __name__: {exporter_class!r}",
        ) from e

    # Prefer class-level _name when provided to avoid unnecessary instantiation.
    class_dict = exporter_class.__dict__
    if "_name" in class_dict:
        class_name_hint = class_dict["_name"]
        if type(class_name_hint) is str and class_name_hint != "":
            return class_name_hint
        raise TelemetryExporterError(
            class_name,
            f"Exporter class attribute _name must be a non-empty string, got {class_name_hint!r}",
        )

    try:
        exporter_instance = exporter_class()
    except Exception as e:  # pragma: no cover - defensive boundary for plugin code
        raise TelemetryExporterError(
            class_name,
            f"Failed to instantiate exporter class during discovery: {e}",
        ) from e

    resolved_name = exporter_instance.name
    if type(resolved_name) is not str or resolved_name == "":
        raise TelemetryExporterError(
            class_name,
            f"Exporter name must be a non-empty string, got {resolved_name!r}",
        )

    return resolved_name


def _discover_exporter_registry(
    exporter_plugins: Iterable[Any] = (),
) -> dict[str, type[ExporterProtocol]]:
    """Discover telemetry exporters via pluggy hooks.

    Registers built-in exporters plus any additional plugin objects provided
    by the caller, then calls ``elspeth_get_exporters`` hooks to build the
    runtime name->class registry.

    Args:
        exporter_plugins: Optional additional plugin objects implementing
            ``elspeth_get_exporters``.

    Returns:
        Mapping of exporter name to exporter class.

    Raises:
        TelemetryExporterError: If plugin registration fails, exporter names
            are invalid, or duplicate exporter names are discovered.
    """
    plugin_manager = pluggy.PluginManager(PROJECT_NAME)
    plugin_manager.add_hookspecs(ElspethTelemetrySpec)

    plugins_to_register: list[Any] = [BuiltinExportersPlugin(), *list(exporter_plugins)]
    for plugin in plugins_to_register:
        try:
            plugin_manager.register(plugin)
            plugin_manager.check_pending()
        except (pluggy.PluginValidationError, ValueError) as e:
            # PluginValidationError: hook spec mismatch (wrong method names, etc.)
            # ValueError: duplicate plugin object or plugin name already registered
            if isinstance(e, pluggy.PluginValidationError):
                plugin_manager.unregister(plugin=plugin)
            raise TelemetryExporterError(
                "telemetry_plugins",
                f"Invalid telemetry exporter plugin {type(plugin).__name__}: {e}",
            ) from e

    registry: dict[str, type[ExporterProtocol]] = {}
    hook_impls = plugin_manager.hook.elspeth_get_exporters.get_hookimpls()
    for hook_impl in hook_impls:
        hook_plugin: Any = hook_impl.plugin
        plugin_name = type(hook_plugin).__name__
        try:
            hook_fn = hook_plugin.elspeth_get_exporters
        except AttributeError as e:
            raise TelemetryExporterError(
                "telemetry_plugins",
                f"Telemetry exporter plugin {plugin_name} is missing callable elspeth_get_exporters hook",
            ) from e
        if not callable(hook_fn):
            raise TelemetryExporterError(
                "telemetry_plugins",
                f"Telemetry exporter plugin {plugin_name} is missing callable elspeth_get_exporters hook",
            )
        try:
            exporters = hook_fn()
        except Exception as e:
            raise TelemetryExporterError(
                "telemetry_plugins",
                f"Telemetry exporter plugin {plugin_name} failed in elspeth_get_exporters: {e}",
            ) from e

        if exporters is None:
            raise TelemetryExporterError(
                "telemetry_plugins",
                f"elspeth_get_exporters in plugin {plugin_name} returned None; expected iterable of exporter classes",
            )
        if type(exporters) in (str, bytes):
            raise TelemetryExporterError(
                "telemetry_plugins",
                f"elspeth_get_exporters in plugin {plugin_name} returned {type(exporters).__name__}; expected iterable of exporter classes",
            )
        try:
            exporter_iter = iter(exporters)
        except TypeError as e:
            raise TelemetryExporterError(
                "telemetry_plugins",
                f"elspeth_get_exporters in plugin {plugin_name} returned {type(exporters).__name__}; expected iterable of exporter classes",
            ) from e

        for exporter_class in exporter_iter:
            exporter_name = _resolve_exporter_name(exporter_class)
            if exporter_name in registry:
                existing = registry[exporter_name].__name__
                duplicate = exporter_class.__name__
                raise TelemetryExporterError(
                    exporter_name,
                    f"Duplicate telemetry exporter name '{exporter_name}' discovered: {existing} and {duplicate}",
                )
            registry[exporter_name] = exporter_class

    return registry


def create_telemetry_manager(
    config: RuntimeTelemetryConfig,
    *,
    exporter_plugins: Iterable[Any] = (),
) -> TelemetryManager | None:
    """Create a TelemetryManager from runtime configuration.

    If telemetry is disabled in config, returns None. Otherwise discovers
    exporters via telemetry hooks, instantiates configured exporters, and
    returns a TelemetryManager ready for use.

    Args:
        config: Runtime telemetry configuration from RuntimeTelemetryConfig.from_settings().
        exporter_plugins: Optional additional exporter plugin objects providing
            ``elspeth_get_exporters`` hooks.

    Returns:
        TelemetryManager instance if telemetry is enabled, None otherwise.

    Raises:
        TelemetryExporterError: If exporter discovery fails, unknown exporter
            names are configured, or exporter configuration fails.
    """
    if not config.enabled:
        logger.debug("telemetry_disabled", reason="config.enabled=False")
        return None

    exporter_registry = _discover_exporter_registry(exporter_plugins)

    # Instantiate and configure exporters
    exporters: list[ExporterProtocol] = []
    for exporter_config in config.exporter_configs:
        # Look up exporter class - raises TelemetryExporterError if unknown
        try:
            exporter_class = exporter_registry[exporter_config.name]
        except KeyError:
            available = sorted(exporter_registry.keys())
            raise TelemetryExporterError(
                exporter_name=exporter_config.name,
                message=f"Unknown exporter. Available exporters: {available}",
            ) from None

        exporter = exporter_class()
        exporter.configure(exporter_config.options)
        exporters.append(exporter)
        logger.debug(
            "exporter_configured",
            exporter=exporter_config.name,
            options_keys=list(exporter_config.options.keys()),
        )

    if not exporters:
        logger.warning(
            "telemetry_enabled_no_exporters",
            message="Telemetry enabled but no exporters configured",
        )

    return TelemetryManager(config, exporters=exporters)
