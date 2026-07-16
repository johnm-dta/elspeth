"""Runtime plugin construction helpers.

This module is L3-neutral plugin infrastructure: CLI and web entry points can
use it to build plugin instances, while L2 engine code continues to consume
already-instantiated primitives.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

from elspeth.contracts import SinkProtocol, SourceProtocol, TransformProtocol
from elspeth.contracts.freeze import freeze_fields
from elspeth.contracts.hashing import stable_hash
from elspeth.contracts.sink_effects import (
    AuditExportFormat,
    ResolvedSinkEffectMode,
    SinkEffectExecutionPurpose,
    SinkEffectRuntimeBinding,
)
from elspeth.engine.orchestrator.preflight import (
    validate_audit_export_sink_type_capability,
    validate_sink_effect_type_capability,
)

if TYPE_CHECKING:
    from elspeth.core.config import AggregationSettings, ElspethSettings, LandscapeExportSettings, SourceSettings, TransformSettings
    from elspeth.core.dag.wiring import WiredTransform


@dataclass(frozen=True, slots=True)
class PluginBundle:
    """Pre-instantiated plugin instances from configuration.

    Typed fields enable mypy checking and IDE autocomplete on all access sites.

    Per ADR-025 §1, the source surface is plural by contract. Callers iterate
    ``sources`` / ``source_settings_map`` keyed by source name; PluginBundle no
    longer exposes a singular ``source`` shim.
    """

    sources: Mapping[str, SourceProtocol]
    source_settings_map: Mapping[str, SourceSettings]
    transforms: Sequence[WiredTransform]
    sinks: Mapping[str, SinkProtocol]
    aggregations: Mapping[str, tuple[TransformProtocol, AggregationSettings]]
    sink_effect_bindings: Mapping[str, SinkEffectRuntimeBinding]

    def __post_init__(self) -> None:
        from elspeth.contracts.errors import OrchestrationInvariantError

        if not self.sources:
            raise OrchestrationInvariantError("PluginBundle requires at least one source")
        if set(self.sources) != set(self.source_settings_map):
            raise OrchestrationInvariantError(
                f"PluginBundle sources and source_settings_map keys must match. "
                f"sources={sorted(self.sources)}, source_settings_map={sorted(self.source_settings_map)}"
            )
        if set(self.sink_effect_bindings) != set(self.sinks):
            raise OrchestrationInvariantError("PluginBundle sink effect bindings must exactly match runtime sinks")
        for sink_name, binding in self.sink_effect_bindings.items():
            if binding.sink_name != sink_name or binding.sink is not self.sinks[sink_name]:
                raise OrchestrationInvariantError("PluginBundle sink effect binding must retain the exact named sink instance")
        freeze_fields(self, "sources", "source_settings_map", "transforms", "sinks", "aggregations", "sink_effect_bindings")

    @property
    def sink_effect_modes(self) -> Mapping[str, str]:
        from types import MappingProxyType

        return MappingProxyType(
            {name: binding.effect_mode.value for name, binding in self.sink_effect_bindings.items() if binding.effect_mode is not None}
        )


def instantiate_plugins_from_config(
    config: ElspethSettings,
    *,
    preflight_mode: bool = False,
    sink_effect_purpose: SinkEffectExecutionPurpose = SinkEffectExecutionPurpose.FRESH,
) -> PluginBundle:
    """Instantiate all plugins from configuration.

    Creates plugin instances before graph construction, enabling schema
    extraction from instance attributes.

    When preflight_mode=True, plugin constructors observe
    plugin_preflight_mode_enabled() and may defer side-effectful client setup
    to lifecycle/operation methods.
    """
    from elspeth.core.dag.wiring import WiredTransform
    from elspeth.plugins.infrastructure.manager import get_shared_plugin_manager
    from elspeth.plugins.infrastructure.preflight import plugin_preflight_mode

    manager = get_shared_plugin_manager()

    with plugin_preflight_mode(preflight_mode):
        source_settings_by_name = config.sources

        sources = {}
        for source_name, source_config in source_settings_by_name.items():
            source_cls = manager.get_source_by_name(source_config.plugin)
            source_instance = source_cls(dict(source_config.options))
            source_instance.on_success = source_config.on_success
            sources[source_name] = source_instance

        transforms: list[WiredTransform] = []
        for plugin_config in config.transforms:
            transform_cls = manager.get_transform_by_name(plugin_config.plugin)
            transform = transform_cls(dict(plugin_config.options))
            transform.on_success = plugin_config.on_success
            transform.on_error = plugin_config.on_error
            transforms.append(WiredTransform(plugin=transform, settings=plugin_config))

        aggregations = {}
        for agg_config in config.aggregations:
            transform_cls = manager.get_transform_by_name(agg_config.plugin)
            transform = transform_cls(dict(agg_config.options))
            transform.on_success = agg_config.on_success
            transform.on_error = agg_config.on_error

            # Aggregations require transforms that can consume multiple rows.
            # A non-batch-aware transform would silently ignore the trigger.
            if not transform.is_batch_aware:
                raise ValueError(
                    f"Aggregation '{agg_config.name}' uses transform '{agg_config.plugin}' "
                    f"which has is_batch_aware=False. Aggregations require batch-aware "
                    f"transforms that can process multiple rows at once. "
                    f"Use a batch-aware transform like 'batch_stats' or 'batch_replicate', "
                    f"or set is_batch_aware=True on your custom transform."
                )

            aggregations[agg_config.name] = (transform, agg_config)

        from elspeth.plugins.infrastructure.base import BaseSink

        sinks = {}
        sink_effect_bindings = {}
        delayed_export_sink = config.landscape.export.sink if config.landscape.export.enabled else None
        for sink_name, sink_config in config.sinks.items():
            sink_cls = manager.get_sink_by_name(sink_config.plugin)
            if sink_name == delayed_export_sink:
                if isinstance(sink_cls, type) and issubclass(sink_cls, BaseSink):
                    options = dict(sink_config.options)
                    config_model = sink_cls.get_config_model(options)
                    if config_model is not None:
                        config_model.from_dict(options, plugin_name=sink_config.plugin)
                continue
            sink = sink_cls(dict(sink_config.options))
            sinks[sink_name] = sink
            sinks[sink_name]._on_write_failure = sink_config.on_write_failure
            resolved_mode = (
                sink_cls._resolve_sink_effect_mode(dict(sink_config.options), purpose=sink_effect_purpose)
                if isinstance(sink_cls, type) and issubclass(sink_cls, BaseSink)
                else None
            )
            if resolved_mode is not None and type(resolved_mode) is not ResolvedSinkEffectMode:
                raise TypeError("Sink _resolve_sink_effect_mode must return ResolvedSinkEffectMode or None")
            sink_effect_bindings[sink_name] = SinkEffectRuntimeBinding(
                sink_name=sink_name,
                sink=sink,
                sink_type=type(sink),
                config_fingerprint=stable_hash(dict(sink_config.options)),
                purpose=sink_effect_purpose,
                effect_mode=resolved_mode,
            )

        bundle = PluginBundle(
            sources=sources,
            source_settings_map=source_settings_by_name,
            transforms=transforms,
            sinks=sinks,
            aggregations=aggregations,
            sink_effect_bindings=sink_effect_bindings,
        )

    # Value-source compliance check. Single source of truth: every entry point
    # that builds a runtime bundle passes through this function, so hand-authored
    # YAML with hallucinated values is rejected at construction time.
    from elspeth.engine.orchestrator.preflight import validate_value_source_compliance

    validate_value_source_compliance(_value_source_wired_transforms(bundle))
    return bundle


def _expand_env_placeholders_for_raw_preflight(
    value: object,
    *,
    deferrable_env_vars: frozenset[str],
) -> tuple[object, bool]:
    """Expand ``${VAR}``/``${VAR:-default}`` for pre-secret-loading preflight checks.

    Mirrors the loader's ``_expand_env_vars`` semantics (same placeholder
    pattern, environment-first resolution, ``:-`` defaults, and unset-variable
    failure) with one difference: a variable named in ``deferrable_env_vars``
    (declared to be populated later by the Key Vault secret-loading phase,
    which overrides any current process value) cannot be known yet, so its
    placeholder is left literal and the returned ``fully_resolved`` flag is
    False. Callers must treat a not-fully-resolved value as
    unvalidatable-yet and defer to the post-expansion gates.

    Deferral is checked before the process environment because the secret
    loader overrides existing environment values: validating against a value
    the loader is about to replace would validate the wrong configuration.

    Returns:
        Tuple of (expanded value, fully_resolved).

    Raises:
        ValueError: A referenced variable is unset, has no default, and is not
            deferrable — the same failure the loader itself would raise later.
    """
    import os
    import re

    from elspeth.core.config import _ENV_VAR_PATTERN

    fully_resolved = True

    def _expand_string(text: str) -> str:
        def replacer(match: re.Match[str]) -> str:
            nonlocal fully_resolved
            var_name = match.group(1)
            default = match.group(2)  # None if no default specified
            if var_name in deferrable_env_vars:
                fully_resolved = False
                return match.group(0)
            env_value = os.environ.get(var_name)
            if env_value is not None:
                return env_value
            if default is not None:
                return default
            raise ValueError(
                f"Required environment variable '{var_name}' is not set. "
                f"Either set the variable or use ${{{var_name}:-default}} syntax for optional values."
            )

        return _ENV_VAR_PATTERN.sub(replacer, text)

    def _expand_value(item: object) -> object:
        if isinstance(item, str):
            return _expand_string(item)
        if isinstance(item, Mapping):
            return {key: _expand_value(entry) for key, entry in item.items()}
        if isinstance(item, list):
            return [_expand_value(entry) for entry in item]
        return item

    return _expand_value(value), fully_resolved


def validate_sink_effect_eligibility_from_raw_config(
    raw_config: Mapping[str, object],
    *,
    purpose: SinkEffectExecutionPurpose,
    expand_env_placeholders: bool = False,
    deferrable_env_vars: frozenset[str] = frozenset(),
) -> Mapping[str, ResolvedSinkEffectMode]:
    """Reject ineligible adapter classes before credentials or constructors.

    This deliberately reads only the bounded ``sinks`` and delayed-export
    selection. Mode interpretation remains adapter-owned through
    ``BaseSink._resolve_sink_effect_mode``; framework code never guesses from
    generic option names.

    When ``expand_env_placeholders`` is True, the loader's supported ``${VAR}``
    / ``${VAR:-default}`` expansion is applied to each selected sink's options
    before configuration-dependent mode, URL, or dialect resolution
    (elspeth-19f2382cf4). Variables named in ``deferrable_env_vars`` are only
    populated by the later secret-loading phase (which overrides any current
    process value), so a sink whose options still reference one is skipped
    here — the post-expansion admission gates re-run adapter mode resolution
    against the fully loaded configuration and remain authoritative. The flag
    defaults to False so in-memory (web-authored) configurations keep treating
    placeholders as literal data rather than host-environment lookups.
    """
    from elspeth.contracts.sink_effects import SinkEffectInputKind
    from elspeth.plugins.infrastructure.base import BaseSink
    from elspeth.plugins.infrastructure.manager import get_shared_plugin_manager

    if not isinstance(purpose, SinkEffectExecutionPurpose):
        raise TypeError("Sink effect eligibility purpose must be exact SinkEffectExecutionPurpose")
    if type(expand_env_placeholders) is not bool:
        raise TypeError("Sink effect eligibility expand_env_placeholders must be an exact bool")
    if type(deferrable_env_vars) is not frozenset or any(type(name) is not str for name in deferrable_env_vars):
        raise TypeError("Sink effect eligibility deferrable_env_vars must be an exact frozenset of strings")
    if deferrable_env_vars and not expand_env_placeholders:
        raise ValueError("Sink effect eligibility deferrable_env_vars requires expand_env_placeholders=True")
    raw_sinks = raw_config.get("sinks")
    if not isinstance(raw_sinks, Mapping):
        raise ValueError("'sinks' must be a mapping/object for sink effect eligibility")

    export_settings = validate_landscape_export_settings_from_raw_config(raw_config)
    delayed_export_name: str | None = None
    if export_settings.enabled and export_settings.sink:
        delayed_export_name = export_settings.sink

    if purpose is SinkEffectExecutionPurpose.AUDIT_EXPORT:
        if delayed_export_name is None:
            raise ValueError("Enabled audit export with a named sink is required for audit-export eligibility")
        selected_names = (delayed_export_name,)
        required_input_kind = SinkEffectInputKind.AUDIT_EXPORT_SNAPSHOT
    else:
        selected_names = tuple(name for name in raw_sinks if name != delayed_export_name)
        required_input_kind = SinkEffectInputKind.PIPELINE_MEMBERS

    manager = None
    modes: dict[str, ResolvedSinkEffectMode] = {}
    for sink_name in selected_names:
        if not isinstance(sink_name, str) or not sink_name:
            raise ValueError("Sink names must be non-empty strings for sink effect eligibility")
        component = raw_sinks[sink_name]
        if not isinstance(component, Mapping):
            raise ValueError(f"Sink {sink_name!r} must be a mapping/object")
        plugin_name = component.get("plugin")
        if not isinstance(plugin_name, str) or not plugin_name:
            raise ValueError(f"Sink {sink_name!r} plugin must be a non-empty string")
        raw_options = component.get("options", {})
        if not isinstance(raw_options, Mapping):
            raise ValueError(f"Sink {sink_name!r} options must be a mapping/object")
        options = dict(raw_options)
        if expand_env_placeholders:
            expanded_options, fully_resolved = _expand_env_placeholders_for_raw_preflight(
                options,
                deferrable_env_vars=deferrable_env_vars,
            )
            if not fully_resolved:
                # These options reference secrets that exist only after the
                # secret-loading phase; this raw gate cannot resolve the
                # adapter mode yet. The post-expansion admission gates re-run
                # adapter mode resolution and remain authoritative.
                continue
            if not isinstance(expanded_options, dict):  # pragma: no cover - dict in, dict out
                raise TypeError("expanded sink options must be a mapping")
            options = expanded_options
        if manager is None:
            manager = get_shared_plugin_manager()
        sink_type = manager.get_sink_by_name(plugin_name)
        mode: ResolvedSinkEffectMode | None = None
        if issubclass(sink_type, BaseSink):
            mode = sink_type._resolve_sink_effect_mode(options, purpose=purpose)
            if mode is not None and type(mode) is not ResolvedSinkEffectMode:
                raise TypeError("Sink _resolve_sink_effect_mode must return ResolvedSinkEffectMode or None")
        validate_sink_effect_type_capability(
            sink_type,
            mode.value if mode is not None else "",
            required_input_kind,
        )
        if purpose is SinkEffectExecutionPurpose.AUDIT_EXPORT:
            validate_audit_export_sink_type_capability(sink_type, AuditExportFormat(export_settings.format))
        assert mode is not None
        modes[sink_name] = mode
    return modes


def validate_landscape_export_settings_from_raw_config(raw_config: Mapping[str, object]) -> LandscapeExportSettings:
    """Normalize bounded raw export settings through the authoritative model."""
    from elspeth.core.config import LandscapeExportSettings

    raw_landscape = raw_config.get("landscape")
    if raw_landscape is None:
        return LandscapeExportSettings()
    if not isinstance(raw_landscape, Mapping):
        raise ValueError("'landscape' must be a mapping/object before sink effect eligibility")
    raw_export = raw_landscape.get("export")
    if raw_export is None:
        return LandscapeExportSettings()
    if not isinstance(raw_export, Mapping):
        raise ValueError("'landscape.export' must be a mapping/object before sink effect eligibility")
    return LandscapeExportSettings.model_validate(dict(raw_export))


def _value_source_wired_transforms(bundle: PluginBundle) -> tuple[WiredTransform, ...]:
    """Return ordinary and aggregation-backed transforms for value-source checks."""
    from elspeth.core.dag.wiring import WiredTransform

    aggregation_transforms = tuple(
        WiredTransform(plugin=transform, settings=cast("TransformSettings", agg_settings))
        for transform, agg_settings in bundle.aggregations.values()
    )
    return (*bundle.transforms, *aggregation_transforms)


def make_sink_factory(config: ElspethSettings) -> Callable[[str], SinkEffectRuntimeBinding]:
    """Create a factory that produces fresh sink instances from config.

    Used by the export phase, which runs after the pipeline's sinks have
    already been closed. The factory creates a new, unstarted instance each time
    it is called. Delayed export sinks are constructed in preflight mode so
    compliant constructors defer credential and client initialization until
    after the export lifecycle capability gate.
    """
    from elspeth.plugins.infrastructure.manager import get_shared_plugin_manager
    from elspeth.plugins.infrastructure.preflight import plugin_preflight_mode

    def factory(sink_name: str) -> SinkEffectRuntimeBinding:
        if sink_name not in config.sinks:
            raise ValueError(f"Export sink '{sink_name}' not found in sink configuration")
        sink_config = config.sinks[sink_name]
        manager = get_shared_plugin_manager()
        sink_cls = manager.get_sink_by_name(sink_config.plugin)
        with plugin_preflight_mode(True):
            sink = sink_cls(dict(sink_config.options))
        sink._on_write_failure = sink_config.on_write_failure
        from elspeth.plugins.infrastructure.base import BaseSink

        resolved_mode = (
            sink_cls._resolve_sink_effect_mode(
                dict(sink_config.options),
                purpose=SinkEffectExecutionPurpose.AUDIT_EXPORT,
            )
            if isinstance(sink_cls, type) and issubclass(sink_cls, BaseSink)
            else None
        )
        if resolved_mode is not None and type(resolved_mode) is not ResolvedSinkEffectMode:
            raise TypeError("Sink _resolve_sink_effect_mode must return ResolvedSinkEffectMode or None")
        return SinkEffectRuntimeBinding(
            sink_name=sink_name,
            sink=sink,
            sink_type=type(sink),
            config_fingerprint=stable_hash(dict(sink_config.options)),
            purpose=SinkEffectExecutionPurpose.AUDIT_EXPORT,
            effect_mode=resolved_mode,
        )

    return factory
