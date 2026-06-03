"""Single-source-of-truth pipeline assembly + route-target preflight.

Both the composer ``/validate`` endpoint and the execution service's run path
must assemble :class:`PipelineConfig` identically and run the same four
route-target validators that the orchestrator runs at run-init. This module
owns that contract.

**Layer note.** This module sits in ``engine/`` (L2). It cannot import
``cli_helpers`` (L3) and therefore takes primitives instead of ``PluginBundle``.
Both call sites unpack their bundle locally before calling.

**Mutation note.** Aggregation transforms have ``node_id`` assigned in this
helper (mirrors ``service.py`` runtime path). The mutation is intentional —
the orchestrator depends on ``node_id`` being set before run, and folding
aggregations into the transforms list is the canonical wiring step. Reorder
with care.

**Idempotency note.** The four validators are pure (no I/O, no mutation of
their inputs). The orchestrator runs them again at run-init
(``core.py:1746-1777`` and the resume site at ``:1940-1967``). Calling them
here in the composer/service surfaces failures earlier with a cleaner error
surface; the second call at run-init either passes again or raises the same
error.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from types import SimpleNamespace
from typing import TYPE_CHECKING, cast

from elspeth.contracts.types import AggregationName
from elspeth.contracts.value_source import (
    CatalogValueSource,
    DerivedFromSiblingValueSource,
    UnknownCatalogIdError,
    ValueSource,
    find_value_source_config,
    get_catalog_missing_dep_hint,
    get_catalog_values,
)
from elspeth.core.config import resolve_config
from elspeth.engine.orchestrator.types import (
    PipelineConfig,
    ValueSourceFinding,
    ValueSourceValidationError,
)
from elspeth.engine.orchestrator.validation import (
    validate_route_destinations,
    validate_sink_failsink_destinations,
    validate_source_quarantine_destination,
    validate_transform_error_sinks,
)

if TYPE_CHECKING:
    from elspeth.contracts import SinkProtocol, SourceProtocol, TransformProtocol
    from elspeth.core.config import AggregationSettings, ElspethSettings
    from elspeth.core.dag import WiredTransform
    from elspeth.core.dag.graph import ExecutionGraph


def assemble_and_validate_pipeline_config(
    *,
    source: SourceProtocol,
    transforms: Sequence[WiredTransform],
    sinks: Mapping[str, SinkProtocol],
    aggregations: Mapping[str, tuple[TransformProtocol, AggregationSettings]],
    settings: ElspethSettings,
    graph: ExecutionGraph,
) -> PipelineConfig:
    """Fold aggregations into transforms, build :class:`PipelineConfig`, and
    run the four orchestrator route-target validators.

    Mirrors the assembly logic in ``src/elspeth/web/execution/service.py`` and
    the four-validator pre-init call site in
    ``src/elspeth/engine/orchestrator/core.py``.

    Args:
        source: Instantiated source plugin.
        transforms: Wired transforms from ``PluginBundle.transforms``.
        sinks: Sink instances keyed by name from ``PluginBundle.sinks``.
        aggregations: Aggregation transforms + settings keyed by aggregation
            name (from ``PluginBundle.aggregations``).
        settings: Full ``ElspethSettings`` (provides ``gates`` and
            ``coalesce`` settings).
        graph: ``ExecutionGraph`` (provides ``aggregation_id_map`` for
            ``node_id`` assignment).

    Returns:
        Assembled ``PipelineConfig`` with aggregations folded into transforms.

    Raises:
        RouteValidationError: A route target references a non-existent sink,
            or a sink failsink violates one of the failsink rules.
        OrchestrationInvariantError: A framework invariant has been broken
            (e.g. a transform ``on_error`` is ``None`` when ``TransformSettings``
            requires it). This is a programmer-bug class — callers in the
            composer/service path should let it propagate to a 500 rather than
            translating it to a per-pipeline validation failure.
    """
    all_transforms: list[TransformProtocol] = [t.plugin for t in transforms]
    aggregation_settings: dict[str, AggregationSettings] = {}
    agg_id_map = graph.get_aggregation_id_map()

    for agg_name, (transform, agg_config) in aggregations.items():
        node_id = agg_id_map[AggregationName(agg_name)]
        aggregation_settings[node_id] = agg_config
        # Intentional mutation: aggregation transforms need node_id set so
        # the orchestrator can index aggregation_settings by node_id at run
        # time. Mirrors service.py:663-667.
        transform.node_id = node_id
        all_transforms.append(transform)

    pipeline_config = PipelineConfig(
        source=source,
        transforms=all_transforms,
        sinks=sinks,
        config=resolve_config(settings),
        gates=list(settings.gates),
        aggregation_settings=aggregation_settings,
        coalesce_settings=(list(settings.coalesce) if settings.coalesce else []),
    )

    available_sinks = set(pipeline_config.sinks.keys())

    validate_route_destinations(
        route_resolution_map=graph.get_route_resolution_map(),
        available_sinks=available_sinks,
        transform_id_map=graph.get_transform_id_map(),
        transforms=pipeline_config.transforms,
        config_gate_id_map=graph.get_config_gate_id_map(),
        config_gates=pipeline_config.gates,
    )

    validate_transform_error_sinks(
        transforms=pipeline_config.transforms,
        available_sinks=available_sinks,
    )

    validate_source_quarantine_destination(
        source=pipeline_config.source,
        available_sinks=available_sinks,
    )

    sink_validation_stubs = {name: SimpleNamespace(on_write_failure=sink._on_write_failure) for name, sink in pipeline_config.sinks.items()}
    sink_plugins = {name: sink.name for name, sink in pipeline_config.sinks.items()}
    validate_sink_failsink_destinations(
        sink_configs=sink_validation_stubs,
        available_sinks=available_sinks,
        sink_plugins=sink_plugins,
    )

    # NB: Value-source compliance is enforced upstream in
    # ``cli_helpers.instantiate_plugins_from_config`` — by the time the
    # bundle reaches this function, declared field values have already
    # been validated against their VALUE_SOURCES contracts. Re-running
    # the walker here would be redundant.

    return pipeline_config


def validate_value_source_compliance(transforms: Sequence[WiredTransform]) -> None:
    """Reject pipelines whose plugin configs violate VALUE_SOURCES declarations.

    Walks each transform's typed config (via the plugin's public ``config``
    accessor when present) and dispatches each
    :class:`elspeth.contracts.value_source.ValueSource` declaration:

    * :class:`CatalogValueSource` — ``getattr(config, field_name)`` must
      appear in the catalog resolved by
      :func:`elspeth.contracts.value_source.get_catalog_values`. An
      empty catalog (e.g. optional dependency missing) is itself a
      structured failure, never a silent pass.
    * :class:`DerivedFromSiblingValueSource` — the field value must equal
      the sibling field's value. With ``allow_empty_default=True``, an
      empty (``""`` / ``None``) value is also accepted.

    Plugins that do not expose a ``config`` attribute, or whose config
    class has no ``VALUE_SOURCES`` ClassVar, are skipped (they have no
    declarations to enforce).

    Raises:
        ValueSourceValidationError: One or more declared sources rejected
            their field's value. The exception's ``findings`` tuple
            carries one :class:`ValueSourceFinding` per offending field,
            with structured ``component_id`` / ``field_name`` / ``reason``
            attributes; the composer ``/validate`` path reads them
            directly to build per-component ``ValidationError`` records.
        UnknownCatalogIdError: A declaration references a catalog id with
            no registered reader. This is a programmer-bug class
            (declaration was wired but plugin pack didn't register the
            reader); callers should let it propagate to a 500.
    """
    findings: list[ValueSourceFinding] = []
    for wired in transforms:
        # The L0 registry returns the typed config for plugins that have
        # explicitly opted into value-source compliance via
        # ``register_value_source_plugin``; ``None`` for everything else.
        # Explicit opt-in instead of duck-typing avoids defensive
        # getattr/hasattr/isinstance patterns and makes the contract
        # discoverable at plugin-pack import time.
        config = find_value_source_config(wired.plugin)
        if config is None:
            continue
        config_cls = type(config)
        # The discriminated-union declarations live on the config class
        # as a ``VALUE_SOURCES`` ClassVar. A registered plugin whose
        # config class has no declarations is a plugin-pack contract
        # bug — let AttributeError surface rather than silently passing.
        # mypy narrows ``type(config)`` to ``type[object]`` and cannot
        # see the ClassVar declared by L3 plugin packs; ``# type: ignore``
        # documents the intentional contract break.
        declarations: tuple[ValueSource, ...] = config_cls.VALUE_SOURCES  # type: ignore[attr-defined]
        if not declarations:
            continue
        # ``settings.name`` is the operator-facing transform identifier
        # (e.g. ``"openrouter_llm_node_1"``) — pinned into each finding
        # so the composer can attribute errors to a specific component
        # without re-walking the bundle.
        component_id = wired.settings.name
        for declaration in declarations:
            finding = _check_value_source(declaration, config, component_id)
            if finding is not None:
                findings.append(finding)
    if findings:
        message = f"{len(findings)} field(s) violated value-source declarations: " + "; ".join(f.format() for f in findings)
        raise ValueSourceValidationError(message, findings=tuple(findings))


def check_config_value_sources(config: object, *, component_id: str) -> tuple[ValueSourceFinding, ...]:
    """Run one already-constructed config's ``VALUE_SOURCES`` declarations.

    Per-config counterpart to :func:`validate_value_source_compliance` (which
    walks a wired bundle). It lets per-node callers — notably the composer's
    pre-wiring option prevalidation — surface the same structured findings
    (catalog membership, sibling derivation) without constructing a bundle, so a
    hallucinated catalog value (e.g. an unknown OpenRouter ``model``) is caught
    at authoring time with an actionable ``list_models`` hint rather than only at
    instantiation. Returns ``()`` when the config's class declares no protocols.
    """
    return tuple(
        finding
        for declaration in _declared_value_sources(type(config))
        if (finding := _check_value_source(declaration, config, component_id)) is not None
    )


def _declared_value_sources(config_cls: type) -> tuple[ValueSource, ...]:
    """Return ``config_cls``'s ``VALUE_SOURCES`` ClassVar, or ``()`` if it declares none.

    ``VALUE_SOURCES`` is an optional ClassVar — config classes with no
    value-source protocols omit it entirely (see
    :mod:`elspeth.contracts.value_source`). Walk the MRO's class ``__dict__``
    directly so a genuinely-absent declaration is distinguished from a
    present-but-empty one, without ``getattr(..., default)`` defensively
    suppressing the attribute.
    """
    for klass in config_cls.__mro__:
        if "VALUE_SOURCES" in klass.__dict__:
            return cast("tuple[ValueSource, ...]", klass.__dict__["VALUE_SOURCES"])
    return ()


def _check_value_source(
    declaration: ValueSource,
    config: object,
    component_id: str,
) -> ValueSourceFinding | None:
    """Run a single declaration against ``config``; return None on pass.

    Dispatches on the concrete variant of the discriminated union via
    ``match``/``case`` (structural dispatch). Returns a structured
    :class:`ValueSourceFinding` so the L3 consumer can read
    ``component_id``/``field_name``/``reason`` directly without parsing
    a formatted string.
    """
    match declaration:
        case CatalogValueSource():
            return _check_catalog_membership(declaration, config, component_id)
        case DerivedFromSiblingValueSource():
            return _check_derived_from_sibling(declaration, config, component_id)
        case _:
            raise TypeError(f"Unknown ValueSource variant {type(declaration).__name__} on {component_id!r}: {declaration!r}")


def _check_catalog_membership(
    declaration: CatalogValueSource,
    config: object,
    component_id: str,
) -> ValueSourceFinding | None:
    # ``applies_when`` predicate: catalog membership is conditional on
    # sibling field values. If any predicate pair doesn't match, the
    # catalog isn't authoritative for this config (e.g. OpenRouter
    # base_url overridden to a chaos test endpoint) — skip the check
    # rather than rejecting values the upstream endpoint would accept.
    for sibling_field, expected_value in declaration.applies_when:
        actual_value = _read_field(config, sibling_field)
        if actual_value != expected_value:
            return None
    value = _read_field(config, declaration.field_name)
    catalog = get_catalog_values(declaration.catalog_id)
    if not catalog:
        # Quote the registrar's actionable hint verbatim when present so
        # the operator sees the specific install command instead of a
        # generic "install the optional dependency". Falls back to the
        # generic message when no hint was registered.
        dep_hint = get_catalog_missing_dep_hint(declaration.catalog_id)
        remediation = (
            dep_hint
            if dep_hint is not None
            else ("install the optional dependency that provides the catalog or pin a static catalog snapshot")
        )
        return ValueSourceFinding(
            component_id=component_id,
            field_name=declaration.field_name,
            reason=(f"catalog '{declaration.catalog_id}' is empty or unavailable; cannot verify field value ({remediation})"),
        )
    # ``value`` is a string for the LLM ``model`` field today; for non-string
    # values we structurally reject (type mismatch is a Pydantic-level fault
    # caught earlier — defense-in-depth here).
    match value:
        case str() if value in catalog:
            return None
        case _:
            return ValueSourceFinding(
                component_id=component_id,
                field_name=declaration.field_name,
                reason=(
                    f"value {value!r} is not in catalog '{declaration.catalog_id}' "
                    f"(catalog has {len(catalog)} entries; pick a valid value via the "
                    "list_models composer tool)"
                ),
            )


def _check_derived_from_sibling(
    declaration: DerivedFromSiblingValueSource,
    config: object,
    component_id: str,
) -> ValueSourceFinding | None:
    field_value = _read_field(config, declaration.field_name)
    sibling_value = _read_field(config, declaration.sibling_field)
    if declaration.allow_empty_default and (field_value is None or field_value == ""):
        return None
    if field_value == sibling_value:
        return None
    return ValueSourceFinding(
        component_id=component_id,
        field_name=declaration.field_name,
        reason=(
            f"value {field_value!r} must equal sibling "
            f"'{declaration.sibling_field}' (currently {sibling_value!r})"
            + ("; leave the field empty to inherit the sibling value" if declaration.allow_empty_default else "")
        ),
    )


def _read_field(config: object, field_name: str) -> object:
    """Read a Pydantic config field by name without ``getattr`` defensive default.

    The walker reads field names declared by the plugin's own
    ``VALUE_SOURCES``. A declared field that does not exist on the config
    is a contract bug in the plugin pack — let the AttributeError surface
    rather than silently substituting a default.
    """
    return config.__getattribute__(field_name)


# Re-export for L3 callers that translate structured findings into per-component
# ValidationError records (composer /validate). UnknownCatalogIdError is
# intentionally NOT caught by the walker — it surfaces unconfigured catalogs
# as 500-class programmer bugs, not per-pipeline validation failures.
__all__ = [
    "UnknownCatalogIdError",
    "assemble_and_validate_pipeline_config",
    "validate_value_source_compliance",
]
