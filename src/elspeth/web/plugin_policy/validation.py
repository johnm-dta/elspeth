"""Pure validation and lowering for one frozen plugin-policy snapshot."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from typing import Literal

from jsonschema import Draft202012Validator

from elspeth.contracts.freeze import deep_thaw, freeze_fields
from elspeth.contracts.plugin_capabilities import ControlMode, WebConfigAuthority
from elspeth.web.catalog.protocol import CatalogService
from elspeth.web.catalog.schemas import PluginSchemaInfo
from elspeth.web.composer.state import CompositionState, NodeSpec, OutputSpec, SourceSpec, ValidationEntry, ValidationSummary
from elspeth.web.interpretation_state import AUTHORING_METADATA_OPTION_KEYS
from elspeth.web.plugin_policy.coverage import control_coverage_findings
from elspeth.web.plugin_policy.models import PluginAvailabilitySnapshot, PluginId, PluginUnavailableReason
from elspeth.web.plugin_policy.profiles import LoweredPluginConfig, OperatorProfileRegistry

PolicyValidationStage = Literal[
    "plugin_enablement",
    "operator_profile_options",
    "required_control_availability",
    "required_control_coverage",
]

_PROFILE_LOWERING_METADATA_OPTION_KEYS = AUTHORING_METADATA_OPTION_KEYS | {"resolved_prompt_template_hash"}


@dataclass(frozen=True, slots=True)
class PluginPolicyFinding:
    stage: PolicyValidationStage
    component_id: str | None
    component_type: str | None
    error_code: str
    message: str


@dataclass(frozen=True, slots=True)
class PluginPolicyValidationResult:
    executable_state: CompositionState = field(repr=False)
    findings: tuple[PluginPolicyFinding, ...]

    def findings_for(self, stage: PolicyValidationStage) -> tuple[PluginPolicyFinding, ...]:
        return tuple(finding for finding in self.findings if finding.stage == stage)


@dataclass(frozen=True, slots=True)
class _Component:
    component_id: str
    component_type: Literal["source", "transform", "sink"]
    plugin_id: PluginId | None
    options: Mapping[str, object]

    def __post_init__(self) -> None:
        freeze_fields(self, "options")


def validate_plugin_policy(
    state: CompositionState,
    *,
    snapshot: PluginAvailabilitySnapshot,
    profile_registry: OperatorProfileRegistry | None,
    catalog: CatalogService,
) -> PluginPolicyValidationResult:
    """Validate identities/controls and lower public profile options in memory.

    The returned state is executable-only. ``state`` remains the audit-safe
    authored value and is never mutated.
    """
    findings: list[PluginPolicyFinding] = []
    components = _components(state)
    unavailable = {item.plugin_id: item.reason for item in snapshot.unavailable}

    for component in components:
        if component.plugin_id is None:
            findings.append(
                PluginPolicyFinding(
                    stage="plugin_enablement",
                    component_id=component.component_id,
                    component_type=component.component_type,
                    error_code="plugin_unavailable",
                    message="The configured plugin identity is invalid or unavailable.",
                )
            )
            continue
        if component.plugin_id in snapshot.available:
            continue
        reason = unavailable.get(component.plugin_id, PluginUnavailableReason.NOT_AUTHORIZED)
        findings.append(
            PluginPolicyFinding(
                stage="plugin_enablement",
                component_id=component.component_id,
                component_type=component.component_type,
                error_code=reason.value,
                message=f"Plugin '{component.plugin_id}' is not available for this request.",
            )
        )

    executable_state = state
    if not findings and snapshot.principal_scope != "local:trained-operator":
        executable_state, profile_findings = _lower_profiled_components(
            state,
            snapshot=snapshot,
            profile_registry=profile_registry,
            catalog=catalog,
        )
        findings.extend(profile_findings)

    selected = dict(snapshot.selected)
    required = tuple(capability for capability, mode in snapshot.control_modes if mode is ControlMode.REQUIRED)
    for capability in required:
        selected_plugin = selected.get(capability)
        if selected_plugin is not None and selected_plugin in snapshot.available:
            continue
        findings.append(
            PluginPolicyFinding(
                stage="required_control_availability",
                component_id=None,
                component_type=None,
                error_code="required_control_unavailable",
                message=f"Required control '{capability.value}' has no available implementation.",
            )
        )

    for capability in required:
        for coverage in control_coverage_findings(state, capability):
            findings.append(
                PluginPolicyFinding(
                    stage="required_control_coverage",
                    component_id=coverage.component_id,
                    component_type="transform",
                    error_code="required_control_coverage",
                    message=(
                        f"Node '{coverage.component_id}' is not covered by the required "
                        f"'{coverage.capability.value}' {coverage.role.value} control."
                    ),
                )
            )

    return PluginPolicyValidationResult(executable_state=executable_state, findings=tuple(findings))


def _plugin_id(kind: Literal["source", "transform", "sink"], name: str) -> PluginId | None:
    try:
        return PluginId(kind, name)
    except ValueError:
        return None


def _components(state: CompositionState) -> tuple[_Component, ...]:
    result: list[_Component] = []
    for source_name, source in sorted(state.sources.items()):
        component_id = "source" if source_name == "source" else f"source:{source_name}"
        result.append(
            _Component(
                component_id=component_id,
                component_type="source",
                plugin_id=_plugin_id("source", source.plugin),
                options=deep_thaw(source.options),
            )
        )
    for node in state.nodes:
        if node.plugin is None:
            continue
        result.append(
            _Component(
                component_id=node.id,
                component_type="transform",
                plugin_id=_plugin_id("transform", node.plugin),
                options=deep_thaw(node.options),
            )
        )
    for output in state.outputs:
        result.append(
            _Component(
                component_id=output.name,
                component_type="sink",
                plugin_id=_plugin_id("sink", output.plugin),
                options=deep_thaw(output.options),
            )
        )
    return tuple(result)


def _lower_profiled_components(
    state: CompositionState,
    *,
    snapshot: PluginAvailabilitySnapshot,
    profile_registry: OperatorProfileRegistry | None,
    catalog: CatalogService,
) -> tuple[CompositionState, tuple[PluginPolicyFinding, ...]]:
    aliases_by_plugin = dict(snapshot.usable_profile_aliases)
    components = _components(state)
    lowered_options: dict[tuple[str, str], dict[str, object]] = {}
    findings: list[PluginPolicyFinding] = []
    lowering_registry = profile_registry
    lowering_catalog = catalog
    for component in components:
        profile_context = _profile_lowering_context(
            component,
            aliases_by_plugin=aliases_by_plugin,
            profile_registry=lowering_registry,
            catalog=lowering_catalog,
            findings=findings,
        )
        if profile_context is None:
            continue
        plugin_id, aliases, resolved_public_schema = profile_context
        authored_options = {
            name: deep_thaw(value) for name, value in component.options.items() if name not in _PROFILE_LOWERING_METADATA_OPTION_KEYS
        }
        authoring_metadata = {
            name: deep_thaw(value) for name, value in component.options.items() if name in _PROFILE_LOWERING_METADATA_OPTION_KEYS
        }
        alias = authored_options.pop("profile", None)
        if not isinstance(alias, str) or alias not in aliases:
            findings.append(_profile_unavailable_finding(component, plugin_id))
            continue
        public_schema = resolved_public_schema.json_schema
        public_options = {"profile": alias, **authored_options}
        schema_errors = sorted(
            Draft202012Validator(public_schema).iter_errors(public_options),
            key=lambda error: tuple(str(part) for part in error.absolute_path),
        )
        if schema_errors:
            findings.append(
                PluginPolicyFinding(
                    stage="operator_profile_options",
                    component_id=component.component_id,
                    component_type=component.component_type,
                    error_code="profile_unavailable",
                    message=f"Plugin '{plugin_id}' options do not match its public operator-profile schema.",
                )
            )
            continue
        try:
            assert lowering_registry is not None
            lowered = _lower_profile_options(
                lowering_registry,
                plugin_id,
                alias=alias,
                safe_options=authored_options,
            )
        except ValueError:
            findings.append(
                PluginPolicyFinding(
                    stage="operator_profile_options",
                    component_id=component.component_id,
                    component_type=component.component_type,
                    error_code="plugin_unavailable",
                    message=f"Plugin '{plugin_id}' operator profile is no longer available.",
                )
            )
            continue
        lowered_options[(component.component_type, component.component_id)] = {
            **deep_thaw(lowered.executable_options),
            **authoring_metadata,
        }

    if findings:
        return state, _normalized_profile_findings(findings)

    sources: dict[str, SourceSpec] = {}
    for source_name, source in state.sources.items():
        component_id = "source" if source_name == "source" else f"source:{source_name}"
        options = lowered_options.get(("source", component_id))
        sources[source_name] = source if options is None else replace(source, options=options)
    nodes: list[NodeSpec] = []
    for node in state.nodes:
        options = lowered_options.get(("transform", node.id))
        nodes.append(node if options is None else replace(node, options=options))
    outputs: list[OutputSpec] = []
    for output in state.outputs:
        options = lowered_options.get(("sink", output.name))
        outputs.append(output if options is None else replace(output, options=options))
    return (
        replace(state, sources=sources, nodes=tuple(nodes), outputs=tuple(outputs)),
        (),
    )


def _profile_unavailable_finding(component: _Component, plugin_id: PluginId) -> PluginPolicyFinding:
    return PluginPolicyFinding(
        stage="operator_profile_options",
        component_id=component.component_id,
        component_type=component.component_type,
        error_code="profile_unavailable",
        message=f"Plugin '{plugin_id}' requires an available operator profile.",
    )


@dataclass(frozen=True, slots=True)
class ProfileAwareValidationResult:
    """One authoritative composer validation result for authored web state."""

    authored_state: CompositionState
    executable_state: CompositionState = field(repr=False)
    policy_findings: tuple[PluginPolicyFinding, ...]
    validation: ValidationSummary


_IMMEDIATE_BLOCKING_STAGES: frozenset[PolicyValidationStage] = frozenset({"plugin_enablement", "operator_profile_options"})


def validate_authored_composition_state(
    state: CompositionState,
    *,
    snapshot: PluginAvailabilitySnapshot,
    profile_registry: OperatorProfileRegistry | None,
    catalog: CatalogService,
) -> ProfileAwareValidationResult:
    """Validate authored state once through the principal's policy boundary."""
    if snapshot.principal_scope == "local:trained-operator":
        return ProfileAwareValidationResult(
            authored_state=state,
            executable_state=state,
            policy_findings=(),
            validation=state.validate(),
        )

    policy = validate_plugin_policy(
        state,
        snapshot=snapshot,
        profile_registry=profile_registry,
        catalog=catalog,
    )
    blocking = tuple(finding for finding in policy.findings if finding.stage in _IMMEDIATE_BLOCKING_STAGES)
    evidence = tuple(finding for finding in policy.findings if finding.stage not in _IMMEDIATE_BLOCKING_STAGES)
    blocking_entries = tuple(_validation_entry(finding, severity="high") for finding in blocking)
    evidence_entries = tuple(_validation_entry(finding, severity="medium") for finding in evidence)

    if blocking_entries:
        summary = ValidationSummary(
            is_valid=False,
            errors=blocking_entries,
            warnings=evidence_entries,
        )
    else:
        executable_summary = policy.executable_state.validate()
        summary = replace(
            executable_summary,
            warnings=(*evidence_entries, *executable_summary.warnings),
        )

    return ProfileAwareValidationResult(
        authored_state=state,
        executable_state=policy.executable_state,
        policy_findings=policy.findings,
        validation=summary,
    )


def _validation_entry(finding: PluginPolicyFinding, *, severity: Literal["high", "medium"]) -> ValidationEntry:
    component = "pipeline"
    if finding.component_id is not None:
        if finding.component_type == "transform":
            component = f"node:{finding.component_id}"
        elif finding.component_type == "sink":
            component = f"output:{finding.component_id}"
        else:
            component = finding.component_id
    return ValidationEntry(
        component=component,
        message=finding.message,
        severity=severity,
        error_code=finding.error_code,
    )


def _profile_lowering_context(
    component: _Component,
    *,
    aliases_by_plugin: Mapping[PluginId, tuple[str, ...]],
    profile_registry: OperatorProfileRegistry | None,
    catalog: CatalogService,
    findings: list[PluginPolicyFinding],
) -> tuple[PluginId, tuple[str, ...], PluginSchemaInfo] | None:
    """Resolve the public profile schema or record one closed failure."""
    plugin_id = component.plugin_id
    if plugin_id is None:
        return None
    full_schema = catalog.get_schema(plugin_id.kind, plugin_id.name)
    requires_profile = full_schema.web_config_authority is WebConfigAuthority.OPERATOR_PROFILED
    if not requires_profile and "profile" not in component.options:
        return None
    aliases = aliases_by_plugin[plugin_id] if plugin_id in aliases_by_plugin else ()
    if profile_registry is None or not aliases:
        findings.append(_profile_unavailable_finding(component, plugin_id))
        return None
    public_schema = profile_registry.public_schema(
        plugin_id,
        full_schema,
        available_aliases=aliases,
    )
    return plugin_id, aliases, public_schema


def _normalized_profile_findings(findings: list[PluginPolicyFinding]) -> tuple[PluginPolicyFinding, ...]:
    """Expose the single public profile failure code at the web boundary."""
    return tuple(
        replace(finding, error_code="profile_unavailable")
        if finding.stage == "operator_profile_options" and finding.error_code == "plugin_unavailable"
        else finding
        for finding in findings
    )


def _lower_profile_options(
    profile_registry: OperatorProfileRegistry,
    plugin_id: PluginId,
    *,
    alias: str,
    safe_options: dict[str, object],
) -> LoweredPluginConfig:
    """Collapse expected resolver outages into the established ValueError path."""
    try:
        return profile_registry.lower_options(plugin_id, alias=alias, safe_options=safe_options)
    except RuntimeError as exc:
        raise ValueError("profile_unavailable") from exc
