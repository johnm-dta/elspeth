"""Pure validation and lowering for one frozen plugin-policy snapshot."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Literal

from jsonschema import Draft202012Validator

from elspeth.contracts.freeze import deep_thaw
from elspeth.contracts.plugin_capabilities import ControlMode
from elspeth.web.catalog.policy_view import PolicyCatalogView
from elspeth.web.composer.state import CompositionState, NodeSpec, OutputSpec, SourceSpec
from elspeth.web.dependencies import create_catalog_service
from elspeth.web.interpretation_state import AUTHORING_METADATA_OPTION_KEYS
from elspeth.web.plugin_policy.coverage import control_coverage_findings
from elspeth.web.plugin_policy.models import PluginAvailabilitySnapshot, PluginId, PluginUnavailableReason
from elspeth.web.plugin_policy.profiles import OperatorProfileRegistry

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
    executable_state: CompositionState
    findings: tuple[PluginPolicyFinding, ...]

    def findings_for(self, stage: PolicyValidationStage) -> tuple[PluginPolicyFinding, ...]:
        return tuple(finding for finding in self.findings if finding.stage == stage)


@dataclass(frozen=True, slots=True)
class _Component:
    component_id: str
    component_type: Literal["source", "transform", "sink"]
    plugin_id: PluginId | None
    options: Mapping[str, object]


def validate_plugin_policy(
    state: CompositionState,
    *,
    snapshot: PluginAvailabilitySnapshot,
    profile_registry: OperatorProfileRegistry | None,
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
        error_code = "plugin_not_enabled" if reason is PluginUnavailableReason.NOT_AUTHORIZED else "plugin_unavailable"
        findings.append(
            PluginPolicyFinding(
                stage="plugin_enablement",
                component_id=component.component_id,
                component_type=component.component_type,
                error_code=error_code,
                message=f"Plugin '{component.plugin_id}' is not available for this request.",
            )
        )

    executable_state = state
    if not findings:
        executable_state, profile_findings = _lower_profiled_components(
            state,
            snapshot=snapshot,
            profile_registry=profile_registry,
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
) -> tuple[CompositionState, tuple[PluginPolicyFinding, ...]]:
    aliases_by_plugin = dict(snapshot.usable_profile_aliases)
    if not aliases_by_plugin:
        return state, ()
    if profile_registry is None:
        return state, (
            PluginPolicyFinding(
                stage="operator_profile_options",
                component_id=None,
                component_type=None,
                error_code="plugin_unavailable",
                message="Operator profile resolution is unavailable.",
            ),
        )

    view = PolicyCatalogView(create_catalog_service(), snapshot, profile_registry)
    lowered_options: dict[tuple[str, str], dict[str, object]] = {}
    findings: list[PluginPolicyFinding] = []
    for component in _components(state):
        plugin_id = component.plugin_id
        if plugin_id is None or plugin_id not in aliases_by_plugin:
            continue
        aliases = aliases_by_plugin[plugin_id]
        authored_options = {name: value for name, value in component.options.items() if name not in _PROFILE_LOWERING_METADATA_OPTION_KEYS}
        authoring_metadata = {name: value for name, value in component.options.items() if name in _PROFILE_LOWERING_METADATA_OPTION_KEYS}
        alias = authored_options.pop("profile", None)
        if not isinstance(alias, str) or alias not in aliases:
            findings.append(
                PluginPolicyFinding(
                    stage="operator_profile_options",
                    component_id=component.component_id,
                    component_type=component.component_type,
                    error_code="plugin_unavailable",
                    message=f"Plugin '{plugin_id}' requires an available operator profile.",
                )
            )
            continue

        public_schema = view.get_schema(plugin_id.kind, plugin_id.name).json_schema
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
                    error_code="plugin_unavailable",
                    message=f"Plugin '{plugin_id}' options do not match its public operator-profile schema.",
                )
            )
            continue
        try:
            lowered = profile_registry.lower_options(plugin_id, alias=alias, safe_options=authored_options)
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
            **dict(lowered.executable_options),
            **authoring_metadata,
        }

    if findings:
        return state, tuple(findings)

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
