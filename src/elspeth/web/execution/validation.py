"""Dry-run validation using real engine code paths.

Calls the same functions as `elspeth run`: load_settings(),
instantiate_runtime_plugins(), build_runtime_graph(),
graph.validate(), assemble_and_validate_pipeline_config() (route targets),
graph.validate_edge_compatibility().

W18 fix: Only typed exceptions are caught. Bare except Exception is forbidden.
Unknown exception types propagate as 500 Internal Server Error, signalling
that this function needs updating — not that the error should be swallowed.

Settings loading uses load_settings_from_yaml_string() — the same in-memory
loader as the execution service. This ensures validation exercises the exact
same code path as execution, and resolved secrets never touch disk.

Route-target validation (issue elspeth-127de6865a) closes the parity gap
where the orchestrator's four pre-init validators
(validate_route_destinations, validate_transform_error_sinks,
validate_source_quarantine_destination, validate_sink_failsink_destinations)
were not reached by /validate, letting dangling references pass preflight
only to be rejected pre-token at /execute.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, cast
from uuid import UUID

import yaml
from pydantic import ValidationError as PydanticValidationError

from elspeth.contracts.blobs import BlobRecord
from elspeth.contracts.blobs_inline import BlobInlineValidationViolation
from elspeth.contracts.secrets import SecretRefPlacementViolation, WebSecretResolver
from elspeth.core.blobs_inline import (
    BLOB_INLINE_AGGREGATE_BYTE_CAP,
    BLOB_INLINE_PER_REF_BYTE_CAP,
    _substitute_blob_content_refs_for_validation,
    _validate_blob_content_refs_sync,
)
from elspeth.core.config import load_settings_from_yaml_string
from elspeth.core.dag.models import EdgeContractError, GraphValidationError
from elspeth.core.secrets import (
    collect_credential_field_violations,
    collect_disallowed_secret_ref_markers,
    resolve_secret_refs,
    secret_env_ref_name,
)
from elspeth.engine.orchestrator.preflight import assemble_and_validate_pipeline_config
from elspeth.engine.orchestrator.types import (
    RouteValidationError,
    ValueSourceValidationError,
)
from elspeth.plugins.infrastructure.config_base import PluginConfigError
from elspeth.plugins.infrastructure.manager import PluginNotFoundError
from elspeth.web.composer._semantic_validator import validate_semantic_contracts
from elspeth.web.composer.state import (
    CompositionState,
    _batch_aware_placement_error,
    _batch_aware_required_input_fields_error,
    _batch_distribution_profile_value_field_entries,
)
from elspeth.web.execution._semantic_helpers import (
    assistance_suggestion_for,
    serialize_semantic_contracts,
)
from elspeth.web.execution.preflight import (
    RUNTIME_CHECK_GRAPH_STRUCTURE,
    RUNTIME_CHECK_PLUGIN_INSTANTIATION,
    RUNTIME_CHECK_SCHEMA_COMPATIBILITY,
    RUNTIME_GRAPH_VALIDATION_CHECKS,
    build_runtime_graph,
    instantiate_runtime_plugins,
    resolve_runtime_yaml_paths,
)
from elspeth.web.execution.protocol import ValidationSettings, YamlGenerator
from elspeth.web.execution.schemas import (
    CHECK_OUTCOME_SECRET_REFS_NO_REFS,
    CHECK_OUTCOME_SECRET_REFS_RESOLVED,
    CHECK_OUTCOME_SECRET_REFS_SKIPPED_NO_SERVICE,
    CHECK_OUTCOME_SECRET_REFS_UNRESOLVED,
    CHECK_OUTCOME_SKIPPED_AFTER_FAILURE,
    ValidationCheck,
    ValidationError,
    ValidationReadiness,
    ValidationReadinessBlocker,
    ValidationResult,
)
from elspeth.web.interpretation_state import (
    INTERPRETATION_REVIEW_PENDING_CODE,
    InterpretationReviewPending,
    InterpretationReviewSite,
    materialize_state_for_authoring,
    materialize_state_for_execution,
)
from elspeth.web.secrets.ref_policy import allowed_secret_ref_fields, allowed_secret_ref_fields_text

# ── Check names (ordered) ─────────────────────────────────────────────
_CHECK_PATH_ALLOWLIST = "path_allowlist"
_CHECK_WEB_SCRAPE_NETWORK_POLICY = "web_scrape_network_policy"
_CHECK_SECRET_REFS = "secret_refs"
_CHECK_BLOB_INLINE_REFS = "blob_inline_refs"
_CHECK_SEMANTIC_CONTRACTS = "semantic_contracts"
_CHECK_BATCH_TRANSFORM_OPTIONS = "batch_transform_options"
_CHECK_INTERPRETATION_REVIEW = "interpretation_review"
_CHECK_SETTINGS = "settings_load"
_CHECK_PLUGINS = RUNTIME_CHECK_PLUGIN_INSTANTIATION
_CHECK_VALUE_SOURCE_COMPLIANCE = "value_source_compliance"
_CHECK_GRAPH = RUNTIME_CHECK_GRAPH_STRUCTURE
_CHECK_ROUTE_TARGETS = "route_target_resolution"
_CHECK_SCHEMA = RUNTIME_CHECK_SCHEMA_COMPATIBILITY
assert RUNTIME_GRAPH_VALIDATION_CHECKS == (_CHECK_PLUGINS, _CHECK_GRAPH, _CHECK_SCHEMA)


def _execution_ready() -> ValidationReadiness:
    return ValidationReadiness(
        authoring_valid=True,
        execution_ready=True,
        completion_ready=True,
        blockers=[],
    )


def _blocked_readiness(
    *,
    code: str,
    detail: str,
    component_id: str | None = None,
    component_type: str | None = None,
    authoring_valid: bool = False,
    completion_ready: bool = False,
) -> ValidationReadiness:
    return ValidationReadiness(
        authoring_valid=authoring_valid,
        execution_ready=False,
        completion_ready=completion_ready,
        blockers=[
            ValidationReadinessBlocker(
                code=code,
                component_id=component_id,
                component_type=component_type,
                detail=detail,
            )
        ],
    )


# Advisory check — non-blocking, multi-entry (one ValidationCheck per
# detected node, all sharing this name).  Deliberately NOT included in
# _ALL_CHECKS: that list governs the "skipped check" propagation when an
# earlier pass/fail check fails.  This advisory uses ``passed=True`` for
# every entry and is emitted only on the happy-path return, so structural
# errors are never drowned in cosmetic noise.
_CHECK_IDENTITY_NODE_ADVISORY = "identity_node_advisory"

# _CHECK_VALUE_SOURCE_COMPLIANCE slots between _CHECK_PLUGINS (typed configs
# now exist) and _CHECK_GRAPH (so a hallucinated model fails before any DAG
# work). The position is asserted by tests/unit/web/execution/test_validation.py
# to prevent silent reordering.
_ALL_CHECKS = [
    _CHECK_PATH_ALLOWLIST,
    _CHECK_WEB_SCRAPE_NETWORK_POLICY,
    _CHECK_SECRET_REFS,
    _CHECK_SEMANTIC_CONTRACTS,
    _CHECK_BATCH_TRANSFORM_OPTIONS,
    _CHECK_INTERPRETATION_REVIEW,
    _CHECK_BLOB_INLINE_REFS,
    _CHECK_SETTINGS,
    _CHECK_PLUGINS,
    _CHECK_VALUE_SOURCE_COMPLIANCE,
    _CHECK_GRAPH,
    _CHECK_ROUTE_TARGETS,
    _CHECK_SCHEMA,
]


@dataclass(frozen=True, slots=True)
class _EdgePatchTarget:
    component_id: str
    component_type: str | None
    display_name: str
    schema_patch_tool_call: str


def _node_schema_patch_target(component_id: str, component_type: str | None) -> _EdgePatchTarget:
    return _EdgePatchTarget(
        component_id=component_id,
        component_type=component_type,
        display_name=f"{component_type or 'node'} '{component_id}'",
        schema_patch_tool_call=f"patch_node_options(node_id='{component_id}', patch={{'schema': {{...}}}})",
    )


def _source_schema_patch_target(source_name: str, plugin_name: str | None) -> _EdgePatchTarget:
    component_id = "source" if source_name == "source" else f"source:{source_name}"
    if source_name == "source":
        display = "source" if plugin_name is None else f"source '{plugin_name}'"
        schema_patch_tool_call = "patch_source_options(patch={'schema': {...}})"
    else:
        display = f"source '{source_name}'" if plugin_name is None else f"source '{source_name}' ({plugin_name})"
        schema_patch_tool_call = f"patch_source_options(source_name={source_name!r}, patch={{'schema': {{...}}}})"
    return _EdgePatchTarget(
        component_id=component_id,
        component_type="source",
        display_name=display,
        schema_patch_tool_call=schema_patch_tool_call,
    )


def _output_schema_patch_target(sink_name: str) -> _EdgePatchTarget:
    return _EdgePatchTarget(
        component_id=sink_name,
        component_type="sink",
        display_name=f"output '{sink_name}'",
        schema_patch_tool_call=f"patch_output_options(sink_name='{sink_name}', patch={{'schema': {{...}}}})",
    )


def _unmapped_schema_patch_target(dag_node_id: str, component_type: str | None) -> _EdgePatchTarget:
    return _EdgePatchTarget(
        component_id=dag_node_id,
        component_type=component_type,
        display_name=f"unmapped DAG node '{dag_node_id}'",
        schema_patch_tool_call="get_pipeline_state(component='all')  # inspect composer IDs before patching this DAG node",
    )


def _source_name_for_dag_source(state: CompositionState, graph: Any, dag_source_id: str) -> str | None:
    node_info = graph.get_node_info(dag_source_id)
    config = node_info.config
    if "source_name" in config:
        source_name = cast(str, config["source_name"])
        if source_name in state.sources:
            return source_name
    if len(state.sources) == 1:
        return next(iter(state.sources))
    return None


def _edge_patch_targets_by_dag_id(state: CompositionState, graph: Any) -> dict[str, _EdgePatchTarget]:
    """Map runtime DAG node IDs back to composer patch-tool targets."""
    targets: dict[str, _EdgePatchTarget] = {}
    nodes_by_id = {node.id: node for node in state.nodes}

    if state.sources:
        for source_id in graph.get_sources():
            dag_source_id = str(source_id)
            source_name = _source_name_for_dag_source(state, graph, dag_source_id)
            if source_name is None:
                continue
            source = state.sources[source_name]
            targets[dag_source_id] = _source_schema_patch_target(source_name, source.plugin)

    transform_nodes = [node for node in state.nodes if node.node_type == "transform"]
    transform_id_map = graph.get_transform_id_map()
    for sequence, dag_node_id in transform_id_map.items():
        if sequence >= len(transform_nodes):
            continue
        node = transform_nodes[sequence]
        targets[str(dag_node_id)] = _node_schema_patch_target(node.id, node.node_type)

    config_gate_id_map = graph.get_config_gate_id_map()
    for gate_name, dag_node_id in config_gate_id_map.items():
        component_id = str(gate_name)
        node_type = nodes_by_id[component_id].node_type if component_id in nodes_by_id else "gate"
        targets[str(dag_node_id)] = _node_schema_patch_target(component_id, node_type)

    aggregation_id_map = graph.get_aggregation_id_map()
    for aggregation_name, dag_node_id in aggregation_id_map.items():
        component_id = str(aggregation_name)
        node_type = nodes_by_id[component_id].node_type if component_id in nodes_by_id else "aggregation"
        targets[str(dag_node_id)] = _node_schema_patch_target(component_id, node_type)

    coalesce_id_map = graph.get_coalesce_id_map()
    for coalesce_name, dag_node_id in coalesce_id_map.items():
        component_id = str(coalesce_name)
        node_type = nodes_by_id[component_id].node_type if component_id in nodes_by_id else "coalesce"
        targets[str(dag_node_id)] = _node_schema_patch_target(component_id, node_type)

    sink_id_map = graph.get_sink_id_map()
    for sink_name, dag_node_id in sink_id_map.items():
        targets[str(dag_node_id)] = _output_schema_patch_target(str(sink_name))

    return targets


def _edge_patch_target_for_node_id(
    dag_node_id: str,
    *,
    state: CompositionState | None = None,
    graph: Any | None = None,
    component_type: str | None = None,
) -> _EdgePatchTarget:
    """Resolve a DAG node ID to the composer component/tool that can patch it."""
    if state is None or graph is None:
        return _node_schema_patch_target(dag_node_id, component_type)

    targets = _edge_patch_targets_by_dag_id(state, graph)
    if not targets:
        return _node_schema_patch_target(dag_node_id, component_type)
    if dag_node_id in targets:
        return targets[dag_node_id]
    return _unmapped_schema_patch_target(dag_node_id, component_type)


def _infer_component_type_from_plugin_error(
    exc: PluginNotFoundError | PluginConfigError,
) -> str | None:
    """Extract component type from plugin error metadata.

    Reads PluginConfigError.component_type directly — set by from_dict()
    from the config class hierarchy's _plugin_component_type attribute.
    Returns None for PluginNotFoundError or when component_type was not set.
    """
    if isinstance(exc, PluginConfigError):
        return exc.component_type
    return None


def _format_edge_contract_failure(
    exc: EdgeContractError,
    *,
    state: CompositionState | None = None,
    graph: Any | None = None,
) -> tuple[str, str]:
    """Build LLM-actionable (message, suggestion) pair from a structured edge-contract error.

    The composer surfaces both fields verbatim into the assistant's reply when
    runtime preflight rejects a completion claim. Empirically (cohort
    diagnosis 2026-05-07), models converge on retry only when the message
    names the producer/consumer node IDs and per-field issues, AND the
    suggestion lists concrete tool-call shapes for the fix. Prose like
    "Type mismatches: f (expected X, got Y)" by itself routinely caused the
    model to surrender mid-loop because there was no obvious next move.

    Format choices:
      - Producer/consumer are introduced by NODE ID first (the model uses
        these as ``node_id=`` arguments), then by SCHEMA NAME (informational
        — schema classes are baked-in plugin contracts, the model can't
        target them directly).
      - Each ``CompatibilityResult`` issue category gets its own bullet
        block. We keep the original "expected ... got ..." nomenclature
        from ``CompatibilityResult.error_message`` for continuity, but
        switch to "consumer requires ... producer emits ..." prose because
        empirically the composer LLM mis-grounds "expected/got" against
        the validator's perspective rather than the data-flow direction.
      - The suggestion leads with option (a) (patch consumer) because the
        dominant captured failure mode is consumer over-declaration. The
        producer-side option is listed second with the caveat that plugin
        output schemas are baked-in.
    """
    result = exc.compatibility_result
    issue_lines: list[str] = []
    if result.missing_fields:
        issue_lines.append("Missing required fields (consumer requires, producer does not guarantee):")
        for field_name in result.missing_fields:
            issue_lines.append(f"  - '{field_name}'")
    if result.type_mismatches:
        issue_lines.append("Type mismatches:")
        for field_name, expected, actual in result.type_mismatches:
            issue_lines.append(f"  - field '{field_name}': consumer requires '{expected}', producer emits '{actual}'")
    if result.constraint_mismatches:
        issue_lines.append("Constraint mismatches:")
        for field_name, reason in result.constraint_mismatches:
            issue_lines.append(f"  - field '{field_name}': {reason}")
    if result.extra_fields:
        issue_lines.append("Extra fields forbidden by consumer (producer emits, consumer rejects):")
        for field_name in result.extra_fields:
            issue_lines.append(f"  - '{field_name}'")

    issues_block = "\n".join(issue_lines) if issue_lines else "(no per-field detail available)"

    message = (
        f"Edge contract violation between producer node '{exc.from_node_id}' "
        f"(schema '{exc.producer_schema_name}') and consumer node '{exc.to_node_id}' "
        f"(schema '{exc.consumer_schema_name}'):\n"
        f"{issues_block}"
    )

    suggestion = _build_edge_contract_suggestion(exc, state=state, graph=graph)
    return message, suggestion


def _build_edge_contract_suggestion(
    exc: EdgeContractError,
    *,
    state: CompositionState | None = None,
    graph: Any | None = None,
) -> str:
    """Compose the action-oriented suggestion text for an edge-contract failure.

    Split out from ``_format_edge_contract_failure`` so the suggestion text
    can be unit-tested without exercising the full message-building flow,
    and so future tuning of the suggestion (e.g., emitting different prose
    for missing-field vs type-mismatch cases) keeps the message format
    stable.
    """
    result = exc.compatibility_result
    has_type_mismatch = bool(result.type_mismatches)
    has_missing = bool(result.missing_fields)
    has_extras = bool(result.extra_fields)
    consumer = _edge_patch_target_for_node_id(
        exc.to_node_id,
        state=state,
        graph=graph,
        component_type=exc.component_type,
    )
    producer = _edge_patch_target_for_node_id(
        exc.from_node_id,
        state=state,
        graph=graph,
        component_type=None,
    )

    parts: list[str] = []
    parts.append("Most edge-contract failures come from the consumer over-declaring fields it doesn't operate on. Try option (a) first.")
    parts.append("")
    parts.append(f"  (a) Relax the consumer's input schema on {consumer.display_name}. Either:")
    if has_type_mismatch:
        parts.append("      - Change the declared field type(s) to match what the producer emits (see Type mismatches above).")
    if has_missing:
        parts.append("      - Drop missing required fields from the consumer's required_fields if the consumer doesn't actually need them.")
    if has_extras:
        parts.append(
            "      - Switch the consumer's input schema mode to 'flexible' or 'observed' so it accepts the producer's extra fields."
        )
    parts.append(
        "      - Or switch the consumer's input schema mode to 'flexible' so it accepts the producer's full output without redeclaring every field."
    )
    parts.append(f"      Tool: {consumer.schema_patch_tool_call}")
    parts.append("")
    parts.append(
        f"  (b) Patch the producer {producer.display_name}. Note: plugin output schemas are largely baked-in by the plugin's contract — "
        f"this option only works if you mis-declared the producer's schema in your initial set_pipeline / upsert_node call. "
        f"If the producer is using its plugin's default output contract, option (a) is the only fix."
    )
    parts.append(f"      Tool: {producer.schema_patch_tool_call}")

    return "\n".join(parts)


def _skipped_checks(from_check: str) -> list[ValidationCheck]:
    """Generate skipped check records for all checks after from_check."""
    skipping = False
    result: list[ValidationCheck] = []
    for name in _ALL_CHECKS:
        if name == from_check:
            skipping = True
            continue
        if skipping:
            result.append(
                ValidationCheck(
                    name=name,
                    passed=False,
                    detail=f"Skipped: {from_check} failed",
                    affected_nodes=(),
                    outcome_code=CHECK_OUTCOME_SKIPPED_AFTER_FAILURE,
                )
            )
    return result


def _format_interpretation_site(site: InterpretationReviewSite) -> str:
    return f"{site.kind.value} review pending for {site.component_type} {site.component_id!r}: {site.user_term}"


def _format_interpretation_sites(sites: Sequence[InterpretationReviewSite]) -> str:
    return ", ".join(_format_interpretation_site(site) for site in sites)


@dataclass(frozen=True, slots=True)
class _IdentityFinding:
    """One detected identity-shaped passthrough between a transform and a sink.

    Emitted by ``_find_identity_node_advisories``; consumed by the advisory
    block in ``validate_pipeline``.  All four fields are scalars, so
    ``frozen=True`` is sufficient (no ``deep_freeze`` guard needed).

    Attributes:
        node_id: ID of the passthrough node itself.
        upstream_id: ID of the producer feeding the passthrough's input
            (or "source" when the source feeds it directly).
        sink_name: Name of the downstream sink (output) the passthrough emits to.
        sink_schema_mode: Schema mode of the sink ("fixed" / "flexible" /
            "observed"), or ``None`` when the sink declares no schema mode.
            Used purely for the advisory's detail string — not a detection input.
    """

    node_id: str
    upstream_id: str
    sink_name: str
    sink_schema_mode: str | None


def _find_identity_node_advisories(state: CompositionState) -> list[_IdentityFinding]:
    """Detect identity-shaped passthrough transforms between a real transform and a sink.

    A node is flagged iff ALL of the following hold:

    1. ``node_type == "transform"`` and ``plugin == "passthrough"`` (literal
       string check — deliberately narrow per dispatch; broader registry-based
       detection of ``passes_through_input`` plugins is out of scope).
    2. Exactly one upstream producer feeds ``node.input`` (single inbound).
    3. ``on_success`` targets exactly one sink (output by name) — the
       downstream must be a sink, not another transform.
    4. The node has no fork machinery (``fork_to``, ``routes`` empty).
    5. ``options["schema"]["fields"]`` is missing or empty (not Concept-5
       schema-anchoring per ``pipeline_composer.md:758-768``).
    6. The upstream node is NOT a ``gate`` (per ``pipeline_composer.md:1517-1518``,
       per-fork-branch passthrough is the documented legitimate pattern).

    Returns:
        List of :class:`_IdentityFinding`, one per detected node.  Empty when
        nothing was detected.
    """
    findings: list[_IdentityFinding] = []

    output_by_name = {output.name: output for output in state.outputs}
    nodes_by_id = {node.id: node for node in state.nodes}

    # Producer index: maps a connection-target name (the value carried by an
    # upstream's on_success / on_error / route value / fork_to entry) back to
    # the producer node id.  Used to find a node's upstream by matching its
    # input field.  Explicit "if key not in dict" preserves first-writer-wins
    # semantics; the schema validator rejects duplicate connection targets
    # earlier in the pipeline so collisions here would already have surfaced.
    producer_by_target: dict[str, str] = {}

    def _record(target: str, producer_id: str) -> None:
        if target not in producer_by_target:
            producer_by_target[target] = producer_id

    for source_name, source in state.sources.items():
        if source.on_success:
            producer_id = "source" if source_name == "source" else f"source:{source_name}"
            producer_by_target[source.on_success] = producer_id
    for upstream in state.nodes:
        if upstream.on_success:
            _record(upstream.on_success, upstream.id)
        if upstream.on_error:
            _record(upstream.on_error, upstream.id)
        if upstream.routes:
            for route_target in upstream.routes.values():
                _record(route_target, upstream.id)
        if upstream.fork_to:
            for fork_target in upstream.fork_to:
                _record(fork_target, upstream.id)

    for node in state.nodes:
        # Rule 1: identity passthrough plugin (literal name).
        if node.node_type != "transform" or node.plugin != "passthrough":
            continue
        # Rule 4: no fork machinery on the node itself.
        if node.fork_to or node.routes:
            continue
        # Rule 3: on_success must point to exactly one sink (output).
        if node.on_success is None or node.on_success not in output_by_name:
            continue
        sink = output_by_name[node.on_success]
        # Rule 2: must have an upstream producer.  Absence means the pipeline
        # has a dangling input ref, which a structural check will already have
        # surfaced; the advisory simply skips the node.
        if node.input not in producer_by_target:
            continue
        upstream_id = producer_by_target[node.input]
        # Rule 6: upstream is not a gate (per skill lines 1517-1518 —
        # per-fork-branch passthrough is the documented legitimate pattern).
        # ``upstream_id == "source"`` is not in nodes_by_id; the source is
        # never a gate, so falling through is correct.
        if upstream_id in nodes_by_id and nodes_by_id[upstream_id].node_type == "gate":
            continue
        # Rule 5: passthrough has no schema.fields anchor (Concept-5 exemption
        # per skill lines 758-768).  ``options`` values are Tier-3 (LLM- or
        # operator-supplied), so isinstance() dispatches the optional schema
        # block legitimately — a non-Mapping value means "no schema declared".
        schema_block = node.options.get("schema")
        if isinstance(schema_block, Mapping):
            fields = schema_block.get("fields")
            if isinstance(fields, (list, tuple)) and len(fields) > 0:
                continue
        # Compute sink schema mode for the advisory's detail string.  Same
        # Tier-3 dispatch: sink options are operator-supplied, schema may be
        # absent or shaped differently than expected.
        sink_schema_mode: str | None = None
        sink_schema_block = sink.options.get("schema")
        if isinstance(sink_schema_block, Mapping):
            mode = sink_schema_block.get("mode")
            if isinstance(mode, str):
                sink_schema_mode = mode
        findings.append(
            _IdentityFinding(
                node_id=node.id,
                upstream_id=upstream_id,
                sink_name=sink.name,
                sink_schema_mode=sink_schema_mode,
            )
        )
    return findings


def _collect_secret_refs(obj: Any, env_ref_names: set[str] | None = None) -> list[str]:
    """Walk a nested dict/list/Mapping structure and collect all secret_ref names."""
    refs: list[str] = []
    if isinstance(obj, Mapping):
        if len(obj) == 1 and "secret_ref" in obj:
            ref = obj["secret_ref"]
            if isinstance(ref, str):
                refs.append(ref)
                return refs
        for v in obj.values():
            refs.extend(_collect_secret_refs(v, env_ref_names))
    elif isinstance(obj, (list, tuple)):
        for item in obj:
            refs.extend(_collect_secret_refs(item, env_ref_names))
    else:
        ref = secret_env_ref_name(obj, env_ref_names or frozenset())
        if ref is not None:
            refs.append(ref)
    return refs


def _blob_inline_component_id(field_path: str) -> str | None:
    if field_path == "(aggregate)":
        return None
    if field_path.startswith("source."):
        return "source"
    if field_path.startswith("node:"):
        component, _separator, _rest = field_path.partition(".")
        return component[len("node:") :]
    if field_path.startswith("output:"):
        component, _separator, _rest = field_path.partition(".")
        return component[len("output:") :]
    return field_path


def _blob_inline_component_type(field_path: str) -> str | None:
    if field_path == "(aggregate)":
        return None
    if field_path.startswith("source."):
        return "source"
    if field_path.startswith("node:"):
        return "transform"
    if field_path.startswith("output:"):
        return "sink"
    return None


def _blob_inline_validation_detail(violations: list[BlobInlineValidationViolation]) -> str:
    return "; ".join(f"{violation.field_path}: {violation.category}" for violation in violations)


def _blob_inline_validation_error(violation: BlobInlineValidationViolation) -> ValidationError:
    return ValidationError(
        component_id=_blob_inline_component_id(violation.field_path),
        component_type=_blob_inline_component_type(violation.field_path),
        message=f"Inline content blob reference at {violation.field_path} is {violation.category}: {violation.detail}",
        suggestion="Verify the blob exists, is ready, is under the configured size caps, and matches the pinned sha256.",
        error_code=f"{violation.category}_inline_blob_content",
    )


def validate_pipeline(
    state: CompositionState,
    settings: ValidationSettings,
    yaml_generator: YamlGenerator,
    *,
    secret_service: WebSecretResolver | None = None,
    user_id: str | None = None,
    blob_get_metadata: Callable[[UUID], BlobRecord | None] | None = None,
    allow_pending_interpretation_placeholders: bool = False,
) -> ValidationResult:
    """Dry-run validation through the real engine code path.

    Steps:
    1. Source path allowlist check (C3/S2 defense-in-depth)
    1b. Secret ref validation (all referenced secrets exist)
    2. Generate YAML from CompositionState
    3. Load settings via load_settings_from_yaml_string() — resolve secret
       refs first if present, matching the execution service path exactly
    4. instantiate_runtime_plugins(settings, preflight_mode=True)
    5. build_runtime_graph(settings, bundle)
    6. graph.validate() + graph.validate_edge_compatibility()

    Catches and converts to structured ``ValidationResult(is_valid=False)``:
    ``PydanticValidationError``, ``ValueError``, ``TypeError`` (settings load),
    ``PluginNotFoundError``, ``PluginConfigError`` (plugin instantiation),
    ``FileExistsError`` (file-sink path collision under ``fail_if_exists`` /
    exhausted ``auto_increment`` — Tier 3 fs-boundary condition),
    ``GraphValidationError`` (structural), ``RouteValidationError`` (route
    target resolution). All other exceptions propagate (W18) — those are
    Tier 1 invariant breaks that must surface as a 500 to the composer
    failure-handling path.

    Args:
        state: CompositionState from the session.
        settings: ValidationSettings — exposes data_dir for path resolution and allowlist check.
        yaml_generator: YamlGenerator module/object with generate_yaml() method.
        secret_service: Optional secret resolver for validating secret refs.
        user_id: User ID for scoped secret resolution (required if secret_service is set).
        blob_get_metadata: Optional sync metadata lookup for validate-time
            inline-content blob checks. Runtime content reads stay in the
            execution preflight; validate checks metadata only.
        allow_pending_interpretation_placeholders: When true, composer
            authoring preflight masks unresolved ``{{interpretation:<term>}}``
            tokens before YAML generation. Runtime execution leaves this false.
    """
    checks: list[ValidationCheck] = []
    errors: list[ValidationError] = []

    # Step 0: Empty-composition short-circuit.
    #
    # A CompositionState with no source, no transforms, and no outputs cannot
    # be assembled into ElspethSettings — pydantic would raise
    # ``2 validation errors for ElspethSettings: source/sinks Field required``,
    # which is a Tier-3 boundary violation (internal pydantic model names
    # leaking to a user-facing UI). It also auto-populates the post-
    # ``exit_to_freeform`` view of any guided session, so the surface is
    # exercised on a normal workflow, not just on hand-crafted empty configs.
    #
    # Return a clean ``empty_pipeline`` ValidationResult instead. The frontend
    # subscription guard treats ``empty_pipeline`` as non-broadcast (no chat
    # injection, no validation feedback sent to the LLM) — see
    # ``stores/subscriptions.ts``.
    if not state.sources and not state.nodes and not state.outputs:
        return ValidationResult(
            is_valid=False,
            checks=[
                ValidationCheck(
                    name=_CHECK_SETTINGS,
                    passed=False,
                    detail="Pipeline has no source, transforms, or outputs.",
                    affected_nodes=(),
                    outcome_code=None,
                ),
                *_skipped_checks(_CHECK_SETTINGS),
            ],
            errors=[
                ValidationError(
                    component_id=None,
                    component_type=None,
                    message=("Pipeline is empty. Add a source and at least one output to begin building."),
                    suggestion=("Pick a source plugin (e.g. text, csv) and an output destination, then validate again."),
                    error_code="empty_pipeline",
                ),
            ],
            readiness=_blocked_readiness(
                code="empty_pipeline",
                detail="Pipeline is empty.",
            ),
        )

    # Step 1: Source + sink path allowlist check (C3/S2 defense-in-depth)
    # Local filesystem keys in source/sink options must resolve under allowed
    # directories. Uses the shared helpers from AD-4.
    from elspeth.web.paths import (
        NESTED_LOCAL_PATH_OPTION_KEYS,
        SINK_LOCAL_PATH_OPTION_KEYS,
        SOURCE_LOCAL_PATH_OPTION_KEYS,
        allowed_sink_directories,
        allowed_source_directories,
        resolve_data_path,
    )

    allowed_source_dirs = allowed_source_directories(str(settings.data_dir))
    allowed_sink_dirs = allowed_sink_directories(str(settings.data_dir))
    path_checked = False
    for source_name, source in state.sources.items():
        source_options = dict(source.options)
        source_component = "source" if source_name == "source" else f"source:{source_name}"
        for key in SOURCE_LOCAL_PATH_OPTION_KEYS:
            value = source_options.get(key)
            if value is not None:
                path_checked = True
                resolved = resolve_data_path(value, str(settings.data_dir))
                if not any(resolved.is_relative_to(d) for d in allowed_source_dirs):
                    return ValidationResult(
                        is_valid=False,
                        checks=[
                            ValidationCheck(
                                name=_CHECK_PATH_ALLOWLIST,
                                passed=False,
                                detail=f"Source '{source_name}' {key} '{value}' is outside allowed source directories",
                                affected_nodes=(source_component,),
                                outcome_code=None,
                            ),
                            *_skipped_checks(_CHECK_PATH_ALLOWLIST),
                        ],
                        errors=[
                            ValidationError(
                                component_id=source_component,
                                component_type="source",
                                message=(
                                    f"Path traversal blocked: source '{source_name}' {key}='{value}' resolves outside allowed directories"
                                ),
                                suggestion="Use a file within the blobs directory.",
                                error_code=None,
                            ),
                        ],
                        readiness=_blocked_readiness(
                            code="path_allowlist",
                            detail=f"source '{source_name}' {key} resolves outside allowed source directories",
                            component_id=source_component,
                            component_type="source",
                        ),
                    )

    # Sink path allowlist — prevents arbitrary file writes via sink options.
    for output in state.outputs or ():
        for key in SINK_LOCAL_PATH_OPTION_KEYS:
            value = output.options.get(key)
            if value is not None:
                path_checked = True
                resolved = resolve_data_path(value, str(settings.data_dir))
                if not any(resolved.is_relative_to(d) for d in allowed_sink_dirs):
                    return ValidationResult(
                        is_valid=False,
                        checks=[
                            ValidationCheck(
                                name=_CHECK_PATH_ALLOWLIST,
                                passed=False,
                                detail=f"Sink '{output.name}' {key} '{value}' is outside allowed output directories",
                                affected_nodes=(),
                                outcome_code=None,
                            ),
                            *_skipped_checks(_CHECK_PATH_ALLOWLIST),
                        ],
                        errors=[
                            ValidationError(
                                component_id=output.name,
                                component_type="sink",
                                message=f"Path traversal blocked: sink '{output.name}' {key}='{value}' resolves outside allowed directories",
                                suggestion="Use a path within the outputs or blobs directory.",
                                error_code=None,
                            ),
                        ],
                        readiness=_blocked_readiness(
                            code="path_allowlist",
                            detail=f"sink {output.name} {key} resolves outside allowed output directories",
                            component_id=output.name,
                            component_type="sink",
                        ),
                    )

    # Nested transform provider_config path allowlist — RAG retrieval
    # transforms carry a local Chroma persist_directory under
    # options.provider_config. It is a read/write target like a sink, so it is
    # confined to the allowed SINK directories.
    for node in state.nodes:
        if node.node_type != "transform":
            continue
        provider_config = node.options.get("provider_config")
        if not isinstance(provider_config, Mapping):
            continue
        for key in NESTED_LOCAL_PATH_OPTION_KEYS:
            value = provider_config.get(key)
            if value is not None:
                path_checked = True
                resolved = resolve_data_path(value, str(settings.data_dir))
                if not any(resolved.is_relative_to(d) for d in allowed_sink_dirs):
                    return ValidationResult(
                        is_valid=False,
                        checks=[
                            ValidationCheck(
                                name=_CHECK_PATH_ALLOWLIST,
                                passed=False,
                                detail=f"Transform '{node.id}' {key} '{value}' is outside allowed output directories",
                                affected_nodes=(),
                                outcome_code=None,
                            ),
                            *_skipped_checks(_CHECK_PATH_ALLOWLIST),
                        ],
                        errors=[
                            ValidationError(
                                component_id=node.id,
                                component_type="transform",
                                message=f"Path traversal blocked: transform '{node.id}' {key}='{value}' resolves outside allowed directories",
                                suggestion="Use a path within the outputs or blobs directory.",
                                error_code=None,
                            ),
                        ],
                        readiness=_blocked_readiness(
                            code="path_allowlist",
                            detail=f"transform {node.id} {key} resolves outside allowed output directories",
                            component_id=node.id,
                            component_type="transform",
                        ),
                    )

    # B11 fix: Always record the path_allowlist check
    if path_checked:
        checks.append(
            ValidationCheck(
                name=_CHECK_PATH_ALLOWLIST,
                passed=True,
                detail="All paths within allowed directories",
                affected_nodes=(),
                outcome_code=None,
            )
        )
    else:
        checks.append(
            ValidationCheck(
                name=_CHECK_PATH_ALLOWLIST,
                passed=True,
                detail="No path option — check skipped",
                affected_nodes=(),
                outcome_code=None,
            )
        )

    web_scrape_network_errors: list[ValidationError] = []
    for node in state.nodes:
        if node.plugin != "web_scrape":
            continue
        http_options = node.options.get("http")
        if not isinstance(http_options, Mapping):
            continue
        if "allowed_hosts" not in http_options:
            continue
        allowed_hosts = http_options["allowed_hosts"]
        if allowed_hosts == "allow_private":
            message = (
                "web_scrape.http.allowed_hosts='allow_private' is not permitted in web execution. "
                "Web-authored pipelines may only use public SSRF policy; private-network fetching requires "
                "an operator-owned runtime outside the web composer."
            )
        elif isinstance(allowed_hosts, Sequence) and not isinstance(allowed_hosts, str):
            message = (
                "web_scrape.http.allowed_hosts CIDR allowlists are not permitted in web execution. "
                "Web-authored pipelines may only use allowed_hosts='public_only'; private-network fetching "
                "requires an operator-owned runtime outside the web composer."
            )
        else:
            continue
        web_scrape_network_errors.append(
            ValidationError(
                component_id=node.id,
                component_type="transform",
                message=message,
                suggestion="Set web_scrape.http.allowed_hosts to 'public_only' or remove the option.",
                error_code="web_scrape_private_network_not_allowed",
            )
        )

    if web_scrape_network_errors:
        affected_nodes = tuple(error.component_id for error in web_scrape_network_errors if error.component_id is not None)
        checks.append(
            ValidationCheck(
                name=_CHECK_WEB_SCRAPE_NETWORK_POLICY,
                passed=False,
                detail="web_scrape private-network allowlists are not permitted in web execution",
                affected_nodes=affected_nodes,
                outcome_code=None,
            )
        )
        checks.extend(_skipped_checks(_CHECK_WEB_SCRAPE_NETWORK_POLICY))
        return ValidationResult(
            is_valid=False,
            checks=checks,
            errors=web_scrape_network_errors,
            readiness=_blocked_readiness(
                code="web_scrape_network_policy",
                detail="web_scrape private-network allowlists are not permitted in web execution.",
                component_id=web_scrape_network_errors[0].component_id,
                component_type="transform",
            ),
        )

    checks.append(
        ValidationCheck(
            name=_CHECK_WEB_SCRAPE_NETWORK_POLICY,
            passed=True,
            detail="No web_scrape private-network allowlists found",
            affected_nodes=(),
            outcome_code=None,
        )
    )

    # Step 1b: Secret ref validation — check that
    #   (a) every wired ``{secret_ref: ...}`` / inventory ``${NAME}`` resolves
    #       (existing missing-refs gate), AND
    #   (b) every credential-bearing field (per ``is_secret_field``) contains
    #       a wired secret rather than a literal placeholder string
    #       (issue elspeth-72d1dccd44 — S1A "WILL_BE_WIRED_FROM_..." defect).
    #
    # The remediation note (docs/composer/evidence/composer-remediation-program-2026-05-01.md)
    # suggested "the catalog already knows which fields require secrets."  In
    # practice the catalog renders the per-plugin schema but does not mark
    # credential fields; the closed list of credential-bearing field names
    # already lives in ``elspeth.core.secrets.is_secret_field`` and is shared
    # with the runtime fingerprinting code path.  Reusing it here keeps
    # validate-time and runtime in lock-step — divergence would re-open the
    # parity gap this issue was filed to close.
    all_refs: list[str] = []
    env_ref_names: set[str] = set()
    fabricated_components: list[tuple[str | None, str | None, list[str]]] = []
    disallowed_secret_ref_components: list[tuple[str | None, str, str, list[SecretRefPlacementViolation]]] = []
    if secret_service is not None and user_id is not None:
        env_ref_names = {item.name for item in secret_service.list_refs(user_id)}
        # Walk source options, node configs, and output options for secret refs
        for source_name, source in state.sources.items():
            source_component = "source" if source_name == "source" else f"source:{source_name}"
            all_refs.extend(_collect_secret_refs(source.options, env_ref_names))
            fabricated = collect_credential_field_violations(source.options, env_ref_names)
            if fabricated:
                fabricated_components.append((source_component, "source", fabricated))
            disallowed = collect_disallowed_secret_ref_markers(
                source.options,
                env_ref_names,
                additional_allowed_fields=allowed_secret_ref_fields("source", source.plugin),
            )
            if disallowed:
                disallowed_secret_ref_components.append((source_component, "source", source.plugin, disallowed))
        for node in state.nodes or ():
            all_refs.extend(_collect_secret_refs(node.options, env_ref_names))
            fabricated = collect_credential_field_violations(node.options, env_ref_names)
            if fabricated:
                fabricated_components.append((node.id, "transform", fabricated))
            node_plugin = node.plugin or "<unset>"
            disallowed = collect_disallowed_secret_ref_markers(
                node.options,
                env_ref_names,
                additional_allowed_fields=allowed_secret_ref_fields("transform", node_plugin),
            )
            if disallowed:
                disallowed_secret_ref_components.append((node.id, "transform", node_plugin, disallowed))
        for output in state.outputs or ():
            all_refs.extend(_collect_secret_refs(output.options, env_ref_names))
            fabricated = collect_credential_field_violations(output.options, env_ref_names)
            if fabricated:
                fabricated_components.append((output.name, "sink", fabricated))
            disallowed = collect_disallowed_secret_ref_markers(
                output.options,
                env_ref_names,
                additional_allowed_fields=allowed_secret_ref_fields("sink", output.plugin),
            )
            if disallowed:
                disallowed_secret_ref_components.append((output.name, "sink", output.plugin, disallowed))

        missing_refs = [ref for ref in all_refs if not secret_service.has_ref(user_id, ref)]
        if missing_refs or fabricated_components or disallowed_secret_ref_components:
            detail_parts: list[str] = []
            if missing_refs:
                names = ", ".join(missing_refs)
                detail_parts.append(f"Missing secret references: {names}")
                errors.append(
                    ValidationError(
                        component_id=None,
                        component_type=None,
                        message=f"Cannot resolve secret references: {names}",
                        suggestion="Add the missing secrets via the Secrets panel before executing.",
                        error_code="missing_secret_ref",
                    )
                )
            if fabricated_components:
                fabricated_summary = ", ".join(f"{cid}:{','.join(fields)}" for cid, _ctype, fields in fabricated_components)
                detail_parts.append("Literal value in credential field(s): " + fabricated_summary)
                # Audit hygiene: name the field, never echo the value.  A value
                # that looks like a placeholder may be a near-miss real secret
                # and the validation response is operator-visible.
                for component_id, component_type, fields in fabricated_components:
                    fields_text = ", ".join(fields)
                    errors.append(
                        ValidationError(
                            component_id=component_id,
                            component_type=component_type,
                            message=(f"Credential field(s) {fields_text} contain a literal value; expected a wired secret reference."),
                            suggestion=(
                                "Wire each credential field through the Secrets panel "
                                "(produces a {secret_ref: NAME} marker) instead of typing "
                                "the value directly."
                            ),
                            error_code="fabricated_secret",
                        )
                    )
            if disallowed_secret_ref_components:
                for component_id, component_type, plugin_name, violations in disallowed_secret_ref_components:
                    violation_text = ", ".join(f"{v.field_path} -> {v.secret_name}" for v in violations)
                    allowed_text = allowed_secret_ref_fields_text(component_type, plugin_name)
                    detail_parts.append(f"Disallowed secret_ref for {plugin_name} {component_id}: {violation_text}")
                    for violation in violations:
                        errors.append(
                            ValidationError(
                                component_id=component_id,
                                component_type=component_type,
                                message=(
                                    f"Plugin '{plugin_name}' field '{violation.field_path}' contains secret_ref "
                                    f"'{violation.secret_name}', but only credential-bearing fields may carry secret_ref "
                                    f"markers. Allowed credential-bearing fields for this plugin: {allowed_text}."
                                ),
                                suggestion=(
                                    "Move the secret_ref marker to an actual credential field, or use a literal "
                                    "non-secret value for wire-visible identity/configuration fields."
                                ),
                                error_code="disallowed_secret_ref",
                            )
                        )
            checks.append(
                ValidationCheck(
                    name=_CHECK_SECRET_REFS,
                    passed=False,
                    detail="; ".join(detail_parts),
                    affected_nodes=(),
                    outcome_code=CHECK_OUTCOME_SECRET_REFS_UNRESOLVED,
                )
            )
            checks.extend(_skipped_checks(_CHECK_SECRET_REFS))
            return ValidationResult(
                is_valid=False,
                checks=checks,
                errors=errors,
                readiness=_blocked_readiness(
                    code="secret_refs",
                    detail="Secret reference validation failed.",
                ),
            )
        checks.append(
            ValidationCheck(
                name=_CHECK_SECRET_REFS,
                passed=True,
                detail=f"All {len(all_refs)} secret reference(s) resolved" if all_refs else "No secret references found",
                affected_nodes=(),
                outcome_code=CHECK_OUTCOME_SECRET_REFS_RESOLVED if all_refs else CHECK_OUTCOME_SECRET_REFS_NO_REFS,
            )
        )
    else:
        checks.append(
            ValidationCheck(
                name=_CHECK_SECRET_REFS,
                passed=True,
                detail="No secret service — check skipped",
                affected_nodes=(),
                outcome_code=CHECK_OUTCOME_SECRET_REFS_SKIPPED_NO_SERVICE,
            )
        )

    semantic_errors, semantic_contracts = validate_semantic_contracts(state)
    if semantic_errors:
        checks.append(
            ValidationCheck(
                name=_CHECK_SEMANTIC_CONTRACTS,
                passed=False,
                detail="Semantic contract check failed",
                affected_nodes=(),
                outcome_code=None,
            )
        )
        for entry in semantic_errors:
            # entry.message already names plugins, fields, requirement code.
            # Suggestion is plugin-owned — fetch from PluginAssistance.
            errors.append(
                ValidationError(
                    component_id=entry.component.removeprefix("node:"),
                    component_type="transform",
                    message=entry.message,
                    suggestion=assistance_suggestion_for(entry, semantic_contracts),
                    error_code=None,
                )
            )
        checks.extend(_skipped_checks(_CHECK_SEMANTIC_CONTRACTS))
        return ValidationResult(
            is_valid=False,
            checks=checks,
            errors=errors,
            readiness=_blocked_readiness(
                code="semantic_contracts",
                detail="Semantic contract check failed.",
            ),
            semantic_contracts=serialize_semantic_contracts(semantic_contracts),
        )

    checks.append(
        ValidationCheck(
            name=_CHECK_SEMANTIC_CONTRACTS,
            passed=True,
            detail=(
                f"All {len(semantic_contracts)} semantic contract(s) satisfied" if semantic_contracts else "No semantic contracts to check"
            ),
            affected_nodes=(),
            outcome_code=None,
        )
    )

    batch_option_errors: list[tuple[str, str]] = []
    for node in state.nodes:
        batch_placement_error = _batch_aware_placement_error(node.id, node.node_type, node.plugin, node.output_mode)
        if batch_placement_error is not None:
            batch_option_errors.append((node.id, batch_placement_error))
        batch_required_error = _batch_aware_required_input_fields_error(node.id, node.plugin, node.options)
        if batch_required_error is not None:
            batch_option_errors.append((node.id, batch_required_error))
    numeric_contract_errors, _numeric_contract_warnings = _batch_distribution_profile_value_field_entries(state.sources, state.nodes)
    for entry in numeric_contract_errors:
        batch_option_errors.append((entry.component.removeprefix("node:"), entry.message))
    if batch_option_errors:
        checks.append(
            ValidationCheck(
                name=_CHECK_BATCH_TRANSFORM_OPTIONS,
                passed=False,
                detail="Batch-aware transform option check failed",
                affected_nodes=(),
                outcome_code=None,
            )
        )
        for node_id, message in batch_option_errors:
            errors.append(
                ValidationError(
                    component_id=node_id,
                    component_type="transform",
                    message=message,
                    suggestion=(
                        "Use node_type='aggregation' for batch-aware plugins; remove required_input_fields from batch-aware transform "
                        "options and use schema.required_fields for batch input validation."
                    ),
                    error_code=None,
                )
            )
        checks.extend(_skipped_checks(_CHECK_BATCH_TRANSFORM_OPTIONS))
        return ValidationResult(
            is_valid=False,
            checks=checks,
            errors=errors,
            readiness=_blocked_readiness(
                code="batch_transform_options",
                detail="Batch-aware transform option check failed.",
            ),
            semantic_contracts=serialize_semantic_contracts(semantic_contracts),
        )
    checks.append(
        ValidationCheck(
            name=_CHECK_BATCH_TRANSFORM_OPTIONS,
            passed=True,
            detail="Batch-aware transform options are compatible with ADR-013",
            affected_nodes=(),
            outcome_code=None,
        )
    )

    materialized_state = (
        materialize_state_for_authoring(state) if allow_pending_interpretation_placeholders else materialize_state_for_execution(state)
    )
    if isinstance(materialized_state, InterpretationReviewPending):
        site_detail = _format_interpretation_sites(materialized_state.sites)
        affected_nodes = tuple(dict.fromkeys(site.component_id for site in materialized_state.sites if site.component_type == "transform"))
        checks.append(
            ValidationCheck(
                name=_CHECK_INTERPRETATION_REVIEW,
                passed=False,
                detail=f"Interpretation review is pending for {site_detail}.",
                affected_nodes=affected_nodes,
                outcome_code=None,
            )
        )
        errors.extend(
            ValidationError(
                component_id=site.component_id,
                component_type=site.component_type,
                message=_format_interpretation_site(site),
                suggestion="Resolve the pending interpretation review before running.",
                error_code=INTERPRETATION_REVIEW_PENDING_CODE,
            )
            for site in materialized_state.sites
        )
        checks.extend(_skipped_checks(_CHECK_INTERPRETATION_REVIEW))
        single_site = materialized_state.sites[0] if len(materialized_state.sites) == 1 else None
        return ValidationResult(
            is_valid=False,
            checks=checks,
            errors=errors,
            readiness=_blocked_readiness(
                code=INTERPRETATION_REVIEW_PENDING_CODE,
                detail=site_detail,
                component_id=single_site.component_id if single_site is not None else None,
                component_type=single_site.component_type if single_site is not None else None,
                authoring_valid=True,
                completion_ready=True,
            ),
            semantic_contracts=serialize_semantic_contracts(semantic_contracts),
        )
    checks.append(
        ValidationCheck(
            name=_CHECK_INTERPRETATION_REVIEW,
            passed=True,
            detail="No pending interpretation review",
            affected_nodes=(),
            outcome_code=None,
        )
    )

    # Step 2: Generate YAML
    pipeline_yaml = yaml_generator.generate_yaml(materialized_state)
    pipeline_yaml = resolve_runtime_yaml_paths(pipeline_yaml, str(settings.data_dir))

    if blob_get_metadata is not None and "blob_ref" in pipeline_yaml and "inline_content" in pipeline_yaml:
        config_dict = yaml.safe_load(pipeline_yaml)
        if type(config_dict) is not dict:
            raise TypeError(
                f"generate_yaml() produced non-dict YAML (got {type(config_dict).__name__}) — this is a bug in the YAML generator"
            )
        blob_violations = _validate_blob_content_refs_sync(
            blob_get_metadata,
            config_dict,
            per_ref_byte_cap=BLOB_INLINE_PER_REF_BYTE_CAP,
            aggregate_byte_cap=BLOB_INLINE_AGGREGATE_BYTE_CAP,
        )
        if blob_violations:
            detail = _blob_inline_validation_detail(blob_violations)
            checks.append(
                ValidationCheck(
                    name=_CHECK_BLOB_INLINE_REFS,
                    passed=False,
                    detail=detail,
                    affected_nodes=(),
                    outcome_code=None,
                )
            )
            errors.extend(_blob_inline_validation_error(violation) for violation in blob_violations)
            checks.extend(_skipped_checks(_CHECK_BLOB_INLINE_REFS))
            return ValidationResult(
                is_valid=False,
                checks=checks,
                errors=errors,
                readiness=_blocked_readiness(
                    code="blob_inline_refs",
                    detail=detail,
                ),
                semantic_contracts=serialize_semantic_contracts(semantic_contracts),
            )
        checks.append(
            ValidationCheck(
                name=_CHECK_BLOB_INLINE_REFS,
                passed=True,
                detail="All inline-content blob references are valid",
                affected_nodes=(),
                outcome_code=None,
            )
        )
        pipeline_yaml = yaml.dump(_substitute_blob_content_refs_for_validation(config_dict), default_flow_style=False)
    elif blob_get_metadata is None:
        checks.append(
            ValidationCheck(
                name=_CHECK_BLOB_INLINE_REFS,
                passed=True,
                detail="No blob metadata service — check skipped",
                affected_nodes=(),
                outcome_code=None,
            )
        )
    else:
        checks.append(
            ValidationCheck(
                name=_CHECK_BLOB_INLINE_REFS,
                passed=True,
                detail="No inline-content blob references found",
                affected_nodes=(),
                outcome_code=None,
            )
        )

    # Step 3: Settings loading
    #
    # Always uses load_settings_from_yaml_string() — the same loader the
    # execution service uses (in _run_pipeline).  This ensures validation
    # exercises the exact same code path as execution, preventing
    # false-pass or false-fail results from loader differences.
    #
    # When secret refs are present, resolve them before loading.
    # Resolved secrets stay in process memory — never written to disk.
    #
    # SecretResolutionError is NOT caught: if a ref is missing here,
    # Step 1b's existence check was wrong — that's an internal bug
    # and must crash per the W18 rule.
    try:
        settings_yaml = pipeline_yaml
        if secret_service is not None and user_id is not None and all_refs:
            config_dict = yaml.safe_load(pipeline_yaml)
            if not isinstance(config_dict, dict):
                raise TypeError(
                    f"generate_yaml() produced non-dict YAML (got {type(config_dict).__name__}) — this is a bug in the YAML generator"
                )
            resolved_dict, _resolutions = resolve_secret_refs(
                config_dict,
                secret_service,
                user_id,
                env_ref_names=env_ref_names,
            )
            settings_yaml = yaml.dump(resolved_dict, default_flow_style=False)

        # Web-authored pipeline YAML must never expand host ${VAR} placeholders
        # (parity with the execution path). Known secret inventory names are
        # resolved above via resolve_secret_refs; any remaining ${VAR} is
        # user-authored data, not a host-environment lookup.
        elspeth_settings = load_settings_from_yaml_string(settings_yaml, expand_env_vars=False)
        checks.append(
            ValidationCheck(
                name=_CHECK_SETTINGS,
                passed=True,
                detail="Settings loaded successfully",
                affected_nodes=(),
                outcome_code=None,
            )
        )
    except (PydanticValidationError, ValueError, TypeError) as exc:
        checks.append(
            ValidationCheck(
                name=_CHECK_SETTINGS,
                passed=False,
                detail=str(exc),
                affected_nodes=(),
                outcome_code=None,
            )
        )
        errors.append(
            ValidationError(
                component_id=None,
                component_type=None,
                message=str(exc),
                suggestion=None,
                error_code=None,
            )
        )
        checks.extend(_skipped_checks(_CHECK_SETTINGS))
        return ValidationResult(
            is_valid=False,
            checks=checks,
            errors=errors,
            readiness=_blocked_readiness(
                code="settings_load",
                detail="Settings failed to load.",
            ),
            semantic_contracts=serialize_semantic_contracts(semantic_contracts),
        )

    # Step 4: Plugin instantiation + value-source compliance
    #
    # ``instantiate_plugins_from_config`` now runs the value-source walker
    # against ``bundle.transforms`` before returning, so a hallucinated
    # model identifier (or any field that violates a plugin's
    # VALUE_SOURCES declaration) raises ``ValueSourceValidationError``
    # from inside this call. We disambiguate the two failure classes by
    # exception type:
    #
    # * ``PluginNotFoundError`` / ``PluginConfigError`` / ``FileExistsError``
    #   → the bundle was never built; PLUGINS check failed, value-source
    #   compliance is skipped via cascade (it could not run).
    # * ``ValueSourceValidationError`` → the bundle was built successfully
    #   but rejected by the walker; PLUGINS check passed, VALUE_SOURCE
    #   compliance check failed, downstream checks skipped via cascade.
    try:
        bundle = instantiate_runtime_plugins(elspeth_settings, preflight_mode=True)
        checks.append(
            ValidationCheck(
                name=_CHECK_PLUGINS,
                passed=True,
                detail="All plugins instantiated",
                affected_nodes=(),
                outcome_code=None,
            )
        )
        checks.append(
            ValidationCheck(
                name=_CHECK_VALUE_SOURCE_COMPLIANCE,
                passed=True,
                detail="All declared value sources satisfied",
                affected_nodes=(),
                outcome_code=None,
            )
        )
    except ValueSourceValidationError as exc:
        # Bundle was built — instantiation succeeded — but the walker
        # rejected one or more declared field values. Surface PLUGINS
        # as passed, VALUE_SOURCE as failed.
        checks.append(
            ValidationCheck(
                name=_CHECK_PLUGINS,
                passed=True,
                detail="All plugins instantiated",
                affected_nodes=(),
                outcome_code=None,
            )
        )
        checks.append(
            ValidationCheck(
                name=_CHECK_VALUE_SOURCE_COMPLIANCE,
                passed=False,
                detail=str(exc),
                affected_nodes=(),
                outcome_code=None,
            )
        )
        # Each finding names a single ``component_id`` field-violation —
        # surface them as separate ValidationError records so the composer
        # UI can attribute each to its node. ``finding`` is a
        # :class:`ValueSourceFinding` carrying the attribution structurally
        # — no string parsing, no silent ``component_id=None`` fallback.
        for finding in exc.findings:
            errors.append(
                ValidationError(
                    component_id=finding.component_id,
                    component_type="transform",
                    message=finding.format(),
                    suggestion=(
                        "Use the list_models composer tool to pick a known "
                        "model identifier; for Azure transforms, leave 'model' "
                        "empty so it inherits from 'deployment_name'."
                    ),
                    error_code=None,
                )
            )
        checks.extend(_skipped_checks(_CHECK_VALUE_SOURCE_COMPLIANCE))
        return ValidationResult(
            is_valid=False,
            checks=checks,
            errors=errors,
            readiness=_blocked_readiness(
                code="value_source_compliance",
                detail="Value-source compliance failed.",
                component_id=errors[-1].component_id if errors else None,
                component_type=errors[-1].component_type if errors else None,
            ),
            semantic_contracts=serialize_semantic_contracts(semantic_contracts),
        )
    except (PluginNotFoundError, PluginConfigError) as exc:
        comp_type = _infer_component_type_from_plugin_error(exc)
        plugin_error_name: str | None = exc.plugin_name if isinstance(exc, PluginConfigError) else None
        # Prefer cause (validation detail) over str(exc) which includes the
        # internal class name prefix (e.g. "Invalid configuration for CSVSourceConfig: ...").
        if isinstance(exc, PluginConfigError) and exc.cause is not None and plugin_error_name is not None:
            detail = f"Invalid configuration for {comp_type} '{plugin_error_name}': {exc.cause}"
        else:
            detail = str(exc)
        checks.append(
            ValidationCheck(
                name=_CHECK_PLUGINS,
                passed=False,
                detail=detail,
                affected_nodes=(),
                outcome_code=None,
            )
        )
        errors.append(
            ValidationError(
                component_id=plugin_error_name,
                component_type=comp_type,
                message=detail,
                suggestion=None,
                error_code=None,
            )
        )
        checks.extend(_skipped_checks(_CHECK_PLUGINS))
        return ValidationResult(
            is_valid=False,
            checks=checks,
            errors=errors,
            readiness=_blocked_readiness(
                code="plugin_instantiation",
                detail="Plugin instantiation failed.",
                component_id=plugin_error_name,
                component_type=comp_type,
            ),
            semantic_contracts=serialize_semantic_contracts(semantic_contracts),
        )
    except FileExistsError as exc:
        # Belt-and-braces conversion for plugins that still perform filesystem
        # collision checks during construction. Built-in local file sinks skip
        # this check during runtime preflight and defer it to first write.
        # Known raise sites in ``plugins/infrastructure/output_paths.py``:
        #
        # * line 48 — ``collision_policy="fail_if_exists"`` and the target path
        #   already exists.
        # * line 73 — ``collision_policy="auto_increment"`` and 9999 sibling
        #   slots are all taken.
        #
        # Per CLAUDE.md trust tiers, the existing-file condition is a Tier 3
        # boundary fact (external filesystem state), not a Tier 1 invariant
        # break — convert to a structured validation result here so the
        # composer ``/validate`` and ``/messages`` paths surface a 422-class
        # diagnostic instead of an opaque 500 ``composer_plugin_error``. The
        # exception does not carry sink-name attribution at this layer
        # (sinks raise ``FileExistsError`` directly without wrapping); the
        # message text contains the path, which is operator-actionable.
        detail = str(exc)
        checks.append(
            ValidationCheck(
                name=_CHECK_PLUGINS,
                passed=False,
                detail=detail,
                affected_nodes=(),
                outcome_code=None,
            )
        )
        errors.append(
            ValidationError(
                component_id=None,
                component_type="sink",
                message=detail,
                suggestion=("Set collision_policy='auto_increment' to pick a free sibling path automatically, or choose a different path."),
                error_code=None,
            )
        )
        checks.extend(_skipped_checks(_CHECK_PLUGINS))
        return ValidationResult(
            is_valid=False,
            checks=checks,
            errors=errors,
            readiness=_blocked_readiness(
                code="plugin_instantiation",
                detail="Plugin filesystem target validation failed.",
                component_type="sink",
            ),
            semantic_contracts=serialize_semantic_contracts(semantic_contracts),
        )

    # Step 5: Graph construction + structural validation
    try:
        graph = build_runtime_graph(elspeth_settings, bundle)
        graph.validate()
        checks.append(
            ValidationCheck(
                name=_CHECK_GRAPH,
                passed=True,
                detail="Graph structure is valid",
                affected_nodes=(),
                outcome_code=None,
            )
        )
    except GraphValidationError as exc:
        checks.append(
            ValidationCheck(
                name=_CHECK_GRAPH,
                passed=False,
                detail=str(exc),
                affected_nodes=(),
                outcome_code=None,
            )
        )
        errors.append(
            ValidationError(
                component_id=exc.component_id,
                component_type=exc.component_type,
                message=str(exc),
                suggestion=None,
                error_code=None,
            )
        )
        checks.extend(_skipped_checks(_CHECK_GRAPH))
        return ValidationResult(
            is_valid=False,
            checks=checks,
            errors=errors,
            readiness=_blocked_readiness(
                code="graph_structure",
                detail="Graph validation failed.",
                component_id=exc.component_id,
                component_type=exc.component_type,
            ),
            semantic_contracts=serialize_semantic_contracts(semantic_contracts),
        )

    # Step 5b: Route target resolution
    #
    # The orchestrator's pre-init runs four route-target validators that
    # ``graph.validate()`` does not cover:
    # ``validate_route_destinations``, ``validate_transform_error_sinks``,
    # ``validate_source_quarantine_destination``, and
    # ``validate_sink_failsink_destinations``. Without this step the composer's
    # ``/validate`` returns ``is_valid: true`` for pipelines whose
    # ``on_error`` / ``on_validation_failure`` / ``on_write_failure`` /
    # ``gates[*].routes[*]`` reference a non-existent sink, and the runtime
    # then rejects them at ``/execute`` pre-token (issue elspeth-127de6865a).
    #
    # ``OrchestrationInvariantError`` is intentionally NOT caught — it
    # signals a framework bug (e.g. transform on_error is None when
    # TransformSettings requires it) and must surface as a 500, not as a
    # per-pipeline validation failure.
    try:
        assemble_and_validate_pipeline_config(
            sources=bundle.sources,
            transforms=bundle.transforms,
            sinks=bundle.sinks,
            aggregations=bundle.aggregations,
            settings=elspeth_settings,
            graph=graph,
        )
        checks.append(
            ValidationCheck(
                name=_CHECK_ROUTE_TARGETS,
                passed=True,
                detail="All route targets resolve to existing sinks",
                affected_nodes=(),
                outcome_code=None,
            )
        )
    except RouteValidationError as exc:
        checks.append(
            ValidationCheck(
                name=_CHECK_ROUTE_TARGETS,
                passed=False,
                detail=str(exc),
                affected_nodes=(),
                outcome_code=None,
            )
        )
        errors.append(
            ValidationError(
                component_id=None,
                component_type=None,
                message=str(exc),
                suggestion=("Use 'discard' to drop rows without routing, or wire the destination to an existing sink."),
                error_code=None,
            )
        )
        checks.extend(_skipped_checks(_CHECK_ROUTE_TARGETS))
        return ValidationResult(
            is_valid=False,
            checks=checks,
            errors=errors,
            readiness=_blocked_readiness(
                code="route_target_resolution",
                detail="Route target validation failed.",
            ),
            semantic_contracts=serialize_semantic_contracts(semantic_contracts),
        )

    # Step 6: Schema compatibility
    try:
        graph.validate_edge_compatibility()
        checks.append(
            ValidationCheck(
                name=_CHECK_SCHEMA,
                passed=True,
                detail="All edge schemas compatible",
                affected_nodes=(),
                outcome_code=None,
            )
        )
    except GraphValidationError as exc:
        # ValidationCheck.detail keeps the legacy single-line prose so the
        # operator-facing /validate response and dashboard run-status panel
        # render the same compact summary they always have. The richer
        # multi-line message + suggestion live on the ValidationError below
        # and surface to the composer LLM.
        checks.append(
            ValidationCheck(
                name=_CHECK_SCHEMA,
                passed=False,
                detail=str(exc),
                affected_nodes=(),
                outcome_code=None,
            )
        )
        if isinstance(exc, EdgeContractError):
            consumer_target = _edge_patch_target_for_node_id(
                exc.to_node_id,
                state=state,
                graph=graph,
                component_type=exc.component_type,
            )
            edge_message, edge_suggestion = _format_edge_contract_failure(exc, state=state, graph=graph)
            errors.append(
                ValidationError(
                    component_id=consumer_target.component_id,
                    component_type=consumer_target.component_type,
                    message=edge_message,
                    suggestion=edge_suggestion,
                    error_code=None,
                )
            )
        else:
            errors.append(
                ValidationError(
                    component_id=exc.component_id,
                    component_type=exc.component_type,
                    message=str(exc),
                    suggestion=None,
                    error_code=None,
                )
            )
        return ValidationResult(
            is_valid=False,
            checks=checks,
            errors=errors,
            readiness=_blocked_readiness(
                code="schema_compatibility",
                detail="Schema compatibility failed.",
                component_id=errors[-1].component_id if errors else None,
                component_type=errors[-1].component_type if errors else None,
            ),
            semantic_contracts=serialize_semantic_contracts(semantic_contracts),
        )

    # Identity-node advisory — non-blocking, multi-entry.  Emitted only on the
    # happy path (after every structural check has passed) so structural errors
    # are not drowned in cosmetic noise.  One ValidationCheck per detected node;
    # the detail string names the offending node, its upstream, and its
    # downstream sink, plus the repair action so the composer LLM can self-correct
    # on the next turn.  See dispatch-prompt-floofy-noodle.md plan + skill lines
    # 758-768 (Concept-5 exemption) and 1517-1518 (fork-branch exemption).
    for identity_finding in _find_identity_node_advisories(state):
        sink_mode_text = (
            f", which uses schema.mode: {identity_finding.sink_schema_mode}" if identity_finding.sink_schema_mode is not None else ""
        )
        checks.append(
            ValidationCheck(
                name=_CHECK_IDENTITY_NODE_ADVISORY,
                passed=True,
                detail=(
                    f"Node '{identity_finding.node_id}' is an identity-shaped passthrough "
                    f"between '{identity_finding.upstream_id}' and sink '{identity_finding.sink_name}'"
                    f"{sink_mode_text}.  The sink accepts the upstream row directly; "
                    f"the passthrough adds an audit hop with no contract benefit.  "
                    f"Consider removing it and wiring '{identity_finding.upstream_id}'.on_success "
                    f"directly to '{identity_finding.sink_name}'."
                ),
                # Structured node attribution for the audit-readiness panel's
                # provenance row (see docs/composer/ux-redesign-2026-05/14a
                # §"Six rows — projection mapping").
                affected_nodes=(identity_finding.node_id,),
                outcome_code=None,
            )
        )

    return ValidationResult(
        is_valid=True,
        checks=checks,
        errors=errors,
        readiness=_execution_ready(),
        semantic_contracts=serialize_semantic_contracts(semantic_contracts),
    )
