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

from collections.abc import Mapping
from typing import Any

import yaml
from pydantic import ValidationError as PydanticValidationError

from elspeth.contracts.secrets import WebSecretResolver
from elspeth.core.config import load_settings_from_yaml_string
from elspeth.core.dag.models import GraphValidationError
from elspeth.core.secrets import is_secret_field, resolve_secret_refs, secret_env_ref_name
from elspeth.engine.orchestrator.preflight import assemble_and_validate_pipeline_config
from elspeth.engine.orchestrator.types import RouteValidationError
from elspeth.plugins.infrastructure.config_base import PluginConfigError
from elspeth.plugins.infrastructure.manager import PluginNotFoundError
from elspeth.web.composer._semantic_validator import validate_semantic_contracts
from elspeth.web.composer.state import CompositionState, _batch_aware_required_input_fields_error
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
    ValidationCheck,
    ValidationError,
    ValidationResult,
)

# ── Check names (ordered) ─────────────────────────────────────────────
_CHECK_PATH_ALLOWLIST = "path_allowlist"
_CHECK_SECRET_REFS = "secret_refs"
_CHECK_SEMANTIC_CONTRACTS = "semantic_contracts"
_CHECK_BATCH_TRANSFORM_OPTIONS = "batch_transform_options"
_CHECK_SETTINGS = "settings_load"
_CHECK_PLUGINS = RUNTIME_CHECK_PLUGIN_INSTANTIATION
_CHECK_GRAPH = RUNTIME_CHECK_GRAPH_STRUCTURE
_CHECK_ROUTE_TARGETS = "route_target_resolution"
_CHECK_SCHEMA = RUNTIME_CHECK_SCHEMA_COMPATIBILITY
assert RUNTIME_GRAPH_VALIDATION_CHECKS == (_CHECK_PLUGINS, _CHECK_GRAPH, _CHECK_SCHEMA)

_ALL_CHECKS = [
    _CHECK_PATH_ALLOWLIST,
    _CHECK_SECRET_REFS,
    _CHECK_SEMANTIC_CONTRACTS,
    _CHECK_BATCH_TRANSFORM_OPTIONS,
    _CHECK_SETTINGS,
    _CHECK_PLUGINS,
    _CHECK_GRAPH,
    _CHECK_ROUTE_TARGETS,
    _CHECK_SCHEMA,
]


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
                )
            )
    return result


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


def _value_is_wired_secret(value: Any, env_ref_names: set[str]) -> bool:
    """A value is "wired" if it is either a ``{secret_ref: NAME}`` marker or
    an exact ``${NAME}`` env-marker for a NAME present in the secret inventory.

    Anything else placed in a credential-bearing field — a literal placeholder
    string like ``"WILL_BE_WIRED_FROM_OPENROUTER_API_KEY"``, an env-marker for
    an unregistered name, etc. — is fabrication: the runtime would treat the
    value as the literal credential and authentication would fail per-row in a
    misleading way (issue elspeth-72d1dccd44).
    """
    if isinstance(value, Mapping) and len(value) == 1 and "secret_ref" in value:
        return isinstance(value["secret_ref"], str)
    if isinstance(value, str):
        return secret_env_ref_name(value, env_ref_names) is not None
    return False


def _collect_credential_field_violations(
    options: Any,
    env_ref_names: set[str],
) -> list[str]:
    """Walk an options tree and return the field names whose value is a
    fabricated literal sitting in a credential-bearing field.

    Returns the list of offending top-level field names (as discovered by
    ``is_secret_field``) — caller decides how to surface them. The returned
    list is intentionally name-only (no value): credential values must never
    be reflected back into the validation response.

    Skips:
    - ``{"secret_ref": ...}`` markers (wired)
    - exact ``${NAME}`` env-markers where NAME is in ``env_ref_names`` (wired)
    - missing keys, ``None`` values, and empty strings (treated as
      "not provided" — the runtime config validator decides if the field is
      required)
    - non-string values in credential fields (Pydantic at settings_load
      will reject these with the proper schema-typed error)

    Recurses through nested mappings and sequences so that
    ``{"auth": {"api_key": "..."}}`` is detected as well as a top-level shape.
    """
    violations: list[str] = []
    if isinstance(options, Mapping):
        # A ``{"secret_ref": ...}`` marker is itself wired — do not descend
        # into it (otherwise the literal NAME inside would be evaluated).
        if len(options) == 1 and "secret_ref" in options:
            return violations
        for key, value in options.items():
            if isinstance(key, str) and is_secret_field(key):
                if _value_is_wired_secret(value, env_ref_names):
                    continue
                if value is None or value == "":
                    continue
                if isinstance(value, str):
                    violations.append(key)
                    continue
                # Non-string in a credential field: defer to Pydantic at
                # settings_load — its error is more precise (shape mismatch
                # vs. fabrication).
                continue
            violations.extend(_collect_credential_field_violations(value, env_ref_names))
    elif isinstance(options, (list, tuple)):
        for item in options:
            violations.extend(_collect_credential_field_violations(item, env_ref_names))
    return violations


def validate_pipeline(
    state: CompositionState,
    settings: ValidationSettings,
    yaml_generator: YamlGenerator,
    *,
    secret_service: WebSecretResolver | None = None,
    user_id: str | None = None,
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

    Only catches: PydanticValidationError, FileNotFoundError, ValueError,
    GraphValidationError. All other exceptions propagate (W18).

    Args:
        state: CompositionState from the session.
        settings: ValidationSettings — exposes data_dir for path resolution and allowlist check.
        yaml_generator: YamlGenerator module/object with generate_yaml() method.
        secret_service: Optional secret resolver for validating secret refs.
        user_id: User ID for scoped secret resolution (required if secret_service is set).
    """
    checks: list[ValidationCheck] = []
    errors: list[ValidationError] = []

    # Step 1: Source + sink path allowlist check (C3/S2 defense-in-depth)
    # Any `path` or `file` key in source/sink options must resolve under
    # an allowed directory. Uses the shared helpers from AD-4.
    from elspeth.web.paths import allowed_sink_directories, allowed_source_directories, resolve_data_path

    allowed_source_dirs = allowed_source_directories(str(settings.data_dir))
    allowed_sink_dirs = allowed_sink_directories(str(settings.data_dir))
    # state is a CompositionState (typed domain object). state.source is a
    # SourceSpec with typed .options attribute (Mapping[str, Any]).
    source_options = dict(state.source.options) if state.source is not None else {}
    path_checked = False
    for key in ("path", "file"):
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
                            detail=f"Source {key} '{value}' is outside allowed source directories",
                        ),
                        *_skipped_checks(_CHECK_PATH_ALLOWLIST),
                    ],
                    errors=[
                        ValidationError(
                            component_id="source",
                            component_type="source",
                            message=f"Path traversal blocked: {key}='{value}' resolves outside allowed directories",
                            suggestion="Use a file within the blobs directory.",
                        ),
                    ],
                )

    # Sink path allowlist — prevents arbitrary file writes via sink options.
    for output in state.outputs or ():
        for key in ("path", "file"):
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
                            ),
                            *_skipped_checks(_CHECK_PATH_ALLOWLIST),
                        ],
                        errors=[
                            ValidationError(
                                component_id=output.name,
                                component_type="sink",
                                message=f"Path traversal blocked: sink '{output.name}' {key}='{value}' resolves outside allowed directories",
                                suggestion="Use a path within the outputs or blobs directory.",
                            ),
                        ],
                    )

    # B11 fix: Always record the path_allowlist check
    if path_checked:
        checks.append(
            ValidationCheck(
                name=_CHECK_PATH_ALLOWLIST,
                passed=True,
                detail="All paths within allowed directories",
            )
        )
    else:
        checks.append(
            ValidationCheck(
                name=_CHECK_PATH_ALLOWLIST,
                passed=True,
                detail="No path option — check skipped",
            )
        )

    # Step 1b: Secret ref validation — check that
    #   (a) every wired ``{secret_ref: ...}`` / inventory ``${NAME}`` resolves
    #       (existing missing-refs gate), AND
    #   (b) every credential-bearing field (per ``is_secret_field``) contains
    #       a wired secret rather than a literal placeholder string
    #       (issue elspeth-72d1dccd44 — S1A "WILL_BE_WIRED_FROM_..." defect).
    #
    # The remediation note (notes/composer-remediation-program-2026-05-01.md)
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
    if secret_service is not None and user_id is not None:
        env_ref_names = {item.name for item in secret_service.list_refs(user_id)}
        # Walk source options, node configs, and output options for secret refs
        if state.source is not None:
            all_refs.extend(_collect_secret_refs(state.source.options, env_ref_names))
            fabricated = _collect_credential_field_violations(state.source.options, env_ref_names)
            if fabricated:
                fabricated_components.append(("source", "source", fabricated))
        for node in state.nodes or ():
            all_refs.extend(_collect_secret_refs(node.options, env_ref_names))
            fabricated = _collect_credential_field_violations(node.options, env_ref_names)
            if fabricated:
                fabricated_components.append((node.id, "transform", fabricated))
        for output in state.outputs or ():
            all_refs.extend(_collect_secret_refs(output.options, env_ref_names))
            fabricated = _collect_credential_field_violations(output.options, env_ref_names)
            if fabricated:
                fabricated_components.append((output.name, "sink", fabricated))

        missing_refs = [ref for ref in all_refs if not secret_service.has_ref(user_id, ref)]
        if missing_refs or fabricated_components:
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
                        )
                    )
            checks.append(
                ValidationCheck(
                    name=_CHECK_SECRET_REFS,
                    passed=False,
                    detail="; ".join(detail_parts),
                )
            )
            checks.extend(_skipped_checks(_CHECK_SECRET_REFS))
            return ValidationResult(is_valid=False, checks=checks, errors=errors)
        checks.append(
            ValidationCheck(
                name=_CHECK_SECRET_REFS,
                passed=True,
                detail=f"All {len(all_refs)} secret reference(s) resolved" if all_refs else "No secret references found",
            )
        )
    else:
        checks.append(
            ValidationCheck(
                name=_CHECK_SECRET_REFS,
                passed=True,
                detail="No secret service — check skipped",
            )
        )

    semantic_errors, semantic_contracts = validate_semantic_contracts(state)
    if semantic_errors:
        checks.append(
            ValidationCheck(
                name=_CHECK_SEMANTIC_CONTRACTS,
                passed=False,
                detail="Semantic contract check failed",
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
                )
            )
        checks.extend(_skipped_checks(_CHECK_SEMANTIC_CONTRACTS))
        return ValidationResult(
            is_valid=False,
            checks=checks,
            errors=errors,
            semantic_contracts=serialize_semantic_contracts(semantic_contracts),
        )

    checks.append(
        ValidationCheck(
            name=_CHECK_SEMANTIC_CONTRACTS,
            passed=True,
            detail=(
                f"All {len(semantic_contracts)} semantic contract(s) satisfied" if semantic_contracts else "No semantic contracts to check"
            ),
        )
    )

    batch_option_errors: list[tuple[str, str]] = []
    for node in state.nodes:
        batch_required_error = _batch_aware_required_input_fields_error(node.id, node.plugin, node.options)
        if batch_required_error is not None:
            batch_option_errors.append((node.id, batch_required_error))
    if batch_option_errors:
        checks.append(
            ValidationCheck(
                name=_CHECK_BATCH_TRANSFORM_OPTIONS,
                passed=False,
                detail="Batch-aware transform option check failed",
            )
        )
        for node_id, message in batch_option_errors:
            errors.append(
                ValidationError(
                    component_id=node_id,
                    component_type="transform",
                    message=message,
                    suggestion="Remove required_input_fields from batch-aware transform options; use schema.required_fields for batch input validation.",
                )
            )
        checks.extend(_skipped_checks(_CHECK_BATCH_TRANSFORM_OPTIONS))
        return ValidationResult(
            is_valid=False,
            checks=checks,
            errors=errors,
            semantic_contracts=serialize_semantic_contracts(semantic_contracts),
        )
    checks.append(
        ValidationCheck(
            name=_CHECK_BATCH_TRANSFORM_OPTIONS,
            passed=True,
            detail="Batch-aware transform options are compatible with ADR-013",
        )
    )

    # Step 2: Generate YAML
    pipeline_yaml = yaml_generator.generate_yaml(state)
    pipeline_yaml = resolve_runtime_yaml_paths(pipeline_yaml, str(settings.data_dir))

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

        elspeth_settings = load_settings_from_yaml_string(settings_yaml)
        checks.append(
            ValidationCheck(
                name=_CHECK_SETTINGS,
                passed=True,
                detail="Settings loaded successfully",
            )
        )
    except (PydanticValidationError, ValueError, TypeError) as exc:
        checks.append(
            ValidationCheck(
                name=_CHECK_SETTINGS,
                passed=False,
                detail=str(exc),
            )
        )
        errors.append(
            ValidationError(
                component_id=None,
                component_type=None,
                message=str(exc),
                suggestion=None,
            )
        )
        checks.extend(_skipped_checks(_CHECK_SETTINGS))
        return ValidationResult(
            is_valid=False,
            checks=checks,
            errors=errors,
            semantic_contracts=serialize_semantic_contracts(semantic_contracts),
        )

    # Step 4: Plugin instantiation
    try:
        bundle = instantiate_runtime_plugins(elspeth_settings, preflight_mode=True)
        checks.append(
            ValidationCheck(
                name=_CHECK_PLUGINS,
                passed=True,
                detail="All plugins instantiated",
            )
        )
    except (PluginNotFoundError, PluginConfigError) as exc:
        comp_type = _infer_component_type_from_plugin_error(exc)
        plugin_name = exc.plugin_name if isinstance(exc, PluginConfigError) else None
        # Prefer cause (validation detail) over str(exc) which includes the
        # internal class name prefix (e.g. "Invalid configuration for CSVSourceConfig: ...").
        if isinstance(exc, PluginConfigError) and exc.cause is not None and plugin_name is not None:
            detail = f"Invalid configuration for {comp_type} '{plugin_name}': {exc.cause}"
        else:
            detail = str(exc)
        checks.append(
            ValidationCheck(
                name=_CHECK_PLUGINS,
                passed=False,
                detail=detail,
            )
        )
        errors.append(
            ValidationError(
                component_id=plugin_name,
                component_type=comp_type,
                message=detail,
                suggestion=None,
            )
        )
        checks.extend(_skipped_checks(_CHECK_PLUGINS))
        return ValidationResult(
            is_valid=False,
            checks=checks,
            errors=errors,
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
            )
        )
    except GraphValidationError as exc:
        checks.append(
            ValidationCheck(
                name=_CHECK_GRAPH,
                passed=False,
                detail=str(exc),
            )
        )
        errors.append(
            ValidationError(
                component_id=exc.component_id,
                component_type=exc.component_type,
                message=str(exc),
                suggestion=None,
            )
        )
        checks.extend(_skipped_checks(_CHECK_GRAPH))
        return ValidationResult(
            is_valid=False,
            checks=checks,
            errors=errors,
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
            source=bundle.source,
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
            )
        )
    except RouteValidationError as exc:
        checks.append(
            ValidationCheck(
                name=_CHECK_ROUTE_TARGETS,
                passed=False,
                detail=str(exc),
            )
        )
        errors.append(
            ValidationError(
                component_id=None,
                component_type=None,
                message=str(exc),
                suggestion=("Use 'discard' to drop rows without routing, or wire the destination to an existing sink."),
            )
        )
        checks.extend(_skipped_checks(_CHECK_ROUTE_TARGETS))
        return ValidationResult(
            is_valid=False,
            checks=checks,
            errors=errors,
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
            )
        )
    except GraphValidationError as exc:
        checks.append(
            ValidationCheck(
                name=_CHECK_SCHEMA,
                passed=False,
                detail=str(exc),
            )
        )
        errors.append(
            ValidationError(
                component_id=exc.component_id,
                component_type=exc.component_type,
                message=str(exc),
                suggestion=None,
            )
        )
        return ValidationResult(
            is_valid=False,
            checks=checks,
            errors=errors,
            semantic_contracts=serialize_semantic_contracts(semantic_contracts),
        )

    return ValidationResult(
        is_valid=True,
        checks=checks,
        errors=errors,
        semantic_contracts=serialize_semantic_contracts(semantic_contracts),
    )
