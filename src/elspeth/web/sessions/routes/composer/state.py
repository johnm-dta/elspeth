from __future__ import annotations

import json
from dataclasses import replace
from typing import TypedDict

from pydantic import BaseModel, ConfigDict, Field

from elspeth.contracts.trust_boundary import trust_boundary
from elspeth.core.secrets import collect_credential_field_violations
from elspeth.web.blobs.protocol import BlobNotFoundError
from elspeth.web.composer.state import CompositionState, SourceSpec
from elspeth.web.composer.yaml_importer import (
    MAX_RUNTIME_YAML_IMPORT_CHARS,
    RuntimeYamlImportError,
    composition_state_from_runtime_yaml,
)
from elspeth.web.paths import SOURCE_LOCAL_PATH_OPTION_KEYS, allowed_source_directories, resolve_data_path
from elspeth.web.secrets.ref_policy import allowed_secret_ref_fields

from .._helpers import (
    UTC,
    UUID,
    Any,
    APIRouter,
    ComposerPreferencesResponse,
    ComposerProgressSnapshot,
    CompositionStateResponse,
    Depends,
    GraphValidationError,
    HTTPException,
    Mapping,
    PluginConfigError,
    PluginNotFoundError,
    Query,
    Request,
    RevertStateRequest,
    SessionServiceProtocol,
    SessionsTelemetry,
    UpdateComposerPreferencesRequest,
    UserIdentity,
    _composer_preferences_response,
    _get_composer_progress_registry,
    _get_session_compose_lock_registry,
    _record_composer_runtime_preflight_telemetry,
    _runtime_preflight_for_state,
    _state_data_from_composer_state,
    _state_from_record,
    _state_response,
    _verify_session_ownership,
    composer_completion_events_table,
    datetime,
    generate_public_yaml,
    get_current_user,
    insert,
    record_session_completed,
    record_session_switched,
    uuid4,
)

router = APIRouter()


class StateYamlResponse(TypedDict, total=False):
    yaml: str
    source_blob_ids: dict[str, str]


class ImportStateYamlRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    yaml: str = Field(min_length=1, max_length=MAX_RUNTIME_YAML_IMPORT_CHARS)
    source_blob_ids: dict[str, str] | None = None


class SeedCompositionStateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    state: dict[str, Any]


@trust_boundary(
    tier=3,
    source="web-authored source options from a pasted/seeded composition state",
    source_param="options",
    suppresses=("R1", "R5"),
    invariant=(
        "missing or non-string path values are skipped (False); a hostile string value "
        "raises ValueError from path resolution rather than being coerced"
    ),
    test_ref="tests/unit/web/sessions/routes/composer/test_state_boundaries.py::test_source_options_reference_blob_storage_raises_on_nul_byte_path",
    test_fingerprint="18ac99a70b815badb2f8ef53eb3061e7cb3d8b13299965bc591e559d2d961bf9",
)
def _source_options_reference_blob_storage(options: Mapping[str, Any], *, data_dir: str) -> bool:
    allowed_dirs = allowed_source_directories(data_dir)
    for key in SOURCE_LOCAL_PATH_OPTION_KEYS:
        value = options.get(key)
        if not isinstance(value, str):
            continue
        resolved = resolve_data_path(value, data_dir)
        if any(resolved.is_relative_to(directory) for directory in allowed_dirs):
            return True
    return False


@trust_boundary(
    tier=3,
    source="pasted/seeded CompositionState carrying web-authored source options",
    source_param="state",
    suppresses=("R1",),
    invariant=(
        "raises HTTPException 400 for any source whose options reference session blob storage "
        "without a blob_ref binding; blob_ref-bound and non-blob sources pass"
    ),
    test_ref="tests/unit/web/sessions/routes/composer/test_state_boundaries.py::test_reject_unbound_blob_storage_sources_raises_400_on_unbound_blob_path",
    test_fingerprint="bdb265d5ce47ac3963b67735a2d485acdf4bc93c9d091f9b6c2da9c0571ca9a9",
)
def _reject_unbound_blob_storage_sources(state: CompositionState, *, data_dir: str) -> None:
    for source_name, source in state.sources.items():
        if source.options.get("blob_ref") is not None:
            continue
        if _source_options_reference_blob_storage(source.options, data_dir=data_dir):
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Source '{source_name}' points at session blob storage but has no source_blob_ids entry. "
                    "Upload the blob into this session and include source_blob_ids for replay imports."
                ),
            )


def _reject_fabricated_secret_literals(
    state: CompositionState,
    *,
    secret_service: Any | None,
    user_id: str,
) -> None:
    """Reject literal credential values in pasted YAML before persistence.

    The composer's tool-call surface (``set_source``, ``patch_node_options``,
    ``set_output``) already rejects a fabricated ("typed the value directly
    instead of wiring a secret_ref") credential *before* persistence, via
    ``_credential_wiring_contract_failure`` in ``composer/tools/_common.py``:
    the mutation handler returns the unchanged state, so the literal is never
    written into CompositionState. Pasted YAML bypasses those handlers
    entirely (it reconstructs the state directly from the parsed document),
    so it has no equivalent gate -- without this check a literal credential
    would sail straight through ``_state_data_from_composer_state``'s
    ``persist_invalid`` policy: written to the DB as plaintext and echoed back
    verbatim in this response's ``sources``/``nodes``/``outputs`` fields,
    regardless of whether the state is later flagged ``is_valid=False``.

    Checking (and rejecting outright, before any persistence) here is
    stricter than the tool-call path affords today -- deliberately so, since
    this is a new, paste-facing entry point. It reuses the same field-name
    predicate and the same audit-hygiene discipline (name the field, never
    the value) as the runtime-preflight ``fabricated_secret`` check in
    ``elspeth.web.execution.validation``, so a legitimately wired
    ``{secret_ref: NAME}`` marker or declared ``${NAME}`` env-inventory
    marker is never mistaken for a fabricated literal.

    Beyond the heuristic name/suffix predicate, each component also feeds its
    plugin-specific credential fields (via ``allowed_secret_ref_fields`` --
    the database sink's whole-DSN ``url`` being the canonical case) so a
    pasted DSN with an embedded password is rejected here exactly as the
    ``set_source``/``set_output`` tool gate rejects it, rather than slipping
    past the suffix predicate and persisting plaintext.
    """
    env_ref_names: frozenset[str] = frozenset()
    if secret_service is not None:
        env_ref_names = frozenset(item.name for item in secret_service.list_refs(user_id))

    violations: dict[str, list[str]] = {}
    for source_name, source in state.sources.items():
        fields = collect_credential_field_violations(
            source.options,
            env_ref_names,
            additional_credential_fields=allowed_secret_ref_fields("source", source.plugin),
        )
        if fields:
            violations[f"source:{source_name}"] = fields
    for node in state.nodes:
        node_plugin_fields = allowed_secret_ref_fields("transform", node.plugin) if node.plugin is not None else frozenset()
        fields = collect_credential_field_violations(
            node.options,
            env_ref_names,
            additional_credential_fields=node_plugin_fields,
        )
        if fields:
            violations[f"node:{node.id}"] = fields
    for output in state.outputs:
        fields = collect_credential_field_violations(
            output.options,
            env_ref_names,
            additional_credential_fields=allowed_secret_ref_fields("sink", output.plugin),
        )
        if fields:
            violations[f"sink:{output.name}"] = fields

    if violations:
        components = "; ".join(f"{component}: {', '.join(fields)}" for component, fields in violations.items())
        raise HTTPException(
            status_code=400,
            detail=(
                f"Pasted YAML contains a literal value in credential-bearing field(s) -- {components}. "
                "Wire each credential through the Secrets panel (produces a {secret_ref: NAME} marker) "
                "instead of pasting the value directly."
            ),
        )


async def _state_with_imported_source_blobs(
    state: CompositionState,
    *,
    source_blob_ids: Mapping[str, str] | None,
    request: Request,
    session_id: UUID,
) -> CompositionState:
    if not source_blob_ids:
        return state

    sources = dict(state.sources)
    requested_blobs: list[tuple[str, UUID]] = []
    for source_name, blob_id_raw in source_blob_ids.items():
        if source_name not in sources:
            raise HTTPException(status_code=400, detail=f"source_blob_ids references unknown source '{source_name}'")
        try:
            blob_id = UUID(blob_id_raw)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=f"source_blob_ids.{source_name} must be a UUID") from exc
        requested_blobs.append((source_name, blob_id))

    blob_service = request.app.state.blob_service
    if blob_service is None:
        raise HTTPException(status_code=409, detail="Blob service unavailable for YAML import")

    for source_name, blob_id in requested_blobs:
        try:
            blob = await blob_service.get_blob(blob_id)
        except BlobNotFoundError:
            raise HTTPException(status_code=404, detail="Blob not found") from None
        if blob.session_id != session_id:
            raise HTTPException(status_code=404, detail="Blob not found")

        source = sources[source_name]
        options = dict(source.options)
        for key in SOURCE_LOCAL_PATH_OPTION_KEYS:
            if key in options:
                del options[key]
        options["path"] = blob.storage_path
        options["blob_ref"] = str(blob.id)
        sources[source_name] = SourceSpec(
            plugin=source.plugin,
            on_success=source.on_success,
            options=options,
            on_validation_failure=source.on_validation_failure,
        )

    return CompositionState(
        sources=sources,
        nodes=state.nodes,
        edges=state.edges,
        outputs=state.outputs,
        metadata=state.metadata,
        version=state.version,
        guided_session=state.guided_session,
    )


@router.get(
    "/{session_id}/composer-progress",
    response_model=ComposerProgressSnapshot,
)
async def get_composer_progress(
    session_id: UUID,
    request: Request,
    user: UserIdentity = Depends(get_current_user),  # noqa: B008
) -> ComposerProgressSnapshot:
    """Return the latest provider-safe composer progress for a session."""
    session = await _verify_session_ownership(session_id, user, request)
    registry = _get_composer_progress_registry(request)
    return await registry.get_latest(str(session.id))


@router.get(
    "/{session_id}/composer/preferences",
    response_model=ComposerPreferencesResponse,
)
async def get_composer_preferences(
    session_id: UUID,
    request: Request,
    user: UserIdentity = Depends(get_current_user),  # noqa: B008
) -> ComposerPreferencesResponse:
    session = await _verify_session_ownership(session_id, user, request)
    service: SessionServiceProtocol = request.app.state.session_service
    prefs = await service.get_composer_preferences(session.id)
    return _composer_preferences_response(prefs)


@router.patch(
    "/{session_id}/composer/preferences",
    response_model=ComposerPreferencesResponse,
)
async def update_composer_preferences(
    session_id: UUID,
    body: UpdateComposerPreferencesRequest,
    request: Request,
    user: UserIdentity = Depends(get_current_user),  # noqa: B008
) -> ComposerPreferencesResponse:
    session = await _verify_session_ownership(session_id, user, request)
    service: SessionServiceProtocol = request.app.state.session_service
    # B2 (load-bearing): the service returns ``(prior, current)`` so
    # Phase 8b's per-session ``composer.session.switched_total``
    # counter can read ``from_mode=transition.prior.trust_mode``
    # without a route-handler read-before-write (which would open a
    # TOCTOU window — see plan §"Option not taken — read-before-write
    # from the route handler"). The PATCH response shape is unchanged;
    # we only project ``current`` into the response model.
    transition = await service.update_composer_preferences(
        session.id,
        trust_mode=body.trust_mode,
        density_default=body.density_default,
        actor=f"user:{user.user_id}",
    )

    # Phase 8 Task 2 Step 3 — per-session ``trust_mode`` switch emit.
    #
    # Guarded on actual change (transition-rate semantic, distinct
    # from the account-level set-rate at preferences/routes.py).
    # The service's ``trust_mode.changed`` audit row at
    # ``sessions/service.py:1605-1619`` fires unconditionally on
    # every PATCH including no-ops; emitting the counter
    # unconditionally would over-count by the no-op rate. Guarding
    # on ``prior != current`` also gives the Q4 contract: a
    # combined PATCH that changes both ``trust_mode`` AND
    # ``density_default`` fires the counter exactly once,
    # attributed to the trust_mode change only.
    #
    # B1 (audit-primacy superset rule): the emit runs AFTER the
    # audit row commits (the service ``_run_sync`` returned),
    # which carries ``prior_trust_mode`` in its payload (B1
    # extension at sessions/service.py:1614). Telemetry attributes
    # are a strict subset of audit-recorded reality.
    #
    # Vocabulary (B1-r2): both attributes come from the per-session
    # ``trust_mode`` CHECK-constraint vocabulary
    # (``explicit_approve`` / ``auto_commit``), NOT the account-
    # level ``default_composer_mode`` vocabulary.
    if transition.prior.trust_mode != transition.current.trust_mode:
        telemetry: SessionsTelemetry = request.app.state.sessions_telemetry
        record_session_switched(
            telemetry,
            from_mode=transition.prior.trust_mode,
            to_mode=transition.current.trust_mode,
        )

    return _composer_preferences_response(transition.current)


@router.get("/{session_id}/state")
async def get_current_state(
    session_id: UUID,
    request: Request,
    user: UserIdentity = Depends(get_current_user),  # noqa: B008
) -> CompositionStateResponse | None:
    """Get the current (highest-version) composition state."""
    session = await _verify_session_ownership(session_id, user, request)
    service = request.app.state.session_service
    state = await service.get_current_state(session.id)
    if state is None:
        return None
    return _state_response(state)


@router.get(
    "/{session_id}/state/versions",
    response_model=list[CompositionStateResponse],
)
async def get_state_versions(
    session_id: UUID,
    request: Request,
    user: UserIdentity = Depends(get_current_user),  # noqa: B008
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
) -> list[CompositionStateResponse]:
    """Get composition state versions for a session."""
    session = await _verify_session_ownership(session_id, user, request)
    service = request.app.state.session_service
    versions = await service.get_state_versions(session.id, limit=limit, offset=offset)
    return [_state_response(v) for v in versions]


@router.post(
    "/{session_id}/state/revert",
    response_model=CompositionStateResponse,
)
async def revert_state(
    session_id: UUID,
    body: RevertStateRequest,
    request: Request,
    user: UserIdentity = Depends(get_current_user),  # noqa: B008
) -> CompositionStateResponse:
    """Revert the pipeline to a prior composition state version (R1).

    Creates a new version that is a copy of the specified prior state.
    Injects a system message recording the revert.
    """
    session = await _verify_session_ownership(session_id, user, request)
    service = request.app.state.session_service

    try:
        new_state = await service.set_active_state(
            session.id,
            body.state_id,
        )
    except ValueError:
        raise HTTPException(
            status_code=404,
            detail="State not found",
        ) from None

    # Look up the original version number for the system message
    original_state = await service.get_state(body.state_id)
    await service.add_message(
        session.id,
        role="system",
        content=f"Pipeline reverted to version {original_state.version}.",
        writer_principal="route_system_message",
    )

    return _state_response(new_state)


@router.post(
    "/{session_id}/state/yaml",
    response_model=CompositionStateResponse,
)
async def import_state_yaml(
    session_id: UUID,
    body: ImportStateYamlRequest,
    request: Request,
    user: UserIdentity = Depends(get_current_user),  # noqa: B008
) -> CompositionStateResponse:
    """Seed a session's composition state from exported runtime YAML."""
    session = await _verify_session_ownership(session_id, user, request)
    compose_lock = await _get_session_compose_lock_registry(request).get_lock(str(session.id))
    async with compose_lock:
        try:
            imported_state = composition_state_from_runtime_yaml(body.yaml)
        except RuntimeYamlImportError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        imported_state = await _state_with_imported_source_blobs(
            imported_state,
            source_blob_ids=body.source_blob_ids,
            request=request,
            session_id=session.id,
        )
        _reject_unbound_blob_storage_sources(
            imported_state,
            data_dir=str(request.app.state.settings.data_dir),
        )
        _reject_fabricated_secret_literals(
            imported_state,
            secret_service=request.app.state.scoped_secret_resolver,
            user_id=str(user.user_id),
        )

        service: SessionServiceProtocol = request.app.state.session_service
        state_data, _validation = await _state_data_from_composer_state(
            imported_state,
            settings=request.app.state.settings,
            secret_service=request.app.state.scoped_secret_resolver,
            user_id=str(user.user_id),
            session_id=session.id,
            runtime_preflight=None,
            preflight_exception_policy="persist_invalid",
            initial_version=imported_state.version,
            telemetry_source="compose",
        )
        state_record = await service.save_composition_state(
            session.id,
            state_data,
            provenance="session_seed",
        )
        return _state_response(state_record)


@router.post(
    "/{session_id}/state/e2e-seed",
    response_model=CompositionStateResponse,
    include_in_schema=False,
)
async def seed_state_for_e2e(
    session_id: UUID,
    request: Request,
    user: UserIdentity = Depends(get_current_user),  # noqa: B008
) -> CompositionStateResponse:
    """Test-only state seed endpoint for Playwright-managed E2E runs."""
    if not request.app.state.settings.e2e_state_seed_enabled:
        raise HTTPException(status_code=404, detail="Not found")

    session = await _verify_session_ownership(session_id, user, request)

    try:
        body = SeedCompositionStateRequest.model_validate(await request.json())
    except (json.JSONDecodeError, TypeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail="Invalid seed request JSON") from exc

    compose_lock = await _get_session_compose_lock_registry(request).get_lock(str(session.id))
    async with compose_lock:
        try:
            seeded_state = CompositionState.from_dict(body.state)
        except (KeyError, TypeError, ValueError) as exc:
            raise HTTPException(status_code=400, detail="Invalid composition state JSON") from exc

        _reject_unbound_blob_storage_sources(
            seeded_state,
            data_dir=str(request.app.state.settings.data_dir),
        )
        _reject_fabricated_secret_literals(
            seeded_state,
            secret_service=request.app.state.scoped_secret_resolver,
            user_id=str(user.user_id),
        )

        service: SessionServiceProtocol = request.app.state.session_service
        state_data, _validation = await _state_data_from_composer_state(
            seeded_state,
            settings=request.app.state.settings,
            secret_service=request.app.state.scoped_secret_resolver,
            user_id=str(user.user_id),
            session_id=session.id,
            runtime_preflight=None,
            preflight_exception_policy="persist_invalid",
            initial_version=seeded_state.version,
            telemetry_source="state_seed",
        )
        state_record = await service.save_composition_state(
            session.id,
            state_data,
            provenance="session_seed",
        )
        return _state_response(state_record)


def _reattach_guided_blob_refs(state: CompositionState) -> CompositionState:
    """Reconstitute the ``blob_ref`` stripped from a guided blob-backed source's
    committed options, using the GuidedSession snapshot's ``step_1_result`` as
    the authoritative signal (elspeth-b5ee205720).

    The manual set_source commit strips ``blob_ref`` from guided sources (it
    cannot prove ``path == storage_path``); it survives only in the persisted
    snapshot. Both export egress channels — the public-YAML storage-path omission
    (``_strip_web_metadata(..., omit_blob_bound_source_paths=True)``) and the
    ``source_blob_ids`` sidecar — key off ``source.options["blob_ref"]``, so
    without this a guided blob source leaks its absolute storage path AND emits no
    sidecar (breaking the export→re-import round-trip). Reattaching here lets the
    existing ``blob_ref``-keyed export machinery treat guided sources exactly like
    freeform blob-bound ones. Mirrors the snapshot cross-reference in
    ``redact_guided_snapshot_storage_paths``; never mutates ``state``.

    Sources are matched to the snapshot by storage-path-string equality (the same
    approach as the reference redactor). A second, distinct source carrying the
    identical absolute path string would also be treated as blob-backed — a
    narrow, pre-existing edge shared with that redactor; equal paths do mean "the
    same underlying file", and guided sessions commit a single source today.
    """
    guided = state.guided_session
    if guided is None or guided.step_1_result is None:
        return state
    snapshot_options = guided.step_1_result.options
    blob_ref = snapshot_options.get("blob_ref")
    # No blob_ref on the snapshot ⇒ operator-typed source; leave every path alone.
    if not blob_ref:
        return state
    blob_backed_paths: set[str] = set()
    for key in SOURCE_LOCAL_PATH_OPTION_KEYS:
        value = snapshot_options.get(key)
        if isinstance(value, str):
            blob_backed_paths.add(value)
    if not blob_backed_paths:
        return state

    reattached: dict[str, SourceSpec] = {}
    changed = False
    for source_name, source in state.sources.items():
        options = source.options
        if "blob_ref" in options or not any(options.get(key) in blob_backed_paths for key in SOURCE_LOCAL_PATH_OPTION_KEYS):
            reattached[source_name] = source
            continue
        merged = dict(options)
        merged["blob_ref"] = str(blob_ref)
        reattached[source_name] = replace(source, options=merged)
        changed = True

    return replace(state, sources=reattached) if changed else state


@router.get("/{session_id}/state/yaml")
async def get_state_yaml(
    session_id: UUID,
    request: Request,
    user: UserIdentity = Depends(get_current_user),  # noqa: B008
) -> StateYamlResponse:
    """Get YAML representation of the current composition state (M1).

    Runs runtime preflight on the exact CompositionState reconstructed
    from the persisted record, then generates deterministic public YAML via
    generate_public_yaml() against that same snapshot. YAML export preflight
    deliberately does not receive the scoped secret resolver: export
    serializes secret_ref markers, and a rejected preflight response must
    not expose resolved secret values through plugin validation prose.
    The two operations see the same Python object — there is no re-fetch
    between preflight and serialization, so a state that passes the gate
    is byte-identical to the state that gets serialized.
    """
    session = await _verify_session_ownership(session_id, user, request)
    service: SessionServiceProtocol = request.app.state.session_service
    state_record = await service.get_current_state(session.id)
    if state_record is None:
        raise HTTPException(status_code=404, detail="No composition state exists")
    state = _state_from_record(state_record)
    try:
        runtime_validation = await _runtime_preflight_for_state(
            state,
            settings=request.app.state.settings,
            secret_service=None,
            user_id=None,
            session_id=session.id,
        )
    except (
        TimeoutError,
        OSError,
        PluginConfigError,
        PluginNotFoundError,
        GraphValidationError,
    ) as exc:
        # Narrowed per CLAUDE.md offensive-programming policy. This
        # tuple covers the user-fixable preflight failure modes:
        #
        # * TimeoutError — asyncio.wait_for exceeded
        #   composer_runtime_preflight_timeout_seconds. Operator
        #   action: increase timeout or fix the slow plugin.
        # * OSError — filesystem error during plugin instantiation
        #   (file not found, permission denied, broken pipe, etc.).
        #   Operator action: fix the file/permissions.
        # * PluginConfigError / PluginNotFoundError — the user's
        #   pipeline references a misconfigured or missing plugin.
        #   Operator action: fix the pipeline config.
        # * GraphValidationError — the pipeline graph is structurally
        #   invalid (validate_pipeline normally absorbs this, but
        #   it's listed here for defense-in-depth in case a future
        #   refactor lets it escape).
        #
        # Programmer-bug classes (AttributeError, TypeError,
        # KeyError, RuntimeError, ImportError, etc.) are deliberately
        # NOT caught — they propagate to FastAPI's default 500
        # handler so operators see real tracebacks rather than the
        # misleading "fix your pipeline" 409 message. The
        # exception-counter is reserved for the user-fixable bucket
        # so dashboards measure real preflight failure rate, not
        # bugs we introduced ourselves.
        _record_composer_runtime_preflight_telemetry(
            "exception",
            source="yaml_export",
            exception_class=type(exc).__name__,
        )
        raise HTTPException(
            status_code=409,
            detail="Runtime preflight could not complete; YAML export aborted.",
        ) from exc
    _record_composer_runtime_preflight_telemetry(
        "passed" if runtime_validation.is_valid else "failed",
        source="yaml_export",
    )
    if not runtime_validation.is_valid:
        # Deliberately a generic message: the YAML-export 409 must not echo
        # preflight error prose (commit "fix: prevent YAML export secret
        # leaks"). With secret_service=None the fabricated-secret check is
        # skipped, so a literally-typed credential is not redacted and could
        # otherwise surface through plugin validation prose. See
        # test_get_state_yaml_does_not_echo_preflight_error_messages.
        detail = "Current composition state failed runtime preflight. Fix validation errors before exporting YAML."
        raise HTTPException(status_code=409, detail=detail)
    # elspeth-b5ee205720: reconstitute blob_ref for guided blob-backed sources
    # (stripped from committed options; retained only in the GuidedSession
    # snapshot) so BOTH export egress channels below — the public-YAML storage-path
    # omission and the source_blob_ids sidecar — treat them as blob-bound. Kept
    # AFTER preflight: blob_ref is extra=forbid for plugin configs and must not
    # reach plugin instantiation. Preflight ran on the raw `state`; export uses
    # the reattached copy.
    export_state = _reattach_guided_blob_refs(state)
    yaml_str = generate_public_yaml(export_state)

    # Phase 6A B3 — sessions-DB audit event for YAML export.
    #
    # Two Tier-1 audit events ship in Phase 6 (mark_ready_for_review and
    # export_yaml). This is the export_yaml site. Sync, crash-on-failure
    # per CLAUDE.md audit primacy — if this write fails the request
    # fails, no YAML is returned, no carve-out is permitted. The write
    # MUST land before the response is returned: the audit row is the
    # legal record that the YAML was exported on the user's behalf.
    #
    # The state record was just read via ``service.get_current_state``
    # above; ``state_record.id`` is the composition_state_id this
    # export is bound to.
    with request.app.state.session_engine.begin() as conn:
        conn.execute(
            insert(composer_completion_events_table).values(
                id=str(uuid4()),
                session_id=str(session_id),
                composition_state_id=str(state_record.id),
                event_type="export_yaml",
                actor=str(user.user_id),
                created_at=datetime.now(UTC),
                payload_digest=None,
                expires_at=None,
            )
        )

    # Phase 8 Sub-task 7c (telemetry-backfill: phase-6).
    # composer.session.completed_total — fires AFTER the audit
    # engine.begin() block has exited cleanly. If the audit INSERT
    # raises, the with-block exits via exception, FastAPI converts
    # it to a 5xx, and control never reaches this line — the counter
    # stays at zero and the superset invariant (counter aggregates
    # over committed audit rows) is structurally enforced.
    record_session_completed(
        request.app.state.sessions_telemetry,
        completion_verb="export_yaml",
    )

    response: StateYamlResponse = {"yaml": yaml_str}
    source_blob_ids = {
        source_name: str(source.options["blob_ref"]) for source_name, source in export_state.sources.items() if "blob_ref" in source.options
    }
    if source_blob_ids:
        response["source_blob_ids"] = source_blob_ids
    return response
