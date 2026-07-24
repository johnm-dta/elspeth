from __future__ import annotations

from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from elspeth.web.blobs.protocol import (
    BlobError,
    BlobForkFenceLostError,
    BlobForkWriteFence,
    BlobQuotaExceededError,
    BlobRecord,
)
from elspeth.web.composer.guided.protocol import BLOB_REF_PATH_PREFIX
from elspeth.web.composer.guided.state_machine import GuidedSession
from elspeth.web.sessions.protocol import (
    GuidedForkSettlementCommand,
    GuidedOperationFailureCode,
    GuidedOperationFenceLostError,
    GuidedSessionResult,
    SessionGuidedOperationInProgressError,
    SessionNotFoundError,
)
from elspeth.web.sessions.routes.guided_operations import (
    GuidedOperationLease,
    guided_response_hash,
    raise_guided_operation_failure,
    reserve_or_replay_guided_operation,
)
from elspeth.web.sessions.titles import mint_default_session_title

from ._helpers import (
    UUID,
    APIRouter,
    AuditIntegrityError,
    BlobServiceProtocol,
    ComposerProgressSnapshot,
    CompositionStateData,
    CreateSessionRequest,
    Depends,
    ForkSessionRequest,
    ForkSessionResponse,
    HTTPException,
    InvalidForkTargetError,
    Query,
    Request,
    SessionResponse,
    SessionServiceProtocol,
    SQLAlchemyError,
    UpdateSessionRequest,
    UserIdentity,
    _get_composer_progress_registry,
    _get_session_compose_lock_registry,
    _session_response,
    _verify_session_ownership,
    deep_thaw,
    get_current_user,
)


def _copied_blob_for_inline_marker(
    marker: Mapping[str, Any],
    blob_map: dict[UUID, BlobRecord],
    *,
    composition_state_id: UUID,
    new_session_id: UUID,
    field_path: str,
) -> BlobRecord:
    old_ref = marker["blob_ref"]
    if type(old_ref) is not str:
        raise AuditIntegrityError(
            f"Tier 1 audit anomaly: composition_state {composition_state_id} "
            f"has inline_content blob_ref type {type(old_ref).__name__} at "
            f"{field_path} (expected UUID string). Fork aborted to prevent "
            f"cross-session blob reference in forked session {new_session_id}."
        )
    try:
        old_uuid = UUID(old_ref)
    except ValueError as exc:
        raise AuditIntegrityError(
            f"Tier 1 audit anomaly: composition_state {composition_state_id} "
            f"has non-UUID inline_content blob_ref {old_ref!r} at {field_path}. "
            f"Fork aborted to prevent cross-session blob reference in forked "
            f"session {new_session_id}."
        ) from exc
    if old_uuid not in blob_map:
        raise AuditIntegrityError(
            f"Tier 1 audit anomaly: composition_state {composition_state_id} "
            f"has inline_content blob_ref {old_ref!r} at {field_path}, but "
            f"the source blob was not copied into forked session {new_session_id}."
        )
    copied_blob = blob_map[old_uuid]
    marker_hash = marker["sha256"]
    if type(marker_hash) is not str:
        raise AuditIntegrityError(
            f"Tier 1 audit anomaly: composition_state {composition_state_id} "
            f"has inline_content marker at {field_path} without string sha256."
        )
    if copied_blob.content_hash != marker_hash:
        raise AuditIntegrityError(
            f"Tier 1 audit anomaly: copied blob {copied_blob.id} hash does "
            f"not match inline_content marker at {field_path} in forked "
            f"session {new_session_id}."
        )
    return copied_blob


def _rewrite_inline_content_blob_refs(
    value: Any,
    blob_map: dict[UUID, BlobRecord],
    *,
    composition_state_id: UUID,
    new_session_id: UUID,
    field_path: str,
) -> bool:
    if type(value) is dict:
        mode = value["mode"] if "mode" in value else None
        if mode == "inline_content" and "blob_ref" in value:
            copied_blob = _copied_blob_for_inline_marker(
                value,
                blob_map,
                composition_state_id=composition_state_id,
                new_session_id=new_session_id,
                field_path=field_path,
            )
            value["blob_ref"] = str(copied_blob.id)
            return True

        rewritten = False
        for key, child in value.items():
            if type(key) is str:
                rewritten = (
                    _rewrite_inline_content_blob_refs(
                        child,
                        blob_map,
                        composition_state_id=composition_state_id,
                        new_session_id=new_session_id,
                        field_path=f"{field_path}.{key}",
                    )
                    or rewritten
                )
        return rewritten

    if type(value) is list:
        rewritten = False
        for index, child in enumerate(value):
            rewritten = (
                _rewrite_inline_content_blob_refs(
                    child,
                    blob_map,
                    composition_state_id=composition_state_id,
                    new_session_id=new_session_id,
                    field_path=f"{field_path}[{index}]",
                )
                or rewritten
            )
        return rewritten

    return False


def _rewrite_source_blob_options(
    options: object,
    blob_map: dict[UUID, BlobRecord],
    source_blob_path_map: dict[str, BlobRecord],
    *,
    field_path: str,
) -> tuple[dict[str, Any], bool]:
    """Strictly rebuild one source options object without touching samples."""
    if type(options) is not dict:
        raise AuditIntegrityError(f"Tier 1 audit anomaly: {field_path} must be an exact dict")
    rebuilt = deep_thaw(options)
    if type(rebuilt) is not dict:  # pragma: no cover - deep_thaw contract
        raise AuditIntegrityError(f"Tier 1 audit anomaly: {field_path} thaw did not produce a dict")
    targets: dict[UUID, BlobRecord] = {}
    id_option_keys = tuple(
        key
        for key, value in rebuilt.items()
        if value is not None and type(key) is str and (key in {"blob_ref", "blob_id"} or key.endswith("_blob_id"))
    )
    for key in id_option_keys:
        old_ref = rebuilt[key]
        if type(old_ref) is not str:
            raise AuditIntegrityError(f"Tier 1 audit anomaly: {field_path}.{key} must be a UUID string")
        try:
            old_blob_id = UUID(old_ref)
        except ValueError as exc:
            raise AuditIntegrityError(f"Tier 1 audit anomaly: {field_path}.{key} is not a UUID string") from exc
        copied = blob_map.get(old_blob_id)
        if copied is None:
            raise AuditIntegrityError(f"Tier 1 audit anomaly: {field_path}.{key} was absent from the frozen fork plan")
        targets[old_blob_id] = copied
        rebuilt[key] = str(copied.id)
    for carrier in ("path", "file"):
        value = rebuilt.get(carrier)
        if carrier in rebuilt and value is not None and type(value) is not str:
            raise AuditIntegrityError(f"Tier 1 audit anomaly: {field_path}.{carrier} must be a string")
        if type(value) is not str:
            continue
        if value.startswith(BLOB_REF_PATH_PREFIX):
            try:
                old_blob_id = UUID(value.removeprefix(BLOB_REF_PATH_PREFIX))
            except ValueError as exc:
                raise AuditIntegrityError(f"Tier 1 audit anomaly: {field_path}.{carrier} has malformed blob sentinel") from exc
            copied = blob_map.get(old_blob_id)
            if copied is None:
                raise AuditIntegrityError(
                    f"Tier 1 audit anomaly: {field_path}.{carrier} blob sentinel was absent from the frozen fork plan"
                )
            targets[old_blob_id] = copied
            rebuilt[carrier] = f"{BLOB_REF_PATH_PREFIX}{copied.id}"
        elif value in source_blob_path_map:
            copied = source_blob_path_map[value]
            targets[next(source_id for source_id, record in blob_map.items() if record.id == copied.id)] = copied
            rebuilt[carrier] = copied.storage_path
    if len(targets) > 1:
        raise AuditIntegrityError(f"Tier 1 audit anomaly: {field_path} binds more than one source blob")
    target = next(iter(targets.values()), None)
    if target is None:
        return rebuilt, False
    rebuilt["blob_ref"] = str(target.id)
    if ("path" in rebuilt and not str(rebuilt["path"]).startswith(BLOB_REF_PATH_PREFIX)) or (
        "path" not in rebuilt and "file" not in rebuilt
    ):
        rebuilt["path"] = target.storage_path
    if "file" in rebuilt and not str(rebuilt["file"]).startswith(BLOB_REF_PATH_PREFIX):
        rebuilt["file"] = target.storage_path
    return rebuilt, True


def _rewrite_session_owned_sink_options(
    options: object,
    *,
    data_dir: Path,
    parent_session_id: UUID,
    child_session_id: UUID,
    field_path: str,
) -> tuple[dict[str, Any], bool]:
    """Rebase managed sink targets from the parent namespace to the child."""
    if type(options) is not dict:
        raise AuditIntegrityError(f"Tier 1 audit anomaly: {field_path} must be an exact dict")
    rebuilt = deep_thaw(options)
    if type(rebuilt) is not dict:  # pragma: no cover - deep_thaw contract
        raise AuditIntegrityError(f"Tier 1 audit anomaly: {field_path} thaw did not produce a dict")

    base = data_dir.resolve()
    rewritten = False
    for key in ("path", "file", "persist_directory"):
        value = rebuilt.get(key)
        if value is None:
            continue
        if type(value) is not str:
            raise AuditIntegrityError(f"Tier 1 audit anomaly: {field_path}.{key} must be a string")
        raw = Path(value)
        resolved = raw.resolve() if raw.is_absolute() else (base / raw).resolve()
        for namespace in ("outputs", "blobs"):
            parent_root = base / namespace / str(parent_session_id)
            try:
                suffix = resolved.relative_to(parent_root)
            except ValueError:
                continue
            rebuilt[key] = str(base / namespace / str(child_session_id) / suffix)
            rewritten = True
            break
    return rebuilt, rewritten


def _contains_exact_string(value: object, needles: frozenset[str]) -> bool:
    if type(value) is str:
        return value in needles or any(value == f"{BLOB_REF_PATH_PREFIX}{needle}" for needle in needles)
    if type(value) is dict:
        return any(_contains_exact_string(item, needles) for item in value.values())
    if type(value) is list:
        return any(_contains_exact_string(item, needles) for item in value)
    return False


def _rewrite_guided_blob_custody(
    composer_meta: Mapping[str, Any] | None,
    blob_map: dict[UUID, BlobRecord],
    source_blob_path_map: dict[str, BlobRecord],
    *,
    data_dir: Path,
    parent_session_id: UUID,
    child_session_id: UUID,
) -> tuple[dict[str, Any] | None, bool]:
    if composer_meta is None:
        return None, False
    if "guided_session" not in composer_meta:
        return dict(composer_meta), False
    guided_raw = composer_meta["guided_session"]
    if type(guided_raw) is not dict:
        raise AuditIntegrityError("Tier 1 audit anomaly: composer_meta.guided_session must be an exact dict")
    # Parse first and again after reconstruction: reviewed and pending sources
    # are schema-owned objects, not arbitrary JSON dictionaries.
    guided = GuidedSession.from_dict(guided_raw)
    rebuilt = guided.to_dict()
    rewritten = False
    for stable_id, reviewed in rebuilt["reviewed_sources"].items():
        reviewed["options"], changed = _rewrite_source_blob_options(
            reviewed["options"],
            blob_map,
            source_blob_path_map,
            field_path=f"guided_session.reviewed_sources[{stable_id!r}].options",
        )
        rewritten = rewritten or changed
    for stable_id, pending in rebuilt["pending_source_intents"].items():
        if pending["options"] is not None:
            pending["options"], changed = _rewrite_source_blob_options(
                pending["options"],
                blob_map,
                source_blob_path_map,
                field_path=f"guided_session.pending_source_intents[{stable_id!r}].options",
            )
            rewritten = rewritten or changed
        inspection = pending["inspection_facts"]
        if inspection is not None:
            identity = inspection["redacted_identity"]
            old_ref = identity.get("blob_id")
            if old_ref is not None:
                try:
                    old_blob_id = UUID(old_ref)
                except (TypeError, ValueError) as exc:
                    raise AuditIntegrityError("Tier 1 audit anomaly: pending source inspection blob_id is not a UUID string") from exc
                copied = blob_map.get(old_blob_id)
                if copied is None:
                    raise AuditIntegrityError(
                        "Tier 1 audit anomaly: pending source inspection blob_id was absent from the frozen fork plan"
                    )
                identity["blob_id"] = str(copied.id)
                rewritten = True
    for stable_id, reviewed in rebuilt["reviewed_outputs"].items():
        reviewed["options"], changed = _rewrite_session_owned_sink_options(
            reviewed["options"],
            data_dir=data_dir,
            parent_session_id=parent_session_id,
            child_session_id=child_session_id,
            field_path=f"guided_session.reviewed_outputs[{stable_id!r}].options",
        )
        rewritten = rewritten or changed
    for stable_id, pending in rebuilt["pending_output_intents"].items():
        if pending["options"] is not None:
            pending["options"], changed = _rewrite_session_owned_sink_options(
                pending["options"],
                data_dir=data_dir,
                parent_session_id=parent_session_id,
                child_session_id=child_session_id,
                field_path=f"guided_session.pending_output_intents[{stable_id!r}].options",
            )
            rewritten = rewritten or changed
    rebuilt = GuidedSession.from_dict(rebuilt).to_dict()
    source_ids = frozenset(str(blob_id) for blob_id in blob_map)
    if _contains_exact_string(rebuilt, source_ids):
        raise AuditIntegrityError("Tier 1 audit anomaly: forked guided metadata retained a parent blob id")
    result = dict(composer_meta)
    result["guided_session"] = rebuilt
    return result, rewritten


def _rewrite_fork_state_blob_custody(
    state: Any,
    blob_map: dict[UUID, BlobRecord],
    source_blob_path_map: dict[str, BlobRecord],
    *,
    data_dir: Path,
    parent_session_id: UUID,
    child_session_id: UUID,
) -> CompositionStateData | None:
    if state is None:
        return None
    sources = deep_thaw(state.sources) if state.sources is not None else None
    nodes = deep_thaw(state.nodes)
    edges = deep_thaw(state.edges)
    outputs = deep_thaw(state.outputs)
    metadata = deep_thaw(state.metadata_)
    composer_meta = deep_thaw(state.composer_meta) if state.composer_meta is not None else None
    rewritten = False
    if sources is not None:
        if type(sources) is not dict:
            raise AuditIntegrityError("Tier 1 audit anomaly: forked composition sources must be an exact dict")
        for source_name, source in sources.items():
            if type(source) is not dict:
                raise AuditIntegrityError(f"Tier 1 audit anomaly: sources.{source_name} must be an exact dict")
            if source.get("options") is not None:
                source["options"], changed = _rewrite_source_blob_options(
                    source["options"],
                    blob_map,
                    source_blob_path_map,
                    field_path=f"sources.{source_name}.options",
                )
                rewritten = rewritten or changed
    if outputs is not None and type(outputs) is not list:
        raise AuditIntegrityError("Tier 1 audit anomaly: forked composition outputs must be an exact list")
    for index, output in enumerate(outputs or []):
        if type(output) is not dict:
            raise AuditIntegrityError(f"Tier 1 audit anomaly: outputs[{index}] must be an exact dict")
        if output.get("options") is not None:
            output["options"], changed = _rewrite_session_owned_sink_options(
                output["options"],
                data_dir=data_dir,
                parent_session_id=parent_session_id,
                child_session_id=child_session_id,
                field_path=f"outputs[{index}].options",
            )
            rewritten = rewritten or changed
    for field_name, value in (("sources", sources), ("nodes", nodes), ("outputs", outputs)):
        rewritten = (
            _rewrite_inline_content_blob_refs(
                value,
                blob_map,
                composition_state_id=state.id,
                new_session_id=child_session_id,
                field_path=field_name,
            )
            or rewritten
        )
    composer_meta, guided_rewritten = _rewrite_guided_blob_custody(
        composer_meta,
        blob_map,
        source_blob_path_map,
        data_dir=data_dir,
        parent_session_id=parent_session_id,
        child_session_id=child_session_id,
    )
    rewritten = rewritten or guided_rewritten
    if not rewritten:
        return None
    return CompositionStateData(
        sources=sources,
        nodes=nodes,
        edges=edges,
        outputs=outputs,
        metadata_=metadata,
        is_valid=state.is_valid,
        validation_errors=list(state.validation_errors) if state.validation_errors else None,
        composer_meta=composer_meta,
    )


def register_session_routes(router: APIRouter) -> None:

    @router.post("", status_code=201, response_model=SessionResponse)
    async def create_session(
        body: CreateSessionRequest,
        request: Request,
        user: UserIdentity = Depends(get_current_user),  # noqa: B008
    ) -> SessionResponse:
        """Create a new session for the authenticated user."""
        service = request.app.state.session_service
        settings = request.app.state.settings
        title = body.title
        if title is None:
            # Mint the app-wide default title server-side (one convention,
            # elspeth-ef8c18a6cb). Archived sessions are included in the
            # collision set so an unarchive never resurfaces a duplicate
            # default row in the switcher. Local server time so the date in
            # the title matches the operator's wall clock, not UTC.
            existing = await service.list_sessions(
                user.user_id,
                settings.auth_provider,
                limit=200,
                offset=0,
                include_archived=True,
            )
            title = mint_default_session_title(
                datetime.now(UTC).astimezone(),
                (existing_session.title for existing_session in existing),
            )
        session = await service.create_session(
            user.user_id,
            title,
            settings.auth_provider,
        )
        return _session_response(session)

    @router.get("", response_model=list[SessionResponse])
    async def list_sessions(
        request: Request,
        user: UserIdentity = Depends(get_current_user),  # noqa: B008
        limit: int = Query(50, ge=1, le=200),
        offset: int = Query(0, ge=0),
        include_archived: bool = Query(False),
    ) -> list[SessionResponse]:
        """List sessions for the authenticated user."""
        service = request.app.state.session_service
        settings = request.app.state.settings
        sessions = await service.list_sessions(
            user.user_id,
            settings.auth_provider,
            limit=limit,
            offset=offset,
            include_archived=include_archived,
        )
        return [_session_response(s) for s in sessions]

    # NOTE: Registered before "/{session_id}" so FastAPI matches "_active"
    # against this exact-path route rather than attempting to parse "_active"
    # as a UUID (which would 422). The leading underscore also guarantees
    # the path can never collide with a real session id (UUIDs only contain
    # hex digits and hyphens).
    @router.get("/_active", response_model=list[ComposerProgressSnapshot])
    async def list_active_composer_requests(
        request: Request,
        user: UserIdentity = Depends(get_current_user),  # noqa: B008
    ) -> list[ComposerProgressSnapshot]:
        """List in-flight composer requests for the authenticated user.

        Closes the operator-visibility gap captured in the source report:
        Uvicorn's access log only writes the POST line when the response
        completes, so an in-flight or client-cancelled composer request
        was previously invisible to operators unless they polled
        ``/composer-progress`` for a specific session id.

        Returns snapshots whose phase is in NON_TERMINAL_PROGRESS_PHASES
        (starting / calling_model / using_tools / validating / saving),
        ordered by ``updated_at`` ascending so the longest-running request
        is at the top — typical triage starting point. Filtered by the
        authenticated user's id against the registry's internal user
        index, so a caller cannot see other users' active sessions even
        when they share a server.

        ``cancelled``, ``failed``, ``complete``, and ``idle`` snapshots
        are intentionally excluded from this view: those requests are no
        longer in flight and the per-session ``/composer-progress`` GET
        is the right surface for inspecting a terminal outcome.
        """
        registry = _get_composer_progress_registry(request)
        snapshots = await registry.list_active(user_id=str(user.user_id))
        return list(snapshots)

    @router.get("/{session_id}", response_model=SessionResponse)
    async def get_session(
        session_id: UUID,
        request: Request,
        user: UserIdentity = Depends(get_current_user),  # noqa: B008
    ) -> SessionResponse:
        """Get a single session. IDOR-protected."""
        session = await _verify_session_ownership(session_id, user, request)
        return _session_response(session)

    @router.patch("/{session_id}", response_model=SessionResponse)
    async def update_session(
        session_id: UUID,
        body: UpdateSessionRequest,
        request: Request,
        user: UserIdentity = Depends(get_current_user),  # noqa: B008
    ) -> SessionResponse:
        """Update a session's user-visible metadata. IDOR-protected."""
        session = await _verify_session_ownership(session_id, user, request)
        service = request.app.state.session_service
        updated = await service.update_session_title(session.id, body.title)
        return _session_response(updated)

    @router.delete("/{session_id}", status_code=204)
    async def delete_session(
        session_id: UUID,
        request: Request,
        user: UserIdentity = Depends(get_current_user),  # noqa: B008
    ) -> None:
        """Archive (delete) a session and all associated data.

        Rejects deletion while a pipeline run is active — archive_session()
        would delete run rows and blob directories out from under the
        background worker, causing status update failures and data loss.
        """
        session = await _verify_session_ownership(session_id, user, request)
        service = request.app.state.session_service
        execution_service = request.app.state.execution_service
        session_key = str(session.id)
        execution_lock = execution_service.get_session_lock(session_key)

        async with execution_lock:
            active_run = await service.get_active_run(session.id)
            if active_run is not None:
                raise HTTPException(
                    status_code=409,
                    detail="Cannot delete session while a pipeline run is active. Cancel the run first.",
                )

            try:
                await service.archive_session(session.id)
            except SessionGuidedOperationInProgressError as exc:
                raise HTTPException(
                    status_code=409,
                    detail="Cannot archive a session while a guided operation is in progress.",
                ) from exc
            # Archive is the durable boundary: preserve the live session's
            # ephemeral coordination state when it fails. Registry cleanup
            # retires held/waited locks only after their current users drain,
            # so deletion cannot split one session across old and new locks.
            execution_service.cleanup_session_lock(session_key)
            compose_lock_registry = _get_session_compose_lock_registry(request)
            await compose_lock_registry.cleanup_session_lock(session_key)
            progress_registry = _get_composer_progress_registry(request)
            await progress_registry.clear(session_key)

    @router.post(
        "/{session_id}/fork",
        status_code=201,
        response_model=ForkSessionResponse,
    )
    async def fork_from_message(
        session_id: UUID,
        body: ForkSessionRequest,
        request: Request,
        user: UserIdentity = Depends(get_current_user),  # noqa: B008
    ) -> ForkSessionResponse:
        """Fork a session from a specific user message.

        Creates a new session inheriting history and composition state up to
        the fork point, with the edited message replacing the original.
        The original session is never mutated.
        """
        await _verify_session_ownership(session_id, user, request)
        service: SessionServiceProtocol = request.app.state.session_service
        blob_service: BlobServiceProtocol = request.app.state.blob_service

        # Give structurally invalid fork targets their stable public status
        # before reserving durable retry state. The service rechecks under the
        # parent write lock so this read is never relied on for integrity.
        parent_messages = await service.get_messages(session_id, limit=None)
        fork_target = next((message for message in parent_messages if message.id == body.from_message_id), None)
        if fork_target is None:
            raise HTTPException(status_code=404, detail=f"Message {body.from_message_id} not found")
        if fork_target.role != "user":
            raise HTTPException(status_code=422, detail=str(InvalidForkTargetError(str(fork_target.id), fork_target.role)))

        async def _replay(result: object) -> ForkSessionResponse:
            if type(result) is not GuidedSessionResult:
                raise AuditIntegrityError("Session fork replay locator has the wrong result kind")
            return ForkSessionResponse(session_id=result.session_id)

        while True:
            try:
                reserved = await reserve_or_replay_guided_operation(
                    service=service,
                    session_id=session_id,
                    kind="session_fork",
                    request=body,
                    replay=_replay,
                )
            except SessionNotFoundError as exc:
                raise HTTPException(status_code=404, detail="Session not found") from exc
            if reserved is None:  # pragma: no cover - reservation is enabled
                raise AuditIntegrityError("Session fork operation was not reserved")
            if not isinstance(reserved, GuidedOperationLease):
                return reserved

            fence = reserved.fence
            staged = None
            try:
                staged = await service.fork_session(
                    fence,
                    fork_message_id=body.from_message_id,
                    new_message_content=body.new_message_content,
                )

                async def _checkpoint() -> None:
                    nonlocal fence
                    fence = await service.renew_guided_operation(
                        fence,
                        actor="composer_route",
                        lease_seconds=300,
                    )

                source_blobs = {entry.source_blob_id: await blob_service.get_blob(entry.source_blob_id) for entry in staged.blob_plan}
                blob_map = await blob_service.copy_blobs_for_fork(
                    session_id,
                    staged.session.id,
                    staged.blob_plan,
                    BlobForkWriteFence(
                        source_session_id=session_id,
                        target_session_id=staged.session.id,
                        operation_id=fence.operation_id,
                        lease_token=fence.lease_token,
                        attempt=fence.attempt,
                    ),
                    checkpoint=_checkpoint,
                )
                source_blob_path_map = {source_blobs[source_id].storage_path: copied for source_id, copied in blob_map.items()}
                rewritten_state = _rewrite_fork_state_blob_custody(
                    staged.state,
                    blob_map,
                    source_blob_path_map,
                    data_dir=Path(request.app.state.settings.data_dir),
                    parent_session_id=session_id,
                    child_session_id=staged.session.id,
                )
                response = ForkSessionResponse(session_id=staged.session.id)
                await _checkpoint()
                await service.settle_guided_fork_operation(
                    GuidedForkSettlementCommand(
                        fence=fence,
                        child_session_id=staged.session.id,
                        expected_current_state_id=staged.state.id if staged.state is not None else None,
                        edited_message_id=staged.messages[-1].id,
                        rewritten_state_id=uuid4() if rewritten_state is not None else None,
                        rewritten_state=rewritten_state,
                        response_hash=guided_response_hash(response),
                        actor="composer_route",
                    )
                )
                return response
            except (GuidedOperationFenceLostError, BlobForkFenceLostError):
                # A stale worker never cleans a child now owned by takeover.
                continue
            except Exception as primary_exc:
                failure_code: GuidedOperationFailureCode = (
                    "quota_exceeded"
                    if isinstance(primary_exc, BlobQuotaExceededError)
                    else "integrity_error"
                    if isinstance(primary_exc, AuditIntegrityError)
                    else "operation_failed"
                )
                try:
                    failed = await service.fail_guided_operation(
                        fence,
                        failure_code=failure_code,
                        actor="composer_route",
                    )
                except GuidedOperationFenceLostError:
                    # Only the fail-CAS winner owns cleanup.
                    continue

                if staged is not None:
                    try:
                        cleanup = await blob_service.cleanup_blobs_for_fork(
                            session_id,
                            staged.session.id,
                            fence.operation_id,
                        )
                    except (AuditIntegrityError, BlobError, SQLAlchemyError, OSError) as cleanup_exc:
                        primary_exc.add_note(
                            f"RecoveryFailed[{type(cleanup_exc).__name__}]: fork blob cleanup failed for "
                            f"child {staged.session.id} ({cleanup_exc})"
                        )
                    else:
                        for error in cleanup.errors:
                            primary_exc.add_note(
                                f"RecoveryFailed[{error.exc_type}]: could not delete fork blob {error.blob_id} "
                                f"from child {staged.session.id} ({error.detail})"
                            )
                    # The failed child is retained as archived audit evidence.
                    # Only its copied blobs are compensatable; deleting the
                    # session would also destroy the frozen plan envelope.

                raise_guided_operation_failure(failed)
