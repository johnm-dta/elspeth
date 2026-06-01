from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from elspeth.web.blobs.protocol import BlobRecord

from ._helpers import (
    UUID,
    APIRouter,
    AuditIntegrityError,
    BlobQuotaExceededError,
    BlobServiceProtocol,
    ChatMessageRecord,
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
    _message_response,
    _session_response,
    _state_response,
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
        raise TypeError(
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
        raise TypeError(
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


async def _archive_session_capturing_failure(
    service: SessionServiceProtocol,
    session_id: UUID,
    *,
    context: str,
) -> str | None:
    """Best-effort rollback archive of a partially-forked session.

    Returns an explicit ``RecoveryFailed[...]`` note (a recorded boundary
    outcome, not a swallow) when ``archive_session`` fails with a recoverable
    IO/DB error; the caller attaches it to the propagating primary error so the
    orphan session row is visible to operators. Returns ``None`` on success.

    The catch is narrowed to ``(SQLAlchemyError, OSError)`` so programmer bugs
    (AttributeError, TypeError) in ``archive_session`` still propagate. The
    cleanup failure is converted to a return value *inside* this helper, so it
    never enters the headline exception's ``__context__`` chain — the caller's
    ``raise ... from None`` / bare ``raise`` traceback semantics are preserved.
    """
    try:
        await service.archive_session(session_id)
        return None
    except (SQLAlchemyError, OSError) as cleanup_exc:
        return (
            f"RecoveryFailed[{type(cleanup_exc).__name__}]: "
            f"could not archive forked session {session_id} {context} "
            f"({cleanup_exc}). Manual cleanup of sessions.id={session_id} required."
        )


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
        session = await service.create_session(
            user.user_id,
            body.title,
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

        active_run = await service.get_active_run(session.id)
        if active_run is not None:
            raise HTTPException(
                status_code=409,
                detail="Cannot delete session while a pipeline run is active. Cancel the run first.",
            )

        try:
            await service.archive_session(session.id)
        finally:
            # Clean up ephemeral per-session state regardless of archive outcome.
            # If archive fails, the session still exists and a retry will re-enter
            # this path. The lock cleanup is idempotent.
            execution_service = request.app.state.execution_service
            execution_service.cleanup_session_lock(str(session.id))
            compose_lock_registry = _get_session_compose_lock_registry(request)
            await compose_lock_registry.cleanup_session_lock(str(session.id))
            progress_registry = _get_composer_progress_registry(request)
            await progress_registry.clear(str(session.id))

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
        settings = request.app.state.settings

        try:
            new_session, new_messages, copied_state = await service.fork_session(
                source_session_id=session_id,
                fork_message_id=body.from_message_id,
                new_message_content=body.new_message_content,
                user_id=user.user_id,
                auth_provider_type=settings.auth_provider,
            )
        except InvalidForkTargetError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

        # Everything after fork_session() is a compensatable post-commit
        # phase.  If ANY step fails, archive the fork to avoid orphaned
        # sessions/blobs/state.  BlobQuotaExceededError gets a specific
        # 413; all other failures re-raise after cleanup.
        blob_service: BlobServiceProtocol = request.app.state.blob_service
        try:
            source_blobs = await blob_service.list_blobs(session_id)
            # Copy blobs from source session into the forked session.
            # Returns old_id → new_blob mapping for source reference rewriting.
            blob_map = await blob_service.copy_blobs_for_fork(session_id, new_session.id)
            source_blob_path_map = {blob.storage_path: blob_map[blob.id] for blob in source_blobs if blob.id in blob_map}

            # Rewrite source references in the forked state so the fork is
            # self-contained.  Without this, blob_ref and path in the source
            # options still point at the original session's assets.
            if copied_state is not None:
                source_dict = deep_thaw(copied_state.source) if copied_state.source is not None else None
                nodes_data = deep_thaw(copied_state.nodes)
                edges_data = deep_thaw(copied_state.edges)
                outputs_data = deep_thaw(copied_state.outputs)
                metadata_data = deep_thaw(copied_state.metadata_)
                composer_meta_data = deep_thaw(copied_state.composer_meta) if copied_state.composer_meta is not None else None
                rewritten = False

                if source_dict is not None and not isinstance(source_dict, dict):
                    raise AuditIntegrityError(
                        f"Tier 1 audit anomaly: composition_state {copied_state.id} "
                        f"has source type {type(source_dict).__name__}, expected dict "
                        f"before fork blob rewrite for session {new_session.id}."
                    )

                if source_dict is not None and "options" in source_dict and source_dict["options"] is not None:
                    options = source_dict["options"]
                    if not isinstance(options, dict):
                        raise AuditIntegrityError(
                            f"Tier 1 audit anomaly: composition_state {copied_state.id} "
                            f"has source.options type {type(options).__name__}, expected "
                            f"dict before fork blob rewrite for session {new_session.id}."
                        )

                    rewrite_target = None
                    # Remap blob_ref to the new blob's ID.
                    # composition_states.source is Tier 1 ("our data") — the
                    # composer writes blob_ref as the blob's UUID string
                    # (composer/tools.py _execute_set_source_from_blob).  A
                    # non-UUID value here means a write-path bug, DB
                    # corruption, or tampering — crash with a diagnostic
                    # rather than silently skipping the remap.  Silent skip
                    # would leave the forked state's blob_ref pointing at
                    # the source session's blob, which is the cross-session
                    # reference class closed at the FK layer by the
                    # current-schema composite FK and is audit-contradictory
                    # on its face.  The enclosing ``except Exception``
                    # block archives the partially-created fork (see the
                    # cleanup-rollback site below), so this crash does
                    # not leak artifacts.
                    if "blob_ref" in options:
                        old_ref = options["blob_ref"]
                    else:
                        old_ref = None
                    if old_ref is not None:
                        if not isinstance(old_ref, str):
                            raise AuditIntegrityError(
                                f"Tier 1 audit anomaly: composition_state "
                                f"{copied_state.id} has blob_ref type "
                                f"{type(old_ref).__name__} in source.options "
                                f"(expected a UUID string written by "
                                f"composer/tools.py). Fork aborted to prevent "
                                f"cross-session blob reference in forked "
                                f"session {new_session.id}."
                            )
                        try:
                            old_uuid = UUID(old_ref)
                        except ValueError as exc:
                            raise AuditIntegrityError(
                                f"Tier 1 audit anomaly: composition_state "
                                f"{copied_state.id} has non-UUID blob_ref "
                                f"{old_ref!r} in source.options (expected a "
                                f"UUID string written by composer/tools.py). "
                                f"Fork aborted to prevent cross-session blob "
                                f"reference in forked session {new_session.id}."
                            ) from exc
                        if old_uuid not in blob_map:
                            raise AuditIntegrityError(
                                f"Tier 1 audit anomaly: composition_state "
                                f"{copied_state.id} has source blob_ref "
                                f"{old_ref!r}, but the source blob was not "
                                f"copied into forked session {new_session.id}."
                            )
                        rewrite_target = blob_map[old_uuid]

                    if rewrite_target is None:
                        for path_key in ("path", "file"):
                            if path_key not in options:
                                continue
                            path_value = options[path_key]
                            if isinstance(path_value, str) and path_value in source_blob_path_map:
                                rewrite_target = source_blob_path_map[path_value]
                                break

                    if rewrite_target is not None:
                        options["blob_ref"] = str(rewrite_target.id)
                        if "path" in options or "file" not in options:
                            options["path"] = rewrite_target.storage_path
                        if "file" in options:
                            options["file"] = rewrite_target.storage_path
                        rewritten = True

                if source_dict is not None:
                    rewritten = (
                        _rewrite_inline_content_blob_refs(
                            source_dict,
                            blob_map,
                            composition_state_id=copied_state.id,
                            new_session_id=new_session.id,
                            field_path="source",
                        )
                        or rewritten
                    )
                rewritten = (
                    _rewrite_inline_content_blob_refs(
                        nodes_data,
                        blob_map,
                        composition_state_id=copied_state.id,
                        new_session_id=new_session.id,
                        field_path="nodes",
                    )
                    or rewritten
                )
                rewritten = (
                    _rewrite_inline_content_blob_refs(
                        outputs_data,
                        blob_map,
                        composition_state_id=copied_state.id,
                        new_session_id=new_session.id,
                        field_path="outputs",
                    )
                    or rewritten
                )

                if rewritten:
                    # Save updated state with remapped source. Preserve the
                    # source state's composer_meta — fork inherits the
                    # operational provenance of the parent compose.
                    state_data = CompositionStateData(
                        source=source_dict,
                        nodes=nodes_data,
                        edges=edges_data,
                        outputs=outputs_data,
                        metadata_=metadata_data,
                        is_valid=copied_state.is_valid,
                        validation_errors=list(copied_state.validation_errors) if copied_state.validation_errors else None,
                        composer_meta=composer_meta_data,
                    )
                    copied_state = await service.save_composition_state(
                        new_session.id,
                        state_data,
                        # Preserves pre-fix labelling. The fork-time source-
                        # storage rewrite previously wrote ``session_seed``
                        # under the hardcoded label and continues to do so.
                        # Whether this row should carry ``session_fork``
                        # (the rewrite is part of the fork operation) or a
                        # new ``fork_storage_rewrite`` discriminator is a
                        # separate audit-attribution question outside the
                        # scope of elspeth-obs-f217c634aa.
                        provenance="session_seed",
                    )

                    # The edited user message (last in list) still references
                    # the pre-rewrite state.  Re-point it at the replacement
                    # state so message-state lineage is self-contained.
                    user_msg = new_messages[-1]
                    await service.update_message_composition_state(
                        user_msg.id,
                        copied_state.id,
                    )
                    new_messages[-1] = ChatMessageRecord(
                        id=user_msg.id,
                        session_id=user_msg.session_id,
                        role=user_msg.role,
                        content=user_msg.content,
                        raw_content=user_msg.raw_content,
                        tool_calls=user_msg.tool_calls,
                        created_at=user_msg.created_at,
                        sequence_no=user_msg.sequence_no,
                        composition_state_id=copied_state.id,
                        writer_principal=user_msg.writer_principal,
                        tool_call_id=user_msg.tool_call_id,
                        parent_assistant_id=user_msg.parent_assistant_id,
                    )
        except BlobQuotaExceededError:
            # Build the HTTPException up-front so cleanup failures can be
            # attached as a note on the object that actually propagates —
            # the inner BlobQuotaExceededError is suppressed by `from None`
            # and any note attached to it would never reach operator logs.
            quota_exc = HTTPException(
                status_code=413,
                detail="Blob quota exceeded during fork — unable to copy files",
            )
            cleanup_note = await _archive_session_capturing_failure(service, new_session.id, context="after blob quota rollback")
            if cleanup_note is not None:
                quota_exc.add_note(cleanup_note)
            raise quota_exc from None
        except Exception as primary_exc:
            # Mirror the RecoveryFailed[...] convention from
            # ``BlobServiceImpl.copy_blobs_for_fork`` and
            # ``BlobServiceImpl.finalize_run_output_blobs`` (web/blobs/service.py):
            # cleanup failures must NOT mask the original error.  The
            # best-effort archive captures any recoverable cleanup failure as a
            # note attached to primary_exc; the bare `raise` preserves
            # primary_exc and its original traceback as the headline.
            cleanup_note = await _archive_session_capturing_failure(service, new_session.id, context="during fork rollback")
            if cleanup_note is not None:
                primary_exc.add_note(cleanup_note)
            raise

        return ForkSessionResponse(
            session=_session_response(new_session),
            messages=[_message_response(m) for m in new_messages],
            composition_state=_state_response(copied_state) if copied_state else None,
        )
