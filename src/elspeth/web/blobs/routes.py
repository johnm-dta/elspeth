"""REST API routes for session-scoped blob management.

All endpoints require authentication and verify session ownership.
Blob ownership is enforced via the session boundary — a blob must
belong to the authenticated user's session.
"""

from __future__ import annotations

from typing import cast
from urllib.parse import quote
from uuid import UUID

from fastapi import APIRouter, Depends, File, HTTPException, Query, Request, UploadFile
from fastapi.responses import Response

from elspeth.web.auth.middleware import get_current_user
from elspeth.web.auth.models import UserIdentity
from elspeth.web.blobs.protocol import (
    ALLOWED_MIME_TYPES,
    AllowedMimeType,
    BlobActiveRunError,
    BlobContentMissingError,
    BlobIntegrityError,
    BlobNotFoundError,
    BlobQuotaExceededError,
    BlobRecord,
    BlobStateError,
)
from elspeth.web.blobs.schemas import BlobMetadataResponse, CreateInlineBlobRequest
from elspeth.web.blobs.service import BlobServiceImpl, sanitize_filename
from elspeth.web.blobs.sniff import detect_mime_type


def _blob_response(record: BlobRecord) -> BlobMetadataResponse:
    """Convert a BlobRecord to a response model (excludes storage_path)."""
    return BlobMetadataResponse(
        id=str(record.id),
        session_id=str(record.session_id),
        filename=record.filename,
        mime_type=record.mime_type,
        size_bytes=record.size_bytes,
        content_hash=record.content_hash,
        created_at=record.created_at,
        created_by=record.created_by,
        source_description=record.source_description,
        status=record.status,
    )


async def _verify_session_and_get_blob_service(
    session_id: UUID,
    user: UserIdentity,
    request: Request,
) -> BlobServiceImpl:
    """Verify session ownership and return the blob service.

    Returns 404 (not 403) for ownership failures to preserve the
    anti-IDOR pattern from the session routes.
    """
    session_service = request.app.state.session_service
    try:
        session = await session_service.get_session(session_id)
    except ValueError:
        raise HTTPException(status_code=404, detail="Session not found") from None

    settings = request.app.state.settings
    if session.user_id != user.user_id or session.auth_provider_type != settings.auth_provider:
        raise HTTPException(status_code=404, detail="Session not found")

    blob_service: BlobServiceImpl = request.app.state.blob_service
    return blob_service


async def _get_owned_blob(
    blob_service: BlobServiceImpl,
    session_id: UUID,
    blob_id: UUID,
) -> BlobRecord:
    """Fetch a blob and verify it belongs to the given session.

    Returns 404 for missing blobs or session mismatches.
    """
    try:
        blob = await blob_service.get_blob(blob_id)
    except BlobNotFoundError:
        raise HTTPException(status_code=404, detail="Blob not found") from None

    if blob.session_id != session_id:
        raise HTTPException(status_code=404, detail="Blob not found")

    return blob


def create_blobs_router() -> APIRouter:
    """Create the blob management router."""
    router = APIRouter(
        prefix="/api/sessions/{session_id}/blobs",
        tags=["blobs"],
    )

    @router.post("", status_code=201, response_model=BlobMetadataResponse)
    async def create_blob_upload(
        session_id: UUID,
        request: Request,
        file: UploadFile = File(...),  # noqa: B008
        user: UserIdentity = Depends(get_current_user),  # noqa: B008
    ) -> BlobMetadataResponse:
        """Create a blob from a multipart file upload."""
        blob_service = await _verify_session_and_get_blob_service(session_id, user, request)
        settings = request.app.state.settings

        # Record the browser-declared MIME type but do NOT reject based on
        # it alone. Browser MIME assignment is Tier 3 (untrusted) — browsers
        # commonly report .csv files as "application/vnd.ms-excel" or other
        # non-standard types. Server-side content sniffing below determines
        # the effective type.
        mime_type = file.content_type or "application/octet-stream"

        # Validate filename at the HTTP boundary (Tier 3) so malformed
        # uploads surface as 422, not an uncaught ValueError propagated
        # from sanitize_filename() deeper in the service layer.
        try:
            original_filename = sanitize_filename(file.filename or "upload")
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from None
        chunks: list[bytes] = []
        total_size = 0
        while chunk := await file.read(8192):
            total_size += len(chunk)
            if total_size > settings.max_upload_bytes:
                raise HTTPException(
                    status_code=413,
                    detail=f"File exceeds maximum size of {settings.max_upload_bytes} bytes",
                )
            chunks.append(chunk)
        content = b"".join(chunks)

        # Server-side MIME detection — reject if declared type doesn't
        # match detected content (defense-in-depth, client MIME is untrusted)
        detected = detect_mime_type(content)
        if detected is not None and detected != mime_type and detected not in ALLOWED_MIME_TYPES:
            raise HTTPException(
                status_code=415,
                detail=f"Detected content type '{detected}' does not match declared '{mime_type}' and is not in the allowed set.",
            )
        # Use detected type if available — record the truth, not the claim
        effective_mime = detected if detected is not None else mime_type
        if effective_mime not in ALLOWED_MIME_TYPES:
            raise HTTPException(
                status_code=415,
                detail=f"Detected content type '{effective_mime}' is not in the allowed set.",
            )
        # The membership check above narrows effective_mime to one of the
        # AllowedMimeType Literal members; mypy cannot follow a frozenset
        # membership test to a Literal, so cast explicitly.  The cast is
        # safe *because* of the runtime check two lines up — removing the
        # check would also be a type-safety regression.
        effective_mime_typed = cast(AllowedMimeType, effective_mime)

        try:
            record = await blob_service.create_blob(
                session_id=session_id,
                filename=original_filename,
                content=content,
                mime_type=effective_mime_typed,
                created_by="user",
                source_description="uploaded",
            )
        except BlobQuotaExceededError as exc:
            raise HTTPException(status_code=413, detail=str(exc)) from None
        return _blob_response(record)

    @router.post("/inline", status_code=201, response_model=BlobMetadataResponse)
    async def create_blob_inline(
        session_id: UUID,
        body: CreateInlineBlobRequest,
        request: Request,
        user: UserIdentity = Depends(get_current_user),  # noqa: B008
    ) -> BlobMetadataResponse:
        """Create a blob from inline text/JSON content.

        Request validation — including filename sanitization, mime_type
        allowlist enforcement, and rejection of unknown fields — is
        performed by CreateInlineBlobRequest at the Pydantic layer.
        This route only enforces size and persistence-layer concerns.
        """
        blob_service = await _verify_session_and_get_blob_service(session_id, user, request)

        settings = request.app.state.settings
        content_bytes = body.content.encode("utf-8")
        if len(content_bytes) > settings.max_upload_bytes:
            raise HTTPException(
                status_code=413,
                detail=f"Content exceeds maximum size of {settings.max_upload_bytes} bytes",
            )

        try:
            record = await blob_service.create_blob(
                session_id=session_id,
                filename=body.filename,
                content=content_bytes,
                mime_type=body.mime_type,
                created_by="user",
                source_description="created inline",
            )
        except BlobQuotaExceededError as exc:
            raise HTTPException(status_code=413, detail=str(exc)) from None
        return _blob_response(record)

    @router.get("", response_model=list[BlobMetadataResponse])
    async def list_blobs(
        session_id: UUID,
        request: Request,
        user: UserIdentity = Depends(get_current_user),  # noqa: B008
        limit: int = Query(50, ge=1, le=500),
        offset: int = Query(0, ge=0),
    ) -> list[BlobMetadataResponse]:
        """List blobs for a session, newest first."""
        blob_service = await _verify_session_and_get_blob_service(session_id, user, request)
        records = await blob_service.list_blobs(session_id, limit=limit, offset=offset)
        return [_blob_response(r) for r in records]

    @router.get("/{blob_id}", response_model=BlobMetadataResponse)
    async def get_blob_metadata(
        session_id: UUID,
        blob_id: UUID,
        request: Request,
        user: UserIdentity = Depends(get_current_user),  # noqa: B008
    ) -> BlobMetadataResponse:
        """Get blob metadata."""
        blob_service = await _verify_session_and_get_blob_service(session_id, user, request)
        blob = await _get_owned_blob(blob_service, session_id, blob_id)
        return _blob_response(blob)

    @router.get("/{blob_id}/content")
    async def download_blob_content(
        session_id: UUID,
        blob_id: UUID,
        request: Request,
        user: UserIdentity = Depends(get_current_user),  # noqa: B008
    ) -> Response:
        """Download blob content.

        Enforces lifecycle state before reading: only ``ready`` blobs
        are downloadable.  Pending/error blobs return 409 (the resource
        exists but its state precludes the operation).
        """
        blob_service = await _verify_session_and_get_blob_service(session_id, user, request)
        blob = await _get_owned_blob(blob_service, session_id, blob_id)

        # Lifecycle guard — defense-in-depth (service layer also checks)
        if blob.status != "ready":
            raise HTTPException(
                status_code=409,
                detail=f"Blob is in '{blob.status}' state and is not downloadable",
            )

        try:
            content = await blob_service.read_blob_content(blob_id)
        except BlobNotFoundError:
            raise HTTPException(status_code=404, detail="Blob content not found") from None
        except BlobStateError:
            # Use a generic message — blob.status was fetched before the
            # service call and may be stale if a concurrent transition
            # happened between the route guard and the service read.
            raise HTTPException(
                status_code=409,
                detail="Blob is not in a downloadable state",
            ) from None
        except (BlobIntegrityError, BlobContentMissingError):
            raise HTTPException(
                status_code=500,
                detail="Blob content integrity verification failed",
            ) from None

        return Response(
            content=content,
            media_type=blob.mime_type,
            headers={"Content-Disposition": f"attachment; filename*=UTF-8''{quote(blob.filename, safe='')}"},
        )

    @router.delete("/{blob_id}", status_code=204)
    async def delete_blob(
        session_id: UUID,
        blob_id: UUID,
        request: Request,
        user: UserIdentity = Depends(get_current_user),  # noqa: B008
    ) -> None:
        """Delete a blob and its backing file."""
        blob_service = await _verify_session_and_get_blob_service(session_id, user, request)
        await _get_owned_blob(blob_service, session_id, blob_id)

        try:
            await blob_service.delete_blob(blob_id)
        except BlobNotFoundError:
            raise HTTPException(status_code=404, detail="Blob not found") from None
        except BlobActiveRunError as exc:
            raise HTTPException(
                status_code=409,
                detail=str(exc),
            ) from None

    return router
