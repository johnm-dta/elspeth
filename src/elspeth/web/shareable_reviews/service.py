"""``ShareableReviewService`` — business logic for the "Save for review" verb.

Responsibilities:

* Validate the composition before marking (validation + readiness gates).
* Freeze a complete snapshot of the composition at mark-time — including
  the readiness panel — into a content-addressable blob in the payload
  store.
* Record the audit event in ``composer_completion_events_table`` BEFORE
  writing the blob (audit-first ordering).
* Mint a signed capability token that encodes the content-address.
* Resolve an inbound token back to the frozen snapshot for the reviewer.

Audit-first ordering (load-bearing):

1. Build snapshot dict in memory.
2. Canonical-JSON-serialize → compute ``payload_digest``.
3. INSERT ``composer_completion_events`` row (sync, crash-on-failure).
4. Write blob to payload store.
5. Sign token.
6. Return.

If the audit insert fails, no blob is ever written. If the blob write fails
after the audit insert, the audit row stands as honest evidence of the
attempt; no token is returned.

Frozen-at-mark-time discipline (load-bearing):

* ``get_shareable_link`` may re-mint only when the current
  ``(session, state)`` already has a ``mark_ready_for_review`` audit row
  whose snapshot blob still exists. It does not create a new share decision.
* ``resolve_token`` reads ``audit_readiness`` directly from the blob; it
  never re-calls ``ReadinessService.compute_snapshot``. This means:
    - The reviewer sees exactly what the owner saw at mark-time, even if
      the live state has drifted.
    - The ``payload_digest`` fingerprint covers the readiness panel too —
      content-addressing is evidentially complete.
    - The reviewer-vs-owner permission question for the readiness service
      never arises at resolve time.

Mark-time gate: ``mark_ready_for_review`` raises
``CompositionNotRunnableError`` if validation fails OR if any readiness row
has ``status == "error"``. ``status == "warning"`` (e.g. pending LLM
interpretations) is permitted; the reviewer sees the warning.

Layer: L3 (web application).
"""

from __future__ import annotations

import hashlib
import json
import secrets
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any, Final, Protocol, TypedDict, cast
from uuid import UUID, uuid4

from sqlalchemy import desc, insert, select
from sqlalchemy.engine import Engine

from elspeth.core.canonical import canonical_json
from elspeth.core.payload_store import FilesystemPayloadStore
from elspeth.web.audit_readiness.models import AuditReadinessSnapshot
from elspeth.web.composer.telemetry_phase8 import (
    SessionsTelemetry,
    record_session_completed,
)
from elspeth.web.composer.yaml_generator import generate_yaml
from elspeth.web.config import WebSettings
from elspeth.web.sessions.converters import state_from_record
from elspeth.web.sessions.models import composer_completion_events_table
from elspeth.web.shareable_reviews.models import (
    MarkReadyForReviewResponse,
    ShareableLinkResponse,
    SharedInspectResponse,
)
from elspeth.web.shareable_reviews.signer import (
    ShareTokenPayload,
    ShareTokenSigner,
)

# Path-only share URL — the frontend's `location.origin` plus this string
# resolves to a full clickable URL. Avoids adding yet another deployment-
# base-URL setting to ``WebSettings``. The ``/#/shared/`` shape matches the
# SPA hash route in ``hooks/useHashRouter.ts`` — the recipient's browser
# opens the SPA at the hash, which calls the
# ``GET /api/sessions/shared/{token}`` backend route.
_SHARE_URL_PREFIX = "/#/shared/"

# All payload-digest values on the wire carry the ``sha256:`` prefix so the
# hash algorithm is self-describing. The payload store accepts only the raw
# hex; the prefix is added/stripped at the service boundary.
_DIGEST_PREFIX = "sha256:"
_NOT_MARKED_READY_DETAIL = "current composition state has not been marked ready for review"


class CompositionNotRunnableError(Exception):
    """Raised when the composition fails the mark-for-review gates.

    Two distinct failure modes carry the same exception type because the
    route layer maps both to HTTP 409 with the same user-facing message
    ("the composition is not in a shareable state — fix the surfaced
    errors and try again").

    The ``reason`` field disambiguates for logs and tests:

    * ``"validation_failed"`` — ``ExecutionService.validate`` returned
      ``is_valid=False``.
    * ``"readiness_error_row"`` — ``ReadinessService.compute_snapshot``
      returned a row with ``status == "error"``. Sharing a known-broken
      readiness state is share-theatre; the gate refuses.
    * ``"not_marked_ready"`` — re-minting was requested before a
      successful ``mark_ready_for_review`` snapshot blob exists for the
      current state.
    * ``"readiness_state_drift"`` — validation and readiness were computed
      against different composition versions.
    * ``"completion_event_missing"`` — re-mint was requested before the
      current state had passed ``mark_ready_for_review``.
    """

    def __init__(self, *, reason: str, detail: str = "") -> None:
        super().__init__(detail or reason)
        self.reason = reason
        self.detail = detail


# ── Injected dependency protocols ────────────────────────────────────────


class _SessionServiceLike(Protocol):
    """Subset of ``SessionServiceProtocol`` the service uses.

    Declared as a Protocol so tests can substitute mocks without inheriting
    the full session-service surface.
    """

    async def get_current_state(self, session_id: UUID) -> Any: ...
    async def get_session(self, session_id: UUID) -> Any: ...


class _ExecutionServiceLike(Protocol):
    async def validate(self, session_id: UUID, *, user_id: str | None = None) -> Any: ...
    async def validate_state(self, state: Any, *, user_id: str | None = None) -> Any: ...


class _ReadinessServiceLike(Protocol):
    async def compute_snapshot(self, *, session_id: UUID, user_id: str) -> AuditReadinessSnapshot: ...


# ── Helpers ──────────────────────────────────────────────────────────────


class _BlobShape(TypedDict):
    """Wire-shape contract for the canonical-JSON snapshot blob.

    The same five keys are produced by ``_build_snapshot`` and consumed by
    ``ShareableReviewService.resolve_token``. Adding a key here is a
    breaking change for outstanding share artifacts — bump the signer
    payload version + ship a migration before extending.
    """

    pipeline_metadata: Any  # CompositionObject (dict[str, JsonValue])
    composition_snapshot: Any  # CompositionObject
    yaml: str
    audit_readiness: Any  # AuditReadinessSnapshot.model_dump output
    created_by_user_id: str


# Closed-set producer-side guard. The digest only proves bytes-on-disk
# integrity; it does NOT detect a producer drift that emits an extra or
# missing key (the buggy bytes round-trip cleanly through the digest).
# ``_assert_blob_shape`` catches that class of drift at the producer so
# the bug surfaces in the owner's request instead of several frames away
# in the reviewer's Pydantic construction. Update both ``_BlobShape`` AND
# this constant in the same commit — and bump the signer payload version
# per the ``_BlobShape`` docstring.
_BLOB_KEYS: Final[frozenset[str]] = frozenset(
    {
        "pipeline_metadata",
        "composition_snapshot",
        "yaml",
        "audit_readiness",
        "created_by_user_id",
    }
)


@dataclass(frozen=True, slots=True)
class _Snapshot:
    """In-memory snapshot built once at mark-time.

    Carries the serialized canonical bytes (for hashing + blob storage)
    plus the live AuditReadinessSnapshot (so the response can include it
    without round-tripping through JSON).
    """

    canonical_bytes: bytes
    payload_digest: str  # sha256:<hex>
    digest_hex: str  # bare hex for payload-store calls
    audit_readiness: AuditReadinessSnapshot
    state_id: UUID
    created_at: datetime


def _has_error_readiness_row(snapshot: AuditReadinessSnapshot) -> bool:
    """Mark-time gate: any row with status='error' blocks the share."""
    return any(row.status == "error" for row in snapshot.rows)


def _build_snapshot(
    *,
    session_id: UUID,
    state_record: Any,
    audit_readiness: AuditReadinessSnapshot,
    created_by_user_id: str,
) -> _Snapshot:
    """Build the canonical-JSON snapshot blob and compute its content-address.

    The blob shape is the same one ``SharedInspectResponse`` deserialises
    on the resolve path — pipeline_metadata, composition_snapshot, yaml,
    audit_readiness, created_by_user_id, created_at. This is the
    contractual wire shape for the share artifact.
    """
    composition_state = state_from_record(state_record)
    yaml_text = generate_yaml(composition_state)
    composition_dict = composition_state.to_dict()
    # ``metadata`` is normalised to ``{"name", "description"}`` by
    # CompositionState.to_dict — that IS the pipeline_metadata wire shape.
    # Direct indexing per CLAUDE.md offensive programming: ``to_dict``
    # always emits a ``metadata`` key. KeyError here is a contract
    # violation of CompositionState.to_dict.
    pipeline_metadata = composition_dict["metadata"]
    # The blob carries ONLY content-addressed content. Mark-time and
    # mint-time live in the token envelope (and the audit row), not here,
    # so two re-mints over an unchanged composition yield the same
    # payload_digest.
    #
    # The readiness snapshot is normalised: its ``checked_at`` is pinned
    # to the composition state's ``created_at`` rather than the live
    # ``datetime.now()``. Without this normalisation the
    # ``AuditReadinessSnapshot`` would carry a fresh wall-clock on each
    # re-mint, breaking digest stability even on an otherwise unchanged
    # composition. Pinning to state.created_at is semantically defensible:
    # the readiness panel describes "what readiness signal accompanied
    # the state when it was committed," not "when was this query run."
    audit_readiness_dict = audit_readiness.model_dump(mode="json")
    audit_readiness_dict["checked_at"] = state_record.created_at.isoformat()
    blob: _BlobShape = {
        "pipeline_metadata": pipeline_metadata,
        "composition_snapshot": composition_dict,
        "yaml": yaml_text,
        "audit_readiness": audit_readiness_dict,
        "created_by_user_id": created_by_user_id,
    }
    # Producer-side drift guard. ``_BlobShape: TypedDict`` is a static
    # type — Python does not enforce it at runtime. Without this assert,
    # a future refactor adding/dropping a key would produce a digest-stable
    # but shape-drifted blob; the bug would surface in Pydantic
    # construction in ``SharedInspectResponse`` several frames downstream.
    # Crash here so the bug surfaces at the producer.
    actual_keys = frozenset(blob.keys())
    if actual_keys != _BLOB_KEYS:
        missing = _BLOB_KEYS - actual_keys
        extra = actual_keys - _BLOB_KEYS
        raise RuntimeError(
            f"shareable-review snapshot blob shape drift: missing={sorted(missing)!r} extra={sorted(extra)!r}. "
            "Update _BlobShape and _BLOB_KEYS together, and bump the signer payload version."
        )
    canonical_str = canonical_json(blob)
    canonical_bytes = canonical_str.encode("utf-8")
    digest_hex = hashlib.sha256(canonical_bytes).hexdigest()
    return _Snapshot(
        canonical_bytes=canonical_bytes,
        payload_digest=_DIGEST_PREFIX + digest_hex,
        digest_hex=digest_hex,
        audit_readiness=audit_readiness,
        state_id=state_record.id,
        created_at=datetime.now(UTC),
    )


# ── Service ──────────────────────────────────────────────────────────────


class ShareableReviewService:
    """Orchestrator for the shareable-reviews capability.

    Dependencies are injected so tests can substitute mocks. In production
    the wiring lives in ``web/app.py``.
    """

    def __init__(
        self,
        *,
        session_service: _SessionServiceLike,
        execution_service: _ExecutionServiceLike,
        readiness_service: _ReadinessServiceLike,
        signer: ShareTokenSigner,
        settings: WebSettings,
        sessions_db_engine: Engine,
        payload_store: FilesystemPayloadStore,
        telemetry: SessionsTelemetry,
    ) -> None:
        self._session_service = session_service
        self._execution_service = execution_service
        self._readiness_service = readiness_service
        self._signer = signer
        self._settings = settings
        self._sessions_db_engine = sessions_db_engine
        self._payload_store = payload_store
        # Phase 8 Sub-task 7c — composer.session.completed_total counter.
        # Mirrors the sessions/service.py pattern at line 48/2446 (the
        # cohort b1 interpretation-opt-out emit). The container is owned
        # by the FastAPI app and threaded through app.state; we hold a
        # reference so the audit-row-then-counter sequence at
        # mark_ready_for_review can fire the helper after engine.begin()
        # exits cleanly.
        self._telemetry = telemetry

    async def mark_ready_for_review(self, *, session_id: UUID, user_id: str) -> MarkReadyForReviewResponse:
        """Build a signed share artifact for ``(session_id, current state)``.

        Sequence (audit-first ordering):

        1. Validate the composition (raises ``CompositionNotRunnableError``
           on failure).
        2. Compute the readiness snapshot using the OWNER's user_id and
           apply the mark-time gate (no row with status='error').
        3. Build the canonical-JSON snapshot bytes; compute payload_digest.
        4. INSERT the audit row into ``composer_completion_events_table``.
        5. Write the blob to the payload store.
        6. Sign the token; return.

        If step (4) fails the request fails before any blob is written. If
        step (5) fails the audit row stands as honest evidence of the
        attempt; no token is returned.
        """
        state_record = await self._session_service.get_current_state(session_id)
        if state_record is None:
            raise CompositionNotRunnableError(
                reason="state_missing",
                detail="No composition state exists for this session",
            )
        composition_state = state_from_record(state_record)
        validation = await self._execution_service.validate_state(composition_state, user_id=user_id)
        if not validation.is_valid:
            raise CompositionNotRunnableError(
                reason="validation_failed",
                detail="composition validation failed; fix errors before sharing",
            )

        audit_readiness = await self._readiness_service.compute_snapshot(session_id=session_id, user_id=user_id)
        if _has_error_readiness_row(audit_readiness):
            raise CompositionNotRunnableError(
                reason="readiness_error_row",
                detail="readiness panel reports an error; resolve before sharing",
            )
        if audit_readiness.composition_version != state_record.version:
            raise CompositionNotRunnableError(
                reason="readiness_state_drift",
                detail="composition changed while preparing the review snapshot; retry against the current state",
            )

        snapshot = _build_snapshot(
            session_id=session_id,
            state_record=state_record,
            audit_readiness=audit_readiness,
            created_by_user_id=user_id,
        )

        # Stamp expiry now so the same value lands in both the audit row
        # and the signed token envelope. The validity period is set at
        # issue-time; the signer's verify() rejects after this point.
        lifetime = timedelta(seconds=self._settings.shareable_link_lifetime_seconds)
        expires_at = snapshot.created_at + lifetime

        # AUDIT FIRST. Sync, crash-on-failure. If this raises, no blob
        # gets written and the caller sees the error.
        with self._sessions_db_engine.begin() as conn:
            conn.execute(
                insert(composer_completion_events_table).values(
                    id=str(uuid4()),
                    session_id=str(session_id),
                    composition_state_id=str(snapshot.state_id),
                    event_type="mark_ready_for_review",
                    actor=user_id,
                    created_at=snapshot.created_at,
                    payload_digest=snapshot.payload_digest,
                    expires_at=expires_at,
                )
            )

        # Phase 8 Sub-task 7c (telemetry-backfill: phase-6).
        # Audit primacy: the helper runs AFTER the engine.begin() block
        # commits the audit row. If the INSERT above raises, control
        # never reaches this line and the counter stays at zero — the
        # superset rule (counter aggregates over committed audit rows)
        # is structurally enforced by the placement.
        record_session_completed(self._telemetry, completion_verb="mark_ready_for_review")

        # Then blob.
        stored_hex = self._payload_store.store(snapshot.canonical_bytes)
        # Defense in depth: the payload store recomputes the digest;
        # a mismatch here is a Tier-1 corruption event.
        if stored_hex != snapshot.digest_hex:
            raise RuntimeError(
                f"payload store returned digest {stored_hex} but we computed {snapshot.digest_hex} — possible canonicalisation drift"
            )

        token = self._sign_token(
            session_id=session_id,
            state_id=snapshot.state_id,
            payload_digest=snapshot.payload_digest,
            created_by_user_id=user_id,
            created_at=snapshot.created_at,
            expires_at=expires_at,
        )

        return MarkReadyForReviewResponse(
            token=token,
            share_url=_SHARE_URL_PREFIX + token,
            expires_at=expires_at,
            payload_digest=snapshot.payload_digest,
        )

    async def get_shareable_link(self, *, session_id: UUID, user_id: str) -> ShareableLinkResponse:
        """Re-mint a token for an already-marked current snapshot.

        Always mints a fresh token. Content-addressing guarantees that two
        calls on an unchanged state produce identical ``payload_digest``
        even though the token strings differ (the nonce in each token's
        envelope ensures the byte sequence varies).

        No audit row or blob is written here: this path is only allowed
        when a prior ``mark_ready_for_review`` row exists for the current
        ``(session, state)`` and the row's snapshot blob still exists. If
        the state drifts, the owner must mark ready again so the gate and
        audit row run on the new share decision.
        """
        state_record = await self._session_service.get_current_state(session_id)
        if state_record is None:
            raise CompositionNotRunnableError(
                reason="state_missing",
                detail="No composition state exists for this session",
            )
        event = self._latest_mark_ready_event(session_id=session_id, state_id=state_record.id, user_id=user_id)
        if event is None:
            raise CompositionNotRunnableError(
                reason="completion_event_missing",
                detail="mark this composition ready for review before requesting a shareable link",
            )
        payload_digest = str(event.payload_digest)
        digest_hex = payload_digest.removeprefix(_DIGEST_PREFIX)
        if not self._payload_store.exists(digest_hex):
            raise CompositionNotRunnableError(
                reason="not_marked_ready",
                detail=_NOT_MARKED_READY_DETAIL,
            )
        lifetime = timedelta(seconds=self._settings.shareable_link_lifetime_seconds)
        created_at = datetime.now(UTC)
        expires_at = created_at + lifetime
        token = self._sign_token(
            session_id=session_id,
            state_id=state_record.id,
            payload_digest=payload_digest,
            created_by_user_id=user_id,
            created_at=created_at,
            expires_at=expires_at,
        )
        return ShareableLinkResponse(
            token=token,
            share_url=_SHARE_URL_PREFIX + token,
            expires_at=expires_at,
            state_id=str(state_record.id),
            payload_digest=payload_digest,
        )

    async def resolve_token(self, *, token: str, requesting_user_id: str) -> SharedInspectResponse:
        """Verify the token and return the frozen-at-mark-time snapshot.

        The audit_readiness is read directly from the blob — this method
        never calls ``ReadinessService.compute_snapshot``. Three reasons:
        audit-trail integrity (the reviewer sees what the owner saw at
        mark-time, not what readiness reports right now); content-addressing
        completeness (the payload_digest covers the readiness panel too);
        and permission-boundary collapse (resolve never calls the readiness
        service, so the reviewer-vs-owner permission question never arises
        at resolve time). ADR-022 §D4 documents the full reasoning.

        Raises:
            InvalidToken: signature mismatch, malformed envelope, or
                expired (per ``ShareTokenSigner.verify``).
            PayloadNotFoundError: token verifies but the payload-store
                blob has been reaped under the retention policy. The route
                layer maps this to HTTP 404 with a "ask the sender for a
                fresh link" message.
        """
        payload = self._signer.verify(token)
        digest_hex = payload.payload_digest.removeprefix(_DIGEST_PREFIX)
        blob_bytes = self._payload_store.retrieve(digest_hex)
        # The payload store has already verified the digest matches —
        # any tampering on the filesystem path raises IntegrityError
        # before we get here.
        blob_dict = self._parse_blob(blob_bytes)
        # ``model_validate_json`` (not ``model_validate``) on the
        # audit_readiness sub-tree because the strict-mode model rejects
        # ISO-string datetimes and list-as-tuple after a JSON round-trip.
        # ``model_validate_json`` activates Pydantic's JSON validators
        # which DO coerce wire-format primitives back to native types.
        audit_readiness = AuditReadinessSnapshot.model_validate_json(json.dumps(blob_dict["audit_readiness"]))
        return SharedInspectResponse(
            session_id=str(payload.session_id),
            state_id=str(payload.state_id),
            pipeline_metadata=blob_dict["pipeline_metadata"],
            composition_snapshot=blob_dict["composition_snapshot"],
            yaml=blob_dict["yaml"],
            audit_readiness=audit_readiness,
            created_by_user_id=blob_dict["created_by_user_id"],
            # ``created_at`` lives in the token envelope rather than the blob —
            # the blob is content-addressed and must not carry mint-time data.
            created_at=payload.created_at,
            expires_at=payload.expires_at,
        )

    # ── private helpers ─────────────────────────────────────────────

    def _latest_mark_ready_event(self, *, session_id: UUID, state_id: UUID, user_id: str) -> Any | None:
        """Return the latest committed mark-ready event for this exact state."""
        with self._sessions_db_engine.begin() as conn:
            return conn.execute(
                select(composer_completion_events_table)
                .where(
                    composer_completion_events_table.c.session_id == str(session_id),
                    composer_completion_events_table.c.composition_state_id == str(state_id),
                    composer_completion_events_table.c.event_type == "mark_ready_for_review",
                    composer_completion_events_table.c.actor == user_id,
                )
                .order_by(desc(composer_completion_events_table.c.created_at))
                .limit(1)
            ).first()

    def _sign_token(
        self,
        *,
        session_id: UUID,
        state_id: UUID,
        payload_digest: str,
        created_by_user_id: str,
        created_at: datetime,
        expires_at: datetime,
    ) -> str:
        return self._signer.sign(
            ShareTokenPayload(
                version=1,
                session_id=session_id,
                state_id=state_id,
                created_at=created_at,
                expires_at=expires_at,
                nonce_hex=secrets.token_hex(16),
                payload_digest=payload_digest,
                created_by_user_id=created_by_user_id,
            )
        )

    @staticmethod
    def _parse_blob(blob_bytes: bytes) -> _BlobShape:
        """Decode the canonical-JSON blob back to the typed wire shape.

        Tier-1 input: the blob came out of OUR payload store, integrity
        verified by digest. Any decode failure here is a corruption event.
        Returning the ``_BlobShape`` TypedDict gives downstream callers a
        typed surface; the ``cast`` reflects "we trust the canonicalisation
        round-trip" rather than re-validating each key.
        """
        return cast(_BlobShape, json.loads(blob_bytes.decode("utf-8")))
