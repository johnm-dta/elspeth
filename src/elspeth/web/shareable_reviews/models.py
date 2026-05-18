"""Pydantic response models for the shareable-reviews endpoints.

Phase 6A Task 4 (UX redesign 2026-05). All three models inherit
``_StrictResponse`` from ``web/execution/schemas.py`` so the Tier-1 wire
discipline (``strict=True, extra="forbid"``) applies:

* Field types must match exactly — ``"7"`` into an ``int`` is rejected.
* Unexpected fields in the constructed model crash rather than being silently
  dropped — drift between producer and consumer fails at construction.

The ``SharedInspectResponse`` re-uses ``AuditReadinessSnapshot`` verbatim so
the shared inspect view shows the same six-row readiness panel the owner
sees in the composer. Phase 2 / Phase 18 own that model; Phase 6 is a
consumer, not a co-owner.

Layer: L3 (web application).
"""

from __future__ import annotations

from datetime import datetime

from elspeth.web.audit_readiness.models import AuditReadinessSnapshot
from elspeth.web.execution.schemas import _StrictResponse
from elspeth.web.sessions.schemas import CompositionObject


class MarkReadyForReviewResponse(_StrictResponse):
    """Returned by ``POST /api/sessions/{session_id}/mark-ready-for-review``.

    Carries the freshly-minted signed token, the URL the operator can copy
    out of the UI, the absolute expiry, and the content-address of the
    snapshot blob the token resolves to. ``payload_digest`` is exposed on
    the wire so the operator UI can correlate "the share URL I just got"
    against "the audit row recorded server-side" without an extra round
    trip.
    """

    token: str
    share_url: str
    expires_at: datetime
    payload_digest: str


class ShareableLinkResponse(_StrictResponse):
    """Returned by ``GET /api/sessions/{session_id}/shareable-link``.

    Re-mints a fresh token for the current ``(session, state)`` snapshot on
    every call. Two calls in succession produce two different tokens (the
    nonce differs) but identical ``payload_digest`` (content-addressing of
    an unchanged snapshot).

    The ``state_id`` is surfaced because the underlying composition may
    have advanced since the last call; the client uses this to confirm
    which version of the pipeline the link inspects.
    """

    token: str
    share_url: str
    expires_at: datetime
    state_id: str
    payload_digest: str


class SharedInspectResponse(_StrictResponse):
    """Returned by ``GET /api/sessions/shared/{token}``.

    The read-only inspect view for a recipient. ``yaml`` is the rendered
    pipeline YAML (regenerated from the snapshot, not stored verbatim, so
    a YAML-generator improvement automatically flows through). The
    ``audit_readiness`` field is the same Phase-2 / Phase-18 snapshot the
    owner sees — including the ``llm_interpretations`` row added in Phase
    18 (see plan 19a §"Post-Phase-18 merge fact"). No extra aggregation
    happens in Phase 6.

    ``created_by_user_id`` is surfaced so the reviewer can see who shared
    the pipeline. ``expires_at`` lets the frontend show "this link expires
    in N days" without re-decoding the token.
    """

    session_id: str
    state_id: str
    pipeline_metadata: CompositionObject
    composition_snapshot: CompositionObject
    yaml: str
    audit_readiness: AuditReadinessSnapshot
    created_by_user_id: str
    created_at: datetime
    expires_at: datetime
