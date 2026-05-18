"""Shareable-review tokens, snapshot blobs, and inspect routes.

Phase 6A (UX redesign 2026-05) — completion-gesture "Save for review" verb.

The package exposes:

* ``ShareTokenSigner`` / ``ShareTokenPayload`` / ``InvalidToken`` — the HMAC
  primitive backing self-verifying capability tokens.
* ``ShareableReviewService`` — the orchestrator that builds a snapshot blob,
  records the audit event, and mints the token.
* ``create_shareable_reviews_router`` — the three new FastAPI routes
  (POST mark-ready-for-review, GET shareable-link, GET sessions/shared/{token}).

Module-layer reminder: all of `elspeth.web` is L3 (application layer) per the
ELSPETH layer model — imports L0/L1/L2 freely, must not be imported by anything
in those lower layers.
"""

from __future__ import annotations

from elspeth.web.shareable_reviews.routes import create_shareable_reviews_router
from elspeth.web.shareable_reviews.service import (
    CompositionNotRunnableError,
    ShareableReviewService,
)
from elspeth.web.shareable_reviews.signer import (
    InvalidToken,
    ShareTokenPayload,
    ShareTokenSigner,
)

__all__ = [
    "CompositionNotRunnableError",
    "InvalidToken",
    "ShareTokenPayload",
    "ShareTokenSigner",
    "ShareableReviewService",
    "create_shareable_reviews_router",
]
