"""Pydantic models for the tutorial-run endpoint."""

from __future__ import annotations

from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class TutorialRunRequest(BaseModel):
    """Request body for ``POST /api/tutorial/run``."""

    model_config = ConfigDict(extra="forbid")

    session_id: UUID


class TutorialRunOutput(BaseModel):
    """Output preview returned by the tutorial run endpoint."""

    # ``frozen=True`` prevents post-construction mutation of attributes; ``rows``
    # is typed as ``tuple`` so the in-memory representation is itself immutable
    # (a list field on a frozen model would still permit ``model.rows.append(...)``).
    # Pydantic accepts a list at construction time and coerces to tuple.
    model_config = ConfigDict(strict=True, extra="forbid", frozen=True)

    rows: tuple[dict[str, Any], ...]
    source_data_hash: str
    # Count of rows the source DISCARDED at the boundary (on_validation_failure=
    # "discard"): recorded in the Landscape validation_errors trail but absent from
    # ``rows``. Surfaced so the UX can show "N rows dropped at source" rather than
    # silently presenting only the survivors (the "5 requested, 2 arrived" gap).
    # Quarantined-to-a-sink rows are NOT counted here — those have a visible
    # destination. Defaults to 0 (a clean run with nothing discarded).
    discarded_row_count: int = Field(default=0, ge=0)


class TutorialRunResponse(BaseModel):
    """Response for ``POST /api/tutorial/run``."""

    model_config = ConfigDict(strict=True, extra="forbid", frozen=True)

    run_id: str
    output: TutorialRunOutput


class TutorialCancelRequest(BaseModel):
    """Request body for ``POST /api/tutorial/cancel``."""

    model_config = ConfigDict(extra="forbid")

    session_id: UUID


class TutorialCancelResponse(BaseModel):
    """Response for ``POST /api/tutorial/cancel``.

    ``cancelled=True`` when the session's active run was cancelled;
    ``cancelled=False`` when no active run exists. The endpoint is
    idempotent — the no-active-run case is never an error.
    """

    model_config = ConfigDict(strict=True, extra="forbid", frozen=True)

    cancelled: bool


class TutorialOrphanCleanupResponse(BaseModel):
    """Response for tutorial orphan cleanup."""

    model_config = ConfigDict(strict=True, extra="forbid")

    deleted_count: int
