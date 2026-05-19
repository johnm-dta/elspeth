"""Pydantic models for the tutorial-run endpoint."""

from __future__ import annotations

from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class TutorialRunRequest(BaseModel):
    """Request body for ``POST /api/tutorial/run``."""

    model_config = ConfigDict(extra="forbid")

    session_id: UUID
    prompt: str = Field(default="", max_length=65536)


class TutorialRunOutput(BaseModel):
    """Output preview returned by the tutorial run endpoint."""

    # ``frozen=True`` prevents post-construction mutation of attributes; ``rows``
    # is typed as ``tuple`` so the in-memory representation is itself immutable
    # (a list field on a frozen model would still permit ``model.rows.append(...)``).
    # Pydantic accepts a list at construction time and coerces to tuple.
    model_config = ConfigDict(strict=True, extra="forbid", frozen=True)

    rows: tuple[dict[str, Any], ...]
    source_data_hash: str


class TutorialRunResponse(BaseModel):
    """Response for ``POST /api/tutorial/run``."""

    model_config = ConfigDict(strict=True, extra="forbid", frozen=True)

    run_id: str
    output: TutorialRunOutput
    seeded_from_cache: bool
    cache_key: str | None


class TutorialOrphanCleanupResponse(BaseModel):
    """Response for tutorial orphan cleanup."""

    model_config = ConfigDict(strict=True, extra="forbid")

    deleted_count: int
