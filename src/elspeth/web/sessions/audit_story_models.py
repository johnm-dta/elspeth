"""Pydantic response model for run audit-story projections."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict


class RunAuditStoryResponse(BaseModel):
    """Audit-story projection for a completed run."""

    model_config = ConfigDict(strict=True, extra="forbid")

    run_id: str
    session_id: str
    llm_call_count: int
    output_file_hash: str
    started_at: datetime
    plugin_versions: dict[str, str]
    seeded_from_cache: bool
    cache_key: str | None
