"""Flat-file tutorial-seed run cache. Absence = miss; corruption = crash.

Key: ``SHA-256(canonical_prompt + ":" + model_id)``.

``model_id`` is an opaque string supplied by the run path. Phase 4 uses a
compound identifier assembled by
``elspeth.web.composer.tutorial_service.tutorial_model_id``.

Invalidation envelope (what causes a different key, hence cache miss):

- ``settings.composer_model`` change.
- Core composer skill markdown (``pipeline_composer.md``) content change.
- Deployment skill overlay (``{data_dir}/skills/pipeline_composer.md``)
  content change.
- Staged guided skill pack (``base.md`` + ``step_1..step_4_wire.md``) content
  change.
- Recipe catalog content change (``composer/recipes.py`` or
  ``composer/guided/recipe_match.py``).

Out of scope (cache does NOT auto-invalidate on these — operator clears the
cache directory manually, consistent with the project's
"operator deletes the artifact" pattern documented elsewhere):

- LLM non-determinism. The composer LLM may pick a different transform model
  on a re-compose with identical inputs; the cache freezes whichever
  pipeline was produced first.
- Plugin pack defaults (``packs/llm/defaults.yaml``) or profile YAMLs that
  bias the composer's transform-model choice but do not change the key
  inputs above. The cached ``pipeline_yaml`` has the chosen model embedded,
  so the audit replay attribution is internally consistent — the operator
  just sees an older canonical experience until they clear
  ``{data_dir}/tutorial_cache/``.

This module does not interpret the ``model_id`` string; any single covered
input changing simply invalidates the cache. Uncovered inputs require
manual cache directory removal.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, ValidationError

# Kept BYTE-IDENTICAL to the frontend constant ``CANONICAL_TUTORIAL_PROMPT`` in
# ``src/elspeth/web/frontend/src/components/tutorial/tutorialMachine.ts``. The
# frontend posts that constant on the Turn-4 run; the cache only engages when the
# posted prompt equals this one. ``test_canonical_seed_matches_frontend_constant``
# enforces the equality so the two can never silently drift again.
CANONICAL_SEED_PROMPT = (
    "Scrape these three synthetic project-brief pages and, for each page, "
    "have an LLM write a short summary of the page. Remove the raw HTML and "
    "write the rows to a json file."
)


class TutorialCacheCorruptError(RuntimeError):
    """Raised when a present cache file is unreadable or self-inconsistent."""

    def __init__(self, path: Path, reason: str) -> None:
        super().__init__(f"tutorial cache file {path}: {reason}")
        self.path = path
        self.reason = reason


def _compute_key(canonical_prompt: str, model_id: str) -> str:
    """Hex SHA-256 of ``f"{canonical_prompt}:{model_id}"``."""
    h = hashlib.sha256()
    h.update(canonical_prompt.encode("utf-8"))
    h.update(b":")
    h.update(model_id.encode("utf-8"))
    return h.hexdigest()


def tutorial_cache_key(canonical_prompt: str, model_id: str) -> str:
    """Public cache-key helper for provenance surfaces."""
    return _compute_key(canonical_prompt, model_id)


class TutorialCacheEntry(BaseModel):
    """Cached deterministic output of a canonical-seed run.

    This model stores reusable output content only. It intentionally rejects
    run/session/user identity fields from the cache-seeding run; replaying a
    cache hit must create fresh audit identity owned by the current session.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    canonical_prompt: str
    model_id: str
    cached_at: datetime
    rows: list[dict[str, Any]]
    source_data_hash: str
    llm_call_count: int
    pipeline_yaml: str


class TutorialCache:
    """Flat-file cache for canonical-seed run outputs."""

    def __init__(self, *, cache_dir: Path) -> None:
        self._dir = cache_dir

    def lookup(self, canonical_prompt: str, model_id: str) -> TutorialCacheEntry | None:
        """Return cached entry, or None on miss. Crashes on corruption."""
        key = tutorial_cache_key(canonical_prompt, model_id)
        path = self._dir / f"{key}.json"
        if not path.exists():
            return None
        try:
            raw = path.read_text(encoding="utf-8")
        except OSError as exc:
            raise TutorialCacheCorruptError(path, "unreadable") from exc
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise TutorialCacheCorruptError(path, "not valid JSON") from exc
        try:
            entry = TutorialCacheEntry.model_validate(data)
        except ValidationError as exc:
            raise TutorialCacheCorruptError(path, "does not match expected shape") from exc
        if entry.canonical_prompt != canonical_prompt or entry.model_id != model_id:
            raise TutorialCacheCorruptError(
                path,
                f"prompt mismatch: file recorded "
                f"({entry.canonical_prompt!r}, {entry.model_id!r}) "
                f"but lookup was for ({canonical_prompt!r}, {model_id!r})",
            )
        return entry

    def store(self, entry: TutorialCacheEntry) -> None:
        """Persist the entry atomically with tempfile + ``os.replace``."""
        self._dir.mkdir(parents=True, exist_ok=True)
        key = tutorial_cache_key(entry.canonical_prompt, entry.model_id)
        final_path = self._dir / f"{key}.json"
        fd, tmp_path_str = tempfile.mkstemp(
            prefix=f"{key}.",
            suffix=".json.tmp",
            dir=str(self._dir),
        )
        tmp_path = Path(tmp_path_str)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(entry.model_dump_json())
            os.replace(tmp_path, final_path)
        except Exception:
            tmp_path.unlink(missing_ok=True)
            raise
