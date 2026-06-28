# Phase 4 Backend Plan — Part 2 (Endpoints + Telemetry)

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development or superpowers:executing-plans.

**Status:** 2026-05-19. Part 2 of the Phase 4 backend plan, split from `21a-phase-4-backend.md` once the file exceeded 4,900 lines.

**Goal:** Ship the tutorial-run endpoint, the audit-story endpoint, the frontend API client wiring, and the launch-critical telemetry counters. Builds on Part 1's infrastructure (preferences contract, cache module, run-path integration, Landscape write repository).

**Prerequisite:** [21a1-phase-4-backend-part-1.md](21a1-phase-4-backend-part-1.md) must be implemented first — Part 2 depends on Part 1's Tasks 6 (cache wiring), 7 (run-path replay), 7.0 (`LandscapeWriteRepository` + `Run` dataclass extensions), and on the cross-plan contract block declared in Part 1.

**Sibling plans:**

- [21-phase-4-hello-world-tutorial.md](21-phase-4-hello-world-tutorial.md) — overview, scope, risks.
- [21a1-phase-4-backend-part-1.md](21a1-phase-4-backend-part-1.md) — backend infrastructure.
- [21b1-phase-4-frontend-part-1.md](21b1-phase-4-frontend-part-1.md) — frontend store, App.tsx, Turns 1–2b.
- [21b2-phase-4-frontend-part-2.md](21b2-phase-4-frontend-part-2.md) — frontend Turns 3–6, finalisation, smoke.

**PR mapping:** This plan ships as part of **PR-21a** alongside Part 1 (21a1 + 21a2 are co-dependent and land in a single PR). PR-21a must merge to `RC5.2` **before** PR-21b (the frontend half from 21b1 + 21b2) — see overview §"PR strategy" for rationale.

**Tech stack:** SQLAlchemy Core, FastAPI, Pydantic v2, pytest, OpenTelemetry.

---

## Task 7.1: Route + service — `POST /api/tutorial/run`

> **Architecture finding M-R2-1 (2026-05-19) — no lazy imports.** The
> original draft had `tutorial_run_routes.py` define
> `TutorialRunRequest/Output/Response`, and `tutorial_service.py` import
> them at module load; the route module then lazy-imported the service
> inside the handler body to break the resulting circular import. That
> is forbidden by CLAUDE.md ("Shifting the Burden" — lazy imports defer
> structural fixes and the pattern recurs).
>
> The structural fix: extract the three Pydantic models to a new module
> `src/elspeth/web/composer/tutorial_models.py`. The route imports from
> `tutorial_models` AND `tutorial_service` at module top; the service
> imports from `tutorial_models` only. No circular dependency, no lazy
> imports anywhere in the call graph. The same shape applies to the
> `from elspeth.web.composer.run_path import execute_pipeline_run`
> import inside the service body — promote it to a module-top import
> (the run-path module imports from `core/landscape` only, not from
> the composer; no cycle).

**Files:**

- Create: `src/elspeth/web/composer/tutorial_run_routes.py` — FastAPI router.
- Create: `src/elspeth/web/composer/tutorial_service.py` — service method.
- Create: `tests/integration/web/test_tutorial_routes.py` — integration tests.
- Modify: `src/elspeth/web/composer/__init__.py` — export the router.
- Modify: the FastAPI app-composition site (identified in Task 6 recon) — `include_router(create_tutorial_run_router())`.

Task 7 wires the cache-consult branch into the *generic* run path (so any
canonical-seed run benefits, including ones initiated from the existing
`POST /api/sessions/{id}/runs` endpoint). Task 7.1 adds the **tutorial-specific
entry point** that the frontend's `runTutorialPipeline` (21b2 Task 8) calls.
The route is a thin façade over the run-path orchestration Task 7 already
wired: it accepts a `(session_id, prompt)` pair, derives the canonical
pipeline from the prompt, invokes the run-path, and returns the
`TutorialRunResponse` shape the frontend consumes.

The route does **not** duplicate Task 7's cache logic — it calls a service
method (`run_tutorial_pipeline`) which delegates to the same
`execute_pipeline_run` (or recon-confirmed equivalent) entry point that
Task 7 modified. The cache-consult branch fires inside that entry point;
Task 7.1 simply guarantees the frontend has a stable, narrow surface.

**Bypass paths (live run, no cache consult, no cache write):**

1. User's `tutorial_completed_at IS NOT NULL` — post-completion users get
   live runs (they're past the tutorial; cache hits would be misleading).
2. User's `default_mode == 'freeform'` — freeform users skipped the tutorial
   by choice; the cache is a tutorial-only optimisation (per Q11). A
   freeform user who *does* invoke `POST /api/tutorial/run` (e.g. via the
   Phase 8 retake button) gets a live run.

Both bypass paths are evaluated **before** any cache lookup. The bypass
decision is logged to the Landscape entry's metadata as
`tutorial_cache_bypass_reason: "completed" | "freeform" | None` so an
auditor can later distinguish a bypass from a miss.

- [ ] **Step 1: Reconnaissance — confirm the run-path service surface.**

```bash
grep -n "def execute_pipeline_run\|def _execute_pipeline_live\|create_run_router" \
  src/elspeth/web/sessions/*.py src/elspeth/web/composer/*.py 2>/dev/null
```

Confirm:

1. The exact name of the function Task 7 modified (its argument shape
   determines `run_tutorial_pipeline`'s pass-through call).
2. The auth dep used by sibling composer routes (`get_current_user` or a
   composer-specific dep) — match it.
3. The existing FastAPI app-composition site where the new router is
   `include_router`'d — same file as `create_preferences_router()` is
   registered.
4. The Pydantic config base (project convention is `ConfigDict(strict=True,
   extra='forbid')` — confirm against an existing model such as
   `UpdateSessionRequest` in `src/elspeth/web/sessions/schemas.py`).
5. **Session-ownership verification is a shared free function**:
   `verify_session_ownership(session_id, user, request)` in
   `src/elspeth/web/sessions/ownership.py`. It raises
   `HTTPException(404)` on any access-control failure (unknown session,
   wrong user, wrong auth provider) — **the IDOR contract is 404, not
   403**, deliberately, to avoid leaking session existence to a UUID
   enumerator. Do NOT reinvent this check inline; do NOT add a
   service-method shim such as `session_exists_for_user`. Reuse the
   free function. Confirm by reading `sessions/ownership.py:26-50`.

Write the findings into the commit message body. Do not guess.

- [ ] **Step 2: Write the failing integration test.**

Create `tests/integration/web/test_tutorial_routes.py`:

```python
"""Integration tests for POST /api/tutorial/run."""

from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy.pool import StaticPool

from elspeth.web.auth.middleware import get_current_user
from elspeth.web.auth.models import UserIdentity
from elspeth.web.composer.tutorial_run_routes import create_tutorial_run_router
from elspeth.web.preferences.routes import create_preferences_router
from elspeth.web.preferences.service import PreferencesService
from elspeth.web.preferences.tutorial_cache import (
    CANONICAL_SEED_PROMPT,
    TutorialCache,
    TutorialCacheEntry,
)
from elspeth.web.sessions.engine import create_session_engine
from elspeth.web.sessions.schema import initialize_session_schema


@pytest.fixture
def cache_dir(tmp_path: Path) -> Path:
    d = tmp_path / "cache"
    d.mkdir()
    return d


@pytest.fixture
def app(cache_dir: Path) -> FastAPI:
    engine = create_session_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    initialize_session_schema(engine)
    app = FastAPI()
    app.state.session_engine = engine
    app.state.preferences_service = PreferencesService(engine)
    app.state.tutorial_cache = TutorialCache(cache_dir=cache_dir)

    identity = UserIdentity(user_id="alice", username="alice")

    async def _mock_user() -> UserIdentity:
        return identity

    app.dependency_overrides[get_current_user] = _mock_user
    app.include_router(create_preferences_router())
    app.include_router(create_tutorial_run_router())
    return app


def _seed_canonical_cache_entry(app: FastAPI) -> None:
    app.state.tutorial_cache.store(
        TutorialCacheEntry(
            canonical_prompt=CANONICAL_SEED_PROMPT,
            model_id="claude-opus-4-7",
            cached_at=datetime(2026, 5, 15, tzinfo=UTC),
            rows=[{"url": "ato.gov.au", "score": 5, "rationale": "clear"}],
            source_data_hash="a7f3e2cached",
            llm_call_count=5,
            pipeline_yaml="<canonical>",
        )
    )


def _create_session(app: FastAPI, user_id: str = "alice") -> str:
    """Create a session row and return its id.

    Reuses the project's established integration-test fixture pattern from
    ``tests/integration/web/conftest.py`` (see ``_make_session``): the
    fixture inserts directly into ``sessions_table`` via the session
    engine. There is no ``create_session_for_user`` free function — the
    public production surface is the ``SessionsServiceImpl.create_session``
    coroutine, but tests bypass it to set up arbitrary ownership.
    Implementer: import ``_make_session`` from the existing conftest, or
    inline the same INSERT pattern here. Confirm during Step 1 recon.
    """
    raise NotImplementedError("use _make_session from tests/integration/web/conftest.py")


def test_post_run_cache_hit_returns_current_session_run_id(
    app: FastAPI, cache_dir: Path
) -> None:
    """Cache hit: response.run_id is owned by the current session; seeded_from_cache=True."""
    _seed_canonical_cache_entry(app)
    session_id = _create_session(app)
    client = TestClient(app)

    response = client.post(
        "/api/tutorial/run",
        json={"session_id": session_id, "prompt": CANONICAL_SEED_PROMPT},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["seeded_from_cache"] is True
    assert isinstance(body["cache_key"], str) and len(body["cache_key"]) == 64
    # The run_id is a fresh identifier owned by the caller's session
    # (cross-validated by querying the session's runs).
    assert isinstance(body["run_id"], str) and len(body["run_id"]) > 0
    # The output's source_data_hash matches the cached entry — content,
    # not identity, was replayed.
    assert body["output"]["source_data_hash"] == "a7f3e2cached"


def test_post_run_cache_miss_executes_fresh(
    app: FastAPI, cache_dir: Path
) -> None:
    """Cache miss: live run, response.seeded_from_cache=False, cache populated post-run."""
    session_id = _create_session(app)
    client = TestClient(app)

    response = client.post(
        "/api/tutorial/run",
        json={"session_id": session_id, "prompt": CANONICAL_SEED_PROMPT},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["seeded_from_cache"] is False
    assert body["cache_key"] is None
    # The cache was written post-run (one file in the cache directory).
    assert len(list(cache_dir.iterdir())) == 1


def test_post_run_bypasses_cache_for_freeform_user(
    app: FastAPI, cache_dir: Path
) -> None:
    """default_mode=='freeform' → cache bypassed even if a hit would be available."""
    _seed_canonical_cache_entry(app)
    session_id = _create_session(app)
    client = TestClient(app)
    # Set the user's mode to freeform.
    client.patch("/api/composer-preferences", json={"default_mode": "freeform"})

    # Snapshot cache directory before run; after the bypass the directory
    # must still hold ONLY the pre-seeded canonical entry (no new file).
    pre_run_files = {p.name for p in cache_dir.iterdir()}
    with patch(
        "elspeth.web.composer.tutorial_service.TutorialCache.lookup"
    ) as mock_lookup:
        response = client.post(
            "/api/tutorial/run",
            json={"session_id": session_id, "prompt": CANONICAL_SEED_PROMPT},
        )
        assert response.status_code == 200
        # The cache lookup was bypassed — never called.
        mock_lookup.assert_not_called()
    body = response.json()
    assert body["seeded_from_cache"] is False
    # Bypass contract: no cache write either (otherwise a bypass-mode run could
    # poison the cache for non-bypass users).
    post_run_files = {p.name for p in cache_dir.iterdir()}
    assert post_run_files == pre_run_files, (
        f"Bypass path must not write the cache; "
        f"new files: {post_run_files - pre_run_files}"
    )


def test_post_run_bypasses_cache_for_completed_user(
    app: FastAPI, cache_dir: Path
) -> None:
    """tutorial_completed_at IS NOT NULL → cache bypassed (post-completion live runs)."""
    _seed_canonical_cache_entry(app)
    session_id = _create_session(app)
    client = TestClient(app)
    client.patch(
        "/api/composer-preferences",
        json={"tutorial_completed_at": "2026-05-14T00:00:00Z"},
    )

    pre_run_files = {p.name for p in cache_dir.iterdir()}
    with patch(
        "elspeth.web.composer.tutorial_service.TutorialCache.lookup"
    ) as mock_lookup:
        response = client.post(
            "/api/tutorial/run",
            json={"session_id": session_id, "prompt": CANONICAL_SEED_PROMPT},
        )
        assert response.status_code == 200
        mock_lookup.assert_not_called()
    body = response.json()
    assert body["seeded_from_cache"] is False
    # Bypass contract: no cache write either (otherwise a bypass-mode run could
    # poison the cache for non-bypass users).
    post_run_files = {p.name for p in cache_dir.iterdir()}
    assert post_run_files == pre_run_files, (
        f"Bypass path must not write the cache; "
        f"new files: {post_run_files - pre_run_files}"
    )


def test_post_run_unknown_session_returns_404(
    app: FastAPI, cache_dir: Path
) -> None:
    """Unknown session → 404, raised by the shared ownership helper.

    Per the live IDOR contract (src/elspeth/web/sessions/ownership.py:33),
    verify_session_ownership raises HTTPException(404) for any
    access-control failure — unknown session, wrong user, or wrong auth
    provider. The 404 is deliberate (not 403) to avoid leaking session
    existence to an attacker enumerating UUIDs.
    """
    client = TestClient(app)
    response = client.post(
        "/api/tutorial/run",
        json={
            "session_id": "00000000-0000-0000-0000-000000000000",
            "prompt": CANONICAL_SEED_PROMPT,
        },
    )
    assert response.status_code == 404


def test_post_run_corrupt_preferences_returns_500(
    app: FastAPI, cache_dir: Path
) -> None:
    """Tier-1: corrupt preferences row → CorruptPreferencesError → 500."""
    session_id = _create_session(app)
    # Write a corrupt tutorial_completed_at directly via the engine (bypassing Pydantic).
    from sqlalchemy import text
    with app.state.session_engine.begin() as conn:
        conn.execute(
            text(
                "INSERT INTO user_preferences_table (user_id, tutorial_completed_at) "
                "VALUES ('alice', 'not-a-datetime')"
            )
        )
    client = TestClient(app)
    response = client.post(
        "/api/tutorial/run",
        json={"session_id": session_id, "prompt": CANONICAL_SEED_PROMPT},
    )
    assert response.status_code == 500


def test_post_run_request_extra_field_rejected(
    app: FastAPI, cache_dir: Path
) -> None:
    """ConfigDict(extra='forbid') invariant: unknown fields in the body → 422."""
    session_id = _create_session(app)
    client = TestClient(app)
    response = client.post(
        "/api/tutorial/run",
        json={
            "session_id": session_id,
            "prompt": CANONICAL_SEED_PROMPT,
            "rogue_field": "should-reject",
        },
    )
    assert response.status_code == 422
```

Placeholders (`_create_session`, the corrupt-row INSERT shape) are filled in
from Step 1's recon; do not guess. The test file may not be valid Python
until then — intentional: TDD must fail against real code, not mocks.

- [ ] **Step 3: Run test to verify it fails.**

```bash
.venv/bin/python -m pytest tests/integration/web/test_tutorial_routes.py -v
```

Expected: FAIL — module-import error (router not yet created) or
behavioural failure.

- [ ] **Step 4: Implement the Pydantic models, route, and service.**

Create `src/elspeth/web/composer/tutorial_run_routes.py`:

```python
"""Tutorial run endpoint — POST /api/tutorial/run.

Phase 4A.7.1. The frontend's runTutorialPipeline (21b2 Task 8) calls this
route. Logic is delegated to TutorialRunService; the route is a thin
FastAPI shell that handles request parsing, auth, and response shaping.

Per CLAUDE.md no-defensive-programming: no try/except wrapping. Errors
from the service propagate to FastAPI's exception handlers (CorruptPreferencesError
maps to 500 via the global handler installed in app composition; HTTPException
short-circuits FastAPI; everything else surfaces as the framework default 500).
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, Request

from elspeth.web.auth.middleware import get_current_user
from elspeth.web.auth.models import UserIdentity
from elspeth.web.composer.tutorial_models import (
    TutorialRunRequest,
    TutorialRunResponse,
)
# Module-top import — no circular dependency because the service imports
# from `tutorial_models` (a leaf module), not from `tutorial_run_routes`
# (M-R2-1, 2026-05-19).
from elspeth.web.composer.tutorial_service import run_tutorial_pipeline


def create_tutorial_run_router() -> APIRouter:
    router = APIRouter(prefix="/api/tutorial", tags=["tutorial"])

    @router.post("/run", response_model=TutorialRunResponse)
    async def run_tutorial(
        request: Request,
        body: TutorialRunRequest,
        user: UserIdentity = Depends(get_current_user),
    ) -> TutorialRunResponse:
        # No try/except wrapping: the service calls verify_session_ownership
        # which raises HTTPException(404) directly on any access-control
        # failure (the IDOR contract; see
        # src/elspeth/web/sessions/ownership.py:33). HTTPException
        # short-circuits FastAPI's response pipeline; CorruptPreferencesError
        # propagates to the global 500 handler installed in app composition.
        return await run_tutorial_pipeline(
            request=request,
            user=user,
            session_id=body.session_id,
            prompt=body.prompt,
        )

    return router
```

Create `src/elspeth/web/composer/tutorial_models.py` (the leaf module
extracted per M-R2-1, 2026-05-19 — both the route and the service
import these models from here, so neither depends on the other):

```python
"""Pydantic models for the tutorial-run endpoint.

This is a leaf module — depends only on `pydantic`. The route module
(`tutorial_run_routes.py`) and the service module (`tutorial_service.py`)
both import from here, so neither has to import the other. Resolves the
circular-import pattern that the original draft worked around with lazy
imports (CLAUDE.md "Shifting the Burden" — lazy imports defer structural
fixes; they are forbidden).
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict


class TutorialRunRequest(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")

    session_id: str
    prompt: str


class TutorialRunOutput(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")

    rows: list[dict[str, Any]]
    source_data_hash: str


class TutorialRunResponse(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")

    run_id: str
    output: TutorialRunOutput
    seeded_from_cache: bool
    cache_key: str | None
```

Create `src/elspeth/web/composer/tutorial_service.py`:

```python
"""Tutorial run service — orchestrates cache consult, bypass, and live run.

Phase 4A.7.1. Layered above Task 7's `execute_pipeline_run` so the
cache-consult branch lives in exactly one place (Task 7's modified run
path). This service adds the **bypass** logic (completed-user, freeform-user)
that is specific to the tutorial entry point.

Tier model:
- session_id, prompt: Tier 3 (untrusted) — the composer LLM `set_pipeline`
  path already handles Tier 3 prompts; this service forwards.
- preferences row: Tier 1 — `CorruptPreferencesError` propagates.
- cache content: server-generated — corruption crashes (Task 5).
"""

from __future__ import annotations

from uuid import UUID

from fastapi import Request

from elspeth.web.auth.models import UserIdentity
from elspeth.web.composer.run_path import execute_pipeline_run
from elspeth.web.composer.tutorial_models import (
    TutorialRunOutput,
    TutorialRunResponse,
)
from elspeth.web.sessions.ownership import verify_session_ownership


async def run_tutorial_pipeline(
    *,
    request: Request,
    user: UserIdentity,
    session_id: str,
    prompt: str,
) -> TutorialRunResponse:
    """Execute the tutorial pipeline for ``user`` against ``session_id``.

    Decision order:

    1. Validate session ownership via the shared
       ``verify_session_ownership`` free function. Per the live IDOR
       contract (``src/elspeth/web/sessions/ownership.py:33``), any
       access-control failure — unknown session, wrong user, wrong auth
       provider — raises ``HTTPException(404)`` directly. The 404 (not
       403) is deliberate: distinguishing "no such session" from "you
       can't access this session" would leak session existence to a UUID
       enumerator. This service does NOT catch and re-raise; the
       HTTPException propagates to FastAPI's exception handlers.
    2. Read the user's preferences. Tier-1 corruption → CorruptPreferencesError
       (propagates to the 500 handler).
    3. Compute bypass reason:
       - prefs.tutorial_completed_at IS NOT NULL  → bypass ("completed")
       - prefs.default_mode == "freeform"         → bypass ("freeform")
       - otherwise                                → consult cache
    4. Build the canonical pipeline from ``prompt`` (the composer LLM
       set_pipeline path handles Tier-3 prompts; result is a pipeline_state
       dict matching Task 7's `_is_canonical_seed_pipeline` contract).
    5. Invoke Task 7's `execute_pipeline_run` — that function already
       contains the cache-consult / cache-store branches gated on
       prefs.tutorial_completed_at being None. On bypass paths we pass a
       pre-built pipeline_state whose mode-flag forces the live path.

    Per CLAUDE.md offensive-programming:
    - No `.get()` / `getattr(default)` against pipeline_state or prefs.
    - Bypass-reason is computed offensively; an unrecognised mode crashes
      (the existing `_VALID_MODES` guard in preferences/service.py catches
      this upstream).
    """
    # Reuse the shared IDOR-safe ownership check. Raises HTTPException(404)
    # on mismatch (no separate SessionNotFoundError indirection needed —
    # the shared helper already raises the framework-native error).
    await verify_session_ownership(
        session_id=UUID(session_id),
        user=user,
        request=request,
    )

    prefs_service = request.app.state.preferences_service
    prefs = await prefs_service.get_composer_preferences(user.user_id)

    bypass_reason: str | None = None
    if prefs.tutorial_completed_at is not None:
        bypass_reason = "completed"
    elif prefs.default_mode == "freeform":
        bypass_reason = "freeform"

    pipeline_state = _build_canonical_pipeline_from_prompt(
        prompt=prompt,
        force_live=bypass_reason is not None,
    )

    # Task 7's execute_pipeline_run owns the cache-consult/cache-store
    # branches. When force_live is set, _is_canonical_seed_pipeline returns
    # False (the pipeline_state carries a flag the helper checks) and the
    # cache is bypassed. `execute_pipeline_run` is imported at module top
    # (M-R2-1, 2026-05-19) — no lazy import.
    result = await execute_pipeline_run(
        request=request,
        user=user,
        session_id=session_id,
        pipeline_state=pipeline_state,
    )

    # Tier-1: every field below is read from a Landscape entry the run-path
    # just wrote (or replayed). Absence is corruption.
    return TutorialRunResponse(
        run_id=result["run_id"],
        output=TutorialRunOutput(
            rows=result["rows"],
            source_data_hash=result["source_data_hash"],
        ),
        seeded_from_cache=result["seeded_from_cache"],
        cache_key=result["cache_key"],
    )


def _build_canonical_pipeline_from_prompt(
    *, prompt: str, force_live: bool
) -> dict[str, object]:
    """Build a pipeline_state dict from the (possibly edited) seed prompt.

    Exact construction depends on Step 1 recon — likely a thin wrapper
    around the existing composer set_pipeline path. The `force_live` flag
    is attached as `pipeline_state["_tutorial_force_live"] = True`; Task 7's
    `_is_canonical_seed_pipeline` reads this flag and returns False when set
    (which bypasses both the cache-consult and the cache-store branches).
    """
    raise NotImplementedError("filled in from Step 1 recon")
```

Modify the FastAPI app-composition site (identified in Step 1 recon) to:

```python
from elspeth.web.composer.tutorial_run_routes import create_tutorial_run_router
app.include_router(create_tutorial_run_router())
```

- [ ] **Step 5: Update Task 7's `_is_canonical_seed_pipeline` to honour the force-live flag.**

In the run-path file Task 7 modified, extend `_is_canonical_seed_pipeline`:

```python
def _is_canonical_seed_pipeline(pipeline_state: Mapping[str, object]) -> bool:
    # Force-live escape hatch (Task 7.1): bypass paths short-circuit the canonical
    # check so cache_consult AND cache_write are both disabled. Use explicit
    # membership + identity check rather than `.get()` to keep the
    # optional-key contract visible (per CLAUDE.md offensive-programming —
    # `.get(...)` is the defensive-default form we avoid in production code).
    if "_tutorial_force_live" in pipeline_state and pipeline_state["_tutorial_force_live"] is True:
        return False
    # ... existing canonical-seed shape checks (recon) ...
```

This is the *only* call-site change Task 7.1 makes outside the new files.

- [ ] **Step 6: Run test to verify it passes.**

```bash
.venv/bin/python -m pytest tests/integration/web/test_tutorial_routes.py -v
```

Expected: PASS — all seven tests green.

- [ ] **Step 7: Run the full integration suite.**

```bash
.venv/bin/python -m pytest tests/integration/web/ -v
```

Expected: PASS — Task 7's `test_non_tutorial_user_skips_cache` and
`test_edited_prompt_skips_cache` still green; the new bypass paths do not
collide with their gates.

- [ ] **Step 8: Commit.**

```bash
git add src/elspeth/web/composer/tutorial_run_routes.py \
        src/elspeth/web/composer/tutorial_service.py \
        src/elspeth/web/composer/__init__.py \
        tests/integration/web/test_tutorial_routes.py \
        <app-composition-site> <run-path-file>
git commit -m "feat(web): POST /api/tutorial/run with completed/freeform bypass (Phase 4A.7.1)"
```

---

## Task 7.2: Route + service — `GET /api/sessions/{session_id}/runs/{run_id}/audit-story`

**Files:**

- Create: `src/elspeth/web/sessions/audit_story_models.py` — Pydantic
  response model (leaf module — neither the route nor the service
  imports the other, R2-8, 2026-05-19).
- Create: `src/elspeth/web/sessions/audit_story_service.py` — service method.
- Modify: `src/elspeth/web/sessions/routes.py` — add the new route handler.
- Create: `tests/integration/web/test_audit_story_routes.py` — integration tests.

> **Reality finding R2-8 (2026-05-19) — no lazy imports.** The original
> draft placed `RunAuditStoryResponse` in `sessions/schemas.py` and had
> the service import it at module load; the route module then
> lazy-imported the service inside the handler body to break the
> resulting circular import. Same anti-pattern as M-R2-1 (CLAUDE.md
> "Shifting the Burden").
>
> Structural fix: extract `RunAuditStoryResponse` to a new module
> `src/elspeth/web/sessions/audit_story_models.py`. The service imports
> the model from there; the route imports the model AND the service at
> module top. `sessions/schemas.py` is unchanged — `RunAuditStoryResponse`
> no longer lives there. The error class `CorruptAuditRowError` stays
> in the service module (it has no consumers outside the service +
> route, and the route imports the service module already).

This endpoint surfaces the **real Landscape audit row** for a specific
`(session_id, run_id)` pair so the frontend's Turn 5 (21b2 Task 9) can
render a load-bearing audit narrative against the user's own run. The
response is derived **entirely from real audit data**. **No field is ever
synthesised, defaulted, or inferred** — if a field is absent from the
audit row, that absence is a Tier-1 corruption signal and the service
raises `CorruptAuditRowError`, which propagates to 500. This is the Q6
no-synthesis invariant: the audit-story endpoint must not lie about what
the audit trail recorded, even by omission.

**Authorization order:**

1. Caller authenticated (route dep).
2. Caller owns `session_id` (session-ownership check, via
   `verify_session_ownership` in `src/elspeth/web/sessions/ownership.py`).
   If not → **404**. The IDOR contract (`sessions/ownership.py:33`) is 404
   on every access-control failure (unknown session, wrong user, wrong
   auth provider) — deliberately, to avoid leaking session existence to a
   UUID enumerator. Do **not** return 403 here; that would expose "this
   session exists, you just can't read it" vs "no such session". 403 is
   reserved for the unauthenticated case (no session at all), which is
   handled by the auth middleware before this route runs.
3. `run_id` belongs to `session_id` (cross-ownership query). If not → 404
   (an unknown `run_id` and a foreign-but-existing `run_id` are
   indistinguishable to the caller, by design). Same-user but
   cross-session also returns 404 (run not found in this session).
4. Tier-1 Landscape read. Missing required field → CorruptAuditRowError → 500.

- [ ] **Step 1: Reconnaissance — confirm the Landscape read surface.**

```bash
grep -n "def get_run_audit\|def read_run_audit\|landscape\.read\|run_id.*session_id" \
  src/elspeth/web/sessions/*.py src/elspeth/core/landscape/*.py 2>/dev/null | head -30
```

Confirm:

1. **The Landscape data is reached via TWO reads composed.** There is no
   `app.state.landscape.read_run(session_id, run_id)` shortcut and no
   `app.state.landscape` attribute on the app state. The composition is:
   - (a) **Composer-DB read** via the per-request session-service:
     `record = await session_service.get_run(UUID(run_id))` returns a
     `RunRecord` (`src/elspeth/web/sessions/protocol.py:430`). Use
     `record.session_id` to enforce the cross-session check (a run whose
     `session_id` differs from the path's `session_id` → 404). Use
     `record.landscape_run_id` as the join key into the Landscape DB.
     `get_run` raises `ValueError` if the run row does not exist —
     translate to 404 in the service.
   - (b) **Landscape read** via `RunLifecycleRepository.get_run(landscape_run_id)`
     (`src/elspeth/core/landscape/run_lifecycle_repository.py:262`), which
     is **synchronous** and takes a single `str` argument. It returns
     `Run | None`; `None` → `CorruptAuditRowError` (the composer
     `runs_table` row claimed a `landscape_run_id` that does not exist in
     the Landscape DB — that is a Tier-1 audit-database inconsistency).
   If the implementer finds the Landscape read surface needs extending
   (e.g., to expose a new column added by Task 7.0 that
   `RunLifecycleRepository.get_run` does not yet project), escalate to
   the operator — this is a meaningful scope addition.
2. The auth dep used by sibling `/api/sessions/{id}/...` routes —
   `get_current_user` from `auth/middleware.py`. The ownership-check
   helper is the shared `verify_session_ownership` free function in
   `sessions/ownership.py` (raises `HTTPException(404)` — same IDOR
   contract as Task 7.1).
3. The exact attribute names of the Landscape row corresponding to:
   `llm_call_count`, `source_data_hash` (the `output_file_hash` in our
   response model), `started_at` (reused from existing column), `plugin_versions`, and the
   `seeded_from_cache` / `cache_key` metadata written by Task 7's
   `_replay_cached_content_to_landscape`. The audit-story service's
   field-presence check must match those exact names. These columns are
   added in Task 7.0 (see above) — verify they exist before implementing
   Step 4.

- [ ] **Step 2: Write the failing integration test.**

Create `tests/integration/web/test_audit_story_routes.py`:

```python
"""Integration tests for GET /api/sessions/{session_id}/runs/{run_id}/audit-story."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy.pool import StaticPool

from elspeth.web.auth.middleware import get_current_user
from elspeth.web.auth.models import UserIdentity
from elspeth.web.sessions.engine import create_session_engine
from elspeth.web.sessions.routes import create_session_router
from elspeth.web.sessions.schema import initialize_session_schema


@pytest.fixture
def app(tmp_path: Path) -> FastAPI:
    engine = create_session_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    initialize_session_schema(engine)
    app = FastAPI()
    app.state.session_engine = engine
    # app.state.session_service: wired during recon (SessionServiceImpl).
    # app.state.landscape_engine: wired during recon (the Landscape DB engine
    # handle the new `get_run_lifecycle_repo` / `get_landscape_write_repo`
    # dependency providers construct repositories against — per Task 6
    # Step 3). NO `app.state.landscape` / `app.state.run_lifecycle_repo`
    # attribute shim exists; the repositories are FastAPI-`Depends`-
    # injected at the route boundary (operator decision CR-3, 2026-05-19).
    # Tests override the dependency via `app.dependency_overrides[
    # get_run_lifecycle_repo] = lambda: fake_repo` rather than poking
    # `app.state` directly.
    # app.state.settings: WebSettings(auth_provider="local", ...) — required
    # by verify_session_ownership.

    identity = UserIdentity(user_id="alice", username="alice")

    async def _mock_user() -> UserIdentity:
        return identity

    app.dependency_overrides[get_current_user] = _mock_user
    app.include_router(create_session_router())
    return app


def _stage_run_for_audit_story(
    app: FastAPI,
    *,
    session_id: str,
    run_id: str,
    user_id: str = "alice",
    llm_call_count: int = 5,
    output_file_hash: str = "cafebabe",
    started_at: datetime | None = None,
    plugin_versions: dict[str, str] | None = None,
    seeded_from_cache: bool = False,
    cache_key: str | None = None,
    omit_field: str | None = None,
) -> None:
    """Stage the THREE rows required to resolve an audit-story request.

    The service composes three reads in sequence (ownership → composer-DB
    run → Landscape-DB run). A test fixture must seed all three or the
    request will 404 partway through, masking the behaviour under test:

    1. ``sessions_table`` row (composer DB): keyed by ``session_id``,
       owned by ``user_id``, matching ``app.state.settings.auth_provider``.
       Required by ``verify_session_ownership``. Reuse the existing
       ``_make_session`` helper from
       ``tests/integration/web/conftest.py``.

    2. ``runs_table`` row (composer DB): keyed by ``run_id``, with
       ``session_id`` and ``landscape_run_id`` populated. The Tier-1
       invariant on ``RunRecord`` (``protocol.py:466``) requires
       ``landscape_run_id`` to be NOT NULL for terminal statuses
       (``completed`` / ``completed_with_failures`` / ``empty``); use
       ``status="completed"`` and assign ``landscape_run_id = "ldr-" + run_id``
       (or similar synthetic key) — fixtures don't need a real Landscape
       binding, just a consistent one.

    3. ``runs_table`` row (Landscape DB): keyed by the
       ``landscape_run_id`` from step 2, carrying the five new audit-story
       columns added by Task 7.0 plus the existing ``started_at`` column
       that the audit-story response reuses. This is the row the
       ``RunLifecycleRepository.get_run`` lookup resolves.

    ``omit_field`` simulates Tier-1 corruption by writing the Landscape
    row WITHOUT the named column populated (or by setting the column to
    None when the schema permits — the service's field-presence check
    treats both as corruption for the required columns).

    Implementation depends on Step 1 recon for the exact insert patterns
    against both DBs.
    """
    raise NotImplementedError("filled in from Step 1 recon")


def test_get_audit_story_returns_real_landscape_data(app: FastAPI) -> None:
    """Response fields exactly match the Landscape row — no synthesis."""
    _stage_run_for_audit_story(
        app,
        session_id="sess-1",
        run_id="run-1",
        llm_call_count=5,
        output_file_hash="cafebabe1234",
        started_at=datetime(2026, 5, 15, 12, 0, tzinfo=UTC),
        plugin_versions={"web_scrape": "1.0.0", "llm_rate": "1.0.0"},
    )
    client = TestClient(app)
    response = client.get("/api/sessions/sess-1/runs/run-1/audit-story")
    assert response.status_code == 200
    body = response.json()
    assert body["run_id"] == "run-1"
    assert body["session_id"] == "sess-1"
    assert body["llm_call_count"] == 5
    assert body["output_file_hash"] == "cafebabe1234"
    assert body["started_at"] == "2026-05-15T12:00:00+00:00"
    assert body["plugin_versions"] == {"web_scrape": "1.0.0", "llm_rate": "1.0.0"}
    assert body["seeded_from_cache"] is False
    assert body["cache_key"] is None


def test_get_audit_story_for_cache_replay_surfaces_seeded_marker(
    app: FastAPI,
) -> None:
    """Cache-replay run → seeded_from_cache=true, cache_key is the SHA-256."""
    _stage_run_for_audit_story(
        app,
        session_id="sess-1",
        run_id="run-cache-replay",
        llm_call_count=0,  # cache replay: no live LLM calls
        seeded_from_cache=True,
        cache_key="a" * 64,
    )
    client = TestClient(app)
    response = client.get("/api/sessions/sess-1/runs/run-cache-replay/audit-story")
    assert response.status_code == 200
    body = response.json()
    assert body["seeded_from_cache"] is True
    assert body["cache_key"] == "a" * 64
    assert body["llm_call_count"] == 0


def test_get_audit_story_cross_session_returns_404(app: FastAPI) -> None:
    """run_id belongs to a different session → 404 (not 200, not 403).

    Same IDOR-safe contract as the cross-user case: an attacker who learns
    a foreign run_id (e.g. via an unrelated leak) cannot probe its
    existence by querying it under their own session_id. The service
    returns 404 whether the run is unknown or simply in a different
    session — the two cases are deliberately indistinguishable.
    """
    _stage_run_for_audit_story(app, session_id="sess-other", run_id="run-1")
    # Caller (alice) attempts to read sess-1's run-1, but run-1 lives in sess-other.
    _stage_run_for_audit_story(app, session_id="sess-1", run_id="run-2")  # alice's
    client = TestClient(app)
    response = client.get("/api/sessions/sess-1/runs/run-1/audit-story")
    assert response.status_code == 404


def test_get_audit_story_cross_user_returns_404(app: FastAPI) -> None:
    """Session not owned by current user → 404 (IDOR contract).

    Per the established IDOR contract (src/elspeth/web/sessions/ownership.py:33),
    cross-user access returns 404 (not 403) to avoid leaking session existence
    to an attacker enumerating UUIDs. Returning 403 would expose "this session
    exists, you just can't read it" vs "no such session". This is enforced by
    the shared `verify_session_ownership` helper, which raises
    HTTPException(404) on any access-control failure (unknown session, wrong
    user, wrong auth provider).
    """
    # Stage a session owned by bob, with a run.
    _stage_run_for_audit_story(
        app, session_id="sess-bob", run_id="run-b", user_id="bob"
    )
    client = TestClient(app)  # current user is alice (fixture)
    response = client.get("/api/sessions/sess-bob/runs/run-b/audit-story")
    assert response.status_code == 404


@pytest.mark.parametrize(
    "missing_field",
    [
        "llm_call_count",
        "source_data_hash",
        "started_at",
        "plugin_versions",
        "seeded_from_cache",
        "cache_key",
    ],
)
def test_get_audit_story_synthesis_forbidden(
    app: FastAPI, missing_field: str
) -> None:
    """Tier-1 invariant: missing required field → named exception → 500.

    Quality finding R2-M2 (2026-05-19): the original test covered only
    `llm_call_count`. Each of the five other required fields needs its
    own coverage so a future regression that fabricates a default for,
    e.g., `plugin_versions={}`, is caught. Parametrized to exercise all
    six.

    Field origins (per R2-S4 final list, 2026-05-19):
    - `llm_call_count`: runs_table column added by Task 7.0; missing on
      the staged row simulates Tier-1 corruption.
    - `source_data_hash`: aggregated from rows_table; missing simulates
      a runs_table row with no associated rows_table entries (or all
      entries missing the column).
    - `started_at`: existing runs_table column; missing simulates a
      pre-Phase-4 row whose `started_at` is NULL.
    - `plugin_versions`: aggregated from nodes_table; missing simulates
      a runs_table row with no associated nodes_table entries.
    - `seeded_from_cache`: runs_table column added by Task 7.0; missing
      is a server_default failure (Tier-1 corruption).
    - `cache_key`: nullable column (NULL is legal for live runs), so the
      "missing" case here is the runs_table row absent entirely — handled
      separately in the test_unknown_run case. For the parametrize, the
      seeded fixture sets `omit_field="cache_key"` only when staging an
      explicit-corruption case (e.g., column dropped from the row dict
      that simulates the dataclass shape). Per CLAUDE.md "no inference"
      a NULL value is data, not absence; absence means the column is not
      in the row's vars() at all.

    Per CLAUDE.md "no inference - if it's not recorded, it didn't happen".
    A Landscape row missing any required field is corruption, not an
    invitation to fabricate a default.
    """
    _stage_run_for_audit_story(
        app,
        session_id="sess-1",
        run_id="run-broken",
        omit_field=missing_field,
    )
    client = TestClient(app)
    response = client.get("/api/sessions/sess-1/runs/run-broken/audit-story")
    assert response.status_code == 500
    # The error body names the missing field so an auditor can identify
    # the corrupt row without grepping logs.
    assert missing_field in response.json().get("detail", "")


def test_get_audit_story_unknown_run_returns_404(app: FastAPI) -> None:
    client = TestClient(app)
    response = client.get(
        "/api/sessions/sess-1/runs/nonexistent-run/audit-story"
    )
    assert response.status_code == 404
```

- [ ] **Step 3: Run test to verify it fails.**

```bash
.venv/bin/python -m pytest tests/integration/web/test_audit_story_routes.py -v
```

Expected: FAIL.

- [ ] **Step 4: Implement Pydantic model, service, and route.**

Create `src/elspeth/web/sessions/audit_story_models.py` (leaf module per
R2-8, 2026-05-19 — extracted from `sessions/schemas.py` so neither the
route nor the service has to import the other):

```python
"""Pydantic response model for GET /api/sessions/{id}/runs/{run_id}/audit-story.

Leaf module — depends only on `pydantic` + stdlib `datetime`. The route
module (`sessions/routes.py`) and the service module
(`sessions/audit_story_service.py`) both import from here, so neither
has to import the other. Resolves the circular-import pattern that the
original draft worked around with lazy imports (R2-8, 2026-05-19;
CLAUDE.md "Shifting the Burden").
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict


class RunAuditStoryResponse(BaseModel):
    """Audit-story response for GET /api/sessions/{id}/runs/{run_id}/audit-story.

    All fields are read from a real Landscape audit row. No field is ever
    synthesised or defaulted. Absence of any field below in the underlying
    row is Tier-1 corruption (CorruptAuditRowError → 500).

    Per R2-S4 (2026-05-19), `source_data_hash` (surfaced as
    `output_file_hash`) and `plugin_versions` are aggregated by the
    service from `rows_table` and `nodes_table` respectively, not read
    from a denormalised `runs_table` column.
    """

    model_config = ConfigDict(strict=True, extra="forbid")

    run_id: str
    session_id: str
    llm_call_count: int
    output_file_hash: str  # aggregated from rows_table.source_data_hash (R2-S4)
    started_at: datetime  # reused from existing schema.py:65 column
    plugin_versions: dict[str, str]  # aggregated from nodes_table.plugin_version (R2-S4)
    seeded_from_cache: bool
    cache_key: str | None
```

Create `src/elspeth/web/sessions/audit_story_service.py`:

```python
"""Audit-story service — read-only Landscape projection for a single run.

Phase 4A.7.2. The frontend's getRunAuditSummary (21b2 Task 9) calls the
route that wraps this service.

Tier model:
- Inputs (session_id, run_id, user): Tier 3 trust — validated via the
  shared `verify_session_ownership` helper before touching the Landscape.
  The helper raises HTTPException(404) on any access-control failure
  (IDOR contract; sessions/ownership.py:33).
- Landscape rows: Tier 1 — every required field MUST be present. Absence
  → CorruptAuditRowError → 500.

The Landscape data is fetched via a TWO-read composition:
  1. composer DB: ``session_service.get_run(UUID(run_id))`` resolves the
     ``(session_id, landscape_run_id)`` binding.
  2. Landscape DB: ``run_lifecycle_repo.get_run(landscape_run_id)`` (sync,
     single arg) returns the row with the audit-story columns.

Per CLAUDE.md no-defensive-programming: NO synthesis, NO `.get(field, default)`,
NO `getattr(row, field, None)`. Direct attribute access only. Absent fields
raise the named exception with `from exc` chaining so an auditor sees which
field was missing.
"""

from __future__ import annotations

from uuid import UUID

from fastapi import HTTPException, Request

from elspeth.core.landscape.run_lifecycle_repository import RunLifecycleRepository
from elspeth.web.auth.models import UserIdentity
from elspeth.web.sessions.audit_story_models import RunAuditStoryResponse
from elspeth.web.sessions.ownership import verify_session_ownership
from elspeth.web.sessions.protocol import SessionServiceProtocol


class CorruptAuditRowError(Exception):
    """Raised when the Landscape row is missing a required field."""

    def __init__(self, *, run_id: str, missing_field: str) -> None:
        super().__init__(
            f"audit row for run {run_id!r} is missing required field "
            f"{missing_field!r}; this is Tier-1 corruption"
        )
        self.run_id = run_id
        self.missing_field = missing_field


async def get_run_audit_story(
    *,
    session_id: str,
    run_id: str,
    user: UserIdentity,
    request: Request,
    session_service: SessionServiceProtocol,
    run_lifecycle_repo: RunLifecycleRepository,
) -> RunAuditStoryResponse:
    """Return the audit-story projection for ``(session_id, run_id)``.

    Authorization order (see Authorization order section above):
      1. Caller authenticated — enforced upstream by the route dep.
      2. Caller owns ``session_id`` — ``verify_session_ownership`` raises
         ``HTTPException(404)`` on mismatch (IDOR contract;
         ``sessions/ownership.py:33``).
      3. ``run_id`` belongs to ``session_id`` — composer-DB read.
      4. Landscape row exists and is complete — Landscape read + Tier-1
         field-presence check.

    The Landscape data is composed from TWO reads (no
    ``app.state.landscape`` shortcut exists):
      (a) composer DB: ``session_service.get_run(UUID(run_id))`` →
          ``RunRecord`` with ``session_id`` (for the cross-session check)
          and ``landscape_run_id`` (the join key);
      (b) Landscape DB: ``run_lifecycle_repo.get_run(landscape_run_id)``
          → the ``Run`` row carrying the audit-story columns added by
          Task 7.0.

    Raises:
        HTTPException(404): session not owned by caller (from
            ``verify_session_ownership``), run not found in the composer
            DB, run belongs to a different session, or the Landscape row
            for the run is missing.
        CorruptAuditRowError: the Landscape row is missing a Tier-1
            required field (→ 500).
    """
    # 1. Ownership check. Raises HTTPException(404) on mismatch — IDOR-safe.
    await verify_session_ownership(
        session_id=UUID(session_id),
        user=user,
        request=request,
    )

    # 2. Composer-DB read: locate the run, verify it belongs to this session.
    try:
        record = await session_service.get_run(UUID(run_id))
    except ValueError:
        # Composer-DB has no row for this run_id.
        raise HTTPException(
            status_code=404, detail=f"Run {run_id!r} not found"
        ) from None
    if str(record.session_id) != session_id:
        # Run exists but in a different session. Return 404 (IDOR-safe):
        # do not reveal that the run exists elsewhere.
        raise HTTPException(
            status_code=404,
            detail=f"Run {run_id!r} not found in session {session_id!r}",
        )
    if record.landscape_run_id is None:
        # The composer DB has no Landscape join key yet — the run did not
        # reach the engine-completion path that writes landscape_run_id.
        # That is not a Tier-1 corruption, it just means there is no
        # audit story to read.
        raise HTTPException(
            status_code=404,
            detail=f"Run {run_id!r} has no Landscape audit row yet",
        )

    # 3. Landscape read. Synchronous single-arg API on the repository.
    landscape_row = run_lifecycle_repo.get_run(record.landscape_run_id)
    if landscape_row is None:
        # The composer-DB row claimed a landscape_run_id that does not
        # exist in the Landscape DB. That is a Tier-1 cross-database
        # inconsistency: the audit trail is broken.
        raise CorruptAuditRowError(
            run_id=run_id, missing_field="<landscape row missing entirely>"
        )

    # 4. Tier-1 field-presence check on the runs_table row. Direct
    # attribute access — no `getattr(row, field, default)`, no `.get(...)`,
    # no synthesis. `started_at` is the existing column (schema.py:65),
    # reused; the other three are added by Task 7.0.
    #
    # Per R2-S4 (2026-05-19), `source_data_hash` and `plugin_versions`
    # are NOT on the runs_table row — they are aggregated from
    # rows_table and nodes_table respectively, by the two queries below.
    # The presence check on those aggregated fields lives in their own
    # query branches (a missing JOIN result is its own corruption signal).
    for required in (
        "llm_call_count",
        "started_at",
        "seeded_from_cache",
        "cache_key",
    ):
        if not _row_has_field(landscape_row, required):
            raise CorruptAuditRowError(run_id=run_id, missing_field=required)

    # 5. Aggregated reads (R2-S4) — source_data_hash from rows_table,
    # plugin_versions from nodes_table. Step 1 recon identified the
    # repository methods or raw SELECTs; both are bound here.
    aggregated_source_hash = _aggregate_source_data_hash(
        run_lifecycle_repo, record.landscape_run_id
    )
    if aggregated_source_hash is None:
        raise CorruptAuditRowError(run_id=run_id, missing_field="source_data_hash")
    aggregated_plugin_versions = _aggregate_plugin_versions(
        run_lifecycle_repo, record.landscape_run_id
    )
    if not aggregated_plugin_versions:
        raise CorruptAuditRowError(run_id=run_id, missing_field="plugin_versions")

    # Tier-1: `llm_call_count` may be NULL on live (non-tutorial) runs
    # per R2-S3 operator decision (Task 7.0). Surface NULL → 0 only when
    # `seeded_from_cache=True` (cache replay writes 0); raise for live
    # runs so the auditor sees the honest gap rather than a fabricated 0.
    if landscape_row.llm_call_count is None and not landscape_row.seeded_from_cache:
        # Live run with no count — Phase 4 baseline behaviour. The
        # endpoint returns 500 with a clear message; the operator (or
        # the sibling phase that adds general counting) backfills.
        raise CorruptAuditRowError(
            run_id=run_id, missing_field="llm_call_count (live-run gap, R2-S3)"
        )

    return RunAuditStoryResponse(
        run_id=run_id,
        session_id=session_id,
        llm_call_count=landscape_row.llm_call_count or 0,
        output_file_hash=aggregated_source_hash,
        started_at=landscape_row.started_at,
        plugin_versions=aggregated_plugin_versions,
        seeded_from_cache=landscape_row.seeded_from_cache,
        cache_key=landscape_row.cache_key,
    )


def _row_has_field(row: object, field: str) -> bool:
    """Check whether ``row`` carries the named attribute.

    Detects column-missing (corruption), NOT value-is-None (which is legal
    for ``cache_key`` on a live run). Implementation depends on the exact
    shape returned by ``RunLifecycleRepository.get_run`` (likely a frozen
    dataclass — confirm during Step 1 recon).

    NOTE: do NOT use ``hasattr()`` here — per CLAUDE.md it is
    unconditionally banned (it swallows arbitrary ``@property`` exceptions).
    Use ``field in vars(row)`` for dataclass rows or
    ``field in row._fields`` for NamedTuple rows; pick the form that
    matches the actual return type.
    """
    raise NotImplementedError("filled in from Step 1 recon")
```

Add the route handler to `src/elspeth/web/sessions/routes.py` alongside the
existing `update_session` PATCH route. Module-top imports — no lazy
imports (R2-8, 2026-05-19):

```python
# At src/elspeth/web/sessions/routes.py module top (alongside existing imports):
from elspeth.web.sessions.audit_story_models import RunAuditStoryResponse
from elspeth.web.sessions.audit_story_service import (
    CorruptAuditRowError,
    get_run_audit_story,
)

@router.get(
    "/{session_id}/runs/{run_id}/audit-story",
    response_model=RunAuditStoryResponse,
)
async def get_run_audit_story_route(
    session_id: str,
    run_id: str,
    request: Request,
    user: UserIdentity = Depends(get_current_user),
    run_lifecycle_repo: RunLifecycleRepository = Depends(get_run_lifecycle_repo),
) -> RunAuditStoryResponse:
    # The Landscape read uses `RunLifecycleRepository` injected via the
    # FastAPI dependency `get_run_lifecycle_repo` (Task 6 Step 3). NO
    # `app.state.landscape` or `app.state.run_lifecycle_repo` attribute
    # shim exists in the Phase 4 surface (operator decision CR-3,
    # 2026-05-19); the repository is constructed per request against
    # the engine on `app.state.landscape_engine`.
    # `session_service` is still read from `app.state.session_service`
    # because that's the pre-existing project convention for the
    # composer-DB session service — out of scope for the Phase 4 repo
    # split. Ownership failures (HTTPException(404) from
    # verify_session_ownership) and "run not found"/"run-in-other-
    # session" cases (HTTPException(404) raised inside the service)
    # propagate directly to FastAPI's handler.
    try:
        return await get_run_audit_story(
            session_id=session_id,
            run_id=run_id,
            user=user,
            request=request,
            session_service=request.app.state.session_service,
            run_lifecycle_repo=run_lifecycle_repo,
        )
    except CorruptAuditRowError as exc:
        # Tier-1 corruption: surface the field name in the 500 body so an
        # auditor reading the response knows exactly which column is broken.
        raise HTTPException(status_code=500, detail=str(exc)) from exc
```

- [ ] **Step 5: Run test to verify it passes.**

```bash
.venv/bin/python -m pytest tests/integration/web/test_audit_story_routes.py -v
```

Expected: PASS — all six tests green, including the no-synthesis test.

- [ ] **Step 6: Run the full integration suite.**

```bash
.venv/bin/python -m pytest tests/integration/web/ -v
```

Expected: PASS.

- [ ] **Step 7: Commit.**

```bash
git add src/elspeth/web/sessions/audit_story_service.py \
        src/elspeth/web/sessions/routes.py \
        src/elspeth/web/sessions/schemas.py \
        tests/integration/web/test_audit_story_routes.py
git commit -m "feat(web): GET /api/sessions/{id}/runs/{run_id}/audit-story (Phase 4A.7.2)"
```

---

## Task 7.3: ~~Frontend client functions~~ — RELOCATED to 21b2 Task 7.5

**Status:** Moved to `21b2-phase-4-frontend-part-2.md` Task 7.5 per
Architecture finding M-R2-2 (2026-05-19). PR-21a (backend) no longer
contains TypeScript changes — the frontend `client.ts` work was
blocking independent revert of the two PRs. See 21b2 §"Task 7.5: Frontend
API client surface — `runTutorialPipeline`, `getRunAuditSummary`,
`renameSession`, `deleteTutorialOrphans`" for the full task content.

---

## Task 8: Tutorial telemetry counters at launch

**Files:**
- Modify: `src/elspeth/web/preferences/service.py` — extend with the
  `completed_total` counter emit site keyed on a server-inferred
  `completion_path` label (same module that owns
  `_PREFERENCES_PATCH_COUNTER`, so the same OTel meter is reused).
- Create: `src/elspeth/web/composer/tutorial_telemetry.py` — module that
  declares the two new counters (kept separate from
  `preferences/service.py` only because the abandon-counter emit site
  is a tutorial-specific route, not a preference write).
- Create: `src/elspeth/web/composer/tutorial_abandon_routes.py` — single
  POST `/api/tutorial/abandon` endpoint that increments the abandon
  counter when the frontend sends a beacon on browser-close.
- Modify: `tests/unit/web/preferences/test_service.py` — add counter-emit
  assertions for the completion-path label discrimination.
- Create: `tests/unit/web/composer/test_tutorial_telemetry.py` — counter
  declaration + abandon-route emit tests.

**Phase 8 forward-fit alignment.** Phase 8 already shipped a `_Counter`
slot named `tutorial_completed_total` on `SessionsTelemetry`
(`src/elspeth/web/sessions/telemetry.py:204`) wired to the OTel name
`composer.tutorial.completed_total` (with 'd';
`sessions/telemetry.py:317-323`) and a write helper
`record_tutorial_completed` (`composer/telemetry_phase8.py:92,255-265`).
Phase 8's `DELIBERATELY ABSENT` markers at
`composer/telemetry_phase8.py:234` (deferred Phase-9 retake-counter
context) and `sessions/telemetry.py:205` (`tutorial_replayed_total` —
deferred per Phase 9 followups Decision 2) are not Phase 4's slot — they
guard a different deferred counter. Phase 4 is the implementer that
fills `composer.tutorial.completed_total`'s emit site (which has had no
caller until now). Per operator decision CR-1 (2026-05-19), Phase 4
conforms to the already-shipped Phase 8 name (`completed_total`, with
'd') rather than introducing a parallel `complete_total` namespace.

**Counters declared.** Two OTel counters back Phase 4's tutorial
telemetry surface, declared in
`src/elspeth/web/composer/tutorial_telemetry.py` using the same pattern
that `src/elspeth/web/preferences/service.py` uses for
`_PREFERENCES_PATCH_COUNTER` (lines 84-88 in the live file — single
OTel meter handle, single counter handle per name, no per-call meter
lookup):

- `composer.tutorial.completed_total` — incremented when
  `update_composer_preferences` writes `tutorial_completed_at`. The
  counter carries a single `completion_path` label whose value is
  inferred server-side from the PATCH payload shape (see
  Discrimination rule below). Permitted label values: `first_time`,
  `skip`, `retake`, `repeat`. This is the same OTel counter name
  declared by Phase 8 at `sessions/telemetry.py:317-323`; Phase 4
  introduces the **emit site** (Phase 8 only declared the slot).
- `composer.tutorial.abandon_total` — incremented by the
  `/api/tutorial/abandon` endpoint when the frontend sends a
  `navigator.sendBeacon` (best-effort signal) on browser-close or
  navigation-away while the user's `tutorial_completed_at IS NULL`
  and there was tutorial-machine state in flight. The endpoint is
  fire-and-forget: it increments the counter, returns 204, and never
  blocks the page-unload path.

**Discrimination rule (completion_path label).** Single source of
truth: the `update_composer_preferences` code path that writes the
column inspects the resolved `payload` AND the prior-row state to
decide which `completion_path` value to attach. The rule is purely
server-side — the frontend does **not** send a discriminator field;
it just sends the right PATCH payload for the gesture the user made
(turn-6 finalise includes `default_mode` AND `tutorial_completed_at`;
turn-1 skip includes only `tutorial_completed_at`; Phase 8 retake
sends `tutorial_completed_at: null`). This keeps the wire contract
identical to Phase 1B and avoids a frontend-supplied label that a
malicious or buggy client could spoof.

The discrimination requires reading the **prior** `tutorial_completed_at`
value before the write — similar to how Task 3's service extension
already reads `existing_raw` for the `default_composer_mode` change-
detection. Concretely, in `update_composer_preferences`, alongside the
`existing_raw` read introduced by Task 3:

```python
# Phase 4A.8 — read the prior tutorial_completed_at value so the
# completion_path label can discriminate first_time / skip / retake /
# repeat without a frontend-supplied flag. Mirrors Task 3's
# `existing_raw["default_composer_mode"]` read for change detection.
existing_tutorial = (
    existing_raw["tutorial_completed_at"] if existing_raw else None
)
```

Then, in the success-branch (the same site that increments
`_PREFERENCES_PATCH_COUNTER`), after the existing counter emit:

```python
# Phase 4A.8 — tutorial completion telemetry. One counter
# (composer.tutorial.completed_total — declared by Phase 8;
# Phase 4 fills the emit site) with a server-inferred
# `completion_path` label. A retake (Phase 8) writes NULL; a
# repeat-finalise (already-completed user re-PATCHing) is a no-op
# for the user but a label value we want to count separately so an
# operator can spot anomalous re-PATCH activity.
tutorial_in_payload = "tutorial_completed_at" in payload.model_fields_set
if tutorial_in_payload:
    new_tutorial = payload.tutorial_completed_at
    addressed_mode = "default_mode" in payload.model_fields_set
    if existing_tutorial is None and new_tutorial is not None and addressed_mode:
        completion_path = "first_time"
    elif existing_tutorial is None and new_tutorial is not None and not addressed_mode:
        completion_path = "skip"
    elif existing_tutorial is not None and new_tutorial is None:
        completion_path = "retake"
    elif existing_tutorial is not None and new_tutorial is not None:
        completion_path = "repeat"
    else:
        completion_path = None  # both None — no-op PATCH; do not emit
    if completion_path is not None:
        _TUTORIAL_COMPLETED_COUNTER.add(
            1, attributes={"completion_path": completion_path}
        )
```

The counter handle is imported from `tutorial_telemetry`. The
attribute vocabulary is fixed at four values
(`first_time | skip | retake | repeat`); adding new values later is
non-breaking on the OTel side, but extension MUST be paired with a
plan addendum so dashboards stay consistent.

**Abandon endpoint (best-effort).** The frontend wires a
`navigator.sendBeacon('/api/tutorial/abandon', '')` on a `beforeunload`
or `pagehide` handler when the tutorial machine state is not in
`done`. The endpoint:

```python
@router.post("/api/tutorial/abandon", status_code=204)
async def tutorial_abandon(_user: UserIdentity = Depends(get_current_user)) -> Response:
    """Fire-and-forget tutorial-abandon beacon.

    Best-effort signal — browsers do not guarantee beacon delivery on
    unload. The counter is operational telemetry, not audit primary
    record; missed beacons are an accepted limitation of this design.
    No request body is consumed; user identity is sufficient.
    """
    _TUTORIAL_ABANDON_COUNTER.add(1, attributes={})
    return Response(status_code=204)
```

**Best-effort nature is explicit.** Per CLAUDE.md primacy (audit →
telemetry → logger): the tutorial-completed event itself is a
Landscape write (the `tutorial_completed_at` timestamp column written
by Task 3's service path) — that is the audit primary record. These
three counters are operational telemetry, layered on top to give the
team a fast aggregate signal ("are users actually finishing the
tutorial, or are they skipping or abandoning?") without scanning audit
rows. The abandon counter in particular is **best-effort** — a user
who closes their laptop, kills the browser process, or loses network
mid-tutorial will not send the beacon, and the counter under-counts.
The plan accepts this; no compensating server-side timeout exists.

**Tests.** Reuse the `_last_patch_counter_attributes` helper pattern
that Task 3 introduces (per P3). For each counter, add a sibling
helper `_last_tutorial_counter_attributes(reader, counter_name)` that
peeks at the in-memory MeterProvider's recorded points; tests assert
exactly one increment per gesture and zero increments per
non-triggering PATCH (e.g., a mode toggle on a user whose
`tutorial_completed_at` is already non-NULL — no counter fires).

**Fixture and helper (Quality finding R2-M1, 2026-05-19).** The
counter-emit tests reference a `tutorial_metric_reader` fixture AND a
`_last_tutorial_counter_attributes` helper that were not defined in
the original draft. Both must be defined as Step 1 of this task:

```python
# tests/unit/web/composer/conftest.py — extend (create if absent):
from collections.abc import Iterator
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import InMemoryMetricReader
import pytest

from elspeth.web.composer import tutorial_telemetry as _tut_module


@pytest.fixture
def tutorial_metric_reader(monkeypatch: pytest.MonkeyPatch) -> Iterator[InMemoryMetricReader]:
    """In-memory metric reader bound to the tutorial-telemetry meter.

    Same shape as the preferences variant (R2-13, see 21a1 Task 3
    conftest) but rebinds the tutorial module's two counter handles
    (`_TUTORIAL_COMPLETED_COUNTER`, `_TUTORIAL_ABANDON_COUNTER`) so
    counter increments land in the in-memory reader.
    """
    reader = InMemoryMetricReader()
    provider = MeterProvider(metric_readers=[reader])
    new_meter = provider.get_meter("tutorial")
    monkeypatch.setattr(_tut_module, "_meter", new_meter)
    monkeypatch.setattr(
        _tut_module,
        "_TUTORIAL_COMPLETED_COUNTER",
        new_meter.create_counter(
            name="composer.tutorial.completed_total",
            description=_tut_module._TUTORIAL_COMPLETED_COUNTER.description,
        ),
    )
    monkeypatch.setattr(
        _tut_module,
        "_TUTORIAL_ABANDON_COUNTER",
        new_meter.create_counter(
            name="composer.tutorial.abandon_total",
            description=_tut_module._TUTORIAL_ABANDON_COUNTER.description,
        ),
    )
    try:
        yield reader
    finally:
        provider.shutdown()
        reader.shutdown()


def _last_tutorial_counter_attributes(
    reader: InMemoryMetricReader, counter_name: str
) -> dict[str, object]:
    """Read the latest data-point attributes for the named tutorial
    counter from the in-memory metric reader.

    Walks the resource_metrics → scope_metrics → metrics tree and
    returns the attributes dict from the most-recently-recorded data
    point for `counter_name`. Raises AssertionError if the counter has
    no data points — that condition indicates the emit-site under test
    did not fire (a Task 8 regression).
    """
    data = reader.get_metrics_data()
    for resource_metric in data.resource_metrics:
        for scope_metric in resource_metric.scope_metrics:
            for metric in scope_metric.metrics:
                if metric.name == counter_name:
                    points = list(metric.data.data_points)
                    if points:
                        return dict(points[-1].attributes)
    raise AssertionError(
        f"{counter_name} had no data points — emit-site did not fire."
    )
```

The completion-path tests (which assert via the *preferences* service
emit site, not the tutorial-telemetry module) reuse the
`in_memory_metric_reader` fixture from 21a1 Task 3's conftest extension
— that conftest is scoped to `tests/unit/web/preferences/` and is the
correct seam for asserting `composer.preferences.patch_total` AND the
sibling `composer.tutorial.completed_total` emit fired from the same
PATCH-path code site. The `tutorial_metric_reader` fixture above is
used only for the abandon-route test (the abandon counter is owned by
`tutorial_telemetry.py`, not by the preferences module).

Test list:

- `test_completed_counter_first_time_label_fires_on_turn_six_finalise` —
  a PATCH carrying both `default_mode` and `tutorial_completed_at`,
  with the prior preference row having `tutorial_completed_at=None`,
  increments `composer.tutorial.completed_total` by 1 with
  `attributes={"completion_path": "first_time"}`.
- `test_completed_counter_skip_label_fires_on_turn_one_skip` — a PATCH
  carrying only `tutorial_completed_at` (no `default_mode`), prior row
  `tutorial_completed_at=None`, increments
  `composer.tutorial.completed_total` by 1 with
  `attributes={"completion_path": "skip"}`.
- `test_completed_counter_retake_label_fires_on_phase8_retake` — a
  PATCH that writes `tutorial_completed_at=null` (Phase 8 retake) on
  a prior-non-null row increments `composer.tutorial.completed_total`
  by 1 with `attributes={"completion_path": "retake"}`.
- `test_completed_counter_repeat_label_fires_on_idempotent_refinalise`
  — a PATCH that writes `tutorial_completed_at` when the prior row
  already had `tutorial_completed_at` non-NULL increments
  `composer.tutorial.completed_total` by 1 with
  `attributes={"completion_path": "repeat"}`.
- `test_completed_counter_does_not_fire_on_mode_only_patch` — a PATCH
  carrying only `default_mode` (no `tutorial_completed_at` key in
  `model_fields_set`) does NOT increment
  `composer.tutorial.completed_total`.
- `test_abandon_route_increments_counter` — a POST to
  `/api/tutorial/abandon` returns 204 and increments
  `composer.tutorial.abandon_total` by 1.

**CLAUDE.md compliance check.** No `slog.debug` calls anywhere in this
task — per memory `feedback_no_slog_recommendations.md`, operational
visibility goes through OTel counters, not the structured logger. The
tutorial-completion fact itself is audit-recorded by Task 3 (the
column write); these counters are the operational-visibility layer on
top.

**PR strategy.** Task 8 lands in **PR-21a** (backend) per
[21-phase-4-hello-world-tutorial.md](21-phase-4-hello-world-tutorial.md)
§"PR strategy". The frontend beacon-emit wiring (the `sendBeacon` call
itself) is owned by Phase 4B (a small addition to the
`HelloWorldTutorial` container's unmount handler); the route exists in
PR-21a so the beacon has a valid target the moment the frontend ships.

- [ ] **Step 0: Define the `tutorial_metric_reader` fixture + helper (R2-M1, 2026-05-19).**

Extend `tests/unit/web/composer/conftest.py` (create if absent) with the
`tutorial_metric_reader` fixture and the
`_last_tutorial_counter_attributes` helper exactly as shown in the
"Fixture and helper" section above. Without these the counter-emit
tests in Step 4 below cannot resolve their references and the entire
TDD loop fails at import time.

- [ ] **Step 1: Declare counters in `tutorial_telemetry.py`.**

Mirror `_PREFERENCES_PATCH_COUNTER`'s declaration block from
`preferences/service.py` (single meter, single counter per name).

- [ ] **Step 2: Add the discriminator + emit block to `service.py`.**

Insert immediately after the existing `_PREFERENCES_PATCH_COUNTER.add(...)`
call (lines ~293-300 in the live file). Imports go alongside the
existing telemetry imports at the top of the file.

- [ ] **Step 3: Add the abandon route.**

One handler, one router. Register via the existing FastAPI
app-composition site that Task 6 already touches.

- [ ] **Step 4: Run tests to verify.**

```bash
.venv/bin/python -m pytest tests/unit/web/preferences/test_service.py \
                            tests/unit/web/composer/test_tutorial_telemetry.py -v
```

Expected: PASS — all six counter-emit tests green, plus existing
service tests unchanged.

- [ ] **Step 5: Commit.**

```bash
git add src/elspeth/web/preferences/service.py \
        src/elspeth/web/composer/tutorial_telemetry.py \
        src/elspeth/web/composer/tutorial_abandon_routes.py \
        tests/unit/web/preferences/test_service.py \
        tests/unit/web/composer/test_tutorial_telemetry.py
git commit -m "feat(web): tutorial telemetry counters at launch (Phase 4A.8)"
```

---

## What Phase 4A leaves the backend in

After Tasks 0–8: new column, Tier-1 guards, cache module (filesystem-backed,
corruption-detecting), run-path cache consult, **tutorial-run route** (POST
/api/tutorial/run, Task 7.1) with completed-user / freeform-user bypass,
**audit-story route** (GET …/audit-story, Task 7.2) with no-synthesis Tier-1
invariant, **frontend client functions** (`runTutorialPipeline`,
`getRunAuditSummary`, renamed `renameSession`, Task 7.3), and **launch
telemetry counters** (`composer.tutorial.completed_total` keyed by a
`completion_path` label with values `first_time | skip | retake | repeat`,
plus `composer.tutorial.abandon_total`, Task 8 — `completed_total` fills
Phase 8's already-shipped slot at `sessions/telemetry.py:317-323`). Phase
4B wires the frontend components against these surfaces.

## Risks and mitigations

Key risks: run-path entry point not where assumed (Task 7 Step 1 recon resolves); cache fires for non-tutorial users (gate on `tutorial_completed_at is None` + regression test); model_id derivation drifts (integration tests assert hit-on-pre-populated-cache); concurrent writers (atomic via `os.replace`); audit-story synthesis (no-synthesis Tier-1 test `test_get_audit_story_synthesis_forbidden` in Task 7.2 pins the invariant); `_is_canonical_seed_pipeline` force-live flag collides with Task 7's existing canonical-seed shape check (Task 7.1 Step 5 adds the flag check at the top of the helper; Task 7's `test_non_tutorial_user_skips_cache` and `test_edited_prompt_skips_cache` must remain green after Task 7.1 lands — verified by Step 7); renamed `updateSessionTitle` call sites missed (Task 7.3 Step 5 runs `tsc --noEmit` to catch).

## Forward compatibility

Phase 8 schema additions wipe `tutorial_completed_at` via DB-delete policy — every user retakes the tutorial. Structural fix (Alembic) owned by the roadmap.

**Phase 8 retake mechanism (co-owned contract).** Phase 8 Task 6 ships a
"Replay hello-world tutorial" button that nulls `tutorial_completed_at` via
`PATCH /api/composer-preferences` with body `{"tutorial_completed_at": null}`.
The schema (`nullable=True`), the Pydantic model (`datetime | None`), the
service (absent-vs-null discrimination via `model_fields_set`), and the route
(no structural change) shipped by Phase 4 are all the preconditions for that
mechanism. See §"Cross-plan contract — `tutorial_completed_at` PATCH
semantics" near the top of this document. The audit emit for the retake
event itself lives in Phase 8; this plan does not change
`composer.preferences.patch_total` or its emit site.

## Memory references

- `project_composer_first_run_tutorial`
- `project_composer_canonical_test_case`
- `project_composer_dynamic_source_from_chat`
- `project_composer_default_guided_with_opt_out`
- `project_db_migration_policy`
- `feedback_no_calendar_shipping_commitments`

---

## Review history

### 2026-05-15 — review panel

| ID | Severity | Status | Summary |
|---|---|---|---|
| 4A-F1 | BLOCKER (Systems) | Applied | DB-delete cadence section added after §Scope boundaries |
| 4A-F2 | CRITICAL (Systems) | Applied (P8 refinement) | Cache path now flows through validated `WebSettings.tutorial_cache_dir` (`Path \| None` + `model_validator` defaulting to `data_dir / "tutorial_cache"`); no env-var lookup inside `TutorialCache`; startup write-permission check at app-composition site; legacy env-var path and hardcoded absolute-path defaults removed |
| 4A-F3 | CRITICAL (Quality) | Applied | Corrupt-cache integration test added to Task 5; orchestrator-propagation integration test added |
| 4A-F4 | IMPORTANT (Architecture) | Applied | Sequencing note added to §DB-delete cadence |
| 4A-F5 | IMPORTANT (Quality) | Applied | Preflight Task 0 Step 4 column check made precise |
| 4A-F6 (from 4B2-F1) | IMPORTANT (Architecture) | Applied | `POST /api/tutorial/run` route specified in §New endpoints |
| 4A-F7 (from 4B2-F3) | IMPORTANT (Quality) | Applied | `GET /api/sessions/{id}/runs/{run_id}/audit-story` specified in §New endpoints |

### 2026-05-19 — cross-plan contract amendment (Phase 4 ↔ Phase 8)

Pass-1 review of Phase 4 surfaced a Systems S1 contract rupture against
Phase 8 Task 6's retake mechanism: Phase 4 originally disallowed
nullification of `tutorial_completed_at` via PATCH (the Pydantic field
default was treated as "leave alone" and operators were told to SQL-UPDATE
directly), while Phase 8 expected to clear the column via the same PATCH
endpoint. Synthesizer-adopted resolution: Option (a) — allow nullification
via explicit `null` in the PATCH body; distinguish absent-from-payload from
explicit-null via Pydantic v2's `model_fields_set`. Same field, same column,
single shared contract co-owned by Phases 4 and 8.

Edits applied:

- §Scope boundaries `update_composer_preferences` bullet rewritten to name
  the three semantic states (absent / datetime / null).
- New §"Cross-plan contract — `tutorial_completed_at` PATCH semantics"
  section inserted between §"DB-delete cadence" and §"New endpoints".
- Task 1 column-add prose: explicit "do not add NOT NULL or server_default"
  note attached to the `nullable=True` line.
- Task 2 model comment rewritten to document the three-state contract and
  the `model_fields_set` discrimination pattern.
- Task 3 service code: absent-vs-null distinguished via `model_fields_set`
  for `tutorial_completed_at` only. `default_mode` and `banner_dismissed_at`
  retain the Phase 1A "None = preserve" convention (no client need to NULL
  either field).
- Task 3 tests: two new tests (`test_explicit_null_clears_tutorial_completed_at`,
  `test_absent_field_and_explicit_null_are_distinguished`).
- Task 4 tests: one new test (`test_patch_with_explicit_null_clears_tutorial`).
- §Forward compatibility: new paragraph naming the Phase 8 retake mechanism
  and citing the shared contract block.

Co-edits applied in `21-phase-4-hello-world-tutorial.md` (Open Question C3
resolution updated, vocabulary block extended, file inventory pointer) and
`20-phase-8-polish-and-telemetry.md` (Task 6 PATCH body changed from
`{"tutorial_completed": false}` to `{"tutorial_completed_at": null}`,
Trust-tier check updated, telemetry-primacy footnote updated, retake
audit-emit boundary surfaced for Phase 8 reviewers).
