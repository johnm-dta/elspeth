> **Part of the [Tutorial Staged Recut plan](./00-overview.md).** Read the [overview](./00-overview.md) first — it holds the Global Constraints (§9.2 gate commands) and the "use VERBATIM" Shared Interfaces every task depends on. Phases execute **P0 → P7 in order**.

## Phase P6 — Entry protocol + profile lifecycle + concurrency

> **Phase dependency note (read first).** This phase CONSUMES symbols owned by **P0**:
> `WorkflowProfile`, `EMPTY_PROFILE`, `TUTORIAL_PROFILE`, `WorkflowProfileKind`
> (all in `src/elspeth/web/composer/guided/profile.py`), the `GuidedSession.profile`
> field + the `GuidedSession.initial(profile=...)` keyword signature + the v6
> `to_dict`/`from_dict` round-trip (`composer/guided/state_machine.py`). **Do not author
> any of those here.** If P0 has not landed, each task below will fail its run-to-fail
> with an `ImportError`/`AttributeError` that names the missing P0 symbol — that is the
> correct signal to land P0 first, not to re-create the symbol in this phase.
>
> This phase OWNS: `WorkflowProfileResponse` (Pydantic, `sessions/schemas.py`); the
> `profile` field on `GuidedSessionResponse`; the TS `WorkflowProfile` interface + the
> `profile` field on the TS `GuidedSession`; `startGuidedSession` (`client.ts`); the
> external `POST /api/sessions/{session_id}/guided/start` route (implemented inside
> the sessions subrouter as `/{session_id}/guided/start`); the `_strip_guided_profile_in_meta` fork
> helper in `sessions/service.py`; the new optional `step_index` field on
> `GuidedRespondRequest` + its 409 guard in `post_guided_respond`.

All file paths are relative to `src/elspeth/web/` unless prefixed. Run every `pytest`
command from the repo root (`/home/john/elspeth`).
Frontend commands run from `src/elspeth/web/frontend`.

---

### Task P6.1: `WorkflowProfileResponse` Pydantic model + `profile` field on `GuidedSessionResponse`

**Files:**
- Modify: `sessions/schemas.py` (add `WorkflowProfileResponse` after `ChatTurnResponse` ~:307; add `profile` field to `GuidedSessionResponse` :323)

**Interfaces:**
- Consumes: `_StrictResponse` (`sessions/schemas.py:39`, `model_config = ConfigDict(strict=True, extra="forbid")`).
- Produces: `class WorkflowProfileResponse(_StrictResponse)` with fields `coaching: bool`, `bookends: bool`, `recipe_match: bool`, `advisor_checkpoints: bool` (wire-visible subset; **`entry_seed` is NOT included** — consumed server-side at start). `GuidedSessionResponse.profile: WorkflowProfileResponse | None = None` (`None` == empty/live-guided profile).

- [ ] **Step 1: Write the failing test for `WorkflowProfileResponse` shape + strictness.**
  Append to `tests/unit/web/sessions/test_schemas.py`:
  ```python
  def test_workflow_profile_response_wire_subset_and_strict() -> None:
      from elspeth.web.sessions.schemas import WorkflowProfileResponse

      model = WorkflowProfileResponse(
          coaching=True, bookends=True, recipe_match=True, advisor_checkpoints=True
      )
      dumped = model.model_dump()
      assert set(dumped.keys()) == {
          "coaching",
          "bookends",
          "recipe_match",
          "advisor_checkpoints",
      }

      import pydantic

      with pytest.raises(pydantic.ValidationError):
          WorkflowProfileResponse(
              coaching=True,
              bookends=True,
              recipe_match=True,
              advisor_checkpoints=True,
              entry_seed="leak",
          )

      with pytest.raises(pydantic.ValidationError):
          WorkflowProfileResponse(
              coaching="yes",
              bookends=True,
              recipe_match=True,
              advisor_checkpoints=True,
          )
  ```
- [ ] **Step 2: Run to fail.**
  `uv run pytest tests/unit/web/sessions/test_schemas.py::test_workflow_profile_response_wire_subset_and_strict -x`
  Expected: `ImportError: cannot import name 'WorkflowProfileResponse' from 'elspeth.web.sessions.schemas'`.
- [ ] **Step 3: Add `WorkflowProfileResponse`.**
  In `sessions/schemas.py`, immediately after `class ChatTurnResponse(_StrictResponse):` block (ends ~:307, before `class GuidedSessionResponse` :323), insert:
  ```python
  class WorkflowProfileResponse(_StrictResponse):
      """Wire-visible subset of a server-owned WorkflowProfile.

      Mirrors :class:`elspeth.web.composer.guided.profile.WorkflowProfile`
      MINUS ``entry_seed``. The seed is consumed server-side at
      ``POST /api/sessions/{session_id}/guided/start`` and must never ride
      the GET wire. ``None`` at the parent ``GuidedSessionResponse.profile``
      level means the empty/live-guided profile.
      """

      coaching: bool
      bookends: bool
      recipe_match: bool
      advisor_checkpoints: bool
  ```
- [ ] **Step 4: Add the `profile` field to `GuidedSessionResponse`.**
  In `sessions/schemas.py`, in `class GuidedSessionResponse(_StrictResponse):`, after `chat_turn_seq: int` (:337), add:
  ```python
      # Server-owned WorkflowProfile (wire-visible subset). ``None`` for the
      # empty/live-guided profile. Defaulted to ``None`` because most
      # GuidedSessionResponse construction sites carry the empty profile; the
      # start/GET path overrides it explicitly.
      profile: WorkflowProfileResponse | None = None
  ```
- [ ] **Step 5: Run to pass.**
  `uv run pytest tests/unit/web/sessions/test_schemas.py::test_workflow_profile_response_wire_subset_and_strict -x`
  Expected: `1 passed`.
- [ ] **Step 6: Commit.**
  `git add src/elspeth/web/sessions/schemas.py tests/unit/web/sessions/test_schemas.py && git commit -m "feat(sessions): add WorkflowProfileResponse + profile field on GuidedSessionResponse

P6.1 — wire-visible WorkflowProfile subset (coaching/bookends/recipe_match/
advisor_checkpoints; entry_seed stays server-side). Defaulted None == empty
profile.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"`

---

### Task P6.2: TS `WorkflowProfile` interface + `profile` field on TS `GuidedSession`

**Files:**
- Modify: `frontend/src/types/guided.ts` (add `WorkflowProfile` interface near :95; add `profile` to `GuidedSession` :89)

**Interfaces:**
- Produces: `export interface WorkflowProfile { coaching: boolean; bookends: boolean; recipe_match: boolean; advisor_checkpoints: boolean }`; `GuidedSession.profile: WorkflowProfile | null`.

- [ ] **Step 1: Write the failing Vitest test.**
  Append to `frontend/src/types/guided.test.ts` (create the file if absent — it is a pure type-shape assertion; see Step 3 for the import surface):
  ```typescript
  import { describe, it, expect } from "vitest";
  import type { WorkflowProfile, GuidedSession } from "@/types/guided";

  describe("WorkflowProfile wire type", () => {
    it("carries the four wire-visible boolean flags and rides GuidedSession.profile", () => {
      const profile: WorkflowProfile = {
        coaching: true,
        bookends: true,
        recipe_match: true,
        advisor_checkpoints: true,
      };
      const session: Pick<GuidedSession, "profile"> = { profile };
      expect(session.profile).not.toBeNull();
      // null is the empty/live-guided profile.
      const empty: Pick<GuidedSession, "profile"> = { profile: null };
      expect(empty.profile).toBeNull();
    });
  });
  ```
- [ ] **Step 2: Run to fail.**
  From `src/elspeth/web/frontend`: `npm test -- --run src/types/guided.test.ts`
  Expected failure: TS compile error `Module '"@/types/guided"' has no exported member 'WorkflowProfile'`.
- [ ] **Step 3: Add the `WorkflowProfile` interface.**
  In `frontend/src/types/guided.ts`, immediately before `export interface GuidedSession {` (:89), insert:
  ```typescript
  /**
   * Wire: WorkflowProfileResponse (schemas.py — WorkflowProfileResponse).
   * Server-owned workflow profile, wire-visible subset. `entry_seed` is
   * consumed server-side at POST /api/sessions/{session_id}/guided/start
   * (`/guided/start` shorthand only) and is NOT on the wire.
   * A `null` `GuidedSession.profile` is the empty/live-guided profile.
   */
  export interface WorkflowProfile {
    coaching: boolean;
    bookends: boolean;
    recipe_match: boolean;
    advisor_checkpoints: boolean;
  }
  ```
- [ ] **Step 4: Add `profile` to `GuidedSession`.**
  In `frontend/src/types/guided.ts`, in `export interface GuidedSession {`, after `chat_turn_seq: number;` (:94), add:
  ```typescript
    /** Server-owned WorkflowProfile, or `null` for the empty/live-guided profile. */
    profile: WorkflowProfile | null;
  ```
- [ ] **Step 5: Run to pass.**
  From `src/elspeth/web/frontend`: `npm test -- --run src/types/guided.test.ts`
  Expected: `1 passed`.
- [ ] **Step 6: Typecheck (guards the new optional field against existing GuidedSession literals).**
  From `src/elspeth/web/frontend`: `npm run typecheck`
  Expected: passes, OR fails at existing object-literal construction sites that build a `GuidedSession` without `profile`. If it fails there, those are mock/fixture sites — add `profile: null` to each. Re-run until clean. (`profile` is required on the TS side, mirroring the always-present wire field — `null`, never absent.)
- [ ] **Step 7: Commit.**
  `git add src/elspeth/web/frontend/src/types/guided.ts src/elspeth/web/frontend/src/types/guided.test.ts && git commit -m "feat(frontend): mirror WorkflowProfile on TS GuidedSession

P6.2 — WorkflowProfile interface (coaching/bookends/recipe_match/
advisor_checkpoints) + GuidedSession.profile: WorkflowProfile | null.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"`

---

### Task P6.3: Thread `profile` onto every `GuidedSessionResponse` construction; helper `_workflow_profile_response`

**Files:**
- Modify: `sessions/routes/_helpers.py` (add `_workflow_profile_response` helper)
- Modify: `sessions/routes/composer/guided.py` (six `GuidedSessionResponse(...)` sites: :431, :667, :881, :1200, :1668, :1851 — thread `profile=...`)

**Interfaces:**
- Consumes: `WorkflowProfile` (P0, `composer/guided/profile.py`), `EMPTY_PROFILE` (P0), `WorkflowProfileResponse` (P6.1), `GuidedSession.profile` (P0).
- Produces: `def _workflow_profile_response(guided: GuidedSession) -> WorkflowProfileResponse | None` — returns `None` when `guided.profile == EMPTY_PROFILE`, else a populated `WorkflowProfileResponse`.

- [ ] **Step 1: Write the failing unit test for the helper.**
  Append to `tests/unit/web/sessions/test_routes_split.py` (the route-helper unit suite):
  ```python
  def test_workflow_profile_response_none_for_empty_profile() -> None:
      from elspeth.web.composer.guided.profile import EMPTY_PROFILE, TUTORIAL_PROFILE
      from elspeth.web.composer.guided.state_machine import GuidedSession
      from elspeth.web.sessions.routes._helpers import _workflow_profile_response

      empty_session = GuidedSession.initial()  # default = EMPTY_PROFILE
      assert empty_session.profile == EMPTY_PROFILE
      assert _workflow_profile_response(empty_session) is None

      tutorial_session = GuidedSession.initial(profile=TUTORIAL_PROFILE)
      resp = _workflow_profile_response(tutorial_session)
      assert resp is not None
      assert resp.coaching is TUTORIAL_PROFILE.coaching
      assert resp.bookends is TUTORIAL_PROFILE.bookends
      assert resp.recipe_match is TUTORIAL_PROFILE.recipe_match
      assert resp.advisor_checkpoints is TUTORIAL_PROFILE.advisor_checkpoints
  ```
- [ ] **Step 2: Run to fail.**
  `uv run pytest tests/unit/web/sessions/test_routes_split.py::test_workflow_profile_response_none_for_empty_profile -x`
  Expected: `ImportError: cannot import name '_workflow_profile_response'`.
- [ ] **Step 3: Add the helper to `_helpers.py`.**
  Near `_initial_composition_state_with_guided_session` (:2190) add:
  ```python
  def _workflow_profile_response(guided: GuidedSession) -> WorkflowProfileResponse | None:
      """Project a GuidedSession's server-owned profile onto the wire subset.

      Returns ``None`` for the empty/live-guided profile (== ``EMPTY_PROFILE``).
      ``entry_seed`` is deliberately excluded: it is the server-side cache-key
      discriminator, not a render input.
      """
      if guided.profile == EMPTY_PROFILE:
          return None
      return WorkflowProfileResponse(
          coaching=guided.profile.coaching,
          bookends=guided.profile.bookends,
          recipe_match=guided.profile.recipe_match,
          advisor_checkpoints=guided.profile.advisor_checkpoints,
      )
  ```
  Add the imports at the top of `_helpers.py` (alongside the existing `from elspeth.web.composer.guided.*` and `from elspeth.web.sessions.schemas import ...` blocks):
  `from elspeth.web.composer.guided.profile import EMPTY_PROFILE` and ensure `WorkflowProfileResponse` is imported from `elspeth.web.sessions.schemas`. `GuidedSession` is already imported.
- [ ] **Step 4: Run to pass (helper unit).**
  `uv run pytest tests/unit/web/sessions/test_routes_split.py::test_workflow_profile_response_none_for_empty_profile -x`
  Expected: `1 passed`.
- [ ] **Step 5: Re-export `_workflow_profile_response` into `guided.py`.**
  In `guided.py`'s `from .._helpers import (` block (starts :3), add `_workflow_profile_response,` (keep alpha order near `_validate_step_indices` / `_state_response`).
- [ ] **Step 6: Thread `profile=` onto all six `GuidedSessionResponse(...)` sites.**
  At EACH of `guided.py:431, 667, 881, 1200, 1668, 1851`, the constructor builds from a local `guided` (or `new_guided`) variable. Add `profile=_workflow_profile_response(<that local>),` as the final kwarg. For the site at :667 (`post_guided_reenter`, builds from `new_guided`) use `profile=_workflow_profile_response(new_guided)`; for :881 (`new_guided`), :1200 (`guided`), etc. — match the variable already feeding `step=<x>.step.value`. (Grep `GuidedSessionResponse(` in `guided.py` to confirm the six; each is preceded by `step=<var>.step.value` — use `<var>`.)
- [ ] **Step 7: Run the existing guided-route suite to confirm no regression in the six sites.**
  `uv run pytest tests/unit/web/sessions/test_routes.py -k "guided" -q`
  Expected: all previously-passing guided route tests still pass (now also returning `profile` in the payload; `_StrictResponse` accepts it because the field exists). If any test asserts an exact `model_dump()` key set, update it to include `"profile"`.
- [ ] **Step 8: Commit.**
  `git add src/elspeth/web/sessions/routes/_helpers.py src/elspeth/web/sessions/routes/composer/guided.py tests/unit/web/sessions/test_routes_split.py && git commit -m "feat(sessions): surface WorkflowProfile on every GuidedSessionResponse

P6.3 — _workflow_profile_response (None for empty profile) threaded onto all
six GuidedSessionResponse constructors so GET/respond/chat/reenter carry profile.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"`

---

### Task P6.4: `POST /api/sessions/{session_id}/guided/start` — idempotent, closed-enum profile, persists

**Files:**
- Modify: `sessions/schemas.py` (add `StartGuidedRequest` request model)
- Modify: `sessions/routes/composer/guided.py` (add `post_guided_start` route after `post_guided_reenter` :578; auto-registers via `routes/composer/__init__.py:19`)
- Modify: `sessions/routes/_helpers.py` (add the profile `entry_seed` materializer used by `post_guided_start`)

**Interfaces:**
- Consumes: `WorkflowProfileKind` (P0 closed StrEnum `{LIVE="live", TUTORIAL="tutorial"}`), `EMPTY_PROFILE` / `TUTORIAL_PROFILE` (P0), `WorkflowProfile.entry_seed` (P0; server-owned, not wire-visible), `_initial_composition_state_with_guided_session(*, profile=...)` (P0-modified signature, `_helpers.py:2190`), `GuidedSession.profile` (P0).
- Produces: `class StartGuidedRequest(_RequestModel): profile: object = "live"` (raw boundary field accepted without echoing arbitrary input; the handler validates it is a short string closed-enum discriminator and returns a generic 400 on invalid input — mirrors the `control_signal` / `step_index` graceful-stale-client convention without leaking `entry_seed` payloads). External route `POST /api/sessions/{session_id}/guided/start` → `GetGuidedResponse`; the internal FastAPI decorator remains `@router.post("/{session_id}/guided/start", ...)` because `create_session_router()` mounts this subrouter under `/api/sessions`.
- Produces: `_initial_composition_state_with_guided_session(profile=...)` still attaches the server-owned `GuidedSession.profile`; `_materialize_profile_entry_seed_state(profile)` wraps it and, when `profile.entry_seed` is present, materializes the canonical tutorial seed/topology into the `CompositionState` before persistence. `entry_seed` is consumed server-side only and never appears in `GuidedSessionResponse.profile`, `StartGuidedRequest`, or the frontend `WorkflowProfile`.

- [ ] **Step 1: Write the failing integration test (idempotency + persistence).**
  Create `tests/unit/web/sessions/test_guided_start.py`:
  ```python
  """POST /api/sessions/{session_id}/guided/start — idempotent profile-seeded guided entry (P6, §4.3, D16)."""

  from __future__ import annotations

  import uuid
  from unittest.mock import AsyncMock, MagicMock

  import pytest
  import structlog
  from fastapi import FastAPI
  from sqlalchemy.pool import StaticPool

  from elspeth.core.payload_store import FilesystemPayloadStore
  from elspeth.web.auth.middleware import get_current_user
  from elspeth.web.auth.models import UserIdentity
  from elspeth.web.catalog.schemas import PluginSchemaInfo, PluginSummary
  from elspeth.web.config import WebSettings
  from elspeth.web.composer.progress import ComposerProgressRegistry
  from elspeth.web.middleware.rate_limit import ComposerRateLimiter
  from elspeth.web.sessions.engine import create_session_engine
  from elspeth.web.sessions.routes import create_session_router
  from elspeth.web.sessions.schema import initialize_session_schema
  from elspeth.web.sessions.service import SessionServiceImpl
  from elspeth.web.sessions.telemetry import build_sessions_telemetry
  from tests.unit.web._sync_asgi_client import SyncASGITestClient as TestClient


  def _make_app(tmp_path, user_id="alice"):
      engine = create_session_engine(
          "sqlite:///:memory:",
          poolclass=StaticPool,
          connect_args={"check_same_thread": False},
      )
      initialize_session_schema(engine)
      service = SessionServiceImpl(
          engine,
          telemetry=build_sessions_telemetry(),
          log=structlog.get_logger("test"),
      )
      app = FastAPI()
      identity = UserIdentity(user_id=user_id, username=user_id)

      async def mock_user():
          return identity

      app.dependency_overrides[get_current_user] = mock_user
      app.state.session_service = service
      app.state.session_engine = engine
      catalog = MagicMock(
          spec=["list_sources", "list_transforms", "list_sinks", "get_schema"]
      )
      catalog.list_sources.return_value = [
          PluginSummary(
              name="inline_blob",
              description="Inline blob source",
              plugin_type="source",
              config_fields=[],
          ),
      ]
      catalog.list_transforms.return_value = []
      catalog.list_sinks.return_value = []
      catalog.get_schema.return_value = PluginSchemaInfo(
          name="inline_blob",
          plugin_type="source",
          description="Inline blob source",
          json_schema={"title": "Config", "properties": {}},
          knob_schema={"fields": []},
      )
      app.state.catalog_service = catalog
      app.state.settings = WebSettings(
          data_dir=tmp_path,
          composer_max_composition_turns=15,
          composer_max_discovery_turns=10,
          composer_timeout_seconds=85.0,
          composer_rate_limit_per_minute=10,
          shareable_link_signing_key=b"\x00" * 32,
      )
      app.state.payload_store = FilesystemPayloadStore(
          app.state.settings.get_payload_store_path()
      )
      app.state.blob_service = MagicMock()
      app.state.blob_service.list_blobs = AsyncMock(return_value=[])
      app.state.blob_service.create_blob = AsyncMock()
      app.state.blob_service.get_blob = AsyncMock()
      app.state.composer_service = None
      app.state.rate_limiter = ComposerRateLimiter(limit=100)
      app.state.composer_progress_registry = ComposerProgressRegistry()
      app.include_router(create_session_router())
      return app, service


  @pytest.mark.asyncio
  async def test_guided_start_seeds_tutorial_profile_and_persists(tmp_path) -> None:
      app, service = _make_app(tmp_path)
      client = TestClient(app)
      session = await service.create_session("alice", "T", "local")

      resp = client.post(
          f"/api/sessions/{session.id}/guided/start",
          json={"profile": "tutorial"},
      )
      assert resp.status_code == 200
      body = resp.json()
      # Wire carries the tutorial profile (advisor_checkpoints on, bookends on).
      assert body["guided_session"]["profile"] is not None
      assert body["guided_session"]["profile"]["advisor_checkpoints"] is True
      assert body["guided_session"]["profile"]["bookends"] is True

      get_resp = client.get(f"/api/sessions/{session.id}/guided")
      assert get_resp.status_code == 200
      assert (
          get_resp.json()["guided_session"]["profile"]["advisor_checkpoints"]
          is True
      )


  @pytest.mark.asyncio
  async def test_guided_start_materializes_tutorial_entry_seed_topology(tmp_path) -> None:
      """Tutorial start consumes profile.entry_seed server-side and persists it.

      The client only sends {"profile": "tutorial"}. The canonical seed/topology
      comes from the SERVER-owned TUTORIAL_PROFILE.entry_seed and never rides the
      request or response wire as a raw profile object.
      """
      from elspeth.web.composer.guided.profile import TUTORIAL_PROFILE

      app, service = _make_app(tmp_path)
      client = TestClient(app)
      session = await service.create_session("alice", "T", "local")

      resp = client.post(
          f"/api/sessions/{session.id}/guided/start",
          json={"profile": "tutorial"},
      )
      assert resp.status_code == 200
      body = resp.json()
      assert "entry_seed" not in str(body["guided_session"]["profile"])

      state = body["composition_state"]
      # P0 owns the concrete entry_seed shape; P6 owns consuming it.
      assert state["sources"] == TUTORIAL_PROFILE.entry_seed.sources
      assert state["nodes"] == TUTORIAL_PROFILE.entry_seed.nodes
      assert state["edges"] == TUTORIAL_PROFILE.entry_seed.edges
      assert state["outputs"] == TUTORIAL_PROFILE.entry_seed.outputs

      persisted = await service.get_current_state(session.id)
      assert persisted is not None
      assert persisted.sources == TUTORIAL_PROFILE.entry_seed.sources
      assert persisted.nodes == TUTORIAL_PROFILE.entry_seed.nodes
      assert persisted.edges == TUTORIAL_PROFILE.entry_seed.edges
      assert persisted.outputs == TUTORIAL_PROFILE.entry_seed.outputs


  @pytest.mark.asyncio
  async def test_guided_start_is_idempotent(tmp_path) -> None:
      app, service = _make_app(tmp_path)
      client = TestClient(app)
      session = await service.create_session("alice", "T", "local")

      first = client.post(
          f"/api/sessions/{session.id}/guided/start",
          json={"profile": "tutorial"},
      )
      assert first.status_code == 200

      second = client.post(
          f"/api/sessions/{session.id}/guided/start",
          json={"profile": "live"},
      )
      assert second.status_code == 200
      assert second.json()["guided_session"]["profile"] is not None
      assert second.json()["guided_session"]["profile"]["advisor_checkpoints"] is True

      from sqlalchemy import text

      with service._engine.connect() as conn:
          versions = conn.execute(
              text("SELECT COUNT(*) FROM composition_states WHERE session_id = :sid"),
              {"sid": str(session.id)},
          ).scalar()
      assert versions == 1


  @pytest.mark.asyncio
  async def test_guided_start_rejects_existing_freeform_state_without_guided_session(tmp_path) -> None:
      """Do not silently convert or overwrite a freeform composition state."""
      from elspeth.web.sessions.protocol import CompositionStateData

      app, service = _make_app(tmp_path)
      client = TestClient(app)
      session = await service.create_session("alice", "Freeform draft", "local")
      existing = await service.save_composition_state(
          session.id,
          CompositionStateData(
              sources={"draft": {"plugin": "csv", "options": {"path": "draft.csv"}}},
              nodes={},
              edges={},
              outputs={},
              composer_meta=None,
          ),
          provenance="post_compose",
      )

      resp = client.post(
          f"/api/sessions/{session.id}/guided/start",
          json={"profile": "tutorial"},
      )
      assert resp.status_code == 409
      assert "existing freeform composition state" in resp.json()["detail"]

      persisted = await service.get_current_state(session.id)
      assert persisted is not None
      assert persisted.id == existing.id
      assert persisted.sources == {
          "draft": {"plugin": "csv", "options": {"path": "draft.csv"}}
      }
      assert persisted.composer_meta is None or "guided_session" not in persisted.composer_meta


  @pytest.mark.asyncio
  async def test_guided_start_rejects_unknown_profile_kind(tmp_path) -> None:
      app, service = _make_app(tmp_path)
      client = TestClient(app)
      session = await service.create_session("alice", "T", "local")

      resp = client.post(
          f"/api/sessions/{session.id}/guided/start",
          json={"profile": "superuser"},
      )
      assert resp.status_code == 400
      assert "profile" in resp.json()["detail"].lower()
      assert "superuser" not in resp.json()["detail"]


  @pytest.mark.asyncio
  async def test_guided_start_rejects_client_supplied_profile_object_without_echo(tmp_path) -> None:
      app, service = _make_app(tmp_path)
      client = TestClient(app)
      session = await service.create_session("alice", "T", "local")

      resp = client.post(
          f"/api/sessions/{session.id}/guided/start",
          json={
              "profile": {
                  "kind": "tutorial",
                  "entry_seed": {"sources": {"evil": "client-owned"}},
                  "advisor_checkpoints": False,
              },
          },
      )
      assert resp.status_code == 400
      detail = resp.json()["detail"]
      assert "profile" in detail.lower()
      assert "entry_seed" not in detail
      assert "evil" not in detail


  @pytest.mark.asyncio
  async def test_guided_start_unowned_session_404(tmp_path) -> None:
      app, service = _make_app(tmp_path, user_id="alice")
      client = TestClient(app)
      resp = client.post(
          f"/api/sessions/{uuid.uuid4()}/guided/start",
          json={"profile": "tutorial"},
      )
      assert resp.status_code == 404
  ```
- [ ] **Step 2: Run to fail.**
  `uv run pytest tests/unit/web/sessions/test_guided_start.py -x`
  Expected: `404 Not Found` on the start POST (route absent) → assertion error on `status_code == 200` (or a routing 405). This confirms the endpoint does not yet exist.
- [ ] **Step 3: Add `StartGuidedRequest` to `schemas.py`.**
  After `class RevertStateRequest(_RequestModel):` (~:268) add:
  ```python
  class StartGuidedRequest(_RequestModel):
      """Request body for POST /api/sessions/{session_id}/guided/start.

      ``profile`` is a raw boundary value whose valid form is a closed-enum
      discriminator (``WorkflowProfileKind``). The route validates that the
      value is a short string, maps it to the SERVER-owned WorkflowProfile
      constant, and rejects anything else with a generic 400.
      """

      profile: object = "live"
  ```
- [ ] **Step 4: Add the profile entry-seed materializer to `_helpers.py`.**
  Near `_initial_composition_state_with_guided_session` (:2190), add:
  ```python
  def _materialize_profile_entry_seed_state(profile: WorkflowProfile) -> CompositionState:
      """Build the initial guided state and consume profile.entry_seed server-side.

      ``entry_seed`` is a server-owned profile field. It may seed the canonical
      tutorial source/topology before the first guided turn, but it must never be
      accepted from the request body or appear in ``WorkflowProfileResponse``.
      """
      state = _initial_composition_state_with_guided_session(profile=profile)
      seed = profile.entry_seed
      if seed is None:
          return state
      return _replace(
          state,
          sources=seed.sources,
          nodes=seed.nodes,
          edges=seed.edges,
          outputs=seed.outputs,
          metadata=seed.metadata,
      )
  ```
  Use the concrete P0 `entry_seed` attributes verbatim. If P0 names the canonical seed
  holder differently, resolve it in `profile.py` and keep this rule unchanged:
  **only the SERVER-owned profile object supplies seed/topology material; request JSON
  never supplies profile fields beyond the discriminator.**
- [ ] **Step 5: Add the `post_guided_start` route to `guided.py`.**
  Immediately after the `post_guided_reenter` handler (starts :578, ends before `@router.post("/{session_id}/guided/respond"...)` :703) insert:
  ```python
@router.post("/{session_id}/guided/start", response_model=GetGuidedResponse)
async def post_guided_start(
    session_id: UUID,
    body: StartGuidedRequest,
    request: Request,
    user: UserIdentity = Depends(get_current_user),  # noqa: B008
) -> GetGuidedResponse:
    """Seed a guided session with a server-owned WorkflowProfile.

    The client supplies a closed-enum ``profile`` discriminator
    (``WorkflowProfileKind``); the SERVER maps it to the concrete profile
    object and persists the resulting GuidedSession, so a client cannot
    inject an arbitrary profile or weaken the advisor gate (D13/§4.3).

    **Idempotent (D16):** a second start for a session that ALREADY has a
    persisted GuidedSession returns the existing session unchanged — it
    never re-initialises or double-creates.
    GET /api/sessions/{session_id}/guided then reads the
    persisted ``GuidedSession.profile``; the lazy no-arg GET default path
    stays for live guided (empty profile).

    Raises 404 if the session does not exist or belong to the requester.
    Raises 409 if the session already has a freeform composition state with
    no GuidedSession; this route does not convert or discard freeform state.
    Raises 400 if ``profile`` is not a recognised WorkflowProfileKind or if
    a client sends anything other than a short discriminator string.
    """
    await _verify_session_ownership(session_id, user, request)
    service: SessionServiceProtocol = request.app.state.session_service
    catalog: CatalogServiceProtocol = request.app.state.catalog_service

    # Tier-3 -> Tier-2 coercion at the profile-kind boundary. A stale client
    # sending an unknown discriminator gets a 400 with a generic message
    # rather than a Pydantic 422; the typed kind then selects a SERVER-owned
    # constant — the client never supplies the profile object. Do not echo
    # the raw value: it may be a long string or an attempted profile object
    # carrying attacker-controlled fields such as entry_seed.
    if not isinstance(body.profile, str) or len(body.profile) > 32:
        raise HTTPException(
            status_code=400,
            detail=(
                "Invalid profile discriminator. "
                f"Valid values: {sorted(k.value for k in WorkflowProfileKind)}."
            ),
        )
    try:
        profile_kind = WorkflowProfileKind(body.profile)
    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=(
                "Unknown profile discriminator. "
                f"Valid values: {sorted(k.value for k in WorkflowProfileKind)}."
            ),
        ) from exc
    profile = TUTORIAL_PROFILE if profile_kind is WorkflowProfileKind.TUTORIAL else EMPTY_PROFILE

    compose_lock = await _get_session_compose_lock_registry(request).get_lock(str(session_id))
    async with compose_lock:
        # Idempotency (D16): if a guided session is already persisted, return
        # it UNCHANGED — never re-init (a second start must not clobber the
        # learner's in-progress wizard or re-seed a fresh profile).
        existing_record = await service.get_current_state(session_id)
        if existing_record is not None:
            existing_state = _state_from_record(existing_record)
            if existing_state.guided_session is not None:
                guided = existing_state.guided_session
                terminal = guided.terminal
                return GetGuidedResponse(
                    guided_session=GuidedSessionResponse(
                        step=guided.step.value,
                        history=[
                            TurnRecordResponse(
                                step=r.step.value,
                                turn_type=r.turn_type.value,
                                payload_hash=r.payload_hash,
                                response_hash=r.response_hash,
                                summary=r.summary,
                                emitter=r.emitter,
                            )
                            for r in guided.history
                        ],
                        terminal=TerminalStateResponse(
                            kind=terminal.kind.value,
                            reason=terminal.reason.value if terminal.reason is not None else None,
                            pipeline_yaml=terminal.pipeline_yaml,
                        )
                        if terminal is not None
                        else None,
                        chat_history=[
                            ChatTurnResponse(
                                role=t.role.value,
                                content=t.content,
                                seq=t.seq,
                                step=t.step.value,
                                ts_iso=t.ts_iso,
                            )
                            for t in guided.chat_history
                        ],
                        chat_turn_seq=guided.chat_turn_seq,
                        profile=_workflow_profile_response(guided),
                    ),
                    next_turn=None,
                    terminal=TerminalStateResponse(
                        kind=terminal.kind.value,
                        reason=terminal.reason.value if terminal.reason is not None else None,
                        pipeline_yaml=terminal.pipeline_yaml,
                    )
                    if terminal is not None
                    else None,
                    composition_state=_state_response(existing_record),
                )
            raise HTTPException(
                status_code=409,
                detail=(
                    "Cannot start guided on a session that already has existing "
                    "freeform composition state. Create a new session or fork before "
                    "starting the tutorial profile."
                ),
            )

        # No persisted guided session yet: construct the profile-seeded
        # initial state, consume profile.entry_seed server-side, and PERSIST
        # it (so GET /api/sessions/{session_id}/guided reads it back).
        # For the tutorial profile this
        # materializes the canonical seed/topology; for the live profile it
        # is the existing empty guided state.
        new_state = _materialize_profile_entry_seed_state(profile)
        guided = new_state.guided_session
        if guided is None:  # pragma: no cover — helper always attaches a guided session
            raise InvariantError("post_guided_start: initial state has no guided_session")
        turn = _build_get_guided_turn(new_state, guided, catalog=catalog)

        new_composer_meta = {"guided_session": guided.to_dict()}
        state_d = new_state.to_dict()
        state_data = CompositionStateData(
            sources=state_d["sources"],
            nodes=state_d["nodes"],
            edges=state_d["edges"],
            outputs=state_d["outputs"],
            metadata_=state_d["metadata"],
            is_valid=False,
            validation_errors=None,
            composer_meta=new_composer_meta,
        )
        state_record_out = await service.save_composition_state(
            session_id,
            state_data,
            # Start endpoint seeds the canonical guided session for a profile;
            # ``session_seed`` is the closest existing provenance category for
            # a fresh server-authored seed state (the closed enum has no
            # guided-specific value — see merge commit message).
            provenance="session_seed",
        )

        return GetGuidedResponse(
            guided_session=GuidedSessionResponse(
                step=guided.step.value,
                history=[
                    TurnRecordResponse(
                        step=r.step.value,
                        turn_type=r.turn_type.value,
                        payload_hash=r.payload_hash,
                        response_hash=r.response_hash,
                        summary=r.summary,
                        emitter=r.emitter,
                    )
                    for r in guided.history
                ],
                terminal=None,
                chat_history=[
                    ChatTurnResponse(
                        role=t.role.value,
                        content=t.content,
                        seq=t.seq,
                        step=t.step.value,
                        ts_iso=t.ts_iso,
                    )
                    for t in guided.chat_history
                ],
                chat_turn_seq=guided.chat_turn_seq,
                profile=_workflow_profile_response(guided),
            ),
            next_turn=TurnPayloadResponse(
                type=turn["type"],
                step_index=turn["step_index"],
                payload=dict(turn["payload"]),
            )
            if turn is not None
            else None,
            terminal=None,
            composition_state=_state_response(state_record_out),
        )
  ```
- [ ] **Step 6: Add the imports to `guided.py`.**
  In the `from .._helpers import (` block (starts :3) add `StartGuidedRequest,`, `WorkflowProfileKind,`, `WorkflowProfile,`, `EMPTY_PROFILE,`, `TUTORIAL_PROFILE,`, and `_materialize_profile_entry_seed_state,` — confirm each is re-exported by `_helpers.py`; if not yet re-exported, add `from elspeth.web.composer.guided.profile import EMPTY_PROFILE, TUTORIAL_PROFILE, WorkflowProfile, WorkflowProfileKind` to `_helpers.py` and add `from elspeth.web.sessions.schemas import StartGuidedRequest` likewise. **Do NOT add `_build_get_guided_turn`** — it is defined locally in `guided.py:91` and is already in scope; importing it from `_helpers.py` would raise `ImportError`. `_state_from_record`, `_state_response`, `InvariantError`, `CompositionStateData`, `TurnPayloadResponse`, `TerminalStateResponse` are already present in `guided.py`'s imports.
- [ ] **Step 7: Run to pass.**
  `uv run pytest tests/unit/web/sessions/test_guided_start.py -x`
  Expected: all start-route tests pass (`7 passed` unless additional local tests already exist).
- [ ] **Step 8: Lint + type the new code.**
  `uv run ruff check src/elspeth/web/sessions/routes/composer/guided.py src/elspeth/web/sessions/routes/_helpers.py src/elspeth/web/sessions/schemas.py && uv run mypy src/elspeth/web/sessions/routes/composer/guided.py src/elspeth/web/sessions/routes/_helpers.py`
  Expected: clean.
- [ ] **Step 9: Commit.**
  `git add src/elspeth/web/sessions/routes/composer/guided.py src/elspeth/web/sessions/routes/_helpers.py src/elspeth/web/sessions/schemas.py tests/unit/web/sessions/test_guided_start.py && git commit -m "feat(sessions): guided/start idempotent profile entry

P6.4 — server-constructed WorkflowProfile (LIVE/TUTORIAL), persists the seeded
GuidedSession and consumes profile.entry_seed server-side to materialize the
canonical tutorial seed/topology; idempotent re-start returns the existing
session unchanged; existing freeform state -> 409; invalid/unknown profile -> 400
without echoing raw profile input. External route:
POST /api/sessions/{session_id}/guided/start.
GET /api/sessions/{session_id}/guided reads the persisted profile.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"`

---

### Task P6.5: `client.ts` — `startGuidedSession(sessionId, profileKind)`

**Files:**
- Modify: `frontend/src/api/client.ts` (add `startGuidedSession` near the guided fns ~:589–:646)

**Interfaces:**
- Consumes: `GetGuidedResponse` (already imported from `@/types/guided`).
- Produces: `export async function startGuidedSession(sessionId: string, profileKind: "live" | "tutorial"): Promise<GetGuidedResponse>`.

- [ ] **Step 1: Write the failing Vitest test.**
  Append to `frontend/src/api/client.guided.test.ts` (the client test suite; if a guided-client test file already exists, append there — grep for `getGuided` to find it):
  ```typescript
  import { describe, it, expect, vi, afterEach } from "vitest";
  import { startGuidedSession } from "@/api/client";

  afterEach(() => vi.restoreAllMocks());

  describe("startGuidedSession", () => {
    it("POSTs the profile discriminator to the full guided-start route", async () => {
      const fetchMock = vi.spyOn(globalThis, "fetch").mockResolvedValue(
        new Response(
          JSON.stringify({
            guided_session: {
              step: "step_1_source",
              history: [],
              terminal: null,
              chat_history: [],
              chat_turn_seq: 0,
              profile: {
                coaching: true,
                bookends: true,
                recipe_match: true,
                advisor_checkpoints: true,
              },
            },
            next_turn: null,
            terminal: null,
            composition_state: null,
          }),
          { status: 200, headers: { "content-type": "application/json" } },
        ),
      );

      const result = await startGuidedSession("sess-1", "tutorial");
      expect(fetchMock).toHaveBeenCalledWith(
        "/api/sessions/sess-1/guided/start",
        expect.objectContaining({
          method: "POST",
          body: JSON.stringify({ profile: "tutorial" }),
        }),
      );
      expect(result.guided_session.profile?.advisor_checkpoints).toBe(true);
    });
  });
  ```
- [ ] **Step 2: Run to fail.**
  From `src/elspeth/web/frontend`: `npm test -- --run src/api/client.guided.test.ts`
  Expected: TS compile error `Module '"@/api/client"' has no exported member 'startGuidedSession'`.
- [ ] **Step 3: Add `startGuidedSession` to `client.ts`.**
  Immediately after the `getGuided` function (ends ~:607, before `respondGuided`), insert:
  ```typescript
  /**
   * Seed a guided session with a server-owned WorkflowProfile.
   *
   * The `profileKind` is a closed-enum discriminator ("live" | "tutorial"); the
   * SERVER constructs the concrete profile object and persists the GuidedSession.
   * Idempotent (D16): a second call for a session that already has a persisted
   * guided session returns the existing session unchanged.
   */
  export async function startGuidedSession(
    sessionId: string,
    profileKind: "live" | "tutorial",
  ): Promise<GetGuidedResponse> {
    const response = await fetch(`/api/sessions/${sessionId}/guided/start`, {
      method: "POST",
      headers: authHeaders("application/json"),
      body: JSON.stringify({ profile: profileKind }),
    });
    return parseResponse<GetGuidedResponse>(response);
  }
  ```
- [ ] **Step 4: Run to pass.**
  From `src/elspeth/web/frontend`: `npm test -- --run src/api/client.guided.test.ts`
  Expected: `1 passed` (plus pre-existing client tests still green).
- [ ] **Step 5: Typecheck.**
  From `src/elspeth/web/frontend`: `npm run typecheck`
  Expected: clean.
- [ ] **Step 6: Commit.**
  `git add src/elspeth/web/frontend/src/api/client.ts src/elspeth/web/frontend/src/api/client.guided.test.ts && git commit -m "feat(frontend): startGuidedSession client (closed-enum profile)

P6.5 — POST /api/sessions/{session_id}/guided/start with profile discriminator;
returns GetGuidedResponse.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"`

---

### Task P6.6: Fork strip — reset `GuidedSession.profile` to empty inside `fork_session`

**Files:**
- Modify: `sessions/service.py` (add module-level `_strip_guided_profile_in_meta` helper; apply at the two verbatim `composer_meta` copies `:5150` and `:5227`)

**Interfaces:**
- Consumes: `EMPTY_PROFILE` (P0, `composer/guided/profile.py`), `deep_thaw` (already imported `service.py:37`).
- Produces: `def _strip_guided_profile_in_meta(composer_meta: Mapping[str, Any] | None) -> dict[str, Any] | None` — returns a copy of `composer_meta` whose `["guided_session"]["profile"]` is replaced with `EMPTY_PROFILE.to_dict()`; passes `None`/no-guided-session through unchanged.

- [ ] **Step 1: Write the failing service-level test (covers the REAL materialised
  canonical source — proves the strip is in `fork_session`, independent of the
  blob-rewrite path).**
  Append to `tests/unit/web/sessions/test_fork.py` (inside `class TestForkSession`):
  ```python
      @pytest.mark.asyncio
      async def test_fork_strips_tutorial_profile_from_guided_session(self, service) -> None:
          """Forking a tutorial-profile guided session yields the EMPTY profile.

          Critical case (finding 10, rev 4 — CORRECTED). The canonical tutorial
          source MATERIALISES (set_pipeline from ``source.inline_blob``) to a real
          ``json`` source whose ``options`` carry ``blob_ref``
          (``composer/tools/sessions.py:425``), so the route-layer blob-rewrite save
          DOES fire (``rewritten=True``). This fixture uses that real shape on
          purpose: it proves the strip survives EVEN on the path that re-saves the
          state — because the blob-rewrite re-save preserves ``composer_meta``
          verbatim (``sessions/routes/sessions.py:479-480``) and never strips the
          profile. The strip therefore lives in ``fork_session`` (both the :5150
          persist copy and the :5227 return copy) and is independent of
          ``rewritten``. (The earlier "no blob_ref => rewritten=False" framing was a
          false premise — see the spec's two-objects ``blob_ref`` note in §5/B4.)
          """
          from elspeth.web.composer.guided.profile import EMPTY_PROFILE, TUTORIAL_PROFILE
          from elspeth.web.composer.guided.state_machine import GuidedSession

          session = await service.create_session("alice", "Tutorial", "local")
          tutorial_guided = GuidedSession.initial(profile=TUTORIAL_PROFILE)
          state = await service.save_composition_state(
              session.id,
              CompositionStateData(
                  # Materialised canonical URL source (sessions.py:420-427): a real
                  # ``json`` plugin with ``blob_ref`` in options => rewritten=True.
                  # The blob-rewrite save fires but preserves composer_meta verbatim,
                  # so the profile strip must still come from fork_session.
                  sources={
                      "urls": {
                          "plugin": "json",
                          "options": {
                              "path": "composer_blobs/canonical-url-list.json",
                              "blob_ref": "a1b2c3d4-0000-0000-0000-000000000099",
                          },
                      }
                  },
                  is_valid=True,
                  composer_meta={"guided_session": tutorial_guided.to_dict()},
              ),
              provenance="session_seed",
          )
          fork_msg = await service.add_message(
              session.id,
              "user",
              "Build this",
              composition_state_id=state.id,
              writer_principal="route_user_message",
          )

          _, _, copied_state = await service.fork_session(
              source_session_id=session.id,
              fork_message_id=fork_msg.id,
              new_message_content="Build something else",
              user_id="alice",
              auth_provider_type="local",
          )

          assert copied_state is not None
          # Returned record (the :5227 copy) carries the EMPTY profile.
          forked_guided = GuidedSession.from_dict(copied_state.composer_meta["guided_session"])
          assert forked_guided.profile == EMPTY_PROFILE
          # And it is PERSISTED that way (the :5150 copy) — re-read from the DB.
          persisted = await service.get_current_state(copied_state.session_id)
          persisted_guided = GuidedSession.from_dict(persisted.composer_meta["guided_session"])
          assert persisted_guided.profile == EMPTY_PROFILE

      @pytest.mark.asyncio
      async def test_fork_without_guided_session_passes_meta_through(self, service) -> None:
          """An ordinary (non-guided) fork is unaffected by the profile strip."""
          session = await service.create_session("alice", "Plain", "local")
          state = await service.save_composition_state(
              session.id,
              CompositionStateData(
                  sources={"s": {"plugin": "csv", "options": {"path": "x.csv"}}},
                  is_valid=True,
                  composer_meta={"repair_turns_used": 2},
              ),
              provenance="session_seed",
          )
          fork_msg = await service.add_message(
              session.id, "user", "Build", composition_state_id=state.id, writer_principal="route_user_message"
          )
          _, _, copied_state = await service.fork_session(
              source_session_id=session.id,
              fork_message_id=fork_msg.id,
              new_message_content="Build other",
              user_id="alice",
              auth_provider_type="local",
          )
          assert copied_state is not None
          # composer_meta passes through verbatim (no guided_session key to strip).
          assert copied_state.composer_meta == {"repair_turns_used": 2}
  ```
- [ ] **Step 2: Run to fail.**
  `uv run pytest tests/unit/web/sessions/test_fork.py::TestForkSession::test_fork_strips_tutorial_profile_from_guided_session -x`
  Expected: `AssertionError: assert <TUTORIAL_PROFILE> == <EMPTY_PROFILE>` — the verbatim copy currently carries the tutorial profile through.
- [ ] **Step 3: Add the `_strip_guided_profile_in_meta` helper.**
  In `sessions/service.py`, add a module-level function (near the other module helpers, after the imports / before `class SessionServiceImpl` — or just above `fork_session` if module helpers live in-class; place it module-level so it is import-testable):
  ```python
  def _strip_guided_profile_in_meta(composer_meta: Mapping[str, Any] | None) -> dict[str, Any] | None:
      """Reset a forked GuidedSession's WorkflowProfile to the empty profile.

      ``composer_meta`` is copied verbatim on fork. A tutorial profile must not
      leak into an ordinary forked session, so the strip lives inside
      ``fork_session`` where the composer_meta copies happen.
      """
      from elspeth.web.composer.guided.profile import EMPTY_PROFILE

      if composer_meta is None:
          return None
      thawed = dict(deep_thaw(composer_meta))
      guided_raw = thawed.get("guided_session")
      if not isinstance(guided_raw, dict) or "profile" not in guided_raw:
          return thawed
      guided_copy = dict(guided_raw)
      guided_copy["profile"] = EMPTY_PROFILE.to_dict()
      thawed["guided_session"] = guided_copy
      return thawed
  ```
  Confirm `Mapping` and `Any` are imported at the top of `service.py` (they are — used throughout); `deep_thaw` is imported at `:37`.
- [ ] **Step 4: Apply the strip at both verbatim copies in `fork_session`.**
  In `fork_session`, after `source_state_record` is loaded (~:4910) and before `_sync`, compute the stripped meta ONCE:
  ```python
    # Profile strip (finding 10, rev 4): never let a tutorial WorkflowProfile
    # leak into a forked session. Computed once, used by BOTH verbatim
    # composer_meta copies below (the :5150 persist copy and the :5227 return
    # copy). The route-layer blob-rewrite save preserves composer_meta
    # verbatim and never strips the profile (and is not in this service
    # method's path), so the strip must live here — independent of rewritten.
    forked_composer_meta = (
        _strip_guided_profile_in_meta(source_state_record.composer_meta)
        if source_state_record is not None
        else None
    )
  ```
  Then change the `:5150` site (inside `StatePayload(... CompositionStateData(... composer_meta=source_state_record.composer_meta))`) to `composer_meta=forked_composer_meta`, and the `:5227` site (the returned `CompositionStateRecord(... composer_meta=source_state_record.composer_meta)`) to `composer_meta=forked_composer_meta`. (Grep `composer_meta=source_state_record.composer_meta` within `fork_session` starting at :4934 to locate both — do not rely solely on line numbers.)
- [ ] **Step 5: Run to pass.**
  `uv run pytest tests/unit/web/sessions/test_fork.py -k "profile or passes_meta_through" -x`
  Expected: `2 passed`.
- [ ] **Step 6: Run the full fork suite (no regression in copy provenance / state lineage).**
  `uv run pytest tests/unit/web/sessions/test_fork.py -q`
  Expected: all pass.
- [ ] **Step 7: Lint + type.**
  `uv run ruff check src/elspeth/web/sessions/service.py && uv run mypy src/elspeth/web/sessions/service.py`
  Expected: clean.
- [ ] **Step 8: Commit.**
  `git add src/elspeth/web/sessions/service.py tests/unit/web/sessions/test_fork.py && git commit -m "fix(sessions): strip WorkflowProfile on fork (no tutorial-profile leak)

P6.6 — _strip_guided_profile_in_meta resets guided_session.profile to EMPTY at
BOTH verbatim composer_meta copies in fork_session (:5150 persist + :5227 return).
The route blob-rewrite save preserves composer_meta verbatim and never strips the
profile (and is route-layer, not part of fork_session), so the strip must live in
fork_session — independent of rewritten. Closes finding 10.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"`

---

### Task P6.7: `POST /api/sessions/{session_id}/guided/respond` optimistic-concurrency `step_index` 409 guard

**Files:**
- Modify: `sessions/schemas.py` (add optional `step_index: str | None = None` to `GuidedRespondRequest` :365)
- Modify: `frontend/src/types/guided.ts` (add `step_index?: GuidedStep | null` to `GuidedRespondRequest` :115)
- Modify: `sessions/routes/composer/guided.py` (add the 409 guard in `post_guided_respond` after the terminal-409 guard ~:914 / before the dispatcher)

**Interfaces:**
- Consumes: `GuidedStep` (already imported), the existing terminal-409 pattern (`guided.py:~914`) and the `POST /api/sessions/{session_id}/guided/chat` step-mismatch 409 pattern (`guided.py:~1394`).
- Produces: `GuidedRespondRequest.step_index: str | None = None` — when supplied, the handler coerces it to `GuidedStep` and 409s if it does not match `guided.step` (carries an expected step on the wire confirm, D16). Absent (`None`) preserves the existing no-guard behaviour (back-compat with non-wire turns that don't carry a step).

- [ ] **Step 1: Write the failing integration test (stale step_index → 409).**
  Append to `tests/unit/web/sessions/test_guided_start.py` (reuses the `_make_app` harness):
  ```python
  @pytest.mark.asyncio
  async def test_guided_respond_stale_step_index_409(tmp_path) -> None:
      app, service = _make_app(tmp_path)
      client = TestClient(app)
      session = await service.create_session("alice", "T", "local")
      client.post(f"/api/sessions/{session.id}/guided/start", json={"profile": "tutorial"})
      client.get(f"/api/sessions/{session.id}/guided")

      resp = client.post(
          f"/api/sessions/{session.id}/guided/respond",
          json={"step_index": "step_3_transforms", "chosen": ["csv"]},
      )
      assert resp.status_code == 409
      assert "step_index" in resp.json()["detail"]

  @pytest.mark.asyncio
  async def test_guided_respond_unknown_step_index_400(tmp_path) -> None:
      app, service = _make_app(tmp_path)
      client = TestClient(app)
      session = await service.create_session("alice", "T", "local")
      client.post(f"/api/sessions/{session.id}/guided/start", json={"profile": "tutorial"})
      client.get(f"/api/sessions/{session.id}/guided")

      resp = client.post(
          f"/api/sessions/{session.id}/guided/respond",
          json={"step_index": "step_99_bogus", "chosen": ["csv"]},
      )
      assert resp.status_code == 400
      assert "step_index" in resp.json()["detail"].lower()

  @pytest.mark.asyncio
  async def test_guided_respond_success_preserves_tutorial_profile(tmp_path) -> None:
      """A normal respond response still carries the persisted tutorial profile."""
      app, service = _make_app(tmp_path)
      client = TestClient(app)
      session = await service.create_session("alice", "T", "local")
      client.post(f"/api/sessions/{session.id}/guided/start", json={"profile": "tutorial"})
      client.get(f"/api/sessions/{session.id}/guided")

      resp = client.post(
          f"/api/sessions/{session.id}/guided/respond",
          json={"step_index": "step_1_source", "chosen": ["inline_blob"]},
      )
      assert resp.status_code == 200
      profile = resp.json()["guided_session"]["profile"]
      assert profile is not None
      assert profile["advisor_checkpoints"] is True
      assert profile["bookends"] is True

      get_resp = client.get(f"/api/sessions/{session.id}/guided")
      assert get_resp.status_code == 200
      assert get_resp.json()["guided_session"]["profile"]["advisor_checkpoints"] is True
  ```
- [ ] **Step 2: Run to fail.**
  `uv run pytest tests/unit/web/sessions/test_guided_start.py -k "stale_step_index or unknown_step_index" -x`
  Expected: the stale-index POST returns 200/400 (not 409) because no guard exists yet → assertion fails on `status_code == 409`. (The unknown-index test may already 422 from Pydantic if the field is absent — confirming the field must be added.)
- [ ] **Step 3: Add the optional `step_index` to `GuidedRespondRequest`.**
  In `sessions/schemas.py`, in `class GuidedRespondRequest(_RequestModel):` (:365), after `control_signal: str | None = None` (:384), add:
  ```python
      # Optimistic-concurrency token (D16): the client's expected current step.
      # When present, the route 409s if it does not match the session's live
      # ``guided.step``. A plain ``str`` (not the enum) makes unknown values fail
      # with a route-handler 400 rather than a Pydantic 422.
      step_index: str | None = None
  ```
- [ ] **Step 4: Add the 409 guard in `post_guided_respond`.**
  In `guided.py`, in `post_guided_respond` (starts :703), after the generic terminal-409 guard (`if guided.terminal is not None: raise HTTPException(409...)` ~:914) and before `current_step = guided.step`, insert:
  ```python
            # Optimistic-concurrency guard (D16): if the client carried an
            # expected step, reject a mismatch with 409 (the wizard advanced
            # under the client between read and write) — the same guard
            # ``POST /api/sessions/{session_id}/guided/chat`` already has
            # (guided.py:~1394). A stale client
            # sending an unknown value gets a 400, not a Pydantic 422,
            # mirroring control_signal. ``None`` (field absent) skips the
            # guard for turns that do not carry an expected step.
            if body.step_index is not None:
                try:
                    expected_step = GuidedStep(body.step_index)
                except ValueError as exc:
                    raise HTTPException(
                        status_code=400,
                        detail=(
                            f"Unknown step_index {body.step_index!r}. "
                            f"Valid values: {sorted(s.value for s in GuidedStep)}."
                        ),
                    ) from exc
                if expected_step is not guided.step:
                    raise HTTPException(
                        status_code=409,
                        detail=(
                            f"step_index {expected_step.value!r} does not match the session's "
                            f"current step {guided.step.value!r}. Re-fetch GET "
                            f"/api/sessions/{{id}}/guided and retry."
                        ),
                    )
  ```
- [ ] **Step 5: Run to pass (backend).**
  `uv run pytest tests/unit/web/sessions/test_guided_start.py -k "step_index" -x`
  Expected: `2 passed`. Also confirm no regression: `uv run pytest tests/unit/web/sessions/test_routes.py -k "guided" -q` → all pass (existing respond callers omit `step_index`, so the guard is skipped).
- [ ] **Step 5b: Run the profile-preservation respond success test.**
  `uv run pytest tests/unit/web/sessions/test_guided_start.py::test_guided_respond_success_preserves_tutorial_profile -x`
  Expected: `1 passed`; proves P6.3's `GuidedSessionResponse` profile projection survives the successful `POST /api/sessions/{session_id}/guided/respond` construction path after tutorial start.
- [ ] **Step 6: Mirror `step_index` on the TS `GuidedRespondRequest`.**
  In `frontend/src/types/guided.ts`, in `export interface GuidedRespondRequest {` (:115), after `control_signal: ControlSignal | null;` (:122), add:
  ```typescript
    /**
     * Optimistic-concurrency token: the client's expected current step. When
     * present the server 409s on mismatch (the wizard advanced under the
     * client). Optional — omit for non-wire turns that don't carry a step.
     */
    step_index?: GuidedStep | null;
  ```
- [ ] **Step 7: Frontend typecheck + the mirror gate.**
  From `src/elspeth/web/frontend`: `npm run typecheck` (expected clean).
  From the repo root: `uv run python scripts/cicd/check_slot_type_cross_language.py` (the SlotType / guided.ts mirror gate — this work edits `guided.ts`; expected: passes, since `step_index` is a request field, not a `SlotType`/`TurnType`/`GuidedStep` enum member).
- [ ] **Step 8: Commit.**
  `git add src/elspeth/web/sessions/schemas.py src/elspeth/web/sessions/routes/composer/guided.py src/elspeth/web/frontend/src/types/guided.ts tests/unit/web/sessions/test_guided_start.py && git commit -m "feat(sessions): optimistic-concurrency step_index 409 on guided respond

P6.7 — optional step_index on GuidedRespondRequest; route 409s on mismatch
(wizard advanced under client), 400 on unknown value — parity with guided chat.
TS mirror added. Closes D16.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"`

---

### Task P6.8: Phase P6 verification sweep

**Files:** none (verification only).

**Interfaces:** none.

- [ ] **Step 1: Run the full P6 backend test set.**
  `uv run pytest tests/unit/web/sessions/test_guided_start.py tests/unit/web/sessions/test_fork.py tests/unit/web/sessions/test_schemas.py tests/unit/web/sessions/test_routes_split.py tests/unit/web/sessions/test_routes.py -q`
  Expected: all pass, zero failures.
- [ ] **Step 2: Ruff check + format-check on every touched file.**
  `uv run ruff check src/ tests/ && uv run ruff format --check src/elspeth/web/sessions/routes/composer/guided.py src/elspeth/web/sessions/routes/_helpers.py src/elspeth/web/sessions/service.py src/elspeth/web/sessions/schemas.py`
  Expected: `All checks passed!` / no files would be reformatted. (If format-check flags a file, run `uv run ruff format <file>` and re-commit.)
- [ ] **Step 3: mypy on the touched modules.**
  `uv run mypy src/elspeth/web/sessions/routes/composer/guided.py src/elspeth/web/sessions/routes/_helpers.py src/elspeth/web/sessions/service.py src/elspeth/web/sessions/schemas.py`
  Expected: `Success: no issues found`.
- [ ] **Step 4: elspeth-lints trust gates over the touched surface.**
  `PYTHONPATH=elspeth-lints/src uv run python -m elspeth_lints.core.cli check --rules trust_tier.tier_model,trust_boundary.tests,trust_boundary.scope,trust_boundary.tier,'composer/*' --root src/elspeth`
  Expected: no NEW findings attributable to P6. The start endpoint validates the profile discriminator against a closed enum and the SERVER constructs the profile object (no Tier-3 profile object crosses the boundary); `step_index` is coerced through `GuidedStep(...)` with a 400 on failure (a clean Tier-3→Tier-2 coercion). If a finding lands on a P6 line, fix at the boundary; do not waiver. (Pre-existing fingerprint_baseline drift is operator-owned — state once, do not bless.)
- [ ] **Step 5: Frontend gates.**
  From `src/elspeth/web/frontend`: `npm run typecheck && npm test -- --run && npm run build`
  Expected: all green.
- [ ] **Step 6: SlotType / guided.ts mirror gate + wardline.**
  Repo root: `uv run python scripts/cicd/check_slot_type_cross_language.py` (expected pass) and `wardline scan . --fail-on ERROR` (expected exit 0; P6 touches the profile/start trust boundary — confirm the closed-enum discriminator + server-constructed profile keeps the taint flow clean; fix at the boundary if a finding lands).
- [ ] **Step 7: Final commit (only if Steps 2/3 required a formatting/type touch-up; otherwise skip).**
  `git add src/elspeth/web/sessions/routes/composer/guided.py src/elspeth/web/sessions/routes/_helpers.py src/elspeth/web/sessions/service.py src/elspeth/web/sessions/schemas.py src/elspeth/web/frontend/src/types/guided.ts src/elspeth/web/frontend/src/api/client.ts tests/unit/web/sessions/test_guided_start.py tests/unit/web/sessions/test_fork.py tests/unit/web/sessions/test_schemas.py tests/unit/web/sessions/test_routes_split.py tests/unit/web/sessions/test_routes.py src/elspeth/web/frontend/src/types/guided.test.ts src/elspeth/web/frontend/src/api/client.guided.test.ts && git commit -m "chore(sessions): P6 verification sweep — ruff/mypy/lints/frontend green

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"`

---
