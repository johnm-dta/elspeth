"""Integration tests for tutorial backend routes."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID, uuid4

import pytest
import structlog
from fastapi import FastAPI
from pydantic import SecretBytes
from sqlalchemy.pool import StaticPool

from elspeth.web.auth.middleware import get_current_user
from elspeth.web.auth.models import UserIdentity
from elspeth.web.composer.tutorial_run_routes import create_tutorial_run_router
from elspeth.web.composer.tutorial_service import tutorial_model_id
from elspeth.web.config import WebSettings
from elspeth.web.middleware.rate_limit import ComposerRateLimiter
from elspeth.web.preferences.routes import create_preferences_router
from elspeth.web.preferences.service import PreferencesService
from elspeth.web.preferences.tutorial_cache import CANONICAL_SEED_PROMPT, TutorialCache, TutorialCacheEntry, tutorial_cache_key
from elspeth.web.sessions.audit_story_service import AuditStoryIntegrityError
from elspeth.web.sessions.engine import create_session_engine
from elspeth.web.sessions.protocol import CompositionStateData
from elspeth.web.sessions.routes import create_session_router
from elspeth.web.sessions.schema import initialize_session_schema
from elspeth.web.sessions.service import SessionServiceImpl
from elspeth.web.sessions.telemetry import build_sessions_telemetry
from tests.integration.web.conftest import _make_session
from tests.unit.web._sync_asgi_client import SyncASGITestClient as TestClient


def _settings(tmp_path: Path) -> WebSettings:
    (tmp_path / "runs").mkdir(exist_ok=True)
    return WebSettings(
        data_dir=tmp_path,
        landscape_url=f"sqlite:///{tmp_path}/runs/audit.db",
        composer_model="gpt-5.5",
        composer_max_composition_turns=15,
        composer_max_discovery_turns=10,
        composer_timeout_seconds=85.0,
        composer_rate_limit_per_minute=10,
        shareable_link_signing_key=SecretBytes(b"\x00" * 32),
    )


def _app(tmp_path: Path) -> FastAPI:
    engine = create_session_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    initialize_session_schema(engine)
    settings = _settings(tmp_path)
    session_service = SessionServiceImpl(
        engine,
        data_dir=tmp_path,
        telemetry=build_sessions_telemetry(),
        log=structlog.get_logger("test.tutorial"),
    )

    app = FastAPI()
    app.state.settings = settings
    app.state.session_engine = engine
    app.state.session_service = session_service
    app.state.preferences_service = PreferencesService(engine)
    tutorial_cache_dir = settings.tutorial_cache_dir
    assert tutorial_cache_dir is not None
    app.state.tutorial_cache = TutorialCache(cache_dir=tutorial_cache_dir)
    app.state.rate_limiter = ComposerRateLimiter(limit=settings.composer_rate_limit_per_minute)

    identity = UserIdentity(user_id="alice", username="alice")

    async def _mock_user() -> UserIdentity:
        return identity

    app.dependency_overrides[get_current_user] = _mock_user
    app.include_router(create_preferences_router())
    app.include_router(create_tutorial_run_router())
    app.include_router(create_session_router())
    return app


def _seed_session_with_state(app: FastAPI, *, match_canonical_cache: bool = False) -> UUID:
    """Seed a session with a composition state.

    When ``match_canonical_cache`` is True, the state's source/transforms/
    outputs topology mirrors the canonical cached tutorial pipeline. The
    cache-replay path requires structural match before attaching cached
    pipeline_yaml + rows to the session's state_id; an unmatched state
    causes fall-through to live compose (per the P2-2 fix in
    ``_state_matches_cached_topology``).
    """
    session_id = uuid4()
    with app.state.session_engine.begin() as conn:
        _make_session(conn, session_id=str(session_id), user_id="alice")
    if match_canonical_cache:
        state_data = CompositionStateData(
            source={"plugin": "null"},
            nodes=[
                {"id": "keep", "node_type": "transform", "plugin": "passthrough"},
            ],
            outputs=[{"name": "out", "plugin": "json"}],
        )
    else:
        state_data = CompositionStateData()
    asyncio.run(
        app.state.session_service.save_composition_state(
            session_id,
            state_data,
            provenance="session_seed",
        )
    )
    return session_id


def _seed_canonical_cache(app: FastAPI) -> str:
    # Use the same compound model_id helper the production path uses. Going
    # through ``settings.composer_model`` directly would seed at a key the
    # service no longer looks up, masking the real cache-hit path.
    model_id = tutorial_model_id(app.state.settings)
    key = tutorial_cache_key(CANONICAL_SEED_PROMPT, model_id)
    app.state.tutorial_cache.store(
        TutorialCacheEntry(
            canonical_prompt=CANONICAL_SEED_PROMPT,
            model_id=model_id,
            cached_at=datetime(2026, 5, 15, tzinfo=UTC),
            rows=[{"url": "ato.gov.au", "score": 5, "rationale": "clear"}],
            source_data_hash="a7f3e2cached",
            llm_call_count=5,
            # Production composer YAML shape (yaml_generator.py): source is
            # a single dict, transforms is list[dict], sinks is dict-keyed.
            pipeline_yaml=(
                "source:\n  plugin: 'null'\n"
                "transforms:\n"
                "  - name: keep\n"
                "    plugin: passthrough\n"
                "    input: source\n"
                "    on_success: out\n"
                "    on_error: abort\n"
                "sinks:\n  out:\n    plugin: json\n"
            ),
        )
    )
    return key


def test_post_run_cache_hit_creates_current_session_run_and_audit_story(tmp_path: Path) -> None:
    app = _app(tmp_path)
    cache_key = _seed_canonical_cache(app)
    # State topology must match the cached pipeline for the P2-2 guard to
    # permit replay (otherwise cache hit falls through to live).
    session_id = _seed_session_with_state(app, match_canonical_cache=True)
    client = TestClient(app)

    response = client.post(
        "/api/tutorial/run",
        json={"session_id": str(session_id), "prompt": CANONICAL_SEED_PROMPT},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["seeded_from_cache"] is True
    assert body["cache_key"] == cache_key
    assert body["output"]["source_data_hash"] == "a7f3e2cached"

    story = client.get(f"/api/sessions/{session_id}/runs/{body['run_id']}/audit-story")
    assert story.status_code == 200
    story_body = story.json()
    assert story_body["run_id"] == body["run_id"]
    assert story_body["session_id"] == str(session_id)
    assert story_body["llm_call_count"] == 0
    assert story_body["seeded_from_cache"] is True
    assert story_body["cache_key"] == cache_key


def test_post_run_rate_limiter_blocks_cache_miss_before_live_execution(tmp_path: Path) -> None:
    app = _app(tmp_path)
    app.state.rate_limiter = ComposerRateLimiter(limit=1)
    _seed_canonical_cache(app)
    cache_hit_session_id = _seed_session_with_state(app, match_canonical_cache=True)
    cache_miss_session_id = _seed_session_with_state(app)

    class _ExplodingExecutionService:
        def __init__(self) -> None:
            self.calls = 0

        async def execute(self, *_args: object, **_kwargs: object) -> UUID:
            self.calls += 1
            raise AssertionError("live tutorial execution should have been throttled before execute()")

    execution_service = _ExplodingExecutionService()
    app.state.execution_service = execution_service
    client = TestClient(app)

    cache_hit = client.post(
        "/api/tutorial/run",
        json={"session_id": str(cache_hit_session_id), "prompt": CANONICAL_SEED_PROMPT},
    )
    assert cache_hit.status_code == 200
    assert cache_hit.json()["seeded_from_cache"] is True

    throttled = client.post(
        "/api/tutorial/run",
        json={"session_id": str(cache_miss_session_id), "prompt": "run the tutorial live"},
    )

    assert throttled.status_code == 429
    assert throttled.json()["detail"]["error_type"] == "rate_limited"
    assert execution_service.calls == 0


def test_audit_story_missing_audit_db_raises_without_creating_file(tmp_path: Path) -> None:
    """A recorded landscape_run_id with no audit DB on disk is a named
    Tier-1 breach — and the read path must not create the database.

    Pins the slice-1 behavior change: the old bare ``LandscapeDB(...)``
    writer constructor silently CREATED an empty audit DB (create_all +
    epoch stamp) and then 500'd with "run not found". Now the route raises
    ``AuditStoryIntegrityError`` before touching disk.
    """
    app = _app(tmp_path)
    _seed_canonical_cache(app)
    session_id = _seed_session_with_state(app, match_canonical_cache=True)
    client = TestClient(app)

    response = client.post(
        "/api/tutorial/run",
        json={"session_id": str(session_id), "prompt": CANONICAL_SEED_PROMPT},
    )
    assert response.status_code == 200
    run_id = response.json()["run_id"]

    audit_db = tmp_path / "runs" / "audit.db"
    assert audit_db.exists()
    for suffix in ("", "-wal", "-shm"):
        Path(f"{audit_db}{suffix}").unlink(missing_ok=True)

    # The bare test app registers no exception handlers, so the named error
    # propagates to the client (production maps it to a structured 500 with
    # the ``error_type`` discriminator via the app.py handler).
    with pytest.raises(AuditStoryIntegrityError, match="audit database does not exist"):
        client.get(f"/api/sessions/{session_id}/runs/{run_id}/audit-story")
    assert not audit_db.exists(), "read path must not recreate the audit DB"


def test_post_run_cache_topology_mismatch_does_not_replay_cache(tmp_path: Path) -> None:
    """P2-2: cache hit with state topology mismatch must NOT attach cached output to the wrong state.

    A client posts the canonical prompt against a session whose composition
    state describes a different pipeline (csv source, no transforms, csv sink)
    than the cached tutorial (null source, passthrough transform, json sink).
    The cache lookup hits, but the topology check refuses replay; the request
    falls through to the live path. The Tier-1 invariant: a 200 cache-replay
    response (``seeded_from_cache=True``) MUST NOT be returned in this case
    — that would be the audit lie the P2-2 review found.
    """
    app = _app(tmp_path)
    _seed_canonical_cache(app)
    # Seed a state whose plugin topology disagrees with the cached pipeline.
    session_id = uuid4()
    with app.state.session_engine.begin() as conn:
        _make_session(conn, session_id=str(session_id), user_id="alice")
    asyncio.run(
        app.state.session_service.save_composition_state(
            session_id,
            CompositionStateData(
                source={"plugin": "csv"},
                nodes=[],
                outputs=[{"name": "out", "plugin": "csv"}],
            ),
            provenance="session_seed",
        )
    )
    client = TestClient(app)

    # The live path requires ``execution_service`` which this fixture
    # intentionally does not wire (the integration scope here is the
    # tutorial-run + audit-story slice, not the full execution backbone).
    # What this test asserts is the *negative* invariant: the cache-replay
    # branch must NOT silently take over when topology mismatches. The
    # audit lie the P2-2 review caught is a 200 with
    # ``seeded_from_cache=True`` attaching cached rows to the wrong state.
    # Fall-through to live presents as 500 (missing execution_service) —
    # that's correct behaviour for this test surface; the cache replay
    # would have been the wrong outcome.
    try:
        response = client.post(
            "/api/tutorial/run",
            json={"session_id": str(session_id), "prompt": CANONICAL_SEED_PROMPT},
        )
    except AttributeError as exc:
        # Live path's app.state.execution_service is the missing dep; fall-
        # through reached it, confirming cache replay was correctly skipped.
        assert "execution_service" in str(exc)
        return

    assert response.json().get("seeded_from_cache") is not True, (
        "Cache replay must refuse to attach cached output to a state whose topology does not match the cached pipeline — Tier-1 audit lie."
    )


def test_post_run_unknown_session_returns_404(tmp_path: Path) -> None:
    app = _app(tmp_path)
    _seed_canonical_cache(app)
    client = TestClient(app)

    response = client.post(
        "/api/tutorial/run",
        json={"session_id": str(uuid4()), "prompt": CANONICAL_SEED_PROMPT},
    )

    assert response.status_code == 404


def test_post_run_rejects_extra_fields(tmp_path: Path) -> None:
    app = _app(tmp_path)
    session_id = _seed_session_with_state(app)
    client = TestClient(app)

    response = client.post(
        "/api/tutorial/run",
        json={"session_id": str(session_id), "prompt": CANONICAL_SEED_PROMPT, "rogue": True},
    )

    assert response.status_code == 422


def test_delete_orphans_soft_renames_incomplete_tutorial_sessions(tmp_path: Path) -> None:
    app = _app(tmp_path)
    orphan_id = uuid4()
    keep_id = uuid4()
    with app.state.session_engine.begin() as conn:
        _make_session(conn, session_id=str(orphan_id), user_id="alice", title="hello-world (cool government pages)")
        _make_session(conn, session_id=str(keep_id), user_id="alice", title="ordinary session")
    client = TestClient(app)

    response = client.delete("/api/tutorial/orphans")

    assert response.status_code == 200
    assert response.json() == {"deleted_count": 1}
    renamed = asyncio.run(app.state.session_service.get_session(orphan_id))
    kept = asyncio.run(app.state.session_service.get_session(keep_id))
    assert renamed.title.startswith("abandoned-hello-world (cool government pages)-")
    assert kept.title == "ordinary session"
