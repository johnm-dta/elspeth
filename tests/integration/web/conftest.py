"""Shared helpers for Phase 1 web integration tests.

Duplicates ``_make_session`` from
``tests/unit/web/conftest.py``; if either copy changes the
other must be updated to match. Engine fixtures are NOT shared because
integration tests own their engine fixture and call ``_make_session``
against whatever connection they have.
"""

from __future__ import annotations

import asyncio
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from queue import Empty, Queue
from threading import Event, Thread
from uuid import UUID, uuid4

import anyio
import pytest
import structlog
from asgi_lifespan import LifespanManager
from fastapi import FastAPI, HTTPException
from sqlalchemy import Connection, event, insert
from sqlalchemy.engine import Engine
from sqlalchemy.exc import OperationalError
from sqlalchemy.pool import StaticPool

from elspeth.web.auth.middleware import get_current_user
from elspeth.web.auth.models import UserIdentity
from elspeth.web.composer.state import (
    CompositionState,
    NodeSpec,
    OutputSpec,
    PipelineMetadata,
    SourceSpec,
)
from elspeth.web.config import WebSettings
from elspeth.web.middleware.rate_limit import ComposerRateLimiter
from elspeth.web.sessions import models
from elspeth.web.sessions.engine import create_session_engine
from elspeth.web.sessions.protocol import CompositionStateData
from elspeth.web.sessions.routes import create_session_router
from elspeth.web.sessions.schema import initialize_session_schema
from elspeth.web.sessions.service import SessionServiceImpl
from elspeth.web.sessions.telemetry import build_sessions_telemetry
from tests.unit.web._sync_asgi_client import SyncASGITestClient as TestClient


def _make_session(
    conn: Connection,
    *,
    session_id: str,
    user_id: str = "test_user",
    auth_provider_type: str = "local",
    title: str = "test session",
    created_at: datetime | None = None,
    updated_at: datetime | None = None,
) -> None:
    """Insert a session row with every NOT NULL column populated."""
    now = created_at or datetime.now(UTC)
    conn.execute(
        insert(models.sessions_table).values(
            id=session_id,
            user_id=user_id,
            auth_provider_type=auth_provider_type,
            title=title,
            created_at=now,
            updated_at=updated_at or now,
        )
    )


@pytest.fixture
def composer_test_client(tmp_path: Path) -> TestClient:
    """FastAPI ``TestClient`` configured for generic compose-route tests."""

    engine = create_session_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    initialize_session_schema(engine)
    session_service = SessionServiceImpl(
        engine,
        data_dir=tmp_path,
        telemetry=build_sessions_telemetry(),
        log=structlog.get_logger("test.phase3.integration"),
    )
    app = FastAPI()
    identity = UserIdentity(user_id="alice", username="alice")

    async def mock_user() -> UserIdentity:
        return identity

    app.dependency_overrides[get_current_user] = mock_user
    app.state.session_service = session_service
    app.state.session_engine = engine
    app.state.settings = WebSettings(
        data_dir=tmp_path,
        composer_max_composition_turns=15,
        composer_max_discovery_turns=10,
        composer_timeout_seconds=85.0,
        composer_rate_limit_per_minute=10,
        shareable_link_signing_key=b"\x00" * 32,
    )
    app.state.composer_service = None
    app.state.rate_limiter = ComposerRateLimiter(limit=100)
    app.state.execution_service = None
    app.state.composer_progress_registry = None
    app.state.scoped_secret_resolver = None
    app.include_router(create_session_router())
    client = TestClient(app)
    client.app.state.phase3_engine = engine
    client.app.state.phase3_sessions_service = session_service
    return client


@pytest.fixture
def session_with_pending_compose_request(composer_test_client: TestClient) -> dict[str, str]:
    """Session with owner and user message ready for a compose request."""

    session_id = str(uuid4())
    user_message_id = str(uuid4())
    now = datetime.now(UTC)
    engine = composer_test_client.app.state.phase3_engine
    with engine.begin() as conn:
        _make_session(conn, session_id=session_id, user_id="alice")
        conn.execute(
            insert(models.chat_messages_table).values(
                id=user_message_id,
                session_id=session_id,
                role="user",
                content="Build a pipeline",
                raw_content=None,
                tool_calls=None,
                tool_call_id=None,
                sequence_no=1,
                writer_principal="route_user_message",
                created_at=now,
                composition_state_id=None,
                parent_assistant_id=None,
            )
        )
    return {"session_id": session_id, "user_message_id": user_message_id}


@pytest.fixture
def session_with_composer_state(session_with_pending_compose_request: dict[str, str]) -> dict[str, str]:
    """Session fixture used by failed-turn route tests."""

    return session_with_pending_compose_request


# ---------------------------------------------------------------------------
# Audit-readiness integration fixtures (Phase 2A Task 5)
# ---------------------------------------------------------------------------
#
# These fixtures stand up the full ELSPETH web app via ``create_app`` so the
# audit-readiness routes exercise the real validate_pipeline / scoped secret
# resolver / sessions paths.  ``_lifespan_test_client`` runs the FastAPI
# lifespan so ``app.state.execution_service`` and
# ``app.state.readiness_service`` are constructed.
#
# Auth is bypassed by overriding ``get_current_user``: the auth provider is
# still created by ``create_app`` but the token-validation path is never
# exercised, matching ``test_preferences_routes.py``'s ``client_anonymous``
# pattern.
#
# Cross-user IDOR fixture: ``audit_readiness_other_user_session_id``
# inserts a session for ``user_id="bob"`` on the SAME engine the
# alice-authenticated TestClient hits, with a saved composition state.  The
# ownership check returns 404 because of the user_id mismatch, not because
# the row is absent (the route would otherwise 404 for either reason; we
# want the mismatch path specifically).

_TEST_AUTHED_USER_ID = "alice"


def _build_audit_readiness_app(
    tmp_path: Path,
    *,
    authed_user_id: str | None,
) -> FastAPI:
    """Construct a real FastAPI app for audit-readiness route tests.

    ``authed_user_id`` of ``None`` installs an override that raises
    ``HTTPException(401)`` — the anonymous fixture variant.
    """
    from elspeth.web.app import create_app

    settings = WebSettings(
        data_dir=tmp_path,
        landscape_url=f"sqlite:///{tmp_path}/runs/audit.db",
        payload_store_path=tmp_path / "payloads",
        composer_max_composition_turns=15,
        composer_max_discovery_turns=10,
        composer_timeout_seconds=85.0,
        composer_rate_limit_per_minute=10,
        shareable_link_signing_key=b"\x00" * 32,
        plugin_allowlist=("transform:passthrough",),
    )
    app = create_app(settings=settings)

    if authed_user_id is None:

        async def _unauthenticated() -> UserIdentity:
            raise HTTPException(status_code=401, detail="Not authenticated")

        app.dependency_overrides[get_current_user] = _unauthenticated
    else:
        identity = UserIdentity(user_id=authed_user_id, username=authed_user_id)

        async def _mock_user() -> UserIdentity:
            return identity

        app.dependency_overrides[get_current_user] = _mock_user

    return app


@contextmanager
def _lifespan_test_client(app: FastAPI) -> Iterator[TestClient]:
    """Run FastAPI lifespan without Starlette's portal-sensitive TestClient."""

    ready: Queue[LifespanManager | BaseException] = Queue(maxsize=1)
    finished: Queue[BaseException | None] = Queue(maxsize=1)
    stop = Event()
    ready_sent = False

    async def _run_lifespan() -> None:
        nonlocal ready_sent
        try:
            # asgi_lifespan's default startup_timeout is 5s, which is too tight
            # for the full ELSPETH app's lifespan startup under concurrent-CI
            # runner contention (the push + pull_request CI runs execute the
            # same job in parallel on shared runners). 5s produced intermittent
            # fixture-setup TimeoutErrors. This inner timeout is the PRIMARY
            # startup guard; it must stay strictly below the outer ``ready.get``
            # backstop below so its specific asgi_lifespan TimeoutError is the
            # one that surfaces, not the generic "thread wedged" backstop.
            async with LifespanManager(app, startup_timeout=15) as manager:
                ready.put(manager)
                ready_sent = True
                while not stop.is_set():
                    await anyio.sleep(0.05)
        except BaseException as exc:
            if not ready_sent:
                ready.put(exc)
                ready_sent = True
            finished.put(exc)
        else:
            finished.put(None)

    def _thread_target() -> None:
        anyio.run(_run_lifespan)

    thread = Thread(target=_thread_target, name="audit-readiness-lifespan", daemon=True)
    thread.start()
    try:
        # Backstop for a wholly-wedged lifespan thread that never puts a result
        # (manager or exception) onto the queue. Kept strictly above the inner
        # ``startup_timeout`` (15s) so the inner asgi_lifespan TimeoutError wins
        # the race on a slow-but-not-wedged startup; this only fires if the
        # thread produces nothing at all.
        started = ready.get(timeout=20)
    except Empty as exc:
        stop.set()
        thread.join(timeout=5)
        raise TimeoutError("FastAPI lifespan did not start within 20 seconds") from exc

    if isinstance(started, BaseException):
        thread.join(timeout=5)
        raise started

    try:
        with TestClient(app, transport_app=started.app) as client:
            yield client
    finally:
        stop.set()
        thread.join(timeout=10)
        if thread.is_alive():
            raise TimeoutError("FastAPI lifespan did not shut down within 10 seconds")
        try:
            error = finished.get_nowait()
        except Empty:
            error = None
        if error is not None:
            raise error


def _passthrough_composition_state(data_dir: Path, session_id: UUID) -> CompositionState:
    """Build a source → passthrough → sink composition.

    The passthrough node MUST trigger ``identity_node_advisory`` in
    ``execution/validation.py`` so the audit-readiness panel's
    ``provenance`` row surfaces as ``status == "warning"`` with a
    populated ``component_ids`` tuple (C1 guard).

    Source / sink paths are anchored to the session-owned subtrees under
    ``data_dir/blobs`` and ``data_dir/outputs`` respectively so they satisfy
    ``allowed_source_directories`` / ``allowed_sink_directories`` in
    ``web/paths.py`` — otherwise validation reports a path-traversal
    error and the happy-path advisory never emits.
    """
    source_path = str(data_dir / "blobs" / str(session_id) / "audit_readiness_fixture.csv")
    sink_path = str(data_dir / "outputs" / str(session_id) / "audit_readiness_fixture_out.csv")
    return CompositionState(
        source=SourceSpec(
            plugin="csv",
            on_success="src_out",
            # schema={mode: observed} satisfies the CSV plugin's
            # required schema field without anchoring a fixed contract;
            # combined with an absent schema block on the passthrough
            # and an observed sink, this is the canonical identity-shaped
            # repro that fires identity_node_advisory.
            options={"path": source_path, "schema": {"mode": "observed"}},
            # "discard" avoids needing a sink named "quarantine"; this
            # is purely a fixture pragmatism choice and does not affect
            # the path the advisory check exercises.
            on_validation_failure="discard",
        ),
        nodes=(
            NodeSpec(
                id="pass",
                node_type="transform",
                plugin="passthrough",
                input="src_out",
                on_success="out",
                # on_error must be a sink name (or "discard") — the YAML
                # generator hard-fails on None, mirroring the
                # mutation-boundary default contract.
                on_error="discard",
                # schema={mode: observed} (no fields) is the canonical
                # identity-shaped case from
                # test_passthrough_with_schema_mode_only_is_flagged in
                # tests/unit/web/execution/test_identity_node_advisory.py:
                # the plugin's required-schema constraint is satisfied
                # but Rule 5's fields anchor is absent, so the advisory
                # still fires.
                options={"schema": {"mode": "observed"}},
                condition=None,
                routes=None,
                fork_to=None,
                branches=None,
                policy=None,
                merge=None,
            ),
        ),
        edges=(),
        outputs=(
            OutputSpec(
                name="out",
                plugin="csv",
                # Sink uses observed schema mode so the advisory's
                # sink_schema_mode field is reported, completing the
                # canonical repro from
                # tests/unit/web/execution/test_identity_node_advisory.py.
                options={"path": sink_path, "schema": {"mode": "observed"}},
                on_write_failure="discard",
            ),
        ),
        metadata=PipelineMetadata(
            name="audit-readiness fixture",
            description="Fixture used by audit-readiness route tests.",
        ),
        version=1,
    )


def _seed_session_with_state(
    client: TestClient,
    *,
    user_id: str,
) -> UUID:
    """Create a session and persist the passthrough composition state.

    Runs against the live ``app.state.session_service`` so the route
    handler later observes the same DB.
    """
    session_service = client.app.state.session_service
    settings: WebSettings = client.app.state.settings

    # Ensure path-allowlisted directories exist so source/sink option
    # paths inside the persisted CompositionState satisfy
    # web/paths.py's resolve_data_path() invariants.
    async def _seed() -> UUID:
        record = await session_service.create_session(
            user_id=user_id,
            title="audit-readiness fixture",
            auth_provider_type=settings.auth_provider,
        )
        (settings.data_dir / "blobs" / str(record.id)).mkdir(parents=True, exist_ok=True)
        (settings.data_dir / "outputs" / str(record.id)).mkdir(parents=True, exist_ok=True)
        state = _passthrough_composition_state(settings.data_dir, record.id)
        state_d = state.to_dict()
        await session_service.save_composition_state(
            record.id,
            CompositionStateData(
                sources=state_d["sources"],
                nodes=state_d["nodes"],
                edges=state_d["edges"],
                outputs=state_d["outputs"],
                metadata_=state_d["metadata"],
                is_valid=True,
                validation_errors=None,
            ),
            provenance="session_seed",
        )
        return record.id

    return asyncio.run(_seed())


def _seed_session_without_state(
    client: TestClient,
    *,
    user_id: str,
) -> UUID:
    """Create a session but DO NOT persist any composition state."""
    session_service = client.app.state.session_service
    settings: WebSettings = client.app.state.settings

    async def _seed() -> UUID:
        record = await session_service.create_session(
            user_id=user_id,
            title="audit-readiness empty fixture",
            auth_provider_type=settings.auth_provider,
        )
        return record.id

    return asyncio.run(_seed())


@pytest.fixture
def audit_readiness_test_client(tmp_path: Path) -> Iterator[TestClient]:
    """Full app with audit-readiness routes wired; auth bypassed to ``alice``."""
    app = _build_audit_readiness_app(tmp_path, authed_user_id=_TEST_AUTHED_USER_ID)
    with _lifespan_test_client(app) as client:
        yield client


@pytest.fixture
def audit_readiness_client_with_state(
    audit_readiness_test_client: TestClient,
) -> tuple[TestClient, UUID]:
    """Client + a session owned by ``alice`` with a passthrough composition.

    The passthrough triggers ``identity_node_advisory`` so the
    audit-readiness panel's ``provenance`` row is ``status="warning"``
    with a non-empty ``component_ids`` (C1 integration guard).
    """
    session_id = _seed_session_with_state(
        audit_readiness_test_client,
        user_id=_TEST_AUTHED_USER_ID,
    )
    return audit_readiness_test_client, session_id


@pytest.fixture
def audit_readiness_client_without_state(
    audit_readiness_test_client: TestClient,
) -> tuple[TestClient, UUID]:
    """Client + a session owned by ``alice`` with no composition state.

    The audit-readiness routes must return 404 on this session because
    ``get_current_state`` returns ``None``.  Ownership check passes
    first; the 404 comes from the missing-state branch.
    """
    session_id = _seed_session_without_state(
        audit_readiness_test_client,
        user_id=_TEST_AUTHED_USER_ID,
    )
    return audit_readiness_test_client, session_id


@pytest.fixture
def audit_readiness_client_anonymous(tmp_path: Path) -> Iterator[TestClient]:
    """Full app + ``get_current_user`` override raising ``HTTPException(401)``.

    The real ``get_current_user`` reads ``app.state.auth_audit_recorder``
    and ``app.state.settings`` before the Authorization header is
    consulted; the override sidesteps that machinery so the test
    asserts route-layer auth-required behaviour without standing up
    the full auth surface.
    """
    app = _build_audit_readiness_app(tmp_path, authed_user_id=None)
    with _lifespan_test_client(app) as client:
        yield client


@pytest.fixture
def audit_readiness_other_user_session_id(
    audit_readiness_test_client: TestClient,
) -> UUID:
    """A session_id owned by ``bob`` (not the authenticated ``alice``).

    Used for the IDOR guard: ``alice``'s GET against ``bob``'s session
    MUST return 404 (not 403) so session existence is not leaked.  The
    session has a persisted composition state so the route would
    otherwise be able to compute a snapshot — the user_id mismatch
    branch in ``verify_session_ownership`` is what produces the 404.
    """
    return _seed_session_with_state(
        audit_readiness_test_client,
        user_id="bob",
    )


def _seed_session_with_mismatched_auth_provider(
    client: TestClient,
    *,
    user_id: str,
    auth_provider_type: str,
) -> UUID:
    """Seed a session whose ``auth_provider_type`` differs from
    ``settings.auth_provider``.

    Used by the second-comparator IDOR test. The session is owned by
    ``user_id`` (typically the authenticated user) so the user_id
    branch of the ownership check passes — only the
    ``auth_provider_type`` branch flips, isolating that comparator.
    """
    session_service = client.app.state.session_service
    settings: WebSettings = client.app.state.settings

    async def _seed() -> UUID:
        record = await session_service.create_session(
            user_id=user_id,
            title="audit-readiness mismatched-provider fixture",
            auth_provider_type=auth_provider_type,  # type: ignore[arg-type]
        )
        (settings.data_dir / "blobs" / str(record.id)).mkdir(parents=True, exist_ok=True)
        (settings.data_dir / "outputs" / str(record.id)).mkdir(parents=True, exist_ok=True)
        state = _passthrough_composition_state(settings.data_dir, record.id)
        state_d = state.to_dict()
        await session_service.save_composition_state(
            record.id,
            CompositionStateData(
                sources=state_d["sources"],
                nodes=state_d["nodes"],
                edges=state_d["edges"],
                outputs=state_d["outputs"],
                metadata_=state_d["metadata"],
                is_valid=True,
                validation_errors=None,
            ),
            provenance="session_seed",
        )
        return record.id

    return asyncio.run(_seed())


@pytest.fixture
def audit_readiness_mismatched_provider_session_id(
    audit_readiness_test_client: TestClient,
) -> UUID:
    """A session owned by ``alice`` but bound to ``auth_provider_type="oidc"``.

    The test client authenticates as ``alice`` with
    ``settings.auth_provider == "local"`` (the default). The user_id
    branch of ``verify_session_ownership`` passes; only the
    ``auth_provider_type`` branch flips the access decision, isolating
    that comparator for an IDOR-branch regression test.
    """
    return _seed_session_with_mismatched_auth_provider(
        audit_readiness_test_client,
        user_id=_TEST_AUTHED_USER_ID,
        auth_provider_type="oidc",
    )


# ---------------------------------------------------------------------------
# Shareable-reviews integration fixtures (Phase 6A Task 6)
# ---------------------------------------------------------------------------
#
# Shareable-reviews tests reuse the audit-readiness app harness (same
# ``create_app`` path, same passthrough composition fixture). For the
# "recipient is not the creator" test, the alice-authed client mints a
# token, then the test swaps ``app.dependency_overrides[get_current_user]``
# to return ``bob`` and re-issues the GET — same app, same signing key,
# same payload store. No additional fixture needed.


@pytest.fixture
def inject_commit_OperationalError() -> object:
    """Integration-scope one-shot SQLAlchemy COMMIT failure hook."""

    def _install(engine: Engine) -> None:
        def _raise(_conn: object) -> None:
            event.remove(engine, "commit", _raise)
            raise OperationalError("COMMIT", {}, RuntimeError("phase3 commit failure"))

        event.listen(engine, "commit", _raise)

    return _install
