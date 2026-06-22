"""Phase 8 Sub-task 7c (telemetry-backfill: phase-6) — route-level
counter emit at ``GET /api/sessions/{session_id}/state/yaml``.

The route inserts a ``composer_completion_events_table`` audit row
with ``event_type="export_yaml"`` and, AFTER the engine.begin() block
exits cleanly, emits ``composer.session.completed_total`` with
``completion_verb="export_yaml"``.

Tests:
- Happy path: successful export emits exactly one counter increment
  AND the audit row is present in the DB (superset rule contract).
- Negative path: when the runtime preflight fails and the route
  returns 409 BEFORE the audit insert, the counter stays at zero
  (audit primacy: no audit row → no counter increment).

Q10 isolation: every test gets a fresh app + fresh telemetry container
via ``_make_app``.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest
import structlog
from fastapi import FastAPI
from sqlalchemy import select
from sqlalchemy.pool import StaticPool

from elspeth.web.auth.middleware import get_current_user
from elspeth.web.auth.models import UserIdentity
from elspeth.web.config import WebSettings
from elspeth.web.execution.schemas import ValidationError, ValidationReadiness, ValidationResult
from elspeth.web.middleware.rate_limit import ComposerRateLimiter
from elspeth.web.sessions.engine import create_session_engine
from elspeth.web.sessions.models import composer_completion_events_table
from elspeth.web.sessions.protocol import CompositionStateData
from elspeth.web.sessions.routes import create_session_router
from elspeth.web.sessions.schema import initialize_session_schema
from elspeth.web.sessions.service import SessionServiceImpl
from elspeth.web.sessions.telemetry import _FakeCounter, build_sessions_telemetry, observed_value
from tests.unit.web._sync_asgi_client import SyncASGITestClient as TestClient


def _ready_readiness() -> ValidationReadiness:
    return ValidationReadiness(authoring_valid=True, execution_ready=True, completion_ready=True, blockers=[])


def _blocked_readiness() -> ValidationReadiness:
    return ValidationReadiness(authoring_valid=False, execution_ready=False, completion_ready=False, blockers=[])


def _make_app_with_telemetry(tmp_path: Path) -> tuple[FastAPI, SessionServiceImpl]:
    """Build a FastAPI app exposing the session routes plus a fresh
    ``sessions_telemetry`` container on ``app.state``.

    Mirrors ``tests/unit/web/sessions/test_routes.py::_make_app`` but
    additionally sets ``app.state.sessions_telemetry`` so the
    ``export_yaml`` emit at routes.py reads a non-None container.
    A future refactor MAY unify the two factories — for now the
    duplication is small and self-contained.
    """
    engine = create_session_engine(
        "sqlite:///:memory:",
        poolclass=StaticPool,
        connect_args={"check_same_thread": False},
    )
    initialize_session_schema(engine)
    telemetry = build_sessions_telemetry()
    service = SessionServiceImpl(
        engine,
        telemetry=telemetry,
        log=structlog.get_logger("test.phase8.subtask7c"),
    )

    app = FastAPI()

    identity = UserIdentity(user_id="alice", username="alice")

    async def mock_user() -> UserIdentity:
        return identity

    app.dependency_overrides[get_current_user] = mock_user
    app.state.session_service = service
    app.state.session_engine = engine
    # Phase 8 Sub-task 7c — the route emits via
    # ``request.app.state.sessions_telemetry``.
    app.state.sessions_telemetry = telemetry
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
    app.state.composer_progress_registry = None
    app.state.scoped_secret_resolver = None
    app.state.execution_service = None

    app.include_router(create_session_router())
    return app, service


def _csv_state() -> CompositionStateData:
    return CompositionStateData(
        source={
            "plugin": "csv",
            "on_success": "out",
            "options": {"path": "/data.csv"},
            "on_validation_failure": "quarantine",
        },
        outputs=[
            {
                "name": "out",
                "plugin": "csv",
                "options": {"path": "outputs/out.csv", "schema": {"mode": "observed"}},
                "on_write_failure": "discard",
            }
        ],
        metadata_={"name": "Test Pipeline", "description": ""},
        is_valid=False,
    )


@pytest.mark.asyncio
async def test_export_yaml_route_emits_completion_counter(tmp_path: Path) -> None:
    """Successful ``GET /state/yaml`` writes the export_yaml audit row
    AND increments ``composer.session.completed_total`` exactly once
    with ``completion_verb="export_yaml"``.
    """
    app, service = _make_app_with_telemetry(tmp_path)
    telemetry = app.state.sessions_telemetry
    client = TestClient(app)

    session = await service.create_session("alice", "Pipeline", "local")
    await service.save_composition_state(
        session.id,
        _csv_state(),
        provenance="session_seed",
    )

    async def _pass_preflight(state, *, settings, secret_service, user_id):  # type: ignore[no-untyped-def]
        del state, settings, secret_service, user_id
        return ValidationResult(is_valid=True, checks=[], errors=[], readiness=_ready_readiness())

    # Baseline.
    assert observed_value(telemetry.session_completed_total) == 0

    with patch("elspeth.web.sessions.routes.composer.state._runtime_preflight_for_state", side_effect=_pass_preflight):
        resp = client.get(f"/api/sessions/{session.id}/state/yaml")

    assert resp.status_code == 200, resp.text

    # Counter incremented exactly once with the DB-vocabulary verb.
    assert observed_value(telemetry.session_completed_total) == 1
    counter = telemetry.session_completed_total
    assert isinstance(counter, _FakeCounter)
    assert counter.calls == [
        (1, {"completion_verb": "export_yaml"}, None),
    ]

    # Audit row IS in the DB (the counter aggregates over THIS row).
    engine = app.state.session_engine
    with engine.connect() as conn:
        rows = conn.execute(
            select(composer_completion_events_table).where(
                composer_completion_events_table.c.session_id == str(session.id),
            )
        ).all()
    assert len(rows) == 1
    assert rows[0].event_type == "export_yaml"
    assert rows[0].actor == "alice"


@pytest.mark.asyncio
async def test_export_yaml_route_runtime_preflight_failure_does_not_emit(tmp_path: Path) -> None:
    """When runtime preflight returns ``is_valid=False`` the route raises
    409 BEFORE the audit insert runs. The counter MUST stay at zero —
    no audit row was written, so the superset rule forbids a counter
    tick. (This is the audit-primacy structural guarantee at the
    route layer: the preflight gate sits BEFORE engine.begin(), so a
    failed preflight skips both the audit row AND the counter.)
    """
    app, service = _make_app_with_telemetry(tmp_path)
    telemetry = app.state.sessions_telemetry
    client = TestClient(app)

    session = await service.create_session("alice", "Pipeline", "local")
    await service.save_composition_state(
        session.id,
        _csv_state(),
        provenance="session_seed",
    )

    async def _fail_preflight(state, *, settings, secret_service, user_id):  # type: ignore[no-untyped-def]
        del state, settings, secret_service, user_id
        return ValidationResult(
            is_valid=False,
            checks=[],
            errors=[
                ValidationError(
                    component_id="src",
                    component_type="source",
                    message="simulated preflight failure",
                    suggestion=None,
                    error_code=None,
                )
            ],
            readiness=_blocked_readiness(),
        )

    with patch("elspeth.web.sessions.routes.composer.state._runtime_preflight_for_state", side_effect=_fail_preflight):
        resp = client.get(f"/api/sessions/{session.id}/state/yaml")

    assert resp.status_code == 409, resp.text

    # No audit row → no counter increment.
    assert observed_value(telemetry.session_completed_total) == 0
    engine = app.state.session_engine
    with engine.connect() as conn:
        rows = conn.execute(
            select(composer_completion_events_table).where(
                composer_completion_events_table.c.session_id == str(session.id),
            )
        ).all()
    assert len(rows) == 0


@pytest.mark.asyncio
async def test_export_yaml_route_missing_session_does_not_emit(tmp_path: Path) -> None:
    """A GET against a non-existent session returns 404 without
    touching the audit row OR the counter. Belt-and-braces case for
    audit primacy on the 'no session, no audit row' branch.
    """
    app, _service = _make_app_with_telemetry(tmp_path)
    telemetry = app.state.sessions_telemetry
    client = TestClient(app)

    bogus_session_id = "00000000-0000-0000-0000-000000000000"
    resp = client.get(f"/api/sessions/{bogus_session_id}/state/yaml")

    assert resp.status_code in {403, 404}, resp.text
    assert observed_value(telemetry.session_completed_total) == 0
