from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import structlog
from fastapi import FastAPI
from sqlalchemy.pool import StaticPool

from elspeth.web.auth.middleware import get_current_user
from elspeth.web.auth.models import UserIdentity
from elspeth.web.composer.progress import ComposerProgressRegistry
from elspeth.web.composer.state import CompositionState, OutputSpec, PipelineMetadata, SourceSpec
from elspeth.web.config import WebSettings
from elspeth.web.dependencies import create_catalog_service
from elspeth.web.execution.schemas import ValidationReadiness, ValidationResult
from elspeth.web.middleware.rate_limit import ComposerRateLimiter
from elspeth.web.plugin_policy.models import PluginAvailabilitySnapshot
from elspeth.web.plugin_policy.profiles import OperatorProfileRegistry
from elspeth.web.sessions.engine import create_session_engine
from elspeth.web.sessions.routes import create_session_router
from elspeth.web.sessions.schema import initialize_session_schema
from elspeth.web.sessions.service import SessionServiceImpl
from elspeth.web.sessions.telemetry import build_sessions_telemetry
from tests.unit.web._sync_asgi_client import SyncASGITestClient as TestClient


def _make_app(
    tmp_path: Path,
    *,
    e2e_state_seed_enabled: bool = False,
    user_id: str = "alice",
) -> tuple[FastAPI, SessionServiceImpl]:
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
        log=structlog.get_logger("test.e2e_state_seed"),
    )
    app = FastAPI()
    identity = UserIdentity(user_id=user_id, username=user_id)

    async def mock_user() -> UserIdentity:
        return identity

    app.dependency_overrides[get_current_user] = mock_user
    app.state.session_service = service
    app.state.session_engine = engine
    app.state.sessions_telemetry = telemetry
    app.state.settings = WebSettings(
        data_dir=tmp_path,
        composer_max_composition_turns=15,
        composer_max_discovery_turns=10,
        composer_timeout_seconds=85.0,
        composer_rate_limit_per_minute=10,
        shareable_link_signing_key=b"\x00" * 32,
        e2e_state_seed_enabled=e2e_state_seed_enabled,
    )
    app.state.composer_service = None
    app.state.rate_limiter = ComposerRateLimiter(limit=100)
    app.state.execution_service = None
    app.state.composer_progress_registry = ComposerProgressRegistry()
    app.state.scoped_secret_resolver = None
    catalog = create_catalog_service()
    snapshot = PluginAvailabilitySnapshot.for_trained_operator(catalog)
    app.state.catalog_service = catalog
    app.state.operator_profile_registry = MagicMock(spec=OperatorProfileRegistry)
    app.state.plugin_snapshot_factory = lambda _user: snapshot
    app.include_router(create_session_router())
    return app, service


def _ready_readiness() -> ValidationReadiness:
    return ValidationReadiness(authoring_valid=True, execution_ready=True, completion_ready=True, blockers=[])


def _valid_state(tmp_path: Path, *, session_id: str) -> CompositionState:
    blob_ref = "00000000-0000-4000-8000-000000000001"
    source_path = tmp_path / "blobs" / session_id / f"{blob_ref}_input.csv"
    source_path.parent.mkdir(parents=True, exist_ok=True)
    source_path.write_text("id\n1\n")
    return CompositionState(
        source=SourceSpec(
            plugin="csv",
            on_success="out",
            options={"path": str(source_path), "blob_ref": blob_ref},
            on_validation_failure="discard",
        ),
        nodes=(),
        edges=(),
        outputs=(
            OutputSpec(
                name="out",
                plugin="csv",
                options={"path": "outputs/out.csv", "schema": {"mode": "observed"}},
                on_write_failure="discard",
            ),
        ),
        metadata=PipelineMetadata(name="Seeded E2E", description="seeded by test route"),
        version=7,
    )


@pytest.mark.asyncio
async def test_e2e_state_seed_route_is_disabled_by_default(tmp_path: Path) -> None:
    app, service = _make_app(tmp_path)
    client = TestClient(app)
    session = await service.create_session("alice", "Seed", "local")

    response = client.post(
        f"/api/sessions/{session.id}/state/e2e-seed",
        json={"state": _valid_state(tmp_path, session_id=str(session.id)).to_dict()},
    )

    assert response.status_code == 404
    assert await service.get_current_state(session.id) is None


@pytest.mark.asyncio
async def test_e2e_state_seed_route_disabled_rejects_before_body_validation(tmp_path: Path) -> None:
    app, service = _make_app(tmp_path)
    client = TestClient(app)
    session = await service.create_session("alice", "Seed", "local")

    response = client.post(
        f"/api/sessions/{session.id}/state/e2e-seed",
        json={},
    )

    assert response.status_code == 404
    assert await service.get_current_state(session.id) is None


@pytest.mark.asyncio
async def test_e2e_state_seed_route_persists_canonical_state(tmp_path: Path) -> None:
    app, service = _make_app(tmp_path, e2e_state_seed_enabled=True)
    client = TestClient(app)
    session = await service.create_session("alice", "Seed", "local")
    seeded = _valid_state(tmp_path, session_id=str(session.id))

    async def _pass_preflight(*_args, **_kwargs):
        return ValidationResult(is_valid=True, checks=[], errors=[], readiness=_ready_readiness())

    with patch("elspeth.web.sessions.routes._helpers._runtime_preflight_for_state", side_effect=_pass_preflight):
        response = client.post(
            f"/api/sessions/{session.id}/state/e2e-seed",
            json={"state": seeded.to_dict()},
        )

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["version"] == 1
    assert body["metadata"]["name"] == "Seeded E2E"
    assert body["sources"]["source"]["plugin"] == "csv"
    assert body["outputs"][0]["name"] == "out"
    assert body["is_valid"] is True

    record = await service.get_current_state(session.id)
    assert record is not None
    assert record.version == 1
    assert record.metadata_ == {"name": "Seeded E2E", "description": "seeded by test route"}
    assert record.sources["source"]["plugin"] == "csv"


@pytest.mark.asyncio
async def test_e2e_state_seed_route_rejects_invalid_state_json(tmp_path: Path) -> None:
    app, service = _make_app(tmp_path, e2e_state_seed_enabled=True)
    client = TestClient(app)
    session = await service.create_session("alice", "Seed", "local")

    response = client.post(
        f"/api/sessions/{session.id}/state/e2e-seed",
        json={"state": {"version": 1}},
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "Invalid composition state JSON"
    assert await service.get_current_state(session.id) is None
