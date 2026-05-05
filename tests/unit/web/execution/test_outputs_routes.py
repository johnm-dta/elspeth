"""Tests for the /api/runs/{rid}/outputs manifest + content endpoints.

Mirrors the structure of test_routes.py — bypasses real DB setup via
mocks on app.state.* and dependency_overrides for the auth middleware.

The manifest endpoint is the authoritative full-list of every sink-write
artefact produced by a run (distinct from the diagnostics endpoint's
20-artifact preview). The content endpoint streams the bytes of one
artefact, gated by a path-allowlist guard that enforces
``allowed_sink_directories(data_dir)``.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from starlette.routing import Route

from elspeth.web.auth.models import UserIdentity
from elspeth.web.execution.schemas import (
    RunOutputArtifact,
    RunOutputsResponse,
    RunStatusResponse,
)

_TEST_USER_ID = "test-user-123"


def _route_endpoint(app: FastAPI, name: str) -> Callable[..., Awaitable[Any]]:
    for route in app.routes:
        if isinstance(route, Route) and route.name == name:
            return cast(Callable[..., Awaitable[Any]], route.endpoint)
    raise AssertionError(f"Route endpoint {name!r} not found")


def _create_test_app(
    execution_service: MagicMock | None = None,
    settings: MagicMock | None = None,
) -> FastAPI:
    from elspeth.web.auth.middleware import get_current_user
    from elspeth.web.execution.routes import create_execution_router

    app = FastAPI()
    app.state.execution_service = execution_service or MagicMock()
    app.state.broadcaster = MagicMock()
    app.state.auth_provider = MagicMock()

    mock_session_service = MagicMock()
    mock_session = MagicMock()
    mock_session.user_id = _TEST_USER_ID
    mock_session.auth_provider_type = "local"
    mock_session_service.get_session = AsyncMock(return_value=mock_session)
    mock_session_service.get_run = AsyncMock(return_value=MagicMock(session_id=uuid4()))
    app.state.session_service = mock_session_service

    if settings is None:
        settings = MagicMock()
        settings.auth_provider = "local"
    app.state.settings = settings

    fake_user = UserIdentity(user_id=_TEST_USER_ID, username="testuser")

    async def _fake_current_user() -> UserIdentity:
        return fake_user

    app.dependency_overrides[get_current_user] = _fake_current_user
    app.include_router(create_execution_router())
    return app


def _running_status(run_id: uuid4) -> RunStatusResponse:
    return RunStatusResponse(
        run_id=str(run_id),
        status="running",
        started_at=datetime.now(UTC),
        finished_at=None,
        rows_processed=0,
        rows_succeeded=0,
        rows_failed=0,
        rows_routed_success=0,
        rows_routed_failure=0,
        rows_quarantined=0,
        error=None,
        landscape_run_id=None,
    )


# ── Manifest endpoint tests ─────────────────────────────────────────


class TestRunOutputsManifestEndpoint:
    """GET /api/runs/{run_id}/outputs"""

    @pytest.mark.asyncio
    async def test_returns_full_artifact_list_no_preview_cap(self, monkeypatch) -> None:
        run_id = uuid4()
        svc = MagicMock()
        svc.get_status = AsyncMock(return_value=_running_status(run_id))

        artifacts = [
            RunOutputArtifact(
                artifact_id=f"art-{i:02d}",
                sink_node_id=f"sink_{i}",
                artifact_type="file",
                path_or_uri=f"/data/outputs/sink_{i}.csv",
                content_hash="a" * 64,
                size_bytes=100 + i,
                created_at=datetime.now(UTC),
                exists_now=True,
            )
            for i in range(25)
        ]

        def fake_load(*args: object, **kwargs: object) -> RunOutputsResponse:
            return RunOutputsResponse(
                run_id=str(run_id),
                landscape_run_id=str(run_id),
                artifacts=artifacts,
            )

        async def fake_to_thread(func, /, *args, **kwargs):
            return func(*args, **kwargs)

        monkeypatch.setattr("elspeth.web.execution.routes.load_run_outputs_for_settings", fake_load)
        monkeypatch.setattr("elspeth.web.execution.routes.asyncio.to_thread", fake_to_thread)

        app = _create_test_app(execution_service=svc)
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.get(f"/api/runs/{run_id}/outputs")

        assert response.status_code == 200
        body = response.json()
        assert body["run_id"] == str(run_id)
        assert len(body["artifacts"]) == 25, "manifest must NOT apply diagnostics preview cap"

    @pytest.mark.asyncio
    async def test_404_when_run_not_found(self, monkeypatch) -> None:
        run_id = uuid4()
        svc = MagicMock()
        svc.get_status = AsyncMock(side_effect=ValueError("not found"))

        app = _create_test_app(execution_service=svc)
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.get(f"/api/runs/{run_id}/outputs")
        assert response.status_code == 404


# ── Content streaming endpoint tests ────────────────────────────────


class TestRunOutputContentEndpoint:
    """GET /api/runs/{run_id}/outputs/{artifact_id}/content"""

    @pytest.mark.asyncio
    async def test_streams_file_bytes_when_inside_sink_allowlist(self, monkeypatch, tmp_path) -> None:
        run_id = uuid4()
        outputs_dir = tmp_path / "outputs"
        outputs_dir.mkdir()
        sink_file = outputs_dir / "results.jsonl"
        sink_file.write_bytes(b'{"interaction_id":"INT-1001"}\n')

        svc = MagicMock()
        svc.get_status = AsyncMock(return_value=_running_status(run_id))

        def fake_load(*args: object, **kwargs: object) -> RunOutputsResponse:
            return RunOutputsResponse(
                run_id=str(run_id),
                landscape_run_id=str(run_id),
                artifacts=[
                    RunOutputArtifact(
                        artifact_id="art-1",
                        sink_node_id="results",
                        artifact_type="file",
                        path_or_uri=f"file://{sink_file}",
                        content_hash="a" * 64,
                        size_bytes=sink_file.stat().st_size,
                        created_at=datetime.now(UTC),
                        exists_now=True,
                    )
                ],
            )

        async def fake_to_thread(func, /, *args, **kwargs):
            return func(*args, **kwargs)

        monkeypatch.setattr("elspeth.web.execution.routes.load_run_outputs_for_settings", fake_load)
        monkeypatch.setattr("elspeth.web.execution.routes.asyncio.to_thread", fake_to_thread)

        settings = MagicMock()
        settings.auth_provider = "local"
        settings.data_dir = str(tmp_path)

        app = _create_test_app(execution_service=svc, settings=settings)
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.get(f"/api/runs/{run_id}/outputs/art-1/content")

        assert response.status_code == 200
        assert response.content == b'{"interaction_id":"INT-1001"}\n'

    @pytest.mark.asyncio
    async def test_403_when_path_outside_sink_allowlist(self, monkeypatch, tmp_path) -> None:
        run_id = uuid4()
        # File lives OUTSIDE data_dir/{outputs,blobs} — must be refused.
        elsewhere = tmp_path / "elsewhere"
        elsewhere.mkdir()
        rogue_file = elsewhere / "rogue.csv"
        rogue_file.write_bytes(b"escaped\n")

        svc = MagicMock()
        svc.get_status = AsyncMock(return_value=_running_status(run_id))

        def fake_load(*args: object, **kwargs: object) -> RunOutputsResponse:
            return RunOutputsResponse(
                run_id=str(run_id),
                landscape_run_id=str(run_id),
                artifacts=[
                    RunOutputArtifact(
                        artifact_id="art-rogue",
                        sink_node_id="rogue",
                        artifact_type="file",
                        path_or_uri=str(rogue_file),
                        content_hash="b" * 64,
                        size_bytes=rogue_file.stat().st_size,
                        created_at=datetime.now(UTC),
                        exists_now=True,
                    )
                ],
            )

        async def fake_to_thread(func, /, *args, **kwargs):
            return func(*args, **kwargs)

        monkeypatch.setattr("elspeth.web.execution.routes.load_run_outputs_for_settings", fake_load)
        monkeypatch.setattr("elspeth.web.execution.routes.asyncio.to_thread", fake_to_thread)

        settings = MagicMock()
        settings.auth_provider = "local"
        # data_dir does not contain elsewhere/, so rogue_file is outside
        # allowed_sink_directories(data_dir) = (data_dir/outputs, data_dir/blobs).
        settings.data_dir = str(tmp_path)

        app = _create_test_app(execution_service=svc, settings=settings)
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.get(f"/api/runs/{run_id}/outputs/art-rogue/content")

        assert response.status_code == 403
        assert response.json()["detail"]["error_type"] == "output_path_outside_allowlist"

    @pytest.mark.asyncio
    async def test_404_when_artifact_id_not_in_run(self, monkeypatch, tmp_path) -> None:
        run_id = uuid4()
        svc = MagicMock()
        svc.get_status = AsyncMock(return_value=_running_status(run_id))

        def fake_load(*args: object, **kwargs: object) -> RunOutputsResponse:
            return RunOutputsResponse(
                run_id=str(run_id),
                landscape_run_id=str(run_id),
                artifacts=[],
            )

        async def fake_to_thread(func, /, *args, **kwargs):
            return func(*args, **kwargs)

        monkeypatch.setattr("elspeth.web.execution.routes.load_run_outputs_for_settings", fake_load)
        monkeypatch.setattr("elspeth.web.execution.routes.asyncio.to_thread", fake_to_thread)

        settings = MagicMock()
        settings.auth_provider = "local"
        settings.data_dir = str(tmp_path)

        app = _create_test_app(execution_service=svc, settings=settings)
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.get(f"/api/runs/{run_id}/outputs/art-missing/content")

        assert response.status_code == 404
        assert response.json()["detail"]["error_type"] == "artifact_not_found"

    @pytest.mark.asyncio
    async def test_410_when_artifact_path_no_longer_exists(self, monkeypatch, tmp_path) -> None:
        run_id = uuid4()
        outputs_dir = tmp_path / "outputs"
        outputs_dir.mkdir()
        sink_file = outputs_dir / "results.jsonl"
        sink_file.write_bytes(b"will-purge\n")
        sink_file.unlink()  # File was purged after the run

        svc = MagicMock()
        svc.get_status = AsyncMock(return_value=_running_status(run_id))

        def fake_load(*args: object, **kwargs: object) -> RunOutputsResponse:
            return RunOutputsResponse(
                run_id=str(run_id),
                landscape_run_id=str(run_id),
                artifacts=[
                    RunOutputArtifact(
                        artifact_id="art-purged",
                        sink_node_id="results",
                        artifact_type="file",
                        path_or_uri=str(sink_file),
                        content_hash="a" * 64,
                        size_bytes=10,
                        created_at=datetime.now(UTC),
                        exists_now=False,
                    )
                ],
            )

        async def fake_to_thread(func, /, *args, **kwargs):
            return func(*args, **kwargs)

        monkeypatch.setattr("elspeth.web.execution.routes.load_run_outputs_for_settings", fake_load)
        monkeypatch.setattr("elspeth.web.execution.routes.asyncio.to_thread", fake_to_thread)

        settings = MagicMock()
        settings.auth_provider = "local"
        settings.data_dir = str(tmp_path)

        app = _create_test_app(execution_service=svc, settings=settings)
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.get(f"/api/runs/{run_id}/outputs/art-purged/content")

        assert response.status_code == 410
        assert response.json()["detail"]["error_type"] == "artifact_purged_or_moved"
