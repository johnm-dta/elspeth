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

import hashlib
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock
from uuid import UUID, uuid4

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from httpx import ASGITransport, AsyncClient
from starlette.routing import Route

from elspeth.web.auth.models import UserIdentity
from elspeth.web.execution.outputs import RunOutputsAuditUnavailableError
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
    mock_run = MagicMock(session_id=uuid4())
    mock_run.landscape_run_id = None
    mock_session_service.get_run = AsyncMock(return_value=mock_run)
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


def _running_status(run_id: UUID) -> RunStatusResponse:
    return RunStatusResponse(
        run_id=str(run_id),
        status="running",
        started_at=datetime.now(UTC),
        finished_at=None,
        accounting=None,
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
                downloadable=True,
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
        monkeypatch.setattr("elspeth.web.execution.routes.run_sync_in_worker", fake_to_thread)

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

    @pytest.mark.asyncio
    async def test_503_when_audit_database_unavailable(self, monkeypatch) -> None:
        run_id = uuid4()
        svc = MagicMock()
        svc.get_status = AsyncMock(return_value=_running_status(run_id))

        def fake_load(*args: object, **kwargs: object) -> RunOutputsResponse:
            raise RunOutputsAuditUnavailableError(landscape_run_id=str(run_id), landscape_url="sqlite:////missing/audit.db")

        async def fake_to_thread(func, /, *args, **kwargs):
            return func(*args, **kwargs)

        monkeypatch.setattr("elspeth.web.execution.routes.load_run_outputs_for_settings", fake_load)
        monkeypatch.setattr("elspeth.web.execution.routes.run_sync_in_worker", fake_to_thread)

        app = _create_test_app(execution_service=svc)
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.get(f"/api/runs/{run_id}/outputs")

        assert response.status_code == 503
        assert response.json()["detail"] == {
            "error_type": "run_outputs_audit_unavailable",
            "landscape_run_id": str(run_id),
            "audit_location": "/missing/audit.db",
        }


# ── Content streaming endpoint tests ────────────────────────────────


class TestRunOutputContentEndpoint:
    """GET /api/runs/{run_id}/outputs/{artifact_id}/content"""

    @pytest.mark.asyncio
    async def test_streams_file_bytes_when_inside_sink_allowlist(self, monkeypatch, tmp_path) -> None:
        run_id = uuid4()
        outputs_dir = tmp_path / "outputs"
        outputs_dir.mkdir()
        sink_file = outputs_dir / "results.jsonl"
        sink_bytes = b'{"interaction_id":"INT-1001"}\n'
        sink_file.write_bytes(sink_bytes)

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
                        content_hash=hashlib.sha256(sink_bytes).hexdigest(),
                        size_bytes=sink_file.stat().st_size,
                        created_at=datetime.now(UTC),
                        exists_now=True,
                        downloadable=True,
                    )
                ],
            )

        async def fake_to_thread(func, /, *args, **kwargs):
            return func(*args, **kwargs)

        monkeypatch.setattr("elspeth.web.execution.routes.load_run_outputs_for_settings", fake_load)
        monkeypatch.setattr("elspeth.web.execution.routes.run_sync_in_worker", fake_to_thread)

        settings = MagicMock()
        settings.auth_provider = "local"
        settings.data_dir = str(tmp_path)

        app = _create_test_app(execution_service=svc, settings=settings)
        endpoint = _route_endpoint(app, "get_run_output_content")
        request = MagicMock()
        request.app = app
        response = await endpoint(
            run_id=run_id,
            artifact_id="art-1",
            request=request,
            user=UserIdentity(user_id=_TEST_USER_ID, username="testuser"),
            service=svc,
        )

        assert isinstance(response, FileResponse)
        assert response.path == sink_file
        assert sink_file.read_bytes() == b'{"interaction_id":"INT-1001"}\n'

    @pytest.mark.asyncio
    async def test_409_when_file_content_drifts_under_same_size(self, monkeypatch, tmp_path) -> None:
        """An overwritten in-allowlist file must not be served as the audited artifact.

        Regression for elspeth-50189c547c: the endpoint streamed the file after a
        path-allowlist + existence check but never compared the current bytes
        against the audit-recorded content_hash/size_bytes. A same-size byte
        substitution (swap content, keep length) would otherwise be served under
        the artifact's identity. The content endpoint must verify the whole-file
        hash and reject drift with 409.
        """
        run_id = uuid4()
        outputs_dir = tmp_path / "outputs"
        outputs_dir.mkdir()
        sink_file = outputs_dir / "results.jsonl"
        original = b'{"interaction_id":"INT-1001"}\n'
        sink_file.write_bytes(original)
        recorded_hash = hashlib.sha256(original).hexdigest()
        recorded_size = len(original)

        # Tamper: overwrite with DIFFERENT bytes of the SAME length, so a size
        # check alone would pass — only the content hash catches it.
        tampered = b'{"interaction_id":"INT-9999"}\n'
        assert len(tampered) == len(original)
        sink_file.write_bytes(tampered)

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
                        content_hash=recorded_hash,
                        size_bytes=recorded_size,
                        created_at=datetime.now(UTC),
                        exists_now=True,
                        downloadable=True,
                    )
                ],
            )

        async def fake_to_thread(func, /, *args, **kwargs):
            return func(*args, **kwargs)

        monkeypatch.setattr("elspeth.web.execution.routes.load_run_outputs_for_settings", fake_load)
        monkeypatch.setattr("elspeth.web.execution.routes.run_sync_in_worker", fake_to_thread)

        settings = MagicMock()
        settings.auth_provider = "local"
        settings.data_dir = str(tmp_path)

        app = _create_test_app(execution_service=svc, settings=settings)
        endpoint = _route_endpoint(app, "get_run_output_content")
        request = MagicMock()
        request.app = app
        with pytest.raises(HTTPException) as exc_info:
            await endpoint(
                run_id=run_id,
                artifact_id="art-1",
                request=request,
                user=UserIdentity(user_id=_TEST_USER_ID, username="testuser"),
                service=svc,
            )
        assert exc_info.value.status_code == 409
        assert exc_info.value.detail["error_type"] == "artifact_content_drift"

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
                        downloadable=True,
                    )
                ],
            )

        async def fake_to_thread(func, /, *args, **kwargs):
            return func(*args, **kwargs)

        monkeypatch.setattr("elspeth.web.execution.routes.load_run_outputs_for_settings", fake_load)
        monkeypatch.setattr("elspeth.web.execution.routes.run_sync_in_worker", fake_to_thread)

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
        monkeypatch.setattr("elspeth.web.execution.routes.run_sync_in_worker", fake_to_thread)

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
                        downloadable=False,
                    )
                ],
            )

        async def fake_to_thread(func, /, *args, **kwargs):
            return func(*args, **kwargs)

        monkeypatch.setattr("elspeth.web.execution.routes.load_run_outputs_for_settings", fake_load)
        monkeypatch.setattr("elspeth.web.execution.routes.run_sync_in_worker", fake_to_thread)

        settings = MagicMock()
        settings.auth_provider = "local"
        settings.data_dir = str(tmp_path)

        app = _create_test_app(execution_service=svc, settings=settings)
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.get(f"/api/runs/{run_id}/outputs/art-purged/content")

        assert response.status_code == 410
        assert response.json()["detail"]["error_type"] == "artifact_purged_or_moved"


# ── Preview endpoint tests ──────────────────────────────────────────


def _file_artifact_in_outputs(
    sink_file: Path,
    *,
    artifact_id: str = "art-1",
    sink_node_id: str = "results",
) -> RunOutputArtifact:
    return RunOutputArtifact(
        artifact_id=artifact_id,
        sink_node_id=sink_node_id,
        artifact_type="file",
        path_or_uri=f"file://{sink_file}",
        content_hash="a" * 64,
        size_bytes=sink_file.stat().st_size,
        created_at=datetime.now(UTC),
        exists_now=True,
        downloadable=True,
    )


def _install_manifest_loader(
    monkeypatch: pytest.MonkeyPatch,
    *,
    artifacts: list[RunOutputArtifact],
    run_id: UUID,
) -> None:
    def fake_load(*args: object, **kwargs: object) -> RunOutputsResponse:
        return RunOutputsResponse(
            run_id=str(run_id),
            landscape_run_id=str(run_id),
            artifacts=artifacts,
        )

    async def fake_to_thread(func, /, *args, **kwargs):
        return func(*args, **kwargs)

    monkeypatch.setattr("elspeth.web.execution.routes.load_run_outputs_for_settings", fake_load)
    monkeypatch.setattr("elspeth.web.execution.routes.run_sync_in_worker", fake_to_thread)


class TestRunOutputPreviewEndpoint:
    """GET /api/runs/{run_id}/outputs/{artifact_id}/preview"""

    @pytest.mark.asyncio
    async def test_returns_csv_preview_for_small_file(self, monkeypatch, tmp_path) -> None:
        run_id = uuid4()
        outputs_dir = tmp_path / "outputs"
        outputs_dir.mkdir()
        sink_file = outputs_dir / "results.csv"
        sink_file.write_text("col1,col2\n1,2\n3,4\n")

        svc = MagicMock()
        svc.get_status = AsyncMock(return_value=_running_status(run_id))
        _install_manifest_loader(monkeypatch, artifacts=[_file_artifact_in_outputs(sink_file)], run_id=run_id)

        settings = MagicMock()
        settings.auth_provider = "local"
        settings.data_dir = str(tmp_path)

        app = _create_test_app(execution_service=svc, settings=settings)
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.get(f"/api/runs/{run_id}/outputs/art-1/preview")

        assert response.status_code == 200
        body = response.json()
        assert body["content_type"] == "csv"
        assert body["preview_text"] == "col1,col2\n1,2\n3,4\n"
        assert body["truncated"] is False
        assert body["row_count_preview"] == 3
        assert body["total_size_bytes"] == sink_file.stat().st_size

    @pytest.mark.asyncio
    async def test_text_file_under_cap_returns_full_content(self, monkeypatch, tmp_path) -> None:
        run_id = uuid4()
        outputs_dir = tmp_path / "outputs"
        outputs_dir.mkdir()
        sink_file = outputs_dir / "log.txt"
        sink_file.write_text("hello world\n")

        svc = MagicMock()
        svc.get_status = AsyncMock(return_value=_running_status(run_id))
        _install_manifest_loader(monkeypatch, artifacts=[_file_artifact_in_outputs(sink_file)], run_id=run_id)

        settings = MagicMock()
        settings.auth_provider = "local"
        settings.data_dir = str(tmp_path)

        app = _create_test_app(execution_service=svc, settings=settings)
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.get(f"/api/runs/{run_id}/outputs/art-1/preview")

        assert response.status_code == 200
        body = response.json()
        assert body["content_type"] == "text"
        assert body["preview_text"] == "hello world\n"
        assert body["truncated"] is False

    @pytest.mark.asyncio
    async def test_binary_file_returns_binary_content_type(self, monkeypatch, tmp_path) -> None:
        run_id = uuid4()
        outputs_dir = tmp_path / "outputs"
        outputs_dir.mkdir()
        sink_file = outputs_dir / "blob.bin"
        sink_file.write_bytes(b"\xff\xd8\xff\xe0\x00\x10JFIF" + b"\x00" * 200)

        svc = MagicMock()
        svc.get_status = AsyncMock(return_value=_running_status(run_id))
        _install_manifest_loader(monkeypatch, artifacts=[_file_artifact_in_outputs(sink_file)], run_id=run_id)

        settings = MagicMock()
        settings.auth_provider = "local"
        settings.data_dir = str(tmp_path)

        app = _create_test_app(execution_service=svc, settings=settings)
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.get(f"/api/runs/{run_id}/outputs/art-1/preview")

        assert response.status_code == 200
        body = response.json()
        assert body["content_type"] == "binary"
        assert body["preview_text"] == ""

    @pytest.mark.asyncio
    async def test_404_when_artifact_id_not_in_run(self, monkeypatch, tmp_path) -> None:
        run_id = uuid4()
        svc = MagicMock()
        svc.get_status = AsyncMock(return_value=_running_status(run_id))
        _install_manifest_loader(monkeypatch, artifacts=[], run_id=run_id)

        settings = MagicMock()
        settings.auth_provider = "local"
        settings.data_dir = str(tmp_path)

        app = _create_test_app(execution_service=svc, settings=settings)
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.get(f"/api/runs/{run_id}/outputs/art-missing/preview")

        assert response.status_code == 404
        assert response.json()["detail"]["error_type"] == "artifact_not_found"

    @pytest.mark.asyncio
    async def test_403_when_path_outside_sink_allowlist(self, monkeypatch, tmp_path) -> None:
        run_id = uuid4()
        elsewhere = tmp_path / "elsewhere"
        elsewhere.mkdir()
        rogue_file = elsewhere / "rogue.csv"
        rogue_file.write_text("escaped\n")

        svc = MagicMock()
        svc.get_status = AsyncMock(return_value=_running_status(run_id))
        artifact = RunOutputArtifact(
            artifact_id="art-rogue",
            sink_node_id="rogue",
            artifact_type="file",
            path_or_uri=str(rogue_file),
            content_hash="b" * 64,
            size_bytes=rogue_file.stat().st_size,
            created_at=datetime.now(UTC),
            exists_now=True,
            downloadable=False,
        )
        _install_manifest_loader(monkeypatch, artifacts=[artifact], run_id=run_id)

        settings = MagicMock()
        settings.auth_provider = "local"
        settings.data_dir = str(tmp_path)

        app = _create_test_app(execution_service=svc, settings=settings)
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.get(f"/api/runs/{run_id}/outputs/art-rogue/preview")

        assert response.status_code == 403
        assert response.json()["detail"]["error_type"] == "output_path_outside_allowlist"

    @pytest.mark.asyncio
    async def test_410_when_file_purged_between_manifest_and_preview(self, monkeypatch, tmp_path) -> None:
        run_id = uuid4()
        outputs_dir = tmp_path / "outputs"
        outputs_dir.mkdir()
        sink_file = outputs_dir / "gone.csv"
        sink_file.write_text("data\n")
        sink_file.unlink()

        svc = MagicMock()
        svc.get_status = AsyncMock(return_value=_running_status(run_id))
        artifact = RunOutputArtifact(
            artifact_id="art-purged",
            sink_node_id="results",
            artifact_type="file",
            path_or_uri=str(sink_file),
            content_hash="a" * 64,
            size_bytes=5,
            created_at=datetime.now(UTC),
            exists_now=False,
            downloadable=False,
        )
        _install_manifest_loader(monkeypatch, artifacts=[artifact], run_id=run_id)

        settings = MagicMock()
        settings.auth_provider = "local"
        settings.data_dir = str(tmp_path)

        app = _create_test_app(execution_service=svc, settings=settings)
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.get(f"/api/runs/{run_id}/outputs/art-purged/preview")

        assert response.status_code == 410
        assert response.json()["detail"]["error_type"] == "artifact_purged_or_moved"

    @pytest.mark.asyncio
    async def test_415_when_artifact_is_object_store_uri(self, monkeypatch, tmp_path) -> None:
        run_id = uuid4()
        svc = MagicMock()
        svc.get_status = AsyncMock(return_value=_running_status(run_id))
        artifact = RunOutputArtifact(
            artifact_id="art-azure",
            sink_node_id="cloud",
            artifact_type="webhook",
            path_or_uri="azure://container/blob.json",
            content_hash="c" * 64,
            size_bytes=42,
            created_at=datetime.now(UTC),
            exists_now=False,
            downloadable=False,
        )
        _install_manifest_loader(monkeypatch, artifacts=[artifact], run_id=run_id)

        settings = MagicMock()
        settings.auth_provider = "local"
        settings.data_dir = str(tmp_path)

        app = _create_test_app(execution_service=svc, settings=settings)
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.get(f"/api/runs/{run_id}/outputs/art-azure/preview")

        assert response.status_code == 415
        assert response.json()["detail"]["error_type"] == "object_store_artifact_not_previewable"

    @pytest.mark.asyncio
    async def test_404_when_run_not_owned_by_user(self, monkeypatch, tmp_path) -> None:
        # IDOR: a different user's run must look not-found, not 403.
        run_id = uuid4()
        svc = MagicMock()

        app = _create_test_app(execution_service=svc)
        # Force the session-ownership check to claim the run belongs to
        # someone else by overriding the session.user_id on the mock.
        app.state.session_service.get_session.return_value.user_id = "other-user"

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.get(f"/api/runs/{run_id}/outputs/art-1/preview")

        assert response.status_code == 404
