"""Tests for the /api/runs/{rid}/outputs manifest + content endpoints.

Mirrors the structure of test_routes.py — bypasses real DB setup via
fakes on app.state.* and dependency_overrides for the auth middleware.

The manifest endpoint is the authoritative full-list of every sink-write
artefact produced by a run (distinct from the diagnostics endpoint's
20-artifact preview). The content endpoint streams the bytes of one
artefact, gated by a path-allowlist guard that enforces
``allowed_sink_directories(data_dir, session_id=run.session_id)``.
"""

from __future__ import annotations

import hashlib
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from uuid import UUID, uuid4

import pytest
from fastapi import FastAPI, HTTPException
from httpx import ASGITransport, AsyncClient
from starlette.routing import Route

from elspeth.web.auth.models import UserIdentity
from elspeth.web.execution import routes as execution_routes
from elspeth.web.execution.outputs import RunOutputsAuditUnavailableError
from elspeth.web.execution.schemas import (
    RunOutputArtifact,
    RunOutputsResponse,
    RunStatusResponse,
)

_TEST_USER_ID = "test-user-123"
_TEST_SESSION_ID = UUID("11111111-1111-4111-8111-111111111111")


@dataclass
class _FakeSettings:
    auth_provider: str = "local"
    data_dir: str = "/tmp/elspeth-test-data"


@dataclass
class _FakeSession:
    user_id: str = _TEST_USER_ID
    auth_provider_type: str = "local"
    archived_at: datetime | None = None


@dataclass
class _FakeRun:
    session_id: UUID = field(default_factory=lambda: _TEST_SESSION_ID)
    landscape_run_id: str | None = None


@dataclass
class _FakeSessionService:
    session: _FakeSession = field(default_factory=_FakeSession)
    run: _FakeRun = field(default_factory=_FakeRun)

    async def get_session(self, session_id: UUID) -> _FakeSession:
        return self.session

    async def get_run(self, run_id: UUID) -> _FakeRun:
        return self.run


@dataclass
class _FakeExecutionService:
    status: RunStatusResponse | None = None
    error: Exception | None = None

    async def get_status(
        self,
        run_id: UUID,
        *,
        accounting: object | None = None,
        run_record: object | None = None,
    ) -> RunStatusResponse:
        if self.error is not None:
            raise self.error
        if self.status is None:
            raise AssertionError("test fake execution service has no status configured")
        return self.status


def _execution_service_for_status(run_id: UUID) -> _FakeExecutionService:
    return _FakeExecutionService(status=_running_status(run_id))


def _request_for_app(app: FastAPI, *, headers: dict[str, str] | None = None) -> SimpleNamespace:
    return SimpleNamespace(app=app, headers=headers or {})


def _route_endpoint(app: FastAPI, name: str) -> Callable[..., Awaitable[Any]]:
    for route in app.routes:
        if isinstance(route, Route) and route.name == name:
            return cast(Callable[..., Awaitable[Any]], route.endpoint)
    raise AssertionError(f"Route endpoint {name!r} not found")


def _create_test_app(
    execution_service: _FakeExecutionService | None = None,
    settings: _FakeSettings | None = None,
) -> FastAPI:
    from elspeth.web.auth.middleware import get_current_user
    from elspeth.web.execution.routes import create_execution_router

    app = FastAPI()
    app.state.execution_service = execution_service or _FakeExecutionService(status=_running_status(uuid4()))
    app.state.broadcaster = object()
    app.state.auth_provider = object()
    app.state.session_service = _FakeSessionService()

    if settings is None:
        settings = _FakeSettings()
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
        svc = _execution_service_for_status(run_id)

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
                storage_kind="sink_file",
                producer_kind="node_state",
                produced_by_state_id="state-legacy",
                sink_effect_id=None,
                publication_performed=True,
                publication_evidence_kind="legacy_returned",
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
        svc = _FakeExecutionService(error=ValueError("not found"))

        app = _create_test_app(execution_service=svc)
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.get(f"/api/runs/{run_id}/outputs")
        assert response.status_code == 404

    @pytest.mark.asyncio
    async def test_503_when_audit_database_unavailable(self, monkeypatch) -> None:
        run_id = uuid4()
        svc = _execution_service_for_status(run_id)

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
        outputs_dir = tmp_path / "outputs" / str(_TEST_SESSION_ID)
        outputs_dir.mkdir(parents=True)
        sink_file = outputs_dir / 'results "é".jsonl'
        sink_bytes = b'{"interaction_id":"INT-1001"}\n'
        sink_file.write_bytes(sink_bytes)

        svc = _execution_service_for_status(run_id)

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
                        storage_kind="sink_file",
                        producer_kind="node_state",
                        produced_by_state_id="state-legacy",
                        sink_effect_id=None,
                        publication_performed=True,
                        publication_evidence_kind="legacy_returned",
                    )
                ],
            )

        async def fake_to_thread(func, /, *args, **kwargs):
            return func(*args, **kwargs)

        monkeypatch.setattr("elspeth.web.execution.routes.load_run_outputs_for_settings", fake_load)
        monkeypatch.setattr("elspeth.web.execution.routes.run_sync_in_worker", fake_to_thread)

        settings = _FakeSettings(data_dir=str(tmp_path))

        app = _create_test_app(execution_service=svc, settings=settings)
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.get(f"/api/runs/{run_id}/outputs/art-1/content")

        assert response.status_code == 200
        assert response.content == sink_bytes
        assert "filename*" in response.headers["content-disposition"]
        assert sink_file.read_bytes() == b'{"interaction_id":"INT-1001"}\n'

    @pytest.mark.asyncio
    async def test_content_serves_decoded_file_uri_candidate_when_raw_percent_file_also_exists(self, monkeypatch, tmp_path) -> None:
        run_id = uuid4()
        outputs_dir = tmp_path / "outputs" / str(_TEST_SESSION_ID)
        outputs_dir.mkdir(parents=True)
        decoded_file = outputs_dir / "results?token=literal.csv"
        raw_percent_file = outputs_dir / "results%3Ftoken%3Dliteral.csv"
        audited_bytes = b"id,name\n1,alice\n"
        decoded_file.write_bytes(audited_bytes)
        raw_percent_file.write_bytes(b"id,name\n2,bob\n")

        svc = _execution_service_for_status(run_id)
        _install_manifest_loader(
            monkeypatch,
            artifacts=[
                RunOutputArtifact(
                    artifact_id="art-encoded",
                    sink_node_id="results",
                    artifact_type="file",
                    path_or_uri=f"file://{outputs_dir}/results%3Ftoken%3Dliteral.csv",
                    content_hash=hashlib.sha256(audited_bytes).hexdigest(),
                    size_bytes=len(audited_bytes),
                    created_at=datetime.now(UTC),
                    exists_now=True,
                    downloadable=True,
                    storage_kind="sink_file",
                    producer_kind="node_state",
                    produced_by_state_id="state-legacy",
                    sink_effect_id=None,
                    publication_performed=True,
                    publication_evidence_kind="legacy_returned",
                )
            ],
            run_id=run_id,
        )

        settings = _FakeSettings(data_dir=str(tmp_path))

        app = _create_test_app(execution_service=svc, settings=settings)
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.get(f"/api/runs/{run_id}/outputs/art-encoded/content")

        assert response.status_code == 200
        assert response.content == audited_bytes

    @pytest.mark.asyncio
    async def test_content_streams_verified_bytes_when_file_rewritten_after_integrity_check(self, monkeypatch, tmp_path) -> None:
        """The bytes returned must be the bytes that passed artifact integrity verification."""
        run_id = uuid4()
        outputs_dir = tmp_path / "outputs" / str(_TEST_SESSION_ID)
        outputs_dir.mkdir(parents=True)
        sink_file = outputs_dir / "results.jsonl"
        original = b'{"interaction_id":"INT-1001"}\n'
        sink_file.write_bytes(original)
        rewritten = b'{"interaction_id":"INT-9999"}\n'

        svc = _execution_service_for_status(run_id)

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
                        content_hash=hashlib.sha256(original).hexdigest(),
                        size_bytes=len(original),
                        created_at=datetime.now(UTC),
                        exists_now=True,
                        downloadable=True,
                        storage_kind="sink_file",
                        producer_kind="node_state",
                        produced_by_state_id="state-legacy",
                        sink_effect_id=None,
                        publication_performed=True,
                        publication_evidence_kind="legacy_returned",
                    )
                ],
            )

        real_snapshot = execution_routes._copy_artifact_to_temp_snapshot

        def fake_copy_artifact_to_temp_snapshot(path: Path, snapshot_dir: Path) -> execution_routes._ArtifactFileSnapshot:
            assert path == sink_file
            assert snapshot_dir == tmp_path / ".run-output-snapshots"
            snapshot = real_snapshot(path, snapshot_dir)
            sink_file.write_bytes(rewritten)
            return snapshot

        async def fake_to_thread(func, /, *args, **kwargs):
            return func(*args, **kwargs)

        monkeypatch.setattr("elspeth.web.execution.routes.load_run_outputs_for_settings", fake_load)
        monkeypatch.setattr("elspeth.web.execution.routes.run_sync_in_worker", fake_to_thread)
        monkeypatch.setattr(
            "elspeth.web.execution.routes._copy_artifact_to_temp_snapshot",
            fake_copy_artifact_to_temp_snapshot,
        )

        settings = _FakeSettings(data_dir=str(tmp_path))

        app = _create_test_app(execution_service=svc, settings=settings)
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.get(f"/api/runs/{run_id}/outputs/art-1/content")

        assert response.status_code == 200
        assert response.content == original
        assert response.content != rewritten

    @pytest.mark.asyncio
    async def test_content_supports_single_range_request_from_verified_snapshot(self, monkeypatch, tmp_path) -> None:
        run_id = uuid4()
        outputs_dir = tmp_path / "outputs" / str(_TEST_SESSION_ID)
        outputs_dir.mkdir(parents=True)
        sink_file = outputs_dir / "results.jsonl"
        sink_bytes = b"0123456789"
        sink_file.write_bytes(sink_bytes)

        svc = _execution_service_for_status(run_id)

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
                        size_bytes=len(sink_bytes),
                        created_at=datetime.now(UTC),
                        exists_now=True,
                        downloadable=True,
                        storage_kind="sink_file",
                        producer_kind="node_state",
                        produced_by_state_id="state-legacy",
                        sink_effect_id=None,
                        publication_performed=True,
                        publication_evidence_kind="legacy_returned",
                    )
                ],
            )

        async def fake_to_thread(func, /, *args, **kwargs):
            return func(*args, **kwargs)

        monkeypatch.setattr("elspeth.web.execution.routes.load_run_outputs_for_settings", fake_load)
        monkeypatch.setattr("elspeth.web.execution.routes.run_sync_in_worker", fake_to_thread)

        settings = _FakeSettings(data_dir=str(tmp_path))

        app = _create_test_app(execution_service=svc, settings=settings)
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.get(
                f"/api/runs/{run_id}/outputs/art-1/content",
                headers={"Range": "bytes=2-5"},
            )

        assert response.status_code == 206
        assert response.content == b"2345"
        assert response.headers["content-range"] == "bytes 2-5/10"
        assert response.headers["accept-ranges"] == "bytes"

    @pytest.mark.asyncio
    async def test_content_removes_temp_snapshot_when_send_fails(self, monkeypatch, tmp_path) -> None:
        run_id = uuid4()
        outputs_dir = tmp_path / "outputs" / str(_TEST_SESSION_ID)
        outputs_dir.mkdir(parents=True)
        sink_file = outputs_dir / "results.jsonl"
        sink_bytes = b'{"interaction_id":"INT-1001"}\n'
        sink_file.write_bytes(sink_bytes)

        svc = _execution_service_for_status(run_id)

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
                        size_bytes=len(sink_bytes),
                        created_at=datetime.now(UTC),
                        exists_now=True,
                        downloadable=True,
                        storage_kind="sink_file",
                        producer_kind="node_state",
                        produced_by_state_id="state-legacy",
                        sink_effect_id=None,
                        publication_performed=True,
                        publication_evidence_kind="legacy_returned",
                    )
                ],
            )

        async def fake_to_thread(func, /, *args, **kwargs):
            return func(*args, **kwargs)

        snapshot_paths: list[Path] = []
        real_snapshot = execution_routes._copy_artifact_to_temp_snapshot

        def fake_copy_artifact_to_temp_snapshot(path: Path, snapshot_dir: Path) -> execution_routes._ArtifactFileSnapshot:
            snapshot = real_snapshot(path, snapshot_dir)
            snapshot_paths.append(snapshot.path)
            return snapshot

        monkeypatch.setattr("elspeth.web.execution.routes.load_run_outputs_for_settings", fake_load)
        monkeypatch.setattr("elspeth.web.execution.routes.run_sync_in_worker", fake_to_thread)
        monkeypatch.setattr(
            "elspeth.web.execution.routes._copy_artifact_to_temp_snapshot",
            fake_copy_artifact_to_temp_snapshot,
        )

        settings = _FakeSettings(data_dir=str(tmp_path))

        app = _create_test_app(execution_service=svc, settings=settings)
        endpoint = _route_endpoint(app, "get_run_output_content")
        request = _request_for_app(app)
        response = await endpoint(
            run_id=run_id,
            artifact_id="art-1",
            request=request,
            user=UserIdentity(user_id=_TEST_USER_ID, username="testuser"),
            service=svc,
        )
        assert len(snapshot_paths) == 1
        snapshot_path = snapshot_paths[0]
        assert snapshot_path.exists()
        assert snapshot_path.parent == tmp_path / ".run-output-snapshots"

        async def receive() -> dict[str, object]:
            return {"type": "http.request", "body": b"", "more_body": False}

        async def send(message: dict[str, object]) -> None:
            if message["type"] == "http.response.body":
                raise RuntimeError("client disconnected")

        with pytest.raises(RuntimeError, match="client disconnected"):
            await response(
                {
                    "type": "http",
                    "asgi": {"spec_version": "2.4"},
                    "method": "GET",
                    "path": "/download",
                    "headers": [],
                    "extensions": {"http.response.pathsend": {}},
                },
                receive,
                send,
            )

        assert not snapshot_path.exists()
        assert sink_file.read_bytes() == sink_bytes

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
        outputs_dir = tmp_path / "outputs" / str(_TEST_SESSION_ID)
        outputs_dir.mkdir(parents=True)
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

        svc = _execution_service_for_status(run_id)

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
                        storage_kind="sink_file",
                        producer_kind="node_state",
                        produced_by_state_id="state-legacy",
                        sink_effect_id=None,
                        publication_performed=True,
                        publication_evidence_kind="legacy_returned",
                    )
                ],
            )

        async def fake_to_thread(func, /, *args, **kwargs):
            return func(*args, **kwargs)

        monkeypatch.setattr("elspeth.web.execution.routes.load_run_outputs_for_settings", fake_load)
        monkeypatch.setattr("elspeth.web.execution.routes.run_sync_in_worker", fake_to_thread)

        settings = _FakeSettings(data_dir=str(tmp_path))

        app = _create_test_app(execution_service=svc, settings=settings)
        endpoint = _route_endpoint(app, "get_run_output_content")
        request = _request_for_app(app)
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

        svc = _execution_service_for_status(run_id)

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
                        storage_kind="unknown",
                        producer_kind="node_state",
                        produced_by_state_id="state-legacy",
                        sink_effect_id=None,
                        publication_performed=True,
                        publication_evidence_kind="legacy_returned",
                    )
                ],
            )

        async def fake_to_thread(func, /, *args, **kwargs):
            return func(*args, **kwargs)

        monkeypatch.setattr("elspeth.web.execution.routes.load_run_outputs_for_settings", fake_load)
        monkeypatch.setattr("elspeth.web.execution.routes.run_sync_in_worker", fake_to_thread)

        # data_dir does not contain elsewhere/, so rogue_file is outside
        # The path is outside both session-owned output and blob subtrees.
        settings = _FakeSettings(data_dir=str(tmp_path))

        app = _create_test_app(execution_service=svc, settings=settings)
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.get(f"/api/runs/{run_id}/outputs/art-rogue/content")

        assert response.status_code == 403
        assert response.json()["detail"]["error_type"] == "output_path_outside_allowlist"

    @pytest.mark.asyncio
    async def test_serves_artifact_in_runs_own_session_blob_subtree(self, monkeypatch, tmp_path) -> None:
        """elspeth-bdc17cfdb1 regression guard: an output artefact stored in
        the run's OWN ``blobs/<session>/`` subtree must still stream. Output
        blobs are stored exactly there (blob.session_id == run.session_id,
        enforced by link_blob_to_run), so the per-session read allowlist must
        not break legitimate blob downloads."""
        run_id = uuid4()
        session_id = uuid4()
        blob_dir = tmp_path / "blobs" / str(session_id)
        blob_dir.mkdir(parents=True)
        blob_file = blob_dir / "out.jsonl"
        blob_bytes = b'{"row":1}\n'
        blob_file.write_bytes(blob_bytes)

        svc = _execution_service_for_status(run_id)

        def fake_load(*args: object, **kwargs: object) -> RunOutputsResponse:
            return RunOutputsResponse(
                run_id=str(run_id),
                landscape_run_id=str(run_id),
                artifacts=[
                    RunOutputArtifact(
                        artifact_id="art-blob",
                        sink_node_id="blob_out",
                        artifact_type="file",
                        path_or_uri=f"file://{blob_file}",
                        content_hash=hashlib.sha256(blob_bytes).hexdigest(),
                        size_bytes=blob_file.stat().st_size,
                        created_at=datetime.now(UTC),
                        exists_now=True,
                        downloadable=True,
                        storage_kind="blob",
                        producer_kind="node_state",
                        produced_by_state_id="state-legacy",
                        sink_effect_id=None,
                        publication_performed=True,
                        publication_evidence_kind="legacy_returned",
                    )
                ],
            )

        async def fake_to_thread(func, /, *args, **kwargs):
            return func(*args, **kwargs)

        monkeypatch.setattr("elspeth.web.execution.routes.load_run_outputs_for_settings", fake_load)
        monkeypatch.setattr("elspeth.web.execution.routes.run_sync_in_worker", fake_to_thread)

        settings = _FakeSettings(data_dir=str(tmp_path))

        app = _create_test_app(execution_service=svc, settings=settings)
        # Pin the run's owning session so the artefact's blob subtree matches.
        app.state.session_service.run.session_id = session_id
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.get(f"/api/runs/{run_id}/outputs/art-blob/content")

        assert response.status_code == 200
        assert response.content == blob_bytes

    @pytest.mark.asyncio
    async def test_403_when_artifact_in_other_session_blob_subtree(self, monkeypatch, tmp_path) -> None:
        """elspeth-bdc17cfdb1: an artefact whose path lands in ANOTHER
        session's ``blobs/<session>/`` subtree must be refused — the read-side
        of the cross-session confinement."""
        run_id = uuid4()
        own_session = uuid4()
        other_session = uuid4()
        other_dir = tmp_path / "blobs" / str(other_session)
        other_dir.mkdir(parents=True)
        foreign_file = other_dir / "secret.jsonl"
        foreign_file.write_bytes(b'{"leak":1}\n')

        svc = _execution_service_for_status(run_id)

        def fake_load(*args: object, **kwargs: object) -> RunOutputsResponse:
            return RunOutputsResponse(
                run_id=str(run_id),
                landscape_run_id=str(run_id),
                artifacts=[
                    RunOutputArtifact(
                        artifact_id="art-foreign",
                        sink_node_id="blob_out",
                        artifact_type="file",
                        path_or_uri=f"file://{foreign_file}",
                        content_hash="c" * 64,
                        size_bytes=foreign_file.stat().st_size,
                        created_at=datetime.now(UTC),
                        exists_now=True,
                        downloadable=True,
                        storage_kind="blob",
                        producer_kind="node_state",
                        produced_by_state_id="state-legacy",
                        sink_effect_id=None,
                        publication_performed=True,
                        publication_evidence_kind="legacy_returned",
                    )
                ],
            )

        async def fake_to_thread(func, /, *args, **kwargs):
            return func(*args, **kwargs)

        monkeypatch.setattr("elspeth.web.execution.routes.load_run_outputs_for_settings", fake_load)
        monkeypatch.setattr("elspeth.web.execution.routes.run_sync_in_worker", fake_to_thread)

        settings = _FakeSettings(data_dir=str(tmp_path))

        app = _create_test_app(execution_service=svc, settings=settings)
        app.state.session_service.run.session_id = own_session
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.get(f"/api/runs/{run_id}/outputs/art-foreign/content")

        assert response.status_code == 403
        assert response.json()["detail"]["error_type"] == "output_path_outside_allowlist"

    @pytest.mark.asyncio
    async def test_404_when_artifact_id_not_in_run(self, monkeypatch, tmp_path) -> None:
        run_id = uuid4()
        svc = _execution_service_for_status(run_id)

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

        settings = _FakeSettings(data_dir=str(tmp_path))

        app = _create_test_app(execution_service=svc, settings=settings)
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.get(f"/api/runs/{run_id}/outputs/art-missing/content")

        assert response.status_code == 404
        assert response.json()["detail"]["error_type"] == "artifact_not_found"

    @pytest.mark.asyncio
    async def test_410_when_artifact_path_no_longer_exists(self, monkeypatch, tmp_path) -> None:
        run_id = uuid4()
        outputs_dir = tmp_path / "outputs" / str(_TEST_SESSION_ID)
        outputs_dir.mkdir(parents=True)
        sink_file = outputs_dir / "results.jsonl"
        sink_file.write_bytes(b"will-purge\n")
        sink_file.unlink()  # File was purged after the run

        svc = _execution_service_for_status(run_id)

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
                        storage_kind="sink_file",
                        producer_kind="node_state",
                        produced_by_state_id="state-legacy",
                        sink_effect_id=None,
                        publication_performed=True,
                        publication_evidence_kind="legacy_returned",
                    )
                ],
            )

        async def fake_to_thread(func, /, *args, **kwargs):
            return func(*args, **kwargs)

        monkeypatch.setattr("elspeth.web.execution.routes.load_run_outputs_for_settings", fake_load)
        monkeypatch.setattr("elspeth.web.execution.routes.run_sync_in_worker", fake_to_thread)

        settings = _FakeSettings(data_dir=str(tmp_path))

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
    content = sink_file.read_bytes()
    return RunOutputArtifact(
        artifact_id=artifact_id,
        sink_node_id=sink_node_id,
        artifact_type="file",
        path_or_uri=f"file://{sink_file}",
        content_hash=hashlib.sha256(content).hexdigest(),
        size_bytes=sink_file.stat().st_size,
        created_at=datetime.now(UTC),
        exists_now=True,
        downloadable=True,
        storage_kind="sink_file",
        producer_kind="node_state",
        produced_by_state_id="state-legacy",
        sink_effect_id=None,
        publication_performed=True,
        publication_evidence_kind="legacy_returned",
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
    async def test_preview_uses_verified_bytes_when_file_rewritten_after_integrity_check(self, monkeypatch, tmp_path) -> None:
        """Preview must be built from the bytes that passed artifact integrity verification."""
        run_id = uuid4()
        outputs_dir = tmp_path / "outputs" / str(_TEST_SESSION_ID)
        outputs_dir.mkdir(parents=True)
        sink_file = outputs_dir / "results.jsonl"
        original = b'{"interaction_id":"INT-1001"}\n'
        sink_file.write_bytes(original)
        rewritten = b'{"interaction_id":"INT-9999"}\n'

        svc = _execution_service_for_status(run_id)
        _install_manifest_loader(
            monkeypatch,
            artifacts=[
                RunOutputArtifact(
                    artifact_id="art-1",
                    sink_node_id="results",
                    artifact_type="file",
                    path_or_uri=f"file://{sink_file}",
                    content_hash=hashlib.sha256(original).hexdigest(),
                    size_bytes=len(original),
                    created_at=datetime.now(UTC),
                    exists_now=True,
                    downloadable=True,
                    storage_kind="sink_file",
                    producer_kind="node_state",
                    produced_by_state_id="state-legacy",
                    sink_effect_id=None,
                    publication_performed=True,
                    publication_evidence_kind="legacy_returned",
                )
            ],
            run_id=run_id,
        )

        real_preview_head = execution_routes._read_artifact_preview_head

        def fake_read_artifact_preview_head(path: Path, *, byte_cap: int) -> execution_routes._ArtifactPreviewHeadSnapshot:
            assert path == sink_file
            snapshot = real_preview_head(path, byte_cap=byte_cap)
            sink_file.write_bytes(rewritten)
            return snapshot

        async def fake_to_thread(func, /, *args, **kwargs):
            return func(*args, **kwargs)

        monkeypatch.setattr("elspeth.web.execution.routes.run_sync_in_worker", fake_to_thread)
        monkeypatch.setattr(
            "elspeth.web.execution.routes._read_artifact_preview_head",
            fake_read_artifact_preview_head,
        )

        settings = _FakeSettings(data_dir=str(tmp_path))

        app = _create_test_app(execution_service=svc, settings=settings)
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.get(f"/api/runs/{run_id}/outputs/art-1/preview")

        assert response.status_code == 200
        assert "INT-1001" in response.json()["preview_text"]
        assert "INT-9999" not in response.json()["preview_text"]

    @pytest.mark.asyncio
    async def test_preview_serves_legacy_raw_percent_candidate_when_it_matches_audit(self, monkeypatch, tmp_path) -> None:
        run_id = uuid4()
        outputs_dir = tmp_path / "outputs" / str(_TEST_SESSION_ID)
        outputs_dir.mkdir(parents=True)
        legacy_raw_file = outputs_dir / "results%3Ftoken=literal.jsonl"
        decoded_decoy = outputs_dir / "results?token=literal.jsonl"
        audited_bytes = b'{"legacy":true}\n'
        legacy_raw_file.write_bytes(audited_bytes)
        decoded_decoy.write_bytes(b'{"legacy":false}\n')

        svc = _execution_service_for_status(run_id)
        _install_manifest_loader(
            monkeypatch,
            artifacts=[
                RunOutputArtifact(
                    artifact_id="art-legacy",
                    sink_node_id="results",
                    artifact_type="file",
                    path_or_uri=f"file://{outputs_dir}/results%3Ftoken=literal.jsonl",
                    content_hash=hashlib.sha256(audited_bytes).hexdigest(),
                    size_bytes=len(audited_bytes),
                    created_at=datetime.now(UTC),
                    exists_now=True,
                    downloadable=True,
                    storage_kind="sink_file",
                    producer_kind="node_state",
                    produced_by_state_id="state-legacy",
                    sink_effect_id=None,
                    publication_performed=True,
                    publication_evidence_kind="legacy_returned",
                )
            ],
            run_id=run_id,
        )

        settings = _FakeSettings(data_dir=str(tmp_path))

        app = _create_test_app(execution_service=svc, settings=settings)
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.get(f"/api/runs/{run_id}/outputs/art-legacy/preview")

        assert response.status_code == 200
        assert response.json()["preview_text"] == '{"legacy":true}\n'

    @pytest.mark.asyncio
    async def test_409_when_preview_file_content_drifts_under_same_size(self, monkeypatch, tmp_path) -> None:
        """Preview must not expose bytes that no longer match the artifact audit row."""
        run_id = uuid4()
        outputs_dir = tmp_path / "outputs" / str(_TEST_SESSION_ID)
        outputs_dir.mkdir(parents=True)
        sink_file = outputs_dir / "results.jsonl"
        original = b'{"interaction_id":"INT-1001"}\n'
        sink_file.write_bytes(original)
        recorded_hash = hashlib.sha256(original).hexdigest()
        recorded_size = len(original)

        tampered = b'{"interaction_id":"INT-9999"}\n'
        assert len(tampered) == len(original)
        sink_file.write_bytes(tampered)

        svc = _execution_service_for_status(run_id)
        _install_manifest_loader(
            monkeypatch,
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
                    storage_kind="sink_file",
                    producer_kind="node_state",
                    produced_by_state_id="state-legacy",
                    sink_effect_id=None,
                    publication_performed=True,
                    publication_evidence_kind="legacy_returned",
                )
            ],
            run_id=run_id,
        )

        settings = _FakeSettings(data_dir=str(tmp_path))

        app = _create_test_app(execution_service=svc, settings=settings)
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.get(f"/api/runs/{run_id}/outputs/art-1/preview")

        assert response.status_code == 409
        assert response.json()["detail"]["error_type"] == "artifact_content_drift"

    @pytest.mark.asyncio
    async def test_returns_csv_preview_for_small_file(self, monkeypatch, tmp_path) -> None:
        run_id = uuid4()
        outputs_dir = tmp_path / "outputs" / str(_TEST_SESSION_ID)
        outputs_dir.mkdir(parents=True)
        sink_file = outputs_dir / "results.csv"
        sink_file.write_text("col1,col2\n1,2\n3,4\n")

        svc = _execution_service_for_status(run_id)
        _install_manifest_loader(monkeypatch, artifacts=[_file_artifact_in_outputs(sink_file)], run_id=run_id)

        settings = _FakeSettings(data_dir=str(tmp_path))

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
        outputs_dir = tmp_path / "outputs" / str(_TEST_SESSION_ID)
        outputs_dir.mkdir(parents=True)
        sink_file = outputs_dir / "log.txt"
        sink_file.write_text("hello world\n")

        svc = _execution_service_for_status(run_id)
        _install_manifest_loader(monkeypatch, artifacts=[_file_artifact_in_outputs(sink_file)], run_id=run_id)

        settings = _FakeSettings(data_dir=str(tmp_path))

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
        outputs_dir = tmp_path / "outputs" / str(_TEST_SESSION_ID)
        outputs_dir.mkdir(parents=True)
        sink_file = outputs_dir / "blob.bin"
        sink_file.write_bytes(b"\xff\xd8\xff\xe0\x00\x10JFIF" + b"\x00" * 200)

        svc = _execution_service_for_status(run_id)
        _install_manifest_loader(monkeypatch, artifacts=[_file_artifact_in_outputs(sink_file)], run_id=run_id)

        settings = _FakeSettings(data_dir=str(tmp_path))

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
        svc = _execution_service_for_status(run_id)
        _install_manifest_loader(monkeypatch, artifacts=[], run_id=run_id)

        settings = _FakeSettings(data_dir=str(tmp_path))

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

        svc = _execution_service_for_status(run_id)
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
            storage_kind="unknown",
            producer_kind="node_state",
            produced_by_state_id="state-legacy",
            sink_effect_id=None,
            publication_performed=True,
            publication_evidence_kind="legacy_returned",
        )
        _install_manifest_loader(monkeypatch, artifacts=[artifact], run_id=run_id)

        settings = _FakeSettings(data_dir=str(tmp_path))

        app = _create_test_app(execution_service=svc, settings=settings)
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.get(f"/api/runs/{run_id}/outputs/art-rogue/preview")

        assert response.status_code == 403
        assert response.json()["detail"]["error_type"] == "output_path_outside_allowlist"

    @pytest.mark.asyncio
    async def test_410_when_file_purged_between_manifest_and_preview(self, monkeypatch, tmp_path) -> None:
        run_id = uuid4()
        outputs_dir = tmp_path / "outputs" / str(_TEST_SESSION_ID)
        outputs_dir.mkdir(parents=True)
        sink_file = outputs_dir / "gone.csv"
        sink_file.write_text("data\n")
        sink_file.unlink()

        svc = _execution_service_for_status(run_id)
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
            storage_kind="sink_file",
            producer_kind="node_state",
            produced_by_state_id="state-legacy",
            sink_effect_id=None,
            publication_performed=True,
            publication_evidence_kind="legacy_returned",
        )
        _install_manifest_loader(monkeypatch, artifacts=[artifact], run_id=run_id)

        settings = _FakeSettings(data_dir=str(tmp_path))

        app = _create_test_app(execution_service=svc, settings=settings)
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.get(f"/api/runs/{run_id}/outputs/art-purged/preview")

        assert response.status_code == 410
        assert response.json()["detail"]["error_type"] == "artifact_purged_or_moved"

    @pytest.mark.asyncio
    async def test_415_when_artifact_is_object_store_uri(self, monkeypatch, tmp_path) -> None:
        run_id = uuid4()
        svc = _execution_service_for_status(run_id)
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
            storage_kind="unknown",
            producer_kind="node_state",
            produced_by_state_id="state-legacy",
            sink_effect_id=None,
            publication_performed=True,
            publication_evidence_kind="legacy_returned",
        )
        _install_manifest_loader(monkeypatch, artifacts=[artifact], run_id=run_id)

        settings = _FakeSettings(data_dir=str(tmp_path))

        app = _create_test_app(execution_service=svc, settings=settings)
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.get(f"/api/runs/{run_id}/outputs/art-azure/preview")

        assert response.status_code == 415
        assert response.json()["detail"]["error_type"] == "object_store_artifact_not_previewable"

    @pytest.mark.asyncio
    async def test_404_when_run_not_owned_by_user(self, monkeypatch, tmp_path) -> None:
        # IDOR: a different user's run must look not-found, not 403.
        run_id = uuid4()
        svc = _FakeExecutionService()

        app = _create_test_app(execution_service=svc)
        # Force the session-ownership check to claim the run belongs to
        # someone else by overriding the fake session record.
        app.state.session_service.session.user_id = "other-user"

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.get(f"/api/runs/{run_id}/outputs/art-1/preview")

        assert response.status_code == 404
