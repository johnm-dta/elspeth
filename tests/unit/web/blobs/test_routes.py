"""Tests for blob REST API routes — upload, download, IDOR, MIME enforcement.

Security boundaries tested:
- IDOR protection: blobs cannot be accessed across session boundaries
- storage_path exclusion: internal filesystem paths never leak to API consumers
- MIME type allowlist: only data-oriented types accepted (no executables, images)
- Size limit enforcement: prevents resource exhaustion via oversized uploads
- Content-Disposition header: safe download filename for audit evidence export
"""

from __future__ import annotations

import io
from typing import Any

import structlog
from fastapi import FastAPI
from sqlalchemy.pool import StaticPool

from elspeth.web.auth.middleware import get_current_user
from elspeth.web.auth.models import UserIdentity
from elspeth.web.blobs.protocol import BlobPendingProposalError
from elspeth.web.blobs.routes import create_blobs_router
from elspeth.web.blobs.service import BlobServiceImpl
from elspeth.web.config import WebSettings
from elspeth.web.sessions.engine import create_session_engine
from elspeth.web.sessions.routes import create_session_router
from elspeth.web.sessions.schema import initialize_session_schema
from elspeth.web.sessions.service import SessionServiceImpl
from elspeth.web.sessions.telemetry import build_sessions_telemetry
from tests.unit.web._sync_asgi_client import SyncASGITestClient as TestClient

# ---------------------------------------------------------------------------
# Test app factory
# ---------------------------------------------------------------------------


def _make_app(
    tmp_path,
    user_id: str = "alice",
    max_upload_bytes: int = 10 * 1024 * 1024,
) -> tuple[FastAPI, SessionServiceImpl, BlobServiceImpl]:
    """Create a test app with session and blob routes."""
    engine = create_session_engine(
        "sqlite:///:memory:",
        poolclass=StaticPool,
        connect_args={"check_same_thread": False},
    )
    initialize_session_schema(engine)
    session_service = SessionServiceImpl(
        engine,
        telemetry=build_sessions_telemetry(),
        log=structlog.get_logger("test"),
    )
    blob_service = BlobServiceImpl(engine, tmp_path)

    app = FastAPI()

    identity = UserIdentity(user_id=user_id, username=user_id)

    async def mock_user():
        return identity

    app.dependency_overrides[get_current_user] = mock_user

    settings = WebSettings(
        data_dir=tmp_path,
        max_upload_bytes=max_upload_bytes,
        composer_max_composition_turns=15,
        composer_max_discovery_turns=10,
        composer_timeout_seconds=85.0,
        composer_rate_limit_per_minute=10,
        shareable_link_signing_key=b"\x00" * 32,
    )
    app.state.settings = settings
    app.state.session_service = session_service
    app.state.blob_service = blob_service

    from elspeth.web.middleware.rate_limit import ComposerRateLimiter

    app.state.rate_limiter = ComposerRateLimiter(limit=100)

    app.include_router(create_session_router())
    app.include_router(create_blobs_router())

    return app, session_service, blob_service


def _create_session(client: TestClient, title: str = "Test") -> str:
    """Create a session and return its ID."""
    resp = client.post("/api/sessions", json={"title": title})
    assert resp.status_code == 201
    session_id: str = resp.json()["id"]
    return session_id


def _upload_blob(
    client: TestClient,
    session_id: str,
    content: bytes = b"col1,col2,col3\na,b,c",
    filename: str = "data.csv",
    content_type: str = "text/csv",
) -> dict[str, Any]:
    """Upload a blob and return the response body."""
    resp = client.post(
        f"/api/sessions/{session_id}/blobs",
        files={"file": (filename, io.BytesIO(content), content_type)},
    )
    assert resp.status_code == 201
    body: dict[str, Any] = resp.json()
    return body


# ---------------------------------------------------------------------------
# Upload tests
# ---------------------------------------------------------------------------


class TestUploadBlob:
    """Blob upload: creation, MIME validation, size enforcement."""

    def test_upload_blob_returns_201_with_metadata(self, tmp_path) -> None:
        app, _, _ = _make_app(tmp_path)
        client = TestClient(app)
        session_id = _create_session(client)

        body = _upload_blob(client, session_id)

        assert "id" in body
        assert body["filename"] == "data.csv"
        assert body["mime_type"] == "text/csv"
        assert body["status"] == "ready"
        # storage_path MUST NOT be in the response — internal implementation detail
        assert "storage_path" not in body

    def test_upload_blob_rejects_disallowed_mime_type(self, tmp_path) -> None:
        """MIME allowlist: image/jpeg is not a data format."""
        app, _, _ = _make_app(tmp_path)
        client = TestClient(app)
        session_id = _create_session(client)

        resp = client.post(
            f"/api/sessions/{session_id}/blobs",
            files={"file": ("photo.jpg", io.BytesIO(b"\xff\xd8\xff"), "image/jpeg")},
        )
        assert resp.status_code == 415

    def test_upload_csv_with_excel_mime_accepted(self, tmp_path) -> None:
        """Browsers report .csv as application/vnd.ms-excel on Windows.

        The server-side content sniff detects text/csv from the actual
        content, overriding the browser's (untrusted) MIME declaration.
        """
        app, _, _ = _make_app(tmp_path)
        client = TestClient(app)
        session_id = _create_session(client)

        resp = client.post(
            f"/api/sessions/{session_id}/blobs",
            files={"file": ("data.csv", io.BytesIO(b"name,age,city\nAlice,30,London"), "application/vnd.ms-excel")},
        )
        assert resp.status_code == 201
        body = resp.json()
        assert body["mime_type"] == "text/csv"

    def test_upload_csv_with_octet_stream_mime_accepted(self, tmp_path) -> None:
        """Some browsers send application/octet-stream for unknown extensions."""
        app, _, _ = _make_app(tmp_path)
        client = TestClient(app)
        session_id = _create_session(client)

        resp = client.post(
            f"/api/sessions/{session_id}/blobs",
            files={"file": ("data.csv", io.BytesIO(b"x,y,z\n1,2,3"), "application/octet-stream")},
        )
        assert resp.status_code == 201
        body = resp.json()
        assert body["mime_type"] == "text/csv"

    def test_upload_blob_enforces_size_limit(self, tmp_path) -> None:
        """Resource exhaustion: oversized uploads rejected before full read."""
        app, _, _ = _make_app(tmp_path, max_upload_bytes=100)
        client = TestClient(app)
        session_id = _create_session(client)

        big_content = b"x" * 200
        resp = client.post(
            f"/api/sessions/{session_id}/blobs",
            files={"file": ("big.csv", io.BytesIO(big_content), "text/csv")},
        )
        assert resp.status_code == 413


# ---------------------------------------------------------------------------
# List and metadata
# ---------------------------------------------------------------------------


class TestListAndMetadata:
    """Blob listing and metadata retrieval."""

    def test_list_blobs_returns_session_blobs(self, tmp_path) -> None:
        app, _, _ = _make_app(tmp_path)
        client = TestClient(app)
        session_id = _create_session(client)

        _upload_blob(client, session_id, filename="a.csv")
        _upload_blob(client, session_id, filename="b.csv")

        resp = client.get(f"/api/sessions/{session_id}/blobs")
        assert resp.status_code == 200
        assert len(resp.json()) == 2

    def test_get_blob_metadata_excludes_storage_path(self, tmp_path) -> None:
        """storage_path is an internal detail — never exposed to API consumers."""
        app, _, _ = _make_app(tmp_path)
        client = TestClient(app)
        session_id = _create_session(client)

        blob = _upload_blob(client, session_id)
        blob_id = blob["id"]

        resp = client.get(f"/api/sessions/{session_id}/blobs/{blob_id}")
        assert resp.status_code == 200
        body = resp.json()
        assert "storage_path" not in body
        assert body["filename"] == "data.csv"


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------


class TestDownloadBlob:
    """Blob download: content integrity and Content-Disposition header."""

    def test_download_blob_returns_content(self, tmp_path) -> None:
        app, _, _ = _make_app(tmp_path)
        client = TestClient(app)
        session_id = _create_session(client)

        content = b"col1,col2\n1,2\n3,4"
        blob = _upload_blob(client, session_id, content=content)
        blob_id = blob["id"]

        resp = client.get(f"/api/sessions/{session_id}/blobs/{blob_id}/content")
        assert resp.status_code == 200
        assert resp.content == content

    def test_download_content_disposition_header(self, tmp_path) -> None:
        """RFC 5987 encoded filename in Content-Disposition for safe export."""
        app, _, _ = _make_app(tmp_path)
        client = TestClient(app)
        session_id = _create_session(client)

        blob = _upload_blob(client, session_id, filename="data.csv")
        blob_id = blob["id"]

        resp = client.get(f"/api/sessions/{session_id}/blobs/{blob_id}/content")
        assert resp.status_code == 200
        cd = resp.headers.get("content-disposition", "")
        assert "attachment" in cd
        assert "data.csv" in cd


# ---------------------------------------------------------------------------
# Preview
# ---------------------------------------------------------------------------


class TestPreviewBlob:
    """Blob preview: bounded inline read, not a full download."""

    def test_preview_returns_bounded_prefix_without_full_content_read(self, tmp_path, monkeypatch) -> None:
        app, _, blob_service = _make_app(tmp_path)
        client = TestClient(app)
        session_id = _create_session(client)
        content = b"a" * 6000
        blob = _upload_blob(client, session_id, content=content, content_type="text/plain", filename="large.txt")

        async def fail_full_read(blob_id):
            raise AssertionError(f"preview must not call read_blob_content({blob_id})")

        monkeypatch.setattr(blob_service, "read_blob_content", fail_full_read)

        resp = client.get(f"/api/sessions/{session_id}/blobs/{blob['id']}/preview?limit=5000")

        assert resp.status_code == 200
        assert resp.content == b"a" * 5000
        assert resp.headers["x-preview-truncated"] == "true"
        assert resp.headers["x-preview-limit"] == "5000"


class TestDeleteBlob:
    def test_pending_proposal_conflict_returns_409(self, tmp_path, monkeypatch) -> None:
        app, _, blob_service = _make_app(tmp_path)
        client = TestClient(app)
        session_id = _create_session(client)
        blob = _upload_blob(client, session_id)

        async def reject_pending_proposal(blob_id) -> None:
            raise BlobPendingProposalError(str(blob_id), proposal_id="proposal-pending")

        monkeypatch.setattr(blob_service, "delete_blob", reject_pending_proposal)

        response = client.delete(f"/api/sessions/{session_id}/blobs/{blob['id']}")

        assert response.status_code == 409
        assert "proposal-pending" in response.json()["detail"]


# ---------------------------------------------------------------------------
# IDOR protection — CRITICAL security boundary
# ---------------------------------------------------------------------------


class TestIDORProtection:
    """IDOR: blob access from a different session must return 404, not the blob.

    These tests protect against the most dangerous class of authorization
    bypass: accessing another user's audit evidence by guessing blob IDs.
    404 (not 403) prevents information leakage about blob existence.
    """

    def _make_two_session_app(self, tmp_path) -> tuple[TestClient, TestClient]:
        """Create shared-DB app with two users, each with a session."""
        engine = create_session_engine(
            "sqlite:///:memory:",
            poolclass=StaticPool,
            connect_args={"check_same_thread": False},
        )
        initialize_session_schema(engine)
        session_service = SessionServiceImpl(
            engine,
            telemetry=build_sessions_telemetry(),
            log=structlog.get_logger("test"),
        )
        blob_service = BlobServiceImpl(engine, tmp_path)

        settings = WebSettings(
            data_dir=tmp_path,
            composer_max_composition_turns=15,
            composer_max_discovery_turns=10,
            composer_timeout_seconds=85.0,
            composer_rate_limit_per_minute=10,
            shareable_link_signing_key=b"\x00" * 32,
        )

        def make_app_for_user(uid: str) -> FastAPI:
            app = FastAPI()
            identity = UserIdentity(user_id=uid, username=uid)

            async def mock_user():
                return identity

            app.dependency_overrides[get_current_user] = mock_user
            app.state.session_service = session_service
            app.state.blob_service = blob_service
            app.state.settings = settings

            from elspeth.web.middleware.rate_limit import ComposerRateLimiter

            app.state.rate_limiter = ComposerRateLimiter(limit=100)

            app.include_router(create_session_router())
            app.include_router(create_blobs_router())
            return app

        alice_app = make_app_for_user("alice")
        bob_app = make_app_for_user("bob")

        return TestClient(alice_app), TestClient(bob_app)

    def test_blob_access_from_wrong_session_returns_404(self, tmp_path) -> None:
        """GET metadata for another user's blob returns 404."""
        alice, bob = self._make_two_session_app(tmp_path)

        alice_session = _create_session(alice, "Alice Session")
        bob_session = _create_session(bob, "Bob Session")
        blob = _upload_blob(alice, alice_session)

        resp = bob.get(f"/api/sessions/{bob_session}/blobs/{blob['id']}")
        assert resp.status_code == 404

    def test_blob_delete_from_wrong_session_returns_404(self, tmp_path) -> None:
        """DELETE another user's blob returns 404."""
        alice, bob = self._make_two_session_app(tmp_path)

        alice_session = _create_session(alice, "Alice Session")
        bob_session = _create_session(bob, "Bob Session")
        blob = _upload_blob(alice, alice_session)

        resp = bob.delete(f"/api/sessions/{bob_session}/blobs/{blob['id']}")
        assert resp.status_code == 404

    def test_blob_download_from_wrong_session_returns_404(self, tmp_path) -> None:
        """GET content for another user's blob returns 404."""
        alice, bob = self._make_two_session_app(tmp_path)

        alice_session = _create_session(alice, "Alice Session")
        bob_session = _create_session(bob, "Bob Session")
        blob = _upload_blob(alice, alice_session)

        resp = bob.get(f"/api/sessions/{bob_session}/blobs/{blob['id']}/content")
        assert resp.status_code == 404

    def test_blob_preview_from_wrong_session_returns_404(self, tmp_path) -> None:
        """GET bounded preview for another user's blob returns 404."""
        alice, bob = self._make_two_session_app(tmp_path)

        alice_session = _create_session(alice, "Alice Session")
        bob_session = _create_session(bob, "Bob Session")
        blob = _upload_blob(alice, alice_session)

        resp = bob.get(f"/api/sessions/{bob_session}/blobs/{blob['id']}/preview")
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# Blob lifecycle enforcement — download guard (elspeth-182cbb262b)
# ---------------------------------------------------------------------------


class TestDownloadLifecycleGuard:
    """Download route must reject non-ready blobs with 409 Conflict.

    Bug: elspeth-182cbb262b — download_blob_content serves blobs that
    are still pending or already error, because the route never enforces
    the blob lifecycle before reading storage.

    Uses direct DB seeding (not asyncio service calls) to create
    non-ready blobs — avoids the deprecated asyncio.get_event_loop()
    pattern and matches the established test style in test_service.py.
    """

    @staticmethod
    def _seed_blob(
        engine,
        session_id: str,
        blob_id: str,
        status: str,
        storage_path: str,
        content_hash: str | None = None,
        size_bytes: int = 0,
    ) -> None:
        """Insert a blob row directly into the DB for test setup."""
        from datetime import UTC, datetime

        from elspeth.web.sessions.models import blobs_table

        with engine.begin() as conn:
            conn.execute(
                blobs_table.insert().values(
                    id=blob_id,
                    session_id=session_id,
                    filename="output.csv",
                    mime_type="text/csv",
                    size_bytes=size_bytes,
                    content_hash=content_hash,
                    storage_path=storage_path,
                    created_at=datetime.now(UTC),
                    created_by="pipeline",
                    source_description="test",
                    status=status,
                )
            )

    def test_download_pending_blob_returns_409(self, tmp_path) -> None:
        """Pending blob: content not finalized, download must be rejected."""
        from uuid import uuid4

        app, _, blob_service = _make_app(tmp_path)
        client = TestClient(app)
        session_id = _create_session(client)

        blob_id = str(uuid4())
        storage = tmp_path / "blobs" / session_id / f"{blob_id}_output.csv"
        storage.parent.mkdir(parents=True, exist_ok=True)
        storage.write_bytes(b"not-yet-finalized")

        self._seed_blob(blob_service._engine, session_id, blob_id, "pending", str(storage))

        resp = client.get(f"/api/sessions/{session_id}/blobs/{blob_id}/content")
        assert resp.status_code == 409, f"Expected 409 Conflict for pending blob, got {resp.status_code}"
        assert "pending" in resp.json()["detail"]

    def test_download_error_blob_returns_409(self, tmp_path) -> None:
        """Error blob: run failed, content must not be served."""
        from uuid import uuid4

        app, _, blob_service = _make_app(tmp_path)
        client = TestClient(app)
        session_id = _create_session(client)

        blob_id = str(uuid4())
        storage = tmp_path / "blobs" / session_id / f"{blob_id}_output.csv"
        storage.parent.mkdir(parents=True, exist_ok=True)
        storage.write_bytes(b"partial-output")

        self._seed_blob(blob_service._engine, session_id, blob_id, "error", str(storage))

        resp = client.get(f"/api/sessions/{session_id}/blobs/{blob_id}/content")
        assert resp.status_code == 409, f"Expected 409 Conflict for error blob, got {resp.status_code}"
        assert "error" in resp.json()["detail"]

    def test_download_ready_blob_still_works(self, tmp_path) -> None:
        """Sanity check: ready blobs are still downloadable after the guard."""
        app, _, _ = _make_app(tmp_path)
        client = TestClient(app)
        session_id = _create_session(client)

        content = b"col1,col2\n1,2"
        blob = _upload_blob(client, session_id, content=content)

        resp = client.get(f"/api/sessions/{session_id}/blobs/{blob['id']}/content")
        assert resp.status_code == 200
        assert resp.content == content

    def test_download_tampered_blob_returns_500(self, tmp_path) -> None:
        """Integrity failure: tampered file on disk returns 500, not content."""
        from pathlib import Path

        app, _, blob_service = _make_app(tmp_path)
        client = TestClient(app)
        session_id = _create_session(client)

        # Upload a valid blob, then tamper with the backing file
        blob = _upload_blob(client, session_id, content=b"original-content")
        blob_id = blob["id"]

        # Find and tamper the backing file
        from elspeth.web.sessions.models import blobs_table

        with blob_service._engine.connect() as conn:
            row = conn.execute(blobs_table.select().where(blobs_table.c.id == blob_id)).first()
        # Tier 1 read guard — the upload we just made cannot have vanished;
        # a None here would indicate a catastrophic test-harness bug
        # (concurrent delete, DB corruption).  Crash offensively rather
        # than carry the Optional through the tamper sequence.
        assert row is not None, f"blob {blob_id} vanished between upload and tamper"
        Path(row.storage_path).write_bytes(b"tampered-content")

        resp = client.get(f"/api/sessions/{session_id}/blobs/{blob_id}/content")
        assert resp.status_code == 500, f"Expected 500 for tampered blob, got {resp.status_code}"
        assert "integrity" in resp.json()["detail"].lower()

    def test_download_ready_blob_with_missing_backing_file_returns_500(self, tmp_path) -> None:
        """Ready blob metadata with a missing file must surface as integrity failure."""
        from pathlib import Path

        app, _, blob_service = _make_app(tmp_path)
        client = TestClient(app)
        session_id = _create_session(client)

        blob = _upload_blob(client, session_id, content=b"original-content")
        blob_id = blob["id"]

        from elspeth.web.sessions.models import blobs_table

        with blob_service._engine.connect() as conn:
            row = conn.execute(blobs_table.select().where(blobs_table.c.id == blob_id)).first()
        assert row is not None, f"blob {blob_id} vanished between upload and unlink"
        Path(row.storage_path).unlink()

        resp = client.get(f"/api/sessions/{session_id}/blobs/{blob_id}/content")
        assert resp.status_code == 500, f"Expected 500 for missing backing file, got {resp.status_code}"
        assert "integrity" in resp.json()["detail"].lower()


# ---------------------------------------------------------------------------
# Filename validation at the HTTP boundary (elspeth-12e778e606, elspeth-3b189ef8a5)
# ---------------------------------------------------------------------------


class TestFilenameValidation:
    """Malformed filenames must produce 4xx, never 500."""

    def test_multipart_upload_rejects_dotdot_filename(self, tmp_path) -> None:
        app, _, _ = _make_app(tmp_path)
        client = TestClient(app)
        session_id = _create_session(client)

        resp = client.post(
            f"/api/sessions/{session_id}/blobs",
            files={"file": ("..", io.BytesIO(b"a,b,c\n1,2,3\n"), "text/csv")},
        )
        assert resp.status_code == 422
        assert "invalid filename" in resp.json()["detail"].lower()

    def test_multipart_upload_rejects_dot_filename(self, tmp_path) -> None:
        app, _, _ = _make_app(tmp_path)
        client = TestClient(app)
        session_id = _create_session(client)

        resp = client.post(
            f"/api/sessions/{session_id}/blobs",
            files={"file": (".", io.BytesIO(b"a,b,c\n1,2,3\n"), "text/csv")},
        )
        assert resp.status_code == 422

    def test_inline_rejects_empty_filename(self, tmp_path) -> None:
        app, _, _ = _make_app(tmp_path)
        client = TestClient(app)
        session_id = _create_session(client)

        resp = client.post(
            f"/api/sessions/{session_id}/blobs/inline",
            json={"filename": "", "content": "{}", "mime_type": "application/json"},
        )
        assert resp.status_code == 422

    def test_inline_rejects_dotdot_filename(self, tmp_path) -> None:
        app, _, _ = _make_app(tmp_path)
        client = TestClient(app)
        session_id = _create_session(client)

        resp = client.post(
            f"/api/sessions/{session_id}/blobs/inline",
            json={"filename": "..", "content": "{}", "mime_type": "application/json"},
        )
        assert resp.status_code == 422


# ---------------------------------------------------------------------------
# Inline endpoint — mime_type contract (elspeth-f7daa8c016)
# ---------------------------------------------------------------------------


class TestInlineBlobRequest:
    """/blobs/inline rejects malformed bodies and uses mime_type (not content_type)."""

    def test_inline_happy_path_records_mime_type(self, tmp_path) -> None:
        app, _, _ = _make_app(tmp_path)
        client = TestClient(app)
        session_id = _create_session(client)

        resp = client.post(
            f"/api/sessions/{session_id}/blobs/inline",
            json={
                "filename": "data.json",
                "content": '{"a": 1}',
                "mime_type": "application/json",
            },
        )
        assert resp.status_code == 201
        body = resp.json()
        assert body["filename"] == "data.json"
        assert body["mime_type"] == "application/json"

    def test_inline_happy_path_records_authenticated_user_authorship(self, tmp_path) -> None:
        """Inline REST uploads are user-authored; assistant blobs come from composer tools."""
        app, _, _ = _make_app(tmp_path, user_id="regular-user")
        client = TestClient(app)
        session_id = _create_session(client)

        resp = client.post(
            f"/api/sessions/{session_id}/blobs/inline",
            json={
                "filename": "data.json",
                "content": '{"a": 1}',
                "mime_type": "application/json",
            },
        )

        assert resp.status_code == 201
        assert resp.json()["created_by"] == "user"

    def test_inline_rejects_legacy_content_type_field(self, tmp_path) -> None:
        """The old `content_type` key must not silently downgrade to text/plain."""
        app, _, _ = _make_app(tmp_path)
        client = TestClient(app)
        session_id = _create_session(client)

        resp = client.post(
            f"/api/sessions/{session_id}/blobs/inline",
            json={
                "filename": "data.json",
                "content": '{"a": 1}',
                "content_type": "application/json",  # wrong key name
            },
        )
        # extra="forbid" + mime_type missing → 422, never a silent text/plain
        assert resp.status_code == 422

    def test_inline_rejects_unsupported_mime_type(self, tmp_path) -> None:
        """Literal-typed mime_type rejects values outside the allowlist."""
        app, _, _ = _make_app(tmp_path)
        client = TestClient(app)
        session_id = _create_session(client)

        resp = client.post(
            f"/api/sessions/{session_id}/blobs/inline",
            json={
                "filename": "photo.png",
                "content": "fake",
                "mime_type": "image/png",
            },
        )
        assert resp.status_code == 422

    def test_inline_default_mime_type_is_text_plain(self, tmp_path) -> None:
        app, _, _ = _make_app(tmp_path)
        client = TestClient(app)
        session_id = _create_session(client)

        resp = client.post(
            f"/api/sessions/{session_id}/blobs/inline",
            json={"filename": "notes.txt", "content": "hello"},
        )
        assert resp.status_code == 201
        assert resp.json()["mime_type"] == "text/plain"


# ---------------------------------------------------------------------------
# UTF-16 upload support (elspeth-3e6a7e0cdb)
# ---------------------------------------------------------------------------


class TestUTF16Uploads:
    """Non-UTF-8 text uploads with BOMs must not be rejected as binary."""

    def test_utf16_le_csv_upload_accepted(self, tmp_path) -> None:
        app, _, _ = _make_app(tmp_path)
        client = TestClient(app)
        session_id = _create_session(client)

        # UTF-16 LE BOM + 3-column CSV (the sniffer's CSV floor is 3 fields)
        content = b"\xff\xfe" + "name,age,city\nAlice,30,London\n".encode("utf-16-le")
        resp = client.post(
            f"/api/sessions/{session_id}/blobs",
            files={"file": ("data.csv", io.BytesIO(content), "text/csv")},
        )
        assert resp.status_code == 201, resp.text
        assert resp.json()["mime_type"] == "text/csv"

    def test_utf16_be_plain_text_upload_accepted(self, tmp_path) -> None:
        app, _, _ = _make_app(tmp_path)
        client = TestClient(app)
        session_id = _create_session(client)

        content = b"\xfe\xff" + "hello world\n".encode("utf-16-be")
        resp = client.post(
            f"/api/sessions/{session_id}/blobs",
            files={"file": ("note.txt", io.BytesIO(content), "text/plain")},
        )
        assert resp.status_code == 201, resp.text
        assert resp.json()["mime_type"] == "text/plain"
