"""Integration tests for /api/sessions/{sid}/{mark-ready-for-review,shareable-link}
and /api/sessions/shared/{token} routes.

Phase 6A Task 6 (UX redesign 2026-05). Reuses the audit-readiness test
harness because the same ``_passthrough_composition_state`` fixture produces
a valid composition that satisfies the mark-time gate.

Fixtures live in ``tests/integration/web/conftest.py``:
    * ``audit_readiness_test_client`` — full app, alice authenticated.
    * ``audit_readiness_client_with_state`` — also seeds a passthrough state.
    * ``audit_readiness_client_anonymous`` — auth raises 401.
    * ``audit_readiness_other_user_session_id`` — session owned by bob.
"""

from __future__ import annotations

import asyncio
import uuid
from datetime import UTC, datetime, timedelta
from uuid import UUID

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select

from elspeth.web.auth.middleware import get_current_user
from elspeth.web.auth.models import UserIdentity
from elspeth.web.sessions.models import composer_completion_events_table
from elspeth.web.sessions.protocol import CompositionStateData
from elspeth.web.shareable_reviews.signer import ShareTokenPayload

from .conftest import _TEST_AUTHED_USER_ID, _passthrough_composition_state

# ── POST /mark-ready-for-review ─────────────────────────────────────────


def test_mark_ready_for_review_happy_path(
    audit_readiness_client_with_state: tuple[TestClient, UUID],
) -> None:
    client, session_id = audit_readiness_client_with_state
    response = client.post(f"/api/sessions/{session_id}/mark-ready-for-review")
    assert response.status_code == 200, response.text
    body = response.json()
    assert isinstance(body["token"], str) and body["token"]
    assert body["share_url"].endswith(body["token"])
    assert body["payload_digest"].startswith("sha256:")


def test_mark_ready_for_review_records_audit_row(
    audit_readiness_client_with_state: tuple[TestClient, UUID],
) -> None:
    """A `mark_ready_for_review` row lands in ``composer_completion_events_table``."""
    client, session_id = audit_readiness_client_with_state
    response = client.post(f"/api/sessions/{session_id}/mark-ready-for-review")
    assert response.status_code == 200
    digest = response.json()["payload_digest"]
    engine = client.app.state.session_engine
    with engine.connect() as conn:
        rows = conn.execute(
            select(composer_completion_events_table).where(composer_completion_events_table.c.session_id == str(session_id))
        ).all()
    assert len(rows) == 1
    assert rows[0].event_type == "mark_ready_for_review"
    assert rows[0].payload_digest == digest
    assert rows[0].actor == "alice"


def test_mark_ready_for_review_requires_auth(
    audit_readiness_client_anonymous: TestClient,
) -> None:
    any_session_id = uuid.uuid4()
    response = audit_readiness_client_anonymous.post(f"/api/sessions/{any_session_id}/mark-ready-for-review")
    assert response.status_code == 401


def test_mark_ready_for_review_idor_returns_404(
    audit_readiness_test_client: TestClient,
    audit_readiness_other_user_session_id: UUID,
) -> None:
    """alice posts against bob's session → 404 (byte-identical to "not found")."""
    response = audit_readiness_test_client.post(f"/api/sessions/{audit_readiness_other_user_session_id}/mark-ready-for-review")
    assert response.status_code == 404


def test_mark_ready_for_review_no_state_returns_409(
    audit_readiness_client_without_state: tuple[TestClient, UUID],
) -> None:
    """Session exists but has no composition state — validation gate fires 409."""
    client, session_id = audit_readiness_client_without_state
    response = client.post(f"/api/sessions/{session_id}/mark-ready-for-review")
    # validate() returns is_valid=False for "no state" → CompositionNotRunnableError → 409.
    assert response.status_code == 409


def _seed_session_with_blob_subtree_sink(client: TestClient, *, user_id: str) -> UUID:
    """Seed a session whose sink writes into the session's OWN blob subtree.

    Mirrors ``conftest._seed_session_with_state`` but anchors the sink path
    at ``data_dir/blobs/<session_id>/...`` — a target the session-scoped
    sink allowlist (elspeth-bdc17cfdb1) only admits when the validator
    receives the caller's session id. The path depends on the session id,
    so the state is built after ``create_session``.
    """
    session_service = client.app.state.session_service
    settings = client.app.state.settings
    (settings.data_dir / "outputs").mkdir(parents=True, exist_ok=True)

    async def _seed() -> UUID:
        record = await session_service.create_session(
            user_id=user_id,
            title="blob-subtree sink fixture",
            auth_provider_type=settings.auth_provider,
        )
        (settings.data_dir / "blobs" / str(record.id)).mkdir(parents=True, exist_ok=True)
        state_d = _passthrough_composition_state(settings.data_dir).to_dict()
        state_d["outputs"][0]["options"]["path"] = str(settings.data_dir / "blobs" / str(record.id) / "review_out.csv")
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


def test_mark_ready_for_review_allows_own_session_blob_sink(
    audit_readiness_test_client: TestClient,
) -> None:
    """A sink targeting the session's own ``blobs/<session_id>/`` subtree
    must pass the mark-time validation gate exactly as it passes /validate:
    ``mark_ready_for_review`` threads the session id into ``validate_state``
    so the session-scoped sink allowlist includes the caller's own subtree.
    """
    client = audit_readiness_test_client
    session_id = _seed_session_with_blob_subtree_sink(client, user_id=_TEST_AUTHED_USER_ID)
    # Parity guard: the same state is valid through /validate...
    validate_response = client.post(f"/api/sessions/{session_id}/validate")
    assert validate_response.status_code == 200, validate_response.text
    assert validate_response.json()["is_valid"] is True, validate_response.text
    # ...so the share gate must agree.
    response = client.post(f"/api/sessions/{session_id}/mark-ready-for-review")
    assert response.status_code == 200, response.text


# ── GET /shareable-link ─────────────────────────────────────────────────


def test_get_shareable_link_remints_stable_digest(
    audit_readiness_client_with_state: tuple[TestClient, UUID],
) -> None:
    """Two GETs yield identical payload_digest but different token strings."""
    client, session_id = audit_readiness_client_with_state
    mark_response = client.post(f"/api/sessions/{session_id}/mark-ready-for-review")
    assert mark_response.status_code == 200, mark_response.text

    r1 = client.get(f"/api/sessions/{session_id}/shareable-link")
    r2 = client.get(f"/api/sessions/{session_id}/shareable-link")
    assert r1.status_code == 200, r1.text
    assert r2.status_code == 200, r2.text
    assert r1.json()["payload_digest"] == r2.json()["payload_digest"]
    assert r1.json()["token"] != r2.json()["token"]


def test_get_shareable_link_requires_auth(
    audit_readiness_client_anonymous: TestClient,
) -> None:
    any_session_id = uuid.uuid4()
    response = audit_readiness_client_anonymous.get(f"/api/sessions/{any_session_id}/shareable-link")
    assert response.status_code == 401


def test_get_shareable_link_idor_returns_404(
    audit_readiness_test_client: TestClient,
    audit_readiness_other_user_session_id: UUID,
) -> None:
    response = audit_readiness_test_client.get(f"/api/sessions/{audit_readiness_other_user_session_id}/shareable-link")
    assert response.status_code == 404


# ── GET /sessions/shared/{token} ────────────────────────────────────────


def _mint_token(client: TestClient, session_id: UUID) -> str:
    response = client.post(f"/api/sessions/{session_id}/mark-ready-for-review")
    assert response.status_code == 200, response.text
    return response.json()["token"]


def test_get_shared_inspect_happy_path(
    audit_readiness_client_with_state: tuple[TestClient, UUID],
) -> None:
    client, session_id = audit_readiness_client_with_state
    token = _mint_token(client, session_id)
    response = client.get(f"/api/sessions/shared/{token}")
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["session_id"] == str(session_id)
    assert body["yaml"]
    # audit_readiness is the six-row snapshot, served from the frozen blob.
    assert {row["id"] for row in body["audit_readiness"]["rows"]} == {
        "validation",
        "plugin_trust",
        "provenance",
        "retention",
        "llm_interpretations",
        "secrets",
    }


def test_get_shared_inspect_recipient_is_not_creator(
    audit_readiness_client_with_state: tuple[TestClient, UUID],
) -> None:
    """alice mints, bob (different authenticated user) resolves successfully."""
    client, session_id = audit_readiness_client_with_state
    token = _mint_token(client, session_id)

    # Swap the get_current_user override to return bob, then re-GET.
    bob_identity = UserIdentity(user_id="bob", username="bob")

    async def _bob() -> UserIdentity:
        return bob_identity

    original_override = client.app.dependency_overrides[get_current_user]
    client.app.dependency_overrides[get_current_user] = _bob
    try:
        response = client.get(f"/api/sessions/shared/{token}")
    finally:
        client.app.dependency_overrides[get_current_user] = original_override
    assert response.status_code == 200, response.text
    assert response.json()["session_id"] == str(session_id)


def test_get_shared_inspect_requires_auth(
    audit_readiness_client_anonymous: TestClient,
) -> None:
    """The token is a CAPABILITY, not an authenticator — auth dep still gates."""
    response = audit_readiness_client_anonymous.get("/api/sessions/shared/some-token")
    assert response.status_code == 401


def test_get_shared_inspect_tampered_token_returns_401(
    audit_readiness_client_with_state: tuple[TestClient, UUID],
) -> None:
    client, session_id = audit_readiness_client_with_state
    token = _mint_token(client, session_id)
    tampered = token[:-2] + ("aa" if token[-2:] != "aa" else "bb")
    response = client.get(f"/api/sessions/shared/{tampered}")
    assert response.status_code == 401
    assert response.json() == {"detail": "Invalid or expired share token"}


def test_get_shared_inspect_expired_token_returns_401(
    audit_readiness_client_with_state: tuple[TestClient, UUID],
) -> None:
    """Token verifies cryptographically but is past its ``expires_at`` → 401.

    Mechanism: mint a real token via the route, decode its payload using the
    production signer, re-sign an identical payload with ``expires_at`` in the
    past. The resulting token is signature-valid against the live signing key
    AND structurally indistinguishable from a normally-minted token that has
    aged past its lifetime — exactly what the route must reject with 401.

    Exercises the full route-stack exception-translation path: the route
    calls ``service.resolve_token`` → ``signer.verify`` raises
    ``InvalidToken("token expired")`` → route maps to 401 with the same body
    shape as the tampered-token case (the error string is intentionally
    indistinguishable to a probing attacker, per the signer docstring).

    Phase 6A backend plan line 19a:966 mandates this coverage. Without it,
    a future broadening of the route's ``except InvalidToken`` clause (e.g.
    catching a wider exception) would silently regress an expired-token 401
    into a 500 with no test failure — the unit-level signer/service tests
    would still pass.

    The frozen-pydantic-WebSettings instance precludes the in-place lifetime
    mutation used by the corresponding unit test
    (``test_resolve_token_rejects_expired_token``), which uses a ``MagicMock``
    settings object. Re-signing with the production signer is the equivalent
    end-state and keeps the wire format identical to a naturally-expired
    token.
    """
    client, session_id = audit_readiness_client_with_state
    real_token = _mint_token(client, session_id)
    service = client.app.state.shareable_review_service
    signer = service._signer
    # Verify the real token to extract the live payload fields, then re-sign
    # an otherwise-identical payload with expires_at in the past. Using
    # ``created_at`` 1h ago + ``expires_at`` 1s ago mirrors the relative
    # ordering of a token that aged out under its normal lifetime.
    payload = signer.verify(real_token)
    expired_payload = ShareTokenPayload(
        version=payload.version,
        session_id=payload.session_id,
        state_id=payload.state_id,
        created_at=datetime.now(UTC) - timedelta(hours=1),
        expires_at=datetime.now(UTC) - timedelta(seconds=1),
        nonce_hex=payload.nonce_hex,
        payload_digest=payload.payload_digest,
        created_by_user_id=payload.created_by_user_id,
    )
    expired_token = signer.sign(expired_payload)
    response = client.get(f"/api/sessions/shared/{expired_token}")
    assert response.status_code == 401
    assert response.json() == {"detail": "Invalid or expired share token"}


def test_get_shared_inspect_blob_expired_returns_404(
    audit_readiness_client_with_state: tuple[TestClient, UUID],
) -> None:
    """Token verifies but payload-store blob has been reaped → 404."""
    client, session_id = audit_readiness_client_with_state
    response = client.post(f"/api/sessions/{session_id}/mark-ready-for-review")
    assert response.status_code == 200
    body = response.json()
    token = body["token"]
    digest_hex = body["payload_digest"].removeprefix("sha256:")
    # Reach into the wired payload store and delete the blob.
    payload_store = client.app.state.payload_store
    assert payload_store.delete(digest_hex), "blob should exist before deletion"
    response = client.get(f"/api/sessions/shared/{token}")
    assert response.status_code == 404


def test_mark_ready_for_review_audit_write_failure_returns_no_token(
    audit_readiness_client_with_state: tuple[TestClient, UUID],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If the audit insert raises, the request fails — no token returned, no blob exposed.

    Mechanism: monkey-patch the engine's ``begin`` to raise. The service's
    ``with self._sessions_db_engine.begin() as conn`` path then fails before
    any blob is written. TestClient defaults to ``raise_server_exceptions=True``,
    so the test catches the exception directly; the assertion that matters is
    that NO blob was written, confirming audit-first ordering.
    """
    client, session_id = audit_readiness_client_with_state

    # Track payload_store.store calls so we can prove no blob was written.
    payload_store = client.app.state.payload_store
    original_store = payload_store.store
    store_calls: list[bytes] = []

    def tracking_store(content: bytes) -> str:
        store_calls.append(content)
        return original_store(content)

    monkeypatch.setattr(payload_store, "store", tracking_store)

    # Break the engine's begin() to raise.
    service = client.app.state.shareable_review_service

    class _AuditWriteBoom(Exception): ...

    class _BadEngine:
        def begin(self):  # type: ignore[no-untyped-def]
            raise _AuditWriteBoom("audit write injected failure")

    monkeypatch.setattr(service, "_sessions_db_engine", _BadEngine())

    with pytest.raises(_AuditWriteBoom):
        client.post(f"/api/sessions/{session_id}/mark-ready-for-review")
    # CRITICAL: no blob was written.
    assert store_calls == [], "audit insert must precede blob write — blob should not exist when audit fails"
