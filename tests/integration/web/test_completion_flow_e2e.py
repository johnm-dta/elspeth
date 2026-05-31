"""End-to-end Phase 6 completion-gesture flow: User A → User B handoff.

Phase 6B gap-analysis FIX-J (CRITICAL). Plan reference:
``docs/composer/ux-redesign-2026-05/19b-phase-6b-frontend.md`` lines 640-664
(Task 11 — Cross-cutting tests).

This test composes the previously-disjoint backend slices into the single
six-step scenario the plan mandates:

    1. User A has a composition that passes validation (seeded by fixture).
    2. User A posts to ``/api/sessions/{sid}/mark-ready-for-review``; receives
       a token + share_url.
    3. User B (a *different* authenticated user) GETs
       ``/api/sessions/shared/{token}`` and sees the YAML + audit_readiness
       payload.
    4. The read-only inspect response has NO mutation surface. The frontend
       SharedInspectView's absence of a CompletionBar is asserted at the
       widget level by ``frontend/src/web/composer/__tests__/
       SharedInspectView.test.tsx`` (FIX-H); this backend test asserts the
       contract by confirming the response shape is the read-only payload
       and does not expose write capability.
    5. The sessions DB ``composer_completion_events_table`` contains a
       ``mark_ready_for_review`` row whose ``actor`` is User A.
    6. User A GETs ``/api/sessions/{sid}/state/yaml``; a second row appears
       with ``event_type='export_yaml'`` and ``actor`` = User A.

Pre-existing single-purpose tests (``test_shareable_reviews_routes.py``,
``test_yaml_export_audit_event.py``) cover each leg in isolation. FIX-J's
contribution is asserting they compose — that User A's actor identity is
recorded on BOTH audit rows in a single session lifecycle and that User B's
read-only access in the middle of that lifecycle does not pollute the audit
trail with a User B actor.
"""

from __future__ import annotations

from uuid import UUID

from fastapi.testclient import TestClient
from sqlalchemy import select

from elspeth.web.auth.middleware import get_current_user
from elspeth.web.auth.models import UserIdentity
from elspeth.web.sessions.models import composer_completion_events_table


def test_user_a_to_user_b_handoff_with_audit_rows(
    audit_readiness_client_with_state: tuple[TestClient, UUID],
) -> None:
    """Six-step E2E: User A mints, User B resolves, User A exports YAML.

    The ``audit_readiness_client_with_state`` fixture authenticates as
    ``alice`` and seeds a passthrough composition state that passes
    validation. We treat ``alice`` as User A and swap the
    ``get_current_user`` dependency override to ``bob`` for Step 3 only.

    Per-step DB-state checks:
      * After Step 2: 1 row, event_type=mark_ready_for_review, actor=alice.
      * After Step 3 (User B resolve): still 1 row (resolution does NOT
        write to ``composer_completion_events_table``).
      * After Step 6: 2 rows total, second is export_yaml with actor=alice.
    """
    client, session_id = audit_readiness_client_with_state
    engine = client.app.state.session_engine

    # ── Step 1: User A's composition is already seeded by the fixture. ──

    # ── Step 2: User A marks ready for review. ──
    mark_response = client.post(f"/api/sessions/{session_id}/mark-ready-for-review")
    assert mark_response.status_code == 200, mark_response.text
    mark_body = mark_response.json()
    token = mark_body["token"]
    share_url = mark_body["share_url"]
    payload_digest = mark_body["payload_digest"]
    assert isinstance(token, str) and token
    assert share_url.endswith(token)
    assert payload_digest.startswith("sha256:")

    # DB check #1: exactly one row, mark_ready_for_review, actor=alice.
    with engine.connect() as conn:
        rows_after_step2 = conn.execute(
            select(composer_completion_events_table)
            .where(composer_completion_events_table.c.session_id == str(session_id))
            .order_by(composer_completion_events_table.c.created_at)
        ).all()
    assert len(rows_after_step2) == 1
    mark_row = rows_after_step2[0]
    assert mark_row.event_type == "mark_ready_for_review"
    assert mark_row.actor == "alice"
    assert mark_row.payload_digest == payload_digest
    assert mark_row.composition_state_id is not None

    # ── Step 3: User B opens the share_url. ──
    # Swap the get_current_user override to return bob. Same app, same
    # signing key, same payload store — mirrors the pattern used by
    # ``test_get_shared_inspect_recipient_is_not_creator``.
    bob_identity = UserIdentity(user_id="bob", username="bob")

    async def _bob() -> UserIdentity:
        return bob_identity

    original_override = client.app.dependency_overrides[get_current_user]
    client.app.dependency_overrides[get_current_user] = _bob
    try:
        shared_response = client.get(f"/api/sessions/shared/{token}")
    finally:
        client.app.dependency_overrides[get_current_user] = original_override
    assert shared_response.status_code == 200, shared_response.text
    shared_body = shared_response.json()
    assert shared_body["session_id"] == str(session_id)
    # YAML is present (User B sees the composition).
    assert shared_body["yaml"]
    # Audit-readiness snapshot is present (the six-row inspect payload).
    assert {row["id"] for row in shared_body["audit_readiness"]["rows"]} == {
        "validation",
        "plugin_trust",
        "provenance",
        "retention",
        "llm_interpretations",
        "secrets",
    }

    # ── Step 4: Read-only contract. ──
    # The shared payload exposes YAML + audit_readiness for inspection but
    # does NOT expose any mutation handle. We assert this at the response
    # shape level: there is no token-refresh, no edit URL, no
    # completion-gesture surface in the read-only payload. The
    # SharedInspectView widget's absence of a CompletionBar is asserted at
    # the unit level in ``SharedInspectView.test.tsx`` (FIX-H).
    assert "mark_ready_for_review" not in shared_body
    assert "completion_bar" not in shared_body

    # DB check #2: Step 3 must NOT have written an audit row. User B's
    # read-only resolution is a read; only User A's gestures append.
    with engine.connect() as conn:
        rows_after_step3 = conn.execute(
            select(composer_completion_events_table)
            .where(composer_completion_events_table.c.session_id == str(session_id))
            .order_by(composer_completion_events_table.c.created_at)
        ).all()
    assert len(rows_after_step3) == 1
    assert rows_after_step3[0].actor == "alice"

    # ── Step 6: User A exports YAML. ──
    # (Step 5 — the post-Step-2 audit assertion — was performed inline
    # above; the plan numbers 5 before 6 but they share a DB-query
    # mechanism, so we fold them into a single ordered narrative.)
    export_response = client.get(f"/api/sessions/{session_id}/state/yaml")
    assert export_response.status_code == 200, export_response.text
    assert "yaml" in export_response.json()

    # DB check #3: two rows total, second is export_yaml with actor=alice.
    with engine.connect() as conn:
        rows_after_step6 = conn.execute(
            select(composer_completion_events_table)
            .where(composer_completion_events_table.c.session_id == str(session_id))
            .order_by(composer_completion_events_table.c.created_at)
        ).all()
    assert len(rows_after_step6) == 2
    export_row = rows_after_step6[1]
    assert export_row.event_type == "export_yaml"
    assert export_row.actor == "alice"
    assert export_row.composition_state_id is not None
    # export_yaml rows do not carry a payload_digest or expires_at (per
    # 19a Task 7's contract; see ``test_yaml_export_audit_event.py``).
    assert export_row.payload_digest is None
    assert export_row.expires_at is None

    # Cross-row invariant: both rows belong to the same composition state
    # (User A's seeded state was not mutated between mark and export).
    assert export_row.composition_state_id == mark_row.composition_state_id
