"""Route-level telemetry tests for Phase 8 Task 2 Site 3.

PATCH /api/sessions/{session_id}/composer/preferences must emit
``record_session_switched`` when (and only when) the persisted
``trust_mode`` actually changes from the prior to the current state.

The audit row (``proposal_events_table`` ``trust_mode.changed``) fires
unconditionally on every PATCH, but the telemetry counter is a
**transition-rate** (guarded on ``prior.trust_mode !=
current.trust_mode``). This is distinct from the account-level
mode-opted-in/out emit at ``preferences/routes.py``, which is a
**set-rate** (fires whenever ``default_mode`` is in the PATCH body).
See ``20-phase-8-polish-and-telemetry.md`` §"Account-level scope
narrowing (B2.b — load-bearing)" for the contrast.

Q4 (combined-PATCH) pinning: a PATCH that changes both ``trust_mode``
AND ``density_default`` in a single request fires the counter exactly
once, attributed to the ``trust_mode`` change only. The density change
is not double-counted and must not suppress the counter.

Q8 (B1 audit-payload contract) pinning: the
``trust_mode.changed`` audit row's payload carries BOTH ``trust_mode``
(the new value) AND ``prior_trust_mode`` (the value the PATCH
overwrote). This locks the B1 audit-payload extension so a future PR
cannot silently revert the audit shape while keeping the telemetry
emit (the superset rule would be violated).

Fixture discipline (Q10): consumes the function-scoped ``test_client``
fixture from ``tests/unit/web/conftest.py``, which builds a fresh
``sessions_telemetry`` container per test.
"""

from __future__ import annotations

import json

from fastapi.testclient import TestClient
from sqlalchemy import select

from elspeth.web.sessions.models import proposal_events_table
from elspeth.web.sessions.telemetry import _FakeCounter, observed_value


def _patch_trust_mode(test_client: TestClient, session_id: str, **fields: object):
    return test_client.patch(
        f"/api/sessions/{session_id}/composer/preferences",
        json=fields,
    )


# ---------------------------------------------------------------------------
# Site 3 — emit on actual trust_mode change
# ---------------------------------------------------------------------------


def test_route_emits_session_switched_on_mode_change(test_client: TestClient) -> None:
    """PATCH that changes ``trust_mode`` from ``auto_commit`` to
    ``explicit_approve`` fires ``record_session_switched`` once with
    attributes ``{from_mode, to_mode}`` drawn from the per-session
    ``trust_mode`` vocabulary."""
    session = test_client.post("/api/sessions", json={"title": "TrustModeChange"}).json()

    telemetry = test_client.app.state.sessions_telemetry
    assert observed_value(telemetry.session_switched_total) == 0

    # New sessions default to ``trust_mode="auto_commit"`` (per
    # sessions_table column default). PATCH flips it to
    # ``explicit_approve``.
    response = _patch_trust_mode(
        test_client,
        session["id"],
        trust_mode="explicit_approve",
        density_default="medium",
    )
    assert response.status_code == 200
    assert response.json()["trust_mode"] == "explicit_approve"

    assert observed_value(telemetry.session_switched_total) == 1

    # Attribute contract: ``from_mode`` / ``to_mode`` drawn from the
    # per-session vocabulary (NOT ``guided`` / ``freeform``).  Narrow
    # the type via isinstance against ``_FakeCounter`` rather than
    # ``hasattr`` (CLAUDE.md unconditionally bans hasattr) — see the
    # canonical pattern in ``test_telemetry_phase8.py``.
    counter = telemetry.session_switched_total
    assert isinstance(counter, _FakeCounter), "test must run against _FakeCounter (build_sessions_telemetry(meter=None))"
    calls = counter.calls
    assert len(calls) == 1
    amount, attributes, _ctx = calls[0]
    assert amount == 1
    assert attributes == {"from_mode": "auto_commit", "to_mode": "explicit_approve"}


def test_route_does_not_emit_session_switched_when_mode_unchanged(
    test_client: TestClient,
) -> None:
    """PATCH that re-asserts the same ``trust_mode`` (and changes only
    ``density_default``) must NOT fire ``record_session_switched`` —
    the counter is a transition-rate, not a set-rate."""
    session = test_client.post("/api/sessions", json={"title": "NoModeChange"}).json()

    telemetry = test_client.app.state.sessions_telemetry

    # Default is ``auto_commit``; PATCH re-asserts it.
    response = _patch_trust_mode(
        test_client,
        session["id"],
        trust_mode="auto_commit",
        density_default="low",
    )
    assert response.status_code == 200

    assert observed_value(telemetry.session_switched_total) == 0


# ---------------------------------------------------------------------------
# Q4 — combined PATCH fires the counter exactly once
# ---------------------------------------------------------------------------


def test_route_emits_session_switched_once_when_mode_and_density_both_change(
    test_client: TestClient,
) -> None:
    """A PATCH that changes BOTH ``trust_mode`` AND ``density_default``
    in a single request fires ``record_session_switched`` exactly once
    (not zero, not twice). Q4 contract.

    The counter is attributed to the trust_mode change; the
    density change must not double-fire it (no separate
    density-related counter exists) and must not suppress it (the
    presence of a co-changed field doesn't gate the trust_mode emit).
    """
    session = test_client.post("/api/sessions", json={"title": "ComboPatch"}).json()
    telemetry = test_client.app.state.sessions_telemetry

    response = _patch_trust_mode(
        test_client,
        session["id"],
        trust_mode="explicit_approve",
        density_default="low",
    )
    assert response.status_code == 200

    assert observed_value(telemetry.session_switched_total) == 1


# ---------------------------------------------------------------------------
# Q8 — B1 audit-payload contract pinning
# ---------------------------------------------------------------------------


def test_trust_mode_changed_audit_event_records_prior_and_new_state(
    test_client: TestClient,
) -> None:
    """The ``trust_mode.changed`` audit row's payload must carry BOTH
    ``trust_mode`` (the new value) AND ``prior_trust_mode`` (the
    value the PATCH overwrote). This locks the B1 audit-payload
    extension landed in Phase 8a-1 (commit 417276bc2). A future PR
    that silently removes ``prior_trust_mode`` from the audit payload
    while keeping the telemetry emit would violate the superset rule
    (telemetry attributes must be a strict subset of audit-recorded
    reality); this test fails in that regression.
    """
    session = test_client.post("/api/sessions", json={"title": "AuditPayload"}).json()
    sid = session["id"]

    # Default is ``auto_commit``; flip to ``explicit_approve`` so the
    # audit row records a non-trivial prior→current transition.
    response = _patch_trust_mode(
        test_client,
        sid,
        trust_mode="explicit_approve",
        density_default="medium",
    )
    assert response.status_code == 200

    # Query the audit row directly via the route (avoids leaking
    # SQLAlchemy session-engine details into the test). The
    # ``proposal-events`` endpoint already deep-thaws ``payload`` to
    # native dicts (sessions/routes.py:328).
    events_resp = test_client.get(f"/api/sessions/{sid}/proposal-events")
    assert events_resp.status_code == 200
    events = events_resp.json()
    trust_mode_events = [e for e in events if e["event_type"] == "trust_mode.changed"]
    assert len(trust_mode_events) == 1, f"expected exactly one trust_mode.changed event, got {len(trust_mode_events)}"

    payload = trust_mode_events[0]["payload"]
    assert payload["trust_mode"] == "explicit_approve", payload
    assert payload["prior_trust_mode"] == "auto_commit", payload


# ---------------------------------------------------------------------------
# DB-level cross-check on the audit payload (avoids route round-trip)
# ---------------------------------------------------------------------------


def test_trust_mode_changed_audit_row_payload_via_db(test_client: TestClient) -> None:
    """Same contract as the previous test, but read the payload
    directly from ``proposal_events_table``. This guards against a
    response-shape bug in the ``/proposal-events`` route silently
    masking a real audit-row payload regression — the DB row is the
    source of truth.
    """
    session = test_client.post("/api/sessions", json={"title": "AuditPayloadDB"}).json()
    sid = session["id"]

    response = _patch_trust_mode(
        test_client,
        sid,
        trust_mode="explicit_approve",
        density_default="medium",
    )
    assert response.status_code == 200

    engine = test_client.app.state.phase3_engine
    with engine.connect() as conn:
        rows = conn.execute(
            select(proposal_events_table.c.payload)
            .where(proposal_events_table.c.session_id == sid)
            .where(proposal_events_table.c.event_type == "trust_mode.changed")
        ).all()
    assert len(rows) == 1
    raw_payload = rows[0][0]
    payload = json.loads(raw_payload) if isinstance(raw_payload, str | bytes) else raw_payload
    assert payload["trust_mode"] == "explicit_approve"
    assert payload["prior_trust_mode"] == "auto_commit"
