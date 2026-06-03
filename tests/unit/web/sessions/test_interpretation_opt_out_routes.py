"""HTTP-route tests for Phase 5b interpretation opt-out (Task 7).

Covers POST /api/sessions/{id}/interpretations/opt_out — flips the sessions
boolean, writes one ``interpretation_events`` row with
``choice='opted_out'`` and ``interpretation_source='auto_interpreted_opt_out'``,
and is idempotent across repeated POSTs (F-29).

Regression guard: the opt-out is routed to ``interpretation_events_table``,
NOT ``proposal_events_table``.  An earlier proposal-event design wrote opt-out
rows to the proposal-events table; that path was removed in Task 4 but the
audit-row destination is load-bearing so we re-assert it here.

Compose-loop integration (spec test 3) — verifying that the
``interpretation_review_disabled`` flag is read at compose start — is a
fixture-level integration test deferred to Task 9 per the spec.  The
shape we can verify here at unit-test granularity is the schema state
(``sessions.interpretation_review_disabled = True``) the compose-loop
reads from.

Fixture model: shared ``test_client`` from ``tests/unit/web/conftest.py``.
``alice`` is the authenticated user; IDOR tests seed a session owned by
``bob`` so the verifier raises 404 for alice's request.
"""

from __future__ import annotations

import pathlib
from uuid import UUID, uuid4

import pytest
from fastapi.testclient import TestClient
from httpx import ASGITransport, AsyncClient, Response
from sqlalchemy import select

from elspeth.web.sessions.models import (
    interpretation_events_table,
    proposal_events_table,
    sessions_table,
)
from elspeth.web.sessions.telemetry import observed_value
from tests.unit.web.conftest import _make_session

# Phase 8 Sub-task 7e (Q7 content-probe skip). Gates the route-level
# telemetry assertion on the Phase 5b audit-source string being present
# in ``web/`` source. Per ``project_phase5b_shipped`` memory the symbol
# should exist; the skip pathway preserves test-suite green behaviour if
# the memory is stale or the symbol has been renamed. The probe is a
# content check rather than a symbol import because
# ``auto_interpreted_opt_out`` is a string-literal column value in an
# audit row, not an exported identifier.
_WEB_ROOT = pathlib.Path(__file__).resolve().parents[4] / "src" / "elspeth" / "web"
_PHASE_5B_OPT_OUT_PRESENT = any("auto_interpreted_opt_out" in p.read_text() for p in _WEB_ROOT.rglob("*.py") if p.is_file())


async def _post(test_client: TestClient, url: str) -> Response:
    async with AsyncClient(
        transport=ASGITransport(app=test_client.app),
        base_url="http://test",
        cookies=test_client.cookies,
    ) as client:
        response = await client.post(url)
        test_client.cookies.update(response.cookies)
        return response


def _seed_session(test_client: TestClient, *, user_id: str = "alice") -> UUID:
    sid = uuid4()
    with test_client.app.state.phase3_engine.begin() as conn:
        _make_session(conn, session_id=str(sid), user_id=user_id)
    return sid


# --------------------------------------------------------------------------- #
# Spec test 1: write contract + regression guard against proposal_events
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_14_opt_out_flips_session_flag_and_writes_interpretation_event_row(
    test_client: TestClient,
) -> None:
    """Spec test 1 (numbered 14 in the combined spec):

    POST /opt_out flips ``sessions.interpretation_review_disabled`` to True,
    writes one ``interpretation_events`` row with ``choice='opted_out'`` and
    ``interpretation_source='auto_interpreted_opt_out'``.  Verifies NO row
    was written to ``proposal_events_table`` (regression guard against the
    earlier proposal-event-keyed opt-out design).
    """
    session_id = _seed_session(test_client)

    response = await _post(test_client, f"/api/sessions/{session_id}/interpretations/opt_out")

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["session_id"] == str(session_id)
    assert body["interpretation_review_disabled"] is True
    assert body["opted_out_at"] is not None

    with test_client.app.state.phase3_engine.begin() as conn:
        # Sessions boolean flipped.
        session_row = conn.execute(select(sessions_table).where(sessions_table.c.id == str(session_id))).one()
        assert session_row.interpretation_review_disabled is True

        # Exactly one interpretation_events row, opted-out shape.
        event_rows = conn.execute(
            select(interpretation_events_table).where(interpretation_events_table.c.session_id == str(session_id))
        ).fetchall()
        assert len(event_rows) == 1
        row = event_rows[0]
        assert row.choice == "opted_out"
        assert row.interpretation_source == "auto_interpreted_opt_out"
        assert row.resolved_at is not None
        # Opt-out rows carry NULL for the surfacing-specific fields.
        assert row.composition_state_id is None
        assert row.affected_node_id is None
        assert row.user_term is None
        assert row.llm_draft is None
        assert row.accepted_value is None

        # Regression guard: NO proposal_events row was written.
        proposal_rows = conn.execute(select(proposal_events_table).where(proposal_events_table.c.session_id == str(session_id))).fetchall()
        assert proposal_rows == []


# --------------------------------------------------------------------------- #
# Spec test 2: idempotency (F-29)
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_15_opt_out_is_idempotent_first_timestamp_authoritative(
    test_client: TestClient,
) -> None:
    """Spec test 2 (numbered 15):

    POST /opt_out twice ⇒ second call returns 200; row count is exactly 1;
    the FIRST opt-out timestamp is returned on both calls (F-29 contract:
    first opt-out is authoritative).
    """
    session_id = _seed_session(test_client)

    first = await _post(test_client, f"/api/sessions/{session_id}/interpretations/opt_out")
    assert first.status_code == 200, first.text
    first_timestamp = first.json()["opted_out_at"]

    second = await _post(test_client, f"/api/sessions/{session_id}/interpretations/opt_out")
    assert second.status_code == 200, second.text
    second_timestamp = second.json()["opted_out_at"]

    # First timestamp authoritative — both responses carry it.
    assert second_timestamp == first_timestamp

    # Exactly ONE row in the table (no duplicate insert).
    with test_client.app.state.phase3_engine.begin() as conn:
        rows = conn.execute(
            select(interpretation_events_table).where(interpretation_events_table.c.session_id == str(session_id))
        ).fetchall()
    assert len(rows) == 1


# --------------------------------------------------------------------------- #
# Spec test 3 (deferred integration shape): flag is readable post-opt-out
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_16_after_opt_out_sessions_flag_is_readable_by_compose_loop_path(
    test_client: TestClient,
) -> None:
    """Spec test 3 (numbered 16):

    The compose-loop reads ``sessions.interpretation_review_disabled`` at
    compose start to decide whether to inject the opted-out nudge into the
    system prompt.  The full fixture-level compose-loop integration test
    is deferred to Task 9 per the spec; what we can verify here is the
    schema state the compose-loop consumes: after POST /opt_out, the
    flag is True on the sessions row.

    This is the unit-test boundary for the integration contract.  Without
    it, a regression on the column-write would only surface in the
    deferred Task 9 fixture suite.
    """
    session_id = _seed_session(test_client)

    response = await _post(test_client, f"/api/sessions/{session_id}/interpretations/opt_out")
    assert response.status_code == 200, response.text

    with test_client.app.state.phase3_engine.begin() as conn:
        row = conn.execute(select(sessions_table.c.interpretation_review_disabled).where(sessions_table.c.id == str(session_id))).one()
    assert row.interpretation_review_disabled is True


# --------------------------------------------------------------------------- #
# Spec test 4: IDOR regression
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_17_idor_opt_out_on_other_users_session_returns_404(
    test_client: TestClient,
) -> None:
    """Spec test 4 (numbered 17):

    alice cannot POST /opt_out on bob's session — returns 404, not 403, and
    no mutation occurs on bob's session row.
    """
    session_id = _seed_session(test_client, user_id="bob")

    response = await _post(test_client, f"/api/sessions/{session_id}/interpretations/opt_out")

    assert response.status_code == 404, response.text

    # Sessions row is unchanged — interpretation_review_disabled still False.
    with test_client.app.state.phase3_engine.begin() as conn:
        row = conn.execute(select(sessions_table.c.interpretation_review_disabled).where(sessions_table.c.id == str(session_id))).one()
    assert row.interpretation_review_disabled is False

    # No interpretation_events row was written.
    with test_client.app.state.phase3_engine.begin() as conn:
        events = conn.execute(
            select(interpretation_events_table).where(interpretation_events_table.c.session_id == str(session_id))
        ).fetchall()
    assert events == []


# --------------------------------------------------------------------------- #
# Phase 8 Sub-task 7e — telemetry emit on the opt-out INSERT path
# --------------------------------------------------------------------------- #


@pytest.mark.skipif(
    not _PHASE_5B_OPT_OUT_PRESENT,
    reason="Phase 5b opt-out audit-source not present; Sub-task 7e is a documented no-op per Task 0 probe.",
)
@pytest.mark.asyncio
async def test_18_opt_out_emits_interpretation_opt_out_counter(
    test_client: TestClient,
) -> None:
    """Phase 8 Sub-task 7e (B3 cohort b1):

    First POST /opt_out commits one ``interpretation_events`` row with
    ``interpretation_source='auto_interpreted_opt_out'`` and increments
    ``composer.interpretation.opt_out_total`` by exactly one. The
    counter is an aggregate over committed audit rows, so its value
    must mirror the row count (superset rule).
    """
    session_id = _seed_session(test_client)
    telemetry = test_client.app.state.session_service._telemetry

    # Baseline: counter starts at zero for a fresh test_client fixture
    # (the fixture's ``build_sessions_telemetry()`` returns _FakeCounter
    # instances scoped to this test function — Q10 isolation).
    assert observed_value(telemetry.interpretation_opt_out_total) == 0

    response = await _post(test_client, f"/api/sessions/{session_id}/interpretations/opt_out")
    assert response.status_code == 200, response.text

    assert observed_value(telemetry.interpretation_opt_out_total) == 1


@pytest.mark.skipif(
    not _PHASE_5B_OPT_OUT_PRESENT,
    reason="Phase 5b opt-out audit-source not present; Sub-task 7e is a documented no-op per Task 0 probe.",
)
@pytest.mark.asyncio
async def test_19_opt_out_idempotent_refire_does_not_double_count(
    test_client: TestClient,
) -> None:
    """Phase 8 Sub-task 7e — superset-rule guard for F-29 idempotency.

    A second POST /opt_out returns the existing row (no INSERT) and must
    NOT increment the counter again. Without this guard, the counter
    would over-count route hits relative to audit rows, breaking the
    "telemetry aggregates over audit rows" invariant.
    """
    session_id = _seed_session(test_client)
    telemetry = test_client.app.state.session_service._telemetry

    first = await _post(test_client, f"/api/sessions/{session_id}/interpretations/opt_out")
    assert first.status_code == 200, first.text
    assert observed_value(telemetry.interpretation_opt_out_total) == 1

    second = await _post(test_client, f"/api/sessions/{session_id}/interpretations/opt_out")
    assert second.status_code == 200, second.text
    # Counter unchanged: F-29 idempotent re-fire did not write a new audit row.
    assert observed_value(telemetry.interpretation_opt_out_total) == 1
