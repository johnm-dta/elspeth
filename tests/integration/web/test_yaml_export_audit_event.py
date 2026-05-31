"""Integration test for the Phase 6A B3 audit-event side effect on YAML export.

Phase 6A Task 7 (UX redesign 2026-05). The pre-existing
``GET /api/sessions/{session_id}/state/yaml`` route gains a Tier-1 audit
write to ``composer_completion_events_table`` with ``event_type='export_yaml'``.
Sync, crash-on-failure per CLAUDE.md audit primacy — no telemetry-class
exemption.
"""

from __future__ import annotations

from uuid import UUID

from fastapi.testclient import TestClient
from sqlalchemy import select

from elspeth.web.sessions.models import composer_completion_events_table


def test_yaml_export_records_audit_event(
    audit_readiness_client_with_state: tuple[TestClient, UUID],
) -> None:
    """GET /state/yaml writes an export_yaml row keyed by (session, state, actor)."""
    client, session_id = audit_readiness_client_with_state
    response = client.get(f"/api/sessions/{session_id}/state/yaml")
    assert response.status_code == 200, response.text
    assert "yaml" in response.json()
    # Verify the audit row landed.
    engine = client.app.state.session_engine
    with engine.connect() as conn:
        rows = conn.execute(
            select(composer_completion_events_table).where(composer_completion_events_table.c.session_id == str(session_id))
        ).all()
    assert len(rows) == 1
    row = rows[0]
    assert row.event_type == "export_yaml"
    assert row.actor == "alice"
    # composition_state_id is populated (UUID string).
    assert row.composition_state_id is not None
    # payload_digest and expires_at stay NULL for export_yaml.
    assert row.payload_digest is None
    assert row.expires_at is None


def test_yaml_export_appends_audit_row_per_call(
    audit_readiness_client_with_state: tuple[TestClient, UUID],
) -> None:
    """Each export adds a row. Append-only audit — distinct ids, distinct
    created_at timestamps."""
    client, session_id = audit_readiness_client_with_state
    r1 = client.get(f"/api/sessions/{session_id}/state/yaml")
    r2 = client.get(f"/api/sessions/{session_id}/state/yaml")
    assert r1.status_code == 200
    assert r2.status_code == 200
    engine = client.app.state.session_engine
    with engine.connect() as conn:
        rows = conn.execute(
            select(composer_completion_events_table).where(composer_completion_events_table.c.session_id == str(session_id))
        ).all()
    assert len(rows) == 2
    assert all(r.event_type == "export_yaml" for r in rows)
    # Distinct primary keys.
    assert rows[0].id != rows[1].id
