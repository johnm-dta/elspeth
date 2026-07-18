"""Integration tests for POST /api/sessions/{id}/guided/convert.

The "Switch to guided" affordance on a freeform session that has already done
composition work used to hit GET /guided, which 400s by design for a session
whose persisted CompositionState carries no ``guided_session`` (freeform).
There was no backend operation to move such a session into guided mode
(elspeth-e2c3dba6b5).

POST /guided/convert is that operation. Per the "fresh wizard + consent"
product decision it does NOT try to walk the retained freeform graph through
the wizard (GuidedSession.initial() starts at STEP_1_SOURCE and the step
handlers would clobber a pre-built graph). Instead it seeds a FRESH guided
wizard as a NEW composition-state version. The prior freeform pipeline stays
reachable via GET /state/versions + POST /state/revert — the same
recoverability contract as YAML import — so nothing is lost.

Branch behaviour:
  * no persisted state (empty session)         -> persist a fresh schema-8 wizard
                                                  with an immutable replay locator
  * guided_session already present             -> idempotent: return it UNCHANGED,
                                                  including any terminal
  * persisted state, guided_session is None    -> THE CONVERSION: reseed a fresh
                                                  wizard as a new version; the prior
                                                  freeform version is recoverable
"""

from __future__ import annotations

import asyncio
from dataclasses import replace
from unittest.mock import patch
from uuid import UUID, uuid4

import pytest

from elspeth.contracts.errors import AuditIntegrityError
from elspeth.contracts.freeze import deep_thaw
from elspeth.web.composer.guided.errors import InvariantError
from elspeth.web.composer.guided.state_machine import TerminalKind, TerminalState
from elspeth.web.sessions.guided_operations import guided_operation_request_hash
from elspeth.web.sessions.guided_replay import load_guided_json_payload
from elspeth.web.sessions.protocol import CompositionStateData, GuidedOperationCompleted
from elspeth.web.sessions.schemas import ConvertGuidedRequest
from tests.unit.web._sync_asgi_client import SyncASGITestClient as TestClient

# ---------------------------------------------------------------------------
# Helpers (self-contained; must not couple to another test module's fixtures)
# ---------------------------------------------------------------------------


def _create_session(client: TestClient, *, profile: str | None = None) -> str:
    resp = client.post("/api/sessions", json={"title": "convert-test"})
    assert resp.status_code == 201, resp.json()
    session_id = resp.json()["id"]
    if profile is not None:
        start = client.post(
            f"/api/sessions/{session_id}/guided/start",
            json={"profile": profile, "operation_id": str(uuid4())},
        )
        assert start.status_code == 200, start.json()
    return session_id


def _convert_raw(client: TestClient, session_id: str, *, operation_id: str | None = None) -> object:
    return client.post(
        f"/api/sessions/{session_id}/guided/convert",
        json={"operation_id": operation_id or str(uuid4())},
    )


def _convert(client: TestClient, session_id: str) -> dict:
    resp = _convert_raw(client, session_id)
    assert resp.status_code == 200, resp.json()
    return resp.json()


def _convert_outcome(client: TestClient, session_id: str, operation_id: str) -> GuidedOperationCompleted:
    request = ConvertGuidedRequest(operation_id=operation_id)
    request_hash = guided_operation_request_hash(
        session_id=UUID(session_id),
        kind="guided_convert",
        request=request,
    )
    outcome = asyncio.run(
        client.app.state.session_service.get_guided_operation(
            session_id=UUID(session_id),
            operation_id=operation_id,
            kind="guided_convert",
            request_hash=request_hash,
        )
    )
    assert isinstance(outcome, GuidedOperationCompleted)
    return outcome


def _seed_freeform_state_with_work(client: TestClient, session_id: str) -> None:
    """Persist a freeform composition state that carries real work.

    ``composer_meta={}`` (no ``guided_session`` key) is what makes GET /guided
    400 — the exact precondition of the bug. A committed source stands in for
    "the operator did freeform composition work worth preserving".
    """
    service = client.app.state.session_service
    freeform_state = CompositionStateData(
        sources={
            "src": {
                "plugin": "csv",
                "on_success": "main",
                "options": {"path": "/tmp/convert-test-source.csv"},
                "on_validation_failure": "discard",
            }
        },
        nodes=(),
        edges=(),
        outputs=(),
        metadata_={"name": "My freeform pipeline", "description": ""},
        is_valid=False,
        validation_errors=None,
        composer_meta={},  # No guided_session key -> freeform.
    )
    asyncio.run(service.save_composition_state(UUID(session_id), freeform_state, provenance="session_seed"))


# ---------------------------------------------------------------------------
# Branch 3 — the conversion + the load-bearing recoverability promise
# ---------------------------------------------------------------------------


class TestConvertFreeformWithWork:
    def test_convert_reseeds_fresh_wizard(self, composer_test_client: TestClient) -> None:
        """A worked freeform session converts into a fresh Step-1 wizard."""
        client = composer_test_client
        session_id = _create_session(client)
        _seed_freeform_state_with_work(client, session_id)

        # Precondition: the bug — GET /guided rejects the freeform session.
        pre = client.get(f"/api/sessions/{session_id}/guided")
        assert pre.status_code == 400, pre.json()

        body = _convert(client, session_id)

        gs = body["guided_session"]
        assert gs is not None
        assert gs["step"] == "step_1_source"
        assert len(gs["history"]) == 1
        assert gs["history"][0]["response_hash"] is None
        assert gs["terminal"] is None
        assert body["terminal"] is None
        assert body["next_turn"] is not None
        loaded_turn = load_guided_json_payload(
            client.app.state.payload_store,
            payload_id=gs["history"][0]["payload_hash"],
            purpose="turn",
        )
        assert deep_thaw(loaded_turn.payload) == body["next_turn"]["payload"]
        # The fresh wizard replaced the freeform graph: the new version carries
        # no source.
        state = body["composition_state"]
        assert state is not None
        assert not state["sources"]

        # And the session is now readable as guided (the bug is gone).
        post = client.get(f"/api/sessions/{session_id}/guided")
        assert post.status_code == 200, post.json()

    def test_prior_freeform_pipeline_is_recoverable_via_version_history(self, composer_test_client: TestClient) -> None:
        """The load-bearing promise behind "fresh wizard + consent": the freeform
        pipeline the conversion set aside is fully recoverable.

        Convert -> the prior freeform version is still listed -> reverting to it
        restores the graph AND drops the session back to freeform (its
        composer_meta had no guided_session, and revert copies composer_meta
        verbatim). If this cannot go green the consent copy is a lie.
        """
        client = composer_test_client
        session_id = _create_session(client)
        _seed_freeform_state_with_work(client, session_id)

        versions_before = client.get(f"/api/sessions/{session_id}/state/versions")
        assert versions_before.status_code == 200, versions_before.json()
        freeform_version = next(v for v in versions_before.json() if v["sources"])
        assert freeform_version["sources"]["src"]["plugin"] == "csv"

        _convert(client, session_id)

        # Revert to the pre-conversion freeform version.
        revert = client.post(
            f"/api/sessions/{session_id}/state/revert",
            json={"operation_id": str(uuid4()), "state_id": freeform_version["id"]},
        )
        assert revert.status_code == 200, revert.json()
        restored = revert.json()
        assert restored["sources"]["src"]["plugin"] == "csv"

        # Reverting restored freeform: GET /guided 400s again.
        after = client.get(f"/api/sessions/{session_id}/guided")
        assert after.status_code == 400, after.json()

    def test_convert_records_recovery_breadcrumb_message(self, composer_test_client: TestClient) -> None:
        """The destructive-but-recoverable conversion leaves a durable trail.

        The closed ``provenance`` enum cannot distinguish the conversion from a
        fresh seed without a governance change, so the conscious audit choice is
        a system message (mirroring revert_state) that names the recoverable
        version.
        """
        client = composer_test_client
        session_id = _create_session(client)
        _seed_freeform_state_with_work(client, session_id)

        _convert(client, session_id)

        service = client.app.state.session_service
        messages = asyncio.run(service.get_messages(UUID(session_id)))
        system_msgs = [m for m in messages if m.role == "system"]
        assert any("guided" in m.content.lower() and "version" in m.content.lower() for m in system_msgs), (
            f"expected a system breadcrumb naming the recoverable version; got {[m.content for m in system_msgs]}"
        )

    def test_same_operation_replays_exactly_without_duplicate_state_or_message(self, composer_test_client: TestClient) -> None:
        client = composer_test_client
        session_id = _create_session(client)
        _seed_freeform_state_with_work(client, session_id)
        operation_id = str(uuid4())

        first = _convert_raw(client, session_id, operation_id=operation_id)
        replay = _convert_raw(client, session_id, operation_id=operation_id)

        assert first.status_code == 200, first.json()
        assert replay.status_code == 200, replay.json()
        assert replay.json() == first.json()
        versions = asyncio.run(client.app.state.session_service.get_state_versions(UUID(session_id)))
        assert [version.version for version in versions] == [1, 2]
        messages = asyncio.run(client.app.state.session_service.get_messages(UUID(session_id)))
        assert len([message for message in messages if message.role == "system"]) == 1
        outcome = _convert_outcome(client, session_id, operation_id)
        assert str(outcome.result.state_id) == first.json()["composition_state"]["id"]


# ---------------------------------------------------------------------------
# Branch 2 — idempotency + terminal fidelity
# ---------------------------------------------------------------------------


class TestConvertIdempotent:
    def test_convert_on_active_guided_returns_it_unchanged(self, composer_test_client: TestClient) -> None:
        """Convert on an existing schema-8 checkpoint settles against that state."""
        client = composer_test_client
        session_id = _create_session(client, profile="tutorial")
        before = client.get(f"/api/sessions/{session_id}/guided")
        assert before.status_code == 200, before.json()

        body = _convert(client, session_id)
        assert body == before.json()
        # Branch 2 must REBUILD and return the live turn for a non-terminal step,
        # exactly like GET /guided (elspeth-e2c3dba6b5 review P2). A double-click
        # / cross-tab race lands the second "Switch to guided" here; if it returns
        # next_turn=None the frontend keeps guidedSession but drops guidedNextTurn,
        # isGuidedBuildActive goes false, and ChatPanel falls back to freeform.
        assert body["next_turn"] is not None, "idempotent convert on a non-terminal step must return the current turn, not null"

    def test_convert_on_completed_session_returns_completed_terminal(self, composer_test_client: TestClient) -> None:
        """Branch 2 must return a non-exit terminal faithfully.

        enterGuided routes completed / solver-exhausted / protocol-violation
        terminals through convert (only exited_to_freeform goes to reenter), and
        ChatPanel renders the CompletionSummary off ``terminal.kind ==
        'completed'``. If convert dropped or reseeded the terminal the completed
        surface would vanish.
        """
        client = composer_test_client
        session_id = _create_session(client, profile="tutorial")
        service = client.app.state.session_service
        current = asyncio.run(service.get_current_state(UUID(session_id)))
        assert current is not None
        from elspeth.web.sessions.routes._helpers import _state_from_record

        state = _state_from_record(current)
        assert state.guided_session is not None
        terminal_guided = replace(
            state.guided_session,
            terminal=TerminalState(kind=TerminalKind.COMPLETED, reason=None, pipeline_yaml="pipeline: {}\n"),
        )
        state_data = state.to_dict()
        asyncio.run(
            service.save_composition_state(
                UUID(session_id),
                CompositionStateData(
                    sources=state_data["sources"],
                    nodes=state_data["nodes"],
                    edges=state_data["edges"],
                    outputs=state_data["outputs"],
                    metadata_=state_data["metadata"],
                    is_valid=current.is_valid,
                    validation_errors=current.validation_errors,
                    composer_meta={"guided_session": terminal_guided.to_dict()},
                ),
                provenance="session_seed",
            )
        )

        body = _convert(client, session_id)
        assert body["terminal"] is not None
        assert body["terminal"]["kind"] == "completed"
        assert body["guided_session"]["terminal"]["kind"] == "completed"
        # The turn/None contract's other half: a TERMINAL session has no live
        # turn, so next_turn stays null (mirrors GET /guided). The non-terminal
        # rebuild above and this null here together pin both branches.
        assert body["next_turn"] is None


# ---------------------------------------------------------------------------
# Branch 1 — empty session persisted for immutable replay
# ---------------------------------------------------------------------------


class TestConvertEmptySession:
    def test_convert_on_empty_session_persists_retry_located_seed(self, composer_test_client: TestClient) -> None:
        """A brand-new session persists the schema-8 seed needed for replay."""
        client = composer_test_client
        session_id = _create_session(client)
        operation_id = str(uuid4())

        first = _convert_raw(client, session_id, operation_id=operation_id)
        replay = _convert_raw(client, session_id, operation_id=operation_id)

        assert first.status_code == 200, first.json()
        assert replay.status_code == 200, replay.json()
        assert replay.json() == first.json()
        body = first.json()
        assert body["guided_session"]["step"] == "step_1_source"
        assert body["composition_state"] is not None

        versions = client.get(f"/api/sessions/{session_id}/state/versions")
        assert versions.status_code == 200, versions.json()
        assert len(versions.json()) == 1
        outcome = _convert_outcome(client, session_id, operation_id)
        assert str(outcome.result.state_id) == body["composition_state"]["id"]

    def test_convert_requires_operation_id_before_persisting(self, composer_test_client: TestClient) -> None:
        client = composer_test_client
        session_id = _create_session(client)

        response = client.post(f"/api/sessions/{session_id}/guided/convert", json={})

        assert response.status_code == 422
        versions = client.get(f"/api/sessions/{session_id}/state/versions")
        assert versions.json() == []

    def test_replay_fails_closed_when_located_response_drifts(self, composer_test_client: TestClient) -> None:
        """Policy/live response drift is intentionally not stored or replayed.

        The operation stores only a state locator and strict response-domain
        hash. Rebuilding under a changed live projection must fail closed.
        """
        client = composer_test_client
        session_id = _create_session(client)
        operation_id = str(uuid4())
        first = _convert_raw(client, session_id, operation_id=operation_id)
        assert first.status_code == 200, first.json()
        original_turn = first.json()["next_turn"]
        drifted_turn = {
            **original_turn,
            "payload": {**original_turn["payload"], "question": "drifted after settlement"},
        }

        with (
            patch(
                "elspeth.web.sessions.routes.composer.guided._build_get_guided_turn",
                return_value=drifted_turn,
            ),
            pytest.raises(AuditIntegrityError, match="persisted occurrence"),
        ):
            _convert_raw(client, session_id, operation_id=operation_id)

    def test_deterministic_response_failure_is_settled_and_replayed(self, composer_test_client: TestClient) -> None:
        client = composer_test_client
        session_id = _create_session(client)
        operation_id = str(uuid4())

        with patch(
            "elspeth.web.sessions.routes.composer.guided._build_get_guided_turn",
            side_effect=InvariantError("tier-3 diagnostic must not escape"),
        ):
            first = _convert_raw(client, session_id, operation_id=operation_id)
        replay = _convert_raw(client, session_id, operation_id=operation_id)

        assert first.status_code == 500
        assert replay.status_code == 500
        assert (
            first.json()
            == replay.json()
            == {
                "detail": {
                    "error_type": "guided_operation_terminal_failure",
                    "failure_code": "operation_failed",
                    "detail": "The operation failed.",
                }
            }
        )
        assert "tier-3 diagnostic" not in first.text

    def test_audit_integrity_failure_is_settled_without_swallowing_diagnostic(
        self,
        composer_test_client: TestClient,
    ) -> None:
        client = composer_test_client
        session_id = _create_session(client)
        operation_id = str(uuid4())
        service = client.app.state.session_service

        from structlog.testing import capture_logs

        with (
            capture_logs() as cap_logs,
            patch.object(
                service,
                "save_state_for_guided_operation",
                side_effect=AuditIntegrityError("diagnostic retained for audit"),
            ),
        ):
            first = _convert_raw(client, session_id, operation_id=operation_id)

        replay = _convert_raw(client, session_id, operation_id=operation_id)
        assert first.status_code == replay.status_code == 500
        assert (
            first.json()
            == replay.json()
            == {
                "detail": {
                    "error_type": "guided_operation_terminal_failure",
                    "failure_code": "integrity_error",
                    "detail": "The operation failed an integrity check.",
                }
            }
        )
        events = [entry for entry in cap_logs if entry.get("event") == "guided.operation_terminal_failure"]
        assert len(events) == 1
        assert events[0]["exc_class"] == "AuditIntegrityError"
        assert events[0]["site"] == "post_guided_convert"
        assert events[0]["frames"]
        assert "diagnostic retained for audit" not in repr(events[0])
