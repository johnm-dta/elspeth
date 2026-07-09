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
  * no persisted state (empty session)         -> lazy fresh wizard, NON-persisting
                                                  (identical to GET /guided's lazy path)
  * guided_session already present             -> idempotent: return it UNCHANGED,
                                                  including any terminal
  * persisted state, guided_session is None    -> THE CONVERSION: reseed a fresh
                                                  wizard as a new version; the prior
                                                  freeform version is recoverable
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, patch
from uuid import UUID

from tests.integration.web.composer.guided.test_step_3_e2e import (
    _confirm_wiring,
    _drive_to_step_3_propose_chain,
    _fake_llm_response_for_passthrough,
)
from tests.unit.web._sync_asgi_client import SyncASGITestClient as TestClient

# ---------------------------------------------------------------------------
# Helpers (self-contained; must not couple to another test module's fixtures)
# ---------------------------------------------------------------------------


def _create_session(client: TestClient, *, profile: str | None = None) -> str:
    resp = client.post("/api/sessions", json={"title": "convert-test"})
    assert resp.status_code == 201, resp.json()
    session_id = resp.json()["id"]
    if profile is not None:
        start = client.post(f"/api/sessions/{session_id}/guided/start", json={"profile": profile})
        assert start.status_code == 200, start.json()
    return session_id


def _convert_raw(client: TestClient, session_id: str) -> object:
    return client.post(f"/api/sessions/{session_id}/guided/convert")


def _convert(client: TestClient, session_id: str) -> dict:
    resp = _convert_raw(client, session_id)
    assert resp.status_code == 200, resp.json()
    return resp.json()


def _seed_freeform_state_with_work(client: TestClient, session_id: str) -> None:
    """Persist a freeform composition state that carries real work.

    ``composer_meta={}`` (no ``guided_session`` key) is what makes GET /guided
    400 — the exact precondition of the bug. A committed source stands in for
    "the operator did freeform composition work worth preserving".
    """
    from elspeth.web.sessions.protocol import CompositionStateData

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
        assert gs["history"] == []
        assert gs["terminal"] is None
        assert body["terminal"] is None
        assert body["next_turn"] is not None
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
            json={"state_id": freeform_version["id"]},
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


# ---------------------------------------------------------------------------
# Branch 2 — idempotency + terminal fidelity
# ---------------------------------------------------------------------------


class TestConvertIdempotent:
    def test_convert_on_active_guided_returns_it_unchanged(self, composer_test_client: TestClient) -> None:
        """Convert on a session already mid-wizard is a no-op return, NOT a reseed.

        Advancing to Step 2 first proves branch 2 returns the EXISTING session
        (step preserved) rather than clobbering it back to a fresh Step-1 seed.
        """
        client = composer_test_client
        session_id = _create_session(client, profile="tutorial")

        from tests.integration.web.composer.guided.test_step_3_e2e import _seed_blob

        _blob_id, storage_path = _seed_blob(client, session_id)
        client.get(f"/api/sessions/{session_id}/guided")
        # Answer Step 1 (choose csv) then submit the schema form -> reach Step 2.
        client.post(f"/api/sessions/{session_id}/guided/respond", json={"chosen": ["csv"]})
        advanced = client.post(
            f"/api/sessions/{session_id}/guided/respond",
            json={
                "edited_values": {
                    "plugin": "csv",
                    "options": {"path": storage_path, "schema": {"mode": "observed"}},
                    "observed_columns": ["text", "note"],
                    "sample_rows": [{"text": "Hello world", "note": "greeting"}],
                }
            },
        )
        assert advanced.status_code == 200, advanced.json()
        step_before = advanced.json()["guided_session"]["step"]
        assert step_before == "step_2_sink"

        body = _convert(client, session_id)
        assert body["guided_session"]["step"] == step_before
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

        with patch(
            "elspeth.web.composer.guided.chain_solver._litellm_acompletion",
            new_callable=AsyncMock,
            return_value=_fake_llm_response_for_passthrough(),
        ):
            _drive_to_step_3_propose_chain(client, session_id)
            client.post(f"/api/sessions/{session_id}/guided/respond", json={"chosen": ["accept"]})

        done = _confirm_wiring(client, session_id)
        assert done["terminal"]["kind"] == "completed"

        body = _convert(client, session_id)
        assert body["terminal"] is not None
        assert body["terminal"]["kind"] == "completed"
        assert body["guided_session"]["terminal"]["kind"] == "completed"
        # The turn/None contract's other half: a TERMINAL session has no live
        # turn, so next_turn stays null (mirrors GET /guided). The non-terminal
        # rebuild above and this null here together pin both branches.
        assert body["next_turn"] is None


# ---------------------------------------------------------------------------
# Branch 1 — empty session lazy, non-persisting
# ---------------------------------------------------------------------------


class TestConvertEmptySession:
    def test_convert_on_empty_session_is_lazy_and_non_persisting(self, composer_test_client: TestClient) -> None:
        """A brand-new session with no persisted state takes GET /guided's lazy
        path: a fresh in-memory wizard, composition_state=None, no version
        written (an empty graph must not start the version history)."""
        client = composer_test_client
        session_id = _create_session(client)

        body = _convert(client, session_id)
        assert body["guided_session"]["step"] == "step_1_source"
        assert body["composition_state"] is None

        versions = client.get(f"/api/sessions/{session_id}/state/versions")
        assert versions.status_code == 200, versions.json()
        assert versions.json() == []
