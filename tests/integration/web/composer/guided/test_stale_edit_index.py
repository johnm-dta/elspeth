"""Package A (A1) — stale ``step_3_edit_index`` must not crash GET /guided.

Reproduces the confirmed defect: the STEP_3 chat path replaces
``step_3_proposal`` without clearing ``step_3_edit_index``.  A stale index
pointing past the end of a *shorter* revised proposal made the GET /guided
turn rebuild raise IndexError — an unsanitised 500.

Two layers under test:

1. The STEP_3 chat apply clears ``step_3_edit_index`` in the same atomic
   replace that installs the replacement proposal.
2. Defence in depth: the GET /guided rebuild degrades a stale/out-of-range
   ``step_3_edit_index`` to no-edit-in-progress (renders ``propose_chain``)
   and sanitises InvariantError into a structured 500 like the POST path.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch
from uuid import uuid4

from elspeth.web.composer.guided.errors import InvariantError
from tests.integration.web.composer.guided.test_step_3_e2e import (
    _create_session,
    _drive_to_step_3_propose_chain,
    _respond,
)
from tests.unit.web._sync_asgi_client import SyncASGITestClient as TestClient


def _fake_chain_response(step_count: int) -> SimpleNamespace:
    """A LiteLLM-shaped emit_turn response proposing ``step_count`` passthrough steps."""
    steps = [
        {
            "plugin": "passthrough",
            "options": {"schema": {"mode": "observed"}},
            "rationale": f"step {i}: pass rows through unchanged",
        }
        for i in range(step_count)
    ]
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    tool_calls=[
                        SimpleNamespace(
                            function=SimpleNamespace(
                                name="emit_turn",
                                arguments=json.dumps(
                                    {
                                        "turn_type": "propose_chain",
                                        "payload": {
                                            "steps": steps,
                                            "why": "two passthrough steps to start; one after revision",
                                        },
                                    }
                                ),
                            )
                        )
                    ],
                )
            )
        ]
    )


class TestStaleEditIndexAfterChatRevise:
    def test_chat_revise_to_shorter_chain_clears_edit_index_and_get_survives(self, composer_test_client: TestClient) -> None:
        """Accept edit at index 1 → chat-revise to a 1-step chain → GET /guided.

        Before the fix the persisted ``step_3_edit_index=1`` outlived the
        2-step proposal, and GET /guided crashed with IndexError rebuilding
        ``steps[1]`` of the 1-step replacement.
        """
        client = composer_test_client
        session_id = _create_session(client)

        # Drive to STEP_3 with a TWO-step proposal.
        with patch(
            "elspeth.web.composer.guided.chain_solver._litellm_acompletion",
            new_callable=AsyncMock,
            return_value=_fake_chain_response(2),
        ):
            body, _blob_id, _output_path = _drive_to_step_3_propose_chain(client, session_id)
        assert body["next_turn"]["type"] == "propose_chain"
        assert len(body["next_turn"]["payload"]["steps"]) == 2

        # Stage an edit on the LAST step (index 1).
        edit_body = _respond(client, session_id, edit_step_index=1)
        assert edit_body["next_turn"]["type"] == "schema_form"
        staged = edit_body["composition_state"]["composer_meta"]["guided_session"]
        assert staged["step_3_edit_index"] == 1

        # Chat-revise the chain down to ONE step.
        with patch(
            "elspeth.web.composer.guided.chain_solver._litellm_acompletion",
            new_callable=AsyncMock,
            return_value=_fake_chain_response(1),
        ):
            chat_resp = client.post(
                f"/api/sessions/{session_id}/guided/chat",
                json={"message": "just one passthrough step please", "step_index": "step_3_transforms"},
            )
        assert chat_resp.status_code == 200, chat_resp.json()
        chat_body = chat_resp.json()
        assert chat_body["next_turn"]["type"] == "propose_chain"
        assert len(chat_body["next_turn"]["payload"]["steps"]) == 1

        # Fix layer 1: the same atomic replace that installed the replacement
        # proposal must have cleared the staged edit index.
        revised = chat_body["composition_state"]["composer_meta"]["guided_session"]
        assert revised["step_3_edit_index"] is None

        # Fix layer 2 (behavioural): GET /guided must not crash and must
        # render the propose_chain turn for the revised proposal.
        get_resp = client.get(f"/api/sessions/{session_id}/guided")
        assert get_resp.status_code == 200, get_resp.text
        get_body = get_resp.json()
        assert get_body["next_turn"] is not None
        assert get_body["next_turn"]["type"] == "propose_chain"
        assert len(get_body["next_turn"]["payload"]["steps"]) == 1


class TestGetGuidedInvariantSanitisation:
    def test_get_guided_rebuild_invariant_error_returns_sanitised_500(self, composer_test_client: TestClient) -> None:
        """An InvariantError from the GET turn rebuild must surface as the same
        static, sanitised 500 detail the POST path uses — never ``str(exc)``
        (which can embed ``{d!r}`` of Tier-1 records incl. Tier-3 sample rows).
        """
        client = composer_test_client
        session_id = _create_session(client)
        # First GET persists nothing (fresh session is non-mutating) but a
        # started session gives the route a state to rebuild from.
        start_resp = client.post(
            f"/api/sessions/{session_id}/guided/start",
            json={"profile": "live", "operation_id": str(uuid4())},
        )
        assert start_resp.status_code == 200, start_resp.json()

        secret_marker = "TIER3-SAMPLE-ROW-CONTENT"
        with patch(
            "elspeth.web.sessions.routes.composer.guided._build_get_guided_turn",
            side_effect=InvariantError(f"corrupted record {{'sample_rows': '{secret_marker}'}}"),
        ):
            resp = client.get(f"/api/sessions/{session_id}/guided")

        assert resp.status_code == 500
        detail = resp.json()["detail"]
        assert detail == "Server invariant violated. See application audit log for diagnostic detail."
        assert secret_marker not in resp.text
