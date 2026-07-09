"""Package A (A2) — guided persists must stamp the REAL validity.

Every guided persist site previously hardcoded ``is_valid=False,
validation_errors=None`` — guided-authored composition-state rows looked
permanently invalid in the DB (and, worse, invalid *without any errors*,
a combination the freeform persist path never produces).

The freeform persist path computes authoring validity from
``CompositionState.validate()`` and stores the error messages (or None when
empty).  Guided persists must mirror that: a genuinely-invalid intermediate
state stores ``is_valid=False`` with errors populated; a state persisted
after the pipeline is fully committed and valid stores ``is_valid=True``.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

from tests.integration.web.composer.guided.test_step_3_e2e import (
    _confirm_wiring,
    _create_session,
    _drive_to_step_3_propose_chain,
    _fake_llm_response_for_passthrough,
    _get_guided,
    _respond,
    _seed_blob,
)
from tests.unit.web._sync_asgi_client import SyncASGITestClient as TestClient


class TestGuidedPersistedValidity:
    def test_intermediate_state_persists_invalid_with_errors_populated(self, composer_test_client: TestClient) -> None:
        """After the Step-1 commit the pipeline has a source but no outputs —
        genuinely invalid — so the persisted row must carry the actual errors,
        not the old ``is_valid=False, validation_errors=None`` stamp."""
        client = composer_test_client
        session_id = _create_session(client)
        _blob_id, storage_path = _seed_blob(client, session_id)

        _get_guided(client, session_id)
        _respond(client, session_id, chosen=["csv"])
        body = _respond(
            client,
            session_id,
            edited_values={
                "plugin": "csv",
                "options": {"path": storage_path, "schema": {"mode": "observed"}},
                "observed_columns": ["text", "note"],
                "sample_rows": [{"text": "Hello world", "note": "greeting"}],
            },
        )

        state = body["composition_state"]
        assert state is not None
        assert state["is_valid"] is False
        errors = state["validation_errors"]
        assert isinstance(errors, list)
        assert errors, "genuinely-invalid intermediate state must persist its validation errors"

    def test_valid_committed_pipeline_persists_is_valid_true(self, composer_test_client: TestClient) -> None:
        """After Step-3 accept + wire confirm, the committed pipeline validates
        clean — the persisted row must say so instead of the hardcoded False.

        Uses the tutorial profile (advisor_checkpoints=False, a documented
        guided-mode exception — see profile.py TUTORIAL_PROFILE) so the wire
        confirm can reach COMPLETED without a live advisor sign-off service;
        the live profile's advisor gate is orthogonal to what this test
        exercises (persisted is_valid/validation_errors correctness)."""
        client = composer_test_client
        session_id = _create_session(client, profile="tutorial")

        with patch(
            "elspeth.web.composer.guided.chain_solver._litellm_acompletion",
            new_callable=AsyncMock,
            return_value=_fake_llm_response_for_passthrough(),
        ):
            _drive_to_step_3_propose_chain(client, session_id)
            accept_body = _respond(client, session_id, chosen=["accept"])

        assert accept_body["guided_session"]["step"] == "step_4_wire"
        accept_state = accept_body["composition_state"]
        assert accept_state is not None
        assert accept_state["is_valid"] is True
        assert accept_state["validation_errors"] is None

        confirm_body = _confirm_wiring(client, session_id)
        assert confirm_body["terminal"]["kind"] == "completed"
        confirm_state = confirm_body["composition_state"]
        assert confirm_state is not None
        assert confirm_state["is_valid"] is True
        assert confirm_state["validation_errors"] is None
