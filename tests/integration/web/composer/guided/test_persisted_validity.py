"""Guided intermediate checkpoints persist their real authoring validity."""

from __future__ import annotations

from tests.integration.web.composer.guided.test_respond import (
    _create_session,
    _get_guided,
    _respond,
    _seed_blob,
)
from tests.unit.web._sync_asgi_client import SyncASGITestClient as TestClient


class TestGuidedPersistedValidity:
    def test_intermediate_state_persists_invalid_with_errors_populated(self, composer_test_client: TestClient) -> None:
        """Reviewed facts are not a committed pipeline, so errors must be real."""
        client = composer_test_client
        session_id = _create_session(client)
        _seed_blob(client, session_id)

        _get_guided(client, session_id)
        configured = _respond(client, session_id, chosen=["csv"])
        prefilled = configured["next_turn"]["payload"]["prefilled"]
        body = _respond(
            client,
            session_id,
            edited_values={
                "plugin": "csv",
                "options": prefilled,
            },
        )

        state = body["composition_state"]
        assert state is not None
        assert state["sources"] == {}
        assert state["is_valid"] is False
        errors = state["validation_errors"]
        assert errors == ["guided_composition_invalid"]
