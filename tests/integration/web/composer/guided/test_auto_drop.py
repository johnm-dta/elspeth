"""Integration tests for Phase 5 Task 5.1: auto-drop to freeform on solver-exhausted.

Drives a full session through Steps 1 + 2 + Step 3 PROPOSE_CHAIN and then
exercises the repair-then-drop flow:

- Both-fail test: first LLM call proposes a chain that fails preview, second
  (repair) also fails → HTTP 200 with terminal kind=exited_to_freeform,
  reason=solver_exhausted; audit record for guided_dropped_to_freeform emitted.

- Repair-succeeds test: first LLM call proposes a bad chain, second proposes a
  working passthrough chain → HTTP 200 with terminal kind=completed; no drop
  event emitted.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch
from uuid import UUID

from tests.unit.web._sync_asgi_client import SyncASGITestClient as TestClient

# ---------------------------------------------------------------------------
# Helpers (mirrors test_step_3_e2e.py — kept local; no cross-file imports)
# ---------------------------------------------------------------------------


def _create_session(client: TestClient) -> str:
    resp = client.post("/api/sessions", json={"title": "auto-drop-test"})
    assert resp.status_code == 201, resp.json()
    return resp.json()["id"]


def _get_guided(client: TestClient, session_id: str) -> dict:
    resp = client.get(f"/api/sessions/{session_id}/guided")
    assert resp.status_code == 200, resp.json()
    return resp.json()


def _respond(client: TestClient, session_id: str, **kwargs) -> dict:
    resp = client.post(f"/api/sessions/{session_id}/guided/respond", json=kwargs)
    assert resp.status_code == 200, resp.json()
    return resp.json()


def _seed_blob(client: TestClient, session_id: str) -> tuple[str, str]:
    content = "text,note\nHello world,greeting\nGoodbye,farewell\n"
    resp = client.post(
        f"/api/sessions/{session_id}/blobs/inline",
        json={"filename": "data.csv", "content": content, "mime_type": "text/csv"},
    )
    assert resp.status_code == 201, resp.json()
    blob_id = resp.json()["id"]
    blob_service = client.app.state.blob_service
    record = asyncio.run(blob_service.get_blob(UUID(blob_id)))
    return blob_id, record.storage_path


def _outputs_path(client: TestClient, filename: str) -> str:
    data_dir: Path = client.app.state.settings.data_dir
    outputs_dir = data_dir / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)
    return str(outputs_dir / filename)


def _fake_llm_response_for_bad_plugin() -> SimpleNamespace:
    """LiteLLM-shaped response proposing a nonexistent plugin.

    ``definitely_not_a_real_plugin_xyzzy`` causes ``_validate_plugin_name`` to
    fail in ``_execute_set_pipeline``, making ``handle_step_3_chain_accept``
    return success=False — the failure mode exercised by the repair flow.
    """
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
                                            "steps": [
                                                {
                                                    "plugin": "definitely_not_a_real_plugin_xyzzy",
                                                    "options": {},
                                                    "rationale": "stub: guaranteed to fail validation",
                                                }
                                            ],
                                            "why": "stub that forces preview_pipeline failure",
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


def _fake_llm_response_for_passthrough() -> SimpleNamespace:
    """LiteLLM-shaped response proposing a valid passthrough chain (will succeed)."""
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
                                            "steps": [
                                                {
                                                    "plugin": "passthrough",
                                                    "options": {"schema": {"mode": "observed"}},
                                                    "rationale": "pass rows through unchanged",
                                                }
                                            ],
                                            "why": "source rows already match sink schema",
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


def _drive_to_step_3_propose_chain(client: TestClient, session_id: str) -> tuple[dict, str, str]:
    """Drive the wizard to the Step 3 ``propose_chain`` turn.

    Returns (response_body_at_step_3, blob_id, output_path).
    Uses ``required_fields=["text"]`` (no classifier keyword) so no recipe
    matches and the chain-solver entry seam fires.
    """
    blob_id, storage_path = _seed_blob(client, session_id)
    output_path = _outputs_path(client, "out.jsonl")

    _get_guided(client, session_id)
    _respond(client, session_id, chosen=["csv"])
    _respond(
        client,
        session_id,
        edited_values={
            "plugin": "csv",
            "options": {"path": storage_path, "schema": {"mode": "observed"}},
            "observed_columns": ["text", "note"],
            "sample_rows": [{"text": "Hello world", "note": "greeting"}],
        },
    )
    _respond(client, session_id, chosen=["json"])
    _respond(
        client,
        session_id,
        edited_values={
            "plugin": "json",
            "options": {
                "path": output_path,
                "schema": {"mode": "observed"},
                "collision_policy": "auto_increment",
            },
            "observed_columns": [],
            "sample_rows": [],
        },
    )
    body = _respond(
        client,
        session_id,
        chosen=["text"],
        custom_inputs=[],
    )
    return body, blob_id, output_path


def _get_tool_messages(client: TestClient, session_id: str) -> list:
    """Return all role=tool messages for this session from the session service."""
    service = client.app.state.session_service
    msgs = asyncio.run(service.get_messages(UUID(session_id), limit=None))
    return [m for m in msgs if m.role == "tool"]


def _extract_guided_drop_invocations(client: TestClient, session_id: str) -> list[dict]:
    """Return all guided_dropped_to_freeform invocation payloads for this session."""
    tool_messages = _get_tool_messages(client, session_id)
    drop_invocations = []
    for msg in tool_messages:
        if not msg.tool_calls:
            continue
        for tc in msg.tool_calls:
            invocation = tc.get("invocation", {})
            if invocation.get("tool_name") == "guided_dropped_to_freeform":
                # arguments_canonical is the JSON-encoded payload dict
                args_canonical = invocation.get("arguments_canonical", "{}")
                drop_invocations.append(json.loads(args_canonical))
    return drop_invocations


# ---------------------------------------------------------------------------
# Test: both initial and repair attempts fail → auto-drop to freeform (200)
# ---------------------------------------------------------------------------


class TestAutoDropOnSolverExhausted:
    def test_both_attempts_fail_returns_200_with_terminal_exited_to_freeform(self, composer_test_client: TestClient) -> None:
        """Both chain-solver attempts fail validation → HTTP 200, terminal=exited_to_freeform."""
        session_id = _create_session(composer_test_client)

        # Drive Steps 1 + 2 to reach PROPOSE_CHAIN.
        with patch(
            "elspeth.web.composer.guided.chain_solver._litellm_acompletion",
            new_callable=AsyncMock,
            return_value=_fake_llm_response_for_bad_plugin(),
        ):
            _drive_to_step_3_propose_chain(composer_test_client, session_id)

        # Accept the (bad) chain. The initial commit fails, triggering one LLM
        # repair call (also a bad plugin).  Both fail → auto-drop.
        with patch(
            "elspeth.web.composer.guided.chain_solver._litellm_acompletion",
            new_callable=AsyncMock,
            side_effect=[_fake_llm_response_for_bad_plugin()],
        ) as mock_llm:
            resp = composer_test_client.post(
                f"/api/sessions/{session_id}/guided/respond",
                json={"chosen": ["accept"]},
            )

        # 1. Status code must be 200 — auto-drop is a clean wizard outcome.
        assert resp.status_code == 200, resp.json()

        body = resp.json()

        # 2. Terminal in body must carry exited_to_freeform / solver_exhausted.
        terminal = body.get("terminal")
        assert terminal is not None, f"expected terminal in body, got: {body}"
        assert terminal["kind"] == "exited_to_freeform", f"unexpected kind: {terminal}"
        assert terminal["reason"] == "solver_exhausted", f"unexpected reason: {terminal}"
        assert terminal["pipeline_yaml"] is None

        # 3. GuidedSession terminal also set correctly.
        gs = body["guided_session"]
        assert gs["terminal"]["kind"] == "exited_to_freeform"
        assert gs["terminal"]["reason"] == "solver_exhausted"

        # 4. Repair call was made exactly once (one repair attempt).
        assert mock_llm.call_count == 1

        # 5. Audit record for guided_dropped_to_freeform must be present.
        drop_invocations = _extract_guided_drop_invocations(composer_test_client, session_id)
        assert len(drop_invocations) == 1, (
            f"expected exactly one guided_dropped_to_freeform audit record, got {len(drop_invocations)}: {drop_invocations}"
        )
        drop_args = drop_invocations[0]
        assert drop_args["drop_reason"] == "solver_exhausted"
        assert drop_args["prev_step"] == "step_3_transforms"
        assert "validation_result" in drop_args, f"spec §9.1 requires validation_result on solver_exhausted drops; got: {drop_args}"
        validation_result = drop_args["validation_result"]
        assert isinstance(validation_result, dict)
        assert validation_result["is_valid"] is False
        assert "errors" in validation_result  # may be empty list, not absent

    def test_no_next_turn_after_auto_drop(self, composer_test_client: TestClient) -> None:
        """After auto-drop the next_turn must be None — wizard is terminal."""
        session_id = _create_session(composer_test_client)

        with patch(
            "elspeth.web.composer.guided.chain_solver._litellm_acompletion",
            new_callable=AsyncMock,
            return_value=_fake_llm_response_for_bad_plugin(),
        ):
            _drive_to_step_3_propose_chain(composer_test_client, session_id)

        with patch(
            "elspeth.web.composer.guided.chain_solver._litellm_acompletion",
            new_callable=AsyncMock,
            side_effect=[_fake_llm_response_for_bad_plugin()],
        ):
            resp = composer_test_client.post(
                f"/api/sessions/{session_id}/guided/respond",
                json={"chosen": ["accept"]},
            )

        assert resp.status_code == 200
        body = resp.json()
        assert body["next_turn"] is None


# ---------------------------------------------------------------------------
# Test: first attempt fails, repair succeeds → COMPLETED (no drop event)
# ---------------------------------------------------------------------------


class TestRepairSucceeds:
    def test_first_fails_repair_succeeds_returns_completed(self, composer_test_client: TestClient) -> None:
        """First chain fails, second (repair) succeeds → HTTP 200, terminal=completed."""
        session_id = _create_session(composer_test_client)

        # Drive Steps 1 + 2 to reach PROPOSE_CHAIN (initial solve: bad plugin).
        with patch(
            "elspeth.web.composer.guided.chain_solver._litellm_acompletion",
            new_callable=AsyncMock,
            return_value=_fake_llm_response_for_bad_plugin(),
        ):
            _drive_to_step_3_propose_chain(composer_test_client, session_id)

        # Accept. Initial commit fails → repair attempt returns passthrough (valid).
        with patch(
            "elspeth.web.composer.guided.chain_solver._litellm_acompletion",
            new_callable=AsyncMock,
            side_effect=[_fake_llm_response_for_passthrough()],
        ) as mock_llm:
            resp = composer_test_client.post(
                f"/api/sessions/{session_id}/guided/respond",
                json={"chosen": ["accept"]},
            )

        assert resp.status_code == 200, resp.json()
        body = resp.json()

        # Terminal must be COMPLETED with rendered YAML.
        terminal = body.get("terminal")
        assert terminal is not None
        assert terminal["kind"] == "completed", f"unexpected kind: {terminal}"
        assert terminal["reason"] is None
        assert terminal["pipeline_yaml"] is not None
        assert "passthrough" in terminal["pipeline_yaml"]

        # next_turn is None — wizard is complete.
        assert body["next_turn"] is None

        # Exactly one repair call made.
        assert mock_llm.call_count == 1

        # No guided_dropped_to_freeform audit event emitted.
        drop_invocations = _extract_guided_drop_invocations(composer_test_client, session_id)
        assert drop_invocations == [], f"unexpected drop event in repair-succeeds path: {drop_invocations}"
