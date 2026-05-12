"""Integration tests for Phase 5 Task 5.3: full-session audit emission contract.

Asserts that spec §9.1 audit events appear (or do NOT appear) across two
complete session lifecycles:

1. Recipe-match happy path:
   - guided_turn_emitted fires at least once
   - guided_turn_answered fires at least once
   - guided_step_advanced fires at least once
   - guided_dropped_to_freeform fires ZERO times

2. Auto-drop path (chain solver exhausted):
   - guided_dropped_to_freeform fires at least once
   - The drop event's ``drop_reason == "solver_exhausted"``
   - The drop event's ``prev_step == "step_3_transforms"``
   - The drop event carries a ``validation_result`` field (spec §9.1 MUST)

These tests cover the audit-emission contract, not HTTP response semantics;
for response-shape tests see test_respond.py and test_auto_drop.py.
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
# Guided audit discriminators (spec §9.1)
# ---------------------------------------------------------------------------

_GUIDED_TOOL_NAMES: frozenset[str] = frozenset(
    {
        "guided_turn_emitted",
        "guided_turn_answered",
        "guided_step_advanced",
        "guided_dropped_to_freeform",
    }
)

# ---------------------------------------------------------------------------
# Low-level helpers (mirrors test_auto_drop.py — no cross-file imports)
# ---------------------------------------------------------------------------


def _create_session(client: TestClient) -> str:
    resp = client.post("/api/sessions", json={"title": "audit-emission-test"})
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
    """Seed a CSV blob for recipe-match path.  Returns (blob_id, storage_path).

    Uses ``text`` + ``category`` columns so the ``classify-rows-llm-jsonl``
    recipe predicate is satisfied when the user declares ``category`` as a
    required field (the classify keyword is a substring of "category").
    """
    content = "text,category\nHello world,greeting\nGoodbye,farewell\n"
    resp = client.post(
        f"/api/sessions/{session_id}/blobs/inline",
        json={"filename": "data.csv", "content": content, "mime_type": "text/csv"},
    )
    assert resp.status_code == 201, resp.json()
    blob_id = resp.json()["id"]
    blob_service = client.app.state.blob_service
    record = asyncio.run(blob_service.get_blob(UUID(blob_id)))
    return blob_id, record.storage_path


def _seed_blob_no_recipe(client: TestClient, session_id: str) -> tuple[str, str]:
    """Seed a CSV blob for the auto-drop path (no recipe match).

    Uses ``text`` + ``note`` columns — ``note`` does not satisfy any
    classify/label/category keyword, so no recipe matches and the
    chain-solver path fires.
    """
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


# ---------------------------------------------------------------------------
# Audit-extraction helpers
# ---------------------------------------------------------------------------


def _get_tool_messages(client: TestClient, session_id: str) -> list:
    """Return all role=tool messages for this session from the session service."""
    service = client.app.state.session_service
    msgs = asyncio.run(service.get_messages(UUID(session_id), limit=None))
    return [m for m in msgs if m.role == "tool"]


def _extract_guided_invocations(client: TestClient, session_id: str) -> dict[str, list[dict]]:
    """Return a mapping of guided-mode tool_name → list of parsed argument payloads.

    Filters to the four guided-mode discriminators from spec §9.1.  Other tool
    names (``set_source``, ``set_output``, ``apply_pipeline_recipe``, etc.) are
    excluded so callers only see guided-protocol events.

    Returns:
        A dict keyed by tool_name, each value a list of parsed argument dicts
        (``arguments_canonical`` decoded from JSON).  Missing keys indicate zero
        events of that type.
    """
    tool_messages = _get_tool_messages(client, session_id)
    result: dict[str, list[dict]] = {name: [] for name in _GUIDED_TOOL_NAMES}
    for msg in tool_messages:
        if not msg.tool_calls:
            continue
        for tc in msg.tool_calls:
            invocation = tc.get("invocation", {})
            tool_name = invocation.get("tool_name")
            if tool_name not in _GUIDED_TOOL_NAMES:
                continue
            args_canonical = invocation.get("arguments_canonical", "{}")
            result[tool_name].append(json.loads(args_canonical))
    return result


# ---------------------------------------------------------------------------
# Scenario drivers
# ---------------------------------------------------------------------------


def _drive_to_recipe_offer(client: TestClient, session_id: str) -> tuple[dict, str]:
    """Drive the wizard to the Step 2.5 RECIPE_OFFER state.

    Uses ``text`` + ``category`` columns with the JSON sink so the
    ``classify-rows-llm-jsonl`` recipe predicate is satisfied.

    Returns (last_response_body, blob_id).  ``blob_id`` is needed for the
    recipe-accept slots where ``_resolve_source_blob`` reads the session DB.
    """
    blob_id, storage_path = _seed_blob(client, session_id)
    output_path = _outputs_path(client, "out_recipe.jsonl")

    _get_guided(client, session_id)
    _respond(client, session_id, chosen=["csv"])
    _respond(
        client,
        session_id,
        edited_values={
            "plugin": "csv",
            "options": {"path": storage_path, "schema": {"mode": "observed"}},
            "observed_columns": ["text", "category"],
            "sample_rows": [{"text": "Hello", "category": "greeting"}],
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
        chosen=["text", "category"],
        custom_inputs=[],
    )
    return body, blob_id


def _drive_to_step_3_propose_chain(client: TestClient, session_id: str) -> tuple[dict, str, str]:
    """Drive the wizard to the Step 3 ``propose_chain`` turn (no recipe).

    Uses ``required_fields=["text"]`` (no classify/label/category keyword) so
    no recipe matches and the chain-solver entry seam fires.

    Returns (response_body_at_step_3, blob_id, output_path).
    """
    blob_id, storage_path = _seed_blob_no_recipe(client, session_id)
    output_path = _outputs_path(client, "out_drop.jsonl")

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


def _fake_llm_bad_plugin() -> SimpleNamespace:
    """LiteLLM-shaped response proposing a nonexistent plugin (validation will fail)."""
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


# ---------------------------------------------------------------------------
# Test 1 — Recipe-match happy path audit emission
# ---------------------------------------------------------------------------


class TestRecipeMatchAuditEmission:
    """Assert spec §9.1 audit events fire (and don't fire) across the recipe-match happy path.

    The happy path terminates at Step 2.5 RECIPE_MATCH via recipe-accept.
    The chain solver is never invoked.  No LLM patches needed.
    """

    def test_recipe_accept_emits_turn_emitted_events(self, composer_test_client: TestClient) -> None:
        """guided_turn_emitted fires at least once across the full recipe-accept lifecycle."""
        session_id = _create_session(composer_test_client)
        recipe_body, blob_id = _drive_to_recipe_offer(composer_test_client, session_id)
        output_path = _outputs_path(composer_test_client, "out_recipe.jsonl")

        assert recipe_body["next_turn"]["type"] == "recipe_offer"
        offered_recipe = recipe_body["next_turn"]["payload"]["recipe_name"]

        _respond(
            composer_test_client,
            session_id,
            chosen=["accept"],
            edited_values={
                "recipe_name": offered_recipe,
                "slots": {
                    "source_blob_id": blob_id,
                    "classifier_template": "Classify: {{ row['text'] }}",
                    "model": "anthropic/claude-3.5-sonnet",
                    "api_key_secret": "OPENROUTER_API_KEY",
                    "required_input_fields": ["text"],
                    "label_field": "category",
                    "output_path": output_path,
                },
            },
        )

        guided_invocations = _extract_guided_invocations(composer_test_client, session_id)
        assert len(guided_invocations["guided_turn_emitted"]) >= 1, (
            f"expected at least one guided_turn_emitted event; got none. All guided events: {guided_invocations}"
        )

    def test_recipe_accept_emits_turn_answered_events(self, composer_test_client: TestClient) -> None:
        """guided_turn_answered fires at least once across the full recipe-accept lifecycle."""
        session_id = _create_session(composer_test_client)
        recipe_body, blob_id = _drive_to_recipe_offer(composer_test_client, session_id)
        output_path = _outputs_path(composer_test_client, "out_recipe.jsonl")

        offered_recipe = recipe_body["next_turn"]["payload"]["recipe_name"]

        _respond(
            composer_test_client,
            session_id,
            chosen=["accept"],
            edited_values={
                "recipe_name": offered_recipe,
                "slots": {
                    "source_blob_id": blob_id,
                    "classifier_template": "Classify: {{ row['text'] }}",
                    "model": "anthropic/claude-3.5-sonnet",
                    "api_key_secret": "OPENROUTER_API_KEY",
                    "required_input_fields": ["text"],
                    "label_field": "category",
                    "output_path": output_path,
                },
            },
        )

        guided_invocations = _extract_guided_invocations(composer_test_client, session_id)
        assert len(guided_invocations["guided_turn_answered"]) >= 1, (
            f"expected at least one guided_turn_answered event; got none. All guided events: {guided_invocations}"
        )

    def test_recipe_accept_emits_step_advanced_events(self, composer_test_client: TestClient) -> None:
        """guided_step_advanced fires at least once across the full recipe-accept lifecycle."""
        session_id = _create_session(composer_test_client)
        recipe_body, blob_id = _drive_to_recipe_offer(composer_test_client, session_id)
        output_path = _outputs_path(composer_test_client, "out_recipe.jsonl")

        offered_recipe = recipe_body["next_turn"]["payload"]["recipe_name"]

        _respond(
            composer_test_client,
            session_id,
            chosen=["accept"],
            edited_values={
                "recipe_name": offered_recipe,
                "slots": {
                    "source_blob_id": blob_id,
                    "classifier_template": "Classify: {{ row['text'] }}",
                    "model": "anthropic/claude-3.5-sonnet",
                    "api_key_secret": "OPENROUTER_API_KEY",
                    "required_input_fields": ["text"],
                    "label_field": "category",
                    "output_path": output_path,
                },
            },
        )

        guided_invocations = _extract_guided_invocations(composer_test_client, session_id)
        assert len(guided_invocations["guided_step_advanced"]) >= 1, (
            f"expected at least one guided_step_advanced event; got none. All guided events: {guided_invocations}"
        )

    def test_recipe_accept_emits_no_drop_events(self, composer_test_client: TestClient) -> None:
        """guided_dropped_to_freeform fires ZERO times on the recipe-accept happy path."""
        session_id = _create_session(composer_test_client)
        recipe_body, blob_id = _drive_to_recipe_offer(composer_test_client, session_id)
        output_path = _outputs_path(composer_test_client, "out_recipe.jsonl")

        offered_recipe = recipe_body["next_turn"]["payload"]["recipe_name"]

        final_body = _respond(
            composer_test_client,
            session_id,
            chosen=["accept"],
            edited_values={
                "recipe_name": offered_recipe,
                "slots": {
                    "source_blob_id": blob_id,
                    "classifier_template": "Classify: {{ row['text'] }}",
                    "model": "anthropic/claude-3.5-sonnet",
                    "api_key_secret": "OPENROUTER_API_KEY",
                    "required_input_fields": ["text"],
                    "label_field": "category",
                    "output_path": output_path,
                },
            },
        )

        # Confirm this is actually the COMPLETED terminal (not a drop).
        assert final_body["terminal"] is not None
        assert final_body["terminal"]["kind"] == "completed", f"expected terminal.kind=completed, got: {final_body['terminal']}"

        guided_invocations = _extract_guided_invocations(composer_test_client, session_id)
        assert guided_invocations["guided_dropped_to_freeform"] == [], (
            f"expected no guided_dropped_to_freeform events on happy path; got: {guided_invocations['guided_dropped_to_freeform']}"
        )


# ---------------------------------------------------------------------------
# Test 2 — Auto-drop path audit emission
# ---------------------------------------------------------------------------


class TestAutoDropAuditEmission:
    """Assert spec §9.1 audit events fire correctly on the solver-exhausted auto-drop path.

    Both the initial chain-solver call and the repair call propose an
    invalid plugin name.  The wizard auto-drops to freeform after the
    repair fails.
    """

    def test_auto_drop_emits_drop_event_with_correct_fields(self, composer_test_client: TestClient) -> None:
        """guided_dropped_to_freeform fires with required fields on solver-exhausted path.

        Asserts:
        - At least one guided_dropped_to_freeform event is present.
        - drop_reason == "solver_exhausted"
        - prev_step == "step_3_transforms"
        - validation_result is present (spec §9.1 MUST when drop_reason=solver_exhausted)
        - validation_result.is_valid is False
        - validation_result.errors is present (may be empty list, not absent)
        """
        session_id = _create_session(composer_test_client)

        # Drive Steps 1 + 2 to PROPOSE_CHAIN with a bad LLM response.
        with patch(
            "elspeth.web.composer.guided.chain_solver._litellm_acompletion",
            new_callable=AsyncMock,
            return_value=_fake_llm_bad_plugin(),
        ):
            _drive_to_step_3_propose_chain(composer_test_client, session_id)

        # Accept the (bad) chain — initial commit fails, repair also fails → auto-drop.
        with patch(
            "elspeth.web.composer.guided.chain_solver._litellm_acompletion",
            new_callable=AsyncMock,
            side_effect=[_fake_llm_bad_plugin()],
        ):
            final_resp = composer_test_client.post(
                f"/api/sessions/{session_id}/guided/respond",
                json={"chosen": ["accept"]},
            )

        assert final_resp.status_code == 200, final_resp.json()
        body = final_resp.json()

        # Confirm the session actually dropped (not a different terminal).
        terminal = body.get("terminal")
        assert terminal is not None, f"expected terminal in response body, got: {body}"
        assert terminal["kind"] == "exited_to_freeform", f"expected terminal.kind=exited_to_freeform, got: {terminal}"

        # Extract guided-mode audit events.
        guided_invocations = _extract_guided_invocations(composer_test_client, session_id)
        drop_events = guided_invocations["guided_dropped_to_freeform"]

        assert len(drop_events) >= 1, (
            f"expected at least one guided_dropped_to_freeform audit event; got none. All guided events: {guided_invocations}"
        )

        drop_args = drop_events[0]

        assert drop_args["drop_reason"] == "solver_exhausted", f"expected drop_reason=solver_exhausted; got: {drop_args}"

        assert drop_args["prev_step"] == "step_3_transforms", f"expected prev_step=step_3_transforms; got: {drop_args}"

        assert "validation_result" in drop_args, (
            f"spec §9.1 requires validation_result on solver_exhausted drops; field is absent. Drop event payload: {drop_args}"
        )

        validation_result = drop_args["validation_result"]
        assert isinstance(validation_result, dict), f"validation_result must be a dict; got {type(validation_result)}: {validation_result}"
        assert validation_result["is_valid"] is False, (
            f"validation_result.is_valid must be False (pipeline was invalid); got: {validation_result}"
        )
        assert "errors" in validation_result, f"validation_result must have an 'errors' key; got: {validation_result}"

    def test_auto_drop_also_emits_turn_answered_events(self, composer_test_client: TestClient) -> None:
        """guided_turn_answered fires at least once on the auto-drop path (responds were made)."""
        session_id = _create_session(composer_test_client)

        with patch(
            "elspeth.web.composer.guided.chain_solver._litellm_acompletion",
            new_callable=AsyncMock,
            return_value=_fake_llm_bad_plugin(),
        ):
            _drive_to_step_3_propose_chain(composer_test_client, session_id)

        with patch(
            "elspeth.web.composer.guided.chain_solver._litellm_acompletion",
            new_callable=AsyncMock,
            side_effect=[_fake_llm_bad_plugin()],
        ):
            composer_test_client.post(
                f"/api/sessions/{session_id}/guided/respond",
                json={"chosen": ["accept"]},
            )

        guided_invocations = _extract_guided_invocations(composer_test_client, session_id)
        assert len(guided_invocations["guided_turn_answered"]) >= 1, (
            f"expected at least one guided_turn_answered event on auto-drop path; got none. All guided events: {guided_invocations}"
        )
