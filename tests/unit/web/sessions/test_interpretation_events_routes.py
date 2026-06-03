"""HTTP-route tests for Phase 5b interpretation events (Tasks 6 + opt_out_summary).

Covers POST /resolve, GET /interpretations, and GET /opt_out_summary on the
:func:`create_session_router` surface.  Test numbering mirrors the spec at
``docs/composer/ux-redesign-2026-05/18a-phase-5b-backend.md`` Task 6 §Tests
(spec tests 1-13) plus the opt_out_summary tests pulled in here (F-22, 18-20).

Fixture model: the shared ``test_client`` from ``tests/unit/web/conftest.py``
provides an in-memory SQLite engine, ``SessionServiceImpl``, and an auth
override returning ``alice``.  IDOR tests seed a session owned by ``bob`` so
the verifier raises 404 for alice's request.

The test file does NOT bypass any production code paths — every assertion
goes through the registered route handler, the service method, and the SQL
schema as a full stack.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from uuid import UUID, uuid4

import pytest
from fastapi.testclient import TestClient
from httpx import ASGITransport, AsyncClient, Response

from elspeth.contracts.composer_interpretation import (
    InterpretationChoice,
    InterpretationKind,
    InterpretationSource,
)
from elspeth.contracts.enums import CreationModality
from elspeth.web.composer.state import CompositionState, NodeSpec, PipelineMetadata, SourceSpec
from elspeth.web.interpretation_state import INTERPRETATION_REQUIREMENTS_KEY, SOURCE_AUTHORING_KEY, SOURCE_COMPONENT_ID
from elspeth.web.sessions.protocol import CompositionStateData
from elspeth.web.sessions.service import SessionServiceImpl
from tests.unit.web.conftest import _make_session


async def _post(test_client: TestClient, url: str, *, json: dict[str, Any]) -> Response:
    async with AsyncClient(
        transport=ASGITransport(app=test_client.app),
        base_url="http://test",
        cookies=test_client.cookies,
    ) as client:
        response = await client.post(url, json=json)
        test_client.cookies.update(response.cookies)
        return response


async def _get(test_client: TestClient, url: str) -> Response:
    async with AsyncClient(
        transport=ASGITransport(app=test_client.app),
        base_url="http://test",
        cookies=test_client.cookies,
    ) as client:
        response = await client.get(url)
        test_client.cookies.update(response.cookies)
        return response


def _llm_node(
    *,
    node_id: str = "llm_transform_1",
    user_term: str = "cool",
    model: str | None = "anthropic/claude-opus-4-7",
    model_version: str | None = "2026-05-01",
) -> dict[str, Any]:
    """Return a production-serialized LLM node carrying one placeholder.

    Shape note: ``prompt_template`` lives inside ``options`` because that is
    the field ``_patch_llm_transform_prompt`` reads and the field NodeSpec
    surfaces to the runtime YAML generator. The returned dict comes from
    ``CompositionState.to_dict()`` so route tests exercise the production
    ``node_type``/``plugin`` discriminator rather than a private ``kind``
    fixture.
    """
    options: dict[str, Any] = {
        "prompt_template": f"Rate how {{{{interpretation:{user_term}}}}} this is.",
    }
    if model is not None:
        options["model"] = model
    if model_version is not None:
        options["model_version"] = model_version
    state = CompositionState(
        source=None,
        nodes=(
            NodeSpec(
                id=node_id,
                node_type="transform",
                plugin="llm",
                input="input",
                on_success="out",
                on_error="quarantine",
                options=options,
                condition=None,
                routes=None,
                fork_to=None,
                branches=None,
                policy=None,
                merge=None,
            ),
        ),
        edges=(),
        outputs=(),
        metadata=PipelineMetadata(name="Phase 5b Routes Test", description=""),
        version=1,
    )
    node = state.to_dict()["nodes"][0]
    return node


def _prompt_template_review_node(*, node_id: str = "identify_colour") -> dict[str, Any]:
    prompt_template = "Read {{ row.html }} and return JSON."
    state = CompositionState(
        source=None,
        nodes=(
            NodeSpec(
                id=node_id,
                node_type="transform",
                plugin="llm",
                input="rows",
                on_success="out",
                on_error="quarantine",
                options={
                    "prompt_template": prompt_template,
                    INTERPRETATION_REQUIREMENTS_KEY: [
                        {
                            "id": "prompt_template_review",
                            "kind": InterpretationKind.LLM_PROMPT_TEMPLATE.value,
                            "user_term": f"llm_prompt_template:{node_id}",
                            "status": "pending",
                            "draft": prompt_template,
                            "event_id": None,
                            "accepted_value": None,
                            "accepted_artifact_hash": None,
                            "resolved_prompt_template_hash": None,
                        }
                    ],
                },
                condition=None,
                routes=None,
                fork_to=None,
                branches=None,
                policy=None,
                merge=None,
            ),
        ),
        edges=(),
        outputs=(),
        metadata=PipelineMetadata(name="Phase 5b Routes Test", description=""),
        version=1,
    )
    return state.to_dict()["nodes"][0]


def _llm_generated_source() -> dict[str, Any]:
    state = CompositionState(
        source=SourceSpec(
            plugin="csv",
            on_success="rows",
            options={
                "path": "/tmp/generated.csv",
                SOURCE_AUTHORING_KEY: {
                    "modality": CreationModality.LLM_GENERATED.value,
                    "content_hash": "0" * 64,
                    "review_event_id": None,
                    "resolved_kind": None,
                },
                INTERPRETATION_REQUIREMENTS_KEY: [
                    {
                        "id": "source_review",
                        "kind": InterpretationKind.INVENTED_SOURCE.value,
                        "user_term": "inline_source_url_list",
                        "status": "pending",
                        "draft": "https://example.gov.au",
                        "event_id": None,
                        "accepted_value": None,
                        "accepted_artifact_hash": None,
                        "resolved_prompt_template_hash": None,
                    }
                ],
            },
            on_validation_failure="quarantine",
        ),
        nodes=(),
        edges=(),
        outputs=(),
        metadata=PipelineMetadata(name="Phase 5b Routes Test", description=""),
        version=1,
    )
    source = state.to_dict()["source"]
    assert source is not None
    return source


async def _seed_session_with_pending_event(
    test_client: TestClient,
    *,
    session_id: UUID | None = None,
    user_id: str = "alice",
    node: dict[str, Any] | None = None,
    user_term: str = "cool",
    kind: InterpretationKind = InterpretationKind.VAGUE_TERM,
    llm_draft: str = "Innovative and creative",
) -> dict[str, Any]:
    """Insert a session row + composition state + pending interpretation event."""
    sid = session_id if session_id is not None else uuid4()
    node = node if node is not None else _llm_node(user_term=user_term)
    service: SessionServiceImpl = test_client.app.state.session_service
    with test_client.app.state.phase3_engine.begin() as conn:
        _make_session(conn, session_id=str(sid), user_id=user_id)
    state = await service.save_composition_state(
        sid,
        CompositionStateData(
            nodes=[node],
            metadata_=CompositionState(
                source=None,
                nodes=(),
                edges=(),
                outputs=(),
                metadata=PipelineMetadata(name="Phase 5b Routes Test", description=""),
                version=1,
            ).to_dict()["metadata"],
            is_valid=True,
        ),
        provenance="tool_call",
    )
    event = await service.create_pending_interpretation_event(
        session_id=sid,
        composition_state_id=state.id,
        affected_node_id=node["id"],
        tool_call_id="call_42",
        user_term=user_term,
        kind=kind,
        llm_draft=llm_draft,
        model_identifier="anthropic/claude-opus-4-7",
        model_version="2026-05-01",
        provider="anthropic",
        composer_skill_hash="a" * 64,
    )
    return {"session_id": sid, "state": state, "event": event}


async def _seed_session_with_source_pending_event(test_client: TestClient) -> dict[str, Any]:
    sid = uuid4()
    service: SessionServiceImpl = test_client.app.state.session_service
    with test_client.app.state.phase3_engine.begin() as conn:
        _make_session(conn, session_id=str(sid), user_id="alice")
    state = await service.save_composition_state(
        sid,
        CompositionStateData(
            source=_llm_generated_source(),
            nodes=[],
            metadata_={"name": "Phase 5b Routes Test", "description": ""},
            is_valid=True,
        ),
        provenance="tool_call",
    )
    event = await service.create_pending_interpretation_event(
        session_id=sid,
        composition_state_id=state.id,
        affected_node_id=SOURCE_COMPONENT_ID,
        tool_call_id="call_source_review",
        user_term="inline_source_url_list",
        kind=InterpretationKind.INVENTED_SOURCE,
        llm_draft="https://example.gov.au",
        model_identifier="anthropic/claude-opus-4-7",
        model_version="2026-05-01",
        provider="anthropic",
        composer_skill_hash="a" * 64,
    )
    return {"session_id": sid, "state": state, "event": event}


# --------------------------------------------------------------------------- #
# POST /resolve happy paths
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_01_resolve_accepted_as_drafted_returns_resolved_event_and_advanced_state(
    test_client: TestClient,
) -> None:
    """Spec test 1: accepted_as_drafted ⇒ accepted_value == llm_draft and version+1."""
    seeded = await _seed_session_with_pending_event(test_client)
    session_id = seeded["session_id"]
    event_id = seeded["event"].id
    initial_version = seeded["state"].version

    response = await _post(
        test_client,
        f"/api/sessions/{session_id}/interpretations/{event_id}/resolve",
        json={"choice": "accepted_as_drafted"},
    )

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["event"]["choice"] == "accepted_as_drafted"
    assert body["event"]["kind"] == "vague_term"
    assert body["event"]["accepted_value"] == "Innovative and creative"
    assert body["event"]["resolved_at"] is not None
    # Composition state version advances by 1 (interpretation_resolve provenance).
    assert body["new_state"]["version"] == initial_version + 1


@pytest.mark.asyncio
async def test_02_resolve_amended_returns_amended_value_as_accepted(
    test_client: TestClient,
) -> None:
    """Spec test 2: amended + valid amended_value ⇒ accepted_value == amended_value."""
    seeded = await _seed_session_with_pending_event(test_client)
    session_id = seeded["session_id"]
    event_id = seeded["event"].id

    response = await _post(
        test_client,
        f"/api/sessions/{session_id}/interpretations/{event_id}/resolve",
        json={"choice": "amended", "amended_value": "Strikingly original"},
    )

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["event"]["choice"] == "amended"
    assert body["event"]["accepted_value"] == "Strikingly original"


@pytest.mark.asyncio
async def test_19_resolve_records_runtime_model_snapshot_from_current_state(
    test_client: TestClient,
) -> None:
    """F-19 supplementary: runtime_model_*_at_resolve carry node config values."""
    seeded = await _seed_session_with_pending_event(test_client)
    session_id = seeded["session_id"]
    event_id = seeded["event"].id

    response = await _post(
        test_client,
        f"/api/sessions/{session_id}/interpretations/{event_id}/resolve",
        json={"choice": "accepted_as_drafted"},
    )

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["event"]["runtime_model_identifier_at_resolve"] == "anthropic/claude-opus-4-7"
    assert body["event"]["runtime_model_version_at_resolve"] == "2026-05-01"


@pytest.mark.asyncio
async def test_19b_resolve_runtime_model_nulls_when_node_has_no_model_config(
    test_client: TestClient,
) -> None:
    """F-19 absence handling: missing ``model`` key ⇒ NULL on the audit row."""
    seeded = await _seed_session_with_pending_event(
        test_client,
        node=_llm_node(model=None, model_version=None),
    )
    session_id = seeded["session_id"]
    event_id = seeded["event"].id

    response = await _post(
        test_client,
        f"/api/sessions/{session_id}/interpretations/{event_id}/resolve",
        json={"choice": "accepted_as_drafted"},
    )

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["event"]["runtime_model_identifier_at_resolve"] is None
    assert body["event"]["runtime_model_version_at_resolve"] is None


# --------------------------------------------------------------------------- #
# POST /resolve validation failures (Pydantic 422)
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_03_resolve_amended_without_amended_value_returns_422(
    test_client: TestClient,
) -> None:
    """Spec test 3: choice=amended without amended_value → 422 (model_validator)."""
    seeded = await _seed_session_with_pending_event(test_client)
    response = await _post(
        test_client,
        f"/api/sessions/{seeded['session_id']}/interpretations/{seeded['event'].id}/resolve",
        json={"choice": "amended"},
    )
    assert response.status_code == 422, response.text


@pytest.mark.asyncio
async def test_04_resolve_accepted_with_amended_value_returns_422(
    test_client: TestClient,
) -> None:
    """Spec test 4: choice=accepted_as_drafted WITH amended_value → 422."""
    seeded = await _seed_session_with_pending_event(test_client)
    response = await _post(
        test_client,
        f"/api/sessions/{seeded['session_id']}/interpretations/{seeded['event'].id}/resolve",
        json={"choice": "accepted_as_drafted", "amended_value": "drift"},
    )
    assert response.status_code == 422, response.text


@pytest.mark.asyncio
async def test_05_resolve_opted_out_choice_returns_422(test_client: TestClient) -> None:
    """Spec test 5: choice=opted_out → 422 (Literal restricts the field set)."""
    seeded = await _seed_session_with_pending_event(test_client)
    response = await _post(
        test_client,
        f"/api/sessions/{seeded['session_id']}/interpretations/{seeded['event'].id}/resolve",
        json={"choice": "opted_out"},
    )
    assert response.status_code == 422, response.text


# --------------------------------------------------------------------------- #
# POST /resolve semantic-error mappings
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_06_double_resolve_returns_409(test_client: TestClient) -> None:
    """Spec test 6: TOCTOU — second resolve for same event → 409 (already resolved)."""
    seeded = await _seed_session_with_pending_event(test_client)
    url = f"/api/sessions/{seeded['session_id']}/interpretations/{seeded['event'].id}/resolve"
    first = await _post(test_client, url, json={"choice": "accepted_as_drafted"})
    assert first.status_code == 200, first.text

    second = await _post(test_client, url, json={"choice": "amended", "amended_value": "different"})
    assert second.status_code == 409, second.text


@pytest.mark.asyncio
async def test_07_resolve_unknown_event_id_returns_404(test_client: TestClient) -> None:
    """Spec test 7: unknown event_id → 404 (IDOR-safe; not 500)."""
    sid = uuid4()
    with test_client.app.state.phase3_engine.begin() as conn:
        _make_session(conn, session_id=str(sid), user_id="alice")
    bogus_event_id = uuid4()
    response = await _post(
        test_client,
        f"/api/sessions/{sid}/interpretations/{bogus_event_id}/resolve",
        json={"choice": "accepted_as_drafted"},
    )
    assert response.status_code == 404, response.text


@pytest.mark.asyncio
async def test_08_resolve_event_from_different_session_returns_404(
    test_client: TestClient,
) -> None:
    """Spec test 8: event_id from a DIFFERENT session (alice owns both) → 404."""
    seeded_a = await _seed_session_with_pending_event(test_client)
    seeded_b = await _seed_session_with_pending_event(test_client)

    # Try to resolve session-B's event_id via session-A's path.  Both sessions
    # belong to alice, so ownership succeeds for session-A's path — but the
    # service's WHERE filter on session_id rejects the cross-session event.
    response = await _post(
        test_client,
        f"/api/sessions/{seeded_a['session_id']}/interpretations/{seeded_b['event'].id}/resolve",
        json={"choice": "accepted_as_drafted"},
    )
    assert response.status_code == 404, response.text


@pytest.mark.asyncio
async def test_08b_resolve_existing_event_with_consumed_placeholder_returns_422(
    test_client: TestClient,
) -> None:
    """A real patch failure must not be laundered as "event not found"."""
    seeded = await _seed_session_with_pending_event(
        test_client,
        node=_llm_node(user_term="cool") | {"options": {"prompt_template": "No interpretation placeholder remains."}},
    )

    response = await _post(
        test_client,
        f"/api/sessions/{seeded['session_id']}/interpretations/{seeded['event'].id}/resolve",
        json={"choice": "accepted_as_drafted"},
    )

    assert response.status_code == 422, response.text
    assert response.json()["detail"]["code"] == "interpretation_placeholder_unavailable"


@pytest.mark.asyncio
async def test_resolve_prompt_template_amended_returns_422(test_client: TestClient) -> None:
    seeded = await _seed_session_with_pending_event(
        test_client,
        node=_prompt_template_review_node(),
        user_term="llm_prompt_template:identify_colour",
        kind=InterpretationKind.LLM_PROMPT_TEMPLATE,
        llm_draft="Read {{ row.html }} and return JSON.",
    )

    response = await _post(
        test_client,
        f"/api/sessions/{seeded['session_id']}/interpretations/{seeded['event'].id}/resolve",
        json={"choice": "amended", "amended_value": "Read row HTML and return CSV."},
    )

    assert response.status_code == 422, response.text
    assert response.json()["detail"]["code"] == "interpretation_resolution_unsupported"


@pytest.mark.asyncio
async def test_resolve_invented_source_amended_returns_422(test_client: TestClient) -> None:
    seeded = await _seed_session_with_source_pending_event(test_client)

    response = await _post(
        test_client,
        f"/api/sessions/{seeded['session_id']}/interpretations/{seeded['event'].id}/resolve",
        json={"choice": "amended", "amended_value": "https://dta.gov.au"},
    )

    assert response.status_code == 422, response.text
    assert response.json()["detail"]["code"] == "interpretation_resolution_unsupported"


# --------------------------------------------------------------------------- #
# GET /interpretations status filter
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_09_list_status_pending_returns_only_pending_events(
    test_client: TestClient,
) -> None:
    """Spec test 9: status=pending ⇒ only choice='pending' rows."""
    seeded = await _seed_session_with_pending_event(test_client)
    session_id = seeded["session_id"]

    # Add a second pending event then resolve the first.
    service: SessionServiceImpl = test_client.app.state.session_service
    second = await service.create_pending_interpretation_event(
        session_id=session_id,
        composition_state_id=seeded["state"].id,
        affected_node_id="llm_transform_1",
        tool_call_id="call_43",
        user_term="cool",
        kind=InterpretationKind.VAGUE_TERM,
        llm_draft="Second draft",
        model_identifier="anthropic/claude-opus-4-7",
        model_version="2026-05-01",
        provider="anthropic",
        composer_skill_hash="a" * 64,
    )
    await _post(
        test_client,
        f"/api/sessions/{session_id}/interpretations/{seeded['event'].id}/resolve",
        json={"choice": "accepted_as_drafted"},
    )

    response = await _get(test_client, f"/api/sessions/{session_id}/interpretations?status=pending")
    assert response.status_code == 200, response.text
    events = response.json()["events"]
    assert len(events) == 1
    assert events[0]["id"] == str(second.id)
    assert events[0]["choice"] == "pending"
    assert events[0]["kind"] == "vague_term"


@pytest.mark.asyncio
async def test_10_list_status_all_returns_every_event(test_client: TestClient) -> None:
    """Spec test 10: status=all ⇒ every row including resolved."""
    seeded = await _seed_session_with_pending_event(test_client)
    session_id = seeded["session_id"]
    await _post(
        test_client,
        f"/api/sessions/{session_id}/interpretations/{seeded['event'].id}/resolve",
        json={"choice": "accepted_as_drafted"},
    )

    response = await _get(test_client, f"/api/sessions/{session_id}/interpretations?status=all")
    assert response.status_code == 200, response.text
    events = response.json()["events"]
    assert len(events) == 1
    assert events[0]["choice"] == "accepted_as_drafted"


# --------------------------------------------------------------------------- #
# IDOR regression
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_11_idor_post_resolve_on_other_users_session_returns_404(
    test_client: TestClient,
) -> None:
    """Spec test 11: alice cannot resolve bob's session-A event → 404, not 403."""
    seeded_bob = await _seed_session_with_pending_event(test_client, user_id="bob")
    response = await _post(
        test_client,
        f"/api/sessions/{seeded_bob['session_id']}/interpretations/{seeded_bob['event'].id}/resolve",
        json={"choice": "accepted_as_drafted"},
    )
    assert response.status_code == 404, response.text


@pytest.mark.asyncio
async def test_12_idor_get_list_on_other_users_session_returns_404(
    test_client: TestClient,
) -> None:
    """Spec test 12: alice cannot list bob's session interpretations → 404."""
    seeded_bob = await _seed_session_with_pending_event(test_client, user_id="bob")
    response = await _get(
        test_client,
        f"/api/sessions/{seeded_bob['session_id']}/interpretations?status=all",
    )
    assert response.status_code == 404, response.text


@pytest.mark.asyncio
async def test_13_idor_f7_cross_session_event_id_returns_404(
    test_client: TestClient,
) -> None:
    """Spec test 13: alice's session-A path + bob's session-B event_id → 404.

    Verifies that ``resolve_interpretation_event`` filters on BOTH
    ``id = :event_id AND session_id = :session_id``, not on event_id alone.
    Even though alice owns session-A, she cannot reach bob's session-B
    event by smuggling its event_id through session-A's path.
    """
    seeded_alice = await _seed_session_with_pending_event(test_client, user_id="alice")
    seeded_bob = await _seed_session_with_pending_event(test_client, user_id="bob")

    response = await _post(
        test_client,
        f"/api/sessions/{seeded_alice['session_id']}/interpretations/{seeded_bob['event'].id}/resolve",
        json={"choice": "accepted_as_drafted"},
    )
    assert response.status_code == 404, response.text

    # Bob's event must remain pending (no cross-session mutation).
    service: SessionServiceImpl = test_client.app.state.session_service
    bob_events = await service.list_interpretation_events(seeded_bob["session_id"], status="all")
    assert len(bob_events) == 1
    assert bob_events[0].choice is InterpretationChoice.PENDING


# --------------------------------------------------------------------------- #
# Opt-out summary route (F-22)
# --------------------------------------------------------------------------- #


def _insert_auto_event(
    test_client: TestClient,
    *,
    session_id: UUID,
    source: InterpretationSource,
    created_at: datetime | None = None,
) -> None:
    """Direct-insert an auto-interpreted row matching schema invariants.

    Mirrors the row shapes that ``record_session_interpretation_opt_out`` and
    ``record_auto_interpreted_no_surfaces_event`` produce, so the inserts
    satisfy ``ck_interpretation_events_opt_out_shape``,
    ``ck_interpretation_events_no_surfaces_shape``, and
    ``ck_interpretation_events_resolved_at_status``.

    Bypasses the service-level writers (which require a real compose-loop
    actor and active session lock semantics) and inserts the canonical row
    shape directly.  The route under test reads via the standard
    ``list_interpretation_events`` path and projects whatever the schema
    permitted in.
    """
    from sqlalchemy import insert

    from elspeth.web.sessions.models import interpretation_events_table

    now = created_at or datetime.now(UTC)
    if source is InterpretationSource.AUTO_INTERPRETED_OPT_OUT:
        # All nine surface/provenance fields NULL per
        # ck_interpretation_events_opt_out_shape; choice=opted_out;
        # resolved_at == created_at (row born resolved).
        provenance: dict[str, Any] = {
            "kind": None,
            "model_identifier": None,
            "model_version": None,
            "provider": None,
            "composer_skill_hash": None,
        }
    elif source is InterpretationSource.AUTO_INTERPRETED_NO_SURFACES:
        # Surface fields NULL, provenance fields NOT NULL per
        # ck_interpretation_events_no_surfaces_shape; choice=opted_out
        # (rate cap is the resolution); resolved_at == created_at.
        provenance = {
            "kind": InterpretationKind.VAGUE_TERM.value,
            "model_identifier": "anthropic/claude-opus-4-7",
            "model_version": "2026-05-01",
            "provider": "anthropic",
            "composer_skill_hash": "0" * 64,
        }
    else:
        raise AssertionError(f"_insert_auto_event only supports auto_* sources; got {source!r}")

    with test_client.app.state.phase3_engine.begin() as conn:
        conn.execute(
            insert(interpretation_events_table).values(
                id=str(uuid4()),
                session_id=str(session_id),
                composition_state_id=None,
                affected_node_id=None,
                tool_call_id=None,
                user_term=None,
                llm_draft=None,
                accepted_value=None,
                choice=InterpretationChoice.OPTED_OUT.value,
                created_at=now,
                resolved_at=now,
                actor="system:test",
                arguments_hash=None,
                hash_domain_version=None,
                interpretation_source=source.value,
                runtime_model_identifier_at_resolve=None,
                runtime_model_version_at_resolve=None,
                resolved_prompt_template_hash=None,
                **provenance,
            )
        )


@pytest.mark.asyncio
async def test_18_opt_out_summary_returns_both_auto_sources_ordered_by_created_at(
    test_client: TestClient,
) -> None:
    """F-22 spec test 18: both auto_interpreted_* sources, oldest-first."""
    session_id = uuid4()
    with test_client.app.state.phase3_engine.begin() as conn:
        _make_session(conn, session_id=str(session_id), user_id="alice")

    older = datetime(2026, 5, 17, 10, 0, 0, tzinfo=UTC)
    newer = datetime(2026, 5, 17, 12, 0, 0, tzinfo=UTC)
    _insert_auto_event(
        test_client,
        session_id=session_id,
        source=InterpretationSource.AUTO_INTERPRETED_OPT_OUT,
        created_at=older,
    )
    _insert_auto_event(
        test_client,
        session_id=session_id,
        source=InterpretationSource.AUTO_INTERPRETED_NO_SURFACES,
        created_at=newer,
    )

    response = await _get(test_client, f"/api/sessions/{session_id}/interpretations/opt_out_summary")
    assert response.status_code == 200, response.text
    events = response.json()["events"]
    assert len(events) == 2
    assert events[0]["interpretation_source"] == "auto_interpreted_opt_out"
    assert events[1]["interpretation_source"] == "auto_interpreted_no_surfaces"


@pytest.mark.asyncio
async def test_19_opt_out_summary_excludes_user_approved_rows(test_client: TestClient) -> None:
    """F-22 spec test 19: user_approved rows are not in the summary."""
    seeded = await _seed_session_with_pending_event(test_client)
    session_id = seeded["session_id"]

    # Resolve the seeded event (now choice=accepted_as_drafted, source=user_approved).
    await _post(
        test_client,
        f"/api/sessions/{session_id}/interpretations/{seeded['event'].id}/resolve",
        json={"choice": "accepted_as_drafted"},
    )
    # Also add an auto_interpreted_no_surfaces row.
    _insert_auto_event(
        test_client,
        session_id=session_id,
        source=InterpretationSource.AUTO_INTERPRETED_NO_SURFACES,
    )

    response = await _get(test_client, f"/api/sessions/{session_id}/interpretations/opt_out_summary")
    assert response.status_code == 200, response.text
    events = response.json()["events"]
    assert len(events) == 1
    assert events[0]["interpretation_source"] == "auto_interpreted_no_surfaces"


@pytest.mark.asyncio
async def test_20_idor_opt_out_summary_on_other_users_session_returns_404(
    test_client: TestClient,
) -> None:
    """F-22 spec test 20: alice cannot read bob's opt-out summary → 404."""
    session_id = uuid4()
    with test_client.app.state.phase3_engine.begin() as conn:
        _make_session(conn, session_id=str(session_id), user_id="bob")
    _insert_auto_event(
        test_client,
        session_id=session_id,
        source=InterpretationSource.AUTO_INTERPRETED_OPT_OUT,
    )

    response = await _get(test_client, f"/api/sessions/{session_id}/interpretations/opt_out_summary")
    assert response.status_code == 404, response.text
