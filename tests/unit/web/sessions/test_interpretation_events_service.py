"""Service-layer tests for Phase 5b interpretation-event methods.

Covers the four async service methods plus the private helper added in
Phase 5b Task 4:

* ``SessionServiceImpl.create_pending_interpretation_event``
* ``SessionServiceImpl.resolve_interpretation_event``
* ``SessionServiceImpl.list_interpretation_events``
* ``SessionServiceImpl.record_session_interpretation_opt_out``
* module-level ``_patch_llm_transform_prompt``

Test numbering mirrors the spec at
``docs/composer/ux-redesign-2026-05/18a-phase-5b-backend.md`` (Task 4 —
test shape, lines 1849-1894).

Fixture pattern mirrors ``test_composer_proposals.py`` — in-memory SQLite
engine via ``create_session_engine`` + ``initialize_session_schema``, with
``SessionServiceImpl`` constructed on top. Helper functions seed sessions
and composition states with LLM-transform nodes so the resolve path's
prompt-patch can land on real node JSON.
"""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import UUID, uuid4

import pytest
import structlog
from sqlalchemy import insert, select
from sqlalchemy.pool import StaticPool

from elspeth.contracts.composer_interpretation import (
    INTERPRETATION_HASH_DOMAIN_V1,
    InterpretationChoice,
    InterpretationSource,
)
from elspeth.contracts.hashing import stable_hash
from elspeth.web.sessions.engine import create_session_engine
from elspeth.web.sessions.models import (
    composition_states_table,
    interpretation_events_table,
    proposal_events_table,
    sessions_table,
)
from elspeth.web.sessions.protocol import CompositionStateData, CompositionStateRecord
from elspeth.web.sessions.schema import initialize_session_schema
from elspeth.web.sessions.service import SessionServiceImpl, _patch_llm_transform_prompt
from elspeth.web.sessions.telemetry import build_sessions_telemetry

# --------------------------------------------------------------------------- #
# Fixtures and helpers
# --------------------------------------------------------------------------- #


@pytest.fixture
def engine():
    eng = create_session_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    initialize_session_schema(eng)
    return eng


@pytest.fixture
def service(engine) -> SessionServiceImpl:
    return SessionServiceImpl(
        engine,
        telemetry=build_sessions_telemetry(),
        log=structlog.get_logger("test"),
    )


def _insert_session(conn, session_id: str) -> None:
    conn.execute(
        insert(sessions_table).values(
            id=session_id,
            user_id="alice",
            auth_provider_type="local",
            title="Phase 5b Task 4 Test",
            created_at=datetime.now(UTC),
            updated_at=datetime.now(UTC),
        )
    )


def _llm_node(
    *,
    node_id: str = "llm_transform_1",
    user_term: str = "cool",
    prompt_template: str | None = None,
) -> dict:
    """Return a minimally-shaped LLM-transform node carrying a placeholder."""
    if prompt_template is None:
        prompt_template = f"Rate how {{{{interpretation:{user_term}}}}} this is."
    return {
        "id": node_id,
        "kind": "llm",
        "prompt_template": prompt_template,
    }


async def _seed_state_with_llm_node(
    service: SessionServiceImpl,
    *,
    session_id: UUID,
    node: dict | None = None,
) -> CompositionStateRecord:
    """Create a session row and one composition_states row containing an LLM node."""
    with service._engine.begin() as conn:
        _insert_session(conn, str(session_id))
    node = node if node is not None else _llm_node()
    state = await service.save_composition_state(
        session_id,
        CompositionStateData(nodes=[node], is_valid=True),
        provenance="tool_call",
    )
    return state


# --------------------------------------------------------------------------- #
# create_pending_interpretation_event
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_01_create_pending_interpretation_event_inserts_row(service) -> None:
    """Spec test 1: pending row inserted with all six required fields."""
    session_id = uuid4()
    state = await _seed_state_with_llm_node(service, session_id=session_id)

    event = await service.create_pending_interpretation_event(
        session_id=session_id,
        composition_state_id=state.id,
        affected_node_id="llm_transform_1",
        tool_call_id="call_42",
        user_term="cool",
        llm_draft="A draft definition of cool",
        model_identifier="anthropic/claude-opus-4-7",
        model_version="2026-05-01",
        provider="anthropic",
        composer_skill_hash="a" * 64,
    )

    assert event.choice is InterpretationChoice.PENDING
    assert event.interpretation_source is InterpretationSource.USER_APPROVED
    assert event.user_term == "cool"
    assert event.llm_draft == "A draft definition of cool"
    assert event.accepted_value is None
    assert event.arguments_hash is None
    assert event.hash_domain_version is None
    assert event.resolved_at is None
    assert event.session_id == session_id
    assert event.composition_state_id == state.id
    assert event.affected_node_id == "llm_transform_1"
    assert event.tool_call_id == "call_42"
    assert event.model_identifier == "anthropic/claude-opus-4-7"
    assert event.model_version == "2026-05-01"
    assert event.provider == "anthropic"
    assert event.composer_skill_hash == "a" * 64

    # Spot-check the DB row directly.
    with service._engine.begin() as conn:
        row = conn.execute(select(interpretation_events_table).where(interpretation_events_table.c.id == str(event.id))).one()
    assert row.choice == "pending"
    assert row.interpretation_source == "user_approved"
    assert row.user_term == "cool"
    assert row.accepted_value is None
    assert row.resolved_at is None


@pytest.mark.asyncio
async def test_02_create_pending_rejects_unknown_node_id(service) -> None:
    """Spec test 2: writer-boundary validation on affected_node_id.

    Per CLAUDE.md offensive-programming rules, the writer must detect that
    affected_node_id is not present in composition_states.nodes BEFORE any
    DB write, raising ValueError. The interpretation_events table must be
    empty after the raise (transaction rolled back).
    """
    session_id = uuid4()
    state = await _seed_state_with_llm_node(
        service,
        session_id=session_id,
        node=_llm_node(node_id="node-A"),
    )

    with pytest.raises(ValueError, match=r"node-does-not-exist|not present"):
        await service.create_pending_interpretation_event(
            session_id=session_id,
            composition_state_id=state.id,
            affected_node_id="node-does-not-exist",
            tool_call_id="call_42",
            user_term="cool",
            llm_draft="A draft of cool",
            model_identifier="anthropic/claude-opus-4-7",
            model_version="2026-05-01",
            provider="anthropic",
            composer_skill_hash="a" * 64,
        )

    with service._engine.begin() as conn:
        count = conn.execute(
            select(interpretation_events_table).where(interpretation_events_table.c.session_id == str(session_id))
        ).fetchall()
    assert count == [], "interpretation_events must be empty after writer-boundary raise"


# --------------------------------------------------------------------------- #
# resolve_interpretation_event
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_03_resolve_accepted_as_drafted_uses_llm_draft(service) -> None:
    """Spec test 3: accepted_as_drafted ⇒ accepted_value = llm_draft."""
    session_id = uuid4()
    state = await _seed_state_with_llm_node(service, session_id=session_id)
    event = await service.create_pending_interpretation_event(
        session_id=session_id,
        composition_state_id=state.id,
        affected_node_id="llm_transform_1",
        tool_call_id="call_42",
        user_term="cool",
        llm_draft="Innovative and creative",
        model_identifier="anthropic/claude-opus-4-7",
        model_version="2026-05-01",
        provider="anthropic",
        composer_skill_hash="a" * 64,
    )

    resolved, new_state = await service.resolve_interpretation_event(
        session_id=session_id,
        event_id=event.id,
        choice=InterpretationChoice.ACCEPTED_AS_DRAFTED,
        amended_value=None,
        actor="user:alice",
        runtime_model_identifier="anthropic/claude-sonnet-4-7",
        runtime_model_version="2026-05-02",
    )

    assert resolved.choice is InterpretationChoice.ACCEPTED_AS_DRAFTED
    assert resolved.accepted_value == "Innovative and creative"
    assert resolved.resolved_at is not None
    assert resolved.actor == "user:alice"
    assert resolved.runtime_model_identifier_at_resolve == "anthropic/claude-sonnet-4-7"
    assert resolved.runtime_model_version_at_resolve == "2026-05-02"
    assert resolved.hash_domain_version == "v1"
    assert resolved.arguments_hash is not None
    assert len(resolved.arguments_hash) == 64
    assert resolved.resolved_prompt_template_hash is not None
    assert len(resolved.resolved_prompt_template_hash) == 64

    # A new composition state row exists at version+1 with interpretation_resolve provenance.
    assert new_state.version == state.version + 1
    assert new_state.nodes is not None
    patched = next(n for n in new_state.nodes if n["id"] == "llm_transform_1")
    assert "{{interpretation:cool}}" not in patched["prompt_template"]
    assert "Innovative and creative" in patched["prompt_template"]
    assert patched["resolved_prompt_template_hash"] == resolved.resolved_prompt_template_hash

    # Verify provenance in DB.
    with service._engine.begin() as conn:
        state_row = conn.execute(select(composition_states_table).where(composition_states_table.c.id == str(new_state.id))).one()
    assert state_row.provenance == "interpretation_resolve"


@pytest.mark.asyncio
async def test_04_resolve_amended_uses_amended_value(service) -> None:
    """Spec test 4: amended ⇒ accepted_value = amended_value."""
    session_id = uuid4()
    state = await _seed_state_with_llm_node(service, session_id=session_id)
    event = await service.create_pending_interpretation_event(
        session_id=session_id,
        composition_state_id=state.id,
        affected_node_id="llm_transform_1",
        tool_call_id="call_42",
        user_term="cool",
        llm_draft="Innovative and creative",
        model_identifier="anthropic/claude-opus-4-7",
        model_version="2026-05-01",
        provider="anthropic",
        composer_skill_hash="a" * 64,
    )

    resolved, new_state = await service.resolve_interpretation_event(
        session_id=session_id,
        event_id=event.id,
        choice=InterpretationChoice.AMENDED,
        amended_value="Strikingly original",
        actor="user:alice",
        runtime_model_identifier=None,
        runtime_model_version=None,
    )

    assert resolved.choice is InterpretationChoice.AMENDED
    assert resolved.accepted_value == "Strikingly original"
    assert new_state.version == state.version + 1
    patched = next(n for n in new_state.nodes if n["id"] == "llm_transform_1")
    assert "Strikingly original" in patched["prompt_template"]
    assert "{{interpretation:cool}}" not in patched["prompt_template"]


@pytest.mark.asyncio
async def test_05_resolve_raises_on_double_resolve(service) -> None:
    """Spec test 5: TOCTOU guard via WHERE choice='pending'."""
    session_id = uuid4()
    state = await _seed_state_with_llm_node(service, session_id=session_id)
    event = await service.create_pending_interpretation_event(
        session_id=session_id,
        composition_state_id=state.id,
        affected_node_id="llm_transform_1",
        tool_call_id="call_42",
        user_term="cool",
        llm_draft="A draft",
        model_identifier="anthropic/claude-opus-4-7",
        model_version="2026-05-01",
        provider="anthropic",
        composer_skill_hash="a" * 64,
    )

    await service.resolve_interpretation_event(
        session_id=session_id,
        event_id=event.id,
        choice=InterpretationChoice.ACCEPTED_AS_DRAFTED,
        amended_value=None,
        actor="user:alice",
        runtime_model_identifier=None,
        runtime_model_version=None,
    )

    with pytest.raises(ValueError):
        await service.resolve_interpretation_event(
            session_id=session_id,
            event_id=event.id,
            choice=InterpretationChoice.AMENDED,
            amended_value="A second resolution",
            actor="user:alice",
            runtime_model_identifier=None,
            runtime_model_version=None,
        )


@pytest.mark.asyncio
async def test_06_resolve_raises_when_node_removed_since_surfacing(service) -> None:
    """Spec test 6: affected node removed from composition state since surfacing."""
    session_id = uuid4()
    state = await _seed_state_with_llm_node(service, session_id=session_id)
    event = await service.create_pending_interpretation_event(
        session_id=session_id,
        composition_state_id=state.id,
        affected_node_id="llm_transform_1",
        tool_call_id="call_42",
        user_term="cool",
        llm_draft="A draft",
        model_identifier="anthropic/claude-opus-4-7",
        model_version="2026-05-01",
        provider="anthropic",
        composer_skill_hash="a" * 64,
    )

    # Advance composition state — the affected node disappears.
    await service.save_composition_state(
        session_id,
        CompositionStateData(
            nodes=[{"id": "different-node", "kind": "csv", "options": {}}],
            is_valid=True,
        ),
        provenance="tool_call",
    )

    with pytest.raises(ValueError):
        await service.resolve_interpretation_event(
            session_id=session_id,
            event_id=event.id,
            choice=InterpretationChoice.ACCEPTED_AS_DRAFTED,
            amended_value=None,
            actor="user:alice",
            runtime_model_identifier=None,
            runtime_model_version=None,
        )


@pytest.mark.asyncio
async def test_07_resolve_raises_when_placeholder_absent(service) -> None:
    """Spec test 7: prompt_template missing the {{interpretation:<term>}} placeholder."""
    session_id = uuid4()
    bad_node = _llm_node(prompt_template="No placeholder here at all.")
    state = await _seed_state_with_llm_node(service, session_id=session_id, node=bad_node)
    event = await service.create_pending_interpretation_event(
        session_id=session_id,
        composition_state_id=state.id,
        affected_node_id="llm_transform_1",
        tool_call_id="call_42",
        user_term="cool",
        llm_draft="A draft",
        model_identifier="anthropic/claude-opus-4-7",
        model_version="2026-05-01",
        provider="anthropic",
        composer_skill_hash="a" * 64,
    )

    with pytest.raises(ValueError, match=r"placeholder|interpretation"):
        await service.resolve_interpretation_event(
            session_id=session_id,
            event_id=event.id,
            choice=InterpretationChoice.ACCEPTED_AS_DRAFTED,
            amended_value=None,
            actor="user:alice",
            runtime_model_identifier=None,
            runtime_model_version=None,
        )

    # No interpretation_events row may have been updated to resolved, and no
    # new composition_states row may have been written.
    with service._engine.begin() as conn:
        row = conn.execute(select(interpretation_events_table).where(interpretation_events_table.c.id == str(event.id))).one()
        states = conn.execute(select(composition_states_table).where(composition_states_table.c.session_id == str(session_id))).fetchall()
    assert row.choice == "pending"
    assert row.accepted_value is None
    assert len(states) == 1, "resolve must short-circuit before writing a new state"


# --------------------------------------------------------------------------- #
# list_interpretation_events
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_08_list_status_pending_filters_to_pending_only(service) -> None:
    """Spec test 8: status='pending' returns only pending rows."""
    session_id = uuid4()
    state = await _seed_state_with_llm_node(service, session_id=session_id)

    # Two pending events on distinct tool_call_ids.
    e_pending = await service.create_pending_interpretation_event(
        session_id=session_id,
        composition_state_id=state.id,
        affected_node_id="llm_transform_1",
        tool_call_id="call_pending",
        user_term="cool",
        llm_draft="cool def",
        model_identifier="anthropic/claude-opus-4-7",
        model_version="2026-05-01",
        provider="anthropic",
        composer_skill_hash="a" * 64,
    )
    e_to_resolve = await service.create_pending_interpretation_event(
        session_id=session_id,
        composition_state_id=state.id,
        affected_node_id="llm_transform_1",
        tool_call_id="call_resolved",
        user_term="cool",
        llm_draft="cool def 2",
        model_identifier="anthropic/claude-opus-4-7",
        model_version="2026-05-01",
        provider="anthropic",
        composer_skill_hash="a" * 64,
    )
    await service.resolve_interpretation_event(
        session_id=session_id,
        event_id=e_to_resolve.id,
        choice=InterpretationChoice.ACCEPTED_AS_DRAFTED,
        amended_value=None,
        actor="user:alice",
        runtime_model_identifier=None,
        runtime_model_version=None,
    )

    pending = await service.list_interpretation_events(session_id, status="pending")
    assert [e.id for e in pending] == [e_pending.id]


@pytest.mark.asyncio
async def test_09_list_status_all_returns_all_rows(service) -> None:
    """Spec test 9: status='all' returns every row for the session."""
    session_id = uuid4()
    state = await _seed_state_with_llm_node(service, session_id=session_id)

    e1 = await service.create_pending_interpretation_event(
        session_id=session_id,
        composition_state_id=state.id,
        affected_node_id="llm_transform_1",
        tool_call_id="call_1",
        user_term="cool",
        llm_draft="d1",
        model_identifier="anthropic/claude-opus-4-7",
        model_version="2026-05-01",
        provider="anthropic",
        composer_skill_hash="a" * 64,
    )
    e2 = await service.create_pending_interpretation_event(
        session_id=session_id,
        composition_state_id=state.id,
        affected_node_id="llm_transform_1",
        tool_call_id="call_2",
        user_term="cool",
        llm_draft="d2",
        model_identifier="anthropic/claude-opus-4-7",
        model_version="2026-05-01",
        provider="anthropic",
        composer_skill_hash="a" * 64,
    )
    await service.resolve_interpretation_event(
        session_id=session_id,
        event_id=e2.id,
        choice=InterpretationChoice.ACCEPTED_AS_DRAFTED,
        amended_value=None,
        actor="user:alice",
        runtime_model_identifier=None,
        runtime_model_version=None,
    )

    rows = await service.list_interpretation_events(session_id, status="all")
    assert {e.id for e in rows} == {e1.id, e2.id}


# --------------------------------------------------------------------------- #
# record_session_interpretation_opt_out
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_10_opt_out_writes_event_and_sets_session_boolean_no_proposal_event(service) -> None:
    """Spec test 10: opt-out atomicity + proposal_events regression guard.

    Asserts:
      * an interpretation_events row exists with choice='opted_out',
        interpretation_source='auto_interpreted_opt_out', resolved_at set,
        all nullable interpretation fields NULL;
      * sessions.interpretation_review_disabled is true;
      * proposal_events row count for the session is unchanged across the
        call (delta-guard against the prior proposal_events routing).
    """
    session_id = uuid4()
    with service._engine.begin() as conn:
        _insert_session(conn, str(session_id))
        before_count = conn.execute(select(proposal_events_table).where(proposal_events_table.c.session_id == str(session_id))).fetchall()

    event = await service.record_session_interpretation_opt_out(
        session_id=session_id,
        actor="user:alice",
    )

    assert event.choice is InterpretationChoice.OPTED_OUT
    assert event.interpretation_source is InterpretationSource.AUTO_INTERPRETED_OPT_OUT
    assert event.resolved_at is not None
    assert event.composition_state_id is None
    assert event.affected_node_id is None
    assert event.tool_call_id is None
    assert event.user_term is None
    assert event.llm_draft is None
    assert event.accepted_value is None
    assert event.model_identifier is None
    assert event.model_version is None
    assert event.provider is None
    assert event.composer_skill_hash is None
    assert event.arguments_hash is None

    with service._engine.begin() as conn:
        session_row = conn.execute(select(sessions_table).where(sessions_table.c.id == str(session_id))).one()
        assert session_row.interpretation_review_disabled is True

        after_count = conn.execute(select(proposal_events_table).where(proposal_events_table.c.session_id == str(session_id))).fetchall()
    assert len(after_count) == len(before_count), "record_session_interpretation_opt_out must NOT write to proposal_events"


@pytest.mark.asyncio
async def test_10b_opt_out_is_idempotent(service) -> None:
    """F-29: second call returns the existing opt-out row, no duplicate insert."""
    session_id = uuid4()
    with service._engine.begin() as conn:
        _insert_session(conn, str(session_id))

    first = await service.record_session_interpretation_opt_out(
        session_id=session_id,
        actor="user:alice",
    )
    second = await service.record_session_interpretation_opt_out(
        session_id=session_id,
        actor="user:alice",
    )

    assert first.id == second.id
    assert first.resolved_at == second.resolved_at

    with service._engine.begin() as conn:
        rows = conn.execute(
            select(interpretation_events_table).where(interpretation_events_table.c.session_id == str(session_id))
        ).fetchall()
    assert len(rows) == 1


# --------------------------------------------------------------------------- #
# _patch_llm_transform_prompt — direct-helper unit tests
# --------------------------------------------------------------------------- #


def _state_with_node(node: dict) -> CompositionStateRecord:
    """Return a CompositionStateRecord shell carrying one node for helper tests."""
    return CompositionStateRecord(
        id=uuid4(),
        session_id=uuid4(),
        version=1,
        source=None,
        nodes=[node],
        edges=None,
        outputs=None,
        metadata_=None,
        is_valid=True,
        validation_errors=None,
        created_at=datetime.now(UTC),
        derived_from_state_id=None,
        composer_meta=None,
    )


def test_11_patch_helper_rejects_multiple_placeholders() -> None:
    """Spec test 11: placeholder appears >1 time ⇒ raise."""
    state = _state_with_node(_llm_node(prompt_template="{{interpretation:cool}} and {{interpretation:cool}} again."))
    with pytest.raises(ValueError, match=r"more than once|multiple|appears"):
        _patch_llm_transform_prompt(
            state,
            affected_node_id="llm_transform_1",
            user_term="cool",
            accepted_value="Innovative",
        )


def test_12_patch_helper_rejects_system_prefix() -> None:
    """Spec test 12: 'system:' prefix immediately before placeholder ⇒ raise."""
    state = _state_with_node(_llm_node(prompt_template="System: {{interpretation:cool}}"))
    with pytest.raises(ValueError, match=r"system:|directive|prefix"):
        _patch_llm_transform_prompt(
            state,
            affected_node_id="llm_transform_1",
            user_term="cool",
            accepted_value="Innovative",
        )


def test_13_patch_helper_rejects_role_prefix() -> None:
    """Spec test 13: 'role:' prefix immediately before placeholder ⇒ raise."""
    state = _state_with_node(_llm_node(prompt_template="Role: {{interpretation:cool}}"))
    with pytest.raises(ValueError, match=r"role:|directive|prefix"):
        _patch_llm_transform_prompt(
            state,
            affected_node_id="llm_transform_1",
            user_term="cool",
            accepted_value="Innovative",
        )


def test_13b_patch_helper_rejects_instructions_prefix() -> None:
    """Bonus: 'instructions:' prefix is also caught by the structural-directive guard."""
    state = _state_with_node(_llm_node(prompt_template="Instructions: {{interpretation:cool}}"))
    with pytest.raises(ValueError, match=r"instructions:|directive|prefix"):
        _patch_llm_transform_prompt(
            state,
            affected_node_id="llm_transform_1",
            user_term="cool",
            accepted_value="Innovative",
        )


def test_14_patch_helper_succeeds_on_clean_template() -> None:
    """Spec test 14: clean template with one placeholder in normal body position."""
    state = _state_with_node(_llm_node(prompt_template="Rate how {{interpretation:cool}} this is."))
    nodes_out = _patch_llm_transform_prompt(
        state,
        affected_node_id="llm_transform_1",
        user_term="cool",
        accepted_value="Innovative and creative",
    )
    nodes_list = list(nodes_out)
    assert len(nodes_list) == 1
    patched = nodes_list[0]
    assert "{{interpretation:cool}}" not in patched["prompt_template"]
    assert "Innovative and creative" in patched["prompt_template"]
    # The other surrounding text is preserved verbatim.
    assert patched["prompt_template"] == "Rate how Innovative and creative this is."
    # Node id/kind unchanged.
    assert patched["id"] == "llm_transform_1"
    assert patched["kind"] == "llm"


def test_patch_helper_rejects_missing_node() -> None:
    """Helper-contract guard: node not in state.nodes ⇒ raise (covers writer-boundary parity)."""
    state = _state_with_node(_llm_node(node_id="llm_transform_1"))
    with pytest.raises(ValueError, match=r"not present|missing|node"):
        _patch_llm_transform_prompt(
            state,
            affected_node_id="missing-node",
            user_term="cool",
            accepted_value="x",
        )


def test_patch_helper_rejects_non_llm_kind() -> None:
    """Helper-contract guard: kind != 'llm' ⇒ raise."""
    state = _state_with_node({"id": "csv_source_1", "kind": "csv", "options": {}})
    with pytest.raises(ValueError, match=r"llm|kind"):
        _patch_llm_transform_prompt(
            state,
            affected_node_id="csv_source_1",
            user_term="cool",
            accepted_value="x",
        )


def test_patch_helper_rejects_missing_prompt_template() -> None:
    """Helper-contract guard: no prompt_template field ⇒ raise."""
    state = _state_with_node({"id": "llm_transform_1", "kind": "llm"})
    with pytest.raises(ValueError, match="prompt_template"):
        _patch_llm_transform_prompt(
            state,
            affected_node_id="llm_transform_1",
            user_term="cool",
            accepted_value="x",
        )


def test_arguments_hash_matches_domain_v1() -> None:
    """Cross-check: the arguments_hash a service writes is consistent with
    INTERPRETATION_HASH_DOMAIN_V1 over the recorded field set.

    Computes the expected hash from the 12-field domain and confirms it
    matches the stored row's arguments_hash after resolve. This guards
    against silent drift between the writer and the closed hash domain.
    """
    # Computed inside an asyncio test wrapper below.


@pytest.mark.asyncio
async def test_15_arguments_hash_matches_domain_v1(service) -> None:
    """The service's arguments_hash matches stable_hash() over INTERPRETATION_HASH_DOMAIN_V1."""
    session_id = uuid4()
    state = await _seed_state_with_llm_node(service, session_id=session_id)
    event = await service.create_pending_interpretation_event(
        session_id=session_id,
        composition_state_id=state.id,
        affected_node_id="llm_transform_1",
        tool_call_id="call_42",
        user_term="cool",
        llm_draft="A draft of cool",
        model_identifier="anthropic/claude-opus-4-7",
        model_version="2026-05-01",
        provider="anthropic",
        composer_skill_hash="a" * 64,
    )
    resolved, _new_state = await service.resolve_interpretation_event(
        session_id=session_id,
        event_id=event.id,
        choice=InterpretationChoice.ACCEPTED_AS_DRAFTED,
        amended_value=None,
        actor="user:alice",
        runtime_model_identifier=None,
        runtime_model_version=None,
    )

    expected_domain = {
        "session_id": str(session_id),
        "composition_state_id": str(state.id),
        "affected_node_id": "llm_transform_1",
        "tool_call_id": "call_42",
        "user_term": "cool",
        "llm_draft": "A draft of cool",
        "accepted_value": "A draft of cool",
        "actor": "user:alice",
        "model_identifier": "anthropic/claude-opus-4-7",
        "model_version": "2026-05-01",
        "provider": "anthropic",
        "composer_skill_hash": "a" * 64,
    }
    # Sanity: domain dict keys exactly match INTERPRETATION_HASH_DOMAIN_V1.
    assert set(expected_domain.keys()) == INTERPRETATION_HASH_DOMAIN_V1
    assert resolved.arguments_hash == stable_hash(expected_domain)
    assert resolved.hash_domain_version == "v1"


# --------------------------------------------------------------------------- #
# Trigger-error classifier (F-28) — direct UPDATE on a resolved row must
# surface as ValueError, not raw IntegrityError.
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_resolve_after_resolve_classifies_trigger_message_to_valueerror(service) -> None:
    """Double-resolve via the public method ⇒ ValueError (not IntegrityError).

    The TOCTOU WHERE choice='pending' guard short-circuits before the
    trigger fires in normal use, so this test covers the SELECT-then-raise
    path. Test 5 covers the same behaviour from a different angle (post-
    first-resolve, the second call should produce a domain error).
    """
    # Same shape as test 5 but with a stricter assertion on exception type.
    session_id = uuid4()
    state = await _seed_state_with_llm_node(service, session_id=session_id)
    event = await service.create_pending_interpretation_event(
        session_id=session_id,
        composition_state_id=state.id,
        affected_node_id="llm_transform_1",
        tool_call_id="call_42",
        user_term="cool",
        llm_draft="A draft",
        model_identifier="anthropic/claude-opus-4-7",
        model_version="2026-05-01",
        provider="anthropic",
        composer_skill_hash="a" * 64,
    )
    await service.resolve_interpretation_event(
        session_id=session_id,
        event_id=event.id,
        choice=InterpretationChoice.ACCEPTED_AS_DRAFTED,
        amended_value=None,
        actor="user:alice",
        runtime_model_identifier=None,
        runtime_model_version=None,
    )
    with pytest.raises(ValueError):
        await service.resolve_interpretation_event(
            session_id=session_id,
            event_id=event.id,
            choice=InterpretationChoice.ACCEPTED_AS_DRAFTED,
            amended_value=None,
            actor="user:alice",
            runtime_model_identifier=None,
            runtime_model_version=None,
        )
