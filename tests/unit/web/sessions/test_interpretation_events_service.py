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
from types import SimpleNamespace
from uuid import UUID, uuid4

import pytest
import structlog
from sqlalchemy import insert, select
from sqlalchemy.pool import StaticPool

from elspeth.contracts.composer_interpretation import (
    INTERPRETATION_HASH_DOMAIN_V2,
    InterpretationChoice,
    InterpretationKind,
    InterpretationSource,
)
from elspeth.contracts.hashing import stable_hash
from elspeth.web.composer.state import CompositionState, NodeSpec, PipelineMetadata
from elspeth.web.interpretation_state import INTERPRETATION_REQUIREMENTS_KEY, PROMPT_TEMPLATE_PARTS_KEY
from elspeth.web.sessions.engine import create_session_engine
from elspeth.web.sessions.models import (
    composition_states_table,
    interpretation_events_table,
    proposal_events_table,
    sessions_table,
)
from elspeth.web.sessions.protocol import CompositionStateData, CompositionStateRecord
from elspeth.web.sessions.schema import initialize_session_schema
from elspeth.web.sessions.service import SessionServiceImpl, _interpretation_event_record_from_row, _patch_llm_transform_prompt
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
    """Return a production-serialized LLM transform node carrying a placeholder.

    Shape note: ``prompt_template`` lives inside ``options`` because that is
    the field ``_patch_llm_transform_prompt`` reads (mirroring
    ``NodeSpec.options["prompt_template"]``). The returned dict comes from
    ``CompositionState.to_dict()`` so resolve tests use the same
    ``node_type``/``plugin`` discriminator that production composer state
    emits. Earlier fixture iterations used a private ``kind`` field and let
    production break while unit tests stayed green.
    """
    if prompt_template is None:
        prompt_template = f"Rate how {{{{interpretation:{user_term}}}}} this is."
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
                options={"prompt_template": prompt_template},
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
        metadata=PipelineMetadata(name="Phase 5b Test", description=""),
        version=1,
    )
    return state.to_dict()["nodes"][0]


def _structured_llm_node(
    *,
    node_id: str = "llm_transform_1",
    user_term: str = "cool",
) -> dict:
    """Return a production-serialized LLM transform with structured pending interpretation state."""
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
                options={
                    "prompt_template": "Rate pending interpretation this is.",
                    PROMPT_TEMPLATE_PARTS_KEY: [
                        {"kind": "text", "text": "Rate "},
                        {"kind": "interpretation_ref", "requirement_id": user_term},
                        {"kind": "text", "text": " this is."},
                    ],
                    INTERPRETATION_REQUIREMENTS_KEY: [
                        {
                            "id": user_term,
                            "user_term": user_term,
                            "status": "pending",
                            "draft": "A draft of cool",
                            "event_id": None,
                            "accepted_value": None,
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
        metadata=PipelineMetadata(name="Phase 5b Test", description=""),
        version=1,
    )
    return state.to_dict()["nodes"][0]


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
        CompositionStateData(
            nodes=[node],
            metadata_={"name": "Phase 5b Test", "description": ""},
            is_valid=True,
        ),
        provenance="tool_call",
    )
    return state


# --------------------------------------------------------------------------- #
# create_pending_interpretation_event
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_01_create_pending_interpretation_event_inserts_row(service) -> None:
    """Spec test 1: pending row inserted with all required fields."""
    session_id = uuid4()
    state = await _seed_state_with_llm_node(service, session_id=session_id)

    event = await service.create_pending_interpretation_event(
        session_id=session_id,
        composition_state_id=state.id,
        affected_node_id="llm_transform_1",
        tool_call_id="call_42",
        user_term="cool",
        kind=InterpretationKind.VAGUE_TERM,
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
    assert event.kind is InterpretationKind.VAGUE_TERM
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
    assert row.kind == "vague_term"
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
            kind=InterpretationKind.VAGUE_TERM,
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


@pytest.mark.asyncio
async def test_create_pending_interpretation_event_requires_explicit_kind(service) -> None:
    """The writer must not silently classify omitted kinds as vague_term."""
    session_id = uuid4()
    state = await _seed_state_with_llm_node(service, session_id=session_id)

    with pytest.raises(TypeError, match="kind"):
        await service.create_pending_interpretation_event(
            session_id=session_id,
            composition_state_id=state.id,
            affected_node_id="llm_transform_1",
            tool_call_id="call_missing_kind",
            user_term="cool",
            llm_draft="A draft of cool",
            model_identifier="anthropic/claude-opus-4-7",
            model_version="2026-05-01",
            provider="anthropic",
            composer_skill_hash="a" * 64,
        )

    rows = await service.list_interpretation_events(session_id, status="all")
    assert rows == []


@pytest.mark.asyncio
async def test_create_pending_interpretation_event_rejects_raw_kind_string(service) -> None:
    """Raw strings must not be coerced into the vague-term enum."""
    session_id = uuid4()
    state = await _seed_state_with_llm_node(service, session_id=session_id)

    with pytest.raises(ValueError, match=r"kind must be InterpretationKind"):
        await service.create_pending_interpretation_event(
            session_id=session_id,
            composition_state_id=state.id,
            affected_node_id="llm_transform_1",
            tool_call_id="call_raw_kind",
            user_term="cool",
            kind="invented_source",  # type: ignore[arg-type]
            llm_draft="A draft of cool",
            model_identifier="anthropic/claude-opus-4-7",
            model_version="2026-05-01",
            provider="anthropic",
            composer_skill_hash="a" * 64,
        )

    rows = await service.list_interpretation_events(session_id, status="all")
    assert rows == []


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
        kind=InterpretationKind.VAGUE_TERM,
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
    assert resolved.kind is InterpretationKind.VAGUE_TERM
    assert resolved.hash_domain_version == "v2"
    assert resolved.arguments_hash is not None
    assert len(resolved.arguments_hash) == 64
    assert resolved.resolved_prompt_template_hash is not None
    assert len(resolved.resolved_prompt_template_hash) == 64

    # A new composition state row exists at version+1 with interpretation_resolve provenance.
    assert new_state.version == state.version + 1
    assert new_state.nodes is not None
    patched = next(n for n in new_state.nodes if n["id"] == "llm_transform_1")
    # Patched prompt and resolved hash live inside ``options`` so they survive
    # ``state_from_record`` and reach the runtime YAML.
    patched_template = patched["options"]["prompt_template"]
    assert "{{interpretation:cool}}" not in patched_template
    assert "Innovative and creative" in patched_template
    assert patched["options"]["resolved_prompt_template_hash"] == resolved.resolved_prompt_template_hash

    # Verify provenance in DB.
    with service._engine.begin() as conn:
        state_row = conn.execute(select(composition_states_table).where(composition_states_table.c.id == str(new_state.id))).one()
    assert state_row.provenance == "interpretation_resolve"


@pytest.mark.asyncio
async def test_03b_resolve_recomputes_validation_for_patched_live_state(service) -> None:
    """Resolve must not carry stale unresolved-placeholder validation errors.

    A compose turn can persist a pending interpretation event and later leave
    the current state row marked invalid because runtime Jinja validation saw
    the unresolved ``{{interpretation:<term>}}`` placeholder. Resolving the
    event patches that placeholder out of the live state; the new
    interpretation_resolve row must recompute authoring validity instead of
    copying the stale error from the pre-resolve live row.
    """
    session_id = uuid4()
    surfacing_state = await _seed_state_with_llm_node(service, session_id=session_id)
    event = await service.create_pending_interpretation_event(
        session_id=session_id,
        composition_state_id=surfacing_state.id,
        affected_node_id="llm_transform_1",
        tool_call_id="call_42",
        user_term="cool",
        kind=InterpretationKind.VAGUE_TERM,
        llm_draft="Innovative and creative",
        model_identifier="anthropic/claude-opus-4-7",
        model_version="2026-05-01",
        provider="anthropic",
        composer_skill_hash="a" * 64,
    )
    stale_error = "Invalid Jinja2 template: expected token 'end of print statement', got ':'"
    await service.save_composition_state(
        session_id,
        CompositionStateData(
            nodes=[_llm_node()],
            metadata_={"name": "Phase 5b Test", "description": ""},
            is_valid=False,
            validation_errors=[stale_error],
        ),
        provenance="session_seed",
    )

    _resolved, new_state = await service.resolve_interpretation_event(
        session_id=session_id,
        event_id=event.id,
        choice=InterpretationChoice.ACCEPTED_AS_DRAFTED,
        amended_value=None,
        actor="user:alice",
    )

    assert stale_error not in list(new_state.validation_errors or ())


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
        kind=InterpretationKind.VAGUE_TERM,
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
    patched_template = patched["options"]["prompt_template"]
    assert "Strikingly original" in patched_template
    assert "{{interpretation:cool}}" not in patched_template


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
        kind=InterpretationKind.VAGUE_TERM,
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
        kind=InterpretationKind.VAGUE_TERM,
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
        kind=InterpretationKind.VAGUE_TERM,
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
        kind=InterpretationKind.VAGUE_TERM,
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
        kind=InterpretationKind.VAGUE_TERM,
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
        kind=InterpretationKind.VAGUE_TERM,
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
        kind=InterpretationKind.VAGUE_TERM,
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
    assert event.kind is None
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


@pytest.mark.asyncio
async def test_create_pending_after_session_opt_out_returns_opt_out_row(service) -> None:
    """Opt-out is enforced at the pending-event writer boundary."""
    session_id = uuid4()
    state = await _seed_state_with_llm_node(service, session_id=session_id)
    opt_out = await service.record_session_interpretation_opt_out(
        session_id=session_id,
        actor="user:alice",
    )

    event = await service.create_pending_interpretation_event(
        session_id=session_id,
        composition_state_id=state.id,
        affected_node_id="llm_transform_1",
        tool_call_id="call_after_opt_out",
        user_term="cool",
        kind=InterpretationKind.VAGUE_TERM,
        llm_draft="A draft definition",
        model_identifier="anthropic/claude-opus-4-7",
        model_version="2026-05-01",
        provider="anthropic",
        composer_skill_hash="a" * 64,
    )

    assert event.id == opt_out.id
    assert event.choice is InterpretationChoice.OPTED_OUT
    assert event.interpretation_source is InterpretationSource.AUTO_INTERPRETED_OPT_OUT

    pending_rows = await service.list_interpretation_events(session_id, status="pending")
    all_rows = await service.list_interpretation_events(session_id, status="all")
    assert pending_rows == []
    assert [row.id for row in all_rows] == [opt_out.id]


@pytest.mark.asyncio
async def test_record_auto_interpreted_no_surfaces_carries_kind(service) -> None:
    """Rate-cap no-surface rows still record the closed interpretation kind."""
    session_id = uuid4()
    with service._engine.begin() as conn:
        _insert_session(conn, str(session_id))

    event = await service.record_auto_interpreted_no_surfaces_event(
        session_id=session_id,
        actor="composer-llm",
        kind=InterpretationKind.LLM_PROMPT_TEMPLATE,
        model_identifier="anthropic/claude-opus-4-7",
        model_version="2026-05-01",
        provider="anthropic",
        composer_skill_hash="a" * 64,
    )

    assert event.choice is InterpretationChoice.OPTED_OUT
    assert event.interpretation_source is InterpretationSource.AUTO_INTERPRETED_NO_SURFACES
    assert event.kind is InterpretationKind.LLM_PROMPT_TEMPLATE
    assert event.arguments_hash is None
    assert event.hash_domain_version is None


@pytest.mark.asyncio
async def test_record_auto_interpreted_no_surfaces_requires_explicit_kind(service) -> None:
    """Rate-cap audit rows require the caller's explicit interpretation class."""
    session_id = uuid4()
    with service._engine.begin() as conn:
        _insert_session(conn, str(session_id))

    with pytest.raises(TypeError, match="kind"):
        await service.record_auto_interpreted_no_surfaces_event(
            session_id=session_id,
            actor="composer-llm",
            model_identifier="anthropic/claude-opus-4-7",
            model_version="2026-05-01",
            provider="anthropic",
            composer_skill_hash="a" * 64,
        )

    rows = await service.list_interpretation_events(session_id, status="all")
    assert rows == []


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


def _interpretation_row(**overrides: object) -> SimpleNamespace:
    values: dict[str, object] = {
        "id": str(uuid4()),
        "session_id": str(uuid4()),
        "composition_state_id": str(uuid4()),
        "affected_node_id": "llm_transform_1",
        "tool_call_id": "tool-call-abc",
        "user_term": "cool",
        "kind": "vague_term",
        "llm_draft": "A draft definition of cool",
        "accepted_value": "A draft definition of cool",
        "choice": "accepted_as_drafted",
        "created_at": datetime.now(UTC),
        "resolved_at": datetime.now(UTC),
        "actor": "user:alice",
        "model_identifier": "anthropic/claude-opus-4-7",
        "model_version": "2026-05-01",
        "provider": "anthropic",
        "composer_skill_hash": "a" * 64,
        "arguments_hash": "b" * 64,
        "hash_domain_version": "v2",
        "interpretation_source": "user_approved",
        "runtime_model_identifier_at_resolve": "anthropic/claude-opus-4-7",
        "runtime_model_version_at_resolve": "2026-05-01",
        "resolved_prompt_template_hash": "c" * 64,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_interpretation_row_conversion_rejects_empty_composition_state_id_as_bad_uuid() -> None:
    """Tier-1 empty-string corruption must reach UUID parsing, not become None."""
    row = _interpretation_row(composition_state_id="")
    with pytest.raises(ValueError, match=r"badly formed|UUID|hexadecimal"):
        _interpretation_event_record_from_row(row)


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
    # The patched template lives in ``options.prompt_template`` so it lands
    # on ``NodeSpec.options`` after ``state_from_record`` and reaches the
    # runtime YAML through ``generate_pipeline_dict``. See Phase 5b Task 9.
    patched_template = patched["options"]["prompt_template"]
    assert "{{interpretation:cool}}" not in patched_template
    assert "Innovative and creative" in patched_template
    # The other surrounding text is preserved verbatim.
    assert patched_template == "Rate how Innovative and creative this is."
    # Production node discriminator unchanged.
    assert patched["id"] == "llm_transform_1"
    assert patched["node_type"] == "transform"
    assert patched["plugin"] == "llm"
    assert "kind" not in patched


def test_patch_helper_resolves_structured_requirement_without_legacy_placeholder() -> None:
    state = _state_with_node(_structured_llm_node())

    nodes_out = _patch_llm_transform_prompt(
        state,
        affected_node_id="llm_transform_1",
        user_term="cool",
        accepted_value="Innovative and creative",
    )

    patched = next(iter(nodes_out))
    patched_options = patched["options"]
    patched_template = patched_options["prompt_template"]
    requirement = patched_options[INTERPRETATION_REQUIREMENTS_KEY][0]

    assert "{{interpretation:cool}}" not in patched_template
    assert patched_template == "Rate Innovative and creative this is."
    assert requirement["status"] == "resolved"
    assert requirement["accepted_value"] == "Innovative and creative"
    assert requirement["resolved_prompt_template_hash"] == stable_hash(patched_template)


def test_patch_helper_accepts_whitespace_tolerant_placeholder() -> None:
    """Resolve accepts the same whitespace placeholder form as staging."""
    state = _state_with_node(_llm_node(prompt_template="Rate how {{ interpretation : cool }} this is."))
    nodes_out = _patch_llm_transform_prompt(
        state,
        affected_node_id="llm_transform_1",
        user_term="cool",
        accepted_value="Innovative and creative",
    )
    patched = next(iter(nodes_out))
    patched_template = patched["options"]["prompt_template"]
    assert "{{ interpretation : cool }}" not in patched_template
    assert patched_template == "Rate how Innovative and creative this is."


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


def test_patch_helper_rejects_legacy_kind_only_llm_shape() -> None:
    """Helper-contract guard: private ``kind`` fixtures are not accepted."""
    state = _state_with_node(
        {
            "id": "llm_transform_1",
            "kind": "llm",
            "options": {"prompt_template": "Rate how {{interpretation:cool}} this is."},
        }
    )
    with pytest.raises(ValueError, match=r"node_type|plugin|LLM discriminator"):
        _patch_llm_transform_prompt(
            state,
            affected_node_id="llm_transform_1",
            user_term="cool",
            accepted_value="x",
        )


def test_patch_helper_rejects_non_llm_production_shape() -> None:
    """Helper-contract guard: production node shape with plugin != 'llm' raises."""
    state = _state_with_node(
        {
            "id": "csv_source_1",
            "node_type": "source",
            "plugin": "csv",
            "options": {},
        }
    )
    with pytest.raises(ValueError, match=r"llm|plugin|node_type"):
        _patch_llm_transform_prompt(
            state,
            affected_node_id="csv_source_1",
            user_term="cool",
            accepted_value="x",
        )


def test_patch_helper_rejects_missing_prompt_template() -> None:
    """Helper-contract guard: no options.prompt_template field ⇒ raise."""
    state = _state_with_node(
        {
            "id": "llm_transform_1",
            "node_type": "transform",
            "plugin": "llm",
            "options": {},
        }
    )
    with pytest.raises(ValueError, match="prompt_template"):
        _patch_llm_transform_prompt(
            state,
            affected_node_id="llm_transform_1",
            user_term="cool",
            accepted_value="x",
        )


def test_patch_helper_rejects_missing_options() -> None:
    """Helper-contract guard: no options mapping at all ⇒ raise.

    Mirrors the runtime contract: options is the carrier for prompt_template,
    so a node without options can never be a valid LLM interpretation target.
    """
    state = _state_with_node(
        {
            "id": "llm_transform_1",
            "node_type": "transform",
            "plugin": "llm",
        }
    )
    with pytest.raises(ValueError, match="options"):
        _patch_llm_transform_prompt(
            state,
            affected_node_id="llm_transform_1",
            user_term="cool",
            accepted_value="x",
        )


def test_arguments_hash_matches_domain_v2() -> None:
    """Cross-check: the arguments_hash a service writes is consistent with
    INTERPRETATION_HASH_DOMAIN_V2 over the recorded field set.

    Computes the expected hash from the 13-field domain and confirms it
    matches the stored row's arguments_hash after resolve. This guards
    against silent drift between the writer and the closed hash domain.
    """
    # Computed inside an asyncio test wrapper below.


@pytest.mark.asyncio
async def test_15_arguments_hash_matches_domain_v2(service) -> None:
    """The service's arguments_hash matches stable_hash() over INTERPRETATION_HASH_DOMAIN_V2."""
    session_id = uuid4()
    state = await _seed_state_with_llm_node(service, session_id=session_id)
    event = await service.create_pending_interpretation_event(
        session_id=session_id,
        composition_state_id=state.id,
        affected_node_id="llm_transform_1",
        tool_call_id="call_42",
        user_term="cool",
        kind=InterpretationKind.VAGUE_TERM,
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
        "kind": "vague_term",
        "llm_draft": "A draft of cool",
        "accepted_value": "A draft of cool",
        "actor": "user:alice",
        "model_identifier": "anthropic/claude-opus-4-7",
        "model_version": "2026-05-01",
        "provider": "anthropic",
        "composer_skill_hash": "a" * 64,
    }
    # Sanity: domain dict keys exactly match INTERPRETATION_HASH_DOMAIN_V2.
    assert set(expected_domain.keys()) == INTERPRETATION_HASH_DOMAIN_V2
    assert resolved.arguments_hash == stable_hash(expected_domain)
    assert resolved.hash_domain_version == "v2"


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
        kind=InterpretationKind.VAGUE_TERM,
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


# --------------------------------------------------------------------------- #
# Phase 5b Task 9 — composer round-trip: resolve must land the resolved
# prompt-template and the hash in NodeSpec.options so the runtime YAML
# carries both. Regression guard against the Task 4 escape where the patch
# helper wrote top-level fields that NodeSpec.from_dict dropped.
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_resolve_round_trips_through_state_from_record_and_yaml(service) -> None:
    """resolve → state_from_record → generate_yaml lands the resolved template
    and the cross-DB hash on the runtime YAML's options.

    The probe at Phase 5b Task 9 discovered that an earlier shape (top-level
    ``prompt_template`` / ``resolved_prompt_template_hash`` on the node JSON)
    was silently dropped by NodeSpec.from_dict — the runtime YAML still
    contained the unresolved placeholder, which would crash the F-17 gate
    on every interpretation-resolved execution. This test pins the fix:
    both fields must survive the round-trip into ``options``.
    """
    from elspeth.web.composer.yaml_generator import generate_yaml
    from elspeth.web.sessions.converters import state_from_record

    session_id = uuid4()

    # Seed a composition state with a NodeSpec-compatible node shape so
    # ``state_from_record`` reconstructs the LLM transform via NodeSpec.
    nodes = [
        {
            "id": "llm_transform_1",
            "node_type": "transform",
            "plugin": "llm",
            "input": "input",
            "on_success": "out",
            "on_error": "quarantine",
            "options": {
                "prompt_template": "Rate how {{interpretation:cool}} this is.",
                "model": "stub-model",
            },
            "condition": None,
            "routes": None,
            "fork_to": None,
            "branches": None,
            "policy": None,
            "merge": None,
        }
    ]
    with service._engine.begin() as conn:
        _insert_session(conn, str(session_id))
    state = await service.save_composition_state(
        session_id,
        CompositionStateData(
            nodes=nodes,
            is_valid=True,
            metadata_={"name": "t", "description": "t"},
        ),
        provenance="tool_call",
    )
    event = await service.create_pending_interpretation_event(
        session_id=session_id,
        composition_state_id=state.id,
        affected_node_id="llm_transform_1",
        tool_call_id="call_round_trip",
        user_term="cool",
        kind=InterpretationKind.VAGUE_TERM,
        llm_draft="modern and clear",
        model_identifier="anthropic/claude-opus-4-7",
        model_version="2026-05-01",
        provider="anthropic",
        composer_skill_hash="a" * 64,
    )
    resolved, new_state = await service.resolve_interpretation_event(
        session_id=session_id,
        event_id=event.id,
        choice=InterpretationChoice.AMENDED,
        amended_value="modern design + clear purpose",
        actor="user:alice",
        runtime_model_identifier="anthropic/claude-opus-4-7",
        runtime_model_version="2026-05-01",
    )

    # NodeSpec view must show the resolved template and the hash inside options.
    cs = state_from_record(new_state)
    assert len(cs.nodes) == 1
    node = cs.nodes[0]
    assert node.id == "llm_transform_1"
    assert "{{interpretation:cool}}" not in node.options["prompt_template"]
    assert "modern design + clear purpose" in node.options["prompt_template"]
    assert node.options["resolved_prompt_template_hash"] == resolved.resolved_prompt_template_hash

    # Generated YAML must carry both fields under transforms[0].options.
    yaml_str = generate_yaml(cs)
    assert "prompt_template: Rate how modern design + clear purpose this is." in yaml_str
    assert f"resolved_prompt_template_hash: {resolved.resolved_prompt_template_hash}" in yaml_str
    # Negative assertion: the placeholder must not survive into the YAML.
    assert "{{interpretation:cool}}" not in yaml_str


@pytest.mark.asyncio
async def test_resolve_structured_requirement_round_trips_without_authoring_metadata_in_yaml(service) -> None:
    from elspeth.web.composer.yaml_generator import generate_yaml
    from elspeth.web.sessions.converters import state_from_record

    session_id = uuid4()
    state = await _seed_state_with_llm_node(
        service,
        session_id=session_id,
        node=_structured_llm_node(),
    )
    event = await service.create_pending_interpretation_event(
        session_id=session_id,
        composition_state_id=state.id,
        affected_node_id="llm_transform_1",
        tool_call_id="call_structured_round_trip",
        user_term="cool",
        kind=InterpretationKind.VAGUE_TERM,
        llm_draft="modern and clear",
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
        runtime_model_identifier="anthropic/claude-opus-4-7",
        runtime_model_version="2026-05-01",
    )

    cs = state_from_record(new_state)
    node = cs.nodes[0]
    requirement = node.options[INTERPRETATION_REQUIREMENTS_KEY][0]
    assert node.options["prompt_template"] == "Rate modern and clear this is."
    assert node.options["resolved_prompt_template_hash"] == resolved.resolved_prompt_template_hash
    assert requirement["status"] == "resolved"
    assert requirement["accepted_value"] == "modern and clear"
    assert requirement["resolved_prompt_template_hash"] == resolved.resolved_prompt_template_hash

    yaml_str = generate_yaml(cs)
    assert "prompt_template: Rate modern and clear this is." in yaml_str
    assert PROMPT_TEMPLATE_PARTS_KEY not in yaml_str
    assert INTERPRETATION_REQUIREMENTS_KEY not in yaml_str


# --------------------------------------------------------------------------- #
# Phase 5b Task 11 — orphan-PENDING rehydration (refreshPending contract)
#
# A compose-loop crash between ``create_pending_interpretation_event``
# returning the row and the ToolResult propagating back to the frontend
# leaves an "orphan" PENDING interpretation_events row: written to the
# session DB, but the frontend never received the in-band notification
# that would normally raise the review affordance.
#
# Recovery contract (18a Task 11, spec lines 3053-3091): on session reload
# the frontend calls ``refreshPending``, which hits
# ``GET /api/sessions/{id}/interpretations?status=pending``. That route
# delegates to ``SessionServiceImpl.list_interpretation_events`` with
# ``status='pending'``. This test pins the service-layer contract that the
# orphan row IS visible to that read path.
#
# Scope discipline: per the spec, the optional cleanup job (auto-resolve
# orphans older than 7 days as ``choice='abandoned'``) is deferred to
# Phase 11 — NOT implemented here. The Task 11 backend deliverable is
# narrowly the rehydration test.
#
# Operates under the operator-acknowledged assumption that 18a Task 0
# (empirical LLM gate ≥ 8/10 staging runs emit ``{{interpretation:<term>}}``)
# passes.
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_16_refresh_pending_rehydrates_orphan_event_without_in_band_notification(
    service,
) -> None:
    """Spec Task 11 deliverable: a PENDING row the frontend never saw in-band
    is still discoverable via ``list_interpretation_events(status='pending')``.

    Models the orphan scenario: the writer commits the row to the session
    DB; the in-band ToolResult is lost (compose-loop crash, transport drop,
    or session abandonment between write and notification). On reload, the
    frontend's ``refreshPending`` path — i.e. ``list_interpretation_events``
    with ``status='pending'`` — must rediscover the orphan and re-raise the
    review affordance.

    Note on simulation: in unit tests there is no "in-band notification"
    channel to selectively suppress — the writer simply commits the row and
    the test then exercises the read path without any intervening tool-loop
    interaction. That mirrors the orphan scenario faithfully: from the read
    path's perspective, an orphan row is indistinguishable from a fresh
    pending row, which is exactly the contract that allows rehydration to
    work.
    """
    session_id = uuid4()
    state = await _seed_state_with_llm_node(service, session_id=session_id)

    # Writer commits the PENDING row. In the orphan scenario, the ToolResult
    # that would normally propagate this event to the frontend in-band is
    # lost between this call returning and the next user action. The unit
    # test simulates that by simply not invoking any further frontend-side
    # path before the refresh.
    orphan = await service.create_pending_interpretation_event(
        session_id=session_id,
        composition_state_id=state.id,
        affected_node_id="llm_transform_1",
        tool_call_id="call_orphan",
        user_term="cool",
        kind=InterpretationKind.VAGUE_TERM,
        llm_draft="An orphan-scenario draft",
        model_identifier="anthropic/claude-opus-4-7",
        model_version="2026-05-01",
        provider="anthropic",
        composer_skill_hash="a" * 64,
    )

    # refreshPending → GET /interpretations?status=pending →
    # list_interpretation_events(status='pending'). Must return the orphan.
    rehydrated = await service.list_interpretation_events(session_id, status="pending")

    assert len(rehydrated) == 1
    record = rehydrated[0]
    assert record.id == orphan.id
    assert record.choice is InterpretationChoice.PENDING
    assert record.interpretation_source is InterpretationSource.USER_APPROVED
    assert record.session_id == session_id
    assert record.composition_state_id == state.id
    assert record.affected_node_id == "llm_transform_1"
    assert record.tool_call_id == "call_orphan"
    assert record.user_term == "cool"
    assert record.kind is InterpretationKind.VAGUE_TERM
    assert record.llm_draft == "An orphan-scenario draft"
    # PENDING invariants: no resolution metadata yet.
    assert record.accepted_value is None
    assert record.resolved_at is None
    assert record.arguments_hash is None
    assert record.hash_domain_version is None


@pytest.mark.asyncio
async def test_17_refresh_pending_is_idempotent_across_repeated_calls(service) -> None:
    """The rehydration path is read-only and stable under repeated reload.

    Refreshing twice without resolving the orphan returns the same row both
    times — no implicit consumption, no side-effects on the DB. This is the
    contract that lets the frontend retry ``refreshPending`` on transient
    network failures without altering the audit state.
    """
    session_id = uuid4()
    state = await _seed_state_with_llm_node(service, session_id=session_id)

    orphan = await service.create_pending_interpretation_event(
        session_id=session_id,
        composition_state_id=state.id,
        affected_node_id="llm_transform_1",
        tool_call_id="call_orphan_idem",
        user_term="cool",
        kind=InterpretationKind.VAGUE_TERM,
        llm_draft="A draft",
        model_identifier="anthropic/claude-opus-4-7",
        model_version="2026-05-01",
        provider="anthropic",
        composer_skill_hash="a" * 64,
    )

    first = await service.list_interpretation_events(session_id, status="pending")
    second = await service.list_interpretation_events(session_id, status="pending")

    assert [e.id for e in first] == [orphan.id]
    assert [e.id for e in second] == [orphan.id]
    # The two reads must produce equal records (same row, same choice, same
    # resolved_at=None). Equality on the dataclass would suffice; we spot-
    # check the load-bearing fields for clarity.
    assert first[0].id == second[0].id
    assert first[0].choice is second[0].choice is InterpretationChoice.PENDING
    assert first[0].resolved_at is None
    assert second[0].resolved_at is None


@pytest.mark.asyncio
async def test_18_refresh_pending_drops_orphan_after_resolution(service) -> None:
    """Once the rehydrated orphan is resolved, ``status='pending'`` no longer
    returns it — the row remains in the table (audit-honest) but the
    refreshPending read filter excludes resolved rows.

    Pins the post-resolution contract: a session that has worked through an
    orphan does not see it on subsequent reloads, which is what the UX
    requires (the review affordance must come down).
    """
    session_id = uuid4()
    state = await _seed_state_with_llm_node(service, session_id=session_id)

    orphan = await service.create_pending_interpretation_event(
        session_id=session_id,
        composition_state_id=state.id,
        affected_node_id="llm_transform_1",
        tool_call_id="call_orphan_resolve",
        user_term="cool",
        kind=InterpretationKind.VAGUE_TERM,
        llm_draft="Draft to be accepted",
        model_identifier="anthropic/claude-opus-4-7",
        model_version="2026-05-01",
        provider="anthropic",
        composer_skill_hash="a" * 64,
    )

    # Rehydrate the orphan, then resolve it.
    before = await service.list_interpretation_events(session_id, status="pending")
    assert [e.id for e in before] == [orphan.id]

    await service.resolve_interpretation_event(
        session_id=session_id,
        event_id=orphan.id,
        choice=InterpretationChoice.ACCEPTED_AS_DRAFTED,
        amended_value=None,
        actor="user:alice",
        runtime_model_identifier=None,
        runtime_model_version=None,
    )

    # status='pending' no longer surfaces it...
    after_pending = await service.list_interpretation_events(session_id, status="pending")
    assert after_pending == []
    # ...but status='all' still does (audit-honest: the row persists).
    after_all = await service.list_interpretation_events(session_id, status="all")
    assert [e.id for e in after_all] == [orphan.id]
    assert after_all[0].choice is InterpretationChoice.ACCEPTED_AS_DRAFTED


@pytest.mark.asyncio
async def test_19_refresh_pending_isolates_orphans_by_session(service) -> None:
    """An orphan in session A does not leak into session B's refreshPending.

    Models the multi-session reload case: two sessions, each with its own
    orphan row. ``refreshPending`` for session A must return only A's
    orphan; session B's orphan must not be surfaced.

    Closes the would-be regression where a missing
    ``session_id`` predicate on the rehydration read would cross-pollinate
    review affordances between sessions — a privacy and correctness fault.
    """
    session_a = uuid4()
    state_a = await _seed_state_with_llm_node(service, session_id=session_a)
    session_b = uuid4()
    state_b = await _seed_state_with_llm_node(service, session_id=session_b)

    orphan_a = await service.create_pending_interpretation_event(
        session_id=session_a,
        composition_state_id=state_a.id,
        affected_node_id="llm_transform_1",
        tool_call_id="call_A",
        user_term="cool",
        kind=InterpretationKind.VAGUE_TERM,
        llm_draft="Draft A",
        model_identifier="anthropic/claude-opus-4-7",
        model_version="2026-05-01",
        provider="anthropic",
        composer_skill_hash="a" * 64,
    )
    orphan_b = await service.create_pending_interpretation_event(
        session_id=session_b,
        composition_state_id=state_b.id,
        affected_node_id="llm_transform_1",
        tool_call_id="call_B",
        user_term="cool",
        kind=InterpretationKind.VAGUE_TERM,
        llm_draft="Draft B",
        model_identifier="anthropic/claude-opus-4-7",
        model_version="2026-05-01",
        provider="anthropic",
        composer_skill_hash="a" * 64,
    )

    a_pending = await service.list_interpretation_events(session_a, status="pending")
    b_pending = await service.list_interpretation_events(session_b, status="pending")

    assert [e.id for e in a_pending] == [orphan_a.id]
    assert [e.id for e in b_pending] == [orphan_b.id]
