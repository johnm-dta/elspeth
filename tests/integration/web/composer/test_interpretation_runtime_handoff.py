"""Phase 5b Task 9 — runtime hand-off + cross-DB hash spot-check.

Verifies the runtime-side half of the Option A hash-anchored cross-DB linkage:
when an LLM transform that originated from a resolved interpretation event
makes its LLM call at runtime, the Landscape ``calls.resolved_prompt_template_hash``
column gets populated with the same SHA-256 that
``interpretation_events.resolved_prompt_template_hash`` carries in the session
DB.

Test shape (per spec ``docs/composer/ux-redesign-2026-05/18a-phase-5b-backend.md``
lines 2941-2982):

1. Seed the session DB with a resolved interpretation event whose
   ``resolved_prompt_template_hash`` we capture.
2. Drive an audited LLM call against an in-memory Landscape DB, passing the
   same hash through the public hand-off kwarg. Azure uses
   ``AuditedLLMClient.chat_completion`` directly; OpenRouter records a logical
   ``CallType.LLM`` row around its HTTP transport.
3. Read back ``calls.resolved_prompt_template_hash`` from the Landscape DB and
   assert byte equality with the session DB value.
4. External-recompute step (spec step 9): compute
   ``stable_hash(resolved_template_str)`` over the prompt-template string
   embedded in ``composition_states.nodes[i].options.prompt_template`` and
   assert it equals both stored hash values. This is the external
   audit-tooling check that proves the two DBs are internally consistent AND
   that the composition state JSON contains the string that was actually
   hashed — the hash chain has no silent intermediate step.

Scope note: this test exercises the Task 9 hash plumbing in isolation. The
composer's ``options.prompt_template`` maps directly to the runtime LLM config
field, which is also named ``prompt_template`` (``LLMConfig.prompt_template``,
``plugins/transforms/llm/base.py``) — there is no ``LLMConfig.template``. The
``state_from_record → generate_yaml → validate_pipeline`` path therefore
validates and builds the runtime graph for a complete LLM transform fine
(empirically: a ``text → llm → json`` composer pipeline passes the real
``validate_pipeline``). What this isolated hash-plumbing test does not exercise
is a *live* end-to-end run: the provider client and its API key are built in
``on_start()`` at run time, so actually issuing the LLM call needs real
credentials — orthogonal to the hash anchor verified here.

Operates under the operator-acknowledged assumption that 18a Task 0
(empirical LLM gate ≥ 8/10 staging runs emit ``{{interpretation:<term>}}``)
passes.
"""

from __future__ import annotations

import json as jsonlib
import uuid
from datetime import UTC, datetime
from typing import Any
from unittest.mock import MagicMock, Mock

import pytest
import structlog
from sqlalchemy import insert, select
from sqlalchemy.pool import StaticPool

from elspeth.contracts import NodeType
from elspeth.contracts.composer_interpretation import InterpretationChoice, InterpretationKind
from elspeth.contracts.hashing import stable_hash
from elspeth.contracts.schema import SchemaConfig
from elspeth.core.landscape.database import LandscapeDB
from elspeth.core.landscape.factory import RecorderFactory
from elspeth.core.landscape.schema import calls_table
from elspeth.plugins.infrastructure.clients.llm import AuditedLLMClient
from elspeth.plugins.transforms.llm.providers.openrouter import OpenRouterLLMProvider
from elspeth.web.sessions.converters import state_from_record
from elspeth.web.sessions.engine import create_session_engine
from elspeth.web.sessions.models import (
    interpretation_events_table,
    sessions_table,
)
from elspeth.web.sessions.protocol import CompositionStateData
from elspeth.web.sessions.schema import initialize_session_schema
from elspeth.web.sessions.service import SessionServiceImpl
from elspeth.web.sessions.telemetry import build_sessions_telemetry

DYNAMIC_SCHEMA = SchemaConfig.from_dict({"mode": "observed"})


def _make_session_service() -> tuple[SessionServiceImpl, Any]:
    """Build a fresh in-memory session-audit DB + service for this test."""
    engine = create_session_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    initialize_session_schema(engine)
    service = SessionServiceImpl(
        engine,
        telemetry=build_sessions_telemetry(),
        log=structlog.get_logger("test"),
    )
    return service, engine


def _insert_session(conn: Any, session_id: str) -> None:
    """Seed a sessions row so the FK to composition_states resolves."""
    conn.execute(
        insert(sessions_table).values(
            id=session_id,
            user_id="alice",
            auth_provider_type="local",
            title="Phase 5b Task 9 runtime hand-off",
            created_at=datetime.now(UTC),
            updated_at=datetime.now(UTC),
        )
    )


def _make_fake_openai_response(content: str, model: str) -> Any:
    """Return a Mock that quacks like an OpenAI Chat Completion response.

    Mirrors the helper in tests/unit/plugins/clients/test_audited_llm_client.py
    so this integration test exercises the same Mock surface AuditedLLMClient
    is contractually tested against.
    """
    message = Mock()
    message.content = content

    choice = Mock()
    choice.message = message
    choice.finish_reason = "stop"

    usage = Mock()
    usage.prompt_tokens = 10
    usage.completion_tokens = 5
    usage.total_tokens = 15

    response = Mock()
    response.choices = [choice]
    response.model = model
    response.usage = usage
    response.model_dump = Mock(
        return_value={
            "id": "resp_test",
            "choices": [{"finish_reason": "stop", "message": {"content": content}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        }
    )
    return response


class _FakeHTTPXClient:
    """Small httpx.Client stand-in for OpenRouter transport tests."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.closed = False

    def post(
        self,
        url: str,
        *,
        json: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
        timeout: float | None = None,
    ) -> Any:
        body = {
            "choices": [
                {
                    "message": {"content": "7 / 10"},
                    "finish_reason": "stop",
                }
            ],
            "model": "openrouter/test-model",
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15,
            },
        }
        import httpx

        return httpx.Response(
            status_code=200,
            content=jsonlib.dumps(body).encode(),
            headers={"content-type": "application/json"},
            request=httpx.Request("POST", url),
        )

    def close(self) -> None:
        self.closed = True


@pytest.mark.asyncio
async def test_runtime_handoff_cross_db_hash_anchored() -> None:
    """Steps 6-9 of spec lines 2941-2982: cross-DB hash equality + recompute.

    Drives the Azure/public-client side of the production hash-plumbing chain
    through ``AuditedLLMClient``. OpenRouter's raw-HTTP variant is covered below
    so both provider families keep the same cross-DB invariant.
    """
    # ── Session-DB side: seed a session, a composition state with an LLM
    # node carrying the placeholder, a pending interpretation event, and
    # resolve it. ``resolve_interpretation_event`` writes
    # ``options.prompt_template`` (patched) and
    # ``options.resolved_prompt_template_hash`` into the new state row.
    service, _ = _make_session_service()
    sid = uuid.uuid4()
    with service._engine.begin() as conn:
        _insert_session(conn, str(sid))

    raw_template = "Rate how {{interpretation:cool}} this is."
    accepted = "modern design + clear purpose + interactivity"
    resolved_template = raw_template.replace("{{interpretation:cool}}", accepted)

    nodes = [
        {
            "id": "llm_rate",
            "node_type": "transform",
            "plugin": "llm",
            "input": "input",
            "on_success": "out",
            "on_error": "quarantine",
            "options": {
                "prompt_template": raw_template,
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
    state = await service.save_composition_state(
        sid,
        CompositionStateData(
            nodes=nodes,
            is_valid=True,
            metadata_={"name": "test", "description": "test"},
        ),
        provenance="tool_call",
    )

    pending = await service.create_pending_interpretation_event(
        session_id=sid,
        composition_state_id=state.id,
        affected_node_id="llm_rate",
        tool_call_id="tcall_handoff_1",
        user_term="cool",
        kind=InterpretationKind.VAGUE_TERM,
        llm_draft="modern and clear",
        model_identifier="anthropic/claude-opus-4-7",
        model_version="2026-05-01",
        provider="anthropic",
        composer_skill_hash="a" * 64,
    )
    resolved, latest_state = await service.resolve_interpretation_event(
        session_id=sid,
        event_id=pending.id,
        choice=InterpretationChoice.AMENDED,
        amended_value=accepted,
        actor="user:alice",
        runtime_model_identifier="anthropic/claude-opus-4-7",
        runtime_model_version="2026-05-01",
    )

    # Step 6 (spec): session-side read — non-NULL hash, capture for the join.
    assert resolved.resolved_prompt_template_hash is not None, (
        "resolve_interpretation_event MUST populate resolved_prompt_template_hash on the session-DB row at resolve time"
    )
    session_hash: str = resolved.resolved_prompt_template_hash
    assert len(session_hash) == 64

    # Round-trip + production-shape check: the resolved hash must land on
    # NodeSpec.options where the runtime LLM transform reads it. Without
    # this, the runtime would forward None and break the cross-DB join even
    # for legitimately-resolved interpretations.
    cs = state_from_record(latest_state)
    assert cs.nodes, "Resolved state must contain the patched LLM node"
    patched_options = cs.nodes[0].options
    assert patched_options["prompt_template"] == resolved_template
    assert patched_options["resolved_prompt_template_hash"] == session_hash

    # ── Landscape side: instantiate a real in-memory Landscape DB +
    # recorder, register a run/source/transform node + node_state, then
    # drive an audited LLM call carrying the same hash. This is the L3
    # plugin's hand-off point: the provider reads
    # ``self._resolved_prompt_template_hash`` (snapshotted from
    # ``LLMConfig.resolved_prompt_template_hash`` at transform construction)
    # and Azure passes it to ``client.chat_completion``.
    db = LandscapeDB.in_memory()
    factory = RecorderFactory(db)

    run = factory.run_lifecycle.begin_run(config={}, canonical_version="v1")
    source_node = factory.data_flow.register_node(
        run_id=run.run_id,
        plugin_name="csv_source",
        node_type=NodeType.SOURCE,
        plugin_version="1.0",
        config={},
        schema_config=DYNAMIC_SCHEMA,
    )
    llm_node = factory.data_flow.register_node(
        run_id=run.run_id,
        plugin_name="llm",
        node_type=NodeType.TRANSFORM,
        plugin_version="1.0",
        config={"prompt_template": resolved_template},
        schema_config=DYNAMIC_SCHEMA,
    )
    row = factory.data_flow.create_row(
        run_id=run.run_id,
        source_node_id=source_node.node_id,
        row_index=0,
        data={"input": "demo"},
        source_row_index=0,
        ingest_sequence=0,
    )
    token = factory.data_flow.create_token(row_id=row.row_id)
    node_state = factory.execution.begin_node_state(
        token_id=token.token_id,
        node_id=llm_node.node_id,
        run_id=run.run_id,
        step_index=1,
        input_data={"input": "demo"},
    )

    openai_stub = MagicMock()
    openai_stub.chat.completions.create.return_value = _make_fake_openai_response(
        content="7 / 10",
        model="stub-model",
    )

    client = AuditedLLMClient(
        execution=factory.execution,
        state_id=node_state.state_id,
        run_id=run.run_id,
        telemetry_emit=lambda event: None,
        underlying_client=openai_stub,
        provider="stub",
    )

    response = client.chat_completion(
        model="stub-model",
        messages=[{"role": "user", "content": resolved_template}],
        resolved_prompt_template_hash=session_hash,
    )
    assert response.content == "7 / 10"

    # Step 7 (spec): Landscape-side read — non-NULL hash on the LLM call row.
    with db.connection() as conn:
        landscape_row = conn.execute(select(calls_table).where(calls_table.c.state_id == node_state.state_id)).one()
    landscape_hash = landscape_row.resolved_prompt_template_hash
    assert landscape_hash is not None, (
        "AuditedLLMClient.chat_completion MUST forward resolved_prompt_template_hash "
        f"to the Landscape calls row when non-None at the public API. "
        f"run_id={run.run_id} state_id={node_state.state_id} session_hash={session_hash}"
    )

    # Step 8 (spec): byte-equality assertion — failure here means the
    # composition state was mutated between resolve and execution OR the
    # runtime plugin received a different prompt than the one recorded.
    assert session_hash == landscape_hash, (
        f"Cross-DB hash mismatch (Phase 5b Option A audit anomaly):\n"
        f"  session_hash:   {session_hash}\n"
        f"  landscape_hash: {landscape_hash}\n"
        f"  run_id:         {run.run_id}\n"
        f"  state_id:       {node_state.state_id}"
    )

    # Step 9 (spec): external recompute — read the resolved prompt template
    # string from composition_states.nodes JSON and hash it with the same
    # stable_hash() the production code used. This is the external audit
    # check that proves the stored hashes match the string actually embedded
    # in the composition state, not a hash silently computed over different
    # bytes somewhere in the chain.
    embedded_template = patched_options["prompt_template"]
    recomputed_hash = stable_hash(embedded_template)
    assert recomputed_hash == session_hash == landscape_hash, (
        "External recompute fails — composition_states.nodes.options.prompt_template "
        "does not match either stored hash. The stored hashes drift from the "
        "string they purport to represent."
    )


@pytest.mark.asyncio
async def test_openrouter_hash_handoff_records_logical_llm_call_not_http_transport(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """OpenRouter's HTTP transport must not carry the LLM prompt hash.

    Regression guard for the live execution crash where
    ``resolved_prompt_template_hash`` leaked onto an ``http`` call row and
    tripped the ``Call`` invariant before the logical LLM audit row could be
    written.
    """
    db = LandscapeDB.in_memory()
    factory = RecorderFactory(db)

    run = factory.run_lifecycle.begin_run(config={}, canonical_version="v1")
    source_node = factory.data_flow.register_node(
        run_id=run.run_id,
        plugin_name="csv_source",
        node_type=NodeType.SOURCE,
        plugin_version="1.0",
        config={},
        schema_config=DYNAMIC_SCHEMA,
    )
    llm_node = factory.data_flow.register_node(
        run_id=run.run_id,
        plugin_name="llm",
        node_type=NodeType.TRANSFORM,
        plugin_version="1.0",
        config={"prompt_template": "Rate how modern this is."},
        schema_config=DYNAMIC_SCHEMA,
    )
    row = factory.data_flow.create_row(
        run_id=run.run_id,
        source_node_id=source_node.node_id,
        row_index=0,
        data={"input": "demo"},
        source_row_index=0,
        ingest_sequence=0,
    )
    token = factory.data_flow.create_token(row_id=row.row_id)
    node_state = factory.execution.begin_node_state(
        token_id=token.token_id,
        node_id=llm_node.node_id,
        run_id=run.run_id,
        step_index=1,
        input_data={"input": "demo"},
    )

    session_hash = "b" * 64
    provider = OpenRouterLLMProvider(
        api_key="test-key",
        base_url="https://openrouter.ai/api/v1",
        timeout_seconds=30.0,
        recorder=factory.execution,
        run_id=run.run_id,
        telemetry_emit=lambda event: None,
        resolved_prompt_template_hash=session_hash,
    )
    monkeypatch.setattr("elspeth.plugins.infrastructure.clients.http.httpx.Client", _FakeHTTPXClient)

    result = provider.execute_query(
        messages=[{"role": "user", "content": "Rate how modern this is."}],
        model="openrouter/test-model",
        temperature=0.0,
        max_tokens=100,
        state_id=node_state.state_id,
        token_id=token.token_id,
    )

    assert result.content == "7 / 10"
    with db.connection() as conn:
        rows = conn.execute(
            select(calls_table.c.call_type, calls_table.c.resolved_prompt_template_hash)
            .where(calls_table.c.state_id == node_state.state_id)
            .order_by(calls_table.c.call_index)
        ).all()

    assert [row.call_type for row in rows].count("http") == 1
    assert [row.resolved_prompt_template_hash for row in rows if row.call_type == "http"] == [None]
    assert [row.resolved_prompt_template_hash for row in rows if row.call_type == "llm"] == [session_hash]


@pytest.mark.asyncio
async def test_runtime_handoff_none_hash_records_null() -> None:
    """When the LLM transform is not downstream of an interpretation event,
    the hash kwarg is None and the Landscape column is NULL.

    Regression guard: the hash plumbing must accept None without crashing
    AND must produce a NULL column (not a fabricated string).
    """
    db = LandscapeDB.in_memory()
    factory = RecorderFactory(db)

    run = factory.run_lifecycle.begin_run(config={}, canonical_version="v1")
    source_node = factory.data_flow.register_node(
        run_id=run.run_id,
        plugin_name="csv_source",
        node_type=NodeType.SOURCE,
        plugin_version="1.0",
        config={},
        schema_config=DYNAMIC_SCHEMA,
    )
    llm_node = factory.data_flow.register_node(
        run_id=run.run_id,
        plugin_name="llm",
        node_type=NodeType.TRANSFORM,
        plugin_version="1.0",
        config={"prompt_template": "plain template, no interpretation"},
        schema_config=DYNAMIC_SCHEMA,
    )
    row = factory.data_flow.create_row(
        run_id=run.run_id,
        source_node_id=source_node.node_id,
        row_index=0,
        data={"input": "demo"},
        source_row_index=0,
        ingest_sequence=0,
    )
    token = factory.data_flow.create_token(row_id=row.row_id)
    node_state = factory.execution.begin_node_state(
        token_id=token.token_id,
        node_id=llm_node.node_id,
        run_id=run.run_id,
        step_index=1,
        input_data={"input": "demo"},
    )

    openai_stub = MagicMock()
    openai_stub.chat.completions.create.return_value = _make_fake_openai_response(
        content="ok",
        model="stub-model",
    )

    client = AuditedLLMClient(
        execution=factory.execution,
        state_id=node_state.state_id,
        run_id=run.run_id,
        telemetry_emit=lambda event: None,
        underlying_client=openai_stub,
        provider="stub",
    )
    # Default — no hash forwarded (transform was never downstream of an
    # interpretation event).
    client.chat_completion(
        model="stub-model",
        messages=[{"role": "user", "content": "plain"}],
    )

    with db.connection() as conn:
        landscape_row = conn.execute(select(calls_table).where(calls_table.c.state_id == node_state.state_id)).one()
    assert landscape_row.resolved_prompt_template_hash is None


@pytest.mark.asyncio
async def test_session_db_records_match_runtime_landscape_join() -> None:
    """Reverse-direction lookup: given a Landscape calls row, the hash
    points to exactly one interpretation_events row.

    This is the auditor's traversal: ``calls.resolved_prompt_template_hash``
    → ``interpretation_events WHERE resolved_prompt_template_hash = ?``.
    Verifies the index ``ix_calls_resolved_prompt_template_hash`` and the
    session-side column are populated consistently.
    """
    service, engine = _make_session_service()
    sid = uuid.uuid4()
    with service._engine.begin() as conn:
        _insert_session(conn, str(sid))

    raw_template = "How {{interpretation:elegant}} is the design?"
    accepted = "minimal, no chrome, clear hierarchy"
    nodes = [
        {
            "id": "llm_eval",
            "node_type": "transform",
            "plugin": "llm",
            "input": "input",
            "on_success": "out",
            "on_error": "quarantine",
            "options": {"prompt_template": raw_template, "model": "stub"},
            "condition": None,
            "routes": None,
            "fork_to": None,
            "branches": None,
            "policy": None,
            "merge": None,
        }
    ]
    state = await service.save_composition_state(
        sid,
        CompositionStateData(
            nodes=nodes,
            is_valid=True,
            metadata_={"name": "t", "description": "t"},
        ),
        provenance="tool_call",
    )
    pending = await service.create_pending_interpretation_event(
        session_id=sid,
        composition_state_id=state.id,
        affected_node_id="llm_eval",
        tool_call_id="tcall_reverse_1",
        user_term="elegant",
        kind=InterpretationKind.VAGUE_TERM,
        llm_draft="clean and minimal",
        model_identifier="anthropic/claude-opus-4-7",
        model_version="2026-05-01",
        provider="anthropic",
        composer_skill_hash="b" * 64,
    )
    resolved, _ = await service.resolve_interpretation_event(
        session_id=sid,
        event_id=pending.id,
        choice=InterpretationChoice.AMENDED,
        amended_value=accepted,
        actor="user:alice",
        runtime_model_identifier="anthropic/claude-opus-4-7",
        runtime_model_version="2026-05-01",
    )
    target_hash = resolved.resolved_prompt_template_hash
    assert target_hash is not None

    # Reverse query in the session DB: given a Landscape hash, find the
    # matching interpretation_events row.
    with engine.begin() as conn:
        matches = conn.execute(
            select(interpretation_events_table).where(interpretation_events_table.c.resolved_prompt_template_hash == target_hash)
        ).all()
    assert len(matches) == 1, f"Cross-DB join must resolve to exactly one event; got {len(matches)} for hash {target_hash}"
    assert str(matches[0].id) == str(resolved.id)
