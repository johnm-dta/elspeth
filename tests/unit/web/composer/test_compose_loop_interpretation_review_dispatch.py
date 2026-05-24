"""Compose-loop dispatch tests for ``request_interpretation_review`` (Phase 5b Task 5 follow-on).

Task 5 (commit a37a08f4a) shipped the tool surface — registry, handler,
rate limits, redaction, F-6 service writer — but the tool was not yet
reachable through the live compose loop. This module covers the
compose-loop wiring delivered in the Task 5 follow-on commit:

* **F-5a startup assert**: ``ComposerServiceImpl.__init__`` re-reads the
  composer skill markdown and asserts its SHA-256 still matches the
  LRU-cached hash. Mid-process on-disk drift raises ``RuntimeError`` with
  an operator-actionable message.
* **F-5c skill_markdown_history upsert**: the first ``_compose_loop``
  entry upserts the in-memory skill text into
  ``skill_markdown_history`` (INSERT OR IGNORE). Subsequent compose()
  calls observe the per-instance flag and skip the upsert.
* **Compose-loop dispatch**: ``_SESSION_AWARE_TOOL_HANDLERS`` is now
  recognised by the loop; ``request_interpretation_review`` is awaited
  with the injected service methods and the result is recorded through
  the same audit envelope discipline as the sync path.
* **F-6 rate-cap branch**: the per-term rate cap raises a
  ``ToolArgumentError`` carrying ``code=RATE_CAP_PER_TERM``; the
  dispatcher emits the F-15 ``interpretation_rate_cap_exceeded``
  operational telemetry signal AND writes the
  AUTO_INTERPRETED_NO_SURFACES audit row BEFORE returning the ARG_ERROR
  to the LLM.

Test-path discipline (CLAUDE.md "Never bypass production code paths"):
all tests drive the compose loop through ``_run_one_turn_for_test``,
which exercises the same ``_compose_loop`` body the live web server
uses.
"""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from unittest.mock import patch
from uuid import UUID, uuid4

import pytest
import structlog
from sqlalchemy import insert, select
from sqlalchemy.pool import StaticPool

from elspeth.contracts.composer_interpretation import (
    InterpretationChoice,
    InterpretationKind,
    InterpretationSource,
)
from elspeth.web.composer.protocol import ToolArgumentError
from elspeth.web.composer.service import ComposerAvailability, ComposerServiceImpl, _pending_interpretation_review_repair_message
from elspeth.web.composer.state import (
    CompositionState,
    NodeSpec,
    PipelineMetadata,
    SourceSpec,
)
from elspeth.web.composer.tools import (
    RATE_CAP_CODE_TO_TELEMETRY_CAP_TYPE,
    RATE_CAP_PER_SESSION_DAY_CODE,
    RATE_CAP_PER_TERM_CODE,
)
from elspeth.web.execution.schemas import ValidationReadiness, ValidationResult
from elspeth.web.interpretation_state import INTERPRETATION_REQUIREMENTS_KEY, SOURCE_AUTHORING_KEY
from elspeth.web.sessions.engine import create_session_engine
from elspeth.web.sessions.models import (
    sessions_table,
    skill_markdown_history_table,
)
from elspeth.web.sessions.protocol import CompositionStateData
from elspeth.web.sessions.schema import initialize_session_schema
from elspeth.web.sessions.service import SessionServiceImpl
from elspeth.web.sessions.telemetry import build_sessions_telemetry, observed_value

# ---------------------------------------------------------------------------
# Lightweight fake LLM that emits a single request_interpretation_review call.
# ---------------------------------------------------------------------------


def _fake_response_with_tool_call(
    *,
    tool_call_id: str,
    tool_name: str,
    arguments: dict[str, Any],
    response_model: str = "anthropic/claude-opus-4-7-20260101",
) -> Any:
    """Build a minimal LiteLLM-shaped response carrying one tool call.

    The real LiteLLM response has a long surface; the compose loop reads
    only ``response.choices[0].message.{content,tool_calls}`` and
    ``response.model`` for ``safe_response_model``. We synthesise the
    minimum the loop needs.
    """
    return _fake_response_with_tool_calls(
        tool_calls=[{"id": tool_call_id, "name": tool_name, "arguments": arguments}],
        response_model=response_model,
    )


def _execution_ready() -> ValidationReadiness:
    return ValidationReadiness(authoring_valid=True, execution_ready=True, completion_ready=True, blockers=[])


def _fake_response_with_tool_calls(
    *,
    tool_calls: list[dict[str, Any]],
    response_model: str = "anthropic/claude-opus-4-7-20260101",
) -> Any:
    """Build a minimal LiteLLM-shaped response carrying one or more tool calls."""

    class _Func:
        def __init__(self, name: str, arguments: str) -> None:
            self.name = name
            self.arguments = arguments

    class _ToolCall:
        def __init__(self, id_: str, function: _Func) -> None:
            self.id = id_
            self.function = function

    class _Message:
        def __init__(self, tool_calls: list[_ToolCall] | None, content: str | None) -> None:
            self.tool_calls = tool_calls
            self.content = content

    class _Choice:
        def __init__(self, message: _Message) -> None:
            self.message = message

    class _Response:
        def __init__(self) -> None:
            self.choices = [
                _Choice(
                    _Message(
                        tool_calls=[
                            _ToolCall(
                                str(call["id"]),
                                _Func(str(call["name"]), json.dumps(call["arguments"])),
                            )
                            for call in tool_calls
                        ],
                        content=None,
                    )
                )
            ]
            self.model = response_model

    return _Response()


def _fake_text_response(text: str) -> Any:
    """Build a no-tool-calls assistant response (terminates the compose loop)."""

    class _Message:
        def __init__(self, content: str) -> None:
            self.content = content
            self.tool_calls = None

    class _Choice:
        def __init__(self, message: _Message) -> None:
            self.message = message

    class _Response:
        def __init__(self) -> None:
            self.choices = [_Choice(_Message(text))]
            self.model = "anthropic/claude-opus-4-7-20260101"

    return _Response()


class _ScriptedLLM:
    """Returns a sequence of pre-built LiteLLM responses in order."""

    def __init__(self, responses: list[Any]) -> None:
        self._responses = list(responses)

    async def __call__(self, _messages: Any, _tools: Any) -> Any:
        if not self._responses:
            return _fake_text_response("Done.")
        return self._responses.pop(0)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


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
def sessions_service(engine) -> SessionServiceImpl:
    return SessionServiceImpl(
        engine,
        telemetry=build_sessions_telemetry(),
        log=structlog.get_logger("test.sessions"),
    )


def _llm_node_spec(term: str = "cool") -> NodeSpec:
    return NodeSpec(
        id="rate_node",
        node_type="transform",
        plugin="llm",
        input="rows",
        on_success="out",
        on_error=None,
        options={"prompt_template": f"Rate how {{{{interpretation:{term}}}}} this row is."},
        condition=None,
        routes=None,
        fork_to=None,
        branches=None,
        policy=None,
        merge=None,
    )


def _llm_node_spec_with_id(node_id: str, *, term: str = "cool") -> NodeSpec:
    """Build an LLM-transform NodeSpec with a caller-chosen id and an
    ``{{interpretation:<term>}}`` placeholder. Used by the rate-cap tests
    that need multiple distinct sites sharing one user_term — the dedup
    gate keys on (kind, user_term, affected_node_id) so distinct ids
    accumulate the per-term count without idempotent intercepts.
    """
    return NodeSpec(
        id=node_id,
        node_type="transform",
        plugin="llm",
        input="rows",
        on_success="out",
        on_error=None,
        options={"prompt_template": f"Rate how {{{{interpretation:{term}}}}} this row is."},
        condition=None,
        routes=None,
        fork_to=None,
        branches=None,
        policy=None,
        merge=None,
    )


def _state_with_llm_node(term: str = "cool") -> CompositionState:
    return CompositionState(
        source=None,
        nodes=(_llm_node_spec(term),),
        edges=(),
        outputs=(),
        metadata=PipelineMetadata(),
        version=1,
    )


def _state_with_prompt_template_review_node() -> CompositionState:
    prompt = "Read {{ row.html }} and return JSON."
    return CompositionState(
        source=None,
        nodes=(
            NodeSpec(
                id="rate_node",
                node_type="transform",
                plugin="llm",
                input="rows",
                on_success="out",
                on_error=None,
                options={
                    "prompt_template": prompt,
                    INTERPRETATION_REQUIREMENTS_KEY: [
                        {
                            "id": "prompt_template_review",
                            "kind": InterpretationKind.LLM_PROMPT_TEMPLATE.value,
                            "user_term": "llm_prompt_template:rate_node",
                            "status": "pending",
                            "draft": prompt,
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
        metadata=PipelineMetadata(),
        version=1,
    )


def _state_with_source_review() -> CompositionState:
    return CompositionState(
        source=SourceSpec(
            plugin="csv",
            on_success="rows",
            options={
                "path": "/tmp/generated.csv",
                SOURCE_AUTHORING_KEY: {
                    "modality": "llm_generated",
                    "content_hash": "0" * 64,
                    "review_event_id": None,
                    "resolved_kind": None,
                },
                INTERPRETATION_REQUIREMENTS_KEY: [
                    {
                        "id": "source_review:inline_source_url_list",
                        "kind": InterpretationKind.INVENTED_SOURCE.value,
                        "user_term": "inline_source_url_list",
                        "status": "pending",
                        "draft": "url\nhttps://example.gov.au/\n",
                        "event_id": None,
                        "accepted_value": None,
                        "accepted_artifact_hash": None,
                        "resolved_prompt_template_hash": None,
                    }
                ],
            },
            on_validation_failure="discard",
        ),
        nodes=(),
        edges=(),
        outputs=(),
        metadata=PipelineMetadata(),
        version=1,
    )


def _state_with_pipeline_decision_review() -> CompositionState:
    return CompositionState(
        source=None,
        nodes=(
            NodeSpec(
                id="drop_raw_html",
                node_type="transform",
                plugin="field_mapper",
                input="scored_rows",
                on_success="clean_rows",
                on_error=None,
                options={
                    "mapping": {
                        "url": "url",
                        "agency": "agency",
                        "primary_colours": "primary_colours",
                    },
                    "select_only": True,
                    INTERPRETATION_REQUIREMENTS_KEY: [
                        {
                            "id": "drop_raw_html_review",
                            "kind": InterpretationKind.PIPELINE_DECISION.value,
                            "user_term": "drop_raw_html_fields",
                            "status": "pending",
                            "draft": "Drop the scraped raw HTML and fingerprint fields before saving the JSON output.",
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
        metadata=PipelineMetadata(),
        version=1,
    )


def _state_with_unreviewed_raw_html_cleanup() -> CompositionState:
    return CompositionState(
        source=None,
        nodes=(
            NodeSpec(
                id="fetch_pages",
                node_type="transform",
                plugin="web_scrape",
                input="rows",
                on_success="scraped_rows",
                on_error=None,
                options={
                    "url_field": "url",
                    "content_field": "content",
                    "fingerprint_field": "content_fingerprint",
                },
                condition=None,
                routes=None,
                fork_to=None,
                branches=None,
                policy=None,
                merge=None,
            ),
            NodeSpec(
                id="drop_raw_html",
                node_type="transform",
                plugin="field_mapper",
                input="scored_rows",
                on_success="clean_rows",
                on_error=None,
                options={
                    "mapping": {
                        "url": "url",
                        "primary_colours": "primary_colours",
                    },
                    "select_only": True,
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
        metadata=PipelineMetadata(),
        version=1,
    )


def _set_pipeline_with_pending_interpretation_args(term: str = "cool") -> dict[str, Any]:
    return {
        "source": {
            "plugin": "null",
            "on_success": "rows",
            "options": {},
        },
        "nodes": [
            {
                "id": "rate_node",
                "node_type": "transform",
                "plugin": "llm",
                "input": "rows",
                "on_success": "scored_rows",
                "on_error": "discard",
                "options": {
                    "provider": "openrouter",
                    "model": "openai/gpt-4o-mini",
                    "api_key": {"secret_ref": "OPENROUTER_API_KEY"},
                    "prompt_template": f"Rate how {{{{interpretation:{term}}}}} this row is.",
                    "schema": {"mode": "observed"},
                },
            }
        ],
        "edges": [],
        "outputs": [],
    }


async def _seed_session_and_state(
    service: SessionServiceImpl,
    *,
    user_id: str = "alice",
    state: CompositionState | None = None,
) -> tuple[UUID, UUID]:
    """Seed a session row + composition_states row carrying the LLM node.

    Returns ``(session_id, composition_state_id)``.
    """
    session_id = uuid4()
    with service._engine.begin() as conn:
        conn.execute(
            insert(sessions_table).values(
                id=str(session_id),
                user_id=user_id,
                auth_provider_type="local",
                title="Phase 5b Task 5 follow-on test",
                created_at=datetime.now(UTC),
                updated_at=datetime.now(UTC),
            )
        )
    state_dict = (state or _state_with_llm_node()).to_dict()
    state_record = await service.save_composition_state(
        session_id,
        CompositionStateData(
            nodes=state_dict["nodes"],
            sources=state_dict["sources"],
            metadata_=state_dict["metadata"],
            is_valid=True,
        ),
        provenance="tool_call",
    )
    return session_id, state_record.id


@pytest.fixture(autouse=True)
def _force_composer_available(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep tests independent of local API keys — same pattern as the wider
    composer test suite (conftest.py ``_composer_available_for_phase3``)."""

    def _available(self: ComposerServiceImpl) -> ComposerAvailability:
        return ComposerAvailability(available=True, model=self._model, provider="anthropic")

    monkeypatch.setattr(ComposerServiceImpl, "_compute_availability", _available)


def _build_composer(tmp_path: Path, sessions_service: SessionServiceImpl) -> ComposerServiceImpl:
    from unittest.mock import MagicMock

    from elspeth.web.catalog.protocol import CatalogService
    from elspeth.web.config import WebSettings

    catalog = MagicMock(spec=CatalogService)
    catalog.list_sources.return_value = []
    catalog.list_transforms.return_value = []
    catalog.list_sinks.return_value = []
    settings = WebSettings(
        data_dir=tmp_path,
        composer_max_composition_turns=15,
        composer_max_discovery_turns=10,
        composer_timeout_seconds=85.0,
        composer_rate_limit_per_minute=10,
        composer_model="anthropic/claude-opus-4-7",
        shareable_link_signing_key=b"\x00" * 32,
    )
    return ComposerServiceImpl(
        catalog=catalog,
        settings=settings,
        sessions_service=sessions_service,
        session_engine=sessions_service._engine,
    )


def test_composer_runtime_preflight_uses_backend_readiness_contract(
    tmp_path: Path,
    sessions_service: SessionServiceImpl,
) -> None:
    """Composer preflight relies on typed readiness, not placeholder masking."""

    composer = _build_composer(tmp_path, sessions_service)
    state = _state_with_llm_node()
    expected = ValidationResult(is_valid=True, checks=[], errors=[], readiness=_execution_ready())

    with patch("elspeth.web.composer.service.validate_pipeline", return_value=expected) as validate:
        result = composer._runtime_preflight(state, user_id="alice")

    assert result is expected
    validate.assert_called_once()
    assert "allow_pending_interpretation_placeholders" not in validate.call_args.kwargs


# ---------------------------------------------------------------------------
# Test 1 — end-to-end dispatch through _SESSION_AWARE_TOOL_HANDLERS
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_compose_loop_dispatches_request_interpretation_review(
    tmp_path: Path,
    sessions_service: SessionServiceImpl,
) -> None:
    """The compose loop intercepts ``request_interpretation_review``, awaits
    the session-aware handler, persists the pending row, and threads the
    SUCCESS result through the audit envelope.

    Validates the full dispatch wiring: the LLM tool call name maps to
    ``_SESSION_AWARE_TOOL_HANDLERS``, the kwargs are built from compose-
    loop context, the F-5a/F-5c skill-hash machinery is exercised, and
    the audit recorder captures the SUCCESS invocation with the
    interpretation_review_pending payload.
    """
    composer = _build_composer(tmp_path, sessions_service)
    state = _state_with_llm_node()
    session_id, state_id = await _seed_session_and_state(sessions_service)

    llm = _ScriptedLLM(
        [
            _fake_response_with_tool_call(
                tool_call_id="call_42",
                tool_name="request_interpretation_review",
                arguments={
                    "affected_node_id": "rate_node",
                    "kind": "vague_term",
                    "user_term": "cool",
                    "llm_draft": "Visually appealing and well-organized.",
                },
            ),
            _fake_text_response("Done — surfaced the interpretation for review."),
        ]
    )

    result = await composer._run_one_turn_for_test(
        llm=llm,
        session_id=str(session_id),
        current_state_id=str(state_id),
        initial_state=state,
    )

    # The tool invocation was recorded.
    invocations = result.tool_invocations
    assert len(invocations) == 1
    invocation = invocations[0]
    assert invocation.tool_name == "request_interpretation_review"
    # SUCCESS finishing status — finish_success was called by the
    # session-aware dispatch branch.
    assert invocation.status.value == "success"

    # A pending interpretation_events row was created.
    events = await sessions_service.list_interpretation_events(session_id, status="pending")
    assert len(events) == 1
    event = events[0]
    assert event.user_term == "cool"
    assert event.llm_draft == "Visually appealing and well-organized."
    assert event.affected_node_id == "rate_node"
    assert event.tool_call_id == "call_42"
    assert event.choice is InterpretationChoice.PENDING
    # Provenance threaded from the compose-loop snapshot.
    assert event.model_identifier == "anthropic/claude-opus-4-7"
    assert event.provider == "anthropic"
    # ``model_version`` was sourced from ``response.model``; with the
    # fixed fake response model this is a deterministic string.
    assert event.model_version == "anthropic/claude-opus-4-7-20260101"
    # The skill hash matches the in-memory cached value from prompts.py.
    from elspeth.web.composer.prompts import PIPELINE_COMPOSER_SKILL_HASH

    assert event.composer_skill_hash == PIPELINE_COMPOSER_SKILL_HASH


@pytest.mark.asyncio
async def test_fresh_session_set_pipeline_then_request_interpretation_review_persists_pending_event(
    tmp_path: Path,
    sessions_service: SessionServiceImpl,
) -> None:
    """A fresh chat session can stage an LLM placeholder, persist the new
    composition state, and request interpretation review on the next model
    turn.

    This is the Phase 5b success-metric path from a first user message:
    no initial ``composition_states`` row exists, the LLM first calls
    ``set_pipeline`` with ``{{interpretation:cool}}`` in
    ``options.prompt_template``, then calls
    ``request_interpretation_review`` after the state-staging tool result.
    The placeholder is a composer-time review token, so authoring
    prevalidation must allow it until the user resolves the pending event.
    """
    composer = _build_composer(tmp_path, sessions_service)
    session_id = uuid4()
    with sessions_service._engine.begin() as conn:
        conn.execute(
            insert(sessions_table).values(
                id=str(session_id),
                user_id="alice",
                auth_provider_type="local",
                title="Fresh interpretation review session",
                created_at=datetime.now(UTC),
                updated_at=datetime.now(UTC),
            )
        )

    llm = _ScriptedLLM(
        [
            _fake_response_with_tool_call(
                tool_call_id="call_set_pipeline",
                tool_name="set_pipeline",
                arguments=_set_pipeline_with_pending_interpretation_args(),
            ),
            _fake_response_with_tool_call(
                tool_call_id="call_review",
                tool_name="request_interpretation_review",
                arguments={
                    "affected_node_id": "rate_node",
                    "kind": "vague_term",
                    "user_term": "cool",
                    "llm_draft": "modern, useful, engaging, and clear for the public.",
                },
            ),
            _fake_text_response("Done — interpretation review is pending."),
        ]
    )

    result = await composer._run_one_turn_for_test(
        llm=llm,
        session_id=str(session_id),
        current_state_id=None,
    )

    invocations = result.tool_invocations
    assert [inv.tool_name for inv in invocations] == [
        "set_pipeline",
        "request_interpretation_review",
    ]
    assert [inv.status.value for inv in invocations] == ["success", "success"]

    events = await sessions_service.list_interpretation_events(session_id, status="pending")
    assert len(events) == 1
    event = events[0]
    assert event.composition_state_id is not None
    assert event.affected_node_id == "rate_node"
    assert event.tool_call_id == "call_review"
    assert event.user_term == "cool"
    assert event.llm_draft == "modern, useful, engaging, and clear for the public."


@pytest.mark.asyncio
async def test_pending_interpretation_placeholder_without_event_forces_review_tool_retry(
    tmp_path: Path,
    sessions_service: SessionServiceImpl,
) -> None:
    """A staged ``{{interpretation:...}}`` token is incomplete until the audit event exists."""

    composer = _build_composer(tmp_path, sessions_service)
    session_id = uuid4()
    with sessions_service._engine.begin() as conn:
        conn.execute(
            insert(sessions_table).values(
                id=str(session_id),
                user_id="alice",
                auth_provider_type="local",
                title="Missing interpretation review repair session",
                created_at=datetime.now(UTC),
                updated_at=datetime.now(UTC),
            )
        )

    llm = _ScriptedLLM(
        [
            _fake_response_with_tool_call(
                tool_call_id="call_set_pipeline",
                tool_name="set_pipeline",
                arguments=_set_pipeline_with_pending_interpretation_args(),
            ),
            _fake_text_response("Done — the pipeline is ready."),
            _fake_response_with_tool_call(
                tool_call_id="call_review_after_repair",
                tool_name="request_interpretation_review",
                arguments={
                    "affected_node_id": "rate_node",
                    "kind": "vague_term",
                    "user_term": "cool",
                    "llm_draft": "modern, useful, engaging, and clear for the public.",
                },
            ),
            _fake_text_response("Done — interpretation review is pending."),
        ]
    )

    result = await composer._run_one_turn_for_test(
        llm=llm,
        session_id=str(session_id),
        current_state_id=None,
        message="create a workflow that rates how cool pages are",
    )

    assert [inv.tool_name for inv in result.tool_invocations] == [
        "set_pipeline",
        "request_interpretation_review",
    ]
    assert [inv.status.value for inv in result.tool_invocations] == ["success", "success"]
    events = await sessions_service.list_interpretation_events(session_id, status="pending")
    assert len(events) == 1
    assert events[0].user_term == "cool"
    assert events[0].affected_node_id == "rate_node"
    assert events[0].tool_call_id == "call_review_after_repair"


@pytest.mark.asyncio
async def test_prompt_template_review_event_does_not_trigger_vague_term_repair(
    tmp_path: Path,
    sessions_service: SessionServiceImpl,
) -> None:
    """Prompt-template reviews are not legacy ``{{interpretation:...}}`` handoffs."""

    composer = _build_composer(tmp_path, sessions_service)
    state = _state_with_prompt_template_review_node()
    session_id, state_id = await _seed_session_and_state(sessions_service, state=state)

    await sessions_service.create_pending_interpretation_event(
        session_id=session_id,
        composition_state_id=state_id,
        affected_node_id="rate_node",
        tool_call_id="call_prompt_review",
        user_term="llm_prompt_template:rate_node",
        kind=InterpretationKind.LLM_PROMPT_TEMPLATE,
        llm_draft="Read {{ row.html }} and return JSON.",
        model_identifier="anthropic/claude-opus-4-7",
        model_version="2026-05-01",
        provider="anthropic",
        composer_skill_hash="a" * 64,
    )

    missing = await composer._missing_pending_interpretation_review_sites(
        state,
        session_id=str(session_id),
    )

    assert missing == ()


@pytest.mark.asyncio
async def test_missing_prompt_template_review_event_forces_review_tool_retry(
    tmp_path: Path,
    sessions_service: SessionServiceImpl,
) -> None:
    """A pending prompt-template requirement is incomplete until its event exists."""

    composer = _build_composer(tmp_path, sessions_service)
    state = _state_with_prompt_template_review_node()
    session_id, _state_id = await _seed_session_and_state(sessions_service, state=state)

    missing = await composer._missing_pending_interpretation_review_sites(
        state,
        session_id=str(session_id),
    )

    assert missing == (("rate_node", "llm_prompt_template:rate_node", InterpretationKind.LLM_PROMPT_TEMPLATE),)


@pytest.mark.asyncio
async def test_missing_invented_source_review_event_forces_review_tool_retry(
    tmp_path: Path,
    sessions_service: SessionServiceImpl,
) -> None:
    """A pending invented-source requirement is incomplete until its event exists."""

    composer = _build_composer(tmp_path, sessions_service)
    state = _state_with_source_review()
    session_id, _state_id = await _seed_session_and_state(sessions_service, state=state)

    missing = await composer._missing_pending_interpretation_review_sites(
        state,
        session_id=str(session_id),
    )

    assert missing == (("source", "inline_source_url_list", InterpretationKind.INVENTED_SOURCE),)


@pytest.mark.asyncio
async def test_missing_pipeline_decision_review_event_forces_review_tool_retry(
    tmp_path: Path,
    sessions_service: SessionServiceImpl,
) -> None:
    """A pending pipeline-decision requirement is incomplete until its event exists."""

    composer = _build_composer(tmp_path, sessions_service)
    state = _state_with_pipeline_decision_review()
    session_id, _state_id = await _seed_session_and_state(sessions_service, state=state)

    missing = await composer._missing_pending_interpretation_review_sites(
        state,
        session_id=str(session_id),
    )

    assert missing == (("drop_raw_html", "drop_raw_html_fields", InterpretationKind.PIPELINE_DECISION),)


def test_pending_interpretation_repair_message_requires_one_call_per_site() -> None:
    message = _pending_interpretation_review_repair_message(
        (
            ("source", "inline_source_url_list", InterpretationKind.INVENTED_SOURCE),
            ("identify_colours", "llm_prompt_template:identify_colours", InterpretationKind.LLM_PROMPT_TEMPLATE),
            ("drop_html", "drop_raw_html_fields", InterpretationKind.PIPELINE_DECISION),
        ),
        next_turn=1,
    )

    assert "one request_interpretation_review tool call per listed handoff" in message
    assert "in this same assistant turn before stopping" in message


@pytest.mark.asyncio
async def test_unreviewed_raw_html_cleanup_forces_pipeline_decision_staging_retry(
    tmp_path: Path,
    sessions_service: SessionServiceImpl,
) -> None:
    """A field_mapper that drops web-scrape raw fields needs a review site."""

    composer = _build_composer(tmp_path, sessions_service)
    state = _state_with_unreviewed_raw_html_cleanup()
    session_id, _state_id = await _seed_session_and_state(sessions_service, state=state)

    missing = await composer._missing_pending_interpretation_review_sites(
        state,
        session_id=str(session_id),
    )

    assert missing == (("drop_raw_html", "drop_raw_html_fields", InterpretationKind.PIPELINE_DECISION),)


@pytest.mark.asyncio
async def test_pending_interpretation_event_with_duplicate_placeholder_forces_prompt_repair(
    tmp_path: Path,
    sessions_service: SessionServiceImpl,
) -> None:
    """A pending event is still broken if the live prompt has two replacement sites."""

    composer = _build_composer(tmp_path, sessions_service)
    session_id = uuid4()
    with sessions_service._engine.begin() as conn:
        conn.execute(
            insert(sessions_table).values(
                id=str(session_id),
                user_id="alice",
                auth_provider_type="local",
                title="Duplicate placeholder repair session",
                created_at=datetime.now(UTC),
                updated_at=datetime.now(UTC),
            )
        )

    duplicate_prompt = (
        "Rate how {{interpretation:cool}} this row is. Use this meaning for {{interpretation:cool}} after the user approves it."
    )
    repaired_prompt = "Rate how {{interpretation:cool}} this row is after the user approves the pending definition."
    llm = _ScriptedLLM(
        [
            _fake_response_with_tool_call(
                tool_call_id="call_set_pipeline",
                tool_name="set_pipeline",
                arguments=_set_pipeline_with_pending_interpretation_args(),
            ),
            _fake_response_with_tool_call(
                tool_call_id="call_review",
                tool_name="request_interpretation_review",
                arguments={
                    "affected_node_id": "rate_node",
                    "kind": "vague_term",
                    "user_term": "cool",
                    "llm_draft": "modern, useful, engaging, and clear for the public.",
                },
            ),
            _fake_response_with_tool_call(
                tool_call_id="call_duplicate_placeholder",
                tool_name="patch_node_options",
                arguments={
                    "node_id": "rate_node",
                    "patch": {"prompt_template": duplicate_prompt},
                },
            ),
            _fake_text_response("Done — interpretation review is pending."),
            _fake_response_with_tool_call(
                tool_call_id="call_repair_duplicate_placeholder",
                tool_name="patch_node_options",
                arguments={
                    "node_id": "rate_node",
                    "patch": {"prompt_template": repaired_prompt},
                },
            ),
            _fake_text_response("Done — interpretation review is still pending."),
        ]
    )

    result = await composer._run_one_turn_for_test(
        llm=llm,
        session_id=str(session_id),
        current_state_id=None,
        message="create a workflow that rates how cool pages are",
    )

    assert [inv.tool_name for inv in result.tool_invocations] == [
        "set_pipeline",
        "request_interpretation_review",
        "patch_node_options",
        "patch_node_options",
    ]
    events = await sessions_service.list_interpretation_events(session_id, status="pending")
    assert len(events) == 1
    state_record = await sessions_service.get_current_state(session_id)
    assert state_record is not None
    [rate_node] = [node for node in state_record.nodes if node["id"] == "rate_node"]
    assert rate_node["options"]["prompt_template"] == repaired_prompt


@pytest.mark.asyncio
async def test_missing_state_interpretation_review_arg_error_forces_staging_retry(
    tmp_path: Path,
    sessions_service: SessionServiceImpl,
) -> None:
    """A build-style request must not stop after calling review too early.

    The live model sometimes tries ``request_interpretation_review`` before
    any composition state exists, receives the missing-current-state ARG_ERROR,
    then emits prose saying it needs to stage the workflow first. For a
    build-style user request, that prose is not a valid stopping point: the
    compose loop should inject a repair instruction and continue so the model
    can call ``set_pipeline`` and then retry the review tool.
    """
    composer = _build_composer(tmp_path, sessions_service)
    session_id = uuid4()
    with sessions_service._engine.begin() as conn:
        conn.execute(
            insert(sessions_table).values(
                id=str(session_id),
                user_id="alice",
                auth_provider_type="local",
                title="Early review repair session",
                created_at=datetime.now(UTC),
                updated_at=datetime.now(UTC),
            )
        )

    llm = _ScriptedLLM(
        [
            _fake_response_with_tool_call(
                tool_call_id="call_too_soon",
                tool_name="request_interpretation_review",
                arguments={
                    "affected_node_id": "rate_node",
                    "kind": "vague_term",
                    "user_term": "cool",
                    "llm_draft": "modern and useful.",
                },
            ),
            _fake_text_response("I need to stage the LLM node first."),
            _fake_response_with_tool_call(
                tool_call_id="call_set_pipeline",
                tool_name="set_pipeline",
                arguments=_set_pipeline_with_pending_interpretation_args(),
            ),
            _fake_response_with_tool_call(
                tool_call_id="call_review",
                tool_name="request_interpretation_review",
                arguments={
                    "affected_node_id": "rate_node",
                    "kind": "vague_term",
                    "user_term": "cool",
                    "llm_draft": "modern and useful.",
                },
            ),
            _fake_text_response("Done — interpretation review is pending."),
        ]
    )

    result = await composer._run_one_turn_for_test(
        llm=llm,
        session_id=str(session_id),
        current_state_id=None,
        message="create a workflow that rates how cool pages are",
    )

    assert [inv.tool_name for inv in result.tool_invocations] == [
        "request_interpretation_review",
        "set_pipeline",
        "request_interpretation_review",
    ]
    assert [inv.status.value for inv in result.tool_invocations] == [
        "arg_error",
        "success",
        "success",
    ]
    events = await sessions_service.list_interpretation_events(session_id, status="pending")
    assert len(events) == 1
    assert events[0].tool_call_id == "call_review"


@pytest.mark.asyncio
async def test_request_interpretation_review_without_persisted_state_returns_arg_error(
    tmp_path: Path,
    sessions_service: SessionServiceImpl,
) -> None:
    """An over-eager request before any state row exists is LLM-correctable,
    not a server crash.

    Fresh sessions legitimately start with ``current_state_id=None``. If the
    model calls ``request_interpretation_review`` before a successful
    state-staging tool has been persisted, the compose loop must return an
    ARG_ERROR instructing it to stage state first instead of raising
    ``RuntimeError``.
    """
    composer = _build_composer(tmp_path, sessions_service)
    session_id = uuid4()
    with sessions_service._engine.begin() as conn:
        conn.execute(
            insert(sessions_table).values(
                id=str(session_id),
                user_id="alice",
                auth_provider_type="local",
                title="Early interpretation review session",
                created_at=datetime.now(UTC),
                updated_at=datetime.now(UTC),
            )
        )

    llm = _ScriptedLLM(
        [
            _fake_response_with_tool_call(
                tool_call_id="call_too_soon",
                tool_name="request_interpretation_review",
                arguments={
                    "affected_node_id": "rate_node",
                    "kind": "vague_term",
                    "user_term": "cool",
                    "llm_draft": "modern and useful.",
                },
            ),
            _fake_text_response("I will stage the LLM node first."),
        ]
    )

    result = await composer._run_one_turn_for_test(
        llm=llm,
        session_id=str(session_id),
        current_state_id=None,
    )

    invocations = result.tool_invocations
    assert len(invocations) == 1
    invocation = invocations[0]
    assert invocation.tool_name == "request_interpretation_review"
    assert invocation.status.value == "arg_error"
    assert invocation.error_class == "ToolArgumentError"
    assert "composition_state_id" in (invocation.error_message or "")

    events = await sessions_service.list_interpretation_events(session_id, status="all")
    assert events == []


# ---------------------------------------------------------------------------
# Test 2 — F-5a startup assert crashes on mid-process disk drift
# ---------------------------------------------------------------------------


def test_f5a_startup_assert_crashes_on_skill_hash_drift(
    tmp_path: Path,
    sessions_service: SessionServiceImpl,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If the on-disk composer skill file changes after the LRU cache
    populates, ``ComposerServiceImpl.__init__`` MUST refuse to start with
    an operator-actionable RuntimeError.

    LRU-cache trap (see advisor note): the cache MUST be primed with
    pristine content BEFORE we mock ``Path.read_text``, otherwise the
    mock seeds the cache with the tampered text and the assert finds
    no mismatch. ``prompts.py`` import-time populates the cache via
    ``load_skill_with_hash`` at module load, so by the time this test
    runs the cache already holds the real on-disk hash.
    """
    # Prime the cache (idempotent — prompts.py already loaded it at import).
    from elspeth.web.composer.skills import load_skill_with_hash

    pristine_text, pristine_hash = load_skill_with_hash("pipeline_composer")
    assert pristine_text  # sanity — non-empty
    assert len(pristine_hash) == 64  # sha256 hex

    # Now mock Path.read_text on the skill file so the assert_skill_hash_unchanged_on_disk
    # call inside __init__ reads tampered content. We target only reads of the
    # specific skill file path; everything else passes through.
    from elspeth.web.composer.skills import _SKILLS_DIR

    target_path = _SKILLS_DIR / "pipeline_composer.md"
    tampered_text = pristine_text + "\n# TAMPERED MARKER\n"
    tampered_hash = hashlib.sha256(tampered_text.encode("utf-8")).hexdigest()
    assert tampered_hash != pristine_hash

    real_read_text = Path.read_text

    def _patched_read_text(self: Path, *args: Any, **kwargs: Any) -> str:
        if self == target_path:
            return tampered_text
        return real_read_text(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", _patched_read_text)

    # Build the service — the F-5a assertion in __init__ should fire.
    with pytest.raises(RuntimeError) as excinfo:
        _build_composer(tmp_path, sessions_service)
    msg = str(excinfo.value)
    assert "skill hash mismatch" in msg.lower() or "skill hash drift" in msg.lower()
    # Operator-actionable: the message must mention service restart so an
    # on-call engineer knows what to do without reading the source.
    assert "restart" in msg.lower()
    # The cached and on-disk hashes are both surfaced so the operator
    # can diff against the deployed file.
    assert pristine_hash in msg
    assert tampered_hash in msg


# ---------------------------------------------------------------------------
# Test 3 — F-5c skill_markdown_history INSERT OR IGNORE on first use
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_f5c_skill_markdown_history_upsert_idempotent(
    tmp_path: Path,
    sessions_service: SessionServiceImpl,
) -> None:
    """On the first compose-loop entry of a service instance, the composer
    skill markdown is upserted into ``skill_markdown_history`` keyed by
    SHA-256. A second compose() call on the same instance does NOT
    duplicate the row (INSERT OR IGNORE semantics)."""
    composer = _build_composer(tmp_path, sessions_service)
    session_id, state_id = await _seed_session_and_state(sessions_service)
    state = _state_with_llm_node()

    # Pre-condition: empty history table.
    with sessions_service._engine.connect() as conn:
        rows = conn.execute(select(skill_markdown_history_table)).fetchall()
    assert rows == []

    # First compose() — assistant terminates immediately, exercising
    # only the F-5c upsert site at the top of _compose_loop.
    llm = _ScriptedLLM([_fake_text_response("Hello.")])
    await composer._run_one_turn_for_test(
        llm=llm,
        session_id=str(session_id),
        current_state_id=str(state_id),
        initial_state=state,
    )

    with sessions_service._engine.connect() as conn:
        rows = conn.execute(select(skill_markdown_history_table)).fetchall()
    assert len(rows) == 1
    row = rows[0]
    from elspeth.web.composer.prompts import PIPELINE_COMPOSER_SKILL_HASH

    assert row.hash == PIPELINE_COMPOSER_SKILL_HASH
    assert row.filename == "pipeline_composer.md"
    assert row.content.startswith("#") or row.content  # non-empty markdown
    assert len(row.content) > 100  # the real skill is ~24KB

    # Second compose() — flag should now be set; upsert is skipped and
    # the row count stays at 1. Even if it weren't skipped, INSERT OR
    # IGNORE would prevent duplication.
    llm2 = _ScriptedLLM([_fake_text_response("Hello again.")])
    await composer._run_one_turn_for_test(
        llm=llm2,
        session_id=str(session_id),
        current_state_id=str(state_id),
        initial_state=state,
    )

    with sessions_service._engine.connect() as conn:
        rows = conn.execute(select(skill_markdown_history_table)).fetchall()
    assert len(rows) == 1  # no duplicate


# ---------------------------------------------------------------------------
# Test 4 — F-6 rate-cap branch (telemetry + AUTO_INTERPRETED_NO_SURFACES row)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_f6_rate_cap_branch_emits_telemetry_and_writes_audit_row(
    tmp_path: Path,
    sessions_service: SessionServiceImpl,
) -> None:
    """When the per-term rate cap fires inside the compose loop, the
    dispatcher MUST (in this order):

    1. Emit the F-15 ``interpretation_rate_cap_exceeded`` telemetry signal
       with ``cap_type='per_term'`` and ``session_id`` (NO user_term —
       PII-clean).
    2. Write the AUTO_INTERPRETED_NO_SURFACES audit row via
       ``record_auto_interpreted_no_surfaces_event`` (NULL surface
       fields; populated LLM provenance fields; choice=opted_out;
       interpretation_source=auto_interpreted_no_surfaces).
    3. Return the ARG_ERROR to the LLM so the skill's fallback nudge
       triggers (bake the interpretation directly into the prompt
       template).
    """
    composer = _build_composer(tmp_path, sessions_service)
    # Swap in a fresh telemetry container so we can inspect the
    # interpretation_rate_cap_exceeded_total counter cleanly.
    telemetry = build_sessions_telemetry()
    composer._telemetry = telemetry  # type: ignore[attr-defined]

    # Multi-node state — the rate cap is keyed on ``user_term``, NOT
    # ``affected_node_id``. Three distinct sites all reference the same
    # vague term ``"cool"``, accumulating the per-term count without
    # triggering the dedup gate (which keys on the tuple
    # ``(kind, user_term, affected_node_id)``).
    state = CompositionState(
        source=None,
        nodes=tuple(_llm_node_spec_with_id(f"rate_node_{i}", term="cool") for i in range(4)),
        edges=(),
        outputs=(),
        metadata=PipelineMetadata(),
        version=1,
    )
    session_id, state_id = await _seed_session_and_state(
        sessions_service,
        state=state,
    )

    # Saturate the per-term cap by issuing three successful compose-loop
    # turns with the same ``user_term`` across DISTINCT sites. Each turn =
    # one fake response that calls request_interpretation_review followed
    # by a terminating text response. The per-term cap default is 3 — the
    # 4th call below trips the cap.
    for i in range(3):
        llm = _ScriptedLLM(
            [
                _fake_response_with_tool_call(
                    tool_call_id=f"call_{i}",
                    tool_name="request_interpretation_review",
                    arguments={
                        "affected_node_id": f"rate_node_{i}",
                        "kind": "vague_term",
                        "user_term": "cool",
                        "llm_draft": f"Visually appealing {i}.",
                    },
                ),
                _fake_text_response(f"Surfaced #{i}."),
            ]
        )
        await composer._run_one_turn_for_test(
            llm=llm,
            session_id=str(session_id),
            current_state_id=str(state_id),
            initial_state=state,
        )

    # Sanity: three PENDING rows exist before the cap is hit.
    pending = await sessions_service.list_interpretation_events(session_id, status="pending")
    assert len(pending) == 3
    # No telemetry yet — no caps breached.
    assert observed_value(telemetry.interpretation_rate_cap_exceeded_total) == 0

    # The fourth call trips the per-term cap. Uses the 4th distinct site
    # so the dedup gate does not intercept — the per-term count (3 prior
    # rows with user_term="cool") is what fires.
    llm = _ScriptedLLM(
        [
            _fake_response_with_tool_call(
                tool_call_id="call_capped",
                tool_name="request_interpretation_review",
                arguments={
                    "affected_node_id": "rate_node_3",
                    "kind": "vague_term",
                    "user_term": "cool",
                    "llm_draft": "Visually appealing 3.",
                },
            ),
            _fake_text_response("Falling back to baked interpretation."),
        ]
    )
    result = await composer._run_one_turn_for_test(
        llm=llm,
        session_id=str(session_id),
        current_state_id=str(state_id),
        initial_state=state,
    )

    # 1. F-15 telemetry signal was emitted with the right attributes.
    assert observed_value(telemetry.interpretation_rate_cap_exceeded_total) == 1
    # ``calls`` is the test-only attribute on _FakeCounter.
    counter_calls = telemetry.interpretation_rate_cap_exceeded_total.calls  # type: ignore[attr-defined]
    assert len(counter_calls) == 1
    _amount, attrs, _ctx = counter_calls[0]
    assert attrs == {
        "cap_type": "per_term",
        "session_id": str(session_id),
    }
    # Explicit PII guard: no user_term, no llm_draft.
    assert "user_term" not in attrs
    assert "llm_draft" not in attrs

    # 2. The AUTO_INTERPRETED_NO_SURFACES audit row exists.
    all_events = await sessions_service.list_interpretation_events(session_id, status="all")
    auto_rows = [e for e in all_events if e.interpretation_source is InterpretationSource.AUTO_INTERPRETED_NO_SURFACES]
    assert len(auto_rows) == 1
    auto = auto_rows[0]
    # Row shape per ck_interpretation_events_no_surfaces_shape: surface
    # fields NULL, LLM provenance fields populated, choice = OPTED_OUT.
    assert auto.composition_state_id is None
    assert auto.affected_node_id is None
    assert auto.tool_call_id is None
    assert auto.user_term is None
    assert auto.llm_draft is None
    assert auto.kind is InterpretationKind.VAGUE_TERM
    assert auto.choice is InterpretationChoice.OPTED_OUT
    assert auto.model_identifier == "anthropic/claude-opus-4-7"
    assert auto.provider == "anthropic"
    assert auto.composer_skill_hash is not None
    assert len(auto.composer_skill_hash) == 64

    # 3. The LLM-facing tool message carries the ARG_ERROR shape (the
    # compose loop's ``_arg_error_payload``). The fourth turn's tool
    # invocation was recorded with the ARG_ERROR status.
    invocations = result.tool_invocations
    arg_error_invocations = [inv for inv in invocations if inv.status.value == "arg_error"]
    assert len(arg_error_invocations) == 1
    arg_error = arg_error_invocations[0]
    assert arg_error.tool_name == "request_interpretation_review"
    assert arg_error.error_class == "ToolArgumentError"

    # The per-term row count remains at 3 — the rejected call did NOT
    # add a pending row; only the AUTO_INTERPRETED_NO_SURFACES row was
    # written.
    final_pending = await sessions_service.list_interpretation_events(session_id, status="pending")
    assert len(final_pending) == 3


@pytest.mark.asyncio
async def test_rate_capped_invalid_review_site_does_not_write_no_surfaces_row(
    tmp_path: Path,
    sessions_service: SessionServiceImpl,
) -> None:
    """Rate-cap audit rows require a valid pending review site first."""

    composer = _build_composer(tmp_path, sessions_service)
    telemetry = build_sessions_telemetry()
    composer._telemetry = telemetry  # type: ignore[attr-defined]

    session_id, state_id = await _seed_session_and_state(sessions_service)
    state = _state_with_llm_node()

    for i in range(3):
        llm = _ScriptedLLM(
            [
                _fake_response_with_tool_call(
                    tool_call_id=f"call_valid_{i}",
                    tool_name="request_interpretation_review",
                    arguments={
                        "affected_node_id": "rate_node",
                        "kind": "vague_term",
                        "user_term": "cool",
                        "llm_draft": f"Draft {i}",
                    },
                ),
                _fake_text_response(f"Surfaced #{i}."),
            ]
        )
        await composer._run_one_turn_for_test(
            llm=llm,
            session_id=str(session_id),
            current_state_id=str(state_id),
            initial_state=state,
        )

    llm = _ScriptedLLM(
        [
            _fake_response_with_tool_call(
                tool_call_id="call_invalid_capped",
                tool_name="request_interpretation_review",
                arguments={
                    "affected_node_id": "rate_node",
                    "kind": "llm_prompt_template",
                    "user_term": "cool",
                    "llm_draft": "Draft 4",
                },
            ),
            _fake_text_response("Invalid review target."),
        ]
    )

    result = await composer._run_one_turn_for_test(
        llm=llm,
        session_id=str(session_id),
        current_state_id=str(state_id),
        initial_state=state,
    )

    assert observed_value(telemetry.interpretation_rate_cap_exceeded_total) == 0
    all_events = await sessions_service.list_interpretation_events(session_id, status="all")
    auto_rows = [event for event in all_events if event.interpretation_source is InterpretationSource.AUTO_INTERPRETED_NO_SURFACES]
    assert auto_rows == []

    arg_error_invocations = [inv for inv in result.tool_invocations if inv.status.value == "arg_error"]
    assert len(arg_error_invocations) == 1
    assert arg_error_invocations[0].tool_name == "request_interpretation_review"
    assert arg_error_invocations[0].error_class == "ToolArgumentError"


# ---------------------------------------------------------------------------
# Test 5 — ToolArgumentError.code parity with telemetry cap_type table
# ---------------------------------------------------------------------------


def test_rate_cap_code_constants_round_trip_to_telemetry_cap_type() -> None:
    """Every ``RATE_CAP_*_CODE`` constant MUST appear in
    ``RATE_CAP_CODE_TO_TELEMETRY_CAP_TYPE`` so the dispatcher's
    ``exc.code`` discriminant can always be translated into a telemetry
    ``cap_type`` attribute. Catches "added a new code, forgot to extend
    the mapping" regressions at the module-load boundary."""
    assert RATE_CAP_PER_TERM_CODE in RATE_CAP_CODE_TO_TELEMETRY_CAP_TYPE
    assert RATE_CAP_CODE_TO_TELEMETRY_CAP_TYPE[RATE_CAP_PER_TERM_CODE] == "per_term"
    assert RATE_CAP_PER_SESSION_DAY_CODE in RATE_CAP_CODE_TO_TELEMETRY_CAP_TYPE
    assert RATE_CAP_CODE_TO_TELEMETRY_CAP_TYPE[RATE_CAP_PER_SESSION_DAY_CODE] == "per_session_day"


def test_tool_argument_error_code_field_is_frozen_after_construction() -> None:
    """The new ``code`` field is part of ``_FROZEN_ATTRS`` so a later
    layer cannot rewrite the dispatcher discriminant after the exception
    has been constructed at the trust boundary."""
    exc = ToolArgumentError(
        argument="user_term",
        expected="at most 3",
        actual_type="4",
        code=RATE_CAP_PER_TERM_CODE,
    )
    assert exc.code == RATE_CAP_PER_TERM_CODE
    with pytest.raises(AttributeError):
        exc.code = "TAMPERED"  # type: ignore[misc]


def test_tool_argument_error_code_defaults_to_none() -> None:
    """Existing call sites do not pass ``code``; the default is ``None``
    so the dispatcher's rate-cap discriminant path is opt-in."""
    exc = ToolArgumentError(
        argument="any_arg",
        expected="any expected description",
        actual_type="any actual type",
    )
    assert exc.code is None
