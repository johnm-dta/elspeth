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
from dataclasses import dataclass
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
from elspeth.web.composer.guided.errors import InvariantError
from elspeth.web.composer.protocol import ToolArgumentError
from elspeth.web.composer.service import (
    AdvisorCheckpointVerdict,
    ComposerAvailability,
    ComposerServiceImpl,
    _pending_interpretation_review_repair_message,
)
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
from tests.unit.web.composer._helpers import _stub_advisor_end_gate_clean  # noqa: F401  (autouse end-gate CLEAN stub)

# ---------------------------------------------------------------------------
# Lightweight fake LLM that emits a single request_interpretation_review call.
# ---------------------------------------------------------------------------


def _fake_response_with_tool_call(
    *,
    tool_call_id: str,
    tool_name: str,
    arguments: dict[str, Any],
    response_model: str = "anthropic/claude-opus-4-7-20260101",
    content: str | None = None,
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
        content=content,
    )


def _execution_ready() -> ValidationReadiness:
    return ValidationReadiness(authoring_valid=True, execution_ready=True, completion_ready=True, blockers=[])


def _fake_response_with_tool_calls(
    *,
    tool_calls: list[dict[str, Any]],
    response_model: str = "anthropic/claude-opus-4-7-20260101",
    content: str | None = None,
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
                        content=content,
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


@dataclass(frozen=True)
class _RecordedAsyncCall:
    args: tuple[Any, ...]
    kwargs: dict[str, Any]


class _AdvisorCheckpointFake:
    """Async advisor test double that records awaits and returns one verdict."""

    def __init__(self, verdict: AdvisorCheckpointVerdict) -> None:
        self._verdict = verdict
        self.await_count = 0
        self.await_args_list: list[_RecordedAsyncCall] = []

    async def __call__(self, *args: Any, **kwargs: Any) -> AdvisorCheckpointVerdict:
        self.await_count += 1
        self.await_args_list.append(_RecordedAsyncCall(args=args, kwargs=kwargs))
        return self._verdict


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


def _build_composer(
    tmp_path: Path,
    sessions_service: SessionServiceImpl,
    *,
    max_composition_turns: int = 15,
) -> ComposerServiceImpl:
    from unittest.mock import MagicMock

    from elspeth.web.catalog.protocol import CatalogService
    from elspeth.web.config import WebSettings

    catalog = MagicMock(spec=CatalogService)
    catalog.list_sources.return_value = []
    catalog.list_transforms.return_value = []
    catalog.list_sinks.return_value = []
    settings = WebSettings(
        data_dir=tmp_path,
        composer_max_composition_turns=max_composition_turns,
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


def _set_pipeline_clean_llm_node_args() -> dict[str, Any]:
    """A ``set_pipeline`` whose LLM node has a CLEAN prompt_template (no bare
    ``{{interpretation:...}}`` token). The node still auto-stages an
    ``llm_prompt_template`` review requirement, but carries no orphan token —
    so the ONLY pending review at finalization is the backend-surfaced PT one.
    """
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
                    "prompt_template": "Summarise {{ row.text }} in one sentence.",
                    "schema": {"mode": "observed"},
                },
            }
        ],
        "edges": [],
        "outputs": [],
    }


def _set_pipeline_clean_llm_node_no_model_args() -> dict[str, Any]:
    """Like ``_set_pipeline_clean_llm_node_args`` but the LLM node carries NO
    ``model`` (nor provider/api_key). With no ``options.model``, no
    ``llm_model_choice`` review site is enumerated
    (``interpretation_state._missing_model_choice_review_sites`` short-circuits on
    an empty model), so the ONLY orphaned pre-check site at finalization is the
    auto-surfaceable ``llm_prompt_template`` one — the exact masking condition the
    END advisor gate must see through. The clean prompt_template (no bare token)
    still auto-stages the PT review requirement (it keys off prompt_template, not
    model)."""
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
                    "prompt_template": "Summarise {{ row.text }} in one sentence.",
                    "schema": {"mode": "observed"},
                },
            }
        ],
        "edges": [],
        "outputs": [],
    }


def test_composer_runtime_preflight_uses_backend_readiness_contract(
    tmp_path: Path,
    sessions_service: SessionServiceImpl,
) -> None:
    """Composer preflight relies on typed readiness, not placeholder masking."""

    composer = _build_composer(tmp_path, sessions_service)
    state = _state_with_llm_node()
    expected = ValidationResult(is_valid=True, checks=[], errors=[], readiness=_execution_ready())

    with patch("elspeth.web.composer.service.validate_pipeline", return_value=expected) as validate:
        result = composer._runtime_preflight(state, user_id="alice", session_id=None)

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
    # The model staged the vague_term review via the tool; the backend also
    # auto-surfaces the node's llm_prompt_template review at finalization
    # (elspeth-e51216d305) as an independent review event.
    vague_events = [e for e in events if e.kind is InterpretationKind.VAGUE_TERM]
    assert len(vague_events) == 1
    event = vague_events[0]
    assert event.composition_state_id is not None
    assert event.affected_node_id == "rate_node"
    assert event.tool_call_id == "call_review"
    assert event.user_term == "cool"
    assert event.llm_draft == "modern, useful, engaging, and clear for the public."
    pt_events = [e for e in events if e.kind is InterpretationKind.LLM_PROMPT_TEMPLATE]
    assert len(pt_events) == 1
    assert pt_events[0].affected_node_id == "rate_node"
    assert pt_events[0].tool_call_id.startswith("backend_auto_surface:")


@pytest.mark.asyncio
async def test_successful_interpretation_review_returns_user_handoff_without_extra_model_turns(
    tmp_path: Path,
    sessions_service: SessionServiceImpl,
) -> None:
    """Once a review is pending, the compose turn should hand off to the user.

    Regression for elspeth-e6ff1b8c13: the freeform model successfully staged
    interpretation review(s), but the loop treated that as an ordinary tool
    result, asked the model for another turn, and the model kept re-surfacing
    reviews until the request hit the wall-clock timeout. A pending review is
    already a user-action boundary; the loop should return a recoverable
    ComposerResult before consuming another LLM/tool turn.
    """

    composer = _build_composer(tmp_path, sessions_service)
    session_id = uuid4()
    with sessions_service._engine.begin() as conn:
        conn.execute(
            insert(sessions_table).values(
                id=str(session_id),
                user_id="alice",
                auth_provider_type="local",
                title="Review handoff terminates compose turn",
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
                content="Surfacing the review card now.",
                arguments={
                    "affected_node_id": "rate_node",
                    "kind": "vague_term",
                    "user_term": "cool",
                    "llm_draft": "modern, useful, engaging, and clear for the public.",
                },
            ),
            _fake_response_with_tool_call(
                tool_call_id="call_duplicate_review",
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
    assert result.assistant_message.startswith("Surfacing the review card now.")
    assert "Interpretation review cards are ready" in result.assistant_message
    assert result.raw_assistant_content == "Surfacing the review card now."
    assert "review" in result.assistant_message.lower()
    events = await sessions_service.list_interpretation_events(session_id, status="pending")
    vague_events = [e for e in events if e.kind is InterpretationKind.VAGUE_TERM]
    assert len(vague_events) == 1
    assert vague_events[0].tool_call_id == "call_review"
    pt_events = [e for e in events if e.kind is InterpretationKind.LLM_PROMPT_TEMPLATE]
    assert len(pt_events) == 1
    assert pt_events[0].tool_call_id.startswith("backend_auto_surface:")


@pytest.mark.asyncio
async def test_interpretation_review_handoff_requires_clean_terminal_review_suffix(
    tmp_path: Path,
    sessions_service: SessionServiceImpl,
) -> None:
    """A review call before a later failed tool call is not enough to stop the loop."""

    composer = _build_composer(tmp_path, sessions_service)
    state = _state_with_llm_node()
    session_id, state_id = await _seed_session_and_state(sessions_service, state=state)

    llm = _ScriptedLLM(
        [
            _fake_response_with_tool_calls(
                tool_calls=[
                    {
                        "id": "call_review",
                        "name": "request_interpretation_review",
                        "arguments": {
                            "affected_node_id": "rate_node",
                            "kind": "vague_term",
                            "user_term": "cool",
                            "llm_draft": "modern, useful, engaging, and clear for the public.",
                        },
                    },
                    {
                        "id": "call_after_review_arg_error",
                        "name": "set_pipeline",
                        "arguments": {"nodes": []},
                    },
                ],
                content="I am staging a review and then trying an invalid change.",
            ),
            _fake_text_response("Final after later tool failure."),
        ]
    )

    result = await composer._run_one_turn_for_test(
        llm=llm,
        session_id=str(session_id),
        current_state_id=str(state_id),
        initial_state=state,
        message="update the pipeline and stage the interpretation review",
    )

    assert [inv.tool_name for inv in result.tool_invocations] == [
        "request_interpretation_review",
        "set_pipeline",
    ]
    assert [inv.status.value for inv in result.tool_invocations] == ["success", "arg_error"]
    assert "Final after later tool failure." in result.assistant_message
    assert "Interpretation review cards are ready" not in result.assistant_message


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
    # The model staged the vague_term "cool" review via the tool; the backend
    # additionally auto-surfaces the node's llm_prompt_template review at
    # finalization (elspeth-e51216d305) as an independent review event.
    vague_events = [e for e in events if e.kind is InterpretationKind.VAGUE_TERM]
    assert len(vague_events) == 1
    assert vague_events[0].user_term == "cool"
    assert vague_events[0].affected_node_id == "rate_node"
    assert vague_events[0].tool_call_id == "call_review_after_repair"
    pt_events = [e for e in events if e.kind is InterpretationKind.LLM_PROMPT_TEMPLATE]
    assert len(pt_events) == 1
    assert pt_events[0].affected_node_id == "rate_node"
    assert pt_events[0].tool_call_id.startswith("backend_auto_surface:")


@pytest.mark.asyncio
async def test_orphaned_interpretation_placeholder_fails_turn_closed_after_repair_budget(
    tmp_path: Path,
    sessions_service: SessionServiceImpl,
) -> None:
    """An orphaned ``{{interpretation:...}}`` token that survives the repair budget fails closed.

    Regression for elspeth-01832796f4 (Fix A, Option 1). The composer writes a
    bare ``{{interpretation:cool}}`` token via ``set_pipeline`` and then NEVER
    calls ``request_interpretation_review`` — so the in-loop repair (which is
    capped at ``_MAX_REPAIR_TURNS``) keeps firing, the model keeps emitting
    plain text, and the budget is exhausted with the placeholder still
    unresolvable (no pending event exists for it).

    Before this fix the no-tool-calls finalizer would finalize the turn as a
    "success" once the budget ran out — the runtime preflight's
    ``InterpretationReviewPending`` shape is indistinguishable between a
    resolvable two-step handoff and an orphan, so the UI would enable run and
    the backend would only reject at ``materialize_state_for_execution`` with
    ``UnresolvedInterpretationPlaceholderError``. The fail-closed gate now
    surfaces a turn-level blocking ``runtime_preflight`` (every readiness axis
    False, distinct ``interpretation_review_orphaned`` code) so the UI never
    enables run on an orphan — making a tutorial run identical to a regular run.
    """

    composer = _build_composer(tmp_path, sessions_service)
    session_id = uuid4()
    with sessions_service._engine.begin() as conn:
        conn.execute(
            insert(sessions_table).values(
                id=str(session_id),
                user_id="alice",
                auth_provider_type="local",
                title="Orphaned interpretation placeholder session",
                created_at=datetime.now(UTC),
                updated_at=datetime.now(UTC),
            )
        )

    # Turn 1 stages the bare token; every subsequent turn (incl. the
    # forced-repair turns) emits plain text only — the model never stages the
    # review, so the placeholder stays orphaned. The scripted LLM keeps
    # returning a no-tool text response once the list is exhausted.
    llm = _ScriptedLLM(
        [
            _fake_response_with_tool_call(
                tool_call_id="call_set_pipeline",
                tool_name="set_pipeline",
                arguments=_set_pipeline_with_pending_interpretation_args(),
            ),
            _fake_text_response("Done — the pipeline is ready."),
            _fake_text_response("It is ready, you can run it."),
            _fake_text_response("All set."),
            _fake_text_response("All set."),
        ]
    )

    result = await composer._run_one_turn_for_test(
        llm=llm,
        session_id=str(session_id),
        current_state_id=None,
        message="create a workflow that rates how cool pages are",
    )

    # The vague_term "cool" review tool was never called, so no VAGUE_TERM event
    # exists for it — the placeholder is a genuine orphan and still fails closed.
    # The backend DOES auto-surface the node's llm_prompt_template review at
    # finalization (elspeth-e51216d305), but that is an independent review and
    # does not resolve the orphaned vague_term token.
    events = await sessions_service.list_interpretation_events(session_id, status="pending")
    assert all(event.kind is not InterpretationKind.VAGUE_TERM for event in events)
    assert all(
        event.tool_call_id.startswith("backend_auto_surface:") for event in events if event.kind is InterpretationKind.LLM_PROMPT_TEMPLATE
    )

    # The turn fails closed: the runtime preflight is blocking on every
    # readiness axis (NOT the resolvable pending-handoff shape).
    preflight = result.runtime_preflight
    assert preflight is not None
    assert preflight.is_valid is False
    assert preflight.readiness.execution_ready is False
    assert preflight.readiness.completion_ready is False
    assert preflight.readiness.authoring_valid is False
    blocker_codes = {blocker.code for blocker in preflight.readiness.blockers}
    assert blocker_codes == {"interpretation_review_orphaned"}
    assert any("cool" in error.message for error in preflight.errors)


def test_orphaned_interpretation_validation_derives_component_type_per_kind() -> None:
    """The fail-closed orphan result labels component_type per interpretation kind.

    The gate fires for every kind ``_missing_pending_interpretation_review_sites``
    can surface, not just legacy vague_term tokens. An ``INVENTED_SOURCE`` site is
    a source-level handoff (component_id ``"source"``, component_type ``"source"``)
    while every other kind is transform-level. The persisted ValidationError /
    readiness blocker must carry the correct component_type into the audit trail,
    and ``affected_nodes`` must exclude source sites (mirroring the runtime
    preflight's ``InterpretationReviewPending`` handling).
    """
    from elspeth.web.composer.service import _orphaned_interpretation_review_validation

    result = _orphaned_interpretation_review_validation(
        (
            ("source", "inline_source_url_list", InterpretationKind.INVENTED_SOURCE),
            ("rate_node", "cool", InterpretationKind.VAGUE_TERM),
        )
    )

    # Every readiness axis is blocking (not the resolvable-handoff shape).
    assert result.is_valid is False
    assert result.readiness.authoring_valid is False
    assert result.readiness.completion_ready is False
    assert result.readiness.execution_ready is False

    # component_type derived per-site: source for invented_source, transform otherwise.
    error_by_component = {error.component_id: error.component_type for error in result.errors}
    assert error_by_component == {"source": "source", "rate_node": "transform"}
    blocker_by_component = {blocker.component_id: blocker.component_type for blocker in result.readiness.blockers}
    assert blocker_by_component == {"source": "source", "rate_node": "transform"}
    assert {blocker.code for blocker in result.readiness.blockers} == {"interpretation_review_orphaned"}

    # affected_nodes excludes the source site (transform-only, per validation.py).
    assert result.checks[0].affected_nodes == ("rate_node",)


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
async def test_missing_prompt_template_review_event_reported_by_orphan_gate(
    tmp_path: Path,
    sessions_service: SessionServiceImpl,
) -> None:
    """A pending prompt-template requirement with no event is reported as a missing site.

    Backend ownership (elspeth-e51216d305 Case B): the LLM no longer surfaces the
    ``llm_prompt_template`` review via the tool, and Task 2b filters it OUT of the
    repair ask — so a missing prompt-template event does NOT force an LLM tool
    retry. ``_missing_pending_interpretation_review_sites`` nonetheless still
    REPORTS the prompt-template site UNFILTERED (D4): this feeds the fail-closed
    orphan gate at finalization, which stays the backstop if the backend
    auto-surfacing ever no-ops. The backend surfaces the event immediately before
    that gate, so in the normal path the site is cleared (see
    ``test_finalization_auto_surfaces_prompt_template_and_does_not_orphan_block``);
    this test pins that the helper itself does not silently drop the kind.
    """

    composer = _build_composer(tmp_path, sessions_service)
    state = _state_with_prompt_template_review_node()
    session_id, _state_id = await _seed_session_and_state(sessions_service, state=state)

    missing = await composer._missing_pending_interpretation_review_sites(
        state,
        session_id=str(session_id),
    )

    assert missing == (("rate_node", "llm_prompt_template:rate_node", InterpretationKind.LLM_PROMPT_TEMPLATE),)


@pytest.mark.asyncio
async def test_auto_surface_prompt_template_creates_pending_event_idempotently(
    tmp_path: Path,
    sessions_service: SessionServiceImpl,
) -> None:
    """The backend surfaces a pending llm_prompt_template event idempotently.

    A node carrying a pending auto-staged ``llm_prompt_template`` requirement and
    NO pending event gets exactly one backend-surfaced event whose ``llm_draft``
    equals the node's ``options.prompt_template`` and whose ``tool_call_id`` is
    the honest ``backend_auto_surface:`` sentinel (D1). A second call is a no-op.
    """

    composer = _build_composer(tmp_path, sessions_service)
    state = _state_with_prompt_template_review_node()
    session_id, state_id = await _seed_session_and_state(sessions_service, state=state)

    await composer._auto_surface_prompt_template_reviews(
        state,
        session_id=str(session_id),
        current_state_id=str(state_id),
    )

    events = await sessions_service.list_interpretation_events(session_id, status="pending")
    pt = [e for e in events if e.kind is InterpretationKind.LLM_PROMPT_TEMPLATE]
    assert len(pt) == 1
    assert pt[0].affected_node_id == "rate_node"
    assert pt[0].user_term == "llm_prompt_template:rate_node"
    assert pt[0].llm_draft == "Read {{ row.html }} and return JSON."
    assert pt[0].tool_call_id.startswith("backend_auto_surface:")

    # Idempotent: a second call must not create a duplicate.
    await composer._auto_surface_prompt_template_reviews(
        state,
        session_id=str(session_id),
        current_state_id=str(state_id),
    )
    events2 = await sessions_service.list_interpretation_events(session_id, status="pending")
    assert len([e for e in events2 if e.kind is InterpretationKind.LLM_PROMPT_TEMPLATE]) == 1


@pytest.mark.asyncio
async def test_finalization_auto_surfaces_prompt_template_and_does_not_orphan_block(
    tmp_path: Path,
    sessions_service: SessionServiceImpl,
) -> None:
    """Finalization surfaces the PT review and does NOT fail closed on it.

    The model finishes (no tool calls) with a pending auto-staged
    ``llm_prompt_template`` requirement and no pending event. The backend
    surfaces the event against the frozen final skeleton immediately before the
    orphan gate, so the turn finalizes (``action == "return"``) instead of
    fail-closing as an orphan on account of the prompt-template review.
    """

    from elspeth.web.composer.audit import BufferingRecorder

    composer = _build_composer(tmp_path, sessions_service)
    state = _state_with_prompt_template_review_node()
    session_id, state_id = await _seed_session_and_state(sessions_service, state=state)

    class _AssistantMessage:
        content = "Done — the pipeline is ready."

    outcome = await composer._try_terminate_no_tools(
        assistant_message=_AssistantMessage(),
        message="rate how cool the pages are",
        llm_messages=[],
        state=state,
        session_id=str(session_id),
        current_state_id=str(state_id),
        initial_version=1,
        user_id="alice",
        last_runtime_preflight=None,
        runtime_preflight_cache=composer._new_runtime_preflight_cache(),
        session_scope=str(session_id),
        mutation_success_seen=True,
        recorder=BufferingRecorder(),
        progress=None,
        repair_turns_used=0,
        persisted_assistant_message_id=None,
        persisted_tool_call_turn=False,
        advisor_checkpoint_passes_used=0,
    )

    events = await sessions_service.list_interpretation_events(session_id, status="pending")
    assert any(e.kind is InterpretationKind.LLM_PROMPT_TEMPLATE for e in events)
    assert outcome.action == "return"
    # Both the orphan-block and the finalize path return action="return", so
    # assert specifically that the turn did NOT fail closed on a prompt-template
    # orphan — the backend surfacing cleared that site.
    result = outcome.result
    assert result is not None
    preflight = result.runtime_preflight
    assert preflight is None or all(blocker.code != "interpretation_review_orphaned" for blocker in preflight.readiness.blockers)


@pytest.mark.asyncio
async def test_budget_exhaustion_finalize_auto_surfaces_prompt_template(
    tmp_path: Path,
    sessions_service: SessionServiceImpl,
) -> None:
    """HIGH-1: the B-4D-3 budget-exhaustion last-chance finalize ALSO surfaces PT.

    Regression for Task 7 (elspeth-e51216d305). The B-4D-3 budget-exhaustion
    last-chance no-tool finalize in ``_classify_and_budget_turn`` is a SECOND
    no-tool finalize path. Before the fix it called ``_finalize_no_tool_response``
    directly, bypassing both ``_auto_surface_prompt_template_reviews`` and the
    fail-closed orphan gate. With ``composer_max_composition_turns=1`` the single
    ``set_pipeline`` mutation exhausts the composition budget; the bonus call
    returns plain text (no tool calls) and finalizes via that path. The node has a
    CLEAN prompt (no bare token), so the only pending review must be the
    backend-surfaced PT one.
    """

    composer = _build_composer(tmp_path, sessions_service, max_composition_turns=1)
    session_id = uuid4()
    with sessions_service._engine.begin() as conn:
        conn.execute(
            insert(sessions_table).values(
                id=str(session_id),
                user_id="alice",
                auth_provider_type="local",
                title="Budget-exhaustion PT surfacing session",
                created_at=datetime.now(UTC),
                updated_at=datetime.now(UTC),
            )
        )

    # Turn 1: a single mutation (set_pipeline) — composition counter 0 -> 1 =
    # exhausted -> B-4D-3 bonus call. Bonus call returns plain text -> the
    # budget-exhaustion no-tool finalize path runs.
    llm = _ScriptedLLM(
        [
            _fake_response_with_tool_call(
                tool_call_id="call_set_pipeline",
                tool_name="set_pipeline",
                arguments=_set_pipeline_clean_llm_node_args(),
            ),
            _fake_text_response("Done — the pipeline is ready."),
        ]
    )

    await composer._run_one_turn_for_test(
        llm=llm,
        session_id=str(session_id),
        current_state_id=None,
        message="summarise each row",
    )

    events = await sessions_service.list_interpretation_events(session_id, status="pending")
    pt = [e for e in events if e.kind is InterpretationKind.LLM_PROMPT_TEMPLATE]
    assert len(pt) == 1, "budget-exhaustion finalize must auto-surface the PT review"
    assert pt[0].affected_node_id == "rate_node"
    assert pt[0].tool_call_id.startswith("backend_auto_surface:")


@pytest.mark.asyncio
async def test_budget_exhaustion_finalize_fails_closed_on_bare_token_orphan(
    tmp_path: Path,
    sessions_service: SessionServiceImpl,
) -> None:
    """HIGH-1: the budget-exhaustion finalize ALSO runs the fail-closed orphan gate.

    A genuine bare ``{{interpretation:cool}}`` vague-term token with no pending
    event must fail closed on the budget-exhaustion path exactly as it does on the
    ``_try_terminate_no_tools`` path — not finalize runnable-pending with no card.
    """

    composer = _build_composer(tmp_path, sessions_service, max_composition_turns=1)
    session_id = uuid4()
    with sessions_service._engine.begin() as conn:
        conn.execute(
            insert(sessions_table).values(
                id=str(session_id),
                user_id="alice",
                auth_provider_type="local",
                title="Budget-exhaustion orphan session",
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
        ]
    )

    result = await composer._run_one_turn_for_test(
        llm=llm,
        session_id=str(session_id),
        current_state_id=None,
        message="create a workflow that rates how cool pages are",
    )

    # The vague_term "cool" review was never staged: a genuine orphan. The
    # budget-exhaustion finalize must fail closed (NOT finalize runnable).
    preflight = result.runtime_preflight
    assert preflight is not None
    assert preflight.is_valid is False
    assert preflight.readiness.execution_ready is False
    assert {blocker.code for blocker in preflight.readiness.blockers} == {"interpretation_review_orphaned"}


@pytest.mark.asyncio
async def test_auto_surface_re_surfaces_after_prompt_edit_not_bricked(
    tmp_path: Path,
    sessions_service: SessionServiceImpl,
) -> None:
    """HIGH-2: draft-aware dedup re-surfaces PT after a multi-turn prompt edit.

    Turn 1 surfaces event E_A for skeleton A. Turn 2 edits the node prompt to B
    (re-stages the PT requirement; the stale pending event E_A survives). With
    node-id-only dedup the second auto-surface would SKIP the node and never
    create E_B (the review is bricked). Draft-aware dedup skips only when a
    pending PT event's ``llm_draft`` equals the CURRENT prompt, so a fresh,
    resolvable E_B (draft == B) is surfaced.
    """

    composer = _build_composer(tmp_path, sessions_service)
    state_a = _state_with_prompt_template_review_node()  # prompt == "Read {{ row.html }} and return JSON."
    session_id, state_id_a = await _seed_session_and_state(sessions_service, state=state_a)

    await composer._auto_surface_prompt_template_reviews(
        state_a,
        session_id=str(session_id),
        current_state_id=str(state_id_a),
    )
    events_a = await sessions_service.list_interpretation_events(session_id, status="pending")
    pt_a = [e for e in events_a if e.kind is InterpretationKind.LLM_PROMPT_TEMPLATE]
    assert len(pt_a) == 1
    assert pt_a[0].llm_draft == "Read {{ row.html }} and return JSON."

    # Turn 2: edit the prompt to B and re-stage the PT requirement against it.
    prompt_b = "Classify {{ row.html }} and return a label."
    node_a = state_a.nodes[0]
    new_options = dict(node_a.options)
    new_options["prompt_template"] = prompt_b
    new_options[INTERPRETATION_REQUIREMENTS_KEY] = [
        {
            "id": "prompt_template_review",
            "kind": InterpretationKind.LLM_PROMPT_TEMPLATE.value,
            "user_term": "llm_prompt_template:rate_node",
            "status": "pending",
            "draft": prompt_b,
            "event_id": None,
            "accepted_value": None,
            "accepted_artifact_hash": None,
            "resolved_prompt_template_hash": None,
        }
    ]
    node_b = NodeSpec(
        id=node_a.id,
        node_type=node_a.node_type,
        plugin=node_a.plugin,
        input=node_a.input,
        on_success=node_a.on_success,
        on_error=node_a.on_error,
        options=new_options,
        condition=node_a.condition,
        routes=node_a.routes,
        fork_to=node_a.fork_to,
        branches=node_a.branches,
        policy=node_a.policy,
        merge=node_a.merge,
    )
    state_b = CompositionState(
        sources=state_a.sources,
        nodes=(node_b,),
        edges=state_a.edges,
        outputs=state_a.outputs,
        metadata=state_a.metadata,
        version=2,
    )
    state_dict_b = state_b.to_dict()
    record_b = await sessions_service.save_composition_state(
        session_id,
        CompositionStateData(
            nodes=state_dict_b["nodes"],
            sources=state_dict_b["sources"],
            metadata_=state_dict_b["metadata"],
            is_valid=True,
        ),
        provenance="tool_call",
    )

    await composer._auto_surface_prompt_template_reviews(
        state_b,
        session_id=str(session_id),
        current_state_id=str(record_b.id),
    )

    events_b = await sessions_service.list_interpretation_events(session_id, status="pending")
    pt_b = [e for e in events_b if e.kind is InterpretationKind.LLM_PROMPT_TEMPLATE]
    drafts = {e.llm_draft for e in pt_b}
    assert prompt_b in drafts, "draft-aware dedup must surface a fresh event for the edited prompt B"


def test_has_pending_prompt_template_requirement_false_on_duplicate() -> None:
    """LOW-a: the helper returns False when TWO pending PT requirements match.

    Mirrors ``_matching_pending_requirement_index``'s exactly-one multiplicity:
    a duplicate-requirement node must be skipped to the fail-closed orphan gate,
    never surfaced (which would raise an opaque 500 at the writer boundary).
    """

    options = {
        "prompt_template": "Rate {{ row.html }}.",
        INTERPRETATION_REQUIREMENTS_KEY: [
            {
                "id": "prompt_template_review_1",
                "kind": InterpretationKind.LLM_PROMPT_TEMPLATE.value,
                "user_term": "llm_prompt_template:rate_node",
                "status": "pending",
                "draft": "Rate {{ row.html }}.",
            },
            {
                "id": "prompt_template_review_2",
                "kind": InterpretationKind.LLM_PROMPT_TEMPLATE.value,
                "user_term": "llm_prompt_template:rate_node",
                "status": "pending",
                "draft": "Rate {{ row.html }}.",
            },
        ],
    }

    assert (
        ComposerServiceImpl._has_pending_prompt_template_requirement(
            options,
            user_term="llm_prompt_template:rate_node",
        )
        is False
    )


def test_has_pending_prompt_template_requirement_rejects_malformed_present_requirements() -> None:
    """Present requirement state is internal composer data, not optional input."""

    options = {
        "prompt_template": "Rate {{ row.html }}.",
        INTERPRETATION_REQUIREMENTS_KEY: [{"kind": InterpretationKind.LLM_PROMPT_TEMPLATE.value, "status": "pending", "user_term": 123}],
    }

    with pytest.raises(InvariantError, match="user_term"):
        ComposerServiceImpl._has_pending_prompt_template_requirement(
            options,
            user_term="llm_prompt_template:rate_node",
        )


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
    """A pre-existing pending event is still broken if the live prompt has two replacement sites."""

    composer = _build_composer(tmp_path, sessions_service)
    state = _state_with_llm_node()
    session_id, state_id = await _seed_session_and_state(sessions_service, state=state)
    await sessions_service.create_pending_interpretation_event(
        session_id=session_id,
        composition_state_id=state_id,
        affected_node_id="rate_node",
        tool_call_id="seed_review",
        user_term="cool",
        kind=InterpretationKind.VAGUE_TERM,
        llm_draft="modern, useful, engaging, and clear for the public.",
        model_identifier="anthropic/claude-opus-4-7",
        model_version="2026-05-01",
        provider="anthropic",
        composer_skill_hash="a" * 64,
    )

    duplicate_prompt = (
        "Rate how {{interpretation:cool}} this row is. Use this meaning for {{interpretation:cool}} after the user approves it."
    )
    repaired_prompt = "Rate how {{interpretation:cool}} this row is after the user approves the pending definition."

    def _set_pipeline_args_with_prompt(prompt: str) -> dict[str, Any]:
        args = _set_pipeline_with_pending_interpretation_args()
        args["nodes"][0]["options"]["prompt_template"] = prompt
        return args

    llm = _ScriptedLLM(
        [
            _fake_response_with_tool_call(
                tool_call_id="call_duplicate_placeholder",
                tool_name="set_pipeline",
                arguments=_set_pipeline_args_with_prompt(duplicate_prompt),
            ),
            _fake_text_response("Done — interpretation review is pending."),
            _fake_response_with_tool_call(
                tool_call_id="call_repair_duplicate_placeholder",
                tool_name="set_pipeline",
                arguments=_set_pipeline_args_with_prompt(repaired_prompt),
            ),
            _fake_text_response("Done — interpretation review is still pending."),
        ]
    )

    result = await composer._run_one_turn_for_test(
        llm=llm,
        session_id=str(session_id),
        current_state_id=str(state_id),
        initial_state=state,
        message="create a workflow that rates how cool pages are",
    )

    assert [inv.tool_name for inv in result.tool_invocations] == [
        "set_pipeline",
        "set_pipeline",
    ]
    events = await sessions_service.list_interpretation_events(session_id, status="pending")
    assert len([e for e in events if e.kind is InterpretationKind.VAGUE_TERM]) == 1
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
    # The model staged the vague_term review (after the early arg_error retry);
    # the backend also auto-surfaces the node's llm_prompt_template review at
    # finalization (elspeth-e51216d305).
    vague_events = [e for e in events if e.kind is InterpretationKind.VAGUE_TERM]
    assert len(vague_events) == 1
    assert vague_events[0].tool_call_id == "call_review"
    pt_events = [e for e in events if e.kind is InterpretationKind.LLM_PROMPT_TEMPLATE]
    assert len(pt_events) == 1
    assert pt_events[0].tool_call_id.startswith("backend_auto_surface:")


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


# ---------------------------------------------------------------------------
# Task 6 review CRITICAL: the END advisor gate must REACH pipelines whose only
# pending-review site is an auto-surfaceable llm_prompt_template (the canonical
# "use an LLM to rate" case). Pre-fix BOTH end-gate pre-checks used the
# UNFILTERED _missing_pending_interpretation_review_sites, so a still-unsurfaced
# PT site (which the finalize tail auto-surfaces) suppressed the advisor — the
# authoritative gate never fired for any LLM-prompt-template pipeline. The fix
# excludes PT sites from the suppress decision (only GENUINE non-PT orphans
# suppress); the tail's FINAL orphan gate stays unfiltered (fail-closed).
#
# These regressions are NON-VACUOUS: unlike the suites that stub
# _missing_pending_interpretation_review_sites -> (), they drive the REAL
# function over a REAL CompositionState and FIRST prove the masking site exists.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_end_advisor_gate_reaches_unsurfaced_prompt_template_pipeline_p2(
    tmp_path: Path,
    sessions_service: SessionServiceImpl,
) -> None:
    """P2 (_try_terminate_no_tools): advisor fires for a PT-only pipeline.

    Half (1): the REAL ``_missing_pending_interpretation_review_sites`` returns a
    NON-EMPTY set containing an ``LLM_PROMPT_TEMPLATE`` site for this state —
    BEFORE the finalize tail auto-surfaces it. This proves the masking condition
    that suppressed the advisor pre-fix is real for this state (without it, half
    (2) would be theater). It MUST be asserted before driving the gate, because
    the tail creates the pending PT event and the same call then returns empty.

    Half (2): with ``_run_advisor_checkpoint`` overridden by an assertable CLEAN
    ``AsyncMock`` (no real Opus call), the END gate is driven and the advisor IS
    awaited; the CLEAN verdict falls through to finalize (``action == "return"``,
    not an advisor-driven ``continue``).

    Pre-fix this test fails: the unfiltered pre-check is non-empty -> advisor
    skipped -> ``await_count == 0``. Post-fix the PT site is filtered out of the
    suppress decision -> advisor runs -> ``await_count >= 1``.
    """

    from elspeth.web.composer.audit import BufferingRecorder

    composer = _build_composer(tmp_path, sessions_service)
    state = _state_with_prompt_template_review_node()
    session_id, state_id = await _seed_session_and_state(sessions_service, state=state)

    # Half (1) — the masking site is REAL for this state (and is PT-kind).
    sites = await composer._missing_pending_interpretation_review_sites(state, session_id=str(session_id))
    assert sites, "expected the unsurfaced PT site to be a (real) pending-review orphan pre-fix"
    assert any(site[2] is InterpretationKind.LLM_PROMPT_TEMPLATE for site in sites)

    # Per-instance assertable advisor stub (instance attr wins over the autouse
    # class-level CLEAN stub) so we can assert it was awaited.
    advisor_mock = _AdvisorCheckpointFake(AdvisorCheckpointVerdict(ok=True, blocking=False, findings_text="CLEAN"))
    composer._run_advisor_checkpoint = advisor_mock  # type: ignore[method-assign]

    class _AssistantMessage:
        content = "Done — the pipeline is ready."

    outcome = await composer._try_terminate_no_tools(
        assistant_message=_AssistantMessage(),
        message="rate how cool the pages are",
        llm_messages=[],
        state=state,
        session_id=str(session_id),
        current_state_id=str(state_id),
        initial_version=1,
        user_id="alice",
        last_runtime_preflight=None,
        runtime_preflight_cache=composer._new_runtime_preflight_cache(),
        session_scope=str(session_id),
        mutation_success_seen=True,
        recorder=BufferingRecorder(),
        progress=None,
        repair_turns_used=0,
        persisted_assistant_message_id=None,
        persisted_tool_call_turn=False,
        advisor_checkpoint_passes_used=0,
    )

    # Half (2) — the advisor WAS reached (the crux: pre-fix it was suppressed).
    assert advisor_mock.await_count >= 1, "END advisor gate must fire for a PT-only pipeline"
    # CLEAN verdict falls through to finalize; it did NOT advisor-block/continue.
    assert outcome.action == "return"


@pytest.mark.asyncio
async def test_end_advisor_gate_reaches_prompt_template_pipeline_p5_budget_exhaustion(
    tmp_path: Path,
    sessions_service: SessionServiceImpl,
) -> None:
    """P5 (_classify_and_budget_turn budget-exhaustion last-chance finalize).

    Same masking + same fix as P2, but on the DISTINCT P5 code site. The session
    starts from a REAL ``_state_with_prompt_template_review_node`` (an LLM node
    with an unsurfaced PT requirement and NO ``model`` — so no genuine
    ``llm_model_choice`` orphan: the ONLY orphaned pre-check site is the
    auto-surfaceable PT one). With ``max_composition_turns=1`` a single
    ``set_metadata`` mutation exhausts the composition budget; the bonus call
    returns plain text and finalizes through the P5 pre-check. We use
    ``set_metadata`` (not ``set_pipeline``) so the existing PT node is not
    re-canonicalized — a model-less LLM node cannot survive ``set_pipeline``
    canonicalization, and a model-bearing one would re-introduce the
    ``llm_model_choice`` orphan that legitimately suppresses the advisor.

    Starting from a NON-empty state means the empty->non-empty early pass does
    NOT fire, so the only advisor call possible is the END gate — but we still
    discriminate on ``phase == "end"`` for robustness. Pre-fix the unfiltered PT
    site suppressed this pre-check (no ``end`` call); post-fix it is filtered and
    the advisor is awaited.
    """

    composer = _build_composer(tmp_path, sessions_service, max_composition_turns=1)
    state = _state_with_prompt_template_review_node()
    session_id, state_id = await _seed_session_and_state(sessions_service, state=state)

    # Half (1) — the masking site is REAL for this state (and is PT-kind), with
    # no genuine non-PT orphan to legitimately suppress the advisor.
    sites = await composer._missing_pending_interpretation_review_sites(state, session_id=str(session_id))
    assert sites, "expected the unsurfaced PT site to be a (real) pending-review orphan pre-fix"
    assert all(site[2] is InterpretationKind.LLM_PROMPT_TEMPLATE for site in sites)

    advisor_mock = _AdvisorCheckpointFake(AdvisorCheckpointVerdict(ok=True, blocking=False, findings_text="CLEAN"))
    composer._run_advisor_checkpoint = advisor_mock  # type: ignore[method-assign]

    llm = _ScriptedLLM(
        [
            _fake_response_with_tool_call(
                tool_call_id="call_set_metadata",
                tool_name="set_metadata",
                arguments={"patch": {"name": "Rate pipeline"}},
            ),
            _fake_text_response("Done — the pipeline is ready."),
        ]
    )

    await composer._run_one_turn_for_test(
        llm=llm,
        session_id=str(session_id),
        current_state_id=str(state_id),
        initial_state=state,
        message="give it a name",
    )

    # Half (2) — the P5 END advisor pre-check was reached (pre-fix the unfiltered
    # PT site suppressed it). Discriminate on the END phase for robustness.
    end_calls = [call for call in advisor_mock.await_args_list if call.kwargs.get("phase") == "end"]
    assert end_calls, "P5 budget-exhaustion END advisor gate must fire for a PT pipeline"
    events = await sessions_service.list_interpretation_events(session_id, status="pending")
    assert any(e.kind is InterpretationKind.LLM_PROMPT_TEMPLATE for e in events)


# ---------------------------------------------------------------------------
# Advisor-blocked terminal return must ALSO surface PT (+ run the orphan gate).
#
# The three advisor-blocked terminal returns (P2 unavailable, P2 exhausted,
# P5 unavailable/exhausted) bypass the shared finalize tail that auto-surfaces
# the llm_prompt_template review and runs the unfiltered orphan gate. A tool
# turn that (re-)drafts an LLM node's prompt_template stages a pending PT
# requirement but no pending EVENT; if the END advisor then blocks/is
# unavailable, the persisted max-version state carries a PT *site* with no
# pending *event*. RUN reads that state via get_current_state and
# materialize_state_for_execution raises UnresolvedInterpretationPlaceholderError
# even though the frontend pendingCount (events) is zero. This is the live
# staging asymmetry (3/8 tutorial RUNs failing with HTTP 500).
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_advisor_unavailable_terminal_return_surfaces_prompt_template(
    tmp_path: Path,
    sessions_service: SessionServiceImpl,
) -> None:
    """P2 advisor-UNAVAILABLE blocked return must surface the PT review event.

    Reproduces the staging asymmetry: a node with a pending auto-staged
    ``llm_prompt_template`` requirement and NO pending event reaches the END
    advisor gate; the advisor is unavailable (``ok=False``) so the
    ``_try_terminate_no_tools`` blocked return fires. Pre-fix that return
    bypasses the auto-surface + orphan gate, so NO pending PT event exists ->
    pendingCount(events) == 0 while interpretation_sites scan == 1, and a
    later RUN raises ``UnresolvedInterpretationPlaceholderError``. Post-fix the
    blocked return surfaces the resolvable pending event first.
    """

    from elspeth.web.composer.audit import BufferingRecorder

    composer = _build_composer(tmp_path, sessions_service)
    state = _state_with_prompt_template_review_node()
    session_id, state_id = await _seed_session_and_state(sessions_service, state=state)

    # Force the blocked-return branch (ok=False == unavailable). Instance attr
    # wins over the autouse class-level CLEAN stub.
    composer._run_advisor_checkpoint = _AdvisorCheckpointFake(  # type: ignore[method-assign]
        AdvisorCheckpointVerdict(ok=False, blocking=False, findings_text="unavailable")
    )

    class _AssistantMessage:
        content = "Done — the pipeline is ready."

    outcome = await composer._try_terminate_no_tools(
        assistant_message=_AssistantMessage(),
        message="rate how cool the pages are",
        llm_messages=[],
        state=state,
        session_id=str(session_id),
        current_state_id=str(state_id),
        initial_version=1,
        user_id="alice",
        last_runtime_preflight=None,
        runtime_preflight_cache=composer._new_runtime_preflight_cache(),
        session_scope=str(session_id),
        mutation_success_seen=True,
        recorder=BufferingRecorder(),
        progress=None,
        repair_turns_used=0,
        persisted_assistant_message_id=None,
        persisted_tool_call_turn=False,
        advisor_checkpoint_passes_used=0,
    )

    assert outcome.action == "return"
    assert outcome.result is not None

    # The events-side realignment: exactly one resolvable pending PT event now
    # exists, surfaced by the backend against the frozen final skeleton. (Pre-fix
    # there are ZERO pending events — the assertion below fails for the right
    # reason: the placeholder reached the runnable state with no card.)
    events = await sessions_service.list_interpretation_events(session_id, status="pending")
    pt = [e for e in events if e.kind is InterpretationKind.LLM_PROMPT_TEMPLATE]
    assert len(pt) == 1
    assert pt[0].affected_node_id == "rate_node"
    assert pt[0].tool_call_id.startswith("backend_auto_surface:")

    # The PT site is now resolvable (a pending event matches it), so the orphan
    # gate sees no orphan for it.
    missing = await composer._missing_pending_interpretation_review_sites(
        state,
        session_id=str(session_id),
    )
    assert all(site[2] is not InterpretationKind.LLM_PROMPT_TEMPLATE for site in missing)

    # GUARD (do NOT weaken execution/service.py:519): the execution materializer
    # scans STRUCTURE, not events; the placeholder is removed only by the BAKE
    # when the user resolves the card. So the state still materializes-as-pending
    # right after the turn. The fix re-aligns the frontend events gate, it does
    # NOT auto-resolve the review.
    from elspeth.web.interpretation_state import (
        InterpretationReviewPending,
        materialize_state_for_execution,
    )

    materialized = materialize_state_for_execution(state)
    assert isinstance(materialized, InterpretationReviewPending)


@pytest.mark.asyncio
async def test_advisor_exhausted_terminal_return_surfaces_prompt_template(
    tmp_path: Path,
    sessions_service: SessionServiceImpl,
) -> None:
    """P2 advisor-EXHAUSTED (blocking on last pass) blocked return surfaces PT.

    Variant of the unavailable case: the advisor FLAGS the pipeline
    (``blocking=True``) on the last budgeted pass, driving the
    ``reason="exhausted"`` blocked return. That return must also surface the
    resolvable pending PT event so the persisted runnable state is resolvable.
    """

    from elspeth.web.composer.audit import BufferingRecorder

    composer = _build_composer(tmp_path, sessions_service)
    state = _state_with_prompt_template_review_node()
    session_id, state_id = await _seed_session_and_state(sessions_service, state=state)

    composer._run_advisor_checkpoint = _AdvisorCheckpointFake(  # type: ignore[method-assign]
        AdvisorCheckpointVerdict(ok=True, blocking=True, findings_text="FLAGGED: review the prompt")
    )

    class _AssistantMessage:
        content = "Done — the pipeline is ready."

    outcome = await composer._try_terminate_no_tools(
        assistant_message=_AssistantMessage(),
        message="rate how cool the pages are",
        llm_messages=[],
        state=state,
        session_id=str(session_id),
        current_state_id=str(state_id),
        initial_version=1,
        user_id="alice",
        last_runtime_preflight=None,
        runtime_preflight_cache=composer._new_runtime_preflight_cache(),
        session_scope=str(session_id),
        mutation_success_seen=True,
        recorder=BufferingRecorder(),
        progress=None,
        repair_turns_used=0,
        persisted_assistant_message_id=None,
        persisted_tool_call_turn=False,
        # Force last-pass so (used + 1) >= max_passes -> the "exhausted" return.
        advisor_checkpoint_passes_used=composer._settings.composer_advisor_checkpoint_max_passes - 1,
    )

    assert outcome.action == "return"
    events = await sessions_service.list_interpretation_events(session_id, status="pending")
    pt = [e for e in events if e.kind is InterpretationKind.LLM_PROMPT_TEMPLATE]
    assert len(pt) == 1
    assert pt[0].tool_call_id.startswith("backend_auto_surface:")


@pytest.mark.asyncio
async def test_p5_budget_exhaustion_advisor_blocked_return_surfaces_prompt_template(
    tmp_path: Path,
    sessions_service: SessionServiceImpl,
) -> None:
    """P5 budget-exhaustion advisor-blocked terminal return surfaces PT.

    The THIRD blocked-return site lives in ``_classify_and_budget_turn`` (the
    B-4D-3 budget-exhaustion last-chance finalize). With
    ``max_composition_turns=1`` the single ``set_pipeline`` mutation exhausts the
    composition budget; the bonus call returns no tool calls, and the END advisor
    gate is forced UNAVAILABLE (``ok=False``) so the P5 blocked return fires
    instead of falling through to the shared finalize tail. Pre-fix that return
    bypasses auto-surface -> zero pending events for the staged PT requirement
    (the eventless-placeholder asymmetry); post-fix it surfaces the resolvable
    event. Threading uses ``persist.current_state_id`` (the mutation persisted
    before classify), exercising the P5-specific plumbing the P2 tests do not.
    """

    composer = _build_composer(tmp_path, sessions_service, max_composition_turns=1)
    state = _state_with_prompt_template_review_node()  # model-less PT node: PT is the ONLY orphan site
    session_id, state_id = await _seed_session_and_state(sessions_service, state=state)

    # Pre-condition: PT is the only orphaned pre-check site (no genuine non-PT
    # orphan that would legitimately suppress the END advisor), so genuine_orphans
    # is empty and the P5 END gate fires (see the CLEAN counterpart
    # test_end_advisor_gate_reaches_prompt_template_pipeline_p5_budget_exhaustion).
    sites = await composer._missing_pending_interpretation_review_sites(state, session_id=str(session_id))
    assert sites and all(site[2] is InterpretationKind.LLM_PROMPT_TEMPLATE for site in sites)

    # Force the P5 blocked-return branch (ok=False == unavailable -> fail closed).
    advisor_mock = _AdvisorCheckpointFake(AdvisorCheckpointVerdict(ok=False, blocking=False, findings_text="unavailable"))
    composer._run_advisor_checkpoint = advisor_mock  # type: ignore[method-assign]

    # set_metadata (not set_pipeline) so the model-less PT node is not
    # re-canonicalized; the single mutation exhausts max_composition_turns=1 ->
    # the B-4D-3 bonus call returns plain text -> the P5 finalize pre-check runs.
    llm = _ScriptedLLM(
        [
            _fake_response_with_tool_call(
                tool_call_id="call_set_metadata",
                tool_name="set_metadata",
                arguments={"patch": {"name": "Rate pipeline"}},
            ),
            _fake_text_response("Done — the pipeline is ready."),
        ]
    )

    await composer._run_one_turn_for_test(
        llm=llm,
        session_id=str(session_id),
        current_state_id=str(state_id),
        initial_state=state,
        message="give it a name",
    )

    # The P5 END advisor gate fired and returned the blocked path; the fix runs
    # the surface+gate pair on that blocked return, so the PT event is surfaced.
    end_calls = [call for call in advisor_mock.await_args_list if call.kwargs.get("phase") == "end"]
    assert end_calls, "P5 budget-exhaustion END advisor gate must fire for a PT pipeline"
    events = await sessions_service.list_interpretation_events(session_id, status="pending")
    pt = [e for e in events if e.kind is InterpretationKind.LLM_PROMPT_TEMPLATE]
    assert len(pt) == 1, "P5 advisor-blocked finalize must auto-surface the PT review"
    assert pt[0].affected_node_id == "rate_node"
    assert pt[0].tool_call_id.startswith("backend_auto_surface:")


@pytest.mark.asyncio
async def test_advisor_blocked_terminal_return_still_fails_closed_on_bare_token_orphan(
    tmp_path: Path,
    sessions_service: SessionServiceImpl,
) -> None:
    """Guard: a GENUINE (non-PT) orphan stays fail-closed at the no-tool finalize.

    A bare ``{{interpretation:cool}}`` vague-term token with no pending event is
    a case-a orphan that auto-surface skips (no PT requirement). With the repair
    budget exhausted, ``genuine_orphans`` is non-empty so the END advisor gate is
    SKIPPED and control falls through to the CLEAN tail's UNFILTERED orphan gate
    (NOT the advisor-blocked return). The turn must fail closed there exactly as
    before this fix — the shared ``_surface_pt_and_gate_orphans_or_none`` helper
    preserves the case-a fail-closed behaviour (surface no-ops, unfiltered gate
    blocks). Pins that the DRY refactor did not regress genuine orphans.
    """

    from elspeth.web.composer.audit import BufferingRecorder

    composer = _build_composer(tmp_path, sessions_service)
    state = _state_with_llm_node()  # bare {{interpretation:cool}}, no PT requirement, no event
    session_id, state_id = await _seed_session_and_state(sessions_service, state=state)

    composer._run_advisor_checkpoint = _AdvisorCheckpointFake(  # type: ignore[method-assign]
        AdvisorCheckpointVerdict(ok=False, blocking=False, findings_text="unavailable")
    )

    class _AssistantMessage:
        content = "Done — the pipeline is ready."

    outcome = await composer._try_terminate_no_tools(
        assistant_message=_AssistantMessage(),
        message="rate how cool the pages are",
        llm_messages=[],
        state=state,
        session_id=str(session_id),
        current_state_id=str(state_id),
        initial_version=1,
        user_id="alice",
        last_runtime_preflight=None,
        runtime_preflight_cache=composer._new_runtime_preflight_cache(),
        session_scope=str(session_id),
        mutation_success_seen=True,
        recorder=BufferingRecorder(),
        progress=None,
        # Exhaust the repair budget so the model-repairable vague-term branch is
        # skipped and control reaches the advisor gate + the orphan gate (the
        # blocked-return path under test).
        repair_turns_used=2,
        persisted_assistant_message_id=None,
        persisted_tool_call_turn=False,
        advisor_checkpoint_passes_used=0,
    )

    assert outcome.action == "return"
    assert outcome.result is not None
    preflight = outcome.result.runtime_preflight
    assert preflight is not None
    assert preflight.is_valid is False
    assert preflight.readiness.execution_ready is False
    assert {blocker.code for blocker in preflight.readiness.blockers} == {"interpretation_review_orphaned"}
