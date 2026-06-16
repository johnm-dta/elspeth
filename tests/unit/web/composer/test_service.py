"""Tests for ComposerServiceImpl — LLM tool-use loop with mock LLM."""

from __future__ import annotations

import asyncio
import contextlib
import json
import threading
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast
from unittest.mock import AsyncMock, patch
from uuid import UUID, uuid4

import pytest
import structlog
from sqlalchemy import select
from sqlalchemy.exc import OperationalError
from sqlalchemy.pool import StaticPool

from elspeth.contracts.composer_progress import ComposerProgressEvent
from elspeth.contracts.hashing import stable_hash
from elspeth.web.catalog.protocol import CatalogService
from elspeth.web.composer.protocol import (
    ComposerConvergenceError,
    ComposerPluginCrashError,
    ComposerResult,
    ComposerRuntimePreflightError,
    ComposerServiceError,
    ToolArgumentError,
)
from elspeth.web.composer.service import (
    AdvisorCheckpointVerdict,
    ComposerAvailability,
    ComposerServiceImpl,
    _compose_preflight_repair_message,
)
from elspeth.web.composer.state import (
    CompositionState,
    OutputSpec,
    PipelineMetadata,
    SourceSpec,
    ValidationSummary,
)
from elspeth.web.composer.tools import ToolResult
from elspeth.web.composer.tools import execute_tool as _execute_tool
from elspeth.web.execution.preflight import runtime_preflight_settings_hash
from elspeth.web.execution.schemas import (
    ValidationCheck,
    ValidationError,
    ValidationReadiness,
    ValidationReadinessBlocker,
)
from elspeth.web.execution.schemas import (
    ValidationResult as ValidationResultModel,
)
from elspeth.web.interpretation_state import INTERPRETATION_REVIEW_PENDING_CODE
from elspeth.web.sessions.engine import create_session_engine
from elspeth.web.sessions.models import chat_messages_table, sessions_table
from elspeth.web.sessions.schema import initialize_session_schema
from elspeth.web.sessions.service import SessionServiceImpl
from elspeth.web.sessions.telemetry import build_sessions_telemetry
from tests.unit.web.composer._helpers import (
    FakeChoice,
    FakeFunction,
    FakeLLMResponse,
    FakeMessage,
    FakeToolCall,
    _empty_state,
    _make_llm_response,
    _make_settings,
    _mock_catalog,
    _stub_advisor_end_gate_clean,  # noqa: F401  (autouse end-gate CLEAN stub)
)


def _execution_ready() -> ValidationReadiness:
    return ValidationReadiness(authoring_valid=True, execution_ready=True, completion_ready=True, blockers=[])


def _not_authoring_ready(code: str = "test_blocker") -> ValidationReadiness:
    return ValidationReadiness(
        authoring_valid=False,
        execution_ready=False,
        completion_ready=False,
        blockers=[
            ValidationReadinessBlocker(
                code=code,
                component_id=None,
                component_type=None,
                detail=code,
            )
        ],
    )


def _pending_interpretation_readiness() -> ValidationReadiness:
    return ValidationReadiness(
        authoring_valid=True,
        execution_ready=False,
        completion_ready=True,
        blockers=[
            ValidationReadinessBlocker(
                code=INTERPRETATION_REVIEW_PENDING_CODE,
                component_id="rate_node",
                component_type="transform",
                detail="rate_node:cool",
            )
        ],
    )


def ValidationResult(
    *,
    is_valid: bool,
    checks: list[ValidationCheck],
    errors: list[ValidationError],
    readiness: ValidationReadiness | None = None,
    **kwargs: Any,
) -> ValidationResultModel:
    return ValidationResultModel(
        is_valid=is_valid,
        checks=checks,
        errors=errors,
        readiness=readiness or (_execution_ready() if is_valid else _not_authoring_ready()),
        **kwargs,
    )


def _assert_no_mutation_empty_state_blocker(
    result: ComposerResult,
    *,
    tool_name: str,
    expected_detail: str,
) -> None:
    """Assert the no-mutation empty-state augmentation contract (post elspeth-861b0c58f5).

    The new shape (vs. the old synthetic-replacement behavior):
    - The model's prose is preserved verbatim at the start of result.message
      (raw_assistant_content carries the same prose unaugmented).
    - A system-attributed suffix is appended carrying the concrete blocker.
    - The runtime_preflight ValidationResult records the state_exists=false
      check and the blocker detail for audit-trail attribution.
    """
    assert result.runtime_preflight is not None
    assert result.runtime_preflight.is_valid is False
    assert [check.name for check in result.runtime_preflight.checks] == ["state_exists"]
    assert result.raw_assistant_content is not None
    # Model's prose preserved verbatim at the start; system suffix appended.
    assert result.message.startswith(result.raw_assistant_content)
    # System suffix carries the operator-facing meta-narration.
    assert "[ELSPETH-SYSTEM]" in result.message
    assert "still empty" in result.message
    # Blocker detail surfaced both in suffix (for the user) and in the
    # runtime_preflight ValidationResult (for audit-trail attribution).
    assert tool_name in result.message
    assert expected_detail in result.message
    blocker_detail = result.runtime_preflight.checks[0].detail
    assert tool_name in blocker_detail
    assert expected_detail in blocker_detail


def _session_engine_with_session() -> tuple[Any, str]:
    engine = create_session_engine(
        "sqlite:///:memory:",
        poolclass=StaticPool,
        connect_args={"check_same_thread": False},
    )
    initialize_session_schema(engine)
    session_id = str(uuid4())
    now = datetime.now(UTC)
    with engine.begin() as conn:
        conn.execute(
            sessions_table.insert().values(
                id=session_id,
                user_id="test-user",
                auth_provider_type="local",
                title="Test Session",
                trust_mode="auto_commit",
                density_default="high",
                created_at=now,
                updated_at=now,
            )
        )
    return engine, session_id


def _insert_user_message(engine: Any, session_id: str, content: str) -> str:
    """Persist a user chat message for tests that create verbatim blobs."""
    user_message_id = str(uuid4())
    with engine.begin() as conn:
        latest = conn.execute(
            select(chat_messages_table.c.sequence_no)
            .where(chat_messages_table.c.session_id == session_id)
            .order_by(chat_messages_table.c.sequence_no.desc())
        ).first()
        sequence_no = 1 if latest is None else latest.sequence_no + 1
        conn.execute(
            chat_messages_table.insert().values(
                id=user_message_id,
                session_id=session_id,
                role="user",
                content=content,
                raw_content=None,
                tool_calls=None,
                tool_call_id=None,
                sequence_no=sequence_no,
                writer_principal="route_user_message",
                created_at=datetime.now(UTC),
                composition_state_id=None,
                parent_assistant_id=None,
            )
        )
    return user_message_id


def _verbatim_blob_context(engine: Any, session_id: str, content: str) -> dict[str, str]:
    user_message_content = f"Use this exact content:\n{content}"
    return {
        "user_message_id": _insert_user_message(engine, session_id, user_message_content),
        "user_message_content": user_message_content,
    }


def _test_sessions_service(engine: Any, data_dir: Path | None = None) -> SessionServiceImpl:
    return SessionServiceImpl(
        engine,
        data_dir=data_dir,
        telemetry=build_sessions_telemetry(),
        log=structlog.get_logger("test.composer.sessions"),
    )


@pytest.fixture(autouse=True)
def _composer_available_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep service tests focused on compose behavior, not local API keys."""

    def _available(self: ComposerServiceImpl) -> ComposerAvailability:
        return ComposerAvailability(available=True, model=self._model, provider="test")

    monkeypatch.setattr(ComposerServiceImpl, "_compute_availability", _available)


@pytest.fixture(autouse=True)
def _composer_to_thread_uses_test_worker(monkeypatch: pytest.MonkeyPatch) -> None:
    """Run composer to_thread seams through a deterministic test worker.

    These unit tests assert the composer offloads synchronous tool and audit
    work; they do not need to test asyncio's executor implementation. The shim
    keeps the off-event-loop-thread property visible while avoiding local
    executor hangs from masking composer behavior.
    """

    async def test_to_thread(func: Any, /, *args: Any, **kwargs: Any) -> Any:
        import threading

        result: list[Any] = []
        failures: list[BaseException] = []

        def run() -> None:
            try:
                result.append(func(*args, **kwargs))
            except BaseException as exc:
                failures.append(exc)

        worker = threading.Thread(target=run, name="composer-test-worker")
        worker.start()
        while worker.is_alive():
            await asyncio.sleep(0.001)
        worker.join()
        if failures:
            raise failures[0]
        if result:
            return result[0]
        return None

    monkeypatch.setattr("asyncio.to_thread", test_to_thread)


class TestComposerTextOnlyResponse:
    @pytest.mark.asyncio
    async def test_non_build_text_only_returns_immediately(self) -> None:
        """Non-build text-only replies still terminate without mutation."""
        catalog = _mock_catalog()
        settings = _make_settings()
        service = ComposerServiceImpl(
            catalog=catalog,
            settings=settings,
        )
        state = _empty_state()

        llm_response = _make_llm_response(content="I'll help you build a pipeline!")

        with patch.object(service, "_call_llm", new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = llm_response
            result = await service.compose("What can this composer do?", [], state)

        assert isinstance(result, ComposerResult)
        assert result.message == "I'll help you build a pipeline!"
        assert result.state.version == 1  # unchanged

    @pytest.mark.asyncio
    async def test_build_request_text_only_on_empty_state_returns_no_mutation_blocker(self) -> None:
        """A build request cannot end with conceptual prose when no mutation ran."""
        catalog = _mock_catalog()
        settings = _make_settings()
        service = ComposerServiceImpl(catalog=catalog, settings=settings)
        state = _empty_state()
        model_prose = "I set up the workflow conceptually and can continue."
        llm_response = _make_llm_response(content=model_prose)

        with patch.object(service, "_call_llm", new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = llm_response
            result = await service.compose("Set this up to actually run from leads_q3.csv.", [], state)

        assert result.state.version == state.version
        assert result.raw_assistant_content == model_prose
        assert result.runtime_preflight is not None
        assert result.runtime_preflight.is_valid is False
        assert [check.name for check in result.runtime_preflight.checks] == ["state_exists"]
        # New contract (post elspeth-861b0c58f5): model prose preserved
        # verbatim, system suffix appended carrying the concrete blocker.
        assert result.message.startswith(model_prose)
        assert "[ELSPETH-SYSTEM]" in result.message
        assert "still empty" in result.message
        assert "the model ended the turn without calling any build/edit tool" in result.message
        # Audit-trail attribution: blocker recorded in the runtime_preflight
        # ValidationResult (so structured downstream consumers can route on it).
        assert "state_exists=false" in result.runtime_preflight.checks[0].detail

    @pytest.mark.asyncio
    async def test_failed_mutation_then_empty_state_reply_names_blocking_tool_error(self) -> None:
        """A failed mutation followed by final prose must surface the tool error."""
        catalog = _mock_catalog()
        settings = _make_settings()
        service = ComposerServiceImpl(catalog=catalog, settings=settings)
        state = _empty_state()
        failed_set_pipeline = _make_llm_response(
            tool_calls=[
                {
                    "id": "call_bad_pipeline",
                    "name": "set_pipeline",
                    "arguments": {
                        "source": {"on_success": "rows", "options": {}},
                        "nodes": [],
                        "edges": [],
                        "outputs": [],
                    },
                }
            ],
        )
        final_prose = "I tried to build it, but the workflow is still conceptual."
        text_response = _make_llm_response(content=final_prose)

        with patch.object(service, "_call_llm", new_callable=AsyncMock) as mock_llm:
            mock_llm.side_effect = [failed_set_pipeline, text_response]
            result = await service.compose("Build the CSV workflow now.", [], state)

        assert result.state.version == state.version
        assert result.raw_assistant_content == final_prose
        assert result.runtime_preflight is not None
        assert result.runtime_preflight.is_valid is False
        # New contract: model prose preserved + system suffix with blocker.
        assert result.message.startswith(final_prose)
        assert "[ELSPETH-SYSTEM]" in result.message
        assert "still empty" in result.message
        assert "set_pipeline failed before mutation" in result.message
        assert "MissingRequiredPaths" in result.message
        assert "source.plugin" in result.message

    @pytest.mark.asyncio
    async def test_blob_only_success_then_empty_state_reply_returns_no_state_mutation_blocker(self, tmp_path: Path) -> None:
        """A blob-side success is not a successful CompositionState mutation."""
        catalog = _mock_catalog()
        engine, session_id = _session_engine_with_session()
        user_message_id = str(uuid4())
        user_message_content = "Build a runnable pipeline from this text."
        now = datetime.now(UTC)
        with engine.begin() as conn:
            conn.execute(
                chat_messages_table.insert().values(
                    id=user_message_id,
                    session_id=session_id,
                    role="user",
                    content=user_message_content,
                    raw_content=None,
                    tool_calls=None,
                    tool_call_id=None,
                    sequence_no=1,
                    writer_principal="route_user_message",
                    created_at=now,
                    composition_state_id=None,
                    parent_assistant_id=None,
                )
            )
        settings = _make_settings(data_dir=tmp_path)
        service = ComposerServiceImpl(
            catalog=catalog,
            settings=settings,
            sessions_service=_test_sessions_service(engine, tmp_path),
            session_engine=engine,
        )
        state = _empty_state()
        create_blob_turn = _make_llm_response(
            tool_calls=[
                {
                    "id": "call_create_blob",
                    "name": "create_blob",
                    "arguments": {
                        "filename": "seed.txt",
                        "mime_type": "text/plain",
                        "content": "hello",
                    },
                }
            ],
        )
        final_prose = "I created the input blob, so the pipeline is ready."
        text_response = _make_llm_response(content=final_prose)

        with patch.object(service, "_call_llm", new_callable=AsyncMock) as mock_llm:
            mock_llm.side_effect = [create_blob_turn, text_response]
            result = await service.compose(
                user_message_content,
                [],
                state,
                session_id=session_id,
                user_message_id=user_message_id,
            )

        assert result.state.version == state.version
        assert result.raw_assistant_content == final_prose
        assert result.runtime_preflight is not None
        assert result.runtime_preflight.is_valid is False
        # New contract: model prose preserved + system suffix with blocker.
        assert result.message.startswith(final_prose)
        assert "[ELSPETH-SYSTEM]" in result.message
        assert "still empty" in result.message
        assert "create_blob succeeded without mutating CompositionState" in result.message
        assert "state_exists=false" in result.runtime_preflight.checks[0].detail
        assert mock_llm.call_count == 2


class TestComposerSingleToolCall:
    @pytest.mark.asyncio
    async def test_tool_dispatch_receives_configured_blob_quota(self) -> None:
        """Composer dispatch must thread WebSettings blob quota into tool execution."""

        service = ComposerServiceImpl(
            catalog=_mock_catalog(),
            settings=_make_settings(max_blob_storage_per_session_bytes=3),
        )
        state = _empty_state()
        captured_quota: list[int | None] = []

        def fake_execute_tool(
            _tool_name: str,
            _arguments: dict[str, Any],
            _state: CompositionState,
            _catalog: CatalogService,
            *args: Any,
            max_blob_storage_per_session_bytes: int | None = None,
            **kwargs: Any,
        ) -> ToolResult:
            del args, kwargs
            captured_quota.append(max_blob_storage_per_session_bytes)
            return ToolResult(
                success=True,
                updated_state=_state.with_metadata({"name": "captured"}),
                validation=ValidationSummary(is_valid=True, errors=()),
                affected_nodes=("metadata",),
            )

        turn = _make_llm_response(
            tool_calls=[
                {
                    "id": "call_set_metadata",
                    "name": "set_metadata",
                    "arguments": {"patch": {"name": "captured"}},
                }
            ],
        )
        done = _make_llm_response(content="Done.")

        with (
            patch("elspeth.web.composer.tool_batch.execute_tool", side_effect=fake_execute_tool),
            patch.object(service, "_call_llm", new_callable=AsyncMock) as mock_llm,
        ):
            mock_llm.side_effect = [turn, done]
            await service.compose("Update metadata", [], state)

        assert captured_quota == [3]

    @pytest.mark.asyncio
    async def test_tool_dispatch_receives_composer_source_provenance_context(self) -> None:
        """Sync tool dispatch receives the user message and audited composer provenance."""

        service = ComposerServiceImpl(
            catalog=_mock_catalog(),
            settings=_make_settings(composer_model="gpt-5.5"),
        )
        service._availability = ComposerAvailability(  # type: ignore[misc]
            available=True,
            model="gpt-5.5",
            provider="test-provider",
        )
        state = _empty_state()
        captured_kwargs: dict[str, Any] = {}
        arguments = {"patch": {"name": "captured"}}

        def fake_execute_tool(
            _tool_name: str,
            _arguments: dict[str, Any],
            _state: CompositionState,
            _catalog: CatalogService,
            *args: Any,
            **kwargs: Any,
        ) -> ToolResult:
            del args, _arguments, _catalog
            captured_kwargs.update(kwargs)
            return ToolResult(
                success=True,
                updated_state=_state.with_metadata({"name": "captured"}),
                validation=ValidationSummary(is_valid=True, errors=()),
                affected_nodes=("metadata",),
            )

        turn = _make_llm_response(
            tool_calls=[
                {
                    "id": "call_set_metadata",
                    "name": "set_metadata",
                    "arguments": arguments,
                }
            ],
        )
        done = _make_llm_response(content="Done.")

        with (
            patch("elspeth.web.composer.tool_batch.execute_tool", side_effect=fake_execute_tool),
            patch.object(service, "_call_llm", new_callable=AsyncMock) as mock_llm,
        ):
            mock_llm.side_effect = [turn, done]
            await service.compose(
                "Create generated source content.",
                [],
                state,
                user_message_id="11111111-1111-1111-1111-111111111111",
            )

        assert captured_kwargs["user_message_id"] == "11111111-1111-1111-1111-111111111111"
        assert captured_kwargs["user_message_content"] == "Create generated source content."
        assert captured_kwargs["composer_model_identifier"] == "gpt-5.5"
        assert captured_kwargs["composer_model_version"] == "gpt-5.5"
        assert captured_kwargs["composer_provider"] == "test-provider"
        assert captured_kwargs["composer_skill_hash"] == service._composer_skill_hash
        assert captured_kwargs["tool_arguments_hash"] == stable_hash(arguments)

    @pytest.mark.asyncio
    async def test_explicit_approve_mutating_tool_creates_pending_proposal_without_state_mutation(
        self,
        composer_service_with_real_sessions: ComposerServiceImpl,
        result_session_id: str,
        fake_llm_one_set_pipeline_tool_call: Any,
    ) -> None:
        """Explicit approval mode stores mutating tool calls as pending proposals."""
        sessions_service = composer_service_with_real_sessions._sessions_service
        assert sessions_service is not None
        session_uuid = UUID(result_session_id)
        state = _empty_state()

        await sessions_service.update_composer_preferences(
            session_uuid,
            trust_mode="explicit_approve",
            density_default="high",
            actor="user:alice",
        )

        result = await composer_service_with_real_sessions._run_one_turn_for_test(
            llm=fake_llm_one_set_pipeline_tool_call,
            session_id=result_session_id,
            initial_state=state,
        )

        proposals = await sessions_service.list_composition_proposals(session_uuid)
        assert len(proposals) == 1
        assert proposals[0].tool_call_id == "call_set_pipeline"
        assert proposals[0].tool_name == "set_pipeline"
        assert proposals[0].status == "pending"
        assert "Replace the pipeline" in proposals[0].summary
        assert proposals[0].composer_model_identifier == composer_service_with_real_sessions._model
        assert proposals[0].composer_model_version == composer_service_with_real_sessions._model
        assert proposals[0].composer_provider == "test"
        assert proposals[0].composer_skill_hash == composer_service_with_real_sessions._composer_skill_hash
        assert proposals[0].tool_arguments_hash == stable_hash(proposals[0].arguments_json)
        assert result.tool_outcomes[0].post_version == state.version

    @pytest.mark.asyncio
    async def test_create_composition_proposal_normalizes_composer_provenance(
        self,
        composer_service_with_real_sessions: ComposerServiceImpl,
        result_session_id: str,
    ) -> None:
        """Proposal provenance is normalized before it becomes audit-bearing state."""
        sessions_service = composer_service_with_real_sessions._sessions_service
        assert sessions_service is not None
        session_uuid = UUID(result_session_id)

        proposal = await sessions_service.create_composition_proposal(
            session_id=session_uuid,
            tool_call_id="call_set_pipeline",
            tool_name="set_pipeline",
            summary="Replace the pipeline.",
            rationale="Composer proposed a pipeline update.",
            affects=["graph"],
            arguments_json={"source": {"plugin": "csv", "options": {}}},
            arguments_redacted_json={"source": {"plugin": "csv", "options": {}}},
            base_state_id=None,
            actor="assistant",
            composer_model_identifier=" openai/gpt-5-mini ",
            composer_model_version=" gpt-5-mini-2026-05-01 ",
            composer_provider=" openai ",
            composer_skill_hash=" sha256:composer-skill ",
            tool_arguments_hash=" sha256:tool-arguments ",
        )

        assert proposal.composer_model_identifier == "openai/gpt-5-mini"
        assert proposal.composer_model_version == "gpt-5-mini-2026-05-01"
        assert proposal.composer_provider == "openai"
        assert proposal.composer_skill_hash == "sha256:composer-skill"
        assert proposal.tool_arguments_hash == "sha256:tool-arguments"

    @pytest.mark.asyncio
    async def test_create_composition_proposal_rejects_blank_composer_provenance(
        self,
        composer_service_with_real_sessions: ComposerServiceImpl,
        result_session_id: str,
    ) -> None:
        """Blank proposal provenance fails before a partial audit row can persist."""
        from elspeth.contracts.errors import AuditIntegrityError

        sessions_service = composer_service_with_real_sessions._sessions_service
        assert sessions_service is not None
        session_uuid = UUID(result_session_id)

        with pytest.raises(AuditIntegrityError, match=r"composer provenance.*composer_provider"):
            await sessions_service.create_composition_proposal(
                session_id=session_uuid,
                tool_call_id="call_set_pipeline",
                tool_name="set_pipeline",
                summary="Replace the pipeline.",
                rationale="Composer proposed a pipeline update.",
                affects=["graph"],
                arguments_json={"source": {"plugin": "csv", "options": {}}},
                arguments_redacted_json={"source": {"plugin": "csv", "options": {}}},
                base_state_id=None,
                actor="assistant",
                composer_model_identifier="openai/gpt-5-mini",
                composer_model_version="gpt-5-mini-2026-05-01",
                composer_provider="\t ",
                composer_skill_hash="sha256:composer-skill",
                tool_arguments_hash="sha256:tool-arguments",
            )

        assert await sessions_service.list_composition_proposals(session_uuid) == []

    @pytest.mark.asyncio
    async def test_explicit_approve_does_not_intercept_create_blob(
        self,
        composer_service_with_real_sessions: ComposerServiceImpl,
        result_session_id: str,
        fake_llm_create_blob_then_set_pipeline: Any,
    ) -> None:
        """Regression for staging session 986fabf6-a723-4eb3-84de-2db1b7ae4e96:
        under trust_mode=explicit_approve the previous behaviour intercepted
        both create_blob and set_pipeline as proposals. The create_blob
        proposal was structurally unacceptable — the accept endpoint
        requires CompositionState.version to advance, but create_blob
        never advances state (it is a blob-store side effect). So users
        clicked 'Accept' on the proposal and got HTTP 409 'did not change
        composition state'.

        After the fix, create_blob executes immediately; only
        composition-mutation tools (set_pipeline here) become pending
        proposals."""
        sessions_service = composer_service_with_real_sessions._sessions_service
        assert sessions_service is not None
        session_uuid = UUID(result_session_id)
        state = _empty_state()

        await sessions_service.update_composer_preferences(
            session_uuid,
            trust_mode="explicit_approve",
            density_default="high",
            actor="user:alice",
        )

        await composer_service_with_real_sessions._run_one_turn_for_test(
            llm=fake_llm_create_blob_then_set_pipeline,
            session_id=result_session_id,
            initial_state=state,
        )

        proposals = await sessions_service.list_composition_proposals(session_uuid)
        proposal_tools = sorted(p.tool_name for p in proposals)
        # create_blob is NOT intercepted — it executes immediately and the
        # resulting blob is available to set_pipeline at proposal-creation
        # time.
        assert "create_blob" not in proposal_tools
        # set_pipeline IS still intercepted — it advances composition state
        # and represents the meaningful operator approval.
        assert "set_pipeline" in proposal_tools

    @pytest.mark.asyncio
    async def test_explicit_approve_invalid_arguments_do_not_crash_compose_loop(
        self,
        composer_service_with_real_sessions: ComposerServiceImpl,
        result_session_id: str,
        fake_llm_set_pipeline_with_misplaced_schema: Any,
    ) -> None:
        """Regression for staging session 100dc5cb-fd66-400b-8041-a1c165cbd8bd:
        under trust_mode=explicit_approve the proposal-interception path
        called redact_tool_call_arguments() with no exception handler. When
        the LLM produced structurally-invalid arguments (schema at the node
        body level instead of inside options), the Pydantic ValidationError
        propagated up, crashed the compose request with HTTP 500, and the
        frontend rendered a bare 'retry' button with no diagnostic.

        After the fix, the redaction failure is caught, the proposal block
        is skipped, and the normal dispatch path produces a clean
        ToolArgumentError that the compose loop surfaces to the model as
        a tool message — letting the model self-correct rather than
        crashing the request."""
        sessions_service = composer_service_with_real_sessions._sessions_service
        assert sessions_service is not None
        session_uuid = UUID(result_session_id)
        state = _empty_state()

        await sessions_service.update_composer_preferences(
            session_uuid,
            trust_mode="explicit_approve",
            density_default="high",
            actor="user:alice",
        )

        # Must not raise — the previous behavior raised a 500 here.
        result = await composer_service_with_real_sessions._run_one_turn_for_test(
            llm=fake_llm_set_pipeline_with_misplaced_schema,
            session_id=result_session_id,
            initial_state=state,
        )

        # No proposal was created — the LLM arguments were invalid.
        proposals = await sessions_service.list_composition_proposals(session_uuid)
        assert proposals == []
        # The set_pipeline outcome surfaces as a tool argument failure that
        # the loop carries into the model's next turn for self-correction.
        outcome_tool_names = [o.call.function.name for o in result.tool_outcomes]
        assert "set_pipeline" in outcome_tool_names
        invalid_outcome = next(o for o in result.tool_outcomes if o.call.function.name == "set_pipeline")
        assert invalid_outcome.error_class is not None
        # State did not advance — invalid arguments are not committed.
        assert invalid_outcome.post_version == state.version

    @pytest.mark.asyncio
    async def test_auto_commit_mutating_tool_preserves_existing_state_mutation_path(
        self,
        composer_service_with_real_sessions: ComposerServiceImpl,
        result_session_id: str,
        fake_llm_one_set_pipeline_tool_call: Any,
    ) -> None:
        """Auto-commit mode still executes mutating tools through the existing path."""
        sessions_service = composer_service_with_real_sessions._sessions_service
        assert sessions_service is not None
        session_uuid = UUID(result_session_id)
        state = _empty_state()

        await sessions_service.update_composer_preferences(
            session_uuid,
            trust_mode="auto_commit",
            density_default="high",
            actor="user:alice",
        )

        result = await composer_service_with_real_sessions._run_one_turn_for_test(
            llm=fake_llm_one_set_pipeline_tool_call,
            session_id=result_session_id,
            initial_state=state,
        )

        proposals = await sessions_service.list_composition_proposals(session_uuid)
        assert proposals == []
        assert result.tool_outcomes[0].post_version > state.version

    @pytest.mark.asyncio
    async def test_single_tool_call_then_text(self) -> None:
        """LLM makes one tool call, then responds with text."""
        catalog = _mock_catalog()
        settings = _make_settings()
        service = ComposerServiceImpl(catalog=catalog, settings=settings)
        state = _empty_state()

        # Turn 1: tool call to set_source
        tool_response = _make_llm_response(
            content=None,
            tool_calls=[
                {
                    "id": "call_1",
                    "name": "set_source",
                    "arguments": {
                        "plugin": "csv",
                        "on_success": "t1",
                        "options": {"path": "/data/blobs/data.csv", "schema": {"mode": "observed"}},
                        "on_validation_failure": "quarantine",
                    },
                }
            ],
        )
        # Turn 2: text response
        text_response = _make_llm_response(content="I've set up a CSV source.")

        passing_preflight = ValidationResult(is_valid=True, checks=[], errors=[])
        with (
            patch.object(service, "_call_llm", new_callable=AsyncMock) as mock_llm,
            patch.object(service, "_runtime_preflight", return_value=passing_preflight),
        ):
            mock_llm.side_effect = [tool_response, text_response]
            result = await service.compose("Use CSV as source", [], state)

        assert result.message == "I've set up a CSV source."
        assert result.state.sources["source"] is not None
        assert result.state.sources["source"].plugin == "csv"
        assert result.state.version == 2

    @pytest.mark.asyncio
    async def test_progress_reports_visible_boundaries_without_tool_payloads(self) -> None:
        """Progress summaries derive from visible lifecycle/tool names, not raw args."""
        catalog = _mock_catalog()
        settings = _make_settings()
        service = ComposerServiceImpl(catalog=catalog, settings=settings)
        state = _empty_state()
        progress_events: list[ComposerProgressEvent] = []

        async def record_progress(event: ComposerProgressEvent) -> None:
            progress_events.append(event)

        tool_response = _make_llm_response(
            content=None,
            tool_calls=[
                {
                    "id": "call_secret",
                    "name": "validate_secret_ref",
                    "arguments": {"name": "OPENROUTER_TOP_SECRET_VALUE"},
                }
            ],
        )
        text_response = _make_llm_response(content="I checked the available credentials.")

        async def inline_to_thread(func: Any, /, *args: Any, **kwargs: Any) -> Any:
            return func(*args, **kwargs)

        with (
            patch.object(service, "_call_llm", new_callable=AsyncMock) as mock_llm,
            patch("elspeth.web.composer.service.asyncio.to_thread", side_effect=inline_to_thread),
        ):
            mock_llm.side_effect = [tool_response, text_response]
            result = await service.compose(
                "Use my OpenRouter secret",
                [],
                state,
                progress=record_progress,
            )

        assert result.message == "I checked the available credentials."
        phases = [event.phase for event in progress_events]
        assert phases[:2] == ["starting", "calling_model"]
        assert "using_tools" in phases
        assert "complete" in phases

        progress_text = "\n".join(line for event in progress_events for line in (event.headline, event.likely_next or "", *event.evidence))
        assert "OPENROUTER_TOP_SECRET_VALUE" not in progress_text
        assert '"name"' not in progress_text
        assert "checking available secret references" in progress_text.lower()

    @pytest.mark.asyncio
    async def test_inline_complete_pipeline_replay_uses_atomic_tool_shape(self, tmp_path: Path) -> None:
        """Fake-model replay for simple inline-data builds stays provider-bounded."""
        catalog = _mock_catalog()
        engine, session_id = _session_engine_with_session()
        settings = _make_settings(data_dir=tmp_path)
        service = ComposerServiceImpl(
            catalog=catalog,
            settings=settings,
            sessions_service=_test_sessions_service(engine, tmp_path),
            session_engine=engine,
        )
        # Stub the EARLY advisory checkpoint (fires on the empty->non-empty
        # transition this test drives) so it makes no advisor LLM call — this
        # test is about the atomic tool shape and its `llm_calls == 3`
        # bookkeeping, not the advisor pass.
        service._run_advisor_checkpoint = AsyncMock(  # type: ignore[method-assign]
            return_value=AdvisorCheckpointVerdict(ok=True, blocking=False, findings_text="CLEAN")
        )
        state = _empty_state()
        user_message_content = "I want a pipeline that takes the string 'hello' and appends ' world' to it."
        user_message_id = _insert_user_message(engine, session_id, user_message_content)
        output_path = tmp_path / "outputs" / "append.csv"
        pipeline_args = {
            "source": {
                "plugin": "text",
                "on_success": "source_out",
                "options": {
                    "column": "text",
                    "schema": {"mode": "observed", "guaranteed_fields": ["text"]},
                },
                "inline_blob": {
                    "filename": "input.txt",
                    "mime_type": "text/plain",
                    "content": "hello",
                },
                "on_validation_failure": "discard",
            },
            "nodes": [
                {
                    "id": "append_world",
                    "node_type": "transform",
                    "plugin": "value_transform",
                    "input": "source_out",
                    "on_success": "main",
                    "on_error": "discard",
                    "options": {
                        "schema": {
                            "mode": "observed",
                            "guaranteed_fields": ["text"],
                            "required_fields": ["text"],
                        },
                        "operations": [{"target": "text", "expression": "row['text'] + ' world'"}],
                    },
                }
            ],
            "edges": [
                {"id": "source_to_append", "from_node": "source", "to_node": "append_world", "edge_type": "on_success"},
                {"id": "append_to_main", "from_node": "append_world", "to_node": "main", "edge_type": "on_success"},
            ],
            "outputs": [
                {
                    "sink_name": "main",
                    "plugin": "csv",
                    "options": {
                        "path": str(output_path),
                        "schema": {"mode": "observed", "required_fields": ["text"]},
                        "mode": "write",
                        "collision_policy": "auto_increment",
                    },
                    "on_write_failure": "discard",
                }
            ],
            "metadata": {"name": "Append literal text"},
        }
        build_turn = _make_llm_response(
            tool_calls=[
                {
                    "id": "call_build",
                    "name": "set_pipeline",
                    "arguments": pipeline_args,
                }
            ],
        )
        preview_turn = _make_llm_response(tool_calls=[{"id": "call_preview", "name": "preview_pipeline", "arguments": {}}])
        final_turn = _make_llm_response(content="Pipeline configured.")

        with (
            patch.object(service, "_call_llm", new_callable=AsyncMock) as mock_llm,
            patch.object(
                service,
                "_cached_runtime_preflight",
                new_callable=AsyncMock,
                return_value=ValidationResult(is_valid=True, checks=[], errors=[]),
            ),
        ):
            mock_llm.side_effect = [build_turn, preview_turn, final_turn]
            result = await service.compose(
                user_message_content,
                [],
                state,
                session_id=session_id,
                user_message_id=user_message_id,
            )

        assert result.message == "Pipeline configured."
        assert mock_llm.call_count == 3
        tool_names = [inv.tool_name for inv in result.tool_invocations]
        assert tool_names == ["set_pipeline", "preview_pipeline"]
        assert "create_blob" not in tool_names
        assert "set_source_from_blob" not in tool_names
        assert "upsert_node" not in tool_names
        assert "set_output" not in tool_names
        assert len(result.llm_calls) == 3
        assert result.state.sources["source"] is not None
        assert "blob_ref" in result.state.sources["source"].options


class TestComposerMultiTurnToolCalls:
    @pytest.mark.asyncio
    async def test_multi_turn_state_accumulates(self) -> None:
        """Multiple tool calls across turns — state accumulates."""
        catalog = _mock_catalog()
        settings = _make_settings()
        service = ComposerServiceImpl(catalog=catalog, settings=settings)
        state = _empty_state()

        # Turn 1: set_source
        turn1 = _make_llm_response(
            tool_calls=[
                {
                    "id": "call_1",
                    "name": "set_source",
                    "arguments": {
                        "plugin": "csv",
                        "on_success": "t1",
                        "options": {"path": "/data/blobs/input.csv", "schema": {"mode": "observed"}},
                        "on_validation_failure": "quarantine",
                    },
                }
            ],
        )
        # Turn 2: set_metadata
        turn2 = _make_llm_response(
            tool_calls=[
                {
                    "id": "call_2",
                    "name": "set_metadata",
                    "arguments": {"patch": {"name": "My Pipeline"}},
                }
            ],
        )
        # Turn 3: text
        turn3 = _make_llm_response(content="Pipeline configured.")

        passing_preflight = ValidationResult(is_valid=True, checks=[], errors=[])
        with (
            patch.object(service, "_call_llm", new_callable=AsyncMock) as mock_llm,
            patch.object(service, "_runtime_preflight", return_value=passing_preflight),
        ):
            mock_llm.side_effect = [turn1, turn2, turn3]
            result = await service.compose("Build a pipeline", [], state)

        assert result.state.sources["source"] is not None
        assert result.state.metadata.name == "My Pipeline"
        assert result.state.version == 3  # two mutations


class TestComposerConvergence:
    @pytest.mark.asyncio
    async def test_discovery_budget_exceeded_raises(self) -> None:
        """Discovery-only turns exhaust the discovery budget."""
        catalog = _mock_catalog()
        settings = _make_settings(composer_max_discovery_turns=1)
        service = ComposerServiceImpl(catalog=catalog, settings=settings)
        state = _empty_state()

        # Two different discovery tools to avoid cache hits
        disc1 = _make_llm_response(
            tool_calls=[{"id": "c1", "name": "list_sources", "arguments": {}}],
        )
        disc2 = _make_llm_response(
            tool_calls=[{"id": "c2", "name": "list_transforms", "arguments": {}}],
        )

        with patch.object(service, "_call_llm", new_callable=AsyncMock) as mock_llm:
            mock_llm.side_effect = [disc1, disc2]
            with pytest.raises(ComposerConvergenceError) as exc_info:
                await service.compose("Loop forever", [], state)
            assert exc_info.value.budget_exhausted == "discovery"

    @pytest.mark.asyncio
    async def test_composition_budget_exceeded_raises(self) -> None:
        """Mutation turns exhaust the composition budget."""
        catalog = _mock_catalog()
        settings = _make_settings(composer_max_composition_turns=1)
        service = ComposerServiceImpl(catalog=catalog, settings=settings)
        state = _empty_state()

        mut = _make_llm_response(
            tool_calls=[
                {
                    "id": "c1",
                    "name": "set_metadata",
                    "arguments": {"patch": {"name": "test"}},
                }
            ],
        )
        # Bonus call also returns tool calls — convergence error
        mut2 = _make_llm_response(
            tool_calls=[
                {
                    "id": "c2",
                    "name": "set_metadata",
                    "arguments": {"patch": {"name": "test2"}},
                }
            ],
        )

        with patch.object(service, "_call_llm", new_callable=AsyncMock) as mock_llm:
            mock_llm.side_effect = [mut, mut2]
            with pytest.raises(ComposerConvergenceError) as exc_info:
                await service.compose("Keep mutating", [], state)
            assert exc_info.value.budget_exhausted == "composition"

    @pytest.mark.asyncio
    async def test_self_correction_on_final_composition_turn_succeeds(self) -> None:
        """B-4D-3: LLM makes mutation on final turn, then text on bonus call."""
        catalog = _mock_catalog()
        settings = _make_settings(composer_max_composition_turns=1)
        service = ComposerServiceImpl(catalog=catalog, settings=settings)
        state = _empty_state()

        mut = _make_llm_response(
            tool_calls=[
                {
                    "id": "c1",
                    "name": "set_metadata",
                    "arguments": {"patch": {"name": "Final"}},
                }
            ],
        )
        text = _make_llm_response(content="Done after final correction.")

        passing_preflight = ValidationResult(is_valid=True, checks=[], errors=[])
        with (
            patch.object(service, "_call_llm", new_callable=AsyncMock) as mock_llm,
            patch.object(service, "_runtime_preflight", return_value=passing_preflight),
        ):
            mock_llm.side_effect = [mut, text]
            result = await service.compose("Do it", [], state)

        assert result.message == "Done after final correction."
        assert result.state.metadata.name == "Final"

    @pytest.mark.asyncio
    async def test_mixed_turns_charge_correct_budgets(self) -> None:
        """Mixed discovery/mutation turns are classified independently.

        Discovery turns charge discovery budget, mutation turns charge
        composition budget. Neither exhausts the other.
        """
        catalog = _mock_catalog()
        settings = _make_settings(
            composer_max_composition_turns=2,
            composer_max_discovery_turns=2,
        )
        service = ComposerServiceImpl(catalog=catalog, settings=settings)
        state = _empty_state()

        # Turn 1: discovery (list_sources) — discovery counter = 1
        disc = _make_llm_response(
            tool_calls=[{"id": "c1", "name": "list_sources", "arguments": {}}],
        )
        # Turn 2: mutation (set_metadata) — composition counter = 1
        mut = _make_llm_response(
            tool_calls=[
                {
                    "id": "c2",
                    "name": "set_metadata",
                    "arguments": {"patch": {"name": "Works"}},
                }
            ],
        )
        # Turn 3: text response — loop terminates
        text = _make_llm_response(content="Done.")

        passing_preflight = ValidationResult(is_valid=True, checks=[], errors=[])
        with (
            patch.object(service, "_call_llm", new_callable=AsyncMock) as mock_llm,
            patch.object(service, "_runtime_preflight", return_value=passing_preflight),
        ):
            mock_llm.side_effect = [disc, mut, text]
            result = await service.compose("Build", [], state)

        assert result.message == "Done."
        assert result.state.metadata.name == "Works"


class TestFailedMutationBudgetClassification:
    """Failed mutation tool calls must charge composition budget, not discovery."""

    @pytest.mark.asyncio
    async def test_failed_mutation_charges_composition_budget(self) -> None:
        """A mutation tool that fails with KeyError/TypeError should exhaust
        composition budget, not discovery budget."""
        catalog = _mock_catalog()
        settings = _make_settings(
            composer_max_composition_turns=1,
            composer_max_discovery_turns=10,
        )
        service = ComposerServiceImpl(catalog=catalog, settings=settings)
        state = _empty_state()

        # Turn 1: set_source with missing required key → KeyError
        # This is a mutation tool, so even though it fails, it should
        # charge composition budget (1/1 → exhausted).
        bad_mutation = _make_llm_response(
            tool_calls=[
                {
                    "id": "c1",
                    "name": "set_source",
                    "arguments": {"plugin": "csv"},  # missing on_success
                }
            ],
        )
        # Bonus call (composition exhausted gives one last chance) also
        # returns a tool call → convergence error
        bad_mutation2 = _make_llm_response(
            tool_calls=[
                {
                    "id": "c2",
                    "name": "set_source",
                    "arguments": {"plugin": "csv"},
                }
            ],
        )

        with patch.object(service, "_call_llm", new_callable=AsyncMock) as mock_llm:
            mock_llm.side_effect = [bad_mutation, bad_mutation2]
            with pytest.raises(ComposerConvergenceError) as exc_info:
                await service.compose("Setup source", [], state)
            assert exc_info.value.budget_exhausted == "composition"

    @pytest.mark.asyncio
    async def test_failed_mutation_json_parse_charges_composition_budget(self) -> None:
        """Mutation tool with unparseable JSON arguments should still
        charge composition budget."""
        catalog = _mock_catalog()
        settings = _make_settings(
            composer_max_composition_turns=1,
            composer_max_discovery_turns=10,
        )
        service = ComposerServiceImpl(catalog=catalog, settings=settings)
        state = _empty_state()

        # Build a tool call with invalid JSON manually
        call = FakeToolCall(
            id="c1",
            function=FakeFunction(
                name="set_source",
                arguments="{invalid json",
            ),
        )
        msg = FakeMessage(content=None, tool_calls=[call])
        response = FakeLLMResponse(choices=[FakeChoice(message=msg)])

        # Bonus call also fails
        call2 = FakeToolCall(
            id="c2",
            function=FakeFunction(
                name="set_source",
                arguments="{still invalid",
            ),
        )
        msg2 = FakeMessage(content=None, tool_calls=[call2])
        response2 = FakeLLMResponse(choices=[FakeChoice(message=msg2)])

        with patch.object(service, "_call_llm", new_callable=AsyncMock) as mock_llm:
            mock_llm.side_effect = [response, response2]
            with pytest.raises(ComposerConvergenceError) as exc_info:
                await service.compose("Setup source", [], state)
            assert exc_info.value.budget_exhausted == "composition"

    @pytest.mark.asyncio
    async def test_failed_discovery_does_not_charge_composition_budget(self) -> None:
        """A failed discovery tool should still charge discovery budget,
        not composition budget."""
        catalog = _mock_catalog()
        settings = _make_settings(
            composer_max_composition_turns=10,
            composer_max_discovery_turns=1,
        )
        service = ComposerServiceImpl(catalog=catalog, settings=settings)
        state = _empty_state()

        # list_sources with invalid JSON → still a discovery turn
        call = FakeToolCall(
            id="c1",
            function=FakeFunction(
                name="list_sources",
                arguments="{bad json",
            ),
        )
        msg = FakeMessage(content=None, tool_calls=[call])
        response = FakeLLMResponse(choices=[FakeChoice(message=msg)])

        # Turn 2: another discovery call
        disc2 = _make_llm_response(
            tool_calls=[{"id": "c2", "name": "list_transforms", "arguments": {}}],
        )

        with patch.object(service, "_call_llm", new_callable=AsyncMock) as mock_llm:
            mock_llm.side_effect = [response, disc2]
            with pytest.raises(ComposerConvergenceError) as exc_info:
                await service.compose("Explore", [], state)
            assert exc_info.value.budget_exhausted == "discovery"


class TestComposerErrorHandling:
    @pytest.mark.asyncio
    async def test_unknown_tool_returns_error_to_llm(self) -> None:
        """Unknown tool name returns error message, LLM can retry."""
        catalog = _mock_catalog()
        settings = _make_settings()
        service = ComposerServiceImpl(catalog=catalog, settings=settings)
        state = _empty_state()

        # Turn 1: invalid tool
        bad_call = _make_llm_response(
            tool_calls=[
                {
                    "id": "call_bad",
                    "name": "nonexistent_tool",
                    "arguments": {},
                }
            ],
        )
        # Turn 2: text response (self-corrected)
        text = _make_llm_response(content="Sorry, let me try again.")

        with patch.object(service, "_call_llm", new_callable=AsyncMock) as mock_llm:
            mock_llm.side_effect = [bad_call, text]
            result = await service.compose("Do something", [], state)

        assert result.message == "Sorry, let me try again."
        # State unchanged — the bad tool call didn't modify anything
        assert result.state.version == 1

    @pytest.mark.asyncio
    async def test_malformed_arguments_returns_error(self) -> None:
        """Malformed tool arguments return error, not crash."""
        catalog = _mock_catalog()
        settings = _make_settings()
        service = ComposerServiceImpl(catalog=catalog, settings=settings)
        state = _empty_state()

        # Turn 1: set_source with missing required field
        bad_call = _make_llm_response(
            tool_calls=[
                {
                    "id": "call_bad",
                    "name": "set_source",
                    "arguments": {"plugin": "csv"},  # missing on_success
                }
            ],
        )
        # Turn 2: text
        text = _make_llm_response(content="Fixed.")

        with patch.object(service, "_call_llm", new_callable=AsyncMock) as mock_llm:
            mock_llm.side_effect = [bad_call, text]
            result = await service.compose("Setup", [], state)

        _assert_no_mutation_empty_state_blocker(
            result,
            tool_name="set_source",
            expected_detail="on_success, options, on_validation_failure",
        )

    @pytest.mark.asyncio
    async def test_wrong_type_tool_arg_returns_error(self) -> None:
        """ToolArgumentError from Tier 3 type guard in tool handler is caught, not crash.

        Tool handlers validate LLM-provided argument types at the Tier 3
        boundary, raising ToolArgumentError for wrong types (e.g. int where
        str expected). The compose loop catches this typed exception and
        feeds the error back to the LLM so it can retry with a corrected
        argument.
        """
        catalog = _mock_catalog()
        settings = _make_settings()
        service = ComposerServiceImpl(catalog=catalog, settings=settings)
        state = _empty_state()

        # Turn 1: tool call that triggers ToolArgumentError from Tier 3 type guard
        bad_call = _make_llm_response(
            tool_calls=[
                {
                    "id": "call_bad",
                    "name": "set_source",
                    "arguments": {
                        "plugin": "csv",
                        "on_success": "out",
                        "options": {},
                        "on_validation_failure": "quarantine",
                    },
                }
            ],
        )
        # Turn 2: LLM self-corrects
        text = _make_llm_response(content="Fixed.")

        with (
            patch(
                "elspeth.web.composer.tool_batch.execute_tool",
                side_effect=ToolArgumentError(
                    argument="content",
                    expected="a string",
                    actual_type="int",
                ),
            ),
            patch.object(service, "_call_llm", new_callable=AsyncMock) as mock_llm,
        ):
            mock_llm.side_effect = [bad_call, text]
            result = await service.compose("Setup", [], state)

        _assert_no_mutation_empty_state_blocker(
            result,
            tool_name="set_source",
            expected_detail="'content' must be a string, got int",
        )

    @pytest.mark.asyncio
    async def test_malformed_set_pipeline_missing_top_level_required_field_returns_error(self) -> None:
        """Top-level required field omission in set_pipeline returns a Tier-3 arg error.

        Renamed from test_malformed_set_pipeline_nested_required_field_returns_error
        — the body always tested top-level ``source.plugin`` omission, not a nested
        required field. Coverage of nested-optional + inner-required semantics lives
        in the two tests below.
        """
        catalog = _mock_catalog()
        settings = _make_settings()
        service = ComposerServiceImpl(catalog=catalog, settings=settings)
        state = _empty_state()

        bad_call = _make_llm_response(
            tool_calls=[
                {
                    "id": "call_bad",
                    "name": "set_pipeline",
                    "arguments": {
                        "source": {
                            "on_success": "source_out",
                            "options": {"path": "/data/in.csv", "schema": {"mode": "observed"}},
                        },
                        "nodes": [
                            {
                                "id": "t1",
                                "node_type": "transform",
                                "plugin": "passthrough",
                                "input": "source_out",
                                "on_success": "main",
                                "options": {},
                            }
                        ],
                        "edges": [
                            {
                                "id": "e1",
                                "from_node": "source",
                                "to_node": "t1",
                                "edge_type": "on_success",
                            }
                        ],
                        "outputs": [
                            {
                                "sink_name": "main",
                                "plugin": "csv",
                                "options": {"path": "/data/out.csv", "schema": {"mode": "observed"}},
                            }
                        ],
                    },
                }
            ],
        )
        text = _make_llm_response(content="Recovered.")

        with patch.object(service, "_call_llm", new_callable=AsyncMock) as mock_llm:
            mock_llm.side_effect = [bad_call, text]
            result = await service.compose("Setup", [], state)

        _assert_no_mutation_empty_state_blocker(
            result,
            tool_name="set_pipeline",
            expected_detail="source.plugin",
        )
        tool_msg = mock_llm.call_args_list[1][0][0][-1]
        error_content = json.loads(tool_msg["content"])
        assert "source.plugin" in error_content["error"]
        assert "missing required" in error_content["error"].lower()
        # Regression guard: the conditional-on-presence walker must NOT
        # surface false-positive errors for the optional ``source.inline_blob``
        # branch when no inline blob was supplied. See elspeth-4e79436719 §Bug A.
        assert "inline_blob" not in error_content["error"]

    @pytest.mark.asyncio
    async def test_set_pipeline_without_inline_blob_passes_predispatch_validation(self) -> None:
        """No-inline set_pipeline payloads must reach the handler.

        Regression guard for elspeth-4e79436719 Bug A: the required-paths
        walker previously lifted ``source.inline_blob.{filename,mime_type,content}``
        into top-level required paths because it recursed into nested
        ``properties`` blocks unconditionally — even when the containing
        property (``inline_blob``) was optional at its parent. Correct
        JSON-Schema semantics: nested ``required`` applies only when the
        containing object is present in the value.
        """
        catalog = _mock_catalog()
        settings = _make_settings()
        service = ComposerServiceImpl(catalog=catalog, settings=settings)
        state = _empty_state()

        good_call = _make_llm_response(
            tool_calls=[
                {
                    "id": "call_ok",
                    "name": "set_pipeline",
                    "arguments": {
                        "sources": {
                            "primary": {
                                "plugin": "csv",
                                "on_success": "t1",
                                "options": {"path": "/data/blobs/in.csv", "schema": {"mode": "observed"}},
                            }
                        },
                        "nodes": [
                            {
                                "id": "t1",
                                "node_type": "transform",
                                "plugin": "passthrough",
                                "input": "main",
                                "on_success": "main",
                                "options": {},
                            }
                        ],
                        "edges": [
                            {
                                "id": "e1",
                                "from_node": "source",
                                "to_node": "t1",
                                "edge_type": "on_success",
                            }
                        ],
                        "outputs": [
                            {
                                "sink_name": "main",
                                "plugin": "csv",
                                "options": {"path": "/data/out.csv", "schema": {"mode": "observed"}},
                            }
                        ],
                    },
                }
            ],
        )
        text = _make_llm_response(content="Pipeline ready.")
        passing_preflight = ValidationResult(is_valid=True, checks=[], errors=[])

        with (
            patch.object(service, "_call_llm", new_callable=AsyncMock) as mock_llm,
            patch.object(service, "_runtime_preflight", return_value=passing_preflight),
        ):
            mock_llm.side_effect = [good_call, text]
            await service.compose("Setup", [], state)

        # The pre-dispatch validator must not have rejected the call.
        # Verify by inspecting the tool-result message content: the
        # MissingRequiredPaths code path produces a JSON object whose
        # ``error`` field starts with ``Tool 'set_pipeline' missing required
        # argument(s):``. A no-inline payload must not trigger that path,
        # regardless of what the tool handler returns afterwards.
        tool_msg = mock_llm.call_args_list[1][0][0][-1]
        assert tool_msg["role"] == "tool"
        try:
            error_content = json.loads(tool_msg["content"])
            error_text = error_content.get("error", "") if isinstance(error_content, dict) else ""
        except json.JSONDecodeError:
            error_text = ""
        assert "missing required argument" not in error_text.lower()
        assert "inline_blob" not in error_text

    @pytest.mark.asyncio
    async def test_set_pipeline_with_partial_inline_blob_returns_tier3_arg_error(self) -> None:
        """When ``inline_blob`` is present but incomplete, the inner required fields ARE enforced.

        Conditional-on-presence semantics: an absent optional object skips
        its inner ``required`` checks; a present-but-incomplete optional
        object surfaces them as Tier-3 arg errors. See elspeth-4e79436719
        §Bug A.
        """
        catalog = _mock_catalog()
        settings = _make_settings()
        service = ComposerServiceImpl(catalog=catalog, settings=settings)
        state = _empty_state()

        partial_call = _make_llm_response(
            tool_calls=[
                {
                    "id": "call_partial",
                    "name": "set_pipeline",
                    "arguments": {
                        "source": {
                            "plugin": "csv",
                            "on_success": "t1",
                            "options": {"path": "/data/blobs/in.csv", "schema": {"mode": "observed"}},
                            "inline_blob": {"filename": "data.csv"},
                        },
                        "nodes": [
                            {
                                "id": "t1",
                                "node_type": "transform",
                                "plugin": "passthrough",
                                "input": "main",
                                "on_success": "main",
                                "options": {},
                            }
                        ],
                        "edges": [
                            {
                                "id": "e1",
                                "from_node": "source",
                                "to_node": "t1",
                                "edge_type": "on_success",
                            }
                        ],
                        "outputs": [
                            {
                                "sink_name": "main",
                                "plugin": "csv",
                                "options": {"path": "/data/out.csv", "schema": {"mode": "observed"}},
                            }
                        ],
                    },
                }
            ],
        )
        text = _make_llm_response(content="Adjusted.")

        with patch.object(service, "_call_llm", new_callable=AsyncMock) as mock_llm:
            mock_llm.side_effect = [partial_call, text]
            result = await service.compose("Setup", [], state)

        _assert_no_mutation_empty_state_blocker(
            result,
            tool_name="set_pipeline",
            expected_detail="source.inline_blob.mime_type, source.inline_blob.content",
        )
        tool_msg = mock_llm.call_args_list[1][0][0][-1]
        error_content = json.loads(tool_msg["content"])
        assert "source.inline_blob.mime_type" in error_content["error"]
        assert "source.inline_blob.content" in error_content["error"]
        # filename WAS supplied — must not be reported.
        assert "source.inline_blob.filename" not in error_content["error"]

    @pytest.mark.asyncio
    async def test_internal_key_error_is_not_swallowed(self) -> None:
        """KeyError from tool handler internals must crash, not be sent to LLM.

        Previously, KeyError from missing LLM arguments and KeyError from
        internal bugs were both caught and fed back to the LLM. Internal
        bugs should crash immediately — the LLM cannot self-correct our code.
        """
        catalog = _mock_catalog()
        settings = _make_settings()
        service = ComposerServiceImpl(catalog=catalog, settings=settings)
        state = _empty_state()

        # Provide all required arguments so pre-validation passes,
        # but patch execute_tool to raise a KeyError from "internal logic"
        valid_call = _make_llm_response(
            tool_calls=[
                {
                    "id": "c1",
                    "name": "set_source",
                    "arguments": {
                        "plugin": "csv",
                        "on_success": "out",
                        "options": {},
                        "on_validation_failure": "quarantine",
                    },
                }
            ],
        )

        with (
            patch.object(service, "_call_llm", new_callable=AsyncMock) as mock_llm,
            patch(
                "elspeth.web.composer.tool_batch.execute_tool",
                side_effect=KeyError("internal_state_key"),
            ),
        ):
            mock_llm.return_value = valid_call
            with pytest.raises(ComposerPluginCrashError) as exc_info:
                await service.compose("Setup", [], state)
        # The underlying KeyError is preserved on the wrapper so callers
        # (server logs, route handler, capture_logs assertions) can still
        # identify the original plugin-internal class.
        assert isinstance(exc_info.value.original_exc, KeyError)
        assert exc_info.value.exc_class == "KeyError"
        assert "internal_state_key" in str(exc_info.value.original_exc)

    @pytest.mark.asyncio
    async def test_missing_args_error_message_lists_keys(self) -> None:
        """Missing required arguments should produce a clear error listing the keys."""
        catalog = _mock_catalog()
        settings = _make_settings()
        service = ComposerServiceImpl(catalog=catalog, settings=settings)
        state = _empty_state()

        # set_source requires plugin, on_success, options, on_validation_failure
        bad_call = _make_llm_response(
            tool_calls=[
                {
                    "id": "c1",
                    "name": "set_source",
                    "arguments": {"plugin": "csv"},  # missing 3 required args
                }
            ],
        )
        text = _make_llm_response(content="Ok.")

        with patch.object(service, "_call_llm", new_callable=AsyncMock) as mock_llm:
            mock_llm.side_effect = [bad_call, text]
            await service.compose("Setup", [], state)

        # Verify the error message sent back to the LLM mentions the missing keys
        tool_msg = mock_llm.call_args_list[1][0][0][-1]  # last message in second call
        error_content = json.loads(tool_msg["content"])
        assert "on_success" in error_content["error"]
        assert "missing required" in error_content["error"].lower()

    @pytest.mark.asyncio
    @pytest.mark.parametrize("arguments", [[], None, "oops"])
    async def test_top_level_non_object_mutation_arguments_return_tool_error(self, arguments: Any) -> None:
        """Non-object mutation arguments are Tier-3 tool errors, not plugin crashes."""
        catalog = _mock_catalog()
        settings = _make_settings()
        service = ComposerServiceImpl(catalog=catalog, settings=settings)
        state = _empty_state()

        bad_call = _make_llm_response(
            tool_calls=[
                {
                    "id": "call_bad",
                    "name": "set_source",
                    "arguments": arguments,
                }
            ],
        )
        text = _make_llm_response(content="Recovered.")

        with (
            patch(
                "elspeth.web.composer.tool_batch.execute_tool",
                wraps=_execute_tool,
            ) as mock_execute_tool,
            patch.object(service, "_call_llm", new_callable=AsyncMock) as mock_llm,
        ):
            mock_llm.side_effect = [bad_call, text]
            result = await service.compose("Setup", [], state)

        _assert_no_mutation_empty_state_blocker(
            result,
            tool_name="set_source",
            expected_detail="arguments (",
        )
        mock_execute_tool.assert_not_called()
        tool_msg = mock_llm.call_args_list[1][0][0][-1]
        error_content = json.loads(tool_msg["content"])
        assert "arguments must be a JSON object" in error_content["error"]

    @pytest.mark.asyncio
    @pytest.mark.parametrize("arguments", [[], None, "oops"])
    async def test_top_level_non_object_discovery_arguments_return_tool_error(self, arguments: Any) -> None:
        """Non-object discovery arguments are rejected before cache lookup or execution."""
        catalog = _mock_catalog()
        settings = _make_settings()
        service = ComposerServiceImpl(catalog=catalog, settings=settings)
        state = _empty_state()

        bad_call = _make_llm_response(
            tool_calls=[
                {
                    "id": "call_bad",
                    "name": "list_sources",
                    "arguments": arguments,
                }
            ],
        )
        text = _make_llm_response(content="Recovered.")

        with (
            patch(
                "elspeth.web.composer.tool_batch.execute_tool",
                wraps=_execute_tool,
            ) as mock_execute_tool,
            patch.object(service, "_call_llm", new_callable=AsyncMock) as mock_llm,
        ):
            mock_llm.side_effect = [bad_call, text]
            result = await service.compose("Explore", [], state)

        assert result.message == "Recovered."
        mock_execute_tool.assert_not_called()
        tool_msg = mock_llm.call_args_list[1][0][0][-1]
        error_content = json.loads(tool_msg["content"])
        assert "arguments must be a JSON object" in error_content["error"]


class TestProviderCacheTokenAudit:
    """Audit-write capture of provider prompt-cache statistics (elspeth-4e79436719 Bug C).

    Provider responses can carry cache-hit information in two distinct
    shapes:
    - OpenAI / OpenRouter: ``usage.prompt_tokens_details.cached_tokens``
    - Anthropic: ``usage.cache_creation_input_tokens`` and
      ``usage.cache_read_input_tokens`` as siblings on usage

    Both must land on the ``ComposerLLMCall`` audit record with their
    canonical names, so the audit DB records what the provider actually
    reported. A missing field must remain ``None`` per the
    "absence as evidence" rule from CLAUDE.md.

    LiteLLM-shape deduplication: when LiteLLM is the wire to an
    Anthropic-family provider (Anthropic direct, Bedrock-Claude,
    Vertex-Claude/Gemini), it populates BOTH the nested OpenAI shape and
    the Anthropic siblings on the same response, deriving
    ``prompt_tokens_details.cached_tokens`` from the sibling
    ``cache_read_input_tokens``. The two are aliases, not independent
    signals — recording both would double-count cache hits in the audit
    sidecar, and creation-only responses (which carry ``cached_tokens=0``
    by LiteLLM default) would fabricate a zero into ``cached_prompt_tokens``
    that the provider never asserted. Presence of an Anthropic sibling is
    the provenance signal that the nested shape is LiteLLM-synthesized.
    """

    @staticmethod
    def _response_with_usage(usage: dict[str, Any]) -> FakeLLMResponse:
        """Build a fake response carrying a usage object as a Mapping.

        FakeLLMResponse only carries .choices; add .usage so the audit
        extractor's Mapping branch fires. Using a separate dataclass keeps
        the typed FakeLLMResponse intact for the rest of the test suite.
        """

        @dataclass
        class FakeResponseWithUsage:
            choices: list[FakeChoice]
            usage: dict[str, Any]
            model: str = "openrouter/openai/gpt-5.5"
            id: str = "chatcmpl-test"

        text = _make_llm_response(content="Done.")
        return FakeResponseWithUsage(choices=text.choices, usage=usage)  # type: ignore[return-value]

    @pytest.mark.asyncio
    async def test_openai_nested_cached_tokens_lands_on_audit_record(self) -> None:
        from elspeth.web.composer.llm_response_parsing import token_usage_from_response

        response = self._response_with_usage(
            {
                "prompt_tokens": 1200,
                "completion_tokens": 80,
                "total_tokens": 1280,
                "prompt_tokens_details": {"cached_tokens": 1024},
            }
        )
        usage = token_usage_from_response(response)
        assert usage.prompt_tokens == 1200
        assert usage.cached_prompt_tokens == 1024
        assert usage.cache_creation_input_tokens is None
        assert usage.cache_read_input_tokens is None

    @pytest.mark.asyncio
    async def test_anthropic_sibling_cache_fields_land_on_audit_record(self) -> None:
        from elspeth.web.composer.llm_response_parsing import token_usage_from_response

        response = self._response_with_usage(
            {
                "prompt_tokens": 8200,
                "completion_tokens": 120,
                "cache_creation_input_tokens": 7000,
                "cache_read_input_tokens": 1100,
            }
        )
        usage = token_usage_from_response(response)
        assert usage.cache_creation_input_tokens == 7000
        assert usage.cache_read_input_tokens == 1100
        assert usage.cached_prompt_tokens is None

    @pytest.mark.asyncio
    async def test_litellm_normalized_anthropic_does_not_double_record_cache_hit(self) -> None:
        """LiteLLM-normalized Anthropic-family response: nested shape is suppressed.

        Mirrors the wire shape produced by
        ``litellm/llms/anthropic/chat/transformation.py:1294-1317`` —
        ``prompt_tokens_details.cached_tokens`` and ``cache_read_input_tokens``
        carry the SAME value because LiteLLM derives the former from the
        latter. The audit row must record only the Anthropic-shape signal.
        """
        from elspeth.web.composer.llm_response_parsing import token_usage_from_response

        response = self._response_with_usage(
            {
                "prompt_tokens": 8200,
                "completion_tokens": 120,
                "prompt_tokens_details": {"cached_tokens": 1100},
                "cache_creation_input_tokens": 7000,
                "cache_read_input_tokens": 1100,
            }
        )
        usage = token_usage_from_response(response)
        assert usage.cache_creation_input_tokens == 7000
        assert usage.cache_read_input_tokens == 1100
        assert usage.cached_prompt_tokens is None

    @pytest.mark.asyncio
    async def test_litellm_creation_only_does_not_leak_zero_into_cached_prompt_tokens(self) -> None:
        """LiteLLM creation-only Anthropic response: nested cached_tokens=0 is suppressed.

        ``PromptTokensDetailsWrapper`` preserves ``cached_tokens=0`` (verified
        empirically), so a cache-creation-only Anthropic call still emits a
        nested shape with ``cached_tokens=0``. Without dedup, the audit row
        would carry ``cached_prompt_tokens=0`` — a fabricated zero that
        misleads the auditor into thinking the provider reported a zero-hit
        cache read, when in fact the provider only reported cache creation.
        """
        from elspeth.web.composer.llm_response_parsing import token_usage_from_response

        response = self._response_with_usage(
            {
                "prompt_tokens": 7000,
                "completion_tokens": 80,
                "prompt_tokens_details": {"cached_tokens": 0},
                "cache_creation_input_tokens": 7000,
                "cache_read_input_tokens": 0,
            }
        )
        usage = token_usage_from_response(response)
        assert usage.cache_creation_input_tokens == 7000
        assert usage.cache_read_input_tokens == 0
        assert usage.cached_prompt_tokens is None

    @pytest.mark.asyncio
    async def test_litellm_normalized_dual_shape_is_deduped_on_attribute_branch(self) -> None:
        """Pydantic-shaped (attribute) usage object also dedups when siblings present.

        Real LiteLLM responses are Pydantic ``Usage`` objects, not Mappings.
        Verifies the elif branch in ``token_usage_from_response`` honors the
        same dedup rule: nested ``prompt_tokens_details.cached_tokens`` is
        dropped when an Anthropic sibling is present on the attribute object.
        """
        from elspeth.web.composer.llm_response_parsing import token_usage_from_response

        @dataclass
        class FakePromptTokensDetails:
            cached_tokens: int | None

        @dataclass
        class FakePydanticUsage:
            prompt_tokens: int
            completion_tokens: int
            total_tokens: int
            prompt_tokens_details: FakePromptTokensDetails
            cache_creation_input_tokens: int | None
            cache_read_input_tokens: int | None

        @dataclass
        class FakeResponseWithPydanticUsage:
            choices: list[FakeChoice]
            usage: FakePydanticUsage
            model: str = "anthropic/claude-3-5-sonnet"
            id: str = "msg_test"

        text = _make_llm_response(content="Done.")
        response = FakeResponseWithPydanticUsage(
            choices=text.choices,
            usage=FakePydanticUsage(
                prompt_tokens=8200,
                completion_tokens=120,
                total_tokens=8320,
                prompt_tokens_details=FakePromptTokensDetails(cached_tokens=1100),
                cache_creation_input_tokens=7000,
                cache_read_input_tokens=1100,
            ),
        )
        usage = token_usage_from_response(response)
        assert usage.cache_creation_input_tokens == 7000
        assert usage.cache_read_input_tokens == 1100
        assert usage.cached_prompt_tokens is None

    @pytest.mark.asyncio
    async def test_no_cache_metadata_leaves_fields_none(self) -> None:
        """Absent cache metadata must NOT be fabricated to zero."""
        from elspeth.web.composer.llm_response_parsing import token_usage_from_response

        response = self._response_with_usage({"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120})
        usage = token_usage_from_response(response)
        assert usage.cached_prompt_tokens is None
        assert usage.cache_creation_input_tokens is None
        assert usage.cache_read_input_tokens is None

    @pytest.mark.asyncio
    async def test_cache_fields_propagate_through_compose_to_llm_call_record(self) -> None:
        """End-to-end: cache fields appear on the buffered ComposerLLMCall record.

        Patches the LLM call to return a response with OpenAI-shape cache
        metadata, then checks the recorded ComposerLLMCall on the
        ``ComposerResult.llm_calls`` tuple.
        """
        catalog = _mock_catalog()
        settings = _make_settings()
        service = ComposerServiceImpl(catalog=catalog, settings=settings)
        state = _empty_state()

        response = self._response_with_usage(
            {
                "prompt_tokens": 500,
                "completion_tokens": 12,
                "total_tokens": 512,
                "prompt_tokens_details": {"cached_tokens": 256},
            }
        )

        with patch.object(service, "_call_llm", new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = response
            result = await service.compose("Hi", [], state)

        assert len(result.llm_calls) == 1
        call_record = result.llm_calls[0]
        assert call_record.prompt_tokens == 500
        assert call_record.cached_prompt_tokens == 256
        assert call_record.cache_creation_input_tokens is None
        assert call_record.cache_read_input_tokens is None


class TestBuildMessages:
    @pytest.mark.asyncio
    async def test_build_messages_returns_new_list(self) -> None:
        """_build_messages must return a new list on every call."""
        catalog = _mock_catalog()
        settings = _make_settings()
        service = ComposerServiceImpl(catalog=catalog, settings=settings)
        state = _empty_state()

        msgs1 = service._build_messages([], state, "Hello")
        msgs2 = service._build_messages([], state, "Hello")

        assert msgs1 is not msgs2  # different list objects
        assert msgs1 == msgs2  # same content

    @pytest.mark.asyncio
    async def test_build_messages_oserror_redacts_filename(self) -> None:
        """OSError from deployment-skill loading MUST NOT leak its filename.

        ``str(OSError)`` expands to "[Errno N] <strerror>: '<absolute
        path>'".  That filename reveals the operator's data-dir layout
        and — when the deployment skill lives under a user-scoped
        subdirectory — the user identifier itself.  The wrapper
        ``ComposerServiceError`` flows into the 502 response body in
        ``sessions/routes.py::send_message`` and ``recompose``, so the
        message MUST contain only the exception class name.

        This test pins the redaction contract: the ``str(exc)`` of the
        raised ``ComposerServiceError`` contains no substring of the
        provoking OSError's filename or its strerror text.  Mirrors
        the regression assertion on the SQLAlchemy-family 422 path in
        ``_handle_convergence_error`` (web/sessions/routes.py).
        """
        catalog = _mock_catalog()
        settings = _make_settings()
        service = ComposerServiceImpl(catalog=catalog, settings=settings)
        state = _empty_state()

        secret_path = "/var/lib/elspeth/users/alice/skills/secret-deployment.md"

        def _raise_oserror(*_args: object, **_kwargs: object) -> list[dict[str, Any]]:
            raise PermissionError(13, "Permission denied", secret_path)

        with (
            patch("elspeth.web.composer.service.build_messages", side_effect=_raise_oserror),
            pytest.raises(ComposerServiceError) as excinfo,
        ):
            service._build_messages([], state, "Hi")

        body = str(excinfo.value)
        # The filename MUST NOT appear in the wrapper message. Test
        # against the full path AND its directory fragments, because
        # partial leaks (e.g. "/var/lib/elspeth/users/alice") are just
        # as damaging as the full path.
        assert secret_path not in body
        assert "alice" not in body
        assert "Permission denied" not in body
        # The class name IS part of the safe surface — operators
        # reading the 502 still need to distinguish PermissionError
        # from IsADirectoryError from FileNotFoundError.
        assert "PermissionError" in body
        # __cause__ preservation: full detail reaches server-side
        # machinery even though the HTTP body is redacted.
        assert isinstance(excinfo.value.__cause__, PermissionError)
        assert excinfo.value.__cause__.filename == secret_path


class TestComposerMultipleToolCallsPerTurn:
    @pytest.mark.asyncio
    async def test_multiple_tool_calls_in_single_turn(self) -> None:
        """LLM returns multiple tool calls in one response — all executed."""
        catalog = _mock_catalog()
        settings = _make_settings()
        service = ComposerServiceImpl(catalog=catalog, settings=settings)
        state = _empty_state()

        # Turn 1: two tool calls in one response
        multi_call = _make_llm_response(
            tool_calls=[
                {
                    "id": "call_1",
                    "name": "set_source",
                    "arguments": {
                        "plugin": "csv",
                        "on_success": "t1",
                        "options": {"path": "/data/blobs/input.csv", "schema": {"mode": "observed"}},
                        "on_validation_failure": "quarantine",
                    },
                },
                {
                    "id": "call_2",
                    "name": "set_metadata",
                    "arguments": {"patch": {"name": "Dual Call Pipeline"}},
                },
            ],
        )
        # Turn 2: text
        text = _make_llm_response(content="Done.")

        passing_preflight = ValidationResult(is_valid=True, checks=[], errors=[])
        with (
            patch.object(service, "_call_llm", new_callable=AsyncMock) as mock_llm,
            patch.object(service, "_runtime_preflight", return_value=passing_preflight),
        ):
            mock_llm.side_effect = [multi_call, text]
            result = await service.compose("Setup", [], state)

        assert result.state.sources["source"] is not None
        assert result.state.metadata.name == "Dual Call Pipeline"
        assert result.state.version == 3  # two mutations


class TestDiscoveryCache:
    """Tests for the discovery cache (F1)."""

    @pytest.mark.asyncio
    async def test_cacheable_tool_returns_cached_result(self) -> None:
        """Repeated cacheable discovery calls return cached results
        without incrementing any budget counter."""
        catalog = _mock_catalog()
        settings = _make_settings(composer_max_discovery_turns=2)
        service = ComposerServiceImpl(catalog=catalog, settings=settings)
        state = _empty_state()

        # Turn 1: list_sources (first call — executes, charges discovery: 1/2)
        # Turn 2: list_sources AGAIN (cache hit — no budget charge)
        # Turn 3: text
        disc1 = _make_llm_response(
            tool_calls=[{"id": "c1", "name": "list_sources", "arguments": {}}],
        )
        disc2 = _make_llm_response(
            tool_calls=[{"id": "c2", "name": "list_sources", "arguments": {}}],
        )
        text = _make_llm_response(content="Found sources.")

        with patch.object(service, "_call_llm", new_callable=AsyncMock) as mock_llm:
            mock_llm.side_effect = [disc1, disc2, text]
            result = await service.compose("List sources", [], state)

        # Should NOT have raised — second list_sources was a cache hit
        assert result.message == "Found sources."
        # Catalog list_sources is called once by build_messages (prompt
        # context) and once by execute_tool (first discovery call).
        # The second discovery call is a cache hit — no catalog call.
        # Total: 2, not 3.
        assert catalog.list_sources.call_count == 2

    @pytest.mark.asyncio
    async def test_cache_hit_rebuilds_result_envelope_from_current_state(self) -> None:
        """Cacheable discovery data is reused, but validation/version stay current."""
        catalog = _mock_catalog()
        settings = _make_settings()
        service = ComposerServiceImpl(catalog=catalog, settings=settings)
        state = _empty_state()

        disc1 = _make_llm_response(
            tool_calls=[{"id": "c1", "name": "list_sources", "arguments": {}}],
        )
        mutate = _make_llm_response(
            tool_calls=[
                {
                    "id": "c2",
                    "name": "set_source",
                    "arguments": {
                        "plugin": "csv",
                        "on_success": "out",
                        "options": {"path": "/data/blobs/input.csv", "schema": {"mode": "observed"}},
                        "on_validation_failure": "quarantine",
                    },
                }
            ],
        )
        disc2 = _make_llm_response(
            tool_calls=[{"id": "c3", "name": "list_sources", "arguments": {}}],
        )
        text = _make_llm_response(content="Done.")

        passing_preflight = ValidationResult(is_valid=True, checks=[], errors=[])
        with (
            patch.object(service, "_call_llm", new_callable=AsyncMock) as mock_llm,
            patch.object(service, "_runtime_preflight", return_value=passing_preflight),
        ):
            mock_llm.side_effect = [disc1, mutate, disc2, text]
            result = await service.compose("Build", [], state)

        assert result.state.version == 2
        cached_tool_message = mock_llm.call_args_list[3][0][0][-1]
        cached_payload = json.loads(cached_tool_message["content"])
        assert cached_payload["version"] == 2
        expected_validation = ToolResult(
            success=True,
            updated_state=result.state,
            validation=result.state.validate(),
            affected_nodes=(),
        ).to_dict()["validation"]
        assert cached_payload["validation"] == expected_validation
        assert catalog.list_sources.call_count == 2

    @pytest.mark.asyncio
    async def test_cache_key_includes_arguments(self) -> None:
        """Different arguments = different cache entries = both execute."""
        catalog = _mock_catalog()
        settings = _make_settings()
        service = ComposerServiceImpl(catalog=catalog, settings=settings)
        state = _empty_state()

        schema1 = _make_llm_response(
            tool_calls=[
                {
                    "id": "c1",
                    "name": "get_plugin_schema",
                    "arguments": {"plugin_type": "source", "name": "csv"},
                }
            ],
        )
        schema2 = _make_llm_response(
            tool_calls=[
                {
                    "id": "c2",
                    "name": "get_plugin_schema",
                    "arguments": {"plugin_type": "source", "name": "json"},
                }
            ],
        )
        text = _make_llm_response(content="Got schemas.")

        with patch.object(service, "_call_llm", new_callable=AsyncMock) as mock_llm:
            mock_llm.side_effect = [schema1, schema2, text]
            await service.compose("Get schemas", [], state)

        # Both calls should have executed (different arguments)
        assert catalog.get_schema.call_count == 2

    @pytest.mark.asyncio
    async def test_mutation_tools_never_cached(self) -> None:
        """Mutation tool results are never cached — always execute."""
        catalog = _mock_catalog()
        settings = _make_settings()
        service = ComposerServiceImpl(catalog=catalog, settings=settings)
        state = _empty_state()

        mut1 = _make_llm_response(
            tool_calls=[
                {
                    "id": "c1",
                    "name": "set_metadata",
                    "arguments": {"patch": {"name": "X"}},
                }
            ],
        )
        mut2 = _make_llm_response(
            tool_calls=[
                {
                    "id": "c2",
                    "name": "set_metadata",
                    "arguments": {"patch": {"name": "Y"}},
                }
            ],
        )
        text = _make_llm_response(content="Done.")

        with patch.object(service, "_call_llm", new_callable=AsyncMock) as mock_llm:
            mock_llm.side_effect = [mut1, mut2, text]
            result = await service.compose("Update metadata", [], state)

        assert result.state.metadata.name == "Y"


class TestComposeTimeout:
    """Tests for the server-side compose timeout (F1)."""

    @pytest.mark.asyncio
    async def test_timeout_raises_convergence_error(self) -> None:
        """Exceeding composer_timeout_seconds raises ComposerConvergenceError
        with budget_exhausted='timeout'."""
        import asyncio

        catalog = _mock_catalog()
        settings = _make_settings(composer_timeout_seconds=0.1)
        service = ComposerServiceImpl(catalog=catalog, settings=settings)
        state = _empty_state()

        async def slow_llm(*args: Any, **kwargs: Any) -> Any:
            await asyncio.sleep(1.0)
            return _make_llm_response(content="Too late.")

        with patch.object(service, "_call_llm", new_callable=AsyncMock) as mock_llm:
            mock_llm.side_effect = slow_llm
            with pytest.raises(ComposerConvergenceError) as exc_info:
                await service.compose("Slow pipeline", [], state)
            assert exc_info.value.budget_exhausted == "timeout"

    @pytest.mark.asyncio
    async def test_mutation_tool_state_preserved_on_timeout(self) -> None:
        """Mutation tools that complete before timeout must have their
        state reflected in partial_state.

        Regression test for the cancel-safety concern: with cooperative
        timeout, the deadline is checked AFTER tool execution completes,
        so side effects and state publication are never split. The
        partial_state must include the mutation that completed.
        """
        import time

        catalog = _mock_catalog()
        settings = _make_settings()
        service = ComposerServiceImpl(catalog=catalog, settings=settings)
        state = _empty_state()

        call_count = 0

        def _slow_mutation_tool(
            _tool_name: str,
            _arguments: dict[str, Any],
            current_state: CompositionState,
            _catalog: Any,
            **kwargs: Any,
        ) -> ToolResult:
            # Simulate a blob mutation that takes time
            time.sleep(0.2)
            from elspeth.web.composer.state import SourceSpec

            new_state = current_state.with_source(
                SourceSpec(
                    plugin="csv",
                    on_success="out",
                    options={"path": "/data/blobs/f.csv", "schema": {"mode": "observed"}},
                    on_validation_failure="quarantine",
                )
            )
            return ToolResult(
                success=True,
                updated_state=new_state,
                validation=new_state.validate(),
                affected_nodes=("source",),
                data=None,
            )

        async def first_tool_then_timeout_llm(*args: Any, **kwargs: Any) -> Any:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                # First call: return tool call (fast)
                return _make_llm_response(
                    tool_calls=[
                        {
                            "id": "c1",
                            "name": "set_source",
                            "arguments": {
                                "plugin": "csv",
                                "on_success": "out",
                                "options": {"path": "/data/blobs/f.csv", "schema": {"mode": "observed"}},
                                "on_validation_failure": "quarantine",
                            },
                        }
                    ],
                )
            # Second call: exercise _call_llm_before_deadline's timeout
            # capture path without depending on wall-clock scheduling.
            raise TimeoutError

        with (
            patch.object(service, "_call_llm", new=first_tool_then_timeout_llm),
            patch(
                "elspeth.web.composer.tool_batch.execute_tool",
                side_effect=_slow_mutation_tool,
            ),
            pytest.raises(ComposerConvergenceError) as exc_info,
        ):
            await service.compose("Build pipeline", [], state)

        assert exc_info.value.budget_exhausted == "timeout"
        assert call_count == 2
        # The mutation tool completed BEFORE the timeout fired on the
        # second LLM call.  With cooperative timeout, partial_state must
        # reflect the completed mutation.
        assert exc_info.value.partial_state is not None, (
            "partial_state is None — mutation tool's state was lost on timeout. "
            "This is the cancel-safety regression: side effects committed but "
            "state was not published."
        )
        assert exc_info.value.partial_state.sources["source"] is not None
        assert exc_info.value.partial_state.sources["source"].plugin == "csv"


class TestConvergenceProgressDispatch:
    """Each ComposerConvergenceError sub-cause must surface a discriminated
    progress event through the sink — covering the contract that fixes
    elspeth-5030f7373d.

    The original symptom: wall-clock timeout, mutation-turn budget, and
    discovery-turn budget all collapsed into one generic ``phase: failed``
    event with the same headline / evidence / likely_next text. These tests
    end-to-end through ``compose()`` to confirm the sink receives the
    discriminated event for each budget value.
    """

    @pytest.mark.asyncio
    async def test_composition_budget_emits_distinct_failure_event(self) -> None:
        catalog = _mock_catalog()
        settings = _make_settings(composer_max_composition_turns=1)
        service = ComposerServiceImpl(catalog=catalog, settings=settings)
        state = _empty_state()
        progress_events: list[ComposerProgressEvent] = []

        async def record_progress(event: ComposerProgressEvent) -> None:
            progress_events.append(event)

        mut1 = _make_llm_response(
            tool_calls=[
                {
                    "id": "c1",
                    "name": "set_metadata",
                    "arguments": {"patch": {"name": "x"}},
                }
            ],
        )
        mut2 = _make_llm_response(
            tool_calls=[
                {
                    "id": "c2",
                    "name": "set_metadata",
                    "arguments": {"patch": {"name": "y"}},
                }
            ],
        )

        with patch.object(service, "_call_llm", new_callable=AsyncMock) as mock_llm:
            mock_llm.side_effect = [mut1, mut2]
            with pytest.raises(ComposerConvergenceError) as exc_info:
                await service.compose("loop forever", [], state, progress=record_progress)
        assert exc_info.value.budget_exhausted == "composition"

        failed_events = [e for e in progress_events if e.phase == "failed"]
        assert len(failed_events) == 1
        assert failed_events[0].reason == "convergence_composition_budget"
        assert "mutation turn budget" in failed_events[0].headline.lower()

    @pytest.mark.asyncio
    async def test_discovery_budget_emits_distinct_failure_event(self) -> None:
        catalog = _mock_catalog()
        settings = _make_settings(composer_max_discovery_turns=1)
        service = ComposerServiceImpl(catalog=catalog, settings=settings)
        state = _empty_state()
        progress_events: list[ComposerProgressEvent] = []

        async def record_progress(event: ComposerProgressEvent) -> None:
            progress_events.append(event)

        disc1 = _make_llm_response(
            tool_calls=[{"id": "c1", "name": "list_sources", "arguments": {}}],
        )
        disc2 = _make_llm_response(
            tool_calls=[{"id": "c2", "name": "list_transforms", "arguments": {}}],
        )

        with patch.object(service, "_call_llm", new_callable=AsyncMock) as mock_llm:
            mock_llm.side_effect = [disc1, disc2]
            with pytest.raises(ComposerConvergenceError) as exc_info:
                await service.compose("discover forever", [], state, progress=record_progress)
        assert exc_info.value.budget_exhausted == "discovery"

        failed_events = [e for e in progress_events if e.phase == "failed"]
        assert len(failed_events) == 1
        assert failed_events[0].reason == "convergence_discovery_budget"
        assert "discovery turn budget" in failed_events[0].headline.lower()

    @pytest.mark.asyncio
    async def test_wall_clock_timeout_emits_distinct_failure_event(self) -> None:
        import asyncio

        catalog = _mock_catalog()
        settings = _make_settings(composer_timeout_seconds=0.1)
        service = ComposerServiceImpl(catalog=catalog, settings=settings)
        state = _empty_state()
        progress_events: list[ComposerProgressEvent] = []

        async def record_progress(event: ComposerProgressEvent) -> None:
            progress_events.append(event)

        async def slow_llm(*args: Any, **kwargs: Any) -> Any:
            await asyncio.sleep(1.0)
            return _make_llm_response(content="Too late.")

        with patch.object(service, "_call_llm", new_callable=AsyncMock) as mock_llm:
            mock_llm.side_effect = slow_llm
            with pytest.raises(ComposerConvergenceError) as exc_info:
                await service.compose("slow pipeline", [], state, progress=record_progress)
        assert exc_info.value.budget_exhausted == "timeout"

        failed_events = [e for e in progress_events if e.phase == "failed"]
        assert len(failed_events) == 1
        assert failed_events[0].reason == "convergence_wall_clock_timeout"
        assert "timed out" in failed_events[0].headline.lower()

    @pytest.mark.asyncio
    async def test_three_sub_causes_produce_three_distinct_progress_reasons(self) -> None:
        """Regression guard against UX-text collapse — the three convergence
        sub-causes must surface three distinct reason codes via the sink."""
        import asyncio
        import contextlib

        async def collect_failure_reason(
            settings: Any,
            llm_side_effect: Any,
        ) -> str | None:
            catalog = _mock_catalog()
            service = ComposerServiceImpl(catalog=catalog, settings=settings)
            state = _empty_state()
            events: list[ComposerProgressEvent] = []

            async def record(event: ComposerProgressEvent) -> None:
                events.append(event)

            with patch.object(service, "_call_llm", new_callable=AsyncMock) as mock_llm:
                mock_llm.side_effect = llm_side_effect
                with contextlib.suppress(ComposerConvergenceError):
                    await service.compose("x", [], state, progress=record)
            failed = [e for e in events if e.phase == "failed"]
            return failed[-1].reason if failed else None

        composition_reason = await collect_failure_reason(
            _make_settings(composer_max_composition_turns=1),
            [
                _make_llm_response(tool_calls=[{"id": "c1", "name": "set_metadata", "arguments": {"patch": {"name": "x"}}}]),
                _make_llm_response(tool_calls=[{"id": "c2", "name": "set_metadata", "arguments": {"patch": {"name": "y"}}}]),
            ],
        )
        discovery_reason = await collect_failure_reason(
            _make_settings(composer_max_discovery_turns=1),
            [
                _make_llm_response(tool_calls=[{"id": "c1", "name": "list_sources", "arguments": {}}]),
                _make_llm_response(tool_calls=[{"id": "c2", "name": "list_transforms", "arguments": {}}]),
            ],
        )

        async def slow(*args: Any, **kwargs: Any) -> Any:
            await asyncio.sleep(1.0)
            return _make_llm_response(content="late")

        timeout_reason = await collect_failure_reason(
            _make_settings(composer_timeout_seconds=0.1),
            slow,
        )

        codes = {composition_reason, discovery_reason, timeout_reason}
        assert len(codes) == 3, (
            "Convergence sub-causes collapsed into fewer than three distinct progress reasons "
            "— elspeth-5030f7373d regression. Got: " + repr(codes)
        )


class TestPartialStatePreservation:
    """Tests for partial state preservation on convergence failure (F2)."""

    @pytest.mark.asyncio
    async def test_convergence_includes_partial_state_when_mutated(self) -> None:
        """When mutations occurred before convergence failure,
        partial_state is attached to the exception."""
        catalog = _mock_catalog()
        settings = _make_settings(composer_max_composition_turns=1)
        service = ComposerServiceImpl(catalog=catalog, settings=settings)
        state = _empty_state()

        # Turn 1: mutation (set_source) — composition budget exhausted (1/1)
        mut = _make_llm_response(
            tool_calls=[
                {
                    "id": "c1",
                    "name": "set_source",
                    "arguments": {
                        "plugin": "csv",
                        "on_success": "t1",
                        "options": {"path": "/data/blobs/input.csv", "schema": {"mode": "observed"}},
                        "on_validation_failure": "quarantine",
                    },
                }
            ],
        )
        # Bonus call also returns tool calls — convergence error
        mut2 = _make_llm_response(
            tool_calls=[
                {
                    "id": "c2",
                    "name": "set_metadata",
                    "arguments": {"patch": {"name": "nope"}},
                }
            ],
        )

        with patch.object(service, "_call_llm", new_callable=AsyncMock) as mock_llm:
            mock_llm.side_effect = [mut, mut2]
            with pytest.raises(ComposerConvergenceError) as exc_info:
                await service.compose("Build pipeline", [], state)

            assert exc_info.value.partial_state is not None
            assert exc_info.value.partial_state.sources["source"] is not None
            assert exc_info.value.partial_state.sources["source"].plugin == "csv"
            assert exc_info.value.partial_state.version == 2

    @pytest.mark.asyncio
    async def test_convergence_no_partial_state_when_no_mutations(self) -> None:
        """When no mutations occurred, partial_state is None."""
        catalog = _mock_catalog()
        settings = _make_settings(composer_max_discovery_turns=1)
        service = ComposerServiceImpl(catalog=catalog, settings=settings)
        state = _empty_state()

        disc1 = _make_llm_response(
            tool_calls=[{"id": "c1", "name": "list_sources", "arguments": {}}],
        )
        disc2 = _make_llm_response(
            tool_calls=[{"id": "c2", "name": "list_transforms", "arguments": {}}],
        )

        with patch.object(service, "_call_llm", new_callable=AsyncMock) as mock_llm:
            mock_llm.side_effect = [disc1, disc2]
            with pytest.raises(ComposerConvergenceError) as exc_info:
                await service.compose("Just looking", [], state)

            assert exc_info.value.partial_state is None


class TestComposerSamplingConfig:
    """Composer LLM sampling must come from operator config.

    Default ``None`` means omit the provider parameter. Configured values are
    sent verbatim and recorded on each ComposerLLMCall, so a reviewer can
    correlate failures with the precise sampling regime.
    """

    @pytest.mark.asyncio
    async def test_call_llm_sends_configured_temperature_and_seed(self) -> None:
        catalog = _mock_catalog()
        settings = _make_settings(composer_temperature=0.0, composer_seed=42)
        service = ComposerServiceImpl(catalog=catalog, settings=settings)
        state = _empty_state()

        # Single text-only response converges the loop immediately.
        completion = _make_llm_response(content="acknowledged")

        # Service may reject because no pipeline was built; we only
        # care about what reached LiteLLM on the first (and possibly only) call.
        with (
            patch(
                "elspeth.web.composer.service._litellm_acompletion",
                new_callable=AsyncMock,
                return_value=completion,
            ) as mock_acomp,
            contextlib.suppress(ComposerServiceError),
        ):
            await service.compose("Hello", [], state)

        assert mock_acomp.call_count >= 1
        first_call_kwargs = mock_acomp.call_args_list[0].kwargs
        assert first_call_kwargs["temperature"] == 0.0
        assert first_call_kwargs["seed"] == 42

    @pytest.mark.asyncio
    async def test_call_llm_omits_sampling_when_operator_leaves_it_unset(self) -> None:
        catalog = _mock_catalog()
        settings = _make_settings(composer_model="anthropic/claude-3-5-sonnet-20241022")
        service = ComposerServiceImpl(catalog=catalog, settings=settings)
        completion = _make_llm_response(content="acknowledged")

        with patch(
            "elspeth.web.composer.service._litellm_acompletion",
            new_callable=AsyncMock,
            return_value=completion,
        ) as mock_acomp:
            await service._call_llm([{"role": "user", "content": "Hello"}], [])

        kwargs = mock_acomp.call_args_list[0].kwargs
        assert "temperature" not in kwargs
        assert "seed" not in kwargs

    @pytest.mark.asyncio
    async def test_call_text_llm_sends_configured_temperature_and_seed(self) -> None:
        catalog = _mock_catalog()
        settings = _make_settings(composer_temperature=0.0, composer_seed=42)
        service = ComposerServiceImpl(catalog=catalog, settings=settings)

        completion = _make_llm_response(content="diagnostic text")

        with patch(
            "elspeth.web.composer.service._litellm_acompletion",
            new_callable=AsyncMock,
            return_value=completion,
        ) as mock_acomp:
            await service._call_text_llm([{"role": "user", "content": "explain"}])

        assert mock_acomp.call_count == 1
        kwargs = mock_acomp.call_args_list[0].kwargs
        assert kwargs["temperature"] == 0.0
        assert kwargs["seed"] == 42

    @pytest.mark.asyncio
    async def test_call_text_llm_omits_sampling_when_operator_leaves_it_unset(self) -> None:
        catalog = _mock_catalog()
        settings = _make_settings(composer_model="anthropic/claude-3-5-sonnet-20241022")
        service = ComposerServiceImpl(catalog=catalog, settings=settings)
        completion = _make_llm_response(content="diagnostic text")

        with patch(
            "elspeth.web.composer.service._litellm_acompletion",
            new_callable=AsyncMock,
            return_value=completion,
        ) as mock_acomp:
            await service._call_text_llm([{"role": "user", "content": "explain"}])

        kwargs = mock_acomp.call_args_list[0].kwargs
        assert "temperature" not in kwargs
        assert "seed" not in kwargs


class TestEmptyChoicesValidation:
    """Tier 3 boundary: LiteLLM can return empty choices."""

    @pytest.mark.asyncio
    async def test_empty_choices_raises_service_error(self) -> None:
        """LiteLLM returning empty choices must raise ComposerServiceError.

        Empty choices can occur on content-filter blocks, rate-limit
        responses, or malformed upstream responses.  Without validation,
        this causes an IndexError at response.choices[0].message.
        """
        catalog = _mock_catalog()
        settings = _make_settings()
        service = ComposerServiceImpl(catalog=catalog, settings=settings)
        state = _empty_state()

        # Patch the lazy LiteLLM wrapper (not _call_llm) so the validation
        # inside _call_llm is exercised through the production code path.
        empty_response = FakeLLMResponse(choices=[])
        with (
            patch(
                "elspeth.web.composer.service._litellm_acompletion",
                new_callable=AsyncMock,
                return_value=empty_response,
            ),
            pytest.raises(ComposerServiceError, match="empty choices"),
        ):
            await service.compose("Hello", [], state)

    @pytest.mark.asyncio
    async def test_empty_choices_on_bonus_turn_raises_service_error(self) -> None:
        """Empty choices on the bonus turn (budget exhaustion) also raises.

        The bonus turn at composition budget exhaustion goes through the
        same _call_llm() path, so the validation protects both sites.
        """
        catalog = _mock_catalog()
        # Budget of 1 composition turn — first mutation exhausts it,
        # then the bonus _call_llm returns empty choices.
        settings = _make_settings(composer_max_composition_turns=1)
        service = ComposerServiceImpl(catalog=catalog, settings=settings)
        state = _empty_state()

        # First call: valid response with a mutation tool call
        mutation_call = _make_llm_response(
            tool_calls=[
                {
                    "id": "c1",
                    "name": "set_source",
                    "arguments": {
                        "plugin": "csv",
                        "on_success": "out",
                        "options": {},
                        "on_validation_failure": "quarantine",
                    },
                }
            ],
        )
        # Second call (bonus turn): empty choices
        empty_response = FakeLLMResponse(choices=[])

        with (
            patch(
                "elspeth.web.composer.service._litellm_acompletion",
                new_callable=AsyncMock,
                side_effect=[mutation_call, empty_response],
            ) as mock_acomp,
            pytest.raises(ComposerServiceError, match="empty choices"),
        ):
            await service.compose("Setup CSV", [], state)

        # Confirm both LLM calls happened — the error is from the bonus
        # turn (second call), not from a tool handler fault on the first.
        assert mock_acomp.call_count == 2


class TestComposerAvailabilityAndBadRequest:
    """Readiness and LiteLLM bad-request failures are normalized by the service."""

    @pytest.mark.asyncio
    async def test_unavailable_composer_short_circuits_before_llm_call(self) -> None:
        catalog = _mock_catalog()
        settings = _make_settings()
        service = ComposerServiceImpl(catalog=catalog, settings=settings)
        service._availability = ComposerAvailability(
            available=False,
            model="bad-model",
            provider=None,
            reason="Composer model bad-model is unavailable.",
        )

        with (
            patch.object(service, "_call_llm", new_callable=AsyncMock) as mock_llm,
            pytest.raises(ComposerServiceError, match="bad-model is unavailable"),
        ):
            mock_llm.return_value = _make_llm_response(content="unexpected")
            await service.compose("Hello", [], _empty_state())

        mock_llm.assert_not_called()

    @pytest.mark.asyncio
    async def test_litellm_bad_request_raises_redacted_service_error(self) -> None:
        from litellm.exceptions import BadRequestError as LiteLLMBadRequestError

        catalog = _mock_catalog()
        settings = _make_settings()
        service = ComposerServiceImpl(catalog=catalog, settings=settings)
        state = _empty_state()
        bad_request = LiteLLMBadRequestError(
            message="bad model leaked-detail",
            model="bad-model",
            llm_provider="bad-provider",
        )

        with (
            patch(
                "elspeth.web.composer.service._litellm_acompletion",
                new_callable=AsyncMock,
                side_effect=bad_request,
            ),
            pytest.raises(ComposerServiceError) as exc_info,
        ):
            await service.compose("Hello", [], state)

        assert str(exc_info.value) == "LLM request rejected (BadRequestError)"
        assert "leaked-detail" not in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_bad_request_llm_error_preserves_provider_detail(self) -> None:
        """``_BadRequestLLMError`` carries the underlying provider message.

        The wrap message (``str(exc)``) is intentionally redacted so the
        route layer does not leak provider details by default. The route
        layer's ``expose_provider_error=True`` path re-uses
        ``provider_detail`` to surface the raw message after scrubbing.

        Ticket: elspeth-9f7f9d5787.
        """
        from litellm.exceptions import BadRequestError as LiteLLMBadRequestError

        from elspeth.web.composer.service import _BadRequestLLMError

        catalog = _mock_catalog()
        settings = _make_settings()
        service = ComposerServiceImpl(catalog=catalog, settings=settings)
        state = _empty_state()
        bad_request = LiteLLMBadRequestError(
            message="provider says bad",
            model="gpt-4o",
            llm_provider="openai",
        )

        with (
            patch(
                "elspeth.web.composer.service._litellm_acompletion",
                new_callable=AsyncMock,
                side_effect=bad_request,
            ),
            pytest.raises(_BadRequestLLMError) as exc_info,
        ):
            await service.compose("Hello", [], state)

        # str(exc) unchanged from the existing redacted wrap message.
        assert str(exc_info.value) == "LLM request rejected (BadRequestError)"
        # New attributes carry the provider detail for the route layer.
        assert exc_info.value.provider_detail is not None
        assert "provider says bad" in exc_info.value.provider_detail
        assert exc_info.value.provider_status_code == 400

    @pytest.mark.asyncio
    async def test_bad_request_llm_error_empty_message_collapses_to_none(self) -> None:
        """Empty-string provider messages collapse to ``None`` so callers can
        distinguish "no detail" from "empty string" via truthiness checks.
        """
        from litellm.exceptions import BadRequestError as LiteLLMBadRequestError

        from elspeth.web.composer.service import _BadRequestLLMError

        catalog = _mock_catalog()
        settings = _make_settings()
        service = ComposerServiceImpl(catalog=catalog, settings=settings)
        state = _empty_state()
        # LiteLLM's BadRequestError formats the message into a longer string;
        # passing an empty message still yields a non-empty rendered str.
        # To exercise the empty-collapse path, raise the exception type
        # directly with an empty rendered form.
        bad_request = LiteLLMBadRequestError(message="", model="m", llm_provider="p")
        # Force str(exc) == "" so the ``or None`` collapse can fire.
        bad_request.args = ("",)
        bad_request.message = ""

        with (
            patch(
                "elspeth.web.composer.service._litellm_acompletion",
                new_callable=AsyncMock,
                side_effect=bad_request,
            ),
            pytest.raises(_BadRequestLLMError) as exc_info,
        ):
            await service.compose("Hello", [], state)

        # str(bad_request) is "" so provider_detail should collapse to None.
        # provider_status_code is still set from the .status_code attribute.
        assert exc_info.value.provider_detail is None
        assert exc_info.value.provider_status_code == 400

    @pytest.mark.asyncio
    async def test_litellm_api_error_is_retried_before_unavailable(self) -> None:
        from litellm.exceptions import APIError as LiteLLMAPIError

        catalog = _mock_catalog()
        settings = _make_settings()
        service = ComposerServiceImpl(catalog=catalog, settings=settings)
        state = _empty_state()
        transient_error = LiteLLMAPIError(
            status_code=503,
            message="provider temporarily unavailable leaked-detail",
            model="openrouter/openai/gpt-5.5",
            llm_provider="openrouter",
        )
        success = _make_llm_response(content="Recovered.")

        with (
            patch(
                "elspeth.web.composer.service._litellm_acompletion",
                new_callable=AsyncMock,
                side_effect=[transient_error, success],
            ) as mock_llm,
            patch("elspeth.web.composer.service.asyncio.sleep", new_callable=AsyncMock) as mock_sleep,
        ):
            result = await service.compose("Hello", [], state)

        assert result.message == "Recovered."
        assert mock_llm.call_count == 2
        mock_sleep.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_bad_request_llm_error_is_not_retried(self) -> None:
        """Provider 400-class errors are deterministic — not transient — and
        must surface to the caller on the first attempt without consuming
        any retry budget. Pins the no-retry policy made mechanically visible
        in commit 361643809 (``except _BadRequestLLMError: raise`` between
        the ``LiteLLMAuthError`` and ``LiteLLMAPIError`` clauses). Without
        this negative-side assertion, a future refactor that either
        (a) changes ``_BadRequestLLMError`` to inherit from
        ``LiteLLMAPIError`` or (b) removes the explicit re-raise clause
        could silently re-route bad-requests through the retry loop and
        every existing positive-side test would still pass.
        """
        from litellm.exceptions import BadRequestError as LiteLLMBadRequestError

        from elspeth.web.composer.service import _BadRequestLLMError

        catalog = _mock_catalog()
        settings = _make_settings()
        service = ComposerServiceImpl(catalog=catalog, settings=settings)
        state = _empty_state()
        bad_request = LiteLLMBadRequestError(
            message="model gpt-foo does not exist",
            model="gpt-foo",
            llm_provider="openai",
        )

        with (
            patch(
                "elspeth.web.composer.service._litellm_acompletion",
                new_callable=AsyncMock,
                side_effect=bad_request,
            ) as mock_llm,
            patch("elspeth.web.composer.service.asyncio.sleep", new_callable=AsyncMock) as mock_sleep,
            pytest.raises(_BadRequestLLMError),
        ):
            await service.compose("Hello", [], state)

        # Pins the no-retry contract: exactly one provider call, and no
        # backoff sleep was awaited (the retry path's only observable
        # side-effect besides the call count).
        assert mock_llm.call_count == 1
        mock_sleep.assert_not_awaited()


class TestPluginBugCrashesFromToolExecution:
    """Plugin-internal TypeError/ValueError/UnicodeError must crash.

    The compose loop catches ONLY ToolArgumentError around execute_tool.
    Any other TypeError/ValueError/UnicodeError is a plugin bug — per
    CLAUDE.md, silently laundering a plugin bug as an LLM-argument error
    is worse than crashing, because the audit trail records a confident
    but wrong Tier-3 story.

    Mirrors test_internal_key_error_is_not_swallowed.
    """

    @pytest.mark.asyncio
    async def test_plugin_value_error_is_not_swallowed(self) -> None:
        catalog = _mock_catalog()
        settings = _make_settings()
        service = ComposerServiceImpl(catalog=catalog, settings=settings)
        state = _empty_state()

        valid_call = _make_llm_response(
            tool_calls=[
                {
                    "id": "c1",
                    "name": "set_source",
                    "arguments": {
                        "plugin": "csv",
                        "on_success": "out",
                        "options": {},
                        "on_validation_failure": "quarantine",
                    },
                }
            ],
        )

        with (
            patch.object(service, "_call_llm", new_callable=AsyncMock) as mock_llm,
            patch(
                "elspeth.web.composer.tool_batch.execute_tool",
                side_effect=ValueError("invalid expression syntax — plugin bug"),
            ),
        ):
            mock_llm.return_value = valid_call
            with pytest.raises(ComposerPluginCrashError) as exc_info:
                await service.compose("Setup", [], state)
        # Crash on first tool call → no prior mutations → partial_state is None.
        assert exc_info.value.partial_state is None
        assert isinstance(exc_info.value.original_exc, ValueError)
        assert "plugin bug" in str(exc_info.value.original_exc)
        assert exc_info.value.exc_class == "ValueError"

    @pytest.mark.asyncio
    async def test_plugin_type_error_is_not_swallowed(self) -> None:
        catalog = _mock_catalog()
        settings = _make_settings()
        service = ComposerServiceImpl(catalog=catalog, settings=settings)
        state = _empty_state()

        valid_call = _make_llm_response(
            tool_calls=[
                {
                    "id": "c1",
                    "name": "set_source",
                    "arguments": {
                        "plugin": "csv",
                        "on_success": "out",
                        "options": {},
                        "on_validation_failure": "quarantine",
                    },
                }
            ],
        )

        with (
            patch.object(service, "_call_llm", new_callable=AsyncMock) as mock_llm,
            patch(
                "elspeth.web.composer.tool_batch.execute_tool",
                side_effect=TypeError("NoneType + int — plugin bug"),
            ),
        ):
            mock_llm.return_value = valid_call
            with pytest.raises(ComposerPluginCrashError) as exc_info:
                await service.compose("Setup", [], state)
        assert exc_info.value.partial_state is None
        assert isinstance(exc_info.value.original_exc, TypeError)
        assert "plugin bug" in str(exc_info.value.original_exc)
        assert exc_info.value.exc_class == "TypeError"

    @pytest.mark.asyncio
    async def test_plugin_unicode_error_is_not_swallowed(self) -> None:
        catalog = _mock_catalog()
        settings = _make_settings()
        service = ComposerServiceImpl(catalog=catalog, settings=settings)
        state = _empty_state()

        valid_call = _make_llm_response(
            tool_calls=[
                {
                    "id": "c1",
                    "name": "set_source",
                    "arguments": {
                        "plugin": "csv",
                        "on_success": "out",
                        "options": {},
                        "on_validation_failure": "quarantine",
                    },
                }
            ],
        )

        with (
            patch.object(service, "_call_llm", new_callable=AsyncMock) as mock_llm,
            patch(
                "elspeth.web.composer.tool_batch.execute_tool",
                side_effect=UnicodeDecodeError("utf-8", b"\xff", 0, 1, "plugin bug"),
            ),
        ):
            mock_llm.return_value = valid_call
            with pytest.raises(ComposerPluginCrashError) as exc_info:
                await service.compose("Setup", [], state)
        assert exc_info.value.partial_state is None
        assert isinstance(exc_info.value.original_exc, UnicodeDecodeError)
        assert exc_info.value.exc_class == "UnicodeDecodeError"

    @pytest.mark.asyncio
    async def test_plugin_crash_after_successful_mutation_preserves_partial_state(
        self,
    ) -> None:
        """When a plugin crashes AFTER at least one prior mutation succeeded
        in the same request, ``ComposerPluginCrashError.partial_state`` MUST
        carry the accumulated post-mutation state so the route handler can
        persist it into composition_states.

        This closes the P1 regression flagged in review: the narrowed
        ``except`` in compose() used to re-raise bare exceptions without
        threading the loop-local ``state``, so any successful mutations
        prior to the crash were silently dropped and recompose restarted
        from the stale pre-request state.
        """
        catalog = _mock_catalog()
        settings = _make_settings()
        service = ComposerServiceImpl(catalog=catalog, settings=settings)
        state = _empty_state()
        initial_version = state.version

        # Two tool calls in a single LLM turn: first succeeds (mutates
        # state), second raises a plugin-bug exception.
        two_calls = _make_llm_response(
            tool_calls=[
                {
                    "id": "c1",
                    "name": "set_source",
                    "arguments": {
                        "plugin": "csv",
                        "on_success": "out",
                        "options": {},
                        "on_validation_failure": "quarantine",
                    },
                },
                {
                    "id": "c2",
                    "name": "set_metadata",
                    "arguments": {"patch": {"name": "after-mutation"}},
                },
            ],
        )

        mutated_state = CompositionState(
            source=None,
            nodes=(),
            edges=(),
            outputs=(),
            metadata=PipelineMetadata(name="after-mutation"),
            version=initial_version + 1,
        )
        successful_result = ToolResult(
            success=True,
            updated_state=mutated_state,
            validation=ValidationSummary(is_valid=True, errors=()),
            affected_nodes=(),
        )

        call_count = {"n": 0}

        def _fake_execute_tool(*args: Any, **kwargs: Any) -> ToolResult:
            call_count["n"] += 1
            if call_count["n"] == 1:
                return successful_result
            raise ValueError("plugin bug: crash AFTER first mutation")

        with (
            patch.object(service, "_call_llm", new_callable=AsyncMock) as mock_llm,
            patch(
                "elspeth.web.composer.tool_batch.execute_tool",
                side_effect=_fake_execute_tool,
            ),
        ):
            mock_llm.return_value = two_calls
            with pytest.raises(ComposerPluginCrashError) as exc_info:
                await service.compose("Setup", [], state)

        assert call_count["n"] == 2, "both tool calls should have been attempted"
        crash = exc_info.value
        assert crash.partial_state is not None, "partial_state MUST be populated when a mutation succeeded before the crash"
        assert crash.partial_state.version == initial_version + 1
        assert crash.partial_state.metadata.name == "after-mutation"
        assert isinstance(crash.original_exc, ValueError)
        assert crash.exc_class == "ValueError"

    @pytest.mark.asyncio
    async def test_tool_argument_error_is_caught_and_fed_to_llm(self) -> None:
        """Positive case: ToolArgumentError IS caught, error fed back for LLM retry."""
        catalog = _mock_catalog()
        settings = _make_settings()
        service = ComposerServiceImpl(catalog=catalog, settings=settings)
        state = _empty_state()

        valid_call = _make_llm_response(
            tool_calls=[
                {
                    "id": "c1",
                    "name": "set_source",
                    "arguments": {
                        "plugin": "csv",
                        "on_success": "out",
                        "options": {},
                        "on_validation_failure": "quarantine",
                    },
                }
            ],
        )
        text = _make_llm_response(content="Got it, trying again.")

        with (
            patch.object(service, "_call_llm", new_callable=AsyncMock) as mock_llm,
            patch(
                "elspeth.web.composer.tool_batch.execute_tool",
                side_effect=ToolArgumentError(
                    argument="plugin",
                    expected="a string",
                    actual_type="int",
                ),
            ),
        ):
            mock_llm.side_effect = [valid_call, text]
            result = await service.compose("Setup", [], state)

        assert isinstance(result, ComposerResult)
        second_call_messages = mock_llm.call_args_list[1].args[0]
        tool_messages = [m for m in second_call_messages if m.get("role") == "tool"]
        assert len(tool_messages) == 1
        error_payload = json.loads(tool_messages[0]["content"])
        assert "'plugin' must be a string, got int" in error_payload["error"]

    @pytest.mark.asyncio
    async def test_tool_argument_error_subclass_cannot_leak_cause_to_llm(self) -> None:
        """Defense-in-depth: if a subclass overrides __str__ to embed the
        __cause__ chain, the LLM-echo path must still use args[0] only.

        Simulates a future regression where a helpful-looking subclass does
        `def __str__(self): return f"{self.args[0]}: caused by {self.__cause__}"`.
        A DB URL or file path leaked through __cause__ would then reach the
        LLM API. The compose loop MUST short-circuit __str__ and emit
        args[0] verbatim, isolating the cause chain to __cause__ (audit-only).
        """

        class LeakyToolArgumentError(ToolArgumentError):
            def __str__(self) -> str:
                return f"{self.args[0]}: caused by {self.__cause__}"

        catalog = _mock_catalog()
        settings = _make_settings()
        service = ComposerServiceImpl(catalog=catalog, settings=settings)
        state = _empty_state()

        secret_path = "/etc/elspeth/secrets/bootstrap.key"
        secret_cause = ValueError(f"bad path: {secret_path}")
        leaky = LeakyToolArgumentError(
            argument="content",
            expected="a string",
            actual_type="int",
        )
        leaky.__cause__ = secret_cause

        valid_call = _make_llm_response(
            tool_calls=[
                {
                    "id": "c1",
                    "name": "set_source",
                    "arguments": {
                        "plugin": "csv",
                        "on_success": "out",
                        "options": {},
                        "on_validation_failure": "quarantine",
                    },
                }
            ],
        )
        text = _make_llm_response(content="Got it.")

        with (
            patch.object(service, "_call_llm", new_callable=AsyncMock) as mock_llm,
            patch(
                "elspeth.web.composer.tool_batch.execute_tool",
                side_effect=leaky,
            ),
        ):
            mock_llm.side_effect = [valid_call, text]
            await service.compose("Setup", [], state)

        second_call_messages = mock_llm.call_args_list[1].args[0]
        tool_messages = [m for m in second_call_messages if m.get("role") == "tool"]
        assert len(tool_messages) == 1
        error_payload = json.loads(tool_messages[0]["content"])
        assert "'content' must be a string, got int" in error_payload["error"]
        # The crucial assertion: the cause-chain content NEVER appears.
        assert secret_path not in error_payload["error"]
        assert "caused by" not in error_payload["error"]


class TestPluginCrashSessionPersistence:
    """Plugin-bug crash must leave a durable session-row breadcrumb.

    "No silent drops" for session records: a plugin crash that leaves
    the session in no recorded terminal state is as bad for audit
    integrity as the laundering behaviour this plan eliminates.

    Given the current sessions_table schema (no status / crashed_at /
    last_exc_class columns), the breadcrumb is a bump of updated_at.
    This test asserts that bump, plus the invariant that NO exception
    message leaks into any column. The follow-up filigree issue tracks
    the schema migration that adds richer crash markers.
    """

    @pytest.fixture(autouse=True)
    def _setup(self, tmp_path: Path) -> None:
        from datetime import UTC, datetime
        from uuid import uuid4

        from sqlalchemy.pool import StaticPool

        from elspeth.web.sessions.engine import create_session_engine
        from elspeth.web.sessions.models import sessions_table
        from elspeth.web.sessions.schema import initialize_session_schema

        self.engine = create_session_engine(
            "sqlite:///:memory:",
            poolclass=StaticPool,
            connect_args={"check_same_thread": False},
        )
        initialize_session_schema(self.engine)

        self.session_id = str(uuid4())
        self.data_dir = tmp_path
        # Seed the sessions row with a DELIBERATELY OLD updated_at so the
        # crash-path bump is unambiguously distinguishable from the seed.
        self.seeded_at = datetime(2020, 1, 1, tzinfo=UTC)
        with self.engine.begin() as conn:
            conn.execute(
                sessions_table.insert().values(
                    id=self.session_id,
                    user_id="test-user",
                    auth_provider_type="local",
                    title="Test",
                    trust_mode="auto_commit",
                    density_default="high",
                    created_at=self.seeded_at,
                    updated_at=self.seeded_at,
                )
            )

    @pytest.mark.asyncio
    async def test_plugin_crash_bumps_session_updated_at(self) -> None:
        from elspeth.web.sessions.models import sessions_table

        catalog = _mock_catalog()
        settings = _make_settings(data_dir=self.data_dir)
        service = ComposerServiceImpl(
            catalog=catalog,
            settings=settings,
            sessions_service=_test_sessions_service(self.engine, self.data_dir),
            session_engine=self.engine,
        )
        state = _empty_state()

        valid_call = _make_llm_response(
            tool_calls=[
                {
                    "id": "c1",
                    "name": "set_source",
                    "arguments": {
                        "plugin": "csv",
                        "on_success": "out",
                        "options": {},
                        "on_validation_failure": "quarantine",
                    },
                }
            ],
        )

        with (
            patch.object(service, "_call_llm", new_callable=AsyncMock) as mock_llm,
            patch(
                "elspeth.web.composer.tool_batch.execute_tool",
                side_effect=ValueError("plugin bug: /etc/secrets/bootstrap.key is bad"),
            ),
        ):
            mock_llm.return_value = valid_call
            with pytest.raises(ComposerPluginCrashError) as exc_info:
                await service.compose("Setup", [], state, session_id=self.session_id)
        # The underlying plugin exception is preserved on the wrapper.
        assert isinstance(exc_info.value.original_exc, ValueError)
        assert "plugin bug" in str(exc_info.value.original_exc)

        # Assertion 1: session row was touched on the crash path.
        with self.engine.begin() as conn:
            row = conn.execute(sessions_table.select().where(sessions_table.c.id == self.session_id)).one()

        # SQLite DateTime(timezone=True) strips tzinfo on read — normalize
        # both sides of the comparison to the same tz-naive representation.
        row_updated_at = row.updated_at
        if row_updated_at.tzinfo is None:
            seed_for_compare = self.seeded_at.replace(tzinfo=None)
        else:
            seed_for_compare = self.seeded_at
        assert row_updated_at > seed_for_compare, "crash path must bump updated_at as audit breadcrumb"

        # Assertion 2: NO column holds the exception message. Stringify
        # the entire row and verify secret fragments / class hints are
        # absent. This is the load-bearing audit-integrity invariant —
        # if a future refactor adds a 'last_error' column, the assertion
        # will catch any attempt to persist the raw message.
        row_text = " | ".join(str(v) for v in row._mapping.values())
        assert "plugin bug" not in row_text
        assert "/etc/secrets" not in row_text
        assert "ValueError" not in row_text

    @pytest.mark.asyncio
    async def test_persist_crashed_session_failure_does_not_mask_plugin_bug(
        self,
    ) -> None:
        """If _persist_crashed_session itself raises a recoverable audit-path
        exception (SQLAlchemyError / OSError), slog.error fires and the
        original plugin-bug exception still propagates unchanged.

        Uses sqlalchemy.exc.OperationalError as the stand-in for a realistic
        DB-write failure (connection drop, locking timeout, disk I/O
        translated to SQLAlchemy layer). The catch at
        service.py ComposerServiceImpl.compose is narrowed to
        (SQLAlchemyError, OSError); substituting RuntimeError here would
        assert the wrong invariant because RuntimeError deliberately
        propagates past this catch (see the programmer-bug companion test).

        Two invariants asserted:
        1. The ORIGINAL ValueError reaches the caller (not the
           OperationalError from the persistence failure).
        2. slog.error is called with the `composer_crash_persistence_failed`
           event — guarantees that an accidental removal of Step 4a-pre's
           structlog import would be caught (without this assertion, a
           regression where slog.error silently fails as NameError would
           pass the test because the original exception still propagates).
        """
        from structlog.testing import capture_logs

        catalog = _mock_catalog()
        settings = _make_settings(data_dir=self.data_dir)
        service = ComposerServiceImpl(
            catalog=catalog,
            settings=settings,
            sessions_service=_test_sessions_service(self.engine, self.data_dir),
            session_engine=self.engine,
        )
        state = _empty_state()

        valid_call = _make_llm_response(
            tool_calls=[
                {
                    "id": "c1",
                    "name": "set_source",
                    "arguments": {
                        "plugin": "csv",
                        "on_success": "out",
                        "options": {},
                        "on_validation_failure": "quarantine",
                    },
                }
            ],
        )

        with (
            patch.object(service, "_call_llm", new_callable=AsyncMock) as mock_llm,
            patch(
                "elspeth.web.composer.tool_batch.execute_tool",
                side_effect=ValueError("original plugin bug"),
            ),
            patch.object(
                service,
                "_persist_crashed_session",
                side_effect=OperationalError("UPDATE sessions", {}, Exception("db unavailable")),
            ),
            capture_logs() as cap_logs,
        ):
            mock_llm.return_value = valid_call
            with pytest.raises(ComposerPluginCrashError) as exc_info:
                await service.compose("Setup", [], state, session_id=self.session_id)
        # Original plugin exception survives the wrap.
        assert isinstance(exc_info.value.original_exc, ValueError)
        assert "original plugin bug" in str(exc_info.value.original_exc)

        # The crash-persistence-failure slog.error MUST fire. This closes
        # the regression risk where Step 4a-pre's structlog import is
        # accidentally removed — the method would then raise NameError
        # inside the except, masking the original ValueError.
        persistence_failure_events = [entry for entry in cap_logs if entry.get("event") == "composer_crash_persistence_failed"]
        assert len(persistence_failure_events) == 1, cap_logs
        event = persistence_failure_events[0]
        assert event["session_id"] == self.session_id
        assert event["original_exc_class"] == "ValueError"
        # audit_exc_class is the class of the *persistence* failure, not the
        # original plugin bug. Present so operators can distinguish "DB
        # write failed with IntegrityError" from "DB write failed with
        # OperationalError" without needing the traceback.
        assert event["audit_exc_class"] == "OperationalError"
        # No traceback / exception message fields — exc_info was deliberately
        # dropped from this slog call to prevent __cause__-chain secret
        # leakage into server logs.
        assert "exc_info" not in event
        assert "exception" not in event
        assert "stack_info" not in event
        # Exception messages MUST NOT appear anywhere in the structured
        # event (defense-in-depth against accidental re-addition of a
        # message= field in a future refactor).
        assert "original plugin bug" not in str(event)
        # The OperationalError carries its SQL statement and __cause__
        # ("db unavailable") — neither may appear in the structured event.
        assert "db unavailable" not in str(event)
        assert "UPDATE sessions" not in str(event)

    @pytest.mark.asyncio
    async def test_persist_crashed_session_real_path_slog_emission(self) -> None:
        """Smoke test for Step 4a-pre: exercise the real _persist_crashed_session
        path (no patching of the private method).  If structlog is not
        imported in service.py, this test will surface the NameError that
        `test_persist_crashed_session_failure_does_not_mask_plugin_bug`
        misses (because that test patches the method itself).

        The real _persist_crashed_session should succeed here (the sessions
        engine is live), so we assert the crash propagates without any
        persistence-failure slog event.
        """
        from structlog.testing import capture_logs

        catalog = _mock_catalog()
        settings = _make_settings(data_dir=self.data_dir)
        service = ComposerServiceImpl(
            catalog=catalog,
            settings=settings,
            sessions_service=_test_sessions_service(self.engine, self.data_dir),
            session_engine=self.engine,
        )
        state = _empty_state()

        valid_call = _make_llm_response(
            tool_calls=[
                {
                    "id": "c1",
                    "name": "set_source",
                    "arguments": {
                        "plugin": "csv",
                        "on_success": "out",
                        "options": {},
                        "on_validation_failure": "quarantine",
                    },
                }
            ],
        )

        with (
            patch.object(service, "_call_llm", new_callable=AsyncMock) as mock_llm,
            patch(
                "elspeth.web.composer.tool_batch.execute_tool",
                side_effect=ValueError("plugin bug"),
            ),
            capture_logs() as cap_logs,
        ):
            mock_llm.return_value = valid_call
            with pytest.raises(ComposerPluginCrashError) as exc_info:
                await service.compose("Setup", [], state, session_id=self.session_id)
        assert isinstance(exc_info.value.original_exc, ValueError)
        assert "plugin bug" in str(exc_info.value.original_exc)

        # No persistence-failure event — the real path succeeded.
        persistence_failure_events = [entry for entry in cap_logs if entry.get("event") == "composer_crash_persistence_failed"]
        assert persistence_failure_events == [], cap_logs

    @pytest.mark.asyncio
    async def test_persist_crashed_session_programmer_bug_propagates_past_catch(
        self,
    ) -> None:
        """Programmer-bug exceptions inside _persist_crashed_session MUST NOT
        be absorbed by the audit-cleanup catch in compose().

        This test is the guardrail for the narrowed catch at
        ComposerServiceImpl.compose: replacing ``except Exception`` with
        ``except (SQLAlchemyError, OSError)`` means AttributeError, TypeError,
        AssertionError, NameError and the like now escape the handler.
        A future regression that re-widens the catch (e.g., "catch everything
        so audit never crashes the request") would silently pass the sibling
        ``test_persist_crashed_session_failure_does_not_mask_plugin_bug``
        test because that path raises an audit-family exception. This test
        closes the loop by asserting AttributeError — a canonical Tier 1/2
        programmer bug — bubbles out of the compose() call unchanged, NOT
        wrapped as ComposerPluginCrashError and NOT logged as
        ``composer_crash_persistence_failed``.

        The original plugin-bug ValueError becomes the ``__context__`` of the
        escaping AttributeError because Python chains implicit exception
        context through the re-raise site; we do not assert on ``__context__``
        directly since that coupling is an implementation detail, but we do
        verify the headline exception type flipped from
        ComposerPluginCrashError to AttributeError.
        """
        from structlog.testing import capture_logs

        catalog = _mock_catalog()
        settings = _make_settings(data_dir=self.data_dir)
        service = ComposerServiceImpl(
            catalog=catalog,
            settings=settings,
            sessions_service=_test_sessions_service(self.engine, self.data_dir),
            session_engine=self.engine,
        )
        state = _empty_state()

        valid_call = _make_llm_response(
            tool_calls=[
                {
                    "id": "c1",
                    "name": "set_source",
                    "arguments": {
                        "plugin": "csv",
                        "on_success": "out",
                        "options": {},
                        "on_validation_failure": "quarantine",
                    },
                }
            ],
        )

        with (
            patch.object(service, "_call_llm", new_callable=AsyncMock) as mock_llm,
            patch(
                "elspeth.web.composer.tool_batch.execute_tool",
                side_effect=ValueError("original plugin bug"),
            ),
            patch.object(
                service,
                "_persist_crashed_session",
                side_effect=AttributeError("sessions_table has no attribute 'c'"),
            ),
            capture_logs() as cap_logs,
        ):
            mock_llm.return_value = valid_call
            # AttributeError escapes the narrowed catch; the outer
            # ComposerPluginCrashError is never re-raised because the
            # audit-site AttributeError propagates first.
            with pytest.raises(AttributeError) as exc_info:
                await service.compose("Setup", [], state, session_id=self.session_id)

        assert "sessions_table" in str(exc_info.value)

        # No slog event — the catch did not fire, so the structured-logging
        # path was not reached. A regression that re-widens the catch would
        # cause this assertion to fail (the event would appear).
        persistence_failure_events = [entry for entry in cap_logs if entry.get("event") == "composer_crash_persistence_failed"]
        assert persistence_failure_events == [], cap_logs

    @pytest.mark.asyncio
    async def test_persist_crashed_session_runs_off_event_loop(self) -> None:
        """_persist_crashed_session must execute in a worker thread, not
        on the event loop thread.

        The method performs a synchronous ``Engine.begin()`` + UPDATE,
        which holds the GIL and (more importantly) blocks the asyncio
        event loop for the duration of the DB round-trip. Every other
        sync DB path in the compose flow is already wrapped in
        ``asyncio.to_thread(...)``; the crash-path persistence was
        hoisted out of the main loop but not wrapped.

        Blast radius: a stalled persist blocks websocket heartbeats,
        rate-limit checks, and the per-session progress broadcasts for
        every concurrent request. Cold path, but the partial DoS
        matches the same class of regression that the tool-execution
        offloading test already guards against.
        """
        import threading

        catalog = _mock_catalog()
        settings = _make_settings(data_dir=self.data_dir)
        service = ComposerServiceImpl(
            catalog=catalog,
            settings=settings,
            sessions_service=_test_sessions_service(self.engine, self.data_dir),
            session_engine=self.engine,
        )
        state = _empty_state()

        event_loop_thread = threading.current_thread()
        persist_thread: threading.Thread | None = None

        original_persist = service._persist_crashed_session

        def capture_thread(session_id: str) -> None:
            nonlocal persist_thread
            persist_thread = threading.current_thread()
            original_persist(session_id)

        valid_call = _make_llm_response(
            tool_calls=[
                {
                    "id": "c1",
                    "name": "set_source",
                    "arguments": {
                        "plugin": "csv",
                        "on_success": "out",
                        "options": {},
                        "on_validation_failure": "quarantine",
                    },
                }
            ],
        )

        with (
            patch.object(service, "_call_llm", new_callable=AsyncMock) as mock_llm,
            patch(
                "elspeth.web.composer.tool_batch.execute_tool",
                side_effect=ValueError("plugin bug"),
            ),
            patch.object(service, "_persist_crashed_session", side_effect=capture_thread),
        ):
            mock_llm.return_value = valid_call
            with pytest.raises(ComposerPluginCrashError):
                await service.compose("Setup", [], state, session_id=self.session_id)

        assert persist_thread is not None, "_persist_crashed_session was never called"
        assert persist_thread is not event_loop_thread, (
            "_persist_crashed_session ran on the event loop thread — "
            "the synchronous Engine.begin() call blocks all concurrent "
            "requests. It must be offloaded via asyncio.to_thread(...)"
        )


class TestToolExecutionThreadOffloading:
    """execute_tool() must run in a worker thread, not the event loop thread.

    Tests capture actual thread identity rather than checking whether
    asyncio.to_thread was called — testing the behavioral property
    (event loop not blocked) regardless of the offloading mechanism.
    """

    @staticmethod
    async def _assert_tool_runs_off_event_loop(
        tool_call_response: FakeLLMResponse,
        text_response: FakeLLMResponse,
        user_message: str,
    ) -> None:
        """Shared helper: verify a tool call executes in a worker thread."""
        import threading

        catalog = _mock_catalog()
        settings = _make_settings()
        service = ComposerServiceImpl(catalog=catalog, settings=settings)
        state = _empty_state()

        event_loop_thread = threading.current_thread()
        tool_execution_thread: threading.Thread | None = None

        def _capture_thread(
            _tool_name: str,
            _arguments: dict[str, Any],
            current_state: CompositionState,
            _catalog: Any,
            **kwargs: Any,
        ) -> ToolResult:
            nonlocal tool_execution_thread
            tool_execution_thread = threading.current_thread()
            return ToolResult(
                success=True,
                updated_state=current_state,
                validation=current_state.validate(),
                affected_nodes=(),
                data={"sources": []},
            )

        with (
            patch.object(service, "_call_llm", new_callable=AsyncMock) as mock_llm,
            patch(
                "elspeth.web.composer.tool_batch.execute_tool",
                side_effect=_capture_thread,
            ),
        ):
            mock_llm.side_effect = [tool_call_response, text_response]
            await service.compose(user_message, [], state)

        assert tool_execution_thread is not None, "execute_tool was never called"
        assert tool_execution_thread is not event_loop_thread, (
            "execute_tool ran on the event loop thread — must be offloaded to a worker thread to avoid blocking"
        )

    @pytest.mark.asyncio
    async def test_discovery_tool_runs_off_event_loop_thread(self) -> None:
        """Discovery tools run in a worker thread (read-only I/O)."""
        await self._assert_tool_runs_off_event_loop(
            tool_call_response=_make_llm_response(
                tool_calls=[{"id": "c1", "name": "list_sources", "arguments": {}}],
            ),
            text_response=_make_llm_response(content="Here are the sources."),
            user_message="List sources",
        )

    @pytest.mark.asyncio
    async def test_mutation_tool_runs_off_event_loop_thread(self) -> None:
        """Mutation tools run in a worker thread (blob/secret I/O).

        Previously only discovery tools were offloaded; mutation tools
        ran synchronously on the event loop, blocking all concurrent
        requests in the single-process server.
        """
        await self._assert_tool_runs_off_event_loop(
            tool_call_response=_make_llm_response(
                tool_calls=[
                    {
                        "id": "c1",
                        "name": "set_source",
                        "arguments": {
                            "plugin": "csv",
                            "on_success": "out",
                            "options": {"path": "/data/blobs/f.csv", "schema": {"mode": "observed"}},
                            "on_validation_failure": "quarantine",
                        },
                    }
                ],
            ),
            text_response=_make_llm_response(content="Source configured."),
            user_message="Set CSV source",
        )

    @pytest.mark.asyncio
    async def test_event_loop_not_blocked_during_tool_execution(self) -> None:
        """Heartbeat regression: compose() must not block the event loop.

        Runs compose() alongside an async heartbeat coroutine. If the
        heartbeat fires on schedule (not delayed by blocking tool work),
        the event loop was free during tool execution.
        """
        import time

        catalog = _mock_catalog()
        settings = _make_settings()
        service = ComposerServiceImpl(catalog=catalog, settings=settings)
        state = _empty_state()

        # Blocking duration must be much larger than the gap threshold
        # to avoid false positives on slow/shared CI runners where OS
        # scheduler jitter can delay asyncio.sleep wakeups by 50-100ms.
        tool_block_seconds = 1.0
        heartbeat_interval = 0.05

        def _blocking_tool(
            _tool_name: str,
            _arguments: dict[str, Any],
            current_state: CompositionState,
            _catalog: Any,
            **kwargs: Any,
        ) -> ToolResult:
            time.sleep(tool_block_seconds)
            return ToolResult(
                success=True,
                updated_state=current_state,
                validation=current_state.validate(),
                affected_nodes=("source",),
                data=None,
            )

        heartbeat_times: list[float] = []

        async def heartbeat() -> None:
            while True:
                heartbeat_times.append(time.monotonic())
                await asyncio.sleep(heartbeat_interval)

        # Use a mutation tool — the original bug was specifically about
        # mutation tools running synchronously on the event loop.
        tool_call = _make_llm_response(
            tool_calls=[
                {
                    "id": "c1",
                    "name": "set_source",
                    "arguments": {
                        "plugin": "csv",
                        "on_success": "out",
                        "options": {"path": "/data/blobs/f.csv", "schema": {"mode": "observed"}},
                        "on_validation_failure": "quarantine",
                    },
                }
            ],
        )
        text = _make_llm_response(content="Done.")

        with (
            patch.object(service, "_call_llm", new_callable=AsyncMock) as mock_llm,
            patch(
                "elspeth.web.composer.tool_batch.execute_tool",
                side_effect=_blocking_tool,
            ),
        ):
            mock_llm.side_effect = [tool_call, text]
            hb_task = asyncio.create_task(heartbeat())
            try:
                await service.compose("List sources", [], state)
            finally:
                hb_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await hb_task

        # With 1.0s block and 50ms interval we expect ~20 heartbeats.
        # Require at least 4 to catch partial stalls, not just total seizure.
        min_expected = int(tool_block_seconds / heartbeat_interval) - 2
        assert len(heartbeat_times) >= min(min_expected, 4), (
            f"Only {len(heartbeat_times)} heartbeat(s) fired during {tool_block_seconds}s tool execution — event loop was likely blocked"
        )

        # Check that no heartbeat interval exceeds a generous threshold.
        # If the event loop were blocked, one interval would be ≈ tool_block_seconds.
        # The 5x multiplier (250ms) gives wide margin for OS scheduler jitter
        # on shared CI runners while still catching a 1.0s event loop block.
        max_allowed_gap = heartbeat_interval * 5  # 250ms threshold vs 1.0s block (4x safety)
        for i in range(1, len(heartbeat_times)):
            gap = heartbeat_times[i] - heartbeat_times[i - 1]
            assert gap < max_allowed_gap, (
                f"Heartbeat gap {gap:.3f}s exceeds {max_allowed_gap:.3f}s — event loop was blocked (tool takes {tool_block_seconds}s)"
            )


class TestToolArgumentError:
    """ToolArgumentError is a composer-domain exception for Tier-3 boundary failures.

    It signals that a tool handler received arguments of the wrong type or
    with semantically invalid values that could not be coerced. The compose
    loop catches this and feeds the message back to the LLM for retry. Any
    OTHER exception escaping execute_tool is a plugin bug and must crash.

    The class is deliberately a structured DTO rather than a free-form
    ``Exception`` subclass: its composed message is echoed verbatim to
    the LLM API AND recorded in the Landscape audit trail, so any
    channel that could carry an LLM-supplied value would be a secret/PII
    leak pathway. Tests below lock in the "safe by construction" shape.
    """

    def test_inherits_from_exception_directly_not_composer_service_error(self) -> None:
        """ToolArgumentError must NOT inherit from ComposerServiceError.

        If it did, the route-level ``except ComposerServiceError`` block
        in ``send_message`` (sessions/routes.py) would silently absorb
        any escaped ToolArgumentError as a 502, recreating the
        silent-laundering channel this plan closes.
        Inheriting from Exception directly ensures an escaped
        ToolArgumentError (a compose-loop bug) surfaces loudly via FastAPI's
        default handler rather than being masked.
        """
        assert issubclass(ToolArgumentError, Exception)
        assert not issubclass(ToolArgumentError, ComposerServiceError)

    def test_structured_fields_compose_canonical_message(self) -> None:
        """Constructor composes args[0] deterministically from the three fields.

        The compose loop reads ``exc.args[0]`` to build the LLM-echo
        payload, so the composition template is a documented wire
        contract — a change here is a change to what the LLM and
        Landscape see.
        """
        exc = ToolArgumentError(
            argument="content",
            expected="a string",
            actual_type="int",
        )
        assert exc.argument == "content"
        assert exc.expected == "a string"
        assert exc.actual_type == "int"
        assert exc.args[0] == "'content' must be a string, got int"
        assert str(exc) == "'content' must be a string, got int"

    def test_constructor_is_keyword_only(self) -> None:
        """Positional construction must fail — structural leak prevention.

        The whole point of the DTO shape is that there is no way to
        sneak a raw LLM-supplied value into the message. A positional
        ``ToolArgumentError(f"bad: {user_input!r}")`` would defeat
        that. Making the constructor keyword-only forces every call
        site through the three-field safe channel.
        """
        with pytest.raises(TypeError):
            cast(Any, ToolArgumentError)("content must be a string, got int")

    def test_empty_argument_rejected(self) -> None:
        """Blank ``argument`` produces a nonsensical audit record and must be rejected."""
        with pytest.raises(ValueError, match="argument must be a non-empty"):
            ToolArgumentError(argument="", expected="a string", actual_type="int")

    def test_empty_expected_rejected(self) -> None:
        """Blank ``expected`` produces a nonsensical audit record and must be rejected."""
        with pytest.raises(ValueError, match="expected must be a non-empty"):
            ToolArgumentError(argument="content", expected="", actual_type="int")

    def test_empty_actual_type_rejected(self) -> None:
        """Blank ``actual_type`` produces a nonsensical audit record and must be rejected."""
        with pytest.raises(ValueError, match="actual_type must be a non-empty"):
            ToolArgumentError(argument="content", expected="a string", actual_type="")

    def test_declared_fields_frozen_after_construction(self) -> None:
        """Declared fields must not be mutable after construction.

        The exception flows into ``composition_states`` / LLM echo as
        an immutable audit artefact. Mirror the _FROZEN_ATTRS pattern
        used by ComposerConvergenceError and ComposerPluginCrashError
        so no intermediate layer can silently rewrite what downstream
        consumers see.
        """
        exc = ToolArgumentError(
            argument="content",
            expected="a string",
            actual_type="int",
        )
        with pytest.raises(AttributeError, match="frozen after construction"):
            exc.argument = "other"
        with pytest.raises(AttributeError, match="frozen after construction"):
            exc.expected = "a dict"
        with pytest.raises(AttributeError, match="frozen after construction"):
            exc.actual_type = "str"

    def test_exception_chain_dunders_remain_writable(self) -> None:
        """__cause__, __context__, __traceback__, __notes__ must stay writable.

        ``raise ... from ...`` and ``add_note()`` rely on these being
        assignable. The freeze guard covers only the three declared
        fields — the rest of the exception machinery must work
        unchanged.
        """
        exc = ToolArgumentError(
            argument="content",
            expected="a string",
            actual_type="int",
        )
        cause = ValueError("deep cause")
        exc.__cause__ = cause
        assert exc.__cause__ is cause
        exc.add_note("diagnostic note")
        assert "diagnostic note" in exc.__notes__

    def test_supports_exception_chaining(self) -> None:
        """raise ToolArgumentError(...) from exc must preserve __cause__.

        Audit-grade error reporting depends on the cause chain surviving
        asyncio.to_thread re-raise and the service-level catch. The
        cause is carried on ``__cause__`` for debug/audit but NEVER
        echoed to the LLM (see test_tool_argument_error_subclass_
        cannot_leak_cause_to_llm).
        """
        original = ValueError("bad input")
        try:
            try:
                raise original
            except ValueError as exc:
                raise ToolArgumentError(
                    argument="content",
                    expected="a string",
                    actual_type="int",
                ) from exc
        except ToolArgumentError as wrapped:
            assert wrapped.__cause__ is original


class TestToolArgumentErrorAcrossThreadBoundary:
    """End-to-end: ToolArgumentError raised inside the worker thread is caught
    correctly by the service-level catch, with message preserved.

    Closes the sleepy-assertion gap in the mocked service-level tests
    (which raise synchronously on the mock and never exercise the real
    asyncio.to_thread re-raise path).
    """

    @pytest.fixture(autouse=True)
    def _setup(self, tmp_path: Path) -> None:
        from datetime import UTC, datetime
        from uuid import uuid4

        from sqlalchemy.pool import StaticPool

        from elspeth.web.sessions.engine import create_session_engine
        from elspeth.web.sessions.models import sessions_table
        from elspeth.web.sessions.schema import initialize_session_schema

        self.engine = create_session_engine(
            "sqlite:///:memory:",
            poolclass=StaticPool,
            connect_args={"check_same_thread": False},
        )
        initialize_session_schema(self.engine)

        self.session_id = str(uuid4())
        self.data_dir = tmp_path
        now = datetime.now(UTC)
        with self.engine.begin() as conn:
            conn.execute(
                sessions_table.insert().values(
                    id=self.session_id,
                    user_id="test-user",
                    auth_provider_type="local",
                    title="Test",
                    trust_mode="auto_commit",
                    density_default="high",
                    created_at=now,
                    updated_at=now,
                )
            )

    @pytest.mark.asyncio
    async def test_real_create_blob_type_guard_feeds_error_to_llm(self) -> None:
        """Tier-3 wrong-type ``content`` routes through ARG_ERROR (Task 13 / Wave 2).

        Post-promotion (Task 13) ``create_blob`` validates via
        :class:`CreateBlobArgumentsModel` BEFORE ``_prepare_blob_create``
        reads any field.  Pydantic rejects ``content: int`` and the
        handler re-raises :class:`pydantic.ValidationError` as
        :class:`ToolArgumentError` (pattern at ``tools.py:2320-2327``,
        rev-2 BLOCKER_A).  The LLM-facing payload is intentionally
        leak-safe: it names the argument-bundle and the expected model,
        not the offending value or per-field detail.  The structured
        Pydantic detail survives on ``__cause__`` for auditors.

        Memory: ``feedback_locked_in_buggy_expectations`` — the prior
        assertion pinned the legacy ``_prepare_blob_create`` message
        ``"'content' must be a string, got int"``; the Pydantic boundary
        now fires first and that message no longer reaches the LLM.
        """
        catalog = _mock_catalog()
        settings = _make_settings(data_dir=self.data_dir)
        service = ComposerServiceImpl(
            catalog=catalog,
            settings=settings,
            sessions_service=_test_sessions_service(self.engine, self.data_dir),
            session_engine=self.engine,
        )
        state = _empty_state()

        bad_call = _make_llm_response(
            tool_calls=[
                {
                    "id": "call_bad",
                    "name": "create_blob",
                    "arguments": {
                        "filename": "x.txt",
                        "mime_type": "text/plain",
                        "content": 42,  # wrong type
                    },
                }
            ],
        )
        text = _make_llm_response(content="Fixed.")

        with patch.object(service, "_call_llm", new_callable=AsyncMock) as mock_llm:
            mock_llm.side_effect = [bad_call, text]
            result = await service.compose("Setup", [], state, session_id=self.session_id)

        _assert_no_mutation_empty_state_blocker(
            result,
            tool_name="create_blob",
            expected_detail="ToolArgumentError",
        )
        second_call_messages = mock_llm.call_args_list[1].args[0]
        tool_messages = [m for m in second_call_messages if m.get("role") == "tool"]
        assert len(tool_messages) == 1
        error_content = json.loads(tool_messages[0]["content"])
        # New (post-promotion) LLM-facing message names the argument-bundle
        # and the Pydantic model — no leak of the raw offending value.
        assert "create_blob arguments" in error_content["error"]
        assert "CreateBlobArgumentsModel" in error_content["error"]
        # The raw value (42 / "int") MUST NOT survive into the LLM echo;
        # actual_type is the exception's class name, not the value's type.
        assert "got int" not in error_content["error"]

    @pytest.mark.asyncio
    async def test_real_set_source_from_blob_options_guard_feeds_error_to_llm(self) -> None:
        """Tier-3 non-dict ``options`` routes through ARG_ERROR (Task 13 / Wave 2).

        Post-promotion (Task 13) ``set_source_from_blob`` validates via
        :class:`SetSourceFromBlobArgumentsModel` BEFORE any blob lookup.
        Pydantic rejects ``options: str`` and the handler re-raises
        :class:`pydantic.ValidationError` as :class:`ToolArgumentError`
        (pattern at ``tools.py:2320-2327``, rev-2 BLOCKER_A).  The
        LLM-facing payload is intentionally leak-safe: argument-bundle
        name + Pydantic model name, not per-field detail.

        Memory: ``feedback_locked_in_buggy_expectations`` — the prior
        assertion pinned the legacy in-handler isinstance message
        ``"'options' must be an object, got str"``; the Pydantic
        boundary now fires earlier and that message no longer reaches
        the LLM.
        """
        from elspeth.web.composer.tools import execute_tool

        catalog = _mock_catalog()
        settings = _make_settings(data_dir=self.data_dir)
        service = ComposerServiceImpl(
            catalog=catalog,
            settings=settings,
            sessions_service=_test_sessions_service(self.engine, self.data_dir),
            session_engine=self.engine,
        )
        state = _empty_state()

        create_result = execute_tool(
            "create_blob",
            {"filename": "seed.txt", "mime_type": "text/plain", "content": "hello"},
            state,
            catalog,
            data_dir=str(self.data_dir),
            session_engine=self.engine,
            session_id=self.session_id,
            **_verbatim_blob_context(self.engine, self.session_id, "hello"),
        )
        blob_id = create_result.data["blob_id"]

        bad_call = _make_llm_response(
            tool_calls=[
                {
                    "id": "call_bad",
                    "name": "set_source_from_blob",
                    "arguments": {
                        "blob_id": blob_id,
                        "on_success": "out",
                        "options": "column=text",
                    },
                }
            ],
        )
        text = _make_llm_response(content="Fixed.")

        with patch.object(service, "_call_llm", new_callable=AsyncMock) as mock_llm:
            mock_llm.side_effect = [bad_call, text]
            result = await service.compose("Setup", [], state, session_id=self.session_id)

        _assert_no_mutation_empty_state_blocker(
            result,
            tool_name="set_source_from_blob",
            expected_detail="ToolArgumentError",
        )
        second_call_messages = mock_llm.call_args_list[1].args[0]
        tool_messages = [m for m in second_call_messages if m.get("role") == "tool"]
        assert len(tool_messages) == 1
        error_content = json.loads(tool_messages[0]["content"])
        # New (post-promotion) LLM-facing message names the argument-bundle
        # and the Pydantic model — no leak of the raw offending value or
        # its type (which the legacy handler echoed as "got str").
        assert "set_source_from_blob arguments" in error_content["error"]
        assert "SetSourceFromBlobArgumentsModel" in error_content["error"]
        assert "got str" not in error_content["error"]
        assert "column=text" not in error_content["error"]


class TestComposerErrorConstructionInvariants:
    """Type-level invariants for composer service exceptions.

    These exceptions flow into HTTP responses (as error_type/detail bodies)
    and into Landscape (via partial_state persistence in composition_states
    and structured-log exc_class correlation). Post-construction attribute
    reassignment would let any layer silently rewrite what downstream HTTP
    and audit consumers see. The class-level freeze and the ``capture()``
    classmethod encode the "partial_state only when state.version >
    initial_version" invariant mechanically rather than relying on each
    raise site to apply the rule by hand.
    """

    def test_plugin_crash_error_attributes_are_frozen_after_construction(self) -> None:
        exc = ComposerPluginCrashError(ValueError("boom"), partial_state=None)

        with pytest.raises(AttributeError, match="frozen"):
            exc.__setattr__("original_exc", RuntimeError("replaced"))

        with pytest.raises(AttributeError, match="frozen"):
            exc.__setattr__("partial_state", _empty_state())

        with pytest.raises(AttributeError, match="frozen"):
            exc.__setattr__("exc_class", "PrettyException")

    def test_plugin_crash_error_allows_exception_chain_machinery(self) -> None:
        # __cause__, __context__, __suppress_context__, __traceback__, and
        # add_note() target BaseException-managed slots, not our declared
        # attrs. The freeze MUST NOT break `raise X from Y` or add_note.
        root = RuntimeError("underlying")
        exc = ComposerPluginCrashError(ValueError("boom"))
        exc.__cause__ = root
        exc.__suppress_context__ = True
        exc.add_note("operator triage hint")

        assert exc.__cause__ is root
        assert exc.__suppress_context__ is True
        assert "operator triage hint" in exc.__notes__

    def test_convergence_error_attributes_are_frozen_after_construction(self) -> None:
        exc = ComposerConvergenceError(
            max_turns=3,
            budget_exhausted="composition",
            partial_state=None,
        )

        with pytest.raises(AttributeError, match="frozen"):
            exc.__setattr__("max_turns", 99)

        with pytest.raises(AttributeError, match="frozen"):
            exc.__setattr__("budget_exhausted", "timeout")

        with pytest.raises(AttributeError, match="frozen"):
            exc.__setattr__("partial_state", _empty_state())

    def test_plugin_crash_capture_returns_none_when_state_not_mutated(self) -> None:
        # Invariant: partial_state is None when state.version == initial_version
        # (no tool call successfully mutated state before the crash).
        state = _empty_state()  # version=1
        exc = ComposerPluginCrashError.capture(
            KeyError("missing"),
            state=state,
            initial_version=state.version,
        )

        assert exc.partial_state is None
        assert exc.exc_class == "KeyError"

    def test_plugin_crash_capture_returns_state_when_mutated(self) -> None:
        # Invariant: partial_state IS the state when state.version moved
        # beyond initial_version (at least one tool call persisted).
        initial = _empty_state()  # version=1
        mutated = CompositionState(
            source=None,
            nodes=(),
            edges=(),
            outputs=(),
            metadata=PipelineMetadata(),
            version=initial.version + 2,
        )
        exc = ComposerPluginCrashError.capture(
            ValueError("boom"),
            state=mutated,
            initial_version=initial.version,
        )

        assert exc.partial_state is mutated
        assert exc.exc_class == "ValueError"

    def test_convergence_capture_returns_none_when_state_not_mutated(self) -> None:
        state = _empty_state()
        exc = ComposerConvergenceError.capture(
            max_turns=5,
            budget_exhausted="composition",
            state=state,
            initial_version=state.version,
        )

        assert exc.partial_state is None
        assert exc.max_turns == 5
        assert exc.budget_exhausted == "composition"

    def test_convergence_capture_returns_state_when_mutated(self) -> None:
        initial = _empty_state()
        mutated = CompositionState(
            source=None,
            nodes=(),
            edges=(),
            outputs=(),
            metadata=PipelineMetadata(),
            version=initial.version + 1,
        )
        exc = ComposerConvergenceError.capture(
            max_turns=7,
            budget_exhausted="timeout",
            state=mutated,
            initial_version=initial.version,
        )

        assert exc.partial_state is mutated
        assert exc.budget_exhausted == "timeout"


class TestComposerRuntimePreflightCacheAndTimeout:
    @pytest.mark.asyncio
    async def test_runtime_preflight_cache_reuses_same_state_version_and_settings_hash(self) -> None:
        service = ComposerServiceImpl(catalog=_mock_catalog(), settings=_make_settings())
        state = _empty_state()
        cache = service._new_runtime_preflight_cache()
        preflight = ValidationResult(is_valid=True, checks=[], errors=[])

        with patch.object(service, "_runtime_preflight", return_value=preflight) as mock_preflight:
            first = await service._cached_runtime_preflight(
                state,
                user_id="user-1",
                cache=cache,
                initial_version=state.version,
                session_scope="session:test",
            )
            second = await service._cached_runtime_preflight(
                state,
                user_id="user-1",
                cache=cache,
                initial_version=state.version,
                session_scope="session:test",
            )

        assert first is preflight
        assert second is preflight
        mock_preflight.assert_called_once_with(state, "user-1")

    @pytest.mark.asyncio
    async def test_runtime_preflight_timeout_is_cached_for_compose_call(self) -> None:
        settings = _make_settings(composer_runtime_preflight_timeout_seconds=0.01)
        service = ComposerServiceImpl(catalog=_mock_catalog(), settings=settings)
        state = _empty_state()
        cache = service._new_runtime_preflight_cache()
        started = threading.Event()
        release = threading.Event()

        def slow_preflight(candidate: CompositionState, user_id: str | None) -> ValidationResult:
            started.set()
            release.wait(timeout=30)
            return ValidationResult(is_valid=True, checks=[], errors=[])

        try:
            with patch.object(service, "_runtime_preflight", side_effect=slow_preflight) as mock_preflight:
                with pytest.raises(ComposerRuntimePreflightError) as first:
                    await service._cached_runtime_preflight(
                        state,
                        user_id="user-1",
                        cache=cache,
                        initial_version=state.version - 1,
                        session_scope="session:test",
                    )
                assert first.value.exc_class == "TimeoutError"
                assert started.is_set()

                with pytest.raises(ComposerRuntimePreflightError) as second:
                    await service._cached_runtime_preflight(
                        state,
                        user_id="user-1",
                        cache=cache,
                        initial_version=state.version - 1,
                        session_scope="session:test",
                    )

                assert second.value.exc_class == "TimeoutError"
                mock_preflight.assert_called_once()
        finally:
            release.set()

    def test_runtime_preflight_settings_hash_is_non_secret(self) -> None:
        class FakeSettings:
            data_dir = Path("/tmp/elspeth-data")
            landscape_passphrase = "SECRET_CANARY_SHOULD_NOT_APPEAR"

        digest = runtime_preflight_settings_hash(FakeSettings())

        assert "SECRET_CANARY" not in digest
        assert len(digest) == 64


class TestComposerRuntimePreflightFinalGate:
    def test_literal_credential_preflight_repair_requires_secret_inventory_diagnosis(self) -> None:
        """elspeth-697b7377a7: literal credential failures must drive
        secret inventory diagnosis, not another copy of the validate complaint.
        """
        invalid_preflight = ValidationResultModel(
            is_valid=False,
            checks=[],
            errors=[
                ValidationError(
                    component_id="summarize",
                    component_type="transform",
                    message="Credential field(s) api_key contain a literal value; expected a wired secret reference.",
                    suggestion="Wire each credential field through the Secrets panel.",
                    error_code="fabricated_secret",
                )
            ],
            readiness=_not_authoring_ready("fabricated_secret"),
        )

        repair = _compose_preflight_repair_message(invalid_preflight, next_turn=1)

        assert "list_secret_refs" in repair
        assert "validate_secret_ref" in repair
        assert "reason" in repair
        assert "fingerprint_resolver_not_configured" in repair
        assert "Do not answer by repeating the runtime preflight complaint" in repair
        assert "Do not inline a literal credential" in repair

    @pytest.mark.asyncio
    async def test_changed_state_completion_is_replaced_when_runtime_preflight_fails(self) -> None:
        catalog = _mock_catalog()
        settings = _make_settings()
        service = ComposerServiceImpl(catalog=catalog, settings=settings)
        state = _empty_state()
        changed_state = state.with_source(
            SourceSpec(
                plugin="csv",
                on_success="main",
                options={"path": "/data/blobs/input.csv", "schema": {"mode": "observed"}},
                on_validation_failure="discard",
            )
        ).with_output(
            OutputSpec(
                name="main",
                plugin="csv",
                options={"path": "/data/outputs/out.csv", "schema": {"mode": "observed"}},
                on_write_failure="discard",
            )
        )
        changed_state = replace(changed_state, version=state.version + 1)

        llm_response = _make_llm_response(content="The pipeline is complete and valid.")
        failed_preflight = ValidationResult(
            is_valid=False,
            checks=[
                ValidationCheck(
                    name="settings_load",
                    passed=False,
                    detail="Forbidden name: 'end_of_source'",
                    affected_nodes=(),
                    outcome_code=None,
                )
            ],
            errors=[
                ValidationError(
                    component_id="agg1",
                    component_type="aggregation",
                    message="Forbidden name: 'end_of_source'",
                    suggestion="Omit trigger for end-of-source-only aggregation.",
                    error_code=None,
                )
            ],
        )

        with patch.object(service, "_call_llm", new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = llm_response
            with patch.object(service, "_runtime_preflight", return_value=failed_preflight) as mock_preflight:
                result = await service._finalize_no_tool_response(
                    content="The pipeline is complete and valid.",
                    state=changed_state,
                    initial_version=state.version,
                    user_id="user-1",
                    last_runtime_preflight=None,
                    runtime_preflight_cache=service._new_runtime_preflight_cache(),
                    session_scope="session:test",
                )

        assert result.message != "The pipeline is complete and valid."
        assert result.raw_assistant_content == "The pipeline is complete and valid."
        assert result.runtime_preflight is failed_preflight
        mock_preflight.assert_called_once_with(changed_state, "user-1")

    @pytest.mark.asyncio
    async def test_pending_interpretation_handoff_is_not_augmented_as_invalid_config(self) -> None:
        catalog = _mock_catalog()
        settings = _make_settings()
        service = ComposerServiceImpl(catalog=catalog, settings=settings)
        state = _empty_state().with_source(
            SourceSpec(
                plugin="csv",
                on_success="rate_node",
                options={"path": "/data/blobs/input.csv", "schema": {"mode": "observed"}},
                on_validation_failure="discard",
            )
        )
        changed_state = replace(state, version=state.version + 1)
        pending_preflight = ValidationResult(
            is_valid=False,
            checks=[
                ValidationCheck(
                    name="interpretation_review",
                    passed=False,
                    detail="Interpretation review is pending for rate_node:cool.",
                    affected_nodes=("rate_node",),
                    outcome_code=None,
                )
            ],
            errors=[
                ValidationError(
                    component_id="rate_node",
                    component_type="transform",
                    message="Interpretation review is pending for 'cool'.",
                    suggestion="Resolve the pending interpretation review before running.",
                    error_code=INTERPRETATION_REVIEW_PENDING_CODE,
                )
            ],
            readiness=_pending_interpretation_readiness(),
        )
        model_prose = "Review is pending for cool."

        with patch.object(service, "_runtime_preflight", return_value=pending_preflight) as mock_preflight:
            result = await service._finalize_no_tool_response(
                content=model_prose,
                state=changed_state,
                initial_version=state.version,
                user_id="user-1",
                last_runtime_preflight=None,
                runtime_preflight_cache=service._new_runtime_preflight_cache(),
                session_scope="session:test",
                mutation_success_seen=True,
            )

        assert result.message == model_prose
        assert result.raw_assistant_content is None
        assert result.runtime_preflight is pending_preflight
        mock_preflight.assert_called_once_with(changed_state, "user-1")

    @pytest.mark.asyncio
    async def test_unchanged_text_without_preview_does_not_run_preflight(self) -> None:
        catalog = _mock_catalog()
        settings = _make_settings()
        service = ComposerServiceImpl(catalog=catalog, settings=settings)
        state = _empty_state()

        with patch.object(service, "_runtime_preflight") as mock_preflight:
            result = await service._finalize_no_tool_response(
                content="I can help with that.",
                state=state,
                initial_version=state.version,
                user_id="user-1",
                last_runtime_preflight=None,
                runtime_preflight_cache=service._new_runtime_preflight_cache(),
                session_scope="session:test",
            )

        assert result.message == "I can help with that."
        assert result.raw_assistant_content is None
        assert result.runtime_preflight is None
        mock_preflight.assert_not_called()

    @pytest.mark.asyncio
    async def test_unchanged_state_reuses_preview_preflight_without_rerun(self) -> None:
        catalog = _mock_catalog()
        settings = _make_settings()
        service = ComposerServiceImpl(catalog=catalog, settings=settings)
        state = _empty_state()
        preview_preflight = ValidationResult(
            is_valid=False,
            checks=[
                ValidationCheck(
                    name="settings_load",
                    passed=False,
                    detail="Forbidden name: 'end_of_source'",
                    affected_nodes=(),
                    outcome_code=None,
                )
            ],
            errors=[
                ValidationError(
                    component_id="agg1",
                    component_type="aggregation",
                    message="Forbidden name: 'end_of_source'",
                    suggestion=None,
                    error_code=None,
                )
            ],
        )

        with patch.object(service, "_runtime_preflight") as mock_preflight:
            result = await service._finalize_no_tool_response(
                content="The pipeline is complete and valid.",
                state=state,
                initial_version=state.version,
                user_id="user-1",
                last_runtime_preflight=preview_preflight,
                runtime_preflight_cache=service._new_runtime_preflight_cache(),
                session_scope="session:test",
            )

        assert result.message != "The pipeline is complete and valid."
        assert result.raw_assistant_content == "The pipeline is complete and valid."
        assert result.runtime_preflight is preview_preflight
        mock_preflight.assert_not_called()

    @pytest.mark.asyncio
    async def test_unchanged_state_reuses_valid_preview_preflight_without_replacement(self) -> None:
        """Reuse path with is_valid=True must preserve the LLM message verbatim.

        Complement to test_unchanged_state_reuses_preview_preflight_without_rerun:
        that test exercises the invalid-cached branch (which replaces the
        message). This test pins the valid-cached branch (which must keep the
        LLM message intact and not populate raw_assistant_content).
        """
        catalog = _mock_catalog()
        settings = _make_settings()
        service = ComposerServiceImpl(catalog=catalog, settings=settings)
        state = _empty_state()
        passing_preview_preflight = ValidationResult(is_valid=True, checks=[], errors=[])

        with patch.object(service, "_runtime_preflight") as mock_preflight:
            result = await service._finalize_no_tool_response(
                content="The pipeline is complete and valid.",
                state=state,
                initial_version=state.version,
                user_id="user-1",
                last_runtime_preflight=passing_preview_preflight,
                runtime_preflight_cache=service._new_runtime_preflight_cache(),
                session_scope="session:test",
            )

        assert result.message == "The pipeline is complete and valid."
        assert result.raw_assistant_content is None
        assert result.runtime_preflight is passing_preview_preflight
        mock_preflight.assert_not_called()

    @pytest.mark.asyncio
    async def test_passing_preflight_preserves_original_message_verbatim(self) -> None:
        catalog = _mock_catalog()
        settings = _make_settings()
        service = ComposerServiceImpl(catalog=catalog, settings=settings)
        state = _empty_state()
        changed_state = replace(state, version=state.version + 1)
        passed_preflight = ValidationResult(is_valid=True, checks=[], errors=[])

        with patch.object(service, "_runtime_preflight", return_value=passed_preflight):
            result = await service._finalize_no_tool_response(
                content="The pipeline is complete and valid.",
                state=changed_state,
                initial_version=state.version,
                user_id="user-1",
                last_runtime_preflight=None,
                runtime_preflight_cache=service._new_runtime_preflight_cache(),
                session_scope="session:test",
            )

        assert result.message == "The pipeline is complete and valid."
        assert result.raw_assistant_content is None
        assert result.runtime_preflight is passed_preflight

    @pytest.mark.asyncio
    async def test_unexpected_preflight_exception_preserves_partial_state(self) -> None:
        catalog = _mock_catalog()
        settings = _make_settings()
        service = ComposerServiceImpl(catalog=catalog, settings=settings)
        state = _empty_state()
        changed_state = replace(state, version=state.version + 1)

        with (
            patch.object(service, "_runtime_preflight", side_effect=RuntimeError("boom")),
            pytest.raises(ComposerRuntimePreflightError) as exc_info,
        ):
            await service._finalize_no_tool_response(
                content="The pipeline is complete.",
                state=changed_state,
                initial_version=state.version,
                user_id="user-1",
                last_runtime_preflight=None,
                runtime_preflight_cache=service._new_runtime_preflight_cache(),
                session_scope="session:test",
            )

        assert exc_info.value.partial_state == changed_state
        assert exc_info.value.exc_class == "RuntimeError"

    def test_runtime_preflight_error_original_exception_is_frozen(self) -> None:
        original = RuntimeError("boom")
        error = ComposerRuntimePreflightError(original_exc=original, partial_state=None)

        with pytest.raises(AttributeError):
            error.original_exc = RuntimeError("replacement")


class TestEmptyStateFinalizePassthrough:
    """Tier 1.5 §7.6 followup — empty-state finalize-time passthrough.

    Audit-DB inspection of three captured rag-text-llm REDs (2026-05-06
    cohort, sessions 2cf59016, 12f061d9, 29ef178e) showed the model spent
    20+ tool calls trying to converge on a valid set_pipeline build, gave
    up, and produced honest prose explaining what it tried and what's
    blocking. The synthesizer was discarding this prose and replacing it
    with raw Pydantic noise ("source: Field required, sinks: Field
    required") that looked like a system bug to a viewer.

    The structural fix: when state is structurally empty (no source, no
    nodes, no outputs) at finalize time AND the cached/computed runtime
    preflight is invalid, the model isn't lying about completion — it's
    reporting honest failure. Pass through its content with a
    system-attributed suffix telling the user what to do next.

    These tests pin the new behaviour. The original synthesizer path
    (non-empty state with invalid preflight) is left unchanged and tested
    by sister tests above.
    """

    # ── Standalone tests on the helpers ─────────────────────────────────

    def test_state_is_structurally_empty_returns_true_for_empty_state(self) -> None:
        from elspeth.web.composer.service import _state_is_structurally_empty

        state = _empty_state()
        assert _state_is_structurally_empty(state) is True

    def test_state_is_structurally_empty_false_with_source(self) -> None:
        from elspeth.web.composer.service import _state_is_structurally_empty

        state = _empty_state().with_source(
            SourceSpec(plugin="csv", on_success="t1", options={"path": "/tmp/x.csv"}, on_validation_failure="discard")
        )
        assert _state_is_structurally_empty(state) is False

    def test_state_is_structurally_empty_false_with_output(self) -> None:
        from elspeth.web.composer.service import _state_is_structurally_empty

        state = _empty_state().with_output(
            OutputSpec(
                name="main", plugin="csv", options={"path": "/tmp/y.csv", "schema": {"mode": "observed"}}, on_write_failure="discard"
            )
        )
        assert _state_is_structurally_empty(state) is False

    def test_compose_empty_state_message_appends_system_suffix(self) -> None:
        from elspeth.web.composer.service import _compose_empty_state_message

        content = "I tried to build the pipeline but couldn't converge."
        msg = _compose_empty_state_message(content)
        # Original content preserved (model's prose is the truthful part).
        assert content in msg
        # System-attributed marker present (UI / scorers can detect).
        assert "[ELSPETH-SYSTEM]" in msg
        # Concrete next-step guidance.
        assert "refine" in msg.lower() or "retry" in msg.lower()

    def test_compose_empty_state_message_handles_empty_content(self) -> None:
        """Edge case: model produced no content at all. Suffix becomes the
        whole message (better than silence)."""
        from elspeth.web.composer.service import _compose_empty_state_message

        msg = _compose_empty_state_message("")
        assert "[ELSPETH-SYSTEM]" in msg
        assert msg.startswith("[ELSPETH-SYSTEM]") or msg.lstrip().startswith("[ELSPETH-SYSTEM]")

    def test_compose_empty_state_message_with_blocker_includes_cause(self) -> None:
        """When a concrete blocker is supplied (no-mutation empty-state augmentation),
        the suffix surfaces it so the operator sees the precise cause without
        having to consult the audit DB. This is defense-in-depth: the model's
        prose usually mentions the blocker, but not always.
        """
        from elspeth.web.composer.service import _compose_empty_state_message

        content = "I tried to build but the source binding failed."
        blocker = "set_pipeline returned success=false: schema: Field required"
        msg = _compose_empty_state_message(content, blocker=blocker)
        # Model prose preserved verbatim at start.
        assert msg.startswith(content)
        # System suffix attribution + cause both present.
        assert "[ELSPETH-SYSTEM]" in msg
        assert "Cause:" in msg
        assert blocker in msg

    def test_compose_empty_state_message_without_blocker_uses_generic_suffix(self) -> None:
        """The preflight-invalid empty-state augmentation does not have a
        single concrete blocker — runtime_result already carries multi-error
        data — so it passes ``blocker=None`` and gets the generic suffix.
        The ``Cause:`` field is omitted to avoid implying a cause that
        isn't there.
        """
        from elspeth.web.composer.service import _compose_empty_state_message

        msg = _compose_empty_state_message("I tried.", blocker=None)
        assert "[ELSPETH-SYSTEM]" in msg
        assert "Cause:" not in msg

    def test_enforce_augmentation_prefix_invariant_accepts_prefixed_message(self) -> None:
        """The contract holds when the augmented message has content as a strict prefix.

        Empty content is also accepted because ``"".startswith("")`` is trivially True
        and the empty-state augmentation builder degenerates to suffix-only output
        for empty inputs.
        """
        from elspeth.web.composer.service import _enforce_augmentation_prefix_invariant

        _enforce_augmentation_prefix_invariant(branch="test", content="model prose", augmented="model prose [ELSPETH-SYSTEM] suffix")
        _enforce_augmentation_prefix_invariant(branch="test", content="model prose", augmented="model prose")
        _enforce_augmentation_prefix_invariant(branch="test", content="", augmented="any suffix")

    def test_enforce_augmentation_prefix_invariant_raises_on_violation(self) -> None:
        """A producer that violates the prefix invariant raises AuditIntegrityError
        rather than committing a corrupt audit row that the consumer-side
        discriminator at routes._composer_history_content would silently misroute
        as replacement (LLM gets [INTERCEPTED] prefixed onto its own prose).
        """
        from elspeth.contracts.errors import AuditIntegrityError
        from elspeth.web.composer.service import _enforce_augmentation_prefix_invariant

        with pytest.raises(AuditIntegrityError) as exc_info:
            _enforce_augmentation_prefix_invariant(
                branch="no_mutation_empty_state_augmentation",
                content="model prose",
                augmented="[ELSPETH-SYSTEM] something else",
            )
        message = str(exc_info.value)
        assert "Tier 1" in message
        assert "no_mutation_empty_state_augmentation" in message
        assert "augmentation" in message
        assert "discriminator" in message

    # ── _last_mutation_was_pending_proposal helper ───────────────────────

    @staticmethod
    def _proposal_invocation(
        *,
        tool_name: str = "set_pipeline",
        version: int = 1,
    ) -> Any:
        """Build an invocation whose result_canonical mirrors the proposal-payload shape.

        Mirrors the proposal_result envelope written at composer/service.py
        line ~2697: ToolResult(success=True, data={status: APPROVAL_REQUIRED, ...}).
        The blocker classifier reads result_canonical, so the test pins the
        wire shape — if the proposal payload moves, this test will catch it.
        """
        from datetime import UTC, datetime

        from elspeth.contracts.composer_audit import ComposerToolInvocation, ComposerToolStatus
        from elspeth.core.canonical import canonical_json, stable_hash

        result_payload = {
            "success": True,
            "data": {
                "status": "APPROVAL_REQUIRED",
                "proposal_id": "00000000-0000-0000-0000-000000000000",
                "tool_name": tool_name,
                "summary": "Replace the pipeline with csv input, 3 transforms, and 1 output.",
                "message": "The requested pipeline change is pending human approval and has not been applied.",
            },
        }
        canon = canonical_json(result_payload)
        h = stable_hash(result_payload)
        t = datetime(2026, 5, 14, 21, 29, 5, tzinfo=UTC)
        return ComposerToolInvocation(
            tool_call_id="call_proposal",
            tool_name=tool_name,
            arguments_canonical=b"{}",
            arguments_hash="0" * 64,
            result_canonical=canon,
            result_hash=h,
            status=ComposerToolStatus.SUCCESS,
            error_class=None,
            error_message=None,
            version_before=version,
            version_after=version,
            started_at=t,
            finished_at=t,
            latency_ms=1,
            actor="test",
        )

    def test_last_mutation_was_pending_proposal_true_for_approval_required_payload(self) -> None:
        from elspeth.web.composer.service import _last_mutation_was_pending_proposal

        invocations = (self._proposal_invocation(),)
        assert _last_mutation_was_pending_proposal(invocations) is True

    def test_last_mutation_was_pending_proposal_false_for_empty_invocations(self) -> None:
        from elspeth.web.composer.service import _last_mutation_was_pending_proposal

        assert _last_mutation_was_pending_proposal(()) is False

    def test_last_mutation_was_pending_proposal_skips_discovery_tools(self) -> None:
        """Discovery tools (get_plugin_schema, list_*) are transparent — the
        model interleaves them between real mutations. A proposal followed by
        a discovery call still counts as a pending-proposal turn."""
        from datetime import UTC, datetime

        from elspeth.contracts.composer_audit import ComposerToolInvocation, ComposerToolStatus
        from elspeth.web.composer.service import _last_mutation_was_pending_proposal

        proposal = self._proposal_invocation()
        discovery = ComposerToolInvocation(
            tool_call_id="call_discovery",
            tool_name="get_plugin_schema",
            arguments_canonical=b"{}",
            arguments_hash="0" * 64,
            result_canonical=b'{"success": true}',
            result_hash="1" * 64,
            status=ComposerToolStatus.SUCCESS,
            error_class=None,
            error_message=None,
            version_before=1,
            version_after=1,
            started_at=datetime(2026, 5, 14, 21, 29, 5, tzinfo=UTC),
            finished_at=datetime(2026, 5, 14, 21, 29, 5, tzinfo=UTC),
            latency_ms=1,
            actor="test",
        )
        assert _last_mutation_was_pending_proposal((proposal, discovery)) is True

    def test_last_mutation_was_pending_proposal_false_when_most_recent_mutation_is_arg_error(self) -> None:
        """An ARG_ERROR after a successful proposal means the model retried
        and the retry failed. The augmentation should fire."""
        from datetime import UTC, datetime

        from elspeth.contracts.composer_audit import ComposerToolInvocation, ComposerToolStatus
        from elspeth.web.composer.service import _last_mutation_was_pending_proposal

        proposal = self._proposal_invocation()
        arg_error = ComposerToolInvocation(
            tool_call_id="call_failed",
            tool_name="set_pipeline",
            arguments_canonical=b"{}",
            arguments_hash="0" * 64,
            result_canonical=None,
            result_hash=None,
            status=ComposerToolStatus.ARG_ERROR,
            error_class="ToolArgumentError",
            error_message="bad shape",
            version_before=1,
            version_after=None,
            started_at=datetime(2026, 5, 14, 21, 29, 5, tzinfo=UTC),
            finished_at=datetime(2026, 5, 14, 21, 29, 5, tzinfo=UTC),
            latency_ms=1,
            actor="test",
        )
        assert _last_mutation_was_pending_proposal((proposal, arg_error)) is False

    def test_last_mutation_was_pending_proposal_false_for_create_blob_success(self) -> None:
        """create_blob success without state advance is the original target
        of the empty-state augmentation — it must continue to fire."""
        from datetime import UTC, datetime

        from elspeth.contracts.composer_audit import ComposerToolInvocation, ComposerToolStatus
        from elspeth.web.composer.service import _last_mutation_was_pending_proposal

        create_blob_inv = ComposerToolInvocation(
            tool_call_id="call_blob",
            tool_name="create_blob",
            arguments_canonical=b"{}",
            arguments_hash="0" * 64,
            result_canonical=b'{"success": true, "data": {"blob_id": "abc"}}',
            result_hash="2" * 64,
            status=ComposerToolStatus.SUCCESS,
            error_class=None,
            error_message=None,
            version_before=1,
            version_after=1,
            started_at=datetime(2026, 5, 14, 21, 29, 5, tzinfo=UTC),
            finished_at=datetime(2026, 5, 14, 21, 29, 5, tzinfo=UTC),
            latency_ms=1,
            actor="test",
        )
        assert _last_mutation_was_pending_proposal((create_blob_inv,)) is False

    # ── End-to-end: pending-proposal suppression of empty-state augmentation ─

    @pytest.mark.asyncio
    async def test_pending_proposal_does_not_trigger_empty_state_augmentation(self) -> None:
        """Reproduces the convergence-failure path from staging session
        d121ba9c-8775-463d-afdf-75fd6b6f2456: under trust_mode=explicit_approve
        a successful set_pipeline produces a pending proposal with
        version_after == version_before. The empty-state augmentation gate
        previously misclassified this as no-mutation and appended
        '[ELSPETH-SYSTEM] The pipeline is still empty', derailing both the
        operator's framing and (via the synthesized suffix being re-read on
        subsequent turns) the model's own state model."""
        catalog = _mock_catalog()
        settings = _make_settings()
        service = ComposerServiceImpl(catalog=catalog, settings=settings)
        state = _empty_state()
        model_prose = (
            "I picked 5 Australian Government agency pages and prepared the workflow, "
            "but the platform has put the pipeline change into human-approval pending "
            "state, so it has not been applied yet."
        )

        with patch.object(service, "_runtime_preflight") as mock_preflight:
            result = await service._finalize_no_tool_response(
                content=model_prose,
                state=state,
                initial_version=state.version,
                user_id="user-1",
                last_runtime_preflight=None,
                runtime_preflight_cache=service._new_runtime_preflight_cache(),
                session_scope="session:test",
                user_message="please build me a pipeline",
                tool_invocations=(self._proposal_invocation(),),
            )

        # Model's truthful pending-approval prose preserved verbatim.
        assert result.message == model_prose
        # No "[ELSPETH-SYSTEM] pipeline is still empty" suffix — the build
        # succeeded as a pending proposal; that is not an empty-state failure.
        assert "[ELSPETH-SYSTEM]" not in result.message
        assert "pipeline is still empty" not in result.message
        # No re-run of runtime preflight (state.version unchanged).
        mock_preflight.assert_not_called()

    @pytest.mark.asyncio
    async def test_empty_state_invalid_preflight_passes_through_model_content(self) -> None:
        """The captured rag-text-llm RED scenario: empty state, invalid
        cached preflight, model gave up with honest prose. Synthesizer
        must NOT fire."""
        catalog = _mock_catalog()
        settings = _make_settings()
        service = ComposerServiceImpl(catalog=catalog, settings=settings)
        state = _empty_state()
        invalid_preflight = ValidationResult(
            is_valid=False,
            checks=[
                ValidationCheck(
                    name="settings_load",
                    passed=False,
                    detail="Pipeline state is empty",
                    affected_nodes=(),
                    outcome_code=None,
                )
            ],
            errors=[
                ValidationError(
                    component_id=None,
                    component_type=None,
                    message="2 validation errors for ElspethSettings — source: Field required, sinks: Field required",
                    suggestion=None,
                    error_code=None,
                )
            ],
        )
        model_prose = (
            "I did discover the needed plugin requirements: web_scrape needs explicit "
            "schema, url_field, content_field, fingerprint_field, and http. The setup "
            "needs one more valid build pass to supply the right options."
        )

        with patch.object(service, "_runtime_preflight") as mock_preflight:
            result = await service._finalize_no_tool_response(
                content=model_prose,
                state=state,
                initial_version=state.version,
                user_id="user-1",
                last_runtime_preflight=invalid_preflight,
                runtime_preflight_cache=service._new_runtime_preflight_cache(),
                session_scope="session:test",
            )

        # Model's prose preserved as the load-bearing part of the message.
        assert model_prose in result.message
        # System suffix appended.
        assert "[ELSPETH-SYSTEM]" in result.message
        # Synthesized Pydantic-noise message did NOT replace the prose.
        assert "Field required" not in result.message
        assert "I cannot mark this pipeline complete yet" not in result.message
        # Audit fields preserved (matches synthesizer-path semantics).
        assert result.raw_assistant_content == model_prose
        assert result.runtime_preflight is invalid_preflight
        # No re-run of runtime preflight (state.version unchanged).
        mock_preflight.assert_not_called()

    @pytest.mark.asyncio
    async def test_empty_state_invalid_preflight_after_state_mutation_run_preflight(self) -> None:
        """Even when the preflight is computed fresh (state.version > initial),
        the empty-state branch fires if the resulting state is empty."""
        catalog = _mock_catalog()
        settings = _make_settings()
        service = ComposerServiceImpl(catalog=catalog, settings=settings)
        state = _empty_state()
        # Simulate state.version bump without populating any state fields
        # (e.g., the model called a mutation that net-emptied the state).
        bumped_state = replace(state, version=state.version + 1)
        invalid_preflight = ValidationResult(
            is_valid=False,
            checks=[],
            errors=[
                ValidationError(
                    component_id=None,
                    component_type=None,
                    message="Pipeline empty",
                    suggestion=None,
                    error_code=None,
                )
            ],
        )

        with patch.object(service, "_runtime_preflight", return_value=invalid_preflight):
            result = await service._finalize_no_tool_response(
                content="Model prose explaining the gap.",
                state=bumped_state,
                initial_version=state.version,
                user_id="user-1",
                last_runtime_preflight=None,
                runtime_preflight_cache=service._new_runtime_preflight_cache(),
                session_scope="session:test",
            )

        assert "Model prose explaining the gap." in result.message
        assert "[ELSPETH-SYSTEM]" in result.message
        assert "Pipeline empty" not in result.message  # synthesizer skipped

    @pytest.mark.asyncio
    async def test_non_empty_state_invalid_preflight_augments_with_validator_suffix(self) -> None:
        """When state is populated AND runtime preflight is invalid, the
        finalizer augments the model's prose with a system-attributed
        suffix naming the validator's objection (issue elspeth-9cfbad6901).

        Panel-evals evidence (fork_coalesce__p4_adversarial_engineer,
        boolean_routing__p3_marketingops) showed the model's prose in
        this case is typically substantive disclosure rather than a
        false completion claim. The earlier replacement-shape policy
        discarded the prose; the augmentation shape preserves it
        verbatim while still surfacing the validator's reason to the
        operator. The model's next preview_pipeline call drives self-
        correction so the [INTERCEPTED] framing is no longer required.
        """
        catalog = _mock_catalog()
        settings = _make_settings()
        service = ComposerServiceImpl(catalog=catalog, settings=settings)
        state = (
            _empty_state()
            .with_source(
                SourceSpec(
                    plugin="csv",
                    on_success="t1",
                    options={"path": "/data/inputs/in.csv"},
                    on_validation_failure="discard",
                )
            )
            .with_output(
                OutputSpec(
                    name="main",
                    plugin="csv",
                    options={"path": "/data/outputs/out.csv", "schema": {"mode": "observed"}},
                    on_write_failure="discard",
                )
            )
        )
        bumped_state = replace(state, version=state.version + 1)
        invalid_preflight = ValidationResult(
            is_valid=False,
            checks=[],
            errors=[
                ValidationError(
                    component_id="t1",
                    component_type="transform",
                    message="Forbidden name: 'end_of_source'",
                    suggestion="Omit trigger for end-of-source-only aggregation.",
                    error_code=None,
                )
            ],
        )
        model_prose = "The pipeline is complete and valid."

        with patch.object(service, "_runtime_preflight", return_value=invalid_preflight):
            result = await service._finalize_no_tool_response(
                content=model_prose,
                state=bumped_state,
                initial_version=state.version,
                user_id="user-1",
                last_runtime_preflight=None,
                runtime_preflight_cache=service._new_runtime_preflight_cache(),
                session_scope="session:test",
            )

        # Augmentation prefix invariant: model prose preserved verbatim
        # at the start of the message.
        assert result.message.startswith(model_prose)
        # System-attributed suffix carrying the validator's objection.
        assert "[ELSPETH-SYSTEM]" in result.message
        assert "Forbidden name" in result.message
        assert "Suggested fix" in result.message
        # Old replacement-shape framing must not appear.
        assert "I cannot mark this pipeline complete" not in result.message
        # raw_assistant_content carries the unaugmented prose for the
        # audit trail and LLM history replay.
        assert result.raw_assistant_content == model_prose
        assert result.runtime_preflight is invalid_preflight

    @pytest.mark.asyncio
    async def test_empty_content_non_empty_state_invalid_preflight_augments_with_suffix_only(self) -> None:
        """Edge case: state is non-empty, runtime preflight is invalid,
        and the model emits empty assistant text on the finalize turn.

        Under the post-elspeth-9cfbad6901 augmentation policy, empty
        content + non-empty content + invalid preflight degenerates to
        suffix-only output (the augmentation prefix invariant is
        trivially satisfied because every string startswith ""). This
        replaces the earlier replacement-shape guard that crashed with
        AuditIntegrityError on this shape: the [INTERCEPTED] framing it
        was protecting no longer exists, so the ambiguity it guarded
        against is no longer load-bearing. The model's next
        preview_pipeline call drives self-correction.
        """
        catalog = _mock_catalog()
        settings = _make_settings()
        service = ComposerServiceImpl(catalog=catalog, settings=settings)
        state = (
            _empty_state()
            .with_source(
                SourceSpec(
                    plugin="csv",
                    on_success="t1",
                    options={"path": "/data/inputs/in.csv"},
                    on_validation_failure="discard",
                )
            )
            .with_output(
                OutputSpec(
                    name="main",
                    plugin="csv",
                    options={"path": "/data/outputs/out.csv", "schema": {"mode": "observed"}},
                    on_write_failure="discard",
                )
            )
        )
        bumped_state = replace(state, version=state.version + 1)
        invalid_preflight = ValidationResult(
            is_valid=False,
            checks=[],
            errors=[
                ValidationError(
                    component_id="t1",
                    component_type="transform",
                    message="Forbidden name: 'end_of_source'",
                    suggestion="Omit trigger for end-of-source-only aggregation.",
                    error_code=None,
                )
            ],
        )

        with patch.object(service, "_runtime_preflight", return_value=invalid_preflight):
            result = await service._finalize_no_tool_response(
                content="",
                state=bumped_state,
                initial_version=state.version,
                user_id="user-1",
                last_runtime_preflight=None,
                runtime_preflight_cache=service._new_runtime_preflight_cache(),
                session_scope="session:test",
            )

        # Suffix-only message — system-attributed, names the validator's
        # objection. The empty-content prefix invariant is trivially
        # satisfied (every string startswith "").
        assert result.message.startswith("[ELSPETH-SYSTEM]")
        assert "Forbidden name" in result.message
        # raw_assistant_content carries the (empty) original prose so
        # the audit row records what the model actually produced.
        assert result.raw_assistant_content == ""
        assert result.runtime_preflight is invalid_preflight

    @pytest.mark.asyncio
    async def test_empty_state_valid_preflight_preserves_message_verbatim(self) -> None:
        """Sanity: empty state but valid preflight is a degenerate but
        well-defined case (an empty pipeline that "passes" preflight is
        only possible if validation is bypassed). The model's content
        must NOT be touched in this case — neither synthesizer nor
        empty-state suffix."""
        catalog = _mock_catalog()
        settings = _make_settings()
        service = ComposerServiceImpl(catalog=catalog, settings=settings)
        state = _empty_state()
        valid_preflight = ValidationResult(is_valid=True, checks=[], errors=[])

        with patch.object(service, "_runtime_preflight"):
            result = await service._finalize_no_tool_response(
                content="All good.",
                state=state,
                initial_version=state.version,
                user_id="user-1",
                last_runtime_preflight=valid_preflight,
                runtime_preflight_cache=service._new_runtime_preflight_cache(),
                session_scope="session:test",
            )

        assert result.message == "All good."
        assert result.raw_assistant_content is None  # no replacement happened

    @pytest.mark.asyncio
    async def test_state_claim_grounding_appends_correction_on_t4_contradiction(self) -> None:
        """Path 3 of issue elspeth-c028f7d186: when the model's prose claims
        a state field has its old value while state has been mutated to a
        new value, the finalizer appends an [ELSPETH-SYSTEM] correction.

        Reproduces the panel-evals T4 case from the
        boolean_routing__p1_compliance cell (Linda the compliance officer):
        prose claims ``on_validation_failure: discard`` while state has
        ``rejected_records`` (the fix had already landed a turn earlier).

        The augmentation must satisfy
        ``_enforce_augmentation_prefix_invariant`` — message starts with
        the model's prose verbatim — and ``raw_assistant_content`` must
        carry the unaugmented prose for LLM history replay.
        """
        catalog = _mock_catalog()
        settings = _make_settings()
        service = ComposerServiceImpl(catalog=catalog, settings=settings)
        # State already has the fix applied: on_validation_failure == "rejected_records".
        state = (
            _empty_state()
            .with_source(
                SourceSpec(
                    plugin="csv",
                    on_success="rows",
                    options={"path": "/data/inputs/in.csv"},
                    on_validation_failure="rejected_records",
                )
            )
            .with_output(
                OutputSpec(
                    name="main",
                    plugin="csv",
                    options={"path": "/data/outputs/out.csv", "schema": {"mode": "observed"}},
                    on_write_failure="discard",
                )
            )
        )
        valid_preflight = ValidationResult(is_valid=True, checks=[], errors=[])
        # Prose contradicts state: claims discard, but state has rejected_records.
        contradicting_prose = "I see the issue — the source still uses `on_validation_failure: discard`, so I'll fix that next."

        with patch.object(service, "_runtime_preflight"):
            result = await service._finalize_no_tool_response(
                content=contradicting_prose,
                state=state,
                initial_version=state.version,
                user_id="user-1",
                last_runtime_preflight=valid_preflight,
                runtime_preflight_cache=service._new_runtime_preflight_cache(),
                session_scope="session:test",
            )

        # Augmentation prefix invariant: model's prose verbatim at start.
        assert result.message.startswith(contradicting_prose)
        # Correction suffix attached.
        assert "[ELSPETH-SYSTEM]" in result.message
        # Actual state value surfaced for the operator to read.
        assert "rejected_records" in result.message
        # raw_assistant_content carries the unaugmented prose so the
        # LLM history-replay path is unaffected by the synthetic suffix.
        assert result.raw_assistant_content == contradicting_prose
        # Preflight was passed through (still valid).
        assert result.runtime_preflight is valid_preflight

    @pytest.mark.asyncio
    async def test_state_claim_grounding_appends_correction_on_t5_unmotivated_action(self) -> None:
        """Path 3 backward-contradiction case from the panel-evals T5 cell:
        prose claims a fresh action ("I just fixed it") while state was
        unchanged this turn and no mutation tool succeeded.

        The action-claim detector flags the un-grounded completion claim
        even though the prose contains no concrete field=value claim.
        """
        catalog = _mock_catalog()
        settings = _make_settings()
        service = ComposerServiceImpl(catalog=catalog, settings=settings)
        # State is the same as the prior turn — no mutation happened this turn.
        state = (
            _empty_state()
            .with_source(
                SourceSpec(
                    plugin="csv",
                    on_success="rows",
                    options={"path": "/data/inputs/in.csv"},
                    on_validation_failure="rejected_records",
                )
            )
            .with_output(
                OutputSpec(
                    name="main",
                    plugin="csv",
                    options={"path": "/data/outputs/out.csv", "schema": {"mode": "observed"}},
                    on_write_failure="discard",
                )
            )
        )
        valid_preflight = ValidationResult(is_valid=True, checks=[], errors=[])
        unmotivated_prose = "I just fixed the workflow behavior. All set now."

        with patch.object(service, "_runtime_preflight"):
            result = await service._finalize_no_tool_response(
                content=unmotivated_prose,
                state=state,
                initial_version=state.version,  # unchanged → no mutation
                user_id="user-1",
                last_runtime_preflight=valid_preflight,
                runtime_preflight_cache=service._new_runtime_preflight_cache(),
                session_scope="session:test",
                mutation_success_seen=False,
            )

        assert result.message.startswith(unmotivated_prose)
        assert "[ELSPETH-SYSTEM]" in result.message
        # Explanation wording is pattern-agnostic across the four
        # action-claim pattern categories (issue elspeth-905fe2a3d8
        # widened the detector beyond "I just <verb>"). The shared
        # phrasing covers all four pattern shapes uniformly.
        assert "no mutation tool succeeded this turn" in result.message
        assert result.raw_assistant_content == unmotivated_prose

    @pytest.mark.asyncio
    async def test_state_claim_grounding_no_op_on_grounded_prose(self) -> None:
        """Negative control: when the model's prose is consistent with
        state, the grounding check is a no-op and the prose is passed
        through verbatim. Critical to avoid corrupting honest reports."""
        catalog = _mock_catalog()
        settings = _make_settings()
        service = ComposerServiceImpl(catalog=catalog, settings=settings)
        state = (
            _empty_state()
            .with_source(
                SourceSpec(
                    plugin="csv",
                    on_success="rows",
                    options={"path": "/data/inputs/in.csv"},
                    on_validation_failure="rejected_records",
                )
            )
            .with_output(
                OutputSpec(
                    name="main",
                    plugin="csv",
                    options={"path": "/data/outputs/out.csv", "schema": {"mode": "observed"}},
                    on_write_failure="discard",
                )
            )
        )
        valid_preflight = ValidationResult(is_valid=True, checks=[], errors=[])
        # Prose accurately reports state: it correctly says rejected_records.
        grounded_prose = "I configured the source with `on_validation_failure: rejected_records`. The pipeline is ready."

        with patch.object(service, "_runtime_preflight"):
            result = await service._finalize_no_tool_response(
                content=grounded_prose,
                state=state,
                initial_version=state.version,
                user_id="user-1",
                last_runtime_preflight=valid_preflight,
                runtime_preflight_cache=service._new_runtime_preflight_cache(),
                session_scope="session:test",
                mutation_success_seen=True,
            )

        # Verbatim pass-through; no augmentation; no raw_assistant_content set.
        assert result.message == grounded_prose
        assert "[ELSPETH-SYSTEM]" not in result.message
        assert result.raw_assistant_content is None

    @pytest.mark.asyncio
    async def test_state_claim_grounding_runs_when_runtime_result_is_none_agreement_language(self) -> None:
        """Bypass-fix coverage for issue elspeth-905fe2a3d8.

        Previously, when state did not change AND no preview_pipeline was
        called this turn, ``_finalize_no_tool_response`` early-returned with
        bare passthrough at the ``runtime_result is None`` branch — the
        state-claim grounding check at the bottom of the function was
        unreachable on that control-flow path. Cells #2/#4 of the
        panel-smoke-2026-05-10 cohort landed in that hole: the model
        agreed verbally ("you're right, I'll change that") without
        calling any mutation tool, and the augmentation never fired.

        After restructure: grounding runs even when ``runtime_result is None``
        (state non-empty, no preview), and the agreement-promise pattern
        category catches the prose."""
        catalog = _mock_catalog()
        settings = _make_settings()
        service = ComposerServiceImpl(catalog=catalog, settings=settings)
        state = (
            _empty_state()
            .with_source(
                SourceSpec(
                    plugin="csv",
                    on_success="rows",
                    options={"path": "/data/inputs/in.csv"},
                    on_validation_failure="discard",
                )
            )
            .with_output(
                OutputSpec(
                    name="main",
                    plugin="csv",
                    options={"path": "/data/outputs/out.csv", "schema": {"mode": "observed"}},
                    on_write_failure="discard",
                )
            )
        )
        agreement_prose = "You're right, I'll change that to rejected_records."

        with patch.object(service, "_runtime_preflight") as mock_preflight:
            result = await service._finalize_no_tool_response(
                content=agreement_prose,
                state=state,
                initial_version=state.version,  # unchanged → no mutation
                user_id="user-1",
                last_runtime_preflight=None,  # no preview was called
                runtime_preflight_cache=service._new_runtime_preflight_cache(),
                session_scope="session:test",
                mutation_success_seen=False,
            )

        # The grounding correction must fire — the prior bare passthrough
        # would have returned the prose verbatim with no [ELSPETH-SYSTEM]
        # block.
        assert result.message.startswith(agreement_prose)
        assert "[ELSPETH-SYSTEM]" in result.message
        assert "no mutation tool succeeded this turn" in result.message
        assert result.raw_assistant_content == agreement_prose
        # No preflight was rerun (state unchanged); the existing
        # branch-skip behaviour for preflight is preserved.
        mock_preflight.assert_not_called()

    @pytest.mark.asyncio
    async def test_state_claim_grounding_runs_when_runtime_result_is_none_t5_widened(self) -> None:
        """Bypass-fix coverage for issue elspeth-c028f7d186 (T5 widened).

        The T5 cell prose ("I fixed the workflow behavior so source
        validation is no longer silently dropping rows...") would land
        on the same previously-bypassed control-flow path: state didn't
        change after the T4 mutation, and the model didn't preview on
        T5 either. The bare-past-with-consequence pattern category
        added in this fix flags the prose; restructured grounding plumb
        ensures the check actually runs."""
        catalog = _mock_catalog()
        settings = _make_settings()
        service = ComposerServiceImpl(catalog=catalog, settings=settings)
        state = (
            _empty_state()
            .with_source(
                SourceSpec(
                    plugin="csv",
                    on_success="rows",
                    options={"path": "/data/inputs/in.csv"},
                    on_validation_failure="rejected_records",
                )
            )
            .with_output(
                OutputSpec(
                    name="main",
                    plugin="csv",
                    options={"path": "/data/outputs/out.csv", "schema": {"mode": "observed"}},
                    on_write_failure="discard",
                )
            )
        )
        t5_prose = "I fixed the workflow behavior so source validation is no longer silently dropping rows from the record set."

        with patch.object(service, "_runtime_preflight") as mock_preflight:
            result = await service._finalize_no_tool_response(
                content=t5_prose,
                state=state,
                initial_version=state.version,
                user_id="user-1",
                last_runtime_preflight=None,
                runtime_preflight_cache=service._new_runtime_preflight_cache(),
                session_scope="session:test",
                mutation_success_seen=False,
            )

        assert result.message.startswith(t5_prose)
        assert "[ELSPETH-SYSTEM]" in result.message
        assert result.raw_assistant_content == t5_prose
        mock_preflight.assert_not_called()

    @pytest.mark.asyncio
    async def test_no_grounding_correction_when_runtime_result_is_none_and_prose_is_innocuous(self) -> None:
        """Negative control: the bypass-fix restructure must NOT regress
        the bare-passthrough behaviour for prose that contains no
        action claim.

        Pins the contract that grounding is additive — it only augments
        when violations are detected, never in the no-violation case.
        Companion test to ``test_unchanged_text_without_preview_does_not_run_preflight``
        which exercises the same control-flow path with empty state."""
        catalog = _mock_catalog()
        settings = _make_settings()
        service = ComposerServiceImpl(catalog=catalog, settings=settings)
        state = _empty_state().with_source(
            SourceSpec(
                plugin="csv",
                on_success="rows",
                options={"path": "/data/inputs/in.csv"},
                on_validation_failure="rejected_records",
            )
        )
        # Bare past tense without a consequence clause — deliberately
        # below the conservative-anchoring threshold of every action
        # pattern. The detector must NOT flag this.
        innocuous_prose = "I fixed it."

        with patch.object(service, "_runtime_preflight") as mock_preflight:
            result = await service._finalize_no_tool_response(
                content=innocuous_prose,
                state=state,
                initial_version=state.version,
                user_id="user-1",
                last_runtime_preflight=None,
                runtime_preflight_cache=service._new_runtime_preflight_cache(),
                session_scope="session:test",
                mutation_success_seen=False,
            )

        assert result.message == innocuous_prose
        assert result.raw_assistant_content is None
        mock_preflight.assert_not_called()

    @pytest.mark.asyncio
    async def test_no_grounding_correction_when_mutation_succeeded_this_turn(self) -> None:
        """Verifier-gate protection on the bypass-fix path.

        Even when the agreement-promise pattern matches, the verifier
        gate (``mutation_success_seen or state_changed``) suppresses
        the violation if a mutation tool actually succeeded. Critical
        to avoid corrupting honest reports where the model agreed AND
        acted in the same turn."""
        catalog = _mock_catalog()
        settings = _make_settings()
        service = ComposerServiceImpl(catalog=catalog, settings=settings)
        state = _empty_state().with_source(
            SourceSpec(
                plugin="csv",
                on_success="rows",
                options={"path": "/data/inputs/in.csv"},
                on_validation_failure="rejected_records",
            )
        )
        agreement_prose = "You're right, I'll change that to rejected_records."

        with patch.object(service, "_runtime_preflight"):
            result = await service._finalize_no_tool_response(
                content=agreement_prose,
                state=state,
                initial_version=state.version,
                user_id="user-1",
                last_runtime_preflight=None,
                runtime_preflight_cache=service._new_runtime_preflight_cache(),
                session_scope="session:test",
                mutation_success_seen=True,  # the model DID act
            )

        # No augmentation: the verifier gate suppresses the action-claim
        # violation. Prose passes through verbatim.
        assert result.message == agreement_prose
        assert result.raw_assistant_content is None


# ---------------------------------------------------------------------------
# Forced-repair loop coverage. When the assistant emits no tool_calls but
# preview_pipeline's proof_diagnostics still reports blocking entries, the
# compose loop synthesises a repair message and continues. Hard cap of
# _MAX_REPAIR_TURNS=2 forced repair turns. The repair NEVER catches plugin
# exceptions — it only feeds the model proof diagnostics on configurations.
# ---------------------------------------------------------------------------


class TestAttemptProofRepair:
    """Direct exercise of ComposerServiceImpl._attempt_proof_repair."""

    @pytest.fixture(autouse=True)
    def _setup(self, tmp_path: Path):
        from datetime import UTC, datetime

        from elspeth.web.blobs.service import content_hash as _content_hash
        from elspeth.web.sessions.models import blobs_table

        catalog = _mock_catalog()
        settings = _make_settings(data_dir=tmp_path)
        engine, session_id = _session_engine_with_session()
        self.engine = engine
        self.session_id = session_id

        # Seed a CSV blob whose observed columns are {order_id, customer, price}
        self.blob_id = str(uuid4())
        storage_dir = tmp_path / "blobs" / session_id
        storage_dir.mkdir(parents=True)
        self.storage_path = storage_dir / f"{self.blob_id}_orders.csv"
        body = b"order_id,customer,price\nO-1,Alice,49.95\n"
        self.storage_path.write_bytes(body)
        with engine.begin() as conn:
            conn.execute(
                blobs_table.insert().values(
                    id=self.blob_id,
                    session_id=session_id,
                    filename="orders.csv",
                    mime_type="text/csv",
                    size_bytes=len(body),
                    content_hash=_content_hash(body),
                    storage_path=str(self.storage_path),
                    created_at=datetime.now(UTC),
                    created_by="user",
                    source_description=None,
                    status="ready",
                )
            )

        self.service = ComposerServiceImpl(
            catalog=catalog,
            settings=settings,
            sessions_service=_test_sessions_service(engine, tmp_path),
            session_engine=engine,
        )

    def _state_with_blocking_csv(self):
        """Build a state whose preview emits csv_fixed_schema_omits_observed_columns."""
        from elspeth.web.composer.tools import execute_tool as exec_tool

        state = _empty_state()
        catalog = _mock_catalog()
        # Wire the source via set_source_from_blob (canonical path).
        result = exec_tool(
            "set_source_from_blob",
            {
                "blob_id": self.blob_id,
                "on_success": "rows",
                "on_validation_failure": "discard",
                "options": {"schema": {"mode": "fixed", "fields": ["order_id: str"]}},
            },
            state,
            catalog,
            session_engine=self.engine,
            session_id=self.session_id,
        )
        assert result.success, result.data
        state = result.updated_state

        result = exec_tool(
            "set_output",
            {
                "sink_name": "out",
                "plugin": "json",
                "options": {
                    "path": "outputs/out.json",
                    "schema": {"mode": "observed"},
                    "mode": "write",
                    "collision_policy": "auto_increment",
                },
                "on_write_failure": "discard",
            },
            state,
            catalog,
        )
        assert result.success, result.data
        return result.updated_state

    def _state_without_blob(self):
        """A path-based source has nothing for proof_diagnostics to inspect."""
        from elspeth.web.composer.tools import execute_tool as exec_tool

        state = _empty_state()
        catalog = _mock_catalog()
        result = exec_tool(
            "set_pipeline",
            {
                "sources": {
                    "primary": {
                        "plugin": "csv",
                        "on_success": "rows",
                        "options": {"path": "/data/in.csv", "schema": {"mode": "observed"}},
                        "on_validation_failure": "discard",
                    }
                },
                "nodes": [],
                "edges": [],
                "outputs": [
                    {
                        "sink_name": "out",
                        "plugin": "csv",
                        "options": {"path": "/data/out.csv", "schema": {"mode": "observed"}},
                        "on_write_failure": "discard",
                    }
                ],
            },
            state,
            catalog,
        )
        assert result.success, result.data
        return result.updated_state

    def test_returns_false_when_no_blocking_diagnostics(self) -> None:
        state = self._state_without_blob()
        messages: list[dict[str, Any]] = []
        attempted = self.service._attempt_proof_repair(
            state=state,
            llm_messages=messages,
            session_id=self.session_id,
            repair_turns_used=0,
        )
        assert attempted is False
        assert messages == []

    def test_returns_false_when_budget_exhausted(self) -> None:
        from elspeth.web.composer.service import _MAX_REPAIR_TURNS

        state = self._state_with_blocking_csv()
        messages: list[dict[str, Any]] = []
        attempted = self.service._attempt_proof_repair(
            state=state,
            llm_messages=messages,
            session_id=self.session_id,
            repair_turns_used=_MAX_REPAIR_TURNS,
        )
        assert attempted is False
        assert messages == []

    def test_appends_repair_message_when_blocking(self) -> None:
        state = self._state_with_blocking_csv()
        messages: list[dict[str, Any]] = []
        attempted = self.service._attempt_proof_repair(
            state=state,
            llm_messages=messages,
            session_id=self.session_id,
            repair_turns_used=0,
        )
        assert attempted is True
        assert len(messages) == 1
        msg = messages[0]
        assert msg["role"] == "user"
        assert "csv_fixed_schema_omits_observed_columns" in msg["content"]
        assert "Suggested repair" in msg["content"]
        assert "preview_pipeline" in msg["content"]
        # Budget note acknowledges the cap
        assert "forced repair turn 1 of 2" in msg["content"]

    def test_second_repair_message_increments_turn_counter_in_text(self) -> None:
        state = self._state_with_blocking_csv()
        messages: list[dict[str, Any]] = []
        # Simulate one already-used repair turn
        attempted = self.service._attempt_proof_repair(
            state=state,
            llm_messages=messages,
            session_id=self.session_id,
            repair_turns_used=1,
        )
        assert attempted is True
        assert "forced repair turn 2 of 2" in messages[0]["content"]

    def test_repair_does_not_catch_plugin_exceptions(self) -> None:
        """Plugin exceptions must propagate — the repair gate only handles configs.

        Patch compute_proof_diagnostics to raise a synthetic 'plugin bug';
        _attempt_proof_repair must not swallow it.
        """
        from elspeth.web.composer import service as svc_module

        with (
            patch.object(svc_module, "compute_proof_diagnostics", side_effect=RuntimeError("simulated plugin crash")),
            pytest.raises(RuntimeError, match="simulated plugin crash"),
        ):
            self.service._attempt_proof_repair(
                state=self._state_with_blocking_csv(),
                llm_messages=[],
                session_id=self.session_id,
                repair_turns_used=0,
            )

    def test_repair_message_caps_at_three_blockers(self) -> None:
        """When 4+ blocking diagnostics fire, the repair message renders only
        the first three to keep the LLM's context window manageable. The
        cap is documented in the synthesizer; this test pins it so future
        edits cannot relax the bound silently.

        The test patches ``compute_proof_diagnostics`` to return a
        deterministic 5-item list of blocking entries — the synthesiser is
        the unit under test, and the diagnostic source is irrelevant.
        """
        from elspeth.web.composer import service as svc_module

        fake_blockers = [
            {
                "severity": "blocking",
                "code": f"fake_blocker_{i}",
                "message": f"fake message {i}",
                "suggested_repair": f"fake repair {i}",
                "evidence_locator": {"path": f"$.blocker[{i}]"},
            }
            for i in range(5)
        ]
        messages: list[dict[str, Any]] = []
        with patch.object(svc_module, "compute_proof_diagnostics", return_value=fake_blockers):
            attempted = self.service._attempt_proof_repair(
                state=self._state_without_blob(),
                llm_messages=messages,
                session_id=self.session_id,
                repair_turns_used=0,
            )
        assert attempted is True
        assert len(messages) == 1
        body = messages[0]["content"]
        # First three blockers rendered.
        assert "fake_blocker_0" in body
        assert "fake_blocker_1" in body
        assert "fake_blocker_2" in body
        # Fourth and fifth blockers omitted from the synthesised message.
        assert "fake_blocker_3" not in body
        assert "fake_blocker_4" not in body


# ---------------------------------------------------------------------------
# End-to-end forced-repair loop coverage. Drives the real ``_compose_loop``
# ``while True`` through real repair turns with a scripted LLM at the
# ``_call_llm`` boundary. Bugs that would land silently otherwise:
# stale ``repair_turns_used``, missing ``continue`` after the repair message,
# ``replace(result, ...)`` clobbering, divergence behaviour when repair
# makes things worse, hard cap not respected.
# ---------------------------------------------------------------------------


class TestComposeLoopForcedRepair:
    """Exercise ``ComposerServiceImpl.compose`` through scripted repair turns.

    The scripted LLM is wired at ``_call_llm`` (network seam); ``execute_tool``
    runs for real, so ``set_source_from_blob`` / ``patch_source_options`` /
    ``set_output`` produce authentic state mutations and ``compute_proof_diagnostics``
    inspects authentic blob bytes. ``_runtime_preflight`` is patched to a
    permissive result so finalization is bounded by the proof gate alone.
    """

    @pytest.fixture(autouse=True)
    def _setup(self, tmp_path: Path):
        from elspeth.web.blobs.service import content_hash as _content_hash
        from elspeth.web.sessions.models import blobs_table

        catalog = _mock_catalog()
        settings = _make_settings(data_dir=tmp_path)
        engine, session_id = _session_engine_with_session()
        self.engine = engine
        self.session_id = session_id

        # CSV with three observed columns. Pipeline configured with
        # mode='fixed' + fields=['order_id: str'] + on_validation_failure='discard'
        # triggers the csv_fixed_schema_omits_observed_columns blocking
        # diagnostic in compute_proof_diagnostics.
        self.blob_id = str(uuid4())
        storage_dir = tmp_path / "blobs" / session_id
        storage_dir.mkdir(parents=True)
        self.storage_path = storage_dir / f"{self.blob_id}_orders.csv"
        body = b"order_id,customer,price\nO-1,Alice,49.95\n"
        self.storage_path.write_bytes(body)
        with engine.begin() as conn:
            conn.execute(
                blobs_table.insert().values(
                    id=self.blob_id,
                    session_id=session_id,
                    filename="orders.csv",
                    mime_type="text/csv",
                    size_bytes=len(body),
                    content_hash=_content_hash(body),
                    storage_path=str(self.storage_path),
                    created_at=datetime.now(UTC),
                    created_by="user",
                    source_description=None,
                    status="ready",
                )
            )

        self.service = ComposerServiceImpl(
            catalog=catalog,
            settings=settings,
            sessions_service=_test_sessions_service(engine, tmp_path),
            session_engine=engine,
        )

    def _wire_blocking_pipeline_tool_calls(self) -> list[dict[str, Any]]:
        """Tool calls that establish the blocking csv_fixed_schema_omits_observed_columns state."""
        return [
            {
                "id": "call_source",
                "name": "set_source_from_blob",
                "arguments": {
                    "blob_id": self.blob_id,
                    "on_success": "rows",
                    "on_validation_failure": "discard",
                    "options": {"schema": {"mode": "fixed", "fields": ["order_id: str"]}},
                },
            },
            {
                "id": "call_output",
                "name": "set_output",
                "arguments": {
                    "sink_name": "out",
                    "plugin": "json",
                    "options": {
                        "path": "outputs/out.json",
                        "schema": {"mode": "observed"},
                        "mode": "write",
                        "collision_policy": "auto_increment",
                    },
                    "on_write_failure": "discard",
                },
            },
        ]

    def _repair_tool_call(self) -> list[dict[str, Any]]:
        """Tool call that fixes the blocking diagnostic.

        Switch from ``mode='fixed'`` (which omitted observed columns) to
        ``mode='observed'`` (auto-infer types from data). ``mode='flexible'``
        without explicit ``fields`` is rejected by the csv source's
        config-model prevalidation, so observed is the cleanest repair.
        """
        return [
            {
                "id": "call_repair",
                "name": "patch_source_options",
                "arguments": {"patch": {"schema": {"mode": "observed"}}},
            },
        ]

    def _wire_uploaded_blob_to_output_tool_calls(self) -> list[dict[str, Any]]:
        """Tool calls for the uploaded-CSV happy path after an empty-state stall."""
        return [
            {
                "id": "call_source",
                "name": "set_source_from_blob",
                "arguments": {
                    "blob_id": self.blob_id,
                    "on_success": "out",
                    "on_validation_failure": "discard",
                    "options": {"schema": {"mode": "observed"}},
                },
            },
            {
                "id": "call_output",
                "name": "set_output",
                "arguments": {
                    "sink_name": "out",
                    "plugin": "json",
                    "options": {
                        "path": "outputs/out.json",
                        "schema": {"mode": "observed"},
                        "mode": "write",
                        "collision_policy": "auto_increment",
                    },
                    "on_write_failure": "discard",
                },
            },
        ]

    @pytest.mark.asyncio
    async def test_empty_state_uploaded_blob_stall_forces_repair_turn(self) -> None:
        """A no-tool prose reply must not end the turn when ready uploaded blobs exist.

        Regression for elspeth-b493ddf810: the hard-mode uploaded-CSV happy
        path produced repeated prose replies, no tool calls, and an empty
        CompositionState even though a ready uploaded CSV blob was present.
        The service should feed the model a concrete repair instruction naming
        the ready blob and continue the loop, instead of finalizing the
        empty-state response.
        """
        passing_preflight = ValidationResult(is_valid=True, checks=[], errors=[])
        turn1_stall = _make_llm_response(
            content="The uploaded CSV appears to contain only the header row, so I cannot build yet.",
            tool_calls=None,
        )
        turn2_build = _make_llm_response(content=None, tool_calls=self._wire_uploaded_blob_to_output_tool_calls())
        turn3_done = _make_llm_response(content="Ready.", tool_calls=None)

        empty = _empty_state()
        with (
            patch.object(self.service, "_call_llm", new_callable=AsyncMock) as mock_llm,
            patch.object(self.service, "_runtime_preflight", return_value=passing_preflight),
        ):
            mock_llm.side_effect = [turn1_stall, turn2_build, turn3_done]
            result = await self.service.compose(
                "Build from my uploaded CSV",
                [],
                empty,
                session_id=self.session_id,
                user_id="test-user",
            )

        assert mock_llm.call_count == 3
        assert result.repair_turns_used == 1
        assert result.state.sources["source"] is not None
        assert result.state.sources["source"].options["blob_ref"] == self.blob_id
        assert result.state.sources["source"].options["schema"] == {"mode": "observed"}
        assert result.state.outputs[0].name == "out"
        assert result.runtime_preflight is not None and result.runtime_preflight.is_valid is True

        turn2_messages = mock_llm.call_args_list[1].args[0]
        repair_msgs = [
            m
            for m in turn2_messages
            if isinstance(m, dict)
            and m.get("role") == "user"
            and "[composer-system]" in str(m.get("content", ""))
            and self.blob_id in str(m.get("content", ""))
        ]
        assert len(repair_msgs) == 1
        repair_text = str(repair_msgs[0]["content"])
        assert "ready uploaded blob" in repair_text
        assert "inspect_source" in repair_text
        assert "source.blob_id" in repair_text
        assert "Do not infer that a CSV is header-only" in repair_text

    def _futile_repair_tool_call(self, call_id: str, name_value: str) -> list[dict[str, Any]]:
        """Tool call that mutates state but does NOT clear the proof blocker.

        Calls ``set_metadata`` (state.version bumps) without touching the
        source's schema. The original ``csv_fixed_schema_omits_observed_columns``
        diagnostic survives. Used to verify the repair-budget cap holds
        when the model mutates without applying a useful repair.
        """
        return [
            {
                "id": call_id,
                "name": "set_metadata",
                "arguments": {"patch": {"name": name_value}},
            },
        ]

    @pytest.mark.asyncio
    async def test_happy_repair_path(self) -> None:
        """Turn 1 establishes blocker, turn 2 claims completion, turn 3 repairs.

        Loop sequence:
          - Turn 1 (LLM): tool_calls = [set_source_from_blob (blocking schema), set_output]
          - Turn 2 (LLM): no tool_calls, claims completion
            → _attempt_proof_repair fires, sees blocking diagnostic, injects
              repair message, loop continues
          - Turn 3 (LLM): tool_calls = [patch_source_options(mode=flexible)]
          - Turn 4 (LLM): no tool_calls, claims completion
            → no blocking diagnostic now, finalize cleanly

        Assertions:
          - result.repair_turns_used == 1
          - synthesised repair message reaches the LLM history
          - is_valid True (no blocking diagnostic remaining)
        """
        passing_preflight = ValidationResult(is_valid=True, checks=[], errors=[])
        turn1 = _make_llm_response(content=None, tool_calls=self._wire_blocking_pipeline_tool_calls())
        turn2_done = _make_llm_response(content="All set.", tool_calls=None)
        turn3 = _make_llm_response(content=None, tool_calls=self._repair_tool_call())
        turn4_done = _make_llm_response(content="Repaired and ready.", tool_calls=None)

        empty = _empty_state()
        with (
            patch.object(self.service, "_call_llm", new_callable=AsyncMock) as mock_llm,
            patch.object(self.service, "_runtime_preflight", return_value=passing_preflight),
        ):
            mock_llm.side_effect = [turn1, turn2_done, turn3, turn4_done]
            result = await self.service.compose(
                "Build a pipeline",
                [],
                empty,
                session_id=self.session_id,
                user_id="test-user",
            )

        # Loop ran exactly four LLM turns: build, claim-complete, repair,
        # claim-complete-again.
        assert mock_llm.call_count == 4, f"expected 4 LLM calls, got {mock_llm.call_count}"
        # repair_turns_used == 1 — the model needed exactly one forced
        # repair turn to clear the diagnostic.
        assert result.repair_turns_used == 1
        # Final state has observed schema (the repair landed) and is_valid.
        assert result.state.sources["source"] is not None
        assert result.state.sources["source"].options["schema"] == {"mode": "observed"}
        assert result.runtime_preflight is not None and result.runtime_preflight.is_valid is True
        # The synthesised repair message reaches the LLM history before
        # turn 3 — inspect the messages of turn 3 (index 2).
        turn3_messages = mock_llm.call_args_list[2].args[0]
        repair_msgs = [
            m
            for m in turn3_messages
            if isinstance(m, dict)
            and m.get("role") == "user"
            and "[composer-system]" in str(m.get("content", ""))
            and "csv_fixed_schema_omits_observed_columns" in str(m.get("content", ""))
        ]
        assert len(repair_msgs) == 1, f"exactly one composer-system repair message must precede turn 3; got {len(repair_msgs)}"
        # Final assistant content reflects turn 4's claim, not turn 2's
        # (which was preempted by the repair gate).
        assert result.message == "Repaired and ready."

    @pytest.mark.asyncio
    async def test_divergence_repair_does_not_clear_blocker(self) -> None:
        """Repair turns mutate state but never address the proof blocker.

        The model claims completion, sees the repair message, then on
        each repair turn issues a tool call that mutates *something else*
        (here: pipeline metadata.name) without addressing the source
        schema. ``state.version`` bumps each turn, the proof gate fires
        each ``no-tool-call`` turn while ``state.version > initial_version``,
        and the loop terminates when ``repair_turns_used`` reaches
        ``_MAX_REPAIR_TURNS``.

        Loop sequence (6 LLM turns):
          - Turn 1: tool_calls = [set_source_from_blob (blocker), set_output]
          - Turn 2: claim complete → repair fires, repair_turns_used=1
          - Turn 3: tool_calls = [set_metadata(name='attempt-1')] (no-op
            for the blocker)
          - Turn 4: claim complete → repair fires, repair_turns_used=2
          - Turn 5: tool_calls = [set_metadata(name='attempt-2')]
          - Turn 6: claim complete → repair_turns_used==_MAX_REPAIR_TURNS,
            repair gate returns False, finalize with blocker still in
            ``proof_diagnostics``.
        """
        from elspeth.web.composer.service import _MAX_REPAIR_TURNS

        # Sanity check on the constant — keeps the test honest if the
        # cap shifts.
        assert _MAX_REPAIR_TURNS == 2

        passing_preflight = ValidationResult(is_valid=True, checks=[], errors=[])
        turns = [
            _make_llm_response(content=None, tool_calls=self._wire_blocking_pipeline_tool_calls()),
            _make_llm_response(content="claim 1", tool_calls=None),
            _make_llm_response(content=None, tool_calls=self._futile_repair_tool_call("call_a", "attempt-1")),
            _make_llm_response(content="claim 2", tool_calls=None),
            _make_llm_response(content=None, tool_calls=self._futile_repair_tool_call("call_b", "attempt-2")),
            _make_llm_response(content="final", tool_calls=None),
        ]

        empty = _empty_state()
        with (
            patch.object(self.service, "_call_llm", new_callable=AsyncMock) as mock_llm,
            patch.object(self.service, "_runtime_preflight", return_value=passing_preflight),
        ):
            mock_llm.side_effect = turns
            result = await self.service.compose(
                "Build something",
                [],
                empty,
                session_id=self.session_id,
                user_id="test-user",
            )

        # Six LLM turns total; the third claim-complete is final because
        # the repair budget is exhausted.
        assert mock_llm.call_count == 6, f"expected 6 LLM calls, got {mock_llm.call_count}"
        assert result.repair_turns_used == _MAX_REPAIR_TURNS
        # Source still has the original fixed-mode blocking schema; the
        # futile mutations never touched the source.
        assert result.state.sources["source"] is not None
        assert result.state.sources["source"].options["schema"]["mode"] == "fixed"
        # Metadata reflects the most recent (futile) repair attempt.
        assert result.state.metadata.name == "attempt-2"
        # Final message is the third claim-complete; finalisation
        # proceeded once the repair budget hit the cap.
        assert result.message == "final"

    @pytest.mark.asyncio
    async def test_cap_respected_when_model_never_repairs(self) -> None:
        """Model claims completion but never tool-calls repairs — the cap stops the loop.

        The blocker remains every time the proof step fires. The repair
        budget is exhausted; finalization proceeds with the blocker
        still visible. ``is_valid`` must reflect the blocking proof
        diagnostic (forced False by the proof gate even when authoring/
        runtime preflight pass).
        """
        from elspeth.web.composer.service import _MAX_REPAIR_TURNS

        passing_preflight = ValidationResult(is_valid=True, checks=[], errors=[])
        # Turn 1 establishes the blocker. Turns 2..N all claim completion
        # without applying a fix. The repair gate fires after each
        # state-mutating turn — but turns 2..N have no tool calls, so
        # state.version doesn't bump on those turns. The proof gate fires
        # only when ``state.version > initial_version``, which holds
        # from turn 2 onwards (turn 1 mutated). Two repair-fires later
        # (turns 2 and 3 since the cap is 2), the third claim-complete
        # finalises.
        turns = [
            _make_llm_response(content=None, tool_calls=self._wire_blocking_pipeline_tool_calls()),
            _make_llm_response(content="claim 1", tool_calls=None),
            _make_llm_response(content="claim 2", tool_calls=None),
            _make_llm_response(content="final", tool_calls=None),
        ]

        empty = _empty_state()
        with (
            patch.object(self.service, "_call_llm", new_callable=AsyncMock) as mock_llm,
            patch.object(self.service, "_runtime_preflight", return_value=passing_preflight),
        ):
            mock_llm.side_effect = turns
            result = await self.service.compose(
                "Build something",
                [],
                empty,
                session_id=self.session_id,
                user_id="test-user",
            )

        # Four LLM turns: 1 mutating + 3 claim-complete (the first two
        # claim-completes fire repair injection, the third is final).
        assert mock_llm.call_count == 4
        assert result.repair_turns_used == _MAX_REPAIR_TURNS
        # The model never applied a fix → schema still 'fixed' with the
        # original blocker config; runtime_preflight may still be valid
        # (it was patched to return is_valid=True), but the runtime gate
        # is not the proof gate. Verify the source state is unchanged.
        assert result.state.sources["source"] is not None
        assert result.state.sources["source"].options["schema"]["mode"] == "fixed"

    @pytest.mark.asyncio
    async def test_repair_gate_fires_on_first_turn_of_resumed_session(self) -> None:
        """Resumed session with a pre-bound blob-backed source + blocking
        diagnostic must trigger the repair gate on its first compose turn,
        even though the LLM has not mutated state yet.

        Regression for the ``state.version > initial_version`` guard issue:
        on a resumed session, the source was bound on a prior turn, so the
        first turn after resume has ``state.version == initial_version``.
        With the version guard, the gate skipped — exactly the cross-turn
        scenario it exists to catch (e.g.
        ``csv_fixed_schema_omits_observed_columns`` blockers persisting
        through session resume). The corrected predicate fires whenever
        the proof step is applicable (source is blob-backed and bound).
        """
        passing_preflight = ValidationResult(is_valid=True, checks=[], errors=[])

        # Resumed-session state: source is already bound to the blob with
        # the blocking schema config. No state mutation occurred this turn —
        # the LLM's first response on resume claims completion.
        # ``path`` mirrors the blob's canonical storage_path because the csv
        # source plugin requires it (set_source_from_blob populates it on
        # the binding turn; we reproduce that here for the resumed state).
        # ``on_success="out"`` matches the named output below so the graph
        # validates (preventing patch_source_options from rejecting the
        # repair due to a dangling connection unrelated to this regression).
        resumed_state = CompositionState(
            source=SourceSpec(
                plugin="csv",
                on_success="out",
                options={
                    "blob_ref": self.blob_id,
                    "path": str(self.storage_path),
                    "schema": {"mode": "fixed", "fields": ["order_id: str"]},
                },
                on_validation_failure="discard",
            ),
            nodes=(),
            edges=(),
            outputs=(
                OutputSpec(
                    name="out",
                    plugin="json",
                    options={
                        "path": "outputs/out.json",
                        "schema": {"mode": "observed"},
                        "mode": "write",
                        "collision_policy": "auto_increment",
                    },
                    on_write_failure="discard",
                ),
            ),
            metadata=PipelineMetadata(),
            version=7,  # Resumed mid-session — version is already non-trivial.
        )

        # Turn 1: claim completion immediately (no tool_calls). The repair
        # gate must fire here even though state.version was not advanced.
        # Turn 2: apply the repair. Turn 3: claim completion again, gate
        # finds no blocker, finalize cleanly.
        turn1_done = _make_llm_response(content="Already configured.", tool_calls=None)
        turn2_repair = _make_llm_response(content=None, tool_calls=self._repair_tool_call())
        turn3_done = _make_llm_response(content="Repaired and ready.", tool_calls=None)

        with (
            patch.object(self.service, "_call_llm", new_callable=AsyncMock) as mock_llm,
            patch.object(self.service, "_runtime_preflight", return_value=passing_preflight),
        ):
            mock_llm.side_effect = [turn1_done, turn2_repair, turn3_done]
            result = await self.service.compose(
                "Continue building",
                [],
                resumed_state,
                session_id=self.session_id,
                user_id="test-user",
            )

        # The gate fired on turn 1 (no-mutation, version unchanged), forced
        # the model into a repair turn, and the loop converged.
        assert mock_llm.call_count == 3, f"expected 3 LLM calls, got {mock_llm.call_count}"
        assert result.repair_turns_used == 1
        # Turn 2 received the synthesised repair message before the model
        # acted on it — verify the diagnostic landed in the LLM history.
        turn2_messages = mock_llm.call_args_list[1].args[0]
        repair_msgs = [
            m
            for m in turn2_messages
            if isinstance(m, dict)
            and m.get("role") == "user"
            and "[composer-system]" in str(m.get("content", ""))
            and "csv_fixed_schema_omits_observed_columns" in str(m.get("content", ""))
        ]
        assert len(repair_msgs) == 1, f"expected one synthesised repair message in turn-2 history, found: {turn2_messages}"
        # Final state is repaired — the schema mode is now observed.
        assert result.state.sources["source"] is not None
        assert result.state.sources["source"].options["schema"] == {"mode": "observed"}

    @pytest.mark.asyncio
    async def test_repair_gate_skipped_when_source_is_not_blob_backed(self) -> None:
        """When the proof step is not applicable (no blob-backed source),
        the gate skips even if the LLM claims completion.

        This pins the converse of the resumed-session test: chat-only
        turns and path-based-source pipelines must not pay the proof
        step's cost when there is nothing to inspect.
        """
        passing_preflight = ValidationResult(is_valid=True, checks=[], errors=[])

        # Path-based source — no blob_ref, so compute_proof_diagnostics
        # would short-circuit and the gate has nothing to do.
        path_source_state = CompositionState(
            source=SourceSpec(
                plugin="csv",
                on_success="rows",
                options={"path": "/tmp/never-read.csv", "schema": {"mode": "observed"}},
                on_validation_failure="discard",
            ),
            nodes=(),
            edges=(),
            outputs=(
                OutputSpec(
                    name="out",
                    plugin="json",
                    options={
                        "path": "outputs/out.json",
                        "schema": {"mode": "observed"},
                        "mode": "write",
                        "collision_policy": "auto_increment",
                    },
                    on_write_failure="discard",
                ),
            ),
            metadata=PipelineMetadata(),
            version=4,
        )

        turn1_done = _make_llm_response(content="All set.", tool_calls=None)

        with (
            patch.object(self.service, "_call_llm", new_callable=AsyncMock) as mock_llm,
            patch.object(self.service, "_runtime_preflight", return_value=passing_preflight),
        ):
            mock_llm.side_effect = [turn1_done]
            result = await self.service.compose(
                "Anything else?",
                [],
                path_source_state,
                session_id=self.session_id,
                user_id="test-user",
            )

        # Gate skipped — only one LLM call, no repair turns.
        assert mock_llm.call_count == 1
        assert result.repair_turns_used == 0


class TestComposeLoopFreeformRecipeIntentRouting:
    @pytest.mark.asyncio
    async def test_fork_coalesce_truncate_intent_applies_recipe_before_llm(self, tmp_path: Path) -> None:
        engine, session_id = _session_engine_with_session()
        service = ComposerServiceImpl(
            catalog=_mock_catalog(),
            settings=_make_settings(data_dir=tmp_path),
            sessions_service=_test_sessions_service(engine, tmp_path),
            session_engine=engine,
        )
        prompt = (
            "Please create a pipeline that processes the following customer rows. "
            "Each row should be processed two ways in parallel and combined into "
            "a single merged output row at outputs/merged.jsonl: path A keeps the "
            "original row unchanged, path B truncates the description field to 30 "
            "characters with suffix '...'. Combine both branches under separate "
            "keys `path_a` and `path_b` in each merged output row -- one input row "
            "produces one output row containing both branches side-by-side. "
            "Customer rows (CSV):\n"
            "name,description\n"
            "alice,this is a moderately long description for testing the truncation behaviour\n"
            "bob,short note\n"
            "charlie,another lengthy customer description that exceeds thirty characters comfortably"
        )
        user_message_id = _insert_user_message(engine, session_id, prompt)

        with patch.object(
            service,
            "_call_llm_before_deadline",
            new_callable=AsyncMock,
            side_effect=AssertionError("intent routing should bypass the cheap model"),
        ) as mock_llm:
            result = await service.compose(
                prompt,
                [],
                _empty_state(),
                session_id=session_id,
                user_id="test-user",
                user_message_id=user_message_id,
            )

        assert mock_llm.call_count == 0
        assert result.state.validate().is_valid is True
        assert result.state.sources["source"] is not None
        assert result.state.sources["source"].plugin == "csv"
        assert result.state.sources["source"].options["blob_ref"]
        assert {node.node_type for node in result.state.nodes} >= {"gate", "coalesce"}
        assert any(node.plugin == "truncate" for node in result.state.nodes)
        assert result.state.outputs[0].name == "merged_rows"
        assert result.state.outputs[0].options["path"] == "outputs/merged.jsonl"
        assert "fork-coalesce-truncate-jsonl" in result.message
