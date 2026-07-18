"""Atomic guided RESPOND/CHAT settlement and replay contracts."""

from __future__ import annotations

import importlib
from collections.abc import Iterator, Mapping
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import uuid4

import pytest
import structlog
from sqlalchemy import select, update

from elspeth.contracts.composer_audit import ComposerToolInvocation, ComposerToolStatus
from elspeth.contracts.composer_interpretation import InterpretationKind
from elspeth.contracts.composer_llm_audit import (
    ComposerChatInitiator,
    ComposerChatTurn,
    ComposerChatTurnStatus,
    ComposerLLMCall,
    ComposerLLMCallStatus,
)
from elspeth.contracts.errors import AuditIntegrityError
from elspeth.contracts.freeze import deep_thaw
from elspeth.contracts.hashing import stable_hash
from elspeth.core.payload_store import FilesystemPayloadStore
from elspeth.web.composer.guided.protocol import ChatRole, ChatTurn, GuidedStep, TurnType
from elspeth.web.composer.guided.state_machine import GuidedSession, TurnRecord
from elspeth.web.composer.pipeline_proposal import composition_content_hash
from elspeth.web.plugin_policy.models import PluginId, PluginUnavailableReason
from elspeth.web.sessions.converters import state_from_record
from elspeth.web.sessions.engine import create_session_engine
from elspeth.web.sessions.models import (
    chat_messages_table,
    composition_states_table,
    guided_operations_table,
    interpretation_events_table,
    sessions_table,
)
from elspeth.web.sessions.protocol import (
    CompositionStateData,
    CompositionStateRecord,
    GuidedAuditEvidence,
    GuidedOperationClaimed,
    GuidedOperationFence,
    GuidedOperationFenceLostError,
    GuidedOperationTakenOver,
    GuidedOriginatingUserMessageDraft,
    GuidedReplayPolicyFinding,
    GuidedReplayTurn,
    GuidedResponseDescriptor,
    GuidedStateOperationCommand,
    PreparedGuidedInterpretationDraft,
    PreparedGuidedJsonPayload,
    StaleComposeStateError,
)
from elspeth.web.sessions.schema import initialize_session_schema
from elspeth.web.sessions.service import SessionServiceImpl
from elspeth.web.sessions.telemetry import build_sessions_telemetry


def _audit_evidence() -> tuple[ComposerToolInvocation, ComposerLLMCall, ComposerChatTurn]:
    now = datetime.now(UTC)
    invocation = ComposerToolInvocation(
        tool_call_id="guided-audit-1",
        tool_name="guided_step_advanced",
        arguments_canonical='{"next_step":"step_2_sink","prev_step":"step_1_source","reason":"user_advanced"}',
        arguments_hash=stable_hash(
            {
                "next_step": "step_2_sink",
                "prev_step": "step_1_source",
                "reason": "user_advanced",
            }
        ),
        result_canonical=None,
        result_hash=None,
        status=ComposerToolStatus.SUCCESS,
        error_class=None,
        error_message=None,
        version_before=1,
        version_after=1,
        started_at=now,
        finished_at=now,
        latency_ms=0,
        actor="worker",
    )
    llm_call = ComposerLLMCall(
        model_requested="test-model",
        model_returned="test-model",
        status=ComposerLLMCallStatus.SUCCESS,
        prompt_tokens=1,
        completion_tokens=1,
        total_tokens=2,
        latency_ms=1,
        provider_request_id="request-1",
        messages_hash="b" * 64,
        tools_spec_hash=None,
        declared_tool_names=(),
        started_at=now,
        finished_at=now,
        error_class=None,
        error_message=None,
        temperature=None,
        seed=None,
        reasoning_content="SECRET-REASONING-CANARY",
        reasoning_details={"provider_diagnostic": "SECRET-PROVIDER-CANARY"},
    )
    chat_turn = ComposerChatTurn(
        step="step_1_source",
        initiator=ComposerChatInitiator.USER,
        chat_turn_seq=0,
        user_message_hash="c" * 64,
        assistant_message_hash="d" * 64,
        latency_ms=1,
        model="test-model",
        status=ComposerChatTurnStatus.SUCCESS,
        started_at=now,
        finished_at=now,
    )
    return invocation, llm_call, chat_turn


def _turn_emitted_evidence(payload_id: str) -> ComposerToolInvocation:
    invocation, _llm_call, _chat_turn = _audit_evidence()
    arguments = {
        "step_index": "step_1_source",
        "turn_type": "single_select",
        "payload_hash": payload_id,
        "payload_payload_id": payload_id,
        "emitter": "server",
    }
    return replace(
        invocation,
        tool_call_id="turn-emitted-1",
        tool_name="guided_turn_emitted",
        arguments_canonical=importlib.import_module("elspeth.contracts.hashing").canonical_json(arguments),
        arguments_hash=stable_hash(arguments),
    )


def test_protocol_exposes_closed_guided_state_settlement_contract() -> None:
    protocol = importlib.import_module("elspeth.web.sessions.protocol")

    assert hasattr(protocol, "PreparedGuidedJsonPayload")
    assert hasattr(protocol, "PreparedGuidedAuditRow")
    assert hasattr(protocol, "GuidedResponseDescriptor")
    assert hasattr(protocol, "GuidedStateOperationCommand")
    assert hasattr(protocol, "GuidedStateOperationSettlement")


def test_replay_turn_accepts_zero_based_step_one_index() -> None:
    turn = GuidedReplayTurn(
        turn_type=TurnType.SINGLE_SELECT,
        step_index=0,
        payload_id="a" * 64,
    )

    assert turn.step_index == 0


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("component_id", "source:/tmp/operator-secret"),
        ("plugin_id", "source:not-valid-hyphen"),
        ("reason_code", "raw provider denied because token=SECRET"),
        ("snapshot_fingerprint", "not-a-sha256"),
    ],
)
def test_replay_policy_finding_rejects_raw_text_and_malformed_identity(field: str, value: str) -> None:
    values = {
        "component_id": "source_1",
        "plugin_id": PluginId.parse("source:csv"),
        "reason_code": PluginUnavailableReason.NOT_INSTALLED,
        "snapshot_fingerprint": "a" * 64,
    }
    values[field] = value

    with pytest.raises((AuditIntegrityError, ValueError)):
        GuidedReplayPolicyFinding(**values)


def test_prepared_payload_hashes_the_detached_frozen_snapshot() -> None:
    class _TwoViewMapping(Mapping[str, str]):
        def __init__(self) -> None:
            self.views = 0

        def __getitem__(self, key: str) -> str:
            if key != "value":
                raise KeyError(key)
            return "first" if self.views == 0 else "second"

        def __iter__(self) -> Iterator[str]:
            return iter(("value",))

        def __len__(self) -> int:
            return 1

        def items(self):
            value = "first" if self.views == 0 else "second"
            self.views += 1
            return (("value", value),)

    payload = PreparedGuidedJsonPayload(
        payload_id=stable_hash({"value": "first"}),
        purpose="turn",
        payload=_TwoViewMapping(),
    )

    assert deep_thaw(payload.payload) == {"value": "first"}


def test_payload_preparation_stores_and_retrieves_the_frozen_canonical_snapshot(tmp_path: Path) -> None:
    preparation = importlib.import_module("elspeth.web.sessions.guided_payloads")
    store = FilesystemPayloadStore(tmp_path / "guided-payloads")

    payload = preparation.prepare_guided_json_payload(
        store,
        purpose="turn",
        payload={"question": "Choose", "options": ["a", "b"]},
    )

    assert store.retrieve(payload.payload_id)
    assert deep_thaw(payload.payload) == {"question": "Choose", "options": ["a", "b"]}


def test_payload_preparation_rejects_store_that_retrieves_altered_bytes() -> None:
    preparation = importlib.import_module("elspeth.web.sessions.guided_payloads")

    class _AlteredStore:
        def store(self, content: bytes) -> str:
            return stable_hash({"question": "Choose"})

        def retrieve(self, content_hash: str) -> bytes:
            return b'{"question":"ALTERED-PAYLOAD-CANARY"}'

        def exists(self, content_hash: str) -> bool:
            return True

        def delete(self, content_hash: str) -> bool:
            return False

    with pytest.raises(AuditIntegrityError, match="retrieval differs"):
        preparation.prepare_guided_json_payload(
            _AlteredStore(),
            purpose="turn",
            payload={"question": "Choose"},
        )


def test_next_turn_payload_requires_turn_purpose() -> None:
    payload = PreparedGuidedJsonPayload(
        payload_id=stable_hash({"question": "Choose"}),
        purpose="turn_response",
        payload={"question": "Choose"},
    )

    with pytest.raises(AuditIntegrityError, match="purpose=turn"):
        GuidedStateOperationCommand(
            fence=GuidedOperationFence(session_id=uuid4(), operation_id="op", lease_token="lease", attempt=1),
            expected_current_state_id=None,
            expected_current_state_version=None,
            expected_current_content_hash=None,
            state_id=uuid4(),
            state=CompositionStateData(is_valid=False),
            provenance="convergence_persist",
            actor="worker",
            response=GuidedResponseDescriptor(
                kind="guided_respond",
                next_turn=GuidedReplayTurn(
                    turn_type=TurnType.SINGLE_SELECT,
                    step_index=0,
                    payload_id=payload.payload_id,
                ),
                assistant_turn_seq=None,
            ),
            payloads=(payload,),
        )


def test_guided_checkpoint_omits_raw_validation_text_from_storage_and_replay() -> None:
    replay = importlib.import_module("elspeth.web.sessions.guided_replay")
    state = CompositionStateData(
        is_valid=False,
        validation_errors=["/home/operator/private.csv token=VALIDATION-CREDENTIAL-CANARY"],
    )

    prepared = replay.with_guided_response_descriptor(
        state,
        GuidedResponseDescriptor(kind="guided_respond", next_turn=None, assistant_turn_seq=None),
    )

    assert prepared.validation_errors is None


def test_audit_preparation_uses_real_typed_evidence_and_omits_hidden_provider_data() -> None:
    preparation = importlib.import_module("elspeth.web.sessions.guided_audit")
    invocation, llm_call, chat_turn = _audit_evidence()
    failing_llm_call = replace(
        llm_call,
        model_returned=None,
        status=ComposerLLMCallStatus.API_ERROR,
        error_class="ProviderError",
        error_message="raw-provider-secret-sk-123456789012345678901234567890",
    )
    failing_tool = replace(
        invocation,
        arguments_canonical='{"credential":"TOOL-CREDENTIAL-CANARY"}',
        arguments_hash=stable_hash({"credential": "TOOL-CREDENTIAL-CANARY"}),
        result_canonical='{"validation":"RAW-VALIDATION-CANARY"}',
        result_hash=stable_hash({"validation": "RAW-VALIDATION-CANARY"}),
        status=ComposerToolStatus.ARG_ERROR,
        error_class="ToolArgumentError",
        error_message="RAW-VALIDATION-CANARY",
        version_after=None,
    )
    unknown_success_tool = replace(
        invocation,
        tool_call_id="unknown-success-1",
        tool_name="unknown_success_tool",
        arguments_canonical='{"credential":"UNKNOWN-SUCCESS-CREDENTIAL"}',
        arguments_hash=stable_hash({"credential": "UNKNOWN-SUCCESS-CREDENTIAL"}),
        result_canonical='{"diagnostic":"UNKNOWN-SUCCESS-DIAGNOSTIC"}',
        result_hash=stable_hash({"diagnostic": "UNKNOWN-SUCCESS-DIAGNOSTIC"}),
    )

    rows = preparation.prepare_guided_audit_rows(
        invocations=(invocation, failing_tool, unknown_success_tool),
        llm_calls=(llm_call, failing_llm_call),
        chat_turns=(chat_turn,),
    )

    assert [row.kind for row in rows] == ["tool", "tool", "tool", "llm", "llm", "chat"]
    persisted = repr(deep_thaw([{"content": row.content, "envelope": row.envelope} for row in rows]))
    assert "SECRET-REASONING-CANARY" not in persisted
    assert "SECRET-PROVIDER-CANARY" not in persisted
    assert "raw-provider-secret" not in persisted
    assert "TOOL-CREDENTIAL-CANARY" not in persisted
    assert "RAW-VALIDATION-CANARY" not in persisted
    assert "UNKNOWN-SUCCESS-CREDENTIAL" not in persisted
    assert "UNKNOWN-SUCCESS-DIAGNOSTIC" not in persisted


def test_synthetic_audit_payload_reference_must_exist_in_verified_payload_set() -> None:
    preparation = importlib.import_module("elspeth.web.sessions.guided_audit")
    rows = preparation.prepare_guided_audit_rows(
        invocations=(_turn_emitted_evidence("a" * 64),),
        llm_calls=(),
        chat_turns=(),
    )

    with pytest.raises(AuditIntegrityError, match="absent or purpose-mismatched"):
        preparation.validate_guided_audit_payload_references(rows, ())


def test_replay_descriptor_replaces_stale_inherited_metadata() -> None:
    replay = importlib.import_module("elspeth.web.sessions.guided_replay")
    descriptor = GuidedResponseDescriptor(
        kind="guided_respond",
        next_turn=None,
        assistant_turn_seq=None,
    )
    state = CompositionStateData(
        is_valid=False,
        composer_meta={
            "guided_session": GuidedSession.initial().to_dict(),
            "guided_operation_replay": {"schema": "stale"},
            "repair_turns_used": 2,
        },
    )

    prepared = replay.with_guided_response_descriptor(state, descriptor)

    assert prepared.composer_meta is not None
    assert prepared.composer_meta["repair_turns_used"] == 2
    assert deep_thaw(prepared.composer_meta["guided_operation_replay"]) == descriptor.to_dict()


def test_respond_replay_projection_uses_persisted_descriptor_not_live_policy() -> None:
    replay = importlib.import_module("elspeth.web.sessions.guided_replay")
    guided = GuidedSession.initial()
    descriptor = GuidedResponseDescriptor(
        kind="guided_respond",
        next_turn=None,
        assistant_turn_seq=None,
    )
    state = CompositionStateData(
        is_valid=False,
        composer_meta={"guided_session": guided.to_dict()},
    )
    state = replay.with_guided_response_descriptor(state, descriptor)
    record = CompositionStateRecord(
        id=uuid4(),
        session_id=uuid4(),
        version=1,
        sources=None,
        source=None,
        nodes=None,
        edges=None,
        outputs=None,
        metadata_=None,
        is_valid=False,
        validation_errors=None,
        created_at=datetime.now(UTC),
        derived_from_state_id=None,
        composer_meta=state.composer_meta,
    )

    response = replay.project_guided_response(record, payloads=())

    assert type(response).__name__ == "GuidedRespondResponse"
    assert response.guided_session.step == guided.step.value
    assert response.next_turn is None
    assert response.composition_state is not None
    assert response.composition_state.plugin_policy_findings == []


def _replay_record(
    *,
    descriptor: GuidedResponseDescriptor,
    guided: GuidedSession,
    validation_errors: list[str] | None = None,
) -> CompositionStateRecord:
    replay = importlib.import_module("elspeth.web.sessions.guided_replay")
    state = replay.with_guided_response_descriptor(
        CompositionStateData(
            is_valid=False,
            validation_errors=validation_errors,
            composer_meta={"guided_session": guided.to_dict()},
        ),
        descriptor,
    )
    return CompositionStateRecord(
        id=uuid4(),
        session_id=uuid4(),
        version=1,
        sources=None,
        source=None,
        nodes=None,
        edges=None,
        outputs=None,
        metadata_=None,
        is_valid=False,
        validation_errors=state.validation_errors,
        created_at=datetime.now(UTC),
        derived_from_state_id=None,
        composer_meta=state.composer_meta,
    )


def test_replay_rejects_descriptor_step_index_that_disagrees_with_turn_record() -> None:
    replay = importlib.import_module("elspeth.web.sessions.guided_replay")
    payload = PreparedGuidedJsonPayload(
        payload_id=stable_hash({"question": "Choose"}),
        purpose="turn",
        payload={"question": "Choose"},
    )
    guided = replace(
        GuidedSession.initial(),
        history=(
            TurnRecord(
                step=GuidedStep.STEP_1_SOURCE,
                turn_type=TurnType.SINGLE_SELECT,
                payload_hash=payload.payload_id,
                response_hash=None,
                emitter="server",
            ),
        ),
    )
    descriptor = GuidedResponseDescriptor(
        kind="guided_respond",
        next_turn=GuidedReplayTurn(
            turn_type=TurnType.SINGLE_SELECT,
            step_index=1,
            payload_id=payload.payload_id,
        ),
        assistant_turn_seq=None,
    )

    with pytest.raises(AuditIntegrityError, match="persisted turn record"):
        replay.project_guided_response(_replay_record(descriptor=descriptor, guided=guided), payloads=(payload,))


def _chat_guided_session() -> GuidedSession:
    return replace(
        GuidedSession.initial(),
        chat_history=(
            ChatTurn(
                role=ChatRole.USER,
                content="What should I choose?",
                seq=0,
                step=GuidedStep.STEP_1_SOURCE,
                ts_iso="2026-07-18T00:00:00Z",
            ),
            ChatTurn(
                role=ChatRole.ASSISTANT,
                content="Choose the source that contains the records you need.",
                seq=1,
                step=GuidedStep.STEP_1_SOURCE,
                ts_iso="2026-07-18T00:00:01Z",
                assistant_message_kind="assistant",
            ),
        ),
        chat_turn_seq=2,
    )


def _guided_with_next_turn(payload_id: str) -> GuidedSession:
    return replace(
        GuidedSession.initial(),
        history=(
            TurnRecord(
                step=GuidedStep.STEP_1_SOURCE,
                turn_type=TurnType.SINGLE_SELECT,
                payload_hash=payload_id,
                response_hash=None,
                emitter="server",
            ),
        ),
    )


@pytest.mark.parametrize("response_kind", ["guided_respond", "guided_chat"])
@pytest.mark.parametrize("tamper", ["answered", "stale_stage"])
def test_replay_requires_final_turn_to_be_current_and_unanswered(response_kind: str, tamper: str) -> None:
    replay = importlib.import_module("elspeth.web.sessions.guided_replay")
    payload = PreparedGuidedJsonPayload(
        payload_id=stable_hash(
            {
                "question": "Choose a source",
                "options": [{"id": "csv", "label": "CSV", "hint": None}],
                "allow_custom": True,
            }
        ),
        purpose="turn",
        payload={
            "question": "Choose a source",
            "options": [{"id": "csv", "label": "CSV", "hint": None}],
            "allow_custom": True,
        },
    )
    guided = _guided_with_next_turn(payload.payload_id)
    if tamper == "answered":
        guided = replace(guided, history=(replace(guided.history[-1], response_hash="b" * 64),))
    else:
        guided = replace(guided, step=GuidedStep.STEP_2_SINK)
    assistant_turn_seq = None
    if response_kind == "guided_chat":
        chat = _chat_guided_session()
        guided = replace(guided, chat_history=chat.chat_history, chat_turn_seq=chat.chat_turn_seq)
        assistant_turn_seq = 1
    descriptor = GuidedResponseDescriptor(
        kind=response_kind,
        next_turn=GuidedReplayTurn(
            turn_type=TurnType.SINGLE_SELECT,
            step_index=0,
            payload_id=payload.payload_id,
        ),
        assistant_turn_seq=assistant_turn_seq,
    )

    with pytest.raises(AuditIntegrityError, match="current unanswered"):
        replay.project_guided_response(_replay_record(descriptor=descriptor, guided=guided), payloads=(payload,))


@pytest.fixture
def service_and_engine(tmp_path: Path):
    engine = create_session_engine(f"sqlite:///{tmp_path / 'guided-atomic.db'}")
    initialize_session_schema(engine)
    service = SessionServiceImpl(
        engine,
        telemetry=build_sessions_telemetry(),
        log=structlog.get_logger("test.guided-atomic"),
    )
    try:
        yield service, engine
    finally:
        engine.dispose()


@pytest.mark.asyncio
async def test_respond_settlement_commits_state_audit_and_operation_as_one_cohort(service_and_engine) -> None:
    service, engine = service_and_engine
    session_id = (await service.create_session("alice", "guided atomic", "local")).id
    claimed = await service.reserve_guided_operation(
        session_id=session_id,
        operation_id="respond-cohort",
        kind="guided_respond",
        request_hash="a" * 64,
        actor="worker",
        lease_seconds=60,
    )
    assert isinstance(claimed, GuidedOperationClaimed)
    state_id = uuid4()
    invocation, llm_call, chat_turn = _audit_evidence()
    command = GuidedStateOperationCommand(
        fence=claimed.fence,
        expected_current_state_id=None,
        expected_current_state_version=None,
        expected_current_content_hash=None,
        state_id=state_id,
        state=CompositionStateData(
            metadata_={},
            is_valid=False,
            composer_meta={"guided_session": GuidedSession.initial().to_dict()},
        ),
        provenance="convergence_persist",
        actor="worker",
        response=GuidedResponseDescriptor(
            kind="guided_respond",
            next_turn=None,
            assistant_turn_seq=None,
        ),
        audit_evidence=GuidedAuditEvidence(
            invocations=(invocation,),
            llm_calls=(llm_call,),
            chat_turns=(chat_turn,),
        ),
    )

    settlement = await service.settle_guided_state_operation(command)

    assert settlement.primary_state.id == state_id
    assert settlement.result_state.id == state_id
    assert settlement.response_hash == stable_hash(settlement.response_json)
    assert [message.sequence_no for message in settlement.audit_messages] == [1, 2, 3]
    with engine.connect() as conn:
        states = conn.execute(select(composition_states_table).where(composition_states_table.c.session_id == str(session_id))).all()
        messages = conn.execute(
            select(chat_messages_table)
            .where(chat_messages_table.c.session_id == str(session_id))
            .order_by(chat_messages_table.c.sequence_no)
        ).all()
        operation = conn.execute(select(guided_operations_table).where(guided_operations_table.c.session_id == str(session_id))).one()
    assert [row.id for row in states] == [str(state_id)]
    assert [row.role for row in messages] == ["audit", "audit", "audit"]
    assert [row.composition_state_id for row in messages] == [str(state_id)] * 3
    assert operation.status == "completed"
    assert operation.result_state_id == str(state_id)
    assert operation.response_hash == settlement.response_hash


@pytest.mark.asyncio
async def test_chat_settlement_commits_exact_final_assistant_replay(service_and_engine) -> None:
    service, engine = service_and_engine
    session_id = (await service.create_session("alice", "guided chat", "local")).id
    claimed = await service.reserve_guided_operation(
        session_id=session_id,
        operation_id="chat-cohort",
        kind="guided_chat",
        request_hash="a" * 64,
        actor="worker",
        lease_seconds=60,
    )
    assert isinstance(claimed, GuidedOperationClaimed)
    command = GuidedStateOperationCommand(
        fence=claimed.fence,
        expected_current_state_id=None,
        expected_current_state_version=None,
        expected_current_content_hash=None,
        state_id=uuid4(),
        state=CompositionStateData(
            is_valid=False,
            composer_meta={"guided_session": _chat_guided_session().to_dict()},
        ),
        provenance="convergence_persist",
        actor="worker",
        response=GuidedResponseDescriptor(kind="guided_chat", next_turn=None, assistant_turn_seq=1),
        originating_message=GuidedOriginatingUserMessageDraft(
            message_id=uuid4(),
            content="What should I choose?",
        ),
    )

    settlement = await service.settle_guided_state_operation(command)

    assert settlement.response_json["assistant_message"] == "Choose the source that contains the records you need."
    assert settlement.response_json["assistant_message_kind"] == "assistant"
    assert settlement.response_hash == stable_hash(settlement.response_json)
    with engine.connect() as conn:
        operation = conn.execute(select(guided_operations_table)).one()
    assert operation.kind == "guided_chat"
    assert operation.status == "completed"


@pytest.mark.asyncio
async def test_settlement_requires_store_retrieval_for_every_payload_and_can_retry(service_and_engine, tmp_path: Path) -> None:
    service, engine = service_and_engine
    session_id = (await service.create_session("alice", "guided CAS", "local")).id
    claimed = await service.reserve_guided_operation(
        session_id=session_id,
        operation_id="respond-cas",
        kind="guided_respond",
        request_hash="a" * 64,
        actor="worker",
        lease_seconds=60,
    )
    assert isinstance(claimed, GuidedOperationClaimed)
    store = FilesystemPayloadStore(tmp_path / "cas-payloads")
    preparation = importlib.import_module("elspeth.web.sessions.guided_payloads")
    payload = preparation.prepare_guided_json_payload(
        store,
        purpose="turn",
        payload={
            "question": "Choose a source",
            "options": [{"id": "csv", "label": "CSV", "hint": None}],
            "allow_custom": True,
        },
    )
    command = GuidedStateOperationCommand(
        fence=claimed.fence,
        expected_current_state_id=None,
        expected_current_state_version=None,
        expected_current_content_hash=None,
        state_id=uuid4(),
        state=CompositionStateData(
            is_valid=False,
            composer_meta={"guided_session": _guided_with_next_turn(payload.payload_id).to_dict()},
        ),
        provenance="convergence_persist",
        actor="worker",
        response=GuidedResponseDescriptor(
            kind="guided_respond",
            next_turn=GuidedReplayTurn(
                turn_type=TurnType.SINGLE_SELECT,
                step_index=0,
                payload_id=payload.payload_id,
            ),
            assistant_turn_seq=None,
        ),
        payloads=(payload,),
        audit_evidence=GuidedAuditEvidence(invocations=(_turn_emitted_evidence(payload.payload_id),)),
    )

    with pytest.raises(AuditIntegrityError, match="PayloadStore"):
        await service.settle_guided_state_operation(command)
    with engine.connect() as conn:
        assert conn.execute(select(composition_states_table)).all() == []
        assert conn.execute(select(chat_messages_table)).all() == []

    settlement = await service.settle_guided_state_operation(command, payload_store=store)
    assert settlement.response_json["next_turn"]["payload"]["question"] == "Choose a source"
    assert settlement.response_hash == stable_hash(settlement.response_json)


@pytest.mark.asyncio
async def test_late_projection_failure_rolls_back_inserted_cohort_and_same_fence_retries(
    service_and_engine,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service, engine = service_and_engine
    session_id = (await service.create_session("alice", "guided late rollback", "local")).id
    claimed = await service.reserve_guided_operation(
        session_id=session_id,
        operation_id="respond-late-rollback",
        kind="guided_respond",
        request_hash="a" * 64,
        actor="worker",
        lease_seconds=60,
    )
    assert isinstance(claimed, GuidedOperationClaimed)
    invocation, llm_call, chat_turn = _audit_evidence()
    command = GuidedStateOperationCommand(
        fence=claimed.fence,
        expected_current_state_id=None,
        expected_current_state_version=None,
        expected_current_content_hash=None,
        state_id=uuid4(),
        state=CompositionStateData(
            is_valid=False,
            composer_meta={"guided_session": GuidedSession.initial().to_dict()},
        ),
        provenance="convergence_persist",
        actor="worker",
        response=GuidedResponseDescriptor(kind="guided_respond", next_turn=None, assistant_turn_seq=None),
        audit_evidence=GuidedAuditEvidence(
            invocations=(invocation,),
            llm_calls=(llm_call,),
            chat_turns=(chat_turn,),
        ),
        originating_message=GuidedOriginatingUserMessageDraft(message_id=uuid4(), content="continue"),
    )
    service_module = importlib.import_module("elspeth.web.sessions.service")
    projector = service_module.project_guided_response

    def _wrong_purpose_after_writes(*_args, **_kwargs):
        raise AuditIntegrityError("injected next-turn purpose=turn_response failure")

    monkeypatch.setattr(service_module, "project_guided_response", _wrong_purpose_after_writes)
    with pytest.raises(AuditIntegrityError, match="purpose=turn_response"):
        await service.settle_guided_state_operation(command)

    with engine.connect() as conn:
        assert conn.execute(select(composition_states_table)).all() == []
        assert conn.execute(select(chat_messages_table)).all() == []
        assert conn.execute(select(interpretation_events_table)).all() == []
        operation = conn.execute(select(guided_operations_table)).one()
    assert operation.status == "in_progress"
    assert operation.result_state_id is None
    assert operation.response_hash is None

    monkeypatch.setattr(service_module, "project_guided_response", projector)
    settlement = await service.settle_guided_state_operation(command)
    assert settlement.result_state.id == command.state_id


@pytest.mark.asyncio
async def test_failure_after_operation_complete_rolls_back_bind_terminal_and_all_cohort_rows(
    service_and_engine,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service, engine = service_and_engine
    session_id = (await service.create_session("alice", "guided completion rollback", "local")).id
    claimed = await service.reserve_guided_operation(
        session_id=session_id,
        operation_id="respond-completion-rollback",
        kind="guided_respond",
        request_hash="a" * 64,
        actor="worker",
        lease_seconds=60,
    )
    assert isinstance(claimed, GuidedOperationClaimed)
    command = _empty_respond_command(claimed.fence)
    complete = service.complete_guided_operation_on_connection

    def _complete_then_fail(*args, **kwargs):
        complete(*args, **kwargs)
        raise AuditIntegrityError("injected failure after terminal update")

    monkeypatch.setattr(service, "complete_guided_operation_on_connection", _complete_then_fail)
    with pytest.raises(AuditIntegrityError, match="after terminal"):
        await service.settle_guided_state_operation(command)

    with engine.connect() as conn:
        assert conn.execute(select(composition_states_table)).all() == []
        assert conn.execute(select(chat_messages_table)).all() == []
        operation = conn.execute(select(guided_operations_table)).one()
    assert operation.status == "in_progress"
    assert operation.originating_message_id is None
    assert operation.result_state_id is None
    assert operation.response_hash is None


def _empty_respond_command(fence: GuidedOperationFence, *, state_id=None) -> GuidedStateOperationCommand:
    return GuidedStateOperationCommand(
        fence=fence,
        expected_current_state_id=None,
        expected_current_state_version=None,
        expected_current_content_hash=None,
        state_id=state_id or uuid4(),
        state=CompositionStateData(
            is_valid=False,
            composer_meta={"guided_session": GuidedSession.initial().to_dict()},
        ),
        provenance="convergence_persist",
        actor="worker",
        response=GuidedResponseDescriptor(kind="guided_respond", next_turn=None, assistant_turn_seq=None),
    )


async def _seed_present_guided_predecessor(service: SessionServiceImpl, session_id) -> CompositionStateRecord:
    return await service.save_composition_state(
        session_id,
        CompositionStateData(
            metadata_={"name": "Guided predecessor", "description": ""},
            is_valid=False,
            composer_meta={"guided_session": GuidedSession.initial().to_dict()},
        ),
        provenance="convergence_persist",
    )


def _present_respond_command(
    fence: GuidedOperationFence,
    predecessor: CompositionStateRecord,
) -> GuidedStateOperationCommand:
    invocation, llm_call, chat_turn = _audit_evidence()
    return GuidedStateOperationCommand(
        fence=fence,
        expected_current_state_id=predecessor.id,
        expected_current_state_version=predecessor.version,
        expected_current_content_hash=composition_content_hash(state_from_record(predecessor)),
        state_id=uuid4(),
        state=CompositionStateData(
            metadata_={"name": "Guided successor", "description": ""},
            is_valid=False,
            composer_meta={"guided_session": GuidedSession.initial().to_dict()},
        ),
        provenance="convergence_persist",
        actor="worker",
        response=GuidedResponseDescriptor(kind="guided_respond", next_turn=None, assistant_turn_seq=None),
        audit_evidence=GuidedAuditEvidence(
            invocations=(invocation,),
            llm_calls=(llm_call,),
            chat_turns=(chat_turn,),
        ),
        originating_message=GuidedOriginatingUserMessageDraft(message_id=uuid4(), content="continue"),
    )


@pytest.mark.asyncio
async def test_present_expected_head_exact_match_settles_derived_successor(service_and_engine) -> None:
    service, engine = service_and_engine
    session_id = (await service.create_session("alice", "guided present head", "local")).id
    predecessor = await _seed_present_guided_predecessor(service, session_id)
    claimed = await service.reserve_guided_operation(
        session_id=session_id,
        operation_id="respond-present-exact",
        kind="guided_respond",
        request_hash="a" * 64,
        actor="worker",
        lease_seconds=60,
    )
    assert isinstance(claimed, GuidedOperationClaimed)
    command = _present_respond_command(claimed.fence, predecessor)

    settlement = await service.settle_guided_state_operation(command)

    assert settlement.primary_state.id == command.state_id
    assert settlement.primary_state.version == predecessor.version + 1
    assert settlement.primary_state.derived_from_state_id == predecessor.id
    with engine.connect() as conn:
        states = conn.execute(
            select(composition_states_table)
            .where(composition_states_table.c.session_id == str(session_id))
            .order_by(composition_states_table.c.version)
        ).all()
    assert [row.id for row in states] == [str(predecessor.id), str(command.state_id)]


@pytest.mark.asyncio
@pytest.mark.parametrize("mismatch", ["id", "version", "content_hash"])
async def test_present_expected_head_mismatch_rolls_back_and_same_fence_retries(service_and_engine, mismatch: str) -> None:
    service, engine = service_and_engine
    session_id = (await service.create_session("alice", f"guided present {mismatch}", "local")).id
    predecessor = await _seed_present_guided_predecessor(service, session_id)
    claimed = await service.reserve_guided_operation(
        session_id=session_id,
        operation_id=f"respond-present-{mismatch}",
        kind="guided_respond",
        request_hash="a" * 64,
        actor="worker",
        lease_seconds=60,
    )
    assert isinstance(claimed, GuidedOperationClaimed)
    exact = _present_respond_command(claimed.fence, predecessor)
    replacements = {
        "id": {"expected_current_state_id": uuid4()},
        "version": {"expected_current_state_version": predecessor.version + 1},
        "content_hash": {"expected_current_content_hash": "f" * 64},
    }[mismatch]
    bad = replace(exact, **replacements)

    with pytest.raises((StaleComposeStateError, AuditIntegrityError)):
        await service.settle_guided_state_operation(bad)

    with engine.connect() as conn:
        states = conn.execute(select(composition_states_table).where(composition_states_table.c.session_id == str(session_id))).all()
        messages = conn.execute(select(chat_messages_table).where(chat_messages_table.c.session_id == str(session_id))).all()
        operation = conn.execute(select(guided_operations_table).where(guided_operations_table.c.session_id == str(session_id))).one()
    assert [row.id for row in states] == [str(predecessor.id)]
    assert messages == []
    assert operation.status == "in_progress"
    assert operation.result_state_id is None
    assert operation.response_hash is None

    settlement = await service.settle_guided_state_operation(exact)
    assert settlement.primary_state.id == exact.state_id
    assert settlement.primary_state.derived_from_state_id == predecessor.id


@pytest.mark.asyncio
async def test_expired_then_taken_over_fence_allows_only_new_attempt_to_write(service_and_engine) -> None:
    service, engine = service_and_engine
    session_id = (await service.create_session("alice", "guided stale race", "local")).id
    operation_id = "respond-stale-race"
    first = await service.reserve_guided_operation(
        session_id=session_id,
        operation_id=operation_id,
        kind="guided_respond",
        request_hash="a" * 64,
        actor="worker-a",
        lease_seconds=60,
    )
    assert isinstance(first, GuidedOperationClaimed)
    with engine.begin() as conn:
        conn.execute(
            update(guided_operations_table)
            .where(guided_operations_table.c.session_id == str(session_id))
            .values(lease_expires_at=datetime.now(UTC) - timedelta(minutes=1))
        )

    with pytest.raises(GuidedOperationFenceLostError):
        await service.settle_guided_state_operation(_empty_respond_command(first.fence))
    with engine.connect() as conn:
        assert conn.execute(select(composition_states_table)).all() == []
        assert conn.execute(select(chat_messages_table)).all() == []

    second = await service.reserve_guided_operation(
        session_id=session_id,
        operation_id=operation_id,
        kind="guided_respond",
        request_hash="a" * 64,
        actor="worker-b",
        lease_seconds=60,
    )
    assert isinstance(second, GuidedOperationTakenOver)
    with pytest.raises(GuidedOperationFenceLostError):
        await service.settle_guided_state_operation(_empty_respond_command(first.fence))

    winning = _empty_respond_command(second.fence)
    settlement = await service.settle_guided_state_operation(winning)
    assert settlement.primary_state.id == winning.state_id
    with engine.connect() as conn:
        states = conn.execute(select(composition_states_table)).all()
    assert [row.id for row in states] == [str(winning.state_id)]


@pytest.mark.asyncio
async def test_cross_kind_settlement_rolls_back_without_cohort_rows(service_and_engine) -> None:
    service, engine = service_and_engine
    session_id = (await service.create_session("alice", "guided cross kind", "local")).id
    claimed = await service.reserve_guided_operation(
        session_id=session_id,
        operation_id="chat-cross-kind",
        kind="guided_chat",
        request_hash="a" * 64,
        actor="worker",
        lease_seconds=60,
    )
    assert isinstance(claimed, GuidedOperationClaimed)

    with pytest.raises(AuditIntegrityError, match="kind"):
        await service.settle_guided_state_operation(_empty_respond_command(claimed.fence))

    with engine.connect() as conn:
        assert conn.execute(select(composition_states_table)).all() == []
        assert conn.execute(select(chat_messages_table)).all() == []
        operation = conn.execute(select(guided_operations_table)).one()
    assert operation.status == "in_progress"


def _llm_interpretation_node(node_id: str = "llm_transform_1") -> dict[str, object]:
    return {
        "id": node_id,
        "node_type": "transform",
        "plugin": "llm",
        "input": "source",
        "on_success": "output",
        "on_error": "quarantine",
        "options": {"prompt_template": "Rate how {{interpretation:cool}} this is."},
        "condition": None,
        "routes": None,
        "fork_to": None,
        "branches": None,
        "policy": None,
        "merge": None,
    }


def _interpretation_draft(*, node_id: str = "llm_transform_1") -> PreparedGuidedInterpretationDraft:
    return PreparedGuidedInterpretationDraft(
        event_id=uuid4(),
        affected_node_id=node_id,
        tool_call_id="interpretation-call-1",
        user_term="cool",
        kind=InterpretationKind.VAGUE_TERM,
        llm_draft="A concise definition of cool",
        model_identifier="test-model",
        model_version="v1",
        provider="test-provider",
        composer_skill_hash="c" * 64,
    )


@pytest.mark.asyncio
async def test_respond_settlement_commits_pending_interpretation_in_same_cohort(service_and_engine) -> None:
    service, engine = service_and_engine
    session_id = (await service.create_session("alice", "guided interpretation", "local")).id
    claimed = await service.reserve_guided_operation(
        session_id=session_id,
        operation_id="respond-interpretation",
        kind="guided_respond",
        request_hash="a" * 64,
        actor="worker",
        lease_seconds=60,
    )
    assert isinstance(claimed, GuidedOperationClaimed)
    draft = _interpretation_draft()
    command = GuidedStateOperationCommand(
        fence=claimed.fence,
        expected_current_state_id=None,
        expected_current_state_version=None,
        expected_current_content_hash=None,
        state_id=uuid4(),
        state=CompositionStateData(
            nodes=[_llm_interpretation_node()],
            metadata_={"name": "Guided opt out", "description": ""},
            is_valid=False,
            composer_meta={"guided_session": GuidedSession.initial().to_dict()},
        ),
        provenance="convergence_persist",
        actor="worker",
        response=GuidedResponseDescriptor(kind="guided_respond", next_turn=None, assistant_turn_seq=None),
        interpretations=(draft,),
    )

    settlement = await service.settle_guided_state_operation(command)

    assert [event.id for event in settlement.interpretations] == [draft.event_id]
    with engine.connect() as conn:
        event = conn.execute(select(interpretation_events_table)).one()
        operation = conn.execute(select(guided_operations_table)).one()
    assert event.composition_state_id == str(command.state_id)
    assert event.choice == "pending"
    assert operation.status == "completed"


@pytest.mark.asyncio
async def test_malformed_interpretation_rolls_back_state_messages_event_and_operation_then_retries(service_and_engine) -> None:
    service, engine = service_and_engine
    session_id = (await service.create_session("alice", "guided rollback", "local")).id
    claimed = await service.reserve_guided_operation(
        session_id=session_id,
        operation_id="respond-rollback",
        kind="guided_respond",
        request_hash="a" * 64,
        actor="worker",
        lease_seconds=60,
    )
    assert isinstance(claimed, GuidedOperationClaimed)
    invocation, llm_call, chat_turn = _audit_evidence()
    bad_command = GuidedStateOperationCommand(
        fence=claimed.fence,
        expected_current_state_id=None,
        expected_current_state_version=None,
        expected_current_content_hash=None,
        state_id=uuid4(),
        state=CompositionStateData(
            nodes=[_llm_interpretation_node()],
            metadata_={"name": "Guided opt out", "description": ""},
            is_valid=False,
            composer_meta={"guided_session": GuidedSession.initial().to_dict()},
        ),
        provenance="convergence_persist",
        actor="worker",
        response=GuidedResponseDescriptor(kind="guided_respond", next_turn=None, assistant_turn_seq=None),
        audit_evidence=GuidedAuditEvidence(
            invocations=(invocation,),
            llm_calls=(llm_call,),
            chat_turns=(chat_turn,),
        ),
        originating_message=GuidedOriginatingUserMessageDraft(message_id=uuid4(), content="please continue"),
        interpretations=(_interpretation_draft(node_id="missing_node"),),
    )

    with pytest.raises(ValueError, match="missing_node"):
        await service.settle_guided_state_operation(bad_command)

    with engine.connect() as conn:
        assert conn.execute(select(composition_states_table)).all() == []
        assert conn.execute(select(chat_messages_table)).all() == []
        assert conn.execute(select(interpretation_events_table)).all() == []
        operation = conn.execute(select(guided_operations_table)).one()
    assert operation.status == "in_progress"
    assert operation.result_state_id is None
    assert operation.response_hash is None

    retry = replace(
        bad_command,
        state_id=uuid4(),
        interpretations=(_interpretation_draft(),),
    )
    settlement = await service.settle_guided_state_operation(retry)
    assert settlement.result_state.id == retry.state_id


@pytest.mark.asyncio
async def test_opted_out_interpretation_resolves_and_binds_derived_result_in_same_cohort(service_and_engine) -> None:
    service, engine = service_and_engine
    session_id = (await service.create_session("alice", "guided opt out", "local")).id
    with engine.begin() as conn:
        conn.execute(update(sessions_table).where(sessions_table.c.id == str(session_id)).values(interpretation_review_disabled=True))
    claimed = await service.reserve_guided_operation(
        session_id=session_id,
        operation_id="respond-opt-out",
        kind="guided_respond",
        request_hash="a" * 64,
        actor="worker",
        lease_seconds=60,
    )
    assert isinstance(claimed, GuidedOperationClaimed)
    command = GuidedStateOperationCommand(
        fence=claimed.fence,
        expected_current_state_id=None,
        expected_current_state_version=None,
        expected_current_content_hash=None,
        state_id=uuid4(),
        state=CompositionStateData(
            nodes=[_llm_interpretation_node()],
            metadata_={"name": "Guided opt out", "description": ""},
            is_valid=False,
            composer_meta={"guided_session": GuidedSession.initial().to_dict()},
        ),
        provenance="convergence_persist",
        actor="worker",
        response=GuidedResponseDescriptor(kind="guided_respond", next_turn=None, assistant_turn_seq=None),
        interpretations=(_interpretation_draft(),),
    )

    settlement = await service.settle_guided_state_operation(command)

    assert settlement.primary_state.id == command.state_id
    assert settlement.result_state.id != settlement.primary_state.id
    assert settlement.interpretations[0].choice.value == "opted_out"
    with engine.connect() as conn:
        operation = conn.execute(select(guided_operations_table)).one()
        events = conn.execute(select(interpretation_events_table)).all()
    assert operation.result_state_id == str(settlement.result_state.id)
    assert len(events) == 2  # one durable opt-out marker and one resolved surface
