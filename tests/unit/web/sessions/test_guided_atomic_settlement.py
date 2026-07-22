"""Atomic guided RESPOND/CHAT settlement and replay contracts."""

from __future__ import annotations

import importlib
import json
from collections.abc import Iterator, Mapping
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from uuid import UUID, uuid4

import pytest
import structlog
from sqlalchemy import func, select, update

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
from elspeth.contracts.hashing import canonical_json, stable_hash
from elspeth.core.payload_store import FilesystemPayloadStore
from elspeth.web.composer.audit import BufferingRecorder
from elspeth.web.composer.guided.audit import emit_intent_cancelled
from elspeth.web.composer.guided.deferred_intents import DeferredIntentAction, DeferredIntentCancelAction, DeferredIntentEditAction
from elspeth.web.composer.guided.errors import InvariantError
from elspeth.web.composer.guided.intent_management import deferred_intent_management_option
from elspeth.web.composer.guided.protocol import ChatRole, ChatTurn, GuidedStep, TurnType
from elspeth.web.composer.guided.stage_subjects import ComponentCountConstraint
from elspeth.web.composer.guided.state_machine import (
    DeferredStageIntent,
    GuidedProposalRef,
    GuidedSession,
    TurnRecord,
    guided_reviewed_anchor_hash,
)
from elspeth.web.composer.pipeline_proposal import AbsentBase, composition_content_hash
from elspeth.web.composer.state import CompositionState, PipelineMetadata
from elspeth.web.interpretation_state import PROMPT_SHIELD_AVAILABLE_DRAFT, PROMPT_SHIELD_WARNING_DRAFT
from elspeth.web.plugin_policy.models import PluginId, PluginUnavailableReason
from elspeth.web.sessions.converters import state_from_record
from elspeth.web.sessions.engine import create_session_engine
from elspeth.web.sessions.models import (
    chat_messages_table,
    composition_states_table,
    guided_operation_events_table,
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
    GuidedOperationSettlementConflictError,
    GuidedOperationTakenOver,
    GuidedOriginatingUserMessageDraft,
    GuidedReplayPolicyFinding,
    GuidedReplayTurn,
    GuidedResponseDescriptor,
    GuidedStateOperationCommand,
    PreparedGuidedInterpretationDraft,
    PreparedGuidedJsonPayload,
    guided_json_payload_id,
)
from elspeth.web.sessions.schema import initialize_session_schema
from elspeth.web.sessions.service import SessionServiceImpl
from elspeth.web.sessions.telemetry import build_sessions_telemetry


def _empty_composition_state() -> CompositionState:
    return CompositionState(
        nodes=(),
        edges=(),
        outputs=(),
        metadata=PipelineMetadata(),
        version=1,
    )


_GUIDED_PROPOSAL_ID = UUID("00000000-0000-4000-8000-000000000001")
_GUIDED_PROPOSAL_DRAFT_HASH = "d" * 64


def _guided_proposal_ref() -> GuidedProposalRef:
    return GuidedProposalRef(
        proposal_id=_GUIDED_PROPOSAL_ID,
        draft_hash=_GUIDED_PROPOSAL_DRAFT_HASH,
        base=AbsentBase(),
        reviewed_anchor_hash=guided_reviewed_anchor_hash(
            source_order=(),
            reviewed_sources={},
            output_order=(),
            reviewed_outputs={},
        ),
        covered_deferred_intent_ids=(),
        creation_event_schema="pipeline_proposal_created.v1",
    )


def _empty_wire_payload(
    *,
    proposal_id: str = str(_GUIDED_PROPOSAL_ID),
    draft_hash: str = _GUIDED_PROPOSAL_DRAFT_HASH,
    warnings: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    return {
        "proposal_id": proposal_id,
        "draft_hash": draft_hash,
        "sources": [],
        "nodes": [],
        "outputs": [],
        "connections": [],
        "semantic_contracts": [],
        "warnings": warnings or [],
        "blockers": [],
        "can_confirm": True,
    }


_MALFORMED_CURRENT_TURNS: tuple[tuple[GuidedStep, TurnType, Mapping[str, object]], ...] = (
    (
        GuidedStep.STEP_1_SOURCE,
        TurnType.SINGLE_SELECT,
        {
            "question": "Choose",
            "options": [{"id": "csv", "label": "CSV", "hint": None, "canary": True}],
            "allow_custom": False,
        },
    ),
    (
        GuidedStep.STEP_1_SOURCE,
        TurnType.INSPECT_AND_CONFIRM,
        {"observed": {"columns": [1], "samples": [], "warnings": []}},
    ),
    (
        GuidedStep.STEP_1_SOURCE,
        TurnType.SCHEMA_FORM,
        {
            "mode": "plugin_options",
            "plugin": "csv",
            "knobs": {
                "fields": [
                    {
                        "name": "path",
                        "label": "Path",
                        "kind": "credential-canary",
                        "required": True,
                        "nullable": False,
                    }
                ]
            },
            "prefilled": {},
        },
    ),
)


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


@pytest.mark.parametrize(
    ("shield_available", "expected_fragment", "forbidden_fragment"),
    [
        (False, PROMPT_SHIELD_WARNING_DRAFT, PROMPT_SHIELD_AVAILABLE_DRAFT),
        (True, PROMPT_SHIELD_AVAILABLE_DRAFT, PROMPT_SHIELD_WARNING_DRAFT),
    ],
)
def test_confirm_wiring_cas_is_exact_finalized_wire_authority(
    tmp_path: Path,
    shield_available: bool,
    expected_fragment: str,
    forbidden_fragment: str,
) -> None:
    from elspeth.web.sessions.routes.composer.guided import (
        _finalize_guided_turn,
        _load_durable_current_turn,
        _prepare_server_turn_occurrence,
        _turn_payload_response,
    )

    store = FilesystemPayloadStore(tmp_path / "wire-authority")
    raw_turn = {
        "type": "confirm_wiring",
        "step_index": 3,
        "payload": _empty_wire_payload(
            warnings=[
                {
                    "component": "node:llm",
                    "message": f"lead {PROMPT_SHIELD_WARNING_DRAFT}",
                    "severity": "medium",
                }
            ]
        ),
    }
    finalized = _finalize_guided_turn(raw_turn, shield_available=shield_available)
    guided = replace(GuidedSession.initial(), step=GuidedStep.STEP_4_WIRE)
    guided, record, _turn_type, prepared = _prepare_server_turn_occurrence(
        guided,
        current_step=GuidedStep.STEP_4_WIRE,
        turn=finalized,
        payload_store=store,
    )
    guided = replace(guided, active_proposal=_guided_proposal_ref())

    response = _turn_payload_response(finalized, guided=guided, shield_available=not shield_available)
    reloaded_turn, reloaded_payload = _load_durable_current_turn(guided, payload_store=store)
    replay = _turn_payload_response(reloaded_turn, guided=guided, shield_available=not shield_available)

    assert response is not None
    assert replay == response
    assert deep_thaw(reloaded_payload.payload) == response.payload
    assert prepared.payload_id == record.payload_hash
    messages = [warning["message"] for warning in response.payload["warnings"]]
    assert any(expected_fragment in message for message in messages)
    assert not any(forbidden_fragment in message for message in messages)


@pytest.mark.parametrize(
    "turn",
    [
        {
            "type": "single_select",
            "step_index": 0,
            "payload": {"question": "Choose", "options": [], "allow_custom": False},
            "credential_canary": "MUST-NOT-ENTER-CAS",
        },
        {
            "type": "confirm_wiring",
            "step_index": 0,
            "payload": _empty_wire_payload(),
        },
    ],
)
def test_turn_construction_rejects_extra_keys_and_wrong_step_matrix(tmp_path: Path, turn: Mapping[str, object]) -> None:
    from elspeth.web.sessions.routes.composer.guided import _prepare_server_turn_occurrence

    store = FilesystemPayloadStore(tmp_path / "invalid-turn-construction")
    with pytest.raises(InvariantError):
        _prepare_server_turn_occurrence(
            GuidedSession.initial(),
            current_step=GuidedStep.STEP_1_SOURCE,
            turn=turn,
            payload_store=store,
        )
    assert not tuple((tmp_path / "invalid-turn-construction").glob("**/*"))


@pytest.mark.parametrize(("step", "turn_type", "payload"), _MALFORMED_CURRENT_TURNS)
def test_turn_construction_rejects_recursively_malformed_payload_before_cas(
    tmp_path: Path,
    step: GuidedStep,
    turn_type: TurnType,
    payload: Mapping[str, object],
) -> None:
    from elspeth.web.sessions.routes.composer.guided import _prepare_server_turn_occurrence

    store = FilesystemPayloadStore(tmp_path / f"invalid-recursive-{turn_type.value}")
    with pytest.raises(InvariantError, match="Constructed current-schema turn is invalid"):
        _prepare_server_turn_occurrence(
            GuidedSession.initial(),
            current_step=step,
            turn={"type": turn_type.value, "step_index": 0, "payload": payload},
            payload_store=store,
        )
    assert not tuple((tmp_path / f"invalid-recursive-{turn_type.value}").glob("**/*"))


def test_direct_turn_append_rejects_unvalidated_turn() -> None:
    from elspeth.web.sessions.routes.composer.guided import _append_server_turn_record

    invalid_turn = {
        "type": "single_select",
        "step_index": 0,
        "payload": {
            "question": "Choose",
            "options": [],
            "allow_custom": False,
            "credential_canary": "MUST-NOT-ENTER-HISTORY",
        },
    }

    with pytest.raises(InvariantError, match="Constructed current-schema turn is invalid"):
        _append_server_turn_record(
            GuidedSession.initial(),
            current_step=GuidedStep.STEP_1_SOURCE,
            turn=invalid_turn,
        )


def test_schema8_prospective_occurrence_validates_before_history_or_cas(tmp_path: Path, monkeypatch) -> None:
    guided_route = importlib.import_module("elspeth.web.sessions.routes.composer.guided")
    store = FilesystemPayloadStore(tmp_path / "prospective-invalid")
    invalid_turn = {
        "type": "single_select",
        "step_index": 0,
        "payload": {
            "question": "Choose",
            "options": [],
            "allow_custom": False,
            "credential_canary": "MUST-NOT-ENTER-CAS",
        },
    }
    monkeypatch.setattr(guided_route, "_build_get_guided_turn", lambda *_args, **_kwargs: invalid_turn)

    with pytest.raises(InvariantError, match="Constructed current-schema turn is invalid"):
        guided_route._schema8_prospective_occurrence(
            _empty_composition_state(),
            GuidedSession.initial(),
            catalog=cast(Any, object()),
            shield_available=False,
            payload_store=store,
        )
    assert not tuple((tmp_path / "prospective-invalid").glob("**/*"))


def test_schema8_projected_next_turn_validates_before_history(monkeypatch) -> None:
    guided_route = importlib.import_module("elspeth.web.sessions.routes.composer.guided")
    current_turn = {
        "type": "single_select",
        "step_index": 0,
        "payload": {"question": "Choose", "options": [], "allow_custom": False},
    }
    guided = replace(
        GuidedSession.initial(),
        history=(
            TurnRecord(
                step=GuidedStep.STEP_1_SOURCE,
                turn_type=TurnType.SINGLE_SELECT,
                payload_hash="a" * 64,
                response_hash=None,
                emitter="server",
            ),
        ),
    )
    invalid_next = {
        **current_turn,
        "payload": {**current_turn["payload"], "credential_canary": "MUST-NOT-ENTER-HISTORY"},
    }
    monkeypatch.setattr(
        guided_route,
        "_schema8_transition",
        lambda *_args, **_kwargs: (guided, {"chosen": ["csv"]}),
    )
    monkeypatch.setattr(guided_route, "_build_get_guided_turn", lambda *_args, **_kwargs: invalid_next)
    body = SimpleNamespace(control_signal=None)

    with pytest.raises(InvariantError, match="Constructed current-schema turn is invalid"):
        guided_route._schema8_answer_and_project_next(
            _empty_composition_state(),
            guided,
            current_turn,
            body,
            catalog=cast(Any, object()),
            shield_available=False,
            new_stable_id=uuid4(),
        )


def test_durable_current_turn_rejects_wrong_step_type_matrix(tmp_path: Path) -> None:
    preparation = importlib.import_module("elspeth.web.sessions.guided_payloads")
    from elspeth.web.sessions.routes.composer.guided import _load_durable_current_turn

    store = FilesystemPayloadStore(tmp_path / "wrong-turn-matrix")
    payload = _empty_wire_payload()
    prepared = preparation.prepare_guided_json_payload(store, purpose="turn", payload=payload)
    guided = replace(
        GuidedSession.initial(),
        history=(
            TurnRecord(
                step=GuidedStep.STEP_1_SOURCE,
                turn_type=TurnType.CONFIRM_WIRING,
                payload_hash=prepared.payload_id,
                response_hash=None,
                emitter="server",
            ),
        ),
    )

    with pytest.raises(AuditIntegrityError, match="current-schema turn"):
        _load_durable_current_turn(guided, payload_store=store)


@pytest.mark.parametrize(
    ("proposal_id", "draft_hash"),
    [
        ("00000000-0000-4000-8000-000000000002", _GUIDED_PROPOSAL_DRAFT_HASH),
        (str(_GUIDED_PROPOSAL_ID), "e" * 64),
    ],
)
def test_durable_confirm_wiring_requires_exact_active_proposal_binding(
    tmp_path: Path,
    proposal_id: str,
    draft_hash: str,
) -> None:
    preparation = importlib.import_module("elspeth.web.sessions.guided_payloads")
    from elspeth.web.sessions.routes.composer.guided import _load_durable_current_turn

    store = FilesystemPayloadStore(tmp_path / f"wrong-proposal-binding-{proposal_id[-1]}-{draft_hash[0]}")
    prepared = preparation.prepare_guided_json_payload(
        store,
        purpose="turn",
        payload=_empty_wire_payload(proposal_id=proposal_id, draft_hash=draft_hash),
    )
    guided = replace(
        GuidedSession.initial(),
        step=GuidedStep.STEP_4_WIRE,
        history=(
            TurnRecord(
                step=GuidedStep.STEP_4_WIRE,
                turn_type=TurnType.CONFIRM_WIRING,
                payload_hash=prepared.payload_id,
                response_hash=None,
                emitter="server",
            ),
        ),
        active_proposal=_guided_proposal_ref(),
    )

    with pytest.raises(AuditIntegrityError, match="active proposal"):
        _load_durable_current_turn(guided, payload_store=store)


@pytest.mark.parametrize(("step", "turn_type", "payload"), _MALFORMED_CURRENT_TURNS)
def test_durable_current_turn_rejects_recursively_malformed_payload(
    tmp_path: Path,
    step: GuidedStep,
    turn_type: TurnType,
    payload: Mapping[str, object],
) -> None:
    preparation = importlib.import_module("elspeth.web.sessions.guided_payloads")
    from elspeth.web.sessions.routes.composer.guided import _load_durable_current_turn

    store = FilesystemPayloadStore(tmp_path / f"invalid-durable-{turn_type.value}")
    prepared = preparation.prepare_guided_json_payload(store, purpose="turn", payload=payload)
    guided = replace(
        GuidedSession.initial(),
        step=step,
        history=(
            TurnRecord(
                step=step,
                turn_type=turn_type,
                payload_hash=prepared.payload_id,
                response_hash=None,
                emitter="server",
            ),
        ),
    )

    with pytest.raises(AuditIntegrityError, match="current-schema turn"):
        _load_durable_current_turn(guided, payload_store=store)


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

    assert protocol.PreparedGuidedJsonPayload is not None
    assert protocol.PreparedGuidedAuditRow is not None
    assert protocol.GuidedResponseDescriptor is not None
    assert protocol.GuidedStateOperationCommand is not None
    assert protocol.GuidedStateOperationSettlement is not None


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
        payload_id=guided_json_payload_id("turn", {"value": "first"}),
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


def test_guided_turn_token_binds_the_persisted_history_occurrence() -> None:
    replay = importlib.import_module("elspeth.web.sessions.guided_replay")
    payload_id = guided_json_payload_id("turn", {"question": "Choose"})
    unanswered = TurnRecord(
        step=GuidedStep.STEP_1_SOURCE,
        turn_type=TurnType.SINGLE_SELECT,
        payload_hash=payload_id,
        response_hash=None,
        emitter="server",
    )
    first = replace(GuidedSession.initial(), history=(unanswered,))
    repeated = replace(
        first,
        history=(replace(unanswered, response_hash="b" * 64), unanswered),
    )

    first_token = replay.guided_turn_token(first)
    repeated_token = replay.guided_turn_token(repeated)

    assert len(first_token) == 64
    assert first_token != repeated_token


@pytest.mark.parametrize("invalid", ["no_history", "answered", "stale_step"])
def test_guided_turn_token_requires_the_final_current_unanswered_record(invalid: str) -> None:
    replay = importlib.import_module("elspeth.web.sessions.guided_replay")
    guided = GuidedSession.initial()
    if invalid != "no_history":
        record = TurnRecord(
            step=GuidedStep.STEP_1_SOURCE,
            turn_type=TurnType.SINGLE_SELECT,
            payload_hash=guided_json_payload_id("turn", {"question": "Choose"}),
            response_hash="b" * 64 if invalid == "answered" else None,
            emitter="server",
        )
        guided = replace(
            guided,
            history=(record,),
            step=GuidedStep.STEP_2_SINK if invalid == "stale_step" else guided.step,
        )

    with pytest.raises(AuditIntegrityError):
        replay.guided_turn_token(guided)


def test_guided_turn_token_survives_checkpoint_round_trip() -> None:
    replay = importlib.import_module("elspeth.web.sessions.guided_replay")
    guided = _guided_with_next_turn(guided_json_payload_id("turn", {"question": "Choose"}))

    assert replay.guided_turn_token(GuidedSession.from_dict(guided.to_dict())) == replay.guided_turn_token(guided)


def test_load_guided_json_payload_revalidates_durable_canonical_bytes(tmp_path: Path) -> None:
    preparation = importlib.import_module("elspeth.web.sessions.guided_payloads")
    replay = importlib.import_module("elspeth.web.sessions.guided_replay")
    store = FilesystemPayloadStore(tmp_path / "guided-payloads")
    prepared = preparation.prepare_guided_json_payload(
        store,
        purpose="turn",
        payload={"question": "Choose", "options": ["a", "b"]},
    )

    loaded = replay.load_guided_json_payload(
        store,
        payload_id=prepared.payload_id,
        purpose="turn",
    )

    assert loaded == prepared


def test_load_guided_json_payload_rejects_valid_but_wrong_durable_purpose(tmp_path: Path) -> None:
    preparation = importlib.import_module("elspeth.web.sessions.guided_payloads")
    replay = importlib.import_module("elspeth.web.sessions.guided_replay")
    store = FilesystemPayloadStore(tmp_path / "guided-payload-purpose")
    prepared = preparation.prepare_guided_json_payload(
        store,
        purpose="turn_response",
        payload={"chosen": ["csv"]},
    )

    with pytest.raises(AuditIntegrityError, match="purpose"):
        replay.load_guided_json_payload(
            store,
            payload_id=prepared.payload_id,
            purpose="turn",
        )


@pytest.mark.parametrize("failure", ["missing", "noncanonical", "malformed", "invalid_purpose"])
def test_load_guided_json_payload_fails_closed_for_invalid_durable_content(tmp_path: Path, failure: str) -> None:
    replay = importlib.import_module("elspeth.web.sessions.guided_replay")
    store = FilesystemPayloadStore(tmp_path / f"guided-payloads-{failure}")
    payload_id = "a" * 64
    purpose = "turn"
    if failure == "noncanonical":
        payload_id = store.store(b'{"question": "Choose"}')
    elif failure == "malformed":
        payload_id = store.store(b'{"question":')
    elif failure == "invalid_purpose":
        payload_id = store.store(b'{"question":"Choose"}')
        purpose = "diagnostic"

    with pytest.raises(AuditIntegrityError):
        replay.load_guided_json_payload(
            store,
            payload_id=payload_id,
            purpose=purpose,
        )


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
        payload_id=guided_json_payload_id("turn_response", {"question": "Choose"}),
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


def test_guided_checkpoint_replaces_raw_validation_text_with_closed_status() -> None:
    replay = importlib.import_module("elspeth.web.sessions.guided_replay")
    state = CompositionStateData(
        is_valid=False,
        validation_errors=["/home/operator/private.csv token=VALIDATION-CREDENTIAL-CANARY"],
    )

    prepared = replay.with_guided_response_descriptor(
        state,
        GuidedResponseDescriptor(kind="guided_respond", next_turn=None, assistant_turn_seq=None),
    )

    assert prepared.validation_errors == ("guided_composition_invalid",)
    assert "VALIDATION-CREDENTIAL-CANARY" not in repr(prepared.validation_errors)


def test_guided_checkpoint_preserves_closed_validation_status() -> None:
    replay = importlib.import_module("elspeth.web.sessions.guided_replay")
    state = CompositionStateData(
        is_valid=False,
        validation_errors=["guided_composition_invalid"],
    )

    prepared = replay.with_guided_response_descriptor(
        state,
        GuidedResponseDescriptor(kind="guided_respond", next_turn=None, assistant_turn_seq=None),
    )

    assert prepared.validation_errors == ("guided_composition_invalid",)


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


def test_intent_cancellation_is_allowlisted_only_for_the_exact_structural_schema() -> None:
    preparation = importlib.import_module("elspeth.web.sessions.guided_audit")
    invocation, _llm_call, _chat_turn = _audit_evidence()
    payload = {
        "intent_id": "00000000-0000-4000-8000-000000000801",
        "receiving_stage": "source",
        "target_stage": "topology",
    }
    authentic = replace(
        invocation,
        tool_call_id="intent-cancel-authentic",
        tool_name="guided_intent_cancelled",
        arguments_canonical=canonical_json(payload),
        arguments_hash=stable_hash(payload),
    )
    malformed_payload = {**payload, "summary": "PRIVATE-CANCELLATION-CANARY"}
    malformed = replace(
        authentic,
        tool_call_id="intent-cancel-malformed",
        arguments_canonical=canonical_json(malformed_payload),
        arguments_hash=stable_hash(malformed_payload),
    )

    authentic_row, malformed_row = preparation.prepare_guided_audit_rows(
        invocations=(authentic, malformed),
        llm_calls=(),
        chat_turns=(),
    )

    authentic_envelope = deep_thaw(authentic_row.envelope["invocation"])
    assert json.loads(authentic_envelope["arguments_canonical"]) == payload
    malformed_persisted = repr(deep_thaw({"content": malformed_row.content, "envelope": malformed_row.envelope}))
    assert "PRIVATE-CANCELLATION-CANARY" not in malformed_persisted
    assert "unmanifested_success_payload_omitted" in malformed_persisted


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
        validation_errors=state.validation_errors,
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
        payload_id=guided_json_payload_id("turn", {"question": "Choose"}),
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


def test_replay_rejects_turn_payload_with_extra_key() -> None:
    replay = importlib.import_module("elspeth.web.sessions.guided_replay")
    payload_data = {
        "question": "Choose a source",
        "options": [],
        "allow_custom": False,
        "credential_canary": "MUST-NOT-REPLAY",
    }
    payload = PreparedGuidedJsonPayload(
        payload_id=guided_json_payload_id("turn", payload_data),
        purpose="turn",
        payload=payload_data,
    )
    guided = _guided_with_next_turn(payload.payload_id)
    descriptor = GuidedResponseDescriptor(
        kind="guided_respond",
        next_turn=GuidedReplayTurn(
            turn_type=TurnType.SINGLE_SELECT,
            step_index=0,
            payload_id=payload.payload_id,
        ),
        assistant_turn_seq=None,
    )

    with pytest.raises(AuditIntegrityError, match="current-schema turn"):
        replay.project_guided_response(_replay_record(descriptor=descriptor, guided=guided), payloads=(payload,))


@pytest.mark.parametrize(("step", "turn_type", "payload_data"), _MALFORMED_CURRENT_TURNS)
def test_replay_rejects_recursively_malformed_current_turn(
    step: GuidedStep,
    turn_type: TurnType,
    payload_data: Mapping[str, object],
) -> None:
    replay = importlib.import_module("elspeth.web.sessions.guided_replay")
    payload = PreparedGuidedJsonPayload(
        payload_id=guided_json_payload_id("turn", payload_data),
        purpose="turn",
        payload=payload_data,
    )
    guided = replace(
        GuidedSession.initial(),
        step=step,
        history=(
            TurnRecord(
                step=step,
                turn_type=turn_type,
                payload_hash=payload.payload_id,
                response_hash=None,
                emitter="server",
            ),
        ),
    )
    descriptor = GuidedResponseDescriptor(
        kind="guided_respond",
        next_turn=GuidedReplayTurn(
            turn_type=turn_type,
            step_index=0,
            payload_id=payload.payload_id,
        ),
        assistant_turn_seq=None,
    )

    with pytest.raises(AuditIntegrityError, match="current-schema turn"):
        replay.project_guided_response(
            _replay_record(descriptor=descriptor, guided=guided),
            payloads=(payload,),
        )


@pytest.mark.parametrize("response_kind", ["guided_respond", "guided_chat"])
@pytest.mark.parametrize("tamper", ["answered", "stale_stage"])
def test_replay_requires_final_turn_to_be_current_and_unanswered(response_kind: str, tamper: str) -> None:
    replay = importlib.import_module("elspeth.web.sessions.guided_replay")
    payload = PreparedGuidedJsonPayload(
        payload_id=guided_json_payload_id(
            "turn",
            {
                "question": "Choose a source",
                "options": [{"id": "csv", "label": "CSV", "hint": None}],
                "allow_custom": True,
            },
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


async def _seed_guided_predecessor_with_intents(
    service: SessionServiceImpl,
    session_id,
    intents: tuple[DeferredStageIntent, ...],
) -> CompositionStateRecord:
    return await service.save_composition_state(
        session_id,
        CompositionStateData(
            metadata_={"name": "Guided predecessor", "description": ""},
            is_valid=False,
            composer_meta={"guided_session": replace(GuidedSession.initial(), deferred_intents=intents).to_dict()},
        ),
        provenance="convergence_persist",
    )


def _deferred_intent(
    *,
    intent_id=None,
    originating_message_id=None,
    message_content: str = "private deferred instruction",
    summary: str = "Future topology transform requirement.",
) -> DeferredStageIntent:
    return DeferredStageIntent.create(
        intent_id=str(intent_id or uuid4()),
        receiving_stage="source",
        target_stage="topology",
        catalog_kind="transform",
        catalog_name="passthrough",
        redacted_summary=summary,
        originating_message_id=str(originating_message_id or uuid4()),
        message_content_hash=stable_hash(message_content),
        constraints=(
            ComponentCountConstraint(
                kind="component_count",
                component_kind="node",
                plugin_kind="transform",
                plugin_name="passthrough",
                operator="at_least",
                count=1,
            ),
        ),
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
@pytest.mark.parametrize(
    "mutation",
    ["omitted_addition", "replacement_same_id", "reorder", "removal", "extra_append", "mismatched_sideband"],
)
async def test_deferred_intent_delta_rejects_every_untyped_or_non_append_mutation_before_cohort_write(
    service_and_engine,
    mutation: str,
) -> None:
    service, engine = service_and_engine
    session_id = (await service.create_session("alice", f"guided deferred delta {mutation}", "local")).id
    first = _deferred_intent()
    second = _deferred_intent()
    prior = () if mutation == "omitted_addition" else (first, second) if mutation in {"reorder", "removal"} else (first,)
    predecessor = await _seed_guided_predecessor_with_intents(service, session_id, prior)
    claimed = await service.reserve_guided_operation(
        session_id=session_id,
        operation_id=f"respond-deferred-delta-{mutation}",
        kind="guided_respond",
        request_hash="a" * 64,
        actor="worker",
        lease_seconds=60,
    )
    assert isinstance(claimed, GuidedOperationClaimed)
    command = _present_respond_command(claimed.fence, predecessor)
    originating = command.originating_message
    assert originating is not None
    appended = _deferred_intent(
        originating_message_id=originating.message_id,
        message_content=originating.content,
    )
    if mutation == "omitted_addition":
        candidate = (appended,)
        sideband = None
    elif mutation == "replacement_same_id":
        candidate = (
            _deferred_intent(
                intent_id=first.intent_id,
                originating_message_id=originating.message_id,
                message_content=originating.content,
                summary="Changed existing intent under the same id.",
            ),
        )
        sideband = UUID(first.intent_id)
    elif mutation == "reorder":
        candidate = (second, first)
        sideband = None
    elif mutation == "removal":
        candidate = (first,)
        sideband = None
    elif mutation == "extra_append":
        candidate = (first, appended, _deferred_intent())
        sideband = UUID(appended.intent_id)
    else:
        candidate = (first, appended)
        sideband = uuid4()
    command = replace(
        command,
        state=replace(
            command.state,
            composer_meta={"guided_session": replace(GuidedSession.initial(), deferred_intents=candidate).to_dict()},
        ),
        retained_deferred_intent_id=sideband,
    )
    with engine.connect() as connection:
        events_before = connection.execute(
            select(guided_operation_events_table).where(guided_operation_events_table.c.session_id == str(session_id))
        ).all()

    with pytest.raises(AuditIntegrityError, match="deferred intent"):
        await service.settle_guided_state_operation(command)

    with engine.connect() as connection:
        states = connection.execute(
            select(composition_states_table)
            .where(composition_states_table.c.session_id == str(session_id))
            .order_by(composition_states_table.c.version)
        ).all()
        messages = connection.execute(select(chat_messages_table).where(chat_messages_table.c.session_id == str(session_id))).all()
        operation = connection.execute(select(guided_operations_table).where(guided_operations_table.c.session_id == str(session_id))).one()
        events_after = connection.execute(
            select(guided_operation_events_table).where(guided_operation_events_table.c.session_id == str(session_id))
        ).all()
    assert [row.id for row in states] == [str(predecessor.id)]
    assert messages == []
    assert operation.status == "in_progress"
    assert operation.originating_message_id is None
    assert operation.result_state_id is None
    assert operation.response_hash is None
    assert events_after == events_before


@pytest.mark.asyncio
async def test_deferred_intent_delta_allows_exact_unchanged_candidate(service_and_engine) -> None:
    service, _engine = service_and_engine
    session_id = (await service.create_session("alice", "guided deferred unchanged", "local")).id
    prior = (_deferred_intent(),)
    predecessor = await _seed_guided_predecessor_with_intents(service, session_id, prior)
    claimed = await service.reserve_guided_operation(
        session_id=session_id,
        operation_id="respond-deferred-unchanged",
        kind="guided_respond",
        request_hash="a" * 64,
        actor="worker",
        lease_seconds=60,
    )
    assert isinstance(claimed, GuidedOperationClaimed)
    command = _present_respond_command(claimed.fence, predecessor)
    command = replace(
        command,
        state=replace(
            command.state,
            composer_meta={"guided_session": replace(GuidedSession.initial(), deferred_intents=prior).to_dict()},
        ),
    )

    settlement = await service.settle_guided_state_operation(command)

    assert state_from_record(settlement.result_state).guided_session.deferred_intents == prior  # type: ignore[union-attr]


@pytest.mark.asyncio
async def test_deferred_intent_delta_allows_one_typed_terminal_append(service_and_engine) -> None:
    service, _engine = service_and_engine
    session_id = (await service.create_session("alice", "guided deferred append", "local")).id
    prior = (_deferred_intent(),)
    predecessor = await _seed_guided_predecessor_with_intents(service, session_id, prior)
    claimed = await service.reserve_guided_operation(
        session_id=session_id,
        operation_id="respond-deferred-append",
        kind="guided_respond",
        request_hash="a" * 64,
        actor="worker",
        lease_seconds=60,
    )
    assert isinstance(claimed, GuidedOperationClaimed)
    command = _present_respond_command(claimed.fence, predecessor)
    originating = command.originating_message
    assert originating is not None
    appended = _deferred_intent(
        originating_message_id=originating.message_id,
        message_content=originating.content,
    )
    candidate = (*prior, appended)
    command = replace(
        command,
        state=replace(
            command.state,
            composer_meta={"guided_session": replace(GuidedSession.initial(), deferred_intents=candidate).to_dict()},
        ),
        retained_deferred_intent_id=UUID(appended.intent_id),
    )

    settlement = await service.settle_guided_state_operation(command)

    result_guided = state_from_record(settlement.result_state).guided_session
    assert result_guided is not None
    assert result_guided.deferred_intents == candidate


@pytest.mark.asyncio
async def test_deferred_intent_delta_allows_one_typed_exact_cancellation_with_custody_and_audit(
    service_and_engine,
) -> None:
    service, engine = service_and_engine
    session_id = (await service.create_session("alice", "guided deferred cancel", "local")).id
    origin = await service.add_message(
        session_id,
        "user",
        "private deferred instruction",
        writer_principal="route_user_message",
    )
    cancelled = _deferred_intent(originating_message_id=origin.id)
    preserved_origin = await service.add_message(
        session_id,
        "user",
        "another private instruction",
        writer_principal="route_user_message",
    )
    preserved = _deferred_intent(
        originating_message_id=preserved_origin.id,
        message_content="another private instruction",
    )
    predecessor = await _seed_guided_predecessor_with_intents(service, session_id, (cancelled, preserved))
    claimed = await service.reserve_guided_operation(
        session_id=session_id,
        operation_id="respond-deferred-cancel",
        kind="guided_respond",
        request_hash="a" * 64,
        actor="worker",
        lease_seconds=60,
    )
    assert isinstance(claimed, GuidedOperationClaimed)
    command = _present_respond_command(claimed.fence, predecessor)
    recorder = BufferingRecorder()
    emit_intent_cancelled(
        recorder,
        intent=cancelled,
        composition_version=1,
        actor="alice",
    )
    command = replace(
        command,
        state=replace(
            command.state,
            composer_meta={"guided_session": replace(GuidedSession.initial(), deferred_intents=(preserved,)).to_dict()},
        ),
        audit_evidence=GuidedAuditEvidence(invocations=recorder.invocations),
        originating_message=GuidedOriginatingUserMessageDraft(
            message_id=uuid4(),
            content=f"cancel exact intent {cancelled.intent_id}",
        ),
        deferred_intent_action=DeferredIntentCancelAction(
            intent_id=cancelled.intent_id,
            selection_token=deferred_intent_management_option(cancelled).selection_token,
        ),
    )

    settlement = await service.settle_guided_state_operation(command)

    result_guided = state_from_record(settlement.result_state).guided_session
    assert result_guided is not None
    assert result_guided.deferred_intents == (preserved,)
    with engine.connect() as connection:
        audit_rows = connection.execute(
            select(chat_messages_table)
            .where(chat_messages_table.c.session_id == str(session_id))
            .where(chat_messages_table.c.role == "audit")
        ).all()
    assert sum("guided_intent_cancelled" in repr(row.tool_calls) for row in audit_rows) == 1


@pytest.mark.parametrize(
    "private_request",
    [
        "cancel the first saved instruction",
        "cancel exact intent {preserved_id}",
        "compare {cancelled_id} with {preserved_id}, then cancel one",
    ],
    ids=["zero-current-uuids", "wrong-current-uuid", "multiple-current-uuids"],
)
@pytest.mark.asyncio
async def test_settlement_rechecks_plural_intent_user_authority_and_rolls_back(
    service_and_engine,
    private_request: str,
) -> None:
    service, engine = service_and_engine
    session_id = (await service.create_session("alice", "guided plural cancellation authority", "local")).id
    cancelled_origin = await service.add_message(
        session_id,
        "user",
        "private deferred instruction",
        writer_principal="route_user_message",
    )
    preserved_origin = await service.add_message(
        session_id,
        "user",
        "another private instruction",
        writer_principal="route_user_message",
    )
    cancelled = _deferred_intent(originating_message_id=cancelled_origin.id)
    preserved = _deferred_intent(
        originating_message_id=preserved_origin.id,
        message_content="another private instruction",
    )
    predecessor = await _seed_guided_predecessor_with_intents(service, session_id, (cancelled, preserved))
    operation_id = f"respond-plural-authority-{uuid4()}"
    claimed = await service.reserve_guided_operation(
        session_id=session_id,
        operation_id=operation_id,
        kind="guided_respond",
        request_hash="b" * 64,
        actor="worker",
        lease_seconds=60,
    )
    assert isinstance(claimed, GuidedOperationClaimed)
    recorder = BufferingRecorder()
    emit_intent_cancelled(recorder, intent=cancelled, composition_version=1, actor="alice")
    command = _present_respond_command(claimed.fence, predecessor)
    command = replace(
        command,
        state=replace(
            command.state,
            composer_meta={"guided_session": replace(GuidedSession.initial(), deferred_intents=(preserved,)).to_dict()},
        ),
        audit_evidence=GuidedAuditEvidence(invocations=recorder.invocations),
        originating_message=GuidedOriginatingUserMessageDraft(
            message_id=uuid4(),
            content=private_request.format(cancelled_id=cancelled.intent_id, preserved_id=preserved.intent_id),
        ),
        deferred_intent_action=DeferredIntentCancelAction(
            intent_id=cancelled.intent_id,
            selection_token=deferred_intent_management_option(cancelled).selection_token,
        ),
    )

    with pytest.raises(AuditIntegrityError, match="one matching UUID"):
        await service.settle_guided_state_operation(command)

    with engine.connect() as connection:
        operation = (
            connection.execute(select(guided_operations_table).where(guided_operations_table.c.operation_id == operation_id))
            .mappings()
            .one()
        )
    assert operation["status"] == "in_progress"
    assert operation["result_state_id"] is None


@pytest.mark.parametrize(
    "corruption",
    ["arg_error", "malformed", "redacted", "hash", "result", "version", "selection_token"],
)
@pytest.mark.asyncio
async def test_deferred_intent_cancellation_rejects_inauthentic_audit_or_binding_and_rolls_back(
    service_and_engine,
    corruption: str,
) -> None:
    service, engine = service_and_engine
    session_id = (await service.create_session("alice", "guided cancellation audit integrity", "local")).id
    origin = await service.add_message(
        session_id,
        "user",
        "private deferred instruction",
        writer_principal="route_user_message",
    )
    cancelled = _deferred_intent(originating_message_id=origin.id)
    predecessor = await _seed_guided_predecessor_with_intents(service, session_id, (cancelled,))
    operation_id = f"respond-cancel-corrupt-{corruption}"
    claimed = await service.reserve_guided_operation(
        session_id=session_id,
        operation_id=operation_id,
        kind="guided_respond",
        request_hash="c" * 64,
        actor="worker",
        lease_seconds=60,
    )
    assert isinstance(claimed, GuidedOperationClaimed)
    recorder = BufferingRecorder()
    emit_intent_cancelled(recorder, intent=cancelled, composition_version=1, actor="alice")
    (authentic,) = recorder.invocations
    if corruption == "arg_error":
        corrupted = replace(authentic, status=ComposerToolStatus.ARG_ERROR, error_class="ToolArgumentError", version_after=None)
    elif corruption == "malformed":
        corrupted = replace(authentic, arguments_canonical="{")
    elif corruption == "redacted":
        redacted = {"_redaction_status": "guided_failure_payload_omitted"}
        corrupted = replace(authentic, arguments_canonical=canonical_json(redacted), arguments_hash=stable_hash(redacted))
    elif corruption == "hash":
        corrupted = replace(authentic, arguments_hash="0" * 64)
    elif corruption == "result":
        corrupted = replace(authentic, result_canonical="{}", result_hash=stable_hash({}))
    elif corruption == "version":
        corrupted = replace(authentic, version_after=2)
    else:
        corrupted = authentic
    command = _present_respond_command(claimed.fence, predecessor)
    command = replace(
        command,
        state=replace(command.state, composer_meta={"guided_session": GuidedSession.initial().to_dict()}),
        audit_evidence=GuidedAuditEvidence(invocations=(corrupted,)),
        deferred_intent_action=DeferredIntentCancelAction(
            intent_id=cancelled.intent_id,
            selection_token=(
                "wrong-server-selection-token"
                if corruption == "selection_token"
                else deferred_intent_management_option(cancelled).selection_token
            ),
        ),
    )
    with engine.connect() as connection:
        before_states = connection.execute(
            select(func.count()).select_from(composition_states_table).where(composition_states_table.c.session_id == str(session_id))
        ).scalar_one()
        before_messages = connection.execute(
            select(func.count()).select_from(chat_messages_table).where(chat_messages_table.c.session_id == str(session_id))
        ).scalar_one()

    with pytest.raises(AuditIntegrityError):
        await service.settle_guided_state_operation(command)

    with engine.connect() as connection:
        after_states = connection.execute(
            select(func.count()).select_from(composition_states_table).where(composition_states_table.c.session_id == str(session_id))
        ).scalar_one()
        after_messages = connection.execute(
            select(func.count()).select_from(chat_messages_table).where(chat_messages_table.c.session_id == str(session_id))
        ).scalar_one()
        operation = (
            connection.execute(select(guided_operations_table).where(guided_operations_table.c.operation_id == operation_id))
            .mappings()
            .one()
        )
    assert (after_states, after_messages) == (before_states, before_messages)
    assert operation["status"] == "in_progress"
    assert operation["result_state_id"] is None


@pytest.mark.parametrize("non_cancel_sideband", ["unchanged", "append", "edit"])
def test_cancellation_audit_is_forbidden_without_exact_cancel_sideband(non_cancel_sideband: str) -> None:
    from elspeth.web.sessions.service import _verify_guided_deferred_intent_mutation

    intent = _deferred_intent()
    recorder = BufferingRecorder()
    emit_intent_cancelled(recorder, intent=intent, composition_version=1, actor="alice")
    command = _empty_respond_command(
        GuidedOperationFence(
            session_id=uuid4(),
            operation_id="non-cancel-sideband",
            lease_token="lease",
            attempt=1,
        )
    )
    command = replace(
        command,
        originating_message=GuidedOriginatingUserMessageDraft(message_id=uuid4(), content="private non-cancel request"),
    )
    if non_cancel_sideband == "append":
        command = replace(command, retained_deferred_intent_id=uuid4())
    elif non_cancel_sideband == "edit":
        command = replace(
            command,
            deferred_intent_action=DeferredIntentEditAction(
                intent_id=intent.intent_id,
                selection_token=deferred_intent_management_option(intent).selection_token,
                replacement=DeferredIntentAction(
                    target_stage=intent.target_stage,
                    catalog_kind=intent.catalog_kind,
                    catalog_name=intent.catalog_name,
                    redacted_summary=intent.redacted_summary,
                    constraints=intent.constraints,
                ),
            ),
        )
    command = replace(command, audit_evidence=GuidedAuditEvidence(invocations=recorder.invocations))

    with pytest.raises(AuditIntegrityError, match="if and only if"):
        _verify_guided_deferred_intent_mutation(
            cast(Any, None),
            session_id=str(command.fence.session_id),
            command=command,
            prior_guided=GuidedSession.initial(),
            candidate_guided=GuidedSession.initial(),
        )


@pytest.mark.asyncio
async def test_deferred_intent_delta_allows_stable_id_edit_with_new_private_message_custody(
    service_and_engine,
) -> None:
    service, _engine = service_and_engine
    session_id = (await service.create_session("alice", "guided deferred edit", "local")).id
    origin = await service.add_message(
        session_id,
        "user",
        "private deferred instruction",
        writer_principal="route_user_message",
    )
    existing = _deferred_intent(originating_message_id=origin.id)
    predecessor = await _seed_guided_predecessor_with_intents(service, session_id, (existing,))
    claimed = await service.reserve_guided_operation(
        session_id=session_id,
        operation_id="respond-deferred-edit",
        kind="guided_respond",
        request_hash="b" * 64,
        actor="worker",
        lease_seconds=60,
    )
    assert isinstance(claimed, GuidedOperationClaimed)
    command = _present_respond_command(claimed.fence, predecessor)
    replacement_action = DeferredIntentAction(
        target_stage="topology",
        catalog_kind="transform",
        catalog_name="passthrough",
        redacted_summary="model text is not durable authority",
        constraints=existing.constraints,
    )
    private_revision = GuidedOriginatingUserMessageDraft(message_id=uuid4(), content="private revised instruction")
    from elspeth.web.composer.guided.deferred_intents import create_deferred_stage_intent

    replacement = create_deferred_stage_intent(
        replacement_action,
        receiving_stage=existing.receiving_stage,
        intent_id=existing.intent_id,
        originating_message_id=str(private_revision.message_id),
        originating_message_content=private_revision.content,
    )
    command = replace(
        command,
        state=replace(
            command.state,
            composer_meta={"guided_session": replace(GuidedSession.initial(), deferred_intents=(replacement,)).to_dict()},
        ),
        originating_message=private_revision,
        deferred_intent_action=DeferredIntentEditAction(
            intent_id=existing.intent_id,
            selection_token=deferred_intent_management_option(existing).selection_token,
            replacement=replacement_action,
        ),
    )

    settlement = await service.settle_guided_state_operation(command)

    result_guided = state_from_record(settlement.result_state).guided_session
    assert result_guided is not None
    assert result_guided.deferred_intents == (replacement,)
    assert settlement.originating_message is not None
    assert settlement.originating_message.content == private_revision.content


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

    expected_error = AuditIntegrityError if mismatch == "content_hash" else GuidedOperationSettlementConflictError
    with pytest.raises(expected_error):
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
