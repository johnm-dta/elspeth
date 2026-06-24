"""Edge-case validation tests for session API schemas."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from elspeth.web.execution.schemas import (
    RunAccounting,
    RunAccountingIntegrity,
    RunAccountingRouting,
    RunAccountingSource,
    RunAccountingTokens,
)
from elspeth.web.sessions.schemas import (
    ChatMessageResponse,
    CompositionStateResponse,
    CreateSessionRequest,
    ForkSessionRequest,
    ForkSessionResponse,
    MessageWithStateResponse,
    RevertStateRequest,
    RunResponse,
    SendMessageRequest,
    SessionResponse,
    ValidationEntryResponse,
)


def _accounting() -> RunAccounting:
    return RunAccounting(
        source=RunAccountingSource(rows_processed=10),
        tokens=RunAccountingTokens(
            emitted=10,
            terminal=10,
            succeeded=10,
            failed=0,
            structural=0,
            pending=0,
        ),
        routing=RunAccountingRouting(
            routed_success=0,
            routed_failure=0,
            quarantined=0,
            discarded=0,
        ),
        integrity=RunAccountingIntegrity(
            closure="closed",
            missing_terminal_outcomes=0,
            duplicate_terminal_outcomes=0,
        ),
    )


class TestCreateSessionRequest:
    def test_default_title(self) -> None:
        req = CreateSessionRequest()
        assert req.title == "New session"

    def test_custom_title(self) -> None:
        req = CreateSessionRequest(title="My pipeline")
        assert req.title == "My pipeline"

    @pytest.mark.parametrize("title", ["", "   ", "\u200b", "\ufeff"])
    def test_rejects_blank_or_invisible_title(self, title: str) -> None:
        with pytest.raises(ValidationError):
            CreateSessionRequest(title=title)


class TestSendMessageRequest:
    def test_rejects_empty_content(self) -> None:
        with pytest.raises(ValidationError, match="content"):
            SendMessageRequest(content="")

    @pytest.mark.parametrize("content", ["   ", "\u200b", "\ufeff"])
    def test_rejects_blank_or_invisible_only_content(self, content: str) -> None:
        with pytest.raises(ValidationError):
            SendMessageRequest(content=content)

    def test_accepts_nonempty_content(self) -> None:
        req = SendMessageRequest(content="hello")
        assert req.content == "hello"

    def test_rejects_invalid_state_id(self) -> None:
        with pytest.raises(ValidationError):
            SendMessageRequest(content="hello", state_id="not-a-uuid")

    def test_accepts_valid_uuid_state_id(self) -> None:
        import uuid

        sid = uuid.uuid4()
        req = SendMessageRequest(content="hello", state_id=sid)
        assert req.state_id == sid

    def test_accepts_string_uuid_state_id(self) -> None:
        req = SendMessageRequest(
            content="hello",
            state_id="550e8400-e29b-41d4-a716-446655440000",
        )
        assert str(req.state_id) == "550e8400-e29b-41d4-a716-446655440000"

    def test_accepts_none_state_id(self) -> None:
        req = SendMessageRequest(content="hello", state_id=None)
        assert req.state_id is None

    def test_accepts_content_at_max_length(self) -> None:
        # Phase 5b.0.5 (F-3): max_length=65536 cap on chat message content.
        # Exact-boundary value must validate to confirm the cap is set at
        # 64 KiB rather than at an off-by-one neighbour.
        req = SendMessageRequest(content="x" * 65536)
        assert len(req.content) == 65536

    def test_rejects_content_exceeding_max_length(self) -> None:
        # Phase 5b.0.5 (F-3): one byte over the cap must raise.  Defense
        # against unbounded payload allocation before interpretation events
        # can be triggered.
        with pytest.raises(ValidationError, match="content"):
            SendMessageRequest(content="x" * 65537)


class TestForkSessionRequest:
    def test_rejects_invalid_uuid(self) -> None:
        with pytest.raises(ValidationError):
            ForkSessionRequest(from_message_id="not-a-uuid", new_message_content="fork")

    @pytest.mark.parametrize("content", ["", "   ", "\u200b", "\ufeff"])
    def test_rejects_blank_or_invisible_new_message_content(self, content: str) -> None:
        with pytest.raises(ValidationError):
            ForkSessionRequest(
                from_message_id="550e8400-e29b-41d4-a716-446655440000",
                new_message_content=content,
            )

    def test_accepts_valid_uuid(self) -> None:
        import uuid

        req = ForkSessionRequest(from_message_id=uuid.uuid4(), new_message_content="fork")
        assert req.new_message_content == "fork"


class TestRevertStateRequest:
    def test_rejects_invalid_uuid(self) -> None:
        with pytest.raises(ValidationError):
            RevertStateRequest(state_id="not-a-uuid")


class TestSessionRequestExtraFieldsRejected:
    @pytest.mark.parametrize(
        ("model_cls", "payload"),
        [
            (CreateSessionRequest, {"title": "Chat", "legacy": "x"}),
            (
                SendMessageRequest,
                {
                    "content": "hello",
                    "stateId": "550e8400-e29b-41d4-a716-446655440000",
                },
            ),
            (
                ForkSessionRequest,
                {
                    "from_message_id": "550e8400-e29b-41d4-a716-446655440000",
                    "new_message_content": "fork",
                    "extra": True,
                },
            ),
            (
                RevertStateRequest,
                {
                    "state_id": "550e8400-e29b-41d4-a716-446655440000",
                    "extra": "x",
                },
            ),
        ],
    )
    def test_request_model_rejects_unknown_keys(self, model_cls: type[object], payload: dict[str, object]) -> None:
        with pytest.raises(ValidationError, match="extra"):
            model_cls.model_validate(payload)


class TestRunResponse:
    def test_rejects_invalid_status(self) -> None:
        from datetime import UTC, datetime

        with pytest.raises(ValidationError, match="status"):
            RunResponse(
                id="run-1",
                session_id="sess-1",
                status="invalid_status",
                accounting=None,
                started_at=datetime.now(UTC),
                composition_version=1,
            )

    def test_accepts_valid_status(self) -> None:
        from datetime import UTC, datetime

        resp = RunResponse(
            id="run-1",
            session_id="sess-1",
            status="completed",
            accounting=_accounting(),
            started_at=datetime.now(UTC),
            composition_version=1,
        )
        assert resp.status == "completed"
        assert resp.accounting is not None
        assert resp.accounting.tokens.succeeded == 10


def _valid_session_response_kwargs() -> dict[str, object]:
    now = datetime.now(UTC)
    return {
        "id": "sess-1",
        "user_id": "user-1",
        "title": "Session",
        "created_at": now,
        "updated_at": now,
    }


def _valid_chat_message_kwargs() -> dict[str, object]:
    return {
        "id": "msg-1",
        "session_id": "sess-1",
        "role": "user",
        "content": "hello",
        "tool_calls": None,
        "created_at": datetime.now(UTC),
    }


def _valid_composition_state_kwargs() -> dict[str, object]:
    return {
        "id": "state-1",
        "session_id": "sess-1",
        "version": 1,
        "sources": None,
        "nodes": None,
        "edges": None,
        "outputs": None,
        "metadata": None,
        "is_valid": True,
        "validation_errors": None,
        "validation_warnings": None,
        "validation_suggestions": None,
        "derived_from_state_id": None,
        "created_at": datetime.now(UTC),
    }


def _valid_run_response_kwargs() -> dict[str, object]:
    return {
        "id": "run-1",
        "session_id": "sess-1",
        "status": "completed",
        "accounting": _accounting(),
        "started_at": datetime.now(UTC),
        "composition_version": 1,
    }


# ── Tier 1 strictness regression tests ───────────────────────────────
#
# Session response models serialize system-owned data (Tier 1).  Silent
# coercion or dropped extras on the way out would let a backend bug
# (wrong type emitted by the service layer, or a stale field lingering
# from a refactor) flow into the audit-visible API surface without
# complaint.  Mirror the execution/schemas.py strictness contract.


class TestSessionStrictCoercionRejected:
    """String-to-int, string-to-bool, and string-to-datetime coercion must crash."""

    def test_session_response_rejects_iso_string_datetime(self) -> None:
        kwargs = _valid_session_response_kwargs()
        kwargs["created_at"] = "2026-04-15T10:00:00+00:00"
        with pytest.raises(ValidationError):
            SessionResponse(**kwargs)  # type: ignore[arg-type]

    def test_chat_message_response_rejects_iso_string_datetime(self) -> None:
        kwargs = _valid_chat_message_kwargs()
        kwargs["created_at"] = "2026-04-15T10:00:00+00:00"
        with pytest.raises(ValidationError):
            ChatMessageResponse(**kwargs)  # type: ignore[arg-type]

    def test_chat_message_response_accepts_tool_call_array(self) -> None:
        kwargs = _valid_chat_message_kwargs()
        kwargs["tool_calls"] = [
            {
                "id": "call-1",
                "type": "function",
                "function": {
                    "name": "set_source",
                    "arguments": '{"type":"csv"}',
                },
            }
        ]
        resp = ChatMessageResponse(**kwargs)  # type: ignore[arg-type]
        assert resp.tool_calls is not None
        assert resp.tool_calls[0]["id"] == "call-1"

    def test_chat_message_response_rejects_non_array_tool_calls(self) -> None:
        kwargs = _valid_chat_message_kwargs()
        kwargs["tool_calls"] = {"name": "set_source"}
        with pytest.raises(ValidationError):
            ChatMessageResponse(**kwargs)  # type: ignore[arg-type]

    def test_composition_state_response_rejects_string_int_version(self) -> None:
        kwargs = _valid_composition_state_kwargs()
        kwargs["version"] = "1"
        with pytest.raises(ValidationError):
            CompositionStateResponse(**kwargs)  # type: ignore[arg-type]

    def test_composition_state_response_rejects_string_bool_is_valid(self) -> None:
        kwargs = _valid_composition_state_kwargs()
        kwargs["is_valid"] = "true"
        with pytest.raises(ValidationError):
            CompositionStateResponse(**kwargs)  # type: ignore[arg-type]

    def test_composition_state_response_rejects_non_mapping_sources(self) -> None:
        kwargs = _valid_composition_state_kwargs()
        kwargs["sources"] = "csv"
        with pytest.raises(ValidationError):
            CompositionStateResponse(**kwargs)  # type: ignore[arg-type]

    def test_composition_state_response_rejects_non_mapping_node_entry(self) -> None:
        kwargs = _valid_composition_state_kwargs()
        kwargs["nodes"] = ["node-1"]
        with pytest.raises(ValidationError):
            CompositionStateResponse(**kwargs)  # type: ignore[arg-type]

    def test_composition_state_response_rejects_nested_non_json_source_value(self) -> None:
        kwargs = _valid_composition_state_kwargs()
        kwargs["sources"] = {"source": {"type": "csv", "bad": object()}}
        with pytest.raises(ValidationError):
            CompositionStateResponse(**kwargs)  # type: ignore[arg-type]

    def test_composition_state_response_rejects_nested_non_json_metadata_value(self) -> None:
        kwargs = _valid_composition_state_kwargs()
        kwargs["metadata"] = {"name": "demo", "bad": object()}
        with pytest.raises(ValidationError):
            CompositionStateResponse(**kwargs)  # type: ignore[arg-type]

    def test_run_response_rejects_string_int_accounting(self) -> None:
        kwargs = _valid_run_response_kwargs()
        kwargs["accounting"] = {
            "sources": {"primary": {"rows_processed": "10"}},
            "tokens": {
                "emitted": 10,
                "terminal": 10,
                "succeeded": 10,
                "failed": 0,
                "structural": 0,
                "pending": 0,
            },
            "routing": {
                "routed_success": 0,
                "routed_failure": 0,
                "quarantined": 0,
                "discarded": 0,
            },
            "integrity": {
                "closure": "closed",
                "missing_terminal_outcomes": 0,
                "duplicate_terminal_outcomes": 0,
            },
        }
        with pytest.raises(ValidationError):
            RunResponse(**kwargs)  # type: ignore[arg-type]

    def test_run_response_rejects_iso_string_started_at(self) -> None:
        kwargs = _valid_run_response_kwargs()
        kwargs["started_at"] = "2026-04-15T10:00:00+00:00"
        with pytest.raises(ValidationError):
            RunResponse(**kwargs)  # type: ignore[arg-type]

    def test_validation_entry_response_rejects_int_as_str(self) -> None:
        with pytest.raises(ValidationError):
            ValidationEntryResponse(component=42, message="m", severity="warning")  # type: ignore[arg-type]


class TestSessionExtraFieldsRejected:
    """Extra fields must raise, not be silently dropped."""

    def test_session_response_rejects_extra(self) -> None:
        kwargs = _valid_session_response_kwargs()
        kwargs["extra_field"] = "nope"
        with pytest.raises(ValidationError, match="extra"):
            SessionResponse(**kwargs)  # type: ignore[arg-type]

    def test_chat_message_response_rejects_extra(self) -> None:
        kwargs = _valid_chat_message_kwargs()
        kwargs["model_version"] = "gpt-5"
        with pytest.raises(ValidationError, match="extra"):
            ChatMessageResponse(**kwargs)  # type: ignore[arg-type]

    def test_composition_state_response_rejects_extra(self) -> None:
        kwargs = _valid_composition_state_kwargs()
        kwargs["hash"] = "deadbeef"
        with pytest.raises(ValidationError, match="extra"):
            CompositionStateResponse(**kwargs)  # type: ignore[arg-type]

    def test_run_response_rejects_extra(self) -> None:
        kwargs = _valid_run_response_kwargs()
        kwargs["duration_ms"] = 1234
        with pytest.raises(ValidationError, match="extra"):
            RunResponse(**kwargs)  # type: ignore[arg-type]

    def test_validation_entry_response_rejects_extra(self) -> None:
        with pytest.raises(ValidationError, match="extra"):
            ValidationEntryResponse(component="c", message="m", severity="warning", code="X1")  # type: ignore[call-arg]

    def test_message_with_state_response_rejects_extra(self) -> None:
        msg = ChatMessageResponse(**_valid_chat_message_kwargs())  # type: ignore[arg-type]
        with pytest.raises(ValidationError, match="extra"):
            MessageWithStateResponse(message=msg, state=None, usage_tokens=100)  # type: ignore[call-arg]

    def test_fork_session_response_rejects_extra(self) -> None:
        session = SessionResponse(**_valid_session_response_kwargs())  # type: ignore[arg-type]
        with pytest.raises(ValidationError, match="extra"):
            ForkSessionResponse(session=session, messages=[], composition_state=None, note="x")  # type: ignore[call-arg]


class TestSessionResponseHappyPath:
    """The strictness contract must still allow the production construction paths."""

    def test_session_response_constructs_from_records(self) -> None:
        kwargs = _valid_session_response_kwargs()
        kwargs["forked_from_session_id"] = "sess-parent"
        kwargs["forked_from_message_id"] = "msg-parent"
        resp = SessionResponse(**kwargs)  # type: ignore[arg-type]
        assert resp.forked_from_session_id == "sess-parent"

    def test_composition_state_response_with_populated_containers(self) -> None:
        kwargs = _valid_composition_state_kwargs()
        kwargs["sources"] = {"orders": {"kind": "csv"}, "refunds": {"kind": "json"}}
        kwargs["nodes"] = [{"id": "n1"}]
        kwargs["validation_errors"] = ["boom"]
        kwargs["validation_warnings"] = [
            ValidationEntryResponse(component="c", message="m", severity="warning"),
        ]
        resp = CompositionStateResponse(**kwargs)  # type: ignore[arg-type]
        assert resp.sources == {"orders": {"kind": "csv"}, "refunds": {"kind": "json"}}
        assert resp.validation_errors == ["boom"]
        assert resp.validation_warnings is not None
        assert resp.validation_warnings[0].component == "c"


def test_workflow_profile_response_wire_subset_and_strict() -> None:
    from elspeth.web.sessions.schemas import WorkflowProfileResponse

    model = WorkflowProfileResponse(coaching=True, bookends=True, recipe_match=True, advisor_checkpoints=True)
    dumped = model.model_dump()
    assert set(dumped.keys()) == {
        "coaching",
        "bookends",
        "recipe_match",
        "advisor_checkpoints",
    }

    import pydantic

    with pytest.raises(pydantic.ValidationError):
        WorkflowProfileResponse(
            coaching=True,
            bookends=True,
            recipe_match=True,
            advisor_checkpoints=True,
            entry_seed="leak",
        )

    with pytest.raises(pydantic.ValidationError):
        WorkflowProfileResponse(
            coaching="yes",
            bookends=True,
            recipe_match=True,
            advisor_checkpoints=True,
        )
