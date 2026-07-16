"""Unit tests for the composer LLM-call audit L0 contract."""

from __future__ import annotations

import dataclasses
from datetime import UTC, datetime

import pytest

from elspeth.contracts.composer_llm_audit import (
    PROVIDER_COST_SOURCE_HIDDEN_PARAMS_RESPONSE_COST,
    ComposerChatInitiator,
    ComposerChatTurn,
    ComposerChatTurnRecorder,
    ComposerChatTurnStatus,
    ComposerLLMCall,
    ComposerLLMCallRecorder,
    ComposerLLMCallStatus,
)
from elspeth.core.canonical import stable_hash


def _make_call(**overrides: object) -> ComposerLLMCall:
    messages = [
        {"role": "system", "content": "System prompt"},
        {"role": "assistant", "content": "Prior turn"},
        {"role": "user", "content": "Current turn"},
    ]
    t = datetime(2026, 5, 4, 12, 0, 0, tzinfo=UTC)
    defaults: dict[str, object] = {
        "model_requested": "openrouter/openai/gpt-5.5",
        "model_returned": "openai/gpt-5.5-2026-05-01",
        "status": ComposerLLMCallStatus.SUCCESS,
        "prompt_tokens": 17,
        "completion_tokens": 5,
        "total_tokens": 22,
        "latency_ms": 123,
        "provider_request_id": "chatcmpl-safe-scalar",
        "messages_hash": stable_hash(messages),
        "tools_spec_hash": stable_hash([{"type": "function", "function": {"name": "set_source"}}]),
        "declared_tool_names": ("set_source",),
        "started_at": t,
        "finished_at": t,
        "error_class": None,
        "error_message": None,
        "temperature": 0.0,
        "seed": 42,
    }
    defaults.update(overrides)
    return ComposerLLMCall(**defaults)  # type: ignore[arg-type]


def test_status_strenum_values() -> None:
    assert ComposerLLMCallStatus.SUCCESS.value == "success"
    assert ComposerLLMCallStatus.TIMEOUT.value == "timeout"
    assert ComposerLLMCallStatus.API_ERROR.value == "api_error"
    assert ComposerLLMCallStatus.AUTH_ERROR.value == "auth_error"
    assert ComposerLLMCallStatus.BAD_REQUEST_ERROR.value == "bad_request_error"
    assert ComposerLLMCallStatus.MALFORMED_RESPONSE.value == "malformed_response"
    assert ComposerLLMCallStatus.CANCELLED.value == "cancelled"


def test_to_dict_serializes_enum_and_datetimes() -> None:
    call = _make_call()

    payload = call.to_dict()

    assert payload["status"] == "success"
    assert isinstance(payload["started_at"], str)
    assert isinstance(payload["finished_at"], str)
    assert payload["declared_tool_names"] == ["set_source"]


@pytest.mark.parametrize(
    ("declared_tool_names", "exc_type", "match"),
    [
        (["set_source"], TypeError, "declared_tool_names must be tuple"),
        (("",), ValueError, "declared_tool_names entries must be non-empty"),
        (("set_source", "set_source"), ValueError, "declared_tool_names entries must be unique"),
        (("set-source",), ValueError, "declared_tool_names entries must be tool names"),
    ],
)
def test_declared_tool_names_are_closed_structural_metadata(
    declared_tool_names: object,
    exc_type: type[Exception],
    match: str,
) -> None:
    with pytest.raises(exc_type, match=match):
        _make_call(declared_tool_names=declared_tool_names)


def test_token_none_values_are_preserved() -> None:
    call = _make_call(prompt_tokens=None, completion_tokens=None, total_tokens=None)

    payload = call.to_dict()

    assert payload["prompt_tokens"] is None
    assert payload["completion_tokens"] is None
    assert payload["total_tokens"] is None


def test_provider_cost_fields_are_serialized_without_fabricating_cost() -> None:
    call = _make_call(provider_cost=0.0037, provider_cost_source="response_usage.cost")

    payload = call.to_dict()

    assert payload["provider_cost"] == 0.0037
    assert payload["provider_cost_source"] == "response_usage.cost"


def test_private_provider_cost_source_is_serialized_with_provenance() -> None:
    call = _make_call(
        provider_cost=0.01234,
        provider_cost_source=PROVIDER_COST_SOURCE_HIDDEN_PARAMS_RESPONSE_COST,
    )

    payload = call.to_dict()

    assert payload["provider_cost"] == 0.01234
    assert payload["provider_cost_source"] == "_hidden_params.response_cost"


def test_model_drift_preserves_requested_and_returned_models() -> None:
    call = _make_call(model_requested="anthropic/claude-sonnet-4.5", model_returned="anthropic/claude-sonnet-4.5-20260501")

    payload = call.to_dict()

    assert payload["model_requested"] == "anthropic/claude-sonnet-4.5"
    assert payload["model_returned"] == "anthropic/claude-sonnet-4.5-20260501"


def test_messages_hash_is_hash_of_full_request_messages_array() -> None:
    full_messages = [
        {"role": "system", "content": "System prompt"},
        {"role": "assistant", "content": "Prior turn"},
        {"role": "user", "content": "Current turn"},
    ]
    without_history = [
        {"role": "system", "content": "System prompt"},
        {"role": "user", "content": "Current turn"},
    ]

    call = _make_call(messages_hash=stable_hash(full_messages))

    assert call.messages_hash == stable_hash(full_messages)
    assert call.messages_hash != stable_hash(without_history)


def test_error_fields_are_safe_class_name_payloads() -> None:
    call = _make_call(
        status=ComposerLLMCallStatus.AUTH_ERROR,
        model_returned=None,
        prompt_tokens=None,
        completion_tokens=None,
        total_tokens=None,
        provider_request_id=None,
        error_class="AuthenticationError",
        error_message="AuthenticationError",
    )

    payload = call.to_dict()

    assert payload["status"] == "auth_error"
    assert payload["error_class"] == "AuthenticationError"
    assert payload["error_message"] == "AuthenticationError"


def test_frozen_dataclass_blocks_mutation() -> None:
    call = _make_call()

    with pytest.raises(dataclasses.FrozenInstanceError):
        call.model_requested = "different"  # type: ignore[misc]


def test_l0_module_has_no_upward_imports() -> None:
    import elspeth.contracts.composer_llm_audit as audit

    forbidden_prefixes = ("elspeth.core", "elspeth.engine", "elspeth.plugins", "elspeth.web", "elspeth.cli")
    for ref in audit.__dict__.values():
        ref_module = getattr(ref, "__module__", None)
        if ref_module is None:
            continue
        for prefix in forbidden_prefixes:
            assert not ref_module.startswith(prefix), f"composer_llm_audit imports {ref!r} from forbidden module {ref_module}"


def test_cache_token_fields_default_to_none() -> None:
    """Cache token fields default to None — absence is evidence, not zero.

    Per CLAUDE.md fabrication policy and elspeth-4e79436719 §Bug C: a
    missing provider cache statistic must NOT be coerced to zero. The
    audit row distinguishes "no cache reported" from "cache reported
    zero hits" — only the latter is a real provider claim.
    """
    call = _make_call()
    payload = call.to_dict()
    assert call.cached_prompt_tokens is None
    assert call.cache_creation_input_tokens is None
    assert call.cache_read_input_tokens is None
    assert payload["cached_prompt_tokens"] is None
    assert payload["cache_creation_input_tokens"] is None
    assert payload["cache_read_input_tokens"] is None


def test_cache_token_fields_round_trip_when_known() -> None:
    """Cache fields persist through to_dict for both provider shapes."""
    call = _make_call(
        cached_prompt_tokens=1024,
        cache_creation_input_tokens=500,
        cache_read_input_tokens=900,
    )
    payload = call.to_dict()
    assert payload["cached_prompt_tokens"] == 1024
    assert payload["cache_creation_input_tokens"] == 500
    assert payload["cache_read_input_tokens"] == 900


def test_provider_reasoning_fields_round_trip_when_reported() -> None:
    """Provider-supplied reasoning artifacts are retained on the hidden audit sidecar.

    These fields are intentionally separate from normal assistant message
    content: they exist so operators can diagnose tool-call/config failures
    against provider metadata without exposing hidden reasoning in the user
    chat transcript.
    """
    reasoning_details = [
        {"type": "reasoning.text", "text": "checked available pipeline tools"},
        {"type": "reasoning.signature", "signature": "opaque-provider-signature"},
    ]
    thinking_blocks = [{"type": "thinking", "thinking": "provider-supplied thinking block"}]

    call = _make_call(
        reasoning_tokens=12,
        reasoning_content="provider supplied reasoning text",
        reasoning_details=reasoning_details,
        thinking_blocks=thinking_blocks,
    )

    payload = call.to_dict()

    assert payload["reasoning_tokens"] == 12
    assert payload["reasoning_content"] == "provider supplied reasoning text"
    assert payload["reasoning_details"] == reasoning_details
    assert payload["thinking_blocks"] == thinking_blocks


def test_provider_reasoning_fields_default_to_none() -> None:
    call = _make_call()
    payload = call.to_dict()

    assert call.reasoning_tokens is None
    assert call.reasoning_content is None
    assert call.reasoning_details is None
    assert call.thinking_blocks is None
    assert payload["reasoning_tokens"] is None
    assert payload["reasoning_content"] is None
    assert payload["reasoning_details"] is None
    assert payload["thinking_blocks"] is None


def test_composer_llm_call_records_temperature_and_seed() -> None:
    """Configured temperature and seed round-trip through to_dict()."""
    call = _make_call(temperature=0.0, seed=42)

    payload = call.to_dict()

    assert call.temperature == 0.0
    assert call.seed == 42
    assert payload["temperature"] == 0.0
    assert payload["seed"] == 42


def test_temperature_accepts_none_for_omitted_request_parameter() -> None:
    call = _make_call(temperature=None)

    payload = call.to_dict()

    assert call.temperature is None
    assert payload["temperature"] is None


def test_composer_llm_call_allows_seed_none_when_provider_omits_it() -> None:
    """Unsupported provider params are omitted; audit records the actual request shape."""
    call = _make_call(model_requested="anthropic/claude-3-5-sonnet-20241022", seed=None)

    payload = call.to_dict()

    assert call.temperature == 0.0
    assert call.seed is None
    assert payload["seed"] is None


@pytest.mark.parametrize(
    ("overrides", "exc_type", "match"),
    [
        ({"status": "success"}, TypeError, "status must be ComposerLLMCallStatus"),
        ({"prompt_tokens": True}, TypeError, "prompt_tokens must be int"),
        ({"completion_tokens": -1}, ValueError, "completion_tokens must be >= 0"),
        ({"total_tokens": 1.5}, TypeError, "total_tokens must be int"),
        ({"latency_ms": -1}, ValueError, "latency_ms must be >= 0"),
        ({"seed": 1.5}, TypeError, "seed must be int"),
        ({"temperature": float("inf")}, ValueError, "temperature must be finite"),
        ({"model_requested": ""}, ValueError, "model_requested must be non-empty"),
        ({"model_returned": ""}, ValueError, "model_returned must be non-empty"),
        ({"messages_hash": ""}, ValueError, "messages_hash must be non-empty"),
        ({"tools_spec_hash": ""}, ValueError, "tools_spec_hash must be non-empty"),
        ({"provider_request_id": ""}, ValueError, "provider_request_id must be non-empty"),
        ({"started_at": "2026-05-04T12:00:00Z"}, TypeError, "started_at must be datetime"),
        ({"finished_at": "2026-05-04T12:00:00Z"}, TypeError, "finished_at must be datetime"),
        ({"finished_at": datetime(2026, 5, 4, 11, 59, 59, tzinfo=UTC)}, ValueError, "finished_at must be >= started_at"),
        ({"error_class": "TimeoutError"}, ValueError, "SUCCESS calls must not include error_class or error_message"),
        (
            {"status": ComposerLLMCallStatus.TIMEOUT, "error_class": None, "error_message": "TimeoutError"},
            ValueError,
            "non-success calls must include error_class and error_message",
        ),
    ],
)
def test_composer_llm_call_rejects_invalid_audit_shape(
    overrides: dict[str, object],
    exc_type: type[Exception],
    match: str,
) -> None:
    with pytest.raises(exc_type, match=match):
        _make_call(**overrides)


def test_recorder_protocol_runtime_check() -> None:
    class _StubRecorder:
        def record_llm_call(self, call: ComposerLLMCall) -> None:
            return

        def resolve_session(self, session_id: str) -> None:
            return

    rec: ComposerLLMCallRecorder = _StubRecorder()
    rec.record_llm_call(_make_call())
    rec.resolve_session("abc")


# ---------------------------------------------------------------------------
# ComposerChatTurn — Phase A slice 5
# ---------------------------------------------------------------------------


def _make_chat_turn(**overrides: object) -> ComposerChatTurn:
    t = datetime(2026, 5, 13, 12, 0, 0, tzinfo=UTC)
    defaults: dict[str, object] = {
        "step": "step_1_source",
        "initiator": ComposerChatInitiator.USER,
        "chat_turn_seq": 0,
        "user_message_hash": stable_hash("what columns?"),
        "assistant_message_hash": stable_hash("col_a, col_b"),
        "latency_ms": 250,
        "model": "openrouter/openai/gpt-5.5",
        "status": ComposerChatTurnStatus.SUCCESS,
        "started_at": t,
        "finished_at": t,
        "error_class": None,
    }
    defaults.update(overrides)
    return ComposerChatTurn(**defaults)  # type: ignore[arg-type]


def test_chat_turn_status_strenum_values() -> None:
    assert ComposerChatTurnStatus.SUCCESS.value == "success"
    assert ComposerChatTurnStatus.SYNTHETIC_UNAVAILABLE.value == "synthetic_unavailable"
    assert ComposerChatTurnStatus.INVARIANT_VIOLATED.value == "invariant_violated"


def test_chat_turn_initiator_strenum_values() -> None:
    assert ComposerChatInitiator.USER.value == "user"
    assert ComposerChatInitiator.STEP_ENTRY_OPENER.value == "step_entry_opener"


def test_chat_turn_to_dict_serializes_enum_and_datetimes() -> None:
    turn = _make_chat_turn()

    payload = turn.to_dict()

    assert payload["status"] == "success"
    assert payload["initiator"] == "user"
    assert isinstance(payload["started_at"], str)
    assert isinstance(payload["finished_at"], str)


def test_chat_turn_negative_seq_rejected() -> None:
    with pytest.raises(ValueError, match="chat_turn_seq"):
        _make_chat_turn(chat_turn_seq=-1)


def test_chat_turn_negative_latency_rejected() -> None:
    with pytest.raises(ValueError, match="latency_ms"):
        _make_chat_turn(latency_ms=-1)


def test_chat_turn_unknown_initiator_rejected() -> None:
    with pytest.raises(ValueError, match="opener"):
        ComposerChatInitiator("opener")  # close but not exact


def test_chat_turn_raw_initiator_string_rejected_at_construction() -> None:
    with pytest.raises(TypeError, match="initiator"):
        _make_chat_turn(initiator="user")


def test_chat_turn_raw_status_string_rejected_at_construction() -> None:
    with pytest.raises(TypeError, match="status"):
        _make_chat_turn(status="success")


@pytest.mark.parametrize(
    ("overrides", "exc_type", "match"),
    [
        ({"step": ""}, ValueError, "step must be non-empty"),
        ({"model": ""}, ValueError, "model must be non-empty"),
        ({"user_message_hash": ""}, ValueError, "user_message_hash must be non-empty"),
        ({"assistant_message_hash": ""}, ValueError, "assistant_message_hash must be non-empty"),
        ({"started_at": "2026-05-13T12:00:00Z"}, TypeError, "started_at must be datetime"),
        ({"finished_at": "2026-05-13T12:00:00Z"}, TypeError, "finished_at must be datetime"),
        ({"finished_at": datetime(2026, 5, 13, 11, 59, 59, tzinfo=UTC)}, ValueError, "finished_at must be >= started_at"),
        (
            {"status": ComposerChatTurnStatus.SYNTHETIC_UNAVAILABLE, "error_class": ""},
            ValueError,
            "error_class must be non-empty",
        ),
    ],
)
def test_chat_turn_rejects_invalid_audit_shape(
    overrides: dict[str, object],
    exc_type: type[Exception],
    match: str,
) -> None:
    with pytest.raises(exc_type, match=match):
        _make_chat_turn(**overrides)


def test_chat_turn_success_requires_no_error_class() -> None:
    with pytest.raises(ValueError, match="error_class"):
        _make_chat_turn(status=ComposerChatTurnStatus.SUCCESS, error_class="TimeoutError")


def test_chat_turn_synthetic_requires_error_class() -> None:
    with pytest.raises(ValueError, match="error_class"):
        _make_chat_turn(status=ComposerChatTurnStatus.SYNTHETIC_UNAVAILABLE, error_class=None)


def test_chat_turn_synthetic_with_error_class_succeeds() -> None:
    turn = _make_chat_turn(
        status=ComposerChatTurnStatus.SYNTHETIC_UNAVAILABLE,
        error_class="TimeoutError",
    )

    payload = turn.to_dict()

    assert payload["status"] == "synthetic_unavailable"
    assert payload["error_class"] == "TimeoutError"


def test_chat_turn_invariant_violated_requires_error_class() -> None:
    with pytest.raises(ValueError, match="error_class"):
        _make_chat_turn(status=ComposerChatTurnStatus.INVARIANT_VIOLATED, error_class=None)


def test_chat_turn_invariant_violated_with_error_class_succeeds() -> None:
    turn = _make_chat_turn(
        status=ComposerChatTurnStatus.INVARIANT_VIOLATED,
        error_class="InvariantError",
    )

    payload = turn.to_dict()

    assert payload["status"] == "invariant_violated"
    assert payload["error_class"] == "InvariantError"


def test_chat_turn_recorder_protocol_runtime_check() -> None:
    class _StubChatRecorder:
        def __init__(self) -> None:
            self.calls: list[ComposerChatTurn] = []

        def record_chat_turn(self, turn: ComposerChatTurn) -> None:
            self.calls.append(turn)

    rec: ComposerChatTurnRecorder = _StubChatRecorder()
    chat_turn = _make_chat_turn()
    rec.record_chat_turn(chat_turn)
    assert rec.calls == [chat_turn]  # type: ignore[attr-defined]
