"""Strict HTTP request contracts for retry-safe composer mutations."""

from __future__ import annotations

from uuid import UUID, uuid4

import pytest
from pydantic import ValidationError

from elspeth.web.sessions.guided_operations import guided_operation_request_hash
from elspeth.web.sessions.schemas import (
    AddComponentAction,
    ConvertGuidedRequest,
    ForkSessionRequest,
    GuidedChatRequest,
    GuidedRespondRequest,
    ReenterGuidedRequest,
    RevertStateRequest,
    StartGuidedRequest,
)

_OPERATION_ID = "00000000-0000-4000-8000-000000000001"


@pytest.mark.parametrize(
    ("model_type", "payload"),
    [
        (StartGuidedRequest, {"profile": "live"}),
        (ConvertGuidedRequest, {}),
        (ReenterGuidedRequest, {}),
        (RevertStateRequest, {"state_id": str(uuid4())}),
        (ForkSessionRequest, {"from_message_id": uuid4(), "new_message_content": "Try this"}),
    ],
)
def test_mutating_composer_requests_require_operation_id(model_type, payload) -> None:
    with pytest.raises(ValidationError, match="operation_id"):
        model_type.model_validate(payload)


@pytest.mark.parametrize(
    "model_type",
    [StartGuidedRequest, ConvertGuidedRequest, ReenterGuidedRequest, RevertStateRequest, ForkSessionRequest],
)
def test_mutating_composer_requests_are_strict_and_extra_forbid(model_type) -> None:
    assert model_type.model_config.get("strict") is True
    assert model_type.model_config.get("extra") == "forbid"


def test_guided_chat_requires_strict_operation_and_turn_tokens() -> None:
    valid = {
        "operation_id": _OPERATION_ID,
        "turn_token": "a" * 64,
        "message": "Use CSV",
    }

    request = GuidedChatRequest.model_validate(valid)

    assert request.model_dump(mode="python") == valid
    assert type(request).model_config.get("strict") is True
    assert type(request).model_config.get("extra") == "forbid"
    for missing in ("operation_id", "turn_token"):
        with pytest.raises(ValidationError, match=missing):
            GuidedChatRequest.model_validate({key: value for key, value in valid.items() if key != missing})
    with pytest.raises(ValidationError, match="turn_token"):
        GuidedChatRequest.model_validate({**valid, "turn_token": None})
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        GuidedChatRequest.model_validate({**valid, "step_index": "step_1_source"})


def test_guided_respond_requires_strict_operation_and_live_turn_tokens() -> None:
    with pytest.raises(ValidationError, match="operation_id"):
        GuidedRespondRequest.model_validate({"turn_token": "a" * 64, "chosen": ["csv"]})
    with pytest.raises(ValidationError, match="turn_token"):
        GuidedRespondRequest.model_validate({"operation_id": _OPERATION_ID, "chosen": ["csv"]})
    with pytest.raises(ValidationError, match="turn_token"):
        GuidedRespondRequest.model_validate({"operation_id": _OPERATION_ID, "turn_token": None, "chosen": ["csv"]})

    request = GuidedRespondRequest.model_validate({"operation_id": _OPERATION_ID, "turn_token": "a" * 64, "chosen": ["csv"]})

    assert request.operation_id == _OPERATION_ID
    assert request.turn_token == "a" * 64
    assert type(request).model_config.get("strict") is True
    assert type(request).model_config.get("extra") == "forbid"


def test_guided_respond_terminal_exit_is_the_only_null_token_shape() -> None:
    request = GuidedRespondRequest.model_validate(
        {
            "operation_id": _OPERATION_ID,
            "turn_token": None,
            "control_signal": "exit_to_freeform",
        }
    )

    assert request.turn_token is None

    with pytest.raises(ValidationError, match="cannot be combined"):
        GuidedRespondRequest.model_validate(
            {
                "operation_id": _OPERATION_ID,
                "turn_token": "a" * 64,
                "control_signal": "exit_to_freeform",
                "chosen": ["csv"],
            }
        )


def test_guided_respond_rejects_empty_actions_and_mixed_control_shapes() -> None:
    with pytest.raises(ValidationError, match="action is required"):
        GuidedRespondRequest.model_validate({"operation_id": _OPERATION_ID, "turn_token": "a" * 64})
    with pytest.raises(ValidationError, match="control_signal"):
        GuidedRespondRequest.model_validate(
            {
                "operation_id": _OPERATION_ID,
                "turn_token": "a" * 64,
                "control_signal": "passthrough",
                "custom_inputs": ["field"],
            }
        )


@pytest.mark.parametrize(
    "component_action",
    [
        {"action": "add", "component_kind": "source"},
        {
            "action": "edit",
            "target": {"kind": "source", "stable_id": "11111111-1111-4111-8111-111111111111"},
        },
        {
            "action": "remove",
            "target": {"kind": "output", "stable_id": "33333333-3333-4333-8333-333333333333"},
        },
        {
            "action": "reorder",
            "component_kind": "output",
            "stable_ids": [
                "33333333-3333-4333-8333-333333333333",
                "44444444-4444-4444-8444-444444444444",
            ],
        },
        {"action": "finish", "component_kind": "source"},
    ],
    ids=["add", "edit", "remove", "reorder", "finish"],
)
def test_guided_respond_accepts_one_closed_component_action(component_action: dict[str, object]) -> None:
    request = GuidedRespondRequest.model_validate(
        {
            "operation_id": _OPERATION_ID,
            "turn_token": "a" * 64,
            "component_action": component_action,
        }
    )

    assert request.component_action is not None
    assert request.component_action.action == component_action["action"]
    assert type(request).model_config.get("strict") is True
    assert isinstance(
        AddComponentAction.model_validate({"action": "add", "component_kind": "source"}),
        AddComponentAction,
    )


@pytest.mark.parametrize(
    "component_action",
    [
        {"action": "add", "component_kind": "node"},
        {"action": "add", "component_kind": "source", "stable_id": "11111111-1111-4111-8111-111111111111"},
        {"action": "edit", "target": {"kind": "node", "stable_id": "11111111-1111-4111-8111-111111111111"}},
        {"action": "remove", "target": {"kind": "edge", "stable_id": "11111111-1111-4111-8111-111111111111"}},
        {"action": "edit", "target": {"kind": "source", "stable_id": "11111111-1111-4111-8111-11111111111A"}},
        {"action": "reorder", "component_kind": "source"},
        {"action": "reorder", "component_kind": "source", "stable_ids": []},
        {
            "action": "reorder",
            "component_kind": "source",
            "stable_ids": [
                "11111111-1111-4111-8111-111111111111",
                "11111111-1111-4111-8111-111111111111",
            ],
        },
        {
            "action": "reorder",
            "component_kind": "source",
            "stable_ids": [str(UUID(int=index + 1)) for index in range(257)],
        },
        {"action": "finish"},
        {"action": "replace", "component_kind": "source"},
    ],
    ids=[
        "closed-kind",
        "extra-field",
        "node-target",
        "edge-target",
        "noncanonical-target",
        "missing-ids",
        "empty-ids",
        "duplicate-ids",
        "too-many-ids",
        "missing-kind",
        "closed-action",
    ],
)
def test_guided_respond_rejects_malformed_component_action(component_action: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        GuidedRespondRequest.model_validate(
            {
                "operation_id": _OPERATION_ID,
                "turn_token": "a" * 64,
                "component_action": component_action,
            }
        )


@pytest.mark.parametrize(
    "other_fields",
    [
        {"chosen": ["csv"]},
        {"edited_values": {"path": "/data/input.csv"}},
        {"custom_inputs": ["field"]},
        {"control_signal": "back"},
        {"proposal_id": "00000000-0000-4000-8000-000000000002", "draft_hash": "b" * 64},
        {
            "proposal_id": "00000000-0000-4000-8000-000000000002",
            "draft_hash": "b" * 64,
            "edit_target": {"kind": "source", "stable_id": "00000000-0000-4000-8000-000000000003"},
        },
    ],
    ids=["chosen", "edited", "custom", "control", "proposal", "proposal-edit"],
)
def test_guided_respond_rejects_component_action_combined_with_other_response(other_fields: dict[str, object]) -> None:
    with pytest.raises(ValidationError, match="component_action"):
        GuidedRespondRequest.model_validate(
            {
                "operation_id": _OPERATION_ID,
                "turn_token": "a" * 64,
                "component_action": {"action": "finish", "component_kind": "source"},
                **other_fields,
            }
        )


@pytest.mark.parametrize(
    "proposal_fields",
    [
        {"proposal_id": "00000000-0000-4000-8000-000000000002"},
        {"draft_hash": "b" * 64},
        {
            "edit_target": {
                "kind": "source",
                "stable_id": "00000000-0000-4000-8000-000000000003",
            }
        },
    ],
)
def test_guided_respond_rejects_partial_proposal_bindings(proposal_fields: dict[str, object]) -> None:
    with pytest.raises(ValidationError, match="proposal"):
        GuidedRespondRequest.model_validate(
            {
                "operation_id": _OPERATION_ID,
                "turn_token": "a" * 64,
                **proposal_fields,
            }
        )


@pytest.mark.parametrize("legacy_field", ["step_index", "accepted_step_index", "edit_step_index"])
def test_guided_respond_rejects_legacy_index_fields(legacy_field: str) -> None:
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        GuidedRespondRequest.model_validate(
            {
                "operation_id": _OPERATION_ID,
                "turn_token": "a" * 64,
                "chosen": ["csv"],
                legacy_field: 0,
            }
        )


def test_guided_respond_preserves_closed_proposal_bindings_for_route_validation() -> None:
    request = GuidedRespondRequest.model_validate(
        {
            "operation_id": _OPERATION_ID,
            "turn_token": "a" * 64,
            "proposal_id": "00000000-0000-4000-8000-000000000002",
            "draft_hash": "b" * 64,
            "edit_target": {
                "kind": "source",
                "stable_id": "00000000-0000-4000-8000-000000000003",
            },
        }
    )
    assert request.proposal_id == "00000000-0000-4000-8000-000000000002"
    assert request.edit_target is not None

    for malformed_stable_id in ("", "0" * 35, "0" * 37, "00000000-0000-4000-8000-00000000000A", "../source"):
        malformed_target = GuidedRespondRequest.model_validate(
            {
                "operation_id": _OPERATION_ID,
                "turn_token": "a" * 64,
                "proposal_id": "00000000-0000-4000-8000-000000000002",
                "draft_hash": "b" * 64,
                "edit_target": {"kind": "source", "stable_id": malformed_stable_id},
            }
        )
        assert malformed_target.edit_target is not None
        assert malformed_target.edit_target.stable_id == malformed_stable_id

    malformed = GuidedRespondRequest.model_validate(
        {
            "operation_id": _OPERATION_ID,
            "turn_token": "a" * 64,
            "proposal_id": "not-canonical",
            "draft_hash": "raw-diagnostic",
            "edit_target": {"kind": "source", "stable_id": "../source"},
        }
    )
    assert malformed.proposal_id == "not-canonical"
    assert malformed.draft_hash == "raw-diagnostic"
    assert malformed.edit_target is not None
    assert malformed.edit_target.stable_id == "../source"


@pytest.mark.parametrize("model_type", [StartGuidedRequest, ReenterGuidedRequest])
def test_operation_id_must_be_a_canonical_uuid(model_type) -> None:
    with pytest.raises(ValidationError, match="canonical UUID"):
        model_type.model_validate({"operation_id": "z" * 36})


def test_operation_hash_ignores_retry_id_but_binds_kind_and_session() -> None:
    session_id = uuid4()
    request_a = ReenterGuidedRequest.model_validate({"operation_id": _OPERATION_ID})
    request_b = ReenterGuidedRequest.model_validate({"operation_id": "00000000-0000-4000-8000-000000000002"})

    reenter_hash = guided_operation_request_hash(session_id=session_id, kind="guided_reenter", request=request_a)

    assert reenter_hash == guided_operation_request_hash(session_id=session_id, kind="guided_reenter", request=request_b)
    assert reenter_hash != guided_operation_request_hash(session_id=session_id, kind="state_revert", request=request_a)
    assert reenter_hash != guided_operation_request_hash(session_id=uuid4(), kind="guided_reenter", request=request_a)
