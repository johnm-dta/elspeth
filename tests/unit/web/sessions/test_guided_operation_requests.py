"""Strict HTTP request contracts for retry-safe composer mutations."""

from __future__ import annotations

from uuid import uuid4

import pytest
from pydantic import ValidationError

from elspeth.web.sessions.guided_operations import guided_operation_request_hash
from elspeth.web.sessions.schemas import (
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
    ],
)
def test_mutating_composer_requests_require_operation_id(model_type, payload) -> None:
    with pytest.raises(ValidationError, match="operation_id"):
        model_type.model_validate(payload)


@pytest.mark.parametrize(
    "model_type",
    [StartGuidedRequest, ConvertGuidedRequest, ReenterGuidedRequest, RevertStateRequest],
)
def test_mutating_composer_requests_are_strict_and_extra_forbid(model_type) -> None:
    assert model_type.model_config.get("strict") is True
    assert model_type.model_config.get("extra") == "forbid"


@pytest.mark.parametrize(
    ("model_type", "payload"),
    [
        (GuidedRespondRequest, {"chosen": ["csv"]}),
        (GuidedChatRequest, {"message": "Use CSV", "step_index": "step_1_source"}),
        (ForkSessionRequest, {"from_message_id": str(uuid4()), "new_message_content": "Try this"}),
    ],
)
def test_pending_handlers_do_not_expose_unenforced_operation_id(model_type, payload) -> None:
    request = model_type.model_validate(payload)

    assert "operation_id" not in type(request).model_fields
    assert type(request).model_config.get("strict") is not True


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
