"""Tests for the ToolRedaction manifest-entry dataclass (spec §4.2.1).

Construction errors are precondition contracts; any future bug that
introduces both-shapes-set or neither-shape-set should fail at import
time, not at walk time.
"""

from __future__ import annotations

import pytest
from pydantic import BaseModel

from elspeth.web.composer.redaction import (
    HandlesNoSensitiveDataReason,
    ToolRedaction,
    ToolRedactionPolicy,
)


class _DummyModel(BaseModel):
    x: str


def _ok_reason() -> HandlesNoSensitiveDataReason:
    return HandlesNoSensitiveDataReason(
        sensitive_data_locations=("nowhere — see test",),
        why_arguments_safe="A" * 32,
        why_responses_safe="B" * 32,
    )


def _ok_policy_no_sensitive() -> ToolRedactionPolicy:
    return ToolRedactionPolicy(
        handles_no_sensitive_data=True,
        handles_no_sensitive_data_reason_struct=_ok_reason(),
    )


def test_type_driven_entry_is_constructable() -> None:
    entry = ToolRedaction(argument_model=_DummyModel)
    assert entry.argument_model is _DummyModel
    assert entry.policy is None


def test_declarative_entry_is_constructable() -> None:
    policy = _ok_policy_no_sensitive()
    entry = ToolRedaction(policy=policy)
    assert entry.policy is policy
    assert entry.argument_model is None


def test_both_shapes_set_raises_value_error() -> None:
    with pytest.raises(ValueError, match="both argument_model and policy"):
        ToolRedaction(argument_model=_DummyModel, policy=_ok_policy_no_sensitive())


def test_neither_shape_set_raises_value_error() -> None:
    with pytest.raises(ValueError, match="neither argument_model nor policy"):
        ToolRedaction()


def test_response_model_without_argument_model_raises() -> None:
    with pytest.raises(ValueError, match="response_model requires argument_model"):
        ToolRedaction(response_model=_DummyModel, policy=_ok_policy_no_sensitive())
