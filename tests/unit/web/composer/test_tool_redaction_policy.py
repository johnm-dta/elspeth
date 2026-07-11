"""Tests for ToolRedactionPolicy construction-time validators (spec §4.2.3 / Task 6).

Six cases covering the four invariants enforced by __post_init__:
  1. Orphan summarizer raises ValueError.
  2. handles_no_sensitive_data=True without reason struct raises ValueError.
  3. handles_no_sensitive_data=False with non-None reason struct raises ValueError.
  4. handles_no_sensitive_data=False without known_response_keys raises ValueError.
  5. Valid construction with sensitive data succeeds (no exception).
  6. Container fields are deeply frozen after construction.
"""

from __future__ import annotations

from types import MappingProxyType

import pytest

from elspeth.web.composer.redaction import (
    HandlesNoSensitiveDataReason,
    ToolRedactionPolicy,
)


def _redact(v: object) -> str:
    return "<redacted>"


def _ok_reason() -> HandlesNoSensitiveDataReason:
    return HandlesNoSensitiveDataReason(
        sensitive_data_locations=("nowhere — see test",),
        why_arguments_safe="A" * 32,
        why_responses_safe="B" * 32,
    )


# ---------------------------------------------------------------------------
# Test 1: Orphan summarizer key not in sensitive_argument_keys raises
# ---------------------------------------------------------------------------


def test_orphan_summarizer_raises() -> None:
    """argument_summarizers key not declared in sensitive_argument_keys raises ValueError."""
    with pytest.raises(ValueError, match="orphan"):
        ToolRedactionPolicy(
            sensitive_argument_keys=("path",),
            known_response_keys=("status",),
            argument_summarizers={"path": _redact, "extra_key": _redact},
        )


def test_known_argument_keys_must_cover_sensitive_arguments_when_declared() -> None:
    """An opt-in argument allowlist must include every sensitive argument key."""
    with pytest.raises(ValueError, match="known_argument_keys"):
        ToolRedactionPolicy(
            sensitive_argument_keys=("path",),
            known_argument_keys=("name",),
            known_response_keys=("status",),
            argument_summarizers={"path": _redact},
        )


# ---------------------------------------------------------------------------
# Test 2: handles_no_sensitive_data=True without reason struct raises
# ---------------------------------------------------------------------------


def test_handles_no_sensitive_data_true_without_reason_raises() -> None:
    """handles_no_sensitive_data=True with no reason struct raises ValueError."""
    with pytest.raises(ValueError, match="handles_no_sensitive_data_reason_struct"):
        ToolRedactionPolicy(
            handles_no_sensitive_data=True,
            handles_no_sensitive_data_reason_struct=None,
        )


# ---------------------------------------------------------------------------
# Test 3: handles_no_sensitive_data=False with non-None reason struct raises
# ---------------------------------------------------------------------------


def test_handles_no_sensitive_data_false_with_reason_raises() -> None:
    """handles_no_sensitive_data=False with a non-None reason struct raises ValueError."""
    with pytest.raises(ValueError, match="handles_no_sensitive_data_reason_struct"):
        ToolRedactionPolicy(
            handles_no_sensitive_data=False,
            handles_no_sensitive_data_reason_struct=_ok_reason(),
            known_response_keys=("status",),
        )


# ---------------------------------------------------------------------------
# Test 4: handles_no_sensitive_data=False without known_response_keys raises
# ---------------------------------------------------------------------------


def test_handles_false_without_known_response_keys_raises() -> None:
    """handles_no_sensitive_data=False with empty known_response_keys raises ValueError."""
    with pytest.raises(ValueError, match="known_response_keys"):
        ToolRedactionPolicy(
            handles_no_sensitive_data=False,
            sensitive_argument_keys=("path",),
            argument_summarizers={"path": _redact},
            known_response_keys=(),
        )


# ---------------------------------------------------------------------------
# Test 5: Valid construction with sensitive data succeeds
# ---------------------------------------------------------------------------


def test_valid_policy_with_sensitive_data_is_constructable() -> None:
    """handles_no_sensitive_data=False with full valid fields constructs without error."""
    policy = ToolRedactionPolicy(
        sensitive_argument_keys=("path",),
        argument_summarizers={"path": _redact},
        known_argument_keys=("path",),
        known_response_keys=("status",),
        handles_no_sensitive_data=False,
        handles_no_sensitive_data_reason_struct=None,
    )
    assert policy.sensitive_argument_keys == ("path",)
    assert policy.known_argument_keys == ("path",)
    assert policy.known_response_keys == ("status",)
    assert policy.handles_no_sensitive_data is False
    assert policy.handles_no_sensitive_data_reason_struct is None


# ---------------------------------------------------------------------------
# Test 6: Container fields are deeply frozen after construction
# ---------------------------------------------------------------------------


def test_container_fields_are_deeply_frozen() -> None:
    """All tuple/Mapping container fields are deeply frozen after __post_init__."""
    policy = ToolRedactionPolicy(
        sensitive_argument_keys=("path",),
        sensitive_response_keys=("body",),
        known_argument_keys=("path",),
        known_response_keys=("status",),
        argument_summarizers={"path": _redact},
        handles_no_sensitive_data=False,
    )

    # Tuples are already immutable; verify they remain tuples (not converted).
    assert isinstance(policy.sensitive_argument_keys, tuple)
    assert isinstance(policy.sensitive_response_keys, tuple)
    assert isinstance(policy.known_argument_keys, tuple)
    assert isinstance(policy.known_response_keys, tuple)

    # argument_summarizers must be wrapped as MappingProxyType (deeply frozen).
    assert isinstance(policy.argument_summarizers, MappingProxyType)

    # The dataclass itself must be immutable (frozen=True prevents assignment).
    with pytest.raises(AttributeError):
        policy.handles_no_sensitive_data = True  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Queue exposure must not narrow the generic persisted transport model
# (elspeth-a5b86149d4). ``_PipelineNodeModel.node_type`` stays a plain ``str``
# so set_pipeline keeps Tier-3 LLM-recoverable feedback for unknown enum
# values; the closed vocabulary lives on the composer state / review DTOs.
# ---------------------------------------------------------------------------


def test_pipeline_node_transport_model_node_type_stays_open_str() -> None:
    from elspeth.web.composer.redaction import _PipelineNodeModel

    assert _PipelineNodeModel.model_fields["node_type"].annotation is str
    model = _PipelineNodeModel.model_validate({"id": "inbound", "node_type": "queue", "input": "inbound", "options": {"description": "x"}})
    assert model.node_type == "queue"
