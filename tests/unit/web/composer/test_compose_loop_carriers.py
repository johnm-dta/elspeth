"""Tests for compose-loop carrier contracts."""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType
from typing import Any, get_args, get_origin, get_type_hints

import pytest

from elspeth.web.composer._compose_loop_carriers import _ToolOutcome, _ToolOutcomeResponse
from elspeth.web.composer.tools._common import ToolResult


def test_tool_outcome_response_has_named_sum_type_contract() -> None:
    """The P3/P4 response serializer dispatches on a declared union."""

    hints = get_type_hints(_ToolOutcome)
    assert hints["response"] is _ToolOutcomeResponse

    response_members = set(get_args(_ToolOutcomeResponse))
    assert ToolResult in response_members
    assert type(None) in response_members
    assert any(get_origin(member) is Mapping for member in response_members)


def test_tool_outcome_freezes_mapping_response() -> None:
    """Mapping response payloads must be frozen before P4 redaction reads them."""

    response_dict: dict[str, Any] = {"ok": True, "nested": {"k": "v"}}
    outcome = _ToolOutcome(
        call={"id": "tc_x", "function": {"name": "request_advisor_hint"}},
        response=response_dict,
        error_class=None,
        error_message=None,
        pre_version=1,
        post_version=1,
    )

    assert isinstance(outcome.response, MappingProxyType)
    assert isinstance(outcome.call, MappingProxyType)
    with pytest.raises(TypeError):
        outcome.response["ok"] = False  # type: ignore[index]
    response_dict["ok"] = False
    assert outcome.response["ok"] is True
