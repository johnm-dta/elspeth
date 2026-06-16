"""Unit tests for durable scheduler contract value objects."""

from __future__ import annotations

from types import MappingProxyType

import pytest

from elspeth.contracts.scheduler import BufferedOutcomeSpec


def test_buffered_outcome_context_is_frozen_after_construction() -> None:
    source_context = {"branch": "left", "nested": {"attempt": 1}}
    spec = BufferedOutcomeSpec(batch_id="batch-1", context=source_context)

    assert isinstance(spec.context, MappingProxyType)
    assert isinstance(spec.context["nested"], MappingProxyType)

    source_context["branch"] = "right"
    assert spec.context["branch"] == "left"

    with pytest.raises(TypeError):
        spec.context["branch"] = "right"  # type: ignore[index]
