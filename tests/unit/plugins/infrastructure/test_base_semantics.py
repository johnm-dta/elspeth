"""Tests for BaseTransform / BaseSink / BaseSource default semantic and assistance hooks."""

from __future__ import annotations

from elspeth.contracts import Determinism
from elspeth.contracts.plugin_semantics import (
    InputSemanticRequirements,
    OutputSemanticDeclaration,
)
from elspeth.plugins.infrastructure.base import BaseSink, BaseSource, BaseTransform


class _StubTransform(BaseTransform):
    name = "stub"
    determinism = Determinism.DETERMINISTIC

    def process(self, row, ctx):  # pragma: no cover — not exercised
        raise NotImplementedError


def test_default_output_semantics_is_empty():
    instance = _StubTransform.__new__(_StubTransform)
    decl = instance.output_semantics()
    assert isinstance(decl, OutputSemanticDeclaration)
    assert decl.fields == ()


def test_default_input_semantic_requirements_is_empty():
    instance = _StubTransform.__new__(_StubTransform)
    reqs = instance.input_semantic_requirements()
    assert isinstance(reqs, InputSemanticRequirements)
    assert reqs.fields == ()


def test_default_get_agent_assistance_returns_none():
    result = _StubTransform.get_agent_assistance(issue_code=None)
    assert result is None
    result_with_code = _StubTransform.get_agent_assistance(issue_code="any.code")
    assert result_with_code is None


def test_default_get_post_call_hints_returns_empty_tuple():
    result = _StubTransform.get_post_call_hints(tool_name="upsert_node", config_snapshot={})
    assert result == ()


def test_basesink_default_get_agent_assistance_returns_none():
    """Confirm BaseSink exposes the assistance hook (dual-use with issue_code)."""
    assert BaseSink.get_agent_assistance(issue_code=None) is None
    assert BaseSink.get_agent_assistance(issue_code="any.code") is None


def test_basesink_default_get_post_call_hints_returns_empty_tuple():
    """Confirm BaseSink exposes the postscript hook with the two-param contract."""
    result = BaseSink.get_post_call_hints(tool_name="set_output", config_snapshot={})
    assert result == ()


def test_basesource_default_get_agent_assistance_returns_none():
    """Confirm BaseSource exposes the assistance hook (dual-use with issue_code)."""
    assert BaseSource.get_agent_assistance(issue_code=None) is None
    assert BaseSource.get_agent_assistance(issue_code="any.code") is None


def test_basesource_default_get_post_call_hints_returns_empty_tuple():
    """Confirm BaseSource exposes the postscript hook with the two-param contract."""
    result = BaseSource.get_post_call_hints(tool_name="set_source", config_snapshot={})
    assert result == ()
