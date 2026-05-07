"""Tests for BaseTransform default semantic declaration methods."""

from __future__ import annotations

from elspeth.contracts.plugin_semantics import (
    InputSemanticRequirements,
    OutputSemanticDeclaration,
)
from elspeth.plugins.infrastructure.base import BaseTransform


class _StubTransform(BaseTransform):
    name = "stub"

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
