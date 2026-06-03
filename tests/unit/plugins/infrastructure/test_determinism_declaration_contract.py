"""Contract: every plugin subclass MUST explicitly declare `determinism`.

The Determinism enum docstring at ``contracts/enums.py`` states "undeclared
determinism crashes at registration time."  Until the ``__init_subclass__``
guards landed on ``BaseTransform`` / ``BaseSink`` / ``BaseSource``, that
sentence was aspirational — silent class-level defaults
(``Determinism.DETERMINISTIC`` / ``IO_WRITE`` / ``IO_READ``) on the bases
meant a subclass that forgot to redeclare would silently inherit a value
and then record the wrong determinism per-node in the Landscape legal
record, misclassify nodes in the readiness panel, render the wrong
audit-characteristic chips in the catalog, and feed a wrong
ReproducibilityGrade.

These tests verify the contract is now mechanically enforced — at class
creation time, not at registration time, not at audit time, not "if anyone
notices."
"""

from __future__ import annotations

import pytest

from elspeth.contracts import Determinism, PluginSchema
from elspeth.plugins.infrastructure.base import BaseSink, BaseSource, BaseTransform


class _DummySchema(PluginSchema):
    """Minimal schema for satisfying base-class typing in contract tests."""

    @classmethod
    def from_dict(cls, data: dict) -> _DummySchema:
        return cls()


def test_basetransform_subclass_without_determinism_fails_at_class_creation() -> None:
    with pytest.raises(TypeError, match=r"does not explicitly declare a `determinism`"):

        class _NoDeclTransform(BaseTransform):
            name = "no_decl_transform"
            input_schema = _DummySchema
            output_schema = _DummySchema


def test_basesink_subclass_without_determinism_fails_at_class_creation() -> None:
    with pytest.raises(TypeError, match=r"does not explicitly declare a `determinism`"):

        class _NoDeclSink(BaseSink):
            name = "no_decl_sink"
            input_schema = _DummySchema


def test_basesource_subclass_without_determinism_fails_at_class_creation() -> None:
    with pytest.raises(TypeError, match=r"does not explicitly declare a `determinism`"):

        class _NoDeclSource(BaseSource):
            name = "no_decl_source"
            output_schema = _DummySchema


def test_basetransform_redeclaring_default_value_is_accepted() -> None:
    """The contract is "rebound, even to the same value" — not "different value"."""

    class _ExplicitDefaultTransform(BaseTransform):
        name = "explicit_default_transform"
        input_schema = _DummySchema
        output_schema = _DummySchema
        determinism = Determinism.DETERMINISTIC

    assert _ExplicitDefaultTransform.determinism is Determinism.DETERMINISTIC


def test_basesink_explicit_external_call_is_accepted() -> None:
    class _NetworkSink(BaseSink):
        name = "network_sink"
        input_schema = _DummySchema
        determinism = Determinism.EXTERNAL_CALL

    assert _NetworkSink.determinism is Determinism.EXTERNAL_CALL


def test_basesource_explicit_seeded_is_accepted() -> None:
    class _SeededSource(BaseSource):
        name = "seeded_source"
        output_schema = _DummySchema
        determinism = Determinism.SEEDED

    assert _SeededSource.determinism is Determinism.SEEDED


def test_intermediate_abc_subclass_must_redeclare() -> None:
    """An intermediate ABC must redeclare too — descendants don't get a pass
    just because their parent chain happens to declare. The contract is
    uniform: every subclass at every level."""

    class _IntermediateBase(BaseTransform):
        name = "intermediate"
        input_schema = _DummySchema
        output_schema = _DummySchema
        determinism = Determinism.DETERMINISTIC

    with pytest.raises(TypeError, match=r"does not explicitly declare a `determinism`"):

        class _LeafThatForgot(_IntermediateBase):
            name = "leaf_that_forgot"

    class _LeafThatRedeclares(_IntermediateBase):
        name = "leaf_that_redeclares"
        determinism = Determinism.NON_DETERMINISTIC

    assert _LeafThatRedeclares.determinism is Determinism.NON_DETERMINISTIC


def test_annotation_only_declaration_does_not_satisfy_contract() -> None:
    """``determinism: Determinism`` (annotation without value) doesn't land
    in ``__dict__`` — only ``__annotations__``.  The guard correctly rejects
    this form because at runtime the class has no actual value bound."""

    with pytest.raises(TypeError, match=r"does not explicitly declare a `determinism`"):

        class _AnnotationOnlyTransform(BaseTransform):
            name = "annotation_only"
            input_schema = _DummySchema
            output_schema = _DummySchema
            determinism: Determinism  # no value — only annotation
