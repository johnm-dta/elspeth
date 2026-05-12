"""Tests for the shared traversal iterator (spec §4.2.5).

Covers every container shape the iterator must descend into. Both the
adequacy guard and the runtime walker consume this iterator; gaps here
silently allow gaps in either consumer. See plan-review B2 in
docs/superpowers/plans/2026-04-30-composer-progress-persistence-phase-2-redaction.review.json.
"""

from __future__ import annotations

from typing import Annotated, Any

import pytest
from pydantic import BaseModel

from elspeth.web.composer.redaction import (
    Sensitive,
    TraversalNode,
    _SensitiveMarker,
    walk_model_schema,
)


class _FlatModel(BaseModel):
    ok: str
    secret: Annotated[str, Sensitive()]


class _NestedModel(BaseModel):
    header: str
    payload: _FlatModel


class _ListModel(BaseModel):
    items: list[_FlatModel]


class _DictModel(BaseModel):
    lookup: dict[str, _FlatModel]


class _TupleModel(BaseModel):
    pair: tuple[_FlatModel, ...]


class _OptionalModel(BaseModel):
    maybe: _FlatModel | None = None


class _UnionModel(BaseModel):
    either: _FlatModel | _NestedModel


class _MarkerLastModel(BaseModel):
    tail: Annotated[str, "irrelevant", Sensitive()]


def _paths(nodes: list[TraversalNode]) -> list[str]:
    return [n.path for n in nodes]


def _has_marker(nodes: list[TraversalNode], path: str) -> bool:
    return any(n.path == path and any(isinstance(m, _SensitiveMarker) for m in n.metadata) for n in nodes)


def test_walks_flat_model() -> None:
    nodes = list(walk_model_schema(_FlatModel))
    assert _paths(nodes) == ["ok", "secret"]
    assert not _has_marker(nodes, "ok")
    assert _has_marker(nodes, "secret")


def test_descends_into_nested_basemodel() -> None:
    nodes = list(walk_model_schema(_NestedModel))
    assert "header" in _paths(nodes)
    assert "payload.ok" in _paths(nodes)
    assert "payload.secret" in _paths(nodes)
    assert _has_marker(nodes, "payload.secret")


def test_descends_into_list_of_basemodel() -> None:
    nodes = list(walk_model_schema(_ListModel))
    assert "items[*].ok" in _paths(nodes)
    assert "items[*].secret" in _paths(nodes)
    assert _has_marker(nodes, "items[*].secret")


def test_descends_into_dict_of_basemodel() -> None:
    nodes = list(walk_model_schema(_DictModel))
    assert "lookup{*}.ok" in _paths(nodes)
    assert "lookup{*}.secret" in _paths(nodes)
    assert _has_marker(nodes, "lookup{*}.secret")


def test_descends_into_tuple_of_basemodel() -> None:
    nodes = list(walk_model_schema(_TupleModel))
    assert "pair[*].ok" in _paths(nodes)
    assert "pair[*].secret" in _paths(nodes)
    assert _has_marker(nodes, "pair[*].secret")


def test_descends_into_optional_basemodel_arm() -> None:
    nodes = list(walk_model_schema(_OptionalModel))
    assert "maybe.ok" in _paths(nodes)
    assert "maybe.secret" in _paths(nodes)
    assert _has_marker(nodes, "maybe.secret")


def test_descends_into_every_union_arm() -> None:
    nodes = list(walk_model_schema(_UnionModel))
    paths = _paths(nodes)
    # Both arms walked
    assert "either.ok" in paths
    assert "either.secret" in paths
    assert "either.payload.secret" in paths


def test_marker_position_in_annotated_does_not_matter() -> None:
    """Sensitive() may appear anywhere in the Annotated tuple, not just args[0]."""
    nodes = list(walk_model_schema(_MarkerLastModel))
    assert _has_marker(nodes, "tail")


def test_three_level_nesting_descends_fully() -> None:
    class L1(BaseModel):
        inner: _NestedModel

    class L0(BaseModel):
        outer: L1

    nodes = list(walk_model_schema(L0))
    assert "outer.inner.payload.secret" in _paths(nodes)
    assert _has_marker(nodes, "outer.inner.payload.secret")


def test_any_field_raises_at_walk_time() -> None:
    """Any-typed fields are inspection-resistant; the iterator surfaces them so
    the adequacy guard can fail with a precise error message."""

    class M(BaseModel):
        junk: Any

    nodes = list(walk_model_schema(M))
    junk_node = next(n for n in nodes if n.path == "junk")
    assert junk_node.field_type is Any


def test_value_provider_extracts_from_root_dict_when_with_values() -> None:
    """walker mode: each node's value_provider returns the value at the
    node's path given the root dict."""
    nodes = list(walk_model_schema(_NestedModel, with_values=True))
    root = {"header": "h", "payload": {"ok": "v", "secret": "S"}}
    ok_node = next(n for n in nodes if n.path == "payload.ok")
    assert ok_node.value_provider is not None
    assert ok_node.value_provider(root) == "v"
    secret_node = next(n for n in nodes if n.path == "payload.secret")
    assert secret_node.value_provider is not None
    assert secret_node.value_provider(root) == "S"


def test_value_provider_handles_list_index_descent() -> None:
    """For list/dict/tuple element descents the provider returns a sequence
    of (key_or_index, value) pairs; the walker iterates."""
    nodes = list(walk_model_schema(_ListModel, with_values=True))
    root = {"items": [{"ok": "a", "secret": "X"}, {"ok": "b", "secret": "Y"}]}
    secret_node = next(n for n in nodes if n.path == "items[*].secret")
    assert secret_node.value_provider is not None
    values = list(secret_node.value_provider(root))
    assert values == [(0, "X"), (1, "Y")]


def test_value_provider_handles_dict_key_descent() -> None:
    nodes = list(walk_model_schema(_DictModel, with_values=True))
    root = {"lookup": {"first": {"ok": "a", "secret": "X"}, "second": {"ok": "b", "secret": "Y"}}}
    secret_node = next(n for n in nodes if n.path == "lookup{*}.secret")
    assert secret_node.value_provider is not None
    values = sorted(secret_node.value_provider(root))
    assert values == [("first", "X"), ("second", "Y")]


def test_with_values_false_yields_no_value_provider() -> None:
    """Adequacy-guard mode: walker does not need value extraction."""
    nodes = list(walk_model_schema(_FlatModel))
    assert all(n.value_provider is None for n in nodes)


def test_walk_duplicate_sensitive_markers() -> None:
    """Duplicate _SensitiveMarker in one field's Annotated tuple raises ValueError
    (spec §4.2.5 promises this; rev-2 M_adequacy quality MAJOR-1 gap A)."""

    class _DuplicateModel(BaseModel):
        bad: Annotated[str, Sensitive(), Sensitive()]

    with pytest.raises(ValueError, match="bad"):
        list(walk_model_schema(_DuplicateModel))


def test_walk_list_of_list_of_basemodel() -> None:
    """Iterator descends through two list levels (rev-2 quality MAJOR-1 gap B)."""

    class _Inner(BaseModel):
        secret: Annotated[str, Sensitive()]

    class _Outer(BaseModel):
        matrix: list[list[_Inner]]

    nodes = list(walk_model_schema(_Outer))
    paths = {n.path for n in nodes}
    assert "matrix[*][*].secret" in paths
    assert any(any(isinstance(m, _SensitiveMarker) for m in n.metadata) for n in nodes if n.path == "matrix[*][*].secret")


def test_walk_field_plus_annotated_combined() -> None:
    """Sensitive() marker is detected regardless of FieldInfo position in Annotated
    metadata tuple (rev-2 quality MAJOR-1 gap C)."""
    from pydantic import Field

    class _FieldAnnotatedModel(BaseModel):
        combined: Annotated[str, Field(description="desc"), Sensitive()]

    nodes = list(walk_model_schema(_FieldAnnotatedModel))
    target = next((n for n in nodes if n.path == "combined"), None)
    assert target is not None
    assert any(isinstance(m, _SensitiveMarker) for m in target.metadata)


def test_walk_optional_annotated_scalar_arm() -> None:
    """Optional[Annotated[str, Sensitive()]] scalar-arm coverage (rev-3 W8a / rev-4 W8a).

    The earlier _OptionalModel test covers Optional[<BaseModel>]; this test
    covers the scalar arm Optional[Annotated[<scalar>, Sensitive()]] which
    goes through a different unwrap path (Optional[X] -> Union[X, None] ->
    Annotated[str, Sensitive()] -> str). The marker must survive the Optional
    unwrap; if it doesn't, every Optional[Annotated[scalar, Sensitive()]]
    field is silently treated as non-sensitive at the runtime walker.
    """

    class _OptionalScalarModel(BaseModel):
        maybe_secret: Annotated[str, Sensitive()] | None = None

    nodes = list(walk_model_schema(_OptionalScalarModel))
    target = next((n for n in nodes if n.path == "maybe_secret"), None)
    assert target is not None, (
        "Optional[Annotated[str, Sensitive()]] field did not appear in walk output. "
        "Walker must descend into the non-None Union arm and yield a node for "
        "the wrapped scalar."
    )
    assert any(isinstance(m, _SensitiveMarker) for m in target.metadata), (
        "Optional unwrap dropped the _SensitiveMarker. The marker MUST survive "
        "Optional[Annotated[T, Sensitive()]] unwrapping; otherwise every "
        "Optional sensitive scalar in the manifest is silently non-sensitive."
    )


def test_walk_optional_non_sensitive_scalar_arm() -> None:
    """``Optional[str]`` (no Sensitive marker) emits a leaf at the field path.

    Closes rev-3 M1 floor-check landing-pin: the floor-check (every
    ``model_fields`` key appears as a walk root) was passing accidentally
    for ``set_source`` because that model has no Optional non-Sensitive
    scalar fields.  Task 13 / Wave 2 brings the first such field
    (``create_blob.description: str | None = None``) into the manifest;
    the walker had been silently dropping it because the legacy "skip
    non-descendable scalar Union arms" rule (introduced to suppress
    spurious yields for ``str | _Model | bool`` per
    :func:`test_walk_three_arm_union`) was too broad.

    The carve-out is structural: a Union with exactly one non-None arm is
    an Optional, single-armed by construction, so duplicate-yield risk
    does not apply.  Multi-arm scalar Unions (``str | int | bool``) still
    skip scalar arms — :func:`test_walk_three_arm_union` keeps that pin.
    """

    class _OptionalNonSensitiveScalarModel(BaseModel):
        description: str | None = None

    nodes = list(walk_model_schema(_OptionalNonSensitiveScalarModel))
    paths = [n.path for n in nodes]
    assert "description" in paths, (
        "Optional[str] field with no Sensitive marker did not appear in walk "
        "output. Walker must emit a leaf at the field path so the adequacy "
        "guard's walker-completeness floor-check (rev-3 M1) holds."
    )
    # The leaf must have NO Sensitive marker — this field is structural,
    # not sensitive.  A future change that conflates "emit a leaf" with
    # "mark as sensitive" would fail this assertion.
    description_node = next(n for n in nodes if n.path == "description")
    assert not any(isinstance(m, _SensitiveMarker) for m in description_node.metadata)


def test_walk_three_arm_union() -> None:
    """Iterator descends into BaseModel arm of a 3-arm Union; skips non-BaseModel
    arms cleanly without spurious yields (rev-2 quality MAJOR-1 gap D)."""

    class _InnerWithSecret(BaseModel):
        secret: Annotated[str, Sensitive()]

    class _UnionOuter(BaseModel):
        field: str | _InnerWithSecret | bool

    nodes = list(walk_model_schema(_UnionOuter))
    paths = {n.path for n in nodes}
    # Must find the nested secret via the BaseModel arm.
    assert "field.secret" in paths
    # Must NOT find spurious scalar-arm yields.
    assert not any(p in paths for p in ("field[str]", "field[bool]"))


def test_walk_bare_list_raises_value_error() -> None:
    """Bare list (no element type) raises ValueError mentioning the field path."""

    class M(BaseModel):
        x: list  # type: ignore[type-arg]

    with pytest.raises(ValueError, match="x"):
        list(walk_model_schema(M))


def test_walk_unparameterised_dict_raises_value_error() -> None:
    """Bare dict (no type parameters) raises ValueError."""

    class M(BaseModel):
        x: dict  # type: ignore[type-arg]

    with pytest.raises(ValueError, match="x"):
        list(walk_model_schema(M))


def test_walk_non_str_dict_key_raises_value_error() -> None:
    """dict[int, str] raises ValueError mentioning the field path."""

    class M(BaseModel):
        x: dict[int, str]

    with pytest.raises(ValueError, match="x"):
        list(walk_model_schema(M))


def test_walk_fixed_length_tuple_raises_value_error() -> None:
    """tuple[int, str] (fixed-length heterogeneous) raises ValueError mentioning
    the path and the requirement for variable-length form tuple[X, ...]."""

    class M(BaseModel):
        x: tuple[int, str]

    with pytest.raises(ValueError, match="x"):
        list(walk_model_schema(M))


def test_walk_short_circuits_under_sensitive_container() -> None:
    """A Sensitive marker on a container suppresses descent into inner Sensitive markers.

    Closes rev-5 architecture A1. The redactor replaces the whole container's
    value with the summarizer output; inner markers below that point have no
    reachable values in the redacted view. The walker therefore must NOT yield
    them, or the property test's value_provider(redacted) call at the inner
    path will attempt to iterate the summarizer's string output and raise
    TypeError.
    """

    class _InnerWithSecret(BaseModel):
        inner_secret: Annotated[str, Sensitive()]

    class _OuterSensitiveContainer(BaseModel):
        outer: Annotated[list[_InnerWithSecret], Sensitive(summarizer=lambda v: f"<list:{len(v)}>")]

    nodes = list(walk_model_schema(_OuterSensitiveContainer))
    paths = {n.path for n in nodes}
    # The outer container is yielded with its marker.
    assert "outer" in paths
    outer = next(n for n in nodes if n.path == "outer")
    assert any(isinstance(m, _SensitiveMarker) for m in outer.metadata)
    # The inner secret is NOT yielded - the outer Sensitive short-circuits descent.
    assert "outer[*].inner_secret" not in paths, (
        "Walker descended past a Sensitive-marked container into an inner "
        "Sensitive node. This is forbidden because the redactor replaces the "
        "whole container with the summarizer output; the inner path has no "
        "reachable value in the redacted view and yielding it would cause "
        "the property test to raise TypeError on value_provider(redacted)."
    )
