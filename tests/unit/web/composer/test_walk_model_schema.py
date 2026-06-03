"""Tests for the shared traversal iterator (spec §4.2.5).

Covers every container shape the iterator must descend into. Both the
adequacy guard and the runtime walker consume this iterator; gaps here
silently allow gaps in either consumer. The current redaction design records
the original B2 finding and its resolution in
docs/superpowers/specs/2026-04-30-composer-progress-persistence-design.md §12.2.
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


def test_value_provider_returns_none_for_path_through_optional_none() -> None:
    """A path that descends through an ``Optional[Model] = None`` intermediate
    returns ``None`` rather than raising (rev-4 sibling-API parity).

    The walker emits inner paths for fields whose declared type is
    ``OptionalModel | None`` (schema-completeness contract); at runtime the
    intermediate may be ``None`` for a given example.  The ``value_provider``
    closure mirrors the sibling ``substitute_provider``'s None-skip pattern
    (which has been correct since Task 8) so the read API does not crash on
    a conforming-to-schema input that happens to populate the Optional with
    ``None``.

    This is the load-bearing regression test for the rev-4 sibling-API
    alignment closing the property test's TypeError-on-None failure mode.
    """
    nodes = list(walk_model_schema(_OptionalModel, with_values=True))
    secret_node = next(n for n in nodes if n.path == "maybe.secret")
    assert secret_node.value_provider is not None
    # maybe = None: the inner path is unreachable; provider returns None.
    assert secret_node.value_provider({"maybe": None}) is None
    # maybe populated: provider returns the inner value as usual.
    assert secret_node.value_provider({"maybe": {"ok": "v", "secret": "S"}}) == "S"


def test_value_provider_returns_empty_list_for_container_through_optional_none() -> None:
    """A container-descent path through an ``Optional[list[...]] = None``
    intermediate returns an empty list rather than raising (rev-4 sibling-API
    parity).
    """

    class _OptionalListModel(BaseModel):
        items: list[_FlatModel] | None = None

    nodes = list(walk_model_schema(_OptionalListModel, with_values=True))
    secret_node = next(n for n in nodes if n.path == "items[*].secret")
    assert secret_node.value_provider is not None
    # items = None: provider returns an empty pair-list.
    assert list(secret_node.value_provider({"items": None})) == []
    # items populated: provider returns the per-element pairs as usual.
    assert sorted(secret_node.value_provider({"items": [{"ok": "a", "secret": "X"}, {"ok": "b", "secret": "Y"}]})) == [
        (0, "X"),
        (1, "Y"),
    ]


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


def test_value_provider_treats_missing_union_arm_leaf_as_unreachable() -> None:
    """Union[ModelA, ModelB] paths are unreachable when the active arm lacks the leaf."""

    class _SuccessPayload(BaseModel):
        content: Annotated[str, Sensitive()]

    class _FailurePayload(BaseModel):
        error: str

    class _Envelope(BaseModel):
        data: _SuccessPayload | _FailurePayload

    nodes = list(walk_model_schema(_Envelope, with_values=True))
    content_node = next(n for n in nodes if n.path == "data.content")
    assert content_node.value_provider is not None
    assert content_node.substitute_provider is not None

    failure_root: dict[str, Any] = {"data": {"error": "not found"}}
    assert content_node.value_provider(failure_root) is None

    def _must_not_be_called(_value: Any) -> Any:
        raise AssertionError("missing union-arm leaf should not be substituted")

    content_node.substitute_provider(failure_root, _must_not_be_called)
    assert failure_root == {"data": {"error": "not found"}}

    success_root: dict[str, Any] = {"data": {"content": "secret"}}
    assert content_node.value_provider(success_root) == "secret"
    content_node.substitute_provider(success_root, lambda value: f"<{value}>")
    assert success_root == {"data": {"content": "<secret>"}}


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


def test_walk_emits_substitute_provider_only_for_attr_path() -> None:
    """``substitute_provider`` is populated iff the node's final step is ``("attr", name)``.

    Pins the ``needs_substitute = bool(steps) and steps[-1][0] == "attr"`` gate
    at :func:`elspeth.web.composer.redaction._walk_type` (currently
    ``redaction.py`` line 524) for the scalar/leaf branch.  Container-element
    scalar leaves (``list[str]``, ``dict[str, str]``) end their step chain
    with ``("list",)``/``("dict",)`` — there is no named destination to
    substitute INTO, so ``substitute_provider`` MUST remain ``None``.

    Without the gate, container-final scalar leaves would receive a
    ``substitute_provider`` whose final-attr-step assertion at
    :func:`_build_substitute_provider` (line 269) would raise
    ``AssertionError`` during *node construction* — masking the design
    intent and making the failure mode an opaque crash rather than a
    cleanly-typed ``None``.

    Test-shape coverage:

      * ``tags``: ``Sensitive[list[str]]`` — Sensitive yield-and-return branch;
        steps end with ``("attr", "tags")`` so substitute_provider MUST be set.
      * ``plain_tags[*]``: bare ``list[str]`` — container-descent then scalar
        leaf; steps end with ``("list",)`` so substitute_provider MUST be None.
      * ``name``: scalar field — steps end with ``("attr", "name")`` so
        substitute_provider MUST be set.

    This combination exercises both sides of the gate in one walk.
    """

    class _MixedModel(BaseModel):
        tags: Annotated[list[str], Sensitive(summarizer=lambda v: f"<n={len(v)}>")]
        plain_tags: list[str]
        name: str

    nodes = list(walk_model_schema(_MixedModel, with_values=True))
    by_path = {n.path: n for n in nodes}

    # All three paths must appear.
    assert "tags" in by_path, "Sensitive[list[str]] field did not appear at its field path."
    assert "plain_tags[*]" in by_path, "Container-element leaf for plain list[str] did not appear."
    assert "name" in by_path, "Scalar field did not appear."

    # Attr-final nodes: substitute_provider MUST be populated.
    assert by_path["tags"].substitute_provider is not None, (
        "Sensitive[list[str]] node at attr-final path 'tags' has no "
        "substitute_provider. The redactor cannot substitute the summarizer "
        "output into the field without one."
    )
    assert by_path["name"].substitute_provider is not None, (
        "Scalar attr-final node 'name' has no substitute_provider. "
        "All attr-final leaves must carry a sibling substitute_provider "
        "alongside value_provider."
    )

    # Container-element-final nodes: substitute_provider MUST be None.
    assert by_path["plain_tags[*]"].substitute_provider is None, (
        "Container-element scalar leaf 'plain_tags[*]' was given a "
        "substitute_provider. Its step chain ends with ('list',) — there "
        "is no named destination to substitute into. If the _walk_type "
        "gate at line 524 is removed, this node would receive a provider "
        "whose construction at _build_substitute_provider line 269 would "
        "raise AssertionError on the non-attr final step."
    )

    # Broader assertion: across ALL emitted nodes, the gate must hold.
    for node in nodes:
        # Reconstruct whether the node's terminal step is ('attr', name) by
        # inspecting the path: container-descent path-segments end with
        # '[*]' or '{*}', so any node whose path ends with one of those
        # markers is container-final and MUST have substitute_provider=None.
        is_container_final = node.path.endswith("[*]") or node.path.endswith("}")
        if is_container_final:
            assert node.substitute_provider is None, (
                f"Node at container-final path {node.path!r} unexpectedly carries a "
                "substitute_provider; the _walk_type gate (line 524) is broken."
            )
        else:
            assert node.substitute_provider is not None, f"Node at attr-final path {node.path!r} is missing its substitute_provider."


def test_walk_handles_optional_basemodel_with_none_runtime_value() -> None:
    """Optional[BaseModel] with inner Sensitive — sibling providers tolerate runtime None.

    Pins two cooperating guards that together let a Sensitive leaf inside an
    ``Optional[InnerModel] = None`` field be walked without crashing when the
    runtime example happens to have ``None`` at the Optional slot:

      1. :func:`_build_substitute_provider`'s None-skip at the attr prefix step
         (``redaction.py`` lines 300-303: ``if value is None: continue``).
         Without this skip, the closure would attempt ``current[leaf_name]``
         on a ``None`` intermediate and raise ``TypeError``.

      2. :func:`_build_value_provider`'s empty-frontier scalar return
         (``redaction.py`` lines 230-231: ``if not frontier: return None``).
         The walker emits the inner Sensitive path because schema-completeness
         requires it (rev-3 M1 floor-check); at runtime the inner path is
         unreachable when the Optional resolves to ``None``, and the closure
         must return ``None`` rather than raise.

    Both guards are documented in the docstring (redaction.py lines 167-187
    for value_provider, 280-292 for substitute_provider) as the rev-4
    sibling-API parity contract.

    If guard (1) regresses: ``substitute_provider({'optional_nested': None}, ...)``
    raises ``TypeError`` on the ``None[leaf_name]`` access — this test would
    fail at the substitute_provider call line.

    If guard (2) regresses: ``value_provider({'optional_nested': None})``
    would either raise ``ValueError`` on the empty-frontier unpack
    ``((_, value),) = frontier`` or raise ``TypeError`` earlier on the
    ``None[name]`` step — this test would fail at the value_provider call
    line.
    """

    class _InnerModel(BaseModel):
        sensitive_field: Annotated[str, Sensitive(summarizer=lambda v: f"<len={len(v)}>")]

    class _OuterWithOptionalInner(BaseModel):
        optional_nested: _InnerModel | None = None

    nodes = list(walk_model_schema(_OuterWithOptionalInner, with_values=True))
    sensitive_node = next(
        (n for n in nodes if n.path == "optional_nested.sensitive_field"),
        None,
    )
    assert sensitive_node is not None, (
        "Walker did not emit a node for the inner Sensitive field through the "
        "Optional[InnerModel] arm. The schema-completeness contract "
        "(rev-3 M1 floor-check) requires this path even though it is "
        "unreachable at runtime when optional_nested is None."
    )
    assert sensitive_node.value_provider is not None
    assert sensitive_node.substitute_provider is not None

    # Pin guard (2): scalar case with unreachable path returns None, not raise.
    # The docstring at redaction.py lines 181-187 ("For the scalar
    # (zero-container) case with an unreachable path the closure returns
    # None") is the load-bearing contract.
    root_with_none: dict[str, Any] = {"optional_nested": None}
    assert sensitive_node.value_provider(root_with_none) is None, (
        "value_provider for inner-Sensitive path through Optional[Model]=None "
        "did not return None on an unreachable scalar leaf. The empty-frontier "
        "scalar return at _build_value_provider lines 230-231 has regressed."
    )

    # Pin guard (1): substitute_provider must be a no-op (no TypeError) when
    # the intermediate Optional is None at runtime.  We pass a transform that
    # would itself raise if called, proving the closure short-circuited at
    # the None-skip rather than reaching the leaf.
    def _must_not_be_called(_value: Any) -> Any:
        raise AssertionError(
            "transform was invoked despite the Optional intermediate being None; "
            "the _build_substitute_provider None-skip guard (lines 300-303) "
            "has regressed."
        )

    # No TypeError, no AssertionError from the transform: clean no-op.
    sensitive_node.substitute_provider(root_with_none, _must_not_be_called)
    # And the root remains untouched (no spurious mutation).
    assert root_with_none == {"optional_nested": None}

    # Sanity: when the Optional IS populated, both providers behave normally
    # (this guards against the regression-direction "guard is too aggressive
    # and skips even when the value is present").
    root_populated: dict[str, Any] = {"optional_nested": {"sensitive_field": "abcdef"}}
    assert sensitive_node.value_provider(root_populated) == "abcdef"
    captured: list[Any] = []

    def _capture(value: Any) -> str:
        captured.append(value)
        return f"<replaced:{value}>"

    sensitive_node.substitute_provider(root_populated, _capture)
    assert captured == ["abcdef"]
    assert root_populated == {"optional_nested": {"sensitive_field": "<replaced:abcdef>"}}
