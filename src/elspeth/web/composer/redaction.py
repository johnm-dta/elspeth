"""Redaction utilities for composer state serialization.

Functions that strip internal implementation details (storage paths, blob
locations) from serialized state dicts before they reach external consumers
(LLM prompts, HTTP responses, MCP tool results).

This module also exposes the shared traversal iterator
``walk_model_schema`` (spec §4.2.5) consumed by the adequacy guard
(§4.4) and the runtime walker (§4.2.6). Centralising the traversal
forecloses the rev-1 BLOCKER 2 pattern where the walker omitted
container types the spec promised and the guard could not detect.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass
from types import UnionType
from typing import Annotated, Any, Union, get_args, get_origin

from pydantic import BaseModel

REDACTED_BLOB_SOURCE_PATH = "<redacted-blob-source-path>"


class _SensitiveMarker:
    """Annotated metadata marker (spec §4.2.2).

    Carried inside ``Annotated[T, Sensitive()]`` field declarations. The
    runtime walker uses ``summarizer`` to replace sensitive values with a
    deterministic placeholder before the value reaches an external
    consumer.
    """

    __slots__ = ("summarizer",)

    def __init__(self, summarizer: Callable[[Any], str] | None = None) -> None:
        self.summarizer = summarizer


def Sensitive(*, summarizer: Callable[[Any], str] | None = None) -> _SensitiveMarker:
    """Field-level annotation requesting redaction (§4.2.2)."""
    return _SensitiveMarker(summarizer=summarizer)


@dataclass(frozen=True, slots=True)
class TraversalNode:
    """One field encountered while walking a model schema (§4.2.5).

    Freeze-guard elision (rev-3 A2 / W3):
      `metadata` is typed as `tuple[Any, ...]` to admit `_SensitiveMarker`
      instances in the metadata position. _SensitiveMarker is a regular
      (non-frozen) class - it holds a `summarizer` callable that we never
      mutate after construction. The freeze-guard CI tool
      (scripts/cicd/enforce_freeze_guards.py) only flags forbidden patterns
      in __post_init__; this dataclass intentionally has no __post_init__
      and no `freeze_fields()` call. The design assumption is:
        1. _SensitiveMarker instances are constructed once (at Annotated[...]
           definition time, module load) and never mutated;
        2. TraversalNode is produced inside walk_model_schema and discarded
           after iteration - there is no long-lived reference path that
           would expose mutation of a marker;
        3. All other metadata entries are either pydantic.FieldInfo (built-in
           immutable for our usage) or scalar/None.
      If a future change introduces stateful metadata objects, ADD a
      freeze_fields() call here AND a deep_freeze() of `metadata` - do not
      rely on the type signature alone, which `tuple[Any, ...]` does not
      constrain.
    """

    path: str
    field_type: Any
    metadata: tuple[Any, ...]
    value_provider: Callable[[dict[str, Any]], Any] | None = None


def _unwrap_annotated(field_type: Any, metadata: tuple[Any, ...]) -> tuple[Any, tuple[Any, ...]]:
    """Strip a single Annotated layer; accumulate metadata.

    Returns the inner type plus the combined metadata. ``Annotated`` may
    nest (``Annotated[Annotated[X, m1], m2]``) so the caller invokes this
    in a loop until ``get_origin`` no longer reports ``Annotated``.
    """
    if get_origin(field_type) is Annotated:
        args = get_args(field_type)
        return args[0], metadata + tuple(args[1:])
    return field_type, metadata


def _normalise(field_type: Any, metadata: tuple[Any, ...]) -> tuple[Any, tuple[Any, ...]]:
    """Repeatedly unwrap Annotated layers until a non-Annotated type remains."""
    while get_origin(field_type) is Annotated:
        field_type, metadata = _unwrap_annotated(field_type, metadata)
    return field_type, metadata


def _is_union(origin: Any) -> bool:
    return origin is Union or origin is UnionType


def _has_sensitive(metadata: tuple[Any, ...]) -> bool:
    return any(isinstance(m, _SensitiveMarker) for m in metadata)


def _count_sensitive(metadata: tuple[Any, ...]) -> int:
    count = 0
    for m in metadata:
        if isinstance(m, _SensitiveMarker):
            count += 1
    return count


# A path step describes one descent operation accumulated during the walk.
# Concrete forms:
#   ("attr", name)  - dict-attribute access by string name (BaseModel field)
#   ("list",)        - list/tuple element iteration (yields (index, item))
#   ("dict",)        - dict element iteration (yields (key, item))
# At leaf emission, the steps are baked into a value_provider closure.
_PathStep = tuple[Any, ...]


def _build_value_provider(
    steps: tuple[_PathStep, ...],
) -> Callable[[dict[str, Any]], Any]:
    """Compile a path-step list into a value_provider closure.

    Semantics (spec §4.2.5):
      - Zero container steps: returns the scalar at the path.
      - N container steps (N >= 1): returns a flat list of
        (key_or_keys, leaf_value) pairs. With one container step the key
        is a single int/str; with two or more it is a tuple of keys/indices
        in descent order.

    The closure assumes the root dict's shape conforms to the model schema.
    KeyError/TypeError on a non-conforming root is correct behaviour: the
    consumer is internal code that promised to pass a serialised state for
    the same model, and a mismatch is a bug, not external input.
    """
    container_step_count = sum(1 for s in steps if s[0] in ("list", "dict"))

    def provider(root: dict[str, Any]) -> Any:
        # Stream of (key_path, value) tuples. key_path is a tuple of keys at
        # container boundaries; for non-container traversal it remains ().
        frontier: list[tuple[tuple[Any, ...], Any]] = [((), root)]
        for step in steps:
            kind = step[0]
            if kind == "attr":
                name = step[1]
                frontier = [(keys, current[name]) for keys, current in frontier]
            elif kind == "list":
                frontier = [((*keys, idx), item) for keys, current in frontier for idx, item in enumerate(current)]
            elif kind == "dict":
                frontier = [((*keys, key), item) for keys, current in frontier for key, item in current.items()]
            else:
                raise AssertionError(f"unknown path step kind: {kind!r}")

        if container_step_count == 0:
            # Single-element frontier with empty key path.
            ((_, value),) = frontier
            return value
        if container_step_count == 1:
            return [(keys[0], value) for keys, value in frontier]
        return [(keys, value) for keys, value in frontier]

    return provider


def walk_model_schema(
    model: Any,
    *,
    with_values: bool = False,
    _path_prefix: str = "",
    _steps: tuple[_PathStep, ...] = (),
) -> Iterator[TraversalNode]:
    """Yield a TraversalNode per field; descend per §4.2.5.

    Container shapes descended into: ``list[X]``, ``tuple[X, ...]``,
    ``dict[str, X]``, ``Optional[X]``/``Union[..., X, ...]`` (where ``X``
    is a ``BaseModel``).

    When ``with_values`` is True, each node receives a ``value_provider``
    closure that, given a root dict, returns either the value at that path
    (when no container descent occurred above it) or a list of
    ``(key_or_index, value)`` pairs (when one or more container descents
    occurred above it).

    A ``_SensitiveMarker`` on a node short-circuits descent into that
    node's children (rev-5 A1): the redactor will replace the whole
    value with the summarizer output, so yielding inner nodes would
    produce paths with no reachable values in the redacted view.

    Duplicate ``_SensitiveMarker`` in a single field's Annotated tuple
    raises ``ValueError`` (the spec permits at most one).
    """
    if not (isinstance(model, type) and issubclass(model, BaseModel)):
        raise TypeError(f"walk_model_schema requires a pydantic BaseModel subclass; got {model!r}.")

    for field_name, field_info in model.model_fields.items():
        # Pydantic stores the field annotation in ``annotation`` (already
        # post-Annotated for FieldInfo purposes); the surrounding metadata
        # is exposed via ``metadata``. Reassemble the Annotated view so a
        # single normalisation path handles all entry forms.
        raw_type = field_info.annotation
        raw_metadata: tuple[Any, ...] = tuple(field_info.metadata)
        field_type, metadata = _normalise(raw_type, raw_metadata)

        sensitive_count = _count_sensitive(metadata)
        if sensitive_count > 1:
            raise ValueError(f"Field {field_name!r} has {sensitive_count} Sensitive() markers; spec §4.2.5 permits at most one per field.")

        path = f"{_path_prefix}{field_name}" if _path_prefix == "" else f"{_path_prefix}.{field_name}"
        field_steps = (*_steps, ("attr", field_name))

        yield from _walk_type(
            field_type=field_type,
            metadata=metadata,
            path=path,
            with_values=with_values,
            steps=field_steps,
        )


def _walk_type(
    *,
    field_type: Any,
    metadata: tuple[Any, ...],
    path: str,
    with_values: bool,
    steps: tuple[_PathStep, ...],
) -> Iterator[TraversalNode]:
    """Yield TraversalNodes for a single typed slot at ``path``.

    Handles Annotated unwrap (already done by caller for the field-level
    entry), Union/Optional fan-out, container descent, BaseModel descent,
    and scalar emission. Sensitive short-circuit applies here: if
    ``metadata`` contains a marker we yield the current node and stop.
    """
    # Re-normalise in case a nested Annotated slipped through (Union arms
    # commonly carry their own Annotated wrappers).
    field_type, metadata = _normalise(field_type, metadata)

    origin = get_origin(field_type)

    # Yield-and-return guard: a Sensitive marker on this slot suppresses
    # descent (rev-5 A1).
    if _has_sensitive(metadata):
        yield TraversalNode(
            path=path,
            field_type=field_type,
            metadata=metadata,
            value_provider=_build_value_provider(steps) if with_values else None,
        )
        return

    # Union/Optional: descend into every non-None arm at the SAME path.
    if _is_union(origin):
        for arm in get_args(field_type):
            if arm is type(None):
                continue
            arm_type, arm_metadata = _normalise(arm, ())
            # A scalar arm with a Sensitive marker MUST be emitted (rev-3 W8a):
            # Optional[Annotated[str, Sensitive()]] is a common shape and the
            # marker would otherwise be silently dropped.
            if not _is_descendable(arm_type) and not _has_sensitive(arm_metadata):
                continue
            yield from _walk_type(
                field_type=arm_type,
                metadata=arm_metadata,
                path=path,
                with_values=with_values,
                steps=steps,
            )
        return

    # Container descent.
    #
    # NOTE: bare `list` and bare `dict` (unparameterised) have get_origin() == None
    # in Python's typing machinery, so they do NOT match the `origin in (list, tuple)`
    # or `origin is dict` branches below. Guard them explicitly here so they raise
    # ValueError instead of silently falling through to the scalar/leaf path.
    if field_type is list:
        raise ValueError(f"Field at path {path!r} uses a bare list; spec §4.2.5 requires a parameterised element type.")
    if field_type is dict:
        raise ValueError(f"Field at path {path!r} uses an unparameterised dict; spec §4.2.5 requires dict[str, X].")

    if origin in (list, tuple):
        args = get_args(field_type)
        if not args:
            raise ValueError(f"Field at path {path!r} uses a bare list/tuple; spec §4.2.5 requires a parameterised element type.")
        # Fixed-length heterogeneous tuples (e.g. tuple[int, str]) are not supported.
        # The only tuple form that can be descended into is the variable-length
        # tuple[X, ...] form, identified by exactly two args where the second is
        # Ellipsis.  Any other multi-arg tuple silently drops all but the first
        # element type — the exact silent-pass failure the walker was written to
        # prevent.
        if origin is tuple and not (len(args) == 2 and args[1] is Ellipsis):
            raise ValueError(
                f"Field at path {path!r} uses a fixed-length tuple; spec §4.2.5 supports only the variable-length form tuple[X, ...]."
            )
        element_type = args[0]
        yield from _walk_type(
            field_type=element_type,
            metadata=(),
            path=f"{path}[*]",
            with_values=with_values,
            steps=(*steps, ("list",)),
        )
        return

    if origin is dict:
        args = get_args(field_type)
        if len(args) != 2:
            raise ValueError(f"Field at path {path!r} uses an unparameterised dict; spec §4.2.5 requires dict[str, X].")
        key_type, element_type = args
        if key_type is not str:
            raise ValueError(f"Field at path {path!r} uses a non-str dict key ({key_type!r}); spec §4.2.5 supports dict[str, X] only.")
        yield from _walk_type(
            field_type=element_type,
            metadata=(),
            path=f"{path}{{*}}",
            with_values=with_values,
            steps=(*steps, ("dict",)),
        )
        return

    # BaseModel descent.
    if isinstance(field_type, type) and issubclass(field_type, BaseModel):
        yield from walk_model_schema(
            field_type,
            with_values=with_values,
            _path_prefix=path,
            _steps=steps,
        )
        return

    # Scalar/Any/object leaf: yield a node and stop.
    yield TraversalNode(
        path=path,
        field_type=field_type,
        metadata=metadata,
        value_provider=_build_value_provider(steps) if with_values else None,
    )


def _is_descendable(t: Any) -> bool:
    """True iff ``t`` is a BaseModel, parameterised container, or Union.

    Used to decide whether a Union arm warrants descent. Scalar arms
    (``str``, ``int``, ``bool``) and ``None`` are skipped.
    """
    if isinstance(t, type) and issubclass(t, BaseModel):
        return True
    origin = get_origin(t)
    if _is_union(origin):
        return True
    return origin in (list, tuple, dict)


def redact_source_storage_path(state_dict: dict[str, Any]) -> dict[str, Any]:
    """Redact internal storage paths from a serialized state dict.

    When source options contain a ``blob_ref``, ``path`` points at an
    internal storage location that must not be exposed to agents or users.
    Preserve the key with a sentinel value so external consumers can still
    tell that the source path contract is satisfied.

    Returns a shallow copy with source options redacted. Does not mutate
    the input dict.
    """
    source = state_dict.get("source")
    if source is None:
        return state_dict

    options = source.get("options")
    if options is None or "blob_ref" not in options:
        return state_dict

    # Shallow copy the chain to avoid mutating the original
    redacted = dict(state_dict)
    redacted_source = dict(source)
    redacted_options = dict(options)
    if "path" in redacted_options:
        redacted_options["path"] = REDACTED_BLOB_SOURCE_PATH
    redacted_source["options"] = redacted_options
    redacted["source"] = redacted_source
    return redacted
