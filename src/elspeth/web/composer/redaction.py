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

import copy
import json
from collections.abc import Callable, Iterator, Mapping
from collections.abc import Set as AbstractSet
from dataclasses import dataclass, field
from types import MappingProxyType, UnionType
from typing import Annotated, Any, Literal, Union, get_args, get_origin

from pydantic import BaseModel, BeforeValidator, ConfigDict, Field

from elspeth.contracts.blobs import AllowedMimeType
from elspeth.contracts.composer_interpretation import InterpretationKind
from elspeth.contracts.errors import AuditIntegrityError
from elspeth.contracts.freeze import freeze_fields
from elspeth.web.composer.redaction_telemetry import RedactionTelemetry

REDACTED_BLOB_SOURCE_PATH = "<redacted-blob-source-path>"
_REDACTED_OPTION_VALUE = "<redacted-option-value>"

# Fixed sentinel for response keys that appear in the input but are not
# declared in the manifest entry's known_response_keys or
# sensitive_response_keys sets.  The value is a closed constant — callers
# MUST compare by ==, not by prefix or regex.  No length disclosure
# (closes W6 / spec §8.1 RSK-03 weak echo).
REDACTED_UNKNOWN_RESPONSE_KEY = "<redacted-unknown-response-key>"

# Fixed sentinel for arguments that appear in the input but are not declared in
# a manifest entry's optional known_argument_keys allowlist. Unknown key names
# are removed too; they are LLM-controlled text and may themselves carry
# sensitive payload. Most declarative tools still preserve historical
# passthrough behavior; tools that persist untyped, LLM-supplied argument dicts
# can opt into this fail-closed mode with redact_unknown_argument_keys=True.
REDACTED_UNKNOWN_ARGUMENT_KEY = "<redacted-unknown-argument-key>"
REDACTED_UNKNOWN_ARGUMENTS_FIELD = "_unknown_arguments"

# Sentinel applied to Sensitive[T] fields whose _SensitiveMarker carries no
# summarizer callable.  A field declared ``Annotated[T, Sensitive()]`` (with
# no ``summarizer=`` keyword) receives this value instead of the raw payload.
# It is distinct from REDACTED_UNKNOWN_RESPONSE_KEY so audit consumers can
# distinguish "known sensitive, no summarizer" from "unknown key, fail-closed".
REDACTED_SENSITIVE_NO_SUMMARIZER = "<redacted>"


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

    ``substitute_provider``: when ``with_values=True``, a sibling
    closure to ``value_provider`` that performs in-place substitution at
    every reachable leaf for this node's path. Signature:
    ``(root: dict, transform: Callable[[Any], Any]) -> None``. The closure
    walks the root dict by the same path steps as ``value_provider`` and
    replaces each leaf value with ``transform(old_value)``. Raises from
    ``transform`` propagate; the caller is responsible for atomicity (build
    on a local; only return on success).
    """

    path: str
    field_type: Any
    metadata: tuple[Any, ...]
    value_provider: Callable[[dict[str, Any]], Any] | None = None
    substitute_provider: Callable[[dict[str, Any], Callable[[Any], Any]], None] | None = None


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
      - Zero container steps: returns the scalar at the path, OR ``None`` if
        the path descends through an Optional sub-model whose value is
        ``None`` at runtime.
      - N container steps (N >= 1): returns a flat list of
        (key_or_keys, leaf_value) pairs (possibly empty if any intermediate
        Optional sub-model is ``None`` at runtime).  With one container step
        the key is a single int/str; with two or more it is a tuple of
        keys/indices in descent order.

    **Optional-intermediate disposition (rev-4 sibling-API parity).**
    ``("attr", name)`` steps descend dict-by-name; the walker emits inner
    paths for fields whose declared type is ``OptionalModel | None`` so the
    schema-completeness contract holds.  At runtime the value at the
    intermediate slot may be ``None`` (Pydantic's ``model_dump`` reflects
    ``Optional[Model] = None`` as ``None``); the inner path is unreachable
    in that example.  This closure skips such frontier entries — mirroring
    the sibling ``_build_substitute_provider``'s None-skip pattern (lines
    250-271). The earlier docstring
    rule ("KeyError/TypeError on non-conforming root is correct
    behaviour") predates Optional sub-model coverage and is no
    longer load-bearing — a None intermediate is conforming-to-schema by
    Pydantic's definition.

    For the scalar (zero-container) case with an unreachable path the
    closure returns ``None``: the property test's only-allowed skip rule
    (``raw_value is None: continue``) treats this as semantically
    consistent with "no sensitive data at this path for this example."
    For the container case the closure returns an empty list of pairs:
    the test's container branch handles this via ``dict([])`` → empty
    key set → no per-key comparisons.

    Non-Optional non-conforming roots (e.g., a missing required key) still
    raise KeyError — that remains a bug-in-caller signal.
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
                # Optional / union-arm disposition: skip frontier entries
                # whose ``current`` is None, and mapping entries where the
                # active runtime arm lacks this field. The walker emits inner
                # paths through Optional and Union sub-models so schema
                # completeness holds; at runtime a particular example may
                # select a different union arm, making the path unreachable.
                # Sibling parity with _build_substitute_provider.
                next_frontier: list[tuple[tuple[Any, ...], Any]] = []
                for keys, current in frontier:
                    if current is None:
                        continue
                    if isinstance(current, Mapping) and name not in current:
                        continue
                    next_frontier.append((keys, current[name]))
                frontier = next_frontier
            elif kind == "list":
                # Defensive against a None intermediate (Optional[list[...]]
                # = None at runtime): treat as empty iteration.
                frontier = [((*keys, idx), item) for keys, current in frontier if current is not None for idx, item in enumerate(current)]
            elif kind == "dict":
                # Defensive against a None intermediate (Optional[dict[...]]
                # = None at runtime): treat as empty iteration.
                frontier = [((*keys, key), item) for keys, current in frontier if current is not None for key, item in current.items()]
            else:
                raise AssertionError(f"unknown path step kind: {kind!r}")

        if container_step_count == 0:
            # Scalar case.  If the frontier is empty (path unreachable due
            # to an Optional intermediate being None), return None — the
            # property test's only-allowed skip rule treats this as "no
            # sensitive data at this path for this example".
            if not frontier:
                return None
            ((_, value),) = frontier
            return value
        if container_step_count == 1:
            return [(keys[0], value) for keys, value in frontier]
        return [(keys, value) for keys, value in frontier]

    return provider


def _build_substitute_provider(
    steps: tuple[_PathStep, ...],
) -> Callable[[dict[str, Any], Callable[[Any], Any]], None]:
    """Compile a path-step list into an in-place substitute closure.

    Sibling to ``_build_value_provider``: walks the root dict by the same
    container-aware path and replaces every reachable leaf with
    ``transform(old_value)``.  Mutates the input dict in place.

    Path semantics mirror the walker:
      - ``("attr", name)`` steps descend dict-by-name.
      - ``("list",)`` and ``("dict",)`` steps iterate every element/value.

    The final step is always ``("attr", name)`` (every TraversalNode lives
    inside a named field).  At the final attr step, the dict at the current
    descent point has the leaf assigned at the name slot.

    Caller atomicity: when ``transform`` raises mid-walk, the in-progress
    dict has been partially mutated.  The walker MUST be invoked on a local
    variable that is only returned/exposed on full success — the response
    walker pattern (rev-3 W8b).  The closure does not catch ``transform``
    exceptions; that is by design.

    No-op for empty input (e.g., empty list/dict containers): the frontier
    is exhausted before reaching the leaf and ``transform`` is never called.
    """
    if not steps:
        raise AssertionError("substitute_provider requires at least one step (the field-level attr).")
    if steps[-1][0] != "attr":
        # walk_model_schema always emits TraversalNodes for a named field
        # (the last step is always ('attr', field_name)); a non-attr final
        # step would mean we're trying to substitute INTO an iteration step
        # itself, which has no destination semantics.
        raise AssertionError(f"substitute_provider requires final attr step; got steps={steps!r}.")
    prefix_steps = steps[:-1]
    leaf_name = steps[-1][1]

    def provider(root: dict[str, Any], transform: Callable[[Any], Any]) -> None:
        # frontier holds dict references whose ``leaf_name`` slot is the
        # final substitution target.  Container steps fan the frontier out.
        #
        # Optional-container handling: an ``("attr", name)`` step targets a
        # field that may be ``None`` at runtime when the field's declared
        # type is ``OptionalModel | None``.  The walker still emits inner-
        # path TraversalNodes through the BaseModel arm (e.g.,
        # ``source.inline_blob.content`` when ``inline_blob`` is None at
        # runtime).  The redactor only invokes ``substitute_provider`` for
        # Sensitive nodes, so the inner Sensitive leaf's provider receives
        # an arguments dict whose intermediate Optional is None.  Skipping
        # any frontier entry whose value at the named field is None (or
        # absent entirely) is the correct disposition: there is no leaf to
        # substitute INTO, and the absence is itself the audit fact.
        frontier: list[Any] = [root]
        for step in prefix_steps:
            kind = step[0]
            if kind == "attr":
                name = step[1]
                next_frontier: list[Any] = []
                for current in frontier:
                    # Optional-intermediate disposition (sibling parity with
                    # _build_value_provider lines 211-216): a None intermediate
                    # or a runtime union arm that lacks this named field has no
                    # leaf to substitute INTO — skip it. Direct subscript on the
                    # present-key case; the value may itself be None (declared
                    # Optional sub-model unset at runtime), which is equally
                    # unsubstitutable, so skip that too.
                    if current is None:
                        continue
                    if isinstance(current, Mapping) and name not in current:
                        continue
                    value = current[name]
                    if value is None:
                        continue
                    next_frontier.append(value)
                frontier = next_frontier
            elif kind == "list":
                frontier = [item for current in frontier for item in current]
            elif kind == "dict":
                frontier = [item for current in frontier for item in current.values()]
            else:
                raise AssertionError(f"unknown path step kind: {kind!r}")
        for container in frontier:
            if container is None:
                continue
            if isinstance(container, Mapping) and leaf_name not in container:
                continue
            container[leaf_name] = transform(container[leaf_name])

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
            substitute_provider=_build_substitute_provider(steps) if with_values else None,
        )
        return

    # Union/Optional: descend into every non-None arm at the SAME path.
    #
    # Two emission rules apply (closes rev-3 M1 floor-check):
    #
    #   1. A descendable arm (BaseModel, parameterised container, nested Union)
    #      is always walked — that's the structural-descent contract Tasks 1/7
    #      established for nested-path redaction.
    #   2. A non-descendable scalar arm (str, int, bool, ...) is walked iff
    #      either (a) it carries a Sensitive marker (rev-3 W8a:
    #      Optional[Annotated[str, Sensitive()]]) OR (b) the Union is
    #      structurally an Optional — exactly one non-None arm.  Case (b)
    #      means the Pydantic model still declared a single field; the walker
    #      must emit a leaf at the field path so the rev-3 M1 floor-check
    #      (every ``model_fields`` key appears as a walk root) holds.  The
    #      legacy "skip non-descendable scalar arms" rule existed to prevent
    #      spurious duplicate yields for genuine multi-arm scalar Unions like
    #      ``str | _Model | bool`` (test_walk_three_arm_union pins this);
    #      Optional-of-scalar is single-armed by construction so duplicate-
    #      yield risk does not apply.
    #
    # The Optional-of-scalar carve-out is deliberately narrow: we still skip
    # scalar arms when ANY other non-None arm is present.  test_walk_three_arm_union
    # (``field: str | _InnerWithSecret | bool``) MUST continue to walk only via
    # the BaseModel arm; this fix is purely additive for the Optional[scalar]
    # case that previously emitted no node at all.
    if _is_union(origin):
        non_none_arms = [a for a in get_args(field_type) if a is not type(None)]
        is_optional_of_scalar = len(non_none_arms) == 1
        for arm in non_none_arms:
            arm_type, arm_metadata = _normalise(arm, ())
            if not _is_descendable(arm_type) and not _has_sensitive(arm_metadata) and not is_optional_of_scalar:
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
    #
    # ``substitute_provider`` is meaningful only when the leaf sits at a named
    # field — that is, when ``steps[-1]`` is ``("attr", name)`` — because
    # substitution writes a new value INTO a dict at the named slot.  Container-
    # element leaves (``list[str]``, ``dict[str, str]`` with non-Sensitive scalar
    # elements) end their step chain with ``("list",)`` or ``("dict",)`` — there
    # is no named destination to substitute into.  The redactor at
    # :func:`_redact_via_schema` only consumes ``substitute_provider`` for nodes
    # carrying a ``_SensitiveMarker``; by spec §4.2.5 Sensitive markers live on
    # named fields (the walker's "Sensitive yield-and-return" branch above
    # always sees ``("attr", name)`` as the last step), so producing ``None``
    # for container-element scalar leaves is correct.  The adequacy guard
    # (§4.4.2) still gets its closed-list scalar check via
    # ``node.field_type`` — substitute_provider being None there is harmless.
    needs_substitute = bool(steps) and steps[-1][0] == "attr"
    yield TraversalNode(
        path=path,
        field_type=field_type,
        metadata=metadata,
        value_provider=_build_value_provider(steps) if with_values else None,
        substitute_provider=_build_substitute_provider(steps) if with_values and needs_substitute else None,
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


@dataclass(frozen=True, slots=True)
class HandlesNoSensitiveDataReason:
    """Structured justification required for handles_no_sensitive_data=True (spec §4.2.3).

    Replaces rev-3's free-text string. Adequacy-guard validates each field at
    construction time; the same constraints are checked at manifest-load time
    by the adequacy guard (§4.4).

    sensitive_data_locations: where sensitive material actually lives if not in
        this tool's arguments or responses (e.g., 'server-side secret resolver',
        'request headers stripped before tool dispatch'). Must be non-empty —
        an empty list provides no signal to auditors.

    why_arguments_safe: prose explanation of why every argument, including any
        string-typed ones, is safe to persist verbatim. Adequacy-guard checks
        length >= 32 chars (post-strip) and that the text does not exact-match
        any other tool's why_arguments_safe (mass-copy uniqueness, §4.4.4).

    why_responses_safe: same as above, for responses.
    """

    sensitive_data_locations: tuple[str, ...]
    why_arguments_safe: str
    why_responses_safe: str

    def __post_init__(self) -> None:
        # Validators FIRST, then freeze-guard. Order matters: a future change
        # that mutates a field as part of validation must not run after freeze.
        if not self.sensitive_data_locations:
            raise ValueError(
                "HandlesNoSensitiveDataReason.sensitive_data_locations is empty. "
                "Every declarative manifest entry must enumerate at least one "
                "location where sensitive data could plausibly appear but does "
                "not for this tool (e.g., 'response.body' for a tool that "
                "returns only structural metadata). If the tool genuinely has "
                "no plausible sensitive surface, declare so explicitly — e.g., "
                "('no LLM-supplied inputs',). The list is part of the audit "
                "trail; an empty list provides no information to auditors. "
                "Alternatively, migrate the tool's arguments to a Pydantic "
                "model with Sensitive[T] annotations (the type-driven "
                "manifest-entry shape, §4.2.1)."
            )
        for label, value in (
            ("why_arguments_safe", self.why_arguments_safe),
            ("why_responses_safe", self.why_responses_safe),
        ):
            if len(value.strip()) < 32:
                raise ValueError(
                    f"HandlesNoSensitiveDataReason.{label} must be at least "
                    "32 characters (excluding leading/trailing whitespace). "
                    f"The current value strips to {len(value.strip())} characters. "
                    "Auditors require enough context to understand WHY the "
                    "tool's argument or response shape cannot carry sensitive "
                    "data; a short label is insufficient. "
                    "Spec §4.2.3 example: 'set_pipeline arguments are "
                    "validated against a Pydantic model with extra=forbid; "
                    "every field is structural metadata.'"
                )
        freeze_fields(self, "sensitive_data_locations")


@dataclass(frozen=True, slots=True)
class ToolRedactionPolicy:
    """Declarative redaction policy (spec §4.2.3). Used inside the ``policy``
    field of a manifest entry whose argument surface is purely structural (no
    Pydantic argument model declared). The type-driven shape (§4.2.1 with
    ``argument_model`` set) is preferred for any tool that has, or would
    benefit from, a redaction-bearing Pydantic argument model.

    Five invariants are enforced at construction time:

    1. **No orphan summarizers** — every key in ``argument_summarizers`` must
       also appear in ``sensitive_argument_keys``. A summarizer for a key that
       is not declared sensitive is dead code and silently misleads auditors
       about which keys are redacted.

    2. **``handles_no_sensitive_data=True`` requires a reason struct** — the
       structured justification is part of the audit trail; an unexplained
       exemption is forbidden (spec §4.2.3).

    3. **``handles_no_sensitive_data=False`` forbids a reason struct** — the
       reason struct is meaningful only for an exemption; for a tool with
       sensitive data the redaction policy is the documentation.

    4. **``handles_no_sensitive_data=False`` requires ``known_response_keys``**
       — the allowlist defends against response-shape drift; unknown keys at
       persistence time are fail-closed redacted with a fixed sentinel.

    5. **``known_argument_keys`` covers ``sensitive_argument_keys`` when set or
       when ``redact_unknown_argument_keys`` is enabled** — the argument
       allowlist is opt-in for legacy compatibility, but once a policy uses it
       to fail-close unknown arguments, every sensitive argument key must also
       be known.

    **Response-walker fail-closed behavior for declarative entries:**

    When ``handles_no_sensitive_data=False``, ``redact_tool_call_response``
    iterates ``response.items()`` and dispatches each key as follows:

    - Keys in ``sensitive_response_keys`` → substituted with
      ``REDACTED_SENSITIVE_NO_SUMMARIZER`` (declarative entries have no
      per-key response summarizers; argument_summarizers covers only
      argument keys).
    - Keys in ``known_response_keys`` that are NOT in
      ``sensitive_response_keys`` → passthrough; the value reaches the
      audit-table record unchanged.
    - Keys in NEITHER set → substituted with the fixed sentinel
      ``REDACTED_UNKNOWN_RESPONSE_KEY``. This is the **fail-closed
      default**: a key the policy author did not declare is sentinel'd,
      not leaked.

    Therefore ``known_response_keys`` MUST enumerate every key the
    handler may emit in normal operation (both success and failure
    branches).  The runtime smoke test
    (``test_declarative_manifest_runtime_smoke.py``) verifies this by
    exercising each declarative entry through a real
    ``redact_tool_call_response`` call.  Adequacy-guard assertion 5
    and additionally checks that ``sensitive_response_keys ⊆
    known_response_keys`` at import time so a misspelled sensitive key
    is caught before any payload reaches the walker.

    NOTE on freeze: ``argument_summarizers`` values are Callables;
    ``deep_freeze`` passes Callables through unchanged (verified against
    ``src/elspeth/contracts/freeze.py:78``). Identity-equality of summarizer
    callables is the policy contract.
    """

    sensitive_argument_keys: tuple[str, ...] = ()
    sensitive_response_keys: tuple[str, ...] = ()
    known_argument_keys: tuple[str, ...] = ()
    known_response_keys: tuple[str, ...] = ()
    argument_summarizers: Mapping[str, Callable[[Any], str]] = field(default_factory=dict)
    handles_no_sensitive_data: bool = False
    handles_no_sensitive_data_reason_struct: HandlesNoSensitiveDataReason | None = None
    redact_unknown_argument_keys: bool = False

    def __post_init__(self) -> None:
        # Validators run BEFORE freeze_fields so they read mutable state.
        # If any raise, the dataclass __init__ raises and the object is
        # never returned — atomic construction failure.

        orphan_summarizers = set(self.argument_summarizers) - set(self.sensitive_argument_keys)
        if orphan_summarizers:
            raise ValueError(
                f"argument_summarizers keys {sorted(orphan_summarizers)} are not declared in "
                f"sensitive_argument_keys; orphan summarizers indicate a policy bug."
            )

        orphan_sensitive_arguments = set(self.sensitive_argument_keys) - set(self.known_argument_keys)
        if (self.known_argument_keys or self.redact_unknown_argument_keys) and orphan_sensitive_arguments:
            raise ValueError(
                f"sensitive_argument_keys {sorted(orphan_sensitive_arguments)} are not declared in "
                "known_argument_keys; the opt-in argument allowlist must cover every sensitive argument key."
            )

        if self.handles_no_sensitive_data and self.handles_no_sensitive_data_reason_struct is None:
            raise ValueError(
                "handles_no_sensitive_data=True requires a non-None "
                "handles_no_sensitive_data_reason_struct. Build a "
                "HandlesNoSensitiveDataReason instance with concrete fields."
            )

        if not self.handles_no_sensitive_data and self.handles_no_sensitive_data_reason_struct is not None:
            raise ValueError("handles_no_sensitive_data_reason_struct is only meaningful when handles_no_sensitive_data=True.")

        if not self.handles_no_sensitive_data and not self.known_response_keys:
            raise ValueError(
                "known_response_keys must be declared (non-empty) when "
                "handles_no_sensitive_data=False. The allowlist defends "
                "against response-shape drift; unknown keys at persistence "
                "time are fail-closed redacted with a fixed sentinel."
            )

        freeze_fields(
            self,
            "sensitive_argument_keys",
            "sensitive_response_keys",
            "known_argument_keys",
            "known_response_keys",
            "argument_summarizers",
        )


@dataclass(frozen=True, slots=True)
class ToolRedaction:
    """One manifest entry. Each entry is keyed by tool name in MANIFEST.

    Exactly one of the two shapes must be populated:
      • type-driven  — argument_model is not None
      • declarative  — argument_model is None AND policy is not None

    Both populated → ValueError (precedence is undefined; use one shape).
    Neither populated → ValueError (every tool must have a redaction
    declaration; the adequacy guard cannot consult what doesn't exist).

    response_model is only valid alongside argument_model. Declarative
    entries express response shape via policy.known_response_keys.
    """

    argument_model: type[BaseModel] | None = None
    response_model: type[BaseModel] | None = None
    policy: ToolRedactionPolicy | None = None

    def __post_init__(self) -> None:
        type_driven = self.argument_model is not None
        declarative = self.policy is not None

        if type_driven and declarative:
            raise ValueError(
                "ToolRedaction declared both argument_model and policy; "
                "each manifest entry must choose exactly one shape. "
                "If a tool has a Pydantic argument model with Sensitive[T] "
                "annotations, the model is the single source of truth — "
                "remove the policy. If the argument surface is purely "
                "structural and does not benefit from Sensitive[T], "
                "remove the argument_model and declare the policy."
            )
        if not type_driven and not declarative:
            raise ValueError(
                "ToolRedaction declared neither argument_model nor policy; "
                "every manifest entry must declare its redaction shape. "
                "If the tool genuinely handles no sensitive material, set "
                "policy=ToolRedactionPolicy(handles_no_sensitive_data=True, "
                "handles_no_sensitive_data_reason_struct=...) — the "
                "structured reason is part of the audit trail."
            )
        if self.response_model is not None and not type_driven:
            raise ValueError(
                "response_model requires argument_model to also be set "
                "(declarative entries express response shape via "
                "policy.known_response_keys)."
            )


def _summarize_option_shape(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _summarize_option_shape(child) for key, child in sorted(value.items(), key=lambda item: str(item[0]))}
    if isinstance(value, (list, tuple)):
        return [_summarize_option_shape(item) for item in value]
    if isinstance(value, AbstractSet):
        return sorted((_summarize_option_shape(item) for item in value), key=repr)
    return _REDACTED_OPTION_VALUE


def _summarize_set_source_options(options: object) -> str:
    """Summarizer for ``set_source.options`` (spec §4.2.6).

    Produces canonical JSON for the option payload's shape: mapping keys,
    nested mappings, and sequence lengths are preserved, but every scalar
    value is replaced before serialization. Plugin options are an open,
    plugin-defined surface and routinely carry filesystem paths, source
    object locators, prompt text, and credential material such as connection
    strings, SAS tokens, API keys, and client secrets. Therefore the summary
    must not rely on per-plugin sensitive-key knowledge or on the blob_ref
    path-only redactor.

    Contract (spec §4.2.6, §9 RSK-03):
      * MUST NOT raise on any reachable input value.
      * MUST return ``str``.

    ``default=str`` on :func:`json.dumps` is retained as a final defensive
    guard for unusual mapping keys or container shapes; reachable scalar
    values are substituted before dumps sees them.
    """
    return json.dumps(
        _summarize_option_shape(options) if isinstance(options, Mapping) else "<invalid-options>",
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )


def _coerce_stringified_json_object(value: Any) -> Any:
    """Tier-3 boundary deserialisation for LLM-supplied object arguments.

    Some models — notably ``openrouter/openai/gpt-5.4-mini``, the deployed
    composer model — intermittently serialise a nested-object tool-call
    parameter as a JSON *string* (``options="{}"``,
    ``patch="{\\"column\\":\\"url\\"}"``) instead of emitting a JSON object
    (``options={}``). This was proven from the staging audit trail (sessions
    ``fd551d98`` / ``71d57b4f``): every free-form ``options`` / ``patch`` field
    arrived as a string while typed sibling fields (``blob_id``, ``nodes``)
    arrived correctly, so the build failed wholesale on a ``dict[str, Any]``
    ``ValidationError`` whenever the model stringified.

    A JSON string is an equivalent wire encoding of the object it encodes;
    parsing it back is *meaning-preserving coercion*, not fabrication
    (CLAUDE.md "Data Manifesto" — Tier-3 boundary, the ``"42" -> 42`` class),
    and is therefore exempt from the defensive-programming ban as a documented
    trust-boundary deserialisation. The raw stringified form is recorded in the
    per-dispatch audit envelope (``service.py`` ``begin_dispatch_or_arg_error``,
    opened from the pre-coercion ``json.loads`` result) BEFORE this validator
    runs, so the audit trail still records exactly what the model emitted.

    The coercion is deliberately narrow: only a string that decodes to a JSON
    *object* is coerced. A non-string, a string that is not valid JSON, or a
    string that decodes to a non-object (list, scalar, ``null``) is returned
    untouched so the field's ``dict[str, Any]`` validation still rejects
    genuinely malformed input (fails closed).
    """
    if not isinstance(value, str):
        return value
    try:
        decoded = json.loads(value)
    except (json.JSONDecodeError, ValueError):
        return value
    return decoded if isinstance(decoded, dict) else value


# Reusable annotation for every LLM-supplied free-form object argument
# (``options`` on the source/node/output binding tools; ``patch`` on the
# patch_* tools). Combines the existing ``Sensitive`` redaction marker with the
# stringified-object coercion above. The redaction walker selects the
# ``_SensitiveMarker`` by ``isinstance`` (``_has_sensitive`` /
# ``_count_sensitive``), so the extra ``BeforeValidator`` metadata entry is
# transparent to the adequacy guard; pydantic's ``BeforeValidator`` is a frozen
# dataclass, so it does not introduce mutable metadata.
_LlmJsonObject = Annotated[
    dict[str, Any],
    Sensitive(summarizer=_summarize_set_source_options),
    BeforeValidator(_coerce_stringified_json_object),
]


class SetSourceArgumentsModel(BaseModel):
    """Redaction-bearing argument model for the ``set_source`` tool.

    Mirrors the JSON schema currently consumed by ``_execute_set_source``
    (``tools.py``) and the required-paths check at ``service.py``
    ``_TOOL_REQUIRED_PATHS["set_source"]``.  The ``Annotated`` on
    ``options`` drives mechanical redaction at the persistence boundary;
    the model itself is also used at the dispatch boundary by
    :meth:`Model.model_validate` so the LLM-supplied dict is validated
    before ``_execute_set_source`` reads any field.

    ``extra="forbid"`` is required (rev-2 M.1): without it the model
    would silently accept fields the walker has no record of, breaking
    the manifest/canonical-arguments parity invariant the adequacy
    guard relies on.

    Field set is exactly the four required keys plus optional
    ``source_name`` from the JSON schema at ``tools.py``.  Fields belonging to neighbouring tools
    (``label``, ``blob_id``, ``inline_blob`` — those are on
    ``set_source_from_blob`` / ``create_blob``) are intentionally absent
    so ``extra="forbid"`` rejects misrouted argument shapes early.
    """

    source_name: str = "source"
    plugin: str
    on_success: str
    options: _LlmJsonObject
    on_validation_failure: str

    model_config = ConfigDict(extra="forbid")


def _summarize_interpretation_term(text: str) -> str:
    """Summarizer for ``request_interpretation_review.user_term`` and
    ``request_interpretation_review.llm_draft`` (F-34).

    Returns the input collapsed to a fixed-form ``<interpretation-term:N-chars>``
    or ``<interpretation-term:N-chars:truncated>`` shape where ``N`` is the
    code-point length of the original string. The fixed-form scalar is
    structurally distinguishable from the raw value at every reachable
    input (including the empty string) so the redaction-completeness
    property test can assert ``redacted_value != raw_value`` uniformly.

    The 64-character truncation guard documented in the spec is preserved
    via the ``:truncated`` suffix when the original exceeded the cap —
    auditors can distinguish "long value redacted" from "short value
    redacted" without seeing either. The authoritative value of the term
    still lives in the ``interpretation_events`` row (``user_term`` /
    ``llm_draft`` columns); the audit-side row in
    ``chat_messages.tool_calls`` carries only this fixed-form scalar.

    Naming follows ``_summarize_inline_blob_content`` (American
    spelling). Contract: MUST NOT raise on any reachable input; MUST
    return ``str``.
    """
    truncated = ":truncated" if len(text) > 64 else ""
    return f"<interpretation-term:{len(text)}-chars{truncated}>"


def _summarize_inline_blob_content(content: str) -> str:
    """Summarizer for blob-content fields (``create_blob.content``,
    ``update_blob.content``) — spec §4.2.6 / rev-2 M.10.

    The blob ``content`` field is the LLM-supplied raw bytes of a session
    blob: URLs, JSON snippets, CSV seed data, or arbitrary text that may
    contain operator-sensitive material.  Persisting the raw payload in
    ``chat_messages.tool_calls`` would mirror that material into the audit
    trail beyond its intended retention surface (the blob row itself, the
    canonical record of file content).  The summarizer collapses the
    payload to a fixed-form scalar ``<inline-blob:N-bytes>`` where ``N``
    is the byte-length of the UTF-8 encoded content — disclosing size
    only (a structural fact already inferable from the persisted blob
    row's ``size_bytes`` column), never content.

    Contract (spec §4.2.6, §9 RSK-03):
      * MUST NOT raise on any reachable input value.
      * MUST return ``str``.

    Type-variability discipline (rev-2 M.10): the type-driven argument
    models for ``create_blob`` and ``update_blob`` declare
    ``content: Annotated[str, Sensitive(summarizer=...)]`` plus
    ``model_config = ConfigDict(extra="forbid")``.  Pydantic's ``str``
    validation rejects ``None``, ``int``, ``bool``, ``list``, ``dict``,
    and any other non-string before the summarizer is invoked, so this
    function is reached only with a genuine ``str``.  ``len(b)`` on a
    Python ``str`` measures code points; we explicitly UTF-8 encode to
    measure the wire-format byte-length the persistence boundary actually
    sees, matching the ``size_bytes`` column the handler computes.
    """
    return f"<inline-blob:{len(content.encode('utf-8'))}-bytes>"


class SetSourceFromBlobArgumentsModel(BaseModel):
    """Redaction-bearing argument model for the ``set_source_from_blob`` tool.

    Mirrors the JSON schema declared at ``tools.py:1329-1361`` for the
    ``set_source_from_blob`` definition and its required-paths
    (``blob_id``, ``on_success``).

    ``options`` is :class:`Sensitive` with the same summarizer
    (:func:`_summarize_set_source_options`) used by
    :class:`SetSourceArgumentsModel`.  ``set_source_from_blob`` merges
    caller-supplied ``options`` with the resolved blob's authoritative
    ``path`` and ``blob_ref`` at ``_resolve_source_blob`` (``tools.py:
    1961-1966``); the merged dict carries an internal storage path that
    must not enter the audit-side ``chat_messages.tool_calls`` row
    verbatim.  Treating the caller-supplied ``options`` slot as
    Sensitive at the argument boundary maintains uniformity with
    ``set_source.options`` so an LLM that happens to include a
    path-like field receives the same redaction discipline regardless
    of which source-binding tool it invoked.

    ``blob_id`` is a UUID reference, not content; ``plugin`` and
    ``on_validation_failure`` are operator-controlled discriminators;
    ``on_success`` is a connection-name string.  None of these carry
    LLM-supplied sensitive material, so only ``options`` is marked.

    ``options`` default
    -------------------
    The JSON schema for ``set_source_from_blob`` does NOT list
    ``options`` as required; the handler at
    :func:`_execute_set_source_from_blob` reads it via
    ``arguments.get("options", {})``.  We mirror that absent-equals-empty
    semantics with ``Field(default_factory=dict)`` rather than declaring
    ``options: dict | None = None``.  Reason: the summarizer needs to
    handle a real mapping value and should record an absent slot as an
    empty option shape, not as a separate ``None`` shape.  Defaulting to
    ``{}`` preserves "absent = no options" at the argument boundary AND
    records it accurately on the audit side as an empty-options dict.

    ``plugin`` and ``on_validation_failure`` keep ``str | None = None``
    because the handler distinguishes their absent semantics: ``plugin``
    absent triggers MIME-type-based inference at
    ``_resolve_source_blob``; ``on_validation_failure`` absent falls back
    to ``_DEFAULT_SOURCE_VALIDATION_FAILURE`` ("discard").  A default of
    ``""`` would conflate "operator did not specify" with "operator
    specified empty string" — fabrication (CLAUDE.md trust model).

    ``extra="forbid"`` is required (rev-2 M.1).  Fields belonging to
    neighbouring tools (``filename``, ``mime_type``, ``content``,
    ``description`` on ``create_blob``/``update_blob``; ``inline_blob``
    on ``set_pipeline``) are intentionally absent so ``extra="forbid"``
    rejects misrouted argument shapes early.
    """

    blob_id: str
    on_success: str
    source_name: str = "source"
    plugin: str | None = None
    on_validation_failure: str | None = None
    options: _LlmJsonObject = Field(default_factory=dict)

    model_config = ConfigDict(extra="forbid")


class UpdateBlobArgumentsModel(BaseModel):
    """Redaction-bearing argument model for the ``update_blob`` tool.

    Mirrors the JSON schema declared at ``tools.py:1398-1415`` for the
    ``update_blob`` definition and its required-paths (``blob_id``,
    ``content``).  The ``Annotated`` on ``content`` substitutes a
    length-disclosing scalar at the persistence boundary so the raw blob
    payload never enters ``chat_messages.tool_calls`` (rev-2 M.10).
    Identical summarizer to :class:`CreateBlobArgumentsModel.content` —
    both blob-mutation tools accept the same payload shape and the
    audit-side redaction must be uniform across them.

    ``extra="forbid"`` is required (rev-2 M.1).  Fields belonging to
    neighbouring tools (``filename``, ``mime_type``, ``description`` —
    those are on ``create_blob``; ``plugin``, ``on_success``, ``options``
    — those are on ``set_source*``) are intentionally absent so
    ``extra="forbid"`` rejects misrouted argument shapes early.
    """

    blob_id: str
    content: Annotated[str, Sensitive(summarizer=_summarize_inline_blob_content)]

    model_config = ConfigDict(extra="forbid")


class _RequestInterpretationReviewRedactionModel(BaseModel):
    """Redaction-bearing argument model for ``request_interpretation_review``
    (F-34).

    The tool is a session-aware async handler; this model is the persistence-
    boundary redaction declaration registered in :data:`MANIFEST` so the
    audit-side ``chat_messages.tool_calls`` row carries summarised content
    rather than the LLM-supplied raw text.

    ``user_term`` is not strictly a secret — it's a word the user typed —
    but it could carry PII if the user typed something like ``"rate how
    cool this transaction involving Jane Doe is"``. ``llm_draft`` is the
    model's draft and may quote or paraphrase user content. Both fields
    carry :class:`Sensitive` with :func:`_summarize_interpretation_term`
    so the truncated form lands in the tool-call column (the full value
    is still in ``interpretation_events_table.user_term`` /
    ``interpretation_events_table.llm_draft`` — the authoritative row).

    ``affected_node_id`` and ``kind`` are structural metadata and are not
    marked :class:`Sensitive`. ``extra="forbid"`` ensures a misrouted
    argument shape fails fast at the persistence boundary rather than
    silently accepting an unknown key.
    """

    affected_node_id: str
    kind: InterpretationKind
    user_term: Annotated[str, Sensitive(summarizer=_summarize_interpretation_term)]
    llm_draft: Annotated[str, Sensitive(summarizer=_summarize_interpretation_term)]

    model_config = ConfigDict(extra="forbid")


class CreateBlobArgumentsModel(BaseModel):
    """Redaction-bearing argument model for the ``create_blob`` tool.

    Mirrors the JSON schema declared at ``tools.py:1362-1397`` for the
    ``create_blob`` definition and the required-paths the schema enforces
    (``filename``, ``mime_type``, ``content``; ``description`` optional).
    The ``Annotated`` on ``content`` substitutes a length-disclosing
    scalar at the persistence boundary so the raw blob payload never
    enters ``chat_messages.tool_calls`` (rev-2 M.10).

    ``extra="forbid"`` is required (rev-2 M.1): the walker enumerates the
    declared field set and the adequacy guard relies on canonical-args/
    walker parity.  A stray field that Pydantic silently accepted would
    leak through every gate (manifest dispatch, persistence, adequacy).

    Fields belonging to neighbouring tools (``blob_id`` — that is on
    ``update_blob`` / ``set_source_from_blob`` / ``delete_blob`` /
    ``get_blob_content`` / ``inspect_source``) are intentionally absent
    so ``extra="forbid"`` rejects misrouted argument shapes early.
    """

    filename: str
    mime_type: AllowedMimeType
    content: Annotated[str, Sensitive(summarizer=_summarize_inline_blob_content)]
    description: str | None = None

    model_config = ConfigDict(extra="forbid")


# ---------------------------------------------------------------------------
# set_pipeline / apply_pipeline_recipe argument models.
#
# set_pipeline is the atomic full-state mutation; apply_pipeline_recipe is the
# recipe-scaffolded variant that delegates to set_pipeline after composing the
# arguments from operator-supplied slot values.  Both are promoted to type-
# driven manifest entries here; the inner ``_execute_set_pipeline`` call from
# apply_pipeline_recipe also re-validates the recipe-built args because the
# handler is the single validation site (rev-3 N7 / rev-4 M1).
#
# Field-shape decisions:
#   * ``source.options`` IS ``Sensitive[dict]`` with the same summarizer
#     :func:`_summarize_set_source_options` already used by ``set_source`` /
#     ``set_source_from_blob`` (uniformity-across-source-binding-tools
#     contract).
#   * ``source.inline_blob.content`` IS ``Sensitive[str]`` with the same
#     summarizer :func:`_summarize_inline_blob_content` already used by
#     ``create_blob`` / ``update_blob`` (uniformity-across-blob-payload-tools
#     contract).
#   * ``nodes[*].options`` and ``outputs[*].options`` ARE ``Sensitive[dict]`` with
#     :func:`_summarize_set_source_options` — the adequacy guard (§4.4.2)
#     fails closed on inspection-resistant ``dict[str, Any]`` field types
#     without Sensitive marking, and the LLM-supplied dicts routinely
#     carry secret-ref markers, filesystem paths, credential material, and
#     prompt-template payloads.  Reusing the source-side summarizer is
#     structurally safe because it records option shape while replacing all
#     scalar values before persistence; future tool-specific summarizers may
#     preserve more non-sensitive structure.
#   * ``nodes[*].routes`` and ``nodes[*].trigger`` are structurally typed
#     models/dicts with closed scalar leaf types, so they are validated and
#     walked directly rather than collapsed by the option summarizer.
#
# Walker coverage at the persistence boundary:
# ``walk_model_schema`` descends into ``list[NestedModel]`` so the Sensitive markers on the nested option
# dicts are discoverable from the outer :class:`SetPipelineArgumentsModel`.
# This produces four Sensitive paths in the model walk:
# ``source.options``, ``source.inline_blob.content``, ``nodes[*].options``,
# ``outputs[*].options``.
# ---------------------------------------------------------------------------


class _InlineBlobModel(BaseModel):
    """Nested model for ``set_pipeline.source.inline_blob``.

    Mirrors the JSON schema's ``inline_blob`` sub-object declared at
    ``tools.py:986-1009`` (under the ``set_pipeline.source`` properties).
    Required: ``filename``, ``mime_type``, ``content``.  ``description``
    is optional and matches :class:`CreateBlobArgumentsModel.description`
    (the inline-blob path at :func:`_execute_set_pipeline` invokes
    :func:`_prepare_blob_create`, which reads ``arguments.get("description")``
    — operator-facing labels are honoured).

    ``content`` carries :class:`Sensitive` with
    :func:`_summarize_inline_blob_content` — the SAME summarizer used by
    :class:`CreateBlobArgumentsModel.content` and
    :class:`UpdateBlobArgumentsModel.content`.  Uniformity discipline:
    the LLM may funnel the same raw payload through any of the three
    blob-creating tools (``create_blob``, ``update_blob``,
    ``set_pipeline``+``inline_blob``); the audit-side redaction must
    collapse all three to the same fixed-form ``<inline-blob:N-bytes>``
    scalar.
    """

    filename: str
    mime_type: AllowedMimeType
    # max_length=262_144 (256 KiB) is the inline-blob payload cap (Phase
    # Inline-blob content policy. The composer is intended for prose-sized inline data
    # — a single CSV with up to a few thousand rows — not for binary
    # transfers.  Pydantic rejects oversize content at the boundary so
    # ``_prepare_blob_create`` and the storage-write layer never see it;
    # ToolArgumentError surfaces through the compose loop's ARG_ERROR
    # routing (CEC1).  The siblings ``CreateBlobArgumentsModel.content``
    # / ``UpdateBlobArgumentsModel.content`` do not yet carry this cap;
    # alignment is deferred to the same task that introduces a shared
    content: Annotated[str, Field(max_length=262_144), Sensitive(summarizer=_summarize_inline_blob_content)]
    description: str | None = None

    model_config = ConfigDict(extra="forbid")


class _SetPipelineNamedSourceModel(BaseModel):
    """Typed fields shared by every named ``set_pipeline.sources`` root.

    Named sources deliberately exclude ``blob_id`` and ``inline_blob``:
    those custody-bearing fields are supported only by the legacy singular
    ``source`` branch. Keeping a distinct runtime model makes that semantic
    boundary visible to the advertised-schema compatibility guard.
    """

    plugin: str
    on_success: str
    options: _LlmJsonObject = Field(default_factory=dict)
    on_validation_failure: str | None = None

    model_config = ConfigDict(extra="forbid")


class _SetPipelineSourceModel(_SetPipelineNamedSourceModel):
    """Nested model for ``set_pipeline.source``.

    Mirrors the JSON schema's ``source`` sub-object declared at
    ``tools.py:949-1012``.  Required: ``plugin``, ``on_success`` (matching
    the inner ``required: ["plugin", "on_success"]`` of the JSON schema).
    All other fields are optional at the boundary; the runtime handler in
    :func:`_execute_set_pipeline` resolves them (``blob_id`` →
    :func:`_resolve_source_blob`; ``inline_blob`` →
    :func:`_prepare_blob_create`; ``options.path`` →
    :func:`_validate_source_path` allowlist).

    ``options`` carries :class:`Sensitive` with
    :func:`_summarize_set_source_options` — the SAME summarizer used by
    :class:`SetSourceArgumentsModel.options` and
    :class:`SetSourceFromBlobArgumentsModel.options` (uniformity contract).
    An LLM that includes a path-like field in
    ``set_pipeline.source.options`` receives the same redaction discipline
    it would receive via ``set_source`` or ``set_source_from_blob``.

    ``options`` default
    -------------------
    Mirrors :class:`SetSourceFromBlobArgumentsModel.options`:
    ``Field(default_factory=dict)`` — absent equals empty.  Reason
    documented at length on
    :class:`SetSourceFromBlobArgumentsModel.options` and applies
    identically here.

    ``on_validation_failure`` default
    ----------------------------------
    The handler at :func:`_execute_set_pipeline` falls back to
    ``_DEFAULT_SOURCE_VALIDATION_FAILURE`` ("discard") when absent
    (``tools.py:4038``).  The model preserves operator-omitted-vs-specified
    semantics with ``str | None = None`` so the handler can apply the
    fallback explicitly (not via fabrication; CLAUDE.md trust model).

    ``blob_id`` / ``inline_blob`` exclusivity
    ------------------------------------------
    The JSON schema does NOT encode "exactly one of (blob_id, inline_blob)
    or neither" (JSON Schema's ``oneOf`` is not used here); the handler at
    :func:`_execute_set_pipeline` (``tools.py:4039``) rejects the
    both-supplied case at runtime.  We deliberately do NOT replicate the
    exclusivity at the Pydantic model layer — the handler's existing check
    produces a recoverable ``_failure_result`` with a repair hint that the
    LLM can act on, whereas a Pydantic-level rejection would surface as a
    bare ARG_ERROR with no repair guidance.  Two channels for two failure
    shapes (type vs semantic) — same pattern as
    ``apply_pipeline_recipe`` empty-``recipe_name`` handling.
    """

    blob_id: str | None = None
    inline_blob: _InlineBlobModel | None = None


class _NodeTriggerModel(BaseModel):
    """Typed sub-model for ``nodes[*].trigger`` (aggregation early-batch trigger).

    Mirrors the JSON schema declared at ``tools.py:735-755`` field-for-field:
    a ``count`` row-threshold (``int | None``), a ``timeout_seconds`` wall-clock
    threshold (``float | None``), and an optional boolean-expression
    ``condition`` (``str | None``) evaluated over runtime batch state
    (``row['batch_count']`` and ``row['batch_age_seconds']``).

    Every field is a closed-list scalar per spec §4.4.2 — no Sensitive marker
    is required.  The ``condition`` string is a composer-author-supplied
    boolean expression (matches the composer's grammar discipline) and does
    NOT carry user-data values.

    ``extra="forbid"`` aligns with the JSON schema's ``additionalProperties:
    False`` so the LLM cannot smuggle unmodelled keys past the redaction
    boundary.
    """

    count: int | None = None
    timeout_seconds: float | None = None
    condition: str | None = None

    model_config = ConfigDict(extra="forbid")


class _PipelineNodeModel(BaseModel):
    """Nested model for ``set_pipeline.nodes[*]``.

    Mirrors the JSON schema's ``nodes`` array-item shape declared at
    ``tools.py:1013-1060`` and its inner ``required: ["id", "node_type",
    "input"]``.

    All optional fields preserve operator-omitted-vs-specified semantics
    (``X | None = None``) because the handler at
    :func:`_execute_set_pipeline` distinguishes "absent" from "operator-
    specified empty/default" for several of them (``plugin`` is None for
    gates and coalesces; ``on_error`` defaults to ``"discard"`` for
    transform/aggregation when None).  ``options`` defaults to ``{}`` via
    :func:`Field(default_factory=dict)` (matches the handler's
    ``n.get("options", {})``).

    Sensitive marker on ``options``
    -------------------------------
    ``options`` is an LLM-supplied dict that routinely carries secret-ref
    markers (``api_key: {"secret_ref": ...}``), filesystem paths
    (``path: "/data/in.csv"``), and prompt templates (``template: "..."``).
    Without a Sensitive annotation the adequacy guard (§4.4.2) fails
    closed on the inspection-resistant ``dict[str, Any]`` field type;
    with the annotation the persistence boundary collapses the field to
    the canonical-JSON shape summary that
    :func:`_summarize_set_source_options` already supplies for source-
    side options.

    The summarizer is reused (NOT a new node-specific one) because it is
    option-surface agnostic: it preserves mapping keys and nested container
    shape while replacing every scalar value. That behaviour is correct for
    ``nodes[*].options`` too because plugin options may carry credentials,
    paths, prompts, or future plugin-defined sensitive fields. Future work
    may introduce a node-shape-aware summarizer that preserves more
    non-sensitive structure.

    ``routes`` and ``trigger`` typing
    ---------------------------------
    ``routes`` is ``dict[str, str]``: route labels (``"true"``, ``"false"``,
    or custom labels) → sink/connection identifier strings (per the JSON
    schema at ``tools.py:715-722``).  The closed-list element type makes the
    field structurally exempt from the §4.4.2 Sensitive requirement: the
    walker descends to a ``str``-typed leaf, which is in the closed-list
    scalar set.

    ``trigger`` is a typed sub-model :class:`_NodeTriggerModel`.  Mirroring
    the JSON schema (``tools.py:735-755``) field-for-field makes every
    walked node a closed-list scalar, so the Sensitive requirement falls
    away.

    These two fields previously carried a Sensitive marker that satisfied
    the adequacy guard MECHANICALLY while losing useful route/trigger
    structure behind a generic option summary. Replacing the
    ``dict[str, Any]`` types with structural typings drops the markers and
    strengthens boundary validation (e.g., ``routes: {"true": 42}`` is now
    rejected at Pydantic validation instead of silently passing through to
    become an audit fact).  See F3 in
    ``docs/composer/evidence/composer-phase-2-followup-prompt-F1-F6.md``.
    """

    id: str
    options: _LlmJsonObject = Field(default_factory=dict)
    on_error: str | None = None
    node_type: str
    input: str
    plugin: str | None = None
    on_success: str | None = None
    condition: str | None = None
    routes: dict[str, str] | None = None
    fork_to: list[str] | None = None
    branches: list[str] | dict[str, str] | None = None
    policy: str | None = None
    merge: str | None = None
    trigger: _NodeTriggerModel | None = None
    output_mode: str | None = None
    expected_output_count: int | None = None

    model_config = ConfigDict(extra="forbid")


class _PipelineEdgeModel(BaseModel):
    """Nested model for ``set_pipeline.edges[*]``.

    Mirrors the JSON schema's ``edges`` array-item shape declared at
    ``tools.py:1062-1075`` and its inner ``required: ["id", "from_node",
    "to_node", "edge_type"]``.  ``label`` is optional; the handler reads
    it via ``e.get("label")``.

    No Sensitive markers on this model: every field is a scalar string
    naming a node id, edge id, edge type, or human-readable label —
    structurally not a leak surface.
    """

    id: str
    from_node: str
    to_node: str
    edge_type: str
    label: str | None = None

    model_config = ConfigDict(extra="forbid")


class _PipelineOutputModel(BaseModel):
    """Nested model for ``set_pipeline.outputs[*]``.

    Mirrors the JSON schema's ``outputs`` array-item shape declared at
    ``tools.py:1077-1115`` and its inner ``required: ["sink_name",
    "plugin"]``.

    ``options`` carries :class:`Sensitive` with
    :func:`_summarize_set_source_options` — sink options routinely carry
    filesystem paths (``path: "outputs/results.json"``), secret-ref
    markers, and other operator-sensitive payload values.  See the
    docstring on :class:`_PipelineNodeModel.options` for the rationale
    behind reusing the source-side summarizer at the node/output
    boundary.

    ``on_write_failure`` preserves operator-omitted-vs-specified semantics
    so the handler can apply its ``"discard"`` fallback explicitly.
    """

    sink_name: str
    plugin: str
    options: _LlmJsonObject = Field(default_factory=dict)
    on_write_failure: str | None = None

    model_config = ConfigDict(extra="forbid")


class _PipelineMetadataModel(BaseModel):
    """Nested model for ``set_pipeline.metadata``.

    Mirrors the JSON schema's ``metadata`` sub-object declared at
    ``tools.py:1122-1129``.  Both fields are optional; the handler at
    :func:`_execute_set_pipeline` constructs
    :class:`elspeth.web.composer.state.PipelineMetadata` only with the
    explicitly-supplied subset and lets the dataclass defaults fill in
    the rest.
    """

    name: str | None = None
    description: str | None = None

    model_config = ConfigDict(extra="forbid")


class ApplyPipelineRecipeArgumentsModel(BaseModel):
    """Redaction-bearing argument model for the ``apply_pipeline_recipe`` tool.

    Mirrors the JSON schema declared at ``tools.py:1462-1485`` and its
    ``required: ["recipe_name", "slots"]``.  ``apply_pipeline_recipe``
    delegates to :func:`_execute_set_pipeline` after composing the
    full-pipeline arguments from operator-supplied slot values; that
    inner call goes through :class:`SetPipelineArgumentsModel` so the
    recipe-built args receive the same validation discipline as a
    hand-authored ``set_pipeline`` call.

    Sensitive marker on ``slots``
    -----------------------------
    Recipe slots routinely carry:
      * filesystem paths (e.g., ``output_path: "outputs/results.jsonl"``),
      * secret-ref names (e.g., ``api_key_secret: "OPENROUTER_API_KEY"``),
      * blob references (UUID strings, e.g., ``source_blob_id``),
      * LLM prompt templates (e.g., ``classifier_template``).

    Marking ``slots`` :class:`Sensitive` with
    :func:`_summarize_set_source_options` collapses the dict to the
    canonical-JSON shape summary. The summarizer is reused (NOT a
    recipe-specific one) for the same structural reasons as
    :class:`_PipelineNodeModel.options`: recipe slots may carry paths,
    secret names, blob ids, or prompt templates, so scalar values are not
    persisted. A future recipe-shape-aware summarizer may preserve more
    non-sensitive structure.

    Empty-string semantic check on ``recipe_name``
    ----------------------------------------------
    The Pydantic model accepts ``recipe_name: str`` including the empty
    string.  The handler at :func:`_execute_apply_pipeline_recipe`
    re-checks for emptiness AFTER Pydantic validation and produces a
    repair-hinting ``_failure_result`` ("Call list_recipes to discover
    available recipes") rather than a bare ``ToolArgumentError``.  We
    deliberately do NOT use ``Field(min_length=1)`` here: the
    repair-hint message is recoverable LLM feedback, whereas a
    ``ValidationError`` for ``min_length`` would surface as an ARG_ERROR
    with the generic envelope text and no recipe-discovery guidance.
    Two channels for two failure shapes (type vs semantic) — same
    pattern as :class:`SetSourceArgumentsModel` plugin-not-in-catalog
    handling.

    ``extra="forbid"`` is required (rev-2 M.1).  Fields belonging to
    neighbouring tools (e.g., ``source``, ``nodes`` on ``set_pipeline``)
    are intentionally absent so ``extra="forbid"`` rejects misrouted
    argument shapes early.
    """

    recipe_name: str
    slots: _LlmJsonObject

    model_config = ConfigDict(extra="forbid")


class SetPipelineArgumentsModel(BaseModel):
    """Redaction-bearing argument model for the ``set_pipeline`` tool.

    Mirrors the JSON schema declared at ``tools.py:940-1132`` for the
    ``set_pipeline`` definition and its required-paths (``source``,
    ``nodes``, ``edges``, ``outputs`` at the top level; nested required
    fields per :class:`_SetPipelineSourceModel`, :class:`_PipelineNodeModel`,
    :class:`_PipelineEdgeModel`, :class:`_PipelineOutputModel`).
    ``metadata`` is optional at the top level.

    LLM-supplied vs dispatcher-wired arguments
    ------------------------------------------
    The dispatcher at :func:`execute_tool` (``tools.py:5530-5540``) calls
    :func:`_execute_set_pipeline` with additional kwargs (``session_engine``,
    ``session_id``) that are NOT part of the LLM-supplied ``arguments``
    dict — they are wired by the composer service from the request context.
    This Pydantic model validates only the LLM-supplied dict; the kwargs
    enter through the handler's function signature, not through Pydantic.

    Sensitive marker surface
    ------------------------
    Four paths carry :class:`Sensitive` markers (see module-level note above):
      * ``source.options`` — :func:`_summarize_set_source_options`.
      * ``source.inline_blob.content`` — :func:`_summarize_inline_blob_content`.
      * ``nodes[*].options`` — :func:`_summarize_set_source_options`.
      * ``outputs[*].options`` — :func:`_summarize_set_source_options`.

    The adequacy guard (§4.4.2) fails closed on any ``dict[str, Any]`` or
    ``Any``-typed field without a Sensitive marker, which mechanically
    enforces that every LLM-supplied open option surface is redacted at the
    persistence boundary. Routes and triggers use structural closed-leaf
    types instead.

    ``extra="forbid"`` is required (rev-2 M.1) at every level — the
    outer model AND every nested sub-model.  Misrouted argument shapes
    (e.g., ``filename`` at the top level — that belongs on
    ``create_blob``) are rejected before any handler-side logic runs.
    """

    source: _SetPipelineSourceModel | None = None
    sources: dict[str, _SetPipelineNamedSourceModel] | None = None
    nodes: list[_PipelineNodeModel]
    edges: list[_PipelineEdgeModel]
    outputs: list[_PipelineOutputModel]
    metadata: _PipelineMetadataModel | None = None

    model_config = ConfigDict(extra="forbid")


# ---------------------------------------------------------------------------
# patch_source_options / patch_node_options / patch_output_options
# argument models.
#
# All three tools accept a ``patch`` argument — a merge-patch dict that is
# applied to the current plugin-options dict via :func:`_apply_merge_patch`.
# The patch carries the same content surface as the original ``options`` on
# ``set_source``, ``upsert_node``, and ``set_output`` respectively: plugin
# option keys including credential-ref markers
# (``api_key: {"secret_ref": "NAME"}``), filesystem paths
# (``path: "outputs/results.json"``), and prompt-template payloads.
#
# ``patch: dict[str, Any]`` without a Sensitive annotation would fail the
# adequacy guard (§4.4.2): the value type resolves to ``Any``, which is
# inspection-resistant — the guard fails closed on that shape.  Wrapping
# ``patch`` with :class:`Sensitive` and :func:`_summarize_set_source_options`
# is mandatory, and reusing the source-side summarizer is structurally sound
# because it records option shape while replacing scalar values. Uniformity
# with the Waves 2-3 source/node/output options surfaces is an explicit
# design goal.
#
# ``patch_node_options`` adds a ``node_id: str`` selector; the routing-key
# guard (_node_routing_option_patch_error) is a SEMANTIC check on patch
# contents (on_success, on_error, input, routes, fork_to), not a shape check.
# It runs after Pydantic validation and is NOT replicated in the model —
# same pattern as set_pipeline.source blob_id/inline_blob mutual exclusion
# which the handler checks post-validation.
#
# ``patch_output_options`` adds a ``sink_name: str`` selector.
# ---------------------------------------------------------------------------


class PatchSourceOptionsArgumentsModel(BaseModel):
    """Redaction-bearing argument model for the ``patch_source_options`` tool.

    Mirrors the JSON schema declared at ``tools.py:883-897`` for the
    ``patch_source_options`` definition and its ``required: ["patch"]``.
    ``source_name`` selects the named source root and is not sensitive.

    ``patch`` carries :class:`Sensitive` with
    :func:`_summarize_set_source_options` — the merge-patch dict has the same
    content surface as ``set_source.options``: plugin option keys including
    credential-ref markers, filesystem paths, and prompt-template payloads.
    Reusing :func:`_summarize_set_source_options` maintains the
    uniformity-across-source-binding-tools contract from Tasks 4 / 13.

    ``extra="forbid"`` is required (rev-2 M.1).  Fields belonging to
    ``patch_node_options`` (``node_id``) and ``patch_output_options``
    (``sink_name``) are intentionally absent so ``extra="forbid"``
    rejects misrouted argument shapes early.
    """

    source_name: str = "source"
    patch: _LlmJsonObject

    model_config = ConfigDict(extra="forbid")


class PatchNodeOptionsArgumentsModel(BaseModel):
    """Redaction-bearing argument model for the ``patch_node_options`` tool.

    Mirrors the JSON schema declared at ``tools.py:899-922`` for the
    ``patch_node_options`` definition and its ``required: ["node_id", "patch"]``.

    ``node_id`` is a plain string naming the target node — structural
    identity, no sensitive surface.

    ``patch`` carries :class:`Sensitive` with
    :func:`_summarize_set_source_options` — node plugin-option dicts carry
    the same content surface as ``set_pipeline.nodes[*].options``
    (:class:`_PipelineNodeModel.options` is already marked Sensitive here).
    Reusing :func:`_summarize_set_source_options` maintains
    uniformity-across-node-options-tools.

    Post-validation semantic check: the routing-key guard
    (:func:`_node_routing_option_patch_error`) rejects routing-field keys
    (``on_error``, ``on_success``, ``input``, ``routes``, ``fork_to``) in
    ``patch`` and is a value-domain check that Pydantic cannot express.
    It runs inside ``_execute_patch_node_options`` AFTER this model's
    validation — the same discipline as ``set_pipeline``'s
    ``blob_id`` / ``inline_blob`` mutual-exclusion check that lives in
    ``_execute_set_pipeline`` post-validation.

    ``extra="forbid"`` is required (rev-2 M.1).  Fields belonging to
    ``patch_source_options`` (no ``node_id``) and ``patch_output_options``
    (``sink_name``) are intentionally absent.
    """

    node_id: str
    patch: _LlmJsonObject

    model_config = ConfigDict(extra="forbid")


class PatchOutputOptionsArgumentsModel(BaseModel):
    """Redaction-bearing argument model for the ``patch_output_options`` tool.

    Mirrors the JSON schema declared at ``tools.py:925-942`` for the
    ``patch_output_options`` definition and its ``required: ["sink_name", "patch"]``.

    ``sink_name`` is a plain string naming the target output — structural
    identity, no sensitive surface.

    ``patch`` carries :class:`Sensitive` with
    :func:`_summarize_set_source_options` — sink option dicts carry the same
    content surface as ``set_pipeline.outputs[*].options``
    (:class:`_PipelineOutputModel.options` is already marked Sensitive here).
    Reusing :func:`_summarize_set_source_options` maintains
    uniformity-across-output-options-tools.

    ``extra="forbid"`` is required (rev-2 M.1).  Fields belonging to
    ``patch_source_options`` (no selector) and ``patch_node_options``
    (``node_id`` vs ``sink_name``) are intentionally absent.
    """

    sink_name: str
    patch: _LlmJsonObject

    model_config = ConfigDict(extra="forbid")


def _redact_via_schema(
    tool_name: str,
    validated: BaseModel,
    model_cls: type[BaseModel],
    *,
    telemetry: RedactionTelemetry,
) -> dict[str, Any]:
    """Walk the validated model's schema; substitute Sensitive fields.

    Operates on ``model_dump()`` output (a plain dict) and substitutes
    in-place on a deep copy so the input model and any external caller
    references are not affected.  Returns the dict ready for serialization.

    Sensitive fields with no summarizer receive ``REDACTED_SENSITIVE_NO_SUMMARIZER``.
    Summarizer failures fire ``telemetry.summarizer_error(tool_name=...)``
    BEFORE raising ``AuditIntegrityError`` (rev-2 M.8 discipline).

    Path coverage: top-level, nested-BaseModel,
    list-element, dict-element, and tuple-element paths are all supported
    via the per-node ``substitute_provider`` closure built alongside
    ``value_provider`` in ``walk_model_schema(with_values=True)``.

    Walker atomicity (rev-3 W8b / rev-4 W8b): substitutions mutate a local
    ``dumped`` dict.  When ``transform`` raises mid-walk, the partial dict
    is discarded by the unwind; only a fully successful walk returns it to
    the caller.
    """
    dumped = copy.deepcopy(validated.model_dump())
    for node in walk_model_schema(model_cls, with_values=True):
        marker = next((m for m in node.metadata if isinstance(m, _SensitiveMarker)), None)
        if marker is None:
            continue
        if node.substitute_provider is None:
            # walk_model_schema(with_values=True) guarantees substitute_provider
            # is populated; a None here is a walker bug, not external input.
            raise AuditIntegrityError(
                f"walk_model_schema yielded TraversalNode at {node.path!r} "
                "with substitute_provider=None despite with_values=True. "
                "This is an internal walker contract violation."
            )
        if marker.summarizer is None:
            # No summarizer: substitute the no-summarizer sentinel at every
            # reachable leaf for this path (top-level, nested, or container).
            # Declarative sensitive keys always use this path; type-driven
            # Sensitive() fields without a summarizer keyword also land here.
            node.substitute_provider(dumped, lambda _v: REDACTED_SENSITIVE_NO_SUMMARIZER)
            continue
        # Narrow marker.summarizer from `Callable | None` to `Callable` by
        # rebinding to a local before forming the closure (the early
        # `continue` on `marker.summarizer is None` above does not propagate
        # through the loop body's closure-default position).
        summarizer: Callable[[Any], str] = marker.summarizer

        # Bind summarizer, tool_name, and node.path into the closure via
        # default-argument captures (B023): the loop reassigns these names
        # on each iteration, so capturing through the enclosing scope would
        # have every closure see the LAST iteration's values.
        def _apply(
            raw_value: Any,
            *,
            _summarizer: Callable[[Any], str] = summarizer,
            _tool_name: str = tool_name,
            _path: str = node.path,
        ) -> Any:
            try:
                summary = _summarizer(raw_value)
            except Exception as exc:
                telemetry.summarizer_error(tool_name=_tool_name)  # BEFORE raise (rev-2 M.8)
                raise AuditIntegrityError(f"Summarizer for {_tool_name!r} path {_path!r} raised {type(exc).__name__}: {exc}") from exc
            if not isinstance(summary, str):
                # _SensitiveMarker.summarizer is typed as Callable[[Any], str]; mypy
                # considers this branch unreachable after isinstance narrowing.  We
                # keep it as an offensive invariant guard: a misconfigured or mocked
                # summarizer may violate the return-type contract at runtime
                # (spec §4.2.6 M.8).  type: ignore suppresses the false-positive.
                telemetry.summarizer_error(tool_name=_tool_name)  # type: ignore[unreachable]  # BEFORE raise (rev-2 M.8)
                raise AuditIntegrityError(
                    f"Summarizer for {_tool_name!r} path {_path!r} returned {type(summary).__name__}, expected str (spec §4.2.6)."
                )
            return summary

        node.substitute_provider(dumped, _apply)
    return dumped


def _redact_via_policy(
    tool_name: str,
    arguments: dict[str, Any],
    policy: ToolRedactionPolicy,
    *,
    telemetry: RedactionTelemetry,
) -> dict[str, Any]:
    """Walk ``arguments`` against a declarative policy.

    Spec §4.2.6 disposition table:
      * Argument summarizer key declared but argument key absent in input →
        no-op (key absence is not a fault; Tier-3 input may omit any key).
      * Argument summarizer key declared AND present in input AND a summarizer
        is registered for it → summarizer output substitutes the value.
      * Argument summarizer key declared AND present in input AND no summarizer
        is registered → REDACTED_SENSITIVE_NO_SUMMARIZER substitutes the value
        (spec §4.3 line 1073: "Plain sensitive key → value replaced by literal
        string '<redacted>'.").
      * Argument key NOT in sensitive_argument_keys → passthrough by default
        for legacy declarative entries. If ``policy.known_argument_keys`` is
        declared or ``policy.redact_unknown_argument_keys`` is enabled, keys
        outside the allowlist are removed and replaced with a generic
        REDACTED_UNKNOWN_ARGUMENT_KEY marker under
        REDACTED_UNKNOWN_ARGUMENTS_FIELD. The boolean allows a closed empty
        argument surface for no-argument tools.

    Walker atomicity (rev-3 W8b / rev-4 W8b): the output dict is built in a
    local variable and only returned on success.  A mid-walk raise leaves no
    partial dict observable to the caller.
    """
    # Build the output dict by shallow-copying the input, then substituting
    # values for every sensitive key present.  Atomicity comes from the fact
    # that any raise during summarization aborts before we hand the dict back.
    # Non-sensitive keys are passthrough unless the policy opts into a closed
    # argument allowlist.
    redacted: dict[str, Any] = dict(arguments)
    if policy.known_argument_keys or policy.redact_unknown_argument_keys:
        known_argument_keys = set(policy.known_argument_keys)
        unknown_keys = [key for key in arguments if key not in known_argument_keys]
        for key in unknown_keys:
            redacted.pop(key, None)
        if unknown_keys:
            redacted[REDACTED_UNKNOWN_ARGUMENTS_FIELD] = REDACTED_UNKNOWN_ARGUMENT_KEY

    summarizers = policy.argument_summarizers
    for key in policy.sensitive_argument_keys:
        if key not in arguments:
            # Key absence is not a fault (Tier-3 input).
            continue
        if key not in summarizers:
            # Plain sensitive key, no summarizer registered → no-summarizer sentinel.
            redacted[key] = REDACTED_SENSITIVE_NO_SUMMARIZER
            continue
        summarizer = summarizers[key]
        raw_value = arguments[key]
        try:
            summary = summarizer(raw_value)
        except Exception as exc:
            telemetry.summarizer_error(tool_name=tool_name)  # BEFORE raise (rev-2 M.8)
            raise AuditIntegrityError(f"Summarizer for {tool_name!r} argument key {key!r} raised {type(exc).__name__}: {exc}") from exc
        if not isinstance(summary, str):
            # ToolRedactionPolicy.argument_summarizers is typed as Mapping[str,
            # Callable[[Any], str]]; mypy considers this branch unreachable
            # after the isinstance narrowing.  We keep it as an offensive
            # invariant guard: a misconfigured or mocked summarizer may
            # violate the return-type contract at runtime (spec §4.2.6 M.8).
            telemetry.summarizer_error(tool_name=tool_name)  # type: ignore[unreachable]  # BEFORE raise (rev-2 M.8)
            raise AuditIntegrityError(
                f"Summarizer for {tool_name!r} argument key {key!r} returned {type(summary).__name__}, expected str (spec §4.2.6)."
            )
        redacted[key] = summary
    return redacted


def redact_tool_call_arguments(
    tool_name: str,
    arguments: dict[str, Any],
    *,
    telemetry: RedactionTelemetry,
) -> dict[str, Any]:
    """Produce the redacted arguments dict that lands in
    ``chat_messages.tool_calls`` JSON (spec §4.2.6).

    Disposition by manifest-entry shape:

    **Type-driven entry (``argument_model`` set):**
      Emit the ``manifest_dispatch`` beacon (BEFORE ``model_validate`` so a
      Tier-3 ``ValidationError`` still records the dispatch); validate the
      raw arguments against ``argument_model``; walk via
      :func:`_redact_via_schema`. ``Sensitive[T]`` fields at any path
      (top-level, nested, list-element, dict-element, tuple-element) are
      substituted with the summarizer output or
      ``REDACTED_SENSITIVE_NO_SUMMARIZER`` when no summarizer is declared.

      The handler MUST catch :class:`pydantic.ValidationError` raised here
      and re-raise as :class:`ToolArgumentError` so the compose loop's
      ARG_ERROR routing at ``service.py:2480`` receives the right exception
      class.  A bare ``ValidationError`` escaping the handler hits the
      catch-all and becomes :class:`ComposerPluginCrashError` → HTTP 500 —
      the wrong disposition for Tier-3 input.

    **Declarative entry (``policy`` set):**
      Emit the ``manifest_dispatch`` beacon; walk ``arguments`` by
      ``policy.sensitive_argument_keys``.  Missing keys are no-ops; present
      keys are summarized (via ``policy.argument_summarizers[key]``) or
      sentinel-substituted (``REDACTED_SENSITIVE_NO_SUMMARIZER``).  Keys
      not in ``sensitive_argument_keys`` are passthrough unless the policy
      declares ``known_argument_keys`` or enables
      ``redact_unknown_argument_keys``; in that case unknown argument key names
      are removed and replaced with a generic ``REDACTED_UNKNOWN_ARGUMENT_KEY``
      marker under ``REDACTED_UNKNOWN_ARGUMENTS_FIELD``.

    **Failure modes (all raise AuditIntegrityError):**
      - Manifest entry missing for ``tool_name`` (registry-consistency
        invariant; distinct from Tier-3 LLM-hallucinated tool name which is
        caught earlier in the dispatcher).
      - Summarizer raises → ``telemetry.summarizer_error(tool_name=...)``
        BEFORE raise; ``AuditIntegrityError`` chained from the underlying
        exception (M.8).
      - Summarizer returns non-str → ``telemetry.summarizer_error(...)``
        BEFORE raise; ``AuditIntegrityError`` with a typed message.

    **Walker atomicity:** the output dict is built in a local variable and
    only returned on success.  A mid-walk raise leaves no partial dict
    observable to the caller (rev-3 W8b / rev-4 W8b).
    """
    if tool_name not in MANIFEST:
        raise AuditIntegrityError(
            f"redact_tool_call_arguments called for unknown tool {tool_name!r}; "
            "the manifest is the source of truth for the registry/redaction "
            "set-equality invariant.  Add a manifest entry for this tool or "
            "verify that the dispatch path passes the correct tool name."
        )
    entry = MANIFEST[tool_name]
    if entry.argument_model is not None:
        # Emit BEFORE model_validate: dispatch happened regardless of Tier-3
        # validation outcome.  A ValidationError escapes here and the handler
        # wraps it as ToolArgumentError; the dispatch beacon must not depend on
        # validation success.
        telemetry.manifest_dispatch(tool_name=tool_name, shape="type_driven")
        validated = entry.argument_model.model_validate(arguments)
        return _redact_via_schema(tool_name, validated, entry.argument_model, telemetry=telemetry)
    # Declarative branch (entry.policy is not None — ToolRedaction.__post_init__
    # guarantees exactly one of {argument_model, policy} is set).
    telemetry.manifest_dispatch(tool_name=tool_name, shape="declarative")
    policy = entry.policy
    assert policy is not None  # offensive: satisfies the type-checker contract
    return _redact_via_policy(tool_name, arguments, policy, telemetry=telemetry)


# ---------------------------------------------------------------------------
# Declarative manifest entries — _DISCOVERY_TOOLS, 12 tools.
#
# Every tool in ``_DISCOVERY_TOOLS`` (tools.py:5500-5513) is read-only over the
# composer state and returns only cached/derived plugin-registry metadata,
# validator output, or structural state summaries.  None of these handlers
# accept LLM-supplied dict-shaped option payloads or return raw blob/secret
# content.  Each is declared with ``handles_no_sensitive_data=True`` and a
# distinct ``HandlesNoSensitiveDataReason`` that an auditor can read to
# understand WHY this specific tool's argument/response surface cannot carry
# sensitive material — copy-paste justification is rejected at CI by the
# mass-copy uniqueness assertion (§4.4.4).
#
# The §4.4.2 ``dict[str, Any]`` fail-closed rule applies only to type-driven
# entries (Pydantic models walked via ``walk_model_schema``).  Declarative
# entries express their argument surface through ``sensitive_argument_keys``;
# all _DISCOVERY_TOOLS take only scalar string arguments (or no arguments
# at all), so ``sensitive_argument_keys`` is empty for every entry here.
# ---------------------------------------------------------------------------


_LIST_SOURCES_REASON = HandlesNoSensitiveDataReason(
    sensitive_data_locations=("plugin registry catalogue — cached, read-only metadata served from CatalogService",),
    why_arguments_safe=(
        "list_sources accepts no arguments — the JSON schema at tools.py:604 declares "
        "an empty properties object with empty required, so the LLM cannot place any "
        "value on this surface; the handler reads zero keys from the arguments dict."
    ),
    why_responses_safe=(
        "Response is the cached source-plugin descriptor list (name + summary per plugin) "
        "produced by CatalogService.list_sources; it is registry metadata composed at "
        "module import time, carries no user payload, and never references credentials."
    ),
)


_LIST_TRANSFORMS_REASON = HandlesNoSensitiveDataReason(
    sensitive_data_locations=("plugin registry catalogue — cached, read-only metadata served from CatalogService",),
    why_arguments_safe=(
        "list_transforms accepts no arguments — the JSON schema at tools.py:609 declares "
        "an empty properties object with empty required, so no LLM-supplied content can "
        "reach the dispatch site; the handler reads zero keys from the arguments dict."
    ),
    why_responses_safe=(
        "Response is the cached transform-plugin descriptor list (name + summary per plugin) "
        "produced by CatalogService.list_transforms; it is registry metadata composed at "
        "module import time, carries no operator data, and never references row content."
    ),
)


_LIST_SINKS_REASON = HandlesNoSensitiveDataReason(
    sensitive_data_locations=("plugin registry catalogue — cached, read-only metadata served from CatalogService",),
    why_arguments_safe=(
        "list_sinks accepts no arguments — the JSON schema at tools.py:614 declares an "
        "empty properties object with empty required, so the LLM cannot place any value "
        "on this surface; the handler reads zero keys from the arguments dict."
    ),
    why_responses_safe=(
        "Response is the cached sink-plugin descriptor list (name + summary per plugin) "
        "produced by CatalogService.list_sinks; it is registry metadata composed at "
        "module import time, carries no destination credentials, and never references payload."
    ),
)


_GET_PLUGIN_SCHEMA_REASON = HandlesNoSensitiveDataReason(
    sensitive_data_locations=("plugin registry catalogue — published JSON schema definitions served from CatalogService",),
    why_arguments_safe=(
        "get_plugin_schema arguments are two scalar strings (plugin_type, name) selecting "
        "a plugin in the catalogue; neither is operator-supplied content and the catalogue "
        "rejects unknown values with ValueError surfaced as a tool-failure result."
    ),
    why_responses_safe=(
        "Response is the published plugin configuration schema — a Pydantic schema dict that "
        "names options and types but contains no operator-supplied values, no credentials, "
        "and no row payloads; it is the same schema available via plugin documentation."
    ),
)


_GET_EXPRESSION_GRAMMAR_REASON = HandlesNoSensitiveDataReason(
    sensitive_data_locations=("gate-expression grammar reference — static reference text returned by get_expression_grammar",),
    why_arguments_safe=(
        "get_expression_grammar accepts no arguments — the JSON schema at tools.py:640 "
        "declares an empty properties object with empty required, so no LLM-supplied "
        "content can reach the dispatch site; the handler returns a constant grammar."
    ),
    why_responses_safe=(
        "Response is the static gate-expression syntax reference returned by "
        "get_expression_grammar(); it is a documentation string composed at module load "
        "time, identical for every invocation, and carries no session or operator data."
    ),
)


_EXPLAIN_VALIDATION_ERROR_REASON = HandlesNoSensitiveDataReason(
    sensitive_data_locations=("validation-error explainer — text-pattern lookup against canned guidance strings",),
    why_arguments_safe=(
        "explain_validation_error accepts a single error_text scalar string the LLM "
        "echoes back from a previous validator response; that text was itself emitted "
        "by the validator and contains no operator-supplied row content or credentials."
    ),
    why_responses_safe=(
        "Response is canned guidance text plus the matched error category; it is a "
        "lookup against module-level guidance tables and contains no session state, "
        "no plugin option values, and no row payload — only fix-it explanations."
    ),
)


_GET_PLUGIN_ASSISTANCE_REASON = HandlesNoSensitiveDataReason(
    sensitive_data_locations=(
        "plugin-published assistance text — registry-side guidance keyed by issue_code or, for discovery, plugin identity",
    ),
    why_arguments_safe=(
        "get_plugin_assistance arguments are three enum/scalar strings (plugin_type ∈ "
        "{source, transform, sink}, plugin_name, optional issue_code) selecting "
        "plugin-published guidance; none carry operator data, and unknown values surface "
        "as a tool-failure result rather than passing payload to the handler."
    ),
    why_responses_safe=(
        "Response is the plugin's published guidance struct (summary, suggested_fixes, "
        "examples, composer_hints); it is documentation authored at plugin packaging "
        "time, carries no session payload, and contains no credentials or row content."
    ),
)


_LIST_MODELS_REASON = HandlesNoSensitiveDataReason(
    sensitive_data_locations=("LLM provider model catalogue — names and counts, no credentials or completions",),
    why_arguments_safe=(
        "list_models accepts only an optional provider prefix string and an optional "
        "integer limit; both are pure filtering parameters and the handler validates "
        "them against the provider registry without touching session state or secrets."
    ),
    why_responses_safe=(
        "Response is a provider summary or model-id list scoped by the provider filter; "
        "model identifiers are public catalogue values published by the LLM providers "
        "themselves, and the handler never includes API keys, completions, or PII."
    ),
)


_LIST_RECIPES_REASON = HandlesNoSensitiveDataReason(
    sensitive_data_locations=("pipeline recipe registry — named recipe declarations bundled at packaging time",),
    why_arguments_safe=(
        "list_recipes accepts no arguments — the JSON schema at tools.py:1466 declares "
        "an empty properties object with empty required, so the LLM cannot place any "
        "value on this surface; the handler enumerates the static recipe registry."
    ),
    why_responses_safe=(
        "Response is the static recipe registry (recipe_name + slot schema per recipe) "
        "produced by list_recipes(); recipes are bundled scaffolds composed at packaging "
        "time and contain no operator slot values — only the slot-schema declarations."
    ),
)


_GET_AUDIT_INFO_REASON = HandlesNoSensitiveDataReason(
    sensitive_data_locations=(
        "Landscape audit-trail metadata — constant strings describing that audit is "
        "mandatory and operator-managed; the audit URL/DSN/backend type is intentionally "
        "NOT included (security fix S1, yaml_generator.py:179).",
    ),
    why_arguments_safe=(
        "get_audit_info accepts no arguments — the JSON schema at tools.py:1306 declares "
        "an empty properties object with empty required, so the LLM cannot place any "
        "value on this surface; the handler returns a hard-coded literal payload that "
        "does not depend on the call arguments at all."
    ),
    why_responses_safe=(
        "Response is a constant dict of operator-facing strings explaining that audit is "
        "always enabled and the backend is not composer-modifiable. The handler "
        "(_execute_get_audit_info, tools.py:5106) constructs the payload from string "
        "literals only — no WebSettings access, no DSN, no path, no encryption-key env "
        "var, no operator-internal config reaches the LLM."
    ),
)


_GET_PIPELINE_STATE_REASON = HandlesNoSensitiveDataReason(
    sensitive_data_locations=("in-memory composition state — structural metadata about the session pipeline",),
    why_arguments_safe=(
        "get_pipeline_state accepts only an optional component selector string (a node id, "
        "an output name, 'source', or a full-state alias); the handler indexes into the "
        "composer state by this name and never echoes LLM-supplied payload into the result."
    ),
    why_responses_safe=(
        "Response mirrors the operator's own composition — plugin names, node ids, and "
        "connection wiring — already visible to the operator in the composer UI; the "
        "redaction policy for the underlying mutation tools (set_source, set_pipeline, "
        "patch_*_options) governs how option values reach this state in the first place."
    ),
)


_PREVIEW_PIPELINE_REASON = HandlesNoSensitiveDataReason(
    sensitive_data_locations=("validation snapshot — composition-state validator output plus runtime preflight",),
    why_arguments_safe=(
        "preview_pipeline accepts no arguments — the JSON schema at tools.py:1296 declares "
        "an empty properties object with empty required, so the LLM cannot place any "
        "value on this surface; the handler reads the current composer state directly."
    ),
    why_responses_safe=(
        "Response is the validation summary plus structural source/node/output overview "
        "and proof diagnostics. The proof step DOES read and parse the bound source blob's "
        "bytes and evaluate sampled rows (it does not run the full pipeline), but every "
        "diagnostic that derives from those bytes redacts the observed headers/columns to "
        "counts and withholds the raw resolver/evaluation error text — mirroring the "
        "deliberate header redaction in compute_proof_diagnostics — so no raw row payload "
        "or observed-value PII crosses this surface. Validator entries carry component, "
        "severity, and a human-readable message in the same class as every other composer "
        "tool's validation surface (the message may name a declared field or path)."
    ),
)


_DIFF_PIPELINE_REASON = HandlesNoSensitiveDataReason(
    sensitive_data_locations=("pre/post state diff — structural change list versus the session baseline",),
    why_arguments_safe=(
        "diff_pipeline accepts no arguments — the JSON schema at tools.py:1323 declares "
        "an empty properties object with empty required, so the LLM cannot place any "
        "value on this surface; the handler diffs current state against the baseline."
    ),
    why_responses_safe=(
        "Response is the structural change-set (added/removed/modified nodes/edges/outputs) "
        "produced by diff_states; both sides of the diff are the operator's own pipeline "
        "composition and the diff reports field names plus before/after summaries — not "
        "row payload — so no execution data crosses this surface."
    ),
)


# ---------------------------------------------------------------------------
# Declarative manifest entries — _MUTATION_TOOLS remaining,
# 8 tools).
#
# Excluding set_source / set_pipeline / patch_*_options, the
# remaining ``_MUTATION_TOOLS`` are graph-shape editors that take node-ids,
# edge-ids, sink names, edge kinds, route slots, and (for upsert_node and
# set_output) an option dict.  The option-dict-bearing tools declare
# ``sensitive_argument_keys`` so the option payload is redacted at the
# argument boundary; the §4.4.2 ``dict[str, Any]`` fail-closed rule applies
# only to type-driven entries (Pydantic-walked argument models) and the
# declarative path expresses the same contract through the sensitive-keys
# tuple.
#
# Reuse of :func:`_summarize_set_source_options` is structurally sound: the
# helper is content-agnostic over option-like dicts because it preserves
# shape while replacing scalar values, so it applies to upsert_node options,
# set_output options, and trigger/routes slots without modification.  The
# same reuse rationale is documented in detail on
# :class:`_PipelineNodeModel.options` and :class:`_PipelineOutputModel.options`.
# ---------------------------------------------------------------------------


def _summarize_set_metadata_patch(patch: object) -> str:
    """Summarizer for ``set_metadata.patch``.

    The patch accepts only ``name`` and ``description`` fields per the tool
    schema — both operator-facing labels with no plugin options, no path
    references, and no credential markers.  The summarizer records only that
    allowlisted field names were present. Unknown patch keys are collapsed to
    a generic marker because key names are LLM-controlled text and may
    themselves carry secrets.

    Contract (spec §4.2.6, §9 RSK-03):
      * MUST NOT raise on any reachable input value.
      * MUST return ``str``.
    """
    if not isinstance(patch, Mapping):
        return "<metadata-patch:invalid>"
    allowed_keys = {"description", "name"}
    keys = sorted(key for key in patch if isinstance(key, str) and key in allowed_keys)
    if any(key not in allowed_keys for key in patch):
        keys.append("unknown")
    return f"<metadata-patch:{','.join(keys)}>" if keys else "<metadata-patch:empty>"


_UPSERT_EDGE_REASON = HandlesNoSensitiveDataReason(
    sensitive_data_locations=("graph topology — edge id, endpoints, kind, label — none are payload-bearing",),
    why_arguments_safe=(
        "upsert_edge accepts only graph-topology scalars (id, from_node, to_node, edge_type, "
        "optional label); the manifest argument allowlist preserves those structural fields "
        "and replaces unexpected LLM-supplied argument keys with the fixed unknown-argument "
        "sentinel before audit persistence."
    ),
    why_responses_safe=(
        "Response is the structural ToolResult — validation summary plus the edge id list "
        "after upsert; edges are pure connection-name strings and the response carries no "
        "plugin option values, no credentials, and no row payload from any execution path."
    ),
)


_REMOVE_NODE_REASON = HandlesNoSensitiveDataReason(
    sensitive_data_locations=("graph topology — single node id selector identifying the deletion target",),
    why_arguments_safe=(
        "remove_node accepts only a single scalar id string naming the node to delete; "
        "the manifest argument allowlist preserves that selector and replaces unexpected "
        "LLM-supplied argument keys with the fixed unknown-argument sentinel before audit "
        "persistence."
    ),
    why_responses_safe=(
        "Response is the structural ToolResult — validation summary for the post-removal "
        "state and the list of affected node ids; the deleted node's options are not "
        "echoed back, and the validation entries name fields by path without payload."
    ),
)


_REMOVE_EDGE_REASON = HandlesNoSensitiveDataReason(
    sensitive_data_locations=("graph topology — single edge id selector identifying the deletion target",),
    why_arguments_safe=(
        "remove_edge accepts only a single scalar id string naming the edge to delete; "
        "the manifest argument allowlist preserves that selector and replaces unexpected "
        "LLM-supplied argument keys with the fixed unknown-argument sentinel before audit "
        "persistence."
    ),
    why_responses_safe=(
        "Response is the structural ToolResult — validation summary for the post-removal "
        "state and the list of affected node ids; edges carry only connection names so "
        "the response surface has no payload, no credentials, and no row content."
    ),
)


_REMOVE_OUTPUT_REASON = HandlesNoSensitiveDataReason(
    sensitive_data_locations=("graph topology — single sink_name selector identifying the output to remove",),
    why_arguments_safe=(
        "remove_output accepts only a single scalar sink_name string naming the output "
        "to delete; the manifest argument allowlist preserves that selector and replaces "
        "unexpected LLM-supplied argument keys with the fixed unknown-argument sentinel "
        "before audit persistence."
    ),
    why_responses_safe=(
        "Response is the structural ToolResult — validation summary for the post-removal "
        "state and the affected node id list; the removed sink's option dict is not "
        "echoed back, so destination paths and credentials do not appear in the response."
    ),
)


_CLEAR_SOURCE_REASON = HandlesNoSensitiveDataReason(
    sensitive_data_locations=("composer state — selected in-memory source root being cleared",),
    why_arguments_safe=(
        "clear_source accepts only an optional source_name selector. Source names are "
        "structural composer identifiers, not plugin options, file paths, source content, "
        "or credentials; the manifest argument allowlist replaces unexpected LLM-supplied "
        "argument keys with the fixed unknown-argument sentinel before audit persistence."
    ),
    why_responses_safe=(
        "Response is the structural ToolResult — validation summary describing the "
        "now-source-less state; the cleared source's option dict is discarded and not "
        "echoed back, so the response surface has no path, no plugin options, no secrets."
    ),
)


# ---------------------------------------------------------------------------
# Declarative manifest entries — _BLOB_DISCOVERY_TOOLS, 5 tools.
#
# ``list_blobs``, ``list_composer_blobs``, ``get_blob_metadata``, and
# ``inspect_source`` return only structural blob facts (id, filename,
# mime_type, size, observed headers, inferred types) and never return raw blob
# content.  ``get_blob_content`` IS the content-returning tool and is the
# singular type-driven entry in this group with a response_model: declarative
# entries have no response_summarizers field, so the per-key byte-count
# summary required by the spec §4.7 disposition can only be applied via the
# schema walker on a response_model with ``Annotated[str,
# Sensitive(summarizer=...)]``.
# ---------------------------------------------------------------------------


_LIST_BLOBS_REASON = HandlesNoSensitiveDataReason(
    sensitive_data_locations=("session blob inventory — id/filename/mime_type/size_bytes per blob, no raw content",),
    why_arguments_safe=(
        "list_blobs accepts no arguments — the JSON schema declares an empty properties "
        "object with additionalProperties=false, and redaction strips any unknown keys "
        "before persistence; the handler enumerates the session inventory directly."
    ),
    why_responses_safe=(
        "Response is the blob-inventory list — operator-uploaded filenames, mime_types, "
        "and structural metadata per blob — but never the raw blob content; payload bytes "
        "are exposed only via get_blob_content whose policy applies a length-only summary."
    ),
)


_LIST_COMPOSER_BLOBS_REASON = HandlesNoSensitiveDataReason(
    sensitive_data_locations=("session blob inventory — id/filename/mime_type/size_bytes/content_hash per ready blob, no raw content",),
    why_arguments_safe=(
        "list_composer_blobs accepts no arguments — its JSON schema declares an "
        "empty properties object with additionalProperties=false, and redaction strips "
        "any unknown keys before persistence; the handler enumerates ready blobs from "
        "the session-scoped inventory."
    ),
    why_responses_safe=(
        "Response is the ADR-025 H4 visibility shape — blob_id, mime_type, "
        "size_bytes, content_hash, and filename. It deliberately excludes "
        "source_description, preview, content bytes, and storage_path so the LLM "
        "can author a pinned ref without seeing the referenced text."
    ),
)


_GET_BLOB_METADATA_REASON = HandlesNoSensitiveDataReason(
    sensitive_data_locations=("session blob metadata — filename / mime_type / size / status / hash, no content bytes",),
    why_arguments_safe=(
        "get_blob_metadata accepts a single blob_id scalar string selecting one inventory "
        "row; blob_ids are UUIDs assigned by the composer service and contain no operator "
        "payload; the schema and redaction allowlist reject or strip any other argument keys."
    ),
    why_responses_safe=(
        "Response is the single-blob metadata row — id, filename, mime_type, size_bytes, "
        "status, content_hash — operator-uploaded labels but never the raw bytes; payload "
        "exposure is gated to get_blob_content whose policy summarises content by length."
    ),
)


def _summarize_blob_content(content: str) -> str:
    """Summarizer for ``get_blob_content.content``.

    Discloses only the byte-length of the UTF-8 encoded content, never the
    bytes themselves.  Mirrors :func:`_summarize_inline_blob_content` in form
    so the audit trail's content-length signal is uniform across blob-write
    and blob-read tools (the LLM may funnel the same payload through
    create_blob / update_blob / get_blob_content).

    Contract (spec §4.2.6, §9 RSK-03):
      * MUST NOT raise on any reachable input value.
      * MUST return ``str``.
    """
    return f"<blob-content:{len(content.encode('utf-8'))}-bytes>"


def _summarize_repair_arguments(arguments: Mapping[str, object]) -> str:
    """Summarizer for ``get_blob_content`` repair-tool-call ``arguments``.

    The validation envelope's ``graph_repair_suggestions[*].tool_sequence[*].
    arguments`` slot is the one genuinely heterogeneous leaf in the
    ``get_blob_content`` response surface. It holds the keyword arguments the
    composer would pass to a repair tool (field paths, node ids, connection
    selectors) and varies per repair tool, so it cannot be closed-typed the way
    the surrounding structural fields can. It is NOT source-data lineage — it is
    derived, advisory repair guidance — so summarizing it to a structural sketch
    in ``chat_messages.tool_calls`` does not break attributability.

    The summary discloses only the SORTED argument key names, never their
    values. Keys are pipeline-structural identifiers (e.g. ``field_path``,
    ``node_id``) chosen by the composer's own repair planner, not operator
    payload, so naming them is non-sensitive and aids audit readability.

    Contract (spec §4.2.6, §9 RSK-03):
      * MUST NOT raise on any reachable input value.
      * MUST return ``str``.

    Pydantic's ``Mapping[str, object]`` validation guarantees a real mapping
    with string keys before this runs, so ``sorted(arguments)`` cannot raise on
    a mixed-key set. An empty mapping yields ``<repair-args:>`` which is a valid
    structural signal (the repair tool takes no arguments).
    """
    return f"<repair-args:{','.join(sorted(arguments))}>"


class _RepairToolCallShadowModel(BaseModel):
    """Redaction shadow for ``_RepairToolCall`` (tools/_common.py).

    Mirrors the serialized ``tool_sequence[*]`` dict: a closed ``tool`` name
    plus the heterogeneous ``arguments`` mapping. ``arguments`` is the only
    leaf in the entire ``get_blob_content`` validation envelope that cannot be
    closed-typed, so it carries a :class:`Sensitive` marker with a structural
    summarizer (:func:`_summarize_repair_arguments`). It is advisory repair
    guidance, not source-data lineage, so summarizing its keys is sound.
    """

    tool: str
    arguments: Annotated[Mapping[str, object], Sensitive(summarizer=_summarize_repair_arguments)]

    model_config = ConfigDict(extra="forbid")


class _AffectedConsumerShadowModel(BaseModel):
    """Redaction shadow for ``_AffectedConsumer`` (tools/_common.py).

    All three fields are pipeline-structural scalar strings (node id and the
    before/after input descriptors); none carries operator payload.
    """

    id: str
    current_input: str
    new_input: str

    model_config = ConfigDict(extra="forbid")


class _GraphRepairSuggestionShadowModel(BaseModel):
    """Redaction shadow for ``_GraphRepairSuggestion`` (tools/_common.py).

    Mirrors a serialized ``graph_repair_suggestions[*]`` dict. Every scalar
    field is a structural repair descriptor; the two nested lists descend into
    :class:`_AffectedConsumerShadowModel` and :class:`_RepairToolCallShadowModel`
    so the walker reaches the single Sensitive ``arguments`` leaf within.
    """

    code: str
    connection: str
    strategy: str
    reason: str
    affected_consumers: list[_AffectedConsumerShadowModel]
    tool_sequence: list[_RepairToolCallShadowModel]

    model_config = ConfigDict(extra="forbid")


class _SemanticEdgeContractShadowModel(BaseModel):
    """Redaction shadow for ``_SemanticEdgeContractPayload`` (tools/_common.py).

    Mirrors a serialized ``semantic_contracts[*]`` dict. All fields are
    structural edge-contract metadata (node ids, plugin names, field names,
    outcome / requirement codes). ``producer_plugin`` is ``str | None`` because
    an unconnected consumer edge has no producer.
    """

    from_id: str
    to_id: str
    consumer_plugin: str
    producer_plugin: str | None
    producer_field: str
    consumer_field: str
    outcome: str
    requirement_code: str

    model_config = ConfigDict(extra="forbid")


class _ValidationEntryShadowModel(BaseModel):
    """Redaction shadow for ``ValidationEntry.to_dict()`` (state.py).

    Mirrors a serialized validation message: ``component`` / ``message`` /
    ``severity`` are all plain strings (``severity`` serializes from the
    ``Severity`` literal alias to its string value). None carries operator
    payload — these are composer-authored diagnostics about pipeline shape.
    """

    component: str
    message: str
    severity: str

    model_config = ConfigDict(extra="forbid")


class GetBlobContentValidationModel(BaseModel):
    """Redaction shadow for the serialized ``ToolResult`` validation envelope.

    Faithfully mirrors the ``"validation"`` dict built in
    ``ToolResult.to_dict()`` (tools/_common.py ~591-605). This replaces the
    previous ``validation: dict[str, Any]`` on
    :class:`GetBlobContentResponseModel`, which was an ``Any``-typed
    redaction-bypass surface the adequacy guard (§4.4.2) fails closed on.

    The validation envelope is NON-sensitive pipeline-validation metadata, not
    blob bytes, so the field as a whole is intentionally NOT marked Sensitive —
    redacting it would wrongly strip useful diagnostics from the audit trail.
    Every leaf here is a closed scalar EXCEPT the single
    ``graph_repair_suggestions[*].tool_sequence[*].arguments`` mapping, which
    carries its own Sensitive structural summarizer in
    :class:`_RepairToolCallShadowModel`.
    """

    is_valid: bool
    errors: list[_ValidationEntryShadowModel]
    warnings: list[_ValidationEntryShadowModel]
    suggestions: list[_ValidationEntryShadowModel]
    semantic_contracts: list[_SemanticEdgeContractShadowModel]
    graph_repair_suggestions: list[_GraphRepairSuggestionShadowModel]

    model_config = ConfigDict(extra="forbid")


class _RequestInterpretationReviewPendingDataModel(BaseModel):
    """Correlation-only response payload for a pending interpretation review."""

    kind_marker: Literal["interpretation_review_pending", "interpretation_review_pending_idempotent"] = Field(alias="_kind")
    event_id: str
    affected_node_id: str
    kind: InterpretationKind
    interpretation_source: str
    message: str

    model_config = ConfigDict(extra="forbid", serialize_by_alias=True)


class _RequestInterpretationReviewPendingTextDataModel(BaseModel):
    """Legacy/drift response payload that accidentally carries review text.

    The live handler omits ``user_term`` and ``llm_draft`` entirely; this model
    exists as a persistence-boundary backstop if a future response shape adds
    them back. Because the raw fields are present, ``message`` is sensitive too:
    past versions interpolated the term into that string.
    """

    kind_marker: Literal["interpretation_review_pending", "interpretation_review_pending_idempotent"] = Field(alias="_kind")
    event_id: str
    affected_node_id: str
    kind: InterpretationKind
    user_term: Annotated[str, Sensitive(summarizer=_summarize_interpretation_term)]
    llm_draft: Annotated[str, Sensitive(summarizer=_summarize_interpretation_term)]
    interpretation_source: str
    message: Annotated[str, Sensitive(summarizer=_summarize_interpretation_term)]

    model_config = ConfigDict(extra="forbid", serialize_by_alias=True)


class _RequestInterpretationReviewSuppressedDataModel(BaseModel):
    """Correlation-only response payload for session opt-out suppression."""

    kind_marker: Literal["interpretation_review_suppressed_by_opt_out"] = Field(alias="_kind")
    event_id: str
    kind: InterpretationKind
    interpretation_source: str
    interpretation_review_disabled: bool
    message: str

    model_config = ConfigDict(extra="forbid", serialize_by_alias=True)


class _RequestInterpretationReviewResponseModel(BaseModel):
    """Redaction-bearing ``ToolResult`` envelope for interpretation reviews."""

    success: bool
    validation: GetBlobContentValidationModel
    affected_nodes: list[str]
    version: int
    data: (
        _RequestInterpretationReviewPendingDataModel
        | _RequestInterpretationReviewPendingTextDataModel
        | _RequestInterpretationReviewSuppressedDataModel
    )

    model_config = ConfigDict(extra="forbid")


class GetBlobContentArgumentsModel(BaseModel):
    """Redaction-bearing argument model for the ``get_blob_content`` tool.

    Mirrors the JSON schema declared at ``tools.py:1444-1455`` and its
    required-paths (``blob_id``).  The argument surface carries no sensitive
    material — ``blob_id`` is a UUID identity — so no fields are Sensitive.

    ``get_blob_content`` is type-driven (not declarative) because the
    response surface IS sensitive: ``content`` carries the operator's
    uploaded blob payload and must be summarized to a length-only scalar
    before reaching ``chat_messages.tool_calls``.  Declarative entries have
    no ``response_summarizers`` field; the only way
    to attach a per-key summarizer is via a ``Annotated[T, Sensitive(...)]``
    on a response model walked by ``walk_model_schema``.

    ``extra="forbid"`` is required (rev-2 M.1).  Fields belonging to
    neighbouring tools (``filename``, ``mime_type``, ``content`` —
    those are on ``create_blob`` / ``update_blob``) are intentionally
    absent so ``extra="forbid"`` rejects misrouted argument shapes early.
    """

    blob_id: str

    model_config = ConfigDict(extra="forbid")


class GetBlobContentDataModel(BaseModel):
    """Redaction-bearing success payload for ``get_blob_content``."""

    blob_id: str
    filename: str
    mime_type: str
    content: Annotated[str, Sensitive(summarizer=_summarize_blob_content)]
    truncated: bool
    size_bytes: int

    model_config = ConfigDict(extra="forbid")


class GetBlobContentFailureDataModel(BaseModel):
    """Recoverable ``get_blob_content`` failure payload."""

    error: str

    model_config = ConfigDict(extra="forbid")


class GetBlobContentResponseModel(BaseModel):
    """Redaction-bearing ``ToolResult`` envelope for ``get_blob_content``.

    The handler returns a normal composer ``ToolResult`` envelope. Its success
    branch stores sensitive blob bytes under ``data.content``; failure branches
    store only ``data.error``. Keep both shapes closed so audit persistence
    redacts content without crashing on recoverable failures.
    """

    success: bool
    validation: GetBlobContentValidationModel
    affected_nodes: list[str]
    version: int
    data: GetBlobContentDataModel | GetBlobContentFailureDataModel

    model_config = ConfigDict(extra="forbid")


_INSPECT_SOURCE_REASON = HandlesNoSensitiveDataReason(
    sensitive_data_locations=("source inspection facts — headers / types / URL candidates, never raw row content",),
    why_arguments_safe=(
        "inspect_source accepts a single blob_id scalar string selecting one inventory "
        "row; the handler reads at most 8 KiB and parses at most 100 rows, returning only "
        "structural facts; the schema and redaction allowlist reject or strip any other keys."
    ),
    why_responses_safe=(
        "Response is the SourceInspectionFacts struct — observed headers, inferred scalar "
        "types per column, URL candidate count, and warnings; the inspector explicitly never "
        "returns raw row content (tools.py:3724 docstring) so payload bytes never appear."
    ),
)


# ---------------------------------------------------------------------------
# Declarative manifest entries — _BLOB_MUTATION_TOOLS
# remaining, 2 tools).
#
# Excluding create_blob / update_blob / set_source_from_blob / apply_pipeline_recipe,
# ``delete_blob`` and ``wire_blob_inline_ref`` remain in
# ``_BLOB_MUTATION_TOOLS``. They take scalar identifiers / field paths and
# return structural ToolResult payloads.
# ---------------------------------------------------------------------------


_DELETE_BLOB_REASON = HandlesNoSensitiveDataReason(
    sensitive_data_locations=("session blob inventory — single blob_id selector identifying the deletion target",),
    why_arguments_safe=(
        "delete_blob accepts a single blob_id scalar string naming the inventory row to "
        "delete; the JSON schema at tools.py:1432-1441 has no other properties, so the "
        "LLM cannot place any blob content or operator payload on this surface."
    ),
    why_responses_safe=(
        "Response is the structural ToolResult after deletion — validation summary plus "
        "the affected node id list reflecting any pipeline references to the now-removed "
        "blob; the deleted blob's content and metadata are not echoed back in the response."
    ),
)


_WIRE_BLOB_INLINE_REF_REASON = HandlesNoSensitiveDataReason(
    sensitive_data_locations=(
        "session blob inventory — blob_id selector and authoritative content_hash only",
        "pipeline plugin option dict — the widened blob_ref marker is wired, not the referenced content",
    ),
    why_arguments_safe=(
        "wire_blob_inline_ref arguments are scalar metadata: field_path, blob_id, "
        "optional encoding, and an optional sha256_override used only as a guardrail. "
        "The tool never accepts content bytes; it reads the authoritative hash from "
        "the blobs table and rejects disagreement."
    ),
    why_responses_safe=(
        "Response is the structural ToolResult after the state mutation. Subsequent "
        "state inspection can show the marker {blob_ref, mode, sha256, encoding}, but "
        "the referenced blob content remains opaque and is resolved only at runtime."
    ),
)


# ---------------------------------------------------------------------------
# Declarative manifest entries — _SECRET_DISCOVERY_TOOLS,
# 2 tools).
#
# Secret values are resolved server-side at execution time by the secret
# service; they never traverse the composer tool surface.  ``list_secret_refs``
# and ``validate_secret_ref`` operate on secret NAMES only — discovering which
# references the operator has wired and whether the current user can resolve
# them at execution.  The named-vs-valued distinction is the entire reason
# the secret-ref mechanism exists; surfacing it explicitly in the audit-trail
# justification is policy-correct.
# ---------------------------------------------------------------------------


_LIST_SECRET_REFS_REASON = HandlesNoSensitiveDataReason(
    sensitive_data_locations=(
        "server-side secret resolver — secret values resolved at execution time, never echoed",
        "secret-ref registry — references carry NAMES and SCOPES only, not values",
    ),
    why_arguments_safe=(
        "list_secret_refs accepts no arguments — the JSON schema at tools.py:1519 declares "
        "an empty properties object with empty required, so the LLM cannot place any "
        "value on this surface; the handler enumerates the user's accessible references."
    ),
    why_responses_safe=(
        "Response is the user-scoped secret reference inventory — names and scopes per "
        "ref — but never values; values live behind the secret-service resolver and are "
        "materialised only at pipeline execution, never in any composer tool response."
    ),
)


_VALIDATE_SECRET_REF_REASON = HandlesNoSensitiveDataReason(
    sensitive_data_locations=(
        "server-side secret resolver — secret values resolved at execution time, never echoed",
        "secret-ref existence check — yes/no flag plus scope, never the value",
    ),
    why_arguments_safe=(
        "validate_secret_ref accepts a single scalar name string identifying the secret "
        "reference to check; names are operator-chosen labels (e.g., 'OPENROUTER_API_KEY') "
        "registered via the secret service and never the underlying credential value."
    ),
    why_responses_safe=(
        "Response is the existence/accessibility flag plus the matching scope — boolean "
        "and scope label only; the secret VALUE is resolved only at pipeline execution "
        "time by the server-side resolver and never enters the composer tool surface."
    ),
)


# ---------------------------------------------------------------------------
# Declarative manifest entries — _SECRET_MUTATION_TOOLS,
# 1 tool).
# ---------------------------------------------------------------------------


_WIRE_SECRET_REF_REASON = HandlesNoSensitiveDataReason(
    sensitive_data_locations=(
        "server-side secret resolver — secret values resolved at execution time, never echoed",
        "pipeline plugin option dict — the secret REFERENCE marker {secret_ref: NAME} is wired, not the value",
    ),
    why_arguments_safe=(
        "wire_secret_ref arguments are four scalar strings (name, target, target_id, "
        "option_key); the handler places a {secret_ref: NAME} marker dict into the chosen "
        "option slot, so only the NAME of the secret is recorded — never any credential."
    ),
    why_responses_safe=(
        "Response is the structural ToolResult after the wire mutation — validation summary "
        "for the post-mutation state plus the affected node id list; the wired marker "
        "appears in subsequent state inspections but the secret VALUE is never returned."
    ),
)


# ---------------------------------------------------------------------------
# Declarative manifest entry — request_advisor_hint, 1 tool.
#
# ``request_advisor_hint`` is the advisor escape hatch intercepted at
# service.py:2103 BEFORE the dispatcher; the result_payload shapes are
# enumerated at service.py:2114-2358 (error, budget-exhausted, timeout,
# advisor-error, success).  ``problem_summary``, ``recent_errors``,
# ``attempted_actions``, and ``schema_excerpt`` are LLM-supplied free text
# that may reproduce prompt-injection vectors, plugin option values, or
# validator-quoted operator content; they ARE summarised at the argument
# boundary so the audit-side record collapses each to a fixed-form scalar.
#
# ``guidance`` in the response is the frontier-model advice text returned
# by ``_call_advisor_with_audit``; the full text is already preserved in the
# audit trail by the ComposerLLMCall recorder, which fires in the same
# request.  For the chat_messages.tool_calls.result_payload row the guidance
# text is classified as ``sensitive_response_keys`` so it is substituted with
# REDACTED_SENSITIVE_NO_SUMMARIZER at the persistence boundary, keeping the
# closed-set status / budget / latency metadata in clear text while preventing
# double-mirroring of the advice text into the tool-call surface.
# ---------------------------------------------------------------------------


def _summarize_advisor_problem_summary(value: str) -> str:
    """Summarizer for ``request_advisor_hint.problem_summary``.

    Records only the character length of the LLM's problem statement; the
    composer's own slog of advisor calls retains the full text under separate
    retention policy.  Contract (spec §4.2.6, §9 RSK-03):

      * MUST NOT raise on any reachable input value.
      * MUST return ``str``.
    """
    return f"<advisor-problem-summary:{len(value)}-chars>"


def _summarize_advisor_recent_errors(value: list[str]) -> str:
    """Summarizer for ``request_advisor_hint.recent_errors``.

    Records only the number of error strings.  The full error texts are
    preserved in the separate validator-output audit records; recording them
    here would double-mirror them into ``chat_messages.tool_calls`` beyond
    their intended retention surface.
    """
    return f"<advisor-recent-errors:{len(value)}-entries>"


def _summarize_advisor_attempted_actions(value: list[str]) -> str:
    """Summarizer for ``request_advisor_hint.attempted_actions``."""
    return f"<advisor-attempted-actions:{len(value)}-entries>"


def _summarize_advisor_schema_excerpt(value: str) -> str:
    """Summarizer for ``request_advisor_hint.schema_excerpt``."""
    return f"<advisor-schema-excerpt:{len(value)}-chars>"


_TOOL_RESULT_REQUIRED_RESPONSE_KEYS: tuple[str, ...] = (
    "success",
    "validation",
    "affected_nodes",
    "version",
)
_TOOL_RESULT_OPTIONAL_RESPONSE_KEYS: tuple[str, ...] = (
    "runtime_preflight",
    "validation_delta",
    "post_call_hints",
    "plugin_schemas",
)


def _tool_result_response_keys(*, data: bool) -> tuple[str, ...]:
    """Return the shared top-level ``ToolResult.to_dict`` response envelope."""
    keys = _TOOL_RESULT_REQUIRED_RESPONSE_KEYS
    if data:
        keys = (*keys, "data")
    return (*keys, *_TOOL_RESULT_OPTIONAL_RESPONSE_KEYS)


# Manifest entries are grouped by tool family. The binding is rebuilt as a
# new ``MappingProxyType`` per the spec §4.2.1
# rule "subsequent task waves extend the manifest by building a new
# dict, then replacing the module-level binding — never by mutating
# the proxy view".
MANIFEST: Mapping[str, ToolRedaction] = MappingProxyType(
    {
        # set_source.
        "set_source": ToolRedaction(argument_model=SetSourceArgumentsModel),
        # blob-write tools.
        "create_blob": ToolRedaction(argument_model=CreateBlobArgumentsModel),
        "update_blob": ToolRedaction(argument_model=UpdateBlobArgumentsModel),
        "set_source_from_blob": ToolRedaction(argument_model=SetSourceFromBlobArgumentsModel),
        # full-pipeline mutations.
        "set_pipeline": ToolRedaction(argument_model=SetPipelineArgumentsModel),
        "apply_pipeline_recipe": ToolRedaction(argument_model=ApplyPipelineRecipeArgumentsModel),
        # option-patch tools.
        "patch_source_options": ToolRedaction(argument_model=PatchSourceOptionsArgumentsModel),
        "patch_node_options": ToolRedaction(argument_model=PatchNodeOptionsArgumentsModel),
        "patch_output_options": ToolRedaction(argument_model=PatchOutputOptionsArgumentsModel),
        # _DISCOVERY_TOOLS, 12 declarative entries.
        "list_sources": ToolRedaction(
            policy=ToolRedactionPolicy(
                handles_no_sensitive_data=True,
                handles_no_sensitive_data_reason_struct=_LIST_SOURCES_REASON,
            )
        ),
        "list_transforms": ToolRedaction(
            policy=ToolRedactionPolicy(
                handles_no_sensitive_data=True,
                handles_no_sensitive_data_reason_struct=_LIST_TRANSFORMS_REASON,
            )
        ),
        "list_sinks": ToolRedaction(
            policy=ToolRedactionPolicy(
                handles_no_sensitive_data=True,
                handles_no_sensitive_data_reason_struct=_LIST_SINKS_REASON,
            )
        ),
        "get_plugin_schema": ToolRedaction(
            policy=ToolRedactionPolicy(
                handles_no_sensitive_data=True,
                handles_no_sensitive_data_reason_struct=_GET_PLUGIN_SCHEMA_REASON,
            )
        ),
        "get_expression_grammar": ToolRedaction(
            policy=ToolRedactionPolicy(
                handles_no_sensitive_data=True,
                handles_no_sensitive_data_reason_struct=_GET_EXPRESSION_GRAMMAR_REASON,
            )
        ),
        "explain_validation_error": ToolRedaction(
            policy=ToolRedactionPolicy(
                handles_no_sensitive_data=True,
                handles_no_sensitive_data_reason_struct=_EXPLAIN_VALIDATION_ERROR_REASON,
            )
        ),
        "get_plugin_assistance": ToolRedaction(
            policy=ToolRedactionPolicy(
                handles_no_sensitive_data=True,
                handles_no_sensitive_data_reason_struct=_GET_PLUGIN_ASSISTANCE_REASON,
            )
        ),
        "list_models": ToolRedaction(
            policy=ToolRedactionPolicy(
                handles_no_sensitive_data=True,
                handles_no_sensitive_data_reason_struct=_LIST_MODELS_REASON,
            )
        ),
        "list_recipes": ToolRedaction(
            policy=ToolRedactionPolicy(
                handles_no_sensitive_data=True,
                handles_no_sensitive_data_reason_struct=_LIST_RECIPES_REASON,
            )
        ),
        "get_audit_info": ToolRedaction(
            policy=ToolRedactionPolicy(
                handles_no_sensitive_data=True,
                handles_no_sensitive_data_reason_struct=_GET_AUDIT_INFO_REASON,
            )
        ),
        "get_pipeline_state": ToolRedaction(
            policy=ToolRedactionPolicy(
                handles_no_sensitive_data=True,
                handles_no_sensitive_data_reason_struct=_GET_PIPELINE_STATE_REASON,
            )
        ),
        "preview_pipeline": ToolRedaction(
            policy=ToolRedactionPolicy(
                handles_no_sensitive_data=True,
                handles_no_sensitive_data_reason_struct=_PREVIEW_PIPELINE_REASON,
            )
        ),
        "diff_pipeline": ToolRedaction(
            policy=ToolRedactionPolicy(
                handles_no_sensitive_data=True,
                handles_no_sensitive_data_reason_struct=_DIFF_PIPELINE_REASON,
            )
        ),
        # _MUTATION_TOOLS remaining, 8 declarative entries.
        "upsert_node": ToolRedaction(
            policy=ToolRedactionPolicy(
                known_argument_keys=(
                    "id",
                    "node_type",
                    "input",
                    "plugin",
                    "on_success",
                    "on_error",
                    "options",
                    "condition",
                    "routes",
                    "fork_to",
                    "branches",
                    "policy",
                    "merge",
                    "trigger",
                    "output_mode",
                    "expected_output_count",
                ),
                sensitive_argument_keys=("options", "routes", "trigger"),
                argument_summarizers={
                    "options": _summarize_set_source_options,
                    "routes": _summarize_set_source_options,
                    "trigger": _summarize_set_source_options,
                },
                redact_unknown_argument_keys=True,
                handles_no_sensitive_data=False,
                # Shared ToolResult.to_dict() envelope plus the data key emitted
                # by failure branches reachable from _execute_upsert_node:
                # _mutation_result → always emits success/validation/affected_nodes/version;
                # _failure_result → adds data={"error": ...};
                # _credential_wiring_contract_failure → adds data={error, credential_fields,
                #   components, repair}.
                known_response_keys=_tool_result_response_keys(data=True),
            )
        ),
        "upsert_edge": ToolRedaction(
            policy=ToolRedactionPolicy(
                known_argument_keys=(
                    "id",
                    "from_node",
                    "to_node",
                    "edge_type",
                    "label",
                ),
                redact_unknown_argument_keys=True,
                handles_no_sensitive_data=True,
                handles_no_sensitive_data_reason_struct=_UPSERT_EDGE_REASON,
            )
        ),
        "remove_node": ToolRedaction(
            policy=ToolRedactionPolicy(
                known_argument_keys=("id",),
                redact_unknown_argument_keys=True,
                handles_no_sensitive_data=True,
                handles_no_sensitive_data_reason_struct=_REMOVE_NODE_REASON,
            )
        ),
        "remove_edge": ToolRedaction(
            policy=ToolRedactionPolicy(
                known_argument_keys=("id",),
                redact_unknown_argument_keys=True,
                handles_no_sensitive_data=True,
                handles_no_sensitive_data_reason_struct=_REMOVE_EDGE_REASON,
            )
        ),
        "remove_output": ToolRedaction(
            policy=ToolRedactionPolicy(
                known_argument_keys=("sink_name",),
                redact_unknown_argument_keys=True,
                handles_no_sensitive_data=True,
                handles_no_sensitive_data_reason_struct=_REMOVE_OUTPUT_REASON,
            )
        ),
        "clear_source": ToolRedaction(
            policy=ToolRedactionPolicy(
                known_argument_keys=("source_name",),
                redact_unknown_argument_keys=True,
                handles_no_sensitive_data=True,
                handles_no_sensitive_data_reason_struct=_CLEAR_SOURCE_REASON,
            )
        ),
        "set_metadata": ToolRedaction(
            policy=ToolRedactionPolicy(
                known_argument_keys=("patch",),
                sensitive_argument_keys=("patch",),
                argument_summarizers={"patch": _summarize_set_metadata_patch},
                redact_unknown_argument_keys=True,
                handles_no_sensitive_data=False,
                # Shared ToolResult.to_dict() envelope. _execute_set_metadata
                # currently calls _mutation_result without data, so data remains
                # excluded while optional ToolResult augmentation keys stay
                # centrally covered.
                known_response_keys=_tool_result_response_keys(data=False),
            )
        ),
        "set_output": ToolRedaction(
            policy=ToolRedactionPolicy(
                known_argument_keys=(
                    "sink_name",
                    "plugin",
                    "options",
                    "on_write_failure",
                ),
                sensitive_argument_keys=("options",),
                argument_summarizers={"options": _summarize_set_source_options},
                redact_unknown_argument_keys=True,
                handles_no_sensitive_data=False,
                # Shared ToolResult.to_dict() envelope plus the data key emitted
                # by failure branches reachable from _execute_set_output:
                # _mutation_result → success/validation/affected_nodes/version;
                # _failure_result → adds data={"error": ...};
                # _credential_wiring_contract_failure → adds data={error, credential_fields,
                #   components, repair}.
                known_response_keys=_tool_result_response_keys(data=True),
            )
        ),
        # _BLOB_DISCOVERY_TOOLS, 5 entries (4 declarative + 1 type-driven).
        "list_blobs": ToolRedaction(
            policy=ToolRedactionPolicy(
                redact_unknown_argument_keys=True,
                handles_no_sensitive_data=True,
                handles_no_sensitive_data_reason_struct=_LIST_BLOBS_REASON,
            )
        ),
        "list_composer_blobs": ToolRedaction(
            policy=ToolRedactionPolicy(
                redact_unknown_argument_keys=True,
                handles_no_sensitive_data=True,
                handles_no_sensitive_data_reason_struct=_LIST_COMPOSER_BLOBS_REASON,
            )
        ),
        "get_blob_metadata": ToolRedaction(
            policy=ToolRedactionPolicy(
                known_argument_keys=("blob_id",),
                redact_unknown_argument_keys=True,
                handles_no_sensitive_data=True,
                handles_no_sensitive_data_reason_struct=_GET_BLOB_METADATA_REASON,
            )
        ),
        "get_blob_content": ToolRedaction(
            argument_model=GetBlobContentArgumentsModel,
            response_model=GetBlobContentResponseModel,
        ),
        "inspect_source": ToolRedaction(
            policy=ToolRedactionPolicy(
                known_argument_keys=("blob_id",),
                redact_unknown_argument_keys=True,
                handles_no_sensitive_data=True,
                handles_no_sensitive_data_reason_struct=_INSPECT_SOURCE_REASON,
            )
        ),
        # _BLOB_MUTATION_TOOLS remaining, 2 entries.
        "delete_blob": ToolRedaction(
            policy=ToolRedactionPolicy(
                known_argument_keys=("blob_id",),
                redact_unknown_argument_keys=True,
                handles_no_sensitive_data=True,
                handles_no_sensitive_data_reason_struct=_DELETE_BLOB_REASON,
            )
        ),
        "wire_blob_inline_ref": ToolRedaction(
            policy=ToolRedactionPolicy(
                known_argument_keys=("field_path", "blob_id", "encoding"),
                redact_unknown_argument_keys=True,
                handles_no_sensitive_data=True,
                handles_no_sensitive_data_reason_struct=_WIRE_BLOB_INLINE_REF_REASON,
            )
        ),
        # _SECRET_DISCOVERY_TOOLS, 2 entries.
        "list_secret_refs": ToolRedaction(
            policy=ToolRedactionPolicy(
                handles_no_sensitive_data=True,
                handles_no_sensitive_data_reason_struct=_LIST_SECRET_REFS_REASON,
                known_argument_keys=(),
                redact_unknown_argument_keys=True,
            )
        ),
        "validate_secret_ref": ToolRedaction(
            policy=ToolRedactionPolicy(
                handles_no_sensitive_data=True,
                handles_no_sensitive_data_reason_struct=_VALIDATE_SECRET_REF_REASON,
                known_argument_keys=("name",),
                redact_unknown_argument_keys=True,
            )
        ),
        # _SECRET_MUTATION_TOOLS, 1 entry.
        "wire_secret_ref": ToolRedaction(
            policy=ToolRedactionPolicy(
                handles_no_sensitive_data=True,
                handles_no_sensitive_data_reason_struct=_WIRE_SECRET_REF_REASON,
                known_argument_keys=("name", "target", "target_id", "option_key"),
                redact_unknown_argument_keys=True,
            )
        ),
        # request_interpretation_review (session-aware async tool).
        # Type-driven entry; both argument and response-side review text fields
        # carry a fixed-form summariser so chat_messages.tool_calls cannot leak
        # unbounded user/LLM text. The live handler returns correlation-only
        # response data, while the response_model keeps a redaction backstop for
        # future drift that accidentally reintroduces user_term / llm_draft.
        "request_interpretation_review": ToolRedaction(
            argument_model=_RequestInterpretationReviewRedactionModel,
            response_model=_RequestInterpretationReviewResponseModel,
        ),
        # request_advisor_hint (intercepted at service.py:2103).
        "request_advisor_hint": ToolRedaction(
            policy=ToolRedactionPolicy(
                known_argument_keys=(
                    "trigger",
                    "problem_summary",
                    "recent_errors",
                    "attempted_actions",
                    "schema_excerpt",
                ),
                sensitive_argument_keys=(
                    "problem_summary",
                    "recent_errors",
                    "attempted_actions",
                    "schema_excerpt",
                ),
                argument_summarizers={
                    "problem_summary": _summarize_advisor_problem_summary,
                    "recent_errors": _summarize_advisor_recent_errors,
                    "attempted_actions": _summarize_advisor_attempted_actions,
                    "schema_excerpt": _summarize_advisor_schema_excerpt,
                },
                handles_no_sensitive_data=False,
                # sensitive_response_keys: guidance is the frontier-model advice text,
                # already preserved by ComposerLLMCall recorder; classifying it sensitive
                # prevents double-mirroring of the advice text into chat_messages.tool_calls
                # while keeping the closed-set status/budget/latency metadata in clear text.
                # Per ToolRedactionPolicy invariant 5 (adequacy guard assertion 5), guidance
                # must also appear in known_response_keys (sensitive ⊆ known).
                sensitive_response_keys=("guidance",),
                # known_response_keys: enumerated from all reachable response shapes at
                # service.py:2114-2358.  Disabled path → {error}.  Budget-exhausted path →
                # {status, budget_used, budget_remaining, guidance}.  ARG_ERROR path →
                # {status, error, error_class}.  Timeout/advisor-error paths →
                # {status, error, error_class, budget_used, budget_remaining}.  Success path
                # → {status, guidance, model, prompt_tokens, completion_tokens,
                # cached_prompt_tokens, advisor_latency_ms, budget_used, budget_remaining,
                # note}.  Union of all shapes, guidance included per the sensitive⊆known rule.
                known_response_keys=(
                    "status",
                    "error",
                    "error_class",
                    "budget_used",
                    "budget_remaining",
                    "guidance",
                    "model",
                    "prompt_tokens",
                    "completion_tokens",
                    "cached_prompt_tokens",
                    "advisor_latency_ms",
                    "note",
                ),
            )
        ),
    }
)


def _is_declarative_response_repair_arguments_path(path: tuple[str, ...]) -> bool:
    if not path or path[-1] != "arguments":
        return False
    if path[0] == "validation" and "graph_repair_suggestions" in path and "tool_sequence" in path:
        return True
    return len(path) >= 3 and path[0] == "data" and path[1] == "repair"


def _redact_declarative_known_response_value(value: Any, *, path: tuple[str, ...]) -> Any:
    """Scrub nested repair-argument payloads inside declarative ToolResult keys.

    Declarative response policies close only the top-level ToolResult envelope
    (``success`` / ``validation`` / ``data`` / etc.). Some known envelopes carry
    nested, open repair-tool-call argument mappings. Those are structurally
    useful audit metadata, but the values can contain credential/config payload,
    so they reuse the same structural ``<repair-args:...>`` summarizer as the
    type-driven validation shadow model.
    """
    if _is_declarative_response_repair_arguments_path(path):
        if isinstance(value, Mapping):
            argument_keys: dict[str, object] = {str(key): child for key, child in value.items()}
            return _summarize_repair_arguments(argument_keys)
        return REDACTED_SENSITIVE_NO_SUMMARIZER
    if isinstance(value, Mapping):
        return {key: _redact_declarative_known_response_value(child, path=(*path, str(key))) for key, child in value.items()}
    if isinstance(value, list):
        return [_redact_declarative_known_response_value(item, path=(*path, "*")) for item in value]
    if isinstance(value, tuple):
        return tuple(_redact_declarative_known_response_value(item, path=(*path, "*")) for item in value)
    return value


def redact_tool_call_response(
    tool_name: str,
    response: dict[str, Any],
    *,
    telemetry: RedactionTelemetry,
) -> dict[str, Any]:
    """Produce the redacted response dict for the persistence boundary (spec §4.2.4, §4.2.6).

    Disposition by manifest-entry shape:

    **Type-driven entry with response_model:**
      Walk via ``walk_model_schema``; ``Sensitive[T]`` fields are substituted
      with the summarizer output, or ``REDACTED_SENSITIVE_NO_SUMMARIZER`` when
      no summarizer is declared. Nested-path substitution delegates to the
      fail-closed walker path, which raises ``NotImplementedError`` for paths
      containing ``.``, ``[``, or ``{``.

    **Type-driven entry without response_model:**
      No response-surface declared; return a shallow copy (passthrough).

    **Declarative entry:**
      Walk ``response.keys()``; each key is one of:
        - In ``policy.sensitive_response_keys`` → ``REDACTED_SENSITIVE_NO_SUMMARIZER``
          (declarative entries have no response_summarizers; the argument_summarizers
          mapping covers only argument keys).
        - In ``policy.known_response_keys`` but not sensitive → passthrough after
          bounded repair-guidance scrubbing.
        - In neither → ``REDACTED_UNKNOWN_RESPONSE_KEY`` (fail-closed; counter fires).

    **Failure modes (all raise AuditIntegrityError):**
      - Manifest entry missing for ``tool_name`` (registry-consistency invariant).
      - Summarizer raises → ``telemetry.summarizer_error(tool_name=...)`` BEFORE
        raise; ``AuditIntegrityError`` chained from the underlying exception (M.8).
      - Summarizer returns non-str → ``telemetry.summarizer_error(tool_name=...)``
        BEFORE raise; ``AuditIntegrityError`` with a typed message.

    **Walker atomicity:** the output dict is built in a local variable and
    only returned on success.  A mid-walk raise leaves no partial dict
    observable to the caller (rev-3 W8b / rev-4 W8b).
    """
    if tool_name not in MANIFEST:
        raise AuditIntegrityError(
            f"redact_tool_call_response called for unknown tool {tool_name!r}; "
            "the manifest is the source of truth for the registry/redaction "
            "set-equality invariant.  Add a manifest entry for this tool or "
            "verify that the dispatch path passes the correct tool name."
        )
    entry = MANIFEST[tool_name]

    # --- Type-driven path ---
    if entry.argument_model is not None:
        # Spec §4.2.4: manifest_dispatch is a per-invocation, walker-wide beacon.
        # Emit regardless of whether a response_model is declared — the dispatch
        # happened and the shape is type_driven in both sub-cases.
        telemetry.manifest_dispatch(tool_name=tool_name, shape="type_driven")
        if entry.response_model is None:
            # No response surface declared: nothing to redact.
            return dict(response)
        # Walk the response via its Pydantic model schema.
        # model_validate coerces the raw response dict to the declared shape;
        # unknown keys raise ValidationError if extra="forbid" is set on the
        # model, but that is a model-design choice, not enforced here.
        validated = entry.response_model.model_validate(response)
        return _redact_via_schema(tool_name, validated, entry.response_model, telemetry=telemetry)

    # --- Declarative path ---
    # entry.policy is not None (ToolRedaction.__post_init__ guarantees exactly
    # one of {argument_model, policy} is set).
    # Spec §4.2.4: emit the manifest_dispatch beacon for the declarative branch too.
    telemetry.manifest_dispatch(tool_name=tool_name, shape="declarative")
    policy = entry.policy
    assert policy is not None  # offensive: satisfies the type-checker contract

    redacted: dict[str, Any] = {}
    for key, value in response.items():
        if key in policy.sensitive_response_keys:
            # Sensitive key: declarative entries have no response summarizers
            # (argument_summarizers covers only argument keys).  Substitute
            # the no-summarizer sentinel.
            redacted[key] = REDACTED_SENSITIVE_NO_SUMMARIZER
        elif key in policy.known_response_keys:
            # Known, non-sensitive: preserve the declared ToolResult envelope
            # while scrubbing nested repair-tool-call arguments.
            redacted[key] = _redact_declarative_known_response_value(value, path=(key,))
        else:
            # Unknown key: fail-closed sentinel + telemetry counter (W6).
            redacted[key] = REDACTED_UNKNOWN_RESPONSE_KEY
            telemetry.unknown_response_key_redacted(tool_name=tool_name)

    return redacted


def redact_source_storage_path(state_dict: dict[str, Any]) -> dict[str, Any]:
    """Redact internal storage paths from a serialized state dict.

    When source options contain a ``blob_ref``, ``path`` (and the equivalent
    ``file`` carrier) point at an internal storage location that must not be
    exposed to agents or users. Preserve the key(s) with a sentinel value so
    external consumers can still tell that the source path contract is satisfied.

    Returns a shallow copy with source options redacted. Does not mutate
    the input dict.

    ``state_dict`` is one of our own serializer outputs, but its shape is
    POLYMORPHIC across callers: full-state / ``{"source": ...}`` views always
    carry a ``"source"`` key, whereas the component-scoped
    ``_execute_get_pipeline_state`` view emits ``{"node": ...}`` or
    ``{"output": ...}`` with NO ``"source"`` key at all. The absent key is a
    legitimate first-party shape ("this view has no source to redact"), not a
    contract violation — so we gate on explicit membership rather than crash.
    Likewise a source dict carries ``"options"`` on the to_dict path but the
    component-scoped ``_serialize_source`` may omit it; gate on membership.
    The composer/LLM-authored VALUES inside ``options`` are Tier 3, but the
    only redaction trigger is the structural ``blob_ref`` marker; a ``None``
    source or blob_ref-less options is the "nothing to redact" outcome,
    recorded explicitly by returning the input unchanged.
    """

    def _redact_one(source: Any) -> tuple[Any, bool]:
        # An absent/None source is the documented "nothing to redact"
        # first-party shape (see this function's docstring). A PRESENT,
        # non-None source that is not a Mapping is NOT a producible
        # first-party shape: every serializer feeding this surface
        # (CompositionState.to_dict, _serialize_source,
        # _serialize_full_pipeline_state) emits source/named-source values
        # as dicts. A non-Mapping here is a corrupted/regressed Tier-1
        # serializer output; returning it unchanged would silently pass an
        # un-redacted storage path through this leak-prevention surface, so
        # fail closed with provenance instead (matches the offensive posture
        # of _state_response's direct index and the MCP read-back's
        # PLUGIN_CRASH reclassification).
        if source is None:
            return source, False
        if not isinstance(source, Mapping):
            raise AuditIntegrityError(
                "redact_source_storage_path received a non-Mapping source value "
                f"(type {type(source).__name__!r}); serialized state source entries "
                "must be mappings. Refusing to pass through a malformed first-party "
                "shape that may carry an un-redacted internal storage path."
            )
        if "options" not in source:
            return source, False
        options = source["options"]
        if options is None or not isinstance(options, Mapping) or "blob_ref" not in options:
            return source, False
        redacted_source = dict(source)
        redacted_options = dict(options)
        # Both "path" and "file" are blob storage-path carriers: blob ownership
        # detection (web/blobs/service.py) and the fork rewrite
        # (web/sessions/routes/sessions.py) treat them equivalently. Mask both so a
        # blob-backed source authored with the "file" option shape cannot leak the
        # internal storage_path through this redaction surface (elspeth-a7aa07b7ce).
        for storage_path_key in ("path", "file"):
            if storage_path_key in redacted_options:
                redacted_options[storage_path_key] = REDACTED_BLOB_SOURCE_PATH
        redacted_source["options"] = redacted_options
        return redacted_source, True

    source = state_dict["source"] if "source" in state_dict else None
    redacted_source, source_changed = _redact_one(source)
    sources = state_dict["sources"] if "sources" in state_dict else None
    redacted_sources: dict[str, Any] | None = None
    sources_changed = False
    if sources is not None:
        redacted_sources = {}
        for source_name, named_source in sources.items():
            redacted_named_source, changed = _redact_one(named_source)
            redacted_sources[source_name] = redacted_named_source
            sources_changed = sources_changed or changed

    if not source_changed and not sources_changed:
        return state_dict

    # Shallow copy the chain to avoid mutating the original
    redacted = dict(state_dict)
    if source_changed:
        redacted["source"] = redacted_source
    if sources_changed and redacted_sources is not None:
        redacted["sources"] = redacted_sources
    return redacted


def redact_guided_snapshot_storage_paths(
    sources: Mapping[str, Any] | None,
    composer_meta: Mapping[str, Any] | None,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    """Redact schema-8 reviewed source paths using each source's blob binding.

    Guided commits can omit ``blob_ref`` from the executable source while retaining
    it in ``guided_session.reviewed_sources``. Each reviewed snapshot is matched to
    the committed source by its persisted name and exact storage-path value. Both
    copies are redacted without mutating the persisted input dictionaries.
    """
    sources_out = dict(sources) if sources is not None else None
    meta_out = dict(composer_meta) if composer_meta is not None else None

    if composer_meta is None or "guided_session" not in composer_meta:
        return sources_out, meta_out
    guided = composer_meta["guided_session"]
    if type(guided) is not dict:
        raise ValueError("redact_guided_snapshot_storage_paths: composer_meta.guided_session must be a dict")
    reviewed_sources = guided["reviewed_sources"]
    if type(reviewed_sources) is not dict:
        raise ValueError("redact_guided_snapshot_storage_paths: guided_session.reviewed_sources must be a dict")
    pending_sources = guided["pending_source_intents"]
    if type(pending_sources) is not dict:
        raise ValueError("redact_guided_snapshot_storage_paths: guided_session.pending_source_intents must be a dict")

    reviewed_out: dict[str, Any] = {}
    pending_out: dict[str, Any] = {}
    rebuilt_sources = dict(sources) if sources is not None else None
    reviewed_bindings: list[tuple[str, frozenset[str]]] = []
    changed = False
    for stable_id, snapshot in reviewed_sources.items():
        if type(stable_id) is not str or type(snapshot) is not dict:
            raise ValueError("redact_guided_snapshot_storage_paths: reviewed_sources entries must be string-keyed dicts")
        name = snapshot["name"]
        if type(name) is not str or not name:
            raise ValueError("redact_guided_snapshot_storage_paths: reviewed_sources.name must be a non-empty str")
        snap_options = snapshot["options"]
        if type(snap_options) is not dict:
            raise ValueError(f"redact_guided_snapshot_storage_paths: guided_session.reviewed_sources[{stable_id!r}].options must be a dict")
        if "blob_ref" not in snap_options:
            reviewed_out[stable_id] = snapshot
            continue

        blob_ref = snap_options["blob_ref"]
        blob_paths = {value for key in ("path", "file") if key in snap_options and type(value := snap_options[key]) is str}
        if type(blob_ref) is str and blob_ref and not blob_paths:
            raise AuditIntegrityError("guided reviewed blob source is missing a string path carrier")
        snap_options_redacted = dict(snap_options)
        for key in ("path", "file"):
            if key in snap_options_redacted:
                snap_options_redacted[key] = REDACTED_BLOB_SOURCE_PATH
        snapshot_redacted = dict(snapshot)
        snapshot_redacted["options"] = snap_options_redacted
        reviewed_out[stable_id] = snapshot_redacted
        changed = True
        if blob_paths:
            reviewed_bindings.append((name, frozenset(blob_paths)))

    if rebuilt_sources is not None and reviewed_bindings:
        all_reviewed_paths = frozenset(path for _name, paths in reviewed_bindings for path in paths)
        for live_name, live_source in tuple(rebuilt_sources.items()):
            if type(live_source) is not dict:
                raise ValueError("redact_guided_snapshot_storage_paths: source entries must be dicts when guided blob redaction is active")
            if "options" not in live_source:
                continue
            live_options = live_source["options"]
            if type(live_options) is not dict:
                raise ValueError("redact_guided_snapshot_storage_paths: source.options must be a dict when guided blob redaction is active")
            live_reviewed_paths = {
                value for key in ("path", "file") if type(value := live_options.get(key)) is str and value in all_reviewed_paths
            }
            if not live_reviewed_paths:
                continue
            candidates = [
                paths for reviewed_name, paths in reviewed_bindings if reviewed_name == live_name and live_reviewed_paths <= paths
            ]
            if len(candidates) != 1:
                raise AuditIntegrityError("guided blob source mapping is inconsistent")
            options_redacted = dict(live_options)
            for key in ("path", "file"):
                if type(value := live_options.get(key)) is str and value in live_reviewed_paths:
                    options_redacted[key] = REDACTED_BLOB_SOURCE_PATH
            source_redacted = dict(live_source)
            source_redacted["options"] = options_redacted
            rebuilt_sources[live_name] = source_redacted

    for stable_id, intent in pending_sources.items():
        if type(stable_id) is not str or type(intent) is not dict:
            raise ValueError("redact_guided_snapshot_storage_paths: pending_source_intents entries must be string-keyed dicts")
        intent_options = intent["options"]
        if intent_options is None:
            pending_out[stable_id] = intent
            continue
        if type(intent_options) is not dict:
            raise ValueError(
                f"redact_guided_snapshot_storage_paths: guided_session.pending_source_intents[{stable_id!r}].options must be a dict or None"
            )
        if "blob_ref" not in intent_options:
            pending_out[stable_id] = intent
            continue
        options_redacted = dict(intent_options)
        for key in ("path", "file"):
            if key in options_redacted:
                options_redacted[key] = REDACTED_BLOB_SOURCE_PATH
        intent_redacted = dict(intent)
        intent_redacted["options"] = options_redacted
        pending_out[stable_id] = intent_redacted
        changed = True

    if changed:
        guided_redacted = dict(guided)
        guided_redacted["reviewed_sources"] = reviewed_out
        guided_redacted["pending_source_intents"] = pending_out
        meta_out = dict(composer_meta)
        meta_out["guided_session"] = guided_redacted
        sources_out = rebuilt_sources

    return sources_out, meta_out


class _AuthoringNodeOptionsModel(BaseModel):
    """Shared redaction-bearing surface for authored transform options."""

    id: str
    options: _LlmJsonObject = Field(default_factory=dict)
    on_error: str | None = None

    model_config = ConfigDict(extra="forbid")


class _SpliceTransformNodeModel(_AuthoringNodeOptionsModel):
    """Caller-authored portion of a transform whose routing is server-derived."""

    plugin: str
    options: _LlmJsonObject

    model_config = ConfigDict(extra="forbid")


class SpliceTransformArgumentsModel(BaseModel):
    """Redaction-bearing arguments for one direct-path transform insertion."""

    predecessor_id: str
    successor_id: str
    node: _SpliceTransformNodeModel

    model_config = ConfigDict(extra="forbid")


MANIFEST = MappingProxyType(
    {
        **MANIFEST,
        "splice_transform": ToolRedaction(argument_model=SpliceTransformArgumentsModel),
    }
)
