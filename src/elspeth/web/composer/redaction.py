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
from dataclasses import dataclass, field
from types import MappingProxyType, UnionType
from typing import Annotated, Any, Union, get_args, get_origin

from pydantic import BaseModel, ConfigDict

from elspeth.contracts.errors import AuditIntegrityError
from elspeth.contracts.freeze import freeze_fields
from elspeth.web.composer.redaction_telemetry import RedactionTelemetry

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

    Four invariants are enforced at construction time:

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

    NOTE on freeze: ``argument_summarizers`` values are Callables;
    ``deep_freeze`` passes Callables through unchanged (verified against
    ``src/elspeth/contracts/freeze.py:78``). Identity-equality of summarizer
    callables is the policy contract.
    """

    sensitive_argument_keys: tuple[str, ...] = ()
    sensitive_response_keys: tuple[str, ...] = ()
    known_response_keys: tuple[str, ...] = ()
    argument_summarizers: Mapping[str, Callable[[Any], str]] = field(default_factory=dict)
    handles_no_sensitive_data: bool = False
    handles_no_sensitive_data_reason_struct: HandlesNoSensitiveDataReason | None = None

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


def _summarize_set_source_options(options: dict[str, Any]) -> str:
    """Summarizer for ``set_source.options`` (spec §4.2.6).

    Wraps the existing :func:`redact_source_storage_path` helper so the
    redacted view of options is computed via the same path-blob-ref logic
    used by the legacy state-serialization redactor.  The output is the
    canonical JSON form of the redacted options dict so it is reusable as
    a hashable / loggable scalar in audit records.

    Contract (spec §4.2.6, §9 RSK-03):
      * MUST NOT raise on any reachable input value.
      * MUST return ``str``.

    ``default=str`` on :func:`json.dumps` ensures RSK-03 holds for values
    that survive Pydantic coercion but are not natively JSON-serialisable
    (``datetime``, ``bytes``, ``UUID``) — a future schema that admits
    those types via :class:`typing.Any` would otherwise raise
    :class:`TypeError` at the summarizer boundary.
    """
    redacted = redact_source_storage_path({"source": {"options": options}})
    return json.dumps(
        redacted["source"]["options"],
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )


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

    Field set is exactly the four required keys from the JSON schema at
    ``tools.py:631-655``.  Fields belonging to neighbouring tools
    (``label``, ``blob_id``, ``inline_blob`` — those are on
    ``set_source_from_blob`` / ``create_blob``) are intentionally absent
    so ``extra="forbid"`` rejects misrouted argument shapes early.
    """

    plugin: str
    on_success: str
    options: Annotated[dict[str, Any], Sensitive(summarizer=_summarize_set_source_options)]
    on_validation_failure: str

    model_config = ConfigDict(extra="forbid")


def _redact_via_schema(
    validated: BaseModel,
    model_cls: type[BaseModel],
) -> dict[str, Any]:
    """Walk the validated model's schema; substitute Sensitive fields.

    Tracer-bullet implementation (Task 4): supports only top-level
    sensitive fields (``set_source.options`` is the only such field in
    this commit).  Task 8 generalises this to nested paths under
    containers and BaseModel descent.

    Operates on ``model_dump()`` output (a plain dict) and substitutes
    in-place on a deep copy so the input model and any external caller
    references are not affected.  Returns the dict ready for serialization.
    """
    dumped = copy.deepcopy(validated.model_dump())
    for node in walk_model_schema(model_cls, with_values=True):
        marker = next((m for m in node.metadata if isinstance(m, _SensitiveMarker)), None)
        if marker is None:
            continue
        if marker.summarizer is None:
            # Task 8 decides the no-summarizer policy.  No reachable node
            # in this tracer commit lacks a summarizer; pin that here so
            # a future field-level omission surfaces immediately rather
            # than passing silently with the original value preserved.
            raise NotImplementedError(f"Sensitive field at path {node.path!r} has no summarizer; Task 8 defines the no-summarizer policy.")
        # Tracer-bullet scope: top-level paths only.  Nested-path
        # substitution requires the value_provider-driven path-walker
        # added in Task 8.
        if "." in node.path or "[" in node.path or "{" in node.path:
            raise NotImplementedError(f"Nested-path Sensitive field at path {node.path!r}; Task 8 implements the general path walker.")
        if node.value_provider is None:
            # walk_model_schema(with_values=True) guarantees value_provider
            # is not None; a None here is a walker bug, not external input.
            raise AuditIntegrityError(
                f"walk_model_schema yielded TraversalNode at {node.path!r} "
                "with value_provider=None despite with_values=True. "
                "This is an internal walker contract violation."
            )
        raw_value = node.value_provider(dumped)
        summary = marker.summarizer(raw_value)
        if not isinstance(summary, str):
            raise AuditIntegrityError(f"Summarizer for path {node.path!r} returned {type(summary).__name__}, expected str (spec §4.2.6).")
        dumped[node.path] = summary
    return dumped


def redact_tool_call_arguments(
    tool_name: str,
    arguments: dict[str, Any],
    *,
    telemetry: RedactionTelemetry,
) -> dict[str, Any]:
    """Minimal tracer-bullet impl (spec §4.2.6).

    Generalised in Task 8 to cover the declarative manifest branch and
    full path-walker substitution.  In this commit only the type-driven
    branch is wired and only the ``set_source`` manifest entry exists.

    The handler MUST catch :class:`pydantic.ValidationError` raised here
    and re-raise as :class:`ToolArgumentError` so the compose loop's
    ARG_ERROR routing at ``service.py:2480`` receives the right exception
    class.  A bare ``ValidationError`` escaping the handler hits the
    catch-all and becomes :class:`ComposerPluginCrashError` → HTTP 500 —
    the wrong disposition for Tier-3 input.
    """
    entry = MANIFEST[tool_name]
    if entry.argument_model is not None:
        telemetry.manifest_dispatch(tool_name=tool_name, shape="type_driven")
        validated = entry.argument_model.model_validate(arguments)
        return _redact_via_schema(validated, entry.argument_model)
    # Declarative branch added in Task 8.
    raise NotImplementedError(f"Declarative manifest entry for {tool_name!r} — see Task 8.")


# Manifest entries are added in waves (Tasks 4, 13, 14, 15, 16).  The
# binding is rebuilt as a new ``MappingProxyType`` per the spec §4.2.1
# rule "subsequent task waves extend the manifest by building a new
# dict, then replacing the module-level binding — never by mutating
# the proxy view".
MANIFEST: Mapping[str, ToolRedaction] = MappingProxyType(
    {
        "set_source": ToolRedaction(argument_model=SetSourceArgumentsModel),
    }
)


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
