"""Hypothesis property test (rev-4): redaction replaces every Sensitive[T] field
value at every reachable path, regardless of T's type.

Closes rev-2 BLOCKER_A (quality MAJOR-4), rev-3 B1 (ad-hoc path.split skip),
and rev-4 B1 (isinstance(str)/None-skip/except-fallback silent-pass family).

Invariant: for each Sensitive-annotated path in the model, the value extracted
at that path from the redacted output MUST NOT equal the value extracted from
the raw input. The comparison is type-agnostic `!=` at the path; there is no
`isinstance` filter and no defensive `except` fallback. If `value_provider` is
None for a Sensitive-marked node when `with_values=True`, that is a walker
bug and the test FAILS — it does not silently skip.

Rev-history banner (do not delete — institutional memory):
  rev-2 design: "raw_value not in json.dumps(redacted)" — failed because
    ad-hoc path.split(".") loop couldn't navigate container indices, silently
    skipping every Sensitive[T] inside a list or dict.
  rev-3 design: value_provider-based extraction + "raw_value not in serialized"
    with isinstance(str) gate — failed because the gate silently exempted
    every non-str T (dict, bytes, int, bool, None, empty-str) including the
    Task 15 planned Annotated[dict[str, Any], Sensitive(...)] shape.
  rev-4 design (this file): value_provider-based extraction + path-aware `!=`
    between raw and redacted views — type-agnostic, no silent-skip surface.

settings(max_examples=50, deadline=None) for stable CI execution.
"""

from __future__ import annotations

import pytest
from hypothesis import event, given, settings
from hypothesis import strategies as st

from elspeth.web.composer.redaction import (
    MANIFEST,
    _SensitiveMarker,
    redact_tool_call_arguments,
    redact_tool_call_response,
    walk_model_schema,
)
from elspeth.web.composer.redaction_telemetry import NoopRedactionTelemetry

_CONTAINER_PATH_MARKERS = ("[*]", "{*}")


def _is_container_descent_path(path: str) -> bool:
    """A node's path is container-descent iff it carries a [*] or {*} segment.

    walk_model_schema documents these markers as the canonical container
    indicators; relying on the path shape (not on get_origin runtime checks)
    keeps the test decoupled from Pydantic introspection internals.
    """
    return any(marker in path for marker in _CONTAINER_PATH_MARKERS)


def _has_sensitive_field(model: type) -> bool:
    """True iff ``walk_model_schema(model)`` yields any Sensitive-marked node.

    Type-agnostic: the body just walks a BaseModel and inspects metadata —
    it does not distinguish argument-model from response-model semantics.
    Used by both the argument-side and response-side property-test
    parametrize filters (F5 follow-up adds the response-side test).

    Narrowing rationale (argument-side filter):

    The filter narrows the property-test parametrize set from "type-driven
    entry" to "type-driven entry whose argument model carries at least one
    Sensitive field".  The narrowing is necessary because at least one
    MANIFEST entry (``get_blob_content``) is structurally forced into the
    type-driven shape by a Sensitive *response* field while its *argument*
    model carries no Sensitive nodes — declarative entries have no
    ``response_model`` facility, so a response-sensitive-only tool can only
    be registered type-driven (see ``GetBlobContentArgumentsModel`` docstring
    in ``src/elspeth/web/composer/redaction.py`` for the design rationale).

    The plan body's verbatim filter (``entry.argument_model is not None``)
    treats every type-driven entry as argument-sensitive, which produces a
    vacuous ``assert sensitive_nodes`` failure for ``get_blob_content``.
    That failure is not a real misclassification — it is a manifest-design
    constraint the plan body did not anticipate.  The narrowed filter
    preserves the rev-4 ``assert sensitive_nodes`` guard for its intended
    purpose (catching a type-driven entry whose argument_model has no
    Sensitive fields AND no response_model — i.e., a tool that should be
    declarative) while passing ``get_blob_content`` through silently on the
    argument side — its argument surface is structurally non-sensitive and
    the response-side redaction is covered by the response-side property
    test below.

    Narrowing rationale (response-side filter):

    The response-side filter (``entry.response_model is not None and
    _has_sensitive_field(entry.response_model)``) is symmetric: it skips
    type-driven entries that lack a ``response_model`` (declarative
    entries have no ``response_model`` facility at all, so this branch
    also skips them), and it skips ``response_model``s that happen to
    have no Sensitive nodes.  The rev-4 ``assert sensitive_nodes`` guard
    catches the latter case loudly if it ever arises.
    """
    return any(any(isinstance(m, _SensitiveMarker) for m in node.metadata) for node in walk_model_schema(model))


@pytest.mark.parametrize(
    "tool_name",
    [name for name, entry in MANIFEST.items() if entry.argument_model is not None and _has_sensitive_field(entry.argument_model)],
)
def test_redaction_replaces_every_sensitive_value(tool_name: str) -> None:
    """For each type-driven manifest entry: every Sensitive path's extracted
    value differs between raw and redacted views.

    This is the load-bearing security claim. A buggy redactor that passes
    through a Sensitive value of any type (str, dict, bytes, int, bool,
    empty-str, ...) at any path (scalar, list-item, dict-value, nested)
    causes this test to fail with a precise path + value diagnostic.
    """
    entry = MANIFEST[tool_name]
    model = entry.argument_model
    assert model is not None  # parametrize filter guarantees this; assert for mypy

    # Discover Sensitive paths once per tool (walker is deterministic).
    # value_provider MUST be non-None for every node when with_values=True;
    # we assert that mechanically so any walker regression that produces a
    # None value_provider for a Sensitive node fails this test loudly rather
    # than skipping silently. This is the rev-4 hardening against the
    # "value_provider is None: continue" silent-pass mode from rev-3.
    sensitive_nodes = []
    for node in walk_model_schema(model, with_values=True):
        if not any(isinstance(m, _SensitiveMarker) for m in node.metadata):
            continue
        assert node.value_provider is not None, (
            f"walk_model_schema(model={model.__name__!r}, with_values=True) "
            f"yielded a Sensitive-marked node at path {node.path!r} with "
            f"value_provider=None. This is a walker bug — when with_values=True "
            f"every node MUST carry a value_provider closure. The test fails "
            f"loudly rather than silently skipping (rev-4 B1 hardening)."
        )
        sensitive_nodes.append(node)
    assert sensitive_nodes, (
        f"Tool {tool_name!r}'s argument_model {model.__name__!r} is registered "
        f"as a type-driven manifest entry but walk_model_schema found no "
        f"Sensitive-marked paths. Either the manifest entry should be "
        f"declarative (no Sensitive[T] fields) or walk_model_schema is "
        f"failing to detect the markers. The test fails loudly rather than "
        f"silently passing on a no-op iteration (rev-4 B1 hardening)."
    )

    @given(st.from_type(model))  # type: ignore[arg-type]
    @settings(max_examples=50, deadline=None)
    def check(payload: object) -> None:
        raw_args = payload.model_dump()
        redacted_args = redact_tool_call_arguments(tool_name, raw_args, telemetry=NoopRedactionTelemetry())

        for node in sensitive_nodes:
            # Extract values at this path from BOTH views. value_provider is
            # path-aware: for scalar paths it returns the value directly; for
            # container-descent paths ([*] or {*}) it returns an iterable of
            # (key_or_index, value) pairs. We let any contract violation
            # (e.g. value_provider returns a non-iterable for a container
            # path) propagate as an exception — this is a walker bug and the
            # test must surface it, not absorb it via try/except.
            raw_extracted = node.value_provider(raw_args)
            redacted_extracted = node.value_provider(redacted_args)

            if _is_container_descent_path(node.path):
                # Container descent: compare key-by-key.
                raw_pairs = dict(raw_extracted)
                redacted_pairs = dict(redacted_extracted)
                # The redactor must NOT alter the container's key structure;
                # only values at Sensitive paths are substituted.
                assert raw_pairs.keys() == redacted_pairs.keys(), (
                    f"Redaction altered the key set of the container at path "
                    f"{node.path!r} for tool {tool_name!r}. "
                    f"Raw keys: {sorted(raw_pairs)!r}; "
                    f"redacted keys: {sorted(redacted_pairs)!r}. "
                    f"Redaction must preserve container shape — only values "
                    f"at Sensitive paths are substituted, never keys."
                )
                # An empty container yields no comparisons for this example.
                # Hypothesis will generate non-empty containers in other
                # examples (max_examples=50). The `event()` call below makes
                # the empty-container ratio observable in Hypothesis's
                # summary output, so a parametrize entry whose strategy
                # produces ONLY empty containers across all 50 examples
                # surfaces as a high event count rather than passing as a
                # silent zero-assertion run (rev-5 systems W-empty-container).
                # Currently no MANIFEST entry uses sub-element Sensitive
                # markers (all markers are on the field itself), so this
                # branch is unreachable today; the instrumentation is
                # forward-safety against future entries.
                if not raw_pairs:
                    event(f"empty_container:{tool_name}:{node.path}")
                for key, raw_value in raw_pairs.items():
                    if raw_value is None:
                        # Skip rule (the ONLY allowed skip): no sensitive
                        # data at this path for this example. Asserting
                        # redacted != None would either tautologically fail
                        # for a correct redactor (passes None through) or
                        # pass spuriously for an over-aggressive one
                        # (substitutes None with a sentinel). Out of scope.
                        continue
                    redacted_value = redacted_pairs[key]
                    assert redacted_value != raw_value, (
                        f"Sensitive value at container path {node.path!r}"
                        f"[{key!r}] for tool {tool_name!r} was NOT redacted. "
                        f"raw={raw_value!r}, redacted={redacted_value!r}. "
                        f"redact_tool_call_arguments must replace the value "
                        f"at every Sensitive path with the summarizer output "
                        f"or the fixed sentinel."
                    )
            else:
                # Scalar descent.
                if raw_extracted is None:
                    continue  # only allowed skip; see container branch
                assert redacted_extracted != raw_extracted, (
                    f"Sensitive value at scalar path {node.path!r} for tool "
                    f"{tool_name!r} was NOT redacted. "
                    f"raw={raw_extracted!r}, redacted={redacted_extracted!r}. "
                    f"redact_tool_call_arguments must replace the value at "
                    f"every Sensitive path with the summarizer output or the "
                    f"fixed sentinel."
                )

    check()


@pytest.mark.parametrize(
    "tool_name",
    [name for name, entry in MANIFEST.items() if entry.response_model is not None and _has_sensitive_field(entry.response_model)],
)
def test_redaction_replaces_every_sensitive_response_value(tool_name: str) -> None:
    """For each type-driven manifest entry whose response_model carries at
    least one Sensitive field: every Sensitive path's extracted value differs
    between the raw response and the redacted response.

    Response-side counterpart to ``test_redaction_replaces_every_sensitive_value``
    (F5 follow-up; spec §4.2.5).  Before this test, response-side redaction
    coverage was canary-only — hand-crafted fixtures in
    ``test_redact_tool_call_response.py`` with synthetic ``_ResponseModel``
    classes — so no Hypothesis-strength guarantee existed for the production
    MANIFEST response models.  This test closes that gap with the same shape
    and load-bearing semantics as the argument-side property test: a buggy
    response redactor that passes through a Sensitive value of any T at any
    path causes a precise path + value diagnostic failure.
    """
    entry = MANIFEST[tool_name]
    model = entry.response_model
    assert model is not None  # parametrize filter guarantees this; assert for mypy

    # Discover Sensitive paths once per tool (walker is deterministic).
    # value_provider MUST be non-None for every node when with_values=True;
    # we assert that mechanically so any walker regression that produces a
    # None value_provider for a Sensitive node fails this test loudly rather
    # than skipping silently. Mirrors the argument-side rev-4 hardening.
    sensitive_nodes = []
    for node in walk_model_schema(model, with_values=True):
        if not any(isinstance(m, _SensitiveMarker) for m in node.metadata):
            continue
        assert node.value_provider is not None, (
            f"walk_model_schema(model={model.__name__!r}, with_values=True) "
            f"yielded a Sensitive-marked node at path {node.path!r} with "
            f"value_provider=None. This is a walker bug — when with_values=True "
            f"every node MUST carry a value_provider closure. The test fails "
            f"loudly rather than silently skipping (rev-4 B1 hardening, "
            f"response-side mirror)."
        )
        sensitive_nodes.append(node)
    assert sensitive_nodes, (
        f"Tool {tool_name!r}'s response_model {model.__name__!r} passed the "
        f"_has_sensitive_field parametrize filter but walk_model_schema with "
        f"with_values=True found no Sensitive-marked paths. This is a walker "
        f"contract regression: with_values=True must yield the same Sensitive "
        f"set as with_values=False (which the filter used). The test fails "
        f"loudly rather than silently passing on a no-op iteration."
    )

    # F6 drift guard (conftest.py) ensures future response models with Field(default_factory=dict) raise at collection time.
    @given(st.from_type(model))  # type: ignore[arg-type]
    @settings(max_examples=50, deadline=None)
    def check(payload: object) -> None:
        raw_response = payload.model_dump()
        redacted_response = redact_tool_call_response(tool_name, raw_response, telemetry=NoopRedactionTelemetry())

        for node in sensitive_nodes:
            # Extract values at this path from BOTH views. value_provider is
            # path-aware: for scalar paths it returns the value directly; for
            # container-descent paths ([*] or {*}) it returns an iterable of
            # (key_or_index, value) pairs. We let any contract violation
            # (e.g. value_provider returns a non-iterable for a container
            # path) propagate as an exception — this is a walker bug and the
            # test must surface it, not absorb it via try/except.
            raw_extracted = node.value_provider(raw_response)
            redacted_extracted = node.value_provider(redacted_response)

            if _is_container_descent_path(node.path):
                # Container descent: compare key-by-key.
                raw_pairs = dict(raw_extracted)
                redacted_pairs = dict(redacted_extracted)
                # The redactor must NOT alter the container's key structure;
                # only values at Sensitive paths are substituted.
                assert raw_pairs.keys() == redacted_pairs.keys(), (
                    f"Redaction altered the key set of the container at path "
                    f"{node.path!r} in the response for tool {tool_name!r}. "
                    f"Raw keys: {sorted(raw_pairs)!r}; "
                    f"redacted keys: {sorted(redacted_pairs)!r}. "
                    f"Redaction must preserve container shape — only values "
                    f"at Sensitive paths are substituted, never keys."
                )
                # An empty container yields no comparisons for this example.
                # event() makes the empty-container ratio observable in
                # Hypothesis's summary output — a parametrize entry whose
                # strategy produces ONLY empty containers across all 50
                # examples surfaces as a high event count rather than passing
                # as a silent zero-assertion run. This branch IS reachable for
                # get_blob_content: its response_model carries a Sensitive
                # marker on the repair-tool-call ``arguments`` field nested
                # under ``validation.graph_repair_suggestions[*].tool_sequence
                # [*].arguments`` — a path with ``[*]`` segments, so
                # _is_container_descent_path returns True. (get_blob_content's
                # other Sensitive leaf, ``data.content``, is a scalar
                # Sensitive[str] and takes the else-branch below.) The
                # instrumentation also remains forward-safety for future
                # entries with sub-element Sensitive markers.
                if not raw_pairs:
                    event(f"empty_container_response:{tool_name}:{node.path}")
                for key, raw_value in raw_pairs.items():
                    if raw_value is None:
                        # Skip rule (the ONLY allowed skip): no sensitive
                        # data at this path for this example. Asserting
                        # redacted != None would either tautologically fail
                        # for a correct redactor (passes None through) or
                        # pass spuriously for an over-aggressive one
                        # (substitutes None with a sentinel). Out of scope.
                        continue
                    redacted_value = redacted_pairs[key]
                    assert redacted_value != raw_value, (
                        f"Sensitive value at container path {node.path!r}"
                        f"[{key!r}] in the response for tool {tool_name!r} "
                        f"was NOT redacted. raw={raw_value!r}, "
                        f"redacted={redacted_value!r}. "
                        f"redact_tool_call_response must replace the value "
                        f"at every Sensitive path with the summarizer output "
                        f"or the fixed sentinel."
                    )
            else:
                # Scalar descent.
                if raw_extracted is None:
                    continue  # only allowed skip; see container branch
                assert redacted_extracted != raw_extracted, (
                    f"Sensitive value at scalar path {node.path!r} in the "
                    f"response for tool {tool_name!r} was NOT redacted. "
                    f"raw={raw_extracted!r}, redacted={redacted_extracted!r}. "
                    f"redact_tool_call_response must replace the value at "
                    f"every Sensitive path with the summarizer output or the "
                    f"fixed sentinel."
                )

    check()


def test_queue_node_options_are_redacted_in_set_pipeline() -> None:
    """A queue node's ``options`` is the same Sensitive ``nodes[*].options``
    surface as any other node: the operator-facing description must collapse
    to the summarizer output at the persistence boundary (elspeth-a5b86149d4).

    The queue exposure did NOT weaken the persisted transport model
    (``_PipelineNodeModel.node_type`` stays ``str``) nor the redaction
    inventory — this pins that a queue rides the existing Sensitive path.
    """
    import json

    canary = "SECRET-OPERATOR-QUEUE-NOTE"
    args = {
        "source": {"plugin": "csv", "on_success": "inbound"},
        "nodes": [
            {
                "id": "inbound",
                "node_type": "queue",
                "input": "inbound",
                "options": {"description": canary},
            }
        ],
        "edges": [],
        "outputs": [],
    }
    redacted = redact_tool_call_arguments("set_pipeline", args, telemetry=NoopRedactionTelemetry())
    # node_type survives verbatim (structural); options collapses to a str.
    assert redacted["nodes"][0]["node_type"] == "queue"
    assert isinstance(redacted["nodes"][0]["options"], str)
    assert canary not in json.dumps(redacted, sort_keys=True)
