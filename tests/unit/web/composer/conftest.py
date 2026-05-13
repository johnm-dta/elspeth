"""Conftest for composer-redaction tests.

Closes rev-5 architecture A2 collection-time concern for
``test_redaction_completeness_property.py``: ``Hypothesis.from_type`` cannot
resolve ``dict[str, typing.Any]`` (raised as ``InvalidArgument`` at example
generation), and several MANIFEST argument models use ``Annotated[dict[str,
Any], Sensitive(summarizer=...)]`` for option/slot/patch fields.

Two distinct Hypothesis-resolution issues are addressed here:

1.  **``dict[str, Any]`` resolution.**  Hypothesis cannot generate values for
    ``typing.Any`` because there is no runtime instance of ``Any``.  We register
    a function-style strategy for ``dict`` that inspects the type arguments
    and produces a value-strategy for ``Any``-typed values.

2.  **``Field(default_factory=dict)`` sentinel leakage.**  Pydantic 2.x exposes
    fields whose default is a factory via the ``FieldInfo.default_factory``
    attribute (public API).  Hypothesis's ``from_type`` for such fields emits
    ``one_of(just(<factory_sentinel>), <value_strategy>)``; the ``just`` arm
    produces the unevaluated sentinel object, which fails Pydantic validation
    (``Input should be a valid dictionary``).  We register explicit
    ``st.builds(Model, options=<dict_strategy>, ...)`` overrides for every
    composer-redaction model whose ``options`` field uses
    ``Field(default_factory=dict)`` so the sentinel arm never appears in the
    generation strategy.

**Why the 4 overrides are not auto-generated.**  Each ``st.builds(...)``
override below carries **load-bearing per-field customizations** beyond the
``options`` sentinel issue.  For example, ``_set_pipeline_source_strategy``
narrows ``inline_blob`` to ``st.one_of(st.none(), st.from_type(_InlineBlobModel))``
and ``_pipeline_node_strategy`` narrows roughly a dozen optional fields to
``st.one_of(st.none(), ...)``.  A blind introspector that auto-generated
strategies for every model with a ``default_factory=dict`` field would drop
these customizations and weaken example generation in ways that are not
immediately observable from test outcomes.  We therefore keep the overrides
explicit and use a collection-time **drift guard** (below) to ensure no new
MANIFEST model slips through without an override.

**Drift guard.**  ``_assert_default_factory_dict_models_have_overrides`` runs
at conftest import time.  It walks every ``ToolRedaction`` in MANIFEST
(transitively through nested Pydantic submodels) and identifies every model
that has at least one ``Field(default_factory=dict)`` field.  If any such
model is missing from ``_OVERRIDE_REGISTERED_MODELS``, the guard raises
``RuntimeError`` with a remediation hint *before any test runs*.  This
surfaces skew offensively at conftest load rather than letting it manifest
later as a cryptic Hypothesis ``InvalidArgument`` at example generation.

Scope and discipline notes:

* The strategies are intentionally scoped to the composer-redaction test
  directory (this conftest.py lives adjacent to the test files).  Hypothesis
  registrations are global within a process, but the affected models are
  composer-private types not used elsewhere in the test suite.
* The value strategy for ``Any`` is deliberately broad (``text | int | bool |
  None``) rather than exhaustive: the property test asserts that the
  *redacted* value differs from the *raw* value at every Sensitive path,
  regardless of T.  The test does not assert anything about the *contents*
  of ``Any``-typed values, so the strategy only needs to produce values
  Pydantic's ``model_validate`` accepts.

Plan task: Phase 2 / Task 19 (Hypothesis property test infrastructure).
Spec section: §4.2.6.
F6 follow-up: drift guard added per ``notes/composer-phase-2-followup-prompt-F1-F6.md``.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, get_args, get_origin

import hypothesis.strategies as st
from pydantic import BaseModel

from elspeth.web.composer.redaction import (
    MANIFEST,
    SetSourceFromBlobArgumentsModel,
    ToolRedaction,
    _InlineBlobModel,
    _NodeTriggerModel,
    _PipelineEdgeModel,
    _PipelineMetadataModel,
    _PipelineNodeModel,
    _PipelineOutputModel,
    _SetPipelineSourceModel,
)


def _dict_strategy(thing: type) -> st.SearchStrategy[dict[Any, Any]]:
    """Resolve ``dict[K, V]`` for the property test.

    Special-cases ``dict[str, Any]`` (and similar ``Any``-valued shapes)
    because Hypothesis cannot resolve ``typing.Any`` to a runtime instance.
    For non-``Any`` value types, defers to ``st.from_type`` for the value
    strategy so other dict shapes (e.g., ``dict[str, int]``) keep their
    default behaviour.
    """
    args = get_args(thing)
    if not args:
        # Bare ``dict`` annotation — produce string-keyed scalar values.
        return st.dictionaries(
            st.text(min_size=1, max_size=8),
            st.one_of(st.text(max_size=16), st.integers(), st.booleans(), st.none()),
            max_size=3,
        )
    key_type, value_type = args
    key_strategy = st.from_type(key_type) if key_type is not Any else st.text(min_size=1, max_size=8)
    if value_type is Any:
        value_strategy: st.SearchStrategy[Any] = st.one_of(
            st.text(max_size=16),
            st.integers(),
            st.booleans(),
            st.none(),
        )
    else:
        value_strategy = st.from_type(value_type)
    return st.dictionaries(key_strategy, value_strategy, max_size=3)


st.register_type_strategy(dict, _dict_strategy)


# ---------------------------------------------------------------------------
# Explicit ``st.builds`` overrides for models with ``Field(default_factory=dict)``.
#
# Without these overrides, Hypothesis emits ``one_of(just(<factory_sentinel>),
# <dict_strategy>)`` for ``options`` and Pydantic rejects the sentinel arm with
# ``Input should be a valid dictionary``.  Registering explicit builds for each
# such model forces Hypothesis to use the dict strategy directly.
#
# Every non-``options`` field is left to ``st.from_type`` so the override does
# not silently freeze unrelated field shapes (e.g., we still want random
# ``plugin`` strings, random ``inline_blob`` shapes, etc.).
#
# These overrides also carry load-bearing per-field narrowings that a blind
# auto-generator would drop (see module docstring).  The drift guard below
# protects against new MANIFEST models slipping through without an override;
# the customizations themselves remain hand-curated.
# ---------------------------------------------------------------------------


_OPTIONS_STRATEGY: st.SearchStrategy[dict[str, Any]] = st.dictionaries(
    st.text(min_size=1, max_size=8),
    st.one_of(st.text(max_size=16), st.integers(), st.booleans(), st.none()),
    max_size=3,
)


def _set_source_from_blob_strategy() -> st.SearchStrategy[SetSourceFromBlobArgumentsModel]:
    return st.builds(
        SetSourceFromBlobArgumentsModel,
        blob_id=st.text(),
        on_success=st.text(),
        plugin=st.one_of(st.none(), st.text()),
        on_validation_failure=st.one_of(st.none(), st.text()),
        options=_OPTIONS_STRATEGY,
    )


def _set_pipeline_source_strategy() -> st.SearchStrategy[_SetPipelineSourceModel]:
    return st.builds(
        _SetPipelineSourceModel,
        plugin=st.text(),
        on_success=st.text(),
        blob_id=st.one_of(st.none(), st.text()),
        options=_OPTIONS_STRATEGY,
        on_validation_failure=st.one_of(st.none(), st.text()),
        inline_blob=st.one_of(st.none(), st.from_type(_InlineBlobModel)),
    )


def _pipeline_node_strategy() -> st.SearchStrategy[_PipelineNodeModel]:
    return st.builds(
        _PipelineNodeModel,
        id=st.text(),
        node_type=st.text(),
        input=st.text(),
        plugin=st.one_of(st.none(), st.text()),
        on_success=st.one_of(st.none(), st.text()),
        on_error=st.one_of(st.none(), st.text()),
        options=_OPTIONS_STRATEGY,
        condition=st.one_of(st.none(), st.text()),
        # F3: ``routes`` is now ``dict[str, str]`` (route-label → sink/connection
        # identifier); generate strings only.  ``trigger`` is a typed
        # :class:`_NodeTriggerModel`; defer to ``st.from_type`` so future field
        # additions to the trigger sub-model are picked up automatically.
        routes=st.one_of(
            st.none(),
            st.dictionaries(st.text(min_size=1, max_size=8), st.text(max_size=16), max_size=3),
        ),
        fork_to=st.one_of(st.none(), st.lists(st.text(), max_size=3)),
        branches=st.one_of(st.none(), st.lists(st.text(), max_size=3)),
        policy=st.one_of(st.none(), st.text()),
        merge=st.one_of(st.none(), st.text()),
        trigger=st.one_of(st.none(), st.from_type(_NodeTriggerModel)),
        output_mode=st.one_of(st.none(), st.text()),
        expected_output_count=st.one_of(st.none(), st.integers()),
    )


def _pipeline_output_strategy() -> st.SearchStrategy[_PipelineOutputModel]:
    return st.builds(
        _PipelineOutputModel,
        sink_name=st.text(),
        plugin=st.text(),
        options=_OPTIONS_STRATEGY,
        on_write_failure=st.one_of(st.none(), st.text()),
    )


st.register_type_strategy(SetSourceFromBlobArgumentsModel, _set_source_from_blob_strategy())
st.register_type_strategy(_SetPipelineSourceModel, _set_pipeline_source_strategy())
st.register_type_strategy(_PipelineNodeModel, _pipeline_node_strategy())
st.register_type_strategy(_PipelineOutputModel, _pipeline_output_strategy())


# Mirror of the four explicit ``st.register_type_strategy`` calls above.  This
# tuple is the single source of truth that the drift guard consults — if you
# add a new ``st.register_type_strategy(Model, ...)`` for a model with
# ``Field(default_factory=dict)``, you MUST add ``Model`` to this tuple in the
# same edit.  Conversely, adding a new MANIFEST model with
# ``Field(default_factory=dict)`` and forgetting an override here will cause
# the drift guard to raise at conftest import time.
_OVERRIDE_REGISTERED_MODELS: tuple[type[BaseModel], ...] = (
    SetSourceFromBlobArgumentsModel,
    _SetPipelineSourceModel,
    _PipelineNodeModel,
    _PipelineOutputModel,
)


# Reference _PipelineEdgeModel and _PipelineMetadataModel here to silence
# unused-import warnings — they are not strategically overridden because
# they have no ``Field(default_factory=dict)`` fields, but the property test
# does build them transitively via st.from_type which works correctly.
_REFERENCED_FOR_DOCUMENTATION: tuple[type, ...] = (_PipelineEdgeModel, _PipelineMetadataModel)


def _iter_nested_models(model: type[BaseModel]) -> set[type[BaseModel]]:
    """Return ``model`` plus every Pydantic BaseModel reachable from its fields.

    Walks ``model_fields`` and inspects each field's annotation.  When an
    annotation contains another Pydantic BaseModel (directly, or as a
    parametrization argument of ``list[...]``, ``dict[..., ...]``,
    ``Optional[...]``, ``Union[..., None]``, etc.), that submodel is added to
    the returned set and recursively walked.

    Cycle-safe: an internal ``visited`` set prevents infinite recursion on
    self-referential model graphs.
    """
    visited: set[type[BaseModel]] = set()

    def _walk(m: type[BaseModel]) -> None:
        if m in visited:
            return
        visited.add(m)
        for field_info in m.model_fields.values():
            for inner in _flatten_annotation(field_info.annotation):
                if isinstance(inner, type) and issubclass(inner, BaseModel):
                    _walk(inner)

    _walk(model)
    return visited


def _flatten_annotation(annotation: object) -> list[object]:
    """Flatten a typing annotation into its constituent runtime types.

    Yields the top-level annotation and recurses into ``get_args(...)`` so
    that wrappers like ``Optional[X]``, ``Union[X, Y]``, ``list[X]``,
    ``dict[K, V]``, and ``Annotated[X, ...]`` expose ``X``, ``Y``, ``K``, ``V``
    as candidate runtime types.  Non-type metadata (e.g., ``Sensitive(...)``
    markers attached via ``Annotated``) flows through unchanged; the caller
    filters via ``isinstance(inner, type) and issubclass(inner, BaseModel)``.
    """
    flattened: list[object] = [annotation]
    origin = get_origin(annotation)
    if origin is not None:
        for arg in get_args(annotation):
            flattened.extend(_flatten_annotation(arg))
    return flattened


def _models_needing_override(manifest: Mapping[str, ToolRedaction]) -> dict[type[BaseModel], list[str]]:
    """Identify every model in the manifest's transitive closure with a ``Field(default_factory=dict)`` field.

    Returns a mapping from offending model class → list of field names whose
    ``default_factory`` is ``dict`` (the bare ``dict`` constructor).  Only
    ``default_factory is dict`` is flagged: other factories (e.g.,
    ``default_factory=list`` for list fields) do not trigger the Hypothesis
    sentinel-arm problem this conftest exists to address.
    """
    needs_override: dict[type[BaseModel], list[str]] = {}
    seen_top_level: set[type[BaseModel]] = set()
    for entry in manifest.values():
        for top in (entry.argument_model, entry.response_model):
            if top is None or top in seen_top_level:
                continue
            seen_top_level.add(top)
            for model in _iter_nested_models(top):
                offending_fields = [
                    field_name for field_name, field_info in model.model_fields.items() if field_info.default_factory is dict
                ]
                if offending_fields:
                    # De-duplicate field names across multiple manifest
                    # entries that point at the same nested model.
                    existing = needs_override.setdefault(model, [])
                    for field_name in offending_fields:
                        if field_name not in existing:
                            existing.append(field_name)
    return needs_override


def _assert_default_factory_dict_models_have_overrides(
    manifest: Mapping[str, ToolRedaction],
    registered: tuple[type[BaseModel], ...],
) -> None:
    """Crash at conftest load if a MANIFEST model lacks an override.

    Walks ``manifest`` transitively through nested Pydantic submodels and
    raises ``RuntimeError`` if any model with ``Field(default_factory=dict)``
    fields is missing from ``registered``.  The error message names every
    offender, lists the offending fields, and points at this conftest with a
    remediation hint.

    Parameterized on ``manifest`` and ``registered`` so the drift guard
    itself is unit-testable from a sibling test module.
    """
    needs_override = _models_needing_override(manifest)
    registered_set: set[type[BaseModel]] = set(registered)
    missing = {model: fields for model, fields in needs_override.items() if model not in registered_set}
    if not missing:
        return
    lines = [
        "Hypothesis strategy drift detected: the following composer-redaction "
        "models have Field(default_factory=dict) fields but no explicit "
        "st.register_type_strategy(...) override:",
        "",
    ]
    for model, fields in sorted(missing.items(), key=lambda kv: kv[0].__name__):
        lines.append(f"  - {model.__module__}.{model.__name__}: fields={fields}")
    lines.extend(
        [
            "",
            "Without an override, Hypothesis emits one_of(just(<factory_sentinel>), "
            "<dict_strategy>) for the default_factory=dict field and the sentinel "
            "arm fails Pydantic validation with 'Input should be a valid dictionary'.",
            "",
            "Add an explicit `st.builds(...)` override and call "
            "`st.register_type_strategy(<Model>, ...)` in "
            "`tests/unit/web/composer/conftest.py`. See the existing 4 overrides "
            "as templates. F6 hardening — see "
            "`notes/composer-phase-2-followup-prompt-F1-F6.md`.",
        ]
    )
    raise RuntimeError("\n".join(lines))


_assert_default_factory_dict_models_have_overrides(MANIFEST, _OVERRIDE_REGISTERED_MODELS)
