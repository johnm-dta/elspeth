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
    fields whose default is a factory via a ``_HAS_DEFAULT_FACTORY`` sentinel.
    Hypothesis's ``from_type`` for such fields emits
    ``one_of(just(<factory_sentinel>), <value_strategy>)``; the ``just`` arm
    produces the unevaluated sentinel object, which fails Pydantic validation
    (``Input should be a valid dictionary``).  We register explicit
    ``st.builds(Model, options=<dict_strategy>, ...)`` overrides for every
    composer-redaction model whose ``options`` field uses
    ``Field(default_factory=dict)`` so the sentinel arm never appears in the
    generation strategy.

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
"""

from __future__ import annotations

from typing import Any, get_args

import hypothesis.strategies as st

from elspeth.web.composer.redaction import (
    SetSourceFromBlobArgumentsModel,
    _InlineBlobModel,
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
        routes=st.one_of(st.none(), _OPTIONS_STRATEGY),
        fork_to=st.one_of(st.none(), st.lists(st.text(), max_size=3)),
        branches=st.one_of(st.none(), st.lists(st.text(), max_size=3)),
        policy=st.one_of(st.none(), st.text()),
        merge=st.one_of(st.none(), st.text()),
        trigger=st.one_of(st.none(), _OPTIONS_STRATEGY),
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


# Reference _PipelineEdgeModel and _PipelineMetadataModel here to silence
# unused-import warnings — they are not strategically overridden because
# they have no ``Field(default_factory=dict)`` fields, but the property test
# does build them transitively via st.from_type which works correctly.
_REFERENCED_FOR_DOCUMENTATION: tuple[type, ...] = (_PipelineEdgeModel, _PipelineMetadataModel)
