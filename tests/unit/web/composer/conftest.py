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
F6 follow-up: drift guard added per ``docs/composer/evidence/composer-phase-2-followup-prompt-F1-F6.md``.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, get_args, get_origin
from unittest.mock import MagicMock
from uuid import uuid4

import hypothesis.strategies as st
import pytest
import structlog
from pydantic import BaseModel
from sqlalchemy import event
from sqlalchemy.engine import Engine
from sqlalchemy.exc import IntegrityError, OperationalError
from sqlalchemy.pool import StaticPool

from elspeth.web.catalog.policy_view import PolicyCatalogView
from elspeth.web.catalog.protocol import CatalogService
from elspeth.web.catalog.schemas import PluginSchemaInfo, PluginSummary
from elspeth.web.composer import tools as tools_module
from elspeth.web.composer.protocol import ToolArgumentError
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
    _RepairToolCallShadowModel,
    _SetPipelineNamedSourceModel,
    _SetPipelineSourceModel,
    _SpliceTransformNodeModel,
)
from elspeth.web.composer.service import ComposerAvailability, ComposerServiceImpl
from elspeth.web.composer.state import CompositionState, PipelineMetadata
from elspeth.web.composer.tools._common import ToolContext
from elspeth.web.config import WebSettings
from elspeth.web.plugin_policy.models import PluginAvailabilitySnapshot
from elspeth.web.sessions.engine import create_session_engine
from elspeth.web.sessions.models import sessions_table
from elspeth.web.sessions.schema import initialize_session_schema
from elspeth.web.sessions.service import SessionServiceImpl
from elspeth.web.sessions.telemetry import build_sessions_telemetry


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


def _set_pipeline_named_source_strategy() -> st.SearchStrategy[_SetPipelineNamedSourceModel]:
    return st.builds(
        _SetPipelineNamedSourceModel,
        plugin=st.text(),
        on_success=st.text(),
        options=_OPTIONS_STRATEGY,
        on_validation_failure=st.one_of(st.none(), st.text()),
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


def _splice_transform_node_strategy() -> st.SearchStrategy[_SpliceTransformNodeModel]:
    return st.builds(
        _SpliceTransformNodeModel,
        id=st.text(),
        plugin=st.text(),
        options=_OPTIONS_STRATEGY,
        on_error=st.one_of(st.none(), st.text()),
    )


def _repair_tool_call_strategy() -> st.SearchStrategy[_RepairToolCallShadowModel]:
    """Resolve ``_RepairToolCallShadowModel`` for the property test.

    The shadow model's ``arguments`` field is typed ``Mapping[str, object]`` to
    mirror ``_RepairToolCall`` (tools/_common.py) faithfully. ``st.from_type``
    would fill ``object`` with arbitrary, sometimes non-JSON, non-deepcopy-able
    instances (e.g. ``super(NoneType)``), which crashes the redactor's
    ``copy.deepcopy(model_dump())`` step — a pure strategy artifact, since at
    runtime the value always originates from a parsed JSON response dict.

    We constrain ``arguments`` to a string-keyed dict of JSON scalars so the
    generated examples are deepcopy-safe and faithful to the real wire shape.
    The ``arguments`` leaf carries a Sensitive summarizer
    (:func:`_summarize_repair_arguments`); the property test asserts only that
    the redacted value differs from the raw value at that Sensitive path, so the
    scalar value-space is sufficient — it does not assert on the contents.
    """
    return st.builds(
        _RepairToolCallShadowModel,
        tool=st.text(),
        arguments=st.dictionaries(
            st.text(min_size=1, max_size=8),
            st.one_of(st.text(max_size=16), st.integers(), st.booleans(), st.none()),
            max_size=3,
        ),
    )


st.register_type_strategy(SetSourceFromBlobArgumentsModel, _set_source_from_blob_strategy())
st.register_type_strategy(_SetPipelineSourceModel, _set_pipeline_source_strategy())
st.register_type_strategy(_SetPipelineNamedSourceModel, _set_pipeline_named_source_strategy())
st.register_type_strategy(_PipelineNodeModel, _pipeline_node_strategy())
st.register_type_strategy(_PipelineOutputModel, _pipeline_output_strategy())
st.register_type_strategy(_SpliceTransformNodeModel, _splice_transform_node_strategy())
st.register_type_strategy(_RepairToolCallShadowModel, _repair_tool_call_strategy())


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
    _SetPipelineNamedSourceModel,
    _PipelineNodeModel,
    _PipelineOutputModel,
    _SpliceTransformNodeModel,
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
            "`docs/composer/evidence/composer-phase-2-followup-prompt-F1-F6.md`.",
        ]
    )
    raise RuntimeError("\n".join(lines))


_assert_default_factory_dict_models_have_overrides(MANIFEST, _OVERRIDE_REGISTERED_MODELS)


# ---------------------------------------------------------------------------
# Phase 3 compose-loop persistence harness fixtures.
#
# These helpers are intentionally defined in the package-level conftest before
# the Phase 3 red tests are authored. The rev-6 quality ground rule forbids red
# tests that fail because their scaffolding names do not exist.
# ---------------------------------------------------------------------------


@dataclass
class _FakeFunction:
    name: str
    arguments: str


@dataclass
class _FakeToolCall:
    id: str
    function: _FakeFunction


@dataclass
class _FakeMessage:
    content: str | None
    tool_calls: list[_FakeToolCall] | None


@dataclass
class _FakeChoice:
    message: _FakeMessage


@dataclass
class _FakeLLMResponse:
    choices: list[_FakeChoice]


class _FakeComposeLLM:
    """Callable fake LLM used by ``_run_one_turn_for_test`` fixtures."""

    def __init__(self, responses: tuple[_FakeLLMResponse, ...]) -> None:
        self._responses = list(responses)
        self.execute_tool_invocations = 0

    async def __call__(self, _messages: Any, _tools: Any) -> _FakeLLMResponse:
        if not self._responses:
            return _fake_llm_response(content="Done.")
        return self._responses.pop(0)


def _fake_llm_response(
    *,
    content: str | None = None,
    tool_calls: tuple[dict[str, Any], ...] = (),
) -> _FakeLLMResponse:
    fake_tool_calls: list[_FakeToolCall] | None = None
    if tool_calls:
        fake_tool_calls = [
            _FakeToolCall(
                id=str(call["id"]),
                function=_FakeFunction(
                    name=str(call["name"]),
                    arguments=json.dumps(call.get("arguments", {})),
                ),
            )
            for call in tool_calls
        ]
    return _FakeLLMResponse(choices=[_FakeChoice(message=_FakeMessage(content=content, tool_calls=fake_tool_calls))])


def _empty_composition_state() -> CompositionState:
    return CompositionState(source=None, nodes=(), edges=(), outputs=(), metadata=PipelineMetadata(), version=1)


def _mock_catalog() -> MagicMock:
    catalog = MagicMock(spec=CatalogService)
    catalog.list_sources.return_value = [
        PluginSummary(name="csv", description="CSV source", plugin_type="source", config_fields=[]),
    ]
    catalog.list_transforms.return_value = [
        PluginSummary(name="passthrough", description="Passthrough", plugin_type="transform", config_fields=[]),
    ]
    catalog.list_sinks.return_value = [
        PluginSummary(name="csv", description="CSV sink", plugin_type="sink", config_fields=[]),
    ]
    catalog.get_schema.return_value = PluginSchemaInfo(
        name="csv",
        plugin_type="source",
        description="CSV source",
        json_schema={"title": "Config", "properties": {}},
        knob_schema={"fields": []},
    )
    return catalog


@pytest.fixture
def tool_context() -> ToolContext:
    """Bare ToolContext for unit tests that only need a catalog.

    Use ``replace(tool_context, ...)`` if a test needs additional fields
    (data_dir, secret_service, etc.).
    """

    catalog = _mock_catalog()
    snapshot = PluginAvailabilitySnapshot.for_trained_operator(catalog)
    return ToolContext(
        catalog=PolicyCatalogView.for_trained_operator(catalog, snapshot),
        plugin_snapshot=snapshot,
    )


@pytest.fixture
def make_tool_context() -> Any:
    """Factory for ToolContext with arbitrary overrides.

    Usage::

        ctx = make_tool_context(data_dir="/tmp/data", secret_service=svc)
    """

    def _factory(**overrides: Any) -> ToolContext:
        catalog = overrides.pop("catalog", _mock_catalog())
        snapshot = PluginAvailabilitySnapshot.for_trained_operator(catalog)
        return ToolContext(
            catalog=PolicyCatalogView.for_trained_operator(catalog, snapshot),
            plugin_snapshot=snapshot,
            **overrides,
        )

    return _factory


def _make_settings(data_dir: Path, **overrides: Any) -> WebSettings:
    values: dict[str, Any] = {
        "data_dir": data_dir,
        "composer_max_composition_turns": 15,
        "composer_max_discovery_turns": 10,
        "composer_timeout_seconds": 85.0,
        "composer_rate_limit_per_minute": 10,
        "shareable_link_signing_key": b"\x00" * 32,
    }
    values.update(overrides)
    return WebSettings(**values)


@pytest.fixture(autouse=True)
def _composer_available_for_phase3(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep Phase 3 compose-loop harness tests independent of local API keys."""

    def _available(self: ComposerServiceImpl) -> ComposerAvailability:
        return ComposerAvailability(available=True, model=self._model, provider="test")

    monkeypatch.setattr(ComposerServiceImpl, "_compute_availability", _available)


@pytest.fixture
def result_session_id(composer_service_with_real_sessions: ComposerServiceImpl) -> str:
    """Session id used by ``_run_one_turn_for_test`` result assertions."""

    session_id = str(uuid4())
    now = datetime.now(UTC)
    sessions_service = composer_service_with_real_sessions._sessions_service
    with sessions_service._engine.begin() as conn:
        conn.execute(
            sessions_table.insert().values(
                id=session_id,
                user_id="phase3-test-user",
                auth_provider_type="local",
                title="Phase 3 test session",
                trust_mode="auto_commit",
                density_default="high",
                created_at=now,
                updated_at=now,
            )
        )
    return session_id


def build_test_sessions_service(
    *,
    engine: Engine | None = None,
    data_dir: Path | None = None,
) -> SessionServiceImpl:
    """Build a real SQLite-backed ``SessionServiceImpl`` for composer tests.

    Uses ``create_session_engine(..., poolclass=StaticPool)`` plus
    ``initialize_session_schema(engine)`` so tests exercise the same schema
    bootstrap path as production. Bare ``metadata.create_all()`` is not used.
    """

    resolved_engine = engine or create_session_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    if engine is None:
        initialize_session_schema(resolved_engine)
    return SessionServiceImpl(
        resolved_engine,
        data_dir=data_dir,
        telemetry=build_sessions_telemetry(),
        log=structlog.get_logger("test.composer.phase3.sessions"),
    )


@pytest.fixture
def composer_service_with_real_sessions(tmp_path: Path) -> ComposerServiceImpl:
    """Return ``ComposerServiceImpl`` wired to a real SQLite sessions service."""

    sessions_service = build_test_sessions_service(data_dir=tmp_path)
    service = ComposerServiceImpl.for_trained_operator(
        catalog=_mock_catalog(),
        settings=_make_settings(tmp_path),
        sessions_service=sessions_service,
    )
    return service


@pytest.fixture
def composer_service_without_sessions_service(tmp_path: Path) -> ComposerServiceImpl:
    """Return ``ComposerServiceImpl`` without ``sessions_service`` wired."""

    return ComposerServiceImpl.for_trained_operator(catalog=_mock_catalog(), settings=_make_settings(tmp_path))


@pytest.fixture
def fake_composer_service(composer_service_with_real_sessions: ComposerServiceImpl) -> ComposerServiceImpl:
    """Lightweight composer with protocol-faithful sessions persistence wired."""

    return composer_service_with_real_sessions


@pytest.fixture
def fake_llm_emitting_n_tool_calls() -> Any:
    """Factory for an LLM whose first assistant response emits ``n`` tool calls."""

    def _factory(n: int) -> _FakeComposeLLM:
        calls = tuple({"id": f"call_{idx}", "name": "get_pipeline_state", "arguments": {}} for idx in range(n))
        return _FakeComposeLLM((_fake_llm_response(tool_calls=calls), _fake_llm_response(content="Done.")))

    return _factory


@pytest.fixture
def fake_llm_two_tool_calls(fake_llm_emitting_n_tool_calls: Any) -> _FakeComposeLLM:
    """Fake LLM for exactly two successful tool calls."""

    return fake_llm_emitting_n_tool_calls(2)


@pytest.fixture
def fake_llm_one_set_pipeline_tool_call(tmp_path: Path) -> _FakeComposeLLM:
    """Fake LLM that proposes one valid full-pipeline replacement."""

    input_path = tmp_path / "blobs" / "input.csv"
    input_path.parent.mkdir(parents=True, exist_ok=True)
    input_path.write_text("value\n1\n", encoding="utf-8")

    return _FakeComposeLLM(
        (
            _fake_llm_response(
                tool_calls=(
                    {
                        "id": "call_set_pipeline",
                        "name": "set_pipeline",
                        "arguments": {
                            "source": {
                                "plugin": "csv",
                                "on_success": "source_out",
                                "options": {"path": str(input_path), "schema": {"mode": "observed"}},
                                "on_validation_failure": "quarantine",
                            },
                            "nodes": [
                                {
                                    "id": "t1",
                                    "node_type": "transform",
                                    "plugin": "passthrough",
                                    "input": "source_out",
                                    "on_success": "main",
                                    "on_error": "discard",
                                    "options": {"schema": {"mode": "observed"}},
                                }
                            ],
                            "edges": [
                                {
                                    "id": "e1",
                                    "from_node": "source",
                                    "to_node": "t1",
                                    "edge_type": "on_success",
                                    "label": None,
                                }
                            ],
                            "outputs": [
                                {
                                    "sink_name": "main",
                                    "plugin": "csv",
                                    "options": {
                                        "path": str(tmp_path / "outputs" / "output.csv"),
                                        "schema": {"mode": "observed"},
                                        "mode": "write",
                                        "collision_policy": "auto_increment",
                                    },
                                    "on_write_failure": "discard",
                                }
                            ],
                            "metadata": {"name": "proposal-test"},
                        },
                    },
                )
            ),
            _fake_llm_response(content="Done."),
        )
    )


@pytest.fixture
def fake_llm_create_blob_then_set_pipeline(tmp_path: Path) -> _FakeComposeLLM:
    """Fake LLM that emits a create_blob proposal followed by a set_pipeline.

    Reproduces the live-staging failure shape from session
    986fabf6-a723-4eb3-84de-2db1b7ae4e96 (2026-05-14): the agent's turn
    emits both a blob-store side-effect (create_blob) and the composition
    mutation that references its content (set_pipeline). Under
    trust_mode="explicit_approve" the previous behaviour intercepted both
    as proposals, but the create_blob proposal could never be accepted
    (the accept endpoint requires CompositionState.version to advance,
    which create_blob does not).

    The corrected behaviour: create_blob executes immediately (blob is
    written to the session store, a fresh UUID is allocated); the independently
    valid set_pipeline becomes a pending proposal awaiting operator approval.
    """

    input_path = tmp_path / "blobs" / "agency_urls.csv"
    input_path.parent.mkdir(parents=True, exist_ok=True)
    input_path.write_text(
        "url\nhttps://www.example.gov\nhttps://www.example2.gov\n",
        encoding="utf-8",
    )
    output_path = tmp_path / "outputs" / "review.csv"

    return _FakeComposeLLM(
        (
            _fake_llm_response(
                tool_calls=(
                    {
                        "id": "call_create_blob_urls",
                        "name": "create_blob",
                        "arguments": {
                            "filename": "agency_urls.txt",
                            "mime_type": "text/plain",
                            "content": "https://www.example.gov\nhttps://www.example2.gov\n",
                            "description": "Five agency URLs for review",
                        },
                    },
                    {
                        "id": "call_set_pipeline_with_blob",
                        "name": "set_pipeline",
                        "arguments": {
                            "source": {
                                "plugin": "csv",
                                "on_success": "url_rows",
                                "options": {
                                    "path": str(input_path),
                                    "schema": {"mode": "observed"},
                                },
                                "on_validation_failure": "discard",
                            },
                            "nodes": [
                                {
                                    "id": "t1",
                                    "node_type": "transform",
                                    "plugin": "passthrough",
                                    "input": "url_rows",
                                    "on_success": "main",
                                    "on_error": "discard",
                                    "options": {"schema": {"mode": "observed"}},
                                }
                            ],
                            "edges": [
                                {
                                    "id": "e1",
                                    "from_node": "source",
                                    "to_node": "t1",
                                    "edge_type": "on_success",
                                    "label": None,
                                }
                            ],
                            "outputs": [
                                {
                                    "sink_name": "main",
                                    "plugin": "csv",
                                    "options": {
                                        "path": str(output_path),
                                        "schema": {"mode": "observed"},
                                        "mode": "write",
                                        "collision_policy": "auto_increment",
                                    },
                                    "on_write_failure": "discard",
                                }
                            ],
                            "metadata": {"name": "blob-then-pipeline-test"},
                        },
                    },
                )
            ),
            _fake_llm_response(content="Blob created; pipeline proposal queued for approval."),
        )
    )


@pytest.fixture
def fake_llm_set_pipeline_with_misplaced_schema(tmp_path: Path) -> _FakeComposeLLM:
    """Fake LLM that emits a structurally-invalid set_pipeline.

    Reproduces the live failure shape from staging session
    100dc5cb-fd66-400b-8041-a1c165cbd8bd (2026-05-14): the LLM placed
    ``schema`` directly on the node body instead of inside ``options``.
    The redaction MANIFEST's ``SetPipelineArgumentsModel`` declares
    ``nodes[*]`` with ``extra="forbid"``, so the model rejects this
    shape with a Pydantic ValidationError during the proposal-redaction
    step under ``trust_mode == "explicit_approve"``.

    The fixture's second response is the model's followup after seeing
    the ToolArgumentError — a final text turn closes the compose loop.
    """

    input_path = tmp_path / "blobs" / "input.csv"
    input_path.parent.mkdir(parents=True, exist_ok=True)
    input_path.write_text("value\n1\n", encoding="utf-8")

    return _FakeComposeLLM(
        (
            _fake_llm_response(
                tool_calls=(
                    {
                        "id": "call_set_pipeline_invalid",
                        "name": "set_pipeline",
                        "arguments": {
                            "source": {
                                "plugin": "csv",
                                "on_success": "source_out",
                                "options": {"path": str(input_path), "schema": {"mode": "observed"}},
                                "on_validation_failure": "quarantine",
                            },
                            "nodes": [
                                {
                                    "id": "t1",
                                    "node_type": "transform",
                                    "plugin": "passthrough",
                                    "input": "source_out",
                                    "on_success": "main",
                                    "on_error": "discard",
                                    # BUG: schema placed at node body level instead of inside options.
                                    # The redaction MANIFEST's argument_model forbids extra fields.
                                    "schema": {"mode": "fixed", "fields": ["url: str"]},
                                    "options": {},
                                }
                            ],
                            "edges": [
                                {
                                    "id": "e1",
                                    "from_node": "source",
                                    "to_node": "t1",
                                    "edge_type": "on_success",
                                    "label": None,
                                }
                            ],
                            "outputs": [
                                {
                                    "sink_name": "main",
                                    "plugin": "csv",
                                    "options": {
                                        "path": str(tmp_path / "outputs" / "output.csv"),
                                        "schema": {"mode": "observed"},
                                        "mode": "write",
                                        "collision_policy": "auto_increment",
                                    },
                                    "on_write_failure": "discard",
                                }
                            ],
                            "metadata": {"name": "regression-misplaced-schema"},
                        },
                    },
                )
            ),
            _fake_llm_response(content="I made a mistake with the schema field placement; I will retry."),
        )
    )


@pytest.fixture
def fake_llm_three_tool_calls(fake_llm_emitting_n_tool_calls: Any) -> _FakeComposeLLM:
    """Fake LLM for exactly three successful tool calls."""

    return fake_llm_emitting_n_tool_calls(3)


@pytest.fixture
def fake_llm_tool_argument_error_on_second(monkeypatch: pytest.MonkeyPatch) -> _FakeComposeLLM:
    """Second tool raises ``ToolArgumentError`` while the loop continues."""

    original_execute_tool = tools_module.execute_tool
    calls = 0

    def _execute(tool_name: str, *args: Any, **kwargs: Any) -> Any:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise ToolArgumentError(argument="phase3", expected="valid fixture input", actual_type="invalid")
        return original_execute_tool(tool_name, *args, **kwargs)

    monkeypatch.setattr("elspeth.web.composer.tool_batch.execute_tool", _execute)
    return _FakeComposeLLM(
        (
            _fake_llm_response(
                tool_calls=(
                    {"id": "call_ok", "name": "set_metadata", "arguments": {"patch": {"name": "ok"}}},
                    {"id": "call_bad", "name": "set_metadata", "arguments": {"patch": {"name": "bad"}}},
                    {"id": "call_ok_after", "name": "set_metadata", "arguments": {"patch": {"name": "after"}}},
                )
            ),
            _fake_llm_response(content="Done."),
        )
    )


@pytest.fixture
def fake_llm_assertion_error_on_second(monkeypatch: pytest.MonkeyPatch) -> _FakeComposeLLM:
    """Second tool raises ``AssertionError`` through the production dispatch seam."""

    original_execute_tool = tools_module.execute_tool
    calls = 0

    def _execute(tool_name: str, *args: Any, **kwargs: Any) -> Any:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise AssertionError("phase3 synthetic invariant")
        return original_execute_tool(tool_name, *args, **kwargs)

    monkeypatch.setattr("elspeth.web.composer.tool_batch.execute_tool", _execute)
    return _FakeComposeLLM(
        (
            _fake_llm_response(
                tool_calls=(
                    {"id": "call_ok", "name": "set_metadata", "arguments": {"patch": {"name": "ok"}}},
                    {"id": "call_assert", "name": "set_metadata", "arguments": {"patch": {"name": "assert"}}},
                    {"id": "call_skipped", "name": "set_metadata", "arguments": {"patch": {"name": "skipped"}}},
                )
            ),
        )
    )


@pytest.fixture
def fake_llm_runtime_error_on_second(monkeypatch: pytest.MonkeyPatch) -> _FakeComposeLLM:
    """Second tool raises ``RuntimeError`` through the production dispatch seam."""

    original_execute_tool = tools_module.execute_tool
    calls = 0

    def _execute(tool_name: str, *args: Any, **kwargs: Any) -> Any:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("phase3 synthetic runtime error")
        return original_execute_tool(tool_name, *args, **kwargs)

    monkeypatch.setattr("elspeth.web.composer.tool_batch.execute_tool", _execute)
    return _FakeComposeLLM(
        (
            _fake_llm_response(
                tool_calls=(
                    {"id": "call_ok", "name": "set_metadata", "arguments": {"patch": {"name": "ok"}}},
                    {"id": "call_crash", "name": "set_metadata", "arguments": {"patch": {"name": "crash"}}},
                    {"id": "call_skipped", "name": "set_metadata", "arguments": {"patch": {"name": "skipped"}}},
                )
            ),
        )
    )


@pytest.fixture
def fake_llm_with_sensitive_tool_call() -> _FakeComposeLLM:
    """Emits a tool call with arguments covered by Phase 2 manifest redaction."""

    return _FakeComposeLLM(
        (
            _fake_llm_response(
                tool_calls=(
                    {
                        "id": "call_sensitive",
                        "name": "create_blob",
                        "arguments": {"filename": "secret.txt", "mime_type": "text/plain", "content": "top-secret"},
                    },
                )
            ),
            _fake_llm_response(content="Done."),
        )
    )


@pytest.fixture
def fake_llm_summarizer_active() -> _FakeComposeLLM:
    """Emits a response shape that exercises ``redact_tool_call_response``."""

    return _FakeComposeLLM(
        (
            _fake_llm_response(
                tool_calls=(
                    {
                        "id": "call_summarizer",
                        "name": "create_blob",
                        "arguments": {"filename": "summary.txt", "mime_type": "text/plain", "content": "summarize me"},
                    },
                )
            ),
            _fake_llm_response(content="Done."),
        )
    )


@pytest.fixture
def fake_llm_preflight_rewrites_content() -> _FakeComposeLLM:
    """Exposes original text for runtime-preflight rewrite assertions."""

    llm = _FakeComposeLLM((_fake_llm_response(content="The pipeline is ready."),))
    llm.original_text = "The pipeline is ready."
    return llm


@pytest.fixture
def fake_llm_tool_call_with_no_content() -> _FakeComposeLLM:
    """Assistant message has ``content=None`` for raw-content NULL assertions."""

    return _FakeComposeLLM(
        (
            _fake_llm_response(
                content=None,
                tool_calls=({"id": "call_state", "name": "get_pipeline_state", "arguments": {}},),
            ),
            _fake_llm_response(content="Done."),
        )
    )


@pytest.fixture
def sqlalchemy_event_listener() -> Any:
    """Return a helper that counts begin/commit/rollback events on an engine."""

    def _install(engine: Engine) -> dict[str, int]:
        counts = {"begin": 0, "commit": 0, "rollback": 0}

        def _inc(name: str) -> None:
            counts[name] += 1

        event.listen(engine, "begin", lambda _conn: _inc("begin"))
        event.listen(engine, "commit", lambda _conn: _inc("commit"))
        event.listen(engine, "rollback", lambda _conn: _inc("rollback"))
        return counts

    return _install


@pytest.fixture
def add_message_spy(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Record caller frames for legacy ``SessionServiceImpl.add_message`` calls."""

    calls: list[str] = []
    original = SessionServiceImpl.add_message

    async def _spy(self: SessionServiceImpl, *args: Any, **kwargs: Any) -> Any:
        import inspect

        frame = inspect.stack()[1]
        calls.append(f"{frame.filename}:{frame.function}")
        return await original(self, *args, **kwargs)

    monkeypatch.setattr(SessionServiceImpl, "add_message", _spy)
    return calls


@pytest.fixture
def inject_commit_OperationalError() -> Any:
    """Install a one-shot SQLAlchemy COMMIT failure hook."""

    def _install(engine: Engine) -> None:
        def _raise(_conn: Any) -> None:
            event.remove(engine, "commit", _raise)
            raise OperationalError("COMMIT", {}, RuntimeError("phase3 commit failure"))

        event.listen(engine, "commit", _raise)

    return _install


@pytest.fixture
def inject_IntegrityError_on_chat_messages(monkeypatch: pytest.MonkeyPatch) -> None:
    """Raise ``IntegrityError`` when assistant chat-message insert is attempted."""

    original_insert = SessionServiceImpl._insert_chat_message

    def _raise_for_chat_messages(self: SessionServiceImpl, *args: Any, **kwargs: Any) -> Any:
        role = kwargs.get("role")
        if role == "assistant":
            raise IntegrityError("INSERT chat_messages", {}, RuntimeError("phase3 assistant insert failure"))
        return original_insert(self, *args, **kwargs)

    monkeypatch.setattr(SessionServiceImpl, "_insert_chat_message", _raise_for_chat_messages)
