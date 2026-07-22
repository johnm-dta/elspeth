"""Tests for ``composer_progress.schemas_loaded / schemas_referenced / schemas_gap`` telemetry.

Convergence aid for the LLM: the per-turn system context surfaces which
plugins have already been introspected via ``get_plugin_schema`` and
which still need to be looked up before the model can safely construct
options. Backs the staging session 47cfbb5e thrash where the model
called ``set_pipeline`` for a 4-plugin pipeline without preloading any
schema, taking 13 tool calls / 18 LLM rounds to converge.

Tracker lives on ``ComposerServiceImpl`` as
``_schemas_loaded_by_session: dict[session_id, set[(kind, plugin)]]``;
prompts.py reads it through the ``schemas_loaded`` kwarg on
``build_context_string`` / ``build_messages``. This module exercises the
pure prompt-building surface — service-level wiring is exercised by the
compose-loop test suite.
"""

from __future__ import annotations

import json
from typing import Any

from elspeth.web.catalog.policy_view import PolicyCatalogView
from elspeth.web.catalog.protocol import CatalogService, PluginKind
from elspeth.web.catalog.schemas import PluginSchemaInfo, PluginSummary
from elspeth.web.composer.prompts import build_context_string as _build_context_string
from elspeth.web.composer.state import (
    CompositionState,
    PipelineMetadata,
)
from elspeth.web.plugin_policy.models import PluginAvailabilitySnapshot


class _StubCatalog:
    """Minimal CatalogService conforming stub for prompt-rendering tests."""

    def list_sources(self) -> list[PluginSummary]:
        return [
            PluginSummary(name="csv", description="CSV source", plugin_type="source", config_fields=[]),
        ]

    def list_transforms(self) -> list[PluginSummary]:
        return [
            PluginSummary(name="web_scrape", description="Web scrape", plugin_type="transform", config_fields=[]),
            PluginSummary(name="openrouter_llm", description="LLM via OpenRouter", plugin_type="transform", config_fields=[]),
        ]

    def list_sinks(self) -> list[PluginSummary]:
        return [
            PluginSummary(name="json", description="JSON sink", plugin_type="sink", config_fields=[]),
        ]

    def get_schema(self, plugin_type: PluginKind, name: str) -> PluginSchemaInfo:
        raise ValueError(f"Not implemented for stub: {plugin_type}/{name}")


def _stub_catalog() -> CatalogService:
    catalog: CatalogService = _StubCatalog()
    return catalog


def build_context_string(
    state: CompositionState,
    catalog: CatalogService,
    **kwargs: Any,
) -> str:
    """Exercise the prompt builder through its explicit policy pair."""
    snapshot = PluginAvailabilitySnapshot.for_trained_operator(catalog)
    return _build_context_string(
        state,
        PolicyCatalogView.for_trained_operator(catalog, snapshot),
        plugin_snapshot=snapshot,
        **kwargs,
    )


def _empty_state() -> CompositionState:
    return CompositionState(
        source=None,
        nodes=(),
        edges=(),
        outputs=(),
        metadata=PipelineMetadata(),
        version=1,
    )


def _three_plugin_state() -> CompositionState:
    """State referencing source=csv, transform=web_scrape, sink=json."""
    return CompositionState.from_dict(
        {
            "source": {
                "plugin": "csv",
                "on_success": "rows",
                "options": {"path": "/data/blobs/in.csv", "schema": {"mode": "observed"}},
                "on_validation_failure": "discard",
            },
            "nodes": [
                {
                    "id": "scrape",
                    "node_type": "transform",
                    "plugin": "web_scrape",
                    "input": "rows",
                    "on_success": "main",
                    "on_error": "discard",
                    "options": {"schema": {"mode": "observed"}},
                }
            ],
            "edges": [],
            "outputs": [
                {
                    "name": "main",
                    "plugin": "json",
                    "options": {"path": "/data/outputs/out.json", "schema": {"mode": "observed"}},
                    "on_write_failure": "discard",
                }
            ],
            "metadata": {"name": "test", "description": ""},
            "version": 2,
        }
    )


def _composer_progress(context_str: str) -> dict[str, Any]:
    """Extract the ``composer_progress`` block from a context string payload."""
    prefix, _, json_text = context_str.partition("\n")
    assert prefix.startswith("Current pipeline state and available plugins")
    parsed = json.loads(json_text)
    return parsed["composer_progress"]


class TestSchemasGapTelemetry:
    def test_schemas_loaded_starts_empty(self) -> None:
        """No get_plugin_schema calls yet — explicit empty frozenset means
        "tracked, nothing loaded" and produces empty lists (the LLM's
        signal to discover plugins, not permission to mutate an empty
        pipeline). Distinct from the
        ``_SCHEMAS_LOADED_UNSET`` sentinel reading exercised below.
        """
        progress = _composer_progress(build_context_string(_empty_state(), _stub_catalog(), schemas_loaded=frozenset()))
        assert progress["schemas_loaded_this_session"] == []
        assert progress["schemas_referenced_by_state"] == []
        assert progress["schemas_gap"] == []
        assert progress["schema_inventory_precondition"] == "discover planned plugin schemas before first mutation"

    def test_schemas_loaded_reflects_passed_set(self) -> None:
        """A loaded set passed to build_context_string surfaces in the payload."""
        loaded = frozenset({("transform", "openrouter_llm")})
        progress = _composer_progress(build_context_string(_empty_state(), _stub_catalog(), schemas_loaded=loaded))
        assert progress["schemas_loaded_this_session"] == ["transform/openrouter_llm"]

    def test_schemas_referenced_reflects_state_plugins(self) -> None:
        """source=csv, transform=web_scrape, sink=json → three referenced pairs sorted."""
        progress = _composer_progress(build_context_string(_three_plugin_state(), _stub_catalog()))
        assert progress["schemas_referenced_by_state"] == [
            "sink/json",
            "source/csv",
            "transform/web_scrape",
        ]

    def test_schemas_gap_is_referenced_minus_loaded(self) -> None:
        """When only csv is loaded, web_scrape and json remain in the gap."""
        loaded = frozenset({("source", "csv")})
        progress = _composer_progress(build_context_string(_three_plugin_state(), _stub_catalog(), schemas_loaded=loaded))
        assert progress["schemas_loaded_this_session"] == ["source/csv"]
        assert progress["schemas_referenced_by_state"] == [
            "sink/json",
            "source/csv",
            "transform/web_scrape",
        ]
        assert progress["schemas_gap"] == ["sink/json", "transform/web_scrape"]

    def test_schemas_gap_empty_after_all_referenced_plugins_loaded(self) -> None:
        """Once every referenced plugin has been schema-loaded, the gap is empty."""
        loaded = frozenset(
            {
                ("source", "csv"),
                ("transform", "web_scrape"),
                ("sink", "json"),
            }
        )
        progress = _composer_progress(build_context_string(_three_plugin_state(), _stub_catalog(), schemas_loaded=loaded))
        assert progress["schemas_gap"] == []
        assert progress["schema_inventory_precondition"] == "satisfied for current referenced state"

    def test_extra_loaded_pairs_do_not_affect_referenced_or_gap(self) -> None:
        """Plugins loaded but not referenced are still surfaced under
        schemas_loaded_this_session — they do NOT shrink the gap below
        the (referenced minus loaded) calculation when they aren't
        referenced."""
        loaded = frozenset(
            {
                ("source", "csv"),  # referenced
                ("transform", "openrouter_llm"),  # NOT referenced by state
            }
        )
        progress = _composer_progress(build_context_string(_three_plugin_state(), _stub_catalog(), schemas_loaded=loaded))
        assert progress["schemas_loaded_this_session"] == [
            "source/csv",
            "transform/openrouter_llm",
        ]
        assert progress["schemas_gap"] == ["sink/json", "transform/web_scrape"]

    def test_gate_nodes_with_null_plugin_do_not_contribute_to_referenced(self) -> None:
        """Gate / coalesce nodes have plugin=None; they have no plugin-options
        schema and therefore must NOT appear in schemas_referenced_by_state."""
        state = CompositionState.from_dict(
            {
                "source": {
                    "plugin": "csv",
                    "on_success": "rows",
                    "options": {"path": "/data/blobs/in.csv", "schema": {"mode": "observed"}},
                    "on_validation_failure": "discard",
                },
                "nodes": [
                    {
                        "id": "g1",
                        "node_type": "gate",
                        "plugin": None,
                        "input": "rows",
                        "on_success": None,
                        "on_error": None,
                        "options": {},
                        "condition": "row['x'] > 0",
                        "routes": {"main": "row['x'] > 0"},
                    }
                ],
                "edges": [],
                "outputs": [
                    {
                        "name": "main",
                        "plugin": "json",
                        "options": {"path": "/data/outputs/out.json", "schema": {"mode": "observed"}},
                        "on_write_failure": "discard",
                    }
                ],
                "metadata": {"name": "test", "description": ""},
                "version": 2,
            }
        )
        progress = _composer_progress(build_context_string(state, _stub_catalog()))
        # source and sink only — the gate contributes nothing.
        assert progress["schemas_referenced_by_state"] == ["sink/json", "source/csv"]


class TestSchemasLoadedUnsetSentinel:
    """The unset-sentinel default surfaces a distinct ``composer_progress``
    marker so a service-side regression (caller stops threading the
    ``schemas_loaded`` kwarg) is observable in every audited turn rather
    than masquerading as "tracked, nothing loaded yet".

    Two adjacent regressions to pin:

    1. The default value is identity-tested by call sites — using the
       sentinel constant directly (rather than constructing a new
       ``frozenset`` with the same poisoned pair on each call) means the
       ``is`` check inside ``build_context_string`` triggers.
    2. ``frozenset()`` passed explicitly continues to mean "tracked,
       nothing loaded" — the sentinel branch must not fire on a real
       empty frozenset.
    """

    def test_sentinel_default_emits_distinct_loaded_marker(self) -> None:
        """No ``schemas_loaded`` kwarg → ``schemas_loaded_this_session``
        renders the ``<schemas-loaded-tracker-not-threaded:loaded>``
        marker, distinguishable both from the legitimate ``[]`` produced
        by ``frozenset()`` (the "tracked, nothing loaded" reading) AND
        from the ``:gap``-suffixed sibling marker (so an auditor reading
        the dump can identify which view tripped).
        """
        progress = _composer_progress(build_context_string(_empty_state(), _stub_catalog()))
        assert progress["schemas_loaded_this_session"] == ["<schemas-loaded-tracker-not-threaded:loaded>"]
        assert progress["schemas_gap"] == ["<schemas-loaded-tracker-not-threaded:gap>"]

    def test_sentinel_default_emits_distinct_gap_marker_even_with_referenced_plugins(self) -> None:
        """The sentinel branch overrides the normal ``referenced - loaded``
        gap calculation; the gap field carries the marker rather than
        the (potentially misleading) ``referenced`` set so a tracker
        regression cannot masquerade as "the model has discovered
        nothing yet"."""
        progress = _composer_progress(build_context_string(_three_plugin_state(), _stub_catalog()))
        assert progress["schemas_loaded_this_session"] == ["<schemas-loaded-tracker-not-threaded:loaded>"]
        assert progress["schemas_gap"] == ["<schemas-loaded-tracker-not-threaded:gap>"]
        # ``schemas_referenced_by_state`` is computed independently from
        # ``state`` and must NOT carry the sentinel marker — the referenced
        # view is a fact about state, not about the tracker.
        assert progress["schemas_referenced_by_state"] == [
            "sink/json",
            "source/csv",
            "transform/web_scrape",
        ]

    def test_explicit_empty_frozenset_is_distinguishable_from_sentinel(self) -> None:
        """Passing ``frozenset()`` explicitly produces empty-list readings
        — the "I tracked, nothing has loaded yet" signal. Pins the
        contract that the sentinel branch fires on identity, not on
        emptiness."""
        progress = _composer_progress(build_context_string(_empty_state(), _stub_catalog(), schemas_loaded=frozenset()))
        assert progress["schemas_loaded_this_session"] == []
        assert progress["schemas_gap"] == []

    def test_loaded_and_gap_unset_markers_are_distinct(self) -> None:
        """The two sentinel markers must be byte-distinct so an auditor
        reading a recorded ``composer_progress`` dump can tell which
        view tripped (loaded-vs-gap field-level fault locality).
        Collapsing the two to a single string would let a tracker
        regression on one view masquerade as a regression on the other.
        """
        from elspeth.web.composer.prompts import (
            _SCHEMAS_GAP_UNSET_MARKER,
            _SCHEMAS_LOADED_UNSET_MARKER,
        )

        assert _SCHEMAS_LOADED_UNSET_MARKER != _SCHEMAS_GAP_UNSET_MARKER
        assert _SCHEMAS_LOADED_UNSET_MARKER.endswith(":loaded>")
        assert _SCHEMAS_GAP_UNSET_MARKER.endswith(":gap>")

    def test_sentinel_constant_identity_is_stable_across_calls(self) -> None:
        """The default-value expression evaluates once at function-def
        time. The sentinel must be module-level so repeated calls see
        the same object (``is`` check), not a fresh frozenset on every
        call site."""
        from elspeth.web.composer.prompts import _SCHEMAS_LOADED_UNSET

        first = _SCHEMAS_LOADED_UNSET
        second = _SCHEMAS_LOADED_UNSET
        assert first is second
        # The sentinel's poisoned pair cannot collide with a real
        # ``(kind, plugin)`` — guards against a future refactor that
        # accidentally normalises the sentinel into a "valid" set.
        assert ("__elspeth_internal__", "__sentinel_schemas_loaded_unset__") in first


class TestComposerServiceTracker:
    """Verify the per-session tracker on ``ComposerServiceImpl``.

    The compose-loop integration is exercised separately; these tests
    cover the accessor/marker contract in isolation so a regression in
    the tracker shape is caught before reaching the dispatch layer.
    """

    def _make_service_with_tracker(self) -> Any:
        """Construct a bare service with only the tracker fields exercised.

        Bypasses the full ``__init__`` because the constructor wires
        catalog, sessions service, and skill-hash gates that are
        unrelated to the tracker. The tracker contract is a tiny,
        independent surface; isolating it keeps these tests fast and
        focused.
        """
        from elspeth.web.composer.service import ComposerServiceImpl

        service = object.__new__(ComposerServiceImpl)
        service._schemas_loaded_by_session = {}  # type: ignore[attr-defined]
        return service

    def test_get_returns_empty_frozenset_for_unseen_session(self) -> None:
        service = self._make_service_with_tracker()
        assert service._schemas_loaded_for_session("session-A") == frozenset()

    def test_get_returns_empty_frozenset_for_none_session(self) -> None:
        service = self._make_service_with_tracker()
        assert service._schemas_loaded_for_session(None) == frozenset()

    def test_mark_then_get_round_trips(self) -> None:
        service = self._make_service_with_tracker()
        service._mark_plugin_schema_loaded("session-A", "transform", "openrouter_llm")
        assert service._schemas_loaded_for_session("session-A") == frozenset({("transform", "openrouter_llm")})

    def test_mark_is_session_scoped(self) -> None:
        service = self._make_service_with_tracker()
        service._mark_plugin_schema_loaded("session-A", "source", "csv")
        service._mark_plugin_schema_loaded("session-B", "sink", "json")
        assert service._schemas_loaded_for_session("session-A") == frozenset({("source", "csv")})
        assert service._schemas_loaded_for_session("session-B") == frozenset({("sink", "json")})

    def test_mark_with_none_session_is_noop(self) -> None:
        """Unsaved sessions have no persistent identity for the tracker;
        marking a None session_id must not silently keep state under a
        sentinel key that could collide with a future real session."""
        service = self._make_service_with_tracker()
        service._mark_plugin_schema_loaded(None, "source", "csv")
        assert service._schemas_loaded_by_session == {}

    def test_get_returns_snapshot_not_live_view(self) -> None:
        """A subsequent mark must not mutate a previously-returned frozenset."""
        service = self._make_service_with_tracker()
        service._mark_plugin_schema_loaded("session-A", "source", "csv")
        snapshot = service._schemas_loaded_for_session("session-A")
        service._mark_plugin_schema_loaded("session-A", "sink", "json")
        # snapshot remains as it was at the moment of read.
        assert snapshot == frozenset({("source", "csv")})


# NOTE: ``TestGetPluginSchemaMarksLoadedContract`` was deleted as part of
# elspeth-59cdfcaf67. Its two tests re-implemented the service guard
# (``if result.success:``) in the test body before asserting the marker
# state — pinning the test's own filter, not the service's. The contract
# it claimed to cover is decomposed across:
#
# - ``TestComposerServiceTracker`` above (the marker semantics in
#   isolation).
# - ``tests/unit/web/composer/test_tools.py::TestToolDispatch`` and
#   ``TestSetSourceTool`` (covers ``execute_tool("get_plugin_schema", ...)``
#   producing ``result.success is True`` on success and ``False`` on a
#   catalog-side ``ValueError`` — see ``test_get_plugin_schema_delegates``
#   and ``test_unknown_plugin_fails``).
#
# The single line that genuinely needed coverage —
# ``service.py`` line 3317's ``if tool_name == "get_plugin_schema" and
# result.success:`` — is the compose-loop dispatch site exercised by
# ``test_compose_loop_audit_wiring.py``'s integration tests, not a
# unit-testable surface on the marker.
