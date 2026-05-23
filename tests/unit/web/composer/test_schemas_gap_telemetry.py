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

from elspeth.web.catalog.protocol import CatalogService, PluginKind
from elspeth.web.catalog.schemas import PluginSchemaInfo, PluginSummary
from elspeth.web.composer.prompts import build_context_string
from elspeth.web.composer.state import (
    CompositionState,
    PipelineMetadata,
)


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
    assert prefix.startswith("Current pipeline state and available plugins:")
    parsed = json.loads(json_text)
    return parsed["composer_progress"]


class TestSchemasGapTelemetry:
    def test_schemas_loaded_starts_empty(self) -> None:
        """No get_plugin_schema calls yet — the loaded list is empty."""
        progress = _composer_progress(build_context_string(_empty_state(), _stub_catalog()))
        assert progress["schemas_loaded_this_session"] == []
        assert progress["schemas_referenced_by_state"] == []
        assert progress["schemas_gap"] == []

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


class TestGetPluginSchemaMarksLoadedContract:
    """Verify the dispatch-side contract feeding ``_mark_plugin_schema_loaded``.

    The compose loop body (see ``service.py``) guards the marking with::

        if tool_name == "get_plugin_schema" and result.success:
            self._mark_plugin_schema_loaded(session_id, plugin_type, plugin_name)

    Two preconditions therefore need coverage at the unit level:

    1. A *successful* ``execute_tool("get_plugin_schema", ...)`` produces
       ``result.success is True`` (so the guard passes and marking happens
       upstream).
    2. A *failed* ``execute_tool("get_plugin_schema", ...)`` produces
       ``result.success is False`` (so the guard rejects and marking is
       skipped, even though the tool name matches).

    A future refactor that drops the ``result.success`` check would slip
    past the dispatch-level tests; this contract pair pins the behaviour
    the service relies on.
    """

    def test_successful_get_plugin_schema_returns_success_true(self) -> None:
        """The marker condition passes when execute_tool's ToolResult is successful."""
        from unittest.mock import MagicMock

        from elspeth.web.catalog.schemas import PluginSchemaInfo
        from elspeth.web.composer.tools import execute_tool

        catalog = MagicMock()
        catalog.list_sources.return_value = []
        catalog.list_transforms.return_value = []
        catalog.list_sinks.return_value = []
        catalog.get_schema.return_value = PluginSchemaInfo(
            name="csv",
            plugin_type="source",
            description="CSV source",
            json_schema={"title": "CsvSourceConfig", "properties": {}},
            knob_schema={"fields": []},
        )

        result = execute_tool(
            "get_plugin_schema",
            {"plugin_type": "source", "name": "csv"},
            _empty_state(),
            catalog,
        )
        assert result.success is True

        # Mirror the service guard exactly. A future regression that
        # forgets the ``result.success`` test would still call the
        # marker here; we want to lock in that the dispatch surface
        # produces the right success flag.
        service = TestComposerServiceTracker._make_service_with_tracker(
            TestComposerServiceTracker()  # type: ignore[arg-type]
        )
        if result.success:
            service._mark_plugin_schema_loaded("session-A", "source", "csv")
        assert service._schemas_loaded_for_session("session-A") == frozenset({("source", "csv")})

    def test_failed_get_plugin_schema_does_not_mark_loaded(self) -> None:
        """A failed schema lookup must NOT cause marking under the service guard."""
        from unittest.mock import MagicMock

        from elspeth.web.composer.tools import execute_tool

        catalog = MagicMock()
        catalog.list_sources.return_value = []
        catalog.list_transforms.return_value = []
        catalog.list_sinks.return_value = []
        catalog.get_schema.side_effect = ValueError("Unknown plugin: ghost")

        result = execute_tool(
            "get_plugin_schema",
            {"plugin_type": "source", "name": "ghost"},
            _empty_state(),
            catalog,
        )
        assert result.success is False

        # Apply the same service guard. The ``result.success`` check is
        # the contract: failed schema lookups must not pollute the
        # tracker. Otherwise the next turn's system context would tell
        # the LLM that ``source/ghost`` has been schema-loaded — wrong
        # signal that would suppress a legitimate retry.
        service = TestComposerServiceTracker._make_service_with_tracker(
            TestComposerServiceTracker()  # type: ignore[arg-type]
        )
        if result.success:
            service._mark_plugin_schema_loaded("session-A", "source", "ghost")
        assert service._schemas_loaded_for_session("session-A") == frozenset()
