"""End-to-end mocked-LLM integration coverage for the convergence-suite scenarios.

Each scenario in ``evals/composer-rgr/scenarios/convergence-suite/`` is a
JSON contract that the live RGR harness uses against a real LLM. Without
mocked-LLM coverage the contracts are checked only by the synthetic-state
sanity tests in ``test_convergence_scenarios.py`` — which proves the green
criteria are jointly satisfiable but does NOT prove the composer's
``_compose_loop`` actually drives a real session through that satisfaction.

These tests close the loop:

  * Real ``ComposerServiceImpl`` with a real session DB and real seeded
    blob (when the scenario uses an inline blob).
  * ``_call_llm`` mocked at the network seam — scripted assistant turns
    drive the conversation through ``_compose_loop`` exactly as the live
    harness would, producing authentic state mutations via real
    ``execute_tool`` calls.
  * After ``compose()`` returns, the resulting state is serialised via
    ``state.to_dict()`` (with ``is_valid`` and ``composer_meta`` added
    matching the API persistence shape) and fed to
    ``evals.lib.composer_rgr_score.score`` against the scenario file.
  * Verdict assertions pin both convergence behaviour (repair-turn count)
    and scoring outcome (GREEN against the scenario contract).

Bugs that would land silently otherwise:

  * The compose loop emits a state shape that diverges from
    ``CompositionState.to_dict()`` — the score function would silently
    AMBER on missing keys.
  * A regression in ``_attempt_proof_repair`` that fails to clear the
    blocking diagnostic before finalisation — the live RGR harness would
    catch this only after burning LLM credits.
  * A scenario whose green criteria are no longer aligned with the proof
    step's blocking diagnostics — the only way to detect drift between
    the scenario contract and the runtime is to drive both together.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest
from evals.lib.composer_rgr_score import score
from sqlalchemy.pool import StaticPool

from elspeth.web.blobs.service import content_hash as _content_hash
from elspeth.web.catalog.protocol import CatalogService
from elspeth.web.catalog.schemas import PluginSchemaInfo, PluginSummary
from elspeth.web.composer.service import ComposerServiceImpl
from elspeth.web.composer.state import CompositionState, PipelineMetadata
from elspeth.web.config import WebSettings
from elspeth.web.execution.schemas import ValidationResult
from elspeth.web.sessions.engine import create_session_engine
from elspeth.web.sessions.models import blobs_table, sessions_table
from elspeth.web.sessions.schema import initialize_session_schema

_REPO_ROOT = Path(__file__).resolve().parents[3]
_SUITE = _REPO_ROOT / "evals" / "composer-rgr" / "scenarios" / "convergence-suite"


# --------------------------------------------------------------------------
# Mock-LLM-response plumbing — same shape as test_service.py uses, kept
# local so this file is independent.
# --------------------------------------------------------------------------


@dataclass
class _FakeFn:
    name: str
    arguments: str


@dataclass
class _FakeTC:
    id: str
    function: _FakeFn


@dataclass
class _FakeMsg:
    content: str | None
    tool_calls: list[_FakeTC] | None


@dataclass
class _FakeChoice:
    message: _FakeMsg


@dataclass
class _FakeLLMResponse:
    choices: list[_FakeChoice]


def _llm_response(content: str | None = None, tool_calls: list[dict[str, Any]] | None = None) -> _FakeLLMResponse:
    fakes: list[_FakeTC] | None = None
    if tool_calls:
        fakes = [_FakeTC(id=tc["id"], function=_FakeFn(name=tc["name"], arguments=json.dumps(tc["arguments"]))) for tc in tool_calls]
    return _FakeLLMResponse(choices=[_FakeChoice(message=_FakeMsg(content=content, tool_calls=fakes))])


# --------------------------------------------------------------------------
# Test infrastructure: real catalog + real session engine + per-scenario blob
# --------------------------------------------------------------------------


def _real_catalog() -> Any:
    """Real PluginManager + CatalogService for authentic schema prevalidation.

    The recipe / proof-step pipelines exercise schema prevalidation on
    csv/llm/type_coerce/json plugins. Mocking the catalog forces fabricated
    schemas; the real catalog is cheap (builtin registration only).
    """
    from elspeth.plugins.infrastructure.manager import PluginManager
    from elspeth.web.catalog.service import CatalogServiceImpl

    pm = PluginManager()
    pm.register_builtin_plugins()
    return CatalogServiceImpl(pm)


def _mock_catalog_for_text() -> Any:
    """Mock catalog for the url-text-smoke scenario.

    The text source plugin is not in the real builtin catalog; we wire a
    minimal mock that satisfies the prevalidation surface area used by
    set_pipeline / set_source_from_blob for text/web_scrape/json.
    """
    catalog = MagicMock(spec=CatalogService)
    catalog.list_sources.return_value = [
        PluginSummary(name="text", description="Text source", plugin_type="source", config_fields=[]),
    ]
    catalog.list_transforms.return_value = [
        PluginSummary(name="web_scrape", description="Web scrape", plugin_type="transform", config_fields=[]),
    ]
    catalog.list_sinks.return_value = [
        PluginSummary(name="json", description="JSON sink", plugin_type="sink", config_fields=[]),
    ]
    catalog.get_schema.return_value = PluginSchemaInfo(
        name="text",
        plugin_type="source",
        description="text source",
        json_schema={"title": "Config", "properties": {}},
    )
    return catalog


def _make_settings(data_dir: Path) -> WebSettings:
    return WebSettings(
        data_dir=data_dir,
        composer_max_composition_turns=15,
        composer_max_discovery_turns=10,
        composer_timeout_seconds=85.0,
        composer_rate_limit_per_minute=10,
    )


def _session_engine() -> tuple[Any, str]:
    engine = create_session_engine(
        "sqlite:///:memory:",
        poolclass=StaticPool,
        connect_args={"check_same_thread": False},
    )
    initialize_session_schema(engine)
    session_id = str(uuid4())
    now = datetime.now(UTC)
    with engine.begin() as conn:
        conn.execute(
            sessions_table.insert().values(
                id=session_id,
                user_id="test-user",
                auth_provider_type="local",
                title="Convergence Test",
                created_at=now,
                updated_at=now,
            )
        )
    return engine, session_id


def _seed_blob(engine: Any, session_id: str, *, body: bytes, filename: str, mime_type: str, storage_dir: Path) -> str:
    blob_id = str(uuid4())
    storage_dir.mkdir(parents=True, exist_ok=True)
    storage_path = storage_dir / f"{blob_id}_{filename}"
    storage_path.write_bytes(body)
    with engine.begin() as conn:
        conn.execute(
            blobs_table.insert().values(
                id=blob_id,
                session_id=session_id,
                filename=filename,
                mime_type=mime_type,
                size_bytes=len(body),
                content_hash=_content_hash(body),
                storage_path=str(storage_path),
                created_at=datetime.now(UTC),
                created_by="user",
                source_description=None,
                status="ready",
            )
        )
    return blob_id


def _empty_state() -> CompositionState:
    return CompositionState(
        source=None,
        nodes=(),
        edges=(),
        outputs=(),
        metadata=PipelineMetadata(),
        version=1,
    )


def _composer_meta_for(result: Any) -> dict[str, Any]:
    """Build the composer_meta dict the score function consumes.

    Mirrors what ``_state_data_from_composer_state`` writes when the API
    persists the compose result (see web/sessions/routes.py).
    """
    return {"repair_turns_used": result.repair_turns_used}


def _state_dict_for_scoring(result: Any) -> dict[str, Any]:
    """Build the score-function input dict from a ComposerComposeResult.

    The score function consumes ``state.get('is_valid')``,
    ``state.get('source')``, ``state.get('nodes')``, ``state.get('outputs')``,
    and ``state.get('composer_meta')``. ``CompositionState.to_dict()`` covers
    source/nodes/outputs but does not include is_valid (a runtime
    determination from preflight) or composer_meta (from the result wrapper).
    Compose them here so the round-trip mirrors API persistence.
    """
    state_dict = result.state.to_dict()
    state_dict["is_valid"] = bool(result.runtime_preflight is not None and result.runtime_preflight.is_valid)
    state_dict["composer_meta"] = _composer_meta_for(result)
    return state_dict


def _load_scenario(name: str) -> dict[str, Any]:
    return json.loads((_SUITE / name / "scenario.json").read_text())


@pytest.fixture(autouse=True)
def _composer_available_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """Bypass real availability check (no API key needed in tests)."""
    from elspeth.web.composer.service import ComposerAvailability

    def _available(self: ComposerServiceImpl) -> ComposerAvailability:
        return ComposerAvailability(available=True, model=self._model, provider="test")

    monkeypatch.setattr(ComposerServiceImpl, "_compute_availability", _available)


@pytest.fixture(autouse=True)
def _composer_to_thread_uses_test_worker(monkeypatch: pytest.MonkeyPatch) -> None:
    """Run composer to_thread seams through a deterministic worker."""
    import asyncio

    real_to_thread = asyncio.to_thread

    async def _shim(func: Any, *args: Any, **kwargs: Any) -> Any:
        return await real_to_thread(func, *args, **kwargs)

    monkeypatch.setattr("elspeth.web.composer.service.asyncio.to_thread", _shim)


# --------------------------------------------------------------------------
# Scenario 1: csv-classifier
#
# Drive the LLM through the recipe-equivalent shape directly via tool calls
# (set_pipeline) so the test exercises the same prevalidation/proof path the
# live RGR harness would. Triggers csv_fixed_schema_omits_observed_columns
# on turn 1 by declaring a fixed schema with only one of the five observed
# columns; turn 2 claims completion → repair fires; turn 3 patches the
# source schema to mode=observed; turn 4 claims completion → GREEN.
# --------------------------------------------------------------------------


class TestCsvClassifierScenario:
    """End-to-end mocked-LLM run for the csv-classifier convergence scenario."""

    @pytest.mark.asyncio
    async def test_csv_classifier_converges_with_one_repair_turn(self, tmp_path: Path) -> None:
        engine, session_id = _session_engine()
        # Five observed columns, mirroring the scenario's prompt CSV.
        body = b"ticket_id,customer_name,subject,body,received_at\nT-001,Alice,Issue,desc,2026-05-06\n"
        blob_id = _seed_blob(
            engine,
            session_id,
            body=body,
            filename="tickets.csv",
            mime_type="text/csv",
            storage_dir=tmp_path / "blobs" / session_id,
        )

        catalog = _real_catalog()
        settings = _make_settings(tmp_path)
        service = ComposerServiceImpl(catalog=catalog, settings=settings, session_engine=engine)

        # Turn 1: blocking pipeline — fixed schema declaring only ticket_id,
        # omitting the other four observed columns. on_validation_failure=
        # discard makes it the documented all-row-drop hazard.
        turn1 = _llm_response(
            content=None,
            tool_calls=[
                {
                    "id": "call_set",
                    "name": "set_pipeline",
                    "arguments": {
                        "source": {
                            "plugin": "csv",
                            "blob_id": blob_id,
                            "on_success": "rows",
                            "options": {
                                "schema": {"mode": "fixed", "fields": ["ticket_id: str"]},
                            },
                            "on_validation_failure": "discard",
                        },
                        "nodes": [
                            {
                                "id": "classifier",
                                "node_type": "transform",
                                "plugin": "llm",
                                "input": "rows",
                                "on_success": "labelled",
                                "on_error": "discard",
                                "options": {
                                    "provider": "openrouter",
                                    "model": "anthropic/claude-3.5-sonnet",
                                    "api_key": {"secret_ref": "OPENROUTER_API_KEY"},
                                    "template": "Classify {{ row['subject'] }}",
                                    "response_field": "urgency",
                                    "schema": {"mode": "observed"},
                                    "required_input_fields": ["subject"],
                                },
                            }
                        ],
                        "edges": [],
                        "outputs": [
                            {
                                "sink_name": "labelled",
                                "plugin": "json",
                                "options": {
                                    "path": "outputs/labelled.jsonl",
                                    "format": "jsonl",
                                    "schema": {"mode": "observed"},
                                    "collision_policy": "auto_increment",
                                },
                                "on_write_failure": "discard",
                            }
                        ],
                        "metadata": {"name": "csv-classifier"},
                    },
                },
            ],
        )
        # Turn 2: claim completion → proof gate fires
        turn2 = _llm_response(content="All set.", tool_calls=None)
        # Turn 3: repair — switch source schema to observed mode
        turn3 = _llm_response(
            content=None,
            tool_calls=[
                {
                    "id": "call_repair",
                    "name": "patch_source_options",
                    "arguments": {"patch": {"schema": {"mode": "observed"}}},
                },
            ],
        )
        # Turn 4: claim completion (clean)
        turn4 = _llm_response(content="Repaired and ready.", tool_calls=None)

        passing_preflight = ValidationResult(is_valid=True, checks=[], errors=[])
        empty = _empty_state()
        with (
            patch.object(service, "_call_llm", new_callable=AsyncMock) as mock_llm,
            patch.object(service, "_runtime_preflight", return_value=passing_preflight),
        ):
            mock_llm.side_effect = [turn1, turn2, turn3, turn4]
            result = await service.compose(
                "Classify these tickets",
                [],
                empty,
                session_id=session_id,
                user_id="test-user",
            )

        # Convergence behaviour: exactly one forced repair turn.
        assert mock_llm.call_count == 4, f"expected 4 LLM calls, got {mock_llm.call_count}"
        assert result.repair_turns_used == 1, f"expected 1 repair turn, got {result.repair_turns_used}"

        # Score against the scenario file.
        scenario = _load_scenario("csv-classifier")
        assistant_messages = [{"role": "assistant", "content": result.message or ""}]
        state_dict = _state_dict_for_scoring(result)
        verdict = score(scenario, assistant_messages, state_dict)

        assert verdict["verdict"] == "GREEN", (
            f"csv-classifier did not score GREEN. red={verdict['red_reasons']} amber={verdict['amber_reasons']}"
        )


# --------------------------------------------------------------------------
# Scenario 2: numeric-gate
#
# The proof step does NOT emit a 'must_handle_field_as_numeric' diagnostic —
# that criterion is scoring-only. So this scenario does not exercise the
# repair loop. Test drives a single-turn build that satisfies the criterion
# (type_coerce + gate + 2 outputs) and asserts repair_turns_used == 0 and
# GREEN scoring.
# --------------------------------------------------------------------------


class TestNumericGateScenario:
    """End-to-end mocked-LLM run for the numeric-gate convergence scenario.

    No proof-step blocker fires for this scenario, so the test is a
    first-pass-success integration: scripted LLM emits a type_coerce + gate
    + two-sink pipeline directly, scoring should GREEN with zero repair
    turns. If a future change adds a 'must_handle_field_as_numeric' blocker
    to compute_proof_diagnostics, this test should be promoted to a
    forced-repair flow analogous to csv-classifier.
    """

    @pytest.mark.asyncio
    async def test_numeric_gate_first_pass_success(self, tmp_path: Path) -> None:
        engine, session_id = _session_engine()
        body = b"order_id,customer,price,shipped_at\nO-1,Alice,49.95,2026-05-01\nO-2,Bob,150.00,2026-05-02\n"
        blob_id = _seed_blob(
            engine,
            session_id,
            body=body,
            filename="orders.csv",
            mime_type="text/csv",
            storage_dir=tmp_path / "blobs" / session_id,
        )

        catalog = _real_catalog()
        settings = _make_settings(tmp_path)
        service = ComposerServiceImpl(catalog=catalog, settings=settings, session_engine=engine)

        # Single-turn build: type_coerce on price + gate + two outputs.
        turn1 = _llm_response(
            content=None,
            tool_calls=[
                {
                    "id": "call_set",
                    "name": "set_pipeline",
                    "arguments": {
                        "source": {
                            "plugin": "csv",
                            "blob_id": blob_id,
                            "on_success": "rows",
                            "options": {"schema": {"mode": "observed"}},
                            "on_validation_failure": "discard",
                        },
                        "nodes": [
                            {
                                "id": "coerce_numeric",
                                "node_type": "transform",
                                "plugin": "type_coerce",
                                "input": "rows",
                                "on_success": "numeric_rows",
                                "on_error": "discard",
                                "options": {
                                    "schema": {"mode": "observed"},
                                    "conversions": [{"field": "price", "to": "float"}],
                                },
                            },
                            {
                                "id": "threshold_gate",
                                "node_type": "gate",
                                "input": "numeric_rows",
                                "condition": "row['price'] >= 100.0",
                                "routes": {"true": "high", "false": "low"},
                            },
                        ],
                        "edges": [],
                        "outputs": [
                            {
                                "sink_name": "high",
                                "plugin": "json",
                                "options": {
                                    "path": "outputs/high.jsonl",
                                    "format": "jsonl",
                                    "schema": {"mode": "observed"},
                                    "collision_policy": "auto_increment",
                                },
                                "on_write_failure": "discard",
                            },
                            {
                                "sink_name": "low",
                                "plugin": "json",
                                "options": {
                                    "path": "outputs/low.jsonl",
                                    "format": "jsonl",
                                    "schema": {"mode": "observed"},
                                    "collision_policy": "auto_increment",
                                },
                                "on_write_failure": "discard",
                            },
                        ],
                        "metadata": {"name": "numeric-gate"},
                    },
                },
            ],
        )
        # Turn 2: claim completion. No blocking diagnostic for this scenario,
        # so the proof gate returns False and the loop terminates cleanly.
        turn2 = _llm_response(content="Pipeline ready.", tool_calls=None)

        passing_preflight = ValidationResult(is_valid=True, checks=[], errors=[])
        empty = _empty_state()
        with (
            patch.object(service, "_call_llm", new_callable=AsyncMock) as mock_llm,
            patch.object(service, "_runtime_preflight", return_value=passing_preflight),
        ):
            mock_llm.side_effect = [turn1, turn2]
            result = await service.compose(
                "Split orders by price threshold",
                [],
                empty,
                session_id=session_id,
                user_id="test-user",
            )

        # First-pass success: zero forced repair turns.
        assert mock_llm.call_count == 2, f"expected 2 LLM calls, got {mock_llm.call_count}"
        assert result.repair_turns_used == 0

        scenario = _load_scenario("numeric-gate")
        state_dict = _state_dict_for_scoring(result)
        verdict = score(
            scenario,
            [{"role": "assistant", "content": result.message or ""}],
            state_dict,
        )
        assert verdict["verdict"] == "GREEN", (
            f"numeric-gate did not score GREEN. red={verdict['red_reasons']} amber={verdict['amber_reasons']}"
        )


# --------------------------------------------------------------------------
# Scenario 3: url-text-smoke
#
# Trigger text_source_url_without_web_scrape on turn 1 by declaring a text
# source whose blob content is a URL with no web_scrape downstream; turn 2
# claims completion → repair fires; turn 3 upserts a web_scrape node; turn
# 4 claims completion → GREEN.
# --------------------------------------------------------------------------


class TestUrlTextSmokeScenario:
    """End-to-end mocked-LLM run for the url-text-smoke convergence scenario.

    The text plugin is not in the builtin catalog, so this test cannot drive
    the same set_pipeline prevalidation path as the other two scenarios
    without registering a custom plugin. The compose loop's proof step
    inspects the source blob directly (via inspect_blob_content) rather
    than going through plugin schemas, so the proof-step coverage is
    valuable; the constraint is that set_pipeline must accept the text
    source. Skip for now and observe the gap rather than fake-pass — the
    live RGR harness still drives this scenario against a real LLM.
    """

    @pytest.mark.skip(
        reason="Text source plugin not in builtin catalog; requires custom plugin registration to drive set_pipeline prevalidation. See filigree-obs (filed alongside Fix 17) for the gap."
    )
    @pytest.mark.asyncio
    async def test_url_text_smoke_converges_with_one_repair_turn(self, tmp_path: Path) -> None:
        # Placeholder retained so a future contributor can wire this once
        # the text source plugin is exposed via the catalog or via test-only
        # plugin registration. The shape would mirror the csv-classifier
        # test: turn 1 = text source + sink (no web_scrape) → turn 2 claim →
        # repair fires text_source_url_without_web_scrape → turn 3 upserts
        # web_scrape → turn 4 claim → GREEN.
        pass
