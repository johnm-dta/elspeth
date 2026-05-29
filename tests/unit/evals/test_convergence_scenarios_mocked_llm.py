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
from typing import Any, cast
from unittest.mock import AsyncMock, patch
from uuid import uuid4

import pytest
import structlog
from evals.lib.composer_rgr_score import score
from sqlalchemy.pool import StaticPool

from elspeth.web.blobs.service import content_hash as _content_hash
from elspeth.web.composer.service import ComposerServiceImpl
from elspeth.web.composer.state import CompositionState, PipelineMetadata
from elspeth.web.config import WebSettings
from elspeth.web.execution.schemas import ValidationReadiness, ValidationResult
from elspeth.web.sessions.engine import create_session_engine
from elspeth.web.sessions.models import blobs_table, sessions_table
from elspeth.web.sessions.schema import initialize_session_schema
from elspeth.web.sessions.service import SessionServiceImpl
from elspeth.web.sessions.telemetry import build_sessions_telemetry

_REPO_ROOT = Path(__file__).resolve().parents[3]
_SUITE = _REPO_ROOT / "evals" / "composer-rgr" / "scenarios" / "convergence-suite"


def _passing_preflight() -> ValidationResult:
    return ValidationResult(
        is_valid=True,
        checks=[],
        errors=[],
        readiness=ValidationReadiness(authoring_valid=True, execution_ready=True, completion_ready=True, blockers=[]),
    )


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


def _make_settings(data_dir: Path) -> WebSettings:
    return WebSettings(
        data_dir=data_dir,
        composer_max_composition_turns=15,
        composer_max_discovery_turns=10,
        composer_timeout_seconds=85.0,
        composer_rate_limit_per_minute=10,
        shareable_link_signing_key=b"\x00" * 32,
    )


def _session_engine() -> tuple[Any, str, SessionServiceImpl]:
    engine = create_session_engine(
        "sqlite:///:memory:",
        poolclass=StaticPool,
        connect_args={"check_same_thread": False},
    )
    initialize_session_schema(engine)
    sessions_service = SessionServiceImpl(
        engine,
        data_dir=None,
        telemetry=build_sessions_telemetry(),
        log=structlog.get_logger("test.convergence-scenarios"),
    )
    session_id = str(uuid4())
    now = datetime.now(UTC)
    with engine.begin() as conn:
        conn.execute(
            sessions_table.insert().values(
                id=session_id,
                user_id="test-user",
                auth_provider_type="local",
                trust_mode="auto_commit",
                title="Convergence Test",
                created_at=now,
                updated_at=now,
            )
        )
    return engine, session_id, sessions_service


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
    return cast("dict[str, Any]", state_dict)


def _load_scenario(name: str) -> dict[str, Any]:
    return cast("dict[str, Any]", json.loads((_SUITE / name / "scenario.json").read_text()))


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
# columns; turn 2 claims completion → schema repair fires; turn 3 patches the
# source schema to mode=observed; turn 4 claims completion → the mandatory
# llm_prompt_template interpretation review re-prompts (turn 5) → GREEN. The
# authored prompt_template now requires an interpretation review, adding a second
# forced repair turn on top of the schema fix (5 calls / 2 repair turns).
# --------------------------------------------------------------------------


class TestCsvClassifierScenario:
    """End-to-end mocked-LLM run for the csv-classifier convergence scenario."""

    @pytest.mark.asyncio
    async def test_csv_classifier_converges_in_two_repair_turns(self, tmp_path: Path) -> None:
        engine, session_id, sessions_service = _session_engine()
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
        service = ComposerServiceImpl(catalog=catalog, settings=settings, sessions_service=sessions_service, session_engine=engine)

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
                                    "prompt_template": "Classify {{ row['subject'] }}",
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
                                    "mode": "write",
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

        passing_preflight = _passing_preflight()
        empty = _empty_state()
        with (
            patch.object(service, "_call_llm", new_callable=AsyncMock) as mock_llm,
            patch.object(service, "_runtime_preflight", return_value=passing_preflight),
        ):
            # Turn 5 (completion): the LLM node carries an authored prompt_template,
            # which now triggers a mandatory llm_prompt_template interpretation review
            # ("surface every authored LLM decision through interpretation review").
            # The compose loop re-prompts for that unresolved review (CALL3 and CALL5
            # are the assumption-review re-prompts) on top of the schema-mode repair,
            # so this scenario converges to a valid (GREEN) structural state in
            # 5 LLM calls / 2 forced repair turns rather than the pre-review 4 / 1.
            mock_llm.side_effect = [turn1, turn2, turn3, turn4, turn4]
            result = await service.compose(
                "Classify these tickets",
                [],
                empty,
                session_id=session_id,
                user_id="test-user",
            )

        # Convergence behaviour: two forced repair turns (schema-mode + the
        # mandatory interpretation-review re-prompt for the authored prompt_template).
        assert mock_llm.call_count == 5, f"expected 5 LLM calls, got {mock_llm.call_count}"
        assert result.repair_turns_used == 2, f"expected 2 repair turns, got {result.repair_turns_used}"

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
# The proof step blocks the direct observed-CSV -> numeric-gate failure shape
# with gate_expression_type_mismatch_against_source_schema. A first-pass
# success path still exists when the model uses the recipe-equivalent
# type_coerce + gate + 2 outputs shape immediately.
# --------------------------------------------------------------------------


class TestNumericGateScenario:
    """End-to-end mocked-LLM run for the numeric-gate convergence scenario.

    One test pins the first-pass success path. The sibling test pins the
    proof-repair path for the historical direct observed-CSV numeric gate
    mismatch.
    """

    @pytest.mark.asyncio
    async def test_numeric_gate_first_pass_success(self, tmp_path: Path) -> None:
        engine, session_id, sessions_service = _session_engine()
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
        service = ComposerServiceImpl(catalog=catalog, settings=settings, sessions_service=sessions_service, session_engine=engine)

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
                                    "mode": "write",
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
                                    "mode": "write",
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

        passing_preflight = _passing_preflight()
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

    @pytest.mark.asyncio
    async def test_numeric_gate_direct_observed_csv_gate_repairs_with_one_turn(self, tmp_path: Path) -> None:
        engine, session_id, sessions_service = _session_engine()
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
        service = ComposerServiceImpl(catalog=catalog, settings=settings, sessions_service=sessions_service, session_engine=engine)

        # Turn 1: structurally valid but runtime-broken pipeline — observed
        # CSV values are raw strings, so the direct numeric gate would fail
        # with ExpressionEvaluationError at run time.
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
                                "id": "threshold_gate",
                                "node_type": "gate",
                                "input": "rows",
                                "condition": "row['price'] >= 100",
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
                                    "mode": "write",
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
                                    "mode": "write",
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
        # Turn 2: claim completion -> proof gate fires.
        turn2 = _llm_response(content="Pipeline ready.", tool_calls=None)
        # Turn 3: repair with the recipe-equivalent type_coerce before gate.
        turn3 = _llm_response(
            content=None,
            tool_calls=[
                {
                    "id": "call_repair",
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
                                    "mode": "write",
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
                                    "mode": "write",
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
        # Turn 4: claim completion after repair.
        turn4 = _llm_response(content="Repaired and ready.", tool_calls=None)

        passing_preflight = _passing_preflight()
        empty = _empty_state()
        with (
            patch.object(service, "_call_llm", new_callable=AsyncMock) as mock_llm,
            patch.object(service, "_runtime_preflight", return_value=passing_preflight),
        ):
            mock_llm.side_effect = [turn1, turn2, turn3, turn4]
            result = await service.compose(
                "Split orders by price threshold",
                [],
                empty,
                session_id=session_id,
                user_id="test-user",
            )

        assert mock_llm.call_count == 4, f"expected 4 LLM calls, got {mock_llm.call_count}"
        assert result.repair_turns_used == 1

        scenario = _load_scenario("numeric-gate")
        state_dict = _state_dict_for_scoring(result)
        verdict = score(
            scenario,
            [{"role": "assistant", "content": result.message or ""}],
            state_dict,
        )
        assert verdict["verdict"] == "GREEN", (
            f"numeric-gate repair flow did not score GREEN. red={verdict['red_reasons']} amber={verdict['amber_reasons']}"
        )


# --------------------------------------------------------------------------
# Scenario 3: url-text-smoke
#
# Trigger text_source_url_without_web_scrape on turn 1 by declaring a text
# source whose blob content is a URL with no web_scrape downstream; turn 2
# claims completion → repair fires; turn 3 re-issues set_pipeline with a
# web_scrape transform between source and sink; turn 4 claims completion →
# GREEN.
# --------------------------------------------------------------------------


class TestUrlTextSmokeScenario:
    """End-to-end mocked-LLM run for the url-text-smoke convergence scenario."""

    @pytest.mark.asyncio
    async def test_url_text_smoke_converges_with_one_repair_turn(self, tmp_path: Path) -> None:
        engine, session_id, sessions_service = _session_engine()
        # Single-line text blob containing only a URL — exactly the shape
        # the proof step's text_source_url_without_web_scrape blocker keys
        # off. inspect_blob_content sets source_kind="text" and populates
        # url_candidates from the blob bytes.
        body = b"https://www.iana.org/help/example-domains\n"
        blob_id = _seed_blob(
            engine,
            session_id,
            body=body,
            filename="urls.txt",
            mime_type="text/plain",
            storage_dir=tmp_path / "blobs" / session_id,
        )

        catalog = _real_catalog()
        settings = _make_settings(tmp_path)
        service = ComposerServiceImpl(catalog=catalog, settings=settings, sessions_service=sessions_service, session_engine=engine)

        # Turn 1: blocking pipeline — text source whose blob content is a
        # URL, but no web_scrape transform downstream. The URL string
        # itself flows to the sink instead of being fetched.
        turn1 = _llm_response(
            content=None,
            tool_calls=[
                {
                    "id": "call_set",
                    "name": "set_pipeline",
                    "arguments": {
                        "source": {
                            "plugin": "text",
                            "blob_id": blob_id,
                            "on_success": "url_rows",
                            "options": {
                                "column": "url",
                                "schema": {"mode": "fixed", "fields": ["url: str"]},
                            },
                            "on_validation_failure": "discard",
                        },
                        "nodes": [],
                        "edges": [],
                        "outputs": [
                            {
                                "sink_name": "url_rows",
                                "plugin": "json",
                                "options": {
                                    "path": "outputs/urls.jsonl",
                                    "format": "jsonl",
                                    "schema": {"mode": "observed"},
                                    "mode": "write",
                                    "collision_policy": "auto_increment",
                                },
                                "on_write_failure": "discard",
                            }
                        ],
                        "metadata": {"name": "url-text-smoke"},
                    },
                },
            ],
        )
        # Turn 2: claim completion → proof gate fires text_source_url_without_web_scrape.
        turn2 = _llm_response(content="All set.", tool_calls=None)
        # Turn 3: repair — re-issue set_pipeline with web_scrape inserted
        # between the source and the sink.
        turn3 = _llm_response(
            content=None,
            tool_calls=[
                {
                    "id": "call_repair",
                    "name": "set_pipeline",
                    "arguments": {
                        "source": {
                            "plugin": "text",
                            "blob_id": blob_id,
                            "on_success": "raw_urls",
                            "options": {
                                "column": "url",
                                "schema": {"mode": "fixed", "fields": ["url: str"]},
                            },
                            "on_validation_failure": "discard",
                        },
                        "nodes": [
                            {
                                "id": "fetch",
                                "node_type": "transform",
                                "plugin": "web_scrape",
                                "input": "raw_urls",
                                "on_success": "fetched",
                                "on_error": "discard",
                                "options": {
                                    "url_field": "url",
                                    "schema": {"mode": "fixed", "fields": ["url: str"]},
                                    "content_field": "content",
                                    "fingerprint_field": "content_fingerprint",
                                    "format": "text",
                                    "text_separator": "\n",
                                    "http": {
                                        "abuse_contact": "test@example.com",
                                        "scraping_reason": "convergence test",
                                        "allowed_hosts": "public_only",
                                    },
                                },
                            }
                        ],
                        "edges": [],
                        "outputs": [
                            {
                                "sink_name": "fetched",
                                "plugin": "json",
                                "options": {
                                    "path": "outputs/fetched.jsonl",
                                    "format": "jsonl",
                                    "schema": {"mode": "observed"},
                                    "mode": "write",
                                    "collision_policy": "auto_increment",
                                },
                                "on_write_failure": "discard",
                            }
                        ],
                        "metadata": {"name": "url-text-smoke"},
                    },
                },
            ],
        )
        # Turn 4: claim completion (clean — proof step no longer blocks).
        turn4 = _llm_response(content="Repaired and ready.", tool_calls=None)

        passing_preflight = _passing_preflight()
        empty = _empty_state()
        with (
            patch.object(service, "_call_llm", new_callable=AsyncMock) as mock_llm,
            patch.object(service, "_runtime_preflight", return_value=passing_preflight),
        ):
            mock_llm.side_effect = [turn1, turn2, turn3, turn4]
            result = await service.compose(
                "Fetch the URL and write a JSONL summary",
                [],
                empty,
                session_id=session_id,
                user_id="test-user",
            )

        # Convergence behaviour: exactly one forced repair turn.
        assert mock_llm.call_count == 4, f"expected 4 LLM calls, got {mock_llm.call_count}"
        assert result.repair_turns_used == 1, f"expected 1 repair turn, got {result.repair_turns_used}"

        scenario = _load_scenario("url-text-smoke")
        assistant_messages = [{"role": "assistant", "content": result.message or ""}]
        state_dict = _state_dict_for_scoring(result)
        verdict = score(scenario, assistant_messages, state_dict)

        assert verdict["verdict"] == "GREEN", (
            f"url-text-smoke did not score GREEN. red={verdict['red_reasons']} amber={verdict['amber_reasons']}"
        )
