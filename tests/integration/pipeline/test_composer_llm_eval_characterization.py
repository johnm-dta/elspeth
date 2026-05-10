"""Replay scoreboard for the 2026-04-28 composer LLM evaluation.

These tests intentionally describe the desired post-remediation behavior for
known defects as strict xfails. Run this module with ``--runxfail`` to get the
red characterization failures that later child tickets must turn green.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import yaml

from elspeth.contracts.schema_contract import SchemaContract
from elspeth.contracts.secrets import SecretInventoryItem
from elspeth.core.secrets import SecretResolutionError, resolve_secret_refs
from elspeth.core.security.secret_loader import EnvSecretLoader
from elspeth.plugins.transforms.batch_stats import BatchStats
from elspeth.testing import make_field, make_row
from elspeth.web.catalog.protocol import CatalogService
from elspeth.web.catalog.schemas import PluginSchemaInfo, PluginSummary
from elspeth.web.composer import yaml_generator as composer_yaml_generator
from elspeth.web.composer.progress import ComposerProgressEvent
from elspeth.web.composer.protocol import ComposerConvergenceError
from elspeth.web.composer.service import ComposerServiceImpl
from elspeth.web.composer.state import (
    CompositionState,
    EdgeSpec,
    NodeSpec,
    OutputSpec,
    PipelineMetadata,
    SourceSpec,
    ValidationSummary,
)
from elspeth.web.composer.tools import execute_tool
from elspeth.web.config import WebSettings
from elspeth.web.execution.schemas import ValidationResult
from elspeth.web.execution.validation import validate_pipeline
from elspeth.web.secrets.server_store import ServerSecretStore
from elspeth.web.secrets.service import ScopedSecretResolver, WebSecretService
from tests.fixtures.factories import make_context

pytestmark = pytest.mark.composer_llm_eval


SOURCE_REPORT = "notes/composer-llm-eval-2026-04-28.md"
EVAL_MODEL = "openrouter/openai/gpt-5.5"
EVAL_USER_ID = "dta_user"

ISSUE_CHARACTERIZATION = "elspeth-a5481032bd"
ISSUE_BLOB_PATH = "elspeth-411435710b"
ISSUE_RUNTIME_PREFLIGHT = "elspeth-34baf10c01"
ISSUE_TRIGGER_END_OF_SOURCE = "elspeth-fa94309e28"
ISSUE_BATCH_STATS_REQUIRED_FIELDS = "elspeth-178f765792"
ISSUE_BATCH_STATS_GROUP_BY = "elspeth-95904149b2"
ISSUE_SECRET_AVAILABILITY = "elspeth-cd5d811121"
ISSUE_INTROSPECTION_PATH = "elspeth-0380d5119f"
ISSUE_PROGRESS_CLASSIFICATION = "elspeth-5030f7373d"

EXPECTED_REDACTED_BLOB_SOURCE_PATH = "<redacted-blob-source-path>"

SCENARIO_1A_SESSION_ID = "c549bb63-47e9-427f-9a27-35467f877395"
SCENARIO_1B_SESSION_ID = "6472ff67-1052-406c-98c3-b3278e9ef4ea"
SCENARIO_2_SESSION_ID = "ae6816aa-1f75-4103-b176-886d14f9e104"
SCENARIO_3_SESSION_ID = "9002ed1f-3046-4c00-86be-2f1e3b3bd932"


@dataclass
class FakeFunction:
    name: str
    arguments: str


@dataclass
class FakeToolCall:
    id: str
    function: FakeFunction


@dataclass
class FakeMessage:
    content: str | None
    tool_calls: list[FakeToolCall] | None


@dataclass
class FakeChoice:
    message: FakeMessage


@dataclass
class FakeLLMResponse:
    choices: list[FakeChoice]


class _NoUserSecretStore:
    """User secret store stub for server-secret-only replay cases."""

    def list_secrets(self, *, user_id: str, auth_provider_type: str) -> list[SecretInventoryItem]:
        del user_id, auth_provider_type
        return []

    def has_secret_record(self, name: str, *, user_id: str, auth_provider_type: str) -> bool:
        del name, user_id, auth_provider_type
        return False

    def has_secret(self, name: str, *, user_id: str, auth_provider_type: str) -> bool:
        del name, user_id, auth_provider_type
        return False


def _mock_catalog() -> MagicMock:
    catalog = MagicMock(spec=CatalogService)
    catalog.list_sources.return_value = [
        PluginSummary(name="csv", description="CSV source", plugin_type="source", config_fields=[]),
    ]
    catalog.list_transforms.return_value = [
        PluginSummary(name="batch_stats", description="Batch stats", plugin_type="transform", config_fields=[]),
    ]
    catalog.list_sinks.return_value = [
        PluginSummary(name="json", description="JSON sink", plugin_type="sink", config_fields=[]),
    ]
    catalog.get_schema.return_value = PluginSchemaInfo(
        name="csv",
        plugin_type="source",
        description="CSV source",
        json_schema={"title": "Config", "properties": {}},
    )
    return catalog


def _make_llm_response(
    content: str | None = None,
    tool_calls: list[dict[str, Any]] | None = None,
) -> FakeLLMResponse:
    fake_tool_calls: list[FakeToolCall] | None = None
    if tool_calls is not None:
        fake_tool_calls = [
            FakeToolCall(
                id=tool_call["id"],
                function=FakeFunction(
                    name=tool_call["name"],
                    arguments=json.dumps(tool_call["arguments"]),
                ),
            )
            for tool_call in tool_calls
        ]
    return FakeLLMResponse(choices=[FakeChoice(message=FakeMessage(content=content, tool_calls=fake_tool_calls))])


def _empty_state() -> CompositionState:
    return CompositionState(
        source=None,
        nodes=(),
        edges=(),
        outputs=(),
        metadata=PipelineMetadata(name=f"Composer LLM eval replay ({ISSUE_CHARACTERIZATION})"),
        version=1,
    )


def _web_settings(data_dir: Path, **overrides: Any) -> WebSettings:
    defaults: dict[str, Any] = {
        "data_dir": data_dir,
        "composer_model": EVAL_MODEL,
        "composer_max_composition_turns": 15,
        "composer_max_discovery_turns": 10,
        "composer_timeout_seconds": 180.0,
        "composer_rate_limit_per_minute": 60,
    }
    defaults.update(overrides)
    return WebSettings(**defaults)


def _write_scenario_csv(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "ticket_id,customer_tier,amount\nT-001,gold,10.0\nT-002,silver,20.0\n",
        encoding="utf-8",
    )


def _output(name: str, path: str | Path) -> OutputSpec:
    return OutputSpec(
        name=name,
        plugin="json",
        options={"path": str(path), "format": "jsonl", "schema": {"mode": "observed"}},
        on_write_failure="discard",
    )


def _source(source_path: str | Path, *, on_success: str, extra_options: dict[str, Any] | None = None) -> SourceSpec:
    options: dict[str, Any] = {
        "path": str(source_path),
        "schema": {"mode": "fixed", "fields": ["ticket_id: str", "customer_tier: str", "amount: float"]},
    }
    options.update(extra_options or {})
    # ``on_validation_failure`` accepts ``"discard"`` or a real sink name
    # (route-target validators reject anything else). The eval scenarios do
    # not exercise quarantine routing, so use ``discard`` to keep fixtures
    # internally consistent. Issue elspeth-127de6865a closed the silent-pass
    # behaviour that previously let dangling quarantine targets through.
    return SourceSpec(
        plugin="csv",
        on_success=on_success,
        options=options,
        on_validation_failure="discard",
    )


def _direct_source_state(
    source_path: str | Path,
    output_path: str | Path,
    *,
    blob_ref: str | None = None,
) -> CompositionState:
    extra_options = {"blob_ref": blob_ref} if blob_ref is not None else None
    return CompositionState(
        source=_source(source_path, on_success="summary", extra_options=extra_options),
        nodes=(),
        edges=(EdgeSpec(id="e_source_summary", from_node="source", to_node="summary", edge_type="on_success", label=None),),
        outputs=(_output("summary", output_path),),
        metadata=PipelineMetadata(name=f"{SOURCE_REPORT} scenario 1B"),
        version=1,
    )


def _aggregation_state(
    source_path: str | Path,
    output_path: str | Path,
    *,
    trigger: dict[str, Any] | None,
    aggregation_options: dict[str, Any],
) -> CompositionState:
    return CompositionState(
        source=_source(source_path, on_success="aggregate_in"),
        nodes=(
            NodeSpec(
                id="tier_summary",
                node_type="aggregation",
                plugin="batch_stats",
                input="aggregate_in",
                on_success="summary",
                on_error="discard",
                options=aggregation_options,
                condition=None,
                routes=None,
                fork_to=None,
                branches=None,
                policy=None,
                merge=None,
                trigger=trigger,
                output_mode="transform",
                expected_output_count=1,
            ),
        ),
        edges=(
            EdgeSpec(id="e_source_agg", from_node="source", to_node="tier_summary", edge_type="on_success", label=None),
            EdgeSpec(id="e_agg_summary", from_node="tier_summary", to_node="summary", edge_type="on_success", label=None),
        ),
        outputs=(_output("summary", output_path),),
        metadata=PipelineMetadata(name=f"{SOURCE_REPORT} scenario 2"),
        version=1,
    )


def _scenario_2_files(tmp_path: Path) -> tuple[Path, Path, Path]:
    data_dir = tmp_path / "data"
    source_path = data_dir / "blobs" / SCENARIO_2_SESSION_ID / "tickets.csv"
    output_path = data_dir / "outputs" / "tier_summary.jsonl"
    _write_scenario_csv(source_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    return data_dir, source_path, output_path


def _format_validation_errors(result: ValidationResult) -> str:
    return "\n".join(error.message for error in result.errors)


def _format_composer_errors(summary: ValidationSummary) -> str:
    return "\n".join(entry.message for entry in summary.errors)


def _make_pipeline_row(data: dict[str, Any]):
    fields = tuple(
        make_field(key, type(value) if value is not None else object, original_name=key, required=False, source="inferred")
        for key, value in data.items()
    )
    contract = SchemaContract(mode="OBSERVED", fields=fields, locked=True)
    return make_row(data, contract=contract)


def test_scenario_1b_blob_service_storage_path_validates_through_runtime_path_allowlist(
    tmp_path: Path,
) -> None:
    """Protects the Scenario 1B/3 blob-path failure captured in the report.

    Originally characterized that a relative ``data/blobs/<sid>/<bid>_<filename>``
    path would resolve via the legacy CWD branch in ``web/paths.py`` and pass
    the runtime path allowlist.  After elspeth-07089fbaa3 closed the
    composer-stored blob source path defect, the canonical contract is that
    blob-backed source paths are absolute (``BlobRecord.storage_path``); the
    legacy relative form is no longer accepted.  This test now pins the
    *post-fix* contract: an absolute canonical path under the configured
    ``data_dir`` validates cleanly through the runtime allowlist.
    """
    data_dir = tmp_path / "data"
    blob_id = "11111111-1111-4111-8111-111111111111"
    canonical_storage_path = data_dir / "blobs" / SCENARIO_1B_SESSION_ID / f"{blob_id}_tickets.csv"
    _write_scenario_csv(canonical_storage_path)

    state = _direct_source_state(
        str(canonical_storage_path),
        str(data_dir / "outputs" / "scenario_1b_summary.jsonl"),
        blob_ref=blob_id,
    )

    composer_summary = state.validate()
    assert composer_summary.is_valid, _format_composer_errors(composer_summary)

    runtime_result = validate_pipeline(
        state,
        _web_settings(data_dir),
        composer_yaml_generator,
    )

    assert runtime_result.is_valid, _format_validation_errors(runtime_result)


def test_scenario_2_end_of_source_condition_rejected_before_runtime_settings_load(tmp_path: Path) -> None:
    """Protects Scenario 2 aggregation trigger shape drift."""
    data_dir, source_path, output_path = _scenario_2_files(tmp_path)
    state = _aggregation_state(
        source_path,
        output_path,
        trigger={"condition": "end_of_source"},
        aggregation_options={"schema": {"mode": "observed"}, "value_field": "amount"},
    )

    runtime_result = validate_pipeline(state, _web_settings(data_dir), composer_yaml_generator)
    assert not runtime_result.is_valid
    assert "end_of_source" in _format_validation_errors(runtime_result)

    composer_summary = state.validate()
    assert not composer_summary.is_valid, "composer accepted an end_of_source token in the boolean condition slot"
    assert "end_of_source" in _format_composer_errors(composer_summary)


def test_scenario_2_omitted_trigger_is_end_of_source_only_contract(tmp_path: Path) -> None:
    """Composer and runtime agree that omitted trigger means end-of-source-only aggregation."""
    data_dir, source_path, output_path = _scenario_2_files(tmp_path)
    state = _aggregation_state(
        source_path,
        output_path,
        trigger=None,
        aggregation_options={"schema": {"mode": "observed"}, "value_field": "amount"},
    )

    composer_summary = state.validate()
    assert composer_summary.is_valid, _format_composer_errors(composer_summary)

    yaml_doc = yaml.safe_load(composer_yaml_generator.generate_yaml(state))
    assert "trigger" not in yaml_doc["aggregations"][0]

    runtime_result = validate_pipeline(state, _web_settings(data_dir), composer_yaml_generator)
    assert runtime_result.is_valid, _format_validation_errors(runtime_result)


def test_scenario_2_batch_stats_required_input_fields_returns_pre_execution_validation_error(tmp_path: Path) -> None:
    """Protects the ADR-013 batch-aware dispatch gap from Scenario 2."""
    data_dir, source_path, output_path = _scenario_2_files(tmp_path)
    state = _aggregation_state(
        source_path,
        output_path,
        trigger={"count": 100},
        aggregation_options={
            "schema": {"mode": "observed"},
            "value_field": "amount",
            "required_input_fields": ["amount"],
        },
    )

    runtime_result = validate_pipeline(state, _web_settings(data_dir), composer_yaml_generator)
    assert not runtime_result.is_valid
    assert "batch-aware" in _format_validation_errors(runtime_result)

    composer_summary = state.validate()
    assert not composer_summary.is_valid
    assert "required_input_fields" in _format_composer_errors(composer_summary)


def test_scenario_2_batch_stats_group_by_emits_per_tier_rollups() -> None:
    """Protects elspeth-95904149b2 with the batch_stats per-group rollup contract."""
    transform = BatchStats(
        {
            "schema": {"mode": "observed"},
            "value_field": "amount",
            "group_by": "customer_tier",
        }
    )
    rows = [
        _make_pipeline_row({"ticket_id": "T-001", "customer_tier": "gold", "amount": 10.0}),
        _make_pipeline_row({"ticket_id": "T-002", "customer_tier": "silver", "amount": 20.0}),
        _make_pipeline_row({"ticket_id": "T-003", "customer_tier": "gold", "amount": 30.0}),
    ]

    result = transform.process(rows, make_context())

    assert result.status == "success"
    assert result.is_multi_row
    assert result.rows is not None
    rollups = {row["customer_tier"]: row for row in result.rows}
    assert set(rollups) == {"gold", "silver"}
    assert rollups["gold"]["count"] == 2
    assert rollups["gold"]["sum"] == 40.0
    assert rollups["silver"]["count"] == 1
    assert rollups["silver"]["sum"] == 20.0


def test_known_secret_env_marker_cannot_bypass_unavailable_web_secret_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Protects elspeth-cd5d811121 without any live provider call."""
    secret_name = "OPENROUTER_API_KEY"
    secret_value = "test-openrouter-key"
    monkeypatch.setenv(secret_name, secret_value)
    monkeypatch.delenv("ELSPETH_FINGERPRINT_KEY", raising=False)

    loaded_value, secret_ref = EnvSecretLoader().get_secret(secret_name)
    assert loaded_value == secret_value
    assert secret_ref.source == "env"

    settings = _web_settings(
        tmp_path / "data",
        composer_model=EVAL_MODEL,
        server_secret_allowlist=(secret_name,),
    )
    composer = ComposerServiceImpl(catalog=_mock_catalog(), settings=settings)
    assert composer._availability.available is True
    assert composer._availability.provider == "openrouter"

    web_secret_service = WebSecretService(
        user_store=_NoUserSecretStore(),  # type: ignore[arg-type]
        server_store=ServerSecretStore(allowlist=(secret_name,)),
    )
    resolver = ScopedSecretResolver(web_secret_service, auth_provider_type=settings.auth_provider)

    result = execute_tool(
        "validate_secret_ref",
        {"name": secret_name},
        _empty_state(),
        _mock_catalog(),
        secret_service=resolver,
        user_id=EVAL_USER_ID,
    )

    assert result.success is True
    assert result.to_dict()["data"] == {"name": secret_name, "available": False}

    with pytest.raises(SecretResolutionError) as exc_info:
        resolve_secret_refs(
            {"api_key": f"${{{secret_name}}}"},
            resolver,
            EVAL_USER_ID,
            env_ref_names=frozenset({secret_name}),
        )

    assert exc_info.value.missing == [secret_name]


def test_scenario_3_get_pipeline_state_preserves_redacted_patched_blob_path_that_yaml_preserves(tmp_path: Path) -> None:
    """Characterizes elspeth-0380d5119f redaction across get_pipeline_state and YAML.

    Originally exercised ``patch_source_options`` to set the blob source
    path; after elspeth-07089fbaa3 closed the composer-stored blob source
    path defect, that flow is forbidden — patches against a blob-backed
    source may not touch ``path`` or ``blob_ref`` because the binding is
    immutable.  The redaction contract (LLM/HTTP surfaces see the sentinel;
    YAML emission preserves the real path for the runtime) is now pinned
    against an initial state that already carries the canonical path
    (the post-fix shape produced by ``set_source_from_blob``).
    """
    data_dir = tmp_path / "data"
    blob_id = "33333333-3333-4333-8333-333333333333"
    source_path = data_dir / "blobs" / SCENARIO_3_SESSION_ID / f"{blob_id}_tickets.csv"
    output_path = data_dir / "outputs" / "scenario_3_summary.jsonl"
    _write_scenario_csv(source_path)

    initial_state = CompositionState(
        source=SourceSpec(
            plugin="csv",
            on_success="summary",
            options={
                "blob_ref": blob_id,
                "path": str(source_path),
                "schema": {"mode": "observed"},
            },
            on_validation_failure="discard",
        ),
        nodes=(),
        edges=(EdgeSpec(id="e_source_summary", from_node="source", to_node="summary", edge_type="on_success", label=None),),
        outputs=(_output("summary", output_path),),
        metadata=PipelineMetadata(name=f"{SOURCE_REPORT} scenario 3 patched path"),
        version=1,
    )

    # patch_source_options against a blob-backed source must reject any
    # patch that touches the immutable (path, blob_ref) binding — see
    # elspeth-07089fbaa3.  Re-binding requires set_source_from_blob.
    rejected = execute_tool(
        "patch_source_options",
        {"patch": {"path": str(source_path)}},
        initial_state,
        _mock_catalog(),
        data_dir=str(data_dir),
    )
    assert rejected.success is False
    assert "blob-backed source" in rejected.data["error"]

    # The redaction contract still holds for canonical-path blob sources
    # (the shape set_source_from_blob produces).
    introspection = execute_tool(
        "get_pipeline_state",
        {"component": "source"},
        initial_state,
        _mock_catalog(),
    )
    assert introspection.success is True
    introspected_source = introspection.to_dict()["data"]["source"]
    assert introspected_source["options"]["path"] == EXPECTED_REDACTED_BLOB_SOURCE_PATH
    assert introspected_source["options"]["blob_ref"] == blob_id
    assert str(source_path) not in json.dumps(introspection.to_dict()["data"])

    yaml_doc = yaml.safe_load(composer_yaml_generator.generate_yaml(initial_state))
    assert yaml_doc["source"]["options"]["path"] == str(source_path)


async def _failed_progress_for_timeout(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> ComposerProgressEvent:
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-openrouter-key")
    settings = _web_settings(tmp_path / "data", composer_timeout_seconds=0.05)
    service = ComposerServiceImpl(catalog=_mock_catalog(), settings=settings)
    events: list[ComposerProgressEvent] = []

    async def record_progress(event: ComposerProgressEvent) -> None:
        events.append(event)

    async def slow_llm(*args: Any, **kwargs: Any) -> FakeLLMResponse:
        del args, kwargs
        await asyncio.sleep(1.0)
        return _make_llm_response(content="too late")

    with patch.object(service, "_call_llm", new_callable=AsyncMock) as mock_llm:
        mock_llm.side_effect = slow_llm
        with pytest.raises(ComposerConvergenceError) as exc_info:
            await service.compose(
                "Scenario 1A monolithic request",
                [],
                _empty_state(),
                session_id=SCENARIO_1A_SESSION_ID,
                user_id=EVAL_USER_ID,
                progress=record_progress,
            )

    assert exc_info.value.budget_exhausted == "timeout"
    return next(event for event in reversed(events) if event.phase == "failed")


async def _failed_progress_for_composition_budget(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> ComposerProgressEvent:
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-openrouter-key")
    settings = _web_settings(tmp_path / "data", composer_max_composition_turns=1)
    service = ComposerServiceImpl(catalog=_mock_catalog(), settings=settings)
    events: list[ComposerProgressEvent] = []

    async def record_progress(event: ComposerProgressEvent) -> None:
        events.append(event)

    mutation = _make_llm_response(
        tool_calls=[
            {
                "id": "call_1",
                "name": "set_metadata",
                "arguments": {"patch": {"name": "budget replay"}},
            }
        ]
    )
    bonus_mutation = _make_llm_response(
        tool_calls=[
            {
                "id": "call_2",
                "name": "set_metadata",
                "arguments": {"patch": {"description": "still mutating"}},
            }
        ]
    )

    with patch.object(service, "_call_llm", new_callable=AsyncMock) as mock_llm:
        mock_llm.side_effect = [mutation, bonus_mutation]
        with pytest.raises(ComposerConvergenceError) as exc_info:
            await service.compose(
                "Keep editing forever",
                [],
                _empty_state(),
                session_id=SCENARIO_1A_SESSION_ID,
                user_id=EVAL_USER_ID,
                progress=record_progress,
            )

    assert exc_info.value.budget_exhausted == "composition"
    return next(event for event in reversed(events) if event.phase == "failed")


def _progress_copy(event: ComposerProgressEvent) -> tuple[str, tuple[str, ...], str | None]:
    return (event.headline, event.evidence, event.likely_next)


@pytest.mark.asyncio
async def test_long_running_compose_failures_expose_distinct_progress_guidance(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Protects the report's long-running failure-classification finding.

    Originally an xfail-strict guard for ``{ISSUE_PROGRESS_CLASSIFICATION}``;
    flipped to a passing characterization test once the discriminator
    landed. Now serves as the regression guard against re-collapsing the
    three convergence sub-causes into a single generic event.
    """
    timeout_event = await _failed_progress_for_timeout(tmp_path, monkeypatch)
    composition_event = await _failed_progress_for_composition_budget(tmp_path, monkeypatch)

    assert _progress_copy(timeout_event) != _progress_copy(composition_event)
    # Tighter assertion than text inequality: the discriminator field must
    # carry the per-sub-cause Literal so frontend / response body / LLM
    # recovery can branch on a stable taxonomy.
    assert timeout_event.reason == "convergence_wall_clock_timeout"
    assert composition_event.reason == "convergence_composition_budget"


def test_runtime_preflight_preview_blocks_scenario_2_invalid_trigger(tmp_path: Path) -> None:
    """Scenario 2: preview must show runtime failure, not authoring-only validity."""
    data_dir, source_path, output_path = _scenario_2_files(tmp_path)
    state = _aggregation_state(
        source_path,
        output_path,
        trigger={"condition": "end_of_source"},
        aggregation_options={"schema": {"mode": "observed"}, "value_field": "amount"},
    )
    settings = _web_settings(data_dir)

    def runtime_preflight(candidate: CompositionState) -> ValidationResult:
        return validate_pipeline(candidate, settings, composer_yaml_generator)

    preview = execute_tool(
        "preview_pipeline",
        {},
        state,
        _mock_catalog(),
        data_dir=str(data_dir),
        runtime_preflight=runtime_preflight,
    )

    preview_data = preview.to_dict()["data"]
    assert preview.success is True
    assert preview_data["is_valid"] is False
    assert preview_data["runtime_preflight"]["is_valid"] is False
    assert "end_of_source" in json.dumps(preview_data["runtime_preflight"])


@pytest.mark.asyncio
async def test_final_completion_claim_is_augmented_with_runtime_preflight_failure(tmp_path: Path) -> None:
    """The composer must augment (not discard) the LLM's prose with the validator's
    objection after a dry-run failure. Issue elspeth-9cfbad6901 unified the
    preflight-fail policy on augmentation: the model's prose is preserved
    verbatim and the technical preflight error is appended as a system-
    attributed suffix.
    """
    data_dir, source_path, output_path = _scenario_2_files(tmp_path)
    settings = _web_settings(data_dir)
    composer = ComposerServiceImpl(catalog=_mock_catalog(), settings=settings)
    state = _aggregation_state(
        source_path,
        output_path,
        trigger={"condition": "end_of_source"},
        aggregation_options={"schema": {"mode": "observed"}, "value_field": "amount"},
    )
    changed_state = replace(state, version=state.version + 1)
    model_prose = "The pipeline is complete and valid."

    result = await composer._finalize_no_tool_response(
        content=model_prose,
        state=changed_state,
        initial_version=state.version,
        user_id=EVAL_USER_ID,
        last_runtime_preflight=None,
        runtime_preflight_cache=composer._new_runtime_preflight_cache(),
        session_scope="session:eval",
    )

    # Augmentation prefix invariant: model prose preserved verbatim at start.
    assert result.message.startswith(model_prose)
    # System-attributed suffix carries the validator's specific objection.
    # The first ValidationError.message for the end_of_source trigger case
    # contains "end_of_source", so a regression that omits the validator
    # detail from the suffix would fail this check.
    assert "[ELSPETH-SYSTEM]" in result.message
    assert "end_of_source" in result.message
    # raw_assistant_content carries the unaugmented prose for the audit
    # trail and LLM history replay.
    assert result.raw_assistant_content == model_prose
    assert result.runtime_preflight is not None
    assert result.runtime_preflight.is_valid is False
