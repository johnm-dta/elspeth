"""Validate-time inline blob checks surface structured results, not 500s."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import UUID, uuid4

import pytest
import yaml
from pydantic import SecretBytes

from elspeth.contracts.enums import CreationModality
from elspeth.contracts.hashing import stable_hash
from elspeth.core.dag.graph import ExecutionGraph
from elspeth.web.blobs.protocol import BlobNotFoundError, BlobRecord
from elspeth.web.composer import yaml_generator as composer_yaml_generator
from elspeth.web.composer.state import CompositionState, NodeSpec, OutputSpec, PipelineMetadata, SourceSpec
from elspeth.web.config import WebSettings
from elspeth.web.execution.progress import ProgressBroadcaster
from elspeth.web.execution.service import ExecutionServiceImpl
from elspeth.web.execution.validation import validate_pipeline
from elspeth.web.interpretation_state import INTERPRETATION_REQUIREMENTS_KEY
from elspeth.web.sessions.telemetry import build_sessions_telemetry

VALID_HASH = "a" * 64
BLOB_ID = UUID("5b7a4e0e-9e4a-4f0b-8d3e-2c0e1f0d3a4b")


def _state_with_inline_prompt(tmp_path: Path) -> CompositionState:
    blobs_dir = tmp_path / "blobs"
    outputs_dir = tmp_path / "outputs"
    blobs_dir.mkdir()
    outputs_dir.mkdir()

    return CompositionState(
        source=SourceSpec(
            plugin="csv",
            on_success="classify_input",
            options={
                "path": str(blobs_dir / "input.csv"),
                "schema": {"mode": "observed"},
            },
            on_validation_failure="discard",
        ),
        nodes=(
            NodeSpec(
                id="classify",
                node_type="transform",
                plugin="llm",
                input="classify_input",
                on_success="results",
                on_error="discard",
                options={
                    "provider": "openrouter",
                    "api_key": {"secret_ref": "OPENROUTER_API_KEY"},
                    "model": "openai/gpt-4o",
                    "prompt_template": {
                        "blob_ref": str(BLOB_ID),
                        "mode": "inline_content",
                        "sha256": VALID_HASH,
                    },
                    "required_input_fields": [],
                    "schema": {"mode": "observed"},
                    # Pre-resolved model-choice review so the
                    # interpretation gate doesn't short-circuit the
                    # validator before the blob-inline check runs. The
                    # auto-stager normally creates a pending requirement
                    # at mutation time; tests that bypass the composer
                    # (constructing NodeSpec directly) must stage the
                    # resolved form themselves.
                    INTERPRETATION_REQUIREMENTS_KEY: [
                        {
                            "id": "model_choice_review:classify",
                            "kind": "llm_model_choice",
                            "user_term": "llm_model_choice:classify",
                            "status": "resolved",
                            "draft": "openai/gpt-4o",
                            "event_id": "model-choice-accepted",
                            "accepted_value": "openai/gpt-4o",
                            "accepted_artifact_hash": None,
                            "resolved_prompt_template_hash": stable_hash("openai/gpt-4o"),
                        }
                    ],
                },
                condition=None,
                routes=None,
                fork_to=None,
                branches=None,
                policy=None,
                merge=None,
            ),
        ),
        edges=(),
        outputs=(
            OutputSpec(
                name="results",
                plugin="json",
                options={
                    "path": str(outputs_dir / "results.jsonl"),
                    "format": "jsonl",
                    "schema": {"mode": "observed"},
                },
                on_write_failure="discard",
            ),
        ),
        metadata=PipelineMetadata(),
        version=1,
    )


def _ready_blob_record(*, session_id: UUID, blob_id: UUID = BLOB_ID, size_bytes: int = 32) -> BlobRecord:
    return BlobRecord(
        id=blob_id,
        session_id=session_id,
        filename="prompt.txt",
        mime_type="text/plain",
        size_bytes=size_bytes,
        content_hash=VALID_HASH,
        storage_path=f"/tmp/{blob_id}_prompt.txt",
        created_at=datetime.now(UTC),
        created_by="user",
        source_description=None,
        status="ready",
        creation_modality=CreationModality.VERBATIM,
        created_from_message_id=None,
        creating_model_identifier=None,
        creating_model_version=None,
        creating_provider=None,
        creating_composer_skill_hash=None,
        creating_arguments_hash=None,
    )


def test_validate_returns_structured_violation_for_missing_inline_blob(tmp_path: Path) -> None:
    result = validate_pipeline(
        _state_with_inline_prompt(tmp_path),
        SimpleNamespace(data_dir=tmp_path),
        composer_yaml_generator,
        blob_get_metadata=lambda _blob_id: None,
    )

    assert result.is_valid is False
    blob_check = next(check for check in result.checks if check.name == "blob_inline_refs")
    assert blob_check.passed is False
    assert any(error.error_code == "missing_inline_blob_content" for error in result.errors)
    assert any(error.component_id == "classify" and error.component_type == "transform" for error in result.errors)


def test_validate_blob_inline_failure_has_no_duplicate_check_results(tmp_path: Path) -> None:
    result = validate_pipeline(
        _state_with_inline_prompt(tmp_path),
        SimpleNamespace(data_dir=tmp_path),
        composer_yaml_generator,
        blob_get_metadata=lambda _blob_id: None,
    )

    check_names = [check.name for check in result.checks]
    assert len(check_names) == len(set(check_names))


@patch("elspeth.web.execution.validation.assemble_and_validate_pipeline_config")
@patch("elspeth.web.execution.validation.build_runtime_graph")
@patch("elspeth.web.execution.validation.instantiate_runtime_plugins")
@patch("elspeth.web.execution.validation.load_settings_from_yaml_string")
def test_validate_substitutes_ready_inline_blob_marker_before_settings_load(
    mock_load_settings: MagicMock,
    mock_instantiate: MagicMock,
    mock_build_graph: MagicMock,
    mock_assemble: MagicMock,
    tmp_path: Path,
) -> None:
    session_id = uuid4()

    def load_settings(yaml_text: str, *, expand_env_vars: bool = True) -> SimpleNamespace:
        # Web preflight must not expand host ${VAR} placeholders.
        assert expand_env_vars is False
        doc = yaml.safe_load(yaml_text)
        prompt_template = doc["transforms"][0]["options"]["prompt_template"]
        assert type(prompt_template) is str
        assert "blob_ref" not in yaml_text
        assert "inline_content" not in yaml_text
        return SimpleNamespace()

    mock_load_settings.side_effect = load_settings
    # Structural fakes (not mocks): instantiate/build_graph/assemble are all
    # patched, so the bundle and graph are threaded through opaquely — a plain
    # SimpleNamespace is honest and keeps the unspecced-mock debt from growing.
    mock_bundle = SimpleNamespace(
        sources={"source": SimpleNamespace()},
        transforms=(),
        sinks={"results": SimpleNamespace()},
        aggregations={},
    )
    mock_instantiate.return_value = mock_bundle
    # spec=ExecutionGraph keeps auto-method behaviour (validate, edge checks…)
    # while satisfying the unspecced-mock discipline gate.
    mock_graph = MagicMock(spec=ExecutionGraph)
    mock_build_graph.return_value = mock_graph

    result = validate_pipeline(
        _state_with_inline_prompt(tmp_path),
        SimpleNamespace(data_dir=tmp_path),
        composer_yaml_generator,
        blob_get_metadata=lambda _blob_id: _ready_blob_record(session_id=session_id),
    )

    assert result.is_valid is True
    mock_load_settings.assert_called_once()
    mock_instantiate.assert_called_once()
    mock_build_graph.assert_called_once()
    mock_assemble.assert_called_once()


@pytest.mark.asyncio
async def test_execution_service_validate_state_passes_blob_metadata_bridge(tmp_path: Path) -> None:
    blob_service = MagicMock(spec=object)
    blob_service.get_blob = AsyncMock(side_effect=BlobNotFoundError(str(BLOB_ID)))
    loop = asyncio.get_running_loop()
    service = ExecutionServiceImpl(
        loop=loop,
        broadcaster=ProgressBroadcaster(loop),
        settings=WebSettings(
            data_dir=tmp_path,
            composer_max_composition_turns=10,
            composer_max_discovery_turns=5,
            composer_timeout_seconds=30.0,
            composer_rate_limit_per_minute=60,
            shareable_link_signing_key=SecretBytes(b"\x00" * 32),
        ),
        session_service=MagicMock(spec=object),
        yaml_generator=composer_yaml_generator,
        telemetry=build_sessions_telemetry(),
        blob_service=blob_service,
    )
    try:
        result = await service.validate_state(_state_with_inline_prompt(tmp_path), user_id="user-1")
    finally:
        await service.shutdown()

    assert result.is_valid is False
    assert any(error.error_code == "missing_inline_blob_content" for error in result.errors)
    blob_service.get_blob.assert_awaited_once_with(BLOB_ID)


@pytest.mark.asyncio
async def test_execution_service_validate_state_treats_cross_session_inline_blob_as_missing(tmp_path: Path) -> None:
    requested_session_id = uuid4()
    other_session_id = uuid4()
    blob_service = MagicMock(spec=object)
    blob_service.get_blob = AsyncMock(return_value=_ready_blob_record(session_id=other_session_id))
    loop = asyncio.get_running_loop()
    service = ExecutionServiceImpl(
        loop=loop,
        broadcaster=ProgressBroadcaster(loop),
        settings=WebSettings(
            data_dir=tmp_path,
            composer_max_composition_turns=10,
            composer_max_discovery_turns=5,
            composer_timeout_seconds=30.0,
            composer_rate_limit_per_minute=60,
            shareable_link_signing_key=SecretBytes(b"\x00" * 32),
        ),
        session_service=MagicMock(spec=object),
        yaml_generator=composer_yaml_generator,
        telemetry=build_sessions_telemetry(),
        blob_service=blob_service,
    )
    try:
        result = await service.validate_state(
            _state_with_inline_prompt(tmp_path),
            user_id="user-1",
            session_id=requested_session_id,
        )
    finally:
        await service.shutdown()

    assert result.is_valid is False
    assert any(error.error_code == "missing_inline_blob_content" for error in result.errors)
    blob_service.get_blob.assert_awaited_once_with(BLOB_ID)
