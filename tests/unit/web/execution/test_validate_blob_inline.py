"""Validate-time inline blob checks surface structured results, not 500s."""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock
from uuid import UUID

import pytest
from pydantic import SecretBytes

from elspeth.web.blobs.protocol import BlobNotFoundError
from elspeth.web.composer import yaml_generator as composer_yaml_generator
from elspeth.web.composer.state import CompositionState, NodeSpec, OutputSpec, PipelineMetadata, SourceSpec
from elspeth.web.config import WebSettings
from elspeth.web.execution.progress import ProgressBroadcaster
from elspeth.web.execution.service import ExecutionServiceImpl
from elspeth.web.execution.validation import validate_pipeline
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


@pytest.mark.asyncio
async def test_execution_service_validate_state_passes_blob_metadata_bridge(tmp_path: Path) -> None:
    blob_service = MagicMock()
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
        session_service=MagicMock(),
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
