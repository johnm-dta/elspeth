"""Phase P5.2 — run_signoff_checkpoint delegates to _run_advisor_checkpoint(phase='end')."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from elspeth.web.catalog.protocol import CatalogService
from elspeth.web.catalog.schemas import PluginSchemaInfo, PluginSummary
from elspeth.web.composer.audit import BufferingRecorder
from elspeth.web.composer.service import (
    _ADVISOR_UNAVAILABLE_USER_DETAIL,
    AdvisorCheckpointVerdict,
    ComposerServiceImpl,
)
from elspeth.web.composer.state import (
    CompositionState,
    NodeSpec,
    OutputSpec,
    PipelineMetadata,
    SourceSpec,
)
from elspeth.web.config import WebSettings


def _mock_catalog() -> MagicMock:
    catalog = MagicMock(spec=CatalogService)
    catalog.list_sources.return_value = [
        PluginSummary(name="csv", description="CSV", plugin_type="source", config_fields=[]),
    ]
    catalog.list_transforms.return_value = []
    catalog.list_sinks.return_value = []
    catalog.get_schema.return_value = PluginSchemaInfo(
        name="csv",
        plugin_type="source",
        description="CSV source",
        json_schema={"type": "object", "properties": {}},
        knob_schema={"fields": []},
    )
    return catalog


def _make_settings() -> WebSettings:
    return WebSettings(
        data_dir=Path("/data"),
        composer_max_composition_turns=15,
        composer_max_discovery_turns=10,
        composer_timeout_seconds=85.0,
        composer_rate_limit_per_minute=10,
        composer_advisor_max_calls_per_compose=4,
        composer_advisor_timeout_seconds=60.0,
        shareable_link_signing_key=b"\x00" * 32,
    )


def _state() -> CompositionState:
    return CompositionState(
        source=SourceSpec(plugin="csv", on_success="rows", options={"path": "in.csv"}, on_validation_failure="discard"),
        nodes=(
            NodeSpec(
                id="rate",
                node_type="transform",
                plugin="llm",
                input="rows",
                on_success="rated",
                on_error=None,
                options={"model": "gpt-5.5"},
                condition=None,
                routes=None,
                fork_to=None,
                branches=None,
                policy=None,
                merge=None,
            ),
        ),
        edges=(),
        outputs=(OutputSpec(name="rated", plugin="csv", options={"path": "out.csv"}, on_write_failure="discard"),),
        metadata=PipelineMetadata(),
        version=2,
    )


@pytest.mark.asyncio
async def test_run_signoff_delegates_to_end_checkpoint() -> None:
    service = ComposerServiceImpl(catalog=_mock_catalog(), settings=_make_settings())
    verdict = AdvisorCheckpointVerdict(ok=True, blocking=False, findings_text="CLEAN: looks good")
    service._run_advisor_checkpoint = AsyncMock(return_value=verdict)
    recorder = BufferingRecorder()

    async def sink(event: object) -> None:
        return None

    out = await service.run_signoff_checkpoint(state=_state(), session_id="s1", recorder=recorder, progress=sink)

    assert out is verdict
    service._run_advisor_checkpoint.assert_awaited_once()
    kwargs = service._run_advisor_checkpoint.await_args.kwargs
    assert kwargs["phase"] == "end"
    assert kwargs["session_id"] == "s1"
    assert kwargs["recorder"] is recorder
    assert kwargs["progress"] is sink


@pytest.mark.asyncio
async def test_run_signoff_progress_defaults_none() -> None:
    service = ComposerServiceImpl(catalog=_mock_catalog(), settings=_make_settings())
    service._run_advisor_checkpoint = AsyncMock(
        return_value=AdvisorCheckpointVerdict(
            ok=False,
            blocking=False,
            findings_text=_ADVISOR_UNAVAILABLE_USER_DETAIL,
        )
    )
    await service.run_signoff_checkpoint(state=_state(), session_id=None, recorder=None)
    assert service._run_advisor_checkpoint.await_args.kwargs["progress"] is None
