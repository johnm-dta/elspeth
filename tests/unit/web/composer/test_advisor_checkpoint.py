"""Tests for the deterministic advisor checkpoint runner (Task 4).

Covers the backend-initiated checkpoint primitives:
- ``_run_advisor_checkpoint`` builds phase-specific arguments, reuses the
  audited ``_call_advisor_with_audit`` call, and maps the guidance to an
  :class:`AdvisorCheckpointVerdict` (FLAGGED => blocking, CLEAN => not).
- A CLEAN-prefixed sign-off yields a non-blocking verdict.
- An advisor call that keeps failing yields ``ok=False`` (unavailable) after
  the bounded retry, never raising.

Only ``_call_advisor_with_audit`` is mocked; ``_build_checkpoint_arguments``
and ``_summarize_pipeline_for_advisor`` run for real against ``simple_state``.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from elspeth.web.catalog.protocol import CatalogService
from elspeth.web.catalog.schemas import PluginSchemaInfo, PluginSummary
from elspeth.web.composer.audit import BufferingRecorder
from elspeth.web.composer.service import AdvisorCheckpointVerdict, ComposerServiceImpl
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


def make_recorder() -> BufferingRecorder:
    """Module-level helper (NOT a fixture): a fresh in-flight recorder."""
    return BufferingRecorder()


@pytest.fixture
def make_service() -> object:
    """Return a zero-arg factory producing a wired ``ComposerServiceImpl``."""

    def _factory() -> ComposerServiceImpl:
        return ComposerServiceImpl(catalog=_mock_catalog(), settings=_make_settings())

    return _factory


@pytest.fixture
def simple_state() -> CompositionState:
    """A small but non-trivial pipeline so the summary renderer is exercised."""
    source = SourceSpec(
        plugin="csv",
        on_success="rows",
        options={"path": "input.csv"},
        on_validation_failure="discard",
    )
    node = NodeSpec(
        id="rate",
        node_type="transform",
        plugin="llm",
        input="rows",
        on_success="rated",
        on_error=None,
        options={"required_input_fields": ["url"], "model": "gpt-5.5"},
        condition=None,
        routes=None,
        fork_to=None,
        branches=None,
        policy=None,
        merge=None,
    )
    output = OutputSpec(
        name="rated",
        plugin="csv",
        options={"path": "out.csv"},
        on_write_failure="discard",
    )
    return CompositionState(
        source=source,
        nodes=(node,),
        edges=(),
        outputs=(output,),
        metadata=PipelineMetadata(),
        version=2,
    )


@pytest.fixture
def empty_state() -> CompositionState:
    """A structurally empty pipeline (source/nodes/outputs all absent)."""
    return CompositionState(
        source=None,
        nodes=(),
        edges=(),
        outputs=(),
        metadata=PipelineMetadata(),
        version=1,
    )


@pytest.fixture
def nonempty_state(simple_state) -> CompositionState:
    """A structurally non-empty pipeline (reuse ``simple_state``)."""
    return simple_state


@pytest.mark.asyncio
async def test_early_checkpoint_runs_on_transition_and_injects(make_service, empty_state, nonempty_state):
    service = make_service()
    service._run_advisor_checkpoint = AsyncMock(
        return_value=AdvisorCheckpointVerdict(ok=True, blocking=True, findings_text="Consider a field_mapper before the sink")
    )
    llm_messages: list[dict[str, object]] = []
    ran = await service._maybe_run_early_checkpoint(
        state=nonempty_state,
        prev_state=empty_state,
        session_id="s1",
        llm_messages=llm_messages,
        recorder=make_recorder(),
    )
    assert ran is True
    assert any("field_mapper" in m["content"] for m in llm_messages if m["role"] == "user")


@pytest.mark.asyncio
async def test_early_checkpoint_skips_when_pipeline_already_nonempty(make_service, nonempty_state):
    service = make_service()
    service._run_advisor_checkpoint = AsyncMock()
    ran = await service._maybe_run_early_checkpoint(
        state=nonempty_state,
        prev_state=nonempty_state,
        session_id="s1",
        llm_messages=[],
        recorder=make_recorder(),
    )
    assert ran is False
    service._run_advisor_checkpoint.assert_not_awaited()


@pytest.mark.asyncio
async def test_early_checkpoint_degrades_on_failure(make_service, empty_state, nonempty_state):
    service = make_service()
    service._run_advisor_checkpoint = AsyncMock(
        return_value=AdvisorCheckpointVerdict(ok=False, blocking=False, findings_text="unavailable")
    )
    llm_messages: list[dict[str, object]] = []
    ran = await service._maybe_run_early_checkpoint(
        state=nonempty_state,
        prev_state=empty_state,
        session_id="s1",
        llm_messages=llm_messages,
        recorder=make_recorder(),
    )
    assert ran is True  # attempted
    assert llm_messages == []  # nothing injected; degraded silently


@pytest.mark.asyncio
async def test_run_advisor_checkpoint_end_returns_verdict(make_service, simple_state):
    service = make_service()
    service._call_advisor_with_audit = AsyncMock(return_value=("FLAGGED: the sink drops the rating field", {}))
    verdict = await service._run_advisor_checkpoint(
        phase="end",
        state=simple_state,
        session_id="s1",
        recorder=make_recorder(),
    )
    assert isinstance(verdict, AdvisorCheckpointVerdict)
    assert verdict.ok is True
    assert verdict.blocking is True
    assert "rating field" in verdict.findings_text
    # The synthesized trigger is the backend-only end trigger.
    args = service._call_advisor_with_audit.call_args.args[0]
    assert args["trigger"] == "deterministic_end_checkpoint"
    # The summary carries topology + the field contract so the advisor can
    # actually evaluate the pipeline, not just see node ids.
    excerpt = args["schema_excerpt"]
    assert "rate" in excerpt  # node id
    assert "requires: url" in excerpt  # declared field contract
    assert "model=gpt-5.5" in excerpt  # intent-bearing option value surfaced


@pytest.mark.asyncio
async def test_summarize_renders_intent_values_but_redacts_secret_shaped_keys(simple_state):
    """The summary surfaces allowlisted intent-bearing option VALUES while
    leaving non-allowlisted (potentially secret) keys as names only.
    """
    from elspeth.web.composer.service import _summarize_pipeline_for_advisor
    from elspeth.web.composer.state import NodeSpec

    leaky_node = NodeSpec(
        id="rate",
        node_type="transform",
        plugin="llm",
        input="rows",
        on_success="rated",
        on_error=None,
        options={"model": "gpt-5.5", "api_key": "sk-SECRET-VALUE"},
        condition=None,
        routes=None,
        fork_to=None,
        branches=None,
        policy=None,
        merge=None,
    )
    state = simple_state.with_node(leaky_node)
    summary = _summarize_pipeline_for_advisor(state)
    assert "model=gpt-5.5" in summary  # allowlisted value rendered
    assert "sk-SECRET-VALUE" not in summary  # secret value NEVER rendered
    assert "api_key" in summary  # but its presence is disclosed by name


@pytest.mark.asyncio
async def test_run_advisor_checkpoint_clean_verdict(make_service, simple_state):
    service = make_service()
    service._call_advisor_with_audit = AsyncMock(return_value=("CLEAN: intent satisfied, contracts consistent", {}))
    verdict = await service._run_advisor_checkpoint(phase="end", state=simple_state, session_id="s1", recorder=make_recorder())
    assert verdict.ok is True and verdict.blocking is False


@pytest.mark.asyncio
async def test_run_advisor_checkpoint_unavailable_after_retries(make_service, simple_state):
    service = make_service()
    service._call_advisor_with_audit = AsyncMock(side_effect=TimeoutError())
    verdict = await service._run_advisor_checkpoint(phase="end", state=simple_state, session_id="s1", recorder=make_recorder())
    assert verdict.ok is False  # unavailable
    assert service._call_advisor_with_audit.await_count >= 2  # bounded retry
