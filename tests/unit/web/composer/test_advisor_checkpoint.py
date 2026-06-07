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
from elspeth.web.execution.schemas import ValidationReadiness, ValidationResult


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


# ---------------------------------------------------------------------------
# Task 6: END authoritative gate (re-review loop; fail-closed; separate budget).
# ---------------------------------------------------------------------------


class _AssistantMessage:
    """Minimal assistant message — the gate only reads ``.content``."""

    content = "Done — the pipeline is ready."


@pytest.fixture
def clean_runnable_state(simple_state) -> CompositionState:
    """A runnable pipeline whose orphan pre-check passes.

    The orphan pre-check (``_missing_pending_interpretation_review_sites``)
    is a SERVICE method, not state — ``drive_try_terminate`` stubs it on the
    service instance so the pre-check returns empty and the end gate runs.
    """
    return simple_state


async def drive_try_terminate(
    service,
    state: CompositionState,
    *,
    advisor_checkpoint_passes_used: int,
    llm_messages: list[dict[str, object]] | None = None,
):
    """Drive ``_try_terminate_no_tools`` with the full kwarg set.

    Stubs the SERVICE-level orphan pre-check to return empty (so the end
    gate runs) and the shared finalize tail to return a canned runnable
    result (so the clean fall-through is isolated from finalize plumbing).
    """
    from elspeth.web.composer.protocol import ComposerResult

    service._missing_pending_interpretation_review_sites = AsyncMock(return_value=())
    service._surface_and_finalize_no_tools = AsyncMock(return_value=ComposerResult(message="Done — the pipeline is ready.", state=state))
    # The advisor-blocked terminal returns now run the surface+orphan-gate pair
    # (``_surface_pt_and_gate_orphans_or_none``) before building the blocked
    # result. These tests isolate the ADVISOR verdict logic, so stub the pair to
    # "no orphan" (return None) — its real behaviour is covered by the
    # interpretation-review-dispatch suite. Without the stub it would call the
    # real ``_auto_surface_prompt_template_reviews`` -> ``_require_sessions_service``
    # which is intentionally unwired in this advisor-focused harness.
    service._surface_pt_and_gate_orphans_or_none = AsyncMock(return_value=None)
    # The END advisor gate only reviews a mechanically valid pipeline: the Fix 2
    # preflight-repair gate runs BEFORE it and would intercept a preflight-invalid
    # state. These tests exercise the ADVISOR, so stub the runtime preflight valid
    # to establish that precondition (the preflight gate is covered separately).
    service._runtime_preflight = lambda candidate, user_id=None: ValidationResult(
        is_valid=True,
        checks=[],
        errors=[],
        readiness=ValidationReadiness(authoring_valid=True, execution_ready=True, completion_ready=True, blockers=[]),
    )
    return await service._try_terminate_no_tools(
        assistant_message=_AssistantMessage(),
        message="rate how cool the pages are",
        llm_messages=[] if llm_messages is None else llm_messages,
        state=state,
        session_id="s1",
        current_state_id="cs1",
        initial_version=1,
        user_id="alice",
        last_runtime_preflight=None,
        runtime_preflight_cache=service._new_runtime_preflight_cache(),
        session_scope="s1",
        mutation_success_seen=True,
        recorder=make_recorder(),
        progress=None,
        repair_turns_used=0,
        persisted_assistant_message_id=None,
        persisted_tool_call_turn=False,
        advisor_checkpoint_passes_used=advisor_checkpoint_passes_used,
    )


@pytest.mark.asyncio
async def test_end_gate_clean_proceeds_to_finalize(make_service, clean_runnable_state):
    service = make_service()
    service._run_advisor_checkpoint = AsyncMock(return_value=AdvisorCheckpointVerdict(ok=True, blocking=False, findings_text="CLEAN"))
    outcome = await drive_try_terminate(service, clean_runnable_state, advisor_checkpoint_passes_used=0)
    assert outcome.action == "return"
    assert outcome.result.runtime_preflight is None or outcome.result.runtime_preflight.is_valid


@pytest.mark.asyncio
async def test_end_gate_flagged_with_budget_repairs(make_service, clean_runnable_state):
    service = make_service()
    service._run_advisor_checkpoint = AsyncMock(
        return_value=AdvisorCheckpointVerdict(ok=True, blocking=True, findings_text="FLAGGED: sink omits rating")
    )
    llm_messages: list[dict[str, object]] = []
    outcome = await drive_try_terminate(service, clean_runnable_state, advisor_checkpoint_passes_used=0, llm_messages=llm_messages)
    assert outcome.action == "continue"
    assert outcome.advisor_passes_delta == 1
    assert any("FLAGGED" in m["content"] for m in llm_messages)


@pytest.mark.asyncio
async def test_end_gate_flagged_on_last_pass_fails_closed(make_service, clean_runnable_state):
    service = make_service()  # composer_advisor_checkpoint_max_passes default 2
    service._run_advisor_checkpoint = AsyncMock(
        return_value=AdvisorCheckpointVerdict(ok=True, blocking=True, findings_text="FLAGGED: still wrong")
    )
    # advisor_checkpoint_passes_used=1 -> next pass is the last (default max=2).
    outcome = await drive_try_terminate(service, clean_runnable_state, advisor_checkpoint_passes_used=1)
    assert outcome.action == "return"
    assert outcome.result.runtime_preflight.is_valid is False
    assert outcome.result.runtime_preflight.readiness.execution_ready is False


@pytest.mark.asyncio
async def test_end_gate_unavailable_fails_closed(make_service, clean_runnable_state):
    service = make_service()
    service._run_advisor_checkpoint = AsyncMock(
        return_value=AdvisorCheckpointVerdict(ok=False, blocking=False, findings_text="unavailable")
    )
    outcome = await drive_try_terminate(service, clean_runnable_state, advisor_checkpoint_passes_used=0)
    assert outcome.action == "return"
    assert outcome.result.runtime_preflight.is_valid is False


@pytest.mark.asyncio
async def test_advisor_budget_does_not_consume_repair_budget(make_service, clean_runnable_state):
    """Gate-order invariant: a flagged advisor repair-continue increments
    advisor_passes_delta, NOT repair_turns_delta."""
    service = make_service()
    service._run_advisor_checkpoint = AsyncMock(return_value=AdvisorCheckpointVerdict(ok=True, blocking=True, findings_text="FLAGGED"))
    outcome = await drive_try_terminate(service, clean_runnable_state, advisor_checkpoint_passes_used=0)
    assert outcome.action == "continue"
    assert outcome.repair_turns_delta == 0
    assert outcome.advisor_passes_delta == 1


@pytest.mark.asyncio
async def test_end_gate_skips_structurally_empty_state(make_service, empty_state):
    """The end gate does NOT fire on a structurally empty pipeline.

    Mirrors the early pass's empty-state skip: a conversational no-tool
    finalize on a pipeline with no source/nodes/sinks has nothing to sign off
    on, so the advisor authority gate is skipped and the turn falls through to
    the shared finalize tail. (Plan deviation: the plan's illustrative code
    omitted this guard — added symmetric with ``_maybe_run_early_checkpoint``.)
    """
    service = make_service()
    service._run_advisor_checkpoint = AsyncMock(
        return_value=AdvisorCheckpointVerdict(ok=True, blocking=True, findings_text="FLAGGED: no source")
    )
    outcome = await drive_try_terminate(service, empty_state, advisor_checkpoint_passes_used=0)
    service._run_advisor_checkpoint.assert_not_awaited()
    assert outcome.action == "return"
