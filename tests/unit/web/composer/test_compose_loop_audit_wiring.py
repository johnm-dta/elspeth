"""Audit wiring tests for ComposerServiceImpl._compose_loop.

These tests pin the panel-review (2026-05-04) blocker fixes:

- **B1**: ``canonical_json`` failure on the success path used to bypass
  the audit recorder; the sentinel-canonical fallback inside
  :func:`elspeth.web.composer.audit.finish_success` (mirroring
  ``composer_mcp/server.py:715-721``) closes the hole.
- **B2**: the narrow-class re-raise (``AssertionError`` /
  ``MemoryError`` / ``RecursionError`` / ``SystemError``) used to exit
  ``_compose_loop`` without recording. ``dispatch_with_audit`` now
  records ``PLUGIN_CRASH`` before propagating.
- **B3**: end-to-end sequence test that drives a real ``_compose_loop``
  through a SUCCESS → ARG_ERROR → PLUGIN_CRASH dispatch chain and
  asserts every invocation lands on the recorder buffer surfaced via
  :class:`ComposerPluginCrashError.tool_invocations`.

The tests use the same ``FakeLLMResponse`` shape and ``execute_tool``
patching pattern established in ``test_service.py`` so the mocking
discipline stays consistent — no new test infrastructure is invented.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import rfc8785

from elspeth.contracts.composer_audit import ComposerToolStatus
from elspeth.web.catalog.protocol import CatalogService
from elspeth.web.catalog.schemas import PluginSchemaInfo, PluginSummary
from elspeth.web.composer.audit import build_canonicalization_sentinel
from elspeth.web.composer.protocol import (
    ComposerConvergenceError,
    ComposerPluginCrashError,
    ComposerRuntimePreflightError,
    ToolArgumentError,
)
from elspeth.web.composer.service import ComposerAvailability, ComposerServiceImpl
from elspeth.web.composer.state import (
    CompositionState,
    PipelineMetadata,
    ValidationSummary,
)
from elspeth.web.composer.tools import ToolResult
from elspeth.web.config import WebSettings
from elspeth.web.execution.schemas import ValidationReadiness, ValidationResult

# ---------------------------------------------------------------------------
# Test doubles — mirror the shapes used by tests/unit/web/composer/test_service.py
# so the mocking discipline is identical.
# ---------------------------------------------------------------------------


@dataclass
class _FakeFunction:
    name: str
    arguments: str


@dataclass
class _FakeToolCall:
    id: str
    function: _FakeFunction


@dataclass
class _FakeMessage:
    content: str | None
    tool_calls: list[_FakeToolCall] | None


@dataclass
class _FakeChoice:
    message: _FakeMessage


@dataclass
class _FakeLLMResponse:
    choices: list[_FakeChoice]


def _empty_state() -> CompositionState:
    return CompositionState(
        source=None,
        nodes=(),
        edges=(),
        outputs=(),
        metadata=PipelineMetadata(),
        version=1,
    )


def _passing_preflight() -> ValidationResult:
    return ValidationResult(
        is_valid=True,
        checks=[],
        errors=[],
        readiness=ValidationReadiness(authoring_valid=True, execution_ready=True, completion_ready=True, blockers=[]),
    )


def _mock_catalog() -> MagicMock:
    catalog = MagicMock(spec=CatalogService)
    catalog.list_sources.return_value = [
        PluginSummary(
            name="csv",
            description="CSV source",
            plugin_type="source",
            config_fields=[],
        ),
    ]
    catalog.list_transforms.return_value = []
    catalog.list_sinks.return_value = []
    catalog.get_schema.return_value = PluginSchemaInfo(
        name="csv",
        plugin_type="source",
        description="CSV source",
        json_schema={"title": "Config", "properties": {}},
        knob_schema={"fields": []},
    )
    return catalog


def _make_settings(**overrides: Any) -> WebSettings:
    defaults: dict[str, Any] = {
        "data_dir": Path("/data"),
        "composer_max_composition_turns": 15,
        "composer_max_discovery_turns": 10,
        "composer_timeout_seconds": 85.0,
        "composer_rate_limit_per_minute": 10,
        "shareable_link_signing_key": b"\x00" * 32,
    }
    defaults.update(overrides)
    return WebSettings(**defaults)


def _make_llm_response(
    content: str | None = None,
    tool_calls: list[dict[str, Any]] | None = None,
) -> _FakeLLMResponse:
    fake_tool_calls: list[_FakeToolCall] | None = None
    if tool_calls:
        fake_tool_calls = [
            _FakeToolCall(
                id=tc["id"],
                function=_FakeFunction(
                    name=tc["name"],
                    arguments=json.dumps(tc["arguments"]),
                ),
            )
            for tc in tool_calls
        ]
    return _FakeLLMResponse(choices=[_FakeChoice(message=_FakeMessage(content=content, tool_calls=fake_tool_calls))])


@pytest.fixture(autouse=True)
def _composer_available_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """Skip the boot-time API key check — these tests target compose behavior."""

    def _available(self: ComposerServiceImpl) -> ComposerAvailability:
        return ComposerAvailability(available=True, model=self._model, provider="test")

    monkeypatch.setattr(ComposerServiceImpl, "_compute_availability", _available)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_compose_loop_records_success_arg_error_plugin_crash_sequence() -> None:
    """End-to-end dispatch sequence: SUCCESS -> ARG_ERROR -> PLUGIN_CRASH.

    This is the canary test for the structural ``dispatch_with_audit``
    helper. Drives ``_compose_loop`` through three turns where
    ``execute_tool`` returns / raises in turn:

    1. SUCCESS — ToolResult with mutated state
    2. ARG_ERROR — ``ToolArgumentError`` raised by handler
    3. PLUGIN_CRASH — bare ``RuntimeError`` raised by handler

    Asserts:

    - ``ComposerPluginCrashError`` propagates out of ``compose()``
    - ``exc.tool_invocations`` carries exactly three records, in order
    - status / version_after / error_class line up with the dispatch
      semantics: SUCCESS bumps the version, ARG_ERROR and PLUGIN_CRASH
      both record ``version_after is None``.

    The buffer is read AFTER ``pytest.raises(...)`` exits so the
    ordering check reflects the dispatch sequence as the loop saw it,
    not a snapshot taken mid-flight.
    """
    catalog = _mock_catalog()
    settings = _make_settings()
    service = ComposerServiceImpl(catalog=catalog, settings=settings)
    state = _empty_state()

    # Three LLM turns — one tool_call each. The arguments here only need
    # to satisfy schema-required-paths validation for set_metadata
    # (whose only required key is ``patch``); the test patches
    # ``execute_tool`` for the actual handler behaviour.
    turn1 = _make_llm_response(
        tool_calls=[
            {
                "id": "call_success",
                "name": "set_metadata",
                "arguments": {"patch": {"name": "Step One"}},
            }
        ],
    )
    turn2 = _make_llm_response(
        tool_calls=[
            {
                "id": "call_arg_error",
                "name": "set_metadata",
                "arguments": {"patch": {"name": "Step Two"}},
            }
        ],
    )
    turn3 = _make_llm_response(
        tool_calls=[
            {
                "id": "call_plugin_crash",
                "name": "set_metadata",
                "arguments": {"patch": {"name": "Step Three"}},
            }
        ],
    )

    # SUCCESS: build a ToolResult that bumps the state version 1 -> 2.
    mutated_state = replace(state, version=2)
    success_result = ToolResult(
        success=True,
        updated_state=mutated_state,
        validation=ValidationSummary(
            is_valid=True,
            errors=(),
            warnings=(),
            suggestions=(),
            semantic_contracts=(),
        ),
        affected_nodes=(),
    )

    with (
        patch.object(service, "_call_llm", new_callable=AsyncMock) as mock_llm,
        patch(
            "elspeth.web.composer.tool_batch.execute_tool",
            side_effect=[
                success_result,
                ToolArgumentError(
                    argument="patch",
                    expected="a string",
                    actual_type="int",
                ),
                RuntimeError("synthetic plugin bug"),
            ],
        ),
    ):
        mock_llm.side_effect = [turn1, turn2, turn3]
        with pytest.raises(ComposerPluginCrashError) as exc_info:
            await service.compose("Drive the sequence", [], state)

    invocations = exc_info.value.tool_invocations
    assert len(invocations) == 3, (
        f"Expected 3 audit invocations (SUCCESS, ARG_ERROR, PLUGIN_CRASH); got {len(invocations)}: {[inv.status for inv in invocations]}"
    )

    # SUCCESS — version advanced from 1 to 2
    success_inv = invocations[0]
    assert success_inv.status == ComposerToolStatus.SUCCESS
    assert success_inv.tool_call_id == "call_success"
    assert success_inv.version_before == 1
    assert success_inv.version_after == 2
    assert success_inv.version_after is not None
    assert success_inv.version_after > success_inv.version_before
    assert success_inv.error_class is None

    # ARG_ERROR — version_after must be None (dispatch did not complete)
    arg_error_inv = invocations[1]
    assert arg_error_inv.status == ComposerToolStatus.ARG_ERROR
    assert arg_error_inv.tool_call_id == "call_arg_error"
    assert arg_error_inv.version_after is None
    assert arg_error_inv.error_class == "ToolArgumentError"

    # PLUGIN_CRASH — version_after None, error_class is the original
    # exception class. error_message MUST be class-name only (redaction
    # discipline; pin against future drift that would echo str(exc)).
    plugin_crash_inv = invocations[2]
    assert plugin_crash_inv.status == ComposerToolStatus.PLUGIN_CRASH
    assert plugin_crash_inv.tool_call_id == "call_plugin_crash"
    assert plugin_crash_inv.version_after is None
    assert plugin_crash_inv.error_class == "RuntimeError"
    assert plugin_crash_inv.error_message == "RuntimeError"

    # Tool-call ordering reflects the dispatch sequence as the loop saw it.
    assert [inv.tool_call_id for inv in invocations] == [
        "call_success",
        "call_arg_error",
        "call_plugin_crash",
    ]


@pytest.mark.asyncio
async def test_compose_loop_records_assertion_error_before_reraise() -> None:
    """Narrow re-raise (AssertionError) records PLUGIN_CRASH before propagating.

    Closes panel-review B2: prior to the ``dispatch_with_audit`` helper
    extraction, the narrow ``except (AssertionError, MemoryError,
    RecursionError, SystemError)`` block re-raised without recording.
    The structural helper now records the dispatch as PLUGIN_CRASH
    before re-raise — building a frozen dataclass from pre-captured
    scalars is not poisoned-memory work.

    AssertionError propagates as ``AssertionError`` (NOT wrapped in
    ``ComposerPluginCrashError``) — the policy that interpreter-level
    invariant breaches MUST NOT be laundered is preserved.
    """
    catalog = _mock_catalog()
    settings = _make_settings()
    service = ComposerServiceImpl(catalog=catalog, settings=settings)
    state = _empty_state()

    turn = _make_llm_response(
        tool_calls=[
            {
                "id": "call_assert",
                "name": "set_metadata",
                "arguments": {"patch": {"name": "Tier-1 invariant"}},
            }
        ],
    )

    # Capture the recorder buffer before the AssertionError exits the
    # loop. The narrow re-raise path does NOT route through
    # ComposerPluginCrashError.capture, so the buffer is not surfaced
    # on a partial-state carrier — we read it directly from the
    # BufferingRecorder via a service-level patch that intercepts the
    # recorder construction.
    captured_recorder: dict[str, Any] = {}
    real_buffering_recorder = (
        # Imported lazily so the module patch is the seam, not the symbol
        # in service.py — exact mirror of how production wires it.
        __import__("elspeth.web.composer.audit", fromlist=["BufferingRecorder"]).BufferingRecorder
    )

    class _SpyRecorder(real_buffering_recorder):  # type: ignore[misc, valid-type]
        def __init__(self) -> None:
            super().__init__()
            captured_recorder["instance"] = self

    with (
        patch.object(service, "_call_llm", new_callable=AsyncMock) as mock_llm,
        patch(
            "elspeth.web.composer.tool_batch.execute_tool",
            side_effect=AssertionError("Tier-1 invariant breach"),
        ),
        patch("elspeth.web.composer.service.BufferingRecorder", _SpyRecorder),
    ):
        mock_llm.return_value = turn
        with pytest.raises(AssertionError):
            await service.compose("Trigger invariant", [], state)

    spy = captured_recorder["instance"]
    invocations = spy.invocations
    assert len(invocations) == 1
    inv = invocations[0]
    assert inv.status == ComposerToolStatus.PLUGIN_CRASH
    assert inv.error_class == "AssertionError"
    assert inv.error_message == "AssertionError"
    assert inv.version_after is None


@pytest.mark.asyncio
async def test_compose_loop_crashes_when_success_canonical_json_fails() -> None:
    """Un-canonicalizable SUCCESS-path output is OUR bug — crash, don't launder.

    ``finish_success`` canonicalizes ``result_payload`` — the output of
    a composer tool handler (``ToolResult.to_dict()``), which is
    first-party authored code building structured catalog/state data.
    The composer-LLM authors tool *arguments* (handled on the ARG_ERROR
    path), never the handler's *result*. So a non-finite float or other
    non-JSON-serializable value in that payload means one of *our*
    handlers produced un-canonicalizable output — a bug in our code, not
    malformed external data.

    Per the trust model, ``canonical_json`` on our own dispatch output is
    a Tier-1-equivalent act: it must crash on anomaly. The earlier
    "sentinel-canonical fallback" substituted a degraded sentinel and
    reported ``status=SUCCESS``, laundering our bug into a clean-looking
    audit row — a confident wrong answer to an auditor. That fallback was
    removed; the ``ValueError`` from ``rfc8785.dumps`` (non-finite float,
    RFC 8785 §3.1) now escapes ``finish_success``. The dispatch
    plugin-crash machinery captures it (with the full partial-state /
    tool-invocation story, so no audit hole) and surfaces it as a
    ``ComposerPluginCrashError`` whose ``__cause__`` is the ``ValueError`` —
    a loud, auditable crash rather than a fabricated SUCCESS row.

    This test substitutes a ToolResult subclass whose ``to_dict()``
    returns a float ``inf`` and confirms the loop crashes rather than
    recording a sentinel SUCCESS row.
    """
    catalog = _mock_catalog()
    settings = _make_settings()
    service = ComposerServiceImpl(catalog=catalog, settings=settings)
    state = _empty_state()

    mutated_state = replace(state, version=2)

    # Subclass ToolResult so the loop's ``state = result.updated_state``
    # rebind still works, but ``to_dict()`` returns a payload that
    # canonical_json refuses (non-finite float).
    class _NonCanonicalizableResult(ToolResult):
        def to_dict(self) -> dict[str, Any]:
            return {
                "success": True,
                "version": self.updated_state.version,
                # rfc8785 raises on non-finite floats; canonical_json on
                # our own dispatch output crashes rather than laundering.
                "non_finite": float("inf"),
            }

    bad_result = _NonCanonicalizableResult(
        success=True,
        updated_state=mutated_state,
        validation=ValidationSummary(
            is_valid=True,
            errors=(),
            warnings=(),
            suggestions=(),
            semantic_contracts=(),
        ),
        affected_nodes=(),
    )

    turn1 = _make_llm_response(
        tool_calls=[
            {
                "id": "call_with_non_finite",
                "name": "set_metadata",
                "arguments": {"patch": {"name": "Non-finite payload"}},
            }
        ],
    )
    turn2 = _make_llm_response(content="Done.")

    passing_preflight = _passing_preflight()

    with (
        patch.object(service, "_call_llm", new_callable=AsyncMock) as mock_llm,
        patch.object(service, "_runtime_preflight", return_value=passing_preflight),
        patch(
            "elspeth.web.composer.tool_batch.execute_tool",
            return_value=bad_result,
        ),
        pytest.raises(ComposerPluginCrashError) as exc_info,
    ):
        mock_llm.side_effect = [turn1, turn2]
        await service.compose("Trigger non-finite payload", [], state)

    # The canonicalization failure on our own dispatch output surfaces as
    # a loud crash whose root cause is the rfc8785 ``ValueError`` — never
    # a laundered SUCCESS row.
    assert isinstance(exc_info.value.__cause__, ValueError)


@pytest.mark.asyncio
async def test_timeout_after_successful_tool_carries_audit_invocations() -> None:
    """A wall-clock timeout after a successful tool must preserve its audit row.

    The timeout is raised by ``_call_llm_before_deadline`` on the model call
    after the tool result was appended. ``_compose_loop`` owns the
    BufferingRecorder, so both deadline-call sites must pass it through or the
    resulting ``ComposerConvergenceError`` reaches the route layer with no
    tool_invocations to persist.
    """
    catalog = _mock_catalog()
    settings = _make_settings()
    service = ComposerServiceImpl(catalog=catalog, settings=settings)
    state = _empty_state()

    mutated_state = replace(state, version=2)
    success_result = ToolResult(
        success=True,
        updated_state=mutated_state,
        validation=ValidationSummary(
            is_valid=True,
            errors=(),
            warnings=(),
            suggestions=(),
            semantic_contracts=(),
        ),
        affected_nodes=(),
    )

    turn = _make_llm_response(
        tool_calls=[
            {
                "id": "call_success_before_timeout",
                "name": "set_metadata",
                "arguments": {"patch": {"name": "Timed"}},
            }
        ],
    )

    calls = {"count": 0}

    async def first_tool_then_timeout_llm(*_args: Any, **_kwargs: Any) -> _FakeLLMResponse:
        calls["count"] += 1
        if calls["count"] == 1:
            return turn
        raise TimeoutError

    with (
        patch.object(service, "_call_llm", new=first_tool_then_timeout_llm),
        patch(
            "elspeth.web.composer.tool_batch.execute_tool",
            return_value=success_result,
        ) as mock_execute_tool,
        pytest.raises(ComposerConvergenceError) as exc_info,
    ):
        await service.compose("Timeout after the tool", [], state)

    assert exc_info.value.budget_exhausted == "timeout"
    assert calls["count"] == 2
    assert mock_execute_tool.call_count == 1
    invocations = exc_info.value.tool_invocations
    assert len(invocations) == 1
    inv = invocations[0]
    assert inv.status == ComposerToolStatus.SUCCESS
    assert inv.tool_call_id == "call_success_before_timeout"
    assert inv.version_after == 2


@pytest.mark.asyncio
async def test_preview_runtime_preflight_failure_records_tool_invocation() -> None:
    """preview_pipeline preflight crashes must still audit the tool call.

    The preview runtime preflight runs after the audit envelope opens but
    before ``dispatch_with_audit``. If it raises, the resulting
    ``ComposerRuntimePreflightError`` must carry a PLUGIN_CRASH
    invocation for the preview tool call that caused the failure.
    """
    catalog = _mock_catalog()
    settings = _make_settings()
    service = ComposerServiceImpl(catalog=catalog, settings=settings)
    state = _empty_state()

    turn = _make_llm_response(
        tool_calls=[
            {
                "id": "call_preview_preflight_crash",
                "name": "preview_pipeline",
                "arguments": {},
            }
        ],
    )

    with (
        patch.object(service, "_call_llm", new_callable=AsyncMock) as mock_llm,
        patch.object(service, "_runtime_preflight", side_effect=RuntimeError("synthetic runtime preflight bug")),
        patch("elspeth.web.composer.tool_batch.execute_tool") as mock_execute_tool,
        pytest.raises(ComposerRuntimePreflightError) as exc_info,
    ):
        mock_llm.return_value = turn
        await service.compose("Preview the current pipeline", [], state)

    mock_execute_tool.assert_not_called()
    invocations = exc_info.value.tool_invocations
    assert len(invocations) == 1
    inv = invocations[0]
    assert inv.status == ComposerToolStatus.PLUGIN_CRASH
    assert inv.tool_call_id == "call_preview_preflight_crash"
    assert inv.tool_name == "preview_pipeline"
    assert inv.error_class == "RuntimeError"
    assert inv.error_message == "RuntimeError"
    assert inv.version_after is None


@pytest.mark.asyncio
async def test_dispatch_records_plugin_crash_on_cancelled_error() -> None:
    """Reviewer fix: ``CancelledError`` MUST be audited before propagation.

    ``asyncio.CancelledError`` inherits from ``BaseException`` (not
    ``Exception``), so the typed handlers inside
    :func:`elspeth.web.composer.audit.dispatch_with_audit` —
    ``except ToolArgumentError``, the narrow PLUGIN_CRASH tuple, the
    generic ``except Exception`` — never catch it. With the original
    try/except shape the helper would return early on success and
    every except branch would record-then-raise; ``CancelledError``
    flowed through all three handlers unrecorded, leaving an audit
    hole whenever an ASGI client disconnected mid-dispatch.

    The ``try/finally`` shape closes the hole: the ``finally`` clause
    runs on every exit (including BaseException propagation),
    reconstructs the propagating exception via :func:`sys.exc_info`,
    and records ``PLUGIN_CRASH`` before the helper unwinds.

    Production-realistic path: the test injects
    ``asyncio.CancelledError`` into the ``execute_tool`` worker call
    (``run_sync_in_worker(execute_tool, ...)``) — the same site where
    cancellation would surface in production when the event loop
    cancels the request task because the client closed the connection.

    Asserts:

    - ``CancelledError`` propagates out of ``compose()`` (NOT wrapped
      in ``ComposerPluginCrashError`` — the typed handlers never
      caught it, so neither does any wrap site).
    - The recorder captured exactly one invocation with
      ``status=PLUGIN_CRASH``, ``error_class="CancelledError"``,
      ``version_after=None``.
    """
    catalog = _mock_catalog()
    settings = _make_settings()
    service = ComposerServiceImpl(catalog=catalog, settings=settings)
    state = _empty_state()

    turn = _make_llm_response(
        tool_calls=[
            {
                "id": "call_cancelled",
                "name": "set_metadata",
                "arguments": {"patch": {"name": "Client disconnect"}},
            }
        ],
    )

    # Same recorder-spy pattern as the AssertionError test: the
    # CancelledError path bypasses ComposerPluginCrashError.capture
    # (the `except Exception` handler in service.py is not active for
    # BaseException subclasses), so we read the buffer directly via
    # the BufferingRecorder spy.
    captured_recorder: dict[str, Any] = {}
    real_buffering_recorder = __import__("elspeth.web.composer.audit", fromlist=["BufferingRecorder"]).BufferingRecorder

    class _SpyRecorder(real_buffering_recorder):  # type: ignore[misc, valid-type]
        def __init__(self) -> None:
            super().__init__()
            captured_recorder["instance"] = self

    with (
        patch.object(service, "_call_llm", new_callable=AsyncMock) as mock_llm,
        patch(
            "elspeth.web.composer.tool_batch.execute_tool",
            side_effect=asyncio.CancelledError(),
        ),
        patch("elspeth.web.composer.service.BufferingRecorder", _SpyRecorder),
    ):
        mock_llm.return_value = turn
        with pytest.raises(asyncio.CancelledError):
            await service.compose("Trigger client disconnect", [], state)

    spy = captured_recorder["instance"]
    invocations = spy.invocations
    assert len(invocations) == 1, f"Expected exactly one PLUGIN_CRASH audit row for the cancelled dispatch; got {len(invocations)}"
    inv = invocations[0]
    assert inv.status == ComposerToolStatus.PLUGIN_CRASH
    assert inv.tool_call_id == "call_cancelled"
    assert inv.error_class == "CancelledError"
    # Redaction discipline: error_message is class-name only.
    assert inv.error_message == "CancelledError"
    # Dispatch did not complete — version_after must be None to
    # satisfy the Tier-1 verifier invariant.
    assert inv.version_after is None


@pytest.mark.asyncio
async def test_compose_loop_records_arg_error_for_non_finite_object_arguments() -> None:
    """Parsed NaN/Infinity inside object arguments must produce ARG_ERROR.

    Python's ``json.loads`` accepts ``NaN``/``Infinity`` constants even
    though canonical JSON rejects them. The compose loop must record a
    corrective ARG_ERROR tool row and continue, rather than raising from
    ``begin_dispatch`` before the recorder fires.
    """
    catalog = _mock_catalog()
    settings = _make_settings()
    service = ComposerServiceImpl(catalog=catalog, settings=settings)
    state = _empty_state()

    turn1 = _make_llm_response(
        tool_calls=[
            {
                "id": "call_non_finite_object",
                "name": "set_metadata",
                "arguments": {"patch": {"name": float("nan")}},
            }
        ],
    )
    turn2 = _make_llm_response(content="Recovered.")
    passing_preflight = _passing_preflight()

    with (
        patch.object(service, "_call_llm", new_callable=AsyncMock) as mock_llm,
        patch.object(service, "_runtime_preflight", return_value=passing_preflight),
        patch("elspeth.web.composer.tool_batch.execute_tool") as mock_execute_tool,
    ):
        mock_llm.side_effect = [turn1, turn2]
        result = await service.compose("Trigger non-finite object arguments", [], state)

    assert result.message == "Recovered."
    mock_execute_tool.assert_not_called()
    assert len(result.tool_invocations) == 1
    inv = result.tool_invocations[0]
    assert inv.status == ComposerToolStatus.ARG_ERROR
    assert inv.tool_call_id == "call_non_finite_object"
    assert inv.error_class == "ValueError"
    assert inv.error_message == "ValueError"
    assert inv.version_after is None

    second_call_messages = mock_llm.call_args_list[1].args[0]
    tool_messages = [msg for msg in second_call_messages if msg["role"] == "tool"]
    assert tool_messages[-1]["tool_call_id"] == "call_non_finite_object"


@pytest.mark.asyncio
async def test_compose_loop_records_arg_error_for_non_finite_non_object_arguments() -> None:
    """Top-level Infinity must use the non-object ARG_ERROR audit path."""
    catalog = _mock_catalog()
    settings = _make_settings()
    service = ComposerServiceImpl(catalog=catalog, settings=settings)
    state = _empty_state()

    turn1 = _make_llm_response(
        tool_calls=[
            {
                "id": "call_non_finite_scalar",
                "name": "set_metadata",
                "arguments": float("inf"),
            }
        ],
    )
    turn2 = _make_llm_response(content="Recovered.")
    passing_preflight = _passing_preflight()

    with (
        patch.object(service, "_call_llm", new_callable=AsyncMock) as mock_llm,
        patch.object(service, "_runtime_preflight", return_value=passing_preflight),
        patch("elspeth.web.composer.tool_batch.execute_tool") as mock_execute_tool,
    ):
        mock_llm.side_effect = [turn1, turn2]
        result = await service.compose("Trigger non-finite scalar arguments", [], state)

    assert result.message == "Recovered."
    mock_execute_tool.assert_not_called()
    assert len(result.tool_invocations) == 1
    inv = result.tool_invocations[0]
    assert inv.status == ComposerToolStatus.ARG_ERROR
    assert inv.tool_call_id == "call_non_finite_scalar"
    assert inv.error_class == "ValueError"
    assert inv.error_message == "ValueError"
    assert inv.version_after is None

    second_call_messages = mock_llm.call_args_list[1].args[0]
    tool_messages = [msg for msg in second_call_messages if msg["role"] == "tool"]
    assert tool_messages[-1]["tool_call_id"] == "call_non_finite_scalar"


# ---------------------------------------------------------------------------
# Discovery-tool audit-payload preservation (elspeth-281f259235).
#
# Pre-fix: ``ToolResult.data`` for the four catalog discovery tools
# carries Pydantic ``PluginSummary`` / ``PluginSchemaInfo`` instances.
# The web composer audit canonicalizes ``ToolResult.to_dict()`` directly
# via :func:`elspeth.core.canonical.canonical_json` →
# :mod:`rfc8785`, which rejects ``BaseModel`` with
# ``CanonicalizationError``. The existing ``except (ValueError,
# TypeError)`` substituted the sentinel, obliterating the result body
# the LLM had just made a decision against — failing the
# attributability test.
#
# Post-fix: :func:`finish_success` runs
# :func:`_normalize_audit_payload` (Pydantic-aware recursion mirroring
# the standalone-MCP ``_ensure_serializable`` pattern) before
# canonicalization, so the audit row preserves the catalog data
# verbatim. The sentinel-canonical fallback still fires for genuinely
# non-canonicalizable payloads (acceptance criterion #3 — pinned by
# ``test_compose_loop_records_success_when_canonical_json_fails``).
#
# Matrix: 4 discovery tools x {cache miss, cache hit} = 8 invocation
# scenarios. The cache-hit path (`service.py:980-992`) hand-builds a
# slim audit dict from raw ``cached_result.data`` — bypassing
# ``ToolResult.to_dict()`` — so the only normalization site that
# catches it is ``finish_success`` itself.
# ---------------------------------------------------------------------------


def _discovery_result(
    tool_name: str,
    state: CompositionState,
    catalog: MagicMock,
) -> ToolResult:
    """Build a real ToolResult mirroring what _handle_list_*/get_plugin_schema produce.

    This goes through the same construction path as the production
    handlers (`tools.py:_discovery_result`) so the freeze-on-data
    discipline (``__post_init__`` calls ``freeze_fields(self, "data")``)
    runs identically and the test exercises the actual audit-ingress
    shape rather than a hand-rolled payload.
    """
    if tool_name == "list_sources":
        data: Any = catalog.list_sources()
    elif tool_name == "list_transforms":
        data = catalog.list_transforms()
    elif tool_name == "list_sinks":
        data = catalog.list_sinks()
    elif tool_name == "get_plugin_schema":
        data = catalog.get_schema("source", "csv")
    else:
        raise AssertionError(f"unexpected discovery tool: {tool_name}")
    return ToolResult(
        success=True,
        updated_state=state,
        validation=ValidationSummary(
            is_valid=True,
            errors=(),
            warnings=(),
            suggestions=(),
            semantic_contracts=(),
        ),
        affected_nodes=(),
        data=data,
    )


def _assert_payload_preserved(payload: dict[str, Any], tool_name: str) -> None:
    """Common assertions for a discovery-tool audit payload.

    Pinned invariants:

    - The sentinel diagnostic key is absent (the actual data made it
      through canonicalization).
    - The payload carries the expected envelope keys for a successful
      discovery (``success``, ``validation``, ``version``, ``data``).
    - The catalog data inside ``payload["data"]`` matches what the
      ``_mock_catalog()`` fixture configured.
    """
    assert "_canonicalization_error" not in payload, (
        f"{tool_name}: sentinel landed where real data was expected — audit primacy violated. Payload: {payload}"
    )
    assert payload["success"] is True
    assert "validation" in payload
    assert "version" in payload
    assert "data" in payload, f"{tool_name}: missing data key in audit payload: {payload}"

    data = payload["data"]
    if tool_name == "list_sources":
        assert isinstance(data, list)
        assert len(data) == 1
        assert data[0]["name"] == "csv"
        assert data[0]["plugin_type"] == "source"
    elif tool_name in ("list_transforms", "list_sinks"):
        assert data == []
    elif tool_name == "get_plugin_schema":
        assert isinstance(data, dict)
        assert data["name"] == "csv"
        assert data["plugin_type"] == "source"
        assert data["json_schema"] == {"title": "Config", "properties": {}}
    else:
        raise AssertionError(f"unexpected tool: {tool_name}")


class TestComposerDiscoveryAuditPreservesResult:
    """elspeth-281f259235: successful discovery rows preserve catalog payload.

    Pre-fix the audit row carried the canonicalization sentinel; the
    LLM saw the real data via the ``_pydantic_default`` callback in
    ``_serialize_tool_result``, but the durable audit trail did not.
    This class pins the invariant per discovery tool, on both the
    cache-miss and cache-hit code paths.
    """

    @pytest.mark.parametrize(
        ("tool_name", "tool_args"),
        [
            ("list_sources", {}),
            ("list_transforms", {}),
            ("list_sinks", {}),
            ("get_plugin_schema", {"plugin_type": "source", "name": "csv"}),
        ],
    )
    @pytest.mark.asyncio
    async def test_cache_miss_audit_preserves_pydantic_payload(self, tool_name: str, tool_args: dict[str, Any]) -> None:
        """A first-time discovery dispatch records the real catalog data.

        Cache miss exercises the regular dispatch path:
        ``execute_tool`` → real ``ToolResult`` →
        ``_result_to_audit_payload`` → ``ToolResult.to_dict()`` →
        ``finish_success`` → ``_normalize_audit_payload`` →
        ``canonical_json``.
        """
        catalog = _mock_catalog()
        settings = _make_settings()
        service = ComposerServiceImpl(catalog=catalog, settings=settings)
        state = _empty_state()

        discovery_result = _discovery_result(tool_name, state, catalog)

        turn1 = _make_llm_response(
            tool_calls=[
                {
                    "id": f"call_{tool_name}",
                    "name": tool_name,
                    "arguments": tool_args,
                }
            ],
        )
        turn2 = _make_llm_response(content="Discovery complete.")

        # Empty state has no source/sinks; bypass the post-loop runtime
        # preflight as in the existing B1 test (orthogonal to the audit
        # invariant we're pinning).
        passing_preflight = _passing_preflight()

        with (
            patch.object(service, "_call_llm", new_callable=AsyncMock) as mock_llm,
            patch.object(service, "_runtime_preflight", return_value=passing_preflight),
            patch(
                "elspeth.web.composer.tool_batch.execute_tool",
                return_value=discovery_result,
            ),
        ):
            mock_llm.side_effect = [turn1, turn2]
            result = await service.compose(f"Run {tool_name}", [], state)

        invocations = result.tool_invocations
        assert len(invocations) == 1, f"{tool_name}: expected exactly one audit row"
        inv = invocations[0]
        assert inv.status == ComposerToolStatus.SUCCESS
        assert inv.tool_call_id == f"call_{tool_name}"
        assert inv.cache_hit is False
        assert inv.result_canonical is not None

        payload = json.loads(inv.result_canonical)
        _assert_payload_preserved(payload, tool_name)

    @pytest.mark.parametrize(
        ("tool_name", "tool_args"),
        [
            ("list_sources", {}),
            ("list_transforms", {}),
            ("list_sinks", {}),
            ("get_plugin_schema", {"plugin_type": "source", "name": "csv"}),
        ],
    )
    @pytest.mark.asyncio
    async def test_cache_hit_audit_preserves_pydantic_payload(self, tool_name: str, tool_args: dict[str, Any]) -> None:
        """Cache-hit replay records the cached catalog data, not the sentinel.

        Cache hit exercises the second dispatch path:
        ``cached_payload = {"success": ..., "data":
        cached_result.data, "cache_hit": True}`` (hand-built in
        ``service.py:980-992``, bypasses ``ToolResult.to_dict()``) →
        ``finish_success`` → ``_normalize_audit_payload`` → ``canonical_json``.

        ``finish_success`` is the single SUCCESS-path audit choke
        point, so normalizing there catches both cache-miss and
        cache-hit paths uniformly. This test pins that invariant
        against any future refactor that might split the two paths.

        Sequence: two compose loop turns each issuing the same
        cacheable tool call, then a final text turn. The first call
        populates ``discovery_cache``; the second hits it.
        """
        catalog = _mock_catalog()
        settings = _make_settings()
        service = ComposerServiceImpl(catalog=catalog, settings=settings)
        state = _empty_state()

        discovery_result = _discovery_result(tool_name, state, catalog)

        # Same arguments dict on both turns → identical cache key.
        turn1 = _make_llm_response(
            tool_calls=[
                {
                    "id": f"call_{tool_name}_first",
                    "name": tool_name,
                    "arguments": tool_args,
                }
            ],
        )
        turn2 = _make_llm_response(
            tool_calls=[
                {
                    "id": f"call_{tool_name}_replay",
                    "name": tool_name,
                    "arguments": tool_args,
                }
            ],
        )
        turn3 = _make_llm_response(content="Cached discovery complete.")

        passing_preflight = _passing_preflight()

        with (
            patch.object(service, "_call_llm", new_callable=AsyncMock) as mock_llm,
            patch.object(service, "_runtime_preflight", return_value=passing_preflight),
            patch(
                "elspeth.web.composer.tool_batch.execute_tool",
                return_value=discovery_result,
            ) as mock_execute_tool,
        ):
            mock_llm.side_effect = [turn1, turn2, turn3]
            result = await service.compose(f"Cache {tool_name}", [], state)

        # Cache hit means execute_tool was called only once across both
        # turns — the second turn served the result from
        # discovery_cache without re-dispatching to the handler.
        assert mock_execute_tool.call_count == 1, (
            f"{tool_name}: expected exactly one execute_tool call across "
            f"both turns (second served from cache); got {mock_execute_tool.call_count}"
        )

        invocations = result.tool_invocations
        assert len(invocations) == 2, f"{tool_name}: expected two audit rows (cache miss + hit)"

        miss_inv, hit_inv = invocations
        assert miss_inv.status == ComposerToolStatus.SUCCESS
        assert miss_inv.cache_hit is False
        assert hit_inv.status == ComposerToolStatus.SUCCESS
        assert hit_inv.cache_hit is True

        # Both audit rows must carry the real data — pre-fix the
        # cache-hit row would have landed the sentinel because the
        # hand-built payload bypassed any normalization.
        assert miss_inv.result_canonical is not None
        assert hit_inv.result_canonical is not None
        miss_payload = json.loads(miss_inv.result_canonical)
        hit_payload = json.loads(hit_inv.result_canonical)

        # Cache-miss payload uses the full ToolResult.to_dict shape;
        # cache-hit uses the slim hand-built shape. Both must contain
        # the catalog data and neither must contain the sentinel.
        _assert_payload_preserved(miss_payload, tool_name)
        assert "_canonicalization_error" not in hit_payload, (
            f"{tool_name}: cache-hit audit row carries sentinel — cache replay path bypassed normalization. Payload: {hit_payload}"
        )
        # Hand-built cache-hit shape has data + success + cache_hit.
        assert hit_payload["success"] is True
        assert hit_payload["cache_hit"] is True
        assert "data" in hit_payload

        if tool_name == "list_sources":
            assert hit_payload["data"][0]["name"] == "csv"
        elif tool_name == "get_plugin_schema":
            assert hit_payload["data"]["name"] == "csv"
            assert hit_payload["data"]["json_schema"] == {"title": "Config", "properties": {}}


@pytest.mark.asyncio
async def test_canonicalization_sentinel_omits_detail_for_non_rfc8785_errors() -> None:
    """Leak prevention: sentinel detail is captured ONLY for rfc8785 errors.

    Other ``ValueError`` paths (e.g. ``core/canonical.py`` Decimal
    check) interpolate offending payload values into the exception
    message. Echoing those into ``_canonicalization_detail`` would
    leak Tier-3 data into the Tier-1 audit row. The allowlist in
    :func:`build_canonicalization_sentinel` ensures detail capture
    fires only when the exception is an
    :class:`rfc8785.CanonicalizationError`.

    This test calls the helper directly with a synthetic
    ``ValueError`` carrying secret-shaped text and asserts the
    detail field is absent.
    """
    sentinel = build_canonicalization_sentinel(
        ValueError("Cannot canonicalize SECRET=hunter2 in field x"),
        {"field_x": "value", "field_y": "value"},
    )
    assert sentinel["_canonicalization_error"] == "ValueError"
    assert "_canonicalization_detail" not in sentinel, (
        "Non-rfc8785 ValueError must not capture exception message — potential Tier-3 leak into Tier-1 audit row."
    )
    # Top-level keys are still safe to capture.
    assert sentinel["_payload_keys"] == ["field_x", "field_y"]


def test_canonicalization_sentinel_captures_detail_for_rfc8785_errors() -> None:
    """rfc8785 errors are message-safe by spec — capture full detail.

    :class:`rfc8785.CanonicalizationError` messages are bounded type
    or rule strings (``"unsupported type: <class 'X'>"``,
    ``"<value> is not representable in JCS"``) that never echo
    arbitrary payload bytes. Capturing them in
    ``_canonicalization_detail`` gives auditors the offending Python
    type name without correlating to operational logs — exactly the
    forensic value the diagnostic upgrade is for.

    The 512-char cap is belt-and-braces: even if a future rfc8785
    inlined a longer schema fragment, the detail field is bounded.
    """
    exc = rfc8785.CanonicalizationError("unsupported type: <class 'elspeth.web.catalog.schemas.PluginSummary'>")
    sentinel = build_canonicalization_sentinel(exc, {"data": "x", "success": True, "validation": {}})
    assert sentinel["_canonicalization_error"] == "CanonicalizationError"
    detail = sentinel["_canonicalization_detail"]
    assert isinstance(detail, str)
    assert "PluginSummary" in detail
    assert "unsupported type" in detail
    assert sentinel["_payload_keys"] == ["data", "success", "validation"]


def test_canonicalization_sentinel_caps_detail_at_512_chars() -> None:
    """Belt-and-braces bound: detail capture is hard-capped at 512 chars.

    If a future rfc8785 release inlines a multi-kilobyte schema
    fragment into its exception message, the audit row stays
    bounded.
    """
    long_message = "unsupported type: " + ("X" * 2000)
    exc = rfc8785.CanonicalizationError(long_message)
    sentinel = build_canonicalization_sentinel(exc, {"k": "v"})
    detail = sentinel["_canonicalization_detail"]
    assert isinstance(detail, str)
    assert len(detail) == 512


def test_canonicalization_sentinel_omits_payload_keys_for_non_mapping() -> None:
    """Non-Mapping payloads (raw strings, lists) yield no key diagnostic.

    The string-truncation path in ``begin_dispatch`` and any future
    list-shaped audit payload would emit a noisy index-key list if
    the helper used ``isinstance(payload, Iterable)`` — the
    ``isinstance(payload, Mapping)`` guard keeps the diagnostic
    schema-meaningful (key NAMES, not indices).
    """
    exc = rfc8785.CanonicalizationError("unsupported type: <class 'X'>")
    sentinel = build_canonicalization_sentinel(exc, ["item1", "item2"])
    assert "_payload_keys" not in sentinel
    sentinel_str = build_canonicalization_sentinel(exc, "raw string")
    assert "_payload_keys" not in sentinel_str
    sentinel_none = build_canonicalization_sentinel(exc, None)
    assert "_payload_keys" not in sentinel_none
