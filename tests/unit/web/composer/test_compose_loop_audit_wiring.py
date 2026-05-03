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

from elspeth.contracts.composer_audit import ComposerToolStatus
from elspeth.web.catalog.protocol import CatalogService
from elspeth.web.catalog.schemas import PluginSchemaInfo, PluginSummary
from elspeth.web.composer.protocol import (
    ComposerPluginCrashError,
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
from elspeth.web.execution.schemas import ValidationResult

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
    )
    return catalog


def _make_settings(**overrides: Any) -> WebSettings:
    defaults: dict[str, Any] = {
        "data_dir": Path("/data"),
        "composer_max_composition_turns": 15,
        "composer_max_discovery_turns": 10,
        "composer_timeout_seconds": 85.0,
        "composer_rate_limit_per_minute": 10,
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
            "elspeth.web.composer.service.execute_tool",
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
            "elspeth.web.composer.service.execute_tool",
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
async def test_compose_loop_records_success_when_canonical_json_fails() -> None:
    """B1 fix: a non-finite float in a tool result must NOT bypass audit.

    Prior to the sentinel-canonical fallback inside ``finish_success``,
    a ToolResult whose ``to_dict()`` produced a non-finite float (or
    other non-JSON-serializable type) would raise ``ValueError`` from
    inside the recorder construction, and the success-side
    ``recorder.record(...)`` call would never fire. The state had
    already advanced (the loop rebound ``state = result.updated_state``
    BEFORE the recorder call), so the audit trail lost the dispatch
    that produced the new state — the worst-case audit-hole shape.

    With the helper, ``finish_success`` catches ``(ValueError,
    TypeError)`` from canonicalization and substitutes a sentinel
    payload (``{"_canonicalization_error": "<exc class>"}``). The
    audit row still lands; an auditor can detect the sentinel by
    parsing ``result_canonical``.

    This test substitutes a ToolResult subclass whose ``to_dict()``
    returns a float ``inf`` (which ``rfc8785.dumps`` rejects per RFC
    8785 §3.1) and confirms the recorder still captured a SUCCESS
    invocation.
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
        def to_dict(self) -> dict[str, Any]:  # type: ignore[override]
            return {
                "success": True,
                "version": self.updated_state.version,
                # rfc8785 raises on non-finite floats; this triggers the
                # sentinel-canonical fallback inside finish_success.
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

    # The empty test state has no source/sinks so the post-loop runtime
    # preflight would fail and replace the assistant text with a
    # synthetic preflight-failure message. That is correct production
    # behaviour but orthogonal to the audit invariant we're pinning
    # here. Stub _runtime_preflight to return a passing ValidationResult
    # so result.message reflects the LLM's text-only turn unchanged —
    # matches the discipline already used in TestComposerSingleToolCall.
    passing_preflight = ValidationResult(is_valid=True, checks=[], errors=[])

    with (
        patch.object(service, "_call_llm", new_callable=AsyncMock) as mock_llm,
        patch.object(service, "_runtime_preflight", return_value=passing_preflight),
        patch(
            "elspeth.web.composer.service.execute_tool",
            return_value=bad_result,
        ),
    ):
        mock_llm.side_effect = [turn1, turn2]
        result = await service.compose("Trigger non-finite payload", [], state)

    # The compose call completed normally — the audit row lands with
    # a sentinel canonical, the success path returns the mutation,
    # the LLM gets to produce its text reply.
    assert result.message == "Done."
    invocations = result.tool_invocations
    assert len(invocations) == 1
    inv = invocations[0]
    assert inv.status == ComposerToolStatus.SUCCESS
    assert inv.tool_call_id == "call_with_non_finite"
    # Sentinel canonical: parsable JSON object carrying the error class
    # under a reserved key. An auditor reading this back recognises
    # the sentinel and can correlate with operational logs.
    assert inv.result_canonical is not None
    payload = json.loads(inv.result_canonical)
    assert "_canonicalization_error" in payload
    # Pin the version-after capture: the success path completed; the
    # state advanced; the audit row carries the post-mutation version.
    assert inv.version_after == 2


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
            "elspeth.web.composer.service.execute_tool",
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
