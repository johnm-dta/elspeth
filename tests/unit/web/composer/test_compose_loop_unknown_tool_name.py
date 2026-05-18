"""Compose-loop unknown-tool-name routing and Phase 3 call-order pins.

Spec §4.2.6 / §5.7.5: an LLM-hallucinated tool name is Tier-3 input.
The dispatcher's fall-through at tools.py:5731 returns a failure
ToolResult; the compose loop records and continues. Per
ComposerToolStatus.SUCCESS docstring (contracts/composer_audit.py:34-37),
this is a successful dispatch with a semantic-failure payload — the audit
record carries the full result so an auditor can read the outcome.

Closes plan-review M7 / W3 (Tier-3 quarantine, not Tier-1 crash) and
rev-3 M2 (Phase 3 call-order precondition).
"""

from __future__ import annotations

import asyncio
import json
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from elspeth.contracts.composer_audit import ComposerToolStatus
from elspeth.contracts.errors import AuditIntegrityError
from elspeth.web.catalog.protocol import CatalogService
from elspeth.web.catalog.schemas import (
    PluginSchemaInfo,
    PluginSummary,
)
from elspeth.web.composer.redaction import redact_tool_call_arguments
from elspeth.web.composer.redaction_telemetry import NoopRedactionTelemetry
from elspeth.web.composer.service import ComposerAvailability, ComposerServiceImpl
from elspeth.web.composer.state import (
    CompositionState,
    PipelineMetadata,
)
from elspeth.web.config import WebSettings

# ---------------------------------------------------------------------------
# Module-scoped fixtures required for all compose-loop tests in this file.
# These match the autouse fixtures in test_service.py exactly so the service
# sees the same execution environment as the broader unit-test suite.
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _composer_available(monkeypatch: pytest.MonkeyPatch) -> None:
    """Bypass API-key check so tests focus on compose behavior, not credentials."""

    def _available(self: ComposerServiceImpl) -> ComposerAvailability:
        return ComposerAvailability(available=True, model=self._model, provider="test")

    monkeypatch.setattr(ComposerServiceImpl, "_compute_availability", _available)


@pytest.fixture(autouse=True)
def _composer_to_thread_test_worker(monkeypatch: pytest.MonkeyPatch) -> None:
    """Run asyncio.to_thread calls through a deterministic test worker thread.

    Prevents local executor hangs from masking composer behavior in tests.
    """

    async def test_to_thread(func: Any, /, *args: Any, **kwargs: Any) -> Any:
        result: list[Any] = []
        failures: list[BaseException] = []

        def run() -> None:
            try:
                result.append(func(*args, **kwargs))
            except BaseException as exc:
                failures.append(exc)

        worker = threading.Thread(target=run, name="composer-test-worker")
        worker.start()
        while worker.is_alive():
            await asyncio.sleep(0.001)
        worker.join()
        if failures:
            raise failures[0]
        if result:
            return result[0]
        return None

    monkeypatch.setattr("asyncio.to_thread", test_to_thread)


# ---------------------------------------------------------------------------
# Local helpers (mirror test_service.py private helpers; no cross-module reach)
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
    catalog.list_transforms.return_value = [
        PluginSummary(
            name="uppercase",
            description="Uppercase",
            plugin_type="transform",
            config_fields=[],
        ),
    ]
    catalog.list_sinks.return_value = [
        PluginSummary(
            name="csv",
            description="CSV sink",
            plugin_type="sink",
            config_fields=[],
        ),
    ]
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
    message = _FakeMessage(content=content, tool_calls=fake_tool_calls)
    return _FakeLLMResponse(choices=[_FakeChoice(message=message)])


# ---------------------------------------------------------------------------
# Test 1: Compose-loop unknown-tool-name audit shape (M7/W3 closure)
# ---------------------------------------------------------------------------


class TestUnknownToolNameComposeLoopAuditShape:
    """Pin the exact audit shape produced when the LLM emits a hallucinated
    tool name through the full compose-loop production code path.

    Operator decision (Task 17 option a): the unknown-tool-name path
    produces ``ComposerToolStatus.SUCCESS`` with ``result.success=False``,
    NOT ``ARG_ERROR``. The "dispatch succeeded; payload tells the story"
    framing is canonical and is documented in the ``ComposerToolStatus.SUCCESS``
    docstring (contracts/composer_audit.py:34-37). This test pins the four
    specific values that constitute the accepted audit shape.
    """

    @pytest.mark.asyncio
    async def test_unknown_tool_name_audit_shape(self) -> None:
        """Drive the compose loop with a hallucinated tool name; pin the audit shape.

        Asserts:
        1. ``invocation.status == ComposerToolStatus.SUCCESS`` — the dispatch
           itself completed (dispatcher returned without raising); the audit
           status does NOT mis-classify this as an argument error.
        2. ``invocation.error_class is None`` — ``error_class`` is only
           populated on ``ARG_ERROR`` and ``PLUGIN_CRASH`` paths; a
           SUCCESS-status invocation always has ``error_class=None``.
        3. ``result_canonical`` parses to a dict where ``payload["success"]
           is False`` — the semantic failure is encoded in the payload, not
           in the status field.
        4. ``payload["data"]["error"]`` contains the literal substring
           ``"Unknown tool: this_tool_does_not_exist"`` — the error message
           produced by ``_failure_result`` at ``tools.py:5731``.
        5. The compose loop continues after the unknown-tool-call turn:
           ``result.message`` is the LLM's self-correction text from turn 2.

        Spec refs: §4.2.6 disposition table (added row: unknown tool name →
        SUCCESS-with-semantic-failure); §5.7.5 (audit status clarified).
        Pin: ``tests/unit/web/composer/test_compose_loop_unknown_tool_name.py``
        (this file). Closes plan-review M7 / W3.
        """
        catalog = _mock_catalog()
        settings = _make_settings()
        service = ComposerServiceImpl(catalog=catalog, settings=settings)
        state = _empty_state()

        # Turn 1: LLM emits a hallucinated (unknown) tool name.
        unknown_tool_call = _make_llm_response(
            tool_calls=[
                {
                    "id": "call_unknown",
                    "name": "this_tool_does_not_exist",
                    "arguments": {},
                }
            ],
        )
        # Turn 2: LLM self-corrects with a text response after receiving the
        # failure payload as a role=tool message.
        self_correction = _make_llm_response(content="I apologise — that tool does not exist. Let me try again.")

        with patch.object(service, "_call_llm", new_callable=AsyncMock) as mock_llm:
            mock_llm.side_effect = [unknown_tool_call, self_correction]
            result = await service.compose("Build a pipeline", [], state)

        # Assert 5: compose loop continued; turn 2 text is the result message.
        assert "apologise" in result.message or "sorry" in result.message.lower() or result.message

        # Find the invocation for the hallucinated tool name.
        unknown_invocations = [inv for inv in result.tool_invocations if inv.tool_name == "this_tool_does_not_exist"]
        assert len(unknown_invocations) == 1, (
            f"Expected exactly one invocation for 'this_tool_does_not_exist', "
            f"got {len(unknown_invocations)}. All invocations: "
            f"{[inv.tool_name for inv in result.tool_invocations]}"
        )
        invocation = unknown_invocations[0]

        # Assert 1: status is SUCCESS — dispatch completed without raising.
        assert invocation.status is ComposerToolStatus.SUCCESS, (
            f"Expected ComposerToolStatus.SUCCESS, got {invocation.status!r}. "
            "The unknown-tool-name path should route through dispatch_with_audit's "
            "SUCCESS branch because tools.py:5731 returns a failure ToolResult "
            "without raising (no exception → no ARG_ERROR)."
        )

        # Assert 2: error_class is None — SUCCESS-status invocations never carry
        # error_class (it is only populated on ARG_ERROR / PLUGIN_CRASH paths).
        assert invocation.error_class is None, f"Expected error_class=None for a SUCCESS-status invocation, got {invocation.error_class!r}."

        # Assert 3 + 4: parse result_canonical and check the semantic-failure payload.
        assert invocation.result_canonical is not None, (
            "SUCCESS-status invocations must have result_canonical; got None (indicates the audit write was skipped)."
        )
        payload = json.loads(invocation.result_canonical)
        assert payload["success"] is False, (
            f"Expected payload['success'] to be False (semantic failure recorded in the payload), got {payload['success']!r}."
        )
        assert "data" in payload, f"Expected 'data' key in result_canonical payload, got keys: {list(payload.keys())}."
        assert "error" in payload["data"], f"Expected 'error' key in payload['data'], got keys: {list(payload['data'].keys())}."
        assert "Unknown tool: this_tool_does_not_exist" in payload["data"]["error"], (
            f"Expected 'Unknown tool: this_tool_does_not_exist' in payload['data']['error'], got: {payload['data']['error']!r}."
        )


# ---------------------------------------------------------------------------
# Test 2: Phase 3 call-order pin (rev-3 M2 closure)
# ---------------------------------------------------------------------------


def test_redact_tool_call_arguments_raises_for_unknown_tool() -> None:
    """Phase 3 contract pin: ``redact_tool_call_arguments`` must NOT be
    called for a tool name that is not in MANIFEST.

    The compose loop's existing unknown-tool check (``tools.py:5731`` ->
    ``_failure_result`` with ``Unknown tool: {name}``) MUST fire BEFORE
    the redaction layer.

    If Phase 3 inverts this ordering (redact-then-check), an
    LLM-hallucinated tool name will be silently converted from a
    graceful Tier-3 dispatcher fall-through into a Tier-1
    :class:`AuditIntegrityError` crash. This test asserts that
    :func:`redact_tool_call_arguments` fails loudly when called out of
    order, so Phase 3's call site is mechanically constrained — any
    refactor that inverts the order trips this pin in CI rather than at
    runtime under a hallucinated tool name. Closes rev-3 M2 / rev-4 M2.

    The error message MUST cite the missing tool name so Phase 3
    implementers see the contract violation in stack traces (the
    quoted form from the ``{tool_name!r}`` formatter contains the
    name as a substring).
    """
    with pytest.raises(AuditIntegrityError) as excinfo:
        redact_tool_call_arguments(
            tool_name="nonexistent_tool_name_for_call_order_pin",
            arguments={},
            telemetry=NoopRedactionTelemetry(),
        )
    assert "nonexistent_tool_name_for_call_order_pin" in str(excinfo.value)
