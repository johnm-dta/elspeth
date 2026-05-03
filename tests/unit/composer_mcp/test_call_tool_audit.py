"""Audit-before-return invariant tests for composer_mcp/server.py call_tool.

Pins the structural guarantee that the try/finally envelope around the
dispatch ALWAYS records before the coroutine yields, on every exit path
(SUCCESS / ARG_ERROR / PLUGIN_CRASH). Mirrors AuditedLLMClient pattern.

If a future refactor replaces try/finally with try/except, these tests
are the canary that catches the regression.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from elspeth.composer_mcp.server import create_server
from elspeth.contracts.composer_audit import ComposerToolInvocation, ComposerToolStatus
from elspeth.web.dependencies import create_catalog_service


class _ProbeRecorder:
    """Captures every invocation in a list for inspection."""

    def __init__(self) -> None:
        self.invocations: list[ComposerToolInvocation] = []

    def record(self, invocation: ComposerToolInvocation) -> None:
        self.invocations.append(invocation)

    def resolve_session(self, session_id: str) -> None:
        return


def _call_handler(handlers, name: str, arguments: dict):
    from mcp.types import CallToolRequest, CallToolRequestParams

    req = CallToolRequest(
        method="tools/call",
        params=CallToolRequestParams(name=name, arguments=arguments),
    )
    return handlers[CallToolRequest](req)


@pytest.mark.asyncio
async def test_success_path_records_before_return() -> None:
    """Successful dispatch must produce exactly one SUCCESS invocation."""
    catalog = create_catalog_service()
    with tempfile.TemporaryDirectory() as td:
        scratch = Path(td)
        probe = _ProbeRecorder()
        server = create_server(catalog, scratch, recorder=probe)
        await _call_handler(server.request_handlers, "new_session", {"name": "AuditTest"})
    assert len(probe.invocations) == 1
    inv = probe.invocations[0]
    assert inv.status == ComposerToolStatus.SUCCESS
    assert inv.tool_name == "new_session"
    assert inv.error_class is None
    # Latency is non-negative; started_at <= finished_at.
    assert inv.latency_ms >= 0
    assert inv.started_at <= inv.finished_at


@pytest.mark.asyncio
async def test_arg_error_path_records_before_return() -> None:
    """Bad LLM args must record ARG_ERROR before the CallToolResult is returned."""
    catalog = create_catalog_service()
    with tempfile.TemporaryDirectory() as td:
        scratch = Path(td)
        probe = _ProbeRecorder()
        server = create_server(catalog, scratch, recorder=probe)
        # NOT_HEX is rejected by InvalidSessionIdError (a ValueError subclass).
        await _call_handler(server.request_handlers, "load_session", {"session_id": "NOT_HEX"})
    assert len(probe.invocations) == 1
    inv = probe.invocations[0]
    assert inv.status == ComposerToolStatus.ARG_ERROR
    assert inv.tool_name == "load_session"
    assert inv.error_class == "InvalidSessionIdError"
    # Per the redaction discipline: error_message is the class name only,
    # NOT exc.args[0] (which could carry filesystem paths via
    # CorruptSessionFileError). This pins the Python-W1 fix.
    assert inv.error_message == "InvalidSessionIdError"
    # ARG_ERROR ⇒ version_after is None (the dispatch did not complete).
    assert inv.version_after is None


@pytest.mark.asyncio
async def test_arg_error_payload_recorded_for_audit_replay() -> None:
    """ARG_ERROR records carry result_canonical with the structured error
    payload so an auditor can replay what the LLM saw."""
    catalog = create_catalog_service()
    with tempfile.TemporaryDirectory() as td:
        scratch = Path(td)
        probe = _ProbeRecorder()
        server = create_server(catalog, scratch, recorder=probe)
        await _call_handler(server.request_handlers, "load_session", {"session_id": "NOT_HEX"})
    inv = probe.invocations[0]
    assert inv.status == ComposerToolStatus.ARG_ERROR
    # H4 fix: result_canonical mirrors what the LLM saw.
    assert inv.result_canonical is not None
    import json as _json

    payload = _json.loads(inv.result_canonical)
    assert payload["isError"] is True
    assert "Tool error" in payload["error"]


@pytest.mark.asyncio
async def test_audit_records_in_order_across_session() -> None:
    """A multi-call session must record invocations in dispatch order."""
    catalog = create_catalog_service()
    with tempfile.TemporaryDirectory() as td:
        scratch = Path(td)
        probe = _ProbeRecorder()
        server = create_server(catalog, scratch, recorder=probe)
        await _call_handler(server.request_handlers, "new_session", {"name": "T"})
        await _call_handler(server.request_handlers, "list_sessions", {})
        await _call_handler(server.request_handlers, "load_session", {"session_id": "NOT_HEX"})
    assert [i.tool_name for i in probe.invocations] == [
        "new_session",
        "list_sessions",
        "load_session",
    ]
    assert [i.status for i in probe.invocations] == [
        ComposerToolStatus.SUCCESS,
        ComposerToolStatus.SUCCESS,
        ComposerToolStatus.ARG_ERROR,
    ]


@pytest.mark.asyncio
async def test_plugin_crash_path_records_before_reraise() -> None:
    """W3 fix: PLUGIN_CRASH must record before the exception propagates.

    The MCP ``call_tool`` handler wraps ``_dispatch_tool`` in a
    ``try/except Exception`` that captures ``status = PLUGIN_CRASH``,
    ``error_class = type(exc).__name__``, ``error_message =
    type(exc).__name__`` (class-name only — pins the redaction
    discipline against future drift that would echo ``str(exc)``)
    and re-raises. The outer ``finally`` then writes the invocation
    record before the exception leaves the coroutine.

    Note on the propagation seam: the MCP SDK's ``call_tool`` request
    handler wraps tool exceptions into a ``CallToolResult(isError=True)``
    response at the protocol-transport layer. So the user-visible
    behaviour is ``isError=True`` content rather than a propagating
    Python exception — but the audit invariant is independent of the
    transport's framing. The ``finally`` block inside the inner
    ``call_tool`` decorator runs on the exception path BEFORE the
    transport layer converts it. This test asserts the audit row
    landed regardless of how the MCP transport surfaced the failure.

    This test was missing from the initial slice — the module
    docstring claimed all three paths were covered but only SUCCESS
    and ARG_ERROR had tests. Closes Python-engineer review W3.
    """
    catalog = create_catalog_service()
    with tempfile.TemporaryDirectory() as td:
        scratch = Path(td)
        probe = _ProbeRecorder()
        server = create_server(catalog, scratch, recorder=probe)
        # Patch the dispatcher seam to raise a plain RuntimeError —
        # the canonical "plugin bug" shape per CLAUDE.md "Plugin
        # Ownership". Anything other than ToolArgumentError /
        # (ValueError, KeyError, TypeError) flows through the
        # PLUGIN_CRASH except branch.
        with patch(
            "elspeth.composer_mcp.server._dispatch_tool",
            side_effect=RuntimeError("synthetic plugin bug"),
        ):
            response = await _call_handler(
                server.request_handlers,
                "new_session",
                {"name": "CrashTest"},
            )

    # Transport-layer framing: the MCP SDK converts the propagating
    # RuntimePLUGIN_CRASH into an ``isError=True`` CallToolResult
    # AFTER the inner handler's ``finally`` ran. The audit row was
    # already written.
    call_result = response.root
    assert call_result.isError is True

    assert len(probe.invocations) == 1
    inv = probe.invocations[0]
    assert inv.status == ComposerToolStatus.PLUGIN_CRASH
    assert inv.tool_name == "new_session"
    assert inv.error_class == "RuntimeError"
    # Class-name-only echo — pins the redaction discipline. A future
    # regression that switched to ``str(exc)`` would echo the
    # operator-readable "synthetic plugin bug" message into the
    # audit trail; this assertion catches that.
    assert inv.error_message == "RuntimeError"
    # PLUGIN_CRASH ⇒ version_after is None (the dispatch did not
    # complete; the SUCCESS reset path inside the finally only fires
    # for status == SUCCESS).
    assert inv.version_after is None
    # No result_canonical or result_hash on the crash path — the LLM
    # never saw a tool result, so the audit row faithfully records
    # "no result was produced".
    assert inv.result_canonical is None
    assert inv.result_hash is None


@pytest.mark.asyncio
async def test_delete_session_does_not_orphan_sidecar() -> None:
    """C1 fix: delete_session must not recreate an orphan sidecar.

    After delete_session succeeds, session_id_ref is cleared; the
    deletion record buffers (in process memory) and dies with the
    process. The previously-active sidecar file is unlinked (by
    SessionManager.delete) and stays unlinked.
    """
    from elspeth.composer_mcp.audit import events_sidecar_path

    catalog = create_catalog_service()
    with tempfile.TemporaryDirectory() as td:
        scratch = Path(td)
        # Use the production JsonlEventRecorder so we can verify the file state.
        # We need the same session_id_ref as create_server, but create_server
        # makes it internally. Instead, drive the API and confirm cleanup.
        server = create_server(catalog, scratch)  # default JsonlEventRecorder
        # Create + save + delete.
        r = await _call_handler(server.request_handlers, "new_session", {"name": "X"})
        import json as _json

        sid = _json.loads(r.root.content[0].text)["data"]["session_id"]
        sidecar = events_sidecar_path(scratch, sid)
        await _call_handler(server.request_handlers, "save_session", {"session_id": sid})
        assert sidecar.exists()
        await _call_handler(server.request_handlers, "delete_session", {"session_id": sid})
        # Pre-fix bug: the deletion record would recreate the sidecar.
        # Post-fix: session_id_ref cleared, delete record buffers, no recreation.
        assert not sidecar.exists()
