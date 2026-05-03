"""In-memory buffering recorder + dispatch helper for the web composer.

The web composer runs the LLM tool-use loop inside a single request scope.
Recording each :class:`ComposerToolInvocation` to the database from inside
the loop would tightly couple the composer service to the SQLAlchemy
engine and fragment a single logical turn into N small transactions —
neither is desirable.

Instead, the service uses a :class:`BufferingRecorder` to collect
invocations during the compose loop, surfaces them on
:class:`ComposerResult` (and on the three partial-state-carrier
exceptions: :class:`ComposerConvergenceError`,
:class:`ComposerPluginCrashError`,
:class:`ComposerRuntimePreflightError`), and the route handler persists
the entire ordered batch in a single transaction alongside the assistant
message.

Each invocation lands as one ``role=tool`` chat message whose
``tool_calls`` JSON column carries a ``{"_kind": "audit", "invocation":
{...}}`` envelope. The discriminator distinguishes audit-side payloads
from assistant-side OpenAI tool_calls request payloads (which use the
same column on assistant rows).

Helper architecture
-------------------

This module owns the per-dispatch audit envelope (`_DispatchAudit`,
`_begin_dispatch`, `_finish_*`) AND the structural enforcement helper
:func:`dispatch_with_audit`. The helper guarantees "audit fires before
return on every path" structurally — mirroring the ``try/finally``
shape used by ``composer_mcp/server.py:call_tool`` and
``AuditedLLMClient.chat_completion``. Pulling these helpers out of
``service.py`` and into a single module makes the contract a structural
property of the helper, not a procedural property of the loop.

Layer: L3 (application). Imports L0 (contracts.composer_audit), L1
(core.canonical), L3 (web.composer.protocol), and stdlib only.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from elspeth.contracts.composer_audit import (
    ComposerToolInvocation,
    ComposerToolRecorder,
    ComposerToolStatus,
)
from elspeth.core.canonical import canonical_json, stable_hash
from elspeth.web.composer.protocol import ToolArgumentError

__all__ = [
    "BufferingRecorder",
    "DispatchAudit",
    "DispatchOutcome",
    "audit_envelope",
    "begin_dispatch",
    "dispatch_with_audit",
    "finish_arg_error",
    "finish_plugin_crash",
    "finish_success",
]


class BufferingRecorder(ComposerToolRecorder):
    """Append-only in-memory buffer of :class:`ComposerToolInvocation`.

    Used inside :meth:`ComposerServiceImpl._compose_loop`. The buffer is
    surfaced on :class:`ComposerResult` and on the three partial-state-
    carrier exceptions so the route handler always has the per-call
    audit trail — including on convergence/plugin-crash/runtime-preflight
    failure paths where the LLM never produced a final assistant text.

    Threading: ``record()`` is safe to call from any thread. The compose
    loop dispatches synchronously to a worker via ``run_sync_in_worker``
    and records from the asyncio event loop, but the lock is cheap
    insurance for any future restructure that spawns parallel workers.
    """

    def __init__(self) -> None:
        self._invocations: list[ComposerToolInvocation] = []
        self._lock = threading.Lock()

    def record(self, invocation: ComposerToolInvocation) -> None:
        with self._lock:
            self._invocations.append(invocation)

    @property
    def invocations(self) -> tuple[ComposerToolInvocation, ...]:
        """Snapshot the current buffer as an immutable tuple."""
        with self._lock:
            return tuple(self._invocations)

    def resolve_session(self, session_id: str) -> None:
        """Protocol no-op — the in-memory buffer has nothing to flush.

        Implemented to satisfy :class:`ComposerToolRecorder`. The
        web composer's session_id is known at compose() entry, so
        there is no pre-resolution buffer to drain. ``session_id``
        is accepted but ignored.
        """
        del session_id  # explicit unused-arg discard
        return


def audit_envelope(invocation: ComposerToolInvocation) -> dict[str, object]:
    """Wrap an invocation in the canonical ``tool_calls`` JSON envelope.

    Returns the dict that lands in the ``tool_calls`` JSON column on a
    ``role=tool`` chat message row::

        {"_kind": "audit", "invocation": {...}}

    The ``_kind`` discriminator distinguishes audit-side payloads from
    assistant-side OpenAI tool-call request payloads (which use the same
    column on assistant rows). Code that reads back ``chat_messages`` and
    needs to handle both shapes can dispatch on ``_kind``.

    The ``invocation`` field is the dataclass's ``to_dict()`` output —
    JSON-friendly (datetime → ISO 8601, StrEnum → str). Auditors verifying
    Tier-1 integrity recompute SHA-256 over ``invocation.arguments_canonical``
    and compare to ``invocation.arguments_hash``; mismatch is evidence of
    tampering.
    """
    return {"_kind": "audit", "invocation": invocation.to_dict()}


# ---------------------------------------------------------------------------
# Per-dispatch audit envelope.
# Hoisted from web/composer/service.py so the helper :func:`dispatch_with_audit`
# below can construct invocations directly. The names DispatchAudit /
# begin_dispatch / finish_* (no leading underscore) reflect that these are now
# the public seam other compose-loop sites import — service.py imports them
# alongside the helper. The leading-underscore aliases (_DispatchAudit etc.)
# remain available as private re-exports for any future callsite that wants to
# preserve the old name.
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class DispatchAudit:
    """Per-call timing + canonical-arguments envelope for the audit trail.

    Captured at the start of every dispatch branch in
    :meth:`ComposerServiceImpl._compose_loop` so each branch's invocation
    record carries consistent ``started_at`` / ``arguments_canonical`` /
    ``version_before`` regardless of which path the call ultimately
    follows. The branch-specific finalizers (``finish_*``) read fields
    from here to construct the final :class:`ComposerToolInvocation`.
    """

    tool_call_id: str
    tool_name: str
    arguments_canonical: str
    arguments_hash: str
    version_before: int
    started_at: datetime
    started_ns: int
    actor: str


def begin_dispatch(
    tool_call_id: str,
    tool_name: str,
    arguments: Mapping[str, Any] | str,
    *,
    version_before: int,
    actor: str,
) -> DispatchAudit:
    """Open a per-call audit envelope.

    ``arguments`` is either a parsed dict (typical path) or a raw string
    (JSON-decode failure path before parse succeeded). For the raw-string
    case the canonical form wraps the (truncated) string in a sentinel
    object so the audit trail still records what the LLM tried even
    when it wasn't valid JSON. Truncation guards against unbounded
    audit-row sizes for pathological LLM output.
    """
    if isinstance(arguments, str):
        # 4 KiB is the same boundary as POSIX PIPE_BUF — a sane upper
        # bound that captures normal tool-call shapes while preventing
        # a malformed multi-megabyte LLM response from blowing up audit
        # rows. The full string is recoverable from upstream LiteLLM
        # tracing if needed; the audit row records intent, not bytes.
        truncated = arguments[:4096]
        canon = canonical_json({"_unparseable_arguments": truncated, "_truncated": len(arguments) > 4096})
        h = stable_hash({"_unparseable_arguments": truncated, "_truncated": len(arguments) > 4096})
    else:
        canon = canonical_json(arguments)
        h = stable_hash(arguments)
    return DispatchAudit(
        tool_call_id=tool_call_id,
        tool_name=tool_name,
        arguments_canonical=canon,
        arguments_hash=h,
        version_before=version_before,
        started_at=datetime.now(UTC),
        started_ns=time.monotonic_ns(),
        actor=actor,
    )


def finish_success(
    audit: DispatchAudit,
    *,
    result_payload: Mapping[str, Any],
    version_after: int,
    cache_hit: bool = False,
) -> ComposerToolInvocation:
    """Build a SUCCESS-status invocation record.

    A "successful dispatch" includes handler returns where the underlying
    tool reported ``success=False`` semantically — that is still a
    successful dispatch, and the semantic outcome is recoverable from
    ``result_canonical``. Only path-level failures (ARG_ERROR,
    PLUGIN_CRASH) get a non-success status.

    Sentinel-canonical fallback (B1 fix)
    ------------------------------------
    Wrap ``canonical_json(result_payload)`` and ``stable_hash(result_payload)``
    in try/except so a non-finite float or non-serializable type in the
    result does not raise out of the success path and either crash the
    request or silently skip the audit record. Mirrors the standalone-MCP
    sentinel at ``composer_mcp/server.py:715-721`` exactly so both
    surfaces have the same canonicalization-failure discipline. The audit
    row still lands; the verifier can detect the sentinel by parsing
    ``result_canonical`` and noticing the ``_canonicalization_error``
    key.
    """
    try:
        canon = canonical_json(result_payload)
        result_hash = stable_hash(result_payload)
    except (ValueError, TypeError) as canon_exc:
        sentinel = {"_canonicalization_error": type(canon_exc).__name__}
        canon = canonical_json(sentinel)
        result_hash = stable_hash(sentinel)
    return ComposerToolInvocation(
        tool_call_id=audit.tool_call_id,
        tool_name=audit.tool_name,
        arguments_canonical=audit.arguments_canonical,
        arguments_hash=audit.arguments_hash,
        result_canonical=canon,
        result_hash=result_hash,
        status=ComposerToolStatus.SUCCESS,
        error_class=None,
        error_message=None,
        version_before=audit.version_before,
        version_after=version_after,
        started_at=audit.started_at,
        finished_at=datetime.now(UTC),
        latency_ms=(time.monotonic_ns() - audit.started_ns) // 1_000_000,
        actor=audit.actor,
        cache_hit=cache_hit,
    )


def finish_arg_error(
    audit: DispatchAudit,
    *,
    error_class: str,
    error_message: str,
    error_payload: Mapping[str, Any] | None = None,
) -> ComposerToolInvocation:
    """Build an ARG_ERROR invocation record.

    ``error_message`` is already-redacted at the dispatch boundary —
    callers MUST pass safe-by-construction text (``ToolArgumentError.args[0]``,
    a known schema-failure summary, or a class-name-only fallback).

    ``error_payload``, when supplied, is canonicalized into
    ``result_canonical`` so the audit trail records what was sent back
    to the LLM. Pre-dispatch ARG_ERROR sites pass a structured dict
    that mirrors the ``role=tool`` content the LLM saw.

    ``version_after = None`` — ARG_ERROR means the dispatch did not
    complete (the handler either was never reached or raised before
    producing a result). Aligned with the standalone MCP recorder
    (``server.py:call_tool``) so a Tier-1 verifier can apply a single
    invariant: "version_after is None on paths that did not complete".
    """
    return ComposerToolInvocation(
        tool_call_id=audit.tool_call_id,
        tool_name=audit.tool_name,
        arguments_canonical=audit.arguments_canonical,
        arguments_hash=audit.arguments_hash,
        result_canonical=canonical_json(error_payload) if error_payload is not None else None,
        result_hash=stable_hash(error_payload) if error_payload is not None else None,
        status=ComposerToolStatus.ARG_ERROR,
        error_class=error_class,
        error_message=error_message,
        version_before=audit.version_before,
        version_after=None,
        started_at=audit.started_at,
        finished_at=datetime.now(UTC),
        latency_ms=(time.monotonic_ns() - audit.started_ns) // 1_000_000,
        actor=audit.actor,
    )


def finish_plugin_crash(
    audit: DispatchAudit,
    *,
    exc: BaseException,
) -> ComposerToolInvocation:
    """Build a PLUGIN_CRASH invocation record.

    ``error_message`` is the class name only — plugin exception
    messages can carry secrets, DB URLs, filesystem paths. The full
    cause chain remains on ``__cause__`` for ASGI / server-log
    machinery; this record is for the audit trail, which is
    operator-readable.

    Accepts ``BaseException`` (not just ``Exception``) so the helper
    can record the narrow re-raise classes (``AssertionError``,
    ``MemoryError``, ``RecursionError``, ``SystemError``) before
    propagating. Constructing a frozen dataclass from pre-captured
    scalar fields is not poisoned-memory work — the audit envelope
    was opened before the crash, the version snapshot and timing
    are already in hand, and writing the record is a single in-memory
    list append.
    """
    cls_name = type(exc).__name__
    return ComposerToolInvocation(
        tool_call_id=audit.tool_call_id,
        tool_name=audit.tool_name,
        arguments_canonical=audit.arguments_canonical,
        arguments_hash=audit.arguments_hash,
        result_canonical=None,
        result_hash=None,
        status=ComposerToolStatus.PLUGIN_CRASH,
        error_class=cls_name,
        error_message=cls_name,
        version_before=audit.version_before,
        version_after=None,
        started_at=audit.started_at,
        finished_at=datetime.now(UTC),
        latency_ms=(time.monotonic_ns() - audit.started_ns) // 1_000_000,
        actor=audit.actor,
    )


# ---------------------------------------------------------------------------
# Structural dispatch helper.
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class DispatchOutcome:
    """Outcome of a successful (handler-returned) dispatch.

    The helper :func:`dispatch_with_audit` returns this when
    ``do_dispatch`` returns normally. It carries the handler's result
    plus the post-dispatch state-version so the caller can advance
    its loop-local ``state`` reference and continue.

    ARG_ERROR and PLUGIN_CRASH paths re-raise instead of returning;
    the caller catches those in narrowed except blocks (just like
    today, but the audit record is already written).

    ``result`` is intentionally typed ``Any`` because the helper is
    generic over tool-handler return types; the caller knows the
    concrete shape (a ``ToolResult`` for ``execute_tool``).
    """

    result: Any
    version_after: int


async def dispatch_with_audit(
    *,
    recorder: ComposerToolRecorder,
    audit: DispatchAudit,
    do_dispatch: Callable[[], Awaitable[Any]],
    version_after_provider: Callable[[Any], int],
    arg_error_payload_factory: Callable[[ToolArgumentError], Mapping[str, Any]],
) -> DispatchOutcome:
    """Run a tool dispatch under the audit envelope.

    Structural guarantee: the recorder records exactly one
    :class:`ComposerToolInvocation` before this coroutine returns or
    raises, on every path. This is the structural counterpart of the
    procedural per-branch ``recorder.record(...)`` calls that the
    earlier ``_compose_loop`` shape relied on; replacing seven
    procedural call sites with one structural helper removes the
    audit-hole class of bugs that the panel review (B1, B2)
    identified.

    Paths
    -----
    SUCCESS
        ``do_dispatch`` returned a value. Build a SUCCESS invocation
        via :func:`finish_success` (which itself wraps
        ``canonical_json`` in a sentinel-fallback try/except so a
        non-finite float in the result cannot bypass audit). Record
        and return :class:`DispatchOutcome`.

    ARG_ERROR
        ``do_dispatch`` raised :class:`ToolArgumentError`. Build an
        ARG_ERROR invocation via :func:`finish_arg_error` with
        ``error_payload`` from ``arg_error_payload_factory`` (which the
        caller uses to also produce the LLM-facing tool message). Record
        and re-raise so the caller's ``except ToolArgumentError`` block
        runs the LLM-message-append branch unchanged.

    PLUGIN_CRASH (narrow re-raise)
        ``do_dispatch`` raised one of
        ``(AssertionError, MemoryError, RecursionError, SystemError)``.
        These were previously re-raised WITHOUT an audit record because
        the original implementation worried about poisoned-memory work.
        That worry was misjudged: the audit envelope was already opened
        (``audit`` is a frozen dataclass of pre-captured scalars), and
        constructing a :class:`ComposerToolInvocation` from those
        scalars plus ``type(exc).__name__`` is also pure scalar work.
        Record :class:`ComposerToolStatus.PLUGIN_CRASH` first, then
        re-raise so the caller's ``except (AssertionError, ...)`` block
        propagates the exception out of ``_compose_loop`` unchanged.
        Closes blocker B2 from the panel review (2026-05-04).

    PLUGIN_CRASH (general)
        ``do_dispatch`` raised any other ``Exception`` subclass. Build
        a PLUGIN_CRASH invocation via :func:`finish_plugin_crash` and
        re-raise so the caller's ``except Exception`` block can wrap
        with :meth:`ComposerPluginCrashError.capture`.

    Why ``BaseException`` is NOT caught generically
    -----------------------------------------------
    ``SystemExit``/``KeyboardInterrupt``/``GeneratorExit`` propagate
    through this helper without recording — they signal interpreter
    shutdown, not plugin behaviour. The dispatch attempt that triggered
    them is not "audited as PLUGIN_CRASH" because the interpreter is
    going away; the audit invariant ("if it's not recorded, it didn't
    happen") only meaningfully applies to outcomes the interpreter is
    around to record.

    Args:
        recorder: the active :class:`ComposerToolRecorder` (typically a
            :class:`BufferingRecorder` in the web composer).
        audit: the open audit envelope from :func:`begin_dispatch`.
        do_dispatch: an async callable that performs the actual handler
            call (e.g. ``await run_sync_in_worker(execute_tool, ...)``).
        version_after_provider: extracts the post-dispatch state version
            from the handler's return value. The compose loop knows the
            shape (``ToolResult.updated_state.version``); the helper
            accepts a callback rather than coupling to that type.
        arg_error_payload_factory: builds the structured ``error_payload``
            recorded into ``result_canonical`` on the ARG_ERROR path so
            the audit row mirrors the LLM-facing tool message.

    Returns:
        :class:`DispatchOutcome` carrying the handler result and the
        captured ``version_after``.

    Raises:
        ``ToolArgumentError``: re-raised after recording ARG_ERROR.
        ``AssertionError``/``MemoryError``/``RecursionError``/``SystemError``:
            re-raised after recording PLUGIN_CRASH.
        ``Exception`` (other classes): re-raised after recording PLUGIN_CRASH.
    """
    try:
        result = await do_dispatch()
    except ToolArgumentError as exc:
        # ARG_ERROR dispatch site — handler raised at the Tier-3 boundary.
        # The redaction discipline (class-name + safe args[0]) is enforced
        # by ToolArgumentError's safe-by-construction message: see the
        # ToolArgumentError docstring in protocol.py.
        safe_message = exc.args[0] if exc.args else "tool argument error"
        error_payload = arg_error_payload_factory(exc)
        recorder.record(
            finish_arg_error(
                audit,
                error_class="ToolArgumentError",
                error_message=str(safe_message),
                error_payload=error_payload,
            )
        )
        raise
    except (AssertionError, MemoryError, RecursionError, SystemError) as narrow_exc:
        # Narrow re-raise — record before propagating. The audit envelope
        # is already populated with pre-captured scalars; building the
        # invocation does not touch poisoned memory or invariant-broken
        # state. Closes B2: the prior bare ``raise`` here exited the
        # compose loop with an unrecorded dispatch — an audit hole on
        # the most invariant-critical class of failure.
        recorder.record(finish_plugin_crash(audit, exc=narrow_exc))
        raise
    except Exception as plugin_exc:
        # General PLUGIN_CRASH path — caller wraps with
        # ComposerPluginCrashError.capture(...) which threads the
        # recorder buffer through to the route handler.
        recorder.record(finish_plugin_crash(audit, exc=plugin_exc))
        raise

    # SUCCESS path. finish_success itself wraps canonical_json /
    # stable_hash in a sentinel-fallback try/except (B1 fix), so a
    # non-finite float in result cannot skip the audit record.
    version_after = version_after_provider(result)
    recorder.record(
        finish_success(
            audit,
            result_payload=_result_to_audit_payload(result),
            version_after=version_after,
        )
    )
    return DispatchOutcome(result=result, version_after=version_after)


def _result_to_audit_payload(result: Any) -> Mapping[str, Any]:
    """Coerce a handler result into the canonicalizable audit payload.

    The compose-loop's tool handler returns a ``ToolResult`` dataclass
    that exposes ``to_dict()``. The helper is callback-driven and
    doesn't import the concrete type, but it does need a ``Mapping``
    for :func:`finish_success`. A ``to_dict`` attribute is the canonical
    seam the compose loop already used at the earlier success site
    (``recorder.record(_finish_success(audit, result_payload=result.to_dict(), ...))``).

    If ``result`` is already a ``Mapping`` (e.g. tests passing a dict
    directly), pass through. Anything else without ``to_dict`` is a
    callsite contract violation and crashes loud — exactly the
    offensive-programming discipline the project prefers.
    """
    if isinstance(result, Mapping):
        return result
    to_dict = result.to_dict
    payload = to_dict()
    if not isinstance(payload, Mapping):
        raise TypeError(f"dispatch_with_audit: result.to_dict() returned {type(payload).__name__}, expected Mapping")
    return payload
