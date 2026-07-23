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
:func:`dispatch_with_audit`. The helper guarantees "exactly one
``recorder.record(...)`` per dispatch, fires in ``finally`` regardless
of which path was taken" — structurally mirroring the ``try/finally``
shape used by ``composer_mcp/server.py:call_tool`` and
``AuditedLLMClient.chat_completion``. Pulling these helpers out of
``service.py`` and into a single module makes the contract a structural
property of the helper, not a procedural property of the loop.

Layer: L3 (application). Imports L0 (contracts.composer_audit), L1
(core.canonical), L3 (web.composer.protocol), and stdlib only.
"""

from __future__ import annotations

import asyncio
import sys
import threading
import time
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from typing import Any, Final

import rfc8785
from pydantic import BaseModel, ValidationError

from elspeth.contracts.composer_audit import (
    ComposerToolInvocation,
    ComposerToolRecorder,
    ComposerToolStatus,
)
from elspeth.contracts.composer_llm_audit import (
    ComposerChatTurn,
    ComposerChatTurnRecorder,
    ComposerLLMCall,
    ComposerLLMCallRecorder,
)
from elspeth.core.canonical import canonical_json, stable_hash
from elspeth.web.composer.protocol import ToolArgumentError

__all__ = [
    "BufferingRecorder",
    "DispatchAudit",
    "DispatchOutcome",
    "audit_envelope",
    "begin_dispatch",
    "begin_dispatch_or_arg_error",
    "build_canonicalization_sentinel",
    "canonicalize_pydantic_cause",
    "dispatch_with_audit",
    "finish_arg_error",
    "finish_cancelled",
    "finish_plugin_crash",
    "finish_success",
    "llm_call_audit_envelope",
    "rebind_dispatch_arguments",
]

_CANCELLATION_REASON_CODES: Final[frozenset[str]] = frozenset(
    {
        "cancelled",
        "coordinator_cancelled",
        "sibling_failure",
    }
)


# ---------------------------------------------------------------------------
# Audit-payload normalization + sentinel-diagnostic helpers.
#
# Centralized at the audit ingress (consumed by ``finish_success`` /
# ``finish_arg_error`` / ``begin_dispatch_or_arg_error``) so every audit
# row goes through the same Pydantic-aware normalization and the same
# sentinel-shape discipline. This is the single choke point on the
# SUCCESS path: regular dispatch (via ``_result_to_audit_payload`` →
# ``ToolResult.to_dict``) AND cache-hit replay (which hand-builds a
# slim audit dict in ``service.py``) both flow through ``finish_success``.
# Mirrors the working pattern already used by the standalone composer
# MCP server (``composer_mcp/server.py:_ensure_serializable``).
# ---------------------------------------------------------------------------


def _normalize_audit_payload(value: Any) -> Any:
    """Convert Pydantic ``BaseModel`` instances to plain dicts recursively.

    The web composer audit path canonicalizes payloads via
    :func:`elspeth.core.canonical.canonical_json`, which delegates to
    :mod:`rfc8785` — a JCS implementation that rejects non-JSON types
    including Pydantic ``BaseModel``. Discovery tools
    (``list_sources`` / ``list_transforms`` / ``list_sinks`` /
    ``get_plugin_schema``) carry ``PluginSummary`` /
    ``PluginSchemaInfo`` instances through ``ToolResult.data``, so
    without this normalization the audit canonicalization fails and
    the sentinel-fallback obliterates the actual result body.

    This is the same shape as
    ``composer_mcp/server.py:_ensure_serializable``. The web composer
    centralizes it at the audit ingress (rather than at the dispatch
    boundary) because the SUCCESS path has two call sites — regular
    dispatch via ``ToolResult.to_dict`` and cache-hit replay via a
    hand-built audit dict — and only ``finish_success`` is downstream
    of both.

    Containers (``Mapping``, ``list``, ``tuple``) are walked
    recursively. Scalars and opaque objects pass through unchanged
    (rfc8785 will reject anything that survives this pass that it
    can't serialize, and the sentinel-canonical fallback will fire).
    """
    if isinstance(value, BaseModel):
        return value.model_dump()
    if isinstance(value, Mapping):
        return {k: _normalize_audit_payload(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_normalize_audit_payload(item) for item in value]
    return value


def build_canonicalization_sentinel(
    exc: BaseException,
    payload: Any,
) -> dict[str, object]:
    """Build the sentinel that replaces a non-canonicalizable audit payload.

    The status-quo sentinel records only
    ``{"_canonicalization_error": "<exc class>"}`` which makes a
    recurrence opaque: an auditor reading
    ``result_canonical='{"_canonicalization_error":"CanonicalizationError"}'``
    cannot tell *what* failed without correlating with operational
    logs. This helper enriches the sentinel with bounded, leak-safe
    diagnostic metadata so the audit trail itself is sufficient
    forensic evidence.

    Sentinel shape:

    - ``_canonicalization_error`` — exception class name (status quo).
    - ``_canonicalization_detail`` — exception ``str(exc)`` only when
      ``exc`` is an :class:`rfc8785.CanonicalizationError`. By spec
      (verified empirically on rfc8785) those messages are
      type-name or JCS-rule strings such as ``"unsupported type:
      <class 'X'>"`` or ``"inf is not representable in JCS"`` — they
      never echo payload values. Other ``ValueError`` / ``TypeError``
      paths can echo Tier-3 data (e.g. ``core/canonical.py``'s Decimal
      check interpolates the offending value into its message); for
      those the detail field is omitted to prevent a Tier-3 leak into
      the Tier-1 audit row.
    - ``_payload_keys`` — sorted top-level keys of the failed payload
      when it is a Mapping. Keys are schema metadata (names like
      "data", "validation", "version"); values are NOT captured. This
      lets an auditor identify the failure shape (discovery-tool
      result vs compose-state mutation vs unparseable arguments)
      without leaking content.

    All captured metadata is bounded: no payload values, no exception
    messages from non-allowlisted classes, no recursion into nested
    structures.
    """
    sentinel: dict[str, object] = {"_canonicalization_error": type(exc).__name__}
    if isinstance(exc, rfc8785.CanonicalizationError):
        # Bounded by spec: rfc8785 messages are short and value-free.
        # The 512-char cap is belt-and-braces against future rfc8785
        # changes that might inline a longer schema fragment.
        sentinel["_canonicalization_detail"] = str(exc)[:512]
    if isinstance(payload, Mapping):
        sentinel["_payload_keys"] = sorted(str(k) for k in payload)
    return sentinel


class BufferingRecorder(ComposerToolRecorder, ComposerLLMCallRecorder, ComposerChatTurnRecorder):
    """Append-only in-memory buffer for composer audit records.

    Used inside :meth:`ComposerServiceImpl._compose_loop`. After Phase 3,
    compose-loop tool rows are committed by
    ``SessionServiceProtocol.persist_compose_turn_async`` inside the loop;
    the route-layer ``tool_invocations`` drain is retained only for older
    non-loop carriers. LLM call and guided chat-turn sidecars still use this
    buffer as their route-persisted staging area.

    Threading: ``record()`` is safe to call from any thread. The compose
    loop dispatches synchronously to a worker via ``run_sync_in_worker``
    and records from the asyncio event loop, but the lock is cheap
    insurance for any future restructure that spawns parallel workers.
    """

    def __init__(self) -> None:
        self._invocations: list[ComposerToolInvocation] = []
        self._llm_calls: list[ComposerLLMCall] = []
        self._chat_turns: list[ComposerChatTurn] = []
        self._lock = threading.Lock()

    def record(self, invocation: ComposerToolInvocation) -> None:
        with self._lock:
            self._invocations.append(invocation)

    def record_llm_call(self, call: ComposerLLMCall) -> None:
        with self._lock:
            self._llm_calls.append(call)

    def record_chat_turn(self, turn: ComposerChatTurn) -> None:
        """Append a :class:`ComposerChatTurn` record (Phase A slice 5).

        Persistence to the audit DB is wired by the route handler via
        the future ``_persist_chat_turns`` helper; this buffer is the
        in-memory staging area for the request's per-turn records.
        """
        with self._lock:
            self._chat_turns.append(turn)

    @property
    def invocations(self) -> tuple[ComposerToolInvocation, ...]:
        """Snapshot the current buffer as an immutable tuple."""
        with self._lock:
            return tuple(self._invocations)

    @property
    def llm_calls(self) -> tuple[ComposerLLMCall, ...]:
        """Snapshot the current LLM-call buffer as an immutable tuple."""
        with self._lock:
            return tuple(self._llm_calls)

    @property
    def chat_turns(self) -> tuple[ComposerChatTurn, ...]:
        """Snapshot the current chat-turn buffer as an immutable tuple."""
        with self._lock:
            return tuple(self._chat_turns)

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


_LLM_CALL_PUBLIC_AUDIT_FIELDS: Final[tuple[str, ...]] = (
    "model_requested",
    "model_returned",
    "status",
    "prompt_tokens",
    "completion_tokens",
    "total_tokens",
    "cached_prompt_tokens",
    "cache_creation_input_tokens",
    "cache_read_input_tokens",
    "reasoning_tokens",
    "latency_ms",
    "provider_request_id",
    "messages_hash",
    "tools_spec_hash",
    "declared_tool_names",
    "started_at",
    "finished_at",
    "error_class",
    "error_message",
    "temperature",
    "seed",
    "provider_cost",
    "provider_cost_source",
    "max_completion_tokens_requested",
    "planner_policy_hash",
    "planner_call_ordinal",
)


def _public_llm_call_audit_payload(call: ComposerLLMCall) -> dict[str, Any]:
    raw = call.to_dict()
    return {field: raw[field] for field in _LLM_CALL_PUBLIC_AUDIT_FIELDS}


def llm_call_audit_envelope(call: ComposerLLMCall) -> dict[str, object]:
    """Wrap a public-safe LLM call projection in the ``tool_calls`` envelope.

    Provider reasoning artifacts (``reasoning_content``, ``reasoning_details``,
    ``thinking_blocks``) can contain hidden prompt or tool context, so the
    session-visible sidecar exposes only metadata, hashes, usage, cost, and
    bounded error fields.
    """

    return {"_kind": "llm_call_audit", "call": _public_llm_call_audit_payload(call)}


def chat_turn_audit_envelope(turn: ComposerChatTurn) -> dict[str, object]:
    """Wrap a chat turn in the canonical ``tool_calls`` JSON envelope.

    Sibling of :func:`llm_call_audit_envelope`.  The ``_kind`` discriminator
    distinguishes this from LLM-call audit payloads so a reader of
    ``chat_messages`` can dispatch on the field without inspecting the body.

    ``turn.to_dict()`` already serialises the enum + datetimes; the envelope
    just adds the kind tag.
    """
    return {"_kind": "chat_turn_audit", "turn": turn.to_dict()}


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


def begin_dispatch_or_arg_error(
    tool_call_id: str,
    tool_name: str,
    arguments: Mapping[str, Any] | str,
    *,
    version_before: int,
    actor: str,
) -> tuple[DispatchAudit, BaseException | None]:
    """Open an audit envelope without letting malformed args bypass audit.

    ``json.loads`` accepts non-standard constants like ``NaN`` and
    ``Infinity``. The canonicalizer rejects those values, which is
    correct, but that rejection can happen before the compose loop has an
    audit row. This helper mirrors the standalone MCP path: store a
    canonical sentinel for the arguments, return the canonicalization
    exception to the caller, and let the caller record ARG_ERROR.
    """
    try:
        return (
            begin_dispatch(
                tool_call_id,
                tool_name,
                arguments,
                version_before=version_before,
                actor=actor,
            ),
            None,
        )
    except (ValueError, TypeError) as exc:
        # Pass ``arguments`` only when it is a Mapping — the raw-string
        # path goes through ``begin_dispatch`` directly and never lands
        # here, but typing-narrow the call so ``_payload_keys`` is only
        # emitted when meaningful (sentinel for ``arguments`` that's a
        # str would emit a noisy index-key list).
        sentinel = build_canonicalization_sentinel(
            exc,
            arguments if isinstance(arguments, Mapping) else None,
        )
        return (
            DispatchAudit(
                tool_call_id=tool_call_id,
                tool_name=tool_name,
                arguments_canonical=canonical_json(sentinel),
                arguments_hash=stable_hash(sentinel),
                version_before=version_before,
                started_at=datetime.now(UTC),
                started_ns=time.monotonic_ns(),
                actor=actor,
            ),
            exc,
        )


def rebind_dispatch_arguments(
    audit: DispatchAudit,
    arguments: Mapping[str, Any],
) -> DispatchAudit:
    """Bind an open dispatch envelope to final custody-safe arguments.

    Timing, actor, tool identity, and version snapshot remain those captured
    before validation.  Only the authority-bearing argument canonicalization
    changes, before custody I/O or any durable proposal write occurs.
    """
    return replace(
        audit,
        arguments_canonical=canonical_json(arguments),
        arguments_hash=stable_hash(arguments),
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

    Crash-on-anomaly (no sentinel fallback)
    ---------------------------------------
    ``result_payload`` is the SUCCESS-path output of a composer tool
    handler (``ToolResult.to_dict()`` — our own code). It is a
    first-party authored value, so a non-finite float or non-serializable
    type here means one of our handlers produced un-canonicalizable
    output: a bug in our code, not malformed external data. Per the trust
    model, ``canonical_json`` / ``stable_hash`` on our own dispatch output
    is a Tier-1-equivalent act — it crashes on anomaly rather than
    substituting a degraded sentinel and reporting SUCCESS (which would
    launder our bug into a clean-looking audit row). The audit record is
    still guaranteed: :func:`dispatch_with_audit`'s ``finally`` clause
    catches the raise, records PLUGIN_CRASH, and re-raises — so the
    failure surfaces loudly *with* an audit row, not silently. (Contrast
    the ARG_ERROR path, where the composer-LLM authors the tool
    *arguments* — genuinely Tier-3 — and
    :func:`build_canonicalization_sentinel` does produce a bounded
    sentinel rather than crash.)

    Pydantic normalization (elspeth-281f259235 fix)
    -----------------------------------------------
    Before canonicalization, ``result_payload`` is run through
    :func:`_normalize_audit_payload` which converts any nested Pydantic
    ``BaseModel`` instances to plain dicts. Discovery tool results
    (``list_sources``, ``list_transforms``, ``list_sinks``,
    ``get_plugin_schema``) carry ``PluginSummary`` / ``PluginSchemaInfo``
    instances on ``ToolResult.data``; without this step canonicalization
    would reject the ``BaseModel`` and crash the success path on output
    that is in fact legitimate — normalization keeps well-formed handler
    results canonicalizable so only genuine anomalies trip the crash.
    """
    normalized = _normalize_audit_payload(result_payload)
    # ``result_payload`` is the SUCCESS-path output of a composer tool handler —
    # ``ToolResult.to_dict()`` (our own code building structured catalog/state
    # data). It is a first-party authored value, not an external-origin one: the
    # composer-LLM authors tool *arguments* (handled on the ARG_ERROR path), never
    # the handler's *result*. A non-finite float or non-serializable object here
    # therefore means one of our handlers produced un-canonicalizable output —
    # a bug in our code, not malformed external data. Per the trust model,
    # ``canonical_json`` on our own dispatch output is a Tier-1-equivalent act:
    # crash on anomaly rather than substituting a degraded sentinel and reporting
    # SUCCESS, which would launder our bug into a clean-looking audit row.
    canon = canonical_json(normalized)
    result_hash = stable_hash(normalized)
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
    normalized_error_payload = _normalize_audit_payload(error_payload) if error_payload is not None else None
    return ComposerToolInvocation(
        tool_call_id=audit.tool_call_id,
        tool_name=audit.tool_name,
        arguments_canonical=audit.arguments_canonical,
        arguments_hash=audit.arguments_hash,
        result_canonical=canonical_json(normalized_error_payload) if normalized_error_payload is not None else None,
        result_hash=stable_hash(normalized_error_payload) if normalized_error_payload is not None else None,
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


def finish_cancelled(
    audit: DispatchAudit,
    *,
    exc: asyncio.CancelledError,
) -> ComposerToolInvocation:
    """Build a CANCELLED invocation with a closed, value-free reason code.

    ``Task.cancel(message)`` carries its message through
    :class:`asyncio.CancelledError`. The message is an external value at this
    audit boundary, so only coordinator-authored codes from the closed set are
    persisted; arbitrary text falls back to the generic ``cancelled`` code.
    """
    reason = "cancelled"
    if len(exc.args) == 1 and type(exc.args[0]) is str and exc.args[0] in _CANCELLATION_REASON_CODES:
        reason = exc.args[0]
    return ComposerToolInvocation(
        tool_call_id=audit.tool_call_id,
        tool_name=audit.tool_name,
        arguments_canonical=audit.arguments_canonical,
        arguments_hash=audit.arguments_hash,
        result_canonical=None,
        result_hash=None,
        status=ComposerToolStatus.CANCELLED,
        error_class="CancelledError",
        error_message=reason,
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

    Structural guarantee: this helper performs **exactly one**
    ``recorder.record(...)`` call per dispatch, in a ``finally`` clause
    that runs regardless of which path the dispatch took — including
    ``BaseException`` subclasses (``asyncio.CancelledError``,
    ``SystemExit``, ``KeyboardInterrupt``, ``GeneratorExit``) that the
    typed except handlers do not catch. This mirrors the
    ``try/finally`` shape used by ``composer_mcp/server.py:call_tool``
    and ``AuditedLLMClient.chat_completion``: every code path that
    enters the dispatch envelope leaves an audit record before
    propagating.

    Paths
    -----
    SUCCESS
        ``do_dispatch`` returned a value. The success branch records
        the post-dispatch state version, and the ``finally`` clause
        builds the SUCCESS invocation via :func:`finish_success`. If
        :func:`finish_success` raises while canonicalizing first-party
        handler output (a bug producing un-canonicalizable data), the
        ``finally`` clause records PLUGIN_CRASH and re-raises — the
        failure cannot bypass audit. Returns :class:`DispatchOutcome`.

    ARG_ERROR
        ``do_dispatch`` raised :class:`ToolArgumentError`. The except
        block captures the exception and the structured
        ``error_payload`` from ``arg_error_payload_factory`` (which the
        caller uses to also produce the LLM-facing tool message); the
        ``finally`` clause records ARG_ERROR via
        :func:`finish_arg_error` and the exception re-raises so the
        caller's ``except ToolArgumentError`` block runs the
        LLM-message-append branch unchanged.

    PLUGIN_CRASH (narrow re-raise)
        ``do_dispatch`` raised one of
        ``(AssertionError, MemoryError, RecursionError, SystemError)``.
        These were originally re-raised WITHOUT an audit record because
        the implementer worried about poisoned-memory work. That worry
        was misjudged: the audit envelope is already populated with
        pre-captured scalars (``audit`` is frozen), and constructing
        a :class:`ComposerToolInvocation` from those scalars plus
        ``type(exc).__name__`` is pure scalar work. The except block
        captures the exception; the ``finally`` clause records
        :class:`ComposerToolStatus.PLUGIN_CRASH` before the exception
        propagates. Closes blocker B2 from the panel review (2026-05-04).

    PLUGIN_CRASH (general)
        ``do_dispatch`` raised any other ``Exception`` subclass. Same
        pattern: capture in except, record in ``finally``, propagate
        so the caller's ``except Exception`` block can wrap with
        :meth:`ComposerPluginCrashError.capture`.

    CANCELLED / PLUGIN_CRASH (BaseException)
        ``do_dispatch`` raised an exception that does NOT inherit from
        ``Exception``. The ``finally`` clause detects the propagating
        exception via :func:`sys.exc_info`. It records
        :class:`asyncio.CancelledError` as CANCELLED with a bounded reason
        code, because coordinator cancellation is not a plugin defect. It
        records ``SystemExit``, ``KeyboardInterrupt``, and ``GeneratorExit``
        as PLUGIN_CRASH. In both cases the exception keeps propagating. The audit
        invariant — "if it's not recorded, it didn't happen" — now
        holds even at interpreter-shutdown boundaries; the in-memory
        list append is safe even if the persistence layer never gets
        the chance to flush.

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
        ``asyncio.CancelledError``: propagates after the ``finally`` clause
            records CANCELLED via :func:`sys.exc_info`.
        Other ``BaseException`` subclasses (``SystemExit``,
            ``KeyboardInterrupt``, ``GeneratorExit``): propagate after the
            ``finally`` clause records PLUGIN_CRASH.
    """
    # Outcome variables — populated by the success branch / except
    # blocks, consumed by the finally clause. The four-status discriminant
    # ensures the finally block records exactly one invocation per path
    # taken; ``status is None`` after finally entry means an unanticipated
    # ``BaseException`` propagated past the typed handlers and the
    # finally block must reconstruct the exception via ``sys.exc_info``.
    status: ComposerToolStatus | None = None
    success_result: Any = None
    success_version_after: int | None = None
    arg_error_exc: ToolArgumentError | None = None
    arg_error_payload: Mapping[str, Any] | None = None
    crash_exc: BaseException | None = None
    try:
        try:
            result = await do_dispatch()
        except ToolArgumentError as exc:
            # ARG_ERROR dispatch site — handler raised at the Tier-3
            # boundary. The redaction discipline (class-name + safe
            # args[0]) is enforced by ToolArgumentError's
            # safe-by-construction message: see the ToolArgumentError
            # docstring in protocol.py. Capture here; record in finally.
            status = ComposerToolStatus.ARG_ERROR
            arg_error_exc = exc
            arg_error_payload = arg_error_payload_factory(exc)
            raise
        except (AssertionError, MemoryError, RecursionError, SystemError) as narrow_exc:
            # Narrow re-raise — capture for finally. The audit envelope
            # is already populated with pre-captured scalars; building
            # the invocation does not touch poisoned memory or
            # invariant-broken state. Closes B2: the prior bare
            # ``raise`` here exited the compose loop with an unrecorded
            # dispatch — an audit hole on the most invariant-critical
            # class of failure.
            status = ComposerToolStatus.PLUGIN_CRASH
            crash_exc = narrow_exc
            raise
        except Exception as plugin_exc:
            # General PLUGIN_CRASH path — caller wraps with
            # ComposerPluginCrashError.capture(...) which threads the
            # recorder buffer through to the route handler.
            status = ComposerToolStatus.PLUGIN_CRASH
            crash_exc = plugin_exc
            raise

        # SUCCESS path. The final recorder.record(...) lives in
        # ``finally`` for structural symmetry; here we only capture the
        # handler result and post-dispatch version. If ``finish_success``
        # raises while canonicalizing first-party handler output, the
        # ``finally`` clause records PLUGIN_CRASH and re-raises (see the
        # try/except around finish_success below), so a canonicalization
        # failure cannot skip the audit record.
        success_result = result
        success_version_after = version_after_provider(result)
        status = ComposerToolStatus.SUCCESS
        return DispatchOutcome(result=result, version_after=success_version_after)
    finally:
        # Single record point — fires on every exit, including
        # BaseException subclasses (CancelledError, SystemExit,
        # KeyboardInterrupt, GeneratorExit) that the typed except
        # handlers above do not catch. ``status`` is ``None`` exactly
        # when such an exception is propagating; we reconstruct it via
        # sys.exc_info() and record CANCELLED for coordinator cancellation
        # or PLUGIN_CRASH for other BaseException subclasses before it leaves.
        if status is None:
            current_exc = sys.exc_info()[1]
            # Offensive guard: status==None with no propagating
            # exception is structurally impossible — every successful
            # path assigns SUCCESS before the return statement
            # evaluates the finally block. If we reach this branch
            # without an exception, an audit row would be dropped
            # silently — fail loudly instead.
            if current_exc is None:
                raise RuntimeError(
                    "dispatch_with_audit: finally entered with status=None and no propagating exception — audit invariant violated"
                )
            if isinstance(current_exc, asyncio.CancelledError):
                recorder.record(finish_cancelled(audit, exc=current_exc))
            else:
                recorder.record(finish_plugin_crash(audit, exc=current_exc))
        elif status is ComposerToolStatus.SUCCESS:
            # success_version_after was assigned alongside status.
            # Offensive guard: the dataclass requires ``int`` here, so
            # a None would crash on construction; mypy prefers the
            # explicit narrow.
            if success_version_after is None:
                raise RuntimeError("dispatch_with_audit: SUCCESS status with no version_after captured")
            try:
                invocation = finish_success(
                    audit,
                    result_payload=_result_to_audit_payload(success_result),
                    version_after=success_version_after,
                )
            except Exception as exc:
                recorder.record(finish_plugin_crash(audit, exc=exc))
                raise
            recorder.record(invocation)
        elif status is ComposerToolStatus.ARG_ERROR:
            if arg_error_exc is None:
                raise RuntimeError("dispatch_with_audit: ARG_ERROR status with no captured exception")
            safe_message = arg_error_exc.args[0] if arg_error_exc.args else "tool argument error"
            recorder.record(
                finish_arg_error(
                    audit,
                    error_class="ToolArgumentError",
                    error_message=str(safe_message),
                    error_payload=arg_error_payload,
                )
            )
        else:  # ComposerToolStatus.PLUGIN_CRASH (typed handlers)
            if crash_exc is None:
                raise RuntimeError("dispatch_with_audit: PLUGIN_CRASH status with no captured exception")
            recorder.record(finish_plugin_crash(audit, exc=crash_exc))


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


# ---------------------------------------------------------------------------
# F2 (spec §4.2.6): Pydantic ``__cause__`` canonicalization for ARG_ERROR.
#
# Placed at module tail to keep the AST body-index ordering of the existing
# functions stable — the tier-model enforcer fingerprints findings by AST
# path (``body[N]``), so inserting a new module-level def in the middle of
# the file would rotate every downstream fingerprint and force a churn of
# allowlist re-keying that has nothing to do with this change.
# ---------------------------------------------------------------------------


def canonicalize_pydantic_cause(exc: BaseException | None) -> list[dict[str, Any]] | None:
    """Canonicalize a Pydantic ``ValidationError`` chained on ``__cause__``.

    F2 disposition (spec §4.2.6): promoted-handler ``ToolArgumentError``
    sites raise ``from pydantic.ValidationError``. The chained
    ``ValidationError`` carries field-name detail (``loc``/``msg``/``type``)
    that is auditably valuable for recovery flows in Phase 3+, but
    ``ValidationError.errors()`` also exposes ``input``/``url``/``ctx``
    fields that are leak vectors — ``input`` is the rejected value
    verbatim (Tier-3, secret-bearing), ``url`` is a Pydantic docs URL
    with no audit value, and ``ctx`` may carry the rejected value in
    its context dict.

    This helper produces a leak-safe canonical projection: a fresh list
    of dicts containing ONLY ``loc`` (stringified), ``msg`` (the
    Pydantic-generated message — NOT user-supplied), and ``type`` (the
    Pydantic error-type discriminator like ``"int_parsing"``,
    ``"missing"``, ``"value_error"``).

    Behaviour
    ---------
    - ``exc is None`` → ``None`` (no chained cause to canonicalize).
    - ``exc`` is not a ``pydantic.ValidationError`` → ``None`` (the
      helper opts out cleanly so non-Pydantic causes don't synthesize
      empty audit fields).
    - ``exc.errors()`` returns empty (shouldn't happen for a real
      ValidationError but defensively) → ``None`` (recording
      ``validation_errors: []`` has no audit value; the absence of the
      key is the signal).
    - Otherwise → ``list[dict[str, Any]]`` with one dict per error.

    Loc-safety note
    ---------------
    The ``loc`` tuple contains field-path elements (model field names
    for typed fields, list indices for sequence fields). For
    ``dict[str, Any]`` fields, Pydantic only validates dict shape and
    does NOT descend into values, so ``loc`` paths cannot contain
    user-supplied dict keys under the current MANIFEST convention
    (``dict[str, Any]`` for all unknown-shape fields). If a future
    model introduces ``dict[str, TypedSubmodel]``, Pydantic WILL
    descend and ``loc`` may then contain user-supplied keys — the
    safety analysis here MUST be re-evaluated at that point.

    Tier discipline
    ---------------
    Pydantic's ``__cause__`` is a Tier-3 boundary input (the LLM's
    tool-call shape). This helper sits at the Tier-3 → Tier-1
    audit-record boundary, so defensive handling (the ``isinstance``
    check, the stringification of non-``str`` loc elements) is the
    correct discipline. ``exc.errors()`` itself is NOT wrapped in a
    ``try/except`` — if Pydantic's own ``errors()`` raises, that is a
    Pydantic-internal bug and must propagate (offensive programming).
    """
    if exc is None:
        return None
    if not isinstance(exc, ValidationError):
        return None
    raw_errors = exc.errors()
    if not raw_errors:
        return None
    canonicalized: list[dict[str, Any]] = []
    for err in raw_errors:
        # Stringify every loc element. Pydantic produces ``tuple[int |
        # str, ...]`` (ints for list-index errors); coerce to ``list[str]``
        # so the recorded shape is uniform and downstream audit consumers
        # don't have to branch on element type.
        loc_stringified = [str(piece) for piece in err["loc"]]
        canonicalized.append(
            {
                "loc": loc_stringified,
                "msg": err["msg"],
                "type": err["type"],
            }
        )
    return canonicalized
