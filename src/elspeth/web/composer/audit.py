"""In-memory buffering recorder for the web composer service.

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

Layer: L3 (application). Imports L0 (contracts.composer_audit) and stdlib only.
"""

from __future__ import annotations

import threading

from elspeth.contracts.composer_audit import ComposerToolInvocation, ComposerToolRecorder

__all__ = ["BufferingRecorder", "audit_envelope"]


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
