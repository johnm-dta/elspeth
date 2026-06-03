"""Dataclasses passed across the async/sync boundary in
``SessionServiceImpl.persist_compose_turn`` (spec §5.2.1).

These types have no async behaviour; they are pure data containers that
the compose loop populates in async land and then hands to the sync
worker via ``_run_sync``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from elspeth.contracts.freeze import freeze_fields
from elspeth.web.sessions.protocol import CompositionStateData


@dataclass(frozen=True, slots=True)
class StatePayload:
    """Snapshot of a CompositionState ready for insertion.

    Composes the existing :class:`CompositionStateData` input DTO (which
    carries the per-column state contents and is already
    ``freeze_fields``-protected by its own ``__post_init__``) with a
    ``derived_from_state_id`` that records lineage from the pre-call
    state.

    **B1 (Phase 1 plan-review synthesis): no ``version`` field.** Earlier
    drafts of this dataclass carried a caller-supplied ``version: int``.
    That contract was unsafe: in Phase 3 the compose loop reads
    ``MAX(version)`` outside the session write lock and then
    dispatches into the ``_session_write_lock``-protected ``_insert_composition_state``
    helper. Two concurrent allocators can both compute ``MAX+1`` before
    either acquires the lock; the loser's INSERT then hits
    ``uq_composition_state_version`` and the locked path's
    ``IntegrityError`` handler classifies it as a Tier-1 audit-integrity
    violation — fabricating a Tier-1 violation from a benign contention
    loss. SLO threshold for ``tool_row_integrity_violation_total`` is 0,
    so the alert fires on a non-event. Under ELSPETH's auditability
    standard this is evidence-tampering-class harm: the audit trail
    asserts a violation that did not occur.

    The fix is structural: the version is no longer a payload field at
    all. ``_insert_composition_state`` allocates it under the held
    session write lock via ``SELECT COALESCE(MAX(version), 0) + 1 FROM
    composition_states WHERE session_id = :sid`` (see Task 10). With
    version off the payload, the dual-allocator race becomes
    structurally impossible: every caller must be inside the lock context to invoke
    the helper, and the SELECT-MAX-then-INSERT sequence is atomic
    against every other writer for that session.

    The contract is fixed in Phase 1 even though the call shape only
    manifests in Phase 3 — making the helper impossible to misuse from
    Phase 3 onward is cheaper than catching the misuse later.

    Why this shape rather than a single JSON blob:

    The ``composition_states`` table has eight content columns
    (``source/nodes/edges/outputs/metadata_/is_valid/validation_errors/derived_from_state_id``).
    The plan's earlier ``payload_json: str`` design was a hallucination —
    no ``payload`` column exists, and the existing
    ``save_composition_state`` insert at ``service.py:1005-1020``
    (function body starts at 948) writes
    each column individually via a method-local ``_enveloped(...)`` helper
    and ``deep_thaw(...)`` patterns. Task 10 extracts that rule to the
    shared ``_enveloped_state_column(...)`` helper. ``StatePayload``
    mirrors that real schema by reusing :class:`CompositionStateData` rather than
    duplicating its fields and freeze-guard machinery.

    ``derived_from_state_id`` is ``str | None`` rather than ``str``
    because the existing inline inserts at ``service.py:1005-1020``
    (initial state) currently set it to ``None``. The compose-loop
    caller in Phase 3 will always supply a non-None value (every
    tool-call-driven state advance has a predecessor).

    Note on freeze-fields. ``derived_from_state_id`` is a scalar (or
    ``None``); ``data`` is a frozen ``CompositionStateData`` with its own
    ``freeze_fields`` discipline. No ``__post_init__`` is required on
    ``StatePayload`` itself — ``frozen=True`` is sufficient because
    every remaining field is either scalar or an already-frozen
    dataclass. (Removing ``version`` did not change this analysis;
    ``version: int`` was scalar too.)
    """

    data: CompositionStateData
    derived_from_state_id: str | None = None


@dataclass(frozen=True, slots=True)
class _ToolOutcome:
    """Result of one tool call within a compose turn.

    The ``call`` and ``response`` fields are typed ``Any`` because the
    compose loop populates them with framework-specific objects (LiteLLM
    ToolCall, Pydantic response models, plain dicts, etc.) that this
    module deliberately does not couple to. At runtime these values are
    typically dicts or ``Mapping`` types, so the ``frozen=True``
    declaration alone is a lie about immutability — the dataclass
    attribute cannot be reassigned, but the dict it points to remains
    fully mutable through the reference.

    CLAUDE.md's ``freeze_fields`` contract is unconditional for frozen
    dataclasses with container/Any fields: ``__post_init__`` must call
    ``freeze_fields`` on every such field. ``deep_freeze`` (which
    ``freeze_fields`` invokes per field) is identity-preserving for
    values that are already frozen, so the cost of running it on
    already-immutable inputs (e.g. an integer-only ``call``, which
    won't happen in practice but is contractually possible) is zero.
    """

    call: Any  # ToolCall — typed in protocol module
    response: Any  # tool response object or None on error
    error_class: str | None
    error_message: str | None
    pre_version: int
    post_version: int

    def __post_init__(self) -> None:
        freeze_fields(self, "call", "response")


@dataclass(frozen=True, slots=True)
class RedactedToolRow:
    """One persisted tool row, with redactions already applied."""

    tool_call_id: str
    content: str  # JSON-serialised redacted response
    composition_state_payload: StatePayload | None  # set iff state advanced


@dataclass(frozen=True, slots=True)
class AuditOutcome:
    """Disposition returned by SessionServiceImpl.persist_compose_turn (§5.2.2).

    Two outcome shapes:

    - **Success.** ``assistant_id`` is set, ``unwind_audit_failed=False``.
      Caller continues with the new assistant message id.
    - **Tool failed AND audit unwind failed.** ``assistant_id=None``,
      ``unwind_audit_failed=True``. Caller raises the captured plugin
      crash; the audit failure is recorded by ``persist_compose_turn``
      via counter increment + ``slog.warning`` (permitted under
      CLAUDE.md primacy because the audit system itself failed).

    There is NO tier-1-violation outcome shape. When the audit
    database fails AND no plugin crash is in flight,
    ``persist_compose_turn`` raises
    :class:`elspeth.contracts.errors.AuditIntegrityError` directly,
    chained from the underlying ``OperationalError`` via ``raise ...
    from audit_exc``. The exception is registered in
    ``TIER_1_ERRORS`` (via the ``@tier_1_error`` decoration on
    ``AuditIntegrityError``) so ``except Exception:`` blocks cannot
    silently swallow it. The caller has no opportunity to ignore the
    failure — this is the Tier-1 crash doctrine ("Bad data in the
    audit trail = crash immediately") encoded structurally rather
    than asked nicely.

    Why ``unwind_audit_failed`` stays a flag-return (not a raise):
    when a tool plugin has crashed in flight, the caller already has
    a captured plugin-crash exception to raise. Surfacing a separate
    audit exception would mask the original tool failure. The flag
    tells the caller "your raise should ALSO record this audit
    failure," and the counter + slog inside ``persist_compose_turn``
    have already done so.

    Closes synthesised review finding H1 (audit primacy via
    return-flag instead of raised exception violates Tier-1 doctrine).
    """

    assistant_id: str | None
    unwind_audit_failed: bool
    current_state_id: str | None = None

    def __post_init__(self) -> None:
        # Success and unwind-failure are the only two valid shapes.
        # Reject any combination that would make the outcome ambiguous.
        if self.assistant_id is not None and self.unwind_audit_failed:
            raise ValueError(
                "AuditOutcome: unwind_audit_failed=True is incompatible with "
                "assistant_id being set; the unwind path cannot have produced "
                "an assistant id"
            )
