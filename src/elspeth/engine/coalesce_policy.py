"""Coalesce policy matrix: the single source of merge/fail/wait verdicts.

Extracted from ``CoalesceExecutor`` (filigree elspeth-2d43291212). The
executor used to dispatch on ``settings.policy`` in three independent
places, each re-expressing the same threshold matrix:

1. **Arrival** — ``_should_merge()``: "merge now, or hold?"
2. **Timeout / flush** — ``_resolve_pending()``: "merge what arrived, or fail?"
3. **Branch loss** — ``_evaluate_after_loss()``: "merge, fail, or keep waiting?"

Any policy change required coordinated edits at all three sites; drift would
silently diverge arrival vs timeout vs loss behaviour of the audit-critical
barrier. :func:`decide_coalesce` now owns the whole matrix — one arm per
policy covering every event — and the executor's three sites are thin
delegates that map the returned :class:`CoalesceDecision` onto their
existing merge/fail/hold actions.

Behaviour-preserving. Invariants held exactly:
  - every predicate is transcribed verbatim from its original site
    (``require_all`` completeness, ``first`` merges-on-first-arrival,
    quorum ``>=`` comparisons, ``best_effort`` arrived+lost accounting,
    quorum-impossibility checked before quorum-met on loss);
  - all failure-reason strings are byte-identical (they are
    audit-load-bearing: hashed by ``compute_error_hash`` and serialized
    to Landscape via ``CoalesceFailureReason``);
  - the ``'first'``-policy arrived-pending invariant crash and the
    unknown-policy crash raise ``RuntimeError`` with the same messages;
  - :func:`require_quorum_count` is consulted only inside the quorum arm,
    so a non-quorum config with ``quorum_count`` unset never crashes.

This module is pure and stateless — no I/O, no clock, no executor state —
so (unlike ``leader_follower_drain``) no injected seams are needed. Audit
writes (``_fail_pending`` / ``_execute_merge``) and pending-state mutation
stay in the executor; only the policy *verdict* lives here.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum

from elspeth.contracts.errors import OrchestrationInvariantError
from elspeth.core.config import CoalesceSettings


class CoalesceEvent(Enum):
    """The executor lifecycle moment a coalesce verdict is needed for."""

    ARRIVAL = "arrival"
    TIMEOUT = "timeout"
    FLUSH = "flush"
    LOSS = "loss"


class CoalesceAction(Enum):
    """The verdict: merge what arrived, fail the group, or keep waiting."""

    MERGE = "merge"
    FAIL = "fail"
    WAIT = "wait"


@dataclass(frozen=True, slots=True)
class CoalesceDecision:
    """A policy verdict plus (for FAIL) its machine-readable reason.

    ``failure_reason`` strings are audit-load-bearing — they feed
    ``compute_error_hash`` and are serialized to Landscape — so they are
    pinned byte-exact by tests/unit/engine/test_coalesce_policy.py.
    """

    action: CoalesceAction
    failure_reason: str | None = None

    def __post_init__(self) -> None:
        if (self.failure_reason is not None) != (self.action is CoalesceAction.FAIL):
            raise OrchestrationInvariantError(
                f"CoalesceDecision: failure_reason must be set if and only if action is FAIL "
                f"(action={self.action.name}, failure_reason={self.failure_reason!r})"
            )

    def require_failure_reason(self) -> str:
        """Return the failure reason or crash — only FAIL decisions carry one."""
        if self.failure_reason is None:
            raise OrchestrationInvariantError(
                f"CoalesceDecision.require_failure_reason() called on a {self.action.name} decision — only FAIL decisions carry a reason"
            )
        return self.failure_reason


_MERGE = CoalesceDecision(action=CoalesceAction.MERGE)
_WAIT = CoalesceDecision(action=CoalesceAction.WAIT)


def _fail(reason: str) -> CoalesceDecision:
    return CoalesceDecision(action=CoalesceAction.FAIL, failure_reason=reason)


def require_quorum_count(settings: CoalesceSettings) -> int:
    """Return quorum_count or crash if None — config validation should have caught this."""
    if settings.quorum_count is None:
        raise RuntimeError(f"quorum_count is None for quorum policy at coalesce '{settings.name}'. This indicates a config validation bug.")
    return settings.quorum_count


def decide_coalesce(
    settings: CoalesceSettings,
    event: CoalesceEvent,
    *,
    arrived_count: int,
    lost_branches: Mapping[str, str],
    row_id: str | None = None,
) -> CoalesceDecision:
    """Decide merge/fail/wait for a coalesce group at a lifecycle event.

    Structural guarantees (relied on by the executor's delegates):
      - ARRIVAL never returns FAIL (an arrival either merges or holds);
      - TIMEOUT and FLUSH never return WAIT (resolution is forced).

    Args:
        settings: Coalesce settings for this point. ``settings.name`` equals
            the coalesce name at every call site — the executor registers
            settings keyed by ``settings.name``.
        event: Which lifecycle moment triggered the evaluation.
        arrived_count: Number of branches that have arrived
            (``len(pending.branches)``).
        lost_branches: Branch name → loss reason for branches that will
            never arrive (``pending.lost_branches``).
        row_id: Source row id, used only in the ``'first'``-policy invariant
            crash message; callers on the TIMEOUT/FLUSH paths always provide it.

    Returns:
        The policy verdict. FAIL decisions always carry a ``failure_reason``.

    Raises:
        RuntimeError: Unknown policy; quorum policy with ``quorum_count``
            unset (config validation bug); ``'first'`` policy with arrived
            branches at TIMEOUT/FLUSH (``'first'`` merges immediately on
            arrival, so an arrived pending branch is a bug in accept()).
    """
    expected_count = len(settings.branches)
    lost_count = len(lost_branches)

    if settings.policy == "require_all":
        if event is CoalesceEvent.ARRIVAL:
            return _MERGE if arrived_count == expected_count else _WAIT
        if event is CoalesceEvent.TIMEOUT or event is CoalesceEvent.FLUSH:
            # Never a partial merge — an unresolved require_all group fails.
            return _fail("incomplete_branches")
        # event is CoalesceEvent.LOSS: ANY lost branch = immediate failure
        return _fail(f"branch_lost:{','.join(sorted(lost_branches.keys()))}")

    elif settings.policy == "first":
        if event is CoalesceEvent.ARRIVAL:
            return _MERGE if arrived_count >= 1 else _WAIT
        if event is CoalesceEvent.TIMEOUT or event is CoalesceEvent.FLUSH:
            # Only a zero-arrival loss-created pending entry can reach
            # resolution: 'first' merges immediately on arrival.
            if arrived_count == 0:
                return _fail("first_timeout_no_arrivals" if event is CoalesceEvent.TIMEOUT else "all_branches_lost")
            raise RuntimeError(
                f"Invariant violation: 'first' policy should never have arrived pending branches "
                f"at coalesce '{settings.name}', row_id='{row_id}'. "
                f"'first' merges immediately on arrival — bug in accept()."
            )
        # event is CoalesceEvent.LOSS: if every branch is lost before any
        # arrival, fail the row cleanly. If any branch arrived, accept()
        # should already have merged and marked the key complete.
        if arrived_count == 0 and arrived_count + lost_count >= expected_count:
            return _fail("all_branches_lost")
        return _WAIT

    elif settings.policy == "quorum":
        quorum = require_quorum_count(settings)
        if event is CoalesceEvent.ARRIVAL:
            return _MERGE if arrived_count >= quorum else _WAIT
        if event is CoalesceEvent.TIMEOUT or event is CoalesceEvent.FLUSH:
            if arrived_count >= quorum:
                return _MERGE
            return _fail("quorum_not_met_at_timeout" if event is CoalesceEvent.TIMEOUT else "quorum_not_met")
        # event is CoalesceEvent.LOSS: check if quorum is now impossible
        max_possible = expected_count - lost_count
        if max_possible < quorum:
            return _fail(f"quorum_impossible:need={quorum},max_possible={max_possible}")
        # Check if arrived count already meets quorum
        if arrived_count >= quorum:
            return _MERGE
        return _WAIT  # Still waiting

    elif settings.policy == "best_effort":
        # Lost branches count as "accounted for" — they won't arrive but we
        # know about them.
        if event is CoalesceEvent.ARRIVAL:
            return _MERGE if arrived_count + lost_count >= expected_count else _WAIT
        if event is CoalesceEvent.TIMEOUT or event is CoalesceEvent.FLUSH:
            if arrived_count > 0:
                return _MERGE
            return _fail("best_effort_timeout_no_arrivals" if event is CoalesceEvent.TIMEOUT else "all_branches_lost")
        # event is CoalesceEvent.LOSS: all branches accounted for (arrived + lost)?
        if arrived_count + lost_count >= expected_count:
            if arrived_count > 0:
                return _MERGE
            return _fail("all_branches_lost")
        return _WAIT  # Still waiting for remaining branches

    else:
        raise RuntimeError(f"Unknown coalesce policy: {settings.policy!r}")
