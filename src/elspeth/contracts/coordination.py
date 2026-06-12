"""Multi-worker run-coordination contracts (epoch 21, ADR-030).

Value objects threaded through the leader/follower coordination protocol
(design: notes/option-c-multi-worker-coordination-design-2026-06-11.md):

- :func:`mint_worker_id` — the §A.1 single-use worker identity. Minted at
  registration (``begin_run`` / ``acquire_run_leadership`` / ``join``),
  doubles as the scheduler ``lease_owner`` string. Role is a registry
  attribute, never parsed from the string.
- :class:`CoordinationToken` — the fencing token carried by value into every
  leader-fenced verb; never re-read mid-run. ``leader_epoch`` is THE fence:
  a takeover bumps it, instantly refusing the deposed leader everywhere.
- :class:`LeaderInfo` — read-only seat snapshot (``live_leader``); the
  slice-4 entry-guard precision upgrade consumes it.
- :class:`CoordinationSnapshot` — returned by ``worker_heartbeat`` so
  followers learn of seat handover on their existing cadence (§A.3;
  consumed by the slice-4 heartbeat thread).
- :class:`RegisteredWorker` — forensic registry row surfaced by the §B.4
  BUSY-takeover diagnostic (``WriteLockHeldError``): pid is the one forensic
  column with a functional consumer (the operator's SIGKILL target).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Final
from uuid import uuid4

__all__ = [
    "DEFAULT_RUN_LIVENESS_WINDOW_SECONDS",
    "CoordinationSnapshot",
    "CoordinationToken",
    "LeaderInfo",
    "RegisteredWorker",
    "mint_worker_id",
]

# Run-level liveness window (design §A.3): window >= 4 x (beat interval +
# busy_timeout) = 4 x (15 s + 5 s) = 80 s at defaults. Sized against
# worst-case write-lock occupancy, NOT the longest LLM call — the slice-4
# heartbeat thread keeps an idle leader live; until then every fenced verb
# extends the seat as a side effect (identity+epoch fence, never expiry).
DEFAULT_RUN_LIVENESS_WINDOW_SECONDS: Final[float] = 80.0


def mint_worker_id(run_id: str) -> str:
    """Mint a fresh single-use worker identity (design §A.1).

    ``worker:{run_id}:{uuid4().hex}`` — minted at registration and used as
    the scheduler ``lease_owner``. Identities are single-use: a ``departed``
    or ``evicted`` registry row never returns to ``active``; a returning
    process mints a fresh identity and re-admits.
    """
    return f"worker:{run_id}:{uuid4().hex}"


@dataclass(frozen=True, slots=True)
class CoordinationToken:
    """Leader fencing token: ``(run_id, worker_id, leader_epoch)``.

    Minted by ``register_run_leader`` (epoch 1, in ``begin_run``'s
    transaction) or ``acquire_run_leadership`` (takeover CAS, epoch+1).
    Threaded by value into every leader-fenced verb; the verify-and-extend
    fence (``verify_and_extend_leader_fence``) CAS-matches all three fields
    against ``run_coordination`` as the first statement of the verb's
    transaction.
    """

    run_id: str
    worker_id: str
    leader_epoch: int


@dataclass(frozen=True, slots=True)
class LeaderInfo:
    """Read-only view of a run's leader seat (``live_leader``).

    ``seat_live`` is the §C.1 liveness predicate evaluated at the caller's
    ``now``: ``leader_heartbeat_expires_at >= now``. A dead seat
    (``seat_live=False``) is the admissible-takeover signal the slice-4
    entry guard consumes.
    """

    run_id: str
    leader_worker_id: str
    leader_epoch: int
    leader_heartbeat_expires_at: datetime
    seat_live: bool


@dataclass(frozen=True, slots=True)
class CoordinationSnapshot:
    """Seat state observed atomically by ``worker_heartbeat`` (§A.3).

    ``worker_active`` is False when the heartbeat CAS missed (this worker is
    no longer ``active`` — departed at finalize, or evicted); the slice-4
    heartbeat thread latches its coordination-lost flag on that, never on a
    DB error. ``leader_worker_id`` is None for a vacant seat. A leader-mode
    process observing a snapshot whose leader is not itself treats that as
    fatal (deposed even if its registry row was not yet evicted).
    """

    leader_worker_id: str | None
    leader_epoch: int
    seat_live: bool
    worker_active: bool


@dataclass(frozen=True, slots=True)
class RegisteredWorker:
    """Forensic ``run_workers`` registry row (§B.4 BUSY-takeover diagnostic)."""

    worker_id: str
    role: str
    status: str
    pid: int | None
    hostname: str | None
