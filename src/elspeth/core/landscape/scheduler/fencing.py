"""Transaction-shape chooser shared by fenced-or-legacy scheduler verbs."""

from __future__ import annotations

from contextlib import AbstractContextManager
from datetime import datetime

from sqlalchemy.engine import Connection

from elspeth.contracts.coordination import DEFAULT_RUN_LIVENESS_WINDOW_SECONDS, CoordinationToken
from elspeth.core.landscape.database import Tier1Engine, begin_write
from elspeth.core.landscape.run_coordination_repository import fenced_leader_transaction


def fenced_or_plain_write(
    engine: Tier1Engine,
    *,
    coordination_token: CoordinationToken | None,
    now: datetime,
    verb: str,
) -> AbstractContextManager[Connection]:
    """One write-intent transaction, leader-fenced when a token is supplied.

    Partial ratchet (slice 4): the following verbs now require the token
    (``None`` raises TypeError in Python before reaching this helper):
    ``complete_barrier``, ``mark_pending_sink_terminal``,
    ``mark_pending_sink_terminal_many``,
    ``terminalize_pending_sinks_with_terminal_outcomes``.

    The following verbs deliberately remain ``Optional[CoordinationToken]``
    in this slice — ``None`` falls through to the unfenced legacy arm:

    - ``complete_run`` / ``update_run_status`` / ``finalize_run``: broad
      test surface (21 / 9 / 4 files); non-leader-plane callers exist;
      ``complete_run`` immutability backstop is independent of the token.
      Tie together when all three can move as a unit.
    - ``recover_expired_leases``: 11 test files; the bare maintenance
      form is load-bearing for crashed-image construction in harness
      helpers that claim under un-registered identities by design.
      Re-evaluate after slice 5 makes the sweep leader-only end-to-end.
    - ``mark_blocked_barrier_terminal`` /
      ``mark_blocked_barrier_pending_sink_many``: legacy barrier wrappers;
      §E.3a late-arrival callers are not yet fully token-threaded; 6 / 3
      test files; defer until the wrappers are retired or fully threaded.
    - ``create_checkpoint`` (CheckpointManager): 21 test files, 113 actual
      call sites (vs the 16-file estimate in the ratchet plan); the
      orchestrator already threads the token via CheckpointCoordinator;
      ratcheting the direct CheckpointManager boundary requires
      coordination-row setup across all integration/property harnesses.
      Deferred until the integration suite is token-aware end-to-end.
    """
    if coordination_token is None:
        return begin_write(engine)
    return fenced_leader_transaction(
        engine,
        token=coordination_token,
        now=now,
        window_seconds=DEFAULT_RUN_LIVENESS_WINDOW_SECONDS,
        verb=verb,
    )
