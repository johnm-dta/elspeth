"""Explicit transaction helpers for fenced and legacy scheduler writes.

This module deliberately has no optional-authority transaction selector.  A
fenced caller must provide a :class:`CoordinationToken`; the one remaining
scheduler legacy arm must opt into a separately named unfenced helper.

Other legacy allowances documented when the scheduler ratchet landed remain
owned by their existing boundaries: ``complete_run`` / ``update_run_status`` /
``finalize_run`` in the run-lifecycle repository, and checkpoint create/delete
in ``CheckpointManager``.  This helper split neither widens nor ratchets those
independent APIs.
"""

from __future__ import annotations

from contextlib import AbstractContextManager
from datetime import datetime

from sqlalchemy.engine import Connection

from elspeth.contracts.coordination import DEFAULT_RUN_LIVENESS_WINDOW_SECONDS, CoordinationToken
from elspeth.core.landscape.database import Tier1Engine, begin_write
from elspeth.core.landscape.run_coordination_repository import fenced_leader_transaction


def require_coordination_token(
    coordination_token: CoordinationToken | None,
    *,
    verb: str,
) -> CoordinationToken:
    """Reject missing authority before a strict scheduler write can transact."""
    if coordination_token is None:
        raise TypeError(f"{verb} requires coordination_token; None cannot select an unfenced write")
    return coordination_token


def fenced_write(
    engine: Tier1Engine,
    *,
    coordination_token: CoordinationToken,
    now: datetime,
    verb: str,
) -> AbstractContextManager[Connection]:
    """Return a leader-fenced write transaction; missing authority refuses.

    The non-optional annotation prevents new Optional-authority call sites,
    while the runtime check protects Python callers that bypass static typing.
    Both contracts reject ``None`` before ``BEGIN IMMEDIATE`` is opened.
    """
    coordination_token = require_coordination_token(coordination_token, verb=verb)
    return fenced_leader_transaction(
        engine,
        token=coordination_token,
        now=now,
        window_seconds=DEFAULT_RUN_LIVENESS_WINDOW_SECONDS,
        verb=verb,
    )


def legacy_unfenced_recover_expired_leases_write(
    engine: Tier1Engine,
) -> AbstractContextManager[Connection]:
    """Return the named legacy transaction for direct-harness lease recovery.

    ``recover_expired_leases`` is intentionally still callable without a
    coordination seat by direct repository/integration harnesses that build a
    crashed image under unregistered worker identities.  Keeping this helper
    token-free and recovery-specific makes the unfenced choice visible at the
    sole authorized source call site without trusting a caller-supplied verb.

    The later decision between a strict public leader-recovery API and a
    separately named public unfenced API is outside this helper split.
    """
    return begin_write(engine)
