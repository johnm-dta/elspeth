"""JoinAdmissionService: §B.1 atomic follower admission (ADR-030).

Extracted from ``Orchestrator.join_run`` / ``Orchestrator._join_preflight``
(filigree elspeth-9e71ae82a4). ``Orchestrator.join_run`` remains the public
entry point and delegates here verbatim.

Behaviour-preserving: preflight ordering (filesystem checks BEFORE touching
the registry), the config-hash recipe (``stable_hash(resolve_config(settings))``),
and the single-transaction ``admit_follower`` call are unchanged. Tests that
previously patched ``…orchestrator.core.resolve_config`` / ``.stable_hash`` /
``.os.access`` patch this module instead.
"""

from __future__ import annotations

import os
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

from elspeth.contracts.coordination import (
    DEFAULT_RUN_LIVENESS_WINDOW_SECONDS,
    mint_worker_id,
)
from elspeth.contracts.errors import JoinRefusedError
from elspeth.core.canonical import stable_hash
from elspeth.core.config import resolve_config
from elspeth.core.landscape.factory import RecorderFactory

if TYPE_CHECKING:
    from elspeth.core.config import ElspethSettings
    from elspeth.core.landscape import LandscapeDB


class JoinAdmissionService:
    """Owns the cooperative follower attach path (NOT a ``resume()`` variant)."""

    def __init__(self, *, db: LandscapeDB) -> None:
        self._db = db

    @staticmethod
    def _join_preflight(db: LandscapeDB, run_id: str) -> None:
        """§B.1 step 0: filesystem write-access preflight for the join path.

        Verifies that the joining process can write to the DB file, its
        containing directory, and any existing ``-wal`` / ``-shm`` sidecars
        BEFORE touching the registry.  Raises :class:`JoinRefusedError` with
        an actionable path and description on any failure so the operator
        knows exactly what permission is missing.

        SQLCipher: passphrase is verified implicitly by LandscapeDB's PRAGMA
        probe at open time (the engine construction that precedes this call
        fails with an opaque SQLite error if the passphrase is wrong).  This
        preflight only handles filesystem-level write access.

        Non-SQLite (Postgres): no filesystem sidecars exist; skip silently.
        """
        url = db.connection_string
        if not url.startswith("sqlite"):
            # Postgres / other backends: no filesystem sidecars to check.
            return

        # Use SQLAlchemy's make_url to reliably extract the database path
        # (handles sqlite:///path, sqlite:////abs/path, and :memory: forms).
        from sqlalchemy.engine.url import make_url as _make_url

        parsed_url = _make_url(url)
        db_file = parsed_url.database
        if db_file is None or db_file in ("", ":memory:"):
            return  # in-memory: no filesystem checks needed

        db_path = Path(db_file)
        if not db_path.is_absolute():
            db_path = Path.cwd() / db_path

        checks: list[tuple[Path, str]] = [
            (db_path, "write access to the audit DB file"),
            (db_path.parent, "write access to the audit DB directory"),
        ]
        for sidecar_suffix in ("-wal", "-shm"):
            sidecar = db_path.parent / (db_path.name + sidecar_suffix)
            if sidecar.exists():
                checks.append((sidecar, f"write access to {sidecar_suffix} sidecar"))

        for path, description in checks:
            if not os.access(path, os.W_OK):
                raise JoinRefusedError(
                    run_id,
                    f"filesystem preflight failed: {description} at {path} — "
                    "ensure the joining process has write permission "
                    "(shared group + group-writable state directory required for "
                    "cross-uid joins; see ADR-030 §B.1 step 0 and the slice-6 runbook)",
                )

    def join_run(
        self,
        run_id: str,
        settings: ElspethSettings,
        *,
        now: datetime | None = None,
        window_seconds: float | None = None,
    ) -> str:
        """§B.1: atomic follower admission — new public entry point (ADR-030).

        NOT a ``resume()`` variant.  ``resume()`` keeps refusing
        RUNNING-with-live-leader; ``join_run`` is the cooperative follower
        attach path.

        Steps performed (design §B.1):

        0. Filesystem preflight — write access to DB file + dir + any
           existing ``-wal``/``-shm`` sidecars.  Raises :class:`JoinRefusedError`
           naming the path if the joining process cannot write.

        1. DB is already open through ``self._db`` (the caller's
           ``LandscapeDB`` carries the PRAGMA probe + epoch check — G28
           cross-process uniformity by construction).

        2. Atomic admission (one ``BEGIN IMMEDIATE`` transaction via
           :meth:`RunCoordinationRepository.admit_follower`):

           - ``SELECT runs.status, runs.config_hash`` — status must be
             ``RUNNING``, else :class:`JoinRefusedError`;
           - joiner's resolved settings hash must equal ``config_hash``,
             else refused (different pipeline ⇒ different graph + barrier
             keys);
           - ``run_coordination`` seat must be live
             (``leader_heartbeat_expires_at > now``), else refused
             ("no live leader — use ``elspeth resume``");
           - ``INSERT run_workers`` (role='follower', status='active') +
             ``worker_register`` event.  COMMIT.

        Args:
            run_id: Landscape run ID to join.
            settings: The joining process's resolved ``ElspethSettings``.
                Its ``stable_hash(resolve_config(settings))`` is compared
                to ``runs.config_hash``; they must be equal.
            now: Clock injection for tests (defaults to ``datetime.now(UTC)``).
            window_seconds: Heartbeat liveness window (defaults to
                :data:`~elspeth.contracts.coordination.DEFAULT_RUN_LIVENESS_WINDOW_SECONDS`).

        Returns:
            The minted ``worker_id`` string (``worker:{run_id}:{uuid4().hex}``)
            so the caller can construct a follower-mode ``RowProcessor`` with
            ``lease_owner=worker_id``.

        Raises:
            JoinRefusedError: Filesystem preflight failed, run is not RUNNING,
                config hash mismatch, or no live leader seat.
        """

        _now = now if now is not None else datetime.now(UTC)
        _window = window_seconds if window_seconds is not None else DEFAULT_RUN_LIVENESS_WINDOW_SECONDS

        # Step 0: filesystem preflight BEFORE touching the registry.
        self._join_preflight(self._db, run_id)

        # Step 1: DB already open through self._db (PRAGMA probe + epoch
        # check inherited at LandscapeDB construction — G28 uniformity).

        # Step 2: atomic admission via the slice-5 RunCoordinationRepository
        # surface.  The joiner computes its own config_hash from its settings
        # using the same stable_hash(resolve_config(settings)) recipe that
        # begin_run stored.
        joiner_config_hash = stable_hash(resolve_config(settings))
        worker_id = mint_worker_id(run_id)

        factory = RecorderFactory(self._db, payload_store=None)
        factory.run_coordination.admit_follower(
            run_id=run_id,
            worker_id=worker_id,
            config_hash=joiner_config_hash,
            now=_now,
            window_seconds=_window,
        )

        return worker_id
