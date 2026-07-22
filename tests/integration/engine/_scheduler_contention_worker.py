# tests/integration/engine/_scheduler_contention_worker.py
"""Child-process entry point for the slice-1 two-process contention test.

Option-c design (notes/option-c-multi-worker-coordination-design-2026-06-11.md
§H "Contention (slice 1)", landing plan slice 1): two real OS processes hammer
``claim_ready`` / ``recover_expired_leases`` against one WAL SQLite file via
the real ``TokenSchedulerRepository`` write-intent path, while a third process
runs dashboard-style reads over a ``read_only=True`` handle.  Each child
serializes a metrics artifact to ``--metrics-out``; the parent test
(``test_two_process_scheduler_contention.py``) owns all assertions.

Underscore-prefixed so pytest never collects it.  Imports ONLY ``elspeth.*``,
sqlalchemy, and stdlib — it is launched by absolute path with
``PYTHONPATH=<worktree>/src`` and must not depend on the ``tests`` package.

Roles
-----
hammer
    Single-threaded loop: ``claim_ready`` every iteration with
    ``--lease-seconds`` (default 0, so every lease is instantly expired and
    the PEER's sweep genuinely reaps it — the hammer is self-feeding);
    ``recover_expired_leases`` every ``--sweep-every`` iterations (the
    self-steal guard means only the peer's leases are reaped — the exact
    multi-worker interleaving slice 1 must survive); plus a synthetic
    "heartbeat" every ``--beat-interval-ms``: the §A.3 beat shape — a
    single-row CAS UPDATE on the runs row in its own write-intent
    transaction — which is the direct latency proxy for the slice-4
    heartbeat thread.

reader
    Dashboard-style read batches on a ``from_url(read_only=True)`` handle
    (the live-dashboard shape: ``mode=ro&uri=true``, non-immutable, WAL
    sidecar visible): work-item status counts, recent scheduler events,
    the run row.

Write-lock instrumentation
--------------------------
The slice-1 write-intent listener emits ``BEGIN IMMEDIATE`` via
``exec_driver_sql`` inside the engine-level ``begin`` event, and that
statement DOES fire ``before_cursor_execute`` / ``after_cursor_execute``
(verified empirically against this worktree's listener).  So:

- ``before_cursor_execute`` on a ``BEGIN*`` statement opens a transaction
  record stamped ``txn_started`` (lock acquisition starts here — for
  IMMEDIATE this is where the busy-timeout poll begins);
- ``after_cursor_execute`` on the same statement stamps ``lock_acquired``
  (BEGIN IMMEDIATE returns ⇒ the WAL write lock is held);
- the verb wrapper stamps verb-return time, so
  ``hold_ms   = t_verb_end - lock_acquired`` (includes COMMIT completion;
  conservative by microseconds of Python) and
  ``lock_wait_ms = lock_acquired - txn_started`` (the quantity
  ``busy_timeout`` bounds).

One verb call == one transaction (claim_ready / recover_expired_leases each
run a single ``begin_write`` block), so finalization windows are unambiguous.
The recorded ``begin_stmt`` lets the parent additionally assert the verbs
actually went through ``BEGIN IMMEDIATE`` cross-process (instrumentation
self-check A12).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from sqlalchemy import event, update
from sqlalchemy.engine import Connection

from elspeth.core.landscape.database import LandscapeDB, begin_write
from elspeth.core.landscape.scheduler_repository import TokenSchedulerRepository
from elspeth.core.landscape.schema import runs_table

ROLE_HAMMER = "hammer"
ROLE_READER = "reader"

# Verb names used in the artifact's write_txns records (parent imports these).
VERB_CLAIM = "claim_ready"
VERB_RECOVER = "recover_expired_leases"
VERB_BEAT = "beat"

# Poll interval while waiting for the parent's go-file (start barrier).
_GO_POLL_SECONDS = 0.005
# Reader pacing: dashboard-style polling, not a busy spin.
_READER_PAUSE_SECONDS = 0.010


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db-url", required=True)
    parser.add_argument("--role", required=True, choices=(ROLE_HAMMER, ROLE_READER))
    parser.add_argument("--owner", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--ready-file", required=True)
    parser.add_argument("--go-file", required=True)
    parser.add_argument("--duration-seconds", type=float, required=True)
    parser.add_argument("--metrics-out", required=True)
    parser.add_argument("--sweep-every", type=int, default=8)
    parser.add_argument("--lease-seconds", type=int, default=0)
    parser.add_argument("--beat-interval-ms", type=int, default=250)
    return parser.parse_args(argv)


def _read_pragmas(db: LandscapeDB) -> dict[str, str]:
    """Actual PRAGMA readback — proves cross-process PRAGMA inheritance (G28)."""
    with db.engine.connect() as conn:
        busy_timeout = conn.exec_driver_sql("PRAGMA busy_timeout").scalar_one()
        journal_mode = conn.exec_driver_sql("PRAGMA journal_mode").scalar_one()
    return {"busy_timeout": str(busy_timeout), "journal_mode": str(journal_mode)}


def _await_start_barrier(ready_file: str, go_file: str) -> None:
    Path(ready_file).touch()
    go = Path(go_file)
    while not go.exists():
        time.sleep(_GO_POLL_SECONDS)


def _write_artifact(metrics_out: str, artifact: dict[str, Any]) -> None:
    tmp = f"{metrics_out}.tmp"
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(artifact, handle)
    os.replace(tmp, metrics_out)


def _install_write_lock_instrumentation(engine: Any, txns: list[dict[str, Any]]) -> None:
    """Stamp txn_started / lock_acquired off the BEGIN statement's cursor events.

    Single-threaded child: ``txns[-1]`` is always the in-flight transaction.
    """

    @event.listens_for(engine, "before_cursor_execute")
    def _begin_started(conn, cursor, statement, parameters, context, executemany) -> None:
        stmt = statement.lstrip().upper()
        if stmt.startswith("BEGIN"):
            txns.append({"txn_started": time.perf_counter(), "begin_stmt": statement.strip()})

    @event.listens_for(engine, "after_cursor_execute")
    def _begin_returned(conn, cursor, statement, parameters, context, executemany) -> None:
        stmt = statement.lstrip().upper()
        if stmt.startswith("BEGIN") and txns and "lock_acquired" not in txns[-1]:
            txns[-1]["lock_acquired"] = time.perf_counter()


def _run_hammer(args: argparse.Namespace) -> dict[str, Any]:
    db = LandscapeDB.from_url(args.db_url, create_tables=False)
    try:
        pragmas = _read_pragmas(db)  # BEFORE instrumentation: keeps txn records verb-only
        engine = db.engine
        repo = TokenSchedulerRepository(engine)

        txns: list[dict[str, Any]] = []
        _install_write_lock_instrumentation(engine, txns)

        write_txns: list[dict[str, Any]] = []
        errors: list[dict[str, str]] = []
        claims = 0
        claim_none = 0
        sweeps = 0
        recovered_total = 0
        claimed_work_item_ids: list[str] = []
        beats: list[float] = []

        def timed(verb: str, fn: Callable[[], Any]) -> Any:
            """Run one repository verb; finalize its write-transaction record."""
            idx = len(txns)
            t_start = time.perf_counter()
            result = fn()
            t_end = time.perf_counter()
            record: dict[str, Any] = {"verb": verb, "round_trip_ms": (t_end - t_start) * 1000.0}
            opened = txns[idx:]
            # One verb call == one transaction; tolerate (and surface via the
            # A12 self-check) anything else rather than crashing the child.
            if len(opened) == 1 and "lock_acquired" in opened[0]:
                rec = opened[0]
                record["begin_stmt"] = rec["begin_stmt"]
                record["lock_wait_ms"] = (rec["lock_acquired"] - rec["txn_started"]) * 1000.0
                record["hold_ms"] = (t_end - rec["lock_acquired"]) * 1000.0
            write_txns.append(record)
            return result

        def beat() -> None:
            """§A.3 beat shape: single-row CAS UPDATE in its own write txn."""
            with begin_write(engine) as conn:
                result = conn.execute(
                    update(runs_table).where(runs_table.c.run_id == args.run_id, runs_table.c.status == "running").values(status="running")
                )
                if result.rowcount != 1:
                    raise RuntimeError(f"beat CAS matched {result.rowcount} rows (expected 1)")

        _await_start_barrier(args.ready_file, args.go_file)

        beat_interval = args.beat_interval_ms / 1000.0
        start = time.monotonic()
        deadline = start + args.duration_seconds
        next_beat = start  # first beat fires immediately
        iteration = 0
        while time.monotonic() < deadline:
            iteration += 1
            try:
                item = timed(
                    VERB_CLAIM,
                    lambda: repo.claim_ready(
                        run_id=args.run_id,
                        lease_owner=args.owner,
                        lease_seconds=args.lease_seconds,
                        now=datetime.now(UTC),
                    ),
                )
                if item is None:
                    claim_none += 1
                else:
                    claims += 1
                    claimed_work_item_ids.append(item.work_item_id)
            except Exception as exc:
                errors.append({"where": VERB_CLAIM, "type": type(exc).__name__, "msg": str(exc)})

            if iteration % args.sweep_every == 0:
                try:
                    recovered = timed(
                        VERB_RECOVER,
                        lambda: repo.recover_expired_leases_legacy_unfenced(
                            run_id=args.run_id,
                            now=datetime.now(UTC),
                            caller_owner=args.owner,
                        ),
                    )
                    sweeps += 1
                    recovered_total += int(recovered)
                except Exception as exc:
                    errors.append({"where": VERB_RECOVER, "type": type(exc).__name__, "msg": str(exc)})

            if time.monotonic() >= next_beat:
                next_beat += beat_interval
                try:
                    timed(VERB_BEAT, beat)
                    beats.append(write_txns[-1]["round_trip_ms"])
                except Exception as exc:
                    errors.append({"where": VERB_BEAT, "type": type(exc).__name__, "msg": str(exc)})

        return {
            "role": ROLE_HAMMER,
            "owner": args.owner,
            "pragmas": pragmas,
            "errors": errors,
            "claims": claims,
            "claim_none": claim_none,
            "sweeps": sweeps,
            "recovered_total": recovered_total,
            "claimed_work_item_ids": claimed_work_item_ids,
            "write_txns": write_txns,
            "beats": beats,
            "reads": [],
            "read_batches": 0,
        }
    finally:
        db.close()


def _run_reader(args: argparse.Namespace) -> dict[str, Any]:
    db = LandscapeDB.from_url(args.db_url, read_only=True, create_tables=False)
    try:
        # Read-only engines skip the PRAGMA probe; read back what actually
        # applies on this handle so the parent can assert the contract.
        pragmas = _read_pragmas(db)

        errors: list[dict[str, str]] = []
        reads: list[float] = []
        read_batches = 0

        def read_batch(conn: Connection) -> None:
            """The dashboard read surfaces named at design §0/F10."""
            status_counts = conn.exec_driver_sql(
                "SELECT status, COUNT(*) FROM token_work_items WHERE run_id = ? GROUP BY status",
                (args.run_id,),
            ).all()
            conn.exec_driver_sql(
                "SELECT * FROM scheduler_events WHERE run_id = ? ORDER BY event_id DESC LIMIT 50",
                (args.run_id,),
            ).all()
            run_rows = conn.exec_driver_sql("SELECT * FROM runs WHERE run_id = ?", (args.run_id,)).all()
            total_items = sum(count for _status, count in status_counts)
            if total_items <= 0:
                raise RuntimeError(f"dashboard sanity: work-item status counts sum to {total_items}")
            if len(run_rows) != 1:
                raise RuntimeError(f"dashboard sanity: expected 1 run row, saw {len(run_rows)}")

        _await_start_barrier(args.ready_file, args.go_file)

        deadline = time.monotonic() + args.duration_seconds
        while time.monotonic() < deadline:
            t_start = time.perf_counter()
            try:
                with db.read_only_connection() as conn:
                    read_batch(conn)
            except Exception as exc:
                errors.append({"where": "read_batch", "type": type(exc).__name__, "msg": str(exc)})
            else:
                reads.append((time.perf_counter() - t_start) * 1000.0)
                read_batches += 1
            time.sleep(_READER_PAUSE_SECONDS)

        return {
            "role": ROLE_READER,
            "owner": args.owner,
            "pragmas": pragmas,
            "errors": errors,
            "claims": 0,
            "claim_none": 0,
            "sweeps": 0,
            "recovered_total": 0,
            "claimed_work_item_ids": [],
            "write_txns": [],
            "beats": [],
            "reads": reads,
            "read_batches": read_batches,
        }
    finally:
        db.close()


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        if args.role == ROLE_HAMMER:
            artifact = _run_hammer(args)
        else:
            artifact = _run_reader(args)
    except Exception as exc:
        print(f"contention worker internal failure: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    _write_artifact(args.metrics_out, artifact)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
