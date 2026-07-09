# tests/integration/engine/test_two_process_scheduler_contention.py
"""Slice-1 two-process scheduler contention test (option-c design §H, slice 1).

Two real OS processes hammer ``claim_ready`` / ``recover_expired_leases``
against ONE file-backed WAL SQLite audit DB through the real
``TokenSchedulerRepository`` (write-intent ``BEGIN IMMEDIATE`` path), while a
third process runs dashboard-style reads over a ``read_only=True`` handle —
the design's contention scenario: "two-process claim hammer + dashboard-style
concurrent reads ⇒ heartbeat latency stays inside the liveness window; max
write-lock hold time measured against ``busy_timeout``"
(notes/option-c-multi-worker-coordination-design-2026-06-11.md §H).

This is the correctness gate for slice 1 itself: under the old DEFERRED begin,
thousands of contended SELECT-then-UPDATE claim transactions across two
processes would produce non-retryable ``SQLITE_BUSY_SNAPSHOT``
("database is locked") aborts that ``busy_timeout`` cannot retry; assertion A1
(zero recorded errors) is the regression gate.  The printed
``SLICE1-CONTENTION`` line is the measured baseline that ADR-030's
barrier-size ceiling guidance (design risk 2) and the slice-4 rerun
(design risk 5) are compared against.

Children are launched by script path (no pickling, no pytest sys.path
dependency) with ``PYTHONPATH=<worktree>/src``; the worker module is imported
here only for its artifact-schema constants.
"""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from sqlalchemy import insert, select

import elspeth
from elspeth.contracts import NodeType
from elspeth.contracts.coordination import DEFAULT_RUN_HEARTBEAT_SECONDS
from elspeth.contracts.scheduler import TokenWorkStatus
from elspeth.contracts.schema_contract import PipelineRow, SchemaContract
from elspeth.core.landscape.database import _SQLITE_PRAGMA_INVARIANTS_FILE, LandscapeDB
from elspeth.core.landscape.scheduler_repository import TokenSchedulerRepository
from elspeth.core.landscape.schema import (
    nodes_table,
    rows_table,
    runs_table,
    token_work_items_table,
    tokens_table,
)
from tests.integration.engine import _scheduler_contention_worker as worker

WORKER_PATH = Path(worker.__file__).resolve()

RUN_ID = "run-slice1-contention"
WORK_ITEMS = 64

# --- Thresholds, derived from the audit DB's probe-enforced PRAGMA contract
# and the design's §A.3 liveness-window form, NOT hardcoded guesses. ----------
#
# busy_timeout: the probe-enforced PRAGMA invariant (database.py
# _SQLITE_PRAGMA_INVARIANTS_FILE) — every cross-process waiter polls for up to
# this long before BEGIN IMMEDIATE raises a retryable OperationalError.
BUSY_TIMEOUT_MS = int(dict(_SQLITE_PRAGMA_INVARIANTS_FILE)["busy_timeout"])  # 5000
# §A.3 heartbeat cadence — imported from the module constant added in slice 4.
RUN_HEARTBEAT_MS = int(DEFAULT_RUN_HEARTBEAT_SECONDS * 1000)  # 15_000
# One beat "allotment": a beat issued at any point must land within its
# interval plus the worst busy-timeout poll.
BEAT_ALLOTMENT_MS = RUN_HEARTBEAT_MS + BUSY_TIMEOUT_MS  # 20_000
# §A.3 window sizing rule: window >= 4 x (beat interval + busy_timeout) —
# four consecutive beat attempts must ALL fail before a worker is evicted.
LIVENESS_WINDOW_MS = 4 * BEAT_ALLOTMENT_MS  # 80_000

# A3: no single write-lock hold may consume more than half of any waiter's
# busy budget.  Expected actual is <5 ms, so this carries ~500x margin.
MAX_HOLD_MS = BUSY_TIMEOUT_MS / 2  # 2_500
# A4: typical-case hold ceiling; thousands of samples make p95 robust to
# isolated CI stalls.
P95_HOLD_MS = 100.0
# A5: a §A.3-shaped beat lands within half of one beat allotment ⇒ >=2x
# per-attempt margin, >=8x inside the 80 s window.  Expected actual <50 ms.
MAX_BEAT_RT_MS = BEAT_ALLOTMENT_MS / 2  # 10_000
# A6: the concrete §A.3 headroom figure reported for the slice-4 rerun.
MIN_LIVENESS_HEADROOM = 8.0
# A7: hottest-verb latency stays interactive under contention.
P99_WRITE_RT_MS = 1_000.0
# A8: dashboard reads stay live while two writers hammer (F10 scenario).
MAX_READ_MS = float(BUSY_TIMEOUT_MS)  # 5_000
P95_READ_MS = 500.0
# A9: progress floors (~100x under expected throughput) proving both hammers
# genuinely ran concurrently and both verbs did real cross-owner work.
MIN_CLAIMS_PER_HAMMER = 10
MIN_SWEEPS_PER_HAMMER = 1
MIN_RECOVERED_PER_HAMMER = 1
MIN_READ_BATCHES = 20
# A12: instrumentation self-check — the IMMEDIATE listener must be observable.
MIN_STAMPED_FRACTION = 0.99

READY_DEADLINE_SECONDS = 30.0
JOIN_TIMEOUT_SECONDS = 60.0


# ---------------------------------------------------------------------------
# Seeding (local copies of the canonical unit-fixture shapes from
# tests/unit/engine/test_lease_recovery_sweep.py — deliberately NOT
# imported from a unit-test module).
# ---------------------------------------------------------------------------


def _seed_database(db_path: Path, *, now: datetime) -> None:
    db = LandscapeDB(f"sqlite:///{db_path}")
    try:
        engine = db.engine
        with engine.begin() as conn:
            conn.execute(
                insert(runs_table).values(
                    run_id=RUN_ID,
                    started_at=now,
                    config_hash="config",
                    settings_json="{}",
                    canonical_version="v1",
                    status="running",
                    openrouter_catalog_sha256="0" * 64,
                    openrouter_catalog_source="bundled",
                )
            )
            for node_id, node_type, plugin in (
                ("source-a", NodeType.SOURCE, "csv"),
                ("normalize", NodeType.TRANSFORM, "identity"),
            ):
                conn.execute(
                    insert(nodes_table).values(
                        run_id=RUN_ID,
                        node_id=node_id,
                        plugin_name=plugin,
                        node_type=node_type.value,
                        plugin_version="1.0",
                        determinism="deterministic",
                        config_hash="config",
                        config_json="{}",
                        registered_at=now,
                    )
                )
        payload = TokenSchedulerRepository.serialize_row_payload(
            PipelineRow({"id": 1}, SchemaContract(mode="OBSERVED", fields=(), locked=True))
        )
        repo = TokenSchedulerRepository(engine)
        for ingest_sequence in range(WORK_ITEMS):
            row_id = f"row-{ingest_sequence}"
            token_id = f"token-{ingest_sequence}"
            with engine.begin() as conn:
                conn.execute(
                    insert(rows_table).values(
                        row_id=row_id,
                        run_id=RUN_ID,
                        source_node_id="source-a",
                        row_index=ingest_sequence,
                        source_row_index=ingest_sequence,
                        ingest_sequence=ingest_sequence,
                        source_data_hash=f"hash-{row_id}",
                        created_at=now,
                    )
                )
                conn.execute(
                    insert(tokens_table).values(
                        token_id=token_id,
                        row_id=row_id,
                        run_id=RUN_ID,
                        created_at=now,
                    )
                )
            repo.enqueue_ready(
                run_id=RUN_ID,
                token_id=token_id,
                row_id=row_id,
                node_id="normalize",
                step_index=1,
                ingest_sequence=ingest_sequence,
                available_at=now,
                row_payload_json=payload,
            )
    finally:
        # Parent must hold NO connection while the hammers run.
        db.close()


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def _spawn_children(
    tmp_path: Path,
    db_url: str,
    *,
    duration_seconds: float,
    hammer_owners: tuple[str, ...],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Run N hammers + 1 reader concurrently; return (hammer_artifacts, reader_artifact)."""
    src_dir = Path(elspeth.__file__).resolve().parents[1]
    env = dict(os.environ)
    env["PYTHONPATH"] = str(src_dir) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")

    go_file = tmp_path / "go"
    specs: list[tuple[str, str]] = [(worker.ROLE_HAMMER, owner) for owner in hammer_owners]
    specs.append((worker.ROLE_READER, "dashboard-reader"))

    procs: list[subprocess.Popen[bytes]] = []
    ready_files: list[Path] = []
    metric_files: list[Path] = []
    log_files: list[Path] = []
    try:
        for role, owner in specs:
            ready = tmp_path / f"ready-{owner}"
            metrics = tmp_path / f"metrics-{owner}.json"
            log = tmp_path / f"log-{owner}.txt"
            ready_files.append(ready)
            metric_files.append(metrics)
            log_files.append(log)
            with open(log, "wb") as log_handle:
                procs.append(
                    subprocess.Popen(
                        [
                            sys.executable,
                            str(WORKER_PATH),
                            "--db-url",
                            db_url,
                            "--role",
                            role,
                            "--owner",
                            owner,
                            "--run-id",
                            RUN_ID,
                            "--ready-file",
                            str(ready),
                            "--go-file",
                            str(go_file),
                            "--duration-seconds",
                            str(duration_seconds),
                            "--metrics-out",
                            str(metrics),
                        ],
                        env=env,
                        stdout=log_handle,
                        stderr=subprocess.STDOUT,
                    )
                )

        # Start barrier: all children up and connected before anyone runs.
        deadline = time.monotonic() + READY_DEADLINE_SECONDS
        while not all(ready.exists() for ready in ready_files):
            for proc, log in zip(procs, log_files, strict=True):
                if proc.poll() is not None:
                    raise AssertionError(f"child exited rc={proc.returncode} before the start barrier:\n{log.read_text()}")
            if time.monotonic() > deadline:
                raise AssertionError(f"children not ready within {READY_DEADLINE_SECONDS}s")
            time.sleep(0.01)
        go_file.touch()

        for proc, log in zip(procs, log_files, strict=True):
            rc = proc.wait(timeout=JOIN_TIMEOUT_SECONDS)
            # A1 (exit codes): exit 1 is harness-internal failure, never a
            # recorded contention error — those land in the artifact.
            assert rc == 0, f"child exited rc={rc}:\n{log.read_text()}"
    finally:
        for proc in procs:
            if proc.poll() is None:
                proc.kill()
                proc.wait(timeout=10)

    artifacts = [json.loads(path.read_text()) for path in metric_files]
    return artifacts[:-1], artifacts[-1]


def _percentile(values: list[float], fraction: float) -> float:
    """Nearest-rank percentile (no interpolation; conservative for upper tails)."""
    assert values, "percentile of empty series"
    ordered = sorted(values)
    rank = math.ceil(fraction * len(ordered))
    return ordered[max(rank, 1) - 1]


def _run_contention_scenario(
    tmp_path: Path,
    *,
    duration_seconds: float,
    hammer_owners: tuple[str, ...],
) -> None:
    db_path = tmp_path / "audit.db"
    _seed_database(db_path, now=datetime.now(UTC))
    db_url = f"sqlite:///{db_path}"

    hammers, reader = _spawn_children(tmp_path, db_url, duration_seconds=duration_seconds, hammer_owners=hammer_owners)

    # ---- Aggregate ----------------------------------------------------------
    write_txns = [rec for hammer in hammers for rec in hammer["write_txns"]]
    verb_txns = [rec for rec in write_txns if rec["verb"] != worker.VERB_BEAT]
    holds = [rec["hold_ms"] for rec in write_txns if "hold_ms" in rec]
    lock_waits = [rec["lock_wait_ms"] for rec in write_txns if "lock_wait_ms" in rec]
    verb_rts = [rec["round_trip_ms"] for rec in verb_txns]
    all_write_rts = [rec["round_trip_ms"] for rec in write_txns]
    beat_rts = [rt for hammer in hammers for rt in hammer["beats"]]
    read_rts = list(reader["reads"])

    assert holds and verb_rts and beat_rts and read_rts, (
        f"degenerate run: missing samples (holds={len(holds)} verbs={len(verb_rts)} beats={len(beat_rts)} reads={len(read_rts)})"
    )

    max_hold = max(holds)
    p95_hold = _percentile(holds, 0.95)
    max_lock_wait = max(lock_waits)
    max_beat_rt = max(beat_rts)
    max_write_rt = max(all_write_rts)
    p99_write_rt = _percentile(verb_rts, 0.99)
    max_read = max(read_rts)
    p95_read = _percentile(read_rts, 0.95)
    headroom = LIVENESS_WINDOW_MS / max_write_rt

    # ---- The measurement line: printed BEFORE any assertion (this is the
    # deliverable that ADR-030 risk 2 and the slice-4 rerun depend on). -------
    claims_by_owner = " ".join(f"claims_{h['owner']}={h['claims']}" for h in hammers)
    recovered_by_owner = " ".join(f"recovered_{h['owner']}={h['recovered_total']}" for h in hammers)
    print(
        f"SLICE1-CONTENTION hammers={len(hammers)} duration_s={duration_seconds} "
        f"max_hold_ms={max_hold:.2f} p95_hold_ms={p95_hold:.2f} "
        f"max_lock_wait_ms={max_lock_wait:.2f} max_beat_rt_ms={max_beat_rt:.2f} "
        f"max_write_rt_ms={max_write_rt:.2f} p99_write_rt_ms={p99_write_rt:.2f} "
        f"liveness_headroom={headroom:.0f}x max_read_ms={max_read:.2f} p95_read_ms={p95_read:.2f} "
        f"{claims_by_owner} {recovered_by_owner} read_batches={reader['read_batches']}"
    )

    # ---- A1: zero contention errors (SQLITE_BUSY_SNAPSHOT closure gate) -----
    for artifact in [*hammers, reader]:
        assert artifact["errors"] == [], f"{artifact['owner']} recorded errors: {artifact['errors']}"

    # ---- A2: cross-process PRAGMA inheritance (G28) --------------------------
    for artifact in [*hammers, reader]:
        assert artifact["pragmas"]["busy_timeout"] == str(BUSY_TIMEOUT_MS), artifact["pragmas"]
        assert artifact["pragmas"]["journal_mode"] == "wal", artifact["pragmas"]

    # ---- A3/A4: max write-lock hold time vs busy_timeout (risk 2) -----------
    assert max_hold <= MAX_HOLD_MS, f"max hold {max_hold:.1f}ms > {MAX_HOLD_MS}ms (busy_timeout/2)"
    assert p95_hold <= P95_HOLD_MS, f"p95 hold {p95_hold:.1f}ms > {P95_HOLD_MS}ms"

    # ---- A5/A6: heartbeat-latency headroom inside the §A.3 window -----------
    assert max_beat_rt <= MAX_BEAT_RT_MS, f"max beat {max_beat_rt:.1f}ms > {MAX_BEAT_RT_MS}ms (allotment/2)"
    assert headroom >= MIN_LIVENESS_HEADROOM, (
        f"liveness headroom {headroom:.1f}x < {MIN_LIVENESS_HEADROOM}x "
        f"(max write round-trip {max_write_rt:.1f}ms vs {LIVENESS_WINDOW_MS}ms window)"
    )

    # ---- A7: claim/recover stay interactive under contention ----------------
    assert p99_write_rt <= P99_WRITE_RT_MS, f"p99 verb round-trip {p99_write_rt:.1f}ms > {P99_WRITE_RT_MS}ms"

    # ---- A8: dashboard reads stay live (F10) ---------------------------------
    assert max_read <= MAX_READ_MS, f"max read batch {max_read:.1f}ms > {MAX_READ_MS}ms"
    assert p95_read <= P95_READ_MS, f"p95 read batch {p95_read:.1f}ms > {P95_READ_MS}ms"

    # ---- A9: genuine concurrent progress on both verb paths -----------------
    for hammer in hammers:
        owner = hammer["owner"]
        assert hammer["claims"] >= MIN_CLAIMS_PER_HAMMER, f"{owner}: claims={hammer['claims']}"
        assert hammer["sweeps"] >= MIN_SWEEPS_PER_HAMMER, f"{owner}: sweeps={hammer['sweeps']}"
        assert hammer["recovered_total"] >= MIN_RECOVERED_PER_HAMMER, (
            f"{owner}: recovered_total={hammer['recovered_total']} — peer-lease reaping never happened"
        )
    assert reader["read_batches"] >= MIN_READ_BATCHES, f"read_batches={reader['read_batches']}"

    # ---- A12: instrumentation self-check (must precede trusting A3/A4) ------
    stamped = [rec for rec in write_txns if "hold_ms" in rec]
    fraction = len(stamped) / len(write_txns)
    assert fraction >= MIN_STAMPED_FRACTION, (
        f"IMMEDIATE listener not observable — only {fraction:.1%} of write txns stamped; adapt instrumentation"
    )
    non_immediate = {rec["begin_stmt"] for rec in stamped if rec["begin_stmt"].upper() != "BEGIN IMMEDIATE"}
    assert not non_immediate, f"write verbs began without IMMEDIATE: {non_immediate}"

    # ---- A10: exactly-once claim under cross-process racing -----------------
    union: list[str] = [wid for hammer in hammers for wid in hammer["claimed_work_item_ids"]]
    assert len(union) == len(set(union)), "a work_item_id was claimed twice across processes"

    verify_db = LandscapeDB.from_url(db_url, read_only=True, create_tables=False)
    try:
        with verify_db.read_only_connection() as conn:
            duplicate_claims = conn.exec_driver_sql(
                "SELECT work_item_id, COUNT(*) AS n FROM scheduler_events "
                "WHERE run_id = ? AND event_type = 'claim_ready' "
                "GROUP BY work_item_id HAVING COUNT(*) > 1",
                (RUN_ID,),
            ).all()
            assert duplicate_claims == [], f"duplicate claim_ready events: {duplicate_claims}"

            # ---- A11: final-state consistency (no wedged/corrupt rows) ------
            rows = conn.execute(
                select(
                    token_work_items_table.c.status,
                    token_work_items_table.c.lease_owner,
                    token_work_items_table.c.attempt,
                ).where(token_work_items_table.c.run_id == RUN_ID)
            ).all()
            assert len(rows) == WORK_ITEMS
            for status, lease_owner, attempt in rows:
                assert status in (TokenWorkStatus.READY.value, TokenWorkStatus.LEASED.value), f"wedged status {status!r}"
                if status == TokenWorkStatus.LEASED.value:
                    assert lease_owner in hammer_owners, f"foreign lease_owner {lease_owner!r}"
                assert attempt >= 1

            # Event-ledger sanity: every claim in the artifacts is in the ledger.
            (claim_events,) = conn.exec_driver_sql(
                "SELECT COUNT(*) FROM scheduler_events WHERE run_id = ? AND event_type = 'claim_ready'",
                (RUN_ID,),
            ).one()
            assert claim_events == len(union), f"ledger has {claim_events} claim events, artifacts {len(union)}"
    finally:
        verify_db.close()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.timeout(120)
def test_two_process_claim_hammer_with_dashboard_reads(tmp_path: Path) -> None:
    """Default-suite gate: 2 hammers + 1 dashboard reader, 3 s, ~10-15 s wall."""
    _run_contention_scenario(
        tmp_path,
        duration_seconds=3.0,
        hammer_owners=("contender-a", "contender-b"),
    )


@pytest.mark.slow
@pytest.mark.timeout(300)
def test_three_process_claim_hammer_soak(tmp_path: Path) -> None:
    """Pre-release soak: 3 hammers + 1 dashboard reader, 30 s (deselected by default)."""
    _run_contention_scenario(
        tmp_path,
        duration_seconds=30.0,
        hammer_owners=("contender-a", "contender-b", "contender-c"),
    )
