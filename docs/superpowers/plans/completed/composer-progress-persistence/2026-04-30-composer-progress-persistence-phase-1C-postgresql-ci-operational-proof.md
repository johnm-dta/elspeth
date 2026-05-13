# Composer Progress Persistence — Phase 1C: PostgreSQL, CI, and Operational Proof

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this schedule task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

> **Traceability:** Task numbers are preserved from the original Phase 1 plan so review findings can cite the same task IDs across Schedule 1A/1B/1C. Do not renumber tasks inside a schedule.

**Goal:** Close Phase 1 by proving the Schedule 1A/1B contracts across PostgreSQL, CI, and final operational handoff.

**Risk controlled:** dialect drift, Docker/testcontainer dependency drift, concurrency hangs, CI-success gating gaps, latency regressions, and stale spec/OQ handoff text.

**Architecture:** This schedule should not invent new schema or primitive behavior except where PostgreSQL proof exposes a portability defect. It owns testcontainer infrastructure, final CI wiring, latency sanity, final spec amendments, and Phase 1 checkpointing.

**Review focus:** PostgreSQL DDL portability, fail-closed dependency checks, thread/future termination assertions, CI branch-protection aggregation, latency test bounds, and final handoff accuracy.

---

## Schedule 1C Scope

**Included tasks:** Task 16, Task 17, Task 19, Task 20, Phase 1 Done When, and the final delivery/stall criteria.

**Explicit exclusions:** New feature behavior beyond fixes required by PostgreSQL/CI proof, Schedule 1A schema/current-writer work, Schedule 1B primitive/audit work, and all later composer/frontend phases.

**Must land after Schedule 1B.** This is the Phase 1 closure schedule; Phase 2 does not start until Schedule 1C has merged.

---

## Schedule 1C Preflight: Environment-Proof Gate

- [ ] **Step 1: Confirm Schedule 1A and 1B are merged**

Verify the branch contains the schema/current-writer safety changes and the compose-turn primitive/audit semantics changes. Do not broaden this PR with new behavior unless PostgreSQL proof exposes a portability defect.

- [ ] **Step 2: Prove schema initialization on PostgreSQL before concurrency claims**

`initialize_session_schema(engine)` must succeed against the PostgreSQL schema used by the testcontainer lane. If current metadata is not portable, fix the metadata or explicitly constrain the test scope before claiming CL-PP-11 coverage.

- [ ] **Step 3: Make dependency checks fail closed**

The `testcontainers.postgres` import check must fail the job when missing. Do not use `|| echo missing` or another success-preserving fallback.

---

## Task 16: CL-PP-11 concurrent multi-session test (testcontainer PostgreSQL)

**Files:**
- Create: `tests/integration/web/test_compose_loop_concurrent_sessions.py`
- Modify: `pyproject.toml` — register the `testcontainer` pytest marker AND add `testcontainers[postgres]` to the dev optional-dependencies group.
- Modify: `uv.lock` — required because CI uses frozen dependency sync.
- Modify: `.github/workflows/ci.yaml` — add a Docker-enabled testcontainer lane/step, update every explicit non-Docker pytest marker expression to exclude `testcontainer`, and include the Docker lane in the aggregate `ci-success` result check.

**Why a dedicated marker.** The default local test invocation
(`pytest tests/`) is configured with
`addopts = ["-m", "not slow and not stress and not performance"]`
in `pyproject.toml` and does NOT auto-skip Docker-required tests.
The GitHub Actions workflow also contains explicit `pytest -m "not slow
and not stress and not performance"` commands, so changing only
`pyproject.toml` is insufficient. Without registering a `testcontainer`
marker AND adding it to the default deselect list and every non-Docker
CI marker expression, every developer or CI job without Docker gets a
noisy fail. With the marker registered (and the default/CI non-Docker
selectors updated to deselect it) the test runs in CI's Docker-enabled
lane and is opt-in locally via `pytest -m testcontainer`. Closes
synthesised review findings M8 / Q-F-07.

- [ ] **Step 1: Register the `testcontainer` marker and add the dependency**

In `pyproject.toml`, extend the existing markers list under
`[tool.pytest.ini_options]` (around line 350) to include:

```toml
markers = [
    "slow: marks tests as slow (deselect with '-m \"not slow\"')",
    "integration: marks tests requiring external services",
    "asyncio: marks tests as async (handled by pytest-asyncio)",
    "performance: marks tests as performance benchmarks",
    "stress: marks tests as stress tests requiring ChaosLLM HTTP server (deselect with '-m \"not stress\"')",
    "composer_llm_eval: characterization/replay tests for the 2026-04-28 composer LLM evaluation scenarios",
    "chaosllm: Configure ChaosLLM server for the test (preset=None, **kwargs)",
    "testcontainer: marks tests requiring Docker via testcontainers-python (deselect with '-m \"not testcontainer\"')",
]
```

In the same `[tool.pytest.ini_options]` block, extend the default
`addopts` deselect list:

```toml
addopts = [
    "-m", "not slow and not stress and not performance and not testcontainer",
]
```

This keeps the default `pytest` invocation Docker-free; CI's
Docker-enabled lane runs `pytest -m testcontainer` explicitly.

In the `[project.optional-dependencies]` table (the `dev` group),
add the testcontainers dependency:

```toml
dev = [
    # ...existing deps...
    "testcontainers[postgres]>=4.0,<5",
]
```

Run `uv sync --extra dev` after editing to install and update `uv.lock`.

- [ ] **Step 2: Confirm the install**

```bash
.venv/bin/python -c "import testcontainers.postgres" 2>&1 || echo "missing"
```
Expected: no output (success).

- [ ] **Step 2b: Add the CI lane**

In `.github/workflows/ci.yaml`, add a Docker-enabled job or explicit
step that installs the dev dependencies from the updated `uv.lock` and
runs:

```bash
.venv/bin/python -m pytest -m testcontainer tests/integration/web/ -v
```

Do not rely on the default unit/integration jobs: `addopts` deselects
`testcontainer` by design. The lane must fail if PostgreSQL cannot be
started through testcontainers. If this is implemented as a separate
GitHub Actions job, add that job name to the aggregate `ci-success`
job's `needs:` list AND update the `ci-success` shell/script logic that
checks `needs.<job>.result` so a failed or skipped testcontainer job
fails the aggregate gate; otherwise branch protection can remain green
while CL-PP-11 is red. If it is implemented as a step inside an existing
job that already feeds `ci-success`, update that job's explicit marker
expressions to:

```bash
-m "not slow and not stress and not performance and not testcontainer"
```

Do this for both coverage and non-coverage test commands. Document the
chosen CI shape in the PR body.

- [ ] **Step 3: Write the test**

Create `tests/integration/web/test_compose_loop_concurrent_sessions.py`:

```python
"""CL-PP-11 (spec §8.2): concurrent writes against two sessions that share
a PostgreSQL connection pool. Verifies (a) per-session sequence_no values
are independently monotonic, (b) advisory-lock collisions serialise but
do not deadlock, (c) the two-argument advisory-lock form
(``pg_advisory_xact_lock(ELSPETH_SESSIONS_LOCK_CLASSID,
hashtext(session_id))``) partitions ELSPETH's
session-write namespace away from any other application or
ELSPETH subsystem on the same Postgres cluster (B3 from the Phase 1
plan-review synthesis). ``hashtext`` returns the required int4 key
directly; do not replace it with ``hashtextextended(... )::int``
because PostgreSQL integer casts are range-checked. Within the
``ELSPETH_SESSIONS_LOCK_CLASSID`` namespace birthday collisions become
probable around ~65k *concurrent* sessions — collisions cause benign
extra serialisation only, never duplicate rows (the unique index
``ix_chat_messages_session_sequence`` is the correctness guarantee)."""
from __future__ import annotations

import threading
import uuid

import pytest
from sqlalchemy import text
from testcontainers.postgres import PostgresContainer

from elspeth.web.sessions.service import SessionServiceImpl
from elspeth.web.sessions._persist_payload import _RedactedToolRow, _StatePayload
from elspeth.web.sessions.protocol import CompositionStateData
from elspeth.web.sessions.engine import create_session_engine
from elspeth.web.sessions.schema import initialize_session_schema
from elspeth.web.sessions.telemetry import build_sessions_telemetry
import structlog

# Integration-suite shared session-insert helper.
from .conftest import _make_session


@pytest.fixture(scope="module")
def pg_engine():
    """Spin up a Postgres testcontainer for the duration of this test
    module. ``create_session_engine`` is a no-op for non-SQLite
    dialects (FK enforcement is the database's responsibility on
    Postgres) — using it here keeps the engine-construction pattern
    uniform with the rest of the suite. ``initialize_session_schema``
    is the same path production uses; we explicitly avoid
    ``metadata.create_all`` here because the production schema
    bootstrap is the only path tested in CI."""
    with PostgresContainer("postgres:16-alpine") as pg:
        engine = create_session_engine(pg.get_connection_url())
        initialize_session_schema(engine)
        yield engine


@pytest.fixture
def service(pg_engine, tmp_path):
    return SessionServiceImpl(
        pg_engine, data_dir=tmp_path,
        telemetry=build_sessions_telemetry(),
        log=structlog.get_logger(),
    )


import traceback


def _worker(
    service,
    session_id: str,
    writer_id: str,
    n: int,
    errors: list,
    tracebacks: list,
) -> None:
    """Concurrent-write worker. Captures the traceback alongside any
    exception so test failures are diagnosable rather than just
    showing the exception class. Closes synthesised review nit P-N-2."""
    try:
        for i in range(n):
            tool_call_id = f"{session_id}_{writer_id}_tc_{i}"
            service.persist_compose_turn(
                session_id=session_id,
                assistant_content=f"turn {i}",
                redacted_assistant_tool_calls=(
                    {"id": tool_call_id, "function": {"name": "f"}},
                ),
                redacted_tool_rows=(
                    _RedactedToolRow(
                        tool_call_id,
                        "{}",
                        None,
                    ),
                ),
                parent_composition_state_id=None,
                expected_current_state_id=None,
                writer_principal="compose_loop",
                plugin_crash_pending=False,
            )
    except Exception as exc:  # noqa: BLE001
        errors.append(exc)
        tracebacks.append(traceback.format_exc())


def test_concurrent_DIFFERENT_sessions_do_not_deadlock(service):
    """Concurrent writes to two DIFFERENT sessions must not interfere.
    Different ``session_id`` values map to different
    ``hashtext(...)`` slots within the
    ``ELSPETH_SESSIONS_LOCK_CLASSID`` namespace, so the advisory locks
    do NOT contend (modulo ~65k-concurrent-session birthday-collision
    probability — see module docstring) — this test verifies the
    *non-contending* path: two sessions can write concurrently with
    both reaching their expected row counts.

    For the advisory-lock CONTENTION path see
    ``test_concurrent_SAME_session_serialises_via_advisory_lock``
    below — that is the test the spec's CL-PP-11 description was
    actually targeting."""
    with service._engine.begin() as conn:
        _make_session(conn, session_id="s_a")
        _make_session(conn, session_id="s_b")

    errors: list[Exception] = []
    tracebacks: list[str] = []

    threads = [
        threading.Thread(target=_worker, args=(service, "s_a", "writer_a", 5, errors, tracebacks)),
        threading.Thread(target=_worker, args=(service, "s_b", "writer_b", 5, errors, tracebacks)),
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)

    assert not errors, f"errors: {errors}\ntracebacks:\n" + "\n---\n".join(tracebacks)

    # Both sessions reached 10 messages each (1 assistant + 1 tool per turn × 5).
    with service._engine.begin() as conn:
        for sid in ("s_a", "s_b"):
            count = conn.execute(text(
                "SELECT COUNT(*) FROM chat_messages WHERE session_id = :sid"
            ), {"sid": sid}).scalar()
            assert count == 10, f"session {sid} count: {count}"
            seqs = [
                row.sequence_no
                for row in conn.execute(text(
                    "SELECT sequence_no FROM chat_messages "
                    "WHERE session_id = :sid ORDER BY sequence_no"
                ), {"sid": sid})
            ]
            # Strictly monotonic; gaps allowed but not duplicates.
            assert all(b > a for a, b in zip(seqs, seqs[1:]))


def test_concurrent_SAME_session_serialises_via_advisory_lock(service):
    """Concurrent writes to the SAME session must serialise via the
    session-scoped PostgreSQL advisory lock
    (``pg_advisory_xact_lock(ELSPETH_SESSIONS_LOCK_CLASSID,
    hashtext(session_id))`` — the two-argument form
    introduced by B3 to partition ELSPETH's namespace from other
    cluster users).

    Without serialisation, two writers would race on
    ``_reserve_sequence_range``'s ``SELECT MAX(sequence_no) ...``
    and produce duplicate ``sequence_no`` values, hitting the
    ``ix_chat_messages_session_sequence`` unique index and raising
    ``IntegrityError``. With the advisory lock, the second writer
    blocks until the first commits, sees the updated max, and
    allocates the next contiguous range — no duplicates, no
    deadlock.

    Closes synthesised review finding F-02 / Q-F-02 (CL-PP-11
    coverage gap)."""
    with service._engine.begin() as conn:
        _make_session(conn, session_id="s_shared")

    errors: list[Exception] = []
    tracebacks: list[str] = []

    threads = [
        threading.Thread(
            target=_worker,
            args=(service, "s_shared", "writer_A", 10, errors, tracebacks),
            name="writer_A",
        ),
        threading.Thread(
            target=_worker,
            args=(service, "s_shared", "writer_B", 10, errors, tracebacks),
            name="writer_B",
        ),
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)

    assert not errors, (
        f"advisory-lock serialisation failed; errors:\n"
        f"{errors}\ntracebacks:\n" + "\n---\n".join(tracebacks)
    )

    # Both writers must have completed; total rows = 2 × 10 turns ×
    # 2 rows/turn (one assistant + one tool) = 40.
    with service._engine.begin() as conn:
        count = conn.execute(text(
            "SELECT COUNT(*) FROM chat_messages WHERE session_id = 's_shared'"
        )).scalar()
        assert count == 40, f"expected 40 rows, got {count}"

        # sequence_no values must be unique across both writers.
        # Duplicates would manifest as IntegrityError above; the
        # explicit check here documents the expected post-condition
        # so a future regression that disables the unique index
        # (instead of the lock) still fails the test.
        seqs = [
            row.sequence_no
            for row in conn.execute(text(
                "SELECT sequence_no FROM chat_messages "
                "WHERE session_id = 's_shared' ORDER BY sequence_no"
            ))
        ]
        assert len(seqs) == len(set(seqs)), (
            f"duplicate sequence_no values: "
            f"{[s for s in seqs if seqs.count(s) > 1]}"
        )
        # Strict monotonicity: 1..40 with no gaps under successful
        # serial allocation. (Gaps are allowed in general — see spec
        # §4.1.1 — but happen only on rollback paths, none of which
        # this test exercises.)
        assert seqs == list(range(min(seqs), min(seqs) + 40))


def test_advisory_lock_actually_acquired_on_postgres(service):
    """Positive assertion against ``pg_locks``: when
    ``_acquire_session_advisory_lock`` is called inside an open
    transaction on PostgreSQL, a row appears in ``pg_locks`` with
    ``locktype='advisory'``, ``granted=true``,
    ``classid=ELSPETH_SESSIONS_LOCK_CLASSID``, and
    ``objid=hashtext('lock_probe')``. Without this,
    a typo in the dialect guard (e.g. checking ``"postgres"``
    instead of ``"postgresql"``) would silently no-op the lock
    on PG and the contention test above would still pass for
    the wrong reason — the SELECT MAX races would simply not
    overlap in test time.

    Filtering on the classid is load-bearing for B3 (Phase 1
    plan-review synthesis): if a future second classid lands in the
    same cluster, an unfiltered ``classid=*`` assertion would pass
    against the wrong subsystem's lock and silently green-light a
    regression in the sessions-DB lock path. The classid is the
    namespace-partitioning guarantee the two-argument advisory-lock
    form was introduced to provide.

    Closes synthesised review finding F-03 / Q-F-03."""
    from elspeth.contracts.advisory_locks import ELSPETH_SESSIONS_LOCK_CLASSID

    with service._engine.begin() as conn:
        _make_session(conn, session_id="lock_probe")
        # Acquire the advisory lock; do NOT release manually —
        # ``pg_advisory_xact_lock`` releases at transaction end.
        service._acquire_session_advisory_lock(conn, "lock_probe")
        # Inspect pg_locks while still inside the transaction.
        # Filter on classid AND objid so this test fails loudly if a
        # future change accidentally drops the classid (collapsing
        # back to single-argument form) or hashes the wrong session_id.
        row = conn.execute(text(
            "SELECT classid, objid FROM pg_locks "
            "WHERE locktype = 'advisory' AND granted = true "
            "AND classid = :classid"
        ), {"classid": ELSPETH_SESSIONS_LOCK_CLASSID}).first()
        assert row is not None, (
            "advisory lock not visible in pg_locks under "
            f"classid={ELSPETH_SESSIONS_LOCK_CLASSID} — dialect guard "
            "in _acquire_session_advisory_lock may have skipped the "
            "Postgres branch, or the two-argument form was reverted "
            "to single-argument (regressing B3)."
        )
        # Verify objid matches hashtext(session_id).
        # Computing the expected value via the database guarantees we
        # exercise the same hash function the helper used.
        expected_objid = conn.execute(text(
            "SELECT hashtext(:sid)"
        ), {"sid": "lock_probe"}).scalar()
        assert row.objid == expected_objid, (
            f"advisory-lock objid mismatch: got {row.objid}, "
            f"expected {expected_objid} — _acquire_session_advisory_lock "
            "may be hashing a different value than session_id."
        )


def test_cross_allocator_save_state_serialises_with_persist_turn(service):
    """B3 (Phase 1 plan-review synthesis): the dual-allocator race.

    Pre-B3, ``persist_compose_turn`` allocated ``composition_states.version``
    under _session_write_lock, but ``save_composition_state``
    relied on a 3-attempt retry loop with NO lock. A concurrent
    interleave produced a real ``IntegrityError`` on the locked path,
    which the locked path's exception handler classified as a Tier-1
    audit-integrity violation — fabricating a Tier-1 violation from a
    benign race. (Fabricated Tier-1 violations are evidence-tampering
    under the audit-grade doctrine.)

    Post-B3, ``save_composition_state`` (and ``set_active_state``)
    also enter ``_session_write_lock`` before their SELECT MAX. The
    persist side also has the stale-current-state guard introduced in
    Task 11: if a save wins after the persist worker reads the current
    state but before it acquires the write lock, ``persist_compose_turn``
    rejects with ``StaleComposeStateError``. That is the correct
    compose-vs-revert disposition and must NOT be counted as an audit
    integrity failure. Successful writers must still leave sequential,
    contiguous version numbers in ``composition_states`` with NO
    ``IntegrityError`` raised on either path.

    This test would have failed pre-B3 by either crashing one path
    with ``IntegrityError`` or, worse, by silently incrementing the
    ``tool_row_integrity_violation_total`` counter (the false-positive
    Tier-1 alert) on the persist path.
    """
    from elspeth.web.sessions._persist_payload import _RedactedToolRow, _StatePayload
    from elspeth.web.sessions.protocol import CompositionStateData
    from elspeth.web.sessions.service import StaleComposeStateError
    from elspeth.web.sessions.telemetry import observed_value

    session_uuid = uuid.uuid4()
    sid = str(session_uuid)
    with service._engine.begin() as conn:
        _make_session(conn, session_id=sid)

    errors: list[Exception] = []
    stale_rejections: list[StaleComposeStateError] = []
    tracebacks: list[str] = []
    persist_count = 5
    save_count = 5
    starting_counter = observed_value(
        service._telemetry.tool_row_integrity_violation_total
    )

    def _persist_worker():
        # B2 (Phase 1 plan-review synthesis): pre-B2 this loop passed
        # ``assistant_id=str(uuid.uuid4())`` and
        # ``assistant_raw_content=None`` to ``persist_compose_turn`` —
        # neither kwarg is declared on the signature.
        # ``assistant_id`` is helper-generated (the test never read it
        # back); ``assistant_raw_content`` was a typo for the new
        # ``raw_content`` parameter, which defaults to ``None`` and so
        # is omitted entirely (this test exercises the
        # cross-allocator race, not the redaction-attribution path).
        try:
            for i in range(persist_count):
                with service._engine.begin() as conn:
                    expected_current_state_id = conn.execute(text(
                        "SELECT id FROM composition_states "
                        "WHERE session_id = :sid ORDER BY version DESC LIMIT 1"
                    ), {"sid": sid}).scalar_one_or_none()
                tool_call_id = f"tc_persist_{i}"
                service.persist_compose_turn(
                    session_id=sid,
                    assistant_content="ok",
                    redacted_assistant_tool_calls=({"id": tool_call_id, "function": {"name": "f"}},),
                    redacted_tool_rows=(
                        # B1: no ``version=``; helper allocates under lock.
                        _RedactedToolRow(tool_call_id, "{}", _StatePayload(data=CompositionStateData(), derived_from_state_id=None)),
                    ),
                    parent_composition_state_id=None,
                    expected_current_state_id=expected_current_state_id,
                    writer_principal="compose_loop",
                    plugin_crash_pending=False,
                )
        except StaleComposeStateError as e:
            stale_rejections.append(e)
        except Exception as e:
            import traceback
            errors.append(e)
            tracebacks.append(traceback.format_exc())

    def _save_worker():
        import asyncio
        try:
            for _ in range(save_count):
                asyncio.run(
                    service.save_composition_state(
                        session_uuid,  # ``save_composition_state`` takes UUID
                        CompositionStateData(),
                    )
                )
        except Exception as e:
            import traceback
            errors.append(e)
            tracebacks.append(traceback.format_exc())

    threads = [
        threading.Thread(target=_persist_worker, name="persist"),
        threading.Thread(target=_save_worker, name="save"),
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)

    assert not errors, (
        f"cross-allocator race not serialised; errors:\n"
        f"{errors}\ntracebacks:\n" + "\n---\n".join(tracebacks)
    )
    assert observed_value(
        service._telemetry.tool_row_integrity_violation_total
    ) == starting_counter

    # Sanity: every allocated version is unique and contiguous from 1.
    with service._engine.begin() as conn:
        versions = sorted(
            row.version
            for row in conn.execute(text(
                "SELECT version FROM composition_states "
                "WHERE session_id = :sid ORDER BY version"
            ), {"sid": sid})
        )
    # Successful writers only: stale compose attempts roll back before
    # inserting state rows. Whatever succeeded must still be contiguous
    # from 1 with no duplicate/gap evidence of allocator races.
    assert versions == list(range(1, len(versions) + 1)), (
        f"cross-allocator versions not contiguous: {versions}"
    )
    assert len(versions) + len(stale_rejections) >= save_count


def test_concurrent_persist_compose_turn_same_state_stale_rejects_without_integrity_alert(service):
    """B1 plus stale-state guard: two concurrent state-changing compose
    persists based on the same current state must not fabricate a Tier-1
    integrity alert.

    Pre-B1, both writers could try to insert the same caller-supplied
    ``composition_states.version`` and one would hit
    ``uq_composition_state_version``. Post-B1 the helper allocates the
    version under the lock, but Task 11's stale-current-state guard adds
    the stronger user-intent rule: after the first writer creates the new
    current state, the second writer's ``expected_current_state_id=None``
    is stale and should be rejected with ``StaleComposeStateError`` before
    it inserts any audit rows. The integrity counter must not move.

    Closes B1 and the compose-vs-revert stale-state race from the Phase 1
    plan-review synthesis."""
    from elspeth.web.sessions._persist_payload import _RedactedToolRow, _StatePayload
    from elspeth.web.sessions.protocol import CompositionStateData
    from elspeth.web.sessions.service import StaleComposeStateError
    from elspeth.web.sessions.telemetry import observed_value

    sid = "s_b1_concurrent"
    with service._engine.begin() as conn:
        _make_session(conn, session_id=sid)

    starting_counter = observed_value(
        service._telemetry.tool_row_integrity_violation_total
    )

    errors: list[Exception] = []
    stale_rejections: list[StaleComposeStateError] = []
    tracebacks: list[str] = []
    barrier = threading.Barrier(2)

    def _stateful_worker(writer_id: str) -> None:
        try:
            barrier.wait(timeout=10)
            tool_call_id = f"{sid}_{writer_id}_tc_0"
            service.persist_compose_turn(
                session_id=sid,
                assistant_content="ok",
                redacted_assistant_tool_calls=(
                    {"id": tool_call_id, "function": {"name": "f"}},
                ),
                redacted_tool_rows=(
                    _RedactedToolRow(
                        tool_call_id,
                        "{}",
                        _StatePayload(
                            data=CompositionStateData(),
                            derived_from_state_id=None,
                        ),
                    ),
                ),
                parent_composition_state_id=None,
                expected_current_state_id=None,
                writer_principal="compose_loop",
                plugin_crash_pending=False,
            )
        except StaleComposeStateError as exc:
            stale_rejections.append(exc)
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)
            tracebacks.append(traceback.format_exc())

    threads = [
        threading.Thread(
            target=_stateful_worker,
            args=("b1_writer_A",),
            name="b1_writer_A",
        ),
        threading.Thread(
            target=_stateful_worker,
            args=("b1_writer_B",),
            name="b1_writer_B",
        ),
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)

    # B1/stale-guard: NO IntegrityError on either path. The loser of the
    # same-state race is a stale compose, not an audit-integrity event.
    assert not errors, (
        f"B1 regression: dual-allocator race not closed. errors:\n"
        f"{errors}\ntracebacks:\n" + "\n---\n".join(tracebacks)
    )
    assert len(stale_rejections) == 1

    # B1: the integrity-violation counter MUST NOT have moved. Any
    # increment here is a fabricated Tier-1 alert (SLO threshold = 0).
    assert observed_value(
        service._telemetry.tool_row_integrity_violation_total
    ) == starting_counter, (
        "B1 regression: tool_row_integrity_violation_total incremented "
        "during concurrent persist_compose_turn. SLO threshold for this "
        "counter is 0; any increment is a fabricated Tier-1 alert and "
        "evidence-tampering-class harm under ELSPETH's audit doctrine."
    )

    # Exactly one writer commits; the stale writer rolls back before
    # inserting any state or chat rows. The one committed state receives
    # version 1.
    with service._engine.begin() as conn:
        versions = [
            row.version
            for row in conn.execute(text(
                "SELECT version FROM composition_states "
                "WHERE session_id = :sid ORDER BY version"
            ), {"sid": sid})
        ]
    assert versions == [1]
```

- [ ] **Step 4: Apply the `testcontainer` marker to the whole module**

The test file uses free functions, not a test class. Add module-level
`pytestmark` immediately after imports so every function in
`tests/integration/web/test_compose_loop_concurrent_sessions.py` is
selected by `pytest -m testcontainer` and deselected by the default
Docker-free lane:

```python
pytestmark = pytest.mark.testcontainer
```

This both signals intent to readers and (with the
`addopts = ["-m", "not ... and not testcontainer"]` deselect from
Step 1) keeps the default `pytest tests/` invocation Docker-free.
CI's Docker-enabled lane runs `pytest -m testcontainer` to opt in.

- [ ] **Step 5: Run the test in the Docker-enabled lane**

```bash
.venv/bin/python -m pytest tests/integration/web/test_compose_loop_concurrent_sessions.py -v -m testcontainer
```
Expected: PASS. Verify locally only when Docker is running; in CI,
the testcontainer lane is the canonical execution path.

- [ ] **Step 6: Commit**

```bash
git add tests/integration/web/test_compose_loop_concurrent_sessions.py pyproject.toml uv.lock .github/workflows/ci.yaml
git commit -m "test(integration): CL-PP-11 concurrent multi-session writes against testcontainer Postgres (composer-progress-persistence phase 1)"
```

---
## Task 17: Latency sanity bound test

**Files:**
- Create: `tests/integration/web/test_compose_loop_latency_sanity.py`

- [ ] **Step 1: Write the test**

```python
"""Spec §1.4 NFR: per-turn DB write overhead p95 ≤ 250 ms with N ≤ 8 tool
calls, measured in CI on standard infra. Order-of-magnitude bound, not a
tight budget; the tight 25 ms target is verified by a nightly bench job."""
from __future__ import annotations

import statistics
import time

import pytest
from sqlalchemy.pool import StaticPool

from elspeth.web.sessions.service import SessionServiceImpl
from elspeth.web.sessions._persist_payload import _RedactedToolRow, _StatePayload
from elspeth.web.sessions.protocol import CompositionStateData
from elspeth.web.sessions.engine import create_session_engine
from elspeth.web.sessions.schema import initialize_session_schema
from elspeth.web.sessions.telemetry import build_sessions_telemetry
import structlog

# Integration-suite shared session-insert helper.
from .conftest import _make_session


@pytest.fixture
def service(tmp_path):
    eng = create_session_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    initialize_session_schema(eng)
    return SessionServiceImpl(
        eng, data_dir=tmp_path,
        telemetry=build_sessions_telemetry(),
        log=structlog.get_logger(),
    )


_WARMUP_TURNS = 5
_MEASURED_TURNS = 50
_TOTAL_TURNS = _WARMUP_TURNS + _MEASURED_TURNS


def _build_turn(turn: int):
    """Construct a (calls, rows) pair for one turn of the latency test.
    Extracted so the warm-up and measured loops share identical
    construction, removing one variable from the timing comparison."""
    rows = tuple(
        _RedactedToolRow(
            f"turn_{turn}_tc_{i}",
            '{"ok": true}',
            _StatePayload(
                data=CompositionStateData(),
                derived_from_state_id=None,
            ),
        )
        for i in range(8)
    )
    calls = tuple(
        {"id": f"turn_{turn}_tc_{i}", "function": {"name": "f"}}
        for i in range(8)
    )
    return calls, rows


def test_per_turn_p95_under_250ms_with_8_tool_calls(service):
    """Sanity bound (spec §1.4 NFR). Excludes the first ``_WARMUP_TURNS``
    iterations from the measurement: SQLAlchemy connection-pool warmup,
    Python bytecode caching, SQLite page-cache fill, and OTel meter
    materialisation all skew the very first calls. Without this
    exclusion the p95 absorbs cold-start noise that does not represent
    steady-state production latency. Closes synthesised review
    finding F-09 / Q-F-09.

    The bound is a sanity threshold, not a tight target — a tight
    25ms target is verified by a nightly bench job (see overview).

    **Scope note (synthesised review S6 / M3 — systems thinker).**
    This test measures the SYNC-ONLY path:
    ``service.persist_compose_turn(...)`` invoked directly. In
    production (Phase 3), the compose loop calls
    ``await service.persist_compose_turn_async(...)``, which
    additionally pays for ``ThreadPoolExecutor`` allocation + the
    ``run_in_executor`` scheduling round-trip. Phase 3 must add a
    parallel test that exercises the full async dispatch path so
    the production p95 — not just the sync function — is bounded.
    For Phase 1 the sync baseline is the right scope: Phase 1's
    only caller of ``persist_compose_turn`` is this test.
    """
    with service._engine.begin() as conn:
        _make_session(conn, session_id="lat")

    expected_current_state_id = None

    # Warm-up: discard timings.
    for turn in range(_WARMUP_TURNS):
        calls, rows = _build_turn(turn)
        service.persist_compose_turn(
            session_id="lat",
            assistant_content="",
            redacted_assistant_tool_calls=calls,
            redacted_tool_rows=rows,
            parent_composition_state_id=None,
            expected_current_state_id=expected_current_state_id,
            writer_principal="compose_loop",
            plugin_crash_pending=False,
        )
        with service._engine.begin() as conn:
            expected_current_state_id = conn.execute(text(
                "SELECT id FROM composition_states "
                "WHERE session_id='lat' ORDER BY version DESC LIMIT 1"
            )).scalar_one()

    # Measured: 50 turns is enough for stable p95 via 20-quantiles
    # (the 19th quantile == p95).
    durations: list[float] = []
    for turn in range(_WARMUP_TURNS, _TOTAL_TURNS):
        calls, rows = _build_turn(turn)
        start = time.perf_counter()
        service.persist_compose_turn(
            session_id="lat",
            assistant_content="",
            redacted_assistant_tool_calls=calls,
            redacted_tool_rows=rows,
            parent_composition_state_id=None,
            expected_current_state_id=expected_current_state_id,
            writer_principal="compose_loop",
            plugin_crash_pending=False,
        )
        durations.append((time.perf_counter() - start) * 1000)
        with service._engine.begin() as conn:
            expected_current_state_id = conn.execute(text(
                "SELECT id FROM composition_states "
                "WHERE session_id='lat' ORDER BY version DESC LIMIT 1"
            )).scalar_one()

    p95 = statistics.quantiles(durations, n=20)[18]  # 19th of 20 quantiles == p95
    assert p95 < 250, (
        f"p95={p95:.1f}ms exceeds 250ms sanity bound (warmup={_WARMUP_TURNS} "
        f"discarded, measured={_MEASURED_TURNS}); durations={durations}"
    )
```

- [ ] **Step 2: Run the test**

```bash
.venv/bin/python -m pytest tests/integration/web/test_compose_loop_latency_sanity.py -v
```
Expected: PASS.

- [ ] **Step 3: Commit**

```bash
git add tests/integration/web/test_compose_loop_latency_sanity.py
git commit -m "test(integration): per-turn latency sanity bound (composer-progress-persistence phase 1)"
```

---
## Task 19: File OQ-1 follow-up + amend stale design/spec handoff text

- [ ] **Step 1: Use the filigree MCP tool to create the OQ-1 issue**

Create a new issue with the following fields. Use the `mcp__filigree__create_issue` tool, or `filigree create` CLI if MCP is unavailable:

- Title: `chat_messages and audit_access_log retention CLI extension`
- Type: `task`
- Priority: `P3`
- Description: `Spec section: docs/superpowers/specs/2026-04-30-composer-progress-persistence-design.md §10 OQ-1. Extend the elspeth purge --retention-days CLI to operate on web session chat_messages and audit_access_log tables in addition to its existing pipeline-payload scope. Currently web tables grow only through cascade-delete with sessions; an explicit retention path is needed before the production composer corpus exists.`
- Labels: `cluster:composer-progress-persistence`, `from-design-spec`

- [ ] **Step 2: Record the OQ-1 ticket ID**

Record the assigned `elspeth-XXXXXXXX` ID. This will be cited in the Phase 1 PR description as the resolution to OQ-1.

- [ ] **Step 3: Amend the design/spec document in the same PR**

Modify `docs/superpowers/specs/2026-04-30-composer-progress-persistence-design.md`
so downstream phases do not copy stale source-of-truth text:

- §4.1.1: extend the `chat_messages.role` values with internal
  `"audit"` and extend `writer_principal` with `"session_fork"`.
  Explain that `role="audit"` rows are excluded from normal chat
  history and are used only for audit breadcrumbs that cannot be
  parented to an assistant row.
- §4.1.1: replace any one-column `parent_assistant_id` FK description
  with the composite same-session FK:
  `(parent_assistant_id, session_id) -> (chat_messages.id, chat_messages.session_id)`,
  backed by `uq_chat_messages_id_session`.
- §5.2 / §5.2.1: remove `version` from `_StatePayload` construction
  and field lists. State versions are allocated by
  `_insert_composition_state` under `_session_write_lock`.
- §5.2.2: add `raw_content: str | None = None` to
  `persist_compose_turn` and plumb it to the assistant
  `_insert_chat_message` call.
- §5.2.2: add `expected_current_state_id: str | None` to
  `persist_compose_turn` and document the under-lock stale-state guard:
  current latest state must equal the expected state or the helper raises
  `StaleComposeStateError` before sequence/version allocation.
- §5.2.2 / protocol handoff: public async callers use
  `SessionServiceProtocol.persist_compose_turn_async(...)`; the sync
  `SessionServiceImpl.persist_compose_turn(...)` stays concrete-only and
  async-loop guarded.
- §5.2.2 helper snippets: replace local `_enveloped(...)` assumptions
  with the shared `_enveloped_state_column(...)` helper used by
  `_insert_composition_state`, `save_composition_state`, and `fork_session`.
- §5.2.2: update `_AuditOutcome` to the two-field shape
  `(assistant_id, unwind_audit_failed)` and make Tier-1 audit-write
  failures raise `AuditIntegrityError` inside the sync worker rather
  than returning a flag for the caller to re-raise.
- §6.3: define `audit_access_log.session_id` with
  `ForeignKey("sessions.id", ondelete="CASCADE")` (or the equivalent SQL
  `ON DELETE CASCADE`) so `archive_session` can delete sessions that have
  audit-grade view rows.
- §5.7.1: replace the single-argument advisory-lock SQL with
  `_session_write_lock`: PostgreSQL uses
  `pg_advisory_xact_lock(ELSPETH_SESSIONS_LOCK_CLASSID,
  hashtext(session_id))` and SQLite uses a process-wide `(database_url,
  session_id)` RLock for the current single-process staging deployment.
- §8.6: replace bare `sqlalchemy.create_engine("sqlite:///:memory:")`
  + `metadata.create_all()` test guidance with
  `create_session_engine(..., poolclass=StaticPool)` +
  `initialize_session_schema()`.
- §10 / OQ-4: replace row-level DELETE framing with the
  session-DB archive/delete/restart recreation procedure.
- CI handoff text: document the `testcontainer` marker, explicit
  non-Docker CI marker expressions, and `ci-success` `needs.<job>.result`
  check for the Docker-enabled lane.

- [ ] **Step 4: Update overview handoff text if needed**

If this Phase 1 plan and the overview are in the same PR, confirm
`2026-04-30-composer-progress-persistence-overview.md` contains the
same supersession language: no row DELETE, no bare `create_engine`,
Phase 3 owns recovery fidelity, and Phase 3 cannot ship without
Tier-1 alert/dashboard/runbook visibility.

- [ ] **Step 5: Commit**

```bash
git add docs/superpowers/specs/2026-04-30-composer-progress-persistence-design.md \
        docs/superpowers/plans/completed/composer-progress-persistence/2026-04-30-composer-progress-persistence-overview.md
git commit -m "docs(composer): align progress-persistence spec with phase 1 execution plan"
```

- [ ] **Step 6: Record OQ-1 in the PR description**

The OQ-1 issue lives in Filigree, not the repo. Replace the
`elspeth-XXXXXXXX` placeholder in the PR body with the actual issue ID.
No `YYYYYYYY` observation placeholders should remain; spec drift is
handled by the doc amendments above, not by observation-only notes.

---
## Task 20: Final Schedule C / Phase 1 closure CI run

- [ ] **Step 1: Run the full sessions test suite**

```bash
.venv/bin/python -m pytest tests/unit/web/sessions/ tests/unit/web/sessions/test_telemetry.py tests/unit/web/composer/test_audit_failure_primacy.py tests/integration/web/test_inv_audit_ahead_backward.py tests/integration/web/test_compose_loop_latency_sanity.py -v
```
Expected: PASS for every test.

- [ ] **Step 1b: Run the testcontainer-marked tests in a Docker-enabled lane (closes B4)**

This step is required before opening the PR. Without it, the
"all CL-PP-* scenarios pass" done-condition (see "Phase 1 Done When"
below) is not actually verified — Task 16's `addopts = ["-m", "not
... and not testcontainer"]` deselect causes Step 2's `pytest tests/ -x`
to silently skip CL-PP-11 and the B3 cross-allocator test. The
synthesised plan-review (B4) ranked this gap as a blocker because
the headline concurrency contract ships unverified.

```bash
.venv/bin/python -m pytest -m testcontainer tests/integration/web/ -v
```
Expected: PASS. Requires Docker on the local machine OR running this
step in CI's Docker-enabled lane.

**If Docker is unavailable locally**, do NOT skip this step. Mark
the PR description with "[ ] CL-PP-11 + B3 cross-allocator pending
CI testcontainer lane" and confirm the testcontainer lane completes
green before merging. The Phase 1 sign-off explicitly depends on
Docker-lane CI green for these tests.

**Sanity check on test discovery** (catches the
"addopts deselected the file before pytest could collect it" failure
mode that B4 was filed against):

```bash
.venv/bin/python -m pytest -m testcontainer tests/integration/web/ --collect-only -q | grep -E "test_concurrent|test_advisory_lock_actually|test_cross_allocator"
```
Expected: at minimum the five CL-PP-11 / B3 / B1 test functions appear
in the collection output (`test_concurrent_DIFFERENT_sessions_do_not_deadlock`,
`test_concurrent_SAME_session_serialises_via_advisory_lock`,
`test_advisory_lock_actually_acquired_on_postgres`,
`test_cross_allocator_save_state_serialises_with_persist_turn`, and
`test_concurrent_persist_compose_turn_same_state_stale_rejects_without_integrity_alert`
— the B1/stale regression for the dual-``persist_compose_turn`` race
that the version-on-payload contract previously fabricated as a Tier-1
violation). If the grep returns empty, the marker registration or
`addopts` deselect is broken and the testcontainer lane is silently
green for the wrong reason.

- [ ] **Step 2: Run the full project test suite**

```bash
.venv/bin/python -m pytest tests/ -x
```
Expected: PASS. No regression in pre-existing tests. The testcontainer
tests are deselected here by design (Step 1b is the canonical run for
those); Step 2 is the Docker-free regression check.

- [ ] **Step 3: Run mypy**

```bash
uv run mypy src/ tests/
```
Expected: clean, matching the CI mypy scope. Address any new errors
before opening the PR.

- [ ] **Step 4: Run ruff**

```bash
uv run ruff check src/ tests/ scripts/ examples/
uv run ruff format --check src/ tests/ scripts/ examples/
```
Expected: clean.

- [ ] **Step 5: Run the tier-model and freeze-guard CI scripts**

```bash
.venv/bin/python scripts/cicd/enforce_tier_model.py check --root src/elspeth --allowlist config/cicd/enforce_tier_model
.venv/bin/python scripts/cicd/enforce_freeze_guards.py check
```
Expected: both green. The new files (`telemetry.py`, `_persist_payload.py`) live in L3 and should not introduce any upward imports. The new `src/elspeth/contracts/advisory_locks.py` is L0 (contracts layer); the import from L3 (`src/elspeth/web/sessions/service.py`) is downward and tier-model-clean — this is the canonical shape for a contracts-layer constant consumed by an L3 plugin (see CLAUDE.md "Layer Dependency Rules").

- [ ] **Step 6: Open the Schedule C closure PR**

```bash
gh pr create --title "test(composer): progress persistence phase 1C — PostgreSQL and CI proof" --body "$(cat <<'EOF'
## Summary

Schedule C closes Phase 1 of composer-progress-persistence after the
Schedule A schema/current-writer safety PR and Schedule B compose-turn
primitive/audit semantics PR have merged.

- Proves `initialize_session_schema(engine)` succeeds on the PostgreSQL
  schema exercised by the testcontainer lane.
- Adds or fixes the Docker-enabled `pytest -m testcontainer
  tests/integration/web/ -v` CI lane and `ci-success` gating.
- Verifies CL-PP-11, advisory-lock acquisition, cross-allocator
  serialization, stale compose-turn rejection, and latency sanity.
- Confirms dependency checks fail closed when `testcontainers.postgres`
  is unavailable.
- Amends final spec/OQ handoff text and records the Phase 1 checkpoint.

## Prior Schedule PRs

- Schedule A: schema and current writer safety PR
- Schedule B: compose-turn primitive and audit semantics PR

## Spec

`docs/superpowers/specs/2026-04-30-composer-progress-persistence-design.md` revision 4.

## Out of scope (later phases)

- Compose-loop integration (Phase 3)
- Redaction primitives (Phase 2)
- Frontend recovery panel (Phase 4)

## Follow-ups filed

- [OQ-1] elspeth-XXXXXXXX (chat_messages retention CLI extension)
- [OQ-3] to be filed in Phase 3
- [OQ-4] staging runbook documents session-DB archive/delete/restart recreation procedure (supersedes the original "pre-deploy DELETE" framing, which did not match codebase reality — no Alembic exists for the session DB)
- [Spec alignment] `docs/superpowers/specs/2026-04-30-composer-progress-persistence-design.md` amended for Phase 1 corrections: audit role, session_fork principal, no `_StatePayload.version`, `persist_compose_turn(raw_content, expected_current_state_id)`, async dispatcher on `SessionServiceProtocol`, two-field `_AuditOutcome`, `_session_write_lock`, audit_access_log cascade, production-equivalent session DB tests, and no row-level DELETE migration framing.

## Test plan

- [ ] Unit tests against in-memory SQLite (`tests/unit/web/sessions/`, `tests/unit/web/sessions/test_telemetry.py`, `tests/unit/web/composer/test_audit_failure_primacy.py`)
- [ ] Integration test: backward-direction INV-AUDIT-AHEAD (`tests/integration/web/test_inv_audit_ahead_backward.py`)
- [ ] Integration test: CL-PP-11 concurrent multi-session against testcontainer Postgres
- [ ] Integration test: per-turn latency sanity bound (p95 <= 250ms)
- [ ] tier-model CI green
- [ ] freeze-guard CI green
- [ ] mypy clean
- [ ] ruff clean

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

- [ ] **Step 7: After the PR merges, record the Phase 1 checkpoint in Filigree**

Do **not** close `elspeth-90b4542b63` after Phase 1. That feature owns
VAL for the whole user-visible recovery capability, and Phase 1 is only
the data-layer/sync-primitive slice. Use `mcp__filigree__add_comment`
(or `filigree add-comment`) on `elspeth-90b4542b63` with the Phase 1 PR
link, commit SHA, and the explicit note that phases 2-4 remain open. If
a dedicated Phase 1 child issue exists by implementation time, close
that child instead and link it from the feature comment.

(Note: do NOT close the parent epic `elspeth-528bde62bb` until all four phases are merged.)

---
## Phase 1 Done When

Schedules A, B, and C have each landed through their own review gate,
and Task 0 plus Tasks 1-20 above are complete. Specifically:

1. [ ] All new tests pass against in-memory SQLite.
2. [ ] CL-PP-11 + the B3 cross-allocator test (`test_cross_allocator_save_state_serialises_with_persist_turn`) + the B1/stale dual-`persist_compose_turn` test (`test_concurrent_persist_compose_turn_same_state_stale_rejects_without_integrity_alert`) pass against testcontainer PostgreSQL **via Task 20 Step 1b explicitly** — passing tests in the Docker-free `pytest tests/ -x` run does NOT count for this gate (the marker deselect silently skips them; B4 from the plan-review synthesis). The B1 regression closes the helper-contract-level fabricated-Tier-1-violation vector and proves stale same-state compose rejects do not increment the integrity counter.
3. [ ] The schema-level backward-direction post-condition is enforced and tested, including intra-turn ordering (B2 fix, see T15).
4. [ ] Existing `add_message` callers updated to pass `writer_principal` — including `_persist_tool_invocations`, `_persist_llm_calls`, and the `fork_session` batch-insert sweep.
5. [ ] tier-model and freeze-guard CI green.
6. [ ] Latency NFR sanity bound green (p95 <= 250 ms with N <= 8 tool calls).
7. [ ] Staging runbook documents the session-DB archive/delete/restart recreation procedure (OQ-4 superseded — see Task 18 rationale).
8. [ ] OQ-1 retention CLI follow-up Filigree ticket filed.
9. [ ] Design/spec and overview handoff text amended so downstream phases do not copy stale Phase 1 snippets.

Phase 2 begins only after the Schedule C closure PR merges. Phase 2
builds on the schema and primitive delivered here but does not require
the compose loop to be wired (Phase 3) or the frontend (Phase 4).

---
## What Phase 1 actually delivers (and does not)

**No user-visible change.** Phase 1 ships:

- New schema columns (`chat_messages.tool_call_id`, `parent_assistant_id`, `sequence_no`, `writer_principal`, `composition_states.provenance`), an internal `chat_messages.role="audit"` storage shape for unparented composer breadcrumbs, and a new `audit_access_log` table.
- A synchronous persistence primitive (`SessionServiceImpl.persist_compose_turn`) with its async-loop guard, protocol-public async dispatcher, stale-current-state guard, `_session_write_lock` + sequence-allocation helpers, and `_AuditOutcome` disposition.
- An advisory-lock classid registry (`src/elspeth/contracts/advisory_locks.py`) defining `ELSPETH_SESSIONS_LOCK_CLASSID` as on-the-wire ABI under change control — the namespace partition that makes the two-argument `pg_advisory_xact_lock` form safe in shared-cluster deployments (B3 fix). Future ELSPETH advisory locks register their classid here.
- The `_StatePayload` contract change closing B1/B3: no `version` field on the payload; `_insert_composition_state` allocates per-session `composition_states.version` under `_session_write_lock`. The dual-allocator race that previously fabricated Tier-1 audit-integrity violations (when two compose turns observed the same `state.version` in async land before the sync worker acquired the lock) is now structurally impossible. This is a contract-level fix that hardens Phase 3's compose-loop wiring against a defect class the Phase 3 plan would otherwise have inherited.
- A telemetry container (`_SessionsTelemetry`) wired into the service constructor and into `app.py`'s production wiring.
- The atomic `add_message` rewrite with the four preserved behaviours, six route/helper call-site updates, and the `fork_session` direct-insert sweep.
- Design/spec and overview handoff text amended so later phases do not copy stale Phase 1 assumptions.
- Schema-level INV-AUDIT-AHEAD backward-direction enforcement and the latency sanity bound.

The new schema columns and table have NO user-facing behaviour
because `persist_compose_turn` is not yet called by anything (the
compose loop wiring is Phase 3) and the `audit_access_log` writer
is also Phase 3 (the `include_tool_rows=true` route extension).
The redaction primitives that produce the `summarizer_errors` and
`unknown_response_key` telemetry are Phase 2.

**Why stage dormant infrastructure through three schedules.** The
schema, the sync primitive, and the audit-primacy disposition are
load-bearing for Phases 2-4, but the review found that landing them in
one PR hides too much risk. Schedule A proves the schema can coexist
with every current writer before any dormant primitive exists. Schedule
B proves the primitive's transcript, audit, and cancellation semantics
before PostgreSQL/CI infrastructure becomes the dominant concern.
Schedule C proves the same contract across PostgreSQL and CI before
Phase 1 is declared complete. This keeps later phases on a stable
persistence contract without forcing one high-blast-radius merge.

**Stall criterion (event-based).** ELSPETH does not commit to
calendar dates for Phase 2-4 (per project policy: ADR-level SLAs
are governance devices, not release deadlines). The stall
criterion for the Phase 1 dormant infrastructure is therefore
event-based:

1. **Phase 2 stalls AND ELSPETH starts onboarding real users.**
   At that point the schema-recreation runbook (Task 18) becomes
   inadequate (real users have chat history that must survive a
   schema change), and the dormant `audit_access_log` table is
   visible to every operator looking at the database without a
   writer to explain its purpose. Decision: introduce migrations
   (cancelling the no-tech-debt-pre-release exemption) OR revert
   Phase 1 to keep the schema in sync with what's actually
   wired.

2. **Phase 2 ships, Phase 3 stalls indefinitely.** The compose
   loop is the first production consumer of
   `persist_compose_turn_async`. If Phase 3 stalls, the concrete sync
   primitive remains uncalled and the async protocol dispatcher remains
   a dormant typed contract. This is acceptable only while there is no
   production caller: the sync primitive stays concrete-only, so routes
   do not accumulate direct `_run_sync` coupling, and the single public
   surface is the async dispatcher that Phase 3 will use.

3. **Phase 3 ships, Phase 4 stalls indefinitely.** Frontend is
   independent — Phase 3's audit-grade view route (with
   `include_tool_rows`) is consumable via direct API calls
   without the recovery panel UI. No revert needed.

The first criterion is the only one that forces a decision; the
second and third are operationally tolerable. Closes synthesised
review finding SA-11 / M15. The framing follows the project's
"no calendar shipping commitments" doctrine — every criterion
fires on a real-world event (user onboarding, phase merge), not a
date.

---

## Schedule 1C Done When

1. [ ] `initialize_session_schema(engine)` succeeds on PostgreSQL for the schema used by the testcontainer lane.
2. [ ] The testcontainer dependency check fails closed when `testcontainers.postgres` is unavailable.
3. [ ] CL-PP-11, advisory-lock acquisition, cross-allocator serialization, and stale compose-turn rejection pass in the Docker-enabled lane.
4. [ ] Concurrency tests assert all worker threads/futures terminate and fail with useful diagnostics on timeout.
5. [ ] CI success aggregation includes the Docker/testcontainer lane.
6. [ ] Latency sanity tests are bounded and diagnostic.
7. [ ] Final spec/OQ text is amended and Phase 1 checkpointing is recorded.
8. [ ] Phase 2 remains blocked until this schedule has merged.
