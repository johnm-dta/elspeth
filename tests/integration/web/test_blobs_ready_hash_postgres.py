"""ck_blobs_ready_hash dialect-equivalence regression for PostgreSQL.

The SQLite parametrized test
(``tests/unit/web/blobs/test_service.py::TestBlobsReadyHashDBConstraint::
test_update_ready_hash_to_malformed_rejected``) exercises eight
malformed-hash cases against ``sqlite:///:memory:`` only. Dialect
equivalence between the SQLite GLOB form
(``content_hash NOT GLOB '*[^a-f0-9]*'``) and the PostgreSQL POSIX
regex form (``content_hash ~ '^[a-f0-9]+$'``) was previously asserted
by code comment alone — the regex variant could mismatch on edge
cases (trailing newline, Unicode lookalikes) and the audit invariant
the Phase 1 schema work was built around would silently break on
PostgreSQL without any test failure.

This module mirrors the SQLite parametrize shape against a
testcontainer Postgres engine, plus regex-specific edge cases that
only manifest on the POSIX regex implementation:

* Trailing newline — POSIX ``^...$`` anchors are line-anchors, NOT
  string-anchors. ``"a"*64 + "\\n"`` matches ``^[a-f0-9]+$`` against
  the first line but fails the ``length() = 64`` clause; both must
  reject in concert.
* Trailing carriage-return — same line-anchor nuance.
* Unicode digit lookalike (Devanagari digit U+0966) — must reject
  even though some locales might consider it numeric.
* Embedded newline mid-hash — ``"a"*32 + "\\n" + "a"*31`` fails
  the length clause; the regex alone would split-line-anchor through
  it.

Together the eight base cases plus four regex-specific edges close
the dialect-equivalence gap that the SQLite-only test left open.
"""

from __future__ import annotations

import uuid
from collections.abc import Iterator
from datetime import UTC, datetime

import pytest
from sqlalchemy import Engine, insert, update
from sqlalchemy.exc import IntegrityError
from testcontainers.postgres import PostgresContainer

from elspeth.web.sessions.engine import create_session_engine
from elspeth.web.sessions.models import blobs_table
from elspeth.web.sessions.schema import initialize_session_schema

# Integration-suite shared session-insert helper. Mirrors the import
# pattern in test_compose_loop_concurrent_sessions.py so all
# testcontainer modules use the same session-row construction.
from .conftest import _make_session

# Skipped: the session-DB schema uses SQLite-only CHECK constraints
# (char()/trim-comma) that fail to CREATE on Postgres, so these tests error at
# schema setup. Postgres is deferred (SQLite-only stance); portability work and
# re-enable steps are tracked in filigree elspeth-f18d996d71.
pytestmark = [
    pytest.mark.testcontainer,
    pytest.mark.skip(reason="Postgres deferred; session-DB CHECK constraints SQLite-only — elspeth-f18d996d71"),
]


@pytest.fixture(scope="module")
def pg_engine() -> Iterator[Engine]:
    """Module-scoped Postgres testcontainer.

    Mirrors the fixture in
    ``tests/integration/web/test_compose_loop_concurrent_sessions.py``
    — same image tag, same engine-construction path, same schema
    bootstrap. Module scope amortises container startup over the
    parametrized cases; the test itself uses fresh blob-record IDs
    per case so there is no cross-test state leak.
    """
    with PostgresContainer("postgres:16-alpine") as pg:
        engine = create_session_engine(pg.get_connection_url())
        initialize_session_schema(engine)
        yield engine


@pytest.fixture
def session_id(pg_engine: Engine) -> str:
    """Insert a fresh session row and return its id.

    Each test case uses a distinct session_id to keep blob inserts
    independent — no row-level cleanup needed because the Postgres
    container is torn down at module exit.
    """
    sid = str(uuid.uuid4())
    with pg_engine.begin() as conn:
        _make_session(
            conn,
            session_id=sid,
            user_id="postgres-test-user",
            title="ck_blobs_ready_hash dialect equivalence",
        )
    return sid


def _insert_pending_blob(pg_engine: Engine, session_id: str) -> str:
    """Insert a pending blob and return its id.

    Pending rows accept NULL content_hash by spec — the CHECK only
    binds when ``status='ready'``. Returning the id lets the caller
    UPDATE the row to ready+malformed-hash and observe the rejection.
    """
    blob_id = str(uuid.uuid4())
    with pg_engine.begin() as conn:
        conn.execute(
            insert(blobs_table).values(
                id=blob_id,
                session_id=session_id,
                filename="legit.csv",
                mime_type="text/csv",
                size_bytes=12,
                content_hash=None,
                storage_path=f"/tmp/postgres-test/{blob_id}",
                created_at=datetime.now(UTC),
                created_by="pipeline",
                status="pending",
            )
        )
    return blob_id


# Eight base parametrizations mirror the SQLite suite verbatim
# (tests/unit/web/blobs/test_service.py:1935-1947) so any future
# divergence between the two surfaces immediately. Adding a base case
# requires updating BOTH this module and the SQLite test.
_BASE_MALFORMED = [
    pytest.param("abc123", id="too-short"),
    pytest.param("a" * 63, id="off-by-one-63"),
    pytest.param("a" * 65, id="off-by-one-65"),
    pytest.param("A" * 64, id="uppercase"),
    pytest.param("g" * 64, id="non-hex-letter"),
    pytest.param("a" * 63 + "Z", id="mostly-hex-with-uppercase"),
    pytest.param("", id="empty"),
    pytest.param("a" * 64 + "\n", id="trailing-newline"),
]

# Regex-specific edge cases that exercise the POSIX ``^...$``
# line-anchor semantics specifically. These cases would not change
# the SQLite GLOB outcome (GLOB has no line-anchor concept) but they
# are the most likely places for the regex variant to diverge.
_REGEX_EDGE_CASES = [
    pytest.param("a" * 64 + "\r", id="trailing-carriage-return"),
    pytest.param("a" * 32 + "\n" + "a" * 31, id="embedded-newline"),
    pytest.param("a" * 63 + "०", id="unicode-devanagari-digit"),  # noqa: RUF001 — Unicode lookalike is the test point
    pytest.param("a" * 63 + "ª", id="unicode-feminine-ordinal"),
]


@pytest.mark.parametrize("bad_hash", _BASE_MALFORMED + _REGEX_EDGE_CASES)
def test_update_ready_hash_to_malformed_rejected_on_postgres(
    pg_engine: Engine,
    session_id: str,
    bad_hash: str,
) -> None:
    """Postgres CHECK constraint rejects malformed content_hash on
    transition to ``status='ready'``.

    Mirrors the SQLite-only test
    ``test_update_ready_hash_to_malformed_rejected`` against the
    Postgres POSIX-regex variant. The base eight cases pin
    dialect equivalence; the four regex-specific edges close the
    line-anchor gap that the SQLite GLOB form does not exercise.

    Each case inserts a fresh pending blob (CHECK does not fire while
    status != 'ready'), then attempts the
    ``status='ready' + content_hash=<bad>`` transition in a single
    UPDATE — exactly the malformed shape a buggy migration script
    would produce. The expected outcome is ``IntegrityError``: the
    Postgres CHECK constraint fires at COMMIT and the transaction
    rolls back.
    """
    blob_id = _insert_pending_blob(pg_engine, session_id)

    with pytest.raises(IntegrityError), pg_engine.begin() as conn:
        conn.execute(update(blobs_table).where(blobs_table.c.id == blob_id).values(content_hash=bad_hash, status="ready"))


def test_inserting_ready_with_valid_hash_succeeds_on_postgres(
    pg_engine: Engine,
    session_id: str,
) -> None:
    """Positive control — a valid 64-char lowercase-hex hash COMMITs.

    Without a positive control, a regression that broke ALL inserts
    (e.g. a misnamed CHECK that always evaluates true) would still
    pass the malformed-rejected parametrization. The valid case pins
    that the constraint accepts the spec-shaped hash.
    """
    blob_id = str(uuid.uuid4())
    valid_hash = "a" * 64

    with pg_engine.begin() as conn:
        conn.execute(
            insert(blobs_table).values(
                id=blob_id,
                session_id=session_id,
                filename="valid.csv",
                mime_type="text/csv",
                size_bytes=42,
                content_hash=valid_hash,
                storage_path=f"/tmp/postgres-test/{blob_id}",
                created_at=datetime.now(UTC),
                created_by="pipeline",
                status="ready",
            )
        )
