"""Schema tests for the rev-4 composition_states.provenance column.

Uses the shared ``engine`` fixture and ``_make_session`` helper from
``tests/unit/web/conftest.py``.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from sqlalchemy import insert
from sqlalchemy.exc import IntegrityError

from elspeth.web.sessions import models
from tests.unit.web.conftest import _make_session


def test_provenance_column_exists():
    cols = {c.name for c in models.composition_states_table.columns}
    assert "provenance" in cols


def test_provenance_check_accepts_known_values(engine):
    """Each provenance value listed in the spec §4.1.2 enum must insert
    cleanly. The test exercises the per-row CHECK in isolation; it does
    not assert anything about the semantic correctness of each value
    (that is the spec's concern, not the schema's)."""
    now = datetime.now(UTC)
    for provenance in (
        "tool_call",
        "convergence_persist",
        "plugin_crash_persist",
        "preflight_persist",
        "tutorial_normalization",
        "session_seed",
        "session_fork",
        "interpretation_resolve",
    ):
        with engine.begin() as conn:
            _make_session(conn, session_id=f"s_{provenance}")
            conn.execute(
                insert(models.composition_states_table).values(
                    id=f"cs_{provenance}",
                    session_id=f"s_{provenance}",
                    version=1,
                    provenance=provenance,
                    created_at=now,
                    # is_valid has a Python-side default of False on the
                    # column declaration, so it does not need to be passed
                    # explicitly. All JSON content columns are nullable.
                )
            )


def test_provenance_check_rejects_unknown_value(engine):
    with engine.begin() as conn:
        _make_session(conn, session_id="s1")
        with pytest.raises(IntegrityError, match="ck_composition_states_provenance"):
            conn.execute(
                insert(models.composition_states_table).values(
                    id="cs_x",
                    session_id="s1",
                    version=1,
                    provenance="rogue_value",
                    created_at=datetime.now(UTC),
                )
            )


def test_provenance_not_null(engine):
    with engine.begin() as conn:
        _make_session(conn, session_id="s1")
        # SQLite reports NOT NULL violations as
        # ``NOT NULL constraint failed: composition_states.provenance``,
        # so anchor on the column reference rather than a constraint name.
        with pytest.raises(IntegrityError, match=r"NOT NULL.*composition_states.provenance"):
            conn.execute(
                insert(models.composition_states_table).values(
                    id="cs_x",
                    session_id="s1",
                    version=1,
                    provenance=None,
                    created_at=datetime.now(UTC),
                )
            )
