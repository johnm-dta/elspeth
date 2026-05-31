"""Spec §1.4 NFR: per-turn DB write overhead p95 ≤ 250 ms with N ≤ 8 tool
calls, measured in CI on standard infra. Order-of-magnitude bound, not a
tight budget; the tight 25 ms target is verified by a nightly bench job."""

from __future__ import annotations

import statistics
import time

import pytest
import structlog

# Plan-body deviation: spec §Task 17 lines 833 and 856 use ``text(...)`` inside
# the warm-up and measured loops but omit the top-level ``from sqlalchemy import
# text`` import. Added here so the file is runnable; preserves spec semantics.
from sqlalchemy import text
from sqlalchemy.pool import StaticPool

from elspeth.web.sessions._persist_payload import RedactedToolRow, StatePayload
from elspeth.web.sessions.engine import create_session_engine
from elspeth.web.sessions.protocol import CompositionStateData
from elspeth.web.sessions.schema import initialize_session_schema
from elspeth.web.sessions.service import SessionServiceImpl
from elspeth.web.sessions.telemetry import build_sessions_telemetry

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
        eng,
        data_dir=tmp_path,
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
        RedactedToolRow(
            f"turn_{turn}_tc_{i}",
            '{"ok": true}',
            StatePayload(
                data=CompositionStateData(),
                derived_from_state_id=None,
            ),
        )
        for i in range(8)
    )
    calls = tuple({"id": f"turn_{turn}_tc_{i}", "function": {"name": "f"}} for i in range(8))
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
            expected_current_state_id = conn.execute(
                text("SELECT id FROM composition_states WHERE session_id='lat' ORDER BY version DESC LIMIT 1")
            ).scalar_one()

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
            expected_current_state_id = conn.execute(
                text("SELECT id FROM composition_states WHERE session_id='lat' ORDER BY version DESC LIMIT 1")
            ).scalar_one()

    p95 = statistics.quantiles(durations, n=20)[18]  # 19th of 20 quantiles == p95
    assert p95 < 250, (
        f"p95={p95:.1f}ms exceeds 250ms sanity bound (warmup={_WARMUP_TURNS} discarded, measured={_MEASURED_TURNS}); durations={durations}"
    )
