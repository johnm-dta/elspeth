"""Tests for ``elspeth.web.composer.telemetry_phase8`` (Phase 8 Task 1).

Per CLAUDE.md primacy, the audit row is the legal record; these counters
are operational signals. Each helper here is verified for:

1. Correct counter slot is incremented.
2. Recorded attribute dict matches the helper's documented shape.
3. Cross-vocabulary inputs raise ``ValueError`` BEFORE the W5 try/except
   wrap (the assert is a programmer-error guard, not an OTel-exporter
   failure).
4. An exporter raising inside ``.add(...)`` is swallowed and the helper
   returns ``None`` (W5 — load-bearing).

Fixture discipline (Q10). The ``sessions_telemetry`` fixture is
``function``-scoped: every test gets a fresh container with fresh
``_FakeCounter`` instances. A wider scope would let
``observed_value(...)`` assertions go order-dependent because the fake's
``.calls`` list accumulates across tests.

Test-file location divergence from the plan. The plan
(``docs/composer/ux-redesign-2026-05/20-phase-8-polish-and-telemetry.md``
"File structure", line 1086) places this file at
``src/elspeth/web/composer/telemetry_phase8_test.py``. The project
convention (verified at task-start by ``find src/elspeth -name
'*_test.py'`` returning nothing, and ``find tests/unit -name
'test_telemetry*'`` returning ``tests/unit/web/sessions/test_telemetry.py``
and three others) is ``tests/unit/<area>/test_<name>.py``. Following the
project convention; surfacing the divergence in the task report.
"""

from __future__ import annotations

import importlib
import sys
from typing import Any

import pytest

from elspeth.web.composer import telemetry_phase8
from elspeth.web.composer.telemetry_phase8 import (
    SessionsTelemetry,
    record_audit_fetch_failure,
    record_interpretation_opt_out,
    record_mode_opted_in,
    record_mode_opted_out,
    record_session_completed,
    record_session_switched,
    record_share_link_expiry_hit,
    record_share_token_verify_failure,
    record_source_dynamic_created,
    record_tutorial_started,
)
from elspeth.web.sessions.telemetry import _FakeCounter, build_sessions_telemetry, observed_value

# ─────────────────────────────────────────────────────────────────────────
# Function-scoped fixture (Q10 — load-bearing)
# ─────────────────────────────────────────────────────────────────────────


@pytest.fixture()
def sessions_telemetry() -> SessionsTelemetry:
    """Fresh container with fresh fake counters per test.

    MUST stay function-scoped. The ``_FakeCounter.calls`` list
    accumulates across the counter's lifetime, so a wider scope would
    cross-contaminate ``observed_value`` assertions across tests.
    """
    return build_sessions_telemetry()


# ─────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────


def _fake_calls(counter: object) -> list[tuple[int | float, dict[str, Any] | None, Any]]:
    """Type-narrow to ``_FakeCounter`` and return its recorded calls.

    Production OTel counters do not expose ``.calls``; this helper
    asserts the test is running against the fake-counter branch of
    ``build_sessions_telemetry`` and surfaces a clear error if it isn't.
    """
    assert isinstance(counter, _FakeCounter), (
        f"expected _FakeCounter but got {type(counter).__name__}; tests must use build_sessions_telemetry(meter=None)"
    )
    return counter.calls


# ─────────────────────────────────────────────────────────────────────────
# Q1 — Module-local probe-failure counter
# ─────────────────────────────────────────────────────────────────────────


def test_phase_8_probe_failed_counter_records_phase_and_probe_attributes(monkeypatch: pytest.MonkeyPatch) -> None:
    """Module-local W8-r2 counter exists and accepts the documented
    attribute shape.

    UNLIKE the other tests in this file, this counter is NOT a slot on
    ``_SessionsTelemetry``: it's a module-local OTel counter constructed
    at ``telemetry_phase8.py`` import time via ``metrics.get_meter()``.
    The ``_FakeCounter.calls`` / ``observed_value`` pattern used by
    sibling tests does not apply. We instead substitute a fake meter
    *before* reimporting the module, then assert on the fake's recorded
    ``create_counter`` + ``.add`` calls. A future maintainer should not
    "harmonise" this test into the container-based shape — the contrast
    is deliberate.
    """

    class _RecordingCounter:
        def __init__(self) -> None:
            self.adds: list[tuple[int | float, dict[str, Any] | None]] = []

        def add(self, amount: int | float, attributes: dict[str, Any] | None = None, context: Any = None) -> None:
            del context
            self.adds.append((amount, dict(attributes) if attributes is not None else None))

    class _RecordingMeter:
        def __init__(self) -> None:
            self.created: dict[str, _RecordingCounter] = {}

        def create_counter(self, name: str, *, description: str = "", unit: str = "") -> _RecordingCounter:
            del description, unit
            counter = _RecordingCounter()
            self.created[name] = counter
            return counter

    fake_meter = _RecordingMeter()

    def _fake_get_meter(name: str) -> _RecordingMeter:
        del name
        return fake_meter

    # Substitute ``metrics.get_meter`` before re-importing the module so
    # the module-level ``_meter = metrics.get_meter(__name__)`` line picks
    # up the fake. We restore the original module afterwards so other
    # tests using ``from elspeth.web.composer.telemetry_phase8 import …``
    # are not affected.
    monkeypatch.setattr("opentelemetry.metrics.get_meter", _fake_get_meter)
    original_module = sys.modules.get("elspeth.web.composer.telemetry_phase8")
    try:
        # ``importlib.reload`` re-executes the module body, so the
        # module-level counter creation happens against our fake meter.
        reloaded = importlib.reload(telemetry_phase8)
        assert "composer.phase_8.probe_failed_total" in fake_meter.created, (
            f"Module-level counter creation did not call create_counter with the expected name. Saw: {sorted(fake_meter.created.keys())}"
        )
        probe_counter = fake_meter.created["composer.phase_8.probe_failed_total"]
        reloaded._PHASE_8_PROBE_FAILED_COUNTER.add(  # type: ignore[attr-defined]
            1,
            attributes={"phase": "Task 4", "probe": "HeaderSessionSwitcher"},
        )
        assert probe_counter.adds == [
            (1, {"phase": "Task 4", "probe": "HeaderSessionSwitcher"}),
        ]
    finally:
        # Restore the original module so importers in other tests see
        # the original singleton again. Without this restore, the
        # reloaded module would persist and any ``from … import name``
        # rebound by reload would point at the post-reload constants
        # only — sibling test files importing at module load already
        # have stable references, but this guard keeps the import system
        # tidy.
        if original_module is not None:
            sys.modules["elspeth.web.composer.telemetry_phase8"] = original_module


# ─────────────────────────────────────────────────────────────────────────
# Q2 — real-meter snapshot test
# ─────────────────────────────────────────────────────────────────────────


def test_factory_registers_canonical_counter_names() -> None:
    """Snapshot test: exact equality on the wire-name set registered by
    ``build_sessions_telemetry`` in the real-meter branch.

    Catches both typos (``opt_out_total`` losing the ``ed``,
    ``token_verify_failures`` with a trailing ``s``) AND accidental
    additions (a future PR adding a counter without updating this
    snapshot fails the test until the snapshot is intentionally
    updated, forcing the new wire name through code review).

    ``composer.phase_8.probe_failed_total`` is NOT in the expected set —
    per W8-r2 it's module-local in ``telemetry_phase8.py``, not a
    ``_SessionsTelemetry`` slot, and is covered by Q1 above.
    ``composer.tutorial.replayed_total`` is also absent (Phase 9
    deferred per Decision 2 / Option C).
    """

    class _RecordingMeter:
        def __init__(self) -> None:
            self.names: list[str] = []

        def create_counter(self, name: str, *, description: str = "", unit: str = "") -> _FakeCounter:
            del description, unit
            self.names.append(name)
            return _FakeCounter()

    meter = _RecordingMeter()
    build_sessions_telemetry(meter=meter)

    # The real-meter branch wires every counter including Phase 1+ ones;
    # this snapshot pins the Phase-8 wire names EXACTLY (set equality) so
    # both directions are caught: a missing name (typo, deletion) AND an
    # accidental extra Phase-8-shaped name (a future PR adds a counter
    # without updating this snapshot or filing a Decision-2-style
    # exception).  The broader full-set assertion lives in
    # ``tests/unit/web/sessions/test_telemetry.py``
    # (``test_production_meter_registers_named_metrics``) and covers the
    # complete cross-phase wire-name set; this assertion is the
    # Phase-8-scoped equivalent for the module's own test surface.
    phase_8_expected: frozenset[str] = frozenset(
        {
            "composer.mode.opted_out_total",
            "composer.mode.opted_in_total",
            "composer.session.switched_total",
            "composer.tutorial.started_total",
            # composer.tutorial.completed_total — counted ONLY by
            # composer/tutorial_telemetry.py (attribute-carrying); the
            # sessions-meter registration was a double-count and was removed.
            # composer.tutorial.replayed_total — deferred to Phase 9
            # per Decision 2 resolution (Option C). Do NOT add it back
            # to expected_names without re-opening that decision.
            "composer.session.completed_total",
            "composer.share.token_verify_failure_total",
            "composer.share.link_expiry_hit_total",
            "composer.interpretation.opt_out_total",
            "composer.audit.fetch_failure_total",
            "composer.source.dynamic_created_total",
        }
    )
    phase_8_actual = frozenset(name for name in meter.names if name.startswith("composer."))
    # Subtract pre-Phase-8 composer counters so the assertion only sees the
    # Phase 8 surface.  The list is hand-maintained alongside this test;
    # when a counter graduates out of Phase 8 (or a pre-Phase-8 counter is
    # retired), update both lists in the same commit.
    pre_phase_8_composer_counters: frozenset[str] = frozenset(
        {
            "composer.audit.tool_row_tier1_violation_total",
            "composer.audit.state_rolled_back_during_persist_total",
            "composer.audit.tool_row_persist_failed_during_unwind_total",
            "composer.audit.tool_row_integrity_violation_total",
            "composer.tool_call_cap_exceeded_total",
            "composer.audit.audit_grade_view_total",
            "composer.audit.audit_access_log_write_failed_total",
            "composer.interpretation_rate_cap_exceeded_total",
            "composer.interpretation_placeholder_unresolved_at_runtime_total",
        }
    )
    phase_8_actual_filtered = phase_8_actual - pre_phase_8_composer_counters
    missing = phase_8_expected - phase_8_actual_filtered
    extra = phase_8_actual_filtered - phase_8_expected
    assert not missing, f"missing Phase 8 wire name(s) in real-meter branch: {sorted(missing)}"
    assert not extra, (
        f"extra Phase-8-shaped wire name(s) registered by factory but not in this test's expected set: "
        f"{sorted(extra)}.  If this is intentional, update phase_8_expected (and reconcile with "
        f"21-phase-9-followups.md if the counter is a Phase-9 graduation)."
    )
    # Probe-failed counter is module-local, not factory-wired.
    assert "composer.phase_8.probe_failed_total" not in meter.names, (
        "probe-failed counter must NOT be wired through build_sessions_telemetry — it is module-local in telemetry_phase8.py per W8-r2."
    )


# ─────────────────────────────────────────────────────────────────────────
# Per-helper Q1-shaped tests
# ─────────────────────────────────────────────────────────────────────────


def test_record_mode_opted_out_increments_counter(sessions_telemetry: SessionsTelemetry) -> None:
    """Account-level opt-out helper: post-state-only, kwarg-free,
    attribute-free per §B2.b.
    """
    record_mode_opted_out(sessions_telemetry)
    assert observed_value(sessions_telemetry.mode_opted_out_total) == 1
    calls = _fake_calls(sessions_telemetry.mode_opted_out_total)
    assert calls == [(1, {}, None)]


def test_record_mode_opted_in_increments_counter(sessions_telemetry: SessionsTelemetry) -> None:
    """Account-level opt-in helper: symmetric to opt-out."""
    record_mode_opted_in(sessions_telemetry)
    assert observed_value(sessions_telemetry.mode_opted_in_total) == 1
    calls = _fake_calls(sessions_telemetry.mode_opted_in_total)
    assert calls == [(1, {}, None)]


def test_record_session_switched_increments_counter(sessions_telemetry: SessionsTelemetry) -> None:
    """Per-session trust_mode switch — records both from_mode and to_mode
    attributes drawn from the per-session ``trust_mode`` vocabulary.
    """
    record_session_switched(sessions_telemetry, from_mode="explicit_approve", to_mode="auto_commit")
    assert observed_value(sessions_telemetry.session_switched_total) == 1
    calls = _fake_calls(sessions_telemetry.session_switched_total)
    assert calls == [
        (1, {"from_mode": "explicit_approve", "to_mode": "auto_commit"}, None),
    ]


@pytest.mark.parametrize(
    ("from_mode", "to_mode"),
    [
        ("explicit_approve", "auto_commit"),
        ("auto_commit", "explicit_approve"),
        ("explicit_approve", "explicit_approve"),
        ("auto_commit", "auto_commit"),
    ],
)
def test_record_session_switched_accepts_all_valid_combinations(
    sessions_telemetry: SessionsTelemetry, from_mode: str, to_mode: str
) -> None:
    """Every combination of per-session vocabulary values is accepted.

    Vocabulary is intentionally NOT parametrised over ``"guided"`` /
    ``"freeform"`` / ``"unknown"`` — those are wrong vocabulary
    (account-level column) or fabricated values that the per-session
    column does not admit. Cross-vocabulary rejection is asserted by
    the test below.
    """
    record_session_switched(sessions_telemetry, from_mode=from_mode, to_mode=to_mode)  # type: ignore[arg-type]
    assert observed_value(sessions_telemetry.session_switched_total) == 1


def test_record_session_switched_rejects_cross_vocabulary_account_mode(sessions_telemetry: SessionsTelemetry) -> None:
    """B1-r2 regression: ``"guided"`` is valid for the account-level
    ``default_composer_mode`` column and INVALID for the per-session
    ``trust_mode`` column. A pass-1 draft sharing a single mode Literal
    would have admitted this call. Re-running this assertion catches
    future regressions that re-share the Literal.
    """
    with pytest.raises(ValueError, match=r"from_mode must be"):
        record_session_switched(sessions_telemetry, from_mode="guided", to_mode="auto_commit")  # type: ignore[arg-type]


def test_record_session_switched_rejects_fabricated_mode(sessions_telemetry: SessionsTelemetry) -> None:
    """Generic out-of-set guard: a fabricated value is rejected."""
    with pytest.raises(ValueError, match=r"from_mode must be"):
        record_session_switched(sessions_telemetry, from_mode="yolo", to_mode="auto_commit")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match=r"to_mode must be"):
        record_session_switched(sessions_telemetry, from_mode="explicit_approve", to_mode="yolo")  # type: ignore[arg-type]


def test_record_tutorial_started_increments_counter(sessions_telemetry: SessionsTelemetry) -> None:
    """Tutorial-started helper (Phase 4 surface; helper shipped Phase 8
    for forward-fit).
    """
    record_tutorial_started(sessions_telemetry)
    assert observed_value(sessions_telemetry.tutorial_started_total) == 1
    calls = _fake_calls(sessions_telemetry.tutorial_started_total)
    assert calls == [(1, {}, None)]


def test_record_session_completed_increments_counter(sessions_telemetry: SessionsTelemetry) -> None:
    """Completion-gesture helper: DB-authoritative completion-verb
    vocabulary sourced from
    ``composer_completion_events_table.event_type`` CHECK constraint
    (``src/elspeth/web/sessions/models.py:735``).
    """
    record_session_completed(sessions_telemetry, completion_verb="mark_ready_for_review")
    assert observed_value(sessions_telemetry.session_completed_total) == 1
    calls = _fake_calls(sessions_telemetry.session_completed_total)
    assert calls == [
        (1, {"completion_verb": "mark_ready_for_review"}, None),
    ]


@pytest.mark.parametrize("completion_verb", ["mark_ready_for_review", "export_yaml"])
def test_record_session_completed_accepts_all_valid_combinations(sessions_telemetry: SessionsTelemetry, completion_verb: str) -> None:
    """Every value in the DB CHECK constraint vocabulary is accepted.

    Note: the parametrize set is INTENTIONALLY narrowed to the two
    values in the audit-row CHECK constraint. ``save_for_review`` is
    UI-only vocabulary (different from the DB audit row) and
    ``run_pipeline`` has no audit row in
    ``composer_completion_events_table`` (its audit lives under
    ``runs/``); both would violate the CLAUDE.md superset rule and
    are rejected by the negative tests below.
    """
    record_session_completed(sessions_telemetry, completion_verb=completion_verb)  # type: ignore[arg-type]
    assert observed_value(sessions_telemetry.session_completed_total) == 1


def test_record_session_completed_rejects_save_for_review_ui_vocab(sessions_telemetry: SessionsTelemetry) -> None:
    """Regression for the Sub-task 7c overall-plan reviewer finding:
    ``"save_for_review"`` is UI-facing vocabulary that DRIFTS from the
    DB CHECK-constraint value (``"mark_ready_for_review"`` is what the
    audit row carries). Accepting it here would silently break the
    superset rule because aggregation by ``completion_verb`` would
    return a value that has no corresponding row in
    ``composer_completion_events_table``.
    """
    with pytest.raises(ValueError, match=r"completion_verb must be"):
        record_session_completed(sessions_telemetry, completion_verb="save_for_review")  # type: ignore[arg-type]


def test_record_session_completed_rejects_run_pipeline(sessions_telemetry: SessionsTelemetry) -> None:
    """``run_pipeline`` is a UX-level verb that does NOT write a
    ``composer_completion_events_table`` row (pipeline runs are audited
    under the ``runs`` table). Including it here would violate the
    superset rule: the counter aggregates over committed
    completion-events rows, and ``run_pipeline`` has none. A future
    aggregate over runs must be a separate counter
    (``composer.run.started_total`` or similar).
    """
    with pytest.raises(ValueError, match=r"completion_verb must be"):
        record_session_completed(sessions_telemetry, completion_verb="run_pipeline")  # type: ignore[arg-type]


def test_record_session_completed_rejects_unknown_completion_verb(sessions_telemetry: SessionsTelemetry) -> None:
    """Out-of-set completion verb is rejected."""
    with pytest.raises(ValueError, match=r"completion_verb must be"):
        record_session_completed(sessions_telemetry, completion_verb="abandon")  # type: ignore[arg-type]


def test_record_share_token_verify_failure_increments_counter(sessions_telemetry: SessionsTelemetry) -> None:
    record_share_token_verify_failure(sessions_telemetry)
    assert observed_value(sessions_telemetry.share_token_verify_failure_total) == 1
    assert _fake_calls(sessions_telemetry.share_token_verify_failure_total) == [(1, {}, None)]


def test_record_share_link_expiry_hit_increments_counter(sessions_telemetry: SessionsTelemetry) -> None:
    record_share_link_expiry_hit(sessions_telemetry)
    assert observed_value(sessions_telemetry.share_link_expiry_hit_total) == 1
    assert _fake_calls(sessions_telemetry.share_link_expiry_hit_total) == [(1, {}, None)]


def test_record_interpretation_opt_out_increments_counter(sessions_telemetry: SessionsTelemetry) -> None:
    record_interpretation_opt_out(sessions_telemetry)
    assert observed_value(sessions_telemetry.interpretation_opt_out_total) == 1
    assert _fake_calls(sessions_telemetry.interpretation_opt_out_total) == [(1, {}, None)]


def test_record_audit_fetch_failure_increments_counter(sessions_telemetry: SessionsTelemetry) -> None:
    record_audit_fetch_failure(sessions_telemetry)
    assert observed_value(sessions_telemetry.audit_fetch_failure_total) == 1
    assert _fake_calls(sessions_telemetry.audit_fetch_failure_total) == [(1, {}, None)]


def test_record_source_dynamic_created_increments_counter(sessions_telemetry: SessionsTelemetry) -> None:
    record_source_dynamic_created(sessions_telemetry)
    assert observed_value(sessions_telemetry.source_dynamic_created_total) == 1
    assert _fake_calls(sessions_telemetry.source_dynamic_created_total) == [(1, {}, None)]


# ─────────────────────────────────────────────────────────────────────────
# W5 — load-bearing: exporter-failure swallow
# ─────────────────────────────────────────────────────────────────────────


class _RaisingCounter:
    """``_Counter``-protocol-shaped fake whose ``add`` raises.

    Simulates an OTel exporter failure mid-emit. The Phase 8 helpers
    MUST swallow this (W5 — load-bearing) so a broken exporter cannot
    500 a PATCH whose audit row already wrote.
    """

    def __init__(self) -> None:
        self.attempts = 0

    def add(self, amount: int | float, attributes: Any = None, context: Any = None) -> None:
        del amount, attributes, context
        self.attempts += 1
        raise RuntimeError("simulated exporter failure")


def _swap_counter(tel: SessionsTelemetry, field: str, replacement: _RaisingCounter) -> SessionsTelemetry:
    """Build a new ``SessionsTelemetry`` with ``field`` replaced by
    ``replacement``.

    ``SessionsTelemetry`` is frozen+slots — we can't mutate in place.
    Rebuild via ``dataclasses.replace``. The shape is verified by the
    swap call type-checking.
    """
    from dataclasses import replace

    return replace(tel, **{field: replacement})  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("field", "invoke"),
    [
        ("mode_opted_out_total", lambda tel: record_mode_opted_out(tel)),
        ("mode_opted_in_total", lambda tel: record_mode_opted_in(tel)),
        (
            "session_switched_total",
            lambda tel: record_session_switched(tel, from_mode="explicit_approve", to_mode="auto_commit"),
        ),
        ("tutorial_started_total", lambda tel: record_tutorial_started(tel)),
        (
            "session_completed_total",
            lambda tel: record_session_completed(tel, completion_verb="mark_ready_for_review"),
        ),
        ("share_token_verify_failure_total", lambda tel: record_share_token_verify_failure(tel)),
        ("share_link_expiry_hit_total", lambda tel: record_share_link_expiry_hit(tel)),
        ("interpretation_opt_out_total", lambda tel: record_interpretation_opt_out(tel)),
        ("audit_fetch_failure_total", lambda tel: record_audit_fetch_failure(tel)),
        ("source_dynamic_created_total", lambda tel: record_source_dynamic_created(tel)),
    ],
)
def test_helper_swallows_exporter_failure(
    sessions_telemetry: SessionsTelemetry,
    field: str,
    invoke: Any,
) -> None:
    """Every record_* helper wraps its ``.add(...)`` call in try/except
    that swallows the exception and returns None (W5 — load-bearing).
    """
    raising = _RaisingCounter()
    tel_with_raise = _swap_counter(sessions_telemetry, field, raising)
    # No raise should propagate out of the helper.
    result = invoke(tel_with_raise)
    assert result is None
    assert raising.attempts == 1, "Helper must call counter.add exactly once even when add raises."


def test_record_session_switched_assert_runs_before_swallow(sessions_telemetry: SessionsTelemetry) -> None:
    """The ValueError from ``_assert_session_trust_mode`` MUST escape —
    it's a programmer-error guard, not an exporter failure, and is
    raised BEFORE the W5 try/except. If a future refactor moves the
    assert inside the try, this test will fail and surface the
    regression. See module docstring §"OTel exporter failure handling
    (W5 — load-bearing)".
    """
    raising = _RaisingCounter()
    tel_with_raise = _swap_counter(sessions_telemetry, "session_switched_total", raising)
    with pytest.raises(ValueError, match=r"from_mode must be"):
        record_session_switched(tel_with_raise, from_mode="guided", to_mode="auto_commit")  # type: ignore[arg-type]
    # The exporter must never have been touched because the assert
    # fired first.
    assert raising.attempts == 0
