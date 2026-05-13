"""Tests for the named OTel counters introduced in spec §1.4 / §5.7.4."""

from __future__ import annotations

from elspeth.web.sessions.telemetry import (
    _FakeCounter,
    build_sessions_telemetry,
    observed_value,
)

# Spec §1.4 NFR table plus Phase 3 audit-grade transcript access counters.
# Verified end-to-end by ``test_production_meter_registers_named_metrics`` below.
EXPECTED_METRIC_NAMES = {
    "composer.audit.tool_row_tier1_violation_total",
    "composer.audit.state_rolled_back_during_persist_total",
    "composer.audit.tool_row_persist_failed_during_unwind_total",
    "composer.audit.tool_row_integrity_violation_total",
    "composer.tool_call_cap_exceeded_total",
    "composer.audit.audit_grade_view_total",
    "composer.audit.audit_access_log_write_failed_total",
}


def test_telemetry_field_names_match_spec_exactly():
    """Use ``set ==`` (not ``issubset``) so an accidental rename — say
    ``tier1_violation_total`` losing its ``tool_row_`` prefix — fails
    the test rather than passing under ``issubset``. Closes synthesised
    review finding L10 / Q-F-13."""
    telem = build_sessions_telemetry()
    expected_fields = {
        "tool_row_tier1_violation_total",
        "state_rolled_back_during_persist_total",
        "tool_row_persist_failed_during_unwind_total",
        "tool_row_integrity_violation_total",
        "tool_call_cap_exceeded_total",
        "audit_grade_view_total",
        "audit_access_log_write_failed_total",
    }
    actual = set(telem.__dataclass_fields__)
    assert actual == expected_fields, f"field-name mismatch — added: {actual - expected_fields}; removed: {expected_fields - actual}"


def test_counter_increments_visible_via_observed_value_helper():
    """Test path: build_sessions_telemetry() with no meter returns
    fake counters; the ``observed_value`` helper extracts cumulative
    sum after type-narrowing to ``_FakeCounter``. Production code
    never reads ``observed_value`` — it only writes via ``add()`` —
    so the helper makes the test-only inspection explicit at the
    call site."""
    telem = build_sessions_telemetry()
    starting = observed_value(telem.tool_row_tier1_violation_total)
    telem.tool_row_tier1_violation_total.add(1)
    assert observed_value(telem.tool_row_tier1_violation_total) == starting + 1


def test_counter_records_attributes_dict():
    """Real OTel ``Counter.add`` accepts ``attributes`` as the second
    positional/keyword argument. Production code at composer/service.py
    and routes.py uses this for structured emission (e.g.,
    ``add(1, {"outcome": "failure"})``). The fake must mirror the
    signature so tests with attributed metrics do not raise
    ``TypeError`` against a fake-narrow ``add(amount)`` signature.
    Closes synthesised review finding H6."""
    telem = build_sessions_telemetry()
    telem.tool_row_tier1_violation_total.add(1, {"reason": "commit_failure", "session_id": "s_test"})
    fake = telem.tool_row_tier1_violation_total
    assert isinstance(fake, _FakeCounter)
    assert fake.calls == [
        (1, {"reason": "commit_failure", "session_id": "s_test"}, None),
    ]


def test_production_meter_registers_named_metrics():
    """Closes synthesised review finding F-10 / L7. Verifies that the
    four Phase-1 ``meter.create_counter(...)`` strings in
    ``build_sessions_telemetry`` match spec §1.4 exactly. Without this
    test, a typo (e.g. ``tool_row_tier1_violations_total`` with a
    spurious ``s``) would pass the field-name check (which inspects
    Python attribute names, not OTel metric names) and silently break
    production observability."""

    class _RecordingMeter:
        """Captures the names passed to ``create_counter`` so the
        test can assert them as a set."""

        def __init__(self) -> None:
            self.registered: dict[str, _FakeCounter] = {}

        def create_counter(self, name: str) -> _FakeCounter:
            counter = _FakeCounter()
            self.registered[name] = counter
            return counter

    meter = _RecordingMeter()
    build_sessions_telemetry(meter=meter)
    assert set(meter.registered.keys()) == EXPECTED_METRIC_NAMES
