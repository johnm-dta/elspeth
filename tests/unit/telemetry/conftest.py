# tests/unit/telemetry/conftest.py
"""Telemetry unit test configuration.

Provides autouse cleanup for TelemetryManager thread leaks and on-demand
in-memory OTEL metric reader for per-counter assertion tests (Prerequisite 2
of ADR-008 runtime cross-check — test #29).

B1-r3 note (2026-05-19): after the MeterProvider precondition lands, a real
PrometheusMetricReader-backed MeterProvider is installed at app module-import
time.  OTel 1.41+ enforces do_once semantics on set_meter_provider — once a
non-NoOp provider is set it cannot be overridden.  The ``in_memory_metric_reader``
fixture below provides a hermetic reader by constructing a local MeterProvider
*without* setting it globally.  Tests that need to assert on counter increments
must create their instruments via ``provider.get_meter()`` (not the global
``metrics.get_meter()``) and monkeypatch any module-level counters they want
to intercept.  See tests/unit/engine/test_executors.py for the canonical pattern.
"""

from collections.abc import Iterator

import pytest
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import InMemoryMetricReader

from tests.fixtures.telemetry import telemetry_manager_cleanup


@pytest.fixture(autouse=True)
def _auto_close_telemetry_managers() -> Iterator[None]:
    """Cleanup TelemetryManager instances after each test."""
    with telemetry_manager_cleanup():
        yield


@pytest.fixture
def in_memory_metric_reader() -> Iterator[InMemoryMetricReader]:
    """Yield an InMemoryMetricReader bound to a local, freshly-constructed MeterProvider.

    IMPORTANT: after B1-r3 this fixture does NOT set the global OTel provider
    (the old save/restore pattern no longer works because OTel 1.41+ enforces
    do_once semantics).  The yielded reader is bound to a local provider that
    is held alive for the duration of the test.  Tests that need to intercept
    module-level counters (e.g. ``pass_through._VIOLATIONS_COUNTER``) must
    construct a local ``MeterProvider`` themselves and monkeypatch those
    module globals with a counter created from it — see
    ``tests/unit/engine/test_executors.py::test_cross_check_increments_telemetry_counter_on_violation``
    for the canonical pattern.  Assert counter increments via
    ``reader.get_metrics_data()``.
    """
    reader = InMemoryMetricReader()
    _provider = MeterProvider(metric_readers=[reader])  # keeps reader registered
    try:
        yield reader
    finally:
        _provider.shutdown()
        reader.shutdown()
