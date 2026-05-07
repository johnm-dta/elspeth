# tests/unit/telemetry/conftest.py
"""Telemetry unit test configuration.

Provides autouse cleanup for TelemetryManager thread leaks and on-demand
in-memory OTEL metric reader for per-counter assertion tests (Prerequisite 2
of ADR-008 runtime cross-check — test #29).
"""

from collections.abc import Iterator

import pytest
from opentelemetry import metrics
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
    """Provide an InMemoryMetricReader bound to a freshly-constructed MeterProvider.

    Yields the reader so tests can call .get_metrics_data() and inspect counter
    increments. Restores the prior meter provider on teardown to prevent
    pytest-xdist worker-state bleed across tests.
    """
    reader = InMemoryMetricReader()
    provider = MeterProvider(metric_readers=[reader])
    prior = metrics.get_meter_provider()
    metrics.set_meter_provider(provider)
    try:
        yield reader
    finally:
        metrics.set_meter_provider(prior)
        reader.shutdown()
