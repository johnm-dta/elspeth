"""Tests for the B1-r3 MeterProvider precondition in elspeth.web.app.

Verifies that ``create_app()`` installs a real ``MeterProvider`` (not the OTel
default ``NoOpMeterProvider``) and that the FastAPI app exposes a Prometheus
scrape endpoint at ``GET /metrics``.

Process-global side effect notice
----------------------------------
``elspeth.web.app.create_app()`` sets the global OTel ``MeterProvider`` during
application construction (process-global per OTel design, do_once semantics in
OTel 1.41+).
Once a real (non-NoOp) provider is set, subsequent ``set_meter_provider``
calls are silently ignored — the old save/restore test pattern no longer works.

Tests in this module that need an isolated reader
(``test_counter_emits_to_in_memory_reader``) use ``provider.get_meter()``
directly rather than the global ``metrics.get_meter()`` API.  Module-level
counters in production code (e.g. ``pass_through._VIOLATIONS_COUNTER``) can
be intercepted by monkeypatching the module global with a counter created from
a local provider — see ``tests/unit/engine/test_executors.py`` for the
canonical pattern.

Future test authors: if you write a test here that asserts no-op counter
behaviour, you will be asserting against the *pre-B1-r3 state* and the test
will fail. The correct baseline after B1-r3 is that counters emit; write tests
that assert on emitted data.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

from opentelemetry import metrics
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import InMemoryMetricReader
from pydantic import SecretBytes
from starlette.responses import Response
from starlette.routing import Route

# ---------------------------------------------------------------------------
# Test 1 — creating the app wires a real MeterProvider
# ---------------------------------------------------------------------------


def test_meter_provider_is_not_noop(tmp_path: Path) -> None:
    """``create_app()`` installs a real ``MeterProvider``, not the OTel default.

    This is the B1-r3 load-bearing precondition: without a real provider every
    counter in the codebase silently discards its data.
    """
    from elspeth.web.app import create_app
    from elspeth.web.config import WebSettings

    settings = WebSettings(
        data_dir=tmp_path,
        composer_max_composition_turns=15,
        composer_max_discovery_turns=10,
        composer_timeout_seconds=85.0,
        composer_rate_limit_per_minute=10,
        shareable_link_signing_key=SecretBytes(b"\x00" * 32),
    )
    create_app(settings)

    provider = metrics.get_meter_provider()
    # Must be the real SDK MeterProvider, NOT the OTel no-op.
    from opentelemetry.sdk.metrics import MeterProvider as SDKMeterProvider

    assert isinstance(provider, SDKMeterProvider), (
        f"Expected MeterProvider but got {type(provider).__name__!r}. B1-r3 precondition not satisfied — set_meter_provider was not called."
    )


# ---------------------------------------------------------------------------
# Test 2 — a counter bound to an in-memory reader records data
# ---------------------------------------------------------------------------


def test_counter_emits_to_in_memory_reader() -> None:
    """A counter created via ``provider.get_meter()`` and ``create_counter()``
    records an increment that appears in the metric reader's output.

    Uses an isolated ``InMemoryMetricReader`` bound to its own ``MeterProvider``
    via direct provider access (NOT via the global ``metrics.get_meter()``).
    This is hermetic with respect to whatever reader ``app.py`` installed
    globally — after B1-r3 the global is a Prometheus provider that cannot
    be overridden (OTel 1.41+ do_once semantics).
    """
    reader = InMemoryMetricReader()
    provider = MeterProvider(metric_readers=[reader])
    try:
        # Use provider.get_meter() directly — NOT the global metrics.get_meter().
        # The global API goes through the process-level singleton set by app.py.
        meter = provider.get_meter("elspeth.test.b1r3")
        counter = meter.create_counter("test_counter_b1r3")
        counter.add(1, {"env": "test"})

        metrics_data = reader.get_metrics_data()
        assert metrics_data is not None

        found = False
        for rm in metrics_data.resource_metrics:
            for sm in rm.scope_metrics:
                for metric in sm.metrics:
                    if metric.name == "test_counter_b1r3":
                        for point in metric.data.data_points:
                            if dict(point.attributes or {}).get("env") == "test":
                                number_point = cast(Any, point)
                                assert number_point.value >= 1
                                found = True

        assert found, (
            "Expected test_counter_b1r3 with env=test in metrics data but it was not found. Counter.add() appears to be discarding data."
        )
    finally:
        reader.shutdown()


# ---------------------------------------------------------------------------
# Test 3 — GET /metrics returns Prometheus exposition format
# ---------------------------------------------------------------------------


def test_metrics_endpoint_returns_prometheus_format(tmp_path: Path) -> None:
    """``GET /metrics`` on the FastAPI app returns HTTP 200 with
    Prometheus exposition format.

    Verifies:
    - Status code 200.
    - Content-Type contains ``text/plain`` (Prometheus default).
    - Body contains at least one non-comment, non-empty line (a real metric or
      the ``# HELP`` / ``# TYPE`` preamble).

    Uses the synchronously registered FastAPI route directly so the test runs
    without the application lifespan or an AnyIO portal. The ``/metrics`` route
    is independent of the lifespan context; it works before ``yield`` completes.
    """
    from opentelemetry import metrics

    from elspeth.web.app import create_app
    from elspeth.web.config import WebSettings

    settings = WebSettings(
        data_dir=tmp_path,
        composer_max_composition_turns=15,
        composer_max_discovery_turns=10,
        composer_timeout_seconds=85.0,
        composer_rate_limit_per_minute=10,
        shareable_link_signing_key=SecretBytes(b"\x00" * 32),
    )

    app = create_app(settings)

    # Pin the emit → exposition round-trip.  Without a known counter that
    # we register and increment, the previous body assertion ("at least
    # one non-comment line OR # HELP/# TYPE preamble") passed against a
    # stub body of just preamble.  A regression where every
    # production-registered counter silently failed to appear in
    # exposition would still produce ``# HELP`` / ``# TYPE`` lines and
    # pass.  By emitting a known counter and asserting its specific wire
    # name, we anchor the test to actual emit behaviour.
    pin_meter_name = "test.meter_provider.endpoint_pin"
    pin_counter_name = "elspeth_test_metrics_endpoint_pin_total"
    pin_meter = metrics.get_meter(pin_meter_name)
    pin_counter = pin_meter.create_counter(
        pin_counter_name.removesuffix("_total"),
        description="Pin counter for /metrics exposition round-trip test.",
    )
    pin_counter.add(1, {"phase8_pr_review": "s2"})

    # Call the registered synchronous route directly. TestClient would add an
    # AnyIO portal and, when used as a context manager, run lifespan startup;
    # neither is part of this route-level meter -> exposition contract.
    metrics_route = next(
        route for route in app.routes if getattr(route, "path", None) == "/metrics" and "GET" in getattr(route, "methods", set())
    )
    assert isinstance(metrics_route, Route)
    response = metrics_route.endpoint()
    assert isinstance(response, Response)

    assert response.status_code == 200, (
        f"Expected 200 from GET /metrics but got {response.status_code}. "
        "Prometheus endpoint not mounted — check create_app() /metrics mount."
    )

    content_type = response.headers.get("content-type", "")
    assert "text/plain" in content_type, f"Expected text/plain content-type from /metrics but got {content_type!r}."

    body = bytes(response.body).decode("utf-8")
    # Hard-pin: the counter we just emitted must appear in exposition with
    # the value we added.  Prometheus normalises the metric name (dot →
    # underscore) but otherwise preserves it; the value line carries the
    # attribute we attached.
    assert pin_counter_name in body, (
        f"Pin counter {pin_counter_name!r} not present in /metrics body — "
        "registered counters are not reaching exposition.  Body head: "
        f"{body[:500]!r}"
    )
    assert 'phase8_pr_review="s2"' in body, (
        "Pin counter attribute not preserved through the meter → reader → "
        "exposition path.  Either the global REGISTRY is wired to a "
        "different reader than the one /metrics reads, or attribute "
        f"serialisation has regressed.  Body head: {body[:500]!r}"
    )
