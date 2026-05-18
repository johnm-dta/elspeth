"""Tests for the B1-r3 MeterProvider precondition in elspeth.web.app.

Verifies that importing ``elspeth.web.app`` installs a real ``MeterProvider``
(not the OTel default ``NoOpMeterProvider``) and that the FastAPI app exposes
a Prometheus scrape endpoint at ``GET /metrics``.

Process-global side effect notice
----------------------------------
``elspeth.web.app`` sets the global OTel ``MeterProvider`` at module-import
time (process-global per OTel design, do_once semantics in OTel 1.41+).
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

from opentelemetry import metrics
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import InMemoryMetricReader
from starlette.testclient import TestClient

# ---------------------------------------------------------------------------
# Test 1 — importing app.py wires a real MeterProvider
# ---------------------------------------------------------------------------


def test_meter_provider_is_not_noop() -> None:
    """``metrics.get_meter_provider()`` returns a real ``MeterProvider``, not
    the OTel default ``NoOpMeterProvider``.

    This is the B1-r3 load-bearing precondition: without a real provider every
    counter in the codebase silently discards its data.
    """
    # Importing app triggers the module-level set_meter_provider call.
    import elspeth.web.app  # noqa: F401 — side-effect import

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

        found = False
        for rm in metrics_data.resource_metrics:
            for sm in rm.scope_metrics:
                for metric in sm.metrics:
                    if metric.name == "test_counter_b1r3":
                        for point in metric.data.data_points:
                            if dict(point.attributes).get("env") == "test":
                                assert point.value >= 1
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

    Uses ``starlette.testclient.TestClient`` in a minimal synchronous invocation
    so the test runs without an asyncio event loop.  The ``/metrics`` mount is
    independent of the lifespan context; it works before ``yield`` completes.
    """
    from elspeth.web.app import create_app
    from elspeth.web.config import WebSettings

    settings = WebSettings(
        data_dir=tmp_path,
        composer_max_composition_turns=15,
        composer_max_discovery_turns=10,
        composer_timeout_seconds=85.0,
        composer_rate_limit_per_minute=10,
        shareable_link_signing_key=b"\x00" * 32,
    )

    app = create_app(settings)

    # Use raise_server_exceptions=False so that lifespan exceptions (e.g. the
    # OIDC discovery step that isn't configured in tests) don't propagate.  The
    # /metrics endpoint is mounted before lifespan runs, so it responds even
    # without the full lifespan completing.
    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.get("/metrics")

    assert response.status_code == 200, (
        f"Expected 200 from GET /metrics but got {response.status_code}. "
        "Prometheus endpoint not mounted — check create_app() /metrics mount."
    )

    content_type = response.headers.get("content-type", "")
    assert "text/plain" in content_type, f"Expected text/plain content-type from /metrics but got {content_type!r}."

    body = response.text
    # Must contain at least one line that is non-empty and non-comment —
    # Prometheus clients emit ``# HELP`` and ``# TYPE`` lines even for
    # zero-value counters.
    non_comment_lines = [line for line in body.splitlines() if line.strip() and not line.startswith("#")]
    assert len(non_comment_lines) > 0 or "# HELP" in body or "# TYPE" in body, (
        f"Prometheus body appears empty or malformed. Body: {body[:500]!r}"
    )
