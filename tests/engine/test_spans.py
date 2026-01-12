# tests/engine/test_spans.py
"""Tests for OpenTelemetry span factory."""

import pytest


class TestSpanFactory:
    """OpenTelemetry span creation."""

    def test_create_run_span(self) -> None:
        from elspeth.engine.spans import NoOpSpan, SpanFactory

        factory = SpanFactory()  # No tracer = no-op mode

        with factory.run_span("run-001") as span:
            # No-op mode returns NoOpSpan (not None) so callers can use uniform interface
            assert isinstance(span, NoOpSpan)

    def test_create_row_span(self) -> None:
        from elspeth.engine.spans import NoOpSpan, SpanFactory

        factory = SpanFactory()

        with factory.row_span("row-001", "token-001") as span:
            assert isinstance(span, NoOpSpan)

    def test_create_transform_span(self) -> None:
        from elspeth.engine.spans import NoOpSpan, SpanFactory

        factory = SpanFactory()

        with factory.transform_span("my_transform", input_hash="abc123") as span:
            assert isinstance(span, NoOpSpan)

    def test_with_tracer(self) -> None:
        """Test with actual tracer if opentelemetry available."""
        pytest.importorskip("opentelemetry")
        from opentelemetry.sdk.trace import TracerProvider

        from elspeth.engine.spans import SpanFactory

        # Create provider locally - do NOT set global state with set_tracer_provider()
        # Global state causes flaky tests when other tests/libraries have already set a provider
        provider = TracerProvider()
        tracer = provider.get_tracer("test")  # Get tracer from local provider

        factory = SpanFactory(tracer=tracer)

        with factory.run_span("run-001") as span:
            assert span is not None
            assert span.is_recording()

    def test_noop_span_interface(self) -> None:
        """Test that NoOpSpan has the same interface as real spans."""
        from elspeth.engine.spans import NoOpSpan, SpanFactory

        # NoOpSpan should be usable in place of real spans
        noop = NoOpSpan()
        noop.set_attribute("key", "value")  # Should not raise
        noop.set_status(None)  # Should not raise
        noop.record_exception(ValueError("test"))  # Should not raise
        assert noop.is_recording() is False

        # Factory in no-op mode should return NoOpSpan, not None
        factory = SpanFactory()  # No tracer = no-op mode
        with factory.run_span("run-001") as span:
            assert isinstance(span, NoOpSpan)


class TestSpanFactoryEnabled:
    """Test SpanFactory.enabled property."""

    def test_enabled_false_without_tracer(self) -> None:
        from elspeth.engine.spans import SpanFactory

        factory = SpanFactory()
        assert factory.enabled is False

    def test_enabled_true_with_tracer(self) -> None:
        pytest.importorskip("opentelemetry")
        from opentelemetry.sdk.trace import TracerProvider

        from elspeth.engine.spans import SpanFactory

        provider = TracerProvider()
        tracer = provider.get_tracer("test")
        factory = SpanFactory(tracer=tracer)
        assert factory.enabled is True


class TestSourceSpan:
    """Test source span creation."""

    def test_source_span_noop(self) -> None:
        from elspeth.engine.spans import NoOpSpan, SpanFactory

        factory = SpanFactory()
        with factory.source_span("csv_source") as span:
            assert isinstance(span, NoOpSpan)

    def test_source_span_with_tracer(self) -> None:
        pytest.importorskip("opentelemetry")
        from opentelemetry.sdk.trace import TracerProvider

        from elspeth.engine.spans import SpanFactory

        provider = TracerProvider()
        tracer = provider.get_tracer("test")
        factory = SpanFactory(tracer=tracer)

        with factory.source_span("csv_source") as span:
            assert span.is_recording()


class TestGateSpan:
    """Test gate span creation."""

    def test_gate_span_noop(self) -> None:
        from elspeth.engine.spans import NoOpSpan, SpanFactory

        factory = SpanFactory()
        with factory.gate_span("my_gate", input_hash="xyz789") as span:
            assert isinstance(span, NoOpSpan)

    def test_gate_span_with_tracer(self) -> None:
        pytest.importorskip("opentelemetry")
        from opentelemetry.sdk.trace import TracerProvider

        from elspeth.engine.spans import SpanFactory

        provider = TracerProvider()
        tracer = provider.get_tracer("test")
        factory = SpanFactory(tracer=tracer)

        with factory.gate_span("my_gate") as span:
            assert span.is_recording()


class TestAggregationSpan:
    """Test aggregation span creation."""

    def test_aggregation_span_noop(self) -> None:
        from elspeth.engine.spans import NoOpSpan, SpanFactory

        factory = SpanFactory()
        with factory.aggregation_span("batch_agg", batch_id="batch-001") as span:
            assert isinstance(span, NoOpSpan)

    def test_aggregation_span_with_tracer(self) -> None:
        pytest.importorskip("opentelemetry")
        from opentelemetry.sdk.trace import TracerProvider

        from elspeth.engine.spans import SpanFactory

        provider = TracerProvider()
        tracer = provider.get_tracer("test")
        factory = SpanFactory(tracer=tracer)

        with factory.aggregation_span("batch_agg") as span:
            assert span.is_recording()


class TestSinkSpan:
    """Test sink span creation."""

    def test_sink_span_noop(self) -> None:
        from elspeth.engine.spans import NoOpSpan, SpanFactory

        factory = SpanFactory()
        with factory.sink_span("csv_sink") as span:
            assert isinstance(span, NoOpSpan)

    def test_sink_span_with_tracer(self) -> None:
        pytest.importorskip("opentelemetry")
        from opentelemetry.sdk.trace import TracerProvider

        from elspeth.engine.spans import SpanFactory

        provider = TracerProvider()
        tracer = provider.get_tracer("test")
        factory = SpanFactory(tracer=tracer)

        with factory.sink_span("csv_sink") as span:
            assert span.is_recording()


class TestNestedSpans:
    """Test span nesting works correctly."""

    def test_nested_spans_noop(self) -> None:
        from elspeth.engine.spans import NoOpSpan, SpanFactory

        factory = SpanFactory()

        with factory.run_span("run-001") as run:
            assert isinstance(run, NoOpSpan)
            with factory.row_span("row-001", "token-001") as row:
                assert isinstance(row, NoOpSpan)
                with factory.transform_span("transform_1") as transform:
                    assert isinstance(transform, NoOpSpan)

    def test_nested_spans_with_tracer(self) -> None:
        pytest.importorskip("opentelemetry")
        from opentelemetry.sdk.trace import TracerProvider

        from elspeth.engine.spans import SpanFactory

        provider = TracerProvider()
        tracer = provider.get_tracer("test")
        factory = SpanFactory(tracer=tracer)

        with factory.run_span("run-001") as run:
            assert run.is_recording()
            with factory.row_span("row-001", "token-001") as row:
                assert row.is_recording()
                with factory.transform_span("transform_1") as transform:
                    assert transform.is_recording()
