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

    def test_span_names_stable(self) -> None:
        """Span names should not include variable IDs."""
        pytest.importorskip("opentelemetry")
        from opentelemetry.sdk.trace import TracerProvider

        from elspeth.engine.spans import SpanFactory

        provider = TracerProvider()
        tracer = provider.get_tracer("test")
        factory = SpanFactory(tracer=tracer)

        with factory.run_span("run-001") as span:
            assert span.name == "run"

        with factory.row_span("row-001", "token-001") as span:
            assert span.name == "row"

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


class TestTokenIdOnChildSpans:
    """Test that child spans (transform, gate, sink) carry correct token.id.

    BUG: P2-2026-01-21 - row_span is created with parent token_id, but child
    tokens from fork/deaggregation have different token_ids. The child spans
    (transform_span, gate_span, sink_span) need to record the actual token_id
    being processed, not inherit from parent row_span.
    """

    def test_transform_span_includes_token_id(self) -> None:
        """Transform span should include token.id when provided."""
        pytest.importorskip("opentelemetry")
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

        from elspeth.engine.spans import SpanFactory

        # Set up in-memory exporter to capture spans
        exporter = InMemorySpanExporter()
        provider = TracerProvider()
        provider.add_span_processor(
            __import__("opentelemetry.sdk.trace.export", fromlist=["SimpleSpanProcessor"]).SimpleSpanProcessor(exporter)
        )
        tracer = provider.get_tracer("test")
        factory = SpanFactory(tracer=tracer)

        # Create transform span WITH token_id
        with factory.transform_span("my_transform", token_id="child-token-001"):
            pass

        spans = exporter.get_finished_spans()
        assert len(spans) == 1
        transform_span = spans[0]

        # Verify token.id attribute is set
        attrs = dict(transform_span.attributes or {})
        assert attrs.get("token.id") == "child-token-001"

    def test_gate_span_includes_token_id(self) -> None:
        """Gate span should include token.id when provided."""
        pytest.importorskip("opentelemetry")
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

        from elspeth.engine.spans import SpanFactory

        exporter = InMemorySpanExporter()
        provider = TracerProvider()
        provider.add_span_processor(
            __import__("opentelemetry.sdk.trace.export", fromlist=["SimpleSpanProcessor"]).SimpleSpanProcessor(exporter)
        )
        tracer = provider.get_tracer("test")
        factory = SpanFactory(tracer=tracer)

        with factory.gate_span("my_gate", token_id="child-token-002"):
            pass

        spans = exporter.get_finished_spans()
        assert len(spans) == 1
        gate_span = spans[0]

        attrs = dict(gate_span.attributes or {})
        assert attrs.get("token.id") == "child-token-002"

    def test_sink_span_includes_token_ids(self) -> None:
        """Sink span should include token.ids (list) when provided."""
        pytest.importorskip("opentelemetry")
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

        from elspeth.engine.spans import SpanFactory

        exporter = InMemorySpanExporter()
        provider = TracerProvider()
        provider.add_span_processor(
            __import__("opentelemetry.sdk.trace.export", fromlist=["SimpleSpanProcessor"]).SimpleSpanProcessor(exporter)
        )
        tracer = provider.get_tracer("test")
        factory = SpanFactory(tracer=tracer)

        # Sink writes multiple tokens at once
        with factory.sink_span("my_sink", token_ids=["token-001", "token-002", "token-003"]):
            pass

        spans = exporter.get_finished_spans()
        assert len(spans) == 1
        sink_span = spans[0]

        attrs = dict(sink_span.attributes or {})
        # token.ids should be a tuple (OpenTelemetry converts lists to tuples)
        assert attrs.get("token.ids") == ("token-001", "token-002", "token-003")

    def test_child_token_has_different_id_than_parent_row_span(self) -> None:
        """Simulate fork scenario: parent row_span has parent token, child transform uses child token.

        This test documents the architectural fix: row_span keeps parent token.id,
        but transform_span/gate_span get the actual child token.id being processed.
        """
        pytest.importorskip("opentelemetry")
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

        from elspeth.engine.spans import SpanFactory

        exporter = InMemorySpanExporter()
        provider = TracerProvider()
        provider.add_span_processor(
            __import__("opentelemetry.sdk.trace.export", fromlist=["SimpleSpanProcessor"]).SimpleSpanProcessor(exporter)
        )
        tracer = provider.get_tracer("test")
        factory = SpanFactory(tracer=tracer)

        # Simulate: parent token enters, forks into 2 child tokens
        parent_token_id = "parent-001"
        child_token_ids = ["child-001", "child-002"]

        with factory.row_span("row-001", parent_token_id):
            # Process parent through transform (before fork)
            with factory.transform_span("pre_fork_transform", token_id=parent_token_id):
                pass

            # Fork happens, now process child tokens
            for child_id in child_token_ids:
                with factory.transform_span("post_fork_transform", token_id=child_id):
                    pass

        spans = exporter.get_finished_spans()
        # 1 row_span + 1 pre_fork + 2 post_fork = 4 spans
        assert len(spans) == 4

        # Find spans by name
        row_spans = [s for s in spans if s.name == "row"]
        pre_fork_spans = [s for s in spans if s.name == "transform:pre_fork_transform"]
        post_fork_spans = [s for s in spans if s.name == "transform:post_fork_transform"]

        assert len(row_spans) == 1
        assert len(pre_fork_spans) == 1
        assert len(post_fork_spans) == 2

        # Row span has parent token.id
        row_attrs = dict(row_spans[0].attributes or {})
        assert row_attrs.get("token.id") == parent_token_id

        # Pre-fork transform has parent token.id
        pre_fork_attrs = dict(pre_fork_spans[0].attributes or {})
        assert pre_fork_attrs.get("token.id") == parent_token_id

        # Post-fork transforms have CHILD token.ids (not parent!)
        post_fork_token_ids = {dict(s.attributes or {}).get("token.id") for s in post_fork_spans}
        assert post_fork_token_ids == set(child_token_ids)

    def test_transform_span_without_token_id_still_works(self) -> None:
        """Backwards compatibility: transform_span without token_id should work."""
        pytest.importorskip("opentelemetry")
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

        from elspeth.engine.spans import SpanFactory

        exporter = InMemorySpanExporter()
        provider = TracerProvider()
        provider.add_span_processor(
            __import__("opentelemetry.sdk.trace.export", fromlist=["SimpleSpanProcessor"]).SimpleSpanProcessor(exporter)
        )
        tracer = provider.get_tracer("test")
        factory = SpanFactory(tracer=tracer)

        # No token_id provided - should still work
        with factory.transform_span("my_transform"):
            pass

        spans = exporter.get_finished_spans()
        assert len(spans) == 1
        attrs = dict(spans[0].attributes or {})
        # token.id should NOT be present
        assert "token.id" not in attrs


class TestTokenIdEdgeCases:
    """Edge case tests for token_id/token_ids parameters.

    Tests for:
    - Explicit None vs missing parameter
    - Empty sequences
    - Batch transform token_ids (multiple tokens)
    """

    def test_transform_span_with_explicit_none_omits_attribute(self) -> None:
        """token_id=None should omit attribute (not set it to 'None' string)."""
        pytest.importorskip("opentelemetry")
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

        from elspeth.engine.spans import SpanFactory

        exporter = InMemorySpanExporter()
        provider = TracerProvider()
        provider.add_span_processor(
            __import__("opentelemetry.sdk.trace.export", fromlist=["SimpleSpanProcessor"]).SimpleSpanProcessor(exporter)
        )
        tracer = provider.get_tracer("test")
        factory = SpanFactory(tracer=tracer)

        with factory.transform_span("my_transform", token_id=None):
            pass

        spans = exporter.get_finished_spans()
        assert len(spans) == 1
        attrs = dict(spans[0].attributes or {})
        # Explicit None should NOT create attribute
        assert "token.id" not in attrs
        assert "token.ids" not in attrs

    def test_sink_span_with_empty_token_ids_records_empty_tuple(self) -> None:
        """token_ids=[] should record empty tuple (distinguishes 'zero tokens' from 'not tracked').

        Design decision: Explicit empty list creates attribute with empty tuple.
        This distinguishes:
        - token_ids=None → attribute omitted → "token tracking not used"
        - token_ids=[] → attribute is () → "batch had zero tokens"

        This is semantically correct for audit: an explicitly provided empty batch
        is different from "we didn't track tokens at all".
        """
        pytest.importorskip("opentelemetry")
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

        from elspeth.engine.spans import SpanFactory

        exporter = InMemorySpanExporter()
        provider = TracerProvider()
        provider.add_span_processor(
            __import__("opentelemetry.sdk.trace.export", fromlist=["SimpleSpanProcessor"]).SimpleSpanProcessor(exporter)
        )
        tracer = provider.get_tracer("test")
        factory = SpanFactory(tracer=tracer)

        # Empty list - should create attribute with empty tuple (explicit is recorded)
        with factory.sink_span("my_sink", token_ids=[]):
            pass

        spans = exporter.get_finished_spans()
        assert len(spans) == 1
        attrs = dict(spans[0].attributes or {})
        # Empty sequence should create attribute with empty tuple
        assert attrs.get("token.ids") == ()

    def test_transform_span_with_token_ids_for_batch(self) -> None:
        """Batch transforms should use token_ids (plural) not token_id."""
        pytest.importorskip("opentelemetry")
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

        from elspeth.engine.spans import SpanFactory

        exporter = InMemorySpanExporter()
        provider = TracerProvider()
        provider.add_span_processor(
            __import__("opentelemetry.sdk.trace.export", fromlist=["SimpleSpanProcessor"]).SimpleSpanProcessor(exporter)
        )
        tracer = provider.get_tracer("test")
        factory = SpanFactory(tracer=tracer)

        batch_tokens = ["batch-token-001", "batch-token-002", "batch-token-003"]
        with factory.transform_span("batch_transform", token_ids=batch_tokens):
            pass

        spans = exporter.get_finished_spans()
        assert len(spans) == 1
        attrs = dict(spans[0].attributes or {})
        # Should have token.ids (plural), not token.id (singular)
        assert "token.id" not in attrs
        assert attrs.get("token.ids") == tuple(batch_tokens)

    def test_transform_span_token_ids_takes_precedence_over_token_id(self) -> None:
        """If both token_id and token_ids are provided, token_ids takes precedence."""
        pytest.importorskip("opentelemetry")
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

        from elspeth.engine.spans import SpanFactory

        exporter = InMemorySpanExporter()
        provider = TracerProvider()
        provider.add_span_processor(
            __import__("opentelemetry.sdk.trace.export", fromlist=["SimpleSpanProcessor"]).SimpleSpanProcessor(exporter)
        )
        tracer = provider.get_tracer("test")
        factory = SpanFactory(tracer=tracer)

        # Provide both - token_ids should win
        with factory.transform_span(
            "mixed_transform",
            token_id="single-token",
            token_ids=["batch-1", "batch-2"],
        ):
            pass

        spans = exporter.get_finished_spans()
        assert len(spans) == 1
        attrs = dict(spans[0].attributes or {})
        # token_ids takes precedence
        assert "token.id" not in attrs
        assert attrs.get("token.ids") == ("batch-1", "batch-2")

    def test_gate_span_with_explicit_none_omits_attribute(self) -> None:
        """gate_span with token_id=None should omit attribute."""
        pytest.importorskip("opentelemetry")
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

        from elspeth.engine.spans import SpanFactory

        exporter = InMemorySpanExporter()
        provider = TracerProvider()
        provider.add_span_processor(
            __import__("opentelemetry.sdk.trace.export", fromlist=["SimpleSpanProcessor"]).SimpleSpanProcessor(exporter)
        )
        tracer = provider.get_tracer("test")
        factory = SpanFactory(tracer=tracer)

        with factory.gate_span("my_gate", token_id=None):
            pass

        spans = exporter.get_finished_spans()
        assert len(spans) == 1
        attrs = dict(spans[0].attributes or {})
        assert "token.id" not in attrs

    def test_aggregation_span_with_token_ids(self) -> None:
        """Aggregation span should support token_ids for batch tracking."""
        pytest.importorskip("opentelemetry")
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

        from elspeth.engine.spans import SpanFactory

        exporter = InMemorySpanExporter()
        provider = TracerProvider()
        provider.add_span_processor(
            __import__("opentelemetry.sdk.trace.export", fromlist=["SimpleSpanProcessor"]).SimpleSpanProcessor(exporter)
        )
        tracer = provider.get_tracer("test")
        factory = SpanFactory(tracer=tracer)

        batch_tokens = ["agg-token-001", "agg-token-002"]
        with factory.aggregation_span(
            "batch_agg",
            batch_id="batch-123",
            token_ids=batch_tokens,
        ):
            pass

        spans = exporter.get_finished_spans()
        assert len(spans) == 1
        attrs = dict(spans[0].attributes or {})
        assert attrs.get("batch.id") == "batch-123"
        assert attrs.get("token.ids") == tuple(batch_tokens)


class TestNodeIdOnSpans:
    """Test that spans include node.id for disambiguation.

    BUG: P2-2026-01-21-span-ambiguous-plugin-instances
    When multiple plugin instances of the same type exist (e.g., two FieldMapper
    transforms), spans are indistinguishable because they only use plugin.name.
    Adding node.id allows correlation with Landscape node_states.
    """

    def test_transform_span_includes_node_id(self) -> None:
        """Transform span should include node.id when provided."""
        pytest.importorskip("opentelemetry")
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

        from elspeth.engine.spans import SpanFactory

        exporter = InMemorySpanExporter()
        provider = TracerProvider()
        provider.add_span_processor(
            __import__("opentelemetry.sdk.trace.export", fromlist=["SimpleSpanProcessor"]).SimpleSpanProcessor(exporter)
        )
        tracer = provider.get_tracer("test")
        factory = SpanFactory(tracer=tracer)

        with factory.transform_span(
            "field_mapper",
            node_id="transform_field_mapper_abc123_0",
            token_id="token-001",
        ):
            pass

        spans = exporter.get_finished_spans()
        assert len(spans) == 1
        attrs = dict(spans[0].attributes or {})
        assert attrs.get("node.id") == "transform_field_mapper_abc123_0"

    def test_gate_span_includes_node_id(self) -> None:
        """Gate span should include node.id when provided."""
        pytest.importorskip("opentelemetry")
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

        from elspeth.engine.spans import SpanFactory

        exporter = InMemorySpanExporter()
        provider = TracerProvider()
        provider.add_span_processor(
            __import__("opentelemetry.sdk.trace.export", fromlist=["SimpleSpanProcessor"]).SimpleSpanProcessor(exporter)
        )
        tracer = provider.get_tracer("test")
        factory = SpanFactory(tracer=tracer)

        with factory.gate_span(
            "safety_gate",
            node_id="gate_safety_gate_def456",
            token_id="token-002",
        ):
            pass

        spans = exporter.get_finished_spans()
        assert len(spans) == 1
        attrs = dict(spans[0].attributes or {})
        assert attrs.get("node.id") == "gate_safety_gate_def456"

    def test_sink_span_includes_node_id(self) -> None:
        """Sink span should include node.id when provided."""
        pytest.importorskip("opentelemetry")
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

        from elspeth.engine.spans import SpanFactory

        exporter = InMemorySpanExporter()
        provider = TracerProvider()
        provider.add_span_processor(
            __import__("opentelemetry.sdk.trace.export", fromlist=["SimpleSpanProcessor"]).SimpleSpanProcessor(exporter)
        )
        tracer = provider.get_tracer("test")
        factory = SpanFactory(tracer=tracer)

        with factory.sink_span(
            "csv_sink",
            node_id="sink_output_ghi789",
            token_ids=["token-001", "token-002"],
        ):
            pass

        spans = exporter.get_finished_spans()
        assert len(spans) == 1
        attrs = dict(spans[0].attributes or {})
        assert attrs.get("node.id") == "sink_output_ghi789"

    def test_aggregation_span_includes_node_id(self) -> None:
        """Aggregation span should include node.id when provided."""
        pytest.importorskip("opentelemetry")
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

        from elspeth.engine.spans import SpanFactory

        exporter = InMemorySpanExporter()
        provider = TracerProvider()
        provider.add_span_processor(
            __import__("opentelemetry.sdk.trace.export", fromlist=["SimpleSpanProcessor"]).SimpleSpanProcessor(exporter)
        )
        tracer = provider.get_tracer("test")
        factory = SpanFactory(tracer=tracer)

        with factory.aggregation_span(
            "batch_stats",
            node_id="aggregation_batch_stats_jkl012",
            batch_id="batch-001",
            token_ids=["token-001", "token-002"],
        ):
            pass

        spans = exporter.get_finished_spans()
        assert len(spans) == 1
        attrs = dict(spans[0].attributes or {})
        assert attrs.get("node.id") == "aggregation_batch_stats_jkl012"
        # Also verify batch.id is still present
        assert attrs.get("batch.id") == "batch-001"

    def test_aggregation_span_includes_input_hash(self) -> None:
        """Aggregation span should include input.hash when provided.

        BUG: P3-2026-02-01-aggregation-flush-span-missing-input-hash
        Aggregation flushes compute input_hash for audit correlation, but
        aggregation_span() was created without input_hash parameter when
        migrating from transform_span(). This breaks trace-to-audit correlation.
        """
        pytest.importorskip("opentelemetry")
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

        from elspeth.engine.spans import SpanFactory

        exporter = InMemorySpanExporter()
        provider = TracerProvider()
        provider.add_span_processor(
            __import__("opentelemetry.sdk.trace.export", fromlist=["SimpleSpanProcessor"]).SimpleSpanProcessor(exporter)
        )
        tracer = provider.get_tracer("test")
        factory = SpanFactory(tracer=tracer)

        with factory.aggregation_span(
            "batch_stats",
            node_id="aggregation_batch_stats_abc123",
            batch_id="batch-001",
            token_ids=["token-001", "token-002"],
            input_hash="sha256:deadbeef1234567890",
        ):
            pass

        spans = exporter.get_finished_spans()
        assert len(spans) == 1
        attrs = dict(spans[0].attributes or {})
        # Core assertion: input.hash must be present for trace-to-audit correlation
        assert attrs.get("input.hash") == "sha256:deadbeef1234567890"
        # Verify other attributes still work
        assert attrs.get("node.id") == "aggregation_batch_stats_abc123"
        assert attrs.get("batch.id") == "batch-001"
        assert attrs.get("token.ids") == ("token-001", "token-002")

    def test_duplicate_plugins_distinguishable_by_node_id(self) -> None:
        """Two instances of same plugin type should have different node.id."""
        pytest.importorskip("opentelemetry")
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

        from elspeth.engine.spans import SpanFactory

        exporter = InMemorySpanExporter()
        provider = TracerProvider()
        provider.add_span_processor(
            __import__("opentelemetry.sdk.trace.export", fromlist=["SimpleSpanProcessor"]).SimpleSpanProcessor(exporter)
        )
        tracer = provider.get_tracer("test")
        factory = SpanFactory(tracer=tracer)

        # Simulate two FieldMapper transforms with different configs
        with factory.transform_span(
            "field_mapper",
            node_id="transform_field_mapper_abc123_0",
            token_id="token-001",
        ):
            pass

        with factory.transform_span(
            "field_mapper",
            node_id="transform_field_mapper_def456_1",
            token_id="token-002",
        ):
            pass

        spans = exporter.get_finished_spans()
        assert len(spans) == 2

        # Both have same plugin.name
        assert spans[0].attributes.get("plugin.name") == "field_mapper"
        assert spans[1].attributes.get("plugin.name") == "field_mapper"

        # But different node.id - NOW DISTINGUISHABLE!
        node_ids = {s.attributes.get("node.id") for s in spans}
        assert node_ids == {
            "transform_field_mapper_abc123_0",
            "transform_field_mapper_def456_1",
        }

    def test_node_id_none_omits_attribute(self) -> None:
        """node_id=None should omit the attribute (backwards compatible)."""
        pytest.importorskip("opentelemetry")
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

        from elspeth.engine.spans import SpanFactory

        exporter = InMemorySpanExporter()
        provider = TracerProvider()
        provider.add_span_processor(
            __import__("opentelemetry.sdk.trace.export", fromlist=["SimpleSpanProcessor"]).SimpleSpanProcessor(exporter)
        )
        tracer = provider.get_tracer("test")
        factory = SpanFactory(tracer=tracer)

        # No node_id provided
        with factory.transform_span("field_mapper", token_id="token-001"):
            pass

        spans = exporter.get_finished_spans()
        assert len(spans) == 1
        attrs = dict(spans[0].attributes or {})
        # node.id should NOT be present when not provided
        assert "node.id" not in attrs
