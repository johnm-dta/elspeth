"""Production-path integration test for ADR-009 §Clause 2 batch-flush cross-check.

Verifies the runtime cross-check fires from
``RowProcessor._process_batch_aggregation_node`` (processor.py:1082, 1192)
when a mis-annotated batch-aware transform drops fields. Uses the real
production assembly path — ``ExecutionGraph.from_plugin_instances()`` and
``Orchestrator.run()`` — with no mocks of the cross-check or its invocation
site.

This complements ``tests/unit/engine/test_cross_check_flush_output.py``
(which exercises ``_cross_check_flush_output`` in isolation with a mocked
transform). The unit test would still pass if a refactor accidentally
removed the call site at ``processor.py:1082`` or ``:1192``; this
integration test catches that — the cross-check must be reachable from the
production execution path, not just callable in isolation.

Per CLAUDE.md "Critical Implementation Patterns": integration tests MUST
use ``ExecutionGraph.from_plugin_instances()`` rather than constructing
graph state by hand.
"""

from __future__ import annotations

from typing import Any

import pytest

from elspeth.contracts.enums import OutputMode
from elspeth.contracts.errors import PassThroughContractViolation
from elspeth.contracts.schema_contract import (
    PipelineRow,
    SchemaContract,
)
from elspeth.contracts.types import AggregationName
from elspeth.core.config import (
    AggregationSettings,
    SourceSettings,
    TriggerConfig,
)
from elspeth.core.dag import ExecutionGraph
from elspeth.engine.orchestrator import Orchestrator, PipelineConfig
from elspeth.plugins.infrastructure.base import BaseTransform
from elspeth.plugins.infrastructure.results import TransformResult
from tests.fixtures.base_classes import _TestSchema, as_sink, as_source, as_transform
from tests.fixtures.landscape import make_landscape_db
from tests.fixtures.plugins import CollectSink, ListSource

# ---------------------------------------------------------------------------
# Fixture transforms
# ---------------------------------------------------------------------------


def _build_dropped_contract(original: SchemaContract, drop_field: str) -> SchemaContract:
    """Build a contract that omits ``drop_field`` while preserving mode/locked."""
    kept_fields = tuple(fc for fc in original.fields if fc.normalized_name != drop_field)
    return SchemaContract(mode=original.mode, fields=kept_fields, locked=original.locked)


class _MisannotatedBatchDropper(BaseTransform):
    """Batch-aware transform that LIES about ``passes_through_input``.

    Claims pass-through but emits rows with ``to_drop`` stripped from BOTH
    the contract AND the payload. The runtime cross-check at
    ``processor._cross_check_flush_output`` must catch this and raise
    ``PassThroughContractViolation``.

    Why drop from both: ``verify_pass_through`` computes
    ``runtime_observed = contract_fields & payload_fields``. A field
    surviving in either alone would still be a one-sided drop and would
    fire the cross-check, but dropping from both is the unambiguous
    "transform deliberately suppressed this field" case that an honest
    pass-through transform would never produce.
    """

    name = "misannotated-batch-dropper"
    plugin_version = "1.0.0"
    source_file_hash: str | None = None
    input_schema = _TestSchema
    output_schema = _TestSchema
    is_batch_aware = True
    passes_through_input = True  # LIE — process() drops 'to_drop' field.
    # on_success/on_error must be set at instance level — orchestrator's
    # validate_transform_error_sinks treats None as a Tier-1 invariant
    # violation (TransformSettings always requires on_error in production).
    on_success = "output"
    on_error = "discard"

    def __init__(self) -> None:
        super().__init__({"schema": {"mode": "observed"}})

    def process(  # type: ignore[override]
        self, rows: list[PipelineRow], ctx: Any
    ) -> TransformResult:
        if not rows:
            return TransformResult.error({"reason": "empty_batch"}, retryable=False)

        # Build ONE shared dropped-contract instance so success_multi's
        # contract-identity invariant is satisfied (all emitted rows must
        # share contract instance — see TransformResult.success_multi).
        dropped_contract = _build_dropped_contract(rows[0].contract, "to_drop")

        emitted: list[PipelineRow] = []
        for row in rows:
            data = row.to_dict()
            data.pop("to_drop", None)
            emitted.append(PipelineRow(data, dropped_contract))

        return TransformResult.success_multi(
            tuple(emitted),
            success_reason={"action": "drop_to_drop"},
        )


class _HonestBatchPreserver(BaseTransform):
    """Batch-aware transform that honestly preserves all input fields.

    Positive control for the cross-check: an annotated transform that
    actually preserves fields must NOT trigger the violation path.
    """

    name = "honest-batch-preserver"
    plugin_version = "1.0.0"
    source_file_hash: str | None = None
    input_schema = _TestSchema
    output_schema = _TestSchema
    is_batch_aware = True
    passes_through_input = True  # Honest annotation.
    on_success = "output"
    on_error = "discard"

    def __init__(self) -> None:
        super().__init__({"schema": {"mode": "observed"}})

    def process(  # type: ignore[override]
        self, rows: list[PipelineRow], ctx: Any
    ) -> TransformResult:
        if not rows:
            return TransformResult.error({"reason": "empty_batch"}, retryable=False)

        # Pass through every row unchanged (same data, same contract instance).
        # All rows in success_multi must share contract identity; we use the
        # first row's contract since the source emits a single shared contract.
        shared_contract = rows[0].contract
        emitted = tuple(PipelineRow(row.to_dict(), shared_contract) for row in rows)
        return TransformResult.success_multi(
            emitted,
            success_reason={"action": "preserve"},
        )


# ---------------------------------------------------------------------------
# Pipeline assembly helper
# ---------------------------------------------------------------------------


def _build_aggregation_pipeline(
    transform: BaseTransform,
    *,
    output_mode: OutputMode,
    source_data: list[dict[str, Any]],
) -> tuple[PipelineConfig, ExecutionGraph, CollectSink]:
    """Build a ``ListSource → Aggregation(transform) → CollectSink`` pipeline.

    Uses ``ExecutionGraph.from_plugin_instances()`` — the real production
    assembly path. The aggregation count trigger is set to the source row
    count so the flush fires exactly once per run.
    """
    source = ListSource(source_data, name="list_source", on_success="agg_in")
    output_sink = CollectSink("output")

    agg_settings = AggregationSettings(
        name="dropper_agg",
        plugin=transform.name,
        input="agg_in",
        on_success="output",
        on_error="discard",
        trigger=TriggerConfig(count=len(source_data), timeout_seconds=3600),
        output_mode=output_mode,
    )

    graph = ExecutionGraph.from_plugin_instances(
        source=as_source(source),
        source_settings=SourceSettings(plugin=source.name, on_success="agg_in", options={}),
        transforms=[],
        sinks={"output": as_sink(output_sink)},
        aggregations={"dropper_agg": (as_transform(transform), agg_settings)},
        gates=[],
    )

    # Wire the agg node_id back onto the transform — graceful_shutdown's
    # _build_interruptible_aggregation_config does the same thing; the
    # transform instance carries it for runtime identification.
    agg_node_id = graph.get_aggregation_id_map()[AggregationName("dropper_agg")]
    transform.node_id = agg_node_id

    config = PipelineConfig(
        source=as_source(source),
        transforms=[as_transform(transform)],
        sinks={"output": as_sink(output_sink)},
        aggregation_settings={agg_node_id: agg_settings},
    )
    return config, graph, output_sink


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestBatchPassThroughFlushCrossCheck:
    """End-to-end verification that ADR-009 §Clause 2 cross-check is wired in.

    Two scenarios verify both the negative (mis-annotation caught) and
    positive (honest annotation passes) paths through the production
    Orchestrator → RowProcessor → _cross_check_flush_output chain.
    """

    def test_misannotated_transform_violation_propagates_through_orchestrator(self, payload_store) -> None:
        """Mis-annotated batch transform → ``PassThroughContractViolation``.

        Verifies the cross-check at ``processor.py:1082`` (count-triggered
        flush path) is invoked from ``_process_batch_aggregation_node`` and
        the resulting violation propagates as a Tier-1 error all the way
        out of ``Orchestrator.run()`` (no on_error swallowing,
        ``TIER_1_ERRORS`` contains ``PassThroughContractViolation``).

        Failure mode if call site is removed: orchestrator returns
        ``RunStatus.COMPLETED`` instead of raising — silent audit
        corruption, exactly the gap ADR-009 §Clause 2 closes.
        """
        db = make_landscape_db()
        config, graph, output_sink = _build_aggregation_pipeline(
            _MisannotatedBatchDropper(),
            output_mode=OutputMode.TRANSFORM,
            # Two rows with 'to_drop' field — ensures intersection includes it.
            source_data=[{"value": 1, "to_drop": "x"}, {"value": 2, "to_drop": "y"}],
        )

        orchestrator = Orchestrator(db)
        with pytest.raises(PassThroughContractViolation) as exc_info:
            orchestrator.run(
                config,
                graph=graph,
                payload_store=payload_store,
            )

        # Verify the violation identifies the offending transform and field.
        violation = exc_info.value
        assert violation.transform == "misannotated-batch-dropper"
        assert "to_drop" in violation.divergence_set, f"Expected 'to_drop' in divergence_set; got {violation.divergence_set!r}"
        # Sink must not have received the broken outputs — violation fires
        # BEFORE _route_transform_results runs (ADR-009 §2.5 placement).
        assert output_sink.results == [], (
            "Sink received outputs despite cross-check violation — "
            "indicates the cross-check ran AFTER routing, violating "
            "ADR-009 §2.5's BEFORE-emit invariant."
        )

    def test_honest_transform_passes_cross_check_end_to_end(self, payload_store) -> None:
        """Honest pass-through batch transform → run completes successfully.

        Positive control: when ``passes_through_input=True`` is correctly
        annotated, the cross-check at processor.py runs and returns
        without raising, the flush completes, and rows reach the sink.

        This test would fail if the cross-check raised a false-positive on
        legitimate pass-through (e.g., off-by-one in field comparison) —
        regression coverage for the cross-check's correctness, not just
        its presence.
        """
        db = make_landscape_db()
        config, graph, output_sink = _build_aggregation_pipeline(
            _HonestBatchPreserver(),
            output_mode=OutputMode.TRANSFORM,
            source_data=[
                {"value": 1, "preserve": "a"},
                {"value": 2, "preserve": "b"},
            ],
        )

        orchestrator = Orchestrator(db)
        result = orchestrator.run(
            config,
            graph=graph,
            payload_store=payload_store,
        )

        # Run completed cleanly — no Tier-1 violation raised.
        from elspeth.contracts import RunStatus

        assert result.status == RunStatus.COMPLETED
        # Rows reached the sink with all input fields preserved.
        assert len(output_sink.results) == 2
        for written in output_sink.results:
            assert "value" in written
            assert "preserve" in written
