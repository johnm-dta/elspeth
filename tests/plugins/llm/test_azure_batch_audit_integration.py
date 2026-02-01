"""Integration tests for Azure Batch LLM audit trail recording.

Tests that LLM calls recorded against batch states are visible via explain().
This validates the fix for P2-aggregation-metadata-hardcoded which ensures
LLM calls are recorded against the batch's node_state rather than individual
row states.
"""

from __future__ import annotations

import json

import pytest

from elspeth.contracts import (
    CallStatus,
    CallType,
    NodeStateStatus,
    NodeType,
    RowOutcome,
)
from elspeth.contracts.schema import SchemaConfig
from elspeth.core.landscape import LandscapeRecorder
from elspeth.core.landscape.lineage import explain

DYNAMIC_SCHEMA = SchemaConfig.from_dict({"fields": "dynamic"})


@pytest.mark.integration
def test_llm_calls_visible_in_explain(real_landscape_db) -> None:
    """Verify LLM calls recorded against batch state are visible via explain()."""
    recorder = LandscapeRecorder(real_landscape_db)

    # Create a run
    run = recorder.begin_run(
        config={"pipeline": "test-batch-audit"},
        canonical_version="v1",
    )

    # Register nodes
    source_node = recorder.register_node(
        run_id=run.run_id,
        plugin_name="csv_source",
        node_type=NodeType.SOURCE,
        plugin_version="1.0",
        config={},
        schema_config=DYNAMIC_SCHEMA,
    )

    batch_node = recorder.register_node(
        run_id=run.run_id,
        plugin_name="azure_batch_llm",
        node_type=NodeType.AGGREGATION,
        plugin_version="1.0",
        config={"deployment_name": "gpt-4o-batch"},
        schema_config=DYNAMIC_SCHEMA,
    )

    # Create source row and token
    row = recorder.create_row(
        run_id=run.run_id,
        source_node_id=source_node.node_id,
        row_index=0,
        data={"text": "Hello world"},
    )

    token = recorder.create_token(row_id=row.row_id)

    # Create node_state for batch operation
    state = recorder.begin_node_state(
        token_id=token.token_id,
        node_id=batch_node.node_id,
        run_id=run.run_id,
        step_index=1,
        input_data={"batch_rows": [{"text": "Hello world"}]},
    )

    # Record LLM call against the batch state (this is what the fix does)
    recorder.record_call(
        state_id=state.state_id,
        call_index=0,
        call_type=CallType.LLM,
        status=CallStatus.SUCCESS,
        request_data={
            "custom_id": "row-0-abc",
            "messages": [{"role": "user", "content": "Analyze: Hello world"}],
            "model": "gpt-4o-batch",
        },
        response_data={
            "choices": [{"message": {"content": "Analysis: Greeting"}}],
            "usage": {"total_tokens": 15},
        },
    )

    # Complete state
    recorder.complete_node_state(
        state_id=state.state_id,
        status=NodeStateStatus.COMPLETED,
        output_data={"llm_response": "Analysis: Greeting"},
        duration_ms=100.0,
    )

    # Record token outcome (required for explain to find the token)
    recorder.record_token_outcome(
        run_id=run.run_id,
        token_id=token.token_id,
        outcome=RowOutcome.COMPLETED,
        sink_name="output",
    )

    # Use explain() to retrieve lineage
    lineage = explain(
        recorder=recorder,
        run_id=run.run_id,
        token_id=token.token_id,
    )

    # Verify lineage has the batch state
    assert lineage is not None
    assert len(lineage.node_states) >= 1

    # Find the batch state
    batch_state = next(
        (s for s in lineage.node_states if s.node_id == batch_node.node_id),
        None,
    )
    assert batch_state is not None, f"Batch state not found. States: {[s.node_id for s in lineage.node_states]}"

    # Verify calls are included in the lineage
    assert len(lineage.calls) >= 1, f"Expected at least 1 call, got {len(lineage.calls)}"

    # Find LLM call
    llm_call = next(
        (c for c in lineage.calls if c.call_type == CallType.LLM),
        None,
    )
    assert llm_call is not None, f"LLM call not found. Calls: {[c.call_type for c in lineage.calls]}"
    # The Call object stores hashes, not raw data (audit trail design)
    assert llm_call.request_hash is not None
    assert llm_call.response_hash is not None
    assert llm_call.status == CallStatus.SUCCESS


@pytest.mark.integration
def test_multiple_llm_calls_recorded_per_batch(real_landscape_db) -> None:
    """Verify multiple LLM calls in a batch are all visible via explain()."""
    recorder = LandscapeRecorder(real_landscape_db)

    run = recorder.begin_run(
        config={"pipeline": "test-multi-call-batch"},
        canonical_version="v1",
    )

    source_node = recorder.register_node(
        run_id=run.run_id,
        plugin_name="csv_source",
        node_type=NodeType.SOURCE,
        plugin_version="1.0",
        config={},
        schema_config=DYNAMIC_SCHEMA,
    )

    batch_node = recorder.register_node(
        run_id=run.run_id,
        plugin_name="azure_batch_llm",
        node_type=NodeType.AGGREGATION,
        plugin_version="1.0",
        config={"deployment_name": "gpt-4o-batch"},
        schema_config=DYNAMIC_SCHEMA,
    )

    # Create multiple rows in the batch
    rows = []
    tokens = []
    for i in range(3):
        row = recorder.create_row(
            run_id=run.run_id,
            source_node_id=source_node.node_id,
            row_index=i,
            data={"text": f"Message {i}"},
        )
        rows.append(row)
        token = recorder.create_token(row_id=row.row_id)
        tokens.append(token)

    # Create a batch state for the first token (batch processes multiple rows)
    state = recorder.begin_node_state(
        token_id=tokens[0].token_id,
        node_id=batch_node.node_id,
        run_id=run.run_id,
        step_index=1,
        input_data={"batch_size": 3},
    )

    # Record LLM calls for each row in the batch
    for i in range(3):
        recorder.record_call(
            state_id=state.state_id,
            call_index=i,
            call_type=CallType.LLM,
            status=CallStatus.SUCCESS,
            request_data={
                "custom_id": f"row-{i}-xyz",
                "messages": [{"role": "user", "content": f"Analyze: Message {i}"}],
                "model": "gpt-4o-batch",
            },
            response_data={
                "choices": [{"message": {"content": f"Analysis: Message {i}"}}],
                "usage": {"total_tokens": 10 + i},
            },
        )

    recorder.complete_node_state(
        state_id=state.state_id,
        status=NodeStateStatus.COMPLETED,
        output_data={"batch_results": ["result0", "result1", "result2"]},
        duration_ms=500.0,
    )

    recorder.record_token_outcome(
        run_id=run.run_id,
        token_id=tokens[0].token_id,
        outcome=RowOutcome.COMPLETED,
        sink_name="output",
    )

    # Verify all calls are visible
    lineage = explain(
        recorder=recorder,
        run_id=run.run_id,
        token_id=tokens[0].token_id,
    )

    assert lineage is not None

    # All 3 LLM calls should be present
    llm_calls = [c for c in lineage.calls if c.call_type == CallType.LLM]
    assert len(llm_calls) == 3, f"Expected 3 LLM calls, got {len(llm_calls)}"

    # Verify call indices are correct
    call_indices = sorted(c.call_index for c in llm_calls)
    assert call_indices == [0, 1, 2], f"Expected call indices [0, 1, 2], got {call_indices}"

    # Each call should have request and response hashes recorded
    for call in llm_calls:
        assert call.request_hash is not None
        assert call.response_hash is not None
        assert call.status == CallStatus.SUCCESS


@pytest.mark.integration
def test_failed_llm_call_recorded(real_landscape_db) -> None:
    """Verify failed LLM calls are recorded with error details."""
    recorder = LandscapeRecorder(real_landscape_db)

    run = recorder.begin_run(
        config={"pipeline": "test-failed-call"},
        canonical_version="v1",
    )

    source_node = recorder.register_node(
        run_id=run.run_id,
        plugin_name="csv_source",
        node_type=NodeType.SOURCE,
        plugin_version="1.0",
        config={},
        schema_config=DYNAMIC_SCHEMA,
    )

    batch_node = recorder.register_node(
        run_id=run.run_id,
        plugin_name="azure_batch_llm",
        node_type=NodeType.AGGREGATION,
        plugin_version="1.0",
        config={"deployment_name": "gpt-4o-batch"},
        schema_config=DYNAMIC_SCHEMA,
    )

    row = recorder.create_row(
        run_id=run.run_id,
        source_node_id=source_node.node_id,
        row_index=0,
        data={"text": "Content that causes error"},
    )

    token = recorder.create_token(row_id=row.row_id)

    state = recorder.begin_node_state(
        token_id=token.token_id,
        node_id=batch_node.node_id,
        run_id=run.run_id,
        step_index=1,
        input_data={"batch_rows": [{"text": "Content that causes error"}]},
    )

    # Record a failed LLM call
    recorder.record_call(
        state_id=state.state_id,
        call_index=0,
        call_type=CallType.LLM,
        status=CallStatus.ERROR,
        request_data={
            "custom_id": "row-0-err",
            "messages": [{"role": "user", "content": "Problematic content"}],
            "model": "gpt-4o-batch",
        },
        response_data=None,
        error={
            "code": "content_filter",
            "message": "Content was blocked by safety filter",
        },
    )

    recorder.complete_node_state(
        state_id=state.state_id,
        status=NodeStateStatus.FAILED,
        output_data=None,
        duration_ms=50.0,
        error={"exception": "LLM call failed", "type": "ContentFilterError"},
    )

    recorder.record_token_outcome(
        run_id=run.run_id,
        token_id=token.token_id,
        outcome=RowOutcome.FAILED,
        error_hash="abc123",
    )

    lineage = explain(
        recorder=recorder,
        run_id=run.run_id,
        token_id=token.token_id,
    )

    assert lineage is not None

    # Find the failed LLM call
    failed_calls = [c for c in lineage.calls if c.status == CallStatus.ERROR]
    assert len(failed_calls) == 1, f"Expected 1 failed call, got {len(failed_calls)}"

    failed_call = failed_calls[0]
    assert failed_call.call_type == CallType.LLM
    assert failed_call.error_json is not None
    # Error is stored as JSON string in error_json field
    error = json.loads(failed_call.error_json)
    assert error["code"] == "content_filter"


@pytest.mark.integration
def test_missing_result_gets_call_record(real_landscape_db) -> None:
    """Verify rows missing from Azure batch output still get Call records.

    Regression test for P2-2026-01-31-azure-batch-missing-call-records:
    When a row's result is missing from Azure batch output, we must still
    record an LLM Call to maintain audit trail completeness.

    Per CLAUDE.md: "External calls - Full request AND response recorded"
    A missing response is still a response (absence of data is data).
    """
    recorder = LandscapeRecorder(real_landscape_db)

    run = recorder.begin_run(
        config={"pipeline": "test-missing-result"},
        canonical_version="v1",
    )

    source_node = recorder.register_node(
        run_id=run.run_id,
        plugin_name="csv_source",
        node_type=NodeType.SOURCE,
        plugin_version="1.0",
        config={},
        schema_config=DYNAMIC_SCHEMA,
    )

    batch_node = recorder.register_node(
        run_id=run.run_id,
        plugin_name="azure_batch_llm",
        node_type=NodeType.AGGREGATION,
        plugin_version="1.0",
        config={"deployment_name": "gpt-4o-batch"},
        schema_config=DYNAMIC_SCHEMA,
    )

    row = recorder.create_row(
        run_id=run.run_id,
        source_node_id=source_node.node_id,
        row_index=0,
        data={"text": "Row that Azure didn't return"},
    )

    token = recorder.create_token(row_id=row.row_id)

    state = recorder.begin_node_state(
        token_id=token.token_id,
        node_id=batch_node.node_id,
        run_id=run.run_id,
        step_index=1,
        input_data={"batch_rows": [{"text": "Row that Azure didn't return"}]},
    )

    # Record a "result_not_found" LLM call - this is what the fix adds
    recorder.record_call(
        state_id=state.state_id,
        call_index=0,
        call_type=CallType.LLM,
        status=CallStatus.ERROR,
        request_data={
            "custom_id": "row-0-missing",
            "row_index": 0,
            "messages": [{"role": "user", "content": "Analyze: Row that Azure didn't return"}],
            "model": "gpt-4o-batch",
        },
        response_data=None,
        error={
            "reason": "result_not_found",
            "custom_id": "row-0-missing",
        },
    )

    recorder.complete_node_state(
        state_id=state.state_id,
        status=NodeStateStatus.FAILED,
        output_data=None,
        duration_ms=100.0,
        error={"reason": "result_not_found"},
    )

    recorder.record_token_outcome(
        run_id=run.run_id,
        token_id=token.token_id,
        outcome=RowOutcome.FAILED,
        error_hash="missing-result",
    )

    lineage = explain(
        recorder=recorder,
        run_id=run.run_id,
        token_id=token.token_id,
    )

    assert lineage is not None

    # Verify Call record exists for the missing result
    error_calls = [c for c in lineage.calls if c.status == CallStatus.ERROR]
    assert len(error_calls) == 1, f"Expected 1 error call, got {len(error_calls)}"

    error_call = error_calls[0]
    assert error_call.call_type == CallType.LLM
    assert error_call.request_hash is not None  # Request was recorded
    assert error_call.error_json is not None  # Error was recorded

    # Verify error contains "result_not_found"
    error = json.loads(error_call.error_json)
    assert error["reason"] == "result_not_found"
