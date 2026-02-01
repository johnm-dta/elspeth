# Azure Batch LLM Audit Trail Fix Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Fix BUG-AZURE-01 (missing LLM payloads) and BUG-AZURE-02 (synthetic state_ids) by recording per-row LLM calls against the existing batch node_state.

**Architecture:** The batch operation has ONE node_state (created by AggregationExecutor). We record N LLM calls against that state using different `call_index` values. This uses existing schema correctly without creating parallel audit mechanisms.

**Tech Stack:** Python, SQLAlchemy (existing schema), Azure OpenAI Batch API

**Bugs Fixed:**
- BUG-AZURE-01: Batch mode doesn't record LLM payloads in audit trail
- BUG-AZURE-02: Batch mode uses synthetic state_ids breaking FK integrity

---

## Architectural Context

### Why the Original Plan Was Rejected

The original plan proposed creating N new node_states (one per row) to record LLM calls. A 4-specialist review board rejected this because:

1. **Violates uniqueness constraint:** `UniqueConstraint("token_id", "step_index", "attempt")` prevents multiple states per token at same step
2. **Creates orphan states:** New states wouldn't link to batch infrastructure (`batches`, `batch_members` tables)
3. **Breaks audit semantics:** One transform execution should create one node_state, not N+1
4. **"Fixes that Fail" pattern:** Creates parallel audit mechanism instead of using existing infrastructure

### The Correct Architecture

The `calls` table supports multiple calls per state via `call_index`:

```
calls table structure for a batch operation:
├── state_id (batch's aggregation_state_id)
│   ├── call_index=0: HTTP file upload
│   ├── call_index=1: HTTP batch create
│   ├── call_index=2: HTTP status check
│   ├── call_index=3: HTTP file download
│   ├── call_index=4: LLM row 0  ← NEW
│   ├── call_index=5: LLM row 1  ← NEW
│   └── call_index=N+3: LLM row N-1  ← NEW
```

**Benefits:**
- Uses existing schema (no migration)
- One node_state per batch execution (correct semantics)
- `custom_id` in request_data maps calls back to specific tokens
- `explain()` already traverses calls table

---

## Prerequisites

Before starting, verify:
1. Existing batch infrastructure works: `pytest tests/engine/test_aggregation_executor.py -v`
2. Azure batch transform tests pass: `pytest tests/plugins/llm/test_azure_batch.py -v`

---

## Task 1: Store Original Requests in Checkpoint

**Files:**
- Modify: `src/elspeth/plugins/llm/azure_batch.py` (around line 375 and 472-481)
- Test: `tests/plugins/llm/test_azure_batch.py`

**Purpose:** Store the original LLM request bodies in checkpoint so we can record them as calls when results arrive.

**Step 1: Write the failing test**

```python
# In tests/plugins/llm/test_azure_batch.py
import pytest
from unittest.mock import MagicMock, patch

from elspeth.contracts import BatchPendingError
from elspeth.plugins.llm.azure_batch import AzureBatchLLMTransform


def test_checkpoint_includes_original_requests():
    """Checkpoint should include original LLM request data for audit recording."""
    config = {
        "deployment_name": "gpt-4o-batch",
        "endpoint": "https://test.openai.azure.com",
        "api_key": "test-key",
        "template": "Analyze: {{ row.text }}",
        "schema": {"fields": "dynamic"},
    }
    transform = AzureBatchLLMTransform(config)

    # Track checkpoint updates
    captured_checkpoint = None

    def capture_checkpoint(data):
        nonlocal captured_checkpoint
        captured_checkpoint = data

    # Create mock context
    ctx = MagicMock()
    ctx.run_id = "test-run"
    ctx.get_checkpoint.return_value = None  # Fresh batch
    ctx.update_checkpoint = capture_checkpoint
    ctx.record_call = MagicMock()

    # Mock Azure client
    mock_file = MagicMock()
    mock_file.id = "file-123"
    mock_file.status = "uploaded"

    mock_batch = MagicMock()
    mock_batch.id = "batch-456"
    mock_batch.status = "validating"

    with patch.object(transform, "_get_client") as mock_client:
        mock_client.return_value.files.create.return_value = mock_file
        mock_client.return_value.batches.create.return_value = mock_batch

        rows = [{"text": "Hello"}, {"text": "World"}]

        # Submit batch (will raise BatchPendingError)
        with pytest.raises(BatchPendingError):
            transform._process_batch(rows, ctx)

    # Verify checkpoint includes requests
    assert captured_checkpoint is not None
    assert "requests" in captured_checkpoint
    assert len(captured_checkpoint["requests"]) == 2

    # Each request should have the full LLM request body
    for custom_id, request in captured_checkpoint["requests"].items():
        assert "messages" in request
        assert "model" in request
        assert request["model"] == "gpt-4o-batch"
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/plugins/llm/test_azure_batch.py::test_checkpoint_includes_original_requests -v`
Expected: FAIL with KeyError "requests"

**Step 3: Write minimal implementation**

In `src/elspeth/plugins/llm/azure_batch.py`, modify the `_submit_batch` method.

After building `requests` list (around line 376), add:

```python
        # Build request lookup for audit recording (custom_id -> request body)
        requests_by_id = {req["custom_id"]: req["body"] for req in requests}
```

Update checkpoint_data (around line 473-481) to include requests:

```python
        # 4. CHECKPOINT immediately after submit
        checkpoint_data = {
            "batch_id": batch.id,
            "input_file_id": batch_file.id,
            "row_mapping": row_mapping,
            "template_errors": template_errors,
            "submitted_at": datetime.now(UTC).isoformat(),
            "row_count": len(rows),
            "requests": requests_by_id,  # ADD: Original requests for LLM call recording
        }
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/plugins/llm/test_azure_batch.py::test_checkpoint_includes_original_requests -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/elspeth/plugins/llm/azure_batch.py tests/plugins/llm/test_azure_batch.py
git commit -m "feat(azure-batch): store original requests in checkpoint

Include full LLM request bodies in checkpoint for audit recording
when results arrive. Part of BUG-AZURE-01 fix.

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
```

---

## Task 2: Record LLM Calls Against Batch State When Processing Results

**Files:**
- Modify: `src/elspeth/plugins/llm/azure_batch.py` (in `_download_results` method)
- Test: `tests/plugins/llm/test_azure_batch.py`

**Purpose:** When batch results arrive, record each LLM interaction as a call against the batch's existing state_id. This is the core fix.

**Key insight:** The batch already has a node_state (created by AggregationExecutor). We use `ctx.record_call()` which increments `call_index` automatically. No new node_states needed.

**Step 1: Write the failing test**

```python
# In tests/plugins/llm/test_azure_batch.py
def test_download_results_records_llm_calls():
    """Processing results should record LLM calls per row against batch state."""
    config = {
        "deployment_name": "gpt-4o-batch",
        "endpoint": "https://test.openai.azure.com",
        "api_key": "test-key",
        "template": "Analyze: {{ row.text }}",
        "schema": {"fields": "dynamic"},
    }
    transform = AzureBatchLLMTransform(config)

    # Track recorded calls
    recorded_calls = []

    def capture_call(**kwargs):
        recorded_calls.append(kwargs)
        return MagicMock()

    # Create mock context with state_id (simulates batch's aggregation state)
    ctx = MagicMock()
    ctx.run_id = "test-run"
    ctx.state_id = "batch-state-123"  # The batch's node_state
    ctx.record_call = capture_call
    ctx.get_checkpoint.return_value = {
        "batch_id": "azure-batch-789",
        "row_mapping": {"row-0-abc": 0, "row-1-def": 1},
        "requests": {
            "row-0-abc": {
                "messages": [{"role": "user", "content": "Analyze: Hello"}],
                "model": "gpt-4o-batch",
            },
            "row-1-def": {
                "messages": [{"role": "user", "content": "Analyze: World"}],
                "model": "gpt-4o-batch",
            },
        },
    }
    ctx.clear_checkpoint = MagicMock()

    # Mock Azure batch completion
    mock_batch = MagicMock()
    mock_batch.id = "azure-batch-789"
    mock_batch.status = "completed"
    mock_batch.output_file_id = "output-file-999"

    # Mock output file content (JSONL)
    output_jsonl = """{"custom_id": "row-0-abc", "response": {"body": {"choices": [{"message": {"content": "Analysis: Greeting"}}], "usage": {"total_tokens": 10}}}}
{"custom_id": "row-1-def", "response": {"body": {"choices": [{"message": {"content": "Analysis: Planet"}}], "usage": {"total_tokens": 12}}}}"""

    mock_output_content = MagicMock()
    mock_output_content.text = output_jsonl

    rows = [{"text": "Hello"}, {"text": "World"}]

    with patch.object(transform, "_get_client") as mock_client:
        mock_client.return_value.files.content.return_value = mock_output_content

        result = transform._download_results(mock_batch, ctx.get_checkpoint(), rows, ctx)

    # Verify result is successful
    assert result.status == "success"
    assert len(result.rows) == 2

    # Verify LLM calls were recorded
    from elspeth.contracts import CallType, CallStatus

    llm_calls = [c for c in recorded_calls if c.get("call_type") == CallType.LLM]
    assert len(llm_calls) == 2, f"Expected 2 LLM calls, got {len(llm_calls)}"

    # Verify first LLM call has correct data
    call1 = llm_calls[0]
    assert "custom_id" in call1["request_data"]
    assert call1["request_data"]["messages"][0]["content"] == "Analyze: Hello"
    assert call1["status"] == CallStatus.SUCCESS

    # Verify second LLM call
    call2 = llm_calls[1]
    assert call2["request_data"]["messages"][0]["content"] == "Analyze: World"
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/plugins/llm/test_azure_batch.py::test_download_results_records_llm_calls -v`
Expected: FAIL (no LLM calls recorded)

**Step 3: Write minimal implementation**

In `src/elspeth/plugins/llm/azure_batch.py`, in the `_download_results` method, add LLM call recording after parsing results but before clearing checkpoint.

Find the line `# Clear checkpoint after successful completion` (around line 792) and add BEFORE it:

```python
        # Record per-row LLM calls against the batch's state
        # Uses existing state_id from context (set by AggregationExecutor)
        requests_data = checkpoint.get("requests", {})

        for custom_id, result in results_by_id.items():
            original_request = requests_data.get(custom_id, {})
            row_index = row_mapping.get(custom_id)

            # Determine call status from result
            if result.get("error"):
                call_status = CallStatus.ERROR
                response_data = None
                error_data = {"error": result.get("error")}
            else:
                call_status = CallStatus.SUCCESS
                response_body = result.get("response", {}).get("body", {})
                response_data = response_body
                error_data = None

            # Record LLM call with custom_id for token mapping
            ctx.record_call(
                call_type=CallType.LLM,
                status=call_status,
                request_data={
                    "custom_id": custom_id,
                    "row_index": row_index,
                    **original_request,
                },
                response_data=response_data,
                error=error_data,
            )
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/plugins/llm/test_azure_batch.py::test_download_results_records_llm_calls -v`
Expected: PASS

**Step 5: Run full Azure batch test suite**

Run: `pytest tests/plugins/llm/test_azure_batch.py -v`
Expected: All tests pass

**Step 6: Commit**

```bash
git add src/elspeth/plugins/llm/azure_batch.py tests/plugins/llm/test_azure_batch.py
git commit -m "feat(azure-batch): record per-row LLM calls against batch state

When processing batch results, record each LLM prompt/response as a call
against the batch's existing node_state. Uses call_index to distinguish
multiple calls per state. Core fix for BUG-AZURE-01 and BUG-AZURE-02.

Architecture: Uses existing calls table with different call_index values
rather than creating new node_states (which was architecturally incorrect).

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
```

---

## Task 3: Handle Error Results Correctly

**Files:**
- Modify: `src/elspeth/plugins/llm/azure_batch.py` (LLM call recording)
- Test: `tests/plugins/llm/test_azure_batch.py`

**Purpose:** Ensure LLM calls that returned errors are recorded with `CallStatus.ERROR`, not `SUCCESS`.

**Step 1: Write the failing test**

```python
# In tests/plugins/llm/test_azure_batch.py
def test_download_results_records_failed_llm_calls_correctly():
    """LLM calls that failed should be recorded with ERROR status."""
    config = {
        "deployment_name": "gpt-4o-batch",
        "endpoint": "https://test.openai.azure.com",
        "api_key": "test-key",
        "template": "Analyze: {{ row.text }}",
        "schema": {"fields": "dynamic"},
    }
    transform = AzureBatchLLMTransform(config)

    recorded_calls = []

    def capture_call(**kwargs):
        recorded_calls.append(kwargs)
        return MagicMock()

    ctx = MagicMock()
    ctx.run_id = "test-run"
    ctx.state_id = "batch-state-123"
    ctx.record_call = capture_call
    ctx.get_checkpoint.return_value = {
        "batch_id": "azure-batch-789",
        "row_mapping": {"row-0-abc": 0, "row-1-def": 1},
        "requests": {
            "row-0-abc": {"messages": [{"role": "user", "content": "Good"}], "model": "gpt-4o-batch"},
            "row-1-def": {"messages": [{"role": "user", "content": "Bad"}], "model": "gpt-4o-batch"},
        },
    }
    ctx.clear_checkpoint = MagicMock()

    mock_batch = MagicMock()
    mock_batch.output_file_id = "output-file-999"

    # One success, one error
    output_jsonl = """{"custom_id": "row-0-abc", "response": {"body": {"choices": [{"message": {"content": "OK"}}]}}}
{"custom_id": "row-1-def", "error": {"code": "content_filter", "message": "Content filtered"}}"""

    mock_output_content = MagicMock()
    mock_output_content.text = output_jsonl

    rows = [{"text": "Good"}, {"text": "Bad"}]

    with patch.object(transform, "_get_client") as mock_client:
        mock_client.return_value.files.content.return_value = mock_output_content

        result = transform._download_results(mock_batch, ctx.get_checkpoint(), rows, ctx)

    from elspeth.contracts import CallType, CallStatus

    llm_calls = [c for c in recorded_calls if c.get("call_type") == CallType.LLM]
    assert len(llm_calls) == 2

    # First call should be SUCCESS
    success_call = next(c for c in llm_calls if "Good" in str(c["request_data"]))
    assert success_call["status"] == CallStatus.SUCCESS
    assert success_call["response_data"] is not None

    # Second call should be ERROR
    error_call = next(c for c in llm_calls if "Bad" in str(c["request_data"]))
    assert error_call["status"] == CallStatus.ERROR
    assert error_call["error"] is not None
    assert "content_filter" in str(error_call["error"])
```

**Step 2: Run test to verify it passes**

If Task 2 was implemented correctly with error handling, this should already pass.

Run: `pytest tests/plugins/llm/test_azure_batch.py::test_download_results_records_failed_llm_calls_correctly -v`
Expected: PASS (if Task 2 implementation handles errors)

**Step 3: Commit**

```bash
git add tests/plugins/llm/test_azure_batch.py
git commit -m "test(azure-batch): verify failed LLM calls recorded with ERROR status

Ensures mixed success/error batch results are recorded correctly in
the audit trail with appropriate CallStatus values.

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
```

---

## Task 4: Handle Missing Requests in Checkpoint (Backwards Compatibility)

**Files:**
- Modify: `src/elspeth/plugins/llm/azure_batch.py`
- Test: `tests/plugins/llm/test_azure_batch.py`

**Purpose:** Batches submitted before this fix won't have `requests` in checkpoint. Handle gracefully with a warning.

**Step 1: Write the test**

```python
# In tests/plugins/llm/test_azure_batch.py
import logging

def test_download_results_handles_checkpoint_without_requests(caplog):
    """Old checkpoints without 'requests' should log warning but not crash."""
    config = {
        "deployment_name": "gpt-4o-batch",
        "endpoint": "https://test.openai.azure.com",
        "api_key": "test-key",
        "template": "Analyze: {{ row.text }}",
        "schema": {"fields": "dynamic"},
    }
    transform = AzureBatchLLMTransform(config)

    recorded_calls = []
    ctx = MagicMock()
    ctx.run_id = "test-run"
    ctx.state_id = "batch-state-123"
    ctx.record_call = lambda **kwargs: recorded_calls.append(kwargs)
    ctx.get_checkpoint.return_value = {
        "batch_id": "old-batch",
        "row_mapping": {"row-0-abc": 0},
        # NO 'requests' field - old checkpoint format
    }
    ctx.clear_checkpoint = MagicMock()

    mock_batch = MagicMock()
    mock_batch.output_file_id = "output-file"

    output_jsonl = '{"custom_id": "row-0-abc", "response": {"body": {"choices": [{"message": {"content": "OK"}}]}}}'
    mock_output_content = MagicMock()
    mock_output_content.text = output_jsonl

    rows = [{"text": "Hello"}]

    with patch.object(transform, "_get_client") as mock_client:
        mock_client.return_value.files.content.return_value = mock_output_content

        with caplog.at_level(logging.WARNING):
            result = transform._download_results(mock_batch, ctx.get_checkpoint(), rows, ctx)

    # Should still succeed
    assert result.status == "success"

    # LLM calls still recorded (with empty request_data)
    from elspeth.contracts import CallType
    llm_calls = [c for c in recorded_calls if c.get("call_type") == CallType.LLM]
    assert len(llm_calls) == 1

    # Should have logged a warning about missing requests
    # (Implementation should add this warning)
```

**Step 2: Update implementation if needed**

In the LLM call recording loop, add a warning if `requests_data` is empty:

```python
        requests_data = checkpoint.get("requests", {})

        if not requests_data:
            import logging
            logger = logging.getLogger(__name__)
            logger.warning(
                "Checkpoint for batch %s has no 'requests' field. "
                "LLM calls will be recorded without original request data. "
                "This batch was likely submitted before the audit fix was deployed.",
                checkpoint.get("batch_id", "unknown"),
            )
```

**Step 3: Commit**

```bash
git add src/elspeth/plugins/llm/azure_batch.py tests/plugins/llm/test_azure_batch.py
git commit -m "fix(azure-batch): handle old checkpoints without requests field

Batches submitted before audit fix deployment won't have 'requests' in
checkpoint. Log warning and continue rather than failing.

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
```

---

## Task 5: Integration Test - Verify explain() Works

**Files:**
- Test: `tests/plugins/llm/test_azure_batch.py`

**Purpose:** End-to-end test verifying that `explain()` can retrieve LLM details for batch rows.

**Step 1: Write integration test**

```python
# In tests/plugins/llm/test_azure_batch.py
def test_batch_llm_calls_visible_in_audit_trail(real_landscape_db):
    """Integration: explain() should return LLM call details for batch operations."""
    from elspeth.core.landscape import LandscapeRecorder
    from elspeth.contracts import CallType

    recorder = LandscapeRecorder(real_landscape_db)

    # Create a run and batch operation
    run = recorder.begin_run(config={"pipeline": "test"}, canonical_version="v1")

    # Register source node
    source_node_id = recorder.register_node(
        run_id=run.run_id,
        node_id="source:csv",
        node_type="source",
        plugin_name="csv_source",
        config={},
    )

    # Register batch transform node
    batch_node_id = recorder.register_node(
        run_id=run.run_id,
        node_id="transform:azure_batch_llm",
        node_type="aggregation",
        plugin_name="azure_batch_llm",
        config={"deployment_name": "gpt-4o-batch"},
    )

    # Create a source row and token
    row_id = recorder.record_source_row(
        run_id=run.run_id,
        node_id=source_node_id,
        row_data={"text": "Hello world"},
    )

    token = recorder.create_token(
        run_id=run.run_id,
        row_id=row_id,
        node_id=source_node_id,
    )

    # Create batch and add member
    batch = recorder.create_batch(
        run_id=run.run_id,
        aggregation_node_id=batch_node_id,
    )

    recorder.add_batch_member(
        batch_id=batch.batch_id,
        token_id=token.token_id,
        ordinal=0,
    )

    # Create node_state for batch operation
    state = recorder.begin_node_state(
        token_id=token.token_id,
        node_id=batch_node_id,
        run_id=run.run_id,
        step_index=1,
        input_data={"batch_rows": [{"text": "Hello world"}]},
    )

    # Record LLM call against the batch state
    call = recorder.record_call(
        state_id=state.state_id,
        call_index=0,
        call_type=CallType.LLM,
        status="SUCCESS",
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
        status="COMPLETED",
        output_data={"llm_response": "Analysis: Greeting"},
    )

    # Verify call was recorded
    from elspeth.core.landscape import Lineage

    lineage = Lineage(recorder)
    explanation = lineage.explain(run_id=run.run_id, token_id=token.token_id)

    # Should have at least one node_state
    assert len(explanation.node_states) >= 1

    # Find the batch state
    batch_state = next(
        (s for s in explanation.node_states if s.node_id == batch_node_id),
        None
    )
    assert batch_state is not None

    # Should have calls
    assert len(batch_state.calls) >= 1

    # Find LLM call
    llm_call = next(
        (c for c in batch_state.calls if c.call_type == CallType.LLM),
        None
    )
    assert llm_call is not None
    assert "messages" in llm_call.request_data
    assert "choices" in llm_call.response_data
```

**Step 2: Run integration test**

Run: `pytest tests/plugins/llm/test_azure_batch.py::test_batch_llm_calls_visible_in_audit_trail -v`
Expected: PASS

**Step 3: Commit**

```bash
git add tests/plugins/llm/test_azure_batch.py
git commit -m "test(azure-batch): integration test for explain() with batch LLM calls

Verifies end-to-end that explain() returns LLM prompt/response details
for batch operations. This confirms BUG-AZURE-01 is fixed.

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
```

---

## Task 6: Update Bug Documentation

**Files:**
- Modify: `docs/bugs/open/BUG-AZURE-01-batch-audit-missing-payloads.md`
- Modify: `docs/bugs/open/BUG-AZURE-02-batch-synthetic-state-ids.md`

**Step 1: Add CLOSURE sections to both bug files**

For BUG-AZURE-01:
```markdown
---

## CLOSURE: 2026-01-29

**Status:** FIXED

**Fixed By:** Claude (with 4-specialist review board)

**Resolution:**

The Azure batch transform now records per-row LLM calls against the batch's existing node_state:

1. Original requests stored in checkpoint during submit phase
2. When results return, each LLM prompt/response is recorded as a call with:
   - `call_type=LLM`
   - Unique `call_index` per row
   - `custom_id` in request_data for token mapping
3. Uses existing `calls` table - no new node_states needed
4. explain() queries now return full LLM interaction details

**Architecture Note:**
Original plan proposed creating N new node_states per batch. This was rejected by review board as it violated audit model semantics. Correct approach uses existing infrastructure.

**Tests Added:**
- `test_checkpoint_includes_original_requests`
- `test_download_results_records_llm_calls`
- `test_download_results_records_failed_llm_calls_correctly`
- `test_download_results_handles_checkpoint_without_requests`
- `test_batch_llm_calls_visible_in_audit_trail`

**Verified By:** 4-specialist review board (2026-01-29)
```

For BUG-AZURE-02, similar closure noting that LLM calls now use real state_ids (the batch's aggregation_state_id) rather than synthetic ones.

**Step 2: Move files to closed directory**

```bash
mkdir -p docs/bugs/closed/llm
git mv docs/bugs/open/BUG-AZURE-01-batch-audit-missing-payloads.md docs/bugs/closed/llm/
git mv docs/bugs/open/BUG-AZURE-02-batch-synthetic-state-ids.md docs/bugs/closed/llm/
```

**Step 3: Update README counts**

Update `docs/bugs/open/README.md` to reflect 2 fewer P1 bugs.

**Step 4: Commit**

```bash
git add docs/bugs/
git commit -m "docs(bugs): close BUG-AZURE-01 and BUG-AZURE-02

Both bugs fixed by recording per-row LLM calls against batch state.
Uses existing calls table with call_index, not new node_states.

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
```

---

## Task 7: Final Verification

**Step 1: Run full test suite**

```bash
pytest tests/plugins/llm/test_azure_batch.py -v
pytest tests/engine/test_aggregation_executor.py -v
pytest tests/core/landscape/ -v
```

Expected: All tests pass

**Step 2: Run type checking**

```bash
.venv/bin/python -m mypy src/elspeth/plugins/llm/azure_batch.py
```

Expected: No type errors

**Step 3: Run linting**

```bash
.venv/bin/python -m ruff check src/elspeth/plugins/llm/azure_batch.py
```

Expected: No lint errors

**Step 4: Create final summary commit**

```bash
git add .
git commit -m "fix(azure-batch): complete audit trail for batch LLM operations

Fixes BUG-AZURE-01 (missing payloads) and BUG-AZURE-02 (synthetic state_ids).

Architecture:
- Records N LLM calls against ONE batch state using call_index
- Uses existing calls table (no schema changes)
- Stores requests in checkpoint for later recording
- custom_id in request_data maps calls back to tokens

Changes:
- Store original requests in checkpoint during submit
- Record LLM calls when processing batch results
- Handle error results with CallStatus.ERROR
- Backwards compatible with old checkpoints (warns if no requests)

Review: Approved by 4-specialist review board (architecture, python,
QA, systems thinking). Original plan rejected for creating orphan
node_states; revised to use existing infrastructure correctly.

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
```

---

## Summary

| Task | Description | Files |
|------|-------------|-------|
| 1 | Store requests in checkpoint | azure_batch.py |
| 2 | Record LLM calls against batch state | azure_batch.py |
| 3 | Handle error results correctly | azure_batch.py |
| 4 | Backwards compatibility for old checkpoints | azure_batch.py |
| 5 | Integration test for explain() | test_azure_batch.py |
| 6 | Update bug docs | docs/bugs/ |
| 7 | Final verification | - |

**Estimated time:** 2-3 hours

**Risk assessment:** Low - uses existing infrastructure, no schema changes

**Key difference from original plan:** Records calls against existing batch state instead of creating new node_states. This is architecturally correct and uses existing schema.

---

## PLAN COMPLETION: 2026-01-29

**Status:** COMPLETED

**Implemented By:** Claude (with subagent-driven development + 4-specialist review)

**Summary:**
All 7 tasks completed successfully. The Azure batch LLM transform now records per-row LLM calls in the audit trail with full FK integrity.

**Implementation Highlights:**
1. ✅ Task 1: Requests stored in checkpoint (`requests_by_id` lookup)
2. ✅ Task 2: LLM calls recorded via `ctx.record_call()` with auto-incrementing `call_index`
3. ✅ Task 3: Error results handled with `CallStatus.ERROR`
4. ✅ Task 5: Integration tests verify `explain()` returns full LLM details
5. ✅ Task 6: Bug docs moved to `docs/bugs/closed/llm/` with CLOSURE sections
6. ✅ Task 7: All 43 tests pass, mypy clean, ruff clean, pre-commit hooks pass

**Notable Decisions:**
- **No backwards compatibility guards** - Per No Legacy Code Policy, updated all test fixtures instead of adding version checks
- **Tier 3 boundary validation** - Added at Azure response parse boundary (lines 707-727)
- **Task 4 deleted** - Was a compat guard, forbidden by CLAUDE.md

**Commit:** `ad8a368` - fix(llm): record Azure batch LLM calls in audit trail

**Files Changed:** 7 files, +713 insertions, -25 deletions
