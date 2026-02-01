# TransformErrorReason TypedDict Implementation Plan

**Status:** ✅ IMPLEMENTED (2026-02-01)

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace `dict[str, Any]` typing on `TransformResult.reason` with a properly typed `TransformErrorReason` TypedDict, enabling compile-time type safety for transform error audit trails.

**Architecture:** Single TypedDict with:
- Required `reason` field using `Literal` type to constrain valid error categories
- Nested TypedDicts for structured fields (`TemplateErrorEntry`, `RowErrorEntry`, `UsageStats`)
- Extensive optional context fields covering all observed usage patterns

**Tech Stack:** Python TypedDict with `NotRequired` and `Literal` (Python 3.12+).

**Bead:** elspeth-rapid-q6s

**Review Status:** Approved with amendments from 4-perspective review (2026-01-31).

---

## Implementation Summary

- Added `TransformErrorCategory` and `TransformErrorReason` TypedDicts (`src/elspeth/contracts/errors.py`).
- `TransformResult.error()` now requires structured reasons (`src/elspeth/contracts/results.py`).
- Contract tests validate required fields and optional context (`tests/contracts/test_errors.py`).

## Task 1: Define TransformErrorReason TypedDict

**Files:**

- Modify: `src/elspeth/contracts/errors.py`

### Step 1: Add nested TypedDicts and TransformErrorReason

Add after the existing `TransformReason` TypedDict (around line 75):

```python
from typing import Any, Literal, NotRequired, TypedDict


# =============================================================================
# Transform Error Reason Types
# =============================================================================


class TemplateErrorEntry(TypedDict):
    """Entry in template_errors list for batch processing failures."""

    row_index: int
    error: str


class RowErrorEntry(TypedDict):
    """Entry in row_errors list for batch processing failures."""

    row_index: int
    reason: str
    error: NotRequired[str]


class UsageStats(TypedDict, total=False):
    """LLM token usage statistics."""

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


# Literal type for compile-time validation of error categories
TransformErrorCategory = Literal[
    # API/Network errors
    "api_error",
    "api_call_failed",
    "llm_call_failed",
    "network_error",
    "permanent_error",
    "retry_timeout",
    # Field/validation errors
    "missing_field",
    "type_mismatch",
    "validation_failed",
    "invalid_input",
    # Template errors
    "template_rendering_failed",
    "all_templates_failed",
    # JSON/response parsing errors
    "json_parse_failed",
    "invalid_json_response",
    "invalid_json_type",
    "empty_choices",
    "malformed_response",
    "missing_output_field",
    "response_truncated",
    # Batch processing errors
    "batch_error",
    "batch_create_failed",
    "batch_failed",
    "batch_cancelled",
    "batch_expired",
    "batch_timeout",
    "batch_retrieve_failed",
    "file_upload_failed",
    "file_download_failed",
    "all_output_lines_malformed",
    "all_rows_failed",
    "result_not_found",
    "query_failed",
    "rate_limited",
    # Content filtering
    "blocked_content",
    "content_filtered",
    "content_safety_violation",
    "prompt_injection_detected",
    # Generic (for tests and edge cases)
    "test_error",
    "property_test_error",
    "simulated_failure",
    "deliberate_failure",
    "intentional_failure",
]


class TransformErrorReason(TypedDict):
    """Reason for transform processing error.

    Used when transforms return TransformResult.error().
    The reason field describes what category of error occurred.
    Additional fields provide context specific to the error type.

    Recorded in the audit trail (Tier 1 - full trust) for legal traceability.
    Every transform error must be attributable to its cause.

    Required field:
        reason: Error category from TransformErrorCategory literal type.
                Compile-time validated to prevent typos.

    Common context fields:
        error: Exception message or detailed error description
        field: Field name for field-related errors
        error_type: Sub-category (e.g., "http_error", "network_error")
        message: Human-readable error message (alternative to error)

    Multi-query/template context:
        query: Which query in multi-query failed
        template_hash: Template version for debugging
        template_errors: List of per-row template failures (batch processing)

    LLM response context:
        max_tokens: Configured max tokens limit
        completion_tokens: Actual tokens used in response
        prompt_tokens: Tokens used in prompt
        raw_response: Truncated raw LLM response content
        raw_response_preview: Alternative name for truncated preview
        content_after_fence_strip: Content after markdown fence removal
        usage: Token usage stats from LLM response
        response: Full response object for debugging
        response_keys: Keys present in response dict
        body_preview: HTTP body preview for errors
        content_type: Content-Type header value

    Type validation context:
        expected: Expected type or value
        actual: Actual type or value received
        value: The actual value (truncated for audit)

    Rate limiting/timeout context:
        elapsed_seconds: Time elapsed before timeout
        max_seconds: Maximum allowed time
        status_code: HTTP status code

    Content filtering context:
        matched_pattern: Regex pattern that matched
        match_context: Context around the match
        categories: Content safety violation categories

    Batch processing context:
        batch_id: Azure/OpenRouter batch job ID
        queries_completed: Number of queries completed before failure
        row_errors: List of per-row error entries

    Example usage:
        # API error with exception details
        TransformResult.error({
            "reason": "api_error",
            "error": str(e),
            "error_type": "http_error",
        })

        # Field-related error
        TransformResult.error({
            "reason": "missing_field",
            "field": "customer_id",
        })

        # LLM response truncation
        TransformResult.error({
            "reason": "response_truncated",
            "error": "Response was truncated at 1000 tokens",
            "query": "sentiment",
            "max_tokens": 1000,
            "completion_tokens": 1000,
        })
    """

    # REQUIRED - error category (Literal-typed for compile-time validation)
    reason: TransformErrorCategory

    # Common context
    error: NotRequired[str]
    field: NotRequired[str]
    error_type: NotRequired[str]
    message: NotRequired[str]

    # Multi-query/template context
    query: NotRequired[str]
    template_hash: NotRequired[str]
    template_errors: NotRequired[list[TemplateErrorEntry]]

    # LLM response context
    max_tokens: NotRequired[int]
    completion_tokens: NotRequired[int]
    prompt_tokens: NotRequired[int]
    raw_response: NotRequired[str]
    raw_response_preview: NotRequired[str]
    content_after_fence_strip: NotRequired[str]
    usage: NotRequired[UsageStats]
    response: NotRequired[dict[str, Any]]
    response_keys: NotRequired[list[str]]
    body_preview: NotRequired[str]
    content_type: NotRequired[str]

    # Type validation context
    expected: NotRequired[str]
    actual: NotRequired[str]
    value: NotRequired[str]

    # Rate limiting/timeout context
    elapsed_seconds: NotRequired[float]
    max_seconds: NotRequired[float]
    status_code: NotRequired[int]

    # Content filtering context
    matched_pattern: NotRequired[str]
    match_context: NotRequired[str]
    categories: NotRequired[list[str]]

    # Batch processing context
    batch_id: NotRequired[str]
    queries_completed: NotRequired[int]
    row_errors: NotRequired[list[RowErrorEntry]]
```

### Step 2: Update imports at top of file

Ensure the imports include `Literal`:

```python
from typing import Any, Literal, NotRequired, TypedDict
```

### Step 3: Verify import works

Run: `.venv/bin/python -c "from elspeth.contracts.errors import TransformErrorReason, TransformErrorCategory; print('OK')"`
Expected: `OK`

### Step 4: Commit

```bash
git add src/elspeth/contracts/errors.py
git commit -m "$(cat <<'EOF'
feat(contracts): add TransformErrorReason TypedDict with Literal reason

Define structured type for TransformResult.error() reason payloads.

Key design decisions:
- reason field uses Literal type for compile-time validation of error categories
- Nested TypedDicts for structured fields (TemplateErrorEntry, RowErrorEntry, UsageStats)
- Extensive optional fields cover all observed usage patterns

Required field: reason (TransformErrorCategory literal)
Nested types: TemplateErrorEntry, RowErrorEntry, UsageStats
Optional fields grouped by category: common, LLM, validation, rate limiting, batch

Part of elspeth-rapid-q6s type safety audit.

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: Update contracts/\_\_init\_\_.py Exports

**Files:**

- Modify: `src/elspeth/contracts/__init__.py`

### Step 1: Add imports

Add all new types to the imports from errors (around line 101-108):

```python
from elspeth.contracts.errors import (
    BatchPendingError,
    ConfigGateReason,
    ExecutionError,
    PluginGateReason,
    RoutingReason,
    RowErrorEntry,  # ADD
    TemplateErrorEntry,  # ADD
    TransformErrorCategory,  # ADD
    TransformErrorReason,  # ADD
    TransformReason,
    UsageStats,  # ADD
)
```

### Step 2: Add to \_\_all\_\_

Add all new types to the errors section of `__all__` (around line 151-158):

```python
    # errors
    "BatchPendingError",
    "ConfigGateReason",
    "ExecutionError",
    "PluginGateReason",
    "RoutingReason",
    "RowErrorEntry",  # ADD
    "TemplateErrorEntry",  # ADD
    "TransformErrorCategory",  # ADD
    "TransformErrorReason",  # ADD
    "TransformReason",
    "UsageStats",  # ADD
```

### Step 3: Verify export works

Run: `.venv/bin/python -c "from elspeth.contracts import TransformErrorReason, TransformErrorCategory, TemplateErrorEntry; print('OK')"`
Expected: `OK`

### Step 4: Commit

```bash
git add src/elspeth/contracts/__init__.py
git commit -m "$(cat <<'EOF'
feat(contracts): export TransformErrorReason and related types

Exports: TransformErrorReason, TransformErrorCategory,
TemplateErrorEntry, RowErrorEntry, UsageStats

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: Update TransformResult.error() Signature

**Files:**

- Modify: `src/elspeth/contracts/results.py`

### Step 1: Add import

Add import at top of file (around line 13):

```python
from typing import TYPE_CHECKING, Any, Literal

from elspeth.contracts.errors import TransformErrorReason  # ADD THIS
```

### Step 2: Update reason field type

Change line 98 from:

```python
    reason: dict[str, Any] | None
```

To:

```python
    reason: TransformErrorReason | None
```

### Step 3: Update error() method signature

Change the `error()` method (around lines 139-153) from:

```python
    @classmethod
    def error(
        cls,
        reason: dict[str, Any],
        *,
        retryable: bool = False,
    ) -> "TransformResult":
        """Create error result with reason."""
        return cls(
            status="error",
            row=None,
            reason=reason,
            retryable=retryable,
            rows=None,
        )
```

To:

```python
    @classmethod
    def error(
        cls,
        reason: TransformErrorReason,
        *,
        retryable: bool = False,
    ) -> "TransformResult":
        """Create error result with structured reason.

        Args:
            reason: Error details with required 'reason' field from
                    TransformErrorCategory (compile-time validated).
                    See TransformErrorReason for all available context fields.
            retryable: Whether the error is transient and should be retried.

        Returns:
            TransformResult with status="error" and the provided reason.
        """
        return cls(
            status="error",
            row=None,
            reason=reason,
            retryable=retryable,
            rows=None,
        )
```

### Step 4: Run mypy

Run: `.venv/bin/python -m mypy src/elspeth/contracts/results.py --no-error-summary`
Expected: Should pass (all plugin usages have valid `reason` field)

### Step 5: Commit

```bash
git add src/elspeth/contracts/results.py
git commit -m "$(cat <<'EOF'
feat(results): type TransformResult.reason as TransformErrorReason

Update error() method signature and reason field to use
TransformErrorReason TypedDict instead of dict[str, Any].

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: Fix Plugin Outliers

**Files:**

- Modify: `src/elspeth/plugins/transforms/field_mapper.py:113`

### Step 1: Fix field\_mapper.py

Change line 113 from:

```python
                    return TransformResult.error({"message": f"Required field '{source}' not found in row"})
```

To:

```python
                    return TransformResult.error({"reason": "missing_field", "field": source})
```

### Step 2: Verify plugin works

Run: `.venv/bin/python -c "from elspeth.plugins.transforms.field_mapper import FieldMapper; print('OK')"`
Expected: `OK`

### Step 3: Commit

```bash
git add src/elspeth/plugins/transforms/field_mapper.py
git commit -m "$(cat <<'EOF'
fix(field_mapper): use standard TransformErrorReason pattern

Change {"message": "..."} to {"reason": "missing_field", "field": "..."}
to match the standard transform error pattern.

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 5: Update recorder.py Signature

**Files:**

- Modify: `src/elspeth/core/landscape/recorder.py`

### Step 1: Add import

Add to imports at top of file:

```python
from elspeth.contracts.errors import TransformErrorReason
```

### Step 2: Update method signature

Change `record_transform_error` signature (around line 2294-2302):

```python
    def record_transform_error(
        self,
        run_id: str,
        token_id: str,
        transform_id: str,
        row_data: dict[str, Any],
        error_details: TransformErrorReason,  # Changed from dict[str, Any]
        destination: str,
    ) -> str:
```

### Step 3: Run mypy

Run: `.venv/bin/python -m mypy src/elspeth/core/landscape/recorder.py --no-error-summary`
Expected: No new errors

### Step 4: Commit

```bash
git add src/elspeth/core/landscape/recorder.py
git commit -m "$(cat <<'EOF'
refactor(recorder): type record_transform_error with TransformErrorReason

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 6: Add Tests for TransformErrorReason

**Files:**

- Modify: `tests/contracts/test_errors.py`

### Step 1: Add comprehensive test classes

Add after `TestRoutingReasonUsage`:

```python
class TestTransformErrorReasonSchema:
    """Tests for TransformErrorReason TypedDict schema."""

    def test_transform_error_reason_required_keys(self) -> None:
        """TransformErrorReason has reason as required."""
        from elspeth.contracts import TransformErrorReason

        assert TransformErrorReason.__required_keys__ == frozenset({"reason"})

    def test_transform_error_reason_has_expected_optional_keys(self) -> None:
        """TransformErrorReason has expected optional keys."""
        from elspeth.contracts import TransformErrorReason

        # Check a subset of important optional keys
        optional = TransformErrorReason.__optional_keys__
        assert "error" in optional
        assert "field" in optional
        assert "error_type" in optional
        assert "query" in optional
        assert "max_tokens" in optional
        assert "status_code" in optional
        assert "template_errors" in optional
        assert "row_errors" in optional

    def test_transform_error_category_literal_values(self) -> None:
        """TransformErrorCategory contains expected error types."""
        from typing import get_args

        from elspeth.contracts import TransformErrorCategory

        categories = get_args(TransformErrorCategory)
        # Verify key categories exist
        assert "api_error" in categories
        assert "missing_field" in categories
        assert "template_rendering_failed" in categories
        assert "response_truncated" in categories
        assert "batch_failed" in categories


class TestTransformErrorReasonUsage:
    """Tests for constructing valid TransformErrorReason values."""

    def test_minimal_error_reason(self) -> None:
        """TransformErrorReason works with only required field."""
        from elspeth.contracts import TransformErrorReason

        reason: TransformErrorReason = {"reason": "api_error"}
        assert reason["reason"] == "api_error"

    def test_api_error_pattern(self) -> None:
        """Common API error pattern with error and error_type."""
        from elspeth.contracts import TransformErrorReason

        reason: TransformErrorReason = {
            "reason": "api_error",
            "error": "Connection refused",
            "error_type": "network_error",
        }
        assert reason["reason"] == "api_error"
        assert reason["error"] == "Connection refused"
        assert reason["error_type"] == "network_error"

    def test_field_error_pattern(self) -> None:
        """Common field-related error pattern."""
        from elspeth.contracts import TransformErrorReason

        reason: TransformErrorReason = {
            "reason": "missing_field",
            "field": "customer_id",
        }
        assert reason["reason"] == "missing_field"
        assert reason["field"] == "customer_id"

    def test_llm_truncation_pattern(self) -> None:
        """LLM response truncation with token counts."""
        from elspeth.contracts import TransformErrorReason

        reason: TransformErrorReason = {
            "reason": "response_truncated",
            "error": "Response truncated at 1000 tokens",
            "query": "sentiment",
            "max_tokens": 1000,
            "completion_tokens": 1000,
            "prompt_tokens": 500,
        }
        assert reason["reason"] == "response_truncated"
        assert reason["max_tokens"] == 1000

    def test_type_mismatch_pattern(self) -> None:
        """Type validation error pattern."""
        from elspeth.contracts import TransformErrorReason

        reason: TransformErrorReason = {
            "reason": "type_mismatch",
            "field": "score",
            "expected": "float",
            "actual": "str",
            "value": "not_a_number",
        }
        assert reason["reason"] == "type_mismatch"
        assert reason["expected"] == "float"
        assert reason["actual"] == "str"

    def test_rate_limit_pattern(self) -> None:
        """Rate limiting/timeout error pattern."""
        from elspeth.contracts import TransformErrorReason

        reason: TransformErrorReason = {
            "reason": "retry_timeout",
            "error": "Max retry time exceeded",
            "elapsed_seconds": 60.5,
            "max_seconds": 60.0,
            "status_code": 429,
        }
        assert reason["reason"] == "retry_timeout"
        assert reason["status_code"] == 429

    def test_batch_job_error_pattern(self) -> None:
        """Azure/OpenRouter batch job failure."""
        from elspeth.contracts import TransformErrorReason

        reason: TransformErrorReason = {
            "reason": "batch_failed",
            "batch_id": "batch_abc123",
            "error": "Job expired",
            "queries_completed": 42,
        }
        assert reason["batch_id"] == "batch_abc123"
        assert reason["queries_completed"] == 42

    def test_batch_template_errors_pattern(self) -> None:
        """Template errors in batch processing with nested TypedDict."""
        from elspeth.contracts import TemplateErrorEntry, TransformErrorReason

        error1: TemplateErrorEntry = {"row_index": 0, "error": "Missing field 'customer_id'"}
        error2: TemplateErrorEntry = {"row_index": 5, "error": "Invalid template syntax"}

        reason: TransformErrorReason = {
            "reason": "all_templates_failed",
            "template_errors": [error1, error2],
        }
        assert len(reason["template_errors"]) == 2
        assert reason["template_errors"][0]["row_index"] == 0

    def test_content_safety_violation_pattern(self) -> None:
        """Content safety API violation."""
        from elspeth.contracts import TransformErrorReason

        reason: TransformErrorReason = {
            "reason": "content_safety_violation",
            "field": "user_input",
            "categories": ["Violence", "SelfHarm"],
            "message": "Content violates safety policy",
        }
        assert reason["reason"] == "content_safety_violation"
        assert "Violence" in reason["categories"]

    def test_json_parsing_failure_pattern(self) -> None:
        """JSON parsing failure with response preview."""
        from elspeth.contracts import TransformErrorReason

        reason: TransformErrorReason = {
            "reason": "invalid_json_type",
            "expected": "object",
            "actual": "list",
            "raw_response_preview": "[1, 2, 3]",
            "query": "classification",
        }
        assert reason["expected"] == "object"
        assert reason["actual"] == "list"

    def test_template_rendering_failure_pattern(self) -> None:
        """Jinja2 template rendering failure."""
        from elspeth.contracts import TransformErrorReason

        reason: TransformErrorReason = {
            "reason": "template_rendering_failed",
            "error": "UndefinedError: 'customer_id' is undefined",
            "query": "sentiment_analysis",
            "template_hash": "sha256:abc123def456",
        }
        assert reason["template_hash"] is not None

    def test_usage_stats_nested_typeddict(self) -> None:
        """UsageStats nested TypedDict works correctly."""
        from elspeth.contracts import TransformErrorReason, UsageStats

        usage: UsageStats = {
            "prompt_tokens": 100,
            "completion_tokens": 50,
            "total_tokens": 150,
        }
        reason: TransformErrorReason = {
            "reason": "response_truncated",
            "usage": usage,
        }
        assert reason["usage"]["total_tokens"] == 150


class TestNestedTypeDicts:
    """Tests for nested TypedDict structures."""

    def test_template_error_entry_structure(self) -> None:
        """TemplateErrorEntry has correct fields."""
        from elspeth.contracts import TemplateErrorEntry

        entry: TemplateErrorEntry = {"row_index": 5, "error": "Missing field"}
        assert entry["row_index"] == 5
        assert entry["error"] == "Missing field"

    def test_row_error_entry_structure(self) -> None:
        """RowErrorEntry has correct fields."""
        from elspeth.contracts import RowErrorEntry

        entry: RowErrorEntry = {"row_index": 3, "reason": "api_error", "error": "Timeout"}
        assert entry["row_index"] == 3
        assert entry["reason"] == "api_error"

    def test_usage_stats_partial(self) -> None:
        """UsageStats allows partial fields (total=False)."""
        from elspeth.contracts import UsageStats

        # Only some fields provided
        usage: UsageStats = {"prompt_tokens": 100}
        assert usage["prompt_tokens"] == 100
```

### Step 2: Run tests

Run: `.venv/bin/python -m pytest tests/contracts/test_errors.py -v`
Expected: All tests pass

### Step 3: Commit

```bash
git add tests/contracts/test_errors.py
git commit -m "$(cat <<'EOF'
test(errors): add comprehensive TransformErrorReason TypedDict tests

Tests cover:
- Schema introspection (required/optional keys, Literal values)
- All common usage patterns (API, field, LLM, batch, content safety)
- Nested TypedDicts (TemplateErrorEntry, RowErrorEntry, UsageStats)
- Edge cases (JSON parsing, template rendering)

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 6.5: Update Property Test Strategies (CRITICAL)

**Files:**

- Modify: `tests/property/engine/test_executor_properties.py`
- Modify: `tests/property/contracts/test_serialization_properties.py`
- Modify: `tests/property/conftest.py` (if strategy defined there)

### Step 1: Update error\_reasons strategy in test\_executor\_properties.py

Find the `error_reasons` strategy (around lines 64-70) and change from:

```python
error_reasons = st.dictionaries(
    keys=st.text(min_size=1, max_size=20, alphabet="abcdefghijklmnopqrstuvwxyz_"),
    values=json_primitives,
    min_size=1,
    max_size=5,
)
```

To:

```python
# Valid TransformErrorReason requires "reason" field with Literal-typed value
# Use a subset of common error categories for property testing
_test_error_categories = [
    "api_error",
    "missing_field",
    "validation_failed",
    "test_error",
    "property_test_error",
]

error_reasons = st.fixed_dictionaries(
    {"reason": st.sampled_from(_test_error_categories)},
    optional={
        "error": st.text(min_size=1, max_size=100),
        "field": st.text(min_size=1, max_size=50, alphabet="abcdefghijklmnopqrstuvwxyz_"),
        "status_code": st.integers(min_value=100, max_value=599),
        "query": st.text(min_size=1, max_size=50),
    },
)
```

### Step 2: Update test\_serialization\_properties.py

Find tests using `reason` parameter (around lines 283-305) and ensure they use the updated strategy.

If there's a local strategy definition, update it similarly:

```python
# If defined locally, change to:
reason_dicts = st.fixed_dictionaries(
    {"reason": st.sampled_from(["test_error", "api_error", "validation_failed"])},
    optional={
        "error": st.text(min_size=0, max_size=100),
        "field": st.text(min_size=0, max_size=50),
    },
)
```

### Step 3: Check conftest.py for shared strategies

Search for error reason strategies in `tests/property/conftest.py` and update if found.

The existing `PropertyConditionalErrorTransform` at line 293 already uses correct pattern:
```python
return TransformResult.error({"reason": "property_test_error"})
```
No change needed there.

### Step 4: Run property tests

Run: `.venv/bin/python -m pytest tests/property/ -v --hypothesis-seed=0`
Expected: All tests pass

### Step 5: Commit

```bash
git add tests/property/
git commit -m "$(cat <<'EOF'
test(property): update error_reasons strategy for TransformErrorReason

Property test strategies now generate valid TransformErrorReason dicts
with required 'reason' field using Literal-typed values.

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 7: Update Test Fixtures (Comprehensive Audit)

**Scope:** All test files with `TransformResult.error()` calls (~50 files, ~100 call sites).

### Step 0: Generate comprehensive fixture list

Run this command to find ALL test fixtures needing review:

```bash
grep -rn "TransformResult\.error(" tests/ --include='*.py' | grep -v ".pyc" | cut -d: -f1 | sort -u > /tmp/error_fixture_files.txt
cat /tmp/error_fixture_files.txt
```

### Step 1: Fix engine test files

**tests/engine/test_processor_quarantine.py** (lines 79, 177):

Change from `{"message": "..."}` to:
```python
# Line 79
{"reason": "validation_failed", "message": "negative values not allowed"}

# Line 177
{"reason": "validation_failed", "message": "missing required_field"}
```

**tests/engine/test_integration.py** (lines 3437, 3555):

Change from:
```python
return TransformResult.error({"message": "Even values fail", "value": row["value"]})
```
To:
```python
return TransformResult.error({"reason": "validation_failed", "message": "Even values fail"})
```

**tests/engine/test_aggregation_audit.py** (line 81):

Change from:
```python
return TransformResult.error({"message": "batch processing failed", "code": "BATCH_ERROR"})
```
To:
```python
return TransformResult.error({"reason": "batch_error", "message": "batch processing failed"})
```

**tests/engine/test_processor_core.py** (line 254):

Change from:
```python
return TransformResult.error({"message": "negative values not allowed"})
```
To:
```python
return TransformResult.error({"reason": "validation_failed", "message": "negative values not allowed"})
```

**tests/engine/test_transform_executor.py** (lines 107, 365, 436):

```python
# Line 107
return TransformResult.error({"reason": "validation_failed", "message": "validation failed"})

# Line 365
return TransformResult.error({"reason": "validation_failed", "message": "invalid input"})

# Line 436
return TransformResult.error({"reason": "validation_failed", "message": "routing to error sink"})
```

### Step 2: Fix contract test files

**tests/contracts/test_results.py** (line 120):

Change from:
```python
error = TransformResult.error({"e": "msg"})
```
To:
```python
error = TransformResult.error({"reason": "test_error", "error": "msg"})
```

**tests/plugins/test_results.py** (lines 96-97):

Change from:
```python
result = TransformResult.error(
    reason={"action": "validation", "fields_modified": ["value"]},
```
To:
```python
result = TransformResult.error(
    reason={"reason": "validation_failed", "field": "value"},
```

### Step 3: Audit remaining test files

Review each file from Step 0 and ensure all `TransformResult.error()` calls have:
1. A `reason` field with value from `TransformErrorCategory`
2. Consistent naming (`*_failed` pattern preferred)

**Additional files to check:**

- `tests/engine/test_batch_adapter.py:225`
- `tests/engine/test_executors.py:373`
- `tests/engine/test_processor_telemetry.py:85`
- `tests/engine/test_orchestrator_errors.py:275`
- `tests/engine/test_processor_outcomes.py:530`
- `tests/engine/test_audit_sweep.py:384,723`
- `tests/engine/test_coalesce_integration.py:828`
- `tests/engine/test_aggregation_integration.py:1335,1669,1837,2010,2169`
- `tests/engine/test_transform_error_routing.py:123,187,246,298,384,520,581`
- `tests/system/recovery/test_crash_recovery.py:77,464`
- `tests/plugins/llm/test_pooled_executor.py:298`

Most of these already use `{"reason": "..."}` pattern but verify each one.

### Step 4: Run full test suite

Run: `.venv/bin/python -m pytest tests/ -v --tb=short -x`
Expected: PASS

### Step 5: Commit

```bash
git add tests/
git commit -m "$(cat <<'EOF'
test: update all fixtures to use valid TransformErrorReason

Comprehensive audit of ~50 test files with TransformResult.error() calls.
All error reason dicts now have required 'reason' field with
Literal-typed value from TransformErrorCategory.

Fixed outliers using 'message' without 'reason' field.
Standardized on '*_failed' naming pattern where appropriate.

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 8: Final Verification

### Step 1: Run full test suite

Run: `.venv/bin/python -m pytest tests/ -v`
Expected: PASS

### Step 2: Run mypy on affected files

Run:

```bash
.venv/bin/python -m mypy \
    src/elspeth/contracts/errors.py \
    src/elspeth/contracts/results.py \
    src/elspeth/core/landscape/recorder.py \
    src/elspeth/plugins/transforms/field_mapper.py \
    --no-error-summary
```

Expected: No errors

### Step 3: Run mypy on tests (verify Literal validation)

Run:

```bash
.venv/bin/python -m mypy tests/contracts/test_errors.py tests/contracts/test_results.py --no-error-summary
```

Expected: No errors (validates that Literal typing works in tests)

### Step 4: Run ruff

Run: `.venv/bin/python -m ruff check src/elspeth/contracts/ src/elspeth/core/landscape/recorder.py`
Expected: No errors

### Step 5: Close bead

```bash
bd close elspeth-rapid-q6s --reason="Implemented TransformErrorReason TypedDict with Literal-typed reason field, nested TypedDicts, and comprehensive test coverage. All tests pass."
```

---

## Summary

### Type Definitions

| Type | Purpose |
|------|---------|
| `TransformErrorCategory` | Literal type with all valid error category strings |
| `TransformErrorReason` | Main TypedDict for error reasons |
| `TemplateErrorEntry` | Nested TypedDict for batch template errors |
| `RowErrorEntry` | Nested TypedDict for batch row errors |
| `UsageStats` | Nested TypedDict for LLM token usage |

### Key Fields

| Field | Type | Purpose |
|-------|------|---------|
| `reason` | `TransformErrorCategory` (REQUIRED) | Error category (Literal-typed for compile-time validation) |
| `error` | `NotRequired[str]` | Exception message or detailed description |
| `field` | `NotRequired[str]` | Field name for field-related errors |
| `error_type` | `NotRequired[str]` | Sub-category (e.g., "http_error", "network_error") |
| `query` | `NotRequired[str]` | Which query failed in multi-query |
| `template_errors` | `NotRequired[list[TemplateErrorEntry]]` | Batch template failures (nested TypedDict) |
| `row_errors` | `NotRequired[list[RowErrorEntry]]` | Batch row failures (nested TypedDict) |
| `usage` | `NotRequired[UsageStats]` | LLM token usage (nested TypedDict) |
| ... | ... | (25+ additional optional fields) |

### Key Design Decisions

1. **Literal-typed `reason` field** - Compile-time validation prevents typos in error categories
2. **Nested TypedDicts** - `template_errors`, `row_errors`, `usage` have structured types instead of `dict[str, Any]`
3. **Simplified unions** - Removed `str | None` where field absence is sufficient
4. **Comprehensive test coverage** - 15+ test cases covering all usage patterns
5. **Property test updates** - Hypothesis strategies generate valid TypedDict structures

### Files Changed

| File | Change |
|------|--------|
| `src/elspeth/contracts/errors.py` | Add TypedDict and nested types |
| `src/elspeth/contracts/__init__.py` | Export all new types |
| `src/elspeth/contracts/results.py` | Update signature |
| `src/elspeth/core/landscape/recorder.py` | Update signature |
| `src/elspeth/plugins/transforms/field_mapper.py` | Fix outlier |
| `tests/contracts/test_errors.py` | Add comprehensive tests |
| `tests/property/*.py` | Update Hypothesis strategies |
| `tests/engine/*.py` | Fix ~20 fixture files |
| `tests/plugins/*.py` | Fix ~5 fixture files |
| `tests/system/*.py` | Fix ~2 fixture files |

### Review Board Amendments Applied

| Amendment | Status |
|-----------|--------|
| P0: Add Task 6.5 for property tests | ✅ Added |
| P0: Expand Task 7 fixture audit | ✅ Expanded to ~50 files |
| P1: Add edge case tests to Task 6 | ✅ Added 15+ test cases |
| P1: Add Literal type for reason | ✅ TransformErrorCategory |
| P2: Define nested TypedDicts | ✅ TemplateErrorEntry, RowErrorEntry, UsageStats |
| Simplify str \| None unions | ✅ Removed where appropriate |
| Standardize test naming | ✅ Using `*_failed` pattern |
