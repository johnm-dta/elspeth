# Type Soup Follow-up Cleanup Plan

**Status:** ✅ IMPLEMENTED (2026-02-01)

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Complete type safety improvements identified during review of RoutingReason and TransformErrorReason plans. Address remaining union type ambiguity, dead code, and redundant optionality patterns.

**Prerequisites:** RoutingReason (elspeth-rapid-5vc) and TransformErrorReason (elspeth-rapid-q6s) plans should be completed first.

**Bead:** TBD (create with `bd create --title="Type soup follow-up cleanup" --type=task --priority=3`)

---

## Implementation Summary

- Added `QueryFailureDetail` and `ErrorDetail` TypedDicts and exports (`src/elspeth/contracts/errors.py`, `src/elspeth/contracts/__init__.py`).
- RoutingReason property test strategies updated to typed variants (`tests/property/engine/test_executor_properties.py`, `tests/property/contracts/test_serialization_properties.py`).
- Error contract tests extended for new TypedDicts (`tests/contracts/test_errors.py`).

## Task 0: Commit Pending Work from Previous Plans

**Context:** There are 4 uncommitted test files from previous type cleanup work.

**Step 1: Verify uncommitted changes**

Run: `git status`
Expected: Modified files in tests/ directory

**Step 2: Run tests to ensure changes are valid**

Run: `.venv/bin/python -m pytest tests/contracts/test_results.py tests/engine/test_aggregation_audit.py tests/engine/test_processor_core.py tests/plugins/llm/test_pooled_executor.py -v`
Expected: PASS

**Step 3: Commit**

```bash
git add tests/contracts/test_results.py tests/engine/test_aggregation_audit.py tests/engine/test_processor_core.py tests/plugins/llm/test_pooled_executor.py
git commit -m "$(cat <<'EOF'
test: update fixtures to use valid TransformErrorReason

- test_results.py: Use {"reason": "test_error"} instead of {"reason": "failed"}
- test_aggregation_audit.py: Use {"reason": "batch_error"} pattern
- test_processor_core.py: Use {"reason": "validation_failed"} pattern
- test_pooled_executor.py: Use {"reason": "api_error"} pattern

All error reason dicts now have required 'reason' field with
Literal-typed value from TransformErrorCategory.

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 1: Define Nested TypedDicts for Union Fields

**Note:** TransformReason is NOT being deleted - it's being wired up properly.
See `2026-01-31-transform-success-reason.md` for that implementation plan.

**Files:**
- Modify: `src/elspeth/contracts/errors.py:276, 317`

**Context:** Two fields use `list[str] | list[dict[str, Any]]` which is confusing - it's unclear what the dict structure should be. Define proper TypedDicts for the detailed variants.

**Step 1: Add QueryFailureDetail TypedDict**

Add after `UsageStats` class (around line 120):

```python
class QueryFailureDetail(TypedDict):
    """Detailed information about a failed query in multi-query transforms.

    Used when transforms need to report more than just the query name.
    """

    query: str
    error: NotRequired[str]
    error_type: NotRequired[str]
    status_code: NotRequired[int]


class ErrorDetail(TypedDict):
    """Detailed information about an error in batch processing.

    Used when more context is needed than a simple error message string.
    """

    message: str
    error_type: NotRequired[str]
    row_index: NotRequired[int]
    details: NotRequired[str]
```

**Step 2: Update failed_queries field**

Change line 276 from:
```python
    failed_queries: NotRequired[list[str] | list[dict[str, Any]]]  # Query names or detailed failures
```

To:
```python
    failed_queries: NotRequired[list[str | QueryFailureDetail]]  # Query names or detailed failures
```

**Step 3: Update errors field**

Change line 317 from:
```python
    errors: NotRequired[list[str] | list[dict[str, Any]]]  # Error messages or structured errors
```

To:
```python
    errors: NotRequired[list[str | ErrorDetail]]  # Error messages or structured errors
```

**Step 4: Update exports**

In `src/elspeth/contracts/__init__.py`, add to imports:
```python
    ErrorDetail,
    QueryFailureDetail,
```

Add to `__all__`:
```python
    "ErrorDetail",
    "QueryFailureDetail",
```

**Step 5: Verify mypy**

Run: `.venv/bin/python -m mypy src/elspeth/contracts/errors.py --no-error-summary`
Expected: No errors

**Step 6: Add tests**

Add to `tests/contracts/test_errors.py`:

```python
class TestQueryFailureDetail:
    """Tests for QueryFailureDetail TypedDict."""

    def test_query_failure_detail_minimal(self) -> None:
        """QueryFailureDetail works with only query field."""
        from elspeth.contracts import QueryFailureDetail

        detail: QueryFailureDetail = {"query": "sentiment"}
        assert detail["query"] == "sentiment"

    def test_query_failure_detail_full(self) -> None:
        """QueryFailureDetail with all fields."""
        from elspeth.contracts import QueryFailureDetail

        detail: QueryFailureDetail = {
            "query": "sentiment",
            "error": "Rate limited",
            "error_type": "rate_limit",
            "status_code": 429,
        }
        assert detail["status_code"] == 429


class TestErrorDetail:
    """Tests for ErrorDetail TypedDict."""

    def test_error_detail_minimal(self) -> None:
        """ErrorDetail works with only message field."""
        from elspeth.contracts import ErrorDetail

        detail: ErrorDetail = {"message": "Processing failed"}
        assert detail["message"] == "Processing failed"

    def test_error_detail_full(self) -> None:
        """ErrorDetail with all fields."""
        from elspeth.contracts import ErrorDetail

        detail: ErrorDetail = {
            "message": "Row processing failed",
            "error_type": "validation_error",
            "row_index": 42,
            "details": "Field 'amount' was negative",
        }
        assert detail["row_index"] == 42
```

**Step 7: Commit**

```bash
git add src/elspeth/contracts/errors.py src/elspeth/contracts/__init__.py tests/contracts/test_errors.py
git commit -m "$(cat <<'EOF'
feat(contracts): add QueryFailureDetail and ErrorDetail TypedDicts

Replace ambiguous list[str] | list[dict[str, Any]] unions with
properly typed alternatives:
- QueryFailureDetail: query name + optional error context
- ErrorDetail: message + optional error type, row index, details

Fields updated:
- failed_queries: list[str | QueryFailureDetail]
- errors: list[str | ErrorDetail]

This maintains backwards compatibility (str still allowed) while
providing type safety for detailed error reporting.

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: Simplify Redundant Optionality

**Files:**
- Modify: `src/elspeth/contracts/errors.py:274, 284, 285, 289, 290`

**Context:** Several fields use `NotRequired[X | None]` which is redundant. `NotRequired` already means "may be absent" - adding `| None` means the field can be present-but-None, which is rarely the intended semantics. Simplify to just `NotRequired[X]`.

**Step 1: Review each field's semantics**

Check if any code actually sets these to explicit `None`:

```bash
grep -rn 'template_source.*=.*None' src/elspeth/
grep -rn 'raw_response.*=.*None' src/elspeth/
grep -rn 'raw_response_preview.*=.*None' src/elspeth/
grep -rn 'response_keys.*=.*None' src/elspeth/
grep -rn 'body_preview.*=.*None' src/elspeth/
```

If any results show intentional `= None` assignment (not just default params), keep `| None` for that field.

**Step 2: Simplify fields (if no explicit None usage)**

Change in `src/elspeth/contracts/errors.py`:

```python
# Line 274: Change
template_source: NotRequired[str | None]
# To:
template_source: NotRequired[str]

# Line 284: Change
raw_response: NotRequired[str | None]
# To:
raw_response: NotRequired[str]

# Line 285: Change
raw_response_preview: NotRequired[str | None]
# To:
raw_response_preview: NotRequired[str]

# Line 289: Change
response_keys: NotRequired[list[str] | None]
# To:
response_keys: NotRequired[list[str]]

# Line 290: Change
body_preview: NotRequired[str | None]
# To:
body_preview: NotRequired[str]
```

**Step 3: Run mypy to catch any breakage**

Run: `.venv/bin/python -m mypy src/elspeth/ --no-error-summary`

If mypy reports errors where code assigns `None` to these fields, revert those specific fields to keep `| None`.

**Step 4: Commit**

```bash
git add src/elspeth/contracts/errors.py
git commit -m "$(cat <<'EOF'
refactor(contracts): simplify redundant NotRequired[X | None] patterns

NotRequired already means "may be absent" - adding | None creates
confusing semantics where a field can be absent OR present-but-None.

Simplified fields:
- template_source: NotRequired[str]
- raw_response: NotRequired[str]
- raw_response_preview: NotRequired[str]
- response_keys: NotRequired[list[str]]
- body_preview: NotRequired[str]

If the field is not present, don't include it. If present, must have value.

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: Update RoutingReason Property Test Strategies

**Files:**
- Modify: `tests/property/engine/test_executor_properties.py:47-55`
- Modify: `tests/property/contracts/test_serialization_properties.py:62-68`

**Context:** The `reason_dicts` strategy for `RoutingAction.reason` generates arbitrary dicts, but `RoutingReason` is now a typed union of `ConfigGateReason | PluginGateReason`. Update strategies to generate valid typed variants.

**Step 1: Update test_executor_properties.py**

Replace lines 47-55:

```python
# Reason dictionaries (matching routing.py's reason field)
reason_dicts = st.one_of(
    st.none(),
    st.dictionaries(
        keys=dict_keys,
        values=json_primitives,
        min_size=0,
        max_size=5,
    ),
)
```

With:

```python
# ConfigGateReason: condition + result (from config-driven gates)
config_gate_reasons = st.fixed_dictionaries({
    "condition": st.text(min_size=1, max_size=100),
    "result": st.text(min_size=1, max_size=30),
})

# PluginGateReason: rule + matched_value + optional threshold fields
plugin_gate_reasons = st.fixed_dictionaries(
    {
        "rule": st.text(min_size=1, max_size=100),
        "matched_value": json_primitives,
    },
    optional={
        "threshold": st.floats(allow_nan=False, allow_infinity=False),
        "field": st.text(min_size=1, max_size=50, alphabet="abcdefghijklmnopqrstuvwxyz_"),
        "comparison": st.sampled_from([">", "<", ">=", "<=", "==", "!="]),
    },
)

# RoutingReason is ConfigGateReason | PluginGateReason
routing_reasons = st.one_of(
    st.none(),
    config_gate_reasons,
    plugin_gate_reasons,
)
```

Update any references from `reason_dicts` to `routing_reasons`.

**Step 2: Update test_serialization_properties.py**

Replace lines 62-68:

```python
# Reason dictionaries for RoutingAction (JSON-safe)
reason_dicts = st.dictionaries(
    keys=st.sampled_from(["condition", "threshold", "rule", "match", "reason"]),
    values=st.one_of(st.text(max_size=50), st.integers(min_value=-1000, max_value=1000), st.booleans()),
    min_size=0,
    max_size=3,
)
```

With:

```python
# ConfigGateReason: condition + result (from config-driven gates)
config_gate_reasons = st.fixed_dictionaries({
    "condition": st.text(min_size=1, max_size=50),
    "result": st.text(min_size=1, max_size=20),
})

# PluginGateReason: rule + matched_value
plugin_gate_reasons = st.fixed_dictionaries(
    {
        "rule": st.text(min_size=1, max_size=50),
        "matched_value": st.one_of(
            st.text(max_size=50),
            st.integers(min_value=-1000, max_value=1000),
            st.booleans(),
        ),
    },
    optional={
        "threshold": st.floats(min_value=-1000, max_value=1000, allow_nan=False),
        "field": st.text(min_size=1, max_size=30),
        "comparison": st.sampled_from([">", "<", ">=", "<="]),
    },
)

# RoutingReason union for property tests
routing_reasons = st.one_of(
    st.none(),
    config_gate_reasons,
    plugin_gate_reasons,
)
```

Update any references from `reason_dicts` to `routing_reasons`.

**Step 3: Run property tests**

Run: `.venv/bin/python -m pytest tests/property/ -v --hypothesis-seed=0`
Expected: PASS

**Step 4: Commit**

```bash
git add tests/property/
git commit -m "$(cat <<'EOF'
test(property): update routing_reasons strategy for typed RoutingReason

Property test strategies now generate valid RoutingReason variants:
- ConfigGateReason: condition + result
- PluginGateReason: rule + matched_value + optional threshold fields

Previously generated arbitrary dicts which could violate the
discriminated union contract.

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: Final Verification

**Step 1: Run full test suite**

Run: `.venv/bin/python -m pytest tests/ -v`
Expected: PASS

**Step 2: Run mypy**

Run: `.venv/bin/python -m mypy src/elspeth/ --no-error-summary`
Expected: No new errors

**Step 3: Run ruff**

Run: `.venv/bin/python -m ruff check src/elspeth/`
Expected: No errors

**Step 4: Verify contract exports**

Run:
```python
.venv/bin/python -c "
from elspeth.contracts import (
    ConfigGateReason,
    PluginGateReason,
    RoutingReason,
    TransformErrorReason,
    TransformErrorCategory,
    QueryFailureDetail,
    ErrorDetail,
    TemplateErrorEntry,
    RowErrorEntry,
    UsageStats,
)
print('All type contracts exported successfully')
"
```

**Step 5: Close bead**

```bash
bd close <bead-id> --reason="Completed type soup follow-up: removed dead TransformReason, added QueryFailureDetail/ErrorDetail TypedDicts, simplified redundant optionality, updated property test strategies."
```

---

## Summary

| Task | Change | Impact |
|------|--------|--------|
| Task 0 | Commit pending test updates | Completes previous plans |
| Task 1 | Add QueryFailureDetail, ErrorDetail | Type safety for union fields |
| Task 2 | Simplify NotRequired[X \| None] | Cleaner semantics |
| Task 3 | Update property test strategies | Type-safe test generation |

**Note:** TransformReason wiring is handled in separate plan: `2026-01-31-transform-success-reason.md`

**Files Changed:**
- `src/elspeth/contracts/errors.py` - Add new TypedDicts, simplify optionality
- `src/elspeth/contracts/__init__.py` - Update exports
- `tests/contracts/test_errors.py` - Add new TypedDict tests
- `tests/property/engine/test_executor_properties.py` - Update routing_reasons strategy
- `tests/property/contracts/test_serialization_properties.py` - Update routing_reasons strategy
- `tests/` (4 files) - Commit pending fixture updates

**Design Rationale:**

1. **Union refinement** - `list[str | TypedDict]` is clearer than `list[str] | list[dict]`
2. **Semantic clarity** - `NotRequired[X]` means "absent or present-with-value", not "absent or present-as-None"
3. **Property test accuracy** - Strategies should generate valid typed data, not arbitrary dicts
