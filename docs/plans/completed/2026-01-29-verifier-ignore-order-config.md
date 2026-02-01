# CallVerifier `ignore_order` Configuration Implementation Plan

**Status:** ✅ IMPLEMENTED (2026-02-01)

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add configurable `ignore_order` parameter to `CallVerifier` so users can opt into order-sensitive verification.

**Architecture:** Extend `CallVerifier.__init__()` with a new `ignore_order: bool = True` keyword argument that passes through to DeepDiff. Default remains `True` (non-breaking) per review board recommendation. Comprehensive tests document both behaviors.

**Tech Stack:** Python 3.13, DeepDiff, pytest

**Related Bugs:**
- Fixes: `docs/bugs/open/core-landscape/P3-2026-01-21-verifier-ignore-order-hides-drift.md` (Phase 1)
- Enables: `docs/bugs/open/core-landscape/P3-2026-01-29-verifier-field-level-order-config.md` (Phase 2)

---

## Implementation Summary

- Added `ignore_order` parameter wiring in `CallVerifier.__init__()` and `verify()` to control DeepDiff order sensitivity (`src/elspeth/plugins/clients/verifier.py`).
- Default behavior preserved (ignore_order=True), with explicit strict option supported for order-sensitive diffs.
- Tests cover default and strict ordering semantics, including nested lists and duplicate elements (`tests/plugins/clients/test_verifier.py`).

## Pre-Implementation Checklist

Before starting:
1. Ensure you're on a clean branch: `git status`
2. Run existing tests to confirm baseline: `pytest tests/plugins/clients/test_verifier.py -v`
3. Read the current implementation: `src/elspeth/plugins/clients/verifier.py`

---

## Task 1: Add Parameter and Basic Wiring

**Files:**
- Modify: `src/elspeth/plugins/clients/verifier.py:120-137` (add parameter to `__init__`)
- Modify: `src/elspeth/plugins/clients/verifier.py:219-225` (use parameter in `verify()`)
- Test: `tests/plugins/clients/test_verifier.py`

**Step 1: Write the failing test for explicit `ignore_order=False`**

Add this test to `tests/plugins/clients/test_verifier.py` in the `TestCallVerifier` class:

```python
def test_verify_order_sensitive_when_configured(self) -> None:
    """Verifier detects order changes when ignore_order=False."""
    recorder = self._create_mock_recorder()
    request_data = {"id": 1}
    request_hash = stable_hash(request_data)

    recorded_response = {"items": ["a", "b", "c"]}
    live_response = {"items": ["c", "b", "a"]}  # Same items, different order

    mock_call = self._create_mock_call(request_hash=request_hash)
    recorder.find_call_by_request_hash.return_value = mock_call
    recorder.get_call_response_data.return_value = recorded_response

    verifier = CallVerifier(recorder, source_run_id="run_abc123", ignore_order=False)
    result = verifier.verify(
        call_type="llm",
        request_data=request_data,
        live_response=live_response,
    )

    # Should NOT match because ignore_order=False
    assert result.is_match is False
    assert result.has_differences is True
    # DeepDiff reports position changes as values_changed
    assert "values_changed" in result.differences
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/plugins/clients/test_verifier.py::TestCallVerifier::test_verify_order_sensitive_when_configured -v`

Expected: FAIL with `TypeError: CallVerifier.__init__() got an unexpected keyword argument 'ignore_order'`

**Step 3: Add `ignore_order` parameter to `__init__`**

Modify `src/elspeth/plugins/clients/verifier.py` - update the `__init__` method signature and body:

```python
def __init__(
    self,
    recorder: LandscapeRecorder,
    source_run_id: str,
    *,
    ignore_paths: list[str] | None = None,
    ignore_order: bool = True,
) -> None:
    """Initialize verifier.

    Args:
        recorder: LandscapeRecorder for looking up recorded calls
        source_run_id: The run_id containing baseline recordings
        ignore_paths: Paths to ignore in comparison (e.g., ["root['latency']"])
                     These paths will be excluded from DeepDiff comparison.
        ignore_order: If True (default), list ordering differences are ignored.
                     If False, list elements must appear in the same order to match.
                     Set to False for order-sensitive data like ranked results.
    """
    self._recorder = recorder
    self._source_run_id = source_run_id
    self._ignore_paths = ignore_paths or []
    self._ignore_order = ignore_order
    self._report = VerificationReport()
    # Sequence counter: (call_type, request_hash) -> next_index
    # Tracks how many times we've seen each unique request
    # Uses defaultdict to avoid .get() which can hide key bugs
    self._sequence_counters: defaultdict[tuple[str, str], int] = defaultdict(int)
```

**Step 4: Update `verify()` to use the parameter**

Modify the DeepDiff call in `verify()` method (around line 220):

```python
# Compare using DeepDiff
diff = DeepDiff(
    recorded_response,
    live_response,
    ignore_order=self._ignore_order,
    exclude_paths=self._ignore_paths,
)
```

**Step 5: Run test to verify it passes**

Run: `pytest tests/plugins/clients/test_verifier.py::TestCallVerifier::test_verify_order_sensitive_when_configured -v`

Expected: PASS

**Step 6: Run all verifier tests to check for regressions**

Run: `pytest tests/plugins/clients/test_verifier.py -v`

Expected: All tests PASS (existing `test_verify_order_independent_comparison` should still pass since default is `True`)

**Step 7: Commit**

```bash
git add src/elspeth/plugins/clients/verifier.py tests/plugins/clients/test_verifier.py
git commit -m "feat(verifier): add ignore_order parameter for order-sensitive verification

Add configurable ignore_order parameter to CallVerifier.__init__().
Default remains True (non-breaking) to preserve existing behavior.
Users can now set ignore_order=False to detect list order changes.

Partially addresses P3-2026-01-21-verifier-ignore-order-hides-drift.

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
```

---

## Task 2: Test Duplicate Elements Behavior

**Files:**
- Test: `tests/plugins/clients/test_verifier.py`

**Step 1: Write test for duplicate element handling**

Add this test to document multiset behavior:

```python
def test_ignore_order_handles_duplicate_elements(self) -> None:
    """List comparisons treat lists as multisets when ignore_order=True."""
    recorder = self._create_mock_recorder()
    request_data = {"id": 1}
    request_hash = stable_hash(request_data)

    recorded_response = {"tags": ["a", "a", "b"]}
    live_response = {"tags": ["b", "a", "a"]}  # Same multiset, different order

    mock_call = self._create_mock_call(request_hash=request_hash)
    recorder.find_call_by_request_hash.return_value = mock_call
    recorder.get_call_response_data.return_value = recorded_response

    # With ignore_order=True (default): should match (same multiset)
    verifier_loose = CallVerifier(recorder, source_run_id="run_abc123", ignore_order=True)
    result_loose = verifier_loose.verify(
        call_type="llm",
        request_data=request_data,
        live_response=live_response,
    )
    assert result_loose.is_match is True

    # With ignore_order=False: should NOT match (different positions)
    verifier_strict = CallVerifier(recorder, source_run_id="run_abc123", ignore_order=False)
    result_strict = verifier_strict.verify(
        call_type="llm",
        request_data=request_data,
        live_response=live_response,
    )
    assert result_strict.is_match is False
```

**Step 2: Run test to verify it passes**

Run: `pytest tests/plugins/clients/test_verifier.py::TestCallVerifier::test_ignore_order_handles_duplicate_elements -v`

Expected: PASS

**Step 3: Commit**

```bash
git add tests/plugins/clients/test_verifier.py
git commit -m "test(verifier): add duplicate element handling test

Documents that ignore_order=True treats lists as multisets
where duplicate counts matter but positions don't.

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
```

---

## Task 3: Test Nested List Behavior

**Files:**
- Test: `tests/plugins/clients/test_verifier.py`

**Step 1: Write test for nested list ordering**

Add this test to document recursive behavior:

```python
def test_ignore_order_applies_recursively_to_nested_lists(self) -> None:
    """Document that ignore_order affects ALL list levels recursively."""
    recorder = self._create_mock_recorder()
    request_data = {"id": 1}
    request_hash = stable_hash(request_data)

    recorded_response = {
        "results": [
            {"id": 1, "tags": ["a", "b"]},
            {"id": 2, "tags": ["x", "y"]},
        ]
    }
    live_response = {
        "results": [
            {"id": 2, "tags": ["y", "x"]},  # Both levels reordered
            {"id": 1, "tags": ["b", "a"]},
        ]
    }

    mock_call = self._create_mock_call(request_hash=request_hash)
    recorder.find_call_by_request_hash.return_value = mock_call
    recorder.get_call_response_data.return_value = recorded_response

    # With ignore_order=True: matches (recursive order-independence)
    verifier_loose = CallVerifier(recorder, source_run_id="run_abc123", ignore_order=True)
    result_loose = verifier_loose.verify(
        call_type="llm",
        request_data=request_data,
        live_response=live_response,
    )
    assert result_loose.is_match is True

    # With ignore_order=False: does NOT match
    verifier_strict = CallVerifier(recorder, source_run_id="run_abc123", ignore_order=False)
    result_strict = verifier_strict.verify(
        call_type="llm",
        request_data=request_data,
        live_response=live_response,
    )
    assert result_strict.is_match is False
```

**Step 2: Run test to verify it passes**

Run: `pytest tests/plugins/clients/test_verifier.py::TestCallVerifier::test_ignore_order_applies_recursively_to_nested_lists -v`

Expected: PASS

**Step 3: Commit**

```bash
git add tests/plugins/clients/test_verifier.py
git commit -m "test(verifier): document nested list ordering behavior

Shows that ignore_order applies recursively to all nested lists.
This is important context for users choosing between modes.

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
```

---

## Task 4: Test Dict Key Ordering Unaffected

**Files:**
- Test: `tests/plugins/clients/test_verifier.py`

**Step 1: Write test confirming dict keys are always order-independent**

```python
def test_ignore_order_does_not_affect_dict_keys(self) -> None:
    """Dict key ordering is always ignored (JSON semantics)."""
    recorder = self._create_mock_recorder()
    request_data = {"id": 1}
    request_hash = stable_hash(request_data)

    # Python dicts maintain insertion order, but JSON treats them as unordered
    recorded_response = {"z": 1, "a": 2, "m": 3}
    live_response = {"a": 2, "m": 3, "z": 1}  # Same keys/values, different order

    mock_call = self._create_mock_call(request_hash=request_hash)
    recorder.find_call_by_request_hash.return_value = mock_call
    recorder.get_call_response_data.return_value = recorded_response

    # Both with and without ignore_order: dicts should match
    verifier_loose = CallVerifier(recorder, source_run_id="run_abc123", ignore_order=True)
    result_loose = verifier_loose.verify(
        call_type="llm",
        request_data=request_data,
        live_response=live_response,
    )
    assert result_loose.is_match is True

    verifier_strict = CallVerifier(recorder, source_run_id="run_abc123", ignore_order=False)
    result_strict = verifier_strict.verify(
        call_type="llm",
        request_data=request_data,
        live_response=live_response,
    )
    assert result_strict.is_match is True
```

**Step 2: Run test to verify it passes**

Run: `pytest tests/plugins/clients/test_verifier.py::TestCallVerifier::test_ignore_order_does_not_affect_dict_keys -v`

Expected: PASS

**Step 3: Commit**

```bash
git add tests/plugins/clients/test_verifier.py
git commit -m "test(verifier): confirm dict key ordering unaffected by ignore_order

Documents that ignore_order only affects list ordering, not dict keys.
Dict comparison follows JSON semantics (unordered) regardless of setting.

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
```

---

## Task 5: Test Empty Lists

**Files:**
- Test: `tests/plugins/clients/test_verifier.py`

**Step 1: Write edge case test for empty lists**

```python
def test_empty_lists_always_match(self) -> None:
    """Empty lists match regardless of ignore_order setting."""
    recorder = self._create_mock_recorder()
    request_data = {"id": 1}
    request_hash = stable_hash(request_data)

    recorded_response = {"items": []}
    live_response = {"items": []}

    mock_call = self._create_mock_call(request_hash=request_hash)
    recorder.find_call_by_request_hash.return_value = mock_call
    recorder.get_call_response_data.return_value = recorded_response

    # Both settings should match
    for ignore_order in [True, False]:
        verifier = CallVerifier(recorder, source_run_id="run_abc123", ignore_order=ignore_order)
        result = verifier.verify(
            call_type="llm",
            request_data=request_data,
            live_response=live_response,
        )
        assert result.is_match is True, f"Failed with ignore_order={ignore_order}"
```

**Step 2: Run test to verify it passes**

Run: `pytest tests/plugins/clients/test_verifier.py::TestCallVerifier::test_empty_lists_always_match -v`

Expected: PASS

**Step 3: Commit**

```bash
git add tests/plugins/clients/test_verifier.py
git commit -m "test(verifier): add empty list edge case test

Confirms empty lists always match regardless of ignore_order.

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
```

---

## Task 6: Test Realistic LLM Response Shape

**Files:**
- Test: `tests/plugins/clients/test_verifier.py`

**Step 1: Write test with realistic LLM tool call structure**

```python
def test_order_sensitivity_with_realistic_llm_response(self) -> None:
    """Verify order handling with actual LLM tool call structure."""
    recorder = self._create_mock_recorder()
    request_data = {"model": "gpt-4", "messages": [{"role": "user", "content": "test"}]}
    request_hash = stable_hash(request_data)

    recorded_response = {
        "choices": [{
            "message": {
                "tool_calls": [
                    {"id": "call_1", "function": {"name": "search", "arguments": "{}"}},
                    {"id": "call_2", "function": {"name": "summarize", "arguments": "{}"}},
                    {"id": "call_3", "function": {"name": "respond", "arguments": "{}"}},
                ]
            }
        }]
    }
    live_response = {
        "choices": [{
            "message": {
                "tool_calls": [
                    {"id": "call_2", "function": {"name": "summarize", "arguments": "{}"}},
                    {"id": "call_1", "function": {"name": "search", "arguments": "{}"}},
                    {"id": "call_3", "function": {"name": "respond", "arguments": "{}"}},
                ]
            }
        }]
    }

    mock_call = self._create_mock_call(request_hash=request_hash)
    recorder.find_call_by_request_hash.return_value = mock_call
    recorder.get_call_response_data.return_value = recorded_response

    # With ignore_order=True (default): matches despite tool call reordering
    verifier_loose = CallVerifier(recorder, source_run_id="run_abc123", ignore_order=True)
    result_loose = verifier_loose.verify(
        call_type="llm",
        request_data=request_data,
        live_response=live_response,
    )
    assert result_loose.is_match is True

    # With ignore_order=False: tool call reordering is detected as drift
    verifier_strict = CallVerifier(recorder, source_run_id="run_abc123", ignore_order=False)
    result_strict = verifier_strict.verify(
        call_type="llm",
        request_data=request_data,
        live_response=live_response,
    )
    assert result_strict.is_match is False, "Tool call reordering should be detected"
```

**Step 2: Run test to verify it passes**

Run: `pytest tests/plugins/clients/test_verifier.py::TestCallVerifier::test_order_sensitivity_with_realistic_llm_response -v`

Expected: PASS

**Step 3: Commit**

```bash
git add tests/plugins/clients/test_verifier.py
git commit -m "test(verifier): add realistic LLM tool call ordering test

Shows practical example where tool call order matters semantically.
Demonstrates why ignore_order=False is important for LLM verification.

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
```

---

## Task 7: Update Existing Test Name for Clarity

**Files:**
- Modify: `tests/plugins/clients/test_verifier.py:488-509`

**Step 1: Rename existing test to be explicit about configuration**

Find the existing `test_verify_order_independent_comparison` test and update its name and docstring:

```python
def test_verify_order_independent_with_default_config(self) -> None:
    """Default configuration (ignore_order=True) ignores list ordering."""
    recorder = self._create_mock_recorder()
    request_data = {"id": 1}
    request_hash = stable_hash(request_data)

    recorded_response = {"items": ["a", "b", "c"]}
    live_response = {"items": ["c", "a", "b"]}  # Same items, different order

    mock_call = self._create_mock_call(request_hash=request_hash)
    recorder.find_call_by_request_hash.return_value = mock_call
    recorder.get_call_response_data.return_value = recorded_response

    verifier = CallVerifier(recorder, source_run_id="run_abc123")
    result = verifier.verify(
        call_type="llm",
        request_data=request_data,
        live_response=live_response,
    )

    # Should match because ignore_order=True by default
    assert result.is_match is True
```

**Step 2: Run test to verify it passes**

Run: `pytest tests/plugins/clients/test_verifier.py::TestCallVerifier::test_verify_order_independent_with_default_config -v`

Expected: PASS

**Step 3: Run all tests to confirm no regressions**

Run: `pytest tests/plugins/clients/test_verifier.py -v`

Expected: All PASS

**Step 4: Commit**

```bash
git add tests/plugins/clients/test_verifier.py
git commit -m "refactor(test): rename order test to clarify default behavior

Makes it explicit that order-independence is the default configuration,
not an inherent property of the verifier.

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
```

---

## Task 8: Update Bug Report Status

**Files:**
- Modify: `docs/bugs/open/core-landscape/P3-2026-01-21-verifier-ignore-order-hides-drift.md`

**Step 1: Add Phase 1 completion note to the bug report**

Add this section at the end of the file:

```markdown
## Phase 1 Implementation (2026-01-29)

**Status: IMPLEMENTED**

Phase 1 adds the `ignore_order` parameter to `CallVerifier` with default `True` (non-breaking).

### Changes Made

1. Added `ignore_order: bool = True` parameter to `CallVerifier.__init__()`
2. Parameter passes through to DeepDiff comparison
3. Default preserves existing behavior (order-independent)
4. Users can now set `ignore_order=False` for order-sensitive verification

### Test Coverage Added

- `test_verify_order_sensitive_when_configured` - explicit False behavior
- `test_ignore_order_handles_duplicate_elements` - multiset semantics
- `test_ignore_order_applies_recursively_to_nested_lists` - recursive behavior
- `test_ignore_order_does_not_affect_dict_keys` - dict key ordering unaffected
- `test_empty_lists_always_match` - edge case
- `test_order_sensitivity_with_realistic_llm_response` - practical LLM example
- `test_verify_order_independent_with_default_config` - renamed from original

### Next Steps

Phase 2 (P3-2026-01-29-verifier-field-level-order-config.md) will add field-level configuration to allow mixed semantics within a single verification session.
```

**Step 2: Commit**

```bash
git add docs/bugs/open/core-landscape/P3-2026-01-21-verifier-ignore-order-hides-drift.md
git commit -m "docs(bugs): mark verifier ignore_order Phase 1 as implemented

Adds implementation notes and test coverage summary.
Links to Phase 2 ticket for field-level configuration.

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
```

---

## Task 9: Final Verification

**Step 1: Run full test suite for verifier**

Run: `pytest tests/plugins/clients/test_verifier.py -v`

Expected: All tests PASS

**Step 2: Run mypy type check**

Run: `.venv/bin/python -m mypy src/elspeth/plugins/clients/verifier.py`

Expected: No errors

**Step 3: Run ruff linter**

Run: `.venv/bin/python -m ruff check src/elspeth/plugins/clients/verifier.py`

Expected: No errors

**Step 4: Review git log**

Run: `git log --oneline -10`

Expected: See 8 commits from this implementation

---

## Summary

This plan implements Phase 1 of the `ignore_order` fix:

| Task | Description | Tests Added |
|------|-------------|-------------|
| 1 | Add parameter and wiring | 1 |
| 2 | Duplicate elements | 1 |
| 3 | Nested lists | 1 |
| 4 | Dict keys | 1 |
| 5 | Empty lists | 1 |
| 6 | Realistic LLM | 1 |
| 7 | Rename existing | 0 (rename) |
| 8 | Update bug report | 0 (docs) |
| 9 | Final verification | 0 (validation) |

**Total new tests: 6**
**Total commits: 8**
