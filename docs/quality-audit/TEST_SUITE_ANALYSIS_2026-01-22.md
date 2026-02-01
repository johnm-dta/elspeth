# Test Suite Analysis Report

**Date:** 2026-01-22
**Scope:** Complete test suite analysis for ELSPETH
**Methodology:** Parallel specialized agent analysis (test-suite-reviewer, coverage-gap-analyst, pyramid-analyzer, mutation-testing-reviewer)

---

## Executive Summary

| Metric | Value | Assessment |
|--------|-------|------------|
| **Total Tests** | 2,696 across 204 files | Comprehensive |
| **Total Test LOC** | ~77,452 lines | Well-invested |
| **Test Pyramid** | 72% unit / 18% integration / 4% E2E / 8% property | Excellent balance |
| **Overall Grade** | **A-** | Excellent with targeted improvements needed |
| **Critical Gaps** | 4 confirmed bugs + coverage holes | Priority fixes required |
| **Anti-patterns** | 3 high-severity, 3 medium-severity | Addressable |

### Bottom Line

The ELSPETH test suite demonstrates **exceptional testing discipline** for a high-stakes audit system. The combination of contract-based testing, property-based verification, and strict error path coverage shows a team that understands the stakes.

However, critical gaps exist that undermine confidence:

1. **Mutation testing excludes the entire engine subsystem** — 3,700+ lines of audit-critical code untested for mutation survival
2. **No property test for the core invariant** — "every row reaches exactly one terminal state" is unverified
3. **Confirmed bugs in audit trail** — source payloads not persisted, secrets leaked to artifacts
4. **511 weak assertions** — tests that pass but protect nothing

---

## Test Distribution Analysis

### Current Pyramid

```
        /\
       /  \      97 System tests (3.6%)
      /____\
     /      \    488 Integration tests (18.1%)
    /________\
   /          \  1,891 Unit tests (71.6%)
  /____________\ 220 Property tests (8.2%)
```

### Distribution by Directory

| Directory | Tests | Unit | Integration | System | Property | Assessment |
|-----------|-------|------|-------------|--------|----------|------------|
| `tests/cli/` | 77 | 79% | 21% | - | - | Good |
| `tests/contracts/` | 242 | 87% | - | - | 13% | Good |
| `tests/core/` | 687 | 69% | 31% | - | - | Good |
| `tests/engine/` | 466 | 28% | 32% | 19% | 21% | Excellent (justified) |
| `tests/integration/` | 75 | 20% | 79% | 1% | - | By design |
| `tests/plugins/` | 952 | 94% | 6% | - | 1% | Excellent |
| `tests/property/` | 79 | - | - | - | 100% | By design |
| `tests/system/` | 10 | - | - | 100% | - | By design |
| `tests/tui/` | 43 | 100% | - | - | - | Good |

**Verdict:** Pyramid is healthy. No inversion. The 28% unit rate in `tests/engine/` is justified by orchestration complexity requiring integration-level verification.

---

## Priority 0: Confirmed Bugs Affecting Audit Integrity

These findings represent **actual bugs**, not just test gaps. They were identified through coverage gap analysis and cross-referenced with existing bug reports.

### P0-1: Source Row Payloads Never Persisted

**Status:** Bug confirmed, filed as `P0-2026-01-22-source-row-payloads-never-persisted.md`

**Impact:**
- Cannot prove data lineage
- Violates "I don't know what happened is never acceptable"
- `explain()` queries cannot reconstruct source state

**Root Cause:** Source plugin records row metadata but not the actual payload content.

**Required Fix:**
1. Modify source recording to persist payload via PayloadStore
2. Add integration test verifying payload retrieval after run completion
3. Add property test for payload round-trip integrity

---

### P0-2: Artifact Descriptor Leaks Secrets

**Status:** Bug confirmed, filed as `P1-2026-01-22-artifact-descriptor-leaks-secrets.md`

**Impact:**
- Database connection strings stored in audit trail
- Non-redactable after the fact
- Security/compliance violation

**Root Cause:** Artifact descriptors serialize sink configuration without filtering sensitive fields.

**Required Fix:**
1. Apply HMAC fingerprinting to sensitive config fields before storage
2. Add test verifying no plaintext secrets in artifact descriptors
3. Audit existing audit trails for exposure

---

### P0-3: Decimal NaN/Infinity Bypasses Rejection

**Status:** Bug confirmed, filed as `P1-2026-01-22-decimal-nan-infinity-bypass-rejection.md`

**Impact:**
- Non-finite values could enter audit trail
- Hash determinism compromised (NaN != NaN)
- Silent data corruption

**Root Cause:** `canonical.py` rejects `float('nan')` and `numpy.nan` but not `Decimal('NaN')`.

**Required Fix:**
```python
# In canonical.py _normalize_for_canonical()
if isinstance(value, Decimal):
    if not value.is_finite():
        raise ValueError(f"Non-finite Decimal value: {value}")
```

**Required Test:**
```python
@given(value=st.from_type(Decimal))
@settings(max_examples=200)
def test_decimal_nan_rejected(value: Decimal) -> None:
    assume(value.is_nan() or value.is_infinite())
    with pytest.raises(ValueError, match="non-finite"):
        canonical_json({"value": value})
```

---

### P0-4: Payload Integrity Verification Untested

**Status:** Coverage gap

**Impact:**
- Corrupted/tampered payloads undetected
- Audit trail returns wrong data silently

**Root Cause:** No test exercises the corruption detection path in PayloadStore.

**Required Test:**
```python
def test_payload_tampering_detected():
    """Verify corrupted payloads raise IntegrityError."""
    store = PayloadStore(db)
    ref = store.store({"key": "value"})

    # Simulate corruption
    db.execute("UPDATE payloads SET content = 'garbage' WHERE ref = ?", ref)

    with pytest.raises(PayloadIntegrityError):
        store.retrieve(ref)
```

---

## Priority 1: High-Severity Test Quality Issues

### Issue 1.1: Sleepy Assertions in Concurrency Tests

**Severity:** High
**Location:** `tests/plugins/llm/test_pooled_executor.py`
**Lines:** 101, 167, 350

**Problem:**
```python
# Current - flaky on slow CI runners
time.sleep(0.01 * (3 - idx))  # idx 0 slowest, idx 2 fastest
time.sleep(0.05)  # Give row 0 time to release semaphore
```

**Impact:**
- CI flakiness
- False positives on fast machines
- Intermittent failures under load

**Fix:**
```python
# Use explicit synchronization
release_event = threading.Event()
# ... in concurrent code ...
release_event.set()
# ... in test ...
assert release_event.wait(timeout=1.0), "Timeout waiting for release"
```

**Effort:** 1-2 hours

---

### Issue 1.2: Weak Assertions (511 instances)

**Severity:** High
**Pattern:** `assert x is not None`
**Scope:** Throughout test suite, concentrated in:
- `tests/core/landscape/test_recorder.py`
- `tests/engine/test_orchestrator.py`
- `tests/plugins/test_manager.py`

**Problem:**
```python
# Vacuous - Python constructors don't return None
executor = CoalesceExecutor(config, db)
assert executor is not None  # What does this prove?
```

**Impact:**
- Tests pass but catch zero regressions
- False confidence in coverage
- Technical debt with compound interest

**Fix:**
```python
# Verify actual behavior
executor = CoalesceExecutor(config, db)
assert executor.pool_size == config.pool_size
assert executor.pending_count == 0
assert isinstance(executor._buffer, ReorderBuffer)
```

**Effort:** 4-6 hours (prioritize core/engine first)

---

### Issue 1.3: Over-Mocking in LLM Tests

**Severity:** High
**Location:** `tests/integration/test_llm_transforms.py`, `tests/plugins/llm/`
**Scope:** 644 mock occurrences across 30 files

**Problem:**
```python
# Deep mock chains test mock setup, not integration
mock_client = MagicMock()
mock_response = MagicMock()
mock_response.choices = [MagicMock()]
mock_response.choices[0].message.content = "Hello"
mock_response.model = "gpt-4"
mock_response.usage = MagicMock()
mock_response.usage.prompt_tokens = 10
mock_response.usage.completion_tokens = 5
mock_response.model_dump.return_value = {}
mock_client.chat.completions.create.return_value = mock_response
```

**Impact:**
- Tests pass despite API changes
- Mock maintenance burden
- False confidence

**Fix:** Use fixture-based fakes or existing `replayer.py`:
```python
@pytest.fixture
def fake_openai_response():
    """Real response structure with test data."""
    return ChatCompletionResponse(
        choices=[Choice(message=Message(content="Hello"))],
        model="gpt-4",
        usage=Usage(prompt_tokens=10, completion_tokens=5)
    )

# Or use record/replay
client = AuditedLLMClient.from_recording("fixtures/openai_chat.json")
```

**Effort:** 3-4 hours

---

### Issue 1.4: Mutation Testing Excludes Engine

**Severity:** Critical
**Location:** `pyproject.toml` lines 227-236

**Current Configuration:**
```toml
[tool.mutmut]
paths_to_mutate = "src/elspeth/core/"
runner = "python -m pytest tests/property/ tests/core/ -x --tb=no -q"
```

**Missing from Mutation Scope:**

| File | Lines | Audit Risk |
|------|-------|------------|
| `orchestrator.py` | 1,674 | Run lifecycle, DAG execution |
| `processor.py` | 1,014 | Row routing, token management |
| `executors.py` | 49KB | Transform dispatch, batch processing |
| `coalesce_executor.py` | 16KB | Fork/join merge logic |
| `retry.py` | - | Attempt recording |
| `expression_parser.py` | - | Gate condition evaluation |

**Impact:**
- 3,700+ lines of audit-critical code never mutated
- Tests exist but aren't proven effective
- Subtle bugs (e.g., `<=` vs `<`) would survive

**Fix:**
```toml
[tool.mutmut]
paths_to_mutate = [
    "src/elspeth/core/",
    "src/elspeth/engine/orchestrator.py",
    "src/elspeth/engine/processor.py",
    "src/elspeth/engine/executors.py",
    "src/elspeth/engine/coalesce_executor.py",
]
runner = "python -m pytest tests/core/ tests/engine/ tests/property/ --tb=short -v"
```

**Effort:** 1-2 hours config + 3-4 hours first mutation run

---

## Priority 2: Coverage Gaps in Critical Paths

### Gap 2.1: Missing Terminal State Property Test

**Severity:** Critical
**Impact:** Core ELSPETH invariant unverified

This is the most important missing test in the entire suite. ELSPETH's fundamental guarantee is that every row reaches exactly one terminal state.

**Required Test:**
```python
# tests/property/engine/test_terminal_state_invariant.py

from hypothesis import given, settings
from hypothesis import strategies as st

class TestTerminalStateInvariant:
    """Property: Every row reaches exactly one terminal state."""

    @given(rows=st.lists(st.dictionaries(st.text(min_size=1), json_values), min_size=1, max_size=50))
    @settings(max_examples=500)
    def test_every_row_reaches_exactly_one_terminal_state(self, rows, orchestrator, landscape):
        """No row is silently dropped or double-counted."""
        run_id = orchestrator.run(source_rows=rows)

        for row_idx in range(len(rows)):
            states = landscape.query_terminal_states(run_id=run_id, row_id=row_idx)

            assert len(states) == 1, (
                f"Row {row_idx} has {len(states)} terminal states, expected exactly 1. "
                f"States: {states}"
            )

            assert states[0].status in {
                RowStatus.COMPLETED,
                RowStatus.ROUTED,
                RowStatus.FORKED,
                RowStatus.CONSUMED_IN_BATCH,
                RowStatus.COALESCED,
                RowStatus.QUARANTINED,
                RowStatus.FAILED,
            }, f"Row {row_idx} has invalid terminal state: {states[0].status}"
```

---

### Gap 2.2: Checkpoint Recovery with Malformed JSON

**Severity:** High
**Impact:** Pipeline crashes on resume with corrupted checkpoint

**Required Test:**
```python
def test_checkpoint_recovery_handles_malformed_json():
    """Checkpoint manager must fail gracefully on corruption."""
    checkpoint_manager.save(run_id, state)

    # Corrupt the checkpoint file
    checkpoint_path = checkpoint_manager._path_for(run_id)
    checkpoint_path.write_text("{ invalid json")

    with pytest.raises(CheckpointCorruptionError) as exc_info:
        checkpoint_manager.restore(run_id)

    assert "malformed" in str(exc_info.value).lower()
    # Verify the run is marked as unrecoverable, not silently ignored
```

---

### Gap 2.3: Multi-Sink Recovery Idempotency

**Severity:** High
**Impact:** Duplicate outputs on pipeline resume

**Required Test:**
```python
def test_multi_sink_recovery_is_idempotent():
    """Resuming a multi-sink pipeline must not duplicate outputs."""
    # Run pipeline with 2 sinks until row 50, then crash
    run_id = orchestrator.run(source_rows=rows_100, stop_after=50)

    # Resume from checkpoint
    orchestrator.resume(run_id)

    # Verify each row appears exactly once per sink
    for sink_name in ["sink_a", "sink_b"]:
        outputs = sink_repository.get_outputs(run_id, sink_name)
        row_ids = [o.row_id for o in outputs]
        assert len(row_ids) == len(set(row_ids)), (
            f"Duplicate outputs in {sink_name}: {[r for r in row_ids if row_ids.count(r) > 1]}"
        )
```

---

### Gap 2.4: Empty Batch Edge Case

**Severity:** Medium
**Location:** `src/elspeth/engine/executors.py`
**Evidence:** Code comment "should not happen" with success return

**Required Test:**
```python
def test_empty_batch_returns_explicit_error():
    """Empty batches must not silently succeed."""
    aggregation = BatchAggregation(batch_size=10)

    # Force empty batch scenario
    result = aggregation.finalize_batch([])

    assert result.is_error
    assert "empty batch" in result.error_message.lower()
```

---

## Priority 3: Medium-Severity Issues

### Issue 3.1: Type Suppressions (133 instances)

**Pattern:** `# type: ignore` scattered across test files

**Example:**
```python
def _get_llm_client(self, ctx: PluginContext) -> AuditedLLMClient:
    return ctx.llm_client  # type: ignore[return-value]
```

**Impact:** Reduces type safety confidence

**Fix:** Proper fixture typing:
```python
@pytest.fixture
def mock_llm_client() -> AuditedLLMClient:
    return cast(AuditedLLMClient, MagicMock(spec=AuditedLLMClient))
```

**Effort:** Ongoing, 30 min per batch of 10

---

### Issue 3.2: Expression Parser Smoke Tests Without Assertions

**Location:** `tests/engine/test_expression_parser.py`
**Count:** 17 tests

**Current:**
```python
def test_random_characters(self) -> None:
    """Random character sequences should not crash the parser."""
    ExpressionParser("!@#$%^&*()")  # No assertion
```

**Fix:**
```python
def test_random_characters(self) -> None:
    """Random character sequences must raise ExpressionSyntaxError."""
    with pytest.raises(ExpressionSyntaxError):
        ExpressionParser("!@#$%^&*()")
```

**Effort:** 1 hour

---

### Issue 3.3: Hypothesis Profile Discoverability

**Problem:** Excellent Hypothesis configuration exists but developers may not know about it.

**Fix:** Add docstring to property test modules:
```python
"""Property-based tests for canonical JSON.

Hypothesis Profiles Available:
    HYPOTHESIS_PROFILE=ci pytest tests/property/       # 100 examples (default)
    HYPOTHESIS_PROFILE=nightly pytest tests/property/  # 1000 examples
    HYPOTHESIS_PROFILE=debug pytest tests/property/    # 10 examples, verbose
"""
```

**Effort:** 30 minutes

---

## Exemplary Practices to Preserve

The analysis identified practices that should be maintained and emulated:

### 1. Zero Test Interdependence

**Evidence:** No `@pytest.mark.order`, no shared module-level state
**Value:** Tests run in any order or parallel without breaking

### 2. Contract-Based Plugin Testing

**Location:** `tests/contracts/`
**Pattern:** Abstract base classes define tests all implementations must pass
**Value:** New plugins inherit full verification automatically

### 3. Property-Based Determinism Verification

**Location:** `tests/property/canonical/test_hash_determinism.py`
**Pattern:** 500+ random inputs verify `canonical_json(x) == canonical_json(x)`
**Value:** Mathematical confidence in audit trail integrity

### 4. Defense-in-Depth NaN Rejection

**Location:** `tests/property/canonical/test_nan_rejection.py`
**Pattern:** Multiple layers test NaN rejection at every nesting level
**Value:** Aligns with "no silent conversion" policy

### 5. Comprehensive Error Path Testing

**Evidence:** 498 uses of `pytest.raises` across 85 files
**Value:** Crashes are verified, not just avoided

### 6. Bug-Hiding Prevention Enforcement

**Location:** `tests/scripts/cicd/test_no_bug_hiding.py`
**Pattern:** Meta-tests verify the enforcement tool works
**Value:** Defensive anti-patterns caught automatically

---

## Implementation Roadmap

### Week 1: Critical Fixes (P0)

| Task | Owner | Effort | Impact |
|------|-------|--------|--------|
| Fix source payload persistence | - | 4h | P0 bug |
| Fix artifact secret leakage | - | 2h | P1 security |
| Add Decimal NaN/Infinity tests + fix | - | 2h | P1 data integrity |
| Expand mutation testing config | - | 2h | Critical coverage |

**Milestone:** All P0 bugs addressed, mutation testing covers engine

### Week 2: High-Severity Quality (P1)

| Task | Owner | Effort | Impact |
|------|-------|--------|--------|
| Replace sleep() with sync primitives | - | 2h | CI stability |
| Strengthen top 50 weak assertions | - | 4h | Test effectiveness |
| Add payload tampering test | - | 1h | Corruption detection |
| Add terminal state property test | - | 2h | Core invariant |

**Milestone:** High-severity anti-patterns resolved, core invariant verified

### Week 3: Coverage Gaps (P2)

| Task | Owner | Effort | Impact |
|------|-------|--------|--------|
| Checkpoint malformed JSON test | - | 1h | Resume reliability |
| Multi-sink recovery idempotency test | - | 2h | Duplicate prevention |
| Empty batch handling test | - | 1h | Edge case coverage |
| Simplify LLM mocks with fixtures | - | 4h | Test maintainability |

**Milestone:** All critical coverage gaps closed

### Ongoing (P3)

| Task | Cadence | Effort |
|------|---------|--------|
| Reduce `# type: ignore` count | Weekly | 30 min |
| Add pytest markers for CI optimization | Once | 1h |
| Review mutation testing results | After runs | 1h |
| Document Hypothesis profiles | Once | 30 min |

---

## Metrics and Tracking

### Before/After Targets

| Metric | Current | Target | Measurement |
|--------|---------|--------|-------------|
| Weak assertions | 511 | <50 | `grep -r "is not None" tests/ \| wc -l` |
| Type ignores | 133 | <30 | `grep -r "type: ignore" tests/ \| wc -l` |
| Mutation scope | 2,900 LOC | 6,600 LOC | mutmut paths |
| Engine mutation score | 0% (excluded) | >85% | mutmut results |
| Property test coverage | 8.2% | 10% | Test count ratio |

### Success Criteria

1. **P0 bugs resolved:** All 4 confirmed bugs fixed with regression tests
2. **Mutation testing expanded:** Engine subsystem under mutation with >85% kill rate
3. **Terminal state invariant verified:** Property test with 500+ examples passing
4. **CI stability improved:** Zero flaky test failures for 2 weeks

---

## Appendix A: Files Requiring Attention

### High-Priority Test Files

| File | Issue | Priority |
|------|-------|----------|
| `tests/plugins/llm/test_pooled_executor.py` | Sleepy assertions | P1 |
| `tests/core/landscape/test_recorder.py` | Weak assertions | P1 |
| `tests/engine/test_orchestrator.py` | Weak assertions | P1 |
| `tests/integration/test_llm_transforms.py` | Over-mocking | P1 |
| `tests/engine/test_expression_parser.py` | Missing assertions | P3 |

### High-Priority Source Files Needing Tests

| File | Gap | Priority |
|------|-----|----------|
| `src/elspeth/core/payload_store.py` | Tampering detection | P0 |
| `src/elspeth/engine/orchestrator.py` | Terminal state invariant | P1 |
| `src/elspeth/engine/processor.py` | Recovery idempotency | P2 |
| `src/elspeth/core/checkpoint/manager.py` | Malformed JSON handling | P2 |

---

## Appendix B: Mutation Testing Configuration

### Recommended Configuration

```toml
# pyproject.toml

[tool.mutmut]
paths_to_mutate = [
    "src/elspeth/core/canonical.py",
    "src/elspeth/core/landscape/",
    "src/elspeth/core/payload_store.py",
    "src/elspeth/core/checkpoint/",
    "src/elspeth/engine/orchestrator.py",
    "src/elspeth/engine/processor.py",
    "src/elspeth/engine/executors.py",
    "src/elspeth/engine/coalesce_executor.py",
    "src/elspeth/engine/tokens.py",
]

runner = "python -m pytest tests/core/ tests/engine/ tests/property/ -x --tb=short"

[tool.mutmut.settings]
# Target mutation scores by module
# canonical.py: 95%+ (hash integrity is critical)
# landscape/: 90%+ (audit trail must be bulletproof)
# engine/: 85%+ (orchestration must be reliable)
```

### Running Mutation Tests

```bash
# Full mutation run (slow, ~3-4 hours)
mutmut run

# Check results
mutmut results

# Show survived mutants (prioritize fixing these)
mutmut show surviving

# HTML report
mutmut html
```

---

## Appendix C: Property Test Template

### Terminal State Invariant Test

```python
# tests/property/engine/test_terminal_state_invariant.py
"""Property tests for ELSPETH's core row processing invariants.

These tests verify mathematical properties that must hold for all inputs,
not just example cases. They use Hypothesis to generate thousands of
random test cases.

Hypothesis Profiles:
    HYPOTHESIS_PROFILE=ci pytest tests/property/       # 100 examples
    HYPOTHESIS_PROFILE=nightly pytest tests/property/  # 1000 examples
    HYPOTHESIS_PROFILE=debug pytest tests/property/    # 10 examples, verbose
"""

from __future__ import annotations

import pytest
from hypothesis import given, settings, assume
from hypothesis import strategies as st

from elspeth.contracts.routing import RowStatus
from elspeth.engine.orchestrator import Orchestrator
from elspeth.core.landscape import LandscapeDB, LandscapeRecorder


# Strategies for generating valid row data
json_primitives = st.one_of(
    st.none(),
    st.booleans(),
    st.integers(),
    st.floats(allow_nan=False, allow_infinity=False),
    st.text(),
)

json_values = st.recursive(
    json_primitives,
    lambda children: st.one_of(
        st.lists(children, max_size=5),
        st.dictionaries(st.text(min_size=1, max_size=10), children, max_size=5),
    ),
    max_leaves=20,
)


class TestTerminalStateInvariant:
    """Every row must reach exactly one terminal state."""

    TERMINAL_STATES = frozenset({
        RowStatus.COMPLETED,
        RowStatus.ROUTED,
        RowStatus.FORKED,
        RowStatus.CONSUMED_IN_BATCH,
        RowStatus.COALESCED,
        RowStatus.QUARANTINED,
        RowStatus.FAILED,
    })

    @pytest.fixture
    def landscape_db(self) -> LandscapeDB:
        return LandscapeDB.in_memory()

    @pytest.fixture
    def orchestrator(self, landscape_db: LandscapeDB, simple_pipeline_config) -> Orchestrator:
        return Orchestrator(config=simple_pipeline_config, db=landscape_db)

    @given(rows=st.lists(
        st.dictionaries(st.text(min_size=1, max_size=20), json_values, min_size=1, max_size=10),
        min_size=1,
        max_size=50,
    ))
    @settings(max_examples=500, deadline=None)
    def test_every_row_reaches_exactly_one_terminal_state(
        self,
        rows: list[dict],
        orchestrator: Orchestrator,
        landscape_db: LandscapeDB,
    ) -> None:
        """Property: No row is silently dropped or double-counted.

        This is ELSPETH's fundamental guarantee. Every row that enters
        the pipeline must reach exactly one of the defined terminal states.
        Silent drops or duplicate processing would compromise audit integrity.
        """
        # Run the pipeline
        run_id = orchestrator.run(source_rows=rows)

        # Verify each row has exactly one terminal state
        for row_idx in range(len(rows)):
            terminal_states = landscape_db.query(
                """
                SELECT DISTINCT status FROM row_states
                WHERE run_id = ? AND row_id = ? AND is_terminal = 1
                """,
                (run_id, row_idx),
            )

            # Must have exactly one terminal state
            assert len(terminal_states) == 1, (
                f"Row {row_idx} has {len(terminal_states)} terminal states, "
                f"expected exactly 1. States: {terminal_states}"
            )

            # Terminal state must be valid
            state = RowStatus(terminal_states[0]["status"])
            assert state in self.TERMINAL_STATES, (
                f"Row {row_idx} has invalid terminal state: {state}"
            )

    @given(rows=st.lists(
        st.dictionaries(st.text(min_size=1), json_values, min_size=1),
        min_size=1,
        max_size=20,
    ))
    @settings(max_examples=200, deadline=None)
    def test_row_count_conservation(
        self,
        rows: list[dict],
        orchestrator: Orchestrator,
        landscape_db: LandscapeDB,
    ) -> None:
        """Property: Total terminal states equals input row count (accounting for forks).

        For pipelines without forking, terminal state count must equal input count.
        For pipelines with forking, we verify the parent-child relationship is preserved.
        """
        run_id = orchestrator.run(source_rows=rows)

        # Count terminal states per original row
        results = landscape_db.query(
            """
            SELECT row_id, COUNT(*) as terminal_count
            FROM row_states
            WHERE run_id = ? AND is_terminal = 1
            GROUP BY row_id
            """,
            (run_id,),
        )

        # Every input row must appear
        row_ids_with_terminals = {r["row_id"] for r in results}
        expected_row_ids = set(range(len(rows)))

        assert row_ids_with_terminals == expected_row_ids, (
            f"Missing terminal states for rows: {expected_row_ids - row_ids_with_terminals}, "
            f"Unexpected rows: {row_ids_with_terminals - expected_row_ids}"
        )
```

---

*Document generated by systematic test suite analysis on 2026-01-22*
