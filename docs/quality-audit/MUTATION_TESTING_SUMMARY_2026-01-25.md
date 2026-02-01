# Mutation Testing Summary - January 25, 2026

## Execution Details

**Date:** 2026-01-25
**Duration:** 36 hours
**Environment:** `/home/john/elspeth` (clone for uninterrupted testing)
**Mutmut Version:** 2.5.1
**Total Mutants:** 2,778

## Results

| Metric | Count | Percentage |
|--------|-------|------------|
| **Killed** | 1,542 | 55.5% |
| **Survived** | 1,139 | 41.0% |
| **Suspicious** | 97 | 3.5% |
| **Timeout** | 0 | 0.0% |

**Kill Rate:** 55.5% (Target: 85%+)

## Critical Findings

### Kill Rate Analysis

The 55.5% kill rate indicates **significant test coverage gaps**. Industry best practice for high-stakes systems is 85%+ kill rate (15% or fewer survivors).

### Survivor Hotspots

#### P0 - Critical Files (356 survivors)

| File | Survivors | Impact |
|------|-----------|--------|
| `orchestrator.py` | 206 | Pipeline orchestration, run lifecycle |
| `executors.py` | 88 | Transform execution, error routing |
| `processor.py` | 62 | Row processing, token management |

**Implication:** Core audit-critical orchestration logic has weak test assertions.

#### P1 - High Priority (93 survivors)

| File | Survivors | Impact |
|------|-----------|--------|
| `recorder.py` | 64 | Audit trail persistence |
| `coalesce_executor.py` | 29 | Fork/join merge logic |

#### P2 - Medium Priority (127 survivors)

| File | Survivors | Impact |
|------|-----------|--------|
| `exporter.py` | 88 | Audit export, HMAC signing |
| `models.py` | 39 | Dataclass field defaults |

#### P3 - Deferred (191 survivors)

| File | Survivors | Notes |
|------|-----------|-------|
| `schema.py` | 191 | SQLAlchemy table definitions - likely equivalent mutants |

## Common Survivor Patterns

### 1. Default Factory Fields (High Frequency)

```python
# Survivor example
child_tokens: list[TokenInfo] = field(default_factory=list)
```

**Why it survives:** Tests don't verify the field is actually a list vs None/dict.

**Fix:**
```python
def test_gate_result_child_tokens_default():
    result = GateResult(result=..., updated_token=...)
    assert isinstance(result.child_tokens, list)
    assert len(result.child_tokens) == 0
```

### 2. Error Message Strings (High Frequency)

```python
# Survivor example
raise ValueError(f"Expected {expected} but got {actual}")
```

**Why it survives:** Tests catch the exception but don't verify the message content.

**Fix:**
```python
with pytest.raises(ValueError, match=r"Expected .* but got .*"):
    function_that_raises()
```

### 3. Context Setting (Medium Frequency)

```python
# Survivor example
ctx.state_id = state.state_id
ctx.node_id = transform.node_id
```

**Why it survives:** Tests don't verify context attributes were set correctly.

**Fix:**
```python
def test_executor_sets_context_attributes():
    result = executor.execute(transform, token, ctx)
    assert ctx.state_id == expected_state_id
    assert ctx.node_id == transform.node_id
```

### 4. Comment Lines (Low Impact)

```python
# Survivor example
# Set state_id and node_id on context for external call recording
```

**Why it survives:** Mutating comments to empty strings doesn't affect execution.

**Action:** Can ignore or exclude comments from mutation testing.

### 5. Type Annotations (Low Impact)

```python
# Survivor example
step_in_pipeline: int
```

**Why it survives:** Python doesn't enforce type hints at runtime.

**Action:** Can ignore - these don't affect execution correctness.

## Actionable Survivor Breakdown

**Total Actionable:** 642 survivors (excluding schema.py)

**By Category:**
- Default factories: ~120 survivors
- Error messages: ~180 survivors
- Context/state setting: ~90 survivors
- Comments: ~80 survivors (can ignore)
- Validation logic: ~172 survivors

## Recommended Fix Strategy

### Phase 1: P0 Files (356 survivors)

**Week 1-2:** Focus on `orchestrator.py` (206 survivors)
- Error message assertions
- Context attribute verification
- Gate routing validation
- Schema compatibility checks

**Week 3:** `executors.py` and `processor.py` (150 survivors)
- Transform result verification
- Token state assertions
- Error routing coverage

### Phase 2: P1 Files (93 survivors)

**Week 4:** `recorder.py` and `coalesce_executor.py`
- Audit trail completeness
- Fork/join correctness

### Phase 3: P2 Files (127 survivors)

**Week 5+:** `exporter.py` and `models.py`
- Export integrity
- Dataclass field defaults

## Differential Testing Workflow

For incremental improvements, use the differential mutation testing script:

```bash
# After fixing tests for orchestrator.py
./scripts/mutmut_differential.sh HEAD~1

# Test specific file
.venv/bin/python -m mutmut run --paths-to-mutate src/elspeth/engine/orchestrator.py
```

This allows targeted verification without 36-hour full runs.

## Files Reference

- **Detailed Checklist:** `docs/analysis/MUTATION_GAPS_CHECKLIST.md` (6,542 lines)
- **Mutmut Cache:** `.mutmut-cache` (936 KB)
- **Differential Script:** `scripts/mutmut_differential.sh`

## Success Criteria

**Target Kill Rates:**
- Overall: 85%+ (currently 55.5%, gap: 29.5%)
- P0 files: 90%+ (orchestrator, executors, processor)
- P1 files: 85%+ (recorder, coalesce)
- P2 files: 80%+ (exporter, models)

**Estimated Effort:**
- P0 fixes: 80-120 hours (356 survivors × 15-20 min avg)
- P1 fixes: 25-35 hours (93 survivors)
- P2 fixes: 30-40 hours (127 survivors)

**Total:** 135-195 hours (~4-6 weeks of dedicated work)

## Next Steps

1. **Review MUTATION_GAPS_CHECKLIST.md** - Prioritize P0 files
2. **Start with orchestrator.py** - Highest survivor count
3. **Use pattern templates** (above) for common fixes
4. **Run differential tests** after each batch of fixes
5. **Track progress** - Check off items in the checklist

---

*Analysis generated from 36-hour mutation testing run on elspeth clone*
*Kill rate: 55.5% | Actionable survivors: 642*
