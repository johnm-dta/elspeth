# Test Suite Uplift Action Plan

**Reference:** [TEST_SUITE_ANALYSIS_2026-01-22.md](TEST_SUITE_ANALYSIS_2026-01-22.md)
**Status:** Ready for execution
**Estimated Total Effort:** 30-40 hours over 3 weeks

---

## Quick Reference: What to Fix

### This Week (P0 - Critical)

- [ ] **Fix source payload persistence bug** — `src/elspeth/core/landscape/recorder.py`
- [ ] **Fix artifact secret leakage** — `src/elspeth/engine/artifacts.py`
- [ ] **Add Decimal NaN/Infinity rejection** — `src/elspeth/core/canonical.py`
- [ ] **Expand mutation testing to engine** — `pyproject.toml`

### Next Week (P1 - High)

- [ ] **Replace sleep() with sync primitives** — `tests/plugins/llm/test_pooled_executor.py`
- [ ] **Add terminal state property test** — `tests/property/engine/` (new)
- [ ] **Strengthen weak assertions** — Start with `tests/core/`, `tests/engine/`
- [ ] **Add payload tampering test** — `tests/core/test_payload_store.py`

### Following Week (P2 - Medium)

- [ ] **Checkpoint malformed JSON test** — `tests/core/checkpoint/`
- [ ] **Multi-sink recovery idempotency test** — `tests/engine/`
- [ ] **Simplify LLM mocks** — `tests/integration/test_llm_transforms.py`
- [ ] **Empty batch handling test** — `tests/engine/test_executors.py`

---

## Detailed Task Breakdown

### Task 1: Fix Source Payload Persistence (P0)

**Problem:** Source payloads are never persisted, breaking audit lineage.

**Files to modify:**
- `src/elspeth/core/landscape/recorder.py` — Add payload storage call
- `src/elspeth/core/payload_store.py` — Verify store() handles source payloads

**Test to add:**
```python
# tests/core/landscape/test_recorder_payloads.py

def test_source_payload_persisted():
    """Source row payloads must be stored for audit lineage."""
    recorder = LandscapeRecorder(db)
    payload = {"user_id": 123, "action": "login"}

    recorder.record_source_row(run_id, row_id=0, payload=payload)

    # Verify payload can be retrieved
    stored = recorder.get_source_payload(run_id, row_id=0)
    assert stored == payload
```

**Acceptance criteria:**
- [ ] `explain(run_id, row_id)` returns source payload
- [ ] Property test verifies round-trip for random payloads
- [ ] Existing tests still pass

---

### Task 2: Fix Artifact Secret Leakage (P1)

**Problem:** Database credentials stored in artifact descriptors.

**Files to modify:**
- `src/elspeth/engine/artifacts.py` — Apply HMAC fingerprinting
- `src/elspeth/core/security/fingerprint.py` — Extend to sink configs

**Test to add:**
```python
# tests/engine/test_artifacts_security.py

def test_artifact_descriptor_redacts_secrets():
    """Sink credentials must be fingerprinted, not stored plaintext."""
    config = DatabaseSinkConfig(
        connection_string="postgresql://user:password123@host/db"
    )

    descriptor = ArtifactDescriptor.from_sink_config(config)

    assert "password123" not in str(descriptor)
    assert "hmac:" in descriptor.connection_fingerprint
```

**Acceptance criteria:**
- [ ] No plaintext secrets in artifact descriptors
- [ ] Fingerprints are deterministic (same secret → same fingerprint)
- [ ] Existing artifact tests pass

---

### Task 3: Add Decimal NaN/Infinity Rejection (P1)

**Problem:** `Decimal('NaN')` and `Decimal('Infinity')` bypass rejection.

**File to modify:**
- `src/elspeth/core/canonical.py` — Add Decimal check in `_normalize_for_canonical()`

**Code change:**
```python
# In _normalize_for_canonical()
if isinstance(value, Decimal):
    if not value.is_finite():
        raise ValueError(f"Non-finite Decimal value not allowed: {value}")
    return float(value)  # or str(value) for precision
```

**Test to add:**
```python
# tests/property/canonical/test_decimal_rejection.py

@given(st.sampled_from([Decimal('NaN'), Decimal('Infinity'), Decimal('-Infinity')]))
def test_decimal_non_finite_rejected(value):
    with pytest.raises(ValueError, match="non-finite"):
        canonical_json({"value": value})

@given(st.decimals(allow_nan=False, allow_infinity=False))
def test_decimal_finite_accepted(value):
    result = canonical_json({"value": value})
    assert isinstance(result, str)
```

**Acceptance criteria:**
- [ ] `Decimal('NaN')` raises ValueError
- [ ] `Decimal('Infinity')` raises ValueError
- [ ] Finite Decimals still work
- [ ] Property tests pass with 500+ examples

---

### Task 4: Expand Mutation Testing (Critical)

**Problem:** Engine subsystem excluded from mutation testing.

**File to modify:** `pyproject.toml`

**Change:**
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

**Validation:**
```bash
# Verify config is valid
mutmut run --paths-to-mutate src/elspeth/engine/orchestrator.py --max-workers 1

# Full run (schedule overnight)
mutmut run
```

**Acceptance criteria:**
- [ ] mutmut runs without config errors
- [ ] Engine files appear in mutation report
- [ ] Baseline mutation score recorded

---

### Task 5: Replace Sleep with Sync Primitives (P1)

**Problem:** `time.sleep()` causes CI flakiness.

**File to modify:** `tests/plugins/llm/test_pooled_executor.py`

**Lines to fix:** 101, 167, 350

**Pattern:**
```python
# Before
time.sleep(0.05)
assert executor.pending_count == 0

# After
release_complete = threading.Event()

def on_release():
    release_complete.set()

executor.on_release_callback = on_release
# ... trigger release ...
assert release_complete.wait(timeout=2.0), "Release did not complete"
assert executor.pending_count == 0
```

**Acceptance criteria:**
- [ ] No `time.sleep()` in concurrency tests
- [ ] Tests pass reliably on slow CI runners
- [ ] 100 consecutive test runs without flakiness

---

### Task 6: Add Terminal State Property Test (P1)

**Problem:** Core ELSPETH invariant unverified.

**New file:** `tests/property/engine/test_terminal_state_invariant.py`

**Content:** See Appendix C in main analysis document.

**Acceptance criteria:**
- [ ] Test passes with 500 random inputs
- [ ] Test fails if terminal state logic is broken (verify with temporary bug)
- [ ] Hypothesis finds edge cases (check statistics)

---

### Task 7: Strengthen Weak Assertions (P1)

**Problem:** 511 instances of `assert x is not None`.

**Priority files:**
1. `tests/core/landscape/test_recorder.py`
2. `tests/engine/test_orchestrator.py`
3. `tests/engine/test_processor.py`
4. `tests/plugins/test_manager.py`

**Pattern:**
```python
# Before
recorder = LandscapeRecorder(db)
assert recorder is not None

# After
recorder = LandscapeRecorder(db)
assert recorder.db is db
assert recorder.run_id is None  # Not started yet
```

**Tracking:**
```bash
# Count remaining weak assertions
grep -r "is not None" tests/ | wc -l
# Target: <50
```

**Acceptance criteria:**
- [ ] Top 50 weak assertions strengthened
- [ ] Total count reduced from 511 to <200
- [ ] No new weak assertions introduced (add to code review checklist)

---

### Task 8: Add Payload Tampering Detection Test (P1)

**Problem:** Corruption detection path never tested.

**File to modify:** `tests/core/test_payload_store.py`

**Test to add:**
```python
def test_payload_tampering_detected():
    """Corrupted payloads must raise IntegrityError."""
    store = PayloadStore(db)
    original = {"key": "value", "nested": {"deep": True}}
    ref = store.store(original)

    # Verify normal retrieval works
    assert store.retrieve(ref) == original

    # Simulate tampering
    db.execute(
        "UPDATE payloads SET content = ? WHERE ref = ?",
        (b'{"tampered": true}', ref),
    )

    # Tampering must be detected
    with pytest.raises(PayloadIntegrityError, match="hash mismatch"):
        store.retrieve(ref)
```

**Acceptance criteria:**
- [ ] Test verifies tampering is detected
- [ ] Hash mismatch message is informative
- [ ] Test covers both content and hash tampering

---

## Progress Tracking

### Week 1 Checklist

| Task | Status | PR | Notes |
|------|--------|-----|-------|
| Source payload persistence | 🔄 | - | Being fixed by separate agent |
| Artifact secret leakage | 🔄 | - | Being fixed by separate agent |
| Decimal NaN/Infinity | 🔄 | - | Being fixed by separate agent |
| Mutation testing config | ✅ | - | Expanded to 10,145 LOC |

### Week 2 Checklist

| Task | Status | PR | Notes |
|------|--------|-----|-------|
| Sleep → sync primitives | ⬜ | - | |
| Terminal state property test | ⬜ | - | |
| Strengthen weak assertions (50) | ⬜ | - | |
| Payload tampering test | ⬜ | - | |

### Week 3 Checklist

| Task | Status | PR | Notes |
|------|--------|-----|-------|
| Checkpoint malformed JSON | ⬜ | - | |
| Multi-sink idempotency | ⬜ | - | |
| Simplify LLM mocks | ⬜ | - | |
| Empty batch handling | ⬜ | - | |

---

## Verification Commands

```bash
# Run all tests
.venv/bin/python -m pytest tests/ -v

# Run property tests with nightly profile (more examples)
HYPOTHESIS_PROFILE=nightly .venv/bin/python -m pytest tests/property/ -v

# Check weak assertion count
grep -r "is not None" tests/ | grep "assert" | wc -l

# Check type ignore count
grep -r "type: ignore" tests/ | wc -l

# Run mutation testing on engine
mutmut run --paths-to-mutate src/elspeth/engine/orchestrator.py

# Check mutation results
mutmut results
mutmut show surviving
```

---

## Definition of Done

The test suite uplift is complete when:

1. ✅ All P0 bugs have fixes with regression tests
2. ✅ Mutation testing covers engine with >85% kill rate
3. ✅ Terminal state property test passes with 500+ examples
4. ✅ Weak assertions reduced to <50
5. ✅ No `time.sleep()` in concurrency tests
6. ✅ Zero flaky test failures for 2 consecutive weeks
7. ✅ All new tests follow established patterns (fixtures, isolation, assertions)

---

*Action plan derived from TEST_SUITE_ANALYSIS_2026-01-22.md*
