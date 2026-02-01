# How to Fix Surviving Mutants - Practical Guide

This guide provides step-by-step instructions for addressing the 642 actionable surviving mutants identified in the mutation testing run.

## Quick Start

1. **Open the checklist:** `docs/analysis/MUTATION_GAPS_CHECKLIST.md`
2. **Start with P0 files:** orchestrator.py, executors.py, processor.py
3. **Use the patterns below** to write effective tests
4. **Run differential mutation testing** after each batch

## Test Pattern Templates

### Pattern 1: Default Factory Fields

**Survivor Example:**
```python
child_tokens: list[TokenInfo] = field(default_factory=list)
```

**Fix - Add Dataclass Default Test:**
```python
def test_gate_result_has_empty_child_tokens_by_default():
    """Verify child_tokens defaults to empty list, not None."""
    result = GateResult(
        result=GateResult(...),
        updated_token=TokenInfo(...)
    )
    assert isinstance(result.child_tokens, list)
    assert len(result.child_tokens) == 0
    assert result.child_tokens == []  # Exact value check
```

**Where to add:** `tests/engine/test_executors.py` or create `test_executors_mutation_gaps.py`

---

### Pattern 2: Error Message Content

**Survivor Example:**
```python
raise ValueError(f"Expected {expected} but got {actual}")
```

**Fix - Add Message Assertion:**
```python
def test_validator_error_message_contains_expected_and_actual():
    """Error messages must include both expected and actual values for debugging."""
    with pytest.raises(ValueError, match=r"Expected .* but got .*"):
        validate_something(expected=5, actual=10)

    # Or more specific:
    with pytest.raises(ValueError, match=r"Expected 5 but got 10"):
        validate_something(expected=5, actual=10)
```

**Where to add:** Same file as the function being tested

---

### Pattern 3: Context Attribute Setting

**Survivor Example:**
```python
ctx.state_id = state.state_id
ctx.node_id = transform.node_id
```

**Fix - Verify Context State:**
```python
def test_executor_sets_context_attributes_before_transform():
    """Context must have state_id and node_id set for external call recording."""
    executor = TransformExecutor(...)
    ctx = PluginContext(...)
    transform = MockTransform(node_id="transform-123")

    executor.execute(transform, token, ctx, step=1)

    # Verify context was populated
    assert ctx.state_id == "expected-state-id"
    assert ctx.node_id == "transform-123"
    assert ctx._call_index == 0
```

**Where to add:** `tests/engine/test_executors.py`

---

### Pattern 4: Routing/Validation Logic

**Survivor Example:**
```python
if destination == "continue":
    continue
```

**Fix - Test Control Flow:**
```python
def test_gate_routing_skips_continue_destination():
    """Routes with 'continue' destination should not validate against sinks."""
    # Setup with a gate that routes to "continue"
    gate_settings = GateSettings(
        name="test_gate",
        routes={"true": "continue", "false": "sink_a"}
    )

    # Should not raise RouteValidationError for "continue"
    validate_routes(gate_settings, available_sinks={"sink_a"})

def test_gate_routing_validates_sink_destinations():
    """Routes to actual sinks must validate sink exists."""
    gate_settings = GateSettings(
        name="test_gate",
        routes={"true": "nonexistent_sink"}
    )

    with pytest.raises(RouteValidationError, match="nonexistent_sink"):
        validate_routes(gate_settings, available_sinks={"sink_a"})
```

**Where to add:** `tests/engine/test_orchestrator_validation.py`

---

### Pattern 5: State Transitions

**Survivor Example:**
```python
result.status = RunStatus.COMPLETED
```

**Fix - Verify State Changes:**
```python
def test_orchestrator_marks_run_completed_on_success():
    """Run status must transition to COMPLETED after successful execution."""
    orchestrator = Orchestrator(config, db)

    result = orchestrator.run(source_rows=[{"id": 1}])

    # Verify status was set
    assert result.status == RunStatus.COMPLETED

    # Verify persisted to database
    run_record = db.get_run(result.run_id)
    assert run_record.status == RunStatus.COMPLETED
```

**Where to add:** `tests/engine/test_orchestrator.py`

---

### Pattern 6: Numeric Constants and Defaults

**Survivor Example:**
```python
rows_quarantined: int = 0
```

**Fix - Verify Initial State:**
```python
def test_run_result_initializes_counters_to_zero():
    """All row counters must default to zero, not None or uninitialized."""
    result = RunResult(
        run_id="test",
        rows_succeeded=10,
        rows_failed=2,
        rows_routed=3
    )

    # Verify optional counters default to 0
    assert result.rows_quarantined == 0
    assert result.rows_forked == 0
    assert result.rows_coalesced == 0
    assert result.rows_expanded == 0
```

**Where to add:** `tests/engine/test_orchestrator.py`

---

## Workflow for Fixing Survivors

### Step 1: Pick a File from Checklist

Start with P0 files (highest impact):
1. `src/elspeth/engine/orchestrator.py` (206 survivors)
2. `src/elspeth/engine/executors.py` (88 survivors)
3. `src/elspeth/engine/processor.py` (62 survivors)

### Step 2: Choose a Section

Open the checklist and find a logical grouping (e.g., all survivors in one function).

Example from `MUTATION_GAPS_CHECKLIST.md`:
```markdown
### Line 169

```python
    168 |         # and batch checkpoint lookup (node_id required)
>>> 169 |         ctx.state_id = state.state_id
    170 |         ctx.node_id = transform.node_id
```

- [ ] Add test to catch mutations on this line
```

### Step 3: Write the Test

Use Pattern 3 from above:

```python
# tests/engine/test_executors_mutation_gaps.py

def test_transform_executor_sets_context_state_id():
    """Context.state_id must be set from NodeState for external call recording."""
    executor = TransformExecutor(...)
    state = NodeState(state_id="state-abc123", ...)
    ctx = PluginContext()

    executor._execute_transform(..., state=state, ctx=ctx)

    assert ctx.state_id == "state-abc123"
```

### Step 4: Verify the Fix

Run differential mutation testing on just that file:

```bash
.venv/bin/python -m mutmut run --paths-to-mutate src/elspeth/engine/executors.py
```

Check that the mutant on line 169 is now killed:

```bash
sqlite3 .mutmut-cache "SELECT status FROM mutant m JOIN line l ON m.line = l.id WHERE l.line_number = 169;"
```

Should show `ok_killed` instead of `bad_survived`.

### Step 5: Check Off in Checklist

Mark the checkbox:
```markdown
- [x] Add test to catch mutations on this line
```

### Step 6: Commit

```bash
git add tests/engine/test_executors_mutation_gaps.py
git commit -m "test(mutation): kill survivor on executors.py:169 (context.state_id)"
```

### Step 7: Repeat

Work through 5-10 survivors at a time, commit frequently.

---

## Batch Fixing Strategy

### Batch 1: Error Messages (Quick Wins)

Search the checklist for lines with `raise ValueError` or `raise RuntimeError`:
- Add `match=` parameter to existing `pytest.raises()` assertions
- Estimated time: 2-3 min per survivor
- Estimated batch: 50-80 survivors in 2-3 hours

### Batch 2: Default Factories

Search for `field(default_factory=`:
- Add dataclass initialization tests
- Estimated time: 5-8 min per survivor
- Estimated batch: 30-40 survivors in 3-4 hours

### Batch 3: Context Setting

Search for `ctx.` assignments:
- Add context attribute assertions to existing tests
- Estimated time: 5-10 min per survivor
- Estimated batch: 20-30 survivors in 2-3 hours

### Batch 4: Validation Logic

Search for `if` statements and control flow:
- Add branch coverage tests
- Estimated time: 10-15 min per survivor
- Estimated batch: 15-20 survivors in 3-4 hours

---

## Common Pitfalls

### Pitfall 1: Testing Implementation Instead of Behavior

**Bad:**
```python
def test_orchestrator_calls_validate_routes():
    """This tests implementation, not behavior."""
    orchestrator = Orchestrator(config, db)
    with mock.patch('elspeth.engine.orchestrator.validate_routes') as mock_validate:
        orchestrator.run(...)
        mock_validate.assert_called_once()  # ❌ Implementation detail
```

**Good:**
```python
def test_orchestrator_rejects_routes_to_nonexistent_sinks():
    """This tests observable behavior."""
    config = Config(
        gates=[GateSettings(name="gate", routes={"true": "nonexistent"})]
    )
    orchestrator = Orchestrator(config, db)

    with pytest.raises(RouteValidationError, match="nonexistent"):
        orchestrator.run(...)  # ✅ Behavior verification
```

### Pitfall 2: Weak Assertions

**Bad:**
```python
def test_context_gets_set():
    result = executor.execute(...)
    assert result is not None  # ❌ Weak - doesn't verify ctx changed
```

**Good:**
```python
def test_executor_sets_context_node_id():
    ctx = PluginContext()
    result = executor.execute(..., ctx=ctx)
    assert ctx.node_id == "expected-node-id"  # ✅ Specific assertion
```

### Pitfall 3: Not Running Mutation Tests After Fix

Always verify your fix actually kills the mutant:

```bash
# Bad: assume the test works
git commit -m "fix mutation survivor"

# Good: verify first
.venv/bin/python -m mutmut run --paths-to-mutate src/elspeth/engine/orchestrator.py
# Check that survivor count decreased
git commit -m "test(mutation): kill orchestrator.py:245 survivor"
```

---

## Progress Tracking

Create a tracking file `docs/analysis/MUTATION_PROGRESS.md`:

```markdown
# Mutation Testing Progress

## P0 Files

### orchestrator.py (206 survivors)
- [x] Lines 63-77 (error messages) - PR #123
- [x] Lines 169-170 (context setting) - PR #124
- [ ] Lines 230-245 (routing validation)
- [ ] ... (remaining)

Current: 180 survivors (-26)

### executors.py (88 survivors)
- [ ] ...

### processor.py (62 survivors)
- [ ] ...
```

Update after each batch of fixes.

---

## Estimated Timeline

**Aggressive (Full-time focus):**
- Week 1-2: P0 orchestrator.py (206 survivors) → ~160 survivors killed
- Week 3: P0 executors.py + processor.py (150 survivors) → ~120 killed
- Week 4: P1 files (93 survivors) → ~70 killed
- Total: 4 weeks, ~350 survivors killed (kill rate: 65% → 75%)

**Sustainable (Part-time):**
- 10-20 survivors per week
- 30-40 weeks for all P0/P1 survivors
- Suggested: Focus on P0 only, defer P2

---

## Success Metrics

After fixing P0 files (356 survivors):
```bash
.venv/bin/python -m mutmut run --paths-to-mutate \
    src/elspeth/engine/orchestrator.py \
    src/elspeth/engine/executors.py \
    src/elspeth/engine/processor.py

# Target kill rate for P0 files: 90%+
```

After fixing P1 files (93 survivors):
```bash
# Target overall kill rate: 75-80%
```

After fixing P2 files (127 survivors):
```bash
# Target overall kill rate: 85%+
```

---

*Use this guide in conjunction with MUTATION_GAPS_CHECKLIST.md*
*Kill rate target: 85% | Current: 55.5% | Gap: 642 actionable survivors*
