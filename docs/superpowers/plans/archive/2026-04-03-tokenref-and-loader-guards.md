# TokenRef Adoption & Loader Guard Hardening

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add explicit AuditIntegrityError guards to RunLoader, CallLoader, and OperationLoader; adopt `list[TokenRef]` in `coalesce_tokens`; defer the deep TokenRef field replacement (Option A) with a documented rationale.

**Architecture:** Three focused changes to the Landscape audit subsystem. Tasks 1-3 add pre-construction guards to loaders (following the TokenOutcomeLoader/NodeStateLoader pattern). Task 4 changes `coalesce_tokens` to accept `list[TokenRef]` instead of `list[str]`, threading run context through the multi-token join path. Task 5 updates the CI guard symmetry allowlist. Task 6 defers the Option A deep refactor with acceptance criteria.

**Tech Stack:** Python dataclasses, SQLAlchemy Core, pytest, Filigree issue tracker

**Tracked Issues:**
- `elspeth-971113c6b0` — Loader guards (Tasks 1-3, 5)
- `elspeth-bb432582c3` — coalesce_tokens TokenRef (Task 4)
- `elspeth-dbfa04ee89` — Option A deferral (Task 6)

---

### Task 1: RunLoader — AuditIntegrityError Guards

**Files:**
- Modify: `src/elspeth/core/landscape/model_loaders.py:53-77`
- Test: `tests/unit/core/landscape/test_model_loaders.py` (TestRunLoader class, line 96)

**Context:** RunLoader currently delegates all validation to `Run.__post_init__()`, which validates enum types. The loader should add explicit guards for status-dependent invariants that `__post_init__` does NOT check: a completed run must have `completed_at`, and export-related fields have consistency rules.

- [ ] **Step 1: Write failing tests for RunLoader guards**

Add to `TestRunLoader` in `tests/unit/core/landscape/test_model_loaders.py`:

```python
def test_completed_without_completed_at_raises_audit_integrity(self) -> None:
    """Completed run with NULL completed_at is a Tier 1 integrity violation."""
    sa_row = self._make_run_row(status="completed", completed_at=None)
    loader = RunLoader()
    with pytest.raises(AuditIntegrityError, match="completed_at"):
        loader.load(sa_row)

def test_failed_without_completed_at_raises_audit_integrity(self) -> None:
    """Failed run with NULL completed_at is a Tier 1 integrity violation."""
    sa_row = self._make_run_row(status="failed", completed_at=None)
    loader = RunLoader()
    with pytest.raises(AuditIntegrityError, match="completed_at"):
        loader.load(sa_row)

def test_interrupted_without_completed_at_raises_audit_integrity(self) -> None:
    """Interrupted run with NULL completed_at is a Tier 1 integrity violation."""
    sa_row = self._make_run_row(status="interrupted", completed_at=None)
    loader = RunLoader()
    with pytest.raises(AuditIntegrityError, match="completed_at"):
        loader.load(sa_row)

def test_running_with_completed_at_raises_audit_integrity(self) -> None:
    """Running run with completed_at set is a Tier 1 integrity violation."""
    sa_row = self._make_run_row(status="running", completed_at=LATER)
    loader = RunLoader()
    with pytest.raises(AuditIntegrityError, match="completed_at"):
        loader.load(sa_row)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/unit/core/landscape/test_model_loaders.py::TestRunLoader -v -x`
Expected: FAIL — loader currently passes values through without guard checks.

- [ ] **Step 3: Implement RunLoader guards**

Replace the `RunLoader.load()` method in `src/elspeth/core/landscape/model_loaders.py:56-77`:

```python
def load(self, row: SARow[Any]) -> Run:
    """Load Run from database row.

    Converts string fields to enums. Validates status-dependent
    invariants before construction.

    Raises:
        AuditIntegrityError: If status/completed_at are inconsistent (Tier 1)
    """
    status = RunStatus(row.status)

    # Tier 1: status-dependent lifecycle invariants
    if status == RunStatus.RUNNING:
        if row.completed_at is not None:
            raise AuditIntegrityError(
                f"Run {row.run_id} has status='running' but completed_at is set — "
                f"audit integrity violation (running runs must not have completed_at)"
            )
    else:
        # COMPLETED, FAILED, INTERRUPTED all require completed_at
        if row.completed_at is None:
            raise AuditIntegrityError(
                f"Run {row.run_id} has status={status.value!r} but completed_at is NULL — "
                f"audit integrity violation (terminal runs must have completed_at)"
            )

    return Run(
        run_id=row.run_id,
        started_at=row.started_at,
        config_hash=row.config_hash,
        settings_json=row.settings_json,
        canonical_version=row.canonical_version,
        status=status,
        completed_at=row.completed_at,
        reproducibility_grade=ReproducibilityGrade(row.reproducibility_grade) if row.reproducibility_grade is not None else None,
        export_status=ExportStatus(row.export_status) if row.export_status is not None else None,
        export_error=row.export_error,
        exported_at=row.exported_at,
        export_format=row.export_format,
        export_sink=row.export_sink,
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/unit/core/landscape/test_model_loaders.py::TestRunLoader -v`
Expected: ALL PASS

- [ ] **Step 5: Update existing test expectations**

The existing `test_valid_load_all_fields` uses `status="completed"` with `completed_at=LATER`, which should still pass. Verify no existing tests break by running the full TestRunLoader suite.

Run: `.venv/bin/python -m pytest tests/unit/core/landscape/test_model_loaders.py::TestRunLoader -v`
Expected: ALL PASS (existing + new tests)

---

### Task 2: CallLoader — AuditIntegrityError Guards

**Files:**
- Modify: `src/elspeth/core/landscape/model_loaders.py:190-213`
- Test: `tests/unit/core/landscape/test_model_loaders.py` (TestCallLoader class, line 495)

**Context:** CallLoader delegates validation to `Call.__post_init__()`, which checks call_index (via `require_int`), enum types, and the XOR constraint (state_id vs operation_id). The loader should add explicit AIE guards for the XOR constraint (the most structurally significant invariant) before construction, so violations are reported as `AuditIntegrityError` rather than a generic `ValueError`.

- [ ] **Step 1: Write failing tests for CallLoader guards**

Add to `TestCallLoader` in `tests/unit/core/landscape/test_model_loaders.py`:

```python
def test_both_state_and_operation_raises_audit_integrity(self) -> None:
    """Both state_id and operation_id set is a Tier 1 integrity violation."""
    sa_row = self._make_call_row(state_id="state-1", operation_id="op-1")
    loader = CallLoader()
    with pytest.raises(AuditIntegrityError, match="exactly one"):
        loader.load(sa_row)

def test_neither_state_nor_operation_raises_audit_integrity(self) -> None:
    """Neither state_id nor operation_id set is a Tier 1 integrity violation."""
    sa_row = self._make_call_row(state_id=None, operation_id=None)
    loader = CallLoader()
    with pytest.raises(AuditIntegrityError, match="exactly one"):
        loader.load(sa_row)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/unit/core/landscape/test_model_loaders.py::TestCallLoader -v -x`
Expected: FAIL — currently raises `ValueError` from `__post_init__`, not `AuditIntegrityError` from loader.

- [ ] **Step 3: Implement CallLoader guards**

Replace the `CallLoader.load()` method in `src/elspeth/core/landscape/model_loaders.py:193-213`:

```python
def load(self, row: SARow[Any]) -> Call:
    """Load Call from database row.

    Handles both state-parented calls (transform processing) and
    operation-parented calls (source/sink I/O). Validates XOR
    constraint before construction.

    Raises:
        AuditIntegrityError: If state_id/operation_id XOR violated (Tier 1)
    """
    # Tier 1: XOR constraint — exactly one parent context
    has_state = row.state_id is not None
    has_operation = row.operation_id is not None
    if has_state == has_operation:
        raise AuditIntegrityError(
            f"Call {row.call_id} requires exactly one of state_id or operation_id. "
            f"Got state_id={row.state_id!r}, operation_id={row.operation_id!r} — "
            f"audit integrity violation"
        )

    return Call(
        call_id=row.call_id,
        call_index=row.call_index,
        call_type=CallType(row.call_type),
        status=CallStatus(row.status),
        request_hash=row.request_hash,
        created_at=row.created_at,
        state_id=row.state_id,
        operation_id=row.operation_id,
        request_ref=row.request_ref,
        response_hash=row.response_hash,
        response_ref=row.response_ref,
        error_json=row.error_json,
        latency_ms=row.latency_ms,
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/unit/core/landscape/test_model_loaders.py::TestCallLoader -v`
Expected: ALL PASS

---

### Task 3: OperationLoader — AuditIntegrityError Guards

**Files:**
- Modify: `src/elspeth/core/landscape/model_loaders.py:609-640`
- Test: `tests/unit/core/landscape/test_model_loaders.py` (TestOperationLoader class, line 1589)

**Context:** OperationLoader delegates all validation to `Operation.__post_init__()`, which checks operation_type, status, and status-dependent lifecycle invariants. The loader should add explicit AIE guards for the status-lifecycle invariants before construction, mirroring how NodeStateLoader handles its state-dependent field validation. The `__post_init__` currently raises generic `ValueError` — the loader should raise `AuditIntegrityError` to correctly attribute the failure as a Tier 1 violation.

- [ ] **Step 1: Write failing tests for OperationLoader guards**

Add to `TestOperationLoader` in `tests/unit/core/landscape/test_model_loaders.py`:

```python
# === AuditIntegrityError guards (pre-construction, Tier 1) ===

def test_invalid_operation_type_raises_audit_integrity(self) -> None:
    """Invalid operation_type is a Tier 1 integrity violation."""
    sa_row = self._make_operation_row(operation_type="kafka_consume")
    loader = OperationLoader()
    with pytest.raises(AuditIntegrityError, match="operation_type"):
        loader.load(sa_row)

def test_invalid_status_raises_audit_integrity(self) -> None:
    """Invalid status is a Tier 1 integrity violation."""
    sa_row = self._make_operation_row(status="running")
    loader = OperationLoader()
    with pytest.raises(AuditIntegrityError, match="status"):
        loader.load(sa_row)

def test_open_with_completed_at_raises_audit_integrity(self) -> None:
    sa_row = self._make_operation_row(status="open", completed_at=NOW)
    loader = OperationLoader()
    with pytest.raises(AuditIntegrityError, match="completed_at"):
        loader.load(sa_row)

def test_open_with_duration_ms_raises_audit_integrity(self) -> None:
    sa_row = self._make_operation_row(status="open", duration_ms=100.0)
    loader = OperationLoader()
    with pytest.raises(AuditIntegrityError, match="duration_ms"):
        loader.load(sa_row)

def test_open_with_error_message_raises_audit_integrity(self) -> None:
    sa_row = self._make_operation_row(status="open", error_message="bad")
    loader = OperationLoader()
    with pytest.raises(AuditIntegrityError, match="error_message"):
        loader.load(sa_row)

def test_completed_without_completed_at_raises_audit_integrity(self) -> None:
    sa_row = self._make_operation_row(
        status="completed", completed_at=None, duration_ms=100.0,
    )
    loader = OperationLoader()
    with pytest.raises(AuditIntegrityError, match="completed_at"):
        loader.load(sa_row)

def test_completed_without_duration_ms_raises_audit_integrity(self) -> None:
    sa_row = self._make_operation_row(
        status="completed", completed_at=LATER, duration_ms=None,
    )
    loader = OperationLoader()
    with pytest.raises(AuditIntegrityError, match="duration_ms"):
        loader.load(sa_row)

def test_completed_with_error_message_raises_audit_integrity(self) -> None:
    sa_row = self._make_operation_row(
        status="completed", completed_at=LATER, duration_ms=100.0,
        error_message="should not be here",
    )
    loader = OperationLoader()
    with pytest.raises(AuditIntegrityError, match="error_message"):
        loader.load(sa_row)

def test_failed_without_error_message_raises_audit_integrity(self) -> None:
    sa_row = self._make_operation_row(
        status="failed", completed_at=LATER, duration_ms=100.0,
        error_message=None,
    )
    loader = OperationLoader()
    with pytest.raises(AuditIntegrityError, match="error_message"):
        loader.load(sa_row)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/unit/core/landscape/test_model_loaders.py::TestOperationLoader -v -x`
Expected: FAIL — currently raises `ValueError` from `__post_init__`, not `AuditIntegrityError` from loader.

- [ ] **Step 3: Implement OperationLoader guards**

Replace the `OperationLoader.load()` method in `src/elspeth/core/landscape/model_loaders.py:617-640`:

```python
def load(self, row: SARow[Any]) -> Operation:
    """Load Operation from database row.

    Validates operation_type, status, and status-dependent lifecycle
    invariants before construction.

    Raises:
        AuditIntegrityError: If operation_type/status invalid or lifecycle
            invariants violated (Tier 1)
    """
    oid = row.operation_id

    # Tier 1: validate constrained literal fields
    allowed_types = ("source_load", "sink_write")
    if row.operation_type not in allowed_types:
        raise AuditIntegrityError(
            f"Operation {oid} has invalid operation_type={row.operation_type!r} "
            f"(expected one of {allowed_types}) — audit integrity violation"
        )

    allowed_statuses = ("open", "completed", "failed", "pending")
    if row.status not in allowed_statuses:
        raise AuditIntegrityError(
            f"Operation {oid} has invalid status={row.status!r} "
            f"(expected one of {allowed_statuses}) — audit integrity violation"
        )

    # Tier 1: status-dependent lifecycle invariants
    if row.status == "open":
        if row.completed_at is not None:
            raise AuditIntegrityError(
                f"Operation {oid}: status='open' but completed_at is set — audit integrity violation"
            )
        if row.duration_ms is not None:
            raise AuditIntegrityError(
                f"Operation {oid}: status='open' but duration_ms is set — audit integrity violation"
            )
        if row.error_message is not None:
            raise AuditIntegrityError(
                f"Operation {oid}: status='open' but error_message is set — audit integrity violation"
            )
    elif row.status in ("completed", "failed", "pending"):
        if row.completed_at is None:
            raise AuditIntegrityError(
                f"Operation {oid}: status={row.status!r} but completed_at is NULL — audit integrity violation"
            )
        if row.duration_ms is None:
            raise AuditIntegrityError(
                f"Operation {oid}: status={row.status!r} but duration_ms is NULL — audit integrity violation"
            )
        if row.status == "failed" and row.error_message is None:
            raise AuditIntegrityError(
                f"Operation {oid}: status='failed' but error_message is NULL — audit integrity violation"
            )
        if row.status == "completed" and row.error_message is not None:
            raise AuditIntegrityError(
                f"Operation {oid}: status='completed' but error_message is set — audit integrity violation"
            )

    return Operation(
        operation_id=oid,
        run_id=row.run_id,
        node_id=row.node_id,
        operation_type=row.operation_type,
        started_at=row.started_at,
        completed_at=row.completed_at,
        status=row.status,
        input_data_ref=row.input_data_ref,
        input_data_hash=row.input_data_hash,
        output_data_ref=row.output_data_ref,
        output_data_hash=row.output_data_hash,
        error_message=row.error_message,
        duration_ms=row.duration_ms,
    )
```

- [ ] **Step 4: Run tests to verify new tests pass**

Run: `.venv/bin/python -m pytest tests/unit/core/landscape/test_model_loaders.py::TestOperationLoader -v`
Expected: New AIE tests PASS. But existing `ValueError` tests now fail (see Step 5).

- [ ] **Step 5: Update existing tests to expect AuditIntegrityError**

The existing tests at lines 1682-1752 expect `ValueError` but the loader now raises `AuditIntegrityError` before construction. Update them:

Replace `pytest.raises(ValueError, match="operation_type")` with `pytest.raises(AuditIntegrityError, match="operation_type")` in `test_invalid_operation_type_raises`.

Replace `pytest.raises(ValueError, match="status")` with `pytest.raises(AuditIntegrityError, match="status")` in `test_invalid_status_raises`.

Replace all remaining `pytest.raises(ValueError, match=...)` for lifecycle invariant tests (`test_open_with_completed_at_raises`, etc.) with `pytest.raises(AuditIntegrityError, match=...)`.

After updating, the new AIE tests from Step 1 are duplicates of these — **remove the new tests from Step 1** and keep the updated existing tests. This avoids test duplication.

- [ ] **Step 6: Run full OperationLoader test suite**

Run: `.venv/bin/python -m pytest tests/unit/core/landscape/test_model_loaders.py::TestOperationLoader -v`
Expected: ALL PASS

---

### Task 4: coalesce_tokens — Accept `list[TokenRef]`

**Files:**
- Modify: `src/elspeth/core/landscape/data_flow_repository.py:554-645` (DataFlowRepository.coalesce_tokens)
- Modify: `src/elspeth/core/landscape/recorder.py:807-819` (LandscapeRecorder.coalesce_tokens)
- Modify: `src/elspeth/engine/tokens.py:265-305` (TokenManager.coalesce_tokens)
- Modify: `src/elspeth/engine/coalesce_executor.py:735` (caller passes run_id)
- Test: `tests/unit/core/landscape/test_data_flow_repository.py`
- Test: `tests/unit/core/landscape/test_token_recording.py`
- Test: `tests/unit/engine/test_tokens.py`
- Test: `tests/unit/engine/test_token_manager_pipeline_row.py`
- Test: `tests/integration/audit/test_recorder_tokens.py`
- Test: `tests/integration/audit/test_recorder_queries.py`
- Test: `tests/property/engine/test_token_lifecycle_state_machine.py`
- Test: `tests/property/engine/test_coalesce_properties.py`

**Context:** `coalesce_tokens` currently takes `parent_token_ids: list[str]` and validates each token individually. The change bundles each parent as `TokenRef(token_id, run_id)`, preventing cross-run contamination at the type level — matching the pattern already used by `fork_token`, `expand_token`, and `record_token_outcome`.

The change flows through 3 layers:
1. `DataFlowRepository.coalesce_tokens()` — the actual implementation
2. `LandscapeRecorder.coalesce_tokens()` — thin delegation facade
3. `TokenManager.coalesce_tokens()` — engine-level wrapper that constructs the refs

- [ ] **Step 1: Write failing test for DataFlowRepository.coalesce_tokens with TokenRef**

Add a test to `tests/unit/core/landscape/test_data_flow_repository.py` (or update existing):

```python
def test_coalesce_tokens_accepts_token_refs(self, repo, run_id):
    """coalesce_tokens should accept list[TokenRef] with bundled run context."""
    # This test verifies the new signature. Existing tests will be updated
    # to pass TokenRef instead of bare strings.
    from elspeth.contracts.audit import TokenRef
    # ... (test body depends on existing fixture patterns in this file)
```

Note: The exact test depends on existing test fixtures. The key assertion is that the method accepts `list[TokenRef]` and uses `ref.token_id` / `ref.run_id` internally.

- [ ] **Step 2: Update DataFlowRepository.coalesce_tokens signature and body**

In `src/elspeth/core/landscape/data_flow_repository.py:554-645`, change:

```python
def coalesce_tokens(
    self,
    parent_refs: list[TokenRef],
    row_id: str,
    *,
    step_in_pipeline: int | None = None,
) -> Token:
    """Coalesce multiple tokens into one (join operation).

    Creates a new token representing the merged result.
    Records all parent relationships.

    Validates that all parent tokens belong to the specified row_id and
    that they all share the same run_id. Cross-run/cross-row contamination
    crashes immediately per Tier 1 trust model.

    Args:
        parent_refs: TokenRefs for tokens being merged (bundled token_id + run_id)
        row_id: Row ID for the merged token
        step_in_pipeline: Step in the DAG where the coalesce occurs

    Returns:
        Merged Token model

    Raises:
        AuditIntegrityError: If parent tokens do not belong to specified row
            or if parent tokens span multiple runs
    """
    if not parent_refs:
        raise AuditIntegrityError(
            "coalesce_tokens requires at least one parent token — a coalesce with zero parents creates an unexplainable audit state"
        )

    # Validate all parent tokens belong to the same row and run (Tier 1 invariant)
    run_id_resolved: str | None = None
    for ref in parent_refs:
        self._validate_token_row_ownership(ref.token_id, row_id)
        self._validate_token_run_ownership(ref)
        if run_id_resolved is None:
            run_id_resolved = ref.run_id
        elif ref.run_id != run_id_resolved:
            raise AuditIntegrityError(
                f"Cross-run contamination prevented in coalesce: parent token {ref.token_id!r} "
                f"belongs to run {ref.run_id!r}, but other parents belong to run {run_id_resolved!r}. "
                f"All parent tokens in a coalesce must belong to the same run."
            )

    if run_id_resolved is None:
        run_id_resolved = self._resolve_run_id_for_row(row_id)

    join_group_id = generate_id()
    token_id = generate_id()
    timestamp = now()

    with self._db.connection() as conn:
        result = conn.execute(
            tokens_table.insert().values(
                token_id=token_id,
                row_id=row_id,
                run_id=run_id_resolved,
                join_group_id=join_group_id,
                step_in_pipeline=step_in_pipeline,
                created_at=timestamp,
            )
        )
        if result.rowcount == 0:
            raise AuditIntegrityError(f"coalesce_tokens: merged token INSERT affected zero rows (token_id={token_id})")

        for ordinal, ref in enumerate(parent_refs):
            result = conn.execute(
                token_parents_table.insert().values(
                    token_id=token_id,
                    parent_token_id=ref.token_id,
                    ordinal=ordinal,
                )
            )
            if result.rowcount == 0:
                raise AuditIntegrityError(
                    f"coalesce_tokens: token_parent INSERT affected zero rows (child={token_id}, parent={ref.token_id})"
                )

    return Token(
        token_id=token_id,
        row_id=row_id,
        join_group_id=join_group_id,
        step_in_pipeline=step_in_pipeline,
        created_at=timestamp,
        run_id=run_id_resolved,
    )
```

- [ ] **Step 3: Update LandscapeRecorder.coalesce_tokens signature**

In `src/elspeth/core/landscape/recorder.py:807-819`, change:

```python
def coalesce_tokens(
    self,
    parent_refs: list[TokenRef],
    row_id: str,
    *,
    step_in_pipeline: int | None = None,
) -> Token:
    """Coalesce multiple tokens. Delegates to DataFlowRepository."""
    return self._data_flow.coalesce_tokens(
        parent_refs,
        row_id,
        step_in_pipeline=step_in_pipeline,
    )
```

- [ ] **Step 4: Update TokenManager.coalesce_tokens to construct TokenRefs**

In `src/elspeth/engine/tokens.py:265-305`, change the delegation to construct `TokenRef` objects. The `TokenManager` has access to `run_id` from its constructor (`self._run_id` or from the parent tokens). Inspect the class to determine how run_id is accessed.

`TokenManager` does NOT store `run_id` — it receives it per-call (like `fork_token` and `expand_token`). Add `run_id: str` as a required parameter:

```python
def coalesce_tokens(
    self,
    parents: list[TokenInfo],
    merged_data: PipelineRow,
    node_id: NodeID,
    run_id: str,
) -> TokenInfo:
    """Coalesce multiple tokens into one.

    Args:
        parents: Parent tokens to merge
        merged_data: Merged row data as PipelineRow (with merged contract)
        node_id: NodeID of the coalesce node performing the merge
        run_id: Run ID for TokenRef construction (prevents cross-run contamination)

    Returns:
        Merged TokenInfo with PipelineRow row_data
    """
    if not parents:
        raise OrchestrationInvariantError("coalesce_tokens requires at least one parent token")

    row_id = parents[0].row_id
    mismatched = [p.token_id for p in parents if p.row_id != row_id]
    if mismatched:
        raise OrchestrationInvariantError(
            f"coalesce_tokens requires all parents to share row_id={row_id}; mismatched token_ids={mismatched}"
        )

    step = self._step_resolver(node_id)

    merged = self._recorder.coalesce_tokens(
        parent_refs=[TokenRef(token_id=p.token_id, run_id=run_id) for p in parents],
        row_id=row_id,
        step_in_pipeline=step,
    )

    return TokenInfo(
        row_id=row_id,
        token_id=merged.token_id,
        row_data=merged_data,
        join_group_id=merged.join_group_id,
    )
```

Then update the caller in `src/elspeth/engine/coalesce_executor.py:735`:

```python
# Before:
merged_token = self._token_manager.coalesce_tokens(
    parents=list(consumed_tokens),
    merged_data=merged_data,
    node_id=node_id,
)

# After:
merged_token = self._token_manager.coalesce_tokens(
    parents=list(consumed_tokens),
    merged_data=merged_data,
    node_id=node_id,
    run_id=self._run_id,
)
```

- [ ] **Step 5: Update all test call sites**

There are ~26 test call sites across 9 test files. Each must change from:

```python
recorder.coalesce_tokens(parent_token_ids=[...], row_id=...)
```

to:

```python
recorder.coalesce_tokens(parent_refs=[TokenRef(token_id=..., run_id=...) for ...], row_id=...)
```

**Test files to update (with approximate call counts):**
- `tests/unit/core/landscape/test_token_recording.py` — 8 calls
- `tests/unit/core/landscape/test_data_flow_repository.py` — 2 calls
- `tests/unit/core/landscape/test_query_methods.py` — 2 calls
- `tests/unit/engine/test_tokens.py` — 6 calls (these call `TokenManager.coalesce_tokens`, different signature — unchanged)
- `tests/unit/engine/test_token_manager_pipeline_row.py` — 1 call (TokenManager, unchanged)
- `tests/integration/audit/test_recorder_tokens.py` — 6 calls
- `tests/integration/audit/test_recorder_queries.py` — 1 call
- `tests/property/engine/test_token_lifecycle_state_machine.py` — 2 calls
- `tests/property/engine/test_coalesce_properties.py` — 1 mock setup

For tests that call `TokenManager.coalesce_tokens(parents=..., merged_data=..., node_id=...)`, the signature is unchanged — `TokenManager` constructs the refs internally.

For tests that call `recorder.coalesce_tokens(parent_token_ids=..., row_id=...)`, update the keyword argument name and wrap IDs in `TokenRef`.

- [ ] **Step 6: Run the full test suite for coalesce**

Run: `.venv/bin/python -m pytest tests/unit/core/landscape/test_token_recording.py tests/unit/core/landscape/test_data_flow_repository.py tests/unit/engine/test_tokens.py tests/integration/audit/test_recorder_tokens.py -v -x`
Expected: ALL PASS

---

### Task 5: Update Guard Symmetry Allowlist

**Files:**
- Modify: `config/cicd/enforce_guard_symmetry/landscape.yaml`

**Context:** After Tasks 1-3, RunLoader, CallLoader, and OperationLoader now have explicit AuditIntegrityError guards. The guard symmetry scanner should no longer flag them as findings. The `max_hits` must be reduced from 12 to 9 (removing the 3 fixed loaders from the finding count).

- [ ] **Step 1: Run the guard symmetry scanner to get the new count**

Run: `.venv/bin/python scripts/cicd/enforce_guard_symmetry.py check --root src/elspeth`
Expected: Observe new finding count (should be 9 if only RunLoader, CallLoader, OperationLoader were fixed).

- [ ] **Step 2: Update the allowlist**

In `config/cicd/enforce_guard_symmetry/landscape.yaml`, change `max_hits: 12` to `max_hits: 9` and update the comment:

```yaml
per_file_rules:
  - pattern: "core/landscape/model_loaders.py"
    rules: [GS1]
    reason: >-
      9 loaders where __post_init__ validation fires during load(),
      providing read-side coverage without explicit AuditIntegrityError.
      Int-only: RowLoader, TokenLoader, TokenParentLoader, ArtifactLoader,
      BatchMemberLoader. Enum+int: EdgeLoader, BatchLoader, RoutingEventLoader.
      Plugin: ValidationErrorLoader (pairs with validation.py, not audit.py).
    expires: null
    max_hits: 9
```

- [ ] **Step 3: Re-run scanner to verify**

Run: `.venv/bin/python scripts/cicd/enforce_guard_symmetry.py check --root src/elspeth`
Expected: PASS (findings ≤ max_hits)

---

### Task 6: Defer Option A (TokenRef Field Replacement) with Acceptance Criteria

**Files:**
- Filigree issue: `elspeth-dbfa04ee89`

**Context:** The deep refactor to embed `TokenRef` directly in Token, TokenOutcome, and TransformErrorRecord dataclasses affects ~80+ sites. This is a separate body of work that should be planned independently when prioritized. For now, transition the issue from `proposed` to `approved` with concrete acceptance criteria.

- [ ] **Step 1: Add acceptance criteria to the issue**

Use Filigree MCP to update `elspeth-dbfa04ee89`:

```
Acceptance Criteria:
1. Token, TokenOutcome, TransformErrorRecord replace (token_id: str, run_id: str) with (token_ref: TokenRef)
2. Convenience properties .token_id and .run_id delegate to token_ref for read access
3. All 3 loaders (TokenLoader, TokenOutcomeLoader, TransformErrorLoader) construct TokenRef from DB rows
4. All ~48 test fixtures updated to construct via TokenRef
5. to_dict() methods decompose TokenRef back to separate fields for DB writes
6. No net regression in unit or integration tests
```

- [ ] **Step 2: Add comment explaining deferral rationale**

Add comment to `elspeth-dbfa04ee89`:

```
Deferring to a separate planning cycle. The coalesce_tokens TokenRef adoption (elspeth-bb432582c3) provides the immediate type-safety win for the multi-token parameter pattern. This deeper refactor is valuable but can wait — the existing __post_init__ validation on the dataclasses catches mismatched pairs at construction time, so the risk of silent bugs is low.
```

---

### Task 7: Commit and Close Issues

- [ ] **Step 1: Run full test suite**

Run: `.venv/bin/python -m pytest tests/unit/core/landscape/test_model_loaders.py tests/unit/core/landscape/test_token_recording.py tests/unit/core/landscape/test_data_flow_repository.py tests/unit/engine/test_tokens.py tests/integration/audit/ -v --tb=short`
Expected: ALL PASS

- [ ] **Step 2: Run guard symmetry scanner**

Run: `.venv/bin/python scripts/cicd/enforce_guard_symmetry.py check --root src/elspeth`
Expected: PASS

- [ ] **Step 3: Run tier model enforcement**

Run: `.venv/bin/python scripts/cicd/enforce_tier_model.py check --root src/elspeth --allowlist config/cicd/enforce_tier_model`
Expected: PASS (no new tier violations from loader guard code)

- [ ] **Step 4: Run type checker**

Run: `.venv/bin/python -m mypy src/elspeth/core/landscape/model_loaders.py src/elspeth/core/landscape/data_flow_repository.py src/elspeth/core/landscape/recorder.py src/elspeth/engine/tokens.py`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/elspeth/core/landscape/model_loaders.py \
        src/elspeth/core/landscape/data_flow_repository.py \
        src/elspeth/core/landscape/recorder.py \
        src/elspeth/engine/tokens.py \
        src/elspeth/engine/coalesce_executor.py \
        config/cicd/enforce_guard_symmetry/landscape.yaml \
        tests/unit/core/landscape/test_model_loaders.py \
        tests/unit/core/landscape/test_token_recording.py \
        tests/unit/core/landscape/test_data_flow_repository.py \
        tests/unit/engine/test_tokens.py \
        tests/integration/audit/test_recorder_tokens.py \
        tests/integration/audit/test_recorder_queries.py \
        tests/property/engine/test_token_lifecycle_state_machine.py \
        tests/property/engine/test_coalesce_properties.py
git commit -m "feat(landscape): AuditIntegrityError loader guards + TokenRef coalesce_tokens

Add explicit read-side guards to RunLoader (status/completed_at lifecycle),
CallLoader (state_id/operation_id XOR), and OperationLoader (full lifecycle
state machine). Adopt list[TokenRef] in coalesce_tokens for type-level
cross-run safety. Update guard symmetry allowlist max_hits 12→9.

Closes: elspeth-971113c6b0, elspeth-bb432582c3"
```

- [ ] **Step 6: Close Filigree issues**

Close `elspeth-971113c6b0` and `elspeth-bb432582c3` via MCP tools.
