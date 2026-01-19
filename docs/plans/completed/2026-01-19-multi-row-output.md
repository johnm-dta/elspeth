# Multi-Row Output Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Enable transforms to output multiple rows from batch inputs (passthrough/transform modes) and single inputs (deaggregation).

**Architecture:** Extend `TransformResult` to support multi-row output via a new `rows` field. The processor checks `output_mode` from aggregation settings and `creates_tokens` flag on transforms to determine how to handle multi-row results. Each output row gets proper token lineage via a new `LandscapeRecorder.expand_token()` method.

**Tech Stack:** Python dataclasses, existing TokenManager for child token creation, LandscapeRecorder for audit trail

---

## Background

Currently:
- `TransformResult.row` holds ONE output row
- Aggregation only supports `output_mode: single` (N rows → 1 row)
- Transforms cannot expand rows (1 → N)

After this change:
- `TransformResult.rows` can hold MULTIPLE output rows
- `output_mode: passthrough` returns N enriched rows from N input rows
- `output_mode: transform` returns M rows from N input rows
- `creates_tokens=True` transforms can return multiple rows from single input (deaggregation)

## Token Lineage

Multi-row output creates audit trail challenges. The solution:
- **Passthrough mode**: Each output row inherits the token_id of its corresponding input row (no new tokens created)
- **Transform mode**: Output rows get new token_ids with parent linkage to batch triggering token
- **Deaggregation**: Output rows get new token_ids with parent linkage to input token

**IMPORTANT:** Token expansion uses `LandscapeRecorder.expand_token()` (added in Task 2), which records parent relationships in `token_parents` table with ordinal tracking.

### Transform Mode Parent Linkage Tradeoff

**Design decision:** In transform mode, the **triggering token** (the Nth row that caused the batch to flush) becomes the parent for all output tokens in `token_parents`.

**Why not link to ALL input tokens?** Creating N×M relationships (N inputs × M outputs) would explode the audit trail for large batches. The triggering token provides a single anchor point.

**How auditors recover full lineage:**
1. Query `token_parents` to find the parent token (triggering token)
2. Query `batch_members` for the batch_id associated with that parent token
3. Query `batch_members` again to find ALL input tokens in that batch
4. The complete lineage is: batch_members → triggering token → output tokens

**Alternative considered:** Store batch_id on output tokens. Rejected because it conflates batch membership (an aggregation concept) with token lineage (a DAG concept).

## Key Design Decisions

### Multi-Row vs Token Creation

There are TWO distinct concepts:

1. **Multi-row output**: Transform returns `TransformResult.success_multi(rows)` instead of `success(row)`
2. **Token creation**: New token_ids are generated for output rows

| Mode | Multi-row? | Creates tokens? | Buffer outcome | Use case |
|------|------------|-----------------|----------------|----------|
| Deaggregation | Yes | Yes | N/A (not batched) | 1 input → N outputs (e.g., JSON explode) |
| Passthrough | Yes | No | `BUFFERED` | N inputs → N enriched outputs (same tokens) |
| Transform | Yes | Yes | `CONSUMED_IN_BATCH` | N inputs → M outputs (batch aggregation) |
| Single | No | No | `CONSUMED_IN_BATCH` | N inputs → 1 output (e.g., batch_stats) |

The `creates_tokens` flag on transforms distinguishes deaggregation (which creates new tokens) from passthrough mode (which preserves original tokens).

### Buffer Outcome Semantics

Two distinct outcomes for buffered rows:

- **`CONSUMED_IN_BATCH`** (terminal): Token is absorbed into an aggregate. Will never reappear. Used by single and transform modes where the batch produces NEW tokens.
- **`BUFFERED`** (non-terminal): Token is held for enrichment. Will reappear as `COMPLETED` on flush. Used by passthrough mode where the SAME tokens continue.

### Triggering Row Semantics

For aggregation with batch size N:
- **single/transform modes**: Rows 1 to N all get `CONSUMED_IN_BATCH` (terminal)
- **passthrough mode**: Rows 1 to N-1 get `BUFFERED` (non-terminal), row N triggers flush, all N get `COMPLETED`

The triggering row is part of the batch, not separate from it.

### Fork/Join Interaction

**Out of scope for this plan.** If a forked path contains an expansion that changes row count, coalesce behavior is undefined. Document this limitation and address in a future plan.

### Error Handling in Batch Modes

When a batch transform returns `error`:
- **All modes are atomic**: ALL buffered rows fail together (including the triggering row)
- This is consistent and auditable - a batch either succeeds completely or fails completely
- Failed rows get `RowOutcome.FAILED` with error details from `TransformResult.reason`

---

### Task 0: Prerequisite Verification

**Purpose:** Verify assumptions about the current codebase before making changes.

**Step 1: Verify current signatures and find call sites**

```bash
# Verify flush_buffer current signature
grep -n "def flush_buffer" src/elspeth/engine/executors.py

# Verify no existing expand_group_id
grep -rn "expand_group_id" src/elspeth/

# Count flush_buffer call sites (IMPORTANT: update all of these in Task 6)
grep -rn "flush_buffer" src/elspeth/engine/
grep -rn "flush_buffer" tests/

# Check TransformExecutor assertion that needs updating
grep -n "success status requires row data" src/elspeth/engine/executors.py

# Check Token construction sites that need expand_group_id
grep -rn "TokenRepository.*load\|def load.*Token\|Token(" src/elspeth/core/landscape/

# IMPORTANT: Check how processor accesses transforms (for Task 8 helper method)
grep -n "def process_row\|self._transforms\|transforms:" src/elspeth/engine/processor.py

# Check quarantine recording mechanism (for Task 11 integration tests)
grep -rn "quarantine\|QuarantineEvent" src/elspeth/core/landscape/

# IMPORTANT: Find _process_batch_aggregation_node callers (Task 8 changes return type)
grep -rn "_process_batch_aggregation_node" src/elspeth/engine/

# Verify validation_errors table exists for quarantine audit (Task 11 tests)
grep -n "validation_errors_table" src/elspeth/core/landscape/schema.py

# IMPORTANT: Verify Orchestrator interface for Task 11 integration tests
grep -n "class Orchestrator\|class RunResult\|def run(" src/elspeth/engine/orchestrator.py

# IMPORTANT: Verify complete_node_state signature for Task 1 update
grep -n "def complete_node_state" src/elspeth/core/landscape/recorder.py
```

**Step 2: Document findings**

Record the line numbers and file paths for:
- `flush_buffer()` call sites - **COMPLETE LIST (update all in Task 6):**
  - Production: `src/elspeth/engine/processor.py:176`
  - Test: `tests/engine/test_executors.py:2420`
- `TokenRepository.load()` method (expect: repositories.py:141)
- TransformExecutor assertion location (expect: executors.py ~line 197)
- How `process_row()` receives transforms: **CONFIRMED via parameter** `transforms: list[Any]` at processor.py:291
- Quarantine recording table/method name (expect: `validation_errors` table via `record_validation_error()`)
- `_process_batch_aggregation_node` callers (expect: single call site at processor.py:451)
- `Orchestrator` interface: `class Orchestrator` at orchestrator.py:91, `class RunResult` at orchestrator.py:67, `result.status` uses `RunStatus` enum
- `complete_node_state` signature: recorder.py:965, `output_data: dict[str, Any] | None` - **needs updating for multi-row**

**Step 3: No commit** (verification only)

---

### Task 1: Extend TransformResult for Multi-Row Output

**Files:**
- Modify: `src/elspeth/contracts/results.py`
- Modify: `src/elspeth/engine/executors.py` (fix assertion)
- Test: `tests/contracts/test_results.py`

**Step 1: Write the failing tests**

```python
# tests/contracts/test_results.py

import pytest
from elspeth.contracts.results import TransformResult


def test_transform_result_multi_row_success():
    """TransformResult.success_multi returns multiple rows."""
    rows = [{"id": 1, "value": "a"}, {"id": 2, "value": "b"}]
    result = TransformResult.success_multi(rows)

    assert result.status == "success"
    assert result.row is None  # Single row field is None
    assert result.rows == rows
    assert len(result.rows) == 2


def test_transform_result_success_single_sets_rows_none():
    """TransformResult.success() sets rows to None for single-row output."""
    result = TransformResult.success({"id": 1})

    assert result.status == "success"
    assert result.row == {"id": 1}
    assert result.rows is None


def test_transform_result_is_multi_row():
    """is_multi_row property distinguishes single vs multi output."""
    single = TransformResult.success({"id": 1})
    multi = TransformResult.success_multi([{"id": 1}, {"id": 2}])

    assert single.is_multi_row is False
    assert multi.is_multi_row is True


def test_transform_result_success_multi_rejects_empty_list():
    """success_multi raises ValueError for empty list."""
    with pytest.raises(ValueError, match="at least one row"):
        TransformResult.success_multi([])


def test_transform_result_error_has_rows_none():
    """TransformResult.error() sets rows to None."""
    result = TransformResult.error({"reason": "failed"})

    assert result.status == "error"
    assert result.row is None
    assert result.rows is None


def test_transform_result_has_output_data():
    """has_output_data property checks if ANY output exists."""
    single = TransformResult.success({"id": 1})
    multi = TransformResult.success_multi([{"id": 1}])
    error = TransformResult.error({"reason": "failed"})

    assert single.has_output_data is True
    assert multi.has_output_data is True
    assert error.has_output_data is False
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/contracts/test_results.py::test_transform_result_multi_row_success -v`
Expected: FAIL with "AttributeError: type object 'TransformResult' has no attribute 'success_multi'"

**Step 3: Write minimal implementation**

```python
# src/elspeth/contracts/results.py - modify TransformResult class

@dataclass
class TransformResult:
    """Result of a transform operation.

    Use the factory methods to create instances.

    IMPORTANT: status uses Literal["success", "error"], NOT enum, per architecture.
    Audit fields (input_hash, output_hash, duration_ms) are populated by executors.

    For multi-row output (batch transforms, deaggregation):
    - Use success_multi(rows) instead of success(row)
    - rows field contains list of output dicts
    - row field is None when rows is set
    - Use has_output_data to check if ANY output exists
    """

    status: Literal["success", "error"]
    row: dict[str, Any] | None
    reason: dict[str, Any] | None
    retryable: bool = False

    # Multi-row output support
    rows: list[dict[str, Any]] | None = None

    # Audit fields - set by executor, not by plugin
    input_hash: str | None = field(default=None, repr=False)
    output_hash: str | None = field(default=None, repr=False)
    duration_ms: float | None = field(default=None, repr=False)

    @property
    def is_multi_row(self) -> bool:
        """Check if this result contains multiple output rows."""
        return self.rows is not None

    @property
    def has_output_data(self) -> bool:
        """Check if this result has ANY output data (single or multi)."""
        return self.row is not None or self.rows is not None

    @classmethod
    def success(cls, row: dict[str, Any]) -> "TransformResult":
        """Create successful result with single output row."""
        return cls(status="success", row=row, reason=None, rows=None)

    @classmethod
    def success_multi(cls, rows: list[dict[str, Any]]) -> "TransformResult":
        """Create successful result with multiple output rows.

        Use for:
        - Batch transforms with passthrough/transform output_mode
        - Deaggregation transforms (1 input -> N outputs)

        Args:
            rows: List of output row dicts (must not be empty)

        Raises:
            ValueError: If rows is empty
        """
        if not rows:
            raise ValueError("success_multi requires at least one row")
        return cls(status="success", row=None, reason=None, rows=rows)

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

**Step 4: Fix TransformExecutor assertion**

The existing assertion at ~line 197 of `executors.py` will break for multi-row results:
```python
# BEFORE (breaks for multi-row):
assert result.row is not None, "success status requires row data"

# AFTER (supports both):
assert result.has_output_data, "success status requires row or rows data"
```

Also update the output_hash calculation:
```python
# BEFORE:
result.output_hash = stable_hash(result.row) if result.row else None

# AFTER:
if result.row is not None:
    result.output_hash = stable_hash(result.row)
elif result.rows is not None:
    result.output_hash = stable_hash(result.rows)
else:
    result.output_hash = None
```

**Step 5: Update LandscapeRecorder.complete_node_state() signature**

The `LandscapeRecorder.complete_node_state()` method at `recorder.py:965` currently has:
```python
def complete_node_state(..., output_data: dict[str, Any] | None = None, ...)
```

Update to accept `list[dict[str, Any]]` for multi-row output:
```python
# src/elspeth/core/landscape/recorder.py - modify complete_node_state signature
def complete_node_state(
    self,
    state_id: str,
    status: NodeStateStatus | str,
    *,
    output_data: dict[str, Any] | list[dict[str, Any]] | None = None,  # UPDATED
    duration_ms: float | None = None,
    error: ExecutionError | dict[str, Any] | None = None,
    context_after: dict[str, Any] | None = None,
) -> NodeStateCompleted | NodeStateFailed:
```

The `output_data` is hashed for audit trail. For multi-row, hash the entire list (which `stable_hash()` already handles via canonical JSON of list).

**Step 6: Run test to verify it passes**

Run: `uv run pytest tests/contracts/test_results.py -v -k "transform_result"`
Expected: PASS

**Step 7: Commit**

```bash
git add src/elspeth/contracts/results.py src/elspeth/engine/executors.py src/elspeth/core/landscape/recorder.py tests/contracts/test_results.py
git commit -m "feat(contracts): add multi-row support to TransformResult

- Add TransformResult.rows field and success_multi() factory
- Add is_multi_row and has_output_data properties
- Update TransformExecutor assertion for multi-row
- Update complete_node_state to accept list[dict] output_data"
```

---

### Task 2: Add expand_token() to LandscapeRecorder

**Rationale:** Token expansion needs proper audit trail recording. The existing `fork_token()` creates tokens for parallel DAG paths with branch names. Expansion creates tokens for sequential children from a single parent (deaggregation). We need a dedicated method to maintain clear audit semantics.

**Files:**
- Modify: `src/elspeth/core/landscape/recorder.py`
- Modify: `src/elspeth/core/landscape/schema.py` (add expand_group_id column)
- Modify: `src/elspeth/contracts/audit.py` (add expand_group_id to Token)
- Modify: `src/elspeth/core/landscape/repositories.py` (update TokenRepository.load)
- Test: `tests/core/landscape/test_recorder.py`

**Step 1: Write the failing test**

```python
# tests/core/landscape/test_recorder.py

import pytest
from elspeth.core.landscape import LandscapeDB, LandscapeRecorder
from elspeth.contracts.enums import NodeType, Determinism
from elspeth.contracts.schema import SchemaConfig


def test_expand_token_creates_children_with_parent_relationship():
    """expand_token creates child tokens linked to parent via token_parents."""
    db = LandscapeDB.in_memory()
    recorder = LandscapeRecorder(db)

    # Setup: create run, node, row, and parent token
    run = recorder.begin_run(config={"test": True}, canonical_version="1.0")
    node = recorder.register_node(
        run_id=run.run_id,
        plugin_name="json_explode",
        node_type=NodeType.TRANSFORM,
        plugin_version="1.0.0",
        config={},
        determinism=Determinism.DETERMINISTIC,
        schema_config=SchemaConfig(fields="dynamic"),
    )
    row = recorder.create_row(
        run_id=run.run_id,
        source_node_id=node.node_id,
        row_index=0,
        data={"items": [1, 2, 3]},
    )
    parent_token = recorder.create_token(row_id=row.row_id)

    # Act: expand parent into 3 children
    children = recorder.expand_token(
        parent_token_id=parent_token.token_id,
        row_id=row.row_id,
        count=3,
        step_in_pipeline=2,
    )

    # Assert: 3 children created
    assert len(children) == 3

    # All children share same row_id (same source row)
    for child in children:
        assert child.row_id == row.row_id
        assert child.token_id != parent_token.token_id

    # All children share same expand_group_id
    expand_group_ids = {c.expand_group_id for c in children}
    assert len(expand_group_ids) == 1
    assert None not in expand_group_ids

    # Verify parent relationships recorded
    for i, child in enumerate(children):
        parents = recorder.get_token_parents(child.token_id)
        assert len(parents) == 1
        assert parents[0].parent_token_id == parent_token.token_id
        assert parents[0].ordinal == i


def test_expand_token_with_zero_count_raises():
    """expand_token raises ValueError for count=0."""
    db = LandscapeDB.in_memory()
    recorder = LandscapeRecorder(db)

    run = recorder.begin_run(config={}, canonical_version="1.0")
    node = recorder.register_node(
        run_id=run.run_id,
        plugin_name="csv",
        node_type=NodeType.SOURCE,
        plugin_version="1.0.0",
        config={},
        determinism=Determinism.IO_READ,
        schema_config=SchemaConfig(fields="dynamic"),
    )
    row = recorder.create_row(
        run_id=run.run_id,
        source_node_id=node.node_id,
        row_index=0,
        data={},
    )
    token = recorder.create_token(row_id=row.row_id)

    with pytest.raises(ValueError, match="at least 1"):
        recorder.expand_token(
            parent_token_id=token.token_id,
            row_id=row.row_id,
            count=0,
            step_in_pipeline=1,
        )
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/core/landscape/test_recorder.py::test_expand_token_creates_children_with_parent_relationship -v`
Expected: FAIL with "AttributeError: 'LandscapeRecorder' object has no attribute 'expand_token'"

**Step 3: Add expand_group_id column to tokens table**

**MIGRATION NOTE:** Adding this column requires either:
- For development/tests: `metadata.create_all()` will create new tables with the column
- For existing databases: Create an Alembic migration:
  ```bash
  alembic revision --autogenerate -m "Add expand_group_id to tokens"
  alembic upgrade head
  ```

```python
# src/elspeth/core/landscape/schema.py - modify tokens_table

tokens_table = Table(
    "tokens",
    metadata,
    Column("token_id", String(32), primary_key=True),
    Column("row_id", String(32), ForeignKey("rows.row_id"), nullable=False, index=True),
    Column("fork_group_id", String(32), nullable=True, index=True),
    Column("join_group_id", String(32), nullable=True, index=True),
    Column("expand_group_id", String(32), nullable=True, index=True),  # NEW: for deaggregation
    Column("branch_name", String(255), nullable=True),
    Column("step_in_pipeline", Integer, nullable=True),
    Column("created_at", DateTime(timezone=True), nullable=False),
)
```

**Step 4: Add expand_group_id to Token contract**

```python
# src/elspeth/contracts/audit.py - modify Token dataclass

@dataclass
class Token:
    """A row instance flowing through a specific DAG path."""

    token_id: str
    row_id: str
    created_at: datetime
    fork_group_id: str | None = None
    join_group_id: str | None = None
    expand_group_id: str | None = None  # NEW: for deaggregation grouping
    branch_name: str | None = None
    step_in_pipeline: int | None = None
```

**Step 5: Update TokenRepository.load() in repositories.py**

Update the `TokenRepository.load()` method to include `expand_group_id`:

```python
# src/elspeth/core/landscape/repositories.py - update TokenRepository.load method

class TokenRepository:
    def load(self, row: Any) -> Token:
        """Load Token from database row."""
        return Token(
        token_id=row.token_id,
        row_id=row.row_id,
        created_at=row.created_at,
        fork_group_id=row.fork_group_id,
        join_group_id=row.join_group_id,
        expand_group_id=row.expand_group_id,  # NEW
        branch_name=row.branch_name,
        step_in_pipeline=row.step_in_pipeline,
    )
```

**Step 6: Implement expand_token() method**

```python
# src/elspeth/core/landscape/recorder.py - add method to LandscapeRecorder

def expand_token(
    self,
    parent_token_id: str,
    row_id: str,
    count: int,
    step_in_pipeline: int,
) -> list[Token]:
    """Expand a token into multiple child tokens (deaggregation).

    Creates N child tokens from a single parent for 1→N expansion.
    All children share the same row_id (same source row) and are
    linked to the parent via token_parents table.

    Unlike fork_token (parallel DAG paths with branch names), expand_token
    creates sequential children for deaggregation transforms.

    Args:
        parent_token_id: Token being expanded
        row_id: Row ID (same for all children)
        count: Number of child tokens to create (must be >= 1)
        step_in_pipeline: Step where expansion occurs

    Returns:
        List of child Token models

    Raises:
        ValueError: If count < 1
    """
    if count < 1:
        raise ValueError("expand_token requires at least 1 child")

    expand_group_id = _generate_id()
    children = []

    with self._db.connection() as conn:
        for ordinal in range(count):
            child_id = _generate_id()
            now = _now()

            # Create child token with expand_group_id
            conn.execute(
                tokens_table.insert().values(
                    token_id=child_id,
                    row_id=row_id,
                    expand_group_id=expand_group_id,
                    step_in_pipeline=step_in_pipeline,
                    created_at=now,
                )
            )

            # Record parent relationship
            conn.execute(
                token_parents_table.insert().values(
                    token_id=child_id,
                    parent_token_id=parent_token_id,
                    ordinal=ordinal,
                )
            )

            children.append(
                Token(
                    token_id=child_id,
                    row_id=row_id,
                    expand_group_id=expand_group_id,
                    step_in_pipeline=step_in_pipeline,
                    created_at=now,
                )
            )

    return children
```

**Step 7: Run test to verify it passes**

Run: `uv run pytest tests/core/landscape/test_recorder.py -v -k "expand_token"`
Expected: PASS

**Step 8: Commit**

```bash
git add src/elspeth/core/landscape/recorder.py src/elspeth/core/landscape/schema.py src/elspeth/core/landscape/repositories.py src/elspeth/contracts/audit.py tests/core/landscape/test_recorder.py
git commit -m "feat(landscape): add expand_token for deaggregation audit trail"
```

---

### Task 3: Add creates_tokens Flag to BaseTransform

**Rationale:** We need to distinguish transforms that CREATE new tokens (deaggregation) from those that just return multiple rows (passthrough). The flag `creates_tokens` is clearer than `can_expand`.

**Files:**
- Modify: `src/elspeth/plugins/base.py`
- Modify: `src/elspeth/plugins/protocols.py`
- Test: `tests/plugins/test_base.py`

**Step 1: Write the failing test**

```python
# tests/plugins/test_base.py

from elspeth.plugins.base import BaseTransform
from elspeth.contracts.results import TransformResult


def test_base_transform_creates_tokens_default_false():
    """BaseTransform.creates_tokens defaults to False."""

    class SimpleTransform(BaseTransform):
        name = "simple"
        input_schema = None  # Not needed for this test
        output_schema = None

        def process(self, row, ctx):
            return TransformResult.success(row)

    transform = SimpleTransform({})
    assert transform.creates_tokens is False


def test_base_transform_creates_tokens_settable():
    """BaseTransform.creates_tokens can be overridden to True."""

    class ExpandingTransform(BaseTransform):
        name = "expander"
        creates_tokens = True  # Deaggregation transform
        input_schema = None
        output_schema = None

        def process(self, row, ctx):
            return TransformResult.success_multi([row, row])

    transform = ExpandingTransform({})
    assert transform.creates_tokens is True
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/plugins/test_base.py::test_base_transform_creates_tokens_default_false -v`
Expected: FAIL with "AttributeError: 'SimpleTransform' object has no attribute 'creates_tokens'"

**Step 3: Write minimal implementation**

```python
# src/elspeth/plugins/base.py - add to BaseTransform class attributes (after is_batch_aware)

    # Token creation flag for deaggregation transforms
    # When True AND process() returns success_multi(), the processor creates
    # new token_ids for each output row with parent linkage to input token.
    # When False AND success_multi() is returned, the processor expects
    # passthrough mode (same number of outputs as inputs, preserve token_ids).
    # Default: False (most transforms don't create new tokens)
    creates_tokens: bool = False
```

```python
# src/elspeth/plugins/protocols.py - add to TransformProtocol (after is_batch_aware)

    # Token creation flag for deaggregation
    # When True, process() may return TransformResult.success_multi(rows)
    # and new tokens will be created for each output row.
    # When False, success_multi() is only valid in passthrough aggregation mode.
    creates_tokens: bool
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/plugins/test_base.py -v -k "creates_tokens"`
Expected: PASS

**Step 5: Commit**

```bash
git add src/elspeth/plugins/base.py src/elspeth/plugins/protocols.py tests/plugins/test_base.py
git commit -m "feat(plugins): add creates_tokens flag for deaggregation transforms"
```

---

### Task 4: Create TokenManager.expand_token() Method

**Files:**
- Modify: `src/elspeth/engine/tokens.py`
- Test: `tests/engine/test_tokens.py`

**Step 1: Write the failing test**

```python
# tests/engine/test_tokens.py

from elspeth.engine.tokens import TokenManager
from elspeth.contracts.identity import TokenInfo
from elspeth.core.landscape import LandscapeDB, LandscapeRecorder
from elspeth.contracts.enums import NodeType, Determinism
from elspeth.contracts.schema import SchemaConfig


def test_expand_token_creates_children():
    """expand_token creates child tokens for each expanded row."""
    # Use real recorder for integration test
    db = LandscapeDB.in_memory()
    recorder = LandscapeRecorder(db)

    # Setup
    run = recorder.begin_run(config={}, canonical_version="1.0")
    node = recorder.register_node(
        run_id=run.run_id,
        plugin_name="csv",
        node_type=NodeType.SOURCE,
        plugin_version="1.0.0",
        config={},
        determinism=Determinism.IO_READ,
        schema_config=SchemaConfig(fields="dynamic"),
    )
    row = recorder.create_row(
        run_id=run.run_id,
        source_node_id=node.node_id,
        row_index=0,
        data={"original": "data"},
    )
    db_token = recorder.create_token(row_id=row.row_id)

    manager = TokenManager(recorder)

    parent = TokenInfo(
        row_id=row.row_id,
        token_id=db_token.token_id,
        row_data={"original": "data"},
    )

    expanded_rows = [
        {"id": 1, "value": "a"},
        {"id": 2, "value": "b"},
        {"id": 3, "value": "c"},
    ]

    # Act
    children = manager.expand_token(
        parent_token=parent,
        expanded_rows=expanded_rows,
        step_in_pipeline=2,
    )

    # Assert: correct number of children
    assert len(children) == 3

    # All children share same row_id (same source row)
    for child in children:
        assert child.row_id == row.row_id
        assert child.token_id != parent.token_id

    # Each child has its expanded row data
    assert children[0].row_data == {"id": 1, "value": "a"}
    assert children[1].row_data == {"id": 2, "value": "b"}
    assert children[2].row_data == {"id": 3, "value": "c"}

    # Verify parent relationships in database
    for i, child in enumerate(children):
        parents = recorder.get_token_parents(child.token_id)
        assert len(parents) == 1
        assert parents[0].parent_token_id == parent.token_id
        assert parents[0].ordinal == i


def test_expand_token_inherits_branch_name():
    """expand_token children inherit parent's branch_name."""
    db = LandscapeDB.in_memory()
    recorder = LandscapeRecorder(db)

    run = recorder.begin_run(config={}, canonical_version="1.0")
    node = recorder.register_node(
        run_id=run.run_id,
        plugin_name="csv",
        node_type=NodeType.SOURCE,
        plugin_version="1.0.0",
        config={},
        determinism=Determinism.IO_READ,
        schema_config=SchemaConfig(fields="dynamic"),
    )
    row = recorder.create_row(
        run_id=run.run_id,
        source_node_id=node.node_id,
        row_index=0,
        data={},
    )
    db_token = recorder.create_token(row_id=row.row_id)

    manager = TokenManager(recorder)

    parent = TokenInfo(
        row_id=row.row_id,
        token_id=db_token.token_id,
        row_data={},
        branch_name="stats_branch",
    )

    children = manager.expand_token(
        parent_token=parent,
        expanded_rows=[{"a": 1}, {"a": 2}],
        step_in_pipeline=1,
    )

    # Children inherit branch_name
    assert all(c.branch_name == "stats_branch" for c in children)
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/engine/test_tokens.py::test_expand_token_creates_children -v`
Expected: FAIL with "AttributeError: 'TokenManager' object has no attribute 'expand_token'"

**Step 3: Write minimal implementation**

```python
# src/elspeth/engine/tokens.py - add method to TokenManager class

def expand_token(
    self,
    parent_token: TokenInfo,
    expanded_rows: list[dict[str, Any]],
    step_in_pipeline: int,
) -> list[TokenInfo]:
    """Create child tokens for deaggregation (1 input -> N outputs).

    Unlike fork_token (which creates parallel paths through the same DAG),
    expand_token creates sequential children that all continue down the
    same path. Used when a transform outputs multiple rows from single input.

    Args:
        parent_token: The token being expanded
        expanded_rows: List of output row dicts
        step_in_pipeline: Current step (for audit)

    Returns:
        List of child TokenInfo, one per expanded row
    """
    # Delegate to recorder which handles DB operations and parent linking
    db_children = self._recorder.expand_token(
        parent_token_id=parent_token.token_id,
        row_id=parent_token.row_id,
        count=len(expanded_rows),
        step_in_pipeline=step_in_pipeline,
    )

    # Build TokenInfo objects with row data
    return [
        TokenInfo(
            row_id=parent_token.row_id,
            token_id=db_child.token_id,
            row_data=row_data,
            branch_name=parent_token.branch_name,  # Inherit branch
        )
        for db_child, row_data in zip(db_children, expanded_rows)
    ]
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/engine/test_tokens.py -v -k "expand_token"`
Expected: PASS

**Step 5: Commit**

```bash
git add src/elspeth/engine/tokens.py tests/engine/test_tokens.py
git commit -m "feat(engine): add TokenManager.expand_token for deaggregation"
```

---

### Task 5: Add EXPANDED and BUFFERED to RowOutcome Enum

**Rationale:** We need two new outcomes:
- `EXPANDED`: Terminal state for parent tokens that were deaggregated into children
- `BUFFERED`: Non-terminal state for tokens held in passthrough mode (will reappear as `COMPLETED`)

The distinction between terminal and non-terminal outcomes is important for orchestrator logic and audit queries.

**Files:**
- Modify: `src/elspeth/contracts/enums.py`
- Test: `tests/contracts/test_enums.py`

**Step 1: Write the tests**

```python
# tests/contracts/test_enums.py

from elspeth.contracts.enums import RowOutcome


def test_row_outcome_expanded_exists():
    """RowOutcome.EXPANDED is available for deaggregation."""
    assert RowOutcome.EXPANDED.value == "expanded"


def test_row_outcome_buffered_exists():
    """RowOutcome.BUFFERED is available for passthrough batching."""
    assert RowOutcome.BUFFERED.value == "buffered"


def test_row_outcome_buffered_is_not_terminal():
    """BUFFERED is non-terminal - token will reappear with final outcome."""
    assert RowOutcome.BUFFERED.is_terminal is False


def test_row_outcome_consumed_in_batch_is_terminal():
    """CONSUMED_IN_BATCH is terminal - token is absorbed into aggregate."""
    assert RowOutcome.CONSUMED_IN_BATCH.is_terminal is True


def test_row_outcome_expanded_is_terminal():
    """EXPANDED is terminal - parent token's journey ends, children continue."""
    assert RowOutcome.EXPANDED.is_terminal is True


def test_row_outcome_completed_is_terminal():
    """COMPLETED is terminal."""
    assert RowOutcome.COMPLETED.is_terminal is True


def test_all_outcomes_have_is_terminal():
    """All RowOutcome values have is_terminal property."""
    for outcome in RowOutcome:
        # Should not raise - property exists for all values
        _ = outcome.is_terminal
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/contracts/test_enums.py::test_row_outcome_buffered_exists -v`
Expected: FAIL with "AttributeError: 'RowOutcome' object has no attribute 'BUFFERED'"

**Step 3: Write minimal implementation**

```python
# src/elspeth/contracts/enums.py - add to RowOutcome enum

class RowOutcome(Enum):
    """Outcome for a token in the pipeline.

    IMPORTANT: These are DERIVED at query time from node_states,
    routing_events, and batch_members - NOT stored in the database.
    Therefore this is plain Enum, not (str, Enum).

    If you need the string value, use .value explicitly.

    Most outcomes are TERMINAL - the token's journey is complete:
    - COMPLETED: Reached output sink successfully
    - ROUTED: Sent to named sink by gate
    - FORKED: Split into multiple parallel paths (parent token)
    - FAILED: Processing failed, not recoverable
    - QUARANTINED: Failed validation, stored for investigation
    - CONSUMED_IN_BATCH: Absorbed into aggregate (single/transform mode)
    - COALESCED: Merged in join from parallel paths
    - EXPANDED: Deaggregated into child tokens (parent token)

    One outcome is NON-TERMINAL - the token will reappear:
    - BUFFERED: Held for batch processing in passthrough mode
    """

    # Terminal outcomes
    COMPLETED = "completed"
    ROUTED = "routed"
    FORKED = "forked"
    FAILED = "failed"
    QUARANTINED = "quarantined"
    CONSUMED_IN_BATCH = "consumed_in_batch"
    COALESCED = "coalesced"
    EXPANDED = "expanded"

    # Non-terminal outcomes
    BUFFERED = "buffered"

    @property
    def is_terminal(self) -> bool:
        """Check if this outcome represents a final state for the token.

        Terminal outcomes mean the token's journey is complete - it won't
        appear again in results. Non-terminal outcomes (BUFFERED) mean
        the token is temporarily held and will reappear with a final outcome.
        """
        return self != RowOutcome.BUFFERED
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/contracts/test_enums.py -v -k "row_outcome"`
Expected: PASS

**Step 5: Commit**

```bash
git add src/elspeth/contracts/enums.py tests/contracts/test_enums.py
git commit -m "feat(contracts): add EXPANDED and BUFFERED to RowOutcome with is_terminal property"
```

---

### Task 6: Modify AggregationExecutor.flush_buffer() to Return Tokens

**Rationale:** The current `flush_buffer()` clears the token buffer before callers can access it for passthrough mode. We need to return both rows AND tokens together.

**Files:**
- Modify: `src/elspeth/engine/executors.py`
- Modify: `src/elspeth/engine/processor.py` (update ALL call sites)
- Modify: `tests/engine/test_executors.py` (update existing test)
- Test: `tests/engine/test_executors.py` (add new test)

**CALL SITES TO UPDATE (from Task 0 verification):**

| Location | Line | Current Code | Update Required |
|----------|------|--------------|-----------------|
| `src/elspeth/engine/processor.py` | 176 | `buffered_rows = ...flush_buffer(node_id)` | Destructure tuple |
| `tests/engine/test_executors.py` | 2420 | `buffered = executor.flush_buffer(...)` | Destructure tuple |

**IMPORTANT:** Run full test suite after changes to catch any missed call sites.

**Step 1: Write the failing test**

```python
# tests/engine/test_executors.py

def test_aggregation_executor_flush_buffer_returns_tokens():
    """flush_buffer returns both rows and tokens."""
    from elspeth.engine.executors import AggregationExecutor
    from elspeth.contracts.identity import TokenInfo
    from elspeth.core.config import AggregationSettings, TriggerConfig
    from elspeth.core.landscape import LandscapeDB, LandscapeRecorder
    from elspeth.engine.spans import SpanFactory

    # Use real recorder for proper integration
    db = LandscapeDB.in_memory()
    recorder = LandscapeRecorder(db)
    span_factory = SpanFactory()  # No tracer = disabled tracing

    settings = {
        "node_1": AggregationSettings(
            name="test_agg",
            plugin="batch_stats",
            trigger=TriggerConfig(count=2),
        )
    }

    run = recorder.begin_run(config={}, canonical_version="1.0")

    executor = AggregationExecutor(
        recorder, span_factory, run.run_id, aggregation_settings=settings
    )

    # Buffer two tokens
    token1 = TokenInfo(row_id="r1", token_id="t1", row_data={"x": 1})
    token2 = TokenInfo(row_id="r2", token_id="t2", row_data={"x": 2})

    executor.buffer_row("node_1", token1)
    executor.buffer_row("node_1", token2)

    # Flush should return both rows AND tokens
    rows, tokens = executor.flush_buffer("node_1")

    assert len(rows) == 2
    assert len(tokens) == 2
    assert rows[0] == {"x": 1}
    assert rows[1] == {"x": 2}
    assert tokens[0].token_id == "t1"
    assert tokens[1].token_id == "t2"

    # Buffer should be cleared
    assert executor.get_buffered_rows("node_1") == []
    assert executor.get_buffered_tokens("node_1") == []
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/engine/test_executors.py::test_aggregation_executor_flush_buffer_returns_tokens -v`
Expected: FAIL (flush_buffer returns only rows, not tuple)

**Step 3: Update flush_buffer() signature and implementation**

```python
# src/elspeth/engine/executors.py - modify AggregationExecutor.flush_buffer

def flush_buffer(self, node_id: str) -> tuple[list[dict[str, Any]], list[TokenInfo]]:
    """Get buffered rows and tokens, then clear the buffer.

    Args:
        node_id: Aggregation node ID

    Returns:
        Tuple of (rows, tokens) - both lists in buffer order
    """
    rows = list(self._buffers.get(node_id, []))
    tokens = list(self._buffer_tokens.get(node_id, []))

    # Clear buffers
    self._buffers[node_id] = []
    self._buffer_tokens[node_id] = []

    # Reset trigger evaluator for next batch
    evaluator = self._trigger_evaluators.get(node_id)
    if evaluator is not None:
        evaluator.reset()

    # Clear batch ID for next batch
    self._batch_ids[node_id] = None

    return rows, tokens
```

**Step 4: Update ALL processor call sites**

Search for `flush_buffer` in processor.py and update each call site:

```python
# src/elspeth/engine/processor.py - update _process_batch_aggregation_node

# BEFORE:
buffered_rows = self._aggregation_executor.flush_buffer(node_id)

# AFTER:
buffered_rows, buffered_tokens = self._aggregation_executor.flush_buffer(node_id)
```

**Step 5: Run test to verify it passes**

Run: `uv run pytest tests/engine/test_executors.py::test_aggregation_executor_flush_buffer_returns_tokens -v`
Expected: PASS

**Step 6: Update existing tests that use flush_buffer()**

Update `tests/engine/test_executors.py:2420`:

```python
# BEFORE (line ~2420):
buffered = executor.flush_buffer(agg_node.node_id)
assert len(buffered) == 2

# AFTER:
buffered_rows, buffered_tokens = executor.flush_buffer(agg_node.node_id)
assert len(buffered_rows) == 2
assert len(buffered_tokens) == 2
```

**Step 7: Run full test suite to catch any missed call sites**

```bash
uv run pytest tests/ -v --tb=short
```

If any tests fail with tuple unpacking errors, update those call sites.

**Step 8: Commit**

```bash
git add src/elspeth/engine/executors.py src/elspeth/engine/processor.py tests/engine/test_executors.py
git commit -m "feat(engine): flush_buffer returns both rows and tokens for passthrough"
```

---

### Task 7: Handle Multi-Row Output in Processor (Deaggregation)

**Files:**
- Modify: `src/elspeth/engine/processor.py` (transform processing section)
- Test: `tests/engine/test_processor.py`

**Context:** The deaggregation code goes in `_process_single_transform()` method (or equivalent) where `TransformResult` is handled after calling `transform.process()`.

**Step 1: Write the failing test**

```python
# tests/engine/test_processor.py

def test_processor_handles_expanding_transform():
    """Processor creates multiple RowResults for expanding transform."""
    from elspeth.plugins.base import BaseTransform
    from elspeth.contracts.results import TransformResult
    from elspeth.contracts.enums import RowOutcome, NodeType, Determinism
    from elspeth.contracts.identity import TokenInfo
    from elspeth.engine.processor import RowProcessor
    from elspeth.core.landscape import LandscapeDB, LandscapeRecorder
    from elspeth.engine.spans import SpanFactory
    from elspeth.plugins.context import PluginContext
    from unittest.mock import MagicMock

    class ExpanderTransform(BaseTransform):
        name = "expander"
        creates_tokens = True  # This is a deaggregation transform
        input_schema = None
        output_schema = None

        def process(self, row, ctx):
            # Expand each row into 2 rows
            return TransformResult.success_multi([
                {**row, "copy": 1},
                {**row, "copy": 2},
            ])

    # Setup real recorder
    from elspeth.contracts.schema import SchemaConfig
    db = LandscapeDB.in_memory()
    recorder = LandscapeRecorder(db)
    span_factory = SpanFactory()  # No tracer = disabled

    run = recorder.begin_run(config={}, canonical_version="1.0")
    source_node = recorder.register_node(
        run_id=run.run_id,
        plugin_name="csv",
        node_type=NodeType.SOURCE,
        plugin_version="1.0.0",
        config={},
        determinism=Determinism.IO_READ,
        schema_config=SchemaConfig(fields="dynamic"),
    )
    transform_node = recorder.register_node(
        run_id=run.run_id,
        plugin_name="expander",
        node_type=NodeType.TRANSFORM,
        plugin_version="1.0.0",
        config={},
        determinism=Determinism.DETERMINISTIC,
        schema_config=SchemaConfig(fields="dynamic"),
    )

    transform = ExpanderTransform({})
    transform.node_id = transform_node.node_id

    processor = RowProcessor(
        recorder=recorder,
        span_factory=span_factory,
        run_id=run.run_id,
        source_node_id=source_node.node_id,
    )

    ctx = MagicMock(spec=PluginContext)

    # Process a row through the expanding transform
    results = processor.process_row(
        row_index=0,
        row_data={"value": 42},
        transforms=[transform],
        ctx=ctx,
    )

    # Should get 3 results: 1 EXPANDED parent + 2 COMPLETED children
    assert len(results) == 3

    # Find the parent (EXPANDED) and children (COMPLETED)
    expanded = [r for r in results if r.outcome == RowOutcome.EXPANDED]
    completed = [r for r in results if r.outcome == RowOutcome.COMPLETED]

    assert len(expanded) == 1
    assert len(completed) == 2

    # Children should have different token_ids but same row_id
    assert completed[0].token_id != completed[1].token_id
    assert completed[0].row_id == completed[1].row_id

    # Children should have the expanded data
    child_copies = {r.final_data["copy"] for r in completed}
    assert child_copies == {1, 2}


def test_processor_rejects_multi_row_without_creates_tokens():
    """Processor raises error if transform returns multi-row but creates_tokens=False."""
    from elspeth.plugins.base import BaseTransform
    from elspeth.contracts.results import TransformResult
    from elspeth.engine.processor import RowProcessor
    from elspeth.core.landscape import LandscapeDB, LandscapeRecorder
    from elspeth.engine.spans import SpanFactory
    from elspeth.contracts.enums import NodeType, Determinism
    from unittest.mock import MagicMock
    import pytest

    class BadTransform(BaseTransform):
        name = "bad"
        creates_tokens = False  # NOT allowed to create new tokens
        input_schema = None
        output_schema = None

        def process(self, row, ctx):
            return TransformResult.success_multi([row, row])  # But returns multi!

    from elspeth.contracts.schema import SchemaConfig
    db = LandscapeDB.in_memory()
    recorder = LandscapeRecorder(db)
    span_factory = SpanFactory()  # No tracer = disabled

    run = recorder.begin_run(config={}, canonical_version="1.0")
    source_node = recorder.register_node(
        run_id=run.run_id,
        plugin_name="csv",
        node_type=NodeType.SOURCE,
        plugin_version="1.0.0",
        config={},
        determinism=Determinism.IO_READ,
        schema_config=SchemaConfig(fields="dynamic"),
    )
    transform_node = recorder.register_node(
        run_id=run.run_id,
        plugin_name="bad",
        node_type=NodeType.TRANSFORM,
        plugin_version="1.0.0",
        config={},
        determinism=Determinism.DETERMINISTIC,
        schema_config=SchemaConfig(fields="dynamic"),
    )

    transform = BadTransform({})
    transform.node_id = transform_node.node_id

    processor = RowProcessor(
        recorder=recorder,
        span_factory=span_factory,
        run_id=run.run_id,
        source_node_id=source_node.node_id,
    )

    ctx = MagicMock()

    # Should raise because creates_tokens=False but returns multi-row
    # (outside of aggregation passthrough context)
    with pytest.raises(RuntimeError, match="creates_tokens=False"):
        processor.process_row(
            row_index=0,
            row_data={"value": 1},
            transforms=[transform],
            ctx=ctx,
        )
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/engine/test_processor.py::test_processor_handles_expanding_transform -v`
Expected: FAIL (processor doesn't handle multi-row results yet)

**Step 3: Write minimal implementation**

Location: In `processor.py`, find the method that handles transform results (likely `_process_single_transform` or similar). Add after the existing success handling.

**IMPORTANT:** This code is ONLY reached for non-aggregation transforms. Aggregation transforms (including passthrough mode) route through `_process_batch_aggregation_node()` where multi-row output is handled differently (see Task 8). The RuntimeError here catches the bug case where a non-aggregation transform incorrectly returns multi-row with `creates_tokens=False`.

```python
# src/elspeth/engine/processor.py - modify transform result handling
# NOTE: This is in _process_single_transform(), NOT the aggregation path

if result.status == "success":
    # Check for multi-row output (deaggregation)
    if result.is_multi_row:
        # Validate transform is allowed to create tokens
        # This check is ONLY for non-aggregation transforms.
        # Aggregation passthrough handles multi-row via Task 8's _process_batch_aggregation_node()
        if not transform.creates_tokens:
            raise RuntimeError(
                f"Transform '{transform.name}' returned multi-row result "
                f"but has creates_tokens=False. Either set creates_tokens=True "
                f"or return single row via TransformResult.success(row). "
                f"(Multi-row is allowed in aggregation passthrough mode.)"
            )

        # Deaggregation: create child tokens for each output row
        child_tokens = self._token_manager.expand_token(
            parent_token=current_token,
            expanded_rows=result.rows,
            step_in_pipeline=step,
        )

        # Queue each child for continued processing
        for child_token in child_tokens:
            child_items.append(_WorkItem(
                token=child_token,
                start_step=step + 1,
                coalesce_at_step=coalesce_at_step,
                coalesce_name=coalesce_name,
            ))

        # Parent token is EXPANDED (terminal for parent)
        return (
            RowResult(
                token=current_token,
                final_data=current_token.row_data,
                outcome=RowOutcome.EXPANDED,
            ),
            child_items,
        )

    # Single row output (existing logic continues below)
    # NOTE: creates_tokens=True transforms CAN return single rows (e.g., JSONExplode
    # with empty array). When is_multi_row=False, we use normal single-row handling
    # regardless of creates_tokens flag - no expansion occurs.
    # ... existing code for updating token with result.row ...
```

**Edge case: creates_tokens=True with single row output**

When a transform with `creates_tokens=True` returns a single row (e.g., JSONExplode with empty array):
- `result.is_multi_row` is `False`
- Code falls through to existing single-row handling
- No token expansion occurs - the row passes through normally
- This is **intentional** - single row = no expansion needed

Test this with `test_empty_array_returns_single_row()` in Task 10.

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/engine/test_processor.py::test_processor_handles_expanding_transform -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/elspeth/engine/processor.py tests/engine/test_processor.py
git commit -m "feat(engine): handle deaggregation in processor with creates_tokens validation"
```

---

### Task 8: Handle passthrough Output Mode in Aggregation

**Files:**
- Modify: `src/elspeth/engine/processor.py` (_process_batch_aggregation_node)
- Test: `tests/engine/test_processor.py`

**API CHANGE:** The return type of `_process_batch_aggregation_node` changes from `tuple[RowResult, list[_WorkItem]]` to `tuple[RowResult | list[RowResult], list[_WorkItem]]`. Callers must be updated to handle both cases:

```python
# Update caller to handle list return:
result, new_items = self._process_batch_aggregation_node(...)
if isinstance(result, list):
    results.extend(result)
else:
    results.append(result)
```

**Key insights:**
1. Passthrough mode does NOT create new tokens - it preserves original token_ids while enriching row data
2. Unlike single/transform modes, passthrough uses `BUFFERED` (non-terminal) while waiting, NOT `CONSUMED_IN_BATCH` (terminal)
3. Buffered tokens will reappear as `COMPLETED` on flush - this is expected and correct

**Semantic distinction:**
- `CONSUMED_IN_BATCH`: Token is absorbed into aggregate, will never reappear (single/transform modes)
- `BUFFERED`: Token is held for enrichment, will reappear as `COMPLETED` (passthrough mode)

**Step 1: Write the failing test**

```python
# tests/engine/test_processor.py

def test_aggregation_passthrough_mode():
    """Passthrough mode: BUFFERED while waiting, COMPLETED on flush with same tokens."""
    from elspeth.plugins.base import BaseTransform
    from elspeth.contracts.results import TransformResult
    from elspeth.contracts.enums import RowOutcome, NodeType, Determinism
    from elspeth.engine.processor import RowProcessor
    from elspeth.core.landscape import LandscapeDB, LandscapeRecorder
    from elspeth.engine.spans import SpanFactory
    from elspeth.core.config import AggregationSettings, TriggerConfig
    from unittest.mock import MagicMock

    class PassthroughEnricher(BaseTransform):
        name = "enricher"
        is_batch_aware = True
        creates_tokens = False  # Passthrough preserves original tokens
        input_schema = None
        output_schema = None

        def process(self, rows, ctx):
            # Enrich each row with batch metadata
            enriched = [
                {**row, "batch_size": len(rows), "enriched": True}
                for row in rows
            ]
            return TransformResult.success_multi(enriched)

    from elspeth.contracts.schema import SchemaConfig
    db = LandscapeDB.in_memory()
    recorder = LandscapeRecorder(db)
    span_factory = SpanFactory()  # No tracer = disabled

    run = recorder.begin_run(config={}, canonical_version="1.0")
    source_node = recorder.register_node(
        run_id=run.run_id,
        plugin_name="csv",
        node_type=NodeType.SOURCE,
        plugin_version="1.0.0",
        config={},
        determinism=Determinism.IO_READ,
        schema_config=SchemaConfig(fields="dynamic"),
    )
    transform_node = recorder.register_node(
        run_id=run.run_id,
        plugin_name="enricher",
        node_type=NodeType.TRANSFORM,
        plugin_version="1.0.0",
        config={},
        determinism=Determinism.DETERMINISTIC,
        schema_config=SchemaConfig(fields="dynamic"),
    )

    transform = PassthroughEnricher({})
    transform.node_id = transform_node.node_id

    aggregation_settings = {
        transform_node.node_id: AggregationSettings(
            name="batch_enrich",
            plugin="enricher",
            trigger=TriggerConfig(count=3),
            output_mode="passthrough",
        )
    }

    processor = RowProcessor(
        recorder=recorder,
        span_factory=span_factory,
        run_id=run.run_id,
        source_node_id=source_node.node_id,
        aggregation_settings=aggregation_settings,
    )

    ctx = MagicMock()

    # Process 3 rows - batch should flush after 3rd row
    all_results = []
    for i in range(3):
        results = processor.process_row(
            row_index=i,
            row_data={"value": i},
            transforms=[transform],
            ctx=ctx,
        )
        all_results.extend(results)

    # First 2 rows: BUFFERED (non-terminal, waiting for flush)
    # 3rd row triggers flush: 3 COMPLETED (all enriched, including the 2 that were buffered)
    buffered = [r for r in all_results if r.outcome == RowOutcome.BUFFERED]
    completed = [r for r in all_results if r.outcome == RowOutcome.COMPLETED]

    assert len(buffered) == 2, f"Expected 2 buffered, got {len(buffered)}"
    assert len(completed) == 3, f"Expected 3 completed, got {len(completed)}"

    # Completed rows should have enrichment
    for r in completed:
        assert r.final_data["enriched"] is True
        assert r.final_data["batch_size"] == 3

    # CRITICAL: Buffered tokens MUST reappear in completed (same token_ids)
    # This is the key semantic of passthrough - tokens are held, not consumed
    buffered_token_ids = {r.token.token_id for r in buffered}
    completed_token_ids = {r.token.token_id for r in completed}
    assert buffered_token_ids.issubset(completed_token_ids), \
        "Buffered tokens must reappear as completed in passthrough mode"


def test_aggregation_passthrough_validates_row_count():
    """Passthrough mode raises error if transform returns wrong row count."""
    from elspeth.plugins.base import BaseTransform
    from elspeth.contracts.results import TransformResult
    from elspeth.engine.processor import RowProcessor
    from elspeth.core.landscape import LandscapeDB, LandscapeRecorder
    from elspeth.engine.spans import SpanFactory
    from elspeth.core.config import AggregationSettings, TriggerConfig
    from elspeth.contracts.enums import NodeType, Determinism
    from unittest.mock import MagicMock
    import pytest

    class BadPassthrough(BaseTransform):
        name = "bad_passthrough"
        is_batch_aware = True
        creates_tokens = False
        input_schema = None
        output_schema = None

        def process(self, rows, ctx):
            # Returns wrong number of rows!
            return TransformResult.success_multi([{"wrong": True}])

    from elspeth.contracts.schema import SchemaConfig
    db = LandscapeDB.in_memory()
    recorder = LandscapeRecorder(db)
    span_factory = SpanFactory()  # No tracer = disabled

    run = recorder.begin_run(config={}, canonical_version="1.0")
    source_node = recorder.register_node(
        run_id=run.run_id,
        plugin_name="csv",
        node_type=NodeType.SOURCE,
        plugin_version="1.0.0",
        config={},
        determinism=Determinism.IO_READ,
        schema_config=SchemaConfig(fields="dynamic"),
    )
    transform_node = recorder.register_node(
        run_id=run.run_id,
        plugin_name="bad_passthrough",
        node_type=NodeType.TRANSFORM,
        plugin_version="1.0.0",
        config={},
        determinism=Determinism.DETERMINISTIC,
        schema_config=SchemaConfig(fields="dynamic"),
    )

    transform = BadPassthrough({})
    transform.node_id = transform_node.node_id

    aggregation_settings = {
        transform_node.node_id: AggregationSettings(
            name="bad",
            plugin="bad_passthrough",
            trigger=TriggerConfig(count=2),
            output_mode="passthrough",
        )
    }

    processor = RowProcessor(
        recorder=recorder,
        span_factory=span_factory,
        run_id=run.run_id,
        source_node_id=source_node.node_id,
        aggregation_settings=aggregation_settings,
    )

    ctx = MagicMock()

    # Process 2 rows to trigger flush
    processor.process_row(row_index=0, row_data={"x": 1}, transforms=[transform], ctx=ctx)

    with pytest.raises(ValueError, match="same number of output rows"):
        processor.process_row(row_index=1, row_data={"x": 2}, transforms=[transform], ctx=ctx)
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/engine/test_processor.py::test_aggregation_passthrough_mode -v`
Expected: FAIL (only single mode handled)

**Step 3: Write minimal implementation**

```python
# src/elspeth/engine/processor.py - modify _process_batch_aggregation_node

def _process_batch_aggregation_node(
    self,
    transform: BaseTransform,
    current_token: TokenInfo,
    ctx: PluginContext,
    step: int,
    total_steps: int,  # NEW: for checking if more transforms exist
    child_items: list[_WorkItem],
    coalesce_at_step: int | None = None,
    coalesce_name: str | None = None,
) -> tuple[RowResult | list[RowResult], list[_WorkItem]]:
    """Process a row at an aggregation node using engine buffering.

    Returns either a single RowResult or a list of RowResults.

    Args:
        total_steps: Total number of transforms in pipeline (for passthrough continuation)

    Outcome semantics by output_mode:
    - single: Buffered rows get CONSUMED_IN_BATCH (terminal), aggregate gets COMPLETED
    - passthrough: Buffered rows get BUFFERED (non-terminal), same tokens get COMPLETED on flush
    - transform: Buffered rows get CONSUMED_IN_BATCH (terminal), new tokens get COMPLETED
    """
    node_id = transform.node_id
    assert node_id is not None

    # Get output_mode early - we need it to decide buffer outcome
    agg_settings = self._aggregation_settings.get(node_id)
    output_mode = agg_settings.output_mode if agg_settings else "single"

    # Buffer the row
    self._aggregation_executor.buffer_row(node_id, current_token)

    # Check if we should flush
    if self._aggregation_executor.should_flush(node_id):
        # Get buffered rows AND tokens before flushing clears them
        buffered_rows, buffered_tokens = self._aggregation_executor.flush_buffer(node_id)

        # Call transform with batch
        result = transform.process(buffered_rows, ctx)  # type: ignore[arg-type]

        if result.status == "success":
            if output_mode == "single":
                # Existing: N rows -> 1 row
                final_data = result.row if result.row is not None else {}
                updated_token = TokenInfo(
                    row_id=current_token.row_id,
                    token_id=current_token.token_id,
                    row_data=final_data,
                    branch_name=current_token.branch_name,
                )
                return (
                    RowResult(
                        token=updated_token,
                        final_data=final_data,
                        outcome=RowOutcome.COMPLETED,
                    ),
                    child_items,
                )

            elif output_mode == "passthrough":
                # N rows -> N enriched rows, preserving original tokens
                if not result.is_multi_row:
                    raise ValueError(
                        f"passthrough mode requires success_multi result, got single row"
                    )
                if len(result.rows) != len(buffered_tokens):
                    raise ValueError(
                        f"passthrough mode requires same number of output rows "
                        f"({len(result.rows)}) as input rows ({len(buffered_tokens)})"
                    )

                # Build COMPLETED results for ALL buffered tokens with their enriched data
                all_results: list[RowResult] = []
                for token, row_data in zip(buffered_tokens, result.rows, strict=True):
                    updated_token = TokenInfo(
                        row_id=token.row_id,
                        token_id=token.token_id,  # PRESERVE original token
                        row_data=row_data,
                        branch_name=token.branch_name,
                    )

                    # If there are more transforms after this, queue as work items
                    # Otherwise mark as COMPLETED
                    if step + 1 < total_steps:
                        child_items.append(_WorkItem(
                            token=updated_token,
                            start_step=step + 1,
                            coalesce_at_step=coalesce_at_step,
                            coalesce_name=coalesce_name,
                        ))
                    else:
                        all_results.append(RowResult(
                            token=updated_token,
                            final_data=row_data,
                            outcome=RowOutcome.COMPLETED,
                        ))

                # Return completed results (may be empty if all went to child_items)
                return all_results, child_items

            elif output_mode == "transform":
                # N rows -> M rows (handled in Task 9)
                raise NotImplementedError("transform output_mode handled in Task 9")

            else:
                raise ValueError(f"Unknown output_mode: {output_mode}")

        else:
            # Error handling - all buffered rows fail (batch is atomic)
            error_results = []
            for token in buffered_tokens:
                error_results.append(RowResult(
                    token=token,
                    final_data=token.row_data,
                    outcome=RowOutcome.FAILED,
                    error=FailureInfo(
                        exception_type="BatchTransformError",
                        message=str(result.reason) if result.reason else "Batch transform failed",
                    ),
                ))
            return error_results, child_items

    # Not flushing yet - outcome depends on output_mode
    if output_mode == "passthrough":
        # BUFFERED: non-terminal, token will reappear as COMPLETED on flush
        return (
            RowResult(
                token=current_token,
                final_data=current_token.row_data,
                outcome=RowOutcome.BUFFERED,
            ),
            child_items,
        )
    else:
        # CONSUMED_IN_BATCH: terminal, token absorbed into aggregate (single/transform modes)
        return (
            RowResult(
                token=current_token,
                final_data=current_token.row_data,
                outcome=RowOutcome.CONSUMED_IN_BATCH,
            ),
            child_items,
        )
```

**Step 4: Pass total_steps as parameter**

The implementation uses `step + 1 < total_steps` to check for remaining transforms. Since `process_row()` receives transforms as a parameter (confirmed in Task 0: `transforms: list[Any]` at processor.py:291), we pass `total_steps` rather than storing transforms on self.

**Update method signature:**

```python
# src/elspeth/engine/processor.py - update _process_batch_aggregation_node signature
def _process_batch_aggregation_node(
    self,
    transform: BaseTransform,
    current_token: TokenInfo,
    ctx: PluginContext,
    step: int,
    total_steps: int,  # NEW parameter
    child_items: list[_WorkItem],
    coalesce_at_step: int | None = None,
    coalesce_name: str | None = None,
) -> tuple[RowResult | list[RowResult], list[_WorkItem]]:
```

**Update call site in process_row()** (processor.py:451):

```python
# BEFORE:
return self._process_batch_aggregation_node(
    transform=transform,
    current_token=current_token,
    ctx=ctx,
    step=step,
    child_items=child_items,
)

# AFTER:
return self._process_batch_aggregation_node(
    transform=transform,
    current_token=current_token,
    ctx=ctx,
    step=step,
    total_steps=len(transforms),  # NEW: pass total steps
    child_items=child_items,
    coalesce_at_step=coalesce_at_step,
    coalesce_name=coalesce_name,
)
```

**In the implementation (Step 3 code), replace `_has_more_transforms_after(step)` with:**

```python
if step + 1 < total_steps:  # Has more transforms after this
    # queue as work items
```

**Step 5: Run test to verify it passes**

Run: `uv run pytest tests/engine/test_processor.py::test_aggregation_passthrough_mode -v`
Expected: PASS

**Step 6: Commit**

```bash
git add src/elspeth/engine/processor.py tests/engine/test_processor.py
git commit -m "feat(engine): implement passthrough output mode with BUFFERED semantics"
```

---

### Task 9: Handle transform Output Mode in Aggregation

**Files:**
- Modify: `src/elspeth/engine/processor.py`
- Test: `tests/engine/test_processor.py`

**Key insight:** Transform mode CREATES new tokens (like deaggregation). The triggering row becomes `CONSUMED_IN_BATCH` like the others, and the batch as a whole produces new tokens for output.

**Step 1: Write the failing test**

```python
# tests/engine/test_processor.py

def test_aggregation_transform_mode():
    """Transform mode returns M rows from N input rows with new tokens."""
    from elspeth.plugins.base import BaseTransform
    from elspeth.contracts.results import TransformResult
    from elspeth.contracts.enums import RowOutcome, NodeType, Determinism
    from elspeth.engine.processor import RowProcessor
    from elspeth.core.landscape import LandscapeDB, LandscapeRecorder
    from elspeth.engine.spans import SpanFactory
    from elspeth.core.config import AggregationSettings, TriggerConfig
    from unittest.mock import MagicMock

    class GroupSplitter(BaseTransform):
        """Splits batch into groups, outputs one row per group."""
        name = "splitter"
        is_batch_aware = True
        creates_tokens = True  # Transform mode creates new tokens
        input_schema = None
        output_schema = None

        def process(self, rows, ctx):
            # Group by 'category' and output one row per group
            groups = {}
            for row in rows:
                cat = row.get("category", "default")
                if cat not in groups:
                    groups[cat] = {"category": cat, "count": 0, "total": 0}
                groups[cat]["count"] += 1
                groups[cat]["total"] += row.get("value", 0)
            return TransformResult.success_multi(list(groups.values()))

    from elspeth.contracts.schema import SchemaConfig
    db = LandscapeDB.in_memory()
    recorder = LandscapeRecorder(db)
    span_factory = SpanFactory()  # No tracer = disabled

    run = recorder.begin_run(config={}, canonical_version="1.0")
    source_node = recorder.register_node(
        run_id=run.run_id,
        plugin_name="csv",
        node_type=NodeType.SOURCE,
        plugin_version="1.0.0",
        config={},
        determinism=Determinism.IO_READ,
        schema_config=SchemaConfig(fields="dynamic"),
    )
    transform_node = recorder.register_node(
        run_id=run.run_id,
        plugin_name="splitter",
        node_type=NodeType.TRANSFORM,
        plugin_version="1.0.0",
        config={},
        determinism=Determinism.DETERMINISTIC,
        schema_config=SchemaConfig(fields="dynamic"),
    )

    transform = GroupSplitter({})
    transform.node_id = transform_node.node_id

    aggregation_settings = {
        transform_node.node_id: AggregationSettings(
            name="group_split",
            plugin="splitter",
            trigger=TriggerConfig(count=5),
            output_mode="transform",
        )
    }

    processor = RowProcessor(
        recorder=recorder,
        span_factory=span_factory,
        run_id=run.run_id,
        source_node_id=source_node.node_id,
        aggregation_settings=aggregation_settings,
    )

    ctx = MagicMock()

    # Process 5 rows with 2 categories
    test_rows = [
        {"category": "A", "value": 10},
        {"category": "B", "value": 20},
        {"category": "A", "value": 30},
        {"category": "B", "value": 40},
        {"category": "A", "value": 50},
    ]

    all_results = []
    for i, row_data in enumerate(test_rows):
        results = processor.process_row(
            row_index=i,
            row_data=row_data,
            transforms=[transform],
            ctx=ctx,
        )
        all_results.extend(results)

    # All 5 input rows get CONSUMED_IN_BATCH
    # The batch produces 2 COMPLETED outputs (one per category)
    consumed = [r for r in all_results if r.outcome == RowOutcome.CONSUMED_IN_BATCH]
    completed = [r for r in all_results if r.outcome == RowOutcome.COMPLETED]

    assert len(consumed) == 5, f"Expected 5 consumed, got {len(consumed)}"
    assert len(completed) == 2, f"Expected 2 completed, got {len(completed)}"

    # Verify group data
    categories = {r.final_data["category"] for r in completed}
    assert categories == {"A", "B"}

    # Verify new token_ids created (not reusing input tokens)
    completed_tokens = {r.token.token_id for r in completed}
    consumed_tokens = {r.token.token_id for r in consumed}
    assert completed_tokens.isdisjoint(consumed_tokens), "Transform mode should create NEW tokens"
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/engine/test_processor.py::test_aggregation_transform_mode -v`
Expected: FAIL (NotImplementedError from Task 8)

**Step 3: Write minimal implementation**

```python
# src/elspeth/engine/processor.py - replace NotImplementedError in transform mode

elif output_mode == "transform":
    # N rows -> M rows with new tokens
    # All input tokens become CONSUMED_IN_BATCH
    # Batch produces new tokens for outputs

    # First, mark the triggering token as consumed (like the others)
    consumed_result = RowResult(
        token=current_token,
        final_data=current_token.row_data,
        outcome=RowOutcome.CONSUMED_IN_BATCH,
    )

    if result.is_multi_row:
        # Multi-row: expand into new tokens
        # Use the triggering token as the "parent" for audit trail
        expanded_tokens = self._token_manager.expand_token(
            parent_token=current_token,
            expanded_rows=result.rows,
            step_in_pipeline=step,
        )

        # Queue expanded tokens for continued processing (or completion)
        output_results = []
        for token in expanded_tokens:
            if step + 1 < total_steps:
                child_items.append(_WorkItem(
                    token=token,
                    start_step=step + 1,
                    coalesce_at_step=coalesce_at_step,
                    coalesce_name=coalesce_name,
                ))
            else:
                output_results.append(RowResult(
                    token=token,
                    final_data=token.row_data,
                    outcome=RowOutcome.COMPLETED,
                ))

        # Return consumed marker + output results
        return [consumed_result] + output_results, child_items

    else:
        # Single row output is valid for transform mode
        # Create one new token for the output
        expanded_tokens = self._token_manager.expand_token(
            parent_token=current_token,
            expanded_rows=[result.row],
            step_in_pipeline=step,
        )
        output_token = expanded_tokens[0]

        if step + 1 < total_steps:
            child_items.append(_WorkItem(
                token=output_token,
                start_step=step + 1,
            ))
            return [consumed_result], child_items
        else:
            return [
                consumed_result,
                RowResult(
                    token=output_token,
                    final_data=output_token.row_data,
                    outcome=RowOutcome.COMPLETED,
                ),
            ], child_items
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/engine/test_processor.py::test_aggregation_transform_mode -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/elspeth/engine/processor.py tests/engine/test_processor.py
git commit -m "feat(engine): implement transform output mode for aggregation"
```

---

### Task 10: Create Example Deaggregation Transform (JSONExplode)

**Files:**
- Create: `src/elspeth/plugins/transforms/json_explode.py`
- Modify: `src/elspeth/plugins/transforms/hookimpl.py`
- Test: `tests/plugins/transforms/test_json_explode.py`

**THREE-TIER TRUST MODEL COMPLIANCE:**

Per the plugin protocol, transforms TRUST that pipeline data types are correct:
- Source validates that required fields exist and have correct types
- Transforms access fields directly without defensive checks
- Type violations (missing field, wrong type) indicate UPSTREAM BUGS and should CRASH

JSONExplode does NOT return `TransformResult.error()` for type violations because:
1. Missing field = source should have validated → crash surfaces config bug
2. Wrong type = source should have validated → crash surfaces config bug
3. There are no VALUE-level operations that can fail in this transform

Therefore, JSONExplode inherits from `DataPluginConfig` (NOT `TransformDataConfig`)
and has no `on_error` configuration.

**Step 1: Write the failing tests**

```python
# tests/plugins/transforms/test_json_explode.py
"""Tests for JSONExplode transform.

THREE-TIER TRUST MODEL:
Per the plugin protocol, JSONExplode TRUSTS that:
- The array field exists (source validated it)
- The field is a list (source validated it)

Type violations (missing field, wrong type) indicate upstream bugs and
should CRASH, not return TransformResult.error(). This surfaces
configuration problems immediately rather than hiding them.
"""

import pytest
from unittest.mock import MagicMock

from elspeth.plugins.transforms.json_explode import JSONExplode


class TestJSONExplodeHappyPath:
    """Tests for valid input conforming to expected schema."""

    def test_explodes_array_into_multiple_rows(self):
        """JSONExplode expands array field into multiple rows."""
        transform = JSONExplode({
            "array_field": "items",
            "schema": {"fields": "dynamic"},
        })
        ctx = MagicMock()

        row = {
            "id": 1,
            "items": [{"name": "a"}, {"name": "b"}, {"name": "c"}],
        }

        result = transform.process(row, ctx)

        assert result.status == "success"
        assert result.is_multi_row
        assert len(result.rows) == 3
        assert result.rows[0] == {"id": 1, "item": {"name": "a"}, "item_index": 0}
        assert result.rows[1] == {"id": 1, "item": {"name": "b"}, "item_index": 1}
        assert result.rows[2] == {"id": 1, "item": {"name": "c"}, "item_index": 2}

    def test_creates_tokens_is_true(self):
        """JSONExplode has creates_tokens=True (deaggregation)."""
        transform = JSONExplode({
            "array_field": "items",
            "schema": {"fields": "dynamic"},
        })
        assert transform.creates_tokens is True

    def test_empty_array_returns_single_row(self):
        """Empty array returns single row with None item."""
        transform = JSONExplode({
            "array_field": "items",
            "schema": {"fields": "dynamic"},
        })
        ctx = MagicMock()

        row = {"id": 1, "items": []}

        result = transform.process(row, ctx)

        # Empty array: return single row (not multi) with None item
        assert result.status == "success"
        assert result.is_multi_row is False
        assert result.row == {"id": 1, "item": None, "item_index": None}

    def test_custom_output_field_name(self):
        """Custom output_field name is respected."""
        transform = JSONExplode({
            "array_field": "items",
            "output_field": "element",
            "schema": {"fields": "dynamic"},
        })
        ctx = MagicMock()

        row = {"id": 1, "items": [{"x": 1}]}

        result = transform.process(row, ctx)

        assert "element" in result.rows[0]
        assert "item" not in result.rows[0]

    def test_include_index_false(self):
        """Can disable item_index field."""
        transform = JSONExplode({
            "array_field": "items",
            "include_index": False,
            "schema": {"fields": "dynamic"},
        })
        ctx = MagicMock()

        row = {"id": 1, "items": [{"x": 1}]}

        result = transform.process(row, ctx)

        assert "item_index" not in result.rows[0]

    def test_preserves_all_non_array_fields(self):
        """All fields except array field are preserved in output."""
        transform = JSONExplode({
            "array_field": "items",
            "schema": {"fields": "dynamic"},
        })
        ctx = MagicMock()

        row = {
            "id": 1,
            "name": "test",
            "metadata": {"key": "value"},
            "items": [{"x": 1}],
        }

        result = transform.process(row, ctx)

        assert result.rows[0]["id"] == 1
        assert result.rows[0]["name"] == "test"
        assert result.rows[0]["metadata"] == {"key": "value"}
        assert "items" not in result.rows[0]


class TestJSONExplodeTypeViolations:
    """Tests for type violations - these should CRASH, not return errors.

    Per the three-tier trust model:
    - Source validates that `items` field exists and is a list
    - Transform trusts this validation
    - If validation didn't happen, that's an UPSTREAM BUG
    - Crashing surfaces the bug immediately
    """

    def test_missing_field_crashes(self):
        """Missing array field is upstream bug - should crash."""
        transform = JSONExplode({
            "array_field": "items",
            "schema": {"fields": "dynamic"},
        })
        ctx = MagicMock()

        row = {"id": 1}  # No 'items' field - upstream should have validated!

        # Crash indicates upstream configuration bug
        # Source should have schema: {fields: ["items: list"]}
        with pytest.raises(KeyError, match="items"):
            transform.process(row, ctx)

    def test_none_value_crashes(self):
        """None value for array field is upstream bug - should crash."""
        transform = JSONExplode({
            "array_field": "items",
            "schema": {"fields": "dynamic"},
        })
        ctx = MagicMock()

        row = {"id": 1, "items": None}  # None should have been quarantined at source!

        # Crash indicates upstream configuration bug
        with pytest.raises(TypeError):
            transform.process(row, ctx)

    def test_non_array_value_crashes(self):
        """Non-array value is upstream bug - should crash."""
        transform = JSONExplode({
            "array_field": "items",
            "schema": {"fields": "dynamic"},
        })
        ctx = MagicMock()

        row = {"id": 1, "items": "not an array"}  # Wrong type should have been quarantined!

        # Crash indicates upstream configuration bug
        with pytest.raises(TypeError):
            transform.process(row, ctx)


class TestJSONExplodeConfiguration:
    """Tests for configuration handling."""

    def test_no_on_error_attribute(self):
        """JSONExplode has no on_error - no legitimate VALUE errors exist."""
        transform = JSONExplode({
            "array_field": "items",
            "schema": {"fields": "dynamic"},
        })

        # _on_error should be None (inherited default from BaseTransform)
        # This transform has no legitimate VALUE-level errors
        assert transform._on_error is None

    def test_array_field_is_required(self):
        """array_field config is required."""
        from elspeth.plugins.config_base import PluginConfigError

        with pytest.raises(PluginConfigError):
            JSONExplode({
                # Missing array_field
                "schema": {"fields": "dynamic"},
            })

    def test_schema_is_required(self):
        """schema config is required (via DataPluginConfig)."""
        from elspeth.plugins.config_base import PluginConfigError

        with pytest.raises(PluginConfigError):
            JSONExplode({
                "array_field": "items",
                # Missing schema
            })
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/plugins/transforms/test_json_explode.py -v`
Expected: FAIL with ModuleNotFoundError

**Step 3: Write minimal implementation**

```python
# src/elspeth/plugins/transforms/json_explode.py
"""JSON array explode transform - deaggregation example.

Expands a JSON array field into multiple rows, one per array element.
Useful for flattening nested data structures.

THREE-TIER TRUST MODEL COMPLIANCE:
This transform TRUSTS that pipeline data types are correct:
- The array_field exists (source validated it)
- The array_field is a list (source validated it)

If these assumptions are violated, the transform CRASHES. This is
intentional - it surfaces upstream configuration bugs immediately
rather than hiding them behind error routing.

Correct pipeline configuration:
    datasource:
      plugin: json
      options:
        schema:
          mode: strict
          fields:
            - "items: list"   # <-- Source validates this!
        on_validation_failure: quarantine_sink

Example:
    Input:  {"id": 1, "items": [{"x": 1}, {"x": 2}]}
    Output: [
        {"id": 1, "item": {"x": 1}, "item_index": 0},
        {"id": 1, "item": {"x": 2}, "item_index": 1},
    ]
"""

from typing import Any

from pydantic import Field

from elspeth.plugins.base import BaseTransform
from elspeth.plugins.config_base import DataPluginConfig  # NOT TransformDataConfig!
from elspeth.plugins.context import PluginContext
from elspeth.contracts.results import TransformResult
from elspeth.plugins.schema_factory import create_schema_from_config


class JSONExplodeConfig(DataPluginConfig):
    """Configuration for JSON explode transform.

    Inherits from DataPluginConfig (NOT TransformDataConfig) because:
    - This transform has no legitimate VALUE-level errors
    - TYPE violations (missing field, wrong type) indicate upstream bugs
    - Upstream bugs should CRASH, not be routed to error sinks

    Therefore, no `on_error` configuration is needed or accepted.
    """

    array_field: str = Field(description="Name of the array field to explode")
    output_field: str = Field(
        default="item",
        description="Name for the exploded element in output rows",
    )
    include_index: bool = Field(
        default=True,
        description="Whether to include item_index field",
    )


class JSONExplode(BaseTransform):
    """Explode a JSON array field into multiple rows.

    This is a deaggregation transform (creates_tokens=True) that takes
    one input row and produces N output rows, one per array element.

    Config options:
        array_field: Required. Name of the array field to explode
        output_field: Name for exploded element (default: "item")
        include_index: Whether to add item_index (default: True)
        schema: Schema config (required via DataPluginConfig)

    Edge cases:
        - Empty array: Returns single row with item=None (not multi-row)

    Type violations (these CRASH - they indicate upstream bugs):
        - Missing field: KeyError
        - None value: TypeError
        - Non-array value: TypeError

    IMPORTANT: Configure source schema to validate the array field!

        datasource:
          options:
            schema:
              mode: strict
              fields:
                - "items: list"
            on_validation_failure: quarantine_sink
    """

    name = "json_explode"
    creates_tokens = True  # CRITICAL: enables new token creation for each output

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(config)
        cfg = JSONExplodeConfig.from_dict(config)
        self._array_field = cfg.array_field
        self._output_field = cfg.output_field
        self._include_index = cfg.include_index

        # NO _on_error - this transform has no legitimate VALUE errors
        # Type violations crash (indicating upstream bugs)

        # Schema setup
        assert cfg.schema_config is not None
        schema = create_schema_from_config(
            cfg.schema_config, "JSONExplodeSchema", allow_coercion=False
        )
        self.input_schema = schema
        self.output_schema = schema

    def process(self, row: dict[str, Any], ctx: PluginContext) -> TransformResult:
        """Explode array field into multiple rows.

        TRUST: array_field exists and is a list (source validated this).
        If not, we crash - that's an upstream configuration bug.
        """
        # Direct access - TRUST that source validated the field exists
        # KeyError here means upstream didn't validate → crash surfaces bug
        array_value = row[self._array_field]

        # Direct iteration - TRUST that source validated it's a list
        # TypeError here means upstream didn't validate type → crash surfaces bug

        # Build base output (all fields except the array field)
        base = {k: v for k, v in row.items() if k != self._array_field}

        # Handle empty array - return single row, not multi
        if len(array_value) == 0:
            output = dict(base)
            output[self._output_field] = None
            if self._include_index:
                output["item_index"] = None
            return TransformResult.success(output)

        # Explode array into multiple rows
        output_rows = []
        for i, item in enumerate(array_value):
            output = dict(base)
            output[self._output_field] = item
            if self._include_index:
                output["item_index"] = i
            output_rows.append(output)

        return TransformResult.success_multi(output_rows)

    def close(self) -> None:
        """No resources to release."""
        pass
```

**Step 4: Register in hookimpl**

```python
# src/elspeth/plugins/transforms/hookimpl.py - add import and registration

# In ElspethBuiltinTransforms.elspeth_get_transforms():
from elspeth.plugins.transforms.batch_stats import BatchStats
from elspeth.plugins.transforms.field_mapper import FieldMapper
from elspeth.plugins.transforms.json_explode import JSONExplode
from elspeth.plugins.transforms.passthrough import PassThrough

return [PassThrough, FieldMapper, BatchStats, JSONExplode]
```

**Step 5: Run test to verify it passes**

Run: `uv run pytest tests/plugins/transforms/test_json_explode.py -v`
Expected: PASS

**Step 6: Commit**

```bash
git add src/elspeth/plugins/transforms/json_explode.py src/elspeth/plugins/transforms/hookimpl.py tests/plugins/transforms/test_json_explode.py
git commit -m "feat(plugins): add JSONExplode deaggregation transform

Implements deaggregation (1→N row expansion) using creates_tokens=True.

Three-tier trust model compliance:
- Trusts source validated array field exists and is list
- Crashes on type violations (surfaces upstream bugs)
- No on_error config (no legitimate VALUE-level errors)"
```

---

### Task 11: Create Integration Test with Example Pipeline

**Files:**
- Create: `examples/deaggregation/settings.yaml`
- Create: `examples/deaggregation/input.json`
- Test: `tests/integration/test_deaggregation.py`

**THREE-TIER TRUST MODEL COMPLIANCE:**

The example settings demonstrate proper configuration:
1. **Source uses strict schema** - validates `items: list` at ingestion boundary
2. **Source has `on_validation_failure`** - required field specifying where invalid rows go
3. **JSONExplode trusts the data** - no defensive checks needed because source validated

This is "defense in depth" - validate early at the trust boundary (source), and
downstream transforms can trust the data types are correct.

**Note:** Use JSON output (not CSV) to preserve nested dict structure in output.

**Step 1: Create example input**

```json
[
  {"order_id": 1, "items": [{"sku": "A1", "qty": 2}, {"sku": "B2", "qty": 1}]},
  {"order_id": 2, "items": [{"sku": "C3", "qty": 5}]},
  {"order_id": 3, "items": [{"sku": "A1", "qty": 1}, {"sku": "D4", "qty": 3}, {"sku": "E5", "qty": 2}]}
]
```

Save to: `examples/deaggregation/input.json`

**Step 2: Create example settings (with strict source schema)**

```yaml
# examples/deaggregation/settings.yaml
#
# Deaggregation example: Explode order items into individual rows
#
# THREE-TIER TRUST MODEL:
# - Source validates that 'items' is a list (strict schema)
# - Invalid rows quarantined BEFORE entering pipeline
# - JSONExplode trusts the field exists and is a list

datasource:
  plugin: json
  options:
    path: examples/deaggregation/input.json
    schema:
      mode: strict  # ENFORCE schema at source (the gatekeeper)
      fields:
        - "order_id: int"
        - "items: list"  # Source validates items is a list!
    on_validation_failure: discard  # REQUIRED: where invalid rows go
    # Alternative: on_validation_failure: invalid_orders (route to error sink)

row_plugins:
  - plugin: json_explode
    options:
      array_field: items
      output_field: item
      include_index: true
      schema:
        fields: dynamic
      # NO on_error - JSONExplode has no legitimate VALUE errors
      # Type violations crash (indicating upstream bug)

sinks:
  output:
    plugin: json
    options:
      path: examples/deaggregation/output/order_items.json
      schema:
        fields: dynamic
  # Uncomment if using on_validation_failure: invalid_orders
  # invalid_orders:
  #   plugin: json
  #   options:
  #     path: examples/deaggregation/output/invalid_orders.json
  #     schema:
  #       fields: dynamic

output_sink: output

landscape:
  url: sqlite:///examples/deaggregation/runs/audit.db
```

**Step 3: Write integration tests**

```python
# tests/integration/test_deaggregation.py
"""Integration tests for deaggregation pipeline.

Tests the JSONExplode transform with strict source schema validation,
demonstrating the three-tier trust model in action.
"""

import json
import sqlite3
import tempfile
import shutil
from pathlib import Path

import pytest

from elspeth.engine.orchestrator import Orchestrator
from elspeth.core.config import load_settings


@pytest.fixture
def temp_output_dir():
    """Create a temporary directory for test outputs."""
    temp_dir = tempfile.mkdtemp(prefix="elspeth_test_")
    yield Path(temp_dir)
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def deaggregation_settings(temp_output_dir):
    """Load deaggregation settings with temp output paths."""
    settings = load_settings("examples/deaggregation/settings.yaml")

    # Override output paths to use temp directory
    settings.sinks["output"].options["path"] = str(temp_output_dir / "order_items.json")
    settings.landscape.url = f"sqlite:///{temp_output_dir}/audit.db"

    return settings, temp_output_dir


class TestDeaggregationPipeline:
    """Tests for the complete deaggregation pipeline."""

    def test_explodes_orders_into_items(self, deaggregation_settings):
        """Full pipeline with deaggregation transform."""
        settings, temp_dir = deaggregation_settings

        # Run pipeline
        orchestrator = Orchestrator(settings)
        result = orchestrator.run()

        assert result.status == "completed", f"Pipeline failed: {result.error}"

        # Check output file exists
        output_path = temp_dir / "order_items.json"
        assert output_path.exists(), "Output file not created"

        # Load and verify output
        with open(output_path) as f:
            rows = json.load(f)

        # Verify row count (3 orders with 2+1+3 = 6 items)
        assert len(rows) == 6, f"Expected 6 rows, got {len(rows)}"

        # Verify structure
        for row in rows:
            assert "order_id" in row, "Missing order_id field"
            assert "item" in row, "Missing item field"
            assert "item_index" in row, "Missing item_index field"
            assert isinstance(row["item"], dict), "item should be a dict"
            assert "sku" in row["item"], "item should have sku"

        # Verify specific values
        order_1_rows = [r for r in rows if r["order_id"] == 1]
        assert len(order_1_rows) == 2, "Order 1 should have 2 items"

    def test_preserves_item_order(self, deaggregation_settings):
        """Items preserve their original array order via item_index."""
        settings, temp_dir = deaggregation_settings

        orchestrator = Orchestrator(settings)
        result = orchestrator.run()
        assert result.status == "completed"

        output_path = temp_dir / "order_items.json"
        with open(output_path) as f:
            rows = json.load(f)

        # Check order 3 which has 3 items
        order_3_rows = sorted(
            [r for r in rows if r["order_id"] == 3],
            key=lambda r: r["item_index"]
        )

        assert order_3_rows[0]["item"]["sku"] == "A1"
        assert order_3_rows[0]["item_index"] == 0
        assert order_3_rows[1]["item"]["sku"] == "D4"
        assert order_3_rows[1]["item_index"] == 1
        assert order_3_rows[2]["item"]["sku"] == "E5"
        assert order_3_rows[2]["item_index"] == 2


class TestDeaggregationAuditTrail:
    """Tests for audit trail correctness."""

    def test_records_token_expansion(self, deaggregation_settings):
        """Verify audit trail records token expansion correctly."""
        settings, temp_dir = deaggregation_settings

        orchestrator = Orchestrator(settings)
        result = orchestrator.run()
        assert result.status == "completed"

        # Check audit database
        db_path = temp_dir / "audit.db"
        assert db_path.exists(), "Audit database not created"

        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()

        # Verify tokens were created
        cursor.execute("SELECT COUNT(*) FROM tokens")
        token_count = cursor.fetchone()[0]
        # 3 source tokens + 6 expanded tokens = 9 total
        assert token_count >= 9, f"Expected at least 9 tokens, got {token_count}"

        conn.close()

    def test_records_parent_relationships(self, deaggregation_settings):
        """Expanded tokens have parent linkage for lineage."""
        settings, temp_dir = deaggregation_settings

        orchestrator = Orchestrator(settings)
        result = orchestrator.run()
        assert result.status == "completed"

        db_path = temp_dir / "audit.db"
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()

        # Verify token_parents recorded for expansion
        cursor.execute("SELECT COUNT(*) FROM token_parents")
        parent_count = cursor.fetchone()[0]
        # 6 expanded children should have parent relationships
        assert parent_count >= 6, f"Expected at least 6 parent relationships, got {parent_count}"

        conn.close()

    def test_expand_group_id_set(self, deaggregation_settings):
        """Expanded tokens have expand_group_id for grouping."""
        settings, temp_dir = deaggregation_settings

        orchestrator = Orchestrator(settings)
        result = orchestrator.run()
        assert result.status == "completed"

        db_path = temp_dir / "audit.db"
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()

        # Verify expand_group_id is set for expanded tokens
        cursor.execute("SELECT COUNT(*) FROM tokens WHERE expand_group_id IS NOT NULL")
        expanded_count = cursor.fetchone()[0]
        assert expanded_count == 6, f"Expected 6 expanded tokens, got {expanded_count}"

        conn.close()


class TestSourceSchemaValidation:
    """Tests demonstrating source-level schema validation.

    These tests show how the three-tier trust model works:
    - Source validates schema at ingestion (the trust boundary)
    - Invalid rows are quarantined BEFORE reaching transforms
    - JSONExplode trusts the data is valid
    """

    def test_invalid_row_quarantined_at_source(self, temp_output_dir):
        """Rows with missing items field are quarantined at source, not transform."""
        # Create input with one invalid row
        input_path = temp_output_dir / "mixed_input.json"
        with open(input_path, "w") as f:
            json.dump([
                {"order_id": 1, "items": [{"sku": "A1"}]},
                {"order_id": 2},  # Missing 'items' field
                {"order_id": 3, "items": [{"sku": "B2"}]},
            ], f)

        # Create settings with strict schema
        settings_yaml = f"""
datasource:
  plugin: json
  options:
    path: {input_path}
    schema:
      mode: strict
      fields:
        - "order_id: int"
        - "items: list"
    on_validation_failure: discard  # Quarantine at source

row_plugins:
  - plugin: json_explode
    options:
      array_field: items
      schema:
        fields: dynamic

sinks:
  output:
    plugin: json
    options:
      path: {temp_output_dir}/output.json
      schema:
        fields: dynamic

output_sink: output

landscape:
  url: sqlite:///{temp_output_dir}/audit.db
"""
        settings_path = temp_output_dir / "settings.yaml"
        with open(settings_path, "w") as f:
            f.write(settings_yaml)

        from elspeth.core.config import load_settings
        settings = load_settings(str(settings_path))

        orchestrator = Orchestrator(settings)
        result = orchestrator.run()

        # Pipeline succeeds (invalid row was quarantined at source)
        assert result.status == "completed"

        # Only 2 orders processed (order 2 was quarantined)
        with open(temp_output_dir / "output.json") as f:
            rows = json.load(f)

        # Order 1: 1 item, Order 3: 1 item = 2 rows total
        assert len(rows) == 2
        order_ids = {r["order_id"] for r in rows}
        assert order_ids == {1, 3}  # Order 2 was quarantined

        # Verify quarantine was recorded in audit via validation_errors table
        # (verified in Task 0: recorder.record_validation_error() writes to this table)
        conn = sqlite3.connect(str(temp_output_dir / "audit.db"))
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM validation_errors")
        error_count = cursor.fetchone()[0]
        assert error_count == 1, "Should have 1 validation error recorded"
        conn.close()
```

**Step 4: Run integration test**

Run: `uv run pytest tests/integration/test_deaggregation.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add examples/deaggregation/ tests/integration/test_deaggregation.py
git commit -m "feat(examples): add deaggregation example with strict source schema

Demonstrates three-tier trust model:
- Source enforces 'items: list' schema (gatekeeper)
- Invalid rows quarantined BEFORE entering pipeline
- JSONExplode trusts validated data

Integration tests verify:
- Token expansion creates parent linkage
- expand_group_id groups sibling tokens
- Source quarantines schema violations"
```

---

### Task 12: Run Full Test Suite

**Step 1: Run all tests**

Run: `uv run pytest tests/ -v --tb=short`
Expected: All tests pass

**Step 2: Run type checker**

Run: `uv run mypy src/elspeth/`
Expected: No errors

**Step 3: Run linter**

Run: `uv run ruff check src/elspeth/`
Expected: No errors

**Step 4: Final commit if any fixes needed**

```bash
git add -A
git commit -m "chore: fix any remaining issues from multi-row implementation"
```

---

## Summary

After completing all tasks:

| Feature | Task | Status |
|---------|------|--------|
| `TransformResult.success_multi()` | 1 | ✅ |
| `TransformResult.is_multi_row` | 1 | ✅ |
| `TransformResult.has_output_data` | 1 | ✅ |
| Fix TransformExecutor assertion | 1 | ✅ |
| `LandscapeRecorder.expand_token()` | 2 | ✅ |
| `Token.expand_group_id` | 2 | ✅ |
| Update `TokenRepository.load()` | 2 | ✅ |
| `BaseTransform.creates_tokens` | 3 | ✅ |
| `TokenManager.expand_token()` | 4 | ✅ |
| `RowOutcome.EXPANDED` | 5 | ✅ |
| `RowOutcome.BUFFERED` | 5 | ✅ |
| `RowOutcome.is_terminal` property | 5 | ✅ |
| `flush_buffer()` returns tokens | 6 | ✅ |
| Processor deaggregation handling | 7 | ✅ |
| `output_mode: passthrough` with `BUFFERED` semantics | 8 | ✅ |
| `output_mode: transform` | 9 | ✅ |
| `JSONExplode` example transform | 10 | ✅ |
| Integration test with audit verification | 11 | ✅ |

**Total estimated time:** 4-5 hours

**Key architectural decisions:**
1. Multi-row output uses separate `rows` field (not overloading `row`)
2. `creates_tokens` flag distinguishes deaggregation from passthrough
3. `LandscapeRecorder.expand_token()` handles audit trail with `expand_group_id`
4. Deaggregation creates child tokens with parent lineage via `token_parents`
5. Passthrough mode preserves original token IDs (no expansion)
6. Transform mode creates new tokens via expansion
7. `creates_tokens=False` transforms returning multi-row results raise RuntimeError (except in passthrough aggregation)
8. Empty array edge case returns single row (not multi-row with empty list)
9. Batch errors fail ALL buffered rows (atomic batch semantics)
10. `BUFFERED` is non-terminal - tokens will reappear with final outcome
11. `CONSUMED_IN_BATCH` is terminal - tokens are absorbed (single/transform modes only)
12. Passthrough mode uses `BUFFERED` → `COMPLETED` flow (same token appears twice, by design)

**Outcome semantics by aggregation mode:**

| Mode | While Buffering | On Flush | Token Identity |
|------|-----------------|----------|----------------|
| **single** | `CONSUMED_IN_BATCH` (terminal) | 1 row → `COMPLETED` | Triggering token reused |
| **passthrough** | `BUFFERED` (non-terminal) | N same tokens → `COMPLETED` | Tokens preserved |
| **transform** | `CONSUMED_IN_BATCH` (terminal) | M new tokens → `COMPLETED` | New tokens created via `expand_token()` |

**Audit trail guarantees:**
- Every **child** token from expansion has `expand_group_id` set (NOT the parent)
- Every child token has entry in `token_parents` linking to parent token
- `RowOutcome.EXPANDED` is **both returned and derivable**:
  - **Returned:** Processor returns `RowResult(outcome=RowOutcome.EXPANDED)` for the parent token (Task 7)
  - **Derivable:** Can be reconstructed from audit trail via query (for `explain()` and auditing):
  ```sql
  -- Query to find EXPANDED tokens:
  SELECT DISTINCT tp.parent_token_id
  FROM token_parents tp
  JOIN tokens t ON tp.token_id = t.token_id
  WHERE t.expand_group_id IS NOT NULL
  ```
- **Note:** `RowResult.outcome` is the processor's immediate result. The audit trail stores evidence (token_parents, expand_group_id) from which outcomes can be independently verified. This dual approach ensures:
  1. Fast processing (outcome in RowResult)
  2. Audit integrity (outcome can be re-derived from evidence)
- `RowOutcome.BUFFERED` tokens always reappear as `COMPLETED` (passthrough mode)
- Integration test verifies audit database correctness

**BUFFERED edge cases:**

1. **Crash recovery for BUFFERED tokens:** If a pipeline crashes while rows are in `BUFFERED` state (passthrough mode), those tokens remain with `BUFFERED` as their final recorded outcome. On pipeline restart, the batch state is `draft` or `executing` (not `completed`). The engine should:
   - Detect incomplete batches on restart (batch.status != 'completed')
   - Either: resume processing from checkpoint, or fail the batch with all `BUFFERED` tokens → `FAILED`
   - This is handled by existing crash recovery for batches (documented in plugin-protocol.md v1.4)

2. **`explain()` query handling for BUFFERED→COMPLETED:** In passthrough mode, a token's outcome changes from `BUFFERED` to `COMPLETED` over time. The `explain()` query should show:
   - The **final** outcome (`COMPLETED`) as the canonical result
   - The `BUFFERED` intermediate state in the timeline/trace for full audit visibility
   - Query implementation: derive outcome from `node_states` ordered by timestamp, taking the last state

**Out of scope (future work):**
- Fork/join interaction with expansion
- Partial batch error recovery
