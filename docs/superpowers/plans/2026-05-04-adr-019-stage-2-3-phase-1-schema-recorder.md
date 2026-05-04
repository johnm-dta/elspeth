# ADR-019 Stage 2/3 — Phase 1: Schema + Recorder + Loader + Contract Dataclasses

> **For Claude:** REQUIRED SUB-SKILL: Use `superpowers:executing-plans` to implement this phase task-by-task.
>
> **CRITICAL — atomic merge:** This phase is part of a five-phase plan ([overview](2026-05-04-adr-019-stage-2-3-overview.md)). Phases sequence as commits within ONE PR; Phase 1 alone leaves the engine module non-importable because every producer site still passes `RowOutcome` to `record_token_outcome`. Do NOT propose to land this commit alone. Phase 2 must follow in the same PR.

**Goal:** Flip the audit foundation — DB schema (`is_terminal` → `completed`, add `path` column, repurpose `outcome` value space), the four contract dataclasses (`TokenOutcome`, `RowResult`, `PendingOutcome`, `TokenCompleted`), the recorder write path (`record_token_outcome` signature + `_validate_outcome_fields` rewrite), and the loader read path (`TokenOutcomeLoader.load` cross-checks per ADR-019 § Implementation Notes invariant-translation table).

**Files touched in this phase:**

Core schema + recorder + dataclasses:
- Modify: `src/elspeth/core/landscape/schema.py:180-216` (table definition)
- Modify: `src/elspeth/contracts/audit.py:673-703` (TokenOutcome dataclass)
- Modify: `src/elspeth/contracts/results.py:379-421` (RowResult dataclass)
- Modify: `src/elspeth/contracts/engine.py:46-100` (PendingOutcome dataclass)
- Modify: `src/elspeth/contracts/events.py:242-249` (TokenCompleted dataclass)
- Modify: `src/elspeth/core/landscape/data_flow_repository.py:203-307, 570-580, 785-795, 802-880, 895-940` (recorder + internal FORKED/EXPANDED callers + read-side query column-rename sites at 899/930)
- Modify: `src/elspeth/core/landscape/model_loaders.py:525-609` (loader)
- Modify: `src/elspeth/testing/__init__.py:31-45, 507-540, 715-740` (test helpers re-export `TerminalOutcome`/`TerminalPath`)

**Downstream consumers of the audit schema (added 2026-05-05 after consumer-surface sweep — the original plan missed these and Phase 1 would crash MCP/Web at deploy without them):**
- Modify: `src/elspeth/mcp/types.py:364` (TypedDict `OutcomeDistributionEntry` — wire-schema field rename `is_terminal` → `completed`, add `path: str`)
- Modify: `src/elspeth/mcp/analyzers/reports.py:659, 663, 700, 708, 709` (5 sites in `get_outcome_analysis` — SQL column rename + dict-key rename + add path grouping)
- Modify: `src/elspeth/mcp/analyzers/diagnostics.py:181` (hardcoded `outcome == "quarantined"` → `path == "quarantined_at_source"` — silent-zero-quarantine bug per CLAUDE.md Tier 1 audit integrity)
- Modify: `src/elspeth/web/execution/diagnostics.py:170` (JOIN condition `is_terminal == 1` → `completed == 1`)
- Modify: `src/elspeth/web/execution/discard_summary.py:92` (WHERE filter `is_terminal == 1` → `completed == 1`)
- Modify: `src/elspeth/core/landscape/exporter.py:430` (JSONL token_outcome export field `is_terminal` → `completed`, add `path` field)
- Modify: `src/elspeth/core/landscape/lineage.py:118` (property read `o.is_terminal` → `o.completed` on `TokenOutcome` dataclass)
- Modify: `src/elspeth/core/landscape/formatters.py:170` (CLI formatter — print `path.name` alongside `outcome.name`; `is_terminal` line becomes `completed`)

Tests:
- Test: `tests/unit/core/landscape/test_data_flow_repository.py` (new tests for the (outcome, path) write path)
- Test: `tests/unit/core/landscape/test_token_outcome_loader.py` (new tests for the new cross-checks)
- Test: `tests/unit/contracts/test_audit.py` (TokenOutcome construction tests for the new shape)
- Test: `tests/unit/mcp/test_outcome_analysis.py` (new — verifies wire-schema rename + path column)
- Test: `tests/unit/mcp/test_diagnose_quarantine_count.py` (new — RED-first regression test for B3)
- Test: `tests/unit/web/execution/test_discard_summary.py` (existing — schema-dependent fixup)
- Test: `tests/unit/core/landscape/test_exporter.py` (existing — JSONL output assertion fixup)

**Background reading:** ADR-019 lines 99-115 (mapping table — the canonical contract), lines 237-269 (cross-check invariants), lines 638-660 (Implementation Notes table). The Stage 1 closed-set partition at `src/elspeth/contracts/enums.py::_LEGAL_TERMINAL_PAIRS` is THE source of truth for legal `(outcome, path)` pairs — every cross-check in this phase consults it.

---

## Schema decisions (read before editing)

### 1. `is_terminal` → `completed` is a rename, not a new column

Same SQLAlchemy `Column(Integer, nullable=False)` semantics. Same 0/1 stored values. Only the field name changes — and the cross-check that consults it. Per ADR-019 sub-decision 3 (panel-resolved), the `completed` field is materially redundant with `outcome IS NOT NULL` but preserved as a materialized column for query ergonomics and operator vocabulary, mirroring the existing `is_terminal` pattern.

### 2. `outcome` column changes value space

| Before | After |
| --- | --- |
| `String(32), nullable=False` | `String(32), nullable=True` |
| Stores `RowOutcome.value` (12 enum values, never NULL) | Stores `TerminalOutcome.value` (3 enum values: `success`, `failure`, `transient`) OR NULL when `completed=False` |
| Old cross-check: `is_terminal == RowOutcome(outcome).is_terminal` | New cross-check: `completed XOR (outcome IS NULL)` — i.e., `completed=True ↔ outcome IS NOT NULL` |

The same column is REUSED, not added. The schema migration is: rename one column, add one column (`path`), change one column's nullability and value space.

### 3. `path` is a new always-populated column

`Column("path", String(64), nullable=False)` — every row in `token_outcomes` has a path, including `BUFFERED` rows which carry `path="buffered"`. This makes the path column lookup-stable and avoids NULL handling at the loader.

### 4. DB migration is delete-and-recreate

Per `MEMORY.md::project_db_migration_policy`, ELSPETH does not run Alembic. The metadata defines the new schema; `metadata.create_all()` creates the new tables on engine startup. Operators delete `audit.db` and `sessions.db` between this PR and any pre-Stage-2 state.

---

## Tasks

### Task 1.1: Update `token_outcomes` schema definition

**Files:**
- Modify: `src/elspeth/core/landscape/schema.py:180-216`

**Step 1: Read the existing schema block to confirm line numbers**

Run: `grep -n "token_outcomes_table = Table" src/elspeth/core/landscape/schema.py`

Expected: line `180` (current HEAD).

**Step 2: Make the schema change**

Apply this edit:

```python
# OLD (lines 180-206 approximately):
token_outcomes_table = Table(
    "token_outcomes",
    metadata,
    Column("outcome_id", String(64), primary_key=True),
    Column("run_id", String(64), nullable=False, index=True),
    Column("token_id", String(64), nullable=False, index=True),
    ForeignKeyConstraint(["token_id", "run_id"], ["tokens.token_id", "tokens.run_id"]),
    Column("outcome", String(32), nullable=False),
    Column("is_terminal", Integer, nullable=False),
    Column("recorded_at", DateTime(timezone=True), nullable=False),
    Column("sink_name", String(128)),
    Column("batch_id", String(64)),
    Column("fork_group_id", String(64)),
    Column("join_group_id", String(64)),
    Column("expand_group_id", String(64)),
    Column("error_hash", String(64)),
    Column("context_json", Text),
    Column("expected_branches_json", Text),
    ForeignKeyConstraint(["batch_id", "run_id"], ["batches.batch_id", "batches.run_id"]),
)

# NEW:
token_outcomes_table = Table(
    "token_outcomes",
    metadata,
    # Identity
    Column("outcome_id", String(64), primary_key=True),
    Column("run_id", String(64), nullable=False, index=True),
    Column("token_id", String(64), nullable=False, index=True),
    # Composite FK: token_id and run_id belong together (prevents cross-run contamination)
    ForeignKeyConstraint(["token_id", "run_id"], ["tokens.token_id", "tokens.run_id"]),
    # ADR-019 two-axis terminal model. ``completed`` mirrors the prior
    # ``is_terminal`` column (sub-decision 3). ``outcome`` value space changed
    # from RowOutcome (12 values, non-NULL) to TerminalOutcome (3 values:
    # success / failure / transient) with NULL when completed=False
    # (only ``BUFFERED`` today). ``path`` is producer-declared per ADR-019
    # § "Classification is producer-declared, not topology-derivable" and
    # always populated, including ``path="buffered"`` for non-terminal rows.
    Column("outcome", String(32), nullable=True),
    Column("path", String(64), nullable=False),
    Column("completed", Integer, nullable=False),
    Column("recorded_at", DateTime(timezone=True), nullable=False),
    # Outcome-specific fields (nullable based on (outcome, path) pair)
    Column("sink_name", String(128)),
    Column("batch_id", String(64)),
    Column("fork_group_id", String(64)),
    Column("join_group_id", String(64)),
    Column("expand_group_id", String(64)),
    Column("error_hash", String(64)),
    # Optional extended context
    Column("context_json", Text),
    Column("expected_branches_json", Text),
    ForeignKeyConstraint(["batch_id", "run_id"], ["batches.batch_id", "batches.run_id"]),
)
```

**Step 3: Update the partial unique index (lines ~210-216)**

The existing index uses `is_terminal == 1`; rename to `completed == 1`:

```python
# OLD:
Index(
    "ix_token_outcomes_terminal_unique",
    token_outcomes_table.c.token_id,
    unique=True,
    sqlite_where=(token_outcomes_table.c.is_terminal == 1),
    postgresql_where=(token_outcomes_table.c.is_terminal == 1),
)

# NEW:
Index(
    "ix_token_outcomes_terminal_unique",
    token_outcomes_table.c.token_id,
    unique=True,
    sqlite_where=(token_outcomes_table.c.completed == 1),
    postgresql_where=(token_outcomes_table.c.completed == 1),
)
```

**Step 4: Verify schema compiles**

Run: `.venv/bin/python -c "from elspeth.core.landscape.schema import token_outcomes_table; print([c.name for c in token_outcomes_table.columns])"`

Expected output:
```
['outcome_id', 'run_id', 'token_id', 'outcome', 'path', 'completed', 'recorded_at', 'sink_name', 'batch_id', 'fork_group_id', 'join_group_id', 'expand_group_id', 'error_hash', 'context_json', 'expected_branches_json']
```

**Definition of Done:**
- [ ] `is_terminal` column renamed to `completed`
- [ ] `outcome` column nullability changed to `True`
- [ ] `path` column added with `String(64), nullable=False`
- [ ] Partial unique index references `completed == 1` instead of `is_terminal == 1`
- [ ] Module imports cleanly
- [ ] No other references to `token_outcomes_table.c.is_terminal` remain in `schema.py` (grep verifies)

---

### Task 1.2: Retype `TokenOutcome` dataclass

**Files:**
- Modify: `src/elspeth/contracts/audit.py:673-703`

**Step 1: Write the failing test FIRST**

Create or extend: `tests/unit/contracts/test_audit.py`

```python
"""Tests for ADR-019 TokenOutcome dataclass shape change."""

from datetime import datetime, timezone

import pytest

from elspeth.contracts.audit import TokenOutcome
from elspeth.contracts.enums import TerminalOutcome, TerminalPath


class TestTokenOutcomeTwoAxis:
    """ADR-019 Phase 1: TokenOutcome carries (outcome, path, completed)."""

    def test_completed_outcome_has_outcome_path_completed(self) -> None:
        """A completed-state TokenOutcome has all three two-axis fields."""
        record = TokenOutcome(
            outcome_id="out_test_01",
            run_id="run_001",
            token_id="tok_001",
            outcome=TerminalOutcome.SUCCESS,
            path=TerminalPath.DEFAULT_FLOW,
            completed=True,
            recorded_at=datetime.now(timezone.utc),
            sink_name="primary",
        )
        assert record.outcome == TerminalOutcome.SUCCESS
        assert record.path == TerminalPath.DEFAULT_FLOW
        assert record.completed is True

    def test_buffered_outcome_has_null_outcome_buffered_path(self) -> None:
        """A non-terminal (BUFFERED) TokenOutcome has outcome=None, path=BUFFERED, completed=False."""
        record = TokenOutcome(
            outcome_id="out_test_02",
            run_id="run_001",
            token_id="tok_001",
            outcome=None,
            path=TerminalPath.BUFFERED,
            completed=False,
            recorded_at=datetime.now(timezone.utc),
            batch_id="batch_001",
        )
        assert record.outcome is None
        assert record.path == TerminalPath.BUFFERED
        assert record.completed is False

    def test_completed_xor_outcome_invariant_completed_true_outcome_none(self) -> None:
        """Tier 1: completed=True with outcome=None is an invariant violation — crash."""
        with pytest.raises(ValueError, match="completed"):
            TokenOutcome(
                outcome_id="out_test_03",
                run_id="run_001",
                token_id="tok_001",
                outcome=None,
                path=TerminalPath.DEFAULT_FLOW,
                completed=True,  # mismatch
                recorded_at=datetime.now(timezone.utc),
            )

    def test_completed_xor_outcome_invariant_completed_false_outcome_set(self) -> None:
        """Tier 1: completed=False with outcome=SUCCESS is an invariant violation — crash."""
        with pytest.raises(ValueError, match="completed"):
            TokenOutcome(
                outcome_id="out_test_04",
                run_id="run_001",
                token_id="tok_001",
                outcome=TerminalOutcome.SUCCESS,
                path=TerminalPath.BUFFERED,
                completed=False,  # mismatch
                recorded_at=datetime.now(timezone.utc),
            )

    def test_legal_pair_required(self) -> None:
        """Tier 1: an unknown (outcome, path) pair is an invariant violation — crash."""
        with pytest.raises(ValueError, match="legal"):
            TokenOutcome(
                outcome_id="out_test_05",
                run_id="run_001",
                token_id="tok_001",
                outcome=TerminalOutcome.SUCCESS,
                path=TerminalPath.UNROUTED,  # SUCCESS+UNROUTED is not a legal pair
                completed=True,
                recorded_at=datetime.now(timezone.utc),
            )
```

**Step 2: Run the tests to verify they fail (RED)**

Run: `.venv/bin/python -m pytest tests/unit/contracts/test_audit.py::TestTokenOutcomeTwoAxis -v`

Expected: All five tests fail with `TypeError` ("unexpected keyword argument 'path'") — the dataclass doesn't have the new fields yet.

**Step 3: Update the dataclass**

Apply this edit to `src/elspeth/contracts/audit.py:673-703`:

```python
# OLD:
@dataclass(frozen=True, slots=True)
class TokenOutcome:
    """Recorded terminal state for a token.

    Captures the moment a token reached its terminal (or buffered) state.
    Part of AUD-001 audit integrity - explicit rather than derived.
    """

    outcome_id: str
    run_id: str
    token_id: str
    outcome: RowOutcome  # Direct type, not forward reference
    is_terminal: bool
    recorded_at: datetime

    # Outcome-specific fields (nullable based on outcome type)
    sink_name: str | None = None
    batch_id: str | None = None
    fork_group_id: str | None = None
    join_group_id: str | None = None
    expand_group_id: str | None = None
    error_hash: str | None = None
    context_json: str | None = None
    expected_branches_json: str | None = None  # Branch contract for FORKED/EXPANDED

    def __post_init__(self) -> None:
        """Validate enum and bool fields - Tier 1 crash on invalid types."""
        _validate_enum(self.outcome, RowOutcome, "outcome")
        if not isinstance(self.is_terminal, bool):
            raise TypeError(f"is_terminal must be bool, got {type(self.is_terminal).__name__}: {self.is_terminal!r}")

# NEW:
@dataclass(frozen=True, slots=True)
class TokenOutcome:
    """Recorded terminal state for a token (ADR-019 two-axis model).

    Captures the moment a token reached its terminal (or buffered) state.
    Part of AUD-001 audit integrity - explicit rather than derived.

    ``outcome`` is the lifecycle answer (TerminalOutcome) when ``completed=True``,
    or ``None`` when ``completed=False`` (only BUFFERED today). ``path`` is the
    provenance answer (TerminalPath), always populated. ``completed`` mirrors
    the prior ``is_terminal`` field per ADR-019 sub-decision 3.

    The ``__post_init__`` invariants enforce three Tier 1 constraints:
    1. ``completed XOR (outcome IS NULL)`` — bool/outcome consistency.
    2. ``(outcome, path) ∈ _LEGAL_TERMINAL_PAIRS`` when completed=True.
    3. ``path == BUFFERED`` when completed=False (only non-terminal path today).
    """

    outcome_id: str
    run_id: str
    token_id: str
    outcome: TerminalOutcome | None
    path: TerminalPath
    completed: bool
    recorded_at: datetime

    # Outcome-specific fields (nullable based on (outcome, path) pair)
    sink_name: str | None = None
    batch_id: str | None = None
    fork_group_id: str | None = None
    join_group_id: str | None = None
    expand_group_id: str | None = None
    error_hash: str | None = None
    context_json: str | None = None
    expected_branches_json: str | None = None  # Branch contract for FORK_PARENT/EXPAND_PARENT

    def __post_init__(self) -> None:
        """Validate two-axis invariants — Tier 1 crash on invalid combinations."""
        # I0a: completed/outcome consistency
        if not isinstance(self.completed, bool):
            raise TypeError(
                f"completed must be bool, got {type(self.completed).__name__}: {self.completed!r}"
            )
        if self.completed and self.outcome is None:
            raise ValueError(
                f"TokenOutcome {self.outcome_id}: completed=True requires non-NULL outcome "
                f"(ADR-019 § Decision invariant: completed XOR (outcome IS NULL))"
            )
        if not self.completed and self.outcome is not None:
            raise ValueError(
                f"TokenOutcome {self.outcome_id}: completed=False requires outcome=None "
                f"(got outcome={self.outcome!r})"
            )

        # I0b: enum types
        if self.outcome is not None:
            _validate_enum(self.outcome, TerminalOutcome, "outcome")
        _validate_enum(self.path, TerminalPath, "path")

        # I0c: legal pair when terminal
        if self.completed:
            assert self.outcome is not None  # invariant from I0a
            if (self.outcome, self.path) not in _LEGAL_TERMINAL_PAIRS:
                raise ValueError(
                    f"TokenOutcome {self.outcome_id}: ({self.outcome!r}, {self.path!r}) "
                    f"is not in _LEGAL_TERMINAL_PAIRS — see ADR-019 § Mapping table."
                )
        else:
            # I0d: non-terminal path
            if self.path != TerminalPath.BUFFERED:
                raise ValueError(
                    f"TokenOutcome {self.outcome_id}: completed=False requires "
                    f"path=BUFFERED (got path={self.path!r})"
                )
```

Update the imports at the top of `contracts/audit.py`:

```python
# Replace ``from elspeth.contracts.enums import RowOutcome`` with:
from elspeth.contracts.enums import (
    _LEGAL_TERMINAL_PAIRS,
    TerminalOutcome,
    TerminalPath,
)
```

**Step 4: Run the tests to verify they pass (GREEN)**

Run: `.venv/bin/python -m pytest tests/unit/contracts/test_audit.py::TestTokenOutcomeTwoAxis -v`

Expected: All five tests pass.

**Step 5: Verify other tests in the contracts test suite still pass**

Run: `.venv/bin/python -m pytest tests/unit/contracts/test_audit.py -v`

Expected: every existing test that constructs `TokenOutcome` will fail because the old shape (`outcome=RowOutcome.X, is_terminal=True`) is gone. Update each existing test fixture in this file in the same commit to use `(outcome=TerminalOutcome.SUCCESS, path=TerminalPath.DEFAULT_FLOW, completed=True)` style. This is schema-dependent test fixup per Phase 5 § Test triage; do NOT defer to Stage 4.

**Definition of Done:**
- [ ] Five new tests in `TestTokenOutcomeTwoAxis` pass
- [ ] All pre-existing tests in `test_audit.py` updated and passing
- [ ] `TokenOutcome.__post_init__` enforces all four I0 invariants
- [ ] No `RowOutcome` references remain in `contracts/audit.py` (grep verifies)
- [ ] mypy passes on `src/elspeth/contracts/audit.py`

---

### Task 1.3: Retype `RowResult` dataclass

**Files:**
- Modify: `src/elspeth/contracts/results.py:379-421`

**Step 1: Write the failing test FIRST**

Extend `tests/unit/contracts/test_results.py` with:

```python
class TestRowResultTwoAxis:
    """ADR-019 Phase 1: RowResult carries (outcome, path) at the producer site."""

    def test_completed_row_result(self) -> None:
        token = TokenInfo(token_id="tok_001", row_id="row_001", run_id="run_001")
        result = RowResult(
            token=token,
            final_data=PipelineRow(row={"k": "v"}, contract=None),
            outcome=TerminalOutcome.SUCCESS,
            path=TerminalPath.DEFAULT_FLOW,
            sink_name="primary",
        )
        assert result.outcome == TerminalOutcome.SUCCESS
        assert result.path == TerminalPath.DEFAULT_FLOW

    def test_routed_on_error_requires_error_field(self) -> None:
        token = TokenInfo(token_id="tok_001", row_id="row_001", run_id="run_001")
        with pytest.raises(OrchestrationInvariantError, match="ON_ERROR_ROUTED"):
            RowResult(
                token=token,
                final_data=PipelineRow(row={"k": "v"}, contract=None),
                outcome=TerminalOutcome.FAILURE,
                path=TerminalPath.ON_ERROR_ROUTED,
                sink_name="error_sink",
                error=None,  # missing — must crash
            )

    def test_buffered_row_result(self) -> None:
        token = TokenInfo(token_id="tok_001", row_id="row_001", run_id="run_001")
        result = RowResult(
            token=token,
            final_data=PipelineRow(row={"k": "v"}, contract=None),
            outcome=None,
            path=TerminalPath.BUFFERED,
        )
        assert result.outcome is None
        assert result.path == TerminalPath.BUFFERED
```

**Step 2: Run RED**

Run: `.venv/bin/python -m pytest tests/unit/contracts/test_results.py::TestRowResultTwoAxis -v`

Expected fail: `TypeError: __init__() got an unexpected keyword argument 'path'`.

**Step 3: Update the dataclass**

Apply edit at `src/elspeth/contracts/results.py:379-421`:

```python
@dataclass(frozen=True, slots=True)
class RowResult:
    """Final result of processing a row through the pipeline (ADR-019 two-axis).

    Producers emit (outcome, path) pairs at the producer site per ADR-019
    § "Classification is producer-declared, not topology-derivable." The
    recorder writes the pair without re-derivation.

    Fields:
        token: Token identity for this row instance
        final_data: Final row data as PipelineRow (may be original if failed early)
        outcome: Lifecycle answer (None for non-terminal BUFFERED rows)
        path: Provenance answer (always populated)
        sink_name: For paths that reach a sink, the destination sink name
        error: For (FAILURE, ON_ERROR_ROUTED), type-safe error details for audit
    """

    token: TokenInfo
    final_data: PipelineRow
    outcome: TerminalOutcome | None
    path: TerminalPath
    sink_name: str | None = None
    error: FailureInfo | None = None

    def __post_init__(self) -> None:
        # The recorder will rerun the (outcome, path) legality check; here we
        # catch obvious construction-site bugs early. The post-Stage-2 contract
        # is: every legal terminal pair has its required fields documented in
        # the ADR Implementation Notes table; we mirror those here.
        if self.outcome is None and self.path != TerminalPath.BUFFERED:
            raise OrchestrationInvariantError(
                f"RowResult: outcome=None requires path=BUFFERED, got path={self.path!r}"
            )
        if self.outcome is not None and self.path == TerminalPath.BUFFERED:
            raise OrchestrationInvariantError(
                f"RowResult: path=BUFFERED requires outcome=None, got outcome={self.outcome!r}"
            )

        # Per-pair sink_name and error invariants
        if self.path == TerminalPath.DEFAULT_FLOW and self.sink_name is None:
            raise OrchestrationInvariantError(
                "(SUCCESS, DEFAULT_FLOW) outcome requires sink_name to be set"
            )
        if self.path == TerminalPath.GATE_ROUTED and self.sink_name is None:
            raise OrchestrationInvariantError(
                "(SUCCESS, GATE_ROUTED) outcome requires sink_name to be set"
            )
        if self.path == TerminalPath.ON_ERROR_ROUTED:
            if self.sink_name is None:
                raise OrchestrationInvariantError(
                    "(FAILURE, ON_ERROR_ROUTED) outcome requires sink_name to be set"
                )
            if self.error is None:
                raise OrchestrationInvariantError(
                    "(FAILURE, ON_ERROR_ROUTED) outcome requires error (FailureInfo) to be set — "
                    "the originating transform error must be captured on the outcome record for "
                    "single-hop audit attributability."
                )
            if not isinstance(self.error, FailureInfo):
                raise OrchestrationInvariantError(
                    "(FAILURE, ON_ERROR_ROUTED) outcome requires error to be a FailureInfo instance"
                )
        if self.path == TerminalPath.COALESCED and self.sink_name is None:
            raise OrchestrationInvariantError(
                "(SUCCESS, COALESCED) outcome requires sink_name to be set"
            )
```

Update imports — replace `from elspeth.contracts.enums import RowOutcome` with `from elspeth.contracts.enums import TerminalOutcome, TerminalPath`.

**Step 4: GREEN**

Run: `.venv/bin/python -m pytest tests/unit/contracts/test_results.py::TestRowResultTwoAxis -v`

Expected: all three tests pass.

**Step 5: Update existing RowResult tests in this file**

Pre-existing test fixtures construct `RowResult(outcome=RowOutcome.X, ...)`. Update each to the new shape with the canonical pair from `tests/unit/contracts/test_enums.py::_ROW_OUTCOME_TO_TWO_AXIS_MAPPING`. This is schema-dependent test fixup per Phase 5; do NOT defer.

**Definition of Done:**
- [ ] Three new TestRowResultTwoAxis tests pass
- [ ] All pre-existing RowResult tests updated and passing
- [ ] `__post_init__` enforces (outcome, path) consistency invariants
- [ ] mypy passes
- [ ] No `RowOutcome` references remain in `contracts/results.py` (grep verifies)

---

### Task 1.4: Retype `PendingOutcome` dataclass

**Files:**
- Modify: `src/elspeth/contracts/engine.py:46-100`

**Step 1: Write the failing test FIRST**

Extend `tests/unit/contracts/test_engine_contracts.py`:

```python
class TestPendingOutcomeTwoAxis:
    """ADR-019 Phase 1: PendingOutcome carries (outcome, path) for sink-durable recording."""

    def test_pending_outcome_completed(self) -> None:
        po = PendingOutcome(
            outcome=TerminalOutcome.SUCCESS,
            path=TerminalPath.DEFAULT_FLOW,
        )
        assert po.outcome == TerminalOutcome.SUCCESS
        assert po.path == TerminalPath.DEFAULT_FLOW
        assert po.error_hash is None

    def test_pending_outcome_routed_on_error_requires_hash(self) -> None:
        with pytest.raises(ValueError, match="error_hash"):
            PendingOutcome(
                outcome=TerminalOutcome.FAILURE,
                path=TerminalPath.ON_ERROR_ROUTED,
                error_hash=None,  # required for ON_ERROR_ROUTED path
            )

    def test_pending_outcome_failed_requires_hash(self) -> None:
        with pytest.raises(ValueError, match="error_hash"):
            PendingOutcome(
                outcome=TerminalOutcome.FAILURE,
                path=TerminalPath.UNROUTED,
                error_hash=None,
            )

    def test_pending_outcome_quarantined_requires_hash(self) -> None:
        with pytest.raises(ValueError, match="error_hash"):
            PendingOutcome(
                outcome=TerminalOutcome.FAILURE,
                path=TerminalPath.QUARANTINED_AT_SOURCE,
                error_hash=None,
            )

    def test_pending_outcome_completed_must_not_have_hash(self) -> None:
        with pytest.raises(ValueError, match="error_hash"):
            PendingOutcome(
                outcome=TerminalOutcome.SUCCESS,
                path=TerminalPath.DEFAULT_FLOW,
                error_hash="abcd1234abcd1234",
            )
```

**Step 2: Run RED**

Run: `.venv/bin/python -m pytest tests/unit/contracts/test_engine_contracts.py::TestPendingOutcomeTwoAxis -v`

Expected fail: `TypeError: __init__() got an unexpected keyword argument 'path'`.

**Step 3: Update the dataclass**

Apply edit at `src/elspeth/contracts/engine.py:46-100`:

```python
# Replace the entire PendingOutcome class:
@dataclass(frozen=True, slots=True)
class PendingOutcome:
    """Pending token outcome waiting for sink durability confirmation (ADR-019).

    Carries (outcome, path) pairs through the pending_tokens queue for sink
    durability sequencing per the original ADR-018 motivation: token outcomes
    must only be recorded after sink write + flush complete successfully.

    The ``_REQUIRES_ERROR_HASH_PATHS`` set encodes the ADR-019 mapping:
    paths that require error_hash for single-hop audit attributability are
    (FAILURE, ON_ERROR_ROUTED), (FAILURE, UNROUTED), (FAILURE, QUARANTINED_AT_SOURCE),
    (TRANSIENT, SINK_FALLBACK_TO_FAILSINK), and (FAILURE, SINK_DISCARDED).
    """

    # Paths that require ``error_hash`` on PendingOutcome. Indexed by path
    # rather than (outcome, path) because the path uniquely identifies the
    # error-carrying scenarios under the new model.
    _REQUIRES_ERROR_HASH_PATHS: ClassVar[frozenset[TerminalPath]] = frozenset(
        {
            TerminalPath.ON_ERROR_ROUTED,
            TerminalPath.UNROUTED,
            TerminalPath.QUARANTINED_AT_SOURCE,
            TerminalPath.SINK_FALLBACK_TO_FAILSINK,
            TerminalPath.SINK_DISCARDED,
        }
    )

    outcome: TerminalOutcome | None
    path: TerminalPath
    error_hash: str | None = None

    def __post_init__(self) -> None:
        if self.path in self._REQUIRES_ERROR_HASH_PATHS and (
            self.error_hash is None or not self.error_hash.strip()
        ):
            raise ValueError(
                f"PendingOutcome with path={self.path.name} requires non-empty error_hash"
            )
        if self.path not in self._REQUIRES_ERROR_HASH_PATHS and self.error_hash is not None:
            raise ValueError(
                f"PendingOutcome with path={self.path.name} must not have error_hash"
            )
```

Update imports: replace `from elspeth.contracts.enums import RowOutcome` with `from elspeth.contracts.enums import TerminalOutcome, TerminalPath`.

**Step 4: GREEN**

Run: `.venv/bin/python -m pytest tests/unit/contracts/test_engine_contracts.py::TestPendingOutcomeTwoAxis -v`

Expected: five tests pass.

**Step 5: Commit so far**

```bash
git add src/elspeth/contracts/audit.py src/elspeth/contracts/results.py src/elspeth/contracts/engine.py \
        tests/unit/contracts/test_audit.py tests/unit/contracts/test_results.py tests/unit/contracts/test_engine_contracts.py
# Don't commit yet — Task 1.5 still touches the contracts package.
```

**Definition of Done:**
- [ ] Five new TestPendingOutcomeTwoAxis tests pass
- [ ] mypy passes
- [ ] No `RowOutcome` references remain in `contracts/engine.py`

---

### Task 1.5: Retype `TokenCompleted` telemetry event

**Files:**
- Modify: `src/elspeth/contracts/events.py:242-249`

**Step 1: Write the failing test FIRST**

Extend `tests/unit/contracts/test_events.py`:

```python
class TestTokenCompletedTwoAxis:
    """ADR-019 Phase 1: TokenCompleted telemetry event carries (outcome, path)."""

    def test_token_completed_carries_outcome_and_path(self) -> None:
        evt = TokenCompleted(
            run_id="run_001",
            row_id="row_001",
            token_id="tok_001",
            outcome=TerminalOutcome.SUCCESS,
            path=TerminalPath.DEFAULT_FLOW,
            sink_name="primary",
        )
        assert evt.outcome == TerminalOutcome.SUCCESS
        assert evt.path == TerminalPath.DEFAULT_FLOW
```

**Step 2: Run RED**

Expected: `TypeError: __init__() got an unexpected keyword argument 'path'`.

**Step 3: Update the dataclass**

```python
# OLD (lines 242-249):
@dataclass(frozen=True, slots=True)
class TokenCompleted(TelemetryEvent):
    """Emitted when a token reaches its terminal state."""

    row_id: str
    token_id: str
    outcome: RowOutcome
    sink_name: str | None

# NEW:
@dataclass(frozen=True, slots=True)
class TokenCompleted(TelemetryEvent):
    """Emitted when a token reaches its terminal state (ADR-019 two-axis)."""

    row_id: str
    token_id: str
    outcome: TerminalOutcome | None
    path: TerminalPath
    sink_name: str | None
```

Update imports at the top of `events.py` to import `TerminalOutcome` and `TerminalPath` instead of `RowOutcome`.

**Step 4: GREEN**

Run: `.venv/bin/python -m pytest tests/unit/contracts/test_events.py::TestTokenCompletedTwoAxis -v`

Expected: pass.

**Step 5: Update telemetry consumers in this file and downstream**

Telemetry consumers that read `TokenCompleted.outcome` and downstream-produce `RowOutcome`-named values may exist. Grep:

```bash
grep -rn "TokenCompleted" src/ tests/
```

For each consumer, update field reads from `evt.outcome.X` (RowOutcome) to `(evt.outcome, evt.path)` pair handling. Most telemetry consumers will simply log both fields verbatim — minimal behavioural change.

**Definition of Done:**
- [ ] TestTokenCompletedTwoAxis passes
- [ ] All TokenCompleted consumers in src/ updated
- [ ] mypy passes

---

### Task 1.6: Update `record_token_outcome` recorder signature

**Files:**
- Modify: `src/elspeth/core/landscape/data_flow_repository.py:203-307` (`_validate_outcome_fields`)
- Modify: `src/elspeth/core/landscape/data_flow_repository.py:802-880` (`record_token_outcome`)
- Modify: `src/elspeth/core/landscape/data_flow_repository.py:570-580, 785-795` (the two internal callers that emit `FORKED` and `EXPANDED` directly)
- Modify: `src/elspeth/core/landscape/data_flow_repository.py:895-940` (`get_token_outcome` ORDER BY at line 899; `get_token_outcomes_for_row` SELECT column list at line 930) — pure column rename `is_terminal` → `completed`; the loader at Task 1.7 already expects `row.completed`

**Step 1: Write the failing test FIRST**

Create or extend `tests/unit/core/landscape/test_data_flow_repository.py` with:

```python
class TestRecordTokenOutcomeTwoAxis:
    """ADR-019 Phase 1: recorder writes (outcome, path, completed) triple."""

    def test_record_completed_default_flow(self, audit_repo, run_id, token_ref) -> None:
        outcome_id = audit_repo.record_token_outcome(
            ref=token_ref,
            outcome=TerminalOutcome.SUCCESS,
            path=TerminalPath.DEFAULT_FLOW,
            sink_name="primary",
        )
        loaded = audit_repo.get_token_outcome(token_ref.token_id)
        assert loaded is not None
        assert loaded.outcome == TerminalOutcome.SUCCESS
        assert loaded.path == TerminalPath.DEFAULT_FLOW
        assert loaded.completed is True

    def test_record_buffered(self, audit_repo, run_id, token_ref) -> None:
        outcome_id = audit_repo.record_token_outcome(
            ref=token_ref,
            outcome=None,
            path=TerminalPath.BUFFERED,
            batch_id="batch_001",
        )
        loaded = audit_repo.get_token_outcome(token_ref.token_id)
        assert loaded.outcome is None
        assert loaded.path == TerminalPath.BUFFERED
        assert loaded.completed is False

    def test_record_illegal_pair_crashes(self, audit_repo, run_id, token_ref) -> None:
        with pytest.raises(ValueError, match="legal"):
            audit_repo.record_token_outcome(
                ref=token_ref,
                outcome=TerminalOutcome.SUCCESS,
                path=TerminalPath.UNROUTED,  # illegal pair
                sink_name="x",
            )

    def test_record_default_flow_requires_sink_name(self, audit_repo, run_id, token_ref) -> None:
        with pytest.raises(ValueError, match="sink_name"):
            audit_repo.record_token_outcome(
                ref=token_ref,
                outcome=TerminalOutcome.SUCCESS,
                path=TerminalPath.DEFAULT_FLOW,
                sink_name=None,
            )

    # Plus tests for every (outcome, path) → required_fields row from the ADR
    # Implementation Notes table at lines 638-660.
```

**Step 2: Run RED**

Expected: each test fails with `TypeError: record_token_outcome() got an unexpected keyword argument 'path'`.

**Step 3: Rewrite `_validate_outcome_fields`**

Replace the entire `_validate_outcome_fields` block (lines 203-307) with a `(outcome, path)`-driven version. The 13 if-branches in the old code map mechanically to the 13 legal pairs:

```python
# Required-field rules per ADR-019 Implementation Notes table (lines 638-660).
# Maps (TerminalOutcome | None, TerminalPath) → tuple of required field names.
# An empty tuple means "no required fields beyond the path itself."
_REQUIRED_FIELDS_BY_PAIR: dict[tuple[TerminalOutcome | None, TerminalPath], tuple[str, ...]] = {
    (TerminalOutcome.SUCCESS, TerminalPath.DEFAULT_FLOW): ("sink_name",),
    (TerminalOutcome.SUCCESS, TerminalPath.GATE_ROUTED): ("sink_name",),
    (TerminalOutcome.FAILURE, TerminalPath.ON_ERROR_ROUTED): ("sink_name", "error_hash"),
    (TerminalOutcome.SUCCESS, TerminalPath.FILTER_DROPPED): (),
    (TerminalOutcome.SUCCESS, TerminalPath.COALESCED): ("sink_name", "join_group_id"),
    (TerminalOutcome.FAILURE, TerminalPath.UNROUTED): ("error_hash",),
    (TerminalOutcome.FAILURE, TerminalPath.QUARANTINED_AT_SOURCE): ("error_hash",),
    (TerminalOutcome.TRANSIENT, TerminalPath.SINK_FALLBACK_TO_FAILSINK): ("sink_name", "error_hash"),
    (TerminalOutcome.FAILURE, TerminalPath.SINK_DISCARDED): ("sink_name", "error_hash"),
    (TerminalOutcome.TRANSIENT, TerminalPath.FORK_PARENT): ("fork_group_id",),
    (TerminalOutcome.TRANSIENT, TerminalPath.EXPAND_PARENT): ("expand_group_id",),
    (TerminalOutcome.TRANSIENT, TerminalPath.BATCH_CONSUMED): ("batch_id",),
    (None, TerminalPath.BUFFERED): ("batch_id",),
}


def _validate_outcome_fields(
    self,
    outcome: TerminalOutcome | None,
    path: TerminalPath,
    *,
    sink_name: str | None,
    batch_id: str | None,
    fork_group_id: str | None,
    join_group_id: str | None,
    expand_group_id: str | None,
    error_hash: str | None,
) -> None:
    """Validate required fields for the (outcome, path) pair.

    Per ADR-019 Implementation Notes invariant-translation table at
    docs/architecture/adr/019-two-axis-terminal-model.md lines 638-660.
    Defense-in-depth: producers SHOULD pass correct fields, but bugs in
    producer code crash here at write time rather than corrupting the
    audit DB.

    Raises:
        ValueError: If the pair is not in _LEGAL_TERMINAL_PAIRS, or if
            a required field for the pair is missing.
    """
    pair = (outcome, path)
    if pair not in _REQUIRED_FIELDS_BY_PAIR:
        raise ValueError(
            f"Unhandled (outcome, path) pair in validation: {pair!r}. "
            f"See ADR-019 § Mapping table (lines 99-115) and update "
            f"_REQUIRED_FIELDS_BY_PAIR with the new pair."
        )
    required = _REQUIRED_FIELDS_BY_PAIR[pair]
    field_values = {
        "sink_name": sink_name,
        "batch_id": batch_id,
        "fork_group_id": fork_group_id,
        "join_group_id": join_group_id,
        "expand_group_id": expand_group_id,
        "error_hash": error_hash,
    }
    for field_name in required:
        if field_values[field_name] is None:
            raise ValueError(
                f"({pair[0].name if pair[0] else 'NULL'}, {pair[1].name}) outcome "
                f"requires {field_name} but got None. "
                f"Contract violation — see ADR-019 § Implementation Notes table."
            )
```

**Step 4: Rewrite `record_token_outcome` signature**

Replace the existing method (lines 802-880) with:

```python
def record_token_outcome(
    self,
    ref: TokenRef,
    outcome: TerminalOutcome | None,
    path: TerminalPath,
    *,
    sink_name: str | None = None,
    batch_id: str | None = None,
    fork_group_id: str | None = None,
    join_group_id: str | None = None,
    expand_group_id: str | None = None,
    error_hash: str | None = None,
    context: Mapping[str, object] | None = None,
) -> str:
    """Record a token's (outcome, path) audit terminal in the audit trail.

    Called at the moment the producer determines the terminal pair. For
    BUFFERED tokens (outcome=None, path=BUFFERED), a second call records
    the actual lifecycle terminal when the batch flushes.

    Validates that the token belongs to the specified run_id before recording.
    Cross-run contamination crashes immediately per Tier 1 trust model.

    Per ADR-019 § "Classification is producer-declared, not topology-derivable":
    the (outcome, path) pair is the producer's declaration; the recorder
    writes it without re-derivation.

    Args:
        ref: TokenRef bundling token_id and run_id
        outcome: TerminalOutcome lifecycle answer, or None for BUFFERED
        path: TerminalPath provenance answer (always required)
        sink_name: For paths that reach a sink (REQUIRED for those)
        batch_id: For BATCH_CONSUMED / BUFFERED (REQUIRED)
        fork_group_id: For FORK_PARENT (REQUIRED)
        join_group_id: For COALESCED (REQUIRED)
        expand_group_id: For EXPAND_PARENT (REQUIRED)
        error_hash: For UNROUTED / QUARANTINED_AT_SOURCE / ON_ERROR_ROUTED /
                    SINK_FALLBACK_TO_FAILSINK / SINK_DISCARDED (REQUIRED)
        context: Optional additional context (stored as JSON)

    Returns:
        outcome_id for tracking

    Raises:
        ValueError: If (outcome, path) is illegal or required fields missing
        AuditIntegrityError: If token does not belong to the specified run, or
            if a cross-table invariant fails (Phase 4 will add I1c, I3 here)
        IntegrityError: If terminal outcome already exists for token
    """
    self._validate_outcome_fields(
        outcome,
        path,
        sink_name=sink_name,
        batch_id=batch_id,
        fork_group_id=fork_group_id,
        join_group_id=join_group_id,
        expand_group_id=expand_group_id,
        error_hash=error_hash,
    )
    self._validate_token_run_ownership(ref)

    outcome_id = f"out_{generate_id()[:12]}"
    completed = outcome is not None  # I0a invariant
    context_json = canonical_json(context) if context is not None else None

    self._ops.execute_insert(
        token_outcomes_table.insert().values(
            outcome_id=outcome_id,
            run_id=ref.run_id,
            token_id=ref.token_id,
            outcome=outcome.value if outcome is not None else None,
            path=path.value,
            completed=1 if completed else 0,
            recorded_at=now(),
            sink_name=sink_name,
            batch_id=batch_id,
            fork_group_id=fork_group_id,
            join_group_id=join_group_id,
            expand_group_id=expand_group_id,
            error_hash=error_hash,
            context_json=context_json,
        )
    )

    return outcome_id
```

**Step 5: Update the two internal recorder callers (FORKED at line 572, EXPANDED at line 787)**

These two call `record_token_outcome` with `outcome=RowOutcome.FORKED` / `RowOutcome.EXPANDED`. Update to:

```python
# Line ~572 (FORKED parent recording):
self.record_token_outcome(
    ref=parent_ref,
    outcome=TerminalOutcome.TRANSIENT,
    path=TerminalPath.FORK_PARENT,
    fork_group_id=fork_group_id,
)

# Line ~787 (EXPANDED parent recording):
self.record_token_outcome(
    ref=parent_ref,
    outcome=TerminalOutcome.TRANSIENT,
    path=TerminalPath.EXPAND_PARENT,
    expand_group_id=expand_group_id,
)
```

**Step 5b: Rename `is_terminal` → `completed` in the read-side query sites (lines 899 and 930)**

The `record_token_outcome` rewrite in Step 4 changes the inserted column name from `is_terminal` to `completed`. Two read-side query sites in the same file still reference the old column name and must be renamed in lock-step or the queries will fail with a SQLAlchemy "no such column" error against the new schema. The loader at Task 1.7 already expects `row.completed`, so these are pure column renames with no semantic change.

```python
# Line ~899 (get_token_outcome ORDER BY clause):
# OLD:
.order_by(
    token_outcomes_table.c.is_terminal.desc(),  # Terminal first
    token_outcomes_table.c.recorded_at.desc(),  # Then by time
)

# NEW:
.order_by(
    token_outcomes_table.c.completed.desc(),    # Terminal first (ADR-019 column rename)
    token_outcomes_table.c.recorded_at.desc(),  # Then by time
)
```

```python
# Line ~930 (get_token_outcomes_for_row SELECT column list):
# OLD:
select(
    token_outcomes_table.c.outcome_id,
    token_outcomes_table.c.run_id,
    token_outcomes_table.c.token_id,
    token_outcomes_table.c.outcome,
    token_outcomes_table.c.is_terminal,
    token_outcomes_table.c.recorded_at,
    ...
)

# NEW:
select(
    token_outcomes_table.c.outcome_id,
    token_outcomes_table.c.run_id,
    token_outcomes_table.c.token_id,
    token_outcomes_table.c.outcome,
    token_outcomes_table.c.path,        # ADR-019 new column (always populated)
    token_outcomes_table.c.completed,   # ADR-019 column rename (was is_terminal)
    token_outcomes_table.c.recorded_at,
    ...
)
```

The `path` column must be added to this SELECT list because `TokenOutcomeLoader.load` (Task 1.7) reads `row.path` as a load-bearing field. Verify the SELECT lists every column the loader reads — `outcome_id`, `run_id`, `token_id`, `outcome`, `path`, `completed`, `recorded_at`, `sink_name`, `batch_id`, `fork_group_id`, `join_group_id`, `expand_group_id`, `error_hash`, `context_json`, `expected_branches_json`.

**Step 6: GREEN**

Run: `.venv/bin/python -m pytest tests/unit/core/landscape/test_data_flow_repository.py::TestRecordTokenOutcomeTwoAxis -v`

Expected: all tests pass.

**Definition of Done:**
- [ ] `_validate_outcome_fields` rewritten with `_REQUIRED_FIELDS_BY_PAIR` table
- [ ] `record_token_outcome` signature flipped to `(outcome, path)`
- [ ] Internal recorder callers updated for FORK_PARENT / EXPAND_PARENT
- [ ] Read-side queries renamed: `get_token_outcome` ORDER BY (line ~899) and `get_token_outcomes_for_row` SELECT column list (line ~930) use `completed` (was `is_terminal`); SELECT list includes the new `path` column
- [ ] All TestRecordTokenOutcomeTwoAxis tests pass
- [ ] mypy passes
- [ ] No `RowOutcome` references remain in `data_flow_repository.py`
- [ ] No `is_terminal` references remain in `data_flow_repository.py` (verify with `grep -n "is_terminal" src/elspeth/core/landscape/data_flow_repository.py` returning empty)

---

### Task 1.7: Update `TokenOutcomeLoader.load` cross-checks

**Files:**
- Modify: `src/elspeth/core/landscape/model_loaders.py:525-609`

**Step 1: Write the failing test FIRST**

Create or extend `tests/unit/core/landscape/test_token_outcome_loader.py`:

```python
class TestTokenOutcomeLoaderTwoAxis:
    """ADR-019 Phase 1: loader runs two-axis cross-checks at read time."""

    def test_loads_completed_default_flow(self, audit_db) -> None:
        # Given a recorded (SUCCESS, DEFAULT_FLOW) outcome,
        # When the loader reads it,
        # Then we get a TokenOutcome with the correct fields.
        ...

    def test_completed_xor_outcome_violation_crashes(self, audit_db) -> None:
        # Given a tampered DB with completed=1 and outcome=NULL,
        # When the loader reads it,
        # Then AuditIntegrityError fires.
        ...

    def test_illegal_pair_in_db_crashes(self, audit_db) -> None:
        # Given a tampered DB with (SUCCESS, UNROUTED) — not legal,
        # Then AuditIntegrityError fires.
        ...

    def test_required_field_missing_crashes(self, audit_db) -> None:
        # Given a tampered DB with (SUCCESS, DEFAULT_FLOW) but sink_name=NULL,
        # Then AuditIntegrityError fires.
        ...
```

(The fixtures in this test class can use direct INSERT statements through the SQLAlchemy connection — bypassing the recorder — to simulate the "tampered DB" scenario. This is a Tier 1 read guard test.)

**Step 2: Run RED**

Expected: tests fail because the loader still reads the old `is_terminal` column.

**Step 3: Rewrite `TokenOutcomeLoader.load`**

Replace lines 525-609 with:

```python
def load(self, row: Row[Any]) -> TokenOutcome:
    """Load a TokenOutcome from a token_outcomes row, with Tier 1 cross-checks.

    Per ADR-019 § Cross-check invariants and CLAUDE.md "Three-Tier Trust Model":
    audit DB is OUR data; crash on any anomaly. The cross-checks fall into
    five layers, all run before constructing the dataclass:

    1. completed type check (must be int 0 or 1, never bool/str/etc.)
    2. outcome value coercion (TerminalOutcome.X or None)
    3. path value coercion (TerminalPath.X)
    4. completed XOR (outcome IS NULL) cross-check
    5. (outcome, path) ∈ _LEGAL_TERMINAL_PAIRS when completed
    6. required-field cross-check per ADR-019 Implementation Notes table

    Raises:
        AuditIntegrityError: any cross-check violation
    """
    oid = row.outcome_id

    # 1. completed type check
    if type(row.completed) is not int or row.completed not in (0, 1):
        raise AuditIntegrityError(
            f"TokenOutcome {oid}: invalid completed={row.completed!r} (expected int 0 or 1) "
            f"— audit integrity violation"
        )
    completed = row.completed == 1

    # 2. outcome value coercion (None for non-terminal)
    outcome: TerminalOutcome | None
    if row.outcome is None:
        outcome = None
    else:
        try:
            outcome = TerminalOutcome(row.outcome)
        except ValueError as exc:
            raise AuditIntegrityError(
                f"TokenOutcome {oid}: invalid outcome={row.outcome!r} not in TerminalOutcome — "
                f"audit integrity violation"
            ) from exc

    # 3. path value coercion (always non-NULL)
    if row.path is None:
        raise AuditIntegrityError(
            f"TokenOutcome {oid}: path is NULL — audit integrity violation "
            f"(path is always populated under ADR-019)"
        )
    try:
        path = TerminalPath(row.path)
    except ValueError as exc:
        raise AuditIntegrityError(
            f"TokenOutcome {oid}: invalid path={row.path!r} not in TerminalPath — "
            f"audit integrity violation"
        ) from exc

    # 4. completed XOR (outcome IS NULL)
    if completed != (outcome is not None):
        raise AuditIntegrityError(
            f"TokenOutcome {oid}: completed={completed} but outcome={outcome!r} — "
            f"completed must be true iff outcome is non-NULL "
            f"(ADR-019 § Decision invariant)"
        )

    # 5. (outcome, path) ∈ _LEGAL_TERMINAL_PAIRS when completed; else path == BUFFERED
    if completed:
        assert outcome is not None  # invariant from check 4
        if (outcome, path) not in _LEGAL_TERMINAL_PAIRS:
            raise AuditIntegrityError(
                f"TokenOutcome {oid}: ({outcome!r}, {path!r}) not in _LEGAL_TERMINAL_PAIRS "
                f"— see ADR-019 § Mapping table"
            )
    else:
        if path != TerminalPath.BUFFERED:
            raise AuditIntegrityError(
                f"TokenOutcome {oid}: completed=False requires path=BUFFERED, got {path!r} "
                f"— audit integrity violation"
            )

    # 6. Required-field cross-check per ADR-019 Implementation Notes table.
    # Mirrors _REQUIRED_FIELDS_BY_PAIR in data_flow_repository.py — the read
    # path repeats the write path's check because audit-DB tampering bypasses
    # the write check.
    pair: tuple[TerminalOutcome | None, TerminalPath] = (outcome, path)
    required_fields_for_pair = _REQUIRED_FIELDS_BY_PAIR.get(pair, ())
    field_values = {
        "sink_name": row.sink_name,
        "batch_id": row.batch_id,
        "fork_group_id": row.fork_group_id,
        "join_group_id": row.join_group_id,
        "expand_group_id": row.expand_group_id,
        "error_hash": row.error_hash,
    }
    for field_name in required_fields_for_pair:
        if field_values[field_name] is None:
            raise AuditIntegrityError(
                f"TokenOutcome {oid}: ({outcome!r}, {path!r}) requires {field_name} but "
                f"DB has NULL — audit integrity violation"
            )

    return TokenOutcome(
        outcome_id=oid,
        run_id=row.run_id,
        token_id=row.token_id,
        outcome=outcome,
        path=path,
        completed=completed,
        recorded_at=row.recorded_at,
        sink_name=row.sink_name,
        batch_id=row.batch_id,
        fork_group_id=row.fork_group_id,
        join_group_id=row.join_group_id,
        expand_group_id=row.expand_group_id,
        error_hash=row.error_hash,
        context_json=row.context_json,
        expected_branches_json=row.expected_branches_json,
    )
```

To avoid duplicating `_REQUIRED_FIELDS_BY_PAIR` in two files, **extract it into `src/elspeth/contracts/audit.py`** (the canonical location alongside `TokenOutcome`) and import in both `data_flow_repository.py` and `model_loaders.py`. The dict is the canonical machine-readable encoding of ADR-019 § Implementation Notes table.

**Step 4: GREEN**

Run: `.venv/bin/python -m pytest tests/unit/core/landscape/test_token_outcome_loader.py -v`

Expected: all tests pass.

**Definition of Done:**
- [ ] Loader runs all six cross-checks
- [ ] `_REQUIRED_FIELDS_BY_PAIR` extracted to `contracts/audit.py` and shared
- [ ] All TestTokenOutcomeLoaderTwoAxis tests pass
- [ ] mypy passes

---

### Task 1.8: Update package re-exports — `contracts/__init__.py` and `testing/__init__.py`

**Why this task exists:** Stage 1 added `TerminalOutcome` and `TerminalPath` to `src/elspeth/contracts/enums.py` but did NOT add them to the public `contracts/__init__.py` re-export block (only `RowOutcome` is currently re-exported there). Phase 2 producer-flip imports use the broad `from elspeth.contracts import RowResult, TerminalOutcome, TerminalPath, ...` pattern (mirroring the existing `from elspeth.contracts import RowOutcome` shape); without the re-exports, every Phase 2 import fails with `ImportError: cannot import name 'TerminalOutcome' from 'elspeth.contracts'`. **Verified 2026-05-05** against current HEAD `60d30551`: `grep "Terminal" src/elspeth/contracts/__init__.py` returns empty.

**Files:**
- Modify: `src/elspeth/contracts/__init__.py` (re-export block around line 150 and `__all__` block around line 388)
- Modify: `src/elspeth/testing/__init__.py:31-45, 507-540, 715-740`

**Step 1a: Add `TerminalOutcome` / `TerminalPath` to `contracts/__init__.py`**

Find the existing `RowOutcome` re-export (line ~150 of `contracts/__init__.py`) and add the two new enums alongside in the same import block:

```python
# OLD (around line 145-152):
from elspeth.contracts.enums import (
    ...,
    RowOutcome,
    ...,
)

# NEW:
from elspeth.contracts.enums import (
    ...,
    RowOutcome,
    TerminalOutcome,
    TerminalPath,
    ...,
)
```

Find the `__all__` block (line ~388) and add the two new names alongside `"RowOutcome"`:

```python
__all__ = [
    ...,
    "RowOutcome",
    "TerminalOutcome",
    "TerminalPath",
    ...,
]
```

**Verify the re-export:**

```bash
.venv/bin/python -c "from elspeth.contracts import TerminalOutcome, TerminalPath, RowOutcome; print('OK')"
```

Expected: `OK`. Without this, Phase 2 producer imports crash with `ImportError`.

**Step 1b: Add `TerminalOutcome` and `TerminalPath` to the testing pack re-exports**

Lines 31-45 of `testing/__init__.py` contain a list of re-exports for `from elspeth.testing import ...`. Add the two new enum types alongside `RowOutcome` (RowOutcome stays — Stage 5 deletes it).

```python
# Around line 35:
from elspeth.contracts.enums import (
    RowOutcome,
    TerminalOutcome,
    TerminalPath,
)
```

**Step 2: Update default-outcome callsites in testing helper builders**

Lines 507-540 and 715-740 use `outcome or RowOutcome.COMPLETED` defaults in test scaffolding helpers. Update each to construct the (outcome, path) pair:

```python
# OLD (line ~520):
resolved_outcome = outcome or RowOutcome.COMPLETED

# NEW:
# Test helpers default to (SUCCESS, DEFAULT_FLOW) — the canonical
# "happy path" pair per ADR-019 mapping table.
if outcome is None and path is None:
    resolved_outcome = TerminalOutcome.SUCCESS
    resolved_path = TerminalPath.DEFAULT_FLOW
else:
    resolved_outcome = outcome
    resolved_path = path
```

The signature of these helpers will need a `path: TerminalPath | None = None` parameter alongside the existing `outcome` parameter. Each helper should accept both for caller flexibility.

**Step 3: Verify both re-exports work**

```bash
.venv/bin/python -c "from elspeth.contracts import TerminalOutcome, TerminalPath, RowOutcome; print('contracts: OK')"
.venv/bin/python -c "from elspeth.testing import TerminalOutcome, TerminalPath, RowOutcome; print('testing: OK')"
```

Expected output: both `OK` lines.

**Definition of Done:**
- [ ] `TerminalOutcome` and `TerminalPath` re-exported from `elspeth.contracts` (the broad `from elspeth.contracts import ...` pattern used in Phase 2 producer files now works)
- [ ] `TerminalOutcome` and `TerminalPath` re-exported from `elspeth.testing`
- [ ] Both names appear in `contracts/__init__.py::__all__` and the corresponding `testing/__init__.py` exports list
- [ ] Helper defaults updated to construct (outcome, path) pairs
- [ ] mypy passes
- [ ] ChaosEngine fixtures (in `tests/`) that depend on these helpers still compile

---

### Task 1.9: Fix downstream schema consumers (MCP analyzers, Web execution, exporter, lineage, formatters)

**Why this task exists:** Phase 1 changes the `token_outcomes` schema (column rename + new column + `outcome` value space). EIGHT downstream consumers in the MCP analyzer pack, the Web execution layer, the JSONL exporter, the lineage helper, and the CLI formatter read those columns directly via SQL or via the `TokenOutcome` dataclass's `is_terminal` property. The original plan missed these — they were discovered during a 2026-05-05 consumer-surface sweep. Without them in Phase 1, MCP `diagnose()` silently returns zero quarantines (Tier 1 audit-integrity violation per CLAUDE.md), the Web run-diagnostics view crashes on the renamed column, and the discard-summary widget under-counts.

**Files:** see "Files touched in this phase" at the top of this document.

**Step 1: Write the silent-failure regression test (RED) for the diagnostics quarantine bug**

This is the most dangerous of the eight bugs: it doesn't crash, it silently lies. Encode it as a regression test FIRST so the fix is anchored.

Create: `tests/unit/mcp/test_diagnose_quarantine_count.py`

```python
"""ADR-019 B3: diagnose() must report the correct quarantine count under
the two-axis terminal model.

Pre-fix bug: mcp/analyzers/diagnostics.py:181 hardcodes
``outcome == "quarantined"``. After Phase 1's recorder writes
``outcome="failure"`` and ``path="quarantined_at_source"``, that filter
matches zero rows. diagnose() reports "0 quarantines" with confidence —
exactly the silent-wrong-result class CLAUDE.md Tier 1 forbids.
"""

import pytest

from elspeth.contracts.enums import TerminalOutcome, TerminalPath
from elspeth.mcp.analyzers.diagnostics import diagnose


def test_diagnose_counts_quarantined_under_new_path(audit_db_with_quarantines):
    """A run with 3 quarantined rows reports 3 quarantines via diagnose()."""
    # Fixture writes 3 (FAILURE, QUARANTINED_AT_SOURCE) outcomes.
    result = diagnose(audit_db_with_quarantines.conn)
    quarantine_problems = [
        p for p in result["problems"] if p["type"] == "quarantined_rows"
    ]
    assert len(quarantine_problems) == 1
    assert quarantine_problems[0]["count"] == 3, (
        f"Expected diagnose() to count 3 quarantined rows, got "
        f"{quarantine_problems[0]['count']}. The filter must match the "
        f"new TerminalPath.QUARANTINED_AT_SOURCE value, not the legacy "
        f"RowOutcome.QUARANTINED string 'quarantined'."
    )
```

**Step 2: Run RED**

```bash
.venv/bin/python -m pytest tests/unit/mcp/test_diagnose_quarantine_count.py -v
```

Expected: test fails with `count == 0` (the bug). After the fix in Step 4, it passes.

**Step 3: Patch `mcp/analyzers/reports.py` (5 sites in `get_outcome_analysis`)**

```python
# Lines 657-665 area (the SELECT and GROUP BY):
# OLD:
stmt = (
    select(
        token_outcomes_table.c.outcome,
        token_outcomes_table.c.is_terminal,
        func.count(token_outcomes_table.c.outcome_id).label("count"),
    )
    .where(token_outcomes_table.c.run_id == run_id)
    .group_by(token_outcomes_table.c.outcome, token_outcomes_table.c.is_terminal)
)

# NEW:
# ADR-019: the outcome distribution is naturally keyed by (outcome, path)
# under the two-axis model. Group by both; the wire schema exposes both
# fields plus the renamed `completed` for query ergonomics.
stmt = (
    select(
        token_outcomes_table.c.outcome,
        token_outcomes_table.c.path,
        token_outcomes_table.c.completed,
        func.count(token_outcomes_table.c.outcome_id).label("count"),
    )
    .where(token_outcomes_table.c.run_id == run_id)
    .group_by(
        token_outcomes_table.c.outcome,
        token_outcomes_table.c.path,
        token_outcomes_table.c.completed,
    )
)
```

```python
# Lines 698-710 area (the dict construction + summary aggregation):
# OLD:
outcomes.append(
    {
        "outcome": row.outcome,
        "is_terminal": bool(row.is_terminal),
        "count": row.count,
    }
)
...
terminal_count = sum(o["count"] for o in outcomes if o["is_terminal"])
non_terminal_count = sum(o["count"] for o in outcomes if not o["is_terminal"])

# NEW:
outcomes.append(
    {
        "outcome": row.outcome,           # TerminalOutcome.value or NULL
        "path": row.path,                 # TerminalPath.value (always populated)
        "completed": bool(row.completed), # ADR-019 rename — replaces is_terminal
        "count": row.count,
    }
)
...
terminal_count = sum(o["count"] for o in outcomes if o["completed"])
non_terminal_count = sum(o["count"] for o in outcomes if not o["completed"])
```

**Step 4: Patch `mcp/analyzers/diagnostics.py:181` — the silent-failure fix**

```python
# OLD (line 181):
.where(token_outcomes_table.c.outcome == "quarantined")

# NEW:
# ADR-019: under the two-axis model, the quarantine signal lives on
# the path column, not the outcome column. (FAILURE, QUARANTINED_AT_SOURCE)
# is the canonical encoding; path alone is sufficient because no other
# legal pair uses QUARANTINED_AT_SOURCE.
.where(token_outcomes_table.c.path == TerminalPath.QUARANTINED_AT_SOURCE.value)
```

Add `from elspeth.contracts.enums import TerminalPath` to the imports.

**Step 5: Patch `mcp/types.py:364` — TypedDict wire schema**

```python
# OLD:
class OutcomeDistributionEntry(TypedDict):
    outcome: str
    is_terminal: bool
    count: int

# NEW:
class OutcomeDistributionEntry(TypedDict):
    """Single entry in outcome distribution (ADR-019 two-axis terminal model).

    Keyed by (outcome, path). ``outcome`` is the TerminalOutcome value or NULL
    for non-terminal rows. ``path`` is the TerminalPath value (always populated).
    ``completed`` mirrors the recorder's ``completed`` column — true iff the
    row reached a terminal state.
    """
    outcome: str | None  # TerminalOutcome.value or NULL (non-terminal)
    path: str            # TerminalPath.value
    completed: bool
    count: int
```

**Wire-format note:** This is a breaking change to the MCP outcome-analysis response shape. Per CLAUDE.md "no legacy code" + "no users yet," breaking is acceptable; the renamed field flows from the recorder schema rename and the added field surfaces a load-bearing dimension that could not be exposed under the single-axis model. Operator MCP clients that destructured `is_terminal` must update to `completed`. Documented in `docs/operator/migrations/adr-019.md` (Phase 5).

**Step 6: Patch `web/execution/diagnostics.py:170`**

```python
# OLD:
.outerjoin(
    token_outcomes_table,
    and_(
        token_outcomes_table.c.token_id == tokens_table.c.token_id,
        token_outcomes_table.c.run_id == tokens_table.c.run_id,
        token_outcomes_table.c.is_terminal == 1,
    ),
)

# NEW:
.outerjoin(
    token_outcomes_table,
    and_(
        token_outcomes_table.c.token_id == tokens_table.c.token_id,
        token_outcomes_table.c.run_id == tokens_table.c.run_id,
        token_outcomes_table.c.completed == 1,
    ),
)
```

**Step 7: Patch `web/execution/discard_summary.py:92`**

```python
# OLD:
.where(token_outcomes_table.c.is_terminal == 1)

# NEW:
.where(token_outcomes_table.c.completed == 1)
```

The discard-summary widget can ALSO benefit from a path-aware filter under the new model — the widget today counts rows where `sink_name == "__discard__"`, which conflates failsink-mode `(TRANSIENT, SINK_FALLBACK_TO_FAILSINK)` (sink_name = the actual failsink name, NOT `__discard__`) with discard-mode `(FAILURE, SINK_DISCARDED)` (sink_name = `__discard__`). The existing `sink_name == DISCARD_SINK_NAME` filter already isolates the discard-mode case, so no semantic change is needed beyond the column rename. Verify the widget's count under the new model produces the same number as before for discard-mode-only runs.

**Step 8: Patch `core/landscape/exporter.py:430`**

```python
# OLD (line 430):
"outcome": outcome.outcome.value,
"is_terminal": outcome.is_terminal,

# NEW:
# ADR-019: outcome value space is TerminalOutcome.value or None;
# completed mirrors the renamed is_terminal column; path is the new
# always-populated provenance field.
"outcome": outcome.outcome.value if outcome.outcome is not None else None,
"path": outcome.path.value,
"completed": outcome.completed,
```

The JSONL export is a wire format. The same breaking-change reasoning as the MCP TypedDict applies — operators regenerate exports from the new audit DB; old JSONL exports remain readable as historical snapshots but the new format adds `path` and renames `is_terminal` → `completed`.

**Step 9: Patch `core/landscape/lineage.py:118`**

```python
# OLD:
terminal_outcomes = [o for o in outcomes if o.is_terminal]

# NEW:
terminal_outcomes = [o for o in outcomes if o.completed]
```

Trivial property rename. The `TokenOutcome` dataclass exposes `completed: bool` after Phase 1 Task 1.2; `is_terminal` no longer exists.

**Step 10: Patch `core/landscape/formatters.py:170`**

```python
# OLD:
lines.append(f"Outcome: {result.outcome.outcome.name}")
if result.outcome.sink_name:
    lines.append(f"Sink: {result.outcome.sink_name}")
lines.append(f"Terminal: {result.outcome.is_terminal}")

# NEW:
# ADR-019: print both axes — operator CLI output should make the new
# (outcome, path) pair visible at a glance.
outcome_name = result.outcome.outcome.name if result.outcome.outcome else "NULL"
lines.append(f"Outcome: {outcome_name}")
lines.append(f"Path: {result.outcome.path.name}")
if result.outcome.sink_name:
    lines.append(f"Sink: {result.outcome.sink_name}")
lines.append(f"Completed: {result.outcome.completed}")
```

**Step 11: Final consumer-surface sweep**

After applying Steps 1-10, run a residual-hit grep to confirm zero misses:

```bash
echo "=== Residual is_terminal column reads (must be 0) ==="
grep -rn "token_outcomes_table.c.is_terminal\|outcome.is_terminal" src/elspeth/ \
    | grep -v "/contracts/\|test_"
# Expected: empty.
# NOTE: the previous version of this sweep excluded data_flow_repository.py to silence
# noise from the in-scope sites at lines 573/788/859/868. That file-level exclusion
# also masked the read-side query sites at lines 899 and 930. Task 1.6 Step 5b now
# covers those lines, so the exclusion is removed — by end of Phase 1, the file is
# 100% migrated and any residual hit indicates a missed consumer.

echo "=== Residual hardcoded RowOutcome.value SQL filters (must be 0) ==="
grep -rn 'outcome\s*==\s*"\(completed\|routed\|routed_on_error\|forked\|failed\|quarantined\|diverted\|consumed_in_batch\|dropped_by_filter\|coalesced\|expanded\|buffered\)"' src/elspeth/
# Expected: empty.

echo "=== Residual is_terminal references in src/ that aren't event-progress (which is unrelated) ==="
grep -rn "is_terminal" src/elspeth/ | grep -v "/web/execution/progress.py" | grep -v "/__pycache__/"
# Expected: only legitimate references in tests, contracts/freeze, or comments.
```

If any sweep shows a hit not addressed by Steps 1-10, STOP and surface to user — there's a missed consumer.

**Step 12: GREEN — both regression tests + suite**

```bash
.venv/bin/python -m pytest tests/unit/mcp/test_diagnose_quarantine_count.py tests/unit/mcp/test_outcome_analysis.py -v
.venv/bin/python -m pytest tests/unit/mcp/ tests/unit/web/execution/ tests/unit/core/landscape/test_exporter.py tests/unit/core/landscape/test_lineage.py -q
```

Expected: all green. Existing tests with stale assertions (`is_terminal` field reads, hardcoded `"quarantined"` string assertions) need fixture updates in the same commit — this is schema-dependent test fixup per Phase 5 § Test triage Category C, but rolled into Phase 1 because the Web/MCP fixtures co-located with the consumer code make a clean commit boundary.

**Definition of Done:**
- [ ] Eight downstream consumer sites patched (5 in `mcp/analyzers/reports.py`, 1 in `mcp/types.py`, 1 in `mcp/analyzers/diagnostics.py`, 1 in `web/execution/diagnostics.py`, 1 in `web/execution/discard_summary.py`, 1 in `core/landscape/exporter.py`, 1 in `core/landscape/lineage.py`, 1 in `core/landscape/formatters.py`)
- [ ] B3 silent-zero-quarantine regression test passes
- [ ] B2 wire-schema test passes (new MCP outcome-analysis test)
- [ ] Final consumer-surface sweep shows zero residual hits
- [ ] mypy clean across `mcp/`, `web/execution/`, `core/landscape/`
- [ ] No `RowOutcome` references remain in `src/elspeth/mcp/` or `src/elspeth/web/execution/`

---

### Task 1.10: Phase 1 commit

**Step 1: Run all Phase 1 tests**

```bash
.venv/bin/python -m pytest \
    tests/unit/contracts/ \
    tests/unit/core/landscape/ \
    tests/unit/mcp/ \
    tests/unit/web/execution/ \
    -v
```

All tests must pass — including the eight downstream-consumer fixes and the B3 regression test added in Task 1.9.

**Step 2: Run quality gates**

```bash
.venv/bin/python -m mypy src/elspeth/contracts/ src/elspeth/core/landscape/ src/elspeth/testing/ \
    src/elspeth/mcp/ src/elspeth/web/execution/
.venv/bin/python -m ruff check src/elspeth/contracts/ src/elspeth/core/landscape/ src/elspeth/testing/ \
    src/elspeth/mcp/ src/elspeth/web/execution/
.venv/bin/python -m ruff format --check src/elspeth/contracts/ src/elspeth/core/landscape/ src/elspeth/testing/ \
    src/elspeth/mcp/ src/elspeth/web/execution/
```

All gates must pass.

**Step 3: Note that the engine module will NOT import yet**

After this commit, importing `elspeth.engine.orchestrator` will fail because every producer site still calls `record_token_outcome(outcome=RowOutcome.X, ...)`. This is expected and only resolved by Phase 2's commit. **Do not push or merge after Phase 1 alone — Phase 2 must follow in the same PR.**

**Step 4: Commit**

```bash
git add src/elspeth/contracts/ \
        src/elspeth/core/landscape/ \
        src/elspeth/testing/ \
        src/elspeth/mcp/ \
        src/elspeth/web/execution/ \
        tests/unit/contracts/ \
        tests/unit/core/landscape/ \
        tests/unit/mcp/ \
        tests/unit/web/execution/

git commit -m "$(cat <<'EOF'
feat(adr-019): phase 1 — schema + recorder + loader + dataclass two-axis flip + downstream consumer fixes

ADR-019 Stage 2/3 Phase 1 of 5 (see docs/superpowers/plans/2026-05-04-adr-019-stage-2-3-overview.md).

Schema:
- token_outcomes: rename is_terminal → completed; add path column; outcome
  column nullability flipped to True (NULL means non-terminal/BUFFERED).
- Partial unique index references completed instead of is_terminal.

Dataclasses retyped to (outcome: TerminalOutcome | None, path: TerminalPath, completed: bool):
- TokenOutcome (audit.py) with four __post_init__ Tier 1 invariants:
  type-checked, completed XOR outcome-NULL, legal pair, BUFFERED-on-non-terminal.
- RowResult (results.py) with per-pair sink_name/error invariants.
- PendingOutcome (engine.py) with _REQUIRES_ERROR_HASH_PATHS encoding the
  five paths that need error_hash.
- TokenCompleted (events.py) telemetry payload extended with path field.

Recorder + loader:
- _REQUIRED_FIELDS_BY_PAIR canonical dict in contracts/audit.py encodes
  ADR-019 Implementation Notes table; consumed by both write and read paths.
- record_token_outcome signature: (outcome, path) pair instead of RowOutcome.
- TokenOutcomeLoader.load runs six layered cross-checks (type, value coercion,
  completed XOR, legal pair, BUFFERED-non-terminal, required-field).

Testing pack:
- testing/__init__.py re-exports TerminalOutcome and TerminalPath alongside
  RowOutcome (Stage 5 drops RowOutcome).
- Default-outcome helper signatures accept the new pair.

Downstream consumer fixes (8 sites — discovered during 2026-05-05
consumer-surface sweep; without these, MCP diagnose() silently lies about
zero quarantines and Web run-diagnostics crashes on the renamed column):
- mcp/types.py: OutcomeDistributionEntry retyped — is_terminal renamed to
  completed, path field added (wire-schema breaking change documented in
  docs/operator/migrations/adr-019.md, Phase 5).
- mcp/analyzers/reports.py: get_outcome_analysis SELECT/GROUP BY/dict-key
  rename + add path grouping (5 sites).
- mcp/analyzers/diagnostics.py:181: ``outcome == "quarantined"`` →
  ``path == "quarantined_at_source"``. The hardcoded RowOutcome.value
  string would silently match zero rows under the new model — Tier 1
  audit-integrity violation per CLAUDE.md.
- web/execution/diagnostics.py:170 + web/execution/discard_summary.py:92:
  is_terminal → completed in JOIN/WHERE clauses.
- core/landscape/exporter.py:430: JSONL token_outcome export field rename
  + path field addition.
- core/landscape/lineage.py:118: TokenOutcome property read renamed.
- core/landscape/formatters.py:170: CLI formatter prints both outcome.name
  AND path.name; Terminal: line renamed to Completed:.

This commit alone breaks engine.orchestrator import — every producer site
still passes RowOutcome to record_token_outcome. Phase 2 is the producer
flip that restores end-to-end imports. The merge is atomic per ADR-019
lines 318-320.

Refs: elspeth-949719575e (Stage 2 ticket)
ADR: docs/architecture/adr/019-two-axis-terminal-model.md

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

**Definition of Done:**
- [ ] All Phase 1 tests pass
- [ ] mypy / ruff / format clean on the affected paths
- [ ] Commit landed
- [ ] Phase 2 starts in the next session/checkpoint
