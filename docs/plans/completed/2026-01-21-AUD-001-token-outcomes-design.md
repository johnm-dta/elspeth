# AUD-001: Explicit Token Outcome Recording

**Date:** 2026-01-21
**Status:** ✅ IMPLEMENTED (2026-01-22)
**Author:** Claude (with architecture review)

## Implementation Status

**✅ COMPLETE** - All planned components implemented and tested:
- ✅ `token_outcomes` table added to schema (schema.py:115-145)
- ✅ `record_token_outcome()` API in recorder (recorder.py:2146+)
- ✅ `TokenOutcome` dataclass in contracts
- ✅ Partial unique index enforces "one terminal outcome per token"
- ✅ Integration tests pass (test_token_outcomes.py, test_processor_outcomes.py)
- ✅ 17 recording sites in processor.py
- ⚠️ 1 remaining bug: P1-2026-01-21-token-outcome-group-ids-mismatch.md

**Evidence:** 9 files reference token_outcomes infrastructure

## Problem Statement

Terminal states (COMPLETED, ROUTED, FORKED, etc.) are currently **derived at query time** from multiple tables (`node_states`, `token_parents`, `batch_members`, `routing_events`). This creates audit fragility:

1. Derivation logic is spread across 900+ lines in `processor.py`
2. No single source of truth exists
3. If derivation logic changes, historical outcomes could change retroactively
4. Cannot audit "how was this outcome determined?"

**Requirement (from requirements.md):**
> AUD-001: Every token reaches exactly one terminal state

## Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Storage model | Explicit records | "If it's not recorded, it didn't happen" (CLAUDE.md) |
| Recording timing | Immediate at determination | Capture exact moment; crash-safe |
| Multiple records | Append-only | BUFFERED → terminal progression preserved |
| Schema approach | Explicit columns + JSON | Queryable core fields; flexible metadata |

## Schema

```python
# In src/elspeth/core/landscape/schema.py

token_outcomes_table = Table(
    "token_outcomes",
    metadata,
    # Identity
    Column("outcome_id", String(64), primary_key=True),
    Column("run_id", String(64), ForeignKey("runs.run_id"), nullable=False, index=True),
    Column("token_id", String(64), ForeignKey("tokens.token_id"), nullable=False, index=True),

    # Core outcome
    Column("outcome", String(32), nullable=False),
    Column("is_terminal", Boolean, nullable=False),  # False only for BUFFERED
    Column("recorded_at", DateTime(timezone=True), nullable=False),

    # Outcome-specific fields (nullable, used based on outcome type)
    Column("sink_name", String(128)),                                    # ROUTED, COMPLETED
    Column("batch_id", String(64), ForeignKey("batches.batch_id")),      # CONSUMED_IN_BATCH
    Column("fork_group_id", String(64)),                                 # FORKED
    Column("join_group_id", String(64)),                                 # COALESCED
    Column("expand_group_id", String(64)),                               # EXPANDED
    Column("error_hash", String(64)),                                    # FAILED, QUARANTINED

    # Optional extended context
    Column("context_json", Text, nullable=True),

    # Constraints
    CheckConstraint(
        "outcome IN ('completed', 'routed', 'forked', 'failed', 'quarantined', "
        "'consumed_in_batch', 'coalesced', 'expanded', 'buffered')",
        name="ck_token_outcomes_outcome"
    ),
)

# Enforce "exactly one terminal outcome per token"
Index(
    "uq_token_terminal_outcome",
    token_outcomes_table.c.token_id,
    unique=True,
    sqlite_where=(token_outcomes_table.c.is_terminal == True),
    postgresql_where=(token_outcomes_table.c.is_terminal == True),
)
```

### Column Usage by Outcome

| Outcome | `sink_name` | `batch_id` | `fork_group_id` | `join_group_id` | `expand_group_id` | `error_hash` |
|---------|-------------|------------|-----------------|-----------------|-------------------|--------------|
| COMPLETED | sink name | | | | | |
| ROUTED | sink name | | | | | |
| FORKED | | | group ID | | | |
| CONSUMED_IN_BATCH | | batch ID | | | | |
| COALESCED | | | | group ID | | |
| EXPANDED | | | | | group ID | |
| FAILED | | | | | | error hash |
| QUARANTINED | | | | | | error hash |
| BUFFERED | | batch ID | | | | |

## Recorder API

```python
# In src/elspeth/core/landscape/recorder.py

def record_token_outcome(
    self,
    run_id: str,
    token_id: str,
    outcome: RowOutcome,
    *,
    sink_name: str | None = None,
    batch_id: str | None = None,
    fork_group_id: str | None = None,
    join_group_id: str | None = None,
    expand_group_id: str | None = None,
    error_hash: str | None = None,
    context: dict[str, Any] | None = None,
) -> str:
    """Record a token's outcome in the audit trail.

    Called at the moment the outcome is determined in processor.py.
    For BUFFERED tokens, a second call records the terminal outcome
    when the batch flushes.

    Args:
        run_id: Current run ID
        token_id: Token that reached this outcome
        outcome: The RowOutcome enum value
        sink_name: For ROUTED/COMPLETED - which sink
        batch_id: For CONSUMED_IN_BATCH/BUFFERED - which batch
        fork_group_id: For FORKED - the fork group
        join_group_id: For COALESCED - the join group
        expand_group_id: For EXPANDED - the expand group
        error_hash: For FAILED/QUARANTINED - hash of error details
        context: Optional additional context (stored as JSON)

    Returns:
        outcome_id for tracking

    Raises:
        IntegrityError: If terminal outcome already exists for token
    """
```

## Processor Changes

17 locations in `processor.py` return `RowResult` with an outcome. Each needs a `record_token_outcome()` call:

| Line | Outcome | Context to Record |
|------|---------|-------------------|
| 205 | FAILED | Aggregation flush failure |
| 228 | COMPLETED | Single-mode aggregation output |
| 287 | COMPLETED | Multi-row aggregation output |
| 319 | CONSUMED_IN_BATCH | Transform mode trigger |
| 344 | COMPLETED | Passthrough flush output |
| 361 | BUFFERED | Passthrough non-flush |
| 370 | CONSUMED_IN_BATCH | Passthrough trigger |
| 638 | ROUTED | Gate routes to sink |
| 669 | FORKED | Gate forks to paths |
| 706 | FAILED | MaxRetriesExceeded |
| 720 | QUARANTINED | Transform error with discard |
| 730 | ROUTED | Transform error to sink |
| 775 | EXPANDED | Multi-row transform output |
| 816 | ROUTED | Config gate routes to sink |
| 848 | FORKED | Config gate forks |
| 883 | COALESCED | Coalesce merge complete |
| 892 | COMPLETED | Default end-of-pipeline |

### Implementation Pattern

```python
# Before (current):
return RowResult(
    token=token,
    outcome=RowOutcome.ROUTED,
    sink_name=sink_name,
)

# After (with outcome recording):
self._recorder.record_token_outcome(
    run_id=self._run_id,
    token_id=token.token_id,
    outcome=RowOutcome.ROUTED,
    sink_name=sink_name,
)
return RowResult(
    token=token,
    outcome=RowOutcome.ROUTED,
    sink_name=sink_name,
)
```

## Migration Strategy

1. **Alembic migration** creates `token_outcomes` table
2. **No backfill** - existing runs continue using derived logic
3. **Forward-only** - new runs get explicit outcomes
4. **Explain query** updated to prefer `token_outcomes` if present, fall back to derivation

This follows ELSPETH's "no backwards compatibility" principle while maintaining query functionality.

## Testing Strategy

1. **Unit tests** for `record_token_outcome()` method
2. **Integration tests** verifying each of 17 processor paths records outcome
3. **Constraint tests** verifying unique terminal outcome enforcement
4. **Explain tests** verifying outcomes appear in lineage queries

### Key Test Cases

```python
def test_terminal_outcome_uniqueness():
    """Inserting two terminal outcomes for same token raises IntegrityError."""

def test_buffered_then_terminal():
    """BUFFERED followed by terminal outcome succeeds."""

def test_all_outcome_types_recorded():
    """Each RowOutcome type can be recorded with appropriate fields."""

def test_explain_includes_outcome():
    """explain() returns recorded outcome, not derived."""
```

## Architecture Review

**Reviewer:** axiom-system-architect:architecture-critic
**Score:** 2/5 (original) → addressed all critical/high issues

### Issues Addressed

| Issue | Resolution |
|-------|------------|
| Missing `run_id` | Added to schema |
| JSON blob for core data | Extracted explicit columns |
| No uniqueness constraint | Added partial unique index |
| BUFFERED ambiguity | Added `is_terminal` column |
| No CHECK constraint | Added enum validation |

## Files to Modify

1. `src/elspeth/core/landscape/schema.py` - Add table definition
2. `src/elspeth/core/landscape/recorder.py` - Add `record_token_outcome()` method
3. `src/elspeth/engine/processor.py` - Add 17 recording calls
4. `src/elspeth/core/landscape/lineage.py` - Update `explain()` to include outcomes
5. `src/elspeth/contracts/__init__.py` - Export new dataclass if needed
6. `alembic/versions/xxx_add_token_outcomes.py` - Migration
7. `tests/core/test_token_outcomes.py` - New test file
8. `tests/engine/test_processor_outcomes.py` - Integration tests
