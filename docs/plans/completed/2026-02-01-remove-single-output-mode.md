# Remove `output_mode='single'` Implementation Plan

**Status:** ✅ IMPLEMENTED (2026-02-01)

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Remove the broken `output_mode='single'` from batch aggregation, fixing bug elspeth-rapid-nd3 where the triggering token is reused instead of creating a new output token.

**Architecture:** Two code paths handle "single" mode - one correct (handle_timeout_flush, lines 570-620) and one buggy (_process_batch_aggregation_node, lines 870-922). We remove "single" entirely and consolidate on "transform" mode semantics. An optional `expected_output_count` parameter preserves N→1 cardinality enforcement.

**Tech Stack:** Python 3.11+, Pydantic 2.x, pytest, SQLAlchemy

**Beads Issues:** elspeth-rapid-nd3 (bug), elspeth-rapid-3me (removal), elspeth-rapid-360 (tests)

---

## Implementation Summary

- `output_mode` restricted to `passthrough`/`transform`, with validator rejecting `single` (`src/elspeth/core/config.py`).
- Aggregation flush paths now handle only `passthrough` and `transform` semantics (`src/elspeth/engine/processor.py`).
- Documentation and examples updated to reflect the removal (see references in this plan).

## Peer Review (Risk/Complexity)

**Code path validation**
- `src/elspeth/engine/processor.py`: `handle_timeout_flush()` currently has a `single` branch that uses `expand_token()` with the first buffered token as parent (matches plan).
- `src/elspeth/engine/processor.py`: `_process_batch_aggregation_node()` currently reuses `current_token.token_id` in the `single` branch (bug is real and matches plan).
- `src/elspeth/core/config.py`: `AggregationSettings.output_mode` is `Literal["single", "passthrough", "transform"]` with default `"single"` (matches plan).

**High-risk gaps / required plan adjustments**
1. **`expected_output_count` is config-only right now.** No runtime path uses it. Add enforcement in BOTH flush paths (`handle_timeout_flush()` and `_process_batch_aggregation_node()`) right after output rows are derived. If `expected_output_count` is set and `len(output_rows)` mismatches, this should be a hard error (plugin contract violation) rather than a silent accept. Without this, the plan's "preserves N→1 cardinality" claim is false.
2. **Docs will be wrong if not updated.** `docs/reference/configuration.md`, `docs/contracts/plugin-protocol.md`, `docs/release/feature-inventory.md`, and `docs/audit-trail/tokens/01-outcome-path-map.md` all describe `output_mode: single` and default behavior. Add a doc update task or you'll ship contradictory docs.
3. **`assert_batch_output_exists()` targets an unpopulated table.** `batch_outputs_table` exists but is never written to in code. Either add population in the aggregation executor/processor or remove/avoid using this helper for now.
4. **Partial commit with failing tests.** Task 3 suggests committing while tests are still failing (pending the mass test update in Task 5). If your workflow expects green commits, move that commit after the test update or mark it as a deliberate WIP.

**Complexity notes**
- Moderate-to-high: touches core aggregation flow, config validation, and a wide test surface. Also requires doc updates to keep contracts accurate.
- Risk concentrates in audit trail correctness: token identity and terminal outcomes. The plan’s identity-based tests help, but only if `expected_output_count` enforcement is added.

---

## Task 1: Add Identity-Based Test Assertion Helpers

**Files:**
- Create: `tests/helpers/audit_assertions.py`

**Step 1: Create the test helper module**

```python
"""Identity-based audit trail assertions.

These helpers verify token IDENTITY (which specific tokens have which outcomes),
not just outcome COUNTS. This prevents bugs like elspeth-rapid-nd3 where
count-based tests pass even when wrong tokens get wrong outcomes.
"""

from elspeth.contracts.enums import RowOutcome
from elspeth.core.landscape import LandscapeRecorder


def get_token_outcome(recorder: LandscapeRecorder, run_id: str, token_id: str) -> RowOutcome | None:
    """Get the terminal outcome for a specific token."""
    with recorder._db.connection() as conn:
        from sqlalchemy import select
        from elspeth.core.landscape.schema import token_outcomes_table

        result = conn.execute(
            select(token_outcomes_table.c.outcome)
            .where(token_outcomes_table.c.token_id == token_id)
        ).fetchone()

        if result is None:
            return None
        return RowOutcome(result.outcome)


def assert_token_outcome(
    recorder: LandscapeRecorder,
    run_id: str,
    token_id: str,
    expected: RowOutcome,
) -> None:
    """Assert a specific token has a specific outcome.

    Use this instead of counting outcomes to verify the RIGHT tokens
    get the RIGHT outcomes.
    """
    actual = get_token_outcome(recorder, run_id, token_id)
    assert actual == expected, (
        f"Token {token_id} has outcome {actual}, expected {expected}"
    )


def assert_all_batch_members_consumed(
    recorder: LandscapeRecorder,
    run_id: str,
    batch_id: str,
) -> None:
    """Assert ALL tokens in a batch have CONSUMED_IN_BATCH outcome.

    This catches the elspeth-rapid-nd3 bug where the triggering token
    was incorrectly marked COMPLETED instead of CONSUMED_IN_BATCH.
    """
    with recorder._db.connection() as conn:
        from sqlalchemy import select
        from elspeth.core.landscape.schema import batch_members_table, token_outcomes_table

        # Get all tokens in the batch
        members = conn.execute(
            select(batch_members_table.c.token_id, batch_members_table.c.ordinal)
            .where(batch_members_table.c.batch_id == batch_id)
            .order_by(batch_members_table.c.ordinal)
        ).fetchall()

        assert len(members) > 0, f"Batch {batch_id} has no members"

        for member in members:
            token_id = member.token_id
            ordinal = member.ordinal

            outcome_row = conn.execute(
                select(token_outcomes_table.c.outcome)
                .where(token_outcomes_table.c.token_id == token_id)
            ).fetchone()

            assert outcome_row is not None, (
                f"Batch member {token_id} (ordinal {ordinal}) has no outcome recorded"
            )

            actual = RowOutcome(outcome_row.outcome)
            assert actual == RowOutcome.CONSUMED_IN_BATCH, (
                f"Batch member {token_id} (ordinal {ordinal}) has outcome {actual}, "
                f"expected CONSUMED_IN_BATCH. This indicates the triggering token "
                f"was incorrectly reused instead of creating a new output token."
            )


def assert_output_token_distinct_from_inputs(
    output_token_id: str,
    input_token_ids: list[str],
) -> None:
    """Assert output token has a DIFFERENT token_id from all inputs.

    Token-producing operations (fork, expand, aggregate, coalesce) MUST
    create new tokens. Reusing input token_ids breaks audit lineage.
    """
    assert output_token_id not in input_token_ids, (
        f"Output token {output_token_id} reuses an input token_id! "
        f"Token-producing operations must create NEW tokens for audit lineage. "
        f"Input token_ids: {input_token_ids}"
    )


def assert_batch_output_exists(
    recorder: LandscapeRecorder,
    batch_id: str,
) -> str:
    """Assert batch_outputs table has an entry for this batch.

    Returns the output_token_id for further assertions.

    NOTE: This test will FAIL until batch_outputs table population is implemented.
    """
    with recorder._db.connection() as conn:
        from sqlalchemy import select
        from elspeth.core.landscape.schema import batch_outputs_table

        result = conn.execute(
            select(batch_outputs_table)
            .where(batch_outputs_table.c.batch_id == batch_id)
        ).fetchone()

        assert result is not None, (
            f"Batch {batch_id} has no entry in batch_outputs table. "
            f"Aggregation must record the output token for audit lineage."
        )

        return result.output_id
```

**Step 2: Run tests to verify module loads**

Run: `.venv/bin/python -c "from tests.helpers.audit_assertions import assert_token_outcome; print('OK')"`
Expected: `OK`

**Step 3: Commit**

```bash
git add tests/helpers/audit_assertions.py
git commit -m "feat(tests): add identity-based audit assertion helpers

Addresses elspeth-rapid-360: These helpers verify token IDENTITY
(which specific tokens have which outcomes), not just outcome COUNTS.

- assert_token_outcome(): verify specific token has specific outcome
- assert_all_batch_members_consumed(): verify ALL batch tokens consumed
- assert_output_token_distinct_from_inputs(): verify no token reuse
- assert_batch_output_exists(): verify batch_outputs table populated

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
```

---

## Task 2: Add Config Validator and expected_output_count Parameter

**Files:**
- Modify: `src/elspeth/core/config.py:156-163`

**Step 1: Write failing test for rejected "single" mode**

Create file `tests/core/test_config_single_rejected.py`:

```python
"""Test that output_mode='single' is rejected with helpful error."""

import pytest
from pydantic import ValidationError

from elspeth.core.config import AggregationSettings, TriggerConfig


def test_aggregation_config_rejects_single_mode() -> None:
    """Config validation must reject 'single' mode with migration hint."""
    with pytest.raises(ValidationError) as exc_info:
        AggregationSettings(
            name="test_agg",
            plugin="test_plugin",
            trigger=TriggerConfig(count=5),
            output_mode="single",
        )

    error_msg = str(exc_info.value)
    assert "single" in error_msg.lower()
    assert "transform" in error_msg.lower()  # Migration hint


def test_aggregation_config_accepts_transform_mode() -> None:
    """Config validation must accept 'transform' mode."""
    settings = AggregationSettings(
        name="test_agg",
        plugin="test_plugin",
        trigger=TriggerConfig(count=5),
        output_mode="transform",
    )
    assert settings.output_mode == "transform"


def test_aggregation_config_accepts_passthrough_mode() -> None:
    """Config validation must accept 'passthrough' mode."""
    settings = AggregationSettings(
        name="test_agg",
        plugin="test_plugin",
        trigger=TriggerConfig(count=5),
        output_mode="passthrough",
    )
    assert settings.output_mode == "passthrough"


def test_aggregation_config_default_is_transform() -> None:
    """Default output_mode should be 'transform' (not 'single')."""
    settings = AggregationSettings(
        name="test_agg",
        plugin="test_plugin",
        trigger=TriggerConfig(count=5),
    )
    assert settings.output_mode == "transform"


def test_aggregation_config_expected_output_count() -> None:
    """expected_output_count validates output cardinality."""
    settings = AggregationSettings(
        name="test_agg",
        plugin="test_plugin",
        trigger=TriggerConfig(count=5),
        output_mode="transform",
        expected_output_count=1,  # N→1 aggregation
    )
    assert settings.expected_output_count == 1
```

**Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/core/test_config_single_rejected.py -v`
Expected: FAIL (single mode still accepted, expected_output_count doesn't exist)

**Step 3: Modify AggregationSettings in config.py**

In `src/elspeth/core/config.py`, replace lines 156-163:

```python
    output_mode: Literal["passthrough", "transform"] = Field(
        default="transform",
        description="How batch produces output rows. 'transform' creates new tokens "
        "with proper audit lineage. 'passthrough' enriches original tokens.",
    )
    expected_output_count: int | None = Field(
        default=None,
        description="Optional: validate aggregation produces exactly this many output rows. "
        "Use expected_output_count=1 for reductions (SUM, COUNT, AVG).",
    )
    options: dict[str, Any] = Field(
        default_factory=dict,
        description="Plugin-specific configuration options",
    )

    @field_validator("output_mode", mode="before")
    @classmethod
    def reject_single_mode(cls, v: Any) -> Any:
        """Reject deprecated 'single' mode with helpful migration message."""
        if v == "single":
            raise ValueError(
                "output_mode='single' has been removed (bug elspeth-rapid-nd3). "
                "Use output_mode='transform' instead. For N→1 aggregations, add "
                "expected_output_count=1 to validate cardinality."
            )
        return v
```

Also add the import at the top of config.py if not present:
```python
from pydantic import field_validator
```

**Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/core/test_config_single_rejected.py -v`
Expected: PASS

**Step 5: Run mypy to verify type safety**

Run: `.venv/bin/python -m mypy src/elspeth/core/config.py --no-error-summary`
Expected: No errors related to AggregationSettings

**Step 6: Commit**

```bash
git add src/elspeth/core/config.py tests/core/test_config_single_rejected.py
git commit -m "feat(config): remove 'single' output_mode, add expected_output_count

BREAKING CHANGE: output_mode='single' is removed.

- Remove 'single' from Literal type
- Change default from 'single' to 'transform'
- Add Pydantic validator with migration hint
- Add optional expected_output_count for N→1 cardinality validation

Fixes elspeth-rapid-nd3: 'single' mode reused triggering token,
breaking audit lineage. 'transform' mode correctly creates new tokens.

Migration: Change output_mode='single' to output_mode='transform'.
For N→1 aggregations, add expected_output_count=1.

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
```

---

## Task 3: Remove "single" Code Path from handle_timeout_flush

**Files:**
- Modify: `src/elspeth/engine/processor.py:570-620`

**Step 1: Write failing test for timeout flush with transform mode**

The existing tests use "single" mode. We need to verify the timeout flush path works with "transform" mode semantics. This path is ALREADY CORRECT (uses expand_token), so this is a verification step.

Run: `.venv/bin/python -m pytest tests/engine/test_aggregation_integration.py::test_aggregation_timeout_fires_during_processing -v`
Expected: FAIL (test uses output_mode="single" which is now rejected)

**Step 2: Update test to use output_mode="transform"**

In `tests/engine/test_aggregation_integration.py`, line 190, change:
```python
output_mode="single",
```
to:
```python
output_mode="transform",
```

**Step 3: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/engine/test_aggregation_integration.py::test_aggregation_timeout_fires_during_processing -v`
Expected: PASS (timeout flush path uses expand_token correctly)

**Step 4: Modify handle_timeout_flush to remove "single" branch**

In `src/elspeth/engine/processor.py`, the `handle_timeout_flush` method (around line 500) has an `if output_mode == "single":` block at lines 570-620. This code is CORRECT (uses expand_token), but we need to consolidate it with the "transform" logic.

Replace lines 570-620 with the "transform" semantics. Since the code is nearly identical, we just need to remove the `if output_mode == "single":` condition and merge into "transform" handling.

Find the section starting at line 570 and replace:
```python
        if output_mode == "single":
```
with:
```python
        if output_mode == "transform":
            # Transform mode: N input rows -> M output rows with NEW tokens
            # (Previously, "single" mode was handled here identically)
```

The body should use the same `expand_token` logic that's already there.

**Step 5: Run full aggregation test suite**

Run: `.venv/bin/python -m pytest tests/engine/test_aggregation_integration.py -v -x`
Expected: Some tests FAIL (they still use output_mode="single")

**Step 6: Commit partial progress**

```bash
git add src/elspeth/engine/processor.py tests/engine/test_aggregation_integration.py
git commit -m "refactor(processor): consolidate handle_timeout_flush to transform mode only

Remove 'single' mode handling from handle_timeout_flush. The code was
already correct (used expand_token), now it's unified under 'transform'.

Part of elspeth-rapid-3me: removing output_mode='single'

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
```

---

## Task 4: Remove "single" Code Path from _process_batch_aggregation_node

**Files:**
- Modify: `src/elspeth/engine/processor.py:870-922`

**Step 1: Identify the buggy code block**

Lines 870-922 contain the BUGGY "single" mode implementation that reuses the triggering token. This entire block must be deleted.

**Step 2: Delete the "single" mode block**

In `src/elspeth/engine/processor.py`, delete lines 870-922 (the entire `if output_mode == "single":` block including the `elif`).

The code should go directly from the output_modes handling comment to the `if output_mode == "passthrough":` block, then to `elif output_mode == "transform":`.

After deletion, the structure should be:
```python
            # Handle output modes
            if output_mode == "passthrough":
                # ... passthrough logic (unchanged)
            elif output_mode == "transform":
                # ... transform logic (unchanged)
            else:
                raise ValueError(f"Unknown output_mode: {output_mode}")
```

**Step 3: Run tests to verify transform mode works**

Run: `.venv/bin/python -m pytest tests/engine/test_processor_modes.py::TestProcessorTransformMode -v`
Expected: PASS

**Step 4: Commit**

```bash
git add src/elspeth/engine/processor.py
git commit -m "fix(processor): remove buggy 'single' mode from _process_batch_aggregation_node

This was the root cause of elspeth-rapid-nd3: the 'single' mode block
(lines 870-922) reused current_token.token_id instead of creating a
new output token via expand_token().

The 'transform' mode implementation correctly creates new tokens with
proper audit lineage. Removing 'single' entirely consolidates on the
correct behavior.

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
```

---

## Task 5: Update All Test Files to Use "transform" Mode

**Files to modify:**
- `tests/engine/test_orchestrator_audit.py:1165`
- `tests/engine/test_processor_modes.py:825, 929`
- `tests/engine/test_aggregation_integration.py:366, 525, 683, 1264, 1418, 1596, 2103, 2414, 2560`
- `tests/engine/test_processor_batch.py:554`
- `tests/engine/test_processor_coalesce.py:1397`
- `tests/engine/test_aggregation_audit.py:615`
- `tests/engine/test_coalesce_integration.py:1112`
- `tests/integration/test_aggregation_recovery.py:501`
- `tests/core/test_dag_schema_propagation.py:338`
- `tests/core/test_config_aggregation.py:147, 152, 162, 176`

**Step 1: Use sed to replace all occurrences**

```bash
# Replace output_mode="single" with output_mode="transform"
sed -i 's/output_mode="single"/output_mode="transform"/g' \
    tests/engine/test_orchestrator_audit.py \
    tests/engine/test_processor_modes.py \
    tests/engine/test_aggregation_integration.py \
    tests/engine/test_processor_batch.py \
    tests/engine/test_processor_coalesce.py \
    tests/engine/test_aggregation_audit.py \
    tests/engine/test_coalesce_integration.py \
    tests/integration/test_aggregation_recovery.py \
    tests/core/test_dag_schema_propagation.py

# Fix test_config_aggregation.py which has assertions about "single"
# This file tests config validation, needs special handling
```

**Step 2: Update test_config_aggregation.py manually**

In `tests/core/test_config_aggregation.py`, the tests assert that `output_mode == "single"`. These need to be changed to test "transform" as the default:

- Line 147: Change `output_mode="single"` to `output_mode="transform"`
- Line 152: Change `assert settings.output_mode == "single"` to `assert settings.output_mode == "transform"`
- Line 162: Change `output_mode="single"` to `output_mode="transform"`
- Line 176: Change `assert settings.output_mode == "single"` to `assert settings.output_mode == "transform"`

**Step 3: Run full test suite**

Run: `.venv/bin/python -m pytest tests/engine/test_aggregation*.py tests/engine/test_processor*.py tests/core/test_config*.py -v`
Expected: PASS (or identify remaining "single" references)

**Step 4: Commit**

```bash
git add tests/
git commit -m "test: update all tests from output_mode='single' to 'transform'

Mass migration of test files to use 'transform' mode instead of
removed 'single' mode.

Files updated:
- test_orchestrator_audit.py
- test_processor_modes.py
- test_aggregation_integration.py
- test_processor_batch.py
- test_processor_coalesce.py
- test_aggregation_audit.py
- test_coalesce_integration.py
- test_aggregation_recovery.py
- test_dag_schema_propagation.py
- test_config_aggregation.py

Part of elspeth-rapid-3me

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
```

---

## Task 6: Update Example Pipeline

**Files:**
- Modify: `examples/batch_aggregation/settings.yaml:28`

**Step 1: Update the example**

Change line 28 from:
```yaml
    output_mode: single  # Produce one summary row per batch
```
to:
```yaml
    output_mode: transform  # Produce one summary row per batch
    # Use expected_output_count: 1 to validate N→1 cardinality
```

**Step 2: Run the example to verify it works**

```bash
rm -rf examples/batch_aggregation/runs/
rm -f examples/batch_aggregation/output/*.csv
.venv/bin/python -m elspeth run -s examples/batch_aggregation/settings.yaml --execute
```
Expected: Pipeline completes successfully

**Step 3: Verify the audit trail is correct**

```bash
sqlite3 examples/batch_aggregation/runs/audit.db "
SELECT '=== Token Outcomes ===';
SELECT outcome, count(*) FROM token_outcomes GROUP BY outcome;

SELECT '';
SELECT '=== All batch members should be CONSUMED_IN_BATCH ===';
SELECT bm.ordinal, tok_out.outcome
FROM batch_members bm
JOIN token_outcomes tok_out ON bm.token_id = tok_out.token_id
ORDER BY bm.batch_id, bm.ordinal;
"
```
Expected: All batch members show `consumed_in_batch`, output tokens show `completed`

**Step 4: Commit**

```bash
git add examples/batch_aggregation/settings.yaml
git commit -m "docs(example): update batch_aggregation to use transform mode

Part of elspeth-rapid-3me: removing output_mode='single'

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
```

---

## Task 7: Add Identity-Based Regression Test

**Files:**
- Create: `tests/engine/test_batch_token_identity.py`

**Step 1: Write the identity verification test**

```python
"""Token identity tests for batch aggregation.

These tests verify that aggregation correctly creates NEW output tokens
instead of reusing input tokens. This catches bug elspeth-rapid-nd3.
"""

import pytest

from elspeth.contracts import NodeType
from elspeth.contracts.enums import RowOutcome
from elspeth.core.config import AggregationSettings, TriggerConfig
from elspeth.core.landscape import LandscapeDB, LandscapeRecorder
from elspeth.core.landscape.schema import DYNAMIC_SCHEMA
from elspeth.engine.processor import RowProcessor
from elspeth.engine.spans import SpanFactory
from elspeth.engine.tokens import NodeID
from elspeth.plugins.base import BaseTransform
from elspeth.plugins.context import PluginContext
from elspeth.plugins.results import TransformResult
from elspeth.contracts import Determinism

from tests.helpers.audit_assertions import (
    assert_all_batch_members_consumed,
    assert_output_token_distinct_from_inputs,
    assert_token_outcome,
)


class _TestSchema:
    fields = "dynamic"


class SumTransform(BaseTransform):
    """Sums values in a batch, outputs single aggregated row."""

    name = "summer"
    input_schema = _TestSchema
    output_schema = _TestSchema
    is_batch_aware = True
    creates_tokens = True  # Transform mode: creates new tokens
    determinism = Determinism.DETERMINISTIC
    plugin_version = "1.0"

    def __init__(self, node_id: str) -> None:
        super().__init__({"schema": {"fields": "dynamic"}})
        self.node_id = node_id

    def process(
        self, rows: list[dict] | dict, ctx: PluginContext
    ) -> TransformResult:
        if isinstance(rows, list):
            total = sum(r.get("value", 0) for r in rows)
            return TransformResult.success({"total": total}, success_reason={"action": "sum"})
        return TransformResult.success(rows, success_reason={"action": "passthrough"})


class TestBatchTokenIdentity:
    """Tests that batch aggregation creates distinct output tokens."""

    def test_all_batch_members_consumed_in_batch(self) -> None:
        """ALL tokens in a batch must have CONSUMED_IN_BATCH outcome.

        This is the core regression test for elspeth-rapid-nd3.
        The bug was that the triggering token (last in batch) got
        COMPLETED instead of CONSUMED_IN_BATCH.
        """
        db = LandscapeDB.in_memory()
        recorder = LandscapeRecorder(db)
        run = recorder.begin_run(config={}, canonical_version="v1")

        source_node = recorder.register_node(
            run_id=run.run_id,
            plugin_name="source",
            node_type=NodeType.SOURCE,
            plugin_version="1.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )
        agg_node = recorder.register_node(
            run_id=run.run_id,
            plugin_name="summer",
            node_type=NodeType.AGGREGATION,
            plugin_version="1.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )

        aggregation_settings = {
            NodeID(agg_node.node_id): AggregationSettings(
                name="batch_sum",
                plugin="summer",
                trigger=TriggerConfig(count=3),  # Batch of 3
                output_mode="transform",
            ),
        }

        processor = RowProcessor(
            recorder=recorder,
            span_factory=SpanFactory(),
            run_id=run.run_id,
            source_node_id=NodeID(source_node.node_id),
            aggregation_settings=aggregation_settings,
        )

        transform = SumTransform(agg_node.node_id)
        ctx = PluginContext(run_id=run.run_id, config={})

        # Process 3 rows to trigger batch flush
        all_results = []
        input_token_ids = []
        for i in range(3):
            results = processor.process_row(
                row_index=i,
                row_data={"value": (i + 1) * 10},  # 10, 20, 30
                transforms=[transform],
                ctx=ctx,
            )
            all_results.extend(results)
            # Collect token IDs as they're created
            for r in results:
                if r.outcome == RowOutcome.CONSUMED_IN_BATCH:
                    input_token_ids.append(r.token.token_id)

        # Get the batch_id from the recorder
        with recorder._db.connection() as conn:
            from sqlalchemy import select
            from elspeth.core.landscape.schema import batches_table

            batch = conn.execute(
                select(batches_table)
                .where(batches_table.c.run_id == run.run_id)
            ).fetchone()
            batch_id = batch.batch_id

        # CRITICAL ASSERTION: All batch members must be CONSUMED_IN_BATCH
        assert_all_batch_members_consumed(recorder, run.run_id, batch_id)

        # Get output token
        completed = [r for r in all_results if r.outcome == RowOutcome.COMPLETED]
        assert len(completed) == 1, f"Expected 1 COMPLETED, got {len(completed)}"
        output_token_id = completed[0].token.token_id

        # CRITICAL ASSERTION: Output token must be DISTINCT from inputs
        assert_output_token_distinct_from_inputs(output_token_id, input_token_ids)

        # Verify aggregation result
        assert completed[0].final_data["total"] == 60  # 10 + 20 + 30


    def test_triggering_token_not_reused(self) -> None:
        """The token that triggers the flush must NOT be reused as output.

        This specifically tests the bug scenario: the 3rd row triggers
        the batch flush. Its token_id must NOT appear in the output.
        """
        db = LandscapeDB.in_memory()
        recorder = LandscapeRecorder(db)
        run = recorder.begin_run(config={}, canonical_version="v1")

        source_node = recorder.register_node(
            run_id=run.run_id,
            plugin_name="source",
            node_type=NodeType.SOURCE,
            plugin_version="1.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )
        agg_node = recorder.register_node(
            run_id=run.run_id,
            plugin_name="summer",
            node_type=NodeType.AGGREGATION,
            plugin_version="1.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )

        aggregation_settings = {
            NodeID(agg_node.node_id): AggregationSettings(
                name="batch_sum",
                plugin="summer",
                trigger=TriggerConfig(count=2),  # Batch of 2
                output_mode="transform",
            ),
        }

        processor = RowProcessor(
            recorder=recorder,
            span_factory=SpanFactory(),
            run_id=run.run_id,
            source_node_id=NodeID(source_node.node_id),
            aggregation_settings=aggregation_settings,
        )

        transform = SumTransform(agg_node.node_id)
        ctx = PluginContext(run_id=run.run_id, config={})

        # Process row 0 - buffered, returns CONSUMED_IN_BATCH
        results_0 = processor.process_row(
            row_index=0,
            row_data={"value": 10},
            transforms=[transform],
            ctx=ctx,
        )
        assert len(results_0) == 1
        assert results_0[0].outcome == RowOutcome.CONSUMED_IN_BATCH
        first_token_id = results_0[0].token.token_id

        # Process row 1 - triggers flush
        results_1 = processor.process_row(
            row_index=1,
            row_data={"value": 20},
            transforms=[transform],
            ctx=ctx,
        )

        # Should have: CONSUMED_IN_BATCH (triggering token) + COMPLETED (output)
        consumed = [r for r in results_1 if r.outcome == RowOutcome.CONSUMED_IN_BATCH]
        completed = [r for r in results_1 if r.outcome == RowOutcome.COMPLETED]

        assert len(consumed) == 1, "Triggering token must be CONSUMED_IN_BATCH"
        assert len(completed) == 1, "Must have exactly 1 COMPLETED output"

        triggering_token_id = consumed[0].token.token_id
        output_token_id = completed[0].token.token_id

        # THE BUG: In the old code, output_token_id == triggering_token_id
        # THE FIX: They must be different
        assert output_token_id != triggering_token_id, (
            f"Output token {output_token_id} should NOT equal triggering token! "
            f"This is the elspeth-rapid-nd3 bug: token reuse breaks audit lineage."
        )
        assert output_token_id != first_token_id, (
            "Output token should not equal first buffered token either"
        )
```

**Step 2: Run the new tests**

Run: `.venv/bin/python -m pytest tests/engine/test_batch_token_identity.py -v`
Expected: PASS (if the processor changes are correct)

**Step 3: Commit**

```bash
git add tests/engine/test_batch_token_identity.py
git commit -m "test: add identity-based regression tests for batch aggregation

These tests specifically catch bug elspeth-rapid-nd3:
- test_all_batch_members_consumed_in_batch: verifies ALL tokens consumed
- test_triggering_token_not_reused: verifies output is a NEW token

Uses the new audit assertion helpers from tests/helpers/audit_assertions.py

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
```

---

## Task 8: Run Full Test Suite and Fix Any Remaining Issues

**Step 1: Run full test suite**

```bash
.venv/bin/python -m pytest tests/ -v --tb=short -x
```

**Step 2: Fix any remaining "single" references**

If tests fail due to remaining "single" references, use:
```bash
grep -rn '"single"' tests/ src/ --include="*.py" | grep -v test_config_single_rejected
```

**Step 3: Run mypy**

```bash
.venv/bin/python -m mypy src/elspeth/engine/processor.py src/elspeth/core/config.py
```

**Step 4: Run ruff**

```bash
.venv/bin/python -m ruff check src/elspeth/engine/processor.py src/elspeth/core/config.py --fix
```

**Step 5: Final commit**

```bash
git add -A
git commit -m "fix: complete removal of output_mode='single' from batch aggregation

Fixes elspeth-rapid-nd3: Batch aggregation now correctly creates new
output tokens via expand_token() instead of reusing the triggering token.

Summary of changes:
- Removed 'single' from Literal type in AggregationSettings
- Changed default output_mode from 'single' to 'transform'
- Added Pydantic validator rejecting 'single' with migration hint
- Added optional expected_output_count parameter for N→1 validation
- Deleted buggy 'single' code path in processor.py
- Updated all tests to use 'transform' mode
- Added identity-based test assertions
- Added regression tests for token identity

BREAKING CHANGE: output_mode='single' is removed.
Migration: Use output_mode='transform' instead.

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
```

---

## Task 9: Update Beads Issues

**Step 1: Close the dependent issues**

```bash
bd close elspeth-rapid-3me --reason="Single mode removed, transform mode is now default"
bd close elspeth-rapid-360 --reason="Identity-based test assertions added"
bd close elspeth-rapid-nd3 --reason="Bug fixed by removing single mode and adding identity tests"
```

**Step 2: Sync beads**

```bash
bd sync
```

---

## Verification Checklist

Before considering this complete:

- [ ] `output_mode="single"` rejected by Pydantic with helpful error
- [ ] `output_mode="transform"` is the new default
- [ ] All batch members have `CONSUMED_IN_BATCH` outcome
- [ ] Output token has DIFFERENT token_id from all inputs
- [ ] `explain()` traces output to ALL input rows (not just triggering row)
- [ ] Full test suite passes
- [ ] mypy passes
- [ ] ruff passes
- [ ] Example pipeline works correctly
