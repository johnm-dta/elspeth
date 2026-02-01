# Bug Report: COMPLETED outcomes missing sink_name for disambiguation

## Summary

- `explain(row_id, sink='X')` disambiguation feature doesn't work for COMPLETED outcomes because `sink_name` is not recorded.
- Processor records COMPLETED outcomes without `sink_name` because it doesn't know the routing destination.
- Only orchestrator knows the branch→sink mapping after routing decisions are made.
- Design doc (AUD-001) specifies COMPLETED outcomes SHOULD have `sink_name` populated, but implementation was incomplete.

## Severity

- Severity: medium (feature doesn't work, but has workaround: use token_id)
- Priority: P2

## Reporter

- Name or handle: john (via bug triage comment in lineage.py:108-113)
- Date: 2026-01-24
- Related run/issue ID: AUD-001 design implementation

## Environment

- Commit/branch: `fix/rc1-bug-burndown-session-4` @ `f97789d`
- OS: Linux 6.8.0-90-generic
- Python version: 3.12+
- Config profile / env vars: N/A
- Data set or fixture: any pipeline with forked paths reaching multiple sinks

## Agent Context (if relevant)

- Goal or task prompt: systematic debugging of COMPLETED outcomes missing sink_name bug
- Model/version: Claude Sonnet 4.5
- Tooling and permissions: full workspace access
- Determinism details: N/A (static analysis + code review)
- Notable tool calls: systematic investigation using architecture critic and code review agents

## Steps To Reproduce

1. Create a pipeline with a gate that forks to multiple paths
2. Each forked path completes normally and routes to different sinks (e.g., "approved", "rejected")
3. Run the pipeline and record outcomes
4. Query lineage: `explain(recorder, run_id, row_id="X", sink="approved")`

## Expected Behavior

- `explain(row_id, sink='approved')` returns the lineage for the token that reached the "approved" sink
- The disambiguation filter `[o for o in terminal_outcomes if o.sink_name == sink]` finds matching outcomes

## Actual Behavior

- `explain(row_id, sink='approved')` returns None because no COMPLETED outcomes have `sink_name` set
- All COMPLETED outcomes have `sink_name=None` in the database
- The filter returns an empty list, so explain() returns None

## Evidence

- **Lineage query filtering**: `src/elspeth/core/landscape/lineage.py:108-110` filters by `sink_name`
- **Processor records without sink**: `src/elspeth/engine/processor.py:1003-1007` calls `record_token_outcome(outcome=COMPLETED)` with no `sink_name`
- **Orchestrator determines sink**: `src/elspeth/engine/orchestrator.py:954-956` determines `sink_name` from branch mapping AFTER outcome already recorded
- **Design intent**: `docs/plans/completed/2026-01-21-AUD-001-token-outcomes-design.md:92` shows COMPLETED should have `sink_name`

**Additional evidence from systematic investigation:**
- Found FOUR locations in processor.py recording COMPLETED without sink_name:
  1. Line 236-240: Aggregation single output mode
  2. Line 300-304: Aggregation passthrough mode
  3. Line 369-373: Aggregation transform mode
  4. Line 1003-1007: Normal transform completion

## Impact

- User-facing impact: `explain(row_id, sink='X')` disambiguation doesn't work for normal completions, only for gate-routed outcomes
- Data integrity impact: audit trail is incomplete - can't determine which sink a COMPLETED token reached without scanning routing events
- Performance or cost impact: users must fall back to `explain(token_id=...)` which requires knowing token IDs upfront

## Root Cause Hypothesis

**Temporal gap between outcome recording and sink determination:**
- Processor records COMPLETED at end of `_process_single_token()` when sink is unknown
- Orchestrator determines sink later based on `branch_name` and routing config
- Information asymmetry: processor knows row outcomes, orchestrator knows sink routing

## Proposed Fix

- Code changes (modules/files):
  - Remove all `record_token_outcome(outcome=COMPLETED)` calls from `processor.py` (4 locations)
  - Add `record_token_outcome(outcome=COMPLETED, sink_name=sink_name)` to `orchestrator.py` after sink determination (4 locations: main path, resume path, aggregation flush single, aggregation flush multi)
- Config or schema changes: none
- Tests to add/update:
  - Verify existing tests still pass (they should - this adds data, doesn't change behavior)
  - Consider adding test that validates `explain(row_id, sink='X')` works for COMPLETED outcomes
- Risks or migration steps:
  - Low risk - adding data to audit trail, not changing behavior
  - All COMPLETED outcomes in new runs will have sink_name populated

## Architectural Deviations

- Spec or doc reference: `docs/plans/completed/2026-01-21-AUD-001-token-outcomes-design.md` (Column Usage table line 92)
- Observed divergence: COMPLETED outcomes recorded without sink_name despite design specifying it should be populated
- Reason (if known): implementation oversight - sink determination happens in orchestrator, but outcome recording was left in processor
- Alignment plan: move COMPLETED outcome recording from processor to orchestrator where sink context exists

## Acceptance Criteria

- All COMPLETED outcomes have `sink_name` populated in token_outcomes table
- `explain(row_id, sink='X')` successfully disambiguates COMPLETED tokens in forked pipelines
- All existing tests pass
- No regression in audit trail integrity

## Tests

- Suggested tests to run: `pytest tests/engine/test_processor_outcomes.py tests/core/test_token_outcomes.py -v`
- New tests required: no (existing tests cover the API, just adding data to existing outcomes)

## Notes / Links

- Related issues/PRs: AUD-001 token outcomes implementation
- Related design docs: `docs/plans/completed/2026-01-21-AUD-001-token-outcomes-design.md`
- Expert review: architecture critic and code review agents both confirmed fix approach

---

## Resolution

**Status:** RESOLVED
**Fixed in:** 2026-01-24
**Commit:** (pending)

**Root Cause:**
Outcome recording responsibility was split incorrectly:
- Processor recorded COMPLETED outcomes at end of row processing, but doesn't know routing destination
- Orchestrator determines sink routing based on branch_name mapping, but doesn't record outcomes
- This created a temporal gap: outcome recorded before sink determination happens

**Fix Applied:**
Moved COMPLETED outcome recording from processor to orchestrator (8 changes total):

**Processor deletions (4 locations):**
1. Line ~236: Aggregation single output mode
2. Line ~296: Aggregation passthrough mode
3. Line ~361: Aggregation transform mode
4. Line ~988: Normal transform completion

**Orchestrator additions (4 locations):**
1. Lines 958-965: Main execution path (after sink determination)
2. Lines 1573-1580: Resume/checkpoint path
3. Lines 1810-1816: Aggregation flush - single output
4. Lines 1836-1842: Aggregation flush - multiple output

Each addition properly sets `sink_name` using the orchestrator's routing logic (branch_name → sink mapping).

**Test Coverage:**
- All existing engine tests pass (94 tests)
- Changes maintain backward compatibility (adds data, doesn't change behavior)
- Audit trail integrity verified by code review agent

**Separation of Concerns:**
- Processor handles row-level transformations and routing decisions (what happened)
- Orchestrator handles sink assignment and destination mapping (where it goes)
- COMPLETED is a terminal outcome that requires sink context, which only orchestrator has
- ROUTED outcomes continue to be recorded in processor (gates know the sink)

**Bonus Finding:**
Code review agent discovered a separate potential issue: tokens merged via `coalesce_executor.flush_pending()` at end-of-source don't have their COALESCED outcome recorded, unlike coalesce operations during normal processing. This may warrant a separate bug ticket for audit completeness.
