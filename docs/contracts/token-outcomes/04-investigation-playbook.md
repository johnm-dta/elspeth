# Token Outcome Gap Investigation Playbook

Use this playbook when tokens are missing terminal outcomes or outcomes
are incorrect. The goal is fast isolation, reproducible failures, and
permanent fixes.

## Inputs to collect

- run_id
- pipeline config used in the run
- any error logs or stack traces
- whether the run completed or failed

## Step 1: Run the audit sweep

Run all queries in `docs/contracts/token-outcomes/02-audit-sweep.md`.
Save results grouped by outcome type.

## Step 2: Classify the gap

For each failing token group, classify by symptom:
- Missing terminal outcome
- Missing required fields
- Sink node_state mismatch
- Parent link missing

## Step 3: Map to path

Use `docs/contracts/token-outcomes/01-outcome-path-map.md` to locate the responsible
code path. This gives you the exact file and function to inspect.

## Step 4: Identify the minimal reproduction

Create the smallest pipeline that triggers the gap. Examples:
- Single gate route to sink for ROUTED gaps
- Fork + coalesce with 2 branches for COALESCED gaps
- Aggregation passthrough for BUFFERED gaps

Prefer in-memory LandscapeDB and minimal plugin stubs.

## Step 5: Add a failing test

Write a regression test that:
- Reproduces the gap
- Runs the audit sweep on the test run
- Fails on current behavior

Place tests in the appropriate tier:
- Unit for narrow path logic
- Integration for full pipeline behaviors

## Step 6: Fix and re-verify

- Fix the code path (no patches or temporary workarounds)
- Re-run the regression test
- Re-run the audit sweep on the failing run_id if possible

## Escalation rules

Escalate immediately if:
- Terminal outcomes are missing after run completion
- Sink node_state and COMPLETED outcomes disagree
- Duplicate terminal outcomes appear

## Closure checklist

- [ ] Gap reproduced with a test
- [ ] Fix applied and test passes
- [ ] Audit sweep passes on the reproduction
- [ ] Outcome path map updated if behavior changed
