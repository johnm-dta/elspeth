# Bug Report: Source quarantine routing is not end-to-end (`route_to_sink` is a stub; invalid rows are dropped)

## Summary

- Sources record validation failures and attempt to route invalid external rows to a quarantine sink via `PluginContext.route_to_sink()`, but `route_to_sink()` is a Phase 2 stub that only logs; sources also do not `yield` invalid rows, so the quarantine sink never receives them.

## Severity

- Severity: major
- Priority: P1

## Reporter

- Name or handle: codex
- Date: 2026-01-19
- Related run/issue ID: N/A

## Environment

- Commit/branch: `main` @ `25468ac9550b481a55b81a05d84bbf2592e6430c`
- OS: Linux (Ubuntu 6.8.0-90-generic)
- Python version: 3.13.1
- Config profile / env vars: N/A (static analysis)
- Data set or fixture: N/A (static analysis)

## Agent Context (if relevant)

- Goal or task prompt: deep dive into subsystems, identify hotspots, write bug reports
- Model/version: GPT-5.2 (Codex CLI)
- Tooling and permissions (sandbox/approvals): sandbox read-only, network restricted
- Determinism details (seed, run ID): N/A
- Notable tool calls or steps: inspected source plugins, plugin context routing hooks, and orchestrator source loop

## Steps To Reproduce

1. Configure a source with a strict schema (e.g., CSV) and set `on_validation_failure` to a sink name (not `"discard"`).
2. Provide input containing at least one invalid row.
3. Run the pipeline.
4. Observe the invalid row is neither processed through the engine nor written to the configured quarantine sink.

## Expected Behavior

- When `on_validation_failure` names a sink, invalid source rows are delivered to that sink as artifacts (and reflected in run metrics/audit trail).

## Actual Behavior

- Invalid rows are recorded as validation errors, but are not yielded to the orchestrator, and `route_to_sink()` only logs; quarantine routing does not reach sinks.

## Evidence

- `PluginContext.route_to_sink()` is explicitly a stub and only logs: `src/elspeth/plugins/context.py:225-248`
- CSV source quarantines invalid rows and `continue`s (no `yield`) after calling `route_to_sink()`: `src/elspeth/plugins/sources/csv_source.py:110-133`
- JSON source behaves the same (no `yield` on invalid rows): `src/elspeth/plugins/sources/json_source.py:143-165`
- Orchestrator only processes what `source.load(ctx)` yields: `src/elspeth/engine/orchestrator.py:571`

## Impact

- User-facing impact: configured quarantine sinks (e.g., “invalid_rows.csv”) never receive invalid rows.
- Data integrity / security impact: violates “no silent drops” expectations for observable quarantine outcomes (rows disappear from outputs even when configured to route).
- Performance or cost impact: operators may rerun or add external validation tooling to recover missing quarantine artifacts.

## Root Cause Hypothesis

- Source-level quarantine is implemented as a context hook (`route_to_sink`) that is not wired into the engine’s sink execution path; source plugins skip invalid rows instead of emitting a first-class “quarantined row” for orchestration/routing.

## Proposed Fix

- Code changes (modules/files):
  - Option A (recommended): make source quarantine a first-class engine outcome (e.g., emit a structured quarantined token/result from sources) so the orchestrator can route it through the normal sink executor path.
  - Option B: wire `PluginContext.route_to_sink()` into an orchestrator-managed sink buffer/callback used during `source.load(ctx)`.
  - Option C: if quarantined inputs must be part of the run graph, persist them as rows/tokens (or a dedicated quarantine table) and route them consistently.
- Config or schema changes:
  - Possibly add/clarify a “quarantine row schema” or dedicated sink contract for validation failures.
- Tests to add/update:
  - Add an integration test where `on_validation_failure` targets a sink and verify the sink receives the invalid row.
  - Add a test for `"discard"` to ensure invalid rows are intentionally dropped (and still auditable via `validation_errors`).
- Risks or migration steps:
  - Defining whether quarantined inputs become first-class rows/tokens affects metrics, checkpointing, and “terminal state” semantics.

## Architectural Deviations

- Spec or doc reference: `CLAUDE.md` (Tier 3 external data: “quarantine rows that can’t be coerced/validated”)
- Observed divergence: quarantine is recorded but the configured quarantine sink does not receive the row; output artifacts do not match operator expectations.
- Reason (if known): routing-to-sink hook intentionally left stubbed for Phase 2.
- Alignment plan or decision needed: choose the canonical quarantine mechanism (engine outcome vs direct sink routing) and document it.

## Acceptance Criteria

- When `on_validation_failure != "discard"`, invalid source rows are actually written to the configured sink.
- Quarantines are visible in run counts/status (either as a first-class terminal outcome or via a clearly defined quarantine counter).
- Audit trail and outputs agree: quarantined inputs have a durable record and an artifact path when routed to a sink.

## Tests

- Suggested tests to run: `pytest tests/`
- New tests required: yes (source quarantine routing integration test)

## Notes / Links

- Related docs: `src/elspeth/plugins/config_base.py` (for `on_validation_failure` config contract)
