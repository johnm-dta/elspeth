# Bug Report: Gate route destination validation assumes “identifier” sink names, but sink keys are not validated

## Summary

- `GateSettings.validate_routes()` requires sink destinations to match `_IDENTIFIER_PATTERN` (letters/underscores, then alphanumerics/underscores).
- `ElspethSettings.sinks` keys (sink names) are not validated against the same pattern; `output_sink` can point to any sink key.
- Result: configurations can legally define sinks with names that work for `output_sink`, but gates cannot route to them (even though the sink exists), creating an inconsistent and surprising config constraint.

## Severity

- Severity: minor
- Priority: P3

## Reporter

- Name or handle: codex
- Date: 2026-01-19
- Related run/issue ID: N/A

## Environment

- Commit/branch: `8cfebea78be241825dd7487fed3773d89f2d7079` (local)
- OS: Linux (Ubuntu kernel 6.8.0-90-generic)
- Python version: 3.13.1
- Config profile / env vars: N/A
- Data set or fixture: N/A

## Agent Context (if relevant)

- Goal or task prompt: deep dive into subsystem 3 (core infrastructure) and create bug tickets
- Model/version: GPT-5.2 (Codex CLI)
- Tooling and permissions (sandbox/approvals): workspace-write, network restricted, approvals on-request
- Determinism details (seed, run ID): N/A
- Notable tool calls or steps: static inspection of `src/elspeth/core/config.py`

## Steps To Reproduce

1. Create a settings YAML with a sink name that is not a valid identifier (e.g., `output-sink`):
   - `sinks: { "output-sink": { plugin: csv } }`
   - `output_sink: "output-sink"` (passes `output_sink` existence check)
2. Add a gate that routes to that sink:
   - `routes: { "true": "output-sink", "false": "continue" }`
3. Call `load_settings()` and observe validation error in `GateSettings.validate_routes()` for the destination name.

## Expected Behavior

One of:

- Sink names are restricted to identifier-like values everywhere (validate sink keys up-front), or
- Gate routing accepts any sink key name that exists in `sinks` (and does not impose a stricter unrelated regex).

## Actual Behavior

- Sink keys are effectively unrestricted, but gate routes require identifier-style sink names, so some valid sink names become “unroutable by gates.”

## Evidence

- Destination identifier regex:
  - `src/elspeth/core/config.py:15-16` (`_IDENTIFIER_PATTERN`)
- Gate route validation requires destination match:
  - `src/elspeth/core/config.py:206-232` (`validate_routes`, `_IDENTIFIER_PATTERN.match(destination)`)
- Sinks keys are only validated for non-emptiness, not naming:
  - `src/elspeth/core/config.py:635-737` (`sinks: dict[str, SinkSettings]` + `validate_sinks_not_empty`)

## Impact

- User-facing impact: confusing config errors; sink names that work for `output_sink` cannot be used as route destinations.
- Data integrity / security impact: low.
- Performance or cost impact: N/A.

## Root Cause Hypothesis

- Gate routing validation was implemented before (or independently of) a system-wide decision about permissible sink naming.

## Proposed Fix

- Code changes (modules/files):
  - Option A (restrict sinks): validate `ElspethSettings.sinks` keys (and `output_sink`) against the same `_IDENTIFIER_PATTERN`, so naming rules are consistent.
  - Option B (loosen gate validation): remove `_IDENTIFIER_PATTERN` enforcement in `GateSettings.validate_routes()` and instead validate route destinations against the actual `sinks` keys during graph compilation (`ExecutionGraph.from_config` already checks existence).
- Tests to add/update:
  - Add a config test that demonstrates consistent behavior (either rejection of invalid sink names early, or support for non-identifier sink names in gates).
- Risks or migration steps:
  - Option A is breaking for existing configs with non-identifier sink names.
  - Option B requires ensuring downstream code can handle arbitrary sink names safely (they’re used in node IDs and maps).

## Architectural Deviations

- Spec or doc reference: N/A (naming contract)
- Observed divergence: gate destination validation imposes stronger constraints than sink definition.
- Reason (if known): implicit assumption that sink names are identifiers.
- Alignment plan or decision needed: define and document sink naming rules explicitly.

## Acceptance Criteria

- Sink naming rules and gate destination validation are consistent and predictable.

## Tests

- Suggested tests to run: `.venv/bin/python -m pytest tests/core/test_config.py`
- New tests required: yes (sink naming + gate routing consistency)

## Notes / Links

- Related issues/PRs: N/A

## Resolution

**Status:** FIXED (2026-01-21)

**Solution Applied:** Option B - removed `_IDENTIFIER_PATTERN` enforcement from `GateSettings.validate_routes()`.

**Changes Made:**
1. `src/elspeth/core/config.py`: Removed the identifier pattern check from `validate_routes()` and deleted the now-unused `_IDENTIFIER_PATTERN` constant (per CLAUDE.md "No Legacy Code Policy"). The DAG builder (`ExecutionGraph.from_config` at `dag.py:388-392`) already validates that route destinations exist as actual sink keys, providing better error messages that reference available sinks.

2. `tests/core/test_config.py`: Updated tests:
   - Replaced `test_gate_settings_invalid_route_destination` with `test_gate_settings_hyphenated_sink_destination_accepted`
   - Replaced `test_gate_settings_route_destination_special_chars` with `test_gate_settings_numeric_prefix_sink_destination_accepted`

3. `tests/core/test_dag.py`: Added integration test `test_hyphenated_sink_names_work_in_dag` to verify end-to-end flow with hyphenated sink names.

**Rationale:**
- Sink names are dict keys, not Python identifiers - no reason to restrict them
- The DAG builder already validates sink existence with helpful error messages
- Option A would have been breaking for existing configs with non-identifier sink names

**Verification:**
- All 200 tests in `test_config.py` and `test_dag.py` pass
- Manual verification confirms hyphenated sink names now work with gates
