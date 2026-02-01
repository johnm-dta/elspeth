# Bug Report: LLM usage report misclassifies call status

## Summary

- LLM usage report checks `if row.status == "completed"` but actual stored value is `"success"` per `CallStatus` enum. All successful calls are misclassified as failures.

## Severity

- Severity: moderate
- Priority: P2

## Reporter

- Name or handle: Codex (triaged by Claude)
- Date: 2026-01-31

## Evidence

- `src/elspeth/mcp/server.py:987` - `if row.status == "completed":`
- `src/elspeth/contracts/enums.py:203-204` - `CallStatus.SUCCESS = "success"`
- 100% of successful calls misclassified

## Impact

- User-facing impact: Completely broken LLM usage reporting
- Data integrity: None (MCP is debugging tool)

## Proposed Fix

- Change to `if row.status == "success":` or use `CallStatus.SUCCESS.value`

## Acceptance Criteria

- Successful calls correctly counted as successful

## Verification (2026-02-01)

**Status: STILL VALID**

- LLM usage still compares `row.status` to `"completed"` while `CallStatus.SUCCESS` is `"success"`. (`src/elspeth/mcp/server.py:987`, `src/elspeth/contracts/enums.py:203-204`)

## Resolution (2026-02-02)

**Status: FIXED**

- Changed comparison to use `CallStatus.SUCCESS.value` instead of string literal `"completed"`
- Added `CallStatus` import from `elspeth.contracts.enums`
- Fix at `src/elspeth/mcp/server.py:988`
