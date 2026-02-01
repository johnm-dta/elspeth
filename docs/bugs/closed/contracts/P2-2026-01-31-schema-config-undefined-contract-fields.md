# Bug Report: Explicit Schema Contracts Allow Undefined Fields

## Summary

- `SchemaConfig.from_dict()` parses `guaranteed_fields`, `required_fields`, `audit_fields` without validating they're subsets of declared fields. Typos in contract fields become "guaranteed" without error.

## Severity

- Severity: moderate
- Priority: P2

## Reporter

- Name or handle: Codex (triaged by Claude)
- Date: 2026-01-31

## Evidence

- `src/elspeth/contracts/schema.py:283-351` - `from_dict()` parses `guaranteed_fields` / `required_fields` / `audit_fields` with no subset validation.
- `src/elspeth/contracts/schema.py:389-409` - `get_effective_guaranteed_fields()` unions explicit guarantees with declared required fields, so typos become “guaranteed.”
- Typos in `guaranteed_fields` create impossible contracts

## Impact

- User-facing impact: DAG validation passes with invalid contract claims
- Data integrity: Audit claims fields exist that don't

## Proposed Fix

- Add subset validation: all contract fields must exist in declared fields

## Acceptance Criteria

- Undefined field names in contracts raise ValueError at config load time

## Verification (2026-02-01)

**Status: STILL VALID**

- Contract fields are still accepted without checking they exist in declared fields. (`src/elspeth/contracts/schema.py:283-351`, `src/elspeth/contracts/schema.py:389-409`)
