# Bug Report: Routing Reason Payloads Never Persisted (reason_ref Always None)

## Summary

- Routing decisions record only a hash; the actual routing reason payload is never persisted to the payload store, and multi-destination routing hard-codes `reason_ref=None`, so audit/export cannot reconstruct gate reasoning.

## Severity

- Severity: major
- Priority: P1

## Reporter

- Name or handle: Codex (triaged by Claude)
- Date: 2026-01-31
- Related run/issue ID: N/A

## Evidence

- `src/elspeth/core/landscape/recorder.py:1157` - record_routing_event computes `reason_hash` but never persists payload
- `src/elspeth/core/landscape/recorder.py:1217` - record_routing_events lacks a `reason_ref` parameter
- `src/elspeth/core/landscape/recorder.py:1251` - record_routing_events hard-codes `reason_ref=None` in the RoutingEvent
- `src/elspeth/engine/executors.py:854` - engine passes `reason` but no `reason_ref`

## Impact

- User-facing impact: Routing explanations and audit exports lack decision reason payloads even when provided
- Data integrity / security impact: Audit trail violates "no inference" for routing decisions; reason content cannot be reconstructed
- Performance or cost impact: May increase audit/debugging time

## Root Cause Hypothesis

- `LandscapeRecorder.record_routing_event(s)` never persist routing reason payloads to the payload store and do not populate `reason_ref`, despite schema support.

## Proposed Fix

- Code changes:
  - Add payload persistence in `record_routing_event` when `reason` is provided and `reason_ref` is not
  - Add `reason_ref` support to `record_routing_events`, storing once per routing group and inserting `reason_ref` into DB rows
- Tests to add/update:
  - Unit test verifying `reason_ref` is set and payload exists for single routing
  - Unit test verifying `reason_ref` is set for multi-destination routing and is shared across group

## Acceptance Criteria

- Routing events with a non-empty `reason` record a non-null `reason_ref`
- Payload store contains a canonical JSON payload for the routing reason
- Export can include routing reason payloads
