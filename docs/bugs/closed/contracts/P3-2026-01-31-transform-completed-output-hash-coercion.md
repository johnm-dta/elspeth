# Bug Report: TransformCompleted telemetry requires output_hash even when no output exists

## Summary

- `TransformCompleted.output_hash` is `str` not `str | None`, forcing coercion to empty string for failed transforms.

## Severity

- Severity: minor
- Priority: P3

## Reporter

- Name or handle: Codex (triaged by Claude)
- Date: 2026-01-31

## Evidence

- `src/elspeth/contracts/events.py:167` - `output_hash: str` (not optional)
- `src/elspeth/engine/processor.py:218` - `output_hash=transform_result.output_hash or ""`
- Failed transforms appear to have output hash (empty string)

## Proposed Fix

- Change to `output_hash: str | None`

## Acceptance Criteria

- Failed transforms have `output_hash=None`, not empty string

## Verification (2026-02-01)

**Status: STILL VALID**

- `TransformCompleted.output_hash` is still a non-optional `str`. (`src/elspeth/contracts/events.py:156-167`)
- Telemetry emission still coerces missing hashes to empty string. (`src/elspeth/engine/processor.py:210-221`)
