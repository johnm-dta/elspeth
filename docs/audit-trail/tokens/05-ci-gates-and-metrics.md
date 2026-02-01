# CI Gates and Metrics for Token Outcomes

This document defines quality gates and metrics that enforce the token
outcome contract in CI and on completed runs.

## Required CI gates

1. Audit sweep passes (no missing terminal outcomes).
2. Required fields present for all outcomes.
3. Sink node_state <-> COMPLETED outcome consistency.

## Suggested CI stages

- PR:
  - Unit tests
  - Integration tests with audit sweep assertions
  - Property-based tests (small sample)

- Main:
  - Full unit + integration
  - Expanded property-based sample

- Nightly:
  - Stress property-based tests
  - Any slow end-to-end pipelines

## Metrics to track

- Outcome gap rate:
  - missing_terminal_outcomes / total_tokens
  - Target: 0

- Required-field violations:
  - count of outcomes with missing required fields
  - Target: 0

- Sink mismatch count:
  - completed outcomes without sink node_state
  - sink node_states without COMPLETED outcome
  - Target: 0

- Test flakiness rate:
  - Target: < 1 percent

## Enforcement rules

- Any non-zero outcome gap rate fails CI.
- Any missing required field fails CI.
- Sink mismatch failures block merges.

## Notes

- If a run is still in progress, BUFFERED outcomes are expected.
  Only enforce these checks after run completion.
- Keep property-based tests bounded on PRs to preserve fast feedback.
