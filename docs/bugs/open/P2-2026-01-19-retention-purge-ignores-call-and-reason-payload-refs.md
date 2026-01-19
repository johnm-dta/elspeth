# Bug Report: Retention purge only targets row payloads; call/reason payloads are never purged

## Summary

- `PurgeManager.find_expired_row_payloads()` only returns `rows.source_data_ref` for completed runs past retention.
- The Landscape schema also stores payload refs for:
  - external calls (`calls.request_ref`, `calls.response_ref`)
  - routing reasons (`routing_events.reason_ref`)
- Result: “call payloads” and “reason payloads” are not subject to retention, contradicting the retention strategy that explicitly calls out both row and call payloads.

## Severity

- Severity: major
- Priority: P1

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
- Notable tool calls or steps: static inspection of `src/elspeth/core/retention/purge.py` + schema

## Steps To Reproduce

1. Run a pipeline that records at least one external call with payload refs (request/response) and/or routing events with `reason_ref`.
2. Set the run status to completed and ensure `completed_at` is older than your retention cutoff.
3. Call `PurgeManager.find_expired_row_payloads(retention_days=...)` and then `purge_payloads(refs)`.
4. Observe only `rows.source_data_ref` blobs are eligible; call/reason blobs remain indefinitely.

## Expected Behavior

- Retention purge identifies *all* purge-eligible payload refs tied to expired completed runs, including:
  - row payloads
  - call request/response payloads
  - routing reason payloads

## Actual Behavior

- Only `rows.source_data_ref` is queried for purge eligibility; call/reason refs are not queried and therefore are never purged.

## Evidence

- Purge query only targets `rows.source_data_ref`:
  - `src/elspeth/core/retention/purge.py:71-114`
- Schema stores additional payload refs not covered by purge:
  - `src/elspeth/core/landscape/schema.py:162-178` (`calls.request_ref`, `calls.response_ref`)
  - `src/elspeth/core/landscape/schema.py:204-217` (`routing_events.reason_ref`)
- Design explicitly calls out call payload retention:
  - `docs/design/architecture.md:571-579` (Row payloads and Call payloads: purge, keep hash)

## Impact

- User-facing impact: storage usage grows without bound for external call and routing reason payloads; operators cannot rely on documented retention semantics.
- Data integrity / security impact: elevated. Retention policy is part of data governance; keeping call payloads longer than intended can violate compliance expectations.
- Performance or cost impact: potentially high storage costs; slower backups/exports.

## Root Cause Hypothesis

- Retention work was implemented for the initial row payload use case only (`rows.source_data_ref`), and additional payload-bearing tables were not integrated into the purge discovery query.

## Proposed Fix

- Code changes (modules/files):
  - `src/elspeth/core/retention/purge.py`:
    - Replace `find_expired_row_payloads()` with something like `find_expired_payload_refs()` that returns a deduplicated union of:
      - `rows.source_data_ref`
      - `calls.request_ref`
      - `calls.response_ref`
      - `routing_events.reason_ref`
    - Ensure each query is constrained to completed runs past cutoff via `runs.completed_at`.
- Config or schema changes:
  - Consider splitting retention days per payload type if the design requires (`row_payloads_days` vs `call_payloads_days`), or explicitly document that `PayloadStoreSettings.retention_days` applies to all payload refs.
- Tests to add/update:
  - Add tests that insert rows + calls + routing events with refs for an expired run and assert all refs are returned and purged.
- Risks or migration steps:
  - Purging additional refs may delete payloads that some workflows assumed were kept indefinitely; confirm policy alignment.

## Architectural Deviations

- Spec or doc reference: `docs/design/architecture.md:569-580`
- Observed divergence: call payloads and routing reason payloads are not included in retention purge logic.
- Reason (if known): initial implementation targeted only row payloads.
- Alignment plan or decision needed: confirm whether *all* payload ref columns are subject to retention under a unified policy.

## Acceptance Criteria

- Purge discovery returns refs from rows, calls, and routing reasons for eligible runs.
- Purge deletes all referenced blobs that exist and reports failures clearly.
- Tests demonstrate that call payload refs and reason refs are purged.

## Tests

- Suggested tests to run: `.venv/bin/python -m pytest tests/`
- New tests required: yes (retention coverage for calls + routing reasons)

## Notes / Links

- Related issues/PRs: N/A
- Related design docs: `docs/design/architecture.md` (payload retention)
