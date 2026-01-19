# Bug Report: Review non-trust-boundary defensive patterns for whitelist

## Summary

- Inventory and review internal uses of `.get()`, `hasattr()`, `getattr()`, and `isinstance()` that are not at trust boundaries, and decide whether to whitelist or refactor them.

## Severity

- Severity: trivial
- Priority: P3

## Reporter

- Name or handle: codex
- Date: 2026-01-15
- Related run/issue ID: N/A

## Environment

- Commit/branch: not checked
- OS: not checked
- Python version: not checked
- Config profile / env vars: not checked
- Data set or fixture: not checked

## Agent Context (if relevant)

- Goal or task prompt: compile non-trust-boundary defensive usage list for review
- Model/version: GPT-5 (Codex)
- Tooling and permissions (sandbox/approvals): sandbox read-only, network restricted, approvals on-request
- Determinism details (seed, run ID): N/A
- Notable tool calls or steps: `rg -n "getattr|hasattr|isinstance|.get(" src`

## Steps To Reproduce

1. Review the listed call sites in Evidence.
2. Classify each as trust-boundary (allowed) or internal invariant (should be refactored).
3. Update code or whitelist documentation accordingly.

## Expected Behavior

- Internal invariants use direct access and explicit errors; only trust-boundary logic uses defensive patterns.

## Actual Behavior

- Internal code uses `.get()` and similar patterns without an explicit trust-boundary justification.

## Evidence

**⚠️ STALE DATA (2026-01-19 Triage Note)**
The line numbers below are from the original audit and no longer match current code. A re-audit is needed.

**Known Status from 2026-01-19 triage:**
- `dag.py`: Now uses direct access (`data["info"]`), not `.get()`. FIXED.
- `lineage_tree.py`: Uses direct access with documented contracts. FIXED.
- `node_detail.py`: Correctly distinguishes required fields (direct access) from optional display fields (`.get()` with explicit fallback for display purposes like `"N/A"`). This is the CORRECT pattern per CLAUDE.md.

**Original list (stale line numbers):**

- `src/elspeth/core/dag.py:129` (data.get("info")) - **FIXED: now direct access**
- `src/elspeth/core/dag.py:164` (data.get("info")) - **FIXED: now direct access**
- `src/elspeth/core/dag.py:177` (data.get("info")) - **FIXED: now direct access**
- `src/elspeth/core/rate_limit/limiter.py:251` (_suppressed_threads.get) - needs re-check
- `src/elspeth/engine/executors.py:327` (route_resolution_map.get) - needs re-check
- `src/elspeth/engine/executors.py:405` (edge_map.get) - needs re-check
- `src/elspeth/engine/executors.py:419` (edge_map.get) - needs re-check
- `src/elspeth/engine/adapters.py:131` (artifact_descriptor.get) - needs re-check
- `src/elspeth/engine/adapters.py:145` (artifact_descriptor.get) - needs re-check
- `src/elspeth/engine/adapters.py:168` (artifact_descriptor.get) - needs re-check
- `src/elspeth/engine/adapters.py:169` (artifact_descriptor.get) - needs re-check
- `src/elspeth/engine/adapters.py:176` (artifact_descriptor.get) - needs re-check
- `src/elspeth/engine/adapters.py:177` (artifact_descriptor.get) - needs re-check
- `src/elspeth/cli.py:419` (PLUGIN_REGISTRY.get) - needs re-check
- `src/elspeth/tui/widgets/node_detail.py` - **REVIEWED: Uses correct pattern (direct access for required, .get() for optional display)**
- `src/elspeth/tui/widgets/lineage_tree.py` - **FIXED: now direct access**

## Impact

- User-facing impact: inconsistent enforcement of the defensive programming prohibition.
- Data integrity / security impact: low (review-only).
- Performance or cost impact: none.

## Root Cause Hypothesis

- Legacy internal patterns and convenience usage were not audited against the prohibition; no whitelist exists for internal display or adapter code paths.

## Proposed Fix

- Code changes (modules/files):
  - Audit each listed site; refactor internal invariants to direct access and explicit errors.
  - If any are judged acceptable, document them as whitelisted trust-boundary cases.
- Config or schema changes: none.
- Tests to add/update: none unless refactors change behavior.
- Risks or migration steps: minimal; mainly doc clarity.

## Architectural Deviations

- Spec or doc reference (e.g., docs/design/architecture.md#L...): `CLAUDE.md#L318` (defensive programming prohibition).
- Observed divergence: internal code uses defensive access patterns without a trust-boundary justification.
- Reason (if known): pre-existing convenience patterns not audited.
- Alignment plan or decision needed: decide whether these are true trust boundaries or should be refactored.

## Acceptance Criteria

- Each listed call site is either refactored to direct access or explicitly documented as a whitelist exception.

## Tests

- Suggested tests to run: none.
- New tests required: no.

## Notes / Links

- Related issues/PRs: N/A
- Related design docs: `CLAUDE.md`
