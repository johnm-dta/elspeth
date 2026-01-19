# Bug Report: LandscapeExporter export is not self-contained (run/node config JSON omitted)

## Summary

- The exporter is described as producing audit data “suitable for compliance review and legal inquiry”.
- `LandscapeExporter` currently exports:
  - `runs.config_hash` but not `runs.settings_json`
  - `nodes.config_hash` but not `nodes.config_json` (nor determinism/schema config fields)
- This makes exported audit trails less useful outside the originating system because configuration required for traceability is missing.

## Severity

- Severity: major
- Priority: P2

## Reporter

- Name or handle: codex
- Date: 2026-01-19
- Related run/issue ID: N/A

## Environment

- Commit/branch: `main` @ `8ca061c9293db459c9a900f2f74b19b59a364a42`
- OS: Linux (Ubuntu kernel 6.8.0-90-generic)
- Python version: 3.13.1
- Config profile / env vars: N/A
- Data set or fixture: N/A

## Agent Context (if relevant)

- Goal or task prompt: deep dive subsystem 4 (Landscape) and create bug tickets
- Model/version: GPT-5.2 (Codex CLI)
- Tooling and permissions (sandbox/approvals): workspace-write, network restricted, approvals on-request
- Determinism details (seed, run ID): N/A
- Notable tool calls or steps: static inspection of export record mappings

## Steps To Reproduce

1. Create any run and nodes (normal pipeline execution).
2. Export the run via `LandscapeExporter.export_run(run_id)`.
3. Observe that:
   - the `run` record lacks `settings_json`
   - the `node` record lacks `config_json`

## Expected Behavior

- Exported audit trail contains the resolved configuration needed to explain decisions:
  - `runs.settings_json`
  - `nodes.config_json`
  - (optionally) node determinism, schema_mode/schema_fields

## Actual Behavior

- Export includes hashes but omits the underlying config JSON payloads.

## Evidence

- Exporter omits settings/config JSON:
  - `src/elspeth/core/landscape/exporter.py:162-185` (`run` record omits `settings_json`)
  - `src/elspeth/core/landscape/exporter.py:173-185` (`node` record omits `config_json`)
- Schema stores these values explicitly:
  - `src/elspeth/core/landscape/schema.py` (`runs.settings_json`, `nodes.config_json`)
- Audit standard requires configuration traceability:
  - `CLAUDE.md` (“Every decision must be traceable to … configuration …”)

## Impact

- User-facing impact: exported audit trail may be insufficient for third-party review without separate access to the original config artifacts.
- Data integrity / security impact: moderate (export incompleteness).
- Performance or cost impact: including config JSON may increase export size; needs explicit decision.

## Root Cause Hypothesis

- Exporter was implemented with a minimal record schema and hasn’t been revisited for “portable audit trail” requirements.

## Proposed Fix

- Code changes (modules/files):
  - Decide export contract:
    - If “self-contained export” is required: include `settings_json` and `config_json` in export records.
    - If not: explicitly document that exported audit trails require separate config artifacts.
  - `src/elspeth/core/landscape/exporter.py` add fields accordingly.
- Config or schema changes: none.
- Tests to add/update:
  - Add exporter tests asserting config JSON presence (if desired behavior).
- Risks or migration steps:
  - Export size growth; may require optional inclusion controlled by config (`landscape.export.include_config_json`).

## Architectural Deviations

- Spec or doc reference: `CLAUDE.md` auditability standard
- Observed divergence: export not portable/self-contained for config traceability.
- Reason (if known): minimal initial exporter schema.
- Alignment plan or decision needed: decide portability requirements for exported audit artifacts.

## Acceptance Criteria

- A documented and tested decision exists:
  - either export includes resolved config JSON, or docs clearly state export is hash-only for config and requires separate artifacts.

## Tests

- Suggested tests to run: `pytest tests/core/landscape/test_exporter.py`
- New tests required: maybe (depends on decision)

## Notes / Links

- Related issues/PRs: N/A
