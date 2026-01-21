# Bug Report: `elspeth purge` ignores `payload_store` settings and defaults to the wrong directory

## Summary

- The `purge` command should purge payload blobs referenced by Landscape rows, but its default payload directory is `./payloads` (or `<db_dir>/payloads`) rather than the configured `payload_store.base_path` (default: `.elspeth/payloads`).
- When the database is inferred from `settings.yaml`, the payload directory still defaults to `./payloads` relative to the current working directory.
- `purge` also ignores `payload_store.backend` and `payload_store.retention_days` settings from config (it always assumes filesystem and uses its own `--retention-days` default).

## Severity

- Severity: major
- Priority: P1

## Reporter

- Name or handle: codex
- Date: 2026-01-20
- Related run/issue ID: N/A

## Environment

- Commit/branch: `8cfebea78be241825dd7487fed3773d89f2d7079` (main)
- OS: Linux (Ubuntu kernel 6.8.0-90-generic)
- Python version: 3.13.1
- Config profile / env vars: N/A
- Data set or fixture: N/A

## Agent Context (if relevant)

- Goal or task prompt: deep dive into subsystem 1 (CLI), identify bugs, create tickets
- Model/version: GPT-5.2 (Codex CLI)
- Tooling and permissions (sandbox/approvals): workspace-write, network restricted, approvals on-request
- Determinism details (seed, run ID): N/A
- Notable tool calls or steps: code inspection of CLI + config defaults

## Steps To Reproduce

1. Use default config (or set `payload_store.base_path` explicitly to `.elspeth/payloads`).
2. Run a pipeline that stores payload blobs under the configured payload store directory.
3. Run `elspeth purge --yes --database ./landscape.db` without specifying `--payload-dir`.
4. Observe that it looks in `./payloads` (or `<db_dir>/payloads`) rather than `.elspeth/payloads`, and therefore purges nothing (or purges the wrong directory if it exists).

## Expected Behavior

- If a settings file is present (or if config can be loaded), `purge` should use:
  - `settings.payload_store.base_path` as the default payload directory
  - `settings.payload_store.retention_days` as the default retention policy
  - `settings.payload_store.backend` to select the appropriate purge strategy or fail fast if unsupported

## Actual Behavior

- `purge` defaults payload directory to `payloads/` unrelated to the configured default `.elspeth/payloads`.
- `purge` always assumes filesystem payload store and always uses its own retention default.

## Evidence

- Purge default payload directory logic:
  - `src/elspeth/cli.py:538-545`
- Config defaults for payload store base path and retention:
  - `src/elspeth/core/config.py:612-619`

## Impact

- User-facing impact: operators run `purge` and see “No payloads older than … found” even when there are eligible payloads; retention policy appears broken.
- Data integrity / security impact: moderate. If the default `payloads/` directory exists and contains unrelated files, an operator could delete unintended data after confirmation.
- Performance or cost impact: potentially high (payload store grows without being purged).

## Root Cause Hypothesis

- `purge` was implemented as a standalone command and does not consult the canonical settings model (`ElspethSettings.payload_store`), despite such settings existing.

## Proposed Fix

- Code changes (modules/files):
  - `src/elspeth/cli.py`:
    - When loading `settings.yaml`, use `config.payload_store.base_path` and `config.payload_store.retention_days` as defaults.
    - If `--payload-dir` is provided, it should override config.
    - Fail fast unless `config.payload_store.backend == "filesystem"` (until other backends exist).
    - Apply `expanduser()` to configured/CLI paths (see separate ticket on tilde handling).
- Config or schema changes: none.
- Tests to add/update:
  - Add a purge test that writes a settings.yaml with `payload_store.base_path: .elspeth/payloads`,
    creates expired refs in the DB, writes blobs under that path, and verifies purge finds/deletes them without `--payload-dir`.
- Risks or migration steps:
  - None, but clarify behavior when `--database` is provided without settings: choose a safe default and document it.

## Architectural Deviations

- Spec or doc reference: `CLAUDE.md` data storage points / retention concepts
- Observed divergence: retention tooling does not match configured payload store location.
- Reason (if known): incremental CLI implementation.
- Alignment plan or decision needed: define whether purge is config-driven (recommended) or purely manual.

## Acceptance Criteria

- With a config specifying `payload_store.base_path`, running `elspeth purge` with just `--database` (and settings.yaml present) purges blobs from that configured directory by default.
- Purge fails fast with a clear message when payload store backend is unsupported.

## Tests

- Suggested tests to run:
  - `pytest tests/cli/test_cli.py -k purge`
- New tests required: yes (default payload_store path + backend handling)

## Notes / Links

- Related issues/PRs: N/A
