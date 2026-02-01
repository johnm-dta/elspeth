# Bug Report: Secret fingerprinting is shallow and skips critical config fields (DSN passwords can leak to Landscape)

## Summary

- Secret fingerprinting only applies to top-level keys in plugin `options` dicts; nested secrets inside dict/list structures are not fingerprinted.
- Secret handling does not cover non-plugin settings fields such as `landscape.url`, which can contain embedded credentials (e.g., PostgreSQL DSNs).
- When `ELSPETH_FINGERPRINT_KEY` is missing, fingerprinting silently “fails open” and preserves secrets as-is (currently covered by tests), making production misconfiguration an audit-data leak.

## Severity

- Severity: critical
- Priority: P0

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
- Notable tool calls or steps: static inspection of `src/elspeth/core/config.py` + design docs

## Steps To Reproduce

### A) DSN password leak via `landscape.url`

1. Create a `settings.yaml` with:
   - `landscape.url: "postgresql://user:password@localhost/db"`
   - minimal required fields (`datasource`, `sinks`, `output_sink`)
2. Call `load_settings(settings.yaml)` and then `resolve_config(settings)`.
3. Observe the resolved config includes the password in `landscape.url`.

### B) Nested secrets are not fingerprinted (even with fingerprint key set)

1. Set `ELSPETH_FINGERPRINT_KEY`.
2. Create a `settings.yaml` with nested secrets, e.g.:
   - `datasource.options.auth.api_key: "sk-..."` (nested dict)
3. Call `load_settings(settings.yaml)`.
4. Observe nested `api_key` remains present (only top-level keys are processed).

### C) “Fail open” when fingerprint key missing

1. Ensure `ELSPETH_FINGERPRINT_KEY` is not set.
2. Configure `datasource.options.api_key: "sk-..."`.
3. Call `load_settings(settings.yaml)` and then run via CLI (which persists `resolve_config()` to Landscape).
4. Observe the secret is stored raw in the resolved config and therefore can be persisted into Landscape.

## Expected Behavior

- Secrets are never written to Landscape configuration records:
  - DSN-style settings are stored without passwords (or with password fingerprints only).
  - Plugin options are recursively scanned and fingerprinted.
- If secrets are present but the fingerprint key is missing, the system fails fast (or at least redacts secrets instead of persisting them).

## Actual Behavior

- Fingerprinting is shallow (top-level only) and limited to plugin options.
- `landscape.url` and other non-option fields are not sanitized for embedded credentials.
- If the fingerprint key is missing, secrets remain raw.

## Evidence

- Fingerprinting is non-recursive and only checks immediate keys:
  - `src/elspeth/core/config.py:753-781` (`_fingerprint_secrets()` only iterates `options.items()`)
- Fingerprinting is only applied to specific plugin option sections (not `landscape.url`):
  - `src/elspeth/core/config.py:784-847` (`_fingerprint_config_options()` processes datasource/sinks/row_plugins/aggregations only)
- Fingerprinting fails open when key missing:
  - `src/elspeth/core/config.py:756-779` (catch `ValueError` and `pass`, preserving raw secrets)
  - `tests/core/test_config.py:1748-1774` (explicitly asserts secrets are preserved when fingerprint key is not set)
- Architecture requires secrets never be stored:
  - `docs/design/architecture.md:741-780` (“Secrets are NEVER written to Landscape”)

## Impact

- User-facing impact: operators/auditors can inadvertently persist credentials into the audit DB by configuration mistake or by using DSN strings.
- Data integrity / security impact: severe. Landscape becomes a secret store, violating the audit system’s data governance model.
- Performance or cost impact: N/A (primary risk is confidentiality).

## Root Cause Hypothesis

- Secret handling was implemented as a shallow convenience layer for common option key names, without:
  - recursive traversal
  - coverage for credentials embedded in DSN strings
  - a fail-closed policy when fingerprint key is unavailable

## Proposed Fix

- Code changes (modules/files):
  - `src/elspeth/core/config.py`:
    - Implement recursive fingerprinting for `options` values (walk dicts/lists; fingerprint matching keys anywhere in the structure).
    - Add sanitization for DSN/URL fields that can embed passwords (at least `LandscapeSettings.url`):
      - parse with SQLAlchemy’s URL parser (or a strict DSN parser) and drop/replace passwords
      - store a `*_password_fingerprint` (HMAC) when a password was present
    - Replace “fail open” behavior with a configurable fail-closed policy:
      - if secret-like keys are present and fingerprint key is missing, raise `ValueError` (production-safe default)
      - optionally allow an explicit dev override (e.g., `ELSPETH_ALLOW_RAW_SECRETS_FOR_DEV=true`) so tests/development can proceed intentionally
- Config or schema changes:
  - Consider adding a first-class “redaction policy” flag (dev vs prod) to make the behavior explicit and auditable.
- Tests to add/update:
  - Add tests asserting:
    - DSN passwords are redacted/fingerprinted in `resolve_config()`
    - nested secret keys are fingerprinted
    - missing fingerprint key causes a failure (or explicit redaction), unless dev override is enabled
- Risks or migration steps:
  - This is a behavior change relative to the current test suite (“fail open”); requires an explicit decision about dev-mode semantics vs audit safety.

## Architectural Deviations

- Spec or doc reference: `docs/design/architecture.md:741-780`
- Observed divergence: secrets can be present in resolved config and therefore persisted into Landscape.
- Reason (if known): implemented as shallow, best-effort convenience rather than an enforced boundary policy.
- Alignment plan or decision needed: define an explicit policy for dev environments that does not compromise audit DB safety.

## Acceptance Criteria

- No resolved config persisted to Landscape contains raw secret values (including DSN passwords and nested option secrets).
- Missing fingerprint key cannot silently leak secrets in production mode.
- New tests cover DSN redaction and recursive fingerprinting.

## Tests

- Suggested tests to run: `.venv/bin/python -m pytest tests/core/test_config.py`
- New tests required: yes (nested secrets + DSN redaction + fail-closed behavior)

## Notes / Links

- Related issues/PRs: N/A
- Related design docs: `docs/design/architecture.md` (Data governance / secret handling)
