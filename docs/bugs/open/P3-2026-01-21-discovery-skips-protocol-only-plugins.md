# Bug Report: Plugin discovery ignores protocol-only implementations

## Summary

- Plugin docs say implementations may “subclass base classes or implement protocols directly.”
- Dynamic discovery only accepts subclasses of `BaseSource`/`BaseTransform`/`BaseSink`.
- A plugin class that implements `SourceProtocol`/`TransformProtocol`/`SinkProtocol` directly (without inheriting the base class) is silently skipped and never registered.

## Severity

- Severity: minor
- Priority: P3

## Reporter

- Name or handle: codex
- Date: 2026-01-21
- Related run/issue ID: N/A

## Environment

- Commit/branch: `ae2c0e6f088f467276582fa8016f91b4d3bb26c7` (local)
- OS: Linux (Ubuntu kernel 6.8.0-90-generic)
- Python version: 3.13.1
- Config profile / env vars: N/A
- Data set or fixture: N/A

## Agent Context (if relevant)

- Goal or task prompt: deep dive into contents of `src/elspeth/plugins` and create bug tickets
- Model/version: GPT-5 (Codex CLI)
- Tooling and permissions (sandbox/approvals): workspace-write, network restricted, approvals on-request
- Determinism details (seed, run ID): N/A
- Notable tool calls or steps: static inspection of `src/elspeth/plugins/base.py` and `src/elspeth/plugins/discovery.py`

## Steps To Reproduce

1. Create a source class in `src/elspeth/plugins/sources/` that implements `SourceProtocol` but does not inherit `BaseSource`.
2. Ensure it defines a non-empty `name` attribute and implements `load()`/`close()`.
3. Call `discover_all_plugins()`.
4. Observe that the class is not returned or registered.

## Expected Behavior

- Protocol-only plugins should be discoverable if the system advertises protocol-based implementations as supported.

## Actual Behavior

- Discovery filters on `issubclass(obj, BaseSource/BaseTransform/BaseSink)` only, so protocol-only implementations are skipped.

## Evidence

- Base class docs say protocol-only implementations are allowed: `src/elspeth/plugins/base.py`.
- Discovery requires subclassing a base class: `src/elspeth/plugins/discovery.py` (`issubclass(obj, base_class)`).

## Impact

- User-facing impact: valid plugin implementations are never registered, leading to confusing “plugin not found” behavior.
- Data integrity / security impact: low.
- Performance or cost impact: low.

## Root Cause Hypothesis

- Discovery logic was implemented against base classes only and never updated to honor protocol-only implementations.

## Proposed Fix

- Code changes (modules/files):
  - `src/elspeth/plugins/discovery.py`: accept classes that satisfy the relevant runtime-checkable protocol (e.g., `issubclass(obj, SourceProtocol)`) in addition to subclassing base classes.
  - Alternatively, update documentation to require subclassing base classes and make discovery failure explicit (raise on protocol-only classes if found).
- Tests to add/update:
  - Add a test plugin that implements `SourceProtocol` without inheriting `BaseSource` and assert it is discovered (or explicitly rejected if policy changes).
- Risks or migration steps:
  - If changing discovery policy, ensure no false positives for helper classes by validating required attributes/methods.

## Architectural Deviations

- Spec or doc reference (e.g., docs/design/architecture.md#L...): `src/elspeth/plugins/base.py` module docstring (protocol-only support).
- Observed divergence: discovery only supports base-class inheritance.
- Reason (if known): discovery keyed to base class for simplicity.
- Alignment plan or decision needed: either support protocol-only discovery or update docs to remove the claim.

## Acceptance Criteria

- Protocol-only plugin classes are discoverable (or explicitly rejected with a clear error if policy is updated).

## Tests

- Suggested tests to run: `.venv/bin/python -m pytest tests/plugins/test_discovery.py`
- New tests required: yes (protocol-only plugin discovery)

## Notes / Links

- Related issues/PRs: N/A
- Related design docs: `src/elspeth/plugins/base.py`
