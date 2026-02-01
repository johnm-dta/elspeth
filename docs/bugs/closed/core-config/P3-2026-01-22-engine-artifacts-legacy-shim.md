# Bug Report: `elspeth.engine.artifacts` is a legacy re-export shim

## Summary

- `src/elspeth/engine/artifacts.py` exists only to re-export `ArtifactDescriptor` from `elspeth.contracts`.
- This is a backwards-compatibility shim, which violates the repository's "No Legacy Code" policy and keeps an old import path alive without need.

## Severity

- Severity: trivial
- Priority: P3

## Reporter

- Name or handle: codex
- Date: 2026-01-22
- Related run/issue ID: N/A

## Environment

- Commit/branch: `ae2c0e6f088f467276582fa8016f91b4d3bb26c7` (main)
- OS: Linux (Ubuntu kernel 6.8.0-90-generic)
- Python version: 3.13.1
- Config profile / env vars: N/A
- Data set or fixture: N/A

## Agent Context (if relevant)

- Goal or task prompt: deep dive into `src/elspeth/engine/artifacts.py`, find bugs, create tickets
- Model/version: GPT-5 (Codex CLI)
- Tooling and permissions (sandbox/approvals): workspace-write, network restricted, approvals on-request
- Determinism details (seed, run ID): N/A
- Notable tool calls or steps: code inspection + repo policy review

## Steps To Reproduce

1. Open `src/elspeth/engine/artifacts.py`.
2. Observe it only re-exports `ArtifactDescriptor` from `elspeth.contracts`.
3. Compare with the "No Legacy Code Policy" in `CLAUDE.md`, which forbids compatibility shims.

## Expected Behavior

- There is a single canonical import path for `ArtifactDescriptor` (`elspeth.contracts`).
- Legacy shim modules are removed and all call sites updated.

## Actual Behavior

- `elspeth.engine.artifacts` persists as a re-export shim with no engine-specific behavior.

## Evidence

- Shim implementation: `src/elspeth/engine/artifacts.py`
- Policy forbidding compatibility shims: `CLAUDE.md`
- Old import path still used in tests/docs: `rg "elspeth.engine.artifacts"`

## Impact

- User-facing impact: none directly.
- Data integrity / security impact: none.
- Performance or cost impact: negligible; main cost is policy violation and maintenance drift risk.

## Root Cause Hypothesis

- The module was left behind after moving `ArtifactDescriptor` into `elspeth.contracts.results` to avoid updating imports.

## Proposed Fix

- Code changes (modules/files):
  - Remove `src/elspeth/engine/artifacts.py`.
  - Update all imports to `from elspeth.contracts import ArtifactDescriptor`.
- Config or schema changes: none.
- Tests to add/update: update existing tests/docs that import `elspeth.engine.artifacts`.
- Risks or migration steps: none; this is an internal codebase path change.

## Architectural Deviations

- Spec or doc reference: `CLAUDE.md` (No Legacy Code Policy)
- Observed divergence: backward-compatibility shim retained in engine namespace.
- Reason (if known): convenience during refactor.
- Alignment plan or decision needed: remove shim and update call sites.

## Acceptance Criteria

- `elspeth.engine.artifacts` module is removed.
- All references updated to `elspeth.contracts.ArtifactDescriptor`.

## Tests

- Suggested tests to run:
  - `pytest tests/engine/test_orchestrator.py -k ArtifactDescriptor`
- New tests required: no

## Notes / Links

- Related issues/PRs: N/A
- Related design docs: N/A

---

## Verification (2026-02-01)

**Status: FIXED**

- Removed `src/elspeth/engine/artifacts.py` and updated all imports to use `elspeth.contracts.ArtifactDescriptor`.

## Closure Report (2026-02-01)

**Status:** CLOSED (FIXED)

### Closure Notes

- Deleted the legacy shim and migrated all code/test imports to the canonical contracts path.

## VERIFICATION: 2026-01-25

**Status:** STILL VALID

**Verified By:** Claude Code P3 verification wave 6 (FINAL)

**Current Code Analysis:**

The legacy shim at `/home/john/elspeth-rapid/src/elspeth/engine/artifacts.py` still exists and contains only a re-export of `ArtifactDescriptor` from `elspeth.contracts`:

```python
# src/elspeth/engine/artifacts.py
"""Unified artifact descriptors for all sink types.

IMPORTANT: ArtifactDescriptor is now defined in elspeth.contracts.results.
This module re-exports it as part of the engine's public API.
"""

from elspeth.contracts import ArtifactDescriptor

__all__ = ["ArtifactDescriptor"]
```

Key findings:

1. **Canonical import path exists:** `ArtifactDescriptor` is properly exported from `elspeth.contracts.__init__.py` (line 64 import, line 166 in `__all__`), which imports from `elspeth.contracts.results` where the actual class is defined.

2. **Shim is NOT part of engine's public API:** `ArtifactDescriptor` is NOT exported from `elspeth.engine.__init__.py` despite the docstring claiming "re-exports it as part of the engine's public API." The engine package exports many other types but deliberately excludes `ArtifactDescriptor`.

3. **No internal engine usage:** No files within `src/elspeth/engine/` import from `artifacts.py`. The shim is unused by the subsystem that hosts it.

4. **Widespread legacy import usage:** 94 occurrences of `from elspeth.engine.artifacts import` exist in the test suite, while only 7 files in `src/` use the canonical `from elspeth.contracts import ArtifactDescriptor` path. This indicates the shim is actively preventing migration to the canonical import.

5. **Plugin code already migrated:** All sink implementations (`csv_sink.py`, `json_sink.py`, `database_sink.py`, `blob_sink.py`) and the plugin base classes properly import from `elspeth.contracts`. Only engine tests still use the legacy path.

**Git History:**

- **Commit cfa8a14 (2026-01-16):** Consolidated result types into `contracts/results.py`, converting `artifacts.py` from a full implementation (123 lines) to a re-export shim (10 lines). Commit message explicitly states "re-exports preserved for backwards compatibility."

- **Commit dcae91a (2026-01-16):** Changed docstring from "backwards compatibility" to "public API" - likely an attempt to justify the shim's existence under the No Legacy Code Policy, but contradicted by the fact that `ArtifactDescriptor` is not actually exported from `elspeth.engine.__init__.py`.

- **No subsequent removal attempts:** No commits since 2026-01-16 have attempted to remove the shim or update the 94 legacy import sites.

**Root Cause Confirmed:**

Yes, this is a textbook legacy compatibility shim that violates CLAUDE.md's "No Legacy Code Policy":

- ✅ Fits definition: "wrapper functions that translate old APIs to new ones"
- ✅ Creates dual import paths: both `elspeth.engine.artifacts` and `elspeth.contracts` work
- ✅ Prevents full migration: 94 test files still use the old path
- ✅ Adds no functionality: file contains only a re-export
- ✅ Violates "DELETE THE OLD CODE COMPLETELY" directive

The docstring's claim that this is "part of the engine's public API" is demonstrably false since `ArtifactDescriptor` is absent from `elspeth.engine.__init__.py.__all__`.

**Recommendation:**

**Keep open - Still requires fix.**

The bug is valid and unfixed. The proposed remediation remains correct:

1. Delete `src/elspeth/engine/artifacts.py` entirely
2. Update all 94 test files to use `from elspeth.contracts import ArtifactDescriptor`
3. Verify no runtime imports remain via static analysis or import hooks

The shim should have been removed as part of commit cfa8a14 when the consolidation occurred. The fact that it was retained "for backwards compatibility" and then rebranded as "public API" represents exactly the kind of technical debt accumulation that the No Legacy Code Policy was designed to prevent.
