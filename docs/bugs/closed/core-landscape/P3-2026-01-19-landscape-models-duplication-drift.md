# Bug Report: `core/landscape/models.py` duplicates audit contracts but diverges from runtime contracts/schema (test drift + confusion)

## Summary

- The repo defines audit dataclasses twice:
  - `src/elspeth/contracts/audit.py` (strict, used by runtime code and public exports)
  - `src/elspeth/core/landscape/models.py` (additional/legacy models)
- `core/landscape/models.py` has drift vs contracts/schema (examples):
  - `Node` is missing `schema_mode` / `schema_fields` even though schema and runtime `Node` include them.
  - `RoutingEvent.mode` is `str` instead of `RoutingMode`.
  - `Checkpoint.created_at` is `datetime | None` despite schema requiring `created_at NOT NULL`.
- Tests import `elspeth.core.landscape.models` directly, so they can pass even when runtime contracts differ.

## Severity

- Severity: minor
- Priority: P3

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
- Notable tool calls or steps: code + test inspection

## Steps To Reproduce

1. Compare audit contracts:
   - `src/elspeth/contracts/audit.py` vs `src/elspeth/core/landscape/models.py`.
2. Observe mismatched fields/types.
3. Note that tests import the legacy models module (e.g., `tests/core/landscape/test_models_enums.py`), so they can validate the wrong contract surface.

## Expected Behavior

- There is a single source of truth for audit record contracts (the contracts subsystem).
- Tests validate the same contracts used in runtime APIs (`elspeth.contracts.audit` / `elspeth.core.landscape` exports).

## Actual Behavior

- Duplicate model definitions exist and have drift, which can hide bugs and confuse contributors.

## Evidence

- Missing schema audit fields in legacy Node:
  - `src/elspeth/core/landscape/models.py:49-64` (no `schema_mode` / `schema_fields`)
  - `src/elspeth/contracts/audit.py:49-82` (includes them)
- Type mismatch example:
  - `src/elspeth/core/landscape/models.py:248-260` (`RoutingEvent.mode: str`)
  - `src/elspeth/contracts/audit.py` (`RoutingEvent.mode: RoutingMode`)
- Schema mismatch example:
  - `src/elspeth/core/landscape/models.py:297-312` (`Checkpoint.created_at: datetime | None`)
  - `src/elspeth/core/landscape/schema.py` `checkpoints.created_at` is `nullable=False`
- Tests importing legacy models:
  - `tests/core/landscape/test_models_enums.py`
  - `tests/core/landscape/test_models.py`

## Impact

- User-facing impact: low (runtime uses contracts), but developer-facing impact is significant.
- Data integrity / security impact: indirect. Drift makes it easier to introduce audit contract mismatches unnoticed.
- Performance or cost impact: N/A

## Root Cause Hypothesis

- Models were initially implemented within Landscape and later moved to Contracts, but the legacy module remained and was kept alive by tests.

## Proposed Fix

- Code changes (modules/files):
  - Prefer a single contract source:
    - Remove `src/elspeth/core/landscape/models.py` and update tests/imports to use `elspeth.contracts.audit` or `elspeth.core.landscape` exports, OR
    - Make `src/elspeth/core/landscape/models.py` a thin re-export of contracts (no duplicate dataclasses).
- Config or schema changes: none.
- Tests to add/update:
  - Update `tests/core/landscape/test_models*.py` to import from `elspeth.contracts.audit` (or `elspeth.core.landscape`) instead of the legacy module.
- Risks or migration steps:
  - If external users import `elspeth.core.landscape.models`, this is a breaking change; repo guidance discourages compatibility shims, so call sites should be updated.

## Architectural Deviations

- Spec or doc reference: `CLAUDE.md` (prohibition on legacy compatibility shims / duplicated interfaces)
- Observed divergence: duplicate “contract” definitions exist and drift.
- Reason (if known): module not removed after contract migration.
- Alignment plan or decision needed: decide whether `core/landscape/models.py` should exist at all.

## Acceptance Criteria

- There is exactly one authoritative audit model definition set.
- Tests validate the same dataclasses used by production code.

## Tests

- Suggested tests to run: `pytest tests/core/landscape/test_models*.py`
- New tests required: no (but update existing)

## Notes / Links

- Related issues/PRs: N/A

---

## VERIFICATION: 2026-01-25

**Status:** STILL VALID

**Verified By:** Claude Code P3 verification wave 1

**Current Code Analysis:**

All drift issues reported in the bug are CONFIRMED present in the current codebase:

1. **Node missing schema_mode/schema_fields** - CONFIRMED
   - `src/elspeth/core/landscape/models.py` lines 50-63: Node class lacks these fields
   - `src/elspeth/contracts/audit.py` lines 68-70: Node class HAS these fields with comment "# Schema configuration for audit trail (WP-11.99)"
   - These fields were added to contracts/audit.py on 2026-01-17 (commit 4b21b51) but models.py was never updated

2. **RoutingEvent.mode type mismatch** - CONFIRMED
   - `src/elspeth/core/landscape/models.py` line 286: `mode: str  # move, copy`
   - `src/elspeth/contracts/audit.py` line 277: `mode: RoutingMode  # Strict: enum only`

3. **Checkpoint.created_at nullable mismatch** - CONFIRMED
   - `src/elspeth/core/landscape/models.py` line 339: `created_at: datetime | None`
   - `src/elspeth/contracts/audit.py` line 332: `created_at: datetime  # Required - schema enforces NOT NULL (Tier 1 audit data)`

4. **Call.call_type and Call.status type mismatch** - CONFIRMED (additional drift not in original report)
   - `src/elspeth/core/landscape/models.py` lines 250-251: `call_type: str` and `status: str`
   - `src/elspeth/contracts/audit.py` lines 238-239: `call_type: CallType` and `status: CallStatus` (both enums with "# Strict: enum only" comments)

**Additional Findings:**

- contracts/audit.py contains 4 additional classes NOT present in models.py:
  - ValidationErrorRecord (added for AUD-003)
  - TransformErrorRecord (added for error recording)
  - TokenOutcome (added for AUD-001)
  - NonCanonicalMetadata (added for non-canonical data handling)

**Git History:**

- 2026-01-16 (commit 687ed8b): contracts/audit.py created with strict enum types
- 2026-01-17 (commit 4b21b51): schema_mode/schema_fields added to contracts/audit.py Node class for WP-11.99
- 2026-01-19: This bug reported
- 2026-01-21 onwards: Multiple additional classes added to contracts/audit.py (TokenOutcome, ValidationErrorRecord, etc.)
- models.py was NEVER updated with any of these changes

**Usage Analysis:**

- Production code (src/): NO imports from models.py - all use contracts/audit.py ✅
- Public API (src/elspeth/core/landscape/__init__.py): Re-exports from contracts ✅
- Tests: Still import from models.py ❌
  - tests/core/landscape/test_models.py (line 12, 31, 52, 69)
  - tests/core/landscape/test_models_enums.py (line 12)
  - tests/core/landscape/test_models_mutation_gaps.py
  - tests/core/landscape/test_schema.py

The models.py file is effectively orphaned - only used by tests, creating the exact test drift risk described in the bug report.

**Root Cause Confirmed:**

YES. The bug is completely valid. models.py is a legacy artifact from before the contracts subsystem was created. When contracts/audit.py was introduced (Jan 16), production code migrated but:
1. models.py was not deleted
2. Tests continued importing from models.py
3. New fields/classes added to contracts/audit.py never propagated to models.py
4. This creates test drift - tests validate wrong contract surface

**Recommendation:**

KEEP OPEN - Valid P3 bug requiring cleanup.

Proposed fix aligns with CLAUDE.md "No Legacy Code Policy":
1. Delete src/elspeth/core/landscape/models.py entirely
2. Update all test imports to use `from elspeth.contracts.audit import` or `from elspeth.core.landscape import` (which re-exports contracts)
3. No compatibility shims per policy

This is safe because:
- Production code already uses contracts/audit.py
- Public API already re-exports from contracts
- Only tests need updating (internal consumers)

Risk: Low - affects only test code, not production.

---

## CLOSURE: 2026-01-28

**Status:** FIXED

**Fixed By:** Unknown (discovered during bug audit)

**Resolution:**

The fix was implemented by deleting `src/elspeth/core/landscape/models.py` entirely, following the CLAUDE.md "No Legacy Code Policy".

**Verification:**

```bash
$ ls src/elspeth/core/landscape/models.py
ls: cannot access 'src/elspeth/core/landscape/models.py': No such file or directory
```

The file no longer exists. All code now uses the single authoritative source:
- `src/elspeth/contracts/audit.py` - canonical audit dataclasses
- `src/elspeth/core/landscape/__init__.py` - re-exports from contracts

**Impact:**

- No more duplicate model definitions
- No more drift between models.py and contracts/audit.py
- Tests now validate the same contracts used by production code
