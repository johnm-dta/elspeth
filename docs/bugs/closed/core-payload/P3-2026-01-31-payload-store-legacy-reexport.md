# Bug Report: Backwards compatibility re-export in payload_store.py

## Summary

- `payload_store.py` re-exports symbols "for backwards compatibility," violating the No Legacy Code Policy.

## Severity

- Severity: trivial
- Priority: P3

## Reporter

- Name or handle: Codex (triaged by Claude)
- Date: 2026-01-31

## Evidence

- `src/elspeth/core/payload_store.py:17-18` - "Re-export for backwards compatibility" comment
- CLAUDE.md:797-841 forbids backwards compatibility code

## Proposed Fix

- Remove re-exports; update imports to use canonical paths

## Acceptance Criteria

- No compatibility re-exports remain

## Verification (2026-02-01)

**Status: FIXED**

- Removed the backwards-compatibility `__all__` re-export and now only exports `FilesystemPayloadStore`.
- Updated imports to use `elspeth.contracts.payload_store` for `PayloadStore` / `IntegrityError`.

## Closure Report (2026-02-01)

**Status:** CLOSED (FIXED)

### Closure Notes

- Deleted the compatibility re-export and migrated code/test imports to the canonical contracts path.
