# Bug Report: CallVerifier lacks field-level order sensitivity configuration

## Summary

- `CallVerifier` only supports instance-level `ignore_order` configuration. Users cannot specify that some fields are order-sensitive (ranked results) while others are order-insensitive (tags) within the same verification session. This forces a global trade-off between false negatives (missing drift) and false positives (noisy alerts).

## Severity

- Severity: minor
- Priority: P3

## Reporter

- Name or handle: claude-review-board
- Date: 2026-01-29
- Related run/issue ID: P3-2026-01-21-verifier-ignore-order-hides-drift (Phase 2)

## Resolution (2026-02-02)

**Status: CLOSED → P4 Enhancement (Backlog)**

### Why This Is Not A Bug

This is explicitly a **feature enhancement request**, not a bug:

1. **Already marked P3/minor** - reporter acknowledged this is low priority
2. **"Phase 2" of planned improvement** - this is roadmap work, not defect repair
3. **Current behavior works correctly** - global `ignore_order` functions as designed
4. **Workaround documented** - "create two CallVerifier instances" per the report itself
5. **Zero users** - no one is experiencing this limitation

### Current State

The `CallVerifier` works correctly with global `ignore_order` configuration. The request is for *more granular* control, which is a nice-to-have enhancement.

### Deferred To

Backlog - revisit when:
- Users request field-level verification control
- Verification becomes a production-critical feature
- Phase 1 (`ignore_order` parameter) sees actual usage

### Closed By

- Reviewer: Claude Opus 4.5
- Date: 2026-02-02
- Reason: Enhancement request, not a bug - working code, zero users, workaround exists
