# CSS Extraction Review — `components/audit/audit.css`

**Source range**: `src/elspeth/web/frontend/src/App.css` lines 1722–2226
**Destination**: `.worktrees/css-split/src/elspeth/web/frontend/src/components/audit/audit.css` (412 lines)
**Reviewer**: complex-reviewer (paired with the css-split extraction)
**Verdict**: **Approved** — extraction is faithful; one design observation below.

---

## Summary

| Aspect | Result |
|---|---|
| Retained selector clusters present | ✓ All three (`.audit-readiness-*`, `.explain-dialog*`, `.readiness-row-detail-*`) |
| Moved-out clusters absent | ✓ `.graph-modal-*` and `.yaml-modal-*` not in `audit.css` (only mentioned in two comments) |
| Selector text byte-identical | ✓ (diff against original returned 0) |
| Declaration order preserved | ✓ All three blocks diff-clean against the original line ranges |
| Top-of-file comment notes the moved-out modals | ✓ Lines 1–5 |
| Trailing `.yaml-modal-body` remnant (orig 2221–2225) | ✓ Correctly absent — landed in `common.css` per Path B |

---

## 1. Intent Fit (per-cluster verification)

### `.audit-readiness*` (original 1740–1931 → new 25–216)

Top-level selectors enumerated in both files match exactly, in order:

```
.audit-readiness, .audit-readiness--collapsed, .audit-readiness--loading,
.audit-readiness--error, .audit-readiness-live-region,
.audit-readiness-loading, .audit-readiness-error, .audit-readiness-summary,
.audit-readiness-summary:hover, .audit-readiness-summary-meta,
.audit-readiness-header, .audit-readiness-title, .audit-readiness-freshness,
.audit-readiness-actions, .audit-readiness-action-btn,
.audit-readiness-action-btn--ghost,
.audit-readiness-action-btn--ghost:hover:not(:disabled):not([aria-disabled="true"]),
.audit-readiness-rows, .audit-readiness-row, .audit-readiness-row:last-child,
.audit-readiness-row--ok, .audit-readiness-row--warning,
.audit-readiness-row--error, .audit-readiness-row--not_applicable,
.audit-readiness-row-btn, .audit-readiness-row-static,
.audit-readiness-row-btn, .audit-readiness-row-btn:hover,
.audit-readiness-glyph, .audit-readiness-row-label,
.audit-readiness-row-summary
```

Body bytes identical (`diff orig new` = empty).

### `.explain-dialog*` (original 2033–2113 → new 226–306)

All 13 selector entry points present in matching order; bodies identical.

### `.readiness-row-detail-*` (original 2125–2219 → new 318–412)

All 14 selector entry points present in matching order; bodies identical.

The trailing `.yaml-modal-body` block at original 2221–2225 — which the brief explicitly identifies as a "trailing remnant" sandwiched after `.readiness-row-detail-component-id` — is **not** in `audit.css`. The brief states it moved to `common.css`. Verified absent here (no `yaml-modal-body` substring outside the two header comments).

## 2. Scope Discipline

- No additions beyond the three declared clusters.
- New top-of-file authoring comment (lines 1–5) is the only material introduced; it is documentation, not a new rule.
- No reordering of declarations within retained clusters.
- No edits to existing selectors or values.

## 3. Structural Integrity

| Invariant | Status | Evidence |
|---|---|---|
| Brace balance | ✓ | Trivial inspection; each `{` paired |
| Top-level selector count match | ✓ | 60 retained selectors counted in both source range and new file (excluding the three moved modal clusters totalling 11 selectors) |
| Three section banners preserved | ✓ | "Audit Readiness panel" (line 7), "ExplainDialog" (line 219), "ReadinessRowDetail" (line 308) |
| Declaration order | ✓ | `diff` clean on each cluster |
| No `.graph-modal-*` rules | ✓ | `grep -E '^\.graph-modal'` returns 0 hits |
| No `.yaml-modal-*` rules | ✓ | `grep -E '^\.yaml-modal'` returns 0 hits |

The two surface mentions of `yaml-modal`/`graph-modal` in the new file are **inside comments**: the top-of-file note (line 3) explaining the move-out, and the ExplainDialog header comment (line 222) describing the dialog/backdrop pattern. Both are documentary and load-bearing — they tell the next reader where to look for the modal base styles.

## 4. Cross-Reference & Boundary Integrity

- The ExplainDialog comment block (lines 219–225) describes the dialog as "mirroring the `.yaml-modal-*` / `.graph-modal-*` pattern". Since those rules now live in `common.css`, the cross-reference still resolves (just to a different file). The comment's wording does not say "above" or "below" or otherwise assume co-location — it remains accurate post-split.
- Top-of-file comment (lines 1–5) **does** explicitly direct the reader to `common.css` for the modal rules. This is the required hand-off; it is present.
- No orphaned transitions: the original had `.audit-readiness-row-summary` (1927) immediately followed by `.graph-modal-backdrop` (1933) with no banner comment between them; the new file replaces that adjacency with a section banner at line 219 introducing ExplainDialog, which is cleaner than the original.

## 5. Style / Behaviour Continuity

The new top-of-file comment matches the established commentary register: terse multi-line, ownership statement, cross-reference to sibling file. Selector blocks are byte-identical so no continuity drift is possible there.

## 6. Observations the Brief Asked For

**Candidate cluster that *could* have moved but did not: `.explain-dialog-backdrop` + `.explain-dialog`.**

These two selectors (lines 226–244) are structurally identical to the `.graph-modal-*` / `.yaml-modal-*` base rules that moved to `common.css`:

| Property | `.graph-modal` / `.yaml-modal` (moved) | `.explain-dialog` (kept) |
|---|---|---|
| `position` | `fixed` | `fixed` |
| `inset` | `32px` | `32px` |
| `z-index` | `var(--z-dialog)` | `var(--z-dialog)` |
| `display` | `flex` | `flex` |
| `flex-direction` | `column` | `column` |
| `background-color` | `var(--color-surface-paper)` | `var(--color-surface-paper)` |
| `border` | `1px solid var(--color-border)` | `1px solid var(--color-border)` |
| `border-radius` | `var(--radius-md)` | `var(--radius-md)` |
| `box-shadow` | `0 8px 32px rgba(0,0,0,0.25)` | `0 8px 32px rgba(0,0,0,0.25)` |
| `overflow` | `hidden` | `hidden` |

`.explain-dialog-backdrop` is likewise identical to the two moved `.*-modal-backdrop` rules. The same goes for the `-header`, `-close`, and `-close:hover` shapes downstream.

**Whether this should have been promoted to `common.css` (e.g. as a `.modal-base` + `.modal-backdrop-base` pair, or a shared `.dialog-base` class) is a design decision outside the strict "preserve byte fidelity during a split" remit of Path B.** The ExplainDialog component-scoped class names are CSS-namespaced, so leaving them in `audit.css` is defensible — they are owned by the audit component tree (`AuditReadinessPanel` → `ExplainDialog`). But the ExplainDialog comment itself signals the duplication ("Mirrors the `.yaml-modal-*` / `.graph-modal-*` pattern"), which is an explicit invitation to factor.

**Recommendation**: file as a follow-up observation for a *subsequent* refactor pass (introduce `.dialog-base` / `.dialog-backdrop-base` in `common.css`, have `.explain-dialog` / `.graph-modal` / `.yaml-modal` `@extend`-equivalent via additional class names in markup). Do NOT block this split on it — Path B's mandate is preservation, not deduplication.

**Dead rules**: none identified. All retained selectors are referenced by `AuditReadinessPanel.tsx`, `AuditReadinessRow.tsx`, `ExplainDialog.tsx`, or `ReadinessRowDetail.tsx` per the comment annotations (consumers not separately verified — see Information Gaps).

## Issues Found

### Critical
None.

### Major
None.

### Minor
None.

---

## Confidence Assessment

**Confidence**: High.

**Basis**: Verified directly via `diff` that each of the three retained clusters' bodies is byte-identical to the corresponding original line range; verified via `grep` that the two moved-out modal clusters have zero selector occurrences in `audit.css` (only documentary mentions in comments); verified via `awk`-extracted selector listings that ordering is preserved; verified the top-of-file hand-off comment is present and accurate.

## Risk Assessment

**Residual Risk**: Low.

- If `common.css` did **not** in fact receive the moved `.graph-modal-*`, `.yaml-modal-*`, and trailing `.yaml-modal-body` rules, the application styling will break — but that is `common.css`'s review, not this file's. Within `audit.css` itself there is no risk surface.
- The duplication between `.explain-dialog` and the moved `.graph-modal` / `.yaml-modal` rules is *pre-existing*, not introduced by the split. It carries the same maintenance risk it did in `App.css`: a change to the modal visual shape (e.g. border-radius) must be made in two files now instead of co-located. This was already the case across the modal cluster pair pre-split, so the split does not regress maintainability — it just makes the duplication visible across files instead of across non-contiguous regions of one file.

## Information Gaps

- I did not verify that `common.css` actually received the moved rules — that is the scope of the `common.css` review.
- I did not run the frontend build (`npm run build`) or the visual Playwright tests to confirm no selectors regressed at runtime. The diff-level fidelity is byte-exact; runtime regression would require a build-system change (e.g. the new `audit.css` not being imported from the entry point), which is also outside the scope of this file review.
- I did not verify that every selector in `audit.css` is actually referenced by a `className` somewhere in the TSX tree — I trusted the existing comments that name the owning components. Dead-rule detection at the application level is a separate audit.

## Caveats

- Path B's mandate is **preservation of behaviour during a split**. This review confirms preservation. It does not endorse any pre-existing design choices (notably the duplication noted in §6); those are separate refactor candidates.
- If the operator's intent for the split was to *also* deduplicate the dialog primitive shape, this extraction does not achieve that and that intent would need to be stated explicitly as a follow-up.
