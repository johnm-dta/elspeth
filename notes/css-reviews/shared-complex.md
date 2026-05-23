# Edit Review: src/elspeth/web/frontend/src/styles/shared.css

## Summary

**File**: `/home/john/elspeth/.worktrees/css-split/src/elspeth/web/frontend/src/styles/shared.css` (588 lines, type: docs / CSS — pure-concatenation extraction from `App.css`)
**Edit report received**: No (reconstructed from caller brief: 9 source ranges from `/home/john/elspeth/src/elspeth/web/frontend/src/App.css`, source-order concatenation, no moves)
**Edits reviewed**: 9 fragment extractions + 1 header comment + 8 inter-fragment blank separators + 1 trailing-orphan stripped
**Overall verdict**: **Approved**

## Intent Fit

Reconstructed concatenation of the 9 declared App.css ranges → `/tmp/concat_expected.css` (576 lines). `diff` against `shared.css` (588 lines):

| Diff hunk | Change | Classification |
|-----------|--------|----------------|
| Lines 1–7 (new) | File header comment naming owners and provenance | Legitimate / desirable |
| Single blank line inserted at 8 fragment seams (concat lines 61, 146, 267, 299, 321, 335, 385, 485 → shared.css lines 69, 155, 277, 310, 333, 348, 399) | Cosmetic separation between fragments | Legitimate cosmetic |
| `/* ---------------------------------------------------------------------------` at concat line 485 stripped | Orphan banner-opener from the *trailing edge of fragment 8* (App.css line 1535 is the opener of the **next** "SuggestionList" block, not part of the side-rail block) | **Correct cleanup** — see Orphan analysis |
| Trailing empty line stripped | Whitespace polish | Cosmetic |

Every rule body inside the diff is byte-identical to the corresponding App.css range. Verified by `diff` returning **only** the four classes of change above.

## Scope Discipline

**Declared scope**: pure concatenation, no moves, preserve cascade order.

**Actually preserved**: all 9 fragments present, in declared source order, byte-identical rule contents.

**Undeclared but defensible changes**:
1. **Added 7-line file-header comment** at top. Documentation; does not alter cascade. Acceptable for a newly-created file even though not strictly "pure concatenation".
2. **Added 8 single blank lines at fragment seams**. Cosmetic; CSS ignores extra whitespace; does not affect cascade.
3. **Stripped 1 orphan comment-opener line** at end of fragment 8 (the unclosed `/* ---` from App.css:1535 which would otherwise dangle without its `--- */` closer because the SuggestionList header was correctly *not* included in the extraction). Without this strip, the file would have an unterminated comment block.

All three are defensible polish around a true concatenation. Surface them so the writer/operator can confirm acceptance.

## Structural Integrity

| Invariant | Status | Evidence |
|-----------|--------|----------|
| Fragment ordering (source-order) | ✓ | `.type-badge` (line 14) → `.status-badge` (73) → `.btn` family (159) → `.side-rail-slot-fill` (271) → `.tab-strip` (281) → `.validation-banner` base (314) → `.empty-state` (337) → `.confirm-dialog*` (352–394) → `.app-layout` (403) → `.banner` (435) → `.side-rail` base (477) → `.validation-banner-*` detailed variants (501–586) |
| `.validation-banner` base before detailed | ✓ | base at line 314 (from 903-924); detailed at lines 501+ (from 5550-5641) — required for cascade |
| `.status-badge-icon--cancelling` inline `@media (prefers-reduced-motion)` preserved | ✓ | shared.css:125 declares `.status-badge-icon--cancelling`; shared.css:130 opens `@media (prefers-reduced-motion: reduce) { ... }` immediately after, byte-identical to App.css:680–693 area |
| Brace balance | ✓ | Visual scan of `@media` block (130–143) shows the nested rule + outer block both close; no dangling braces in the diff |
| Comment-block balance | ✓ | Header `/* … */` closes on line 7; orphan opener at App.css:1535 correctly stripped (would otherwise have produced unterminated comment) |
| Selector text byte-identical | ✓ | Diff hunk list contains zero in-rule differences |
| Declaration order within rules | ✓ | Same evidence (no in-rule diff) |
| Build / lint | Not run | Out of scope for CSS-split fidelity review; recommend `npm run build` and Playwright smoke before merge |

## Cross-Reference / Call-Site Integrity (TSX consumers)

Spot-checked candidates for "moved selectors":

| Selector | TSX consumer files | Status |
|----------|--------------------|--------|
| `.type-badge`, `.type-badge-source/-transform/-gate/-sink/-coalesce/-aggregation` | Present in `GraphView.tsx` etc. | ✓ |
| `.status-badge*` (12 variants) | Present (incl. `statusBadgeAccessibility.test.ts`) | ✓ |
| `.btn`, `.btn-primary`, `.btn-danger`, `.btn-compact`, `.btn-small` | Multiple consumers (HeaderSessionSwitcher, SaveForReviewDialog, ComposerPreferencesPanel, etc.) | ✓ |
| `.tab-strip*` | Present (CatalogDrawer, etc.) | ✓ |
| `.confirm-dialog*` | `ConfirmDialog.tsx` | ✓ |
| `.app-layout`, `.layout-chat`, `.layout-siderail` | `Layout.tsx`, `Layout.test.tsx`, `App.test.tsx` | ✓ |
| `.banner`, `.banner-info`, `.banner-dismiss-btn` | Present | ✓ |
| `.side-rail` base | `SideRail.tsx` | ✓ |
| `.validation-banner-*` detailed (content, summary, checks, warnings, suggestion, component-btn--warning/--error) | `ValidationResult.tsx`, `SideRailValidationBanner.tsx` | ✓ |

**Old-class search**: Not applicable — this is a *split*, not a *rename*. Both `App.css` and `shared.css` should be loaded together for the cutover; the writer/operator must ensure `shared.css` is imported wherever `App.css` was, or that the relevant rules remain in `App.css` until the migration completes. **This file does NOT delete the original from App.css**, so we cannot verify here that downstream import wiring has been updated. Flagging as Information Gap.

## Orphan / Dead-Code / Boundary Findings

**Boundary-by-boundary inspection (each ±5 lines):**

1. **Header / fragment-1 seam (line 7→14)**: Clean — header `*/` closes, blank line, `.type-badge {` starts. No orphan.
2. **Fragment 1 → 2 seam (`.type-badge-coalesce` end → `.status-badge` start, shared.css ~69)**: Clean — added blank line, no broken comment, both rules complete.
3. **Fragment 2 → 3 seam (`.status-badge-empty` → `.btn`, ~155)**: Clean.
4. **Fragment 3 → 4 seam (`.side-rail-slot-fill` → `.tab-strip`, ~277)**: Clean.
5. **Fragment 4 → 5 seam (`.tab-strip-tab-active` → `.validation-banner`, ~310)**: Clean — base `.validation-banner` rule starts a new block.
6. **Fragment 5 → 6 seam (`.validation-banner-fail` → `.empty-state`, ~333)**: Clean.
7. **Fragment 6 → 7 seam (`.empty-state` → `.confirm-dialog-backdrop`, ~348)**: Clean.
8. **Fragment 7 → 8 seam (`.confirm-dialog-btn` → `.app-layout`, ~399)**: Clean.
9. **Fragment 8 → 9 seam (`.side-rail` block end ~498 → ValidationResult comment header ~500)**: **Critical save.** The writer correctly stripped one orphan `/* ---------------------------------------------------------------------------` line that came from App.css:1535 (which is the *opener* of an unrelated SuggestionList section). Had the writer included it verbatim, `shared.css` would have an unterminated comment block consuming everything after line 498. Verified by reading App.css:1533–1537 — line 1535 is `/* ---` of the next section header, not the close of the side-rail section.

**Dead-rule candidates (no TSX consumer; pre-existing in App.css, surfaced for visibility only — out of scope for this split):**

| Selector | TSX consumers found | Note |
|----------|---------------------|------|
| `.side-rail-slot-fill` | 0 | Pre-existing dead in App.css; not introduced by split |
| `.status-badge-empty` | 0 | Pre-existing dead in App.css; not introduced by split |
| `.status-badge-completed-with-failures` | 0 | Pre-existing dead in App.css; not introduced by split |

These are not split-introduced regressions. They warrant a follow-up cleanup ticket but should not block this PR.

## Style / Behaviour Continuity

| Edit | Surrounding style | Inserted style | Match? |
|------|-------------------|----------------|--------|
| Header comment | App.css uses multi-line `/* ----- TITLE ----- */` banners | Writer used `/* shared.css — … */` prose block | Different but standard for a file header; acceptable |
| Inter-fragment blank lines | App.css uses 1 blank line between top-level rules | shared.css uses 1 blank line at seams | ✓ Matches |

**Cascade preservation (the load-bearing property of this split):**
- Within-file cascade order: ✓ source order preserved.
- Cross-file cascade: depends on import order in the entry point — **out of scope of this file-scoped review**. The caller must confirm `shared.css` is imported at the same cascade position the corresponding App.css ranges occupied, or that App.css still contains the ranges as well during the transition.

## Issues Found

### Critical
None.

### Major
1. **Information Gap (not a defect in this file)**: This review only inspects `shared.css` content fidelity. It does not verify that:
   - The 9 ranges have been deleted (or are scheduled to be deleted) from `App.css`. If left in both files, the **detailed validation-banner cascade may double-apply** without issue (idempotent re-declaration) but the `.btn` base rule duplication would cause unnecessary specificity churn in devtools. Confirm before merging.
   - `shared.css` is imported in a location that preserves cascade ordering vs. the rest of `App.css`'s rules (specifically, rules originally between lines 540 and 5550 in App.css must continue to cascade in the same relative order).

### Minor
1. The file-header comment is descriptive but does not cite the *source line ranges* it was composed from. Adding `/* From App.css lines: 480-540, 541-625, 626-746, 747-778, 903-924, 925-938, 939-988, 1437-1535, 5550-5641 */` would make future re-syncs trivially auditable. (Suggestion only.)
2. The three dead rules (`.side-rail-slot-fill`, `.status-badge-empty`, `.status-badge-completed-with-failures`) are pre-existing dead code carried over from App.css. Surface as a follow-up cleanup ticket; do not block this split.

## Out-of-Scope Observations

- **Migration sequencing** (does `App.css` still contain these ranges? is the import wired?) → not a single-file review; needs a multi-file diff at the PR level. Refer to the writer or to a structure-analyst-style review.
- **Visual regression risk** → recommend Playwright snapshot before merge.
- **Pre-existing dead rules** → unrelated to this split; file a tidy-up ticket.

## Confidence Assessment

**Confidence**: **High** on fidelity of the extraction. The diff between expected-concatenation and `shared.css` reduces to exactly 4 categories of intentional polish (header comment, fragment-seam blanks, orphan-opener strip, trailing-newline strip). Zero in-rule edits, zero reordering, zero selector text drift.

**Basis**: Direct `sed`-extraction of each range from `App.css`, concatenation, `diff` against `shared.css`. Verified ordering by grepping selectors. Verified `@media (prefers-reduced-motion)` presence at the correct adjacency. Verified the orphan-stripped line by reading App.css:1533–1537 in situ.

## Risk Assessment

**Residual risks** (not detectable from a file-scoped review):

1. **Double-declaration cascade noise** if `App.css` was not edited to remove the 9 ranges in the same commit. CSS will still render correctly (identical rules), but specificity debugging becomes harder.
2. **Import-order regression** if `shared.css` is imported earlier or later than the original ranges' position in `App.css` — could affect specificity-tied overrides downstream in the original file (lines 540–5550 region).
3. **Visual regression on `.status-badge-icon--cancelling`** if the nested `@media (prefers-reduced-motion)` block has any whitespace-sensitive interaction with subsequent rules — extremely unlikely but warrants Playwright snapshot.

## Information Gaps

- Did not verify deletion of the 9 ranges from `App.css` (requires cross-file diff vs. main).
- Did not verify CSS import wiring at the entry point (`main.tsx` / `App.tsx`).
- Did not run `npm run build` or Playwright snapshots.
- Did not verify that the trailing-newline strip is a deliberate choice vs. an accident; standard POSIX would prefer the file end with newline. **Recommend the writer confirm the file ends in newline.**

## Caveats

- This review establishes **byte-fidelity of the extraction** within the new file; it does **not** establish merge-readiness of the broader CSS-split work program. The PR-level review must additionally check App.css deletion, import wiring, and visual regression.
- The "no moves in/out" claim is verified for this file's contents; verifying the converse (nothing was *also* moved into a different sibling file from these same ranges) requires a sibling-file review.
