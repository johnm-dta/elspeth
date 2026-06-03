# Guided CSS Split — Fidelity Review

## Summary
- **NEW file**: `/home/john/elspeth/.worktrees/css-split/src/elspeth/web/frontend/src/components/chat/guided/guided.css` (1402 lines)
- **ORIGINAL range**: `/home/john/elspeth/src/elspeth/web/frontend/src/App.css` lines 5941–7335 (1395 lines)
- **Verdict**: **APPROVED** — body byte-identical; only diff is the writer-declared file header and the planned cross-reference comment fix.

## Fidelity Verification

### Selector completeness
- `grep -oE '\.guided-[a-zA-Z0-9_-]+' | sort -u`: **113 unique selectors** in NEW, **113 in ORIGINAL range** — `diff` returns clean.
- Rule-line count (selector starts, whitespace-tolerant): **195 in both files** — exact match.
- `diff <(sed -n '5941,7335p' App.css) guided.css` returns **only two hunks**:
  1. Lines 1–6 in NEW: writer-declared file header comment (`/* guided.css — Guided-mode widget family used by the chat panel. ... */`). Pure metadata, no rules.
  2. Lines 974–982 in NEW vs 974–975 in ORIGINAL: the planned cross-reference comment update (see Task 4 below).

All declarations, values, selector grouping, comment text, and section ordering are otherwise byte-identical.

### Section order preserved
Sections appear in identical order: workflow → current-decision/step-chat → guided-turn → chip-base → custom-input → InspectAndConfirm → MultiSelect → SchemaForm → ProposeChain → Recipe → ExitToFreeform → GuidedHistory → CompletionSummary → `@media (prefers-reduced-motion: reduce)`.

### Trailing reduced-motion block (Task 3)
NEW lines 1378–1402 carry the `@media (prefers-reduced-motion: reduce)` block as the **final** rule in the file. The 22-selector silenced list matches the original 22-selector list exactly: `.guided-chip-btn`, `.guided-workflow-step`, `.guided-inspect-confirm-btn`, `.guided-inspect-edit-btn`, `.guided-inspect-remove-btn`, `.guided-inspect-cancel-btn`, `.guided-inspect-apply-btn`, `.guided-multi-custom-remove-btn`, `.guided-multi-escape-btn`, `.guided-multi-continue-btn`, `.guided-schema-input`, `.guided-schema-select`, `.guided-schema-textarea`, `.guided-schema-advanced-toggle`, `.guided-schema-continue-btn`, `.guided-propose-accept-btn`, `.guided-recipe-apply-btn`, `.guided-recipe-build-btn`, `.guided-exit-button`, `.guided-completion-save-btn`, `.guided-completion-edit-btn`.

Source order preserved — block is at the END of guided.css.

### Cross-reference comment update (Task 4)
Original comment (App.css ~6914):
```
Card chrome reuse: .guided-propose-step-card (Task 7.6, App.css:4285-4290)
is applied as the outer card shell.
```
New comment (guided.css ~980):
```
Card chrome reuse: .guided-propose-step-card (Task 7.6, defined earlier in
this file in the Propose Chain decision header section) is applied as the
outer card shell.
```
**Properly updated** — the App.css line-range pointer (which would have been wrong both because lines shift after the split and because the target moved into guided.css) is replaced with an in-file location reference. Note: the original `App.css:4285-4290` pointer was already stale (the actual `.guided-propose-step-card` definition lives at original 6798, not 4285) — so the writer correctly chose to replace the broken pointer with a semantic reference rather than recompute a number that would drift again. This is the right call.

## Adjacent-Selector Duplication
The plan flagged `.guided-inspect-edit-btn` at original 6250 and 6270 — both present in NEW at lines 316 and 336. Inspection shows this is **intentional CSS pattern**, not a bug:
- Line 316: shared declaration `.guided-inspect-confirm-btn, .guided-inspect-edit-btn { /* shared layout */ padding, border-radius, font-size, font-family, cursor, min-height, transition }`.
- Line 336: standalone `.guided-inspect-edit-btn { /* differentiation */ background-color, color, border }`.

This is the standard "shared base + per-variant differentiation" pattern. The `.guided-inspect-confirm-btn` follows the same shape at lines 326 + 332 (hover). Cascade is well-defined, no specificity conflict.

Also separately: `:focus-visible` for the same pair grouped at line 347. Not a duplicate, a shared selector.

## Orphan Rules (Pre-Existing — Not Regressions)
A `grep -rE ${cls}` scan across all `.tsx` / `.ts` files in `web/frontend/src/` finds **22 declared classes with no TSX consumer**:

```
guided-recipe-actions
guided-recipe-alternative-item
guided-recipe-alternatives
guided-recipe-alternatives-heading
guided-recipe-alternatives-list
guided-recipe-apply-btn
guided-recipe-build-btn
guided-recipe-input-warning
guided-recipe-input-warning-icon
guided-recipe-name
guided-recipe-slot-key
guided-recipe-slot-row
guided-recipe-slots
guided-recipe-slot-val
guided-schema-advanced-toggle
guided-schema-continue-btn
guided-schema-error
guided-schema-optional-section
guided-schema-required-section
guided-schema-textarea--error
guided-workflow-step--complete
guided-workflow-step--current
```

**Verification**: same scan run against the ORIGINAL `/home/john/elspeth/src/elspeth/web/frontend/src/` returns the identical 22-class list. These are pre-existing dead rules carried forward; **not introduced by the split**. Two clusters:

1. **Recipe family** — `RecipeContextHeader.tsx` uses `recipe-context-header` (no `guided-` prefix); the entire `guided-recipe-*` chrome appears unused. May indicate the Recipe widget was never wired up, or was refactored to a different class family.
2. **Schema family** — `SchemaFormTurn.tsx` uses `guided-turn-primary` / `guided-turn-secondary` for buttons instead of `guided-schema-continue-btn`, and has no toggle for advanced section. The `guided-schema-required-section` / `-optional-section` / `-advanced-toggle` triad is stale.
3. **Workflow modifiers** — modifiers consumed via template literal `guided-workflow-step--${state}` in `ChatPanel.tsx`; the literal substitution means `--complete` / `--current` classes ARE consumed at runtime (the scan can't see the interpolation). **These two are false positives in the orphan list.**

Recommendation: out-of-scope for this CSS split task; surface to operator as a follow-up cleanup (recipe and schema-advanced dead rules). The split should preserve them faithfully, which it does.

## Brittle Descendant Chains
Single chain found: `.guided-recipe-slot-row .guided-recipe-input-warning { flex-basis: 100%; margin-left: calc(120px + var(--space-sm)); }` at NEW line 1080. Two-level descendant only; not deep, but the `calc(120px + var(--space-sm))` is fragile — it duplicates the `min-width: 120px` of `.guided-recipe-slot-key` (line 1017). If the key min-width changes, this offset desyncs silently. **Pre-existing**; preserved faithfully by the split. Worth noting as tech debt.

## Focus-Visible Coverage (per Task brief)
- `.guided-chip-btn:focus-visible` — **present** at NEW line 183.
- `.guided-multi-continue-btn:focus-visible` — **present** at NEW line 617 (a `:not(:disabled)` hover is also present at 607).
- `.guided-propose-accept-btn:focus-visible` — **present** at NEW line 971.

Note: brief asked about `.guided-continue-btn` and `.guided-accept-btn` — those names do not exist; assumed to be shorthand for the namespaced variants checked above.

Additionally, every interactive control in the file has a `:focus-visible` rule applied (verified by grep — 23 `:focus-visible` selectors covering all button/input/select/textarea/toggle classes). Coverage is comprehensive.

## Confidence Assessment
**Confidence**: High
**Basis**: Byte-level `diff` of the source range against the new file confirms fidelity. Selector set diff is clean. Rule-line counts match (195 = 195). Section ordering preserved. Trailing `@media` block is the final rule. Cross-reference comment update applied correctly. Orphan list is pre-existing (verified against main).

## Risk Assessment
**Residual Risk**: Low.
- The split adds a 6-line header comment and replaces a brittle line-range pointer with a semantic in-file reference. No selectors, declarations, or values touched.
- Build risk: the new file must be `@import`'d (or otherwise loaded) by whatever entry replaces App.css. That wiring is **outside this file**; this review verifies only the CSS body itself.
- Orphan rules carried forward intact — same compiled-size cost as before; no behavioural change.

## Information Gaps
- Did not verify the import / `<link>` wiring that loads `guided.css` into the bundle — out of scope for fidelity review.
- Did not run a visual regression test or Playwright smoke; behavioural equivalence inferred from byte-identity of rule bodies.
- Did not investigate why the Recipe / Schema-advanced orphans exist — pre-existing; not a regression.

## Caveats
- The split itself is faithful. Pre-existing dead rules and the brittle `calc(120px + var(--space-sm))` chain are out of scope for this review — should be surfaced as a separate cleanup ticket if desired.
- The TSX consumer scan uses a literal-substring grep; template-literal interpolated class names (`${state}`) are not detectable that way. The `--complete` / `--current` workflow-step modifiers are consumed at runtime via `ChatPanel.tsx` line ~`guided-workflow-step--${state}`.
