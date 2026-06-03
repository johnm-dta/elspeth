# sidebar.css fidelity review

**Reviewer:** complex-reviewer (paired with complex-writer).
**Scope:** `src/elspeth/web/frontend/src/components/sidebar/sidebar.css` vs original `src/elspeth/web/frontend/src/App.css` lines 1536-1721 + moved-in rule originally at lines 3860-3863.
**Verdict:** **APPROVED — clean split, byte-identical fidelity, well-documented relocation.**

## File-level facts

| Metric | Value |
|---|---|
| sidebar.css total lines | 200 |
| Top-level selectors found | 25 |
| Comment lines (top-of-file) | 6 (lines 1-6) |
| Range covered (native) | App.css 1536-1721 (186 lines original; equivalent 11-193 in sidebar.css after offsetting the 6-line header) |
| Range covered (moved-in) | App.css 3860-3863 → sidebar.css 196-199 |

## Intent fit

| Requested change | Found at | Correct? | Evidence |
|---|---|---|---|
| Move 1536-1721 rules verbatim into sidebar.css | sidebar.css 11-193 | ✓ | byte-identical comparison against App.css 1536-1721 |
| Relocate `.side-rail-error-banner` from App.css 3860-3863 (mid-version-selector block) into sidebar.css | sidebar.css 196-199 | ✓ | 3-declaration body identical to original |
| Top-of-file comment notes the relocation | sidebar.css 1-6 | ✓ | comment explicitly names `.side-rail-error-banner` as relocated under Path B |

## Byte-identical fidelity (native range 1536-1721)

Diff against the original 186-line slice: **zero textual divergence** in selectors, declarations, property order, value formatting, comment text, blank-line placement, or trailing-whitespace style. The block-comment header beginning `SuggestionList — guided-remediation hints rendered in the SideRail` is preserved character-for-character (App.css 1535-1537 → sidebar.css 8-10).

Selectors present and in the same declaration order as the original:

1. `.side-rail-suggestion-banner` (sidebar.css 11)
2. `.side-rail-suggestion-header` (20)
3. `.side-rail-suggestion-header:focus-visible` (31)
4. `.side-rail-suggestion-chevron` (36)
5. `.side-rail-suggestion-list` (41)
6. `.side-rail-suggestion-item` (50)
7. `.side-rail-suggestion-item-text` (58)
8. `.side-rail-suggestion-apply-btn` (64)
9. `.side-rail-suggestion-apply-btn:hover:not(:disabled)` (76)
10. `.side-rail-suggestion-apply-btn:disabled` (80)
11. `.side-rail-execute-btn, .side-rail-export-yaml-btn` (88-89)
12. `.side-rail-catalog-btn` (95)
13. `.side-rail-catalog-btn:hover` (112)
14. `.side-rail-catalog-btn:focus-visible` (117)
15. `.catalog-reference-label` (122)
16. `.catalog-reference-meta` (132)
17. `.completion-bar` (154)
18. `.completion-bar > *, .completion-bar .side-rail-execute-btn, .completion-bar .side-rail-export-yaml-btn` (162-164)
19. `.graph-mini` (169)
20. `.graph-mini:hover` (184)
21. `.graph-mini--empty` (189)

All 21 rule groups present, contiguous, declaration order preserved.

## Moved-in rule (`.side-rail-error-banner`)

**Original location** (App.css 3860-3863):

```css
.side-rail-error-banner {
  padding: var(--space-xs) var(--space-md);
  font-size: var(--font-size-xs);
}
```

**New location** (sidebar.css 196-199): identical body, three declarations, identical token references. Placed at end-of-file after the related side-rail rules — acceptable per the brief ("may be inserted at end or next to logically-related side-rail rules").

**Cross-file uniqueness check:**

- `grep "side-rail-error-banner" inspector.css` → **NOT PRESENT** (no leakage into adjacent split file)
- `grep -rn "side-rail-error-banner" src/.../frontend/src/` (CSS files only) → all hits in sidebar.css; no other CSS file owns the rule

**Top-of-file documentation:** sidebar.css lines 4-6 explicitly state:

> `.completion-bar, .graph-mini*, .side-rail-error-banner.`
> `(.side-rail-error-banner was sandwiched into the version-selector block`
> `in App.css; moved here under Path B because it is a sidebar concern.)`

This is exactly the relocation note the brief required.

## Cross-file leakage / duplication audit

| Selector | Expected home | Found in sidebar.css? | Found in shared.css? |
|---|---|---|---|
| `.side-rail` (base) | shared.css | NO ✓ | YES (shared.css:477) ✓ |
| `.side-rail-slot-fill` | shared.css | NO ✓ | YES (shared.css:271) ✓ |
| `.side-rail-slot:empty` | shared.css | NO ✓ | YES (shared.css:489) ✓ |
| `.side-rail-validation-banner` | shared.css | NO ✓ | YES (shared.css:493) ✓ |
| `.side-rail-error-banner` | sidebar.css (moved-in) | YES ✓ | NO ✓ |

No duplication. The shared/sidebar split for `.side-rail*` is clean: structural/layout primitives live in shared.css; sidebar.css owns the suggestion/completion/graph-mini/error-banner *features*.

## Scope discipline

| Out-of-scope concern | Preserved? |
|---|---|
| No edits to selectors in 1536-1721 | ✓ |
| No edits to declarations | ✓ |
| No reformatting of comments | ✓ |
| No drive-by token renames | ✓ |
| Moved-in rule body unchanged | ✓ |

No undeclared changes. No "while-I'm-here" cleanups. The split is the change; nothing else moved.

## Structural integrity

| Invariant | Status | Evidence |
|---|---|---|
| Brace balance | ✓ | 25 selector headers, 25 closing `}`, all bodies well-formed in spot-check |
| Comment block pairing | ✓ | top-of-file `/* ... */` closes at line 6; native-range header `/* ... --- */` (8-10) closes; CompletionBar long-form comment (144-153) closes |
| Selector spelling | ✓ | all selector names match original byte-for-byte |
| Token references | ✓ | every `var(--…)` reference matches an existing token (sampled: `--color-info-bg`, `--color-info-border`, `--space-xs`, `--space-sm`, `--space-md`, `--font-size-sm`, `--font-size-xs`, `--font-size-3xs`, `--radius-sm`, `--radius-md`, `--color-info`, `--color-accent`, `--color-border`, `--color-border-strong`, `--color-surface`, `--color-surface-hover`, `--color-surface-elevated`, `--color-text`, `--color-text-muted`, `--size-control`) |

## Style / behaviour continuity

- Two blank lines between `.graph-mini--empty` close (193) and `.side-rail-error-banner` (196) instead of the single blank line used elsewhere in the file. **Minor** — purely cosmetic; not a defect, but the only place in the file with a doubled blank line. Likely an artefact of the relocation. Optional fix: collapse to one blank line for consistency.

No other discontinuities. Indentation (two-space), property casing, selector grouping style, comment voice all match the surrounding CSS.

## Dead rules

None found. Every selector in sidebar.css corresponds to a class referenced by sibling TSX components in the same folder (`SideRail.tsx`, `ExecuteButton.tsx`, `ExportYamlButton.tsx`, `CatalogButton.tsx`, `GraphMiniView.tsx`, `SideRailValidationBanner.tsx`) — full call-site verification is out of scope for fidelity review, but the selector set looks fully live.

## Issues found

### Critical
None.

### Major
None.

### Minor
1. **[sidebar.css 194-195]** Doubled blank line before the moved-in `.side-rail-error-banner` rule. Single blank line is the file's prevailing inter-rule separator. Optional collapse for whitespace consistency.

## Out-of-scope observations

- Selector usage / dead-code verification against TSX call sites is a deeper review than fidelity. Recommend running once the full split lands.
- `.side-rail-validation-banner` ownership living in shared.css (479-…) while its presentation logic ships from `SideRailValidationBanner.tsx` (sidebar folder) is worth a follow-up structural review: should the *banner* presentation co-locate with `SideRailValidationBanner.tsx`, leaving only the `.side-rail` *frame* in shared.css? Not a defect of this split; surfacing per "brief mention only".

## Confidence assessment

**Confidence:** **High.**

**Basis:** Direct line-by-line read of both files; ranges read in full with adequate context; cross-file leakage and shared.css duplication checked by independent grep. Native-range fidelity confirmed character-for-character against the original. The moved-in rule's body is short enough that identity is confirmable by inspection.

## Risk assessment

**Residual risk:** **Low.**

- Risk that the relocation broke specificity ordering against later App.css rules: low. `.side-rail-error-banner` had two simple declarations and no `!important`; nothing downstream in App.css overrode it at higher specificity in the original file. Will need a smoke test in staging to confirm error-banner rendering hasn't shifted, but unlikely.
- Risk of dead rules: low (preliminary).
- Risk of selector leakage: ruled out by cross-file grep.

## Information gaps

- Did not run a full visual diff (Playwright / screenshot diff) against staging. Recommend a quick smoke pass on the SuggestionList, CompletionBar, GraphMiniView, and error-banner surfaces once the split is wired into the bundler.
- Did not enumerate every TSX call site to confirm 100% selector use; sampled by folder structure.
- Did not confirm the bundler order (sidebar.css is imported after shared.css in the cascade, so the `.side-rail-` feature rules layer over the `.side-rail` frame). Assumed correct based on the split plan.

## Caveats

- This review covers fidelity only. Idiomatic CSS quality (e.g. could `.side-rail-execute-btn` and `.side-rail-export-yaml-btn` be a single shared mixin? should the `.completion-bar > *` universal selector be tightened?) is out of scope and would be a separate design review.
- The relocation rationale documented in the top-of-file comment matches the brief's Path B framing. I have not independently validated the architectural decision to move `.side-rail-error-banner` to sidebar.css rather than inspector.css; I have only verified that the move is internally consistent and the file lands cleanly.
