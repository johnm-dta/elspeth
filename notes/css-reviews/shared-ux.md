# UX Review: shared.css
**Date:** 2026-05-23
**Reviewer:** UX Critic Agent (lyra-ux-designer)
**File:** `src/elspeth/web/frontend/src/styles/shared.css` (589 lines)
**Status:** Acceptable — strong foundation with four fixable gaps

---

## Summary

The file is well-disciplined in the areas it has been designed to control. Disabled-button contrast is tokenised with a measured rationale (comment at line 184), badge backgrounds use the correct rgba-on-token layering approach documented in team comments, and dialog semantics are complete (role="alertdialog", aria-modal, aria-labelledby, aria-describedby, Escape key, focus-trap, backdrop click to cancel). No z-index magic numbers are present. The `.side-rail-slot-fill` class resolves to a dead rule in TSX consumers (sidebar.css replicates its effect via a comma-grouped selector on the concrete button names) but is not harmful. Four issues require remediation before the next release.

---

## Visual Design

**Strengths:**
- Disabled button uses tokenised `--color-bg` / `--color-text-muted` pair rather than `opacity:0.4`, and the rationale (4.78:1 dark / 6.32:1 light) is cited in the comment. Verified by `colorContrast.test.ts` lines 186–208.
- `--color-btn-primary-bg` and `--color-btn-danger-bg` both use `--color-text-inverse` and are verified WCAG AA in both themes by `colorContrast.test.ts` lines 119–137.
- `box-shadow: 0 8px 32px rgba(0,0,0,0.25)` on `.confirm-dialog` (line 371) and `rgba(0,0,0,0.45)` backdrop (line 355) are the two legitimately hardcoded rgba values: they are decoration / scrim, not semantic colour, so they do not need tokens. Acceptable.

**Issues:**

| Issue | Severity | Location | Fix |
|-------|----------|----------|-----|
| Status-badge backgrounds are hardcoded `rgba()` values (8 occurrences, lines 86–151) while the foreground text uses design tokens. In high-contrast or forced-colors mode these alpha backgrounds collapse unpredictably — themes.css has no `.status-badge-*` override block. | Major | Lines 86, 91, 96, 101, 106, 116, 146, 151 | Introduce `--color-status-<name>-bg` tokens in tokens.css (parallel to the existing `--color-badge-*-bg` token pattern) and replace all raw `rgba()` values. Add forced-colors override block in themes.css (parallel to the `.type-badge-*` block at line 46). |
| `.btn-primary` and `.btn-danger` have no `:active` state (lines 237–257). The base `.btn` has no `:active` state either. Without a depressed visual, users on touch screens or accessibility input devices get no tactile confirmation. | Minor | Lines 237–257 | Add `&:active:not(:disabled) { filter: brightness(0.9); }` or equivalent tokenised colour pair to both. |

---

## Information Architecture

**Strengths:**
- `.side-rail-slot:empty { display: none; }` (line 489) correctly collapses unused slots, preventing visual ghost-space in the rail.
- `.side-rail-validation-banner` is logically grouped with the rail section, not buried in a component-specific file.

**Issues:**

| Issue | Severity | Location | Fix |
|-------|----------|----------|-----|
| `.side-rail-slot-fill` (lines 271–275) is a dead class: `sidebar.css` lines 88–93 replicates the exact same width+margin rule as a comma-grouped selector on the concrete element names (`.side-rail-execute-btn, .side-rail-export-yaml-btn`). No TSX file applies `.side-rail-slot-fill` directly. The class comment instructs "do not re-declare width or margin — compose this class instead," which contradicts sidebar.css's actual behaviour. | Minor | Lines 271–275 | Either (a) delete `.side-rail-slot-fill` and update the comment in sidebar.css to own the rule, or (b) migrate sidebar.css to compose `.side-rail-slot-fill` as originally intended. Option (b) is cleaner; option (a) is consistent with the no-legacy-code policy. |
| `.app-layout--overlay` (lines 413–415) is never applied in any TSX file. The only Layout consumer (`Layout.tsx`) renders a fixed two-column grid with no conditional class. | Minor | Lines 413–415 | Confirm whether overlay mode is planned (if so, add a TODO issue); if not planned, delete the rule. |

---

## Interaction Design

**Strengths:**
- Base `.btn` has `min-height: var(--size-control)` tokenised to 44px, verified by `colorContrast.test.ts` line 221.
- `.btn-compact` has `min-height: var(--size-control-compact)` tokenised to ≥32px, satisfying WCAG 2.5.8 AA.
- `.confirm-dialog-btn` sets `min-height: var(--size-control)` (line 397), giving dialog buttons the full 44px touch target.
- Transitions use `var(--transition-fast)` throughout — no raw `ms` literals.

**Issues:**

| Issue | Severity | Location | Fix |
|-------|----------|----------|-----|
| `.btn`, `.btn-compact`, `.btn-primary`, and `.btn-danger` have no `:focus-visible` rule. The global `base.css` rule (`:focus-visible { outline: 2px solid var(--color-focus-ring); outline-offset: 2px; }`) provides a default, but `.btn-primary` and `.btn-danger` use high-saturation filled backgrounds where the default `--color-focus-ring` may provide insufficient 3:1 contrast against the button surface rather than the page background — WCAG 1.4.11 Non-text Contrast requires 3:1 between the focus indicator and its adjacent colours. | Major | Lines 237–257 | Add explicit `:focus-visible` on `.btn-primary` and `.btn-danger` with `outline-offset: 3px` (enough clearance from the filled surface) or a contrasting `outline-color`. Add a `colorContrast.test.ts` assertion that checks focus-ring contrast against `--color-btn-primary-bg` and `--color-btn-danger-bg`. |
| No `@media (hover: none)` block anywhere in shared.css. The `:hover` rules on `.btn`, `.btn-compact`, `.tab-strip-tab`, `.banner-dismiss-btn`, `.validation-banner-component-btn`, etc. do not fall through to a touch-equivalent active state. On mobile/tablet, sticky hover states can appear after tap and persist visually. | Minor | Lines 179, 224, 243, 254, 301 | Add `@media (hover: none) and (pointer: coarse)` block that resets hover `background-color` overrides and optionally adds a `:active` rule for tap feedback. |

---

## Accessibility

**Quick Check:**

- [x] 1.4.3 Contrast (text): Pass — tokenised pairs with CI-gate tests in `colorContrast.test.ts`
- [ ] 1.4.11 Non-text Contrast (focus indicator on filled buttons): Needs verification — see Interaction Design finding
- [ ] 1.4.11 Non-text Contrast (status badges in forced-colors): Fail — no forced-colors override for `.status-badge-*`
- [x] 2.1.1 Keyboard nav: Pass — focus trap in ConfirmDialog, Escape key, backdrop dismiss
- [x] 2.4.7 Focus Visible: Pass (base.css global rule covers all elements)
- [x] 1.1.1 Alt Text: N/A — CSS file
- [x] 3.3.1 Error Identification: Pass — `.validation-banner-fail` uses `role="alert"` in ValidationResult.tsx (line 120), `.validation-banner-pass` uses `role="status"` (line 50)

**Additional a11y notes:**

- `statusBadgeAccessibility.test.ts` correctly asserts that status badges do not inject symbols via CSS `::before content:` — icon presence is a component responsibility, not a CSS responsibility. This is the right separation. The test verifies the negative case only; a positive assertion that icon glyphs are present in consumer TSX would complete the contract (out of scope for this CSS review).
- `themes.css` `@media (prefers-contrast: more)` correctly widens `.type-badge` border to 2px and increases focus outline to 3px — good progressive enhancement. Same block should be extended to cover `.status-badge` border width.
- `.validation-banner-checks` (line 523) hardcodes `color: var(--color-success)` — this is correct for the pass state but means the check list is semantically styled with success colour even for failed checks within a pass banner (`check.passed ? "✓" : "✗"`). Minor semantic mismatch; doesn't affect accessibility.

---

## Platform-Specific Notes

**Mobile / Touch:** No `@media (hover: none)` handling (see Interaction Design finding). `.btn-small` (line 260–264) does not set a `min-height`, inheriting from the parent `.btn` rule only if composed correctly — consumers using `.btn.btn-small` get the 44px floor, but the small variant visually communicates "small" while maintaining the touch target. This is an intended pattern; the comment at line 259 acknowledges it.

**Forced colors (Windows High Contrast):** `.status-badge-*` lacks a `forced-color-adjust` or border override in themes.css, meaning the rgba backgrounds collapse to the user's Canvas colour and the badges become visually flat. Only `.type-badge-*` has this override (themes.css lines 46–55).

---

## Priority Recommendations

**Major (Fix Before Launch):**
1. **Status badge backgrounds — tokenise rgba and add forced-colors override.** Replace the 8 hardcoded `rgba()` values (lines 86, 91, 96, 101, 106, 116, 146, 151) with `--color-status-<name>-bg` tokens. Add `.status-badge-*` block to the `@media (forced-colors: active)` section in themes.css. This matches the existing pattern for `.type-badge-*`.
2. **Explicit focus-visible on filled buttons.** Add `:focus-visible` rules to `.btn-primary` and `.btn-danger` with `outline-offset: 3px` to ensure the focus ring is legible against the filled button surface rather than relying solely on the page-background assumption in `base.css`. Back with a contrast assertion in `colorContrast.test.ts`.

**Minor (Improvement Before Next Review):**
3. **Add `:active` state to `.btn-primary` and `.btn-danger`.** A depressed-state visual (brightness or scale) closes the interaction loop for touch and mouse users.
4. **Resolve `.side-rail-slot-fill` dead-class ambiguity.** Either delete and let `sidebar.css` own the rule, or migrate `sidebar.css` to compose it. The conflicting comment is a maintenance trap.
5. **Add `@media (hover: none)` block.** Reset sticky hover states and provide a touch `:active` alternative.
6. **Confirm or delete `.app-layout--overlay`.** Zero TSX consumers found; the rule is either forward-declared (add a TODO) or dead (delete it).

---

## Confidence Assessment

**High (85%)** on findings 1–4 — directly evidenced by CSS source + consumer grep. Finding 1 (status badge forced-colors) is verified by cross-referencing themes.css. Finding 2 (focus-visible on filled buttons) is a structural gap confirmed by absence; actual contrast ratio against the resolved token values cannot be computed without the token hex values from tokens.css (not read in this session), so a small residual uncertainty remains.

**Medium (70%)** on finding 5 (hover:none) — the absence of a media query is confirmed, but whether the current `:hover` rules actually produce sticky-hover artefacts in production depends on the browser/device matrix tested.

## Risk Assessment

**Critical path risk: None.** All four major issues are bounded to visual/interaction layer with no data integrity or pipeline-logic impact. The status-badge forced-colors gap affects users on Windows High Contrast Mode; this is a WCAG A-adjacent failure (forced-colors is not formally a WCAG success criterion but is required under EN 301 549 clause 9).

## Information Gaps

- Token hex values for `--color-btn-primary-bg`, `--color-btn-danger-bg`, and `--color-focus-ring` were not resolved (would require reading `tokens.css`). The focus-ring contrast assertion above is structural rather than numerically verified.
- Whether `.app-layout--overlay` is a planned-but-unimplemented feature or dead code is not determinable from CSS alone — requires checking git history or open issues.

## Caveats

This review covers `shared.css` in isolation. Component-level CSS files (sidebar.css, execution.css, etc.) were read only where needed to confirm consumer patterns. A full audit of interaction between shared.css and component overrides is out of scope here.
