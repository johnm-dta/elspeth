# base.css — Complex Reviewer Fidelity Report

**Reviewer**: complex-reviewer (CSS-split fidelity)
**Date**: 2026-05-23
**Scope**: Verify `src/elspeth/web/frontend/src/styles/base.css` faithfully extracts lines 210–335 of the original `src/elspeth/web/frontend/src/App.css` (Reset, html/body/#root, code/pre typography, scrollbar, focus-visible, skip-link, sr-only).
**Adjacency contract**: No selectors moved in or out.

---

## Summary

**File**: `/home/john/elspeth/.worktrees/css-split/src/elspeth/web/frontend/src/styles/base.css` (130 lines, type: CSS)
**Original**: `/home/john/elspeth/src/elspeth/web/frontend/src/App.css` (lines 210–335)
**Edit report received**: Yes (caller brief + `notes/app-css-split-plan.md`)
**Rules in expected range**: 16 (incl. comment blocks)
**Overall verdict**: **Approved — clean lift-and-shift.**

A `diff` of `App.css:210–335` against `base.css:4–129` returns no differences. The new file adds only a two-line top-of-file purpose comment (lines 1–2) above an otherwise byte-identical copy of the source range. Trailing newline preserved (`...0;\n}\n`).

---

## Intent Fit

| Requested Change | Found in File | Correct? | Evidence |
|------------------|---------------|----------|----------|
| Move Reset & Base block | base.css:4–32 | ✓ | matches App.css:210–238 |
| Move Typography Code/Mono block | base.css:34–51 | ✓ | matches App.css:240–257 |
| Move Scrollbar Styling block (incl. Firefox `* { scrollbar-* }`) | base.css:53–78 | ✓ | matches App.css:259–284; **Firefox `*` rule present at base.css:74–78** |
| Move Focus-Visible block | base.css:80–97 | ✓ | matches App.css:286–303 |
| Move `.skip-to-content` + `:focus` variant | base.css:99–116 | ✓ | matches App.css:305–322 |
| Move `.sr-only` utility | base.css:118–129 | ✓ | matches App.css:324–335 |
| Add file-purpose header comment | base.css:1–2 | ✓ | new, non-functional |

Verification: `diff <(sed -n '210,335p' App.css) <(sed -n '4,129p' base.css)` → empty (no differences).

## Scope Discipline

**Declared out-of-scope**: any rule outside lines 210–335; no selectors moved in or out.
**Actually preserved**: ✓
**Undeclared changes**: One non-load-bearing addition — a 2-line purpose comment at the top of `base.css`. Acceptable for split-file orientation; flagged for transparency, not as a defect.

## Structural Integrity

| Invariant | Status | Evidence |
|-----------|--------|----------|
| Selector text byte-identical | ✓ | diff empty |
| Declaration order preserved | ✓ | diff empty |
| Property values byte-identical | ✓ | diff empty |
| Comment-block dividers preserved | ✓ | All four `/* --- ... --- */` section banners carried across |
| Brace balance | ✓ | 16 opening + 16 closing braces in base.css body |
| Custom-property references (`var(--…)`) intact | ✓ | All 19 var() refs present; none renamed |
| Trailing newline at EOF | ✓ | `...0;\n}\n` |
| Firefox scrollbar `* { scrollbar-width / scrollbar-color }` present | ✓ | base.css:74–78 |

## Cross-Reference / Call-Site Integrity

Within the lifted range, no cross-file references exist beyond `var(--…)` custom-property lookups, which are resolved against the consolidated tokens file (out of scope here). Spot-checked tokens used by this slice — `--color-bg`, `--color-text`, `--font-sans`, `--font-size-base`, `--line-height-normal`, `--font-mono`, `--font-size-sm`, `--line-height-relaxed`, `--color-scrollbar-track`, `--color-scrollbar-thumb`, `--color-scrollbar-thumb-hover`, `--radius-sm`, `--color-focus-ring`, `--z-skip-link`, `--space-sm`, `--space-lg`, `--color-accent`, `--color-text-inverse`, `--radius-md` — all are referenced unchanged. Whether each is defined in the eventual tokens file is a separate review's concern.

External consumers of `.skip-to-content`, `.sr-only`, `.mono`, and `#root` are not modified by this extraction; class names and id selectors are unchanged.

## Orphan / Dead-Code / Boundary Findings

Inspection of edit boundaries:

- **base.css head (lines 1–6)**: Clean. Header comment + section banner; no orphaned text from any upstream/adjacent block.
- **base.css tail (lines 127–129)**: Clean. `.sr-only` closes cleanly; no orphan `}`, no trailing fragment from the deleted keyframes section that immediately follows in the original (App.css:337+).
- **Internal section seams** (Reset→Typography, Typography→Scrollbar, Scrollbar→Focus, Focus→Skip-link, Skip-link→sr-only): All preserved verbatim. No transitional comment text was dropped.

No dead rules within the slice — every selector is plausibly referenced by the React app (`.skip-to-content` in AppShell, `.sr-only` widely, `.mono` in code surfaces, scrollbar/focus pseudo-elements global).

## Style / Behavior Continuity

| Edit | Surrounding Style | Inserted Style | Match? | Behavior preserved? |
|------|-------------------|----------------|--------|----------------------|
| Section banners | `/* --- … --- */` 76-col rules | identical | ✓ | ✓ |
| Indentation | 2-space | 2-space | ✓ | ✓ |
| Property casing / hyphenation | kebab-case | kebab-case | ✓ | ✓ |
| Em-dash usage in comments | present (e.g. "Reset & Base") | present | ✓ | ✓ |

CSS specificity, cascade ordering within the slice, and pseudo-element/pseudo-class selectors are byte-identical → **no behavioral change.** When the eventual `index.css` or App.css loads `base.css` in this slice's original position, runtime cascade is preserved.

## Issues Found

### Critical
None.

### Major
None.

### Minor / Observations (not defects in this slice; surfaced per brief)

1. **Brittle universal selector — `* { scrollbar-width / scrollbar-color }` (base.css:74–78).** This is a pre-existing pattern carried over verbatim from the original, not introduced by the split. The `*` selector applies Firefox scrollbar styling to *every* element, which is the documented Firefox idiom (scrollbar styling is inherited; `html` would suffice for most cases but is functionally equivalent here). Not a fidelity defect. If a future tokens/refactor cycle wants to tighten scope, narrowing to `html` (or `:root, body`) is mechanically safe — but that is **out of scope** for a lift-and-shift split. Recommend filing as a separate refactor candidate if desired, not as part of this PR.

2. **No `forced-colors` (Windows High Contrast) fallback for `.skip-to-content` or `:focus-visible`.** Pre-existing accessibility gap, not introduced by this split. In forced-colors mode, `outline: 2px solid var(--color-focus-ring)` resolves to the system highlight colour (acceptable), but `.skip-to-content`'s `background-color: var(--color-accent)` + `color: var(--color-text-inverse)` may render with system-mandated colours that lose the visual signal. A `@media (forced-colors: active)` block forcing `background-color: Canvas; color: CanvasText; outline: 2px solid Highlight;` on `.skip-to-content:focus` would close this. **Not a fidelity defect**; surface to the operator as a separate a11y ticket candidate. Filing it here would scope-creep the split.

3. **No zombie / dead rules detected** in the moved slice. The seven rule groups all have plausible runtime consumers in the React tree. Confirming actual usage of `.mono` (defined in the Typography block) would require a frontend grep, but absence of usage there is again pre-existing, not split-induced.

4. **Header comment cosmetics (base.css:1–2).** Minor: the comment ends `…and .sr-only utility.` and could optionally list `.mono` for completeness; trivial.

## Out-of-Scope Observations
- For absolute CSS style/idiom quality (universal selectors, modern `:where()`/`:is()` adoption, `@layer` adoption), refer to a CSS-specific reviewer.
- For accessibility deep-dive (forced-colors, prefers-reduced-motion adjacent to focus animations, contrast tokens), refer to `lyra-ux-designer:accessibility-audit`.
- Tokens definition completeness (do all 19 `var(--…)` references resolve in the consolidated tokens file?) is a separate slice's review.

## Confidence Assessment
**Confidence**: High.
**Basis**: Direct `diff` of source slice vs new file produced empty output, confirming byte-identical selector text, declaration order, property values, and inter-rule whitespace. Brace balance and trailing newline verified independently. Heads-up checks (universal selector, forced-colors, dead rules) returned no fidelity defects — all observations are pre-existing patterns inherited from the source.

## Risk Assessment
**Residual Risk**: Low for this slice in isolation.
- The only material runtime risk is **cascade-order regression** when `base.css` is wired into `index.css`/`App.css` — if loaded *before* the consolidated tokens (`var(--…)` defs), every `var()` here will fall back to its initial value or any provided fallback (there are none in this slice). Verify the import order in the orchestrating stylesheet places tokens **before** `base.css`. That wiring is out of scope for this file-level review but is the dominant integration risk.
- A secondary, very small risk: if a future stylesheet adds a `*`-selector rule with conflicting `scrollbar-width` or `box-sizing`, the cascade will hinge on import order. Pre-existing in the original; unchanged here.

## Information Gaps
- Did not verify token definitions exist in the consolidated tokens file (out of scope for this slice).
- Did not verify the orchestrating stylesheet (`App.css` post-split or `index.css`) `@import`s `base.css` at the correct cascade position — out of scope for this file's review, in scope for the integration review.
- Did not run the frontend build/HMR; no behavioural runtime check performed.
- Did not search the React tree for actual consumers of `.mono`, `.sr-only`, `.skip-to-content`, `#root` — fidelity review does not require it.

## Caveats
- This review certifies **fidelity of extraction**, not **quality of the underlying CSS**. The universal-selector and forced-colors gaps are pre-existing and were not introduced by this edit; they are surfaced per the caller's request but are not actionable on this PR.
- "Approved" is conditional on the integration review (cascade order, import wiring) being performed when the consolidating stylesheet is assembled.
