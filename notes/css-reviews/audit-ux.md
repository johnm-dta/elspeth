# UX / A11y Review: audit.css

**File:** `src/elspeth/web/frontend/src/components/audit/audit.css`
**Consumers:** `AuditReadinessPanel.tsx`, `AuditReadinessRow.tsx`, `ExplainDialog.tsx`, `ReadinessRowDetail.tsx`
**Overall:** Acceptable — one Major layout bug, several Minor token/markup gaps, no Critical blockers.
**Critical Issues:** 0
**Major Issues:** 2
**Minor Issues:** 5

---

## 1. Readiness State Distinguishability

**Strengths:**
- Three-channel status encoding: 3px left-edge stripe (colour), glyph (✓ ⚠ ✗ —), and `.sr-only` text (aria status label). Colour is explicitly "reinforcement", not primary — this is correct.
- `.sr-only` is defined in `base.css` (line 119) and confirmed canonical.
- Warning/error background tints are low-opacity washes (~12-14% alpha), not saturated: `rgba(232,86,83,0.12)` / `rgba(227,132,68,0.14)`. Visual gravitas without alarm fatigue — correct for a legal-record surface.
- `audit-readiness-row--not_applicable` uses `--color-border-strong` stripe (neutral) — appropriate for a non-actionable state.

**Issues:**

| Issue | Severity | Location | Fix |
|-------|----------|----------|-----|
| Hover washes out warning/error row tint | Minor | `.audit-readiness-row-btn:hover` (line 198) | The inner button's `--color-surface-hover` background overpaints the row's `--color-warning-bg` / `--color-error-bg` while the cursor is present. The stripe and glyph remain, so this is not a WCAG failure, but it erodes the signal. Fix: set hover background at the `li` level (`audit-readiness-row--warning:hover`) or paint it with `mix-blend-mode: multiply` / `color-mix()` so the tint persists. |
| `--audit-readiness-row--llm-interpretations` modifier applied by TSX but has no rule | Minor | `AuditReadinessPanel.tsx` line 543; `audit.css` has no matching rule | The panel assigns `extraClassName: "audit-readiness-row--llm-interpretations"` to LLM-interpretation rows. The CSS has no rule for this class. Currently harmless (the status modifier still applies), but any future need to visually distinguish opt-out or "not yet surfaced" state has no styling hook to target. Either add an empty comment-anchored rule or remove the class from the TSX. |

---

## 2. Explain Dialog

**Strengths:**
- Focus trap via `useFocusTrap(dialogRef, true, ".explain-dialog-close")` — initial focus lands on Close, Tab wraps inside, focus restores on unmount.
- Document-level Escape (`document.addEventListener("keydown", …)`) fires even if focus drifts outside the trap.
- Backdrop is a sibling of the dialog `<div>` (not nested inside it) — correct; nesting confuses SR dialog-boundary detection.
- `aria-modal="true"`, `role="dialog"`, `aria-labelledby` all present.
- No separate "action row" — Close in the header is the only action. The pinned-action-row requirement passes vacuously (nothing to pin).
- Narrative body: `overflow: auto` on `.explain-dialog-narrative` (line 287) + `flex: 1 1 auto` (line 301) + `min-height: 0` on the content wrapper (line 251). The scrollable body with pinned header is mechanically correct.

**Issues:**

| Issue | Severity | Location | Fix |
|-------|----------|----------|-----|
| Close button width below iOS touch standard | Major | `.explain-dialog-close` (line 268) | `min-width: 36px` — below the 44px iOS / 48dp Android minimum. `min-height: var(--size-control-compact)` resolves to 36px (tokens.css line 189) — also under 44px in both dimensions. The `×` character's rendered hit area is determined by padding alone; there is no padding rule on this element. Fix: increase `min-width` and `min-height` to 44px, or add symmetric padding of at least 8px. |
| No `:focus-visible` override on `.explain-dialog-close` | Minor | audit.css lines 267–280 | This button is not a `.btn`, so it relies on the global `:focus-visible` rule in `base.css` (2px ring). The global rule is present and sufficient, but the close button's visual treatment (border, background, `cursor:pointer`) is manually assembled rather than composing from `.btn`. If a future base.css change alters the global rule, this button's focus state is invisible. Document the dependency or extract a `btn-icon` utility class. |

---

## 3. Row Detail Drawer

**Strengths:**
- Intentionally non-modal (`aria-modal="false"`, comment in TSX explaining why: "Jump to node" keeps the drawer open while the user inspects the graph). No focus trap is correct here.
- Mount-time focus move to Close button (`closeBtnRef.current?.focus()`) ensures keyboard users land inside the drawer on open.
- Escape closes via `onKeyDown` on the root `<div role="dialog">` — fires when focus is inside (which it is after the mount focus move).
- `overflow: hidden` on the root, `overflow: auto` on `.readiness-row-detail-components` — the components list scrolls independently.

**Issues:**

| Issue | Severity | Location | Fix |
|-------|----------|----------|-----|
| Close button touch-target same defect as ExplainDialog | Major | `.readiness-row-detail-close` (line 348) | Identical `min-width: 36px; min-height: var(--size-control-compact)` = 36×36px. Same fix: raise to 44px or add 8px padding. This is the same token — one fix in both rules covers both components. |
| Drawer body has no scroll affordance on `.readiness-row-detail-body` | Minor | Lines 363–369 | `.readiness-row-detail-body` has no `overflow: auto`. If `row.detail` is long (multi-paragraph), it pushes `.readiness-row-detail-components` down and the user must scroll the fixed-height drawer root. The root has `overflow: hidden`. Wrapping body + components in a flex column with `flex: 1 1 auto; overflow: auto; min-height: 0` would let them share the vertical space properly. Currently `.readiness-row-detail-components` has `overflow: auto` (line 379) but only the components list — not the prose — can scroll independently. |
| Row detail button has no `aria-haspopup` hint | Minor | `AuditReadinessRow.tsx` lines 89–101; no matching CSS | Actionable rows open the `ReadinessRowDetail` drawer (role="dialog"). The button has no `aria-haspopup="dialog"`. Screen readers announce only "button" with no hint that activation opens a dialog. CSS cannot fix this — it is a TSX change — but the review notes it because the CSS provides no chevron or disclosure affordance either. Together these omissions mean keyboard users have no pre-click indication that a panel will open. Fix (TSX): add `aria-haspopup="dialog"` to the `<button>` in `AuditReadinessRow.tsx` line 89. |

---

## 4. Readiness Panel Collapsed View

**Issues:**

| Issue | Severity | Location | Fix |
|-------|----------|----------|-----|
| Collapsed-pill renders three flex lines, not the promised two | Major | `.audit-readiness-summary` (line 64) + `AuditReadinessPanel.tsx` lines 418–425 | The CSS comment on lines 11–18 documents "glyph + label on line 1, freshness meta on line 2." But the JSX renders `<span aria-hidden="true">✓</span> Audit ready <span class="…-meta">…</span>`. Because `.audit-readiness-summary` is `flex-direction: column`, the anonymous text node "Audit ready" becomes its own flex item, producing **three stacked lines** (✓ / Audit ready / meta) rather than two. The comment is a load-bearing design spec being violated. Fix option A (markup): wrap glyph and label text in a single `<span>` — `<span><span aria-hidden>✓</span> Audit ready</span>`. Fix option B (CSS): change to `flex-direction: row; flex-wrap: wrap` and use `flex-basis: 100%` on the meta span to force it onto a second line. |

---

## 5. Long Readiness List / Dense Data

**Strengths:**
- Panel rows are `<li>` items in a `<ul>` — `list-style: none` is set, so VoiceOver/NVDA do not announce list roles unnecessarily (though Safari/VoiceOver omits the role on `list-style:none` lists; acceptable here as the individual rows carry their own ARIA semantics).
- Six fixed rows — no virtualisation needed at this scale.
- Monospace font (`--font-mono`) correctly applied to `.readiness-row-detail-component-id` (line 410), which displays node IDs.

**Issues:**

| Issue | Severity | Location | Fix |
|-------|----------|----------|-----|
| No hover layout-shift guard on readiness rows | Minor | `.audit-readiness-row` (line 149) | The 3px left-border is `transparent` by default and coloured per status. This means the border-width is constant — no layout shift on hover (good). But the hover background on `.audit-readiness-row-btn` (line 199) is applied to the inner button, not the `<li>`. If the `<li>`'s `border-left` were ever moved to the button, the 3px jump would cause layout shift. The current split is safe but fragile; a comment on line 149 noting "border-width is intentionally constant (no layout shift)" would prevent a future regression. |

---

## 6. Accessibility Quick Check

| Criterion | WCAG | Result |
|-----------|------|--------|
| 1.4.3 Contrast — text | AA (4.5:1) | Cannot measure without rendered tokens, but token values are semantic-named (`--color-text`, `--color-text-muted`); no raw colour overrides in this file that would introduce a new contrast risk. |
| 1.4.11 Non-text contrast — status stripes | AA (3:1) | `--color-success` (#14b0ae), `--color-error` (#e85653), `--color-warning` (#e38444) against `--color-surface-elevated` — all saturated hues; 3:1 is plausible but unverified without token-to-hex rendering. Flag for tooling verification. |
| 2.1.1 Keyboard navigation | Pass | All interactive elements are `<button>` — natively keyboard-operable. |
| 2.4.7 Focus visible | Pass (with caveat) | Global `base.css` 2px ring covers all focusable elements. Close buttons (manually assembled, not `.btn`) inherit this but the dependency is implicit. |
| 1.1.1 Alt text | Pass | All decorative glyphs carry `aria-hidden="true"`; status glyphs are supplemented by `.sr-only` text. |
| 2.5.5 / 2.5.8 Target size | Fail (Major) | `.explain-dialog-close` and `.readiness-row-detail-close` are 36×36px. WCAG 2.5.8 (AA, WCAG 2.2) requires 24×24px minimum; 2.5.5 (AAA) requires 44×44px. Both buttons are between the targets — they clear 2.5.8 but fail 2.5.5. Flag as Major because this is a high-stakes legal-record surface used under stress. |
| 3.3.2 Labels | Pass | All inputs and buttons have visible labels or `aria-label`. |

---

## 7. Token Discipline

Raw values that should be promoted to tokens:

| Value | Line(s) | Suggested token |
|-------|---------|-----------------|
| `3px` (stripe width) | 151, 159, 167, 173 | `--size-status-stripe` |
| `24px` (glyph grid column) | 178 | `--size-glyph-col` |
| `32px` (dialog inset) | 236 | `--space-dialog-inset` |
| `36px` (close button min-width) | 268, 348 | Addressed by raising to `--size-touch-target` (44px) |
| `rgba(0,0,0,0.45)` (backdrop opacity) | 229 | `--color-backdrop` |
| `rgba(0,0,0,0.25)` (dialog/drawer shadow) | 242, 330 | `--shadow-modal` |

None of these omissions break anything today; tokenising them ensures future design changes propagate without hunting raw values.

---

## 8. Dead Rules

None. Every selector in `audit.css` maps to a className used in one of the four consumer components. The `audit-readiness-row--llm-interpretations` modifier (noted in section 1) is the reverse: a TSX className with no corresponding CSS rule.

---

## Priority Recommendations

**Major (Fix Before Launch):**
1. **Collapsed-pill three-line bug** — wrap glyph and "Audit ready" text in a shared `<span>` or switch `.audit-readiness-summary` to `flex-direction: row; flex-wrap: wrap` with `flex-basis: 100%` on the meta span.
2. **Close button touch targets** — raise `min-width` and `min-height` on `.explain-dialog-close` and `.readiness-row-detail-close` from 36px to 44px. One token change (`--size-control-compact: 36px → 44px`) would cascade, but verify that raising it doesn't break other `.btn`-sized elements.

**Minor (Improvement Cycle):**
1. **Hover washes warning/error tint** — apply hover background at `<li>` level or use `color-mix()` to layer over the status tint.
2. **`aria-haspopup="dialog"` on row buttons** — TSX change in `AuditReadinessRow.tsx` line 89; no CSS change needed.
3. **Drawer body scroll** — wrap `.readiness-row-detail-body` and `.readiness-row-detail-components` in a shared flex column with `overflow: auto` so long detail text doesn't require scrolling the fixed-height drawer root.
4. **`audit-readiness-row--llm-interpretations` CSS hook** — add an empty anchored rule or remove the TSX class.
5. **Tokenise raw values** — `3px` stripe, `24px` glyph column, `32px` dialog inset, backdrop opacity, shadow value.

---

## Confidence Assessment

**High confidence:** Collapsed-pill layout bug (structural — JSX flex-item count is deterministic), close-button size (token value confirmed: 36px), `.sr-only` presence (found in `base.css` line 119), `--size-control-compact` value (tokens.css line 189), hover-overwrites-status-tint (paint order is deterministic).

**Medium confidence:** WCAG 1.4.11 non-text contrast on status stripes (token hex values are available but not rendered against all surface colours in all themes; flag for tooling pass). Drawer-body scroll behaviour (would require a running browser to confirm whether long `row.detail` content overflows visibly or is clipped).

**Lower confidence:** Overall "gravitas" tone assessment — this is subjective and context-dependent on the complete visual environment, which is not visible in CSS alone.

## Risk Assessment

**Highest risk:** The collapsed-pill three-line bug is operator-visible (matches the "weirdly wide and full of text" symptom cited in the CSS comment — the comment was written to document the fix, but the fix was not applied to the markup). It is the first thing an auditor or demo observer sees when the panel is all-green.

**Second:** Close button targets — 36×36px on a legal-record surface used under conditions that may include stress or time pressure.

## Information Gaps

- Rendered token values in dark theme were not checked for non-text contrast (stripe vs. surface).
- The `useFocusTrap` implementation was not inspected — correctness of Tab-wrap and initial-focus behaviour is assumed from the call site.
- The `--size-control-compact` token is used by other components; raising it to 44px needs a project-wide impact check before landing.

## Caveats

This review is based on static CSS and TSX source only. No browser rendering, no screen reader test, no colour-contrast tool was run. Findings that depend on computed layout (scroll, overflow, three-line flex) are structurally derived and high-confidence but should be confirmed visually before the fixes are closed.
