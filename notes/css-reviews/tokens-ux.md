# tokens.css — UX / Accessibility Review

**File:** `src/elspeth/web/frontend/src/styles/tokens.css`
**Reviewed:** 2026-05-23
**Reviewer:** lyra-ux-designer (UX Critic Agent)
**Scope:** Token coverage, naming hygiene, contrast pairs, semantic vs raw, dead tokens, forced-colors isolation.

---

## Overall Status: MINOR

- Critical issues: 0
- Major issues: 2
- Minor issues: 4

The token layer is well-structured. Dark/light theme symmetry on colour tokens is complete (verified programmatically — zero drift). Naming conventions are consistent and prefix-disciplined. The two major issues are both contrast failures that are real but bounded in scope.

---

## 1. Theme Symmetry

**Pass.** All `--color-*` tokens declared in `:root` have corresponding overrides in `[data-theme="light"]`. No asymmetry found via automated diff. Non-colour tokens (spacing, typography, radius, transitions, z-index) are correctly absent from the light override block — they are theme-invariant by design.

---

## 2. Naming Hygiene

**Pass with one minor cleanup item.**

All tokens follow the `--{category}-{qualifier}` convention consistently:
- `--color-*` for all colour roles
- `--space-*` for spacing scale
- `--font-size-*`, `--font-*`, `--line-height-*` for typography
- `--radius-*`, `--size-*`, `--transition-*`, `--z-*` for structural values

**Minor — grep artifact token `--lg` (line 159, comment text):**
The regex extraction picks up `--lg` from the comment `/* was 18px duplicate of --lg */`. This is not a real declared token; it is a reference to a now-deleted token preserved in documentation. No action required on the token itself, but the comment could be tightened to avoid confusing future static analysis passes: "was 18px, formerly `--font-size-lg`" would be unambiguous.

---

## 3. Contrast Pairs

### 3a. Dark theme — `--color-error` as text (Major)

**Measured:** `#e85653` on `#0f2d35` = **4.07:1**
**WCAG 1.4.3 threshold:** 4.5:1 for normal text
**Status: FAILS AA for body text**

`--color-error` is applied as `color:` (text foreground) in at least 15 CSS rules across `chat.css`, `header.css`, `blobs.css`, `settings.css`, `execution.css`, `tutorial.css`, `guided.css`, and `catalog.css`. The badge font size is 12px (`--font-size-xs`), well below the 18px large-text threshold. The 3:1 UI-component bar passes, but the token is used for text, not icons or borders alone.

The same value also surfaces as `--color-status-failed`, which is applied as badge text at 12px.

**Fix:** Raise dark `--color-error` from `#e85653` to approximately `#ec6a67` or `#f07370` to reach 4.5:1 on `#0f2d35`. Verify the light theme value `#c93b38` (currently 4.71:1 on `#f4f8f9`) is preserved — it already passes.

**Note:** The test suite in `colorContrast.test.ts` does not appear to include a `--color-error`-on-`--color-bg` assertion. Adding one would prevent regression.

### 3b. Light theme — status badge text on tinted background (Major)

Status badge classes (`.status-badge-*`) in `shared.css` use hardcoded `rgba(...)` tint backgrounds at 15% opacity. The text colour is drawn from the corresponding `--color-status-*` token, which in light mode is a darkened variant. However, the hardcoded background tints use the same RGB values as the dark theme (not light-adjusted), creating a higher-luminance composited background than intended.

**Measured contrast (text on composited badge-bg), 12px font, light theme:**

| Status | Token value | Ratio | WCAG 1.4.3 |
|--------|-------------|-------|------------|
| pending | `#5a7a84` on rgba(122,154,154,0.15)+`#f4f8f9` | 3.77:1 | FAIL |
| running | `#2890b8` on rgba(97,218,255,0.15)+`#f4f8f9` | 3.18:1 | FAIL |
| completed | `#0d8a88` on rgba(20,176,174,0.15)+`#f4f8f9` | 3.42:1 | FAIL |
| cancelled | `#b86830` on rgba(227,132,68,0.15)+`#f4f8f9` | 3.40:1 | FAIL |

Dark theme badges also fail for pending (3.81:1), completed (4.22:1), and cancelled (4.29:1). Running passes at 6.26:1.

**Fix options (in order of preference):**

1. Raise the `--color-status-*` token values in light mode to ensure 4.5:1 against the composited badge background. The composited backgrounds are near-white; the tokens need to reach approximately:
   - `pending`: darken `#5a7a84` → `#4a6a74` (approx 4.5:1 on composited bg)
   - `running`: darken `#2890b8` → `#1e7599` (4.5:1 on composited bg)
   - `completed`: darken `#0d8a88` → `#0a6f6d` (4.5:1 on composited bg)
   - `cancelled`: darken `#b86830` → `#985225`

2. Alternatively, tokenise the badge background colours in `shared.css` (replacing hardcoded `rgba(...)` with `--color-status-*-bg` tokens) and provide light-adjusted values. This is the more robust fix because it eliminates the hidden coupling between `shared.css` hardcoded rgba and token values, but it requires changes outside `tokens.css`.

3. If badge text is always paired with an icon glyph providing redundant status information, the 3:1 UI-component threshold would technically apply — but that relies on `statusBadgeAccessibility.test.ts` enforcing the pairing, which is worth verifying.

**Note:** `statusBadgeAccessibility.test.ts` exists but must be checked to confirm it covers light-theme badge text contrast specifically.

### 3c. Body text and primary actions — Pass

All body text, secondary text, and muted text pairs pass AA in both themes with comfortable margins:

| Token pair | Dark | Light |
|------------|------|-------|
| `--color-text` on `--color-bg` | 12.33:1 | 13.57:1 |
| `--color-text-secondary` on `--color-bg` | 8.71:1 | 6.95:1 |
| `--color-text-muted` on `--color-bg` | 4.78:1 | 6.32:1 |
| `--color-text-inverse` on `--color-btn-primary-bg` | 6.92:1 | 5.28:1 |
| `--color-text-inverse` on `--color-btn-danger-bg` | 5.98:1 | 5.98:1 |
| All `--color-badge-*` on `--color-bg` | ≥4.5:1 | ≥4.36:1 |

`--color-badge-coalesce` on light `--color-bg` is 4.36:1, just below 4.5:1. This is marginal but passes the 3:1 UI-component bar. If badge labels are rendered as text (not just icons), this is a borderline concern worth watching.

---

## 4. Semantic vs Raw Values

**Pass.** Raw hex values appear only in the token layer itself; no raw hex/rgba values were found in consumer CSS or TSX files where a corresponding token exists. Two inline fallback values in consumer files are correctly bounded:
- `var(--color-danger, #b00020)` — intentional fallback for an undeclared token (see §6)
- `var(--color-success, #16a34a)` — intentional fallback, same pattern
- `var(--color-bg-hover, rgba(143, 200, 200, 0.08))` — intentional fallback in `header.css`

The scrollbar tokens correctly reference `var(--color-bg)` rather than repeating the hex, and `--color-node-unchecked` correctly aliases `var(--color-border-strong)`.

The `--color-success` / `--color-state-positive` pair carries byte-identical values in dark mode (`#14b0ae`). These are semantically distinct roles (completion state vs positive state indicator), so the duplication is intentional. In light mode they diverge (`#0d8a88` vs `#056e6c`), confirming the semantic split is real.

---

## 5. Dead Tokens

The following tokens are declared but have no `var(--...)` consumer found in any `.css` or `.tsx` file in `src/`:

| Token | Lines | Assessment |
|-------|-------|------------|
| `--color-node-valid` | 91, 302 | **Not dead.** Used via `getComputedStyle()` in `GraphView.tsx` (line 57-64) at runtime; not reachable by static `var()` grep. Confirmed by `readThemeColor()` function pattern. Actually: `VALIDATION_COLORS` in `tokens.ts` maps to `--color-success`, `--color-warning`, `--color-error` — so `--color-node-*` may be genuinely unused if `VALIDATION_COLORS` is the canonical path. |
| `--color-node-warning` | 92, 303 | Same as above. |
| `--color-node-invalid` | 93, 304 | Same as above. |
| `--color-node-unchecked` | 94, 305 | Same as above. |
| `--opacity-dimmed` | 126 | **Likely dead.** No `var()` reference found anywhere in `src/`. Confirm and delete if unused. |
| `--inspector-default-width` | 143 | **Likely dead.** No `var()` reference in `src/`. May be used by a layout component directly with a hardcoded `320px` — worth checking. |
| `--inspector-min-width` | 144 | Same as above. |
| `--transition-slow` | 194 | **Likely dead.** `--transition-fast` and `--transition-normal` are both used; `--transition-slow` has zero references. |
| `--z-overlay-backdrop` | 200 | **Likely dead.** `--z-overlay` is used (2 references) but `--z-overlay-backdrop` has zero. Backdrop may be hardcoded. |

**Action for node tokens:** Verify whether `--color-node-*` are consumed via `getPropertyValue()` in `GraphView.tsx` or superseded by `tokens.ts`'s `VALIDATION_COLORS`. If the latter, remove the `--color-node-*` declarations to eliminate the confusion: two separate systems asserting node validation colours is a maintenance liability.

**Action for the five likely-dead tokens:** Run a project-wide search including string literals and `getPropertyValue` calls. If zero matches, delete all five. They are safe to remove — no consumer in `src/` references them by any mechanism found in this review.

---

## 6. Undeclared Tokens (Referenced but Not Defined)

Two tokens are referenced in consumers but not declared in `tokens.css`:

| Token | Reference | Assessment |
|-------|-----------|------------|
| `--color-danger` | `ComposerPreferencesPanel.tsx:91`, `InlineOptOutCheckbox.tsx:74` | **Minor.** Used with hardcoded fallback `#b00020`. Should either be declared in `tokens.css` (with light theme override, since `#b00020` is light-appropriate) or replaced with `--color-error` which is the established error semantic. The fallback means it works but bypasses the theme system entirely. |
| `--color-bg-hover` | `header.css:128` | **Minor.** Used with fallback `rgba(143, 200, 200, 0.08)`. This value is close to `--color-surface-hover` (dark: `rgba(255,255,255,0.04)`). Should be declared or consolidated with `--color-surface-hover`. The fallback makes it functionally invisible to theme switching. |

---

## 7. Forced-Colors / High-Contrast Mode

No `@media (forced-colors: active)` tokens or rules are present in `tokens.css`. This is not flagged as a defect — forced-colors mode uses the UA's system colour keywords and typically overrides custom properties correctly for solid backgrounds and foregrounds. The primary risk would be for the many `rgba(...)` semi-transparent values (borders, bubble backgrounds, badge backgrounds) which collapse to opaque colours under forced-colors; those are borderline invisible concerns that only become active defects if any functionality is conveyed solely by the transparent layer.

No tokens are claimed to be forced-colors-specific, so no isolation requirement applies.

---

## Priority Summary

### Major (Fix Before Next RC)

1. **tokens.css:75 — `--color-error` dark: 4.07:1 fails AA as text**
   Raise from `#e85653` to reach 4.5:1 on `#0f2d35`. Error messages are displayed as text in at least 15 CSS locations at 12-13px. Add a `colorContrast.test.ts` assertion to lock this pair.

2. **shared.css — Light status badge text fails AA (pending/running/completed/cancelled)**
   Either darken the four `--color-status-*` light-theme token values, or tokenise the badge background colours so the pairs are verifiable together. Assess whether `statusBadgeAccessibility.test.ts` covers light-mode text contrast.

### Minor (Fix Before Ship)

3. **tokens.css:91-94, 302-305 — `--color-node-*` tokens: resolve dual-system ambiguity**
   Verify whether `tokens.ts`'s `VALIDATION_COLORS` (which references `--color-success/warning/error`) has superseded `--color-node-*`. If yes, delete `--color-node-*` from both theme blocks. If no, document which system owns node border colours.

4. **tokens.css:126, 143-144, 194, 200 — Five likely-dead tokens**
   `--opacity-dimmed`, `--inspector-default-width`, `--inspector-min-width`, `--transition-slow`, `--z-overlay-backdrop` have no consumer references. Verify with getPropertyValue search and delete if confirmed dead.

5. **Two undeclared tokens with fallbacks**
   `--color-danger` (two components): declare in `tokens.css` with light-theme override, or replace with `--color-error`.
   `--color-bg-hover` (header.css): declare or consolidate with `--color-surface-hover`.

6. **tokens.css:159 — Comment reference to `--lg`**
   Tighten comment wording to avoid confusing static analysis. Change "was 18px duplicate of --lg" to "was 18px, formerly `--font-size-lg`".

---

## Confidence Assessment

**High (85%)** for contrast measurements — values computed from hex token declarations using the WCAG relative luminance formula; accuracy depends on tokens matching runtime values (they should, absent CSS override layers). The status badge analysis assumes the hardcoded rgba values in `shared.css` represent the actual compositing context; if those elements render on a different background in practice, ratios shift.

**Medium (70%)** for dead token assessment — static `var()` grep cannot find `getPropertyValue(cssVarName)` runtime reads. The `--color-node-*` tokens are the primary unknown; the other five (opacity, inspector, transition-slow, z-overlay-backdrop) have no dynamic consumption pattern and are more likely genuinely dead.

**High (90%)** for theme symmetry and naming hygiene — both are deterministic from the file content.

## Risk Assessment

**Contrast failures carry AA compliance risk.** The dark `--color-error` failure is the most impactful because it affects error messaging — the exact content users most need to read under stress. Light theme status badge failures affect operational state visibility; users running pipelines on light mode will have reduced badge legibility.

**Dead tokens carry no functional risk** but increase maintenance cost and create misleading surface area for future implementers.

## Information Gaps

- The `statusBadgeAccessibility.test.ts` file was found but not fully read — it may already assert light-mode badge contrast, in which case finding 3b would be surfaced by CI.
- `--color-node-*` consumption pattern via `getPropertyValue()` not confirmed or denied by this review — needs a targeted search for `getPropertyValue` calls.
- Badge rendering background: if `.status-badge-*` elements always appear on `--color-surface` (white in light mode) rather than `--color-bg`, the composited ratios improve slightly but the failures persist.

## Caveats

This review does not assess whether the contrast test suite is comprehensive (that is `accessibility-auditor` scope). Contrast ratios were computed against `--color-bg` as the canonical background; actual in-situ contrast depends on the component's parent background. The `--color-error`-on-`--color-error-bg` pair (3.66:1 dark) is a further failure if error text appears inside error-tinted panels — that pairing was not counted in the main findings because it requires a component-level audit to confirm the layering.
