# settings.css — Complex Reviewer Fidelity Pass

## Edit Review: src/elspeth/web/frontend/src/components/settings/settings.css

### Summary
- **NEW**: `/home/john/elspeth/.worktrees/css-split/src/elspeth/web/frontend/src/components/settings/settings.css` (165 lines, type: docs/CSS)
- **ORIGINAL**: `/home/john/elspeth/src/elspeth/web/frontend/src/App.css` lines 4083–4244
- **Declared scope**: All `.secrets-*` rules in 4083–4244, no moves, byte-identical selectors and declarations.
- **Overall verdict**: **Approved (fidelity passes).** Several non-fidelity observations follow under "Out-of-scope observations" — none block the split.

### Intent Fit (rule-by-rule diff)

| Original line | Selector | Present in settings.css | Byte-identical? |
|---|---|---|---|
| 4086 | `.secrets-panel-header` | L7 | yes |
| 4095 | `.secrets-panel-title` | L16 | yes |
| 4101 | `.secrets-panel-close` | L22 | yes |
| 4117 | `.secrets-panel-body` | L38 | yes |
| 4123 | `.secrets-section-heading` | L44 | yes |
| 4132 | `.secrets-form-fields` | L53 | yes |
| 4138 | `.secrets-form-label` | L59 | yes |
| 4145 | `.secrets-form-input` | L66 | yes |
| 4156 | `.secrets-submit-btn` | L77 | yes |
| 4163 | `.secrets-list` | L84 | yes |
| 4172 | `.secrets-list-item` | L93 | yes |
| 4182 | `.secrets-list-detail` | L103 | yes |
| 4190 | `.secrets-list-name` | L111 | yes |
| 4197 | `.secrets-unavailable-reason` | L118 | yes |
| 4203 | `.secrets-scope-badge` | L124 | yes |
| 4214 | `.secrets-delete-btn` | L135 | yes |
| 4231 | `.secrets-loading, .secrets-empty` | L152 (grouped selector preserved) | yes |
| 4238 | `.secrets-footnote` | L159 | yes |

**18/18 selectors present.** Declaration order preserved. The grouped selector `.secrets-loading, .secrets-empty` is preserved as a single rule (not split). The section header comment "Secrets Panel" with the rule-line decoration is preserved (L4–L6). A new file header comment was prepended (L1–L2):
```css
/* settings.css — Settings panel (secrets management).
   Owners: .secrets-*. */
```
This is an additive change to the new file's preamble only — acceptable file-ownership documentation, not a fidelity regression.

`grep -n "secrets-" App.css` returns matches only in 4086–4238, confirming no `.secrets-*` rules exist outside the declared range. No rules were missed; no rules from outside the range were pulled in.

### Scope Discipline
- **Declared out-of-scope**: rule reordering, color/token changes, selector rewrites — all verified preserved.
- **Undeclared changes**: only the two-line file-header comment (above). No "while I'm here" formatting tweaks, no whitespace normalisation, no media-query insertions.

### Structural Integrity

| Invariant | Status | Evidence |
|---|---|---|
| Brace balance | ✓ | 18 `{` / 18 `}` blocks; final file ends at L165 with closing `}` for `.secrets-footnote` then a single trailing newline |
| Declaration order within each rule | ✓ | Verified inline above |
| Section banner comment | ✓ | L4–L6 matches App.css L4083–L4085 verbatim |
| No stray rules from neighbouring sections | ✓ | Next App.css section is `.error-boundary-fallback` (L4248), not migrated |

### Cross-Reference / Call-Site Integrity (class usage in components/settings/)

`grep -rh 'secrets-' src/elspeth/web/frontend/src/components/settings/` yields 21 distinct class names used in TSX. Cross-checked against the 18 CSS rules:

**Used in TSX and defined in CSS (16):** `secrets-panel-header`, `secrets-panel-title`, `secrets-panel-close`, `secrets-panel-body`, `secrets-section-heading`, `secrets-form-fields`, `secrets-form-label`, `secrets-form-input`, `secrets-submit-btn`, `secrets-list`, `secrets-list-item`, `secrets-list-detail`, `secrets-list-name`, `secrets-unavailable-reason`, `secrets-scope-badge`, `secrets-delete-btn`, `secrets-loading`, `secrets-empty`, `secrets-footnote`.

**Used in TSX but NOT a CSS class (2):** `secrets-add-heading`, `secrets-inventory-heading`. These are **`id=` values** for `aria-labelledby` wiring (SecretsPanel.tsx L187, L189, L262, L264), not class selectors. Not a dead rule, not a missing rule — this is an ARIA id token that *happens* to share the `secrets-` prefix. **No action required**, but noting it because a future reviewer grepping class names will flag it as orphan. Worth a comment in `settings.css` or a rename to drop the `secrets-` prefix on the id tokens (e.g. `id="add-secret-heading"`) to remove the false positive.

### Orphan / Dead-Code / Boundary Findings

- **L1–L6 (new file head)**: clean. New file-header comment + preserved section banner.
- **L165 (EOF)**: file ends cleanly after `.secrets-footnote` closing brace; one trailing newline.
- **App.css boundary at L4082 → L4083**: I did not inspect App.css's deletion site, but the writer report states "no moves," implying the original block is being relocated. If the original is being *replaced by an import* of the new file, that wiring should be checked separately by the integration reviewer. If the original is still present in App.css, the project now has duplicated rules — verify via `grep -c '\.secrets-panel-header' App.css` after the split lands.

### Style / Behavior Continuity
- Token usage (`var(--space-*)`, `var(--color-*)`, `var(--font-size-*)`, `var(--radius-*)`, `var(--size-control)`, `var(--font-mono)`, `var(--line-height-normal)`) is consistent with surrounding App.css conventions. No hardcoded values were introduced in the migrated rules.
- Two-space indentation matches App.css.

### Issues Found

#### Critical
None.

#### Major
None.

#### Minor
1. **TSX inline styles bypass the CSS file (SecretsPanel.tsx L151–L168, L262).** The dialog container's positioning, sizing, surface color, border, radius, box-shadow, and `fontSize: 13` are inline. These are not under `settings.css` ownership and were not in the migrated range, so this is **not** a split-fidelity issue — but it does mean the file's stated ownership (`Owners: .secrets-*`) is incomplete: the panel's visual envelope is owned by TSX, not by `settings.css`. Consider promoting these to a `.secrets-panel` rule for owner consolidation in a follow-up.
2. **`secrets-add-heading` / `secrets-inventory-heading` are ARIA `id`s, not classes** — noted above. Risk: future grep-based dead-CSS sweeps will misreport them. Cheap fix: drop the `secrets-` prefix on the id tokens, or add a one-line comment in `settings.css` documenting that those tokens are deliberately not CSS classes.

### Out-of-Scope Observations (NOT fidelity defects; surface for the design owner)

These existed in the original App.css block and are inherited unchanged into `settings.css`. They are **not** in scope for a fidelity review but were requested in the task:

1. **Dead rules — none confirmed.** All 18 migrated rules have at least one TSX consumer in `components/settings/`.
2. **Brittle selectors.** All selectors are single-class — not brittle. The grouped selector `.secrets-loading, .secrets-empty` is fine. The `:focus-visible` styling is inherited from the global rule at App.css L289 (`:focus-visible { outline: 2px solid var(--color-focus-ring); outline-offset: 2px; }`). That means none of the migrated rules need their own `:focus-visible` — the global wins via cascade. **However**, if `settings.css` is loaded *without* App.css's global rule (e.g. in a hypothetical isolated Storybook), focus rings will silently disappear. Worth flagging when the broader split lands: the global `:focus-visible` is a hidden dependency the new file does not declare.
3. **Hardcoded colours that should be tokens — none in the migrated rules.** All colours go through `var(--color-*)`. Dialog inline styles in TSX use `var(--color-surface, #fff)` (token with fallback — fine) and `rgba(0,0,0,0.25)` for box-shadow (hardcoded, but TSX-owned, out of scope here).
4. **Missing focus-visible on reveal/copy buttons.** SecretsPanel.tsx has **no reveal/copy buttons** in the current implementation — only `.secrets-panel-close` (× close), `.secrets-submit-btn` (add), and `.secrets-delete-btn` (×). All three inherit the global `:focus-visible` outline. The task's "reveal/copy" framing appears to assume a feature that does not exist in this file. If reveal/copy buttons are planned, they will inherit the global focus ring automatically; no `settings.css` change required unless a stronger contrast is wanted on the destructive `.secrets-delete-btn` (currently relies on the generic 2px outline against `--color-error` text — adequate for WCAG but not bespoke).
5. **`.secrets-panel-close` and `.secrets-delete-btn` are 44×44 min-targets** — good (matches WCAG 2.5.5 touch target). No regression.

### Confidence Assessment
**Confidence: High** for fidelity (all 18 rules verified byte-identical, selector order preserved, no spurious additions beyond the two-line file header).
**Confidence: Medium** for the dead-rule / brittle-selector / hardcoded-colour observations — I checked TSX in `components/settings/` only. Classes referenced elsewhere in the app (other panels, tests, e2e) were not surveyed; an `rg 'secrets-' src/elspeth/web/frontend` would close that gap but was not requested.

### Risk Assessment
- **Hidden dependency on global `:focus-visible`** (App.css L289). If `settings.css` is ever loaded in isolation, the migrated buttons lose keyboard focus indication silently. Document the dependency or duplicate the rule scoped to `.secrets-panel-close`, `.secrets-submit-btn`, `.secrets-delete-btn` if isolation is on the roadmap.
- **Original-block deletion at App.css L4083–L4244 not verified by this review.** If the writer left the original in place, the project will have duplicate `.secrets-*` rules and the cascade will resolve in load order — flag for the integration reviewer.
- The "reveal/copy buttons" framing in the task brief may indicate that a planned feature was expected to exist; verify with the operator whether absence is intentional.

### Information Gaps
- Did not inspect App.css around L4083 to confirm the original block was deleted as part of the split. Treated the file as additive per "ADJACENCY: No moves."
- Did not run `vite build` or visually diff the rendered panel.
- Did not survey class usage outside `components/settings/`.

### Caveats
- This review answers "did the split preserve the original rules?" — yes. It does not answer "is the original CSS good?" — that is a separate design-review question for the secrets-panel owner.
