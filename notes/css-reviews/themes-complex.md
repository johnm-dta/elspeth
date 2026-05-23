# themes.css — Complex Fidelity Review

**Files**

- NEW: `src/elspeth/web/frontend/src/styles/themes.css` (96 lines)
- ORIGINAL: `src/elspeth/web/frontend/src/App.css` lines 3368–3457

## Verdict

**Approved.** Byte-identical block migration; no fidelity loss.

## Evidence

### 1. Byte-precise diff (lines 3369–3456 of App.css vs lines 9–96 of themes.css)

```
diff <(sed -n '3369,3456p' src/elspeth/web/frontend/src/App.css) \
     <(sed -n '9,96p'    src/elspeth/web/frontend/src/styles/themes.css)
```

Returned **zero output**. The two @media blocks plus the intervening
section comment are character-for-character identical, including
whitespace and the column-aligned `--color-*` token values.

### 2. Both @media conditions present and intact

- Line 12: `@media (prefers-contrast: more)` — opens; contains nested
  `:root { ... }` (lines 13–19) and `[data-theme="light"] { ... }`
  (lines 21–27) at the *correct* (nested, non-top-level) depth. The
  token-test parsers — which key on top-level `:root` / `[data-theme]`
  blocks for the dark/light token contract — correctly ignore these.
  Confirmed by reading the file structure: the two declarations sit
  inside the `@media` brace, indented two spaces, not at column 0.
- Line 38: `@media (forced-colors: active)` — opens; closes at line 96
  (matching brace). All 12 selector groups preserved.

### 3. Selector inventory (forced-colors block)

All present, in original order:

- `.validation-banner-pass, .validation-banner-fail, .alert-banner`
  (group, with `border: 2px solid CanvasText` + `forced-color-adjust: none`)
- `.type-badge-source, .type-badge-transform, .type-badge-gate,
  .type-badge-sink, .type-badge-aggregation, .type-badge-coalesce`
  (group, `border-width: 2px` + `forced-color-adjust: none`)
- `.react-flow__edge-path` — `stroke: ButtonText`
- `:focus-visible` — `outline: 2px solid Highlight`
- `.yaml-toolbar-btn[data-copied="true"]` — `border: 2px solid Highlight`
- `.tutorial-graph-node` — `border: 1px solid CanvasText` +
  `forced-color-adjust: none`
- `.tutorial-graph-chevron` — `color: CanvasText`
- `.tutorial-progress-dot` — `border: 1px solid CanvasText`
- `.tutorial-progress-dot--active` — `background: Highlight`
- `.tutorial-progress-bar` — `border: 1px solid CanvasText`
- `.tutorial-progress-bar::after` — `background: Highlight`

### 4. No leaked selectors

Every selector inside themes.css is either:
- a theme token override (`:root`, `[data-theme="light"]` — both nested
  inside `@media (prefers-contrast: more)`), or
- a high-contrast / forced-colors override.

No component-default rules slipped in.

### 5. Import order — themes.css is last

`styles/index.css:36` ends with `@import "./themes.css";`, preceded by a
header comment (line 10) explicitly noting "themes.css imports LAST".
This matches the Path B design and preserves the original cascade
ordering (the block was the file's last @media in App.css before the
markdown rules at 3458+).

## Surfaced observations

### `forced-color-adjust` symmetry

The rule of thumb checked: any selector setting an explicit
non-system color *needs* `forced-color-adjust: none` to prevent the UA
from overriding it; selectors setting system colors (`CanvasText`,
`Highlight`, `ButtonText`) inside `forced-colors` mode do **not**
require the opt-out, because system colors are the expected vocabulary
in that mode.

Audit of each rule:

| Selector | Sets | `forced-color-adjust: none`? | Verdict |
|---|---|---|---|
| `.validation-banner-*, .alert-banner` | `border: 2px solid CanvasText` | yes | OK — `border` width is non-default; opt-out keeps the 2px |
| `.type-badge-*` | `border-width: 2px` | yes | OK — same reasoning |
| `.react-flow__edge-path` | `stroke: ButtonText` (SVG) | no | OK — SVG `stroke` to a system color; no override needed |
| `:focus-visible` | `outline: 2px solid Highlight` | no | OK — Highlight is system; UA already respects |
| `.yaml-toolbar-btn[data-copied="true"]` | `border: 2px solid Highlight` | no | **Minor risk** — see below |
| `.tutorial-graph-node` | `border: 1px solid CanvasText` | yes | OK |
| `.tutorial-graph-chevron` | `color: CanvasText` | no | OK — system color |
| `.tutorial-progress-dot{,--active}` | `border` / `background` system | no | OK |
| `.tutorial-progress-bar{,::after}` | `border` / `background` system | no | OK |

**Minor (carried from original, not introduced by split):**
`.yaml-toolbar-btn[data-copied="true"]` gets a 2px Highlight border
without `forced-color-adjust: none`. The other 2px borders in this
block (validation banners, alert banner, tutorial-graph-node) all pair
the explicit width with the opt-out. Because the system *might*
collapse the border to its own width in forced-colors mode, this
selector could lose its 2px in some browsers while its siblings keep
theirs. This is a **fidelity preservation** — the original at App.css
line 3424–3426 has the same shape — so it is **not** a split defect,
but worth surfacing if a follow-up sweep ever audits forced-colors
discipline.

### Possibly-redundant prefers-contrast rules

- `.type-badge { border-width: 2px; }` (line 30) — defaults elsewhere
  set `.type-badge` border width to **1px** (verified by grep finding
  `.type-badge` defined in `common.css` / sibling files). This is a
  real override, not a duplicate. OK.
- `:focus-visible { outline-width: 3px; }` (line 34) — default
  focus-visible outline is 2px elsewhere; this widens to 3px under
  high-contrast preference. Real override. OK.

No prefers-contrast rule duplicates a default.

### Dangling-override risk

All base selectors referenced in this file resolve to at least one
non-themes.css source file (grep audit, 15/15 selectors hit ≥1 other
file). No silently orphaned override.

### Cascade implications under Path B

Original position: ~line 3372 of a single 7335-line stylesheet — last
@media block before the `Markdown rendering (A2)` section. Under Path B
themes.css is imported last in `styles/index.css`, so it applies *after*
every component CSS file. For these rules (all inside `@media` queries
that wrap targeted overrides), this matches the original cascade
behaviour: they continue to win against unconditional component rules
of equal specificity, and the relative ordering against any later
@import would only matter if a *later* file also defined
`@media (forced-colors)` or `@media (prefers-contrast)` rules on the
same selectors — none do.

## Confidence Assessment

**High.** Verified by byte-precise diff (returned empty), structural read of
the file, sibling-file grep for every overridden selector, and import-
order check in `styles/index.css`. The migration is a clean lift.

## Risk Assessment

**Negligible** for this split. The pre-existing `forced-color-adjust`
asymmetry on `.yaml-toolbar-btn[data-copied="true"]` is preserved
from the original, not introduced. No selectors leaked in or out.

## Information Gaps

- Did not visually verify rendering in forced-colors mode (Windows
  High Contrast / Firefox `MSHighContrast`). Pure source review.
- Did not enumerate every component CSS file post-split to confirm
  the *full* set of base rules for each selector is intact — only
  confirmed each selector exists in ≥1 non-themes file.

## Caveats

- Review scope was themes.css fidelity only. Sibling-file completeness,
  the rest of the App.css → styles/* split, and the index.css import
  graph beyond the `themes.css` last-position invariant were not
  re-verified here; trust them to their own reviews.
