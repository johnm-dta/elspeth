# themes.css — UX / Accessibility Review

**Scope:** `/src/elspeth/web/frontend/src/styles/themes.css` — coverage sufficiency of the
`@media (prefers-contrast: more)` and `@media (forced-colors: active)` blocks. This
complements the fidelity review in `themes-complex.md`, which confirmed the split was
byte-identical. This review asks the different question: *Is the coverage adequate?*

---

## Coverage Matrix

| Visual cue | Class(es) | Colour-only signal? | Forced-colors override | Severity |
|---|---|---|---|---|
| Type badges (source/transform/gate/sink/aggregation/coalesce) | `.type-badge-*` | No — border + text + bg | `border-width: 2px` + `forced-color-adjust: none` | OK |
| Validation banner pass / fail | `.validation-banner-pass`, `.validation-banner-fail` | Yes in FC | `border: 2px solid CanvasText` + `forced-color-adjust: none` | OK |
| App-level alert banner (error) | `.alert-banner` | Yes in FC | Same as above | OK |
| App-level alert banner (info) | `.alert-banner--info` | Yes in FC | None — sibling of `.alert-banner` is covered but `--info` modifier is not | **Minor** |
| Validation banner (warning state) | `.validation-banner-warnings-section/-title/-list/-warn-item` | Yes in FC | None | **Major** |
| Validation banner button links | `.validation-banner-component-btn--warning/--error` | Yes — text-decoration colour only | None | **Minor** |
| Run status badges (7 states) | `.status-badge-pending/running/completed/failed/cancelled/cancelling/empty/completed-with-failures` | Yes — bg tint + text colour only | None | **Major** |
| Status badge cancelling pulse | `.status-badge-icon--cancelling` | Already has `prefers-reduced-motion` guard | None for FC (animation still active) | **Minor** |
| Chat bubble role | `.bubble-user`, `.bubble-assistant`, `.bubble-system` | Yes — alpha tint only | None | **Major** |
| Tool-call card state (pending/committed/rejected) | `.tool-call-card`, `--committed`, `--rejected` | Yes — `border-left-color` only | None | **Major** |
| Pending-proposal / pending-overlay cues | `.pending-proposals-banner`, `.pending-overlay-pill`, `.spec-pending-proposal`, `.yaml-pending-summary`, `.runs-pending-proposal` | Yes — dashed warning border + bg tint | None | **Major** |
| Graph canvas validation dot | `.graph-validation-dot` (8 px dot) | Yes — colour only; maps to `--color-node-valid/-warning/-invalid` | None | **Major** |
| Execution status banners | `.progress-ws-banner`, `.progress-cancelled-msg`, `.progress-failed-msg`, `.progress-errors-container`, `.progress-errors-title` | Yes — colour only | None | **Major** |
| Chat input upload error | `.chat-input-upload-alert` | Yes | None | **Minor** |
| Chat panel inline error | `.chat-panel-error` | Yes | None | **Minor** |
| Message send button active state | `.chat-input-send-btn:not(:disabled)` | Yes — accent-only colour change from disabled | None | **Minor** |
| YAML copy button copied state | `.yaml-toolbar-btn[data-copied="true"]` | Yes | `border: 2px solid Highlight` ✓ | OK (`forced-color-adjust: none` gap noted in themes-complex.md §Surfaced observations) |
| Focus indicator | `:focus-visible` | — | `outline: 2px solid Highlight` ✓ | OK |
| Graph edge | `.react-flow__edge-path` | — | `stroke: ButtonText` ✓ | OK |
| Tutorial graph nodes / chevrons / progress | `.tutorial-graph-*`, `.tutorial-progress-*` | Yes | Fully covered with CanvasText / Highlight | OK |
| Link text | `--color-link` / `<a>` elements | Yes — `#61daff` / `#176d8a` brand colour | No `LinkText` mapping | **Minor** |

**Summary:** 8 Major gaps, 6 Minor gaps.

---

## 1. Forced-Colors Mode

### 1a. System Colour Keywords

The file uses `CanvasText`, `ButtonText`, and `Highlight` correctly — these are the appropriate system-colour vocabularies and correctly omit `forced-color-adjust: none`. No hardcoded hex values appear inside the `forced-colors` block. This is correct.

**Gap:** `LinkText` is absent. `.validation-banner-component-btn` is rendered as underlined text-buttons whose purpose is navigation; under forced-colors this should map to `LinkText` so the UA can apply the correct system link colour.

### 1b. Status Badge Family (Major)

`shared.css:85–153` defines 8 `status-badge-*` states. Every state is differentiated **exclusively** by background tint and text colour (both `rgba` values that forced-colors collapses). In forced-colors mode all 8 states render identically — the operator cannot distinguish pending / running / completed / failed / cancelled / cancelling / empty / completed-with-failures at a glance.

`status-badge-icon--cancelling` does add a pulsing glyph as a secondary signal, and `prefers-reduced-motion` correctly disables it, but this glyph only differentiates *cancelling* from *cancelled* — it does not help distinguish any other pair.

**Recommended override pattern:**

```css
@media (forced-colors: active) {
  .status-badge-pending    { border: 1px solid GrayText; color: GrayText; }
  .status-badge-running    { border: 1px solid Highlight; color: Highlight; }
  .status-badge-completed,
  .status-badge-completed-with-failures { border: 1px solid ButtonText; }
  .status-badge-failed     { border: 2px solid ButtonText; }
  .status-badge-cancelled,
  .status-badge-cancelling { border: 1px dashed ButtonText; }
  .status-badge-empty      { border: 1px solid GrayText; color: GrayText; }
}
```

Using border-width, border-style, and system colour *combinations* rather than only colour encodes state in two channels simultaneously.

### 1c. Chat Bubble Role (Major)

`.bubble-user`, `.bubble-assistant`, and `.bubble-system` use rgba backgrounds (`rgba(40,130,100,0.14)`, `rgba(255,255,255,0.05)`, `rgba(255,255,255,0.03)`) and transparent borders that all collapse to Canvas under forced-colors. A screen-reader user with forced-colors active is also likely using AT — they may already have semantic context — but sighted forced-colors users lose the left/right alignment cue (already present) but also lose colour distinction. The structural alignment (`.message-row--user { justify-content: flex-end }`) survives forced-colors and partially compensates, but `.bubble-system` (centred, italic) loses its background distinction entirely.

**Recommended:** Add a `CanvasText` border to each bubble type, varying border-width (1px user, 1px assistant, 2px system) to create a non-colour shape cue.

### 1d. Tool-Call Card State (Major)

`.tool-call-card` uses `border-left: 4px solid var(--color-warning)` for pending, overridden to `--color-success` for committed and `--color-error` for rejected. All three collapse to `CanvasText`. An operator reviewing proposals in forced-colors cannot tell committed from rejected from pending.

**Recommended:** Introduce border-style variation — solid (committed), dashed (rejected), dotted (pending) — combined with the existing 4px left accent structure, using `ButtonText` or `CanvasText` as the stroke.

### 1e. Pending-Proposal State Cues (Major)

`.pending-proposals-banner`, `.pending-overlay-pill`, `.spec-pending-proposal`, `.yaml-pending-summary`, and `.runs-pending-proposal` all rely on `--color-warning` dashed borders and `color-mix()` background tints. The dashed border style survives forced-colors (border-style is not a colour), which is a partial win — the structural cue remains. However, the tinted background and warning colour disappear. The dashed border alone may suffice to signal "pending", but the components lack `CanvasText` border declarations that would ensure the dashed border is visible at all against the Canvas background.

**Recommended:** Add `border: 2px dashed CanvasText` to the pending-proposal group in forced-colors, explicitly preserving the dashed style with a system colour stroke.

### 1f. Execution and Chat Status Banners (Major / Minor)

The covered banners (`.validation-banner-pass`, `.validation-banner-fail`, `.alert-banner`) establish a pattern: `border: 2px solid CanvasText` + `forced-color-adjust: none`. The following banners with equivalent semantic weight are uncovered:

- `.progress-ws-banner` (WebSocket connectivity warning)
- `.progress-cancelled-msg` (cancellation notice)
- `.progress-failed-msg` (failure notice)
- `.progress-errors-container` / `.progress-errors-title` (error list)
- `.chat-input-upload-alert` (file-upload error)
- `.chat-panel-error` (session error)
- `.alert-banner--info` (info-level system alert — the `--info` modifier changes background/border colour but receives no override)

All of these use `--color-warning-bg/border` or `--color-error-bg/border` that collapse in forced-colors. They should be added to the existing `.validation-banner-pass, .validation-banner-fail, .alert-banner` group, or into a new parallel rule group.

### 1g. Graph Validation Dot (Major)

`.graph-validation-dot` (`inspector.css:192`) is an 8 px filled circle whose colour is set inline via `--color-node-valid/-warning/-invalid`. These three states (valid/warning/invalid per node) become indistinguishable in forced-colors — all render as `CanvasText`. A 3-state colour-only indicator with no shape fallback fails at the forced-colors boundary.

**Recommended:** Override in forced-colors with explicit shape variation. Because `.graph-validation-dot` background is set via inline style (React Flow assigns the custom property per node), a CSS forced-colors override cannot target the three states directly unless they are given class modifiers. If class modifiers exist (confirm with `.tsx` inspection), add:

```css
.graph-validation-dot--valid   { outline: 1px solid ButtonText; border-radius: 50%; }
.graph-validation-dot--warning { border: 2px solid ButtonText; }
.graph-validation-dot--invalid { outline: 2px solid ButtonText; }
```

If the dot is colour-only with no class modifiers, the component needs a structural change (not just a themes.css override).

---

## 2. Prefers-Contrast: More

### 2a. Token Deltas

The block overrides six tokens:

| Token | Dark default | Dark HC | Delta | Light default | Light HC | Delta |
|---|---|---|---|---|---|---|
| `--color-text` | `#dff0ee` | `#ffffff` | +white | `#0f2d35` | `#000000` | +black |
| `--color-text-secondary` | `#a8d0d0` | `#b0d8d8` | +~4% L | `#3a5a64` | `#1a3a44` | −25% L |
| `--color-text-muted` | `#7a9a9a` | `#96c0c0` | +~14% L | `#426069` | `#2a4a54` | −22% L |
| `--color-border` | rgba .12 | rgba .30 | +2.5× alpha | rgba .12 | rgba .30 | +2.5× alpha |
| `--color-border-strong` | rgba .25 | rgba .50 | +2× alpha | rgba .25 | rgba .50 | +2× alpha |

The alpha multiplications on border tokens are meaningful — a 2–2.5× increase in border visibility is a real help for users who rely on border cues for panel and control separation.

The text token changes are real but modest: muted text moves toward secondary, secondary moves toward primary. These lift contrast legitimately without breaking hierarchy.

### 2b. What the Block Does Not Touch

The prefers-contrast block does not override semantic colours (`--color-success`, `--color-error`, `--color-warning`, `--color-info`) or any badge colours (`--color-badge-source`, etc.) or status badge text (`--color-status-*`). This means:

- A low-vision user relying on prefers-contrast gets enhanced text and borders, but the functional badge palette (source/transform/gate/sink) remains unchanged.
- Status badge text (`--color-status-running` = `#61daff` dark, `--color-status-pending` = `#7a9a9a`) receives no lift.

Whether this is a deliberate choice (the badge hues already pass AA per `colorContrast.test.ts`) or an oversight is unclear. The existing contrast tests pass, so this is a design question rather than a WCAG failure — but it merits a comment in the file explaining the decision, especially since the block already adjusts border tokens to improve separation.

### 2c. Missing `.type-badge` Semantic Colour Override

`prefers-contrast: more` widens `.type-badge` border to 2px (good — structural, non-colour cue). But badge text colour is drawn from `--color-badge-source/-transform/-gate/-sink/-aggregation/-coalesce`, none of which are overridden. Under the dark theme these range from 7:1+ against their badge backgrounds, so no WCAG failure — but the block's border-widening intervention suggests a conscious accessibility intent. Making that intent explicit (either add semantic-colour lifts or add a comment saying "badge hues already meet AAA; border-width is the only needed intervention") would clarify the design logic for future maintainers.

---

## 3. `forced-color-adjust: none` Consistency

The existing review (`themes-complex.md`, §Surfaced observations) notes `.yaml-toolbar-btn[data-copied="true"]` sets `border: 2px solid Highlight` without `forced-color-adjust: none`, inconsistent with the 2px-border rules for validation banners and alert banners which both carry the opt-out.

**Status:** Pre-existing, carried from original App.css. The risk is that some browser/OS combinations collapse the 2px to the UA default border width under forced-colors. Adding `forced-color-adjust: none` to this rule closes the inconsistency. Severity: Minor.

---

## 4. Priority Recommendations

### Major (Fix Before Demo / Accessibility Review)

1. **Status badge family** — add forced-colors overrides using border-width + border-style + system-colour combinations. 8 states, 0 overrides.
2. **Tool-call card pending/committed/rejected** — discriminate by border-style (solid/dashed/dotted) with `CanvasText` stroke.
3. **Chat bubbles** — add `CanvasText` borders; vary border-width by role.
4. **Pending-proposal group** — add `border: 2px dashed CanvasText` to preserve dashed-style signal explicitly.
5. **Execution and chat status banners** — extend the `.validation-banner-pass, .validation-banner-fail, .alert-banner` group to include the 6 uncovered semantic banners.
6. **Graph validation dot** — requires component investigation; if class modifiers exist, add forced-colors shape overrides; if not, flag as a component-layer fix.

### Minor (Improvement Sweep)

7. **`.alert-banner--info` modifier** — add to the `.alert-banner` forced-colors group (`.alert-banner` is covered; the `--info` modifier colour-override is not).
8. **`.validation-banner-component-btn`** — add `color: LinkText` in forced-colors block for the underlined navigation text-buttons.
9. **`.yaml-toolbar-btn[data-copied]`** — add `forced-color-adjust: none` for consistency with adjacent 2px border rules.
10. **`status-badge-icon--cancelling`** — animation survives forced-colors (only `prefers-reduced-motion` suppresses it); this is acceptable, but confirm the glyph colour is not set via a custom property that collapses.
11. **Prefers-contrast intent comment** — add a short comment explaining whether the omission of semantic-colour overrides is deliberate (existing AA/AAA pass) or deferred.

---

## Confidence Assessment

**High** for gap identification and severity ratings — all findings are grounded in direct file reads. The contrast ratio claims for badge colours are inferred from the existing `colorContrast.test.ts` mention in tokens.css; those test results were not independently verified during this review.

**Not verified:** Windows High Contrast and Firefox forced-colors rendering (pure source review). Graph validation dot class-modifier structure (requires `.tsx` file inspection not performed).

---

## Risk Assessment

**Zero regression risk** from adding the recommended overrides — all proposed changes add new `@media (forced-colors: active)` rules to the existing block. No existing rules are modified or removed. The `forced-color-adjust: none` addition to `.yaml-toolbar-btn[data-copied]` is the only edit to an existing rule; it only affects forced-colors mode and only affects border rendering.

**Functional risk of not fixing:** The status badge family and tool-call card state are decision-critical surfaces. An operator using forced-colors mode has no way to tell, at a glance, whether a run completed, failed, or is still running, or whether a tool-call proposal is pending or already committed.

---

## Information Gaps

- Graph validation dot: requires inspection of the React Flow node component (`.tsx`) to determine whether the dot is assigned state classes (enabling a CSS override) or is purely colour-via-inline-style (requiring a component-layer fix).
- Contrast ratios for badge and status colours under `prefers-contrast: more` were not independently measured; relying on the assertion that `colorContrast.test.ts` covers these.
- Visual verification in an actual forced-colors environment was not performed.

---

## Caveats

- This review covers themes.css coverage sufficiency. The correctness of the split from App.css, cascade ordering, and fidelity to the original are covered by `themes-complex.md` and are out of scope here.
- Recommended overrides use system colour keywords (`CanvasText`, `ButtonText`, `GrayText`, `Highlight`, `LinkText`) only — not hardcoded values. This matches the existing pattern in the file and is the correct approach in forced-colors context; `forced-color-adjust: none` is only appropriate when preserving non-default structural properties (border-width, border-style), not brand colours.
