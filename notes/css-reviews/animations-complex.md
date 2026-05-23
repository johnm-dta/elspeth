# animations.css — Extraction Fidelity Review

**Source file**: `src/elspeth/web/frontend/src/App.css` lines 337–479
**Target file**: `src/elspeth/web/frontend/src/styles/animations.css`
**Plan**: `notes/app-css-split-plan.md`
**Adjacency**: No moves declared.
**Verdict**: **Approved — faithful byte-for-byte extraction with one banner-comment addition (not content).**

---

## 1. Intent Fit

The brief: extract App.css 337–479 (4 keyframes + their consuming rules + the global reduced-motion block) into a new `styles/animations.css`. No reordering, no rewording, no behaviour change.

| Element | Source (App.css) | Target (animations.css) | Match |
|---|---|---|---|
| `@keyframes composing-bounce` | 343–354 | 12–23 | ✓ identical frames `0%,60%,100%` and `30%`; same `transform`/`opacity` values |
| `.composing-dot` + `:nth-child(1..3)` | 356–376 | 25–45 | ✓ |
| `@keyframes progress-stripe` | 380–387 | 49–56 | ✓ `0% → background-position: 0 0`, `100% → 40px 0` |
| `.progress-bar`, `.progress-bar-stripe`, `.progress-bar-complete` | 389–420 | 58–89 | ✓ linear-gradient stops `25/25/50/50/75/75`, `background-size: 40px 40px` preserved |
| `@keyframes pulse-dot` | 422–430 | 91–99 | ✓ `0%,100% → opacity:1`, `50% → opacity:0.3` |
| `@keyframes spin` | 433–440 | 102–109 | ✓ `from 0deg → to 360deg` |
| `.spinner` | 442–450 | 111–119 | ✓ |
| `@media (prefers-reduced-motion: reduce)` block | 455–479 | 124–148 | ✓ silences `.composing-dot`, `.progress-bar-stripe`, `.spinner`, `.progress-bar-complete` (transition), `.react-flow__edge.animated path`; opacity/border-top-color values preserved |

The only non-comment difference: a 4-line header comment added at the top of `animations.css` (`/* animations.css — @keyframes (...) plus the rules that consume them plus the global @media ... */`). This is a banner orientation comment, not a content change.

## 2. Scope Discipline

- **In-scope**: lines 337–479 of App.css.
- **Declared out-of-scope**: any non-animation rules, any adjacency moves.
- **Verified**: no rules from outside 337–479 leaked in; no rules from 337–479 were dropped.
- **Undeclared additions**: only the banner comment (acceptable, informational).

## 3. Structural Integrity

| Invariant | Status |
|---|---|
| Brace balance | ✓ all blocks closed |
| Keyframe count | ✓ 4 expected, 4 present |
| `@media` fence | ✓ opens at line 124, closes at line 148 |
| CSS variable references | ✓ `--color-text-muted`, `--color-surface-elevated`, `--radius-sm`, `--color-info`, `--color-success`, `--transition-normal` — all defined elsewhere in the design-token layer (unchanged from the original; not this file's concern to verify the defs, only that the references match the source verbatim) |
| Trailing blank line inside `@media` | ✓ preserved at line 147 (matches App.css 478) |

## 4. Cross-Reference / Animation-Name Integrity

All four `@keyframes` names defined here are referenced exactly once each across the frontend codebase, and every `animation:` shorthand referring to one of these names resolves:

| Keyframe | Defined at | Consumed at |
|---|---|---|
| `composing-bounce` | `animations.css:12` | `animations.css:32` (`.composing-dot`) |
| `progress-stripe` | `animations.css:49` | `animations.css:81` (`.progress-bar-stripe`) |
| `pulse-dot` | `animations.css:91` | `shared.css:127` (`.status-badge-icon--cancelling`) — **cross-file ref, resolves** |
| `spin` | `animations.css:102` | `animations.css:118` (`.spinner`) |

**No orphan animation references** detected anywhere in `src/elspeth/web/frontend/src/`. Every `animation: <name>` shorthand points to a defined `@keyframes`:

- `composing-bounce`, `progress-stripe`, `spin`, `pulse-dot` → defined in `animations.css`
- `tutorial-progress-slide` → defined and consumed inside `tutorial.css` (self-contained, not this file's concern)

## 5. Reduced-Motion Guard Coverage

The global reduced-motion block in `animations.css` (124–148) silences:

- `.composing-dot` — guards `composing-bounce` ✓
- `.progress-bar-stripe` — guards `progress-stripe` ✓
- `.spinner` — guards `spin` ✓
- `.progress-bar-complete` transition ✓ (preserved verbatim)
- `.react-flow__edge.animated path` ✓ (preserved verbatim)

**`pulse-dot` is not silenced by this block** — but that is faithful to the original (App.css 422–430 had no reduced-motion entry for it either). The actual guard for `pulse-dot`'s only consumer lives co-located with that consumer at `shared.css:130–135`:

```css
@media (prefers-reduced-motion: reduce) {
  .status-badge-icon--cancelling {
    animation: none;
    opacity: 0.7;
  }
}
```

This is a legitimate co-location pattern (the badge rule and its guard travel together in `shared.css`), not a bypass. No animation rule in the frontend bypasses a reduced-motion guard:

- `tutorial.css` uses the inverse idiom — `@media (prefers-reduced-motion: no-preference)` to *gate the animation in*, with a separate `reduce` block providing a static fallback. This is the safer pattern (default = no motion) and equivalent in effect.

## 6. Style / Behaviour Continuity

- Indentation (2 spaces), property ordering, blank-line cadence, and comment phrasing match the original character of App.css.
- No semantic drift: timings (`1.2s`, `0.8s`, `0.6s`), easing curves (`ease-in-out`, `linear`), iteration counts (`infinite`), opacity/transform values, and CSS-variable bindings are byte-identical.

---

## Issues Found

### Critical
None.

### Major
None.

### Minor
1. **Banner comment is added but not declared in the plan.** `animations.css:1–4` is new. It is informational and useful, but the plan said "no moves" and didn't explicitly authorise comment additions. Flag for the human reviewer; not a defect.

---

## Out-of-Scope Observations

- `pulse-dot` and its reduced-motion guard are split across two files (`animations.css` defines the keyframe, `shared.css` defines both the consumer and its guard). This is acceptable but a future consolidation could either (a) move the `@keyframes pulse-dot` definition next to `.status-badge-icon--cancelling` in `shared.css`, or (b) move the consumer rule + guard into `animations.css`. Not in scope for this review.
- `tutorial.css` keeps its own `tutorial-progress-slide` keyframe self-contained — consistent with the "co-locate animation with its sole consumer" approach. Future passes may want to decide whether `animations.css` is the single source of truth for keyframes or whether feature-scoped keyframes stay co-located. Out of scope here.

---

## Confidence Assessment
**Confidence**: High.
**Basis**: Direct line-by-line comparison of App.css 337–479 against animations.css 6–148; independent grep across `src/elspeth/web/frontend/src/` for every `animation:` shorthand and every `@keyframes` definition.

## Risk Assessment
**Residual Risk**: Low. The extraction is byte-faithful aside from the documented banner comment. No animation references were broken; no reduced-motion guard was bypassed. Risk that the new file is not imported into the bundle is **not assessed here** — that is App.css refactor scope, not extraction-fidelity scope.

## Information Gaps
- Did not verify that `animations.css` is imported by the build (e.g. via App.tsx / index.css / main.tsx). The extraction may be faithful but unused until the import wiring lands.
- Did not run a visual regression / Playwright check.
- Did not verify the corresponding **deletion** of lines 337–479 from App.css (this review covers the new file only; the cut side belongs to a paired review).

## Caveats
- This review treats the extraction as a 1-to-1 move. If the broader plan intends App.css 337–479 to be **removed** after extraction, that deletion must be verified by a separate review of the modified App.css.
- The banner comment addition is the only non-content delta; if the project standard is "extraction means byte-identical, no decorations," this would be a minor reportable.
