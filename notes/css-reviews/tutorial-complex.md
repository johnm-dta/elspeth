# tutorial.css — CSS-split fidelity review (complex)

**Reviewer**: complex-reviewer (CSS-split fidelity track)
**Date**: 2026-05-23
**Scope**: Verify `src/elspeth/web/frontend/src/components/tutorial/tutorial.css`
is a byte-faithful extraction of lines 2708–3181 of
`src/elspeth/web/frontend/src/App.css`, with no orphans, no scope sprawl,
and no missed tutorial-owned content elsewhere in App.css.

---

## Summary

**File**: `src/elspeth/web/frontend/src/components/tutorial/tutorial.css`
(481 lines; new file)
**Source**: `src/elspeth/web/frontend/src/App.css` lines 2708–3181
**Edits reviewed**: 1 extraction (no moves)
**Overall verdict**: **Approved with one Major finding** — the extracted
range is byte-identical to the source, but 6 tutorial-owned rules in the
`@media (forced-colors: active)` block at App.css 3428–3455 are NOT in
the brief's "expected source line range" and have NOT been migrated. The
extraction as performed matches the brief, but the brief itself appears
to under-scope the tutorial surface. Surfacing for operator decision.

---

## Intent Fit

| Requested Change | Found in File | Correct? | Evidence |
|------------------|---------------|----------|----------|
| Extract `.tutorial-*` rules from App.css 2708–3181 | tutorial.css 11–481 | ✓ | byte-diff identical (below) |
| Include `@keyframes tutorial-progress-slide` (co-located intentional) | tutorial.css 294–301 | ✓ | present, identical |
| Preserve `@media (prefers-reduced-motion: no-preference / reduce)` | tutorial.css 279–292 | ✓ | both blocks present, identical |
| Preserve `@media (max-width: 760px)` | tutorial.css 457–481 | ✓ | identical, includes `.tutorial-shell`, `.tutorial-turn`, `.tutorial-layer-grid`/`.tutorial-summary-grid`/`.tutorial-audit-list`, `.tutorial-graph`, `.tutorial-graph-chevron` overrides |
| Add file header docstring | tutorial.css 1–6 | ✓ | one-block ownership comment, lists owned selectors + keyframes |

**Byte-fidelity proof** (run during review):

```
$ diff <(sed -n '2711,3181p' .../App.css) <(sed -n '11,481p' .../tutorial.css)
(no output — BYTE-IDENTICAL)
```

All 56 `.tutorial-*` selector lines in App.css 2708–3181 reproduce
exactly in tutorial.css (selector count matches: 56 in each range).

---

## Scope Discipline

**Declared scope (per brief)**: App.css lines 2708–3182, comprising
`.tutorial-*` selectors + `@keyframes tutorial-progress-slide` +
`(max-width: 760px)` media + inline `(prefers-reduced-motion)`.

**Actually extracted**: exactly that range, byte-identical. No
opportunistic edits, no reflow, no whitespace drift. The only
non-source addition is the 6-line header docstring at tutorial.css 1–6.

**Undeclared changes**: none.

---

## Structural Integrity

| Invariant | Status | Evidence |
|-----------|--------|----------|
| Brace balance | ✓ | extracted block self-contained; opens and closes within the 471-line body |
| `@media` blocks closed | ✓ | 3 media blocks (lines 279, 285, 457) each open + close cleanly |
| `@keyframes` block closed | ✓ | tutorial.css 294–301 |
| Selector count parity | ✓ | 56 `.tutorial-*` rules in source range, 56 in extracted file |
| Comment preservation | ✓ | All institutional-memory comments preserved verbatim (e.g. the "static 60% read as stuck" rationale at 247–254, the "neutral emphasis" green-success rationale at 100–109, the "compact button — does not require .btn's 44px hit area" comment at 377–390) |
| CSS variable references unchanged | ✓ | every `var(--...)` token identical to source |
| Reduced-motion guard present | ✓ | both `no-preference` (animates) and `reduce` (static centred chunk) variants migrated |

---

## Cross-Reference Integrity

**Custom-property references** (all defined in App.css base / variables;
must remain resolvable when tutorial.css is loaded alongside App.css or
its base/variables successor):

- Spacing: `--space-xs`, `--space-sm`, `--space-md`, `--space-lg`,
  `--space-xl`, `--space-2xl`
- Color: `--color-bg`, `--color-surface`, `--color-surface-elevated`,
  `--color-surface-input`, `--color-surface-hover`, `--color-border`,
  `--color-border-strong`, `--color-text`, `--color-text-muted`,
  `--color-text-secondary`, `--color-info`, `--color-link`,
  `--color-error`, `--color-error-bg`, `--color-error-border`,
  `--color-warning`, `--color-warning-bg`, `--color-warning-border`,
  `--color-state-positive`, `--color-badge-source-border`,
  `--color-badge-source-bg`
- Radii / sizes / type: `--radius-sm`, `--radius-md`, `--radius-lg`,
  `--radius-pill`, `--size-control`, `--font-size-xs`,
  `--font-size-sm`, `--font-size-lg`, `--font-size-xl`,
  `--line-height-tight`, `--font-mono`

All of these are present in App.css's `:root` block (verified
incidentally — none were renamed). **No broken `var()` references.**

**Internal cross-reference** — `animation: tutorial-progress-slide` at
line 281 resolves to `@keyframes tutorial-progress-slide` at line 294
**within the same file**. Co-location is intentional per the brief
(tutorial-specific keyframes do not move to animations.css). Verified.

---

## Orphan / Dead-Code / Boundary Findings

- **App.css 2706–2707 (boundary above extraction)**: the section header
  "First-run tutorial" (lines 2708–2710) was extracted with the rules.
  No orphaned header left behind in App.css. *(Verification of App.css
  post-extraction state is out of scope for this review — the brief
  scopes the new file only — but worth flagging to the App.css editor.)*
- **App.css 3182–3183 (boundary below extraction)**: not part of this
  review's scope.
- **tutorial.css internal boundaries**: no dangling lead-ins, no broken
  transitions, no list-introducer/list mismatches. All comments still
  reference visible adjacent CSS (e.g. the "see media query below"
  comment at 156–157 still refers to the live `(max-width: 760px)`
  block at 457).
- **No dead rules** in the extracted body.

---

## ALSO-SURFACE Items (per brief)

### 1. Tutorial-specific keyframes that should have moved to animations.css

**None outside tutorial.css.** The only tutorial keyframe is
`tutorial-progress-slide`, intentionally co-located per the plan. The
indeterminate-progress shimmer is tutorial-only chrome and would have
no consumers in animations.css. **Co-location is correct.**

The opposite check — did tutorial.css end up depending on a keyframe
*owned by* animations.css? — also clean: the only `animation:`
declaration in tutorial.css (line 281) names `tutorial-progress-slide`,
which is defined locally.

### 2. Tutorial reduced-motion guard missing

**Not missing.** Both halves of the guard are present and correct:

- `@media (prefers-reduced-motion: no-preference)` (line 279) gates the
  sliding animation — so motion is opt-in by default-no-preference, not
  always-on.
- `@media (prefers-reduced-motion: reduce)` (line 285) provides the
  static-chunk fallback so the affordance still communicates "something
  is happening" without motion.

This is the architecturally correct pattern (animation gated by
no-preference rather than the more common "always animate, then
suppress under reduce"). Preserved verbatim.

### 3. Dead rules

**None found** in the extracted range.

---

## MAJOR FINDING — Scope under-statement in the brief

The brief lists the expected source range as **App.css 2708–3182**.
However, App.css lines **3428–3455** contain six additional
tutorial-owned rules inside the `@media (forced-colors: active)` block
(App.css 3398–3456):

```css
/* App.css 3428–3455 — inside @media (forced-colors: active) */
.tutorial-graph-node      { border: 1px solid CanvasText; forced-color-adjust: none; }
.tutorial-graph-chevron   { color: CanvasText; }
.tutorial-progress-dot    { border: 1px solid CanvasText; }
.tutorial-progress-dot--active { background: Highlight; }
.tutorial-progress-bar    { border: 1px solid CanvasText; }
.tutorial-progress-bar::after  { background: Highlight; }
```

These are introduced by a load-bearing comment (App.css 3428–3431):

> "First-run tutorial — high-contrast fallbacks. Without these the
> graph nodes render borderless, the active progress dot is
> indistinguishable from inactive dots, and the indeterminate progress
> bar disappears into the surface."

**Severity**: Major (not Critical). The extraction as performed is
faithful to the *brief*. But the brief under-scoped: these six rules
are tutorial-owned chrome by every reasonable criterion (selectors all
in the tutorial namespace, comment ties them directly to tutorial UI,
removing them degrades only tutorial UX), and shipping the CSS split
without migrating them leaves a permanent split-ownership pattern
where tutorial chrome lives in two files.

**Recommended resolution** (operator-level decision):

- **Option A (recommended)**: Migrate these six rules into tutorial.css
  inside a new `@media (forced-colors: active)` block at the end of
  the file. Delete from App.css in the same commit. This keeps the
  one-file-per-component invariant the split is establishing.
- **Option B**: If the plan deliberately keeps `forced-colors: active`
  consolidated in a single App.css block (cross-component contrast
  policy lives together), document that explicitly — both in the
  tutorial.css header docstring and in the App.css block comment — so
  the next CSS-split increment doesn't re-litigate this.

This is a Major rather than Critical because the build still works
and the tutorial still renders correctly under forced-colors today; it
is a *structural-debt* finding, not a behavioural-defect finding.

**Note on review scope**: The brief explicitly bounded the review to
the new file's fidelity vs. lines 2708–3181. I am flagging this as an
ALSO-SURFACE item (per the brief's explicit instruction to surface
"any tutorial-specific [content] that should have moved"). The
extraction itself is approved; the scope of the extraction is the
finding.

---

## Style / Behavior Continuity

| Aspect | Source | Extracted | Match? |
|--------|--------|-----------|--------|
| Indentation (2-space) | yes | yes | ✓ |
| Comment style (`/* ... */`, multi-line block comments aligned) | yes | yes | ✓ |
| Property ordering within rules | source-order | source-order | ✓ |
| Trailing semicolons on final property | present | present | ✓ |
| Custom-property naming convention | hyphenated | hyphenated | ✓ |
| Selector grouping (e.g. `.tutorial-layer-grid, .tutorial-summary-grid`) | preserved | preserved | ✓ |

No discontinuities at the file head (lines 1–10 docstring → 11 first
rule). The header comment uses the same `/* --- ... --- */` divider
style that surrounds section headers elsewhere in App.css. Consistent.

---

## Issues Found

### Critical
None.

### Major
1. **App.css 3428–3455 — six tutorial-owned high-contrast fallbacks
   not migrated.** Detailed above. Resolution: either extend the
   extraction to include the `@media (forced-colors: active)` tutorial
   subset, or document the deliberate consolidation policy.

### Minor
1. **Header docstring line 6** lists `.tutorial-progress-dot,
   @keyframes tutorial-progress-slide + responsive media` on one line.
   Cosmetic. The brief says "responsive media" — accurate; the
   `prefers-reduced-motion` blocks are not separately enumerated. If
   the docstring intends to be authoritative for grep-ability, list
   the two `prefers-reduced-motion` queries explicitly. Optional.

---

## Out-of-Scope Observations
- This review does not inspect the App.css *post-extraction* state
  (whether lines 2708–3181 have been deleted, whether the entry point
  now `@import`s tutorial.css, etc.). That belongs to a paired App.css
  delta review.
- General CSS quality (idiomatic patterns, redundancy with other
  components, BEM-vs-loose-naming) is out of scope — this is
  extraction-fidelity, not style review. For absolute style review,
  defer to a CSS-focused reviewer.

---

## Confidence Assessment

**Confidence**: **High** for the within-file extraction; **High** for
the scope-gap finding (App.css 3428–3455 verified by direct grep and
read).

**Basis**: Byte-level diff of the source range against the extracted
body returned identical output. Selector count parity (56 = 56)
independently confirms no rule was dropped or duplicated.
Cross-reference scan (`var(--...)` tokens) confirms no renamed or
dropped custom property. The forced-colors scope gap is verified by
direct read of App.css 3398–3456 plus a search for all `tutorial`
occurrences outside 2708–3181 (returns exactly the 6 selectors plus
the load-bearing comment, nothing else).

---

## Risk Assessment

**Residual Risk**: Low for the file itself. The only behavioural risk
is the forced-colors scope-gap finding — *if* App.css's
`@media (forced-colors: active)` block is later refactored or moved
without preserving the six tutorial fallbacks, the tutorial will lose
high-contrast affordances silently. The mitigation (Option A above)
removes that latent risk.

---

## Information Gaps

- Did not run a CSS parser / build (e.g. `vite build` or `postcss`) to
  confirm tutorial.css parses standalone. Visual inspection is
  sufficient (no unclosed braces, no dangling at-rules), but a build
  check would be a stronger validation.
- Did not verify the React component (`Tutorial*.tsx`) imports
  tutorial.css. Out of scope for the brief, but a paired component
  edit must exist for the split to function.
- Did not verify the App.css post-extraction state.
- Did not check tutorial.css against the broader `components/` CSS
  layout convention (does the rest of the migration use a
  `components/<name>/<name>.css` shape? — assumed yes from the path).

---

## Caveats

- This review treats the brief's expected range (2708–3182) as the
  contract for *what was extracted*, and the byte-diff confirms
  conformance. The Major finding is that **the brief's range is
  incomplete relative to the semantic ownership of the tutorial
  component**. The writer executed the brief correctly; the brief
  itself is the gap. Either the operator extends the scope, or
  documents the deliberate consolidation.
- The `@keyframes tutorial-progress-slide` co-location is approved per
  explicit brief instruction. This review does not relitigate that
  choice; the only check performed was that no *other* tutorial
  keyframes exist that should have migrated to animations.css (none
  do — verified by full-file scan of App.css for `@keyframes` near
  tutorial selectors).
