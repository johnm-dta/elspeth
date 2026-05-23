# recovery.css — Split Fidelity Review

**File reviewed**: `src/elspeth/web/frontend/src/components/recovery/recovery.css` (146 lines)
**Compared against**: `src/elspeth/web/frontend/src/App.css` lines 989–1131 (canonical pre-split source)
**TSX consumers inspected**: `RecoveryPanel.tsx`, `RecoveryDiff.tsx`, `RecoveryTranscript.tsx` (plus their `.test.tsx` companions)
**Verdict**: **APPROVED.** The split is byte-identical against the declared source range. Several non-blocking observations follow.

---

## 1. Intent Fit — PASS

Brief: extract `.recovery-panel*`, `.recovery-diff*`, `.recovery-transcript*` and the trailing `@media (max-width: 900px)` block from `App.css` 989–1131 into a component-local stylesheet, no moves, no edits.

Performed exactly. New file body (lines 4–145) is the requested 142-line slice with a 2-line ownership header prepended (lines 1–2) and a final newline (line 146). The leading `Composer Recovery Panel` banner comment from line 989 is preserved verbatim at lines 4–6.

## 2. Scope Discipline — PASS

`diff <(sed -n '989,1130p' App.css) <(sed -n '4,145p' recovery.css)` → **empty.** No reformatting, no token substitution, no rule reordering, no whitespace normalisation. The added two-line ownership comment is the only divergence from the source range, and it is outside the rule body.

## 3. Structural Integrity — PASS

| Invariant | Status | Evidence |
|---|---|---|
| Brace balance | OK | 18 `{` / 18 `}` (16 rule blocks + the media-query wrapper, the inner of which contributes both) |
| Media-query block preserved | OK | Lines 133–145, identical inner declarations and selector list |
| Declaration order | OK | All 16 rule blocks appear in the same order as App.css 992–1116 |
| Selector text byte-identical | OK | `grep -oE '\.recovery-[a-z-]+'` on both files yields identical sorted sets of 21 distinct selectors |
| Custom-property usage | OK | All `var(--…)` references unchanged (`--z-dialog-backdrop`, `--z-dialog`, `--space-{sm,md,lg,xl}`, `--color-surface-paper`, `--color-border`, `--color-warning{,-bg,-border}`, `--color-text-secondary`, `--radius-{sm,md,lg}`, `--font-size-{xs,sm}`) |
| Trailing newline | OK | File ends with newline after line 145 |

## 4. Cross-Reference / Call-Site Integrity — PASS (for the split itself)

Selectors emitted by the new file fully match the `.recovery-*` rules in App.css 989–1131. Independently verified there are **no `.recovery-*` rules anywhere else in App.css** — every selector in the original range moved, none were left behind. (`grep -n '\.recovery-' App.css` lists only lines 992–1127.)

## 5. Orphan / Dead-Code / Boundary Findings

### 5a. Selectors with no observed TSX consumer (dead-or-elsewhere)

Cross-referencing CSS selectors against `className=` strings in `components/recovery/*.tsx`:

| CSS selector | Used by recovery TSX? | Notes |
|---|---|---|
| `.recovery-panel-backdrop` | ✓ `RecoveryPanel` | |
| `.recovery-panel` | ✓ `RecoveryPanel` | |
| `.recovery-panel-header` | ✓ `RecoveryPanel` | |
| `.recovery-panel-evidence` | ✓ `RecoveryPanel` | |
| `.recovery-panel-actions` | ✓ `RecoveryPanel` | |
| `.recovery-panel-confirm-actions` | ✓ `RecoveryPanel` | |
| `.recovery-panel-confirm` | ✓ `RecoveryPanel` | |
| `.recovery-panel-reason` | ✓ `RecoveryPanel` | |
| `.recovery-panel-body` | ✓ `RecoveryPanel` | |
| `.recovery-panel-transcript-controls` | ✓ `RecoveryPanel` | |
| `.recovery-diff-summary` | ✓ `RecoveryDiff` | |
| `.recovery-diff-list` | ✓ `RecoveryDiff` | |
| `.recovery-diff-row` | ✓ `RecoveryDiff` | |
| `.recovery-diff-row-title` | ✓ `RecoveryDiff` | |
| `.recovery-diff-group-header` | ✓ `RecoveryDiff` | |
| `.recovery-transcript` | ✓ `RecoveryTranscript` | |
| `.recovery-transcript-tools` | ✓ `RecoveryTranscript` | (descendant `ul`) |
| `.recovery-transcript-tool-rows` | ✓ `RecoveryTranscript` | |
| `.recovery-transcript-tool-call` | ✓ `RecoveryTranscript` | |
| `.recovery-transcript-tool-title` | ✓ `RecoveryTranscript` | |

**All 21 selectors in recovery.css are consumed.** No dead rules introduced by the split.

### 5b. TSX classes with no CSS rule (pre-existing, not split-induced)

The TSX components reference 13 `recovery-*` classes for which **no rule exists in the original App.css and therefore none in the new file**:

```
recovery-diff-compact
recovery-diff-group
recovery-diff-group--{added,removed,changed,…}      ← BEM modifier, kind interpolated
recovery-diff-row--{added,removed,changed,…}         ← BEM modifier, kind interpolated
recovery-diff-row-change
recovery-diff-title
recovery-panel-apply
recovery-panel-discard
recovery-panel-title
recovery-transcript-assistant
recovery-transcript-missing
recovery-transcript-note
recovery-transcript-title
```

These are **pre-existing** in `RC5.2` — they are styling hooks the components emit that the stylesheet has never defined. The split correctly does not invent rules for them. `recovery-panel-apply` and `recovery-panel-discard` are paired with `btn btn-primary` / `btn btn-danger` so the button gets base styling from `.btn`; the recovery-specific modifier is genuinely unstyled. The BEM-style `--{kind}` variants are dynamic template strings (`` `recovery-diff-row recovery-diff-row--${entry.kind}` ``), so this is a missing-modifier-style situation — likely intentional graceful degradation, but worth a follow-up ticket to either define the variants or drop the class composition. **Out of scope for this PR.**

### 5c. Boundary inspection

- No content immediately precedes line 4 of the new file other than the 2-line ownership banner; the original `Composer Recovery Panel` heading-comment is intact at lines 4–6.
- The file ends immediately after the media-query closing brace (line 145) plus a single newline. No trailing dead content, no doubled newlines.
- No orphaned commas, no truncated selector lists, no stray `}` from a neighbouring block.

## 6. Style / Behaviour Continuity — PASS

Indentation (two-space), declaration order (`position`, `inset`, then visual properties), comment style (`/* --- banner --- */`), and BEM-flavoured selector naming all match neighbouring split files (`base-complex.md`, `animations-complex.md` reviews exist alongside, confirming this is the established pattern). The added file-header banner (lines 1–2) is short, factual, and uses the same comment delimiters as the section banners.

## 7. Hardcoded colours that should be tokens (surfaced per task brief)

Two rgba literals survive in the new file (and survived in the source):

| Line | Declaration | Note |
|---|---|---|
| 10 | `background-color: rgba(0, 0, 0, 0.45);` (`.recovery-panel-backdrop`) | Modal backdrop scrim. Worth tokenising as `--color-scrim` or similar — likely shared with other modal/dialog backdrops elsewhere in the codebase. |
| 28 | `box-shadow: 0 8px 32px rgba(0, 0, 0, 0.25);` (`.recovery-panel`) | Modal lift shadow. Worth tokenising as `--shadow-modal` / `--elevation-modal`. |

Both are **carried forward from the source** — not introduced by this split — so they are out of scope for this PR but should be filed as a tokenisation follow-up across the modal/dialog family (the same literals likely recur in `ExplainDialog`, `RecoveryPanel`, any preferences modal, etc.).

## 8. Duplicate rules — NONE

No selector appears in two rule blocks. The grouped selectors (lines 16–22, 28–30, 34–38, 89–92, 98–100, 105–106) are single rule blocks with comma-separated selector lists — that is intentional, not duplication.

---

## Issues Found

### Critical — NONE

### Major — NONE

### Minor — NONE (split-scoped)

### Out-of-scope observations (do not block this PR)

1. **13 unstyled `recovery-*` classes** are emitted by the TSX (see §5b). Either define them or remove them. Tracking ticket recommended.
2. **Two hardcoded rgba literals** (scrim + modal shadow) should be promoted to design tokens shared across the dialog family (see §7).
3. The ownership header comment at line 2 mentions `+ media query` — accurate, but if more split files follow this convention it might be worth a more uniform "Owners:" syntax across files. Not actionable here.

---

## Confidence Assessment

**Confidence**: High.
**Basis**: byte-level `diff` of the declared source range against the new file body returned empty; brace counts, selector sets, and declaration order verified independently; TSX consumer cross-reference performed via grep across all three components and their tests; original App.css confirmed to contain no `.recovery-*` rules outside the declared range.

## Risk Assessment

**Residual risk**: low. The only ways this split could still break runtime styling are (a) the new file isn't imported by the recovery components — **not verified here**, the brief scoped the review to fidelity, not wiring; (b) cascade-order semantics with other split files where overlapping selectors could land in a different order at import time — no overlapping `.recovery-*` selectors exist in App.css, so this is not a concern for this slice; (c) build tooling that processes App.css differently from a component-local CSS file (e.g., PostCSS plugin scope). Recommend a smoke test of the recovery panel under `npm run build` + Playwright open-the-panel to close (a) and (c).

## Information Gaps

- Did not verify that `recovery.css` is imported by any of `RecoveryPanel.tsx`, `RecoveryDiff.tsx`, `RecoveryTranscript.tsx`, or by a barrel/index file. The brief scoped this review to source-range fidelity, not wiring; flag for the wiring reviewer.
- Did not run `npm run build` or Playwright to confirm visual parity.
- Did not check whether App.css lines 989–1131 have been deleted (or are still duplicated) in the post-split App.css — the brief named the original file as the canonical pre-split reference, so I treated it as immutable for this review.

## Caveats

- This review verifies **fidelity of extraction**, not **correctness of the CSS itself**. Pre-existing issues (unstyled classes, hardcoded colours) are surfaced for follow-up but do not affect the split verdict.
- The 142-line slice was verified against the path `/home/john/elspeth/src/elspeth/web/frontend/src/App.css` — the **main checkout**, not the worktree. If App.css differs between main and worktree, that delta is not captured here.
