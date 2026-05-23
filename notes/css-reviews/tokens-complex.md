# tokens.css — CSS Split Fidelity Review

## Summary

**Status:** PASS

**File under review:** `/home/john/elspeth/.worktrees/css-split/src/elspeth/web/frontend/src/styles/tokens.css` (334 lines)

**Original:** `/home/john/elspeth/src/elspeth/web/frontend/src/App.css`, lines 8–209 (`:root` dark) and lines 3183–3307 (`[data-theme="light"]`).

Both source ranges land in the new file byte-for-byte. Both selector blocks are top-level. No content was reordered, dropped, or rewritten. The only additions are a 5-line file-preamble comment at the top of the new file and one extra blank line between the two blocks — both are intentional metadata permitted by `notes/app-css-split-plan.md`.

## Fidelity Verification

### 1:1 Byte-Diff Results

Compared the two source ranges with the corresponding ranges in the new file using `diff`:

| Source range (App.css) | Destination range (tokens.css) | Result |
|---|---|---|
| 11–208 (`:root { ... }`) | 10–207 | `diff` RC=0 — identical |
| 3183–3306 (`[data-theme="light"] { ... }`) | 210–333 | `diff` RC=0 — identical |

This means every property declaration, every value, every inline comment, every whitespace pattern (including the column-aligned spacing on values) is preserved.

### Declaration Counts (`grep -cE "^[[:space:]]*--"`)

| Block | Original | New |
|---|---|---|
| Dark `:root` | 121 | 121 |
| Light `[data-theme="light"]` | 78 | 78 |
| **Total** | **199** | **199** |

### Top-Level Selector Anchoring

`grep -nE "^(:root|\[data-theme)" tokens.css`:

```
10::root {
215:[data-theme="light"] {
```

Both selectors start in column 0, with no enclosing `@media`, `@supports`, or `@layer` wrapper. This satisfies the contract required by `src/styles/colorContrast.test.ts` and `src/styles/statusBadgeAccessibility.test.ts`, which parse `^:root` and `^[data-theme="light"]` as anchored regexes.

### Intentional Additions (Not in Original)

| New file lines | Content | Justification |
|---|---|---|
| 1–5 | `/* tokens.css — Design tokens. Primary: :root ... ^[data-theme="light"] as anchored selectors. */` | File-level preamble documenting the top-level-selector contract. Acts as institutional memory for future editors. Plan permits ("the barrel may emit … via a 7-line file `styles/_header.css`"). |
| 6 (blank line), 8 (blank line), 208–209 (two blank lines between blocks), 334 (trailing newline) | Whitespace | Cosmetic separation; identical or near-identical to the monolith's inter-block spacing (App.css had one blank line between line 208 and 210; new file has two). Has no semantic effect. |

The single-extra-blank-line between the dark and light blocks is the only non-byte-identical difference outside the preamble. CSS treats whitespace between rule sets as insignificant; no test or build rule depends on it.

### Declaration Order

Diff RC=0 implies order is preserved within each block. Spot-checks confirm the ordering subgroups (Backgrounds → Text → Borders → Bubbles → Badges → Semantic → Node validation → Interactive → Status → Surface variants → Scrollbar → Canvas grid → Sizing → Typography → Spacing → Radius → Control heights → Transitions → Z-index in dark; the analogous re-ordering in light) match the original.

### Comment Preservation

All in-block comments survive: the DTA/AGDS family-palette callouts, the coalesce-collision rationale, the EMPTY-status rationale, the canvas-grid contrast rationale, the WCAG 2.5.5/2.5.8 control-size rationale, the WCAG-AA contrast rationale on the light info/link tokens. No comment was dropped.

## Small-Scope Findings (Surfacing, Not Introduced by Split)

These are bugs/inconsistencies that already existed in the monolith. Surfacing them per the task's "ALSO SURFACE" directive; none of them are caused by the split.

### F1 — `--color-bg-hover` referenced but never defined (pre-existing)

- **Reference:** `src/components/header/header.css:128` — `background: var(--color-bg-hover, rgba(143, 200, 200, 0.08));`
- **Definition site:** none, in the worktree or in the monolith.
- **Status in monolith:** identical — `grep` of original `App.css` finds the same fallback-only usage at line 1253.
- **Likely intent:** the token `--color-surface-hover` (defined in both dark and light) is the canonical one. `--color-bg-hover` looks like a stale or misnamed reference relying on its inline fallback.
- **Risk:** low — fallback works, but the reference is dead-code-equivalent.
- **Recommendation:** out of scope for this PR; rename the consumer to `--color-surface-hover` or define `--color-bg-hover` formally. File as a follow-up issue.

### F2 — `--color-danger` referenced but never defined (pre-existing)

- **References:**
  - `src/components/settings/ComposerPreferencesPanel.tsx:91` — `color: "var(--color-danger, #b00020)"`
  - `src/components/chat/guided/InlineOptOutCheckbox.tsx:74` — `color: "var(--color-danger, #b00020)"`
- **Definition site:** none.
- **Status in monolith:** same — `--color-danger` never defined in `App.css`.
- **Naming inconsistency:** the semantic palette uses `--color-error` (defined), the button family uses `--color-btn-danger-bg` (defined), but consumers reach for `--color-danger` (undefined). Three names for two concepts.
- **Risk:** low — fallback `#b00020` paints, but it's a separate red from `--color-error` (`#e85653` dark / `#c93b38` light), creating a third red on the page.
- **Recommendation:** out of scope; rename TSX consumers to `--color-error`. Follow-up issue.

### F3 — Defined-but-unreferenced tokens (pre-existing dead tokens)

Tokens defined in `tokens.css` but with zero `var(--name)` consumers across the entire worktree `src/` tree:

| Token | Lines |
|---|---|
| `--color-node-valid` | 91, 302 |
| `--color-node-warning` | 92, 303 |
| `--color-node-invalid` | 93, 304 |
| `--color-node-unchecked` | 94, 305 |
| `--inspector-default-width` | 143 |
| `--inspector-min-width` | 144 |
| `--opacity-dimmed` | 126 |
| `--transition-slow` | 194 |
| `--z-overlay-backdrop` | 200 |

- **Status in monolith:** same — none of these have `var(--*)` consumers in the original tree either.
- **Likely intent:** reserved-for-future-use (the node-validation borders look intentionally pre-declared; the inspector widths look like an aborted JS-injected sizing scheme).
- **Risk:** none — dead tokens cost nothing.
- **Recommendation:** out of scope; consider pruning in a follow-up cleanup pass. The `--color-node-*` family is a quartet and may be load-bearing for a planned feature, so do not delete blindly.

### F4 — No intra-block duplicates

`grep -oE "^[[:space:]]*--[a-zA-Z0-9_-]+:" | sort | uniq -d` returns empty for both blocks individually. The cross-block duplication (every dark token re-declared in the light block) is by design — that's how the theme override works. The two pseudo-triples I initially saw (`--color-success`, `--color-info`) were false positives caused by the token names appearing in inline-comment text, not extra declarations.

### F5 — Naming consistency within tokens.css

Within the defined set:

- All colour tokens use `--color-*` (consistent).
- All sizing tokens use `--space-*`, `--radius-*`, `--size-*`, `--font-size-*`, `--line-height-*` (consistent).
- All transitions use `--transition-*` (consistent).
- All z-indexes use `--z-*` (consistent).

No `--text-color` / `--fg-color` style drift inside the file. Inconsistencies (F1, F2) are between the file and its **consumers**, not within the file.

## Adjacency / Boundary Inspection

The plan states this file is the variable layer extracted to be globally importable; no selectors moved in or out from other ranges. Verified:

- The original boundary at App.css line 208 (closing `}` of `:root`) is preserved at tokens.css line 207.
- The original boundary at App.css line 3306 (closing `}` of `[data-theme="light"]`) is preserved at tokens.css line 333.
- The original line immediately after the dark block (App.css 209, blank) and immediately before the light block (App.css 3187, `/* Light theme — applied via ... */`) are both faithfully reproduced (tokens.css 208–209 blank, 210–214 light-theme banner + selector).
- No "as described above" / cross-rule pronoun in the original points across the boundary into territory that has moved; the file is self-contained for its scope.

## Confidence Assessment

**Confidence: High.**

Basis: byte-level `diff RC=0` on both source ranges against both destination ranges; anchored-grep on `^(:root|\[data-theme)` confirms top-level; declaration counts match exactly (121 + 78 = 199); zero intra-block duplicates; the only divergences are an additive 5-line preamble and one extra inter-block blank line, both whitespace-grade and explicitly permitted by the plan.

## Risk Assessment

**Residual risk:** very low.

- The two CSS test parsers (`colorContrast.test.ts`, `statusBadgeAccessibility.test.ts`) read the file via `readFileSync`. They will see anchored `^:root` and `^[data-theme="light"]` exactly as before. The added preamble does not interfere because it starts on a comment line, not on a selector.
- The build/import path is not verified here (this review only checks file content); the barrel-import order in `index.css` is a separate review concern.
- The findings F1–F3 are pre-existing and not in scope for the split; they do not block this change.

## Information Gaps

- I did not run the frontend test suite (`npm test src/styles/colorContrast.test.ts statusBadgeAccessibility.test.ts`). A green run would convert the test-anchoring claim from inference to evidence.
- I did not verify the import order in `styles/index.css` puts tokens.css before any consumer file.
- I did not check that `App.css` no longer contains the duplicated ranges (the split's "delete-from-source" half) — that is out of scope for a tokens-file review.
- I did not inspect `tokens.ts` (the JS-side companion) for token-name drift against `tokens.css`; spot-check shows it consistently uses defined names.

## Caveats

- This review verifies fidelity of the extracted file against the monolith range. It does not verify the **integration** (import path, build order, test-runner config).
- Findings F1–F3 (`--color-bg-hover`, `--color-danger`, dead-token list) are surfaced per the task brief but are pre-existing tree state. They should be filed as separate follow-up issues, not folded into the CSS-split PR's blast radius.
- The `--color-node-*` quartet, while currently unreferenced, looks deliberately reserved. Recommend not pruning without consulting the node-validation UI plan.
