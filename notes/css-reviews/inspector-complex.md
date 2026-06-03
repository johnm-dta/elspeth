# Edit Review: src/elspeth/web/frontend/src/components/inspector/inspector.css

## Summary
**File**: `src/elspeth/web/frontend/src/components/inspector/inspector.css` (325 lines, type: CSS)
**Edit report received**: Yes (paths, expected source ranges, adjacency moves)
**Source**: `src/elspeth/web/frontend/src/App.css` lines 3762–3859 ∪ 3865–4082
**Edits reviewed**: 1 extraction (version-selector + graph-view fragments) + 1 declared move-out (`.side-rail-error-banner`)
**Overall verdict**: **Approved with minor fixes** (two trivial whitespace deltas; no semantic drift)

## Intent Fit

| Requested change | Found in file | Correct? | Evidence |
|------------------|---------------|----------|----------|
| Carry rules from App.css 3762–3859 (.version-selector*) | Lines 12–105 | Yes | Selector-by-selector diff against `/tmp/expected.css` clean except whitespace |
| Drop `.side-rail-error-banner` (3860–3863) | Absent from file | Yes | `grep side-rail-error-banner inspector.css` → 0 hits |
| Carry rules from App.css 3865–4082 (react-flow theming + graph-view + graph-node + graph-validation-dot + graph-config + media query) | Lines 108–324 | Yes | Includes the two react-flow theming blocks at 111 and 138, `:focus-visible` rule at 144, `@media (max-width: 760px)` at 309 |

## Scope Discipline
**Declared out-of-scope**: nothing else moves IN or OUT besides `.side-rail-error-banner` OUT.
**Actually preserved**: Verified — no other selectors added or removed.
**Undeclared changes**: A header comment was added (lines 1–7) describing ownership and the GraphView.test.tsx path-binding. This is informational documentation, not a selector change; acceptable for a split deliverable but technically undeclared. Flag, not block.

## Structural Integrity

| Invariant | Status | Evidence |
|-----------|--------|----------|
| Brace balance | Pass | All `{ ... }` blocks close; 27 opens / 27 closes from grep |
| Selector text byte-identical | Pass (modulo whitespace) | `diff /tmp/expected.css /tmp/actual.css` produces only `98a99` (extra blank line) and `316d316` (missing trailing newline) — no selector-body deltas |
| `.side-rail-error-banner` absence here | Pass | `grep -n side-rail-error-banner inspector.css` → 0 hits |
| `.side-rail-error-banner` presence in sidebar.css | Pass | `sidebar.css:196:.side-rail-error-banner {` |
| `:root .react-flow.react-flow` block present | Pass | Line 111 |
| `[data-theme="light"] .react-flow.react-flow` block present | Pass | Line 138 |
| `.react-flow__controls-button:focus-visible` present | Pass | Line 144 |
| `@media (max-width: 760px)` present and closed | Pass | Lines 309–324 |

## Cross-Reference / Call-Site Integrity

### GraphView.test.tsx path binding (the load-bearing one)
The edit report flagged that `GraphView.test.tsx` reads this file by path. Verified:

- `GraphView.test.tsx:353` reads `"src/components/inspector/inspector.css"` — matches the new path exactly.
- `GraphView.test.tsx:363` regex:
  ```
  /\[data-theme="light"\]\s+\.react-flow\.react-flow\s*\{[\s\S]*--xy-minimap-mask-background-color-default:\s*rgba\(15, 45, 53, 0\.12\);/
  ```
  Matches against `inspector.css:138–139`:
  ```
  [data-theme="light"] .react-flow.react-flow {
    --xy-minimap-mask-background-color-default: rgba(15, 45, 53, 0.12);
  ```
  Regex pattern matches; will pass.
- Other `toContain` assertions at lines 355–361, 364–365 all verified present in inspector.css (background, controls-button-background, controls-button-color, minimap-background, minimap-mask-stroke, edge-stroke-selected, `:root .react-flow.react-flow`, `react-flow__controls-button:focus-visible`, `outline: 2px solid var(--color-focus-ring);`).

### TSX class consumers
All `.graph-*`, `.version-selector*`, and `.react-flow__controls-button` selectors trace to TSX:
- `version-selector*` → `components/header/HeaderVersionSelector.tsx`
- `graph-view*`, `graph-node*`, `graph-validation-dot`, `graph-config*` → `components/inspector/GraphView.tsx` (confirmed at lines 163, 167, 170, 182, 185, 212, 215, 237, 241, 254, 259, 267, 275)
- `react-flow__controls-button:focus-visible` is a global override against the react-flow library DOM (no TSX className needed — react-flow injects it).

No orphaned `.graph-*` rules detected.

## Orphan / Dead-Code / Boundary Findings

- **Lines 105–110** (excision boundary where `.side-rail-error-banner` was removed): contains an extra blank line vs source. Source had `}` → blank → `.side-rail-error-banner` block → blank → `/* Graph View */` comment. New file has `}` → blank → blank → `/* Graph View */`. Cosmetic only; no rule orphaned. **Minor.**
- **End of file** (line 325): missing the trailing newline that the source carried after `}` at App.css:4081 (with App.css:4082 blank). **Minor.**
- No dangling comments, no rules referencing removed selectors.

## Style / Behavior Continuity

| Edit | Surrounding style | Inserted style | Match? |
|------|-------------------|----------------|--------|
| Whole-file extraction | App.css uses 2-space indent, lowercase hex, kebab-case selectors, banner comments with hyphen separators | Identical | Pass |

Header comment block (lines 1–7) follows the convention seen in `sidebar.css` (also a split target with a comparable banner). Consistent.

## Issues Found

### Critical
None.

### Major
None.

### Minor
1. **[Line 99]** Extra blank line between `.version-selector-loading {}` block (closes at line 105) and the `/* Graph View */` banner comment.
   **Evidence**: `diff` shows `98a99 > [blank]`. Source had 1 blank; new file has 2.
   **Fix**: Delete one blank line at line 99 of inspector.css. Cosmetic — does not affect CSS parsing or test outcomes.

2. **[Line 325 / EOF]** Missing trailing newline after the final `}`.
   **Evidence**: `tail -c 20` shows `}` as the last byte. Source App.css ends with `}\n` (line 4081 closes a brace, line 4082 is blank/EOF).
   **Fix**: Append a single `\n` at EOF. Many editors and lint tools (POSIX, prettier, stylelint) flag missing-final-newline.

## Out-of-Scope Observations
(Brief mentions; do not fix as part of this review.)

- **Brittle `:focus-visible` coverage on react-flow controls (line 144).** The rule targets only `.react-flow__controls-button:focus-visible`. React-flow renders several other interactive elements inside `.react-flow__controls` (the controls panel itself, minimap nodes, attribution link) — none of those get the project's focus-ring treatment. If react-flow ships an internal markup change (e.g. renames `__controls-button` to `__controls-Button` per their own class-style shifts in 11.x → 12.x), this rule silently stops applying and focus visibility regresses without a test failure (the `toContain` assertion will still pass against the literal string, but the live UI loses the outline). Worth a follow-up: either widen the selector to `.react-flow__controls *:focus-visible` or add a DOM-side test that asserts computed `outline-width` on a focused control. Not in scope for the split.
- **Missing dark/light parity in react-flow var overrides.** The `:root .react-flow.react-flow {...}` block defines ~21 variables; the `[data-theme="light"]` override only re-defines 3 (`--xy-minimap-mask-background-color-default`, `--xy-selection-background-color-default`, `--xy-controls-box-shadow-default`). All the other variables (controls-button colors, edge stroke, handle background, minimap background, minimap node colors) flow through underlying `--color-*` tokens which presumably already retheme in `[data-theme="light"]`. Worth a one-pass audit to confirm every `var(--color-*)` reference here picks up a defined light-theme value in `tokens.css`/`themes.css` — but that audit belongs to the themes/tokens reviewers, not this split.
- **No orphaned `.graph-*` rules.** Every `.graph-*` selector traces to `GraphView.tsx`. Clean.
- For style polish on the banner comment block, refer to `doc-critic` if desired.

## Confidence Assessment
**Confidence**: **High** for fidelity verification.
**Basis**:
- Direct byte-level diff of concatenated source ranges (3762–3859 ∪ 3865–4082) against the new file body (lines 9–325).
- `diff` output reduced to 2 whitespace-only deltas; all selectors and declarations byte-identical.
- TSX consumers grep-verified.
- `GraphView.test.tsx` path-binding confirmed (file path matches; all `toContain` strings present; regex pattern syntactically matches against actual content).

## Risk Assessment
**Residual Risk**:
- I did not actually execute `vitest` to run `GraphView.test.tsx`. The regex match is verified by inspection only. **Low** risk — the assertions are simple string-containment plus one regex whose body I read directly off the new file.
- I did not verify the CSS is loaded by the bundler in the new path. If `inspector.css` is not imported from the inspector tree (e.g. by `GraphView.tsx` or a barrel), the rules are dead at runtime even though the test passes (the test reads the file from disk, not the rendered DOM). Worth confirming during the broader split rollup.
- I did not verify `--color-canvas-grid` (referenced by GraphView.test.tsx:339) is defined anywhere — but that token belongs to tokens.css, not this file.

## Information Gaps
- Did not run `vitest src/components/inspector/GraphView.test.tsx` to empirically confirm the test passes against the new file.
- Did not verify that `inspector.css` is imported by any TSX/index entry point in the new structure (the test reads it by raw `fs.readFileSync`, which bypasses the bundler — so a missing import would not surface in the test but would break the live UI).
- Did not check stylelint / prettier policy on trailing newlines for this project (the missing-newline finding is graded Minor on general convention, not on a verified project rule).
- The header comment is the only undeclared addition; I treated it as benign per the convention visible in `sidebar.css`, but did not search for a project policy that forbids per-file banner comments.

## Caveats
- This review is scoped to **fidelity of the extraction**, not to the broader CSS architecture, theme correctness, or react-flow integration health. Items in "Out-of-Scope Observations" are surfaced as flags only.
- The two minor findings (extra blank line, missing trailing newline) do not block the split landing. They are recommended cleanups that can land in a follow-up commit or be folded into the next pass over this file.
- Verdict assumes the broader split work (sidebar.css ownership of `.side-rail-error-banner`, tokens.css ownership of `--color-*` tokens) is being reviewed in companion passes; I confirmed only the adjacency move-out target for `.side-rail-error-banner`, not the full sidebar.css landing.
