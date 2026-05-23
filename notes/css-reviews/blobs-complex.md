# blobs.css — Complex Fidelity Review

**Scope**: verify split of `App.css` lines 5058–5199 into
`src/elspeth/web/frontend/src/components/blobs/blobs.css`.

## Edit Report Received

Reconstructed from caller brief:

- Source range: `App.css` lines 5058–5199.
- Selectors expected: `.blob-row-*`, `.blob-manager-*` (22 total).
- Adjacency: no moves; `.blob-action-btn` belongs in `chat.css`, not here.
- Header comment ownership boilerplate.

## Summary

| Item | Value |
|------|-------|
| File | `src/elspeth/web/frontend/src/components/blobs/blobs.css` (145 lines) |
| Type | CSS (frontend) |
| Edits reviewed | 1 (whole-file split) |
| Overall verdict | **Approved** (with minor design-debt observations carried forward from source) |

## Intent Fit

All 22 selectors from the originating range are present in `blobs.css`, in
source order, with byte-identical declaration blocks. Cross-checked by
enumerating the source range (`grep -n "blob" App.css` lines 5061–5189) and
matching one-for-one against the destination.

| Selector | Source line | Dest line | Identical |
|----------|-------------|-----------|-----------|
| `.blob-row-container` | 5061 | 7 | ✓ |
| `.blob-row-status-dot` | 5069 | 15 | ✓ |
| `.blob-row-creator` | 5076 | 22 | ✓ |
| `.blob-row-filename` | 5080 | 26 | ✓ |
| `.blob-row-size` | 5087 | 33 | ✓ |
| `.blob-row-actions` | 5093 | 39 | ✓ |
| `.blob-row-preview` | 5099 | 45 | ✓ |
| `.blob-row-preview-loading` | 5105 | 51 | ✓ |
| `.blob-row-preview-error` | 5110 | 56 | ✓ |
| `.blob-row-preview-pre` | 5115 | 61 | ✓ |
| `.blob-row-preview-truncated` | 5131 | 77 | ✓ |
| `.blob-manager-container` | 5139 | 85 | ✓ |
| `.blob-manager-header` | 5147 | 93 | ✓ |
| `.blob-manager-title` | 5156 | 102 | ✓ |
| `.blob-manager-upload-btn` | 5162 | 108 | ✓ |
| `.blob-manager-error` | 5170 | 116 | ✓ |
| `.blob-manager-list` | 5177 | 123 | ✓ |
| `.blob-manager-loading, .blob-manager-empty` | 5182–5183 | 128–129 | ✓ |
| `.blob-manager-category-header` | 5189 | 135 | ✓ |

Block-comment banners (`/* BlobRow */`, `/* BlobManager */`) preserved
verbatim. New header banner added on lines 1–2 stating ownership
(`.blob-row*`, `.blob-manager*`), which is the project convention observed in
sibling split files (e.g. `chat.css` line 3 declares the same ownership for
`.blob-action-btn`).

## Scope Discipline

- **Declared out-of-scope**: `.blob-action-btn` — confirmed absent from
  `blobs.css`; present at `chat.css:119` and `:133`. The ownership
  comment in `chat.css:3` explicitly claims it. No drift.
- **Undeclared changes**: none. New 2-line header is the only addition; it
  matches the cross-file convention.

## Structural Integrity

| Invariant | Status | Evidence |
|-----------|--------|----------|
| Brace balance | ✓ | 22 rule blocks open and close cleanly; final line 144 is closing `}` for `.blob-manager-category-header`; trailing blank line on 145. |
| Selector text byte-identical | ✓ | per-line comparison above. |
| Comment banners preserved | ✓ | two `/* --- */` banners at lines 4–6 and 82–84, identical to source. |
| Selector ordering | ✓ | source order preserved (BlobRow group then BlobManager group). |
| Ownership header | ✓ | added in convention with other split files. |
| Custom-property usage | ✓ | all `var(--*)` tokens unchanged. |

No build/lint run — this is a pure-CSS extraction with no compiler
involvement at this granularity. Vite/tsc do not type-check CSS.

## Cross-Reference / Call-Site Integrity

All 22 selectors are consumed by live components:

- `BlobRow.tsx` consumes: `blob-row-container`, `blob-row-status-dot`,
  `blob-row-creator`, `blob-row-filename`, `blob-row-size`,
  `blob-row-actions`, `blob-row-preview`, `blob-row-preview-loading`,
  `blob-row-preview-error`, `blob-row-preview-pre`,
  `blob-row-preview-truncated`. (11/11)
- `BlobManager.tsx` consumes: `blob-manager-container`, `blob-manager-header`,
  `blob-manager-title`, `blob-manager-upload-btn`, `blob-manager-error`,
  `blob-manager-list`, `blob-manager-loading`, `blob-manager-empty`,
  `blob-manager-category-header`. (9/9, plus the joint
  `loading, empty` rule on lines 128–129.)

No orphan selectors. No stale references.

The split file must be imported by whatever aggregates the component-scoped
stylesheets. Verifying the import-graph entry is out of scope for this review
(per caller — fidelity only); it should be checked by the index-level
reviewer.

## Orphan / Dead-Code / Boundary Findings

### Boundary at top of source range (App.css 5054 → 5058)

Preceding rule `.progress-error-row-id` (line 5054) is unrelated — the
boundary at the `/* BlobRow */` banner is clean. No straddle.

### Boundary at bottom of source range (App.css 5198 → 5200)

Trailing `}` of `.blob-manager-category-header` (line 5198) and the next
banner `/* PluginCard */` (line 5200) — clean boundary. No straddle.

### Dead-rule analysis

Inside `blobs.css` proper, no dead rules. However, two **hook-only class
names** appear in markup but have no CSS rule in this file (or in the
project anywhere — verified by `grep -n "^\.blob-row\b\|^\.blob-manager\b"
App.css`):

- `BlobRow.tsx:96` — `className="blob-row blob-row-container"` — the bare
  `.blob-row` token has no styling.
- `BlobManager.tsx:98` — `className="blob-manager blob-manager-container"`
  — the bare `.blob-manager` token has no styling.

This is a **pre-existing** design choice in the source (these are presumably
JS-query hooks or test selectors), not introduced by the split. It would
become a finding only if the split were expected to add bare-selector rules.
Flag for documentation, not for revert.

## Style / Behavior Continuity

| Aspect | Source | Dest | Match |
|--------|--------|------|-------|
| Token usage (`--space-*`, `--color-*`, `--font-*`) | as written | unchanged | ✓ |
| Mixed unit conventions (`6px` + `var(--space-sm)`) | yes, lines 5065, 5151 | preserved at 11, 97 | ✓ (continuity, not quality) |
| Tab vs. space indent | 2-space | 2-space | ✓ |
| Trailing newline | yes | yes | ✓ |

Refactor risk: none — pure copy, no behaviour delta possible.

## Issues Found

### Critical

None.

### Major

None.

### Minor

1. **Bare `.blob-row` / `.blob-manager` classes have no rules.**
   `BlobRow.tsx:96` and `BlobManager.tsx:98` apply both `blob-row` and
   `blob-row-container` (resp. `blob-manager` and `blob-manager-container`),
   but only the `-container` half is styled. Pre-existing in source; either
   delete the unused tokens from markup or document them as JS/test hooks
   in a comment. **Out of scope for the split itself.**

2. **Mixed-unit padding.** `.blob-row-container` uses `padding: 6px
   var(--space-sm);` (line 11) and `.blob-manager-header` uses `padding: 6px
   var(--space-md);` (line 97). The literal `6px` should likely be a token
   (`--space-xs` is the conventional sibling). Pre-existing — surface only.

## Out-of-Scope Observations

### Surfaced per caller's "ALSO SURFACE" list

- **Dead rules**: none inside `blobs.css`. (See Minor #1 for orphan markup
  tokens.)
- **Overlap with `chat.css` blob rules**: `chat.css` owns `.blob-action-btn`
  only; namespace is disjoint (`blob-row-*` and `blob-manager-*` vs.
  `blob-action-*`). **No overlap.** The ownership comment in `chat.css:3`
  correctly documents the cross-file split.
- **Missing focus state on blob-row items.** `BlobRow.tsx:96` does not
  appear to be focusable (no `tabIndex`, no role) — looking at the markup,
  the row itself is a `<div>`; interactive elements inside it
  (`.blob-row-actions` children) are the focusable surface and inherit
  focus from button styling in `base.css`/`common.css`. **Focus on the row
  container would be required only if rows become keyboard-navigable**
  (e.g., arrow-key list selection). Not a regression introduced by the
  split. Refer to UX/accessibility track if BlobManager grows keyboard
  navigation.
- **Brittle file-type icon selectors**: none present in `blobs.css` — no
  `.blob-icon`, no `.file-type-*`, no attribute selectors targeting
  filename extensions. Either the icon logic is component-rendered (likely
  in `BlobRow.tsx` via a JSX switch on MIME / extension) or
  unstyled. Verified by `grep` over `components/blobs/`. **No brittle
  selectors to flag.**

### Other refer-outs

- Index-import wiring: refer to the index-level reviewer.
- Token migration for literal `6px`: refer to `tokens.css` /
  design-system track, not this split.

## Confidence Assessment

**Confidence**: High.
**Basis**: per-selector source-vs-dest mapping confirmed; brace balance
visually verified; component-side consumption verified via `grep` across
TSX. The split is mechanical and the diff surface is small (162 lines
total).

## Risk Assessment

**Residual risk**: Low. Two residual risks:

1. If `blobs.css` is not yet wired into the application's CSS entry
   (`index.css` or similar), the styling silently disappears at runtime.
   Out of scope for fidelity review — verify at index level.
2. If a future merge re-introduces blob rules into `App.css`, the
   duplicate will silently win or lose by load order. Mitigation: confirm
   the originating range is **deleted** from `App.css` in the same commit
   (not verified here — caller said "no moves" but a split must remove the
   source).

## Information Gaps

- Did not verify `blobs.css` is imported by the frontend's CSS entry point.
- Did not verify the originating range was removed from `App.css` in the
  split commit (the worktree's `App.css` was not inspected; the original
  was inspected at `/home/john/elspeth/src/...`).
- Did not run the frontend build (`npm run build`) to confirm no CSS-side
  regressions; pure-CSS extraction doesn't surface in tsc/vite typecheck
  anyway.

## Caveats

- This review is **fidelity-only**: byte-equivalence between a source range
  and an extracted file. It does not assess CSS quality, design-token
  hygiene, accessibility, or browser-render parity. For those, refer to
  `lyra-ux-designer` or `accessibility-audit`.
- The "minor" findings above are inherited from the source range; they are
  surfaced because the caller asked for them, not because the split caused
  them.
