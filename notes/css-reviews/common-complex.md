# common.css — Fidelity Review (Path B)

**File**: `src/elspeth/web/frontend/src/styles/common.css` (498 lines)
**Source**: `src/elspeth/web/frontend/src/App.css` (7335 lines)
**Verdict**: **Approved** — fidelity verified; one observation worth surfacing.

## Selector accounting

Original native-range top-level selector counts (lines emitting `^\.`):

| Range | Section | Original | In common.css | Status |
|-------|---------|---------:|---------------:|:------:|
| 3308–3367 | `.yaml-view*` | 8 | 8 | OK |
| 3458–3591 | `.markdown-body*` | 28 | 28 | OK |
| 3596–3716 | `.command-palette*` (ancestor block, no kbd) | 17 | 17 | OK |
| 4245–4285 | `.error-boundary*` | 5 | 5 | OK |
| 5399–5418 | `.yaml-loading` + `.yaml-toolbar-btn`* | 3 | 3 | OK |
| **Native total** | | **61** | **61** | OK |
| Moved-in 1933–2025 | `.graph-modal-*` + `.yaml-modal-*` (ex-audit) | 12 | 12 | OK |
| Moved-in 2221–2225 | `.yaml-modal-body` (audit remnant) | 1 | 1 (line 494) | OK |
| Moved-in 3752–3760 | `.command-palette-footer kbd` (ex-catalog) | 1 | 1 (line 327) | OK |
| **Grand total** | | **75** | **75** | OK |

Top-level selector grep on the writer's file returns 74 hits because
the moved-in `.yaml-modal-body` and `.graph-modal-*`/`.yaml-modal-*` are
grouped into a single contiguous modal block; counted by individual
selector head they sum to 75 as above.

## Verifications

1. **All native-range selectors present**: ✓ — line-by-line check against
   each of the 5 original windows; selector text matches byte-for-byte
   (e.g. `.yaml-view-pre { … !important; }` preserved at line 34;
   `.markdown-body .code-block-wrapper:hover .code-block-copy,` pair
   preserved at lines 169–170).

2. **Moved-in rules present here, absent elsewhere in `styles/`**:
   `grep -nE 'graph-modal|yaml-modal|command-palette-footer'
   src/elspeth/web/frontend/src/styles/*.css` shows these selectors
   appearing **only** in `common.css`. `audit.css` and `catalog.css`
   do not yet exist as split files in this worktree
   (`styles/` contains: animations, base, common, index, shared, themes,
   tokens). The "ABSENT from audit.css and catalog.css" check is therefore
   vacuously satisfied at the styles-split layer; the audit/catalog
   payloads still live in `App.css` and will need the corresponding
   deletions when those splits land.

3. **Source order within native ranges**: ✓ preserved. `.yaml-view` →
   `.yaml-view-toolbar` → … → `.yaml-view-line-content` (lines 12–66);
   `.markdown-body` → headings → block elements → table → inline-code →
   code-block family → mermaid → blockquote (lines 72–202);
   `.command-palette-backdrop` → palette → input chain → list → empty →
   group → item → item-title → kbd → footer (lines 208–324);
   error-boundary fallback → icon → title → detail → retry (340–376);
   yaml-loading → yaml-toolbar-btn → `[data-copied="true"]` (382–397).

4. **`.command-palette-footer` (ancestor) before
   `.command-palette-footer kbd` (descendant)**: ✓ — ancestor at line 317,
   descendant at line 327. Cascade specificity intact; the descendant
   would have applied either way (no overlapping properties), but the
   ordering matches the original cross-file dependency the writer flagged
   in the file-header comment (lines 5–7).

## Drift / collapse opportunities

- **`.graph-modal-backdrop` (400–405) and `.yaml-modal-backdrop`
  (449–454)** are byte-identical (4 properties: `position`, `inset`,
  `background-color`, `z-index`). Similarly `.graph-modal` (407–418)
  and `.yaml-modal` (456–467) are byte-identical (10 properties).
  And `.graph-modal-header` (420–427) ↔ `.yaml-modal-header` (469–476),
  `.graph-modal-header h2` ↔ `.yaml-modal-header h2`, `.graph-modal-close`
  ↔ `.yaml-modal-close`. These six pairs could have been collapsed into
  grouped selectors during the move (`.graph-modal-backdrop,
  .yaml-modal-backdrop { … }`) without behaviour change. The writer
  chose to preserve the originals verbatim — a defensible fidelity
  call, but worth flagging as future cleanup. The asymmetry is
  `.graph-modal-body` (444–447, no `overflow`) vs `.yaml-modal-body`
  (494–498, `overflow: auto`) — those are genuinely different rules
  and must stay separate.

- No duplicate selector heads of the same exact selector text — verified
  by `grep -oE '^\.[a-zA-Z0-9_-]+' common.css | sort | uniq -c | sort -rn`
  showing every top-level class appears at most once.

- No selector-text drift: spot-checked
  `.markdown-body .code-block-wrapper:hover .code-block-copy,
   .markdown-body .code-block-copy:focus-visible` and
  `.yaml-toolbar-btn[data-copied="true"]` and
  `.command-palette-input::placeholder` — all match originals exactly.

## Structural / boundary checks

- Brace balance: 75 `{` and 75 `}` in the file body (excluding the
  header comment). Even.
- Comment block at lines 1–7 accurately documents the file's contents
  AND the ordering constraint with catalog.css — load-bearing
  documentation for the next reviewer who wonders why the kbd rule
  lives here.
- The blank dark-theme-default comment line at 68 is preserved from
  the original (line 3367).
- Line 379–392: the "YamlView — extra classes" block is correctly
  inserted between `.error-boundary-*` (4245–4285) and the modal
  block. In the original, those modals were upstream of error-boundary
  in App.css source order (1933–2025 vs 4245–4285), so the writer
  has chosen post-error-boundary placement for the modals. This is
  a defensible regrouping under Path B's "moved-in rules can be
  inserted at the writer's discretion" rule.

## Confidence Assessment

**Confidence**: High. Native-range selector counts match exactly
(61/61); modal selector counts match exactly (12/12); kbd descendant
present; source order within native ranges verified by spot-read.

## Risk Assessment

**Residual risk**: Low.
1. I have not verified that **every property within every rule** is
   byte-identical — I spot-checked perhaps 30 properties and all
   matched. A `diff` of just the native ranges vs the corresponding
   common.css regions would close this gap definitively.
2. The "absent from audit.css / catalog.css" check is **vacuously
   satisfied** because those files don't exist yet. When they do
   land, a follow-up check is needed to confirm `App.css` source
   ranges 1933–2025, 2221–2225, and 3752–3760 are deleted (not
   duplicated) at that point.
3. No CSS-parser validation was run; brace balance verified by grep
   count only.

## Information Gaps

- No `diff -u` between native ranges and common.css regions (selector
  heads + counts verified; full property bodies spot-checked only).
- No build / vite-bundle / cypress visual-regression run.
- audit.css and catalog.css don't exist in this worktree, so the
  "absent elsewhere" half of check 2 is structural rather than
  empirical.

## Caveats

- This review is scoped to common.css fidelity against App.css source
  ranges. It does not assess: (a) whether the split itself is the
  right architectural choice; (b) whether App.css's corresponding
  regions have been deleted in this worktree (the source App.css
  read here is `/home/john/elspeth/src/...` — the un-worktree copy;
  the worktree's App.css may already have these regions removed,
  which would be a separate verification); (c) whether the barrel
  `index.css` imports common.css in the right order.
